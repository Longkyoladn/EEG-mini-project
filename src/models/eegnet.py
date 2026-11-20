"""Kiến trúc EEGNet-v4 điều chỉnh cho bài toán phân loại single-trial P300.

Tham chiếu:
    Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P.,
    & Lance, B. J. (2018). EEGNet: a compact convolutional neural network for
    EEG-based brain-computer interfaces. Journal of Neural Engineering, 15(5).

Phiên bản này được cấu hình cho tensor đầu vào 3D chuẩn `(B, 127, 251)`
(127 kênh EEG, cửa sổ [-0.2, 0.8] s ở tần số lấy mẫu 250 Hz), trả về raw logits
`(B, 1)` tương thích với `BCEWithLogitsLoss` và `Captum IntegratedGradients`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


def _compute_same_padding(kernel_size: int) -> Tuple[int, int]:
    """Tính cặp padding trái/phải để bảo toàn độ dài chuỗi thời gian."""
    total_pad = kernel_size - 1
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return pad_left, pad_right


def _renorm_weight_(
    weight: torch.Tensor,
    max_norm: Optional[float],
    dims: Tuple[int, ...],
    eps: float = 1e-8,
) -> None:
    """Ràng buộc L2 max-norm in-place trên tensor trọng số (gọi trong no_grad)."""
    if max_norm is None:
        return
    norms = weight.norm(p=2, dim=dims, keepdim=True)
    scale = torch.clamp(max_norm / (norms + eps), max=1.0)
    weight.mul_(scale)


class EEGNetV4(nn.Module):
    """Mạng tích chập gọn nhẹ EEGNet-v4 làm đối chứng nội bộ (Go/No-Go baseline).

    Args:
        n_channels: Số kênh EEG đầu vào (mặc định 127).
        n_times: Số mẫu thời gian mỗi trial (mặc định 251 @ 250 Hz).
        n_classes: Số đầu ra logit (mặc định 1 cho phân loại nhị phân).
        F1: Số bộ lọc thời gian ở Block 1.
        D: Hệ số nhân bộ lọc không gian theo chiều sâu (depth multiplier).
        F2: Số bộ lọc ở Block 2; nếu ``None`` tự đặt bằng ``F1 * D``.
        kernel_length: Độ dài kernel tích chập thời gian đầu tiên (~500 ms @ 250 Hz).
        sep_kernel_length: Độ dài kernel tích chập tách rời theo thời gian ở Block 2.
        pool1: Hệ số AvgPool2d sau Block 1.
        pool2: Hệ số AvgPool2d sau Block 2.
        dropout: Xác suất Dropout (bắt buộc ``inplace=False``).
        spatial_max_norm: Ngưỡng L2 max-norm cho lớp Depthwise Spatial Conv.
        linear_max_norm: Ngưỡng L2 max-norm cho lớp Linear phân loại cuối.
    """

    def __init__(
        self,
        n_channels: int = 127,
        n_times: int = 251,
        n_classes: int = 1,
        F1: int = 8,
        D: int = 2,
        F2: Optional[int] = 16,
        kernel_length: int = 125,
        sep_kernel_length: int = 15,
        pool1: int = 4,
        pool2: int = 8,
        dropout: float = 0.25,
        spatial_max_norm: Optional[float] = 1.0,
        linear_max_norm: Optional[float] = 0.25,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_times <= 0 or n_classes <= 0:
            raise ValueError("n_channels, n_times và n_classes phải là số nguyên dương.")
        if kernel_length <= 0 or sep_kernel_length <= 0:
            raise ValueError("kernel_length và sep_kernel_length phải > 0.")
        if pool1 <= 0 or pool2 <= 0:
            raise ValueError("pool1 và pool2 phải > 0.")

        resolved_f2 = F1 * D if F2 is None else F2
        out_times = (n_times // pool1) // pool2
        if out_times <= 0:
            raise ValueError(
                f"n_times={n_times} quá nhỏ so với pool1={pool1} và pool2={pool2}."
            )

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = resolved_f2
        self.kernel_length = kernel_length
        self.sep_kernel_length = sep_kernel_length
        self.pool1_size = pool1
        self.pool2_size = pool2
        self.dropout_rate = dropout
        self.spatial_max_norm = spatial_max_norm
        self.linear_max_norm = linear_max_norm

        # Block 1: Tích chập thời gian -> Tích chập không gian theo chiều sâu
        pad1_left, pad1_right = _compute_same_padding(kernel_length)
        self.temp_pad = nn.ConstantPad2d((pad1_left, pad1_right, 0, 0), 0.0)
        self.temporal_conv = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, kernel_length),
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(
            F1,
            F1 * D,
            kernel_size=(n_channels, 1),
            groups=F1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.act1 = nn.ELU(inplace=False)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool1), stride=(1, pool1))
        self.drop1 = nn.Dropout(p=dropout, inplace=False)

        # Block 2: Separable Conv2d (Depthwise thời gian + Pointwise)
        pad2_left, pad2_right = _compute_same_padding(sep_kernel_length)
        self.sep_pad = nn.ConstantPad2d((pad2_left, pad2_right, 0, 0), 0.0)
        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, sep_kernel_length),
            padding=0,
            groups=F1 * D,
            bias=False,
        )
        self.sep_pointwise = nn.Conv2d(
            F1 * D,
            resolved_f2,
            kernel_size=(1, 1),
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(resolved_f2)
        self.act2 = nn.ELU(inplace=False)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, pool2), stride=(1, pool2))
        self.drop2 = nn.Dropout(p=dropout, inplace=False)

        # Classifier
        flat_dim = resolved_f2 * out_times
        self.flatten = nn.Flatten(start_dim=1)
        self.classifier = nn.Linear(flat_dim, n_classes)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"Đầu vào EEGNetV4 phải là tensor 3D (B, C, T), nhận được shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_times:
            raise ValueError(
                f"Kỳ vọng shape (B, {self.n_channels}, {self.n_times}), "
                f"nhận được {tuple(x.shape)}."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi từ tensor EEG `(B, C, T)` sang raw logits `(B, 1)`."""
        self._validate_input(x)

        h = x.unsqueeze(1)
        h = self.temp_pad(h)
        h = self.temporal_conv(h)
        h = self.bn1(h)
        h = self.spatial_conv(h)
        h = self.bn2(h)
        h = self.act1(h)
        h = self.pool1(h)
        h = self.drop1(h)

        h = self.sep_pad(h)
        h = self.sep_depthwise(h)
        h = self.sep_pointwise(h)
        h = self.bn3(h)
        h = self.act2(h)
        h = self.pool2(h)
        h = self.drop2(h)

        h = self.flatten(h)
        return self.classifier(h)

    def apply_max_norm(self) -> None:
        """Áp dụng ràng buộc L2 max-norm sau `optimizer.step()` (không gọi trong `forward`)."""
        with torch.no_grad():
            _renorm_weight_(
                self.spatial_conv.weight,
                self.spatial_max_norm,
                dims=(1, 2, 3),
            )
            _renorm_weight_(
                self.classifier.weight,
                self.linear_max_norm,
                dims=(1,),
            )

    def get_spatial_weights(self) -> torch.Tensor:
        """Trả về bản sao trọng số lớp Depthwise Spatial Conv shape `(F1*D, 1, C, 1)`."""
        return self.spatial_conv.weight.detach().clone()

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Đếm tổng số tham số của mô hình."""
        params = (
            (p for p in self.parameters() if p.requires_grad)
            if trainable_only
            else self.parameters()
        )
        return sum(p.numel() for p in params)
