"""Kiến trúc P300-ATCNet tinh chỉnh cho phân loại single-trial P300.

Tham chiếu kiến trúc gốc:
    Altaheri, H., Muhammad, G., & Alsulaiman, M. (2023). Physics-informed
    attention temporal convolutional network for EEG-based motor imagery
    classification. IEEE Transactions on Industrial Informatics, 19(2), 2249-2258.

Bốn tinh chỉnh chuyên biệt cho dữ liệu P300 `(B, 127, 251)` @ 250 Hz:
    1. CV Block giữ độ phân giải thời gian (nén 16x: 251 -> 62 -> 15 bước ~64 ms/bước),
       tích chập đầu dài ~500 ms (mặc định 125 mẫu, hỗ trợ 96..125) bắt dải Delta 1-4 Hz,
       và separable conv ở tầng 2 để kiểm soát số tham số; spatial_max_norm mặc định nới lỏng (2.0).
    2. Thu gọn Sliding Window (`n_windows` trong 1..5, mặc định 2) kèm tùy chọn
       `shared_branch` kiểm soát ngân sách tham số (~35k-60k cho mặc định).
    3. Nhánh Attention + Causal Dilated TCN (`dilation=1, 2`, `kernel_size=4`).
    4. Readout `avg_pool` / `attn_pool` thay cho `last_step` nhằm bắt toàn bộ đỉnh P300
       khóa pha trong cửa sổ thời gian (giữ `last_step` cho ablation).
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ReadoutMode = Literal["avg_pool", "attn_pool", "last_step"]


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


class _MultiHeadSelfAttentionBlock(nn.Module):
    """Khối Pre-LayerNorm Multi-Head Self-Attention với kết nối dư."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Nhận `(B, F, W)` và trả về `(B, F, W)`."""
        x_seq = x.transpose(1, 2)
        normed = self.norm(x_seq)
        attn_out, _ = self.mha(normed, normed, normed, need_weights=False)
        out = x_seq + self.dropout(attn_out)
        return out.transpose(1, 2)


class _CausalConv1dStage(nn.Module):
    """Một tầng tích chập 1D nhân quả: left-pad -> Conv1d -> BN -> ELU -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU(inplace=False)
        self.dropout = nn.Dropout(p=dropout, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.left_pad, 0))
        h = self.conv(h)
        h = self.bn(h)
        h = self.act(h)
        return self.dropout(h)


class _CausalConv1dResidualBlock(nn.Module):
    """Khối TCN nhân quả gồm 2 tầng tích chập giãn nở và kết nối dư."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stages = nn.ModuleList(
            [
                _CausalConv1dStage(
                    in_channels=in_channels if idx == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
                for idx in range(2)
            ]
        )
        self.downsample: nn.Module = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.out_act = nn.ELU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for stage in self.stages:
            h = stage(h)
        res = self.downsample(x)
        return self.out_act(h + res)


class _ATCTCBranch(nn.Module):
    """Nhánh Attention + Causal Dilated TCN cho một cửa sổ trượt."""

    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        at_dropout: float,
        tcn_depth: int,
        tcn_kernel_size: int,
        tcn_filters: int,
        tcn_dropout: float,
    ) -> None:
        super().__init__()
        self.attention = _MultiHeadSelfAttentionBlock(
            embed_dim=in_channels,
            num_heads=num_heads,
            dropout=at_dropout,
        )
        self.tcn_blocks = nn.ModuleList(
            [
                _CausalConv1dResidualBlock(
                    in_channels=in_channels if i == 0 else tcn_filters,
                    out_channels=tcn_filters,
                    kernel_size=tcn_kernel_size,
                    dilation=2**i,
                    dropout=tcn_dropout,
                )
                for i in range(tcn_depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attention(x)
        for block in self.tcn_blocks:
            h = block(h)
        return h


class P300ATCNet(nn.Module):
    """Mạng P300-ATCNet cho phân loại single-trial P300 từ tensor `(B, 127, 251)`.

    Args:
        n_channels: Số kênh EEG đầu vào (mặc định 127).
        n_times: Số mẫu thời gian mỗi trial (mặc định 251 @ 250 Hz).
        n_classes: Số đầu ra logit (mặc định 1).
        F1: Số bộ lọc thời gian ở tầng CV đầu tiên.
        D: Hệ số nhân bộ lọc không gian theo chiều sâu.
        F2: Số kênh đặc trưng sau CV Block; mặc định ``F1 * D = 32``.
        temp_kernel_1: Độ dài kernel tích chập thời gian 1 (mặc định 125 ~500 ms).
        sep_kernel_2: Độ dài kernel tích chập tách rời ở tầng CV thứ 2.
        pool1: Hệ số AvgPool2d sau tầng CV 1.
        pool2: Hệ số AvgPool2d sau tầng CV 2 (tổng nén 16x với pool1=4, pool2=4).
        cv_dropout: Xác suất Dropout trong CV Block.
        n_windows: Số cửa sổ trượt (1..5, mặc định 2 giữ tổng tham số ~49.7k).
        shared_branch: Nếu ``True``, dùng chung 1 nhánh AT+TC cho mọi cửa sổ.
        num_heads: Số đầu chú ý trong Multi-Head Self-Attention.
        at_dropout: Xác suất Dropout trong khối Attention.
        tcn_depth: Số khối TCN nhân quả giãn nở (mặc định 2 với dilation 1, 2).
        tcn_kernel_size: Độ dài kernel Conv1d trong TCN.
        tcn_filters: Số kênh đầu ra mỗi khối TCN.
        tcn_dropout: Xác suất Dropout trong TCN.
        readout: Chiến lược gộp trục thời gian sau TCN (``"avg_pool"``,
            ``"attn_pool"`` hoặc ``"last_step"``).
        spatial_max_norm: Ngưỡng L2 max-norm cho Depthwise Spatial Conv (mặc định 2.0).
        linear_max_norm: Ngưỡng L2 max-norm cho các lớp Linear phân loại (mặc định 0.25).
    """

    def __init__(
        self,
        n_channels: int = 127,
        n_times: int = 251,
        n_classes: int = 1,
        F1: int = 16,
        D: int = 2,
        F2: Optional[int] = 32,
        temp_kernel_1: int = 125,
        sep_kernel_2: int = 15,
        pool1: int = 4,
        pool2: int = 4,
        cv_dropout: float = 0.3,
        n_windows: int = 2,
        shared_branch: bool = False,
        num_heads: int = 2,
        at_dropout: float = 0.3,
        tcn_depth: int = 2,
        tcn_kernel_size: int = 4,
        tcn_filters: int = 32,
        tcn_dropout: float = 0.3,
        readout: ReadoutMode = "avg_pool",
        spatial_max_norm: Optional[float] = 2.0,
        linear_max_norm: Optional[float] = 0.25,
    ) -> None:
        super().__init__()
        resolved_f2 = F1 * D if F2 is None else F2
        if n_channels <= 0 or n_times <= 0 or n_classes <= 0:
            raise ValueError("n_channels, n_times và n_classes phải là số nguyên dương.")
        if temp_kernel_1 <= 0 or sep_kernel_2 <= 0 or tcn_kernel_size <= 0:
            raise ValueError("Kích thước kernel phải > 0.")
        if pool1 <= 0 or pool2 <= 0 or tcn_depth <= 0:
            raise ValueError("pool1, pool2 và tcn_depth phải > 0.")
        if resolved_f2 % num_heads != 0:
            raise ValueError(
                f"F2 ({resolved_f2}) phải chia hết cho num_heads ({num_heads})."
            )
        if readout not in ("avg_pool", "attn_pool", "last_step"):
            raise ValueError(
                f"readout phải thuộc ('avg_pool', 'attn_pool', 'last_step'), nhận được {readout!r}."
            )

        cv_steps = (n_times // pool1) // pool2
        if not (1 <= n_windows <= 5):
            raise ValueError(f"n_windows phải nằm trong [1, 5], nhận được {n_windows}.")
        if n_windows > cv_steps:
            raise ValueError(
                f"n_windows ({n_windows}) vượt quá độ dài chuỗi sau CV ({cv_steps})."
            )

        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        self.F1 = F1
        self.D = D
        self.F2 = resolved_f2
        self.temp_kernel_1 = temp_kernel_1
        self.sep_kernel_2 = sep_kernel_2
        self.pool1_size = pool1
        self.pool2_size = pool2
        self.cv_steps = cv_steps
        self.n_windows = n_windows
        self.window_len = cv_steps - n_windows + 1
        self.shared_branch = shared_branch
        self.tcn_filters = tcn_filters
        self.readout: ReadoutMode = readout
        self.spatial_max_norm = spatial_max_norm
        self.linear_max_norm = linear_max_norm

        # 1. CV Block (tổng nén 16x: 251 -> 62 -> 15)
        pad1_left, pad1_right = _compute_same_padding(temp_kernel_1)
        self.temp_pad = nn.ConstantPad2d((pad1_left, pad1_right, 0, 0), 0.0)
        self.temporal_conv = nn.Conv2d(
            1,
            F1,
            kernel_size=(1, temp_kernel_1),
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
        self.drop1 = nn.Dropout(p=cv_dropout, inplace=False)

        pad2_left, pad2_right = _compute_same_padding(sep_kernel_2)
        self.sep_pad = nn.ConstantPad2d((pad2_left, pad2_right, 0, 0), 0.0)
        self.sep_depthwise = nn.Conv2d(
            F1 * D,
            F1 * D,
            kernel_size=(1, sep_kernel_2),
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
        self.drop2 = nn.Dropout(p=cv_dropout, inplace=False)

        # 2. Các nhánh Sliding Window (AT + TC)
        num_branches = 1 if shared_branch else n_windows
        self.branches = nn.ModuleList(
            [
                _ATCTCBranch(
                    in_channels=resolved_f2,
                    num_heads=num_heads,
                    at_dropout=at_dropout,
                    tcn_depth=tcn_depth,
                    tcn_kernel_size=tcn_kernel_size,
                    tcn_filters=tcn_filters,
                    tcn_dropout=tcn_dropout,
                )
                for _ in range(num_branches)
            ]
        )

        # 3. Readout & Classifier theo cửa sổ
        self.attn_pools: Optional[nn.ModuleList] = (
            nn.ModuleList([nn.Linear(tcn_filters, 1) for _ in range(num_branches)])
            if readout == "attn_pool"
            else None
        )
        self.classifiers = nn.ModuleList(
            [nn.Linear(tcn_filters, n_classes) for _ in range(n_windows)]
        )

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"Đầu vào P300ATCNet phải là tensor 3D (B, C, T), nhận được shape {tuple(x.shape)}."
            )
        if x.shape[1] != self.n_channels or x.shape[2] != self.n_times:
            raise ValueError(
                f"Kỳ vọng shape (B, {self.n_channels}, {self.n_times}), "
                f"nhận được {tuple(x.shape)}."
            )

    def _encode_cv(self, x: torch.Tensor) -> torch.Tensor:
        """Biến đổi `(B, C, T)` qua CV Block thành chuỗi `(B, F2, T_cv)`."""
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
        return h.squeeze(2)

    def _apply_readout(self, tcn_out: torch.Tensor, branch_idx: int) -> torch.Tensor:
        """Tổng hợp chuỗi đầu ra TCN `(B, tcn_filters, W)` thành `(B, tcn_filters)`."""
        if self.readout == "avg_pool":
            return tcn_out.mean(dim=-1)
        if self.readout == "last_step":
            return tcn_out[:, :, -1]
        assert self.attn_pools is not None
        seq = tcn_out.transpose(1, 2)
        scores = self.attn_pools[branch_idx](seq)
        weights = torch.softmax(scores, dim=1)
        return (seq * weights).sum(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền xuôi từ `(B, C, T)` sang raw logits `(B, 1)`."""
        self._validate_input(x)
        cv_seq = self._encode_cv(x)

        window_logits: List[torch.Tensor] = []
        for win_idx in range(self.n_windows):
            branch_idx = 0 if self.shared_branch else win_idx
            window_slice = cv_seq[:, :, win_idx : win_idx + self.window_len]
            tcn_out = self.branches[branch_idx](window_slice)
            feat = self._apply_readout(tcn_out, branch_idx)
            window_logits.append(self.classifiers[win_idx](feat))

        return torch.stack(window_logits, dim=0).mean(dim=0)

    def apply_max_norm(self) -> None:
        """Áp dụng ràng buộc L2 max-norm sau `optimizer.step()` (không gọi trong `forward`)."""
        with torch.no_grad():
            _renorm_weight_(
                self.spatial_conv.weight,
                self.spatial_max_norm,
                dims=(1, 2, 3),
            )
            for clf in self.classifiers:
                _renorm_weight_(clf.weight, self.linear_max_norm, dims=(1,))

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
