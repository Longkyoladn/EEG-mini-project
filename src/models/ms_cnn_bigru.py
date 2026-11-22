"""MS-CNN-BiGRU cho phân loại EEG P300 trên từng trial."""

from collections import OrderedDict

import torch
from torch import nn


class MSCNNBiGRU(nn.Module):
    """Nhận (B, 127, T), trả raw logits (B, 1) cho BCEWithLogitsLoss.

    Khi return_attention=True, trả thêm trọng số thời gian (B, T // 8).
    """

    def __init__(self) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            nn.Sequential(OrderedDict(
                temporal_conv=nn.Conv2d(1, 6, (1, k), padding=(0, k // 2), bias=False),
                temporal_bn=nn.BatchNorm2d(6),
                spatial_conv=nn.Conv2d(6, 12, (127, 1), groups=6, bias=False),
                spatial_bn=nn.BatchNorm2d(12),
                activation=nn.ELU(),
                pool=nn.AvgPool2d((1, 4)),
                dropout=nn.Dropout(0.25),
            ))
            for k in (125, 63, 31)
        )
        self.fusion = nn.Sequential(
            nn.Conv1d(36, 36, 15, padding=7, groups=36, bias=False),
            nn.Conv1d(36, 32, 1, bias=False),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(2),
            nn.Dropout(0.30),
        )
        self.bigru = nn.GRU(32, 24, num_layers=2, batch_first=True,
                            dropout=0.30, bidirectional=True)
        self.layer_norm = nn.LayerNorm(48)
        self.attention = nn.Sequential(nn.Linear(48, 24), nn.Tanh(), nn.Linear(24, 1, bias=False))
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(48, 1)

    @torch.no_grad()
    def apply_max_norm(self) -> None:
        """Giới hạn chuẩn từng bộ lọc không gian và hàng trọng số phân loại."""
        for branch in self.branches:
            weight = branch.spatial_conv.weight
            norms = weight.norm(p=2, dim=(1, 2, 3), keepdim=True)
            weight.mul_((2.0 / (norms + 1e-8)).clamp(max=1.0))
        weight = self.classifier.weight
        weight.mul_((0.5 / (weight.norm(dim=1, keepdim=True) + 1e-8)).clamp(max=1.0))

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1] != 127 or x.shape[2] < 8:
            raise ValueError(f"Đầu vào phải có shape (B, 127, T) với T >= 8, nhận {tuple(x.shape)}")

        eeg = x.unsqueeze(1)
        features = torch.cat([branch(eeg).squeeze(2) for branch in self.branches], dim=1)
        sequence = self.fusion(features).transpose(1, 2)
        if sequence.is_cuda and not self.training and sequence.requires_grad:
            # Captum cần backward qua GRU trong eval; cuDNN chỉ hỗ trợ RNN backward khi train.
            with torch.backends.cudnn.flags(enabled=False):
                states, _ = self.bigru(sequence)
        else:
            states, _ = self.bigru(sequence)

        states = self.layer_norm(states)
        alpha = self.attention(states).squeeze(-1).softmax(dim=1)
        pooled = torch.bmm(alpha.unsqueeze(1), states).squeeze(1)
        logits = self.classifier(self.dropout(pooled))
        if return_attention:
            return logits, alpha
        return logits
