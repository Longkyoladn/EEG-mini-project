"""Module chứa các kiến trúc mô hình Deep Learning cho bài toán phân loại EEG P300."""

from src.models.eegnet import EEGNetV4
from src.models.p300_atcnet import P300ATCNet

__all__ = [
    "EEGNetV4",
    "P300ATCNet",
]
