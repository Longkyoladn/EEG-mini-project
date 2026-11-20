"""Các kiến trúc Deep Learning nhánh 2 cho phân loại single-trial EEG P300."""

from src.models.eegnet import EEGNetV4
from src.models.p300_atcnet import P300ATCNet

__all__ = ["EEGNetV4", "P300ATCNet"]
