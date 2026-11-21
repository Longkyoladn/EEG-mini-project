"""Các kiến trúc Deep Learning nhánh 2 cho phân loại single-trial EEG P300."""
"""Module chứa các kiến trúc mô hình và pipeline giải mã cho bài toán phân loại EEG."""

from src.models.eegnet import EEGNetV4
from src.models.p300_atcnet import P300ATCNet
from src.models.spatial_linear import get_spatial_linear_models

__all__ = ["EEGNetV4",
         "P300ATCNet",
         "get_spatial_linear_models"]

