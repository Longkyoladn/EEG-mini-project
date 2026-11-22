"""Các kiến trúc Deep Learning cho phân loại single-trial EEG P300."""

from src.models.ms_cnn_bigru import MSCNNBiGRU
from src.models.eegnet_v4 import EEGNetV4
from src.models.p300_atc_net import P300ATCNet
__all__ = ["EEGNetV4", "MSCNNBiGRU", "P300ATCNet"]
