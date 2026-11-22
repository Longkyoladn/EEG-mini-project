"""Các kiến trúc Deep Learning cho phân loại single-trial EEG P300."""

from src.models.ms_cnn_bigru import MSCNNBiGRU

__all__ = ["EEGNetV4", "MSCNNBiGRU", "P300ATCNet"]


def __getattr__(name):
    """Chỉ nạp kiến trúc DL2 khi người gọi yêu cầu model đó."""
    if name == "EEGNetV4":
        from .eegnet import EEGNetV4
        return EEGNetV4
    if name == "P300ATCNet":
        from .p300_atcnet import P300ATCNet
        return P300ATCNet
    raise AttributeError(name)
