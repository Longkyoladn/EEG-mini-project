"""Trích xuất đặc trưng miền thời gian ERP đơn lượt thử (Zero Data Leakage).

Tất cả phép biến đổi đều chạy độc lập trên từng trial từ mảng Volts (``~1e-5 V``)
chuyển sang ``µV`` ở độ chính xác ``float64``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import uniform_filter1d

CHANNEL_PRESETS: Dict[str, Tuple[str, ...]] = {
    "midline_10": ("Fz", "Cz", "CPz", "Pz", "POz", "Oz", "CP1", "CP2", "P1", "P2"),
    "core_14": (
        "Fz",
        "FC1",
        "FC2",
        "Cz",
        "CPz",
        "Pz",
        "POz",
        "Oz",
        "CP1",
        "CP2",
        "P1",
        "P2",
        "P3",
        "P4",
    ),
    "extended_20": (
        "Fz",
        "F3",
        "F4",
        "FC1",
        "FC2",
        "Cz",
        "C3",
        "C4",
        "CPz",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
        "Pz",
        "P1",
        "P2",
        "P3",
        "P4",
        "POz",
        "Oz",
    ),
}

BINNED_CORE_CHANNELS: Tuple[str, ...] = (
    "Fz",
    "Cz",
    "CPz",
    "Pz",
    "POz",
    "Oz",
    "CP1",
    "CP2",
)

ERP_PEAK_WINDOWS: Dict[str, Tuple[float, float, str]] = {
    "N100": (0.080, 0.150, "min"),
    "P200": (0.150, 0.250, "max"),
    "N200": (0.200, 0.300, "min"),
    "P300": (0.300, 0.600, "max"),
}

ERP_MEAN_WINDOWS: Dict[str, Tuple[float, float]] = {
    "N100": (0.080, 0.150),
    "P200": (0.150, 0.250),
    "N200": (0.200, 0.300),
    "P300_early": (0.300, 0.450),
    "P300_late": (0.450, 0.600),
    "Late_SW": (0.600, 0.800),
}

BIN_EDGES_S: Tuple[Tuple[float, float], ...] = tuple(
    (round(0.05 * i, 4), round(0.05 * (i + 1), 4)) for i in range(16)
)

VALID_FEATURE_SETS: Tuple[str, ...] = (
    "erp_peaks_only",
    "erp_windows_and_peaks",
    "erp_full_binned",
)

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_data_geometry(data_dir: Union[str, Path]) -> Tuple[List[str], np.ndarray]:
    """Đọc danh sách tên kênh chuẩn (127 kênh) và trục thời gian ``times.npy``."""
    d = Path(data_dir)
    with open(d / "channel_names.json", "r", encoding="utf-8") as f:
        channel_names = [str(ch) for ch in json.load(f)]
    times = np.asarray(np.load(d / "times.npy"), dtype=np.float64)
    return channel_names, times


def _resolve_time_indices(
    times: np.ndarray,
    t_start: float,
    t_end: float,
    *,
    right_inclusive: bool = True,
    label: str = "window",
) -> np.ndarray:
    """Trả về mảng chỉ số thời gian nằm trong khoảng ``[t_start, t_end]`` (hoặc nửa mở)."""
    tol = 1e-9
    if right_inclusive:
        mask = (times >= t_start - tol) & (times <= t_end + tol)
    else:
        mask = (times >= t_start - tol) & (times < t_end - tol)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(
            f"Cửa sổ '{label}' [{t_start}, {t_end}]s không chứa mẫu nào trên trục times."
        )
    return idx


class ERPFeatureExtractor:
    """Trích xuất đặc trưng miền thời gian ERP trên từng lượt thử (single-trial).

    Hỗ trợ 3 cấu hình ``feature_set``:
    - ``erp_peaks_only``: 9 đặc trưng/kênh (peak amp + peak latency cho N100, P200, N200, P300 và P2P).
    - ``erp_windows_and_peaks``: 16 đặc trưng/kênh (thêm 6 window mean + P300 signed AUC).
    - ``erp_full_binned``: bổ sung thêm 16 bin 50 ms trên 8 kênh đường giữa (128 chiều).
    """

    def __init__(
        self,
        channel_names: Sequence[str],
        times: np.ndarray,
        *,
        channel_preset: str = "core_14",
        channels: Optional[Sequence[str]] = None,
        feature_set: str = "erp_windows_and_peaks",
        smooth_window: int = 5,
        scale_to_uv: float = 1e6,
    ) -> None:
        if feature_set not in VALID_FEATURE_SETS:
            raise ValueError(
                f"feature_set không hợp lệ: {feature_set!r}. Hợp lệ: {VALID_FEATURE_SETS}"
            )
        if not isinstance(smooth_window, int) or smooth_window < 1:
            raise ValueError(f"smooth_window phải là số nguyên >= 1, nhận {smooth_window!r}.")

        self.channel_names = [str(ch) for ch in channel_names]
        self.times = np.asarray(times, dtype=np.float64)
        if self.times.ndim != 1 or self.times.size < 2:
            raise ValueError(f"times phải là mảng 1D có >= 2 mẫu, nhận shape {self.times.shape}.")
        self._times_ms = self.times * 1000.0

        self.feature_set = feature_set
        self.smooth_window = smooth_window
        self.scale_to_uv = float(scale_to_uv)

        if channels is not None:
            selected = tuple(str(ch) for ch in channels)
            if len(selected) == 0:
                raise ValueError("Danh sách channels tùy chỉnh không được rỗng.")
            self.channel_preset = "custom"
        else:
            if channel_preset not in CHANNEL_PRESETS:
                raise ValueError(
                    f"channel_preset không hợp lệ: {channel_preset!r}. "
                    f"Hợp lệ: {tuple(CHANNEL_PRESETS.keys())}"
                )
            self.channel_preset = channel_preset
            selected = CHANNEL_PRESETS[channel_preset]

        if len(set(selected)) != len(selected):
            raise ValueError(f"Danh sách kênh chọn chứa tên trùng lặp: {selected}")

        ch_to_idx = {ch: i for i, ch in enumerate(self.channel_names)}
        missing_sel = [ch for ch in selected if ch not in ch_to_idx]
        if missing_sel:
            raise ValueError(f"Kênh không tồn tại trong channel_names: {missing_sel}")

        self.selected_channels: Tuple[str, ...] = selected
        self._ch_indices = np.asarray([ch_to_idx[ch] for ch in selected], dtype=np.int64)

        self._include_windows = feature_set in ("erp_windows_and_peaks", "erp_full_binned")
        self._include_binned = feature_set == "erp_full_binned"

        if self._include_binned:
            missing_bin = [ch for ch in BINNED_CORE_CHANNELS if ch not in ch_to_idx]
            if missing_bin:
                raise ValueError(
                    f"Kênh binned không tồn tại trong channel_names: {missing_bin}"
                )
            self._binned_ch_indices = np.asarray(
                [ch_to_idx[ch] for ch in BINNED_CORE_CHANNELS], dtype=np.int64
            )
        else:
            self._binned_ch_indices = np.empty(0, dtype=np.int64)

        self._peak_specs: List[Tuple[str, np.ndarray, str]] = [
            (
                wname,
                _resolve_time_indices(self.times, t0, t1, right_inclusive=True, label=wname),
                mode,
            )
            for wname, (t0, t1, mode) in ERP_PEAK_WINDOWS.items()
        ]
        self._mean_specs: List[Tuple[str, np.ndarray]] = [
            (
                wname,
                _resolve_time_indices(self.times, t0, t1, right_inclusive=True, label=wname),
            )
            for wname, (t0, t1) in ERP_MEAN_WINDOWS.items()
        ]
        self._auc_p300_idx = _resolve_time_indices(
            self.times, 0.300, 0.600, right_inclusive=True, label="auc_P300"
        )
        self._auc_p300_times = self.times[self._auc_p300_idx]

        n_bins = len(BIN_EDGES_S)
        self._bin_specs: List[Tuple[int, int, np.ndarray]] = [
            (
                int(round(t0 * 1000)),
                int(round(t1 * 1000)),
                _resolve_time_indices(
                    self.times,
                    t0,
                    t1,
                    right_inclusive=(i == n_bins - 1),
                    label=f"bin_{i}",
                ),
            )
            for i, (t0, t1) in enumerate(BIN_EDGES_S)
        ]

        self.feature_names: List[str] = self._build_feature_names()
        self.n_features: int = len(self.feature_names)

    def _build_feature_names(self) -> List[str]:
        names: List[str] = []
        for ch in self.selected_channels:
            for wname in ERP_PEAK_WINDOWS:
                names.append(f"{ch}__{wname}_peak_amp_uv")
                names.append(f"{ch}__{wname}_peak_lat_ms")
            names.append(f"{ch}__p2p_P300_N200_uv")

        if self._include_windows:
            for ch in self.selected_channels:
                for wname in ERP_MEAN_WINDOWS:
                    names.append(f"{ch}__{wname}_mean_uv")
                names.append(f"{ch}__auc_P300_uv_s")

        if self._include_binned:
            for ch in BINNED_CORE_CHANNELS:
                for t0_ms, t1_ms, _ in self._bin_specs:
                    names.append(f"{ch}__bin_{t0_ms:03d}_{t1_ms:03d}ms_mean_uv")

        return names

    def get_feature_names(self) -> List[str]:
        """Trả về bản sao danh sách tên đặc trưng theo đúng thứ tự cột."""
        return list(self.feature_names)

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "ERPFeatureExtractor":
        """Hàm no-op tương thích scikit-learn (không học tham số để tránh leakage)."""
        _ = (X, y)
        return self

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        arr = np.asarray(X)
        if arr.ndim == 2:
            arr = arr[None, :, :]
        elif arr.ndim != 3:
            raise ValueError(
                f"X phải có 2 chiều (C, T) hoặc 3 chiều (N, C, T), nhận shape {arr.shape}."
            )
        if arr.shape[1] != len(self.channel_names):
            raise ValueError(
                f"Số kênh của X ({arr.shape[1]}) không khớp channel_names ({len(self.channel_names)})."
            )
        if arr.shape[2] != self.times.size:
            raise ValueError(
                f"Số mẫu thời gian của X ({arr.shape[2]}) không khớp times ({self.times.size})."
            )
        if not np.isfinite(arr).all():
            raise ValueError("Tín hiệu EEG đầu vào chứa NaN hoặc Inf.")
        return arr

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Trích xuất ma trận đặc trưng ``(N, D)`` kiểu ``float64`` từ EEG thô/sạch."""
        arr = self._validate_input(X)
        n_epochs = int(arr.shape[0])
        if n_epochs == 0:
            return np.empty((0, self.n_features), dtype=np.float64)

        n_sel = len(self.selected_channels)
        out = np.empty((n_epochs, self.n_features), dtype=np.float64)

        # Chỉ cắt đúng các kênh cần dùng trước khi ép kiểu float64 để tối ưu RAM
        sig_uv = np.asarray(arr[:, self._ch_indices, :], dtype=np.float64) * self.scale_to_uv
        if self.smooth_window > 1:
            sig_smooth = uniform_filter1d(
                sig_uv, size=self.smooth_window, axis=-1, mode="nearest"
            )
        else:
            sig_smooth = sig_uv

        # Block 1: Peak amplitudes, peak latencies & P300-N200 peak-to-peak (9 đặc trưng/kênh)
        n_peak_cols = n_sel * 9
        peak_block = out[:, :n_peak_cols].reshape(n_epochs, n_sel, 9)
        n200_amp = None
        p300_amp = None

        for w_idx, (wname, t_idx, mode) in enumerate(self._peak_specs):
            seg = sig_smooth[:, :, t_idx]
            rel_idx = np.argmin(seg, axis=-1) if mode == "min" else np.argmax(seg, axis=-1)
            amp = np.take_along_axis(seg, rel_idx[..., None], axis=-1)[..., 0]
            lat = self._times_ms[t_idx][rel_idx]
            peak_block[:, :, 2 * w_idx] = amp
            peak_block[:, :, 2 * w_idx + 1] = lat
            if wname == "N200":
                n200_amp = amp
            elif wname == "P300":
                p300_amp = amp

        peak_block[:, :, 8] = p300_amp - n200_amp
        col_offset = n_peak_cols

        # Block 2: Window means (6) + P300 signed AUC (1) = 7 đặc trưng/kênh
        if self._include_windows:
            n_win_cols = n_sel * 7
            win_block = out[:, col_offset : col_offset + n_win_cols].reshape(
                n_epochs, n_sel, 7
            )
            for m_idx, (_, t_idx) in enumerate(self._mean_specs):
                win_block[:, :, m_idx] = sig_uv[:, :, t_idx].mean(axis=-1)
            win_block[:, :, 6] = _trapezoid(
                sig_uv[:, :, self._auc_p300_idx], x=self._auc_p300_times, axis=-1
            )
            col_offset += n_win_cols

        # Block 3: 16 bin 50 ms trên 8 kênh đường giữa = 128 đặc trưng
        if self._include_binned:
            n_bin_ch = len(BINNED_CORE_CHANNELS)
            n_bins = len(self._bin_specs)
            binned_uv = (
                np.asarray(arr[:, self._binned_ch_indices, :], dtype=np.float64)
                * self.scale_to_uv
            )
            bin_block = out[:, col_offset : col_offset + n_bin_ch * n_bins].reshape(
                n_epochs, n_bin_ch, n_bins
            )
            for b_idx, (_, _, t_idx) in enumerate(self._bin_specs):
                bin_block[:, :, b_idx] = binned_uv[:, :, t_idx].mean(axis=-1)

        if not np.isfinite(out).all():
            raise ValueError("Ma trận đặc trưng ERP đầu ra chứa NaN hoặc Inf.")
        return out

    def transform_chunked(
        self,
        X: np.ndarray,
        chunk_size: int = 2048,
    ) -> np.ndarray:
        """Trích xuất đặc trưng theo từng chunk để tiết kiệm RAM trên mảng memmap lớn."""
        if chunk_size < 1:
            raise ValueError(f"chunk_size phải >= 1, nhận {chunk_size}.")
        arr = np.asarray(X) if not isinstance(X, np.memmap) else X
        if arr.ndim == 2:
            return self.transform(arr)
        if arr.ndim != 3:
            raise ValueError(
                f"X phải có 2 hoặc 3 chiều, nhận shape {arr.shape}."
            )
        n_epochs = int(arr.shape[0])
        out = np.empty((n_epochs, self.n_features), dtype=np.float64)
        for start in range(0, n_epochs, chunk_size):
            end = min(start + chunk_size, n_epochs)
            out[start:end] = self.transform(arr[start:end])
        return out
