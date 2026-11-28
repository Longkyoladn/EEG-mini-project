"""Unit tests cho bộ trích xuất đặc trưng miền thời gian ERP (Giai đoạn 2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.erp_time_features import (
    CHANNEL_PRESETS,
    ERPFeatureExtractor,
)


def _make_geometry() -> tuple[List[str], np.ndarray]:
    chs: List[str] = list(CHANNEL_PRESETS["extended_20"])
    for idx in range(len(chs), 127):
        chs.append(f"CH_{idx:03d}")
    times = np.linspace(-0.2, 0.8, 251, dtype=np.float64)
    return chs, times


def test_synthetic_peak_amplitude_latency_and_auc() -> None:
    """Kiểm tra khôi phục chính xác đỉnh, độ trễ, P2P, mean và AUC trên tín hiệu tổng hợp."""
    chs, times = _make_geometry()
    pz_idx = chs.index("Pz")
    cz_idx = chs.index("Cz")

    # Tạo tín hiệu đơn vị Volts (nhân 1e-6) với 4 đỉnh Gaussian đúng độ trễ ở Pz
    # Lưu ý: các đỉnh đủ tách biệt để không cộng chéo vào cửa sổ lân cận
    pz_uv = (
        -6.0 * np.exp(-0.5 * ((times - 0.112) / 0.010) ** 2)
        + 4.0 * np.exp(-0.5 * ((times - 0.200) / 0.010) ** 2)
        - 5.0 * np.exp(-0.5 * ((times - 0.252) / 0.010) ** 2)
        + 10.0 * np.exp(-0.5 * ((times - 0.480) / 0.030) ** 2)
    )
    X = np.zeros((2, 127, 251), dtype=np.float32)
    X[0, pz_idx, :] = (pz_uv * 1e-6).astype(np.float32)
    X[1, pz_idx, :] = (-0.5 * pz_uv * 1e-6).astype(np.float32)
    X[:, cz_idx, :] = 3.0e-6  # Hằng số +3.0 µV tại Cz

    ext_raw = ERPFeatureExtractor(
        chs, times, feature_set="erp_windows_and_peaks", smooth_window=1
    )
    feats_raw = ext_raw.transform(X)
    names = ext_raw.get_feature_names()
    idx_map = {n: i for i, n in enumerate(names)}

    assert np.isclose(feats_raw[0, idx_map["Pz__N100_peak_amp_uv"]], -6.0, atol=1e-2)
    assert np.isclose(feats_raw[0, idx_map["Pz__N100_peak_lat_ms"]], 112.0, atol=1e-4)
    assert np.isclose(feats_raw[0, idx_map["Pz__P200_peak_amp_uv"]], 4.0, atol=1e-2)
    assert np.isclose(feats_raw[0, idx_map["Pz__P200_peak_lat_ms"]], 200.0, atol=1e-4)
    assert np.isclose(feats_raw[0, idx_map["Pz__N200_peak_amp_uv"]], -5.0, atol=1e-2)
    assert np.isclose(feats_raw[0, idx_map["Pz__N200_peak_lat_ms"]], 252.0, atol=1e-4)
    assert np.isclose(feats_raw[0, idx_map["Pz__P300_peak_amp_uv"]], 10.0, atol=1e-2)
    assert np.isclose(feats_raw[0, idx_map["Pz__P300_peak_lat_ms"]], 480.0, atol=1e-4)
    assert np.isclose(feats_raw[0, idx_map["Pz__p2p_P300_N200_uv"]], 15.0, atol=2e-2)
    assert feats_raw[0, idx_map["Pz__auc_P300_uv_s"]] > 0.5
    assert np.isclose(feats_raw[0, idx_map["Cz__P300_late_mean_uv"]], 3.0, atol=1e-4)

    # Kiểm tra smooth_window=5 vẫn giữ đúng vị trí đỉnh trên tín hiệu trơn
    ext_sm = ERPFeatureExtractor(
        chs, times, feature_set="erp_windows_and_peaks", smooth_window=5
    )
    feats_sm = ext_sm.transform(X)
    assert np.isclose(feats_sm[0, idx_map["Pz__P300_peak_lat_ms"]], 480.0, atol=4.1)
    assert feats_sm[0, idx_map["Pz__P300_peak_amp_uv"]] > 9.0
    assert np.isclose(feats_sm[0, idx_map["Cz__P300_late_mean_uv"]], 3.0, atol=1e-4)


def test_feature_sets_shapes_and_names() -> None:
    """Kiểm tra đúng số chiều và tên duy nhất cho cả 3 feature_set và preset."""
    chs, times = _make_geometry()
    rng = np.random.default_rng(42)
    X = rng.normal(0.0, 1e-5, size=(4, 127, 251)).astype(np.float32)

    expected_dims = {
        ("core_14", "erp_peaks_only"): 126,
        ("core_14", "erp_windows_and_peaks"): 224,
        ("core_14", "erp_full_binned"): 352,
        ("midline_10", "erp_windows_and_peaks"): 160,
    }
    for (preset, fset), exp_dim in expected_dims.items():
        ext = ERPFeatureExtractor(chs, times, channel_preset=preset, feature_set=fset)
        feats = ext.transform(X)
        names = ext.get_feature_names()
        assert feats.shape == (4, exp_dim)
        assert feats.dtype == np.float64
        assert len(names) == exp_dim
        assert len(set(names)) == exp_dim


def test_single_trial_equals_batch_zero_leakage() -> None:
    """Khẳng định đặc trưng trích trên single-trial khớp tuyệt đối batch (Zero Leakage)."""
    chs, times = _make_geometry()
    rng = np.random.default_rng(7)
    X = rng.normal(0.0, 2e-5, size=(7, 127, 251)).astype(np.float32)

    ext = ERPFeatureExtractor(chs, times, feature_set="erp_full_binned")
    ext_fitted = ext.fit(X[:3], np.array([0, 1, 0]))
    assert ext_fitted is ext

    batch_feats = ext.transform(X)
    chunk_feats = ext.transform_chunked(X, chunk_size=3)
    assert np.allclose(batch_feats, chunk_feats, atol=1e-12, rtol=0.0)

    for i in (0, 3, 6):
        single_3d = ext.transform(X[i : i + 1])
        single_2d = ext.transform(X[i])
        assert np.allclose(single_3d, batch_feats[i : i + 1], atol=1e-12, rtol=0.0)
        assert np.allclose(single_2d, batch_feats[i : i + 1], atol=1e-12, rtol=0.0)


def test_invalid_inputs_raise() -> None:
    """Kiểm tra các trường hợp cấu hình hoặc shape đầu vào không hợp lệ phải báo lỗi."""
    chs, times = _make_geometry()

    with pytest.raises(ValueError):
        ERPFeatureExtractor(chs, times, feature_set="unknown_set")
    with pytest.raises(ValueError):
        ERPFeatureExtractor(chs, times, channel_preset="unknown_preset")
    with pytest.raises(ValueError):
        ERPFeatureExtractor(chs, times, channels=["Pz", "NON_EXISTENT_CH"])

    ext = ERPFeatureExtractor(chs, times)
    with pytest.raises(ValueError):
        ext.transform(np.zeros((2, 126, 251), dtype=np.float32))
    with pytest.raises(ValueError):
        ext.transform(np.zeros((2, 127, 200), dtype=np.float32))

    bad = np.zeros((1, 127, 251), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ext.transform(bad)


def test_real_data_sample_finite_and_plausible() -> None:
    """Nếu có data/processed, kiểm tra mẫu thật gồm cả Target và Frequent không NaN/Inf."""
    data_dir = Path("data/processed")
    if not (data_dir / "X_clean.npy").exists():
        pytest.skip("Không tìm thấy data/processed/X_clean.npy")

    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    times = np.load(data_dir / "times.npy")
    ch_names = json.loads((data_dir / "channel_names.json").read_text(encoding="utf-8"))

    freq_idx = np.flatnonzero(y == 0)[:8]
    tgt_idx = np.flatnonzero(y == 1)[:8]
    sample_idx = np.concatenate([freq_idx, tgt_idx])
    X_sub = np.asarray(X_clean[sample_idx], dtype=np.float32)

    ext = ERPFeatureExtractor(ch_names, times, feature_set="erp_full_binned")
    feats = ext.transform(X_sub)

    assert feats.shape == (16, 352)
    assert feats.dtype == np.float64
    assert np.isfinite(feats).all()

    names = ext.get_feature_names()
    amp_cols = [i for i, n in enumerate(names) if not n.endswith("_ms")]
    assert np.max(np.abs(feats[:, amp_cols])) < 500.0
