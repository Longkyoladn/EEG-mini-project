"""Bộ kiểm thử tự động Data Contract & Chống rò rỉ dữ liệu cho EEG Oddball P300.

Tệp: tests/test_data_contract.py

Đọc đường dẫn dữ liệu từ biến môi trường `EEG_DATA_DIR` (mặc định dùng
`data/processed`).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

EXCLUDED_SUBJECTS = {"sub-02", "sub-14", "sub-17"}
EXPECTED_META_COLS = [
    "trial_uid",
    "subject_id",
    "trial_idx",
    "onset_sample",
    "onset_sec",
    "event_type",
    "label",
    "rt_ms",
    "rt_group",
]


def _resolve_data_dir() -> Path:
    env_dir = os.environ.get("EEG_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    return Path("data/processed")


@pytest.fixture(scope="module")
def data_dir() -> Path:
    d = _resolve_data_dir()
    if not (d / "X_clean.npy").exists():
        pytest.fail(f"Không tìm thấy dữ liệu tiền xử lý tại: {d.resolve()}")
    return d


def test_shapes_dtypes_and_excluded_subjects(data_dir: Path) -> None:
    """Kiểm tra shape, dtype, không NaN/Inf, loại đúng 3 subjects & khớp nhánh no_ica."""
    X_clean = np.load(data_dir / "X_clean.npy")
    y = np.load(data_dir / "y.npy")
    subject_ids = np.load(data_dir / "subject_ids.npy")
    times = np.load(data_dir / "times.npy")

    assert X_clean.ndim == 3
    assert X_clean.shape[1:] == (127, 251)
    assert X_clean.dtype == np.float32
    assert np.isfinite(X_clean).all(), "X_clean chứa NaN hoặc Inf"

    n_clean = X_clean.shape[0]
    assert n_clean > 0
    assert y.shape == (n_clean,)
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()).issubset({0, 1})

    assert subject_ids.shape == (n_clean,)
    assert subject_ids.dtype.kind == "U"
    present_subjects = set(np.unique(subject_ids).tolist())
    assert present_subjects.isdisjoint(EXCLUDED_SUBJECTS), (
        f"Phát hiện subject bị cấm trong dữ liệu sạch: {present_subjects & EXCLUDED_SUBJECTS}"
    )

    assert times.shape == (251,)
    assert times.dtype == np.float64
    assert np.isclose(times[0], -0.2, atol=1e-6)
    assert np.isclose(times[-1], 0.8, atol=1e-6)

    # Kiểm tra nhánh đối chứng no_ica cho XAI RQ3
    X_no_ica = np.load(data_dir / "X_no_ica_no_reject.npy")
    y_no_ica = np.load(data_dir / "y_no_ica_no_reject.npy")
    sub_no_ica = np.load(data_dir / "subject_ids_no_ica_no_reject.npy")
    df_no_ica = pd.read_csv(data_dir / "metadata_no_ica.csv")
    df_clean = pd.read_csv(data_dir / "metadata.csv")

    assert X_no_ica.ndim == 3 and X_no_ica.shape[1:] == (127, 251)
    assert X_no_ica.dtype == np.float32
    assert np.isfinite(X_no_ica).all()
    assert X_no_ica.shape[0] >= n_clean
    assert y_no_ica.shape == (X_no_ica.shape[0],) and y_no_ica.dtype == np.int64
    assert sub_no_ica.shape == (X_no_ica.shape[0],)
    assert len(df_no_ica) == X_no_ica.shape[0]
    assert "kept_in_clean" in df_no_ica.columns
    assert df_no_ica["kept_in_clean"].dtype == bool or set(df_no_ica["kept_in_clean"].unique()).issubset({True, False})

    kept_uids = df_no_ica.loc[df_no_ica["kept_in_clean"].astype(bool), "trial_uid"].tolist()
    assert kept_uids == df_clean["trial_uid"].tolist(), (
        "Danh sách trial_uid có kept_in_clean=True không khớp 1-1 với metadata.csv"
    )


def test_channels_and_colin27_montage(data_dir: Path) -> None:
    """Kiểm tra 127 kênh chuẩn và độ khớp 100% với montage colin27_1005."""
    with open(data_dir / "channel_names.json", "r", encoding="utf-8") as f:
        ch_names = json.load(f)

    assert len(ch_names) == 127
    assert len(set(ch_names)) == 127

    ch_set = set(ch_names)
    for expected in ("TP9", "TP10", "I1", "I2"):
        assert expected in ch_set, f"Thiếu kênh chuẩn '{expected}'"
    for legacy in ("O9", "O10", "TP9 LEFT EAR", "TP10 RIGHT EAR"):
        assert legacy not in ch_set, f"Vẫn còn tên kênh cũ chưa đổi '{legacy}'"

    montage = mne.channels.make_standard_montage("colin27_1005")
    montage_chs = set(montage.ch_names)
    missing = [ch for ch in ch_names if ch not in montage_chs]
    assert not missing, f"Các kênh không khớp colin27_1005: {missing}"


def test_metadata_and_rt_validity(data_dir: Path) -> None:
    """Kiểm tra schema metadata.csv, trial_uid duy nhất, và tính hợp lệ của rt_ms."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    subject_ids = np.load(data_dir / "subject_ids.npy")

    df = pd.read_csv(data_dir / "metadata.csv", keep_default_na=True)
    assert list(df.columns) == EXPECTED_META_COLS
    assert len(df) == X_clean.shape[0]
    assert df["trial_uid"].is_unique
    assert np.array_equal(df["label"].to_numpy(dtype=np.int64), y)
    assert np.array_equal(df["subject_id"].astype(str).to_numpy(), subject_ids)

    target_mask = df["label"] == 1
    freq_mask = df["label"] == 0

    target_rts = df.loc[target_mask, "rt_ms"]
    assert target_rts.notna().all(), "Có trial Target bị thiếu rt_ms"
    assert ((target_rts >= 100.0) & (target_rts <= 1500.0)).all(), (
        "Có trial Target có rt_ms ngoài khoảng [100, 1500] ms"
    )

    freq_rts = df.loc[freq_mask, "rt_ms"]
    assert freq_rts.isna().all(), "Có trial Frequent nhưng rt_ms không phải NaN"
    assert (df.loc[freq_mask, "rt_group"].astype(str) == "na").all()

    # Kiểm tra phân nhóm rt_group cho Target
    fast_ok = (df.loc[target_mask & (df["rt_ms"] < 450.0), "rt_group"] == "fast").all()
    mid_ok = (
        df.loc[
            target_mask & (df["rt_ms"] >= 450.0) & (df["rt_ms"] <= 550.0), "rt_group"
        ]
        == "mid"
    ).all()
    slow_ok = (df.loc[target_mask & (df["rt_ms"] > 550.0), "rt_group"] == "slow").all()
    assert fast_ok and mid_ok and slow_ok

    qc_path = data_dir / "preprocessing_qc_summary.csv"
    assert qc_path.exists()
    df_qc = pd.read_csv(qc_path)
    assert set(np.unique(subject_ids).tolist()).issubset(set(df_qc["subject_id"].tolist()))


def test_splits_zero_leakage(data_dir: Path) -> None:
    """Kiểm tra tuyệt đối không rò rỉ trial hoặc subject trên cả 3 giao thức splits.json."""
    df = pd.read_csv(data_dir / "metadata.csv")
    with open(data_dir / "splits.json", "r", encoding="utf-8") as f:
        splits = json.load(f)

    n_samples = len(df)
    subject_ids = df["subject_id"].astype(str).to_numpy()
    expected_sha256 = hashlib.sha256(
        "\n".join(df["trial_uid"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()

    assert splits["n_samples"] == n_samples
    assert splits["n_subjects"] == len(np.unique(subject_ids))
    assert splits["trial_uid_sha256"] == expected_sha256

    # 1. Within-subject
    within = splits["within_subject"]
    assert set(within.keys()) == set(np.unique(subject_ids).tolist())
    for sub_id, folds in within.items():
        sub_all = set(np.where(subject_ids == sub_id)[0].tolist())
        assert len(folds) == 5
        for fold in folds:
            tr = set(fold["train_idx"])
            va = set(fold["val_idx"])
            te = set(fold["test_idx"])
            assert tr.isdisjoint(va)
            assert tr.isdisjoint(te)
            assert va.isdisjoint(te)
            assert tr | va | te == sub_all

    # 2. Cross-subject 5-fold
    if splits["n_subjects"] >= 5:
        folds_5 = splits["cross_subject_5fold"]
        assert len(folds_5) == 5
        for fold in folds_5:
            tr = set(fold["train_idx"])
            va = set(fold["val_idx"])
            te = set(fold["test_idx"])
            assert tr.isdisjoint(va)
            assert tr.isdisjoint(te)
            assert va.isdisjoint(te)
            assert tr | va | te == set(range(n_samples))

            tr_subs = set(subject_ids[list(tr)].tolist())
            va_subs = set(subject_ids[list(va)].tolist())
            te_subs = set(subject_ids[list(te)].tolist())
            assert tr_subs.isdisjoint(va_subs)
            assert tr_subs.isdisjoint(te_subs)
            assert va_subs.isdisjoint(te_subs)

    # 3. Cross-subject LOSO
    folds_loso = splits["cross_subject_loso"]
    assert len(folds_loso) == splits["n_subjects"]
    for fold in folds_loso:
        tr = set(fold["train_idx"])
        va = set(fold["val_idx"])
        te = set(fold["test_idx"])
        assert tr.isdisjoint(va)
        assert tr.isdisjoint(te)
        assert va.isdisjoint(te)
        assert tr | va | te == set(range(n_samples))

        tr_subs = set(subject_ids[list(tr)].tolist())
        va_subs = set(subject_ids[list(va)].tolist())
        te_subs = set(subject_ids[list(te)].tolist())
        assert tr_subs.isdisjoint(va_subs)
        assert tr_subs.isdisjoint(te_subs)
        assert va_subs.isdisjoint(te_subs)
        assert len(te_subs) == 1


def test_physiological_p300_sanity_check(data_dir: Path) -> None:
    """Kiểm tra sinh lý Grand Average ERP (Target - Frequent) tại kênh Pz."""
    X_clean = np.load(data_dir / "X_clean.npy")
    y = np.load(data_dir / "y.npy")
    times = np.load(data_dir / "times.npy")
    with open(data_dir / "channel_names.json", "r", encoding="utf-8") as f:
        ch_names = json.load(f)

    pz_idx = ch_names.index("Pz")
    target_erp = X_clean[y == 1, pz_idx, :].mean(axis=0)
    freq_erp = X_clean[y == 0, pz_idx, :].mean(axis=0)
    diff_pz = target_erp - freq_erp

    baseline_mask = (times >= -0.20) & (times <= 0.00)
    p300_mask = (times >= 0.30) & (times <= 0.60)

    baseline_mean_v = float(diff_pz[baseline_mask].mean())
    p300_window_diff = diff_pz[p300_mask]
    p300_window_times = times[p300_mask]

    peak_idx = int(np.argmax(p300_window_diff))
    peak_amp_uv = float(p300_window_diff[peak_idx] * 1e6)
    peak_latency_ms = float(p300_window_times[peak_idx] * 1000.0)

    print(
        f"\n[Sanity Check ERP @ Pz] baseline_mean={baseline_mean_v * 1e6:.4f} µV | "
        f"P300 peak={peak_amp_uv:.2f} µV @ {peak_latency_ms:.1f} ms"
    )

    assert abs(baseline_mean_v) < 1e-6, (
        f"Baseline ERP lệch quá lớn: {baseline_mean_v * 1e6:.4f} µV"
    )
    assert peak_amp_uv > 0.0, f"Không thấy đỉnh dương P300 trong [300, 600] ms: {peak_amp_uv:.2f} µV"
