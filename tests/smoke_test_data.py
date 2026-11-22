"""Kiểm thử dữ liệu Lớp 2 (Data Smoke Test) cho bài toán EEG Oddball P300.

Tệp: tests/smoke_test_data.py

Xác thực bộ dữ liệu Single Source of Truth tại ``data/processed/`` và các tệp
chia tập cố định tại ``splits/`` trước khi 5 thành viên huấn luyện mô hình.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Set, Tuple

import mne
import numpy as np
import pandas as pd
import pytest

EXPECTED_N_SUBJECTS = 39
EXCLUDED_SUBJECTS: Set[str] = {"sub-02", "sub-14", "sub-17"}
EXPECTED_INCLUDED_SUBJECTS: Set[str] = {"sub-08", "sub-38"}

EXPECTED_N_CLEAN = 20886
EXPECTED_N_CLEAN_FREQ = 18036
EXPECTED_N_CLEAN_TARGET = 2850

EXPECTED_N_NO_ICA = 22477
EXPECTED_N_NO_ICA_FREQ = 19475
EXPECTED_N_NO_ICA_TARGET = 3002

EXPECTED_N_CHANNELS = 127
EXPECTED_N_TIMES = 251
EXPECTED_SFREQ = 250.0
EXPECTED_SEED = 42

EXPECTED_TRIAL_UID_SHA256 = (
    "7c5ce364eec45fc9c7d2ac3a2fc5d5723fd961b5953d9653cf403acda19eb524"
)
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


@pytest.fixture(scope="module")
def data_dir() -> Path:
    """Trả về thư mục dữ liệu đã tiền xử lý hoặc fail rõ ràng nếu thiếu."""
    env_dir = os.environ.get("EEG_DATA_DIR")
    d = Path(env_dir) if env_dir else Path("data/processed")
    if not (d / "X_clean.npy").exists():
        pytest.fail(f"Không tìm thấy dữ liệu tiền xử lý (X_clean.npy) tại: {d.resolve()}")
    return d


@pytest.fixture(scope="module")
def splits_dir() -> Path:
    """Trả về thư mục splits cố định của nhóm."""
    env_dir = os.environ.get("EEG_SPLITS_DIR")
    d = Path(env_dir) if env_dir else Path("splits")
    if not d.is_dir():
        pytest.fail(f"Không tìm thấy thư mục splits tại: {d.resolve()}")
    return d


def _scan_mmap_array(
    arr: np.ndarray,
    chunk_size: int = 2048,
) -> Tuple[bool, float, float]:
    """Quét 100% mảng (N, C, T) theo từng chunk để tiết kiệm RAM."""
    n_total = int(arr.shape[0])
    if n_total == 0:
        return True, 0.0, 0.0

    all_finite = True
    max_abs = 0.0
    min_epoch_std = float("inf")

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk = np.asarray(arr[start:end], dtype=np.float64)
        if not np.isfinite(chunk).all():
            all_finite = False
        chunk_max = float(np.max(np.abs(chunk)))
        if chunk_max > max_abs:
            max_abs = chunk_max
        chunk_min_std = float(np.min(np.std(chunk, axis=(1, 2))))
        if chunk_min_std < min_epoch_std:
            min_epoch_std = chunk_min_std

    return all_finite, max_abs, min_epoch_std


def _validate_disjoint_fold(
    fold: Dict[str, Any],
    expected_universe: Set[int],
    context: str,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Kiểm tra 3 tập train/val/test rời nhau từng đôi và phủ đúng tập chuẩn."""
    tr = set(fold["train_idx"])
    va = set(fold["val_idx"])
    te = set(fold["test_idx"])
    assert tr.isdisjoint(va), f"{context}: rò rỉ giữa train_idx và val_idx"
    assert tr.isdisjoint(te), f"{context}: rò rỉ giữa train_idx và test_idx"
    assert va.isdisjoint(te), f"{context}: rò rỉ giữa val_idx và test_idx"
    assert (tr | va | te) == expected_universe, f"{context}: hợp 3 tập không khớp tập mẫu chuẩn"
    return tr, va, te


def _validate_subject_disjoint(
    tr: Iterable[int],
    va: Iterable[int],
    te: Iterable[int],
    subject_ids: np.ndarray,
    context: str,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Kiểm tra tuyệt đối không trùng subject_id giữa train/val/test."""
    tr_subs = set(subject_ids[list(tr)].tolist())
    va_subs = set(subject_ids[list(va)].tolist())
    te_subs = set(subject_ids[list(te)].tolist())
    assert tr_subs.isdisjoint(va_subs), f"{context}: rò rỉ subject giữa train và val"
    assert tr_subs.isdisjoint(te_subs), f"{context}: rò rỉ subject giữa train và test"
    assert va_subs.isdisjoint(te_subs), f"{context}: rò rỉ subject giữa val và test"
    return tr_subs, va_subs, te_subs


def test_clean_and_no_ica_contract(data_dir: Path) -> None:
    """Kiểm tra shape, dtype, đơn vị Volts, số nhãn và khớp 1-1 với tập no_ica."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    subject_ids = np.load(data_dir / "subject_ids.npy")
    times = np.load(data_dir / "times.npy")

    assert X_clean.shape == (EXPECTED_N_CLEAN, EXPECTED_N_CHANNELS, EXPECTED_N_TIMES)
    assert X_clean.dtype == np.float32
    clean_finite, clean_max_abs, clean_min_std = _scan_mmap_array(X_clean)
    assert clean_finite, "X_clean chứa NaN hoặc Inf"
    assert clean_max_abs < 1e-2, f"Biên độ X_clean vượt ngưỡng Volts (<1e-2 V): {clean_max_abs}"
    assert clean_min_std > 0.0, "Phát hiện epoch phẳng hoàn toàn (std == 0) trong X_clean"

    assert y.shape == (EXPECTED_N_CLEAN,)
    assert y.dtype == np.int64
    assert set(np.unique(y).tolist()) == {0, 1}
    assert int((y == 0).sum()) == EXPECTED_N_CLEAN_FREQ
    assert int((y == 1).sum()) == EXPECTED_N_CLEAN_TARGET

    assert subject_ids.shape == (EXPECTED_N_CLEAN,)
    assert subject_ids.dtype.kind == "U"

    assert times.shape == (EXPECTED_N_TIMES,)
    assert times.dtype == np.float64
    assert np.isclose(times[0], -0.2, atol=1e-6)
    assert np.isclose(times[-1], 0.8, atol=1e-6)
    assert np.allclose(np.diff(times), 1.0 / EXPECTED_SFREQ, atol=1e-6)

    X_no_ica = np.load(data_dir / "X_no_ica_no_reject.npy", mmap_mode="r")
    y_no_ica = np.load(data_dir / "y_no_ica_no_reject.npy")
    sub_no_ica = np.load(data_dir / "subject_ids_no_ica_no_reject.npy")
    df_no_ica = pd.read_csv(data_dir / "metadata_no_ica.csv")
    df_clean = pd.read_csv(data_dir / "metadata.csv")

    assert X_no_ica.shape == (EXPECTED_N_NO_ICA, EXPECTED_N_CHANNELS, EXPECTED_N_TIMES)
    assert X_no_ica.dtype == np.float32
    no_ica_finite, _, _ = _scan_mmap_array(X_no_ica)
    assert no_ica_finite, "X_no_ica_no_reject chứa NaN hoặc Inf"

    assert y_no_ica.shape == (EXPECTED_N_NO_ICA,)
    assert y_no_ica.dtype == np.int64
    assert int((y_no_ica == 0).sum()) == EXPECTED_N_NO_ICA_FREQ
    assert int((y_no_ica == 1).sum()) == EXPECTED_N_NO_ICA_TARGET
    assert sub_no_ica.shape == (EXPECTED_N_NO_ICA,)
    assert sub_no_ica.dtype.kind == "U"

    assert len(df_no_ica) == EXPECTED_N_NO_ICA
    assert "kept_in_clean" in df_no_ica.columns
    kept_mask = df_no_ica["kept_in_clean"].astype(bool)
    assert int(kept_mask.sum()) == EXPECTED_N_CLEAN
    kept_uids = df_no_ica.loc[kept_mask, "trial_uid"].astype(str).tolist()
    assert kept_uids == df_clean["trial_uid"].astype(str).tolist(), (
        "Danh sách trial_uid có kept_in_clean=True không khớp 1-1 với metadata.csv"
    )


def test_cohort_39_subjects_and_sha256(data_dir: Path) -> None:
    """Xác thực đúng 39 subjects, loại 3 subject hành vi lỗi, giữ sub-38 và khớp SHA-256."""
    subject_ids = np.load(data_dir / "subject_ids.npy")
    df = pd.read_csv(data_dir / "metadata.csv")

    subs_npy = set(np.unique(subject_ids).tolist())
    subs_csv = set(df["subject_id"].astype(str).unique().tolist())

    assert subs_npy == subs_csv
    assert len(subs_npy) == EXPECTED_N_SUBJECTS
    assert subs_npy.isdisjoint(EXCLUDED_SUBJECTS), (
        f"Phát hiện subject bị loại trong tập sạch: {subs_npy & EXCLUDED_SUBJECTS}"
    )
    assert EXPECTED_INCLUDED_SUBJECTS.issubset(subs_npy), (
        f"Thiếu subject bắt buộc phải giữ lại: {EXPECTED_INCLUDED_SUBJECTS - subs_npy}"
    )

    actual_sha = hashlib.sha256(
        "\n".join(df["trial_uid"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    assert actual_sha == EXPECTED_TRIAL_UID_SHA256, (
        f"trial_uid_sha256 không khớp chuẩn nhóm: {actual_sha} != {EXPECTED_TRIAL_UID_SHA256}"
    )


def test_sub08_and_signal_sanity(data_dir: Path) -> None:
    """Kiểm tra sub-08 đã được crop sạch đoạn bão hòa rút mũ (không còn biên độ cực đại)."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    subject_ids = np.load(data_dir / "subject_ids.npy")

    sub08_mask = subject_ids == "sub-08"
    assert int(sub08_mask.sum()) == 545

    X08_uv = np.asarray(X_clean[sub08_mask], dtype=np.float64) * 1e6
    max_abs_uv = float(np.max(np.abs(X08_uv)))
    median_std_uv = float(np.median(np.std(X08_uv, axis=-1)))

    assert max_abs_uv < 500.0, f"sub-08 còn biên độ bão hòa bất thường: {max_abs_uv:.2f} µV"
    assert median_std_uv < 50.0, f"sub-08 còn std bất thường: {median_std_uv:.2f} µV"


def test_channels_and_colin27_montage(data_dir: Path) -> None:
    """Kiểm tra 127 kênh chuẩn, đã đổi 4 alias và khớp 100% montage colin27_1005."""
    with open(data_dir / "channel_names.json", "r", encoding="utf-8") as f:
        ch_names = json.load(f)

    assert len(ch_names) == EXPECTED_N_CHANNELS
    assert len(set(ch_names)) == EXPECTED_N_CHANNELS

    ch_set = set(ch_names)
    for expected in ("TP9", "TP10", "I1", "I2"):
        assert expected in ch_set, f"Thiếu kênh chuẩn '{expected}'"
    for legacy in ("O9", "O10", "TP9 LEFT EAR", "TP10 RIGHT EAR"):
        assert legacy not in ch_set, f"Vẫn còn tên kênh cũ chưa ánh xạ '{legacy}'"

    montage = mne.channels.make_standard_montage("colin27_1005")
    montage_chs = set(montage.ch_names)
    missing = [ch for ch in ch_names if ch not in montage_chs]
    assert not missing, f"Các kênh không khớp colin27_1005: {missing}"


def test_metadata_rt_and_qc_summary(data_dir: Path) -> None:
    """Kiểm tra schema metadata.csv, tính hợp lệ rt_ms/rt_group và preprocessing_qc_summary.csv."""
    y = np.load(data_dir / "y.npy")
    subject_ids = np.load(data_dir / "subject_ids.npy")

    df = pd.read_csv(data_dir / "metadata.csv", keep_default_na=True)
    assert list(df.columns) == EXPECTED_META_COLS
    assert len(df) == EXPECTED_N_CLEAN
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
    assert (df.loc[target_mask & (df["rt_ms"] < 450.0), "rt_group"] == "fast").all()
    assert (
        df.loc[
            target_mask & (df["rt_ms"] >= 450.0) & (df["rt_ms"] <= 550.0), "rt_group"
        ]
        == "mid"
    ).all()
    assert (df.loc[target_mask & (df["rt_ms"] > 550.0), "rt_group"] == "slow").all()

    assert df.loc[freq_mask, "rt_ms"].isna().all(), "Có trial Frequent có rt_ms khác NaN"
    assert (df.loc[freq_mask, "rt_group"].astype(str) == "na").all()

    qc_df = pd.read_csv(data_dir / "preprocessing_qc_summary.csv")
    assert len(qc_df) == EXPECTED_N_SUBJECTS
    assert set(qc_df["subject_id"].astype(str).tolist()) == set(subject_ids.tolist())

    expected_counts = df["subject_id"].astype(str).value_counts().to_dict()
    for _, row in qc_df.iterrows():
        sub = str(row["subject_id"])
        assert int(row["n_clean_epochs_kept"]) == int(expected_counts[sub]), (
            f"QC summary không khớp số epoch sạch tại {sub}"
        )


def test_splits_dir_consistency_and_zero_leakage(
    data_dir: Path,
    splits_dir: Path,
) -> None:
    """Kiểm tra trực tiếp 3 file trong splits/, zero leakage và đồng bộ với splits.json."""
    df = pd.read_csv(data_dir / "metadata.csv")
    subject_ids = df["subject_id"].astype(str).to_numpy()
    universe = set(range(EXPECTED_N_CLEAN))

    split_files = {
        "within_subject": splits_dir / "within_5fold.json",
        "cross_subject_5fold": splits_dir / "cross_5fold.json",
        "cross_subject_loso": splits_dir / "loso.json",
    }
    loaded: Dict[str, Dict[str, Any]] = {}
    for key_name, fpath in split_files.items():
        assert fpath.exists(), f"Thiếu file split bắt buộc: {fpath}"
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        assert int(payload["seed"]) == EXPECTED_SEED
        assert int(payload["n_epochs"]) == EXPECTED_N_CLEAN
        assert int(payload["n_samples"]) == EXPECTED_N_CLEAN
        assert int(payload["n_subjects"]) == EXPECTED_N_SUBJECTS
        assert payload["trial_uid_sha256"] == EXPECTED_TRIAL_UID_SHA256
        loaded[key_name] = payload

    # 1. Within-Subject 5-Fold (39 subjects x 5 folds = 195 folds)
    within = loaded["within_subject"]["within_subject"]
    assert set(within.keys()) == set(subject_ids.tolist())
    total_within_folds = 0
    for sub_id, folds in within.items():
        sub_universe = set(np.where(subject_ids == sub_id)[0].tolist())
        assert len(folds) == 5
        te_union: Set[int] = set()
        for idx, fold in enumerate(folds):
            _, _, te = _validate_disjoint_fold(
                fold, sub_universe, f"within_5fold/{sub_id}/fold_{idx}"
            )
            assert te_union.isdisjoint(te), f"within_5fold/{sub_id}: test_idx bị lặp giữa các fold"
            te_union |= te
            total_within_folds += 1
        assert te_union == sub_universe, f"within_5fold/{sub_id}: các fold test không phủ đủ subject"
    assert total_within_folds == EXPECTED_N_SUBJECTS * 5

    # 2. Cross-Subject 5-Fold
    folds_cs5 = loaded["cross_subject_5fold"]["cross_subject_5fold"]
    assert len(folds_cs5) == 5
    cs5_te_union: Set[int] = set()
    for idx, fold in enumerate(folds_cs5):
        tr, va, te = _validate_disjoint_fold(fold, universe, f"cross_5fold/fold_{idx}")
        _validate_subject_disjoint(tr, va, te, subject_ids, f"cross_5fold/fold_{idx}")
        assert cs5_te_union.isdisjoint(te), "cross_5fold: test_idx bị lặp giữa các fold"
        cs5_te_union |= te
    assert cs5_te_union == universe, "cross_5fold: 5 tập test không phủ đủ 20,886 mẫu"

    # 3. Cross-Subject LOSO (39 folds)
    folds_loso = loaded["cross_subject_loso"]["cross_subject_loso"]
    assert len(folds_loso) == EXPECTED_N_SUBJECTS
    loso_te_union: Set[int] = set()
    seen_te_subs: Set[str] = set()
    for idx, fold in enumerate(folds_loso):
        tr, va, te = _validate_disjoint_fold(fold, universe, f"loso/fold_{idx}")
        _, _, te_subs = _validate_subject_disjoint(
            tr, va, te, subject_ids, f"loso/fold_{idx}"
        )
        assert len(te_subs) == 1, f"loso/fold_{idx}: kỳ vọng 1 test subject, nhận {te_subs}"
        sub = next(iter(te_subs))
        assert sub not in seen_te_subs, f"loso: subject {sub} xuất hiện ở >1 fold test"
        seen_te_subs.add(sub)
        assert loso_te_union.isdisjoint(te)
        loso_te_union |= te
    assert seen_te_subs == set(subject_ids.tolist())
    assert loso_te_union == universe

    # 4. Đối chiếu khớp 100% với data/processed/splits.json (nếu có)
    master_splits_path = data_dir / "splits.json"
    if master_splits_path.exists():
        master = json.loads(master_splits_path.read_text(encoding="utf-8"))
        assert master["trial_uid_sha256"] == EXPECTED_TRIAL_UID_SHA256
        for key_name, payload in loaded.items():
            assert payload[key_name] == master[key_name], (
                f"splits/ ({key_name}) không khớp với data/processed/splits.json"
            )


def test_physiological_p300_pz_sanity(data_dir: Path) -> None:
    """Kiểm tra sinh lý Grand Average ERP (Target - Frequent) tại Pz."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    times = np.load(data_dir / "times.npy")
    ch_names = json.loads((data_dir / "channel_names.json").read_text(encoding="utf-8"))

    pz_idx = ch_names.index("Pz")
    pz_uv = np.asarray(X_clean[:, pz_idx, :], dtype=np.float64) * 1e6
    diff_pz_uv = pz_uv[y == 1].mean(axis=0) - pz_uv[y == 0].mean(axis=0)

    baseline_mask = (times >= -0.20) & (times <= 0.00)
    p300_mask = (times >= 0.30) & (times <= 0.60)

    baseline_mean_uv = float(diff_pz_uv[baseline_mask].mean())
    peak_amp_uv = float(np.max(diff_pz_uv[p300_mask]))

    assert abs(baseline_mean_uv) < 1.0, f"Baseline ERP lệch bất thường: {baseline_mean_uv:.4f} µV"
    assert peak_amp_uv > 3.0, f"Đỉnh dương P300 tại Pz quá yếu: {peak_amp_uv:.2f} µV"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
