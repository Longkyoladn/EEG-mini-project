"""Kiểm thử Schema bàn giao Mục 3.5 cho 4 file ML2 trong results/handover_standard/."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import compute_binary_metrics

HANDOVER_DIR = Path("results/handover_standard")
DATA_DIR = Path("data/processed")

EXPECTED_CSV_COLS = [
    "subject_id",
    "fold",
    "bal_acc",
    "roc_auc",
    "pr_auc",
    "f1_target",
    "sensitivity",
    "specificity",
]

EXPECTED_NPZ_KEYS = {
    "subject_id",
    "fold",
    "trial_idx",
    "trial_uid",
    "y_true",
    "y_prob",
    "y_pred",
}


@pytest.mark.parametrize(
    ("prefix", "expected_rows"),
    [("ml2_erp_svm_within", 195), ("ml2_erp_svm_loso", 39)],
)
def test_ml2_handover_per_fold_csv_schema(prefix: str, expected_rows: int) -> None:
    """Xác thực đúng 8 cột, số dòng 195/39, 39 subjects và miền giá trị [0, 1]."""
    csv_path = HANDOVER_DIR / f"{prefix}_per_fold.csv"
    if not csv_path.exists():
        pytest.fail(f"Thiếu file handover CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    assert list(df.columns) == EXPECTED_CSV_COLS
    assert len(df) == expected_rows
    assert not df.isna().any().any()
    assert df["subject_id"].nunique() == 39

    for col in EXPECTED_CSV_COLS[2:]:
        vals = df[col].to_numpy(dtype=np.float64)
        assert np.isfinite(vals).all()
        assert ((vals >= 0.0) & (vals <= 1.0)).all()


@pytest.mark.parametrize("prefix", ["ml2_erp_svm_within", "ml2_erp_svm_loso"])
def test_ml2_handover_preds_npz_schema_and_alignment(prefix: str) -> None:
    """Xác thực NPZ đọc được với allow_pickle=False, đủ 7 mảng và khớp metadata.csv."""
    npz_path = HANDOVER_DIR / f"{prefix}_preds.npz"
    csv_path = HANDOVER_DIR / f"{prefix}_per_fold.csv"
    meta_path = DATA_DIR / "metadata.csv"

    if not npz_path.exists():
        pytest.fail(f"Thiếu file handover NPZ: {npz_path}")
    if not meta_path.exists():
        pytest.skip("Không tìm thấy data/processed/metadata.csv")

    meta_df = pd.read_csv(meta_path)
    n_samples = len(meta_df)
    assert n_samples == 20886

    with np.load(npz_path, allow_pickle=False) as z:
        assert set(z.files) == EXPECTED_NPZ_KEYS
        for key in EXPECTED_NPZ_KEYS:
            arr = z[key]
            assert arr.shape == (n_samples,), f"{prefix}/{key}: shape {arr.shape}"
            assert arr.dtype.kind != "O", f"{prefix}/{key}: không được dùng dtype object"

        assert np.array_equal(z["trial_idx"], np.arange(n_samples, dtype=np.int64))
        assert np.array_equal(
            z["trial_uid"].astype(str), meta_df["trial_uid"].astype(str).to_numpy()
        )
        assert np.array_equal(
            z["subject_id"].astype(str), meta_df["subject_id"].astype(str).to_numpy()
        )
        assert np.array_equal(
            z["y_true"].astype(np.int64), meta_df["label"].to_numpy(dtype=np.int64)
        )
        y_prob = z["y_prob"].astype(np.float64)
        y_pred = z["y_pred"].astype(np.int64)
        assert np.isfinite(y_prob).all()
        assert ((y_prob >= 0.0) & (y_prob <= 1.0)).all()
        assert set(np.unique(y_pred).tolist()).issubset({0, 1})

        # Đối chiếu lại roc_auc & pr_auc từng fold với _per_fold.csv
        fold_df = pd.read_csv(csv_path)
        sub_arr = z["subject_id"].astype(str)
        fold_arr = z["fold"]
        y_true = z["y_true"]
        for _, row in fold_df.iterrows():
            mask = (sub_arr == str(row["subject_id"])) & (fold_arr == int(row["fold"]))
            m = compute_binary_metrics(y_true[mask], y_prob[mask], threshold=0.5)
            assert m["roc_auc"] == pytest.approx(float(row["roc_auc"]), abs=1e-9)
            assert m["pr_auc"] == pytest.approx(float(row["pr_auc"]), abs=1e-9)


def test_external_benchmark_grouping_contract() -> None:
    """Kiểm tra trực tiếp đúng hợp đồng tổng hợp 39 subjects cho Fig 3 / Fig 3b."""
    for prefix in ("ml2_erp_svm_within", "ml2_erp_svm_loso"):
        df = pd.read_csv(HANDOVER_DIR / f"{prefix}_per_fold.csv")
        grouped = df.groupby("subject_id", sort=False, as_index=False)[
            ["roc_auc", "bal_acc"]
        ].mean()
        assert len(grouped) == 39
        assert np.isfinite(grouped[["roc_auc", "bal_acc"]].to_numpy()).all()


def test_visualization_loaders_compatibility() -> None:
    """Đảm bảo pipeline vẽ Fig 3 & Fig 3b của nhóm đọc mượt 4 file ML2 (nếu có captum)."""
    pytest.importorskip("captum")
    from src.evaluation.visualization import (
        _load_external_benchmark_rows,
        _load_preds_npz,
    )

    rows = _load_external_benchmark_rows(HANDOVER_DIR)
    df = pd.DataFrame(rows)
    assert {"ml2 erp svm within", "ml2 erp svm loso"}.issubset(set(df["label"].unique()))
    for lbl in ("ml2 erp svm within", "ml2 erp svm loso"):
        sub_df = df[df["label"] == lbl]
        assert len(sub_df) == 39

    for prefix in ("ml2_erp_svm_within", "ml2_erp_svm_loso"):
        payload = _load_preds_npz(HANDOVER_DIR / f"{prefix}_preds.npz")
        assert payload["y_true"].shape == (20886,)
        assert payload["y_prob"].shape == (20886,)
        assert payload["subject_id"].shape == (20886,)
