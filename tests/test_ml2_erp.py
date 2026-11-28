"""Unit tests cho pipeline huấn luyện ML2 (SVM, KNN, GaussianNB) & chống Data Leakage."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ml2.ml2_erp import (
    VALID_ML2_MODELS,
    fit_predict_fold,
    run_ml2_experiment,
)


def _make_synthetic_dataset(
    n_subjects: int = 4,
    n_per_sub: int = 40,
    n_features: int = 20,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rows: List[Dict[str, Any]] = []
    global_idx = 0

    for s_idx in range(n_subjects):
        sub_id = f"sub-{s_idx + 1:02d}"
        # Lệch lớp ~6:1 (6 target, 34 frequent)
        y_sub = np.array([1] * 6 + [0] * (n_per_sub - 6), dtype=np.int64)
        rng.shuffle(y_sub)

        X_sub = rng.normal(0.0, 1.0, size=(n_per_sub, n_features))
        # Tạo tín hiệu tách biệt ở 4 đặc trưng đầu cho lớp Target
        X_sub[y_sub == 1, :4] += 2.2
        X_list.append(X_sub)
        y_list.append(y_sub)

        for t_idx in range(n_per_sub):
            lbl = int(y_sub[t_idx])
            rows.append(
                {
                    "trial_uid": f"{sub_id}_{t_idx}",
                    "subject_id": sub_id,
                    "trial_idx": global_idx,
                    "label": lbl,
                }
            )
            global_idx += 1

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    meta_df = pd.DataFrame(rows)
    return X, y, meta_df


def _write_synthetic_splits(splits_dir: Path, meta_df: pd.DataFrame) -> str:
    splits_dir.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(
        "\n".join(meta_df["trial_uid"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    subs = sorted(meta_df["subject_id"].unique().tolist())

    # 1. Within-subject 5-fold (giả lập 2-fold nhỏ gọn cho test hoặc 5-fold đầy đủ)
    within_dict: Dict[str, List[Dict[str, Any]]] = {}
    for sub in subs:
        sub_idx = np.flatnonzero(meta_df["subject_id"].to_numpy() == sub)
        pos = sub_idx[meta_df.iloc[sub_idx]["label"].to_numpy() == 1]
        neg = sub_idx[meta_df.iloc[sub_idx]["label"].to_numpy() == 0]
        # Tạo 5 folds phủ kín sub_idx
        pos_folds = np.array_split(pos, 5)
        neg_folds = np.array_split(neg, 5)
        folds_sub: List[Dict[str, Any]] = []
        for f in range(5):
            te = np.concatenate([pos_folds[f], neg_folds[f]])
            va_f = (f + 1) % 5
            va = np.concatenate([pos_folds[va_f], neg_folds[va_f]])
            tr = np.setdiff1d(sub_idx, np.concatenate([te, va]))
            folds_sub.append(
                {
                    "fold": f,
                    "train_idx": tr.tolist(),
                    "val_idx": va.tolist(),
                    "test_idx": te.tolist(),
                }
            )
        within_dict[sub] = folds_sub

    (splits_dir / "within_5fold.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "n_epochs": len(meta_df),
                "n_samples": len(meta_df),
                "n_subjects": len(subs),
                "trial_uid_sha256": sha,
                "within_subject": within_dict,
            }
        ),
        encoding="utf-8",
    )

    # 2. LOSO
    loso_list: List[Dict[str, Any]] = []
    for f, te_sub in enumerate(subs):
        va_sub = subs[(f + 1) % len(subs)]
        tr_subs = [s for s in subs if s not in (te_sub, va_sub)]
        te_idx = np.flatnonzero(meta_df["subject_id"].to_numpy() == te_sub)
        va_idx = np.flatnonzero(meta_df["subject_id"].to_numpy() == va_sub)
        tr_idx = np.flatnonzero(meta_df["subject_id"].isin(tr_subs).to_numpy())
        loso_list.append(
            {
                "fold": f,
                "train_subjects": tr_subs,
                "val_subjects": [va_sub],
                "test_subjects": [te_sub],
                "train_idx": tr_idx.tolist(),
                "val_idx": va_idx.tolist(),
                "test_idx": te_idx.tolist(),
            }
        )

    (splits_dir / "loso.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "n_epochs": len(meta_df),
                "n_samples": len(meta_df),
                "n_subjects": len(subs),
                "trial_uid_sha256": sha,
                "cross_subject_loso": loso_list,
            }
        ),
        encoding="utf-8",
    )
    return sha


@pytest.mark.parametrize("model_name", VALID_ML2_MODELS)
def test_fit_predict_fold_all_models_and_zero_leakage(model_name: str) -> None:
    """Kiểm tra hội tụ, xác suất hợp lệ và bất biến scaler/threshold trước nhiễu Test."""
    X, y, _ = _make_synthetic_dataset(n_subjects=4, n_per_sub=40, seed=42)
    tr = np.arange(0, 80)
    va = np.arange(80, 120)
    te = np.arange(120, 160)

    res_clean = fit_predict_fold(
        model_name, X[tr], y[tr], X[va], y[va], X[te], y[te], seed=42
    )
    m = res_clean["metrics"]
    assert 0.0 <= m["roc_auc"] <= 1.0 and m["roc_auc"] > 0.85
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert 0.05 <= res_clean["threshold"] <= 0.95
    assert np.isfinite(res_clean["y_prob"]).all()
    assert ((res_clean["y_prob"] >= 0.0) & (res_clean["y_prob"] <= 1.0)).all()
    assert set(np.unique(res_clean["y_pred"]).tolist()).issubset({0, 1})

    # Phá nhiễu dữ liệu Test cực mạnh (+1e6): không được làm đổi scaler, best_params hay threshold!
    X_te_corrupt = X[te].copy() + 1e6
    res_corrupt = fit_predict_fold(
        model_name, X[tr], y[tr], X[va], y[va], X_te_corrupt, y[te], seed=42
    )
    assert res_corrupt["scaler_mean_norm"] == pytest.approx(
        res_clean["scaler_mean_norm"], rel=0.0, abs=1e-12
    )
    assert res_corrupt["best_params"] == res_clean["best_params"]
    assert res_corrupt["threshold"] == pytest.approx(
        res_clean["threshold"], rel=0.0, abs=1e-12
    )


def test_run_ml2_experiment_end_to_end(tmp_path: Path) -> None:
    """Kiểm tra chạy trọn bộ within và loso trên dữ liệu tổng hợp & xuất file đúng chuẩn."""
    X, y, meta_df = _make_synthetic_dataset(n_subjects=4, n_per_sub=40, seed=11)
    splits_dir = tmp_path / "splits"
    out_dir = tmp_path / "out"
    sha = _write_synthetic_splits(splits_dir, meta_df)

    for proto, expected_folds in (("within", 20), ("loso", 4)):
        res = run_ml2_experiment(
            model_name="svm",
            protocol_alias=proto,
            X_feat=X,
            y=y,
            meta_df=meta_df,
            splits_dir=splits_dir,
            output_dir=out_dir,
            feature_set="erp_full_binned",
            n_jobs=1,
            expected_sha256=sha,
        )
        assert res["n_folds"] == expected_folds
        assert res["summary"]["roc_auc"]["mean"] > 0.85

        json_path = Path(res["metrics_json"])
        csv_path = Path(res["predictions_csv"])
        assert json_path.exists() and csv_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["trial_uid_sha256"] == sha
        assert len(payload["per_fold"]) == expected_folds

        pred_df = pd.read_csv(csv_path)
        assert len(pred_df) == len(meta_df)
        assert np.array_equal(pred_df["y_true"].to_numpy(), y)
        assert (pred_df["fold"] >= 0).all()
