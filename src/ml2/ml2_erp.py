"""Pipeline huấn luyện & đánh giá ML Nhánh 2 (SVM, KNN, GaussianNB) trên đặc trưng ERP.

Tuân thủ tuyệt đối Zero Data Leakage:
- StandardScaler chỉ fit trên ``train_idx`` của từng fold.
- Siêu tham số, Platt calibration (với SVM) và ngưỡng quyết định chỉ học trên ``val_idx``.
- Tập ``test_idx`` chỉ được dự đoán và chấm điểm đúng 1 lần cuối cùng.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

from src.evaluation.metrics import (
    _sanitize_json_value,
    aggregate_metrics,
    compute_binary_metrics,
    find_optimal_threshold,
    save_metrics_json,
    save_predictions_csv,
)
from src.features.erp_time_features import ERPFeatureExtractor, load_data_geometry

VALID_ML2_MODELS: Tuple[str, ...] = ("svm", "knn", "gnb")
VALID_PROTOCOLS: Tuple[str, ...] = ("within", "loso")

PROTOCOL_META: Dict[str, Tuple[str, str, str]] = {
    "within": ("within_5fold.json", "within_subject", "within_subject"),
    "loso": ("loso.json", "cross_subject_loso", "cross_subject_loso"),
}

EXPECTED_TRIAL_UID_SHA256 = (
    "7c5ce364eec45fc9c7d2ac3a2fc5d5723fd961b5953d9653cf403acda19eb524"
)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _fit_platt_calibrator(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> Optional[LogisticRegression]:
    """Hiệu chuẩn Platt 1D trên điểm decision_function của tập Validation."""
    scores_arr = np.asarray(val_scores, dtype=np.float64).ravel()
    y_arr = np.asarray(y_val, dtype=np.int64).ravel()
    if np.unique(y_arr).size < 2 or float(np.std(scores_arr)) < 1e-12:
        return None
    cal = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed)
    cal.fit(scores_arr.reshape(-1, 1), y_arr)
    if float(cal.coef_[0, 0]) <= 0.0:
        return None
    return cal


def _apply_platt(
    calibrator: Optional[LogisticRegression],
    scores: np.ndarray,
) -> np.ndarray:
    scores_arr = np.asarray(scores, dtype=np.float64).ravel()
    if calibrator is None:
        return _sigmoid(scores_arr)
    probs = calibrator.predict_proba(scores_arr.reshape(-1, 1))[:, 1]
    return np.clip(probs, 0.0, 1.0)


def _safe_val_roc_auc(y_val: np.ndarray, val_scores: np.ndarray) -> float:
    if np.unique(y_val).size < 2 or not np.isfinite(val_scores).all():
        return 0.5
    return float(roc_auc_score(y_val, val_scores))


def _build_svm_candidates(seed: int) -> List[Tuple[Dict[str, Any], Any]]:
    candidates: List[Tuple[Dict[str, Any], Any]] = []
    for c_val in (0.3, 1.0, 3.0):
        for gamma_val in ("scale", 0.003):
            params = {"kernel": "rbf", "C": c_val, "gamma": gamma_val}
            clf = SVC(
                kernel="rbf",
                C=c_val,
                gamma=gamma_val,
                class_weight="balanced",
                probability=False,
                cache_size=1000,
                tol=1e-3,
                random_state=seed,
            )
            candidates.append((params, clf))
    for c_val in (0.01, 0.1, 1.0):
        params = {"kernel": "linear", "C": c_val}
        clf = LinearSVC(
            C=c_val,
            class_weight="balanced",
            dual="auto",
            max_iter=5000,
            random_state=seed,
        )
        candidates.append((params, clf))
    return candidates


def _build_knn_candidates(n_train: int) -> List[Tuple[Dict[str, Any], Any]]:
    candidates: List[Tuple[Dict[str, Any], Any]] = []
    max_k = max(1, n_train)
    ks = sorted({min(k, max_k) for k in (5, 11, 21, 35)})
    for k_val in ks:
        for metric in ("euclidean", "manhattan"):
            params = {"n_neighbors": k_val, "metric": metric, "weights": "distance"}
            clf = KNeighborsClassifier(
                n_neighbors=k_val,
                weights="distance",
                metric=metric,
            )
            candidates.append((params, clf))
    return candidates


def _build_gnb_candidates() -> List[Tuple[Dict[str, Any], Any]]:
    candidates: List[Tuple[Dict[str, Any], Any]] = []
    for vs in (1e-9, 1e-6, 1e-3, 1e-1):
        params = {"var_smoothing": vs}
        clf = GaussianNB(var_smoothing=vs)
        candidates.append((params, clf))
    return candidates


def fit_predict_fold(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """Huấn luyện, chọn siêu tham số/ngưỡng trên Val và đánh giá đúng 1 lần trên Test."""
    if model_name not in VALID_ML2_MODELS:
        raise ValueError(
            f"model_name không hợp lệ: {model_name!r}. Hợp lệ: {VALID_ML2_MODELS}"
        )

    t0 = time.perf_counter()

    # Zero Data Leakage: chỉ fit StandardScaler trên tập Train của fold hiện tại
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(np.asarray(X_train, dtype=np.float64))
    X_va_sc = scaler.transform(np.asarray(X_val, dtype=np.float64))
    X_te_sc = scaler.transform(np.asarray(X_test, dtype=np.float64))

    if model_name == "svm":
        candidates = _build_svm_candidates(seed)
    elif model_name == "knn":
        candidates = _build_knn_candidates(len(X_tr_sc))
    else:
        candidates = _build_gnb_candidates()

    best_auc = -1.0
    best_params: Dict[str, Any] = {}
    best_clf: Any = None
    best_val_raw: Optional[np.ndarray] = None

    for params, clf in candidates:
        clf.fit(X_tr_sc, y_train)
        if model_name == "svm":
            val_raw = np.asarray(clf.decision_function(X_va_sc), dtype=np.float64).ravel()
        else:
            val_raw = np.asarray(clf.predict_proba(X_va_sc)[:, 1], dtype=np.float64).ravel()

        val_auc = _safe_val_roc_auc(y_val, val_raw)
        if val_auc > best_auc:
            best_auc = val_auc
            best_params = params
            best_clf = clf
            best_val_raw = val_raw

    assert best_clf is not None and best_val_raw is not None

    if model_name == "svm":
        calibrator = _fit_platt_calibrator(best_val_raw, y_val, seed=seed)
        val_prob = _apply_platt(calibrator, best_val_raw)
        test_raw = np.asarray(
            best_clf.decision_function(X_te_sc), dtype=np.float64
        ).ravel()
        test_prob = _apply_platt(calibrator, test_raw)
    else:
        val_prob = np.clip(best_val_raw, 0.0, 1.0)
        test_prob = np.clip(
            np.asarray(best_clf.predict_proba(X_te_sc)[:, 1], dtype=np.float64).ravel(),
            0.0,
            1.0,
        )

    threshold = find_optimal_threshold(y_val, val_prob, metric="balanced_accuracy")
    metrics = compute_binary_metrics(y_test, test_prob, threshold=threshold)
    y_pred = (test_prob >= threshold).astype(np.int64)
    train_time_sec = time.perf_counter() - t0

    return {
        "metrics": metrics,
        "threshold": float(threshold),
        "val_roc_auc": float(best_auc),
        "best_params": best_params,
        "y_prob": test_prob,
        "y_pred": y_pred,
        "train_time_sec": float(train_time_sec),
        "scaler_mean_norm": float(np.linalg.norm(scaler.mean_)),
    }


def _run_single_fold_task(
    task_idx: int,
    subject_id: str,
    fold_id: int,
    test_idx: np.ndarray,
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    res = fit_predict_fold(
        model_name=model_name,
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_va,
        y_val=y_va,
        X_test=X_te,
        y_test=y_te,
        seed=seed,
    )
    return {
        "task_idx": task_idx,
        "subject_id": subject_id,
        "fold": int(fold_id),
        "test_idx": test_idx,
        **res,
    }


def load_protocol_folds(
    splits_dir: Path,
    protocol_alias: str,
    expected_sha256: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]], str]:
    """Nạp danh sách các fold từ ``splits/within_5fold.json`` hoặc ``splits/loso.json``."""
    if protocol_alias not in PROTOCOL_META:
        raise ValueError(
            f"protocol không hợp lệ: {protocol_alias!r}. Hợp lệ: {VALID_PROTOCOLS}"
        )
    fname, json_key, protocol_name = PROTOCOL_META[protocol_alias]
    payload = json.loads((splits_dir / fname).read_text(encoding="utf-8"))

    sha = str(payload["trial_uid_sha256"])
    if expected_sha256 is not None and sha != expected_sha256:
        raise ValueError(
            f"{fname}: trial_uid_sha256 không khớp ({sha} != {expected_sha256})"
        )

    tasks: List[Dict[str, Any]] = []
    if protocol_alias == "within":
        within_dict = payload[json_key]
        for sub_id in sorted(within_dict.keys()):
            for fold_obj in within_dict[sub_id]:
                tasks.append(
                    {
                        "subject_id": str(sub_id),
                        "fold": int(fold_obj["fold"]),
                        "train_idx": np.asarray(fold_obj["train_idx"], dtype=np.int64),
                        "val_idx": np.asarray(fold_obj["val_idx"], dtype=np.int64),
                        "test_idx": np.asarray(fold_obj["test_idx"], dtype=np.int64),
                    }
                )
    else:
        for fold_obj in payload[json_key]:
            te_subs = fold_obj["test_subjects"]
            sub_id = str(te_subs[0])
            tasks.append(
                {
                    "subject_id": sub_id,
                    "fold": int(fold_obj["fold"]),
                    "train_idx": np.asarray(fold_obj["train_idx"], dtype=np.int64),
                    "val_idx": np.asarray(fold_obj["val_idx"], dtype=np.int64),
                    "test_idx": np.asarray(fold_obj["test_idx"], dtype=np.int64),
                }
            )
    return protocol_name, tasks, sha


def run_ml2_experiment(
    *,
    model_name: str,
    protocol_alias: str,
    X_feat: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    splits_dir: Path,
    output_dir: Path,
    feature_set: str,
    channel_preset: str = "core_14",
    exp_name: Optional[str] = None,
    n_jobs: int = -1,
    seed: int = 42,
    expected_sha256: Optional[str] = EXPECTED_TRIAL_UID_SHA256,
) -> Dict[str, Any]:
    """Chạy toàn bộ các fold của một cấu hình (model, protocol, feature_set) và lưu kết quả."""
    if model_name not in VALID_ML2_MODELS:
        raise ValueError(f"model_name không hợp lệ: {model_name!r}")

    actual_sha = hashlib.sha256(
        "\n".join(meta_df["trial_uid"].astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError(
            f"SHA-256 của metadata.csv không khớp chuẩn nhóm: {actual_sha}"
        )

    protocol_name, fold_tasks, split_sha = load_protocol_folds(
        splits_dir, protocol_alias, expected_sha256=actual_sha
    )

    resolved_exp = exp_name or f"ml2_{feature_set}_{model_name}_{protocol_alias}"
    wall_t0 = time.perf_counter()

    parallel_jobs = [
        delayed(_run_single_fold_task)(
            task_idx=i,
            subject_id=task["subject_id"],
            fold_id=task["fold"],
            test_idx=task["test_idx"],
            model_name=model_name,
            X_tr=X_feat[task["train_idx"]],
            y_tr=y[task["train_idx"]],
            X_va=X_feat[task["val_idx"]],
            y_va=y[task["val_idx"]],
            X_te=X_feat[task["test_idx"]],
            y_te=y[task["test_idx"]],
            seed=seed,
        )
        for i, task in enumerate(fold_tasks)
    ]

    if n_jobs == 1:
        fold_outputs = [
            _run_single_fold_task(
                task_idx=i,
                subject_id=task["subject_id"],
                fold_id=task["fold"],
                test_idx=task["test_idx"],
                model_name=model_name,
                X_tr=X_feat[task["train_idx"]],
                y_tr=y[task["train_idx"]],
                X_va=X_feat[task["val_idx"]],
                y_va=y[task["val_idx"]],
                X_te=X_feat[task["test_idx"]],
                y_te=y[task["test_idx"]],
                seed=seed,
            )
            for i, task in enumerate(fold_tasks)
        ]
    else:
        fold_outputs = Parallel(n_jobs=n_jobs)(parallel_jobs)

    fold_outputs.sort(key=lambda r: r["task_idx"])
    wall_time_sec = time.perf_counter() - wall_t0

    n_samples = len(meta_df)
    oof_fold = np.full(n_samples, -1, dtype=np.int64)
    oof_prob = np.full(n_samples, np.nan, dtype=np.float64)
    oof_pred = np.full(n_samples, -1, dtype=np.int64)
    oof_thr = np.full(n_samples, np.nan, dtype=np.float64)

    per_fold_records: List[Dict[str, Any]] = []
    for task, out in zip(fold_tasks, fold_outputs):
        te_idx = out["test_idx"]
        oof_fold[te_idx] = out["fold"]
        oof_prob[te_idx] = out["y_prob"]
        oof_pred[te_idx] = out["y_pred"]
        oof_thr[te_idx] = out["threshold"]

        m = out["metrics"]
        per_fold_records.append(
            {
                "subject_id": out["subject_id"],
                "fold": int(out["fold"]),
                "protocol": protocol_name,
                "balanced_accuracy": float(m["balanced_accuracy"]),
                "roc_auc": float(m["roc_auc"]),
                "pr_auc": float(m["pr_auc"]),
                "f1_target": float(m["f1_target"]),
                "sensitivity": float(m["sensitivity"]),
                "specificity": float(m["specificity"]),
                "threshold": float(out["threshold"]),
                "val_roc_auc": float(out["val_roc_auc"]),
                "best_params": out["best_params"],
                "train_time_sec": float(out["train_time_sec"]),
                "n_train": int(len(task["train_idx"])),
                "n_val": int(len(task["val_idx"])),
                "n_test": int(len(te_idx)),
                "tp": int(m["tp"]),
                "tn": int(m["tn"]),
                "fp": int(m["fp"]),
                "fn": int(m["fn"]),
            }
        )

    if (oof_fold < 0).any() or not np.isfinite(oof_prob).all():
        raise RuntimeError(f"{resolved_exp}: dự đoán OOF không phủ đủ {n_samples} mẫu.")

    agg = aggregate_metrics(per_fold_records, protocol=protocol_name)

    pred_df = pd.DataFrame(
        {
            "trial_uid": meta_df["trial_uid"].astype(str),
            "subject_id": meta_df["subject_id"].astype(str),
            "protocol": protocol_name,
            "fold": oof_fold,
            "y_true": y.astype(np.int64),
            "y_prob": oof_prob,
            "y_pred": oof_pred,
            "threshold": oof_thr,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    preds_csv_path = save_predictions_csv(pred_df, output_dir / f"predictions_{resolved_exp}.csv")

    config = {
        "exp_name": resolved_exp,
        "model": model_name,
        "protocol": protocol_alias,
        "protocol_name": protocol_name,
        "feature_set": feature_set,
        "channel_preset": channel_preset,
        "n_features": int(X_feat.shape[1]),
        "n_samples": int(n_samples),
        "n_folds": len(per_fold_records),
        "seed": int(seed),
        "n_jobs": int(n_jobs),
        "wall_time_sec": float(wall_time_sec),
        "sum_fold_train_time_sec": float(sum(r["train_time_sec"] for r in per_fold_records)),
    }
    metrics_json_path = save_metrics_json(
        out_path=output_dir / f"metrics_{resolved_exp}.json",
        config=config,
        trial_uid_sha256=split_sha,
        summary=agg["summary"],
        per_fold=per_fold_records,
        per_subject=agg["per_subject"],
    )

    return {
        "exp_name": resolved_exp,
        "model": model_name,
        "protocol": protocol_alias,
        "feature_set": feature_set,
        "n_features": int(X_feat.shape[1]),
        "n_folds": len(per_fold_records),
        "wall_time_sec": float(wall_time_sec),
        "summary": agg["summary"],
        "metrics_json": str(metrics_json_path),
        "predictions_csv": str(preds_csv_path),
    }


def extract_features_for_ml2(
    data_dir: Path,
    *,
    feature_set: str,
    channel_preset: str = "core_14",
    chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """Nạp ``X_clean.npy`` và trích xuất ma trận đặc trưng ERP ``float64``."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    channel_names, times = load_data_geometry(data_dir)

    extractor = ERPFeatureExtractor(
        channel_names=channel_names,
        times=times,
        channel_preset=channel_preset,
        feature_set=feature_set,
    )
    X_feat = extractor.transform_chunked(X_clean, chunk_size=chunk_size)
    return X_feat, y, meta_df, extractor.get_feature_names()


def format_summary_row(res: Dict[str, Any]) -> Dict[str, Any]:
    """Chuyển kết quả một thực nghiệm thành dòng tóm tắt chuẩn."""
    s = res["summary"]
    return {
        "exp_name": res["exp_name"],
        "model": res["model"],
        "protocol": res["protocol"],
        "feature_set": res["feature_set"],
        "n_features": res["n_features"],
        "n_folds": res["n_folds"],
        "roc_auc_mean": s["roc_auc"]["mean"],
        "roc_auc_std": s["roc_auc"]["std"],
        "pr_auc_mean": s["pr_auc"]["mean"],
        "pr_auc_std": s["pr_auc"]["std"],
        "bal_acc_mean": s["balanced_accuracy"]["mean"],
        "bal_acc_std": s["balanced_accuracy"]["std"],
        "f1_target_mean": s["f1_target"]["mean"],
        "f1_target_std": s["f1_target"]["std"],
        "sensitivity_mean": s["sensitivity"]["mean"],
        "sensitivity_std": s["sensitivity"]["std"],
        "specificity_mean": s["specificity"]["mean"],
        "specificity_std": s["specificity"]["std"],
        "wall_time_sec": res["wall_time_sec"],
    }


def save_comparison_table(
    rows: Sequence[Dict[str, Any]],
    out_dir: Path,
    stem: str,
) -> Tuple[Path, Path]:
    """Lưu bảng tổng hợp CSV và JSON sạch cho nhiều lần chạy ML2."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(list(rows)), f, indent=2, allow_nan=False)
    return csv_path, json_path
