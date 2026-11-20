"""Đánh giá, tối ưu ngưỡng Validation và chuẩn hóa xuất kết quả cho nhánh DL2."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_NAMES: Tuple[str, ...] = (
    "balanced_accuracy",
    "roc_auc",
    "pr_auc",
    "f1_target",
    "sensitivity",
    "specificity",
)

PREDICTION_COLUMNS: List[str] = [
    "trial_uid",
    "subject_id",
    "protocol",
    "fold",
    "y_true",
    "y_prob",
    "y_pred",
    "threshold",
]


def _natural_sort_key(value: str) -> List[Union[int, str]]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
) -> float:
    """Tìm ngưỡng quyết định tối ưu trên tập Validation trong đoạn ``[0.05, 0.95]``.

    Kết hợp lưới cố định 181 điểm trong ``[0.05, 0.95]`` (bước ``0.005``) với
    các phân vị thực nghiệm của ``y_prob`` được lọc/kẹp gọn vào ``[0.05, 0.95]``
    nhằm tránh trôi ngưỡng cực trị khi xác suất lệch. Khi nhiều ngưỡng hòa nhau
    trong sai số ``1e-12``, ưu tiên chọn ứng viên gần ``0.5`` nhất. Khi tập chỉ
    chứa 1 lớp hoặc có giá trị không hữu hạn, hàm trả về ``0.5``.
    """
    if metric not in ("balanced_accuracy", "f1_target"):
        raise ValueError(
            f"metric tối ưu ngưỡng phải là 'balanced_accuracy' hoặc 'f1_target', nhận {metric!r}."
        )

    y_arr = np.asarray(y_true).ravel()
    p_arr = np.asarray(y_prob, dtype=np.float64).ravel()

    if y_arr.shape[0] != p_arr.shape[0]:
        raise ValueError(
            f"Độ dài y_true ({y_arr.shape[0]}) và y_prob ({p_arr.shape[0]}) không khớp."
        )
    if y_arr.size == 0 or not np.isfinite(p_arr).all():
        return 0.5

    unique_classes = np.unique(y_arr)
    if unique_classes.size < 2:
        return 0.5

    fixed_grid = np.linspace(0.05, 0.95, 181, dtype=np.float64)
    quantiles = np.quantile(p_arr, np.linspace(0.01, 0.99, 99))
    quantiles = quantiles[(quantiles >= 0.05) & (quantiles <= 0.95)]
    candidates = np.clip(
        np.unique(np.concatenate([fixed_grid, quantiles])),
        0.05,
        0.95,
    )

    preds = p_arr[None, :] >= candidates[:, None]
    y_pos = (y_arr == 1)[None, :]
    y_neg = (y_arr == 0)[None, :]

    tp = np.sum(preds & y_pos, axis=1, dtype=np.float64)
    fp = np.sum(preds & y_neg, axis=1, dtype=np.float64)
    fn = np.sum((~preds) & y_pos, axis=1, dtype=np.float64)
    tn = np.sum((~preds) & y_neg, axis=1, dtype=np.float64)

    if metric == "balanced_accuracy":
        sens = tp / np.maximum(tp + fn, 1.0)
        spec = tn / np.maximum(tn + fp, 1.0)
        scores = 0.5 * (sens + spec)
    else:
        denom = 2.0 * tp + fp + fn
        scores = np.where(denom > 0.0, (2.0 * tp) / np.maximum(denom, 1.0), 0.0)

    best_score = float(np.max(scores))
    best_indices = np.where(np.isclose(scores, best_score, atol=1e-12, rtol=0.0))[0]
    best_candidates = candidates[best_indices]
    tie_idx = int(np.argmin(np.abs(best_candidates - 0.5)))
    return float(np.clip(best_candidates[tie_idx], 0.05, 0.95))


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Tính đầy đủ 6 chỉ số phân loại nhị phân và ma trận nhầm lẫn."""
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    p_arr = np.asarray(y_prob, dtype=np.float64).ravel()

    if y_arr.shape[0] != p_arr.shape[0]:
        raise ValueError(
            f"Độ dài y_true ({y_arr.shape[0]}) và y_prob ({p_arr.shape[0]}) không khớp."
        )
    if y_arr.size == 0:
        raise ValueError("Không thể tính metrics trên mảng rỗng.")

    thr = float(threshold)
    y_pred = (p_arr >= thr).astype(np.int64)

    pos_mask = y_arr == 1
    neg_mask = y_arr == 0
    pred_pos = y_pred == 1
    pred_neg = y_pred == 0

    tp = int(np.sum(pred_pos & pos_mask))
    fn = int(np.sum(pred_neg & pos_mask))
    fp = int(np.sum(pred_pos & neg_mask))
    tn = int(np.sum(pred_neg & neg_mask))

    n_samples = int(y_arr.size)
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)

    sensitivity = float(tp / n_pos) if n_pos > 0 else float("nan")
    specificity = float(tn / n_neg) if n_neg > 0 else float("nan")

    if n_pos > 0 and n_neg > 0:
        balanced_accuracy = float(0.5 * (sensitivity + specificity))
    elif n_pos > 0:
        balanced_accuracy = sensitivity
    else:
        balanced_accuracy = specificity

    f1_denom = 2 * tp + fp + fn
    f1_target = float((2 * tp) / f1_denom) if f1_denom > 0 else 0.0

    if n_pos > 0 and n_neg > 0 and np.isfinite(p_arr).all():
        roc_auc = float(roc_auc_score(y_arr, p_arr))
        pr_auc = float(average_precision_score(y_arr, p_arr))
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    return {
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_target": f1_target,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "threshold": thr,
        "n_samples": n_samples,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _summarize_metric_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Tổng hợp mean, std, median, min, max cho 6 chỉ số từ danh sách bản ghi."""
    summary: Dict[str, Dict[str, Any]] = {}
    for metric_name in METRIC_NAMES:
        raw_vals = []
        for rec in records:
            val = rec.get(metric_name)
            if val is not None:
                fval = float(val)
                if math.isfinite(fval):
                    raw_vals.append(fval)
        if not raw_vals:
            summary[metric_name] = {
                "mean": float("nan"),
                "std": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "n": 0,
            }
            continue
        arr = np.asarray(raw_vals, dtype=np.float64)
        summary[metric_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1 if arr.size > 1 else 0)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "n": int(arr.size),
        }
    return summary


def aggregate_metrics(
    fold_results: Sequence[Dict[str, Any]],
    protocol: Optional[str] = None,
) -> Dict[str, Any]:
    """Tổng hợp metrics từ các folds.

    Với giao thức ``within_subject``: tính trung bình các folds theo từng
    ``subject_id`` trước (``per_subject``), rồi tính ``Mean ± SD`` trên toàn bộ
    các subjects để phục vụ ghép cặp thống kê chuẩn bài báo.
    """
    if not fold_results:
        raise ValueError("fold_results rỗng, không thể tổng hợp metrics.")

    resolved_protocol = protocol or fold_results[0].get("protocol")
    is_within = resolved_protocol == "within_subject"

    if not is_within and resolved_protocol is None:
        sub_ids = [str(r.get("subject_id", "")) for r in fold_results if r.get("subject_id")]
        if sub_ids and len(sub_ids) > len(set(sub_ids)) and "all" not in set(sub_ids):
            is_within = True
            resolved_protocol = "within_subject"

    per_subject: Dict[str, Dict[str, Any]] = {}

    if is_within:
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for rec in fold_results:
            sub_id = str(rec.get("subject_id", "unknown"))
            grouped.setdefault(sub_id, []).append(rec)

        for sub_id in sorted(grouped.keys(), key=_natural_sort_key):
            sub_folds = grouped[sub_id]
            sub_stats = _summarize_metric_records(sub_folds)
            sub_entry: Dict[str, Any] = {
                "subject_id": sub_id,
                "n_folds": len(sub_folds),
            }
            for metric_name in METRIC_NAMES:
                sub_entry[metric_name] = sub_stats[metric_name]["mean"]
                sub_entry[f"{metric_name}_std"] = sub_stats[metric_name]["std"]
            per_subject[sub_id] = sub_entry

        summary = _summarize_metric_records(list(per_subject.values()))
        n_subjects = len(per_subject)
    else:
        if resolved_protocol == "cross_subject_loso":
            for rec in fold_results:
                sub_id = str(rec.get("subject_id", rec.get("fold_key", "unknown")))
                sub_entry = {
                    "subject_id": sub_id,
                    "fold": rec.get("fold"),
                    **{m: rec.get(m) for m in METRIC_NAMES},
                }
                per_subject[sub_id] = sub_entry
        summary = _summarize_metric_records(fold_results)
        n_subjects = len(per_subject) if per_subject else len(fold_results)

    result: Dict[str, Any] = {
        "protocol": resolved_protocol,
        "aggregation_unit": "subject" if is_within else "fold",
        "n_folds": len(fold_results),
        "n_subjects": n_subjects,
        "summary": summary,
        "per_subject": per_subject,
    }
    for metric_name in METRIC_NAMES:
        result[metric_name] = summary[metric_name]
    return result


def _sanitize_json_value(obj: Any) -> Any:
    """Chuyển NaN/Inf và kiểu numpy thành kiểu Python hợp chuẩn JSON RFC 8259."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json_value(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _sanitize_json_value(obj.tolist())
    return obj


def save_predictions_csv(df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """Kiểm tra schema bắt buộc và lưu `predictions_{exp_name}.csv`."""
    missing = [col for col in PREDICTION_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"DataFrame predictions thiếu các cột bắt buộc: {missing}")

    ordered = df[PREDICTION_COLUMNS].copy()
    if ordered.isna().any().any():
        nan_cols = ordered.columns[ordered.isna().any()].tolist()
        raise ValueError(f"DataFrame predictions chứa NaN ở các cột: {nan_cols}")

    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    ordered.to_csv(target, index=False)
    return target


def save_metrics_json(
    out_path: Union[str, Path],
    config: Dict[str, Any],
    trial_uid_sha256: str,
    summary: Dict[str, Any],
    per_fold: Sequence[Dict[str, Any]],
    per_subject: Optional[Dict[str, Any]] = None,
) -> Path:
    """Lưu `metrics_{exp_name}.json` theo đúng cấu trúc Data Contract."""
    payload = {
        "config": config,
        "trial_uid_sha256": str(trial_uid_sha256),
        "summary": summary,
        "per_subject": per_subject or {},
        "per_fold": list(per_fold),
    }
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(payload), f, indent=2, allow_nan=False)
    return target
