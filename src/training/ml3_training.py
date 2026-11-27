"""ML3 cross-subject (CS5 & LOSO) XGBoost training, checkpointing, metrics, and plots."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    METRIC_NAMES,
    _sanitize_json_value,
    compute_binary_metrics,
    find_optimal_threshold,
)
from src.features.ml3_feature_store import (
    CS5_MAIN_EXPERIMENTS,
    FEATURE_SCENARIO_SPECS,
    FeatureScenarioSpec,
    LOSO_MAIN_EXPERIMENTS,
    ML3FeatureDataset,
    ScenarioMatrix,
    ensure_drive_path_ready,
    inspect_feature_store_metadata,
    load_ml3_feature_dataset,
)
from src.models.Ml3 import ML3XGBoostConfig, ML3XGBoostModel

ProtocolName = Literal["cs5", "loso"]

SPLIT_FILE_BY_PROTOCOL: Dict[ProtocolName, Tuple[str, str]] = {
    "cs5": ("cross_5fold.json", "cross_subject_5fold"),
    "loso": ("loso.json", "cross_subject_loso"),
}

CANONICAL_EXPERIMENT_ORDER: Tuple[str, ...] = (
    "temporal",
    "wavelet_w5",
    "wavelet_w8",
    "combined_w5",
    "combined_w8",
    "wavelet_w5_gamma",
    "wavelet_w8_gamma",
    "combined_w5_gamma",
    "combined_w8_gamma",
)

PRIMARY_METRICS: Tuple[str, ...] = ("pr_auc", "roc_auc")
SUPPORTING_METRICS: Tuple[str, ...] = (
    "balanced_accuracy",
    "f1_target",
    "sensitivity",
    "specificity",
)
ALL_REPORTED_METRICS: Tuple[str, ...] = PRIMARY_METRICS + SUPPORTING_METRICS


def _natural_sort_key(value: str) -> list[int | str]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(value))]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        _sanitize_json_value(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.stem}.tmp.json")
    tmp_path.write_text(
        json.dumps(_sanitize_json_value(dict(payload)), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    tmp_path.replace(path)
    return path


def normalize_protocol_name(protocol: str) -> ProtocolName:
    cleaned = str(protocol).strip().lower()
    alias_map: Dict[str, ProtocolName] = {
        "cs5": "cs5",
        "cross_5fold": "cs5",
        "cross_subject_5fold": "cs5",
        "loso": "loso",
        "cross_subject_loso": "loso",
    }
    if cleaned in {"within", "within_subject", "within_5fold", "ws"}:
        raise ValueError(
            "Within-subject protocol is intentionally not supported in ML3 v1; "
            "use 'cs5' or 'loso'."
        )
    if cleaned not in alias_map:
        raise ValueError(f"Unsupported ML3 protocol {protocol!r}; expected 'cs5' or 'loso'")
    return alias_map[cleaned]


def default_experiments_for_protocol(protocol: ProtocolName) -> Tuple[str, ...]:
    norm = normalize_protocol_name(protocol)
    if norm == "cs5":
        return CS5_MAIN_EXPERIMENTS
    return LOSO_MAIN_EXPERIMENTS


def validate_experiment_list(
    protocol: ProtocolName,
    experiments: Optional[Sequence[str]],
    *,
    allow_gamma_loso: bool = False,
) -> Tuple[str, ...]:
    norm = normalize_protocol_name(protocol)
    chosen = (
        tuple(dict.fromkeys(str(e).strip() for e in experiments))
        if experiments
        else default_experiments_for_protocol(norm)
    )
    if not chosen:
        raise ValueError("At least one ML3 experiment must be specified")

    unknown = [exp for exp in chosen if exp not in FEATURE_SCENARIO_SPECS]
    if unknown:
        raise ValueError(
            f"Unknown ML3 experiment(s): {unknown}. "
            f"Valid options: {list(CANONICAL_EXPERIMENT_ORDER)}"
        )

    if norm == "loso" and not allow_gamma_loso:
        gamma_exps = [exp for exp in chosen if FEATURE_SCENARIO_SPECS[exp].uses_gamma]
        if gamma_exps:
            raise ValueError(
                f"Gamma experiments {gamma_exps} do not run on LOSO by default. "
                "Run gamma verification on CS5 first (or pass --allow-gamma-loso explicitly)."
            )
    return chosen


def required_feature_groups_for_experiments(experiments: Sequence[str]) -> Tuple[str, ...]:
    groups: list[str] = []
    for exp in experiments:
        spec = FEATURE_SCENARIO_SPECS[exp]
        if spec.uses_temporal and "temporal" not in groups:
            groups.append("temporal")
        if spec.wavelet_windows is not None and "wavelet" not in groups:
            groups.append("wavelet")
    return tuple(groups)


@dataclass(frozen=True)
class SubjectFoldSplit:
    """Subject-level train/val/test partition for one CS5 or LOSO fold."""

    protocol: ProtocolName
    fold: int
    train_subjects: Tuple[str, ...]
    val_subjects: Tuple[str, ...]
    test_subjects: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError(f"fold index must be non-negative, got {self.fold}")
        if not self.train_subjects or not self.val_subjects or not self.test_subjects:
            raise ValueError(
                f"Fold {self.fold} must have non-empty train_subjects, val_subjects, and test_subjects"
            )
        s_train = set(self.train_subjects)
        s_val = set(self.val_subjects)
        s_test = set(self.test_subjects)
        if len(s_train) != len(self.train_subjects) or len(s_val) != len(self.val_subjects) or len(s_test) != len(self.test_subjects):
            raise ValueError(f"Fold {self.fold} contains duplicate subject IDs within a split")

        overlap_tv = sorted(s_train & s_val)
        overlap_tt = sorted(s_train & s_test)
        overlap_vt = sorted(s_val & s_test)
        if overlap_tv or overlap_tt or overlap_vt:
            raise ValueError(
                f"Subject leakage in {self.protocol} fold {self.fold}: "
                f"train&val={overlap_tv}, train&test={overlap_tt}, val&test={overlap_vt}"
            )

    def split_signature(self) -> Dict[str, Any]:
        return {
            "protocol": self.protocol,
            "fold": int(self.fold),
            "train_subjects": list(self.train_subjects),
            "val_subjects": list(self.val_subjects),
            "test_subjects": list(self.test_subjects),
        }


def load_ml3_splits(
    splits_dir: Path | str,
    protocol: ProtocolName | str,
) -> Tuple[Tuple[SubjectFoldSplit, ...], Dict[str, Any]]:
    """Load CS5 or LOSO subject folds directly from ``splits/*.json`` (no metadata.csv needed)."""
    norm = normalize_protocol_name(protocol)
    filename, json_key = SPLIT_FILE_BY_PROTOCOL[norm]
    split_path = Path(splits_dir) / filename
    if not split_path.is_file():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    raw = json.loads(split_path.read_text(encoding="utf-8"))
    if json_key not in raw or not isinstance(raw[json_key], list) or not raw[json_key]:
        raise ValueError(f"Split file {split_path} is missing non-empty fold list '{json_key}'")

    folds: list[SubjectFoldSplit] = []
    seen_folds: set[int] = set()
    for entry in raw[json_key]:
        fold_idx = int(entry["fold"])
        if fold_idx in seen_folds:
            raise ValueError(f"Duplicate fold index {fold_idx} in {split_path}")
        seen_folds.add(fold_idx)
        folds.append(
            SubjectFoldSplit(
                protocol=norm,
                fold=fold_idx,
                train_subjects=tuple(str(s) for s in entry["train_subjects"]),
                val_subjects=tuple(str(s) for s in entry["val_subjects"]),
                test_subjects=tuple(str(s) for s in entry["test_subjects"]),
            )
        )

    folds_sorted = tuple(sorted(folds, key=lambda f: f.fold))
    meta = {
        "split_path": str(split_path),
        "schema_version": raw.get("schema_version"),
        "seed": raw.get("seed"),
        "n_subjects": raw.get("n_subjects"),
        "subjects": raw.get("subjects"),
        "trial_uid_sha256": raw.get("trial_uid_sha256"),
        "n_folds": len(folds_sorted),
    }
    return folds_sorted, meta


def resolve_subject_fold_indices(
    split: SubjectFoldSplit,
    subject_ids: np.ndarray,
    trial_uids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map ``train_subjects / val_subjects / test_subjects`` to verified disjoint row indices."""
    subs = np.asarray(subject_ids, dtype=str)
    uids = np.asarray(trial_uids, dtype=str)
    if len(subs) != len(uids):
        raise ValueError("subject_ids and trial_uids must have equal length")

    available = set(np.unique(subs).tolist())
    required = set(split.train_subjects) | set(split.val_subjects) | set(split.test_subjects)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"Split {split.protocol} fold {split.fold} references subjects absent from features: {missing}"
        )

    train_idx = np.flatnonzero(np.isin(subs, split.train_subjects))
    val_idx = np.flatnonzero(np.isin(subs, split.val_subjects))
    test_idx = np.flatnonzero(np.isin(subs, split.test_subjects))

    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(f"Empty partition encountered in {split.protocol} fold {split.fold}")

    if (
        np.intersect1d(train_idx, val_idx).size
        or np.intersect1d(train_idx, test_idx).size
        or np.intersect1d(val_idx, test_idx).size
    ):
        raise ValueError(f"Trial index leakage detected in {split.protocol} fold {split.fold}")

    u_tr = set(uids[train_idx].tolist())
    u_va = set(uids[val_idx].tolist())
    u_te = set(uids[test_idx].tolist())
    if (u_tr & u_va) or (u_tr & u_te) or (u_va & u_te):
        raise ValueError(f"trial_uid leakage detected in {split.protocol} fold {split.fold}")

    return train_idx, val_idx, test_idx


def compute_fold_config_hash(
    *,
    protocol: ProtocolName,
    scenario: FeatureScenarioSpec,
    model_config: ML3XGBoostConfig,
    split: SubjectFoldSplit,
    threshold_metric: str = "balanced_accuracy",
) -> str:
    payload = {
        "schema_version": 1,
        "protocol": protocol,
        "feature_policy": scenario.policy_dict(),
        "model_parameters": model_config.semantic_dict(),
        "split": split.split_signature(),
        "threshold_metric": threshold_metric,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _format_metric_with_confusion(raw_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    data = dict(raw_metrics)
    data["confusion_matrix"] = {
        "tp": int(data["tp"]),
        "tn": int(data["tn"]),
        "fp": int(data["fp"]),
        "fn": int(data["fn"]),
    }
    return data


def _compute_pooled_metrics_with_fixed_preds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute trial-pooled metrics across folds where each fold used its own val threshold."""
    base = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    y_arr = np.asarray(y_true, dtype=np.int64).ravel()
    pred_arr = np.asarray(y_pred, dtype=np.int64).ravel()

    pos_mask = y_arr == 1
    neg_mask = y_arr == 0
    pred_pos = pred_arr == 1
    pred_neg = pred_arr == 0

    tp = int(np.sum(pred_pos & pos_mask))
    fn = int(np.sum(pred_neg & pos_mask))
    fp = int(np.sum(pred_pos & neg_mask))
    tn = int(np.sum(pred_neg & neg_mask))
    n_pos = int(tp + fn)
    n_neg = int(tn + fp)

    sensitivity = float(tp / n_pos) if n_pos > 0 else float("nan")
    specificity = float(tn / n_neg) if n_neg > 0 else float("nan")
    if n_pos > 0 and n_neg > 0:
        bal_acc = float(0.5 * (sensitivity + specificity))
    elif n_pos > 0:
        bal_acc = sensitivity
    else:
        bal_acc = specificity
    f1_denom = 2 * tp + fp + fn
    f1_target = float((2 * tp) / f1_denom) if f1_denom > 0 else 0.0

    return {
        "pr_auc": base["pr_auc"],
        "roc_auc": base["roc_auc"],
        "balanced_accuracy": bal_acc,
        "f1_target": f1_target,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "n_samples": int(y_arr.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def summarize_scalar_values(values: Sequence[float | int | None]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1 if arr.size > 1 else 0)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def summarize_metric_records(records: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        metric: summarize_scalar_values([rec.get(metric) for rec in records])
        for metric in ALL_REPORTED_METRICS
    }


def _per_subject_metrics(
    subject_list: Sequence[str],
    subject_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, Dict[str, Any]]:
    subs = np.asarray(subject_ids, dtype=str)
    result: Dict[str, Dict[str, Any]] = {}
    for sub in sorted(subject_list, key=_natural_sort_key):
        mask = subs == sub
        if not np.any(mask):
            continue
        m = _format_metric_with_confusion(
            compute_binary_metrics(y_true[mask], y_prob[mask], threshold=threshold)
        )
        m["subject_id"] = sub
        result[sub] = m
    return result


def _validate_and_load_fold_checkpoint(
    fold_dir: Path,
    *,
    expected_hash: str,
    resume: bool,
    overwrite: bool,
) -> Optional[Tuple[Dict[str, Any], Dict[str, np.ndarray]]]:
    if not fold_dir.exists():
        return None

    manifest_path = fold_dir / "manifest.json"
    metrics_path = fold_dir / "metrics.json"
    preds_path = fold_dir / "predictions.npz"
    model_path = fold_dir / "model.json"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        saved_hash = str(manifest.get("config_hash", ""))
        if saved_hash != expected_hash:
            if overwrite and not resume:
                return None
            raise ValueError(
                f"Checkpoint không tương thích tại '{fold_dir}': config_hash không khớp "
                f"(saved={saved_hash[:12]}..., expected={expected_hash[:12]}...). "
                "Model parameters, feature policy hoặc split đã thay đổi."
            )

        if overwrite and not resume:
            return None

        if not resume:
            raise ValueError(
                f"Checkpoint đã tồn tại tại '{fold_dir}'. Dùng --resume để tái sử dụng "
                "hoặc --overwrite để ghi đè."
            )

        missing = [
            p.name
            for p in (model_path, metrics_path, preds_path)
            if not p.is_file()
        ]
        if missing:
            raise ValueError(
                f"Checkpoint tại '{fold_dir}' bị thiếu file {missing}; "
                "hãy xóa thư mục fold này hoặc chạy lại với --overwrite."
            )

        fold_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with np.load(preds_path, allow_pickle=False) as payload:
            preds = {k: np.asarray(payload[k]) for k in payload.files}
        return fold_metrics, preds

    # Incomplete fold directory without manifest.json (e.g., interrupted Colab session):
    for partial in fold_dir.glob("*"):
        if partial.is_file():
            partial.unlink()
    return None


def train_or_load_fold(
    *,
    scenario_matrix: ScenarioMatrix,
    split: SubjectFoldSplit,
    model_config: ML3XGBoostConfig,
    fold_dir: Path,
    threshold_metric: str = "balanced_accuracy",
    resume: bool = False,
    overwrite: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], str]:
    """Train a single CS5 or LOSO fold or reuse its verified checkpoint."""
    expected_hash = compute_fold_config_hash(
        protocol=split.protocol,
        scenario=scenario_matrix.scenario,
        model_config=model_config,
        split=split,
        threshold_metric=threshold_metric,
    )

    cached = _validate_and_load_fold_checkpoint(
        fold_dir,
        expected_hash=expected_hash,
        resume=resume,
        overwrite=overwrite,
    )
    if cached is not None:
        fold_metrics, preds = cached
        return fold_metrics, preds, "reused"

    train_idx, val_idx, test_idx = resolve_subject_fold_indices(
        split,
        scenario_matrix.subject_ids,
        scenario_matrix.trial_uids,
    )

    X_train = scenario_matrix.values[train_idx]
    y_train = scenario_matrix.targets[train_idx]
    X_val = scenario_matrix.values[val_idx]
    y_val = scenario_matrix.targets[val_idx]
    X_test = scenario_matrix.values[test_idx]
    y_test = scenario_matrix.targets[test_idx]

    model = ML3XGBoostModel(config=model_config)
    t_start = time.perf_counter()
    model.fit_fold(X_train, y_train, X_val, y_val)
    train_duration_sec = float(time.perf_counter() - t_start)

    val_prob = model.predict_proba_positive(X_val)
    test_prob = model.predict_proba_positive(X_test)

    # Threshold is strictly optimized on validation only; never on test.
    threshold = find_optimal_threshold(y_val, val_prob, metric=threshold_metric)

    val_pred = (val_prob >= threshold).astype(np.int64)
    test_pred = (test_prob >= threshold).astype(np.int64)

    val_subs = scenario_matrix.subject_ids[val_idx]
    test_subs = scenario_matrix.subject_ids[test_idx]

    val_pooled = _format_metric_with_confusion(
        compute_binary_metrics(y_val, val_prob, threshold=threshold)
    )
    val_per_subject = _per_subject_metrics(
        split.val_subjects, val_subs, y_val, val_prob, threshold=threshold
    )
    val_subject_macro = summarize_metric_records(list(val_per_subject.values()))

    test_pooled = _format_metric_with_confusion(
        compute_binary_metrics(y_test, test_prob, threshold=threshold)
    )
    test_per_subject = _per_subject_metrics(
        split.test_subjects, test_subs, y_test, test_prob, threshold=threshold
    )
    test_subject_macro = summarize_metric_records(list(test_per_subject.values()))

    assert model.best_iteration_ is not None
    assert model.scale_pos_weight_ is not None

    fold_metrics: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": split.protocol,
        "experiment": scenario_matrix.scenario.name,
        "fold": int(split.fold),
        "config_hash": expected_hash,
        "n_features": int(scenario_matrix.n_features),
        "train_subjects": list(split.train_subjects),
        "val_subjects": list(split.val_subjects),
        "test_subjects": list(split.test_subjects),
        "n_train": int(len(y_train)),
        "n_train_frequent": int(np.sum(y_train == 0)),
        "n_train_target": int(np.sum(y_train == 1)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "scale_pos_weight": float(model.scale_pos_weight_),
        "best_iteration": int(model.best_iteration_),
        "n_trees_used": int(model.best_iteration_ + 1),
        "best_val_aucpr": model.best_val_score_,
        "threshold_metric": threshold_metric,
        "threshold": float(threshold),
        "train_duration_sec": train_duration_sec,
        "validation_pooled": val_pooled,
        "validation_per_subject": val_per_subject,
        "validation_subject_macro": val_subject_macro,
        "test_pooled": test_pooled,
        "test_per_subject": test_per_subject,
        "test_subject_macro": test_subject_macro,
    }

    predictions_payload: Dict[str, np.ndarray] = {
        "subject_id": np.asarray(test_subs, dtype=str),
        "fold": np.full(len(test_idx), int(split.fold), dtype=np.int64),
        "trial_idx": np.asarray(test_idx, dtype=np.int64),
        "trial_uid": np.asarray(scenario_matrix.trial_uids[test_idx], dtype=str),
        "y_true": np.asarray(y_test, dtype=np.int64),
        "y_prob": np.asarray(test_prob, dtype=np.float64),
        "y_pred": np.asarray(test_pred, dtype=np.int64),
        "threshold": np.full(len(test_idx), float(threshold), dtype=np.float64),
        "val_subject_id": np.asarray(val_subs, dtype=str),
        "val_trial_idx": np.asarray(val_idx, dtype=np.int64),
        "val_trial_uid": np.asarray(scenario_matrix.trial_uids[val_idx], dtype=str),
        "val_y_true": np.asarray(y_val, dtype=np.int64),
        "val_y_prob": np.asarray(val_prob, dtype=np.float64),
        "val_y_pred": np.asarray(val_pred, dtype=np.int64),
    }

    fold_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(fold_dir / "model.json")

    tmp_preds = fold_dir / "predictions.tmp.npz"
    np.savez_compressed(tmp_preds, **predictions_payload)
    tmp_preds.replace(fold_dir / "predictions.npz")

    _atomic_write_json(fold_dir / "metrics.json", fold_metrics)

    manifest_payload: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": split.protocol,
        "experiment": scenario_matrix.scenario.name,
        "fold": int(split.fold),
        "config_hash": expected_hash,
        "feature_policy": scenario_matrix.scenario.policy_dict(),
        "model_config": model_config.report_dict(),
        "split": split.split_signature(),
        "threshold_metric": threshold_metric,
        "n_features": int(scenario_matrix.n_features),
        "best_iteration": int(model.best_iteration_),
        "train_duration_sec": train_duration_sec,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": ["model.json", "metrics.json", "predictions.npz"],
    }
    _atomic_write_json(fold_dir / "manifest.json", manifest_payload)

    del model
    gc.collect()
    return fold_metrics, predictions_payload, "trained"


def aggregate_experiment_results(
    *,
    protocol: ProtocolName,
    experiment: str,
    exp_dir: Path,
    model_config: ML3XGBoostConfig,
    fold_metrics_list: Sequence[Mapping[str, Any]],
    fold_preds_list: Sequence[Mapping[str, np.ndarray]],
) -> Dict[str, Any]:
    """Aggregate both trial-level pooled metrics and subject-macro metrics across folds."""
    if not fold_metrics_list:
        raise ValueError(f"No fold results available to aggregate for {experiment}")

    ordered_pairs = sorted(
        zip(fold_metrics_list, fold_preds_list),
        key=lambda pair: int(pair[0]["fold"]),
    )
    folds_sorted = [dict(p[0]) for p in ordered_pairs]
    preds_sorted = [p[1] for p in ordered_pairs]

    # 1. Pooled out-of-fold test predictions & validation predictions
    all_test_y = np.concatenate([p["y_true"] for p in preds_sorted], axis=0)
    all_test_prob = np.concatenate([p["y_prob"] for p in preds_sorted], axis=0)
    all_test_pred = np.concatenate([p["y_pred"] for p in preds_sorted], axis=0)
    overall_test_pooled = _compute_pooled_metrics_with_fixed_preds(
        all_test_y, all_test_prob, all_test_pred
    )

    all_val_y = np.concatenate([p["val_y_true"] for p in preds_sorted], axis=0)
    all_val_prob = np.concatenate([p["val_y_prob"] for p in preds_sorted], axis=0)
    all_val_pred = np.concatenate([p["val_y_pred"] for p in preds_sorted], axis=0)
    overall_val_pooled = _compute_pooled_metrics_with_fixed_preds(
        all_val_y, all_val_prob, all_val_pred
    )

    # Fold-level pooled summary (Mean ± SD & median across folds)
    test_fold_pooled_summary = summarize_metric_records(
        [f["test_pooled"] for f in folds_sorted]
    )
    val_fold_pooled_summary = summarize_metric_records(
        [f["validation_pooled"] for f in folds_sorted]
    )

    # 2. Subject-macro aggregation (equal weight per subject; primary cross-subject metric)
    subject_buckets: Dict[str, list[Dict[str, Any]]] = {}
    for f_rec in folds_sorted:
        fold_idx = int(f_rec["fold"])
        for sub_id, sub_m in f_rec["test_per_subject"].items():
            entry = dict(sub_m)
            entry["fold"] = fold_idx
            subject_buckets.setdefault(sub_id, []).append(entry)

    per_subject_test: Dict[str, Dict[str, Any]] = {}
    for sub_id in sorted(subject_buckets, key=_natural_sort_key):
        entries = subject_buckets[sub_id]
        if len(entries) == 1:
            per_subject_test[sub_id] = entries[0]
        else:
            agg_sub = summarize_metric_records(entries)
            per_subject_test[sub_id] = {
                "subject_id": sub_id,
                "fold": [e["fold"] for e in entries],
                **{m: agg_sub[m]["mean"] for m in ALL_REPORTED_METRICS},
            }

    test_subject_macro_summary = summarize_metric_records(list(per_subject_test.values()))
    fold_subject_macro_means = [
        {m: f["test_subject_macro"][m]["mean"] for m in ALL_REPORTED_METRICS}
        for f in folds_sorted
    ]
    test_fold_subject_macro_summary = summarize_metric_records(fold_subject_macro_means)

    val_fold_subject_macro_means = [
        {m: f["validation_subject_macro"][m]["mean"] for m in ALL_REPORTED_METRICS}
        for f in folds_sorted
    ]
    val_fold_subject_macro_summary = summarize_metric_records(val_fold_subject_macro_means)

    best_iter_stats = summarize_scalar_values([f["best_iteration"] for f in folds_sorted])
    duration_values = [float(f["train_duration_sec"]) for f in folds_sorted]
    duration_stats = summarize_scalar_values(duration_values)
    duration_stats["total_sec"] = float(np.sum(duration_values))

    n_features = int(folds_sorted[0]["n_features"])
    scenario_spec = FEATURE_SCENARIO_SPECS[experiment]

    summary_payload: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol,
        "experiment": experiment,
        "feature_policy": scenario_spec.policy_dict(),
        "model_config": model_config.report_dict(),
        "n_features": n_features,
        "n_folds": len(folds_sorted),
        "n_test_subjects": len(per_subject_test),
        "best_iteration": best_iter_stats,
        "train_duration_sec": duration_stats,
        "test": {
            "subject_macro": test_subject_macro_summary,
            "fold_subject_macro": test_fold_subject_macro_summary,
            "fold_pooled": test_fold_pooled_summary,
            "overall_trial_pooled": overall_test_pooled,
        },
        "validation": {
            "fold_subject_macro": val_fold_subject_macro_summary,
            "fold_pooled": val_fold_pooled_summary,
            "overall_trial_pooled": overall_val_pooled,
        },
        "per_subject": per_subject_test,
        "per_fold": folds_sorted,
    }
    _atomic_write_json(exp_dir / "experiment_summary.json", summary_payload)

    # Export convenient CSV tables for per-fold and per-subject inspection
    fold_rows = []
    for f in folds_sorted:
        tp_m = f["test_pooled"]
        tsm = f["test_subject_macro"]
        vp_m = f["validation_pooled"]
        fold_rows.append(
            {
                "protocol": protocol,
                "experiment": experiment,
                "fold": int(f["fold"]),
                "test_subjects": ",".join(f["test_subjects"]),
                "n_features": int(f["n_features"]),
                "scale_pos_weight": float(f["scale_pos_weight"]),
                "best_iteration": int(f["best_iteration"]),
                "threshold": float(f["threshold"]),
                "train_duration_sec": float(f["train_duration_sec"]),
                "val_pr_auc": vp_m["pr_auc"],
                "val_roc_auc": vp_m["roc_auc"],
                "val_balanced_accuracy": vp_m["balanced_accuracy"],
                "test_pooled_pr_auc": tp_m["pr_auc"],
                "test_pooled_roc_auc": tp_m["roc_auc"],
                "test_pooled_balanced_accuracy": tp_m["balanced_accuracy"],
                "test_pooled_f1_target": tp_m["f1_target"],
                "test_pooled_sensitivity": tp_m["sensitivity"],
                "test_pooled_specificity": tp_m["specificity"],
                "test_subject_macro_pr_auc": tsm["pr_auc"]["mean"],
                "test_subject_macro_roc_auc": tsm["roc_auc"]["mean"],
                "test_subject_macro_balanced_accuracy": tsm["balanced_accuracy"]["mean"],
                "test_subject_macro_f1_target": tsm["f1_target"]["mean"],
                "tp": int(tp_m["tp"]),
                "tn": int(tp_m["tn"]),
                "fp": int(tp_m["fp"]),
                "fn": int(tp_m["fn"]),
            }
        )
    pd.DataFrame(fold_rows).to_csv(exp_dir / "per_fold_metrics.csv", index=False)

    subject_rows = []
    for sub_id in sorted(per_subject_test, key=_natural_sort_key):
        s_rec = per_subject_test[sub_id]
        subject_rows.append(
            {
                "protocol": protocol,
                "experiment": experiment,
                "subject_id": sub_id,
                "fold": s_rec.get("fold"),
                "pr_auc": s_rec.get("pr_auc"),
                "roc_auc": s_rec.get("roc_auc"),
                "balanced_accuracy": s_rec.get("balanced_accuracy"),
                "f1_target": s_rec.get("f1_target"),
                "sensitivity": s_rec.get("sensitivity"),
                "specificity": s_rec.get("specificity"),
                "n_samples": s_rec.get("n_samples"),
                "n_pos": s_rec.get("n_pos"),
                "n_neg": s_rec.get("n_neg"),
                "tp": s_rec.get("tp"),
                "tn": s_rec.get("tn"),
                "fp": s_rec.get("fp"),
                "fn": s_rec.get("fn"),
            }
        )
    pd.DataFrame(subject_rows).to_csv(exp_dir / "per_subject_metrics.csv", index=False)

    oof_tmp = exp_dir / "oof_predictions.tmp.npz"
    np.savez_compressed(
        oof_tmp,
        subject_id=np.concatenate([p["subject_id"] for p in preds_sorted], axis=0),
        fold=np.concatenate([p["fold"] for p in preds_sorted], axis=0),
        trial_idx=np.concatenate([p["trial_idx"] for p in preds_sorted], axis=0),
        trial_uid=np.concatenate([p["trial_uid"] for p in preds_sorted], axis=0),
        y_true=all_test_y,
        y_prob=all_test_prob,
        y_pred=all_test_pred,
        threshold=np.concatenate([p["threshold"] for p in preds_sorted], axis=0),
    )
    oof_tmp.replace(exp_dir / "oof_predictions.npz")

    return summary_payload


def _discover_completed_experiment_summaries(protocol_dir: Path) -> Dict[str, Dict[str, Any]]:
    summaries: Dict[str, Dict[str, Any]] = {}
    if not protocol_dir.is_dir():
        return summaries

    ordered_names = list(CANONICAL_EXPERIMENT_ORDER) + sorted(
        p.name
        for p in protocol_dir.iterdir()
        if p.is_dir() and p.name not in CANONICAL_EXPERIMENT_ORDER
    )
    for name in ordered_names:
        summary_file = protocol_dir / name / "experiment_summary.json"
        if summary_file.is_file():
            summaries[name] = json.loads(summary_file.read_text(encoding="utf-8"))
    return summaries


def _compute_paired_subject_deltas(
    summaries: Mapping[str, Mapping[str, Any]],
    exp_a: str,
    exp_b: str,
) -> Optional[Dict[str, Any]]:
    """Compute paired ``exp_a - exp_b`` differences across shared test subjects and folds."""
    if exp_a not in summaries or exp_b not in summaries:
        return None
    subs_a = summaries[exp_a].get("per_subject", {})
    subs_b = summaries[exp_b].get("per_subject", {})
    shared_subs = sorted(set(subs_a) & set(subs_b), key=_natural_sort_key)
    if not shared_subs:
        return None

    subject_deltas: Dict[str, Dict[str, Any]] = {}
    per_sub_vectors: Dict[str, List[float]] = {}
    for metric in ALL_REPORTED_METRICS:
        diffs: list[float] = []
        for sub in shared_subs:
            va = subs_a[sub].get(metric)
            vb = subs_b[sub].get(metric)
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):
                diffs.append(float(va) - float(vb))
        subject_deltas[metric] = summarize_scalar_values(diffs)
        per_sub_vectors[metric] = diffs

    folds_a = {int(f["fold"]): f for f in summaries[exp_a].get("per_fold", [])}
    folds_b = {int(f["fold"]): f for f in summaries[exp_b].get("per_fold", [])}
    shared_folds = sorted(set(folds_a) & set(folds_b))
    fold_deltas: Dict[str, Dict[str, Any]] = {}
    per_fold_vectors: Dict[str, List[float]] = {}
    for metric in ALL_REPORTED_METRICS:
        f_diffs: list[float] = []
        for k in shared_folds:
            va = folds_a[k]["test_pooled"].get(metric)
            vb = folds_b[k]["test_pooled"].get(metric)
            if va is not None and vb is not None and math.isfinite(float(va)) and math.isfinite(float(vb)):
                f_diffs.append(float(va) - float(vb))
        fold_deltas[metric] = summarize_scalar_values(f_diffs)
        per_fold_vectors[metric] = f_diffs

    return {
        "comparison": f"{exp_a} - {exp_b}",
        "lhs": exp_a,
        "rhs": exp_b,
        "n_shared_subjects": len(shared_subs),
        "n_shared_folds": len(shared_folds),
        "subject_paired_delta": subject_deltas,
        "fold_pooled_paired_delta": fold_deltas,
        "_subject_vectors": per_sub_vectors,
        "_fold_vectors": per_fold_vectors,
    }


def write_protocol_comparison_and_plots(
    protocol: ProtocolName,
    protocol_dir: Path,
) -> Dict[str, Any]:
    """Build protocol comparison tables and generate CS5/LOSO plots without auto-picking a winner."""
    summaries = _discover_completed_experiment_summaries(protocol_dir)
    if not summaries:
        return {}

    comparison_rows = []
    for exp_name, summary in summaries.items():
        sm = summary["test"]["subject_macro"]
        fp = summary["test"]["fold_pooled"]
        op = summary["test"]["overall_trial_pooled"]
        bi = summary["best_iteration"]
        td = summary["train_duration_sec"]
        comparison_rows.append(
            {
                "protocol": protocol,
                "experiment": exp_name,
                "n_features": summary["n_features"],
                "n_folds": summary["n_folds"],
                "n_subjects": summary["n_test_subjects"],
                "subject_macro_pr_auc_mean": sm["pr_auc"]["mean"],
                "subject_macro_pr_auc_std": sm["pr_auc"]["std"],
                "subject_macro_pr_auc_median": sm["pr_auc"]["median"],
                "subject_macro_roc_auc_mean": sm["roc_auc"]["mean"],
                "subject_macro_roc_auc_std": sm["roc_auc"]["std"],
                "subject_macro_roc_auc_median": sm["roc_auc"]["median"],
                "subject_macro_bal_acc_mean": sm["balanced_accuracy"]["mean"],
                "subject_macro_bal_acc_std": sm["balanced_accuracy"]["std"],
                "subject_macro_f1_target_mean": sm["f1_target"]["mean"],
                "subject_macro_f1_target_std": sm["f1_target"]["std"],
                "fold_pooled_pr_auc_mean": fp["pr_auc"]["mean"],
                "fold_pooled_pr_auc_std": fp["pr_auc"]["std"],
                "fold_pooled_roc_auc_mean": fp["roc_auc"]["mean"],
                "fold_pooled_roc_auc_std": fp["roc_auc"]["std"],
                "overall_pooled_pr_auc": op["pr_auc"],
                "overall_pooled_roc_auc": op["roc_auc"],
                "overall_pooled_bal_acc": op["balanced_accuracy"],
                "overall_pooled_f1_target": op["f1_target"],
                "best_iteration_mean": bi["mean"],
                "best_iteration_median": bi["median"],
                "train_time_mean_sec": td["mean"],
                "train_time_std_sec": td["std"],
                "train_time_total_sec": td["total_sec"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(protocol_dir / "comparison_table.csv", index=False)

    candidate_pairs = [
        ("wavelet_w8", "wavelet_w5"),
        ("combined_w8", "combined_w5"),
        ("combined_w5", "temporal"),
        ("combined_w8", "temporal"),
        ("combined_w5", "wavelet_w5"),
        ("combined_w8", "wavelet_w8"),
        ("wavelet_w5", "temporal"),
        ("wavelet_w8", "temporal"),
        ("combined_w5_gamma", "combined_w5"),
        ("combined_w8_gamma", "combined_w8"),
    ]
    paired_deltas: Dict[str, Any] = {}
    paired_with_vectors: Dict[str, Any] = {}
    for lhs, rhs in candidate_pairs:
        delta = _compute_paired_subject_deltas(summaries, lhs, rhs)
        if delta is not None:
            key = f"{lhs}_minus_{rhs}"
            paired_with_vectors[key] = delta
            clean_delta = {k: v for k, v in delta.items() if not k.startswith("_")}
            paired_deltas[key] = clean_delta

    generated_plots: List[str] = []
    if protocol == "cs5":
        generated_plots.extend(_generate_cs5_plots(protocol_dir, summaries, paired_with_vectors))
    else:
        generated_plots.extend(_generate_loso_plots(protocol_dir, summaries, paired_with_vectors))

    comparison_payload = {
        "schema_version": 1,
        "protocol": protocol,
        "note": (
            "No automatic winner rule is applied; inspect subject-macro metrics, "
            "trial-pooled metrics, paired deltas, feature counts, and training times."
        ),
        "experiments": list(summaries.keys()),
        "rows": comparison_rows,
        "paired_deltas": paired_deltas,
        "plots": generated_plots,
    }
    _atomic_write_json(protocol_dir / "comparison_summary.json", comparison_payload)
    return comparison_payload


# ── Visualization Helpers (CS5 & LOSO) ───────────────────────────────────────

_EXP_COLORS: Dict[str, str] = {
    "temporal": "#1f77b4",
    "wavelet_w5": "#ff7f0e",
    "wavelet_w8": "#d62728",
    "combined_w5": "#2ca02c",
    "combined_w8": "#9467bd",
    "combined_w5_gamma": "#17becf",
    "combined_w8_gamma": "#e377c2",
    "wavelet_w5_gamma": "#bcbd22",
    "wavelet_w8_gamma": "#8c564b",
}

_EXP_LABELS: Dict[str, str] = {
    "temporal": "Temporal (T)",
    "wavelet_w5": "Wavelet W5",
    "wavelet_w8": "Wavelet W8",
    "combined_w5": "Combined W5 (C5)",
    "combined_w8": "Combined W8 (C8)",
    "combined_w5_gamma": "Combined W5+Gamma",
    "combined_w8_gamma": "Combined W8+Gamma",
    "wavelet_w5_gamma": "Wavelet W5+Gamma",
    "wavelet_w8_gamma": "Wavelet W8+Gamma",
}


def _exp_color(name: str) -> str:
    return _EXP_COLORS.get(name, "#4d4d4d")


def _exp_label(name: str) -> str:
    return _EXP_LABELS.get(name, name)


def _generate_cs5_plots(
    protocol_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    paired_deltas: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    exp_names = list(summaries.keys())
    x = np.arange(len(exp_names))
    colors = [_exp_color(e) for e in exp_names]
    labels = [_exp_label(e) for e in exp_names]
    rng = np.random.default_rng(42)
    saved_files: List[str] = []

    # 1. cs5_metric_comparison.png (Trial-pooled fold Mean ± SD, individual folds, feature count, train time)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Trial-Pooled Metrics, Feature Count & Train Time", fontsize=13, fontweight="bold")

    for ax, metric, title in (
        (axes[0, 0], "pr_auc", "Test Trial-Pooled PR-AUC (Mean ± SD + Fold Dots)"),
        (axes[0, 1], "roc_auc", "Test Trial-Pooled ROC-AUC (Mean ± SD + Fold Dots)"),
    ):
        means = [summaries[e]["test"]["fold_pooled"][metric]["mean"] for e in exp_names]
        stds = [summaries[e]["test"]["fold_pooled"][metric]["std"] for e in exp_names]
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.82, edgecolor="black", linewidth=0.8)
        for idx, e in enumerate(exp_names):
            fold_vals = [f["test_pooled"][metric] for f in summaries[e]["per_fold"]]
            jitter = rng.uniform(-0.12, 0.12, size=len(fold_vals))
            ax.scatter(np.full(len(fold_vals), idx) + jitter, fold_vals, color="black", s=26, zorder=4, alpha=0.85)
            if math.isfinite(means[idx]):
                ax.text(
                    bars[idx].get_x() + bars[idx].get_width() / 2.0,
                    means[idx] + (stds[idx] if math.isfinite(stds[idx]) else 0.0) + 0.01,
                    f"{means[idx]:.3f}±{stds[idx]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylim(0.0, 1.08)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel C: Supporting metrics (Bal Acc, F1 Target, Sensitivity, Specificity)
    ax_sup = axes[1, 0]
    width = 0.18
    sup_metrics = [
        ("balanced_accuracy", "Bal Acc"),
        ("f1_target", "F1 Target"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
    ]
    for m_i, (m_key, m_label) in enumerate(sup_metrics):
        m_means = [summaries[e]["test"]["fold_pooled"][m_key]["mean"] for e in exp_names]
        m_stds = [summaries[e]["test"]["fold_pooled"][m_key]["std"] for e in exp_names]
        offset = (m_i - 1.5) * width
        ax_sup.bar(x + offset, m_means, width=width, yerr=m_stds, capsize=3, label=m_label, alpha=0.85)
    ax_sup.set_xticks(x)
    ax_sup.set_xticklabels(labels, rotation=20, ha="right")
    ax_sup.set_ylim(0.0, 1.05)
    ax_sup.set_title("Supporting Metrics (Fold Pooled Mean ± SD)")
    ax_sup.legend(ncol=2, fontsize=8.5)
    ax_sup.grid(axis="y", linestyle="--", alpha=0.4)

    # Panel D: Feature Count & Train Time (Mean ± SD per fold)
    ax_cost = axes[1, 1]
    feat_counts = [summaries[e]["n_features"] for e in exp_names]
    time_means = [summaries[e]["train_duration_sec"]["mean"] for e in exp_names]
    time_stds = [summaries[e]["train_duration_sec"]["std"] for e in exp_names]

    b_feat = ax_cost.bar(x - 0.18, feat_counts, width=0.34, color="#4c78a8", alpha=0.85, label="Feature Count")
    ax_cost.set_ylabel("Feature Count", color="#4c78a8")
    ax_cost.tick_params(axis="y", labelcolor="#4c78a8")
    for idx, val in enumerate(feat_counts):
        ax_cost.text(b_feat[idx].get_x() + b_feat[idx].get_width() / 2.0, val, f"{val:,}", ha="center", va="bottom", fontsize=8)

    ax_time = ax_cost.twinx()
    ax_time.bar(x + 0.18, time_means, width=0.34, yerr=time_stds, capsize=4, color="#f58518", alpha=0.85, label="Fold Train Time (s)")
    ax_time.set_ylabel("Fold Train Time (s, Mean ± SD)", color="#f58518")
    ax_time.tick_params(axis="y", labelcolor="#f58518")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=20, ha="right")
    ax_cost.set_title("Feature Dimensionality & Training Duration")
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)

    p1 = protocol_dir / "cs5_metric_comparison.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    saved_files.append(p1.name)

    # 2. cs5_subject_macro_comparison.png (Equal-weight subject-macro metrics + per-subject scatter)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Subject-Macro Metrics (Equal Subject Weight)", fontsize=13, fontweight="bold")
    for ax, (metric, title) in zip(
        axes.ravel(),
        (
            ("pr_auc", "Subject-Macro PR-AUC (Mean ± SD + Per-Subject Dots)"),
            ("roc_auc", "Subject-Macro ROC-AUC (Mean ± SD + Per-Subject Dots)"),
            ("balanced_accuracy", "Subject-Macro Balanced Accuracy (Mean ± SD)"),
            ("f1_target", "Subject-Macro F1 Target (Mean ± SD)"),
        ),
    ):
        means = [summaries[e]["test"]["subject_macro"][metric]["mean"] for e in exp_names]
        stds = [summaries[e]["test"]["subject_macro"][metric]["std"] for e in exp_names]
        medians = [summaries[e]["test"]["subject_macro"][metric]["median"] for e in exp_names]
        ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.72, edgecolor="black", linewidth=0.8)
        ax.scatter(x, medians, marker="D", color="gold", edgecolor="black", s=45, zorder=5, label="Median")
        for idx, e in enumerate(exp_names):
            sub_vals = [
                float(s[metric])
                for s in summaries[e]["per_subject"].values()
                if s.get(metric) is not None and math.isfinite(float(s[metric]))
            ]
            jitter = rng.uniform(-0.18, 0.18, size=len(sub_vals))
            ax.scatter(np.full(len(sub_vals), idx) + jitter, sub_vals, color="#222222", s=16, alpha=0.55, zorder=4)
            if math.isfinite(means[idx]):
                ax.text(
                    idx,
                    min(1.03, means[idx] + (stds[idx] if math.isfinite(stds[idx]) else 0.0) + 0.015),
                    f"{means[idx]:.3f}±{stds[idx]:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.09)
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    p2 = protocol_dir / "cs5_subject_macro_comparison.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    saved_files.append(p2.name)

    # 3. cs5_fold_profiles.png (Fold-by-fold trajectories across scenarios)
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Per-Fold Profiles Across Scenarios", fontsize=13, fontweight="bold")
    for ax, field_kind, title in (
        (axes[0, 0], "pr_auc", "Fold Test PR-AUC Profile"),
        (axes[0, 1], "roc_auc", "Fold Test ROC-AUC Profile"),
        (axes[1, 0], "balanced_accuracy", "Fold Test Balanced Accuracy Profile"),
        (axes[1, 1], "best_iteration", "Fold Early-Stopping Best Iteration"),
    ):
        for e in exp_names:
            f_recs = sorted(summaries[e]["per_fold"], key=lambda r: int(r["fold"]))
            f_ids = [int(r["fold"]) for r in f_recs]
            if field_kind == "best_iteration":
                y_vals = [int(r["best_iteration"]) for r in f_recs]
            else:
                y_vals = [float(r["test_pooled"][field_kind]) for r in f_recs]
            ax.plot(f_ids, y_vals, marker="o", linewidth=2.0, color=_exp_color(e), label=_exp_label(e))
        ax.set_xlabel("Fold Index")
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8.5)

    p3 = protocol_dir / "cs5_fold_profiles.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    saved_files.append(p3.name)

    # 4. cs5_window_paired_deltas.png (Paired differences W8-W5, Combined-Temporal, Combined-Wavelet)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    fig.suptitle("ML3 Cross-Subject 5-Fold — Paired Subject-Level Differences (Mean ± SD)", fontsize=13, fontweight="bold")
    delta_keys = [
        k
        for k in (
            "wavelet_w8_minus_wavelet_w5",
            "combined_w8_minus_combined_w5",
            "combined_w5_minus_temporal",
            "combined_w8_minus_temporal",
            "combined_w5_minus_wavelet_w5",
            "combined_w8_minus_wavelet_w8",
            "combined_w5_gamma_minus_combined_w5",
            "combined_w8_gamma_minus_combined_w8",
        )
        if k in paired_deltas
    ]
    for ax, metric, title in (
        (axes[0], "pr_auc", "Paired Δ PR-AUC (Per-Subject)"),
        (axes[1], "roc_auc", "Paired Δ ROC-AUC (Per-Subject)"),
    ):
        ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.2)
        if delta_keys:
            dx = np.arange(len(delta_keys))
            d_labels = [paired_deltas[k]["comparison"] for k in delta_keys]
            d_means = [paired_deltas[k]["subject_paired_delta"][metric]["mean"] for k in delta_keys]
            d_stds = [paired_deltas[k]["subject_paired_delta"][metric]["std"] for k in delta_keys]
            ax.bar(dx, d_means, yerr=d_stds, capsize=5, color="#54a24b", alpha=0.75, edgecolor="black")
            for idx, k in enumerate(delta_keys):
                vec = paired_deltas[k]["_subject_vectors"][metric]
                jitter = rng.uniform(-0.15, 0.15, size=len(vec))
                ax.scatter(np.full(len(vec), idx) + jitter, vec, color="#1f1f1f", s=16, alpha=0.55, zorder=4)
                if math.isfinite(d_means[idx]):
                    ax.text(
                        idx,
                        d_means[idx],
                        f"{d_means[idx]:+.4f}",
                        ha="center",
                        va="bottom" if d_means[idx] >= 0 else "top",
                        fontsize=8.5,
                        fontweight="bold",
                    )
            ax.set_xticks(dx)
            ax.set_xticklabels(d_labels, rotation=25, ha="right")
        else:
            ax.text(0.5, 0.5, "Run ≥2 comparable scenarios to view paired deltas", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    p4 = protocol_dir / "cs5_window_paired_deltas.png"
    fig.savefig(p4, dpi=200)
    plt.close(fig)
    saved_files.append(p4.name)
    return saved_files


def _generate_loso_plots(
    protocol_dir: Path,
    summaries: Mapping[str, Mapping[str, Any]],
    paired_deltas: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    exp_names = list(summaries.keys())
    x = np.arange(len(exp_names))
    colors = [_exp_color(e) for e in exp_names]
    labels = [_exp_label(e) for e in exp_names]
    rng = np.random.default_rng(42)
    saved_files: List[str] = []

    # 1. loso_metric_distribution.png (Boxplots + 39 subject dots + Mean ± SD + Feature Count & Train Time)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle("ML3 LOSO — Held-Out Subject Distributions, Feature Count & Train Time", fontsize=13, fontweight="bold")
    for ax, metric, title in (
        (axes[0, 0], "pr_auc", "LOSO Per-Subject PR-AUC Distribution"),
        (axes[0, 1], "roc_auc", "LOSO Per-Subject ROC-AUC Distribution"),
        (axes[1, 0], "balanced_accuracy", "LOSO Per-Subject Balanced Accuracy Distribution"),
    ):
        series = []
        for e in exp_names:
            vals = [
                float(s[metric])
                for s in summaries[e]["per_subject"].values()
                if s.get(metric) is not None and math.isfinite(float(s[metric]))
            ]
            series.append(vals if vals else [float("nan")])
        bp = ax.boxplot(series, positions=x, widths=0.45, patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for idx, e in enumerate(exp_names):
            vals = series[idx]
            mean_v = summaries[e]["test"]["subject_macro"][metric]["mean"]
            std_v = summaries[e]["test"]["subject_macro"][metric]["std"]
            jitter = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(np.full(len(vals), idx) + jitter, vals, color="#222222", s=18, alpha=0.65, zorder=4)
            ax.errorbar([idx], [mean_v], yerr=[std_v], fmt="D", color="gold", markeredgecolor="black", capsize=5, zorder=5)
            if math.isfinite(mean_v):
                ax.text(idx, min(1.03, max(vals) + 0.02), f"{mean_v:.3f}±{std_v:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.08)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    ax_cost = axes[1, 1]
    feat_counts = [summaries[e]["n_features"] for e in exp_names]
    time_means = [summaries[e]["train_duration_sec"]["mean"] for e in exp_names]
    time_stds = [summaries[e]["train_duration_sec"]["std"] for e in exp_names]
    b_feat = ax_cost.bar(x - 0.18, feat_counts, width=0.34, color="#4c78a8", alpha=0.85, label="Feature Count")
    ax_cost.set_ylabel("Feature Count", color="#4c78a8")
    ax_cost.tick_params(axis="y", labelcolor="#4c78a8")
    for idx, val in enumerate(feat_counts):
        ax_cost.text(b_feat[idx].get_x() + b_feat[idx].get_width() / 2.0, val, f"{val:,}", ha="center", va="bottom", fontsize=8)
    ax_time = ax_cost.twinx()
    ax_time.bar(x + 0.18, time_means, width=0.34, yerr=time_stds, capsize=4, color="#f58518", alpha=0.85, label="Fold Train Time (s)")
    ax_time.set_ylabel("Fold Train Time (s, Mean ± SD)", color="#f58518")
    ax_time.tick_params(axis="y", labelcolor="#f58518")
    ax_cost.set_xticks(x)
    ax_cost.set_xticklabels(labels, rotation=15, ha="right")
    ax_cost.set_title("Feature Count & Per-Subject Fold Train Time")
    ax_cost.grid(axis="y", linestyle="--", alpha=0.3)

    p1 = protocol_dir / "loso_metric_distribution.png"
    fig.savefig(p1, dpi=200)
    plt.close(fig)
    saved_files.append(p1.name)

    # 2. loso_per_subject_metrics.png (Subject-by-subject PR-AUC & ROC-AUC profiles)
    all_subjects = sorted(
        {sub for e in exp_names for sub in summaries[e]["per_subject"].keys()},
        key=_natural_sort_key,
    )
    sx = np.arange(len(all_subjects))
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True, sharex=True)
    fig.suptitle("ML3 LOSO — Per-Subject Generalization Profile (Temporal vs Combined)", fontsize=13, fontweight="bold")
    for ax, metric, title in (
        (axes[0], "pr_auc", "Per-Subject Test PR-AUC"),
        (axes[1], "roc_auc", "Per-Subject Test ROC-AUC"),
    ):
        for e in exp_names:
            sub_map = summaries[e]["per_subject"]
            y_vals = [
                float(sub_map[s][metric]) if s in sub_map and sub_map[s].get(metric) is not None else float("nan")
                for s in all_subjects
            ]
            ax.plot(sx, y_vals, marker="o", markersize=4.5, linewidth=1.5, color=_exp_color(e), label=_exp_label(e), alpha=0.9)
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="lower right", ncol=len(exp_names), fontsize=9)
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(all_subjects, rotation=60, ha="right", fontsize=8)

    p2 = protocol_dir / "loso_per_subject_metrics.png"
    fig.savefig(p2, dpi=200)
    plt.close(fig)
    saved_files.append(p2.name)

    # 3. loso_c5_vs_c8_deltas.png (Paired subject deltas: C8-C5, C5-Temporal, C8-Temporal)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    fig.suptitle("ML3 LOSO — Paired Per-Subject Deltas (C5 vs C8 & Combined vs Temporal)", fontsize=13, fontweight="bold")
    loso_delta_keys = [
        k
        for k in (
            "combined_w8_minus_combined_w5",
            "combined_w5_minus_temporal",
            "combined_w8_minus_temporal",
        )
        if k in paired_deltas
    ]
    for ax, metric, title in (
        (axes[0], "pr_auc", "LOSO Paired Δ PR-AUC (Mean ± SD Across Subjects)"),
        (axes[1], "roc_auc", "LOSO Paired Δ ROC-AUC (Mean ± SD Across Subjects)"),
    ):
        ax.axhline(0.0, color="crimson", linestyle="--", linewidth=1.2)
        if loso_delta_keys:
            dx = np.arange(len(loso_delta_keys))
            d_labels = [paired_deltas[k]["comparison"] for k in loso_delta_keys]
            d_means = [paired_deltas[k]["subject_paired_delta"][metric]["mean"] for k in loso_delta_keys]
            d_stds = [paired_deltas[k]["subject_paired_delta"][metric]["std"] for k in loso_delta_keys]
            ax.bar(dx, d_means, yerr=d_stds, capsize=5, color="#72b7b2", alpha=0.78, edgecolor="black")
            for idx, k in enumerate(loso_delta_keys):
                vec = paired_deltas[k]["_subject_vectors"][metric]
                jitter = rng.uniform(-0.14, 0.14, size=len(vec))
                ax.scatter(np.full(len(vec), idx) + jitter, vec, color="#1f1f1f", s=18, alpha=0.6, zorder=4)
                if math.isfinite(d_means[idx]):
                    ax.text(
                        idx,
                        d_means[idx],
                        f"{d_means[idx]:+.4f}±{d_stds[idx]:.4f}",
                        ha="center",
                        va="bottom" if d_means[idx] >= 0 else "top",
                        fontsize=8.5,
                        fontweight="bold",
                    )
            ax.set_xticks(dx)
            ax.set_xticklabels(d_labels, rotation=20, ha="right")
        else:
            ax.text(0.5, 0.5, "Run ≥2 LOSO experiments (temporal, combined_w5, combined_w8) for paired deltas", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    p3 = protocol_dir / "loso_c5_vs_c8_deltas.png"
    fig.savefig(p3, dpi=200)
    plt.close(fig)
    saved_files.append(p3.name)
    return saved_files


# ── High-Level Planning & Execution Orchestrator ─────────────────────────────

def build_ml3_execution_plan(
    *,
    features_dir: Path | str,
    splits_dir: Path | str,
    protocol: ProtocolName | str,
    experiments: Optional[Sequence[str]] = None,
    output_dir: Path | str,
    model_config: ML3XGBoostConfig = ML3XGBoostConfig(),
    folds_filter: Optional[Sequence[int]] = None,
    threshold_metric: str = "balanced_accuracy",
    allow_gamma_loso: bool = False,
    mount_drive: bool = False,
    drive_mount_point: Path | str = "/content/drive",
) -> Dict[str, Any]:
    """Validate inputs and build a dry-run execution summary (used by ``--plan-only``)."""
    norm_protocol = normalize_protocol_name(protocol)
    chosen_experiments = validate_experiment_list(
        norm_protocol, experiments, allow_gamma_loso=allow_gamma_loso
    )
    req_groups = required_feature_groups_for_experiments(chosen_experiments)
    all_folds, split_meta = load_ml3_splits(splits_dir, norm_protocol)

    if folds_filter is not None:
        wanted = set(int(f) for f in folds_filter)
        available = {f.fold for f in all_folds}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested fold indices not present in {norm_protocol}: {missing}")
        selected_folds = tuple(f for f in all_folds if f.fold in wanted)
    else:
        selected_folds = all_folds

    store_info = inspect_feature_store_metadata(
        features_dir,
        required_groups=req_groups,
        mount_drive=mount_drive,
        mount_point=drive_mount_point,
    )
    out_root = ensure_drive_path_ready(
        output_dir, mount_drive=mount_drive, mount_point=drive_mount_point
    )

    split_subjects = set()
    for f in selected_folds:
        split_subjects.update(f.train_subjects)
        split_subjects.update(f.val_subjects)
        split_subjects.update(f.test_subjects)
    missing_subs = sorted(split_subjects - set(store_info["subjects"]))
    if missing_subs:
        raise ValueError(
            f"Feature store at {store_info['features_dir']} is missing subjects required by splits: {missing_subs}"
        )

    n_ch = int(store_info["n_channels"])
    scenario_details = []
    for exp in chosen_experiments:
        spec = FEATURE_SCENARIO_SPECS[exp]
        n_feat = 0
        if spec.uses_temporal:
            n_feat += 8 * n_ch * 7
        if spec.wavelet_windows is not None:
            n_feat += len(spec.wavelet_windows) * n_ch * (6 if spec.uses_gamma else 5)
        scenario_details.append(
            {
                "experiment": exp,
                "expected_n_features": n_feat,
                "output_dir": str(out_root / norm_protocol / exp),
                "policy": spec.policy_dict(),
            }
        )

    return {
        "protocol": norm_protocol,
        "features_dir": store_info["features_dir"],
        "output_dir": str(out_root / norm_protocol),
        "required_feature_groups": list(req_groups),
        "n_subjects_available": store_info["n_subjects"],
        "n_channels": n_ch,
        "split_metadata": split_meta,
        "n_folds_selected": len(selected_folds),
        "fold_indices": [f.fold for f in selected_folds],
        "threshold_metric": threshold_metric,
        "model_config": model_config.report_dict(),
        "experiments": scenario_details,
    }


def run_ml3_experiments(
    *,
    features_dir: Path | str,
    splits_dir: Path | str,
    protocol: ProtocolName | str,
    experiments: Optional[Sequence[str]] = None,
    output_dir: Path | str,
    model_config: ML3XGBoostConfig = ML3XGBoostConfig(),
    folds_filter: Optional[Sequence[int]] = None,
    threshold_metric: str = "balanced_accuracy",
    resume: bool = False,
    overwrite: bool = False,
    allow_gamma_loso: bool = False,
    mount_drive: bool = False,
    drive_mount_point: Path | str = "/content/drive",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Execute requested ML3 feature scenarios across CS5 or LOSO folds."""
    norm_protocol = normalize_protocol_name(protocol)
    chosen_experiments = validate_experiment_list(
        norm_protocol, experiments, allow_gamma_loso=allow_gamma_loso
    )
    req_groups = required_feature_groups_for_experiments(chosen_experiments)
    needs_gamma = any(FEATURE_SCENARIO_SPECS[e].uses_gamma for e in chosen_experiments)

    all_folds, split_meta = load_ml3_splits(splits_dir, norm_protocol)
    if folds_filter is not None:
        wanted = set(int(f) for f in folds_filter)
        available = {f.fold for f in all_folds}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested fold indices not present in {norm_protocol}: {missing}")
        selected_folds = tuple(f for f in all_folds if f.fold in wanted)
    else:
        selected_folds = all_folds

    needed_subjects = sorted(
        {
            s
            for f in selected_folds
            for s in (*f.train_subjects, *f.val_subjects, *f.test_subjects)
        },
        key=_natural_sort_key,
    )

    out_root = ensure_drive_path_ready(
        output_dir, mount_drive=mount_drive, mount_point=drive_mount_point
    )
    protocol_dir = out_root / norm_protocol
    protocol_dir.mkdir(parents=True, exist_ok=True)

    dataset: ML3FeatureDataset = load_ml3_feature_dataset(
        features_dir,
        subjects=needed_subjects,
        required_groups=req_groups,
        include_gamma_support=needs_gamma,
        mount_drive=mount_drive,
        mount_point=drive_mount_point,
    )

    experiment_summaries: Dict[str, Any] = {}
    fold_statuses: Dict[str, Dict[str, int]] = {}

    for exp_name in chosen_experiments:
        scenario_matrix = dataset.build_scenario(exp_name)
        exp_dir = protocol_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(
                f"[ML3][{norm_protocol.upper()}][{exp_name}] "
                f"samples={scenario_matrix.n_samples:,}, features={scenario_matrix.n_features:,}, "
                f"folds={len(selected_folds)}, device={model_config.device}"
            )

        fold_metrics_list: list[Dict[str, Any]] = []
        fold_preds_list: list[Dict[str, np.ndarray]] = []
        trained_count = 0
        reused_count = 0

        for split in selected_folds:
            fold_dir = exp_dir / f"fold_{split.fold:02d}"
            f_metrics, f_preds, status = train_or_load_fold(
                scenario_matrix=scenario_matrix,
                split=split,
                model_config=model_config,
                fold_dir=fold_dir,
                threshold_metric=threshold_metric,
                resume=resume,
                overwrite=overwrite,
            )
            fold_metrics_list.append(f_metrics)
            fold_preds_list.append(f_preds)
            if status == "reused":
                reused_count += 1
            else:
                trained_count += 1

            if verbose:
                tp = f_metrics["test_pooled"]
                print(
                    f"  fold={split.fold:02d} [{status}] "
                    f"best_iter={f_metrics['best_iteration']} "
                    f"thr={f_metrics['threshold']:.3f} "
                    f"PR-AUC={tp['pr_auc']:.4f} "
                    f"ROC-AUC={tp['roc_auc']:.4f} "
                    f"BalAcc={tp['balanced_accuracy']:.4f} "
                    f"time={f_metrics['train_duration_sec']:.1f}s"
                )

        summary = aggregate_experiment_results(
            protocol=norm_protocol,
            experiment=exp_name,
            exp_dir=exp_dir,
            model_config=model_config,
            fold_metrics_list=fold_metrics_list,
            fold_preds_list=fold_preds_list,
        )
        experiment_summaries[exp_name] = summary
        fold_statuses[exp_name] = {"trained": trained_count, "reused": reused_count}

        del scenario_matrix
        gc.collect()

    comparison = write_protocol_comparison_and_plots(norm_protocol, protocol_dir)

    return {
        "protocol": norm_protocol,
        "output_dir": str(protocol_dir),
        "split_metadata": split_meta,
        "fold_statuses": fold_statuses,
        "experiments": experiment_summaries,
        "comparison": comparison,
    }
