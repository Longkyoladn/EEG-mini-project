"""Điều phối chạy Go/No-Go Pilot Benchmark (Bước 4) trên Full 39 subjects và xuất báo cáo.

Chạy tuần tự 8 thực nghiệm chuẩn Zero Leakage qua `scripts/train_dl2.py`:
  - Nhóm A (Cross-Subject 5-Fold, 39 subjects, batch_size=64):
      1. pilot_cs5_eegnet
      2. pilot_cs5_atcnet_avg
      3. pilot_cs5_atcnet_attn
      4. pilot_cs5_atcnet_last
  - Nhóm B (Within-Subject 5-Fold, 4 subjects: sub-03, sub-05, sub-07, sub-10, batch_size=32):
      5. pilot_ws_eegnet
      6. pilot_ws_atcnet_avg
      7. pilot_ws_atcnet_attn
      8. pilot_ws_atcnet_shared

Sau khi chạy xong, tự động đọc `results/metrics/dl2/metrics_{exp_name}.json` và sinh báo cáo
`reports/STEP4_PILOT_BENCHMARK.md` gồm 4 bảng tổng hợp theo yêu cầu.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.train_dl2 import build_arg_parser as build_train_arg_parser
from scripts.train_dl2 import run_experiment
from src.evaluation.metrics import METRIC_NAMES
from src.training.trainer_dl2 import build_model


@dataclass(frozen=True)
class PilotExperimentSpec:
    """Đặc tả một thực nghiệm trong Step 4 Go/No-Go Pilot Benchmark."""

    exp_name: str
    group: str
    display_name: str
    cli_args: Tuple[str, ...]


GROUP_A_SPECS: Tuple[PilotExperimentSpec, ...] = (
    PilotExperimentSpec(
        exp_name="pilot_cs5_eegnet",
        group="A",
        display_name="EEGNet-v4",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--model", "eegnet",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_cs5_atcnet_avg",
        group="A",
        display_name="P300-ATCNet (avg_pool)",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--model", "p300_atcnet",
            "--readout", "avg_pool",
            "--n-windows", "2",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_cs5_atcnet_attn",
        group="A",
        display_name="P300-ATCNet (attn_pool)",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--model", "p300_atcnet",
            "--readout", "attn_pool",
            "--n-windows", "2",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_cs5_atcnet_last",
        group="A",
        display_name="P300-ATCNet (last_step)",
        cli_args=(
            "--protocol", "cross_subject_5fold",
            "--batch-size", "64",
            "--model", "p300_atcnet",
            "--readout", "last_step",
            "--n-windows", "2",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
)

GROUP_B_SUBJECTS: Tuple[str, ...] = ("sub-03", "sub-05", "sub-07", "sub-10")

GROUP_B_SPECS: Tuple[PilotExperimentSpec, ...] = (
    PilotExperimentSpec(
        exp_name="pilot_ws_eegnet",
        group="B",
        display_name="EEGNet-v4",
        cli_args=(
            "--protocol", "within_subject",
            "--subjects", *GROUP_B_SUBJECTS,
            "--batch-size", "32",
            "--model", "eegnet",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_ws_atcnet_avg",
        group="B",
        display_name="P300-ATCNet (avg_pool)",
        cli_args=(
            "--protocol", "within_subject",
            "--subjects", *GROUP_B_SUBJECTS,
            "--batch-size", "32",
            "--model", "p300_atcnet",
            "--readout", "avg_pool",
            "--n-windows", "2",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_ws_atcnet_attn",
        group="B",
        display_name="P300-ATCNet (attn_pool)",
        cli_args=(
            "--protocol", "within_subject",
            "--subjects", *GROUP_B_SUBJECTS,
            "--batch-size", "32",
            "--model", "p300_atcnet",
            "--readout", "attn_pool",
            "--n-windows", "2",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
    PilotExperimentSpec(
        exp_name="pilot_ws_atcnet_shared",
        group="B",
        display_name="P300-ATCNet (avg_pool, shared)",
        cli_args=(
            "--protocol", "within_subject",
            "--subjects", *GROUP_B_SUBJECTS,
            "--batch-size", "32",
            "--model", "p300_atcnet",
            "--readout", "avg_pool",
            "--n-windows", "2",
            "--shared-branch",
            "--tcn-filters", "32",
            "--temp-kernel-1", "125",
            "--spatial-max-norm", "2.0",
            "--dropout", "0.3",
        ),
    ),
)

ALL_SPECS: Tuple[PilotExperimentSpec, ...] = GROUP_A_SPECS + GROUP_B_SPECS


def _fmt_mean_sd(mean_val: Optional[float], std_val: Optional[float], decimals: int = 4) -> str:
    if mean_val is None or not math.isfinite(float(mean_val)):
        return "NaN"
    if std_val is None or not math.isfinite(float(std_val)):
        return f"{float(mean_val):.{decimals}f}"
    return f"{float(mean_val):.{decimals}f} ± {float(std_val):.{decimals}f}"


def _fmt_duration(seconds: float) -> str:
    return f"{seconds:.1f}s ({seconds / 60.0:.2f}m)"


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.2f} phút ({seconds:.0f}s)"


def _count_model_params(config: Dict[str, Any]) -> int:
    model_name = str(config["model"])
    model_kwargs = dict(config.get("model_kwargs") or {})
    model, _ = build_model(model_name, model_kwargs)
    return int(model.count_parameters(trainable_only=True))


def _load_metrics_json(metrics_dir: Path, exp_name: str) -> Dict[str, Any]:
    path = metrics_dir / f"metrics_{exp_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file metrics: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_experiment_integrity(
    metrics_data: Dict[str, Any],
    pred_path: Path,
) -> Tuple[int, int, int]:
    """Kiểm tra fold dừng sớm < 5 epochs, số NaN trong metrics/fold và predictions.csv."""
    per_fold = metrics_data["per_fold"]
    early_lt5 = sum(1 for rec in per_fold if int(rec.get("best_epoch", 0)) < 5)

    nan_fold_count = 0
    check_keys = (*METRIC_NAMES, "best_val_metric", "threshold")
    for rec in per_fold:
        has_bad = False
        for key in check_keys:
            val = rec.get(key)
            if val is None or not math.isfinite(float(val)):
                has_bad = True
                break
        if has_bad:
            nan_fold_count += 1

    pred_nan_count = 0
    if not pred_path.exists():
        pred_nan_count = -1
    else:
        df_pred = pd.read_csv(pred_path)
        if df_pred.isna().any().any() or not np.isfinite(df_pred["y_prob"].to_numpy(dtype=np.float64)).all():
            pred_nan_count = int(df_pred.isna().sum().sum())

    return early_lt5, nan_fold_count, pred_nan_count


def _compute_split_scale_factors(data_dir: Path) -> Dict[str, float]:
    """Tính tỷ lệ kích thước tập train thực tế từ `splits.json` cho ước lượng Bước 5."""
    splits_path = data_dir / "splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    ws_dict = splits["within_subject"]
    ws_pilot_trains = [
        len(fold["train_idx"])
        for sub_id in GROUP_B_SUBJECTS
        for fold in ws_dict[sub_id]
    ]
    ws_full_trains = [
        len(fold["train_idx"])
        for folds in ws_dict.values()
        for fold in folds
    ]

    cs5_trains = [len(fold["train_idx"]) for fold in splits["cross_subject_5fold"]]
    loso_trains = [len(fold["train_idx"]) for fold in splits["cross_subject_loso"]]

    ws_pilot_mean = float(np.mean(ws_pilot_trains))
    ws_full_mean = float(np.mean(ws_full_trains))
    cs5_mean = float(np.mean(cs5_trains))
    loso_mean = float(np.mean(loso_trains))

    return {
        "n_ws_pilot_folds": float(len(ws_pilot_trains)),
        "n_ws_full_folds": float(len(ws_full_trains)),
        "n_cs5_folds": float(len(cs5_trains)),
        "n_loso_folds": float(len(loso_trains)),
        "ws_pilot_mean_train": ws_pilot_mean,
        "ws_full_mean_train": ws_full_mean,
        "ws_scale": ws_full_mean / ws_pilot_mean,
        "cs5_mean_train": cs5_mean,
        "loso_mean_train": loso_mean,
        "loso_scale": loso_mean / cs5_mean,
    }


def _build_qc_lookup(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    qc_path = data_dir / "preprocessing_qc_summary.csv"
    if not qc_path.exists():
        return {}
    df_qc = pd.read_csv(qc_path)
    return {str(row["subject_id"]): row.to_dict() for _, row in df_qc.iterrows()}


def generate_benchmark_report(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
) -> Path:
    """Đọc kết quả 8 thực nghiệm từ JSON/CSV và ghi `reports/STEP4_PILOT_BENCHMARK.md`."""
    metrics_dir = output_dir / "metrics" / "dl2"
    loaded: Dict[str, Dict[str, Any]] = {
        spec.exp_name: _load_metrics_json(metrics_dir, spec.exp_name)
        for spec in ALL_SPECS
    }
    scale_info = _compute_split_scale_factors(data_dir)
    qc_lookup = _build_qc_lookup(data_dir)

    gpu_name = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else "CPU"
    )

    lines: List[str] = [
        "# BƯỚC 4 — BÁO CÁO GO/NO-GO PILOT BENCHMARK (EEGNet-v4 vs P300-ATCNet)",
        "",
        f"- **Tập dữ liệu**: `{data_dir.as_posix()}` (39 subjects sạch, 20,886 epochs)",
        f"- **Phần cứng thực thi**: `{gpu_name}`",
        "- **Siêu tham số chung**: `epochs=80`, `patience=15`, `lr=1e-3`, `weight_decay=1e-2`, `seed=42`, `early_stop_metric=val_roc_auc`, `threshold_metric=balanced_accuracy`",
        "",
    ]

    # ------------------------------------------------------------------
    # BẢNG 1 — Cross-Subject 5-Fold (39 subjects)
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 1 — Cross-Subject 5-Fold (39 subjects, `batch_size=64`)",
        "",
        "| Thực nghiệm | Mô hình / Readout | Params | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) | F1-Target (Mean ± SD) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | Mean Best Epoch | Thời gian chạy |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
    ])
    for spec in GROUP_A_SPECS:
        data = loaded[spec.exp_name]
        summary = data["summary"]
        per_fold = data["per_fold"]
        n_params = _count_model_params(data["config"])
        mean_best_ep = float(np.mean([f["best_epoch"] for f in per_fold]))
        total_sec = float(sum(f["train_time_sec"] for f in per_fold))
        lines.append(
            f"| `{spec.exp_name}` | {spec.display_name} | {n_params:,} | "
            f"{_fmt_mean_sd(summary['roc_auc']['mean'], summary['roc_auc']['std'])} | "
            f"{_fmt_mean_sd(summary['pr_auc']['mean'], summary['pr_auc']['std'])} | "
            f"{_fmt_mean_sd(summary['balanced_accuracy']['mean'], summary['balanced_accuracy']['std'])} | "
            f"{_fmt_mean_sd(summary['f1_target']['mean'], summary['f1_target']['std'])} | "
            f"{_fmt_mean_sd(summary['sensitivity']['mean'], summary['sensitivity']['std'])} | "
            f"{_fmt_mean_sd(summary['specificity']['mean'], summary['specificity']['std'])} | "
            f"{mean_best_ep:.1f} | {_fmt_duration(total_sec)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 2 — Within-Subject 5-Fold Tổng hợp (4 subjects)
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 2 — Within-Subject 5-Fold Tổng hợp (4 subjects: `sub-03`, `sub-05`, `sub-07`, `sub-10`, `batch_size=32`)",
        "",
        "| Thực nghiệm | Mô hình / Cấu hình | Params | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) | F1-Target (Mean ± SD) | Sensitivity (Mean ± SD) | Specificity (Mean ± SD) | Mean Best Epoch | Thời gian chạy |",
        "| :--- | :--- | ---: | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
    ])
    for spec in GROUP_B_SPECS:
        data = loaded[spec.exp_name]
        summary = data["summary"]
        per_fold = data["per_fold"]
        n_params = _count_model_params(data["config"])
        mean_best_ep = float(np.mean([f["best_epoch"] for f in per_fold]))
        total_sec = float(sum(f["train_time_sec"] for f in per_fold))
        lines.append(
            f"| `{spec.exp_name}` | {spec.display_name} | {n_params:,} | "
            f"{_fmt_mean_sd(summary['roc_auc']['mean'], summary['roc_auc']['std'])} | "
            f"{_fmt_mean_sd(summary['pr_auc']['mean'], summary['pr_auc']['std'])} | "
            f"{_fmt_mean_sd(summary['balanced_accuracy']['mean'], summary['balanced_accuracy']['std'])} | "
            f"{_fmt_mean_sd(summary['f1_target']['mean'], summary['f1_target']['std'])} | "
            f"{_fmt_mean_sd(summary['sensitivity']['mean'], summary['sensitivity']['std'])} | "
            f"{_fmt_mean_sd(summary['specificity']['mean'], summary['specificity']['std'])} | "
            f"{mean_best_ep:.1f} | {_fmt_duration(total_sec)} |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 3 — Phân rã Within-Subject theo từng cá nhân
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 3 — Phân rã Within-Subject theo từng cá nhân (`sub-03`, `sub-05`, `sub-07`, `sub-10`)",
        "",
        "| Thực nghiệm | Mô hình | Subject / Nhóm | Epochs (Target) | ROC-AUC (Mean ± SD) | PR-AUC (Mean ± SD) | Balanced Acc (Mean ± SD) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ])
    sub_tags = {
        "sub-03": "sub-03 (Sạch)",
        "sub-05": "sub-05 (Sạch)",
        "sub-07": "sub-07 (Ít Target)",
        "sub-10": "sub-10 (Ít Target)",
    }
    for spec in GROUP_B_SPECS:
        per_sub = loaded[spec.exp_name]["per_subject"]
        for sub_id in GROUP_B_SUBJECTS:
            entry = per_sub[sub_id]
            qc_row = qc_lookup.get(sub_id, {})
            ep_str = (
                f"{int(qc_row['n_clean_epochs_kept'])} ({int(qc_row['n_target'])})"
                if qc_row
                else "—"
            )
            lines.append(
                f"| `{spec.exp_name}` | {spec.display_name} | {sub_tags[sub_id]} | {ep_str} | "
                f"{_fmt_mean_sd(entry['roc_auc'], entry['roc_auc_std'])} | "
                f"{_fmt_mean_sd(entry['pr_auc'], entry['pr_auc_std'])} | "
                f"{_fmt_mean_sd(entry['balanced_accuracy'], entry['balanced_accuracy_std'])} |"
            )

        clean_roc = float(np.mean([per_sub["sub-03"]["roc_auc"], per_sub["sub-05"]["roc_auc"]]))
        clean_pr = float(np.mean([per_sub["sub-03"]["pr_auc"], per_sub["sub-05"]["pr_auc"]]))
        clean_bacc = float(
            np.mean([per_sub["sub-03"]["balanced_accuracy"], per_sub["sub-05"]["balanced_accuracy"]])
        )
        low_roc = float(np.mean([per_sub["sub-07"]["roc_auc"], per_sub["sub-10"]["roc_auc"]]))
        low_pr = float(np.mean([per_sub["sub-07"]["pr_auc"], per_sub["sub-10"]["pr_auc"]]))
        low_bacc = float(
            np.mean([per_sub["sub-07"]["balanced_accuracy"], per_sub["sub-10"]["balanced_accuracy"]])
        )
        lines.append(
            f"| `{spec.exp_name}` | {spec.display_name} | **TB Nhóm sạch (03, 05)** | — | "
            f"**{clean_roc:.4f}** | **{clean_pr:.4f}** | **{clean_bacc:.4f}** |"
        )
        lines.append(
            f"| `{spec.exp_name}` | {spec.display_name} | **TB Nhóm ít Target (07, 10)** | — | "
            f"**{low_roc:.4f}** | **{low_pr:.4f}** | **{low_bacc:.4f}** |"
        )
        lines.append(
            f"| `{spec.exp_name}` | {spec.display_name} | **Δ (Sạch − Ít Target)** | — | "
            f"**{clean_roc - low_roc:+.4f}** | **{clean_pr - low_pr:+.4f}** | **{clean_bacc - low_bacc:+.4f}** |"
        )
    lines.append("")

    # ------------------------------------------------------------------
    # BẢNG 4 — Kiểm tra Val vs Test & Ước lượng Full Bước 5
    # ------------------------------------------------------------------
    lines.extend([
        "### Bảng 4 — Kiểm tra chênh lệch Validation vs Test & Ước lượng thời gian chạy Full Bước 5",
        "",
        "#### Bảng 4A — Kiểm tra chênh lệch `best_val_metric` (Val ROC-AUC) vs `test_roc_auc` & An toàn hội tụ",
        "",
        "| Thực nghiệm | Protocol | Val ROC-AUC (Mean ± SD) | Test ROC-AUC (Mean ± SD) | Δ (Val − Test) | best_epoch (Mean [Min..Max]) | Folds ep < 5 | NaN (Fold / Pred) |",
        "| :--- | :--- | :---: | :---: | ---: | :---: | ---: | :---: |",
    ])

    for spec in ALL_SPECS:
        data = loaded[spec.exp_name]
        per_fold = data["per_fold"]
        val_scores = np.array([float(f["best_val_metric"]) for f in per_fold], dtype=np.float64)
        test_scores = np.array([float(f["roc_auc"]) for f in per_fold], dtype=np.float64)
        best_eps = [int(f["best_epoch"]) for f in per_fold]

        val_mean = float(np.mean(val_scores))
        val_std = float(np.std(val_scores, ddof=1 if len(val_scores) > 1 else 0))
        test_mean = float(np.mean(test_scores))
        test_std = float(np.std(test_scores, ddof=1 if len(test_scores) > 1 else 0))
        gap = val_mean - test_mean

        pred_path = metrics_dir / f"predictions_{spec.exp_name}.csv"
        early_lt5, nan_folds, pred_nans = _check_experiment_integrity(data, pred_path)

        lines.append(
            f"| `{spec.exp_name}` | `{data['config']['protocol']}` | "
            f"{_fmt_mean_sd(val_mean, val_std)} | "
            f"{_fmt_mean_sd(test_mean, test_std)} | "
            f"{gap:+.4f} | "
            f"{float(np.mean(best_eps)):.1f} [{min(best_eps)}..{max(best_eps)}] | "
            f"{early_lt5}/{len(per_fold)} | {nan_folds} / {pred_nans} |"
        )

    lines.extend([
        "",
        "#### Bảng 4B — Ước lượng thời gian chạy Full Bước 5 trên GPU RTX 5060 Ti (195 folds Within-Subject + 39 folds LOSO)",
        "",
        f"- Hệ số hiệu chỉnh quy mô tập train từ `splits.json`: Within-Subject `342.4 / 262.4 = {scale_info['ws_scale']:.4f}x` (195 folds vs 20 pilot folds); LOSO `18,195.6 / 13,487.2 = {scale_info['loso_scale']:.4f}x` (39 folds vs 5 CS5 folds).",
        "",
        "| Kiến trúc / Cấu hình | WS Pilot (20f) | WS-195 Trực tiếp | WS-195 Hiệu chỉnh | CS5 Pilot (5f) | LOSO-39 Trực tiếp | LOSO-39 Hiệu chỉnh | Tổng Bước 5 (Hiệu chỉnh) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    step5_pairs: Tuple[Tuple[str, Optional[str], Optional[str]], ...] = (
        ("EEGNet-v4", "pilot_ws_eegnet", "pilot_cs5_eegnet"),
        ("P300-ATCNet (avg_pool)", "pilot_ws_atcnet_avg", "pilot_cs5_atcnet_avg"),
        ("P300-ATCNet (attn_pool)", "pilot_ws_atcnet_attn", "pilot_cs5_atcnet_attn"),
        ("P300-ATCNet (last_step)", None, "pilot_cs5_atcnet_last"),
        ("P300-ATCNet (avg_pool, shared)", "pilot_ws_atcnet_shared", None),
    )

    total_step5_adj_sec = 0.0
    for label, ws_exp, cs5_exp in step5_pairs:
        if ws_exp is not None:
            ws_sec = float(sum(f["train_time_sec"] for f in loaded[ws_exp]["per_fold"]))
            ws_per_fold = ws_sec / len(loaded[ws_exp]["per_fold"])
            ws195_direct = ws_per_fold * scale_info["n_ws_full_folds"]
            ws195_adj = ws195_direct * scale_info["ws_scale"]
            ws_pilot_str = f"{ws_sec:.1f}s ({ws_per_fold:.2f}s/f)"
            ws_dir_str = _fmt_minutes(ws195_direct)
            ws_adj_str = _fmt_minutes(ws195_adj)
        else:
            ws195_adj = 0.0
            ws_pilot_str = "—"
            ws_dir_str = "—"
            ws_adj_str = "—"

        if cs5_exp is not None:
            cs5_sec = float(sum(f["train_time_sec"] for f in loaded[cs5_exp]["per_fold"]))
            cs5_per_fold = cs5_sec / len(loaded[cs5_exp]["per_fold"])
            loso_direct = cs5_per_fold * scale_info["n_loso_folds"]
            loso_adj = loso_direct * scale_info["loso_scale"]
            cs5_pilot_str = f"{cs5_sec:.1f}s ({cs5_per_fold:.2f}s/f)"
            loso_dir_str = _fmt_minutes(loso_direct)
            loso_adj_str = _fmt_minutes(loso_adj)
        else:
            loso_adj = 0.0
            cs5_pilot_str = "—"
            loso_dir_str = "—"
            loso_adj_str = "—"

        if ws_exp is not None and cs5_exp is not None:
            combo_adj = ws195_adj + loso_adj
            combo_str = f"**{_fmt_minutes(combo_adj)}**"
            if label in ("EEGNet-v4", "P300-ATCNet (avg_pool)"):
                total_step5_adj_sec += combo_adj
        elif ws_exp is not None:
            combo_str = f"{ws_adj_str} (chỉ WS)"
        else:
            combo_str = f"{loso_adj_str} (chỉ LOSO)"

        lines.append(
            f"| {label} | {ws_pilot_str} | {ws_dir_str} | {ws_adj_str} | "
            f"{cs5_pilot_str} | {loso_dir_str} | {loso_adj_str} | {combo_str} |"
        )

    lines.extend([
        "",
        f"> **Tổng thời gian thực chạy 8 thực nghiệm Pilot**: `{_fmt_duration(sum(float(sum(f['train_time_sec'] for f in loaded[s.exp_name]['per_fold'])) for s in ALL_SPECS))}`.",
        f"> **Dự kiến tổng thời gian chạy cặp mô hình chính Bước 5 (`EEGNet-v4` + `P300-ATCNet avg_pool`, 195 WS + 39 LOSO mỗi mô hình)**: `{_fmt_minutes(total_step5_adj_sec)}` (hiệu chỉnh theo quy mô tập train).",
        "",
    ])

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_all_pilot_experiments(
    data_dir: Path,
    output_dir: Path,
    report_path: Path,
    *,
    report_only: bool = False,
) -> Path:
    """Chạy tuần tự 8 thực nghiệm Pilot và xuất báo cáo `STEP4_PILOT_BENCHMARK.md`."""
    if not report_only:
        train_parser = build_train_arg_parser()
        common_args = [
            "--data-dir", str(data_dir),
            "--epochs", "80",
            "--patience", "15",
            "--lr", "1e-3",
            "--weight-decay", "1e-2",
            "--seed", "42",
            "--resume",
            "--output-dir", str(output_dir),
        ]

        t_all_start = time.perf_counter()
        for idx, spec in enumerate(ALL_SPECS, start=1):
            print(f"\n========== [{idx}/{len(ALL_SPECS)}] Running {spec.exp_name} ==========")
            argv = ["--exp-name", spec.exp_name, *common_args, *spec.cli_args]
            train_args = train_parser.parse_args(argv)
            run_experiment(train_args)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        total_wall = time.perf_counter() - t_all_start
        print(f"\n[run_step4_pilot] Hoàn tất 8/8 thực nghiệm trong {total_wall:.2f}s ({total_wall / 60.0:.2f}m).")

    saved_report = generate_benchmark_report(
        data_dir=data_dir,
        output_dir=output_dir,
        report_path=report_path,
    )
    print(f"[run_step4_pilot] Đã sinh báo cáo tại: {saved_report}")
    return saved_report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chạy 8 thực nghiệm Go/No-Go Pilot Benchmark (Bước 4) và sinh báo cáo."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục dữ liệu đã tiền xử lý (mặc định: data/processed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Thư mục gốc lưu metrics và checkpoints (mặc định: results).",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("reports/STEP4_PILOT_BENCHMARK.md"),
        help="Đường dẫn file báo cáo markdown đầu ra.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Chỉ tổng hợp lại báo cáo từ các file metrics JSON đã có.",
    )
    args = parser.parse_args(argv)
    run_all_pilot_experiments(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        report_path=args.report_path,
        report_only=args.report_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
