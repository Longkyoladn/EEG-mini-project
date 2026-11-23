"""Adapter xuất bộ bàn giao chuẩn Đề cương nhóm (Mục 3.5) và 3 file splits rời.

Chạy độc lập, không train lại mô hình và không sinh lại seed khác:
    py -3.14 scripts/evaluation/export_group_handover.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PER_FOLD_COLUMNS: List[str] = [
    "subject_id",
    "fold",
    "bal_acc",
    "roc_auc",
    "pr_auc",
    "f1_target",
    "sensitivity",
    "specificity",
]

METRIC_COLS: List[str] = [
    "bal_acc",
    "roc_auc",
    "pr_auc",
    "f1_target",
    "sensitivity",
    "specificity",
]

# (output_prefix, source_metrics_json, expected_rows)
PER_FOLD_SPECS: List[Tuple[str, str, int]] = [
    ("p300_atcnet_within", "metrics_dl2_ws_p300_atcnet.json", 195),
    ("p300_atcnet_loso", "metrics_dl2_loso_p300_atcnet.json", 39),
    ("p300_atcnet_cs5_clean", "metrics_dl2_cs5_p300_atcnet_clean.json", 5),
    ("p300_atcnet_cs5_no_ica", "metrics_dl2_cs5_p300_atcnet_no_ica.json", 5),
]

# (output_prefix, source_predictions_csv)
PREDS_SPECS: List[Tuple[str, str]] = [
    ("p300_atcnet_within", "predictions_dl2_ws_p300_atcnet.csv"),
    ("p300_atcnet_loso", "predictions_dl2_loso_p300_atcnet.csv"),
    ("p300_atcnet_cs5_clean", "predictions_dl2_cs5_p300_atcnet_clean.csv"),
]


def _compute_trial_uid_sha256(trial_uids: Sequence[str]) -> str:
    payload = "\n".join(trial_uids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _format_compact_json(payload: Dict[str, Any]) -> str:
    """Format JSON với indent=2 nhưng gom gọn danh sách chỉ số/chuỗi nguyên thủy trên 1 dòng."""
    raw = json.dumps(payload, indent=2)
    return re.sub(
        r"\[\s+([^\[\]\{\}]+?)\s+\]",
        lambda m: "[" + " ".join(line.strip() for line in m.group(1).splitlines()) + "]",
        raw,
    )


def export_per_fold_csvs(metrics_dir: Path, handover_dir: Path) -> List[Path]:
    """Xuất các file `{model}_{setting}_per_fold.csv` chuẩn 8 cột từ metrics JSON."""
    handover_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Path] = []

    for prefix, src_name, expected_rows in PER_FOLD_SPECS:
        src_path = metrics_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Thiếu file metrics nguồn: {src_path}")

        with open(src_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        per_fold = payload.get("per_fold", [])
        if len(per_fold) != expected_rows:
            raise ValueError(
                f"{src_name}: kỳ vọng {expected_rows} dòng trong per_fold, nhận {len(per_fold)}"
            )

        rows: List[Dict[str, Any]] = []
        for item in per_fold:
            row = {
                "subject_id": str(item["subject_id"]),
                "fold": int(item["fold"]),
                "bal_acc": float(item["balanced_accuracy"]),
                "roc_auc": float(item["roc_auc"]),
                "pr_auc": float(item["pr_auc"]),
                "f1_target": float(item["f1_target"]),
                "sensitivity": float(item["sensitivity"]),
                "specificity": float(item["specificity"]),
            }
            rows.append(row)

        df = pd.DataFrame(rows, columns=PER_FOLD_COLUMNS)
        if df.isna().any().any():
            raise ValueError(f"{prefix}_per_fold.csv chứa giá trị NaN")

        for col in METRIC_COLS:
            vals = df[col].to_numpy(dtype=np.float64)
            if not np.isfinite(vals).all():
                raise ValueError(f"{prefix}_per_fold.csv cột {col} chứa NaN/Inf")
            if not ((vals >= 0.0) & (vals <= 1.0)).all():
                raise ValueError(
                    f"{prefix}_per_fold.csv cột {col} có giá trị ngoài [0, 1]: "
                    f"[{vals.min()}, {vals.max()}]"
                )

        out_path = handover_dir / f"{prefix}_per_fold.csv"
        df.to_csv(out_path, index=False)
        exported.append(out_path)
        print(
            f"[Export CSV] {out_path} ({len(df)} rows) <- {src_name} | "
            f"mean bal_acc={df['bal_acc'].mean():.4f}, roc_auc={df['roc_auc'].mean():.4f}"
        )

    return exported


def export_preds_npzs(
    data_dir: Path,
    metrics_dir: Path,
    handover_dir: Path,
) -> List[Path]:
    """Xuất các file `{model}_{setting}_preds.npz` căn chỉnh 1-1 theo data/processed/metadata.csv."""
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu metadata.csv: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    n_samples = len(meta_df)
    meta_uids = np.asarray(meta_df["trial_uid"].astype(str).tolist(), dtype=str)
    meta_subs = np.asarray(meta_df["subject_id"].astype(str).tolist(), dtype=str)
    meta_labels = meta_df["label"].to_numpy(dtype=np.int64)
    global_trial_idx = np.arange(n_samples, dtype=np.int64)

    if not meta_df["trial_uid"].is_unique:
        raise ValueError("metadata.csv có trial_uid trùng lặp")

    handover_dir.mkdir(parents=True, exist_ok=True)
    exported: List[Path] = []

    for prefix, src_name in PREDS_SPECS:
        src_path = metrics_dir / src_name
        if not src_path.exists():
            raise FileNotFoundError(f"Thiếu file predictions nguồn: {src_path}")

        pred_df = pd.read_csv(src_path)
        if len(pred_df) != n_samples:
            raise ValueError(
                f"{src_name}: kỳ vọng {n_samples} dòng, nhận {len(pred_df)}"
            )
        if not pred_df["trial_uid"].is_unique:
            raise ValueError(f"{src_name}: chứa trial_uid trùng lặp")

        pred_df = pred_df.copy()
        pred_df["trial_uid"] = pred_df["trial_uid"].astype(str)
        pred_df["subject_id"] = pred_df["subject_id"].astype(str)

        # Căn chỉnh đúng thứ tự dòng 0..n_samples-1 của metadata.csv
        aligned = (
            meta_df[["trial_uid"]]
            .assign(trial_uid=meta_df["trial_uid"].astype(str))
            .merge(pred_df, on="trial_uid", how="left", validate="one_to_one")
        )
        if aligned[["fold", "y_true", "y_prob", "y_pred", "subject_id"]].isna().any().any():
            raise ValueError(f"{src_name}: thiếu trial_uid khi khớp với metadata.csv")

        subject_id = np.asarray(aligned["subject_id"].astype(str).tolist(), dtype=str)
        fold = aligned["fold"].to_numpy(dtype=np.int64)
        trial_uid = np.asarray(aligned["trial_uid"].astype(str).tolist(), dtype=str)
        y_true = aligned["y_true"].to_numpy(dtype=np.int64)
        y_prob = aligned["y_prob"].to_numpy(dtype=np.float64)
        y_pred = aligned["y_pred"].to_numpy(dtype=np.int64)

        if not np.array_equal(trial_uid, meta_uids):
            raise ValueError(f"{prefix}: trial_uid sau căn chỉnh không khớp metadata.csv")
        if not np.array_equal(subject_id, meta_subs):
            raise ValueError(f"{prefix}: subject_id không khớp metadata.csv")
        if not np.array_equal(y_true, meta_labels):
            raise ValueError(f"{prefix}: y_true không khớp 100% nhãn trong metadata.csv")
        if not np.isfinite(y_prob).all():
            raise ValueError(f"{prefix}: y_prob chứa NaN/Inf")
        if not ((y_prob >= 0.0) & (y_prob <= 1.0)).all():
            raise ValueError(f"{prefix}: y_prob nằm ngoài [0, 1]")
        if not set(np.unique(y_pred).tolist()).issubset({0, 1}):
            raise ValueError(f"{prefix}: y_pred chứa giá trị ngoài {{0, 1}}")

        out_path = handover_dir / f"{prefix}_preds.npz"
        np.savez_compressed(
            out_path,
            subject_id=subject_id,
            fold=fold,
            trial_idx=global_trial_idx,
            trial_uid=trial_uid,
            y_true=y_true,
            y_prob=y_prob,
            y_pred=y_pred,
        )
        exported.append(out_path)
        size_kb = out_path.stat().st_size / 1024.0
        print(
            f"[Export NPZ] {out_path} ({size_kb:.1f} KB, n={n_samples}) <- {src_name}"
        )

    return exported


def export_split_files(data_dir: Path, splits_out_dir: Path) -> List[Path]:
    """Xuất 3 file splits rời từ data/processed/splits.json hiện có (không đổi seed)."""
    splits_path = data_dir / "splits.json"
    meta_path = data_dir / "metadata.csv"
    if not splits_path.exists():
        raise FileNotFoundError(f"Thiếu splits.json: {splits_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Thiếu metadata.csv: {meta_path}")

    meta_df = pd.read_csv(meta_path)
    expected_sha = _compute_trial_uid_sha256(meta_df["trial_uid"].astype(str).tolist())

    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    if splits["trial_uid_sha256"] != expected_sha:
        raise ValueError(
            "trial_uid_sha256 trong splits.json không khớp metadata.csv: "
            f"{splits['trial_uid_sha256']} != {expected_sha}"
        )
    if int(splits["n_samples"]) != len(meta_df):
        raise ValueError(
            f"n_samples trong splits.json ({splits['n_samples']}) != {len(meta_df)}"
        )

    splits_out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("within_5fold.json", "within_subject", "within_subject"),
        ("cross_5fold.json", "cross_subject_5fold", "cross_subject_5fold"),
        ("loso.json", "cross_subject_loso", "cross_subject_loso"),
    ]

    exported: List[Path] = []
    for fname, protocol_name, key_name in specs:
        payload = {
            "schema_version": splits["schema_version"],
            "protocol": protocol_name,
            "seed": int(splits["seed"]),
            "n_epochs": int(splits["n_samples"]),
            "n_samples": int(splits["n_samples"]),
            "n_subjects": int(splits["n_subjects"]),
            "subjects": splits["subjects"],
            "trial_uid_sha256": splits["trial_uid_sha256"],
            key_name: splits[key_name],
        }
        out_path = splits_out_dir / fname
        out_path.write_text(_format_compact_json(payload), encoding="utf-8")
        exported.append(out_path)
        size_kb = out_path.stat().st_size / 1024.0
        print(
            f"[Export Split] {out_path} ({size_kb:.1f} KB, n_epochs={payload['n_epochs']}, "
            f"sha256={payload['trial_uid_sha256'][:12]}...)"
        )

    return exported


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Xuất file bàn giao chuẩn Đề cương nhóm & 3 file splits rời."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed"),
        help="Thư mục dữ liệu đã tiền xử lý (mặc định: data/processed).",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/metrics/dl2"),
        help="Thư mục chứa metrics_*.json & predictions_*.csv (mặc định: results/metrics/dl2).",
    )
    parser.add_argument(
        "--handover-dir",
        type=Path,
        default=Path("results/handover_standard"),
        help="Thư mục đích xuất handover chuẩn (mặc định: results/handover_standard).",
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("splits"),
        help="Thư mục đích xuất 3 file splits rời (mặc định: splits).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    export_per_fold_csvs(args.metrics_dir, args.handover_dir)
    export_preds_npzs(args.data_dir, args.metrics_dir, args.handover_dir)
    export_split_files(args.data_dir, args.splits_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
