"""CLI huấn luyện & đánh giá ML Nhánh 2 (SVM, KNN, GaussianNB) trên đặc trưng ERP.

Ví dụ:
    python scripts/train_ml2.py
    python scripts/train_ml2.py --models svm knn gnb --protocols within loso --run-ablation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.erp_time_features import VALID_FEATURE_SETS
from src.ml2.ml2_erp import (
    VALID_ML2_MODELS,
    VALID_PROTOCOLS,
    extract_features_for_ml2,
    format_summary_row,
    run_ml2_experiment,
    save_comparison_table,
)


def _print_result_line(row: Dict[str, Any]) -> None:
    print(
        f"  {row['exp_name']:<38} | "
        f"ROC={row['roc_auc_mean']:.4f}±{row['roc_auc_std']:.4f} | "
        f"PR={row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f} | "
        f"bAcc={row['bal_acc_mean']:.4f}±{row['bal_acc_std']:.4f} | "
        f"F1={row['f1_target_mean']:.4f}±{row['f1_target_std']:.4f} | "
        f"Sens={row['sensitivity_mean']:.4f} | "
        f"Spec={row['specificity_mean']:.4f} | "
        f"{row['wall_time_sec']:.1f}s"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Huấn luyện các mô hình ML2 (SVM, KNN, GaussianNB) trên đặc trưng ERP."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/ml2"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["svm", "knn", "gnb"],
        choices=list(VALID_ML2_MODELS),
        help="Danh sách mô hình ML2 cần chạy (mặc định: svm knn gnb).",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["within", "loso"],
        choices=list(VALID_PROTOCOLS),
        help="Giao thức đánh giá (mặc định: within loso).",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="erp_full_binned",
        choices=list(VALID_FEATURE_SETS),
        help="Bộ đặc trưng chính (mặc định: erp_full_binned).",
    )
    parser.add_argument("--channel-preset", type=str, default="core_14")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Chạy thêm ablation SVM trên erp_peaks_only và erp_windows_and_peaks.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    total_t0 = time.perf_counter()

    print(
        f"[ML2] Trích xuất đặc trưng chính: feature_set={args.feature_set}, "
        f"channel_preset={args.channel_preset}..."
    )
    X_feat, y, meta_df, _ = extract_features_for_ml2(
        args.data_dir,
        feature_set=args.feature_set,
        channel_preset=args.channel_preset,
        chunk_size=args.chunk_size,
    )
    print(f"[ML2] X_feat={X_feat.shape} ({X_feat.dtype}), n_jobs={args.n_jobs}")

    main_rows: List[Dict[str, Any]] = []
    print("\n=== KẾT QUẢ SO SÁNH 3 MÔ HÌNH ML2 (Mean ± SD) ===")
    for proto in args.protocols:
        for model_name in args.models:
            exp_name = f"ml2_erp_{model_name}_{proto}"
            res = run_ml2_experiment(
                model_name=model_name,
                protocol_alias=proto,
                X_feat=X_feat,
                y=y,
                meta_df=meta_df,
                splits_dir=args.splits_dir,
                output_dir=args.output_dir,
                feature_set=args.feature_set,
                channel_preset=args.channel_preset,
                exp_name=exp_name,
                n_jobs=args.n_jobs,
                seed=args.seed,
            )
            row = format_summary_row(res)
            main_rows.append(row)
            _print_result_line(row)

    csv_main, json_main = save_comparison_table(
        main_rows, args.output_dir, "ml2_comparison_summary"
    )
    print(f"\n[Saved] {csv_main}")
    print(f"[Saved] {json_main}")

    if args.run_ablation:
        ablation_rows: List[Dict[str, Any]] = [
            r for r in main_rows if r["model"] == "svm"
        ]
        print("\n=== ABLATION 3 BỘ ĐẶC TRƯNG ERP TRÊN SVM ===")
        for fset in ("erp_peaks_only", "erp_windows_and_peaks", "erp_full_binned"):
            if fset == args.feature_set:
                continue
            X_ab, y_ab, meta_ab, _ = extract_features_for_ml2(
                args.data_dir,
                feature_set=fset,
                channel_preset=args.channel_preset,
                chunk_size=args.chunk_size,
            )
            for proto in args.protocols:
                exp_name = f"ml2_ablation_{fset}_svm_{proto}"
                res = run_ml2_experiment(
                    model_name="svm",
                    protocol_alias=proto,
                    X_feat=X_ab,
                    y=y_ab,
                    meta_df=meta_ab,
                    splits_dir=args.splits_dir,
                    output_dir=args.output_dir,
                    feature_set=fset,
                    channel_preset=args.channel_preset,
                    exp_name=exp_name,
                    n_jobs=args.n_jobs,
                    seed=args.seed,
                )
                row = format_summary_row(res)
                ablation_rows.append(row)
                _print_result_line(row)

        csv_ab, json_ab = save_comparison_table(
            ablation_rows, args.output_dir, "ml2_ablation_summary"
        )
        print(f"\n[Saved] {csv_ab}")
        print(f"[Saved] {json_ab}")

    total_sec = time.perf_counter() - total_t0
    print(f"\n[ML2] Hoàn tất toàn bộ trong {total_sec:.1f}s ({total_sec/60.0:.2f} phút).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
