"""CLI chạy trích xuất thử và chẩn đoán đặc trưng miền thời gian ERP (Giai đoạn 2).

Ví dụ:
    python scripts/extract_erp_features.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.erp_time_features import (
    VALID_FEATURE_SETS,
    ERPFeatureExtractor,
    load_data_geometry,
)


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    return obj


def _benchmark_single_trial(
    extractor: ERPFeatureExtractor,
    sample_epoch: np.ndarray,
    *,
    warmup: int,
    runs: int,
) -> Dict[str, float]:
    """Đo độ trễ trích xuất đặc trưng single-trial (batch_size=1) trên CPU."""
    for _ in range(warmup):
        _ = extractor.transform(sample_epoch)

    timings_ms: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = extractor.transform(sample_epoch)
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)

    arr = np.asarray(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr, ddof=1)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
    }


def _compute_feature_stats(
    X_feat: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Tính trung bình, SD, Cohen's d và single-feature ROC-AUC cho từng đặc trưng."""
    tgt_mask = y == 1
    freq_mask = y == 0
    n_t = int(tgt_mask.sum())
    n_f = int(freq_mask.sum())

    rows: List[Dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        col = X_feat[:, j]
        t_vals = col[tgt_mask]
        f_vals = col[freq_mask]

        t_mean = float(np.mean(t_vals))
        t_std = float(np.std(t_vals, ddof=1))
        f_mean = float(np.mean(f_vals))
        f_std = float(np.std(f_vals, ddof=1))

        pooled_var = ((n_t - 1) * (t_std ** 2) + (n_f - 1) * (f_std ** 2)) / (
            n_t + n_f - 2
        )
        pooled_sd = math.sqrt(max(pooled_var, 0.0))
        cohens_d = (t_mean - f_mean) / pooled_sd if pooled_sd > 0.0 else 0.0

        roc_signed = float(roc_auc_score(y, col))
        roc_abs = max(roc_signed, 1.0 - roc_signed)
        rows.append(
            {
                "feature": name,
                "target_mean": t_mean,
                "target_std": t_std,
                "freq_mean": f_mean,
                "freq_std": f_std,
                "cohens_d": float(cohens_d),
                "abs_cohens_d": float(abs(cohens_d)),
                "roc_auc_signed": roc_signed,
                "roc_auc_abs": roc_abs,
            }
        )

    df = pd.DataFrame(rows).sort_values("abs_cohens_d", ascending=False).reset_index(
        drop=True
    )
    return df


def _summarize_latency_series(vals: np.ndarray) -> Dict[str, float]:
    q25, q50, q75 = np.percentile(vals, [25.0, 50.0, 75.0])
    return {
        "n": int(vals.size),
        "mean_ms": float(np.mean(vals)),
        "std_ms": float(np.std(vals, ddof=1)),
        "median_ms": float(q50),
        "q25_ms": float(q25),
        "q75_ms": float(q75),
    }


def run_diagnostics(
    data_dir: Path,
    output_dir: Path,
    *,
    channel_preset: str = "core_14",
    chunk_size: int = 2048,
    warmup: int = 50,
    runs: int = 500,
) -> Dict[str, Any]:
    """Chạy benchmark 3 feature sets và xuất chẩn đoán nhiễu / lệch pha ERP."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    channel_names, times = load_data_geometry(data_dir)

    sample_epoch = np.array(X_clean[0:1], dtype=np.float32, copy=True)

    benchmarks: Dict[str, Any] = {}
    diag_feat_matrix: Optional[np.ndarray] = None
    diag_feat_names: Optional[List[str]] = None

    for fset in VALID_FEATURE_SETS:
        extractor = ERPFeatureExtractor(
            channel_names=channel_names,
            times=times,
            channel_preset=channel_preset,
            feature_set=fset,
        )
        t0 = time.perf_counter()
        X_feat = extractor.transform_chunked(X_clean, chunk_size=chunk_size)
        elapsed_sec = time.perf_counter() - t0

        single_latency = _benchmark_single_trial(
            extractor, sample_epoch, warmup=warmup, runs=runs
        )
        benchmarks[fset] = {
            "n_features": int(X_feat.shape[1]),
            "n_epochs": int(X_feat.shape[0]),
            "dtype": str(X_feat.dtype),
            "all_finite": bool(np.isfinite(X_feat).all()),
            "full_extraction_sec": float(elapsed_sec),
            "single_trial_latency": single_latency,
        }
        if fset == "erp_windows_and_peaks":
            diag_feat_matrix = X_feat
            diag_feat_names = extractor.get_feature_names()

    assert diag_feat_matrix is not None and diag_feat_names is not None

    stats_df = _compute_feature_stats(diag_feat_matrix, y, diag_feat_names)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats_csv_path = output_dir / "erp_feature_stats.csv"
    stats_df.to_csv(stats_csv_path, index=False)

    # Chẩn đoán độ trễ & jitter tại kênh Pz
    pz_idx = channel_names.index("Pz")
    pz_uv = np.asarray(X_clean[:, pz_idx, :], dtype=np.float64) * 1e6
    diff_pz_uv = pz_uv[y == 1].mean(axis=0) - pz_uv[y == 0].mean(axis=0)
    p300_mask = (times >= 0.300 - 1e-9) & (times <= 0.600 + 1e-9)
    ga_rel_pk = int(np.argmax(diff_pz_uv[p300_mask]))
    ga_peak_amp_uv = float(diff_pz_uv[p300_mask][ga_rel_pk])
    ga_peak_lat_ms = float(times[p300_mask][ga_rel_pk] * 1000.0)

    name_to_col = {n: i for i, n in enumerate(diag_feat_names)}
    pz_lat_col = diag_feat_matrix[:, name_to_col["Pz__P300_peak_lat_ms"]]
    tgt_mask = y == 1
    freq_mask = y == 0

    tgt_lats = pz_lat_col[tgt_mask]
    freq_lats = pz_lat_col[freq_mask]
    tgt_rts = meta_df.loc[tgt_mask, "rt_ms"].to_numpy(dtype=np.float64)

    sp_r, sp_p = stats.spearmanr(tgt_lats, tgt_rts)
    pe_r, pe_p = stats.pearsonr(tgt_lats, tgt_rts)

    stat_lookup = stats_df.set_index("feature").to_dict(orient="index")
    pz_comparison = {
        fname: {
            "cohens_d": float(stat_lookup[fname]["cohens_d"]),
            "roc_auc_abs": float(stat_lookup[fname]["roc_auc_abs"]),
            "target_mean": float(stat_lookup[fname]["target_mean"]),
            "target_std": float(stat_lookup[fname]["target_std"]),
            "freq_mean": float(stat_lookup[fname]["freq_mean"]),
            "freq_std": float(stat_lookup[fname]["freq_std"]),
        }
        for fname in (
            "Pz__P300_peak_amp_uv",
            "Pz__P300_early_mean_uv",
            "Pz__P300_late_mean_uv",
            "Pz__auc_P300_uv_s",
            "Pz__P300_peak_lat_ms",
        )
    }

    payload: Dict[str, Any] = {
        "channel_preset": channel_preset,
        "chunk_size": chunk_size,
        "benchmarks": benchmarks,
        "top_15_features": stats_df.head(15).to_dict(orient="records"),
        "pz_latency_jitter_diagnostics": {
            "grand_average_diff_peak_amp_uv": ga_peak_amp_uv,
            "grand_average_diff_peak_lat_ms": ga_peak_lat_ms,
            "single_trial_target_p300_lat": _summarize_latency_series(tgt_lats),
            "single_trial_freq_p300_lat": _summarize_latency_series(freq_lats),
            "target_rt_ms": _summarize_latency_series(tgt_rts),
            "corr_target_p300_lat_vs_rt_ms": {
                "spearman_r": float(sp_r),
                "spearman_p": float(sp_p),
                "pearson_r": float(pe_r),
                "pearson_p": float(pe_p),
            },
            "pz_peak_vs_window_comparison": pz_comparison,
        },
    }

    diag_json_path = output_dir / "erp_feature_diagnostics.json"
    with open(diag_json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, indent=2, allow_nan=False)

    print("\n=== BENCHMARK TRÍCH XUẤT ĐẶC TRƯNG ERP (20,886 epochs, float64, µV) ===")
    for fset, info in benchmarks.items():
        lat = info["single_trial_latency"]
        print(
            f"  - {fset:<22} | dims={info['n_features']:>3} | "
            f"full={info['full_extraction_sec']:.3f}s | "
            f"single-trial={lat['mean_ms']:.3f}±{lat['std_ms']:.3f} ms (median={lat['median_ms']:.3f}, p95={lat['p95_ms']:.3f})"
        )

    print("\n=== TOP 10 ĐẶC TRƯNG PHÂN BIỆT MẠNH NHẤT (erp_windows_and_peaks) ===")
    for idx, row in stats_df.head(10).iterrows():
        print(
            f"  {idx+1:>2}. {row['feature']:<28} | d={row['cohens_d']:+.4f} | "
            f"AUC={row['roc_auc_abs']:.4f} | Tgt={row['target_mean']:+.2f}±{row['target_std']:.2f} vs "
            f"Freq={row['freq_mean']:+.2f}±{row['freq_std']:.2f}"
        )

    print("\n=== CHẨN ĐOÁN LATENCY JITTER TẠI KÊNH Pz ===")
    print(
        f"  - Grand Average (Target - Freq) P300 peak: {ga_peak_amp_uv:+.2f} µV @ {ga_peak_lat_ms:.1f} ms"
    )
    print(
        f"  - Single-trial Target P300 peak latency: {tgt_lats.mean():.1f} ± {tgt_lats.std(ddof=1):.1f} ms "
        f"(median={np.median(tgt_lats):.1f}, IQR=[{np.percentile(tgt_lats,25):.1f}, {np.percentile(tgt_lats,75):.1f}])"
    )
    print(
        f"  - Single-trial Freq   P300 peak latency: {freq_lats.mean():.1f} ± {freq_lats.std(ddof=1):.1f} ms"
    )
    print(
        f"  - Corr(Pz__P300_peak_lat_ms, rt_ms) [N={tgt_lats.size}]: "
        f"Spearman r={sp_r:+.4f} (p={sp_p:.2e}), Pearson r={pe_r:+.4f} (p={pe_p:.2e})"
    )
    for fname, vals in pz_comparison.items():
        print(
            f"  - {fname:<24} | d={vals['cohens_d']:+.4f} | single-feat AUC={vals['roc_auc_abs']:.4f}"
        )
    print(f"\n[Saved] {diag_json_path}")
    print(f"[Saved] {stats_csv_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trích xuất thử và chẩn đoán đặc trưng miền thời gian ERP."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/metrics/ml2"))
    parser.add_argument("--channel-preset", type=str, default="core_14")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_diagnostics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        channel_preset=args.channel_preset,
        chunk_size=args.chunk_size,
        warmup=args.warmup,
        runs=args.runs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
