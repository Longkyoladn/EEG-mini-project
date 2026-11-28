"""Đo chi phí tính toán (Train Time & Single-Trial Inference Latency) cho ML Nhánh 2.

Chạy độc lập, không cần train lại toàn bộ 195/39 folds:
    python scripts/benchmark_latency_ml2.py
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.metrics import _sanitize_json_value
from src.features.erp_time_features import ERPFeatureExtractor, load_data_geometry
from src.ml2.ml2_erp import _apply_platt, _fit_platt_calibrator, load_protocol_folds


def _summarize_timings(timings_ms: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(timings_ms, dtype=np.float64)
    q25, q75 = np.percentile(arr, [25.0, 75.0])
    mean_ms = float(np.mean(arr))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(arr, ddof=1)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95.0)),
        "p99_ms": float(np.percentile(arr, 99.0)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "iqr_ms": float(q75 - q25),
        "throughput_epochs_per_sec": float(1000.0 / mean_ms) if mean_ms > 0 else 0.0,
    }


def _benchmark_callable(
    fn: Callable[[], Any],
    *,
    n_warmup: int,
    n_runs: int,
) -> Dict[str, float]:
    for _ in range(n_warmup):
        _ = fn()

    timings_ms: List[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _ = fn()
        t1 = time.perf_counter()
        timings_ms.append((t1 - t0) * 1000.0)
    return _summarize_timings(timings_ms)


def _load_train_time_summary(metrics_dir: Path) -> List[Dict[str, Any]]:
    specs = [
        "metrics_ml2_erp_svm_within.json",
        "metrics_ml2_erp_knn_within.json",
        "metrics_ml2_erp_gnb_within.json",
        "metrics_ml2_erp_svm_loso.json",
        "metrics_ml2_erp_knn_loso.json",
        "metrics_ml2_erp_gnb_loso.json",
    ]
    rows: List[Dict[str, Any]] = []
    for fname in specs:
        fpath = metrics_dir / fname
        if not fpath.exists():
            continue
        payload = json.loads(fpath.read_text(encoding="utf-8"))
        cfg = payload["config"]
        pf = payload["per_fold"]
        fold_times = [float(r["train_time_sec"]) for r in pf]
        rows.append(
            {
                "exp_name": cfg["exp_name"],
                "model": cfg["model"],
                "protocol": cfg["protocol"],
                "n_features": int(cfg["n_features"]),
                "n_folds": int(cfg["n_folds"]),
                "wall_time_sec": float(cfg["wall_time_sec"]),
                "wall_time_min": float(cfg["wall_time_sec"]) / 60.0,
                "sum_fold_train_time_sec": float(cfg["sum_fold_train_time_sec"]),
                "mean_fold_train_time_sec": float(np.mean(fold_times)),
                "std_fold_train_time_sec": float(np.std(fold_times, ddof=1)) if len(fold_times) > 1 else 0.0,
            }
        )
    return rows


def _build_representative_models(seed: int = 42) -> List[Tuple[str, str, Callable[[], Any]]]:
    return [
        (
            "ml2_erp_svm (Linear C=0.01)",
            "svm_linear",
            lambda: LinearSVC(
                C=0.01,
                class_weight="balanced",
                dual="auto",
                max_iter=5000,
                random_state=seed,
            ),
        ),
        (
            "ml2_erp_svm (RBF C=1.0, scale)",
            "svm_rbf",
            lambda: SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=False,
                cache_size=1000,
                tol=1e-3,
                random_state=seed,
            ),
        ),
        (
            "ml2_erp_knn (k=35, manhattan)",
            "knn",
            lambda: KNeighborsClassifier(
                n_neighbors=35,
                weights="distance",
                metric="manhattan",
            ),
        ),
        (
            "ml2_erp_gnb (var_smooth=0.1)",
            "gnb",
            lambda: GaussianNB(var_smoothing=0.1),
        ),
    ]


def run_latency_benchmark(
    data_dir: Path = Path("data/processed"),
    splits_dir: Path = Path("splits"),
    metrics_dir: Path = Path("results/metrics/ml2"),
    output_path: Path = Path("results/metrics/ml2/inference_latency_ml2.json"),
    *,
    feature_set: str = "erp_full_binned",
    channel_preset: str = "core_14",
    n_warmup: int = 50,
    n_runs: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Đo độ trễ suy luận single-trial (batch_size=1) và tổng hợp Train Time cho ML2."""
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    y = np.load(data_dir / "y.npy")
    channel_names, times = load_data_geometry(data_dir)

    extractor = ERPFeatureExtractor(
        channel_names=channel_names,
        times=times,
        channel_preset=channel_preset,
        feature_set=feature_set,
    )

    # Nạp đúng 1 mẫu thật vào RAM để đo thuần CPU (không dính I/O mmap)
    sample_raw = np.array(X_clean[0:1], dtype=np.float32, copy=True)
    fe_stats = _benchmark_callable(
        lambda: extractor.transform(sample_raw),
        n_warmup=n_warmup,
        n_runs=n_runs,
    )
    sample_feat = extractor.transform(sample_raw)

    # Chỉ trích xuất tập con các chỉ số cần dùng cho fold 0 của within và loso
    fold_specs: List[Tuple[str, Dict[str, Any]]] = []
    for proto_alias in ("within", "loso"):
        _, tasks, _ = load_protocol_folds(splits_dir, proto_alias)
        fold_specs.append((proto_alias, tasks[0]))

    needed_idx = np.unique(
        np.concatenate(
            [
                np.concatenate([f["train_idx"], f["val_idx"]])
                for _, f in fold_specs
            ]
        )
    )
    idx_map = {int(orig_i): pos for pos, orig_i in enumerate(needed_idx.tolist())}
    X_needed_feat = extractor.transform_chunked(X_clean[needed_idx], chunk_size=2048)

    benchmarks: List[Dict[str, Any]] = []
    for proto_alias, fold0 in fold_specs:
        tr_pos = np.asarray([idx_map[int(i)] for i in fold0["train_idx"]], dtype=np.int64)
        va_pos = np.asarray([idx_map[int(i)] for i in fold0["val_idx"]], dtype=np.int64)

        X_tr = X_needed_feat[tr_pos]
        y_tr = y[fold0["train_idx"]]
        X_va = X_needed_feat[va_pos]
        y_va = y[fold0["val_idx"]]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        for display_name, kind, factory in _build_representative_models(seed=seed):
            clf = factory()
            clf.fit(X_tr_sc, y_tr)

            if kind.startswith("svm"):
                val_scores = np.asarray(
                    clf.decision_function(X_va_sc), dtype=np.float64
                ).ravel()
                calibrator = _fit_platt_calibrator(val_scores, y_va, seed=seed)

                def _predict_clf_only() -> np.ndarray:
                    x_sc = scaler.transform(sample_feat)
                    sc = clf.decision_function(x_sc)
                    return _apply_platt(calibrator, sc)

                def _predict_end_to_end() -> np.ndarray:
                    f = extractor.transform(sample_raw)
                    x_sc = scaler.transform(f)
                    sc = clf.decision_function(x_sc)
                    return _apply_platt(calibrator, sc)

            else:

                def _predict_clf_only() -> np.ndarray:
                    x_sc = scaler.transform(sample_feat)
                    return clf.predict_proba(x_sc)[:, 1]

                def _predict_end_to_end() -> np.ndarray:
                    f = extractor.transform(sample_raw)
                    x_sc = scaler.transform(f)
                    return clf.predict_proba(x_sc)[:, 1]

            # Kiểm tra đầu ra hợp lệ trước khi benchmark
            prob_check = _predict_end_to_end()
            if prob_check.shape != (1,) or not np.isfinite(prob_check).all():
                raise RuntimeError(f"Đầu ra bất thường ở {display_name}: {prob_check}")

            clf_stats = _benchmark_callable(
                _predict_clf_only, n_warmup=n_warmup, n_runs=n_runs
            )
            e2e_stats = _benchmark_callable(
                _predict_end_to_end, n_warmup=n_warmup, n_runs=n_runs
            )

            n_sv = int(clf.support_.size) if hasattr(clf, "support_") else None
            benchmarks.append(
                {
                    "model_name": display_name,
                    "model_kind": kind,
                    "protocol_fit": proto_alias,
                    "fold_fit": int(fold0["fold"]),
                    "subject_test_fold0": str(fold0.get("test_subjects", ["sub-01"])[0]),
                    "n_train_samples": int(len(y_tr)),
                    "n_support_vectors": n_sv,
                    "device": "cpu",
                    "batch_size": 1,
                    "input_shape": [1, len(channel_names), int(times.size)],
                    "feature_set": feature_set,
                    "n_features": int(extractor.n_features),
                    "n_warmup": n_warmup,
                    "n_runs": n_runs,
                    **e2e_stats,
                    "classifier_only": clf_stats,
                    "feature_extraction_only": fe_stats,
                }
            )

    train_time_summary = _load_train_time_summary(metrics_dir)

    dl2_ref: Optional[Dict[str, Any]] = None
    dl2_latency_path = Path("results/metrics/dl2/inference_latency.json")
    if dl2_latency_path.exists():
        dl2_raw = json.loads(dl2_latency_path.read_text(encoding="utf-8"))
        for entry in dl2_raw.get("benchmarks", []):
            if entry.get("device") == "cpu":
                dl2_ref = {
                    "model_name": entry.get("model_name"),
                    "device": "cpu",
                    "mean_ms": float(entry["mean_ms"]),
                    "std_ms": float(entry["std_ms"]),
                    "median_ms": float(entry["median_ms"]),
                    "p95_ms": float(entry["p95_ms"]),
                    "throughput_epochs_per_sec": float(entry["throughput_epochs_per_sec"]),
                }
                break

    payload: Dict[str, Any] = {
        "protocol": "single_trial_inference_latency_end_to_end",
        "batch_size": 1,
        "input_shape": [1, len(channel_names), int(times.size)],
        "feature_set": feature_set,
        "channel_preset": channel_preset,
        "n_features": int(extractor.n_features),
        "n_warmup": n_warmup,
        "n_runs": n_runs,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cpu_name": platform.processor() or platform.machine(),
        "python_version": platform.python_version(),
        "feature_extraction_latency": fe_stats,
        "benchmarks": benchmarks,
        "train_time_summary": train_time_summary,
        "dl2_cpu_reference": dl2_ref,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json_value(payload), f, indent=2, allow_nan=False)

    print(
        "\n=== ML2 SINGLE-TRIAL INFERENCE LATENCY (CPU, batch_size=1, shape=(1, 127, 251)) ==="
    )
    print(
        f"  [Feature Extraction Only: {feature_set}, {extractor.n_features} dims] "
        f"{fe_stats['mean_ms']:.3f} ± {fe_stats['std_ms']:.3f} ms/epoch "
        f"(median={fe_stats['median_ms']:.3f}, p95={fe_stats['p95_ms']:.3f})"
    )
    header = (
        f"  {'Model':<30} | {'Fit':<6} | {'N_tr':>6} | {'E2E Mean±SD (ms)':>18} | "
        f"{'Median':>8} | {'p95':>8} | {'Clf-only':>9} | {'Thr (ep/s)':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in benchmarks:
        mean_sd = f"{r['mean_ms']:.3f} ± {r['std_ms']:.3f}"
        clf_m = r["classifier_only"]["mean_ms"]
        print(
            f"  {r['model_name']:<30} | {r['protocol_fit']:<6} | {r['n_train_samples']:>6} | "
            f"{mean_sd:>18} | {r['median_ms']:>8.3f} | {r['p95_ms']:>8.3f} | "
            f"{clf_m:>7.3f}ms | {r['throughput_epochs_per_sec']:>10.1f}"
        )
    if dl2_ref is not None:
        print(
            f"  [DL2 Ref CPU] {dl2_ref['model_name']}: "
            f"{dl2_ref['mean_ms']:.3f} ± {dl2_ref['std_ms']:.3f} ms/epoch "
            f"({dl2_ref['throughput_epochs_per_sec']:.1f} ep/s)"
        )

    print("\n=== TỔNG HỢP THỜI GIAN HUẤN LUYỆN (Giai đoạn 3) ===")
    for t in train_time_summary:
        print(
            f"  - {t['exp_name']:<24} | folds={t['n_folds']:>3} | "
            f"wall={t['wall_time_sec']:>7.2f}s ({t['wall_time_min']:.2f}m) | "
            f"fold_avg={t['mean_fold_train_time_sec']:.2f}s"
        )
    print(f"\n[Saved] {output_path}")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Đo Train Time & Single-Trial Latency (ms/epoch) cho ML Nhánh 2."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics/ml2"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/metrics/ml2/inference_latency_ml2.json"),
    )
    parser.add_argument("--feature-set", type=str, default="erp_full_binned")
    parser.add_argument("--channel-preset", type=str, default="core_14")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_latency_benchmark(
        data_dir=args.data_dir,
        splits_dir=args.splits_dir,
        metrics_dir=args.metrics_dir,
        output_path=args.output,
        feature_set=args.feature_set,
        channel_preset=args.channel_preset,
        n_warmup=args.warmup,
        n_runs=args.runs,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
