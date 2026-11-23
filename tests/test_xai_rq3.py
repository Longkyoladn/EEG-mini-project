"""Bộ kiểm thử tự động cho Pipeline XAI 4 lớp (RQ3) & 6 biểu đồ chuẩn bài báo."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.evaluation.visualization import generate_all_paper_figures
from src.evaluation.xai_rq3 import (
    EOG_ROI,
    FREQUENCY_BANDS,
    LEFT_MOTOR_ROI,
    MONTAGE_NAME,
    P300_ROI,
    RIGHT_MOTOR_ROI,
    SFREQ,
    TIME_WINDOWS_ORDER,
    apply_frequency_occlusion,
    create_colin27_info,
    load_channel_names,
    verify_ig_completeness,
    verify_rois,
)
from src.training.trainer_dl2 import load_checkpoint_for_eval

DATA_DIR = Path("data/processed")
LOSO_CKPT_DIR = Path("results/checkpoints/dl2/dl2_loso_p300_atcnet")
PAPER_FIG_DIR = Path("results/figures/paper")

EXPECTED_FIGURES: List[str] = [
    "fig1_erp_overview.png",
    "fig2_single_trial_and_rt.png",
    "fig3_benchmark_rq1_rq2.png",
    "fig4_xai_spatiotemporal.png",
    "fig5_xai_eog_and_motor_confound.png",
    "fig6_frequency_and_error_analysis.png",
]


def _assert_valid_300dpi_png(png_path: Path) -> None:
    assert png_path.exists(), f"Thiếu file biểu đồ: {png_path}"
    assert png_path.stat().st_size > 10_000, f"File biểu đồ quá nhỏ / nghi rỗng: {png_path}"

    with Image.open(png_path) as img:
        img.load()
        width, height = img.size
        assert width >= 800 and height >= 600, f"Kích thước ảnh quá nhỏ ({width}x{height}): {png_path}"

        dpi = img.info.get("dpi")
        assert dpi is not None, f"Thiếu metadata DPI trong {png_path}"
        assert math.isclose(float(dpi[0]), 300.0, abs_tol=1.0), f"DPI X != 300 ({dpi[0]}): {png_path}"
        assert math.isclose(float(dpi[1]), 300.0, abs_tol=1.0), f"DPI Y != 300 ({dpi[1]}): {png_path}"

        arr = np.asarray(img.convert("RGB"))
        assert float(arr.std()) > 5.0, f"Ảnh bị đồng màu / rỗng (std={arr.std():.3f}): {png_path}"


def test_ig_completeness_theorem() -> None:
    """Kiểm chứng định lý Completeness của Captum IG trên checkpoint LOSO thực tế."""
    ckpt_path = LOSO_CKPT_DIR / "sub-01_fold0.pt"
    assert ckpt_path.exists(), f"Không tìm thấy checkpoint {ckpt_path}"
    assert (DATA_DIR / "X_clean.npy").exists(), "Thiếu data/processed/X_clean.npy"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, scaler, _ = load_checkpoint_for_eval(ckpt_path, device=device)

    X_clean = np.load(DATA_DIR / "X_clean.npy", mmap_mode="r")
    y_clean = np.load(DATA_DIR / "y.npy")
    sub_ids = np.load(DATA_DIR / "subject_ids.npy").astype(str)

    sub01_tgt = np.where((sub_ids == "sub-01") & (y_clean == 1))[0][:4]
    sub01_freq = np.where((sub_ids != "sub-01") & (y_clean == 0))[0][:64]
    x_sample = scaler.transform(np.asarray(X_clean[sub01_tgt], dtype=np.float32))
    base_freq = np.mean(
        scaler.transform(np.asarray(X_clean[sub01_freq], dtype=np.float32)).astype(np.float64),
        axis=0,
        keepdims=True,
    ).astype(np.float32)
    base_zero = np.zeros((1, 127, 251), dtype=np.float32)

    res_freq = verify_ig_completeness(
        model, x_sample, base_freq, target=0, n_steps=64, internal_batch_size=32, device=device
    )
    res_zero = verify_ig_completeness(
        model, x_sample, base_zero, target=0, n_steps=64, internal_batch_size=32, device=device
    )

    print(
        f"\n[Completeness] freq_base max|err|={res_freq['max_abs_error']:.5f} "
        f"(rel={res_freq['max_rel_error']:.5f}) | "
        f"zero_base max|err|={res_zero['max_abs_error']:.5f} "
        f"(rel={res_zero['max_rel_error']:.5f})"
    )

    assert res_freq["max_abs_error"] < 0.05
    assert res_freq["max_rel_error"] < 0.02
    assert res_freq["captum_max_abs_delta"] < 0.05
    assert res_zero["max_abs_error"] < 0.05
    assert res_zero["max_rel_error"] < 0.02


def test_rois_and_colin27_montage() -> None:
    """Kiểm tra toàn bộ 4 ROI tồn tại trong channel_names.json và khớp 127/127 colin27_1005."""
    ch_names = load_channel_names(DATA_DIR)
    assert len(ch_names) == 127

    roi_indices = verify_rois(ch_names)
    assert len(roi_indices["eog"]) == len(EOG_ROI) == 8
    assert len(roi_indices["p300"]) == len(P300_ROI) == 7
    assert len(roi_indices["left_motor"]) == len(LEFT_MOTOR_ROI) == 6
    assert len(roi_indices["right_motor"]) == len(RIGHT_MOTOR_ROI) == 6

    ch_set = set(ch_names)
    for roi in (EOG_ROI, P300_ROI, LEFT_MOTOR_ROI, RIGHT_MOTOR_ROI):
        for ch in roi:
            assert ch in ch_set, f"Kênh ROI '{ch}' không nằm trong channel_names.json"

    info = create_colin27_info(ch_names, sfreq=SFREQ, montage_name=MONTAGE_NAME)
    assert len(info.ch_names) == 127
    assert info.ch_names == ch_names


@pytest.mark.parametrize("band", list(FREQUENCY_BANDS.keys()))
def test_frequency_occlusion_psd_and_shape(band: str) -> None:
    """Kiểm tra bộ lọc Frequency Occlusion bảo toàn shape, không NaN/Inf, giảm > 90% PSD dải bị che."""
    rng = np.random.default_rng(123)
    t = np.arange(251, dtype=np.float64) / SFREQ
    test_freqs = [0.6, 2.5, 6.0, 10.5, 20.0, 35.0]
    wave = sum(np.sin(2.0 * np.pi * f * t) for f in test_freqs)
    X_syn = (wave[None, None, :] + 0.05 * rng.standard_normal((8, 127, 251))).astype(np.float32)

    X_occ = apply_frequency_occlusion(X_syn, band=band, sfreq=SFREQ, order=4)
    assert X_occ.shape == (8, 127, 251)
    assert X_occ.dtype == np.float32
    assert np.isfinite(X_occ).all()

    low, high = FREQUENCY_BANDS[band]
    if band == "sub_delta":
        # Đo suy giảm đúng thành phần sin 0.6 Hz
        comp = np.sin(2.0 * np.pi * 0.6 * t)
        X_sub = np.broadcast_to(comp[None, None, :], (4, 127, 251)).astype(np.float32)
        Y_sub = apply_frequency_occlusion(X_sub, band="sub_delta", sfreq=SFREQ, order=4)
        p_before = float(np.mean(X_sub.astype(np.float64) ** 2))
        p_after = float(np.mean(Y_sub.astype(np.float64) ** 2))
        reduction = 1.0 - (p_after / p_before)
    else:
        freqs = np.fft.rfftfreq(251, d=1.0 / SFREQ)
        psd_before = (np.abs(np.fft.rfft(X_syn.astype(np.float64), axis=-1)) ** 2).mean(axis=(0, 1))
        psd_after = (np.abs(np.fft.rfft(X_occ.astype(np.float64), axis=-1)) ** 2).mean(axis=(0, 1))
        b_mask = (freqs >= low) & (freqs <= high)
        assert int(np.sum(b_mask)) > 0
        reduction = 1.0 - float(psd_after[b_mask].sum() / psd_before[b_mask].sum())

    print(f"\n[FreqOcc Test] band={band} ({low}-{high} Hz) -> PSD reduction = {reduction * 100:.2f}%")
    assert reduction > 0.90, f"Dải {band} chỉ giảm {reduction * 100:.2f}% (< 90%)"

    # Kiểm tra thêm trên 8 trials thật của X_clean.npy
    if (DATA_DIR / "X_clean.npy").exists():
        X_real = np.asarray(np.load(DATA_DIR / "X_clean.npy", mmap_mode="r")[:8], dtype=np.float32)
        Y_real = apply_frequency_occlusion(X_real, band=band, sfreq=SFREQ, order=4)
        assert Y_real.shape == (8, 127, 251)
        assert Y_real.dtype == np.float32
        assert np.isfinite(Y_real).all()


def _create_mock_xai_artifacts(mock_metrics_dir: Path, data_dir: Path) -> None:
    """Tạo bộ cache/summary tối giản chuẩn schema để test vẽ 6 hình trước khi chạy full."""
    mock_metrics_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    ch_names = load_channel_names(data_dir)
    times = np.load(data_dir / "times.npy").astype(np.float64)
    times_ms = np.round(times * 1000.0, 3)

    handover_src = Path("results/metrics/dl2/handover_per_subject_summary_dl2.csv")
    df_h = pd.read_csv(handover_src)
    df_h.to_csv(mock_metrics_dir / "handover_per_subject_summary_dl2.csv", index=False)
    subs = df_h["subject_id"].astype(str).tolist()

    n_subs, n_ch, n_t = len(subs), 127, 251
    t_gauss = np.exp(-0.5 * ((times_ms - 420.0) / 80.0) ** 2)

    erp_tgt = (rng.normal(0, 1e-6, (n_subs, n_ch, n_t)) + 4e-6 * t_gauss[None, None, :]).astype(np.float32)
    erp_freq = rng.normal(0, 1e-6, (n_subs, n_ch, n_t)).astype(np.float32)
    abs_ig = (np.abs(rng.normal(0.01, 0.005, (n_subs, n_ch, n_t))) + 0.05 * t_gauss[None, None, :]).astype(np.float32)
    sp_w = np.abs(rng.normal(0.05, 0.02, (n_subs, n_ch))).astype(np.float32)

    np.savez(
        mock_metrics_dir / "xai_loso_cache.npz",
        channel_names=np.asarray(ch_names, dtype=str),
        times=times,
        times_ms=times_ms,
        loso_subjects=np.asarray(subs, dtype=str),
        loso_erp_target=erp_tgt,
        loso_erp_freq=erp_freq,
        loso_target_abs_ig=abs_ig,
        loso_spatial_weights_abs=sp_w,
        cs5_clean_per_sub_abs=abs_ig,
        cs5_noica_per_sub_abs_matched=abs_ig * 1.15,
    )

    curve_t = abs_ig.mean(axis=(0, 1))
    curve_f = curve_t * 0.4
    ci = curve_t * 0.1
    summary = {
        "layer1_temporal": {
            "grand_peak_latency_ms": 420.0,
            "p300_to_baseline_ratio": {"mean": 4.2},
            "windows": {
                w: {"energy_share_pct": {"mean": v}}
                for w, v in zip(TIME_WINDOWS_ORDER, [8.0, 22.0, 52.0, 18.0])
            },
            "spatial_weights_vs_ig_300_500": {"pearson_r": 0.65, "spearman_rho": 0.62},
            "temporal_curve_target_abs": {"mean": curve_t.tolist(), "ci95": ci.tolist()},
            "temporal_curve_freq_abs": {"mean": curve_f.tolist(), "ci95": (ci * 0.5).tolist()},
        },
        "layer2_eog": {
            "uniform_baseline_share_pct": 8.0 / 127.0 * 100.0,
            "matched_1to1_subject_level": {"wilcoxon_share_full": {"wilcoxon_p": 0.012}},
        },
        "layer3_motor_and_rt": {
            "roi_300_600": {
                "p300_to_left_motor_ratio": {"mean": 1.65},
                "motor_laterality": {"mean": 0.05},
                "wilcoxon_p300_vs_left_motor": {"wilcoxon_p": 1e-5},
                "wilcoxon_left_vs_right_motor": {"wilcoxon_p": 0.25},
            },
            "rt_stratified": {
                grp: {
                    "n_trials": 800,
                    "rt_ms": {"mean": rt_m},
                    "peak_xai_latency_ms": pk_m,
                    "temporal_curve_mean": curve_t.tolist(),
                    "temporal_curve_ci95": ci.tolist(),
                    "topo_300_500": abs_ig[:, :, 125:176].mean(axis=(0, 2)).tolist(),
                    "roi_300_600": {"p300_to_left_motor_ratio": 1.6, "motor_laterality_mean": 0.04},
                }
                for grp, rt_m, pk_m in (("fast", 410.0, 412.0), ("mid", 500.0, 420.0), ("slow", 630.0, 436.0))
            },
            "rt_vs_performance_correlation": {
                "mean_rt_vs_loso_roc_auc": {"spearman_rho": -0.45, "spearman_p": 0.004},
                "per_subject_rt_perf": [
                    {
                        "subject_id": s,
                        "mean_rt_ms": float(450.0 + i * 5.0),
                        "loso_roc_auc": float(df_h.loc[i, "loso_roc_auc"]),
                    }
                    for i, s in enumerate(subs)
                ],
            },
            "early_window_cutoff": {
                "loso_inference_masking": {
                    "comparison_vs_orig": {
                        m: {"masked_mean": 0.88} for m in ("roc_auc", "pr_auc", "balanced_accuracy")
                    }
                },
                "cs5_comparison": {
                    m: {"orig_mean": 0.94, "inference_masked_mean": 0.85, "retrain_early_mean": 0.90}
                    for m in ("roc_auc", "pr_auc", "balanced_accuracy")
                },
            },
        },
        "layer4_frequency_and_neuro": {
            "frequency_occlusion": {
                b: {
                    "delta_vs_orig": {
                        "roc_auc": {"delta_mean": dm, "delta_std": 0.02, "wilcoxon_p": 0.001},
                        "pr_auc": {"delta_mean": dm * 1.4, "delta_std": 0.03, "wilcoxon_p": 0.001},
                    }
                }
                for b, dm in zip(
                    ("sub_delta", "delta", "theta", "alpha", "beta", "gamma", "delta_theta"),
                    (-0.02, -0.09, -0.05, -0.01, -0.01, -0.002, -0.18),
                )
            },
            "spatial_correlation": {
                "win_300_500_abs_ig_vs_abs_erp_diff": {"pearson_r": 0.72, "spearman_rho": 0.70},
                "win_300_500_abs_ig_vs_erp_diff": {"pearson_r": 0.68, "spearman_rho": 0.66},
            },
            "error_analysis": {
                "n_tp": 2580,
                "n_fn": 270,
                "tp_pz_p300_amp_uv": {"mean": 4.5},
                "fn_pz_p300_amp_uv": {"mean": 1.8},
                "pz_p300_amp_mannwhitney_p": 1e-6,
                "curves": {
                    "tp_pz_mean_uv": (4.5 * t_gauss).tolist(),
                    "tp_pz_ci95_uv": (0.3 * np.ones_like(t_gauss)).tolist(),
                    "fn_pz_mean_uv": (1.8 * t_gauss).tolist(),
                    "fn_pz_ci95_uv": (0.5 * np.ones_like(t_gauss)).tolist(),
                },
            },
        },
    }
    (mock_metrics_dir / "xai_rq3_summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_paper_figures_300dpi(tmp_path: Path) -> None:
    """Xác nhận cả 6 biểu đồ PNG 300 DPI + PDF được tạo đủ, hợp lệ và không rỗng."""
    all_real_exist = all((PAPER_FIG_DIR / fname).exists() for fname in EXPECTED_FIGURES)
    if all_real_exist:
        for fname in EXPECTED_FIGURES:
            png_path = PAPER_FIG_DIR / fname
            pdf_path = png_path.with_suffix(".pdf")
            _assert_valid_300dpi_png(png_path)
            assert pdf_path.exists() and pdf_path.stat().st_size > 1_000, f"Thiếu/rỗng PDF: {pdf_path}"
        return

    # Giai đoạn unit test trước khi chạy full: kiểm thử pipeline vẽ hình vào tmp_path
    mock_metrics = tmp_path / "metrics"
    mock_figs = tmp_path / "figures"
    _create_mock_xai_artifacts(mock_metrics, DATA_DIR)
    outputs = generate_all_paper_figures(
        data_dir=DATA_DIR,
        metrics_dir=mock_metrics,
        figures_dir=mock_figs,
    )
    assert len(outputs) == 6
    for fname in EXPECTED_FIGURES:
        png_path = mock_figs / fname
        pdf_path = png_path.with_suffix(".pdf")
        _assert_valid_300dpi_png(png_path)
        assert pdf_path.exists() and pdf_path.stat().st_size > 1_000


def test_step68_handover_splits_latency_and_fig3b() -> None:
    """Kiểm tra chuẩn bàn giao Đề cương Mục 3.5/3.6, 3 file splits rời & fig3b_roc_pr_curves."""
    handover_dir = Path("results/handover_standard")
    splits_dir = Path("splits")
    latency_path = Path("results/metrics/dl2/inference_latency.json")
    meta_df = pd.read_csv(DATA_DIR / "metadata.csv")
    splits_master = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))

    expected_cols = [
        "subject_id",
        "fold",
        "bal_acc",
        "roc_auc",
        "pr_auc",
        "f1_target",
        "sensitivity",
        "specificity",
    ]
    csv_specs = {
        "p300_atcnet_within_per_fold.csv": 195,
        "p300_atcnet_loso_per_fold.csv": 39,
        "p300_atcnet_cs5_clean_per_fold.csv": 5,
        "p300_atcnet_cs5_no_ica_per_fold.csv": 5,
    }
    for fname, n_rows in csv_specs.items():
        p = handover_dir / fname
        assert p.exists(), f"Thiếu file per-fold chuẩn: {p}"
        df = pd.read_csv(p)
        assert list(df.columns) == expected_cols
        assert len(df) == n_rows
        assert not df.isna().any().any()
        for col in expected_cols[2:]:
            vals = df[col].to_numpy(dtype=np.float64)
            assert np.isfinite(vals).all()
            assert ((vals >= 0.0) & (vals <= 1.0)).all()

    npz_files = [
        "p300_atcnet_within_preds.npz",
        "p300_atcnet_loso_preds.npz",
        "p300_atcnet_cs5_clean_preds.npz",
    ]
    req_arrays = {"subject_id", "fold", "trial_idx", "trial_uid", "y_true", "y_prob", "y_pred"}
    meta_labels = meta_df["label"].to_numpy(dtype=np.int64)
    meta_uids = np.asarray(meta_df["trial_uid"].astype(str).tolist(), dtype=str)
    expected_idx = np.arange(len(meta_df), dtype=np.int64)
    for fname in npz_files:
        p = handover_dir / fname
        assert p.exists(), f"Thiếu file preds NPZ chuẩn: {p}"
        with np.load(p, allow_pickle=False) as z:
            assert req_arrays.issubset(set(z.files))
            assert np.array_equal(z["trial_idx"], expected_idx)
            assert np.array_equal(z["trial_uid"], meta_uids)
            assert np.array_equal(z["y_true"], meta_labels)
            y_prob = z["y_prob"]
            assert np.isfinite(y_prob).all()
            assert ((y_prob >= 0.0) & (y_prob <= 1.0)).all()
            assert set(np.unique(z["y_pred"]).tolist()).issubset({0, 1})

    for split_name, key in (
        ("within_5fold.json", "within_subject"),
        ("cross_5fold.json", "cross_subject_5fold"),
        ("loso.json", "cross_subject_loso"),
    ):
        sp_path = splits_dir / split_name
        assert sp_path.exists(), f"Thiếu split rời: {sp_path}"
        sp = json.loads(sp_path.read_text(encoding="utf-8"))
        assert sp["n_epochs"] == len(meta_df)
        assert sp["trial_uid_sha256"] == splits_master["trial_uid_sha256"]
        assert key in sp

    assert latency_path.exists(), f"Thiếu file benchmark latency: {latency_path}"
    lat = json.loads(latency_path.read_text(encoding="utf-8"))
    pairs = {(b["model_name"], b["device"]): b for b in lat["benchmarks"]}
    assert ("P300ATCNet (attn_pool)", "cpu") in pairs
    if torch.cuda.is_available():
        assert ("P300ATCNet (attn_pool)", "cuda") in pairs
    for b in lat["benchmarks"]:
        assert b["mean_ms"] > 0.0 and b["median_ms"] > 0.0 and b["p95_ms"] > 0.0

    fig3b_png = PAPER_FIG_DIR / "fig3b_roc_pr_curves.png"
    fig3b_pdf = PAPER_FIG_DIR / "fig3b_roc_pr_curves.pdf"
    _assert_valid_300dpi_png(fig3b_png)
    assert fig3b_pdf.exists() and fig3b_pdf.stat().st_size > 1_000
