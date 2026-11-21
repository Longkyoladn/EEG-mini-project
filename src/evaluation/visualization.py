"""Bộ hàm trực quan hóa chuẩn bài báo khoa học (300 DPI PNG + Vector PDF) cho Bước 6."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy import stats
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.evaluation.xai_rq3 import (
    BOTTOM5_LOSO_SUBJECTS,
    EOG_ROI,
    LEFT_MOTOR_ROI,
    MONTAGE_NAME,
    P300_ROI,
    RIGHT_MOTOR_ROI,
    SFREQ,
    TIME_WINDOWS_LABELS,
    TIME_WINDOWS_ORDER,
    create_colin27_info,
    load_channel_names,
    verify_rois,
)

# Bảng màu Okabe-Ito thân thiện mù màu
OKABE_ITO: Dict[str, str] = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "black": "#1F2937",
    "grey": "#6B7280",
    "red": "#B91C1C",
}


def setup_publication_style() -> None:
    """Thiết lập cấu hình Matplotlib đồng nhất chuẩn bài báo."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "figure.titlesize": 12.0,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "grid.alpha": 0.28,
            "grid.linestyle": "--",
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _p_to_stars(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "n.s."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _fmt_p_text(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "p = NaN"
    if p < 1e-4:
        return f"p = {p:.1e}"
    return f"p = {p:.4f}"


def _mean_ci95_1d(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, dtype=np.float64)
    n = a.shape[0]
    mean = np.mean(a, axis=0)
    if n <= 1:
        return mean, np.zeros_like(mean)
    std = np.std(a, axis=0, ddof=1)
    ci = float(stats.t.ppf(0.975, df=n - 1)) * std / math.sqrt(n)
    return mean, ci


def _plot_topomap_clean(
    values: np.ndarray,
    info: mne.Info,
    ax: plt.Axes,
    *,
    cmap: str = "RdBu_r",
    vlim: Tuple[Optional[float], Optional[float]] = (None, None),
    sensors: bool = False,
    contours: int = 4,
    mask: Optional[np.ndarray] = None,
) -> Any:
    mask_params = (
        dict(marker="o", markerfacecolor="white", markeredgecolor="black", linewidth=0.5, markersize=3.8)
        if mask is not None
        else None
    )
    im, _ = mne.viz.plot_topomap(
        np.asarray(values, dtype=np.float64),
        info,
        axes=ax,
        cmap=cmap,
        vlim=vlim,
        sensors=sensors,
        contours=contours,
        outlines="head",
        extrapolate="head",
        mask=mask,
        mask_params=mask_params,
        show=False,
    )
    return im


def _add_sig_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    fontsize: float = 8.2,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.0, c=OKABE_ITO["black"])
    ax.text((x1 + x2) * 0.5, y + h * 1.15, text, ha="center", va="bottom", fontsize=fontsize, color=OKABE_ITO["black"])


# ==============================================================================
# FIGURE 1 — GRAND AVERAGE ERP & TOPOMAP SEQUENCE
# ==============================================================================
def plot_fig1_erp_overview(
    cache: Dict[str, Any],
    info: mne.Info,
    channel_names: Sequence[str],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Fig 1: Grand Average ERP tại Pz, Cz, Fz + chuỗi 6 Topomap hiệu (Target - Frequent)."""
    setup_publication_style()
    times_ms = cache["times_ms"].astype(np.float64)
    erp_tgt = cache["loso_erp_target"].astype(np.float64) * 1e6  # (39, 127, 251) µV
    erp_freq = cache["loso_erp_freq"].astype(np.float64) * 1e6
    erp_diff = erp_tgt - erp_freq

    fig = plt.figure(figsize=(12.2, 6.8), constrained_layout=False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.25, 0.95], hspace=0.38, figure=fig)

    # Tầng trên: 3 kênh Pz, Cz, Fz
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.16)
    axes_erp = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    target_chs = ["Pz", "Cz", "Fz"]

    y_min_all, y_max_all = float("inf"), float("-inf")
    for ch in target_chs:
        idx = channel_names.index(ch)
        mt, ct = _mean_ci95_1d(erp_tgt[:, idx, :])
        mf, cf = _mean_ci95_1d(erp_freq[:, idx, :])
        y_min_all = min(y_min_all, float(np.min(mt - ct)), float(np.min(mf - cf)))
        y_max_all = max(y_max_all, float(np.max(mt + ct)), float(np.max(mf + cf)))

    y_pad = 0.12 * (y_max_all - y_min_all)
    ylim = (y_min_all - y_pad, y_max_all + y_pad)

    for i, ch in enumerate(target_chs):
        ax = axes_erp[i]
        idx = channel_names.index(ch)
        mt, ct = _mean_ci95_1d(erp_tgt[:, idx, :])
        mf, cf = _mean_ci95_1d(erp_freq[:, idx, :])
        md = mt - mf

        ax.axvspan(300, 600, color=OKABE_ITO["yellow"], alpha=0.15, label="P300 window (300–600 ms)" if i == 0 else None)
        ax.axvline(0, color=OKABE_ITO["black"], ls="--", lw=0.9, alpha=0.75)
        ax.axhline(0, color=OKABE_ITO["grey"], ls=":", lw=0.8, alpha=0.75)

        ax.fill_between(times_ms, mt - ct, mt + ct, color=OKABE_ITO["orange"], alpha=0.22)
        ax.plot(times_ms, mt, color=OKABE_ITO["orange"], lw=2.0, label="Target (Mean ± 95% CI)" if i == 0 else None)

        ax.fill_between(times_ms, mf - cf, mf + cf, color=OKABE_ITO["blue"], alpha=0.20)
        ax.plot(times_ms, mf, color=OKABE_ITO["blue"], lw=1.7, label="Frequent (Mean ± 95% CI)" if i == 0 else None)

        ax.plot(times_ms, md, color=OKABE_ITO["black"], lw=1.2, ls="-.", alpha=0.85, label="Difference (Target − Freq)" if i == 0 else None)

        p300_mask = (times_ms >= 300) & (times_ms <= 600)
        pk_pos = int(np.argmax(md[p300_mask]))
        pk_t = float(times_ms[p300_mask][pk_pos])
        pk_v = float(md[p300_mask][pk_pos])
        ax.annotate(
            f"ΔPeak: {pk_v:+.2f} µV\n@ {pk_t:.0f} ms",
            xy=(pk_t, mt[p300_mask][pk_pos]),
            xytext=(pk_t + 35, ylim[1] * 0.78),
            fontsize=7.8,
            color=OKABE_ITO["black"],
            arrowprops=dict(arrowstyle="->", lw=0.8, color=OKABE_ITO["grey"]),
        )

        ax.set_title(f"Electrode {ch} (N = 39 subjects)")
        ax.set_xlabel("Time relative to stimulus (ms)")
        if i == 0:
            ax.set_ylabel("Amplitude (µV)")
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlim(-200, 800)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25)

    handles, labels = axes_erp[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False)

    # Tầng dưới: 6 MNE Topomap tại 100, 200, 300, 400, 500, 600 ms
    topo_times = [100, 200, 300, 400, 500, 600]
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.14], subplot_spec=gs[1], wspace=0.14)
    grand_diff = erp_diff.mean(axis=0)  # (127, 251)

    topo_slices = []
    for t_ms in topo_times:
        t_idx = int(np.argmin(np.abs(times_ms - t_ms)))
        topo_slices.append(grand_diff[:, t_idx])
    vmax = float(max(np.max(np.abs(s)) for s in topo_slices))
    vmax = max(vmax, 1.0)

    im_ref = None
    for j, (t_ms, vals) in enumerate(zip(topo_times, topo_slices)):
        ax_t = fig.add_subplot(gs_bot[0, j])
        im_ref = _plot_topomap_clean(vals, info, ax_t, cmap="RdBu_r", vlim=(-vmax, vmax), sensors=False)
        ax_t.set_title(f"{t_ms} ms", fontsize=9.5, pad=4)

    cax = fig.add_subplot(gs_bot[0, 6])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label("Target − Frequent (µV)", fontsize=8.8)

    fig.suptitle("Grand Average ERP Waveforms and Topographic Evolution (colin27_1005)", y=0.995)
    return _save_figure(fig, out_dir, "fig1_erp_overview")


# ==============================================================================
# FIGURE 2 — SINGLE-TRIAL JITTER & REACTION TIME DISTRIBUTION
# ==============================================================================
def plot_fig2_single_trial_and_rt(
    data_dir: Path,
    channel_names: Sequence[str],
    out_dir: Path,
    *,
    subject_id: str = "sub-03",
    seed: int = 42,
) -> Tuple[Path, Path]:
    """Fig 2: 10 single-trial Target ngẫu nhiên xếp chồng lên Grand Average tại Pz + phân bố RT."""
    setup_publication_style()
    X_clean = np.load(data_dir / "X_clean.npy", mmap_mode="r")
    times_ms = np.round(np.load(data_dir / "times.npy") * 1000.0, 3)
    meta_df = pd.read_csv(data_dir / "metadata.csv")
    pz_idx = channel_names.index("Pz")

    # Panel (a): sub-03 (hoặc fallback sang subject đầu tiên có Target)
    sub_mask = (meta_df["subject_id"].astype(str) == subject_id) & (meta_df["label"] == 1)
    if int(sub_mask.sum()) == 0:
        subject_id = str(meta_df.loc[meta_df["label"] == 1, "subject_id"].iloc[0])
        sub_mask = (meta_df["subject_id"].astype(str) == subject_id) & (meta_df["label"] == 1)

    sub_tgt_indices = np.where(sub_mask.to_numpy())[0]
    rng = np.random.default_rng(seed)
    chosen_idx = np.sort(rng.choice(sub_tgt_indices, size=min(10, len(sub_tgt_indices)), replace=False))
    single_trials_uv = np.asarray(X_clean[chosen_idx, pz_idx, :], dtype=np.float64) * 1e6
    sub_ga_uv = np.asarray(X_clean[sub_tgt_indices, pz_idx, :], dtype=np.float64).mean(axis=0) * 1e6

    all_tgt_indices = np.where((meta_df["label"] == 1).to_numpy())[0]
    all_ga_uv = np.asarray(X_clean[all_tgt_indices, pz_idx, :], dtype=np.float64).mean(axis=0) * 1e6

    # Panel (b): Phân bố RT của 2,850 Target sạch
    rts = meta_df.loc[meta_df["label"] == 1, "rt_ms"].dropna().to_numpy(dtype=np.float64)
    rt_med = float(np.median(rts))
    rt_q25, rt_q75 = float(np.percentile(rts, 25)), float(np.percentile(rts, 75))
    pct_in_p300 = float(np.mean((rts >= 300.0) & (rts <= 600.0)) * 100.0)
    pct_fast = float(np.mean(rts < 450.0) * 100.0)
    pct_slow = float(np.mean(rts > 550.0) * 100.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.6), gridspec_kw={"wspace": 0.24})

    # (a)
    ax1.axvspan(300, 600, color=OKABE_ITO["yellow"], alpha=0.14, label="P300 window (300–600 ms)")
    ax1.axvline(0, color=OKABE_ITO["black"], ls="--", lw=0.9, alpha=0.75)
    ax1.axhline(0, color=OKABE_ITO["grey"], ls=":", lw=0.8)
    for k in range(single_trials_uv.shape[0]):
        ax1.plot(
            times_ms,
            single_trials_uv[k],
            color=OKABE_ITO["grey"],
            alpha=0.35,
            lw=0.95,
            label=f"10 single trials ({subject_id})" if k == 0 else None,
        )
    ax1.plot(times_ms, sub_ga_uv, color=OKABE_ITO["blue"], lw=2.1, label=f"{subject_id} Target Avg (N={len(sub_tgt_indices)})")
    ax1.plot(times_ms, all_ga_uv, color=OKABE_ITO["orange"], lw=2.3, label=f"Global Target Avg (N={len(all_tgt_indices):,})")
    ax1.set_xlim(-200, 800)
    ax1.set_xlabel("Time relative to stimulus (ms)")
    ax1.set_ylabel("Amplitude at Pz (µV)")
    ax1.set_title(f"(a) Single-Trial SNR & Latency Jitter at Pz ({subject_id})")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.88, edgecolor="none")
    ax1.grid(True, alpha=0.25)

    # (b)
    ax2.axvspan(300, 600, color=OKABE_ITO["yellow"], alpha=0.18, label=f"P300 window [300, 600] ms ({pct_in_p300:.2f}%)")
    ax2.axvspan(rt_q25, rt_q75, color=OKABE_ITO["sky"], alpha=0.16, label=f"IQR [{rt_q25:.0f}–{rt_q75:.0f} ms]")
    bins = np.linspace(250, 1200, 39)
    ax2.hist(rts, bins=bins, density=True, color=OKABE_ITO["blue"], alpha=0.45, edgecolor="white", linewidth=0.5)
    kde = stats.gaussian_kde(rts)
    xs = np.linspace(250, 1200, 300)
    ax2.plot(xs, kde(xs), color=OKABE_ITO["blue"], lw=2.1, label="KDE density")
    ax2.axvline(rt_med, color=OKABE_ITO["orange"], ls="--", lw=1.8, label=f"Median RT = {rt_med:.1f} ms")
    ax2.axvline(450, color=OKABE_ITO["green"], ls=":", lw=1.3, label=f"Cutoff = 450 ms (Fast: {pct_fast:.1f}%, Slow >550: {pct_slow:.1f}%)")

    ax2.text(
        0.97,
        0.54,
        f"N = {len(rts):,} Target Hits\nMean = {np.mean(rts):.1f} ± {np.std(rts, ddof=1):.1f} ms\n"
        f"In [300, 600] ms: {pct_in_p300:.2f}%",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#D1D5DB", alpha=0.94),
    )
    ax2.set_xlim(250, 1150)
    ax2.set_ylim(0, float(np.max(kde(xs))) * 1.36)
    ax2.set_xlabel("Reaction Time (ms)")
    ax2.set_ylabel("Probability Density")
    ax2.set_title("(b) Target Reaction Time Distribution vs P300 Window")
    ax2.legend(loc="upper right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.5)
    ax2.grid(True, alpha=0.25)

    return _save_figure(fig, out_dir, "fig2_single_trial_and_rt")


# ==============================================================================
# FIGURE 3 — BENCHMARK RQ1 & RQ2 (+ ABLATION & SENSITIVITY)
# ==============================================================================
def _load_external_benchmark_rows(external_dir: Optional[Path]) -> List[Dict[str, Any]]:
    """Tùy chọn nạp thêm bảng điểm per-subject của các nhánh ML1–ML3 & DL1 nếu có."""
    if external_dir is None or not external_dir.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for csv_path in sorted(external_dir.glob("*.csv")):
        if csv_path.stem.startswith("p300_atcnet_"):
            continue
        try:
            df = pd.read_csv(csv_path)
            ba_col = "bal_acc" if "bal_acc" in df.columns else (
                "balanced_accuracy" if "balanced_accuracy" in df.columns else None
            )
            if ba_col is not None and {"subject_id", "roc_auc"}.issubset(df.columns):
                label = (
                    csv_path.stem.replace("handover_", "")
                    .replace("_per_fold", "")
                    .replace("_", " ")
                )
                grouped = df.groupby("subject_id", sort=False, as_index=False)[
                    ["roc_auc", ba_col]
                ].mean()
                for _, r in grouped.iterrows():
                    rows.append(
                        {
                            "label": label,
                            "subject_id": str(r["subject_id"]),
                            "roc_auc": float(r["roc_auc"]),
                            "balanced_accuracy": float(r[ba_col]),
                        }
                    )
        except Exception:
            continue
    return rows


def plot_fig3_benchmark_rq1_rq2(
    data_dir: Path,
    metrics_dir: Path,
    out_dir: Path,
    *,
    external_results_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Fig 3: Phân bố 39 subjects (P300-ATCNet Within vs LOSO) + CS5 Clean vs No-ICA & Sensitivity (39/37/31)."""
    setup_publication_style()
    handover_df = pd.read_csv(metrics_dir / "handover_per_subject_summary_dl2.csv")
    qc_df = pd.read_csv(data_dir / "preprocessing_qc_summary.csv")
    qc_map = {str(r["subject_id"]): r for _, r in qc_df.iterrows()}

    # Thu thập dữ liệu Panel (a): Boxplot + jittered points
    models_a = [
        ("Within-Subject\nP300-ATCNet", handover_df["ws_roc_auc"].to_numpy(), handover_df["ws_balanced_accuracy"].to_numpy(), OKABE_ITO["blue"]),
        ("Cross-Subject LOSO\nP300-ATCNet", handover_df["loso_roc_auc"].to_numpy(), handover_df["loso_balanced_accuracy"].to_numpy(), OKABE_ITO["orange"]),
    ]

    ext_rows = _load_external_benchmark_rows(external_results_dir)
    if ext_rows:
        ext_df = pd.DataFrame(ext_rows)
        for lbl, grp in ext_df.groupby("label", sort=False):
            if len(grp) == 39:
                models_a.insert(0, (lbl, grp["roc_auc"].to_numpy(), grp["balanced_accuracy"].to_numpy(), OKABE_ITO["grey"]))

    fig = plt.figure(figsize=(12.8, 5.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.12, 1.0], wspace=0.25, figure=fig)

    # Panel (a)
    ax1 = fig.add_subplot(gs[0])
    n_groups = len(models_a)
    rng = np.random.default_rng(42)
    positions_roc = np.arange(n_groups) * 2.2 - 0.36
    positions_ba = np.arange(n_groups) * 2.2 + 0.36

    for idx, (lbl, roc_vals, ba_vals, col) in enumerate(models_a):
        # ROC-AUC box (filled)
        bp1 = ax1.boxplot(
            [roc_vals],
            positions=[positions_roc[idx]],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", lw=1.6),
            whiskerprops=dict(color=col, lw=1.1),
            capprops=dict(color=col, lw=1.1),
            boxprops=dict(facecolor=col, edgecolor=col, alpha=0.78),
        )
        _ = bp1
        jit = rng.uniform(-0.14, 0.14, size=len(roc_vals))
        ax1.scatter(positions_roc[idx] + jit, roc_vals, s=13, color=col, alpha=0.55, zorder=3)

        # Balanced Acc box (hatched / light)
        bp2 = ax1.boxplot(
            [ba_vals],
            positions=[positions_ba[idx]],
            widths=0.52,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=OKABE_ITO["black"], lw=1.6),
            whiskerprops=dict(color=col, lw=1.1, ls="--"),
            capprops=dict(color=col, lw=1.1),
            boxprops=dict(facecolor="white", edgecolor=col, linewidth=1.4, hatch="///"),
        )
        _ = bp2
        ax1.scatter(positions_ba[idx] + jit, ba_vals, s=13, facecolors="white", edgecolors=col, linewidths=0.8, alpha=0.75, zorder=3)

    # Wilcoxon stars cho ROC-AUC
    w_loso = stats.wilcoxon(handover_df["ws_roc_auc"], handover_df["loso_roc_auc"]).pvalue
    idx_ws = n_groups - 2
    idx_lo = n_groups - 1
    _add_sig_bracket(ax1, positions_roc[idx_ws], positions_roc[idx_lo], 1.030, 0.012, f"ROC: {_p_to_stars(w_loso)} ({_fmt_p_text(w_loso)})", fontsize=7.6)

    ax1.set_xticks(np.arange(n_groups) * 2.2)
    ax1.set_xticklabels([m[0] for m in models_a])
    ax1.set_ylim(0.68, 1.095)
    ax1.set_ylabel("Score (39 Subjects)")
    ax1.set_title("(a) Within-Subject & LOSO Performance (N = 39 Subjects)")
    ax1.grid(True, axis="y", alpha=0.25)

    # Legend giả cho ROC-AUC vs Balanced Acc
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=OKABE_ITO["blue"], edgecolor=OKABE_ITO["blue"], alpha=0.78, label="ROC-AUC (solid)"),
        Patch(facecolor="white", edgecolor=OKABE_ITO["blue"], hatch="///", label="Balanced Acc (hatched)"),
    ]
    ax1.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.9, edgecolor="none")

    # Panel (b): CS5 Clean vs No-ICA & Sensitivity Analysis (39 vs 37 vs 31)
    ax2 = fig.add_subplot(gs[1])
    ablation_specs = [
        ("CS5\nClean", "metrics_dl2_cs5_p300_atcnet_clean.json", OKABE_ITO["sky"]),
        ("CS5\nNo-ICA", "metrics_dl2_cs5_p300_atcnet_no_ica.json", OKABE_ITO["purple"]),
    ]
    x_pos: List[float] = []
    labels_b: List[str] = []
    roc_means, roc_sds, pr_means, pr_sds = [], [], [], []
    colors_b: List[str] = []

    cur_x = 0.0
    for lbl, fname, col in ablation_specs:
        fpath = metrics_dir / fname
        if fpath.exists():
            j = json.loads(fpath.read_text(encoding="utf-8"))
            roc_means.append(float(j["summary"]["roc_auc"]["mean"]))
            roc_sds.append(float(j["summary"]["roc_auc"]["std"]))
            pr_means.append(float(j["summary"]["pr_auc"]["mean"]))
            pr_sds.append(float(j["summary"]["pr_auc"]["std"]))
            x_pos.append(cur_x)
            labels_b.append(lbl)
            colors_b.append(col)
            cur_x += 1.05

    cur_x += 0.65  # Khoảng tách giữa Ablation và Sensitivity
    subs_all = handover_df["subject_id"].astype(str).tolist()
    mask_37 = np.array([float(qc_map[s]["drop_rate_pct"]) <= 30.0 for s in subs_all])
    mask_31 = np.array([float(r["hit_rate_pct"]) >= 90.0 for _, r in handover_df.iterrows()])

    for cohort_lbl, m_arr, col in (
        ("Full\n(N=39)", np.ones(len(subs_all), dtype=bool), OKABE_ITO["orange"]),
        ("Drop≤30%\n(N=37)", mask_37, OKABE_ITO["green"]),
        ("Hit≥90%\n(N=31)", mask_31, OKABE_ITO["yellow"]),
    ):
        r_v = handover_df.loc[m_arr, "loso_roc_auc"].to_numpy(dtype=np.float64)
        p_v = handover_df.loc[m_arr, "loso_pr_auc"].to_numpy(dtype=np.float64)
        roc_means.append(float(np.mean(r_v)))
        roc_sds.append(float(np.std(r_v, ddof=1)))
        pr_means.append(float(np.mean(p_v)))
        pr_sds.append(float(np.std(p_v, ddof=1)))
        x_pos.append(cur_x)
        labels_b.append(cohort_lbl)
        colors_b.append(col)
        cur_x += 1.05

    x_arr = np.array(x_pos)
    w_bar = 0.35
    ax2.bar(x_arr - w_bar / 2, roc_means, yerr=roc_sds, width=w_bar, color=colors_b, alpha=0.88, capsize=3, label="ROC-AUC (Mean ± SD)")
    ax2.bar(x_arr + w_bar / 2, pr_means, yerr=pr_sds, width=w_bar, color=colors_b, alpha=0.45, edgecolor=colors_b, linewidth=1.2, capsize=3, hatch="..", label="PR-AUC (Mean ± SD)")

    for xp, rm, rs, pm, ps in zip(x_arr, roc_means, roc_sds, pr_means, pr_sds):
        ax2.text(xp - w_bar / 2, rm + rs + 0.008, f"{rm:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)
        ax2.text(xp + w_bar / 2, pm + ps + 0.008, f"{pm:.3f}", ha="center", va="bottom", fontsize=7.2, rotation=90)

    if len(x_arr) >= 5:
        sep_x = 0.5 * (x_arr[1] + x_arr[2])
        ax2.axvline(sep_x, color=OKABE_ITO["grey"], ls="--", lw=1.0)
        ax2.text(0.5 * (x_arr[0] + x_arr[1]), 1.105, "ICA Ablation\n(CS 5-Fold)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=OKABE_ITO["black"])
        ax2.text(x_arr[3], 1.105, "Cohort Sensitivity (LOSO)", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=OKABE_ITO["black"])

    ax2.set_xticks(x_arr)
    ax2.set_xticklabels(labels_b, fontsize=8.0)
    ax2.set_ylim(0.70, 1.27)
    ax2.set_ylabel("Score (Mean ± SD)")
    ax2.set_title("(b) ICA Ablation & Cohort Sensitivity")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.8)

    return _save_figure(fig, out_dir, "fig3_benchmark_rq1_rq2")


def _load_preds_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(npz_path, allow_pickle=False) as z:
            out: Dict[str, np.ndarray] = {
                "y_true": np.asarray(z["y_true"], dtype=np.int64),
                "y_prob": np.asarray(z["y_prob"], dtype=np.float64),
            }
            if "subject_id" in z.files:
                out["subject_id"] = np.asarray(z["subject_id"]).astype(str)
            return out
    except ValueError:
        with np.load(npz_path, allow_pickle=True) as z:
            out = {
                "y_true": np.asarray(z["y_true"], dtype=np.int64),
                "y_prob": np.asarray(z["y_prob"], dtype=np.float64),
            }
            if "subject_id" in z.files:
                out["subject_id"] = np.asarray(z["subject_id"]).astype(str)
            return out


def _load_preds_csv(csv_path: Path) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    out: Dict[str, np.ndarray] = {
        "y_true": df["y_true"].to_numpy(dtype=np.int64),
        "y_prob": df["y_prob"].to_numpy(dtype=np.float64),
    }
    if "subject_id" in df.columns:
        out["subject_id"] = df["subject_id"].astype(str).to_numpy()
    return out


def _format_npz_label(stem: str) -> str:
    clean = stem.replace("_preds", "")
    mapping = {
        "p300_atcnet_within": "P300-ATCNet Within (5-Fold)",
        "p300_atcnet_loso": "P300-ATCNet LOSO (39 Subs)",
        "p300_atcnet_cs5_clean": "P300-ATCNet CS5 (Clean)",
    }
    return mapping.get(clean, clean.replace("_", " ").title())


def _compute_roc_pr_summary(
    payload: Dict[str, np.ndarray],
    grid: np.ndarray,
) -> Dict[str, Any]:
    y_true = payload["y_true"]
    y_prob = payload["y_prob"]
    subject_ids = payload.get("subject_id")

    if subject_ids is not None:
        uniq_subs = pd.unique(subject_ids)
        if len(uniq_subs) >= 2:
            tpr_curves, prec_curves, roc_aucs, pr_aucs = [], [], [], []
            for sub in uniq_subs:
                mask = subject_ids == sub
                yt = y_true[mask]
                yp = y_prob[mask]
                if len(np.unique(yt)) < 2:
                    continue
                fpr, tpr, _ = roc_curve(yt, yp)
                tpr_i = np.interp(grid, fpr, tpr)
                tpr_i[0] = 0.0
                tpr_i[-1] = 1.0
                tpr_curves.append(tpr_i)
                roc_aucs.append(float(roc_auc_score(yt, yp)))

                prec, rec, _ = precision_recall_curve(yt, yp)
                prec_i = np.interp(grid, rec[::-1], prec[::-1])
                prec_curves.append(prec_i)
                pr_aucs.append(float(average_precision_score(yt, yp)))

            if len(tpr_curves) >= 2:
                roc_m, roc_ci = _mean_ci95_1d(np.asarray(tpr_curves, dtype=np.float64))
                pr_m, pr_ci = _mean_ci95_1d(np.asarray(prec_curves, dtype=np.float64))
                return {
                    "n_groups": len(tpr_curves),
                    "roc_mean_curve": roc_m,
                    "roc_ci_curve": roc_ci,
                    "pr_mean_curve": pr_m,
                    "pr_ci_curve": pr_ci,
                    "roc_auc_mean": float(np.mean(roc_aucs)),
                    "roc_auc_std": float(np.std(roc_aucs, ddof=1)),
                    "pr_auc_mean": float(np.mean(pr_aucs)),
                    "pr_auc_std": float(np.std(pr_aucs, ddof=1)),
                    "prevalence": float(np.mean(y_true)),
                }

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    tpr_i = np.interp(grid, fpr, tpr)
    tpr_i[0] = 0.0
    tpr_i[-1] = 1.0
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    prec_i = np.interp(grid, rec[::-1], prec[::-1])
    roc_val = float(roc_auc_score(y_true, y_prob))
    pr_val = float(average_precision_score(y_true, y_prob))
    zeros = np.zeros_like(grid, dtype=np.float64)
    return {
        "n_groups": 1,
        "roc_mean_curve": tpr_i,
        "roc_ci_curve": zeros,
        "pr_mean_curve": prec_i,
        "pr_ci_curve": zeros,
        "roc_auc_mean": roc_val,
        "roc_auc_std": 0.0,
        "pr_auc_mean": pr_val,
        "pr_auc_std": 0.0,
        "prevalence": float(np.mean(y_true)),
    }


def plot_fig3b_roc_pr_curves(
    metrics_dir: Path,
    out_dir: Path,
    *,
    external_results_dir: Optional[Path] = None,
    handover_dir: Path = Path("results/handover_standard"),
) -> Tuple[Path, Path]:
    """Fig 3b: Overlaid ROC & Precision-Recall curves từ `*_preds.npz` (hỗ trợ `--external-results-dir`)."""
    setup_publication_style()
    dl2_preview_specs = [
        ("p300_atcnet_loso_preds.npz", "predictions_dl2_loso_p300_atcnet.csv", "P300-ATCNet LOSO (39 Subs)", OKABE_ITO["orange"], "-"),
        ("p300_atcnet_within_preds.npz", "predictions_dl2_ws_p300_atcnet.csv", "P300-ATCNet Within (5-Fold)", OKABE_ITO["blue"], "-"),
        ("p300_atcnet_cs5_clean_preds.npz", "predictions_dl2_cs5_p300_atcnet_clean.csv", "P300-ATCNet CS5 (Clean)", OKABE_ITO["green"], "--"),
    ]
    palette_cycle = [
        OKABE_ITO["green"],
        OKABE_ITO["purple"],
        OKABE_ITO["yellow"],
        OKABE_ITO["red"],
        OKABE_ITO["black"],
        OKABE_ITO["blue"],
        OKABE_ITO["orange"],
        OKABE_ITO["sky"],
    ]
    canonical_style = {npz_name: (lbl, col, ls) for npz_name, _, lbl, col, ls in dl2_preview_specs}

    entries: List[Tuple[str, Dict[str, np.ndarray], str, str]] = []
    loaded_names: set[str] = set()

    if external_results_dir is not None and external_results_dir.exists():
        ext_npzs = sorted(external_results_dir.glob("*_preds.npz"))
        ext_color_idx = 0
        for npz_p in ext_npzs:
            try:
                payload = _load_preds_npz(npz_p)
            except Exception:
                continue
            if npz_p.name in canonical_style:
                lbl, col, ls = canonical_style[npz_p.name]
            else:
                lbl = _format_npz_label(npz_p.stem)
                col = palette_cycle[ext_color_idx % len(palette_cycle)]
                ls = "-"
                ext_color_idx += 1
            entries.append((lbl, payload, col, ls))
            loaded_names.add(npz_p.name)

    # Nếu chưa có đủ bộ đối chiếu DL2 thì nạp thêm từ handover_dir hoặc metrics_dir CSV
    dl2_stems_present = any(n in loaded_names for n in canonical_style)
    if not entries or not dl2_stems_present:
        for npz_name, csv_name, lbl, col, ls in dl2_preview_specs:
            if npz_name in loaded_names:
                continue
            npz_path = handover_dir / npz_name
            csv_path = metrics_dir / csv_name
            if npz_path.exists():
                entries.append((lbl, _load_preds_npz(npz_path), col, ls))
            elif csv_path.exists():
                entries.append((lbl, _load_preds_csv(csv_path), col, ls))

    grid = np.linspace(0.0, 1.0, 201)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0), gridspec_kw={"wspace": 0.24})

    prevalences: List[float] = []
    for lbl, payload, col, ls in entries:
        stats_d = _compute_roc_pr_summary(payload, grid)
        prevalences.append(stats_d["prevalence"])
        if stats_d["n_groups"] > 1:
            roc_lbl = f"{lbl} ({stats_d['roc_auc_mean']:.3f} ± {stats_d['roc_auc_std']:.3f})"
            pr_lbl = f"{lbl} ({stats_d['pr_auc_mean']:.3f} ± {stats_d['pr_auc_std']:.3f})"
        else:
            roc_lbl = f"{lbl} ({stats_d['roc_auc_mean']:.3f})"
            pr_lbl = f"{lbl} ({stats_d['pr_auc_mean']:.3f})"

        ax1.fill_between(
            grid,
            np.clip(stats_d["roc_mean_curve"] - stats_d["roc_ci_curve"], 0.0, 1.0),
            np.clip(stats_d["roc_mean_curve"] + stats_d["roc_ci_curve"], 0.0, 1.0),
            color=col,
            alpha=0.14,
        )
        ax1.plot(grid, stats_d["roc_mean_curve"], color=col, ls=ls, lw=2.1, label=roc_lbl)

        ax2.fill_between(
            grid,
            np.clip(stats_d["pr_mean_curve"] - stats_d["pr_ci_curve"], 0.0, 1.0),
            np.clip(stats_d["pr_mean_curve"] + stats_d["pr_ci_curve"], 0.0, 1.0),
            color=col,
            alpha=0.14,
        )
        ax2.plot(grid, stats_d["pr_mean_curve"], color=col, ls=ls, lw=2.1, label=pr_lbl)

    prev = float(np.mean(prevalences)) if prevalences else 0.136
    ax1.plot([0.0, 1.0], [0.0, 1.0], color=OKABE_ITO["grey"], ls=":", lw=1.2, label="Chance (AUC = 0.500)")
    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 1.02)
    ax1.set_xlabel("False Positive Rate (1 − Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("(a) Overlaid ROC Curves (Subject Mean ± 95% CI)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.8)
    ax1.grid(True, alpha=0.25)

    ax2.axhline(prev, color=OKABE_ITO["grey"], ls=":", lw=1.2, label=f"Chance (Prevalence = {prev:.3f})")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.02)
    ax2.set_xlabel("Recall (Sensitivity)")
    ax2.set_ylabel("Precision (Positive Predictive Value)")
    ax2.set_title("(b) Overlaid Precision–Recall Curves (Subject Mean ± 95% CI)")
    ax2.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.8)
    ax2.grid(True, alpha=0.25)

    return _save_figure(fig, out_dir, "fig3b_roc_pr_curves")


# ==============================================================================
# FIGURE 4 — XAI SPATIOTEMPORAL ALIGNMENT (ERP vs IG vs SPATIAL WEIGHTS)
# ==============================================================================
def plot_fig4_xai_spatiotemporal(
    cache: Dict[str, Any],
    summary: Dict[str, Any],
    info: mne.Info,
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Fig 4: 3 Topomap 127 kênh (300–500 ms) + Đường cong Temporal Attribution [-200, 800] ms."""
    setup_publication_style()
    times_ms = cache["times_ms"].astype(np.float64)
    mask_300_500 = (times_ms >= 300.0 - 1e-6) & (times_ms <= 500.0 + 1e-6)

    erp_diff_3d = (cache["loso_erp_target"] - cache["loso_erp_freq"]).astype(np.float64) * 1e6
    abs_ig_3d = cache["loso_target_abs_ig"].astype(np.float64)
    sp_weights = cache["loso_spatial_weights_abs"].astype(np.float64)

    topo_erp = erp_diff_3d[:, :, mask_300_500].mean(axis=(0, 2))
    topo_ig = abs_ig_3d[:, :, mask_300_500].mean(axis=(0, 2))
    topo_sp = sp_weights.mean(axis=0)

    l1 = summary["layer1_temporal"]
    sc = summary["layer4_frequency_and_neuro"]["spatial_correlation"]
    corr_abs = sc["win_300_500_abs_ig_vs_abs_erp_diff"]
    corr_sp = l1["spatial_weights_vs_ig_300_500"]

    fig = plt.figure(figsize=(12.2, 7.2))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.12], hspace=0.34, figure=fig)

    # Hàng trên: 3 Topomaps
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0], wspace=0.25)
    vmax_erp = float(np.max(np.abs(topo_erp)))

    ax_t1 = fig.add_subplot(gs_top[0, 0])
    im1 = _plot_topomap_clean(topo_erp, info, ax_t1, cmap="RdBu_r", vlim=(-vmax_erp, vmax_erp))
    cb1 = fig.colorbar(im1, ax=ax_t1, fraction=0.046, pad=0.04)
    cb1.set_label("µV", fontsize=8.2)
    ax_t1.set_title("(a) Grand Avg ERP (Target − Freq)\n300–500 ms", fontsize=9.8)

    ax_t2 = fig.add_subplot(gs_top[0, 1])
    im2 = _plot_topomap_clean(topo_ig, info, ax_t2, cmap="viridis", vlim=(float(np.min(topo_ig)), float(np.max(topo_ig))))
    cb2 = fig.colorbar(im2, ax=ax_t2, fraction=0.046, pad=0.04)
    cb2.set_label("Mean |IG|", fontsize=8.2)
    ax_t2.set_title(
        f"(b) P300-ATCNet Captum |IG| (300–500 ms)\n"
        f"|IG| vs |ERP|: r = {corr_abs['pearson_r']:+.3f} ({_fmt_p_text(corr_abs.get('pearson_p'))}), "
        f"ρ = {corr_abs['spearman_rho']:+.3f}",
        fontsize=9.1,
    )

    ax_t3 = fig.add_subplot(gs_top[0, 2])
    im3 = _plot_topomap_clean(topo_sp, info, ax_t3, cmap="magma", vlim=(float(np.min(topo_sp)), float(np.max(topo_sp))))
    cb3 = fig.colorbar(im3, ax=ax_t3, fraction=0.046, pad=0.04)
    cb3.set_label("Mean |W_spatial|", fontsize=8.2)
    ax_t3.set_title(
        f"(c) Depthwise Spatial Conv |W| (39 LOSO)\n"
        f"vs |IG|: r = {corr_sp['pearson_r']:+.3f}, ρ = {corr_sp['spearman_rho']:+.3f}",
        fontsize=9.3,
    )

    # Hàng dưới: Đường cong Temporal Attribution
    ax_b = fig.add_subplot(gs[1])
    t_mean = np.asarray(l1["temporal_curve_target_abs"]["mean"], dtype=np.float64)
    t_ci = np.asarray(l1["temporal_curve_target_abs"]["ci95"], dtype=np.float64)
    f_mean = np.asarray(l1["temporal_curve_freq_abs"]["mean"], dtype=np.float64)
    f_ci = np.asarray(l1["temporal_curve_freq_abs"]["ci95"], dtype=np.float64)

    win_colors = {
        "baseline": "#9CA3AF",
        "early_visual": OKABE_ITO["sky"],
        "p300": OKABE_ITO["yellow"],
        "late_post_response": OKABE_ITO["purple"],
    }
    win_bounds = {
        "baseline": (-200, 0),
        "early_visual": (0, 300),
        "p300": (300, 600),
        "late_post_response": (600, 800),
    }

    y_top = float(np.max(t_mean + t_ci)) * 1.24
    for w_name in TIME_WINDOWS_ORDER:
        x0, x1 = win_bounds[w_name]
        share = l1["windows"][w_name]["energy_share_pct"]["mean"]
        ax_b.axvspan(x0, x1, color=win_colors[w_name], alpha=0.13)
        ax_b.text(
            0.5 * (x0 + x1),
            y_top * 0.93,
            f"{TIME_WINDOWS_LABELS[w_name]}\n{share:.1f}% energy",
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color=OKABE_ITO["black"],
        )

    ax_b.axvline(0, color=OKABE_ITO["black"], ls="--", lw=1.0, alpha=0.8)
    ax_b.fill_between(times_ms, t_mean - t_ci, t_mean + t_ci, color=OKABE_ITO["orange"], alpha=0.25)
    ax_b.plot(times_ms, t_mean, color=OKABE_ITO["orange"], lw=2.2, label="Target |IG| (Mean ± 95% CI, 39 subs)")
    ax_b.fill_between(times_ms, f_mean - f_ci, f_mean + f_ci, color=OKABE_ITO["blue"], alpha=0.20)
    ax_b.plot(times_ms, f_mean, color=OKABE_ITO["blue"], lw=1.6, ls="-.", label="Frequent |IG| (Mean ± 95% CI, 39 subs)")

    pk_ms = float(l1["grand_peak_latency_ms"])
    pk_val = float(t_mean[int(np.argmin(np.abs(times_ms - pk_ms)))])
    ratio_pb = float(l1["p300_to_baseline_ratio"]["mean"])
    ax_b.annotate(
        f"Peak XAI @ {pk_ms:.0f} ms\n(P300/Baseline = {ratio_pb:.2f}×)",
        xy=(pk_ms, pk_val),
        xytext=(pk_ms + 70, y_top * 0.66),
        fontsize=8.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D1D5DB", alpha=0.92),
        arrowprops=dict(arrowstyle="->", lw=0.9, color=OKABE_ITO["black"]),
    )

    ax_b.set_xlim(-200, 800)
    ax_b.set_ylim(0, y_top)
    ax_b.set_xlabel("Time relative to stimulus (ms)")
    ax_b.set_ylabel("Temporal Attribution A(t) = mean(|IG|)")
    ax_b.set_title("(d) Temporal Attribution Profile Across Physiological Windows (Layer 1)")
    ax_b.legend(loc="upper left", bbox_to_anchor=(0.01, 0.78), frameon=True, framealpha=0.9, edgecolor="none")
    ax_b.grid(True, alpha=0.25)

    return _save_figure(fig, out_dir, "fig4_xai_spatiotemporal")


# ==============================================================================
# FIGURE 5 — LAYER 2 (EOG) & LAYER 3 (MOTOR CONFOUND + EARLY CUTOFF)
# ==============================================================================
def plot_fig5_xai_eog_and_motor_confound(
    cache: Dict[str, Any],
    summary: Dict[str, Any],
    info: mne.Info,
    channel_names: Sequence[str],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Fig 5: (a) EOG clean vs no_ica; (b) RT-stratified XAI; (c) ROI Motor vs P300 & Early Cutoff."""
    setup_publication_style()
    times_ms = cache["times_ms"].astype(np.float64)
    roi_idx_map = verify_rois(channel_names)
    eog_idx = roi_idx_map["eog"]

    l2 = summary["layer2_eog"]
    l3 = summary["layer3_motor_and_rt"]

    fig = plt.figure(figsize=(13.6, 9.6))
    gs_outer = gridspec.GridSpec(3, 1, height_ratios=[0.95, 1.05, 1.0], hspace=0.40, figure=fig)

    # ------------------------------------------------------------------
    # PANEL (a): EOG Artifact Check (Clean vs No-ICA)
    # ------------------------------------------------------------------
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[1.25, 0.85, 0.85], subplot_spec=gs_outer[0], wspace=0.26)
    ax_a1 = fig.add_subplot(gs_a[0, 0])

    clean_sub = cache["cs5_clean_per_sub_abs"].astype(np.float64)
    noica_matched_sub = cache["cs5_noica_per_sub_abs_matched"].astype(np.float64)
    c_share = clean_sub[:, eog_idx, :].sum(axis=(1, 2)) / np.maximum(clean_sub.sum(axis=(1, 2)), 1e-15) * 100.0
    n_share = noica_matched_sub[:, eog_idx, :].sum(axis=(1, 2)) / np.maximum(noica_matched_sub.sum(axis=(1, 2)), 1e-15) * 100.0

    bp_a = ax_a1.boxplot(
        [n_share, c_share],
        positions=[0, 1],
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=1.6),
    )
    for patch, col in zip(bp_a["boxes"], [OKABE_ITO["red"], OKABE_ITO["green"]]):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    for xv, yv, col in ((0, n_share, OKABE_ITO["red"]), (1, c_share, OKABE_ITO["green"])):
        ax_a1.scatter(np.full_like(yv, xv) + np.linspace(-0.08, 0.08, len(yv)), yv, s=13, color=col, alpha=0.6, zorder=3)
    for y_n, y_c in zip(n_share, c_share):
        ax_a1.plot([0, 1], [y_n, y_c], color=OKABE_ITO["grey"], alpha=0.22, lw=0.7)

    ax_a1.axhline(l2["uniform_baseline_share_pct"], color=OKABE_ITO["black"], ls="--", lw=1.0, label=f"Uniform chance ({l2['uniform_baseline_share_pct']:.2f}%)")
    p_eog = l2["matched_1to1_subject_level"]["wilcoxon_share_full"]["wilcoxon_p"]
    y_top_a = max(float(np.max(n_share)), float(np.max(c_share)))
    _add_sig_bracket(ax_a1, 0, 1, y_top_a * 1.05, y_top_a * 0.04, f"{_p_to_stars(p_eog)} ({_fmt_p_text(p_eog)})", fontsize=8.0)

    ax_a1.set_xticks([0, 1])
    ax_a1.set_xticklabels(["No-ICA\n(Matched 1-1)", "Clean (ICA)\n(Matched 1-1)"])
    ax_a1.set_ylim(0, y_top_a * 1.38)
    ax_a1.set_ylabel("EOG_ROI |IG| Share (%)")
    ax_a1.set_title("(a1) Frontal EOG Attribution Share (39 Subjects, CS5)")
    ax_a1.legend(loc="upper left", frameon=True, framealpha=0.92, edgecolor="none", fontsize=7.6)
    ax_a1.grid(True, axis="y", alpha=0.25)

    # Topomap No-ICA vs Clean (Difference No-ICA - Clean)
    topo_noica = noica_matched_sub.mean(axis=(0, 2))
    topo_clean = clean_sub.mean(axis=(0, 2))
    topo_diff_eog = topo_noica - topo_clean
    vmax_eog = float(np.max(np.abs(topo_diff_eog)))
    eog_mask = np.isin(np.arange(127), eog_idx)

    ax_a2 = fig.add_subplot(gs_a[0, 1])
    im_n = _plot_topomap_clean(topo_noica, info, ax_a2, cmap="viridis", vlim=(0, float(max(np.max(topo_noica), np.max(topo_clean)))), mask=eog_mask)
    fig.colorbar(im_n, ax=ax_a2, fraction=0.046, pad=0.04)
    ax_a2.set_title("(a2) No-ICA Mean |IG|\n(white dots = EOG_ROI)", fontsize=9.2)

    ax_a3 = fig.add_subplot(gs_a[0, 2])
    im_d = _plot_topomap_clean(topo_diff_eog, info, ax_a3, cmap="RdBu_r", vlim=(-vmax_eog, vmax_eog), mask=eog_mask)
    fig.colorbar(im_d, ax=ax_a3, fraction=0.046, pad=0.04)
    ax_a3.set_title("(a3) Difference (No-ICA − Clean)\nFrontal Artifact Attribution", fontsize=9.2)

    # ------------------------------------------------------------------
    # PANEL (b): RT-Stratified XAI (Fast < 450 ms vs Slow > 550 ms)
    # ------------------------------------------------------------------
    gs_b = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[1.45, 0.78, 0.78], subplot_spec=gs_outer[1], wspace=0.24)
    ax_b1 = fig.add_subplot(gs_b[0, 0])

    fast_d = l3["rt_stratified"]["fast"]
    mid_d = l3["rt_stratified"]["mid"]
    slow_d = l3["rt_stratified"]["slow"]

    f_curve = np.asarray(fast_d["temporal_curve_mean"], dtype=np.float64)
    f_ci = np.asarray(fast_d["temporal_curve_ci95"], dtype=np.float64)
    s_curve = np.asarray(slow_d["temporal_curve_mean"], dtype=np.float64)
    s_ci = np.asarray(slow_d["temporal_curve_ci95"], dtype=np.float64)
    m_curve = np.asarray(mid_d["temporal_curve_mean"], dtype=np.float64)

    ax_b1.axvspan(300, 600, color=OKABE_ITO["yellow"], alpha=0.13)
    ax_b1.axvline(0, color=OKABE_ITO["black"], ls="--", lw=0.9)
    ax_b1.fill_between(times_ms, f_curve - f_ci, f_curve + f_ci, color=OKABE_ITO["green"], alpha=0.22)
    ax_b1.plot(times_ms, f_curve, color=OKABE_ITO["green"], lw=2.0, label=f"Fast RT <450 ms (N={fast_d['n_trials']}, Peak={fast_d['peak_xai_latency_ms']:.0f} ms)")
    ax_b1.plot(times_ms, m_curve, color=OKABE_ITO["blue"], lw=1.4, ls=":", alpha=0.85, label=f"Mid RT 450–550 ms (N={mid_d['n_trials']}, Peak={mid_d['peak_xai_latency_ms']:.0f} ms)")
    ax_b1.fill_between(times_ms, s_curve - s_ci, s_curve + s_ci, color=OKABE_ITO["orange"], alpha=0.22)
    ax_b1.plot(times_ms, s_curve, color=OKABE_ITO["orange"], lw=2.0, label=f"Slow RT >550 ms (N={slow_d['n_trials']}, Peak={slow_d['peak_xai_latency_ms']:.0f} ms)")

    ax_b1.axvline(fast_d["rt_ms"]["mean"], color=OKABE_ITO["green"], ls="--", lw=1.1, alpha=0.8)
    ax_b1.axvline(slow_d["rt_ms"]["mean"], color=OKABE_ITO["orange"], ls="--", lw=1.1, alpha=0.8)
    ax_b1.set_xlim(-200, 800)
    ax_b1.set_xlabel("Time relative to stimulus (ms)")
    ax_b1.set_ylabel("Temporal |IG| A(t)")
    ax_b1.set_title("(b1) RT-Stratified Temporal Attribution (Dashed = Mean RT)")
    ax_b1.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.6)
    ax_b1.grid(True, alpha=0.25)

    t_fast_35 = np.asarray(fast_d["topo_300_500"], dtype=np.float64)
    t_slow_35 = np.asarray(slow_d["topo_300_500"], dtype=np.float64)
    vmax_rt = float(max(np.max(t_fast_35), np.max(t_slow_35)))
    motor_mask = np.isin(np.arange(127), roi_idx_map["left_motor"] + roi_idx_map["p300"])

    ax_b2 = fig.add_subplot(gs_b[0, 1])
    im_f = _plot_topomap_clean(t_fast_35, info, ax_b2, cmap="viridis", vlim=(0, vmax_rt), mask=motor_mask)
    fig.colorbar(im_f, ax=ax_b2, fraction=0.046, pad=0.04)
    ax_b2.set_title(
        f"(b2) Fast RT (<450 ms) 300–500 ms\n"
        f"P300/L-Mot={fast_d['roi_300_600']['p300_to_left_motor_ratio']:.2f}× | Lat={fast_d['roi_300_600']['motor_laterality_mean']:+.3f}",
        fontsize=8.6,
    )

    ax_b3 = fig.add_subplot(gs_b[0, 2])
    im_s = _plot_topomap_clean(t_slow_35, info, ax_b3, cmap="viridis", vlim=(0, vmax_rt), mask=motor_mask)
    fig.colorbar(im_s, ax=ax_b3, fraction=0.046, pad=0.04)
    ax_b3.set_title(
        f"(b3) Slow RT (>550 ms) 300–500 ms\n"
        f"P300/L-Mot={slow_d['roi_300_600']['p300_to_left_motor_ratio']:.2f}× | Lat={slow_d['roi_300_600']['motor_laterality_mean']:+.3f}",
        fontsize=8.6,
    )

    # ------------------------------------------------------------------
    # PANEL (c): Spatial ROI Density (P300 vs Left/Right Motor) & Early-Window Cutoff
    # ------------------------------------------------------------------
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[1.0, 1.15], subplot_spec=gs_outer[2], wspace=0.25)
    ax_c1 = fig.add_subplot(gs_c[0, 0])

    abs_3d = cache["loso_target_abs_ig"].astype(np.float64)
    mask_300_600 = (times_ms >= 300.0 - 1e-6) & (times_ms <= 600.0 + 1e-6)
    p300_vals = abs_3d[:, roi_idx_map["p300"], :][:, :, mask_300_600].mean(axis=(1, 2))
    lm_vals = abs_3d[:, roi_idx_map["left_motor"], :][:, :, mask_300_600].mean(axis=(1, 2))
    rm_vals = abs_3d[:, roi_idx_map["right_motor"], :][:, :, mask_300_600].mean(axis=(1, 2))

    bp_c = ax_c1.boxplot(
        [p300_vals, lm_vals, rm_vals],
        positions=[0, 1, 2],
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="white", lw=1.6),
    )
    roi_cols = [OKABE_ITO["blue"], OKABE_ITO["orange"], OKABE_ITO["sky"]]
    for patch, col in zip(bp_c["boxes"], roi_cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.78)
    for xv, yv, col in zip([0, 1, 2], [p300_vals, lm_vals, rm_vals], roi_cols):
        ax_c1.scatter(np.full_like(yv, xv) + np.linspace(-0.08, 0.08, len(yv)), yv, s=12, color=col, alpha=0.6, zorder=3)

    r36 = l3["roi_300_600"]
    p_pl = r36["wilcoxon_p300_vs_left_motor"]["wilcoxon_p"]
    p_lr = r36["wilcoxon_left_vs_right_motor"]["wilcoxon_p"]
    y_max_c1 = max(float(np.max(p300_vals)), float(np.max(lm_vals)), float(np.max(rm_vals)))
    _add_sig_bracket(ax_c1, 0, 1, y_max_c1 * 1.04, y_max_c1 * 0.03, f"Ratio={r36['p300_to_left_motor_ratio']['mean']:.2f}× ({_p_to_stars(p_pl)})", fontsize=7.6)
    _add_sig_bracket(ax_c1, 1, 2, y_max_c1 * 1.15, y_max_c1 * 0.03, f"Lat={r36['motor_laterality']['mean']:+.3f} ({_p_to_stars(p_lr)})", fontsize=7.6)

    ax_c1.set_xticks([0, 1, 2])
    ax_c1.set_xticklabels(["P300_ROI\n(7 chs)", "LEFT_MOTOR\n(6 chs)", "RIGHT_MOTOR\n(6 chs)"])
    ax_c1.set_ylim(0, y_max_c1 * 1.28)
    ax_c1.set_ylabel("Mean |IG| per channel (300–600 ms)")
    ax_c1.set_title("(c1) Spatial ROI Attribution Density (39 Subjects)")
    ax_c1.grid(True, axis="y", alpha=0.25)

    # Panel (c2): Early-Window Cutoff
    ax_c2 = fig.add_subplot(gs_c[0, 1])
    ew = l3.get("early_window_cutoff")
    if ew is not None:
        cs5_c = ew["cs5_comparison"]
        loso_c = ew["loso_inference_masking"]["comparison_vs_orig"]
        cats = ["ROC-AUC", "PR-AUC", "Balanced Acc"]
        keys = ["roc_auc", "pr_auc", "balanced_accuracy"]
        x_ew = np.arange(len(cats))
        bw = 0.21

        v_cs5_orig = [cs5_c[k]["orig_mean"] for k in keys]
        v_cs5_mask = [cs5_c[k]["inference_masked_mean"] for k in keys]
        v_cs5_ret = [cs5_c[k]["retrain_early_mean"] for k in keys]
        v_loso_mask = [loso_c[k]["masked_mean"] for k in keys]

        b1 = ax_c2.bar(x_ew - 1.5 * bw, v_cs5_orig, width=bw, color=OKABE_ITO["black"], alpha=0.82, label="CS5 Full [-0.2, 0.8]s")
        b2 = ax_c2.bar(x_ew - 0.5 * bw, v_cs5_mask, width=bw, color=OKABE_ITO["grey"], alpha=0.82, label="CS5 Mask >450ms")
        b3 = ax_c2.bar(x_ew + 0.5 * bw, v_cs5_ret, width=bw, color=OKABE_ITO["green"], alpha=0.88, label="CS5 Retrain [-0.2, 0.45]s")
        b4 = ax_c2.bar(x_ew + 1.5 * bw, v_loso_mask, width=bw, color=OKABE_ITO["sky"], alpha=0.88, label="LOSO Mask >450ms")

        for bars in (b1, b2, b3, b4):
            for bar in bars:
                val = bar.get_height()
                ax_c2.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.3f}", ha="center", va="bottom", fontsize=7.0, rotation=90)

        ax_c2.set_xticks(x_ew)
        ax_c2.set_xticklabels(cats)
        ax_c2.set_ylim(0.28, 1.14)
        ax_c2.set_ylabel("Score")
        ax_c2.set_title("(c2) Early-Window Cutoff (<= 450 ms, Pre-Median-RT)")
        ax_c2.legend(loc="lower center", frameon=True, framealpha=0.94, edgecolor="none", fontsize=7.6, ncol=2)
        ax_c2.grid(True, axis="y", alpha=0.25)

    return _save_figure(fig, out_dir, "fig5_xai_eog_and_motor_confound")


# ==============================================================================
# FIGURE 6 — LAYER 4 (FREQUENCY OCCLUSION & ERROR ANALYSIS)
# ==============================================================================
def plot_fig6_frequency_and_error_analysis(
    cache: Dict[str, Any],
    summary: Dict[str, Any],
    out_dir: Path,
) -> Tuple[Path, Path]:
    """Fig 6: (a) Frequency Occlusion ΔROC/ΔPR; (b) Error Analysis TP vs FN & RT vs ROC-AUC."""
    setup_publication_style()
    times_ms = cache["times_ms"].astype(np.float64)
    l4 = summary["layer4_frequency_and_neuro"]
    l3 = summary["layer3_motor_and_rt"]
    fo = l4["frequency_occlusion"]
    ea = l4["error_analysis"]

    fig = plt.figure(figsize=(13.2, 5.4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.25], wspace=0.25, figure=fig)

    # Panel (a): Frequency Occlusion Barplot
    ax1 = fig.add_subplot(gs[0])
    band_order = [b for b in ("sub_delta", "delta", "theta", "alpha", "beta", "gamma", "delta_theta") if b in fo]
    short_labels = {
        "sub_delta": "Sub-Delta\n(<1 Hz)",
        "delta": "Delta\n(1–4 Hz)",
        "theta": "Theta\n(4–8 Hz)",
        "alpha": "Alpha\n(8–13 Hz)",
        "beta": "Beta\n(13–30 Hz)",
        "gamma": "Gamma\n(30–40 Hz)",
        "delta_theta": "Delta+Theta\n(1–8 Hz)",
    }
    x_b = np.arange(len(band_order))
    bw = 0.36
    d_roc_m = np.array([fo[b]["delta_vs_orig"]["roc_auc"]["delta_mean"] for b in band_order])
    d_roc_s = np.array([fo[b]["delta_vs_orig"]["roc_auc"]["delta_std"] for b in band_order])
    p_roc = [fo[b]["delta_vs_orig"]["roc_auc"]["wilcoxon_p"] for b in band_order]

    d_pr_m = np.array([fo[b]["delta_vs_orig"]["pr_auc"]["delta_mean"] for b in band_order])
    d_pr_s = np.array([fo[b]["delta_vs_orig"]["pr_auc"]["delta_std"] for b in band_order])

    ax1.axhline(0.0, color=OKABE_ITO["black"], lw=0.9)
    ax1.bar(x_b - bw / 2, d_roc_m, yerr=d_roc_s, width=bw, color=OKABE_ITO["blue"], alpha=0.85, capsize=3, label="ΔROC-AUC (Mean ± SD)")
    ax1.bar(x_b + bw / 2, d_pr_m, yerr=d_pr_s, width=bw, color=OKABE_ITO["orange"], alpha=0.82, capsize=3, label="ΔPR-AUC (Mean ± SD)")

    for i, (rm, rs, pm, ps, pv) in enumerate(zip(d_roc_m, d_roc_s, d_pr_m, d_pr_s, p_roc)):
        star = _p_to_stars(pv)
        y_pos = min(rm - rs, pm - ps) - 0.012
        ax1.text(x_b[i], y_pos, f"{star}\n{rm:+.3f}", ha="center", va="top", fontsize=7.2, color=OKABE_ITO["black"])

    y_low = float(min(np.min(d_roc_m - d_roc_s), np.min(d_pr_m - d_pr_s))) - 0.065
    y_high = max(0.03, float(max(np.max(d_roc_m + d_roc_s), np.max(d_pr_m + d_pr_s))) + 0.025)
    ax1.set_xticks(x_b)
    ax1.set_xticklabels([short_labels[b] for b in band_order], fontsize=7.6)
    ax1.set_ylim(y_low, y_high)
    ax1.set_ylabel("Performance Drop (Occluded − Original)")
    ax1.set_title("(a) Frequency Occlusion Sensitivity (39 LOSO Subjects)")
    ax1.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="none")
    ax1.grid(True, axis="y", alpha=0.25)

    # Panel (b): 2 sub-panels stacked vertically (TP vs FN ERP tại Pz + Scatter Mean RT vs LOSO ROC-AUC)
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace=0.40)
    ax2_top = fig.add_subplot(gs_b[0])
    ax2_bot = fig.add_subplot(gs_b[1])

    curves = ea["curves"]
    tp_m = np.asarray(curves["tp_pz_mean_uv"], dtype=np.float64)
    tp_c = np.asarray(curves["tp_pz_ci95_uv"], dtype=np.float64)
    fn_m = np.asarray(curves["fn_pz_mean_uv"], dtype=np.float64)
    fn_c = np.asarray(curves["fn_pz_ci95_uv"], dtype=np.float64)

    ax2_top.axvspan(300, 600, color=OKABE_ITO["yellow"], alpha=0.14)
    ax2_top.axvline(0, color=OKABE_ITO["black"], ls="--", lw=0.9)
    ax2_top.axhline(0, color=OKABE_ITO["grey"], ls=":", lw=0.8)
    ax2_top.fill_between(times_ms, tp_m - tp_c, tp_m + tp_c, color=OKABE_ITO["green"], alpha=0.22)
    ax2_top.plot(times_ms, tp_m, color=OKABE_ITO["green"], lw=2.0, label=f"True Positive (N={ea['n_tp']:,}, Pz={ea['tp_pz_p300_amp_uv']['mean']:+.2f} µV)")
    ax2_top.fill_between(times_ms, fn_m - fn_c, fn_m + fn_c, color=OKABE_ITO["red"], alpha=0.22)
    ax2_top.plot(times_ms, fn_m, color=OKABE_ITO["red"], lw=2.0, ls="--", label=f"False Negative (N={ea['n_fn']:,}, Pz={ea['fn_pz_p300_amp_uv']['mean']:+.2f} µV)")
    ax2_top.set_xlim(-200, 800)
    ax2_top.set_ylabel("Pz ERP (µV)")
    ax2_top.set_xlabel("Time relative to stimulus (ms)")
    ax2_top.set_title(f"(b1) Error Analysis: Pz ERP of TP vs FN Targets ({_fmt_p_text(ea['pz_p300_amp_mannwhitney_p'])})")
    ax2_top.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.6)
    ax2_top.grid(True, alpha=0.25)

    # Scatter Mean RT vs LOSO ROC-AUC
    rt_df = pd.DataFrame(l3["rt_vs_performance_correlation"]["per_subject_rt_perf"])
    bot5_set = set(BOTTOM5_LOSO_SUBJECTS)
    is_bot5 = rt_df["subject_id"].isin(bot5_set)

    ax2_bot.scatter(
        rt_df.loc[~is_bot5, "mean_rt_ms"],
        rt_df.loc[~is_bot5, "loso_roc_auc"],
        color=OKABE_ITO["blue"],
        alpha=0.78,
        s=32,
        label="Other 34 subjects",
    )
    ax2_bot.scatter(
        rt_df.loc[is_bot5, "mean_rt_ms"],
        rt_df.loc[is_bot5, "loso_roc_auc"],
        color=OKABE_ITO["red"],
        s=54,
        marker="D",
        zorder=4,
        label="Bottom-5 LOSO subjects",
    )

    # Trendline
    xs_rt = rt_df["mean_rt_ms"].to_numpy(dtype=np.float64)
    ys_auc = rt_df["loso_roc_auc"].to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(xs_rt, ys_auc, 1)
    x_line = np.linspace(float(np.min(xs_rt)) - 10, float(np.max(xs_rt)) + 10, 100)
    ax2_bot.plot(x_line, slope * x_line + intercept, color=OKABE_ITO["black"], ls="--", lw=1.2, alpha=0.8)

    offsets = {
        "sub-07": (-42, 0.007),
        "sub-01": (-44, -0.012),
        "sub-41": (6, -0.005),
        "sub-32": (-42, -0.014),
        "sub-21": (-48, 0.003),
    }
    for _, row in rt_df.loc[is_bot5].iterrows():
        s = str(row["subject_id"])
        dx, dy = offsets.get(s, (5, 0.004))
        ax2_bot.annotate(
            s,
            xy=(float(row["mean_rt_ms"]), float(row["loso_roc_auc"])),
            xytext=(float(row["mean_rt_ms"]) + dx, float(row["loso_roc_auc"]) + dy),
            fontsize=7.6,
            fontweight="bold",
            color=OKABE_ITO["red"],
        )

    c_rt = l3["rt_vs_performance_correlation"]["mean_rt_vs_loso_roc_auc"]
    ax2_bot.set_xlabel("Subject Mean Reaction Time (ms)")
    ax2_bot.set_ylabel("LOSO ROC-AUC")
    ax2_bot.set_title(
        f"(b2) Mean RT vs LOSO ROC-AUC (Spearman ρ = {c_rt['spearman_rho']:+.3f}, {_fmt_p_text(c_rt['spearman_p'])})"
    )
    ax2_bot.legend(loc="lower left", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.6)
    ax2_bot.grid(True, alpha=0.25)

    return _save_figure(fig, out_dir, "fig6_frequency_and_error_analysis")


# ==============================================================================
# ĐIỀU PHỐI XUẤT TRỌN BỘ 6 BIỂU ĐỒ
# ==============================================================================
def generate_all_paper_figures(
    data_dir: Union[str, Path] = Path("data/processed"),
    metrics_dir: Union[str, Path] = Path("results/metrics/dl2"),
    figures_dir: Union[str, Path] = Path("results/figures/paper"),
    external_results_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Tuple[Path, Path]]:
    """Tạo đủ 6 hình chuẩn bài báo (300 DPI PNG + PDF) vào `figures_dir`."""
    data_root = Path(data_dir)
    met_root = Path(metrics_dir)
    fig_root = Path(figures_dir)
    ext_root = Path(external_results_dir) if external_results_dir is not None else None

    channel_names = load_channel_names(data_root)
    info = create_colin27_info(channel_names, sfreq=SFREQ, montage_name=MONTAGE_NAME)

    cache = np.load(met_root / "xai_loso_cache.npz", allow_pickle=False)
    summary = json.loads((met_root / "xai_rq3_summary.json").read_text(encoding="utf-8"))

    outputs = {
        "fig1": plot_fig1_erp_overview(cache, info, channel_names, fig_root),
        "fig2": plot_fig2_single_trial_and_rt(data_root, channel_names, fig_root),
        "fig3": plot_fig3_benchmark_rq1_rq2(data_root, met_root, fig_root, external_results_dir=ext_root),
        "fig4": plot_fig4_xai_spatiotemporal(cache, summary, info, fig_root),
        "fig5": plot_fig5_xai_eog_and_motor_confound(cache, summary, info, channel_names, fig_root),
        "fig6": plot_fig6_frequency_and_error_analysis(cache, summary, fig_root),
    }
    fig3b_paths = plot_fig3b_roc_pr_curves(met_root, fig_root, external_results_dir=ext_root)
    for k, (png_p, pdf_p) in outputs.items():
        print(f"[Figures] Saved {k}: {png_p} & {pdf_p}")
    print(f"[Figures] Saved fig3b: {fig3b_paths[0]} & {fig3b_paths[1]}")
    return outputs
