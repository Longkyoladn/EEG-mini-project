"""Pipeline Explainable AI (RQ3 — 4 lớp kiểm chứng) cho P300-ATCNet.

Triển khai toàn bộ 4 lớp kiểm chứng giải thích mô hình (XAI) chuẩn Zero Leakage:
  - Lớp 1: Chiều thời gian & Sanity Check (Baseline [-200, 0) ms vs P300 [300, 600] ms),
           đối chiếu baseline trung bình lớp Frequent với baseline = 0, và trích xuất
           trọng số không gian Depthwise Spatial Conv (`get_spatial_weights()`).
  - Lớp 2: Kiểm chứng nhiễu chớp mắt EOG (`EOG_ROI` trên `clean` vs `no_ica` qua 5 folds
           Cross-Subject, cả tập khớp 1-1 `kept_in_clean == True` và toàn tập, kèm kiểm tra
           riêng 2 ca nội suy `Fp1/Fp2` là `sub-18` và `sub-40`).
  - Lớp 3: Bóc tách nhiễu vận động bấm tay (Motor Confound):
           (a) Tỷ số mật độ ROI không gian (`P300_ROI` vs `LEFT_MOTOR_ROI` vs `RIGHT_MOTOR_ROI`)
               và chỉ số lệch bán cầu `Motor Laterality`;
           (b) Phân tầng Reaction Time (`Fast < 450 ms`, `Mid 450–550 ms`, `Slow > 550 ms`) +
               tương quan Spearman giữa `subject_mean_rt_ms` và `LOSO ROC-AUC`;
           (c) Kiểm chứng cắt cửa sổ sớm (`Early-Window Cutoff <= 450 ms`): cả inference-only
               masking trên 39 checkpoints LOSO và retrain-from-scratch 5-fold Cross-Subject
               trên `[-0.2, 0.45] s` (`n_times=163`).
  - Lớp 4: Chiều tần số & đối chiếu thần kinh học cổ điển:
           (a) Frequency Occlusion (triệt tiêu từng dải tần bằng Butterworth SOS bậc 4);
           (b) Spatial Correlation (Pearson `r`, Spearman `ρ`) giữa Captum IG và
               Grand Average ERP (`Target − Frequent`) tại `300–500 ms` và `300–600 ms`;
           (c) Error Analysis (`True Positive` vs `False Negative`) & cụm Bottom-5 LOSO.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from scipy import signal, stats

from src.evaluation.metrics import (
    METRIC_NAMES,
    PREDICTION_COLUMNS,
    aggregate_metrics,
    compute_binary_metrics,
    find_optimal_threshold,
    save_metrics_json,
    save_predictions_csv,
)
from src.training.dataset_dl2 import (
    get_protocol_folds,
    load_eeg_dataset,
    natural_sort_key,
)
from src.training.trainer_dl2 import (
    load_checkpoint_for_eval,
    train_single_fold,
)

# ==============================================================================
# HẰNG SỐ CHUẨN HÓA KHÔNG GIAN, THỜI GIAN & TẦN SỐ (RQ3)
# ==============================================================================
MONTAGE_NAME: str = "colin27_1005"
SFREQ: float = 250.0
EARLY_CUTOFF_MS: float = 450.0

EOG_ROI: Tuple[str, ...] = ("Fp1", "Fp2", "AF7", "AF3", "AF8", "AF4", "AFp1", "AFp2")
P300_ROI: Tuple[str, ...] = ("Pz", "CPz", "Cz", "P1", "P2", "CP1", "CP2")
LEFT_MOTOR_ROI: Tuple[str, ...] = ("C3", "CP3", "FCC3h", "CCP3h", "CCP1h", "FC3")
RIGHT_MOTOR_ROI: Tuple[str, ...] = ("C4", "CP4", "FCC4h", "CCP4h", "CCP2h", "FC4")

ROI_CHANNELS: Dict[str, Tuple[str, ...]] = {
    "eog": EOG_ROI,
    "p300": P300_ROI,
    "left_motor": LEFT_MOTOR_ROI,
    "right_motor": RIGHT_MOTOR_ROI,
}

# 4 cửa sổ sinh lý (ms): [-200, 0), [0, 300), [300, 600], (600, 800]
TIME_WINDOWS_ORDER: Tuple[str, ...] = (
    "baseline",
    "early_visual",
    "p300",
    "late_post_response",
)

TIME_WINDOWS_LABELS: Dict[str, str] = {
    "baseline": "[-200, 0) ms",
    "early_visual": "[0, 300) ms",
    "p300": "[300, 600] ms",
    "late_post_response": "(600, 800] ms",
}

FREQUENCY_BANDS: Dict[str, Tuple[float, float]] = {
    "sub_delta": (0.0, 1.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 40.0),
    "delta_theta": (1.0, 8.0),
}

FREQUENCY_BAND_LABELS: Dict[str, str] = {
    "sub_delta": "Sub-Delta (<1 Hz)",
    "delta": "Delta (1–4 Hz)",
    "theta": "Theta (4–8 Hz)",
    "alpha": "Alpha (8–13 Hz)",
    "beta": "Beta (13–30 Hz)",
    "gamma": "Gamma (30–40 Hz)",
    "delta_theta": "Delta+Theta (1–8 Hz)",
}

BOTTOM5_LOSO_SUBJECTS: Tuple[str, ...] = ("sub-07", "sub-01", "sub-41", "sub-32", "sub-21")
FP1_FP2_INTERP_SUBJECTS: Tuple[str, ...] = ("sub-18", "sub-40")

EARLY_CUTOFF_ATCNET_KWARGS: Dict[str, Any] = {
    "n_channels": 127,
    "n_times": 163,
    "n_classes": 1,
    "readout": "attn_pool",
    "n_windows": 2,
    "shared_branch": False,
    "tcn_filters": 32,
    "temp_kernel_1": 125,
    "spatial_max_norm": 2.0,
    "cv_dropout": 0.3,
    "at_dropout": 0.3,
    "tcn_dropout": 0.3,
}


# ==============================================================================
# TIỆN ÍCH MONTAGE, ROI, THỐNG KÊ & SERIALIZATION
# ==============================================================================
def load_channel_names(data_dir: Union[str, Path]) -> List[str]:
    """Đọc danh sách 127 tên kênh chuẩn từ `channel_names.json`."""
    ch_path = Path(data_dir) / "channel_names.json"
    if not ch_path.exists():
        raise FileNotFoundError(f"Không tìm thấy channel_names.json tại: {ch_path.resolve()}")
    with open(ch_path, "r", encoding="utf-8") as f:
        ch_names = json.load(f)
    if len(ch_names) != 127 or len(set(ch_names)) != 127:
        raise ValueError(f"channel_names.json phải có đúng 127 kênh duy nhất, nhận {len(ch_names)}.")
    return [str(c) for c in ch_names]


def create_colin27_info(
    channel_names: Sequence[str],
    sfreq: float = SFREQ,
    montage_name: str = MONTAGE_NAME,
) -> mne.Info:
    """Tạo `mne.Info` và gán montage `colin27_1005` với `on_missing='raise'`."""
    ch_list = [str(c) for c in channel_names]
    info = mne.create_info(ch_names=ch_list, sfreq=float(sfreq), ch_types="eeg")
    montage = mne.channels.make_standard_montage(montage_name)
    info.set_montage(montage, on_missing="raise", verbose=False)

    missing_coords = []
    for ch in info["chs"]:
        pos = ch["loc"][:3]
        if not np.all(np.isfinite(pos)) or float(np.linalg.norm(pos)) < 1e-10:
            missing_coords.append(ch["ch_name"])
    if missing_coords:
        raise ValueError(
            f"Phát hiện {len(missing_coords)} kênh thiếu tọa độ 3D trong {montage_name}: {missing_coords}"
        )
    return info


def verify_rois(channel_names: Sequence[str]) -> Dict[str, List[int]]:
    """Xác thực toàn bộ kênh của 4 ROI nằm trong `channel_names` và trả về index."""
    ch_to_idx = {str(ch): idx for idx, ch in enumerate(channel_names)}
    roi_indices: Dict[str, List[int]] = {}
    for roi_name, roi_chs in ROI_CHANNELS.items():
        missing = [ch for ch in roi_chs if ch not in ch_to_idx]
        if missing:
            raise ValueError(f"ROI '{roi_name}' thiếu các kênh trong channel_names.json: {missing}")
        roi_indices[roi_name] = [ch_to_idx[ch] for ch in roi_chs]
    return roi_indices


def get_time_window_masks(times_ms: np.ndarray) -> Dict[str, np.ndarray]:
    """Trả về boolean mask theo 4 cửa sổ sinh lý chuẩn của Bước 6."""
    t = np.asarray(times_ms, dtype=np.float64)
    return {
        "baseline": (t >= -200.0 - 1e-6) & (t < 0.0 - 1e-6),
        "early_visual": (t >= 0.0 - 1e-6) & (t < 300.0 - 1e-6),
        "p300": (t >= 300.0 - 1e-6) & (t <= 600.0 + 1e-6),
        "late_post_response": (t > 600.0 + 1e-6) & (t <= 800.0 + 1e-6),
    }


def _sanitize_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return _sanitize_json(obj.tolist())
    return obj


def _mean_std_ci95(arr: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính Mean, SD (ddof=1) và bán kính khoảng tin cậy 95% theo phân phối Student-t."""
    a = np.asarray(arr, dtype=np.float64)
    n = a.shape[axis]
    mean = np.mean(a, axis=axis)
    if n <= 1:
        zeros = np.zeros_like(mean)
        return mean, zeros, zeros
    std = np.std(a, axis=axis, ddof=1)
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci95 = t_crit * std / math.sqrt(n)
    return mean, std, ci95


def _scalar_summary(vals: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(vals, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None, "median": None, "q25": None, "q75": None}
    mean, std, ci95 = _mean_std_ci95(a, axis=0)
    return {
        "n": int(a.size),
        "mean": float(mean),
        "std": float(std),
        "ci95": float(ci95),
        "median": float(np.median(a)),
        "q25": float(np.percentile(a, 25.0)),
        "q75": float(np.percentile(a, 75.0)),
    }


def _paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Kiểm định Wilcoxon signed-rank ghép cặp 2 phía kèm Paired t-test đối chiếu."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[mask], vb[mask]
    diff = va - vb
    n = int(diff.size)
    if n == 0:
        return {"n": 0, "mean_diff": None, "wilcoxon_stat": None, "wilcoxon_p": None, "ttest_p": None}
    if np.allclose(diff, 0.0, atol=1e-15):
        return {"n": n, "mean_diff": 0.0, "wilcoxon_stat": 0.0, "wilcoxon_p": 1.0, "ttest_p": 1.0}
    w_res = stats.wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
    t_res = stats.ttest_rel(va, vb)
    return {
        "n": n,
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1 if n > 1 else 0)),
        "median_diff": float(np.median(diff)),
        "wilcoxon_stat": float(w_res.statistic),
        "wilcoxon_p": float(w_res.pvalue),
        "ttest_p": float(t_res.pvalue),
    }


def _corr_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Tính tương quan Pearson r, Spearman rho và Cosine similarity trên 2 vector."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(va) & np.isfinite(vb)
    va, vb = va[mask], vb[mask]
    if va.size < 3 or np.std(va) < 1e-15 or np.std(vb) < 1e-15:
        return {
            "n": int(va.size),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "cosine_sim": None,
        }
    pr, pp = stats.pearsonr(va, vb)
    sr, sp = stats.spearmanr(va, vb)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    cos = float(np.dot(va, vb) / denom) if denom > 1e-15 else 0.0
    return {
        "n": int(va.size),
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "cosine_sim": cos,
    }


def _fmt_pval(p: Optional[float]) -> str:
    if p is None or not math.isfinite(p):
        return "NaN"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4f}"


# ==============================================================================
# BỘ LỌC TẦN SỐ & CAPTUM INTEGRATED GRADIENTS
# ==============================================================================
def apply_frequency_occlusion(
    X: np.ndarray,
    band: str,
    sfreq: float = SFREQ,
    order: int = 4,
) -> np.ndarray:
    """Triệt tiêu 1 dải tần bằng bộ lọc zero-phase Butterworth SOS (`sosfiltfilt`).

    - `sub_delta (< 1 Hz)`: dùng Butterworth highpass 1.0 Hz bậc 4.
    - Các dải `(low, high)`: dùng Butterworth bandstop `[low, high]` bậc 4.
    Bảo toàn tuyệt đối shape `(N, C, T)` và dtype `float32`, không sinh NaN/Inf.
    """
    if band not in FREQUENCY_BANDS:
        raise ValueError(f"Dải tần không hợp lệ: {band!r}. Hợp lệ: {sorted(FREQUENCY_BANDS)}")

    arr = np.asarray(X)
    if arr.ndim != 3:
        raise ValueError(f"Kỳ vọng X shape (N, C, T), nhận {arr.shape}.")
    if arr.shape[0] == 0:
        return arr.astype(np.float32, copy=True)

    low, high = FREQUENCY_BANDS[band]
    if band == "sub_delta" or low <= 0.0:
        sos = signal.butter(order, float(high), btype="highpass", fs=float(sfreq), output="sos")
    else:
        sos = signal.butter(
            order,
            [float(low), float(high)],
            btype="bandstop",
            fs=float(sfreq),
            output="sos",
        )

    filtered = signal.sosfiltfilt(sos, arr.astype(np.float64), axis=-1)
    if not np.isfinite(filtered).all():
        raise ValueError(f"Bộ lọc occlusion cho dải '{band}' sinh ra giá trị NaN/Inf.")
    return filtered.astype(np.float32)


def _resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_captum_ig(
    model: nn.Module,
    x_scaled: np.ndarray,
    baselines: np.ndarray,
    *,
    target: int = 0,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
    return_convergence_delta: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Tính Captum `IntegratedGradients(model)` theo mini-batch an toàn VRAM.

    Đảm bảo kích thước mỗi mini-batch ngoài `<= internal_batch_size` để Captum không
    phát cảnh báo và giữ mức sử dụng bộ nhớ GPU ổn định.
    """
    arr_x = np.asarray(x_scaled, dtype=np.float32)
    arr_b = np.asarray(baselines, dtype=np.float32)
    if arr_x.ndim != 3:
        raise ValueError(f"x_scaled phải có 3 chiều (N, C, T), nhận {arr_x.shape}.")

    if arr_b.ndim == 2:
        arr_b = arr_b[None, :, :]
    if arr_b.ndim != 3 or arr_b.shape[1:] != arr_x.shape[1:]:
        raise ValueError(f"baselines shape {arr_b.shape} không tương thích x_scaled {arr_x.shape}.")
    per_sample_baseline = arr_b.shape[0] == arr_x.shape[0] and arr_x.shape[0] > 1
    if not per_sample_baseline and arr_b.shape[0] != 1:
        raise ValueError(
            f"baselines.shape[0] phải bằng 1 hoặc N={arr_x.shape[0]}, nhận {arr_b.shape[0]}."
        )

    torch_device = _resolve_device(device)
    model.eval()
    model.to(torch_device)

    ig = IntegratedGradients(model)
    n_samples = int(arr_x.shape[0])
    outer_bs = max(1, min(int(internal_batch_size), 32))

    b_shared = None if per_sample_baseline else torch.from_numpy(
        np.ascontiguousarray(arr_b, dtype=np.float32)
    ).to(torch_device)

    attr_chunks: List[np.ndarray] = []
    delta_chunks: List[np.ndarray] = []

    for start in range(0, n_samples, outer_bs):
        end = min(start + outer_bs, n_samples)
        xb = torch.from_numpy(
            np.ascontiguousarray(arr_x[start:end], dtype=np.float32)
        ).to(torch_device)
        if per_sample_baseline:
            bb = torch.from_numpy(
                np.ascontiguousarray(arr_b[start:end], dtype=np.float32)
            ).to(torch_device)
        else:
            assert b_shared is not None
            bb = b_shared.expand(xb.shape[0], -1, -1)

        if return_convergence_delta:
            attr_b, delta_b = ig.attribute(
                xb,
                baselines=bb,
                target=target,
                n_steps=int(n_steps),
                internal_batch_size=int(internal_batch_size),
                return_convergence_delta=True,
            )
            delta_chunks.append(delta_b.detach().cpu().numpy().astype(np.float64))
        else:
            attr_b = ig.attribute(
                xb,
                baselines=bb,
                target=target,
                n_steps=int(n_steps),
                internal_batch_size=int(internal_batch_size),
                return_convergence_delta=False,
            )
        attr_chunks.append(attr_b.detach().cpu().numpy().astype(np.float32))

    attrs = np.concatenate(attr_chunks, axis=0)
    deltas = np.concatenate(delta_chunks, axis=0) if return_convergence_delta else None
    return attrs, deltas


def verify_ig_completeness(
    model: nn.Module,
    x_scaled: np.ndarray,
    baselines: np.ndarray,
    *,
    target: int = 0,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, float]:
    """Kiểm chứng trực tiếp định lý Completeness của Captum Integrated Gradients.

    Kiểm tra hiệu `|sum_{c,t} IG(x) - (f(x) - f(baseline))|` và so sánh với
    `convergence_delta` từ Captum.
    """
    torch_device = _resolve_device(device)
    attrs, deltas = compute_captum_ig(
        model,
        x_scaled,
        baselines,
        target=target,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        device=torch_device,
        return_convergence_delta=True,
    )
    arr_x = np.asarray(x_scaled, dtype=np.float32)
    arr_b = np.asarray(baselines, dtype=np.float32)
    if arr_b.ndim == 2:
        arr_b = arr_b[None, :, :]
    if arr_b.shape[0] == 1 and arr_x.shape[0] > 1:
        arr_b = np.broadcast_to(arr_b, arr_x.shape)

    model.eval()
    with torch.inference_mode():
        fx = model(torch.from_numpy(arr_x).to(torch_device))[:, target].cpu().numpy().astype(np.float64)
        fb = model(torch.from_numpy(np.ascontiguousarray(arr_b)).to(torch_device))[:, target].cpu().numpy().astype(np.float64)

    attr_sum = attrs.astype(np.float64).sum(axis=(1, 2))
    diff_logits = fx - fb
    manual_abs_err = np.abs(attr_sum - diff_logits)
    rel_err = manual_abs_err / np.maximum(np.abs(diff_logits), 1e-6)
    captum_abs_delta = np.abs(deltas) if deltas is not None else manual_abs_err

    return {
        "max_abs_error": float(np.max(manual_abs_err)),
        "mean_abs_error": float(np.mean(manual_abs_err)),
        "max_rel_error": float(np.max(rel_err)),
        "mean_rel_error": float(np.mean(rel_err)),
        "captum_max_abs_delta": float(np.max(captum_abs_delta)),
        "captum_mean_abs_delta": float(np.mean(captum_abs_delta)),
    }


@torch.inference_mode()
def _predict_probs_numpy(
    model: nn.Module,
    X_scaled: np.ndarray,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    n = int(X_scaled.shape[0])
    out = []
    for start in range(0, n, batch_size):
        xb = torch.from_numpy(
            np.ascontiguousarray(X_scaled[start : start + batch_size], dtype=np.float32)
        ).to(device)
        logits = model(xb).squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


# ==============================================================================
# TÍNH IG TRÊN 39 FOLDS LOSO + 5 FOLDS CS5 (CLEAN VS NO-ICA) VÀ CACHE .NPZ
# ==============================================================================
def compute_and_save_xai_cache(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    cache_path: Optional[Union[str, Path]] = None,
    *,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Chạy Captum IG trên toàn bộ 39 folds LOSO & 5 folds CS5 (clean vs no_ica) và lưu `.npz`."""
    t_start = time.perf_counter()
    torch_device = _resolve_device(device)
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    ckpt_root = out_root / "checkpoints" / "dl2"
    target_cache = Path(cache_path) if cache_path is not None else (metrics_dir / "xai_loso_cache.npz")

    channel_names = load_channel_names(data_root)
    create_colin27_info(channel_names, sfreq=SFREQ, montage_name=MONTAGE_NAME)
    verify_rois(channel_names)

    times = np.load(data_root / "times.npy").astype(np.float64)
    times_ms = np.round(times * 1000.0, 3)

    clean_bundle = load_eeg_dataset(data_root, data_variant="clean")
    loso_folds = get_protocol_folds(clean_bundle.splits, protocol="cross_subject_loso")
    if len(loso_folds) != 39:
        raise ValueError(f"Kỳ vọng 39 folds LOSO, nhận {len(loso_folds)}.")

    pred_loso_path = metrics_dir / "handover_predictions_dl2_loso.csv"
    if not pred_loso_path.exists():
        pred_loso_path = metrics_dir / "predictions_dl2_loso_p300_atcnet.csv"
    pred_loso_df = pd.read_csv(pred_loso_path)
    meta_clean = clean_bundle.metadata
    pred_aligned = (
        pred_loso_df.set_index("trial_uid")
        .loc[meta_clean["trial_uid"].astype(str).tolist()]
        .reset_index()
    )
    y_pred_all = pred_aligned["y_pred"].to_numpy(dtype=np.int64)
    y_prob_all = pred_aligned["y_prob"].to_numpy(dtype=np.float64)

    n_subs = len(loso_folds)
    n_ch, n_t = 127, 251
    rng = np.random.default_rng(seed)

    loso_subjects: List[str] = []
    loso_target_abs_ig = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_target_signed_ig = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_target_abs_ig_zero = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_target_abs_ig_testfreq = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_freq_abs_ig = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_spatial_weights_abs = np.zeros((n_subs, n_ch), dtype=np.float32)
    loso_spatial_weights_raw = np.zeros((n_subs, 32, n_ch), dtype=np.float32)

    loso_erp_target = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)
    loso_erp_freq = np.zeros((n_subs, n_ch, n_t), dtype=np.float32)

    # Mảng theo từng trial Target (N=2,850) để phân tầng RT và TP/FN
    target_trial_uids: List[str] = []
    target_trial_subs: List[str] = []
    target_trial_rt_ms: List[float] = []
    target_trial_rt_group: List[str] = []
    target_trial_y_pred: List[int] = []
    target_trial_y_prob: List[float] = []
    target_trial_pz_erp: List[np.ndarray] = []
    target_trial_temp_abs_ig: List[np.ndarray] = []
    target_trial_temp_signed_ig: List[np.ndarray] = []
    target_trial_temp_abs_ig_zero: List[np.ndarray] = []
    target_trial_ch_abs_p300: List[np.ndarray] = []
    target_trial_ch_abs_300_500: List[np.ndarray] = []
    target_trial_ch_abs_450_650: List[np.ndarray] = []
    target_trial_ch_signed_p300: List[np.ndarray] = []
    target_trial_ch_signed_300_500: List[np.ndarray] = []
    target_trial_deltas: List[np.ndarray] = []

    freq_trial_temp_abs_ig: List[np.ndarray] = []

    pz_idx = channel_names.index("Pz")
    mask_300_600 = (times_ms >= 300.0 - 1e-6) & (times_ms <= 600.0 + 1e-6)
    mask_300_500 = (times_ms >= 300.0 - 1e-6) & (times_ms <= 500.0 + 1e-6)
    mask_450_650 = (times_ms >= 450.0 - 1e-6) & (times_ms <= 650.0 + 1e-6)

    print(f"[XAI-RQ3] Bắt đầu chạy Captum IG trên 39 folds LOSO (device={torch_device})...")
    for s_idx, spec in enumerate(loso_folds):
        sub_id = str(spec["subject_id"])
        fold_key = str(spec["fold_key"])
        ckpt_path = ckpt_root / "dl2_loso_p300_atcnet" / f"{fold_key}.pt"
        model, scaler, _ = load_checkpoint_for_eval(ckpt_path, device=torch_device)

        tr_idx = np.asarray(spec["train_idx"], dtype=np.int64)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        tr_freq_idx = tr_idx[clean_bundle.y[tr_idx] == 0]
        te_tgt_idx = te_idx[clean_bundle.y[te_idx] == 1]
        te_freq_idx = te_idx[clean_bundle.y[te_idx] == 0]

        # Baseline Frequent trung bình tập train đã scale (chuẩn Zero Leakage)
        x_tr_freq_scaled = scaler.transform(clean_bundle.X[tr_freq_idx])
        baseline_train_freq = np.mean(x_tr_freq_scaled.astype(np.float64), axis=0, keepdims=True).astype(
            np.float32
        )

        # Baseline Frequent của đúng tập test đã scale (đối chứng khử offset từng subject)
        x_te_freq_scaled = scaler.transform(clean_bundle.X[te_freq_idx])
        baseline_test_freq = np.mean(x_te_freq_scaled.astype(np.float64), axis=0, keepdims=True).astype(
            np.float32
        )

        baseline_zero = np.zeros((1, n_ch, n_t), dtype=np.float32)

        x_te_tgt_raw = clean_bundle.X[te_tgt_idx]
        x_te_tgt_scaled = scaler.transform(x_te_tgt_raw)

        # Sample Frequent đối chứng cùng số lượng Target ở mỗi subject
        n_tgt = len(te_tgt_idx)
        n_samp = min(n_tgt, len(te_freq_idx))
        samp_pos = np.sort(rng.choice(len(te_freq_idx), size=n_samp, replace=False))
        x_te_freq_samp_scaled = x_te_freq_scaled[samp_pos]

        # Chạy IG
        attr_tgt_freqbase, deltas_tgt = compute_captum_ig(
            model,
            x_te_tgt_scaled,
            baseline_train_freq,
            target=0,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=torch_device,
            return_convergence_delta=True,
        )
        attr_tgt_zero, _ = compute_captum_ig(
            model,
            x_te_tgt_scaled,
            baseline_zero,
            target=0,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=torch_device,
            return_convergence_delta=False,
        )
        attr_tgt_testfreq, _ = compute_captum_ig(
            model,
            x_te_tgt_scaled,
            baseline_test_freq,
            target=0,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=torch_device,
            return_convergence_delta=False,
        )
        attr_freq_samp, _ = compute_captum_ig(
            model,
            x_te_freq_samp_scaled,
            baseline_train_freq,
            target=0,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=torch_device,
            return_convergence_delta=False,
        )

        # Trọng số Depthwise Spatial Conv (32, 1, 127, 1) -> (32, 127)
        w_sp = model.get_spatial_weights().detach().cpu().numpy().squeeze((1, 3)).astype(np.float32)

        loso_subjects.append(sub_id)
        loso_target_abs_ig[s_idx] = np.mean(np.abs(attr_tgt_freqbase), axis=0)
        loso_target_signed_ig[s_idx] = np.mean(attr_tgt_freqbase, axis=0)
        loso_target_abs_ig_zero[s_idx] = np.mean(np.abs(attr_tgt_zero), axis=0)
        loso_target_abs_ig_testfreq[s_idx] = np.mean(np.abs(attr_tgt_testfreq), axis=0)
        loso_freq_abs_ig[s_idx] = np.mean(np.abs(attr_freq_samp), axis=0)
        loso_spatial_weights_raw[s_idx] = w_sp
        loso_spatial_weights_abs[s_idx] = np.mean(np.abs(w_sp), axis=0)

        loso_erp_target[s_idx] = np.mean(x_te_tgt_raw.astype(np.float64), axis=0).astype(np.float32)
        loso_erp_freq[s_idx] = np.mean(
            clean_bundle.X[te_freq_idx].astype(np.float64), axis=0
        ).astype(np.float32)

        # Lưu thông tin trial Target
        abs_tgt = np.abs(attr_tgt_freqbase)
        abs_zero = np.abs(attr_tgt_zero)
        for local_i, global_idx in enumerate(te_tgt_idx):
            row = meta_clean.iloc[int(global_idx)]
            target_trial_uids.append(str(row["trial_uid"]))
            target_trial_subs.append(sub_id)
            target_trial_rt_ms.append(float(row["rt_ms"]))
            target_trial_rt_group.append(str(row["rt_group"]))
            target_trial_y_pred.append(int(y_pred_all[int(global_idx)]))
            target_trial_y_prob.append(float(y_prob_all[int(global_idx)]))

        target_trial_pz_erp.append((x_te_tgt_raw[:, pz_idx, :] * 1e6).astype(np.float32))
        target_trial_temp_abs_ig.append(abs_tgt.mean(axis=1).astype(np.float32))
        target_trial_temp_signed_ig.append(attr_tgt_freqbase.mean(axis=1).astype(np.float32))
        target_trial_temp_abs_ig_zero.append(abs_zero.mean(axis=1).astype(np.float32))
        target_trial_ch_abs_p300.append(abs_tgt[:, :, mask_300_600].mean(axis=2).astype(np.float32))
        target_trial_ch_abs_300_500.append(abs_tgt[:, :, mask_300_500].mean(axis=2).astype(np.float32))
        target_trial_ch_abs_450_650.append(abs_tgt[:, :, mask_450_650].mean(axis=2).astype(np.float32))
        target_trial_ch_signed_p300.append(
            attr_tgt_freqbase[:, :, mask_300_600].mean(axis=2).astype(np.float32)
        )
        target_trial_ch_signed_300_500.append(
            attr_tgt_freqbase[:, :, mask_300_500].mean(axis=2).astype(np.float32)
        )
        if deltas_tgt is not None:
            target_trial_deltas.append(deltas_tgt)

        freq_trial_temp_abs_ig.append(np.abs(attr_freq_samp).mean(axis=1).astype(np.float32))

        if (s_idx + 1) % 10 == 0 or (s_idx + 1) == n_subs:
            print(f"  [LOSO IG] {s_idx + 1}/{n_subs} subjects hoàn tất ({sub_id})")

    # ------------------------------------------------------------------
    # LỚP 2 — Captum IG trên CS5 Clean vs CS5 No-ICA
    # ------------------------------------------------------------------
    print("[XAI-RQ3] Chạy Captum IG trên 5 folds CS5 (clean vs no_ica)...")
    no_ica_bundle = load_eeg_dataset(data_root, data_variant="no_ica")

    def _run_cs5_ig(bundle, ckpt_subdir: str) -> Dict[str, Any]:
        folds = get_protocol_folds(bundle.splits, protocol="cross_subject_5fold")
        uids, subs, ch_full_list, ch_p300_list, temp_list, kept_list = [], [], [], [], [], []
        sub_abs_maps: Dict[str, List[np.ndarray]] = {}
        sub_abs_maps_matched: Dict[str, List[np.ndarray]] = {}
        has_kept_col = "kept_in_clean" in bundle.metadata.columns

        for spec in folds:
            fold_key = str(spec["fold_key"])
            ckpt_path = ckpt_root / ckpt_subdir / f"{fold_key}.pt"
            model, scaler, _ = load_checkpoint_for_eval(ckpt_path, device=torch_device)

            tr_idx = np.asarray(spec["train_idx"], dtype=np.int64)
            te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
            tr_freq_idx = tr_idx[bundle.y[tr_idx] == 0]
            te_tgt_idx = te_idx[bundle.y[te_idx] == 1]

            x_tr_freq_scaled = scaler.transform(bundle.X[tr_freq_idx])
            base_freq = np.mean(x_tr_freq_scaled.astype(np.float64), axis=0, keepdims=True).astype(
                np.float32
            )
            x_te_tgt_scaled = scaler.transform(bundle.X[te_tgt_idx])

            attrs, _ = compute_captum_ig(
                model,
                x_te_tgt_scaled,
                base_freq,
                target=0,
                n_steps=n_steps,
                internal_batch_size=internal_batch_size,
                device=torch_device,
                return_convergence_delta=False,
            )
            abs_attrs = np.abs(attrs)
            for local_i, g_idx in enumerate(te_tgt_idx):
                row = bundle.metadata.iloc[int(g_idx)]
                sub_id = str(row["subject_id"])
                kept_flag = bool(row["kept_in_clean"]) if has_kept_col else True
                uids.append(str(row["trial_uid"]))
                subs.append(sub_id)
                kept_list.append(kept_flag)
                sub_abs_maps.setdefault(sub_id, []).append(abs_attrs[local_i])
                if kept_flag:
                    sub_abs_maps_matched.setdefault(sub_id, []).append(abs_attrs[local_i])

            ch_full_list.append(abs_attrs.mean(axis=2).astype(np.float32))
            ch_p300_list.append(abs_attrs[:, :, mask_300_600].mean(axis=2).astype(np.float32))
            temp_list.append(abs_attrs.mean(axis=1).astype(np.float32))

        ordered_subs = sorted(sub_abs_maps.keys(), key=natural_sort_key)
        per_sub_map = np.stack(
            [np.mean(sub_abs_maps[s], axis=0) for s in ordered_subs], axis=0
        ).astype(np.float32)
        per_sub_matched = np.stack(
            [np.mean(sub_abs_maps_matched[s], axis=0) for s in ordered_subs], axis=0
        ).astype(np.float32)

        return {
            "trial_uids": np.asarray(uids, dtype=str),
            "subject_ids": np.asarray(subs, dtype=str),
            "kept_in_clean": np.asarray(kept_list, dtype=bool),
            "ch_abs_full": np.concatenate(ch_full_list, axis=0),
            "ch_abs_p300": np.concatenate(ch_p300_list, axis=0),
            "temp_abs": np.concatenate(temp_list, axis=0),
            "per_sub_abs_full": per_sub_map,
            "per_sub_abs_matched": per_sub_matched,
            "ordered_subs": np.asarray(ordered_subs, dtype=str),
        }

    cs5_clean_ig = _run_cs5_ig(clean_bundle, "dl2_cs5_p300_atcnet_clean")
    cs5_noica_ig = _run_cs5_ig(no_ica_bundle, "dl2_cs5_p300_atcnet_no_ica")

    # Sắp xếp CS5 no_ica theo đúng trial_uid của CS5 clean cho nhánh khớp 1-1
    clean_uid_order = cs5_clean_ig["trial_uids"].tolist()
    noica_uid_to_pos = {u: i for i, u in enumerate(cs5_noica_ig["trial_uids"].tolist())}
    matched_pos_in_noica = np.asarray([noica_uid_to_pos[u] for u in clean_uid_order], dtype=np.int64)

    target_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target_cache,
        channel_names=np.asarray(channel_names, dtype=str),
        times=times,
        times_ms=times_ms,
        loso_subjects=np.asarray(loso_subjects, dtype=str),
        loso_target_abs_ig=loso_target_abs_ig,
        loso_target_signed_ig=loso_target_signed_ig,
        loso_target_abs_ig_zero=loso_target_abs_ig_zero,
        loso_target_abs_ig_testfreq=loso_target_abs_ig_testfreq,
        loso_freq_abs_ig=loso_freq_abs_ig,
        loso_spatial_weights_abs=loso_spatial_weights_abs,
        loso_spatial_weights_raw=loso_spatial_weights_raw,
        loso_erp_target=loso_erp_target,
        loso_erp_freq=loso_erp_freq,
        target_trial_uids=np.asarray(target_trial_uids, dtype=str),
        target_trial_subs=np.asarray(target_trial_subs, dtype=str),
        target_trial_rt_ms=np.asarray(target_trial_rt_ms, dtype=np.float64),
        target_trial_rt_group=np.asarray(target_trial_rt_group, dtype=str),
        target_trial_y_pred=np.asarray(target_trial_y_pred, dtype=np.int64),
        target_trial_y_prob=np.asarray(target_trial_y_prob, dtype=np.float64),
        target_trial_pz_erp=np.concatenate(target_trial_pz_erp, axis=0),
        target_trial_temp_abs_ig=np.concatenate(target_trial_temp_abs_ig, axis=0),
        target_trial_temp_signed_ig=np.concatenate(target_trial_temp_signed_ig, axis=0),
        target_trial_temp_abs_ig_zero=np.concatenate(target_trial_temp_abs_ig_zero, axis=0),
        target_trial_ch_abs_p300=np.concatenate(target_trial_ch_abs_p300, axis=0),
        target_trial_ch_abs_300_500=np.concatenate(target_trial_ch_abs_300_500, axis=0),
        target_trial_ch_abs_450_650=np.concatenate(target_trial_ch_abs_450_650, axis=0),
        target_trial_ch_signed_p300=np.concatenate(target_trial_ch_signed_p300, axis=0),
        target_trial_ch_signed_300_500=np.concatenate(target_trial_ch_signed_300_500, axis=0),
        target_trial_deltas=np.concatenate(target_trial_deltas, axis=0) if target_trial_deltas else np.empty(0),
        freq_trial_temp_abs_ig=np.concatenate(freq_trial_temp_abs_ig, axis=0),
        cs5_subjects=cs5_clean_ig["ordered_subs"],
        cs5_clean_trial_uids=cs5_clean_ig["trial_uids"],
        cs5_clean_trial_subs=cs5_clean_ig["subject_ids"],
        cs5_clean_ch_abs_full=cs5_clean_ig["ch_abs_full"],
        cs5_clean_ch_abs_p300=cs5_clean_ig["ch_abs_p300"],
        cs5_clean_temp_abs=cs5_clean_ig["temp_abs"],
        cs5_clean_per_sub_abs=cs5_clean_ig["per_sub_abs_full"],
        cs5_noica_trial_uids=cs5_noica_ig["trial_uids"],
        cs5_noica_trial_subs=cs5_noica_ig["subject_ids"],
        cs5_noica_kept_in_clean=cs5_noica_ig["kept_in_clean"],
        cs5_noica_ch_abs_full=cs5_noica_ig["ch_abs_full"],
        cs5_noica_ch_abs_p300=cs5_noica_ig["ch_abs_p300"],
        cs5_noica_temp_abs=cs5_noica_ig["temp_abs"],
        cs5_noica_per_sub_abs_full=cs5_noica_ig["per_sub_abs_full"],
        cs5_noica_per_sub_abs_matched=cs5_noica_ig["per_sub_abs_matched"],
        cs5_noica_matched_pos=matched_pos_in_noica,
    )
    elapsed = time.perf_counter() - t_start
    print(f"[XAI-RQ3] Đã lưu xai_loso_cache.npz tại {target_cache} ({elapsed:.1f}s).")
    return target_cache


# ==============================================================================
# LỚP 3.3 — EARLY-WINDOW CUTOFF (<= 450 MS): INFERENCE MASKING & RETRAIN 5-FOLD
# ==============================================================================
def run_early_window_cutoff(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    cutoff_ms: float = EARLY_CUTOFF_MS,
    *,
    resume: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Path:
    """Chạy kiểm chứng cắt cửa sổ sớm `[-0.2, 0.45] s` ở cả 2 chế độ:
      (a) Inference-only masking (`t > 450 ms` -> 0 sau khi scale) trên 39 checkpoints LOSO & 5 folds CS5.
      (b) Retrain-from-scratch `P300ATCNet(n_times=163)` trên `cross_subject_5fold`.
    """
    torch_device = _resolve_device(device)
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    ckpt_root = out_root / "checkpoints" / "dl2"
    out_json_path = metrics_dir / "early_window_cutoff.json"

    bundle = load_eeg_dataset(data_root, data_variant="clean")
    times_ms = np.round(np.load(data_root / "times.npy") * 1000.0, 3)
    keep_mask = times_ms <= float(cutoff_ms) + 1e-6
    n_times_early = int(np.sum(keep_mask))
    if not np.array_equal(keep_mask[:n_times_early], np.ones(n_times_early, dtype=bool)):
        raise ValueError("Trục thời gian không liên tục trước mốc cutoff_ms.")

    # (a.1) Inference-only masking trên 39 checkpoints LOSO
    loso_folds = get_protocol_folds(bundle.splits, protocol="cross_subject_loso")
    loso_orig_json = json.loads((metrics_dir / "metrics_dl2_loso_p300_atcnet.json").read_text(encoding="utf-8"))
    loso_masked_folds: List[Dict[str, Any]] = []
    for spec in loso_folds:
        fold_key = str(spec["fold_key"])
        sub_id = str(spec["subject_id"])
        ckpt_path = ckpt_root / "dl2_loso_p300_atcnet" / f"{fold_key}.pt"
        model, scaler, ckpt = load_checkpoint_for_eval(ckpt_path, device=torch_device)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        x_te_scaled = scaler.transform(bundle.X[te_idx])
        x_te_masked = x_te_scaled.copy()
        x_te_masked[:, :, ~keep_mask] = 0.0
        probs = _predict_probs_numpy(model, x_te_masked, torch_device)
        m = compute_binary_metrics(bundle.y[te_idx], probs, threshold=float(ckpt["best_threshold"]))
        loso_masked_folds.append({"fold_key": fold_key, "protocol": "cross_subject_loso", "fold": spec["fold"], "subject_id": sub_id, **m})

    loso_masked_agg = aggregate_metrics(loso_masked_folds, protocol="cross_subject_loso")
    loso_delta: Dict[str, Dict[str, Any]] = {}
    subs_ordered = sorted(loso_orig_json["per_subject"].keys(), key=natural_sort_key)
    for m_name in METRIC_NAMES:
        a = np.array([float(loso_masked_agg["per_subject"][s][m_name]) for s in subs_ordered])
        b = np.array([float(loso_orig_json["per_subject"][s][m_name]) for s in subs_ordered])
        loso_delta[m_name] = {
            "masked_mean": float(np.mean(a)),
            "masked_std": float(np.std(a, ddof=1)),
            "orig_mean": float(np.mean(b)),
            "orig_std": float(np.std(b, ddof=1)),
            **_paired_wilcoxon(a, b),
        }

    # (a.2) Inference-only masking trên 5 checkpoints CS5 clean
    cs5_folds = get_protocol_folds(bundle.splits, protocol="cross_subject_5fold")
    cs5_orig_json = json.loads((metrics_dir / "metrics_dl2_cs5_p300_atcnet_clean.json").read_text(encoding="utf-8"))
    cs5_masked_folds: List[Dict[str, Any]] = []
    for spec in cs5_folds:
        fold_key = str(spec["fold_key"])
        ckpt_path = ckpt_root / "dl2_cs5_p300_atcnet_clean" / f"{fold_key}.pt"
        model, scaler, ckpt = load_checkpoint_for_eval(ckpt_path, device=torch_device)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        x_te_masked = scaler.transform(bundle.X[te_idx]).copy()
        x_te_masked[:, :, ~keep_mask] = 0.0
        probs = _predict_probs_numpy(model, x_te_masked, torch_device)
        m = compute_binary_metrics(bundle.y[te_idx], probs, threshold=float(ckpt["best_threshold"]))
        cs5_masked_folds.append({"fold_key": fold_key, "protocol": "cross_subject_5fold", "fold": spec["fold"], "subject_id": spec["subject_id"], **m})
    cs5_masked_agg = aggregate_metrics(cs5_masked_folds, protocol="cross_subject_5fold")

    # (b) Retrain-from-scratch 5-fold Cross-Subject trên X_clean[:, :, :163]
    early_exp_name = "dl2_cs5_p300_atcnet_early450"
    early_ckpt_dir = ckpt_root / early_exp_name
    early_cache_dir = early_ckpt_dir / "cache"
    early_metrics_path = metrics_dir / f"metrics_{early_exp_name}.json"

    X_early = np.ascontiguousarray(bundle.X[:, :, :n_times_early], dtype=np.float32)
    model_kwargs = dict(EARLY_CUTOFF_ATCNET_KWARGS)
    model_kwargs["n_times"] = n_times_early

    trial_uids = bundle.metadata["trial_uid"].astype(str).tolist()
    subject_ids = bundle.subject_ids.astype(str).tolist()
    retrain_fold_results: List[Dict[str, Any]] = []
    retrain_pred_frames: List[pd.DataFrame] = []

    print(f"[Early-Window Cutoff] Retrain P300ATCNet(n_times={n_times_early}, <= {cutoff_ms:.0f} ms) trên 5-fold CS5...")
    for idx, spec in enumerate(cs5_folds):
        fold_key = str(spec["fold_key"])
        cached_ok = False
        if resume:
            c_ckpt = early_ckpt_dir / f"{fold_key}.pt"
            c_pred = early_cache_dir / f"{fold_key}_pred.csv"
            c_met = early_cache_dir / f"{fold_key}_metrics.json"
            if c_ckpt.exists() and c_pred.exists() and c_met.exists():
                retrain_pred_frames.append(pd.read_csv(c_pred))
                retrain_fold_results.append(json.loads(c_met.read_text(encoding="utf-8")))
                cached_ok = True
                print(f"  [Early450 {idx + 1}/5] {fold_key}: RESUME (cached)")

        if cached_ok:
            continue

        fold_meta = {
            "fold": spec["fold"],
            "subject_id": spec["subject_id"],
            "protocol": "cross_subject_5fold",
            "fold_key": fold_key,
            "train_subjects": spec["train_subjects"],
            "val_subjects": spec["val_subjects"],
            "test_subjects": spec["test_subjects"],
        }
        res = train_single_fold(
            X=X_early,
            y=bundle.y,
            train_idx=spec["train_idx"],
            val_idx=spec["val_idx"],
            test_idx=spec["test_idx"],
            model_name="p300_atcnet",
            model_kwargs=model_kwargs,
            protocol="cross_subject_5fold",
            fold_key=fold_key,
            fold_meta=fold_meta,
            trial_uids=trial_uids,
            subject_ids=subject_ids,
            epochs=80,
            patience=15,
            batch_size=64,
            lr=1e-3,
            weight_decay=1e-2,
            seed=42,
            device=torch_device,
            checkpoint_dir=early_ckpt_dir,
            save_checkpoint=True,
        )
        fm = res["fold_metrics"]
        pred_df = res["predictions_df"]
        retrain_fold_results.append(fm)
        retrain_pred_frames.append(pred_df)

        early_cache_dir.mkdir(parents=True, exist_ok=True)
        pred_df[PREDICTION_COLUMNS].to_csv(early_cache_dir / f"{fold_key}_pred.csv", index=False)
        (early_cache_dir / f"{fold_key}_metrics.json").write_text(json.dumps(fm, indent=2), encoding="utf-8")
        print(
            f"  [Early450 {idx + 1}/5] {fold_key}: "
            f"bAcc={fm['balanced_accuracy']:.4f} | ROC={fm['roc_auc']:.4f} | "
            f"PR={fm['pr_auc']:.4f} | ep={fm['best_epoch']} | {fm['train_time_sec']:.1f}s"
        )

    all_early_preds = pd.concat(retrain_pred_frames, ignore_index=True)
    save_predictions_csv(all_early_preds, metrics_dir / f"predictions_{early_exp_name}.csv")
    retrain_agg = aggregate_metrics(retrain_fold_results, protocol="cross_subject_5fold")
    save_metrics_json(
        out_path=early_metrics_path,
        config={
            "exp_name": early_exp_name,
            "cutoff_ms": float(cutoff_ms),
            "n_times": n_times_early,
            "model": "p300_atcnet",
            "model_kwargs": model_kwargs,
            "protocol": "cross_subject_5fold",
            "epochs": 80,
            "patience": 15,
            "batch_size": 64,
            "lr": 1e-3,
            "weight_decay": 1e-2,
            "seed": 42,
        },
        trial_uid_sha256=bundle.trial_uid_sha256,
        summary=retrain_agg["summary"],
        per_fold=retrain_fold_results,
        per_subject=retrain_agg["per_subject"],
    )

    # So sánh CS5 retrain vs CS5 gốc & CS5 inference-masked
    cs5_compare: Dict[str, Any] = {}
    for m_name in METRIC_NAMES:
        orig_vals = np.array([float(f[m_name]) for f in cs5_orig_json["per_fold"]])
        mask_vals = np.array([float(f[m_name]) for f in cs5_masked_folds])
        ret_vals = np.array([float(f[m_name]) for f in retrain_fold_results])
        cs5_compare[m_name] = {
            "orig_mean": float(np.mean(orig_vals)),
            "orig_std": float(np.std(orig_vals, ddof=1)),
            "inference_masked_mean": float(np.mean(mask_vals)),
            "inference_masked_std": float(np.std(mask_vals, ddof=1)),
            "retrain_early_mean": float(np.mean(ret_vals)),
            "retrain_early_std": float(np.std(ret_vals, ddof=1)),
            "delta_masked_vs_orig": float(np.mean(mask_vals - orig_vals)),
            "delta_retrain_vs_orig": float(np.mean(ret_vals - orig_vals)),
        }

    payload = {
        "cutoff_ms": float(cutoff_ms),
        "n_times_kept": n_times_early,
        "window_sec": [-0.2, float(times_ms[n_times_early - 1] / 1000.0)],
        "loso_inference_masking": {
            "summary": loso_masked_agg["summary"],
            "comparison_vs_orig": loso_delta,
            "per_subject": loso_masked_agg["per_subject"],
        },
        "cs5_inference_masking": {
            "summary": cs5_masked_agg["summary"],
            "per_fold": cs5_masked_folds,
        },
        "cs5_retrain_early450": {
            "exp_name": early_exp_name,
            "summary": retrain_agg["summary"],
            "per_fold": retrain_fold_results,
        },
        "cs5_comparison": cs5_compare,
    }
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(payload), f, indent=2, allow_nan=False)
    print(f"[Early-Window Cutoff] Đã lưu kết quả tại {out_json_path}.")
    return out_json_path


# ==============================================================================
# LỚP 4.1 — FREQUENCY OCCLUSION (BAND PERTURBATION) TRÊN 39 FOLDS LOSO
# ==============================================================================
def run_frequency_occlusion(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    bands: Sequence[str] = ("delta", "theta", "alpha", "beta", "gamma", "sub_delta", "delta_theta"),
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Đánh giá mức tụt ROC-AUC / PR-AUC khi che từng dải tần trên 39 folds LOSO."""
    torch_device = _resolve_device(device)
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    ckpt_root = out_root / "checkpoints" / "dl2"

    bundle = load_eeg_dataset(data_root, data_variant="clean")
    loso_folds = get_protocol_folds(bundle.splits, protocol="cross_subject_loso")
    orig_json = json.loads((metrics_dir / "metrics_dl2_loso_p300_atcnet.json").read_text(encoding="utf-8"))
    subs_ordered = sorted(orig_json["per_subject"].keys(), key=natural_sort_key)

    band_results: Dict[str, Any] = {}
    print(f"[XAI-RQ3] Chạy Frequency Occlusion ({len(bands)} dải tần × 39 folds LOSO)...")

    # Nạp 39 checkpoints 1 lần hoặc theo fold? Nạp 1 fold đánh cả 7 bands nhanh hơn rất nhiều!
    per_band_folds: Dict[str, List[Dict[str, Any]]] = {b: [] for b in bands}
    for spec in loso_folds:
        fold_key = str(spec["fold_key"])
        sub_id = str(spec["subject_id"])
        ckpt_path = ckpt_root / "dl2_loso_p300_atcnet" / f"{fold_key}.pt"
        model, scaler, ckpt = load_checkpoint_for_eval(ckpt_path, device=torch_device)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        x_te_raw = bundle.X[te_idx]
        y_te = bundle.y[te_idx]
        thr = float(ckpt["best_threshold"])

        for band in bands:
            x_occ = apply_frequency_occlusion(x_te_raw, band=band, sfreq=SFREQ)
            x_occ_scaled = scaler.transform(x_occ)
            probs = _predict_probs_numpy(model, x_occ_scaled, torch_device)
            m = compute_binary_metrics(y_te, probs, threshold=thr)
            per_band_folds[band].append(
                {
                    "fold_key": fold_key,
                    "protocol": "cross_subject_loso",
                    "fold": spec["fold"],
                    "subject_id": sub_id,
                    **m,
                }
            )

    for band in bands:
        agg = aggregate_metrics(per_band_folds[band], protocol="cross_subject_loso")
        delta_stats: Dict[str, Any] = {}
        for m_name in ("roc_auc", "pr_auc", "balanced_accuracy", "f1_target"):
            occ_vals = np.array([float(agg["per_subject"][s][m_name]) for s in subs_ordered])
            orig_vals = np.array([float(orig_json["per_subject"][s][m_name]) for s in subs_ordered])
            diff_drop = occ_vals - orig_vals
            w = _paired_wilcoxon(occ_vals, orig_vals)
            delta_stats[m_name] = {
                "occluded_mean": float(np.mean(occ_vals)),
                "occluded_std": float(np.std(occ_vals, ddof=1)),
                "orig_mean": float(np.mean(orig_vals)),
                "delta_mean": float(np.mean(diff_drop)),
                "delta_std": float(np.std(diff_drop, ddof=1)),
                "drop_mean": float(-np.mean(diff_drop)),
                "drop_std": float(np.std(-diff_drop, ddof=1)),
                "wilcoxon_p": w["wilcoxon_p"],
                "ttest_p": w["ttest_p"],
            }
        band_results[band] = {
            "band_hz": list(FREQUENCY_BANDS[band]),
            "label": FREQUENCY_BAND_LABELS[band],
            "summary": agg["summary"],
            "delta_vs_orig": delta_stats,
            "per_subject": agg["per_subject"],
        }
        r_drop = delta_stats["roc_auc"]
        p_drop = delta_stats["pr_auc"]
        print(
            f"  [FreqOcc] {FREQUENCY_BAND_LABELS[band]:<22}: "
            f"ΔROC={r_drop['delta_mean']:+.4f}±{r_drop['delta_std']:.4f} (p={_fmt_pval(r_drop['wilcoxon_p'])}) | "
            f"ΔPR={p_drop['delta_mean']:+.4f}±{p_drop['delta_std']:.4f}"
        )

    return band_results


# ==============================================================================
# TỔNG HỢP 4 LỚP XAI & XUẤT JSON + BÁO CÁO MARKDOWN
# ==============================================================================
def _compute_roi_densities_per_subject(
    abs_map_3d: np.ndarray,
    roi_idx_map: Dict[str, List[int]],
    t_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Tính mật độ `mean |IG|` của từng subject trên các ROI trong cửa sổ `t_mask`."""
    sub_win = abs_map_3d[:, :, t_mask]
    out: Dict[str, np.ndarray] = {}
    for roi_name, idxs in roi_idx_map.items():
        out[roi_name] = sub_win[:, idxs, :].mean(axis=(1, 2)).astype(np.float64)
    out["all_channels"] = sub_win.mean(axis=(1, 2)).astype(np.float64)
    return out


def analyze_all_layers(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    cache_path: Optional[Union[str, Path]] = None,
    early_cutoff_path: Optional[Union[str, Path]] = None,
    freq_occlusion_results: Optional[Dict[str, Any]] = None,
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Tính toàn bộ số liệu định lượng 4 lớp kiểm chứng XAI từ cache và lưu `xai_rq3_summary.json`."""
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    c_path = Path(cache_path) if cache_path is not None else (metrics_dir / "xai_loso_cache.npz")
    e_path = Path(early_cutoff_path) if early_cutoff_path is not None else (metrics_dir / "early_window_cutoff.json")

    cache = np.load(c_path, allow_pickle=False)
    channel_names = [str(c) for c in cache["channel_names"].tolist()]
    times_ms = cache["times_ms"].astype(np.float64)
    win_masks = get_time_window_masks(times_ms)
    roi_idx_map = verify_rois(channel_names)

    loso_subs = [str(s) for s in cache["loso_subjects"].tolist()]
    n_subs = len(loso_subs)

    # ------------------------------------------------------------------
    # LỚP 1 — Chiều Thời gian & Sanity Check
    # ------------------------------------------------------------------
    abs_3d = cache["loso_target_abs_ig"].astype(np.float64)        # (39, 127, 251)
    signed_3d = cache["loso_target_signed_ig"].astype(np.float64)
    abs_zero_3d = cache["loso_target_abs_ig_zero"].astype(np.float64)
    abs_testfreq_3d = cache["loso_target_abs_ig_testfreq"].astype(np.float64)
    freq_abs_3d = cache["loso_freq_abs_ig"].astype(np.float64)

    temp_abs_sub = abs_3d.mean(axis=1)             # (39, 251)
    temp_signed_sub = signed_3d.mean(axis=1)       # (39, 251)
    temp_abs_zero_sub = abs_zero_3d.mean(axis=1)
    temp_freq_sub = freq_abs_3d.mean(axis=1)

    t_mean, t_std, t_ci95 = _mean_std_ci95(temp_abs_sub, axis=0)
    ts_mean, ts_std, ts_ci95 = _mean_std_ci95(temp_signed_sub, axis=0)
    tf_mean, tf_std, tf_ci95 = _mean_std_ci95(temp_freq_sub, axis=0)

    # Tỷ trọng năng lượng (%) & mật độ trung bình trên 4 cửa sổ sinh lý
    total_energy_sub = temp_abs_sub.sum(axis=1)  # (39,)
    window_stats: Dict[str, Any] = {}
    window_shares_sub: Dict[str, np.ndarray] = {}
    window_densities_sub: Dict[str, np.ndarray] = {}

    for w_name in TIME_WINDOWS_ORDER:
        w_mask = win_masks[w_name]
        w_energy = temp_abs_sub[:, w_mask].sum(axis=1)
        w_share_pct = (w_energy / np.maximum(total_energy_sub, 1e-15)) * 100.0
        w_density = temp_abs_sub[:, w_mask].mean(axis=1)
        window_shares_sub[w_name] = w_share_pct
        window_densities_sub[w_name] = w_density
        window_stats[w_name] = {
            "label": TIME_WINDOWS_LABELS[w_name],
            "n_samples": int(np.sum(w_mask)),
            "time_fraction_pct": float(np.sum(w_mask) / len(times_ms) * 100.0),
            "energy_share_pct": _scalar_summary(w_share_pct),
            "mean_abs_ig": _scalar_summary(w_density),
        }

    # Peak latency trên đường cong trung bình và từng subject
    post_stim_mask = times_ms >= 0.0
    grand_peak_idx = int(np.argmax(t_mean[post_stim_mask]))
    grand_peak_ms = float(times_ms[post_stim_mask][grand_peak_idx])
    sub_peak_ms = np.array(
        [float(times_ms[post_stim_mask][int(np.argmax(row[post_stim_mask]))]) for row in temp_abs_sub]
    )

    # Sanity checks: P300 vs Baseline, Target vs Frequent, Baseline=Freq vs Zero vs TestFreq
    p300_vs_base_density = _paired_wilcoxon(
        window_densities_sub["p300"], window_densities_sub["baseline"]
    )
    p300_vs_base_ratio = window_densities_sub["p300"] / np.maximum(window_densities_sub["baseline"], 1e-15)
    target_vs_freq_p300 = _paired_wilcoxon(
        window_densities_sub["p300"], temp_freq_sub[:, win_masks["p300"]].mean(axis=1)
    )

    deltas_arr = cache["target_trial_deltas"].astype(np.float64)
    spatial_w_abs = cache["loso_spatial_weights_abs"].astype(np.float64)  # (39, 127)
    mask_300_500 = (times_ms >= 300.0 - 1e-6) & (times_ms <= 500.0 + 1e-6)

    ig_freq_topo = abs_3d[:, :, mask_300_500].mean(axis=(0, 2))
    ig_zero_topo = abs_zero_3d[:, :, mask_300_500].mean(axis=(0, 2))
    ig_testfreq_topo = abs_testfreq_3d[:, :, mask_300_500].mean(axis=(0, 2))
    spatial_w_grand = spatial_w_abs.mean(axis=0)

    layer1 = {
        "n_subjects": n_subs,
        "n_target_trials": int(len(cache["target_trial_uids"])),
        "n_freq_sample_trials": int(len(cache["freq_trial_temp_abs_ig"])),
        "completeness_delta": {
            "max_abs_delta": float(np.max(np.abs(deltas_arr))) if deltas_arr.size else None,
            "mean_abs_delta": float(np.mean(np.abs(deltas_arr))) if deltas_arr.size else None,
            "p95_abs_delta": float(np.percentile(np.abs(deltas_arr), 95.0)) if deltas_arr.size else None,
        },
        "grand_peak_latency_ms": grand_peak_ms,
        "subject_peak_latency_ms": _scalar_summary(sub_peak_ms),
        "windows": window_stats,
        "p300_vs_baseline_density": p300_vs_base_density,
        "p300_to_baseline_ratio": _scalar_summary(p300_vs_base_ratio),
        "target_vs_freq_p300_window": target_vs_freq_p300,
        "baseline_comparison": {
            "train_freq_vs_zero_temporal": _corr_metrics(t_mean, temp_abs_zero_sub.mean(axis=0)),
            "train_freq_vs_zero_topo_300_500": _corr_metrics(ig_freq_topo, ig_zero_topo),
            "train_freq_vs_test_freq_temporal": _corr_metrics(t_mean, abs_testfreq_3d.mean(axis=(0, 1))),
            "train_freq_vs_test_freq_topo_300_500": _corr_metrics(ig_freq_topo, ig_testfreq_topo),
        },
        "spatial_weights_top_channels": [
            {"channel": channel_names[int(i)], "mean_abs_weight": float(spatial_w_grand[int(i)])}
            for i in np.argsort(spatial_w_grand)[::-1][:10]
        ],
        "spatial_weights_vs_ig_300_500": _corr_metrics(spatial_w_grand, ig_freq_topo),
        "temporal_curve_target_abs": {
            "mean": t_mean,
            "std": t_std,
            "ci95": t_ci95,
        },
        "temporal_curve_target_signed": {
            "mean": ts_mean,
            "std": ts_std,
            "ci95": ts_ci95,
        },
        "temporal_curve_freq_abs": {
            "mean": tf_mean,
            "std": tf_std,
            "ci95": tf_ci95,
        },
    }

    # ------------------------------------------------------------------
    # LỚP 2 — Kiểm chứng Nhiễu Chớp mắt EOG (Clean vs No-ICA)
    # ------------------------------------------------------------------
    eog_idx = roi_idx_map["eog"]
    non_eog_idx = [i for i in range(127) if i not in set(eog_idx)]
    cs5_subs = [str(s) for s in cache["cs5_subjects"].tolist()]

    clean_per_sub = cache["cs5_clean_per_sub_abs"].astype(np.float64)            # (39, 127, 251)
    noica_per_sub_matched = cache["cs5_noica_per_sub_abs_matched"].astype(np.float64)
    noica_per_sub_full = cache["cs5_noica_per_sub_abs_full"].astype(np.float64)

    def _eog_metrics_per_sub(arr_3d: np.ndarray) -> Dict[str, np.ndarray]:
        eog_sum_full = arr_3d[:, eog_idx, :].sum(axis=(1, 2))
        tot_sum_full = arr_3d.sum(axis=(1, 2))
        share_full = (eog_sum_full / np.maximum(tot_sum_full, 1e-15)) * 100.0

        p300_slice = arr_3d[:, :, win_masks["p300"]]
        eog_sum_p300 = p300_slice[:, eog_idx, :].sum(axis=(1, 2))
        tot_sum_p300 = p300_slice.sum(axis=(1, 2))
        share_p300 = (eog_sum_p300 / np.maximum(tot_sum_p300, 1e-15)) * 100.0

        eog_density = arr_3d[:, eog_idx, :].mean(axis=(1, 2))
        non_eog_density = arr_3d[:, non_eog_idx, :].mean(axis=(1, 2))
        density_ratio = eog_density / np.maximum(non_eog_density, 1e-15)
        return {
            "share_full_pct": share_full,
            "share_p300_pct": share_p300,
            "eog_density": eog_density,
            "non_eog_density": non_eog_density,
            "density_ratio": density_ratio,
        }

    m_clean = _eog_metrics_per_sub(clean_per_sub)
    m_noica_matched = _eog_metrics_per_sub(noica_per_sub_matched)
    m_noica_full = _eog_metrics_per_sub(noica_per_sub_full)

    # Phân tích riêng 2 ca nội suy Fp1/Fp2 (sub-18, sub-40) vs 37 còn lại
    fp_interp_set = set(FP1_FP2_INTERP_SUBJECTS)
    idx_interp = [i for i, s in enumerate(cs5_subs) if s in fp_interp_set]
    idx_other = [i for i, s in enumerate(cs5_subs) if s not in fp_interp_set]

    interp_breakdown = {}
    for i in idx_interp:
        s = cs5_subs[i]
        interp_breakdown[s] = {
            "clean_eog_share_full_pct": float(m_clean["share_full_pct"][i]),
            "noica_matched_eog_share_full_pct": float(m_noica_matched["share_full_pct"][i]),
            "noica_full_eog_share_full_pct": float(m_noica_full["share_full_pct"][i]),
            "delta_matched_pct": float(m_clean["share_full_pct"][i] - m_noica_matched["share_full_pct"][i]),
            "clean_eog_density": float(m_clean["eog_density"][i]),
            "noica_matched_eog_density": float(m_noica_matched["eog_density"][i]),
        }

    # Cũng tính ở mức toàn trial (2,850 matched & 3,002 no_ica full)
    c_tr_full = cache["cs5_clean_ch_abs_full"].astype(np.float64)
    n_tr_full = cache["cs5_noica_ch_abs_full"].astype(np.float64)
    m_pos = cache["cs5_noica_matched_pos"].astype(np.int64)
    n_tr_matched = n_tr_full[m_pos]

    c_trial_eog_share = (c_tr_full[:, eog_idx].sum(axis=1) / np.maximum(c_tr_full.sum(axis=1), 1e-15)) * 100.0
    n_matched_eog_share = (n_tr_matched[:, eog_idx].sum(axis=1) / np.maximum(n_tr_matched.sum(axis=1), 1e-15)) * 100.0
    n_full_eog_share = (n_tr_full[:, eog_idx].sum(axis=1) / np.maximum(n_tr_full.sum(axis=1), 1e-15)) * 100.0

    # Đọc hiệu năng CS5 clean vs no_ica từ Step 5 để báo cáo song hành
    cs5_clean_m = json.loads((metrics_dir / "metrics_dl2_cs5_p300_atcnet_clean.json").read_text(encoding="utf-8"))
    cs5_noica_m = json.loads((metrics_dir / "metrics_dl2_cs5_p300_atcnet_no_ica.json").read_text(encoding="utf-8"))

    layer2 = {
        "uniform_baseline_share_pct": float(len(eog_idx) / 127.0 * 100.0),
        "n_matched_target_trials": int(len(c_tr_full)),
        "n_noica_full_target_trials": int(len(n_tr_full)),
        "matched_1to1_subject_level": {
            "clean_share_full_pct": _scalar_summary(m_clean["share_full_pct"]),
            "noica_share_full_pct": _scalar_summary(m_noica_matched["share_full_pct"]),
            "wilcoxon_share_full": _paired_wilcoxon(m_clean["share_full_pct"], m_noica_matched["share_full_pct"]),
            "clean_share_p300_pct": _scalar_summary(m_clean["share_p300_pct"]),
            "noica_share_p300_pct": _scalar_summary(m_noica_matched["share_p300_pct"]),
            "wilcoxon_share_p300": _paired_wilcoxon(m_clean["share_p300_pct"], m_noica_matched["share_p300_pct"]),
            "clean_eog_density": _scalar_summary(m_clean["eog_density"]),
            "noica_eog_density": _scalar_summary(m_noica_matched["eog_density"]),
            "wilcoxon_eog_density": _paired_wilcoxon(m_clean["eog_density"], m_noica_matched["eog_density"]),
            "clean_density_ratio": _scalar_summary(m_clean["density_ratio"]),
            "noica_density_ratio": _scalar_summary(m_noica_matched["density_ratio"]),
            "wilcoxon_density_ratio": _paired_wilcoxon(m_clean["density_ratio"], m_noica_matched["density_ratio"]),
        },
        "full_set_subject_level": {
            "noica_full_share_full_pct": _scalar_summary(m_noica_full["share_full_pct"]),
            "noica_full_share_p300_pct": _scalar_summary(m_noica_full["share_p300_pct"]),
            "noica_full_eog_density": _scalar_summary(m_noica_full["eog_density"]),
            "wilcoxon_share_full_clean_vs_noica_full": _paired_wilcoxon(
                m_clean["share_full_pct"], m_noica_full["share_full_pct"]
            ),
        },
        "trial_level": {
            "clean_eog_share_pct": _scalar_summary(c_trial_eog_share),
            "noica_matched_eog_share_pct": _scalar_summary(n_matched_eog_share),
            "noica_full_eog_share_pct": _scalar_summary(n_full_eog_share),
            "wilcoxon_matched_trials": _paired_wilcoxon(c_trial_eog_share, n_matched_eog_share),
        },
        "fp1_fp2_interpolated_check": {
            "interpolated_subjects": interp_breakdown,
            "others_37_clean_share_pct": _scalar_summary(m_clean["share_full_pct"][idx_other]),
            "others_37_noica_matched_share_pct": _scalar_summary(m_noica_matched["share_full_pct"][idx_other]),
            "others_37_wilcoxon": _paired_wilcoxon(
                m_clean["share_full_pct"][idx_other], m_noica_matched["share_full_pct"][idx_other]
            ),
        },
        "classification_performance": {
            "clean_roc_auc": cs5_clean_m["summary"]["roc_auc"],
            "noica_roc_auc": cs5_noica_m["summary"]["roc_auc"],
            "clean_f1_target": cs5_clean_m["summary"]["f1_target"],
            "noica_f1_target": cs5_noica_m["summary"]["f1_target"],
        },
    }

    # ------------------------------------------------------------------
    # LỚP 3 — Bóc tách Nhiễu Vận động Bấm tay (Motor Confound)
    # ------------------------------------------------------------------
    roi_p300 = _compute_roi_densities_per_subject(abs_3d, roi_idx_map, win_masks["p300"])
    roi_300_500 = _compute_roi_densities_per_subject(abs_3d, roi_idx_map, mask_300_500)

    ratio_p300_left = roi_p300["p300"] / np.maximum(roi_p300["left_motor"], 1e-15)
    laterality_300_600 = (roi_p300["left_motor"] - roi_p300["right_motor"]) / np.maximum(
        roi_p300["left_motor"] + roi_p300["right_motor"], 1e-15
    )
    laterality_300_500 = (roi_300_500["left_motor"] - roi_300_500["right_motor"]) / np.maximum(
        roi_300_500["left_motor"] + roi_300_500["right_motor"], 1e-15
    )

    # Phân tầng RT trên 2,850 Target trials
    rt_ms_all = cache["target_trial_rt_ms"].astype(np.float64)
    rt_grp_all = np.asarray([str(x) for x in cache["target_trial_rt_group"].tolist()])
    tr_subs_all = np.asarray([str(x) for x in cache["target_trial_subs"].tolist()])
    tr_temp_abs = cache["target_trial_temp_abs_ig"].astype(np.float64)
    tr_ch_300_500 = cache["target_trial_ch_abs_300_500"].astype(np.float64)
    tr_ch_450_650 = cache["target_trial_ch_abs_450_650"].astype(np.float64)
    tr_ch_300_600 = cache["target_trial_ch_abs_p300"].astype(np.float64)

    rt_stratified: Dict[str, Any] = {}
    for grp_name in ("fast", "mid", "slow"):
        g_mask = rt_grp_all == grp_name
        g_temp = tr_temp_abs[g_mask]
        g_mean, g_std, g_ci = _mean_std_ci95(g_temp, axis=0)
        g_peak_ms = float(times_ms[post_stim_mask][int(np.argmax(g_mean[post_stim_mask]))])

        g_300_500 = tr_ch_300_500[g_mask]
        g_450_650 = tr_ch_450_650[g_mask]
        g_300_600 = tr_ch_300_600[g_mask]

        p300_d = g_300_600[:, roi_idx_map["p300"]].mean(axis=1)
        left_d = g_300_600[:, roi_idx_map["left_motor"]].mean(axis=1)
        right_d = g_300_600[:, roi_idx_map["right_motor"]].mean(axis=1)
        lat_d = (left_d - right_d) / np.maximum(left_d + right_d, 1e-15)

        p300_d_late = g_450_650[:, roi_idx_map["p300"]].mean(axis=1)
        left_d_late = g_450_650[:, roi_idx_map["left_motor"]].mean(axis=1)
        right_d_late = g_450_650[:, roi_idx_map["right_motor"]].mean(axis=1)
        lat_d_late = (left_d_late - right_d_late) / np.maximum(left_d_late + right_d_late, 1e-15)

        rt_stratified[grp_name] = {
            "n_trials": int(np.sum(g_mask)),
            "rt_ms": _scalar_summary(rt_ms_all[g_mask]),
            "peak_xai_latency_ms": g_peak_ms,
            "temporal_curve_mean": g_mean,
            "temporal_curve_ci95": g_ci,
            "topo_300_500": g_300_500.mean(axis=0),
            "topo_450_650": g_450_650.mean(axis=0),
            "roi_300_600": {
                "p300_mean": float(np.mean(p300_d)),
                "left_motor_mean": float(np.mean(left_d)),
                "right_motor_mean": float(np.mean(right_d)),
                "p300_to_left_motor_ratio": float(np.mean(p300_d) / max(float(np.mean(left_d)), 1e-15)),
                "motor_laterality_mean": float(np.mean(lat_d)),
            },
            "roi_450_650": {
                "p300_mean": float(np.mean(p300_d_late)),
                "left_motor_mean": float(np.mean(left_d_late)),
                "right_motor_mean": float(np.mean(right_d_late)),
                "p300_to_left_motor_ratio": float(np.mean(p300_d_late) / max(float(np.mean(left_d_late)), 1e-15)),
                "motor_laterality_mean": float(np.mean(lat_d_late)),
            },
        }

    # Tương quan Subject Mean RT vs LOSO ROC-AUC trên 39 subjects
    handover_sum_df = pd.read_csv(metrics_dir / "handover_per_subject_summary_dl2.csv")
    sub_rt_means: List[float] = []
    sub_rt_medians: List[float] = []
    sub_loso_aucs: List[float] = []
    sub_loso_prs: List[float] = []
    sub_loso_baccs: List[float] = []
    sub_ws_aucs: List[float] = []
    per_sub_rt_records: List[Dict[str, Any]] = []

    handover_map = {str(r["subject_id"]): r for _, r in handover_sum_df.iterrows()}
    for s in loso_subs:
        s_rts = rt_ms_all[tr_subs_all == s]
        r_row = handover_map[s]
        m_rt = float(np.mean(s_rts))
        med_rt = float(np.median(s_rts))
        l_auc = float(r_row["loso_roc_auc"])
        l_pr = float(r_row["loso_pr_auc"])
        l_bacc = float(r_row["loso_balanced_accuracy"])
        w_auc = float(r_row["ws_roc_auc"])
        sub_rt_means.append(m_rt)
        sub_rt_medians.append(med_rt)
        sub_loso_aucs.append(l_auc)
        sub_loso_prs.append(l_pr)
        sub_loso_baccs.append(l_bacc)
        sub_ws_aucs.append(w_auc)
        per_sub_rt_records.append(
            {
                "subject_id": s,
                "mean_rt_ms": m_rt,
                "median_rt_ms": med_rt,
                "loso_roc_auc": l_auc,
                "loso_pr_auc": l_pr,
                "loso_balanced_accuracy": l_bacc,
                "ws_roc_auc": w_auc,
            }
        )

    early_cutoff_data = json.loads(e_path.read_text(encoding="utf-8")) if e_path.exists() else None

    layer3 = {
        "roi_300_600": {
            "p300_roi": _scalar_summary(roi_p300["p300"]),
            "left_motor_roi": _scalar_summary(roi_p300["left_motor"]),
            "right_motor_roi": _scalar_summary(roi_p300["right_motor"]),
            "all_channels": _scalar_summary(roi_p300["all_channels"]),
            "p300_to_left_motor_ratio": _scalar_summary(ratio_p300_left),
            "motor_laterality": _scalar_summary(laterality_300_600),
            "wilcoxon_p300_vs_left_motor": _paired_wilcoxon(roi_p300["p300"], roi_p300["left_motor"]),
            "wilcoxon_left_vs_right_motor": _paired_wilcoxon(roi_p300["left_motor"], roi_p300["right_motor"]),
        },
        "roi_300_500": {
            "p300_roi": _scalar_summary(roi_300_500["p300"]),
            "left_motor_roi": _scalar_summary(roi_300_500["left_motor"]),
            "right_motor_roi": _scalar_summary(roi_300_500["right_motor"]),
            "motor_laterality": _scalar_summary(laterality_300_500),
            "wilcoxon_left_vs_right_motor": _paired_wilcoxon(
                roi_300_500["left_motor"], roi_300_500["right_motor"]
            ),
        },
        "rt_stratified": rt_stratified,
        "rt_vs_performance_correlation": {
            "mean_rt_vs_loso_roc_auc": _corr_metrics(np.array(sub_rt_means), np.array(sub_loso_aucs)),
            "mean_rt_vs_loso_pr_auc": _corr_metrics(np.array(sub_rt_means), np.array(sub_loso_prs)),
            "mean_rt_vs_loso_bacc": _corr_metrics(np.array(sub_rt_means), np.array(sub_loso_baccs)),
            "mean_rt_vs_ws_roc_auc": _corr_metrics(np.array(sub_rt_means), np.array(sub_ws_aucs)),
            "per_subject_rt_perf": per_sub_rt_records,
        },
        "early_window_cutoff": early_cutoff_data,
    }

    # ------------------------------------------------------------------
    # LỚP 4 — Chiều Tần số, Spatial Correlation & Error Analysis
    # ------------------------------------------------------------------
    erp_tgt_3d = cache["loso_erp_target"].astype(np.float64) * 1e6  # µV (39, 127, 251)
    erp_freq_3d = cache["loso_erp_freq"].astype(np.float64) * 1e6
    erp_diff_3d = erp_tgt_3d - erp_freq_3d

    erp_diff_300_500_grand = erp_diff_3d[:, :, mask_300_500].mean(axis=(0, 2))
    erp_diff_300_600_grand = erp_diff_3d[:, :, win_masks["p300"]].mean(axis=(0, 2))
    ig_abs_300_500_grand = abs_3d[:, :, mask_300_500].mean(axis=(0, 2))
    ig_abs_300_600_grand = abs_3d[:, :, win_masks["p300"]].mean(axis=(0, 2))
    ig_sgn_300_500_grand = signed_3d[:, :, mask_300_500].mean(axis=(0, 2))
    ig_sgn_300_600_grand = signed_3d[:, :, win_masks["p300"]].mean(axis=(0, 2))

    # Tương quan ở mức từng subject (39 subjects)
    sub_r_300_500, sub_rho_300_500 = [], []
    sub_r_signed_300_500 = []
    for i in range(n_subs):
        e_vec = erp_diff_3d[i][:, mask_300_500].mean(axis=1)
        a_vec = abs_3d[i][:, mask_300_500].mean(axis=1)
        s_vec = signed_3d[i][:, mask_300_500].mean(axis=1)
        c_abs = _corr_metrics(a_vec, np.abs(e_vec))
        c_sgn = _corr_metrics(s_vec, e_vec)
        if c_abs["pearson_r"] is not None:
            sub_r_300_500.append(c_abs["pearson_r"])
            sub_rho_300_500.append(c_abs["spearman_rho"])
        if c_sgn["pearson_r"] is not None:
            sub_r_signed_300_500.append(c_sgn["pearson_r"])

    spatial_correlation = {
        "win_300_500_abs_ig_vs_erp_diff": _corr_metrics(ig_abs_300_500_grand, erp_diff_300_500_grand),
        "win_300_500_abs_ig_vs_abs_erp_diff": _corr_metrics(ig_abs_300_500_grand, np.abs(erp_diff_300_500_grand)),
        "win_300_500_signed_ig_vs_erp_diff": _corr_metrics(ig_sgn_300_500_grand, erp_diff_300_500_grand),
        "win_300_600_abs_ig_vs_erp_diff": _corr_metrics(ig_abs_300_600_grand, erp_diff_300_600_grand),
        "win_300_600_abs_ig_vs_abs_erp_diff": _corr_metrics(ig_abs_300_600_grand, np.abs(erp_diff_300_600_grand)),
        "win_300_600_signed_ig_vs_erp_diff": _corr_metrics(ig_sgn_300_600_grand, erp_diff_300_600_grand),
        "per_subject_300_500_abs_ig_vs_abs_erp_r": _scalar_summary(np.array(sub_r_300_500)),
        "per_subject_300_500_abs_ig_vs_abs_erp_rho": _scalar_summary(np.array(sub_rho_300_500)),
        "per_subject_300_500_signed_ig_vs_erp_r": _scalar_summary(np.array(sub_r_signed_300_500)),
    }

    # Error analysis: TP vs FN trên 2,850 Target trials
    y_pred_tgt = cache["target_trial_y_pred"].astype(np.int64)
    y_prob_tgt = cache["target_trial_y_prob"].astype(np.float64)
    pz_erp_tgt = cache["target_trial_pz_erp"].astype(np.float64)  # µV (2850, 251)

    tp_mask = y_pred_tgt == 1
    fn_mask = y_pred_tgt == 0

    tp_pz_mean, _, tp_pz_ci = _mean_std_ci95(pz_erp_tgt[tp_mask], axis=0)
    fn_pz_mean, _, fn_pz_ci = _mean_std_ci95(pz_erp_tgt[fn_mask], axis=0)
    tp_ig_mean, _, tp_ig_ci = _mean_std_ci95(tr_temp_abs[tp_mask], axis=0)
    fn_ig_mean, _, fn_ig_ci = _mean_std_ci95(tr_temp_abs[fn_mask], axis=0)

    tp_pz_p300_amp = pz_erp_tgt[tp_mask][:, win_masks["p300"]].mean(axis=1)
    fn_pz_p300_amp = pz_erp_tgt[fn_mask][:, win_masks["p300"]].mean(axis=1)
    mw_pz = stats.mannwhitneyu(tp_pz_p300_amp, fn_pz_p300_amp, alternative="two-sided")
    mw_rt = stats.mannwhitneyu(rt_ms_all[tp_mask], rt_ms_all[fn_mask], alternative="two-sided")

    tp_p300_roi = tr_ch_300_600[tp_mask][:, roi_idx_map["p300"]].mean(axis=1)
    fn_p300_roi = tr_ch_300_600[fn_mask][:, roi_idx_map["p300"]].mean(axis=1)
    tp_lm_roi = tr_ch_300_600[tp_mask][:, roi_idx_map["left_motor"]].mean(axis=1)
    fn_lm_roi = tr_ch_300_600[fn_mask][:, roi_idx_map["left_motor"]].mean(axis=1)
    mw_ig_p300 = stats.mannwhitneyu(tp_p300_roi, fn_p300_roi, alternative="two-sided")

    bottom5_details: List[Dict[str, Any]] = []
    loso_sub_to_idx = {s: i for i, s in enumerate(loso_subs)}
    for s in BOTTOM5_LOSO_SUBJECTS:
        sm = tr_subs_all == s
        s_tp = int(np.sum(sm & tp_mask))
        s_fn = int(np.sum(sm & fn_mask))
        s_tot = int(np.sum(sm))
        s_idx = loso_sub_to_idx[s]
        h_row = handover_map[s]
        s_pz_p300 = float(pz_erp_tgt[sm][:, win_masks["p300"]].mean())
        s_pz_peak = float(np.max(pz_erp_tgt[sm][:, win_masks["p300"]].mean(axis=0)))
        bottom5_details.append(
            {
                "subject_id": s,
                "n_target": s_tot,
                "tp": s_tp,
                "fn": s_fn,
                "fn_rate_pct": float(s_fn / max(s_tot, 1) * 100.0),
                "mean_rt_ms": float(np.mean(rt_ms_all[sm])),
                "median_rt_ms": float(np.median(rt_ms_all[sm])),
                "pct_rt_gt_550ms": float(np.mean(rt_ms_all[sm] > 550.0) * 100.0),
                "pz_p300_mean_uv": s_pz_p300,
                "pz_p300_peak_uv": s_pz_peak,
                "loso_roc_auc": float(h_row["loso_roc_auc"]),
                "loso_pr_auc": float(h_row["loso_pr_auc"]),
                "loso_balanced_accuracy": float(h_row["loso_balanced_accuracy"]),
                "p300_to_left_motor_ratio": float(ratio_p300_left[s_idx]),
                "motor_laterality": float(laterality_300_600[s_idx]),
                "drop_rate_pct": float(h_row["drop_rate_pct"]),
                "hit_rate_pct": float(h_row["hit_rate_pct"]),
            }
        )

    error_analysis = {
        "n_tp": int(np.sum(tp_mask)),
        "n_fn": int(np.sum(fn_mask)),
        "tp_rate_pct": float(np.mean(tp_mask) * 100.0),
        "fn_rate_pct": float(np.mean(fn_mask) * 100.0),
        "tp_y_prob": _scalar_summary(y_prob_tgt[tp_mask]),
        "fn_y_prob": _scalar_summary(y_prob_tgt[fn_mask]),
        "tp_rt_ms": _scalar_summary(rt_ms_all[tp_mask]),
        "fn_rt_ms": _scalar_summary(rt_ms_all[fn_mask]),
        "rt_mannwhitney_p": float(mw_rt.pvalue),
        "tp_pz_p300_amp_uv": _scalar_summary(tp_pz_p300_amp),
        "fn_pz_p300_amp_uv": _scalar_summary(fn_pz_p300_amp),
        "pz_p300_amp_mannwhitney_p": float(mw_pz.pvalue),
        "tp_p300_roi_abs_ig": _scalar_summary(tp_p300_roi),
        "fn_p300_roi_abs_ig": _scalar_summary(fn_p300_roi),
        "tp_left_motor_abs_ig": _scalar_summary(tp_lm_roi),
        "fn_left_motor_abs_ig": _scalar_summary(fn_lm_roi),
        "p300_roi_ig_mannwhitney_p": float(mw_ig_p300.pvalue),
        "curves": {
            "tp_pz_mean_uv": tp_pz_mean,
            "tp_pz_ci95_uv": tp_pz_ci,
            "fn_pz_mean_uv": fn_pz_mean,
            "fn_pz_ci95_uv": fn_pz_ci,
            "tp_ig_mean": tp_ig_mean,
            "tp_ig_ci95": tp_ig_ci,
            "fn_ig_mean": fn_ig_mean,
            "fn_ig_ci95": fn_ig_ci,
        },
        "bottom5_loso": bottom5_details,
    }

    if freq_occlusion_results is None:
        freq_occlusion_results = run_frequency_occlusion(data_root, out_root, device=device)

    summary_payload = {
        "montage": MONTAGE_NAME,
        "sfreq_hz": SFREQ,
        "rois": {k: list(v) for k, v in ROI_CHANNELS.items()},
        "layer1_temporal": layer1,
        "layer2_eog": layer2,
        "layer3_motor_and_rt": layer3,
        "layer4_frequency_and_neuro": {
            "frequency_occlusion": freq_occlusion_results,
            "spatial_correlation": spatial_correlation,
            "error_analysis": error_analysis,
        },
    }

    summary_json_path = metrics_dir / "xai_rq3_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary_payload), f, indent=2, allow_nan=False)
    print(f"[XAI-RQ3] Đã lưu tổng hợp 4 lớp vào {summary_json_path}.")
    return summary_payload


# ==============================================================================
# SINH BÁO CÁO MARKDOWN `reports/STEP6_XAI_RQ3_REPORT.md`
# ==============================================================================
def generate_step6_report(
    summary: Dict[str, Any],
    report_path: Union[str, Path] = Path("reports/STEP6_XAI_RQ3_REPORT.md"),
) -> Path:
    """Sinh báo cáo đầy đủ 4 lớp XAI + Early-Window Cutoff + Error Analysis chuẩn Overleaf."""
    l1 = summary["layer1_temporal"]
    l2 = summary["layer2_eog"]
    l3 = summary["layer3_motor_and_rt"]
    l4 = summary["layer4_frequency_and_neuro"]

    lines: List[str] = [
        "# BƯỚC 6 — BÁO CÁO EXPLAINABLE AI (RQ3 — 4 LỚP KIỂM CHỨNG) & PHÂN TÍCH THẦN KINH HỌC",
        "",
        "- **Mô hình giải thích**: `P300ATCNet(readout='attn_pool', n_windows=2, shared_branch=False)` trên toàn bộ **39 folds LOSO** (`dl2_loso_p300_atcnet`, **Zero Leakage**).",
        f"- **Phương pháp XAI**: Captum `IntegratedGradients(model)` (`target=0`, `n_steps=64`, `internal_batch_size=32`, baseline = trung bình lớp Frequent tập train đã scale, đối chiếu với `baseline = 0` và baseline Frequent tập test).",
        f"- **Định lý Completeness**: trên toàn bộ `{l1['n_target_trials']:,}` trials Target, `max |delta| = {l1['completeness_delta']['max_abs_delta']:.4e}`, `mean |delta| = {l1['completeness_delta']['mean_abs_delta']:.4e}`.",
        f"- **Montage không gian**: `{summary['montage']}` (`127/127` kênh khớp tọa độ 3D từ `channel_names.json`, `on_missing='raise'`).",
        "",
        "---",
        "",
        "## Lớp 1 — Chiều Thời gian & Sanity Check (Baseline vs P300)",
        "",
        f"- **Đỉnh Temporal Attribution toàn tập (`|IG|`)**: **`{l1['grand_peak_latency_ms']:.1f} ms`** (per-subject peak: `{l1['subject_peak_latency_ms']['mean']:.1f} ± {l1['subject_peak_latency_ms']['std']:.1f} ms`, median `{l1['subject_peak_latency_ms']['median']:.1f} ms`).",
        f"- **Tỷ số cường độ P300 `[300, 600] ms` / Baseline `[-200, 0) ms`**: **`{l1['p300_to_baseline_ratio']['mean']:.2f}×`** (`± {l1['p300_to_baseline_ratio']['std']:.2f}`, median `{l1['p300_to_baseline_ratio']['median']:.2f}×`, Wilcoxon `p = {_fmt_pval(l1['p300_vs_baseline_density']['wilcoxon_p'])}`).",
        f"- **Đối chiếu Baseline (`Frequent mean` vs `0`)**: Temporal `r = {l1['baseline_comparison']['train_freq_vs_zero_temporal']['pearson_r']:.4f}` (`ρ = {l1['baseline_comparison']['train_freq_vs_zero_temporal']['spearman_rho']:.4f}`); Topomap `300–500 ms`: `r = {l1['baseline_comparison']['train_freq_vs_zero_topo_300_500']['pearson_r']:.4f}` (`ρ = {l1['baseline_comparison']['train_freq_vs_zero_topo_300_500']['spearman_rho']:.4f}`).",
        f"- **Đối chiếu Baseline (`Train Frequent mean` vs `Test Frequent mean`)**: Temporal `r = {l1['baseline_comparison']['train_freq_vs_test_freq_temporal']['pearson_r']:.4f}`; Topomap `300–500 ms`: `r = {l1['baseline_comparison']['train_freq_vs_test_freq_topo_300_500']['pearson_r']:.4f}`.",
        "",
        "### Bảng 1 — Phân bố Năng lượng & Cường độ Attribution qua 4 Cửa sổ Sinh lý (39 subjects)",
        "",
        "| Cửa sổ thời gian | Số mẫu | Tỷ lệ thời lượng (%) | Tỷ trọng năng lượng `|IG|` (%) (Mean ± SD) | 95% CI (%) | Cường độ `mean |IG|` (Mean ± SD) |",
        "| :--- | ---: | ---: | :---: | :---: | :---: |",
    ]

    for w_name in TIME_WINDOWS_ORDER:
        ws = l1["windows"][w_name]
        es = ws["energy_share_pct"]
        ma = ws["mean_abs_ig"]
        lines.append(
            f"| {ws['label']} | {ws['n_samples']} | {ws['time_fraction_pct']:.1f}% | "
            f"{es['mean']:.2f} ± {es['std']:.2f}% | ±{es['ci95']:.2f}% | {ma['mean']:.5f} ± {ma['std']:.5f} |"
        )

    top_sp_str = ", ".join(
        f"`{item['channel']}` ({item['mean_abs_weight']:.4f})"
        for item in l1["spatial_weights_top_channels"][:7]
    )
    lines.extend([
        "",
        f"- **Top 7 kênh có trọng số Depthwise Spatial Conv (`|W_spatial|`) cao nhất qua 39 folds LOSO**: {top_sp_str}.",
        "",
        "---",
        "",
        "## Lớp 2 — Kiểm chứng Nhiễu Chớp mắt EOG (`clean` vs `no_ica`, CS 5-Fold)",
        "",
        f"- **Cụm `EOG_ROI` (8/127 kênh = `{l2['uniform_baseline_share_pct']:.2f}%` nếu phân bố đều)**: `Fp1, Fp2, AF7, AF3, AF8, AF4, AFp1, AFp2`.",
    ])

    m11 = l2["matched_1to1_subject_level"]
    mfu = l2["full_set_subject_level"]
    c_perf = l2["classification_performance"]
    lines.extend([
        f"- **Hiệu năng phân loại CS5**: `clean` ROC-AUC = **`{c_perf['clean_roc_auc']['mean']:.4f} ± {c_perf['clean_roc_auc']['std']:.4f}`** (F1 = `{c_perf['clean_f1_target']['mean']:.4f}`) vs `no_ica` ROC-AUC = **`{c_perf['noica_roc_auc']['mean']:.4f} ± {c_perf['noica_roc_auc']['std']:.4f}`** (F1 = `{c_perf['noica_f1_target']['mean']:.4f}`).",
        "",
        "### Bảng 2 — Tỷ trọng Attribution rơi vào cụm trán `EOG_ROI` Trước vs. Sau ICA (39 subjects)",
        "",
        "| Chỉ số `EOG_ROI` | `clean` (N=2,850 Target) | `no_ica` khớp 1-1 (N=2,850) | `no_ica` toàn tập (N=3,002) | Wilcoxon p (khớp 1-1) |",
        "| :--- | :---: | :---: | :---: | ---: |",
        f"| Tỷ trọng toàn epoch (%) | {m11['clean_share_full_pct']['mean']:.2f} ± {m11['clean_share_full_pct']['std']:.2f}% | {m11['noica_share_full_pct']['mean']:.2f} ± {m11['noica_share_full_pct']['std']:.2f}% | {mfu['noica_full_share_full_pct']['mean']:.2f} ± {mfu['noica_full_share_full_pct']['std']:.2f}% | {_fmt_pval(m11['wilcoxon_share_full']['wilcoxon_p'])} |",
        f"| Tỷ trọng cửa sổ `300–600 ms` (%) | {m11['clean_share_p300_pct']['mean']:.2f} ± {m11['clean_share_p300_pct']['std']:.2f}% | {m11['noica_share_p300_pct']['mean']:.2f} ± {m11['noica_share_p300_pct']['std']:.2f}% | {mfu['noica_full_share_p300_pct']['mean']:.2f} ± {mfu['noica_full_share_p300_pct']['std']:.2f}% | {_fmt_pval(m11['wilcoxon_share_p300']['wilcoxon_p'])} |",
        f"| Mật độ TB mỗi kênh `EOG_ROI` | {m11['clean_eog_density']['mean']:.5f} ± {m11['clean_eog_density']['std']:.5f} | {m11['noica_eog_density']['mean']:.5f} ± {m11['noica_eog_density']['std']:.5f} | {mfu['noica_full_eog_density']['mean']:.5f} ± {mfu['noica_full_eog_density']['std']:.5f} | {_fmt_pval(m11['wilcoxon_eog_density']['wilcoxon_p'])} |",
        f"| Tỷ số mật độ `EOG / Non-EOG` | {m11['clean_density_ratio']['mean']:.3f} ± {m11['clean_density_ratio']['std']:.3f} | {m11['noica_density_ratio']['mean']:.3f} ± {m11['noica_density_ratio']['std']:.3f} | — | {_fmt_pval(m11['wilcoxon_density_ratio']['wilcoxon_p'])} |",
        "",
        "#### Kiểm tra riêng 2 ca nội suy đồng thời `Fp1/Fp2` (`sub-18`, `sub-40`)",
        "",
        "| Nhóm subject | `clean` EOG share (%) | `no_ica` khớp 1-1 EOG share (%) | `no_ica` toàn tập (%) | Chênh lệch (clean − no_ica) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])

    interp = l2["fp1_fp2_interpolated_check"]
    for sub_id, vals in interp["interpolated_subjects"].items():
        lines.append(
            f"| `{sub_id}` (nội suy Fp1/Fp2) | {vals['clean_eog_share_full_pct']:.2f}% | "
            f"{vals['noica_matched_eog_share_full_pct']:.2f}% | {vals['noica_full_eog_share_full_pct']:.2f}% | "
            f"{vals['delta_matched_pct']:+.2f}% |"
        )
    o37_c = interp["others_37_clean_share_pct"]
    o37_n = interp["others_37_noica_matched_share_pct"]
    lines.append(
        f"| 37 subjects còn lại (Mean ± SD) | {o37_c['mean']:.2f} ± {o37_c['std']:.2f}% | "
        f"{o37_n['mean']:.2f} ± {o37_n['std']:.2f}% | — | {o37_c['mean'] - o37_n['mean']:+.2f}% (p={_fmt_pval(interp['others_37_wilcoxon']['wilcoxon_p'])}) |"
    )

    # ------------------------------------------------------------------
    # LỚP 3 — Motor Confound & Early-Window Cutoff
    # ------------------------------------------------------------------
    r36 = l3["roi_300_600"]
    r35 = l3["roi_300_500"]
    corr_rt = l3["rt_vs_performance_correlation"]
    ew = l3["early_window_cutoff"]

    lines.extend([
        "",
        "---",
        "",
        "## Lớp 3 — Bóc tách Nhiễu Vận động Bấm tay (Motor Confound — Trọng tâm RQ3)",
        "",
        "### Bảng 3A — Tỷ số ROI Không gian & Chỉ số Lệch Bán cầu Vận động (39 subjects LOSO)",
        "",
        "| Cửa sổ | `P300_ROI` (7 chs) | `LEFT_MOTOR_ROI` (6 chs) | `RIGHT_MOTOR_ROI` (6 chs) | Tỷ số `P300 / Left_Motor` | `Motor Laterality (L−R)/(L+R)` | Wilcoxon `P300 vs Left` | Wilcoxon `Left vs Right` |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
        f"| `300–600 ms` | {r36['p300_roi']['mean']:.5f} ± {r36['p300_roi']['std']:.5f} | "
        f"{r36['left_motor_roi']['mean']:.5f} ± {r36['left_motor_roi']['std']:.5f} | "
        f"{r36['right_motor_roi']['mean']:.5f} ± {r36['right_motor_roi']['std']:.5f} | "
        f"**{r36['p300_to_left_motor_ratio']['mean']:.2f} ± {r36['p300_to_left_motor_ratio']['std']:.2f}×** | "
        f"**{r36['motor_laterality']['mean']:+.4f} ± {r36['motor_laterality']['std']:.4f}** | "
        f"{_fmt_pval(r36['wilcoxon_p300_vs_left_motor']['wilcoxon_p'])} | "
        f"{_fmt_pval(r36['wilcoxon_left_vs_right_motor']['wilcoxon_p'])} |",
        f"| `300–500 ms` | {r35['p300_roi']['mean']:.5f} ± {r35['p300_roi']['std']:.5f} | "
        f"{r35['left_motor_roi']['mean']:.5f} ± {r35['left_motor_roi']['std']:.5f} | "
        f"{r35['right_motor_roi']['mean']:.5f} ± {r35['right_motor_roi']['std']:.5f} | "
        f"{r35['p300_roi']['mean'] / max(r35['left_motor_roi']['mean'], 1e-15):.2f}× | "
        f"{r35['motor_laterality']['mean']:+.4f} ± {r35['motor_laterality']['std']:.4f} | "
        f"— | {_fmt_pval(r35['wilcoxon_left_vs_right_motor']['wilcoxon_p'])} |",
        "",
        "### Bảng 3B — Phân tầng Reaction Time (`Fast < 450 ms`, `Mid 450–550 ms`, `Slow > 550 ms`)",
        "",
        "| Nhóm RT | Số trials | RT (ms) (Mean ± SD) | Peak XAI (ms) | `P300 / Left_Motor` (`300–600`) | Laterality (`300–600`) | `P300 / Left_Motor` (`450–650`) | Laterality (`450–650`) |",
        "| :--- | ---: | :---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for g_key, g_label in (("fast", "Fast (`< 450 ms`)"), ("mid", "Mid (`450–550 ms`)"), ("slow", "Slow (`> 550 ms`)")):
        gd = l3["rt_stratified"][g_key]
        lines.append(
            f"| {g_label} | {gd['n_trials']} | {gd['rt_ms']['mean']:.1f} ± {gd['rt_ms']['std']:.1f} | "
            f"**{gd['peak_xai_latency_ms']:.1f}** | "
            f"{gd['roi_300_600']['p300_to_left_motor_ratio']:.2f}× | {gd['roi_300_600']['motor_laterality_mean']:+.4f} | "
            f"{gd['roi_450_650']['p300_to_left_motor_ratio']:.2f}× | {gd['roi_450_650']['motor_laterality_mean']:+.4f} |"
        )

    c_roc = corr_rt["mean_rt_vs_loso_roc_auc"]
    c_pr = corr_rt["mean_rt_vs_loso_pr_auc"]
    c_ba = corr_rt["mean_rt_vs_loso_bacc"]
    lines.extend([
        "",
        f"- **Tương quan `subject_mean_rt_ms` vs hiệu năng LOSO (39 subjects)**:",
        f"  - vs `LOSO ROC-AUC`: Spearman `ρ = {c_roc['spearman_rho']:+.4f}` (`p = {_fmt_pval(c_roc['spearman_p'])}`), Pearson `r = {c_roc['pearson_r']:+.4f}` (`p = {_fmt_pval(c_roc['pearson_p'])}`).",
        f"  - vs `LOSO PR-AUC`: Spearman `ρ = {c_pr['spearman_rho']:+.4f}` (`p = {_fmt_pval(c_pr['spearman_p'])}`).",
        f"  - vs `LOSO Balanced Acc`: Spearman `ρ = {c_ba['spearman_rho']:+.4f}` (`p = {_fmt_pval(c_ba['spearman_p'])}`).",
        "",
        "### Bảng 3C — Kiểm chứng Cửa sổ Cắt Sớm (`Early-Window Cutoff <= 450 ms`, Pre-Median-RT)",
        "",
        "| Giao thức / Mô hình | Cửa sổ tín hiệu | ROC-AUC | PR-AUC | Balanced Acc | F1-Target | ΔROC-AUC vs Gốc |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | ---: |",
    ])

    if ew is not None:
        l_orig = ew["loso_inference_masking"]["comparison_vs_orig"]["roc_auc"]
        l_mask = ew["loso_inference_masking"]["summary"]
        lines.append(
            f"| LOSO 39 folds (Gốc đủ cửa sổ) | `[-200, 800] ms` | "
            f"{l_orig['orig_mean']:.4f} ± {l_orig['orig_std']:.4f} | "
            f"{ew['loso_inference_masking']['comparison_vs_orig']['pr_auc']['orig_mean']:.4f} ± {ew['loso_inference_masking']['comparison_vs_orig']['pr_auc']['orig_std']:.4f} | "
            f"{ew['loso_inference_masking']['comparison_vs_orig']['balanced_accuracy']['orig_mean']:.4f} ± {ew['loso_inference_masking']['comparison_vs_orig']['balanced_accuracy']['orig_std']:.4f} | "
            f"{ew['loso_inference_masking']['comparison_vs_orig']['f1_target']['orig_mean']:.4f} ± {ew['loso_inference_masking']['comparison_vs_orig']['f1_target']['orig_std']:.4f} | — |"
        )
        lines.append(
            f"| LOSO 39 folds (Inference-only mask `> 450 ms`) | `[-200, 450] ms` | "
            f"{l_mask['roc_auc']['mean']:.4f} ± {l_mask['roc_auc']['std']:.4f} | "
            f"{l_mask['pr_auc']['mean']:.4f} ± {l_mask['pr_auc']['std']:.4f} | "
            f"{l_mask['balanced_accuracy']['mean']:.4f} ± {l_mask['balanced_accuracy']['std']:.4f} | "
            f"{l_mask['f1_target']['mean']:.4f} ± {l_mask['f1_target']['std']:.4f} | "
            f"{l_orig['mean_diff']:+.4f} (p={_fmt_pval(l_orig['wilcoxon_p'])}) |"
        )

        cs5_c = ew["cs5_comparison"]
        lines.append(
            f"| CS 5-Fold (Gốc đủ cửa sổ) | `[-200, 800] ms` | "
            f"{cs5_c['roc_auc']['orig_mean']:.4f} ± {cs5_c['roc_auc']['orig_std']:.4f} | "
            f"{cs5_c['pr_auc']['orig_mean']:.4f} ± {cs5_c['pr_auc']['orig_std']:.4f} | "
            f"{cs5_c['balanced_accuracy']['orig_mean']:.4f} ± {cs5_c['balanced_accuracy']['orig_std']:.4f} | "
            f"{cs5_c['f1_target']['orig_mean']:.4f} ± {cs5_c['f1_target']['orig_std']:.4f} | — |"
        )
        lines.append(
            f"| CS 5-Fold (Inference-only mask `> 450 ms`) | `[-200, 450] ms` | "
            f"{cs5_c['roc_auc']['inference_masked_mean']:.4f} ± {cs5_c['roc_auc']['inference_masked_std']:.4f} | "
            f"{cs5_c['pr_auc']['inference_masked_mean']:.4f} ± {cs5_c['pr_auc']['inference_masked_std']:.4f} | "
            f"{cs5_c['balanced_accuracy']['inference_masked_mean']:.4f} ± {cs5_c['balanced_accuracy']['inference_masked_std']:.4f} | "
            f"{cs5_c['f1_target']['inference_masked_mean']:.4f} ± {cs5_c['f1_target']['inference_masked_std']:.4f} | "
            f"{cs5_c['roc_auc']['delta_masked_vs_orig']:+.4f} |"
        )
        lines.append(
            f"| **CS 5-Fold (Retrain-from-scratch `n_times=163`)** | **`[-200, 450] ms`** | "
            f"**{cs5_c['roc_auc']['retrain_early_mean']:.4f} ± {cs5_c['roc_auc']['retrain_early_std']:.4f}** | "
            f"**{cs5_c['pr_auc']['retrain_early_mean']:.4f} ± {cs5_c['pr_auc']['retrain_early_std']:.4f}** | "
            f"**{cs5_c['balanced_accuracy']['retrain_early_mean']:.4f} ± {cs5_c['balanced_accuracy']['retrain_early_std']:.4f}** | "
            f"**{cs5_c['f1_target']['retrain_early_mean']:.4f} ± {cs5_c['f1_target']['retrain_early_std']:.4f}** | "
            f"**{cs5_c['roc_auc']['delta_retrain_vs_orig']:+.4f}** |"
        )

    # ------------------------------------------------------------------
    # LỚP 4 — Frequency Occlusion, Spatial Correlation & Error Analysis
    # ------------------------------------------------------------------
    fo = l4["frequency_occlusion"]
    sc = l4["spatial_correlation"]
    ea = l4["error_analysis"]

    lines.extend([
        "",
        "---",
        "",
        "## Lớp 4 — Chiều Tần số, Đối chiếu ERP & Error Analysis",
        "",
        "### Bảng 4A — Frequency Occlusion trên toàn bộ 39 folds LOSO (Butterworth SOS bậc 4)",
        "",
        "| Dải tần bị triệt tiêu | ROC-AUC sau occlusion | `ΔROC-AUC` (Mean ± SD) | Wilcoxon p (ROC) | PR-AUC sau occlusion | `ΔPR-AUC` (Mean ± SD) | Wilcoxon p (PR) |",
        "| :--- | :---: | :---: | ---: | :---: | :---: | ---: |",
    ])

    for b_key in ("sub_delta", "delta", "theta", "alpha", "beta", "gamma", "delta_theta"):
        if b_key not in fo:
            continue
        bd = fo[b_key]
        dr = bd["delta_vs_orig"]["roc_auc"]
        dp = bd["delta_vs_orig"]["pr_auc"]
        lines.append(
            f"| {bd['label']} | {dr['occluded_mean']:.4f} ± {dr['occluded_std']:.4f} | "
            f"**{dr['delta_mean']:+.4f} ± {dr['delta_std']:.4f}** | {_fmt_pval(dr['wilcoxon_p'])} | "
            f"{dp['occluded_mean']:.4f} ± {dp['occluded_std']:.4f} | "
            f"**{dp['delta_mean']:+.4f} ± {dp['delta_std']:.4f}** | {_fmt_pval(dp['wilcoxon_p'])} |"
        )

    c_35_abs = sc["win_300_500_abs_ig_vs_erp_diff"]
    c_35_abs_abs = sc["win_300_500_abs_ig_vs_abs_erp_diff"]
    c_35_sgn = sc["win_300_500_signed_ig_vs_erp_diff"]
    c_36_abs = sc["win_300_600_abs_ig_vs_erp_diff"]
    c_36_abs_abs = sc["win_300_600_abs_ig_vs_abs_erp_diff"]
    c_36_sgn = sc["win_300_600_signed_ig_vs_erp_diff"]

    lines.extend([
        "",
        "### Bảng 4B — Tương quan Không gian 127 Kênh (XAI `IG` vs Grand Average ERP `Target − Frequent`)",
        "",
        "| Cửa sổ | So sánh cặp bản đồ 127 kênh | Pearson `r` (p-value) | Spearman `ρ` (p-value) | Cosine Sim |",
        "| :--- | :--- | :---: | :---: | ---: |",
        f"| `300–500 ms` | `|IG|` vs `ERP (Target − Frequent)` | {c_35_abs['pearson_r']:+.4f} ({_fmt_pval(c_35_abs['pearson_p'])}) | {c_35_abs['spearman_rho']:+.4f} ({_fmt_pval(c_35_abs['spearman_p'])}) | {c_35_abs['cosine_sim']:+.4f} |",
        f"| `300–500 ms` | `|IG|` vs `|ERP (Target − Frequent)|` | **{c_35_abs_abs['pearson_r']:+.4f}** ({_fmt_pval(c_35_abs_abs['pearson_p'])}) | **{c_35_abs_abs['spearman_rho']:+.4f}** ({_fmt_pval(c_35_abs_abs['spearman_p'])}) | {c_35_abs_abs['cosine_sim']:+.4f} |",
        f"| `300–500 ms` | `Signed IG` vs `ERP (Target − Frequent)` | {c_35_sgn['pearson_r']:+.4f} ({_fmt_pval(c_35_sgn['pearson_p'])}) | {c_35_sgn['spearman_rho']:+.4f} ({_fmt_pval(c_35_sgn['spearman_p'])}) | {c_35_sgn['cosine_sim']:+.4f} |",
        f"| `300–600 ms` | `|IG|` vs `ERP (Target − Frequent)` | {c_36_abs['pearson_r']:+.4f} ({_fmt_pval(c_36_abs['pearson_p'])}) | {c_36_abs['spearman_rho']:+.4f} ({_fmt_pval(c_36_abs['spearman_p'])}) | {c_36_abs['cosine_sim']:+.4f} |",
        f"| `300–600 ms` | `|IG|` vs `|ERP (Target − Frequent)|` | {c_36_abs_abs['pearson_r']:+.4f} ({_fmt_pval(c_36_abs_abs['pearson_p'])}) | {c_36_abs_abs['spearman_rho']:+.4f} ({_fmt_pval(c_36_abs_abs['spearman_p'])}) | {c_36_abs_abs['cosine_sim']:+.4f} |",
        f"| `300–600 ms` | `Signed IG` vs `ERP (Target − Frequent)` | {c_36_sgn['pearson_r']:+.4f} ({_fmt_pval(c_36_sgn['pearson_p'])}) | {c_36_sgn['spearman_rho']:+.4f} ({_fmt_pval(c_36_sgn['spearman_p'])}) | {c_36_sgn['cosine_sim']:+.4f} |",
        "",
        f"- **Tương quan không gian trung bình từng subject (`300–500 ms`, `|IG|` vs `|ERP diff|`, 39 subjects)**: `r = {sc['per_subject_300_500_abs_ig_vs_abs_erp_r']['mean']:+.4f} ± {sc['per_subject_300_500_abs_ig_vs_abs_erp_r']['std']:.4f}`, `ρ = {sc['per_subject_300_500_abs_ig_vs_abs_erp_rho']['mean']:+.4f} ± {sc['per_subject_300_500_abs_ig_vs_abs_erp_rho']['std']:.4f}`.",
        "",
        "### Bảng 4C — Error Analysis (`True Positive` vs `False Negative` trên 2,850 Target trials LOSO)",
        "",
        f"- **True Positive (TP)**: **`{ea['n_tp']:,}`** trials (`{ea['tp_rate_pct']:.2f}%`) | **False Negative (FN)**: **`{ea['n_fn']:,}`** trials (`{ea['fn_rate_pct']:.2f}%`).",
        f"- **Biên độ ERP `Pz` trung bình trong `[300, 600] ms`**: TP = **`{ea['tp_pz_p300_amp_uv']['mean']:+.2f} ± {ea['tp_pz_p300_amp_uv']['std']:.2f} µV`** vs FN = **`{ea['fn_pz_p300_amp_uv']['mean']:+.2f} ± {ea['fn_pz_p300_amp_uv']['std']:.2f} µV`** (Mann-Whitney `p = {_fmt_pval(ea['pz_p300_amp_mannwhitney_p'])}`).",
        f"- **Phân bố Reaction Time (`rt_ms`)**: TP = **`{ea['tp_rt_ms']['mean']:.1f} ± {ea['tp_rt_ms']['std']:.1f} ms`** (median `{ea['tp_rt_ms']['median']:.1f} ms`) vs FN = **`{ea['fn_rt_ms']['mean']:.1f} ± {ea['fn_rt_ms']['std']:.1f} ms`** (median `{ea['fn_rt_ms']['median']:.1f} ms`, Mann-Whitney `p = {_fmt_pval(ea['rt_mannwhitney_p'])}`).",
        f"- **Cường độ `|IG|` tại `P300_ROI` (`300–600 ms`)**: TP = `{ea['tp_p300_roi_abs_ig']['mean']:.5f}` vs FN = `{ea['fn_p300_roi_abs_ig']['mean']:.5f}` (Mann-Whitney `p = {_fmt_pval(ea['p300_roi_ig_mannwhitney_p'])}`).",
        "",
        "#### Phân tích cụm Bottom-5 LOSO (`sub-07, sub-01, sub-41, sub-32, sub-21`)",
        "",
        "| Subject | Target (TP / FN) | FN Rate (%) | Mean RT (ms) | `% RT > 550ms` | `Pz` P300 Mean (µV) | LOSO ROC-AUC | LOSO PR-AUC | `P300/Left_Motor` | Laterality |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for b in ea["bottom5_loso"]:
        lines.append(
            f"| `{b['subject_id']}` | {b['n_target']} ({b['tp']} / {b['fn']}) | {b['fn_rate_pct']:.1f}% | "
            f"{b['mean_rt_ms']:.1f} | {b['pct_rt_gt_550ms']:.1f}% | {b['pz_p300_mean_uv']:+.2f} | "
            f"{b['loso_roc_auc']:.4f} | {b['loso_pr_auc']:.4f} | "
            f"{b['p300_to_left_motor_ratio']:.2f}× | {b['motor_laterality']:+.4f} |"
        )

    add65 = summary.get("addendum_step6_5")
    if add65 is not None:
        ew_rt = add65["early_window_rt_and_xai"]
        rel_c = add65["relative_completeness"]
        b5_diag = add65["bottom5_threshold_diagnostic"]

        lines.extend([
            "",
            "---",
            "",
            "## Phụ lục Bước 6.5 — Kiểm chứng Bổ sung Phản biện (Addendum RQ3)",
            "",
            "### Phụ lục A — Hiệu năng & XAI theo phân tầng RT của mô hình Retrain Early-Window (`[-200, 450] ms`, `n_times=163`)",
            "",
            "| Phân tầng Target | Số Target (N) | Early450 Fold ROC-AUC (Mean ± SD) | Early450 Fold PR-AUC (Mean ± SD) | Early450 Pooled ROC / PR | Early450 Sens (%) | Full `[-200, 800]` Fold ROC-AUC | Full Fold PR-AUC |",
            "| :--- | ---: | :---: | :---: | :---: | ---: | :---: | :---: |",
        ])
        for st_key in ("all", "fast", "mid", "slow", "very_slow"):
            st_e = ew_rt["performance"]["early450"][st_key]
            st_f = ew_rt["performance"]["full_clean"][st_key]
            lines.append(
                f"| {st_e['label']} | {st_e['n_target_total']:,} | "
                f"{st_e['fold_roc_auc']['mean']:.4f} ± {st_e['fold_roc_auc']['std']:.4f} | "
                f"{st_e['fold_pr_auc']['mean']:.4f} ± {st_e['fold_pr_auc']['std']:.4f} | "
                f"{st_e['pooled']['roc_auc']:.4f} / {st_e['pooled']['pr_auc']:.4f} | "
                f"{st_e['pooled']['sensitivity'] * 100.0:.2f}% | "
                f"{st_f['fold_roc_auc']['mean']:.4f} ± {st_f['fold_roc_auc']['std']:.4f} | "
                f"{st_f['fold_pr_auc']['mean']:.4f} ± {st_f['fold_pr_auc']['std']:.4f} |"
            )

        x_e = ew_rt["xai_300_450_early450"]
        x_f = ew_rt["xai_300_450_full_cs5_ref"]
        lines.extend([
            "",
            "#### Tỷ số ROI `[300, 450] ms` trên mô hình Retrain `[-200, 450] ms` (Captum IG, `n_steps=64`, baseline = `Frequent mean` test fold)",
            "",
            "| Mô hình & Phân nhóm Target | Đơn vị / N | `P300_ROI` (7 chs) | `LEFT_MOTOR_ROI` (6 chs) | `RIGHT_MOTOR_ROI` (6 chs) | `P300 / Left_Motor` | `Motor Laterality (L−R)/(L+R)` | Wilcoxon `P300 vs Left` | Wilcoxon `Left vs Right` |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | ---: | ---: |",
            f"| Full `[-200, 800]` CS5 (All Target, `[300, 450]`) | 39 subs | {x_f['p300_roi']['mean']:.5f} | {x_f['left_motor_roi']['mean']:.5f} | {x_f['right_motor_roi']['mean']:.5f} | **{x_f['p300_to_left_motor_ratio']['mean']:.2f}×** | **{x_f['motor_laterality']['mean']:+.4f}** | {_fmt_pval(x_f['wilcoxon_p300_vs_left_motor']['wilcoxon_p'])} | {_fmt_pval(x_f['wilcoxon_left_vs_right_motor']['wilcoxon_p'])} |",
            f"| **Early450 `[-200, 450]` (All Target, subject-level)** | **39 subs** | {x_e['subject_level']['p300_roi']['mean']:.5f} | {x_e['subject_level']['left_motor_roi']['mean']:.5f} | {x_e['subject_level']['right_motor_roi']['mean']:.5f} | **{x_e['subject_level']['p300_to_left_motor_ratio']['mean']:.2f} ± {x_e['subject_level']['p300_to_left_motor_ratio']['std']:.2f}×** | **{x_e['subject_level']['motor_laterality']['mean']:+.4f} ± {x_e['subject_level']['motor_laterality']['std']:.4f}** | {_fmt_pval(x_e['subject_level']['wilcoxon_p300_vs_left_motor']['wilcoxon_p'])} | {_fmt_pval(x_e['subject_level']['wilcoxon_left_vs_right_motor']['wilcoxon_p'])} |",
        ])
        for st_key in ("all", "fast", "mid", "slow", "very_slow"):
            st_roi = x_e["trial_strata"][st_key]
            lines.append(
                f"| Early450 (`{st_roi['label']}`) | {st_roi['n_trials']:,} trials | "
                f"{st_roi['p300_roi_mean']:.5f} | {st_roi['left_motor_mean']:.5f} | {st_roi['right_motor_mean']:.5f} | "
                f"**{st_roi['p300_to_left_motor_ratio']:.2f}×** | **{st_roi['motor_laterality_mean']:+.4f}** | "
                f"{_fmt_pval(st_roi['wilcoxon_p300_vs_left_motor']['wilcoxon_p'])} | {_fmt_pval(st_roi['wilcoxon_left_vs_right_motor']['wilcoxon_p'])} |"
            )

        abs_st = rel_c["abs_delta_stats"]
        rel_st = rel_c["rel_delta_pct_stats"]
        lines.extend([
            "",
            "### Phụ lục B — Sai số Tương đối Completeness của Captum IG trên toàn bộ `2,850` Target trials LOSO",
            "",
            f"- **`|delta|` (tuyệt đối)**: Median = `{abs_st['median']:.4e}`, P95 = `{abs_st['p95']:.4e}`, P99 = `{abs_st['p99']:.4e}`, Max = `{abs_st['max']:.4e}` (Mean = `{abs_st['mean']:.4e}`).",
            f"- **`rel_delta = |delta| / (|F(x) − F(x')| + 1e-8)`**: Median = **`{rel_st['median']:.4f}%`**, P95 = **`{rel_st['p95']:.4f}%`**, P99 = **`{rel_st['p99']:.4f}%`**, Max = **`{rel_st['max']:.4f}%`** (Mean = `{rel_st['mean']:.4f}%`).",
            "",
            "| Hạng | `trial_uid` | Subject | `rt_ms` | `F(x)` | `F(x')` | `F(x) − F(x')` | `|delta|` | `rel_delta (%)` |",
            "| ---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for rank_i, t_row in enumerate(rel_c["top5_abs_delta"], start=1):
            lines.append(
                f"| #{rank_i} | `{t_row['trial_uid']}` | `{t_row['subject_id']}` | {t_row['rt_ms']:.1f} | "
                f"{t_row['f_x']:+.4f} | {t_row['f_baseline']:+.4f} | {t_row['logit_diff']:+.4f} | "
                f"{t_row['abs_delta']:.5f} | {t_row['rel_delta_pct']:.4f}% |"
            )

        lines.extend([
            "",
            "### Phụ lục C — Chẩn đoán Threshold Miscalibration trên cụm Bottom-5 LOSO *(Diagnostic Only — Không thay thế điểm test chính thức)*",
            "",
            "| Subject | `thr_loso` | Gốc: Sens / Spec / F1 / FN (FN%) | `thr_oracle (bAcc)` | Oracle bAcc: Sens / Spec / F1 / FN (FN%) | FN khôi phục | `thr_oracle (F1)` (F1) |",
            "| :--- | ---: | :---: | ---: | :---: | :---: | :---: |",
        ])
        for row in b5_diag["per_subject"]:
            o = row["orig_loso"]
            ob = row["oracle_bacc"]
            of1 = row["oracle_f1"]
            lines.append(
                f"| `{row['subject_id']}` | {row['thr_loso']:.3f} | "
                f"{o['sensitivity'] * 100:.1f}% / {o['specificity'] * 100:.1f}% / {o['f1_target']:.3f} / {o['fn']} ({o['fn_rate_pct']:.1f}%) | "
                f"{row['thr_oracle_bacc']:.3f} | "
                f"{ob['sensitivity'] * 100:.1f}% / {ob['specificity'] * 100:.1f}% / {ob['f1_target']:.3f} / {ob['fn']} ({ob['fn_rate_pct']:.1f}%) | "
                f"**{row['fn_recovered_bacc']}/{o['fn']} ({row['fn_recovered_pct_of_orig_fn']:.1f}%)** | "
                f"{row['thr_oracle_f1']:.3f} ({of1['f1_target']:.3f}) |"
            )

    lines.append("")
    out_path = Path(report_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[XAI-RQ3] Đã sinh báo cáo tại {out_path}.")
    return out_path


# ==============================================================================
# BƯỚC 6.5 — CÁC HÀM KIỂM CHỨNG BỔ SUNG (ADDENDUM VERIFICATION)
# ==============================================================================
def _distribution_stats(arr: np.ndarray) -> Dict[str, Any]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "median": None, "p95": None, "p99": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=1 if a.size > 1 else 0)),
        "median": float(np.median(a)),
        "p95": float(np.percentile(a, 95.0)),
        "p99": float(np.percentile(a, 99.0)),
        "max": float(np.max(a)),
    }


def analyze_early_window_rt_and_xai(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    *,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Đánh giá hiệu năng & Captum IG theo phân tầng RT trên mô hình Retrain Early-Window `[-200, 450] ms`."""
    torch_device = _resolve_device(device)
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    ckpt_root = out_root / "checkpoints" / "dl2"

    bundle = load_eeg_dataset(data_root, data_variant="clean")
    channel_names = load_channel_names(data_root)
    roi_idx_map = verify_rois(channel_names)
    times_ms = np.round(np.load(data_root / "times.npy") * 1000.0, 3)

    early_ckpt_dir = ckpt_root / "dl2_cs5_p300_atcnet_early450"
    early_pred_path = metrics_dir / "predictions_dl2_cs5_p300_atcnet_early450.csv"
    full_pred_path = metrics_dir / "predictions_dl2_cs5_p300_atcnet_clean.csv"
    if not early_pred_path.exists() or not (early_ckpt_dir / "fold0.pt").exists():
        run_early_window_cutoff(data_dir=data_root, output_dir=out_root, resume=True, device=torch_device)

    meta_df = bundle.metadata[["trial_uid", "subject_id", "label", "rt_ms", "rt_group"]].copy()
    meta_df["trial_uid"] = meta_df["trial_uid"].astype(str)

    strata_defs = (
        ("all", "All Targets (N=2,850)", lambda df: df["y_true"] == 1),
        ("fast", "Fast RT (< 450 ms)", lambda df: (df["y_true"] == 1) & (df["rt_ms"] < 450.0)),
        ("mid", "Mid RT (450–550 ms)", lambda df: (df["y_true"] == 1) & (df["rt_ms"] >= 450.0) & (df["rt_ms"] <= 550.0)),
        ("slow", "Slow RT (> 550 ms)", lambda df: (df["y_true"] == 1) & (df["rt_ms"] > 550.0)),
        ("very_slow", "Very Slow RT (> 600 ms)", lambda df: (df["y_true"] == 1) & (df["rt_ms"] > 600.0)),
    )

    def _eval_pred_strata(pred_csv: Path) -> Dict[str, Any]:
        p_df = pd.read_csv(pred_csv)
        merged = p_df.merge(meta_df[["trial_uid", "rt_ms", "rt_group"]], on="trial_uid", how="left")
        res: Dict[str, Any] = {}
        for st_key, st_label, tgt_fn in strata_defs:
            tgt_mask = tgt_fn(merged)
            freq_mask = merged["y_true"] == 0
            sub_all = merged.loc[tgt_mask | freq_mask]
            pooled_m = compute_binary_metrics(
                sub_all["y_true"].to_numpy(dtype=np.int64),
                sub_all["y_prob"].to_numpy(dtype=np.float64),
                threshold=float(np.median(sub_all["threshold"])),
            )
            # Sensitivity OOF chuẩn theo ngưỡng từng fold của chính trial đó
            oof_sens = float(
                np.mean(
                    merged.loc[tgt_mask, "y_pred"].to_numpy(dtype=np.int64)
                )
            )
            pooled_m["sensitivity"] = oof_sens

            fold_rocs, fold_prs, fold_sens = [], [], []
            for _, f_df in merged.groupby("fold", sort=True):
                f_tgt = tgt_fn(f_df)
                f_freq = f_df["y_true"] == 0
                f_sub = f_df.loc[f_tgt | f_freq]
                fm = compute_binary_metrics(
                    f_sub["y_true"].to_numpy(dtype=np.int64),
                    f_sub["y_prob"].to_numpy(dtype=np.float64),
                    threshold=float(f_sub["threshold"].iloc[0]),
                )
                fold_rocs.append(fm["roc_auc"])
                fold_prs.append(fm["pr_auc"])
                fold_sens.append(fm["sensitivity"])

            res[st_key] = {
                "label": st_label,
                "n_target_total": int(np.sum(tgt_mask)),
                "n_freq_total": int(np.sum(freq_mask)),
                "fold_roc_auc": _scalar_summary(np.array(fold_rocs)),
                "fold_pr_auc": _scalar_summary(np.array(fold_prs)),
                "fold_sensitivity": _scalar_summary(np.array(fold_sens)),
                "pooled": pooled_m,
            }
        return res

    perf_compare = {
        "early450": _eval_pred_strata(early_pred_path),
        "full_clean": _eval_pred_strata(full_pred_path),
    }

    # Chạy Captum IG trên 5 folds Retrain Early450 (n_times=163)
    keep_mask = times_ms <= EARLY_CUTOFF_MS + 1e-6
    n_times_early = int(np.sum(keep_mask))
    times_early_ms = times_ms[:n_times_early]
    mask_300_450 = (times_early_ms >= 300.0 - 1e-6) & (times_early_ms <= 450.0 + 1e-6)
    X_early = np.ascontiguousarray(bundle.X[:, :, :n_times_early], dtype=np.float32)

    cs5_folds = get_protocol_folds(bundle.splits, protocol="cross_subject_5fold")
    trial_subs: List[str] = []
    trial_rts: List[float] = []
    trial_p300: List[np.ndarray] = []
    trial_lm: List[np.ndarray] = []
    trial_rm: List[np.ndarray] = []
    sub_maps: Dict[str, List[np.ndarray]] = {}

    print("[Step 6.5] Chạy Captum IG trên 5 folds Retrain Early450 (n_times=163, baseline=test Frequent mean)...")
    for spec in cs5_folds:
        fold_key = str(spec["fold_key"])
        ckpt_path = early_ckpt_dir / f"{fold_key}.pt"
        model, scaler, _ = load_checkpoint_for_eval(ckpt_path, device=torch_device)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        te_freq_idx = te_idx[bundle.y[te_idx] == 0]
        te_tgt_idx = te_idx[bundle.y[te_idx] == 1]

        x_te_freq_scaled = scaler.transform(X_early[te_freq_idx])
        baseline_test_freq = np.mean(x_te_freq_scaled.astype(np.float64), axis=0, keepdims=True).astype(
            np.float32
        )
        x_te_tgt_scaled = scaler.transform(X_early[te_tgt_idx])

        attrs, _ = compute_captum_ig(
            model,
            x_te_tgt_scaled,
            baseline_test_freq,
            target=0,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            device=torch_device,
            return_convergence_delta=False,
        )
        abs_win = np.abs(attrs)[:, :, mask_300_450]  # (N_tgt, 127, 38)
        trial_p300.append(abs_win[:, roi_idx_map["p300"], :].mean(axis=(1, 2)))
        trial_lm.append(abs_win[:, roi_idx_map["left_motor"], :].mean(axis=(1, 2)))
        trial_rm.append(abs_win[:, roi_idx_map["right_motor"], :].mean(axis=(1, 2)))

        for local_i, g_idx in enumerate(te_tgt_idx):
            row = bundle.metadata.iloc[int(g_idx)]
            s_id = str(row["subject_id"])
            trial_subs.append(s_id)
            trial_rts.append(float(row["rt_ms"]))
            sub_maps.setdefault(s_id, []).append(np.abs(attrs[local_i]))

    t_p300 = np.concatenate(trial_p300, axis=0).astype(np.float64)
    t_lm = np.concatenate(trial_lm, axis=0).astype(np.float64)
    t_rm = np.concatenate(trial_rm, axis=0).astype(np.float64)
    t_rt = np.asarray(trial_rts, dtype=np.float64)

    ordered_subs = sorted(sub_maps.keys(), key=natural_sort_key)
    sub_abs_early = np.stack([np.mean(sub_maps[s], axis=0) for s in ordered_subs], axis=0).astype(np.float64)
    roi_sub_early = _compute_roi_densities_per_subject(sub_abs_early, roi_idx_map, mask_300_450)
    sub_ratio_early = roi_sub_early["p300"] / np.maximum(roi_sub_early["left_motor"], 1e-15)
    sub_lat_early = (roi_sub_early["left_motor"] - roi_sub_early["right_motor"]) / np.maximum(
        roi_sub_early["left_motor"] + roi_sub_early["right_motor"], 1e-15
    )

    trial_strata_roi: Dict[str, Any] = {}
    rt_strata_masks = (
        ("all", "All Targets", np.ones_like(t_rt, dtype=bool)),
        ("fast", "Fast RT < 450 ms", t_rt < 450.0),
        ("mid", "Mid RT 450–550 ms", (t_rt >= 450.0) & (t_rt <= 550.0)),
        ("slow", "Slow RT > 550 ms", t_rt > 550.0),
        ("very_slow", "Very Slow RT > 600 ms", t_rt > 600.0),
    )
    for k, lbl, m in rt_strata_masks:
        p_m, lm_m, rm_m = t_p300[m], t_lm[m], t_rm[m]
        lat_tr = (lm_m - rm_m) / np.maximum(lm_m + rm_m, 1e-15)
        trial_strata_roi[k] = {
            "label": lbl,
            "n_trials": int(np.sum(m)),
            "p300_roi_mean": float(np.mean(p_m)),
            "left_motor_mean": float(np.mean(lm_m)),
            "right_motor_mean": float(np.mean(rm_m)),
            "p300_to_left_motor_ratio": float(np.mean(p_m) / max(float(np.mean(lm_m)), 1e-15)),
            "motor_laterality_mean": float(np.mean(lat_tr)),
            "wilcoxon_p300_vs_left_motor": _paired_wilcoxon(p_m, lm_m),
            "wilcoxon_left_vs_right_motor": _paired_wilcoxon(lm_m, rm_m),
        }

    # Đối chiếu cùng cửa sổ [300, 450] ms trên Full CS5 Clean từ cache
    cache_path = metrics_dir / "xai_loso_cache.npz"
    cache = np.load(cache_path, allow_pickle=False)
    full_cs5_sub = cache["cs5_clean_per_sub_abs"].astype(np.float64)[:, :, :n_times_early]
    roi_sub_full_ref = _compute_roi_densities_per_subject(full_cs5_sub, roi_idx_map, mask_300_450)
    sub_ratio_full_ref = roi_sub_full_ref["p300"] / np.maximum(roi_sub_full_ref["left_motor"], 1e-15)
    sub_lat_full_ref = (roi_sub_full_ref["left_motor"] - roi_sub_full_ref["right_motor"]) / np.maximum(
        roi_sub_full_ref["left_motor"] + roi_sub_full_ref["right_motor"], 1e-15
    )

    return {
        "performance": perf_compare,
        "xai_300_450_early450": {
            "subject_level": {
                "n_subjects": len(ordered_subs),
                "p300_roi": _scalar_summary(roi_sub_early["p300"]),
                "left_motor_roi": _scalar_summary(roi_sub_early["left_motor"]),
                "right_motor_roi": _scalar_summary(roi_sub_early["right_motor"]),
                "p300_to_left_motor_ratio": _scalar_summary(sub_ratio_early),
                "motor_laterality": _scalar_summary(sub_lat_early),
                "wilcoxon_p300_vs_left_motor": _paired_wilcoxon(
                    roi_sub_early["p300"], roi_sub_early["left_motor"]
                ),
                "wilcoxon_left_vs_right_motor": _paired_wilcoxon(
                    roi_sub_early["left_motor"], roi_sub_early["right_motor"]
                ),
            },
            "trial_strata": trial_strata_roi,
        },
        "xai_300_450_full_cs5_ref": {
            "p300_roi": _scalar_summary(roi_sub_full_ref["p300"]),
            "left_motor_roi": _scalar_summary(roi_sub_full_ref["left_motor"]),
            "right_motor_roi": _scalar_summary(roi_sub_full_ref["right_motor"]),
            "p300_to_left_motor_ratio": _scalar_summary(sub_ratio_full_ref),
            "motor_laterality": _scalar_summary(sub_lat_full_ref),
            "wilcoxon_p300_vs_left_motor": _paired_wilcoxon(
                roi_sub_full_ref["p300"], roi_sub_full_ref["left_motor"]
            ),
            "wilcoxon_left_vs_right_motor": _paired_wilcoxon(
                roi_sub_full_ref["left_motor"], roi_sub_full_ref["right_motor"]
            ),
        },
    }


@torch.inference_mode()
def analyze_relative_completeness_error(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    *,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Kiểm tra sai số tương đối `|delta| / (|F(x) - F(x')| + 1e-8)` trên 2,850 Target trials LOSO."""
    torch_device = _resolve_device(device)
    data_root = Path(data_dir)
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    ckpt_root = out_root / "checkpoints" / "dl2"

    bundle = load_eeg_dataset(data_root, data_variant="clean")
    cache = np.load(metrics_dir / "xai_loso_cache.npz", allow_pickle=False)
    cache_uids = [str(u) for u in cache["target_trial_uids"].tolist()]
    cache_subs = [str(s) for s in cache["target_trial_subs"].tolist()]
    cache_rts = cache["target_trial_rt_ms"].astype(np.float64)
    cache_deltas = cache["target_trial_deltas"].astype(np.float64)

    loso_folds = get_protocol_folds(bundle.splits, protocol="cross_subject_loso")
    uid_to_logits: Dict[str, Tuple[float, float]] = {}

    for spec in loso_folds:
        fold_key = str(spec["fold_key"])
        ckpt_path = ckpt_root / "dl2_loso_p300_atcnet" / f"{fold_key}.pt"
        model, scaler, _ = load_checkpoint_for_eval(ckpt_path, device=torch_device)
        model.eval()

        tr_idx = np.asarray(spec["train_idx"], dtype=np.int64)
        te_idx = np.asarray(spec["test_idx"], dtype=np.int64)
        tr_freq_idx = tr_idx[bundle.y[tr_idx] == 0]
        te_tgt_idx = te_idx[bundle.y[te_idx] == 1]

        x_tr_freq_scaled = scaler.transform(bundle.X[tr_freq_idx])
        baseline_train_freq = np.mean(x_tr_freq_scaled.astype(np.float64), axis=0, keepdims=True).astype(
            np.float32
        )
        x_te_tgt_scaled = scaler.transform(bundle.X[te_tgt_idx])

        fb = float(model(torch.from_numpy(baseline_train_freq).to(torch_device))[0, 0].item())
        fx_all = model(torch.from_numpy(x_te_tgt_scaled).to(torch_device))[:, 0].cpu().numpy().astype(np.float64)
        for local_i, g_idx in enumerate(te_tgt_idx):
            uid = str(bundle.metadata.iloc[int(g_idx)]["trial_uid"])
            uid_to_logits[uid] = (float(fx_all[local_i]), fb)

    fx_arr = np.array([uid_to_logits[u][0] for u in cache_uids], dtype=np.float64)
    fb_arr = np.array([uid_to_logits[u][1] for u in cache_uids], dtype=np.float64)
    logit_diff = fx_arr - fb_arr
    abs_delta = np.abs(cache_deltas)
    rel_delta = abs_delta / (np.abs(logit_diff) + 1e-8)
    rel_delta_pct = rel_delta * 100.0

    top5_idx = np.argsort(abs_delta)[::-1][:5]
    top5_records = []
    for idx in top5_idx:
        i = int(idx)
        top5_records.append(
            {
                "trial_uid": cache_uids[i],
                "subject_id": cache_subs[i],
                "rt_ms": float(cache_rts[i]),
                "f_x": float(fx_arr[i]),
                "f_baseline": float(fb_arr[i]),
                "logit_diff": float(logit_diff[i]),
                "abs_delta": float(abs_delta[i]),
                "rel_delta": float(rel_delta[i]),
                "rel_delta_pct": float(rel_delta_pct[i]),
            }
        )

    return {
        "n_trials": len(cache_uids),
        "abs_delta_stats": _distribution_stats(abs_delta),
        "rel_delta_stats": _distribution_stats(rel_delta),
        "rel_delta_pct_stats": _distribution_stats(rel_delta_pct),
        "abs_logit_diff_stats": _distribution_stats(np.abs(logit_diff)),
        "top5_abs_delta": top5_records,
    }


def analyze_bottom5_threshold_miscalibration(
    output_dir: Union[str, Path] = Path("results"),
) -> Dict[str, Any]:
    """Chẩn đoán hậu nghiệm (diagnostic only) độ lệch ngưỡng trên cụm Bottom-5 LOSO."""
    metrics_dir = Path(output_dir) / "metrics" / "dl2"
    pred_path = metrics_dir / "handover_predictions_dl2_loso.csv"
    df = pd.read_csv(pred_path)

    rows: List[Dict[str, Any]] = []
    for sub_id in BOTTOM5_LOSO_SUBJECTS:
        sub_df = df.loc[df["subject_id"].astype(str) == sub_id]
        y_true = sub_df["y_true"].to_numpy(dtype=np.int64)
        y_prob = sub_df["y_prob"].to_numpy(dtype=np.float64)
        thr_loso = float(sub_df["threshold"].iloc[0])

        m_orig = compute_binary_metrics(y_true, y_prob, threshold=thr_loso)
        m_orig["fn_rate_pct"] = float(m_orig["fn"] / max(m_orig["n_pos"], 1) * 100.0)

        thr_bacc = find_optimal_threshold(y_true, y_prob, metric="balanced_accuracy")
        m_bacc = compute_binary_metrics(y_true, y_prob, threshold=thr_bacc)
        m_bacc["fn_rate_pct"] = float(m_bacc["fn"] / max(m_bacc["n_pos"], 1) * 100.0)

        thr_f1 = find_optimal_threshold(y_true, y_prob, metric="f1_target")
        m_f1 = compute_binary_metrics(y_true, y_prob, threshold=thr_f1)
        m_f1["fn_rate_pct"] = float(m_f1["fn"] / max(m_f1["n_pos"], 1) * 100.0)

        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]
        fn_recovered = int(m_orig["fn"] - m_bacc["fn"])

        rows.append(
            {
                "subject_id": sub_id,
                "thr_loso": thr_loso,
                "thr_oracle_bacc": float(thr_bacc),
                "thr_oracle_f1": float(thr_f1),
                "orig_loso": m_orig,
                "oracle_bacc": m_bacc,
                "oracle_f1": m_f1,
                "fn_recovered_bacc": fn_recovered,
                "fn_recovered_pct_of_orig_fn": float(fn_recovered / max(m_orig["fn"], 1) * 100.0),
                "target_y_prob_quantiles": {
                    "q10": float(np.percentile(pos_probs, 10)),
                    "q25": float(np.percentile(pos_probs, 25)),
                    "median": float(np.percentile(pos_probs, 50)),
                    "q75": float(np.percentile(pos_probs, 75)),
                },
                "freq_y_prob_quantiles": {
                    "q90": float(np.percentile(neg_probs, 90)),
                    "q95": float(np.percentile(neg_probs, 95)),
                    "q99": float(np.percentile(neg_probs, 99)),
                },
            }
        )

    return {
        "diagnostic_only": True,
        "note": "Post-hoc threshold diagnostic on test subjects; NEVER replaces official LOSO metrics.",
        "per_subject": rows,
    }


def run_step6_5_addendum(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    report_path: Union[str, Path] = Path("reports/STEP6_XAI_RQ3_REPORT.md"),
    *,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Chạy đủ 3 kiểm chứng Bước 6.5 và cập nhật `xai_rq3_summary.json` + `STEP6_XAI_RQ3_REPORT.md`."""
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    summary_path = metrics_dir / "xai_rq3_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Chưa có {summary_path}; hãy chạy pipeline Bước 6 trước.")

    early_rt_xai = analyze_early_window_rt_and_xai(
        data_dir=data_dir,
        output_dir=out_root,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        device=device,
    )
    rel_comp = analyze_relative_completeness_error(
        data_dir=data_dir,
        output_dir=out_root,
        device=device,
    )
    b5_diag = analyze_bottom5_threshold_miscalibration(output_dir=out_root)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["addendum_step6_5"] = {
        "early_window_rt_and_xai": early_rt_xai,
        "relative_completeness": rel_comp,
        "bottom5_threshold_diagnostic": b5_diag,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_sanitize_json(summary), f, indent=2, allow_nan=False)
    print(f"[Step 6.5] Đã cập nhật {summary_path}.")

    generate_step6_report(summary, report_path=report_path)
    return summary


def run_full_xai_rq3_pipeline(
    data_dir: Union[str, Path] = Path("data/processed"),
    output_dir: Union[str, Path] = Path("results"),
    report_path: Union[str, Path] = Path("reports/STEP6_XAI_RQ3_REPORT.md"),
    *,
    n_steps: int = 64,
    internal_batch_size: int = 32,
    seed: int = 42,
    resume: bool = True,
    force_recompute: bool = False,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, Any]:
    """Điều phối chạy trọn bộ Pipeline XAI 4 lớp (Cache IG -> Cutoff -> Freq Occlusion -> JSON & Report)."""
    out_root = Path(output_dir)
    metrics_dir = out_root / "metrics" / "dl2"
    cache_path = metrics_dir / "xai_loso_cache.npz"
    cutoff_path = metrics_dir / "early_window_cutoff.json"

    if force_recompute or not cache_path.exists():
        compute_and_save_xai_cache(
            data_dir=data_dir,
            output_dir=out_root,
            cache_path=cache_path,
            n_steps=n_steps,
            internal_batch_size=internal_batch_size,
            seed=seed,
            device=device,
        )
    else:
        print(f"[XAI-RQ3] Dùng lại cache sẵn có: {cache_path}")

    if force_recompute or not cutoff_path.exists():
        run_early_window_cutoff(
            data_dir=data_dir,
            output_dir=out_root,
            cutoff_ms=EARLY_CUTOFF_MS,
            resume=resume and not force_recompute,
            device=device,
        )
    else:
        print(f"[XAI-RQ3] Dùng lại kết quả Early-Window Cutoff sẵn có: {cutoff_path}")

    freq_results = run_frequency_occlusion(
        data_dir=data_dir,
        output_dir=out_root,
        device=device,
    )
    analyze_all_layers(
        data_dir=data_dir,
        output_dir=out_root,
        cache_path=cache_path,
        early_cutoff_path=cutoff_path,
        freq_occlusion_results=freq_results,
        device=device,
    )
    return run_step6_5_addendum(
        data_dir=data_dir,
        output_dir=out_root,
        report_path=report_path,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        device=device,
    )
