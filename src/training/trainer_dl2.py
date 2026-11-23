"""Vòng lặp huấn luyện chuẩn Zero Leakage cho các mô hình DL nhánh 2."""

from __future__ import annotations

import inspect
import random
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import (
    PREDICTION_COLUMNS,
    compute_binary_metrics,
    find_optimal_threshold,
)
from src.models.p300_atcnet import P300ATCNet
from src.training.dataset_dl2 import ChannelStandardScaler, EEGTrialDataset


def set_seed(seed: int) -> None:
    """Cố định seed tái lập cho random, numpy, torch và cuDNN."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _canonical_model_name(model_name: str) -> str:
    key = model_name.strip().lower()
    if key in ("p300_atcnet", "atcnet", "p300atcnet"):
        return "p300_atcnet"
    raise ValueError(
        f"model_name phải là 'p300_atcnet', nhận được {model_name!r}."
    )


def _fast_spatial_conv_forward(self: nn.Conv2d, x: torch.Tensor) -> torch.Tensor:
    """Thực thi Depthwise Spatial Conv (C, 1) bằng cuBLAS batched GEMM (không đổi toán học)."""
    bsz, f1, n_ch, n_t = x.shape
    d_mult = self.weight.shape[0] // f1
    return torch.matmul(self.weight.view(1, f1, d_mult, n_ch), x).reshape(
        bsz, f1 * d_mult, 1, n_t
    )


def build_model(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Khởi tạo mới hoàn toàn mô hình DL2 và chuẩn hóa kwargs để lưu checkpoint."""
    canonical = _canonical_model_name(model_name)
    cls = P300ATCNet

    sig = inspect.signature(cls.__init__)
    defaults: Dict[str, Any] = {
        name: param.default
        for name, param in sig.parameters.items()
        if name != "self" and param.default is not inspect.Parameter.empty
    }
    resolved = dict(defaults)
    if model_kwargs:
        unknown = set(model_kwargs.keys()) - set(defaults.keys())
        if unknown:
            raise ValueError(
                f"Tham số không hợp lệ cho {cls.__name__}: {sorted(unknown)}"
            )
        resolved.update(model_kwargs)

    model = cls(**resolved)
    if hasattr(model, "spatial_conv") and isinstance(model.spatial_conv, nn.Conv2d):
        model.spatial_conv.forward = types.MethodType(
            _fast_spatial_conv_forward, model.spatial_conv
        )
    return model, resolved


def _to_plain_python(obj: Any) -> Any:
    """Chuyển dữ liệu metadata sang kiểu thuần Python để checkpoint an toàn với weights_only=True."""
    if isinstance(obj, dict):
        return {str(k): _to_plain_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain_python(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def load_checkpoint_for_eval(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
) -> Tuple[nn.Module, ChannelStandardScaler, Dict[str, Any]]:
    """Nạp lại mô hình, ChannelStandardScaler và metadata từ file checkpoint `.pt`."""
    target = Path(checkpoint_path)
    if not target.exists():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {target.resolve()}")

    torch_device = torch.device(device)
    try:
        ckpt = torch.load(target, map_location=torch_device, weights_only=True)
    except Exception:
        ckpt = torch.load(target, map_location=torch_device, weights_only=False)

    model, _ = build_model(ckpt["model_name"], ckpt.get("model_kwargs"))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(torch_device)
    model.eval()

    scaler = ChannelStandardScaler.from_state_dict(ckpt["scaler_state_dict"])
    return model, scaler, ckpt


@torch.inference_mode()
def _predict_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Chạy inference trả về xác suất lớp Target (sigmoid) trên toàn bộ DataLoader."""
    model.eval()
    probs_list = []
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=(device.type == "cuda"))
        if amp_enabled and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = model(xb)
        else:
            logits = model(xb)
        probs = torch.sigmoid(logits.float()).squeeze(-1)
        probs_list.append(probs.detach().cpu().numpy())
    if not probs_list:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(probs_list, axis=0).astype(np.float64)


@torch.inference_mode()
def _predict_probabilities_tensor(
    model: nn.Module,
    X_t: torch.Tensor,
    batch_size: int,
    amp_enabled: bool = False,
) -> np.ndarray:
    """Chạy inference theo mini-batch trực tiếp trên tensor đã nằm sẵn trên GPU."""
    model.eval()
    n_samples = int(X_t.shape[0])
    if n_samples == 0:
        return np.empty(0, dtype=np.float64)
    bs = max(1, int(batch_size))
    probs_list = []
    use_amp_cuda = bool(amp_enabled and X_t.device.type == "cuda")
    for start in range(0, n_samples, bs):
        xb = X_t[start : start + bs]
        if use_amp_cuda:
            with torch.amp.autocast("cuda"):
                logits = model(xb)
        else:
            logits = model(xb)
        probs = torch.sigmoid(logits.float()).squeeze(-1)
        probs_list.append(probs)
    return torch.cat(probs_list, dim=0).detach().cpu().numpy().astype(np.float64)


def _validate_disjoint_indices(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> None:
    """Đảm bảo tuyệt đối train/val/test không rỗng và không giao nhau."""
    if train_idx.size == 0 or val_idx.size == 0 or test_idx.size == 0:
        raise ValueError(
            f"Chỉ số split không được rỗng: train={train_idx.size}, "
            f"val={val_idx.size}, test={test_idx.size}."
        )
    tr_set = set(train_idx.tolist())
    va_set = set(val_idx.tolist())
    te_set = set(test_idx.tolist())
    if tr_set & va_set or tr_set & te_set or va_set & te_set:
        raise ValueError(
            "Phát hiện rò rỉ dữ liệu: train_idx, val_idx và test_idx bị chồng chéo!"
        )


def train_single_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    *,
    protocol: str = "within_subject",
    fold_key: str = "fold0",
    fold_meta: Optional[Dict[str, Any]] = None,
    trial_uids: Optional[Sequence[str]] = None,
    subject_ids: Optional[Sequence[str]] = None,
    epochs: int = 80,
    patience: int = 15,
    min_delta: float = 1e-4,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    eta_min: float = 1e-6,
    max_grad_norm: float = 1.0,
    early_stop_metric: str = "val_roc_auc",
    threshold_metric: str = "balanced_accuracy",
    seed: int = 42,
    device: Optional[Union[str, torch.device]] = None,
    use_amp: bool = False,
    checkpoint_dir: Optional[Union[str, Path]] = None,
    save_checkpoint: bool = True,
    num_workers: int = 0,
) -> Dict[str, Any]:
    """Huấn luyện và đánh giá 1 fold chuẩn Zero Leakage.

    Quy trình chuẩn:
        1. Fit ``ChannelStandardScaler`` chỉ trên ``X[train_idx]``, transform ``val`` và ``test``.
        2. Khởi tạo model mới + ``BCEWithLogitsLoss(pos_weight)`` từ tỷ lệ lớp ``y_train``.
        3. Tối ưu bằng ``AdamW`` + ``CosineAnnealingLR``, clip gradient và ``apply_max_norm()``.
        4. Early stopping theo ``val_roc_auc`` (hoặc ``val_pr_auc``), giữ trọng số tốt nhất trong RAM.
        5. Nạp lại trọng số tốt nhất, tìm ``best_threshold`` trên ``val`` và chấm ``test`` đúng 1 lần.
        6. Lưu checkpoint ``.pt`` kèm ``scaler_state_dict`` và ``best_threshold``.
    """
    if epochs <= 0:
        raise ValueError(f"epochs phải >= 1, nhận được {epochs}.")
    if early_stop_metric not in ("val_roc_auc", "val_pr_auc"):
        raise ValueError(
            f"early_stop_metric phải là 'val_roc_auc' hoặc 'val_pr_auc', nhận {early_stop_metric!r}."
        )

    t_start = time.perf_counter()
    set_seed(seed)

    torch_device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None
        else torch.device(device)
    )
    amp_enabled = bool(use_amp and torch_device.type == "cuda")

    tr_idx = np.asarray(train_idx, dtype=np.int64).ravel()
    va_idx = np.asarray(val_idx, dtype=np.int64).ravel()
    te_idx = np.asarray(test_idx, dtype=np.int64).ravel()
    _validate_disjoint_indices(tr_idx, va_idx, te_idx)

    y_train = np.asarray(y[tr_idx], dtype=np.int64).ravel()
    y_val = np.asarray(y[va_idx], dtype=np.int64).ravel()
    y_test = np.asarray(y[te_idx], dtype=np.int64).ravel()

    effective_bs = min(int(batch_size), 32) if protocol == "within_subject" else int(batch_size)
    effective_bs = max(2, effective_bs)
    n_train = int(tr_idx.size)
    drop_last_train = n_train > effective_bs and (n_train % effective_bs == 1)
    eval_bs = max(effective_bs, 64)

    train_val_bytes = int(tr_idx.size + va_idx.size) * int(X.shape[1]) * int(X.shape[2]) * 4
    gpu_resident_bytes_limit = 6 * (1024**3)
    use_gpu_resident = (
        torch_device.type == "cuda" and train_val_bytes < gpu_resident_bytes_limit
    )

    scaler = ChannelStandardScaler()
    pin_mem = torch_device.type == "cuda"
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    test_loader: Optional[DataLoader] = None
    X_train_t: Optional[torch.Tensor] = None
    y_train_t: Optional[torch.Tensor] = None
    X_val_t: Optional[torch.Tensor] = None
    X_test_t: Optional[torch.Tensor] = None

    if not use_gpu_resident:
        # Zero-Leakage Scaling trên CPU khi không dùng nhánh GPU-Resident
        X_train_scaled = scaler.fit_transform(X[tr_idx])
        X_val_scaled = scaler.transform(X[va_idx])
        X_test_scaled = scaler.transform(X[te_idx])

        train_ds = EEGTrialDataset(X_train_scaled, y_train)
        val_ds = EEGTrialDataset(X_val_scaled, y_val)
        test_ds = EEGTrialDataset(X_test_scaled, y_test)
        dl_generator = torch.Generator()
        dl_generator.manual_seed(seed)

        train_loader = DataLoader(
            train_ds,
            batch_size=effective_bs,
            shuffle=True,
            drop_last=drop_last_train,
            num_workers=num_workers,
            pin_memory=pin_mem,
            generator=dl_generator,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=eval_bs,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
        )

    canonical_name = _canonical_model_name(model_name)
    model, resolved_kwargs = build_model(canonical_name, model_kwargs)
    model.to(torch_device)

    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = int(np.sum(y_train == 0))
    pos_weight_val = float(n_neg_train / max(n_pos_train, 1))
    pos_weight_tensor = torch.tensor([pos_weight_val], dtype=torch.float32, device=torch_device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=eta_min
    )
    grad_scaler = torch.amp.GradScaler("cuda") if amp_enabled else None

    best_val_metric = -float("inf")
    best_epoch = 1
    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    def _iter_train_batches(epoch_idx: int):
        if use_gpu_resident:
            assert X_train_t is not None and y_train_t is not None
            g_gpu = torch.Generator(device=torch_device)
            g_gpu.manual_seed(int(seed) + int(epoch_idx))
            perm = torch.randperm(n_train, device=torch_device, generator=g_gpu)
            if drop_last_train:
                perm = perm[: (n_train // effective_bs) * effective_bs]
            for start in range(0, int(perm.numel()), effective_bs):
                batch_idx = perm[start : start + effective_bs]
                yield X_train_t.index_select(0, batch_idx), y_train_t.index_select(0, batch_idx)
        else:
            assert train_loader is not None
            for xb, yb in train_loader:
                yield (
                    xb.to(torch_device, non_blocking=pin_mem),
                    yb.to(torch_device, non_blocking=pin_mem),
                )

    def _eval_split_probs(split_name: str) -> np.ndarray:
        if use_gpu_resident:
            t_arr = X_val_t if split_name == "val" else X_test_t
            assert t_arr is not None
            return _predict_probabilities_tensor(
                model, t_arr, eval_bs, amp_enabled=amp_enabled
            )
        ldr = val_loader if split_name == "val" else test_loader
        assert ldr is not None
        return _predict_probabilities(model, ldr, torch_device, amp_enabled=amp_enabled)

    try:
        if use_gpu_resident:
            X_train_t = torch.from_numpy(
                np.ascontiguousarray(X[tr_idx], dtype=np.float32)
            ).to(torch_device)
            X_val_t = torch.from_numpy(
                np.ascontiguousarray(X[va_idx], dtype=np.float32)
            ).to(torch_device)
            X_test_t = torch.from_numpy(
                np.ascontiguousarray(X[te_idx], dtype=np.float32)
            ).to(torch_device)
            y_train_t = (
                torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.float32))
                .unsqueeze(1)
                .to(torch_device)
            )

            # Zero-Leakage GPU Scaling (float64 theo từng kênh để khớp chuẩn CPU)
            n_ch = int(X_train_t.shape[1])
            mean_g = torch.empty((1, n_ch, 1), dtype=torch.float64, device=torch_device)
            std_g = torch.empty((1, n_ch, 1), dtype=torch.float64, device=torch_device)
            for ch_idx in range(n_ch):
                ch_uv = X_train_t[:, ch_idx, :].to(torch.float64) * scaler.scale_factor
                mean_g[0, ch_idx, 0] = ch_uv.mean()
                std_g[0, ch_idx, 0] = ch_uv.std(correction=0)
            mean_np = mean_g.detach().cpu().numpy()
            std_np = std_g.detach().cpu().numpy()
            if not (np.isfinite(mean_np).all() and np.isfinite(std_np).all()):
                raise ValueError(
                    "Phát hiện giá trị NaN/Inf trong X_train khi fit ChannelStandardScaler."
                )
            scaler.mean_ = mean_np
            scaler.std_ = std_np
            scaler.n_channels_ = n_ch

            mean_f32 = (mean_g / scaler.scale_factor).to(torch.float32)
            denom_f32 = ((std_g + scaler.eps) / scaler.scale_factor).to(torch.float32)
            X_train_t.sub_(mean_f32).div_(denom_f32)
            X_val_t.sub_(mean_f32).div_(denom_f32)
            X_test_t.sub_(mean_f32).div_(denom_f32)
            del mean_g, std_g, mean_f32, denom_f32

        for epoch in range(1, epochs + 1):
            model.train()
            for xb, yb in _iter_train_batches(epoch):
                optimizer.zero_grad(set_to_none=True)

                if amp_enabled and grad_scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(xb)
                        loss = criterion(logits, yb)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                    optimizer.step()

                if hasattr(model, "apply_max_norm"):
                    model.apply_max_norm()

            scheduler.step()

            # Đánh giá trên Validation sau mỗi epoch
            p_val_epoch = _eval_split_probs("val")
            val_metrics_epoch = compute_binary_metrics(y_val, p_val_epoch, threshold=0.5)
            monitored = (
                val_metrics_epoch["roc_auc"]
                if early_stop_metric == "val_roc_auc"
                else val_metrics_epoch["pr_auc"]
            )
            if not np.isfinite(monitored):
                monitored = val_metrics_epoch["balanced_accuracy"]

            if np.isfinite(monitored) and monitored > best_val_metric + min_delta:
                best_val_metric = float(monitored)
                best_epoch = int(epoch)
                best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        # Nạp lại trọng số tốt nhất để tuning threshold trên Val và đánh giá Test 1 lần duy nhất
        model.load_state_dict({k: v.to(torch_device) for k, v in best_state_dict.items()})
        model.eval()

        p_val_best = _eval_split_probs("val")
        best_threshold = find_optimal_threshold(y_val, p_val_best, metric=threshold_metric)
        val_metrics_best = compute_binary_metrics(y_val, p_val_best, threshold=best_threshold)

        p_test = _eval_split_probs("test")
        test_metrics = compute_binary_metrics(y_test, p_test, threshold=best_threshold)
    finally:
        if use_gpu_resident:
            del X_train_t, y_train_t, X_val_t, X_test_t
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed_sec = float(time.perf_counter() - t_start)

    clean_fold_meta = _to_plain_python(dict(fold_meta or {}))
    clean_fold_meta.setdefault("protocol", protocol)
    clean_fold_meta.setdefault("fold_key", fold_key)
    clean_fold_meta["n_train"] = int(tr_idx.size)
    clean_fold_meta["n_val"] = int(va_idx.size)
    clean_fold_meta["n_test"] = int(te_idx.size)
    clean_fold_meta["effective_batch_size"] = int(effective_bs)

    ckpt_path_str: Optional[str] = None
    if save_checkpoint and checkpoint_dir is not None:
        ckpt_root = Path(checkpoint_dir)
        ckpt_root.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_root / f"{fold_key}.pt"
        ckpt_payload = {
            "model_name": canonical_name,
            "model_kwargs": _to_plain_python(resolved_kwargs),
            "model_state_dict": best_state_dict,
            "scaler_state_dict": scaler.state_dict(),
            "best_threshold": float(best_threshold),
            "best_epoch": int(best_epoch),
            "best_val_metric": float(best_val_metric),
            "early_stop_metric": early_stop_metric,
            "threshold_metric": threshold_metric,
            "fold_meta": clean_fold_meta,
            "seed": int(seed),
        }
        torch.save(ckpt_payload, ckpt_path)
        ckpt_path_str = str(ckpt_path)

    test_trial_uids = (
        [str(trial_uids[i]) for i in te_idx]
        if trial_uids is not None
        else [f"{fold_key}_trial_{int(i)}" for i in te_idx]
    )
    default_sub = str(clean_fold_meta.get("subject_id", "unknown"))
    test_subject_ids = (
        [str(subject_ids[i]) for i in te_idx]
        if subject_ids is not None
        else [default_sub] * int(te_idx.size)
    )
    fold_label = clean_fold_meta.get("fold", fold_key)

    predictions_df = pd.DataFrame(
        {
            "trial_uid": test_trial_uids,
            "subject_id": test_subject_ids,
            "protocol": [protocol] * int(te_idx.size),
            "fold": [fold_label] * int(te_idx.size),
            "y_true": y_test.astype(np.int64),
            "y_prob": p_test.astype(np.float64),
            "y_pred": (p_test >= best_threshold).astype(np.int64),
            "threshold": [float(best_threshold)] * int(te_idx.size),
        },
        columns=PREDICTION_COLUMNS,
    )

    fold_metrics_record: Dict[str, Any] = {
        "fold_key": fold_key,
        "protocol": protocol,
        "fold": fold_label,
        "subject_id": default_sub,
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_val_metric),
        "pos_weight": float(pos_weight_val),
        "train_time_sec": elapsed_sec,
        **test_metrics,
    }
    for extra_key in ("train_subjects", "val_subjects", "test_subjects"):
        if extra_key in clean_fold_meta:
            fold_metrics_record[extra_key] = clean_fold_meta[extra_key]

    return {
        "fold_metrics": fold_metrics_record,
        "val_metrics": val_metrics_best,
        "predictions_df": predictions_df,
        "checkpoint_path": ckpt_path_str,
        "best_threshold": float(best_threshold),
        "best_epoch": int(best_epoch),
        "best_val_metric": float(best_val_metric),
        "train_time_sec": elapsed_sec,
    }
