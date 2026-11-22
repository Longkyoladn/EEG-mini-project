"""Huấn luyện và đánh giá MS-CNN-BiGRU trên một fold đã khóa."""

import random
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.evaluation.metrics import PREDICTION_COLUMNS, compute_binary_metrics, find_optimal_threshold
from src.models.ms_cnn_bigru import MSCNNBiGRU
from .dataset_dl1 import (
    VALID_PROTOCOLS, ChannelStandardScaler, EEGDatasetBundle, FoldDataset, fit_fold_scaler,
)


def _split_indices(bundle, fold):
    """Chặn index sai, trial trùng và subject bị rò rỉ trước khi fit scaler."""
    protocol = fold["protocol"]
    if protocol not in VALID_PROTOCOLS:
        raise ValueError(f"Giao thức không hợp lệ: {protocol}")
    parts = []
    for name in ("train_idx", "val_idx", "test_idx"):
        idx = np.asarray(fold[name])
        if (idx.ndim != 1 or not idx.size or idx.dtype.kind not in "iu"
                or idx.min() < 0 or idx.max() >= len(bundle.y)):
            raise ValueError(f"{name} rỗng hoặc chứa index không hợp lệ")
        parts.append(idx)
    indices = np.concatenate(parts)
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Có trial trùng trong fold")
    subjects = [set(bundle.subject_ids[idx]) for idx in parts]
    tr, va, te = subjects
    if protocol == "within_subject":
        if len(tr | va | te) != 1:
            raise ValueError("Within-Subject chỉ được chứa một subject")
    elif tr & va or tr & te or va & te or (protocol == "cross_subject_loso" and len(te) != 1):
        raise ValueError("Fold cross-subject không tách subject đúng giao thức")
    return parts


@torch.inference_mode()
def _predict(model, loader, device):
    model.eval()
    return torch.cat([model(x.to(device)).sigmoid().flatten().cpu() for x, _ in loader]).numpy()


def load_checkpoint_for_eval(path, device="cpu"):
    """Nạp model ở eval mode, scaler và ngưỡng trong checkpoint DL1."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt["model_name"] != "ms_cnn_bigru":
        raise ValueError(f"Checkpoint không phải DL1: {ckpt['model_name']}")
    model = MSCNNBiGRU(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval(), ChannelStandardScaler.from_state_dict(ckpt["scaler_state_dict"]), ckpt


def train_single_fold(
    bundle: EEGDatasetBundle, fold: dict, *, epochs: int = 80, patience: int = 15,
    batch_size: int | None = None, seed: int = 42, device=None,
    checkpoint_dir: str | Path | None = None,
) -> dict:
    """Nhận một fold từ get_protocol_folds; truyền checkpoint_dir để lưu file .pt."""
    tr, va, te = _split_indices(bundle, fold)
    if batch_size is None:
        batch_size = 32 if fold["protocol"] == "within_subject" else 64
    if min(epochs, patience, batch_size) < 1:
        raise ValueError("epochs, patience và batch_size phải dương")
    counts = np.bincount(bundle.y[tr], minlength=2)
    if len(counts) != 2 or min(counts) == 0:
        raise ValueError("Tập train phải có đủ nhãn 0 và 1")
    pos_weight = float(counts[0] / counts[1])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    start = time.perf_counter()
    scaler = fit_fold_scaler(bundle.X, tr)
    train_loader = DataLoader(FoldDataset(bundle, tr, scaler), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(FoldDataset(bundle, va, scaler), batch_size=batch_size)
    test_loader = DataLoader(FoldDataset(bundle, te, scaler), batch_size=batch_size)
    model = MSCNNBiGRU().to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    best_epoch, best_score = 0, -np.inf
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            model.apply_max_norm()
            batch_losses.append(loss.item())
        scheduler.step()
        train_loss = float(np.mean(batch_losses))
        epoch_losses.append(train_loss)
        val_prob = _predict(model, val_loader, device)
        metrics = compute_binary_metrics(bundle.y[va], val_prob, threshold=0.5)
        score = metrics["roc_auc"] if np.isfinite(metrics["roc_auc"]) else metrics["balanced_accuracy"]
        is_best = score > best_score + 1e-4
        if is_best:
            best_score, best_epoch = float(score), epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_prob = val_prob
        marker = " ✓" if is_best else ""
        no_improve = epoch - best_epoch
        print(
            f"  Epoch {epoch:3d}/{epochs} | loss={train_loss:.4f}"
            f" | val_auc={metrics['roc_auc']:.4f}"
            f" | val_ba={metrics['balanced_accuracy']:.4f}"
            f" | best_ep={best_epoch} (+{no_improve}){marker}",
            flush=True,
        )
        if not is_best and no_improve >= patience:
            print(f"  Early stop tại epoch {epoch} (patience={patience}).", flush=True)
            break

    if not best_epoch:
        raise ValueError("Validation không cho metric hữu hạn")
    model.load_state_dict(best_state)
    threshold = float(find_optimal_threshold(bundle.y[va], best_val_prob, metric="balanced_accuracy"))
    test_prob = _predict(model, test_loader, device)
    elapsed = time.perf_counter() - start
    fold_meta = {k: v for k, v in fold.items() if k not in ("train_idx", "val_idx", "test_idx")}
    path = None
    if checkpoint_dir is not None:
        path = Path(checkpoint_dir) / (fold["fold_key"] + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
        try:
            torch.save(dict(model_name="ms_cnn_bigru", model_kwargs={}, model_state_dict=best_state,
                            scaler_state_dict=scaler.state_dict(), best_threshold=threshold,
                            best_epoch=best_epoch, best_val_metric=best_score, seed=seed,
                            n_times=bundle.X.shape[2], trial_uid_sha256=bundle.trial_uid_sha256,
                            fold_meta=fold_meta), tmp)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    predictions = pd.DataFrame(dict(
        trial_uid=bundle.metadata["trial_uid"].to_numpy()[te], subject_id=bundle.subject_ids[te],
        protocol=fold["protocol"], fold=fold["fold"], y_true=bundle.y[te], y_prob=test_prob,
        y_pred=(test_prob >= threshold).astype(np.int64), threshold=threshold,
    ), columns=PREDICTION_COLUMNS)
    return dict(
        fold_metrics={**fold_meta, **compute_binary_metrics(bundle.y[te], test_prob, threshold=threshold),
                      "pos_weight": pos_weight, "train_time_sec": elapsed},
        val_metrics=compute_binary_metrics(bundle.y[va], best_val_prob, threshold=threshold),
        predictions_df=predictions, best_threshold=threshold, best_epoch=best_epoch,
        best_val_metric=best_score, train_time_sec=elapsed, checkpoint_path=str(path) if path else None,
    )
