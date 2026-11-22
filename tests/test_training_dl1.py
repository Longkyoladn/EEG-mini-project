"""Smoke test huấn luyện MS-CNN-BiGRU trên EEG giả, chạy độc lập với dữ liệu Drive."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.metrics import find_optimal_threshold
from src.models.ms_cnn_bigru import MSCNNBiGRU
from src.training.dataset_dl1 import ChannelStandardScaler


@pytest.fixture(scope="module")
def trained_fold():
    """Một fold LOSO thu nhỏ: hai subject train, một subject val và một subject test."""
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    subjects = np.repeat(["sub-01", "sub-03", "sub-04", "sub-05"], 4)
    x = rng.normal(0, 1e-5, (16, 127, 251)).astype(np.float32)
    y = np.array([0, 0, 0, 1] * 2 + [0, 1, 0, 1] * 2)
    train = np.isin(subjects, ["sub-01", "sub-03"])
    val = subjects == "sub-04"
    test = subjects == "sub-05"
    x[val] += 2e-5
    x[test] -= 2e-5

    scaler = ChannelStandardScaler()
    train_x = torch.from_numpy(scaler.fit_transform(x[train]))
    train_y = torch.tensor(y[train], dtype=torch.float32).unsqueeze(1)
    model = MSCNNBiGRU()
    initial_weight = model.classifier.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=(1 - train_y).sum() / train_y.sum())
    losses = []
    model.train()
    for xb, yb in zip(train_x.split(4), train_y.split(4)):
        optimizer.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        model.apply_max_norm()
        losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_prob = model(torch.from_numpy(scaler.transform(x[val]))).sigmoid().flatten().numpy()
    threshold = find_optimal_threshold(y[val], val_prob, metric="balanced_accuracy")
    return model, scaler, x, y, train, test, losses, initial_weight, threshold


def test_fold_updates_weights_and_scales_train_only(trained_fold):
    """Hai mini-batch phải cập nhật classifier và scaler phải khớp thống kê riêng tập train."""
    model, scaler, x, _, train, _, losses, initial_weight, _ = trained_fold
    assert len(losses) == 2 and np.isfinite(losses).all()
    assert not torch.equal(initial_weight, model.classifier.weight)
    train_uv = x[train].astype(np.float64) * 1e6
    np.testing.assert_allclose(scaler.mean_, train_uv.mean(axis=(0, 2), keepdims=True))
    np.testing.assert_allclose(scaler.std_, train_uv.std(axis=(0, 2), keepdims=True))


def test_loso_checkpoint_reproduces_test_predictions(trained_fold):
    """Nạp lại checkpoint, scaler và ngưỡng đã chọn trên val để tái tạo dự đoán test."""
    model, scaler, x, _, _, test, _, _, threshold = trained_fold
    assert 0.05 <= threshold <= 0.95
    with TemporaryDirectory(prefix="eeg-dl1-") as folder:
        path = Path(folder) / "loso_sub-05.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_threshold": float(threshold),
        }, path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    restored = MSCNNBiGRU().eval()
    restored.load_state_dict(checkpoint["model_state_dict"])
    loaded_scaler = ChannelStandardScaler.from_state_dict(checkpoint["scaler_state_dict"])
    with torch.inference_mode():
        expected = model(torch.from_numpy(scaler.transform(x[test]))).sigmoid()
        actual = restored(torch.from_numpy(loaded_scaler.transform(x[test]))).sigmoid()
    assert actual.shape == (4, 1) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert checkpoint["best_threshold"] == threshold
    assert torch.equal(actual >= checkpoint["best_threshold"], expected >= threshold)
