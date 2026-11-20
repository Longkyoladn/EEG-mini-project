"""Bộ kiểm thử tự động Pytest + Captum cho các mô hình DL nhánh 2."""

from __future__ import annotations

import itertools
from typing import List

import pytest
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from src.models import EEGNetV4, P300ATCNet


def _available_devices() -> List[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.mark.parametrize("device", _available_devices())
@pytest.mark.parametrize("model_cls", [EEGNetV4, P300ATCNet])
def test_shapes_and_forward_backward(model_cls: type[nn.Module], device: str) -> None:
    """Kiểm tra forward shape `(4, 1)`, dtype float32 và backward trên CPU/CUDA."""
    torch.manual_seed(42)
    model = model_cls().to(device)
    model.train()

    x = torch.randn(4, 127, 251, dtype=torch.float32, device=device, requires_grad=True)
    logits = model(x)

    assert logits.shape == (4, 1)
    assert logits.dtype == torch.float32
    assert logits.device.type == device
    assert torch.isfinite(logits).all()

    loss = logits.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("model_cls", [EEGNetV4, P300ATCNet])
@pytest.mark.parametrize(
    "bad_shape",
    [
        (127, 251),
        (4, 64, 251),
        (4, 127, 200),
        (2, 1, 127, 251),
    ],
)
def test_invalid_input_raises_value_error(
    model_cls: type[nn.Module], bad_shape: tuple[int, ...]
) -> None:
    """Đảm bảo đầu vào sai số chiều hoặc sai (127, 251) đều raise ValueError."""
    model = model_cls()
    x_bad = torch.randn(*bad_shape, dtype=torch.float32)
    with pytest.raises(ValueError):
        model(x_bad)


@pytest.mark.parametrize(
    "readout,n_windows,shared_branch,temp_kernel_1",
    list(
        itertools.product(
            ["avg_pool", "attn_pool", "last_step"],
            [2, 3],
            [False, True],
            [96, 125],
        )
    ),
)
def test_p300_atcnet_variants(
    readout: str,
    n_windows: int,
    shared_branch: bool,
    temp_kernel_1: int,
) -> None:
    """Kiểm tra toàn bộ tổ hợp biến thể cấu hình của P300ATCNet."""
    torch.manual_seed(7)
    model = P300ATCNet(
        temp_kernel_1=temp_kernel_1,
        n_windows=n_windows,
        shared_branch=shared_branch,
        readout=readout,  # type: ignore[arg-type]
    )
    assert model.cv_steps == 15
    assert model.window_len == 15 - n_windows + 1

    x = torch.randn(4, 127, 251, dtype=torch.float32)
    out = model(x)
    assert out.shape == (4, 1)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_parameter_budget() -> None:
    """Kiểm tra ngân sách tham số và in bảng tổng hợp cho các cấu hình."""
    configs = [
        ("EEGNetV4 (default)", EEGNetV4()),
        ("P300ATCNet (default: n_win=2, shared=False, avg_pool)", P300ATCNet()),
        (
            "P300ATCNet (n_win=2, shared=False, attn_pool)",
            P300ATCNet(n_windows=2, shared_branch=False, readout="attn_pool"),
        ),
        (
            "P300ATCNet (n_win=3, shared=False, avg_pool)",
            P300ATCNet(n_windows=3, shared_branch=False, readout="avg_pool"),
        ),
        (
            "P300ATCNet (n_win=3, shared=False, tcn_filters=24)",
            P300ATCNet(n_windows=3, shared_branch=False, tcn_filters=24),
        ),
        (
            "P300ATCNet (n_win=2, shared=True, avg_pool)",
            P300ATCNet(n_windows=2, shared_branch=True, readout="avg_pool"),
        ),
        (
            "P300ATCNet (n_win=3, shared=True, avg_pool)",
            P300ATCNet(n_windows=3, shared_branch=True, readout="avg_pool"),
        ),
    ]

    print("\n=== Bảng Tổng Hợp Tham Số Mô Hình DL2 ===")
    print(f"{'Cấu hình':<56} | {'Trainable Params':>16}")
    print("-" * 75)
    counts = {}
    for name, model in configs:
        n_params = model.count_parameters()
        counts[name] = n_params
        print(f"{name:<56} | {n_params:>16,}")

    eegnet_params = counts["EEGNetV4 (default)"]
    atc_default_params = counts["P300ATCNet (default: n_win=2, shared=False, avg_pool)"]

    assert eegnet_params < 10_000
    assert 30_000 <= atc_default_params <= 65_000
    assert 35_000 <= atc_default_params <= 60_000


@pytest.mark.parametrize(
    "model_factory,spatial_limit,linear_limit",
    [
        (lambda: EEGNetV4(spatial_max_norm=1.0, linear_max_norm=0.25), 1.0, 0.25),
        (lambda: P300ATCNet(spatial_max_norm=2.0, linear_max_norm=0.25), 2.0, 0.25),
        (
            lambda: P300ATCNet(
                readout="attn_pool", spatial_max_norm=2.0, linear_max_norm=0.25
            ),
            2.0,
            0.25,
        ),
    ],
)
def test_gradients_and_max_norm(
    model_factory: object,
    spatial_limit: float,
    linear_limit: float,
) -> None:
    """Kiểm tra gradient BCEWithLogitsLoss và hiệu lực thực tế của apply_max_norm()."""
    torch.manual_seed(123)
    model: nn.Module = model_factory()  # type: ignore[operator]
    model.train()

    x = torch.randn(6, 127, 251, dtype=torch.float32, requires_grad=True)
    y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0]], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([6.5]))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    optimizer.zero_grad()
    logits = model(x)
    loss = criterion(logits, y)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum().item() > 0.0

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Thiếu gradient ở tham số {name}"
        assert torch.isfinite(param.grad).all(), f"NaN/Inf ở gradient của {name}"

    assert model.temporal_conv.weight.grad.abs().sum().item() > 0.0  # type: ignore[union-attr]
    assert model.spatial_conv.weight.grad.abs().sum().item() > 0.0  # type: ignore[union-attr]
    if hasattr(model, "classifier"):
        assert model.classifier.weight.grad.abs().sum().item() > 0.0  # type: ignore[union-attr]
    else:
        for clf in model.classifiers:  # type: ignore[union-attr]
            assert clf.weight.grad.abs().sum().item() > 0.0

    optimizer.step()

    # Cố tình phóng đại trọng số vượt ngưỡng để kiểm chứng apply_max_norm() cắt chuẩn xác
    with torch.no_grad():
        model.spatial_conv.weight.mul_(25.0)  # type: ignore[union-attr]
        if hasattr(model, "classifier"):
            model.classifier.weight.mul_(25.0)  # type: ignore[union-attr]
        else:
            for clf in model.classifiers:  # type: ignore[union-attr]
                clf.weight.mul_(25.0)

    model.apply_max_norm()  # type: ignore[operator]

    spatial_norms = model.spatial_conv.weight.norm(p=2, dim=(1, 2, 3))  # type: ignore[union-attr]
    assert torch.all(spatial_norms <= spatial_limit + 1e-6)

    if hasattr(model, "classifier"):
        linear_norms = model.classifier.weight.norm(p=2, dim=1)  # type: ignore[union-attr]
        assert torch.all(linear_norms <= linear_limit + 1e-6)
    else:
        for clf in model.classifiers:  # type: ignore[union-attr]
            linear_norms = clf.weight.norm(p=2, dim=1)
            assert torch.all(linear_norms <= linear_limit + 1e-6)


def test_get_spatial_weights() -> None:
    """Kiểm tra helper get_spatial_weights() trả về bản sao đúng chiều 127 kênh."""
    eegnet = EEGNetV4()
    w_eeg = eegnet.get_spatial_weights()
    assert w_eeg.shape == (16, 1, 127, 1)
    assert not w_eeg.requires_grad

    w_eeg_orig = eegnet.spatial_conv.weight.detach().clone()
    w_eeg.zero_()
    assert torch.equal(eegnet.spatial_conv.weight.detach(), w_eeg_orig)

    atcnet = P300ATCNet()
    w_atc = atcnet.get_spatial_weights()
    assert w_atc.shape == (32, 1, 127, 1)
    assert not w_atc.requires_grad


@pytest.mark.parametrize(
    "label,model_factory",
    [
        ("EEGNetV4", lambda: EEGNetV4()),
        ("P300ATCNet(avg_pool)", lambda: P300ATCNet(readout="avg_pool")),
        ("P300ATCNet(attn_pool)", lambda: P300ATCNet(readout="attn_pool")),
    ],
)
def test_captum_integrated_gradients(label: str, model_factory: object) -> None:
    """Kiểm tra Captum IntegratedGradients end-to-end với target=0, n_steps=50."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(99)

    model: nn.Module = model_factory().to(device)  # type: ignore[operator]
    model.eval()

    x = torch.randn(2, 127, 251, dtype=torch.float32, device=device)
    baselines = torch.zeros_like(x)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(
        x,
        baselines=baselines,
        target=0,
        n_steps=50,
        return_convergence_delta=True,
    )

    assert attributions.shape == (2, 127, 251)
    assert torch.isfinite(attributions).all()
    assert attributions.abs().sum().item() > 0.0
    assert torch.isfinite(delta).all()

    print(
        f"\n[Captum IG | {device.upper()} | {label}] "
        f"attr_abs_mean={attributions.abs().mean().item():.6e}, "
        f"convergence_delta={delta.detach().cpu().tolist()}"
    )
