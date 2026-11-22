"""Hợp đồng PyTorch cho MS-CNN-BiGRU. Chạy bằng python -m pytest -v tests/test_model_dl1.py."""

import pytest
import torch
from torch import nn


@pytest.fixture
def model():
    """Tạo model mới với cùng seed cho mỗi ca test để các lần sửa trọng số độc lập."""
    from src.models.ms_cnn_bigru import MSCNNBiGRU

    torch.manual_seed(42)
    return MSCNNBiGRU()


def test_architecture(model):
    """Đối chiếu 28.103 tham số và cấu hình các lớp CNN, BiGRU, attention với thiết kế DL1."""
    layers = list(model.modules())
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 28_103
    temporal = [m for m in layers if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 1]
    assert sorted((m.out_channels, m.kernel_size[1]) for m in temporal) == [
        (6, 31), (6, 63), (6, 125)
    ]
    spatial = [m for m in layers if isinstance(m, nn.Conv2d) and m.kernel_size == (127, 1)]
    assert [(m.in_channels, m.out_channels, m.groups) for m in spatial] == [(6, 12, 6)] * 3
    # Fusion lọc riêng từng đặc trưng trong 36 đặc trưng rồi trộn chúng thành 32 đầu vào cho BiGRU.
    fusion = [m for m in layers if isinstance(m, nn.Conv1d)]
    assert [(m.in_channels, m.out_channels, m.kernel_size, m.groups) for m in fusion] == [
        (36, 36, (15,), 36), (36, 32, (1,), 1)
    ]
    gru, = [m for m in layers if isinstance(m, nn.GRU)]
    assert (gru.input_size, gru.hidden_size, gru.num_layers, gru.bidirectional) == (32, 24, 2, True)
    assert gru.dropout == pytest.approx(0.30)
    assert any(isinstance(m, nn.LayerNorm) and m.normalized_shape == (48,) for m in layers)
    assert sorted((m.in_features, m.out_features) for m in layers if isinstance(m, nn.Linear)) == [
        (24, 1), (48, 1), (48, 24)
    ]
    assert all(not m.inplace for m in layers if isinstance(m, (nn.ELU, nn.Dropout)))


@pytest.mark.parametrize("batch,samples,steps", [(1, 251, 31), (3, 163, 20)])
def test_logits_and_attention(model, batch, samples, steps):
    """Kiểm tra mỗi trial có một logit và attention có tổng bằng 1 trên trục thời gian."""
    # Hai lớp pooling rút cửa sổ chuẩn từ 251 xuống 31 bước, cửa sổ cắt sớm từ 163 xuống 20.
    model.eval()
    x = torch.randn(batch, 127, samples)
    with torch.no_grad():
        default = model(x)
        logits, alpha = model(x, return_attention=True)
    assert logits.shape == (batch, 1)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()
    # Yêu cầu trả thêm attention phải cho cùng dự đoán với forward mặc định ở chế độ eval.
    torch.testing.assert_close(default, logits)
    assert alpha.shape == (batch, steps)
    assert torch.isfinite(alpha).all() and (alpha >= 0).all()
    torch.testing.assert_close(alpha.sum(dim=1), torch.ones(batch))

    # Ép đầu ra âm để phát hiện sigmoid được gắn nhầm vào đầu phân loại.
    head, = [m for m in model.modules() if isinstance(m, nn.Linear)
             and (m.in_features, m.out_features) == (48, 1)]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-2)
        torch.testing.assert_close(model(x), torch.full((batch, 1), -2.0))


def test_train_step_and_max_norm(model):
    """Chạy một bước AdamW với BCE có trọng số và kiểm tra gradient của từng tham số học được."""
    model.train()
    x = torch.randn(3, 127, 163)
    y = torch.tensor([[0.0], [1.0], [0.0]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))(model(x), y).backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
    optimizer.step()

    spatial = [m for m in model.modules() if isinstance(m, nn.Conv2d) and m.kernel_size == (127, 1)]
    head, = [m for m in model.modules() if isinstance(m, nn.Linear)
             and (m.in_features, m.out_features) == (48, 1)]
    assert len(spatial) == 3
    # Chủ động đẩy trọng số vượt ngưỡng để kiểm tra apply_max_norm() thực sự giới hạn
    # chuẩn L2 của từng bộ lọc không gian ở 2,0 và của đầu phân loại ở 0,5.
    with torch.no_grad():
        for layer in [*spatial, head]:
            layer.weight.fill_(2.0)
    model.apply_max_norm()
    for layer, limit in [*((m, 2.0) for m in spatial), (head, 0.5)]:
        norms = layer.weight.flatten(1).norm(dim=1)
        assert ((norms > 0) & (norms <= limit + 1e-6)).all()


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
))]


@pytest.mark.parametrize("device", DEVICES)
def test_eval_input_gradient(model, device):
    """Kiểm tra ở chế độ eval, logit vẫn có gradient hữu hạn và khác 0 theo tín hiệu EEG."""
    # XAI cần lan truyền ngược qua BiGRU ở chế độ eval, kể cả khi chạy trên CUDA.
    model.to(device).eval()
    x = torch.randn(2, 127, 251, device=device, requires_grad=True)
    grad, = torch.autograd.grad(model(x).sum(), x)
    assert grad.shape == x.shape
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_eval_hidden_gradient_without_input_grad(model, device):
    """XAI theo tầng cần backward qua GRU cả khi tín hiệu EEG đầu vào không yêu cầu gradient."""
    model.to(device).eval()
    features = []

    def capture_fusion(module, inputs, output):
        features.append(output)

    with model.fusion.register_forward_hook(capture_fusion):
        logits = model(torch.randn(1, 127, 163, device=device))
    grad, = torch.autograd.grad(logits.sum(), features[0])
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


@pytest.mark.parametrize("device", DEVICES)
def test_integrated_gradients(model, device):
    """Chạy Captum trên cửa sổ 163 mẫu với baseline 0 và kiểm tra bản đồ đóng góp từng kênh, từng mẫu."""
    captum = pytest.importorskip("captum.attr", reason="Install captum to test Integrated Gradients")
    model.to(device).eval()
    x = torch.randn(1, 127, 163, device=device)
    attribution = captum.IntegratedGradients(model).attribute(
        x, baselines=torch.zeros_like(x), target=0, n_steps=8, internal_batch_size=2
    )
    assert attribution.shape == x.shape
    assert torch.isfinite(attribution).all() and attribution.abs().sum() > 0
