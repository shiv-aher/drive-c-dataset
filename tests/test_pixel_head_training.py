from __future__ import annotations

import sys
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.train_perception_health_net import pixel_loss
from src.models.perception_health_net import ModelConfig, PerceptionHealthNet


def test_pixel_head_is_in_optimizer_and_receives_gradients() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PerceptionHealthNet(
        ModelConfig(num_classes=12, pixel_head=True), pretrained=False
    ).to(device)
    model.initialize_pixel_head((64, 128), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

    optimizer_parameters = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    pixel_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("_pixel_head.")
    }
    assert pixel_parameters
    assert all(id(parameter) in optimizer_parameters for parameter in pixel_parameters.values())

    model.train()
    output = model(torch.randn(2, 3, 64, 128, device=device))
    assert output["pred_pix"].shape == (2, 1, 64, 128)
    target = torch.zeros_like(output["pred_pix"])
    has_mask = torch.ones(2, dtype=torch.bool, device=device)
    presence = torch.zeros(2, 12, device=device)
    severity = torch.zeros(2, 12, device=device)
    presence[:, 0] = 1.0
    severity[:, 0] = 0.5
    loss = pixel_loss(
        output["pred_pix"], target, has_mask, presence, severity, [0]
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert pixel_parameters["_pixel_head.out.weight"].grad is not None
    assert torch.count_nonzero(pixel_parameters["_pixel_head.out.weight"].grad) > 0


def test_pixel_head_returns_logits_not_probabilities() -> None:
    source = (REPO / "src/models/perception_health_net.py").read_text()
    pixel_forward = source.split("class PixelHead", 1)[1].split(
        "class PerceptionHealthNet", 1
    )[0]
    assert "return torch.sigmoid(x)" not in pixel_forward
    assert "return x" in pixel_forward
