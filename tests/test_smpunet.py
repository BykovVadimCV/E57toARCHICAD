"""Чекпойнт smp.Unet(ResNet): своя реализация (simscan/smpunet.py) даёт те же логиты, что smp;
предсказание без регрессионных голов проходит весь infer_input; чужой чекпойнт - понятная ошибка."""
import numpy as np
import pytest
import torch

from simscan.infer import describe_checkpoint, infer_input, load_model


def _smp_ckpt(tmp_path, encoder: str, classes: int = 8):
    smp = pytest.importorskip("segmentation_models_pytorch")
    torch.manual_seed(0)
    ref = smp.Unet(encoder, encoder_weights=None, in_channels=8, classes=classes)
    for m in ref.modules():                               # BN не единичные - проверка честнее
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2)
            m.running_var.uniform_(0.5, 1.5)
    ref.eval()
    path = tmp_path / f"{encoder}.pt"
    torch.save({"model": ref.state_dict(), "epoch": 3}, path)
    return ref, path


@pytest.mark.parametrize("encoder", ["resnet18", "resnet34", "resnet50"])
def test_same_logits_as_smp(tmp_path, encoder):
    ref, path = _smp_ckpt(tmp_path, encoder)
    m, _ = load_model(path, device="cpu", log=lambda *a: None)
    x = torch.rand(2, 8, 96, 128)
    with torch.no_grad():
        a, b = ref(x), m(x)["sem"]
    assert torch.allclose(a, b, atol=1e-4), float((a - b).abs().max())
    info = describe_checkpoint(path)
    assert info["format"] == "smp_unet" and encoder in info["text"]


def test_prediction_without_regression_heads(tmp_path):
    _, path = _smp_ckpt(tmp_path, "resnet18")
    m, dev = load_model(path, device="cpu", log=lambda *a: None)
    x = np.random.default_rng(0).random((8, 70, 150)).astype(np.float32)   # не кратно 32
    d = infer_input(m, dev, x, tmp_path / "L1", tta="flip", log=lambda *a: None)
    prob = np.load(d / "prob.npz")["prob"]
    assert prob.shape == (8, 70, 150) and np.allclose(prob.astype(np.float32).sum(0), 1, atol=1e-2)
    assert not (d / "orient.npy").exists() and not (d / "heights.npy").exists()


def test_unknown_checkpoint_is_explained(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"model": {"backbone.conv.weight": torch.zeros(4, 3, 3, 3), "head.weight": torch.zeros(2, 4)}}, path)
    with pytest.raises(ValueError, match="неизвестная архитектура.*backbone"):
        load_model(path, device="cpu", log=lambda *a: None)
