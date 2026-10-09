"""Сеть плана: U-Net с кодировщиком ResNet-34, вход 8 каналов (netinput.CHANNELS), 2 см/пиксель.

Головы (как разметка датасета, simscan/netinput.py):
    sem      [8]  логиты классов 0..7 (255 в разметке - ignore)
    dist     [1]  усечённое расстояние до видимой грани стены, 0..1 (= 0..30 см), сигмоида
    orient   [2]  (cos 2θ, sin 2θ) направления стены, нормирован на единичную длину
    heights  [2]  низ и верх проёма в долях высоты потолка, сигмоида

Только torch: на Colab ставится без segmentation_models_pytorch/timm, веса ImageNet
для 8-канального входа всё равно не подходят.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

N_CLASSES = 8


class _Basic(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.down is None else self.down(x)), inplace=True)


def _layer(cin: int, cout: int, n: int, stride: int) -> nn.Sequential:
    return nn.Sequential(_Basic(cin, cout, stride), *[_Basic(cout, cout) for _ in range(n - 1)])


class _Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class PlanNet(nn.Module):
    def __init__(self, in_channels: int = 8, n_classes: int = N_CLASSES, width: int = 64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(in_channels, w, 7, 2, 3, bias=False), nn.BatchNorm2d(w),
                                  nn.ReLU(inplace=True))                                   # /2
        self.pool = nn.MaxPool2d(3, 2, 1)                                                     # /4
        self.l1 = _layer(w, w, 3, 1)                                                          # /4
        self.l2 = _layer(w, 2 * w, 4, 2)                                                      # /8
        self.l3 = _layer(2 * w, 4 * w, 6, 2)                                                  # /16
        self.l4 = _layer(4 * w, 8 * w, 3, 2)                                                  # /32
        self.u4 = _Up(8 * w, 4 * w, 4 * w)
        self.u3 = _Up(4 * w, 2 * w, 2 * w)
        self.u2 = _Up(2 * w, w, w)
        self.u1 = _Up(w, w, w // 2)
        self.u0 = _Up(w // 2, in_channels, w // 2)                                            # полный размер
        self.sem = nn.Conv2d(w // 2, n_classes, 1)
        self.reg = nn.Conv2d(w // 2, 5, 1)                                                    # dist, orient×2, heights×2

    def forward(self, x):
        H, W = x.shape[-2:]
        ph, pw = (-H) % 32, (-W) % 32
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        s0 = x
        s1 = self.stem(x)
        s2 = self.l1(self.pool(s1))
        s3 = self.l2(s2)
        s4 = self.l3(s3)
        y = self.l4(s4)
        y = self.u4(y, s4)
        y = self.u3(y, s3)
        y = self.u2(y, s2)
        y = self.u1(y, s1)
        y = self.u0(y, s0)
        y = y[..., :H, :W]
        r = self.reg(y)
        return {"sem": self.sem(y),
                "dist": torch.sigmoid(r[:, :1]),
                "orient": F.normalize(r[:, 1:3], dim=1, eps=1e-6),
                "heights": torch.sigmoid(r[:, 3:5])}


# ----------------------------------------------------------------------
# Потери
# ----------------------------------------------------------------------
WALLISH = (2, 3, 4, 5)


def focal_loss(logits, target, alpha=None, gamma: float = 2.0, ignore_index: int = 255):
    """Focal Loss (Lin et al., 2017) для борьбы с дисбалансом классов.

    FL(pt) = -α(1-pt)^γ log(pt)

    Фокусируется на сложных примерах: если pt высока (лёгкий пример), вес (1-pt)^γ мал.
    Полезно для редких классов (door, window).
    """
    ce = F.cross_entropy(logits, target, weight=alpha, ignore_index=ignore_index, reduction="none")
    pt = torch.exp(-ce)
    focal = (1 - pt) ** gamma * ce
    valid = target != ignore_index
    return focal[valid].mean() if valid.any() else torch.tensor(0.0, device=logits.device)


def _soft_skel(p, iters: int = 10):
    """Мягкий скелет (Shit et al., clDice 2021)."""
    def erode(t):
        return -F.max_pool2d(-t, 3, 1, 1)

    def open_(t):
        return F.max_pool2d(erode(t), 3, 1, 1)

    skel = F.relu(p - open_(p))
    for _ in range(iters):
        p = erode(p)
        d = F.relu(p - open_(p))
        skel = skel + F.relu(d - skel * d)
    return skel


def losses(out: dict, batch: dict, class_weights=None, w: dict | None = None, use_focal: bool = True) -> dict:
    w = {"ce": 1.0, "dice": 1.0, "cldice": 0.5, "dist": 1.0, "orient": 0.5, "heights": 0.5, **(w or {})}
    sem = batch["sem"]
    valid = sem != 255

    # Focal loss вместо обычного CE для борьбы с дисбалансом классов
    if use_focal:
        ce = focal_loss(out["sem"], sem, alpha=class_weights, gamma=2.0, ignore_index=255)
    else:
        ce = F.cross_entropy(out["sem"], sem, weight=class_weights, ignore_index=255)

    prob = out["sem"].softmax(1)
    vm = valid.unsqueeze(1).float()
    onehot = F.one_hot(sem.clamp(max=N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2).float() * vm
    p = prob * vm
    inter = (p * onehot).sum((0, 2, 3))
    den = p.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    dice = 1 - ((2 * inter + 1) / (den + 1))[present].mean()
    # clDice по объединению «стена + проёмы»: связность контура важнее толщины
    pw = prob[:, list(WALLISH)].sum(1, keepdim=True) * vm
    tw = onehot[:, list(WALLISH)].sum(1, keepdim=True)
    sp, st = _soft_skel(pw), _soft_skel(tw)
    tprec = ((sp * tw).sum() + 1) / (sp.sum() + 1)
    tsens = ((st * pw).sum() + 1) / (st.sum() + 1)
    cldice = 1 - 2 * tprec * tsens / (tprec + tsens)
    dist = (F.l1_loss(out["dist"], batch["dist"], reduction="none") * vm).sum() / vm.sum().clamp(min=1)
    mo = batch["mask_orient"].float()
    cos = (out["orient"] * batch["orient"]).sum(1)
    orient = ((1 - cos) * mo).sum() / mo.sum().clamp(min=1)
    mh = batch["mask_heights"].float().unsqueeze(1)
    heights = (F.l1_loss(out["heights"], batch["heights"], reduction="none") * mh).sum() / (2 * mh.sum()).clamp(min=1)
    parts = {"ce": ce, "dice": dice, "cldice": cldice, "dist": dist, "orient": orient, "heights": heights}
    parts["total"] = sum(w[k] * v for k, v in parts.items())
    return parts


def confusion(pred, sem, n: int = N_CLASSES):
    """Матрица ошибок [n, n] (строки - эталон) по пикселям с разметкой."""
    m = sem != 255
    idx = sem[m] * n + pred[m]
    return torch.bincount(idx, minlength=n * n).reshape(n, n)


def iou_from_confusion(cm) -> list[float]:
    cm = cm.double()
    tp = cm.diag()
    den = cm.sum(0) + cm.sum(1) - tp
    return [float(t / d) if d > 0 else float("nan") for t, d in zip(tp, den)]
