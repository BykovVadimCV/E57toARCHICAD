"""Вероятности «сети» для синтетической квартиры test_floorplan._apartment (без обучения):
разметка как в netinput (наружная стена - полоса 10 см от внутренней грани), сглаженная."""
import json
import math
from pathlib import Path

import numpy as np

A_DEG, SHIFT = 12.0, np.array([3.0, -2.0])


def _classes(u, v):
    """Координаты квартиры (до поворота) -> класс netinput."""
    c = np.zeros(u.shape, np.uint8)
    inside = (u > 0) & (u < 8) & (v > 0) & (v < 4)
    c[inside] = 1
    ext = ~inside & (u > -0.1) & (u < 8.1) & (v > -0.1) & (v < 4.1)
    c[ext] = 2
    c[ext & (v >= 4) & (u > 1.0) & (u < 2.4)] = 4                       # окно в стене y = 4
    part = (u >= 3.95) & (u <= 4.05) & (v > 0) & (v < 4)
    c[part] = 2
    c[part & (v > 1.5) & (v < 2.4)] = 3                                # дверь
    return c


def write_pred(out: Path, px: float = 0.02, floor_z: float = 10.0, ceiling_z: float = 12.8,
               classes=_classes) -> Path:
    import cv2

    a = math.radians(A_DEG)
    corners = np.array([[-1, -1], [9, -1], [9, 5], [-1, 5]], float)
    Rm = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    w = corners @ Rm.T + SHIFT
    x0, y0 = w.min(0)
    W, H = (np.ceil((w.max(0) - w.min(0)) / px)).astype(int)
    jj, ii = np.meshgrid(np.arange(W), np.arange(H))
    xw, yw = x0 + (jj + 0.5) * px, y0 + (H - 0.5 - ii) * px
    q = np.stack([xw - SHIFT[0], yw - SHIFT[1]], -1) @ Rm                # мир -> квартира
    c = classes(q[..., 0], q[..., 1])
    prob = np.stack([cv2.GaussianBlur((c == k).astype(np.float32), (0, 0), 1.0) for k in range(8)])
    prob = 0.9 * prob / np.maximum(prob.sum(0, keepdims=True), 1e-6) + 0.1 / 8
    d = Path(out) / "apt" / "L1"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "prob.npz", prob=prob.astype(np.float16))
    scene = {"level": "L1", "floor_z": floor_z, "ceiling_z": ceiling_z,
             "pixel_to_world": [[px, 0.0, x0 + 0.5 * px], [0.0, -px, y0 + (H - 0.5) * px]]}
    (d / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    return Path(out)
