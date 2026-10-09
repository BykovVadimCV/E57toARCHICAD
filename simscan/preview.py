"""Превью сцены: плотностной срез, прочитанный из записанного E57, поверх эталона.

Читаем именно файл (pye57.read_scan с позами), а не внутренние массивы -
так превью заодно проверяет запись поз и координат.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .e57io import read_e57_world
from .groundtruth import RasterFrame
from .transform import WorldTransform


def density_image(points_layout: np.ndarray, frame: RasterFrame, z_band) -> np.ndarray:
    z = points_layout[:, 2]
    sel = (z >= z_band[0]) & (z < z_band[1])
    i, j = frame.xy_to_ij(points_layout[sel, 0], points_layout[sel, 1])
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    ok = (i >= 0) & (i < frame.height) & (j >= 0) & (j < frame.width)
    img = np.zeros((frame.height, frame.width), np.float32)
    np.add.at(img, (i[ok], j[ok]), 1.0)
    return img


def _maxpool(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    return a[:h, :w].reshape(h // k, k, w // k, k).max(axis=(1, 3))


def render_preview(e57_path: str | Path, masks: dict, frame: RasterFrame, world: WorldTransform,
                   out_png: str | Path, z_band=(1.0, 1.5), pool: int = 2) -> dict:
    """Фон - эталон (стены розовые, двери зелёные, окна голубые), сверху точки полосы z_band
    (темнее = больше), жёлтые круги - станции. Растр уменьшен max-pooling, чтобы грани
    не терялись при уменьшении."""
    import cv2

    scans = read_e57_world(e57_path)
    pts = np.vstack([world.to_layout(s["xyz"]) for s in scans])
    dens = _maxpool(density_image(pts, frame, z_band), pool)

    img = np.full((*dens.shape, 3), 255, np.uint8)
    for name, color in (("walls", (190, 190, 255)), ("doors", (170, 230, 170)),
                        ("windows", (250, 215, 160))):
        img[_maxpool(masks[name].astype(np.uint8), pool) > 0] = color
    # не нормировать по максимуму: плотность падает с дальностью и гасит дальние стены
    g = np.clip(np.log1p(dens) / np.log1p(20.0), 0, 1)
    shade = (170 - 170 * g).astype(np.uint8)
    occ = dens > 0
    img[occ] = shade[occ][:, None]

    px = frame.pixel_m * pool
    radius = max(3, int(0.12 / px))
    for s in scans:
        p = world.to_layout(s["position"][None])[0]
        i, j = frame.xy_to_ij(p[0], p[1])
        cv2.circle(img, (int(round(j / pool)), int(round(i / pool))), radius, (0, 200, 255), -1)
    cv2.imwrite(str(out_png), img)
    return {"points_read": int(len(pts)), "scans_read": len(scans)}
