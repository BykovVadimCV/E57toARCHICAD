"""Треугольные сетки примитивов: разбитый параллелепипед, цилиндр, штора, внешняя модель.

Все функции возвращают (vertices (n, 3) float64, triangles (m, 3) int64) в СК планировки.
"""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .layout import Box


def _rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def box_mesh(box: Box, cell: float):
    """Параллелепипед, каждая грань которого разбита на сетку с шагом не больше cell.

    Вершины на рёбрах дублируются между гранями; поле деформации зависит только от
    положения точки, поэтому после деформации оболочка остаётся замкнутой."""
    half = np.asarray(box.size, float) / 2
    verts, tris = [], []
    base = 0
    for axis in range(3):
        a1, a2 = [k for k in range(3) if k != axis]
        n1 = max(1, int(math.ceil(2 * half[a1] / cell)))
        n2 = max(1, int(math.ceil(2 * half[a2] / cell)))
        u = np.linspace(-half[a1], half[a1], n1 + 1)
        v = np.linspace(-half[a2], half[a2], n2 + 1)
        uu, vv = np.meshgrid(u, v, indexing="ij")
        for sign in (-1.0, 1.0):
            p = np.zeros((uu.size, 3))
            p[:, axis] = sign * half[axis]
            p[:, a1], p[:, a2] = uu.ravel(), vv.ravel()
            i = np.arange(n1)[:, None] * (n2 + 1) + np.arange(n2)[None]
            q00, q10, q01, q11 = i, i + (n2 + 1), i + 1, i + n2 + 2
            t = np.concatenate([np.stack([q00, q10, q11], -1).reshape(-1, 3),
                                np.stack([q00, q11, q01], -1).reshape(-1, 3)])
            verts.append(p)
            tris.append(t + base)
            base += len(p)
    v = np.vstack(verts) @ _rot(box.yaw).T + np.asarray(box.center)
    return v, np.vstack(tris)


def cylinder_mesh(center, radius: float, height: float, segments: int = 12):
    """Вертикальный цилиндр с крышками; center - центр объёма."""
    cx, cy, cz = center
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ring = np.c_[cx + radius * np.cos(a), cy + radius * np.sin(a)]
    z0, z1 = cz - height / 2, cz + height / 2
    v = np.vstack([np.c_[ring, np.full(segments, z0)], np.c_[ring, np.full(segments, z1)],
                   [[cx, cy, z0], [cx, cy, z1]]])
    i = np.arange(segments)
    j = (i + 1) % segments
    side = np.concatenate([np.stack([i, j, j + segments], -1), np.stack([i, j + segments, i + segments], -1)])
    bottom = np.stack([np.full(segments, 2 * segments), j, i], -1)
    top = np.stack([np.full(segments, 2 * segments + 1), i + segments, j + segments], -1)
    return v, np.concatenate([side, bottom, top])


def curtain_mesh(p0, p1, z0: float, z1: float, amp: float, period: float, rows: int = 3):
    """Штора со складками: полоса от p0 до p1 (план), складки - синус поперёк полосы."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-6:
        return np.zeros((0, 3)), np.zeros((0, 3), int)
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    cols = max(2, int(math.ceil(L / (period / 4))) + 1)
    s = np.linspace(0, L, cols)
    off = amp * np.sin(2 * math.pi * s / period)
    xy = p0[None] + s[:, None] * u[None] + off[:, None] * n[None]
    z = np.linspace(z0, z1, rows + 1)
    v = np.array([[x, y, zz] for zz in z for x, y in xy])
    tris = []
    for r in range(rows):
        for c in range(cols - 1):
            a, b = r * cols + c, r * cols + c + 1
            d, e = a + cols, b + cols
            tris += [[a, b, e], [a, e, d]]
    return v, np.array(tris)


@lru_cache(maxsize=256)
def _load(path: str):
    import open3d as o3d

    m = o3d.io.read_triangle_mesh(path)
    v = np.asarray(m.vertices, float)
    t = np.asarray(m.triangles, np.int64)
    if len(v) == 0 or len(t) == 0:
        raise ValueError(f"пустая модель: {path}")
    return v, t


def asset_mesh(path: str, center, size, yaw: float, z_up: bool = True, native: bool = False,
               scale: float = 1.0):
    """Внешняя модель, повёрнутая на yaw.

    native=False - вписывается в габарит size (вдоль, поперёк, высота) с центром center;
    native=True  - свой (метрический) размер x scale, центр по XY в center, низ на center[2]
    (так ставятся сканы: Google Scanned Objects - в метрах).
    Модели из библиотек часто с осью Y вверх (glTF) - тогда z_up=False."""
    v, t = _load(str(Path(path)))
    v = v.copy()
    if not z_up:
        v = v[:, [0, 2, 1]] * np.array([1, -1, 1])
    lo, hi = v.min(0), v.max(0)
    if native:
        v = (v - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])) * scale
        v = v @ _rot(yaw).T + np.asarray(center, float)
        return v, t
    ext = np.maximum(hi - lo, 1e-9)
    v = (v - (lo + hi) / 2) / ext * np.asarray(size, float)
    v = v @ _rot(yaw).T + np.asarray(center, float)
    return v, t


def prim_mesh(prim: dict):
    kind = prim["type"]
    if kind == "cylinder":
        return cylinder_mesh(prim["center"], prim["radius"], prim["height"])
    if kind == "curtain":
        return curtain_mesh(prim["p0"], prim["p1"], prim["z0"], prim["z1"], prim["amp"],
                            prim["period"])
    if kind == "mesh":
        return asset_mesh(prim["path"], prim["center"], prim.get("size", (1, 1, 1)),
                          prim.get("yaw", 0.0), prim.get("z_up", True), prim.get("native", False),
                          prim.get("scale", 1.0))
    if kind == "trimesh":
        return np.asarray(prim["v"], float), np.asarray(prim["f"], np.int64)
    raise ValueError(f"неизвестный примитив: {kind}")
