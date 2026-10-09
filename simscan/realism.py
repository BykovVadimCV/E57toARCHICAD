"""Неидеальная геометрия здания.

Реальные стены - не плоскости: штукатурка гуляет на миллиметры, стены завалены от
вертикали, углы не ровно 90 градусов, толщины отличаются от номинальных. Всё это
делается непрерывными преобразованиями вершин, поэтому оболочка сцены остаётся
замкнутой (лучи не утекают в щели):

  1. разброс толщин - меняется сама планировка (это и есть «правда» для эталона);
  2. неровность и завал - гладкое случайное поле смещений D(p) на вершинах архитектуры
     (стены, колонны, перекрытия), разбитых на сетку tessellation_m. Поле - сумма
     случайных косинусов (Random Fourier Features), то есть гауссово поле с заданными
     СКО и длиной корреляции; в эталон не попадает (это «шум поверхности»);
  3. сдвиг плана - аффинное преобразование A всей сцены в плане: угол между осями
     90 +- shear. Применяется и к эталону (на 10 м это сантиметры).
"""
from __future__ import annotations

import math

import numpy as np

from .config import RealismConfig
from .layout import Layout


def sample_params(cfg: RealismConfig, rng: np.random.Generator) -> dict:
    if not cfg.enabled:
        return {"enabled": False, "warp": np.eye(2).tolist()}
    u = lambda r: float(rng.uniform(r[0], r[1]))  # noqa: E731
    shear = math.radians(float(rng.normal(0, cfg.shear_deg_sigma)))
    k = 32
    corr = u(cfg.unevenness_corr_m)
    lean_corr = 6.0
    return {
        "enabled": True,
        "tessellation_m": cfg.tessellation_m,
        "unevenness_m": u(cfg.wall_unevenness_mm) / 1000,
        "corr_m": corr,
        "lean_per_m": u(cfg.lean_mm_per_m) / 1000,
        "shear_deg": math.degrees(shear),
        "warp": [[1.0, math.tan(shear)], [0.0, 1.0]],
        # частоты и фазы поля: по набору на каждую ось смещения
        "w": rng.normal(0, 1 / corr, (3, k, 3)).tolist(),
        "phi": rng.uniform(0, 2 * math.pi, (3, k)).tolist(),
        "w_lean": rng.normal(0, 1 / lean_corr, (2, 8, 2)).tolist(),
        "phi_lean": rng.uniform(0, 2 * math.pi, (2, 8)).tolist(),
    }


def jitter_thickness(layout: Layout, mm: float, rng: np.random.Generator) -> None:
    """Фактические толщины стен; концы удлиняются на 1 см, чтобы стыки не раскрылись."""
    if mm <= 0:
        return
    for w in layout.walls:
        w.thickness = round(max(0.05, w.thickness + float(rng.normal(0, mm / 1000))), 4)
        w.ext0 += 0.01
        w.ext1 += 0.01


def displacement(p: np.ndarray, params: dict) -> np.ndarray:
    """Поле смещений D(p): неровность (3 компоненты) + завал от вертикали (z * g(x, y))."""
    w = np.asarray(params["w"])                  # (3, k, 3)
    phi = np.asarray(params["phi"])              # (3, k)
    k = w.shape[1]
    d = np.empty_like(p)
    for axis in range(3):
        arg = p @ w[axis].T + phi[axis]          # (n, k)
        d[:, axis] = params["unevenness_m"] * math.sqrt(2.0 / k) * np.cos(arg).sum(1)
    wl = np.asarray(params["w_lean"])            # (2, 8, 2)
    pl = np.asarray(params["phi_lean"])
    for axis in range(2):
        g = math.sqrt(2.0 / wl.shape[1]) * np.cos(p[:, :2] @ wl[axis].T + pl[axis]).sum(1)
        d[:, axis] += p[:, 2] * params["lean_per_m"] * g
    return d


def warp_xy(xy: np.ndarray, params: dict) -> np.ndarray:
    A = np.asarray(params.get("warp", np.eye(2)))
    return np.asarray(xy, float) @ A.T


def apply_to_mesh(mesh, params: dict):
    """Деформировать вершины сцены на месте и вернуть её же."""
    if not params.get("enabled"):
        return mesh
    v = mesh.vertices.astype(np.float64)
    m = mesh.vert_deform
    if m.any():
        v[m] += displacement(v[m], params)
    v[:, :2] = warp_xy(v[:, :2], params)
    mesh.vertices = v.astype(np.float32)
    return mesh
