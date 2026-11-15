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
     90 +- shear. Применяется и к эталону (на 10 м это сантиметры);
  4. углы между стенами не 90° и дуговые стены - нелинейная деформация плана
     (planwarp.py, разыгрывается в generate.py по готовой планировке); A входит в неё.
"""
from __future__ import annotations

import math
import hashlib
from statistics import NormalDist

import numpy as np

from .config import RealismConfig
from .layout import Layout


def sample_params(cfg: RealismConfig, rng: np.random.Generator) -> dict:
    if not cfg.enabled:
        return {"enabled": False, "warp": np.eye(2).tolist()}
    u = lambda r: float(rng.uniform(r[0], r[1]))  # noqa: E731
    probability = float(cfg.wall_roughness_tail_probability)
    tail_lo, tail_hi = map(float, cfg.wall_roughness_tail_sigma_mm)
    base_lo, base_hi = map(float, cfg.wall_unevenness_mm)
    if not 0.0 <= probability < 0.5:
        raise ValueError("wall_roughness_tail_probability must be in [0, 0.5)")
    if tail_lo <= 0 or tail_hi < tail_lo:
        raise ValueError("wall_roughness_tail_sigma_mm must be a positive ordered range")
    if cfg.wall_roughness_tail_corr_m <= 0 or cfg.wall_roughness_tail_transition_sigma <= 0:
        raise ValueError("roughness tail correlation and transition must be positive")
    if not math.isfinite(cfg.wall_roughness_tail_fine_corr_m) or cfg.wall_roughness_tail_fine_corr_m < 0:
        raise ValueError("wall_roughness_tail_fine_corr_m must be finite and >= 0")
    if probability > 0 and tail_lo < base_hi:
        raise ValueError("roughness tail sigma must be >= the maximum base wall unevenness")
    shear = math.radians(float(rng.normal(0, cfg.shear_deg_sigma)))
    k = 32
    corr = u(cfg.unevenness_corr_m)
    lean_corr = 6.0
    # Keep the historical random stream exactly intact: the tail uses a
    # deterministic transform of existing RFF basis and the already sampled
    # base amplitude. Enabling CAL-07 therefore cannot move walls/stations.
    unevenness_mm = u(cfg.wall_unevenness_mm)
    roughness_fraction = (0.5 if base_hi == base_lo else
                          float(np.clip((unevenness_mm - base_lo) / (base_hi - base_lo), 0, 1)))
    tail_sigma_mm = tail_lo + roughness_fraction * (tail_hi - tail_lo)
    return {
        "enabled": True,
        "tessellation_m": cfg.tessellation_m,
        "unevenness_m": unevenness_mm / 1000,
        "roughness_tail_probability": probability,
        "roughness_tail_sigma_m": tail_sigma_mm / 1000,
        "roughness_tail_corr_m": float(cfg.wall_roughness_tail_corr_m),
        "roughness_tail_fine_corr_m": float(cfg.wall_roughness_tail_fine_corr_m),
        "roughness_tail_transition_sigma": float(cfg.wall_roughness_tail_transition_sigma),
        "corr_m": corr,
        "lean_per_m": u(cfg.lean_mm_per_m) / 1000,
        "shear_deg": math.degrees(shear),
        "warp": [[1.0, math.tan(shear)], [0.0, 1.0]],
        # Existing random fields are unchanged; the z-field also supplies a
        # spatially smooth envelope for CAL-07's horizontal wall-normal tails.
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


def roughness_tail_envelope(p: np.ndarray, params: dict) -> np.ndarray:
    """Smooth clustered envelope in [0,1]; probability is its nominal Gaussian tail."""
    probability = float(params.get("roughness_tail_probability", 0.0))
    if probability <= 0:
        return np.zeros(len(p), dtype=np.float64)
    if not 0 < probability < 0.5:
        raise ValueError("roughness tail probability must be in (0, 0.5)")
    w = np.asarray(params["w"])[2]              # reuse existing RFF: no RNG-stream shift
    phi = np.asarray(params["phi"])[2]
    corr = float(params["corr_m"])
    tail_corr = float(params["roughness_tail_corr_m"])
    if corr <= 0 or tail_corr <= 0:
        raise ValueError("roughness tail correlation lengths must be positive")
    # Scale the existing 3-D RFF frequencies to the desired cluster length.
    # The envelope is spatially smooth, so neighbouring wall patches form zones
    # rather than isolated vertex spikes or discontinuous mesh seams.
    wt = w * (corr / tail_corr)
    raw = math.sqrt(2.0 / w.shape[0]) * np.cos(p @ wt.T + phi).sum(1)
    threshold = NormalDist().inv_cdf(1.0 - probability)
    transition = float(params.get("roughness_tail_transition_sigma", 0.15))
    if transition <= 0:
        raise ValueError("roughness tail transition must be positive")
    u = np.clip((raw - (threshold - transition)) / (2.0 * transition), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def fine_defect_field(p: np.ndarray, params: dict) -> np.ndarray:
    """Independent deterministic RFF texture for the CAL-09 defect interior.

    Derive a private seed from the existing surface basis, not the layout RNG.
    Changing the fine correlation length changes only texture frequencies;
    the realization, large-zone envelope and scanner stations stay paired.
    """
    corr = float(params.get("roughness_tail_fine_corr_m", 0.0))
    if not np.isfinite(corr) or corr <= 0:
        raise ValueError("CAL-09 fine correlation length must be positive")
    key = np.asarray(params["w"], dtype="<f8").tobytes()
    seed = int.from_bytes(hashlib.sha256(b"cal09-fine-wall-v1" + key).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    k = 32
    freq = rng.normal(0.0, 1.0 / corr, (2, k, 3))
    phases = rng.uniform(0.0, 2 * math.pi, (2, k))
    return np.stack([math.sqrt(2.0 / k) * np.cos(p @ freq[a].T + phases[a]).sum(1)
                     for a in range(2)], axis=1)


def displacement(p: np.ndarray, params: dict) -> np.ndarray:
    """Поле неровности; CAL-07 плавно усиливает XY-смещение в редких 3-D зонах."""
    w = np.asarray(params["w"])                  # (3, k, 3)
    phi = np.asarray(params["phi"])              # (3, k)
    k = w.shape[1]
    base_sigma = float(params["unevenness_m"])
    envelope = roughness_tail_envelope(p, params)
    tail_sigma = float(params.get("roughness_tail_sigma_m", base_sigma))
    fine_corr = float(params.get("roughness_tail_fine_corr_m", 0.0))
    if not np.isfinite(fine_corr) or fine_corr < 0:
        raise ValueError("roughness_tail_fine_corr_m must be finite and >= 0")
    local_sigma = base_sigma + envelope * (tail_sigma - base_sigma)
    d = np.empty_like(p)
    for axis in range(3):
        arg = p @ w[axis].T + phi[axis]          # (n, k)
        # Wall-normal displacement is horizontal; retain the calibrated base
        # vertical component so the floor/ceiling datum is not globally inflated.
        sigma = (local_sigma if axis < 2 and fine_corr == 0 else base_sigma)
        d[:, axis] = sigma * math.sqrt(2.0 / k) * np.cos(arg).sum(1)
    if fine_corr > 0 and np.any(envelope):
        # Only add a local high-frequency component: never rescale global noise.
        d[:, :2] += (envelope * (tail_sigma - base_sigma))[:, None] * fine_defect_field(p, params)
    wl = np.asarray(params["w_lean"])            # (2, 8, 2)
    pl = np.asarray(params["phi_lean"])
    for axis in range(2):
        g = math.sqrt(2.0 / wl.shape[1]) * np.cos(p[:, :2] @ wl[axis].T + pl[axis]).sum(1)
        d[:, axis] += p[:, 2] * params["lean_per_m"] * g
    return d


def warp_xy(xy: np.ndarray, params: dict) -> np.ndarray:
    from .planwarp import PlanWarp

    return PlanWarp(params.get("warp"))(xy)


def tessellation(params: dict) -> float | None:
    """Шаг разбиения архитектуры: realism.tessellation_m; при нелинейной деформации плана
    (наклоны, дуги) - всегда, иначе стены и перекрытия не повторят её."""
    from .planwarp import PlanWarp

    t = params.get("tessellation_m") if params.get("enabled") else None
    if t is None and not PlanWarp(params.get("warp")).linear:
        t = 0.3
    return t


def apply_to_mesh(mesh, params: dict):
    """Деформировать вершины сцены на месте и вернуть её же."""
    from .planwarp import PlanWarp

    W = PlanWarp(params.get("warp"))
    if not params.get("enabled") and W.identity:
        return mesh
    v = mesh.vertices.astype(np.float64)
    m = mesh.vert_deform
    if params.get("enabled") and m.any():
        v[m] += displacement(v[m], params)
    if not W.identity:
        v[:, :2] = W(v[:, :2])
    mesh.vertices = v.astype(np.float32)
    return mesh
