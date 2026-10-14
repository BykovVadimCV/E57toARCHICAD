"""Деформация плана: углы между стенами не 90° и дуговые стены.

Планировка, обстановка и стоянки строятся в манхэттенской СК (layout.py, apartment.py,
interior.py), а затем вся сцена - вершины сетки, стоянки, люди и эталон - проходит через
непрерывное обратимое отображение плана

    W(p) = A · Arc(Grid(p))

  A    - сдвиг плана (realism.shear_deg_sigma), матрица 2x2, как и раньше;
  Grid - наклон линий стен: X = x + A(x)(y - yc), Y = y + B(y)(x - xc). A(x) постоянна
         (= a_i) в полосе линии стен x = X_i на всю толщину её стен и линейна между полосами,
         B(y) - так же по линиям y = Y_j. Стена в своей полосе только наклоняется: грани
         остаются параллельными (толщина та же), угол в узле - 90° + a_i - b_j (до
         ±2·max_tilt); между полосами линия стены может преломиться на доли градуса
         (второй порядок: a_i·B'·Δx). Разница наклонов соседних полос ограничена так, что
         расстояние между линиями меняется не больше чем на squeeze (15 %);
  Arc  - дуговая стена: пролёт наружной стены между примыканиями выгибается наружу по
         окружности со стрелой h. Ось уходит на дугу, грани - на концентрические дуги
         (смещение по радиусу: толщина та же); у концов пролёта (на длине ends) радиус
         плавно переходит в нормаль хорды - стык с примыкающей стеной непрерывен. Внутрь
         помещения и наружу смещение гаснет гладко (smoothstep) на falloff.

Тела, которые должны повторить дугу (стены, перекрытия), в сетке разбиваются
(realism.tessellation_m); эталон рисуется по сгущённым контурам (PlanWarp.poly).
Спецификация - словарь (layout.meta["warp"]); старый формат - матрица 2x2.
"""
from __future__ import annotations

import math

import numpy as np

AXIS_TOL = 1e-6
SQUEEZE = 0.15


class PlanWarp:
    def __init__(self, spec=None):
        if spec is None:
            spec = {}
        if isinstance(spec, (list, tuple, np.ndarray)):          # старый формат: матрица
            spec = {"A": np.asarray(spec, float).tolist()}
        self.spec = spec
        self.A = np.asarray(spec.get("A", np.eye(2)), float)
        g = spec.get("grid")
        self.grid = None
        if g:
            self.grid = {"A": _profile(g["xb"], g["a"]), "B": _profile(g["yb"], g["b"]),
                         "xc": float(g["xc"]), "yc": float(g["yc"])}
        self.arcs = [dict(a) for a in spec.get("arcs", [])]

    @property
    def linear(self) -> bool:
        """Только сдвиг A: отрезки остаются отрезками точно, сгущать контуры не нужно."""
        return self.grid is None and not self.arcs

    @property
    def identity(self) -> bool:
        return self.linear and np.allclose(self.A, np.eye(2))

    def to_dict(self) -> dict:
        return self.spec

    def arc_walls(self) -> set:
        return {int(a["wall_id"]) for a in self.arcs}

    # --- прямое отображение ------------------------------------------------------
    def __call__(self, xy) -> np.ndarray:
        p = np.atleast_2d(np.asarray(xy, float))[:, :2].copy()
        if self.grid is not None:
            p = self._grid(p)
        for arc in self.arcs:
            p = p + _arc_disp(p, arc)
        return p @ self.A.T

    def _grid(self, p: np.ndarray) -> np.ndarray:
        g = self.grid
        A = np.interp(p[:, 0], *g["A"])
        B = np.interp(p[:, 1], *g["B"])
        return np.c_[p[:, 0] + A * (p[:, 1] - g["yc"]), p[:, 1] + B * (p[:, 0] - g["xc"])]

    # --- контуры --------------------------------------------------------------------
    def poly(self, pts, step: float = 0.05, closed: bool = True) -> np.ndarray:
        """Контур в СК планировки -> в деформированной; нелинейная деформация - со сгущением
        сторон до step, чтобы прямая сторона могла стать дугой или ломаной."""
        pts = np.atleast_2d(np.asarray(pts, float))[:, :2]
        if self.linear:
            return self(pts)
        return self(densify(pts, step, closed))


def _arc_disp(p: np.ndarray, arc: dict) -> np.ndarray:
    p0, p1 = np.asarray(arc["p0"], float), np.asarray(arc["p1"], float)
    L = float(np.linalg.norm(p1 - p0))
    u = (p1 - p0) / L
    nv = np.asarray(arc["out"], float)
    h, R = float(arc["h"]), float(arc["R"])
    t2, fin, fout = float(arc["t"]) / 2, float(arc["falloff_in"]), float(arc["falloff_out"])
    ends = float(arc.get("ends", 0.3))
    s = (p - p0) @ u
    n = (p - p0) @ nv
    out = np.zeros_like(p)
    m = (s > 0) & (s < L)
    if not m.any():
        return out
    s, n = s[m], n[m]
    x = s - L / 2
    hs = np.maximum(np.sqrt(np.maximum(R * R - x * x, 0.0)) - (R - h), 0.0)
    rad = np.c_[x / R, (R - h + hs) / R]                       # радиус в осях (u, nv)
    w = _smooth(np.minimum(s, L - s) / ends)[:, None]
    nu = (1 - w) * np.array([0.0, 1.0]) + w * rad
    nu /= np.linalg.norm(nu, axis=1, keepdims=True)
    g = np.ones(len(s))
    lo = n < -t2
    g[lo] = _smooth((n[lo] + t2 + fin) / fin)
    hi = n > t2
    g[hi] = _smooth(1 - (n[hi] - t2) / fout)
    # радиальная поправка - только в толще стены (n прижат к граням): дальше от стены
    # остаётся затухающий перенос, иначе у концов пролёта отображение сворачивается
    nc = np.clip(n, -t2, t2)
    du = g * nc * nu[:, 0]                                      # вдоль хорды
    dn = g * (hs + nc * (nu[:, 1] - 1))                         # по нормали хорды
    out[m] = du[:, None] * u[None, :] + dn[:, None] * nv[None, :]
    return out


def _profile(bands, tilts):
    """Кусочно-линейная A(x): tilts[k] на полосе bands[k] = (lo, hi), линейно между полосами."""
    xs, ys = [], []
    for (lo, hi), t in sorted(zip(map(tuple, bands), tilts)):
        xs += [lo, hi]
        ys += [t, t]
    return np.asarray(xs, float), np.asarray(ys, float)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def densify(pts: np.ndarray, step: float, closed: bool = True) -> np.ndarray:
    pts = np.asarray(pts, float)
    seq = np.vstack([pts, pts[:1]]) if closed else pts
    out = []
    for a, b in zip(seq[:-1], seq[1:]):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        t = np.arange(n)[:, None] / n
        out.append(a + (b - a) * t)
    if not closed:
        out.append(seq[-1:])
    return np.vstack(out)


def layout_warp(layout) -> PlanWarp | None:
    """Деформация планировки (layout.meta["warp"]) или None, если её нет."""
    spec = layout.meta.get("warp")
    if spec is None:
        return None
    w = PlanWarp(spec)
    return None if w.identity else w


# ======================================================================
# Розыгрыш
# ======================================================================

def _axis_lines(layouts) -> tuple[list, list]:
    """Линии стен по осям: [(координата, полутолщина)] для x = const и y = const."""
    xs, ys = [], []
    for lay in layouts:
        for w in lay.walls:
            (x0, y0), (x1, y1) = w.p0, w.p1
            if abs(x1 - x0) < AXIS_TOL:
                xs.append((x0, w.thickness / 2))
            elif abs(y1 - y0) < AXIS_TOL:
                ys.append((y0, w.thickness / 2))
    return sorted(xs), sorted(ys)


def _bands(lines: list, pad: float, merge_m: float) -> list[tuple[float, float]]:
    """Полосы линий стен: от грани до грани плюс pad; близкие (зазор < merge_m) - одна полоса."""
    out = []
    for c, ht in lines:
        lo, hi = c - ht - pad, c + ht + pad
        if out and lo - out[-1][1] <= merge_m:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def _tilts(bands: list, half_span: float, p_line: float, max_rad: float, rng) -> list:
    """Наклон на полосу; соседние полосы - разница не больше SQUEEZE * зазор / полупролёт."""
    tilts = []
    for k, (lo, hi) in enumerate(bands):
        t = float(rng.uniform(-max_rad, max_rad)) if rng.random() < p_line else 0.0
        if k:
            lim = SQUEEZE * (lo - bands[k - 1][1]) / max(half_span, 1e-6)
            t = float(np.clip(t, tilts[-1] - lim, tilts[-1] + lim))
        tilts.append(t)
    return tilts


def _arc_candidates(layouts, min_len: float) -> list:
    """Пролёты фасада между соседними примыканиями (как эркер): (стена, s0, s1, наружу, глубина
    помещения). Пролёт сужен на полутолщину примыкающей стены - её тело не деформируется;
    в пролёте только окна и целиком; снаружи в 2,5 м ничего нет (лоджия, балкон)."""
    lay = layouts[0]
    walls = [w for L in layouts for w in L.walls]
    out = []
    for w in lay.walls:
        if w.kind != "exterior" or w.role not in ("", "facade") or not w.outward:
            continue
        p0, u = np.asarray(w.p0, float), w.u
        nv = w.n * w.outward
        cuts = [(0.0, 0.0), (w.length, 0.0)]
        blocked = False
        for v in walls:
            if v is w:
                continue
            for q in (v.p0, v.p1):
                r = np.asarray(q, float) - p0
                s, n = float(r @ u), float(r @ nv)
                if -0.05 < s < w.length + 0.05 and abs(n) < w.thickness + 0.05:
                    cuts.append((s, v.thickness / 2))           # примыкание (или угол)
            a, b = np.asarray(v.p0, float) - p0, np.asarray(v.p1, float) - p0
            if max(a @ nv, b @ nv) > w.thickness and min(a @ nv, b @ nv) < 2.5 and \
                    max(a @ u, b @ u) > -0.5 and min(a @ u, b @ u) < w.length + 0.5:
                blocked = True                                  # снаружи что-то есть
        if blocked:
            continue
        merged: dict = {}
        for sc, tc in cuts:                                     # один узел - наибольшая полутолщина
            k = next((k for k in merged if abs(k - sc) < 0.02), sc)
            merged[k] = max(merged.get(k, 0.0), tc)
        cuts = sorted(merged.items())
        for (sa, ta), (sb, tb) in zip(cuts[:-1], cuts[1:]):
            s0, s1 = sa + ta, sb - tb
            if s1 - s0 < min_len:
                continue
            ops = [o for o in lay.openings_on(w.id) if o.s + o.width / 2 > s0 and o.s - o.width / 2 < s1]
            if any(o.kind != "window" or o.s - o.width / 2 < s0 + 0.1 or o.s + o.width / 2 > s1 - 0.1
                   for o in ops):
                continue
            depth = 3.0                                         # до ближайшей грани стены внутрь
            for v in walls:
                if v is w:
                    continue
                r = np.array([v.point(x) for x in np.linspace(-v.ext0, v.length + v.ext1,
                                                              max(2, int(v.length / 0.05)))]) - p0
                sv, nn = r @ u, r @ nv
                m = (sv > s0) & (sv < s1) & (nn < -w.thickness / 2)
                if m.any():
                    depth = min(depth, float(-nn[m].max()) - v.thickness / 2)
            out.append((w, s0, s1, nv, depth))
    return out


def sample_warp(layouts, A, cfg, rng) -> dict:
    """Спецификация деформации этажа (layouts: [нижний, верхний] у двухуровневых)."""
    spec = {"A": np.asarray(A, float).tolist()}
    if not getattr(cfg, "enabled", True):
        return spec
    lay = layouts[0]
    x0, y0, x1, y1 = lay.bbox()
    max_rad = math.radians(cfg.angle_jitter_deg)
    if max_rad > 0 and rng.random() < cfg.p_angle_jitter:
        xs, ys = _axis_lines(layouts)
        if xs and ys:
            xb, yb = _bands(xs, 0.03, cfg.line_merge_m), _bands(ys, 0.03, cfg.line_merge_m)
            a = _tilts(xb, (y1 - y0) / 2 + 1.0, cfg.p_line_tilt, max_rad, rng)
            b = _tilts(yb, (x1 - x0) / 2 + 1.0, cfg.p_line_tilt, max_rad, rng)
            if any(a) or any(b):
                spec["grid"] = {"xb": [list(x) for x in xb], "a": a, "yb": [list(y) for y in yb], "b": b,
                                "xc": (x0 + x1) / 2, "yc": (y0 + y1) / 2}
    arcs = []
    if cfg.max_arcs > 0 and rng.random() < cfg.p_arc:
        grid = PlanWarp({"grid": spec["grid"]}) if "grid" in spec else None
        cand = _arc_candidates(layouts, cfg.arc_min_len_m)
        rng.shuffle(cand)
        used = []
        for w, s0, s1, nv, depth in cand:
            if len(arcs) >= cfg.max_arcs:
                break
            a, b = w.point(s0), w.point(s1)
            if any(np.allclose(p, q, atol=0.05) for p in (a, b) for q in used):
                continue                                       # соседние дуги не делят узел
            p0, p1 = np.asarray(a, float), np.asarray(b, float)
            if grid is not None:                               # дуга - по наклонённой линии
                p0, p1 = grid(np.array([p0, p1]))
                d = (p1 - p0) / np.linalg.norm(p1 - p0)
                nd = np.array([-d[1], d[0]])
                nv = nd * np.sign(nd @ nv)
            L = float(np.linalg.norm(p1 - p0))
            r_min = max(2.0, 0.75 * L)                         # дуга не круче 84°
            h_geo = r_min - math.sqrt(r_min * r_min - L * L / 4)
            h_lo = max(cfg.arc_sagitta_m[0], 0.06 * L)          # заметная дуга, не почти прямая
            h_hi = min(cfg.arc_sagitta_m[1], h_geo, (depth - w.thickness / 2 - 0.3) / 3)
            if h_hi < h_lo:
                continue
            h = float(rng.uniform(h_lo, h_hi))
            R = (L * L / 4 + h * h) / (2 * h)
            arcs.append({"wall_id": w.id, "s": [s0, s1], "p0": p0.tolist(), "p1": p1.tolist(),
                         "out": np.asarray(nv).tolist(), "h": h, "R": R, "t": w.thickness,
                         "falloff_in": 3 * h, "falloff_out": 3 * h, "ends": max(0.5, 1.5 * w.thickness)})
            used += [a, b]
    if arcs:
        spec["arcs"] = arcs
    return spec
