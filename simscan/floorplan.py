"""План из сведённого облака без нейросети: стены, окна, двери, комнаты.

    python -m simscan floorplan D:/scans/kvartira.e57 --out D:/plan
    -> plan.png (на уровень - plan_L1.png, plan_L2.png ...), walls.png (развёртки стен),
       elements.json

Для пустых квартир (новостройки) в формате заказчика: одно сведённое облако E57.
Структура результата - как в ReFloorBRUSNIKA (core/models.py): стены (WallSegment:
прямоугольник, толщина, наружная или нет), проёмы (Opening: door / window / gap, ширина,
стена-хозяйка) и помещения (Room: многоугольник, площадь). Координаты - метры в
выпрямленной СК плана (оси вдоль стен); как вернуться в СК файла - в elements.json.

Как это работает (на каждый уровень, найденный detect_levels):
  1. два прохода по файлу: по выборке - уровни, поворот плана и кадр; затем все точки
     раскладываются по растру: пол, потолок, полосы высоты и битовая маска слоёв по 10 см
     (бит k - в пикселе есть точки на высоте k*10 см над полом);
  2. грани стен - наложение срезов: пиксель - грань, если занят хотя бы в 4 слоях. Окна и
     двери так закрываются сами (под окном и над ним стена, над дверью перемычка); пол,
     потолок, светильники, низкий потолок за дверью - 1-2 слоя, отпадают;
  3. помещения - заливка свободного (виден пол или потолок), грани - барьеры;
  4. контур помещения - видимые грани его стен - разбивается на прямые и дуги: каждый раз
     самый длинный кусок, который прямая или дуга покрывает с СКО не больше 12 мм; углы -
     пересечения соседних кусков. Форма и размер не меняются, ничего не достраивается;
  5. стены: встречные грани соседних помещений (параллельны, 2-50 см) - стена с толщиной;
     грань без пары - наружная (толщина не видна, на плане - линия);
  6. проёмы - по развёртке стены (длина x слои): участок без точек на 1,0-1,8 м.
     Подоконник есть - окно, нет и есть перемычка - дверь, нет ничего - проём.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

BANDS = {"low": (0.15, 0.7), "mid": (1.0, 1.8)}
LAYER_M = 0.1


@dataclass
class WallSegment:
    id: str
    points: list                      # ось стены, метры
    rooms: list                       # помещения по обе стороны (одно - стена наружная или за ней не снято)
    thickness: float | None          # между гранями соседних помещений; None - видна одна грань
    is_outer: bool
    length: float


@dataclass
class Opening:
    id: str
    opening_type: str                 # door | window | gap
    wall_id: str
    a: list                           # начало и конец по оси стены, метры
    b: list
    points: list                      # четырёхугольник проёма (на всю толщину стены)
    width: float
    sill: float | None = None         # низ проёма над полом (окно)
    head: float | None = None         # верх проёма над полом


@dataclass
class Room:
    id: str
    name: str
    points: list                      # многоугольник по внутренним граням стен, метры
    area: float
    perimeter: float
    label_xy: tuple = (0.0, 0.0)
    faces: list = field(default_factory=list)   # грани: прямые (a, b) и дуги (a, b, center, radius)


@dataclass
class LevelPlan:
    level: int
    floor_z: float
    ceiling_z: float
    walls: list = field(default_factory=list)
    openings: list = field(default_factory=list)
    rooms: list = field(default_factory=list)
    slab: list = field(default_factory=list)    # толща стен между двумя видимыми гранями (для рисунка)


# ----------------------------------------------------------------------
# Чтение: два прохода
# ----------------------------------------------------------------------

def _quat_R(q) -> np.ndarray:
    from .inspect_e57 import _quat_R as qr

    return qr(q)


def _scan_points(reader, index: int, chunk: int):
    s = reader.scans[index]
    R = _quat_R(s.rotation) if s.rotation is not None else np.eye(3)
    t = np.asarray(s.translation, float) if s.translation is not None else np.zeros(3)
    names = s.field_names
    cart = all(n in names for n in ("cartesianX", "cartesianY", "cartesianZ"))
    fields = ["cartesianX", "cartesianY", "cartesianZ"] if cart else \
        ["sphericalRange", "sphericalAzimuth", "sphericalElevation"]
    inv_name = "cartesianInvalidState" if cart else "sphericalInvalidState"
    if inv_name in names:
        fields.append(inv_name)
    for d in reader.iter_points(index, fields, chunk):
        if cart:
            xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
        else:
            r, az, el = d["sphericalRange"], d["sphericalAzimuth"], d["sphericalElevation"]
            xyz = np.c_[r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)]
        ok = np.isfinite(xyz).all(1)
        if inv_name in d:
            ok &= d[inv_name] == 0
        yield xyz[ok] @ R.T + t


def dominant_angle(xy: np.ndarray, bin_m: float = 0.02, max_points: int = 400_000) -> float:
    """Главное направление стен (градусы, 0..90): угол, при котором проекции точек среза на
    оси x и y дают самые острые пики (стены вдоль осей собираются в узкие столбцы).
    Грубо через 1°, затем уточнение через 0,05°."""
    if len(xy) > max_points:
        xy = xy[np.random.default_rng(0).choice(len(xy), max_points, replace=False)]
    xy = xy - xy.mean(0)

    def score(deg):
        t = math.radians(deg)
        c, s_ = math.cos(t), math.sin(t)
        tot = 0.0
        for v in (xy[:, 0] * c + xy[:, 1] * s_, -xy[:, 0] * s_ + xy[:, 1] * c):
            h = np.bincount(np.floor((v - v.min()) / bin_m).astype(np.int64)).astype(float)
            tot += float((h * h).sum())
        return tot

    coarse = np.arange(0.0, 90.0, 1.0)
    best = coarse[int(np.argmax([score(d) for d in coarse]))]
    fine = np.arange(best - 1.0, best + 1.0, 0.05)
    return float(fine[int(np.argmax([score(d) for d in fine]))] % 90.0)


@dataclass
class Rasters:
    origin: np.ndarray                # (x0, y0) в выпрямленной СК
    px: float
    shape: tuple
    counts: dict                      # полоса -> int32 (H, W)
    bits: np.ndarray                  # int32 (H, W): бит k - есть точки в слое высоты k
    layer_m: float = 0.1              # толщина слоя
    layers: int = 30


def build_rasters(path, px: float = 0.01, chunk: int = 2_000_000, sample_points: int = 3_000_000,
                  log=print) -> tuple[list, float, list]:
    from .e57read import E57Reader
    from .rasterize import detect_levels, floor_ceiling

    with E57Reader(path) as r:
        total = sum(s.points for s in r.scans)
        p_keep = min(1.0, sample_points / max(total, 1))
        rng = np.random.default_rng(0)
        sample = []
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                sample.append(p[rng.random(len(p)) < p_keep].astype(np.float32))
        sample = np.concatenate(sample).astype(float)
        levels = detect_levels(sample)
        if not levels:
            fl, ce = floor_ceiling(sample[:, 2])
            levels = [{"floor_z": fl, "ceiling_z": ce, "height_m": ce - fl}]
        hz = sample[:, 2] - levels[0]["floor_z"]
        angle = dominant_angle(sample[(hz > 1.0) & (hz < 1.8), :2])
        c, s_ = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
        rot = np.array([[c, -s_], [s_, c]])
        q = sample[:, :2] @ rot.T
        lo = np.percentile(q, 0.2, axis=0) - 0.6
        hi = np.percentile(q, 99.8, axis=0) + 0.6
        px = max(px, float((hi - lo).max()) / 4000)
        shape = (int(math.ceil((hi[1] - lo[1]) / px)), int(math.ceil((hi[0] - lo[0]) / px)))
        log(f"уровней {len(levels)}, поворот {angle:.2f}°, растр {shape[1]}x{shape[0]} px по {px * 1000:.0f} мм")
        rasters = []
        for lv in levels:
            rasters.append(Rasters(lo, px, shape,
                                   {k: np.zeros(shape, np.int32) for k in ("floor", "ceiling", "low", "mid", "high", "wb", "wc")},
                                   np.zeros(shape, np.int32), LAYER_M,
                                   int(min(31, math.ceil((lv["ceiling_z"] - lv["floor_z"]) / LAYER_M)))))
        H, W = shape
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                q = p[:, :2] @ rot.T
                jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                inside = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
                for lv, ras in zip(levels, rasters):
                    h = p[:, 2] - lv["floor_z"]
                    top = lv["ceiling_z"] - lv["floor_z"]
                    m_lv = inside & (h > -0.1) & (h < top + 0.1)
                    flat = ii * W + jj
                    # «потолок» - любая точка выше 2 м: подвесные потолки и короба ниже основного
                    bands = {"floor": np.abs(h) < 0.04, "ceiling": (h > 2.0) & (h < top + 0.06),
                             "low": (h > BANDS["low"][0]) & (h < BANDS["low"][1]),
                             "mid": (h > BANDS["mid"][0]) & (h < BANDS["mid"][1]),
                             "high": (h > 1.85) & (h < top - 0.06),
                             "wb": (h > 2.15) & (h < 2.5), "wc": (h > 2.5) & (h < top - 0.1)}
                    for k, m in bands.items():
                        m = m & m_lv
                        ras.counts[k] += np.bincount(flat[m], minlength=H * W).reshape(H, W).astype(np.int32)
                    # слои по 10 см: битовая маска на пиксель - развёртка стены по высоте
                    lay = np.floor(h / ras.layer_m).astype(np.int64)
                    bits = ras.bits.reshape(-1)
                    for kk in range(ras.layers):
                        m = m_lv & (lay == kk)
                        if m.any():
                            bits[np.unique(flat[m])] |= np.int32(1 << kk)
    return levels, angle, rasters


# ----------------------------------------------------------------------
# Разбор уровня
# ----------------------------------------------------------------------

def _disk(r_px: float):
    import cv2

    r = max(1, int(round(r_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    import cv2

    inv = (mask == 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    H, W = mask.shape
    out = mask.copy()
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):
            out[lab == k] = 1
    return out


def _rect_kernel(w: int, h: int):
    return np.ones((max(1, h), max(1, w)), np.uint8)


def _components(mask, conn=8):
    import cv2

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), conn)
    return n, lab, stats


def _sandwiched(faces: np.ndarray, L: int) -> np.ndarray:
    """Пиксели между двумя гранями не дальше L по x или по y (толща стены)."""
    import cv2

    out = np.zeros_like(faces)
    for kw, kh, a1, a2 in ((L, 1, (0, 0), (L - 1, 0)), (1, L, (0, 0), (0, L - 1))):
        k_ = _rect_kernel(kw, kh)
        out |= cv2.dilate(faces, k_, anchor=a1) & cv2.dilate(faces, k_, anchor=a2)
    return out


def _drop_small(mask: np.ndarray, min_extent_px: int, min_area_px: int = 0) -> np.ndarray:
    n, lab, stats = _components(mask)
    bad = np.flatnonzero((np.maximum(stats[:, 2], stats[:, 3]) < min_extent_px) | (stats[:, 4] < min_area_px))
    out = mask.copy()
    out[np.isin(lab, bad[bad > 0])] = 0
    return out


# ----------------------------------------------------------------------
# Контур помещения -> прямые и дуги
# ----------------------------------------------------------------------

class _Moments:
    """Префиксные суммы по замкнутому контуру (удвоенному): за O(1) - СКО отклонения от
    лучшей прямой и от лучшей окружности для любого куска контура."""

    def __init__(self, p: np.ndarray):
        q = np.vstack([p, p])
        x, y = q[:, 0], q[:, 1]
        z = x * x + y * y
        cols = np.c_[np.ones_like(x), x, y, x * x, x * y, y * y, z, x * z, y * z, z * z]
        self.S = np.vstack([np.zeros(cols.shape[1]), np.cumsum(cols, 0)])

    def sums(self, i, j):
        return self.S[j] - self.S[i]                 # точки i .. j-1

    @staticmethod
    def line_rms(s):
        n = s[..., 0]
        mx, my = s[..., 1] / n, s[..., 2] / n
        cxx = s[..., 3] / n - mx * mx
        cxy = s[..., 4] / n - mx * my
        cyy = s[..., 5] / n - my * my
        lam = (cxx + cyy) / 2 - np.sqrt(((cxx - cyy) / 2) ** 2 + cxy ** 2)
        return np.sqrt(np.maximum(lam, 0))

    @staticmethod
    def circle(s):
        """Окружность по Касе: z = A x + B y + C; центр (A/2, B/2). -> (cx, cy, r, rms)."""
        n, x, y, xx, xy, yy, z, xz, yz, zz = (s[..., k] for k in range(10))
        M = np.stack([np.stack([xx, xy, x], -1), np.stack([xy, yy, y], -1), np.stack([x, y, n], -1)], -2)
        rhs = np.stack([xz, yz, z], -1)
        M = M + np.eye(3) * 1e-9
        sol = np.linalg.solve(M, rhs[..., None])[..., 0]
        A, B, C = sol[..., 0], sol[..., 1], sol[..., 2]
        cx, cy = A / 2, B / 2
        r2 = C + cx * cx + cy * cy
        r = np.sqrt(np.maximum(r2, 1e-12))
        res = zz - 2 * (A * xz + B * yz + C * z) + (A * A * xx + B * B * yy + C * C * n
                                                    + 2 * A * B * xy + 2 * A * C * x + 2 * B * C * y)
        rms = np.sqrt(np.maximum(res, 0) / n) / (2 * r)
        return cx, cy, r, rms


def _longest_runs(mom: _Moments, N: int, ok_fn, min_pts: int) -> np.ndarray:
    """Для каждого начала i - наибольшая длина куска, который ok_fn считает годным
    (бинарный поиск разом для всех начал)."""
    lo = np.full(N, min_pts - 1)
    hi = np.full(N, N)
    i = np.arange(N)
    good = ok_fn(mom.sums(i, i + min_pts))
    lo[~good] = 0
    hi[~good] = 0
    while True:
        act = hi > lo
        if not act.any():
            break
        mid = (lo + hi + 1) // 2
        m = np.where(act, mid, min_pts)
        g = ok_fn(mom.sums(i, i + m)) & act
        lo = np.where(g, mid, lo)
        hi = np.where(act & ~g, mid - 1, hi)
    return lo


def _fit_primitives(cnt: np.ndarray, tol: float, min_len: float, r_min: float, r_max: float) -> list:
    """Жадно: каждый раз - самая длинная прямая или дуга по ещё не занятому куску контура
    (СКО отклонения <= tol). Возвращает куски по порядку обхода."""
    N = len(cnt)
    c0 = cnt.mean(0)
    p = cnt - c0
    mom = _Moments(p)
    min_pts = max(3, int(min_len))
    line_ok = lambda s: _Moments.line_rms(s) <= tol            # noqa: E731

    def arc_ok(s):
        cx, cy, r, rms = _Moments.circle(s)
        return (rms <= tol) & (r >= r_min) & (r <= r_max)

    L_line = _longest_runs(mom, N, line_ok, min_pts)
    L_arc = _longest_runs(mom, N, arc_ok, min_pts)
    taken = np.zeros(N, bool)
    prims = []
    while True:
        if taken.all():
            break
        # до ближайшего занятого вперёд по обходу
        idx = np.flatnonzero(taken)
        if len(idx):
            nxt = np.searchsorted(idx, np.arange(N))
            dist = np.where(nxt < len(idx), idx[np.minimum(nxt, len(idx) - 1)], idx[0] + N) - np.arange(N)
        else:
            dist = np.full(N, N)
        ll = np.where(taken, 0, np.minimum(L_line, dist))
        la = np.where(taken, 0, np.minimum(L_arc, dist))
        # дуга берётся, только если заметно длиннее прямой с того же начала
        score = np.maximum(ll, np.where(la > 1.15 * ll, la, 0))
        i = int(score.argmax())
        n = int(score[i])
        if n < min_pts:
            break
        kind = "arc" if la[i] > 1.15 * ll[i] and la[i] == n else "line"
        sl = np.arange(i, i + n) % N
        taken[sl] = True
        prims.append({"kind": kind, "start": i, "n": n, "pts": cnt[sl]})
    prims.sort(key=lambda q: q["start"])
    for q in prims:
        pts = q["pts"]
        m = pts.mean(0)
        if q["kind"] == "arc":
            s = _Moments(pts - m).sums(0, len(pts))
            cx, cy, r, _ = _Moments.circle(s)
            q["c"], q["r"] = np.array([cx, cy]) + m, float(r)
            sag = r - math.sqrt(max(r * r - (np.linalg.norm(pts[-1] - pts[0]) / 2) ** 2, 0))
            if sag < 2 * tol:
                q["kind"] = "line"
        if q["kind"] == "line":
            _, _, vt = np.linalg.svd(pts - m, full_matrices=False)
            q["p"], q["d"] = m, vt[0]
    return prims


def _proj(q, xy):
    if q["kind"] == "line":
        return q["p"] + np.dot(xy - q["p"], q["d"]) * q["d"]
    v = xy - q["c"]
    return q["c"] + v / max(np.linalg.norm(v), 1e-9) * q["r"]


def _intersect(a, b, near):
    """Пересечение двух кусков, ближайшее к near (или None)."""
    if a["kind"] == "line" and b["kind"] == "line":
        M = np.c_[a["d"], -b["d"]]
        if abs(np.linalg.det(M)) < 0.05:                # почти параллельны
            return None
        t = np.linalg.solve(M, b["p"] - a["p"])
        return a["p"] + t[0] * a["d"]
    if a["kind"] == "arc" and b["kind"] == "arc":
        d = np.linalg.norm(b["c"] - a["c"])
        if d < 1e-9 or d > a["r"] + b["r"] or d < abs(a["r"] - b["r"]):
            return None
        l_ = (a["r"] ** 2 - b["r"] ** 2 + d * d) / (2 * d)
        h = math.sqrt(max(a["r"] ** 2 - l_ * l_, 0))
        u = (b["c"] - a["c"]) / d
        base = a["c"] + u * l_
        cand = [base + np.array([-u[1], u[0]]) * h, base - np.array([-u[1], u[0]]) * h]
    else:
        ln, ar = (a, b) if a["kind"] == "line" else (b, a)
        f = ln["p"] - ar["c"]
        B = 2 * np.dot(f, ln["d"])
        C = np.dot(f, f) - ar["r"] ** 2
        disc = B * B - 4 * C
        if disc < 0:
            return None
        cand = [ln["p"] + ln["d"] * t for t in ((-B + math.sqrt(disc)) / 2, (-B - math.sqrt(disc)) / 2)]
    return min(cand, key=lambda c: np.linalg.norm(c - near))


def _room_outline(mask: np.ndarray, P, tol_px: float) -> tuple[np.ndarray, list]:
    """Контур помещения: прямые и дуги, углы - пересечения соседних кусков.
    -> (многоугольник (дуги - хордами по 10 см), куски с концами a, b)."""
    import cv2

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
    prims = _fit_primitives(cnt, tol_px, P(0.1), P(1.5), P(40.0))
    if len(prims) < 3:
        poly = cv2.approxPolyDP(cnt.astype(np.int32).reshape(-1, 1, 2), P(0.03), True)[:, 0, :].astype(float)
        return poly, []
    k = len(prims)
    for q in prims:
        q["a"], q["b"] = _proj(q, q["pts"][0]), _proj(q, q["pts"][-1])
    for j in range(k):
        A, B = prims[j], prims[(j + 1) % k]
        near = (A["pts"][-1] + B["pts"][0]) / 2
        x = _intersect(A, B, near)
        if x is not None and np.linalg.norm(x - near) <= P(0.35):
            A["b"], B["a"] = x, x
    poly = []
    for q in prims:
        poly.append(q["a"])
        if q["kind"] == "arc":
            a0 = math.atan2(*(q["a"] - q["c"])[::-1])
            a1 = math.atan2(*(q["b"] - q["c"])[::-1])
            mid = math.atan2(*(q["pts"][len(q["pts"]) // 2] - q["c"])[::-1])
            # направление обхода дуги - через середину куска
            da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            dm = (mid - a0 + math.pi) % (2 * math.pi) - math.pi
            if np.sign(dm) != np.sign(da) or abs(dm) > abs(da):
                da = da - np.sign(da) * 2 * math.pi
            steps = max(2, int(abs(da) * q["r"] / P(0.1)))
            for t in np.linspace(0, 1, steps + 1)[1:-1]:
                ang = a0 + da * t
                poly.append(q["c"] + q["r"] * np.array([math.cos(ang), math.sin(ang)]))
        if np.linalg.norm(q["b"] - prims[(prims.index(q) + 1) % k]["a"]) > 0.5:
            poly.append(q["b"])                     # разрыв между кусками - отрезком
    return np.array(poly), prims


def _offset_polygon(poly: np.ndarray, d: float) -> np.ndarray:
    """Многоугольник, раздвинутый наружу на d (стыки - продолжением сторон)."""
    n = len(poly)
    area = 0.5 * np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sgn = 1.0 if area > 0 else -1.0
    lines = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        L = np.linalg.norm(e)
        if L < 1e-9:
            continue
        e = e / L
        nrm = sgn * np.array([e[1], -e[0]])
        lines.append((a + nrm * d, e))
    out = []
    for i in range(len(lines)):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]
        M = np.c_[d1, -d2]
        if abs(np.linalg.det(M)) < 1e-6:
            out.append(p2)
            continue
        t = np.linalg.solve(M, p2 - p1)
        x = p1 + t[0] * d1
        if np.linalg.norm(x - p2) > 4 * abs(d):        # острый угол - срез
            out.extend([p1 + d1 * (np.dot(p2 - p1, d1)), p2])
        else:
            out.append(x)
    return np.array(out)


def _face_segments(prims, P):
    """Куски контура как отрезки (дуги - хордами ~30 см) для поиска парных граней."""
    segs = []
    for qi, q in enumerate(prims):
        if q["kind"] == "line":
            segs.append((qi, q["a"], q["b"]))
            continue
        a0 = math.atan2(*(q["a"] - q["c"])[::-1])
        a1 = math.atan2(*(q["b"] - q["c"])[::-1])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        steps = max(1, int(abs(da) * q["r"] / P(0.3)))
        pts = [q["c"] + q["r"] * np.array([math.cos(a0 + da * t), math.sin(a0 + da * t)])
               for t in np.linspace(0, 1, steps + 1)]
        pts[0], pts[-1] = q["a"], q["b"]
        segs.extend((qi, pts[j], pts[j + 1]) for j in range(steps))
    return segs


def analyse_level(ras: Rasters, lv: dict, k: int, max_wall_m: float = 0.45, outer_m: float = 0.2,
                  max_opening_m: float = 2.0, min_opening_m: float = 0.5, min_room_m2: float = 1.0,
                  tol_m: float = 0.012, min_layers: int = 4) -> tuple[LevelPlan, dict]:
    import cv2

    px = ras.px
    H, W = ras.shape
    P = lambda m: max(1, int(round(m / px)))                      # noqa: E731 - метры -> пиксели
    occ = {b: (ras.counts[b] > 0).astype(np.uint8) for b in ras.counts}
    k3 = _rect_kernel(3, 3)

    # --- свободно: виден пол или потолок --------------------------------------------------
    free = cv2.morphologyEx(occ["floor"] | occ["ceiling"], cv2.MORPH_CLOSE, k3)
    building = _fill_holes(cv2.morphologyEx(free, cv2.MORPH_CLOSE, _rect_kernel(P(0.3), P(0.3))))
    building = cv2.morphologyEx(building, cv2.MORPH_OPEN, _rect_kernel(P(0.3), P(0.3)))

    # --- грани стен: наложение срезов по 10 см. Пиксель - грань стены, если в нём (с соседями)
    # есть точки хотя бы в min_layers слоях. Окно и дверь так закрываются сами: под окном и над
    # ним стена, над дверью перемычка. Пол, потолок, светильник, низкий потолок - 1-2 слоя ----
    nl = ras.layers
    top = lv["ceiling_z"] - lv["floor_z"]
    keep = sum(1 << kk for kk in range(1, nl) if (kk + 1) * ras.layer_m < top - 0.12)
    b = ras.bits & np.int32(keep)                                        # без пола и потолка
    bp = np.pad(b, 1)
    nb = np.zeros_like(b)
    for dy in range(3):
        for dx in range(3):
            nb |= bp[dy:dy + H, dx:dx + W]
    layers_cnt = np.unpackbits(np.ascontiguousarray(nb).view(np.uint8).reshape(H, W, 4), axis=-1).sum(-1)
    own = np.unpackbits(np.ascontiguousarray(b).view(np.uint8).reshape(H, W, 4), axis=-1).sum(-1)
    faces = ((layers_cnt >= min_layers) & (own >= 2)).astype(np.uint8)
    faces &= cv2.dilate(building, _rect_kernel(2 * P(0.15) + 1, 2 * P(0.15) + 1))
    faces = _drop_small(faces, P(0.2))
    barrier = cv2.morphologyEx(faces, cv2.MORPH_CLOSE, k3)
    struct = barrier | _sandwiched(barrier, P(max_wall_m))       # толща - только между двумя гранями

    # --- помещения: заливка свободного, грани стен (с перемычками и подоконниками) - барьер ---
    space = (free & (1 - barrier) & building).astype(np.uint8)
    space = cv2.morphologyEx(space, cv2.MORPH_OPEN, k3)
    n, lab, stats = _components(space, 4)
    tol_px = max(tol_m / px, 0.6)
    rooms = []
    for c in range(1, n):
        if stats[c][4] * px * px < min_room_m2:
            continue
        m = (lab == c).astype(np.uint8)
        # вернуть пиксель, снятый расширением барьера: грань - вплотную к стене
        m = cv2.dilate(m, k3) & (1 - barrier) & (free | m)
        m = _fill_holes(m)                                       # светильник, колонна внутри - не дырка
        poly, prims = _room_outline(m, P, tol_px)
        area_poly = abs(cv2.contourArea(poly.astype(np.float32))) * px * px
        sgn = 1.0 if cv2.contourArea(poly.astype(np.float32), oriented=True) > 0 else -1.0
        dm = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        li, lj = np.unravel_index(dm.argmax(), dm.shape)
        rooms.append({"mask": m, "poly": poly, "prims": prims, "area": area_poly, "sgn": sgn,
                      "label": (lj, li)})
    rooms.sort(key=lambda r: -r["area"])

    # --- стены: грани помещений. Пара встречных граней соседних помещений (параллельны,
    # между ними 2-50 см) - внутренняя стена с толщиной; без пары - наружная ---------------
    segs = []
    for ri, R in enumerate(rooms):
        for qi, a, b in _face_segments(R["prims"], P):
            e = b - a
            L = np.linalg.norm(e)
            if L < P(0.05):
                continue
            e = e / L
            # наружу от помещения: проверка точкой
            nrm = np.array([e[1], -e[0]])
            mid = (a + b) / 2
            if cv2.pointPolygonTest(R["poly"].astype(np.float32), tuple(map(float, mid + nrm * 2)), False) > 0:
                nrm = -nrm
            segs.append({"room": ri, "prim": qi, "a": a, "b": b, "e": e, "n": nrm, "L": L, "pairs": []})
    for i, s in enumerate(segs):
        for j in range(i + 1, len(segs)):
            t = segs[j]
            if t["room"] == s["room"] or np.dot(s["n"], t["n"]) > -0.995:
                continue
            dist = np.dot(t["a"] - s["a"], s["n"])
            if not (P(0.02) <= dist <= P(max_wall_m + 0.05)):
                continue
            ta, tb = sorted((np.dot(t["a"] - s["a"], s["e"]), np.dot(t["b"] - s["a"], s["e"])))
            o0, o1 = max(0.0, ta), min(s["L"], tb)
            if o1 - o0 < P(0.15):
                continue
            s["pairs"].append((o0, o1, dist, j))
            u0 = np.dot(s["a"] + s["e"] * o0 - t["a"], t["e"])
            u1 = np.dot(s["a"] + s["e"] * o1 - t["a"], t["e"])
            t["pairs"].append((min(u0, u1), max(u0, u1), dist, i))

    walls = []                                       # a, b (ось), t, outer, rooms, seg, (s0, s1) на грани
    outside = (1 - building).astype(np.uint8)
    for i, s in enumerate(segs):
        for (o0, o1, dist, j) in s["pairs"]:
            if j < i:
                continue
            a = s["a"] + s["e"] * o0 + s["n"] * dist / 2
            b = s["a"] + s["e"] * o1 + s["n"] * dist / 2
            walls.append({"a": a, "b": b, "t": dist, "outer": False, "rooms": (s["room"], segs[j]["room"]),
                          "seg": i, "s": (o0, o1), "depth": dist})
        # непарные куски грани
        cov = sorted((o0, o1) for o0, o1, _, _ in s["pairs"])
        free_iv, cur = [], 0.0
        for o0, o1 in cov:
            if o0 > cur:
                free_iv.append((cur, o0))
            cur = max(cur, o1)
        if cur < s["L"]:
            free_iv.append((cur, s["L"]))
        for o0, o1 in free_iv:
            if o1 - o0 < P(0.1):
                continue
            mid = s["a"] + s["e"] * (o0 + o1) / 2 + s["n"] * P(0.3)
            z = np.round(mid).astype(int)
            out = (not (0 <= z[0] < W and 0 <= z[1] < H)) or bool(outside[z[1], z[0]])
            T = P(outer_m)
            walls.append({"a": s["a"] + s["e"] * o0 + s["n"] * T / 2, "b": s["a"] + s["e"] * o1 + s["n"] * T / 2,
                          "t": float(T), "outer": out, "one_sided": True, "rooms": (s["room"],), "seg": i,
                          "s": (o0, o1), "depth": P(max_wall_m)})

    # --- проёмы по развёртке стены: позиция вдоль грани x слои по 10 см; в сечении - от грани
    # вглубь стены (на толщину или 45 см). Окно на контур не влияет - ставится поверх --------
    openings = []
    elevations = []
    nl, lm = ras.layers, ras.layer_m
    mid_l = slice(int(1.0 / lm), int(1.8 / lm))
    for wi, w in enumerate(walls):
        s = segs[w["seg"]]
        s0, s1 = w["s"]
        if s1 - s0 < P(0.5):
            continue
        ss = np.arange(s0, s1, 1.0)
        offs = np.arange(1.0, max(2.0, w["depth"] + P(0.03)), 1.0)
        acc = np.zeros(len(ss), np.int64)
        for i, sv in enumerate(ss):
            z = np.round(s["a"][None] + s["e"][None] * sv + s["n"][None] * offs[:, None]).astype(int)
            ok = (z[:, 0] >= 0) & (z[:, 0] < W) & (z[:, 1] >= 0) & (z[:, 1] < H)
            acc[i] = np.bitwise_or.reduce(ras.bits[z[ok, 1], z[ok, 0]]) if ok.any() else 0
        E = ((acc[:, None] >> np.arange(nl)[None]) & 1).astype(bool)
        ops = []
        if s1 - s0 >= P(min_opening_m):
            kw = max(1, P(0.04))
            mid_occ = np.convolve(E[:, mid_l].mean(1), np.ones(kw) / kw, "same") > 0.3
            runs = np.flatnonzero(np.diff(np.r_[0, (~mid_occ).astype(int), 0]))
            for a, b in zip(runs[::2], runs[1::2]):
                if not (P(min_opening_m) <= b - a <= P(max_opening_m + 0.5)):
                    continue
                v = E[a:b].mean(0) > 0.3
                below = np.flatnonzero(v[1:mid_l.start]) + 1
                above = np.flatnonzero(v[mid_l.stop:nl - 1]) + mid_l.stop
                sill = (below.max() + 1) * lm if len(below) else 0.0
                head = above.min() * lm if len(above) else None
                if sill >= 0.25:
                    kind = "window"
                elif head is not None:
                    kind = "door"
                else:
                    kind = "gap"
                fa = s["a"] + s["e"] * ss[a]
                fb = s["a"] + s["e"] * ss[b - 1]
                th = w["t"]
                quad = np.array([fa, fb, fb + s["n"] * th, fa + s["n"] * th])
                o = {"type": kind, "wall": wi, "a": fa + s["n"] * th / 2, "b": fb + s["n"] * th / 2,
                     "quad": quad, "n": s["n"], "width": (ss[b - 1] - ss[a] + 1) * px,
                     "sill": sill if kind == "window" else None, "head": head,
                     "s": ((ss[a] - s0) * px, (ss[b - 1] - s0) * px)}
                ops.append(o)
                openings.append(o)
        elevations.append((wi, E, ops))

    # --- в метры ---------------------------------------------------------------------------
    def to_m(xy):
        return [round(float(ras.origin[0] + xy[0] * px), 3), round(float(ras.origin[1] + (H - xy[1]) * px), 3)]

    plan = LevelPlan(k, lv["floor_z"], lv["ceiling_z"])
    room_id = [f"L{k}-R{i + 1}" for i in range(len(rooms))]
    for i, R in enumerate(rooms):
        per = float(np.sum(np.linalg.norm(np.diff(np.vstack([R["poly"], R["poly"][:1]]), axis=0), axis=1))) * px
        plan.rooms.append(Room(room_id[i], f"Помещение {i + 1}", [to_m(p) for p in R["poly"]],
                               round(R["area"], 2), round(per, 2), tuple(to_m(R["label"])),
                               [{"kind": q["kind"], "a": to_m(q["a"]), "b": to_m(q["b"]),
                                 **({"center": to_m(q["c"]), "radius": round(q["r"] * px, 3)}
                                    if q["kind"] == "arc" else {})} for q in R["prims"]]))
    wall_id = {}
    for wi, w in enumerate(walls):
        wall_id[wi] = f"L{k}-W{wi + 1}"
        plan.walls.append(WallSegment(wall_id[wi], [to_m(w["a"]), to_m(w["b"])],
                                      [room_id[r] for r in w["rooms"]],
                                      None if w.get("one_sided") else round(w["t"] * px, 3), bool(w["outer"]),
                                      round(float(np.linalg.norm(w["b"] - w["a"])) * px, 3)))
    for i, o in enumerate(openings):
        plan.openings.append(Opening(f"L{k}-O{i + 1}", o["type"], wall_id[o["wall"]], to_m(o["a"]), to_m(o["b"]),
                                     [to_m(p) for p in o["quad"]], round(o["width"], 3),
                                     None if o["sill"] is None else round(o["sill"], 3),
                                     None if o["head"] is None else round(o["head"], 3)))
    # толща стен - как видна: между двумя гранями, не больше max_wall_m; ничего не достраивается
    union = np.zeros((H, W), np.uint8)
    for R in rooms:
        cv2.fillPoly(union, [np.round(R["poly"]).astype(np.int32)], 1)
    fill = (struct & (1 - union)).astype(np.uint8)
    # толща - только там, где с обеих сторон помещения (откос окна снаружи - не стена)
    near = np.zeros((H, W), np.int32)
    kn = _rect_kernel(2 * P(max_wall_m) + 1, 2 * P(max_wall_m) + 1)
    for R in rooms:
        near += cv2.dilate(R["mask"], kn)
    fill = (fill & (near >= 2)).astype(np.uint8)
    fill = _drop_small(fill, P(0.1))
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plan.slab = [[to_m(p) for p in c[:, 0, :].astype(float)] for c in cnts
                 if cv2.contourArea(c) * px * px >= 0.005]
    masks = {"free": free, "building": building, "faces": faces, "struct": struct, "space": space,
             "elevations": [(wall_id[wi], E, ops) for wi, E, ops in elevations]}
    return plan, masks


# ----------------------------------------------------------------------
# Рисунок
# ----------------------------------------------------------------------

ROOM_TINTS = ("#dce9f8", "#fbe3d6", "#d6f0e5", "#fbefcc", "#f8e1ea", "#d9ecd9", "#e3dff3", "#f8dcdc")
WALL, WALL_FILL = "#1f1f1f", "#8a8a8a"


def render(plan: LevelPlan, ras: Rasters, masks: dict, path: Path, title: str = "") -> None:
    from matplotlib.patches import Arc, Patch, Polygon

    from .debug import AQUA, INK, INK2, ORANGE, SURFACE, _plt

    plt = _plt()
    xy = np.array([p for s in plan.slab for p in s] + [p for r in plan.rooms for p in r.points])
    if not len(xy):
        xy = np.array([[0.0, 0.0], [1.0, 1.0]])
    xlim = (xy[:, 0].min() - 0.4, xy[:, 0].max() + 0.4)
    ylim = (xy[:, 1].min() - 0.4, xy[:, 1].max() + 0.4)
    span_x, span_y = xlim[1] - xlim[0], ylim[1] - ylim[0]
    fig = plt.figure(figsize=(11 * span_x / max(span_x, span_y) + 4.2, 11 * span_y / max(span_x, span_y) + 0.6))
    ax = fig.add_axes([0.02, 0.03, 0.70, 0.94])
    side = fig.add_axes([0.74, 0.03, 0.25, 0.94])
    side.axis("off")
    ax.set_facecolor(SURFACE)
    for s in plan.slab:
        ax.add_patch(Polygon(s, closed=True, facecolor=WALL_FILL, edgecolor="none", zorder=1))
    for i, rm in enumerate(plan.rooms):
        ax.add_patch(Polygon(rm.points, closed=True, facecolor=ROOM_TINTS[i % len(ROOM_TINTS)],
                             edgecolor="none", zorder=2))
        # грани - векторы и дуги, как видны из помещения
        ax.add_patch(Polygon(rm.points, closed=True, fill=False, edgecolor=WALL, lw=1.6, zorder=5,
                             joinstyle="miter"))
        ax.text(*rm.label_xy, f"{i + 1}\n{rm.area:.1f} м²".replace(".", ","), ha="center", va="center",
                fontsize=10, color=INK, zorder=6)
    for o in plan.openings:
        a, b = np.array(o.a), np.array(o.b)
        q = np.array(o.points)
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        nrm = q[3] - q[0]
        t = np.linalg.norm(nrm)
        nrm = nrm / max(t, 1e-9)
        ax.add_patch(Polygon(q, closed=True, facecolor=SURFACE, edgecolor="none", zorder=3))
        if o.opening_type == "window":
            for f in (-0.18, 0.18):
                ax.plot(*np.c_[a + nrm * t * f, b + nrm * t * f], color=AQUA, lw=1.3, zorder=4)
            ax.add_patch(Polygon(q, closed=True, fill=False, edgecolor=AQUA, lw=0.8, zorder=4))
        elif o.opening_type == "door":
            # створка - внутрь помещения (против нормали стены)
            r = o.width
            hinge = q[0]
            leaf = -nrm
            ax.plot(*np.c_[hinge, hinge + leaf * r], color=ORANGE, lw=1.4, zorder=4)
            a0 = math.degrees(math.atan2(leaf[1], leaf[0]))
            a1 = math.degrees(math.atan2(d[1], d[0]))
            lo, hi = (a0, a1) if (a1 - a0) % 360 < 180 else (a1, a0)
            ax.add_patch(Arc(tuple(hinge), 2 * r, 2 * r, theta1=lo, theta2=hi, color=ORANGE, lw=1.0, zorder=4))
        c = (a + b) / 2 + nrm * (t / 2 + 0.18)
        rot = math.degrees(math.atan2(d[1], d[0]))
        rot = rot - 180 if rot > 90 else rot + 180 if rot < -90 else rot
        ax.text(*c, f"{o.width * 1000:.0f}", fontsize=7, color=INK2, ha="center", va="center", rotation=rot,
                zorder=6)
    sx, sy = xlim[0] + 0.3, ylim[0] + 0.15
    ax.plot([sx, sx + 1], [sy, sy], color=INK, lw=2)
    ax.text(sx + 0.5, sy + 0.08, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    nd = sum(o.opening_type == "door" for o in plan.openings)
    nw = sum(o.opening_type == "window" for o in plan.openings)
    ng = sum(o.opening_type == "gap" for o in plan.openings)
    na = sum(q["kind"] == "arc" for r in plan.rooms for q in r.faces)
    lines = [title, f"пол {plan.floor_z:+.2f} м, потолок {plan.ceiling_z - plan.floor_z:.2f} м над полом",
             f"стен {len(plan.walls)} (дуговых граней {na})", f"дверей {nd}, окон {nw}, проёмов {ng}", ""]
    for i, rm in enumerate(plan.rooms):
        lines.append(f"{i + 1:>2}  {rm.area:6.1f} м²   периметр {rm.perimeter:5.1f} м".replace(".", ","))
    lines.append(f"    {sum(r.area for r in plan.rooms):6.1f} м²   всего".replace(".", ","))
    lines.append("")
    for o in plan.openings:
        kind = {"door": "дверь", "window": "окно", "gap": "проём"}[o.opening_type]
        hs = []
        if o.sill is not None:
            hs.append(f"низ {o.sill:.2f}")
        if o.head is not None:
            hs.append(f"верх {o.head:.2f}")
        lines.append(f"{kind:<6} {o.width * 1000:5.0f} мм  {' '.join(hs)}".replace(".", ","))
    side.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=8.5, family="monospace", color=INK)
    side.legend(handles=[Patch(color=WALL_FILL, label="толща стены между видимыми гранями"),
                         Patch(facecolor=SURFACE, edgecolor=AQUA, label="окно"),
                         Patch(color=ORANGE, label="дверь")],
                loc="lower left", frameon=False, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def render_elevations(items, px: float, layer_m: float, path: Path, min_len_m: float = 1.0) -> None:
    """Развёртки стен: по горизонтали - длина вдоль стены, по вертикали - слои по 10 см;
    тёмное - точки есть. Рамки - найденные проёмы."""
    from matplotlib.patches import Rectangle

    from .debug import AQUA, INK, INK2, ORANGE, _plt

    plt = _plt()
    items = [it for it in items if len(it[1]) * px >= min_len_m]
    if not items:
        return
    cols = 3
    rows = math.ceil(len(items) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(15, 1.9 * rows + 0.3), squeeze=False)
    colors = {"door": ORANGE, "window": AQUA, "gap": INK2}
    for ax, (wid, E, ops) in zip(axs.flat, items):
        L, nl = len(E) * px, E.shape[1]
        ax.imshow(E.T, origin="lower", aspect="auto", cmap="Greys", vmin=0, vmax=1.6,
                  extent=[0, L, 0, nl * layer_m], interpolation="nearest")
        for o in ops:
            s0, s1 = o["s"]
            z0 = o["sill"] or 0.0
            z1 = o["head"] if o["head"] is not None else nl * layer_m
            ax.add_patch(Rectangle((s0, z0), s1 - s0, z1 - z0, fill=False, edgecolor=colors[o["type"]], lw=1.6))
        ax.set_title(f"{wid}  {L:.2f} м".replace(".", ","), loc="left", fontsize=9, color=INK)
        ax.set_yticks([0, 1, 2, 3])
        ax.tick_params(labelsize=7, colors=INK2)
    for ax in list(axs.flat)[len(items):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------

def floorplan(path, out_dir, px: float = 0.01, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels, angle, rasters = build_rasters(path, px=px, log=log)
    result = {"file": Path(path).name, "rotation_deg": round(angle, 3),
              "transform": "plan = Rz(-rotation_deg) * world (x, y); z - над полом уровня",
              "levels": []}
    for k, (lv, ras) in enumerate(zip(levels, rasters), 1):
        plan, masks = analyse_level(ras, lv, k)
        name = "plan.png" if len(levels) == 1 else f"plan_L{k}.png"
        render(plan, ras, masks, out / name, title=f"Уровень {k}" if len(levels) > 1 else "")
        render_elevations(masks["elevations"], ras.px, ras.layer_m,
                          out / ("walls.png" if len(levels) == 1 else f"walls_L{k}.png"))
        d = asdict(plan)
        result["levels"].append(d)
        log(f"уровень {k}: стен {len(plan.walls)}, проёмов {len(plan.openings)}, помещений {len(plan.rooms)}, "
            f"{sum(r.area for r in plan.rooms):.1f} м²")
    (out / "elements.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    return result
