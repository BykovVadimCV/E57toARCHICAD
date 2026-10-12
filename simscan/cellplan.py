"""План через разбиение на ячейки и глобальную разметку (indoor modelling / scan-to-BIM).

    python -m simscan cellplan D:/scans/kvartira.e57 --out D:/plan [--pred pred/kvartira]
    -> plan.png (на уровень - plan_L1.png ...), cells.png (разбиение и разметка), walls.png
       (развёртки стен с проёмами), elements.json, plan_L1.json ... (оси и толщины стен для
       export-ifc)

С --pred (каталог infer для этого файла: L1/prob.npz + scene.json ...) вероятности сети входят
в задачу (раздел «Вероятности сети» ниже): кандидаты стен - объединение правил wallheat и
стен / проёмов сети, данные ячейки - свидетельство пустоты вместе с вероятностью помещения,
стоимость ребра каждого типа - вместе с вероятностью этого класса сети вдоль ребра. Без
--pred - только правила wallheat.

Вместо заливки помещений (протекает в разрывы, двери без перемычки не замыкает):
  1. срезы по 10 см, тепловая карта и разметка стена / дверь / окно / шум - wallheat;
  2. кандидаты стен - грани маски «стена или дверь», разбитые на прямые и дуги (самый
     длинный кусок, который прямая или дуга покрывает с СКО <= tol); каждый кандидат
     продлевается до первого пересечения с другим или с краем (кинетическое разбиение с
     k = 1, Bauchet & Lafarge): разбиение замкнуто по построению, разрывы в тени сканера
     закрываются продлением, лишние продления потом снимает разметка;
  3. разбиение плоскости продлёнными кандидатами - ячейки (растровая аранжировка: линии
     толщиной в пиксель, ячейки - 4-связные области между ними);
  4. видимость: лучи в плане от стоянок (позы сферических снимков в E57) до стены -
     пройденное лучом пусто (Oesau, Adan & Huber); без стоянок - только пол и потолок;
  5. разметка ячеек «снаружи / помещение k» - целочисленная задача (Ochmann 2016/2019,
     Mura 2016): данные ячейки - доля пола, потолка и видимости (свидетельство пустоты) и
     доля стены; ребро между ячейками разной метки стоит lambda * длина * (1 - опора), где
     опора - доля ребра на стене или двери: резать по стене дёшево, по пустоте дорого.
     Начальные помещения (затравки) - свободное, разрезанное стенами и перемычками и сжатое
     на 25 см (узкие протечки рвутся); метка помещения разрешена только у своей затравки;
  6. стены - то, что между помещениями (толщина - как видна), наружные - граница
     помещения со «снаружи»; контур помещения - прямые и дуги;
  7. проёмы - по развёртке каждой грани: в каждой позиции доли занятых срезов внизу
     (< 0,9 м), посередине и вверху (>= 2 м), точная разметка вдоль грани (Витерби) на
     стена / окно / дверь / проём со штрафом за смену (вместо порогов 1,0-1,8 м; одномерный
     аналог разметки поверхности стены у Michailidis & Pajarola).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

OUT = 0


# ----------------------------------------------------------------------
# Кандидаты стен и кинетическое продление
# ----------------------------------------------------------------------

def wall_candidates(wallmask: np.ndarray, P, tol_px: float, min_len_m: float = 0.3) -> list:
    """Грани маски стен -> куски-прямые и дуги {kind, a, b, (c, r), pts}."""
    import cv2

    from .floorplan import _fit_primitives, _proj

    cnts, _ = cv2.findContours(wallmask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        if len(c) < P(min_len_m):
            continue
        cnt = c[:, 0, :].astype(float)
        for q in _fit_primitives(cnt, tol_px, P(0.1), P(1.5), P(40.0)):
            q["a"], q["b"] = _proj(q, q["pts"][0]), _proj(q, q["pts"][-1])
            if np.linalg.norm(q["b"] - q["a"]) >= P(min_len_m) or (q["kind"] == "arc" and len(q["pts"]) >= P(min_len_m)):
                out.append(q)
    return out


def _poly(q, step: float = 2.0) -> np.ndarray:
    """Кусок как ломаная (дуга - хордами по step пикселей)."""
    if q["kind"] == "line":
        return np.array([q["a"], q["b"]])
    a0 = math.atan2(*(q["a"] - q["c"])[::-1])
    a1 = math.atan2(*(q["b"] - q["c"])[::-1])
    mid = math.atan2(*(q["pts"][len(q["pts"]) // 2] - q["c"])[::-1])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    dm = (mid - a0 + math.pi) % (2 * math.pi) - math.pi
    if np.sign(dm) != np.sign(da) or abs(dm) > abs(da):
        da -= np.sign(da) * 2 * math.pi
    n = max(2, int(abs(da) * q["r"] / step))
    t = a0 + da * np.linspace(0, 1, n + 1)
    pts = q["c"] + q["r"] * np.c_[np.cos(t), np.sin(t)]
    pts[0], pts[-1] = q["a"], q["b"]
    return pts


def extend_candidates(cands: list, shape, max_ext_px: float, k_hits: int = 2) -> list:
    """Кинетическое продление: каждый конец идёт по касательной, проходит k_hits - 1 чужих
    кандидатов и останавливается на k_hits-м (или на краю растра; max_ext_px - предел).
    k_hits = 1 оставляет T-стыки недоразрезанными: стена, упёршаяся в поперечную, не делит
    пространство за ней. Возвращает ломаные (с продлениями)."""
    import cv2

    H, W = shape
    ids = np.zeros((H, W), np.int32)
    polys = [_poly(q) for q in cands]
    for k, p in enumerate(polys, 1):
        cv2.polylines(ids, [np.round(p).astype(np.int32)], False, k, 1, lineType=cv2.LINE_8)
    out = []
    for k, p in enumerate(polys, 1):
        p = p.copy()
        for end in (0, -1):
            nb = p[1] if end == 0 else p[-2]
            d = p[end] - nb
            L = np.linalg.norm(d)
            if L < 1e-9:
                continue
            d /= L
            hit = None
            seen, last = 0, 0
            for s in range(2, int(max_ext_px) + 1):
                x, y = np.round(p[end] + d * s).astype(int)
                if not (0 <= x < W and 0 <= y < H):
                    break
                v = ids[y, x]
                if v and v != k and v != last:
                    seen += 1
                    last = v
                    if seen >= k_hits:
                        hit = p[end] + d * (s + 1)        # пиксель за линией - стык без щели
                        break
                elif not v:
                    last = 0
            if hit is None and s >= 2:                       # до края растра
                hit = p[end] + d * s
            if hit is not None:
                p = np.vstack([hit, p]) if end == 0 else np.vstack([p, hit])
        out.append(p)
    return out


# ----------------------------------------------------------------------
# Разбиение на ячейки
# ----------------------------------------------------------------------

def arrangement(polys: list, shape) -> tuple[np.ndarray, np.ndarray, int]:
    """Растровая аранжировка: линии толщиной в пиксель -> (метки ячеек, маска линий, число)."""
    import cv2

    H, W = shape
    lines = np.zeros((H, W), np.uint8)
    for p in polys:
        cv2.polylines(lines, [np.round(p).astype(np.int32)], False, 1, 1, lineType=cv2.LINE_8)
    lines[0, :] = lines[-1, :] = lines[:, 0] = lines[:, -1] = 1
    n, cells = cv2.connectedComponents((1 - lines).astype(np.uint8), connectivity=4)
    return cells, lines.astype(bool), n


# ----------------------------------------------------------------------
# Видимость со стоянок
# ----------------------------------------------------------------------

def visibility(stations_px: np.ndarray, barrier: np.ndarray, max_px: float, step_deg: float = 0.1) -> np.ndarray:
    """Лучи в плане от каждой стоянки до барьера: пройденные пиксели пусты."""
    H, W = barrier.shape
    vis = np.zeros((H, W), bool)
    ang = np.radians(np.arange(0, 360, step_deg))
    dx, dy = np.cos(ang), np.sin(ang)
    for x0, y0 in stations_px:
        if not (0 <= x0 < W and 0 <= y0 < H):
            continue
        alive = np.ones(len(ang), bool)
        for t in np.arange(0, max_px, 0.7):
            x = np.round(x0 + dx * t).astype(int)
            y = np.round(y0 + dy * t).astype(int)
            alive &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
            if not alive.any():
                break
            xa, ya = x[alive], y[alive]
            hit = barrier[ya, xa]
            idx = np.flatnonzero(alive)
            alive[idx[hit]] = False
            vis[ya[~hit], xa[~hit]] = True
    return vis


def image_stations(path) -> np.ndarray:
    """Стоянки - по позам сферических снимков E57 (6 граней куба на стоянку). Если все позы
    в одной точке (поза облака, а не стоянок) - пусто."""
    from .e57read import E57Reader
    from .inspect_e57 import _image_stations

    with E57Reader(path) as r:
        st = _image_stations(r.tree)
    if len(st) < 2 or float(np.ptp(st[:, :2], axis=0).max()) < 0.3:
        return np.zeros((0, 3))
    return st


# ----------------------------------------------------------------------
# Вероятности сети
# ----------------------------------------------------------------------

NET_SPACE = (1, 6)                                       # помещение, мебель и инженерия
NET_EDGE = {"wall": 2, "door": 3, "window": 4, "gap": 5}  # тип ребра -> класс сети (netinput.CLASSES)


def load_net(pred, level: dict, frame: dict, tol_z: float = 0.5) -> dict | None:
    """prob.npz уровня из каталога infer (уровень - по ближайшей отметке пола, не дальше tol_z)
    -> в кадр cellplan: {"prob" [8, H, W] float32, "valid" [H, W], "dir"}; None - нет уровня."""
    import cv2

    best = None
    for sc in sorted(Path(pred).rglob("scene.json")):
        if not (sc.parent / "prob.npz").exists():
            continue
        meta = json.loads(sc.read_text(encoding="utf-8"))
        dz = abs(float(meta.get("floor_z", 0.0)) - float(level["floor_z"]))
        if dz <= tol_z and (best is None or dz < best[0]):
            best = (dz, sc.parent, meta)
    if best is None:
        return None
    _, d, meta = best
    prob = np.load(d / "prob.npz")["prob"].astype(np.float32)
    M = np.asarray(meta["pixel_to_world"], float)             # (j, i, 1) -> мир (x, y)
    H, W = frame["shape"]
    px = frame["px"]
    jj, ii = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xp = frame["origin"][0] + (jj + 0.5) * px                 # центр пикселя в СК плана
    yp = frame["origin"][1] + (H - 1 - ii + 0.5) * px
    a = math.radians(frame["rotation_deg"])                   # мир = Rz(rotation) * план
    xw = math.cos(a) * xp - math.sin(a) * yp - M[0, 2]
    yw = math.sin(a) * xp + math.cos(a) * yp - M[1, 2]
    inv = np.linalg.inv(M[:, :2])
    mx = (inv[0, 0] * xw + inv[0, 1] * yw).astype(np.float32)
    my = (inv[1, 0] * xw + inv[1, 1] * yw).astype(np.float32)
    out = np.stack([cv2.remap(c, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for c in prob])
    valid = cv2.remap(np.ones(prob.shape[1:], np.float32), mx, my, cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    out[:, ~valid] = 0.0
    return {"prob": out, "valid": valid, "dir": str(d)}


# ----------------------------------------------------------------------
# Затравки помещений
# ----------------------------------------------------------------------

def room_seeds(space: np.ndarray, core_px: int, narrow_px: int, min_area_px: float) -> np.ndarray:
    """Пустота -> области-затравки помещений (0 - не помещение). Комнаты - «пузыри» карты
    расстояний до стен, двери и протечки - перешейки."""
    import cv2

    dt = cv2.distanceTransform(space, cv2.DIST_L2, 5)
    n1, wide = cv2.connectedComponents((dt > core_px).astype(np.uint8), connectivity=8)
    n2, narrow = cv2.connectedComponents((dt > narrow_px).astype(np.uint8), connectivity=8)
    markers = np.zeros(space.shape, np.int32)
    k = 0
    for c in range(1, n1):
        m = wide == c
        if m.sum() >= 4:
            k += 1
            markers[m] = k
    for c in range(1, n2):
        m = narrow == c
        if not (markers[m] > 0).any() and m.sum() >= 4:
            k += 1
            markers[m] = k
    # затравка = ядро, расширенное обратно на свой радиус (открытие): полосы вдоль стен и
    # перешейки уже 2 * радиуса в затравку не попадают - их потом распределит разметка
    union = np.zeros(space.shape, np.uint8)
    for v in range(1, k + 1):
        m = (markers == v).astype(np.uint8)
        r = core_px if (dt[m > 0] > core_px).any() else narrow_px
        union |= cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    union &= space
    bg = k + 1
    markers[union == 0] = bg
    img = np.clip(255 - dt / max(dt.max(), 1e-6) * 255, 0, 255).astype(np.uint8)
    cv2.watershed(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), markers)
    markers[(markers == bg) | (markers < 0)] = 0
    markers[union == 0] = 0
    for v in np.unique(markers):
        if v > 0 and (markers == v).sum() < min_area_px:
            markers[markers == v] = 0
    return markers


# ----------------------------------------------------------------------
# Шаг 1. Признаки пикселя (высота не сворачивается)
# ----------------------------------------------------------------------

FEATURES = ("low", "mid", "top", "floor", "ceil", "density", "speckle")


def pixel_features(L: dict, occ: np.ndarray, zc: np.ndarray) -> dict:
    """Каналы на пиксель: доли занятых срезов внизу (< 0,9 м), посередине (0,9-2,0) и вверху
    (>= 2,0 м до 10 см под потолком); виден пол; виден потолок; плотность точек (лог, к 99-му
    процентилю); крапчатость - доля соседей 7x7, занятых «не гранью» (1-3 среза из всех)."""
    import cv2

    top = L["level"]["ceiling_z"] - L["level"]["floor_z"]
    k3 = np.ones((3, 3), np.uint8)
    f = {}
    for name, m in (("low", zc < 0.9), ("mid", (zc >= 0.9) & (zc < 2.0)), ("top", (zc >= 2.0) & (zc < top - 0.1))):
        f[name] = occ[m].mean(0).astype(np.float32) if m.any() else np.zeros(occ.shape[1:], np.float32)
    f["floor"] = cv2.morphologyEx(L["floor"].astype(np.uint8), cv2.MORPH_CLOSE, k3).astype(np.float32)
    f["ceil"] = cv2.morphologyEx(L["ceil"].astype(np.uint8), cv2.MORPH_CLOSE, k3).astype(np.float32)
    pts = np.log1p(L["points"].astype(np.float32))
    nz = pts[pts > 0]
    f["density"] = np.clip(pts / (np.percentile(nz, 99) if len(nz) else 1.0), 0, 1)
    n_occ = occ.sum(0)
    sparse = ((n_occ >= 1) & (n_occ <= 3)).astype(np.float32)
    f["speckle"] = cv2.blur(sparse, (7, 7))
    return f


# ----------------------------------------------------------------------
# Шаг 2. Рёбра разбиения и их профили
# ----------------------------------------------------------------------

def edge_pixels(cells: np.ndarray, lines: np.ndarray) -> dict:
    """Пиксели линий, разделяющие пару ячеек: {(a, b): индексы (ys, xs)}. Пиксель на стыке
    трёх ячеек идёт в пару (наименьшая, наибольшая)."""
    pad = np.pad(cells, 1)
    ys, xs = np.nonzero(lines)
    nb = np.stack([pad[ys + 1 + dy, xs + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1)
    big = np.iinfo(np.int32).max
    lo = np.where(nb > 0, nb, big).min(1)
    hi = nb.max(1)
    ok = (lo < hi) & (lo != big)
    code = lo[ok].astype(np.int64) * (int(cells.max()) + 1) + hi[ok]
    order = np.argsort(code, kind="stable")
    code, yy, xx = code[order], ys[ok][order], xs[ok][order]
    cuts = np.flatnonzero(np.diff(code)) + 1
    out = {}
    for c, y_, x_ in zip(np.split(code, cuts), np.split(yy, cuts), np.split(xx, cuts)):
        if len(c):
            a, b = divmod(int(c[0]), int(cells.max()) + 1)
            out[(a, b)] = (y_, x_)
    return out


def edge_profile(pix, cells: np.ndarray, F: dict, Fmax: dict, P, bin_m: float = 0.05, side_m: float = 0.25,
                 N: dict | None = None):
    """Профиль ребра с шагом bin_m: занятость полос на ребре (±2 px), с каждой стороны -
    пол или потолок виден, крапчатость; с сетью (N) - вероятности её классов на ребре
    (n_wall, n_door, n_window, n_gap). -> dict или None."""
    ys, xs = pix
    H, W = cells.shape
    p = np.c_[xs, ys].astype(float)
    m = p.mean(0)
    if len(p) < 3:
        return None
    _, _, vt = np.linalg.svd(p - m, full_matrices=False)
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    t = (p - m) @ d
    step = max(1.0, P(bin_m))
    k = np.floor((t - t.min()) / step).astype(int)
    nb = int(k.max()) + 1
    cnt = np.bincount(k, minlength=nb).astype(float)
    good = cnt > 0
    prof = {}
    for name in ("low", "mid", "top"):
        v = np.bincount(k, Fmax[name][ys, xs], minlength=nb) / np.maximum(cnt, 1)
        prof[name] = v
    for name in (NET_EDGE if N is not None else ()):
        prof["n_" + name] = np.bincount(k, N[name][ys, xs], minlength=nb) / np.maximum(cnt, 1)
    centers = m + d * (t.min() + (np.arange(nb) + 0.5) * step)[:, None]
    # стороны: + нормаль и - нормаль; какая из них ячейка a - по большинству
    side = {}
    for sg in (1, -1):
        q = np.round(centers + sg * nrm * P(side_m)).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, W - 1)
        q[:, 1] = np.clip(q[:, 1], 0, H - 1)
        # чья сторона - вплотную к ребру (за тонкой толщей стены на 25 см уже чужая ячейка)
        c = np.round(centers + sg * nrm * 2.0).astype(int)
        c[:, 0] = np.clip(c[:, 0], 0, W - 1)
        c[:, 1] = np.clip(c[:, 1], 0, H - 1)
        side[sg] = {"floor": np.maximum(Fmax["floor"][q[:, 1], q[:, 0]], Fmax["ceil"][q[:, 1], q[:, 0]]),
                    "speckle": F["speckle"][q[:, 1], q[:, 0]],
                    "cell": cells[c[:, 1], c[:, 0]]}
    t_lo, t_hi = t.min(), t.max()
    return {"m": m, "d": d, "n": nrm, "t0": t_lo, "t1": t_hi, "step": step, "nb": nb, "good": good,
            **prof, "side": side, "len_px": float(t_hi - t_lo + 1)}


# Типы ребра: стоимость на бин (м) и пределы длины, м
EDGE_TYPES = {
    "wall":   (0.05, 1e9),
    "window": (0.4, 3.0),
    "door":   (0.6, 1.3),
    "gap":    (0.6, 3.0),
    "shadow": (0.1, 1e9),
}
CONTEXT_TYPES = {"RR": ("wall", "door", "gap"), "RO": ("wall", "window", "door", "shadow")}


def type_costs(pr: dict, ctx: str, out_side: int | None) -> dict:
    """Стоимость типа в каждом бине (таблица соответствия признакам)."""
    lo, mi, tp = pr["low"], pr["mid"], pr["top"]
    fa, fb = pr["side"][1]["floor"], pr["side"][-1]["floor"]
    if ctx == "RO":
        fo = pr["side"][out_side]["floor"]
        so = pr["side"][out_side]["speckle"]
    c = {"wall": (1 - lo) + (1 - mi) + (1 - tp),
         "door": lo + mi + (1 - tp) + 0.5 * (1 - fa) + 0.5 * (1 - fb),
         # проём во всю высоту - открытый проход: штраф 0,6 / м за сложность, иначе любая
         # пустая линия режет комнату
         "gap": lo + mi + tp + 0.5 * (1 - fa) + 0.5 * (1 - fb) + 0.6}
    if ctx == "RO":
        c["window"] = (1 - lo) + mi + (1 - tp) + 0.7 * fo + 0.3 * (1 - np.minimum(1, 3 * so))
        c["shadow"] = lo + mi + tp + 1.0 * fo + 0.3
        c["door"] = lo + mi + (1 - tp) + 0.5 * (1 - fa) + 0.5 * (1 - fb)
    if "n_wall" in pr:                    # сеть: тип ребра дешевле там, где сеть видит его класс
        w = pr.get("net_w", 1.0)
        n_any = _net_any(pr)
        for t in c:
            c[t] = c[t] + w * (n_any if t == "shadow" else 1 - pr["n_" + t])
    return {k: v for k, v in c.items() if k in CONTEXT_TYPES[ctx]}


def _net_any(pr: dict) -> np.ndarray:
    """Вероятность сети «на ребре стена или проём» (тень и ложное продление - против неё)."""
    return np.minimum(1.0, sum(pr["n_" + t] for t in NET_EDGE))


def segment_edge(pr: dict, ctx: str, out_side: int | None, bin_m: float, tau: float = 0.15):
    """Точная полумарковская разметка ребра на куски типов с пределами длины и штрафом tau
    за каждый кусок-проём. -> (стоимость, [(тип, бин от, бин до)])."""
    costs = type_costs(pr, ctx, out_side)
    n = pr["nb"]
    names = list(costs)
    cum = {k: np.r_[0, np.cumsum(v * bin_m)] for k, v in costs.items()}
    best = np.full(n + 1, np.inf)
    best[0] = 0.0
    back = [None] * (n + 1)
    lim = {k: (max(1, int(round(EDGE_TYPES[k][0] / bin_m))), int(EDGE_TYPES[k][1] / bin_m)) for k in names}
    for j in range(1, n + 1):
        for k in names:
            dmin, dmax = lim[k]
            if j < dmin:
                continue
            d = np.arange(dmin, min(dmax, j) + 1)
            v = best[j - d] + (cum[k][j] - cum[k][j - d]) + (0.0 if k == "wall" else tau)
            i = int(v.argmin())
            if v[i] < best[j]:
                best[j] = v[i]
                back[j] = (k, int(d[i]))
    if not np.isfinite(best[n]):
        return float(np.sum(costs["wall"]) * bin_m), [("wall", 0, n)]
    segs, j = [], n
    while j > 0:
        k, d = back[j]
        segs.append((k, j - d, j))
        j -= d
    merged = []
    for k, a, b in segs[::-1]:                           # соседние куски стены - один кусок
        if merged and merged[-1][0] == k == "wall":
            merged[-1] = (k, merged[-1][1], b)
        else:
            merged.append((k, a, b))
    return float(best[n]), merged


def same_cost(pr: dict, bin_m: float) -> float:
    """Ребро внутри одного помещения - ложное продление: вдоль него ничего нет. Верх весит
    0,3: короб или балка под потолком внутри комнаты допустимы (иначе короб режет комнату
    как перемычка)."""
    c = pr["low"] + pr["mid"] + 0.3 * pr["top"]
    if "n_wall" in pr:
        c = c + pr.get("net_w", 1.0) * _net_any(pr)
    return float(np.sum(c) * bin_m)


# ----------------------------------------------------------------------
# Шаг 3. Совместная разметка ячеек и рёбер (ILP)
# ----------------------------------------------------------------------

def joint_labels(D: np.ndarray, allowed: np.ndarray, E: list, time_limit: float = 120.0) -> np.ndarray:
    """Ячейки: метка 0 - снаружи, 1..K - помещения. Ребро e = (a, b, c_same, c_RR, c_ROa, c_ROb)
    (c_ROa - снаружи сторона a). Стоимость ребра = c_same (одна метка-помещение) | c_RR
    (разные помещения) | c_ROa / c_ROb (помещение и снаружи) | 0 (обе снаружи).
    Переменные: x[c, l] бинарные; на ребро z (метки различны), qa, qb (снаружи ровно a / b),
    t (обе снаружи) - точные, через x. Цель (без константы): (c_RR - c_same) z +
    (c_ROa - c_RR) qa + (c_ROb - c_RR) qb - c_same t."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    nc, nl = D.shape
    var = -np.ones((nc, nl), np.int64)
    idx = np.argwhere(allowed)
    var[idx[:, 0], idx[:, 1]] = np.arange(len(idx))
    nx, ne = len(idx), len(E)
    base = nx
    cost = np.zeros(nx + 4 * ne)
    cost[:nx] = D[idx[:, 0], idx[:, 1]]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = [0]

    def row(terms, lb, ub):
        for c_, v_ in terms:
            rows.append(r[0])
            cols.append(c_)
            vals.append(v_)
        lo.append(lb)
        hi.append(ub)
        r[0] += 1

    for c in range(nc):
        v = var[c][var[c] >= 0]
        row([(int(j), 1.0) for j in v], 1.0, 1.0)
    for e, (a, b, cs, crr, croa, crob) in enumerate(E):
        z, qa, qb, t = base + 4 * e, base + 4 * e + 1, base + 4 * e + 2, base + 4 * e + 3
        cost[z], cost[qa], cost[qb], cost[t] = crr - cs, croa - crr, crob - crr, -cs
        for l in np.flatnonzero(allowed[a] | allowed[b]):
            va, vb = var[a, l], var[b, l]
            ta = [(int(va), 1.0)] if va >= 0 else []
            tb = [(int(vb), 1.0)] if vb >= 0 else []
            # z >= xa - xb, z >= xb - xa
            row([(z, 1.0)] + [(j, -w) for j, w in ta] + [(j, w) for j, w in tb], 0.0, np.inf)
            row([(z, 1.0)] + [(j, w) for j, w in ta] + [(j, -w) for j, w in tb], 0.0, np.inf)
            if va >= 0 and vb >= 0:                      # z <= 2 - xa - xb
                row([(z, 1.0), (int(va), 1.0), (int(vb), 1.0)], -np.inf, 2.0)
        oa, ob = var[a, OUT], var[b, OUT]
        OA = [(int(oa), 1.0)] if oa >= 0 else []
        OB = [(int(ob), 1.0)] if ob >= 0 else []
        neg = lambda T: [(j, -w) for j, w in T]          # noqa: E731
        # qa = oa * (1 - ob): qa >= oa - ob, qa <= oa, qa <= 1 - ob
        row([(qa, 1.0)] + neg(OA) + OB, 0.0, np.inf)
        row([(qa, 1.0)] + neg(OA), -np.inf, 0.0)
        row([(qa, 1.0)] + OB, -np.inf, 1.0)
        row([(qb, 1.0)] + neg(OB) + OA, 0.0, np.inf)
        row([(qb, 1.0)] + neg(OB), -np.inf, 0.0)
        row([(qb, 1.0)] + OA, -np.inf, 1.0)
        # t = oa * ob
        row([(t, 1.0)] + neg(OA) + neg(OB), -1.0, np.inf)
        row([(t, 1.0)] + neg(OA), -np.inf, 0.0)
        row([(t, 1.0)] + neg(OB), -np.inf, 0.0)
    A = coo_matrix((vals, (rows, cols)), shape=(r[0], nx + 4 * ne)).tocsr()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi),
               integrality=np.r_[np.ones(nx), np.zeros(4 * ne)],
               bounds=Bounds(np.zeros(nx + 4 * ne), np.ones(nx + 4 * ne)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-3, "disp": False})
    if res.x is None:
        return np.where(allowed, D, np.inf).argmin(1)
    lab = np.zeros(nc, np.int64)
    best = np.full(nc, -1.0)
    for (c, l), v in zip(idx, res.x[:nx]):
        if v > best[c]:
            best[c] = v
            lab[c] = l
    return lab


# ----------------------------------------------------------------------
# Уровень целиком
# ----------------------------------------------------------------------

def analyse_level(L: dict, A: dict, frame: dict, stations_px: np.ndarray, k: int, bin_m: float = 0.05,
                  tol_m: float = 0.012, min_room_m2: float = 1.0, seed_m: float = 0.35, room_core_m: float = 0.6,
                  net: dict | None = None, net_w: float = 1.0, log=print):
    import cv2

    from .floorplan import LevelPlan, Opening, Room, WallSegment, _room_outline
    from .wallheat import DOOR, WALL

    px = frame["px"]
    H, W = frame["shape"]
    P = lambda m: max(1, int(round(m / px)))                      # noqa: E731
    lab = A["lab"]
    occ, zc = A["occ"], A["zc"]
    k3 = np.ones((3, 3), np.uint8)
    k5 = np.ones((5, 5), np.uint8)

    # --- шаг 1: признаки -------------------------------------------------------------------------
    F = pixel_features(L, occ, zc)
    Fmax = {n: cv2.dilate(F[n], k5) for n in ("low", "mid", "top", "floor", "ceil")}

    N = None
    walls = (lab == WALL) | (lab == DOOR)
    if net is not None:                                   # сеть: стены и проёмы - тоже кандидаты
        npr = net["prob"]
        N = {t: cv2.dilate(npr[c], k5) for t, c in NET_EDGE.items()}
        n_space = npr[list(NET_SPACE)].sum(0)
        walls |= npr[[NET_EDGE["wall"], NET_EDGE["door"], NET_EDGE["window"]]].sum(0) > 0.5

    # --- шаг 2: кандидаты, разбиение, профили рёбер -------------------------------------------------
    walls = cv2.morphologyEx(walls.astype(np.uint8), cv2.MORPH_CLOSE, k3)
    cands = wall_candidates(walls, P, max(tol_m / px, 0.6))
    polys = extend_candidates(cands, (H, W), 1e9)
    cells, lines, nc = arrangement(polys, (H, W))
    epix = edge_pixels(cells, lines)
    log(f"  кандидатов {len(cands)}, ячеек {nc - 1}, рёбер {len(epix)}")

    # --- свидетельства по ячейкам ------------------------------------------------------------------
    barrier = cv2.dilate(cv2.morphologyEx((lab > 0).astype(np.uint8), cv2.MORPH_CLOSE, k3), k3).astype(bool)
    vis = visibility(stations_px, barrier, P(20.0)) if len(stations_px) else np.zeros((H, W), bool)
    vis = cv2.dilate(vis.astype(np.uint8), k3).astype(bool)
    empty = (F["floor"] > 0) | (F["ceil"] > 0) | vis
    area = np.bincount(cells.ravel(), minlength=nc).astype(float)
    f_empty = np.bincount(cells.ravel(), empty.ravel(), minlength=nc) / np.maximum(area, 1)
    f_room = f_empty                                      # свидетельство «помещение»
    if net is not None:                                   # вместе с вероятностью помещения у сети
        f_nv = np.bincount(cells.ravel(), net["valid"].ravel(), minlength=nc) / np.maximum(area, 1)
        f_space = np.bincount(cells.ravel(), n_space.ravel(), minlength=nc) / np.maximum(area * f_nv, 1)
        f_room = np.where(f_nv > 0.5, (f_empty + net_w * f_space) / (1 + net_w), f_empty)
    space = (empty & ~cv2.dilate(walls, k3).astype(bool)).astype(np.uint8)
    space = cv2.morphologyEx(space, cv2.MORPH_OPEN, k3)
    seeds = room_seeds(space, P(room_core_m), P(seed_m), min_area_px=0.3 / (px * px))
    keep = [int(v) for v in np.unique(seeds) if v > 0]
    nl = 1 + len(keep)
    log(f"  затравок помещений {len(keep)}")
    seed_frac = np.zeros((nc, nl))
    allowed = np.zeros((nc, nl), bool)
    allowed[:, OUT] = True
    reach_k = np.ones((2 * P(1.5) + 1,) * 2, np.uint8)
    for li, s in enumerate(keep, 1):
        m = seeds == s
        seed_frac[:, li] = np.bincount(cells[m], minlength=nc) / np.maximum(area, 1)
        reach = cv2.dilate(m.astype(np.uint8), reach_k).astype(bool)
        allowed[np.unique(cells[reach]), li] = True
    allowed[f_empty < 0.02, 1:] = False                  # ни пола, ни потолка, ни луча - снаружи
    allowed[0, 1:] = False                               # 0 - пиксели линий, не ячейка
    A_m2 = area * px * px
    D = np.zeros((nc, nl))
    D[:, OUT] = A_m2 * 1.5 * f_room
    seed_any = seed_frac[:, 1:].sum(1)
    for li in range(1, nl):
        # штраф - только за долю чужой затравки: ничья пустота (ниша, полоса у стены) идёт к
        # соседнему помещению по свидетельству пустоты
        D[:, li] = A_m2 * ((1 - f_room) + 0.5 * (seed_any - seed_frac[:, li]))
    D[0] = 0

    # --- профили и стоимости рёбер ---------------------------------------------------------------
    E, prof = [], {}
    can_room = allowed[:, 1:].any(1)
    for (a, b), pix in epix.items():
        if not (can_room[a] or can_room[b]):
            continue
        pr = edge_profile(pix, cells, F, Fmax, P, bin_m, N=N)
        if pr is None:
            continue
        pr["net_w"] = net_w
        pr["bin_m"] = pr["step"] * px                   # фактическая длина бина (шаг округлён до px)
        # сторона +нормали - ячейка a?
        sa = 1 if np.mean(pr["side"][1]["cell"] == a) >= np.mean(pr["side"][-1]["cell"] == a) else -1
        pr["side_a"] = sa
        bm = pr["bin_m"]
        c_same = same_cost(pr, bm)
        c_rr, _ = segment_edge(pr, "RR", None, bm)
        c_roa, _ = segment_edge(pr, "RO", sa, bm)                # снаружи - сторона a
        c_rob, _ = segment_edge(pr, "RO", -sa, bm)
        prof[(a, b)] = pr
        E.append((a, b, c_same, c_rr, c_roa, c_rob))
    log(f"  рёбер в задаче {len(E)}, переменных ячеек {int(allowed.sum())}")

    # --- шаг 3: совместная разметка ----------------------------------------------------------------
    labels = joint_labels(D, allowed, E)
    lab_px = labels[cells].astype(np.int32)
    # пиксели линий разбиения - метка соседей (линия - и есть грань)
    lab_px[lines] = cv2.dilate(np.where(lines, 0, lab_px).astype(np.float32), k3)[lines].astype(np.int32)
    # помещение связно: оторванный кусок метки - к соседнему помещению с самой длинной общей
    # границей (или наружу, если соседей-помещений нет)
    for li in range(1, nl):
        m = (lab_px == li).astype(np.uint8)
        if not m.any():
            continue
        n_m, cl = cv2.connectedComponents(m, connectivity=4)
        if n_m <= 2:
            continue
        sizes = np.bincount(cl.ravel(), minlength=n_m)
        sizes[0] = 0
        main = int(sizes.argmax())
        for c in range(1, n_m):
            if c == main:
                continue
            piece = cl == c
            # соседи - в кольце 15 см: между кусками бывают тонкие полосы «снаружи»
            ring = (cv2.dilate(piece.astype(np.uint8), np.ones((2 * P(0.15) + 1,) * 2, np.uint8)) > 0) & ~piece
            nb = lab_px[ring]
            nb = nb[(nb > 0) & (nb != li)]
            lab_px[piece] = int(np.bincount(nb).argmax()) if len(nb) else OUT
    # помещение - до видимой грани: дорастает до 10 см через незанятое «снаружи» (тонкая
    # полоса между почти совпадающими кандидатами - не стена)
    blocked = cv2.dilate(walls, np.ones((1, 1), np.uint8)).astype(bool)
    for _ in range(P(0.1)):
        grow = cv2.dilate(lab_px.astype(np.float32), k3).astype(np.int32)
        tgt = (lab_px == OUT) & ~blocked & (grow > 0)
        if not tgt.any():
            break
        lab_px[tgt] = grow[tgt]

    # --- помещения --------------------------------------------------------------------------------
    rooms, room_of_label = [], {}
    for li in range(1, nl):
        m = (lab_px == li).astype(np.uint8)
        if m.sum() * px * px < min_room_m2:
            continue
        n_m, cl = cv2.connectedComponents(m, connectivity=4)
        big = max(range(1, n_m), key=lambda c: (cl == c).sum())
        m = (cl == big).astype(np.uint8)
        poly, prims = _room_outline(m, P, max(tol_m / px, 0.6))
        dm = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        y0, x0 = np.unravel_index(dm.argmax(), dm.shape)
        room_of_label[li] = len(rooms)
        rooms.append({"mask": m, "poly": poly, "prims": prims, "label": (x0, y0), "li": li,
                      "area": abs(cv2.contourArea(poly.astype(np.float32))) * px * px})
    order = sorted(range(len(rooms)), key=lambda i: -rooms[i]["area"])
    rank = {old: new for new, old in enumerate(order)}
    rooms = [rooms[i] for i in order]
    room_of_label = {li: rank[i] for li, i in room_of_label.items()}

    # снаружи: связная с краем растра часть - улица; остальное «снаружи» - толща стен
    outside = (lab_px == OUT).astype(np.uint8)
    n_o, ocl = cv2.connectedComponents(outside, connectivity=4)
    border = set(np.unique(np.r_[ocl[0], ocl[-1], ocl[:, 0], ocl[:, -1]])) - {0}
    exterior = np.isin(ocl, list(border))

    # --- стены и проёмы: рёбра между разными метками ----------------------------------------------
    walls_out, ops_all, elev = [], [], []
    for (a, b, cs, crr, croa, crob) in E:
        la, lb = labels[a], labels[b]
        if la == lb:
            continue
        pr = prof[(a, b)]
        sa = pr["side_a"]
        if la != OUT and lb != OUT:
            ctx, out_side = "RR", None
        else:
            ctx, out_side = "RO", (sa if la == OUT else -sa)
        bm = pr["bin_m"]
        _, segs = segment_edge(pr, ctx, out_side, bm)
        p0 = pr["m"] + pr["d"] * pr["t0"]
        p1 = pr["m"] + pr["d"] * pr["t1"]
        rooms_here = [room_of_label[x] for x in (la, lb) if x in room_of_label]
        if not rooms_here:
            continue
        outer = False
        if ctx == "RO":
            q = np.round(pr["m"] + out_side * pr["n"] * P(0.3)).astype(int)
            outer = bool(exterior[np.clip(q[1], 0, H - 1), np.clip(q[0], 0, W - 1)])
        wi = len(walls_out)
        walls_out.append({"a": p0, "b": p1, "rooms": rooms_here, "ctx": ctx, "outer": outer, "segs": segs,
                          "pr": pr, "out_side": out_side, "key": (a, b)})
        # развёртка по срезам: занятость в ±2 px от ребра
        ys, xs = epix[(a, b)]
        tt = (np.c_[xs, ys] - pr["m"]) @ pr["d"]
        kk = np.clip(((tt - pr["t0"]) / pr["step"]).astype(int), 0, pr["nb"] - 1)
        occ_d = np.stack([cv2.dilate(o.view(np.uint8), k5)[ys, xs] for o in occ], 1).astype(bool)
        Ev = np.zeros((pr["nb"], len(zc)), bool)
        np.logical_or.at(Ev, kk, occ_d)
        ops = []
        for kind, s0, s1 in segs:
            if kind in ("wall", "shadow"):
                continue
            prof_z = Ev[s0:s1].mean(0) > 0.3
            below = np.flatnonzero(prof_z & (zc < 1.2))
            above = np.flatnonzero(prof_z & (zc >= 1.6))
            sill = float(zc[below].max() + 0.05) if len(below) and kind == "window" else None
            head = float(zc[above].min() - 0.05) if len(above) and kind != "gap" else None
            pa = pr["m"] + pr["d"] * (pr["t0"] + s0 * pr["step"])
            pb = pr["m"] + pr["d"] * (pr["t0"] + s1 * pr["step"])
            o = {"type": kind, "wall": wi, "a": pa, "b": pb, "width": (s1 - s0) * bm, "sill": sill,
                 "head": head, "s": (s0 * bm, s1 * bm), "n": pr["n"] * (out_side or 1)}
            ops.append(o)
            ops_all.append(o)
        elev.append((wi, np.repeat(Ev, max(1, int(round(pr["step"]))), axis=0), ops))

    # один проём на двух гранях одной стены - оставить один
    kept = []
    for o in sorted(ops_all, key=lambda o: -o["width"]):
        c = (o["a"] + o["b"]) / 2
        if all(np.linalg.norm(c - (q["a"] + q["b"]) / 2) > P(0.5) for q in kept):
            kept.append(o)

    # --- в метры --------------------------------------------------------------------------------------
    def to_m(xy):
        return [round(float(frame["origin"][0] + xy[0] * px), 3),
                round(float(frame["origin"][1] + (H - 1 - xy[1]) * px), 3)]

    plan = LevelPlan(k, L["level"]["floor_z"], L["level"]["ceiling_z"])
    room_id = [f"L{k}-R{i + 1}" for i in range(len(rooms))]
    for i, R in enumerate(rooms):
        per = float(np.sum(np.linalg.norm(np.diff(np.vstack([R["poly"], R["poly"][:1]]), axis=0), axis=1))) * px
        plan.rooms.append(Room(room_id[i], f"Помещение {i + 1}", [to_m(p) for p in R["poly"]], round(R["area"], 2),
                               round(per, 2), tuple(to_m(R["label"])),
                               [{"kind": q["kind"], "a": to_m(q["a"]), "b": to_m(q["b"]),
                                 **({"center": to_m(q["c"]), "radius": round(q["r"] * px, 3)}
                                    if q["kind"] == "arc" else {})} for q in R["prims"]]))
    wall_id = {}
    for wi, w in enumerate(walls_out):
        if np.linalg.norm(w["b"] - w["a"]) < P(0.1) and not any(o["wall"] == wi for o in kept):
            continue                                     # обрезок на стыке - не стена
        wall_id[wi] = f"L{k}-W{len(plan.walls) + 1}"
        plan.walls.append(WallSegment(wall_id[wi], [to_m(w["a"]), to_m(w["b"])], [room_id[r] for r in w["rooms"]],
                                      None, w["outer"], round(float(np.linalg.norm(w["b"] - w["a"])) * px, 3)))
    for i, o in enumerate(kept):
        dep = P(0.15)
        n = o["n"]
        quad = np.array([o["a"] - n * dep / 2, o["b"] - n * dep / 2, o["b"] + n * dep / 2, o["a"] + n * dep / 2])
        plan.openings.append(Opening(f"L{k}-O{i + 1}", o["type"], wall_id[o["wall"]], to_m(o["a"]), to_m(o["b"]),
                                     [to_m(p) for p in quad], round(o["width"], 3),
                                     None if o["sill"] is None else round(o["sill"], 2),
                                     None if o["head"] is None else round(o["head"], 2)))
    # толща стен между двумя помещениями - как видна
    any_room = np.zeros((H, W), np.uint8)
    nearc = np.zeros((H, W), np.int32)
    kn = np.ones((2 * P(0.5) + 1,) * 2, np.uint8)
    for R in rooms:
        any_room |= R["mask"]
        nearc += cv2.dilate(R["mask"], kn)
    fill = ((nearc >= 2) & (any_room == 0)).astype(np.uint8)
    cnts, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plan.slab = [[to_m(p) for p in c[:, 0, :].astype(float)] for c in cnts if cv2.contourArea(c) * px * px >= 0.005]
    items = [(wall_id[wi], Ev, ops) for wi, Ev, ops in elev if wi in wall_id]
    ifc = ifc_plan(plan, walls_out, wall_id, kept, epix, walls, P, px, to_m, room_id, frame)
    dbg = {"cells": cells, "lines": lines, "labels": lab_px, "polys": polys, "vis": vis, "free": empty,
           "stations": stations_px, "seeds": seeds, "elevations": items, "features": F, "walls_out": walls_out,
           "ifc_plan": ifc, "net": net is not None}
    return plan, dbg


# ----------------------------------------------------------------------
# Оси и толщины стен для IFC
# ----------------------------------------------------------------------

def _run(mask: np.ndarray, c: np.ndarray, o: np.ndarray, max_px: int, gap_px: int) -> int:
    """Пикселей маски подряд от c по направлению o (начало - не дальше gap_px)."""
    H, W = mask.shape
    n, started = 0, False
    for s in range(max_px):
        x, y = np.round(c + o * s).astype(int)
        if not (0 <= x < W and 0 <= y < H):
            break
        if mask[y, x]:
            started = True
            n += 1
        elif started or s > gap_px:
            break
    return n


def _axis_points(pix, pr: dict, offset: np.ndarray, P) -> np.ndarray:
    """Ось стены: хорда ребра, сдвинутая на offset; ребро по дуге (отклонение > 3 см) -
    ломаная по пикселям ребра со сдвигом по местной нормали."""
    import cv2

    ys, xs = pix
    p = np.c_[xs, ys].astype(float)
    t = (p - pr["m"]) @ pr["d"]
    dev = np.abs((p - pr["m"]) @ pr["n"])
    a, b = pr["m"] + pr["d"] * pr["t0"], pr["m"] + pr["d"] * pr["t1"]
    if dev.max() <= P(0.03) or len(p) < 8:
        return np.array([a + offset, b + offset])
    step = max(1, P(0.05))
    k = np.floor((t - t.min()) / step).astype(int)
    cnt = np.bincount(k)
    ok = cnt > 0
    q = np.c_[np.bincount(k, p[:, 0])[ok] / cnt[ok], np.bincount(k, p[:, 1])[ok] / cnt[ok]]
    q = cv2.approxPolyDP(q.astype(np.float32)[:, None], max(0.5, P(0.01)), False)[:, 0].astype(float)
    off_len = float(np.linalg.norm(offset))
    if off_len < 1e-9:
        return q
    sgn = offset / off_len
    tang = np.gradient(q, axis=0)
    nrm = np.c_[-tang[:, 1], tang[:, 0]]
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    nrm *= np.sign(nrm @ sgn)[:, None]
    return q + nrm * off_len


def _merge_axes(items: list, P, gap_m: float = 0.3) -> list:
    """Две грани одной стены (видны из двух помещений) дают почти одну ось - одна стена; так же
    соосные участки с зазором до gap_m (наружная стена, разрезанная примыкающей перегородкой).
    Уже проёма зазор не бывает, поэтому настоящий проход так не закрывается."""
    alive = [True] * len(items)
    changed = True
    while changed:
        changed = False
        for i, A in enumerate(items):
            if not alive[i] or len(A["pts"]) != 2:
                continue
            a0, a1 = A["pts"]
            u = (a1 - a0) / max(np.linalg.norm(a1 - a0), 1e-9)
            nrm = np.array([-u[1], u[0]])
            for j in range(i + 1, len(items)):
                B = items[j]
                if not alive[j] or len(B["pts"]) != 2:
                    continue
                b0, b1 = B["pts"]
                v = (b1 - b0) / max(np.linalg.norm(b1 - b0), 1e-9)
                if abs(u[0] * v[1] - u[1] * v[0]) > 0.05:
                    continue
                off = ((b0 + b1) / 2 - a0) @ nrm
                if abs(off) > max(P(0.04), 0.25 * max(A["t"], B["t"])):
                    continue
                sa = sorted([0.0, (a1 - a0) @ u])
                sb = sorted([(b0 - a0) @ u, (b1 - a0) @ u])
                if min(sa[1], sb[1]) - max(sa[0], sb[0]) < -P(gap_m) or A["outer"] != B["outer"] and \
                        min(sa[1], sb[1]) - max(sa[0], sb[0]) <= 0:
                    continue
                la, lb = sa[1] - sa[0], sb[1] - sb[0]
                base = a0 + nrm * off * lb / (la + lb)
                lo, hi = min(sa[0], sb[0]), max(sa[1], sb[1])
                A["pts"] = np.array([base + u * lo, base + u * hi])
                A["t"] = (A["t"] * la + B["t"] * lb) / (la + lb)
                A["outer"] = A["outer"] or B["outer"]
                A["rooms"] = sorted(set(A["rooms"]) | set(B["rooms"]))
                A["src"] += B["src"]
                alive[j] = False
                changed = True
                a0, a1 = A["pts"]
    return [x for x, ok in zip(items, alive) if ok]


def _join_axes(items: list, P, tol_m: float = 0.05) -> None:
    """Стыки: конец прямой стены, лежащий у оси другой (не параллельной) стены не дальше её
    полутолщины и своей толщины, продлевается до пересечения осей (L- и T-стыки)."""
    straight = [x for x in items if len(x["pts"]) == 2]
    for A in straight:
        for e in (0, 1):
            p, q = A["pts"][e], A["pts"][1 - e]
            u = (p - q) / max(np.linalg.norm(p - q), 1e-9)
            best = None
            for B in straight:
                if B is A:
                    continue
                b0, b1 = B["pts"]
                Lb = np.linalg.norm(b1 - b0)
                v = (b1 - b0) / max(Lb, 1e-9)
                den = u[0] * v[1] - u[1] * v[0]
                if abs(den) < 0.3:                            # почти параллельны
                    continue
                r = b0 - p
                s_a = (r[0] * v[1] - r[1] * v[0]) / den           # p + s_a u = b0 + s_b v
                s_b = (r[0] * u[1] - r[1] * u[0]) / den
                reach = max(A["t"], B["t"]) + P(tol_m)
                if not (-reach <= s_a <= reach and -reach <= s_b <= Lb + reach):
                    continue
                if best is None or abs(s_a) < abs(best):
                    best = s_a
            if best is not None:
                A["pts"][e] = p + u * best


def ifc_plan(plan, walls_out: list, wall_id: dict, kept: list, epix: dict, wallmask: np.ndarray, P, px: float,
             to_m, room_id: list, frame: dict, t_min: float = 0.05, t_max: float = 0.6,
             t_default: float = 0.1) -> dict:
    """Уровень для export-ifc: стены - ось и толщина (по маске стен поперёк грани, от
    помещения наружу; между двумя помещениями - в обе стороны), две грани одной стены -
    одна ось; проёмы - на своих стенах; помещения как есть. Координаты - СК плана."""
    items = []
    has_op = {o["wall"] for o in kept}
    for wi, w in enumerate(walls_out):
        if wi not in wall_id:
            continue
        pr = w["pr"]
        if (pr["t1"] - pr["t0"]) * px < 0.15 and wi not in has_op:
            continue                                     # торец стены или обрезок на стыке
        centers = pr["m"] + pr["d"] * (pr["t0"] + (np.arange(pr["nb"]) + 0.5) * pr["step"])[:, None]
        if w["ctx"] == "RO":
            o = pr["n"] * w["out_side"]
            runs = np.array([_run(wallmask, c, o, P(t_max), P(0.05)) for c in centers])
            runs = runs[runs > 0]
            t = float(np.median(runs)) if len(runs) else t_default / px
            t = float(np.clip(t * px, t_min, t_max)) / px
            offset = o * t / 2
        else:
            rp = np.array([_run(wallmask, c, pr["n"], P(t_max), 0) for c in centers])
            rm = np.array([_run(wallmask, c, -pr["n"], P(t_max), 0) for c in centers])
            both = (rp + rm) > 0
            t = float(np.median(rp[both] + rm[both] - 1)) if both.any() else t_default / px
            t = float(np.clip(t * px, t_min, t_max)) / px
            offset = pr["n"] * (float(np.median(rp[both] - rm[both])) / 2 if both.any() else 0.0)
        items.append({"pts": _axis_points(epix[w["key"]], pr, offset, P), "t": t, "outer": w["outer"],
                      "rooms": [room_id[r] for r in w["rooms"]], "src": [wi]})
    items = _merge_axes(items, P)
    _join_axes(items, P)
    lvl = f"L{plan.level}"
    walls, owner = [], {}
    for i, it in enumerate(items):
        wid = f"{lvl}-W{i + 1}"
        for wi in it["src"]:
            owner[wi] = wid
        pts = [to_m(p) for p in it["pts"]]
        walls.append({"id": wid, "points": pts, "thickness": round(it["t"] * px, 3), "is_outer": bool(it["outer"]),
                      "rooms": it["rooms"],
                      "length": round(float(sum(math.dist(a, b) for a, b in zip(pts, pts[1:]))), 3)})
    openings = []
    for i, o in enumerate(kept):
        if o["wall"] not in owner:
            continue
        openings.append({"id": f"{lvl}-O{i + 1}", "opening_type": "opening" if o["type"] == "gap" else o["type"],
                         "wall_id": owner[o["wall"]], "a": to_m(o["a"]), "b": to_m(o["b"]),
                         "width": round(o["width"], 3),
                         "sill": None if o["sill"] is None else round(o["sill"], 2),
                         "head": None if o["head"] is None else round(o["head"], 2)})
    rooms = [{"id": r.id, "name": r.name, "points": r.points, "area": r.area, "perimeter": r.perimeter}
             for r in plan.rooms]
    return {"level": lvl, "floor_z": plan.floor_z, "ceiling_z": plan.ceiling_z, "walls": walls,
            "openings": openings, "rooms": rooms,
            "frame": {"cs": "plan = Rz(-rotation_deg) * world (x, y)", "rotation_deg": frame["rotation_deg"]},
            "metadata": {"method": "cellplan"}}


def _plot_cells(dbg, path, px):
    from matplotlib.colors import ListedColormap

    from .debug import INK, ORANGE, _plt
    from .floorplan import ROOM_TINTS
    from .wallheat import _crop

    plt = _plt()
    lab = dbg["labels"]
    sl = _crop(((lab > 0) | dbg["free"]).astype(float), pad=40)
    oy, ox = sl[0].start or 0, sl[1].start or 0
    Hc, Wc = lab[sl].shape
    fig, axs = plt.subplots(1, 3, figsize=(21, 7 * Hc / max(Wc, 1) + 0.6))
    ax = axs[2]
    ev = np.zeros((Hc, Wc, 3))
    ev[..., 0] = dbg["free"][sl]
    ev[..., 1] = dbg["vis"][sl]
    ev[..., 2] = dbg["seeds"][sl] > 0
    ax.imshow(1 - 0.6 * ev, interpolation="nearest")
    ax.set_title("Пустота: пол/потолок (голубое), лучи (сиреневое), затравки (жёлтое)", loc="left",
                 fontsize=10, color=INK)
    cmap = ListedColormap(["#f2f1ec"] + list(ROOM_TINTS) * 4)
    ax = axs[0]
    rng = np.random.default_rng(1)
    for a in axs[:2]:
        a.set_xlim(-0.5, Wc - 0.5)
        a.set_ylim(Hc - 0.5, -0.5)
    shuffle = rng.permutation(dbg["cells"].max() + 1)
    ax.imshow(shuffle[dbg["cells"][sl]] % 20, cmap="tab20", interpolation="nearest")
    ax.set_title(f"Разбиение: ячеек {dbg['cells'].max()}", loc="left", fontsize=10, color=INK)
    ax = axs[1]
    ax.imshow(np.where(lab[sl] > 0, (lab[sl] - 1) % (len(ROOM_TINTS) * 4) + 1, 0), cmap=cmap,
              vmin=0, vmax=len(ROOM_TINTS) * 4, interpolation="nearest")
    for p in dbg["polys"]:
        ax.plot(p[:, 0] - ox, p[:, 1] - oy, color=INK, lw=0.6)
    st = dbg["stations"]
    if len(st):
        ax.scatter(st[:, 0] - ox, st[:, 1] - oy, s=18, color=ORANGE, zorder=5, label="стоянки")
        ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Разметка ячеек: помещения и «снаружи»; линии - кандидаты стен", loc="left", fontsize=10,
                 color=INK)
    for a in axs:
        a.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def check_against_heat(dbg: dict, heat_walls: np.ndarray, px: float, tol_m: float = 0.03) -> dict:
    """Сверка плана с тепловой картой: грани плана (границы помещений) против граней карты
    (стена или дверь по разметке wallheat), допуск tol_m. -> маски и точность / полнота."""
    import cv2

    lab = dbg["labels"]
    k3 = np.ones((3, 3), np.uint8)
    pred = np.zeros(lab.shape, bool)
    for li in np.unique(lab[lab > 0]):
        m = cv2.morphologyEx((lab == li).astype(np.uint8), cv2.MORPH_CLOSE, k3)
        pred |= (cv2.dilate(m, k3) - m).astype(bool)              # пиксель снаружи у грани помещения
    tol = max(1, int(round(tol_m / px)))
    kt = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    hw = heat_walls.astype(bool)
    # грань карты - пиксель стены, обращённый к пустоте (пол, потолок, лучи): толща стены и
    # наружная сторона наружной стены в сверку не идут - их сканер не видит
    face = hw & (cv2.dilate((dbg["free"] & ~hw).astype(np.uint8), k3) > 0)
    near_room = cv2.dilate((lab > 0).astype(np.uint8), np.ones((2 * int(round(0.5 / px)) + 1,) * 2, np.uint8)) > 0
    hw_rel = face & near_room                                   # грани у помещений (не улица)
    hit = hw_rel & (cv2.dilate(pred.astype(np.uint8), kt) > 0)
    missed = hw_rel & ~hit
    support = pred & (cv2.dilate(hw.astype(np.uint8), kt) > 0)
    invented = pred & ~support
    return {"pred": pred, "hit": hit, "missed": missed, "invented": invented,
            "recall": float(hit.sum() / max(hw_rel.sum(), 1)), "precision": float(support.sum() / max(pred.sum(), 1))}


def _plot_check(dbg: dict, heat: np.ndarray, chk: dict, plan, frame: dict, path):
    import cv2
    from matplotlib.patches import Patch

    from .debug import AQUA, INK, ORANGE, _plt
    from .wallheat import _crop

    plt = _plt()
    px, (H, W) = frame["px"], frame["shape"]
    sl = _crop(((dbg["labels"] > 0) | (heat > 0.3)).astype(float), pad=30)
    oy, ox = sl[0].start or 0, sl[1].start or 0
    h = heat[sl]
    Hc, Wc = h.shape
    img = np.ones((Hc, Wc, 3))
    img *= (1 - 0.35 * np.clip(h, 0, 1))[..., None]             # карта - серым
    img[dbg["labels"][sl] > 0] *= np.array([0.93, 0.96, 1.0])  # помещения - чуть голубым
    thick = max(Hc, Wc) > 1500
    for key, col in (("hit", (0.05, 0.05, 0.05)), ("missed", (0.85, 0.1, 0.1)), ("invented", (0.15, 0.35, 0.95))):
        m = chk[key][sl]
        if thick:
            m = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        img[m] = col
    fig = plt.figure(figsize=(14 * Wc / max(Hc, Wc) + 0.5, 14 * Hc / max(Hc, Wc) + 0.8))
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.88])
    ax.imshow(img, interpolation="nearest")

    def to_px(xy):
        return ((xy[0] - frame["origin"][0]) / px - ox, H - 1 - (xy[1] - frame["origin"][1]) / px - oy)

    for o in plan.openings:
        (ja, ia), (jb, ib) = to_px(o.a), to_px(o.b)
        col = {"door": ORANGE, "window": AQUA, "gap": "#9a63c7"}[o.opening_type]
        ax.plot([ja, jb], [ia, ib], color=col, lw=4, solid_capstyle="butt")
    ax.set_title(f"Сверка с тепловой картой: стены - полнота {chk['recall']:.0%}, точность {chk['precision']:.0%}",
                 loc="left", fontsize=11, color=INK)
    ax.axis("off")
    fig.legend(handles=[Patch(color=(0.05, 0.05, 0.05), label="совпало"),
                        Patch(color=(0.85, 0.1, 0.1), label="грань на карте, нет в плане"),
                        Patch(color=(0.15, 0.35, 0.95), label="стена в плане, нет на карте"),
                        Patch(color=ORANGE, label="дверь"), Patch(color=AQUA, label="окно"),
                        Patch(color="#9a63c7", label="проём")],
               loc="lower left", ncol=6, frameon=False, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _plot_features(F: dict, path):
    from .debug import INK, _plt
    from .wallheat import _crop

    plt = _plt()
    sl = _crop(np.maximum(F["low"], F["floor"]), pad=30)
    names = {"low": "низ < 0,9 м", "mid": "середина 0,9-2,0", "top": "верх >= 2,0", "floor": "виден пол",
             "ceil": "виден потолок", "density": "плотность", "speckle": "крапчатость"}
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    for ax, n in zip(axs.flat, FEATURES):
        ax.imshow(F[n][sl], cmap="magma_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(names[n], loc="left", fontsize=10, color=INK)
        ax.axis("off")
    axs.flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def cellplan(path, out_dir, px: float = 0.01, pred=None, net_w: float = 1.0, log=print) -> dict:
    """pred - каталог infer для этого файла (уровни с prob.npz и scene.json) или None."""
    from dataclasses import asdict

    from .floorplan import render, render_elevations
    from .wallheat import analyse_slices, slice_counts, to_raster

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, frame = slice_counts(path, px=px, log=log)
    st_world = image_stations(path)
    log(f"стоянок по снимкам: {len(st_world)}")
    result = {"file": Path(path).name, "rotation_deg": round(frame["rotation_deg"], 3),
              "transform": "plan = Rz(-rotation_deg) * world (x, y); z - над полом уровня", "levels": []}
    for k, L in enumerate(data, 1):
        A = analyse_slices(L, frame["px"])
        lv = L["level"]
        on = (st_world[:, 2] > lv["floor_z"]) & (st_world[:, 2] < lv["ceiling_z"]) if len(st_world) else []
        st_px = to_raster(st_world[on], frame) if len(st_world) and np.any(on) else np.zeros((0, 2))
        log(f"уровень {k}: стоянок {len(st_px)}")
        net = load_net(pred, lv, frame) if pred is not None else None
        if pred is not None:
            log(f"  сеть: {net['dir'] if net else 'нет уровня с этой отметкой пола - только правила'}")
        plan, dbg = analyse_level(L, A, frame, st_px, k, net=net, net_w=net_w, log=log)
        sfx = "" if len(data) == 1 else f"_L{k}"

        render(plan, None, {}, out / f"plan{sfx}.png", title=f"Уровень {k}" if len(data) > 1 else "")
        render_elevations(dbg["elevations"], frame["px"], L["slice_m"], out / f"walls{sfx}.png")
        _plot_cells(dbg, out / f"cells{sfx}.png", frame["px"])
        _plot_features(dbg["features"], out / f"features{sfx}.png")
        from .wallheat import DOOR, WALL
        chk = check_against_heat(dbg, (A["lab"] == WALL) | (A["lab"] == DOOR), frame["px"])
        _plot_check(dbg, A["bands"]["all"], chk, plan, frame, out / f"check{sfx}.png")
        log(f"  сверка с картой: полнота стен {chk['recall']:.0%}, точность {chk['precision']:.0%}")
        (out / f"plan_L{k}.json").write_text(json.dumps(dbg["ifc_plan"], indent=1, ensure_ascii=False, default=float),
                                             encoding="utf-8")
        np.savez_compressed(out / f"features{sfx}.npz", **{n: dbg["features"][n].astype(np.float16) for n in FEATURES})
        result["levels"].append({**asdict(plan), "net": dbg["net"],
                                 "check": {"wall_recall": round(chk["recall"], 3),
                                           "wall_precision": round(chk["precision"], 3)}})
        log(f"уровень {k}: помещений {len(plan.rooms)}, стен {len(plan.walls)}, проёмов {len(plan.openings)}, "
            f"{sum(r.area for r in plan.rooms):.1f} м²")
    (out / "elements.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float),
                                       encoding="utf-8")
    return result
