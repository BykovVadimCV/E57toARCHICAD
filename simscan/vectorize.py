"""Векторизация предсказаний сети: вероятности классов -> стены (оси, толщины, прямые и дуги),
проёмы, помещения. Только по выходу сети (prob.npz, orient.npy, heights.npy, scene.json) -
без облака; с облаком лучше cellplan.py, он берёт эти же вероятности.

    python -m simscan vectorize pred/kvartira/L1 --out vectors/L1
    -> plan.json (для export-ifc), plan.svg, plan.png

Шаги (пиксели растра сети, в конце - мировые метры через scene.json pixel_to_world):
  1. конструкция = стена + дверь + окно + проём (проёмы не рвут стену), порог с гистерезисом
     по сумме вероятностей: слабые пиксели остаются, если связаны с уверенными; мелочь - вон;
  2. скелет (Zhang-Suen) -> граф: узлы - стыки и концы, рёбра - цепочки; короткие отростки у
     стыков (артефакт утончения толстых стен) отрезаются;
  3. каждое ребро -> прямые и дуги рекурсивным разбиением: прямая, если все точки ближе tol,
     иначе дуга (окружность Касы, R 1,5-40 м), иначе разрез в самой дальней от хорды точке.
     Толщина - по карте расстояний маски на оси; направление короткой прямой - по голове
     ориентации сети (длинной - по точкам);
  4. направления прямых - кластеры по 1° по длинным стенам (от 1 м), короткая прямая
     притягивается к ближайшему направлению ближе 5° (walls.regularize): дрожание уходит,
     85-95° остаётся;
  5. соосные прямые - одна стена; концы - до пересечения осей (L, T, дуги); оборванный конец
     (пропуск сети) дотягивается до GAP_M (walls.merge_collinear, walls.join);
     Чистка: висящий конец ещё раз тянется к соседней стене; вон - одиночный кусок короче
     1 м (пятно шума), огрызок с висящим концом короче 0,5 м, перемычка короче 25 см между
     двумя стенами (остаток утончения на стыке), короткая тень вдоль длинной стены (walls.tidy);
  6. проёмы: компоненты вероятности двери / окна / проёма (гистерезис), хозяин - ближайшая
     стена; концы - проекция на её ось; низ и верх - медиана головы высот. Проём через стык
     двух соединённых стен - они одна стена-ломаная, излом сохраняется (chain_at_openings);
  7. помещения: где они - по сети (что сеть считает помещением, за вычетом стен; сужения уже
     40 см рвутся - комнаты не сливаются), где их грани - по векторным стенам: область
     дорастает в обход тел стен до их граней, контур - стороны на гранях стен, углы - их
     пересечения (как у cellplan). Не замкнутое стенами (площадка за входной дверью) - вон.
  Наружная стена - помещение с одной стороны, за другой по сети улица. Рисунок (plan.png,
  plan.svg) - тот же, что у cellplan (floorplan.render).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .walls import _unit, join, merge_collinear, regularize, tidy

STRUCT = (2, 3, 4, 5)
INTERIOR = (1, 6, 7)                 # помещение, мебель, проём перекрытия
OPEN_KIND = {3: "door", 4: "window", 5: "opening"}
OPEN_MIN_M = {"door": 0.55, "window": 0.3, "opening": 0.5}   # уже - обрывок, не проём

P_HI, P_LO = 0.5, 0.3                # гистерезис конструкции
MIN_WALL_M = 0.15                    # прямая короче - не стена
MIN_SPLIT_M = 0.25                   # кусок короче не дробится (шум скелета - не излом)
SMOOTH_PX = 5                        # окно сглаживания цепочки перед подгонкой
TOL_PX = 1.5                         # допуск ломаной от точек скелета
ARC_R = (1.5, 40.0)                  # радиусы дуг, м
ARC_MIN_CHORD_M = 1.0                # дуга: хорда не короче,
ARC_MIN_SAG_M = 0.08                 # стрела не меньше и окружность в 3 раза точнее прямой
MIN_ROOM_M2 = 1.0
MIN_ENCLOSED = 0.85                   # доля границы помещения на стенах и проёмах
GAP_M = 0.5                          # закрываемый разрыв стены
GAP_MAX_M = 0.8                      # и до стольких - если в щели есть хоть слабая конструкция


# ----------------------------------------------------------------------
# Маска и скелет
# ----------------------------------------------------------------------

def struct_mask(prob: np.ndarray, px: float) -> np.ndarray:
    """Конструкция с гистерезисом: слабая компонента остаётся, если в ней есть уверенные
    пиксели или она протяжённая (габарит от 0,6 м - тонкая наружная полоса, у которой
    сеть неуверенна по всей длине); мелочь (меньше ~0,3 м x 3 px) - шум."""
    import cv2

    p = prob[list(STRUCT)].sum(0)
    weak = (p > P_LO).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    strong = np.bincount(lab[p > P_HI], minlength=n) > 0
    long_ = np.hypot(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT]) * px >= 0.6
    keep = (strong | long_) & (st[:, cv2.CC_STAT_AREA] >= max(8, int(0.3 / px * 3)))
    keep[0] = False
    m = keep[lab].astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)


def thin(mask: np.ndarray) -> np.ndarray:
    """Скелет Zhang-Suen (numpy, в габарите маски)."""
    ys, xs = np.nonzero(mask)
    out = np.zeros_like(mask, bool)
    if not len(ys):
        return out
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    img = np.pad(mask[y0:y1, x0:x1], 1).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = np.pad(img, 1)
            p2, p3, p4, p5 = P[:-2, 1:-1], P[:-2, 2:], P[1:-1, 2:], P[2:, 2:]
            p6, p7, p8, p9 = P[2:, 1:-1], P[2:, :-2], P[1:-1, :-2], P[:-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9, p2]
            B = sum(nb[:8])
            A = sum(((nb[k] == 0) & (nb[k + 1] == 1)).astype(np.uint8) for k in range(8))
            if step == 0:
                c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            rem = (img == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
            if rem.any():
                img[rem] = 0
                changed = True
        if not changed:
            break
    out[y0:y1, x0:x1] = img[1:-1, 1:-1].astype(bool)
    return out


_NB = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def unstair(sk: np.ndarray) -> np.ndarray:
    """Лишние пиксели «лесенки» наклонной линии: угол, чьи два соседа (по вертикали и по
    горизонтали) сами соседи по диагонали, а с противоположной стороны пусто. После
    Zhang-Suen у них три соседа, и граф принял бы каждую ступеньку за стык."""
    sk = sk.copy()
    H, W = sk.shape
    # (соседи, которые должны быть, и которые должны быть пусты) для четырёх поворотов угла
    pats = [(((-1, 0), (0, 1)), ((1, 0), (0, -1), (1, -1))),
            (((0, 1), (1, 0)), ((-1, 0), (0, -1), (-1, -1))),
            (((1, 0), (0, -1)), ((-1, 0), (0, 1), (-1, 1))),
            (((0, -1), (-1, 0)), ((1, 0), (0, 1), (1, 1)))]
    for on, off in pats:
        P = np.pad(sk, 1)
        sh = lambda dy, dx: P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W]       # noqa: E731
        m = sk.copy()
        for dy, dx in on:
            m &= sh(dy, dx)
        for dy, dx in off:
            m &= ~sh(dy, dx)
        sk &= ~m
    return sk


def _degree(sk: np.ndarray) -> np.ndarray:
    P = np.pad(sk.astype(np.uint8), 1)
    H, W = sk.shape
    return sum(P[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dy, dx in _NB) * sk


def graph(sk: np.ndarray) -> list[dict]:
    """Скелет -> рёбра {"pix": [(i, j)...], "ends": (узел или -1, ...)}. Узел - связная группа
    пикселей степени >= 3 или конец (степень 1); замкнутая цепочка без узлов - ребро-кольцо."""
    import cv2

    deg = _degree(sk)
    node_px = sk & ((deg >= 3) | (deg == 1))
    _, nid = cv2.connectedComponents(node_px.astype(np.uint8), connectivity=8)
    H, W = sk.shape
    chain = sk & ~node_px
    seen = np.zeros_like(sk, bool)
    edges = []

    def nbrs(i, j):
        for dy, dx in _NB:
            y, x = i + dy, j + dx
            if 0 <= y < H and 0 <= x < W and sk[y, x]:
                yield y, x

    def walk(start, first):
        path = [start, first]
        seen[first] = True
        prev, cur = start, first
        while True:
            nxt = None
            for q in nbrs(*cur):
                if q == prev or (len(path) > 2 and q == path[-3]):
                    continue
                if node_px[q]:
                    if nid[q] != nid[start] or len(path) > 3 or not node_px[start]:
                        path.append(q)
                        return path
                    continue
                if not seen[q]:
                    nxt = q
                    break
            if nxt is None:
                return path
            seen[nxt] = True
            path.append(nxt)
            prev, cur = cur, nxt

    for i, j in zip(*np.nonzero(node_px)):
        for q in nbrs(i, j):
            if chain[q] and not seen[q]:
                p = walk((i, j), q)
                end = p[-1]
                edges.append({"pix": p, "ends": (int(nid[i, j]), int(nid[end]) if node_px[end] else -1)})
            elif node_px[q] and nid[q] != nid[i, j] and (i, j) < q:
                edges.append({"pix": [(i, j), q], "ends": (int(nid[i, j]), int(nid[q]))})
    for i, j in zip(*np.nonzero(chain & ~seen)):           # кольца без узлов
        if seen[i, j]:
            continue
        seen[i, j] = True
        q = next(nbrs(i, j), None)
        if q is None:
            continue
        p = walk((i, j), q)
        edges.append({"pix": p, "ends": (-1, -1), "ring": True})
    return edges


def prune_spurs(sk: np.ndarray, dt: np.ndarray, rounds: int = 2) -> np.ndarray:
    """Отростки: ребро «стык - конец» короче полуторной толщины в стыке (утончение толстой
    стены оставляет усы к углам)."""
    sk = sk.copy()
    for _ in range(rounds):
        deg = _degree(sk)
        removed = False
        for e in graph(sk):
            pix = e["pix"]
            if len(pix) < 2:
                continue
            a, b = pix[0], pix[-1]
            da, db = deg[a], deg[b]
            if (da >= 3 and db == 1) or (db >= 3 and da == 1):
                junc = a if da >= 3 else b
                if len(pix) <= max(3.0, 1.5 * 2 * dt[junc]):
                    for q in pix:
                        if q != junc:
                            sk[q] = False
                    removed = True
        if not removed:
            break
    return unstair(thin(sk))


# ----------------------------------------------------------------------
# Прямые и дуги по цепочке
# ----------------------------------------------------------------------

def _circle(pts: np.ndarray):
    """Окружность Касы: (центр, радиус, СКО отклонения, макс. отклонение)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.c_[x, y, np.ones_like(x)]
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2] / 2
    r = math.sqrt(max(sol[2] + c @ c, 1e-12))
    e = np.abs(np.linalg.norm(pts - c, axis=1) - r)
    return c, r, float(np.sqrt((e * e).mean())), float(e.max())


def _smooth_chain(pts: np.ndarray, w: int) -> np.ndarray:
    """Скользящее среднее вдоль цепочки (концы на месте): дрожание скелета ±1-2 px."""
    if len(pts) <= w or w < 3:
        return pts
    k = np.ones(w) / w
    pad = w // 2
    q = np.vstack([np.repeat(pts[:1], pad, 0), pts, np.repeat(pts[-1:], pad, 0)])
    out = np.c_[np.convolve(q[:, 0], k, "valid"), np.convolve(q[:, 1], k, "valid")]
    out[0], out[-1] = pts[0], pts[-1]
    return out


def fit_chain(pts: np.ndarray, px: float, tol: float = TOL_PX) -> list[dict]:
    """Цепочка точек -> [{"kind": "line"|"arc", "idx": (i0, i1), ...}] рекурсивным разбиением
    (по сглаженной цепочке; кусок короче MIN_SPLIT_M не дробится)."""
    out = []
    pts = _smooth_chain(pts, SMOOTH_PX)
    min_split = MIN_SPLIT_M / px

    def rec(i0, i1):
        q = pts[i0:i1 + 1]
        if len(q) < 2:
            return
        a, b = q[0], q[-1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L > 1e-9:
            n = np.array([-d[1], d[0]]) / L
            dev = np.abs((q - a) @ n)
        else:
            dev = np.linalg.norm(q - a, axis=1)
        if dev.max() <= tol or L < 2 * min_split:
            out.append({"kind": "line", "idx": (i0, i1)})
            return
        if len(q) >= 12 and L * px >= ARC_MIN_CHORD_M:
            c, r, rms, emax = _circle(q)
            sag = r - math.sqrt(max(r * r - (L / 2) ** 2, 0.0))
            m_ = q.mean(0)
            line_rms = float(np.linalg.svd(q - m_, compute_uv=False)[-1] / math.sqrt(len(q)))
            if rms <= tol and emax <= 2.5 * tol and ARC_R[0] / px <= r <= ARC_R[1] / px and \
                    sag * px >= ARC_MIN_SAG_M and line_rms >= 3 * rms:
                out.append({"kind": "arc", "idx": (i0, i1), "c": c, "r": r})
                return
        k = int(dev.argmax())
        if k in (0, len(q) - 1):
            k = len(q) // 2
        rec(i0, i0 + k)
        rec(i0 + k, i1)

    if len(pts) >= 2:
        if np.linalg.norm(pts[0] - pts[-1]) < 2 and len(pts) > 4:      # кольцо: режем в дальней точке
            k = int(np.linalg.norm(pts - pts[0], axis=1).argmax())
            rec(0, k)
            rec(k, len(pts) - 1)
        else:
            rec(0, len(pts) - 1)
    return out


def _arc_poly(c, r, a, b, mid, step):
    a0 = math.atan2(a[1] - c[1], a[0] - c[0])
    a1 = math.atan2(b[1] - c[1], b[0] - c[0])
    am = math.atan2(mid[1] - c[1], mid[0] - c[0])
    da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
    dm = (am - a0 + math.pi) % (2 * math.pi) - math.pi
    if np.sign(dm) != np.sign(da) or abs(dm) > abs(da):
        da -= np.sign(da) * 2 * math.pi
    n = max(2, int(abs(da) * r / step))
    t = a0 + da * np.linspace(0, 1, n + 1)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def walls_from_mask(mask: np.ndarray, px: float, orient: np.ndarray | None = None) -> list[dict]:
    """Маска конструкции -> стены в пикселях (j, i): {"pts", "t", "free", "arc"?}."""
    import cv2

    # край растра - фон (иначе маска во весь растр даёт бесконечные расстояния)
    dt = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    sk = prune_spurs(unstair(thin(mask)), dt)
    deg = _degree(sk)
    items = []
    for e in graph(sk):
        pix = np.array(e["pix"])
        if len(pix) < 2:
            continue
        xy = pix[:, ::-1].astype(float)                       # (j, i)
        for seg in fit_chain(xy, px):
            i0, i1 = seg["idx"]
            q = xy[i0:i1 + 1]
            qp = pix[i0:i1 + 1]
            t = float(np.clip(2 * float(np.median(dt[qp[:, 0], qp[:, 1]])) - 1.0, 2.0, 1.0 / px))   # 4 см - 1 м
            free = (bool(deg[tuple(qp[0])] == 1), bool(deg[tuple(qp[-1])] == 1))
            if seg["kind"] == "arc":
                items.append({"t": t, "free": free, "arc": {"c": seg["c"], "r": seg["r"]}, "fit": q})
                continue
            L = float(np.linalg.norm(q[-1] - q[0]))
            if L * px < MIN_WALL_M and not any(free) and len(q) < 3:
                continue
            m = q.mean(0)
            if len(q) >= 8 and L * px >= 0.6:                  # длинная - по точкам
                _, _, vt = np.linalg.svd(q - m, full_matrices=False)
                d = vt[0]
            elif orient is not None:                          # короткая - по голове ориентации
                o = orient[:, qp[:, 0], qp[:, 1]].mean(1)
                th = math.atan2(o[1], o[0]) / 2
                d = np.array([math.cos(th), -math.sin(th)])   # растр: строки вниз
            else:
                d = (q[-1] - q[0]) / max(L, 1e-9)
            if (q[-1] - q[0]) @ d < 0:
                d = -d
            s = (q - m) @ d
            if s.max() - s.min() < 1.0:                       # вырожденный кусок (точка)
                continue
            items.append({"pts": np.array([m + d * s.min(), m + d * s.max()]), "t": t, "free": free})
    arcs = _merge_arcs([w for w in items if "arc" in w], px)
    for w in arcs:
        q = w.pop("fit")
        w["pts"] = _arc_poly(w["arc"]["c"], w["arc"]["r"], q[0], q[-1], q[len(q) // 2], 0.1 / px)
    return [w for w in items if "arc" not in w] + arcs


def _merge_arcs(arcs: list, px: float) -> list:
    """Соседние дуги одной окружности (ребро скелета разрезано стыком или шумом) - одна,
    с переподгонкой по всем точкам."""
    changed = True
    while changed:
        changed = False
        for i in range(len(arcs)):
            for j in range(i + 1, len(arcs)):
                A, B = arcs[i], arcs[j]
                ra, rb = A["arc"]["r"], B["arc"]["r"]
                if abs(ra - rb) > 0.08 * max(ra, rb) or \
                        np.linalg.norm(A["arc"]["c"] - B["arc"]["c"]) > 0.08 * max(ra, rb):
                    continue
                qa, qb = A["fit"], B["fit"]
                ends = [(np.linalg.norm(qa[x] - qb[y]), x, y) for x in (0, -1) for y in (0, -1)]
                d, x, y = min(ends)
                if d > 0.3 / px:
                    continue
                qa = qa if x == -1 else qa[::-1]
                qb = qb if y == 0 else qb[::-1]
                q = np.vstack([qa, qb])
                c, r, rms, _ = _circle(q)
                if rms > 2 * TOL_PX:
                    continue
                L = (len(qa) * A["t"] + len(qb) * B["t"]) / len(q)
                arcs[i] = {"t": L, "free": (A["free"][0 if x == -1 else 1], B["free"][1 if y == 0 else 0]),
                           "arc": {"c": c, "r": r}, "fit": q}
                arcs.pop(j)
                changed = True
                break
            if changed:
                break
    return arcs


# ----------------------------------------------------------------------
# Проёмы, помещения
# ----------------------------------------------------------------------

def _hysteresis(p: np.ndarray, hi: float = 0.5, lo: float = 0.3, min_px: int = 6):
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats((p > lo).astype(np.uint8), connectivity=8)
    strong = np.bincount(lab[p > hi], minlength=n) > 0
    return [lab == k for k in range(1, n) if strong[k] and st[k, cv2.CC_STAT_AREA] >= min_px]


def _project(w: dict, xy: np.ndarray):
    """Точки -> (параметр вдоль оси, расстояние до оси) для ломаной стены."""
    pts = w["pts"]
    best_d = np.full(len(xy), np.inf)
    best_s = np.zeros(len(xy))
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        t = np.clip((xy - a) @ d / (L * L), 0, 1)
        dist = np.linalg.norm(xy - (a + t[:, None] * d), axis=1)
        m = dist < best_d
        best_d[m], best_s[m] = dist[m], acc + t[m] * L
        acc += L
    return best_s, best_d


def _point_at(w: dict, s: float) -> np.ndarray:
    pts = w["pts"]
    acc = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        L = float(np.linalg.norm(b - a))
        if s <= acc + L or b is pts[-1]:
            return a + (b - a) * (min(max(s - acc, 0.0), L) / max(L, 1e-9))
        acc += L
    return pts[-1]


def chain_at_openings(prob: np.ndarray, items: list, px: float, max_deg: float = 20.0) -> list:
    """Проём сети через стык двух соединённых прямых (излом до max_deg) - одна стена-ломаная
    с этим изломом (стены бывают кривыми - излом не выпрямляется), иначе проём достаётся
    одной из них и обрезается по стыку. Проём должен заходить на обе стороны стыка."""
    tol = 0.05 / px
    cos_max = math.cos(math.radians(max_deg))
    side = 0.1 / px
    for c in OPEN_KIND:
        for comp in _hysteresis(prob[c], min_px=max(6, int((0.25 / px) ** 2 / 4))):
            ys, xs = np.nonzero(comp)
            xy = np.c_[xs, ys].astype(float)
            pair = None
            for i, A in enumerate(items):
                if pair or len(A["pts"]) != 2 or A.get("arc"):
                    continue
                for j, B in enumerate(items):
                    if j <= i or pair or len(B["pts"]) != 2 or B.get("arc"):
                        continue
                    ends = [(ea, eb) for ea in (0, 1) for eb in (0, 1)
                            if np.linalg.norm(A["pts"][ea] - B["pts"][eb]) <= tol + (A["t"] + B["t"]) / 2]
                    if not ends:
                        continue
                    ea, eb = ends[0]
                    u = A["pts"][ea] - A["pts"][1 - ea]
                    v = B["pts"][1 - eb] - B["pts"][eb]
                    if u @ v < cos_max * np.linalg.norm(u) * np.linalg.norm(v):
                        continue
                    J = (A["pts"][ea] + B["pts"][eb]) / 2
                    ua, ub = _unit(u), _unit(v)
                    da, db = _project(A, xy)[1], _project(B, xy)[1]
                    on_a = (da <= A["t"] / 2 + 0.25 / px) & ((xy - J) @ ua < -side)
                    on_b = (db <= B["t"] / 2 + 0.25 / px) & ((xy - J) @ ub > side)
                    if on_a.sum() >= 0.05 * len(xy) and on_b.sum() >= 0.05 * len(xy):
                        pair = (i, j, ea, eb, J)
            if pair:
                i, j, ea, eb, J = pair
                A, B = items[i], items[j]
                la = float(np.linalg.norm(A["pts"][1] - A["pts"][0]))
                lb = float(np.linalg.norm(B["pts"][1] - B["pts"][0]))
                new = dict(A if la >= lb else B)
                new["pts"] = np.array([A["pts"][1 - ea], J, B["pts"][1 - eb]], float)
                new["t"] = (A["t"] * la + B["t"] * lb) / max(la + lb, 1e-9)
                items = [w for k, w in enumerate(items) if k not in (i, j)] + [new]
    return items


def openings(prob: np.ndarray, heights: np.ndarray | None, items: list, px: float, ceil_m: float) -> list[dict]:
    out = []
    for c, kind in OPEN_KIND.items():
        for comp in _hysteresis(prob[c], min_px=max(6, int((0.25 / px) ** 2 / 4))):
            ys, xs = np.nonzero(comp)
            xy = np.c_[xs, ys].astype(float)
            cen = xy.mean(0)
            best = None
            for k, w in enumerate(items):
                _, dist = _project(w, cen[None])
                if dist[0] <= w["t"] / 2 + 0.25 / px and (best is None or dist[0] < best[1]):
                    best = (k, dist[0])
            if best is None:
                continue
            w = items[best[0]]
            s, _ = _project(w, xy)
            s0, s1 = float(np.percentile(s, 2)), float(np.percentile(s, 98)) + 1.0
            if (s1 - s0) * px < 0.3:
                continue
            o = {"kind": kind, "wall": best[0], "s": (s0, s1), "sill": None, "head": None}
            if heights is not None:
                h0 = float(np.median(heights[0][comp])) * ceil_m
                h1 = float(np.median(heights[1][comp])) * ceil_m
                o["sill"] = round(h0, 2) if kind == "window" else 0.0
                o["head"] = round(h1, 2) if kind != "opening" else None
            out.append(o)
    # один проём, распавшийся на куски вдоль стены - один
    out.sort(key=lambda o: (o["wall"], o["kind"], o["s"][0]))
    merged = []
    for o in out:
        p = merged[-1] if merged else None
        if p and p["wall"] == o["wall"] and p["kind"] == o["kind"] and o["s"][0] <= p["s"][1] + 0.1 / px:
            p["s"] = (p["s"][0], max(p["s"][1], o["s"][1]))
        else:
            merged.append(o)
    return [o for o in merged if (o["s"][1] - o["s"][0]) * px >= OPEN_MIN_M[o["kind"]]]


def _wall_body(w: dict, ext=(0.0, 0.0)) -> np.ndarray | None:
    """Тело прямой стены - четырёхугольник по оси ± полутолщина, концы продлены на ext."""
    if len(w["pts"]) != 2 or w.get("arc"):
        return None
    a, b = np.asarray(w["pts"][0], float), np.asarray(w["pts"][1], float)
    d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
    n = np.array([-d[1], d[0]]) * w["t"] / 2
    a, b = a - d * ext[0], b + d * ext[1]
    return np.array([a + n, b + n, b - n, a - n])


def _barrier(items: list, shape, up: int) -> np.ndarray:
    """Тела стен в растре в up раз мельче пикселя сети (грани помещений - по ним). Конец,
    упёртый в другую стену, продлевается на её полутолщину - до её дальней грани, не дальше
    (иначе толстая стена в T-стыке с тонкой торчит в помещение)."""
    import cv2

    from .walls import _dist_to_wall

    H, W = shape
    bar = np.zeros((H * up, W * up), np.uint8)
    sh = 4                                                  # подпиксельные координаты cv2
    for w in items:
        ext = []
        for e in (w["pts"][0], w["pts"][-1]):
            near = [B["t"] / 2 for B in items if B is not w and _dist_to_wall(np.asarray(e, float), B) <= (w["t"] + B["t"]) / 2 + 1.0]
            ext.append(max(near) if near else 0.0)
        for e, x in zip((w["pts"][0], w["pts"][-1]), ext):     # стык: клин между телами - тоже стена
            if x > 0:
                c = (np.asarray(e, float) + 0.5) * up - 0.5
                cv2.circle(bar, tuple(np.round(c * (1 << sh)).astype(int)), int(round(max(x, w["t"] / 2) * up * (1 << sh))),
                           1, -1, cv2.LINE_8, sh)
        q = _wall_body(w, ext)
        if q is not None:
            q = ((q + 0.5) * up - 0.5) * (1 << sh)
            cv2.fillPoly(bar, [np.round(q).astype(np.int32)], 1, cv2.LINE_8, sh)
        else:
            q = (np.asarray(w["pts"]) + 0.5) * up - 0.5
            cv2.polylines(bar, [np.round(q).astype(np.int32)], False, 1, max(1, int(round(w["t"] * up))))
    return bar


def _line_x(p0, p1, q0, q1):
    """Пересечение прямых p0p1 и q0q1 (None - почти параллельны)."""
    u, v = p1 - p0, q1 - q0
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < 0.3 * np.linalg.norm(u) * np.linalg.norm(v):
        return None
    r = q0 - p0
    return p0 + u * (r[0] * v[1] - r[1] * v[0]) / den


def clean_outline(m: np.ndarray, P, eps_m: float = 0.015, short_m: float = 0.12, jog_m: float = 0.05,
                  col_deg: float = 3.0) -> np.ndarray:
    """Контур области -> чистый многоугольник: упрощение с допуском eps_m; почти соосные
    соседние стороны (до col_deg) - одна; короткая сторона (до short_m) на стыке двух
    непараллельных граней - угол в их пересечении; ступенька до jog_m между параллельными - вон.
    Грани помещения - грани стен, поэтому многоугольник выходит из немногих длинных сторон."""
    import cv2

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    q = cv2.approxPolyDP(c, max(1.0, P(eps_m)), True)[:, 0, :].astype(float)
    sin_col = math.sin(math.radians(col_deg))
    for _ in range(50):
        n = len(q)
        if n <= 3:
            break
        changed = False
        for i in range(n):                                  # соосные стороны
            a, b, c_ = q[i - 1], q[i], q[(i + 1) % n]
            u, v = b - a, c_ - b
            lu, lv = np.linalg.norm(u), np.linalg.norm(v)
            if lu > 1e-9 and lv > 1e-9 and abs(u[0] * v[1] - u[1] * v[0]) / (lu * lv) < sin_col and u @ v > 0:
                q = np.delete(q, i, axis=0)
                changed = True
                break
        if changed:
            continue
        for i in range(n):                                  # короткая сторона q[i] -> q[i+1]
            a0, a1 = q[i - 1], q[i]
            b0, b1 = q[(i + 1) % n], q[(i + 2) % n]
            L = float(np.linalg.norm(b0 - a1))
            if L >= P(short_m):
                continue
            x = _line_x(a0, a1, b0, b1)
            if x is not None and np.linalg.norm(x - (a1 + b0) / 2) <= L + P(0.03):     # дальше - шип
                q[i] = x
                q = np.delete(q, (i + 1) % n, axis=0)
                changed = True
                break
            if x is None and L <= P(jog_m):                  # ступенька между параллельными
                q = np.delete(q, [i, (i + 1) % n], axis=0)
                changed = True
                break
        if not changed:
            break
    return q


def _faces(items: list, margin: float) -> list:
    """Грани стен: по прямой каждого участка оси ± полутолщина -> (точка, направление, от, до)."""
    out = []
    for w in items:
        p = np.asarray(w["pts"], float)
        for a, b in zip(p[:-1], p[1:]):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            d = (b - a) / L
            n = np.array([-d[1], d[0]])
            for sg in (1, -1):
                out.append((a + n * sg * w["t"] / 2, d, -margin, L + margin))
    return out


def snap_outline(poly: np.ndarray, items: list, px: float, tol_m: float = 0.04, ang_deg: float = 5.0) -> np.ndarray:
    """Стороны контура помещения -> на грани стен (ближе tol_m, угол до ang_deg); соседние
    стороны на одной грани - одна; угол - пересечение граней соседних сторон (параллельных -
    прямоугольная ступенька); непривязанные стороны до 40 см между двумя гранями (щель на
    стыке) - вон. Как у cellplan: грань
    помещения - грань стены. Если результат странный - исходный контур."""
    F = _faces(items, 0.15 / px)
    tol, sin_a = tol_m / px, math.sin(math.radians(ang_deg))
    n = len(poly)

    def face_of(p, q):
        u = q - p
        L = float(np.linalg.norm(u))
        if L < 1e-9:
            return None
        u = u / L
        best = None
        for k, (c, d, s0, s1) in enumerate(F):
            if abs(u[0] * d[1] - u[1] * d[0]) > sin_a:
                continue
            nrm = np.array([-d[1], d[0]])
            dp, dq = abs((p - c) @ nrm), abs((q - c) @ nrm)
            sp, sq = (p - c) @ d, (q - c) @ d
            if max(dp, dq) > tol or max(sp, sq) < s0 or min(sp, sq) > s1:
                continue
            if best is None or dp + dq < best[1]:
                best = (k, dp + dq)
        return None if best is None else best[0]

    edges = [[poly[i], poly[(i + 1) % n], face_of(poly[i], poly[(i + 1) % n])] for i in range(n)]
    merged = []                                         # подряд на одной грани - одна сторона
    for e in edges:
        if merged and e[2] is not None and merged[-1][2] == e[2]:
            merged[-1][1] = e[1]
        else:
            merged.append(e)
    if len(merged) > 1 and merged[0][2] is not None and merged[0][2] == merged[-1][2]:
        merged[0][0] = merged[-1][0]
        merged.pop()

    def line_x(f, g):
        (c1, d1, *_), (c2, d2, *_) = F[f], F[g]
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 0.2:
            return None
        r = c2 - c1
        return c1 + d1 * (r[0] * d2[1] - r[1] * d2[0]) / den

    # щели на стыках: цепочка непривязанных сторон до run_m между двумя гранями - вон
    run_m = 0.4 / px
    while len(merged) > 3:
        m = len(merged)
        cut = None
        for i in range(m):
            if merged[i][2] is None or merged[(i + 1) % m][2] is not None:
                continue
            j, L = (i + 1) % m, 0.0
            while merged[j][2] is None and L <= run_m and j != i:
                L += float(np.linalg.norm(merged[j][1] - merged[j][0]))
                j = (j + 1) % m
            if merged[j][2] is not None and j != i and L <= run_m:
                cut = (i, j)
                break
        if cut is None:
            break
        i, j = cut
        drop = []
        k = (i + 1) % len(merged)
        while k != j:
            drop.append(k)
            k = (k + 1) % len(merged)
        merged = [e for k, e in enumerate(merged) if k not in drop]
    out = []
    m = len(merged)
    for i in range(m):
        a, b = merged[i], merged[(i + 1) % m]
        if a[2] is not None and b[2] is not None:
            x = line_x(a[2], b[2])
            if x is not None and np.linalg.norm(x - (a[1] + b[0]) / 2) <= 0.5 / px:
                out.append(x)
                continue
            if x is None:                                # параллельные - прямоугольная ступенька
                c, d, *_ = F[a[2]]
                v = c + d * ((a[1] - c) @ d)
                c2, d2, *_ = F[b[2]]
                out += [v, c2 + d2 * ((v - c2) @ d2)]
                continue
        for v, f in ((a[1], a[2]), (b[0], b[2])):        # свободная сторона - проекции на грани
            if f is not None:
                c, d, *_ = F[f]
                out.append(c + d * ((v - c) @ d))
            else:
                out.append(v)
    q = [out[0]]
    for v in out[1:]:
        if np.linalg.norm(v - q[-1]) > 0.01 / px:
            q.append(v)
    if len(q) > 1 and np.linalg.norm(q[0] - q[-1]) <= 0.01 / px:
        q.pop()
    q = np.array(q)
    area = lambda z: 0.5 * abs(float(z[:, 0] @ np.roll(z[:, 1], -1) - z[:, 1] @ np.roll(z[:, 0], -1)))  # noqa: E731
    if len(q) < 3 or abs(area(q) - area(poly)) > 0.05 * area(poly):
        return poly
    return q


def rooms(prob: np.ndarray, items: list, mask: np.ndarray, px: float, up: int = 4):
    """Помещения: где они - по сети, где их грани - по стенам.
    1. затравки - то, что сеть считает помещением, за вычетом преграды (итоговые стены + маска
       конструкции), сжатое на 20 см (щели уже 40 см рвутся, комнаты не сливаются); остальное
       свободное - к ближайшей затравке геодезическим ростом;
    2. вон - меньше MIN_ROOM_M2 и не замкнутое стенами: граница меньше чем на MIN_ENCLOSED
       проходит по стенам и проёмам (площадка за входной дверью, кусок улицы: у комнат 0,98-1,0,
       у площадки на реальном скане 0,62);
    3. область помещения - в растре в up раз мельче дорастает в обход тел векторных стен (до
       15 см): помещение доходит ровно до их граней;
    4. контур - clean_outline, затем snap_outline: стороны - на гранях стен, углы - их
       пересечения, щели на стыках - вон.
    -> ([{poly, faces, label}] в пикселях сети, растр меток up, up)."""
    import cv2

    H, W = prob.shape[1:]
    bar = mask.astype(np.uint8).copy()
    for w in items:
        cv2.polylines(bar, [np.round(w["pts"]).astype(np.int32)], False, 1, max(2, int(round(w["t"]))))
    free = ((prob[list(INTERIOR)].sum(0) > 0.5) & (bar == 0)).astype(np.uint8)
    r = max(1, int(round(0.2 / px)))
    core = cv2.erode(free, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, seeds = cv2.connectedComponents(core, connectivity=4)
    area = np.bincount(seeds.ravel(), minlength=n)
    keep = area * px * px >= 0.3
    keep[0] = False
    lab = np.where(keep[seeds], seeds, 0).astype(np.int32)
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(3 * r):                                  # геодезический рост внутри свободного
        grow = cv2.dilate(lab.astype(np.float32), k3).astype(np.int32)
        tgt = (lab == 0) & (free > 0) & (grow > 0)
        if not tgt.any():
            break
        lab[tgt] = grow[tgt]
    good = []
    for k in np.unique(lab):
        m = (lab == k).astype(np.uint8)
        if k == 0 or m.sum() * px * px < MIN_ROOM_M2:
            continue
        ring = (cv2.dilate(m, k3) > 0) & (m == 0)
        if ring.any() and bar[ring].mean() < MIN_ENCLOSED:
            continue                                       # граница - не по стенам: не помещение
        good.append(k)
    lab = np.where(np.isin(lab, good), lab, 0)
    # п. 3: метки - в мелкий растр, до граней стен
    lab_up = cv2.resize(lab.astype(np.float32), (W * up, H * up), interpolation=cv2.INTER_NEAREST)
    wall_up = _barrier(items, (H, W), up) > 0
    lab_up[wall_up] = 0
    # рост в обход стен (не сквозь тонкую стену и не за её оборванный конец дальше 15 см)
    kc = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(int(round(0.15 / px * up))):
        grow = cv2.dilate(lab_up, kc)
        tgt = (lab_up == 0) & ~wall_up & (grow > 0)
        if not tgt.any():
            break
        lab_up[tgt] = grow[tgt]
    lab_up = lab_up.astype(np.int32)
    out, lab_fin = [], np.zeros_like(lab_up)
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(round(0.04 / px * up)) + 1,) * 2)
    for k in good:
        # размыкание: щели между телами стен на стыках (пальцы в 2-5 см) - не помещение;
        # скруглённые им углы clean_outline вернёт пересечением граней
        m = cv2.morphologyEx((lab_up == k).astype(np.uint8), cv2.MORPH_OPEN, ko)
        nm, cl = cv2.connectedComponents(m, connectivity=4)
        if nm < 2:
            continue
        sizes = np.bincount(cl.ravel())
        sizes[0] = 0
        m = (cl == sizes.argmax()).astype(np.uint8)
        if m.sum() * (px / up) ** 2 < MIN_ROOM_M2:
            continue
        poly = clean_outline(m, lambda v: v / px * up)
        if len(poly) < 3:
            continue
        dm = cv2.distanceTransform(m, cv2.DIST_L2, 5)
        y0, x0 = np.unravel_index(int(dm.argmax()), dm.shape)
        to_px = lambda q: (np.asarray(q, float) + 0.5) / up - 0.5          # noqa: E731
        poly = (snap_outline(to_px(poly), items, px) + 0.5) * up - 0.5
        faces = [{"kind": "line", "a": to_px(poly[k]), "b": to_px(poly[(k + 1) % len(poly)])} for k in range(len(poly))]
        lab_fin[m > 0] = len(out) + 1
        out.append({"poly": to_px(poly), "faces": faces, "label": to_px([x0, y0])})
    return out, lab_fin, up


def wall_rooms(items: list, lab_up: np.ndarray, up: int, px: float) -> list[list[int]]:
    """Помещения по сторонам каждой стены (номера с 0): метки в 6 см за гранями вдоль оси."""
    Hu, Wu = lab_up.shape
    out = []
    for w in items:
        p = np.asarray(w["pts"], float)
        seg = []
        for a, b in zip(p[:-1], p[1:]):
            d = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
            nrm = np.array([-d[1], d[0]])
            for t in np.linspace(0.1, 0.9, 9):
                for sg in (1, -1):
                    q = a + (b - a) * t + sg * nrm * (w["t"] / 2 + 0.06 / px)
                    j, i = int(round((q[0] + 0.5) * up - 0.5)), int(round((q[1] + 0.5) * up - 0.5))
                    if 0 <= i < Hu and 0 <= j < Wu and lab_up[i, j] > 0:
                        seg.append(int(lab_up[i, j]) - 1)
        cnt = np.bincount(seg) if seg else np.zeros(0)
        out.append([int(k) for k in np.flatnonzero(cnt >= 3)])
    return out


def is_outer(prob: np.ndarray, w: dict, px: float) -> bool:
    """За одной из граней (на 15 см) - улица."""
    H, W = prob.shape[1:]
    a, b = w["pts"][0], w["pts"][-1]
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    n = np.array([-d[1], d[0]])
    sides = []
    for sg in (1, -1):
        q = np.array([w["pts"][k] for k in range(len(w["pts"]))]).mean(0) + sg * n * (w["t"] / 2 + 0.15 / px)
        j, i = int(np.clip(round(q[0]), 0, W - 1)), int(np.clip(round(q[1]), 0, H - 1))
        sides.append(prob[0, i, j])
    return max(sides) > 0.5


# ----------------------------------------------------------------------
# Уровень
# ----------------------------------------------------------------------

def _load(pred_dir: Path):
    import cv2

    if (pred_dir / "prob.npz").exists():
        prob = np.load(pred_dir / "prob.npz")["prob"].astype(np.float32)
    else:                                                  # только классы - вероятности 0/1
        sem = cv2.imread(str(pred_dir / "sem.png"), cv2.IMREAD_UNCHANGED)
        prob = np.stack([(sem == k).astype(np.float32) for k in range(8)])
    opt = {k: np.load(pred_dir / f"{k}.npy").astype(np.float32) for k in ("orient", "heights")
           if (pred_dir / f"{k}.npy").exists()}
    scene = json.loads((pred_dir / "scene.json").read_text(encoding="utf-8"))
    return prob, opt.get("orient"), opt.get("heights"), scene


def vectorize(pred_dir, log=print) -> dict:
    """Каталог уровня из infer -> план (словарь в формате export-ifc, мировые метры)."""
    prob, orient, heights, scene = _load(Path(pred_dir))
    return vectorize_arrays(prob, orient, heights, scene, log=log)


def vectorize_arrays(prob: np.ndarray, orient: np.ndarray | None, heights: np.ndarray | None, scene: dict,
                     log=print) -> dict:
    """Выход сети (prob [8,H,W], orient [2,H,W], heights [2,H,W]) и scene.json уровня -> план."""
    prob = np.asarray(prob, np.float32)
    M = np.asarray(scene["pixel_to_world"], float)
    px = float(abs(M[0, 0]))
    floor_z = float(scene.get("floor_z", 0.0))
    ceil_z = float(scene.get("ceiling_z", floor_z + 2.7))
    ceil_m = ceil_z - floor_z if 1.5 < ceil_z - floor_z < 10 else 2.7
    P = lambda m: m / px                                     # noqa: E731

    mask = struct_mask(prob, px)
    items = walls_from_mask(mask, px, orient)
    n0 = len(items)
    p_struct = prob[list(STRUCT)].sum(0)

    def bridge_ok(a, b):
        """В щели между соосными кусками сеть хоть немного видит конструкцию."""
        n = max(2, int(np.linalg.norm(b - a)))
        q = np.round(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, p_struct.shape[1] - 1)
        q[:, 1] = np.clip(q[:, 1], 0, p_struct.shape[0] - 1)
        return float(p_struct[q[:, 1], q[:, 0]].mean()) >= 0.12

    def merge(gap):
        # разрыв на линии стены - почти всегда пропуск сети: проход без двери - класс «проём»,
        # он в конструкции; поэтому соосные куски сливаются через зазор до GAP_M, а до
        # GAP_MAX_M - если в щели сеть хоть немного видит конструкцию
        return merge_collinear(items, P, gap_m=gap, keep_outer_apart=False, gap_max_m=GAP_MAX_M, bridge_ok=bridge_ok)

    regularize(items, P=P)
    items = merge(GAP_M)
    join(items, P, free_reach_m=GAP_M, bridge_ok=bridge_ok)
    regularize(items, P=P)
    items = merge(0.05)
    items = [w for w in items if np.linalg.norm(w["pts"][-1] - w["pts"][0]) * px >= MIN_WALL_M or w.get("arc")]
    items = tidy(items, P, reach_m=GAP_M, bridge_ok=bridge_ok)
    items = merge(0.05)                                     # стыки после чистки - соосные куски впритык
    items = chain_at_openings(prob, items, px)
    ops = openings(prob, heights, items, px, ceil_m)
    room_list, lab_up, up = rooms(prob, items, mask, px)
    w_rooms = wall_rooms(items, lab_up, up, px)

    def to_w(xy):
        xy = np.atleast_2d(xy)
        return (M @ np.c_[xy, np.ones(len(xy))].T).T

    lvl = str(scene.get("level", "L1"))
    room_id = [f"{lvl}-R{k + 1}" for k in range(len(room_list))]
    walls_out = []
    for k, w in enumerate(items):
        pts = to_w(w["pts"])
        # наружная: помещение с одной стороны, за другой по сети улица
        outer = len(w_rooms[k]) < 2 and bool(is_outer(prob, w, px))
        rec = {"id": f"{lvl}-W{k + 1}", "points": pts.round(3).tolist(), "thickness": round(w["t"] * px, 3),
               "is_outer": outer, "rooms": [room_id[r] for r in w_rooms[k]],
               "length": round(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), 3)}
        if w.get("arc"):
            rec["arc"] = {"center": to_w(w["arc"]["c"])[0].round(3).tolist(), "radius": round(w["arc"]["r"] * px, 3)}
        walls_out.append(rec)
    ops_out = []
    for k, o in enumerate(ops):
        w = items[o["wall"]]
        a, b = to_w(np.array([_point_at(w, o["s"][0]), _point_at(w, o["s"][1])]))
        ops_out.append({"id": f"{lvl}-O{k + 1}", "opening_type": o["kind"], "wall_id": walls_out[o["wall"]]["id"],
                        "a": a.round(3).tolist(), "b": b.round(3).tolist(),
                        "width": round(float(np.linalg.norm(b - a)), 3), "sill": o["sill"], "head": o["head"]})
    rooms_out = []
    for k, R in enumerate(room_list):
        q = to_w(R["poly"])
        area = 0.5 * abs(float(q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1)))
        per = float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum())
        faces = [{"kind": f["kind"], "a": to_w(f["a"])[0].round(3).tolist(), "b": to_w(f["b"])[0].round(3).tolist(),
                  **({"center": to_w(f["c"])[0].round(3).tolist(), "radius": round(f["r"] * px, 3)}
                     if f["kind"] == "arc" else {})} for f in R["faces"]]
        rooms_out.append({"id": room_id[k], "name": f"Помещение {k + 1}", "points": q.round(3).tolist(),
                          "area": round(area, 2), "perimeter": round(per, 2),
                          "label_xy": to_w(R["label"])[0].round(3).tolist(), "faces": faces})
    order = sorted(range(len(rooms_out)), key=lambda i: -rooms_out[i]["area"])   # большие - первыми
    ren = {rooms_out[i]["id"]: f"{lvl}-R{n + 1}" for n, i in enumerate(order)}
    rooms_out = [rooms_out[i] for i in order]
    for n, r in enumerate(rooms_out):
        r["id"], r["name"] = ren[r["id"]], f"Помещение {n + 1}"
    for w in walls_out:
        w["rooms"] = [ren[x] for x in w["rooms"]]
    log(f"{lvl}: стен {len(walls_out)} (кусков скелета {n0}), дуг {sum('arc' in w for w in walls_out)}, "
        f"проёмов {len(ops_out)}, помещений {len(rooms_out)}")
    frame = dict(scene.get("frame") or {})
    frame.setdefault("width", prob.shape[2])
    frame.setdefault("height", prob.shape[1])
    return {"level": lvl, "floor_z": floor_z, "ceiling_z": ceil_z, "walls": walls_out, "openings": ops_out,
            "rooms": rooms_out, "frame": frame, "pixel_to_world": M.tolist(),
            "metadata": {"method": "skeleton-graph", "pixel_m": px}}


# ----------------------------------------------------------------------
# Запись
# ----------------------------------------------------------------------

def save_json(plan: dict, path) -> None:
    Path(path).write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")


def _inside(pt, poly) -> bool:
    x, y = pt
    q = np.asarray(poly, float)
    xi, yi, xj, yj = q[:, 0], q[:, 1], np.roll(q[:, 0], 1), np.roll(q[:, 1], 1)
    c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj == yi, 1e-12, yj - yi) + xi)
    return bool(c.sum() % 2)


def level_plan(plan: dict):
    """plan.json (vectorize) -> floorplan.LevelPlan для того же рисунка, что у cellplan: тела
    стен (концы в стыках - до дальней грани соседа), проёмы на всю толщину стены, створка
    двери - как принято: из прихожей (помещение с наибольшим числом дверей) внутрь комнаты,
    у санузла и кладовой (до 4,5 м²) - наружу, входная - внутрь квартиры."""
    from .floorplan import LevelPlan, Opening, Room, WallSegment
    from .walls import _dist_to_wall

    W = [{"pts": np.asarray(w["points"], float), "t": float(w["thickness"])} for w in plan["walls"]]
    by_id = {w["id"]: (w, W[k]) for k, w in enumerate(plan["walls"])}
    rooms = plan.get("rooms", [])
    lp = LevelPlan(int(str(plan.get("level", "L1")).lstrip("L") or 1), float(plan.get("floor_z", 0.0)),
                   float(plan.get("ceiling_z", 2.7)))
    for w, d in zip(plan["walls"], W):
        lp.walls.append(WallSegment(w["id"], w["points"], w.get("rooms", []), w["thickness"], w["is_outer"],
                                    w["length"]))
        p = d["pts"]
        for k in range(len(p) - 1):
            ext = [0.0, 0.0]
            for e, end in ((0, k == 0), (1, k == len(p) - 2)):
                if not end:
                    continue
                near = [B["t"] / 2 for B in W if B is not d and _dist_to_wall(p[k + e], B) <= (d["t"] + B["t"]) / 2 + 0.05]
                ext[e] = max(near) if near else 0.0
            q = _wall_body({"pts": p[k:k + 2], "t": d["t"]}, ext)
            if q is not None:
                lp.slab.append(q.tolist())
        for v in p[1:-1]:                                  # излом ломаной - без щели
            a = np.linspace(0, 2 * math.pi, 13)[:-1]
            lp.slab.append((v + d["t"] / 2 * np.c_[np.cos(a), np.sin(a)]).tolist())
    def sides(o):
        w, d = by_id[o["wall_id"]]
        a, b = np.asarray(o["a"], float), np.asarray(o["b"], float)
        u = (b - a) / max(float(np.linalg.norm(b - a)), 1e-9)
        n = np.array([-u[1], u[0]])
        c = (a + b) / 2
        return n, [next((r for r in rooms if _inside(c + sg * n * (d["t"] / 2 + 0.3), r["points"])), None)
                   for sg in (1, -1)]

    n_doors = {}
    for o in plan["openings"]:
        if o["opening_type"] == "door":
            for r in sides(o)[1]:
                if r:
                    n_doors[r["id"]] = n_doors.get(r["id"], 0) + 1
    for o in plan["openings"]:
        w, d = by_id[o["wall_id"]]
        a, b = np.asarray(o["a"], float), np.asarray(o["b"], float)
        n, rs = sides(o)

        def score(r):                                      # куда открывается: больше - туда
            if r is None:
                return -1e9                                # наружу из квартиры - нет
            if r["area"] < 4.5:
                return -1e6                                # санузел, кладовая - наружу
            return -n_doors.get(r["id"], 0) * 1e3 + r["area"]

        if score(rs[0]) > score(rs[1]):                    # створка - против n: n - от помещения створки
            n = -n
        t = d["t"]
        quad = [a - n * t / 2, b - n * t / 2, b + n * t / 2, a + n * t / 2]
        kind = "gap" if o["opening_type"] == "opening" else o["opening_type"]
        lp.openings.append(Opening(o["id"], kind, o["wall_id"], o["a"], o["b"], [q.tolist() for q in quad],
                                   o["width"], o.get("sill"), o.get("head")))
    for r in rooms:
        lab = r.get("label_xy") or np.mean(r["points"], axis=0).tolist()
        lp.rooms.append(Room(r["id"], r["name"], r["points"], r["area"], r["perimeter"], tuple(lab),
                             r.get("faces", [])))
    return lp


def render_plan(plan: dict, path, title: str = "") -> None:
    """Рисунок плана - как у cellplan (floorplan.render): помещения с площадями, тела стен,
    окна, двери со створками, ширины проёмов, таблица. Формат - по расширению (png, svg)."""
    from .floorplan import render

    render(level_plan(plan), None, {}, Path(path), title=title, wall_label="стена (ось ± полутолщина)")


def save_png(plan: dict, path) -> None:
    render_plan(plan, path)


def save_svg(plan: dict, path) -> None:
    render_plan(plan, path)


def main_vectorize(pred_dir, out_dir, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = vectorize(pred_dir, log=log)
    save_json(plan, out / "plan.json")
    save_svg(plan, out / "plan.svg")
    save_png(plan, out / "plan.png")
    log(f"записано: {out / 'plan.json'}, plan.svg, plan.png")
    return plan
