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
     стена; концы - проекция на её ось; низ и верх - медиана головы высот;
  7. помещения: что сеть считает помещением, за вычетом стен (итоговых и маски); сужения уже
     40 см рвутся - комнаты не сливаются через щели; контур - упрощённый.
  Наружная стена - та, у которой за одной из граней улица.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .walls import join, merge_collinear, regularize, tidy

STRUCT = (2, 3, 4, 5)
INTERIOR = (1, 6, 7)                 # помещение, мебель, проём перекрытия
OPEN_KIND = {3: "door", 4: "window", 5: "opening"}

P_HI, P_LO = 0.5, 0.3                # гистерезис конструкции
MIN_WALL_M = 0.15                    # прямая короче - не стена
MIN_SPLIT_M = 0.25                   # кусок короче не дробится (шум скелета - не излом)
SMOOTH_PX = 5                        # окно сглаживания цепочки перед подгонкой
TOL_PX = 1.5                         # допуск ломаной от точек скелета
ARC_R = (1.5, 40.0)                  # радиусы дуг, м
ARC_MIN_CHORD_M = 1.0                # дуга: хорда не короче,
ARC_MIN_SAG_M = 0.08                 # стрела не меньше и окружность в 3 раза точнее прямой
MIN_ROOM_M2 = 1.0
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
    return merged


def rooms(prob: np.ndarray, items: list, mask: np.ndarray, px: float) -> list[np.ndarray]:
    """Помещения -> контуры (j, i). Свободное = то, что сеть считает помещением, за вычетом
    преграды (итоговые стены с толщиной + маска конструкции: стены закрывают дыры маски,
    маска - несомкнутые углы стен). Затравки - свободное, сжатое на 20 см (сужения и щели уже
    40 см рвутся, комнаты не сливаются); остальное свободное - к ближайшей затравке
    геодезическим ростом."""
    import cv2

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
    out = []
    for k in np.unique(lab):
        if k == 0:
            continue
        m = (lab == k).astype(np.uint8)
        if m.sum() * px * px < MIN_ROOM_M2:
            continue
        m = cv2.dilate(m, k3, iterations=max(1, int(round(0.03 / px))))        # до грани стены
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cnts, key=cv2.contourArea)
        poly = cv2.approxPolyDP(c, max(1.0, 0.02 / px), True)[:, 0, :].astype(float)
        if len(poly) >= 3:
            out.append(poly)
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
    join(items, P, free_reach_m=GAP_M)
    regularize(items, P=P)
    items = merge(0.05)
    items = [w for w in items if np.linalg.norm(w["pts"][-1] - w["pts"][0]) * px >= MIN_WALL_M or w.get("arc")]
    items = tidy(items, P, reach_m=GAP_M)
    items = merge(0.05)                                     # стыки после чистки - соосные куски впритык
    ops = openings(prob, heights, items, px, ceil_m)
    room_polys = rooms(prob, items, mask, px)

    def to_w(xy):
        xy = np.atleast_2d(xy)
        return (M @ np.c_[xy, np.ones(len(xy))].T).T

    lvl = str(scene.get("level", "L1"))
    walls_out = []
    for k, w in enumerate(items):
        pts = to_w(w["pts"])
        rec = {"id": f"{lvl}-W{k + 1}", "points": pts.round(3).tolist(), "thickness": round(w["t"] * px, 3),
               "is_outer": bool(is_outer(prob, w, px)), "rooms": [],
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
    for k, poly in enumerate(room_polys):
        q = to_w(poly)
        area = 0.5 * abs(float(q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1)))
        per = float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum())
        rooms_out.append({"id": f"{lvl}-R{k + 1}", "name": f"Помещение {k + 1}", "points": q.round(3).tolist(),
                          "area": round(area, 2), "perimeter": round(per, 2)})
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


def _draw(plan: dict, path, svg: bool) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    fig, ax = plt.subplots(figsize=(10, 10))
    for r in plan["rooms"]:
        ax.add_patch(Polygon(r["points"], closed=True, fc="#e8f1fa", ec="none"))
        c = np.mean(r["points"], axis=0)
        ax.text(c[0], c[1], f"{r['area']:.1f} м²", ha="center", va="center", fontsize=7, color="#555")
    for w in plan["walls"]:
        p = np.asarray(w["points"])
        lw = max(1.0, w["thickness"] * 72 / 0.0254 / 100 * 0.35)
        ax.plot(p[:, 0], p[:, 1], color="#c04040" if w["is_outer"] else "#303030", lw=lw, solid_capstyle="butt")
    col = {"door": "#d6604d", "window": "#4393c3", "opening": "#f4a582"}
    for o in plan["openings"]:
        a, b = np.asarray(o["a"]), np.asarray(o["b"])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col.get(o["opening_type"], "k"), lw=4, solid_capstyle="butt")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{plan['level']}: стен {len(plan['walls'])}, проёмов {len(plan['openings'])}, "
                 f"помещений {len(plan['rooms'])}", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight", format="svg" if svg else "png")
    plt.close(fig)


def save_png(plan: dict, path) -> None:
    _draw(plan, path, False)


def save_svg(plan: dict, path) -> None:
    _draw(plan, path, True)


def main_vectorize(pred_dir, out_dir, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = vectorize(pred_dir, log=log)
    save_json(plan, out / "plan.json")
    save_svg(plan, out / "plan.svg")
    save_png(plan, out / "plan.png")
    log(f"записано: {out / 'plan.json'}, plan.svg, plan.png")
    return plan
