"""Привязка векторного плана к облаку точек: сеть говорит, ГДЕ стены, проёмы и помещения
(растр 2 см), облако - ТОЧНО где их грани (тысячи точек на грань, мм вместо см).

    python -m simscan snap pred/kvartira/L1/plan.json kvartira.e57 [--out plan.json]

Для каждой прямой стены плана (мировые метры):
  1. точки уровня в полосе высот стен (от пола + 15 см до потолка - 15 см), вдоль стены без
     концов (у стыков - грани соседних стен), с каждой стороны оси - от оси до полутолщины
     + WIN_M;
  2. грань с каждой стороны - самый плотный слой у ожидаемой грани (перебор наклонов до
     1,5°, гистограмма по 5 мм с весом-гауссом вокруг грани сети), затем прямая по
     точкам ближе 1 см к слою и ещё раз по уточнённой выборке; грань
     принимается, если точек хватает и они покрывают заметную часть длины стены;
  3. видны обе грани и они параллельны - ось посередине, толщина - между ними; видна одна -
     ближняя по облаку, дальняя по сети (до оси доходит половина ошибки сети); ни одной - как
     было. Помещение только с одной стороны (наружная стена, соседнее неотснятое) - грань
     ищется только с неё: снаружи изнутри видны рамы и откосы окон, а не фасад; толщина
     наружу - по сети, как в разметке;
     Сдвиг больше MAX_MOVE_M - не та поверхность (мебель, соседняя стена): как было;
  4. углы - заново пересечением осей (walls.join), проёмы - проекцией на новые оси, стороны
     помещений - на новые грани своих стен, углы помещений - их пересечения.
Цепочки - стены «конец в конец» с изломом до CHAIN_DEG и ломаные (окно через излом) - сначала
подгоняются целиком: одна грань по облаку на всю цепочку. Если у какого-то куска своя грань
отходит от общей прямой (KINK_M, KINK_DEG), излом настоящий - цепочка не трогается, её куски
идут по одному; иначе цепочка - одна прямая (ломаные - прямые стены). Так куски одной
физической стены не получают каждый свою ошибку угла (изломы 2-5° на стыках).
Дуги не трогаются.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

Z_MARGIN_M = 0.15                    # полоса высот стен: от пола и до потолка
NEAR_M = 0.35                        # точки дальше от оси любой стены не нужны
WIN_M = 0.10                         # окно поиска грани за полутолщиной
END_M = 0.15                         # концы стены (стыки) - без точек
MAX_MOVE_M = 0.15
MIN_PTS = 60
MIN_COVER = 0.35                     # доля длины стены, покрытая точками грани
PRIOR_M = 0.04                       # разброс грани сети относительно настоящей
MAX_FACE_M = 0.08                    # грань дальше от грани сети - не та поверхность
CHAIN_DEG = 30.0                     # излом в цепочке до этого угла - кандидат на одну прямую (решает облако)
CHAIN_DEV_M = 0.5                    # вершина цепочки дальше от опорной прямой - не одна стена
CHAIN_MOVE_M = 0.5                   # сдвиг вершин цепочки больше - не та поверхность
KINK_M = 0.025                       # своя грань куска дальше от общей прямой (в середине куска) -
KINK_DEG = 2.0                       # или повёрнута сильнее - излом настоящий
MAX_TURN_DEG = 3.0                   # и повёрнутая сильнее (на реальном скане сеть ошибалась в направлении до 2,3°)


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def level_points(chunks, plans: list[dict], cell: float = 0.02) -> list[np.ndarray]:
    """Точки (x, y) каждого уровня: полоса высот стен и окрестность стен его плана."""
    import cv2

    masks = []
    for plan in plans:
        pts = [np.asarray(w["points"], float) for w in plan["walls"] if len(w["points"]) >= 2]
        if not pts:
            masks.append(None)
            continue
        allp = np.vstack(pts)
        lo = allp.min(0) - 1.0
        hi = allp.max(0) + 1.0
        W, H = int(np.ceil((hi[0] - lo[0]) / cell)) + 1, int(np.ceil((hi[1] - lo[1]) / cell)) + 1
        m = np.zeros((H, W), np.uint8)
        for w in plan["walls"]:
            p = (np.asarray(w["points"], float) - lo) / cell
            th = int(round((float(w.get("thickness") or 0.1) + 2 * NEAR_M) / cell))
            cv2.polylines(m, [np.round(p).astype(np.int32)], False, 1, max(1, th))
        z0 = float(plan["floor_z"]) + Z_MARGIN_M
        z1 = float(plan["ceiling_z"]) - Z_MARGIN_M
        masks.append((m.astype(bool), lo, cell, z0, z1))
    out = [[] for _ in plans]
    for p in chunks:
        p = np.asarray(p, np.float64)
        for k, mk in enumerate(masks):
            if mk is None:
                continue
            m, lo, c, z0, z1 = mk
            q = p[(p[:, 2] > z0) & (p[:, 2] < z1)]
            if not len(q):
                continue
            ij = np.floor((q[:, :2] - lo) / c).astype(np.int64)
            ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < m.shape[1]) & (ij[:, 1] < m.shape[0])
            ij, q = ij[ok], q[ok]
            sel = m[ij[:, 1], ij[:, 0]]
            out[k].append(q[sel, :2].astype(np.float32))
    return [np.concatenate(o).astype(np.float64) if o else np.zeros((0, 2)) for o in out]


def _fit_face(s: np.ndarray, o: np.ndarray, L: float, expect: float, why: dict | None = None,
              prior: float = PRIOR_M):
    """Точки одной стороны в СК стены (s - вдоль, o - от оси наружу) -> (o0 на s=0, наклон
    do/ds, rms, точек) или None.
    Слой грани: перебор наклонов до MAX_TURN_DEG (сеть могла повернуть стену - грань плывёт
    вдоль оси), для каждого - гистограмма отступов по 5 мм, взвешенная гауссом вокруг
    ожидаемой грани expect (PRIOR_M): рама окна, мебель вплотную и прочие слои проигрывают
    плотной плоской грани. Затем прямая по точкам ближе 1 см к слою, ещё раз по уточнённой.
    why (словарь) получает причину отказа: few_points, no_layer, low_cover (и долю покрытия)."""
    why = {} if why is None else why
    if len(o) < MIN_PTS:
        why.update(reason="few_points", points=int(len(o)))
        return None
    best = None
    for k in np.tan(np.radians(np.arange(-MAX_TURN_DEG, MAX_TURN_DEG + 1e-9, 0.1))):
        r = o - k * s
        lo, hi = r.min(), r.max()
        if hi - lo < 0.005:
            c, h = np.array([np.median(r)]), np.array([float(len(r))])
        else:
            h, e = np.histogram(r, np.arange(lo, hi + 0.005, 0.005))
            h = np.convolve(h, [1, 2, 1], "full")[1:-1].astype(float)
            c = e[:-1] + 0.0025
        w = h * np.exp(-0.5 * ((c - expect) / prior) ** 2)
        i = int(w.argmax())
        if best is None or w[i] > best[0]:
            best = (w[i], float(c[i]), float(k))
    _, o0, k = best
    sel = np.abs(o - (o0 + k * s)) < 0.01
    for _ in range(2):
        if sel.sum() < MIN_PTS:
            why.update(reason="no_layer", points=int(len(o)))
            return None
        A = np.c_[np.ones(sel.sum()), s[sel]]
        coef, *_ = np.linalg.lstsq(A, o[sel], rcond=None)
        res = o - (coef[0] + coef[1] * s)
        sel = np.abs(res) < 0.01
    if sel.sum() < MIN_PTS:
        why.update(reason="no_layer", points=int(len(o)))
        return None
    cov = np.histogram(s[sel], np.linspace(0, L, max(2, int(L / 0.05)) + 1))[0] > 0
    if cov.mean() < MIN_COVER:
        why.update(reason="low_cover", cover=round(float(cov.mean()), 2))
        return None
    rms = float(np.sqrt(np.mean(res[sel] ** 2)))
    return float(coef[0]), float(coef[1]), rms, int(sel.sum())


def snap_wall(w: dict, xy: np.ndarray, sides=(1, -1), why: dict | None = None) -> dict | None:
    """Прямая стена {points, thickness} -> {a, b, t, sides, move, rms} по облаку или None.
    sides - стороны (по нормали оси a->b), где искать грань: у наружной - только помещение
    (наружу толщина - по сети, как в разметке; фасад в окне - не грань).
    why (словарь) получает причину отказа (reason: short, no_face, thickness, move) и по каждой
    искавшейся стороне - причину или смещение найденной грани от грани сети (offset_cm)."""
    why = {} if why is None else why
    a, b = np.asarray(w["points"][0], float), np.asarray(w["points"][-1], float)
    L = float(np.linalg.norm(b - a))
    if L < 2 * END_M + 0.2:
        why["reason"] = "short"
        return None
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    t = float(w.get("thickness") or 0.1)
    r = xy - a
    s, o = r @ d, r @ n
    along = (s > END_M) & (s < L - END_M)
    faces = {}
    why["sides"] = {}
    for sg in sides:
        m = along & (sg * o >= 0) & (sg * o <= t / 2 + WIN_M)
        wf: dict = {}
        f = _fit_face(s[m], sg * o[m], L, t / 2, wf)
        # дальше от грани сети или повёрнута (ступенька толщины, мебель) - чужая поверхность
        if f is not None:
            off = max(abs(f[0] - t / 2), abs(f[0] + f[1] * L - t / 2))
            wf.update(offset_cm=round(off * 100, 1), turn_deg=round(math.degrees(math.atan(f[1])), 2))
            if off > MAX_FACE_M:
                wf["reason"] = "far"
            elif abs(f[1]) > math.tan(math.radians(MAX_TURN_DEG)):
                wf["reason"] = "turned"
            else:
                faces[sg] = f
        why["sides"][sg] = wf
    if not faces:
        why["reason"] = "no_face"
        return None

    def face_line(sg):                                   # грань sg: точка на s=0 и направление
        o0, k, *_ = faces[sg]
        return a + n * sg * o0, _unit(d + n * sg * k)

    if len(faces) == 2:
        (p1, u1), (p2, u2) = face_line(1), face_line(-1)
        if abs(u1[0] * u2[1] - u1[1] * u2[0]) > math.sin(math.radians(2.0)):
            faces.pop(min(faces, key=lambda k: faces[k][3]))   # не параллельны - оставить плотную
    if len(faces) == 2:
        (p1, u1), (p2, u2) = face_line(1), face_line(-1)
        u = _unit(u1 + u2)
        nu = np.array([-u[1], u[0]])
        tn = float((p1 - p2) @ nu)
        c = (p1 + p2) / 2
        t_new = abs(tn)
        if not 0.03 <= t_new <= 1.2:
            why.update(reason="thickness", t_cm=round(t_new * 100, 1))
            return None
    else:                                                # одна грань: ближняя - облако, дальняя - сеть
        sg = next(iter(faces))
        p1, u = face_line(sg)
        nu = np.array([-u[1], u[0]])
        far = a - n * sg * t / 2                         # дальняя грань сети
        m_near = p1 + u * (((a + b) / 2 - p1) @ u)
        m_far = far + d * (((a + b) / 2 - far) @ d)
        t_new = abs(float((m_near - m_far) @ nu))
        c = (m_near + m_far) / 2
        if not 0.03 <= t_new <= 1.2:
            why.update(reason="thickness", t_cm=round(t_new * 100, 1))
            return None
    # концы - проекция старых на новую ось
    a2 = c + u * ((a - c) @ u)
    b2 = c + u * ((b - c) @ u)
    move = max(abs((a - c) @ nu), abs((b - c) @ nu))
    if move > MAX_MOVE_M:
        why.update(reason="move", move_cm=round(move * 100, 1))
        return None
    return {"a": a2, "b": b2, "t": t_new, "sides": len(faces), "move": move,
            "rms": max(f[2] for f in faces.values())}


def _inside(pt, poly) -> bool:
    x, y = pt
    xi, yi, xj, yj = poly[:, 0], poly[:, 1], np.roll(poly[:, 0], 1), np.roll(poly[:, 1], 1)
    c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj == yi, 1e-12, yj - yi) + xi)
    return bool(c.sum() % 2)


def _faces_of(walls: list[dict]) -> list:
    """Грани прямых стен и участков ломаных: (номер стены, сторона, участок, точка, направление, длина)."""
    out = []
    for k, w in enumerate(walls):
        p = np.asarray(w["points"], float)
        if len(p) < 2 or w.get("arc"):
            continue
        t = float(w.get("thickness") or 0.1)
        for j, (a, b) in enumerate(zip(p[:-1], p[1:])):
            L = float(np.linalg.norm(b - a))
            if L < 1e-6:
                continue
            d = (b - a) / L
            n = np.array([-d[1], d[0]])
            for sg in (1, -1):
                out.append((k, sg, j, a + n * sg * t / 2, d, L))
    return out


def _x(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 0.2:
        return None
    r = p2 - p1
    return p1 + d1 * (r[0] * d2[1] - r[1] * d2[0]) / den


def _move_room(poly: np.ndarray, old: list, new: list, tol: float = 0.05, ang_deg: float = 5.0) -> np.ndarray:
    """Стороны многоугольника помещения - на новые грани тех же стен (ломаная, ставшая прямой, -
    все её участки на одну грань); углы - пересечения."""
    newmap = {(k, sg, j): (p, d) for k, sg, j, p, d, _ in new}
    n = len(poly)
    lines = []
    sin_a = math.sin(math.radians(ang_deg))
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        u = _unit(q - p)
        best = None
        for k, sg, j, c, d, L in old:
            if abs(u[0] * d[1] - u[1] * d[0]) > sin_a:
                continue
            nr = np.array([-d[1], d[0]])
            e = abs((p - c) @ nr) + abs((q - c) @ nr)
            sp, sq = (p - c) @ d, (q - c) @ d
            if e > 2 * tol or max(sp, sq) < -0.3 or min(sp, sq) > L + 0.3:
                continue
            if best is None or e < best[0]:
                best = (e, (k, sg, j))
        lines.append((newmap.get(best[1]) or newmap.get((best[1][0], best[1][1], 0))) if best else None)
    out = []
    for i in range(n):
        l0, l1 = lines[i - 1], lines[i]
        v = poly[i]
        x = _x(*l0, *l1) if l0 is not None and l1 is not None else None
        if x is not None and np.linalg.norm(x - v) <= 0.3:
            out.append(x)
            continue
        for ln in (l0, l1):                              # проекция на ту грань, что есть
            if ln is not None:
                c, d = ln
                v = c + d * ((v - c) @ d)
        out.append(v)
    return np.array(out)


def chains(walls: list[dict], tol_m: float = 0.04) -> list[list[tuple[int, bool]]]:
    """Цепочки стен «конец в конец» с изломом до CHAIN_DEG: [(номер стены, развёрнута)] по
    порядку. В узле с несколькими стенами в пару идут концы с наименьшим изломом. Ломаная без
    соседей по цепочке - цепочка из одной стены; одиночная прямая стена - не цепочка."""
    ok = [k for k, w in enumerate(walls) if not w.get("arc") and len(w["points"]) >= 2]
    pts = {k: np.asarray(walls[k]["points"], float) for k in ok}

    def end(k, e):
        return pts[k][0] if e == 0 else pts[k][-1]

    def away(k, e):                                       # направление от стены наружу у конца e
        p = pts[k]
        return _unit(p[0] - p[1]) if e == 0 else _unit(p[-1] - p[-2])

    ends = [(k, e) for k in ok for e in (0, 1)]
    pairs = []
    for i, (k1, e1) in enumerate(ends):
        for k2, e2 in ends[i + 1:]:
            if k1 == k2 or np.linalg.norm(end(k1, e1) - end(k2, e2)) > tol_m:
                continue
            ang = math.degrees(math.acos(float(np.clip(-away(k1, e1) @ away(k2, e2), -1, 1))))
            if ang <= CHAIN_DEG:
                pairs.append((ang, (k1, e1), (k2, e2)))
    link = {}
    for _, x, y in sorted(pairs):
        if x not in link and y not in link:
            link[x], link[y] = y, x
    seen, out = set(), []
    for k in ok:
        if k in seen or ((k, 0) in link and (k, 1) in link):
            continue                                       # начало цепочки - стена со свободным концом
        e_in = 0 if (k, 0) not in link else 1
        cur, rev, seq = k, e_in == 1, []
        while cur is not None and cur not in seen:
            seen.add(cur)
            seq.append((cur, rev))
            nxt = link.get((cur, 0 if rev else 1))
            if nxt is None:
                break
            cur, rev = nxt[0], nxt[1] == 1
        if len(seq) > 1 or len(pts[seq[0][0]]) > 2:
            out.append(seq)
    return out


def fit_chain(walls: list[dict], chain: list, xy: np.ndarray, rooms: list, why: dict) -> dict | None:
    """Цепочка -> {номер стены: (новые концы, толщина)} по одной грани облака на всю цепочку или
    None (why["reason"]: short, bent, no_face, kink, move, thickness)."""
    V, segs = [], []
    for k, rev in chain:
        p = np.asarray(walls[k]["points"], float)
        p = p[::-1] if rev else p
        i0 = len(V) - 1 if V else 0
        V.extend(p if not V else p[1:])
        segs.append((k, i0, len(V) - 1, float(walls[k].get("thickness") or 0.1)))
    V = np.array(V)
    a, b = V[0], V[-1]
    L = float(np.linalg.norm(b - a))
    if L < 1.0:
        why["reason"] = "short"
        return None
    d = _unit(b - a)
    n = np.array([-d[1], d[0]])
    sv, ov = (V - a) @ d, (V - a) @ n
    if np.abs(ov).max() > 2 * CHAIN_DEV_M or np.any(np.diff(sv) <= 0):
        why["reason"] = "bent"
        return None
    # стороны с помещением (как у одиночной стены)
    sides = []
    for sg in (1, -1):
        for k, i0, i1, t in segs:
            for f in (0.25, 0.5, 0.75):
                m = V[i0] + (V[i1] - V[i0]) * f
                if any(_inside(m + n * sg * (t / 2 + 0.3), q) for q in rooms):
                    sides.append(sg)
                    break
            if sides and sides[-1] == sg:
                break
    sides = tuple(sides) or (1, -1)
    r = xy - a
    s, o = r @ d, r @ n
    tw = np.array([t for *_, t in segs])
    # опорная ось - прямая по вершинам (вес - длины участков у вершины): у прямой стены с изломом
    # сети она ближе к правде, чем ломаная, и полоса от неё не захватывает противоположную грань
    wv = np.zeros(len(V))
    wv[:-1] += np.diff(sv)
    wv[1:] += np.diff(sv)
    beta, alpha = np.polyfit(sv, ov, 1, w=np.sqrt(wv))
    dev = float(np.abs(ov - (alpha + beta * sv)).max())
    if dev > CHAIN_DEV_M:
        why["reason"] = "bent"
        return None
    base = alpha + beta * s
    prior = max(PRIOR_M, dev)                               # сеть ошиблась на dev - ожидание размыто
    faces = {}
    why["sides"] = {}
    for sg in sides:
        rel = sg * (o - base)
        # с другой стороны помещение - не заходить за ось дальше полутолщины (там чужая грань)
        lo = -dev if len(sides) == 1 else -min(dev, max(0.0, float(tw.min()) / 2 - 0.02))
        m = (s > END_M) & (s < L - END_M) & (rel >= lo) & (rel <= tw.max() / 2 + WIN_M + dev)
        wf: dict = {}
        f = _fit_face(s[m], rel[m], L, float(np.median(tw)) / 2, wf, prior)
        why["sides"][sg] = wf
        if f is None:
            continue
        c, k_ = f[0], f[1]
        cf, kf = alpha + sg * c, beta + sg * k_                # грань в СК хорды: o = cf + kf s
        # настоящий излом: у куска своя протяжённая прямая грань вдоль его направления, не общая
        for jv in range(len(V) - 1):
            pa, pb = V[jv], V[jv + 1]
            Ls = float(np.linalg.norm(pb - pa))
            if Ls < 0.6:
                continue
            ds = _unit(pb - pa)
            ns = np.array([-ds[1], ds[0]])
            t_s = float(np.median(tw))
            rs = xy - pa
            ss, os_ = rs @ ds, sg * (rs @ ns) * np.sign(ns @ n)
            mm = (ss > END_M) & (ss < Ls - END_M) & (os_ >= 0) & (os_ <= t_s / 2 + WIN_M)
            fs = _fit_face(ss[mm], os_[mm], Ls, t_s / 2)
            if fs is None:
                continue                                       # своей грани нет (диагональ зигзага)
            mid = pa + ds * Ls / 2 + ns * np.sign(ns @ n) * sg * (fs[0] + fs[1] * Ls / 2)
            own_dir = _unit(ds + ns * np.sign(ns @ n) * sg * fs[1])
            r_mid = mid - a
            off = abs(float(r_mid @ n) - (cf + kf * float(r_mid @ d)))
            dang = math.degrees(math.acos(min(1.0, abs(float(own_dir @ _unit(d + n * kf))))))
            if off > KINK_M or dang > KINK_DEG:
                why.update(reason="kink", at_m=round(float((sv[jv] + sv[jv + 1]) / 2), 2),
                           off_cm=round(off * 100, 1), deg=round(dang, 2))
                return None
        faces[sg] = (cf, kf, f[3])
    if not faces:
        why["reason"] = "no_face"
        return None
    if len(faces) == 2 and abs(math.degrees(math.atan(faces[1][1]) - math.atan(faces[-1][1]))) > 2.0:
        faces.pop(min(faces, key=lambda q: faces[q][2]))   # не параллельны - остаётся плотная
    res = {}
    for k, i0, i1, t in segs:
        if len(faces) == 2:
            (c1, k1, _), (c2, k2, _) = faces[1], faces[-1]
            t_new = (c1 - c2) + (k1 - k2) * (sv[i0] + sv[i1]) / 2      # грани: o = c1 + k1 s и o = c2 + k2 s
            axis = lambda x: ((c1 + k1 * x) + (c2 + k2 * x)) / 2       # noqa: E731
        else:
            sg, (c, k_, _) = next(iter(faces.items()))
            t_new = t
            axis = lambda x, sg=sg, c=c, k_=k_, t=t: c + k_ * x - sg * t / 2   # noqa: E731
        if not 0.03 <= t_new <= 1.2:
            why.update(reason="thickness", t_cm=round(t_new * 100, 1))
            return None
        new = np.array([a + d * x + n * axis(x) for x in (sv[i0], sv[i1])])
        mv = max(abs(axis(x) - ov[i]) for x, i in ((sv[i], i) for i in range(i0, i1 + 1)))
        # помещения с обеих сторон, а грань одна: сдвиг больше полутолщины - почти наверняка чужая грань
        lim = CHAIN_MOVE_M if len(faces) == 2 or len(sides) == 1 else min(CHAIN_MOVE_M, max(0.03, t / 2))
        if mv > lim:
            why.update(reason="move", move_cm=round(mv * 100, 1))
            return None
        if any(rv for kk, rv in chain if kk == k):
            new = new[::-1]
        res[k] = (new, float(t_new), float(mv))
    why.update(reason="ok", faces=len(faces))
    return res


def snap_plan(plan: dict, xy: np.ndarray, log=print) -> dict:
    """План (мировые метры) + точки уровня (x, y) -> уточнённый план; в metadata.snap - сводка."""
    from .walls import join

    walls = plan["walls"]
    old_faces = _faces_of(walls)
    n_two = n_one = 0
    moves = []
    items, per_wall = [], []
    rooms = [np.asarray(r["points"], float) for r in plan.get("rooms", [])]
    done, n_chain = {}, 0                                  # цепочки - целиком, одной гранью
    for ch in (chains(walls) if len(xy) else []):
        cw: dict = {}
        res = fit_chain(walls, ch, xy, rooms, cw)
        for k, _ in ch:
            done[k] = (res[k] if res else None, cw, len(ch))
        n_chain += res is not None
    for kw, w in enumerate(walls):
        p = np.asarray(w["points"], float)
        it = {"pts": p.copy(), "t": float(w.get("thickness") or 0.1), "arc": w.get("arc")}
        rec = {"id": w.get("id"), "faces": 0, "t_net_cm": round(it["t"] * 100, 1), "status": "not_straight"}
        if kw in done and done[kw][0] is not None:
            (new, t_new, mv), cw, nch = done[kw]
            it["pts"], it["t"] = new, t_new
            rec.update(status="snapped", faces=int(cw.get("faces", 0)), chain=nch, move_cm=round(mv * 100, 2),
                       t_cloud_cm=round(t_new * 100, 1))
            moves.append(mv)
            n_two += cw.get("faces") == 2
            n_one += cw.get("faces") == 1
            items.append(it)
            per_wall.append(rec)
            continue
        if kw in done:
            rec["chain_reason"] = done[kw][1].get("reason")
        if len(p) == 2 and not w.get("arc") and len(xy):
            # помещение с одной стороны - грань только с неё: снаружи изнутри видны рамы и
            # откосы окон, а не фасад (толщина наружу - по сети, как в разметке)
            d = _unit(p[1] - p[0])
            n = np.array([-d[1], d[0]])
            t = float(w.get("thickness") or 0.1)
            probe = [p[0] + (p[1] - p[0]) * f for f in (0.25, 0.5, 0.75)]
            sides = tuple(sg for sg in (1, -1)
                          if any(_inside(m + n * sg * (t / 2 + 0.3), q) for m in probe for q in rooms)) or (1, -1)
            why: dict = {}
            r = snap_wall(w, xy, sides, why)
            sw = why.get("sides", {})
            rec.update(status=why.get("reason", "no_face"), searched_sides=len(sides),
                       **{k: why[k] for k in ("t_cm", "move_cm") if k in why},
                       **{f"side{'+' if sg > 0 else '-'}": ",".join(f"{k}={v}" for k, v in d.items())
                          for sg, d in sw.items()})
            if why.get("reason") == "no_face" and all(d.get("reason") == "few_points" for d in sw.values()):
                rec["status"] = "no_points"
            if r is not None:
                rec.update(status="snapped", faces=int(r["sides"]), move_cm=round(float(r["move"]) * 100, 2),
                           t_cloud_cm=round(float(r["t"]) * 100, 1), rms_mm=round(float(r["rms"]) * 1000, 1))
                it["pts"] = np.array([r["a"], r["b"]])
                it["t"] = r["t"]
                moves.append(r["move"])
                n_two += r["sides"] == 2
                n_one += r["sides"] == 1
        items.append(it)
        per_wall.append(rec)
    join(items, lambda m: m, tol_m=0.03, free_reach_m=0.0)  # углы - пересечением новых осей
    out = json.loads(json.dumps(plan))
    for w, it in zip(out["walls"], items):
        w["points"] = np.round(it["pts"], 4).tolist()
        w["thickness"] = round(float(it["t"]), 4)
        w["length"] = round(float(np.linalg.norm(np.diff(it["pts"], axis=0), axis=1).sum()), 4)
    by_id = {w["id"]: it for w, it in zip(out["walls"], items)}
    for o in out.get("openings", []):
        it = by_id.get(o["wall_id"])
        if it is None or len(it["pts"]) != 2:
            continue
        a, b = it["pts"]
        d = _unit(b - a)
        pa, pb = (np.asarray(o["a"], float) - a) @ d, (np.asarray(o["b"], float) - a) @ d
        o["a"], o["b"] = np.round(a + d * pa, 4).tolist(), np.round(a + d * pb, 4).tolist()
        o["width"] = round(abs(pb - pa), 4)
    new_faces = _faces_of(out["walls"])
    for r in out.get("rooms", []):
        q = _move_room(np.asarray(r["points"], float), old_faces, new_faces)
        r["points"] = np.round(q, 4).tolist()
        r["area"] = round(0.5 * abs(float(q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1))), 2)
        r["perimeter"] = round(float(np.linalg.norm(q - np.roll(q, -1, 0), axis=1).sum()), 2)
        r["faces"] = [{"kind": "line", "a": q[k].round(4).tolist(), "b": q[(k + 1) % len(q)].round(4).tolist()}
                      for k in range(len(q))]
    rep = {"points": int(len(xy)), "walls": len(walls), "two_faces": int(n_two), "one_face": int(n_one),
           "move_cm_median": round(float(np.median(moves)) * 100, 2) if moves else None,
           "move_cm_max": round(float(np.max(moves)) * 100, 2) if moves else None, "chains_straightened": n_chain,
           "per_wall": per_wall}
    out.setdefault("metadata", {})["snap"] = rep
    log(f"{plan.get('level', '')}: к облаку - стен {n_two + n_one} из {len(walls)} (обе грани {n_two}, одна "
        f"{n_one}), сдвиг медиана {rep['move_cm_median']} см, макс {rep['move_cm_max']} см")
    return out


def snap_e57(plans: list[dict], e57, log=print) -> list[dict]:
    """Планы уровней одного E57 -> привязанные к его облаку (один проход по файлу)."""
    from .netinput import E57Source

    src = E57Source(e57)
    try:
        xy = level_points(src.chunks(), plans)
    finally:
        src.close()
    return [snap_plan(p, q, log=log) for p, q in zip(plans, xy)]


def main_snap(plan_paths, e57, out=None, log=print) -> list[Path]:
    """plan.json (один или несколько уровней одного E57) -> привязанные; рядом - plan.png/svg,
    исходный план сети (при записи на место) - plan_net.json."""
    from .vectorize import save_json, save_png, save_svg

    paths = [Path(p) for p in plan_paths]
    plans = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    done = []
    for p, src, plan in zip(paths, plans, snap_e57(plans, e57, log=log)):
        dst = Path(out) if out and len(paths) == 1 else p
        if dst == p:                                      # исходный план сети - рядом, для сравнения
            save_json(src, p.with_name(p.stem + "_net.json"))
        save_json(plan, dst)
        save_png(plan, dst.with_suffix(".png"))
        save_svg(plan, dst.with_suffix(".svg"))
        done.append(dst)
    return done
