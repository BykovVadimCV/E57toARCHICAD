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
Дуги и ломаные пока не трогаются.
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
MAX_TURN_DEG = 1.5                   # и повёрнутая сильнее (сеть ошибается в направлении меньше)


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


def _fit_face(s: np.ndarray, o: np.ndarray, L: float, expect: float):
    """Точки одной стороны в СК стены (s - вдоль, o - от оси наружу) -> (o0 на s=0, наклон
    do/ds, rms, точек) или None.
    Слой грани: перебор наклонов до MAX_TURN_DEG (сеть могла повернуть стену - грань плывёт
    вдоль оси), для каждого - гистограмма отступов по 5 мм, взвешенная гауссом вокруг
    ожидаемой грани expect (PRIOR_M): рама окна, мебель вплотную и прочие слои проигрывают
    плотной плоской грани. Затем прямая по точкам ближе 1 см к слою, ещё раз по уточнённой."""
    if len(o) < MIN_PTS:
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
        w = h * np.exp(-0.5 * ((c - expect) / PRIOR_M) ** 2)
        i = int(w.argmax())
        if best is None or w[i] > best[0]:
            best = (w[i], float(c[i]), float(k))
    _, o0, k = best
    sel = np.abs(o - (o0 + k * s)) < 0.01
    for _ in range(2):
        if sel.sum() < MIN_PTS:
            return None
        A = np.c_[np.ones(sel.sum()), s[sel]]
        coef, *_ = np.linalg.lstsq(A, o[sel], rcond=None)
        res = o - (coef[0] + coef[1] * s)
        sel = np.abs(res) < 0.01
    if sel.sum() < MIN_PTS:
        return None
    cov = np.histogram(s[sel], np.linspace(0, L, max(2, int(L / 0.05)) + 1))[0] > 0
    if cov.mean() < MIN_COVER:
        return None
    rms = float(np.sqrt(np.mean(res[sel] ** 2)))
    return float(coef[0]), float(coef[1]), rms, int(sel.sum())


def snap_wall(w: dict, xy: np.ndarray, sides=(1, -1)) -> dict | None:
    """Прямая стена {points, thickness} -> {a, b, t, sides, move, rms} по облаку или None.
    sides - стороны (по нормали оси a->b), где искать грань: у наружной - только помещение
    (наружу толщина - по сети, как в разметке; фасад в окне - не грань)."""
    a, b = np.asarray(w["points"][0], float), np.asarray(w["points"][-1], float)
    L = float(np.linalg.norm(b - a))
    if L < 2 * END_M + 0.2:
        return None
    d = (b - a) / L
    n = np.array([-d[1], d[0]])
    t = float(w.get("thickness") or 0.1)
    r = xy - a
    s, o = r @ d, r @ n
    along = (s > END_M) & (s < L - END_M)
    faces = {}
    for sg in sides:
        m = along & (sg * o >= 0) & (sg * o <= t / 2 + WIN_M)
        f = _fit_face(s[m], sg * o[m], L, t / 2)
        # дальше от грани сети или повёрнута (ступенька толщины, мебель) - чужая поверхность
        if f is not None and max(abs(f[0] - t / 2), abs(f[0] + f[1] * L - t / 2)) <= MAX_FACE_M \
                and abs(f[1]) <= math.tan(math.radians(MAX_TURN_DEG)):
            faces[sg] = f
    if not faces:
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
            return None
    # концы - проекция старых на новую ось
    a2 = c + u * ((a - c) @ u)
    b2 = c + u * ((b - c) @ u)
    move = max(abs((a - c) @ nu), abs((b - c) @ nu))
    if move > MAX_MOVE_M:
        return None
    return {"a": a2, "b": b2, "t": t_new, "sides": len(faces), "move": move,
            "rms": max(f[2] for f in faces.values())}


def _inside(pt, poly) -> bool:
    x, y = pt
    xi, yi, xj, yj = poly[:, 0], poly[:, 1], np.roll(poly[:, 0], 1), np.roll(poly[:, 1], 1)
    c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / np.where(yj == yi, 1e-12, yj - yi) + xi)
    return bool(c.sum() % 2)


def _faces_of(walls: list[dict]) -> list:
    """Грани прямых стен: (номер стены, сторона, точка, направление, длина)."""
    out = []
    for k, w in enumerate(walls):
        p = np.asarray(w["points"], float)
        if len(p) != 2 or w.get("arc"):
            continue
        a, b = p
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        d = (b - a) / L
        n = np.array([-d[1], d[0]])
        t = float(w.get("thickness") or 0.1)
        for sg in (1, -1):
            out.append((k, sg, a + n * sg * t / 2, d, L))
    return out


def _x(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 0.2:
        return None
    r = p2 - p1
    return p1 + d1 * (r[0] * d2[1] - r[1] * d2[0]) / den


def _move_room(poly: np.ndarray, old: list, new: list, tol: float = 0.05, ang_deg: float = 5.0) -> np.ndarray:
    """Стороны многоугольника помещения - на новые грани тех же стен; углы - пересечения."""
    newmap = {(k, sg): (p, d) for k, sg, p, d, _ in new}
    n = len(poly)
    lines = []
    sin_a = math.sin(math.radians(ang_deg))
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        u = _unit(q - p)
        best = None
        for k, sg, c, d, L in old:
            if abs(u[0] * d[1] - u[1] * d[0]) > sin_a:
                continue
            nr = np.array([-d[1], d[0]])
            e = abs((p - c) @ nr) + abs((q - c) @ nr)
            sp, sq = (p - c) @ d, (q - c) @ d
            if e > 2 * tol or max(sp, sq) < -0.3 or min(sp, sq) > L + 0.3:
                continue
            if best is None or e < best[0]:
                best = (e, (k, sg))
        lines.append(newmap.get(best[1]) if best else None)
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


def snap_plan(plan: dict, xy: np.ndarray, log=print) -> dict:
    """План (мировые метры) + точки уровня (x, y) -> уточнённый план; в metadata.snap - сводка."""
    from .walls import join

    walls = plan["walls"]
    old_faces = _faces_of(walls)
    n_two = n_one = 0
    moves = []
    items, per_wall = [], []
    rooms = [np.asarray(r["points"], float) for r in plan.get("rooms", [])]
    for w in walls:
        p = np.asarray(w["points"], float)
        it = {"pts": p.copy(), "t": float(w.get("thickness") or 0.1), "arc": w.get("arc")}
        rec = {"id": w.get("id"), "faces": 0, "t_net_cm": round(it["t"] * 100, 1), "status": "not_straight"}
        if len(p) == 2 and not w.get("arc") and len(xy):
            # помещение с одной стороны - грань только с неё: снаружи изнутри видны рамы и
            # откосы окон, а не фасад (толщина наружу - по сети, как в разметке)
            d = _unit(p[1] - p[0])
            n = np.array([-d[1], d[0]])
            t = float(w.get("thickness") or 0.1)
            probe = [p[0] + (p[1] - p[0]) * f for f in (0.25, 0.5, 0.75)]
            sides = tuple(sg for sg in (1, -1)
                          if any(_inside(m + n * sg * (t / 2 + 0.3), q) for m in probe for q in rooms)) or (1, -1)
            r = snap_wall(w, xy, sides)
            rec.update(status="no_face", searched_sides=len(sides))
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
           "move_cm_max": round(float(np.max(moves)) * 100, 2) if moves else None, "per_wall": per_wall}
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
