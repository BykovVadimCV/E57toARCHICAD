"""Двухуровневая квартира: второй этаж над первым, лестница и проём в перекрытии.

Верхний этаж - копия планировки нижнего (те же наружные и межквартирные стены, окна, несущие
перегородки - как в реальных двухуровневых квартирах секционного дома) с другой высотой потолка,
своей обстановкой и без входной двери. Пол верхнего этажа - на dz = H нижнего + плита.

Лестница:
  straight - прямой марш вдоль глухой стены (без проёмов на участке марша), открытая сторона
             с ограждением; проём в плите - над той частью марша, где под плитой меньше 2 м;
  spiral   - винтовая в углу помещения вокруг стойки, проём - квадрат по диаметру.
На верхнем этаже вокруг проёма - ограждение (стойки, поручень, балясины или стекло), кроме
стороны выхода с лестницы и сторон вдоль стен.

Плита между этажами - два слоя по SLAB_M/2: низ - потолок первого этажа, верх - пол второго;
проём вырезается в обоих. Разметка (netinput.label_level): VOID (7) - проём в плите на верхнем
уровне и след лестницы на нижнем (meta["void_rects"] каждого уровня).
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .layout import Box, Item, Layout

SLAB_M = 0.2
STAIR_KINDS = ("room", "corridor", "kitchen")


def _u(rng, r) -> float:
    return float(rng.uniform(r[0], r[1]))


# ----------------------------------------------------------------------
# Верхний этаж
# ----------------------------------------------------------------------
def make_upper(lower: Layout, rng, ceiling_range=(2.5, 3.1)) -> Layout:
    up = Layout.from_dict(lower.to_dict())
    H0, H1 = lower.ceiling_height, round(_u(rng, ceiling_range), 3)
    up.ceiling_height = H1
    for r in up.rooms:
        r.ceiling_z = round(H1 - max(0.0, H0 - r.ceiling_z), 3)
    keep = []
    for o in up.openings:
        w = up.wall(o.wall_id)
        if o.kind == "door" and (w.role == "party" or (w.kind == "exterior" and len(o.rooms) == 1)):
            continue                                       # вход - только на нижнем этаже
        o.z1 = min(o.z1, H1 - (0.15 if o.kind == "window" else 0.05))
        keep.append(o)
    for k, o in enumerate(keep):
        o.id = k
    up.openings = keep
    up.items = []
    up.meta = {k: v for k, v in lower.meta.items() if k in ("warp", "program", "shape")}
    up.meta["level"] = 1
    return up


# ----------------------------------------------------------------------
# Место под лестницу
# ----------------------------------------------------------------------
def _side_line(rect, side):
    x0, y0, x1, y1 = rect
    return {"B": ((x0, y0), (x1, y0)), "T": ((x0, y1), (x1, y1)),
            "L": ((x0, y0), (x0, y1)), "R": ((x1, y0), (x1, y1))}[side]


def _openings_near(layout: Layout, p0, p1, a: float, b: float, tol: float = 0.35) -> bool:
    """Есть ли проём на отрезке [a, b] линии p0-p1 (параметр - метры от p0)."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / max(L, 1e-9)
    n = np.array([-u[1], u[0]])
    for o in layout.openings:
        w = layout.wall(o.wall_id)
        c = w.point(o.s)
        if abs(float((c - p0) @ n)) > w.thickness / 2 + tol:
            continue
        if abs(abs(float(w.u @ u)) - 1) > 1e-3:
            continue
        s = float((c - p0) @ u)
        if s + o.width / 2 + 0.15 > a and s - o.width / 2 - 0.15 < b:
            return True
    return False


def _rect_of(room):
    return max(room.rects(), key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))


def plan_stair(layout: Layout, dz: float, rng, p_spiral: float = 0.3) -> dict | None:
    """Выбрать помещение, тип и место лестницы. None - не нашлось."""
    H0 = layout.ceiling_height
    n = int(math.ceil(dz / 0.2))
    rise = dz / n
    rooms = [r for r in layout.rooms if r.kind in STAIR_KINDS and r.scanned]
    rng.shuffle(rooms)
    kinds = ["spiral", "straight"] if rng.random() < p_spiral else ["straight", "spiral"]
    for kind in kinds:
        for room in rooms:
            rect = _rect_of(room)
            x0, y0, x1, y1 = rect
            if kind == "straight":
                width, tread = _u(rng, (0.85, 1.05)), _u(rng, (0.22, 0.26))
                run = n * tread
                sides = ["B", "T", "L", "R"]
                rng.shuffle(sides)
                for side in sides:
                    p0, p1 = _side_line(rect, side)
                    along = math.dist(p0, p1)
                    depth = (y1 - y0) if side in ("B", "T") else (x1 - x0)
                    if along < run + 0.1 or depth < width + 0.9:
                        continue
                    # выход: в торец, если за концом марша не меньше 0,8 м помещения, иначе вбок
                    # с последних ступеней (марш упирается в угол - частый случай в квартирах)
                    starts = [(float(x), d) for x in np.arange(0.0, along - run + 1e-6, 0.1)
                              for d in (1, -1) if not _openings_near(layout, p0, p1, x, x + run)]
                    if not starts:
                        continue
                    a, up_dir = starts[int(rng.integers(len(starts)))]
                    st = _straight(room, rect, side, a, run, width, tread, rise, n, up_dir, H0)
                    st["exit"] = "end" if (along - a - run if up_dir > 0 else a) >= 0.8 else "side"
                    return st
            else:
                D = _u(rng, (1.4, 1.8))
                if min(x1 - x0, y1 - y0) < D + 1.0:
                    continue
                corners = [(sx, sy) for sx in (0, 1) for sy in (0, 1)]
                rng.shuffle(corners)
                for sx, sy in corners:
                    cx0 = x0 if sx == 0 else x1 - D
                    cy0 = y0 if sy == 0 else y1 - D
                    sq = (cx0, cy0, cx0 + D, cy0 + D)
                    pv = _side_line(rect, "L" if sx == 0 else "R")
                    ph = _side_line(rect, "B" if sy == 0 else "T")
                    if _openings_near(layout, *pv, cy0 - y0, cy0 - y0 + D) or \
                            _openings_near(layout, *ph, cx0 - x0, cx0 - x0 + D):
                        continue
                    # последняя ступень смотрит из угла в помещение (выход не в стену)
                    turn = 1 if rng.random() < 0.5 else -1
                    away = math.degrees(math.atan2(1 if sy == 0 else -1, 1 if sx == 0 else -1))
                    end_deg = away + float(rng.uniform(-35, 35))
                    start_deg = end_deg - 0.95 * 360 * turn * (n - 0.5) / n
                    return {"type": "spiral", "room_id": room.id, "footprint": list(sq),
                            "hole": [sq[0] - 0.05 * (sx == 1), sq[1] - 0.05 * (sy == 1),
                                     sq[2] + 0.05 * (sx == 0), sq[3] + 0.05 * (sy == 0)],
                            "center": [(sq[0] + sq[2]) / 2, (sq[1] + sq[3]) / 2], "diameter": D,
                            "n": n, "rise": rise, "start_deg": start_deg, "end_deg": end_deg,
                            "turn": turn, "corner": [sx, sy],
                            "wall_sides": ["L" if sx == 0 else "R", "B" if sy == 0 else "T"]}
    return None


def _straight(room, rect, side, a, run, width, tread, rise, n, up_dir, H0) -> dict:
    x0, y0, x1, y1 = rect
    if side in ("B", "T"):
        ya, yb = (y0, y0 + width) if side == "B" else (y1 - width, y1)
        fp = (x0 + a, ya, x0 + a + run, yb)
        axis = "x"
    else:
        xa, xb = (x0, x0 + width) if side == "L" else (x1 - width, x1)
        fp = (xa, y0 + a, xb, y0 + a + run)
        axis = "y"
    # без проёма - начальные ступени, над которыми до плиты не меньше 2,05 м
    k = max(0, min(n - 3, int((H0 - 2.05) / rise)))
    cut = k * tread
    if axis == "x":
        hole = (fp[0] + cut, fp[1], fp[2], fp[3]) if up_dir > 0 else (fp[0], fp[1], fp[2] - cut, fp[3])
    else:
        hole = (fp[0], fp[1] + cut, fp[2], fp[3]) if up_dir > 0 else (fp[0], fp[1], fp[2], fp[3] - cut)
    return {"type": "straight", "room_id": room.id, "footprint": list(fp), "hole": list(hole), "axis": axis,
            "up_dir": up_dir, "wall_side": side, "width": width, "tread": tread, "rise": rise, "n": n}


# ----------------------------------------------------------------------
# Геометрия лестницы и ограждения
# ----------------------------------------------------------------------
def stair_item(st: dict, rng, item_id: int, refl: float) -> Item:
    boxes, prims = [], []
    if st["type"] == "straight":
        closed = rng.random() < 0.4                      # монолит под маршем или ступени на косоурах
        x0, y0, x1, y1 = st["footprint"]
        n, rise, tread, w = st["n"], st["rise"], st["tread"], st["width"]
        th = 0.05
        for i in range(n):
            t = (i + 0.5) * tread
            z_top = (i + 1) * rise
            zb = 0.0 if closed else z_top - th
            if st["axis"] == "x":
                cx = x0 + t if st["up_dir"] > 0 else x1 - t
                boxes.append(Box((cx, (y0 + y1) / 2, (zb + z_top) / 2), (tread + 0.02, w, z_top - zb), 0.0))
            else:
                cy = y0 + t if st["up_dir"] > 0 else y1 - t
                boxes.append(Box(((x0 + x1) / 2, cy, (zb + z_top) / 2), (w, tread + 0.02, z_top - zb), 0.0))
        # ограждение марша с открытой стороны: стойки на каждой 3-й ступени, поручень ступенями
        rail_h = _u(rng, (0.85, 0.95))
        for i in range(0, n, 3):
            t = (i + 0.5) * tread
            z0 = (i + 1) * rise
            if st["axis"] == "x":
                px = x0 + t if st["up_dir"] > 0 else x1 - t
                py = y1 - 0.04 if st["wall_side"] == "B" else y0 + 0.04
            else:
                py = y0 + t if st["up_dir"] > 0 else y1 - t
                px = x1 - 0.04 if st["wall_side"] == "L" else x0 + 0.04
            boxes.append(Box((px, py, z0 + rail_h / 2), (0.04, 0.04, rail_h), 0.0))
            seg = 3 * tread
            size = (seg, 0.05, 0.05) if st["axis"] == "x" else (0.05, seg, 0.05)
            cx, cy = px, py
            if st["axis"] == "x":
                cx += (seg / 2 - tread / 2) * st["up_dir"]
            else:
                cy += (seg / 2 - tread / 2) * st["up_dir"]
            boxes.append(Box((cx, cy, z0 + rail_h + 1.5 * rise), size, 0.0))
    else:
        cx, cy = st["center"]
        R = st["diameter"] / 2
        n, rise = st["n"], st["rise"]
        step = 2 * math.pi * 0.95 / n * st["turn"]       # почти полный оборот
        a0 = math.radians(st["start_deg"])
        prims.append({"type": "cylinder", "center": [cx, cy, n * rise / 2], "radius": 0.06, "height": n * rise})
        for i in range(n):
            ang = a0 + (i + 0.5) * step
            r_mid = R / 2 + 0.04
            wide = abs(step) * R * 1.05
            c = (cx + r_mid * math.cos(ang), cy + r_mid * math.sin(ang), (i + 1) * rise - 0.025)
            boxes.append(Box(c, (R - 0.08, wide * 0.75, 0.05), ang))
            if i % 2 == 0:                               # стойка ограждения на краю ступени
                px, py = cx + (R - 0.03) * math.cos(ang), cy + (R - 0.03) * math.sin(ang)
                boxes.append(Box((px, py, (i + 1) * rise + 0.45), (0.03, 0.03, 0.9), 0.0))
    return Item(item_id, "stairs", "stairs", boxes, refl, st["room_id"], prims)


def railing_item(st: dict, layout: Layout, rng, item_id: int, refl: float) -> Item:
    """Ограждение вокруг проёма на верхнем этаже (кроме выхода с лестницы и сторон у стен)."""
    hx0, hy0, hx1, hy1 = st["hole"]
    room = layout.rooms[st["room_id"]]
    rx0, ry0, rx1, ry1 = _rect_of(room)
    near = 0.08
    sides = {"B": ((hx0, hy0), (hx1, hy0)), "T": ((hx0, hy1), (hx1, hy1)),
             "L": ((hx0, hy0), (hx0, hy1)), "R": ((hx1, hy0), (hx1, hy1))}
    at_wall = {"B": hy0 - ry0 < near, "T": ry1 - hy1 < near, "L": hx0 - rx0 < near, "R": rx1 - hx1 < near}
    if st["type"] == "straight":
        exit_side = {("x", 1): "R", ("x", -1): "L", ("y", 1): "T", ("y", -1): "B"}[(st["axis"], st["up_dir"])]
    else:
        ang = math.radians(st["end_deg"])
        dx, dy = math.cos(ang), math.sin(ang)
        exit_side = ("R" if dx > 0 else "L") if abs(dx) > abs(dy) else ("T" if dy > 0 else "B")
    h = _u(rng, (0.9, 1.05))
    glass = rng.random() < 0.3
    boxes = []
    side_exit = st.get("exit") == "side"
    if side_exit:                                     # сход вбок: торец ограждён, открытая сторона - не до верха
        open_side = {"B": "T", "T": "B", "L": "R", "R": "L"}[st["wall_side"]]
    for s, (p, q) in sides.items():
        if at_wall[s] or (s == exit_side and not side_exit):
            continue
        p, q = np.asarray(p), np.asarray(q)
        if side_exit and s == open_side:              # оставить 0,9 м у верхнего конца марша
            top = q if st["up_dir"] > 0 else p
            d = (q - p) / max(float(np.linalg.norm(q - p)), 1e-9)
            if st["up_dir"] > 0:
                q = top - d * 0.9
            else:
                p = top + d * 0.9
        L = float(np.linalg.norm(q - p))
        if L < 0.2:
            continue
        u = (q - p) / L
        yaw = math.atan2(u[1], u[0])
        out = np.array([-u[1], u[0]])
        mid = (p + q) / 2
        # наружу от проёма на 3 см, чтобы стойки стояли на плите
        if float(out @ (mid - np.array([(hx0 + hx1) / 2, (hy0 + hy1) / 2]))) < 0:
            out = -out
        off = out * 0.03
        c = mid + off
        boxes.append(Box((c[0], c[1], h - 0.025), (L, 0.05, 0.05), yaw))              # поручень
        boxes.append(Box((c[0], c[1], 0.1), (L, 0.03, 0.03), yaw))                     # нижний пояс
        k = max(2, int(round(L / 1.0)) + 1)
        for t in np.linspace(0, L, k):
            pp = p + u * t + off
            boxes.append(Box((pp[0], pp[1], h / 2), (0.04, 0.04, h), 0.0))
        if glass:
            boxes.append(Box((c[0], c[1], 0.12 + (h - 0.2) / 2), (L - 0.08, 0.012, h - 0.25), yaw))
        else:
            for t in np.arange(0.12, L - 0.06, 0.12):
                pp = p + u * t + off
                boxes.append(Box((pp[0], pp[1], (0.115 + h - 0.05) / 2), (0.02, 0.02, h - 0.165), 0.0))
    return Item(item_id, "railing", "stairs", boxes, refl, st["room_id"], [])


# ----------------------------------------------------------------------
# Сборка
# ----------------------------------------------------------------------
def make_duplex(lower: Layout, cfg, rng) -> dict | None:
    """Верхний этаж + лестница; меняет lower на месте. None - лестница не встала."""
    lc = cfg.layout
    dz = round(lower.ceiling_height + SLAB_M, 4)
    st = plan_stair(lower, dz, rng, lc.duplex_p_spiral)
    if st is None:
        return None
    up = make_upper(lower, rng, lc.ceiling_height_m)
    room = lower.rooms[st["room_id"]]
    room.ceiling_z = lower.ceiling_height             # без подвесного потолка над лестницей
    refl = _u(rng, (0.2, 0.7))
    lower.items.append(stair_item(st, rng, len(lower.items), refl))
    up.items.append(railing_item(st, up, rng, len(up.items), refl))
    half = SLAB_M / 2
    lower.meta.update({"duplex": {"dz": dz, "stair": st}, "reserved": [st["footprint"]],
                       "void_rects": [st["footprint"]], "slab": {"ceiling": {"holes": [st["hole"]], "thickness": half}}})
    up.meta.update({"duplex": {"dz": dz, "stair": st}, "reserved": [st["hole"]], "void_rects": [st["hole"]],
                    "slab": {"floor": {"holes": [st["hole"]], "thickness": half}}})
    return {"upper": up, "dz": dz, "stair": st}


def subtract_rect(r, h) -> list:
    """Прямоугольник r минус прямоугольник h -> до 4 прямоугольников."""
    x0, y0, x1, y1 = r
    a0, b0, a1, b1 = max(h[0], x0), max(h[1], y0), min(h[2], x1), min(h[3], y1)
    if a0 >= a1 or b0 >= b1:
        return [tuple(r)]
    out = []
    if b0 > y0:
        out.append((x0, y0, x1, b0))
    if b1 < y1:
        out.append((x0, b1, x1, y1))
    if a0 > x0:
        out.append((x0, b0, a0, b1))
    if a1 < x1:
        out.append((a1, b0, x1, b1))
    return out


def upper_realism(real: dict, rng) -> dict:
    """Неровность верхнего этажа - независимая (свои фазы), сдвиг плана и завал - те же."""
    if not real.get("enabled"):
        return real
    r = dict(real)
    r["phi"] = rng.uniform(0, 2 * math.pi, np.asarray(real["phi"]).shape).tolist()
    return r


def build_duplex_scene(lower: Layout, upper: Layout, dz: float, real: dict, real_up: dict):
    """Сетка двух этажей. Каждый этаж деформируется в своей СК (завал от своего пола)."""
    from . import realism
    from .scene import SceneMesh, build_scene

    mesh0, solids0 = build_scene(lower, real.get("tessellation_m"))
    realism.apply_to_mesh(mesh0, real)
    mesh1, solids1 = build_scene(upper, real.get("tessellation_m"))
    realism.apply_to_mesh(mesh1, real_up)
    mesh1.vertices = mesh1.vertices + np.array([0, 0, dz], np.float32)
    off = int(mesh0.tri_instance.max()) + 1 if len(mesh0.tri_instance) else 1
    mesh1.tri_instance = mesh1.tri_instance + off
    mesh1.instances = [{**d, "instance": d["instance"] + off, "level": 1} for d in mesh1.instances]
    for s in solids1:
        s.box = replace(s.box, center=(s.box.center[0], s.box.center[1], s.box.center[2] + dz))
        s.instance += off
    return mesh0.concatenate(mesh1), solids0, solids1
