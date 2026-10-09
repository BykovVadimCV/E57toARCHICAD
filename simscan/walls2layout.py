"""Стены -> помещения, контур перекрытий, участки стен для проёмов.

Общая часть для внешних планировок (datasetgen) и генератора квартир:
стены считаются первичной геометрией, помещения - связные области между ними
на сжатой сетке по граням стен (точно, без растеризации).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .config import LayoutConfig
from .layout import Layout, Opening, Room, Wall

OUTSIDE, SOLID = -2, -1


class _Grid:
    """Сжатая сетка по граням стен: клетка - стена, улица или помещение."""

    def __init__(self, rects: list, pad: float = 1.0):
        xs = {r[0] for r in rects} | {r[2] for r in rects}
        ys = {r[1] for r in rects} | {r[3] for r in rects}
        self.xs = np.array(sorted(xs | {min(xs) - pad, max(xs) + pad}))
        self.ys = np.array(sorted(ys | {min(ys) - pad, max(ys) + pad}))
        ny, nx = len(self.ys) - 1, len(self.xs) - 1
        solid = np.zeros((ny, nx), bool)
        for x0, y0, x1, y1 in rects:
            i0, i1 = np.searchsorted(self.xs, [x0, x1])
            j0, j1 = np.searchsorted(self.ys, [y0, y1])
            solid[j0:j1, i0:i1] = True
        lab, n = ndimage.label(~solid)
        border = set(lab[0]) | set(lab[-1]) | set(lab[:, 0]) | set(lab[:, -1])
        self.cls = np.full((ny, nx), SOLID, int)
        self.cls[np.isin(lab, list(border - {0}))] = OUTSIDE
        self.components = [np.argwhere(lab == k) for k in range(1, n + 1) if k not in border]
        self.solid = solid

    def cell_rect(self, j, i):
        return (float(self.xs[i]), float(self.ys[j]), float(self.xs[i + 1]), float(self.ys[j + 1]))

    def classify(self, x: float, y: float) -> int:
        i = int(np.searchsorted(self.xs, x, side="right")) - 1
        j = int(np.searchsorted(self.ys, y, side="right")) - 1
        if not (0 <= i < self.cls.shape[1] and 0 <= j < self.cls.shape[0]):
            return OUTSIDE
        return int(self.cls[j, i])


def _merge_cells(grid: _Grid, cells: np.ndarray) -> list:
    """Клетки компоненты -> покрытие прямоугольниками (полосы по строкам, затем по столбцам)."""
    rows: dict[int, list] = {}
    for j, i in sorted(map(tuple, cells)):
        runs = rows.setdefault(j, [])
        if runs and runs[-1][1] == i:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    rects = []
    open_: dict[tuple, list] = {}
    for j in sorted(rows):
        nxt = {}
        for i0, i1 in rows[j]:
            key = (i0, i1)
            if key in open_ and open_[key][3] == j:
                open_[key][3] = j + 1
                nxt[key] = open_.pop(key)
            else:
                nxt[key] = [i0, j, i1, j + 1]
        rects += list(open_.values())
        open_ = nxt
    rects += list(open_.values())
    return [(float(grid.xs[i0]), float(grid.ys[j0]), float(grid.xs[i1]), float(grid.ys[j1]))
            for i0, j0, i1, j1 in rects]


def _wall_edges(grid: _Grid, cells: np.ndarray) -> list:
    """Грани между клетками помещения и стенами, нормаль внутрь помещения."""
    own = {tuple(c) for c in cells}
    edges = []
    for j, i in own:
        x0, y0, x1, y1 = grid.cell_rect(j, i)
        for dj, di, edge, n in ((0, -1, (x0, y0, x0, y1), (1, 0)), (0, 1, (x1, y0, x1, y1), (-1, 0)),
                                (-1, 0, (x0, y0, x1, y0), (0, 1)), (1, 0, (x0, y1, x1, y1), (0, -1))):
            jj, ii = j + dj, i + di
            if (jj, ii) in own:
                continue
            if 0 <= jj < grid.solid.shape[0] and 0 <= ii < grid.solid.shape[1] and grid.solid[jj, ii]:
                edges.append((*edge, *n))
    # слить соседние коллинеарные
    edges.sort(key=lambda e: (e[4], e[5], e[0] if e[4] else e[1], e[1] if e[4] else e[0]))
    out = []
    for e in edges:
        if out:
            p = out[-1]
            same_line = p[4:] == e[4:] and ((e[4] and p[0] == e[0] and abs(p[3] - e[1]) < 1e-9)
                                            or (e[5] and p[1] == e[1] and abs(p[2] - e[0]) < 1e-9))
            if same_line:
                out[-1] = (p[0], p[1], e[2], e[3], p[4], p[5])
                continue
        out.append(e)
    return out


def wall_body(w: Wall) -> tuple:
    pts = np.array([w.point(s, o) for s in (-w.ext0, w.length + w.ext1)
                    for o in (-w.thickness / 2, w.thickness / 2)])
    return (*pts.min(0).round(6), *pts.max(0).round(6))


def rooms_from_walls(walls: list[Wall], H: float, min_room_m2: float):
    """Помещения - связные области между телами стен; щели меньше min_room_m2 - пустоты."""
    grid = _Grid([wall_body(w) for w in walls])
    rooms: list[Room] = []
    for cells in grid.components:
        region = _merge_cells(grid, cells)
        area = sum((c - a) * (d - b) for a, b, c, d in region)
        if area < min_room_m2:
            continue
        rid = len(rooms)
        for j, i in cells:
            grid.cls[j, i] = rid
        arr = np.array(region)
        bbox = (float(arr[:, 0].min()), float(arr[:, 1].min()),
                float(arr[:, 2].max()), float(arr[:, 3].max()))
        rooms.append(Room(rid, bbox, bbox, "room", H, region=region,
                          wall_edges=_wall_edges(grid, cells)))
    if not rooms:
        raise ValueError("нет замкнутых помещений")
    return rooms, grid


def footprint_rects(grid: _Grid) -> list:
    return _merge_cells(grid, np.argwhere(grid.cls != OUTSIDE))


def label_overlap(room: Room, rect) -> float:
    x0, y0, x1, y1 = rect
    return sum(max(0.0, min(c, x1) - max(a, x0)) * max(0.0, min(d, y1) - max(b, y0))
               for a, b, c, d in room.rects())


def wall_spans(layout: Layout, grid: _Grid):
    """Участки стен по сторонам: что за гранью - помещение, улица или другая стена."""
    int_spans, ext_spans, balcony_spans = [], [], []
    rooms = layout.rooms
    for w in layout.walls:
        if w.kind == "parapet":
            continue
        horizontal = abs(w.u[0]) > 0.5
        lo = w.p0[0] if horizontal else w.p0[1]
        brk = grid.xs if horizontal else grid.ys
        ss = sorted({0.0, w.length, *[float(v - lo) for v in brk if 0 < v - lo < w.length]})
        segs = []
        for a, b in zip(ss[:-1], ss[1:]):
            m = (a + b) / 2
            sides = tuple(grid.classify(*w.point(m, sd * (w.thickness / 2 + 1e-3))) for sd in (1, -1))
            if segs and segs[-1][2] == sides and abs(segs[-1][1] - a) < 1e-9:
                segs[-1][1] = b
            else:
                segs.append([a, b, sides])
        outside_votes = 0
        for a, b, (left, right) in segs:
            if left >= 0 and right >= 0 and left != right:
                pair = (min(left, right), max(left, right))
                if "balcony" not in (rooms[left].kind, rooms[right].kind):
                    int_spans.append((w.id, a, b, pair))
                elif w.role == "balcony_interface":     # боковые стены балкона - без проёмов
                    balcony_spans.append((w.id, a, b, pair))
            elif w.kind == "exterior" and (left == OUTSIDE) != (right == OUTSIDE):
                room = left if left >= 0 else right
                if room >= 0:
                    ext_spans.append((w.id, a, b, room))
                outside_votes += (b - a) * (1 if left == OUTSIDE else -1)
        if w.kind == "exterior":
            w.outward = 1 if outside_votes >= 0 else -1
        if w.role == "balcony_interface":
            for a, b, (left, right) in segs:
                if left >= 0 and rooms[left].kind == "balcony":
                    w.outward = 1
                elif right >= 0 and rooms[right].kind == "balcony":
                    w.outward = -1
    return int_spans, ext_spans, balcony_spans


def place_balcony_blocks(layout: Layout, spans, cfg: LayoutConfig, rng) -> None:
    """Балконный блок: дверь в балкон и окно рядом на одной стене."""
    by_pair: dict = {}
    for wid, a, b, pair in spans:
        by_pair.setdefault(pair, []).append((wid, a, b))
    sill = round(float(rng.uniform(*cfg.window_sill_m)), 3)
    for pair, items in by_pair.items():
        wid, a, b = max(items, key=lambda x: x[2] - x[1])
        wall = layout.wall(wid)
        room = pair[0] if layout.rooms[pair[0]].kind != "balcony" else pair[1]
        dw = round(float(rng.uniform(0.7, 0.8)), 2)
        ww = round(float(rng.uniform(0.9, 1.5)), 2)
        m, gap = 0.25, 0.15
        if b - a < dw + 2 * m:
            continue
        head = round(min(float(rng.uniform(*cfg.door_head_m)), layout.ceiling_height - 0.1), 3)
        ww = min(ww, b - a - 2 * m - dw - gap)
        door_first = rng.random() < 0.5
        s_door = a + m + dw / 2 if door_first else b - m - dw / 2
        swing = 1 if layout.rooms[room].contains(*wall.point(s_door, wall.thickness / 2 + 0.02)) else -1
        layout.openings.append(Opening(len(layout.openings), wid, "door", round(s_door, 3), dw, 0.0,
                                       head, rooms=tuple(pair), hinge=int(rng.choice((-1, 1))),
                                       swing=swing))
        if ww >= 0.6:
            s_win = s_door + (dw / 2 + gap + ww / 2) * (1 if door_first else -1)
            layout.openings.append(Opening(len(layout.openings), wid, "window", round(s_win, 3), ww,
                                           sill, head, rooms=tuple(pair)))
