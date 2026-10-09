"""Наполнение помещений: мебель, инженерные элементы, колонны, зеркала, окружение.

Всё задаётся ориентированными параллелепипедами (Box). Для эталонного плана
важны только стены и колонны; остальное - помехи, которые конвертер должен
пережить (мебель у стен, шкафы до потолка, радиаторы под окнами, короба).
"""
from __future__ import annotations

import math

import numpy as np

from . import furniture as fu
from .config import InteriorConfig
from .layout import Box, Item, Layout, Room, Wall

RES = 0.05  # шаг сетки занятости помещения, м


class RoomGrid:
    """Сетка занятости одного помещения по его чистому прямоугольнику."""

    def __init__(self, room: Room):
        self.x0, self.y0, self.x1, self.y1 = room.clear
        self.nx = max(1, int(round((self.x1 - self.x0) / RES)))
        self.ny = max(1, int(round((self.y1 - self.y0) / RES)))
        self.occ = np.zeros((self.ny, self.nx), bool)        # ничего ставить нельзя
        self.no_tall = np.zeros((self.ny, self.nx), bool)    # нельзя высокое (окна, зеркала)
        if room.region:                                      # непрямоугольное помещение
            self.occ[:] = True
            for rect in room.region:
                self.occ[self._slices_inner(rect)] = False

    def _slices_inner(self, rect):
        """Клетки, целиком лежащие в прямоугольнике."""
        xa, ya, xb, yb = rect
        i0 = int(math.ceil((xa - self.x0) / RES - 1e-6))
        i1 = int(math.floor((xb - self.x0) / RES + 1e-6))
        j0 = int(math.ceil((ya - self.y0) / RES - 1e-6))
        j1 = int(math.floor((yb - self.y0) / RES + 1e-6))
        return (slice(max(j0, 0), max(min(j1, self.ny), 0)),
                slice(max(i0, 0), max(min(i1, self.nx), 0)))

    def _slices(self, rect):
        xa, ya, xb, yb = rect
        i0 = int(math.floor((xa - self.x0) / RES + 1e-6))
        i1 = int(math.ceil((xb - self.x0) / RES - 1e-6))
        j0 = int(math.floor((ya - self.y0) / RES + 1e-6))
        j1 = int(math.ceil((yb - self.y0) / RES - 1e-6))
        return (slice(max(j0, 0), max(min(j1, self.ny), 0)),
                slice(max(i0, 0), max(min(i1, self.nx), 0)))

    def inside(self, rect) -> bool:
        xa, ya, xb, yb = rect
        return (xa >= self.x0 - 1e-6 and ya >= self.y0 - 1e-6
                and xb <= self.x1 + 1e-6 and yb <= self.y1 + 1e-6)

    def is_free(self, rect, tall: bool) -> bool:
        if not self.inside(rect):
            return False
        sl = self._slices(rect)
        if self.occ[sl].any():
            return False
        return not (tall and self.no_tall[sl].any())

    def mark(self, rect, margin: float = 0.05, grid: str = "occ") -> None:
        xa, ya, xb, yb = rect
        getattr(self, grid)[self._slices((xa - margin, ya - margin, xb + margin, yb + margin))] = True

    def free_points(self, clearance: float) -> np.ndarray:
        """Центры свободных клеток не ближе clearance к стенам и занятым клеткам."""
        from scipy.ndimage import distance_transform_edt

        free = ~self.occ
        padded = np.pad(free, 1, constant_values=False)
        dist = distance_transform_edt(padded)[1:-1, 1:-1] * RES
        jj, ii = np.nonzero(dist >= clearance)
        return np.c_[self.x0 + (ii + 0.5) * RES, self.y0 + (jj + 0.5) * RES]


def wall_rect(wall: Wall, s_a: float, s_b: float, off_a: float, off_b: float):
    """Осевой прямоугольник участка стены: s - вдоль оси, off - по левой нормали."""
    pts = np.array([wall.point(s, o) for s in (s_a, s_b) for o in (off_a, off_b)])
    return (*pts.min(0), *pts.max(0))


def room_side(wall: Wall, room: Room, s: float | None = None) -> int:
    """С какой стороны стены (по левой нормали) лежит помещение (у точки s, если задана)."""
    if s is not None:
        for side in (1, -1):
            p = wall.point(s, side * (wall.thickness / 2 + 0.02))
            if room.contains(*p):
                return side
    x0, y0, x1, y1 = room.cell
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    return 1 if float(np.dot(c - np.asarray(wall.p0), wall.n)) > 0 else -1


def _box_on_side(clear, side: str, pos: float, w: float, d: float, z0: float, h: float,
                 gap: float = 0.02):
    """Прямоугольник и Box предмета, стоящего спиной к стороне чистого прямоугольника."""
    x0, y0, x1, y1 = clear
    if side == "B":
        rect, yaw = (pos, y0 + gap, pos + w, y0 + gap + d), 0.0
    elif side == "T":
        rect, yaw = (pos, y1 - gap - d, pos + w, y1 - gap), math.pi
    elif side == "L":
        rect, yaw = (x0 + gap, pos, x0 + gap + d, pos + w), -math.pi / 2
    else:
        rect, yaw = (x1 - gap - d, pos, x1 - gap, pos + w), math.pi / 2
    cx, cy = (rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2
    return rect, Box((cx, cy, z0 + h / 2), (w, d, h), yaw)


def _wall_point(clear, side: str, pos: float) -> tuple[float, float]:
    """Точка на стороне чистого прямоугольника (pos - координата вдоль стороны)."""
    x0, y0, x1, y1 = clear
    return {"B": (pos, y0), "T": (pos, y1), "L": (x0, pos), "R": (x1, pos)}[side]


def _side_len(clear, side: str) -> tuple[float, float]:
    x0, y0, x1, y1 = clear
    return (x0, x1) if side in "BT" else (y0, y1)


def _local_box(base: Box, du: float, dv: float, z0: float, size) -> Box:
    """Box в системе предмета: du - вдоль, dv - от стены (+ к комнате)."""
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    x = base.center[0] + du * c - dv * s
    y = base.center[1] + du * s + dv * c
    return Box((x, y, z0 + size[2] / 2), tuple(size), base.yaw)


# Каталог предметов у стены: (вид, ширина, глубина, высота, где ставить)
WALL_ITEMS = {
    "wardrobe": ((0.8, 2.4), (0.45, 0.65), (1.8, 2.4), ("room", "corridor")),
    "shelf": ((0.6, 1.2), (0.25, 0.4), (1.6, 2.2), ("room",)),
    "sofa": ((1.6, 2.4), (0.8, 1.0), (0.75, 0.9), ("room",)),
    "bed": ((0.9, 1.8), (1.9, 2.1), (0.45, 0.55), ("room",)),
    "desk": ((1.0, 1.6), (0.6, 0.8), (0.73, 0.77), ("room",)),
    "tv_stand": ((1.2, 2.0), (0.35, 0.45), (0.4, 0.6), ("room",)),
    "fridge": ((0.6, 0.7), (0.6, 0.7), (1.7, 2.0), ("kitchen",)),
    "counter": ((1.5, 3.5), (0.6, 0.6), (0.88, 0.92), ("kitchen",)),
    "bathtub": ((1.5, 1.8), (0.7, 0.8), (0.55, 0.6), ("bath",)),
    "sink": ((0.5, 0.8), (0.45, 0.5), (0.8, 0.9), ("bath", "kitchen")),
    "toilet": ((0.36, 0.4), (0.6, 0.7), (0.75, 0.8), ("bath",)),
    "shoe_rack": ((0.6, 1.2), (0.3, 0.4), (0.4, 1.0), ("corridor",)),
}
MANDATORY = {"kitchen": ("counter", "fridge"), "bath": ("bathtub", "sink", "toilet")}


class Furnisher:
    def __init__(self, cfg: InteriorConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _u(self, r) -> float:
        return float(self.rng.uniform(r[0], r[1]))

    def _refl(self) -> float:
        if self.rng.random() < self.cfg.p_dark_furniture:
            return self._u((0.03, 0.1))
        return self._u((0.15, 0.8))

    def _add(self, layout: Layout, kind: str, label: str, boxes, refl: float, room_id,
             prims=None, transmit: float = 0.0) -> None:
        layout.items.append(Item(len(layout.items), kind, label, list(boxes), float(refl), room_id,
                                 list(prims or []), float(transmit)))

    # --- внешние модели ------------------------------------------------
    def _assets(self) -> dict:
        if not hasattr(self, "_asset_index"):
            from pathlib import Path

            idx = {}
            root = Path(self.cfg.asset_dir) if self.cfg.asset_dir else None
            if root and root.is_dir():
                for sub in root.iterdir():
                    files = sorted(str(f) for f in sub.glob("*")
                                   if f.suffix.lower() in (".obj", ".ply", ".stl", ".glb", ".gltf", ".off"))
                    if sub.is_dir() and files:
                        idx[sub.name] = files
            self._asset_index = idx
        return self._asset_index

    def _maybe_asset(self, kind: str, base: Box):
        """Заменить предмет внешней моделью, вписанной в его габарит (или None)."""
        files = self._assets().get(kind)
        if not files or self.rng.random() >= self.cfg.p_asset:
            return None
        path = files[int(self.rng.integers(len(files)))]
        return [{"type": "mesh", "path": path, "center": list(base.center), "size": list(base.size),
                 "yaw": base.yaw, "z_up": False}]

    # ------------------------------------------------------------------
    def furnish(self, layout: Layout) -> dict[int, RoomGrid]:
        rng = self.rng
        layout.materials = {
            "wall": self._u((0.5, 0.9)), "floor": self._u((0.15, 0.6)),
            "ceiling": self._u((0.7, 0.9)), "door": self._u((0.2, 0.8)),
            "window": 0.8, "glass": 0.05, "mirror": 0.9, "exterior": self._u((0.1, 0.4)),
        }
        self.mess = self._u(self.cfg.mess)
        layout.meta["mess"] = round(self.mess, 3)
        self.bare = bool(rng.random() < self.cfg.p_bare)
        layout.meta["bare"] = self.bare
        self._set_doors(layout)
        grids = {r.id: RoomGrid(r) for r in layout.rooms}
        for rect in layout.meta.get("reserved", []):        # лестница, проём в перекрытии
            for g in grids.values():
                g.mark(tuple(rect), 0.4)
        self._reserve_openings(layout, grids)
        if self.cfg.door_frames:
            self._door_frames(layout)
        for room in layout.rooms:
            g = grids[room.id]
            self._wall_fixtures(layout, room, g)
            self._columns(layout, room, g)
            self._risers(layout, room, g)
            if self.cfg.furniture and self.bare:
                self._bare_fixtures(layout, room, g)
            elif self.cfg.furniture:
                self._furniture(layout, room, g)
                self._extras(layout, room, g)
            if not self.bare:
                self._curtains(layout, room)
            self._clutter(layout, room, g, 0.3 if self.bare else 1.0)
            if self.cfg.furniture and not self.bare:
                self._mess(layout, room, g)
                self._scans(layout, room, g)
        if self.cfg.baseboards and not self.bare:
            for room in layout.rooms:
                self._baseboards(layout, room)
        if self.cfg.exterior_ground:
            self._exterior(layout)
        return grids

    def _set_doors(self, layout: Layout) -> None:
        for o in layout.openings:
            if o.kind != "door":
                continue
            if self.rng.random() < self.cfg.p_closed_door:
                o.angle_deg = 0.0
            else:
                o.angle_deg = round(self._u(self.cfg.door_open_deg), 1)

    def _reserve_openings(self, layout: Layout, grids) -> None:
        for o in layout.openings:
            wall = layout.wall(o.wall_id)
            for rid in o.rooms:
                room = layout.rooms[rid]
                side = room_side(wall, room, o.s)
                face = side * wall.thickness / 2
                if o.kind == "window":
                    rect = wall_rect(wall, o.s - o.width / 2 - 0.1, o.s + o.width / 2 + 0.1,
                                     face, face + side * 0.6)
                    grids[rid].mark(rect, 0.0, "no_tall")
                else:
                    depth = max(o.width, 0.9)
                    rect = wall_rect(wall, o.s - o.width / 2 - 0.1, o.s + o.width / 2 + 0.1,
                                     face, face + side * depth)
                    grids[rid].mark(rect, 0.0)

    # --- инженерные элементы на стенах ----------------------------------
    def _wall_fixtures(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        cfg, rng = self.cfg, self.rng
        if cfg.radiators:
            for o in layout.openings:
                if o.kind != "window" or room.id not in o.rooms or o.z0 < 0.45:
                    continue
                wall = layout.wall(o.wall_id)
                side = room_side(wall, room, o.s)
                face = side * wall.thickness / 2
                w = min(o.width, 1.2)
                h = min(0.5, o.z0 - 0.15)
                if cfg.procedural:
                    boxes = fu.radiator(wall, o.s, face, side, w, h, rng)
                else:
                    center = wall.point(o.s, face + side * 0.08)
                    boxes = [Box((*center, 0.1 + h / 2), (w, 0.08, h), wall.yaw)]
                self._add(layout, "radiator", "fixture", boxes, self._u((0.6, 0.9)), room.id)
                g.mark(wall_rect(wall, o.s - w / 2, o.s + w / 2, face, face + side * 0.15), 0.0)

        sides = ["B", "T", "L", "R"]
        if rng.random() < cfg.p_soffit and len(room.rects()) == 1:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            d, h = self._u((0.2, 0.5)), self._u((0.2, 0.4))
            _, box = _box_on_side(room.clear, side, a, b - a, d, room.ceiling_z - h, h, gap=0.0)
            self._add(layout, "soffit", "fixture", [box], layout.materials["ceiling"], room.id)

        if rng.random() < cfg.p_pilaster:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            w, d = self._u((0.3, 0.5)), self._u((0.1, 0.25))
            if b - a > w + 0.4:
                pos = self._u((a + 0.2, b - 0.2 - w))
                rect, box = _box_on_side(room.clear, side, pos, w, d, 0.0, layout.ceiling_height,
                                         gap=-0.01)
                if not g.occ[g._slices(rect)].any():
                    self._add(layout, "pilaster", "column", [box], layout.materials["wall"],
                              room.id)
                    g.mark(rect)

        p_mirror = cfg.p_mirror_bath if room.kind == "bath" else cfg.p_mirror_other
        if rng.random() < p_mirror:
            side = sides[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            w, h = self._u((0.5, 0.8)), self._u((0.6, 0.9))
            if b - a > w + 0.2:
                pos = self._u((a + 0.1, b - 0.1 - w))
                z0 = self._u((1.0, 1.2))
                rect, box = _box_on_side(room.clear, side, pos, w, 0.005, z0, h, gap=0.003)
                if not g.occ[g._slices(rect)].any():
                    self._add(layout, "mirror", "mirror", [box], layout.materials["mirror"],
                              room.id)
                    g.mark(rect, 0.4, "no_tall")

    def _columns(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        if room.area < self.cfg.column_min_area_m2 or self.rng.random() >= self.cfg.p_column:
            return
        pts = g.free_points(0.8)
        if len(pts) == 0:
            return
        x, y = pts[int(self.rng.integers(len(pts)))]
        s = self._u((0.3, 0.5))
        rect = (x - s / 2, y - s / 2, x + s / 2, y + s / 2)
        box = Box((x, y, layout.ceiling_height / 2), (s, s, layout.ceiling_height), 0.0)
        self._add(layout, "column", "column", [box], layout.materials["wall"], room.id)
        g.mark(rect, 0.3)

    # --- мебель ----------------------------------------------------------
    def _furniture(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        rng = self.rng
        kinds = list(MANDATORY.get(room.kind, ()))
        n_extra = int(rng.poisson(room.area * self.cfg.furniture_per_m2))
        allowed = [k for k, v in WALL_ITEMS.items() if room.kind in v[3]]
        if room.kind == "kitchen":
            allowed += [k for k, v in WALL_ITEMS.items() if "room" in v[3] and k != "bed"]
        if allowed:
            kinds += [allowed[int(rng.integers(len(allowed)))] for _ in range(n_extra)]
        for kind in kinds:
            self._place_wall_item(layout, room, g, kind)
        if room.kind in ("room", "kitchen") and room.area > 8 and rng.random() < 0.6:
            self._place_table(layout, room, g)

    def _place_wall_item(self, layout: Layout, room: Room, g: RoomGrid, kind: str) -> bool:
        rng = self.rng
        wr, dr, hr, _ = WALL_ITEMS[kind]
        w, d, h = self._u(wr), self._u(dr), self._u(hr)
        if kind == "wardrobe" and rng.random() < self.cfg.p_wardrobe_to_ceiling:
            h = room.ceiling_z - 0.005
        h = min(h, room.ceiling_z - 0.02)
        tall = h > 1.0
        refl = self._refl()
        for _ in range(15):
            side = ("B", "T", "L", "R")[int(rng.integers(4))]
            a, b = _side_len(room.clear, side)
            if b - a < w + 0.04:
                continue
            pos = self._u((a + 0.02, b - 0.02 - w))
            rect, base = _box_on_side(room.clear, side, pos, w, d, 0.0, h)
            if not g.is_free(rect, tall):
                continue
            prims = self._maybe_asset(kind, base)
            boxes = [] if prims else self._compose(kind, base, w, d, h)
            self._add(layout, kind, "furniture", boxes, refl, room.id, prims)
            g.mark(rect)
            if kind == "counter" and room.ceiling_z > 2.3:
                upper = _local_box(base, 0.0, -d / 2 + 0.175, 1.45, (w, 0.35, 0.7))
                self._add(layout, "upper_cabinet", "furniture", [upper], refl, room.id)
            return True
        return False

    def _compose(self, kind: str, base: Box, w: float, d: float, h: float) -> list[Box]:
        """Составные предметы - несколько параллелепипедов."""
        if self.cfg.procedural:
            make = {"sofa": fu.sofa, "bed": fu.bed, "shelf": fu.shelf, "wardrobe": fu.wardrobe,
                    "tv_stand": fu.tv_stand, "counter": fu.counter, "shoe_rack": fu.shoe_rack,
                    "desk": fu.table}.get(kind)
            if make is not None:
                return make(base, self.rng)
        if kind == "sofa":
            seat = _local_box(base, 0, 0.1, 0.0, (w, d - 0.2, 0.45))
            back = _local_box(base, 0, -d / 2 + 0.1, 0.0, (w, 0.2, h))
            return [seat, back]
        if kind == "bed":
            mattress = _local_box(base, 0, 0, 0.0, (w, d, h))
            head = _local_box(base, 0, -d / 2 + 0.03, 0.0, (w, 0.06, 1.0))
            return [mattress, head]
        if kind == "desk":
            return _table_boxes(base, w, d, h)
        if kind == "toilet":
            bowl = _local_box(base, 0, 0.08, 0.0, (w, d - 0.16, 0.42))
            tank = _local_box(base, 0, -d / 2 + 0.09, 0.0, (w, 0.18, h))
            return [bowl, tank]
        return [base]

    def _place_table(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        rng = self.rng
        w, d, h = self._u((1.0, 1.8)), self._u((0.7, 1.0)), self._u((0.72, 0.76))
        pts = g.free_points(max(w, d) / 2 + 0.5)
        if len(pts) == 0:
            return
        x, y = pts[int(rng.integers(len(pts)))]
        yaw = float(rng.uniform(-0.3, 0.3)) + (math.pi / 2 if rng.random() < 0.5 else 0.0)
        base = Box((x, y, h / 2), (w, d, h), yaw)
        prims = self._maybe_asset("dining_table", base)
        boxes = [] if prims else (fu.table(base, rng) if self.cfg.procedural
                                  else _table_boxes(base, w, d, h))
        self._add(layout, "dining_table", "furniture", boxes, self._refl(), room.id, prims)
        for k in range(int(rng.integers(0, 5))):
            du = (k % 2 * 2 - 1) * w / 4
            dv = (1 if k < 2 else -1) * (d / 2 + 0.3)
            if self.cfg.procedural:
                # стул «спиной» от стола: ось dv стула смотрит к столу
                c = _local_box(base, du, dv, 0.0, (0.44, 0.42, 0.9)).center
                cyaw = base.yaw + (0.0 if dv < 0 else math.pi)
                cbase = Box((c[0], c[1], 0.45), (0.44, 0.42, 0.9), cyaw)
                cprims = self._maybe_asset("chair", cbase)
                cboxes = [] if cprims else fu.chair(cbase, rng)
                self._add(layout, "chair", "furniture", cboxes, self._refl(), room.id, cprims)
            else:
                seat = _local_box(base, du, dv, 0.0, (0.45, 0.45, 0.45))
                back = _local_box(base, du, dv + np.sign(dv) * 0.2, 0.0, (0.45, 0.05, 0.9))
                self._add(layout, "chair", "furniture", [seat, back], self._refl(), room.id)
        r = max(w, d) / 2 + 0.6
        g.mark((x - r, y - r, x + r, y + r), 0.0)

    def _clutter(self, layout: Layout, room: Room, g: RoomGrid, scale: float = 1.0) -> None:
        n = int(self.rng.poisson(room.area * self.cfg.clutter_per_m2 * scale))
        for _ in range(n):
            s = self._u((0.25, 0.6))
            pts = g.free_points(s)
            if len(pts) == 0:
                return
            x, y = pts[int(self.rng.integers(len(pts)))]
            h = self._u((0.2, 0.7))
            box = Box((x, y, h / 2), (s, self._u((0.25, 0.6)), h), float(self.rng.uniform(0, math.pi)))
            self._add(layout, "box", "clutter", [box], self._refl(), room.id)
            g.mark((x - s / 2, y - s / 2, x + s / 2, y + s / 2), 0.1)

    # --- беспорядок: мягкие и неровные вещи -----------------------------------
    def _mess(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        """Куртка на коробке, плед на диване, сумки, кучи белья, стопки коробок, доски у стены.

        Всё это - не параллелепипеды и не мебель из каталога; уровень беспорядка
        self.mess (0..1) задаётся на сцену и масштабирует вероятности и количества."""
        m, rng = self.mess, self.rng
        if m <= 0 or room.kind == "bath":
            return
        # ткань на опорах: стулья, диваны, кровати, коробки, столы
        for it in [it for it in layout.items if it.room_id == room.id]:
            if it.kind not in ("chair", "sofa", "bed", "box", "table", "desk") or not it.boxes:
                continue
            if rng.random() >= 0.35 * m:
                continue
            if it.kind == "chair":
                support = max(it.boxes, key=lambda b: b.center[2] + b.size[2] / 2)
            else:
                big = [b for b in it.boxes if b.size[0] * b.size[1] >= 0.1] or it.boxes
                support = max(big, key=lambda b: b.center[2] + b.size[2] / 2)
            prim = fu.drape(support, rng)
            self._add(layout, "cloth", "clutter", [], self._refl(), room.id, [prim])
        # одежда на крючках у стены (прихожая - чаще)
        lam = (2.0 if room.kind == "corridor" else 0.4) * m
        for _ in range(int(rng.poisson(lam))):
            side = ("B", "T", "L", "R")[int(rng.integers(4))]
            lo, hi = _side_len(room.clear, side)
            w = self._u((0.35, 0.55))
            if hi - lo < w + 0.2:
                continue
            pos = self._u((lo + 0.1, hi - 0.1 - w))
            rect, box = _box_on_side(room.clear, side, pos, w, 0.25, 0.0, 1.8, gap=0.0)
            if not g.inside(rect) or g.occ[g._slices(rect)].any():
                continue
            along = np.array([math.cos(box.yaw), math.sin(box.yaw), 0.0])
            normal = np.array([-math.sin(box.yaw), math.cos(box.yaw), 0.0])
            top = np.array([*_wall_point(room.clear, side, pos + w / 2), self._u((1.6, 1.85))])
            self._add(layout, "jacket", "clutter", [], self._refl(), room.id,
                      [fu.hanging_cloth(top + 0.02 * normal, along, normal, rng, width=w)])
            g.mark(rect, 0.0)
        # сумки, рюкзаки, кучи белья на полу
        for _ in range(int(rng.poisson(0.08 * room.area * m))):
            s = (self._u((0.25, 0.6)), self._u((0.2, 0.45)), self._u((0.15, 0.45)))
            pts = g.free_points(max(s[0], s[1]) / 2 + 0.05)
            if not len(pts):
                break
            x, y = pts[int(rng.integers(len(pts)))]
            self._add(layout, "bag", "clutter", [], self._refl(), room.id,
                      [fu.blob((float(x), float(y)), s, rng)])
            g.mark((x - s[0] / 2, y - s[0] / 2, x + s[0] / 2, y + s[0] / 2), 0.05)
        # стопки коробок (переезд, хранение)
        for _ in range(int(rng.poisson(0.03 * room.area * m))):
            w, d = self._u((0.3, 0.6)), self._u((0.25, 0.5))
            pts = g.free_points(max(w, d) / 2 + 0.05)
            if not len(pts):
                break
            x, y = pts[int(rng.integers(len(pts)))]
            yaw, z, boxes = float(rng.uniform(0, math.pi)), 0.0, []
            for k in range(int(rng.integers(2, 5))):
                h = self._u((0.2, 0.45))
                bw, bd = w * self._u((0.75, 1.0)), d * self._u((0.75, 1.0))
                ox, oy = rng.normal(0, 0.03, 2)
                boxes.append(Box((x + ox, y + oy, z + h / 2), (bw, bd, h),
                                 yaw + float(rng.normal(0, 0.12))))
                z += h
            self._add(layout, "box_stack", "clutter", boxes, self._refl(), room.id)
            r = max(w, d) / 2 + 0.05
            g.mark((x - r, y - r, x + r, y + r), 0.05)
        # доски / листы, прислонённые к стене
        if room.kind in ("room", "corridor") and rng.random() < 0.15 * m:
            side = ("B", "T", "L", "R")[int(rng.integers(4))]
            lo, hi = _side_len(room.clear, side)
            if hi - lo > 1.6:
                pos = self._u((lo + 0.3, hi - 1.3))
                rect, box = _box_on_side(room.clear, side, pos, 1.0, 0.7, 0.0, 2.0, gap=0.0)
                if g.inside(rect) and not g.occ[g._slices(rect)].any():
                    along = np.array([math.cos(box.yaw), math.sin(box.yaw)])
                    normal = np.array([-math.sin(box.yaw), math.cos(box.yaw)])
                    base = np.asarray(_wall_point(room.clear, side, pos + 0.5)) + 0.01 * normal
                    prim = fu.leaning_board(base, along, normal, rng)
                    self._add(layout, "board", "clutter", [], self._refl(), room.id, [prim])
                    g.mark(rect, 0.0)

    # --- отсканированные предметы (Google Scanned Objects и др.) ---------------
    def _scan_index(self) -> dict:
        if not hasattr(self, "_scan_idx"):
            from pathlib import Path

            idx = {}
            root = Path(self.cfg.scan_dir) if self.cfg.scan_dir else None
            if root and root.is_dir():
                for sub in sorted(root.iterdir()):
                    files = sorted(str(f) for f in sub.glob("*.obj")) if sub.is_dir() else []
                    if files:
                        idx[sub.name] = files
            self._scan_idx = idx
        return self._scan_idx

    def _scan_prim(self, role: str, x: float, y: float, z: float, max_xy: float | None = None):
        from .meshes import _load

        files = self._scan_index().get(role)
        if not files:
            return None
        path = files[int(self.rng.integers(len(files)))]
        v, _ = _load(path)
        ext = v.max(0) - v.min(0)
        if max_xy is not None and max(ext[0], ext[1]) > max_xy:
            return None
        return {"type": "mesh", "path": path, "center": [float(x), float(y), float(z)],
                "yaw": float(self.rng.uniform(0, 2 * math.pi)), "native": True, "z_up": True,
                "size": [float(ext[0]), float(ext[1]), float(ext[2])]}

    def _scans(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        """Реальные сканы в натуральную величину: обувь у входа, вещи на столах и полках,
        сумки и игрушки на полу. Роли - подкаталоги scan_dir (см. simscan assets)."""
        idx, rng, m = self._scan_index(), self.rng, self.mess
        if not idx or m <= 0:
            return
        lam = self.cfg.scans_per_m2 * room.area * m
        if room.kind == "corridor" and "shoes" in idx:
            for _ in range(int(rng.poisson(1.5 + 2 * m))):              # пары обуви
                pts = g.free_points(0.25)
                if not len(pts):
                    break
                x, y = pts[int(rng.integers(len(pts)))]
                pr = self._scan_prim("shoes", x, y, 0.0)
                if pr is None:
                    break
                twin = dict(pr, center=[x + 0.12 * math.cos(pr["yaw"]),
                                        y + 0.12 * math.sin(pr["yaw"]), 0.0],
                            yaw=pr["yaw"] + float(rng.normal(0, 0.15)))
                self._add(layout, "scan:shoes", "clutter", [], self._refl(), room.id, [pr, twin])
                g.mark((x - 0.25, y - 0.25, x + 0.25, y + 0.25), 0.0)
        tops = [b for it in layout.items if it.room_id == room.id and it.kind in
                ("table", "desk", "counter", "tv_stand", "shelf", "shoe_rack", "nightstand",
                 "dresser")
                for b in [max(it.boxes, key=lambda b: b.center[2] + b.size[2] / 2)] if it.boxes]
        if tops and "tabletop" in idx:
            for _ in range(int(rng.poisson(lam))):
                b = tops[int(rng.integers(len(tops)))]
                if b.center[2] + b.size[2] / 2 > 1.9:
                    continue
                du, dv = (rng.uniform(-0.5, 0.5, 2) * np.maximum(np.array(b.size[:2]) - 0.15, 0))
                c, s = math.cos(b.yaw), math.sin(b.yaw)
                x, y = b.center[0] + du * c - dv * s, b.center[1] + du * s + dv * c
                pr = self._scan_prim("tabletop", x, y, b.center[2] + b.size[2] / 2,
                                     max_xy=min(b.size[0], b.size[1]))
                if pr is not None:
                    self._add(layout, "scan:tabletop", "clutter", [], self._refl(), room.id, [pr])
        if "floor" in idx and room.kind != "bath":
            for _ in range(int(rng.poisson(0.5 * lam))):
                pts = g.free_points(0.3)
                if not len(pts):
                    break
                x, y = pts[int(rng.integers(len(pts)))]
                pr = self._scan_prim("floor", x, y, 0.0)
                if pr is None:
                    break
                self._add(layout, "scan:floor", "clutter", [], self._refl(), room.id, [pr])
                r = max(pr["size"][:2]) / 2 + 0.05
                g.mark((x - r, y - r, x + r, y + r), 0.0)

    # --- архитектурные детали и предметы, которых нет в коробочной модели ----
    def _door_frames(self, layout: Layout) -> None:
        """Дверная коробка внутри проёма и наличники на обеих гранях стены."""
        rng = self.rng
        for o in layout.openings:
            if o.kind == "window" or (o.kind == "passage" and rng.random() < 0.5):
                continue
            wall = layout.wall(o.wall_id)
            t, yaw = wall.thickness, wall.yaw
            sa, sb = o.s - o.width / 2, o.s + o.width / 2
            boxes = []

            def piece(s0, s1, off, depth, z0, z1):
                c = wall.point((s0 + s1) / 2, off)
                boxes.append(Box((c[0], c[1], (z0 + z1) / 2), (s1 - s0, depth, z1 - z0), yaw))

            lw, aw, ad = 0.03, 0.07, 0.012
            piece(sa, sa + lw, 0.0, t + 0.01, 0.0, o.z1)                 # коробка
            piece(sb - lw, sb, 0.0, t + 0.01, 0.0, o.z1)
            piece(sa, sb, 0.0, t + 0.01, o.z1 - lw, o.z1)
            for side in (-1, 1):                                         # наличники
                off = side * (t / 2 + ad / 2)
                piece(sa - aw + 0.015, sa + 0.015, off, ad, 0.0, o.z1 + aw - 0.015)
                piece(sb - 0.015, sb + aw - 0.015, off, ad, 0.0, o.z1 + aw - 0.015)
                piece(sa - aw + 0.015, sb + aw - 0.015, off, ad, o.z1 - 0.015, o.z1 + aw - 0.015)
            self._add(layout, "door_frame", "door", boxes, layout.materials.get("door", 0.5), None)

    def _curtains(self, layout: Layout, room: Room) -> None:
        """Шторы (непрозрачные) и тюль (пропускает часть лучей) на окнах жилых комнат и кухни."""
        cfg, rng = self.cfg, self.rng
        if room.kind not in ("room", "kitchen"):
            return
        for o in layout.openings:
            if o.kind != "window" or room.id not in o.rooms:
                continue
            wall = layout.wall(o.wall_id)
            side = room_side(wall, room, o.s)
            face = side * wall.thickness / 2
            rail = min(room.ceiling_z - 0.03, o.z1 + self._u((0.1, 0.3)))
            s0 = o.s - o.width / 2 - self._u((0.1, 0.4))
            s1 = o.s + o.width / 2 + self._u((0.1, 0.4))
            bottom = max(0.01, o.z0 - 0.02) if room.kind == "kitchen" and rng.random() < 0.6 \
                else self._u((0.01, 0.04))
            if rng.random() < cfg.p_tulle:
                off = face + side * self._u((0.06, 0.1))
                prim = {"type": "curtain", "p0": wall.point(s0, off).tolist(),
                        "p1": wall.point(s1, off).tolist(), "z0": bottom, "z1": rail,
                        "amp": self._u((0.015, 0.035)), "period": self._u((0.08, 0.14))}
                self._add(layout, "tulle", "curtain", [], self._u((0.4, 0.8)), room.id, [prim],
                          transmit=self._u(cfg.tulle_transmit))
            if rng.random() < cfg.p_curtains:
                off = face + side * self._u((0.11, 0.17))
                span = s1 - s0
                closed = rng.random() < 0.15
                prims = []
                for k, (a, sign) in enumerate(((s0, 1), (s1, -1))):
                    frac = 0.5 if closed else self._u((0.12, 0.45))
                    b = a + sign * span * frac
                    prims.append({"type": "curtain", "p0": wall.point(min(a, b), off).tolist(),
                                  "p1": wall.point(max(a, b), off).tolist(), "z0": bottom,
                                  "z1": rail, "amp": self._u((0.03, 0.06)),
                                  "period": self._u((0.12, 0.2))})
                self._add(layout, "curtains", "curtain", [], self._u((0.1, 0.7)), room.id, prims)

    def _risers(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        """Стояки в углу санузла: трубы или зашитый короб до потолка."""
        rng = self.rng
        if room.kind != "bath" or rng.random() >= self.cfg.p_risers:
            return
        x0, y0, x1, y1 = room.clear
        cx = x0 if rng.random() < 0.5 else x1
        cy = y0 if rng.random() < 0.5 else y1
        sx, sy = (1 if cx == x0 else -1), (1 if cy == y0 else -1)
        H = room.ceiling_z
        if rng.random() < 0.35:
            a = self._u((0.25, 0.45))
            box = Box((cx + sx * a / 2, cy + sy * a / 2, H / 2), (a, a, H), 0.0)
            self._add(layout, "riser_box", "fixture", [box], self._u((0.5, 0.9)), room.id)
            g.mark((min(cx, cx + sx * a), min(cy, cy + sy * a), max(cx, cx + sx * a),
                    max(cy, cy + sy * a)), 0.05)
            return
        prims = []
        for k in range(int(rng.integers(1, 3))):
            r = self._u((0.025, 0.055))
            px = cx + sx * (0.06 + r + k * 0.12)
            py = cy + sy * (0.06 + r)
            prims.append({"type": "cylinder", "center": [px, py, H / 2], "radius": r, "height": H})
        self._add(layout, "riser", "fixture", [], self._u((0.3, 0.8)), room.id, prims)
        g.mark((min(cx, cx + sx * 0.4), min(cy, cy + sy * 0.25), max(cx, cx + sx * 0.4),
                max(cy, cy + sy * 0.25)), 0.0)

    def _bare_fixtures(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        """Квартира под отделку: сантехника в санузле и патрон с лампой на проводе под потолком."""
        rng = self.rng
        if room.kind == "bath":
            for kind in MANDATORY["bath"]:
                self._place_wall_item(layout, room, g, kind)
        if room.kind == "balcony" or rng.random() < 0.2:
            return
        a, b, c, d = max(room.rects(), key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        x, y = (a + c) / 2 + rng.normal(0, 0.15), (b + d) / 2 + rng.normal(0, 0.15)
        drop = self._u((0.05, 0.4))
        z = room.ceiling_z - drop
        prims = [{"type": "cylinder", "center": [x, y, z], "radius": self._u((0.03, 0.05)), "height": 0.1},
                 {"type": "cylinder", "center": [x, y, (z + room.ceiling_z) / 2], "radius": 0.004,
                  "height": drop}]
        self._add(layout, "lamp", "fixture", [], self._u((0.5, 0.9)), room.id, prims)

    def _extras(self, layout: Layout, room: Room, g: RoomGrid) -> None:
        """Растения, вешалки, светильники, телевизоры и картины на стенах."""
        cfg, rng = self.cfg, self.rng
        if not cfg.procedural:
            return
        if room.kind in ("room", "kitchen") and rng.random() < cfg.p_plants:
            pts = g.free_points(0.4)
            if len(pts):
                x, y = pts[int(rng.integers(len(pts)))]
                boxes, prims = fu.plant(float(x), float(y), rng)
                self._add(layout, "plant", "furniture", boxes, self._u((0.1, 0.4)), room.id, prims)
                g.mark((x - 0.3, y - 0.3, x + 0.3, y + 0.3), 0.0)
        if room.kind == "corridor" and rng.random() < 0.4:
            pts = g.free_points(0.35)
            if len(pts):
                x, y = pts[int(rng.integers(len(pts)))]
                boxes, prims = fu.coat_rack(float(x), float(y), rng)
                self._add(layout, "coat_rack", "furniture", boxes, self._refl(), room.id, prims)
                g.mark((x - 0.3, y - 0.3, x + 0.3, y + 0.3), 0.0)
        if cfg.lamps and room.kind != "balcony":
            a, b, c, d = max(room.rects(), key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            x, y = (a + c) / 2, (b + d) / 2
            if rng.random() < 0.5:
                h = self._u((0.1, 0.2))
                z = room.ceiling_z - self._u((0.3, 0.8))
                prims = [{"type": "cylinder", "center": [x, y, z], "radius": self._u((0.12, 0.25)),
                          "height": h},
                         {"type": "cylinder", "center": [x, y, (z + h / 2 + room.ceiling_z) / 2],
                          "radius": 0.006, "height": room.ceiling_z - z - h / 2}]
            else:
                prims = [{"type": "cylinder", "center": [x, y, room.ceiling_z - 0.04],
                          "radius": self._u((0.15, 0.3)), "height": 0.08}]
            self._add(layout, "lamp", "fixture", [], self._u((0.5, 0.9)), room.id, prims)
        if room.kind == "room" and rng.random() < cfg.p_wall_decor:
            side = ("B", "T", "L", "R")[int(rng.integers(4))]
            lo, hi = _side_len(room.clear, side)
            tv = rng.random() < 0.5
            w = self._u((0.9, 1.4)) if tv else self._u((0.4, 1.0))
            h = w * 0.58 if tv else self._u((0.3, 0.8))
            depth = self._u((0.04, 0.08)) if tv else 0.02
            if hi - lo > w + 0.4:
                pos = self._u((lo + 0.2, hi - 0.2 - w))
                z0 = self._u((1.0, 1.4)) if tv else self._u((1.2, 1.7))
                rect, box = _box_on_side(room.clear, side, pos, w, depth, z0, h, gap=0.005)
                if not g.occ[g._slices(rect)].any():
                    self._add(layout, "tv" if tv else "picture", "furniture", [box],
                              self._refl(), room.id)

    # --- плинтусы ----------------------------------------------------------
    def _door_gaps(self, layout: Layout, vertical: bool, coord: float) -> list:
        """Интервалы дверей и арок на линии грани (x = coord или y = coord)."""
        gaps = []
        for o in layout.openings:
            if o.z0 > 0.05:
                continue
            wall = layout.wall(o.wall_id)
            if (abs(wall.u[0]) < 0.5) != vertical:
                continue
            line = wall.p0[0] if vertical else wall.p0[1]
            if abs(line - coord) > wall.thickness / 2 + 0.02:
                continue
            c = wall.point(o.s)[1 if vertical else 0]
            gaps.append((c - o.width / 2, c + o.width / 2))
        return gaps

    def _baseboards(self, layout: Layout, room: Room) -> None:
        x0, y0, x1, y1 = room.clear
        h, d = 0.07, 0.015
        refl = self._u((0.3, 0.9))
        boxes = []
        if room.wall_edges:
            for ex0, ey0, ex1, ey1, nx, ny in room.wall_edges:
                vertical = abs(ex0 - ex1) < 1e-9
                a, b = (ey0, ey1) if vertical else (ex0, ex1)
                coord = ex0 if vertical else ey0
                for sa, sb in _subtract_intervals((min(a, b), max(a, b)),
                                                  self._door_gaps(layout, vertical, coord)):
                    if sb - sa < 0.1:
                        continue
                    if vertical:
                        c = (coord + nx * d / 2, (sa + sb) / 2)
                        box = Box((c[0], c[1], h / 2), (sb - sa, d, h), math.pi / 2)
                    else:
                        c = ((sa + sb) / 2, coord + ny * d / 2)
                        box = Box((c[0], c[1], h / 2), (sb - sa, d, h), 0.0)
                    boxes.append(box)
            if boxes:
                self._add(layout, "baseboard", "fixture", boxes, refl, room.id)
            return
        for side in ("B", "T", "L", "R"):
            a, b = _side_len(room.clear, side)
            coord = {"B": y0, "T": y1, "L": x0, "R": x1}[side]
            gaps = []
            for o in layout.openings:
                if o.z0 > 0.05:
                    continue
                wall = layout.wall(o.wall_id)
                vertical_wall = abs(wall.u[0]) < 0.5
                if vertical_wall != (side in "LR"):
                    continue
                line = wall.p0[0] if vertical_wall else wall.p0[1]
                if abs(line - coord) > wall.thickness / 2 + 0.02:
                    continue
                c = wall.point(o.s)[1 if vertical_wall else 0]
                gaps.append((c - o.width / 2, c + o.width / 2))
            for sa, sb in _subtract_intervals((a, b), gaps):
                if sb - sa < 0.1:
                    continue
                _, box = _box_on_side(room.clear, side, sa, sb - sa, d, 0.0, h, gap=0.0)
                boxes.append(box)
        if boxes:
            self._add(layout, "baseboard", "fixture", boxes, refl, room.id)

    # --- окружение за окнами -------------------------------------------
    def _exterior(self, layout: Layout) -> None:
        rng = self.rng
        x0, y0, x1, y1 = layout.bbox()
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        drop = self._u(self.cfg.ground_drop_m)
        refl = layout.materials["exterior"]
        ground = Box((cx, cy, -drop - 0.35), (160.0, 160.0, 0.2), 0.0)
        self._add(layout, "ground", "exterior", [ground], refl, None)
        lo, hi = self.cfg.neighbour_buildings
        for _ in range(int(rng.integers(lo, hi + 1))):
            ang = rng.uniform(0, 2 * math.pi)
            sx, sy, sz = rng.uniform(8, 30), rng.uniform(8, 30), rng.uniform(10, 40)
            dist = rng.uniform(15, 40) + math.hypot(x1 - x0, y1 - y0) / 2 + math.hypot(sx, sy) / 2
            c = (cx + dist * math.cos(ang), cy + dist * math.sin(ang), -drop - 0.25 + sz / 2)
            self._add(layout, "building", "exterior", [Box(c, (sx, sy, sz), float(ang))],
                      self._u((0.1, 0.6)), None)


def _table_boxes(base: Box, w: float, d: float, h: float) -> list[Box]:
    top = _local_box(base, 0, 0, h - 0.03, (w, d, 0.03))
    legs = [_local_box(base, su * (w / 2 - 0.05), sv * (d / 2 - 0.05), 0.0, (0.05, 0.05, h - 0.03))
            for su in (-1, 1) for sv in (-1, 1)]
    return [top, *legs]


def _subtract_intervals(span, gaps):
    a, b = span
    out = []
    cur = a
    for ga, gb in sorted(gaps):
        if gb <= cur or ga >= b:
            continue
        if ga > cur:
            out.append((cur, ga))
        cur = max(cur, gb)
    if cur < b:
        out.append((cur, b))
    return out
