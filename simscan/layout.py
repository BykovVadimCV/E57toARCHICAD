"""Планировка этажа: модель данных и процедурный генератор.

Система координат планировки (layout): метры, X вправо, Y вверх по плану,
Z вверх, чистый пол на Z = 0. Планировка всегда Manhattan в своей СК;
поворот и сдвиг в СК объекта задаются отдельно (generate.py).

Генератор:
  1. прямоугольник здания режется BSP на ячейки-помещения;
  2. с вероятностью p_l_shape угловая ячейка удаляется (Г-образный план);
  3. границы ячеек раскладываются на элементарные отрезки: с двух сторон
     ячейки -> внутренняя стена (ось на границе), с одной -> наружная
     (внутренняя грань на границе, тело наружу);
  4. двери по остовному дереву смежности помещений + случайные лишние,
     входная дверь и окна на наружных стенах.

Формат JSON (Layout.to_dict) - это и есть точка подключения внешних
генераторов планировок (datasetgen): достаточно собрать такой же словарь.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import LayoutConfig

EPS = 1e-6


@dataclass
class Wall:
    id: int
    p0: tuple[float, float]          # ось стены, начало
    p1: tuple[float, float]          # ось стены, конец
    thickness: float
    kind: str                        # "exterior" | "interior" | "parapet" (ограждение балкона)
    outward: int = 0                 # наружная: +1/-1 - сторона улицы по левой нормали
    ext0: float = 0.0                # удлинение тела за p0 (наружные углы, стыки)
    ext1: float = 0.0                # удлинение тела за p1
    height: float | None = None      # None - до перекрытия
    role: str = ""                   # "balcony_interface" - стена между комнатой и балконом

    @property
    def length(self) -> float:
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    @property
    def u(self) -> np.ndarray:
        d = np.subtract(self.p1, self.p0)
        return d / np.linalg.norm(d)

    @property
    def n(self) -> np.ndarray:
        u = self.u
        return np.array([-u[1], u[0]])

    @property
    def yaw(self) -> float:
        u = self.u
        return float(np.arctan2(u[1], u[0]))

    def point(self, s: float, off: float = 0.0) -> np.ndarray:
        """Точка на расстоянии s вдоль оси от p0 и off по левой нормали."""
        return np.asarray(self.p0, float) + self.u * s + self.n * off


@dataclass
class Opening:
    id: int
    wall_id: int
    kind: str                        # "door" | "passage" | "window"
    s: float                         # центр проёма вдоль оси стены от p0
    width: float
    z0: float                        # низ проёма (0 у дверей, подоконник у окон)
    z1: float                        # верх проёма
    rooms: tuple[int, ...] = ()
    hinge: int = 0                   # -1: петли у начала проёма (меньшее s), +1: у конца
    swing: int = 0                   # сторона открывания по нормали стены
    angle_deg: float = 0.0           # 0 = закрыта
    leaf_thickness: float = 0.04


@dataclass
class Room:
    id: int
    cell: tuple[float, float, float, float]     # xmin, ymin, xmax, ymax по осям стен
    clear: tuple[float, float, float, float]    # чистый прямоугольник (консервативно)
    kind: str = "room"                          # room | corridor | bath | kitchen | balcony
    ceiling_z: float = 0.0                      # низ потолка (подвесной ниже перекрытия)
    scanned: bool = True
    # Непрямоугольное помещение: точное покрытие чистой площади прямоугольниками
    # (пусто - помещение совпадает с clear) и грани стен вдоль контура
    # (x0, y0, x1, y1, nx, ny), нормаль внутрь помещения - для плинтусов.
    region: list = field(default_factory=list)
    wall_edges: list = field(default_factory=list)
    name: str = ""                              # подпись на плане: «Кухня», «Спальня»...

    @property
    def area(self) -> float:
        return sum(max(0.0, x1 - x0) * max(0.0, y1 - y0) for x0, y0, x1, y1 in self.rects())

    def rects(self) -> list:
        """Прямоугольники чистой площади."""
        return list(self.region) if self.region else [self.clear]

    def contains(self, x: float, y: float) -> bool:
        if self.region:
            return any(a <= x < c and b <= y < d for a, b, c, d in self.region)
        x0, y0, x1, y1 = self.cell
        return x0 < x < x1 and y0 < y < y1


@dataclass
class Box:
    center: tuple[float, float, float]
    size: tuple[float, float, float]            # вдоль, поперёк, высота
    yaw: float = 0.0                            # рад, поворот вокруг Z


@dataclass
class Item:
    id: int
    kind: str
    label: str
    boxes: list[Box]
    reflectance: float
    room_id: int | None = None
    # Не-параллелепипеды: {"type": "cylinder", "center", "radius", "height"} |
    # {"type": "curtain", "p0", "p1", "z0", "z1", "amp", "period"} |
    # {"type": "mesh", "path", "center", "size", "yaw"} - внешняя модель, вписанная в габарит
    prims: list = field(default_factory=list)
    transmit: float = 0.0          # доля лучей, проходящих насквозь (тюль)


@dataclass
class Layout:
    ceiling_height: float
    rooms: list[Room]
    walls: list[Wall]
    openings: list[Opening] = field(default_factory=list)
    items: list[Item] = field(default_factory=list)
    materials: dict = field(default_factory=dict)   # отражательная способность поверхностей
    meta: dict = field(default_factory=dict)
    footprint: list = field(default_factory=list)   # прямоугольники под перекрытиями (пусто - ячейки помещений)

    def wall(self, wall_id: int) -> Wall:
        return self.walls[wall_id]

    def openings_on(self, wall_id: int) -> list[Opening]:
        return [o for o in self.openings if o.wall_id == wall_id]

    def bbox(self) -> tuple[float, float, float, float]:
        """Габарит здания вместе с телом наружных стен."""
        pts = []
        for w in self.walls:
            for s in (-w.ext0, w.length + w.ext1):
                for off in (-w.thickness / 2, w.thickness / 2):
                    pts.append(w.point(s, off))
        pts = np.array(pts)
        return (*pts.min(0), *pts.max(0))

    def room_at(self, x: float, y: float) -> int | None:
        for r in self.rooms:
            if r.contains(x, y):
                return r.id
        return None

    def slab_rects(self) -> list:
        return list(self.footprint) if self.footprint else [r.cell for r in self.rooms]

    # --- JSON -----------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "units": "m",
            "frame": "layout: X вправо, Y вверх, Z вверх, пол Z=0",
            "ceiling_height": self.ceiling_height,
            "rooms": [asdict(r) for r in self.rooms],
            "walls": [asdict(w) for w in self.walls],
            "openings": [asdict(o) for o in self.openings],
            "items": [asdict(i) for i in self.items],
            "materials": self.materials,
            "footprint": [list(r) for r in self.footprint],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        def tup(x):
            return tuple(x) if isinstance(x, list) else x

        rooms = []
        for r in d["rooms"]:
            r = dict(r)
            region = [tuple(x) for x in r.pop("region", [])]
            edges = [tuple(x) for x in r.pop("wall_edges", [])]
            rooms.append(Room(**{k: tup(v) for k, v in r.items()}, region=region, wall_edges=edges))
        walls = [Wall(**{k: tup(v) for k, v in w.items()}) for w in d["walls"]]
        openings = [Opening(**{k: tup(v) for k, v in o.items()}) for o in d["openings"]]
        items = []
        for it in d.get("items", []):
            boxes = [Box(tuple(b["center"]), tuple(b["size"]), b["yaw"]) for b in it["boxes"]]
            items.append(Item(it["id"], it["kind"], it["label"], boxes,
                              it["reflectance"], it.get("room_id"), it.get("prims", []),
                              it.get("transmit", 0.0)))
        return cls(d["ceiling_height"], rooms, walls, openings, items,
                   d.get("materials", {}), d.get("meta", {}),
                   [tuple(x) for x in d.get("footprint", [])])

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "Layout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ======================================================================
# Генератор
# ======================================================================

@dataclass
class _Cell:
    x0: float
    y0: float
    x1: float
    y1: float
    # толщина стены по сторонам (L, R, B, T); None - сторона на контуре здания
    side_t: list = field(default_factory=lambda: [None, None, None, None])

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.w * self.h

    def contains(self, x: float, y: float) -> bool:
        return self.x0 < x < self.x1 and self.y0 < y < self.y1


class LayoutGenerator:
    def __init__(self, cfg: LayoutConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def _u(self, r) -> float:
        return float(self.rng.uniform(r[0], r[1]))

    def _snap(self, v: float) -> float:
        q = self.cfg.snap_m
        return round(v / q) * q

    # ------------------------------------------------------------------
    def generate(self) -> Layout:
        cfg, rng = self.cfg, self.rng
        W = self._snap(self._u(cfg.footprint_x_m))
        D = self._snap(self._u(cfg.footprint_y_m))
        n_rooms = int(rng.integers(cfg.rooms[0], cfg.rooms[1] + 1))
        cells = self._split(W, D, n_rooms)
        if len(cells) >= 3 and rng.random() < cfg.p_l_shape:
            cells = self._remove_corner(cells, W, D)

        H = round(self._u(cfg.ceiling_height_m), 3)
        t_ext = float(rng.choice(cfg.exterior_wall_m))
        walls, int_spans, ext_spans = self._build_walls(cells, t_ext)
        rooms = self._make_rooms(cells, H)
        layout = Layout(H, rooms, walls, meta={"footprint_m": [W, D], "t_ext": t_ext})
        self._place_doors(layout, int_spans)
        self._place_entrance(layout, ext_spans)
        self._place_windows(layout, ext_spans)
        return layout

    # --- BSP -------------------------------------------------------------
    def _split(self, W: float, D: float, n_target: int) -> list[_Cell]:
        cfg, rng = self.cfg, self.rng
        m = cfg.min_room_side_m
        cells = [_Cell(0.0, 0.0, W, D)]
        frozen: set[int] = set()
        while len(cells) < n_target:
            cand = [i for i in range(len(cells)) if i not in frozen]
            if not cand:
                break
            areas = np.array([cells[i].area for i in cand])
            i = cand[int(rng.choice(len(cand), p=areas / areas.sum()))]
            c = cells[i]
            axes = []
            if c.w >= 2 * m + EPS:
                axes.append("x")
            if c.h >= 2 * m + EPS:
                axes.append("y")
            if not axes:
                frozen.add(i)
                continue
            if len(axes) == 2:
                longer = "x" if c.w >= c.h else "y"
                axis = longer if rng.random() < 0.75 else ("y" if longer == "x" else "x")
            else:
                axis = axes[0]
            lo, hi = (c.x0, c.x1) if axis == "x" else (c.y0, c.y1)
            pos = self._snap(rng.uniform(lo + m, hi - m))
            pos = min(max(pos, lo + m), hi - m)
            t = float(rng.choice(cfg.interior_wall_m))
            if axis == "x":
                a = _Cell(c.x0, c.y0, pos, c.y1, list(c.side_t))
                b = _Cell(pos, c.y0, c.x1, c.y1, list(c.side_t))
                a.side_t[1] = t
                b.side_t[0] = t
            else:
                a = _Cell(c.x0, c.y0, c.x1, pos, list(c.side_t))
                b = _Cell(c.x0, pos, c.x1, c.y1, list(c.side_t))
                a.side_t[3] = t
                b.side_t[2] = t
            cells[i] = a
            cells.append(b)
            frozen.clear()
        return cells

    def _remove_corner(self, cells: list[_Cell], W: float, D: float) -> list[_Cell]:
        total = sum(c.area for c in cells)
        order = self.rng.permutation(len(cells))
        for i in order:
            c = cells[i]
            at_corner = (c.x0 < EPS or c.x1 > W - EPS) and (c.y0 < EPS or c.y1 > D - EPS)
            if not at_corner or c.area > 0.4 * total:
                continue
            rest = [cells[j] for j in range(len(cells)) if j != i]
            if _connected(rest):
                return rest
        return cells

    # --- стены -----------------------------------------------------------
    def _build_walls(self, cells: list[_Cell], t_ext: float):
        xs = sorted({v for c in cells for v in (c.x0, c.x1)})
        ys = sorted({v for c in cells for v in (c.y0, c.y1)})

        def cell_at(x, y):
            for j, c in enumerate(cells):
                if c.contains(x, y):
                    return j
            return None

        interior = {}   # (orient, coord, t) -> list[(a, b, pair)]
        exterior = {}   # (orient, coord, sign) -> list[(a, b, room)]
        for i, c in enumerate(cells):
            # orient "v": линия x = coord, вдоль y; "h": линия y = coord, вдоль x
            sides = (("v", c.x0, -1, c.y0, c.y1, 0), ("v", c.x1, +1, c.y0, c.y1, 1),
                     ("h", c.y0, -1, c.x0, c.x1, 2), ("h", c.y1, +1, c.x0, c.x1, 3))
            for orient, coord, sign, a0, a1, k in sides:
                brk = [v for v in (ys if orient == "v" else xs) if a0 - EPS <= v <= a1 + EPS]
                for a, b in zip(brk[:-1], brk[1:]):
                    if b - a < EPS:
                        continue
                    mid = 0.5 * (a + b)
                    probe = (coord + sign * 1e-4, mid) if orient == "v" else (mid, coord + sign * 1e-4)
                    j = cell_at(*probe)
                    if j is None:
                        exterior.setdefault((orient, coord, sign), []).append((a, b, i))
                    elif j > i:
                        t = c.side_t[k]
                        interior.setdefault((orient, coord, t), []).append((a, b, (i, j)))

        walls: list[Wall] = []
        int_spans = []   # (wall_id, a, b, pair) - участки общих стен двух помещений
        ext_spans = []   # (wall_id, a, b, room)
        for (orient, coord, t), segs in sorted(interior.items(), key=lambda kv: kv[0][:2]):
            for a, b, parts in _merge_segments(segs):
                wid = len(walls)
                p0, p1 = _line_points(orient, coord, a, b)
                walls.append(Wall(wid, p0, p1, float(t), "interior", 0, 0.01, 0.01))
                for pa, pb, pair in _merge_by_tag(parts):
                    int_spans.append((wid, pa - a, pb - a, pair))

        for (orient, coord, sign), segs in sorted(exterior.items(), key=lambda kv: kv[0][:2]):
            for a, b, parts in _merge_segments(segs):
                wid = len(walls)
                center = coord + sign * t_ext / 2
                p0, p1 = _line_points(orient, center, a, b)
                w = Wall(wid, p0, p1, t_ext, "exterior")
                w.outward = int(np.sign(np.dot(_axis_vec(orient, sign), w.n)))
                # наружный угол (за концом стены с внутренней стороны - улица) -> удлиняем
                for end, val in ((0, a - 1e-4), (1, b + 1e-4)):
                    inner = coord - sign * 1e-4
                    probe = (inner, val) if orient == "v" else (val, inner)
                    if cell_at(*probe) is None:
                        if end == 0:
                            w.ext0 = t_ext
                        else:
                            w.ext1 = t_ext
                walls.append(w)
                for pa, pb, room in _merge_by_tag(parts):
                    ext_spans.append((wid, pa - a, pb - a, room))
        return walls, int_spans, ext_spans

    def _make_rooms(self, cells: list[_Cell], H: float) -> list[Room]:
        cfg, rng = self.cfg, self.rng
        rooms = []
        for i, c in enumerate(cells):
            sh = [0.0 if t is None else t / 2 for t in c.side_t]
            clear = (c.x0 + sh[0], c.y0 + sh[2], c.x1 - sh[1], c.y1 - sh[3])
            room = Room(i, (c.x0, c.y0, c.x1, c.y1), clear, ceiling_z=H)
            short, long_ = sorted((clear[2] - clear[0], clear[3] - clear[1]))
            if short < 1.9 and long_ / max(short, EPS) > 2.2:
                room.kind = "corridor"
            elif room.area < 4.5 and rng.random() < 0.7:
                room.kind = "bath"
            if rng.random() < cfg.p_suspended_ceiling:
                room.ceiling_z = round(H - self._u(cfg.suspended_drop_m), 3)
            rooms.append(room)
        big = [r for r in rooms if r.kind == "room" and r.area > 6.0]
        if big:
            big[int(rng.integers(len(big)))].kind = "kitchen"
        return rooms

    # --- проёмы ----------------------------------------------------------
    def _fits(self, layout: Layout, wall_id: int, s: float, w: float, gap: float = 0.2) -> bool:
        for o in layout.openings_on(wall_id):
            if abs(o.s - s) < (o.width + w) / 2 + gap:
                return False
        return True

    def _try_place(self, layout, wall_id, a, b, width, margin, near_end_bias=0.0):
        """Центр проёма шириной width на участке [a, b] стены; None, если не влезает."""
        lo, hi = a + margin + width / 2, b - margin - width / 2
        if hi < lo:
            return None
        for _ in range(12):
            if self.rng.random() < near_end_bias:
                off = self.rng.uniform(0, min(0.3, hi - lo))
                s = lo + off if self.rng.random() < 0.5 else hi - off
            else:
                s = self.rng.uniform(lo, hi)
            s = round(s, 3)
            if self._fits(layout, wall_id, s, width):
                return s
        return None

    def _door_leaf(self, kind: str) -> dict:
        if kind == "passage":
            return {}
        return {"hinge": int(self.rng.choice((-1, 1))), "swing": int(self.rng.choice((-1, 1)))}

    def _place_doors(self, layout: Layout, int_spans) -> None:
        cfg, rng = self.cfg, self.rng
        min_len = cfg.door_width_m[0] + 2 * cfg.door_margin_m
        by_pair: dict[tuple, list] = {}
        for wid, a, b, pair in int_spans:
            if b - a >= min_len:
                by_pair.setdefault(pair, []).append((wid, a, b))
        pairs = list(by_pair)
        rng.shuffle(pairs)
        parent = list(range(len(layout.rooms)))

        def find(k):
            while parent[k] != k:
                parent[k] = parent[parent[k]]
                k = parent[k]
            return k

        chosen = []
        for p in pairs:
            ra, rb = find(p[0]), find(p[1])
            if ra != rb:
                parent[ra] = rb
                chosen.append(p)
            elif rng.random() < cfg.p_extra_door:
                chosen.append(p)

        for pair in chosen:
            wid, a, b = max(by_pair[pair], key=lambda x: x[2] - x[1])
            kind = "passage" if rng.random() < cfg.p_passage else "door"
            if kind == "passage":
                width = round(self._u(cfg.passage_width_m), 2)
                head = self._u(cfg.passage_head_m)
            else:
                width = round(self._u(cfg.door_width_m), 2)
                head = self._u(cfg.door_head_m)
            width = min(width, b - a - 2 * cfg.door_margin_m)
            if width < cfg.door_width_m[0] - EPS:
                continue
            s = self._try_place(layout, wid, a, b, width, cfg.door_margin_m, near_end_bias=0.6)
            if s is None:
                continue
            head = round(min(head, layout.ceiling_height - 0.1), 3)
            layout.openings.append(Opening(len(layout.openings), wid, kind, s, width, 0.0, head,
                                           rooms=tuple(pair), **self._door_leaf(kind)))

    def _place_entrance(self, layout: Layout, ext_spans) -> None:
        cfg, rng = self.cfg, self.rng
        if rng.random() >= cfg.p_entrance:
            return
        width = round(self._u(cfg.entrance_width_m), 2)
        spans = [sp for sp in ext_spans if sp[2] - sp[1] >= width + 2 * cfg.window_margin_m]
        if not spans:
            return
        wid, a, b, room = spans[int(rng.integers(len(spans)))]
        s = self._try_place(layout, wid, a, b, width, cfg.window_margin_m, near_end_bias=0.3)
        if s is None:
            return
        head = round(min(self._u(cfg.door_head_m), layout.ceiling_height - 0.1), 3)
        wall = layout.wall(wid)
        layout.openings.append(Opening(len(layout.openings), wid, "door", s, width, 0.0, head,
                                       rooms=(room,), hinge=int(rng.choice((-1, 1))),
                                       swing=-wall.outward))

    def _place_windows(self, layout: Layout, ext_spans) -> None:
        cfg, rng = self.cfg, self.rng
        sill = round(self._u(cfg.window_sill_m), 3)
        head = round(min(self._u(cfg.window_head_m), layout.ceiling_height - 0.15), 3)
        if head - sill < 0.6:
            return
        for wid, a, b, room_id in ext_spans:
            room = layout.rooms[room_id]
            if room.area < cfg.min_window_room_area_m2 or room.kind == "corridor":
                continue
            if rng.random() >= cfg.p_window_per_side:
                continue
            free = b - a - 2 * cfg.window_margin_m
            if free < cfg.window_width_m[0]:
                continue
            n = 1 if free < 3.5 else int(rng.integers(1, 3))
            for k in range(n):
                seg_a = a + cfg.window_margin_m + free * k / n
                seg_b = a + cfg.window_margin_m + free * (k + 1) / n
                width = round(min(self._u(cfg.window_width_m), seg_b - seg_a - 0.2), 2)
                if width < cfg.window_width_m[0]:
                    continue
                s = self._try_place(layout, wid, seg_a, seg_b, width, 0.1)
                if s is None:
                    continue
                layout.openings.append(Opening(len(layout.openings), wid, "window", s, width,
                                               sill, head, rooms=(room_id,)))


# --- вспомогательные функции ---------------------------------------------

def _axis_vec(orient: str, sign: int) -> np.ndarray:
    return np.array([sign, 0.0]) if orient == "v" else np.array([0.0, sign])


def _line_points(orient: str, coord: float, a: float, b: float):
    if orient == "v":
        return (float(coord), float(a)), (float(coord), float(b))
    return (float(a), float(coord)), (float(b), float(coord))


def _merge_segments(segs):
    """Сливает соприкасающиеся отрезки одной линии. -> [(a, b, [(a_i, b_i, tag_i), ...])]."""
    segs = sorted(segs, key=lambda s: s[0])
    out = []
    for a, b, tag in segs:
        if out and a <= out[-1][1] + EPS:
            out[-1][1] = max(out[-1][1], b)
            out[-1][2].append((a, b, tag))
        else:
            out.append([a, b, [(a, b, tag)]])
    return [(a, b, parts) for a, b, parts in out]


def _merge_by_tag(parts):
    """Сливает соседние элементарные участки с одинаковым тегом (пара помещений / помещение)."""
    out = []
    for a, b, tag in sorted(parts, key=lambda p: p[0]):
        if out and out[-1][2] == tag and a <= out[-1][1] + EPS:
            out[-1][1] = b
        else:
            out.append([a, b, tag])
    return [tuple(x) for x in out]


def _connected(cells: list[_Cell]) -> bool:
    if not cells:
        return False
    adj = {i: set() for i in range(len(cells))}
    for i, a in enumerate(cells):
        for j, b in enumerate(cells[i + 1:], i + 1):
            ov_y = min(a.y1, b.y1) - max(a.y0, b.y0)
            ov_x = min(a.x1, b.x1) - max(a.x0, b.x0)
            touch_v = (abs(a.x1 - b.x0) < EPS or abs(b.x1 - a.x0) < EPS) and ov_y > 1.0
            touch_h = (abs(a.y1 - b.y0) < EPS or abs(b.y1 - a.y0) < EPS) and ov_x > 1.0
            if touch_v or touch_h:
                adj[i].add(j)
                adj[j].add(i)
    seen, stack = {0}, [0]
    while stack:
        k = stack.pop()
        for m in adj[k] - seen:
            seen.add(m)
            stack.append(m)
    return len(seen) == len(cells)
