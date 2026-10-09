"""Генератор квартир: выращивание комнат на сетке + правила квартир в секционном доме.

Основа - метод Lopes et al., 2010 («A Constrained Growth Method for Procedural Floor
Plan Generation») в реализации ProcTHOR (allenai/procthor, Apache License 2.0):
  - grow_rect / grow_l_shape / select_room - перенесены из
    procthor/generation/floorplan_generation.py с минимальными правками;
  - угловые вырезы контура - по мотивам procthor/generation/interior_boundaries.py;
  - порядок дверей (сначала явные связи, затем всё к прихожей, затем приватные
    к общим, затем проверка достижимости) - из описания в floorplan_generation.py
    и логики procthor/generation/doors.py;
  - лучший из N кандидатов по совпадению долей площадей - score_floorplan ProcTHOR.

Добавлено для квартир (в ProcTHOR этого нет - там отдельные дома):
  - фасад(ы) с окнами и глухие межквартирные стены; вход из прихожей с лестничной клетки;
  - семена жилых комнат и кухни вдоль фасада по долям ширины, прихожая у входа,
    санузлы рядом с прихожей и в глубине корпуса;
  - жёсткие проверки: жилые комнаты и кухня выходят на фасад (или лоджию), минимальные
    ширины, без U-образных комнат, санузлы без окон, все помещения достижимы;
  - лоджия (открытая с ограждением или остеклённая) с балконным блоком;
  - типы стен: наружная, межквартирная, перегородки, несущая внутренняя стена.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .config import LayoutConfig
from .layout import Layout, LayoutGenerator, Opening, Wall
from .walls2layout import (OUTSIDE, footprint_rects, label_overlap, rooms_from_walls,
                           wall_spans)

EMPTY, OUT, LOGGIA = -1, -2, -3
DIRS = {"bottom": (0, -1), "top": (0, 1), "left": (-1, 0), "right": (1, 0)}


@dataclass(frozen=True)
class RoomType:
    label: str
    kind: str                      # класс simscan: room | corridor | bath | kitchen
    area: tuple[float, float]      # м2
    min_width: float               # м, комната вмещает квадрат min_width x min_width
    window: bool = False           # нужен выход на фасад
    wet: bool = False
    door: tuple[float, float] = (0.8, 0.8)


TYPES = {
    "hall": RoomType("Прихожая", "corridor", (3.5, 8.0), 1.2),
    "corridor": RoomType("Коридор", "corridor", (3.0, 6.0), 1.0, door=(0.8, 0.9)),
    "bath": RoomType("Ванная", "bath", (2.8, 4.5), 1.5, wet=True, door=(0.6, 0.7)),
    "wc": RoomType("Туалет", "bath", (1.1, 1.6), 0.85, wet=True, door=(0.6, 0.6)),
    "combined": RoomType("Санузел", "bath", (3.5, 5.5), 1.5, wet=True, door=(0.7, 0.7)),
    "kitchen": RoomType("Кухня", "kitchen", (8.0, 13.0), 2.3, window=True, door=(0.7, 0.8)),
    "kitchen_living": RoomType("Кухня-гостиная", "kitchen", (18.0, 28.0), 3.0, window=True),
    "living": RoomType("Гостиная", "room", (15.0, 22.0), 3.0, window=True),
    "bedroom": RoomType("Спальня", "room", (10.0, 15.0), 2.6, window=True),
    "closet": RoomType("Гардероб", "room", (2.0, 4.0), 1.1, door=(0.6, 0.7)),
}

# Программы квартир: дерево зон (MetaRoom ProcTHOR). Строка - комната, кортеж - зона.
# «a|b» - один из вариантов, «a+b» - обе комнаты, «a?» - с вероятностью 0,5.
PROGRAMS = {
    "studio": (0.8, [("entry", ["hall", "combined"]), "kitchen_living"]),
    "1k": (2.0, [("entry", ["hall", "bath+wc|combined", "closet?"]), "kitchen", "bedroom"]),
    "2k": (2.0, [("entry", ["hall", "bath+wc|combined"]), ("day", ["kitchen", "living"]),
                 "bedroom"]),
    # 3k и 4k пока строятся редко (см. docs/ANALYSIS.md, раздел 7) - малый вес
    "3k": (0.3, [("entry", ["hall", "bath+wc", "closet?"]), "corridor?", "kitchen", "living",
                 "bedroom", "bedroom"]),
    "4k": (0.05, [("entry", ["hall", "bath+wc"]), "corridor", "kitchen", "living", "bedroom",
                 "bedroom", "bedroom"]),
}

# Кому можно открываться: по убыванию приоритета (по мотивам doors.py ProcTHOR).
DOOR_PARENTS = {
    "bath": ("hall", "corridor"), "wc": ("hall", "corridor"),
    "combined": ("hall", "corridor"),
    "kitchen": ("hall", "corridor", "living"), "kitchen_living": ("hall", "corridor"),
    "living": ("hall", "corridor"), "bedroom": ("corridor", "hall", "living"),
    "corridor": ("hall", "living"), "closet": ("hall", "corridor", "bedroom"),
}


class InvalidPlan(Exception):
    pass


@dataclass
class _GRoom:
    """Помещение или зона на сетке (поля как у LeafRoom / MetaRoom ProcTHOR)."""
    idx: int
    type: str                      # тип комнаты или "zone:<имя>"
    target: float                  # целевая площадь в клетках
    children: list = field(default_factory=list)
    min_x: int = 0
    min_y: int = 0
    max_x: int = 0
    max_y: int = 0

    @property
    def ratio(self) -> float:
        return self.target

    def leaves(self) -> list:
        return [self] if not self.children else [x for c in self.children for x in c.leaves()]

    def needs_facade(self) -> bool:
        return any(TYPES[x.type].window for x in self.leaves())

    def has(self, t: str) -> bool:
        return any(x.type == t for x in self.leaves())


@dataclass
class _Shell:
    """Контур квартиры: сетка, фасады, вход, лоджия."""
    grid: np.ndarray               # [j, i], j вверх; EMPTY / OUT / LOGGIA
    cell: float
    facades: tuple[str, ...]
    entrance: str
    loggia: dict | None = None
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Перенесено из ProcTHOR (floorplan_generation.py), Apache-2.0
# ----------------------------------------------------------------------

def select_room(rooms, rng):
    total = sum(r.ratio for r in rooms)
    r = rng.random() * total
    for room in rooms:
        r -= room.ratio
        if r <= 0:
            return room
    return rooms[-1]


def grow_rect(room, fp, rng) -> bool:
    if (room.max_x - room.min_x) * (room.max_y - room.min_y) >= room.target:
        return False
    sizes = {
        "right": room.max_y - room.min_y if room.max_x < fp.shape[1]
        and (fp[room.min_y:room.max_y, room.max_x] == EMPTY).all() else 0,
        "left": room.max_y - room.min_y if room.min_x > 0
        and (fp[room.min_y:room.max_y, room.min_x - 1] == EMPTY).all() else 0,
        "up": room.max_x - room.min_x if room.max_y < fp.shape[0]
        and (fp[room.max_y, room.min_x:room.max_x] == EMPTY).all() else 0,
        "down": room.max_x - room.min_x if room.min_y > 0
        and (fp[room.min_y - 1, room.min_x:room.max_x] == EMPTY).all() else 0,
    }
    best = max(sizes.values())
    if best == 0:
        return False
    d = rng.choice([k for k, v in sizes.items() if v == best])
    if d == "right":
        room.max_x += 1
        fp[room.min_y:room.max_y, room.max_x - 1] = room.idx
    elif d == "left":
        room.min_x -= 1
        fp[room.min_y:room.max_y, room.min_x] = room.idx
    elif d == "up":
        room.max_y += 1
        fp[room.max_y - 1, room.min_x:room.max_x] = room.idx
    else:
        room.min_y -= 1
        fp[room.min_y, room.min_x:room.max_x] = room.idx
    return True


def grow_l_shape(room, fp, rng) -> bool:
    r = room
    sizes = {
        "right": [y for y in range(r.min_y, r.max_y) if r.max_x < fp.shape[1]
                  and fp[y, r.max_x] == EMPTY and fp[y, r.max_x - 1] == r.idx],
        "left": [y for y in range(r.min_y, r.max_y) if r.min_x > 0
                 and fp[y, r.min_x - 1] == EMPTY and fp[y, r.min_x] == r.idx],
        "up": [x for x in range(r.min_x, r.max_x) if r.max_y < fp.shape[0]
               and fp[r.max_y, x] == EMPTY and fp[r.max_y - 1, x] == r.idx],
        "down": [x for x in range(r.min_x, r.max_x) if r.min_y > 0
                 and fp[r.min_y - 1, x] == EMPTY and fp[r.min_y, x] == r.idx],
    }
    best = max(len(v) for v in sizes.values())
    if best == 0:
        return False
    d = rng.choice([k for k, v in sizes.items() if len(v) == best])
    if d == "right":
        fp[sizes[d], r.max_x] = r.idx
        r.max_x += 1
    elif d == "left":
        fp[sizes[d], r.min_x - 1] = r.idx
        r.min_x -= 1
    elif d == "up":
        fp[r.max_y, sizes[d]] = r.idx
        r.max_y += 1
    else:
        fp[r.min_y - 1, sizes[d]] = r.idx
        r.min_y -= 1
    return True


def expand_rooms(rooms, fp, rng) -> None:
    grow = list(rooms)
    while grow:
        room = select_room(grow, rng)
        if not grow_rect(room, fp, rng):
            grow.remove(room)
    grow = list(rooms)
    while grow:
        room = select_room(grow, rng)
        if not grow_l_shape(room, fp, rng):
            grow.remove(room)


# ----------------------------------------------------------------------
# Квартира
# ----------------------------------------------------------------------

class ApartmentGenerator:
    def __init__(self, cfg: LayoutConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.c = cfg.apartment_cell_m

    def _u(self, r) -> float:
        return float(self.rng.uniform(r[0], r[1]))

    # --- программа и контур ---------------------------------------------
    def _program(self) -> tuple[str, list]:
        """Программа квартиры: (имя, дерево [тип | (зона, [типы])])."""
        names = list(PROGRAMS)
        weights = np.array([PROGRAMS[n][0] for n in names])
        name = names[int(self.rng.choice(len(names), p=weights / weights.sum()))]

        def expand(items):
            out = []
            for item in items:
                if isinstance(item, tuple):
                    out.append((item[0], expand(item[1])))
                    continue
                if item.endswith("?"):
                    if self.rng.random() < 0.5:
                        out.append(item[:-1])
                    continue
                options = item.split("|")
                out += options[int(self.rng.integers(len(options)))].split("+")
            return out

        return name, expand(PROGRAMS[name][1])

    def _build_nodes(self, tree) -> tuple[list, list]:
        """Дерево программы -> узлы с целевыми площадями; возвращает (верхний уровень, листья)."""
        leaves: list[_GRoom] = []
        zone_id = [100]

        def build(items):
            nodes = []
            for item in items:
                if isinstance(item, tuple):
                    kids = build(item[1])
                    zone_id[0] += 1
                    nodes.append(_GRoom(zone_id[0], "zone:" + item[0],
                                        sum(k.target for k in kids), kids))
                else:
                    leaf = _GRoom(len(leaves), item, self._u(TYPES[item].area) / self.c ** 2)
                    leaves.append(leaf)
                    nodes.append(leaf)
            return nodes

        return build(tree), leaves

    def _shell(self, types: list[str], areas: list[float]) -> _Shell:
        cfg, rng, c = self.cfg, self.rng, self.c
        total = sum(areas) * 1.04                         # запас на стены
        windowed = [TYPES[t] for t in types if TYPES[t].window]
        facade_need = sum(t.min_width + 0.6 for t in windowed)
        kinds = ("single", "through", "corner")
        w = np.array(cfg.apartment_facade_weights)
        kind = kinds[int(rng.choice(3, p=w / w.sum()))]
        if kind == "single" and facade_need > 10.0:
            kind = str(rng.choice(["corner", "through"]))
        if kind == "single":
            depth = self._u((5.4, 7.4))
            width = max(total / depth, facade_need)
            facades, entrance = ("bottom",), "top"
        elif kind == "through":
            depth = self._u((9.0, 12.0))
            width = max(total / depth, facade_need / 2 + 1.0, 4.2)
            facades, entrance = ("bottom", "top"), str(rng.choice(["left", "right"]))
        else:
            depth = self._u((6.0, 9.0))
            width = max(total / depth, facade_need * 0.6)
            side = str(rng.choice(["left", "right"]))
            facades = ("bottom", side)
            entrance = "top"
        nx = max(int(round(width / c)), 8)
        ny = max(int(round(depth / c)), 8)
        grid = np.full((ny, nx), EMPTY, int)

        # угловой вырез контура (interior_boundaries.py ProcTHOR) - только у входа,
        # чтобы фасад оставался прямым
        if rng.random() < cfg.apartment_p_cut and nx > 14 and ny > 14:
            cw = int(rng.integers(3, max(4, nx // 4)))
            ch = int(rng.integers(3, max(4, ny // 4)))
            if entrance in ("top", "bottom"):
                rows = slice(ny - ch, ny) if entrance == "top" else slice(0, ch)
                cols = slice(0, cw) if rng.random() < 0.5 else slice(nx - cw, nx)
            else:
                cols = slice(0, cw) if entrance == "left" else slice(nx - cw, nx)
                rows = slice(ny - ch, ny) if rng.random() < 0.5 else slice(0, ch)
            grid[rows, cols] = OUT

        loggia = None
        if rng.random() < cfg.apartment_p_loggia and windowed:
            side = facades[int(rng.integers(len(facades)))]
            depth_c = int(round(self._u((1.2, 1.6)) / c))
            along = nx if side in ("bottom", "top") else ny
            length_c = int(min(along - 2, round(self._u((2.8, 6.0)) / c)))
            if length_c >= 8:
                start = int(rng.integers(0, along - length_c + 1))
                sl = slice(start, start + length_c)
                if side == "bottom":
                    grid[:depth_c, sl] = LOGGIA
                elif side == "top":
                    grid[ny - depth_c:, sl] = LOGGIA
                elif side == "left":
                    grid[sl, :depth_c] = LOGGIA
                else:
                    grid[sl, nx - depth_c:] = LOGGIA
                loggia = {"side": side, "glazed": bool(rng.random() < cfg.apartment_p_glazed_loggia)}
        return _Shell(grid, c, facades, entrance, loggia, {"shape": kind})

    # --- семена ----------------------------------------------------------
    def _facade_mask(self, shell: _Shell, dist: int) -> np.ndarray:
        """Клетки не дальше dist клеток от фасада (или от лоджии)."""
        g = shell.grid
        ny, nx = g.shape
        m = np.zeros_like(g, bool)
        for side in shell.facades:
            if side == "bottom":
                m[:dist, :] = True
            elif side == "top":
                m[ny - dist:, :] = True
            elif side == "left":
                m[:, :dist] = True
            else:
                m[:, nx - dist:] = True
        if (g == LOGGIA).any():
            m |= ndimage.binary_dilation(g == LOGGIA, iterations=dist)
        return m & (g == EMPTY)

    def _side_band(self, shell: _Shell, side: str, d0: int, d1: int) -> np.ndarray:
        g = shell.grid
        ny, nx = g.shape
        m = np.zeros_like(g, bool)
        if side == "bottom":
            m[d0:d1, :] = True
        elif side == "top":
            m[ny - d1:ny - d0, :] = True
        elif side == "left":
            m[:, d0:d1] = True
        else:
            m[:, nx - d1:nx - d0] = True
        return m & (g == EMPTY)

    def _place_seeds(self, shell: _Shell, nodes: list, fp: np.ndarray, masks: dict) -> None:
        """Семена узлов одного уровня в окне fp (вид на часть сетки; masks - те же окна).

        Узлы с жилыми комнатами - вдоль фасада, каждый на своём участке фасадной полосы;
        узел с прихожей - у входа; остальные (санузлы, коридор, гардероб) - в глубине,
        рядом с прихожей."""
        rng = self.rng
        free = fp == EMPTY
        taken = np.zeros_like(free)

        def put(node, weights):
            w = weights & free & ~taken
            if not w.any():
                w = free & ~taken
                if not w.any():
                    raise InvalidPlan("нет места для семени")
            cand = np.argwhere(w)
            j, i = cand[int(rng.integers(len(cand)))]
            node.min_x, node.min_y, node.max_x, node.max_y = i, j, i + 1, j + 1
            fp[j, i] = node.idx
            r = 2
            taken[max(0, j - r):j + r + 1, max(0, i - r):i + r + 1] = True

        hall_nodes = [n for n in nodes if n.has("hall")]
        facade_nodes = [n for n in nodes if n.needs_facade() and n not in hall_nodes]
        other = [n for n in nodes if n not in facade_nodes and n not in hall_nodes]
        rng.shuffle(facade_nodes)
        band = masks["facade_near"] & free
        if not band.any():
            band = masks["facade_any"] & free
        idx = np.argwhere(band)
        for k, node in enumerate(facade_nodes):
            sel = np.zeros_like(band)
            if len(idx):
                key = idx[:, 1] if shell.facades[0] in ("bottom", "top") else idx[:, 0]
                part = np.array_split(np.argsort(key + rng.random(len(key)) * 0.1),
                                      len(facade_nodes))[k]
                sel[idx[part, 0], idx[part, 1]] = True
            put(node, sel)
        for node in hall_nodes:
            put(node, masks["entrance"] & ~masks["facade_any"] if (masks["entrance"] & free
                & ~masks["facade_any"]).any() else masks["entrance"])
        near_hall = masks["near_hall"] & ~masks["facade_any"]
        for node in other:
            put(node, near_hall if (near_hall & free).any() else ~masks["facade_any"])

    # --- проверка кандидата ----------------------------------------------
    def _fill_zone(self, sub, zone, ids) -> None:
        """Остаток зоны - соседней комнате этой же зоны."""
        for _ in range(100):
            empty = np.argwhere(zone & (sub == EMPTY))
            if len(empty) == 0:
                return
            changed = False
            for j, i in empty:
                nb = [sub[jj, ii] for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
                      if 0 <= jj < sub.shape[0] and 0 <= ii < sub.shape[1] and sub[jj, ii] in ids]
                if nb:
                    vals, cnt = np.unique(nb, return_counts=True)
                    sub[j, i] = vals[int(np.argmax(cnt))]
                    changed = True
            if not changed:
                raise InvalidPlan("пустота в зоне")

    def _fill_rest(self, fp: np.ndarray) -> None:
        """Последняя фаза метода Lopes: пустые клетки - соседу с наибольшим контактом."""
        for _ in range(200):
            empty = np.argwhere(fp == EMPTY)
            if len(empty) == 0:
                return
            changed = False
            for j, i in empty:
                nb = [fp[jj, ii] for jj, ii in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1))
                      if 0 <= jj < fp.shape[0] and 0 <= ii < fp.shape[1] and fp[jj, ii] >= 0]
                if nb:
                    vals, cnt = np.unique(nb, return_counts=True)
                    fp[j, i] = vals[int(np.argmax(cnt))]
                    changed = True
            if not changed:
                raise InvalidPlan("изолированные пустоты")

    def _contacts(self, shell: _Shell, fp: np.ndarray):
        """Длины контактов: комната-комната, комната-фасад/лоджия, комната-вход."""
        c = self.c
        g = np.pad(fp, 1, constant_values=OUT)
        ny, nx = fp.shape
        side_of_pad = np.full(g.shape, "", object)
        side_of_pad[0, :], side_of_pad[-1, :] = "bottom", "top"
        side_of_pad[:, 0], side_of_pad[:, -1] = "left", "right"
        pair, facade, entrance = {}, {}, {}
        for (dj, di), name in (((0, 1), "right"), ((0, -1), "left"), ((1, 0), "top"), ((-1, 0), "bottom")):
            a = g[1:-1, 1:-1]
            b = g[1 + dj:ny + 1 + dj, 1 + di:nx + 1 + di]
            mask = a >= 0
            for v_a, v_b in zip(a[mask], b[mask]):
                if v_b >= 0 and v_b != v_a:
                    key = (min(v_a, v_b), max(v_a, v_b))
                    pair[key] = pair.get(key, 0) + c / 2      # каждая грань посчитана дважды
                elif v_b == LOGGIA or (v_b == OUT and name in shell.facades):
                    facade[v_a] = facade.get(v_a, 0) + c
                elif v_b == OUT and name == shell.entrance:
                    entrance[v_a] = entrance.get(v_a, 0) + c
        return pair, facade, entrance

    def _shape(self, r, m) -> float:
        """Форма одного помещения; бросает InvalidPlan, возвращает компактность."""
        c, t = self.c, TYPES[r.type]
        if not m.any():
            raise InvalidPlan(f"{r.type}: пусто")
        if ndimage.label(m)[1] != 1:
            raise InvalidPlan(f"{r.type}: несвязно")
        area = m.sum() * c * c
        lo, hi = t.area
        if area < 0.8 * lo or area > 1.3 * hi:
            raise InvalidPlan(f"{r.type}: площадь {area:.1f}")
        k = int(math.ceil(t.min_width / c - 1e-6))
        if not ndimage.minimum_filter(m.astype(np.uint8), size=k, mode="constant").any():
            raise InvalidPlan(f"{r.type}: уже {t.min_width} м")
        corridor_like = r.type in ("hall", "corridor")
        if _corners(m) > (8 if corridor_like else 6):
            raise InvalidPlan(f"{r.type}: много углов")
        # без узких отростков: комната - в основном объединение квадратов min_width
        kk = int(math.ceil((1.0 if corridor_like else t.min_width * 0.8) / c - 1e-6))
        opened = ndimage.binary_opening(m, structure=np.ones((kk, kk)))
        if opened.sum() < (0.85 if corridor_like else 0.92) * m.sum():
            raise InvalidPlan(f"{r.type}: узкий отросток")
        jj, ii = np.nonzero(m)
        bw, bh = np.ptp(ii) + 1, np.ptp(jj) + 1
        compact = m.sum() / (bw * bh)
        if compact < (0.55 if corridor_like else 0.72):
            raise InvalidPlan(f"{r.type}: сложная форма")
        if max(bw, bh) / min(bw, bh) > (6.0 if corridor_like else 2.6):
            raise InvalidPlan(f"{r.type}: вытянута")
        return compact

    ZONE_MIN_WIDTH = {"zone:entry": 2.0, "zone:day": 3.0, "zone:night": 2.6}

    def _zone_shape(self, node, m) -> None:
        """Зона должна быть «нарезаемой»: широкой, без отростков и лесенок."""
        c = self.c
        w = self.ZONE_MIN_WIDTH.get(node.type, 2.4)
        k = int(math.ceil(w / c - 1e-6))
        if not ndimage.minimum_filter(m.astype(np.uint8), size=k, mode="constant").any():
            raise InvalidPlan(f"{node.type}: уже {w} м")
        if _corners(m) > 6:
            raise InvalidPlan(f"{node.type}: много углов")
        opened = ndimage.binary_opening(m, structure=np.ones((k, k)))
        if opened.sum() < 0.9 * m.sum():
            raise InvalidPlan(f"{node.type}: узкий отросток")

    def _check(self, shell, rooms, fp):
        c = self.c
        pair, facade, entrance = self._contacts(shell, fp)
        score = 0.0
        for r in rooms:
            t = TYPES[r.type]
            m = fp == r.idx
            compact = self._shape(r, m)
            if t.window and facade.get(r.idx, 0) < max(1.2, 0.6 * t.min_width):
                raise InvalidPlan(f"{r.type}: нет фасада")
            if t.wet and facade.get(r.idx, 0) > 0:
                raise InvalidPlan(f"{r.type}: санузел на фасаде")
            area = m.sum() * c * c
            lo, hi = t.area
            score -= max(0.0, lo - area) / lo + max(0.0, area - hi) / hi
            score += 0.3 * compact
        hall = next(r for r in rooms if r.type == "hall")
        if entrance.get(hall.idx, 0) < 1.5:
            raise InvalidPlan("прихожая не у входа")
        # двери: родитель по приоритетам + достижимость от прихожей
        doors = self._door_plan(rooms, pair)
        wet = [r.idx for r in rooms if TYPES[r.type].wet]
        if len(wet) == 2 and (min(wet), max(wet)) in pair:
            score += 0.3                       # совмещённые стояки
        return score, doors

    def _door_plan(self, rooms, pair) -> list[tuple[int, int, str]]:
        need = 0.9                             # минимум общей стены под дверь
        hall = next(r for r in rooms if r.type == "hall")
        doors, parent = [], {hall.idx: None}
        pending = [r for r in rooms if r.type != "hall"]
        for _ in range(len(rooms) + 2):
            rest = []
            for r in pending:
                ok = None
                for ptype in DOOR_PARENTS[r.type]:
                    cands = [q for q in rooms if q.type == ptype and q.idx in parent
                             and pair.get((min(q.idx, r.idx), max(q.idx, r.idx)), 0) >= need]
                    if cands:
                        ok = max(cands, key=lambda q: pair[(min(q.idx, r.idx), max(q.idx, r.idx))])
                        break
                if ok is None:
                    rest.append(r)
                else:
                    parent[r.idx] = ok.idx
                    doors.append((ok.idx, r.idx, "door"))
            if not rest:
                break
            if len(rest) == len(pending):
                raise InvalidPlan("недостижимые помещения: " + ", ".join(r.type for r in rest))
            pending = rest
        # кухня-гостиная: открытый проём с вероятностью, если кухня не открывается в гостиную
        k = [r for r in rooms if r.type == "kitchen"]
        lv = [r for r in rooms if r.type == "living"]
        if k and lv and self.rng.random() < 0.35:
            key = (min(k[0].idx, lv[0].idx), max(k[0].idx, lv[0].idx))
            if pair.get(key, 0) >= 1.6 and parent.get(k[0].idx) != lv[0].idx:
                doors.append((lv[0].idx, k[0].idx, "passage"))
        return doors

    # --- генерация -------------------------------------------------------
    def _masks(self, shell: _Shell, hall_pos=None) -> dict:
        g = shell.grid
        m = {"facade_near": self._facade_mask(shell, 3) & ~self._facade_mask(shell, 1),
             "facade_any": self._facade_mask(shell, 4),
             "entrance": self._side_band(shell, shell.entrance, 0, 2)}
        if hall_pos is None:
            m["near_hall"] = np.ones_like(g, bool)
        else:
            yy, xx = np.mgrid[:g.shape[0], :g.shape[1]]
            d = np.hypot(yy - hall_pos[0], xx - hall_pos[1])
            m["near_hall"] = (d >= 2) & (d <= 8)
        return m

    def _expand(self, shell, nodes, fp, masks, oy=0, ox=0, top=True) -> None:
        """recursively_expand_rooms ProcTHOR: вырастить уровень, затем зоны изнутри."""
        self._place_seeds(shell, nodes, fp, masks)
        expand_rooms(nodes, fp, self.rng)
        for node in nodes:
            if node.children:
                self._zone_shape(node, fp == node.idx)
        for node in nodes:
            if not node.children:
                continue
            sl = (slice(node.min_y, node.max_y), slice(node.min_x, node.max_x))
            sub = fp[sl]                                     # вид: изменения идут в fp
            sub[sub == node.idx] = EMPTY
            if self._hall_pos is None and node.has("hall"):
                self._hall_pos = (oy + (node.min_y + node.max_y) // 2,
                                  ox + (node.min_x + node.max_x) // 2)
            full = self._masks(shell, self._hall_pos)
            gy, gx = slice(oy + node.min_y, oy + node.max_y), slice(ox + node.min_x, ox + node.max_x)
            sub_masks = {k: v[gy, gx] for k, v in full.items()}
            # лучшая из нескольких нарезок зоны (у ProcTHOR - только лучший план целиком)
            base = sub.copy()
            best, last, n_ok = None, None, 0
            for _ in range(self.cfg.apartment_zone_trials):
                sub[...] = base
                try:
                    self._expand(shell, node.children, sub, sub_masks, oy + node.min_y,
                                 ox + node.min_x, top=False)
                    zone = (base == EMPTY)
                    self._fill_zone(sub, zone, {x.idx for x in node.children})
                    score = sum(self._shape(x, sub == x.idx) for x in node.leaves())
                except InvalidPlan as e:
                    last = e
                    continue
                if best is None or score > best[0]:
                    best = (score, sub.copy())
                n_ok += 1
                if n_ok >= 3:
                    break
            if best is None:
                raise last or InvalidPlan("зона не нарезалась")
            sub[...] = best[1]
        if top:
            self._fill_rest(fp)

    def generate(self) -> Layout:
        cfg = self.cfg
        last = None
        for _ in range(cfg.apartment_shell_trials):
            prog, tree = self._program()
            top, leaves = self._build_nodes(tree)
            shell = self._shell([x.type for x in leaves], [x.target * self.c ** 2 for x in leaves])
            best, n_ok = None, 0
            for _ in range(cfg.apartment_candidates):
                fp = shell.grid.copy()
                top, leaves = self._build_nodes(tree)
                self._hall_pos = None
                try:
                    self._expand(shell, top, fp, self._masks(shell))
                    score, doors = self._check(shell, leaves, fp)
                except InvalidPlan as e:
                    last = str(e)
                    continue
                if best is None or score > best[0]:
                    best = (score, fp, leaves, doors)
                n_ok += 1
                if n_ok >= cfg.apartment_keep_best_of:
                    break
            if best is not None:
                try:
                    return self._to_layout(prog, shell, *best[1:])
                except InvalidPlan as e:
                    last = str(e)
        raise InvalidPlan(f"не удалось построить квартиру: {last}")

    # --- сетка -> стены -> Layout -----------------------------------------
    def _to_layout(self, prog, shell: _Shell, fp, rooms, doors) -> Layout:
        cfg, rng, c = self.cfg, self.rng, self.c
        H = round(self._u(cfg.ceiling_height_m), 3)
        t_fac = float(rng.choice(cfg.exterior_wall_m))
        t_party = float(rng.choice(cfg.apartment_party_wall_m))
        t_part = float(rng.choice((0.08, 0.1, 0.12)))
        t_wet = float(rng.choice((0.08, 0.1)))
        wet = {r.idx for r in rooms if TYPES[r.type].wet}

        g = np.pad(fp, 1, constant_values=OUT)
        ny, nx = fp.shape
        segs: dict[tuple, list] = {}

        def add(orient, coord, a, b, key):
            segs.setdefault((orient, round(coord, 6), key), []).append((a, b))

        for j in range(ny + 1):              # горизонтальные грани между строками j-1 и j
            for i in range(nx):
                lo, hi = g[j, i + 1], g[j + 1, i + 1]
                key = self._edge_type(lo, hi, "top", "bottom", shell, wet, t_fac, t_party, t_part, t_wet)
                if key:
                    add("h", j * c, i * c, (i + 1) * c, key)
        for i in range(nx + 1):              # вертикальные грани между столбцами i-1 и i
            for j in range(ny):
                lo, hi = g[j + 1, i], g[j + 1, i + 1]
                key = self._edge_type(lo, hi, "right", "left", shell, wet, t_fac, t_party, t_part, t_wet)
                if key:
                    add("v", i * c, j * c, (j + 1) * c, key)

        walls: list[Wall] = []
        for (orient, coord, key), parts in sorted(segs.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            kind, role, t, height = key
            parts.sort()
            merged = [list(parts[0])]
            for a, b in parts[1:]:
                if a <= merged[-1][1] + 1e-9:
                    merged[-1][1] = max(merged[-1][1], b)
                else:
                    merged.append([a, b])
            for a, b in merged:
                p0, p1 = ((coord, a), (coord, b)) if orient == "v" else ((a, coord), (b, coord))
                walls.append(Wall(len(walls), (round(p0[0], 6), round(p0[1], 6)),
                                  (round(p1[0], 6), round(p1[1], 6)), t, kind,
                                  height=height, role=role))
        # несущая внутренняя стена: самая длинная перегородка между сухими помещениями
        inner = [w for w in walls if w.kind == "interior" and w.role == "" and w.thickness == t_part]
        if inner and rng.random() < cfg.apartment_p_bearing_wall:
            longest = max(inner, key=lambda w: w.length)
            if longest.length > 3.0:
                longest.thickness = float(rng.choice((0.16, 0.18, 0.2)))
                longest.role = "bearing"
        _close_wall_ends(walls)

        lay_rooms, grid = rooms_from_walls(walls, H, 0.3)
        # помещение на сетке -> помещение между стенами (по перекрытию клеток)
        cells = {r.idx: self._cell_rects(fp, r.idx) for r in rooms}
        logg = self._cell_rects(fp, LOGGIA)
        comp_of = {}
        for room in lay_rooms:
            scores = {k: sum(label_overlap(room, rc) for rc in v) for k, v in cells.items()}
            lg = sum(label_overlap(room, rc) for rc in logg)
            k_best = max(scores, key=scores.get) if scores else None
            if lg > (scores.get(k_best, 0) if k_best is not None else 0):
                room.kind, room.name = "balcony", "Лоджия"
                continue
            t = TYPES[rooms[k_best].type]
            room.kind, room.name = t.kind, t.label
            if prog == "1k" and rooms[k_best].type == "bedroom":
                room.name = "Комната"
            comp_of[k_best] = room.id
            if rng.random() < cfg.p_suspended_ceiling and t.wet:
                room.ceiling_z = round(H - self._u(cfg.suspended_drop_m), 3)
        if len(comp_of) != len(rooms):
            raise InvalidPlan("помещения слились при построении стен")

        layout = Layout(H, lay_rooms, walls, footprint=footprint_rects(grid), meta={
            "source": "apartment", "program": prog, "shape": shell.meta.get("shape"),
            "facades": list(shell.facades), "entrance_side": shell.entrance,
            "loggia": shell.loggia, "t_ext": t_fac, "t_party": t_party, "t_part": t_part,
            "cell_m": c, "algorithm": "constrained growth (Lopes 2010) after ProcTHOR"})
        self._openings(layout, grid, rooms, comp_of, doors, shell)
        return layout

    def _edge_type(self, a, b, b_dir, a_dir, shell, wet, t_fac, t_party, t_part, t_wet):
        """Тип стены на грани между клетками a (ниже/левее) и b (выше/правее)."""
        if a == b:
            return None
        if (a >= 0 and b == LOGGIA) or (b >= 0 and a == LOGGIA):
            return ("interior", "balcony_interface", t_fac, None)
        if a >= 0 and b >= 0:
            t = t_wet if (a in wet or b in wet) else t_part
            return ("interior", "", t, None)
        # одна из сторон - комната или лоджия, другая - улица/вход/лоджия
        room_side, other, out_dir = (a, b, b_dir) if a >= 0 or a == LOGGIA else (b, a, a_dir)
        if other == OUT:
            if room_side == LOGGIA:
                if out_dir in shell.facades:
                    if shell.loggia and shell.loggia["glazed"]:
                        return ("exterior", "loggia_glazing", self.cfg.parapet_thickness_m, None)
                    return ("parapet", "", self.cfg.parapet_thickness_m,
                            round(self._u(self.cfg.parapet_height_m), 3))
                return ("exterior", "party", 0.16, None)
            if out_dir in shell.facades:
                return ("exterior", "facade", t_fac, None)
            return ("exterior", "party", t_party, None)
        return None

    def _cell_rects(self, fp, value) -> list:
        """Клетки со значением value -> прямоугольники в метрах."""
        c = self.c
        return [(i * c, j * c, (i + 1) * c, (j + 1) * c) for j, i in np.argwhere(fp == value)]

    def _openings(self, layout: Layout, grid, rooms, comp_of, doors, shell) -> None:
        cfg, rng = self.cfg, self.rng
        int_spans, ext_spans, balcony_spans = wall_spans(layout, grid)
        gen = LayoutGenerator(cfg, rng)
        by_pair: dict = {}
        for wid, a, b, pair in int_spans:
            by_pair.setdefault(pair, []).append((wid, a, b))

        def place(wid, a, b, width, margin, kind, z1, rooms_, swing_room=None, bias=0.7):
            """Дверь или арка; полотно открывается в swing_room (None - наружу)."""
            s = gen._try_place(layout, wid, a, b, width, margin, near_end_bias=bias)
            if s is None:
                return False
            wall = layout.wall(wid)
            extra = {}
            if kind == "door":
                if swing_room is None:
                    swing = wall.outward or 1
                else:
                    p = wall.point(s, wall.thickness / 2 + 0.02)
                    swing = 1 if layout.rooms[swing_room].contains(*p) else -1
                extra = {"hinge": int(rng.choice((-1, 1))), "swing": swing}
            layout.openings.append(Opening(len(layout.openings), wid, kind, s, width, 0.0, z1,
                                           rooms=tuple(rooms_), **extra))
            return True

        head = round(min(self._u(cfg.door_head_m), layout.ceiling_height - 0.1), 3)
        for parent, child, kind in doors:
            ca, cb = comp_of[parent], comp_of[child]
            spans = by_pair.get((min(ca, cb), max(ca, cb)), [])
            spans = sorted(spans, key=lambda x: x[2] - x[1], reverse=True)
            t = TYPES[rooms[child].type]
            width = round(self._u(t.door), 2) if kind == "door" else round(self._u((1.0, 1.6)), 2)
            for wid, a, b in spans:
                if place(wid, a, b, width, 0.15, kind, head if kind == "door" else
                         round(min(self._u(cfg.passage_head_m), layout.ceiling_height - 0.1), 3),
                         (ca, cb), swing_room=cb):
                    break
            else:
                raise InvalidPlan(f"дверь {rooms[parent].type}->{rooms[child].type} не встала")

        # вход: на межквартирной стене прихожей
        hall = comp_of[next(r.idx for r in rooms if r.type == "hall")]
        party = [sp for sp in ext_spans if sp[3] == hall and layout.wall(sp[0]).role == "party"]
        party.sort(key=lambda x: x[2] - x[1], reverse=True)
        width = round(self._u(cfg.entrance_width_m), 2)
        # входная дверь открывается наружу, на лестничную площадку
        if not any(place(wid, a, b, width, 0.2, "door", head, (hall,), swing_room=None, bias=0.3)
                   for wid, a, b, _ in party):
            raise InvalidPlan("вход не встал")

        # окна: по центру фасадного участка каждой жилой комнаты и кухни
        sill = round(self._u(cfg.window_sill_m), 3)
        whead = round(min(self._u(cfg.window_head_m), layout.ceiling_height - 0.15), 3)
        windowed = {comp_of[r.idx] for r in rooms if TYPES[r.type].window}
        for wid, a, b, rid in ext_spans:
            wall, room = layout.wall(wid), layout.rooms[rid]
            if wall.role != "facade" or rid not in windowed:
                continue
            L = b - a
            if L < 1.2:
                continue
            n = 2 if (L > 5.0 and room.area > 16) else 1
            for k in range(n):
                seg_a, seg_b = a + L * k / n, a + L * (k + 1) / n
                width = round(min(self._u((1.2, 1.8)), seg_b - seg_a - 0.6), 2)
                if width < 0.6:
                    continue
                s = round((seg_a + seg_b) / 2 + self._u((-0.15, 0.15)), 3)
                if gen._fits(layout, wid, s, width):
                    layout.openings.append(Opening(len(layout.openings), wid, "window", s, width,
                                                   sill, whead, rooms=(rid,)))

        # лоджия: балконный блок в одно помещение, окна - в остальные
        if balcony_spans:
            by_room: dict = {}
            for wid, a, b, pair in balcony_spans:
                rid = pair[0] if layout.rooms[pair[0]].kind != "balcony" else pair[1]
                by_room.setdefault(rid, []).append((wid, a, b, pair))
            prio = {"kitchen": 0, "room": 1}
            order = sorted(by_room, key=lambda r: (prio.get(layout.rooms[r].kind, 2),
                                                   -max(x[2] - x[1] for x in by_room[r])))
            for n_room, rid in enumerate(order):
                wid, a, b, pair = max(by_room[rid], key=lambda x: x[2] - x[1])
                wall = layout.wall(wid)
                if n_room == 0 and b - a >= 0.75 + 0.4:
                    dw = 0.75
                    s_d = round(a + 0.2 + dw / 2, 3) if rng.random() < 0.5 else round(b - 0.2 - dw / 2, 3)
                    swing = 1 if layout.rooms[rid].contains(*wall.point(s_d, wall.thickness / 2 + 0.02)) else -1
                    layout.openings.append(Opening(len(layout.openings), wid, "door", s_d, dw, 0.0, head,
                                                   rooms=pair, hinge=int(rng.choice((-1, 1))), swing=swing))
                ww = round(min(self._u((1.0, 1.6)), b - a - 0.5), 2)
                if ww >= 0.6:
                    s = gen._try_place(layout, wid, a, b, ww, 0.15)
                    if s is not None:
                        layout.openings.append(Opening(len(layout.openings), wid, "window", s, ww,
                                                       sill, whead, rooms=pair))
        # остеклённая лоджия: панели остекления по всей длине над ограждением
        for w in layout.walls:
            if w.role != "loggia_glazing":
                continue
            ph = round(self._u(cfg.parapet_height_m), 3)
            top = round(layout.ceiling_height - 0.25, 3)
            n = max(1, int(round(w.length / 1.4)))
            pane = (w.length - 0.1 * (n + 1)) / n
            for k in range(n):
                s = 0.1 + pane / 2 + k * (pane + 0.1)
                layout.openings.append(Opening(len(layout.openings), w.id, "window", round(s, 3),
                                               round(pane, 3), ph, top, rooms=()))


def _corners(mask: np.ndarray) -> int:
    """Число вершин ортогонального контура (по окнам 2x2: 1 или 3 клетки - угол,
    2 клетки по диагонали - два угла)."""
    m = np.pad(mask.astype(np.int8), 1)
    a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
    s = a + b + c + d
    diag = (s == 2) & (a == d)
    return int(((s == 1) | (s == 3)).sum() + 2 * diag.sum())


def _close_wall_ends(walls: list[Wall]) -> None:
    """Удлинить концы стен на половину толщины перпендикулярных стен, сходящихся в этой точке:
    закрывает углы и стыки без щелей (на прямых продолжениях - на 1 см)."""
    for w in walls:
        for end, p in ((0, np.asarray(w.p0)), (1, np.asarray(w.p1))):
            ext = 0.01
            for v in walls:
                if v is w or abs(float(np.dot(v.u, w.u))) > 0.5:
                    continue
                rel = p - np.asarray(v.p0)
                s, off = float(rel @ v.u), float(rel @ v.n)
                if abs(off) < 1e-6 and -1e-6 <= s <= v.length + 1e-6:
                    ext = max(ext, v.thickness / 2)
            if end == 0:
                w.ext0 = ext
            else:
                w.ext1 = ext
