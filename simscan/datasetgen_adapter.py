"""Планировки из datasetgen (ReFloorBRUSNIKA) -> Layout simscan.

datasetgen живёт в пикселях и рисует двери и окна сразу в растр: в структурах
данных есть только стены (оси, толщина в px, наружная / ограждение / стена балкона),
прямоугольники комнат-меток и балконы. Поэтому:

  1. extract_plan() повторяет геометрическую часть generate_single_plan
     (стратегия + балконы) и отдаёт словарь «plan» - его же можно сохранить
     в JSON на стороне datasetgen и подать сюда без импорта модуля;
  2. layout_from_plan() переводит px в метры (длинная сторона здания
     разыгрывается в метрах), толщины стен разыгрывает заново в метрах
     (в px они от 3 до 200 и физического смысла не имеют, но тонкие перегородки
     остаются тонкими), помещения строит как связные области между стенами
     (у части стратегий прямоугольники комнат не покрывают план или
     перекрываются), а двери и окна ставит сам - уже с высотами.

Стены - первичная геометрия: прямоугольники комнат используются только для
типа помещения (кухня, санузел, коридор, балкон).
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np

from .config import LayoutConfig
from .layout import Layout, LayoutGenerator, Wall
from .walls2layout import footprint_rects, place_balcony_blocks, rooms_from_walls, wall_spans

DEFAULT_STRATEGIES = ("central", "linear", "radial", "graph", "open_plan",
                      "asymmetric", "courtyard", "spine")
ROOM_KIND = {"bathroom": "bath", "laundry": "bath", "kitchen": "kitchen",
             "corridor": "corridor", "walk_in_closet": "corridor"}



# ----------------------------------------------------------------------
# 1. Извлечение геометрии из datasetgen
# ----------------------------------------------------------------------

def load_datasetgen(path: str | Path):
    """Импорт datasetgen.py по пути к файлу или к каталогу с ним."""
    path = Path(path)
    folder = path.parent if path.suffix == ".py" else path
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    return importlib.import_module("datasetgen")


def extract_plan(dg, seed: int, strategies=DEFAULT_STRATEGIES,
                 balcony_probability: float | None = None) -> dict:
    """Геометрия одного плана datasetgen (без рендера) в виде словаря."""
    kwargs = {"seed": seed, "strategies": list(strategies)}
    if balcony_probability is not None:
        kwargs["balcony_probability"] = balcony_probability
    cfg = dg.FloorPlanConfig(**kwargs)
    gen = dg.SmartFloorPlanGenerator(cfg)
    rng = gen.rng
    # толщины в px - как в generate_dataset; нужны только их отношение
    ext_th = rng.randint(*cfg.wall_thickness_range) * gen.super_sampling
    if rng.random() < cfg.thin_partition_prob:
        ratio = rng.uniform(*cfg.thin_partition_ratio_range)
    else:
        ratio = rng.uniform(*cfg.int_wall_ratio_range)
    int_th = max(int(ext_th * ratio), cfg.int_wall_min_px * gen.super_sampling)

    margin = int(gen.size * rng.uniform(0.12, 0.28))
    bounds = (margin, margin, gen.size - margin, gen.size - margin)
    brief = gen.generate_brief()
    strategy = gen.strategies[brief.circulation_type]
    walls, rooms = strategy.generate(bounds, ext_th, int_th, gen.solver, brief, cfg)
    balconies = []
    if brief.has_balcony:
        walls, rooms, balconies = gen._integrate_balconies(walls, rooms, ext_th)
    return {
        "source": "datasetgen", "seed": seed, "strategy": brief.circulation_type,
        "size_px": gen.size, "bounds_px": list(bounds),
        "ext_th_px": ext_th, "int_th_px": int_th,
        "walls": [{"x1": w.x1, "y1": w.y1, "x2": w.x2, "y2": w.y2, "thickness": w.thickness,
                   "is_external": bool(w.is_external), "is_guardrail": bool(w.is_guardrail),
                   "is_balcony_interface": bool(w.is_balcony_interface)} for w in walls],
        "rooms": [{"bounds": [r.x1, r.y1, r.x2, r.y2], "type": r.room_type} for r in rooms],
        "balconies": [{"bounds": [min(b.x1, b.x2), min(b.y1, b.y2), max(b.x1, b.x2),
                                  max(b.y1, b.y2)], "side": b.side} for b in balconies],
    }


# ----------------------------------------------------------------------
# 2. Перевод в Layout
# ----------------------------------------------------------------------
def layout_from_plan(plan: dict, cfg: LayoutConfig, rng: np.random.Generator) -> Layout:
    left, top, right, bottom = plan["bounds_px"]
    long_px = max(right - left, bottom - top)
    scale = float(rng.uniform(*cfg.datasetgen_long_side_m)) / long_px

    def xy(px, py):                      # px -> м, ось Y вверх
        return round((px - left) * scale, 4), round((bottom - py) * scale, 4)

    t_ext = float(rng.choice(cfg.exterior_wall_m))
    thin = plan["int_th_px"] < 0.4 * plan["ext_th_px"]
    pool = [t for t in cfg.interior_wall_m if (t <= 0.12 if thin else t >= 0.12)] or list(cfg.interior_wall_m)
    t_int = float(rng.choice(pool))
    t_par = cfg.parapet_thickness_m
    H = round(float(rng.uniform(*cfg.ceiling_height_m)), 3)
    W, D = (right - left) * scale, (bottom - top) * scale

    # --- стены -----------------------------------------------------------
    walls: list[Wall] = []
    seen = set()
    for w in plan["walls"]:
        a, b = xy(w["x1"], w["y1"]), xy(w["x2"], w["y2"])
        if a == b:
            continue
        a, b = (a, b) if a <= b else (b, a)
        if abs(a[0] - b[0]) > 1e-6 and abs(a[1] - b[1]) > 1e-6:
            continue                     # наклонных стен в datasetgen нет; на всякий случай
        if w["is_guardrail"]:
            kind, t, height = "parapet", t_par, round(float(rng.uniform(*cfg.parapet_height_m)), 3)
        elif w["is_external"]:
            kind, t, height = "exterior", t_ext, None
        elif w["is_balcony_interface"]:
            kind, t, height = "interior", t_ext, None
        else:
            kind, t, height = "interior", t_int, None
        key = (a, b, kind)
        if key in seen:
            continue
        seen.add(key)
        wall = Wall(len(walls), a, b, t, kind, height=height,
                    role="balcony_interface" if w["is_balcony_interface"] else "")
        # наружные стены лежат осью на контуре: углы замыкаем удлинением на t/2
        for end, p in ((0, a), (1, b)):
            at_corner = (abs(p[0]) < 1e-6 or abs(p[0] - W) < 1e-6) and \
                        (abs(p[1]) < 1e-6 or abs(p[1] - D) < 1e-6)
            ext = t / 2 if (kind != "interior" and at_corner) else 0.01
            if end == 0:
                wall.ext0 = ext
            else:
                wall.ext1 = ext
        walls.append(wall)

    rooms, grid = rooms_from_walls(walls, H, cfg.datasetgen_min_room_m2)
    _assign_kinds(plan, rooms, xy, cfg, rng)

    layout = Layout(H, rooms, walls, footprint=footprint_rects(grid),
                    meta={"source": "datasetgen", "datasetgen": {
                        k: plan[k] for k in ("seed", "strategy", "size_px", "bounds_px",
                                             "ext_th_px", "int_th_px")},
                        "scale_m_per_px": scale, "footprint_m": [W, D], "t_ext": t_ext,
                        "t_int": t_int})

    # --- проёмы ----------------------------------------------------------
    int_spans, ext_spans, balcony_spans = wall_spans(layout, grid)
    gen = LayoutGenerator(cfg, rng)
    gen._place_doors(layout, int_spans)
    place_balcony_blocks(layout, balcony_spans, cfg, rng)
    ext_spans = [sp for sp in ext_spans if layout.rooms[sp[3]].kind != "balcony"]
    gen._place_entrance(layout, ext_spans)
    gen._place_windows(layout, ext_spans)
    return layout


def _assign_kinds(plan, rooms, xy, cfg, rng) -> None:
    """Тип помещения - по прямоугольнику-метке datasetgen с наибольшим перекрытием."""
    def overlap(r, rect):
        x0, y0, x1, y1 = rect
        return sum(max(0.0, min(c, x1) - max(a, x0)) * max(0.0, min(d, y1) - max(b, y0))
                   for a, b, c, d in r.rects())

    def to_m(bounds):
        (ax, ay), (bx, by) = xy(bounds[0], bounds[1]), xy(bounds[2], bounds[3])
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    labels = [(to_m(r["bounds"]), r["type"]) for r in plan["rooms"]]
    balconies = [to_m(b["bounds"]) for b in plan["balconies"]]
    for room in rooms:
        if any(overlap(room, b) > 0.5 * room.area for b in balconies):
            room.kind = "balcony"
            continue
        best = max(labels, key=lambda lt: overlap(room, lt[0]), default=None)
        if best and overlap(room, best[0]) > 0:
            room.kind = ROOM_KIND.get(best[1], "room")
        if rng.random() < cfg.p_suspended_ceiling and room.kind != "balcony":
            room.ceiling_z = round(room.ceiling_z - float(rng.uniform(*cfg.suspended_drop_m)), 3)


