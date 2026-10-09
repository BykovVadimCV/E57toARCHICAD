"""План из сведённого облака: две комнаты, перегородка с дверью, окно в наружной стене."""
import math

import numpy as np

from simscan.e57io import write_merged_e57
from simscan.floorplan import floorplan

STEP = 0.02
TOP = 2.8


def _grid(u0, u1, v0, v1):
    u, v = np.meshgrid(np.arange(u0, u1, STEP), np.arange(v0, v1, STEP))
    return u.ravel(), v.ravel()


def _wall_x(x, y0, y1, holes=()):
    y, z = _grid(y0, y1, 0.0, TOP)
    keep = np.ones(len(y), bool)
    for (a, b, lo, hi) in holes:
        keep &= ~((y > a) & (y < b) & (z > lo) & (z < hi))
    return np.c_[np.full(keep.sum(), x), y[keep], z[keep]]


def _wall_y(y, x0, x1, holes=()):
    x, z = _grid(x0, x1, 0.0, TOP)
    keep = np.ones(len(x), bool)
    for (a, b, lo, hi) in holes:
        keep &= ~((x > a) & (x < b) & (z > lo) & (z < hi))
    return np.c_[x[keep], np.full(keep.sum(), y), z[keep]]


def _apartment():
    door = (1.5, 2.4, 0.0, 2.1)
    window = (1.0, 2.4, 0.9, 2.3)
    parts = [_wall_x(0.0, 0, 4), _wall_x(8.0, 0, 4), _wall_y(0.0, 0, 8), _wall_y(4.0, 0, 8, [window]),
             _wall_x(3.95, 0, 4, [door]), _wall_x(4.05, 0, 4, [door])]
    # торцы проёма двери: перемычка снизу и откосы
    x, y = _grid(3.95, 4.05, 1.5, 2.4)
    parts.append(np.c_[x, y, np.full(len(x), 2.1)])
    for yy in (1.5, 2.4):
        x, z = _grid(3.95, 4.05, 0.0, 2.1)
        parts.append(np.c_[x, np.full(len(x), yy), z])
    for zz in (0.0, TOP):
        x, y = _grid(0, 8, 0, 4)
        m = ~((x > 3.95) & (x < 4.05) & ((zz > 0) | (y < 1.5) | (y > 2.4)))
        parts.append(np.c_[x[m], y[m], np.full(m.sum(), zz)])
    p = np.vstack(parts)
    p += np.random.default_rng(0).normal(0, 0.002, p.shape)
    a = math.radians(12)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    return p @ R.T + [3.0, -2.0, 10.0]


def test_rooms_doors_windows_and_closed_contours(tmp_path):
    p = _apartment()
    path = tmp_path / "apt.e57"
    write_merged_e57(path, p, np.full(len(p), 0.5, np.float32))
    res = floorplan(path, tmp_path / "out", px=0.02, log=lambda *a: None)
    lv = res["levels"][0]
    areas = sorted(r["area"] for r in lv["rooms"])
    assert len(areas) == 2, areas
    for a in areas:
        assert abs(a - 15.8) < 1.2, areas
    doors = [o for o in lv["openings"] if o["opening_type"] == "door"]
    windows = [o for o in lv["openings"] if o["opening_type"] == "window"]
    assert len(doors) == 1 and abs(doors[0]["width"] - 0.9) < 0.12 and abs(doors[0]["head"] - 2.1) < 0.15
    assert len(windows) == 1 and abs(windows[0]["width"] - 1.4) < 0.15 and abs(windows[0]["sill"] - 0.9) < 0.15
    # перегородка - стена между двумя помещениями с толщиной между гранями
    inner = [w for w in lv["walls"] if len(w["rooms"]) == 2]
    assert inner and all(abs(w["thickness"] - 0.1) < 0.05 for w in inner), inner
    # контур помещения - четыре прямые грани, без зубцов у окна
    for r in lv["rooms"]:
        assert all(f["kind"] == "line" for f in r["faces"]) and len(r["faces"]) <= 6, r["faces"]
    assert (tmp_path / "out" / "plan.png").exists() and (tmp_path / "out" / "walls.png").exists()
