import numpy as np

from simscan.groundtruth import RasterFrame, build_masks, frame_for, raster_box
from simscan.interior import Furnisher
from simscan.layout import Box, LayoutGenerator
from simscan.scene import build_scene

from conftest import box_room_config


def _box_room(w=5.0, d=3.0, seed=0):
    cfg = box_room_config(w, d)
    rng = np.random.default_rng(seed)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    Furnisher(cfg.interior, rng).furnish(lay)
    return lay


def test_room_area_exact_in_pixels():
    lay = _box_room(5.0, 3.0)
    _, solids = build_scene(lay)
    frame = frame_for(lay, 10, 0.5)
    m = build_masks(lay, solids, frame, 1.25)
    assert (m["rooms"] == 1).sum() == 500 * 300
    t = lay.meta["t_ext"]
    outer = round((5.0 + 2 * t) * 100) * round((3.0 + 2 * t) * 100)
    assert m["walls"].sum() == outer - 500 * 300


def test_scale_side_length():
    """Коробка 5,000 x 3,000 м: длина стороны по маске = 5,000 м с ошибкой меньше 1 px."""
    lay = _box_room(5.0, 3.0)
    _, solids = build_scene(lay)
    frame = frame_for(lay, 10, 0.5)
    rooms = build_masks(lay, solids, frame, 1.25)["rooms"] == 1
    rows, cols = np.nonzero(rooms)
    x0, _ = frame.ij_to_xy(0, cols.min())
    x1, _ = frame.ij_to_xy(0, cols.max())
    width_m = (x1 - x0) + frame.pixel_m
    assert abs(width_m - 5.0) < frame.pixel_m


def test_pixel_formula_roundtrip():
    f = RasterFrame(-1.23, 4.56, 0.01, 640, 480)
    i, j = f.xy_to_ij(2.345, 6.789)
    x, y = f.ij_to_xy(i, j)
    assert abs(x - 2.345) < 1e-9 and abs(y - 6.789) < 1e-9
    # нижняя строка растра - самый малый Y
    _, y_bottom = f.ij_to_xy(f.height - 1, 0)
    assert abs(y_bottom - (f.origin_y + 0.5 * f.pixel_m)) < 1e-12


def test_rotated_box_area():
    f = RasterFrame(-2, -2, 0.005, 800, 800)
    m = np.zeros((800, 800), bool)
    raster_box(f, m, Box((0.0, 0.0, 0.0), (1.2, 0.4, 1.0), 0.6))
    assert abs(m.sum() * f.pixel_m ** 2 - 0.48) < 0.48 * 0.01


def test_openings_cut_from_walls(coarse_config):
    rng = np.random.default_rng(5)
    lay = LayoutGenerator(coarse_config.layout, rng).generate()
    Furnisher(coarse_config.interior, rng).furnish(lay)
    _, solids = build_scene(lay)
    frame = frame_for(lay, 10, 0.5)
    m = build_masks(lay, solids, frame, 1.25)
    assert not (m["walls"] & m["doors"]).any()
    assert not (m["walls"] & m["windows"]).any()
    n_doors = sum(o.kind != "window" for o in lay.openings)
    if n_doors:
        assert m["doors"].sum() > 0
