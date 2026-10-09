"""Адаптер datasetgen -> Layout.

Первые тесты - на «ручном» плане в формате extract_plan (datasetgen не нужен).
Интеграционные - на настоящем datasetgen.py: путь в DATASETGEN_PATH
(или рядом лежащий клон ReFloorBRUSNIKA), иначе пропускаются.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from simscan.config import load_config
from simscan.datasetgen_adapter import layout_from_plan
from simscan.groundtruth import UNET_VALUES, build_masks, frame_for, unet_frame, unet_mask
from simscan.interior import Furnisher
from simscan.scene import build_scene


def _wall(x1, y1, x2, y2, th, ext=False, guard=False, iface=False):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "thickness": th, "is_external": ext,
            "is_guardrail": guard, "is_balcony_interface": iface}


# 10 x 8 м при 1 см/px; ось Y картинки вниз. Справа балкон 1 м за стеной x = 9 м.
PLAN = {
    "source": "datasetgen", "seed": 0, "strategy": "manual", "size_px": 1200,
    "bounds_px": [0, 0, 1000, 800], "ext_th_px": 40, "int_th_px": 16,
    "walls": [
        _wall(0, 0, 1000, 0, 40, ext=True), _wall(0, 800, 1000, 800, 40, ext=True),
        _wall(0, 0, 0, 800, 40, ext=True),
        _wall(1000, 0, 1000, 800, 2, ext=True, guard=True),
        _wall(900, 0, 900, 800, 40, iface=True),
        _wall(500, 0, 500, 800, 16), _wall(0, 400, 500, 400, 16),
    ],
    "rooms": [{"bounds": [0, 0, 500, 400], "type": "bedroom"},
              {"bounds": [0, 400, 500, 800], "type": "kitchen"},
              {"bounds": [500, 0, 900, 800], "type": "living"}],
    "balconies": [{"bounds": [900, 0, 1000, 800], "side": "right"}],
}


def _layout(seed=0, **layout_over):
    cfg = load_config(overrides={"layout": {"datasetgen_long_side_m": [10.0, 10.0],
                                            "exterior_wall_m": [0.4], "interior_wall_m": [0.1],
                                            **layout_over}})
    rng = np.random.default_rng(seed)
    return layout_from_plan(PLAN, cfg.layout, rng), cfg, rng


def _room_with(lay, x, y):
    rid = lay.room_at(x, y)
    assert rid is not None
    return lay.rooms[rid]


def test_rooms_are_regions_between_walls():
    lay, _, _ = _layout()
    assert len(lay.rooms) == 4
    # спальня (левый верх картинки = левый верх плана): x 0,2..4,95, y 4,05..7,8
    bed = _room_with(lay, 2.0, 6.0)
    assert bed.area == pytest.approx(4.75 * 3.75, abs=1e-6)
    assert _room_with(lay, 2.0, 2.0).kind == "kitchen"
    balcony = _room_with(lay, 9.5, 4.0)
    assert balcony.kind == "balcony"
    assert balcony.area == pytest.approx((9.94 - 9.2) * 7.6, abs=1e-6)


def test_walls_converted_to_metres():
    lay, _, _ = _layout()
    kinds = sorted(w.kind for w in lay.walls)
    assert kinds.count("parapet") == 1 and kinds.count("exterior") == 3
    par = next(w for w in lay.walls if w.kind == "parapet")
    assert 1.0 <= par.height <= 1.1 and par.thickness == pytest.approx(0.12)
    iface = next(w for w in lay.walls if w.role == "balcony_interface")
    assert iface.thickness == pytest.approx(0.4)


def test_openings_physically_placed():
    for seed in range(20):
        lay, _, _ = _layout(seed)
        for o in lay.openings:
            w = lay.wall(o.wall_id)
            assert -1e-6 <= o.s - o.width / 2 and o.s + o.width / 2 <= w.length + 1e-6
            assert w.kind != "parapet"
            if o.kind == "window":
                assert w.kind == "exterior" or w.role == "balcony_interface"
                assert all(lay.rooms[r].kind != "balcony" for r in o.rooms) \
                    or w.role == "balcony_interface"
        # балконный блок: дверь и окно на стене балкона
        iface_ops = [o.kind for o in lay.openings if lay.wall(o.wall_id).role == "balcony_interface"]
        assert "door" in iface_ops


def test_masks_and_unet_mask():
    lay, cfg, rng = _layout()
    # колонны на плане - тоже стены; здесь мешают мерить толщину стены
    interior = load_config(overrides={"interior": {"p_column": 0.0, "p_pilaster": 0.0}}).interior
    Furnisher(interior, rng).furnish(lay)
    _, solids = build_scene(lay)
    frame = frame_for(lay, 10, 0.5)
    m = build_masks(lay, solids, frame, 1.25)
    # ограждение балкона ниже секущей плоскости, но на плане - стена
    i, j = frame.xy_to_ij(10.0, 4.0)
    assert m["walls"][int(round(i)), int(round(j))]
    uf = unet_frame(lay, 30.0, 0.5)
    assert uf.width == uf.height
    um = unet_mask(lay, solids, uf, 1.25)
    assert set(np.unique(um)) <= set(UNET_VALUES.values())
    # толщина самой толстой стены в пикселях ~ 30 (как меряет core/scale_norm.py ReFloor)
    import cv2

    walls = (um == UNET_VALUES["wall"]).astype(np.uint8)
    dt = cv2.distanceTransform(np.pad(walls, 1), cv2.DIST_L2, 5)
    vals = dt[dt > 0]
    estimate = 2 * np.median(vals[vals > np.percentile(vals, 85)])   # estimate_wall_stroke_thickness_px
    assert abs(estimate - 30) <= 3


# --- интеграция с настоящим datasetgen ----------------------------------------

def _datasetgen_path():
    env = os.environ.get("DATASETGEN_PATH")
    cands = [env] if env else []
    cands += [str(Path(__file__).resolve().parents[2] / "bykovvadimcv/refloorbrusnika/datasetgen")]
    for c in cands:
        if c and (Path(c) / "datasetgen.py").exists() or (c and c.endswith(".py") and Path(c).exists()):
            return c
    return None


DG = _datasetgen_path()
needs_dg = pytest.mark.skipif(DG is None, reason="datasetgen.py не найден (DATASETGEN_PATH)")


@needs_dg
def test_real_datasetgen_plans_convert():
    from simscan.datasetgen_adapter import extract_plan, load_datasetgen

    dg = load_datasetgen(DG)
    cfg = load_config()
    for seed in range(30):
        plan = extract_plan(dg, seed)
        rng = np.random.default_rng(seed)
        lay = layout_from_plan(plan, cfg.layout, rng)
        Furnisher(cfg.interior, rng).furnish(lay)
        _, solids = build_scene(lay)
        m = build_masks(lay, solids, frame_for(lay, 20, 0.5), 1.25)
        assert m["walls"].any() and len(lay.rooms) >= 1


@needs_dg
def test_real_datasetgen_scene_points_on_surfaces(tmp_path):
    import json

    import pye57

    from simscan.generate import generate_scene
    from simscan.layout import Layout
    from simscan.scene import surface_distance
    from simscan.transform import WorldTransform

    from conftest import CLEAN_EFFECTS, CLEAN_SCENE

    cfg = load_config(overrides={
        "layout": {"source": "datasetgen", "datasetgen_path": DG},
        "scanner": {"angular_step_deg": 0.5},
        "effects": dict(CLEAN_EFFECTS), "export": {"preview": False}, **CLEAN_SCENE})
    generate_scene(cfg, tmp_path / "s", seed=3, index=1)
    doc = json.loads((tmp_path / "s" / "layout.json").read_text(encoding="utf-8"))
    lay = Layout.from_dict(doc)
    w = doc["meta"]["layout_to_world"]
    world = WorldTransform(w["yaw_deg"], tuple(w["offset_m"]))
    _, solids = build_scene(lay)
    with pye57.E57(str(tmp_path / "s" / "scan.e57")) as e57:
        d = e57.read_scan(0, ignore_missing_fields=True)
    p = world.to_layout(np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]])
    p = p[np.random.default_rng(0).choice(len(p), min(len(p), 20000), replace=False)]
    assert surface_distance(p, solids).max() < 1e-4
    assert (tmp_path / "s" / "gt" / "unet_mask.png").exists()
