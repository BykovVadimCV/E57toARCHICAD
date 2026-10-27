"""Открытый балкон (лоджия без потолка, пол ниже, ограждения вместо боковых стен): геометрия сцены,
стоянка над его полом, на плане и в разметке - улица."""
import json

import cv2
import numpy as np
import pytest

from simscan.config import load_config
from simscan.generate import generate_scene, make_layout, scene_rng
from simscan.interior import Furnisher
from simscan.netinput import OUTSIDE, SceneGT, build_scene, world_to_px
from simscan.scanner import place_stations
from simscan.scene import build_scene as build_mesh

OVER = {"layout": {"apartment_p_loggia": 1.0, "apartment_p_open_balcony": 1.0, "p_duplex": 0.0},
        "scanner": {"angular_step_deg": 0.5},
        "interior": {"furniture": False, "clutter_per_m2": 0.0},
        "export": {"preview": False, "write_merged": False, "free_space_input": False, "unet_mask": False}}


def _first_with_balcony(cfg, seed=5):
    for i in range(20):
        try:
            lay = make_layout(cfg, scene_rng(seed, i), seed, i)
        except Exception:                                    # планировка не построилась - следующая
            continue
        if any(r.open_sky for r in lay.rooms):
            return i
    pytest.skip("нет планировки с открытым балконом")


def test_open_balcony_geometry():
    cfg = load_config(overrides=OVER)
    i = _first_with_balcony(cfg)
    rng = scene_rng(5, i)
    lay = make_layout(cfg, rng, 5, i)
    b = next(r for r in lay.rooms if r.open_sky)
    assert -0.25 - 1e-6 <= b.floor_z <= -0.12 + 1e-6 and b.kind == "balcony"
    rails = [w for w in lay.walls if w.role == "open_balcony"]
    assert rails and all(w.kind == "parapet" and w.base == b.floor_z for w in rails)
    assert all(0.95 <= w.height - b.floor_z <= 1.15 for w in rails)          # ограждение от своего пола
    grids = Furnisher(cfg.interior, rng).furnish(lay)
    st = [s for s in place_stations(lay, grids, cfg.scanner, rng) if s.room_id == b.id]
    assert st and all(abs(s.position[2] - b.floor_z - np.mean(cfg.scanner.height_m)) < 0.3 for s in st)
    _, solids = build_mesh(lay)
    cx, cy = (b.cell[0] + b.cell[2]) / 2, (b.cell[1] + b.cell[3]) / 2

    def over(s):
        c, h = s.box.center, np.asarray(s.box.size) / 2
        return abs(cx - c[0]) <= h[0] and abs(cy - c[1]) <= h[1]

    floors = [s for s in solids if s.label == "floor" and over(s)]
    ceils = [s for s in solids if s.label == "ceiling" and over(s)]
    assert len(floors) == 1 and abs(floors[0].box.center[2] + floors[0].box.size[2] / 2 - b.floor_z) < 1e-6
    assert not ceils                                                           # над балконом - небо


def test_open_balcony_is_outside_in_labels(tmp_path):
    cfg = load_config(overrides=OVER)
    i = _first_with_balcony(cfg)
    generate_scene(cfg, tmp_path / "scene", seed=5, index=i)
    build_scene(tmp_path / "scene", tmp_path / "ds", log=lambda *a: None)
    d = tmp_path / "ds" / "scenes" / "scene" / "L1"
    sem = cv2.imread(str(d / "sem.png"), cv2.IMREAD_UNCHANGED)
    sc = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    gt = SceneGT(tmp_path / "scene")
    b = next(r for r in gt.layout.rooms if r.open_sky)
    for x, y in ((b.clear[0] + 0.2, b.clear[1] + 0.2), (b.clear[2] - 0.2, b.clear[3] - 0.2)):
        j, k = np.round(world_to_px(gt.xy([(x, y)]), sc["frame"])[0]).astype(int)
        assert sem[k, j] == OUTSIDE
    v = json.loads((d / "vector.json").read_text(encoding="utf-8"))
    assert all(r["kind"] != "balcony" for r in v["rooms"]) and all(w["kind"] != "parapet" for w in v["walls"])
    x = np.load(d / "input.npy").astype(np.float32)
    m = np.zeros(sem.shape, np.uint8)
    q = world_to_px(gt.xy([(b.clear[0], b.clear[1]), (b.clear[2], b.clear[1]), (b.clear[2], b.clear[3]),
                           (b.clear[0], b.clear[3])]), sc["frame"])
    cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1)
    below, floor = (x[7][m > 0] > 0).mean(), (x[3][m > 0] > 0).mean()     # пол ниже уровня, не его пол
    assert below > 0.03 and below > 3 * floor, (below, floor)               # (скан тут грубый: 0,5°)
