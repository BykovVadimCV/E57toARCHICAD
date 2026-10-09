"""Двухуровневая квартира: лестница, проём в плите, два уровня в датасете с классом VOID."""
import cv2
import numpy as np
import pytest

from simscan.config import load_config
from simscan.duplex import subtract_rect
from simscan.generate import generate_scene
from simscan.netinput import VOID, SceneGT, build_scene


def test_subtract_rect():
    parts = subtract_rect((0, 0, 4, 3), (1, 1, 2, 2))
    assert len(parts) == 4
    assert sum((c - a) * (d - b) for a, b, c, d in parts) == pytest.approx(12 - 1)
    assert subtract_rect((0, 0, 1, 1), (2, 2, 3, 3)) == [(0, 0, 1, 1)]


@pytest.fixture(scope="module")
def duplex(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dup")
    cfg = load_config(overrides={"layout": {"p_duplex": 1.0}, "scanner": {"angular_step_deg": 0.4},
                                 "export": {"preview": False, "free_space_input": False, "unet_mask": False}})
    for seed in range(3, 9):                         # лестница встаёт не в каждой квартире
        r = generate_scene(cfg, tmp / "scene_00000", seed=seed, index=0)
        if r.get("levels") == 2:
            break
    assert r["levels"] == 2
    dirs = build_scene(tmp / "scene_00000", tmp / "ds", log=lambda *a: None)
    return tmp / "scene_00000", tmp / "ds", dirs


def test_two_levels_and_stations(duplex):
    scene, _, dirs = duplex
    g0, g1 = SceneGT(scene, 0), SceneGT(scene, 1)
    st = g0.doc["meta"]["duplex"]
    assert g1.floor_z - g0.floor_z == pytest.approx(st["dz"])
    assert len(g0.stations()) and len(g1.stations())
    assert not any(o.kind == "door" and g1.layout.wall(o.wall_id).role == "party" for o in g1.layout.openings)
    assert [d.split("/")[-1] for d in dirs] == ["L1", "L2"]


def test_void_matches_hole_in_floor(duplex):
    _, ds, _ = duplex
    for lvl in ("L1", "L2"):
        sem = cv2.imread(str(ds / "scenes" / "scene_00000" / lvl / "sem.png"), cv2.IMREAD_UNCHANGED)
        assert (sem == VOID).sum() > 500, lvl
    # на верхнем уровне в проёме пол почти не виден, а вокруг - виден
    x = np.load(ds / "scenes" / "scene_00000" / "L2" / "input.npy").astype(np.float32)
    sem = cv2.imread(str(ds / "scenes" / "scene_00000" / "L2" / "sem.png"), cv2.IMREAD_UNCHANGED)
    floor_vis = x[3]
    assert floor_vis[sem == VOID].mean() < 0.5 * floor_vis[sem == 1].mean()
