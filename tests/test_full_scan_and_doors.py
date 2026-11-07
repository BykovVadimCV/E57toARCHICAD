"""Full-apartment acquisition and closed-door invariants."""
import numpy as np
import pytest

from simscan.config import InteriorConfig, ScannerConfig
from simscan.coverage import verify_room_scans
from simscan.interior import Furnisher, RoomGrid
from simscan.layout import Layout, Opening, Room, Wall
from simscan.scanner import place_stations
from simscan.scene import SceneBuilder, SceneMesh


def two_rooms():
    rooms = [Room(0, (0., 0., 3., 3.), (0.2, 0.2, 2.8, 2.8)),
             Room(1, (3., 0., 6., 3.), (3.2, 0.2, 5.8, 2.8))]
    wall = Wall(0, (3., 0.), (3., 3.), 0.18, "interior")
    door = Opening(0, 0, "door", 1.5, 0.9, 0., 2.1, rooms=(0, 1),
                   hinge=1, swing=1)
    return Layout(2.7, rooms, [wall], [door])


def test_every_room_is_scanned_even_when_skip_probability_is_one():
    layout = two_rooms()
    grids = {room.id: RoomGrid(room) for room in layout.rooms}
    cfg = ScannerConfig(require_all_rooms_scanned=True, p_room_unscanned=1.0,
                        unscanned_kinds=("room",))
    stations = place_stations(layout, grids, cfg, np.random.default_rng(7))
    assert {s.room_id for s in stations} == {0, 1}
    assert all(room.scanned for room in layout.rooms)


def test_full_coverage_rejects_room_with_no_safe_station():
    layout = two_rooms()
    grids = {room.id: RoomGrid(room) for room in layout.rooms}
    grids[1].occ[:] = True
    cfg = ScannerConfig(require_all_rooms_scanned=True)
    with pytest.raises(RuntimeError, match="no safe scanner position in room 1"):
        place_stations(layout, grids, cfg, np.random.default_rng(7))


def test_coverage_is_based_on_actual_hits_per_room():
    layout = two_rooms()
    cfg = ScannerConfig(require_all_rooms_scanned=True,
                        min_valid_returns_per_room=100,
                        min_architectural_returns_per_room=30)
    good = [{"level": 0, "room_id": 0, "valid": 280, "architectural_valid": 180},
            {"level": 0, "room_id": 1, "valid": 220, "architectural_valid": 90}]
    report = verify_room_scans([layout], good, cfg)
    assert report["verified_rooms"] == report["total_rooms"] == 2
    with pytest.raises(RuntimeError, match=r"L1/R1.*architectural returns"):
        verify_room_scans([layout], [good[0], dict(good[1], architectural_valid=10)], cfg)
    with pytest.raises(RuntimeError, match=r"L1/R1.*0 stations"):
        verify_room_scans([layout], [good[0]], cfg)


def test_closed_interior_door_guaranteed_when_random_rate_zero():
    layout = two_rooms()
    cfg = InteriorConfig(p_closed_door=0.0, min_closed_interior_doors=1)
    Furnisher(cfg, np.random.default_rng(17))._set_doors(layout)
    door = layout.openings[0]
    assert door.angle_deg == 0.0
    assert layout.meta["closed_doors"] == 1
    assert layout.meta["closed_interior_doors"] == 1
    solids = SceneBuilder(layout).build()
    assert any(s.label == "door" and s.source == "opening:0:leaf" for s in solids)


def test_closed_leaf_physically_blocks_laser_while_open_does_not():
    o3d = pytest.importorskip("open3d")
    layout = two_rooms()
    door = layout.openings[0]
    door.angle_deg = 0.0
    builder = SceneBuilder(layout)
    builder._door_leaf(layout.walls[0], door, 0.7)
    closed_mesh = SceneMesh.from_solids(builder.solids)
    assert closed_mesh.tri_label.size > 0
    scene = closed_mesh.to_raycasting_scene()
    # Aim through the doorway at mid-height, from room 0 towards room 1.
    rays = o3d.core.Tensor(np.array([[2.2, 1.5, 1.0, 1., 0., 0.]], np.float32))
    t_closed = scene.cast_rays(rays)["t_hit"].numpy()[0]
    assert np.isfinite(t_closed), "Closed door must block the ray"
    door.angle_deg = 90.0
    builder = SceneBuilder(layout)
    builder._door_leaf(layout.walls[0], door, 0.7)
    opened_scene = SceneMesh.from_solids(builder.solids).to_raycasting_scene()
    t_opened = opened_scene.cast_rays(rays)["t_hit"].numpy()[0]
    assert not np.isfinite(t_opened), "Open door must allow the ray through"
