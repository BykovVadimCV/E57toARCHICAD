"""Круговой путь: сцена -> E57 (pye57.write_scan_raw) -> pye57.read_scan -> исходная геометрия."""
import json

import numpy as np
import pye57
from pyquaternion import Quaternion

from simscan.config import load_config
from simscan.generate import generate_scene
from simscan.layout import Layout
from simscan.scene import build_scene, surface_distance
from simscan.transform import WorldTransform, quat_from_matrix, rot_x, rot_y, rot_z

from conftest import CLEAN_EFFECTS, CLEAN_SCENE


def test_quaternion_matches_pyquaternion():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = rot_z(rng.uniform(-3, 3)) @ rot_y(rng.uniform(-3, 3)) @ rot_x(rng.uniform(-3, 3))
        q = quat_from_matrix(R)
        assert np.allclose(Quaternion(q).rotation_matrix, R, atol=1e-12)


def _clean_scene(tmp_path, **effects):
    cfg = load_config(overrides={
        "scanner": {"angular_step_deg": 0.5},
        "effects": {**CLEAN_EFFECTS, **effects},
        "export": {"preview": False, "world_offset_m": [100000.0, 100000.0]},
        **CLEAN_SCENE,
    })
    info = generate_scene(cfg, tmp_path / "s", seed=7, index=0)
    doc = json.loads((tmp_path / "s" / "layout.json").read_text(encoding="utf-8"))
    return info, doc


def test_points_land_on_scene_surfaces(tmp_path):
    """Без шума и ошибки регистрации точки из E57 лежат на сетке сцены точнее 1 мм,
    даже при сдвиге СК объекта на 100 км (позы - float64, точки - float32 в СК станции)."""
    _, doc = _clean_scene(tmp_path)
    lay = Layout.from_dict(doc)
    w = doc["meta"]["layout_to_world"]
    world = WorldTransform(w["yaw_deg"], tuple(w["offset_m"]))
    _, solids = build_scene(lay)
    # surface_distance считает только параллелепипеды; точки на сетках (цилиндры, ткань) - мимо
    meshed = {s.instance for s in solids if s.mesh is not None}
    rng = np.random.default_rng(0)
    with pye57.E57(str(tmp_path / "s" / "scan.e57")) as e57:
        assert e57.scan_count == len(doc["stations"])
        for i, st in enumerate(doc["stations"]):
            d = e57.read_scan(i, transform=True, ignore_missing_fields=True)
            xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
            lab = np.load(tmp_path / "s" / "labels" / f"scan_{i:03d}.npz")
            assert len(xyz) == lab["valid"].sum()
            p = world.to_layout(xyz)[~np.isin(lab["instance"][lab["valid"]], list(meshed))]
            p = p[rng.choice(len(p), min(len(p), 20000), replace=False)]
            dist = surface_distance(p, solids)
            assert dist.max() < 1e-4, f"скан {i}: {dist.max()}"
            pos = e57.scan_position(i)[0]
            assert np.allclose(pos, st["pose_true"]["t"], atol=1e-6)


def test_invalid_rays_kept_as_directions(tmp_path):
    _, doc = _clean_scene(tmp_path)
    nrow, ncol = doc["meta"]["scanner"]["nrow"], doc["meta"]["scanner"]["ncol"]
    with pye57.E57(str(tmp_path / "s" / "scan.e57")) as e57:
        raw = e57.read_scan_raw(0)
        assert len(raw["cartesianX"]) == nrow * ncol
        assert set(np.unique(raw["cartesianInvalidState"])) <= {0, 1}
        h = e57.get_header(0)
        assert h.has_pose()


def test_registration_error_written_not_applied(tmp_path):
    """С ошибкой регистрации записанная поза отличается от истинной, первая станция - опорная."""
    _, doc = _clean_scene(tmp_path, registration_error=True,
                          registration_sigma_mm=[5.0, 5.0])
    st = doc["stations"]
    assert np.allclose(st[0]["pose_true"]["t"], st[0]["pose_written"]["t"])
    if len(st) > 1:
        dt = np.subtract(st[1]["pose_true"]["t"], st[1]["pose_written"]["t"])
        assert 0 < np.linalg.norm(dt) < 0.05
