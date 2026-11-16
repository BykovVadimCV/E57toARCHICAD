"""CAL-09.6 blind XYZ wall segmentation and topology invariants."""
import inspect

import numpy as np
import pytest

from tools.cal096_blind_geometry import (
    DEFAULTS, extract_wall_geometry, infer_topology, read_e57_xyz, voxel_sample,
)


def _face(axis, value, start, end, z0=.08, z1=2.68, step=.035,
          gap=None, seed=7):
    rng = np.random.default_rng(seed)
    along = np.arange(start, end + step * .25, step)
    height = np.arange(z0, z1 + step * .25, step)
    aa, zz = np.meshgrid(along, height)
    aa, zz = aa.ravel(), zz.ravel()
    if gap is not None:
        keep = (aa < gap[0]) | (aa > gap[1])
        aa, zz = aa[keep], zz[keep]
    jitter = rng.normal(0, .0015, (len(aa), 3))
    if axis == "y":
        pts = np.column_stack((aa, np.full(len(aa), value), zz))
        pts += jitter
    else:
        pts = np.column_stack((np.full(len(aa), value), aa, zz))
        pts += jitter
    return pts


def _room_cloud():
    parts = [
        _face("y", 0., 0., 5., seed=1),
        _face("y", 5., 0., 5., seed=2),
        _face("x", 0., 0., 5., seed=3),
        _face("x", 5., 0., 5., seed=4),
        _face("x", 2.5, 0., 2.5, seed=5),
        # A tall-ish cabinet, but it does not span the wall height.
        _face("y", 1.1, 1., 2.0, z0=.05, z1=1.25, seed=6),
    ]
    rng = np.random.default_rng(9)
    clutter = np.column_stack((
        rng.uniform(-.2, 5.2, 1500),
        rng.uniform(-.2, 5.2, 1500),
        rng.uniform(0., 2.8, 1500),
    ))
    return np.vstack(parts + [clutter])


def test_voxel_sample_is_finite_bounded_and_returns_source_indices():
    xyz = np.array([[0., 0., 0.], [.001, .001, .001],
                    [.1, 0., 0.], [np.nan, 0., 0.]])
    sampled, source = voxel_sample(xyz, voxel_m=.02, max_points=10)
    assert len(sampled) == 2
    assert set(source.tolist()) <= {0, 2}
    np.testing.assert_allclose(sampled, xyz[source])
    with pytest.raises(ValueError):
        voxel_sample(np.zeros((3, 2)))


def test_blind_detector_api_accepts_only_xyz_and_has_no_gt_dependency():
    params = inspect.signature(extract_wall_geometry).parameters
    assert list(params) == ["xyz", "config", "seed"]
    with pytest.raises(TypeError):
        extract_wall_geometry(np.zeros((10, 3)), labels=np.zeros(10))


def test_extracts_vertical_wall_planes_and_rejects_short_clutter():
    result = extract_wall_geometry(_room_cloud(), seed=11)
    candidates = [s for s in result["segments"] if s["wall_candidate"]]
    assert result["uses_ground_truth"] is False
    assert result["detector_inputs"] == ["cartesianX", "cartesianY", "cartesianZ"]
    assert len(candidates) >= 5, result["segments"]
    assert all(s["height_m"] >= DEFAULTS["min_wall_height_m"] for s in candidates)
    assert all(s["length_m"] >= DEFAULTS["min_wall_length_m"] for s in candidates)
    assert not any(abs(s["z_max_m"] - s["z_min_m"]) < 1.8 for s in candidates)


def test_door_width_gap_splits_collinear_wall_segments_without_false_bridge():
    wall = _face("y", 0., 0., 5., gap=(2., 3.), seed=21)
    result = extract_wall_geometry(wall, seed=22)
    candidates = [s for s in result["segments"] if s["wall_candidate"]]
    assert len(candidates) == 2, [(s["s_min_m"], s["s_max_m"]) for s in candidates]
    assert not any(e["type"] == "collinear_gap" for e in result["topology"]["edges"])
    assert all(s["height_m"] > 2.4 for s in candidates)


def test_topology_detects_corner_and_tee_from_plane_geometry():
    # Horizontal wall ends at x=5; vertical branch ends on its interior.
    cloud = np.vstack([
        _face("y", 0., 0., 5., seed=31),
        _face("x", 0., 0., 4., seed=32),
        _face("x", 2.5, 0., 2.5, seed=33),
    ])
    result = extract_wall_geometry(cloud, seed=34)
    kinds = {e["type"] for e in result["topology"]["edges"]}
    assert "corner" in kinds, (result["topology"], [(s["id"], s["normal_xy"], s["plane_bias"], s["s_min_m"], s["s_max_m"]) for s in result["segments"]])
    assert "tee" in kinds, result["topology"]


def test_ransac_miss_retries_before_abandoning_remaining_xyz(monkeypatch):
    import tools.cal096_blind_geometry as blind

    cloud = _face("y", 0., 0., 5., seed=71)
    original = blind._ransac_plane
    calls = {"count": 0}

    def intermittent(points, active_ids, rng, cfg):
        calls["count"] += 1
        if calls["count"] == 1:
            return None
        return original(points, active_ids, rng, cfg)

    monkeypatch.setattr(blind, "_ransac_plane", intermittent)
    result = blind.extract_wall_geometry(cloud, seed=72)
    assert calls["count"] >= 2
    assert any(s["wall_candidate"] for s in result["segments"])


def test_ransac_stops_after_frozen_number_of_empty_rounds(monkeypatch):
    import tools.cal096_blind_geometry as blind

    calls = {"count": 0}

    def always_miss(points, active_ids, rng, cfg):
        calls["count"] += 1
        return None

    monkeypatch.setattr(blind, "_ransac_plane", always_miss)
    result = blind.extract_wall_geometry(
        _face("y", 0., 0., 5., seed=73),
        config={"max_failed_plane_rounds": 2}, seed=74)
    assert calls["count"] == 2
    assert result["segments"] == []


def test_parallel_offset_faces_are_hypotheses_not_adjacency_edges():
    cloud = np.vstack([
        _face("y", 0., 0., 4., seed=41),
        _face("y", .12, 0., 4., seed=42),
    ])
    result = extract_wall_geometry(cloud, seed=43)
    candidates = [s for s in result["segments"] if s["wall_candidate"]]
    assert len(candidates) == 2
    assert result["topology"]["parallel_face_candidates"], [(s["id"], s["normal_xy"], s["plane_bias"], s["s_min_m"], s["s_max_m"]) for s in candidates]
    assert not result["topology"]["edges"]


def test_translation_and_yaw_preserve_segment_and_topology_counts():
    xyz = _room_cloud()
    base = extract_wall_geometry(xyz, seed=51)
    angle = np.radians(23.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    moved = xyz.copy()
    moved[:, :2] = moved[:, :2] @ rotation.T + np.array([13., -7.])
    moved[:, 2] += 2.4
    other = extract_wall_geometry(moved, seed=51)
    count = lambda r: sum(s["wall_candidate"] for s in r["segments"])
    assert count(other) == count(base)
    assert len(other["topology"]["edges"]) == len(base["topology"]["edges"])


def test_e57_loader_applies_scan_pose_and_returns_only_xyz(tmp_path):
    pytest.importorskip("pye57")
    from simscan.e57io import write_merged_e57

    rng = np.random.default_rng(61)
    xyz = rng.uniform(-2., 3., (300, 3))
    path = tmp_path / "blind_xyz.e57"
    write_merged_e57(path, xyz, rng.uniform(0., 1., len(xyz)).astype(np.float32))
    loaded, info = read_e57_xyz(path, max_points=1000, chunk_size=73)
    assert info["scans"] == 1
    assert info["valid_points_seen"] == len(xyz)
    assert info["sampled_points"] == len(xyz)
    assert info["uses_station_ids_in_detector"] is False
    np.testing.assert_allclose(np.sort(loaded, axis=0), np.sort(xyz, axis=0), atol=2e-5)
