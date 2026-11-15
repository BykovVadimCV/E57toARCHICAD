"""Synthetic known-ground-truth fixtures for CAL-09.1 welded wall mesh topology."""
import numpy as np
import pytest

from tools.cal091_mesh_truth import mesh_truth_metrics


def quads(cols=2, rows=1, different_instance=False):
    points, tris, instances, values = [], [], [], []
    for x in range(cols):
        for z in range(rows):
            start = len(points)
            points.extend([(x, 0, z), (x + 1, 0, z),
                           (x + 1, 0, z + 1), (x, 0, z + 1)])
            tris.extend([(start, start + 1, start + 2),
                         (start, start + 2, start + 3)])
            instances.extend([x + 1 if different_instance else 1] * 2)
            values.extend([1.0] * 4)
    return np.array(points, float), np.array(tris, int), np.array(instances), np.array(values)


def measure(vertices, triangles, instances, envelope, delta=None):
    if delta is None:
        delta = np.zeros((len(vertices), 2))
    return mesh_truth_metrics(vertices, triangles, instances, envelope, delta)


def test_seam_duplicates_are_welded_into_one_connected_zone():
    v, t, inst, env = quads(2)
    result = measure(v, t, inst, env)
    assert result['vertical_wall_area_m2'] == 2.0
    assert result['connected_components'] == 1
    assert result['largest_component_area_m2'] == 2.0
    assert result['fragmentation_index'] == 0.


def test_cross_wall_connections_are_forbidden():
    v, t, inst, env = quads(2, different_instance=True)
    result = measure(v, t, inst, env)
    assert result['connected_components'] == 2
    assert result['largest_component_area_m2'] == 1.0
    assert result['fragmentation_index'] == 0.5


def test_only_corner_contact_is_not_connected():
    v, t, inst, env = quads(2, 2)
    # Keep only diagonally touching quads (bottom left, top right).
    q = [0, 3]
    ids = np.array([i for qid in q for i in (2 * qid, 2 * qid + 1)])
    result = measure(v, t[ids], inst[ids], env)
    assert result['connected_components'] == 2


def test_outside_background_is_unchanged_and_shape_is_validated():
    v, t, inst, env = quads(2)
    env[4:] = 0
    d = np.zeros((len(v), 2))
    d[:4, 1] = .01
    result = measure(v, t, inst, env, d)
    assert result['defect_area_m2'] == 1.0
    assert result['outside_support_vertex_count'] == 4
    assert result['outside_support_rms_mm'] == 0.0
    assert result['inside_wall_normal_rms_mm'] == 10.0
    with pytest.raises(ValueError):
        measure(v, t, inst, np.ones(3))
    with pytest.raises(ValueError):
        measure(v, t, inst, np.full(len(v), np.nan))


def test_horizontal_faces_are_excluded():
    v, t, inst, env = quads(2)
    horizontal = np.array([(0, 0, 10), (1, 0, 10), (0, 1, 10)], float)
    vertices = np.vstack((v, horizontal))
    tris = np.vstack((t, [len(v), len(v) + 1, len(v) + 2]))
    m = measure(vertices, tris, np.append(inst, 1), np.ones(len(vertices)))
    assert m['vertical_wall_triangles'] == len(t)


def test_fixture_output_is_deterministic():
    v, t, inst, env = quads(3, 2)
    assert measure(v,t,inst,env) == measure(v,t,inst,env)
