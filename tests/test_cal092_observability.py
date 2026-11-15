"""CAL-09.2 known-ground-truth direct-ray and paired-patch invariants."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.cal092_ray_observability import (
    ABLATIONS, barycentric_preimage, confusion, evaluate_rays,
    physical_keys, priority_sample,
)
from tools.validate_cal092_observability import (
    SEEDS, DEFAULT_VARIANTS, RESOLUTIONS_M, summarize,
)


def _mesh():
    vertices = np.array([[0, 0, 0], [3, 0, 0], [3, 0, 3]], float)
    return {
        'vertices_pre': vertices,
        'vertices_post': vertices + np.array([0, 1, 0]),
        'triangles': np.array([[0, 1, 2]], int),
        'tri_instance': np.array([7], int),
        'tri_label': np.array([3], np.uint8),
        'realism': {'roughness_tail_probability': 0.0},
    }


def test_barycentric_maps_deformed_triangle_back_exactly():
    mesh = _mesh()
    xyz = np.array([[2.4, 1., 1.2], [2.7, 1., 2.1]])
    p, w, n, valid = barycentric_preimage(xyz, np.zeros(len(xyz), int), mesh)
    np.testing.assert_allclose(p, xyz - np.array([0, 1, 0]), atol=1e-12)
    assert valid.tolist() == [True, True]
    np.testing.assert_allclose(w.sum(axis=1), 1.)
    assert np.all(np.abs(n[:, 2]) < .2)


def test_bad_barycentric_provenance_does_not_become_gt():
    mesh = _mesh()
    _, _, _, valid = barycentric_preimage(np.array([[5., 1., 5.]]), np.array([0]), mesh)
    assert not valid[0]


def test_physical_patch_separates_wall_faces():
    pts = np.array([[.2, 0, .3], [.2, 0, .3], [.2, 0, .3]])
    normals = np.array([[0, 1., 0], [0, -1., 0], [0, 1., 0]])
    ids = np.array([1, 1, 2])
    keys, _ = physical_keys(pts, normals, ids)
    assert len({tuple(row) for row in keys}) == 3


def test_patch_confusion_handles_missing_positive_denominators():
    rows = [
        {'gt_visible_fraction': 1., 'mad_mm': 7.},
        {'gt_visible_fraction': 1., 'mad_mm': 3.},
        {'gt_visible_fraction': 0., 'mad_mm': 7.},
        {'gt_visible_fraction': 0., 'mad_mm': 2.},
    ]
    d = confusion(rows)
    assert (d['tp'], d['fp'], d['tn'], d['fn']) == (1, 1, 1, 1)
    assert d['precision'] == .5 and d['recall'] == .5
    assert confusion(rows[2:])['recall'] is None
    assert confusion(rows[:2])['false_positive_rate'] is None


def test_priority_sample_only_direct_wall_returns_and_is_deterministic():
    a = SimpleNamespace(
        straight_valid=np.array([True, True, False, True, True]),
        label=np.array([3, 2, 3, 3, 3]),
        primitive_id=np.array([11, 12, -1, -1, 15]),
        true_range_m=np.array([1., 2., np.nan, np.nan, 2.5]),
        xyz_local=np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0],
                            [4, 0, 0], [2.51, 0, 0.]], float),
    )
    one = priority_sample([a], max_sample=1000)
    two = priority_sample([a], max_sample=1000)
    assert one['total_direct_wall_returns'] == 2
    np.testing.assert_array_equal(one['primitive'], [11, 15])
    np.testing.assert_array_equal(one['ideal_local'][:, 0], [1., 2.5])
    np.testing.assert_equal(one['measured_local'], two['measured_local'])
    with pytest.raises(ValueError):
        priority_sample([SimpleNamespace(primitive_id=None, true_range_m=None)],
                        max_sample=1000)


def test_2x2_counterfactual_patches_share_identical_members():
    from simscan.realism import sample_params
    from simscan.config import RealismConfig

    mesh = _mesh()
    mesh['realism'] = sample_params(RealismConfig(wall_roughness_tail_probability=0.),
                                   np.random.default_rng(42))
    rng = np.random.default_rng(44)
    n = 12000
    # Uniformly sample the interior of the 3x3 triangular wall.
    u, v = rng.random(n), rng.random(n)
    reflect = (u + v) > 1
    u[reflect], v[reflect] = 1-u[reflect], 1-v[reflect]
    pos = mesh['vertices_post'][0] + u[:, None] * (
        mesh['vertices_post'][1] - mesh['vertices_post'][0]) + v[:, None] * (
        mesh['vertices_post'][2] - mesh['vertices_post'][0])
    measured = pos.copy()
    measured[:, 1] += rng.normal(0, .003, n)
    sample = {'station': np.zeros(n, np.int32),
              'primitive': np.zeros(n, np.int32),
              'measured_local': measured,
              'ideal_local': pos}
    pose = {'R': np.eye(3).tolist(), 't': [0, 0, 0]}
    written = {'R': np.eye(3).tolist(), 't': [0, .001, 0]}
    layout = {'meta': {'layout_to_world': {'yaw_deg':0.,'offset_m':[0,0,0]}},
              'stations': [{'pose_true':pose,'pose_written':written}]}
    result = evaluate_rays(sample, mesh, layout)
    assert result['n_geometrically_valid_rays'] == n
    assert result['n_physical_wall_patches'] > 0
    assert set(result['modes']) == set(ABLATIONS)
    counts = [v['n_patches'] for v in result['modes'].values()]
    assert len(set(counts)) == 1
    assert result['modes']['ideal']['median_patch_mad_mm'] == 0
    assert result['modes']['range_only']['median_patch_mad_mm'] > 0
    # One uniform translation of one station affects fitted offset, not scatter.
    assert result['modes']['registration_only']['median_patch_mad_mm'] == 0


def test_protocol_gates_reject_incomplete_design():
    row = {
        'seed': SEEDS[0], 'variant': DEFAULT_VARIANTS[0],
        'tessellation_m': RESOLUTIONS_M[0],
        'layout_station_signature': 'same', 'stations': 3, 'rooms': 3,
        'direct_wall_returns_sampled': 1000,
        'metrics': {'n_geometrically_valid_rays': 700,
                    'n_physical_wall_patches': 1,
                    'modes': {name: {
                        'n_gt_positive':0,'n_patches':1,
                        'recall':None,'false_positive_rate':0.,
                        'median_patch_mad_mm':1.5,
                    } for name in ABLATIONS},
                    'paired_ablation_shifts': {
                        'range_only_minus_ideal_median_mad_mm':0.,
                        'registration_only_minus_ideal_median_mad_mm':0.}}
    }
    report = summarize([row], seeds=(SEEDS[0],), variants=(DEFAULT_VARIANTS[0],),
                       resolutions=(RESOLUTIONS_M[0],), angular_step_deg=.5)
    assert not report['checks']['complete_canonical_design']
    assert not report['observability_supported_on_oracle_patches']
    with pytest.raises(ValueError):
        summarize([row, row], seeds=(SEEDS[0],),
                  variants=(DEFAULT_VARIANTS[0],), resolutions=(RESOLUTIONS_M[0],),
                  angular_step_deg=.5)
