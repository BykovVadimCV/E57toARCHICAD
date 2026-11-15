"""CAL-02 regression: synthetic *measurement* fixtures, not scanner calibration."""
from __future__ import annotations

import numpy as np

from tools.calibrate_e57_surfaces import (
    FORMAT, PointReservoir, analyze_sample, assess_wall_patches,
    fit_vertical_plane, report_md,
)


def wall_fixture(n=8500, sigma=0.002, seed=42):
    r = np.random.default_rng(seed)
    y = r.uniform(0, 4.5, n)
    z = r.uniform(0.3, 2.5, n)
    x = r.normal(0, sigma, n)
    return np.c_[x, y, z]


def test_plane_residual_reasonable_but_not_parameter_named_sigma():
    p = wall_fixture(6500, sigma=.002, seed=6)
    result = fit_vertical_plane(p)
    assert result is not None
    assert 1.5 < result['observed_plane_mad_mm'] < 2.5
    assert 1.5 < result['observed_plane_rms_mm'] < 2.5
    assert result['normal_z_abs'] < .08
    assert 'range_sigma_mm' not in result


def test_robust_surface_fitting_rejects_material_outliers():
    r = np.random.default_rng(5)
    p = wall_fixture(4500, sigma=.001, seed=3)
    outlier = np.c_[r.uniform(-0.4, 0.4, 800), r.uniform(0, 4.5, 800), r.uniform(.3, 2.5, 800)]
    result = fit_vertical_plane(np.concatenate([p, outlier]))
    assert result is not None
    assert result['inlier_fraction'] < .99
    assert result['observed_plane_mad_mm'] < 3.0


def test_horizontal_floor_is_not_wall():
    r = np.random.default_rng(4)
    p = np.c_[r.uniform(-2, 2, 3500), r.uniform(-2, 2, 3500), r.normal(0, .001, 3500)]
    assert fit_vertical_plane(p) is None
    assert assess_wall_patches(p)['summary']['n_patches'] == 0


def test_reservoir_uniform_and_bounded_and_chunk_agnostic():
    r = np.random.default_rng(1)
    pts = r.random((15000, 3))
    a, b = PointReservoir(1000, seed=7), PointReservoir(1000, seed=7)
    a.add(pts)
    for chunk in np.array_split(pts, 30):
        b.add(chunk)
    assert a.seen == b.seen == 15000
    assert a.valid == b.valid == 15000
    assert len(a.points) == len(b.points) == 1000
    assert np.allclose(np.sort(a.keys), np.sort(b.keys))
    assert np.isclose(a.sampling_fraction, 1000 / 15000)
    assert np.median(a.points[:, 0]) > .40
    assert np.median(a.points[:, 0]) < .60


def test_reservoir_filters_invalid_records():
    p = np.array([[1., 2., 3.], [np.nan, 0, 1], [8, 9, 10]])
    a = PointReservoir(1000)
    a.add(p, valid=np.array([True, True, False]))
    assert a.seen == 3 and a.valid == 1 and a.points.shape == (1, 3)


def test_wall_patches_detected_without_classification_gt():
    r = np.random.default_rng(7)
    pts = wall_fixture(11000, sigma=.002, seed=9)
    floor = np.c_[r.uniform(-1, 3, 1200), r.uniform(0, 4.5, 1200), r.normal(0, .002, 1200)]
    out = assess_wall_patches(np.r_[pts, floor], seed=16)
    assert out['candidate_wall_lines'] >= 1
    assert out['summary']['n_patches'] >= 1
    assert out['summary']['median_observed_plane_mad_mm'] < 5
    assert 'нет данных' not in report_md([analyze_sample(np.r_[pts, floor], seed=16)])
    assert 'range_sigma_mm' not in str(out['summary'])


def test_no_wall_evidence_does_not_fabricate_sigma():
    r = np.random.default_rng(11)
    xyz = r.random((600, 3))
    doc = analyze_sample(xyz)
    assert doc['format'] == FORMAT
    assert doc['surfaces']['summary']['n_patches'] == 0
    assert 'DO_NOT_FIT_SCANNER_NOISE' in doc['calibration_decision']
    assert 'нет данных' in report_md([doc])


def test_e57_scan_header_does_not_assume_physical_station():
    from types import SimpleNamespace
    from tools.calibrate_e57_surfaces import _scan_header
    s = SimpleNamespace(index=0, points=2500000, field_names=['cartesianX','cartesianY','cartesianZ'],
                        rotation=None, translation=None, index_bounds={})
    r = _scan_header(s)
    assert r['row_and_column_indices_available'] is False
    assert r['one_data3d_node_is_one_physical_station'] == 'unverified'
    assert not r['pose_rotation_available']
