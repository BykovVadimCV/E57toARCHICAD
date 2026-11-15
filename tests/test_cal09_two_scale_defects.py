"""CAL-09: coarse defect zones and fine roughness are independently controlled."""
import copy

import numpy as np
import pytest

from simscan.config import RealismConfig
from simscan.realism import (
    sample_params, roughness_tail_envelope, displacement, fine_defect_field,
)


def params():
    cfg = RealismConfig(
        wall_unevenness_mm=(2.0, 4.0),
        wall_roughness_tail_probability=0.2,
        wall_roughness_tail_sigma_mm=(8.0, 12.0),
        wall_roughness_tail_corr_m=1.8,
        wall_roughness_tail_fine_corr_m=0.35,
    )
    return cfg, sample_params(cfg, np.random.default_rng(46001))


def points():
    x, z = np.meshgrid(np.linspace(0, 9, 121), np.linspace(0, 3, 41))
    return np.column_stack((x.ravel(), np.zeros(x.size), z.ravel()))


def test_fine_scale_never_modifies_zone_mask_or_layout_random_stream():
    base, p = params()
    modified = copy.deepcopy(base)
    modified.wall_roughness_tail_fine_corr_m = 0.65
    rng_a, rng_b = np.random.default_rng(123), np.random.default_rng(123)
    a, b = sample_params(base, rng_a), sample_params(modified, rng_b)
    assert rng_a.random() == rng_b.random()
    assert a["w"] == b["w"] and a["phi"] == b["phi"]
    pts = points()
    np.testing.assert_array_equal(roughness_tail_envelope(pts, a),
                                  roughness_tail_envelope(pts, b))
    np.testing.assert_array_equal(fine_defect_field(pts, a),
                                  fine_defect_field(pts, a))
    assert not np.array_equal(fine_defect_field(pts, a), fine_defect_field(pts, b))


def test_coarse_scale_changes_zone_mask_but_not_fine_texture():
    _, a = params()
    b = copy.deepcopy(a)
    b["roughness_tail_corr_m"] = 3.0
    pts = points()
    np.testing.assert_array_equal(fine_defect_field(pts, a),
                                  fine_defect_field(pts, b))
    assert np.any(roughness_tail_envelope(pts, a) != roughness_tail_envelope(pts, b))


def test_outside_zone_stays_identical_to_global_baseline():
    _, a = params()
    off = copy.deepcopy(a)
    off["roughness_tail_probability"] = 0.
    pts = points()
    mask = roughness_tail_envelope(pts, a) == 0
    assert mask.any(), "fixture must contain uncontaminated surface"
    np.testing.assert_array_equal(displacement(pts[mask], a),
                                  displacement(pts[mask], off))


def test_zero_fine_scale_keeps_cal08_legacy_exactly():
    cfg, _ = params()
    cfg.wall_roughness_tail_fine_corr_m = 0.
    a = sample_params(cfg, np.random.default_rng(46002))
    legacy = copy.deepcopy(a)
    legacy.pop("roughness_tail_fine_corr_m")
    np.testing.assert_array_equal(displacement(points(), a),
                                  displacement(points(), legacy))


def test_zero_tail_never_adds_fine_noise():
    cfg, _ = params()
    cfg.wall_roughness_tail_probability = 0.
    a = sample_params(cfg, np.random.default_rng(46003))
    b = copy.deepcopy(a)
    b["roughness_tail_fine_corr_m"] = 0.
    np.testing.assert_array_equal(displacement(points(), a),
                                  displacement(points(), b))


@pytest.mark.parametrize("corr", [-1.0, float("nan"), float("inf")])
def test_reject_invalid_fine_scale(corr):
    cfg, _ = params()
    cfg.wall_roughness_tail_fine_corr_m = corr
    with pytest.raises(ValueError):
        sample_params(cfg, np.random.default_rng(46004))
