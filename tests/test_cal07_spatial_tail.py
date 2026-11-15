"""CAL-07: spatially clustered roughness tail invariants and quality gates."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from simscan.config import config_to_dict, load_config
from simscan.realism import displacement, roughness_tail_envelope, sample_params
from tools.validate_cal07_spatial_tail import (
    NEW_SEEDS, USED_BEFORE, VARIANTS, validate_seeds, footprint_for_seed,
    fixed_base_config, variant_config, patch_metrics, assess,
)

ROOT = Path(__file__).resolve().parents[1]


def _real():
    return json.loads((ROOT / "configs/calibration_real_e57_surfaces.json").read_text())


def _patches(base=2.0, tail_value=7.0, tail_count=4, isolated=False):
    rows = []
    tail_positions = [(i * 2 if isolated else i, 0) for i in range(tail_count)]
    for i in range(24):
        is_tail = i >= 24 - tail_count
        col, row = (tail_positions[i - (24 - tail_count)] if is_tail else (i % 4, i // 4))
        rows.append({
            "observed_plane_mad_mm": tail_value if is_tail else base,
            "observed_plane_rms_mm": tail_value + .4 if is_tail else base + .4,
            "wall_candidate_index": 0, "patch_col": col, "patch_row": row,
            "inlier_points": 1000 - i, "inlier_fraction": .9,
        })
    return rows


def _records(*, median_shift=0.1, no_cluster=False, patch_count=24, return_ratio=1.0):
    rows = []
    for seed in NEW_SEEDS:
        signature = f"signature_{seed}"
        for variant in VARIANTS:
            tail = variant != "uniform_2_4"
            tail_count = 0 if variant == "uniform_2_4" else (4 if variant == "tail_10pct" else 6)
            tail_value = 4.2 if variant == "uniform_2_4" else (7.0 if variant == "tail_10pct" else 9.0)
            vals = _patches(base=2.0 + (median_shift if tail else 0.0),
                            tail_value=tail_value, tail_count=tail_count,
                            isolated=no_cluster)
            vals = vals[:patch_count]
            rows.append({
                "seed": seed, "variant": variant, "layout_signature": signature,
                "layout_station_matched": True, "rooms": 3, "stations": 4,
                "valid_returns": int(100000 * (return_ratio if tail else 1.0)),
                "n_patches": len(vals), "surface_patches": vals,
            })
    return rows


def test_seed_set_is_frozen_and_disjoint_from_previous_stages():
    assert len(NEW_SEEDS) == 8 and len(set(NEW_SEEDS)) == 8
    assert not (set(NEW_SEEDS) & USED_BEFORE)
    assert validate_seeds(NEW_SEEDS) == NEW_SEEDS
    assert [footprint_for_seed(x) for x in NEW_SEEDS[:4]] == ["original_footprint"] * 4
    assert [footprint_for_seed(x) for x in NEW_SEEDS[4:]] == ["larger_footprint"] * 4
    with pytest.raises(ValueError):
        validate_seeds(NEW_SEEDS[:-1])


def test_variant_changes_only_spatial_tail_and_keeps_base_roughness_2_4():
    base = fixed_base_config(ROOT / "configs/customer_like.yaml", NEW_SEEDS[0])
    assert base.realism.wall_unevenness_mm == (2.0, 4.0)
    snapshots = []
    for variant in VARIANTS:
        cfg, changes = variant_config(base, variant)
        assert cfg.realism.wall_unevenness_mm == (2.0, 4.0)
        assert cfg.scanner.angular_step_deg == .1
        assert cfg.realism.wall_roughness_tail_probability == {
            "uniform_2_4": 0.0, "tail_10pct": .10, "tail_20pct": .20}[variant]
        d = config_to_dict(cfg)
        for key in ("wall_roughness_tail_probability", "wall_roughness_tail_sigma_mm",
                    "wall_roughness_tail_corr_m", "wall_roughness_tail_transition_sigma"):
            d["realism"].pop(key)
        snapshots.append(d)
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_tail_parameters_do_not_advance_or_perturb_the_legacy_random_stream():
    a = load_config(overrides={"realism": {"wall_unevenness_mm": [2, 4],
        "wall_roughness_tail_probability": 0.0}})
    b = load_config(overrides={"realism": {"wall_unevenness_mm": [2, 4],
        "wall_roughness_tail_probability": 0.1}})
    ra, rb = np.random.default_rng(73), np.random.default_rng(73)
    pa, pb = sample_params(a.realism, ra), sample_params(b.realism, rb)
    assert np.array_equal(np.asarray(pa["w"]), np.asarray(pb["w"]))
    assert np.array_equal(np.asarray(pa["phi"]), np.asarray(pb["phi"]))
    assert np.array_equal(np.asarray(pa["w_lean"]), np.asarray(pb["w_lean"]))
    assert ra.random() == rb.random()


def test_tail_envelope_is_smooth_bounded_and_spatially_sparse():
    cfg = load_config(overrides={"realism": {
        "wall_unevenness_mm": [2, 4], "wall_roughness_tail_probability": .1,
        "wall_roughness_tail_sigma_mm": [8, 12], "wall_roughness_tail_corr_m": .8,
        "wall_roughness_tail_transition_sigma": .15,
    }})
    params = sample_params(cfg.realism, np.random.default_rng(5))
    x = np.linspace(0, 10, 101)
    y = np.linspace(0, 8, 81)
    z = np.linspace(0, 3, 13)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    env = roughness_tail_envelope(points, params).reshape(len(x), len(y), len(z))
    assert np.isfinite(env).all()
    assert env.min() >= 0 and env.max() <= 1
    assert .02 < np.mean(env > .95) < .20
    assert .06 < np.mean(env > 0) < .35
    x2 = np.arange(0, 10.0001, .025)
    y2 = np.arange(0, 8.0001, .025)
    xx2, yy2 = np.meshgrid(x2, y2, indexing="ij")
    p2 = np.column_stack((xx2.ravel(), yy2.ravel(), np.full(xx2.size, 1.5)))
    env2 = roughness_tail_envelope(p2, params).reshape(len(x2), len(y2))
    assert np.max(np.abs(np.diff(env2, axis=0))) < .6
    assert np.max(np.abs(np.diff(env2, axis=1))) < .6


def test_zero_tail_is_exact_base_field_and_tail_only_increases_horizontal_sigma():
    cfg0 = load_config(overrides={"realism": {"wall_unevenness_mm": [2, 4],
        "wall_roughness_tail_probability": 0.0}})
    cfg1 = load_config(overrides={"realism": {"wall_unevenness_mm": [2, 4],
        "wall_roughness_tail_probability": .1}})
    p0 = sample_params(cfg0.realism, np.random.default_rng(17))
    p1 = sample_params(cfg1.realism, np.random.default_rng(17))
    points = np.random.default_rng(9).uniform([0, 0, 0], [8, 8, 3], (5000, 3))
    d0, d1 = displacement(points, p0), displacement(points, p1)
    assert np.array_equal(d0[:, 2], d1[:, 2])
    assert np.median(np.linalg.norm(d1[:, :2], axis=1)) >= np.median(np.linalg.norm(d0[:, :2], axis=1))
    assert np.quantile(np.linalg.norm(d1[:, :2], axis=1), .9) > \
           np.quantile(np.linalg.norm(d0[:, :2], axis=1), .9)


def test_invalid_tail_parameters_are_rejected():
    cfg = load_config(overrides={"realism": {"wall_roughness_tail_probability": .7}})
    with pytest.raises(ValueError):
        sample_params(cfg.realism, np.random.default_rng(0))
    cfg = load_config(overrides={"realism": {"wall_unevenness_mm": [2, 4],
        "wall_roughness_tail_probability": .1, "wall_roughness_tail_sigma_mm": [3, 6]}})
    with pytest.raises(ValueError):
        sample_params(cfg.realism, np.random.default_rng(0))


def test_patch_metrics_capture_tail_and_local_clusters():
    p = _patches()
    m = patch_metrics(p)
    assert m["n_patches"] == 24
    assert m["tail_patch_count"] == 4
    assert m["largest_tail_cluster"] == 4
    assert m["tail_neighbor_pairs"] >= 3
    isolated = patch_metrics(_patches(isolated=True))
    assert isolated["tail_patch_count"] == 4
    assert isolated["largest_tail_cluster"] == 1


def test_cal07_prefers_lower_tail_probability_when_both_variants_pass():
    result = assess(_real(), _records(), seeds=NEW_SEEDS)
    assert result["candidate_decision"] == "PROVISIONAL_SPATIAL_TAIL_10PCT_SUPPORTED"
    assert result["selected_candidate"] == "tail_10pct"
    assert result["variants"]["tail_10pct"]["all_gates_pass"]
    assert result["variants"]["tail_20pct"]["all_gates_pass"]
    assert result["auto_update_production_config"] is False


def test_cal07_rejects_variant_that_degrades_typical_surface_median():
    result = assess(_real(), _records(median_shift=.9), seeds=NEW_SEEDS)
    assert not result["variants"]["tail_10pct"]["gates"]["typical_median_preserved_in_six_of_eight"]
    assert result["candidate_decision"] == "NO_SPATIAL_TAIL_VARIANT_PASSES"


def test_cal07_rejects_tail_without_spatial_clustering():
    result = assess(_real(), _records(no_cluster=True), seeds=NEW_SEEDS)
    assert not result["variants"]["tail_10pct"]["gates"]["clustered_tail_in_at_least_four_of_eight"]


def test_cal07_rejects_return_or_patch_collapse():
    result = assess(_real(), _records(patch_count=7), seeds=NEW_SEEDS)
    assert not result["variants"]["tail_10pct"]["gates"]["all_eight_pairs_eligible"]
    result = assess(_real(), _records(return_ratio=.5), seeds=NEW_SEEDS)
    assert not result["variants"]["tail_10pct"]["gates"]["all_eight_pairs_eligible"]


def test_cal07_rejects_duplicate_or_unmatched_records():
    rows = _records()
    with pytest.raises(ValueError):
        assess(_real(), rows[:-1])
    rows[1]["layout_signature"] = "different"
    with pytest.raises(ValueError):
        assess(_real(), rows)


def test_no_coordinates_or_physical_sigma_are_written():
    result = assess(_real(), _records())
    text = json.dumps(result)
    for forbidden in ('"xyz"', '"pose_true"', '"pose_written"', '"origins"',
                      '"intrinsic_sensor_noise_sigma_mm"'):
        assert forbidden not in text
    assert result["auto_update_production_config"] is False
