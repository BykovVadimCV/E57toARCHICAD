"""CAL-09.5: synthetic geometry/soft scoring/held-out-seed invariants."""
from copy import deepcopy
import numpy as np
import pytest

from tools.cal095_soft_confidence import (
    SEEDS, DEV, HOLDOUT, VARIANTS, MESH_M, FEATURE_SETS,
    fit_logistic, confidence, features, crossfit_dev,
    select_on_dev, evaluate_protocol,
)


def cell(col, score, truth, *, wall=1, row=0):
    return {
        "cell_key": [wall, 0, 20, col, row],
        "sampled_direct_rays": 250,
        "gt_visible_fraction": float(truth),
        "scores_mm": {
            mode: {"subcell_shape": score, "legacy_mad": max(0., score * .48),
                   "macro_offset": max(0., score * .7)}
            for mode in ("ideal", "range_only", "registration_only", "combined")
        },
    }


def synthetic_cases():
    scenes = []
    for seed in SEEDS:
        for variant in VARIANTS:
            cells = []
            for i in range(15):
                gt = variant != "uniform_2_4" and i in (1, 2, 3, 4)
                value = (7.0 if gt else (5.0 if i == 11 else 1.0))
                cells.append(cell(i, value, gt))
            scenes.append({
                "seed": seed, "variant": variant, "tessellation_m": MESH_M,
                "angular_step_deg": .1, "layout_station_signature": f"layout-{seed}",
                "patch_scores": {
                    "format": "cal093-multiscale-observed-scores-v1",
                    "same_rays_each_ablation": True,
                    "cells": cells,
                },
            })
    return scenes


def test_no_gt_used_by_feature_extractor_or_order():
    cells = [cell(2, 4.1, True), cell(3, 3.3, False), cell(5, 2.0, False)]
    for set_name in FEATURE_SETS:
        original = features(cells, set_name)
        toggled = deepcopy(cells)
        for row in toggled:
            row["gt_visible_fraction"] = 1 - row["gt_visible_fraction"]
        np.testing.assert_allclose(features(toggled, set_name), original)
        reversed_out = features(cells[::-1], set_name)
        np.testing.assert_allclose(reversed_out[::-1], original)


def test_neighbor_context_does_not_bridge_wall_or_diagonal():
    center = cell(0, 2, False)
    side = cell(1, 10, True)
    other_wall = cell(0, 50, True, wall=9)
    diagonal = cell(1, 50, True, row=1)
    baseline = features([center], "shape_context")[0]
    adjacent = features([center, side], "shape_context")[0]
    separated = features([center, other_wall, diagonal], "shape_context")[0]
    assert adjacent[-2] > baseline[-2]
    np.testing.assert_allclose(separated, baseline)


def test_missing_shape_has_explicit_indicator_not_gt_label():
    p = cell(0, 0., False)
    q = cell(1, 0., True)
    q["scores_mm"]["combined"]["subcell_shape"] = None
    for spec in FEATURE_SETS:
        x = features([p, q], spec)
        assert x[0, 3] == 0.
        assert x[1, 3] == 1.
        assert np.isfinite(x).all()
    with pytest.raises(ValueError):
        features([p, deepcopy(p)], "local")


def test_logistic_training_is_deterministic_and_bounded():
    rng = np.random.default_rng(993)
    X = rng.normal(size=(200, 3))
    y = (X[:, 0] + .4 * X[:, 1] > .3).astype(float)
    a = fit_logistic(X, y, l2=.01)
    b = fit_logistic(X, y, l2=.01)
    pa, pb = confidence(a, X), confidence(b, X)
    np.testing.assert_allclose(pa, pb, atol=1e-12)
    assert np.all((pa > 0.) & (pa < 1.))
    assert pa[y > .5].mean() > pa[y < .5].mean()
    with pytest.raises(ValueError):
        fit_logistic(X, np.zeros(200), l2=.01)


def test_group_crossfit_leaves_entire_seed_out():
    scenes = synthetic_cases()
    dev = [x for x in scenes if x["seed"] in DEV]
    folds = crossfit_dev(dev, "shape_context", .01)
    assert set(folds) == set(DEV)
    assert all(len(folds[s]) == 3 for s in DEV)
    assert all(x["seed"] == s for s in DEV for x, probs in folds[s])


def test_untouched_holdout_cannot_affect_model_selection():
    scenes = synthetic_cases()
    initial = evaluate_protocol(scenes)
    assert initial["selected_on_development_only"] is not None
    changed = deepcopy(scenes)
    for case in changed:
        if case["seed"] in HOLDOUT:
            for row in case["patch_scores"]["cells"]:
                row["gt_visible_fraction"] = 1. - row["gt_visible_fraction"]
                row["scores_mm"]["combined"]["subcell_shape"] = .02
                row["scores_mm"]["combined"]["legacy_mad"] = .02
                row["scores_mm"]["combined"]["macro_offset"] = .02
    after = evaluate_protocol(changed)
    assert after["selected_on_development_only"] == initial["selected_on_development_only"]
    assert after["development_crossfit_selection"] == initial["development_crossfit_selection"]
    assert after["holdout_predeclared_comparators"] != initial["holdout_predeclared_comparators"]


def test_protocol_disallows_missing_scans_and_nonpaired_station_metadata():
    scenes = synthetic_cases()
    with pytest.raises(ValueError, match="Missing/duplicate"):
        evaluate_protocol(scenes[:-1])
    changed = deepcopy(scenes)
    changed[0]["layout_station_signature"] = "different"
    with pytest.raises(ValueError, match="layout"):
        evaluate_protocol(changed)
    changed = deepcopy(scenes)
    changed[0]["angular_step_deg"] = .5
    with pytest.raises(ValueError, match="scanner"):
        evaluate_protocol(changed)
