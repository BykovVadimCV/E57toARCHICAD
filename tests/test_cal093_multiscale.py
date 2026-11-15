"""CAL-09.3 known-ground-truth feature and holdout-selection protocol fixtures."""
from copy import deepcopy

import numpy as np
import pytest

from tools.cal093_multiscale_features import (
    robust_vertical_line, subcell_shape_mm,
)
from tools.cal093_threshold_selection import (
    DEV_SEEDS, HOLDOUT_SEEDS, RESOLUTIONS_M, VARIANTS,
    CANDIDATE_BY_NAME, evaluate_cells, analyze_holdout,
)


def test_observed_robust_face_line_rejects_few_outliers():
    rng = np.random.default_rng(83)
    along = rng.uniform(0, 4, 1200)
    y = rng.normal(0, .0005, 1200)
    xyz = np.column_stack((along, y, rng.uniform(0, 3, 1200)))
    xyz[:25, 1] += .09
    line = robust_vertical_line(xyz)
    assert line is not None
    n, b = line
    assert abs(n[1]) > .999
    assert abs(float(np.median(xyz[25:, :2] @ n + b))) < .0003


def test_subcell_feature_detects_nonplanar_patch_without_truth_labels():
    rng = np.random.default_rng(1909)
    s = rng.uniform(0, .9, 6000)
    z = rng.uniform(0, 1.25, 6000)
    y = .008 * np.sin(2 * np.pi * s / .9) * np.sin(2 * np.pi * z / 1.25)
    curved = np.column_stack((s, y, z))
    flat = np.column_stack((s, np.zeros(len(s)), z))
    bent = subcell_shape_mm(curved, s, z, robust_vertical_line(curved))
    zero = subcell_shape_mm(flat, s, z, robust_vertical_line(flat))
    assert bent is not None and bent > 1.0
    assert zero == pytest.approx(0., abs=1e-8)


def _cell(instance, col, row, gt, mad=1., macro=1., shape=.3):
    return {
        "cell_key": [instance, 0, 20, col, row],
        "gt_visible_fraction": gt,
        "scores_mm": {mode: {
            "legacy_mad":mad, "macro_offset":macro, "subcell_shape":shape
        } for mode in ("ideal","range_only","registration_only","combined")},
    }


def test_cluster_requires_two_adjacently_detected_true_cells():
    cells = [
        _cell(1, 0, 0, 1., mad=3),
        _cell(1, 1, 0, 1., mad=1),
        _cell(1, 1, 1, 1., mad=3),
        _cell(1, 4, 1, 0., mad=1),
    ]
    x = evaluate_cells(cells, CANDIDATE_BY_NAME["mad2"])
    assert x["n_gt_clusters_size_ge_2"] == 1
    assert x["cluster_recall"] == 0
    cells[1]["scores_mm"]["combined"]["legacy_mad"] = 3
    x = evaluate_cells(cells, CANDIDATE_BY_NAME["mad2"])
    assert x["cluster_recall"] == 1.


def test_missing_feature_counts_as_negative_and_not_arbitrary_zero_score():
    items = [_cell(1, 0, 0, 1., macro=None),
             _cell(1, 1, 0, 0., macro=4.)]
    result = evaluate_cells(items, CANDIDATE_BY_NAME["macro2"])
    assert (result["tp"],result["fp"],result["fn"]) == (0, 1, 1)
    assert result["fraction_missing_feature"] == .5


def fake_cases():
    cases = []
    for seed in DEV_SEEDS + HOLDOUT_SEEDS:
        for resolution in RESOLUTIONS_M:
            for variant in VARIANTS:
                base = []
                for i in range(10):
                    is_positive = variant != "uniform_2_4" and i in (0, 1, 2)
                    base.append(_cell(
                        instance=1, col=i, row=0, gt=float(is_positive),
                        mad=3.5 if is_positive else 1.,
                        macro=4.5 if is_positive else .6,
                        shape=2.5 if is_positive else .3))
                cases.append({
                    "seed":seed, "variant":variant,
                    "tessellation_m":resolution, "layout_station_signature":str(seed),
                    "patch_scores":{
                        "format":"cal093-multiscale-observed-scores-v1",
                        "same_rays_each_ablation":True,"cells":base,
                    },
                })
    return cases


def test_selection_never_reads_holdout_scores():
    cases=fake_cases()
    first=analyze_holdout(cases)
    assert first["selected_using_development_only"] == "mad2"
    assert not first["design"]["heldout_seeds_used_in_selection"]
    for case in cases:
        if case["seed"] in HOLDOUT_SEEDS:
            for cell in case["patch_scores"]["cells"]:
                for scores in cell["scores_mm"].values():
                    scores["legacy_mad"] = 0.
                    scores["macro_offset"] = 0.
                    scores["subcell_shape"] = 0.
    changed=analyze_holdout(cases)
    assert changed["selected_using_development_only"] == first["selected_using_development_only"]
    assert changed["development_threshold_sweep"] == first["development_threshold_sweep"]
    assert (changed["holdout_selected_vs_fixed_mad6"]["mad2"]["0.10"]["mean_recall"]
            < first["holdout_selected_vs_fixed_mad6"]["mad2"]["0.10"]["mean_recall"])


def test_protocol_refuses_missing_or_mismatched_cases():
    cases=fake_cases()
    with pytest.raises(ValueError, match="Missing/duplicate"):
        analyze_holdout(cases[:-1])
    other=deepcopy(cases)
    other[0]["layout_station_signature"]="broken"
    with pytest.raises(ValueError, match="Pairing"):
        analyze_holdout(other)
    other=deepcopy(cases)
    other[0]["patch_scores"]["same_rays_each_ablation"]=False
    with pytest.raises(ValueError, match="Pairing"):
        analyze_holdout(other)
