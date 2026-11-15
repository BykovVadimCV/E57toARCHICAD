"""CAL-09.4 frozen spatial filtering and true new-seed holdout invariants."""
from copy import deepcopy
import json

import pytest

from tools.cal094_spatial_selection import (
    SEED_DEV, SEED_HOLDOUT, SEEDS, VARIANTS, MESHES,
    CANDIDATE_BY_NAME, REFERENCE, predictions,
    evaluate_one, evaluate_seed, evaluate_protocol, select_on_development,
)
from tools.aggregate_cal094 import aggregate
from tools.validate_cal094_spatial import RUN_FORMAT


def cell(col, row, score, truth=0., wall=12, variant=None):
    return {
        "cell_key": [wall, 0, 20, col, row],
        "gt_visible_fraction": truth,
        "scores_mm": {name: {
            "subcell_shape": score, "legacy_mad": score * .5,
            "macro_offset": .5,
        } for name in ("ideal", "range_only", "registration_only", "combined")},
    }


def test_cc2_removes_isolated_false_positive_and_keeps_connected_defect():
    rows = [cell(0, 0, 4.3, 1.), cell(1, 0, 4.3, 1.),
            cell(7, 0, 7.5, 0.), cell(10, 0, 2.5, 0.)]
    m = evaluate_one(rows, CANDIDATE_BY_NAME["shape4_cc2"])
    assert (m["tp"], m["fp"], m["fn"]) == (2, 0, 0)
    assert m["recovered_gt_clusters_ge_2"] == 1
    control = evaluate_one(rows, REFERENCE["cal093_shape4"])
    assert control["fp"] == 1


def test_hysteresis_retains_weak_neighbours_only_with_strong_anchor():
    rows = [cell(0, 0, 5.2), cell(1, 0, 3.4),
            cell(8, 0, 3.7), cell(9, 0, 3.9)]
    result = predictions(rows, CANDIDATE_BY_NAME["hys3_5_cc2"])
    assert (12, 0, 20, 0, 0) in result
    assert (12, 0, 20, 1, 0) in result
    assert (12, 0, 20, 8, 0) not in result
    assert (12, 0, 20, 9, 0) not in result


def test_predictions_use_no_gt_labels_and_never_bridge_faces_or_walls():
    rows = [cell(0, 0, 4.5, 0., wall=1),
            cell(1, 0, 4.5, 0., wall=2),
            cell(1, 0, 4.5, 0., wall=3)]
    first = predictions(rows, CANDIDATE_BY_NAME["shape4_cc2"])
    assert first == set()
    changed = deepcopy(rows)
    for x in changed:
        x["gt_visible_fraction"] = 1.
    assert predictions(changed, CANDIDATE_BY_NAME["shape4_cc2"]) == first


def test_corner_only_contact_is_not_four_connected_and_malformed_rejected():
    rows = [cell(0, 0, 5.0), cell(1, 1, 5.0)]
    assert predictions(rows, CANDIDATE_BY_NAME["shape4_cc2"]) == set()
    with pytest.raises(ValueError):
        predictions(rows + [deepcopy(rows[0])], CANDIDATE_BY_NAME["shape4_cc2"])


def cases_fixture():
    cases = []
    for seed in SEEDS:
        for mesh in MESHES:
            for variant in VARIANTS:
                rows = []
                for col in range(12):
                    positive = variant != "uniform_2_4" and col < 4
                    score = 4.5 if positive else (5.0 if col == 8 else 1.)
                    rows.append(cell(col, 0, score, float(positive)))
                cases.append({
                    "seed": seed, "variant": variant,
                    "tessellation_m": mesh,
                    "layout_station_signature": f"layout-{seed}",
                    "patch_scores": {
                        "format": "cal093-multiscale-observed-scores-v1",
                        "same_rays_each_ablation": True,
                        "cells": rows,
                    }
                })
    return cases


def test_no_cross_variant_contamination_when_merging_confusion():
    cases = cases_fixture()
    out = evaluate_seed(cases, SEED_DEV[0], .10,
                        CANDIDATE_BY_NAME["shape4_cc2"])
    assert out["tp"] == 8 and out["fp"] == 0
    assert out["fn"] == 0
    assert out["cluster_recall"] == 1.
    assert out["by_variant"]["uniform_2_4"]["tp"] == 0


def test_selection_does_not_consult_new_holdout_even_when_scores_change():
    cases = cases_fixture()
    first = evaluate_protocol(cases)
    assert first["selected_on_development_only"] == "shape4_cc2"
    assert first["engineering_oracle_gates_pass"]
    modified = deepcopy(cases)
    for c in modified:
        if c["seed"] in SEED_HOLDOUT:
            for row in c["patch_scores"]["cells"]:
                row["scores_mm"]["combined"]["subcell_shape"] = .1
    second = evaluate_protocol(modified)
    assert (second["selected_on_development_only"] ==
            first["selected_on_development_only"])
    assert second["development_selection"] == first["development_selection"]
    assert not second["engineering_oracle_gates_pass"]


def test_protocol_rejects_missing_duplicates_and_broken_pairing():
    fixture = cases_fixture()
    with pytest.raises(ValueError, match="Missing or duplicate"):
        evaluate_protocol(fixture[:-1])
    broken = deepcopy(fixture)
    broken[1]["layout_station_signature"] = "wrong"
    with pytest.raises(ValueError, match="Different layout"):
        evaluate_protocol(broken)


def test_aggregate_exact_shards_and_rejects_config_mismatch(tmp_path):
    cases = cases_fixture()
    paths = []
    for seed in SEEDS:
        doc = {
            "format": RUN_FORMAT,
            "auto_update_production_config": False,
            "variants": list(VARIANTS),
            "resolutions_m": list(MESHES),
            "angular_step_deg": .1,
            "max_sample": 180000,
            "seeds": [seed],
            "n_cases": 6,
            "cases": [x for x in cases if x["seed"] == seed],
            "run_fingerprint": "hash-v1",
            "source_config_sha256": "same",
        }
        path = tmp_path / f"{seed}.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        paths.append(path)
    assert aggregate(paths)["n_complete_scans"] == 48
    wrong = json.loads(paths[1].read_text())
    wrong["run_fingerprint"] = "wrong"
    paths[1].write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="Missing/repeated seed"):
        aggregate(paths)
