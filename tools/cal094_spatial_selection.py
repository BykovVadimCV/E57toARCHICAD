"""CAL-09.4: predeclared spatial-coherence detector and truly fresh holdout.

Selection uses *only* development seeds 48101/48102/48105/48106.
Holdout seeds 49101/49102/49105/49106 are never used to select a
threshold, topology operator, tie break or rule. Old 461xx and 471xx
are intentionally excluded as previously inspected.

Patches are still GT-oracle-associated (CAL-09.3); predictions use
only observed subcell_shape scores and physical patch adjacency.
A component is never connected across different wall instances,
wall-face orientations, geometry variants or scene seeds.

No production calibration, GT labeling or blind wall detection.
"""
from __future__ import annotations

import numpy as np

from tools.cal092_ray_observability import confusion
from tools.cal093_threshold_selection import _components

FORMAT = "cal094-spatial-coherence-holdout-v1"
SEED_DEV = (48101, 48102, 48105, 48106)
SEED_HOLDOUT = (49101, 49102, 49105, 49106)
SEEDS = SEED_DEV + SEED_HOLDOUT
VARIANTS = ("uniform_2_4", "zone08_fine35", "zone18_fine35")
MESHES = (.30, .10)
ORACLE_GT_FRACTION = .40

# (name, weak_shape_mm, strong_shape_mm, minimum_4_connected_component_size)
# The no-topology CAL-09.3 shape4 is retained only as a fixed reference.
# All new candidates and selection rules were frozen before collecting 49xxx.
CANDIDATES = (
    ("shape4_cc2",       4., 4., 2),
    ("shape4_cc3",       4., 4., 3),
    ("shape3_cc2",       3., 3., 2),
    ("shape3_cc3",       3., 3., 3),
    ("shape3p5_cc2",     3.5, 3.5, 2),
    ("shape3p5_cc3",     3.5, 3.5, 3),
    ("shape3p5_cc4",     3.5, 3.5, 4),
    ("shape4p5_cc2",     4.5, 4.5, 2),
    ("shape4p5_cc3",     4.5, 4.5, 3),
    ("hys3_4_cc2",       3., 4., 2),
    ("hys3_5_cc2",       3., 5., 2),
    ("hys3p5_5_cc2",     3.5, 5., 2),
    ("hys3p5_6_cc2",     3.5, 6., 2),
    ("hys3_5_cc3",       3., 5., 3),
    ("hys3p5_5_cc3",     3.5, 5., 3),
    ("hys4_5_cc2",       4., 5., 2),
)
CANDIDATE_BY_NAME = {x[0]: x for x in CANDIDATES}
# Fixed comparators do not participate in selection.
REFERENCE_SHAPE4 = ("cal093_shape4", 4., 4., 1)
REFERENCE_MAD6 = ("cal09_mad6",)
REFERENCE = {REFERENCE_SHAPE4[0]: REFERENCE_SHAPE4,
             REFERENCE_MAD6[0]: REFERENCE_MAD6}
DEV_MAX_MEAN_FPR = .08
DEV_MAX_WORST_FPR = .15
HOLDOUT_MIN_RECALL = .40
HOLDOUT_MAX_MEAN_FPR = .10
HOLDOUT_MIN_CLUSTER_RECALL = .35
HOLDOUT_MIN_GAIN_OVER_MAD6 = .15


def _score(cell, name="subcell_shape"):
    score = cell["scores_mm"]["combined"].get(name)
    return float(score) if score is not None and np.isfinite(score) else None


def predictions(cells, rule):
    """Predict only from observed feature scores and 4-neighbour topology.

    A weak connected component is retained if its size is sufficient and it
    contains at least one strong-score seed. No GT fractions are consulted.
    """
    if not rule or rule[0] not in CANDIDATE_BY_NAME | REFERENCE:
        raise ValueError("Unsupported spatial candidate")
    keys = [tuple(row["cell_key"]) for row in cells]
    if any(len(k) != 5 for k in keys) or len(keys) != len(set(keys)):
        raise ValueError("Duplicate/invalid physical patch key within one scene")
    if rule[0] == REFERENCE_MAD6[0]:
        return {k for k, row in zip(keys, cells)
                if (_score(row, "legacy_mad") is not None and
                    _score(row, "legacy_mad") >= 6.)}
    name, low, high, min_size = rule
    if min_size < 1 or high < low:
        raise ValueError("Invalid frozen spatial rule")
    weak, strong = set(), set()
    for key, row in zip(keys, cells):
        value = _score(row)
        if value is not None and value >= low:
            weak.add(key)
            if value >= high:
                strong.add(key)
    retained = set()
    for component in _components(weak):
        if len(component) >= min_size and component & strong:
            retained.update(component)
    return retained


def evaluate_one(cells, rule):
    """Count scores on ONE geometry variant without cross-variant adjacency."""
    selected = predictions(cells, rule)
    truth = {tuple(row["cell_key"]) for row in cells
             if float(row["gt_visible_fraction"]) >= ORACLE_GT_FRACTION}
    observed = {tuple(row["cell_key"]) for row in cells}
    positive = sorted(truth)
    false = selected - truth
    mat = confusion([
        {"gt_visible_fraction": float(k in truth),
         "mad_mm": float(k in selected)}
        for k in observed], threshold=.5, gt_fraction=.5)
    gt_clusters = [c for c in _components(truth) if len(c) >= 2]
    recovered = sum(any(len(part) >= 2 for part in
                        _components(c & selected)) for c in gt_clusters)
    fp_components = _components(false)
    return {
        **mat,
        "gt_clusters_ge_2": len(gt_clusters),
        "recovered_gt_clusters_ge_2": recovered,
        "fp_components": len(fp_components),
        "fp_singletons": sum(len(c) == 1 for c in fp_components),
        "n_available_shape_scores": sum(_score(x) is not None for x in cells),
        "n_nonempty_predictions": len(selected),
    }


def _safe_ratio(a, b):
    return round(float(a) / b, 6) if b else None


def evaluate_seed(cases, seed, mesh, rule):
    """Pool confusion counts across three *independent* geometry variants.

    Never union wall-cell keys from distinct variant scans: they may be
    numerically identical but represent different physical surfaces.
    """
    these = [c for c in cases if c["seed"] == seed and c["tessellation_m"] == mesh]
    if len(these) != len(VARIANTS) or {x["variant"] for x in these} != set(VARIANTS):
        raise ValueError("Missing/duplicated variant in one seed and resolution")
    groups = {c["variant"]: evaluate_one(c["patch_scores"]["cells"], rule)
              for c in these}
    count_fields = ("tp", "fp", "tn", "fn", "gt_clusters_ge_2",
                    "recovered_gt_clusters_ge_2", "fp_components", "fp_singletons",
                    "n_available_shape_scores", "n_nonempty_predictions")
    combined = {k: sum(g[k] for g in groups.values()) for k in count_fields}
    combined.update({
        "seed": seed, "mesh_m": mesh,
        "recall": _safe_ratio(combined["tp"], combined["tp"] + combined["fn"]),
        "fpr": _safe_ratio(combined["fp"], combined["fp"] + combined["tn"]),
        "precision": _safe_ratio(combined["tp"], combined["tp"] + combined["fp"]),
        "cluster_recall": _safe_ratio(combined["recovered_gt_clusters_ge_2"],
                                     combined["gt_clusters_ge_2"]),
        "by_variant": {name: {
            "tp": g["tp"], "fp": g["fp"], "tn": g["tn"], "fn": g["fn"],
            "cluster_recall": _safe_ratio(g["recovered_gt_clusters_ge_2"],
                                          g["gt_clusters_ge_2"])
        } for name, g in groups.items()},
    })
    return combined


def _average(xs):
    return round(float(np.mean(xs)), 5) if xs and all(x is not None for x in xs) else None


def summarise_rule(cases, seeds, rule):
    outcomes = [evaluate_seed(cases, s, mesh, rule)
                for s in seeds for mesh in MESHES]
    by_mesh = {}
    for mesh in MESHES:
        subset = [x for x in outcomes if x["mesh_m"] == mesh]
        by_mesh[f"{mesh:.2f}"] = {
            "n_seeds": len(subset),
            "mean_recall": _average([x["recall"] for x in subset]),
            "mean_fpr": _average([x["fpr"] for x in subset]),
            "worst_seed_fpr": max(x["fpr"] for x in subset),
            "mean_cluster_recall": _average([x["cluster_recall"] for x in subset]),
            "pooled": {k: sum(x[k] for x in subset)
                       for k in ("tp", "fp", "tn", "fn", "gt_clusters_ge_2",
                                 "recovered_gt_clusters_ge_2", "fp_singletons")},
            "per_seed": subset,
        }
    return {
        "rule": rule[0], "by_mesh": by_mesh,
        "mean_recall_across_both_meshes": _average([x["recall"] for x in outcomes]),
        "mean_fpr_across_both_meshes": _average([x["fpr"] for x in outcomes]),
        "worst_seed_mesh_fpr": max(x["fpr"] for x in outcomes),
        "mean_cluster_recall_across_both_meshes":
            _average([x["cluster_recall"] for x in outcomes]),
    }


def select_on_development(cases):
    """New untouched 481xx only; selected specification is then frozen."""
    sweep = []
    for rank, rule in enumerate(CANDIDATES):
        metrics = summarise_rule(cases, SEED_DEV, rule)
        valid = (metrics["mean_recall_across_both_meshes"] is not None and
                 metrics["mean_fpr_across_both_meshes"] is not None and
                 metrics["mean_fpr_across_both_meshes"] <= DEV_MAX_MEAN_FPR and
                 metrics["worst_seed_mesh_fpr"] <= DEV_MAX_WORST_FPR)
        sweep.append({
            "name": rule[0], "eligible": bool(valid),
            "mean_recall": metrics["mean_recall_across_both_meshes"],
            "mean_fpr": metrics["mean_fpr_across_both_meshes"],
            "worst_fpr": metrics["worst_seed_mesh_fpr"],
            "mean_cluster_recall": metrics["mean_cluster_recall_across_both_meshes"],
            "predeclared_tiebreak_order": rank,
        })
    eligible = [x for x in sweep if x["eligible"]]
    if not eligible:
        return None, sweep
    eligible.sort(key=lambda x: (-x["mean_recall"],
                                  -(x["mean_cluster_recall"] or 0.),
                                  x["mean_fpr"],
                                  x["predeclared_tiebreak_order"]))
    return eligible[0]["name"], sweep


def evaluate_protocol(cases):
    """Require full factorial and select before referencing any holdout scores."""
    expected = {(seed, variant, step) for seed in SEEDS
                for variant in VARIANTS for step in MESHES}
    indexed = {(x["seed"], x["variant"], x["tessellation_m"]): x for x in cases}
    if len(indexed) != len(cases) or set(indexed) != expected:
        raise ValueError("Missing or duplicate canonical CAL-09.4 cases")
    for seed in SEEDS:
        group = [x for x in cases if x["seed"] == seed]
        if len({x["layout_station_signature"] for x in group}) != 1:
            raise ValueError("Different layout/stations among paired variants")
        for x in group:
            if (x["patch_scores"]["format"] != "cal093-multiscale-observed-scores-v1"
                    or x["patch_scores"]["same_rays_each_ablation"] is not True):
                raise ValueError("Unknown patch scores or unpaired ablations")
    development = [x for x in cases if x["seed"] in SEED_DEV]
    selected, sweep = select_on_development(development)
    # Only after freezing the selection are the 491xx holdout scores evaluated.
    holdout = [x for x in cases if x["seed"] in SEED_HOLDOUT]
    reference = {
        "cal09_mad6": summarise_rule(holdout, SEED_HOLDOUT, REFERENCE_MAD6),
        "cal093_shape4": summarise_rule(holdout, SEED_HOLDOUT, REFERENCE_SHAPE4),
    }
    if selected is not None:
        reference[selected] = summarise_rule(holdout, SEED_HOLDOUT,
                                             CANDIDATE_BY_NAME[selected])
    chosen = (reference[selected]["by_mesh"]["0.10"] if selected else None)
    old = reference["cal09_mad6"]["by_mesh"]["0.10"]
    pass_gate = bool(
        chosen and chosen["mean_recall"] is not None and
        chosen["mean_recall"] >= HOLDOUT_MIN_RECALL and
        chosen["mean_fpr"] is not None and
        chosen["mean_fpr"] <= HOLDOUT_MAX_MEAN_FPR and
        chosen["mean_cluster_recall"] is not None and
        chosen["mean_cluster_recall"] >= HOLDOUT_MIN_CLUSTER_RECALL and
        old["mean_recall"] is not None and
        chosen["mean_recall"] - old["mean_recall"] >= HOLDOUT_MIN_GAIN_OVER_MAD6)
    return {
        "format": FORMAT,
        "design": {
            "development_seeds": list(SEED_DEV),
            "independent_holdout_seeds": list(SEED_HOLDOUT),
            "previously_seen_seeds_excluded_from_selection":
                list((46101, 46102, 46105, 46106, 47101, 47102, 47105, 47106)),
            "variants": list(VARIANTS), "meshes_m": list(MESHES),
            "angular_step_deg": .1,
            "direct_ray_sample_limit": 180000,
            "oracle_gt_positive_fraction": ORACLE_GT_FRACTION,
            "candidate_grid_frozen": [list(x) for x in CANDIDATES],
            "dev_max_mean_fpr": DEV_MAX_MEAN_FPR,
            "dev_max_worst_fpr": DEV_MAX_WORST_FPR,
            "holdout_min_recall": HOLDOUT_MIN_RECALL,
            "holdout_max_mean_fpr": HOLDOUT_MAX_MEAN_FPR,
            "holdout_min_cluster_recall": HOLDOUT_MIN_CLUSTER_RECALL,
            "holdout_min_gain_over_mad6": HOLDOUT_MIN_GAIN_OVER_MAD6,
            "uses_ground_truth_to_form_physical_cells": True,
            "uses_ground_truth_in_detector": False,
            "uses_holdout_for_selection": False,
        },
        "development_selection": sweep,
        "selected_on_development_only": selected,
        "independent_holdout": reference,
        "engineering_oracle_gates_pass": pass_gate,
        "decision": ("CAL094_ORACLE_SPATIAL_DIAGNOSTIC_PASS_NOT_CAL09_ACCEPTED"
                     if pass_gate else "CAL094_ORACLE_SPATIAL_DIAGNOSTIC_NOT_ESTABLISHED"),
        "auto_update_production_config": False,
        "limitations": [
            "Wall/face/physical-cell grouping is ground-truth ORACLE, not blind real-world detection.",
            "Detector uses only observed subcell shape scores and physical grid adjacency.",
            "Cells from different geometry variants are never joined into spatial components.",
            "Four fresh 481xx development and four fresh 491xx holdout seeds; small-sample diagnostic.",
            "Prior 461xx/471xx were used in earlier studies and are excluded from tuning.",
            "Per-seed FPR and recall aggregate all three geometry variants incl. uniform control.",
            "Direct rays only; absence of return is not missing-at-random.",
            "True-positive labels are based on fraction of observed direct rays, not wall area.",
            "Mesh changes alter ray intersections; not the same ray set across resolutions.",
            "Real E57 stations and independently surveyed defect truth not available.",
            "Any oracle diagnostic pass cannot override the original failed CAL-09 blind top-24 gates.",
        ],
    }
