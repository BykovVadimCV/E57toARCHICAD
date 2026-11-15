"""CAL-09.3 frozen score sweep, independent-seed model selection and topology.

Thresholds and selection objective must not depend on new holdout seeds.
Only the development seeds 46101, 46102, 46105, 46106 can select a
candidate. On the holdout seeds 47101, 47102, 47105, 47106 report ONLY the
preselected candidate and the fixed legacy MAD>=6mm comparator.

Observed scores use GT-oracle physical patch association, not blind
architecture recovery. All component connectivity is within wall face and
must share a whole horizontal/vertical cell boundary (4-connectivity).
"""
from __future__ import annotations

import math
import numpy as np

from tools.cal092_ray_observability import confusion

DEV_SEEDS = (46101, 46102, 46105, 46106)
HOLDOUT_SEEDS = (47101, 47102, 47105, 47106)
RESOLUTIONS_M = (.30, .10)
VARIANTS = ("uniform_2_4", "zone08_fine35", "zone18_fine35")
LABEL_POSITIVE_FRACTION = .40
# Frozen before any new holdout results were inspected.
CANDIDATES = (
    ("mad2", "legacy_mad", 2.0),
    ("mad3", "legacy_mad", 3.0),
    ("mad4", "legacy_mad", 4.0),
    ("mad5", "legacy_mad", 5.0),
    ("mad6", "legacy_mad", 6.0),
    ("macro2", "macro_offset", 2.0),
    ("macro3", "macro_offset", 3.0),
    ("macro4", "macro_offset", 4.0),
    ("macro5", "macro_offset", 5.0),
    ("macro6", "macro_offset", 6.0),
    ("macro8", "macro_offset", 8.0),
    ("shape1", "subcell_shape", 1.0),
    ("shape1p5", "subcell_shape", 1.5),
    ("shape2", "subcell_shape", 2.0),
    ("shape3", "subcell_shape", 3.0),
    ("shape4", "subcell_shape", 4.0),
    ("or_mad3_macro4", "or_mad3_macro4", None),
    ("or_mad4_macro5", "or_mad4_macro5", None),
    ("or_mad5_macro6", "or_mad5_macro6", None),
)
CANDIDATE_BY_NAME = {c[0]: c for c in CANDIDATES}
BASELINE = "mad6"
# Engineering observability diagnostics, NOT physical-E57 realism gates.
DEVELOPMENT_MAX_MEAN_FPR = .10
DEVELOPMENT_MAX_WORST_FPR = .20
HOLDOUT_MIN_RECALL = .45
HOLDOUT_MAX_FPR = .10
HOLDOUT_MIN_CLUSTER_RECALL = .30


def _score_positive(cell, candidate, *, mode="combined"):
    _, kind, threshold = candidate
    scores = cell["scores_mm"][mode]
    if kind.startswith("or_mad"):
        options = {
            "or_mad3_macro4": (3., 4.),
            "or_mad4_macro5": (4., 5.),
            "or_mad5_macro6": (5., 6.),
        }
        a, b = options[kind]
        return ((scores.get("legacy_mad") is not None and scores["legacy_mad"] >= a)
                or (scores.get("macro_offset") is not None and scores["macro_offset"] >= b))
    score = scores.get(kind)
    return score is not None and np.isfinite(score) and score >= threshold


def _components(cell_keys):
    """4-neighbour connected components for a single wall face and grid."""
    remain = set(tuple(k) for k in cell_keys)
    components = []
    while remain:
        root = remain.pop()
        component = {root}
        stack = [root]
        while stack:
            current = stack.pop()
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = current[:3] + (current[3] + dc, current[4] + dr)
                if nxt in remain:
                    remain.remove(nxt)
                    component.add(nxt)
                    stack.append(nxt)
        components.append(component)
    return components


def evaluate_cells(cells, candidate, *, mode="combined"):
    truth = [float(x["gt_visible_fraction"]) >= LABEL_POSITIVE_FRACTION for x in cells]
    predictions = [_score_positive(x, candidate, mode=mode) for x in cells]
    cm = confusion([
        {"gt_visible_fraction": float(gt), "mad_mm": float(pred)}
        for gt, pred in zip(truth, predictions)
    ], threshold=.5, gt_fraction=.5)

    gt_keys = {tuple(row["cell_key"]) for row, positive in zip(cells, truth)
               if positive}
    detected_gt = {tuple(row["cell_key"]) for row, gt, pred in zip(
        cells, truth, predictions) if gt and pred}
    gt_clusters = [component for component in _components(gt_keys)
                   if len(component) >= 2]
    detected_connected = [component for component in gt_clusters
                          if any(len(found) >= 2 for found in
                                 _components(component & detected_gt))]
    component_recall = (
        round(len(detected_connected) / len(gt_clusters), 4)
        if gt_clusters else None)
    missed_scores = sum(
        row["scores_mm"][mode].get(candidate[1]) is None
        for row in cells
    ) if not candidate[1].startswith("or_") else 0
    return {
        **cm,
        "cluster_recall": component_recall,
        "n_gt_clusters_size_ge_2": len(gt_clusters),
        "n_detected_gt_clusters_size_ge_2": len(detected_connected),
        "fraction_missing_feature": round(missed_scores / len(cells), 4)
                                    if cells else None,
    }


def _median(values):
    seq = [x for x in values if x is not None]
    return round(float(np.median(seq)), 4) if seq else None


def _mean(values):
    seq = [x for x in values if x is not None]
    return round(float(np.mean(seq)), 5) if seq else None


def merged_seed_cells(cases, seed, resolution):
    group = [case for case in cases
             if case["seed"] == seed and case["tessellation_m"] == resolution]
    if {c["variant"] for c in group} != set(VARIANTS):
        raise ValueError("Missing variant or duplicated variant in seed grouping")
    return [cell for entry in group for cell in entry["patch_scores"]["cells"]]


def _per_seed_metrics(cases, seeds, spec):
    records = []
    for seed in seeds:
        for resolution in RESOLUTIONS_M:
            cells = merged_seed_cells(cases, seed, resolution)
            result = evaluate_cells(cells, spec)
            records.append({"seed": seed, "resolution_m": resolution, **result})
    return records


def select_on_development(cases):
    """Freeze the selected candidate using ONLY development seeds and controls."""
    sweep = []
    for spec in CANDIDATES:
        metrics = _per_seed_metrics(cases, DEV_SEEDS, spec)
        recall = [x["recall"] for x in metrics]
        fpr = [x["false_positive_rate"] for x in metrics]
        if any(x is None for x in fpr):
            raise ValueError("No negative cells for calibration")
        mean_fpr = _mean(fpr)
        worst = max(fpr)
        permitted = (mean_fpr <= DEVELOPMENT_MAX_MEAN_FPR
                     and worst <= DEVELOPMENT_MAX_WORST_FPR)
        sweep.append({
            "name": spec[0], "mean_recall": _mean(recall),
            "mean_fpr": mean_fpr,
            "worst_seed_resolution_fpr": worst,
            "mean_cluster_recall": _mean([x["cluster_recall"] for x in metrics]),
            "per_development_seed_and_resolution": metrics,
            "permitted": bool(permitted),
        })
    candidates = [r for r in sweep
                  if r["permitted"] and r["mean_recall"] is not None]
    if not candidates:
        return None, sweep
    # Max recall first, then connected GT recall; minimize FPR then prefer
    # predeclared order. No holdout scores are available here.
    priority = {spec[0]: i for i, spec in enumerate(CANDIDATES)}
    candidates.sort(key=lambda r: (-r["mean_recall"],
                                    -(r["mean_cluster_recall"] or 0.),
                                    r["mean_fpr"], priority[r["name"]]))
    return candidates[0]["name"], sweep


def analyze_holdout(cases):
    """Strictly require all eight predeclared seeds, select before holdout."""
    expected = {(seed, var, resolution)
                for seed in DEV_SEEDS + HOLDOUT_SEEDS
                for var in VARIANTS for resolution in RESOLUTIONS_M}
    indexed = {(c["seed"], c["variant"], c["tessellation_m"]): c for c in cases}
    if len(indexed) != len(cases) or set(indexed) != expected:
        raise ValueError("Missing/duplicate CAL-09.3 canonical cases")
    if any(c["patch_scores"]["format"] !=
           "cal093-multiscale-observed-scores-v1" for c in cases):
        raise ValueError("Unexpected score schema")
    paired = []
    for seed in DEV_SEEDS + HOLDOUT_SEEDS:
        same = [c for c in cases if c["seed"] == seed]
        paired.append({
            "seed": seed,
            "matched_layout_and_station_poses": len({
                c["layout_station_signature"] for c in same}) == 1,
            "all_modes_identical_direct_rays": all(
                c["patch_scores"]["same_rays_each_ablation"] for c in same),
        })
    if not all(x["matched_layout_and_station_poses"] and
               x["all_modes_identical_direct_rays"] for x in paired):
        raise ValueError("Pairing/provenance validation failed")

    # Function selects on development only. Holdout scores are NOT consulted
    # by either threshold selection or tie breaking.
    development = [c for c in cases if c["seed"] in DEV_SEEDS]
    selected, sweep = select_on_development(development)
    holdout = [c for c in cases if c["seed"] in HOLDOUT_SEEDS]
    reporting = [BASELINE] + ([selected] if selected is not None and selected != BASELINE else [])
    comparisons = {}
    for name in reporting:
        spec = CANDIDATE_BY_NAME[name]
        metrics = _per_seed_metrics(holdout, HOLDOUT_SEEDS, spec)
        by_resolution = {}
        for step in RESOLUTIONS_M:
            group = [m for m in metrics if m["resolution_m"] == step]
            by_resolution[f"{step:.2f}"] = {
                "n_independent_holdout_seeds": len(group),
                "mean_recall": _mean([x["recall"] for x in group]),
                "median_recall": _median([x["recall"] for x in group]),
                "mean_fpr": _mean([x["false_positive_rate"] for x in group]),
                "mean_cluster_recall": _mean([x["cluster_recall"] for x in group]),
                "gt_positive_patches": sum(x["n_gt_positive"] for x in group),
                "all_evaluated_patches": sum(x["n_patches"] for x in group),
                "per_seed": group,
            }
        comparisons[name] = by_resolution
    selected_at_010 = (comparisons.get(selected, {}).get("0.10", {})
                       if selected is not None else {})
    qualifies = (selected is not None and
                 selected_at_010.get("mean_recall") is not None and
                 selected_at_010["mean_recall"] >= HOLDOUT_MIN_RECALL and
                 selected_at_010["mean_fpr"] is not None and
                 selected_at_010["mean_fpr"] <= HOLDOUT_MAX_FPR and
                 selected_at_010.get("mean_cluster_recall") is not None and
                 selected_at_010["mean_cluster_recall"] >= HOLDOUT_MIN_CLUSTER_RECALL)

    return {
        "format": "cal093-holdout-multiscale-observability-v1",
        "design": {
            "development_seeds": list(DEV_SEEDS),
            "independent_holdout_seeds": list(HOLDOUT_SEEDS),
            "variants": list(VARIANTS),
            "mesh_resolution_m": list(RESOLUTIONS_M),
            "sampled_direct_ray_step_deg": .1,
            "heldout_seeds_used_in_selection": False,
            "gt_oracle_grouping": True,
            "gt_positive_fraction_threshold": LABEL_POSITIVE_FRACTION,
            "candidate_rules_predeclared": list(CANDIDATES),
            "dev_max_mean_fpr": DEVELOPMENT_MAX_MEAN_FPR,
            "dev_max_worst_fpr": DEVELOPMENT_MAX_WORST_FPR,
            "holdout_min_mean_recall": HOLDOUT_MIN_RECALL,
            "holdout_max_mean_fpr": HOLDOUT_MAX_FPR,
            "holdout_min_mean_cluster_recall": HOLDOUT_MIN_CLUSTER_RECALL,
        },
        "paired_seed_checks": paired,
        "development_threshold_sweep": sweep,
        "selected_using_development_only": selected,
        "holdout_selected_vs_fixed_mad6": comparisons,
        "engineering_oracle_gates_pass": bool(qualifies),
        "decision": "ORACLE_MULTISCALE_DIAGNOSTIC_GATES_PASS_NOT_CAL09_ACCEPTED"
                    if qualifies else "CAL093_ORACLE_OBSERVABILITY_NOT_ESTABLISHED",
        "auto_update_production_config": False,
        "limitations": [
            "Patches are grouped by oracle wall/face/physical cell. This is NOT blind detection.",
            "CAL-09.3 scores use observed XYZ only but patches are formed using the known mesh.",
            "Ground-truth-positive definition is >=40% sampled valid direct returns; not wall area.",
            "The global-face reference may absorb a wall-sized deformation and is not a known plane.",
            "Four holdout seeds are independent of CAL-09/09.1/09.2 but exploratory for production.",
            "Control uniform_2_4 has no injected CAL-09 envelope and remains a true-negative control.",
            "Only four fixed development seeds tune one candidate; holdout candidates are not swept.",
            "Differences between 0.30 and 0.10 mesh resolutions are not ray-for-ray paired.",
            "Observed E57 physical-station provenance remains unavailable; no real-world validation.",
            "Original CAL-09 blind top-24 fails and is NOT overridden by an oracle result.",
        ],
    }
