"""CAL-09.5: soft spatial confidence and dev-only grouped calibration.

No detector receives ground-truth labels. Training uses GT on 501xx
development scenes only. Feature computation uses CAL-09.3's observed XYZ
scores and 4-neighbour *oracle-associated* wall-cell keys. The untouched
511xx holdout is evaluated exactly once after feature set, L2 and operating
threshold are selected by leave-one-seed-out development CV.

This is an oracle-cell *diagnostic*, not blind E57 geometry reconstruction.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from tools.cal093_threshold_selection import _components
from tools.cal094_spatial_selection import REFERENCE, evaluate_one

FORMAT = "cal095-soft-confidence-independent-holdout-v1"
DEV = (50101, 50102, 50105, 50106)
HOLDOUT = (51101, 51102, 51105, 51106)
SEEDS = DEV + HOLDOUT
VARIANTS = ("uniform_2_4", "zone08_fine35", "zone18_fine35")
MESH_M = .10
GT_FRACTION = .40
# Frozen prior to obtaining any 511xx result.
FEATURE_SETS = ("local", "shape_context", "full_context")
L2_GRID = (.01, .1)
NEGATIVE_QUANTILES = (.80, .85, .90, .925, .95, .975, .99)
DEV_MAX_MEAN_FPR = .08
DEV_MAX_WORST_FPR = .15
HOLDOUT_MIN_RECALL = .40
HOLDOUT_MAX_MEAN_FPR = .10
HOLDOUT_MIN_CLUSTER_RECALL = .35
HOLDOUT_MIN_RECALL_GAIN_VS_MAD6 = .15


def _scalar(row, key):
    value = row["scores_mm"]["combined"].get(key)
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) and v >= 0. else None


def features(cells, feature_set):
    """Observed-only, permutation-invariant, same-wall 4-neighbour features.

    Missing shape values are imputed as 0 with an explicit missing indicator;
    missing neighbours never count as positive votes. The function NEVER
    consults gt_visible_fraction, variant or scene seed.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError("Unknown predeclared feature set")
    keys = [tuple(int(x) for x in row["cell_key"]) for row in cells]
    if len(set(keys)) != len(keys) or any(len(k) != 5 for k in keys):
        raise ValueError("Invalid/duplicate oracle cell keys")
    by_key = {key: row for key, row in zip(keys, cells)}
    if not cells:
        return np.empty((0, {"local": 4, "shape_context": 6, "full_context": 7}[feature_set]))
    out = []
    for key, row in zip(keys, cells):
        shape, mad, macro = (_scalar(row, x) for x in
                             ("subcell_shape", "legacy_mad", "macro_offset"))
        base = [
            np.log1p(shape if shape is not None else 0.),
            np.log1p(mad if mad is not None else 0.),
            np.log1p(macro if macro is not None else 0.),
            float(shape is None),
        ]
        if feature_set != "local":
            neighbor_values = []
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                near = by_key.get(key[:3] + (key[3] + dc, key[4] + dr))
                if near is None:
                    continue
                value = _scalar(near, "subcell_shape")
                if value is not None:
                    neighbor_values.append(value)
            base.extend((
                np.log1p(float(np.mean(neighbor_values))) if neighbor_values else 0.,
                len(neighbor_values) / 4.,
            ))
            if feature_set == "full_context":
                base.append(
                    np.log1p(float(max(neighbor_values))) if neighbor_values else 0.)
        out.append(base)
    return np.asarray(out, dtype=np.float64).reshape(len(cells), -1)


def fit_logistic(X, y, *, l2):
    """Deterministic regularised logistic regression without sklearn.

    Standardization parameters are estimated exclusively from the training
    fold. Unweighted log-loss keeps the score on a prevalence-like scale;
    its calibration is checked separately on the untouched holdout.
    """
    x = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if (x.ndim != 2 or len(x) != len(labels) or len(x) < 15 or
            not np.isfinite(x).all() or
            set(np.unique(labels)) != {0., 1.} or l2 <= 0):
        raise ValueError("Invalid training fold: requires both classes, finite data")
    mu = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-9] = 1.
    z = (x - mu) / scale
    z = np.column_stack((np.ones(len(z)), z))
    def loss(beta):
        f = z @ beta
        probs = expit(f)
        penal = .5 * l2 * np.sum(beta[1:] ** 2)
        obj = np.mean(np.logaddexp(0., f) - labels * f) + penal
        gradient = z.T @ (probs - labels) / len(labels)
        gradient[1:] += l2 * beta[1:]
        return float(obj), gradient
    initial = np.zeros(z.shape[1], float)
    rate = float(labels.mean())
    initial[0] = np.log(rate / (1 - rate))
    result = minimize(loss, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 250, "ftol": 1e-12})
    if not result.success and np.linalg.norm(result.jac) > 1e-5:
        raise RuntimeError("CAL-09.5 logistic solver did not converge")
    return {"mean": mu, "std": scale, "weights": result.x, "l2": l2}


def confidence(model, X):
    x = np.asarray(X, float)
    return expit(model["weights"][0] +
                 ((x - model["mean"]) / model["std"]) @ model["weights"][1:])


def labels(cells):
    return np.array(
        [float(row["gt_visible_fraction"]) >= GT_FRACTION for row in cells],
        dtype=bool)


def _group(cases, seed):
    subset = [case for case in cases if case["seed"] == seed]
    if len(subset) != len(VARIANTS) or {c["variant"] for c in subset} != set(VARIANTS):
        raise ValueError("Missing or duplicate geometry variant for seed")
    return sorted(subset, key=lambda c: VARIANTS.index(c["variant"]))


def _matrix(cases, feature_set):
    rows = []
    for case in cases:
        cells = case["patch_scores"]["cells"]
        x = features(cells, feature_set)
        y = labels(cells)
        if len(x):
            rows.append((x, y))
    if not rows:
        raise ValueError("No eligible direct-ray physical cells")
    return np.vstack([v[0] for v in rows]), np.concatenate([v[1] for v in rows])


def _confusion(cells, probabilities, threshold):
    if len(cells) != len(probabilities):
        raise ValueError("Predictions/cells length mismatch")
    keys = [tuple(int(v) for v in row["cell_key"]) for row in cells]
    truth = {key for key, gt in zip(keys, labels(cells)) if gt}
    selected = {key for key, score in zip(keys, probabilities) if score >= threshold}
    pos = len(truth & selected)
    fp = len(selected - truth)
    neg = len(cells) - len(truth)
    clusters = [c for c in _components(truth) if len(c) >= 2]
    recovered = sum(any(len(part) >= 2 for part in _components(c & selected))
                    for c in clusters)
    return {
        "tp": pos, "fp": fp, "fn": len(truth) - pos, "tn": neg - fp,
        "n_gt_clusters": len(clusters), "n_recovered_clusters": recovered,
    }


def _safe(num, denom):
    return round(float(num) / denom, 6) if denom else None


def _mean(vals):
    if not vals or any(v is None for v in vals):
        return None
    return round(float(np.mean(vals)), 5)


def _metrics_for_seed(case_preds, threshold):
    """All variants kept topologically separate, counts pooled within seed."""
    counts = [_confusion(case["patch_scores"]["cells"], probs, threshold)
              for case, probs in case_preds]
    total = {key: sum(d[key] for d in counts) for key in counts[0]}
    return {
        **total,
        "recall": _safe(total["tp"], total["tp"] + total["fn"]),
        "fpr": _safe(total["fp"], total["fp"] + total["tn"]),
        "precision": _safe(total["tp"], total["tp"] + total["fp"]),
        "cluster_recall": _safe(total["n_recovered_clusters"],
                                total["n_gt_clusters"]),
    }


def _summarize(per_seed):
    keys = ("recall", "fpr", "precision", "cluster_recall")
    return {
        "mean_" + k: _mean([x[k] for x in per_seed]) for k in keys
    } | {
        "worst_seed_fpr": max(x["fpr"] for x in per_seed),
        "pooled": {k: sum(x[k] for x in per_seed)
                   for k in ("tp", "fp", "tn", "fn",
                             "n_gt_clusters", "n_recovered_clusters")},
        "per_seed": per_seed,
    }


def _seed_predictions(cases, model, feature_set, seed):
    return [
        (case, confidence(model, features(case["patch_scores"]["cells"], feature_set)))
        for case in _group(cases, seed)
    ]


def crossfit_dev(cases, feature_set, l2):
    """Leave-one-*scene-seed*-out scores, no wall/variant leakage."""
    predictions = {}
    for held in DEV:
        train = [case for case in cases if case["seed"] in DEV and case["seed"] != held]
        x, y = _matrix(train, feature_set)
        model = fit_logistic(x, y, l2=l2)
        predictions[held] = _seed_predictions(cases, model, feature_set, held)
    return predictions


def _negative_scores(folds):
    out = []
    for scenes in folds.values():
        for case, scores in scenes:
            out.extend(scores[~labels(case["patch_scores"]["cells"])].tolist())
    return np.asarray(out, float)


def select_on_dev(dev_cases):
    """Choose feature set, regularisation and confidence threshold on 501xx."""
    if {c["seed"] for c in dev_cases} != set(DEV):
        raise ValueError("Only complete frozen 501xx development is selectable")
    sweep = []
    for feature_set in FEATURE_SETS:
        for l2 in L2_GRID:
            folds = crossfit_dev(dev_cases, feature_set, l2)
            negatives = _negative_scores(folds)
            if len(negatives) < 40:
                raise ValueError("Too few negative development patches to set FPR")
            for quantile in NEGATIVE_QUANTILES:
                threshold = float(np.quantile(negatives, quantile))
                seed_metrics = []
                for seed in DEV:
                    d = _metrics_for_seed(folds[seed], threshold)
                    seed_metrics.append({"seed": seed, **d})
                summary = _summarize(seed_metrics)
                valid = (summary["mean_fpr"] is not None and
                         summary["mean_recall"] is not None and
                         summary["mean_fpr"] <= DEV_MAX_MEAN_FPR and
                         summary["worst_seed_fpr"] <= DEV_MAX_WORST_FPR)
                sweep.append({
                    "feature_set": feature_set, "l2": l2,
                    "negative_quantile": quantile,
                    "threshold": round(threshold, 8),
                    "eligible": bool(valid),
                    "mean_recall": summary["mean_recall"],
                    "mean_fpr": summary["mean_fpr"],
                    "worst_seed_fpr": summary["worst_seed_fpr"],
                    "mean_cluster_recall": summary["mean_cluster_recall"],
                })
    good = [x for x in sweep if x["eligible"]]
    if not good:
        return None, sweep
    # Frozen selection hierarchy. No holdout scores can influence sorting.
    order = {s: i for i, s in enumerate(FEATURE_SETS)}
    good.sort(key=lambda x: (-x["mean_recall"],
                              -(x["mean_cluster_recall"] or 0.),
                              x["mean_fpr"],
                              order[x["feature_set"]], x["l2"],
                              x["negative_quantile"]))
    return good[0], sweep


def _reference_metrics(cases, seed, name):
    rule = REFERENCE[name]
    ref = [evaluate_one(case["patch_scores"]["cells"], rule)
           for case in _group(cases, seed)]
    total = {k: sum(entry[k] for entry in ref)
             for k in ("tp", "fp", "tn", "fn",
                       "gt_clusters_ge_2", "recovered_gt_clusters_ge_2")}
    return {
        "tp": total["tp"], "fp": total["fp"], "tn": total["tn"],
        "fn": total["fn"],
        "n_gt_clusters": total["gt_clusters_ge_2"],
        "n_recovered_clusters": total["recovered_gt_clusters_ge_2"],
        "recall": _safe(total["tp"], total["tp"] + total["fn"]),
        "fpr": _safe(total["fp"], total["fp"] + total["tn"]),
        "precision": _safe(total["tp"], total["tp"] + total["fp"]),
        "cluster_recall": _safe(total["recovered_gt_clusters_ge_2"],
                                total["gt_clusters_ge_2"]),
    }


def _holdout_score(cases, dev_cases, selected):
    model_x, model_y = _matrix(dev_cases, selected["feature_set"])
    model = fit_logistic(model_x, model_y, l2=selected["l2"])
    records = []
    brier_parts = []
    bins = np.linspace(0, 1, 6)
    for seed in HOLDOUT:
        group = _seed_predictions(cases, model, selected["feature_set"], seed)
        scores = _metrics_for_seed(group, selected["threshold"])
        records.append({"seed": seed, **scores})
        for case, probs in group:
            true = labels(case["patch_scores"]["cells"]).astype(float)
            brier_parts.extend(zip(probs.tolist(), true.tolist()))
    summ = _summarize(records)
    # Diagnostic calibration quality, not a calibrated-probability guarantee.
    xy = np.asarray(brier_parts, float)
    if len(xy):
        summ["brier_score"] = round(float(np.mean((xy[:, 0] - xy[:, 1]) ** 2)), 6)
        ece = 0.
        for k in range(len(bins) - 1):
            pick = ((xy[:, 0] >= bins[k]) & (xy[:, 0] < bins[k + 1]))
            if k == len(bins) - 2:
                pick |= xy[:, 0] == 1.
            if pick.any():
                ece += pick.mean() * abs(xy[pick, 0].mean() - xy[pick, 1].mean())
        summ["ece_5bins"] = round(float(ece), 6)
    return summ


def evaluate_protocol(cases):
    """Strict factorial validation, isolate selection from all 511xx data."""
    expected = {(seed, variant, MESH_M) for seed in SEEDS for variant in VARIANTS}
    indexed = {(x["seed"], x["variant"], x["tessellation_m"]): x for x in cases}
    if len(indexed) != len(cases) or set(indexed) != expected:
        raise ValueError("Missing/duplicate CAL-09.5 canonical 24 scans")
    for seed in SEEDS:
        group = _group(cases, seed)
        if len({x["layout_station_signature"] for x in group}) != 1:
            raise ValueError("Paired layout and stations diverge")
        for c in group:
            if (c["patch_scores"]["format"] != "cal093-multiscale-observed-scores-v1"
                    or c["patch_scores"]["same_rays_each_ablation"] is not True
                    or c["angular_step_deg"] != .1):
                raise ValueError("Invalid physical patch provenance or scanner config")
    # Explicitly partition before selection. Never pass holdout into tuner.
    dev = [x for x in cases if x["seed"] in DEV]
    selected, sweep = select_on_dev(dev)
    holdout = [x for x in cases if x["seed"] in HOLDOUT]
    fixed = {
        name: _summarize([{"seed": seed, **_reference_metrics(holdout, seed, name)}
                          for seed in HOLDOUT])
        for name in ("cal09_mad6", "cal093_shape4")
    }
    chosen = _holdout_score(holdout, dev, selected) if selected else None
    if selected is not None:
        fixed["cal095_selected"] = chosen
    old = fixed["cal09_mad6"]
    pass_gate = bool(
        chosen and chosen["mean_recall"] is not None and
        chosen["mean_recall"] >= HOLDOUT_MIN_RECALL and
        chosen["mean_fpr"] is not None and
        chosen["mean_fpr"] <= HOLDOUT_MAX_MEAN_FPR and
        chosen["mean_cluster_recall"] is not None and
        chosen["mean_cluster_recall"] >= HOLDOUT_MIN_CLUSTER_RECALL and
        old["mean_recall"] is not None and
        chosen["mean_recall"] - old["mean_recall"] >= HOLDOUT_MIN_RECALL_GAIN_VS_MAD6)
    return {
        "format": FORMAT,
        "design": {
            "development_seeds": list(DEV),
            "untouched_holdout_seeds": list(HOLDOUT),
            "geometry_variants": list(VARIANTS),
            "mesh_m": MESH_M,
            "direct_ray_angular_step_deg": .1,
            "max_sampled_direct_returns": 180000,
            "score_uses_truth_label_in_features": False,
            "physical_cell_association_is_gt_oracle": True,
            "used_461xx_471xx_481xx_491xx_for_selection": False,
            "feature_sets": list(FEATURE_SETS), "l2_grid": list(L2_GRID),
            "negative_threshold_quantiles": list(NEGATIVE_QUANTILES),
            "development_max_mean_fpr": DEV_MAX_MEAN_FPR,
            "development_max_worst_fpr": DEV_MAX_WORST_FPR,
            "holdout_min_mean_recall": HOLDOUT_MIN_RECALL,
            "holdout_max_fpr": HOLDOUT_MAX_MEAN_FPR,
            "holdout_min_cluster_recall": HOLDOUT_MIN_CLUSTER_RECALL,
            "holdout_min_recall_gain_over_mad6": HOLDOUT_MIN_RECALL_GAIN_VS_MAD6,
        },
        "development_crossfit_selection": sweep,
        "selected_on_development_only": selected,
        "holdout_predeclared_comparators": fixed,
        "engineering_oracle_gates_pass": pass_gate,
        "decision": ("CAL095_ORACLE_SOFT_CONFIDENCE_PASS_NOT_CAL09_ACCEPTED" if pass_gate
                     else "CAL095_ORACLE_SOFT_CONFIDENCE_NOT_ESTABLISHED"),
        "auto_update_production_config": False,
        "limitations": [
            "Oracle physical-cell association uses true wall triangle IDs; not blind E57 detection.",
            "Detector's local and neighboring features are based only on measured XYZ scores.",
            "Supervised training sees GT labels on four NEW 501xx development scenes only.",
            "501xx leave-one-seed-out CV calibrates threshold without using 511xx.",
            "Refitted full-development model may shift score distribution relative to crossfit.",
            "Four 511xx holdout seeds form a small independent synthetic diagnostic only.",
            "A synthetic PASS cannot overrule the original CAL-09 blind top-24 FAIL.",
            "True labels represent a visible direct-ray fraction, not physical wall area.",
            "Every component and neighbor relationship is restricted to one scan wall face.",
            "Missing subcell_shape is flagged, never interpreted as a measured zero.",
            "Real E57 station provenance and independent geometric defect truth remain unavailable.",
        ],
    }
