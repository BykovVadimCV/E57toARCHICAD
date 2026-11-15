"""Aggregate one CAL-09.6 development or untouched holdout seed shard set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.validate_cal096_blind_geometry import (
    DEV_SEEDS, HOLDOUT_SEEDS, VARIANTS, MESH_M, FORMAT,
)

SHARD_FORMAT = FORMAT
GATES = {
    "mean_point_precision": .90,
    "mean_point_recall": .70,
    "mean_point_fpr": .10,
    "worst_seed_point_fpr": .15,
    "mean_wall_instance_recall": .80,
    "mean_topology_edge_precision": .80,
    "mean_topology_edge_recall": .70,
    "mean_topology_edge_type_accuracy": .80,
}


def _mean(values):
    values = [float(x) for x in values if x is not None]
    return round(float(np.mean(values)), 6) if values else None


def aggregate(paths, phase):
    if phase not in ("development", "holdout"):
        raise ValueError("phase must be development or holdout")
    expected_seeds = DEV_SEEDS if phase == "development" else HOLDOUT_SEEDS
    if len(paths) != len(expected_seeds):
        raise ValueError("Expected one shard for each predeclared phase seed")
    cases, fingerprints, config_hashes, seen = [], set(), set(), []
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("format") != SHARD_FORMAT or doc.get("auto_update_production_config") is not False:
            raise ValueError("Not a CAL-09.6 seed shard")
        if (doc.get("variants") != list(VARIANTS) or doc.get("mesh_m") != MESH_M or
                doc.get("angular_step_deg") != .1 or doc.get("max_points") != 200_000 or
                doc.get("n_cases") != len(VARIANTS) or len(doc.get("cases", [])) != len(VARIANTS)):
            raise ValueError("Noncanonical CAL-09.6 seed shard")
        if set(c["seed"] for c in doc["cases"]) != {doc["seed"]}:
            raise ValueError("Shard case seed mismatch")
        if {c["variant"] for c in doc["cases"]} != set(VARIANTS):
            raise ValueError("Shard variant mismatch")
        if any(c["detector_uses_ground_truth"] is not False or
               c["detector_inputs"] != ["cartesianX", "cartesianY", "cartesianZ"]
               for c in doc["cases"]):
            raise ValueError("Blind XYZ information boundary was violated")
        seen.append(doc["seed"])
        fingerprints.add(doc["run_fingerprint"])
        config_hashes.add(doc["source_config_sha256"])
        cases.extend(doc["cases"])
    if (set(seen) != set(expected_seeds) or len(set(seen)) != len(seen) or
            len(fingerprints) != 1 or len(config_hashes) != 1):
        raise ValueError("Missing/duplicate seed or mixed run/config fingerprint")
    for seed in expected_seeds:
        group = [c for c in cases if c["seed"] == seed]
        if len({c["layout_station_signature"] for c in group}) != 1:
            raise ValueError("Paired variants have different layouts/stations")
    keys = {
        "point_precision": lambda c: c["metrics"]["point"]["precision"],
        "point_recall": lambda c: c["metrics"]["point"]["recall"],
        "point_fpr": lambda c: c["metrics"]["point"]["false_positive_rate"],
        "point_f1": lambda c: c["metrics"]["point"]["f1"],
        "wall_instance_recall": lambda c: c["metrics"]["segments"]["instance_recall"],
        "topology_edge_precision": lambda c: c["metrics"]["topology"]["edge_precision"],
        "topology_edge_recall": lambda c: c["metrics"]["topology"]["edge_recall"],
        "topology_edge_f1": lambda c: c["metrics"]["topology"]["edge_f1"],
        "topology_edge_type_accuracy": lambda c: c["metrics"]["topology"]["edge_type_accuracy"],
    }
    per_seed = []
    for seed in expected_seeds:
        group = [c for c in cases if c["seed"] == seed]
        record = {"seed": seed, "cases": len(group)}
        for name, get in keys.items():
            record[name] = _mean([get(c) for c in group])
        record["predicted_segments"] = int(sum(
            c["metrics"]["segments"]["candidate_count"] for c in group))
        record["predicted_edges"] = int(sum(
            c["metrics"]["topology"]["predicted_edge_count"] for c in group))
        per_seed.append(record)
    summary = {name: _mean([s[name] for s in per_seed]) for name in keys}
    summary["worst_seed_point_fpr"] = (
        max(s["point_fpr"] for s in per_seed if s["point_fpr"] is not None)
        if any(s["point_fpr"] is not None for s in per_seed) else None)
    gate_results = {}
    if phase == "holdout":
        for name, threshold in GATES.items():
            value = summary.get(name)
            gate_results[name] = bool(
                value is not None and
                (value <= threshold if "fpr" in name else value >= threshold))
    passed = phase == "holdout" and bool(gate_results) and all(gate_results.values())
    return {
        "format": "cal096-blind-xyz-aggregate-v1",
        "phase": phase,
        "development_seeds": list(DEV_SEEDS),
        "holdout_seeds": list(HOLDOUT_SEEDS),
        "evaluated_seeds": list(expected_seeds),
        "variants": list(VARIANTS),
        "mesh_m": MESH_M,
        "angular_step_deg": .1,
        "max_points": 200_000,
        "detector_receives_only_xyz": True,
        "source_config_sha256": next(iter(config_hashes)),
        "run_fingerprint": next(iter(fingerprints)),
        "case_count": len(cases),
        "summary_mean_of_seed_means": summary,
        "per_seed": per_seed,
        "preregistered_holdout_gates": GATES,
        "holdout_gate_results": gate_results,
        "decision": (
            "CAL096_BLIND_XYZ_TOPOLOGY_PASS_ENGINEERING_DIAGNOSTIC_ONLY"
            if passed else
            "CAL096_DEVELOPMENT_METRICS_ONLY" if phase == "development" else
            "CAL096_BLIND_XYZ_TOPOLOGY_NOT_ESTABLISHED"),
        "auto_update_production_config": False,
        "limitations": [
            "The detector receives only registered XYZ; the separate evaluator uses synthetic point labels and layout walls.",
            "Synthetic scanner geometry and materials do not represent the full range of real E57 occlusion, clutter and registration errors.",
            "The point sample is voxel-thinned; metrics are measured on the same sampled points, not raw point multiplicity.",
            "Wall instance mapping is evaluation-only and relies on the synthetic SceneBuilder wall creation order.",
            "The algorithm proposes vertical planar surfaces; flush furniture or a perfectly coplanar non-wall object can be geometrically indistinguishable.",
            "A holdout PASS is not real-E57 validation, defect detection acceptance, or production approval.",
            "Real E57 inputs with independently surveyed wall geometry and topology are still required.",
        ],
    }


def markdown(doc):
    s = doc["summary_mean_of_seed_means"]
    fmt = lambda x: "n/a" if x is None else f"{x:.3f}"
    lines = [
        f"# CAL-09.6 blind XYZ geometry: {doc['phase']}",
        "",
        f"Seeds: {doc['evaluated_seeds']}; variants: {doc['variants']}; "
        f"mesh={doc['mesh_m']:.2f}m; scanner={doc['angular_step_deg']:.1f}°.",
        "Detector inputs: Cartesian XYZ only. No labels, triangle IDs, wall IDs, station IDs or layout are passed into inference.",
        "",
        "| Metric | Mean of seed means |",
        "|---|---:|",
    ]
    for name, value in s.items():
        lines.append(f"| {name} | {fmt(value)} |")
    lines += ["", "| Seed | Point precision | Point recall | Point FPR | Wall instance recall | Topology edge P | Topology edge R |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in doc["per_seed"]:
        lines.append(
            f"| {row['seed']} | {fmt(row['point_precision'])} | "
            f"{fmt(row['point_recall'])} | {fmt(row['point_fpr'])} | "
            f"{fmt(row['wall_instance_recall'])} | "
            f"{fmt(row['topology_edge_precision'])} | {fmt(row['topology_edge_recall'])} |")
    if doc["phase"] == "holdout":
        lines += ["", "## Preregistered holdout gates", "",
                  "| Gate | Required | Result | Pass |",
                  "|---|---:|---:|---|"]
        for name, threshold in doc["preregistered_holdout_gates"].items():
            value = s.get(name)
            comparator = "<=" if "fpr" in name else ">="
            lines.append(
                f"| {name} | {comparator} {threshold:.2f} | {fmt(value)} | "
                f"{'yes' if doc['holdout_gate_results'].get(name) else 'no'} |")
    lines += ["", f"**Decision: {doc['decision']}**", "",
              "## Limitations"]
    lines += ["- " + x for x in doc["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("development", "holdout"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    args = p.parse_args()
    result = aggregate(args.shards, args.phase)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"cal096_{args.phase}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.out / f"CAL096_{args.phase.upper()}_REPORT.md").write_text(
        markdown(result), encoding="utf-8")
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
