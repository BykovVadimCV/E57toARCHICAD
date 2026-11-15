"""CAL-09.6 blind XYZ wall-geometry experiment.

The detector sees only registered valid XYZ returns. Synthetic labels,
instances and layout walls are retained in a separate evaluator namespace
and are never passed to tools.cal096_blind_geometry.extract_wall_geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.cal096_blind_geometry import extract_wall_geometry
from tools.calibrate_factorial_surfaces import paired_layout_signature
from tools.calibrate_multiroom_surfaces import prepare_multroom_config
from tools.validate_cal09_two_scale_defects import variant_config

FORMAT = "cal096-blind-xyz-seed-shard-v1"
DEV_SEEDS = (60101, 60102, 60105, 60106)
HOLDOUT_SEEDS = (61101, 61102, 61105, 61106)
SEEDS = DEV_SEEDS + HOLDOUT_SEEDS
VARIANTS = ("uniform_2_4", "zone08_fine35", "zone18_fine35")
MESH_M = .10


def frozen_config(config_path, seed, variant, angular_step_deg):
    cfg = prepare_multroom_config(config_path)
    if seed % 100 >= 5:
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    cfg = variant_config(cfg, variant)
    cfg.realism.tessellation_m = MESH_M
    cfg.scanner.angular_step_deg = angular_step_deg
    return cfg


def _gt_wall_edges(walls, visible_wall_indices, ceiling_height):
    """Evaluation-only topology from layout axes; never available to detector."""
    tol, gap_max = .18, .25
    visible = sorted(set(int(x) for x in visible_wall_indices))
    edges = {}
    for pos, i in enumerate(visible):
        if i < 0 or i >= len(walls):
            continue
        a = walls[i]
        a0, a1 = np.asarray(a["p0"], float), np.asarray(a["p1"], float)
        da = a1 - a0
        la = float(np.linalg.norm(da))
        if la < 1e-6:
            continue
        ua = da / la
        na = np.array([-ua[1], ua[0]])
        za0 = float(a.get("base", 0.))
        za1 = za0 + float(a.get("height") or ceiling_height)
        for j in visible[pos + 1:]:
            if j < 0 or j >= len(walls):
                continue
            b = walls[j]
            b0, b1 = np.asarray(b["p0"], float), np.asarray(b["p1"], float)
            db = b1 - b0
            lb = float(np.linalg.norm(db))
            if lb < 1e-6:
                continue
            ub = db / lb
            nb = np.array([-ub[1], ub[0]])
            zb0 = float(b.get("base", 0.))
            zb1 = zb0 + float(b.get("height") or ceiling_height)
            if min(za1, zb1) - max(za0, zb0) < 1.6:
                continue
            dot = abs(float(ua @ ub))
            relation = None
            if dot >= np.cos(np.radians(5.)):
                offset = abs(float(na @ (b0 - a0)))
                if offset <= .10:
                    ib = sorted([float((b0-a0) @ ua), float((b1-a0) @ ua)])
                    gap = max(0., max(ib[0] - la, 0. - ib[1]))
                    if 1e-4 < gap <= gap_max:
                        relation = "collinear_gap"
            elif dot <= np.sin(np.radians(10.)):
                matrix = np.vstack((na, nb))
                if abs(float(np.linalg.det(matrix))) > 1e-8:
                    cross = np.linalg.solve(matrix, np.array([na @ a0, nb @ b0]))
                    sa, sb = float((cross-a0) @ ua), float((cross-b0) @ ub)
                    inside_a = -tol <= sa <= la + tol
                    inside_b = -tol <= sb <= lb + tol
                    if inside_a and inside_b:
                        end_a = min(abs(sa), abs(sa-la)) <= tol
                        end_b = min(abs(sb), abs(sb-lb)) <= tol
                        if end_a and end_b:
                            relation = "corner"
                        elif end_a != end_b:
                            relation = "tee"
            if relation:
                edges[tuple(sorted((i, j)))] = relation
    return edges


def evaluate_geometry(result, labels, instances, walls, ceiling_height):
    """Separate oracle evaluator; result was computed from XYZ alone."""
    from simscan.labels import LABEL_ID

    labels = np.asarray(labels)
    instances = np.asarray(instances)
    sampled_source = np.asarray(result["sampled_source_indices"], np.int64)
    sampled_labels = labels[sampled_source]
    sampled_instances = instances[sampled_source]
    gt_wall = sampled_labels == LABEL_ID["wall"]
    pred_wall = np.zeros(len(sampled_source), bool)
    candidate_segments = [s for s in result["segments"] if s["wall_candidate"]]
    segment_matches = {}
    visible_instances = set(int(v) for v in np.unique(sampled_instances[gt_wall]))
    visible_wall_indices = {v - 1 for v in visible_instances if 1 <= v <= len(walls)}
    matched_wall_indices = set()
    for segment in candidate_segments:
        ids = np.asarray(segment["point_indices"], np.int64)
        pred_wall[ids] = True
        if len(ids) == 0:
            continue
        wall_ids = sampled_instances[ids][sampled_labels[ids] == LABEL_ID["wall"]]
        wall_fraction = len(wall_ids) / len(ids)
        if len(wall_ids):
            unique, counts = np.unique(wall_ids, return_counts=True)
            dominant = int(unique[int(np.argmax(counts))])
            share = float(counts.max() / len(ids))
            wall_index = dominant - 1
        else:
            dominant, share, wall_index = None, 0., None
        matched = bool(wall_fraction >= .55 and share >= .50 and
                       wall_index is not None and 0 <= wall_index < len(walls))
        segment_matches[int(segment["id"])] = wall_index if matched else None
        if matched:
            matched_wall_indices.add(wall_index)
    tp = int(np.count_nonzero(pred_wall & gt_wall))
    fp = int(np.count_nonzero(pred_wall & ~gt_wall))
    fn = int(np.count_nonzero(~pred_wall & gt_wall))
    tn = int(np.count_nonzero(~pred_wall & ~gt_wall))
    ratio = lambda n, d: round(float(n)/d, 6) if d else None
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    fpr = ratio(fp, fp + tn)
    f1 = ratio(2*tp, 2*tp + fp + fn)

    pred_pairs, pred_types = {}, {}
    for edge in result["topology"]["edges"]:
        a = segment_matches.get(int(edge["a"]))
        b = segment_matches.get(int(edge["b"]))
        if a is None or b is None or a == b:
            continue
        pair = tuple(sorted((a, b)))
        pred_pairs[pair] = edge["type"]
        pred_types[pair] = edge["type"]
    gt_pairs = _gt_wall_edges(walls, visible_wall_indices, ceiling_height)
    true_pairs = set(pred_pairs) & set(gt_pairs)
    edge_precision = ratio(len(true_pairs), len(pred_pairs))
    edge_recall = ratio(len(true_pairs), len(gt_pairs))
    edge_f1 = ratio(2*len(true_pairs), len(pred_pairs) + len(gt_pairs))
    type_accuracy = ratio(
        sum(pred_types[p] == gt_pairs[p] for p in true_pairs), len(true_pairs))
    segment_precision = ratio(len(matched_wall_indices), len(candidate_segments))
    instance_recall = ratio(len(matched_wall_indices & visible_wall_indices),
                            len(visible_wall_indices))
    return {
        "point": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "false_positive_rate": fpr, "f1": f1,
        },
        "segments": {
            "candidate_count": len(candidate_segments),
            "matched_count": len(matched_wall_indices),
            "precision": segment_precision,
            "visible_wall_instances": len(visible_wall_indices),
            "matched_visible_wall_instances": len(matched_wall_indices & visible_wall_indices),
            "instance_recall": instance_recall,
        },
        "topology": {
            "predicted_edge_count": len(pred_pairs),
            "ground_truth_visible_edge_count": len(gt_pairs),
            "matched_edge_count": len(true_pairs),
            "edge_precision": edge_precision,
            "edge_recall": edge_recall,
            "edge_f1": edge_f1,
            "edge_type_accuracy": type_accuracy,
            "predicted_types": {
                key: sum(v == key for v in pred_types.values())
                for key in ("corner", "tee", "collinear_gap")
            },
        },
    }


def one_case(config_path, seed, variant, out, *, angular_step_deg=.1,
             max_points=200_000):
    from simscan.generate import generate_scene

    cfg = frozen_config(config_path, seed, variant, angular_step_deg)
    work = Path(out) / "_scratch" / f"{seed}_{variant}_{MESH_M:.2f}"
    try:
        generated = generate_scene(
            cfg, work, seed=seed, index=0, write_scan=False,
            training_only=True, capture_truth_hits=False)
        scans, poses = generated.pop("_scans")
        layout_doc = json.loads((work / "layout.json").read_text(encoding="utf-8"))
        xyz_parts, label_parts, instance_parts = [], [], []
        for scan, (rotation, translation) in zip(scans, poses):
            valid = np.asarray(scan.valid, bool) & np.isfinite(scan.xyz_local).all(axis=1)
            if not valid.any():
                continue
            # Every valid return enters the cloud, including mixed/virtual clutter.
            # Semantic labels are retained in a separate evaluator-only array.
            xyz_parts.append(np.asarray(scan.xyz_local[valid], float) @
                             np.asarray(rotation, float).T + np.asarray(translation, float))
            label_parts.append(np.asarray(scan.label[valid]))
            instance_parts.append(np.asarray(scan.instance[valid]))
        if not xyz_parts:
            raise ValueError("Synthetic scanner produced no valid XYZ returns")
        xyz = np.vstack(xyz_parts)
        labels = np.concatenate(label_parts)
        instances = np.concatenate(instance_parts)
        # Crucial information boundary: only XYZ crosses into the blind detector.
        blind = extract_wall_geometry(xyz)
        metrics = evaluate_geometry(
            blind, labels, instances, layout_doc["walls"],
            float(layout_doc["ceiling_height"]))
        signature = paired_layout_signature(layout_doc)
        compact_segments = [{
            k: v for k, v in s.items()
            if k not in ("point_indices", "source_point_indices")
        } for s in blind["segments"]]
        return {
            "seed": int(seed), "variant": variant, "mesh_m": MESH_M,
            "angular_step_deg": angular_step_deg,
            "layout_station_signature": signature,
            "input_valid_returns": int(len(xyz)),
            "sampled_xyz_points": int(blind["sampled_point_count"]),
            "detected_plane_segments": compact_segments,
            "predicted_topology": blind["topology"],
            "rejected_planes": blind["rejected_planes"],
            "metrics": metrics,
            "detector_inputs": blind["detector_inputs"],
            "detector_uses_ground_truth": blind["uses_ground_truth"],
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_seed(out, config=Path("configs/customer_like.yaml"), *,
             seed, variants=VARIANTS, angular_step_deg=.1,
             max_points=200_000):
    if seed not in SEEDS:
        raise ValueError("Only predeclared CAL-09.6 seeds are allowed")
    variants = tuple(variants)
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("Invalid or repeated geometry variants")
    if not .05 <= angular_step_deg <= .75 or max_points < 1000:
        raise ValueError("Invalid scanner/sample settings")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = Path(config)
    source_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "format": FORMAT, "source_sha256": source_sha,
        "angular_step_deg": angular_step_deg, "max_points": max_points,
        "mesh_m": MESH_M,
    }, sort_keys=True).encode()).hexdigest()
    cache = out / "_completed_cases"
    cache.mkdir(exist_ok=True)
    cases = []
    for variant in variants:
        path = cache / f"{seed}_{variant}_{MESH_M:.2f}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("fingerprint") == fingerprint:
                cases.append(saved["case"])
                print(f"CAL096 resume seed={seed} variant={variant}", flush=True)
                continue
        print(f"CAL096 seed={seed} variant={variant} mesh={MESH_M:.2f} "
              f"angular={angular_step_deg:.2f}", flush=True)
        case = one_case(config, seed, variant, out,
                        angular_step_deg=angular_step_deg, max_points=max_points)
        cases.append(case)
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"fingerprint": fingerprint, "case": case},
                                   ensure_ascii=False, allow_nan=False),
                        encoding="utf-8")
        temp.replace(path)
    summary = {
        "format": FORMAT, "seed": int(seed), "variants": list(variants),
        "mesh_m": MESH_M, "angular_step_deg": angular_step_deg,
        "max_points": max_points, "source_config_sha256": source_sha,
        "run_fingerprint": fingerprint, "n_cases": len(cases),
        "cases": cases, "auto_update_production_config": False,
    }
    (out / f"cal096_seed_{seed}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/customer_like.yaml"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--angular-step-deg", type=float, default=.1)
    parser.add_argument("--max-points", type=int, default=200_000)
    args = parser.parse_args()
    result = run_seed(args.out, args.config, seed=args.seed,
                      variants=args.variants,
                      angular_step_deg=args.angular_step_deg,
                      max_points=args.max_points)
    print(json.dumps({
        "seed": result["seed"], "scenes": result["n_cases"],
        "point_f1": round(float(np.mean([
            x["metrics"]["point"]["f1"] or 0. for x in result["cases"]])), 4),
        "wall_instance_recall": round(float(np.mean([
            x["metrics"]["segments"]["instance_recall"] or 0.
            for x in result["cases"]])), 4),
    }), flush=True)


if __name__ == "__main__":
    main()
