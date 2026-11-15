"""CAL-07: spatially clustered heavy-tail roughness on top of the CAL-06 2–4 mm base.

Three variants are paired on eight unseen multiroom seeds:
uniform 2–4 mm, 10% nominal tail zones, and 20% nominal tail zones.
Only the spatial roughness envelope changes. No physical scanner sigma is inferred.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.calibrate_factorial_surfaces import MIN_PATCHES, paired_layout_signature
from tools.calibrate_multiroom_surfaces import (
    prepare_multroom_config, sample_registered_with_stations, patch_evidence,
)
from tools.calibrate_real_synthetic_surfaces import real_reference

VERSION = "cal07-spatial-roughness-tail-v1"
REAL_REF = Path("configs/calibration_real_e57_surfaces.json")
NEW_SEEDS = tuple(range(44101, 44109))
USED_BEFORE = {
    42111, 42112, 42121, 42122, 42123, 42124,
    43101, 43102, 43103, 43104, 43105, 43106, 43107, 43108,
}
VARIANTS = ("uniform_2_4", "tail_10pct", "tail_20pct")
TAIL_PATCH_THRESHOLD_MM = 6.0
MIN_PATCH_RETENTION = 0.75
MIN_RETURN_RATIO = 0.70
MAX_RETURN_RATIO = 1.30
MEDIAN_PRESERVATION_TOLERANCE_MM = 0.50
MIN_SCENES_PER_GATE = 6
MIN_P90_GAIN_RATIO = 1.20
MIN_CLUSTERED_SCENES = 4
MIN_P90_TARGET_MM = 5.0
MAX_P90_TARGET_MARGIN_MM = 1.0


def validate_seeds(seeds):
    seeds = tuple(int(x) for x in seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("CAL-07 requires exactly eight distinct unseen seeds")
    if set(seeds) != set(NEW_SEEDS) or set(seeds) & USED_BEFORE:
        raise ValueError("CAL-07 seed set is frozen and disjoint from CAL-03–CAL-06")
    return seeds


def footprint_for_seed(seed):
    if seed not in NEW_SEEDS:
        raise ValueError("Unknown CAL-07 seed")
    return "original_footprint" if seed in NEW_SEEDS[:4] else "larger_footprint"


def fixed_base_config(config_path, seed):
    cfg = prepare_multroom_config(config_path)
    cfg.realism.wall_unevenness_mm = (2.0, 4.0)
    cfg.realism.wall_roughness_tail_probability = 0.0
    cfg.realism.wall_roughness_tail_sigma_mm = (8.0, 12.0)
    cfg.realism.wall_roughness_tail_corr_m = 0.8
    cfg.realism.wall_roughness_tail_transition_sigma = 0.15
    if seed in NEW_SEEDS[4:]:
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    if cfg.layout.rooms != (3, 3) or cfg.scanner.angular_step_deg != 0.1:
        raise ValueError("CAL-07 requires three rooms and a fixed 0.1 degree scanner grid")
    return cfg


def variant_config(base, name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown CAL-07 variant: {name}")
    cfg = copy.deepcopy(base)
    probability = {"uniform_2_4": 0.0, "tail_10pct": 0.10, "tail_20pct": 0.20}[name]
    cfg.realism.wall_roughness_tail_probability = probability
    cfg.realism.wall_roughness_tail_sigma_mm = (8.0, 12.0)
    cfg.realism.wall_roughness_tail_corr_m = 0.8
    cfg.realism.wall_roughness_tail_transition_sigma = 0.15
    return cfg, {
        "realism.wall_unevenness_mm": [2.0, 4.0],
        "realism.wall_roughness_tail_probability": probability,
        "realism.wall_roughness_tail_sigma_mm": [8.0, 12.0],
        "realism.wall_roughness_tail_corr_m": 0.8,
        "realism.wall_roughness_tail_transition_sigma": 0.15,
    }


def _median(values):
    return round(float(np.median(values)), 4) if values else None


def _quantile(values, q):
    return round(float(np.quantile(values, q)), 4) if values else None


def _cluster_stats(patches, threshold=TAIL_PATCH_THRESHOLD_MM):
    """Connected components in local wall-patch grid indices; no world coordinates."""
    cells = set()
    for p in patches:
        value = p.get("observed_plane_mad_mm")
        if value is None or float(value) < threshold:
            continue
        if not all(k in p for k in ("wall_candidate_index", "patch_col", "patch_row")):
            continue
        cells.add((int(p["wall_candidate_index"]), int(p["patch_col"]), int(p["patch_row"])))
    remaining = set(cells)
    largest = 0
    while remaining:
        start = remaining.pop()
        stack = [start]
        size = 0
        while stack:
            wall, col, row = stack.pop()
            size += 1
            for nxt in ((wall, col - 1, row), (wall, col + 1, row),
                        (wall, col, row - 1), (wall, col, row + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    neighbor_pairs = sum(
        int((wall, col + 1, row) in cells) + int((wall, col, row + 1) in cells)
        for wall, col, row in cells
    )
    return {
        "tail_patch_count": len(cells),
        "tail_patch_fraction": round(len(cells) / max(len(patches), 1), 4),
        "largest_tail_cluster": largest,
        "tail_neighbor_pairs": int(neighbor_pairs),
    }


def patch_metrics(patches):
    values = [float(p["observed_plane_mad_mm"]) for p in patches
              if p.get("observed_plane_mad_mm") is not None
              and np.isfinite(p["observed_plane_mad_mm"])]
    if not values:
        return {"n_patches": 0, "median_mad_mm": None, "p90_mad_mm": None,
                "p95_mad_mm": None, **_cluster_stats(patches)}
    return {
        "n_patches": len(values),
        "median_mad_mm": _quantile(values, 0.50),
        "p90_mad_mm": _quantile(values, 0.90),
        "p95_mad_mm": _quantile(values, 0.95),
        **_cluster_stats(patches),
    }


def _brief_patches(evidence):
    allowed = (
        "observed_plane_mad_mm", "observed_plane_rms_mm",
        "wall_candidate_index", "patch_col", "patch_row",
        "inlier_points", "inlier_fraction",
    )
    return [{k: p[k] for k in allowed if k in p} for p in evidence["patches"]]


def assess(real_doc, records, seeds=NEW_SEEDS):
    seeds = validate_seeds(seeds)
    if len(records) != len(seeds) * len(VARIANTS):
        raise ValueError("CAL-07 requires exactly 24 completed scene-variant records")
    site_medians = [float(x["patch_mad_mm"]["median"]) for x in real_doc["scans"]]
    site_p90 = [float(x["patch_mad_mm"]["p90"]) for x in real_doc["scans"]]
    if len(site_medians) != 2 or len(site_p90) != 2:
        raise ValueError("CAL-07 expects the two audited CAL-02 E57 references")
    if not all(np.isfinite(v) and v > 0 for v in site_medians + site_p90):
        raise ValueError("Invalid real-site CAL-02 quantiles")

    by_seed = {}
    seen = set()
    for row in records:
        seed, variant = int(row["seed"]), row["variant"]
        if seed not in seeds or variant not in VARIANTS or (seed, variant) in seen:
            raise ValueError("Duplicate or unexpected CAL-07 seed/variant")
        seen.add((seed, variant))
        by_seed.setdefault(seed, {})[variant] = row
    if any(set(by_seed.get(seed, {})) != set(VARIANTS) for seed in seeds):
        raise ValueError("Missing CAL-07 variant for a frozen seed")

    real_median_low = min(site_medians) - 0.5
    real_median_high = max(site_medians) + 0.5
    real_p90_high = max(site_p90) + MAX_P90_TARGET_MARGIN_MM
    per_variant = {}
    for variant in VARIANTS:
        pairs = []
        for seed in seeds:
            group = by_seed[seed]
            base = group["uniform_2_4"]
            row = group[variant]
            if not base["layout_station_matched"] or not row["layout_station_matched"]:
                raise ValueError(f"Unmatched true layout/station signature for seed {seed}")
            if base["layout_signature"] != row["layout_signature"]:
                raise ValueError(f"Different true layout/stations for seed {seed}")
            if base["rooms"] != row["rooms"] or base["stations"] != row["stations"]:
                raise ValueError(f"Room/station count changed for seed {seed}")
            bm = patch_metrics(base["surface_patches"])
            vm = patch_metrics(row["surface_patches"])
            patch_ratio = vm["n_patches"] / max(bm["n_patches"], 1)
            return_ratio = row["valid_returns"] / max(base["valid_returns"], 1)
            eligible = (
                bm["n_patches"] >= MIN_PATCHES and vm["n_patches"] >= MIN_PATCHES
                and patch_ratio >= MIN_PATCH_RETENTION
                and MIN_RETURN_RATIO <= return_ratio <= MAX_RETURN_RATIO
                and bm["median_mad_mm"] is not None and vm["median_mad_mm"] is not None
                and bm["p90_mad_mm"] is not None and vm["p90_mad_mm"] is not None
            )
            gain = (vm["p90_mad_mm"] / max(bm["p90_mad_mm"], 1e-6)
                    if bm["p90_mad_mm"] is not None and vm["p90_mad_mm"] is not None else None)
            pairs.append({
                "seed": seed, "cohort": footprint_for_seed(seed),
                "rooms": row["rooms"], "stations": row["stations"],
                "eligible": bool(eligible),
                "n_patches_baseline": bm["n_patches"], "n_patches_variant": vm["n_patches"],
                "relative_patch_retention": round(patch_ratio, 4),
                "relative_valid_return_count": round(return_ratio, 4),
                "median_mad_baseline_mm": bm["median_mad_mm"],
                "median_mad_variant_mm": vm["median_mad_mm"],
                "median_delta_mm": round(vm["median_mad_mm"] - bm["median_mad_mm"], 4)
                    if eligible else None,
                "p90_baseline_mm": bm["p90_mad_mm"], "p90_variant_mm": vm["p90_mad_mm"],
                "p95_variant_mm": vm["p95_mad_mm"],
                "p90_gain_ratio": round(gain, 4) if gain is not None else None,
                "tail_patch_count": vm["tail_patch_count"],
                "tail_patch_fraction": vm["tail_patch_fraction"],
                "largest_tail_cluster": vm["largest_tail_cluster"],
                "tail_neighbor_pairs": vm["tail_neighbor_pairs"],
            })
        eligible = [x for x in pairs if x["eligible"]]
        medians = [x["median_mad_variant_mm"] for x in eligible]
        p90s = [x["p90_variant_mm"] for x in eligible]
        p95s = [x["p95_variant_mm"] for x in eligible if x["p95_variant_mm"] is not None]
        median_preserved = sum(abs(x["median_delta_mm"]) <= MEDIAN_PRESERVATION_TOLERANCE_MM
                               for x in eligible)
        median_in_real = sum(real_median_low <= x["median_mad_variant_mm"] <= real_median_high
                             for x in eligible)
        p90_up = sum(x["p90_gain_ratio"] is not None and x["p90_gain_ratio"] >= MIN_P90_GAIN_RATIO
                     for x in eligible)
        p90_below_ceiling = sum(x["p90_variant_mm"] is not None and x["p90_variant_mm"] <= real_p90_high
                                for x in eligible)
        clustered = sum(x["tail_patch_count"] >= 2 and x["largest_tail_cluster"] >= 2
                        for x in eligible)
        by_cohort = {
            cohort: sum(x["p90_gain_ratio"] is not None and x["p90_gain_ratio"] >= MIN_P90_GAIN_RATIO
                        for x in eligible if x["cohort"] == cohort)
            for cohort in ("original_footprint", "larger_footprint")
        }
        median_p90 = _median(p90s)
        gates = {
            "all_eight_pairs_eligible": len(eligible) == 8,
            "typical_median_preserved_in_six_of_eight": median_preserved >= MIN_SCENES_PER_GATE,
            "typical_median_inside_real_site_bracket_in_six_of_eight": median_in_real >= MIN_SCENES_PER_GATE,
            "p90_increases_at_least_20pct_in_six_of_eight": p90_up >= MIN_SCENES_PER_GATE,
            "p90_does_not_exceed_real_upper_site_plus_1mm_in_six_of_eight":
                p90_below_ceiling >= MIN_SCENES_PER_GATE,
            "median_p90_inside_5_to_real_upper_plus_1mm":
                median_p90 is not None and MIN_P90_TARGET_MM <= median_p90 <= real_p90_high,
            "clustered_tail_in_at_least_four_of_eight": clustered >= MIN_CLUSTERED_SCENES,
            "both_footprint_cohorts_have_p90_gain_in_three_of_four":
                all(v >= 3 for v in by_cohort.values()),
        }
        per_variant[variant] = {
            "paired_scenes": pairs, "eligible_scenes": len(eligible),
            "median_scene_mad_mm": _median(medians),
            "median_scene_p90_mm": median_p90,
            "median_scene_p95_mm": _median(p95s),
            "median_p90_gain_ratio": _median([x["p90_gain_ratio"] for x in eligible
                                               if x["p90_gain_ratio"] is not None]),
            "median_tail_patch_fraction": _median([x["tail_patch_fraction"] for x in eligible]),
            "median_cluster_size": _median([x["largest_tail_cluster"] for x in eligible]),
            "median_preserved_scenes": median_preserved,
            "median_in_real_bracket_scenes": median_in_real,
            "p90_improved_scenes": p90_up,
            "p90_below_ceiling_scenes": p90_below_ceiling,
            "clustered_tail_scenes": clustered,
            "p90_gain_by_cohort": by_cohort,
            "gates": gates,
            "all_gates_pass": all(gates.values()) if variant != "uniform_2_4" else None,
        }

    passing = [v for v in ("tail_10pct", "tail_20pct") if per_variant[v]["all_gates_pass"]]
    selected = passing[0] if passing else None
    decision = {
        "tail_10pct": "PROVISIONAL_SPATIAL_TAIL_10PCT_SUPPORTED",
        "tail_20pct": "PROVISIONAL_SPATIAL_TAIL_20PCT_SUPPORTED",
    }.get(selected, "NO_SPATIAL_TAIL_VARIANT_PASSES")
    return {
        "format": VERSION,
        "design": {
            "holdout_seeds": list(seeds), "previous_seeds": sorted(USED_BEFORE),
            "disjoint_from_cal03_cal06": True, "fixed_angle_deg": 0.1,
            "variants": list(VARIANTS), "base_wall_unevenness_mm": [2.0, 4.0],
            "tail_sigma_mm": [8.0, 12.0], "tail_corr_m": 0.8,
            "tail_transition_sigma": 0.15, "tail_patch_threshold_mm": TAIL_PATCH_THRESHOLD_MM,
            "same_layout_and_true_stations_within_seed": True,
            "matched_physical_wall_patches": False,
        },
        "real_reference": {
            "site_count": 2, "site_median_mad_mm": site_medians,
            "site_patch_p90_mad_mm": site_p90,
            "median_bracket_mm": [round(real_median_low, 4), round(real_median_high, 4)],
            "p90_upper_site_plus_margin_mm": round(real_p90_high, 4),
            "physical_station_ray_metadata": "UNAVAILABLE",
        },
        "variants": per_variant,
        "selected_candidate": selected,
        "candidate_decision": decision,
        "auto_update_production_config": False,
        "quality_flags": [
            "ONLY_TWO_REAL_E57_SITES_WITH_UNKNOWN_SOURCE_STATIONS",
            "CAL02_REAL_PATCHES_ARE_TOP_24_DENSE_PATCHES_NOT_RANDOM_WALL_SAMPLE",
            "PLANE_RESIDUALS_CONFLATE_SURFACE_ROUGHNESS_RANGE_NOISE_AND_REGISTRATION",
            "CAL07_PATCH_GRID_IS_LOCAL_FITTED_WALL_INDEX_NOT_GLOBAL_WORLD_COORDINATES",
            "NO_MATCHED_PHYSICAL_PATCHES_OR_REAL_STATION_RAY_METADATA",
            "TAIL_THRESHOLD_AND_GATES_ARE_EXPLORATORY_ENGINEERING_CHOICES",
        ],
        "limitations": [
            "The CAL-02 real reference contains only two E57 sites and the 24 densest eligible patches per site.",
            "A local-plane MAD tail cannot be attributed uniquely to plaster, range noise, registration, edge contamination or material effects.",
            "The 10% and 20% tail probabilities describe a smooth synthetic spatial envelope, not a measured real-world defect prevalence.",
            "Local grid adjacency is descriptive within the fitted synthetic wall candidate; it is not matched to real wall coordinates.",
            "Passing this stage supports a synthetic heterogeneity option only; it does not calibrate intrinsic scanner sigma or authorize production config changes.",
        ],
    }


def report_md(doc):
    def fmt(value):
        return f"{value:.3f}" if value is not None else "н/д"
    lines = [
        "# CAL-07: пространственная неоднородность и тяжёлый хвост неровности",
        "",
        "Сравниваются три режима при неизменной базовой неровности стен 2–4 мм:",
        "однородный контроль, пространственный хвост 10% и пространственный хвост 20%.",
        "В каждой зоне амплитуда плавно повышается до 8–12 мм; границы не создают разрывов сетки.",
        "",
        "| Режим | Валидных пар | Медиана MAD, мм | Медиана p90 MAD, мм | Медианный прирост p90 | Доля tail-патчей | Кластер, медиана |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in VARIANTS:
        v = doc["variants"][name]
        vals = [str(v["eligible_scenes"]), fmt(v["median_scene_mad_mm"]),
                fmt(v["median_scene_p90_mm"]), fmt(v["median_p90_gain_ratio"]),
                fmt(v["median_tail_patch_fraction"]), fmt(v["median_cluster_size"])]
        lines.append("| " + name + " | " + " | ".join(vals) + " |")
    lines += ["", "## Парные результаты", ""]
    for name in ("tail_10pct", "tail_20pct"):
        v = doc["variants"][name]
        lines += [f"### {name}", "",
                  "| Seed | MAD база | MAD вариант | p90 база | p90 вариант | Retention патчей | Крупнейший tail-кластер |",
                  "|---:|---:|---:|---:|---:|---:|---:|"]
        for p in v["paired_scenes"]:
            lines.append(f"| {p['seed']} | {fmt(p['median_mad_baseline_mm'])} | "
                         f"{fmt(p['median_mad_variant_mm'])} | {fmt(p['p90_baseline_mm'])} | "
                         f"{fmt(p['p90_variant_mm'])} | {fmt(p['relative_patch_retention'])} | "
                         f"{p['largest_tail_cluster']} |")
        lines += ["", "Проверки:"]
        lines.extend(f"- {'PASS' if ok else 'FAIL'}: {key}" for key, ok in v["gates"].items())
        lines.append("")
    lines += [
        f"## Решение: **{doc['candidate_decision']}**", "",
        "customer_like.yaml, исходные E57 и веса модели не изменены.",
        "Физическая калибровка дальномера и регистрации не объявляется завершённой.",
        "", "## Ограничения",
    ]
    lines.extend("- " + x for x in doc["limitations"])
    return "\n".join(lines) + "\n"


def run(*, out: Path, config=Path("configs/customer_like.yaml"),
        reference=REAL_REF, seeds=NEW_SEEDS, max_sample=250000, max_patches=24):
    from simscan.generate import generate_scene
    seeds = validate_seeds(seeds)
    if max_sample < 1000 or max_patches < MIN_PATCHES:
        raise ValueError("CAL-07 sampling must retain at least eight local wall patches")
    real = real_reference(reference)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    config_sha = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    for seed in seeds:
        base = fixed_base_config(config, seed)
        expected_signature = None
        for variant in VARIANTS:
            cfg, changes = variant_config(base, variant)
            scratch = out / "_scratch" / f"seed_{seed}" / variant
            try:
                print(f"CAL07 seed={seed} variant={variant}", flush=True)
                generated = generate_scene(cfg, scratch, seed=seed, index=0,
                                           write_scan=False, training_only=True)
                with (scratch / "layout.json").open(encoding="utf-8") as f:
                    signature = paired_layout_signature(json.load(f))
                if expected_signature is None:
                    expected_signature = signature
                matched = signature == expected_signature
                if not matched:
                    raise RuntimeError(f"CAL-07 true layout/station mismatch: seed={seed}, variant={variant}")
                scans, poses = generated.pop("_scans")
                reservoir = sample_registered_with_stations(scans, poses,
                                                            max_sample=max_sample, seed=42)
                evidence = patch_evidence(reservoir, seed=42, max_patches=max_patches,
                                          include_patch_grid=True)
                patches = _brief_patches(evidence)
                metrics = patch_metrics(patches)
                row = {
                    "seed": seed, "cohort": footprint_for_seed(seed), "variant": variant,
                    "changed_parameters": changes, "layout_station_matched": matched,
                    "layout_signature": signature,
                    "rooms": generated["rooms"], "stations": generated["stations"],
                    "valid_returns": reservoir.valid,
                    "sampling_fraction": round(reservoir.sampling_fraction, 6),
                    **metrics, "surface_patches": patches,
                }
                rows.append(row)
                print(f"  stations={row['stations']} patches={row['n_patches']} "
                      f"median={row['median_mad_mm']} p90={row['p90_mad_mm']} "
                      f"tail={row['tail_patch_count']} cluster={row['largest_tail_cluster']}",
                      flush=True)
                del scans, poses, reservoir
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        (out / "cal07_partial_progress.json").write_text(json.dumps({
            "format": VERSION, "completed_records": len(rows), "expected_records": len(seeds) * 3,
            "records_no_coordinates": [
                {k: v for k, v in x.items() if k not in ("surface_patches", "layout_signature")}
                for x in rows],
            "source_config_sha256": config_sha,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    result = assess(real, rows, seeds=seeds)
    result["source_config_sha256"] = config_sha
    (out / "cal07_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    (out / "CAL07_REPORT.md").write_text(report_md(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/customer_like.yaml"))
    parser.add_argument("--reference", type=Path, default=REAL_REF)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(NEW_SEEDS))
    parser.add_argument("--max-sample", type=int, default=250000)
    parser.add_argument("--max-patches", type=int, default=24)
    args = parser.parse_args()
    result = run(out=args.out, config=args.config, reference=args.reference,
                 seeds=args.seeds, max_sample=args.max_sample, max_patches=args.max_patches)
    print(report_md(result), flush=True)


if __name__ == "__main__":
    main()
