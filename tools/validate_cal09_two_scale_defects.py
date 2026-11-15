"""CAL-09 paired two-scale wall-defect experiment (40 synthetic scans).

Run from repository root:
    PYTHONPATH=. python tools/validate_cal09_two_scale_defects.py --out /tmp/CAL09

2 x 2 factorial: coarse zones 0.8/1.8 m, independent fine texture
0.15/0.35 m; plus unchanged 2-4 mm uniform control. Never updates YAML.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.validate_cal08_tail_zones import (
    _brief_patches, patch_metrics, cluster_metrics, PRIMARY_PATCHES,
    EXTENDED_PATCHES, REAL_REF, PREDECLARED_GATES,
)
from tools.calibrate_factorial_surfaces import paired_layout_signature
from tools.calibrate_multiroom_surfaces import (
    prepare_multroom_config, sample_registered_with_stations, patch_evidence,
)
from tools.calibrate_real_synthetic_surfaces import real_reference

FORMAT = "cal09-two-scale-wall-defects-v1"
SEEDS = tuple(range(46101, 46109))
VARIANTS = {
    "uniform_2_4": (0., .8, 0.),
    "zone08_fine15": (.2, .8, .15),
    "zone08_fine35": (.2, .8, .35),
    "zone18_fine15": (.2, 1.8, .15),
    "zone18_fine35": (.2, 1.8, .35),
}


def variant_config(source, name):
    if name not in VARIANTS:
        raise ValueError(f"unknown CAL-09 variant: {name}")
    cfg = copy.deepcopy(source)
    probability, zone, fine = VARIANTS[name]
    cfg.realism.wall_unevenness_mm = (2., 4.)
    cfg.realism.wall_roughness_tail_probability = probability
    cfg.realism.wall_roughness_tail_sigma_mm = (8., 12.)
    cfg.realism.wall_roughness_tail_corr_m = zone
    cfg.realism.wall_roughness_tail_fine_corr_m = fine
    cfg.realism.wall_roughness_tail_transition_sigma = .15
    return cfg


def base_config(path, seed):
    if seed not in SEEDS:
        raise ValueError("CAL-09 requires previously unused frozen seeds")
    cfg = prepare_multroom_config(path)
    if seed in SEEDS[4:]:
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    if cfg.layout.rooms != (3, 3) or cfg.scanner.angular_step_deg != .1:
        raise ValueError("fixed CAL-09 acquisition was changed")
    return cfg


def assess(records, real_doc):
    if len(records) != len(SEEDS) * len(VARIANTS):
        raise ValueError("expected exactly 40 CAL-09 pairs")
    grouped = {}
    for r in records:
        key = (r["seed"], r["variant"])
        if r["seed"] not in SEEDS or r["variant"] not in VARIANTS or key in grouped:
            raise ValueError("invalid or duplicate seed/variant pair")
        grouped[key] = r
    real_medians = [float(s["patch_mad_mm"]["median"]) for s in real_doc["scans"]]
    real_p90s = [float(s["patch_mad_mm"]["p90"]) for s in real_doc["scans"]]
    if len(real_medians) != 2 or len(real_p90s) != 2:
        raise ValueError("two measured CAL-02 reference sites required")
    bracket = (min(real_medians) - .5, max(real_medians) + .5)
    ceiling = max(real_p90s) + 1.
    summary = {}
    for name in VARIANTS:
        pairs = []
        for seed in SEEDS:
            group = [grouped[(seed, variant)] for variant in VARIANTS]
            if len({r["layout_signature"] for r in group}) != 1 or not all(
                r["layout_station_matched"] for r in group
            ):
                raise ValueError("CAL-09 true layout/station mismatch")
            if len({(r["rooms"], r["stations"]) for r in group}) != 1:
                raise ValueError("CAL-09 station/room count changed")
            row = grouped[(seed, name)]
            baseline = grouped[(seed, "uniform_2_4")]
            a, b = patch_metrics(row["primary_patches"]), patch_metrics(baseline["primary_patches"])
            expanded = patch_metrics(row["extended_patches"])
            ratio = row["valid_returns"] / max(1, baseline["valid_returns"])
            eligible = (
                a["n_patches"] >= PREDECLARED_GATES["min_eligible_patches"]
                and b["n_patches"] >= PREDECLARED_GATES["min_eligible_patches"]
                and a["n_patches"] / max(1, b["n_patches"]) >= .75
                and .7 <= ratio <= 1.3
            )
            pairs.append({
                "seed": seed, "eligible": eligible,
                "cohort": "original_footprint" if seed in SEEDS[:4] else "larger_footprint",
                "median_mad_mm": a["median_mad_mm"],
                "p90_mad_mm": a["p90_mad_mm"], "p95_mad_mm": a["p95_mad_mm"],
                "median_shift_mm": a["median_mad_mm"] - b["median_mad_mm"],
                "p90_gain": a["p90_mad_mm"] / max(1e-9, b["p90_mad_mm"]),
                "top24_cluster": a["largest_tail_cluster"],
                "top72_cluster": expanded["largest_tail_cluster"],
                "valid_returns_ratio": round(ratio, 4),
            })
        valid = [p for p in pairs if p["eligible"]]
        med = lambda key: (round(float(np.median([p[key] for p in valid])), 4) if valid else None)
        gates = {
            "eight_valid_pairs": len(valid) == 8,
            "median_preserved_6_of_8": sum(abs(p["median_shift_mm"]) <= .5 for p in valid) >= 6,
            "median_in_real_bracket_6_of_8": sum(bracket[0] <= p["median_mad_mm"] <= bracket[1] for p in valid) >= 6,
            "p90_gain_6_of_8": sum(p["p90_gain"] >= 1.2 for p in valid) >= 6,
            "p90_below_ceiling_6_of_8": sum(p["p90_mad_mm"] <= ceiling for p in valid) >= 6,
            "p90_scene_median_in_bounds": bool(valid) and 5. <= med("p90_mad_mm") <= ceiling,
            "top24_clusters_4_of_8": sum(p["top24_cluster"] >= 2 for p in valid) >= 4,
            "top72_clusters_4_of_8": sum(p["top72_cluster"] >= 2 for p in valid) >= 4,
            "p90_gain_in_both_cohorts": all(
                sum(p["p90_gain"] >= 1.2 for p in valid if p["cohort"] == cohort) >= 3
                for cohort in ("original_footprint", "larger_footprint")
            ),
        }
        summary[name] = {
            "pairs": pairs, "median_mad_mm": med("median_mad_mm"),
            "median_p90_mm": med("p90_mad_mm"), "median_p95_mm": med("p95_mad_mm"),
            "median_p90_gain": med("p90_gain"),
            "clustered_top24_scenes": sum(p["top24_cluster"] >= 2 for p in valid),
            "clustered_top72_scenes": sum(p["top72_cluster"] >= 2 for p in valid),
            "gates": gates, "pass": all(gates.values()) if name != "uniform_2_4" else None,
        }
    passed = [n for n in VARIANTS if n != "uniform_2_4" and summary[n]["pass"]]
    return {
        "format": FORMAT,
        "design": {"seeds": list(SEEDS), "variants": VARIANTS,
                   "patches_primary": PRIMARY_PATCHES, "patches_extended": EXTENDED_PATCHES,
                   "real_reference_sites": 2, "threshold_patch_mad_mm": 6.,
                   "independent_zone_and_texture_scales": True},
        "variants": summary,
        "selected_candidate": passed[0] if passed else None,
        "decision": "EXPERIMENTAL_CANDIDATE" if passed else "NO_CAL09_VARIANT_PASSED",
        "auto_update_production_config": False,
        "quality_flags": [
            "ONLY_TWO_REAL_E57_SITES", "TOP24_DENSITY_SELECTION_BIAS",
            "PATCH_GRID_NOT_GROUND_TRUTH", "UNKNOWN_REAL_STATION_PROVENANCE",
            "OBSERVED_MAD_MIXES_SURFACE_AND_SCANNER_REGISTRATION",
        ],
    }


def report(result):
    lines = ["# CAL-09: двухмасштабная модель дефектов стен", "",
             "Масштаб зоны и корреляция мелкой текстуры варьируются независимо.",
             "Глобальные 2–4 мм, амплитуда 8–12 мм и вероятность 20% зафиксированы.", "",
             "| Вариант | MAD p50 | MAD p90 | p90/контроль | Кластеры top24 | Кластеры top72 | Итог |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for n, r in result["variants"].items():
        f = lambda v: "н/д" if v is None else f"{v:.3f}"
        lines.append(f"| {n} | {f(r['median_mad_mm'])} | {f(r['median_p90_mm'])} | "
                     f"{f(r['median_p90_gain'])} | {r['clustered_top24_scenes']}/8 | "
                     f"{r['clustered_top72_scenes']}/8 | {r['pass']} |")
    lines += ["", f"Решение: {result['decision']}.",
              "Production YAML, GT, оригинальные E57 и веса модели не изменяются.",
              "Топ-72 только диагностический. Это не калибровка физического шума сканера."]
    return "\n".join(lines) + "\n"


def run(out, config=Path("configs/customer_like.yaml"), reference=REAL_REF,
        max_sample=250000):
    from simscan.generate import generate_scene
    if max_sample < 1000:
        raise ValueError("max_sample must be >= 1000")
    real = real_reference(reference)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    records = []
    for seed in SEEDS:
        original = base_config(config, seed)
        signature = None
        for name in VARIANTS:
            cfg = variant_config(original, name)
            scratch = out / "_scratch" / str(seed) / name
            try:
                print(f"CAL09 seed={seed} variant={name}", flush=True)
                scene = generate_scene(cfg, scratch, seed=seed, index=0,
                                       write_scan=False, training_only=True)
                signature_now = paired_layout_signature(
                    json.loads((scratch / "layout.json").read_text(encoding="utf-8")))
                if signature is None:
                    signature = signature_now
                if signature != signature_now:
                    raise RuntimeError("CAL-09 geometry and station mismatch")
                scans, poses = scene.pop("_scans")
                reservoir = sample_registered_with_stations(
                    scans, poses, max_sample=max_sample, seed=42)
                evidence = patch_evidence(reservoir, seed=42, max_patches=EXTENDED_PATCHES,
                                          include_patch_grid=True)
                patches = _brief_patches(evidence)
                records.append({
                    "seed": seed, "variant": name, "layout_signature": signature_now,
                    "layout_station_matched": True,
                    "rooms": scene["rooms"], "stations": scene["stations"],
                    "valid_returns": reservoir.valid,
                    "primary_patches": patches[:PRIMARY_PATCHES],
                    "extended_patches": patches,
                })
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        (out / "cal09_progress.json").write_text(
            json.dumps({"complete_seed_groups": sum(r["variant"] == "zone18_fine35" for r in records),
                        "expected_seed_groups": len(SEEDS),
                        "aggregate_no_coordinates": [
                            {"seed": r["seed"], "variant": r["variant"],
                             "primary": patch_metrics(r["primary_patches"]),
                             "extended": patch_metrics(r["extended_patches"])}
                            for r in records]},
                       indent=2), encoding="utf-8")
    result = assess(records, real)
    result["source_config_sha256"] = config_sha
    (out / "cal09_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "CAL09_REPORT.md").write_text(report(result), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/customer_like.yaml"))
    parser.add_argument("--reference", type=Path, default=REAL_REF)
    parser.add_argument("--max-sample", type=int, default=250000)
    args = parser.parse_args()
    print(report(run(args.out, args.config, args.reference, args.max_sample)))


if __name__ == "__main__":
    main()
