"""CAL-09.4 ray-level feature collection for frozen independent-seed holdout.

Canonical: 4 new development seeds + 4 entirely NEW holdout seeds, each scanned
with uniform/zone08/zone18 at 0.30 and 0.10m mesh, 0.1 degree rays (48 scans).

Use one seed per independent GitHub Actions runner. Aggregate eight JSON shards
with tools/aggregate_cal093.py; DO NOT select thresholds on holdout data.

Smoke (not canonical):
 PYTHONPATH=. python tools/validate_cal094_spatial.py \
 --out /tmp/cal093_smoke --seeds 48101 --variants uniform_2_4 zone18_fine35 \
 --resolutions 0.30 --angular-step-deg 0.50 --max-sample 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from tools.calibrate_factorial_surfaces import paired_layout_signature
from tools.calibrate_multiroom_surfaces import prepare_multroom_config
from tools.validate_cal09_two_scale_defects import variant_config
from tools.cal092_ray_observability import priority_sample
from tools.cal093_multiscale_features import matched_patch_scores, FORMAT
from tools.cal094_spatial_selection import (
    SEED_DEV, SEED_HOLDOUT, MESHES, VARIANTS,
)

RUN_FORMAT = "cal094-observed-feature-seed-shard-v1"
SEEDS = SEED_DEV + SEED_HOLDOUT


def frozen_config(config_path, seed, variant, resolution, angular_step_deg):
    """Match CAL-09 geometry/scanner recipe while admitting **new** seeds.

    CAL-09's original base_config disallows any new seed. This replacement
    fixes the same layout distribution, footprint split, all CAL09 defect
    parameters and 0.1 degree canonical scanner acquisition.
    """
    cfg = prepare_multroom_config(config_path)
    if seed % 100 >= 5:
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    cfg = variant_config(cfg, variant)
    cfg.realism.tessellation_m = resolution
    cfg.scanner.angular_step_deg = angular_step_deg
    return cfg


def one_case(config_path, seed, variant, resolution, out, *,
             max_sample=180000, angular_step_deg=.1):
    from simscan.generate import generate_scene
    cfg = frozen_config(config_path, seed, variant, resolution,
                        angular_step_deg)
    work = out / "_scratch" / f"{seed}_{variant}_{resolution:.2f}"
    try:
        result = generate_scene(
            cfg, work, seed=seed, index=0, write_scan=False,
            training_only=True, capture_truth_hits=True)
        scans, poses = result.pop("_scans")
        mesh = result.pop("_truth_mesh")
        doc = json.loads((work / "layout.json").read_text(encoding="utf-8"))
        signature = paired_layout_signature(doc)
        selected = priority_sample(scans, max_sample=max_sample, seed=42)
        del scans, poses
        scores = matched_patch_scores(selected, mesh, doc)
        return {
            "seed": seed, "variant": variant, "tessellation_m": resolution,
            "angular_step_deg": angular_step_deg,
            "layout_station_signature": signature,
            "stations": result["stations"],
            "rooms": result["rooms"],
            "total_direct_wall_rays": selected["total_direct_wall_returns"],
            "sampled_direct_wall_rays": selected["sampled_direct_wall_returns"],
            "patch_scores": scores,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run(out, config=Path("configs/customer_like.yaml"), *,
        seeds=SEEDS, variants=VARIANTS, resolutions=MESHES,
        angular_step_deg=.1, max_sample=180000):
    seeds = tuple(seeds)
    variants = tuple(variants)
    resolutions = tuple(resolutions)
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError("CAL-09.4 requires distinct declared development/holdout seeds")
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("Unsupported or repeated experiment variants")
    if not resolutions or len(set(resolutions)) != len(resolutions) or not set(resolutions) <= set(MESHES):
        raise ValueError("Unsupported or repeated mesh resolutions")
    if max_sample < 1000 or not .05 <= angular_step_deg <= .75:
        raise ValueError("Invalid sampling or angular step")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config_sha = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "version": RUN_FORMAT, "source_sha256": config_sha,
        "max_sample": max_sample, "angular_step_deg": angular_step_deg,
    }, sort_keys=True).encode()).hexdigest()
    cache = out / "_completed_cases"
    cache.mkdir(exist_ok=True)
    cases = []
    for seed in seeds:
        for resolution in resolutions:
            for variant in variants:
                key = f"{seed}_{variant}_{resolution:.2f}"
                path = cache / f"{key}.json"
                if path.exists():
                    saved = json.loads(path.read_text(encoding="utf-8"))
                    if saved.get("fingerprint") == fingerprint:
                        print(f"CAL094 resume {key}", flush=True)
                        cases.append(saved["case"])
                        continue
                print(f"CAL094 seed={seed} variant={variant} mesh={resolution:.2f} "
                      f"angular={angular_step_deg:.2f}", flush=True)
                row = one_case(config, seed, variant, resolution, out,
                               max_sample=max_sample, angular_step_deg=angular_step_deg)
                cases.append(row)
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps({"fingerprint": fingerprint, "case": row},
                                           ensure_ascii=False, allow_nan=False),
                                encoding="utf-8")
                temp.replace(path)
        (out / "cal094_progress.json").write_text(
            json.dumps({"format": RUN_FORMAT, "completed_cases": len(cases),
                        "expected_cases": len(seeds) * len(resolutions) * len(variants),
                        "source_config_sha256": config_sha, "run_fingerprint": fingerprint},
                       indent=2), encoding="utf-8")

    expected = {(s, v, r) for s in seeds for v in variants for r in resolutions}
    assert {(x["seed"], x["variant"], x["tessellation_m"]) for x in cases} == expected
    summary = {
        "format": RUN_FORMAT,
        "source_config_sha256": config_sha,
        "run_fingerprint": fingerprint,
        "seeds": list(seeds),
        "variants": list(variants),
        "resolutions_m": list(resolutions),
        "angular_step_deg": angular_step_deg,
        "max_sample": max_sample,
        "n_cases": len(cases),
        "score_schema": FORMAT,
        "cases": cases,
        "auto_update_production_config": False,
    }
    (out / "cal094_scores.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/customer_like.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--resolutions", type=float, nargs="+", default=list(MESHES))
    parser.add_argument("--angular-step-deg", type=float, default=.1)
    parser.add_argument("--max-sample", type=int, default=180000)
    args = parser.parse_args()
    result = run(args.out, args.config, seeds=args.seeds,
                 variants=args.variants, resolutions=args.resolutions,
                 angular_step_deg=args.angular_step_deg, max_sample=args.max_sample)
    print(f"CAL-09.4 feature cases completed: {result['n_cases']}", flush=True)


if __name__ == "__main__":
    main()
