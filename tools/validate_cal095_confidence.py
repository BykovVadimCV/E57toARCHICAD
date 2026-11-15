"""CAL-09.5: 24 fresh scanner scenes, direct ray / physical cell score export.

8 entirely new seeds (501xx dev, 511xx untouched holdout) x three wall
variants x 0.10m mesh. 0.1-degree scanner, fixed 180000 direct rays.
No wall-amplitude, production config, real E57 or raycaster changes.

The existing CAL-09.3 physical patch feature extractor is reused *without*
modification; only supervised spatial scoring is new in CAL-09.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tools.cal095_soft_confidence import SEEDS, VARIANTS, MESH_M
from tools.validate_cal094_spatial import one_case

FORMAT = "cal095-score-seed-shard-v1"


def run(out, config=Path("configs/customer_like.yaml"), *,
        seeds=SEEDS, variants=VARIANTS, angular_step_deg=.1,
        max_sample=180000):
    seeds, variants = tuple(seeds), tuple(variants)
    if (not seeds or len(set(seeds)) != len(seeds) or
            not set(seeds) <= set(SEEDS)):
        raise ValueError("Only distinct CAL-09.5 declared fresh seeds accepted")
    if not variants or len(set(variants)) != len(variants) or not set(variants) <= set(VARIANTS):
        raise ValueError("Duplicate or invalid CAL-09.5 geometry variant")
    if not .05 <= angular_step_deg <= .75 or max_sample < 1000:
        raise ValueError("Unsafe sampling settings")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = Path(config)
    sha = hashlib.sha256(config.read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        "format": FORMAT, "config_sha256": sha, "step": angular_step_deg,
        "max_sample": max_sample, "mesh_m": MESH_M,
    }, sort_keys=True).encode()).hexdigest()
    cache = out / "_completed_cases"
    cache.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        for variant in variants:
            cached = cache / f"{seed}_{variant}_{MESH_M:.2f}.json"
            if cached.exists():
                doc = json.loads(cached.read_text(encoding="utf-8"))
                if doc.get("fingerprint") == fingerprint:
                    print(f"CAL095 resume seed={seed} variant={variant}", flush=True)
                    rows.append(doc["case"])
                    continue
            print(f"CAL095 seed={seed} variant={variant} mesh={MESH_M} "
                  f"angular={angular_step_deg}", flush=True)
            case = one_case(config, seed, variant, MESH_M, out,
                            angular_step_deg=angular_step_deg,
                            max_sample=max_sample)
            rows.append(case)
            tmp = cached.with_suffix(".tmp")
            tmp.write_text(json.dumps({"fingerprint": fingerprint, "case": case},
                                      ensure_ascii=False, allow_nan=False),
                           encoding="utf-8")
            tmp.replace(cached)
        (out / "cal095_progress.json").write_text(
            json.dumps({"completed_scans": len(rows),
                        "expected_scans": len(seeds) * len(variants),
                        "completed_seeds": len({x["seed"] for x in rows}),
                        "source_config_sha256": sha,
                        "run_fingerprint": fingerprint}, indent=2),
            encoding="utf-8")
    summary = {
        "format": FORMAT, "seeds": list(seeds),
        "variants": list(variants), "mesh_m": MESH_M,
        "angular_step_deg": angular_step_deg,
        "max_sample": max_sample,
        "source_config_sha256": sha,
        "run_fingerprint": fingerprint,
        "n_scans": len(rows),
        "cases": rows,
        "auto_update_production_config": False,
    }
    (out / "cal095_scores.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--config", type=Path, default=Path("configs/customer_like.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--angular-step-deg", type=float, default=.1)
    p.add_argument("--max-sample", type=int, default=180000)
    a = p.parse_args()
    result = run(a.out, a.config, seeds=a.seeds,
                 variants=a.variants, angular_step_deg=a.angular_step_deg,
                 max_sample=a.max_sample)
    print(f"CAL095 complete scenes: {result['n_scans']}", flush=True)


if __name__ == "__main__":
    main()
