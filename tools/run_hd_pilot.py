"""Generate a 10-scene native-HD observation pilot as two isolated cohorts.

The default ``diagnostic`` mode is a *cheap functionality audit*, not a
production-density calibration. ``full`` uses the customer-like acquisition
settings, and can require considerable memory and CPU time.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simscan.config import load_config
from simscan.makedata import make_dataset


def prepare_config(config: str, mode: str, passage_enriched: bool):
    cfg = load_config(config)
    if mode == 'diagnostic':
        cfg.layout.source = 'simscan'
        cfg.layout.rooms = (2, 4)
        cfg.layout.footprint_x_m = (6.0, 8.5)
        cfg.layout.footprint_y_m = (5.0, 7.0)
        cfg.layout.p_duplex = 0.0
        cfg.scanner.angular_step_deg = 0.45
        cfg.scanner.max_stations_per_room = min(2, cfg.scanner.max_stations_per_room)
    elif mode != 'full':
        raise ValueError('Mode must be diagnostic or full')
    # The enriched cohort is deliberately distinct, never merged into the
    # production distribution, and never called a representative sample.
    if passage_enriched:
        cfg.layout.p_passage = 0.75
    return cfg


def run_pilot(out: Path, *, config: str, seed: int, mode: str,
              baseline_count: int = 8, enriched_count: int = 2,
              workers: int = 1) -> dict:
    if (baseline_count < 1 or enriched_count < 1 or workers < 1):
        raise ValueError('Each cohort must contain at least 1 scene')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cohorts = [
        ('baseline', baseline_count, 0, False),
        ('passage_enriched', enriched_count, baseline_count, True),
    ]
    (out / 'pilot_setup.json').write_text(json.dumps({
        'mode': mode, 'seed': seed, 'baseline_count': baseline_count,
        'enriched_count': enriched_count,
        'diagnostic_only': mode == 'diagnostic',
        'acquisition': 'scanner 0.45 degree; 2-4 rooms; single-level' if mode == 'diagnostic'
                       else 'customer-like 0.1 degree; original room/duplex distributions',
        'source': 'configs/customer_like.yaml',
        'policy': 'No GT-only positives; no invalid ray background inference.',
    }, indent=2), encoding='utf-8')
    summaries = []
    for name, count, start, enriched in cohorts:
        cfg = prepare_config(config, mode, enriched)
        print(f'COHORT {name}: {count} scenes; start={start}; full={mode == "full"}', flush=True)
        result = make_dataset(
            cfg=cfg, out_root=out / name, count=count, seed=seed,
            start=start, workers=workers, keep_scans=False, variants=1,
            p_thin=0.3, p_shift=0.5, unseen_policy='ignore',
            hd_masks=True, hd_pixel_mm=2., hd_min_size=(2048,1152),
            log=lambda msg: print(msg, flush=True))
        summaries.append({'name': name, **result})
        if result['errors']:
            raise RuntimeError(f'Incomplete pilot cohort {name}: {result["errors"]} error(s)')
    from audit_hd_observation_pilot import audit_roots, save_report
    report, rows = audit_roots([out/'baseline', out/'passage_enriched'],
                              expected_scenes=baseline_count + enriched_count)
    save_report(report, rows, out / 'reports')
    return {'cohorts': summaries, 'summary': report}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--config', default='configs/customer_like.yaml')
    p.add_argument('--seed', type=int, default=31111)
    p.add_argument('--mode', choices=['diagnostic', 'full'], default='diagnostic')
    p.add_argument('--baseline-count', type=int, default=8)
    p.add_argument('--enriched-count', type=int, default=2)
    p.add_argument('--workers', type=int, default=1)
    a=p.parse_args()
    result=run_pilot(a.out, config=a.config, seed=a.seed, mode=a.mode,
                     baseline_count=a.baseline_count,
                     enriched_count=a.enriched_count, workers=a.workers)
    print('PILOT COMPLETE:', json.dumps({
        'scene_count': result['summary']['scene_count'],
        'level_count': result['summary']['level_count'],
        'warnings': result['summary']['warnings'],
    }, indent=2))

if __name__ == '__main__':
    main()
