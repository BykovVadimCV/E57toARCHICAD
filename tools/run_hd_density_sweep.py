"""Generate same single-room scene under three angular scan sampling rates.

THIS IS A SENSITIVITY SWEEP, not proof that a scanner uses a particular angle.
All candidates use same procedural scene seed, two maximum stations, no thinning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simscan.config import load_config
from simscan.makedata import make_dataset
from calibrate_hd_density import compare, from_synthetic_roots, make_report


def sweep(root: Path, reference: Path, angles: list[float], seed: int = 42111):
    if not angles or any(not 0.04 <= a <= 1.0 for a in angles):
        raise ValueError('All angular steps must be in the 0.04..1 degree range')
    root.mkdir(parents=True, exist_ok=True)
    settings = []
    for angle in angles:
        cfg = load_config('configs/customer_like.yaml')
        cfg.layout.source = 'simscan'
        cfg.layout.rooms = (1, 1)
        cfg.layout.footprint_x_m = (5.0, 5.0)
        cfg.layout.footprint_y_m = (4.5, 4.5)
        cfg.layout.p_duplex = 0.0
        cfg.layout.p_l_shape = 0.0
        cfg.scanner.angular_step_deg = float(angle)
        cfg.scanner.stations_per_m2 = 0.09
        cfg.scanner.max_stations_per_room = 2
        cfg.scanner.station_random_fraction = 0.0
        cfg.scanner.p_limited_ceiling_view = 0.0
        # No arbitrary point thinning, z shifts, or augmentation.
        target = root / ('angle_' + str(angle).replace('.', 'p'))
        print(f'GENERATE {target.name}, angle={angle}', flush=True)
        result = make_dataset(cfg, target, count=1, workers=1, seed=seed,
                              p_thin=0.0, p_shift=0.0, keep_scans=False,
                              variants=1, unseen_policy='ignore',
                              hd_masks=True, hd_pixel_mm=2.0,
                              hd_min_size=(2048, 1152),
                              log=lambda x: print(x, flush=True))
        if result['ok'] != 1 or result['errors']:
            raise RuntimeError(f'Failed to generate candidate {angle}: {result}')
        settings.append({'angle_deg': angle, 'name': target.name,
                         'station_target_per_m2': 0.09, 'max_stations_per_room': 2,
                         'result': result})
    real = json.loads(reference.read_text(encoding='utf-8'))['levels']
    fake = from_synthetic_roots([root / s['name'] for s in settings])
    report = compare(real, fake)
    report['candidate_settings'] = settings
    report['calibration_decision'] = 'NO_AUTO_OVERWRITE: density confounded by unknown real scanner stations and merged E57'
    report['same_scene_seed'] = seed
    (root/'density_calibration.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (root/'density_calibration.md').write_text(make_report(report),encoding='utf-8')
    print(make_report(report))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=Path('configs/calibration_real_e57_observed.json'))
    p.add_argument('--angles', type=float, nargs='+', default=[0.15,0.1,0.0667])
    p.add_argument('--seed', type=int, default=42111)
    args=p.parse_args()
    sweep(args.out, args.reference, args.angles, args.seed)


if __name__=='__main__': main()
