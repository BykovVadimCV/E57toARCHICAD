"""CAL-03: compare measured real/synthetic vertical surface residuals with CAL-02.

A diagnostic *sensitivity* study. Residuals include registration, physical wall
roughness, glancing angle and material. Neither the best matching candidate nor
any quantile is an estimate of intrinsic scanner ranging noise.

Runs synthetic data in-memory, never writes or uploads raw scan coordinates.
The native synthetic XYZ are transformed by their *written* station poses,
matching how downstream consumers see a registered scan, before passing them to
CAL-02's SAME priority reservoir and robust local plane evaluator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np

from tools.calibrate_e57_surfaces import PointReservoir, assess_wall_patches

VERSION = 'cal03-registered-synthetic-vs-real-plane-residuals-v1'
REFERENCE_NAME = 'configs/calibration_real_e57_surfaces.json'
METRICS = ('observed_plane_mad_mm', 'observed_plane_rms_mm')


def real_reference(path: Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding='utf8'))
    if doc.get('format') != 'real_e57_local_surface_reference_v1':
        raise ValueError('Invalid CAL-02 reference version')
    source = doc.get('scans', [])
    if len(source) < 2 or any(int(x['selected_patches']) < 8 for x in source):
        raise ValueError('At least two real examples with multiple wall patches are required')
    if any(x.get('actual_station_count') != 'not_identifiable' for x in source):
        raise ValueError('Unexpected CAL-02 station provenance contract')
    if any(x.get('patch_mad_mm', {}).get('median') is None for x in source):
        raise ValueError('Missing real planar residuals')
    return doc


def sample_registered_returns(scans, poses, max_sample=250000, seed=42):
    """Uniform sample of valid XYZ after written pose, identical CAL-02 reservoir.

    Flattening per-station data before sampling prevents each station receiving
    an artificially equal quota. No model labels or GT masks are consulted.
    """
    if len(scans) != len(poses):
        raise ValueError('Scans/poses mismatch')
    reservoir = PointReservoir(max_sample, seed=seed)
    for scan, (rotation, translation) in zip(scans, poses):
        R = np.asarray(rotation, dtype=float)
        t = np.asarray(translation, dtype=float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError('Invalid registered pose')
        pts = np.asarray(scan.xyz_local)
        valid = np.asarray(scan.valid, dtype=bool)
        if pts.shape != (len(valid), 3):
            raise ValueError('Mismatched synthetic scan shape')
        for start in range(0, len(valid), 200000):
            xyz = pts[start:start+200000] @ R.T + t
            reservoir.add(xyz, valid[start:start+200000])
    return reservoir


def analyze_registered_scans(scans, poses, *, seed=42, max_sample=250000,
                             max_lines=5, max_patches=24) -> dict:
    reservoir = sample_registered_returns(scans, poses, max_sample=max_sample, seed=seed)
    surfaces = assess_wall_patches(reservoir.points, max_lines=max_lines,
                                   max_patches=max_patches, seed=seed)
    return {
        'source_type': 'registered_synthetic_returns_no_gt',
        'data3d_station_count': len(scans),
        'valid_cartesian_points': reservoir.valid,
        'reservoir_points': len(reservoir.points),
        'surfaces': surfaces,
        'warning': 'Registered synthetic coordinates, not device intrinsic noise',
    }


def variant_config(base, name):
    cfg = copy.deepcopy(base)
    # Keep all settings constant apart from ONE factor per sensitivity variant.
    variants = {
        'baseline': {},
        'low_unevenness': {'realism.wall_unevenness_mm': (2.0, 4.0)},
        'high_unevenness': {'realism.wall_unevenness_mm': (12.0, 20.0)},
        'low_registration': {'effects.registration_sigma_mm': (0.5, 1.0)},
        'high_registration': {'effects.registration_sigma_mm': (7.0, 11.0)},
    }
    if name not in variants:
        raise ValueError('Unknown sensitivity variant: ' + name)
    for path, value in variants[name].items():
        parent, field = path.split('.')
        setattr(getattr(cfg, parent), field, value)
    return cfg, variants[name]


def prepare_base_config(config_path):
    from simscan.config import load_config
    cfg = load_config(config_path)
    cfg.layout.source = 'simscan'
    cfg.layout.rooms = (1, 1)
    cfg.layout.footprint_x_m = (5.0, 5.0)
    cfg.layout.footprint_y_m = (4.5, 4.5)
    cfg.layout.p_duplex = 0.0
    cfg.layout.p_l_shape = 0.0
    cfg.scanner.angular_step_deg = 0.1
    cfg.scanner.stations_per_m2 = 0.09
    cfg.scanner.max_stations_per_room = 2
    cfg.scanner.station_random_fraction = 0.0
    cfg.scanner.p_limited_ceiling_view = 0.0
    cfg.export.preview = False
    cfg.export.debug = False
    cfg.export.write_mesh = False
    cfg.export.write_merged = False
    cfg.export.unet_mask = False
    cfg.export.free_space_input = False
    return cfg


def compare(reference, measurements: list[dict]) -> dict:
    """Compare *scene* quantiles; real source has only two non-random sites.

    Distances are descriptive, NOT an optimization objective that authorizes
    scanner tuning. MAD real q10/q50/q90 is biased toward densest wall patches.
    """
    real = reference['scans']
    real_meds = [float(x['patch_mad_mm']['median']) for x in real]
    grouped = {}
    for row in measurements:
        grouped.setdefault(row['variant'], []).append(row)
    comparison = {}
    for name, rows in sorted(grouped.items()):
        valid = [x for x in rows if x['surfaces']['summary'].get('n_patches', 0) >= 8]
        if valid:
            mids = [float(x['surfaces']['summary']['median_observed_plane_mad_mm']) for x in valid]
            rms = [float(x['surfaces']['summary']['median_observed_plane_rms_mm']) for x in valid]
            # Raw quantiles of patch estimates (same estimator, same max-24 policy).
            patches = [float(p['observed_plane_mad_mm']) for x in valid for p in x['surfaces']['patches']]
            synthetic_quantiles = {k: round(float(np.percentile(patches, q)), 4)
                                   for k, q in [('p10', 10), ('median', 50), ('p90', 90)]}
            comparison[name] = {
                'valid_scenes': len(valid), 'attempted_scenes': len(rows),
                'per_scene_median_mad_mm': [round(x, 4) for x in mids],
                'per_scene_median_rms_mm': [round(x, 4) for x in rms],
                'median_mad_mm': round(float(np.median(mids)), 4),
                'median_rms_mm': round(float(np.median(rms)), 4),
                'per_patch_mad_mm': synthetic_quantiles,
                'real_site_mad_medians_mm': real_meds,
                # Descriptive deviation only, never a fitted physical noise.
                'median_mad_difference_from_real_low_mm': round(float(np.median(mids)) - min(real_meds), 4),
                'median_mad_difference_from_real_high_mm': round(float(np.median(mids)) - max(real_meds), 4),
            }
        else:
            comparison[name] = {'valid_scenes': 0, 'attempted_scenes': len(rows),
                                'warning': 'NO_ELIGIBLE_VERTICAL_PATCHES'}
    return {
        'format': VERSION,
        'real_reference': {'files': len(real), 'median_mad_mm_by_file': real_meds,
                           'site_heterogeneity_ratio': round(max(real_meds) / min(real_meds), 3),
                           'physical_station_count': 'unknown'},
        'comparisons': comparison,
        'all_scenes': measurements,
        'auto_update_production_config': False,
        'calibration_decision': 'DIAGNOSTIC_ONLY_NO_UNIQUE_SCANNER_PARAMETER_FIT',
        'limitations': [
            'Two real scenes only. Median over densest 24 candidate wall patches, not random building sample.',
            'Small one-room synthetic scenes cannot match real multiroom incidence, occlusion or number of stations.',
            'Plane residual combines wall material, geometry, registration, range noise and outliers.',
            'Changing registration error or roughness may alter return support and patch selection.',
            'The underlying real E57 files lost original per-station ray provenance.',
            'All comparisons reuse the exact CAL-02 estimator; no geometry ground truth is used.',
            'No candidate can establish physical calibration without matched wall patches and multiple seeds.',
        ],
    }


def report_md(doc):
    lines = ['# CAL-03: сравнение реальных и синтетических стен', '',
             'Один измеритель CAL-02, одинаковые параметры отбора и MAD/RMS.',
             'Данные измерений, а не подтверждение характеристик конкретного сканера.', '',
             '| Группа | Сцен с патчами | MAD медиана, мм | RMS медиана, мм | MAD p10/p50/p90, мм |',
             '|---|---:|---:|---:|---|']
    for k, vals in doc['comparisons'].items():
        if not vals['valid_scenes']:
            lines.append(f'| {k} | 0/{vals["attempted_scenes"]} | н/д | н/д | н/д |')
        else:
            q = vals['per_patch_mad_mm']
            lines.append(f'| {k} | {vals["valid_scenes"]}/{vals["attempted_scenes"]} | '
                         f'{vals["median_mad_mm"]:.2f} | {vals["median_rms_mm"]:.2f} | '
                         f'{q["p10"]:.2f}/{q["median"]:.2f}/{q["p90"]:.2f} |')
    lines += ['', 'Реальные E57: медианы MAD по двум объектам: ' + ', '.join(
        f'{v:.2f} мм' for v in doc['real_reference']['median_mad_mm_by_file']) + '.',
        '', '**Решение:** параметры `customer_like.yaml` автоматически не изменять.',
        'Это эксперимент чувствительности, а не идентификация шума дальномера.', '',
        '## Ограничения', '']
    lines.extend('- ' + x for x in doc['limitations'])
    return '\n'.join(lines) + '\n'


def run(*, out: Path, reference: Path, config: Path, seeds: list[int], variants: list[str],
        max_sample: int = 250000, max_lines: int = 5, max_patches: int = 24) -> dict:
    from simscan.generate import generate_scene
    if not seeds or not variants or len(set(seeds)) != len(seeds):
        raise ValueError('Expected distinct seeds and at least one variant')
    real = real_reference(reference)
    base = prepare_base_config(config)
    out.mkdir(parents=True, exist_ok=True)
    measurements = []
    for seed in seeds:
        for name in variants:
            cfg, modified = variant_config(base, name)
            work = out / '_scratch' / f'{name}_{seed}'
            print(f'CAL03: seed={seed} variant={name}', flush=True)
            try:
                generated = generate_scene(cfg, work, seed=seed, index=0,
                                           write_scan=False, training_only=True)
                scans, poses = generated.pop('_scans')
                measured = analyze_registered_scans(scans, poses, max_sample=max_sample,
                                                    seed=42, max_lines=max_lines,
                                                    max_patches=max_patches)
                del scans, poses
                measured.update({'seed': seed, 'variant': name, 'modified_parameter': modified,
                                 'generated_station_count': generated['stations'],
                                 'generated_valid_returns': generated['points_valid'],
                                 'room_count': generated['rooms'],
                                 'angle_deg': cfg.scanner.angular_step_deg})
                measurements.append(measured)
                print(f'  patches={measured["surfaces"]["summary"]["n_patches"]} '
                      f'MAD={measured["surfaces"]["summary"]["median_observed_plane_mad_mm"]}mm', flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)
    report = compare(real, measurements)
    (out / 'cal03_comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    (out / 'CAL03_REPORT.md').write_text(report_md(report), encoding='utf8')
    shutil.rmtree(out / '_scratch', ignore_errors=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--real-reference', type=Path, default=Path(REFERENCE_NAME))
    ap.add_argument('--config', type=Path, default=Path('configs/customer_like.yaml'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[42111, 42112])
    ap.add_argument('--variants', nargs='+', default=['baseline','low_unevenness',
                                                     'high_unevenness','low_registration','high_registration'])
    ap.add_argument('--max-sample', type=int, default=250000)
    ap.add_argument('--max-patches', type=int, default=24)
    a = ap.parse_args()
    report = run(out=a.out, reference=a.real_reference, config=a.config,
                 seeds=a.seeds, variants=a.variants, max_sample=a.max_sample,
                 max_patches=a.max_patches)
    print(report_md(report), flush=True)


if __name__ == '__main__':
    main()
