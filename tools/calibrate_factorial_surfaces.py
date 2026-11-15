"""CAL-05: paired, one-factor-at-a-time synthetic wall residual experiment.

Runs a fixed layout and scanner placement for each seed while independently
changing roughness, registration translation sigma or range noise. Uses the
exact CAL-02 fitter and the CAL-04 station/ray stratifier. It does not fit
an instrument or update the production YAML. Raw XYZ never enter reports.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.calibrate_real_synthetic_surfaces import real_reference
from tools.calibrate_multiroom_surfaces import (
    prepare_multroom_config, sample_registered_with_stations, patch_evidence,
    _group, _median,
)

VERSION = 'cal05-paired-one-factor-multiroom-v1'
REAL_REF = Path('configs/calibration_real_e57_surfaces.json')
DEFAULT_NAMES = ('baseline', 'roughness_low', 'roughness_high',
                 'registration_low', 'registration_high',
                 'range_noise_low', 'range_noise_high')
VARIANTS = {
    'baseline': {},
    'roughness_low': {'realism.wall_unevenness_mm': (2.0, 4.0)},
    'roughness_high': {'realism.wall_unevenness_mm': (12.0, 20.0)},
    'registration_low': {'effects.registration_sigma_mm': (0.5, 1.0)},
    'registration_high': {'effects.registration_sigma_mm': (7.0, 11.0)},
    'range_noise_low': {'effects.range_sigma_mm': (0.10, 0.20)},
    'range_noise_high': {'effects.range_sigma_mm': (1.2, 2.0)},
}
MIN_PATCHES = 8


def variant_config(base, name):
    """Change ONE range, leaving all other config keys (including angle) intact."""
    if name not in VARIANTS:
        raise ValueError(f'Unknown CAL-05 variant: {name}')
    cfg = copy.deepcopy(base)
    change = VARIANTS[name]
    if len(change) > 1:
        raise AssertionError('CAL-05 is a one-factor design')
    for path, val in change.items():
        part, field = path.split('.')
        setattr(getattr(cfg, part), field, val)
    return cfg, {k: list(v) for k, v in change.items()}


def paired_layout_signature(doc: dict) -> str:
    """Hash geometry/station *true* positions; never publish the coordinates.

    Written poses are deliberately omitted, since they must differ when
    registration error changes. No noisy return count or effect params enter.
    """
    stations = doc.get('stations', [])
    if not stations:
        raise ValueError('Missing station provenance in generated layout')
    selected = []
    for s in stations:
        if 'pose_true' not in s or 'position_layout_m' not in s:
            raise ValueError('Missing physical station pose')
        selected.append({k: s[k] for k in ('id', 'room_id', 'level',
                         'position_layout_m', 'heading_deg', 'tilt_deg', 'pose_true')})
    ground = {k: doc.get(k) for k in ('rooms', 'walls', 'openings')}
    if not ground['rooms']:
        raise ValueError('Missing room geometry')
    ground['stations'] = selected
    ground['layout_to_world'] = doc.get('meta', {}).get('layout_to_world')
    raw = json.dumps(ground, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def _brief_patches(evidence: dict):
    """Keep patch-level residuals and ray *summaries*, no raw XYZ or poses."""
    allowed = ('observed_plane_mad_mm', 'observed_plane_rms_mm',
               'median_range_m', 'median_incidence_deg',
               'range_bin', 'incidence_bin', 'coverage',
               'well_supported_stations', 'direct_return_fraction',
               'interstation_normal_offset_span_mm')
    return [{k: p[k] for k in allowed if k in p} for p in evidence['patches']]


def _pair_effect(by_seed):
    """Paired scene median delta, not matched-patch or causal sensor sigma."""
    base = {s: variants['baseline'] for s, variants in by_seed.items()
            if 'baseline' in variants and variants['baseline']['valid_for_comparison']}
    effects = {}
    for name in DEFAULT_NAMES[1:]:
        pairs = []
        for seed, baseline in sorted(base.items()):
            variant = by_seed[seed].get(name)
            if variant and variant['valid_for_comparison'] and variant['layout_station_matched']:
                delta = round(variant['median_mad_mm'] - baseline['median_mad_mm'], 4)
                pairs.append({'seed': seed, 'baseline_mad_mm': baseline['median_mad_mm'],
                              'variant_mad_mm': variant['median_mad_mm'],
                              'delta_mad_mm': delta,
                              'patches_baseline': baseline['n_patches'],
                              'patches_variant': variant['n_patches']})
        effects[name] = {
            'paired_scenes': len(pairs), 'pairs': pairs,
            'median_delta_mad_mm': _median([p['delta_mad_mm'] for p in pairs]),
            'positive_delta_count': sum(p['delta_mad_mm'] > 0 for p in pairs),
            'negative_delta_count': sum(p['delta_mad_mm'] < 0 for p in pairs),
            'adequate_exploratory_paired_scenes': len(pairs) >= 4,
        }
    return effects


def summarize(real_doc: dict, rows: list[dict], *, expected_seeds=()):
    if not rows:
        raise ValueError('No CAL-05 scenes')
    keys = [(r['seed'], r['variant']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate seed-variant rows')
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r['seed'], {})[r['variant']] = r
    effects = _pair_effect(by_seed)
    comparisons = {}
    for name in DEFAULT_NAMES:
        rr = [r for r in rows if r['variant'] == name]
        ok = [r for r in rr if r['valid_for_comparison'] and r['layout_station_matched']]
        patches = [{**p, 'seed': r['seed']} for r in ok for p in r['surface_patches']]
        comparisons[name] = {
            'attempted_scenes': len(rr), 'valid_paired_scenes': len(ok),
            'per_scene': [{'seed': r['seed'], 'stations': r['stations'],
                           'rooms': r['rooms'], 'n_patches': r['n_patches'],
                           'median_mad_mm': r['median_mad_mm'],
                           'median_rms_mm': r['median_rms_mm'],
                           'layout_station_matched': r['layout_station_matched']}
                          for r in sorted(rr, key=lambda x: x['seed'])],
            'median_of_scene_mads_mm': _median([r['median_mad_mm'] for r in ok]),
            'patches_total': len(patches),
            'coverage_groups': _group(patches, 'coverage'),
            'range_groups': _group(patches, 'range_bin'),
            'incidence_groups': _group(patches, 'incidence_bin'),
        }
    flags = ['ONLY_TWO_REAL_SITES_WITH_UNKNOWN_ORIGINAL_STATIONS',
             'CAL05_DOES_NOT_INFER_INTRINSIC_RANGE_OR_REGISTRATION_SIGMA',
             'PATCH_SELECTION_CAN_DIFFER_ACROSS_PAIRED_VARIANTS',
             'FAR_RANGE_AND_GRAZING_STRATA_REQUIRE_INDEPENDENT_REAL_EVIDENCE']
    if len(by_seed) < 4:
        flags.append('FEWER_THAN_FOUR_INDEPENDENT_SYNTHETIC_LAYOUTS')
    if any(not r['layout_station_matched'] for r in rows):
        flags.append('PAIRED_LAYOUT_OR_STATION_MISMATCH')
    if any(len(by_seed.get(seed, {})) != len(DEFAULT_NAMES) for seed in expected_seeds):
        flags.append('MISSING_FACTORIAL_RUNS')
    if any(r['n_patches'] < MIN_PATCHES for r in rows):
        flags.append('LOW_PLANAR_PATCH_SUPPORT_IN_SOME_SCENES')
    for factor in effects.values():
        if factor['paired_scenes'] < 4:
            flags.append('INSUFFICIENT_VALID_PAIRED_SCENES_FOR_SOME_FACTORS')
            break
    return {
        'format': VERSION, 'design': {'seeds': sorted(by_seed),
          'factors_one_at_a_time': {k: {p: list(v) for p, v in VARIANTS[k].items()} for k in DEFAULT_NAMES},
          'fixed_angular_step_deg': 0.1, 'same_layout_and_true_station_poses_guarded': True,
          'same_noisy_returns_and_matched_physical_wall_patches': False},
        'real': {'site_count': len(real_doc['scans']),
                 'site_median_mad_mm': [r['patch_mad_mm']['median'] for r in real_doc['scans']],
                 'original_ray_station_metadata': 'UNAVAILABLE'},
        'variants': comparisons, 'paired_effects': effects,
        'quality_flags': sorted(set(flags)),
        'calibration_decision': 'NO_AUTO_TUNING_PENDING_INDEPENDENT_HOLDOUT_AND_MORE_REAL_SITES',
        'auto_update_production_config': False,
        'limitations': [
            'One-factor-at-a-time controlled input ranges do not uniquely identify physical scanner components.',
            'Same true station poses and room geometry are checked; noisy return realization and patch selection may differ.',
            'Real sites have no original per-return station geometry, incidence or ranging metadata.',
            'The CAL-02 candidate patches are the highest-support local vertical walls, not a random architectural sample.',
            'Paired deltas compare per-scene estimates, not exactly corresponding physical patches.',
            'Only synthetic range/incidence/overlap strata can be computed; they cannot be fitted to real per-ray values.',
            'No production YAML change or new model training on this experiment alone.',
        ]
    }


def report_md(result: dict) -> str:
    lines = ['# CAL-05: независимое изменение трёх факторов', '',
             'Эксперимент на одинаковых планировках и истинных положениях станций по каждому seed.',
             'Сравнение зарегистрированных синтетических XYZ с реальными плоскостями CAL-02.',
             'Это исследование чувствительности, **не измерение паспортного шума сканера**.', '',
             '| Вариант | Парных сцен | Медианный MAD сцен, мм | Изменение относительно baseline, мм |',
             '|---|---:|---:|---:|']
    for name in DEFAULT_NAMES:
        x = result['variants'][name]
        e = result['paired_effects'].get(name)
        med = x['median_of_scene_mads_mm']
        d = (e['median_delta_mad_mm'] if e else None)
        lines.append(f"| {name} | {x['valid_paired_scenes']} | {f'{med:.3f}' if med is not None else 'н/д'} | "
                     f"{f'{d:+.3f}' if d is not None else 'н/д'} |")
    lines.extend(['', '### Реальные ориентиры',
                  ', '.join(f'{v:.3f} мм' for v in result['real']['site_median_mad_mm']) +
                  ' (медианы двух исходных E57; это не шум дальномера).',
                  '', '### Предупреждения и ограничения'])
    lines.extend('- ' + x for x in result['quality_flags'])
    lines.extend(['', 'Распределения по дальности, углу и перекрытию содержатся в JSON.',
                  'Недостаточные группы не используются для выбора диапазонов.',
                  '', '**Решение:** параметры production YAML и веса не менять.',
                  'До CAL-06 необходимы независимые сцены и проверка качества на реальных объектах.'])
    return '\n'.join(lines) + '\n'


def run(*, out: Path, config: Path = Path('configs/customer_like.yaml'),
        reference: Path = REAL_REF, seeds=(42121, 42122, 42123, 42124),
        max_sample=250000, max_patches=24):
    from simscan.generate import generate_scene
    if len(seeds) < 4 or len(set(seeds)) != len(seeds):
        raise ValueError('CAL-05 requires four distinct seed geometries')
    if max_sample < 1000 or max_patches < MIN_PATCHES:
        raise ValueError('Insufficient CAL-05 sampling or patch limit')
    real = real_reference(reference)
    base = prepare_multroom_config(config)
    if base.scanner.angular_step_deg != .1 or base.layout.rooms != (3, 3):
        raise ValueError('Cannot change calibrated scanner-step/multroom study footprint')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        paired_hash = None
        for name in DEFAULT_NAMES:
            cfg, changes = variant_config(base, name)
            scratch = out / '_scratch' / f'seed_{seed}' / name
            try:
                print(f'CAL05: seed={seed} variant={name}', flush=True)
                generated = generate_scene(cfg, scratch, seed=seed, index=0,
                                           write_scan=False, training_only=True)
                with (scratch / 'layout.json').open(encoding='utf-8') as f:
                    signature = paired_layout_signature(json.load(f))
                if paired_hash is None:
                    paired_hash = signature
                matched = signature == paired_hash
                if not matched:
                    # Avoid false inference when the supposed paired experiment
                    # actually varied the room shape or scanner locations.
                    raise RuntimeError(f'Paired scene/station signature changed: seed={seed}, variant={name}')
                scans, poses = generated.pop('_scans')
                reservoir = sample_registered_with_stations(
                    scans, poses, max_sample=max_sample, seed=42)
                evidence = patch_evidence(reservoir, seed=42, max_patches=max_patches)
                del scans, poses
                n_patches = evidence['summary']['n_patches']
                row = {
                    'seed': seed, 'variant': name, 'changed_input_range': changes,
                    'layout_station_matched': matched,
                    'rooms': generated['rooms'], 'stations': generated['stations'],
                    'valid_returns': reservoir.valid,
                    'sampling_fraction': round(reservoir.sampling_fraction, 6),
                    'n_patches': n_patches,
                    'valid_for_comparison': n_patches >= MIN_PATCHES,
                    'median_mad_mm': evidence['summary']['median_observed_plane_mad_mm'],
                    'median_rms_mm': evidence['summary']['median_observed_plane_rms_mm'],
                    'surface_patches': _brief_patches(evidence),
                }
                rows.append(row)
                print(f"  stations={row['stations']} patches={n_patches} MAD={row['median_mad_mm']}mm", flush=True)
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        # Allow committed partial summaries on successful completed full seeds.
        partial = summarize(real, rows, expected_seeds=seeds)
        (out / 'cal05_results.json').write_text(
            json.dumps(partial, ensure_ascii=False, indent=2), encoding='utf-8')
        (out / 'CAL05_REPORT.md').write_text(report_md(partial), encoding='utf-8')
    return partial


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/customer_like.yaml'))
    p.add_argument('--reference', type=Path, default=REAL_REF)
    p.add_argument('--seeds', type=int, nargs='+', default=[42121,42122,42123,42124])
    p.add_argument('--max-sample', type=int, default=250000)
    p.add_argument('--max-patches', type=int, default=24)
    a = p.parse_args()
    r = run(out=a.out, config=a.config, reference=a.reference,
            seeds=a.seeds, max_sample=a.max_sample, max_patches=a.max_patches)
    print(report_md(r), flush=True)


if __name__ == '__main__':
    main()
