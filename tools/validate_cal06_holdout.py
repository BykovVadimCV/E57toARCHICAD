"""CAL-06: predeclared independent holdout of the CAL-05 roughness candidate.

Compare baseline [6,14] mm wall unevenness with [2,4] mm on eight entirely new
multiroom seeds. Hold geometry and true station poses fixed *within each pair*.
The same CAL-02 planar estimator is reused; no GT patch selection, station
coordinates, raw E57 samples, model weights, or production YAML are written.

This assesses external validation of an observed-plane metric, NOT a calibrated
instrument range sigma, true wall roughness or real-world model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.calibrate_factorial_surfaces import (
    MIN_PATCHES, paired_layout_signature, variant_config, _brief_patches,
)
from tools.calibrate_multiroom_surfaces import (
    prepare_multroom_config, sample_registered_with_stations, patch_evidence,
)
from tools.calibrate_real_synthetic_surfaces import real_reference

VERSION = 'cal06-independent-holdout-v1'
NEW_SEEDS = tuple(range(43101, 43109))
USED_BEFORE = {42111, 42112, 42121, 42122, 42123, 42124}
VARIANTS = ('baseline', 'roughness_low')
REAL_REF = Path('configs/calibration_real_e57_surfaces.json')

# Freeze before generating any holdout scene. An acceptance test is not a
# statistically validated confidence interval with only two real buildings.
PREDECLARED_GATE = {
    'min_eligible_paired_scenes': 8,
    'min_scenes_improved': 7,
    'min_median_paired_reduction_mm': 1.0,
    'min_candidate_to_baseline_patch_count': 0.75,
    'min_candidate_to_baseline_valid_return_count': 0.70,
    'max_candidate_to_baseline_valid_return_count': 1.30,
    'real_site_median_bracket_margin_mm': 0.50,
    'min_scenes_in_expanded_real_bracket': 6,
    'min_improved_per_footprint_cohort': 3,
}


def validate_seeds(seeds):
    seeds = tuple(int(x) for x in seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError('CAL-06 requires eight distinct untouched holdout seeds')
    if set(seeds) != set(NEW_SEEDS) or (set(seeds) & USED_BEFORE):
        raise ValueError('Holdout seed set is frozen, and must not overlap CAL-03/04/05')
    return seeds


def footprint_for_seed(seed):
    if seed not in NEW_SEEDS:
        raise ValueError('Unknown holdout seed')
    return 'original_footprint' if seed in NEW_SEEDS[:4] else 'larger_footprint'


def fixed_base_config(config_path, seed):
    cfg = prepare_multroom_config(config_path)
    if seed in NEW_SEEDS[4:]:
        # Held-out distribution shift: same 3 rooms but larger area.
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    if cfg.layout.rooms != (3, 3) or cfg.scanner.angular_step_deg != 0.1:
        raise ValueError('CAL-06 room and scanner-step contract broken')
    return cfg


def _median(vals):
    return round(float(np.median(vals)), 4) if vals else None


def _quantiles(vals):
    return ({k: round(float(np.quantile(vals, q)), 4)
             for k, q in (('p10', .1), ('median', .5), ('p90', .9))}
            if vals else None)


def assess(real_doc, records, seeds=NEW_SEEDS):
    """Pure holdout decision, using locked thresholds and per-seed pairs only."""
    seeds = validate_seeds(seeds)
    site = [float(r['patch_mad_mm']['median']) for r in real_doc['scans']]
    if len(site) != 2 or any(not np.isfinite(v) or v <= 0 for v in site):
        raise ValueError('Expected two finite real-site CAL-02 references')
    if len(records) != 2 * len(seeds):
        raise ValueError('The holdout must contain exactly 16 completed records')
    seen = set()
    by_seed = {}
    for r in records:
        seed, variant = int(r['seed']), r['variant']
        if seed not in seeds or variant not in VARIANTS or (seed, variant) in seen:
            raise ValueError('Duplicate/unexpected seed or variant in holdout')
        seen.add((seed, variant))
        by_seed.setdefault(seed, {})[variant] = r
    if any(set(by_seed.get(seed, {})) != set(VARIANTS) for seed in seeds):
        raise ValueError('Missing a required paired holdout result')

    paired = []
    for seed in seeds:
        a, b = by_seed[seed]['baseline'], by_seed[seed]['roughness_low']
        if not (a['layout_station_matched'] and b['layout_station_matched']
                and a['layout_signature'] == b['layout_signature']):
            raise ValueError(f'Unmatched physical room/station pair for seed {seed}')
        if a['rooms'] != b['rooms'] or a['stations'] != b['stations']:
            raise ValueError(f'Room/station count changed within seed {seed}')
        base, candidate = a['median_mad_mm'], b['median_mad_mm']
        finite_mad = (base is not None and candidate is not None
                      and np.isfinite(base) and np.isfinite(candidate)
                      and base > 0 and candidate > 0)
        patch_ratio = b['n_patches'] / max(a['n_patches'], 1)
        return_ratio = b['valid_returns'] / max(a['valid_returns'], 1)
        quality = (a['n_patches'] >= MIN_PATCHES and b['n_patches'] >= MIN_PATCHES
                   and patch_ratio >= PREDECLARED_GATE['min_candidate_to_baseline_patch_count']
                   and PREDECLARED_GATE['min_candidate_to_baseline_valid_return_count'] <= return_ratio
                   <= PREDECLARED_GATE['max_candidate_to_baseline_valid_return_count']
                   and finite_mad)
        paired.append({
            'seed': seed, 'cohort': footprint_for_seed(seed),
            'rooms': a['rooms'], 'stations': a['stations'],
            'n_patches_baseline': a['n_patches'],
            'n_patches_candidate': b['n_patches'],
            'relative_patch_retention': round(patch_ratio, 4),
            'relative_valid_return_count': round(return_ratio, 4),
            'median_mad_baseline_mm': base,
            'median_mad_candidate_mm': candidate,
            'reduction_mm': round(base - candidate, 4) if finite_mad else None,
            'eligible': bool(quality),
        })

    eligible = [r for r in paired if r['eligible']]
    real_lower = min(site) - PREDECLARED_GATE['real_site_median_bracket_margin_mm']
    real_upper = max(site) + PREDECLARED_GATE['real_site_median_bracket_margin_mm']
    n_improved = sum(r['reduction_mm'] > 0 for r in eligible)
    n_in_bracket = sum(real_lower <= r['median_mad_candidate_mm'] <= real_upper for r in eligible)
    grouped_improved = {name: sum(r['reduction_mm'] > 0 for r in eligible if r['cohort'] == name)
                        for name in ('original_footprint', 'larger_footprint')}
    gates = {
        'all_eight_eligible': len(eligible) >= PREDECLARED_GATE['min_eligible_paired_scenes'],
        'at_least_seven_improved': n_improved >= PREDECLARED_GATE['min_scenes_improved'],
        'median_reduction_at_least_1mm': (_median([r['reduction_mm'] for r in eligible]) or -1)
            >= PREDECLARED_GATE['min_median_paired_reduction_mm'],
        'six_in_expanded_real_interval': n_in_bracket >= PREDECLARED_GATE['min_scenes_in_expanded_real_bracket'],
        'both_footprint_cohorts_respond': all(
            n >= PREDECLARED_GATE['min_improved_per_footprint_cohort']
            for n in grouped_improved.values()),
    }
    # We may reject the candidate, but even a full holdout pass is only
    # provisional. Two real sites cannot establish scanner physical parameters.
    pass_holdout = all(gates.values())
    patch_quantiles = {}
    for variant in VARIANTS:
        selected = [r for r in records if r['variant'] == variant and
                    any(p['seed'] == r['seed'] and p['eligible'] for p in eligible)]
        patch_quantiles[variant] = _quantiles([
            p['observed_plane_mad_mm'] for r in selected for p in r['surface_patches']
            if 'observed_plane_mad_mm' in p])

    return {
        'format': VERSION,
        'design': {'holdout_seeds': list(seeds), 'cal05_tuning_seeds': sorted(USED_BEFORE),
                   'disjoint_from_cal05': True, 'fixed_angle_deg': 0.1,
                   'comparison': 'baseline_vs_preselected_wall_2_4mm_only',
                   'two_footprint_cohorts': True, 'predeclared_gate': PREDECLARED_GATE,
                   'same_layout_and_true_stations_within_pairs': True,
                   'matched_physical_wall_patches': False},
        'real_reference': {'site_count': 2, 'site_median_mad_mm': site,
                           'expanded_median_interval_mm': [round(real_lower, 4), round(real_upper, 4)],
                           'physical_station_ray_metadata': 'UNAVAILABLE'},
        'paired_scenes': paired,
        'holdout': {'eligible_scenes': len(eligible), 'improved_scenes': n_improved,
                    'candidate_in_real_interval_scenes': n_in_bracket,
                    'improved_by_cohort': grouped_improved,
                    'median_baseline_mad_mm': _median([r['median_mad_baseline_mm'] for r in eligible]),
                    'median_candidate_mad_mm': _median([r['median_mad_candidate_mm'] for r in eligible]),
                    'median_paired_reduction_mm': _median([r['reduction_mm'] for r in eligible]),
                    'patch_mad_quantiles_mm': patch_quantiles,
                    'gates': gates, 'all_predeclared_gates_pass': pass_holdout},
        'candidate_decision': ('PROVISIONAL_HOLDOUT_SUPPORT_NOT_PHYSICAL_CALIBRATION'
                               if pass_holdout else 'HOLDOUT_REJECT_OR_INSUFFICIENT'),
        'auto_update_production_config': False,
        'quality_flags': [
            'ONLY_TWO_REAL_E57_SITES_WITH_UNKNOWN_SOURCE_STATIONS',
            'REAL_SITE_PATCHES_ARE_DENSE_SELECTION_NOT_RANDOM_WALL_SAMPLE',
            'PLANAR_RESIDUALS_INCLUDE_WALL_ROUGHNESS_AND_REGISTRATION',
            'CANDIDATE_SELECTED_USING_CAL05_DIFFERENT_SEEDS',
            'PAIR_MATCHES_LAYOUT_STATIONS_NOT_IDENTICAL_LOCAL_PHYSICAL_PATCHES',
            'NO_REAL_MODEL_EVALUATION_OR_PIXEL_CLASS_PERFORMANCE',
        ],
        'limitations': [
            'Predeclared acceptance bounds are exploratory engineering gates, not confidence intervals.',
            'Two real sites have incompatible unknown sensor/station provenance; a single true scanner sigma cannot be inferred.',
            'MAD measured over different dense planar patches from each rendered scan can reflect selection differences.',
            'This stage validates observed-plane metric only, not measured range/incidence bins on real E57.',
            'A positive holdout never automatically changes production scanner config or retrains model.',
        ],
    }


def report_md(doc):
    h = doc['holdout'];rr = doc['real_reference']
    s = ['# CAL-06: независимая проверка диапазона неровности стен 2–4 мм', '',
         'Предварительно зафиксированное сравнение на **восьми новых seed** и двух размерах планировки.',
         'Одинаковые планировка и физические положения станций внутри каждой пары;',
         'кандидат выбран ранее на CAL-05. Измерение: CAL-02 MAD зарегистрированных точек.', '',
         '| Seed | Размер планировки | Стены, baseline MAD, мм | Кандидат MAD, мм | Разница, мм | Число патчей кандидат/база | Достаточно данных |',
         '|---:|---|---:|---:|---:|---:|---|']
    for r in doc['paired_scenes']:
        def fmt(val, sign=False):
            return f'{val:+.3f}' if sign and val is not None else (f'{val:.3f}' if val is not None else 'н/д')
        s.append(f"| {r['seed']} | {r['cohort']} | {fmt(r['median_mad_baseline_mm'])} | "
                 f"{fmt(r['median_mad_candidate_mm'])} | {fmt(r['reduction_mm'], True)} | "
                 f"{r['n_patches_candidate']}/{r['n_patches_baseline']} | "
                 f"{'да' if r['eligible'] else 'нет'} |")
    s += ['', '### Итоги', '',
          f"Валидных пар: **{h['eligible_scenes']}/8**; улучшились: **{h['improved_scenes']}**.",
          f"Медиана baseline: **{h['median_baseline_mad_mm']} мм**, кандидат: **{h['median_candidate_mad_mm']} мм**.",
          f"Медиана парного снижения: **{h['median_paired_reduction_mm']} мм**.",
          f"В расширенном реальном интервале {rr['expanded_median_interval_mm']} мм: "
          f"**{h['candidate_in_real_interval_scenes']}** сцен.", '',
          '### Предварительно заданные проверки', '']
    for key, value in h['gates'].items():
        s.append(f"- {'PASS' if value else 'FAIL'}: {key}")
    s += ['', f"Решение: **{doc['candidate_decision']}**.", '',
          '**Физическая калибровка сканера не объявляется завершённой.**',
          '`customer_like.yaml`, исходные E57 и веса модели не изменены.', '',
          'Ограничения:']
    s.extend('- ' + v for v in doc['limitations'])
    return '\n'.join(s) + '\n'


def run(*, out: Path, config=Path('configs/customer_like.yaml'),
        reference=REAL_REF, seeds=NEW_SEEDS, max_sample=250000, max_patches=24):
    from simscan.generate import generate_scene
    seeds = validate_seeds(seeds)
    if max_sample < 1000 or max_patches < MIN_PATCHES:
        raise ValueError('Sampling must allow at least eight wall patches')
    real = real_reference(reference)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    # Freeze and print config sha with no spatial/station coordinates.
    base_sha = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    for seed in seeds:
        baseline = fixed_base_config(config, seed)
        expected_hash = None
        for variant in VARIANTS:
            cfg, changes = variant_config(baseline, variant)
            scratch = out / '_scratch' / f'seed_{seed}' / variant
            try:
                print(f'CAL06 seed={seed} variant={variant} cohort={footprint_for_seed(seed)}',flush=True)
                generated = generate_scene(cfg, scratch, seed=seed, index=0,
                                           write_scan=False, training_only=True)
                with (scratch/'layout.json').open(encoding='utf-8') as f:
                    sign = paired_layout_signature(json.load(f))
                if expected_hash is None:
                    expected_hash = sign
                if sign != expected_hash:
                    raise RuntimeError(f'Unmatched pair for seed {seed}')
                scans, poses = generated.pop('_scans')
                sampled = sample_registered_with_stations(scans, poses, max_sample=max_sample, seed=42)
                evidence = patch_evidence(sampled, seed=42, max_patches=max_patches)
                rows.append({
                    'seed': seed, 'variant': variant, 'changed_input_range': changes,
                    'layout_station_matched': sign == expected_hash,
                    'layout_signature': sign,  # non-reversible hash only, never original coordinates
                    'rooms': generated['rooms'], 'stations': generated['stations'],
                    'valid_returns': sampled.valid, 'sampling_fraction': round(sampled.sampling_fraction, 6),
                    'n_patches': evidence['summary']['n_patches'],
                    'median_mad_mm': evidence['summary'].get('median_observed_plane_mad_mm'),
                    'median_rms_mm': evidence['summary'].get('median_observed_plane_rms_mm'),
                    'surface_patches': _brief_patches(evidence),
                })
                print(f"   rooms={generated['rooms']} stations={generated['stations']} "
                      f"patches={evidence['summary']['n_patches']} "
                      f"MAD={evidence['summary'].get('median_observed_plane_mad_mm')}",flush=True)
                del scans, poses
            finally:
                shutil.rmtree(scratch, ignore_errors=True)
        # Save incremental aggregate results, but do not claim complete CAL06
        # until all 16 rows exist and the locked holdout evaluation is run.
        (out/'cal06_partial_progress.json').write_text(json.dumps({
            'format': VERSION, 'complete_pairs': sum(1 for a in rows if a['variant']=='roughness_low'),
            'needed_pairs': 8,
            'paired_scene_stats_no_coordinates': [
                {k: v for k,v in a.items() if k not in ('surface_patches','layout_signature')}
                for a in rows], 'source_config_sha256': base_sha,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    result = assess(real, rows, seeds=seeds)
    result['source_config_sha256'] = base_sha
    (out/'cal06_results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    (out/'CAL06_REPORT.md').write_text(report_md(result), encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/customer_like.yaml'))
    p.add_argument('--reference', type=Path, default=REAL_REF)
    p.add_argument('--seeds', nargs='+', type=int, default=list(NEW_SEEDS))
    p.add_argument('--max-sample', type=int, default=250000)
    p.add_argument('--max-patches', type=int, default=24)
    a=p.parse_args()
    result=run(out=a.out,config=a.config,reference=a.reference,seeds=a.seeds,
               max_sample=a.max_sample,max_patches=a.max_patches)
    print(report_md(result),flush=True)

if __name__=='__main__':
    main()
