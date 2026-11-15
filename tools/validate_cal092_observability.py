"""CAL-09.2: end-to-end direct-ray defect observability, paired physical patches.

Run from repository root:
 PYTHONPATH=. python tools/validate_cal092_observability.py --out results/CAL092

Canonical core: 8 CAL-09 seeds x (uniform, zone08/fine35, zone18/fine35)
x (0.30m, 0.10m) = 48 actual scanner simulations at 0.1-degree ray step.
Four ablations of each simulation are exact same sampled rays. No E57 writes.

For smoke only:
 PYTHONPATH=. python tools/validate_cal092_observability.py --out /tmp/cal092_smoke \
   --seeds 46101 --variants uniform_2_4 zone18_fine35 \
   --resolutions 0.30 --angular-step-deg 0.50 --max-sample 20000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from tools.cal092_ray_observability import (
    ABLATIONS, DETECTED_MAD_MM, MIN_GT_VISIBLE_FRACTION,
    evaluate_rays, priority_sample,
)
from tools.calibrate_factorial_surfaces import paired_layout_signature
from tools.validate_cal09_two_scale_defects import (
    SEEDS, VARIANTS, base_config, variant_config,
)

FORMAT = 'cal092-direct-ray-observability-v1'
DEFAULT_VARIANTS = ('uniform_2_4', 'zone08_fine35', 'zone18_fine35')
RESOLUTIONS_M = (.30, .10)
# Gate definitions are exploratory and cannot override CAL09 observed-patch FAIL.
GATES = {
    'min_paired_physical_patches': 12,
    'min_positive_gt_patches': 3,
    'min_seed_support': 6,
    'min_combined_recall': .60,
    'max_combined_false_positive_rate': .20,
}


def one_case(config_path, seed, variant, resolution, out, *,
             max_sample=180000, angular_step_deg=.1):
    from simscan.generate import generate_scene
    cfg = variant_config(base_config(config_path, seed), variant)
    cfg.realism.tessellation_m = resolution
    cfg.scanner.angular_step_deg = angular_step_deg
    work = Path(out) / '_scratch' / f'{seed}_{variant}_{resolution:.2f}'
    try:
        generated = generate_scene(cfg, work, seed=seed, index=0,
                                   write_scan=False, training_only=True,
                                   capture_truth_hits=True)
        scans, poses = generated.pop('_scans')
        mesh = generated.pop('_truth_mesh')
        doc = json.loads((work / 'layout.json').read_text(encoding='utf-8'))
        layout_signature = paired_layout_signature(doc)
        # Experiment is memory-limited by the original simulated rays,
        # not by the fixed-size sampled physical patch evaluation.
        sample = priority_sample(scans, max_sample=max_sample, seed=42)
        del scans, poses
        measured = evaluate_rays(sample, mesh, doc)
        return {
            'seed': seed, 'variant': variant, 'tessellation_m': resolution,
            'angular_step_deg': angular_step_deg,
            'layout_station_signature': layout_signature,
            'stations': generated['stations'], 'rooms': generated['rooms'],
            'direct_wall_returns_total': sample['total_direct_wall_returns'],
            'direct_wall_returns_sampled': sample['sampled_direct_wall_returns'],
            'metrics': measured,
        }
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarize(rows, *, seeds=SEEDS, variants=DEFAULT_VARIANTS,
              resolutions=RESOLUTIONS_M, angular_step_deg=.1):
    expected = {(seed, var, step) for seed in seeds for var in variants for step in resolutions}
    indexed = {(r['seed'], r['variant'], r['tessellation_m']): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Duplicate or missing CAL-09.2 scene')
    pairing = []
    for seed in seeds:
        one = [r for r in rows if r['seed'] == seed]
        pairing.append({
            'seed': seed,
            'same_layout_and_stations': len({r['layout_station_signature'] for r in one}) == 1,
            'same_station_count': len({r['stations'] for r in one}) == 1,
        })
    grouped = {}
    median = lambda x: round(float(np.median(x)), 4) if x else None
    for variant in variants:
        for step in resolutions:
            subset = [indexed[(seed, variant, step)] for seed in seeds]
            val = lambda r, mode, k: r['metrics']['modes'][mode][k]
            gt_support = [r for r in subset if
                          val(r, 'combined', 'n_gt_positive') >= GATES['min_positive_gt_patches']
                          and val(r, 'combined', 'n_patches') >= GATES['min_paired_physical_patches']]
            recall = [val(r, 'combined', 'recall') for r in gt_support
                      if val(r, 'combined', 'recall') is not None]
            fpr = [val(r, 'combined', 'false_positive_rate') for r in gt_support
                   if val(r, 'combined', 'false_positive_rate') is not None]
            grouped[f'{variant}_{step:.2f}'] = {
                'variant': variant, 'tessellation_m': step, 'seed_count': len(subset),
                'median_physical_patches': median([r['metrics']['n_physical_wall_patches'] for r in subset]),
                'gt_supported_seeds': len(gt_support),
                'median_gt_positive_patches': median([val(r, 'combined', 'n_gt_positive') for r in subset]),
                'median_combined_recall': median(recall),
                'median_combined_fpr': median(fpr),
                'median_mad_mm_by_mode': {mode: median([
                    val(r, mode, 'median_patch_mad_mm') for r in subset
                    if val(r, mode, 'median_patch_mad_mm') is not None])
                    for mode in ABLATIONS},
                'median_noise_increment_mm': median([
                    r['metrics']['paired_ablation_shifts']['range_only_minus_ideal_median_mad_mm']
                    for r in subset
                    if r['metrics']['paired_ablation_shifts']['range_only_minus_ideal_median_mad_mm'] is not None]),
                'median_registration_increment_mm': median([
                    r['metrics']['paired_ablation_shifts']['registration_only_minus_ideal_median_mad_mm']
                    for r in subset
                    if r['metrics']['paired_ablation_shifts']['registration_only_minus_ideal_median_mad_mm'] is not None]),
                'recall_supported_seeds': len(recall),
                'recall_gate_scenes': sum(r >= GATES['min_combined_recall'] for r in recall),
                'fpr_gate_scenes': sum(r <= GATES['max_combined_false_positive_rate'] for r in fpr),
            }
    canonical = (
        tuple(seeds) == SEEDS and tuple(variants) == DEFAULT_VARIANTS and
        tuple(resolutions) == RESOLUTIONS_M and angular_step_deg == .1)
    checks = {
        'complete_canonical_design': canonical,
        'matched_layouts_and_station_counts': all(
            x['same_layout_and_stations'] and x['same_station_count'] for x in pairing),
        'four_paired_ablations_per_case': all(
            set(r['metrics']['modes']) == set(ABLATIONS) for r in rows),
        'sampled_direct_ray_provenance_valid': all(
            r['metrics']['n_geometrically_valid_rays'] > 0 and
            r['metrics']['n_geometrically_valid_rays'] <=
            r['direct_wall_returns_sampled'] for r in rows),
    }
    candidate_check = {
        key: {'sufficient_gt_support': x['gt_supported_seeds'] >= GATES['min_seed_support'],
              'recall_6_of_8': x['recall_gate_scenes'] >= GATES['min_seed_support'],
              'false_positive_rate_6_of_8': x['fpr_gate_scenes'] >= GATES['min_seed_support']}
        for key, x in grouped.items() if x['variant'] != 'uniform_2_4'
    }
    observed_pass = canonical and all(checks.values()) and bool(candidate_check) and any(
        all(z.values()) for z in candidate_check.values())
    return {
        'format': FORMAT,
        'design': {
            'seeds': list(seeds), 'variants': list(variants),
            'resolutions_m': list(resolutions), 'angular_step_deg': angular_step_deg,
            'gates': GATES, 'patch_size_m': [.9, 1.25],
            'gt_visible_fraction_threshold': MIN_GT_VISIBLE_FRACTION,
            'detected_patch_mad_threshold_mm': DETECTED_MAD_MM,
            'physical_patch_assignment_uses_ground_truth': True,
            'same_ray_counterfactual_ablation': list(ABLATIONS),
        },
        'checks': checks, 'paired_layout_checks': pairing,
        'candidate_observability_gates': candidate_check,
        'observability_supported_on_oracle_patches': observed_pass,
        'grouped': grouped,
        'cases': rows,
        'decision': 'ORACLE_OBSERVABILITY_SUPPORTED_NOT_CAL09_ACCEPTED' if observed_pass
                    else 'CAL092_OBSERVABILITY_NOT_ESTABLISHED',
        'auto_update_production_config': False,
        'limitations': [
            'The physical wall grouping is GT-oracle and is not a blind defect or wall detector.',
            'Only direct wall rays; glass, edge-beam, mixed and virtual returns are excluded.',
            'Returned rays sample only visible surfaces; GT positive rate is not wall area coverage.',
            'Patch MAD is computed with the original robust 3D vertical-plane estimator.',
            'Same rays are reused for 2x2 range-noise and registration counterfactuals.',
            'Changing tessellation changes ray intersections and thus ray sets; resolution comparison is not ray-for-ray.',
            'The experiment uses CAL-09 seeds, not new real E57 or a statistical holdout.',
            'Only two real E57 references with unknown station provenance remain available.',
            'An oracle-patch PASS cannot override the CAL-09 top-24 blind-patch FAIL.',
        ],
    }


def report(doc):
    lines = ['# CAL-09.2: наблюдаемость дефектов стен после сканирования', '',
             'Дефекты сопоставлены с прямыми отражениями по физическим треугольникам.',
             'Четыре абляции используют одни и те же отражённые лучи.',
             'Разбиение на физические патчи использует GT: это диагностическая верхняя оценка.',
             '', '| Критерий | Результат |', '|---|---|']
    lines += [f"| {key} | {'PASS' if val else 'FAIL'} |" for key, val in doc['checks'].items()]
    lines += ['', '| Вариант | Mesh, м | Median MAD (ideal) | MAD (range) | MAD (registration) | MAD (combined) | GT-supported seeds | Recall | FPR |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda x: 'n/a' if x is None else f'{x:.3f}'
    for group in doc['grouped'].values():
        m = group['median_mad_mm_by_mode']
        lines.append(f"| {group['variant']} | {group['tessellation_m']:.2f} | "
                     f"{fmt(m['ideal'])} | {fmt(m['range_only'])} | "
                     f"{fmt(m['registration_only'])} | {fmt(m['combined'])} | "
                     f"{group['gt_supported_seeds']} | {fmt(group['median_combined_recall'])} | "
                     f"{fmt(group['median_combined_fpr'])} |")
    lines += ['', '**Решение:** ' + doc['decision'], '',
              'CAL-09 остаётся непринятым независимо от данного результата.',
              'Production YAML и оригинальные E57 не изменяются.', '',
              '### Ограничения']
    lines += ['- ' + x for x in doc['limitations']]
    return '\n'.join(lines) + '\n'


def run(out, config=Path('configs/customer_like.yaml'), *, seeds=SEEDS,
        variants=DEFAULT_VARIANTS, resolutions=RESOLUTIONS_M,
        angular_step_deg=.1, max_sample=180000):
    seeds, variants, resolutions = tuple(seeds), tuple(variants), tuple(resolutions)
    if not seeds or not variants or not resolutions or max_sample < 1000:
        raise ValueError('Empty design or insufficient sampling')
    if len(seeds) != len(set(seeds)) or any(s not in SEEDS for s in seeds):
        raise ValueError('Only distinct CAL-09 paired seeds accepted')
    if len(variants) != len(set(variants)) or any(v not in VARIANTS for v in variants):
        raise ValueError('Invalid variants')
    if len(resolutions) != len(set(resolutions)) or any(r not in RESOLUTIONS_M for r in resolutions):
        raise ValueError('Only 0.30 and 0.10 m meshes accepted')
    if not .05 <= angular_step_deg <= .75:
        raise ValueError('Angular step must be 0.05..0.75 degree; 0.1 is canonical')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({
        'format': FORMAT, 'source': sha, 'max_sample': max_sample,
        'angular_step_deg': angular_step_deg}, sort_keys=True).encode()).hexdigest()
    cache = out / '_completed_rows'
    cache.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        for resolution in resolutions:
            for variant in variants:
                path = cache / f'{seed}_{variant}_{resolution:.2f}.json'
                if path.exists():
                    saved = json.loads(path.read_text(encoding='utf-8'))
                    if saved.get('fingerprint') == fingerprint:
                        print(f'CAL092 cached seed={seed} step={resolution} variant={variant}', flush=True)
                        rows.append(saved['row'])
                        continue
                print(f'CAL092 seed={seed} step={resolution:.2f} variant={variant} '
                      f'angular={angular_step_deg}°', flush=True)
                row = one_case(config, seed, variant, resolution, out,
                               max_sample=max_sample, angular_step_deg=angular_step_deg)
                rows.append(row)
                temp = path.with_suffix('.tmp')
                temp.write_text(json.dumps({'fingerprint': fingerprint, 'row': row},
                                           allow_nan=False), encoding='utf-8')
                temp.replace(path)
        (out / 'cal092_progress.json').write_text(json.dumps({
            'completed_cases': len(rows), 'total_cases': len(seeds) * len(variants) * len(resolutions),
            'completed_seeds': len({r['seed'] for r in rows}),
            'source_config_sha256': sha, 'fingerprint': fingerprint},
            indent=2), encoding='utf-8')
    result = summarize(rows, seeds=seeds, variants=variants,
                       resolutions=resolutions, angular_step_deg=angular_step_deg)
    result['source_config_sha256'] = sha
    result['run_fingerprint'] = fingerprint
    (out / 'cal092_results.json').write_text(json.dumps(result, ensure_ascii=False,
                                                        indent=2, allow_nan=False), encoding='utf-8')
    (out / 'CAL092_REPORT.md').write_text(report(result), encoding='utf-8')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path('configs/customer_like.yaml'))
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    p.add_argument('--variants', nargs='+', default=list(DEFAULT_VARIANTS))
    p.add_argument('--resolutions', type=float, nargs='+', default=list(RESOLUTIONS_M))
    p.add_argument('--max-sample', type=int, default=180000)
    p.add_argument('--angular-step-deg', type=float, default=.1)
    a = p.parse_args()
    print(report(run(a.out, a.config, seeds=a.seeds, variants=a.variants,
                     resolutions=a.resolutions, max_sample=a.max_sample,
                     angular_step_deg=a.angular_step_deg)), flush=True)


if __name__ == '__main__':
    main()
