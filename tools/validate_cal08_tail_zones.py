"""CAL-08 preregistered spatial-length experiment, following rejected CAL-07 tails.

Only the *correlation length of the roughness-tail envelope* varies.  The
amplitude (8-12 mm), nominal probability (20%) and base wall roughness
(2-4 mm), station placement and acquisition parameters stay unchanged.
This is a metric/morphology study, never a physical scanner calibration.
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

VERSION = 'cal08-spatial-tail-length-holdout-v1'
REAL_REF = Path('configs/calibration_real_e57_surfaces.json')
NEW_SEEDS = tuple(range(45101, 45109))
USED_BEFORE = frozenset([42111,42112,42121,42122,42123,42124] +
                        list(range(43101,43109)) + list(range(44101,44109)))
VARIANTS = ('uniform_2_4', 'tail20_corr08', 'tail20_corr18', 'tail20_corr30')
CORRELATION_M = {'uniform_2_4': .8, 'tail20_corr08': .8,
                 'tail20_corr18': 1.8, 'tail20_corr30': 3.0}
PRIMARY_PATCHES = 24  # same biased CAL-02 top-24 reference
EXTENDED_PATCHES = 72  # synthetic-only sensitivity diagnostic
TAIL_PATCH_THRESHOLD_MM = 6.0
PREDECLARED_GATES = {
    'min_eligible_patches': MIN_PATCHES,
    'min_retention_fraction': .75,
    'min_valid_return_ratio': .70,
    'max_valid_return_ratio': 1.30,
    'max_typical_median_shift_mm': .5,
    'min_scenes_preserving_typical_median': 6,
    'min_scenes_inside_real_median_bracket': 6,
    'min_scenes_p90_gain_vs_uniform_20pct': 6,
    'min_p90_gain_vs_uniform_ratio': 1.2,
    'min_scene_median_p90_mm': 5.0,
    'max_scene_median_p90_margin_above_real_p90_mm': 1.0,
    'min_scenes_below_real_p90_ceiling': 6,
    'min_scenes_with_top24_tail_cluster': 4,
    'min_scenes_with_extended_tail_cluster': 4,
    'min_scenes_p90_gain_per_cohort': 3,
    'min_scenes_cluster_gain_vs_short_corr': 4,
}


def validate_seeds(seeds):
    seeds = tuple(int(seed) for seed in seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8 or set(seeds) != set(NEW_SEEDS):
        raise ValueError('CAL-08 requires exactly the eight preregistered new seeds')
    if set(seeds) & USED_BEFORE:
        raise ValueError('CAL-08 seeds overlap earlier calibrations')
    return seeds


def footprint_for_seed(seed):
    if seed not in NEW_SEEDS:
        raise ValueError(f'Unknown CAL-08 seed: {seed}')
    return 'original_footprint' if seed in NEW_SEEDS[:4] else 'larger_footprint'


def fixed_base_config(config_path, seed):
    cfg = prepare_multroom_config(config_path)
    cfg.realism.wall_unevenness_mm = (2., 4.)
    cfg.realism.wall_roughness_tail_probability = 0.
    cfg.realism.wall_roughness_tail_sigma_mm = (8., 12.)
    cfg.realism.wall_roughness_tail_corr_m = .8
    cfg.realism.wall_roughness_tail_transition_sigma = .15
    if seed in NEW_SEEDS[4:]:
        cfg.layout.footprint_x_m = (9.5, 9.5)
        cfg.layout.footprint_y_m = (7.5, 7.5)
    if cfg.layout.rooms != (3, 3) or cfg.scanner.angular_step_deg != .1:
        raise ValueError('CAL-08 fixed footprint or scan acquisition was changed')
    return cfg


def variant_config(base, variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown CAL-08 variant: {variant}')
    cfg = copy.deepcopy(base)
    cfg.realism.wall_roughness_tail_probability = 0. if variant == 'uniform_2_4' else .2
    cfg.realism.wall_roughness_tail_corr_m = CORRELATION_M[variant]
    return cfg, {
        'realism.wall_unevenness_mm': [2., 4.],
        'realism.wall_roughness_tail_sigma_mm': [8., 12.],
        'realism.wall_roughness_tail_probability': cfg.realism.wall_roughness_tail_probability,
        'realism.wall_roughness_tail_corr_m': cfg.realism.wall_roughness_tail_corr_m,
        'realism.wall_roughness_tail_transition_sigma': .15,
    }


def _median(vals):
    return round(float(np.median(vals)), 4) if vals else None


def _quantile(vals, q):
    return round(float(np.quantile(vals, q)), 4) if vals else None


def cluster_metrics(patches):
    """Adjacent wall-candidate grid cells with >=6mm observed-plane MAD.

    Grid indices belong to an estimated wall-line candidate. They are not
    physical wall IDs, world positions or an assertion of actual defects.
    """
    cells = set()
    for p in patches:
        v = p.get('observed_plane_mad_mm')
        if v is None or not np.isfinite(v) or v < TAIL_PATCH_THRESHOLD_MM:
            continue
        required = ('wall_candidate_index', 'patch_col', 'patch_row')
        if any(k not in p for k in required):
            raise ValueError('CAL-08 requires local patch-grid keys from the estimator')
        cells.add(tuple(int(p[k]) for k in required))
    remaining = set(cells)
    largest = 0
    components = 0
    while remaining:
        components += 1
        stack = [remaining.pop()]
        size = 0
        while stack:
            wall, col, row = stack.pop()
            size += 1
            for nxt in ((wall,col-1,row),(wall,col+1,row),
                        (wall,col,row-1),(wall,col,row+1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return {
        'tail_patch_count': len(cells),
        'tail_patch_fraction': round(len(cells)/max(len(patches), 1), 4),
        'largest_tail_cluster': largest,
        'tail_component_count': components,
    }


def patch_metrics(patches):
    vals = []
    for p in patches:
        v = p.get('observed_plane_mad_mm')
        if v is None or not np.isfinite(v) or v < 0:
            raise ValueError('Nonfinite or invalid CAL-08 observed-plane MAD')
        vals.append(float(v))
    return {
        'n_patches': len(vals),
        'median_mad_mm': _median(vals),
        'p90_mad_mm': _quantile(vals, .9),
        'p95_mad_mm': _quantile(vals, .95),
        'mad_iqr_mm': (round(float(np.quantile(vals,.75)-np.quantile(vals,.25)),4)
                       if vals else None),
        **cluster_metrics(patches),
    }


def _brief_patches(evidence):
    keep = ('observed_plane_mad_mm', 'wall_candidate_index', 'patch_col', 'patch_row')
    return [{k: p[k] for k in keep if k in p} for p in evidence['patches']]


def assess(real_doc, records, seeds=NEW_SEEDS):
    seeds = validate_seeds(seeds)
    if len(records) != len(seeds)*len(VARIANTS):
        raise ValueError('CAL-08 requires all 32 scene-variant records')
    medians = [float(x['patch_mad_mm']['median']) for x in real_doc['scans']]
    p90s = [float(x['patch_mad_mm']['p90']) for x in real_doc['scans']]
    if len(medians) != 2 or len(p90s) != 2 or any(not np.isfinite(x) or x<=0 for x in medians+p90s):
        raise ValueError('Expected two complete measured CAL-02 real E57 references')
    bracket = (min(medians)-.5, max(medians)+.5)
    p90_ceiling = max(p90s) + PREDECLARED_GATES['max_scene_median_p90_margin_above_real_p90_mm']

    by_seed = {}
    for r in records:
        seed, variant = int(r['seed']), r['variant']
        if seed not in seeds or variant not in VARIANTS:
            raise ValueError('Unrecognized CAL-08 variant or seed')
        dst = by_seed.setdefault(seed,{})
        if variant in dst:
            raise ValueError('Duplicate CAL-08 pair record')
        dst[variant] = r
    if any(set(by_seed.get(seed,{})) != set(VARIANTS) for seed in seeds):
        raise ValueError('Missing CAL-08 paired scan')

    per_variant = {}
    for variant in VARIANTS:
        pairs=[]
        for seed in seeds:
            group=by_seed[seed]
            uniform=group['uniform_2_4']
            short=group['tail20_corr08']
            row=group[variant]
            if any(not x.get('layout_station_matched') for x in group.values()):
                raise ValueError(f'CAL-08 true layout/station mismatch: seed {seed}')
            if len({x['layout_signature'] for x in group.values()}) != 1:
                raise ValueError(f'CAL-08 true layout hash changed: seed {seed}')
            if len({(x['rooms'], x['stations']) for x in group.values()}) != 1:
                raise ValueError(f'CAL-08 room or station count changed: seed {seed}')
            bm = patch_metrics(uniform['primary_patches'])
            sm = patch_metrics(short['primary_patches'])
            vm = patch_metrics(row['primary_patches'])
            expanded = patch_metrics(row['extended_patches'])
            patch_ratio = vm['n_patches']/max(bm['n_patches'],1)
            returns_ratio = row['valid_returns']/max(uniform['valid_returns'],1)
            g=PREDECLARED_GATES
            short_returns_ratio = short['valid_returns']/max(uniform['valid_returns'],1)
            quality = (bm['n_patches'] >= g['min_eligible_patches']
                       and sm['n_patches'] >= g['min_eligible_patches']
                       and vm['n_patches'] >= g['min_eligible_patches']
                       and patch_ratio >= g['min_retention_fraction']
                       and sm['n_patches']/max(bm['n_patches'],1) >= g['min_retention_fraction']
                       and g['min_valid_return_ratio'] <= returns_ratio <= g['max_valid_return_ratio']
                       and g['min_valid_return_ratio'] <= short_returns_ratio <= g['max_valid_return_ratio']
                       and all(v is not None and v > 0 for v in (
                           bm['median_mad_mm'], sm['median_mad_mm'], vm['median_mad_mm'],
                           bm['p90_mad_mm'], sm['p90_mad_mm'], vm['p90_mad_mm'])))
            p90_gain = (vm['p90_mad_mm']/max(bm['p90_mad_mm'], 1e-9)
                        if bm['p90_mad_mm'] is not None and vm['p90_mad_mm'] is not None else None)
            pairs.append({
                'seed':seed, 'cohort':footprint_for_seed(seed),
                'rooms':row['rooms'],'stations':row['stations'],
                'eligible':bool(quality),
                'primary_patches':vm['n_patches'],'expanded_patches':expanded['n_patches'],
                'primary_retention':round(patch_ratio,4),
                'relative_valid_returns':round(returns_ratio,4),
                'baseline_median_mm':bm['median_mad_mm'],
                'variant_median_mm':vm['median_mad_mm'],
                'median_shift_mm':(round(vm['median_mad_mm']-bm['median_mad_mm'],4)
                                   if vm['median_mad_mm'] is not None and bm['median_mad_mm'] is not None else None),
                'baseline_p90_mm':bm['p90_mad_mm'],'short_corr_p90_mm':sm['p90_mad_mm'],
                'variant_p90_mm':vm['p90_mad_mm'], 'variant_p95_mm':vm['p95_mad_mm'],
                'p90_gain_vs_uniform':round(p90_gain,4) if p90_gain is not None else None,
                'top24_tail_patches':vm['tail_patch_count'],
                'top24_max_cluster':vm['largest_tail_cluster'],
                'extended_tail_patches':expanded['tail_patch_count'],
                'extended_max_cluster':expanded['largest_tail_cluster'],
                'short_corr_max_cluster':patch_metrics(short['primary_patches'])['largest_tail_cluster'],
            })
        valid=[p for p in pairs if p['eligible']]
        median_preserved=sum(abs(p['median_shift_mm']) <= PREDECLARED_GATES['max_typical_median_shift_mm'] for p in valid)
        med_inside=sum(bracket[0] <= p['variant_median_mm'] <= bracket[1] for p in valid)
        p90_gain_count=sum(p['p90_gain_vs_uniform'] >= PREDECLARED_GATES['min_p90_gain_vs_uniform_ratio'] for p in valid)
        p90_ceiling_count=sum(p['variant_p90_mm'] <= p90_ceiling for p in valid)
        primary_clusters=sum(p['top24_max_cluster']>=2 for p in valid)
        extended_clusters=sum(p['extended_max_cluster']>=2 for p in valid)
        cluster_gains=sum(p['top24_max_cluster'] > p['short_corr_max_cluster'] for p in valid)
        p90_cohort={cohort:sum(p['p90_gain_vs_uniform'] >= PREDECLARED_GATES['min_p90_gain_vs_uniform_ratio']
                                for p in valid if p['cohort']==cohort)
                    for cohort in ('original_footprint','larger_footprint')}
        scene_p90=_median([p['variant_p90_mm'] for p in valid])
        gates={
            'all_eight_pairs_eligible':len(valid)==8,
            'typical_median_preserved_six_of_eight':median_preserved>=6,
            'scene_median_in_real_bracket_six_of_eight':med_inside>=6,
            'p90_gain_20pct_six_of_eight':p90_gain_count>=6,
            'p90_below_real_upper_site_plus_1mm_six_of_eight':p90_ceiling_count>=6,
            'median_scene_p90_between_5_and_real_upper_plus_1mm':scene_p90 is not None and 5.0<=scene_p90<=p90_ceiling,
            'top24_cluster_in_four_of_eight':primary_clusters>=4,
            'extended_cluster_in_four_of_eight':extended_clusters>=4,
            'p90_gain_in_both_cohorts_three_of_four':all(v>=3 for v in p90_cohort.values()),
            'cluster_gain_over_short_corr_in_four_of_eight':cluster_gains>=4,
        }
        per_variant[variant]={
            'eligible_scenes':len(valid), 'paired_scenes':pairs,
            'median_scene_mad_mm':_median([p['variant_median_mm'] for p in valid]),
            'median_scene_p90_mm':scene_p90,
            'median_scene_p95_mm':_median([p['variant_p95_mm'] for p in valid]),
            'median_p90_gain_ratio':_median([p['p90_gain_vs_uniform'] for p in valid]),
            'median_top24_tail_patches':_median([p['top24_tail_patches'] for p in valid]),
            'median_top24_largest_cluster':_median([p['top24_max_cluster'] for p in valid]),
            'median_extended_largest_cluster':_median([p['extended_max_cluster'] for p in valid]),
            'typical_median_preserved_scenes':median_preserved,
            'p90_gain_scenes':p90_gain_count,
            'p90_gain_by_cohort':p90_cohort,
            'primary_cluster_scenes':primary_clusters,
            'extended_cluster_scenes':extended_clusters,
            'cluster_gain_vs_short_scenes':cluster_gains,
            'gates':gates,
            'all_gates_pass':all(gates.values()) if variant in ('tail20_corr18','tail20_corr30') else None,
        }
    passed=[name for name in ('tail20_corr18','tail20_corr30') if per_variant[name]['all_gates_pass']]
    selected=passed[0] if passed else None
    return {
        'format':VERSION,
        'design':{'seeds':list(seeds),'excluded_previous_seeds':sorted(USED_BEFORE),
                  'variants':list(VARIANTS),'tail_amplitude_mm':[8,12],
                  'tail_probability':.2,'base_roughness_mm':[2,4],
                  'correlation_m':CORRELATION_M,'primary_patch_limit':PRIMARY_PATCHES,
                  'extended_patch_limit':EXTENDED_PATCHES,
                  'tail_patch_threshold_mad_mm':TAIL_PATCH_THRESHOLD_MM,
                  'predeclared_gates':PREDECLARED_GATES,
                  'matched_layout_and_true_stations_within_seed':True,
                  'matched_physical_surface_patches':False},
        'real_reference':{'n_sites':2,'site_median_mad_mm':medians,
                          'site_p90_mad_mm':p90s,'expanded_median_bracket_mm':list(bracket),
                          'p90_ceiling_mm':p90_ceiling,'station_ray_provenance':'UNKNOWN'},
        'variants':per_variant,
        'selected_candidate':selected,
        'decision':('PROVISIONAL_SPATIAL_LENGTH_'+selected+'_SUPPORTED' if selected
                    else 'NO_CORRELATION_LENGTH_PASSED_PREDECLARED_GATES'),
        'auto_update_production_config':False,
        'quality_flags':[
            'TWO_REAL_E57_SITES_ONLY',
            'DENSE_PATCH_SELECTION_BIAS',
            'REAL_RAY_STATION_PROVENANCE_UNAVAILABLE',
            'OBSERVED_PLANE_MAD_CONFLATES_SURFACE_SCANNER_AND_REGISTRATION',
            'EXTENDED_72_PATCH_CLUSTER_IS_SYNTHETIC_ONLY',
            'LOCAL_PATCH_GRID_IS_NOT_GROUND_TRUTH_WALL_GEOMETRY',
        ],
    }


def report_md(doc):
    def f(x): return f'{x:.4f}' if x is not None else 'н/д'
    lines=['# CAL-08: длина пространственных зон неровности', '',
           'Сравнение 0,8 / 1,8 / 3,0 м корреляции при постоянных 20% и 8–12 мм.',
           'Обязательный контроль: однородный диапазон 2–4 мм.', '',
           '| Вариант | Валидных сцен | MAD p50, мм | MAD p90, мм | MAD p95, мм | p90 / контроль | Кластеров top24 | Кластеров top72 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in VARIANTS:
        d=doc['variants'][name]
        lines.append(f"| {name} | {d['eligible_scenes']}/8 | {f(d['median_scene_mad_mm'])} | {f(d['median_scene_p90_mm'])} | {f(d['median_scene_p95_mm'])} | {f(d['median_p90_gain_ratio'])} | {d['primary_cluster_scenes']} | {d['extended_cluster_scenes']} |")
    for name in VARIANTS[2:]:
        d=doc['variants'][name]
        lines+=['',f'## {name}', '']
        lines.extend('- '+('PASS' if v else 'FAIL')+' '+k for k,v in d['gates'].items())
        lines+=['', '| Seed | MAD p50 | MAD p90 | p90/контроль | Кластер top24 | Кластер top72 | Возвраты к контролю |',
                '|---:|---:|---:|---:|---:|---:|---:|']
        for p in d['paired_scenes']:
            lines.append(f"| {p['seed']} | {f(p['variant_median_mm'])} | {f(p['variant_p90_mm'])} | {f(p['p90_gain_vs_uniform'])} | {p['top24_max_cluster']} | {p['extended_max_cluster']} | {f(p['relative_valid_returns'])} |")
    lines+=['',f"## Решение: {doc['decision']}", '',
            'Промышленный YAML, исходные E57 и веса модели не изменены.',
            'Это не физическая калибровка; всего два реальных E57, топ-24 патча с селекционным смещением,',
            'а реальные станции и исходные лучи не идентифицируемы.',
            'Топ-72 учитывается только как диагностическая проверка пространственных кластеров.',
            'Геометрия и истинные позы внутри каждой группы seed проверены хешем.', '',
            'Quality flags: '+', '.join(doc['quality_flags'])]
    return '\n'.join(lines)+'\n'


def run(*,out:Path,config=Path('configs/customer_like.yaml'),reference=REAL_REF,
        seeds=NEW_SEEDS,max_sample=250000,max_patches=EXTENDED_PATCHES):
    from simscan.generate import generate_scene
    seeds=validate_seeds(seeds)
    if max_sample<1000 or max_patches<EXTENDED_PATCHES:
        raise ValueError('CAL-08 requires full 72-patch expanded sensitivity selection')
    real=real_reference(reference)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    config_sha=hashlib.sha256(Path(config).read_bytes()).hexdigest()
    rows=[]
    for seed in seeds:
        base=fixed_base_config(config,seed)
        signature=None
        for variant in VARIANTS:
            cfg,changes=variant_config(base,variant)
            scratch=out/'_scratch'/f'seed_{seed}'/variant
            try:
                print(f'CAL08 seed={seed} variant={variant}',flush=True)
                generated=generate_scene(cfg,scratch,seed=seed,index=0,
                                         write_scan=False,training_only=True)
                with (scratch/'layout.json').open(encoding='utf-8') as handle:
                    current=paired_layout_signature(json.load(handle))
                if signature is None:signature=current
                if signature!=current:raise RuntimeError(f'Geometry/station mismatch: {seed}/{variant}')
                scans,poses=generated.pop('_scans')
                reservoir=sample_registered_with_stations(scans,poses,
                                                             max_sample=max_sample,seed=42)
                evidence=patch_evidence(reservoir,seed=42,max_patches=max_patches,
                                        include_patch_grid=True)
                expanded=_brief_patches(evidence)
                primary=expanded[:PRIMARY_PATCHES]
                row={'seed':seed,'variant':variant,'cohort':footprint_for_seed(seed),
                     'changes':changes,'layout_station_matched':True,
                     'layout_signature':current,'rooms':generated['rooms'],
                     'stations':generated['stations'],'valid_returns':reservoir.valid,
                     'primary_patches':primary,'extended_patches':expanded,
                     'primary':patch_metrics(primary),'extended':patch_metrics(expanded)}
                rows.append(row)
                print('   p50=',row['primary']['median_mad_mm'],
                      'p90=',row['primary']['p90_mad_mm'],
                      'top24 cluster=',row['primary']['largest_tail_cluster'],
                      'top72 cluster=',row['extended']['largest_tail_cluster'],flush=True)
                del scans,poses,reservoir
            finally:
                shutil.rmtree(scratch,ignore_errors=True)
        (out/'cal08_partial_progress.json').write_text(json.dumps({
            'format':VERSION,'complete_seed_groups':sum(r['variant']=='tail20_corr30' for r in rows),
            'expected_seed_groups':len(seeds),'config_sha256':config_sha,
            'aggregate_rows_no_coordinates':[{k:v for k,v in r.items()
                                              if k not in ('layout_signature','primary_patches','extended_patches')}
                                             for r in rows],
        },indent=2,ensure_ascii=False),encoding='utf-8')
    result=assess(real,rows,seeds=seeds)
    result['source_config_sha256']=config_sha
    (out/'cal08_results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'CAL08_REPORT.md').write_text(report_md(result),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/customer_like.yaml'))
    parser.add_argument('--reference',type=Path,default=REAL_REF)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(NEW_SEEDS))
    parser.add_argument('--max-sample',type=int,default=250000)
    args=parser.parse_args()
    print(report_md(run(out=args.out,config=args.config,reference=args.reference,
                        seeds=args.seeds,max_sample=args.max_sample)),flush=True)


if __name__=='__main__': main()
