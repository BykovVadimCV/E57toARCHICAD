"""CAL-09.1 mesh-truth study; NO scanner simulation or production changes.

Run from repository root:
 PYTHONPATH=. python tools/validate_cal091_ground_truth.py --out results/CAL091
Full design: same eight CAL-09 seeds x five variants x 0.30/0.15/0.10m wall meshes.
Outputs aggregate JSON/MD and optional compressed vertex/triangle truth NPZ.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.cal091_mesh_truth import mesh_truth_metrics
from tools.validate_cal09_two_scale_defects import SEEDS, VARIANTS, base_config, variant_config

RESOLUTIONS_M = (0.30, 0.15, 0.10)
FORMAT = 'cal091-ground-truth-spatial-validation-v1'
# All gates are algorithmic; geometry-only result cannot override failed CAL-09 E57 gates.
GATES = {
    'max_background_delta_rms_mm': 1e-8,
    'max_area_fraction_change_015_vs_010': 0.02,
    'max_wall_area_relative_change_015_vs_010': 0.005,
    'min_converged_seed_count_per_variant': 6,
}


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def make_wall_mesh(cfg, seed):
    """Replay initial generate_scene RNG up to the wall mesh, without the scanner.

    Real wall pieces and box tessellation are reused from SceneBuilder; decor and
    rays are intentionally absent. Since sample_params and jitter_thickness use
    unchanged RNG draws, layout and real stochastic basis remain paired.
    """
    from simscan.generate import scene_rng, make_layout
    from simscan import realism
    from simscan.planwarp import sample_warp
    from simscan.scene import SceneBuilder, SceneMesh

    rng = scene_rng(seed, 0)
    layout = make_layout(cfg, rng, seed, 0)
    params = realism.sample_params(cfg.realism, rng)
    if params['enabled']:
        realism.jitter_thickness(layout, cfg.realism.thickness_jitter_mm, rng)
        params['warp'] = sample_warp([layout], params['warp'], cfg.realism, rng)
    # Do not call Furnisher or ray simulator: wall geometry itself is sufficient.
    layout.meta['warp'] = params['warp']
    solids = [s for s in SceneBuilder(layout).build()
              if s.label == 'wall' and s.deform and s.mesh is None]
    mesh = SceneMesh.from_solids(solids, cfg.realism.tessellation_m)
    doc = layout.to_dict()
    layout_hash = _sha_json({k: doc.get(k) for k in ('walls', 'rooms', 'openings')})
    return mesh, params, layout_hash


def _digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def one_scene(config_path, seed, variant, resolution, *, save_npz=None):
    from simscan import realism
    from simscan.labels import LABEL_ID

    cfg = variant_config(base_config(config_path, seed), variant)
    cfg.realism.tessellation_m = resolution
    mesh, params, layout_hash = make_wall_mesh(cfg, seed)
    xyz = mesh.vertices.astype(np.float64)
    mask = realism.roughness_tail_envelope(xyz, params)
    base = copy.copy(params)
    base['roughness_tail_probability'] = 0.0
    residual = (realism.displacement(xyz, params) -
                realism.displacement(xyz, base))[:, :2]
    fine = (realism.fine_defect_field(xyz, params)
            if params['roughness_tail_probability'] > 0 and
            params['roughness_tail_fine_corr_m'] > 0 else None)
    metrics = mesh_truth_metrics(xyz, mesh.triangles, mesh.tri_instance,
                                 mask, residual, fine,
                                 tri_label=mesh.tri_label,
                                 vert_deform=mesh.vert_deform,
                                 wall_label=LABEL_ID['wall'])
    if save_npz is not None:
        save_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_npz,
                            vertices_before_deformation_m=xyz.astype('f4'),
                            triangles=mesh.triangles.astype('i4'),
                            triangle_wall_instance=mesh.tri_instance.astype('i4'),
                            envelope=mask.astype('f4'),
                            vertex_mask=(mask >= .5),
                            injected_displacement_xy_m=residual.astype('f4'))
    return {
        'seed': seed, 'variant': variant, 'tessellation_m': resolution,
        'layout_hash': layout_hash,
        'mesh_vertex_hash': _digest(xyz.astype('<f4')),
        'envelope_hash': _digest(mask.astype('<f8')),
        'fine_field_hash': _digest(fine.astype('<f8')) if fine is not None else None,
        'metrics': metrics,
    }


def summarize(rows, *, seeds=SEEDS, resolutions=RESOLUTIONS_M):
    keys = [(r['seed'], r['variant'], r['tessellation_m']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate ground truth scene')
    expected = {(s, name, step) for s in seeds for name in VARIANTS for step in resolutions}
    if set(keys) != expected:
        raise ValueError('Incomplete CAL-09.1 factorial study')
    grouped = {(r['seed'], r['variant'], r['tessellation_m']): r for r in rows}
    scale_independent = []
    for seed in seeds:
        for step in resolutions:
            rr = [grouped[(seed, name, step)] for name in VARIANTS]
            scales = {r['variant']: r for r in rr}
            hashes_match = len({r['layout_hash'] for r in rr}) == 1 and len({r['mesh_vertex_hash'] for r in rr}) == 1
            fine_change_preserves_envelope = (
                scales['zone08_fine35']['envelope_hash'] == scales['zone08_fine65']['envelope_hash'] and
                scales['zone18_fine35']['envelope_hash'] == scales['zone18_fine65']['envelope_hash'])
            zone_change_preserves_texture = (
                scales['zone08_fine35']['fine_field_hash'] == scales['zone18_fine35']['fine_field_hash'] and
                scales['zone08_fine65']['fine_field_hash'] == scales['zone18_fine65']['fine_field_hash'])
            off_rms = [r['metrics']['outside_support_rms_mm'] for r in rr if r['variant'] != 'uniform_2_4']
            background_preserved = all(x is not None and x <= GATES['max_background_delta_rms_mm'] for x in off_rms)
            scale_independent.append({'seed': seed, 'resolution_m': step,
                                     'paired_geometry': hashes_match,
                                     'fine_change_preserves_envelope': fine_change_preserves_envelope,
                                     'zone_change_preserves_fine_texture': zone_change_preserves_texture,
                                     'background_preserved': background_preserved})

    convergence = {}
    if 0.15 in resolutions and 0.10 in resolutions:
        for name in VARIANTS:
            comparisons = []
            for seed in seeds:
                a = grouped[(seed, name, 0.15)]['metrics']
                b = grouped[(seed, name, 0.10)]['metrics']
                delta = abs(a['defect_area_fraction'] - b['defect_area_fraction'])
                wall_delta = abs(a['vertical_wall_area_m2'] - b['vertical_wall_area_m2']) / max(1e-9, b['vertical_wall_area_m2'])
                comparisons.append({'seed': seed, 'area_fraction_abs_delta': round(delta, 6),
                                    'relative_wall_area_delta': round(wall_delta, 7),
                                    'within_threshold': delta <= GATES['max_area_fraction_change_015_vs_010'] and
                                                        wall_delta <= GATES['max_wall_area_relative_change_015_vs_010']})
            convergence[name] = {'seeds_converged': sum(x['within_threshold'] for x in comparisons),
                                 'per_seed': comparisons}
    checks = {
        'full_eight_seeds': tuple(seeds) == SEEDS,
        'all_three_predeclared_resolutions': tuple(resolutions) == RESOLUTIONS_M,
        'all_paired_geometry_hashes_match': all(x['paired_geometry'] for x in scale_independent),
        'fine_does_not_move_coarse_mask': all(x['fine_change_preserves_envelope'] for x in scale_independent),
        'coarse_does_not_move_fine_texture': all(x['zone_change_preserves_fine_texture'] for x in scale_independent),
        'outside_defect_background_is_preserved': all(x['background_preserved'] for x in scale_independent),
        'area_convergence_6_of_8_per_variant': bool(convergence) and all(
            x['seeds_converged'] >= GATES['min_converged_seed_count_per_variant']
            for x in convergence.values()),
    }
    return {
        'format': FORMAT,
        'design': {'seeds': list(seeds), 'variants': VARIANTS,
                   'resolutions_m': list(resolutions), 'gates': GATES,
                   'mesh_source': 'real SceneBuilder wall solids, scanner skipped',
                   'mask': 'mean envelope over 3 vertices >=0.5',
                   'cal09_original_seeds_reused_diagnostic_not_independent_holdout': True},
        'checks': checks, 'algorithmic_gates_pass': all(checks.values()),
        'spatial_orthogonality_per_seed': scale_independent,
        'convergence': convergence,
        'scene_metrics': rows,
        'decision': ('CAL091_MESH_TRUTH_GATES_PASS_CAL09_NOT_PROMOTED' if all(checks.values())
                     else 'CAL091_MESH_TRUTH_GATES_FAIL'),
        'auto_update_production_config': False,
        'limitations': [
            'Only vertical wall faces; all wall pieces come from actual SceneBuilder box tessellation.',
            'The geometry-only replay skips furnishing, scanner, range noise and registration.',
            'Truth topology uses same-wall-instance edge welding; does not bridge different wall IDs.',
            'Source triangles on wall seams may be duplicated; edge welding prevents false fragmentation.',
            'Correlation values are distance-binned diagnostics, NOT physical scanner resolution or a fitted material length.',
            'Fine texture only measured in the >=0.95 envelope core; cells without enough pairs report null.',
            'CAL-09 original 40 simulated scans failed MAD/cluster gates; this cannot overturn that outcome.',
            'These reused seeds are NOT a separate independent holdout.',
        ],
    }


def markdown(doc):
    lines = ['# CAL-09.1: истинная пространственная геометрия дефектов', '',
             'Проверка только по истинной стеновой сетке SceneBuilder. Не по E57/MAD-патчам.',
             'Результат CAL-09 (40 сканов) остается отклоненным.', '',
             '| Гейт | Итог |', '|---|---|']
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in doc['checks'].items()]
    lines += ['', '| Вариант | Шаг, м | Медианная доля площади дефектов | Медианная площадь крупнейшего кластера, м² | Медианная фрагментация |',
              '|---|---:|---:|---:|---:|']
    for name in VARIANTS:
        for resolution in doc['design']['resolutions_m']:
            rr = [r['metrics'] for r in doc['scene_metrics'] if r['variant'] == name and r['tessellation_m'] == resolution]
            med = lambda k: np.median([x[k] for x in rr if x[k] is not None]) if any(x[k] is not None for x in rr) else None
            format_metric = lambda val: 'н/д' if val is None else f'{val:.4f}'
            lines.append(f"| {name} | {resolution:.2f} | {format_metric(med('defect_area_fraction'))} | "
                         f"{format_metric(med('largest_component_area_m2'))} | {format_metric(med('fragmentation_index'))} |")
    lines += ['', '## Решение', doc['decision'], '',
              'Исходный CAL-09 не подтвержден, production не меняется.', '',
              '## Ограничения']
    lines += ['- ' + x for x in doc['limitations']]
    return '\n'.join(lines) + '\n'


def run(out, config=Path('configs/customer_like.yaml'), seeds=SEEDS,
        resolutions=RESOLUTIONS_M, save_truth='one'):
    seeds, resolutions = tuple(seeds), tuple(resolutions)
    if not seeds or not resolutions or any(s not in SEEDS for s in seeds):
        raise ValueError('Unsupported seeds or resolutions')
    if any(r not in RESOLUTIONS_M for r in resolutions):
        raise ValueError('Resolution must be one of 0.30, 0.15 or 0.10m')
    if save_truth not in ('none', 'one', 'all'):
        raise ValueError('save_truth must be none, one or all')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config_hash = hashlib.sha256(Path(config).read_bytes()).hexdigest()
    rows = []
    for seed in seeds:
        for resolution in resolutions:
            for variant in VARIANTS:
                print(f'CAL091 seed={seed} tessellation={resolution:.2f} variant={variant}', flush=True)
                save = (save_truth == 'all' or (save_truth == 'one' and seed == seeds[0] and resolution == min(resolutions)))
                target = out / 'mesh_truth' / f'{seed}_{variant}_{resolution:.2f}.npz' if save else None
                rows.append(one_scene(config, seed, variant, resolution, save_npz=target))
        (out / 'cal091_progress.json').write_text(json.dumps({
            'format': FORMAT, 'completed_seeds': len({r['seed'] for r in rows}),
            'expected_seeds': len(seeds), 'completed_meshes': len(rows),
            'source_config_sha256': config_hash}, indent=2), encoding='utf-8')
    doc = summarize(rows, seeds=seeds, resolutions=resolutions)
    doc['source_config_sha256'] = config_hash
    (out / 'cal091_results.json').write_text(json.dumps(doc, indent=2, ensure_ascii=False,
                                                       allow_nan=False), encoding='utf-8')
    (out / 'CAL091_REPORT.md').write_text(markdown(doc), encoding='utf-8')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=Path('configs/customer_like.yaml'))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    parser.add_argument('--resolutions', type=float, nargs='+', default=list(RESOLUTIONS_M))
    parser.add_argument('--save-truth', choices=['none', 'one', 'all'], default='one')
    args = parser.parse_args()
    result = run(args.out, args.config, args.seeds, args.resolutions, args.save_truth)
    print(markdown(result), flush=True)


if __name__ == '__main__':
    main()
