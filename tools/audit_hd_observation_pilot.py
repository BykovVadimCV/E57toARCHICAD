"""Pixel-level audit for a small native-HD dataset pilot.

This is a *data availability* audit, not a claim that a return near a wall is
semantically confirmed structural evidence or that a model does not hallucinate.
Unobserved pixel != empty space; direct-ray evidence is height-specific.

Usage:
    python tools/audit_hd_observation_pilot.py /path/to/pilot --out /path/to/report
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASS_FILES = {
    'wall': ('wall_mask_hd.png', 255),
    'door': ('openings_hd.png', 1),
    'window': ('openings_hd.png', 2),
    'passage': ('openings_hd.png', 3),
    'floor_void': ('floor_void_hd.png', 255),
}
BINARY_MASKS = {
    'legacy_valid': 'supervision_valid_hd.png',
    'training_valid': 'training_valid_hd.png',
    'return_proximity': 'observation_support_hd.png',
    'observed_return': 'observation_occupied_hd.png',
    'ray_free': 'observation_free_hd.png',
    'unknown': 'observation_unknown_hd.png',
    'unsupported_positive': 'observation_uncertain_hd.png',
    'confirmed_passage': 'observation_confirmed_passage_hd.png',
    'face_candidate': 'wall_face_candidates_hd.png',
    'surface_observed': 'observed_wall_surface_hd.png',
}


def _read(path: Path, shape: tuple[int, int]) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 2 or im.shape != shape or im.dtype != np.uint8:
        raise ValueError(f'Invalid uint8 mask (expected {shape}): {path}')
    return im


def _ratio(a: int, b: int) -> float | None:
    return round(a / b, 6) if b else None


def inspect_level(path: Path) -> dict:
    hd = json.loads((path / 'wall_mask_hd.json').read_text(encoding='utf-8'))
    evidence = json.loads((path / 'scan_evidence_hd.json').read_text(encoding='utf-8'))
    frame = hd['frame']
    shape = (int(frame['height']), int(frame['width']))
    if frame['pixel_m'] != 0.002:
        raise ValueError(f'Pilot must use native 2 mm masks: {path}')
    obs = evidence['observation']
    if obs['format'] != 'native-hd-observation-v3':
        raise ValueError(f'Unsupported observation contract: {path}')
    if not obs.get('ray_provenance', {}).get('available'):
        raise ValueError(f'Missing station-linked direct-ray evidence: {path}')
    masks = {key: _read(path / filename, shape) for key, filename in BINARY_MASKS.items()}
    for key, mask in masks.items():
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError(f'Non-binary mask: {path}/{key}')
    classes = {name: _read(path / filename, shape) == value
               for name, (filename, value) in CLASS_FILES.items()}
    positives = np.zeros(shape, dtype=bool)
    for name, class_mask in classes.items():
        if np.any(positives & class_mask):
            raise ValueError(f'Overlapping physical classes in {path}: {name}')
        positives |= class_mask
    valid = masks['training_valid'] != 0
    legacy = masks['legacy_valid'] != 0
    proximity = masks['return_proximity'] != 0
    confirmed = masks['confirmed_passage'] != 0
    face_candidate = masks['face_candidate'] != 0
    surface_observed = masks['surface_observed'] != 0
    unresolved = masks['unsupported_positive'] != 0
    occupied = masks['observed_return'] != 0
    ray_free = masks['ray_free'] != 0
    unknown = masks['unknown'] != 0
    if np.any(valid & ~legacy):
        raise ValueError(f'Training validity exceeds old supervision in {path}')
    if np.any(valid & unresolved):
        raise ValueError(f'Unsupported positive pixels erroneously trained: {path}')
    if np.any(confirmed & ~classes['passage']):
        raise ValueError(f'Confirmed passage outside passage class: {path}')
    if np.any(face_candidate & ~classes['wall']):
        raise ValueError(f'Wall face candidates outside physical wall body: {path}')
    if np.any(surface_observed & ~(face_candidate & proximity)):
        raise ValueError(f'Fabricated wall surface without sensor proximity: {path}')
    if np.any(valid & classes['wall'] & ~surface_observed):
        raise ValueError(f'Unobserved physical wall interior admitted as a training positive: {path}')
    if np.any(valid & classes['passage'] & ~confirmed):
        raise ValueError(f'Passage proximity, not rays, admitted to positive training: {path}')
    supported = (surface_observed
                 | (((classes['door'] | classes['window'] | classes['floor_void'])) & proximity)
                 | confirmed)
    if np.any(unresolved != (positives & ~supported)):
        raise ValueError(f'Unsupported-positive mask inconsistent: {path}')
    if np.any(unknown != ~(occupied | ray_free)):
        raise ValueError(f'Unknown mask does not match return/ray evidence: {path}')

    counts = {k: int(np.count_nonzero(m)) for k, m in masks.items()}
    metrics = {
        'positive': int(positives.sum()),
        'positive_kept': int((positives & valid).sum()),
        'unsupported_positive': counts['unsupported_positive'],
        'legacy_valid': counts['legacy_valid'],
        'training_valid': counts['training_valid'],
        'negative_kept': int((~positives & valid).sum()),
        'negative_legacy': int((~positives & legacy).sum()),
        'confirmed_passage': counts['confirmed_passage'],
        'occupied': counts['observed_return'],
        'free_rays': counts['ray_free'],
        'unknown': counts['unknown'],
        'return_proximity': counts['return_proximity'],
        'wall_face_candidates': counts['face_candidate'],
        'wall_surface_observed': counts['surface_observed'],
        'wall_surface_trained': int(np.count_nonzero(surface_observed & valid)),
        'pixels': int(shape[0] * shape[1]),
    }
    for key, n in [
        ('observed_return_pixels', counts['observed_return']),
        ('return_proximity_pixels', counts['return_proximity']),
        ('unsupported_positive_pixels', counts['unsupported_positive']),
        ('training_valid_pixels', counts['training_valid']),
        ('confirmed_passage_pixels', counts['confirmed_passage']),
        ('free_ray_pixels', counts['ray_free']),
        ('wall_face_candidate_pixels', counts['face_candidate']),
        ('observed_wall_surface_pixels', counts['surface_observed']),
    ]:
        if int(obs[key]) != n:
            raise ValueError(f'Mismatch in sidecar count for {key}: {path}: {obs[key]} != {n}')
    class_stats = {}
    for name, mask in classes.items():
        total = int(mask.sum())
        kept = int(np.count_nonzero(mask & valid))
        class_stats[name] = {
            'physical_px': total,
            'trained_px': kept,
            'proximate_px': int(np.count_nonzero(mask & proximity)),
            'kept_fraction': _ratio(kept, total),
            'area_m2': round(total * float(frame['pixel_m']) ** 2, 6),
        }
        if name == 'wall':
            eligible_faces = int(np.count_nonzero(face_candidate & legacy))
            class_stats[name]['face_candidate_px'] = int(face_candidate.sum())
            class_stats[name]['face_candidate_legacy_valid_px'] = eligible_faces
            class_stats[name]['surface_observed_px'] = int(surface_observed.sum())
            class_stats[name]['surface_trained_px'] = kept
            class_stats[name]['face_coverage_fraction'] = _ratio(kept, eligible_faces)

    objects = obs.get('objects', {})
    walls = objects.get('walls', [])
    aperture_objects = objects.get('openings', [])
    crossing = objects.get('passage_ray_crossings', {})
    crossing_items = crossing.get('openings', [])
    n_confirmed = sum(int(x['confirmed_for_supervision']) for x in crossing_items)
    if n_confirmed != crossing.get('confirmed_count', 0):
        raise ValueError(f'Passage confirmation sidecar mismatch: {path}')
    from simscan.ray_evidence import MAX_PASSAGE_PROOF_FRACTION
    for item in crossing_items:
        total = int(item.get('physical_area_px', -1))
        proof = int(item.get('confirmed_area_px', -1))
        if (total < 0 or proof < 0 or proof > MAX_PASSAGE_PROOF_FRACTION * total
                or (not item.get('confirmed_for_supervision') and proof)
                or (item.get('confirmed_for_supervision') and not proof)
                or len(item.get('representative_ray_offsets_m', [])) > 3):
            raise ValueError(f'Passage per-opening proof area exceeds physical evidence cap: {path}')
    if sum(int(x['confirmed_area_px']) for x in crossing_items) < int(confirmed.sum()):
        raise ValueError(f'Confirmed passage raster not accounted for by aperture audit: {path}')
    for key, expected in [('wall_count', len(walls)), ('opening_count', len(aperture_objects))]:
        if objects.get(key) != expected:
            raise ValueError(f'Object sidecar mismatch ({key}): {path}')
    row = {
        'level': path.parent.name + '/' + path.name,
        'shape': f'{shape[1]}x{shape[0]}',
        'pixels': metrics['pixels'],
        'metrics': metrics,
        'classes': class_stats,
        'walls': len(walls),
        'walls_both_face_proximity_mean': (
            round(float(np.mean([w['both_faces_covered_fraction_upper_bound'] for w in walls])), 5)
            if walls else None),
        'openings_total': len(aperture_objects),
        'openings_both_jambs': sum(bool(x['both_jambs_near_returns']) for x in aperture_objects),
        'passage_objects': len(crossing_items),
        'passage_objects_confirmed': n_confirmed,
        'passage_proof_per_opening': crossing_items,
        'ray_count': int(obs['ray_provenance'].get('ray_count', 0)),
        'note': 'Return proximity is NOT proof of architectural object identity.',
    }
    return row


def summarize(levels: list[dict], expected_scenes: int | None = None) -> dict:
    scenes = {row['level'].split('/')[0] for row in levels}
    aggregate = Counter()
    by_class = defaultdict(Counter)
    for row in levels:
        aggregate.update(row['metrics'])
        aggregate['wall_objects'] += row['walls']
        aggregate['opening_objects'] += row['openings_total']
        aggregate['opening_objects_both_jambs'] += row['openings_both_jambs']
        aggregate['passage_objects'] += row['passage_objects']
        aggregate['passage_objects_confirmed'] += row['passage_objects_confirmed']
        aggregate['ray_count'] += row['ray_count']
        for name, cls in row['classes'].items():
            by_class[name].update({k: v for k, v in cls.items()
                                   if k in {'physical_px', 'trained_px', 'proximate_px',
                                            'face_candidate_px', 'face_candidate_legacy_valid_px',
                                            'surface_observed_px', 'surface_trained_px'}})
    summary = {
        'scene_count': len(scenes), 'level_count': len(levels),
        'metrics': dict(aggregate),
        'class_retention': {
            k: {**dict(v), 'trained_fraction': _ratio(v['trained_px'], v['physical_px']),
                'return_proximity_fraction': _ratio(v['proximate_px'], v['physical_px'])}
            for k, v in sorted(by_class.items())
        },
        'positive_training_retention': _ratio(aggregate['positive_kept'], aggregate['positive']),
        'unsupported_positive_fraction': _ratio(aggregate['unsupported_positive'], aggregate['positive']),
        'passage_object_confirmation_fraction': _ratio(
            aggregate['passage_objects_confirmed'], aggregate['passage_objects']),
        'negative_background_retention': _ratio(aggregate['negative_kept'], aggregate['negative_legacy']),
        'audit_kind': 'synthetic_observation_support_diagnostic_not_model_quality',
    }
    warnings = []
    if expected_scenes is not None and len(scenes) != expected_scenes:
        warnings.append(f'Expected {expected_scenes} scenes, found {len(scenes)}')
    if aggregate['passage_objects'] == 0:
        warnings.append('No passage objects: this pilot cannot assess passage confirmation')
    elif aggregate['passage_objects_confirmed'] == 0:
        warnings.append('No confirmed passages; inspect threshold/provenance, not merely passage IoU')
    if not aggregate['positive']:
        warnings.append('No architectural positive pixels')
    if aggregate['positive'] and aggregate['positive_kept'] == 0:
        warnings.append('All architectural positive pixels excluded from training')
    wall = summary['class_retention'].get('wall', {})
    face_coverage = _ratio(wall.get('surface_trained_px', 0),
                           wall.get('face_candidate_legacy_valid_px', 0))
    summary['wall_face_training_coverage'] = face_coverage
    # Low physical-body retention is *expected* because the body is not the
    # supervision target. Report the ratio but do not treat it as a failure.
    if face_coverage is not None and face_coverage < .35:
        warnings.append('Low measured wall-face coverage (<35%): insufficient observed face targets')
    passages = by_class.get('passage', {})
    passage_pixels = passages.get('physical_px', 0)
    confirmed_px_fraction = _ratio(aggregate['confirmed_passage'], passage_pixels)
    summary['confirmed_passage_area_fraction'] = confirmed_px_fraction
    if confirmed_px_fraction is not None and confirmed_px_fraction > .20:
        warnings.append('Passage ray-confirmed pixels cover >20% of passage area: per-opening proof area cap violated')
    if by_class.get('floor_void', {}).get('physical_px', 0) == 0:
        warnings.append('No floor-void targets: multi-level / upper slab holes were not tested')
    # Even a defect-free *diagnostic* synthetic pilot cannot certify that the
    # dataset will perform on a real E57. Keep this gate explicit.
    summary['quality_gate'] = {
        'mask_consistency': 'passed',
        'approved_for_training': False,
        'reason': 'diagnostic synthetic pilot; requires full-density and real-E57 validation before training',
    }
    summary['warnings'] = warnings
    return summary


def _read_entries(root: Path) -> list[Path]:
    paths = []
    for split in ('train', 'val', 'test'):
        file = root / 'splits' / (split + '.txt')
        if not file.is_file():
            raise ValueError(f'Missing split manifest: {file}')
        for entry in file.read_text(encoding='utf-8').splitlines():
            if not entry.strip():
                continue
            candidate = root / 'scenes' / entry
            if '..' in Path(entry).parts or candidate.is_symlink() or not candidate.is_dir():
                raise ValueError(f'Invalid level reference: {entry}')
            paths.append(candidate)
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate level in split manifest')
    return paths


def audit_roots(roots: list[Path], expected_scenes: int | None = 10,
                check_integrity: bool = True) -> tuple[dict, list[dict]]:
    from audit_hd_dataset import audit
    all_levels = []
    integrity = {}
    for root in roots:
        if check_integrity:
            integrity[root.name] = audit(root, deep=True)
        paths = _read_entries(root)
        for path in paths:
            row = inspect_level(path)
            row['cohort'] = root.name
            all_levels.append(row)
    summary = summarize(all_levels, expected_scenes)
    summary['cohorts'] = integrity
    summary['cohort_statistics'] = {
        root.name: summarize([row for row in all_levels if row['cohort'] == root.name], None)
        for root in roots
    }
    return summary, all_levels


def save_report(summary: dict, levels: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'pilot_quality.json').write_text(json.dumps(
        {'summary': summary, 'levels': levels}, ensure_ascii=False, indent=2), encoding='utf-8')
    columns = ['cohort', 'level', 'shape', 'walls', 'openings_total', 'openings_both_jambs',
               'passage_objects', 'passage_objects_confirmed', 'ray_count',
               'positive_training_retention', 'unsupported_positive_fraction',
               'wall_retention', 'wall_face_coverage', 'door_retention', 'window_retention',
               'passage_retention', 'floor_void_retention']
    with (output_dir / 'pilot_levels.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in levels:
            m = row['metrics']
            w.writerow({'cohort': row['cohort'], 'level': row['level'], 'shape': row['shape'],
                'walls': row['walls'], 'openings_total': row['openings_total'],
                'openings_both_jambs': row['openings_both_jambs'],
                'passage_objects': row['passage_objects'],
                'passage_objects_confirmed': row['passage_objects_confirmed'],
                'ray_count': row['ray_count'],
                'positive_training_retention': _ratio(m['positive_kept'], m['positive']),
                'wall_face_coverage': row['classes']['wall']['face_coverage_fraction'],
                'unsupported_positive_fraction': _ratio(m['unsupported_positive'], m['positive']),
                **{key + '_retention': row['classes'][key]['kept_fraction'] for key in CLASS_FILES}})
    s = summary
    lines = [
        '# Native-HD observation pilot (diagnostic only)', '',
        f"Scenes: **{s['scene_count']}**, levels: **{s['level_count']}**.", '',
        f"Physical GT pixel fraction retained (not wall training coverage): **{s['positive_training_retention']}**.",
        f"Measured wall-face coverage (training target): **{s['wall_face_training_coverage']}**.",
        f"Ray-confirmed passage area fraction: **{s['confirmed_passage_area_fraction']}** (cap <=20% per opening).",
        f"Unsupported physical positives: **{s['unsupported_positive_fraction']}**.",
        f"Background-validity retention: **{s['negative_background_retention']}**.",
        f"Physical passage objects confirmed by rays: **{s['metrics']['passage_objects_confirmed']} / {s['metrics']['passage_objects']}**.",
        '', '## Per-class retention', '',
        '| Class | Physical pixels | Train-valid pixels | Fraction of physical GT |',
        '|---|---:|---:|---:|',
    ]
    for k, v in s['class_retention'].items():
        lines.append(f"| {k} | {v['physical_px']} | {v['trained_px']} | {v['trained_fraction']} |")
    lines += ['', '## Cohorts (must not be confused with production distribution)', '',
              '| Cohort | Scenes | Physical positive kept | Wall-face coverage | Passages confirmed |',
              '|---|---:|---:|---:|---:|']
    for cohort, c in s.get('cohort_statistics', {}).items():
        lines.append(f"| {cohort} | {c['scene_count']} | {c['positive_training_retention']} | {c['wall_face_training_coverage']} | {c['metrics']['passage_objects_confirmed']}/{c['metrics']['passage_objects']} |")
    lines += ['', '## Warnings', '']
    lines += [f'- {w}' for w in s['warnings']] if s['warnings'] else ['No structural audit warnings.']
    lines += ['', '## Interpretation limits', '',
        '- The wall training target is the observed **surface**, not full wall volume.',
        '- Return proximity may be furniture, glass or other nonstructural matter.',
        '- A free-ray pixel confirms a sampled 3D path at *some* height, not a whole empty column.',
        '- An unknown pixel does not prove absence of walls or openings.',
        '- This does not measure network hallucinations, IoU, real E57 accuracy or Archicad export.',
        '- Diagnostic scanner settings are intentionally cheaper than the 0.1° production profile.',
        '- **Training approval: NO**, even when data integrity passes.', '']
    (output_dir / 'PILOT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots', nargs='+', type=Path, help='One or more compatible cohort roots')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--expected-scenes', type=int, default=10)
    a = p.parse_args()
    summary, levels = audit_roots(a.roots, a.expected_scenes)
    save_report(summary, levels, a.out)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('cohorts',)}, indent=2))
    if summary['scene_count'] != a.expected_scenes:
        raise SystemExit('Pilot scene count mismatch')


if __name__ == '__main__':
    main()
