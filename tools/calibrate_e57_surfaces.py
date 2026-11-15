"""CAL-02: bounded-memory structural planar-patch audit of original E57 files.

This is *not* an estimator of intrinsic scanner ranging noise: a merged E57
may contain different stations and an observed surface can be rough or warped.
Use the emitted local plane thicknesses to compare matched real/synthetic
surfaces, never to silently overwrite scanner parameters.

Usage:
    PYTHONPATH=. python tools/calibrate_e57_surfaces.py \
      --e57 /path/to/scan.e57 --out /path/to/CAL02 --max-sample 250000
    PYTHONPATH=. python tools/calibrate_e57_surfaces.py \
      --sample-npz wall_fixture.npz --out /path/to/CAL02

Requires numpy. Raw E57 reading uses the repository's own streaming reader;
no point arrays from a large E57 are loaded into RAM all at once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

FORMAT = 'e57-local-wall-surfaces-calibration-v1'


class PointReservoir:
    """Priority reservoir with uniform inclusion chance for streamed finite xyz."""
    def __init__(self, capacity: int, seed: int = 42):
        if capacity < 1000:
            raise ValueError('capacity must be >=1000')
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.points = np.empty((0, 3), dtype=np.float64)
        self.keys = np.empty(0, dtype=np.float64)
        self.seen = 0
        self.valid = 0

    def add(self, xyz: np.ndarray, valid: np.ndarray | None = None) -> None:
        pts = np.asarray(xyz, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError('xyz must have shape (n,3)')
        self.seen += len(pts)
        mask = np.isfinite(pts).all(axis=1)
        if valid is not None:
            mask &= np.asarray(valid, dtype=bool)
        pts = pts[mask]
        self.valid += len(pts)
        if not len(pts):
            return
        keys = self.rng.random(len(pts))
        if len(self.keys) == self.capacity:
            keep = keys < self.keys.max()
            pts, keys = pts[keep], keys[keep]
            if not len(pts):
                return
        combined_p = np.concatenate([self.points, pts], axis=0)
        combined_k = np.concatenate([self.keys, keys])
        if len(combined_k) > self.capacity:
            indices = np.argpartition(combined_k, self.capacity - 1)[:self.capacity]
            combined_p = combined_p[indices]
            combined_k = combined_k[indices]
        self.points, self.keys = combined_p, combined_k

    @property
    def sampling_fraction(self) -> float:
        return len(self.points) / max(self.valid, 1)


def fit_vertical_plane(points: np.ndarray, max_abs_normal_z: float = 0.20) -> dict | None:
    """Trimmed 3D total-least-squares plane; residual includes material+registration.

    Does not return a value named 'range_sigma': that would be a false claim.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 40 or pts.ndim != 2 or pts.shape[1] != 3:
        return None
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if len(pts) < 40:
        return None
    inliers = pts
    # Robust iteration initialized by eigendecomposition, without projecting
    # onto a global perfect wall body.
    for _ in range(4):
        center = inliers.mean(0)
        cov = (inliers - center).T @ (inliers - center) / len(inliers)
        eig, vec = np.linalg.eigh(cov)
        normal = vec[:, int(np.argmin(eig))]
        if abs(float(normal[2])) > max_abs_normal_z:
            return None
        residual_all = (pts - center) @ normal
        med = float(np.median(residual_all))
        scale = 1.4826 * float(np.median(np.abs(residual_all - med)))
        threshold = min(0.05, max(0.007, 3.5 * scale))
        new = pts[np.abs(residual_all - med) <= threshold]
        if len(new) < 40:
            break
        inliers = new
    center = inliers.mean(0)
    eig, vec = np.linalg.eigh((inliers - center).T @ (inliers - center) / len(inliers))
    normal = vec[:, int(np.argmin(eig))]
    if abs(float(normal[2])) > max_abs_normal_z:
        return None
    r = (inliers - center) @ normal
    med = float(np.median(r))
    mad = 1.4826 * float(np.median(np.abs(r - med)))
    return {
        'inlier_points': int(len(inliers)),
        'source_points': int(len(pts)),
        'inlier_fraction': round(len(inliers) / len(pts), 5),
        'normal_z_abs': round(abs(float(normal[2])), 6),
        'observed_plane_mad_mm': round(mad * 1000, 4),
        'observed_plane_rms_mm': round(float(np.sqrt(np.mean(r * r))) * 1000, 4),
        'observed_plane_abs_p95_mm': round(float(np.percentile(np.abs(r), 95)) * 1000, 4),
    }


def _refit_xy_line(xy: np.ndarray) -> tuple[np.ndarray, float]:
    cen = xy.mean(axis=0)
    C = xy - cen
    eig, vec = np.linalg.eigh(C.T @ C / len(C))
    n = vec[:, 0]
    return n, -float(n @ cen)


def _discover_wall_lines(points: np.ndarray, max_lines: int, seed: int,
                         threshold_m: float = 0.025) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """2D RANSAC for long vertical wall planes, not a full wall segmentation."""
    if not 0.005 <= threshold_m <= 0.1:
        raise ValueError('threshold_m outside plausible wall-fitting range')
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 1000:
        return []
    zlo, zhi = np.quantile(pts[:, 2], [0.08, 0.92])
    # Remove the dominant floor and ceiling slices to reduce false walls.
    margin = min(0.55, max(0.15, float(zhi - zlo) * 0.15))
    eligible = np.flatnonzero((pts[:, 2] > zlo + margin) & (pts[:, 2] < zhi - margin))
    if len(eligible) < 500:
        eligible = np.arange(len(pts))
    rng = np.random.default_rng(seed)
    remaining = eligible
    walls = []
    for _ in range(max_lines):
        if len(remaining) < 500:
            break
        test = rng.choice(remaining, size=min(len(remaining), 18000), replace=False)
        xy = pts[test, :2]
        best_count, best = 0, None
        for _ in range(340):
            a, b = rng.integers(0, len(xy), size=2)
            d = xy[b] - xy[a]
            norm = float(np.linalg.norm(d))
            if norm < 0.9:
                continue
            n = np.array([-d[1], d[0]], dtype=float) / norm
            bias = -float(n @ xy[a])
            votes = np.abs(xy @ n + bias) < threshold_m
            count = int(votes.sum())
            if count > best_count:
                best_count, best = count, (n, bias)
        if best is None or best_count < 125:
            break
        n, bias = best
        inliers = remaining[np.abs(pts[remaining, :2] @ n + bias) < threshold_m]
        if len(inliers) < 220:
            break
        # Require a substantial vertical surface, not a floor slice line.
        if np.ptp(np.quantile(pts[inliers, 2], [0.05, 0.95])) < 0.9:
            remaining = remaining[~np.isin(remaining, inliers)]
            continue
        n, bias = _refit_xy_line(pts[inliers, :2])
        inliers = remaining[np.abs(pts[remaining, :2] @ n + bias) < threshold_m]
        tang = np.array([-n[1], n[0]])
        if np.ptp(np.quantile(pts[inliers, :2] @ tang, [0.05, 0.95])) < 1.3:
            remaining = remaining[~np.isin(remaining, inliers)]
            continue
        walls.append((n, bias, inliers))
        remaining = remaining[np.abs(pts[remaining, :2] @ n + bias) >= 1.5 * threshold_m]
    return walls


def assess_wall_patches(points: np.ndarray, max_lines: int = 5,
                        max_patches: int = 24, seed: int = 42,
                        threshold_m: float = 0.025) -> dict:
    """Report local wall-patch thickness, not device sigma nor ground truth."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('expected (n,3) points')
    pts = pts[np.isfinite(pts).all(axis=1)]
    if not len(pts):
        return {'sample_points': 0, 'candidate_wall_lines': 0, 'patches': [], 'summary': {}}
    lines = _discover_wall_lines(pts, max_lines=max_lines, seed=seed, threshold_m=threshold_m)
    patches = []
    for li, (normal, bias, ids) in enumerate(lines):
        wall = pts[ids]
        tangent = np.array([-normal[1], normal[0]])
        along = wall[:, :2] @ tangent
        z = wall[:, 2]
        tmin = float(np.quantile(along, 0.03))
        tmax = float(np.quantile(along, 0.97))
        zmin = float(np.quantile(z, 0.03))
        zmax = float(np.quantile(z, 0.97))
        for s0 in np.arange(tmin, tmax, 0.9):
            for z0 in np.arange(zmin, zmax, 1.25):
                sel = (along >= s0) & (along < s0 + 0.9) & (z >= z0) & (z < z0 + 1.25)
                sub = wall[sel]
                if len(sub) < 110:
                    continue
                span_t = float(np.quantile(along[sel], 0.95) - np.quantile(along[sel], 0.05))
                span_z = float(np.quantile(z[sel], 0.95) - np.quantile(z[sel], 0.05))
                if span_t < 0.4 or span_z < 0.65:
                    continue
                res = fit_vertical_plane(sub)
                if res is None or res['inlier_fraction'] < 0.55:
                    continue
                patches.append({'wall_candidate_index': li,
                                'span_length_m': round(span_t, 3),
                                'span_height_m': round(span_z, 3),
                                **res})
    # Prefer well-observed patches but include different wall candidates where possible.
    patches.sort(key=lambda p: (p['inlier_points'], -p['observed_plane_mad_mm']), reverse=True)
    patches = patches[:max_patches]
    def med(key):
        return round(float(np.median([p[key] for p in patches])), 4) if patches else None
    return {'sample_points': int(len(pts)), 'candidate_wall_lines': len(lines),
            'patches': patches, 'summary': {
                'n_patches': len(patches),
                'median_observed_plane_mad_mm': med('observed_plane_mad_mm'),
                'median_observed_plane_rms_mm': med('observed_plane_rms_mm'),
                'median_observed_plane_abs_p95_mm': med('observed_plane_abs_p95_mm'),
            }, 'limitations': [
                'Patch residual combines surface roughness, registration, mixed materials and range noise.',
                'Wall candidates are RANSAC hypotheses; they are not ground-truth architecture.',
                'Uniform sampling changes point density; do not use sampled point counts as density.',
                'No independent scanner station or angular step is inferred from a merged E57 node.',
            ]}


def _scan_header(s) -> dict:
    fields = s.field_names
    return {
        'data3d_index': int(s.index), 'point_count': int(s.points),
        'fields': fields, 'row_and_column_indices_available': 'rowIndex' in fields and 'columnIndex' in fields,
        'pose_rotation_available': s.rotation is not None,
        'pose_translation_available': s.translation is not None,
        'index_bounds_available': bool(s.index_bounds),
        'one_data3d_node_is_one_physical_station': 'unverified',
    }


def analyze_sample(xyz: np.ndarray, seed: int = 42, max_lines: int = 5,
                   max_patches: int = 24) -> dict:
    sample = assess_wall_patches(xyz, seed=seed, max_lines=max_lines, max_patches=max_patches)
    return {'format': FORMAT, 'source_type': 'supplied_xyz_sample', 'surfaces': sample,
            'calibration_decision': 'DO_NOT_FIT_SCANNER_NOISE_FROM_UNREGISTERED_OR_MERGED_PLANE_RESIDUALS'}


def read_e57_metadata(path: Path) -> dict:
    """Fast XML/header-only read: no point decompression, no station inference."""
    from simscan.e57read import E57Reader
    path = Path(path)
    with E57Reader(path) as reader:
        scans = [_scan_header(s) for s in reader.scans]
    return {'format': FORMAT, 'source_type': 'original_e57_metadata_only',
            'file_name': path.name, 'file_size_bytes': path.stat().st_size,
            'data3d_node_count': len(scans), 'scans': scans,
            'calibration_decision': 'NO_AUTOMATIC_SCANNER_PARAMETER_CHANGE',
            'limitations': ['Metadata scan count does not prove a physical station count.',
                            'A registration/export process may have merged original stations.']}


def analyze_e57(path: Path, max_sample: int = 250000, chunk: int = 250000,
                max_lines: int = 5, max_patches: int = 24, seed: int = 42) -> dict:
    from simscan.e57read import E57Reader
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    scans = []
    with E57Reader(path) as reader:
        for scan in reader.scans:
            entry = _scan_header(scan)
            if not {'cartesianX', 'cartesianY', 'cartesianZ'}.issubset(scan.field_names):
                entry['error'] = 'missing_cartesian_fields'
                scans.append(entry)
                continue
            reservoir = PointReservoir(max_sample, seed=seed + scan.index)
            use_invalid_flag = 'cartesianInvalidState' in scan.field_names
            fields = ['cartesianX', 'cartesianY', 'cartesianZ']
            if use_invalid_flag:
                fields.append('cartesianInvalidState')
            for batch in reader.iter_points(scan.index, fields, chunk=chunk):
                xyz = np.column_stack([batch[k] for k in ('cartesianX', 'cartesianY', 'cartesianZ')])
                valid = batch.get('cartesianInvalidState', np.zeros(len(xyz), np.int64)) == 0
                reservoir.add(xyz, valid=valid)
            entry.update({'records_seen': reservoir.seen,
                          'valid_cartesian_points': reservoir.valid,
                          'reservoir_points': len(reservoir.points),
                          'reservoir_fraction': round(reservoir.sampling_fraction, 6),
                          'surfaces': assess_wall_patches(reservoir.points,
                                      max_lines=max_lines, max_patches=max_patches,
                                      seed=seed + scan.index)})
            scans.append(entry)
    return {'format': FORMAT, 'source_type': 'original_e57_streamed',
            'file_name': path.name, 'file_size_bytes': path.stat().st_size,
            'data3d_node_count': len(scans), 'scans': scans,
            'configuration': {'max_sample_per_data3d_node': max_sample,
                              'read_chunk_records': chunk, 'max_wall_lines': max_lines,
                              'max_patches': max_patches, 'seed': seed},
            'calibration_decision': 'NO_AUTOMATIC_SCANNER_PARAMETER_CHANGE',
            'limitations': [
                'A data3D node may be a merged export, not a single instrument station.',
                'Row/column indices and one pose, even if present, do not prove original station provenance.',
                'An original 3D surface may be rough; patch residual is an UPPER-BOUND-like mixed error statistic, NOT intrinsic range sigma.',
                'Sampling is uniform and bounded; absolute point density and angular resolution are not estimated.',
                'No new supervised geometry is inferred for unseen areas.',
            ]}


def report_md(documents: list[dict]) -> str:
    lines = ['# CAL-02: локальные плоскости оригинальных E57', '',
             'Статистика рассчитана по измеренным XYZ. Идеальная геометрия генератора не использовалась.', '',
             '| Файл | Узлы data3D | Сеточные индексы (по узлам) | Патчи | MAD, мм | RMS, мм |',
             '|---|---:|---|---:|---:|---:|']
    for d in documents:
        scans = d.get('scans', [])
        p = [s['surfaces']['summary'] for s in scans if 'surfaces' in s]
        if not scans and 'surfaces' in d:
            p = [d['surfaces']['summary']]
        vals = [s.get('median_observed_plane_mad_mm') for s in p if s.get('median_observed_plane_mad_mm') is not None]
        rms = [s.get('median_observed_plane_rms_mm') for s in p if s.get('median_observed_plane_rms_mm') is not None]
        fmt = lambda v: f'{float(np.median(v)):.2f}' if v else 'нет данных'
        name = d.get('file_name', 'XYZ fixture')
        grid = ', '.join('да' if s['row_and_column_indices_available'] else 'нет' for s in scans) or 'неизвестно'
        lines.append(f'| {name} | {len(scans)} | {grid} | {sum(s.get("n_patches", 0) for s in p)} | {fmt(vals)} | {fmt(rms)} |')
    lines.extend(['', '**Параметры `range_sigma_mm`, `registration_sigma_mm` и `angular_step_deg` не изменены.**', '',
                  'MAD и RMS относительно подогнанной плоскости включают неровность поверхности, смешение отражений и возможную погрешность регистрации. Это не чистый шум дальномера.',
                  'Для разделения этих причин нужны оригинальные станции, их траектории и контрольные плоские поверхности. Если файл представляет собой объединённое облако, эти причины могут быть неразделимы.',
                  '', 'Формат отчёта: `' + FORMAT + '`.'])
    return '\n'.join(lines) + '\n'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--e57', type=Path, nargs='+')
    src.add_argument('--sample-npz', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--max-sample', type=int, default=250000)
    p.add_argument('--metadata-only', action='store_true', help='E57 header only, no point decode')
    p.add_argument('--chunk', type=int, default=250000)
    p.add_argument('--max-lines', type=int, default=5)
    p.add_argument('--max-patches', type=int, default=24)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.max_sample < 1000 or a.chunk < 1000 or a.max_lines < 1 or a.max_patches < 1:
        p.error('sample/chunk >=1000, max-lines/max-patches >=1 required')
    if a.sample_npz:
        with np.load(a.sample_npz, allow_pickle=False) as z:
            sample = np.asarray(z['xyz'])
        docs = [analyze_sample(sample, seed=a.seed, max_lines=a.max_lines, max_patches=a.max_patches)]
    elif a.metadata_only:
        docs = [read_e57_metadata(path) for path in a.e57]
    else:
        docs = [analyze_e57(path, max_sample=a.max_sample, chunk=a.chunk,
                            max_lines=a.max_lines, max_patches=a.max_patches,
                            seed=a.seed) for path in a.e57]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out/'cal02_surface_calibration.json').write_text(json.dumps({'format': FORMAT, 'files': docs}, ensure_ascii=False, indent=2), encoding='utf8')
    txt = report_md(docs)
    (a.out/'CAL02_REPORT.md').write_text(txt, encoding='utf8')
    print(txt)


if __name__ == '__main__':
    main()
