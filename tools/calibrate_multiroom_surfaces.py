"""CAL-04: multi-room synthetic E57 return residuals by ray geometry and coverage.

All surface discovery, 3D planar residuals, and top-24 patch selection mirror
CAL-02. We augment the same measured patches with synthetic *known* station
ray ranges, estimated incidence to the fitted wall, and multi-station coverage.
These ray quantities are unavailable for the original merged real E57 files.
No GT surfaces or semantic class labels are used to choose wall patches.
No production scanner config is altered. Never infer a device sigma from MAD.

Run from the repository root:
  PYTHONPATH=. python tools/calibrate_multirooom_surfaces.py --out /tmp/cal04 \
    --seeds 42121 42122 42123
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from tools.calibrate_e57_surfaces import _discover_wall_lines, fit_vertical_plane, assess_wall_patches
from tools.calibrate_real_synthetic_surfaces import real_reference

VERSION = 'cal04-multiroom-observed-surfaces-v1'
DEFAULT_REAL = Path('configs/calibration_real_e57_surfaces.json')
RANGE_EDGES = (0.0, 2.0, 4.0, 6.0, 9.0, float('inf'))
ANGLE_EDGES = (0.0, 30.0, 60.0, 75.0, 90.00001)


class TaggedReservoir:
    """Uniform priority sample across ALL valid returns and ALL stations.

    The same finite/valid inclusion policy as CAL-02 is applied. Extra tags
    remain attached to the retained returns but are never persisted in reports.
    """
    def __init__(self, capacity=250000, seed=42):
        if int(capacity) < 1000:
            raise ValueError('capacity must be >=1000')
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.xyz = np.empty((0, 3), dtype=np.float64)
        self.origins = np.empty((0, 3), dtype=np.float64)
        self.ranges = np.empty(0, dtype=np.float64)
        self.stations = np.empty(0, dtype=np.int32)
        self.direct = np.empty(0, dtype=bool)
        self.keys = np.empty(0, dtype=np.float64)
        self.seen = self.valid = 0

    def add(self, xyz, origin, range_m, station_id, valid=None, direct=None):
        xyz = np.asarray(xyz, dtype=np.float64)
        ranges = np.asarray(range_m, dtype=np.float64)
        origin = np.asarray(origin, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or ranges.shape != (len(xyz),) or origin.shape != (3,):
            raise ValueError('Invalid xyz/range/origin shapes')
        n = len(xyz)
        if valid is None:
            valid = np.ones(n, dtype=bool)
        if direct is None:
            direct = np.ones(n, dtype=bool)
        valid = np.asarray(valid, dtype=bool)
        direct = np.asarray(direct, dtype=bool)
        if valid.shape != (n,) or direct.shape != (n,):
            raise ValueError('Invalid return flags')
        self.seen += n
        allowed = valid & np.isfinite(xyz).all(axis=1) & np.isfinite(ranges) & (ranges > 0)
        self.valid += int(np.count_nonzero(allowed))
        if not allowed.any():
            return
        # Generate priorities only for valid observations. This means chunk
        # segmentation does not alter the sample when records are ordered alike.
        x, d, rr = xyz[allowed], direct[allowed], ranges[allowed]
        keys = self.rng.random(len(x))
        if len(self.keys) == self.capacity:
            select = keys < self.keys.max()
            x, d, rr, keys = x[select], d[select], rr[select], keys[select]
            if not len(keys):
                return
        keys_all = np.concatenate((self.keys, keys))
        xyz_all = np.concatenate((self.xyz, x))
        range_all = np.concatenate((self.ranges, rr))
        origin_all = np.concatenate((self.origins, np.broadcast_to(origin, (len(keys), 3))))
        st_all = np.concatenate((self.stations, np.full(len(keys), station_id, dtype=np.int32)))
        direct_all = np.concatenate((self.direct, d))
        if len(keys_all) > self.capacity:
            ids = np.argpartition(keys_all, self.capacity - 1)[:self.capacity]
        else:
            ids = np.arange(len(keys_all))
        self.keys = keys_all[ids]
        self.xyz = xyz_all[ids]
        self.ranges = range_all[ids]
        self.origins = origin_all[ids]
        self.stations = st_all[ids]
        self.direct = direct_all[ids]

    @property
    def sampling_fraction(self):
        return len(self.xyz) / max(self.valid, 1)


def sample_registered_with_stations(scans, poses, max_sample=250000, seed=42):
    if len(scans) != len(poses):
        raise ValueError('Station scan/pose mismatch')
    out = TaggedReservoir(max_sample, seed=seed)
    for si, (scan, (rotation, translation)) in enumerate(zip(scans, poses)):
        R = np.asarray(rotation, float)
        t = np.asarray(translation, float)
        if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
            raise ValueError('Invalid written pose')
        xyz = np.asarray(scan.xyz_local)
        valid = np.asarray(scan.valid, bool)
        if xyz.ndim != 2 or xyz.shape != (len(valid), 3):
            raise ValueError('Invalid scan Cartesian dimensions')
        direct = getattr(scan, 'straight_valid', None)
        if direct is None:
            mixed = np.asarray(getattr(scan, 'mixed', np.zeros(len(valid), bool)), bool)
            virtual = np.asarray(getattr(scan, 'virtual', np.zeros(len(valid), bool)), bool)
            direct = valid & ~mixed & ~virtual
        else:
            direct = np.asarray(direct, bool)
        if direct.shape != valid.shape:
            raise ValueError('Direct path flags mismatch')
        for a in range(0, len(xyz), 200000):
            v = xyz[a:a+200000]
            out.add(v @ R.T + t, t, np.linalg.norm(v, axis=1), si,
                    valid[a:a+200000], direct[a:a+200000])
    return out


def _bin(value, edges, unit):
    if not np.isfinite(value) or value < edges[0] or value >= edges[-1]:
        return None
    i = int(np.searchsorted(edges, value, side='right') - 1)
    if i < 0 or i >= len(edges)-1:
        return None
    a, b = edges[i:i+2]
    return f'{a:g}–{b:g} {unit}' if np.isfinite(b) else f'{a:g}+ {unit}'


def _median(values):
    return round(float(np.median(values)), 4) if values else None


def patch_evidence(sample: TaggedReservoir, *, seed=42, max_lines=5, max_patches=24,
                   threshold_m=0.025):
    """Same CAL-02 local patches, plus tags from observed synthetic rays.

    Incidence is relative to a wall normal *estimated from measured XYZ*.
    An overlap requires >=25 direct returns and >=10% share from each station.
    """
    xyz = sample.xyz
    if not len(xyz):
        return {'patches': [], 'summary': {'n_patches': 0}}
    lines = _discover_wall_lines(xyz, max_lines=max_lines, seed=seed, threshold_m=threshold_m)
    patches = []
    for li, (normal, bias, ids) in enumerate(lines):
        wall = xyz[ids]
        tangent = np.array([-normal[1], normal[0]])
        along = wall[:, :2] @ tangent
        zz = wall[:, 2]
        tmin, tmax = np.quantile(along, [.03,.97])
        zmin, zmax = np.quantile(zz, [.03,.97])
        for s0 in np.arange(tmin, tmax, 0.9):
            for z0 in np.arange(zmin, zmax, 1.25):
                sel = (along >= s0) & (along < s0+.9) & (zz >= z0) & (zz < z0+1.25)
                sub = wall[sel]
                if len(sub) < 110:
                    continue
                span_t = float(np.quantile(along[sel], .95)-np.quantile(along[sel], .05))
                span_z = float(np.quantile(zz[sel], .95)-np.quantile(zz[sel], .05))
                if span_t < .4 or span_z < .65:
                    continue
                res = fit_vertical_plane(sub)
                if res is None or res['inlier_fraction'] < .55:
                    continue
                subids = ids[sel]
                direct = sample.direct[subids]
                n_direct = int(direct.sum())
                row = {'wall_candidate_index': li, 'span_length_m': round(span_t,3),
                       'span_height_m': round(span_z,3), **res,
                       'direct_return_fraction': round(n_direct / len(sub), 4),
                       'direct_return_count_sampled': n_direct,
                       'median_range_m': None, 'median_incidence_deg': None,
                       'range_bin': None, 'incidence_bin': None,
                       'well_supported_stations': 0, 'coverage': 'unclassified',
                       'interstation_normal_offset_span_mm': None}
                if n_direct:
                    picks = subids[direct]
                    ranges = sample.ranges[picks]
                    dr = xyz[picks] - sample.origins[picks]
                    dr_norm = np.linalg.norm(dr, axis=1)
                    ok = dr_norm > 1e-6
                    cos = np.abs((dr[ok, :2] / dr_norm[ok, None]) @ normal)
                    incident = np.degrees(np.arccos(np.clip(cos, 0, 1)))
                    if len(incident):
                        row['median_range_m'] = round(float(np.median(ranges[ok])),4)
                        row['median_incidence_deg'] = round(float(np.median(incident)),4)
                        row['range_bin'] = _bin(row['median_range_m'], RANGE_EDGES, 'm')
                        row['incidence_bin'] = _bin(row['median_incidence_deg'], ANGLE_EDGES, 'deg')
                    unique, counts = np.unique(sample.stations[picks], return_counts=True)
                    covered = unique[(counts >= 25) & (counts >= max(1, int(np.ceil(.1*len(picks)))))]
                    row['well_supported_stations'] = int(len(covered))
                    row['coverage'] = 'multi_station' if len(covered)>=2 else ('single_station' if len(covered)==1 else 'insufficient_support')
                    if len(covered)>=2:
                        # Relative fitted wall-normal station offsets. This is
                        # not a sensor registration-sigma estimate: wall patch
                        # and incidence differ between stations.
                        offsets=[]
                        for sid in covered:
                            p=xyz[picks[sample.stations[picks] == sid], :2]
                            offsets.append(float(np.median(p @ normal + bias)))
                        row['interstation_normal_offset_span_mm'] = round((max(offsets)-min(offsets))*1000,4)
                patches.append(row)
    patches.sort(key=lambda p:(p['inlier_points'],-p['observed_plane_mad_mm']),reverse=True)
    patches=patches[:max_patches]
    return {'patches':patches,'summary': {
        'n_patches':len(patches),
        'median_observed_plane_mad_mm':_median([p['observed_plane_mad_mm'] for p in patches]),
        'median_observed_plane_rms_mm':_median([p['observed_plane_rms_mm'] for p in patches]),
        'n_range_classified':sum(p['range_bin'] is not None for p in patches),
        'n_incidence_classified':sum(p['incidence_bin'] is not None for p in patches),
        'n_multi_station_patches':sum(p['coverage']=='multi_station' for p in patches),
        'n_single_station_patches':sum(p['coverage']=='single_station' for p in patches),
    }}


def _group(patches, key):
    groups={}
    for p in patches:
        k=p.get(key)
        if k is not None and k!='insufficient_support':
            groups.setdefault(k, []).append(p)
    out = {}
    for key, rows in sorted(groups.items()):
        per_seed = {}
        for row in rows:
            per_seed.setdefault(row['seed'], []).append(row['observed_plane_mad_mm'])
        n_scenes = len(per_seed)
        out[key] = {
            'n_patches': len(rows),
            'median_observed_plane_mad_mm': _median([p['observed_plane_mad_mm'] for p in rows]),
            'median_observed_plane_rms_mm': _median([p['observed_plane_rms_mm'] for p in rows]),
            'n_independent_scenes': n_scenes,
            'per_scene': [{'seed': seed, 'n_patches': len(v),
                           'median_observed_plane_mad_mm': _median(v)}
                          for seed, v in sorted(per_seed.items())],
            # For diagnostic tabulation only; NOT a statistical inference claim.
            'minimum_diagnostic_coverage': n_scenes >= 3 and len(rows) >= 5,
        }
    return out


def summarize(real_reference_dict, scenes):
    real=real_reference_dict['scans']
    patches=[{**p,'seed':sc['seed']} for sc in scenes for p in sc['surface_patches']]
    medians=[sc['summary']['median_observed_plane_mad_mm'] for sc in scenes
             if sc['summary']['n_patches']>=8]
    coverage=_group(patches,'coverage')
    summaries={
        'format': VERSION,
        'real_reference': {'sites':len(real),
            'site_median_mad_mm':[x['patch_mad_mm']['median'] for x in real],
            'per_station_range_or_incidence':'UNAVAILABLE_ORIGINAL_E57_MERGED',
            'actual_station_count':'UNKNOWN'},
        'synthetic': {'attempted_scenes':len(scenes),
            'valid_scenes':len(medians),
            'per_scene_median_mad_mm':medians,
            'median_of_scene_medians_mm':_median(medians),
            'stations_per_scene':[sc['stations'] for sc in scenes],
            'rooms_per_scene':[sc['rooms'] for sc in scenes],
            'range_groups':_group(patches,'range_bin'),
            'incidence_groups':_group(patches,'incidence_bin'),
            'coverage_groups':coverage,
            'n_multistation_patches':sum(p['coverage']=='multi_station' for p in patches),
            'n_patches':len(patches),
            'all_scene_details':scenes},
        'calibration_decision':'DESCRIPTIVE_ONLY_NO_SCANNER_PARAMETER_IDENTIFICATION',
        'auto_update_production_config':False,
        'quality_flags':[],
        'limitations':[
           'Real E57 exports have no original ray-to-station indexing; range/incidence/overlap cannot be compared cross-domain.',
           'Synthetic incidence uses a local observed wall normal fitted from XYZ, not physical mesh ground truth.',
           'Sampled top-24 dense patches per scene bias patch distribution; grouped medians are descriptive.',
           'Angle/range/overlap are correlated, not controlled independent experiments.',
           'No new training or automatic changes to original customer_like.yaml are authorized.',
        ],
    }
    f=summaries['quality_flags']
    if len(medians)<3:
        f.append('INSUFFICIENT_INDEPENDENT_MULTROOM_SCENES')
    if len(patches)<24:
        f.append('FEW_ELIGIBLE_PLANAR_PATCHES')
    if sum(p['coverage']=='multi_station' for p in patches)<5:
        f.append('SPARSE_MULTISTATION_WALL_PATCHES')
    if not any(v['minimum_diagnostic_coverage'] for v in summaries['synthetic']['range_groups'].values()):
        f.append('NO_WELL_SUPPORTED_RANGE_STRATUM')
    if not any(v['minimum_diagnostic_coverage'] for v in summaries['synthetic']['incidence_groups'].values()):
        f.append('NO_WELL_SUPPORTED_INCIDENCE_STRATUM')
    if len(real)<5:
        f.append('ONLY_TWO_REAL_SITES_NO_STATION_PROVENANCE')
    if len(medians)<5:
        f.append('THREE_SEED_EXPLORATORY_STUDY_NOT_STATISTICAL_CALIBRATION')
    far = [v for k,v in summaries['synthetic']['range_groups'].items()
           if k.startswith(('4–','6–','9+'))]
    if not any(v['minimum_diagnostic_coverage'] for v in far):
        f.append('FAR_RANGE_4M_PLUS_UNDERSAMPLED')
    grazing = [v for k,v in summaries['synthetic']['incidence_groups'].items()
               if k.startswith(('60–','75–'))]
    if not any(v['minimum_diagnostic_coverage'] for v in grazing):
        f.append('GRAZING_INCIDENCE_60DEG_PLUS_UNDERSAMPLED')
    if len(medians)>1 and min(medians)>0 and max(medians)/min(medians)>1.5:
        f.append('HIGH_BETWEEN_SCENE_MAD_VARIABILITY')
    return summaries


def report_md(doc):
    s=doc['synthetic'];rr=doc['real_reference']
    lines=['# CAL-04: многокомнатные сцены, дальность, угол падения и перекрытие станций','',
           'Плоскости выделены только по измеренным XYZ, без GT-классов.',
           'Дальности и углы доступны только для синтетических станций, исходные реальные E57 их не сохраняют.','',
           '| Seed | Комнат | Станций | Патчей | MAD, мм | Доля прямых отражений в выборке |',
           '|---:|---:|---:|---:|---:|---:|']
    for v in s['all_scene_details']:
        lines.append(f"| {v['seed']} | {v['rooms']} | {v['stations']} | {v['summary']['n_patches']} | "
                     f"{v['summary']['median_observed_plane_mad_mm']} | {v['direct_valid_fraction']:.3f} |")
    for title,key in [('Расстояние до наблюдаемой плоскости','range_groups'),
                      ('Угол падения к нормали наблюдаемой плоскости','incidence_groups'),
                      ('Поддержка станций','coverage_groups')]:
        lines+=['',f'## {title}', '', '| Группа | Патчей | Независимых сцен | MAD, мм | Мин. диагностическое покрытие? |','|---|---:|---:|---:|---|']
        for name,row in s[key].items():
            lines.append(f"| {name} | {row['n_patches']} | {row['n_independent_scenes']} | "
                         f"{row['median_observed_plane_mad_mm']} | {'да' if row['minimum_diagnostic_coverage'] else 'нет'} |")
    lines+=['', 'Таблицы агрегируют патчи; медианы отдельно по seed содержатся в JSON. Ни одна группа не является доказательством причинно-следственной связи.', '', '### Реальный ориентир CAL-02',
            f"Медианы двух объектов: {', '.join(str(x) for x in rr['site_median_mad_mm'])} мм.",
            'Для реальных E57 дальность, угол и перекрытие исходных станций **не наблюдаемы**.',
            '', '### Проверки качества', '']
    lines.extend(f'- {x}' for x in doc['quality_flags'])
    if not doc['quality_flags']:
        lines.append('Встроенные технические пороги пройдены; это не означает физическую калибровку.')
    lines+=['','**Вывод: только диагностическое сравнение. Параметры сканера, YAML и веса модели не изменены.**','',
            'Ограничения:']
    lines.extend(f'- {x}' for x in doc['limitations'])
    return '\n'.join(lines)+'\n'


def prepare_multroom_config(path):
    from simscan.config import load_config
    cfg=load_config(path)
    cfg.layout.source='simscan'
    cfg.layout.rooms=(3,3)
    cfg.layout.footprint_x_m=(8.,8.)
    cfg.layout.footprint_y_m=(6.5,6.5)
    cfg.layout.p_duplex=0.0
    cfg.layout.p_l_shape=0.0
    cfg.scanner.angular_step_deg=0.1
    cfg.scanner.max_stations_per_room=2
    cfg.scanner.stations_per_m2=.14
    cfg.scanner.p_limited_ceiling_view=0.0
    cfg.export.preview=False
    cfg.export.debug=False
    cfg.export.write_mesh=False
    cfg.export.write_merged=False
    cfg.export.unet_mask=False
    cfg.export.free_space_input=False
    return cfg


def run(*,out:Path, reference:Path=DEFAULT_REAL, config:Path=Path('configs/customer_like.yaml'),
        seeds=(42121,42122,42123),max_sample=250000,max_patches=24):
    from simscan.generate import generate_scene
    if len(seeds)<3 or len(set(seeds))!=len(seeds):
        raise ValueError('CAL-04 requires at least three distinct seeds')
    real=real_reference(reference)
    cfg=prepare_multroom_config(config)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        work=out/'_scratch'/f'scene_{seed}'
        try:
            print(f'CAL04 seed={seed}',flush=True)
            meta=generate_scene(cfg,work,seed=seed,index=0,write_scan=False,training_only=True)
            scans,poses=meta.pop('_scans')
            reservoir=sample_registered_with_stations(scans,poses,max_sample=max_sample,seed=42)
            evidence=patch_evidence(reservoir,seed=42,max_patches=max_patches)
            del scans,poses
            if meta['rooms']<2:
                raise RuntimeError('Not a multroom scan')
            row={'seed':seed,'rooms':meta['rooms'],'stations':meta['stations'],
                 'valid_returns':reservoir.valid,'sampled_returns':len(reservoir.xyz),
                 'sampling_fraction':round(reservoir.sampling_fraction,6),
                 'direct_valid_fraction':round(float(reservoir.direct.mean()),6),
                 'summary':evidence['summary'],
                 # Stored metadata only: deliberately no XYZ or sensor poses.
                 'surface_patches':evidence['patches']}
            rows.append(row)
            print(f"  rooms={row['rooms']} stations={row['stations']} patches={row['summary']['n_patches']} "
                  f"MAD={row['summary']['median_observed_plane_mad_mm']}mm",flush=True)
        finally:
            shutil.rmtree(work,ignore_errors=True)
    summary=summarize(real,rows)
    (out/'cal04_comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'CAL04_REPORT.md').write_text(report_md(summary),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--real-reference',type=Path,default=DEFAULT_REAL)
    p.add_argument('--config',type=Path,default=Path('configs/customer_like.yaml'))
    p.add_argument('--seeds',type=int,nargs='+',default=[42121,42122,42123])
    p.add_argument('--max-sample',type=int,default=250000)
    p.add_argument('--max-patches',type=int,default=24)
    a=p.parse_args()
    doc=run(out=a.out,reference=a.real_reference,config=a.config,seeds=a.seeds,
            max_sample=a.max_sample,max_patches=a.max_patches)
    print(report_md(doc),flush=True)


if __name__=='__main__':
    main()
