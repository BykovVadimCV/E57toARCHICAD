"""Independent recalculation of CAL-08 derived scalar artifact, without any scan XYZ.

Does not import validate_cal08_tail_zones; expected thresholds are written here
independently to reduce correlated mistakes in the acceptance evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median

EXPECTED_SEEDS=set(range(45101,45109))
EXPECTED_VARIANTS={'uniform_2_4','tail20_corr08','tail20_corr18','tail20_corr30'}


def audit(doc:dict,production_sha:str|None=None)->dict:
    assert doc['format']=='cal08-spatial-tail-length-holdout-v1'
    assert set(doc['design']['seeds'])==EXPECTED_SEEDS
    assert set(doc['design']['variants'])==EXPECTED_VARIANTS
    assert doc['auto_update_production_config'] is False
    assert not set(doc['design']['seeds'])&set(doc['design']['excluded_previous_seeds'])
    assert doc['real_reference']['n_sites']==2
    assert doc['real_reference']['station_ray_provenance']=='UNKNOWN'
    assert doc['real_reference']['site_p90_mad_mm']==[1.7205,10.0135]
    assert doc['real_reference']['site_median_mad_mm']==[1.4731,2.7332]
    if production_sha:
        assert doc['source_config_sha256']==production_sha
    assert set(doc['variants'])==EXPECTED_VARIANTS
    rejected=[]
    passed=[]
    data=[]
    for name in sorted(EXPECTED_VARIANTS):
        v=doc['variants'][name]
        rows=v['paired_scenes']
        assert len(rows)==8 and {int(r['seed']) for r in rows}==EXPECTED_SEEDS
        assert {r['cohort'] for r in rows}=={'original_footprint','larger_footprint'}
        for r in rows:
            assert r['rooms']==3 and r['stations']>=1
            assert r['primary_patches']<=24 and r['expanded_patches']<=72
            assert 0<=r['top24_max_cluster']<=r['top24_tail_patches']
            assert 0<=r['extended_max_cluster']<=r['extended_tail_patches']
            assert 0.0 < r['relative_valid_returns'] < 2.0
        ok=[r for r in rows if r['eligible']]
        assert len(ok)==v['eligible_scenes']
        def med(k):return round(float(median([r[k] for r in ok])),4) if ok else None
        assert med('variant_median_mm')==v['median_scene_mad_mm']
        assert med('variant_p90_mm')==v['median_scene_p90_mm']
        assert med('variant_p95_mm')==v['median_scene_p95_mm']
        assert med('p90_gain_vs_uniform')==v['median_p90_gain_ratio']
        assert med('top24_max_cluster')==v['median_top24_largest_cluster']
        assert med('extended_max_cluster')==v['median_extended_largest_cluster']
        median_kept=sum(abs(r['median_shift_mm'])<=.5 for r in ok)
        inside=sum(.9731<=r['variant_median_mm']<=3.2332 for r in ok)
        p90_gain=sum(r['p90_gain_vs_uniform']>=1.2 for r in ok)
        below=sum(r['variant_p90_mm']<=11.0135 for r in ok)
        c24=sum(r['top24_max_cluster']>=2 for r in ok)
        c72=sum(r['extended_max_cluster']>=2 for r in ok)
        spatial_gain=sum(r['top24_max_cluster']>r['short_corr_max_cluster'] for r in ok)
        footprint={g:sum(r['p90_gain_vs_uniform']>=1.2 and r['cohort']==g for r in ok)
                   for g in ('original_footprint','larger_footprint')}
        gates={
            'all_eight_pairs_eligible':len(ok)==8,
            'typical_median_preserved_six_of_eight':median_kept>=6,
            'scene_median_in_real_bracket_six_of_eight':inside>=6,
            'p90_gain_20pct_six_of_eight':p90_gain>=6,
            'p90_below_real_upper_site_plus_1mm_six_of_eight':below>=6,
            'median_scene_p90_between_5_and_real_upper_plus_1mm':med('variant_p90_mm') is not None and 5.0<=med('variant_p90_mm')<=11.0135,
            'top24_cluster_in_four_of_eight':c24>=4,
            'extended_cluster_in_four_of_eight':c72>=4,
            'p90_gain_in_both_cohorts_three_of_four':min(footprint.values())>=3,
            'cluster_gain_over_short_corr_in_four_of_eight':spatial_gain>=4,
        }
        assert gates==v['gates'],(name,gates,v['gates'])
        if name in ('tail20_corr18','tail20_corr30'):
            assert all(gates.values())==v['all_gates_pass']
            (passed if all(gates.values()) else rejected).append(name)
        else:assert v['all_gates_pass'] is None
        data.append({'variant':name,'scene_median_mad_mm':v['median_scene_mad_mm'],
                     'scene_median_p90_mm':v['median_scene_p90_mm'],
                     'scene_median_p95_mm':v['median_scene_p95_mm'],
                     'p90_gain_ratio':v['median_p90_gain_ratio'],
                     'primary_cluster_scenes':c24,'extended_cluster_scenes':c72,
                     'return_ratio_range':[min(r['relative_valid_returns'] for r in rows),
                                           max(r['relative_valid_returns'] for r in rows)],
                     'failed_gates':[k for k,passed in gates.items() if not passed]})
    chosen=passed[0] if passed else None
    assert doc['selected_candidate']==chosen
    assert doc['decision']==('PROVISIONAL_SPATIAL_LENGTH_'+chosen+'_SUPPORTED' if chosen else 'NO_CORRELATION_LENGTH_PASSED_PREDECLARED_GATES')
    # Outer report is aggregate only. Data inventory check on all source fields.
    raw=json.dumps(doc)
    for forbidden in ('"xyz"','"origin"','"pose_true"','"pose_written"',
                      '"station_world_pose"','"physical_sensor_sigma"'):
        assert forbidden not in raw
    return {'verified':True,'decision':doc['decision'], 'selected':chosen,
            'variants':data,'source_config_sha256':doc.get('source_config_sha256'),
            'warning':'Two real reference sites, densely selected patches; not physical sensor calibration.'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('report',type=Path)
    p.add_argument('--production-yaml',type=Path)
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    sha=hashlib.sha256(a.production_yaml.read_bytes()).hexdigest() if a.production_yaml else None
    obj=audit(json.loads(a.report.read_text(encoding='utf8')),sha)
    s=json.dumps(obj,indent=2,ensure_ascii=False)
    if a.out:a.out.write_text(s+'\n',encoding='utf8')
    print(s)

if __name__=='__main__':main()
