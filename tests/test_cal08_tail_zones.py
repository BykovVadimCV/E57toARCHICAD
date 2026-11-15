"""CAL-08 preregistered correlation-length sensitivity and guardrail tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_cal08_tail_zones import (
    NEW_SEEDS, USED_BEFORE, VARIANTS, CORRELATION_M, PREDECLARED_GATES,
    validate_seeds, footprint_for_seed, fixed_base_config, variant_config,
    patch_metrics, cluster_metrics, assess, report_md,
)

ROOT=Path(__file__).resolve().parents[1]


def real():
    return json.loads((ROOT/'configs/calibration_real_e57_surfaces.json').read_text())


def patches(*, n=24, high=0, value=7., isolate=False, median_shift=0.):
    out=[]
    for i in range(n):
        bad=i>=n-high
        index=i-(n-high)
        col=(index*5 if isolate else index) if bad else (i%8)
        row=0 if bad else (2+i//8)
        out.append({'wall_candidate_index':0,'patch_col':col,'patch_row':row,
                    'observed_plane_mad_mm':value if bad else 2.0+median_shift})
    return out


def records(*, shift=0., clustered=True, returns=.99, patches_limit=24, clean=True):
    rows=[]
    for seed in NEW_SEEDS:
        for variant in VARIANTS:
            is_long=variant in ('tail20_corr18','tail20_corr30')
            p=patches(n=patches_limit, high=(4 if variant=='tail20_corr18' else
                                            6 if variant=='tail20_corr30' else
                                            2 if variant=='tail20_corr08' else 0),
                      value=7.3 if variant!='tail20_corr30' else 8.2,
                      isolate=(variant=='tail20_corr08' or (is_long and not clustered)),
                      median_shift=shift if is_long else 0.)
            ext=p+patches(n=48,high=8 if is_long else 1,isolate=not clustered,
                          value=7.5,median_shift=shift if is_long else 0.)
            rows.append({'seed':seed,'variant':variant,'layout_signature':'scene_'+str(seed),
                         'layout_station_matched':True,'rooms':3,'stations':5,
                         'valid_returns':100_000 if variant=='uniform_2_4' else int(100_000*returns),
                         'primary_patches':p,'extended_patches':ext})
    return rows


def test_new_seeds_are_frozen_and_never_reused():
    assert len(NEW_SEEDS)==8 and not set(NEW_SEEDS)&set(USED_BEFORE)
    assert validate_seeds(NEW_SEEDS)==NEW_SEEDS
    assert [footprint_for_seed(s) for s in NEW_SEEDS[:4]]==['original_footprint']*4
    assert [footprint_for_seed(s) for s in NEW_SEEDS[4:]]==['larger_footprint']*4
    for invalid in (NEW_SEEDS[:-1],NEW_SEEDS[:-1]+(43101,),NEW_SEEDS[:-1]+(45101,)):
        with pytest.raises(ValueError):validate_seeds(invalid)


def test_vary_only_tail_zone_spatial_correlation_length():
    base=fixed_base_config(ROOT/'configs/customer_like.yaml',NEW_SEEDS[0])
    assert base.realism.wall_unevenness_mm==(2.,4.)
    assert base.scanner.angular_step_deg==.1
    assert base.layout.rooms==(3,3)
    assert set(CORRELATION_M)==set(VARIANTS)
    common=[]
    for variant in VARIANTS:
        cfg, changes=variant_config(base,variant)
        assert cfg.realism.wall_unevenness_mm==(2.,4.)
        assert cfg.realism.wall_roughness_tail_sigma_mm==(8.,12.)
        assert cfg.realism.wall_roughness_tail_probability==(0. if variant=='uniform_2_4' else .2)
        assert cfg.realism.wall_roughness_tail_corr_m==CORRELATION_M[variant]
        assert cfg.scanner==base.scanner and cfg.effects==base.effects
        d=copy.deepcopy(vars(cfg.realism))
        d.pop('wall_roughness_tail_probability')
        d.pop('wall_roughness_tail_corr_m')
        common.append(d)
    assert common[0]==common[1]==common[2]==common[3]
    with pytest.raises(ValueError):variant_config(base,'tail50_corr99')


def test_local_patch_cluster_has_no_spurious_cross_line_edges():
    assert cluster_metrics(patches(high=4))['largest_tail_cluster']==4
    assert cluster_metrics(patches(high=4,isolate=True))['largest_tail_cluster']==1
    a=patches(high=4)
    for p in a[-2:]:p['wall_candidate_index']=1
    assert cluster_metrics(a)['largest_tail_cluster']==2
    assert patch_metrics(patches(high=4))['p90_mad_mm']>=7.0
    with pytest.raises(ValueError):cluster_metrics([{'observed_plane_mad_mm':7.0}])
    with pytest.raises(ValueError):patch_metrics([{'observed_plane_mad_mm':float('nan')}])


def test_passes_controlled_fixture_and_prefers_shorter_supported_length():
    d=assess(real(),records())
    assert d['decision']=='PROVISIONAL_SPATIAL_LENGTH_tail20_corr18_SUPPORTED'
    assert d['selected_candidate']=='tail20_corr18'
    assert d['auto_update_production_config'] is False
    assert d['variants']['tail20_corr18']['all_gates_pass']
    assert d['variants']['tail20_corr30']['all_gates_pass']
    assert d['variants']['tail20_corr18']['primary_cluster_scenes']==8
    assert d['variants']['tail20_corr18']['extended_cluster_scenes']==8
    assert len(d['variants']['tail20_corr18']['paired_scenes'])==8
    assert 'Физическая' not in report_md(d) or 'не физическая калибровка' in report_md(d)


def test_reject_median_degradation_even_when_p90_improves():
    d=assess(real(),records(shift=.8))
    assert d['selected_candidate'] is None
    assert not d['variants']['tail20_corr18']['gates']['typical_median_preserved_six_of_eight']


def test_reject_isolated_tail_even_when_upper_quantile_improves():
    d=assess(real(),records(clustered=False))
    assert d['selected_candidate'] is None
    assert not d['variants']['tail20_corr18']['gates']['top24_cluster_in_four_of_eight']


def test_reject_missing_patches_or_large_return_losses():
    d=assess(real(),records(patches_limit=7))
    assert d['selected_candidate'] is None
    assert not d['variants']['tail20_corr18']['gates']['all_eight_pairs_eligible']
    d=assess(real(),records(returns=.6))
    assert d['selected_candidate'] is None
    assert not d['variants']['tail20_corr18']['gates']['all_eight_pairs_eligible']


def test_reject_duplicates_or_differently_registered_true_stations():
    rows=records()
    with pytest.raises(ValueError):assess(real(),rows[:-1])
    dup=copy.deepcopy(rows)
    dup[-1]=dup[0]
    with pytest.raises(ValueError):assess(real(),dup)
    different=copy.deepcopy(rows)
    different[-1]['layout_signature']='changed'
    with pytest.raises(ValueError):assess(real(),different)
    different=copy.deepcopy(rows)
    different[-1]['stations']=6
    with pytest.raises(ValueError):assess(real(),different)


def test_do_not_export_raw_positions_or_claim_physical_scanner_calibration():
    result=assess(real(),records())
    data=json.dumps(result)
    for forbidden in ('"xyz"','"origins"','"pose_true"','"pose_written"','"raw_points"',
                      '"intrinsic_scanner_sigma"','"site_coordinates"'):
        assert forbidden not in data
    assert result['auto_update_production_config'] is False
    assert PREDECLARED_GATES['min_scene_median_p90_mm']==5.0
    assert result['real_reference']['station_ray_provenance']=='UNKNOWN'
