"""CAL-06 gate regression on fictional tiny summary data. NOT actual holdout results."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.validate_cal06_holdout import (
    NEW_SEEDS, USED_BEFORE, PREDECLARED_GATE, VERSION,
    VARIANTS, validate_seeds, footprint_for_seed, fixed_base_config,
    assess, report_md,
)


def _real():
    return json.loads((Path(__file__).resolve().parents[1] /
                      'configs/calibration_real_e57_surfaces.json').read_text())


def _pair_rows(*, baseline=5.4, candidate=2.0, candidate_n=22,
               candidate_returns=90_000):
    rows=[]
    for seed in NEW_SEEDS:
        sign=f'reproducible_{seed}'
        for name, val, patch_count, valid in (
            ('baseline', baseline, 24, 100_000),
            ('roughness_low', candidate, candidate_n, candidate_returns),
        ):
            rows.append({
                'seed': seed, 'variant': name, 'layout_signature': sign,
                'layout_station_matched': True, 'rooms': 3, 'stations': 4,
                'n_patches': patch_count, 'valid_returns': valid,
                'median_mad_mm': val,
                'surface_patches': [{'observed_plane_mad_mm': val}] * patch_count,
            })
    return rows


def test_holdout_seeds_frozen_and_no_reuse():
    assert not (set(NEW_SEEDS) & USED_BEFORE)
    assert len(NEW_SEEDS) == 8 and len(set(NEW_SEEDS)) == 8
    assert validate_seeds(NEW_SEEDS) == NEW_SEEDS
    assert set(footprint_for_seed(s) for s in NEW_SEEDS) == {'original_footprint','larger_footprint'}
    for x in (NEW_SEEDS[:7], NEW_SEEDS[:7] + (42121,), [NEW_SEEDS[0]] * 8):
        with pytest.raises(ValueError): validate_seeds(x)


def test_only_baseline_and_locked_cal05_candidate():
    assert VARIANTS == ('baseline', 'roughness_low')
    assert PREDECLARED_GATE['min_eligible_paired_scenes'] == 8
    assert PREDECLARED_GATE['min_scenes_improved'] == 7
    assert PREDECLARED_GATE['min_candidate_to_baseline_patch_count'] == 0.75
    # Imports original CAL-05 variants; no new candidate tuning allowed here.
    from tools.calibrate_factorial_surfaces import VARIANTS as old
    assert old['roughness_low'] == {'realism.wall_unevenness_mm': (2.,4.)}
    assert old['baseline'] == {}


def test_baseline_and_candidate_share_unchanged_other_configs():
    from tools.calibrate_factorial_surfaces import variant_config
    from simscan.config import config_to_dict
    a=fixed_base_config(Path('configs/customer_like.yaml'), NEW_SEEDS[0])
    b=fixed_base_config(Path('configs/customer_like.yaml'), NEW_SEEDS[-1])
    assert a.layout.rooms == (3,3) and b.layout.rooms == (3,3)
    assert a.layout.footprint_x_m != b.layout.footprint_x_m
    assert a.scanner.angular_step_deg == b.scanner.angular_step_deg == 0.1
    x,_=variant_config(a,'baseline'); y, changes=variant_config(a,'roughness_low')
    assert changes == {'realism.wall_unevenness_mm': [2.,4.]}
    z=config_to_dict(y); z['realism']['wall_unevenness_mm'] = config_to_dict(x)['realism']['wall_unevenness_mm']
    assert z == config_to_dict(x)
    assert a.realism.wall_unevenness_mm == (6., 14.)


def test_provisional_gate_passes_plausible_holdout():
    result=assess(_real(), _pair_rows())
    assert result['format'] == VERSION
    assert result['holdout']['all_predeclared_gates_pass'] is True
    assert result['holdout']['eligible_scenes'] == 8
    assert result['holdout']['candidate_in_real_interval_scenes'] == 8
    assert result['holdout']['median_paired_reduction_mm'] == 3.4
    assert result['candidate_decision'].startswith('PROVISIONAL_')
    assert result['auto_update_production_config'] is False
    assert 'не объявляется завершённой' in report_md(result)
    assert 'XYZ' not in json.dumps(result)


def test_reject_when_candidate_drops_patches_even_if_MAD_improves():
    result=assess(_real(),_pair_rows(candidate=.7,candidate_n=11))
    assert result['holdout']['eligible_scenes'] == 0
    assert not result['holdout']['all_predeclared_gates_pass']
    assert result['candidate_decision'].startswith('HOLDOUT_REJECT')
    assert 'FAIL: all_eight_eligible' in report_md(result)


def test_reject_when_candidate_drops_returns():
    result=assess(_real(),_pair_rows(candidate=2.0,candidate_returns=20000))
    assert result['holdout']['eligible_scenes'] == 0
    assert not result['holdout']['all_predeclared_gates_pass']


def test_reject_when_no_improvement_or_far_outside_real_target():
    x=assess(_real(),_pair_rows(baseline=5.4,candidate=5.1))
    assert x['holdout']['improved_scenes'] == 8
    assert not x['holdout']['gates']['median_reduction_at_least_1mm']
    assert not x['holdout']['gates']['six_in_expanded_real_interval']
    assert not x['holdout']['all_predeclared_gates_pass']


def test_reject_if_only_one_footprint_responds():
    rows=_pair_rows()
    for r in rows:
        if r['seed'] in NEW_SEEDS[4:] and r['variant'] == 'roughness_low':
            r['median_mad_mm'] = 5.7
    result=assess(_real(),rows)
    assert result['holdout']['improved_by_cohort']['larger_footprint']==0
    assert not result['holdout']['gates']['both_footprint_cohorts_respond']
    assert not result['holdout']['all_predeclared_gates_pass']


def test_no_mismatched_or_duplicated_pair_can_be_accepted():
    for altered in ('duplicate','missing','signature','stations'):
        rows=_pair_rows()
        if altered=='duplicate':rows[0]=rows[1]
        if altered=='missing': rows.pop()
        if altered=='signature': rows[1]['layout_signature']='different'
        if altered=='stations': rows[1]['stations']=8
        with pytest.raises(ValueError):assess(_real(),rows)


def test_no_inference_of_intrinsic_scanner_physics():
    result=assess(_real(),_pair_rows())
    assert result['real_reference']['physical_station_ray_metadata']=='UNAVAILABLE'
    assert result['auto_update_production_config'] is False
    assert 'intrinsic_sensor_noise_sigma_mm' not in result
    assert result['candidate_decision'] != 'PHYSICALLY_CALIBRATED'
    assert result['design']['matched_physical_wall_patches'] is False

