"""CAL-05: pairing and non-calibration quality gate regression."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.calibrate_factorial_surfaces import (
    DEFAULT_NAMES, VERSION, VARIANTS, MIN_PATCHES, paired_layout_signature,
    variant_config, summarize, report_md, prepare_multroom_config,
    _brief_patches, _pair_effect,
)
from tools.calibrate_real_synthetic_surfaces import real_reference

ROOT = Path(__file__).resolve().parents[1]


def test_variants_only_modify_one_factor_and_preserve_production_config():
    original = prepare_multroom_config(ROOT / 'configs/customer_like.yaml')
    assert len(DEFAULT_NAMES) == 7
    assert len(set(DEFAULT_NAMES)) == len(DEFAULT_NAMES)
    assert original.scanner.angular_step_deg == .1
    rough = original.realism.wall_unevenness_mm
    registration = original.effects.registration_sigma_mm
    ranging = original.effects.range_sigma_mm
    for name in DEFAULT_NAMES:
        cfg, delta = variant_config(original, name)
        assert len(delta) == (0 if name == 'baseline' else 1)
        assert cfg.scanner.angular_step_deg == .1
        assert cfg.layout.rooms == (3, 3)
        assert original.realism.wall_unevenness_mm == rough
        assert original.effects.registration_sigma_mm == registration
        assert original.effects.range_sigma_mm == ranging
        expected = {k.split('.')[1] for k in delta}
        assert (cfg.realism.wall_unevenness_mm != rough) == ('wall_unevenness_mm' in expected)
        assert (cfg.effects.registration_sigma_mm != registration) == ('registration_sigma_mm' in expected)
        assert (cfg.effects.range_sigma_mm != ranging) == ('range_sigma_mm' in expected)
    with pytest.raises(ValueError):
        variant_config(original, 'made_up')


def fixture_layout():
    st = {'id': 0, 'room_id': 3, 'level': 0,
          'position_layout_m': [1, 2, 1.3], 'heading_deg': 5,
          'tilt_deg': [0, 0], 'pose_true': {'R': np.eye(3).tolist(), 't': [1, 2, 1.3]},
          'pose_written': {'R': np.eye(3).tolist(), 't': [1.01, 2, 1.3]},
          'valid': 100, 'rays': 200}
    return {'rooms': [{'id': 3, 'kind': 'room'}], 'walls': [], 'openings': [],
            'meta': {'layout_to_world': {'yaw': 0}, 'effects_sampled': {}},
            'stations': [st]}


def test_pair_signature_ignores_registered_error_and_measurement_counts():
    a = fixture_layout()
    b = copy.deepcopy(a)
    b['stations'][0]['pose_written']['t'][0] += 1
    b['stations'][0]['valid'] = 10
    b['meta']['effects_sampled']['range_sigma_m'] = 999
    assert paired_layout_signature(a) == paired_layout_signature(b)
    b['stations'][0]['pose_true']['t'][0] += .001
    assert paired_layout_signature(a) != paired_layout_signature(b)
    with pytest.raises(ValueError):
        paired_layout_signature({'stations': []})
    with pytest.raises(ValueError):
        paired_layout_signature({'stations': [{'id': 0}], 'rooms': [{'id': 0}]})


def mkrow(seed,variant,mad,patches=12,match=True):
    p={'observed_plane_mad_mm':mad,'observed_plane_rms_mm':mad+.5,
       'range_bin':'2–4 m','incidence_bin':'30–60 deg','coverage':'multi_station'}
    return {'seed':seed, 'variant':variant, 'n_patches':patches,
            'valid_for_comparison':patches>=MIN_PATCHES,
            'layout_station_matched':match,'median_mad_mm':mad,
            'median_rms_mm':mad+.5,'surface_patches':[p]*patches,
            'rooms':3,'stations':4}


def test_paired_effect_is_by_scene_not_by_individual_patches():
    rows={1:{'baseline':mkrow(1,'baseline',4.0),
             'roughness_low':mkrow(1,'roughness_low',2.0)},
          2:{'baseline':mkrow(2,'baseline',6.0),
             'roughness_low':mkrow(2,'roughness_low',7.0)}}
    x=_pair_effect(rows)['roughness_low']
    assert x['paired_scenes']==2
    assert x['median_delta_mad_mm']==-.5
    assert x['positive_delta_count']==1
    assert x['negative_delta_count']==1
    assert not x['adequate_exploratory_paired_scenes']
    rows[2]['roughness_low']['layout_station_matched']=False
    assert _pair_effect(rows)['roughness_low']['paired_scenes']==1


def test_quality_gate_refuses_sparse_pairing_and_missing_variants():
    ref=real_reference(ROOT / 'configs/calibration_real_e57_surfaces.json')
    rows=[mkrow(1,'baseline',5.), mkrow(1,'roughness_low',3.),
          mkrow(2,'baseline',6.), mkrow(2,'roughness_low',3.,patches=4)]
    d=summarize(ref,rows,expected_seeds=(1,2,3,4))
    assert d['format']==VERSION
    assert d['auto_update_production_config'] is False
    assert 'MISSING_FACTORIAL_RUNS' in d['quality_flags']
    assert 'INSUFFICIENT_VALID_PAIRED_SCENES_FOR_SOME_FACTORS' in d['quality_flags']
    assert d['variants']['roughness_low']['valid_paired_scenes']==1
    assert d['paired_effects']['roughness_low']['paired_scenes']==1
    assert 'параметры production YAML и веса не менять' in report_md(d)


def test_full_four_seed_fixture_has_no_fake_physical_calibration():
    ref=real_reference(ROOT / 'configs/calibration_real_e57_surfaces.json')
    rows=[mkrow(seed,var,5 + .5*(seed-1) + (1 if var.endswith('high') else -1 if var.endswith('low') else 0))
          for seed in range(1,5) for var in DEFAULT_NAMES]
    d=summarize(ref,rows,expected_seeds=range(1,5))
    assert d['variants']['baseline']['valid_paired_scenes']==4
    assert d['paired_effects']['roughness_high']['median_delta_mad_mm']==1
    assert d['paired_effects']['range_noise_low']['median_delta_mad_mm']==-1
    assert d['paired_effects']['range_noise_high']['adequate_exploratory_paired_scenes']
    assert d['real']['original_ray_station_metadata']=='UNAVAILABLE'
    assert d['auto_update_production_config'] is False
    assert 'range_sigma_mm' not in d['calibration_decision']


def test_summary_rejects_duplicate_scene_and_brief_patches_no_positions():
    ref=real_reference(ROOT/'configs/calibration_real_e57_surfaces.json')
    with pytest.raises(ValueError):
        summarize(ref,[mkrow(1,'baseline',4.),mkrow(1,'baseline',6.)])
    with pytest.raises(ValueError):
        summarize(ref,[])
    b=_brief_patches({'patches':[{'observed_plane_mad_mm':2.0,'x':123,
                                   'station_world_pose':[1,2,3], 'median_range_m':2.5}]})
    assert b==[{'observed_plane_mad_mm':2.0,'median_range_m':2.5}]


def test_json_serialization_contains_no_cloud_coordinates():
    ref=real_reference(ROOT/'configs/calibration_real_e57_surfaces.json')
    d=summarize(ref,[mkrow(1,'baseline',5.)])
    s=json.dumps(d)
    for forbidden in ('"xyz"','"pose_true"','"pose_written"','"origins"'):
        assert forbidden not in s
