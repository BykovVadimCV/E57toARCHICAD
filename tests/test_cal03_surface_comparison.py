"""CAL-03: compare like-for-like surface measurements without auto-fitting noise."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.calibrate_real_synthetic_surfaces import (
    VERSION, analyze_registered_scans, compare, real_reference, report_md,
    sample_registered_returns, variant_config,
)

REF = Path(__file__).resolve().parents[1] / 'configs/calibration_real_e57_surfaces.json'


def fixture_scan(sigma=0.002, n=7000, seed=4):
    rng=np.random.default_rng(seed)
    xyz=np.c_[rng.normal(0,sigma,n), rng.uniform(0,4.5,n), rng.uniform(.3,2.5,n)]
    return SimpleNamespace(xyz_local=xyz, valid=np.ones(n,bool))


def test_registration_pose_applied_and_not_groundtruth():
    scan=fixture_scan(n=1200)
    R=np.eye(3)
    p=sample_registered_returns([scan], [(R,np.array([6., -2., 3.]))], 1000,seed=10)
    assert len(p.points)==1000
    assert 5.9 < p.points[:,0].mean() < 6.1
    assert -2.0 < p.points[:,1].mean() < 2.5
    assert p.valid==1200


def test_small_but_measured_vertical_patch():
    scan=fixture_scan(n=16000,sigma=.002)
    result=analyze_registered_scans([scan], [(np.eye(3),np.zeros(3))],max_sample=16000,seed=9)
    assert result['surfaces']['summary']['n_patches']>=1
    assert 1 < result['surfaces']['summary']['median_observed_plane_mad_mm'] < 4
    assert result['source_type']=='registered_synthetic_returns_no_gt'


def test_missing_registration_pose_rejected():
    scan=fixture_scan(n=1200)
    with pytest.raises(ValueError):
        sample_registered_returns([scan], [(np.zeros((3,2)), np.zeros(3))],1000)
    with pytest.raises(ValueError):
        sample_registered_returns([scan], [],1000)


def test_reference_is_measured_and_unidentifiable():
    real=real_reference(REF)
    assert len(real['scans'])==2
    assert real['scans'][0]['actual_station_count']=='not_identifiable'
    assert 'range_sigma_mm' not in real['scans'][0]


def test_comparison_never_auto_tunes_parameters():
    reference=real_reference(REF)
    measured={'variant':'baseline','surfaces':{'summary':{
        'n_patches':8,'median_observed_plane_mad_mm':2.0,'median_observed_plane_rms_mm':2.2},
        'patches':[{'observed_plane_mad_mm':float(i)} for i in [1.,1.5,1.7,2.,2.,2.1,2.5,3.]]}}
    doc=compare(reference,[measured]);assert doc['format']==VERSION
    assert doc['auto_update_production_config'] is False
    assert len(doc['real_reference']['median_mad_mm_by_file'])==2
    assert doc['comparisons']['baseline']['valid_scenes']==1
    assert 'noise' in doc['calibration_decision'].lower() or 'PARAMETER' in doc['calibration_decision']
    assert 'автоматически не изменять' in report_md(doc)


def test_empty_sample_does_not_invent_a_match():
    reference=real_reference(REF)
    bad={'variant':'baseline','surfaces':{'summary':{'n_patches':0},'patches':[]}}
    doc=compare(reference,[bad]);assert doc['comparisons']['baseline']['valid_scenes']==0
    assert 'NO_ELIGIBLE_VERTICAL_PATCHES' in str(doc)


def test_variants_change_only_one_family():
    cfg=SimpleNamespace(realism=SimpleNamespace(wall_unevenness_mm=(6.,14.)),
      effects=SimpleNamespace(registration_sigma_mm=(2.,6.)))
    b,changes=variant_config(cfg,'baseline')
    assert not changes and b.realism.wall_unevenness_mm==(6.,14.)
    q,changes=variant_config(cfg,'low_registration')
    assert q.effects.registration_sigma_mm == (0.5,1.0)
    assert q.realism.wall_unevenness_mm==(6.,14.)
    assert cfg.effects.registration_sigma_mm==(2.,6.)
    with pytest.raises(ValueError):variant_config(cfg,'invented_parameter')
