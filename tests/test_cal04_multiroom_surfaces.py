"""CAL-04 measurement semantics: only returns/poses, never synthetic GT geometry."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.calibrate_e57_surfaces import assess_wall_patches
from tools.calibrate_multiroom_surfaces import (
    VERSION, TaggedReservoir, _bin, patch_evidence, prepare_multroom_config,
    sample_registered_with_stations, summarize, report_md,
)
from tools.calibrate_real_synthetic_surfaces import real_reference

REF=Path(__file__).resolve().parents[1]/'configs/calibration_real_e57_surfaces.json'
ROOT=Path(__file__).resolve().parents[1]


def fixture_station(station, n=8500, offset=0.0, seed=3):
    rng=np.random.default_rng(seed)
    x=rng.uniform(0, 4.5, n)
    z=rng.uniform(0.3, 2.7, n)
    y=rng.normal(offset, .002, n)
    world=np.c_[x,y,z]
    # Sensor at y=-2 or y=-4 on the opposite side of the wall.
    t=np.array([2.0, -2.0-2.0*station, 1.4])
    scan=SimpleNamespace(xyz_local=world-t, valid=np.ones(n,bool),
                         straight_valid=np.ones(n,bool))
    return scan,(np.eye(3),t)


def test_tagged_reservoir_chunk_independent_and_uniform():
    rng=np.random.default_rng(123)
    v=np.c_[rng.uniform(0,3,10000),rng.normal(0,.01,10000),rng.uniform(.3,2,10000)]
    R=np.linalg.norm(v,axis=1)
    a=TaggedReservoir(1000,seed=9)
    b=TaggedReservoir(1000,seed=9)
    a.add(v,np.zeros(3),R,2)
    for i in range(0,len(v),250):
        b.add(v[i:i+250],np.zeros(3),R[i:i+250],2)
    assert a.valid==b.valid==10000
    assert len(a.xyz)==len(b.xyz)==1000
    assert np.array_equal(np.sort(a.keys),np.sort(b.keys))
    assert np.all(a.stations==2)
    assert abs(a.sampling_fraction-.1)<1e-9


def test_valid_and_direct_filtered_independently():
    s=TaggedReservoir(1000)
    xyz=np.array([[1.,0,0],[2,0,0],[3,0,0],[4.,0,0]])
    s.add(xyz,np.zeros(3),np.linalg.norm(xyz,axis=1),1,
          valid=[1,1,0,1],direct=[1,0,1,0])
    assert s.valid==3 and len(s.xyz)==3
    assert int(s.direct.sum())==1
    assert s.seen==4
    with pytest.raises(ValueError):
        s.add(xyz,np.zeros(3),np.ones(2),1)
    with pytest.raises(ValueError):
        s.add(xyz,np.zeros(2),np.ones(4),1)


def test_registration_uses_written_pose_and_explicit_ray_origin():
    scan,pose=fixture_station(0,n=1000)
    sr=sample_registered_with_stations([scan],[pose],max_sample=1000)
    assert abs(np.median(sr.xyz[:,1]))<.02
    assert np.allclose(sr.origins[:,1],-2.0)
    assert 1.5<np.median(sr.ranges)<5
    with pytest.raises(ValueError):
        sample_registered_with_stations([scan],[],max_sample=1000)
    with pytest.raises(ValueError):
        sample_registered_with_stations([scan],[(np.eye(2),np.zeros(3))],max_sample=1000)


def test_patches_reuse_cal02_residual_and_detect_real_cross_station_overlap():
    entries=[fixture_station(0,n=16000,seed=1),fixture_station(1,n=16000,offset=.004,seed=2)]
    s=sample_registered_with_stations([x[0] for x in entries],[x[1] for x in entries],max_sample=32000,seed=42)
    derived=patch_evidence(s,seed=42,max_patches=24)
    baseline=assess_wall_patches(s.xyz,seed=42,max_patches=24)
    assert derived['summary']['n_patches']>=5
    # Exactly the same CAL-02 estimator and selected 24 patches.
    assert derived['summary']['n_patches']==baseline['summary']['n_patches']
    assert derived['summary']['median_observed_plane_mad_mm']==baseline['summary']['median_observed_plane_mad_mm']
    assert derived['summary']['n_multi_station_patches']>=1
    both=[p for p in derived['patches'] if p['coverage']=='multi_station']
    assert all(2 <= p['well_supported_stations'] for p in both)
    assert any(p['interstation_normal_offset_span_mm'] is not None for p in both)
    assert all(0 <= p['median_incidence_deg'] <90 for p in derived['patches'] if p['median_incidence_deg'] is not None)
    # Do not publish any raw point coordinate, position or fitted line equation.
    assert not any(k in both[0] for k in ('xyz','origin','x','y','z','normal','bias'))


def test_independent_inference_guard_and_missing_real_station_metadata():
    real=real_reference(REF)
    mk=lambda seed,mad:{'seed':seed,'rooms':3,'stations':4,'direct_valid_fraction':.9,
     'summary':{'n_patches':8,'median_observed_plane_mad_mm':mad},
     'surface_patches':[{'observed_plane_mad_mm':mad,'observed_plane_rms_mm':mad+0.2,
       'range_bin':'2–4 m','incidence_bin':'0–30 deg','coverage':'single_station'}]}
    doc=summarize(real,[mk(1,2.1),mk(2,2.3)])
    assert doc['format']==VERSION
    assert doc['auto_update_production_config'] is False
    assert doc['real_reference']['per_station_range_or_incidence']=='UNAVAILABLE_ORIGINAL_E57_MERGED'
    assert 'INSUFFICIENT_INDEPENDENT_MULTROOM_SCENES' in doc['quality_flags']
    assert not doc['synthetic']['range_groups']['2–4 m']['minimum_diagnostic_coverage']
    assert 'ONLY_TWO_REAL_SITES_NO_STATION_PROVENANCE' in doc['quality_flags']
    assert 'FAR_RANGE_4M_PLUS_UNDERSAMPLED' in doc['quality_flags']
    assert 'GRAZING_INCIDENCE_60DEG_PLUS_UNDERSAMPLED' in doc['quality_flags']
    assert len(doc['synthetic']['range_groups']['2–4 m']['per_scene'])==2
    assert 'физическую калибровку' in report_md(doc) or 'Параметры сканера' in report_md(doc)


def test_bins_are_disjoint_and_declared():
    assert _bin(1.999,(0.,2.,4.,float('inf')),'m')=='0–2 m'
    assert _bin(2.,(0.,2.,4.,float('inf')),'m')=='2–4 m'
    assert _bin(30.,(0.,30.,60.,90.00001),'deg')=='30–60 deg'
    assert _bin(-1.,(0.,30.,90.00001),'deg') is None
    assert _bin(np.nan,(0.,30.,90.00001),'deg') is None


def test_three_rooms_and_locked_scanner_step_and_no_config_mutation():
    cfg=prepare_multroom_config(ROOT/'configs/customer_like.yaml')
    assert cfg.layout.source=='simscan'
    assert cfg.layout.rooms==(3,3)
    assert cfg.layout.p_duplex==0
    assert cfg.scanner.angular_step_deg==.1
    assert cfg.scanner.require_all_rooms_scanned is True
    assert cfg.scanner.max_stations_per_room==2


def test_coverage_station_minimum_does_not_count_one_weak_return():
    a=fixture_station(0,n=8000,seed=2)
    b=fixture_station(1,n=8000,seed=3)
    sb=b[0]
    sb.straight_valid[:]=False
    sb.straight_valid[:8]=True
    s=sample_registered_with_stations([a[0],b[0]],[a[1],b[1]],max_sample=16000)
    out=patch_evidence(s)
    assert out['summary']['n_patches']>=1
    assert out['summary']['n_multi_station_patches']==0
