import numpy as np
import pytest
from shapely.geometry import Polygon
from shapely import union_all
from simscan.door_fit import fit_doors
from simscan.wall_footprint import mask_geometry,regions,primitive


def fixture():
    m=np.zeros((80,180),bool);m[20:28,10:60]=True;m[20:28,100:170]=True
    p=np.zeros((8,*m.shape),np.float32);p[0]=1;p[:,18:30,60:100]=0;p[3,18:30,60:100]=1
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))],wall_faces=[dict(start=[10,24],end=[170,24])])
    return m,p,r


def test_door_fits_the_actual_gap_and_preserves_walls():
    m,p,r=fixture();before=m.copy();d=fit_doors(p,r,m,.02,.02)
    assert len(d['doors'])==1
    door=d['doors'][0];g=union_all([Polygon(c['polygon'],c['holes']) for c in door['footprint_regions']])
    assert g.intersection(mask_geometry(m)).area==0
    assert np.array_equal(before,m)
    assert abs(door['opening_width_m']-.8)<.005
    assert abs(door['host_thickness_m']-.16)<.005
    assert all(door['host_region_ids'])
    assert door['swing_status']==door['hinge_status']==door['height_status']=='not_estimated'
    assert door['scan_evidence']['status']=='not_available'


def test_closed_leaf_returns_do_not_veto_a_real_opening():
    m,p,r=fixture();barrier=np.zeros_like(m);barrier[20,10:170]=True
    d=fit_doors(p,r,m,.02,.02,barrier=barrier)
    assert len(d['doors'])==1
    assert d['doors'][0]['scan_evidence']['status']=='both_jambs_supported'


def test_single_host_seed_is_retained_for_review():
    m,p,r=fixture();m[:,100:]=False;r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    d=fit_doors(p,r,m,.02,.02)
    assert not d['doors'] and d['door_review']
    assert 'missing_two_wall_abutments' in d['door_review'][0]['reasons']


def test_window_predictions_and_existing_window_geometry_are_protected():
    m,p,r=fixture()
    window=dict(footprint_regions=[dict(polygon=[[59.5,19.5],[99.5,19.5],[99.5,27.5],[59.5,27.5]],holes=[])])
    d=fit_doors(p,r,m,.02,.02,windows=[window])
    assert not d['doors'] and 'window_conflict' in d['door_review'][0]['reasons']
    p[:,20:28,78:82]=0;p[4,20:28,78:82]=1
    assert not fit_doors(p,r,m,.02,.02)['doors']


def test_generic_opening_is_not_turned_into_a_door():
    m,p,r=fixture();p[5]=p[3];p[3]=0
    d=fit_doors(p,r,m,.02,.02)
    assert not d['doors'] and not d['door_review']


def test_door_at_transverse_wall_corner_can_use_the_real_corner_as_a_jamb():
    m,p,r=fixture();m[:,100:]=False;m[8:74,100:106]=True;r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    d=fit_doors(p,r,m,.02,.02)
    assert len(d['doors'])==1
    assert 'transverse_wall_corner' in d['doors'][0]['host_types']


def test_no_scan_evidence_is_reported_without_hallucinating_support():
    m,p,r=fixture();d=fit_doors(p,r,m,.02,.02,barrier=np.zeros_like(m))
    assert len(d['doors'])==1
    assert d['doors'][0]['scan_evidence']['status']=='partial_or_unobserved_jambs'


def test_bad_frames_are_rejected():
    m,p,r=fixture()
    with pytest.raises(ValueError,match='frames'):fit_doors(p,r,m,.0025,.02)
    with pytest.raises(ValueError,match='scan frames'):fit_doors(p,r,m,.02,.02,barrier=np.zeros((2,2),bool))


def corner_fixture():
    m=np.zeros((120,180),bool);m[10:100,20:28]=True;m[10:100,100:108]=True
    p=np.zeros((8,*m.shape),np.float32);p[0]=1;p[:,46:54,28:100]=0;p[3,46:54,28:100]=1
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))])
    counts=np.zeros_like(m,dtype=np.uint32);counts[46,28:100]=250;counts[53,28:100]=250
    return m,p,r,counts


def test_both_transverse_jambs_use_scan_band_even_without_full_height_wall_gate():
    m,p,r,counts=corner_fixture();out=fit_doors(p,r,m,.02,.02,counts=counts,barrier=np.zeros_like(m))
    assert len(out['doors'])==1,out
    d=out['doors'][0]
    assert d['host_thickness_source']=='paired_scan_edges'
    assert d['host_types']==['scan_band_transverse_abutment']*2
    assert d['paired_scan_source']=='density_qualified_door_returns'
    assert abs(d['opening_width_m']-1.44)<.01
    assert all(d['host_region_ids'])


def test_low_density_returns_cannot_supply_the_missing_corner_hosts():
    m,p,r,counts=corner_fixture();counts[counts>0]=150
    out=fit_doors(p,r,m,.02,.02,counts=counts,barrier=np.zeros_like(m))
    assert not out['doors'] and out['door_review']


def test_thick_wall_hosts_are_supported_without_displacing_thin_wall_fits():
    m,p,r=fixture();m[:]=False;m[12:42,10:60]=True;m[12:42,100:170]=True
    p[:]=0;p[0]=1;p[:,12:42,60:100]=0;p[3,12:42,60:100]=1
    r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    out=fit_doors(p,r,m,.02,.02)
    assert len(out['doors'])==1,out
    assert abs(out['doors'][0]['host_thickness_m']-.6)<.005


def test_native_2p5mm_export_uses_coarse_mask_pixel_centers_correctly():
    import cv2
    m,p,_=fixture();m=cv2.resize(m.astype('uint8'),None,fx=8,fy=8,interpolation=cv2.INTER_NEAREST)>0
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))])
    out=fit_doors(p,r,m,.0025,.02)
    assert len(out['doors'])==1,out
    door=out['doors'][0]
    assert abs(door['opening_width_m']-.8)<.0025
    assert abs(door['host_thickness_m']-.16)<.0025
    assert abs(door['axis'][0][0]-479.5)<.01
    assert abs(door['axis'][1][0]-799.5)<.01
