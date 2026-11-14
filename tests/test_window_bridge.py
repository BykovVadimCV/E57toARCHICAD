import numpy as np
from shapely.geometry import Polygon
from shapely import union_all
from simscan.window_bridge import fit_windows
from simscan.wall_footprint import mask_geometry,regions,primitive


def fixture():
    m=np.zeros((80,180),bool);m[20:28,10:60]=True;m[20:28,100:170]=True
    p=np.zeros((8,*m.shape),np.float32);p[0]=1;p[:,18:30,60:100]=0;p[4,18:30,60:100]=1
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))],wall_faces=[dict(start=[10,24],end=[170,24])])
    return m,p,r


def test_window_uses_wall_width_and_closes_gap_without_wall_overlap():
    m,p,r=fixture();before=m.copy();out=fit_windows(p,r,m,.02,.02)
    assert len(out['windows'])==1
    w=out['windows'][0];assert abs(w['width_m']-.16)<.005
    g=union_all([Polygon(a['polygon'],a['holes']) for a in w['footprint_regions']])
    walls=mask_geometry(m)
    assert g.intersection(walls).area<1e-6
    assert g.distance(Polygon(r['primitives'][0]['polygon']))<1e-6
    assert g.distance(Polygon(r['primitives'][1]['polygon']))<1e-6
    assert np.array_equal(m,before)


def test_single_wall_support_requires_review():
    m,p,r=fixture();m[:,100:]=False;r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    out=fit_windows(p,r,m,.02,.02)
    assert not out['windows'] and out['window_review']


def test_window_cannot_fill_door_region():
    m,p,r=fixture();p[:,20:28,76:84]=0;p[3,20:28,76:84]=1
    out=fit_windows(p,r,m,.02,.02)
    assert not out['windows']


def test_window_can_terminate_at_a_transverse_corner_wall():
    m,p,r=fixture();m[:,100:]=False;m[8:74,100:106]=True
    r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    out=fit_windows(p,r,m,.02,.02)
    assert len(out['windows'])==1
    assert 'transverse_wall_corner' in out['windows'][0]['host_types']


def test_different_host_widths_are_connected_without_changing_walls():
    m,p,r=fixture();m[18:30,100:170]=True
    r['primitives']=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    out=fit_windows(p,r,m,.02,.02)
    assert len(out['windows'])==1
    widths=out['windows'][0]['host_widths_m']
    assert abs(widths[0]-widths[1])>.04
    assert out['windows'][0]['height_status']=='not_estimated'


def test_window_axis_ends_at_gap_and_records_host_regions():
    m,p,r=fixture();w=fit_windows(p,r,m,.02,.02)['windows'][0]
    assert abs(w['length_m']-.8)<.01
    assert len(w['host_region_ids'])==2 and all(w['host_region_ids'])
    assert abs(w['axis'][0][0]-59.5)<.1
    assert abs(w['axis'][1][0]-99.5)<.1
