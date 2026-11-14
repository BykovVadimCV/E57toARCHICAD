import numpy as np
from simscan.wall_footprint import clean_footprint,mask_geometry,primitive
from simscan.wall_runs import _union
from simscan.decompose import rectangle


def test_variable_width_overlap_becomes_one_footprint():
    a=rectangle([35,30],50,10,0);b=rectangle([65,30],45,16,0)
    original=_union([a,b],(70,100));r=dict(primitives=[a,b],mask=original)
    d=clean_footprint(r,original,np.zeros_like(original),.02)
    assert len(d['primitives'])==1
    assert np.array_equal(_union(d['primitives'],original.shape),d['mask'])
    assert not d['contacts']
    assert d['source_primitives'] and 'parts' not in d['primitives'][0]


def test_pixel_cell_geometry_preserves_hole_and_diagonal_pixels():
    m=np.zeros((35,40),bool);m[3:30,4:35]=True;m[9:22,12:28]=False;m[1,1]=True
    from simscan.wall_footprint import regions
    ps=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))]
    assert np.array_equal(_union(ps,m.shape),m)


def test_aperture_is_not_filled_by_outline_simplification():
    m=np.zeros((50,100),bool);m[20:32,10:90]=True;m[20:24,45:52]=False
    f=np.zeros_like(m);f[20:24,45:52]=True
    from simscan.wall_footprint import regions
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))],mask=m)
    d=clean_footprint(r,m,np.zeros_like(m),.02,forbidden=f)
    assert not (d['mask']&f).any()


def test_thin_allowed_strip_is_recovered_without_filling_window():
    m=np.zeros((80,120),bool);m[30:40,10:110]=True
    f=np.zeros_like(m);f[31:40,45:75]=True;m&=~f
    broken=m.copy();broken[:,45:75]=False
    from simscan.wall_footprint import regions
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(broken)))],mask=broken)
    b=np.zeros_like(m);b[30,10:110]=True
    d=clean_footprint(r,m,b,.0025,forbidden=f)
    assert d['mask'][30,45:75].all()
    assert not (d['mask']&f).any()


def test_unobserved_T_stem_overshoot_is_trimmed():
    stem=rectangle([60,47.5],65,12,np.pi/2)
    crossbar=rectangle([60,25],100,10,0)
    m=_union([stem,crossbar],(100,130));b=np.zeros_like(m)
    b[30,10:110]=True;b[31:80,54]=True;b[31:80,66]=True
    r=dict(primitives=[stem,crossbar],mask=m)
    d=clean_footprint(r,m,b,.01)
    assert any(e['operation']=='trim_unobserved_T_overshoot' for e in d['footprint_cleanup']['edits'])
    assert not d['mask'][15:19,56:65].any()
    assert d['mask'][40:75,56:65].all()


def test_enclosed_fitting_slit_filled_but_semantic_hole_preserved():
    from simscan.wall_footprint import regions
    target=np.zeros((80,160),bool);target[20:44,10:150]=True
    m=target.copy();m[30:32,40:120]=False
    r=dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(m)))],mask=m)
    d=clean_footprint(r,target,np.zeros_like(m),.0025)
    assert d['mask'][30:32,40:120].all()
    forbidden=np.zeros_like(m);forbidden[30:32,40:120]=True
    allowed=target&~forbidden
    d=clean_footprint(r,allowed,np.zeros_like(m),.0025,forbidden=forbidden)
    assert not (d['mask']&forbidden).any()
