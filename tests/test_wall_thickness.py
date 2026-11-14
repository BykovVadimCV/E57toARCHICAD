import numpy as np
from simscan.wall_thickness import stabilize_thickness
from simscan.wall_footprint import mask_geometry,regions,primitive


def source(mask):return dict(primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(mask)))],mask=mask)


def test_long_run_inferred_width_is_more_consistent():
    m=np.zeros((130,700),bool)
    for x in range(20,680):m[40:40+int(20+4*np.sin(x/35)),x]=True
    b=np.zeros_like(m);b[40,20:680]=True
    d=stabilize_thickness(source(m),m,b,.005)
    assert d['thickness_regularization']['edits']
    assert all(e['changed_pixels']>0 for e in d['thickness_regularization']['edits'])
    before=np.std(m[:,60:640].sum(0));after=np.std(d['mask'][:,60:640].sum(0))
    assert after<before*.7
    assert d['mask'][40,20:680].all()


def test_opposite_measured_face_and_aperture_remain():
    m=np.zeros((130,700),bool)
    for x in range(20,680):m[40:40+int(20+4*np.sin(x/35)),x]=True
    b=np.zeros_like(m);b[40,20:680]=True
    for x in range(20,680):b[np.where(m[:,x])[0][-1],x]=True
    forbidden=np.zeros_like(m);forbidden[:,340:355]=True;m&=~forbidden;b&=~forbidden
    d=stabilize_thickness(source(m),m,b,.005,forbidden)
    assert not (d['mask']&forbidden).any()
    assert np.all(d['mask'][b])


def test_perpendicular_branch_stays_connected():
    import cv2
    m=np.zeros((160,700),bool)
    for x in range(20,680):m[40:40+int(20+3*np.sin(x/35)),x]=True
    m[40:140,340:360]=True
    b=np.zeros_like(m);b[40,20:680]=True;b[40:140,340]=True;b[40:140,359]=True
    d=stabilize_thickness(source(m),m,b,.005)
    assert d['mask'][60:135,344:356].all()
    assert cv2.connectedComponents(d['mask'].astype('uint8'),8)[0]==2


def test_short_scan_fragments_are_assembled_before_length_filter():
    m=np.zeros((130,700),bool)
    for x in range(20,680):m[40:40+int(20+4*np.sin(x/35)),x]=True
    b=np.zeros_like(m)
    for x in range(20,650,100):b[40,x:x+85]=True
    d=stabilize_thickness(source(m),m,b,.005)
    info=d['thickness_regularization']
    assert info['assembled_scan_runs']<info['raw_scan_runs']
    assert any(e['merged_scan_fragments']>1 for e in info['edits'])
    assert np.std(d['mask'][:,60:640].sum(0))<np.std(m[:,60:640].sum(0))


def test_paired_faces_remove_short_raster_notch_in_long_wall():
    m=np.zeros((130,700),bool);m[40:61,20:680]=True;m[40:47,320:340]=False
    b=np.zeros_like(m);b[40,20:680]=True;b[60,20:680]=True
    d=stabilize_thickness(source(m),m,b,.005)
    assert d['mask'][41:47,320:340].all()
    assert any(e['mode']=='paired_parallel_faces' for e in d['thickness_regularization']['edits'])


def test_paired_faces_do_not_bridge_a_semantic_opening():
    m=np.zeros((130,700),bool);m[40:61,20:680]=True
    f=np.zeros_like(m);f[40:61,320:340]=True;m&=~f
    b=np.zeros_like(m);b[40,20:680]=True;b[60,20:680]=True
    d=stabilize_thickness(source(m),m,b,.005,forbidden=f)
    assert not (d['mask']&f).any()
