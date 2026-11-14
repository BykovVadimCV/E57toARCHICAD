from pathlib import Path
import numpy as np
import cv2
from simscan.wall_small_features import recover_small_features
from simscan.wall_footprint import mask_geometry,regions,primitive
from simscan.wall_runs import _union


def source(mask):return dict(mask=mask,primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(mask)))])


def fixture():
    target=np.zeros((150,220),bool);target[40:110,140:158]=True;target[70:85,20:158]=True
    current=np.zeros_like(target);current[70:85,20:158]=True
    barrier=np.zeros_like(target);barrier[40:110,157]=True;barrier[40:70,140]=True;barrier[85:110,140]=True
    barrier[70,20:140]=True;barrier[84,20:140]=True
    f=np.zeros_like(target);f[:40,140:158]=True;f[110:,140:158]=True
    return current,target,barrier,f


def test_T_cap_recovers_with_one_parent_and_partial_opposite_faces():
    current,target,b,f=fixture();d=recover_small_features(source(current),target,b,.005,f)
    assert d['mask'][50:65,144:153].all()
    assert d['mask'][90:105,144:153].all()
    assert d['mask'][current].all()
    assert not (d['mask']&f).any()
    assert d['small_feature_recovery']['edits']
    assert np.array_equal(_union(d['primitives'],current.shape),d['mask'])
    assert cv2.connectedComponents(d['mask'].astype('uint8'),8)[0]==2


def test_sparse_returns_do_not_grow_a_cap():
    current,target,b,f=fixture();d=recover_small_features(source(current),target,np.zeros_like(b),.005,f)
    assert np.array_equal(d['mask'],current)


def test_a_single_dense_face_cannot_invent_an_opposite_wall_edge():
    current,target,b,f=fixture();b[:]=False;b[40:110,157]=True
    d=recover_small_features(source(current),target,b,.005,f)
    assert np.array_equal(d['mask'],current)


def test_detached_short_prediction_is_not_promoted_to_an_active_wall():
    current,target,b,f=fixture();current[:]=False
    d=recover_small_features(source(current),target,b,.005,f)
    assert not d['mask'].any()


def test_real_cap_between_two_doors_retains_both_openings():
    z=np.load(Path(__file__).parent/'fixtures/short_T_cap_good.npz');m=z['mask'];d=recover_small_features(source(m),z['target'],z['barrier'],.0025,z['forbidden'])
    assert d['mask'][92,175] and not m[92,175]
    assert d['mask'][162,180] and not m[162,180]
    assert d['mask'][m].all()
    assert not (d['mask']&z['forbidden']).any()
    assert d['small_feature_recovery']['edits']


def test_a_minor_edge_bite_is_left_as_it_was():
    _,target,b,f=fixture();current=target.copy();current[42:45,145:150]=False
    d=recover_small_features(source(current),target,b,.005,f)
    assert np.array_equal(d['mask'],current)


def test_thick_wall_T_cap_is_not_rejected_by_a_thin_wall_width_limit():
    target=np.zeros((150,230),bool);target[40:95,120:200]=True;target[60:76,10:200]=True
    current=np.zeros_like(target);current[60:76,10:200]=True
    b=np.zeros_like(target);b[40:95,199]=True;b[40:60,120]=True;b[76:95,120]=True;b[60,10:120]=True;b[75,10:120]=True
    f=np.zeros_like(target);f[:40,120:200]=True;f[95:,120:200]=True
    d=recover_small_features(source(current),target,b,.005,f)
    assert d['mask'][46:54,130:190].all()
    assert d['mask'][81:89,130:190].all()
    assert not (d['mask']&f).any()
