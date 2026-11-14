import numpy as np
import pytest
from simscan.wall_reference_guard import select_native_fit,barrier_sha256
from simscan.wall_footprint import mask_geometry,regions,primitive
from simscan.decompose import Config


def result(mask):return dict(mask=mask,primitives=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(mask)))])


def fixture():
    target=np.zeros((80,180),bool);target[20:30,10:170]=True
    barrier=np.zeros_like(target);barrier[20,10:170]=True
    f=np.zeros_like(target)
    ref=dict(result=result(target),target=target.copy(),forbidden=f.copy(),pixel_m=.02,barrier_sha256=barrier_sha256(barrier))
    return target,barrier,f,ref


def test_previously_matched_native_face_is_not_lost_to_a_fresh_refit():
    target,b,f,ref=fixture();m=target.copy();m[:,70:90]=False
    d=select_native_fit(result(m),ref,target,b,f,.02,Config())
    assert np.array_equal(d['mask'],target)
    assert d['native_reference_guard']['selected']=='accepted_native'
    assert d['native_reference_guard']['lost_candidate_face_samples']>0


def test_unchanged_or_improved_native_fit_is_kept():
    target,b,f,ref=fixture();d=select_native_fit(result(target.copy()),ref,target,b,f,.02,Config())
    assert d['native_reference_guard']['selected']=='fresh_fit'


def test_reference_for_different_prediction_or_scan_is_rejected():
    target,b,f,ref=fixture();ref['target'][50,50]=True
    with pytest.raises(ValueError,match='does not match'):select_native_fit(result(target),ref,target,b,f,.02,Config())
    ref['target']=target.copy();ref['barrier_sha256']='different scan'
    with pytest.raises(ValueError,match='does not match'):select_native_fit(result(target),ref,target,b,f,.02,Config())
