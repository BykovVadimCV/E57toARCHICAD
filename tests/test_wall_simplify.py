import numpy as np
from simscan.decompose import rectangle,rasterize,Config
from simscan.wall_simplify import simplify_result


def fixture():
 shape=(180,800);parts=[rectangle([195,102.5],350,35,0),rectangle([575,100],410,40,0)]
 mask=np.zeros(shape,bool)
 for c in parts:mask|=rasterize(c,shape)
 barrier=np.zeros(shape,bool);barrier[80,20:780]=1;barrier[120,20:780]=1
 return dict(primitives=parts,mask=mask,config={}),mask,barrier


def test_small_fitting_steps_become_one_scan_fitted_wall():
 result,target,barrier=fixture()
 out=simplify_result(result,target,barrier,.005,target.astype(float))
 assert out['simplification']['edits']
 assert len(out['primitives'])==1
 assert out['mask'][80:121,30:770].all()
 assert out['simplification']['after']['matched_length_m']>=out['simplification']['before']['matched_length_m']
 assert out['simplification']['edits'][0]['max_boundary_shift_m']<=.04+.005


def test_no_flattening_when_only_one_face_is_observed():
 result,target,barrier=fixture();barrier[120]=False
 out=simplify_result(result,target,barrier,.005,target.astype(float))
 assert not out['simplification']['edits']
 assert np.array_equal(out['mask'],target)


def test_forbidden_aperture_is_not_filled_by_simplification():
 result,target,barrier=fixture();forbidden=np.zeros_like(target);forbidden[80:85,100:120]=1
 assert not (target&forbidden).any()
 out=simplify_result(result,target,barrier,.005,target.astype(float),forbidden)
 assert not (out['mask']&forbidden).any()


def test_true_large_thickness_change_in_scan_remains():
 result,target,barrier=fixture();barrier[:]=False;barrier[60,20:370]=1;barrier[80,370:780]=1;barrier[120,20:780]=1
 out=simplify_result(result,target,barrier,.005,target.astype(float))
 assert not out['simplification']['edits']


def test_only_small_narrow_enclosed_slits_are_ignored_for_outline_measurement():
 from simscan.wall_simplify import _outline_measurement
 mask=np.zeros((160,800),bool);mask[80:121,20:780]=1
 mask[99:102,390:410]=0;mask[90:110,450:480]=0
 filled,protected=_outline_measurement(mask,.005,Config())
 assert filled[99:102,390:410].all()
 assert not filled[90:110,450:480].any()
 assert protected[90:110,450:480].all()
