import numpy as np
from shapely.geometry import Polygon,box
from simscan.dense_face_fit import align_dense_faces


def test_continuous_dense_face_aligns_existing_edge():
    wall=box(0,0,5,.2)
    points=np.c_[np.linspace(.02,4.98,1000),np.full(1000,-.02)]
    result,edits=align_dense_faces([wall],points,Polygon())
    assert len(edits)==1
    assert abs(result[0].bounds[1]+.02)<.003
    assert abs(result[0].bounds[3]-.2)<1e-9
    assert result[0].bounds[0]==0 and result[0].bounds[2]==5


def test_scattered_clutter_cannot_pull_wall_face():
    wall=box(0,0,5,.2)
    points=np.c_[np.linspace(2,2.2,1000),np.full(1000,-.02)]
    result,edits=align_dense_faces([wall],points,Polygon())
    assert not edits and result[0].equals(wall)


def test_empty_input_preserves_geometry():
    wall=box(0,0,5,.2)
    result,edits=align_dense_faces([wall],np.empty((0,2)),Polygon())
    assert not edits and result[0].equals(wall)


def test_nearby_opposite_face_does_not_collapse_wall():
    wall=box(0,0,5,.1)
    points=np.c_[np.linspace(.02,4.98,1000),np.full(1000,.095)]
    result,edits=align_dense_faces([wall],points,Polygon(),max_shift=.12)
    assert result[0].bounds[3]-result[0].bounds[1] >= .07


def test_wide_wall_end_cap_is_not_treated_as_long_face():
    wall=box(0,0,5,.6)
    points=np.c_[np.full(1000,5.03),np.linspace(.02,.58,1000)]
    result,edits=align_dense_faces([wall],points,Polygon(),min_length=.4,max_shift=.05)
    assert not edits and result[0].equals(wall)


def test_skewed_end_caps_stay_on_original_lines():
    wall=Polygon([(0,0),(5,0),(5.04,.2),(.04,.2)])
    points=np.c_[np.linspace(.1,4.9,1000),np.full(1000,-.02)]
    result,edits=align_dense_faces([wall],points,Polygon())
    assert len(edits)==1
    coords=np.array(result[0].exterior.coords)
    assert abs(coords[0,0]-.2*coords[0,1])<1e-8
    assert abs(coords[1,0]-(5+.2*coords[1,1]))<1e-8
