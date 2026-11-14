import cv2
import numpy as np
from simscan.wall_face_alignment import align_observed_faces
from simscan.wall_runs import extract_runs
from simscan.decompose import Config


def example():
    # A 48cm exposed face, with a formerly vertical 7cm wall behind it.
    mask=np.zeros((300,240),bool);mask[40:232,70:98]=True
    barrier=np.zeros_like(mask)
    cv2.line(barrier.view('uint8'),(98,40),(117,231),1,2)
    target=np.zeros_like(mask)
    cv2.fillPoly(target.view('uint8'),[np.array([[62,40],[98,40],[117,231],[81,231]])],1)
    return mask,target,barrier


def test_short_one_sided_face_changes_direction_without_inventing_width():
    mask,target,barrier=example();cfg=Config();f=np.zeros_like(mask)
    # This is precisely the valid extracted face missed by paired recovery and
    # the >=1m width regularizer in the old pipeline.
    runs=extract_runs(mask,barrier,.0025,cfg,f)
    out,d=align_observed_faces(mask,target,barrier,f,runs,.0025,cfg)
    assert d['edits']
    assert out[210,110] and not mask[210,110]
    assert all(e['face_distance_p90_after_m']<=.0075 for e in d['edits'])
    assert all(abs(e['width_m']-.07)<.0075 for e in d['edits'])
    assert np.max(out.sum(1))<=31


def test_crossing_blob_does_not_replace_the_continuous_face():
    mask,target,barrier=example();barrier[120:128,72:148]=True
    cfg=Config();f=np.zeros_like(mask);runs=extract_runs(mask,barrier,.0025,cfg,f)
    out,d=align_observed_faces(mask,target,barrier,f,runs,.0025,cfg)
    # If the blob survives the upstream density/verticality gates, it is a
    # constraint, not an excuse to select its perpendicular edge as the wall.
    assert all(e['line']['u'][1]>.9 for e in d['edits'])
    if not d['edits']:
        assert np.array_equal(out,mask)
    else:
        assert out[210,110]


def test_semantic_opening_is_never_filled():
    mask,target,barrier=example();f=np.zeros_like(mask);f[125:145]=True
    mask&=~f;target&=~f;cfg=Config()
    out,_=align_observed_faces(mask,target,barrier,f,extract_runs(mask,barrier,.0025,cfg,f),.0025,cfg)
    assert not (out&f).any()


def test_an_already_measured_opposite_face_is_preserved():
    mask,target,barrier=example();barrier[40:232,70]=True;cfg=Config();f=np.zeros_like(mask)
    out,_=align_observed_faces(mask,target,barrier,f,extract_runs(mask,barrier,.0025,cfg,f),.0025,cfg)
    # An inferred back must never erase this independently measured edge.
    assert out[50:220,70].all()


def test_real_good_scan_screenshot_corner_tracks_the_exposed_face():
    from pathlib import Path
    z=np.load(Path(__file__).parent/'fixtures/good_short_exterior_face.npz')
    cfg=Config();mask=z['mask'];target=z['target'];barrier=z['barrier'];f=z['forbidden']
    runs=extract_runs(mask,barrier,.0025,cfg,f)
    out,d=align_observed_faces(mask,target,barrier,f,runs,.0025,cfg)
    assert d['edits'],d
    # Interior of the screenshot's displaced lower half is recovered, while
    # the window below and the unrelated blob remain outside the wall.
    assert out[265,90] and not mask[265,90]
    assert not out[f].any()
    e=next(e for e in d['edits'] if e['line']['u'][1]>.9)
    assert e['face_distance_p90_before_m']>.03
    assert e['face_distance_p90_after_m']<=.0075
    assert .065<=e['width_m']<=.08
