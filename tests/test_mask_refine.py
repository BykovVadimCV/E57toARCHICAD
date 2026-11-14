import numpy as np
from simscan.mask_refine import refine

def fixture():
 m=np.zeros((80,100),bool);m[10:70,43:53]=1;p=m.astype(float)
 x=np.zeros((8,*m.shape));return m,p,x

def test_sparse_is_exactly_neutral():
 m,p,x=fixture();x[5]=.2
 r=refine(m,p,x);assert np.array_equal(r['mask'],m);assert r['metrics']['graph_nodes']==0

def test_dense_unstructured_clutter_is_neutral():
 m,p,x=fixture();x[5]=1
 assert np.array_equal(refine(m,p,x)['mask'],m)

def test_face_moves_boundary_without_hollowing_wall():
 m,p,x=fixture();x[:3,8:72,40]=1;x[5,8:72,40]=1;x[6,8:72,40]=1
 r=refine(m,p,x)
 assert r['added'][20:60,40:43].any()
 assert r['mask'][20:60,46:50].all()
 assert np.array_equal(r['mask'][~r['editable']],m[~r['editable']])

def test_dense_returns_cannot_create_wall_from_empty():
 m,p,x=fixture();m[:]=0;p[:]=0;x[:]=1
 assert not refine(m,p,x)['mask'].any()


def test_simple_flips_preserve_thin_connections_and_holes():
 from simscan.mask_refine import _preserve_topology
 from scipy import ndimage as ndi
 original=np.zeros((30,40),bool);original[5:25,5:16]=1;original[5:25,25:35]=1;original[14,15:26]=1
 original[10:13,9:12]=0
 target=original.copy();target[14,17:24]=0;target[10:13,9:12]=1;target[1,1]=1
 out=_preserve_topology(original,target,np.zeros(original.shape))
 assert ndi.label(out,np.ones((3,3)))[1]==1
 assert ndi.label(~out)[1]==ndi.label(~original)[1]
 assert not out[1,1]


def test_fine_tiles_use_native_subcells_and_preserve_sparse_areas():
 import cv2
 from simscan.mask_refine import refine_high_resolution
 m=np.zeros((32,40),bool);m[4:28,16:22]=1
 p=cv2.resize(m.astype(np.float32),(20,16),interpolation=cv2.INTER_AREA)
 scan=np.zeros((8,32,40),np.float32);scan[:3]=1;scan[6]=1
 counts=np.zeros((128,160),np.uint32);counts[16:112,61]=10
 r=refine_high_resolution(m,p,scan,counts,tile_pixels=64)
 original=np.repeat(np.repeat(m,4,0),4,1)
 assert (r['mask']!=original).any()
 blocks=r['mask'].reshape(32,4,40,4)
 assert (blocks.max((1,3))!=blocks.min((1,3))).any()
 assert np.array_equal(r['mask'][~r['editable']],original[~r['editable']])
 empty=refine_high_resolution(m,p,scan,np.zeros_like(counts),tile_pixels=64)
 assert np.array_equal(empty['mask'],original)


def test_native_count_frame_must_match_physical_extent():
 import pytest
 from simscan.mask_refine import refine_high_resolution
 with pytest.raises(ValueError):refine_high_resolution(np.zeros((20,20),bool),np.zeros((10,10)),np.zeros((8,20,20)),np.zeros((79,80)))


def test_face_locality_keeps_opposite_wall_face_fixed():
 from simscan.mask_refine import Config
 m,p,x=fixture();m[:]=False;m[10:70,43:47]=True;p=m.astype(float)
 x[:3,10:70,40]=1;x[5,10:70,40]=1;x[6,10:70,40]=1
 r=refine(m,p,x,config=Config(boundary_locality_m=.011))
 assert not r['editable'][20:60,46].any()
 assert np.array_equal(r['mask'][20:60,46],m[20:60,46])


def test_aggressive_supported_face_moves_more_and_sparse_stays_fixed():
 from simscan.mask_refine import Config
 m,p,x=fixture();x[:3,10:70,39]=1;x[5,10:70,39]=1;x[6,10:70,39]=1
 conservative=refine(m,p,x,config=Config(boundary_locality_m=.02))
 aggressive=refine(m,p,x,config=Config(boundary_locality_m=.05,boundary_ownership=True,
                    prior_weight=.08,smoothness=12,boundary_attraction=.995,influence_m=.1))
 assert aggressive['added'].sum()>conservative['added'].sum()
 assert np.array_equal(aggressive['mask'][~aggressive['editable']],m[~aggressive['editable']])


def test_aperture_exclusion_stays_empty_despite_dense_face_evidence():
 m,p,x=fixture();forbidden=np.zeros_like(m);forbidden[35:45,39:56]=1;m&=~forbidden
 x[:3,10:70,40]=1;x[5,10:70,40]=1;x[6,10:70,40]=1
 r=refine(m,p,x,forbidden=forbidden)
 assert not r['mask'][forbidden].any()
 assert not r['editable'][forbidden].any()


def test_native_refinement_quarantines_weak_detached_patch_globally():
 import cv2
 from simscan.mask_refine import refine_high_resolution
 m=np.zeros((32,64),bool);m[4:28,10:16]=1;m[22:27,45:50]=1
 p=cv2.resize(m.astype(np.float32),(32,16),interpolation=cv2.INTER_AREA);p[10:15,21:27]=.4
 scan=np.zeros((8,32,64),np.float32);scan[:3]=1;scan[6]=1
 counts=np.zeros((128,256),np.uint32);counts[16:112,39]=10
 result=refine_high_resolution(m,p,scan,counts,tile_pixels=64)
 assert not result['mask'][88:108,180:200].any()
 assert result['mask'][30:80,48:56].all()
 assert result['metrics']['confidence_removed_pixels']>0
 assert result['metrics']['topology_blocked_changes']>=0
 assert result['quarantine']
