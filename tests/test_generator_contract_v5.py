import copy
import json
import importlib.util
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest
from simscan.netinput import (_fill,SceneGT,label_level,vector_doc,unseen_to_outside,
    rasterize_points,build_scene,mask_runs)
from simscan.layout import Layout,Room,Item,Box,Wall
from simscan.furniture import blob
from simscan.torchdata import FloorplanSamples
from simscan.dataset_contract import complete_level
from simscan.makedata import _done


def frame():return dict(origin_x=0.,origin_y=0.,pixel_m=.02,width=250,height=250)

def gt(tmp_path,items=(),walls=()):
    lay=Layout(2.8,[Room(0,(0,0,5,5),(0,0,5,5),ceiling_z=2.8)],list(walls),items=list(items))
    lay.save_json(tmp_path/'layout.json');return SceneGT(tmp_path)

def test_exact_centres_and_translations():
    for angle in [0,17,90]:
        for off in [0,.003,.01]:
            a=np.deg2rad(angle);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            poly=np.array([[-1,-.05],[1,-.05],[1,.05],[-1,.05]])@R.T+[2.5+off,2.5]
            m=np.zeros((250,250),np.uint8);_fill(m,poly,frame())
            yy,xx=np.indices(m.shape);q=(np.stack([(xx+.5)*.02,(249.5-yy)*.02],-1)-[2.5+off,2.5])@R
            expected=(q[...,0]>=-1-1e-9)&(q[...,0]<1-1e-9)&(q[...,1]>=-.05-1e-9)&(q[...,1]<.05-1e-9)
            # Exclude exact edge ties, whose half-open convention depends on polygon direction.
            edge=(abs(abs(q[...,0])-1)<1e-8)|(abs(abs(q[...,1])-.05)<1e-8)
            np.testing.assert_array_equal(m[~edge],expected[~edge])
    m=np.zeros((250,250),np.uint8);_fill(m,[[2,1],[2.1,1],[2.1,4],[2,4]],frame())
    assert m[125].sum()==5

def test_unknown_independent_of_hidden_station_flag():
    x=np.zeros((8,100,100),np.float32);x[3,:20]=1
    lab=dict(sem=np.ones((100,100),np.uint8),sem_raw=np.ones((100,100),np.uint8),
        room_id=np.ones((100,100),np.uint16),dist=np.zeros((1,100,100)))
    a,b=copy.deepcopy(lab),copy.deepcopy(lab)
    unseen_to_outside(a,x,.02,policy='ignore');unseen_to_outside(b,x,.02,no_station=[1],policy='ignore')
    np.testing.assert_array_equal(a['sem'],b['sem']);assert (a['sem']==255).any()
    assert not (a['sem']==0).any()

def test_mesh_clutter_and_column_vectors(tmp_path):
    mesh=blob([2.5,2.5],[1,.7,1.3],np.random.default_rng(4))
    g=gt(tmp_path,[Item(0,'bag','clutter',[],.5,0,[mesh])])
    assert (label_level(g,frame())['sem']==6).sum()>100
    g.layout.items=[Item(0,'column','column',[Box((2.5,2.5,1.4),(.4,.4,2.8))],.7,0)]
    assert len(vector_doc(g)['walls'])==4

def test_continuous_face_distance(tmp_path):
    g=gt(tmp_path,walls=[Wall(0,(1,2),(4,2),.1,'interior')])
    lab=label_level(g,frame());j=125;i=149 # world y=2.01; face y=2.05: 4 cm
    assert float(lab['dist'][0,i,j])==pytest.approx(.04/.3,abs=.001)

def test_incomplete_resume_rejected(tmp_path):
    d=tmp_path/'scenes/a/L1';d.mkdir(parents=True);(d/'scene.json').write_text('{}')
    assert not complete_level(d) and not _done(tmp_path,'a')

def test_bad_level_is_rejected_before_export(tmp_path,monkeypatch):
    import simscan.netinput as n
    gt(tmp_path)
    result=dict(frame=frame(),level=dict(floor_z=.2,ceiling_z=2.8),input=np.zeros((8,250,250)))
    monkeypatch.setattr(n,'rasterize_points',lambda *a,**k:[result])
    with pytest.raises(ValueError,match='Level reference error'):
        build_scene(tmp_path,tmp_path/'ds',source=object())
    assert not list((tmp_path/'ds').glob('scenes/*/L*/complete.json'))

class Points:
    def __init__(self):
        xy=np.array(np.meshgrid(np.arange(.1,2,.1),np.arange(.1,2,.1))).reshape(2,-1).T
        self.p=np.vstack([np.c_[xy,np.full(len(xy),z)] for z in [0.,.5,1.,1.5,2.8]])
        self.total=len(self.p);self.stations=np.zeros((0,3))
    def chunks(self):yield self.p

def test_reference_shift_changes_features():
    a=rasterize_points(Points(),levels=[dict(floor_z=0,ceiling_z=2.8)],log=lambda *a:None)[0]['input']
    b=rasterize_points(Points(),levels=[dict(floor_z=0,ceiling_z=2.8)],reference_shift=.04,log=lambda *a:None)[0]['input']
    assert not np.array_equal(a,b)
    assert not b[3].any() and a[3].any()

def test_vector_scope_ignores_hidden_walls_and_rooms():
    from simscan.evaluate import vector_metrics
    valid=np.ones((100,100),bool);valid[:,50:]=False
    scope=dict(frame=dict(origin_x=0,origin_y=0,pixel_m=.1,width=100,height=100),valid_runs=mask_runs(valid))
    ref=dict(walls=[dict(id='w',geometry=dict(type='segment',p0=[8,1],p1=[8,9]),thickness=.1)],
        rooms=[dict(polygon=[[7,1],[9,1],[9,9],[7,9]])],openings=[],evaluation_scope=scope)
    a=vector_metrics(dict(walls=[],rooms=[],openings=[]),ref)
    assert a['walls_gt']==0 and a['rooms_gt']==0

def test_default_erasing_is_unknown():
    import inspect
    assert inspect.signature(FloorplanSamples).parameters['erasing_policy'].default=='ignore'
    ds=object.__new__(FloorplanSamples);ds.rng=np.random.default_rng(3)
    ds.mult_noise=ds.add_noise=0;ds.p_erase=1;ds.erasing_policy='ignore';ds.unseen_outside=False
    s=dict(input=np.ones((8,60,60)),sem=np.ones((60,60),np.uint8),dist=np.zeros((1,60,60)))
    r=ds._augment(s);assert (r['sem'][(r['input']==0).all(0)]==255).all()

def test_mesh_batch_assembly_preserves_all_attributes():
    from simscan.scene import SceneMesh,Solid
    solids=[Solid(Box((i,0,1),(.1,.2,2)),'wall',i,.5,deform=True) for i in range(6)]
    fast=SceneMesh.from_solids(solids,.5)
    slow=SceneMesh.from_solids([],None)
    for solid in solids:slow=slow.concatenate(SceneMesh.from_solids([solid],.5))
    for key in ['vertices','triangles','tri_label','tri_instance','tri_reflectance','tri_transmit','vert_deform']:
        np.testing.assert_array_equal(getattr(fast,key),getattr(slow,key))

def test_resume_refuses_different_seed_without_generating(tmp_path):
    from simscan.config import SynthConfig
    from simscan.makedata import make_dataset
    cfg=SynthConfig()
    make_dataset(cfg,tmp_path,0,seed=3,log=lambda *a:None)
    make_dataset(cfg,tmp_path,0,seed=3,log=lambda *a:None)
    assert cfg.export.preview is True
    with pytest.raises(ValueError,match='build changed'):
        make_dataset(SynthConfig(),tmp_path,0,seed=4,log=lambda *a:None)

def test_scope_masks_predicted_as_well_as_reference_geometry():
    from simscan.evaluate import vector_metrics
    valid=np.ones((100,100),bool);valid[:,50:]=False
    scope=dict(frame=dict(origin_x=0,origin_y=0,pixel_m=.1,width=100,height=100),valid_runs=mask_runs(valid))
    ref=dict(walls=[dict(id='w',geometry=dict(type='segment',p0=[2,1],p1=[2,9]),thickness=.1)],
        rooms=[],openings=[],evaluation_scope=scope)
    plan=dict(walls=[dict(points=[[2,1],[2,9]],thickness=.1),dict(points=[[8,1],[8,9]],thickness=.1)],rooms=[],openings=[])
    result=vector_metrics(plan,ref)
    assert result['axis_precision']==1 and result['axis_recall']==1


def test_upper_floor_not_outvoted_by_lower_ceiling():
    from simscan.rasterize import detect_levels
    xy=np.array(np.meshgrid(np.arange(0,5,.1),np.arange(0,4,.1))).reshape(2,-1).T
    parts=[np.tile(np.c_[xy,np.full(len(xy),z)],(repeats,1)) for z,repeats in [(0,2),(2.8,12),(3.,1),(5.8,2)]]
    rng=np.random.default_rng(6)
    parts.append(np.c_[np.zeros(10000),rng.uniform(0,4,10000),rng.uniform(0,5.8,10000)])
    result=detect_levels(np.vstack(parts),stations_z=[1.5,4.5])
    assert len(result)==2 and abs(result[1]['floor_z']-3)<.03


def test_voxel_accumulation_matches_sorted_counts():
    from simscan.netinput import rasterize_points
    src=Points();r=rasterize_points(src,levels=[dict(floor_z=0,ceiling_z=2.8)],log=lambda *a:None)[0]
    assert np.isfinite(r['input']).all()
    fr=r['frame'];j=int((.1-fr['origin_x'])/.02);i=fr['height']-1-int((.1-fr['origin_y'])/.02)
    assert r['input'][5,i,j]>0


def test_voxel_dense_and_sparse_saturate_identically():
    from simscan.netinput import accumulate_voxels
    for n in [100,2_000_001]:
        x=np.zeros(n,np.uint16);x[5]=65000
        ids=np.r_[np.full(1000,5),np.full(3,7)]
        accumulate_voxels(x,ids)
        assert x[5]==65535 and x[7]==3 and x.sum(dtype=np.int64)==65538


def test_json_uses_utf8_even_with_windows_locale(tmp_path,monkeypatch):
    from simscan import dataset_contract as c
    original=c.tempfile.NamedTemporaryFile
    def windows_default(*args,**kwargs):
        kwargs.setdefault('encoding','cp1251')
        return original(*args,**kwargs)
    monkeypatch.setattr(c.tempfile,'NamedTemporaryFile',windows_default)
    doc={'orientation':'cos 2θ','label':'проём 🧱'}
    c.atomic_json(tmp_path/'meta.json',doc)
    assert json.loads((tmp_path/'meta.json').read_bytes().decode('utf-8'))==doc
