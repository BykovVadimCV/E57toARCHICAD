import json
import numpy as np
import cv2
from simscan.hddata import load_patch
from simscan.hd_evidence import SCAN_HD_FILES,project_scan_bands
from simscan.semantic_contract import HD_MASK_FORMAT, HD_OPENING_VALUES, OPENING_SEMANTICS_VERSION
from simscan.observation import OBSERVATION_CONTRACT, OBSERVATION_FILES


def tiny_level(p):
    frame=dict(origin_x=0.,origin_y=0.,pixel_m=.002,width=100,height=80)
    coarse=dict(origin_x=0.,origin_y=0.,pixel_m=.02,width=10,height=8)
    (p/'wall_mask_hd.json').write_text(json.dumps(dict(
        frame=frame,format=HD_MASK_FORMAT,opening_semantics=OPENING_SEMANTICS_VERSION,
        observation_contract=OBSERVATION_CONTRACT,
        opening_values=dict(HD_OPENING_VALUES))))
    (p/'scan_evidence_hd.json').write_text(json.dumps(dict(frame=frame,low_res_frame=coarse,
        observation=dict(format=OBSERVATION_CONTRACT))))
    x=np.zeros((8,8,10),np.float16);x[5]=.5;np.save(p/'input.npy',x)
    for name in SCAN_HD_FILES:cv2.imwrite(str(p/name),np.zeros((80,100),np.uint8))
    for name in ['wall_mask_hd.png','openings_hd.png','floor_void_hd.png','wall_edges_hd.png','opening_edges_hd.png']:
        cv2.imwrite(str(p/name),np.zeros((80,100),np.uint8))
    cv2.imwrite(str(p/'supervision_valid_hd.png'),np.full((80,100),255,np.uint8))
    for name in OBSERVATION_FILES:
        cv2.imwrite(str(p/name),np.full((80,100),255 if name=='training_valid_hd.png' else 0,np.uint8))
    np.savez_compressed(p/'scan_counts_hd.npz',counts=np.zeros((80,100),np.uint32))


def test_hd_input_does_not_leak_geometry_targets(tmp_path):
    tiny_level(tmp_path);a=load_patch(tmp_path,8,8,64)
    wall=np.zeros((80,100),np.uint8);wall[20:30,30:50]=255;cv2.imwrite(str(tmp_path/'wall_mask_hd.png'),wall)
    surf=np.zeros((80,100),np.uint8);surf[20,30:50]=255
    cv2.imwrite(str(tmp_path/'observed_wall_surface_hd.png'),surf)
    valid=np.full((80,100),255,np.uint8);valid[20:30,30:50]=0;valid[20,30:50]=255
    cv2.imwrite(str(tmp_path/'training_valid_hd.png'),valid)
    b=load_patch(tmp_path,8,8,64)
    assert a['input'].shape==(15,64,64)
    np.testing.assert_array_equal(a['input'],b['input'])
    assert (b['sem']==1).sum()==20  # only measured surface, not 200-px wall volume
    assert np.allclose(b['input'][12,12:40,12:40],.5)  # coarse density feature


def test_raw_count_multiplicity_is_preserved():
    class S:
        def chunks(self):yield np.array([[.01,.01,1.3]]*25+[[.03,.03,1.3]])
    frame=dict(origin_x=0.,origin_y=0.,pixel_m=.002,width=40,height=40)
    bands,stats,counts=project_scan_bands(S(),frame,0.,2.7,return_counts=True)
    assert counts.dtype==np.uint32 and counts.max()==25 and counts.sum()==26
    assert np.count_nonzero(bands['scan_mid_hd.png'])==2


def test_hd_model_training_step_when_torch_available():
    import pytest
    torch=pytest.importorskip('torch')
    from simscan.hdnet import HDRefiner,loss
    m=HDRefiner(width=8);x=torch.rand(2,15,32,40)
    b=dict(sem=torch.zeros(2,32,40,dtype=torch.long),valid=torch.ones(2,32,40,dtype=torch.bool),edge=torch.zeros(2,1,32,40))
    out=m(x);assert out['sem'].shape==(2,6,32,40)
    v=loss(out,b);assert torch.isfinite(v);v.backward()
    assert m.a[0].weight.grad is not None


def test_native_writer_completion_counts_and_training_patch(tmp_path):
    from simscan.layout import Layout,Room,Wall
    from simscan.hdmask import write_hd_level,complete_hd_level
    scene=tmp_path/'scene';scene.mkdir();level=tmp_path/'level';level.mkdir()
    layout=Layout(2.7,[Room(0,(0,0,.12,.08),(0,0,.12,.08),ceiling_z=2.7)],
                  [Wall(0,(0,0),(.12,0),.02,'interior')])
    (scene/'layout.json').write_text(json.dumps(layout.to_dict()))
    frame=dict(origin_x=-.02,origin_y=-.02,pixel_m=.02,width=10,height=8)
    (level/'scene.json').write_text(json.dumps(dict(level='L1',frame=frame,floor_z=0.,ceiling_z=2.7,aug={},raster_sample_seed=1)))
    np.save(level/'input.npy',np.zeros((8,8,10),np.float16));np.save(level/'supervision_valid.npy',np.ones((8,10),bool))
    class Source:
        def chunks(self):yield np.array([[.02,.01,1.3]]*25+[[.03,.02,0.]])
    from simscan.ray_evidence import RaySamples
    samples = RaySamples(np.array([[.01,.06,1.3]]), np.array([[.25,.06,1.3]]),
                         np.array([0],dtype=np.int64), 26, 1)
    write_hd_level(scene,level,2.,(32,32),source=Source(),ray_samples=samples)
    assert complete_hd_level(level,2.,(32,32))
    metadata=json.loads((level/'scan_evidence_hd.json').read_text())
    assert metadata['observation']['free_rays_available'] is True
    assert metadata['observation']['ray_provenance']['station_count']==1
    assert cv2.imread(str(level/'ray_free_mid_hd.png'),0).any()
    assert cv2.imread(str(level/'observation_confirmed_passage_hd.png'),0).max()==0
    with np.load(level/'scan_counts_hd.npz') as counts:assert counts['counts'].sum()==25
    p=load_patch(level,0,0,64);assert p['input'].shape==(15,64,64)
    marker=json.loads((level/'hd_complete.json').read_text());assert 'input.npy' in marker['files']
    (level/'scan_counts_hd.npz').write_bytes(b'broken')
    assert not complete_hd_level(level,2.,(32,32))


def test_hd_patch_physical_opening_classes_are_consistent(tmp_path):
    tiny_level(tmp_path)
    aperture=np.zeros((80,100),np.uint8)
    aperture[8:12,8:12]=1   # door
    aperture[18:22,18:22]=2 # window
    aperture[28:32,28:32]=3 # passage
    cv2.imwrite(str(tmp_path/'openings_hd.png'),aperture)
    result=load_patch(tmp_path,0,0,64)
    assert result['sem'][10,10]==2
    assert result['sem'][20,20]==3
    assert result['sem'][30,30]==4


def test_hd_loader_rejects_wall_body_positive_leakage(tmp_path):
    import pytest
    tiny_level(tmp_path)
    wall=np.zeros((80,100),np.uint8);wall[20:30,30:50]=255
    cv2.imwrite(str(tmp_path/'wall_mask_hd.png'),wall)
    with pytest.raises(ValueError,match='Full wall interior'):
        load_patch(tmp_path,8,8,64)
