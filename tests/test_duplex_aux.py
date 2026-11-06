import json
import numpy as np
import pytest
from simscan.config import SynthConfig
from simscan.duplex import make_duplex
from simscan.layout import Layout,Room
from simscan.netinput import SceneGT,label_level


def test_one_duplex_geometry_and_auxiliary_labels(tmp_path):
    cfg=SynthConfig();cfg.layout.duplex_slab_m=(.31,.31);cfg.layout.duplex_p_spiral=0
    lower=Layout(2.8,[Room(0,(0,0,8,6),(0,0,8,6),ceiling_z=2.8)],[])
    made=make_duplex(lower,cfg,np.random.default_rng(7));assert made is not None
    assert made['dz']==pytest.approx(3.11)
    assert lower.meta['duplex']['slab_m']==pytest.approx(.31)
    assert lower.meta['slab']['ceiling']['thickness']==pytest.approx(.155)
    upper=made['upper'];doc=dict(levels=[dict(z0=0,layout=lower.to_dict()),dict(z0=made['dz'],layout=upper.to_dict())],meta={})
    (tmp_path/'layout.json').write_text(json.dumps(doc))
    frame=dict(origin_x=0.,origin_y=0.,pixel_m=.02,width=400,height=300)
    a=label_level(SceneGT(tmp_path,0),frame);b=label_level(SceneGT(tmp_path,1),frame)
    assert a['aux']['stair_footprint'].any() and a['aux']['ceiling_opening'].any()
    assert not a['aux']['floor_opening'].any()
    assert b['aux']['floor_opening'].any() and not b['aux']['stair_footprint'].any()
    assert b['aux']['railing'].any()
    assert a['sem'].shape==b['sem'].shape==(frame['height'],frame['width'])


def test_turning_stair_and_landing_are_generated_when_space_allows(tmp_path):
    cfg=SynthConfig();cfg.layout.duplex_p_turning=1.;cfg.layout.duplex_p_spiral=0.
    lower=Layout(2.8,[Room(0,(0,0,8,6),(0,0,8,6),ceiling_z=2.8)],[])
    made=make_duplex(lower,cfg,np.random.default_rng(4));assert made and made['stair']['type']=='turning'
    stair=made['stair'];assert stair['n1']+stair['n2']==stair['n']
    boxes=next(i.boxes for i in lower.items if i.kind=='stairs');assert len(boxes)>stair['n']+1
    assert np.isclose(boxes[stair['n']].center[2]+boxes[stair['n']].size[2]/2,made['dz'])
    doc=dict(levels=[dict(z0=0,layout=lower.to_dict()),dict(z0=made['dz'],layout=made['upper'].to_dict())],meta={})
    (tmp_path/'layout.json').write_text(json.dumps(doc))
    frame=dict(origin_x=0.,origin_y=0.,pixel_m=.02,width=400,height=300)
    aux=label_level(SceneGT(tmp_path,0),frame)['aux']
    assert aux['landing'].any() and np.all(aux['stair_footprint'][aux['landing']>0])
    assert aux['railing'].any()


def test_aux_loader_handles_old_and_partially_annotated_datasets(tmp_path):
    from simscan.torchdata import load_level,AUX_NAMES
    np.save(tmp_path/'input.npy',np.zeros((8,12,14),np.float32))
    old=load_level(tmp_path,use_aux=True);assert old['aux'].shape==(5,12,14) and not old['aux_valid'].any()
    mask=np.zeros((12,14),np.uint8);mask[3:6,4:8]=1
    np.savez_compressed(tmp_path/'structure_aux.npz',stair_footprint=mask)
    new=load_level(tmp_path,use_aux=True);assert np.array_equal(new['aux'][0],mask)
    assert new['aux_valid'][0].all() and not new['aux_valid'][1:].any()
    assert 'aux' not in load_level(tmp_path)


def test_auxiliary_targets_follow_exact_spatial_augmentation():
    from simscan.torchdata import _flip_x,_rot90
    x=np.zeros((5,8,10),np.float32);x[0,1:3,2:5]=1
    sample=dict(input=np.zeros((8,8,10),np.float32),aux=x,aux_valid=np.ones_like(x,dtype=bool))
    flipped=_flip_x(sample);turned=_rot90(sample)
    assert np.array_equal(flipped['aux'][0],x[0,:,::-1])
    assert np.array_equal(turned['aux'][0],np.rot90(x[0]))
    assert turned['aux_valid'].all()


def test_random_station_placement_uses_clear_spots():
    from simscan.scanner import place_stations
    cfg=SynthConfig().scanner;cfg.station_random_fraction=1.;cfg.stations_per_m2=.2;cfg.max_stations_per_room=4
    layout=Layout(2.8,[Room(0,(0,0,8,6),(0,0,8,6),ceiling_z=2.8)],[])
    points=np.array([(x,y) for x in range(1,8) for y in range(1,6)],float)
    class Grid:
        def free_points(self,clearance): return points
    stations=place_stations(layout,{0:Grid()},cfg,np.random.default_rng(8))
    assert len(stations)==4
    xy=np.array([s.position[:2] for s in stations])
    assert all(any(np.array_equal(p,q) for q in points) for p in xy)
    assert all(np.linalg.norm(xy[i]-xy[j])>=.5 for i in range(len(xy)) for j in range(i))


def test_turning_stair_has_reserved_approach_at_both_levels():
    cfg=SynthConfig();cfg.layout.duplex_p_turning=1.;cfg.layout.duplex_p_spiral=0.
    lower=Layout(2.8,[Room(0,(0,0,8,6),(0,0,8,6),ceiling_z=2.8)],[])
    made=make_duplex(lower,cfg,np.random.default_rng(4));st=made['stair']
    assert st['type']=='turning'
    a=st['approach']; fp=st['footprint']
    assert a[2]-a[0]>=.89 and (a[2]==fp[0] or a[0]==fp[2])
    assert a in lower.meta['reserved'] and a in made['upper'].meta['reserved']
    assert a[0]>=lower.rooms[0].clear[0]


def test_upper_terrace_changes_only_eligible_room_and_exterior_segment(tmp_path):
    from simscan.layout import Wall,Opening
    from simscan.duplex import _make_upper_terrace
    from simscan.scene import SceneBuilder
    rooms=[Room(0,(0,0,4,5),(0,0,4,5),ceiling_z=2.8), Room(1,(4,0,8,5),(4,0,8,5),ceiling_z=2.5)]
    walls=[Wall(0,(4,0),(8,0),.3,'exterior'),Wall(1,(4,0),(4,5),.15,'interior')]
    opening=Opening(0,0,'window',2.,1.,.8,2.,(1,))
    upper=Layout(2.8,rooms,walls,[opening])
    assert _make_upper_terrace(upper,{'room_id':0},np.random.default_rng(3))
    assert not rooms[0].open_sky and rooms[1].open_sky
    assert walls[0].kind=='parapet' and walls[0].role=='open_balcony'
    assert walls[1].kind=='interior' and not upper.openings
    solids=SceneBuilder(upper).build()
    assert not any(s.source=='suspended:1' for s in solids)
    assert any(s.source=='slab:balcony1' for s in solids)
    (tmp_path/'layout.json').write_text(json.dumps(upper.to_dict()))
    lab=label_level(SceneGT(tmp_path),dict(origin_x=0.,origin_y=0.,pixel_m=.1,width=80,height=50))
    # The existing dataset contract treats an open terrace as outside.
    assert lab['sem'][25,60] == 0 and lab['sem'][25,20] == 1


def test_grouped_panels_are_nonstructural_and_respect_reservations():
    from simscan.interior import Furnisher,RoomGrid
    cfg=SynthConfig().interior
    room=Room(0,(0,0,8,6),(0,0,8,6),ceiling_z=2.8)
    layout=Layout(2.8,[room],[]);grid=RoomGrid(room)
    furnisher=Furnisher(cfg,np.random.default_rng(1))
    furnisher._clutter_group(layout,room,grid)
    assert len(layout.items)==3 and all(i.label=='clutter' for i in layout.items)
    assert grid.occ.any()


def test_limited_elevation_sampling_preserves_original_config():
    from simscan.generate import choose_scan_config
    cfg=SynthConfig().scanner
    cfg.p_limited_ceiling_view=1.;cfg.limited_elevation_max_deg=(30.,40.)
    limited=choose_scan_config(cfg,np.random.default_rng(1))
    assert 30 <= limited.elevation_max_deg <= 40 and cfg.elevation_max_deg==90.
    assert limited.elevation_min_deg==cfg.elevation_min_deg
    cfg.limited_elevation_max_deg=(95.,100.)
    with pytest.raises(ValueError): choose_scan_config(cfg,np.random.default_rng(1))
