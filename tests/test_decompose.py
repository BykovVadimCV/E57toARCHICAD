import numpy as np
from simscan.decompose import Config, decompose, rectangle, arc, rasterize, dense_evidence


def test_width_step_and_recess():
    mask=np.zeros((90,150),bool);mask[20:28,10:75]=1;mask[16:32,75:140]=1
    result=decompose(mask,config=Config(spill_weight=3.,spill_limit=.12,proposal_fill=.9,coverage_stop=1.,overlap_weight=0))
    assert result['metrics']['iou']>.95
    widths=[p['width'] for p in result['primitives'] if p['kind']=='rectangle']
    assert max(widths)-min(widths)>5
    assert not result['mask'][10:15].any()


def test_curved_band_stays_curved():
    mask=rasterize(arc([100,100],65,8,-2.8,1.9),(160,180))
    result=decompose(mask,config=Config(spill_weight=3.,spill_limit=.12,proposal_fill=.9,coverage_stop=1.,overlap_weight=0))
    assert any(p['kind']=='arc' for p in result['primitives'])
    assert result['metrics']['iou']>.85
    assert not result['mask'][90:110,90:110].any()


def test_sparse_points_exactly_neutral():
    mask=rasterize(rectangle([65,40],100,8,0),(80,130))
    a=decompose(mask);b=decompose(mask,density=np.full(mask.shape,.3))
    assert a['primitives']==b['primitives']
    assert b['density']['active_pixels']==0


def test_dense_clutter_not_wall_and_count_floor():
    p=np.zeros((40,60),np.float32);p[15:22,5:55]=1
    d=np.ones_like(p);dense,info=dense_evidence(d,p,.02,Config())
    assert not dense[p==0].any()
    assert info['count_threshold_2cm']>=200
    d[:]=np.log1p(100)/np.log1p(4000)
    assert not dense_evidence(d,p,.02,Config())[0].any()


def test_empty_and_real_diagonal():
    assert not decompose(np.zeros((20,20),bool))['primitives']
    mask=rasterize(rectangle([60,60],90,8,.18),(120,120))
    r=decompose(mask)
    assert r['metrics']['iou']>.85
    assert any(abs(p.get('angle',0))>.1 for p in r['primitives'])


def test_density_cannot_create_wall_in_empty_mask():
    mask=np.zeros((40,60),bool)
    result=decompose(mask,density=np.ones(mask.shape,np.float32))
    assert not result['primitives']
    assert not result['mask'].any()


def test_topology_and_union_match_saved_footprints():
    mask=np.zeros((80,100),bool);mask[10:18,10:90]=1;mask[10:65,45:55]=1
    result=decompose(mask)
    union=np.zeros_like(mask)
    for item in result['primitives']:union|=rasterize(item,mask.shape)
    assert np.array_equal(union,result['mask'])
    assert result['contacts']
    assert all(c['a']<len(result['primitives']) and c['b']<len(result['primitives']) for c in result['contacts'])


def test_polygon_hole_preserved():
    p=dict(kind='junction',polygon=[[2,2],[18,2],[18,18],[2,18]],holes=[[[7,7],[13,7],[13,13],[7,13]]])
    mask=rasterize(p,(22,22))
    assert mask[4,4] and not mask[10,10]


def test_dense_evidence_is_bounded_face_bonus():
    mask=rasterize(rectangle([65,40],100,8,0),(80,130))
    result=decompose(mask,density=np.ones(mask.shape,np.float32))
    fits=[c['fit'] for c in result['primitives'] if 'fit' in c]
    assert any(c['cloud_bonus']>0 for c in fits)
    assert all(c['cloud_bonus']<=Config().density_weight*c['gain']+1e-6 for c in fits)


def test_relaxed_defaults_stop_at_coverage():
    mask=np.zeros((70,140),bool);mask[15:25,10:100]=1;mask[45:49,110:130]=1
    result=decompose(mask,config=Config(coverage_stop=.9))
    assert result['metrics']['recall']>=.9
    assert not result['mask'][45:49,110:130].any()
    assert Config().spill_weight==.75 and Config().spill_limit==.35


def test_native_scoring_with_coarser_proposals_keeps_world_alignment():
    mask=np.zeros((151,317),bool);mask[43:67,23:291]=1
    result=decompose(mask,pixel_m=.0025,config=Config(coverage_stop=.98))
    assert result['metrics']['recall']>=.98
    assert result['metrics']['iou']>.94
    union=np.zeros_like(mask)
    for p in result['primitives']:union|=rasterize(p,mask.shape)
    assert np.array_equal(union,result['mask'])


def test_overlap_penalty_reduces_redundant_rectangles_with_coverage():
    # A width step otherwise encourages a long thin strip plus overlapping bands.
    from simscan.decompose import rectangle_overlap_metrics
    mask=np.zeros((100,180),bool);mask[30:40,10:170]=1;mask[25:45,70:150]=1
    old=decompose(mask,config=Config(coverage_stop=.98,overlap_weight=0))
    new=decompose(mask,config=Config(coverage_stop=.98,overlap_weight=2))
    a=rectangle_overlap_metrics(old['primitives'],mask.shape,.02)
    b=rectangle_overlap_metrics(new['primitives'],mask.shape,.02)
    assert b['redundant_area_m2']<a['redundant_area_m2']
    assert new['metrics']['recall']>=.95


def test_corner_overlap_still_allows_connected_footprints():
    mask=np.zeros((100,100),bool);mask[20:30,10:90]=1;mask[20:90,80:90]=1
    result=decompose(mask,config=Config(coverage_stop=.98))
    assert result['metrics']['recall']>=.95
    assert result['contacts']
