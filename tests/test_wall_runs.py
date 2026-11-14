import numpy as np
from simscan.decompose import Config,rectangle,rasterize,_topology
from simscan.wall_runs import refine_structure,filter_detached,extract_runs


def seed(primitives,shape):
    m=np.zeros(shape,bool)
    for p in primitives:m|=rasterize(p,shape)
    return dict(primitives=primitives,mask=m,config={})


def test_recovers_existing_supported_gap_but_not_a_semantic_aperture():
    shape=(130,380);mask=np.zeros(shape,bool);mask[50:62,20:360]=1
    scan=np.zeros(shape,bool);scan[50,20:360]=1
    pieces=[rectangle([90,56],140,12,0),rectangle([280,56],160,12,0)]
    initial=seed(pieces,shape);cfg=Config(structure_joint=False,structure_filter=False)
    r=refine_structure(initial,mask,scan,.005,mask.astype(float),config=cfg)
    assert r['mask'][50:62,165:195].any(0).all()
    assert r['structure']['after']['matched_length_m']>r['structure']['before']['matched_length_m']
    forbidden=np.zeros(shape,bool);forbidden[:,170:190]=1;target=mask&~forbidden
    split=refine_structure(initial,target,scan,.005,target.astype(float),forbidden,cfg)
    assert not (split['mask']&forbidden).any()


def test_detached_filter_needs_original_weak_confidence_and_real_scan():
    mask=np.zeros((100,160),bool);mask[20:30,10:120]=1;mask[65:75,130:150]=1
    p=mask.astype(float);p[65:75,130:150]=.4;scan=np.zeros_like(mask);scan[20,10:120]=1
    clean,removed,records=filter_detached(mask,p,scan,.01)
    assert removed[65:75,130:150].all() and clean[20:30,10:120].all()
    assert len(records)==1
    assert not filter_detached(mask,None,scan,.01)[1].any()
    assert not filter_detached(mask,p,np.zeros_like(scan),.01)[1].any()
    p[65:75,130:150]=.95
    assert not filter_detached(mask,p,scan,.01)[1].any()


def test_attached_occluded_wall_body_is_not_removed_for_missing_returns():
    mask=np.zeros((100,160),bool);mask[20:30,10:120]=1;mask[20:80,110:120]=1
    p=np.full(mask.shape,.4);scan=np.zeros_like(mask);scan[20,10:90]=1
    clean,removed,_=filter_detached(mask,p,scan,.01)
    assert np.array_equal(clean,mask) and not removed.any()


def test_scan_runs_preserve_real_oblique_faces():
    shape=(200,300);angle=.17
    mask=rasterize(rectangle([150,100],230,14,angle),shape)
    y,x=np.mgrid[:shape[0],:shape[1]];q=-(x-150)*np.sin(angle)+(y-100)*np.cos(angle)
    t=(x-150)*np.cos(angle)+(y-100)*np.sin(angle)
    barrier=(np.abs(q+7)<.7)&(np.abs(t)<110)
    runs=extract_runs(mask,barrier,.005,Config())
    assert runs
    assert min(abs(np.arctan2(r['u'][1],r['u'][0])-angle) for r in runs)<.01


def test_shared_run_fit_keeps_exact_geometry_mask_and_width_step():
    shape=(130,400);target=np.zeros(shape,bool);target[50:62,20:200]=1;target[50:72,200:380]=1
    scan=np.zeros(shape,bool);scan[50,20:380]=1
    pieces=[rectangle([70,56],100,12,.002),rectangle([160,56],80,12,-.002),rectangle([240,61],80,22,.002),rectangle([330,61],100,22,-.002)]
    r=refine_structure(seed(pieces,shape),target,scan,.005,target.astype(float),config=Config())
    union=np.zeros(shape,bool)
    for c in r['primitives']:union|=rasterize(c,shape)
    assert np.array_equal(union,r['mask'])
    assert r['mask'][55,30:370].all()
    assert not r['mask'][65,30:160].any()
    assert r['mask'][65,230:370].all()


def test_weak_diagonal_connector_cannot_protect_itself_with_mask_connectivity():
    shape=(220,300);vertical=rectangle([45,105],150,10,np.pi/2);horizontal=rectangle([150,165],220,10,0)
    diagonal=rectangle([121,102],190,18,.68);parts=[vertical,horizontal,diagonal]
    initial=seed(parts,shape);target=initial['mask'];p=np.full(shape,.4,dtype=np.float32)
    p[rasterize(vertical,shape)|rasterize(horizontal,shape)]=.95
    scan=np.zeros(shape,bool);scan[30:180,40]=1;scan[170,40:260]=1
    result=refine_structure(initial,target,scan,.005,p,config=Config(structure_joint=False))
    quarantined=[q for q in result['structure']['quarantine'] if q['reason']=='weak_diagonal_without_matching_scan_direction']
    assert quarantined
    assert result['structure']['after']['matched_length_m']>=result['structure']['before']['matched_length_m']
    assert result['mask'][100,45] and result['mask'][165,180]


def test_edits_refresh_world_footprints_from_authoritative_native_geometry():
    shape=(130,380);target=np.zeros(shape,bool);target[50:62,20:360]=1
    scan=np.zeros(shape,bool);scan[50,20:360]=1
    initial=seed([rectangle([90,56],140,12,0),rectangle([280,56],160,12,0)],shape)
    matrix=np.array([[.005,0.,100.],[0.,-.005,200.]])
    initial['pixel_to_world']=matrix.tolist();initial['world_regions']=[{'footprint':'stale'}]
    result=refine_structure(initial,target,scan,.005,target.astype(float),config=Config())
    for c,world in zip(result['primitives'],result['world_regions']):
        expected=np.c_[c['polygon'],np.ones(len(c['polygon']))]@matrix.T
        assert np.allclose(world['footprint'],expected)
        assert c['id']==world['id']


def test_curve_is_not_replaced_by_straight_wall_or_filled_in():
    from simscan.decompose import arc
    shape=(240,240);curved=arc([120,120],70,12,.2,2.8);initial=seed([curved],shape)
    y,x=np.mgrid[:240,:240];r=np.hypot(x-120,y-120);barrier=(np.abs(r-76)<.8)
    result=refine_structure(initial,initial['mask'],barrier,.005,initial['mask'].astype(float),config=Config())
    assert any(c['kind']=='arc' for c in result['primitives'])
    assert not result['mask'][110:130,110:130].any()


def test_weak_junction_protrusion_removed_but_two_observed_parent_joint_retained():
    shape=(220,300);h=rectangle([70,55],100,10,0)
    tail=dict(kind='junction',polygon=[[70,60],[85,60],[90,85]],holes=[])
    initial=seed([h,tail],shape);p=np.full(shape,.4,dtype=np.float32);p[rasterize(h,shape)]=.95
    scan=np.zeros(shape,bool);scan[50,20:120]=1
    r=refine_structure(initial,initial['mask'],scan,.005,p,config=Config(structure_joint=False))
    assert not r['mask'][75,85]
    assert any(q['reason']=='weak_junction_tail_without_two_observed_parents' for q in r['structure']['quarantine'])
    v=rectangle([130,115],110,10,np.pi/2)
    joint=dict(kind='junction',polygon=[[118,53],[136,53],[136,66],[118,66]],holes=[])
    initial=seed([h,v,joint],shape);p=np.full(shape,.4,dtype=np.float32)
    p[rasterize(h,shape)|rasterize(v,shape)]=.95;scan[60:170,135]=1
    r=refine_structure(initial,initial['mask'],scan,.005,p,config=Config(structure_joint=False))
    assert r['mask'][58,123]


def test_float32_scan_reference_and_export_are_json_serializable():
    import json
    from simscan.decompose import _axis_reference
    c=rectangle([100,100],120,12,.02)
    ref=_axis_reference(c,np.array([[50,94,150,94]],np.float32),[],.01,Config())
    assert ref is not None
    json.dumps(ref)


def test_original_confidence_does_not_reweight_the_area_fit():
    from simscan.decompose import decompose
    shape=(100,220);mask=np.zeros(shape,bool);mask[30:40,10:180]=1;mask[70:80,195:215]=1
    probability=np.full(shape,.1,np.float32);barrier=np.zeros(shape,bool);barrier[30,10:180]=1
    result=decompose(mask,.005,barrier=barrier,semantic_probability=probability,
                     config=Config(axis_weight=0,structure_joint=False))
    assert result['mask'][32:38,20:170].all()
    assert not result['mask'][70:80,195:215].any()
    # Geometry fitting remains on the established binary target scale, while
    # original 0.1 confidence correctly quarantines the unsupported island.
    assert max(c.get('fit',{}).get('gain',0) for c in result['primitives'])>1200
    assert result['structure']['quarantine']
