import numpy as np
from simscan.wall_constraints import wall_only,dense_barriers
from simscan.decompose import decompose,Config


def test_openings_excluded_from_target_and_cannot_be_bridged():
    p=np.zeros((8,80,140),np.float32);p[0]=1
    p[:,30:40,10:130]=0;p[2,30:40,10:130]=1
    p[:,30:40,60:80]=0;p[3,30:40,60:80]=1
    w=wall_only(p)
    assert not w['mask'][:,60:80].any()
    d=decompose(w['mask'],forbidden=w['forbidden'],barrier=np.zeros(w['mask'].shape,bool))
    assert not (d['mask']&w['forbidden']).any()


def test_clear_spill_cheaper_but_dense_barrier_not_crossed():
    from scipy import ndimage as ndi
    m=np.zeros((90,130),bool);m[30:40,10:110]=1;m[30:37,25:95]=0
    uniform=decompose(m)
    clear=decompose(m,barrier=np.zeros_like(m))
    barrier=np.zeros_like(m);barrier[34,25:95]=1
    blocked=decompose(m,barrier=barrier)
    assert clear['metrics']['spill_pixels']>uniform['metrics']['spill_pixels']
    assert not (blocked['mask']&~m&barrier).any()


def test_barriers_do_not_depend_on_mask_and_ignore_sparse_or_nonvertical():
    x=np.zeros((8,20,20),np.float32);x[:3]=1;x[6]=1
    counts=np.zeros((80,80),np.uint32);counts[10:70,40:43]=10
    assert dense_barriers(counts,x).any()
    assert not dense_barriers(np.ones_like(counts),x).any()
    x[6]=0
    assert not dense_barriers(counts,x).any()


def test_windows_retained_doors_and_generic_openings_excluded():
    p=np.zeros((8,70,160),np.float32);p[0]=1
    p[:,25:37,10:150]=0;p[2,25:37,10:150]=1
    for c,lo,hi in [(3,30,45),(4,65,85),(5,110,125)]:
        p[:,25:37,lo:hi]=0;p[c,25:37,lo:hi]=1
    w=wall_only(p,include_windows=True)
    assert w['mask'][25:37,65:85].all()
    assert not w['forbidden'][25:37,65:85].any()
    assert w['probability'][25:37,65:85].min()==1
    assert not w['mask'][25:37,30:45].any()
    assert not w['mask'][25:37,110:125].any()


def test_dense_barrier_fit_can_adjust_coarse_angle_before_hard_rejection():
    from simscan.decompose import rectangle,rasterize
    import cv2
    shape=(220,900);center=np.array([450.,110.]);angle=.08
    mask=rasterize(rectangle(center,800,24,angle),shape)
    y,x=np.mgrid[:shape[0],:shape[1]];dx=x-center[0];dy=y-center[1]
    along=dx*np.cos(angle)+dy*np.sin(angle);across=-dx*np.sin(angle)+dy*np.cos(angle)
    barrier=(np.abs(along)<400)&(np.abs(across)>12)&(np.abs(across)<=13)
    result=decompose(mask,.0025,barrier=barrier,config=Config(coverage_stop=.95))
    assert result['metrics']['recall']>.90
    depth=cv2.distanceTransform(np.pad(result['mask'].astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]*.0025
    assert not (result['mask']&~mask&barrier&(depth>.005)).any()


def test_windows_excluded_by_default():
    p=np.zeros((8,60,120),np.float32);p[0]=1
    p[:,20:32,10:110]=0;p[2,20:32,10:110]=1
    p[:,20:32,40:80]=0;p[4,20:32,40:80]=1
    r=wall_only(p)
    assert r['forbidden'][20:32,40:80].all()
    assert not r['mask'][20:32,40:80].any()
    assert not r['probability'][20:32,40:80].any()


def test_scan_attraction_peaks_on_face_and_penalizes_overshoot():
    from simscan.decompose import _scan_face_terms,rectangle
    shape=(80,120);guide=np.zeros(shape,bool);guide[15:65,70]=1;claimed=np.zeros_like(guide)
    cfg=Config();lo=np.array([0,0])
    def terms(right):return _scan_face_terms(rectangle([(40+right)/2,40],right-40,60,0),guide,claimed,lo,.0025,cfg)
    aligned=terms(70);short=terms(66);overshot=terms(76)
    assert aligned[0]>short[0] and aligned[1]==0
    assert overshot[1]>aligned[1] and overshot[0]<aligned[0]
    claimed[guide]=1
    assert terms(70)[0]==0


def test_scan_attraction_arc_endpoint_wrap_is_symmetric():
    from simscan.decompose import _scan_face_terms,arc
    cfg=Config();c=arc([60,60],30,8,0,np.pi/2);claimed=np.zeros((120,120),bool)
    a=np.zeros_like(claimed);b=a.copy();a[59,90]=1;b[90,59]=1
    ra=_scan_face_terms(c,a,claimed,np.array([0,0]),.0025,cfg)
    rb=_scan_face_terms(c,b,claimed,np.array([0,0]),.0025,cfg)
    assert abs(ra[0]-rb[0])<1e-6
    assert ra[0]>0 and ra[1]==0


def test_local_axis_consensus_preserves_oblique_and_perpendicular_walls():
    from simscan.decompose import _axis_reference,rectangle
    cfg=Config();c=rectangle([100,100],120,12,np.radians(31))
    lines=np.array([[40,65,160,65+120*np.tan(np.radians(30))]])
    ref=_axis_reference(c,lines,[],.01,cfg)
    assert ref is not None and abs(np.degrees(ref['target'])-30)<1e-5
    assert ref['sources']==['scan']
    neighbour=rectangle([165,120],60,12,np.radians(120))
    assert _axis_reference(c,np.empty((0,4)),[neighbour],.01,cfg) is None
    neighbour['scan_verified']=True
    ref=_axis_reference(c,np.empty((0,4)),[neighbour],.01,cfg)
    assert ref is not None and abs(np.degrees(ref['target'])-30)<1e-5
    assert _axis_reference(rectangle([100,100],120,12,np.radians(20)),lines,[],.01,cfg) is None
    assert _axis_reference(c,lines+1000,[],.01,cfg) is None


def test_axis_adjustment_rejects_lost_contact_and_allows_connected_translation():
    from simscan.decompose import rectangle,_contact_guards,_keeps_contacts,_local_mask
    shape=(100,160);c=rectangle([80,50],60,10,0)
    neighbours=[rectangle([30,50],40,10,0),rectangle([130,50],40,10,0)]
    guards=_contact_guards(c,neighbours,shape)
    assert len(guards)==2
    sl=(slice(0,100),slice(0,160))
    def keeps(c):return _keeps_contacts((sl,_local_mask(c,np.array([0,0]),shape)),guards)
    assert keeps(c)
    assert keeps(rectangle([80,51],60,10,0))
    assert not keeps(rectangle([80,50],54,10,0))
    assert not keeps(rectangle([80,50],60,10,np.radians(25)))


def test_final_axis_guard_detects_split_but_allows_added_isolated_coverage():
    from simscan.decompose import _connectivity_breaks
    old=np.zeros((50,80),bool);old[20:25,10:70]=1
    added=old.copy();added[5:10,5:10]=1
    assert not _connectivity_breaks(old,added)
    split=old.copy();split[:,40]=0
    assert len(_connectivity_breaks(old,split))==1
    assert len(_connectivity_breaks(old,np.zeros_like(old)))==1


def test_final_axis_guard_returns_connected_reference_instead_of_filling_gap(monkeypatch):
    import simscan.decompose as module
    from dataclasses import asdict
    old=np.zeros((50,80),bool);old[20:25,10:70]=1
    split=old.copy();split[:,40]=0
    def fit(mask,pixel_m,probability,density,cfg,barrier,forbidden):
        return dict(mask=(split if cfg.axis_weight else old).copy(),config=asdict(cfg),
                    primitives=[dict(fit=dict(axis_reference={'target':0}))] if cfg.axis_weight else [])
    monkeypatch.setattr(module,'_decompose',fit)
    result=module.decompose(old)
    assert np.array_equal(result['mask'],old)
    assert not result['axis_regularization']['accepted']
    assert result['config']['axis_weight']==Config().axis_weight
