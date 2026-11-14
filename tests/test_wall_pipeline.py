import numpy as np
import pytest
from simscan.wall_pipeline import fit_walls, STAGE_ORDER, audit
from simscan.wall_runs import _union


def test_staged_fit_keeps_semantic_gap_and_exports_each_stage():
    target=np.zeros((80,140),bool);target[30:40,10:130]=True
    forbidden=np.zeros_like(target);forbidden[:,62:72]=True;target&=~forbidden
    barrier=np.zeros_like(target);barrier[30,10:130]=True;barrier[39,10:130]=True
    snapshots=[]
    def stage(name,result,info):
        snapshots.append((name,result['mask'].copy(),info))
    result=fit_walls(target,.02,semantic_probability=target.astype('float32'),
                     barrier=barrier,forbidden=forbidden,on_stage=stage)
    assert [s[0] for s in snapshots]==list(STAGE_ORDER)
    assert result['mask'].any()
    assert all(not (mask&forbidden).any() for _,mask,_ in snapshots)
    assert np.array_equal(_union(result['primitives'],target.shape),result['mask'])
    assert np.array_equal(snapshots[3][1],snapshots[4][1])
    assert np.array_equal(snapshots[7][1],snapshots[8][1])


def test_saved_area_fit_uses_three_distinct_structural_passes(monkeypatch):
    import simscan.wall_pipeline as job
    from simscan.decompose import decompose
    mask=np.zeros((60,100),bool);mask[20:28,10:90]=True
    initial=decompose(mask,.02)
    original=job.refine_structure;settings=[]
    def structural(*args,**kwargs):
        cfg=args[6];settings.append((cfg.structure_filter,cfg.structure_joint))
        return original(*args,**kwargs)
    monkeypatch.setattr(job,'refine_structure',structural)
    def unexpected(*args,**kwargs):raise AssertionError('Saved area fit was recomputed')
    monkeypatch.setattr(job,'decompose',unexpected)
    job.fit_walls(mask,.02,initial=initial,historical_replay=True)
    assert settings==[(False,False),(True,False),(True,True)]


def test_audit_rejects_forbidden_fill():
    m=np.ones((5,5),bool);f=np.zeros_like(m);f[2,2]=True
    with pytest.raises(ValueError,match='excluded_pixels_filled'):
        audit(m,m,np.zeros_like(m),f,.0025)


def test_reference_guard_restores_previously_matched_straight_wall():
    from simscan.decompose import decompose, Config
    from simscan.wall_pipeline import finish_walls
    target=np.zeros((80,140),bool);target[30:40,10:130]=True
    barrier=np.zeros_like(target);barrier[30,10:130]=True;barrier[39,10:130]=True
    broken=target.copy();broken[:,60:80]=False
    cfg=Config(structure_refine=False,merge_wall_runs=False,simplify_wall_runs=False)
    area=decompose(target,.02,config=cfg);staged=decompose(broken,.02,config=cfg)
    result=finish_walls(staged,area,target,barrier,.02,probability=target.astype('float32'))
    assert result['regression_guard']['selected']=='combined_reference'
    assert result['regression_guard']['lost_reference_samples']>0
    assert result['mask'][:,60:80].any()


def test_two_scan_faces_recover_width_without_filling_excluded_opening():
    from simscan.decompose import rectangle
    from simscan.wall_face_recovery import recover_scan_faces
    from simscan.wall_runs import _union
    target=np.zeros((180,420),bool);target[75:118,15:405]=True
    barrier=np.zeros_like(target);barrier[80,15:405]=True;barrier[112,15:405]=True
    # A thin accepted rectangle follows the lower face but misses the upper one.
    c=rectangle([210,103],380,18,0);c['id']=0
    initial=dict(primitives=[c],mask=_union([c],target.shape))
    forbidden=np.zeros_like(target);forbidden[:,408:]=True
    result=recover_scan_faces(initial,target,barrier,.0025,target.astype('float32'),forbidden)
    info=result['scan_face_recovery']
    assert info['edits'] and info['after']['coverage']>info['before']['coverage']
    assert not (result['mask']&forbidden).any()
    assert np.array_equal(_union(result['primitives'],target.shape),result['mask'])
