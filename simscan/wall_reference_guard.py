"""Optional final native-fit guard, restricted to identical target and evidence."""
import hashlib
from copy import deepcopy
import numpy as np
from .wall_runs import extract_runs,_profile,_point_distances,_union


def barrier_sha256(barrier):
    a=np.asarray(barrier,bool)
    return hashlib.sha256(np.packbits(a).tobytes()).hexdigest()


def select_native_fit(candidate,reference,target,barrier,forbidden,pixel_m,cfg):
    if reference is None:
        candidate['native_reference_guard']=dict(enabled=False,reason='no matching accepted native fit supplied')
        return candidate
    if not (reference['pixel_m']==pixel_m and np.array_equal(reference['target'],target) and
            np.array_equal(reference['forbidden'],forbidden) and reference['barrier_sha256']==barrier_sha256(barrier)):
        raise ValueError('Accepted reference does not match target, exclusions and scan evidence')
    old=reference['result'];oldmask=np.asarray(old['mask'],bool)
    if oldmask.shape!=target.shape or (oldmask&forbidden).any() or not np.array_equal(_union(old['primitives'],target.shape),oldmask):
        raise ValueError('Accepted native reference geometry/frame is invalid')
    lost=gained=0
    for run in extract_runs(target,barrier,pixel_m,cfg,forbidden):
        pr=_profile(run,target,forbidden,pixel_m,cfg)
        a=_point_distances(oldmask,pr['points'],pixel_m)<=cfg.run_face_tolerance_m
        b=_point_distances(candidate['mask'],pr['points'],pixel_m)<=cfg.run_face_tolerance_m
        lost+=int((pr['valid']&a&~b).sum());gained+=int((pr['valid']&~a&b).sum())
    out=deepcopy(old) if lost else candidate
    out['native_reference_guard']=dict(enabled=True,selected='accepted_native' if lost else 'fresh_fit',
        lost_candidate_face_samples=lost,gained_candidate_face_samples=gained,
        scope='identical target, semantic exclusions, native dense barrier and pixel scale',reference=reference.get('source','supplied accepted fit'))
    return out
