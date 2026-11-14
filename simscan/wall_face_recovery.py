"""Recover refined-mask faces using two independently observed parallel scan lines."""
from copy import deepcopy
import numpy as np
from .wall_runs import (extract_runs,_profile,_point_distances,_intervals,_band,
    _accept_edit,_commit,_union,face_metrics)
from .wall_simplify import _fit_two_faces
from .decompose import Config,_bounds,_local_mask,_topology,mask_metrics,rectangle_overlap_metrics


def recover_scan_faces(result,target,barrier,pixel_m=.0025,probability=None,forbidden=None,config=None):
    cfg=config or Config();target=np.asarray(target,bool);barrier=np.asarray(barrier,bool)
    if barrier.shape!=target.shape:raise ValueError('Recovery raster frames differ')
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense']})
    parts=deepcopy([p for c in result['primitives'] for p in (c['parts'] if c['kind']=='wall_run' else [c])])
    union=_union(parts,target.shape);runs=extract_runs(target,barrier,pixel_m,cfg,forbidden)
    profiles=[_profile(r,target,forbidden,pixel_m,cfg) for r in runs]
    before=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m)
    log=[];reject={};tried=set()
    for run,pr in zip(runs,profiles):
        distance=_point_distances(union,pr['points'],pixel_m)
        for a,b in _intervals(pr['valid']&(distance>cfg.run_face_tolerance_m)):
            if (b-a)*pr['step']*pixel_m<.1:continue
            side=int(np.median(pr['side'][a:b]));width=float(np.median(pr['width'][a:b]))
            if not side or width<cfg.min_width_m:continue
            # Follow the same continuous allowed profile; do not bridge apertures.
            aa,bb=next((v for v in _intervals(pr['valid']) if v[0]<=a and v[1]>=b),(a,b))
            band=_band(run,pr['t'][aa]-.01/pixel_m,pr['t'][bb-1]+.01/pixel_m,width,side,pixel_m)
            candidate,reason=_fit_two_faces([band],barrier,pixel_m,cfg)
            if candidate is None:reject[reason]=reject.get(reason,0)+1;continue
            token=tuple(np.round(np.asarray(candidate['polygon']).ravel(),2))
            if token in tried:continue
            tried.add(token)
            lo,hi=_bounds(candidate,target.shape);shape=tuple((hi-lo)[::-1]);body=_local_mask(candidate,lo,shape)
            remove=[]
            # Replace contained old tiles instead of stacking a new rectangle
            # over them. Protruding ends, branches and inferred geometry survive.
            for i,c in enumerate(parts):
                bounds=np.asarray(c['polygon'])
                if np.any(bounds.min(0)<lo) or np.any(bounds.max(0)>=hi):continue
                patch=_local_mask(c,lo,shape)
                if patch.any() and not (patch&~body).any():remove.append(i)
            edit=_accept_edit(parts,union,remove,[candidate],target,barrier,forbidden,probability,
                profiles,pixel_m,cfg,diagnostics=reject)
            if edit and edit[2].get('new_face_samples',0):
                parts=_commit(parts,union,remove,[candidate],edit,log,'recover_two_observed_faces')
    for i,c in enumerate(parts):c['id']=i
    out.update(primitives=parts,mask=union,contacts=_topology(parts,target.shape),
        metrics=mask_metrics(target,union),rectangle_overlap=rectangle_overlap_metrics(parts,target.shape,pixel_m),
        scan_face_recovery=dict(before=before,after=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m),edits=log,rejections=reject))
    # Geometry may have changed; callers regenerate world footprints from frame.
    out.pop('world_regions',None)
    return out
