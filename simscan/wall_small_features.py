"""Recover short scan-bounded piers and T caps next to semantic apertures.

This additive pass retains existing long-wall fitting. It permits one attached
parent, unlike gap recovery, but requires short coherent scan faces on both sides
of a small wall band. Door leaves failing the full-height wall gate do not vote.
"""
from copy import deepcopy
from dataclasses import replace
import numpy as np
import cv2
from .decompose import Config,_bounds,_local_mask,mask_metrics,_topology
from .wall_runs import extract_runs,_profile,_band,_sample,_intervals
from .wall_footprint import mask_geometry,regions,primitive


def recover_small_features(result,target,barrier,pixel_m=.0025,forbidden=None,config=None):
    cfg=config or Config();mask=np.asarray(result['mask'],bool).copy();target=np.asarray(target,bool);barrier=np.asarray(barrier,bool)
    forbidden=np.zeros_like(mask) if forbidden is None else np.asarray(forbidden,bool)
    if target.shape!=mask.shape or barrier.shape!=mask.shape or forbidden.shape!=mask.shape:raise ValueError('Small-feature frames differ')
    short=replace(cfg,run_min_m=.05,run_max_width_m=min(cfg.max_width_m,.8))
    max_width=short.run_max_width_m
    runs=extract_runs(target,barrier,pixel_m,short,forbidden)
    edits=[];rejected={};considered=0
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    for run in runs:
        length=(run['t1']-run['t0'])*pixel_m
        if not .05<=length<=.70:continue
        pr=_profile(run,target,forbidden,pixel_m,short)
        for side in [-1,1]:
            good=pr['valid']&(pr['side']==side)&(pr['width']<max_width-.5*pixel_m)
            if good.sum()*pr['step']*pixel_m<.04:continue
            # A T stem broadens some profiles far beyond the cap itself. The
            # lower quartile seeds the cap width; a measured opposite face
            # must still confirm that width before any material is added.
            width=float(np.percentile(pr['width'][good],25))
            if not cfg.min_width_m<=width<=max_width:continue
            band=_band(run,run['t0'],run['t1'],width,side,pixel_m)
            lo,hi=_bounds(band,mask.shape);margin=int(np.ceil(.08/pixel_m))+3
            lo=np.maximum(0,lo-margin);hi=np.minimum(mask.shape[::-1],hi+margin);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));old=mask[sl]
            if not forbidden[sl].any():continue
            considered+=1
            yy,xx=np.nonzero(barrier[sl]);pts=np.c_[xx+lo[0],yy+lo[1]].astype(float)
            u=run['u'];v=run['normal'];t=pts@u;q=(pts@v-run['offset'])*side*pixel_m
            interior=(t>=run['t0'])&(t<=run['t1'])
            # The unseen back is not guessed from a nearby door leaf. Partial
            # opposite faces are valid at a T where the stem interrupts them.
            opposite=interior&(np.abs(q-width)<=.025)&(q>=.025)
            if opposite.sum()<max(8,int(.04/pixel_m)):reject('missing_opposite_face');continue
            q2=q[opposite];back=float(np.median(q2))
            coherent=opposite&(np.abs(q-back)<=.0075)
            ts=t[coherent];nb=max(1,int(np.ceil(length/.01)))
            occupied=np.unique(np.clip(((ts-run['t0'])*pixel_m/.01).astype(int),0,nb-1))
            if len(occupied)<max(4,int(np.ceil(nb*.30))):reject('fragmented_opposite_face');continue
            # A T stem contributes perpendicular returns in this band. They
            # must not veto the coherent opposite-face portions on either side.
            front=interior&(np.abs(q)<=.015)
            if front.sum()<max(8,int(.04/pixel_m)):reject('missing_front_face');continue
            qfront=float(np.clip(np.percentile(q[front],90),-.0075,.0075))
            qback=float(np.percentile(q[coherent],10))
            if not cfg.min_width_m*.75<=qback-qfront<=max_width:reject('width');continue
            yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];along=xx*u[0]+yy*u[1];normal=(xx*v[0]+yy*v[1]-run['offset'])*side*pixel_m
            patch=(along>=run['t0'])&(along<=run['t1'])&(normal>=qfront)&(normal<=qback)
            patch&=~forbidden[sl]
            if not patch.any():continue
            if (patch&target[sl]).sum()/patch.sum()<.75:reject('target_support');continue
            added=patch&~old
            if added.sum()*pixel_m**2<.0005:continue
            if added.sum()/patch.sum()<.20:reject('minor_edge_residual');continue
            if added.sum()*pixel_m**2>.08:reject('area');continue
            touch=cv2.dilate(patch.astype('uint8'),np.ones((3,3),'uint8'))>0
            if not (touch&old).any():reject('detached');continue
            distance=cv2.distanceTransform((~old).astype('uint8'),cv2.DIST_L2,5)*pixel_m
            if distance[added].max()>.20:reject('extension');continue
            new=old|patch
            depth=cv2.distanceTransform(np.pad(new.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
            if (new&~target[sl]&barrier[sl]&(depth>cfg.barrier_contact_m)).any():reject('dense_crossing');continue
            # All full-height dense surfaces remain exposed, including faces
            # inside the model mask. Extra coverage cannot swallow a jamb edge.
            old_depth=cv2.distanceTransform(np.pad(old.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
            buried=barrier[sl]&old&(old_depth<=.015)&(depth>.015)
            if buried.any():reject('buried_existing_face');continue
            if (added&barrier[sl]&(depth>.015)).any():reject('buried_new_face');continue
            mask[sl]=new
            edits.append(dict(operation='recover_scan_bounded_pier_or_T_cap',length_m=float(length),width_m=qback-qfront,
                added_pixels=int(added.sum()),restored_fraction=float(added.sum()/patch.sum()),opposite_face_coverage=float(len(occupied)/nb),single_parent_allowed=True,
                line=dict(u=u.tolist(),normal=v.tolist(),offset=float(run['offset']),t0=float(run['t0']),t1=float(run['t1']))))
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense','world_regions','contacts']})
    final=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(mask)))]
    out.update(mask=mask,primitives=final,metrics=mask_metrics(target,mask),contacts=_topology(final,mask.shape),
        small_feature_recovery=dict(edits=edits,rejections=rejected,considered=considered,min_face_length_m=.05,max_face_length_m=.70,
            max_added_area_m2=.08,max_extension_m=.20,max_width_m=max_width,min_missing_body_fraction=.20,policy='additive attached short wall bands next to apertures; two dense full-height faces; openings unchanged'))
    return out
