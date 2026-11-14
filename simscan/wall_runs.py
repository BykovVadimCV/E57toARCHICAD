"""Scan-supported wall faces, structural edits and final-union diagnostics.

No free-space inference from absent points. All coordinates are native pixel
centres. Uncertain detached predictions are quarantined separately from geometry.
"""
from __future__ import annotations
import copy
import cv2
import numpy as np
from scipy.ndimage import median_filter


def _intervals(bits):
    e=np.diff(np.pad(np.asarray(bits,np.int8),(1,1)))
    return list(zip(np.where(e==1)[0],np.where(e==-1)[0]))


def filter_detached(mask,probability,barrier,pixel_m,threshold=.75):
    """Only weak detached components without reliable boundary evidence are removed.

    With no reliable scan or original probabilities this is deliberately neutral.
    Attached occluded material and strongly predicted isolated columns survive.
    """
    mask=np.asarray(mask,bool);removed=np.zeros_like(mask);records=[]
    if probability is None or barrier is None or not np.any(barrier):return mask.copy(),removed,records
    p=np.asarray(probability,np.float32);barrier=np.asarray(barrier,bool)
    if p.shape!=mask.shape or barrier.shape!=mask.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Evidence must match the mask and probabilities must be finite in [0,1]')
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    radius=max(1,int(np.ceil(.04/pixel_m)));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,)*2)
    for i in range(1,n):
        x,y,w,h,area=stats[i];lo=np.maximum(0,[x-radius,y-radius]);hi=np.minimum(mask.shape[::-1],[x+w+radius,y+h+radius])
        sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));component=labels[sl]==i
        confidence=float(p[sl][component].mean())
        if confidence>=threshold:continue
        edge=component&~cv2.erode(component.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
        near=cv2.dilate(edge.astype('uint8'),kernel)>0
        support=int((near&barrier[sl]).sum())
        if support>=max(3,int(round(.06/pixel_m))):continue
        removed[sl]|=component
        records.append(dict(component=i,area_m2=float(area*pixel_m**2),mean_probability=confidence,
                            support_pixels=support,reason='weak_detached_without_reliable_face'))
    return mask&~removed,removed,records


def extract_runs(target,barrier,pixel_m,cfg,forbidden=None):
    """Robust native TLS faces from dense evidence; neither mask edges nor neighbors vote."""
    from .decompose import _scan_axis_lines
    from dataclasses import replace
    target=np.asarray(target,bool);barrier=np.asarray(barrier,bool)
    inside=cv2.distanceTransform(target.astype('uint8'),cv2.DIST_L2,5)*pixel_m
    outside=cv2.distanceTransform((~target).astype('uint8'),cv2.DIST_L2,5)*pixel_m
    eligible=barrier&(np.where(target,inside,outside)<=.06)
    if forbidden is not None:eligible&=~forbidden
    lines=_scan_axis_lines(eligible,pixel_m,replace(cfg,axis_weight=1.,axis_scan_min_length_m=cfg.run_min_m))
    runs=[];pad=int(np.ceil(.035/pixel_m));step=max(1,int(round(.01/pixel_m)))
    for line in lines:
        a=line[:2];b=line[2:];u=(b-a)/np.linalg.norm(b-a);v=np.array([-u[1],u[0]])
        lo=np.maximum(0,np.floor(np.minimum(a,b)-pad).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(np.maximum(a,b)+pad+1).astype(int))
        yy,xx=np.nonzero(eligible[lo[1]:hi[1],lo[0]:hi[0]]);pts=np.c_[xx+lo[0],yy+lo[1]].astype(float)
        if len(pts)<max(12,cfg.run_min_m/pixel_m/2):continue
        t=(pts-a)@u;q=(pts-a)@v
        pts=pts[(np.abs(q)*pixel_m<=.025)&(t>=-pad)&(t<=np.linalg.norm(b-a)+pad)]
        if len(pts)<12:continue
        fit=cv2.fitLine(pts.astype('float32'),cv2.DIST_HUBER,0,.01,.001).ravel()
        u=fit[:2].astype(float);u/=np.linalg.norm(u)
        if u[0]<0 or (abs(u[0])<1e-8 and u[1]<0):u=-u
        v=np.array([-u[1],u[0]]);q=pts@v;offset=float(np.median(q));scatter=float(np.percentile(np.abs(q-offset),90)*pixel_m)
        pts=pts[np.abs(q-offset)*pixel_m<=max(.006,min(.015,scatter))]
        if len(pts)<12:continue
        t=pts@u;t0,t1=np.percentile(t,[1,99]);length=(t1-t0)*pixel_m
        if length<cfg.run_min_m:continue
        # A long bbox with only two point clusters is not a continuous observed face.
        bins=np.floor((t-t0)/step).astype(int);nb=max(1,int(np.ceil((t1-t0)/step)))
        support=np.bincount(np.clip(bins,0,nb-1),minlength=nb)>0
        # Sensor-scale sampling gaps (at most 2cm) may connect; apertures are split later.
        for i,j in _intervals(~support):
            if i and j<nb and (j-i)*step*pixel_m<=.02:support[i:j]=True
        for i,j in _intervals(support):
            if (j-i)*step*pixel_m<cfg.run_min_m:continue
            start=t0+i*step;end=min(t1,t0+j*step)
            run=dict(u=u,normal=v,offset=offset,t0=float(start),t1=float(end),scatter_m=scatter)
            duplicate=False
            for old in runs:
                dot=float(np.clip(u@old['u'],-1,1))
                if np.arccos(abs(dot))>np.radians(1.5):continue
                center=u*((start+end)/2)+v*offset
                if abs(center@old['normal']-old['offset'])*pixel_m>.012:continue
                ends=np.array([u*start+v*offset,u*end+v*offset])@old['u']
                overlap=min(max(ends),old['t1'])-max(min(ends),old['t0'])
                if overlap>=.75*min(end-start,old['t1']-old['t0']):duplicate=True;break
            if not duplicate:runs.append(run)
    # Long faces first; scan sampling and input order cannot privilege tiny fragments.
    return sorted(runs,key=lambda r:-(r['t1']-r['t0']))


def _sample(array,points,default=0):
    xy=np.rint(points).astype(int);valid=(xy[:,0]>=0)&(xy[:,1]>=0)&(xy[:,0]<array.shape[1])&(xy[:,1]<array.shape[0])
    out=np.full(len(points),default,dtype=array.dtype);out[valid]=array[xy[valid,1],xy[valid,0]]
    return out


def _profile(run,target,forbidden,pixel_m,cfg):
    step=max(1,int(round(.01/pixel_m)));ts=np.arange(run['t0'],run['t1']+step/2,step)
    points=ts[:,None]*run['u']+run['offset']*run['normal']
    maxw=int(np.ceil(cfg.run_max_width_m/pixel_m));qs=np.arange(-maxw,maxw+1)
    xy=points[:,None,:]+qs[None,:,None]*run['normal']
    values=_sample(target,xy.reshape(-1,2)).reshape(len(ts),len(qs))
    veto=np.zeros_like(values) if forbidden is None else _sample(forbidden,xy.reshape(-1,2)).reshape(values.shape)
    widths=np.zeros(len(ts));sides=np.zeros(len(ts));valid=np.zeros(len(ts),bool)
    for i,row in enumerate(values):
        intervals=_intervals(row)
        if not intervals:continue
        a,b=min(intervals,key=lambda ab:max(ab[0]-maxw,maxw-ab[1]+1,0))
        if max(a-maxw,maxw-b+1,0)*pixel_m>.025:continue
        left=maxw-a;right=b-1-maxw;side=1 if right>=left else -1
        width=max(left,right)*pixel_m
        if width<cfg.min_width_m*.7:continue
        width=float(np.clip(width,cfg.min_width_m,cfg.run_max_width_m));extent=int(round(width/pixel_m))
        indices=np.arange(maxw,maxw+side*(extent+1),side);indices=np.clip(indices,0,len(qs)-1)
        if veto[i,indices].any() or values[i,indices].mean()<.65:continue
        valid[i]=True;widths[i]=width;sides[i]=side
    # Persistent changes of width remain separate; pixel-scale tails do not.
    smooth=median_filter(widths,size=max(3,int(round(.15/(step*pixel_m)))|1),mode='nearest')
    pieces=[]
    for a,b in _intervals(valid):
        start=a
        for i in range(a+1,b):
            if sides[i]!=sides[start] or abs(smooth[i]-smooth[start])>.025:
                if (i-start)*step*pixel_m>=.15:pieces.append((start,i));start=i
        if b>start:pieces.append((start,b))
    return dict(points=points,t=ts,valid=valid,width=widths,side=sides,step=step,pieces=pieces)


def _band(run,t0,t1,width,side,pixel_m):
    from .decompose import rectangle
    center=run['u']*((t0+t1)/2)+run['normal']*(run['offset']+side*width/pixel_m/2)
    c=rectangle(center,max(0,t1-t0),width/pixel_m,float(np.arctan2(run['u'][1],run['u'][0])))
    c['scan_verified']=True;c['face_fit']=dict(offset_pixels=run['offset'],side=int(side),width_m=float(width),
                                             observed_face=True,opposite_face='mask_inferred',scatter_m=run['scatter_m'])
    return c


def _verified_long_face(c,runs,pixel_m,tolerance=.015):
    """A nearby perpendicular edge is not evidence for this rectangle's long face."""
    if c['kind']!='rectangle':return c['kind']=='arc'
    u=np.array([np.cos(c['angle']),np.sin(c['angle'])]);v=np.array([-u[1],u[0]]);center=np.array(c['center'])
    for run in runs:
        dot=float(u@run['u'])
        if np.arccos(min(1.,abs(dot)))>np.radians(3):continue
        for side in [-1,1]:
            face=center+side*v*c['width']/2
            span=np.sort(np.array([-c['length']/2,c['length']/2])*dot+face@run['u'])
            start=max(span[0],run['t0']);end=min(span[1],run['t1'])
            if (end-start)*pixel_m<min(.15,c['length']*pixel_m*.7):continue
            s=(np.array([start,end])-face@run['u'])/dot
            points=face+s[:,None]*u
            if np.max(np.abs(points@run['normal']-run['offset']))*pixel_m<=tolerance:return True
    return False


def _union(primitives,shape):
    from .decompose import _bounds,_local_mask
    out=np.zeros(shape,bool)
    for c in primitives:
        lo,hi=_bounds(c,shape)
        if np.any(hi<=lo):continue
        out[lo[1]:hi[1],lo[0]:hi[0]]|=_local_mask(c,lo,tuple((hi-lo)[::-1]))
    return out


def _quarantine_detached_geometry(primitives,shape,p,barrier,pixel_m,cfg):
    from .decompose import _bounds,_local_mask
    union=_union(primitives,shape)
    _,weak,_=filter_detached(union,p,barrier,pixel_m,cfg.detached_probability)
    keep=[];records=[]
    for c in primitives:
        lo,hi=_bounds(c,shape);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));patch=_local_mask(c,lo,tuple((hi-lo)[::-1]))
        if patch.any() and np.all(weak[sl][patch]):records.append(dict(stage='vector',reason='weak_detached_without_reliable_face',primitive=c))
        else:keep.append(c)
    return keep,records


def _boundary_distance(mask,pixel_m):
    edge=mask&~cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8'),borderType=cv2.BORDER_CONSTANT,borderValue=0).astype(bool)
    return cv2.distanceTransform((~edge).astype('uint8'),cv2.DIST_L2,5)*pixel_m


def _point_distances(mask,points,pixel_m):
    margin=int(np.ceil(.08/pixel_m))
    lo=np.maximum(0,np.floor(points.min(0)-margin).astype(int));hi=np.minimum(mask.shape[::-1],np.ceil(points.max(0)+margin+1).astype(int))
    local=mask[lo[1]:hi[1],lo[0]:hi[0]]
    return _sample(_boundary_distance(local,pixel_m),points-lo,default=np.float32(1e3))


def face_metrics(mask,profiles,pixel_m,tolerance=.015):
    distance=_boundary_distance(mask,pixel_m);total=matched=longest=error=0.;records=[]
    for profile in profiles:
        valid=profile['valid'];d=_sample(distance,profile['points'],default=np.float32(1e3));step=profile['step']*pixel_m
        missing=valid&(d>tolerance)
        gap=max([(b-a)*step for a,b in _intervals(missing)],default=0.)
        length=float(valid.sum()*step);hit=float((valid&~missing).sum()*step)
        total+=length;matched+=hit;longest=max(longest,gap);error+=float(np.minimum(d[valid],.06).sum()*step)
        records.append(dict(length_m=length,matched_m=hit,longest_gap_m=gap))
    return dict(observed_length_m=total,matched_length_m=matched,coverage=matched/max(total,1e-9),
                longest_unexplained_m=longest,clipped_distance_integral_m2=error,runs=records)


def _accept_edit(primitives,union,remove,add,target,barrier,forbidden,p,profiles,pixel_m,cfg,allow_prune=False,diagnostics=None,coherence_gain=0.,target_loss_fraction=.01):
    """Score the actual union, preserving contacts, observed faces and aperture vetoes."""
    from .decompose import _local_mask,_connectivity_breaks
    def reject(reason):
        if diagnostics is not None:diagnostics[reason]=diagnostics.get(reason,0)+1
        return None
    changed=[primitives[i] for i in remove]+add
    bounds=np.vstack([c['polygon'] for c in changed]);margin=int(np.ceil(.07/pixel_m))+2
    lo=np.maximum(0,np.floor(bounds.min(0)-margin).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(bounds.max(0)+margin+1).astype(int))
    if np.any(bounds.min(0)<0) or np.any(bounds.max(0)>np.array(target.shape[::-1])-1):return reject('frame')
    sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));shape=tuple((hi-lo)[::-1]);old=union[sl];new=np.zeros(shape,bool)
    for i,c in enumerate(primitives):
        if i in remove:continue
        b=np.array(c['polygon'])
        if np.any(b.max(0)<lo) or np.any(b.min(0)>hi):continue
        new|=_local_mask(c,lo,shape)
    for c in add:new|=_local_mask(c,lo,shape)
    if np.array_equal(old,new):
        return (sl,new,dict(objective_gain_m2=0.,representation_only=True)) if len(add)<len(remove) else None
    if forbidden is not None and (new&forbidden[sl]).any():return reject('aperture')
    # Preserve every existing local wall connection. Deliberate whole-component
    # quarantine occurs outside this edit path and is recorded separately.
    breaks=_connectivity_breaks(old,new)
    if breaks and not allow_prune:return reject('connection')
    target_local=target[sl];reliable=barrier[sl]
    old_depth=cv2.distanceTransform(np.pad(old.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
    new_depth=cv2.distanceTransform(np.pad(new.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
    forbidden_cross=(new&~target_local&reliable&(new_depth>cfg.barrier_contact_m))
    if forbidden_cross.any():return reject('outside_dense_barrier')
    # Native dense points represent a noisy face, not a volume to fill. Do not
    # newly bury previously exposed reliable points, including inside the mask.
    newly_buried=reliable&old&(old_depth<=cfg.run_face_tolerance_m)&(new_depth>cfg.run_face_tolerance_m)
    if newly_buried.sum()>max(2,.001*reliable.sum()):return reject('buried_raw_points')
    added=new&~old;lost=old&~new
    # The spill allowance describes a wall footprint, not only the newly added
    # edge pixels. A 2.5mm face correction can otherwise look like 100% spill.
    old_spill=float((old&~target_local).sum())/max(1,old.sum())
    if (new&~target_local).sum()>max(cfg.clear_spill_limit,old_spill)*max(1,new.sum()):return reject('spill')
    if added.any():
        distance=cv2.distanceTransform((~target_local).astype('uint8'),cv2.DIST_L2,5)*pixel_m
        if (added&~target_local&(distance>cfg.max_clear_extension_m)).any():return reject('extension')
    if (lost&target_local).sum()*pixel_m**2>max(.001,target_loss_fraction*(old&target_local).sum()*pixel_m**2) and not allow_prune:return reject('target_loss')
    before=_boundary_distance(old,pixel_m);after=_boundary_distance(new,pixel_m)
    face_gain=0.;new_matches=0
    for profile in profiles:
        pts=profile['points'];inside=((pts>=lo)&(pts<hi)).all(1)&profile['valid']
        if not inside.any():continue
        local=pts[inside]-lo;bd=_sample(before,local,default=np.float32(1e3));ad=_sample(after,local,default=np.float32(1e3))
        bm=bd<=cfg.run_face_tolerance_m;am=ad<=cfg.run_face_tolerance_m
        if (bm&~am).any():return reject('matched_face_loss')
        step=profile['step']*pixel_m;new_matches+=int((am&~bm).sum())
        face_gain+=.10*float((am.astype(int)-bm.astype(int)).sum())*step
        face_gain+=.25*float((np.minimum(bd,.06)-np.minimum(ad,.06)).sum())*step
    prob=target_local.astype(float) if p is None else p[sl]
    weights=np.where(target_local,prob,-.5*(1-prob))
    semantic=float((weights*(new.astype(int)-old.astype(int))).sum())*pixel_m**2
    if face_gain < -cfg.run_face_tolerance_m*pixel_m:return reject('face_distance_regression')
    gain=semantic+face_gain+.002*(len(remove)-len(add))+coherence_gain
    if gain<=1e-7 and not allow_prune:return reject('objective')
    return sl,new,dict(objective_gain_m2=gain,new_face_samples=new_matches,semantic_gain_m2=semantic,face_gain_m2=face_gain,coherence_gain_m2=coherence_gain,
                       quarantined_unsupported_connections=breaks if allow_prune else [])


def _commit(primitives,union,remove,add,edit,log,operation):
    sl,patch,detail=edit;union[sl]=patch
    removed=[primitives[i].get('id',i) for i in remove]
    keep=[c for i,c in enumerate(primitives) if i not in remove]+add
    for i,c in enumerate(keep):c['id']=i
    log.append(dict(operation=operation,removed_ids=removed,added=len(add),**detail))
    return keep


def refine_structure(result,target,barrier,pixel_m,probability=None,forbidden=None,config=None):
    """Three ordered stages: recover faces, quarantine weak islands, fit shared runs."""
    from .decompose import Config,mask_metrics,_topology,rectangle_overlap_metrics,_local_mask
    cfg=config or Config();target=np.asarray(target,bool);barrier=np.asarray(barrier,bool)
    if barrier.shape!=target.shape or pixel_m<=0:raise ValueError('Invalid scan/frame')
    if forbidden is not None:
        forbidden=np.asarray(forbidden,bool)
        if forbidden.shape!=target.shape or (target&forbidden).any():raise ValueError('Apertures must be absent from target')
    p=None if probability is None else np.asarray(probability,np.float32)
    if p is not None and (p.shape!=target.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1))):raise ValueError('Invalid semantic probability')
    out=copy.deepcopy({k:v for k,v in result.items() if k not in ['mask','dense','merging']})
    # Re-refinement operates on original analytic components, not a lossy
    # representative centerline of a previously merged wall object.
    primitives=copy.deepcopy([p for c in result['primitives'] for p in (c['parts'] if c['kind']=='wall_run' else [c])])
    for i,c in enumerate(primitives):c['id']=i
    union=_union(primitives,target.shape);log=[];quarantine=[];rejections={}
    clean=target.copy();removed=np.zeros_like(target)
    if cfg.structure_filter:
        clean,removed,records=filter_detached(target,p,barrier,pixel_m,cfg.detached_probability)
        quarantine.extend(dict(stage='raster',**r) for r in records)
    runs=extract_runs(clean,barrier,pixel_m,cfg,forbidden);profiles=[_profile(r,clean,forbidden,pixel_m,cfg) for r in runs]
    before=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m)
    # Recover uncovered runs first, including gaps that already existed in the
    # old vector graph. Evidence is sampled per centimetre, not per dense point.
    for run,profile in zip(runs,profiles):
        distance=_point_distances(union,profile['points'],pixel_m)
        missing=profile['valid']&(distance>cfg.run_face_tolerance_m)
        for start,end in _intervals(missing):
            if (end-start)*profile['step']*pixel_m<.05:continue
            side=int(np.median(profile['side'][start:end]));width=float(np.median(profile['width'][start:end]))
            if not side or width<cfg.min_width_m:continue
            t0=profile['t'][start]-.02/pixel_m;t1=profile['t'][end-1]+.02/pixel_m
            c=_band(run,t0,t1,width,side,pixel_m)
            edit=_accept_edit(primitives,union,[],[c],clean,barrier,forbidden,p,profiles,pixel_m,cfg,diagnostics=rejections)
            if edit and edit[2].get('new_face_samples',0):primitives=_commit(primitives,union,[],[c],edit,log,'recover_supported_run')
    if cfg.structure_filter:
        # Filter final geometry as well: a primitive can become detached even
        # when its parent raster component was attached through a weak thin tail.
        primitives,records=_quarantine_detached_geometry(primitives,target.shape,p,barrier,pixel_m,cfg)
        quarantine.extend(records);union=_union(primitives,target.shape)
        # Unsupported diagonal tails may be attached, but may not borrow their
        # own direction as evidence. Remove only weak tails near a verified run,
        # with preserved connections and no loss of matched observed faces.
        for c in sorted(list(primitives),key=lambda x:x.get('length',1)*x.get('width',1)):
            if not any(n is c for n in primitives):continue
            if c['kind']!='rectangle' or c.get('scan_verified'):continue
            center=np.array(c['center']);corners=np.array(c['polygon']);near=[];verified=_verified_long_face(c,runs,pixel_m,cfg.run_face_tolerance_m)
            for r in runs:
                along=corners@r['u'];cross=corners@r['normal']-r['offset']
                normal=0. if cross.min()<=0<=cross.max() else np.abs(cross).min()*pixel_m
                if along.max()>=r['t0']-.10/pixel_m and along.min()<=r['t1']+.10/pixel_m and normal<=.08:
                    delta=abs((c['angle']-np.arctan2(r['u'][1],r['u'][0])+np.pi/4)%(np.pi/2)-np.pi/4)
                    near.append(delta)
            if verified or not near or min(near)<=np.radians(10) or p is None:continue
            b=np.array(c['polygon']);lo=np.maximum(0,np.floor(b.min(0)-1).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(b.max(0)+2).astype(int));sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));patch=_local_mask(c,lo,tuple((hi-lo)[::-1]))
            if float(p[sl][patch].mean())>=cfg.detached_probability:continue
            index=next(i for i,x in enumerate(primitives) if x is c)
            remove=[index]
            edit=_accept_edit(primitives,union,remove,[],clean,barrier,forbidden,p,profiles,pixel_m,cfg,allow_prune=True,diagnostics=rejections)
            if edit is None:
                # A false tail must not be protected merely because deleting it
                # detaches another weak unsupported fragment. Quarantine that
                # fragment as a group; scan-backed continuations still veto edits.
                trial_union=_union([n for i,n in enumerate(primitives) if i!=index],target.shape)
                _,orphan,_=filter_detached(trial_union,p,barrier,pixel_m,cfg.detached_probability)
                for i,n in enumerate(primitives):
                    if i==index:continue
                    bb=np.array(n['polygon']);ll=np.maximum(0,np.floor(bb.min(0)-1).astype(int));hh=np.minimum(target.shape[::-1],np.ceil(bb.max(0)+2).astype(int));ss=(slice(ll[1],hh[1]),slice(ll[0],hh[0]));pp=_local_mask(n,ll,tuple((hh-ll)[::-1]))
                    if pp.any() and np.all(orphan[ss][pp]):remove.append(i)
                if len(remove)>1:edit=_accept_edit(primitives,union,remove,[],clean,barrier,forbidden,p,profiles,pixel_m,cfg,allow_prune=True,diagnostics=rejections)
            if edit:
                quarantine.extend(dict(stage='vector',reason='weak_diagonal_without_matching_scan_direction' if i==index else 'unsupported_fragment_after_tail_removal',primitive=copy.deepcopy(primitives[i])) for i in remove)
                primitives=_commit(primitives,union,remove,[],edit,log,'quarantine_diagonal_tail')
        # Remove weak fragments newly orphaned by quarantining a false connector.
        primitives,records=_quarantine_detached_geometry(primitives,target.shape,p,barrier,pixel_m,cfg)
        quarantine.extend(records);union=_union(primitives,target.shape)
    if cfg.structure_joint:
        for run,profile in zip(runs,profiles):
            for start,end in profile['pieces']:
                if (end-start)*profile['step']*pixel_m<cfg.run_min_m:continue
                side=int(np.median(profile['side'][start:end]));width=float(np.median(profile['width'][start:end]))
                if not side:continue
                c=_band(run,profile['t'][start],profile['t'][end-1],width,side,pixel_m)
                b=np.array(c['polygon']);lo=np.maximum(0,np.floor(b.min(0)-1).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(b.max(0)+2).astype(int));shape=tuple((hi-lo)[::-1]);candidate=_local_mask(c,lo,shape);remove=[]
                residual=[];old_widths=[]
                for i,old in enumerate(primitives):
                    if old['kind']!='rectangle':continue
                    delta=abs((old['angle']-c['angle']+np.pi/2)%np.pi-np.pi/2)
                    if delta>np.radians(3):continue
                    area=old['length']*old['width']
                    intersection,_=cv2.intersectConvexConvex(np.array(old['polygon'],np.float32),np.array(c['polygon'],np.float32))
                    old_u=np.array([np.cos(old['angle']),np.sin(old['angle'])]);old_center=np.array(old['center'])
                    denominator=float(old_u@run['u'])
                    if abs(denominator)<.9:continue
                    cut=np.sort((np.array([profile['t'][start],profile['t'][end-1]])-old_center@run['u'])/denominator)
                    left=max(-old['length']/2,cut[0]);right=min(old['length']/2,cut[1]);overlap=max(0.,right-left)
                    if overlap*pixel_m<.15 or intersection<.65*overlap*old['width']:continue
                    remove.append(i)
                    if abs(old['width']*pixel_m-width)<=.04:old_widths.append(old['width']*pixel_m)
                    # Preserve unobserved continuations exactly in their own
                    # frame; a local face observation cannot delete either end.
                    from .decompose import rectangle
                    for a,b in [(-old['length']/2,left),(right,old['length']/2)]:
                        if (b-a)>.5:
                            segment=rectangle(old_center+old_u*(a+b)/2,b-a+.5,old['width'],old['angle'])
                            segment['preserved_continuation']=True
                            residual.append(segment)
                if not remove:continue
                if old_widths:width=float(np.median(old_widths))
                # Share run endpoints with already fitted neighbors instead of
                # shortening to LSD's interior endpoint and opening a tiny seam.
                t0=profile['t'][start];t1=profile['t'][end-1]
                ends=np.vstack([primitives[i]['polygon'] for i in remove])@run['u']
                if t0-ends.min()<=.03/pixel_m:t0=min(t0,float(ends.min()))
                if ends.max()-t1<=.03/pixel_m:t1=max(t1,float(ends.max()))
                best=None
                # Optimize the group's common face offset and opposite face
                # together. Every trial uses the final union and all nearby
                # observed faces; individual pieces cannot vote on their own fit.
                for shift in [0.,-pixel_m,pixel_m,-.005,.005]:
                    shifted=dict(run,offset=run['offset']+shift/pixel_m)
                    for dw in [0.,-.005,.005]:
                        trial_width=float(np.clip(width+dw,cfg.min_width_m,cfg.run_max_width_m))
                        trial=_band(shifted,t0,t1,trial_width,side,pixel_m)
                        edit=_accept_edit(primitives,union,remove,[trial]+residual,clean,barrier,forbidden,p,profiles,pixel_m,cfg,diagnostics=rejections)
                        if edit and (best is None or edit[2]['objective_gain_m2']>best[0][2]['objective_gain_m2']):best=(edit,trial)
                if best:primitives=_commit(primitives,union,remove,[best[1]]+residual,best[0],log,'fit_shared_face_and_width')
        # Carry a verified face through its preserved occluded continuation.
        # Only scan-verified neighbors may supply this reference.
        from .decompose import rectangle,_axis_reference
        for c in list(primitives):
            if not c.get('preserved_continuation'):continue
            reference=_axis_reference(c,np.empty((0,4)),primitives,pixel_m,cfg)
            if reference is None:continue
            angle=reference['target'];u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
            peers=[n for n in primitives if n is not c and n.get('scan_verified') and n['kind']=='rectangle' and
                   abs((n['angle']-angle+np.pi/2)%np.pi-np.pi/2)<np.radians(3)]
            corners=np.array(c['polygon']);along=corners@u;center=np.array(c['center']);offsets=[]
            for n in peers:
                nc=np.array(n['center']);span=c['length']/2+n['length']/2+.03/pixel_m
                if abs((nc-center)@u)>span:continue
                for a in [-1,1]:
                    for b in [-1,1]:offsets.append(float(nc@v+a*n['width']/2-(center@v+b*c['width']/2)))
            if not offsets:continue
            shift=min(offsets,key=abs)
            if abs(shift)*pixel_m>.02:continue
            trial=rectangle(u*(along.min()+along.max())/2+v*(center@v+shift),float(np.ptp(along)),c['width'],angle)
            trial['preserved_continuation']=True;trial['orientation_source']='scan_verified_neighbor'
            index=next(i for i,n in enumerate(primitives) if n is c)
            delta=abs((c['angle']-angle+np.pi/2)%np.pi-np.pi/2)
            coherence=cfg.axis_weight*c['length']*c['width']*pixel_m**2*(delta/np.radians(cfg.axis_max_degrees))**2
            edit=_accept_edit(primitives,union,[index],[trial],clean,barrier,forbidden,p,profiles,pixel_m,cfg,diagnostics=rejections,coherence_gain=coherence)
            if edit:primitives=_commit(primitives,union,[index],[trial],edit,log,'align_verified_continuation')
        # At a verified T/L junction, adjust the endpoint to the neighbor's
        # actual footprint plane, rather than intersecting arbitrary centerlines.
        for c in list(primitives):
            if c['kind']!='rectangle' or not c.get('scan_verified'):continue
            u=np.array([np.cos(c['angle']),np.sin(c['angle'])]);center=np.array(c['center'])
            for n in list(primitives):
                if n is c or n['kind']!='rectangle' or not n.get('scan_verified'):continue
                cross=abs(np.sin(c['angle']-n['angle']))
                if cross<.8:continue
                pts=np.array(n['polygon']);normal=np.array([-u[1],u[0]])
                if (pts@normal).min()>center@normal+c['width']/2 or (pts@normal).max()<center@normal-c['width']/2:continue
                projection=(pts-center)@u
                for sign in [-1,1]:
                    endpoint=sign*c['length']/2;goal=float(projection.max() if sign>0 else projection.min())
                    if abs(goal-endpoint)*pixel_m>.03:continue
                    change=goal-endpoint
                    trial=rectangle(center+u*change/2,c['length']+sign*change,c['width'],c['angle'])
                    if trial['length']<=0:continue
                    trial['scan_verified']=True;trial['junction_endpoint']=True
                    index=next((i for i,x in enumerate(primitives) if x is c),None)
                    if index is None:break
                    edit=_accept_edit(primitives,union,[index],[trial],clean,barrier,forbidden,p,profiles,pixel_m,cfg,diagnostics=rejections)
                    if edit:primitives=_commit(primitives,union,[index],[trial],edit,log,'fit_verified_junction_endpoint');break
                if not any(x is c for x in primitives):break
    # Remove weak unsupported protrusions left by formerly two-parent junctions.
    # Material genuinely joining two independently observed wall bodies survives.
    if cfg.structure_filter and p is not None:
        for c in list(primitives):
            if c['kind']!='junction':continue
            b=np.array(c['polygon']);lo=np.maximum(0,np.floor(b.min(0)-2).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(b.max(0)+3).astype(int));shape=tuple((hi-lo)[::-1]);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));patch=_local_mask(c,lo,shape)
            if not patch.any() or float(p[sl][patch].mean())>=cfg.detached_probability:continue
            touch=cv2.dilate(patch.astype('uint8'),np.ones((3,3),'uint8'))>0;parents=0
            for n in primitives:
                if n is c or n['kind'] not in ['rectangle','arc']:continue
                bb=np.array(n['polygon'])
                if np.any(bb.max(0)<lo) or np.any(bb.min(0)>hi):continue
                if (_local_mask(n,lo,shape)&touch).any() and _verified_long_face(n,runs,pixel_m,cfg.run_face_tolerance_m):parents+=1
            if parents>=2:continue
            index=next(i for i,n in enumerate(primitives) if n is c)
            edit=_accept_edit(primitives,union,[index],[],clean,barrier,forbidden,p,profiles,pixel_m,cfg,allow_prune=True,diagnostics=rejections)
            if edit:
                quarantine.append(dict(stage='vector',reason='weak_junction_tail_without_two_observed_parents',primitive=copy.deepcopy(c)))
                primitives=_commit(primitives,union,[index],[],edit,log,'quarantine_junction_tail')
        primitives,records=_quarantine_detached_geometry(primitives,target.shape,p,barrier,pixel_m,cfg)
        quarantine.extend(records);union=_union(primitives,target.shape)
    # Remove redundant junction patches when they contribute exactly zero pixels.
    # Never blindly fill holes or smooth over a genuine aperture.
    for i in range(len(primitives)-1,-1,-1):
        if primitives[i]['kind']!='junction':continue
        c=primitives[i];b=np.array(c['polygon']);lo=np.maximum(0,np.floor(b.min(0)-1).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(b.max(0)+2).astype(int));shape=tuple((hi-lo)[::-1]);own=_local_mask(c,lo,shape);other=np.zeros(shape,bool)
        for j,n in enumerate(primitives):
            if j==i:continue
            bb=np.array(n['polygon'])
            if np.any(bb.max(0)<lo) or np.any(bb.min(0)>hi):continue
            other|=_local_mask(n,lo,shape)
        if not (own&~other).any():primitives.pop(i);log.append(dict(operation='remove_redundant_junction'))
    for i,c in enumerate(primitives):c['id']=i
    out.update(primitives=primitives,mask=union,contacts=_topology(primitives,target.shape),metrics=mask_metrics(target,union),
               rectangle_overlap=rectangle_overlap_metrics(primitives,target.shape,pixel_m),
               structure=dict(scan_runs=len(runs),before=before,after=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m),
                              edits=log,rejections=rejections,quarantine=quarantine,filtered_target_pixels=int(removed.sum()),
                              policy='native reliable scan faces; sparse returns neutral; original semantic confidence for detached filtering'),
               wall_faces=[dict(angle=float(np.arctan2(r['u'][1],r['u'][0])),offset_pixels=r['offset'],
                                start=(r['u']*r['t0']+r['normal']*r['offset']).tolist(),
                                end=(r['u']*r['t1']+r['normal']*r['offset']).tolist(),scatter_m=r['scatter_m']) for r in runs])
    if 'dense' in result:out['dense']=result['dense']
    from dataclasses import asdict
    out['config']=asdict(cfg)
    # Never keep world footprints from a pre-edit cached result.
    out.pop('world_regions',None)
    if 'pixel_to_world' in out:
        matrix=np.asarray(out['pixel_to_world'])
        def world(points):
            points=np.asarray(points);return (np.c_[points,np.ones(len(points))]@matrix.T).tolist()
        out['world_regions']=[dict(id=c['id'],kind=c['kind'],footprint=world(c['polygon']),
                                  holes=[world(h) for h in c.get('holes',[])],axis=world(c['axis']) if 'axis' in c else None,
                                  mask_width_m=c.get('width',0)*pixel_m) for c in primitives]
    return out
