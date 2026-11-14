"""Bounded simplification of fragmented straight walls against TWO observed faces."""
from copy import deepcopy
import numpy as np
import cv2
from .wall_runs import _union,_profile,extract_runs,_accept_edit,_commit,_boundary_distance,face_metrics,_intervals
from .decompose import rectangle,_bounds,_local_mask,_topology,mask_metrics,rectangle_overlap_metrics


def _angle_delta(a,b):return abs((a-b+np.pi/2)%np.pi-np.pi/2)


def _outline_measurement(mask,pixel_m,cfg):
    """Ignore only small narrow enclosed fitting slits, not rooms/apertures."""
    n,labels,stats,_=cv2.connectedComponentsWithStats((~mask).astype('uint8'),8)
    exterior=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]));filled=mask.copy();protected=np.zeros_like(mask)
    for i in range(1,n):
        if i in exterior:continue
        x,y,w,h,area=stats[i];region=labels[y:y+h,x:x+w]==i
        if area*pixel_m**2<=cfg.simplify_hole_area_m2 and min(w,h)*pixel_m<=cfg.simplify_seam_width_m:
            filled[y:y+h,x:x+w]|=region
        else:protected[y:y+h,x:x+w]|=region
    return filled,protected


def _connected(a,b,shape):
    lo=np.maximum(0,np.floor(np.minimum(np.min(a['polygon'],0),np.min(b['polygon'],0))-2).astype(int))
    hi=np.minimum(shape[::-1],np.ceil(np.maximum(np.max(a['polygon'],0),np.max(b['polygon'],0))+3).astype(int))
    # Bounding-box gate keeps long distant parallel walls out of raster tests.
    aa=np.array(a['polygon']);bb=np.array(b['polygon'])
    if np.any(aa.min(0)>bb.max(0)+1) or np.any(bb.min(0)>aa.max(0)+1):return False
    size=tuple((hi-lo)[::-1]);ma=_local_mask(a,lo,size);mb=_local_mask(b,lo,size)
    return bool((cv2.dilate(ma.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)&mb).any())


def _preserve_aperture_ends(candidate,group,forbidden,pixel_m,cfg):
    """Fit the longest clear interior span; retain original aperture/end geometry."""
    if forbidden is None:return [candidate],None
    lo,hi=_bounds(candidate,forbidden.shape);shape=tuple((hi-lo)[::-1]);patch=_local_mask(candidate,lo,shape)
    yy,xx=np.nonzero(patch&forbidden[lo[1]:hi[1],lo[0]:hi[0]])
    if not len(xx):return [candidate],None
    u=np.array([np.cos(candidate['angle']),np.sin(candidate['angle'])]);center=np.array(candidate['center']);mid=center@u
    t0=mid-candidate['length']/2;t1=mid+candidate['length']/2
    blocked=np.zeros(int(np.ceil(t1-t0))+1,bool);ids=np.clip(np.floor(np.c_[xx+lo[0],yy+lo[1]]@u-t0).astype(int),0,len(blocked)-1);blocked[ids]=True
    blocked=cv2.dilate(blocked.astype('uint8')[None,:],np.ones((1,5),'uint8')).ravel()>0
    intervals=_intervals(~blocked)
    if not intervals:return None,'aperture'
    a,b=max(intervals,key=lambda ab:ab[1]-ab[0]);start=t0+a+1;end=t0+b-1
    if (end-start)*pixel_m<cfg.simplify_min_length_m or (end-start)<.65*(t1-t0):return None,'interior_aperture'
    main=rectangle(center+u*((start+end)/2-mid),end-start,candidate['width'],candidate['angle']);main['scan_verified']=True;main['two_face_fit']=candidate['two_face_fit']
    additions=[main]
    for old in group:
        d=np.array([np.cos(old['angle']),np.sin(old['angle'])]);c=np.array(old['center']);dot=d@u
        cut=np.sort((np.array([start+2,end-2])-c@u)/dot)
        # Residuals are subsets of the originals, with a two-pixel overlap into
        # the fitted body so a slight angular difference cannot open a seam.
        for left,right in [(-old['length']/2,min(old['length']/2,cut[0])),(max(-old['length']/2,cut[1]),old['length']/2)]:
            if right-left>.25:additions.append(rectangle(c+d*(left+right)/2,right-left,old['width'],old['angle']))
    from .wall_merge import polygon_union
    boundary=polygon_union([p['polygon'] for p in additions])
    if boundary is not None:
        combined=dict(main,kind='wall_run',**boundary,parts=additions,axis_role='representative',preserved_aperture_ends=True,
                      merged_from=[p.get('id',i) for i,p in enumerate(group)],
                      width_range_m=[min(p['width'] for p in additions)*pixel_m,max(p['width'] for p in additions)*pixel_m])
        return [combined],None
    return additions,None


def _fit_two_faces(parts,barrier,pixel_m,cfg):
    anchor=max(parts,key=lambda c:c['length']);angle=anchor['angle'];u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
    corners=np.vstack([c['polygon'] for c in parts]);t=corners@u;q=corners@v;t0,t1=float(t.min()),float(t.max())
    if (t1-t0)*pixel_m<cfg.simplify_min_length_m:return None,'short'
    # Stable initial face offsets from the existing UNION cross sections; each
    # centimetre votes equally, so many little tiles cannot dominate the fit.
    lo=np.maximum(0,np.floor(corners.min(0)-cfg.simplify_max_shift_m/pixel_m-3).astype(int));hi=np.minimum(barrier.shape[::-1],np.ceil(corners.max(0)+cfg.simplify_max_shift_m/pixel_m+4).astype(int));size=tuple((hi-lo)[::-1])
    body=np.zeros(size,bool)
    for c in parts:body|=_local_mask(c,lo,size)
    yy,xx=np.nonzero(body);points=np.c_[xx+lo[0],yy+lo[1]];bt=points@u;bq=points@v
    step=max(1,int(round(.02/pixel_m)));bins=np.floor((bt-t0)/step).astype(int);lower=[];upper=[]
    order=np.argsort(bins,kind='stable');splits=np.r_[0,np.where(np.diff(bins[order]))[0]+1,len(order)]
    for a,b in zip(splits,splits[1:]):
        values=bq[order[a:b]]
        if len(values):lower.append(float(values.min()));upper.append(float(values.max()))
    if not lower:return None,'empty'
    q0=float(np.median(lower));q1=float(np.median(upper));width=(q1-q0)*pixel_m
    if not cfg.min_width_m<=width<=.5:return None,'width'
    yy,xx=np.nonzero(barrier[lo[1]:hi[1],lo[0]:hi[0]]);scan=np.c_[xx+lo[0],yy+lo[1]].astype(float)
    st=scan@u;sq=scan@v;trim=min(.08/pixel_m,.10*(t1-t0))
    inner=(st>=t0+trim)&(st<=t1-trim)
    face=[];directions=[];coverage=[]
    for offset in [q0,q1]:
        eligible=inner&(np.abs(sq-offset)*pixel_m<=cfg.simplify_max_shift_m)
        cloud=scan[eligible]
        if len(cloud)<max(12,.2/pixel_m):return None,'missing_second_face'
        fit=cv2.fitLine(cloud.astype('float32'),cv2.DIST_HUBER,0,.01,.001).ravel();d=fit[:2].astype(float);d/=np.linalg.norm(d)
        if d@u<0:d=-d
        if _angle_delta(np.arctan2(d[1],d[0]),angle)>np.radians(2):return None,'unsupported_direction'
        normal=np.array([-d[1],d[0]]);residual=cloud@normal;center=float(np.median(residual))
        cloud=cloud[np.abs(residual-center)*pixel_m<=.0125]
        if len(cloud)<12:return None,'face_scatter'
        along=cloud@u;nb=max(1,int(np.ceil((t1-t0-2*trim)/step)))
        occupied=np.unique(np.clip(np.floor((along-t0-trim)/step).astype(int),0,nb-1))
        support=len(occupied)/nb
        if support<cfg.simplify_min_face_coverage:return None,'fragmented_face'
        coverage.append(float(support));face.append(cloud);directions.append(d)
    direction=directions[0]+directions[1];direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
    if _angle_delta(np.arctan2(directions[0][1],directions[0][0]),np.arctan2(directions[1][1],directions[1][0]))>np.radians(.5):return None,'nonparallel_faces'
    offsets=[float(np.median(c@normal)) for c in face];q0,q1=sorted(offsets)
    width=(q1-q0)*pixel_m
    if not cfg.min_width_m<=width<=.5:return None,'width'
    along=corners@direction;t0,t1=float(along.min()),float(along.max())
    proposal=rectangle(direction*(t0+t1)/2+normal*(q0+q1)/2,t1-t0,q1-q0,float(np.arctan2(direction[1],direction[0])))
    proposal['scan_verified']=True;proposal['two_face_fit']=dict(coverage=coverage,width_m=width,source_components=len(parts))
    return proposal,None


def simplify_result(result,target,barrier,pixel_m=.0025,probability=None,forbidden=None,config=None):
    from .decompose import Config
    cfg=config or Config();target=np.asarray(target,bool);barrier=np.asarray(barrier,bool)
    if barrier.shape!=target.shape or result['mask'].shape!=target.shape:raise ValueError('Simplification frames differ')
    parts=deepcopy([p for c in result['primitives'] for p in (c['parts'] if c['kind']=='wall_run' else [c])])
    for i,c in enumerate(parts):c['id']=i
    original=_union(parts,target.shape);union=original.copy();log=[];rejected={};proposals=[]
    runs=extract_runs(target,barrier,pixel_m,cfg,forbidden);profiles=[_profile(r,target,forbidden,pixel_m,cfg) for r in runs]
    before=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m)
    processed=set()
    for anchor in sorted(parts,key=lambda c:-c.get('length',0)):
        if anchor['kind']!='rectangle' or id(anchor) in processed or not any(c is anchor for c in parts):continue
        group=[anchor];changed=True
        while changed:
            changed=False
            for c in parts:
                if c['kind']!='rectangle' or any(c is x for x in group):continue
                if _angle_delta(c['angle'],anchor['angle'])>np.radians(2):continue
                if any(_connected(c,n,target.shape) for n in group):group.append(c);changed=True
        processed.update(id(c) for c in group)
        if len(group)<2:continue
        diagnostic=dict(anchor=anchor['center'],components=len(group));proposals.append(diagnostic)
        candidate,reason=_fit_two_faces(group,barrier,pixel_m,cfg)
        if candidate is None:diagnostic['reason']=reason;rejected[reason]=rejected.get(reason,0)+1;continue
        best=None;base_candidate=candidate;group_diagnostic=diagnostic
        normal=np.array([-np.sin(candidate['angle']),np.cos(candidate['angle'])])
        offsets=sorted(set([0.,pixel_m,.005]))
        for low,high in [(a,b) for a in offsets for b in offsets if a+b<=.005+1e-9]:
            width=base_candidate['width']-(low+high)/pixel_m
            if width*pixel_m<cfg.min_width_m:continue
            candidate=rectangle(np.array(base_candidate['center'])+normal*(low-high)/pixel_m/2,base_candidate['length'],width,base_candidate['angle'])
            candidate['scan_verified']=True;candidate['two_face_fit']=dict(base_candidate['two_face_fit'],inward_offsets_m=[low,high],width_m=width*pixel_m)
            diagnostic=dict(inward_offsets_m=[low,high])
            additions,reason=_preserve_aperture_ends(candidate,group,forbidden,pixel_m,cfg)
            if additions is None:diagnostic['reason']=reason;rejected[reason]=rejected.get(reason,0)+1;continue
            remove=[i for i,c in enumerate(parts) if any(c is n for n in group)]
            bounds=np.vstack([c['polygon'] for c in additions]+[c['polygon'] for c in group]);lo=np.maximum(0,np.floor(bounds.min(0)-3).astype(int));hi=np.minimum(target.shape[::-1],np.ceil(bounds.max(0)+4).astype(int));shape=tuple((hi-lo)[::-1]);candidate_only=np.zeros(shape,bool)
            for c in additions:candidate_only|=_local_mask(c,lo,shape)
            new=candidate_only.copy();old=union[lo[1]:hi[1],lo[0]:hi[0]]
            # Include other tiles and the end junctions when measuring the actual
            # wall outline. Omitting a transverse divider invents an interior gap.
            for i,c in enumerate(parts):
                if i in remove:continue
                b=np.asarray(c['polygon'])
                if np.any(b.max(0)<lo) or np.any(b.min(0)>hi):continue
                new|=_local_mask(c,lo,shape)
            if forbidden is not None:
                fy,fx=np.nonzero(new&forbidden[lo[1]:hi[1],lo[0]:hi[0]])
                if len(fx):diagnostic['aperture_overlap']=dict(pixels=len(fx),bounds=[int(fx.min()+lo[0]),int(fy.min()+lo[1]),int(fx.max()+lo[0]),int(fy.max()+lo[1])])
            # Movement is measured against the OUTER union, never internal seams.
            old_outline,protected_voids=_outline_measurement(old,pixel_m,cfg)
            if (new&protected_voids).any():rejected['protected_void']=rejected.get('protected_void',0)+1;continue
            new_outline,_=_outline_measurement(new,pixel_m,cfg)
            old_edge=old_outline&~cv2.erode(old_outline.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
            new_edge=new_outline&~cv2.erode(new_outline.astype('uint8'),np.ones((3,3),'uint8')).astype(bool)
            od=_boundary_distance(old_outline,pixel_m);nd=_boundary_distance(new_outline,pixel_m)
            shift=max(float(od[new_edge].max(initial=0)),float(nd[old_edge].max(initial=0)))
            diagnostic['max_boundary_shift_m']=shift
            if old_edge.any():
                yy,xx=np.nonzero(old_edge);k=int(np.argmax(nd[old_edge]));diagnostic['largest_old_edge_shift_xy']=[int(xx[k]+lo[0]),int(yy[k]+lo[1])]
            if shift>cfg.simplify_max_shift_m+pixel_m:rejected['boundary_shift']=rejected.get('boundary_shift',0)+1;continue
            # Absorb transverse dividers only if the new rectangle contains them.
            for i,c in enumerate(parts):
                if i in remove or c['kind']!='rectangle':continue
                bb=np.array(c['polygon'])
                if np.any(bb.min(0)<lo) or np.any(bb.max(0)>hi):continue
                patch=_local_mask(c,lo,shape)
                if patch.any() and not (patch&~candidate_only).any():remove.append(i)
            edit=_accept_edit(parts,union,remove,additions,target,barrier,forbidden,probability,profiles,pixel_m,cfg,diagnostics=rejected,target_loss_fraction=.03)
            if edit:
                edit[2]['max_boundary_shift_m']=shift
                if best is None or edit[2]['objective_gain_m2']>best[0][2]['objective_gain_m2']:best=(edit,remove,additions,diagnostic.copy())
        if best:
            group_diagnostic.update(best[3])
            parts=_commit(parts,union,best[1],best[2],best[0],log,'simplify_two_observed_faces')
        else:group_diagnostic['reason']='no_safe_fit'
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense','merging','world_regions']})
    for i,c in enumerate(parts):c['id']=i
    out.update(primitives=parts,mask=union,contacts=_topology(parts,target.shape),metrics=mask_metrics(target,union),rectangle_overlap=rectangle_overlap_metrics(parts,target.shape,pixel_m),
               simplification=dict(edits=log,rejections=rejected,proposals=proposals,changed_pixels=int((union!=original).sum()),before=before,after=face_metrics(union,profiles,pixel_m,cfg.run_face_tolerance_m)))
    if 'dense' in result:out['dense']=result['dense']
    if 'pixel_to_world' in out:
        matrix=np.asarray(out['pixel_to_world'])
        def world(p):return (np.c_[p,np.ones(len(p))]@matrix.T).tolist()
        out['world_regions']=[dict(id=c['id'],kind=c['kind'],footprint=world(c['polygon']),holes=[world(h) for h in c.get('holes',[])],axis=world(c['axis']) if 'axis' in c else None,mask_width_m=c.get('width',0)*pixel_m) for c in parts]
    return out
