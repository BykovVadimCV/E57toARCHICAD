"""Area-first wall decomposition: rectangles, annular sectors and junction polygons.

Coordinates are pixel centres (x=column, y=row). Footprints are authoritative;
axes are derived export conveniences. The area fit stops at configurable mask
coverage; a subsequent evidence pass can recover observed missing wall runs.
Dense point returns are positive, bounded FACE evidence; absent/sparse returns
are neutral. Input density must use netinput's log1p(count)/log1p(4000) encoding.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
import math
import numpy as np
import cv2


@dataclass(frozen=True)
class Config:
    min_length_m: float = .20
    min_width_m: float = .04
    max_width_m: float = 1.20
    spill_limit: float = .35
    spill_weight: float = .75
    clear_spill_weight: float = .08
    clear_spill_limit: float = .65
    max_clear_extension_m: float = .20
    barrier_contact_m: float = .005
    coverage_stop: float = .95
    proposal_pixel_m: float = .01
    proposal_fill: float = .75
    overlap_weight: float = 2.0
    overlap_free_fraction: float = .10
    complexity_m2: float = .004
    min_gain_m2: float = .006
    max_primitives: int = 180
    max_angles: int = 6
    proposal_limit: int = 1500
    constraint_refine_limit: int = 16
    dense_count_2cm: float = 200.0
    density_quantile: float = .90
    density_weight: float = .25
    scan_face_reward: float = 4.0
    scan_overshoot_weight: float = 8.0
    scan_face_sigma_m: float = .005
    scan_face_range_m: float = .04
    scan_face_scope_m: float = .05
    axis_weight: float = .12
    axis_max_degrees: float = 3.0
    axis_scan_min_length_m: float = .40
    axis_neighbour_m: float = .08
    axis_preserve_connectivity: bool = True
    structure_refine: bool = True
    structure_filter: bool = True
    structure_joint: bool = True
    run_min_m: float = .30
    run_face_tolerance_m: float = .015
    run_max_width_m: float = .40
    detached_probability: float = .75
    merge_wall_runs: bool = True
    simplify_wall_runs: bool = True
    simplify_max_shift_m: float = .04
    simplify_min_length_m: float = .60
    simplify_min_face_coverage: float = .70
    simplify_hole_area_m2: float = .0025
    simplify_seam_width_m: float = .02
    face_tolerance_m: float = .04
    junction_area_m2: float = .10


def dense_evidence(density, probability, pixel_m, cfg):
    """Only contiguous high-count, semantically plausible returns may contribute.

    Counts are normalized to 2cm cells. The absolute floor cannot be weakened by
    a low-density scan's quantile. Saturated inputs cannot resolve >4000 counts.
    """
    if density is None:
        return np.zeros(probability.shape, np.float32), {'available': False, 'active_pixels': 0}
    d = np.asarray(density, dtype=np.float32)
    if d.shape != probability.shape or not np.isfinite(d).all() or np.any((d < 0) | (d > 1)):
        raise ValueError('density must be finite normalized [H,W] in [0,1]')
    counts = np.expm1(d * np.log1p(4000.0)) * (.02 / pixel_m)**2
    eligible = (probability >= .5) & (counts > 0)
    threshold = max(cfg.dense_count_2cm,
                    float(np.quantile(counts[eligible], cfg.density_quantile)) if eligible.any() else 0)
    active = (counts >= threshold) & eligible
    # Isolated dense returns are not a structural cue. Do not dilate into empty space.
    neighbours = cv2.filter2D(active.astype(np.float32), -1, np.ones((3, 3), np.float32),
                             borderType=cv2.BORDER_CONSTANT)
    active &= neighbours >= 3
    strength = active.astype(np.float32)  # bounded: 4000 returns cannot dominate 200
    return strength, {'available': True, 'count_threshold_2cm': threshold,
                      'active_pixels': int(active.sum()), 'sparse_policy': 'neutral'}


def rasterize(primitive, shape):
    """Rasterize analytic geometry, including polygon holes, at pixel centres."""
    h, w = shape
    bounds = np.asarray(primitive['polygon'], float)
    x0, y0 = np.maximum(0, np.floor(bounds.min(0)-1).astype(int))
    x1, y1 = np.minimum([w, h], np.ceil(bounds.max(0)+2).astype(int))
    out = np.zeros(shape, bool)
    if x1 <= x0 or y1 <= y0:
        return out
    if primitive['kind']=='wall_region':
        from shapely.geometry import Polygon
        from shapely import intersects_xy
        geometry=Polygon(primitive['polygon'],primitive.get('holes',[]))
        xx=np.arange(x0,x1)[None,:]
        for row in range(y0,y1,128):
            stop=min(y1,row+128)
            out[row:stop,x0:x1]=intersects_xy(geometry,xx,np.arange(row,stop)[:,None])
        return out
    y, x = np.mgrid[y0:y1, x0:x1]
    if primitive['kind'] in ['wall_run','wall_region']:
        from .wall_merge import polygon_mask
        inside=polygon_mask(primitive['polygon'],primitive.get('holes',[]),(x0,y0),(y1-y0,x1-x0))
    elif primitive['kind'] == 'rectangle':
        c = np.asarray(primitive['center']);a = primitive['angle'];u = np.array([np.cos(a),np.sin(a)])
        dx,dy=x-c[0],y-c[1]
        inside=(np.abs(dx*u[0]+dy*u[1]) <= primitive['length']/2+1e-6) & (
            np.abs(-dx*u[1]+dy*u[0]) <= primitive['width']/2+1e-6)
    elif primitive['kind'] == 'arc':
        c=primitive['center'];dx=x-c[0];dy=y-c[1];r=np.hypot(dx,dy)
        a=(np.arctan2(dy,dx)-primitive['start']) % (2*np.pi)
        inside=(np.abs(r-primitive['radius']) <= primitive['width']/2+1e-6) & (a <= primitive['sweep']+1e-6)
    else:
        patch=np.zeros((y1-y0,x1-x0),np.uint8)
        cv2.fillPoly(patch,[np.rint(bounds-[x0,y0]).astype(np.int32)],1)
        for hole in primitive.get('holes',[]):
            cv2.fillPoly(patch,[np.rint(np.asarray(hole)-[x0,y0]).astype(np.int32)],0)
        inside=patch>0
    out[y0:y1,x0:x1]=inside
    return out


def rectangle(center, length, width, angle):
    c=np.asarray(center,float);u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
    corners=np.array([c+s*length/2*u+t*width/2*v for s,t in [(-1,-1),(1,-1),(1,1),(-1,1)]])
    return dict(kind='rectangle',center=c.tolist(),length=float(length),width=float(width),angle=float(angle),
                polygon=corners.tolist(),axis=[(c-length/2*u).tolist(),(c+length/2*u).tolist()])


def arc(center, radius, width, start, sweep):
    a=np.linspace(start,start+sweep,max(12,int(radius*sweep/2)+1));c=np.asarray(center)
    u=np.c_[np.cos(a),np.sin(a)]
    return dict(kind='arc',center=c.tolist(),radius=float(radius),width=float(width),start=float(start),
                sweep=float(sweep),polygon=np.vstack([c+(radius+width/2)*u,c+(radius-width/2)*u[::-1]]).tolist(),
                axis=(c+radius*u).tolist())


def _angles(mask, limit):
    lines=cv2.createLineSegmentDetector().detect(mask.astype(np.uint8)*255)[0]
    bins={0:0.0,90:0.0}
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            length=float(np.hypot(x2-x1,y2-y1))
            if length < 10:continue
            a=int(round(np.degrees(np.arctan2(y2-y1,x2-x1))))%180
            # Only numerical near-axis drift is snapped, not a 7-degree forced prior.
            if min(a,180-a)<=1:a=0
            if abs(a-90)<=1:a=90
            bins[a]=bins.get(a,0)+length
    chosen=[]
    for a in sorted(bins,key=lambda k:-bins[k]):
        if all(min(abs(a-b),180-abs(a-b))>=3 for b in chosen):chosen.append(a)
        if len(chosen)>=limit:break
    return chosen


def _rectangles(mask, px, cfg):
    """Sliding strips propose variable-width rectangles, scored at native resolution.

    Integral rows avoid cubic max-subrectangle storage. Rotations use a padded
    canvas; candidate coordinates are mapped back exactly. No global downsample.
    """
    ys,xs=np.nonzero(mask)
    if not len(xs):return []
    x0,x1=max(0,xs.min()-2),min(mask.shape[1],xs.max()+3)
    y0,y1=max(0,ys.min()-2),min(mask.shape[0],ys.max()+3)
    crop=mask[y0:y1,x0:x1].astype(np.uint8);h,w=crop.shape
    side=int(np.ceil(np.hypot(h,w)))+4
    lo=max(2,int(round(cfg.min_width_m/px)));hi=max(lo,int(round(cfg.max_width_m/px)))
    widths=np.unique(np.rint(np.geomspace(lo,hi,min(18,hi-lo+1))).astype(int))
    candidates=[]
    for deg in _angles(crop,cfg.max_angles):
        M=cv2.getRotationMatrix2D((w/2,h/2),deg,1);M[:,2]+=np.array([side/2-w/2,side/2-h/2])
        rot=cv2.warpAffine(crop,M,(side,side),flags=cv2.INTER_NEAREST)
        cs=np.vstack([np.zeros((1,side),np.int32),np.cumsum(rot,axis=0,dtype=np.int32)])
        inv=cv2.invertAffineTransform(M)
        vertical_edges=np.diff(np.pad(rot.astype(np.int8),((1,1),(0,0))),axis=0)
        tops=np.where(vertical_edges.T==1)[1];bottoms=np.where(vertical_edges.T==-1)[1]
        observed=bottoms-tops
        observed=observed[(observed>=lo)&(observed<=hi)]
        freq=np.bincount(observed,minlength=hi+1)
        extra=np.argsort(freq)[-12:];extra=extra[freq[extra]>0]
        for width in np.unique(np.r_[widths,extra]):
            if width>side:continue
            valid=(cs[width:]-cs[:-width]) >= max(1,math.ceil(cfg.proposal_fill*width))
            # Extract every horizontal interval; keep several offsets per region.
            edges=np.diff(np.pad(valid.astype(np.int8),((0,0),(1,1))),axis=1)
            rows,starts=np.where(edges==1);_,ends=np.where(edges==-1)
            lengths=ends-starts;good=np.flatnonzero(lengths>=max(width, cfg.min_length_m/px))
            if not len(good):continue
            order=good[np.argsort(lengths[good])[::-1]]
            selected=[]
            for i in order:
                row=int(rows[i]);left=int(starts[i]);right=int(ends[i])
                if any(abs(row-r)<max(1,width//3) and abs(left-l)<3 and abs(right-e)<3 for r,l,e in selected):continue
                selected.append((row,left,right))
                c=inv@np.array([(left+right-1)/2,row+(width-1)/2,1.0])+[x0,y0]
                candidates.append(rectangle(c,right-left-.01,width-.01,np.radians(deg)))
                if len(selected)>=24:break
    candidates.sort(key=lambda p:(-p['length']*p['width'],-p['width']))
    return candidates[:cfg.proposal_limit]


def _arcs(mask, px, cfg):
    # Skeleton is only a proposal source here; full area must justify every arc.
    from .vectorize import thin, graph, _circle
    dt=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)
    out=[]
    for edge in graph(thin(mask)):
        ij=np.asarray(edge['pix']);q=ij[:,::-1].astype(float)
        if len(q)<16:continue
        ranges={(0,len(q))}
        for parts in (2,4):
            step=max(16,len(q)//parts)
            for start in range(0,len(q)-15,max(8,step//2)):ranges.add((start,min(len(q),start+step)))
        for start,end in sorted(ranges):
            pts=q[start:end]
            if len(pts)<16:continue
            c,r,rms,emax=_circle(pts)
            if not (.30/px<=r<=40/px) or rms>1.5 or emax>4:continue
            angles=np.unwrap(np.arctan2(pts[:,1]-c[1],pts[:,0]-c[0]))
            if np.median(np.diff(angles))<0:angles=angles[::-1]
            sweep=float(angles[-1]-angles[0])
            if not (.12<sweep<2*np.pi) or r*(1-np.cos(min(sweep,np.pi)/2))*px < .05:continue
            if np.mean(np.diff(angles)>=-.01)<.95:continue
            width=float(np.clip(2*np.median(dt[ij[start:end,0],ij[start:end,1]])-1,
                                cfg.min_width_m/px,cfg.max_width_m/px))
            out.append(arc(c,r,width,float(angles[0]),sweep))
    return out


def _scan_axis_lines(barrier,pixel_m,cfg):
    """Detect long directions only in already gated dense scan evidence."""
    if barrier is None or not cfg.axis_weight:return np.empty((0,4))
    scale=max(1,int(round(.01/pixel_m)))
    padded=np.pad(barrier,((0,(-barrier.shape[0])%scale),(0,(-barrier.shape[1])%scale)))
    pooled=padded.reshape(padded.shape[0]//scale,scale,padded.shape[1]//scale,scale).max((1,3))
    lines=cv2.createLineSegmentDetector().detect(pooled.astype(np.uint8)*255)[0]
    if lines is None:return np.empty((0,4))
    lines=lines.reshape(-1,4)*scale+(scale-1)/2
    lengths=np.linalg.norm(lines[:,2:]-lines[:,:2],axis=1)
    keep=np.where(lengths*pixel_m>=cfg.axis_scan_min_length_m)[0]
    return lines[keep[np.argsort(-lengths[keep],kind='stable')[:256]]]


def _axis_reference(c,lines,neighbours,pixel_m,cfg):
    """Local parallel/perpendicular consensus; no assumed global Manhattan frame."""
    if not cfg.axis_weight or c['kind']!='rectangle':return None
    angles=[];weights=[];sources=[];center=np.array(c['center']);a=c['angle']
    max_delta=np.radians(cfg.axis_max_degrees)
    def add(angle,weight,source):
        delta=(angle-a+np.pi/4)%(np.pi/2)-np.pi/4
        if abs(delta)<=max_delta and weight>0:
            angles.append(a+delta);weights.append(weight);sources.append(source)
    if len(lines):
        starts=lines[:,:2];v=lines[:,2:]-starts;length=np.linalg.norm(v,axis=1)
        t=np.clip(((center-starts)*v).sum(1)/(length**2),0,1)
        dist=np.linalg.norm(center-(starts+t[:,None]*v),axis=1)*pixel_m
        for line,l,d in zip(lines,length,dist):
            # Side-face support must be local; lines across a room are irrelevant.
            if d<=c['width']*pixel_m/2+.06:
                add(np.arctan2(line[3]-line[1],line[2]-line[0]),min(l*pixel_m,3.),'scan')
    bounds=np.array(c['polygon']);lo=bounds.min(0);hi=bounds.max(0)
    for n in neighbours:
        if n['kind']!='rectangle':continue
        # A fitted rectangle is not independent evidence for its own direction.
        nr=n.get('fit',{}).get('axis_reference') or {}
        if not n.get('scan_verified') and 'scan' not in nr.get('sources',[]):continue
        b=np.array(n['polygon']);gap=np.maximum(np.maximum(lo-b.max(0),b.min(0)-hi),0)
        if np.linalg.norm(gap)*pixel_m<=cfg.axis_neighbour_m:
            add(n['angle'],min(n['length']*pixel_m,2.),'neighbour')
    if not weights:return None
    target=float(np.arctan2(np.dot(weights,np.sin(4*np.array(angles))),np.dot(weights,np.cos(4*np.array(angles))))/4)
    target=a+(target-a+np.pi/4)%(np.pi/2)-np.pi/4
    return dict(target=target,strength=float(min(1.,sum(weights)/2)),sources=sorted(set(sources)))


def _contact_guards(c,neighbours,shape):
    """Keep each existing raster contact during fitting, including arc contacts."""
    bounds=np.array(c['polygon']);lo=np.maximum(0,np.floor(bounds.min(0)-2).astype(int))
    hi=np.minimum(shape[::-1],np.ceil(bounds.max(0)+3).astype(int));size=tuple((hi-lo)[::-1])
    own=_local_mask(c,lo,size);guards=[]
    for n in neighbours:
        b=np.array(n['polygon'])
        if np.any(b.max(0)<lo) or np.any(b.min(0)>hi):continue
        touch=cv2.dilate(_local_mask(n,lo,size).astype(np.uint8),np.ones((3,3),np.uint8))>0
        if (own&touch).any():guards.append((lo,hi,touch))
    return guards


def _keeps_contacts(footprint,guards):
    sl,patch=footprint;lo=np.array([sl[1].start,sl[0].start]);hi=np.array([sl[1].stop,sl[0].stop])
    for glo,ghi,touch in guards:
        a=np.maximum(lo,glo);b=np.minimum(hi,ghi)
        if np.any(b<=a):return False
        if not (patch[a[1]-lo[1]:b[1]-lo[1],a[0]-lo[0]:b[0]-lo[0]]&touch[a[1]-glo[1]:b[1]-glo[1],a[0]-glo[0]:b[0]-glo[0]]).any():return False
    return True


def _connectivity_breaks(before,after):
    """Conservative native-raster test: every prior component must survive whole.

    Even a one-pixel surviving fragment counts, so tiny accidental splits are
    rejected rather than silently hidden under a reporting area threshold.
    This does not establish geometric truth or detect all local holes/gaps.
    """
    old_n,old=cv2.connectedComponents(np.asarray(before,np.uint8),connectivity=8)
    _,new=cv2.connectedComponents(np.asarray(after,np.uint8),connectivity=8)
    breaks=[]
    for i in range(1,old_n):
        ids,counts=np.unique(new[old==i],return_counts=True)
        parts=[dict(component=int(j),overlap_pixels=int(n)) for j,n in zip(ids,counts) if j]
        if len(parts)!=1:breaks.append(dict(previous_component=i,parts=parts))
    return breaks


def decompose(mask, pixel_m=.02, probability=None, density=None, config=None,
              barrier=None, forbidden=None,semantic_probability=None):
    """Fit geometry, rejecting an axis pass that breaks prior connected wall runs.

    The reference is the same dense-face fit with angular regularization off.
    This conservative guard can retain the entire reference fit; it never joins
    pieces by filling unobserved material or bridging an excluded aperture.
    """
    cfg=config or Config()
    confidence=semantic_probability if semantic_probability is not None else probability
    fit_mask=mask
    if cfg.structure_refine and cfg.structure_filter and confidence is not None and barrier is not None and np.any(barrier):
        from .wall_runs import filter_detached
        fit_mask,_,_=filter_detached(mask,confidence,barrier,pixel_m,cfg.detached_probability)
    result=_decompose(fit_mask,pixel_m,probability,density,cfg,barrier,forbidden)
    if cfg.axis_weight and cfg.axis_preserve_connectivity:
        reference=_decompose(fit_mask,pixel_m,probability,density,replace(cfg,axis_weight=0),barrier,forbidden)
        breaks=_connectivity_breaks(reference['mask'],result['mask'])
        if breaks:
            reference['config']=asdict(cfg)
            reference['axis_regularization']=dict(accepted=False,reason='previous_wall_component_split_or_lost',breaks=breaks)
            result=reference
        else:result['axis_regularization']=dict(accepted=True,breaks=[])
    if cfg.structure_refine and barrier is not None and np.any(barrier):
        from .wall_runs import refine_structure
        result=refine_structure(result,mask,barrier,pixel_m,confidence,forbidden,cfg)
    if cfg.simplify_wall_runs and barrier is not None and np.any(barrier):
        from .wall_simplify import simplify_result
        result=simplify_result(result,mask,barrier,pixel_m,confidence,forbidden,cfg)
    if cfg.merge_wall_runs and barrier is not None:
        from .wall_merge import merge_result
        result=merge_result(result,pixel_m)
    return result


def _decompose(mask, pixel_m=.02, probability=None, density=None, config=None,
              barrier=None, forbidden=None):
    cfg=config or Config();mask=np.asarray(mask,bool)
    if mask.ndim!=2 or pixel_m<=0:raise ValueError('Expected 2D mask and positive pixel size')
    p=mask.astype(np.float32) if probability is None else np.asarray(probability,np.float32)
    if p.shape!=mask.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1.0001)):
        raise ValueError('probability must match mask and lie in [0,1]')
    if not (0<=cfg.spill_limit<1 and cfg.spill_weight>0 and cfg.min_width_m>0 and
            cfg.max_width_m>=cfg.min_width_m and cfg.max_primitives>0 and 0<=cfg.density_quantile<=1 and
            cfg.dense_count_2cm>0 and cfg.density_weight>=0 and
            0<cfg.coverage_stop<=1 and 0<cfg.proposal_fill<=1 and
            cfg.overlap_weight>=0 and 0<=cfg.overlap_free_fraction<1 and cfg.axis_weight>=0 and
            0<cfg.axis_max_degrees<45 and cfg.axis_scan_min_length_m>0 and cfg.axis_neighbour_m>=0 and
            cfg.run_min_m>0 and cfg.run_face_tolerance_m>0 and cfg.run_max_width_m>=cfg.min_width_m and
            0<=cfg.detached_probability<=1 and cfg.simplify_max_shift_m>0 and cfg.simplify_min_length_m>0 and
            0<cfg.simplify_min_face_coverage<=1 and cfg.simplify_hole_area_m2>=0 and cfg.simplify_seam_width_m>=0):raise ValueError('Invalid decomposition config')
    if forbidden is not None:
        forbidden=np.asarray(forbidden,bool)
        if forbidden.shape!=mask.shape or (mask&forbidden).any():raise ValueError('Forbidden regions must match and be excluded from target mask')
    if barrier is not None:
        barrier=np.asarray(barrier,bool)
        if barrier.shape!=mask.shape:raise ValueError('Barrier must match target mask')
    distance_to_mask=None if barrier is None else cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)*pixel_m
    dense,density_info=dense_evidence(density,p,pixel_m,cfg)
    guidance=None;face_claimed=None
    if barrier is not None and (cfg.scan_face_reward or cfg.scan_overshoot_weight):
        inside_distance=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)*pixel_m
        guidance=barrier&(np.where(mask,inside_distance,distance_to_mask)<=cfg.scan_face_scope_m)
        if forbidden is not None:guidance&=~forbidden
        face_claimed=np.zeros(mask.shape,bool)
        del inside_distance
    covered=np.zeros(mask.shape,bool);rect_covered=np.zeros(mask.shape,bool);primitives=[]
    scan_lines=_scan_axis_lines(barrier,pixel_m,cfg)
    penalty=cfg.complexity_m2/pixel_m**2;gain_floor=cfg.min_gain_m2/pixel_m**2
    # Face scoring is computed on the boundary, never rewards occupying the interior.
    face_radius=max(1,int(round(cfg.face_tolerance_m/pixel_m)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*face_radius+1,)*2)
    def score(candidate, strict=True):
        # Work only in a primitive's bounding box: full-floor arrays here would
        # turn candidate refinement into an unnecessary memory-bandwidth bottleneck.
        bounds=np.asarray(candidate['polygon'])
        if np.any(bounds.min(0)<-.5) or np.any(bounds.max(0)>np.asarray(mask.shape[::-1])-.5):
            return -np.inf,None,None
        margin=1 if barrier is None else max(2,int(np.ceil(max(cfg.barrier_contact_m,cfg.scan_face_range_m)/pixel_m))+2)
        lo=np.maximum(0,np.floor(bounds.min(0)-margin).astype(int))
        hi=np.minimum(mask.shape[::-1],np.ceil(bounds.max(0)+margin+1).astype(int))
        if np.any(hi<=lo):return -np.inf,None,None
        sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        local=dict(candidate,polygon=(bounds-lo).tolist(),center=(np.asarray(candidate['center'])-lo).tolist())
        footprint=rasterize(local,(hi[1]-lo[1],hi[0]-lo[0]));area=int(footprint.sum())
        if area==0:return -np.inf,None,None
        target=mask[sl];prob=p[sl]
        spill_pixels=footprint&~target
        spill=int(spill_pixels.sum())/area
        limit=cfg.spill_limit if barrier is None else cfg.clear_spill_limit
        if spill>limit:return -np.inf,None,None
        if forbidden is not None and (footprint&forbidden[sl]).any():return -np.inf,None,None
        new=footprint&target&~covered[sl];gain=float(prob[new].sum())
        if int(new.sum())<gain_floor:return -np.inf,None,None
        # Distance to the candidate face allows contact with scan surfaces,
        # while rejecting spill that extends THROUGH a reliable dense surface.
        interior=cv2.distanceTransform(np.pad(footprint.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]
        dense_touch=0;crossing_pixels=0
        if barrier is not None:
            if (spill_pixels&(distance_to_mask[sl]>cfg.max_clear_extension_m)).any():return -np.inf,None,None
            touched=spill_pixels&barrier[sl]
            if strict:
                union=footprint|covered[sl]
                union_distance=cv2.distanceTransform(np.pad(union.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
                crossing_pixels=int(((union&~target&barrier[sl])&(union_distance>cfg.barrier_contact_m)).sum())
            else:
                crossing_pixels=int((touched&(interior*pixel_m>cfg.barrier_contact_m)).sum())
            if strict and crossing_pixels:return -np.inf,None,None
            dense_touch=int(touched.sum())
            cost=cfg.clear_spill_weight*float(((1-prob)*spill_pixels*~barrier[sl]).sum())
            cost+=cfg.spill_weight*float(((1-prob)*touched).sum())
        else:
            cost=cfg.spill_weight*float(((1-prob)*spill_pixels).sum())
        overlap_pixels=int((footprint&rect_covered[sl]).sum()) if candidate['kind']=='rectangle' else 0
        overlap_excess=max(0.,overlap_pixels-cfg.overlap_free_fraction*area)
        overlap_cost=cfg.overlap_weight*overlap_excess
        base=gain-cost-penalty-overlap_cost
        if base<=0:return -np.inf,None,None
        if crossing_pixels:
            # Infeasible initial proposals can be adjusted before the hard check.
            # Bounded penalty retains a useful gradient instead of dropping them.
            base-=min(.9*base,8.0*crossing_pixels)+.01*crossing_pixels
        boundary=footprint&(interior<=face_radius)
        bonus=cfg.density_weight*min(gain,float((dense[sl]*boundary*new).sum()))
        face_bonus=overshoot=0.
        if guidance is not None:
            face_bonus,overshoot,_,_=_scan_face_terms(candidate,guidance[sl],face_claimed[sl],lo,pixel_m,cfg)
            face_bonus=min(.5*gain,face_bonus)
        axis_cost=0.
        reference=candidate.get('_axis_reference')
        if reference is not None:
            delta=(candidate['angle']-reference['target']+np.pi/4)%(np.pi/2)-np.pi/4
            axis_cost=cfg.axis_weight*gain*reference['strength']*(delta/np.radians(cfg.axis_max_degrees))**2
        total=base+bonus+face_bonus-overshoot-axis_cost
        if strict and total<=0:return -np.inf,None,None
        return total+1e-5*candidate.get('width',0),(sl,footprint),dict(
            gain=gain,spill_fraction=spill,rectangle_overlap_fraction=overlap_pixels/area,rectangle_overlap_cost=overlap_cost,dense_surface_contact_pixels=dense_touch,barrier_crossing_pixels=crossing_pixels,cloud_bonus=bonus,scan_face_bonus=face_bonus,scan_overshoot_cost=overshoot,axis_cost=axis_cost,axis_reference=reference,score=total)
    def refine(candidate):
        reference=_axis_reference(candidate,scan_lines,primitives,pixel_m,cfg)
        guards=_contact_guards(candidate,primitives,mask.shape) if reference else []
        if reference:candidate=dict(candidate,_axis_reference=reference)
        best=candidate;s,_,best_detail=score(best,strict=False)
        best_cross=best_detail['barrier_crossing_pixels'] if best_detail else float('inf')
        if reference:
            trial=rectangle(best['center'],best['length'],best['width'],reference['target'])
            trial['_axis_reference']=reference
            ns,fp,nd=score(trial,strict=False)
            if nd is not None and _keeps_contacts(fp,guards):
                crosses=nd['barrier_crossing_pixels']
                if crosses<best_cross or (crosses==best_cross and ns>s+1e-6):best,s,best_cross=trial,ns,crosses
        # Native pixel coordinate descent, same objective and constraints as selection.
        for step in ((4.,2.,1.,.5) if barrier is not None else (2.,1.,.5)):
            for _ in range(1 if barrier is not None else 2):
                changed=False
                fields=('cx','cy','length','width','angle') if best['kind']=='rectangle' else ('radius','width','start','sweep')
                for key in fields:
                    for sign in (-1,1):
                        delta=sign*step
                        if best['kind']=='rectangle':
                            c=np.array(best['center']);length=best['length'];width=best['width'];angle=best['angle']
                            if key=='cx':c[0]+=delta
                            elif key=='cy':c[1]+=delta
                            elif key=='length':length+=2*delta
                            elif key=='angle':angle+=np.radians(delta*.125)
                            else:width+=2*delta
                            if length<cfg.min_length_m/pixel_m or not cfg.min_width_m/pixel_m<=width<=cfg.max_width_m/pixel_m:continue
                            trial=rectangle(c,length,width,angle)
                        else:
                            v={k:best[k] for k in ('radius','width','start','sweep')}
                            v[key]+=delta/best['radius'] if key in ('start','sweep') else delta
                            if v['radius']<=v['width']/2 or not cfg.min_width_m/pixel_m<=v['width']<=cfg.max_width_m/pixel_m or not 0<v['sweep']<2*np.pi:continue
                            trial=arc(best['center'],**v)
                        if reference:trial['_axis_reference']=reference
                        ns,fp,nd=score(trial,strict=False)
                        if nd is None:continue
                        if guards and not _keeps_contacts(fp,guards):continue
                        crosses=nd['barrier_crossing_pixels']
                        if crosses<best_cross or (crosses==best_cross and ns>s+1e-6):
                            best,s,best_cross,changed=trial,ns,crosses,True
                if not changed:break
        return best
    total_mask=int(mask.sum());covered_target=0
    def filled():
        return total_mask==0 or covered_target>=cfg.coverage_stop*total_mask
    # Refresh proposals on residuals so width changes survive early placements.
    for round_id in range(3):
        remaining=mask&~covered
        if filled() or remaining.sum()<gain_floor:break
        scale=max(1,int(round(cfg.proposal_pixel_m/pixel_m)))
        proposal_mask=remaining
        # Pad to an exact multiple before pooling so proposal coordinates have
        # a fixed world scale even when the native dimensions are not divisible.
        if scale>1:
            padded=np.pad(remaining,((0,(-mask.shape[0])%scale),(0,(-mask.shape[1])%scale)))
            proposal_mask=padded.reshape(padded.shape[0]//scale,scale,padded.shape[1]//scale,scale).mean((1,3))>=.5
        arc_mask=mask
        if scale>1:
            padded=np.pad(mask,((0,(-mask.shape[0])%scale),(0,(-mask.shape[1])%scale)))
            arc_mask=padded.reshape(padded.shape[0]//scale,scale,padded.shape[1]//scale,scale).mean((1,3))>=.5
        candidates=_rectangles(proposal_mask,pixel_m*scale,cfg)+(_arcs(arc_mask,pixel_m*scale,cfg) if round_id==0 else [])
        if scale>1:
            candidates=[_scale_primitive(c,scale) for c in candidates]
        # Lazy greedy: coverage only reduces scores, so cached values are upper bounds.
        import heapq
        queue=[];rescue=[]
        for i,c in enumerate(candidates):
            s,_,detail=score(c,strict=False)
            if not np.isfinite(s):continue
            entry=(-s,i,c)
            if detail['barrier_crossing_pixels'] or s<=0:
                rescue.append(entry)
            else:heapq.heappush(queue,entry)
        # Rescue only the best coarse infeasible proposals. Trying every
        # rejected shape at native resolution is needlessly expensive.
        for entry in heapq.nsmallest(cfg.constraint_refine_limit,rescue):heapq.heappush(queue,entry)
        rescued=0
        while queue and len(primitives)<cfg.max_primitives and not filled():
            _,i,c=heapq.heappop(queue);s,footprint,detail=score(c,strict=False)
            if not np.isfinite(s):continue
            if queue and s < -queue[0][0]-1e-6:
                heapq.heappush(queue,(-s,i,c));continue
            if detail['barrier_crossing_pixels'] or s<=0:
                if rescued>=cfg.constraint_refine_limit:continue
                rescued+=1
            c=refine(c);s,footprint,detail=score(c)
            if not np.isfinite(s):continue
            c['fit']=detail;c['id']=len(primitives);primitives.append(c)
            sl,patch=footprint
            covered_target+=int((patch&mask[sl]&~covered[sl]).sum());covered[sl]|=patch
            if c['kind']=='rectangle':rect_covered[sl]|=patch
            if guidance is not None:
                lo=np.array([sl[1].start,sl[0].start])
                _,_,ys,xs=_scan_face_terms(c,guidance[sl],face_claimed[sl],lo,pixel_m,cfg)
                face_claimed[ys,xs]=True
        if len(primitives)>=cfg.max_primitives:break
    # Preserve small irregular junctions only if touching two accepted primitives.
    remaining=(mask&~covered).astype(np.uint8)
    n,labels,stats,_=cv2.connectedComponentsWithStats(remaining,8)
    for label in range(1,n):
        if filled():break
        area=stats[label,cv2.CC_STAT_AREA]
        if area*pixel_m**2>cfg.junction_area_m2 or area<2:continue
        x,y,w,h=stats[label,:4]
        margin=1 if barrier is None else max(2,int(np.ceil(cfg.barrier_contact_m/pixel_m))+2)
        lo=np.array([max(0,x-margin),max(0,y-margin)])
        hi=np.minimum(mask.shape[::-1],[x+w+margin,y+h+margin])
        sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));local_shape=(hi[1]-lo[1],hi[0]-lo[0])
        region=labels[sl]==label;touch=cv2.dilate(region.astype(np.uint8),np.ones((3,3),np.uint8))>0
        parents=[]
        for c in primitives:
            if c['kind']=='junction':continue
            bounds=np.asarray(c['polygon'])
            if np.any(bounds.max(0)<lo) or np.any(bounds.min(0)>hi):continue
            if (_local_mask(c,lo,local_shape)&touch).any():parents.append(c['id'])
        if len(parents)<2 or len(primitives)>=cfg.max_primitives:continue
        contours,hierarchy=cv2.findContours(region.astype(np.uint8),cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:continue
        for j,contour in enumerate(contours):
            if hierarchy[0,j,3]!=-1 or len(contour)<3:continue
            holes=[(c[:,0]+lo).tolist() for k,c in enumerate(contours) if hierarchy[0,k,3]==j]
            c=dict(kind='junction',polygon=(contour[:,0]+lo).tolist(),holes=holes,parents=parents,id=len(primitives))
            footprint=_local_mask(c,lo,local_shape)
            if (footprint&~mask[sl]).any():continue
            if barrier is not None:
                union=footprint|covered[sl]
                depth=cv2.distanceTransform(np.pad(union.astype(np.uint8),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
                if (union&~mask[sl]&barrier[sl]&(depth>cfg.barrier_contact_m)).any():continue
            primitives.append(c);covered_target+=int((footprint&mask[sl]&~covered[sl]).sum());covered[sl]|=footprint
    selected_primitives=len(primitives)
    coverage_search=dict(requested=cfg.coverage_stop,achieved=covered_target/max(1,total_mask),
        target_met=filled(),remaining_pixels=total_mask-covered_target,
        stop_reason=('empty_target' if total_mask==0 else 'target_reached' if filled() else
                     'primitive_limit' if selected_primitives>=cfg.max_primitives else
                     'residual_below_minimum_gain' if total_mask-covered_target<gain_floor else
                     'proposal_rounds_exhausted_under_current_constraints'),
        rounds_attempted=round_id+1,selected_primitives=selected_primitives,
        min_new_area_m2=cfg.min_gain_m2,proposal_pixel_m=cfg.proposal_pixel_m)
    primitives=_consolidate(primitives,mask.shape,cfg,pixel_m)
    return dict(coverage_search=coverage_search,primitives=primitives,contacts=_topology(primitives,mask.shape),mask=covered,dense=dense,metrics=mask_metrics(mask,covered),
                density=density_info,rectangle_overlap=rectangle_overlap_metrics(primitives,mask.shape,pixel_m),spill_policy='cheap_no_observed_barrier' if barrier is not None else 'uniform',
                forbidden_pixels=int(forbidden.sum()) if forbidden is not None else 0,config=asdict(cfg),pixel_m=float(pixel_m),
                thickness_semantics='mask footprint width; exterior training bands are not physical wall thickness')


def mask_metrics(target, predicted):
    target=np.asarray(target,bool);predicted=np.asarray(predicted,bool)
    tp=int((target&predicted).sum());fp=int((~target&predicted).sum());fn=int((target&~predicted).sum())
    return dict(iou=tp/max(1,tp+fp+fn),recall=tp/max(1,tp+fn),precision=tp/max(1,tp+fp),
                missed_pixels=fn,spill_pixels=fp,target_pixels=int(target.sum()))


def _consolidate(primitives, shape, cfg, px):
    """Merge only when the replacement loses zero target footprint pixels and
    introduces zero new footprint pixels. This is exact representation cleanup,
    not another gap-closing heuristic. Width steps remain separate.
    """
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(primitives):
            if a['kind']!='rectangle':continue
            for j in range(i+1,len(primitives)):
                b=primitives[j]
                if b['kind']!='rectangle':continue
                da=abs((a['angle']-b['angle']+np.pi/2)%np.pi-np.pi/2)
                if da>np.radians(1) or abs(a['width']-b['width'])>1.01:continue
                u=np.array([np.cos(a['angle']),np.sin(a['angle'])]);v=np.array([-u[1],u[0]])
                d=np.asarray(b['center'])-a['center']
                if abs(d@v)>1 or abs(d@u)>(a['length']+b['length'])/2+1:continue
                corners=np.vstack([a['polygon'],b['polygon']]);along=corners@u;cross=corners@v
                c=rectangle((along.min()+along.max())/2*u+(cross.min()+cross.max())/2*v,
                            np.ptp(along),np.ptp(cross),a['angle'])
                if c['width']>cfg.max_width_m/px:continue
                lo,hi=_bounds(c,shape)
                local_shape=(hi[1]-lo[1],hi[0]-lo[0])
                old=_local_mask(a,lo,local_shape)|_local_mask(b,lo,local_shape);new=_local_mask(c,lo,local_shape)
                if not np.array_equal(old,new):continue
                c['merged_from']=[a.get('id'),b.get('id')]
                primitives[i]=c;primitives.pop(j);changed=True;break
            if changed:break
    for i,c in enumerate(primitives):c['id']=i
    # Parent ids from before consolidation are replaced by the final topology.
    for c in primitives:c.pop('parents',None)
    return primitives


def _topology(primitives, shape):
    """Region contacts, without moving any wall face or forcing axis intersections."""
    bounds=[(np.min(c['polygon'],axis=0),np.max(c['polygon'],axis=0)) for c in primitives]
    contacts=[]
    for i,a in enumerate(primitives):
        lo,hi=bounds[i]
        for j in range(i+1,len(primitives)):
            blo,bhi=bounds[j]
            if np.any(lo>bhi+1) or np.any(blo>hi+1):continue
            xy0=np.maximum(0,np.floor(np.minimum(lo,blo)-1).astype(int))
            xy1=np.minimum(shape[::-1],np.ceil(np.maximum(hi,bhi)+2).astype(int))
            sl=(slice(xy0[1],xy1[1]),slice(xy0[0],xy1[0]))
            local_shape=(xy1[1]-xy0[1],xy1[0]-xy0[0])
            aa=_local_mask(a,xy0,local_shape);bb=_local_mask(primitives[j],xy0,local_shape);overlap=int((aa&bb).sum())
            if overlap or (cv2.dilate(aa.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)&bb).any():
                contacts.append(dict(a=i,b=j,relation='overlap' if overlap else 'touch',overlap_pixels=overlap))
    return contacts


def _scale_primitive(c,scale):
    center=(np.asarray(c['center'])+.5)*scale-.5
    if c['kind']=='rectangle':return rectangle(center,c['length']*scale,c['width']*scale,c['angle'])
    return arc(center,c['radius']*scale,c['width']*scale,c['start'],c['sweep'])


def _bounds(c,shape):
    points=np.asarray(c['polygon'])
    return np.maximum(0,np.floor(points.min(0)-1).astype(int)),np.minimum(shape[::-1],np.ceil(points.max(0)+2).astype(int))


def _local_mask(c,lo,shape):
    c=dict(c,polygon=(np.asarray(c['polygon'])-lo).tolist())
    if 'center' in c:c['center']=(np.asarray(c['center'])-lo).tolist()
    if 'holes' in c:c['holes']=[(np.asarray(h)-lo).tolist() for h in c['holes']]
    if c['kind']=='wall_run':
        c['parts']=[dict(p,polygon=(np.asarray(p['polygon'])-lo).tolist(),center=(np.asarray(p['center'])-lo).tolist()) for p in c['parts']]
    return rasterize(c,shape)


def _overlaps_raster(c,raster):
    lo,hi=_bounds(c,raster.shape)
    if np.any(hi<=lo):return False
    sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
    if not raster[sl].any():return False
    return bool((_local_mask(c,lo,(hi[1]-lo[1],hi[0]-lo[0]))&raster[sl]).any())


def rectangle_overlap_metrics(primitives,shape,pixel_m,return_counts=False):
    primitives=[part for c in primitives for part in (c['parts'] if c['kind']=='wall_run' else [c])]
    counts=np.zeros(shape,np.uint16);summed=0
    for c in primitives:
        if c['kind']!='rectangle':continue
        lo,hi=_bounds(c,shape)
        if np.any(hi<=lo):continue
        patch=_local_mask(c,lo,(hi[1]-lo[1],hi[0]-lo[0]));sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        counts[sl]+=patch;summed+=int(patch.sum())
    union=int((counts>0).sum());multiple=int((counts>1).sum())
    metrics=dict(rectangle_footprint_m2=summed*pixel_m**2,rectangle_union_m2=union*pixel_m**2,
                 redundant_area_m2=(summed-union)*pixel_m**2,multiply_covered_m2=multiple*pixel_m**2,
                 redundant_fraction=(summed-union)/max(1,summed),max_multiplicity=int(counts.max()) if counts.size else 0)
    return (metrics,counts) if return_counts else metrics


def _scan_face_terms(candidate,guide,claimed,lo,pixel_m,cfg):
    """Attraction to actual primitive faces, plus one-sided penetration cost.

    Evaluate only reliable scan pixels; do not reward occupying wall interiors.
    Already explained scan pixels cannot repeatedly earn attraction rewards.
    """
    yy,xx=np.nonzero(guide)
    if not len(xx):return 0.,0.,yy,xx
    x=xx+lo[0];y=yy+lo[1];c=candidate
    dx=x-c['center'][0];dy=y-c['center'][1]
    if c['kind']=='rectangle':
        u=np.array([np.cos(c['angle']),np.sin(c['angle'])])
        a=np.abs(dx*u[0]+dy*u[1])-c['length']/2
        b=np.abs(-dx*u[1]+dy*u[0])-c['width']/2
        signed=-(np.hypot(np.maximum(a,0),np.maximum(b,0))+np.minimum(np.maximum(a,b),0))*pixel_m
    else:
        radius=np.hypot(dx,dy);angle=(np.arctan2(dy,dx)-c['start'])%(2*np.pi)
        in_angle=angle<=c['sweep'];radial=c['width']/2-np.abs(radius-c['radius'])
        distance=np.where(in_angle,np.abs(radial),np.inf)
        for end in [c['start'],c['start']+c['sweep']]:
            u=np.array([np.cos(end),np.sin(end)])
            projection=np.clip(dx*u[0]+dy*u[1],c['radius']-c['width']/2,c['radius']+c['width']/2)
            distance=np.minimum(distance,np.hypot(dx-projection*u[0],dy-projection*u[1]))
        signed=distance*np.where(in_angle&(radial>=0),1.,-1.)*pixel_m
    sigma=cfg.scan_face_sigma_m
    weights=np.maximum(0.,1-np.abs(signed)/cfg.scan_face_range_m)
    reward=cfg.scan_face_reward*float(weights[~claimed[yy,xx]].sum())
    depth=np.clip((signed-cfg.barrier_contact_m)/sigma,0,10)
    penalty=cfg.scan_overshoot_weight*float(depth.sum())
    matched=np.abs(signed)<=sigma
    return reward,penalty,y[matched],x[matched]
