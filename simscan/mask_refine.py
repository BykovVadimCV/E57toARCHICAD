"""Confidence-gated, seeded graph cut on a fine raster. No vector geometry.

Scan surfaces lower boundary costs, not foreground costs: empty wall interiors
are not treated as background. Sparse areas retain the exact mask. At native
resolution a separate whole-frame confidence check can quarantine weak detached
predictions without scan-face support; attached occluded material is retained.
"""
from dataclasses import dataclass,asdict
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow,breadth_first_order

@dataclass(frozen=True)
class Config:
    band_m: float=.12
    influence_m: float=.07
    dense_count_2cm: float=200.
    vertical_min: float=.6
    min_face_m: float=.06
    smoothness: float=6.
    boundary_attraction: float=.95
    prior_weight: float=.30
    prior_scale_m: float=.025
    preserve_topology: bool=True
    boundary_locality_m: float | None=None
    boundary_ownership: bool=False
    filter_detached: bool=True
    detached_probability: float=.75


def refine(mask,probability,scan,pixel_m=.01,config=None,forbidden=None):
    cfg=config or Config();mask=np.asarray(mask,bool);p=np.asarray(probability,float);scan=np.asarray(scan,float)
    if mask.ndim!=2 or p.shape!=mask.shape or scan.shape!=(8,*mask.shape):raise ValueError('Expected mask, probability and fine [8,H,W] scan')
    if pixel_m<=0 or not np.isfinite(scan).all() or not np.isfinite(p).all():raise ValueError('Invalid scale or nonfinite input')
    if np.any((scan<0)|(scan>1)) or np.any((p<0)|(p>1)):raise ValueError('Inputs must be in [0,1]')
    if cfg.band_m<=0 or cfg.influence_m<=0 or cfg.dense_count_2cm<=0 or cfg.smoothness<0 or not 0<=cfg.boundary_attraction<=1:raise ValueError('Invalid config')
    def result(out,guide,editable,info):
        added=out&~mask;removed=mask&~out
        return dict(mask=out,guide=guide,editable=editable,added=added,removed=removed,
                    metrics=dict(added_pixels=int(added.sum()),removed_pixels=int(removed.sum()),
                                 changed_area_m2=float((added.sum()+removed.sum())*pixel_m**2),
                                 original_components=int(ndi.label(mask,np.ones((3,3)))[1]),refined_components=int(ndi.label(out,np.ones((3,3)))[1]),**info),config=asdict(cfg))
    if forbidden is not None:
        forbidden=np.asarray(forbidden,bool)
        if forbidden.shape!=mask.shape or (mask&forbidden).any():raise ValueError('Forbidden regions must match and be absent from mask')
    blank=np.zeros_like(mask)
    if not mask.any() or mask.all():return result(mask.copy(),blank,blank,{'active_face_pixels':0,'graph_nodes':0})
    inside=ndi.distance_transform_edt(mask);outside=ndi.distance_transform_edt(~mask)
    signed=(inside-outside)*pixel_m
    band=np.abs(signed)<=cfg.band_m
    counts=np.expm1(scan[5]*np.log1p(4000))*(.02/pixel_m)**2
    bands=(scan[:3]>.08).sum(0)
    face=(counts>=cfg.dense_count_2cm)&(scan[6]>=cfg.vertical_min)&(bands>=2)&band
    # Small disconnected dense blobs do not count as reliable wall faces.
    labels,n=ndi.label(face,np.ones((3,3)))
    sizes=np.bincount(labels.ravel());keep=sizes>=max(3,int(round(cfg.min_face_m/pixel_m)));keep[0]=False
    face=keep[labels]
    if not face.any():return result(mask.copy(),face,blank,{'active_face_pixels':0,'graph_nodes':0})
    face_distance=ndi.distance_transform_edt(~face)*pixel_m
    editable=band&(face_distance<=cfg.influence_m)
    if cfg.boundary_locality_m is not None:
        # A return belongs to its nearest existing face. Do not let evidence
        # on one side of a wall remodel the unobserved opposite side.
        boundary_pixels=mask&~ndi.binary_erosion(mask)
        _, nearest=ndi.distance_transform_edt(~boundary_pixels,return_indices=True)
        supported_boundary=np.zeros_like(mask)
        supported_boundary[nearest[0][face],nearest[1][face]]=True
        editable &= ndi.distance_transform_edt(~supported_boundary)*pixel_m<=cfg.boundary_locality_m
        if cfg.boundary_ownership:
            editable &= supported_boundary[nearest[0],nearest[1]]
    # Keep a central foreground seed in thin sections without eroding them away.
    ridge=mask&(inside>=ndi.maximum_filter(inside,size=3)-1e-6)&(p>=.65)
    editable&=~ridge
    if forbidden is not None:editable&=~forbidden
    ids=np.full(mask.shape,-1,np.int32);ids[editable]=np.arange(editable.sum());n=int(editable.sum())
    if n==0:return result(mask.copy(),face,editable,{'active_face_pixels':int(face.sum()),'graph_nodes':0})
    prior=.7/(1+np.exp(-np.clip(signed/cfg.prior_scale_m,-30,30)))+.3*p
    prior=np.clip(prior,.001,.999)
    cost_fg=-np.log(prior[editable])*cfg.prior_weight
    cost_bg=-np.log1p(-prior[editable])*cfg.prior_weight
    boundary=cfg.smoothness*(1-cfg.boundary_attraction*np.exp(-.5*(face_distance/max(pixel_m,.01))**2))
    rows=[];cols=[];data=[]
    for dy,dx in [(0,1),(1,0)]:
        aa=(slice(0,mask.shape[0]-dy or None),slice(0,mask.shape[1]-dx or None))
        bb=(slice(dy,None),slice(dx,None))
        u=ids[aa];v=ids[bb];weight=np.minimum(boundary[aa],boundary[bb])
        both=(u>=0)&(v>=0)
        rows.extend([u[both],v[both]]);cols.extend([v[both],u[both]]);data.extend([weight[both],weight[both]])
        for a,b,fixed in [(u,v,mask[bb]),(v,u,mask[aa])]:
            at=(a>=0)&(b<0)
            np.add.at(cost_fg,a[at&~fixed],weight[at&~fixed])
            np.add.at(cost_bg,a[at&fixed],weight[at&fixed])
    # Source side = foreground. s->node is paid when node is background.
    rows.extend([np.full(n,n),np.arange(n)]);cols.extend([np.arange(n),np.full(n,n+1)]);data.extend([cost_bg,cost_fg])
    capacity=np.maximum(0,np.rint(np.concatenate(data)*1000)).astype(np.int64)
    graph=coo_matrix((capacity,(np.concatenate(rows),np.concatenate(cols))),shape=(n+2,n+2)).tocsr()
    flow=maximum_flow(graph,n,n+1)
    residual=(graph-flow.flow).tocsr();residual.data=(residual.data>0).astype(np.int8);residual.eliminate_zeros()
    reachable=breadth_first_order(residual,n,directed=True,return_predecessors=False)
    out=mask.copy();selected=np.zeros(n,bool);selected[reachable[reachable<n]]=True;out[editable]=selected
    # Refinement cannot establish an entirely new detached foreground component.
    labs,num=ndi.label(out,np.ones((3,3)));anchored=np.bincount(labs[mask],minlength=num+1)>0;anchored[0]=False
    out=anchored[labs]
    # Pixels outside the edit band are an exact hard constraint, including fragments.
    out[~editable]=mask[~editable]
    proposed=int(np.count_nonzero(out!=mask))
    if cfg.preserve_topology:
        out=_preserve_topology(mask,out,face_distance)
    return result(out,face,editable,{'proposed_changes':proposed,'topology_blocked_changes':proposed-int(np.count_nonzero(out!=mask)),
                                   'active_face_pixels':int(face.sum()),'graph_nodes':n,
                                   'max_boundary_displacement_m':cfg.band_m})


def _preserve_topology(original, target, distance):
    """Accept only simple pixel flips (8-connected foreground/4-connected background).

    Local tests are conservative; no new components, severed links or holes.
    Iterate to allow adjacent safe changes to follow previously accepted ones.
    """
    offsets=[(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    lut=np.zeros(256,bool)
    for value in range(256):
        fg=np.zeros((3,3),bool)
        for bit,(dy,dx) in enumerate(offsets):fg[dy+1,dx+1]=bool(value&(1<<bit))
        bg=~fg;bg[1,1]=False
        lut[value]=(ndi.label(fg,np.ones((3,3)))[1]==1 and ndi.label(bg)[1]==1
                    and any(bg[dy+1,dx+1] for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]))
    out=np.pad(original,1)
    ys,xs=np.where(original!=target);order=np.argsort(distance[ys,xs],kind='stable')
    pending=list(order)
    for _ in range(4):
        next_pending=[];accepted=0
        for i in pending:
            y,x=int(ys[i])+1,int(xs[i])+1
            bits=sum(int(out[y+dy,x+dx])<<bit for bit,(dy,dx) in enumerate(offsets))
            if lut[bits]:out[y,x]=target[y-1,x-1];accepted+=1
            else:next_pending.append(i)
        pending=next_pending
        if not accepted:break
    return out[1:-1,1:-1]


def refine_high_resolution(mask10, probability20, scan10, counts, pixel_m=.0025,
                           tile_pixels=768, config=None, log=lambda *_:None, forbidden=None):
    """Progressive tiled refinement against native fine XY return counts.

    Verticality/height support remains at 10mm, explicitly nearest-expanded.
    A halo protects tile cores; topology is enforced once after assembly. This
    bounds graph memory and is not an exact global graph-cut optimum.
    """
    import cv2
    from dataclasses import replace
    counts=np.asarray(counts)
    if counts.ndim!=2 or pixel_m<=0 or tile_pixels<32:raise ValueError('Invalid output grid/tile')
    h,w=counts.shape;mask10=np.asarray(mask10,bool);scan10=np.asarray(scan10)
    factor=.01/pixel_m
    if not np.isclose(factor,round(factor)) or counts.shape!=(int(mask10.shape[0]*factor),int(mask10.shape[1]*factor)):
        raise ValueError('Fine counts must cover the exact 10mm frame at integer scale')
    if scan10.shape!=(8,*mask10.shape) or np.any(counts<0):raise ValueError('Invalid scan/counts')
    # Energy scaling preserves the unary-area / boundary-length ratio in metres.
    cfg=config or replace(Config(),smoothness=Config().smoothness*.01/pixel_m,preserve_topology=False,boundary_locality_m=.02)
    local_cfg=replace(cfg,preserve_topology=False)
    original=cv2.resize(mask10.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    if forbidden is not None:
        forbidden=np.asarray(forbidden,bool)
        if forbidden.shape!=counts.shape:raise ValueError('Forbidden regions must match fine counts')
        original&=~forbidden
    out=original.copy();faces=np.zeros((h,w),bool);editable=np.zeros((h,w),bool)
    halo=int(np.ceil((cfg.band_m+cfg.influence_m+.03)/pixel_m))
    # Mapping native cell centres avoids independently rescaling cropped tiles.
    def sample(a,y0,y1,x0,x1,scale,linear=False):
        xs=(np.arange(x0,x1,dtype=np.float32)+.5)/scale-.5
        ys=(np.arange(y0,y1,dtype=np.float32)+.5)/scale-.5
        xx,yy=np.meshgrid(xs,ys)
        return cv2.remap(a.astype(np.float32),xx,yy,cv2.INTER_LINEAR if linear else cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_REPLICATE)
    graphs=0;nodes=0
    for y in range(0,h,tile_pixels):
        for x in range(0,w,tile_pixels):
            y1=min(h,y+tile_pixels);x1=min(w,x+tile_pixels)
            ya=max(0,y-halo);yb=min(h,y1+halo);xa=max(0,x-halo);xb=min(w,x1+halo)
            patch=original[ya:yb,xa:xb]
            if not patch.any() or patch.all():continue
            core=(slice(y-ya,y1-ya),slice(x-xa,x1-xa))
            scan=np.zeros((8,yb-ya,xb-xa),np.float32)
            for ch in [0,1,2,6]:scan[ch]=sample(scan10[ch],ya,yb,xa,xb,factor)
            scan[5]=np.clip(np.log1p(counts[ya:yb,xa:xb])/np.log1p(4000),0,1)
            p=sample(probability20,ya,yb,xa,xb,.02/pixel_m,linear=True)
            r=refine(patch,np.clip(p,0,1),scan,pixel_m,local_cfg,
                     forbidden=forbidden[ya:yb,xa:xb] if forbidden is not None else None)
            out[y:y1,x:x1]=r['mask'][core];faces[y:y1,x:x1]=r['guide'][core];editable[y:y1,x:x1]=r['editable'][core]
            graphs+=1;nodes+=r['metrics']['graph_nodes']
        log(f'fine row {y1}/{h}, graph tiles {graphs}')
    proposed=int(np.count_nonzero(out!=original))
    # Ordering by local count confidence uses uint8 storage instead of a full
    # floating-point distance image. It does not alter the topology criterion.
    priority=(255-np.minimum(counts,255)).astype(np.uint8)
    out=_preserve_topology(original,out,priority)
    out[~editable]=original[~editable]
    topology_changed=int(np.count_nonzero(out!=original));quarantine=[];confidence_removed=0
    if cfg.filter_detached and faces.any():
        # Run once on the complete frame. A tile border must never turn an
        # attached, occluded continuation into a supposedly detached component.
        from .wall_runs import filter_detached
        probability=np.empty((h,w),np.float32)
        for row in range(0,h,tile_pixels):
            probability[row:min(h,row+tile_pixels)]=sample(probability20,row,min(h,row+tile_pixels),0,w,.02/pixel_m,linear=True)
        out,weak,quarantine=filter_detached(out,np.clip(probability,0,1),faces,pixel_m,cfg.detached_probability)
        confidence_removed=int(weak.sum());editable|=weak
    changed=int(np.count_nonzero(out!=original))
    return dict(mask=out,guide=faces,editable=editable,added=out&~original,removed=original&~out,
                original=original,config=asdict(cfg),quarantine=quarantine,metrics=dict(graph_tiles=graphs,graph_nodes_with_halos=nodes,
                proposed_changes=proposed,topology_blocked_changes=proposed-topology_changed,confidence_removed_pixels=confidence_removed,
                added_pixels=int((out&~original).sum()),removed_pixels=int((original&~out).sum()),
                changed_area_m2=changed*pixel_m**2,pixel_m=pixel_m,halo_pixels=halo,
                verticality_resolution_m=.01,xy_count_resolution_m=pixel_m))
