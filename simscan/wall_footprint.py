"""Native wall footprint cleanup; fitted tiles are provenance, not active walls."""
from copy import deepcopy
import numpy as np
import cv2
from shapely import box, union_all
from shapely.geometry import Polygon
from .decompose import _local_mask,_bounds,_connectivity_breaks,mask_metrics,_topology,rectangle_overlap_metrics
from .wall_runs import _union,extract_runs,_profile,_point_distances,_sample
from .wall_merge import polygon_mask


def mask_geometry(mask, origin=(0,0)):
    """Exact union of occupied pixel cells, avoiding contour rounding/hole loss."""
    edge=np.diff(np.pad(mask.astype('int8'),((0,0),(1,1))),axis=1)
    yy,x0=np.where(edge==1);_,x1=np.where(edge==-1)
    if not len(yy):return Polygon()
    ox,oy=origin
    return union_all(box(x0+ox-.5,yy+oy-.5,x1+ox-.5,yy+oy+.5))


def regions(geometry):
    if geometry.is_empty:return []
    return [geometry] if geometry.geom_type=='Polygon' else [g for g in geometry.geoms if g.geom_type=='Polygon']


def primitive(g,index):
    return dict(id=index,kind='wall_region',polygon=np.asarray(g.exterior.coords)[:-1].tolist(),
        holes=[np.asarray(r.coords)[:-1].tolist() for r in g.interiors],footprint_role='authoritative')


def clean_footprint(result,target,barrier,pixel_m=.0025,probability=None,forbidden=None,config=None):
    from .decompose import Config
    cfg=config or Config();target=np.asarray(target,bool);shape=target.shape
    forbidden=np.zeros(shape,bool) if forbidden is None else np.asarray(forbidden,bool)
    source=deepcopy(result['primitives']);parts=[p for c in source for p in (c['parts'] if c['kind']=='wall_run' else [c])]
    mask=result['mask'].copy();profiles=[_profile(r,target,forbidden,pixel_m,cfg) for r in extract_runs(target,barrier,pixel_m,cfg,forbidden)]
    edits=[];rejections={}
    def safe(old,new,lo):
        if (new&forbidden[lo[1]:lo[1]+new.shape[0],lo[0]:lo[0]+new.shape[1]]).any():return False
        if _connectivity_breaks(old,new):return False
        sl=(slice(lo[1],lo[1]+new.shape[0]),slice(lo[0],lo[0]+new.shape[1]))
        depth=cv2.distanceTransform(np.pad(new.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
        if (new&~target[sl]&barrier[sl]&(depth>cfg.barrier_contact_m)).any():return False
        from .wall_runs import _boundary_distance
        bd=_boundary_distance(old,pixel_m);ad=_boundary_distance(new,pixel_m)
        for pr in profiles:
            pts=pr['points'];inside=pr['valid']&((pts>=lo+3)&(pts<lo+np.array(new.shape[::-1])-3)).all(1)
            if inside.any():
                q=pts[inside]-lo
                if ((_sample(bd,q)<=.015)&(_sample(ad,q)>.015)).any():return False
        return True
    # Remove only small unique junction residue when surrounding fitted tiles
    # already carry the connection and observed faces. Never delete a joint wholesale.
    coverage=np.zeros(shape,np.uint16)
    for p in parts:
        lo,hi=_bounds(p,shape);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));coverage[sl]+=_local_mask(p,lo,tuple((hi-lo)[::-1]))
    for p in parts:
        if p['kind']!='junction':continue
        lo0,hi0=_bounds(p,shape);own=_local_mask(p,lo0,tuple((hi0-lo0)[::-1]));sl0=(slice(lo0[1],hi0[1]),slice(lo0[0],hi0[0]))
        unique=own&(coverage[sl0]==1)
        if not unique.any() or unique.sum()*pixel_m**2>.03:continue
        pad=int(np.ceil(.08/pixel_m));lo=np.maximum(0,lo0-pad);hi=np.minimum(shape[::-1],hi0+pad);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        old=mask[sl];new=old.copy();offset=lo0-lo
        new[offset[1]:offset[1]+unique.shape[0],offset[0]:offset[0]+unique.shape[1]][unique]=False
        # Residue farther than 5 cm from retained walls is not a fitting seam.
        distance=cv2.distanceTransform((~new).astype('uint8'),cv2.DIST_L2,5)*pixel_m
        if distance[old&~new].max()>.05 or not safe(old,new,lo):continue
        mask[sl]=new;edits.append(dict(operation='remove_unsupported_junction_residue',pixels=int(unique.sum())))
    del coverage
    # A T stem may overshoot the back of its crossbar. Trim that endpoint
    # locally, only when the removed material has no dense scan support.
    rectangles=[p for p in parts if p['kind']=='rectangle']
    for stem in rectangles:
        if stem['length']*pixel_m<.4:continue
        for crossbar in rectangles:
            if crossbar is stem or crossbar['length']*pixel_m<1.:continue
            delta=abs((stem['angle']-crossbar['angle']+np.pi/2)%np.pi-np.pi/2)
            if not np.radians(80)<=delta<=np.pi/2:continue
            u=np.array([np.cos(crossbar['angle']),np.sin(crossbar['angle'])]);v=np.array([-u[1],u[0]])
            center=np.asarray(crossbar['center']);points=np.asarray(stem['polygon'])
            side=np.sign((np.asarray(stem['center'])-center)@v)
            if not side:continue
            q=(points-center)@v*side;edge=-crossbar['width']/2;extension=(edge-q.min())*pixel_m
            if not .005<extension<=.20 or q.max()*pixel_m<.3:continue
            along=(np.asarray(stem['center'])-center)@u
            if abs(along)>crossbar['length']/2-.02/pixel_m:continue
            pad=int(.10/pixel_m);lo,hi=_bounds(stem,shape);lo=np.maximum(0,lo-pad);hi=np.minimum(shape[::-1],hi+pad);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
            yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]];xy=np.stack([xx-center[0],yy-center[1]],-1)
            behind=(xy@v*side<edge)&(xy@v*side>=edge-.20/pixel_m)&(np.abs(xy@u-along)<=stem['width']/2+pad)
            old=mask[sl];retained=np.zeros_like(old)
            for support in rectangles:
                angle=abs((support['angle']-crossbar['angle']+np.pi/2)%np.pi-np.pi/2)
                if angle<=np.radians(2):retained|=_local_mask(support,lo,old.shape)
            new=old&~(behind&~retained);removed=old&~new
            if not removed.any() or removed.sum()*pixel_m**2>.08:continue
            near=cv2.dilate(barrier[sl].astype('uint8'),np.ones((9,9),'uint8'))>0
            if (removed&near).any() or not safe(old,new,lo):continue
            mask[sl]=new;edits.append(dict(operation='trim_unobserved_T_overshoot',pixels=int(removed.sum())))
    # Recover only scan-backed target residuals that bridge two existing regions.
    missing=target&~mask&(cv2.dilate(barrier.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0)
    n,labels,stats,_=cv2.connectedComponentsWithStats(missing.astype('uint8'),8)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area*pixel_m**2>.02 or max(w,h)*pixel_m>1.5:continue
        pad=int(np.ceil(.04/pixel_m));lo=np.maximum(0,[x-pad,y-pad]);hi=np.minimum(shape[::-1],[x+w+pad,y+h+pad]);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        part=labels[sl]==i;old=mask[sl];touch=cv2.dilate(part.astype('uint8'),np.ones((3,3),'uint8'))>0
        _,cc=cv2.connectedComponents(old.astype('uint8'),8);parents=np.unique(cc[touch&old]);parents=parents[parents>0]
        near=cv2.dilate(part.astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0
        if len(parents)<2 or (near&barrier[sl]).sum()<max(6,int(.06/pixel_m)):continue
        new=old|part
        if safe(old,new,lo):mask[sl]=new;edits.append(dict(operation='recover_allowed_scan_backed_strip',pixels=int(part.sum())))
    del labels,missing
    # Keep isolated elongated extensions available for review, not silently as
    # architectural walls. Compact columns and short aperture piers stay active.
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    quarantined=np.zeros(shape,bool);review=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i];long=max(w,h)*pixel_m;short=min(w,h)*pixel_m
        if not (.5<=long<=2 and short<=.18 and max(w,h)/max(1,min(w,h))>=4 and area*pixel_m**2<.2):continue
        pad=int(.20/pixel_m);lo=np.maximum(0,[x-pad,y-pad]);hi=np.minimum(shape[::-1],[x+w+pad,y+h+pad]);sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
        if np.any((labels[sl]!=i)&(labels[sl]>0)):continue
        component=labels[y:y+h,x:x+w]==i;quarantined[y:y+h,x:x+w]|=component
        mask[y:y+h,x:x+w][component]=False
        review.append(dict(bbox=[int(x),int(y),int(w),int(h)],area_m2=float(area)*pixel_m**2,reason='isolated_elongated_scan_feature_requires_apartment_context'))
    del labels
    # Close only tiny enclosed fitting slits inside the allowed wall target.
    # Real apertures, rooms, and observed scan faces cannot be filled here.
    n,labels,stats,_=cv2.connectedComponentsWithStats((~mask).astype('uint8'),8)
    exterior=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]))
    for i in range(1,n):
        if i in exterior:continue
        x,y,w,h,area=stats[i]
        if area*pixel_m**2>.01:continue
        sl=(slice(y,y+h),slice(x,x+w));hole=labels[sl]==i
        if (hole&forbidden[sl]).any() or (hole&barrier[sl]).any() or target[sl][hole].mean()<.95:continue
        depth=cv2.distanceTransform(np.pad(hole.astype('uint8'),1),cv2.DIST_L2,5)*pixel_m
        if depth.max()*2>.02:continue
        mask[sl]|=hole;edits.append(dict(operation='close_enclosed_fitting_slit',pixels=int(area)))
    del labels
    # Construct exact non-overlapping connected footprints, then simplify each
    # outline locally. Grid stairs are not retained merely to preserve old tiles.
    geometry=mask_geometry(mask);final=[]
    for g in regions(geometry):
        x0,y0,x1,y1=g.bounds;pad=int(.08/pixel_m);lo=np.maximum(0,np.floor([x0-pad,y0-pad]).astype(int));hi=np.minimum(shape[::-1],np.ceil([x1+pad,y1+pad]).astype(int));sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]));old=mask[sl]
        selected=g
        for tolerance in [.02,.015,.01,.005]:
            trial=g.simplify(tolerance/pixel_m,preserve_topology=True)
            patch=polygon_mask(np.asarray(trial.exterior.coords)[:-1],[np.asarray(r.coords)[:-1] for r in trial.interiors],lo,tuple((hi-lo)[::-1]))
            # Other components in the crop are immutable.
            original=polygon_mask(np.asarray(g.exterior.coords)[:-1],[np.asarray(r.coords)[:-1] for r in g.interiors],lo,patch.shape)
            other=old&~original
            if (patch&other).any():continue
            # Keep the measured face band pixel-exact while simplifying the
            # inferred side; clip real semantic exclusions rather than rejecting
            # an entire connected apartment for a tiny door-edge movement.
            radius=max(1,int(np.ceil(.015/pixel_m))+2)
            protected=cv2.dilate(barrier[sl].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2))>0
            patch[protected]=original[protected]
            patch&=~forbidden[sl]
            if (patch&other).any():continue
            new=other|patch
            if safe(old,new,lo):
                candidate_regions=regions(mask_geometry(patch,lo))
                if len(candidate_regions)!=1:continue
                selected=candidate_regions[0];mask[sl]=new;edits.append(dict(operation='simplify_joint_outline',tolerance_m=tolerance,vertices_before=len(g.exterior.coords),vertices_after=len(selected.exterior.coords)));break
        compact=selected.simplify(.45,preserve_topology=True)
        exact=_local_mask(primitive(selected,0),lo,tuple((hi-lo)[::-1]))
        approximation=_local_mask(primitive(compact,0),lo,exact.shape)
        if np.array_equal(exact,approximation):selected=compact
        final.append(primitive(selected,len(final)))
    rebuilt=_union(final,shape)
    if not np.array_equal(mask,rebuilt):raise ValueError('Final footprint geometry differs from raster')
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense','world_regions']})
    out.update(primitives=final,source_primitives=source,mask=mask,review_mask=quarantined,
        footprint_cleanup=dict(edits=edits,review_features=review,source_rectangle_overlap=rectangle_overlap_metrics(source,shape,pixel_m),
            active_overlap_area_m2=0.,policy='nonoverlapping connected footprints; source tiles are provenance only'),
        contacts=_topology(final,shape),metrics=mask_metrics(target,mask))
    return out
