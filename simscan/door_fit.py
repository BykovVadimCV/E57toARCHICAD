"""Door masks locate openings; stable wall abutments establish their fitted spans.

Exports opening geometry only. A closed leaf can produce dense returns across a
valid door: scan evidence supports jambs but cannot veto the opening interior or
supply a hinge, swing direction or height.
"""
from dataclasses import dataclass,asdict,replace
import numpy as np
import cv2
from shapely.geometry import Polygon,Point,LineString
from shapely import affinity,union_all
from .window_bridge import Config as HostConfig,_host,_directions
from .wall_footprint import mask_geometry,regions
from .wall_runs import _sample,_intervals


@dataclass(frozen=True)
class Config(HostConfig):
    min_area_m2: float=.008
    min_length_m: float=.50
    max_length_m: float=2.4
    search_m: float=.6
    max_extension_m: float=.25
    max_width_m: float=.8


def dense_door_returns(counts,pixel_m=.0025,min_count_2cm=200.):
    """Keep the high count threshold, without requiring full-height wall returns.

    Door leaves and opening edges often fail the wall-specific two-height gate.
    Their dense returns may refine a mask-seeded door, never propose a new wall.
    """
    dense=(np.asarray(counts)>=min_count_2cm*(pixel_m/.02)**2).astype('uint8')
    r=max(1,int(round(.005/pixel_m)))
    dense=cv2.morphologyEx(dense,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,)*2))
    n,labels,stats,_=cv2.connectedComponentsWithStats(dense,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=max(3,int(round(.06/pixel_m)));keep[0]=False
    return keep[labels]


def _jamb_support(barrier,axis,thickness,px):
    """Each 1cm longitudinal bin votes once; point count cannot dominate a jamb."""
    if barrier is None:return dict(status='not_available',jambs=[])
    axis=np.asarray(axis,float);u=axis[1]-axis[0];u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
    q=np.arange(-thickness/px/2-.015/px,thickness/px/2+.015/px+1)
    rows=[]
    for point,sign in zip(axis,[-1,1]):
        # Sample the host just outside each opening, not the closed leaf inside.
        t=np.arange(.01,.071,.01)/px*sign
        xy=point+t[:,None,None]*u+q[None,:,None]*v
        occupied=_sample(barrier,xy.reshape(-1,2)).reshape(len(t),len(q)).any(1)
        rows.append(dict(occupied_bins=int(occupied.sum()),bins=int(len(t)),coverage=float(occupied.mean())))
    return dict(status='both_jambs_supported' if all(r['occupied_bins']>=3 for r in rows) else 'partial_or_unobserved_jambs',jambs=rows)


def _scan_corner_host(mask,points,u,barrier,px,cfg):
    """Two measured door-band sides can connect two transverse wall ends."""
    if barrier is None:return None,'corner_scan_not_available'
    n=np.array([-u[1],u[0]]);st=points@u;sq=points@n;t0,t1=st.min(),st.max()
    if not cfg.min_length_m<=(t1-t0)*px<=cfg.max_length_m:return None,'length'
    offsets=np.percentile(sq,[3,97]);pad=.05/px
    corners=np.array([u*t+n*q for t in [t0,t1] for q in [offsets[0]-pad,offsets[1]+pad]])
    lo=np.maximum(0,np.floor(corners.min(0)-2).astype(int));hi=np.minimum(mask.shape[::-1],np.ceil(corners.max(0)+3).astype(int))
    yy,xx=np.nonzero(barrier[lo[1]:hi[1],lo[0]:hi[0]]);scan=np.c_[xx+lo[0],yy+lo[1]].astype(float)
    if len(scan)<20:return None,'missing_paired_scan_edges'
    along=scan@u;normal=scan@n;trim=min(.06/px,.1*(t1-t0));inner=(along>=t0+trim)&(along<=t1-trim)
    faces=[];dirs=[];coverage=[]
    for offset in offsets:
        cloud=scan[inner&(np.abs(normal-offset)*px<=.04)]
        if len(cloud)<12:return None,'missing_paired_scan_edges'
        line=cv2.fitLine(cloud.astype('float32'),cv2.DIST_HUBER,0,.01,.001).ravel();d=line[:2].astype(float);d/=np.linalg.norm(d)
        if d@u<0:d=-d
        if d@u<np.cos(np.radians(8)):return None,'corner_scan_direction'
        v=np.array([-d[1],d[0]]);q=cloud@v;center=float(np.median(q));cloud=cloud[np.abs(q-center)*px<=.01]
        if len(cloud)<12:return None,'corner_scan_scatter'
        bins=max(1,int(np.ceil((t1-t0-2*trim)*px/.02)))
        occupied=np.unique(np.clip(((cloud@u-t0-trim)*px/.02).astype(int),0,bins-1));support=len(occupied)/bins
        if support<.70:return None,'fragmented_corner_scan_edges'
        faces.append(cloud);dirs.append(d);coverage.append(float(support))
    if dirs[0]@dirs[1]<np.cos(np.radians(1)):return None,'nonparallel_corner_scan_edges'
    direction=dirs[0]+dirs[1];direction/=np.linalg.norm(direction);normal=np.array([-direction[1],direction[0]])
    a,b=sorted(float(np.median(c@normal)) for c in faces);width=(b-a)*px
    if not cfg.min_width_m<=width<=cfg.max_width_m:return None,'corner_scan_width'
    center=(a+b)/2;along=points@direction;t0,t1=along.min(),along.max()
    qs=np.arange(a,b+1)
    def support(edge,sign):
        # Start just inside the mask, so a coarse seed that reaches a jamb can
        # still terminate at that measured wall end. Never search beyond 25cm.
        for distance in np.arange(-.06,cfg.max_extension_m+.001,.01):
            t=edge+sign*distance/px
            hit=[]
            for delta in [0.,.01]:
                line=(t+sign*delta/px)*direction+qs[:,None]*normal
                hit.append(float(_sample(mask,line).mean()))
            if min(hit)>=.45:return t
        return None
    left=support(t0,-1);right=support(t1,1)
    if left is None or right is None:return None,'missing_transverse_wall_ends'
    aa=direction*(left-.005/px)+normal*center;bb=direction*(right+.005/px)+normal*center
    quad=Polygon([aa-normal*width/px/2,bb-normal*width/px/2,bb+normal*width/px/2,aa+normal*width/px/2])
    extension=max(0.,(t0-left)*px,(right-t1)*px)
    angle=np.degrees(np.arccos(np.clip(direction@u,-1,1)))
    item=dict(polygon=quad,axis=[aa.tolist(),bb.tolist()],width_m=width,host_widths_m=[width,width],
        abutments=[aa.tolist(),bb.tolist()],host_types=['scan_band_transverse_abutment']*2,direction_correction_degrees=float(angle),
        paired_scan_edge_coverage=coverage)
    return (extension+.005,item),None


def fit_doors(prob,wall_result,wall_mask,pixel_m=.0025,prob_pixel_m=.02,*,windows=None,barrier=None,counts=None,config=None):
    cfg=config or Config();p=np.asarray(prob,np.float32);mask=np.asarray(wall_mask,bool);factor=prob_pixel_m/pixel_m
    if p.ndim!=3 or p.shape[0]!=8 or pixel_m<=0 or prob_pixel_m<=0 or tuple(int(round(v*factor)) for v in p.shape[1:])!=mask.shape:
        raise ValueError('Door/network/wall frames differ')
    if not np.isfinite(p).all():raise ValueError('Nonfinite door probabilities')
    if barrier is not None:
        barrier=np.asarray(barrier,bool)
        if barrier.shape!=mask.shape:raise ValueError('Door/scan frames differ')
    door_evidence=barrier
    if counts is not None:
        counts=np.asarray(counts)
        if counts.shape!=mask.shape or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Invalid door count frame')
        door_evidence=dense_door_returns(counts,pixel_m)
    labels=p.argmax(0);seed_mask=(labels==3)&(p[3]>=.30)
    n,cc,stats,_=cv2.connectedComponentsWithStats(seed_mask.astype('uint8'),8)
    walls=union_all([Polygon(c['polygon'],c.get('holes',[])) for c in wall_result['primitives']])
    window_geoms=[Polygon(g['polygon'],g.get('holes',[])) for w in (windows or []) for g in w['footprint_regions']]
    window_seed=mask_geometry(labels==4)
    window_seed=affinity.translate(affinity.scale(window_seed,xfact=factor,yfact=factor,origin=(0,0)),(factor-1)/2,(factor-1)/2)
    blocked=union_all([g for g in window_geoms+[window_seed] if not g.is_empty]);claimed=Polygon();doors=[];review=[]
    for i in range(1,n):
        x,y,w,h,area=stats[i];component=cc[y:y+h,x:x+w]==i
        if area*prob_pixel_m**2<cfg.min_area_m2:continue
        yy,xx=np.nonzero(component);points=(np.c_[xx+x,yy+y]+.5)*factor-.5
        center=points.mean(0);_,s,v=np.linalg.svd(points-center,full_matrices=False);direction=v[0];aspect=float(s[0]/max(s[-1],1e-6))
        seed=mask_geometry(component,(x,y));seed=affinity.translate(affinity.scale(seed,xfact=factor,yfact=factor,origin=(0,0)),(factor-1)/2,(factor-1)/2)
        candidates=[];reasons=[]
        for u in _directions(wall_result,seed,direction,aspect,pixel_m):
            candidate,reason=_host(mask,seed,points,u,pixel_m,replace(cfg,max_width_m=min(.5,cfg.max_width_m)))
            if candidate is None and cfg.max_width_m>.5:
                candidate,reason=_host(mask,seed,points,u,pixel_m,cfg)
            if candidate is None:
                reasons.append(reason)
                candidate,reason=_scan_corner_host(mask,points,u,barrier,pixel_m,cfg)
                scan_source='wall_qualified_dense_returns'
                if candidate is None and counts is not None:
                    reasons.append(reason)
                    candidate,reason=_scan_corner_host(mask,points,u,door_evidence,pixel_m,cfg)
                    scan_source='density_qualified_door_returns'
                if candidate is None:reasons.append(reason);continue
                candidate[1]['paired_scan_source']=scan_source
            score,item=candidate;quad=item['polygon']
            if quad.intersection(blocked).area*pixel_m**2>pixel_m**2:reasons.append('window_conflict');continue
            if quad.intersection(claimed).area*pixel_m**2>.001:reasons.append('duplicate_opening');continue
            gap=quad.difference(walls).difference(blocked).difference(claimed)
            if gap.area*pixel_m**2<.008 or gap.area/max(1,quad.area)<.25:reasons.append('no_clear_wall_gap');continue
            pieces=sorted(regions(gap),key=lambda g:-g.area)
            major=[g for g in pieces if g.area>=max(.0005/pixel_m**2,pieces[0].area*.02)]
            if len(major)!=1:reasons.append('multiple_separate_gaps');continue
            gap=major[0]
            if gap.intersection(seed).area/max(1,gap.area)<.20:reasons.append('gap_away_from_seed');continue
            axis=np.asarray(item['axis']);u=axis[1]-axis[0];u/=np.linalg.norm(u)
            # The extreme vertices can be tiny half-pixel shoulders extending
            # along the host surface. Measure the actual opening on its axis.
            crossing=gap.intersection(LineString(axis))
            segments=[crossing] if crossing.geom_type=='LineString' else [g for g in getattr(crossing,'geoms',[]) if g.geom_type=='LineString']
            segments=sorted(segments,key=lambda g:-g.length)
            if not segments:reasons.append('no_axis_gap');continue
            if any(g.length*pixel_m>.05 and g.length>.1*segments[0].length for g in segments[1:]):reasons.append('multiple_axis_gaps');continue
            along=(np.asarray(segments[0].coords)-axis[0])@u
            fitted_axis=np.array([axis[0]+u*along.min(),axis[0]+u*along.max()]);span=float(np.ptp(along)*pixel_m)
            normal=np.array([-u[1],u[0]])*cfg.max_width_m/pixel_m
            aa,bb=fitted_axis
            gap=gap.intersection(Polygon([aa-normal,bb-normal,bb+normal,aa+normal]))
            pieces=regions(gap)
            if not pieces:reasons.append('empty_axis_gap');continue
            gap=max(pieces,key=lambda g:g.area)
            if not cfg.min_length_m<=span<=cfg.max_length_m:reasons.append('fitted_opening_width');continue
            # Both endpoints must actually meet the fitted wall footprint. A
            # long mask, empty space or a free-floating scan line is not a host.
            ends=[Point(a) for a in fitted_axis]
            if any(walls.distance(a)*pixel_m>.015 for a in ends):reasons.append('unattached_jamb');continue
            support=_jamb_support(barrier,fitted_axis,item['width_m'],pixel_m)
            # Weak/missing returns do not prove absence. At equal geometric fit,
            # favor measured jambs; keep unsupported evidence explicit in output.
            scan_bonus=.005*sum(r['coverage'] for r in support['jambs'])
            candidates.append((score-scan_bonus,item,gap,fitted_axis,span,support))
        metadata=dict(seed_component=int(i),seed_bbox_px=[int(x),int(y),int(w),int(h)],
            mean_door_probability=float(p[3,y:y+h,x:x+w][component].mean()))
        if not candidates:
            review.append(dict(**metadata,reasons=sorted(set(reasons))));continue
        _,item,gap,axis,span,support=min(candidates,key=lambda a:a[0]);claimed=claimed.union(gap)
        geometry=[dict(polygon=np.asarray(g.exterior.coords)[:-1].tolist(),holes=[np.asarray(h.coords)[:-1].tolist() for h in g.interiors]) for g in regions(gap)]
        host_ids=[[c['id'] for c in wall_result['primitives'] if Polygon(c['polygon'],c.get('holes',[])).distance(Point(point))*pixel_m<=.015] for point in axis]
        doors.append(dict(id=f'D{len(doors)+1}',kind='door',**metadata,axis=axis.tolist(),opening_width_m=span,
            host_thickness_source='paired_scan_edges' if 'paired_scan_edge_coverage' in item else 'neighboring_wall_bands',
            paired_scan_edge_coverage=item.get('paired_scan_edge_coverage'),paired_scan_source=item.get('paired_scan_source'),host_thickness_m=item['width_m'],host_thicknesses_m=item['host_widths_m'],host_types=item['host_types'],host_region_ids=host_ids,
            footprint_regions=geometry,scan_evidence=support,direction_correction_degrees=item['direction_correction_degrees'],
            placement='wall_abutment_opening',source='class-3 seed; fitted wall jambs; dense scan support at hosts',
            hinge_status='not_estimated',swing_status='not_estimated',height_status='not_estimated'))
    return dict(doors=doors,door_review=review,door_config=asdict(cfg),probability_pixel_m=prob_pixel_m,pixel_m=pixel_m,
        door_scan_policy=dict(min_count_2cm=200.,wall_height_gate=False,source='raw_density' if counts is not None else 'provided_wall_barrier'),
        policy='separate door openings; walls/windows unchanged; class-5 openings not converted; no inferred hinge/swing/height')
