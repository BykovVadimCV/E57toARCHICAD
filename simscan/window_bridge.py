"""Window masks seed location/direction; neighboring wall bands define gap closers."""
from dataclasses import dataclass,asdict
import numpy as np
import cv2
from shapely.geometry import Polygon,LineString,Point
from shapely import affinity,union_all
from .wall_footprint import mask_geometry,regions
from .wall_runs import _sample,_intervals

@dataclass(frozen=True)
class Config:
    min_area_m2: float=.006
    min_length_m: float=.25
    max_length_m: float=4.
    search_m: float=.8
    max_extension_m: float=.4
    max_center_mismatch_m: float=.04
    min_width_m: float=.035
    max_width_m: float=.5


def _directions(result,seed,pca,aspect,px):
    candidates=[pca];limit=np.radians(30 if aspect>=2 else 70)
    for face in result.get('wall_faces',[]):
        line=LineString([face['start'],face['end']])
        if seed.distance(line)*px>.65:continue
        d=np.asarray(face['end'],float)-face['start'];d/=np.linalg.norm(d)
        if abs(d@pca)>=np.cos(limit):candidates.append(d)
    for edit in result.get('thickness_regularization',{}).get('edits',[]):
        r=edit['line'];u=np.array(r['u']);n=np.array(r['normal']);points=[u*r[t]+n*r['offset'] for t in ['t0','t1']]
        if seed.distance(LineString(points))*px<=.65 and abs(u@pca)>=np.cos(limit):candidates.append(u)
    out=[]
    for u in candidates[1:]+candidates[:1]:
        if u[0]<0:u=-u
        if not any(abs(u@v)>np.cos(np.radians(.7)) for v in out):out.append(u)
    return out


def _host(mask,seed,points,u,px,cfg):
    n=np.array([-u[1],u[0]]);t=points@u;q=points@n;tlo,thi=np.percentile(t,[0,100]);q0=float(np.median(q))
    if not cfg.min_length_m<=(thi-tlo)*px<=cfg.max_length_m:return None,'length'
    qs=np.arange(q0-.65/px,q0+.65/px+1)
    step=.02/px
    def bands(t):
        positions=t*u+qs[:,None]*n
        occupied=_sample(mask,positions)
        out=[]
        for a,b in _intervals(occupied):
            width=(b-a)*px
            if cfg.min_width_m<=width<=cfg.max_width_m:out.append((float(qs[a]-.5),float(qs[b-1]+.5)))
        return out
    def supports(edge,sign):
        out=[]
        for k in range(1,int(cfg.search_m/.02)+1):
            tv=edge+sign*k*step
            for a,b in bands(tv):
                center=(a+b)/2
                # A stable 6cm wall band is required on each side, not one hit.
                stable=True
                for offset in [step,2*step]:
                    following=bands(tv+sign*offset)
                    if not any(abs((c+d)/2-center)*px<=.025 and abs((d-c)-(b-a))*px<=.025 for c,d in following):stable=False;break
                if stable:out.append(dict(t=float(tv),a=a,b=b,center=center))
            if len(out)>=8:break
        return out
    left=supports(tlo,-1);right=supports(thi,1)
    def corner_support(edge,sign,other):
        if not other:return []
        reference=other[0];half=(reference['b']-reference['a'])/2;center=reference['center']
        for k in range(1,int(cfg.search_m/.02)+1):
            tv=edge+sign*k*step;hits=[]
            for shift in [0.,step]:
                occupied=_sample(mask,(tv+sign*shift)*u+qs[:,None]*n)
                hits.append(any(qs[a]<=center<=qs[b-1] and (b-a)*px>.5 for a,b in _intervals(occupied)))
            if all(hits):return [dict(t=float(tv),a=center-half,b=center+half,center=center,host_type='transverse_wall_corner')]
        return []
    if not left:left=corner_support(tlo,-1,right)
    if not right:right=corner_support(thi,1,left)
    if not left or not right:return None,'missing_two_wall_abutments' 
    options=[]
    for a in left:
        for b in right:
            wa=(a['b']-a['a'])*px;wb=(b['b']-b['a'])*px
            if abs(wa-wb)>max(.12,.65*max(wa,wb)):continue
            # Correct a small direction error using the two host band centers.
            pa=a['t']*u+a['center']*n;pb=b['t']*u+b['center']*n
            direction=pb-pa;direction/=np.linalg.norm(direction)
            angle=np.arccos(np.clip(direction@u,-1,1))
            if angle>np.radians(8):continue
            if abs(a['center']-b['center'])*px>cfg.max_center_mismatch_m+.03*(b['t']-a['t'])*px:continue
            aa=pa-direction*.005/px;bb=pb+direction*.005/px;normal=np.array([-direction[1],direction[0]]);width=(wa+wb)/2
            poly=Polygon([aa-normal*wa/px/2,bb-normal*wb/px/2,bb+normal*wb/px/2,aa+normal*wa/px/2])
            if poly.intersection(seed).area/max(poly.area,1)<.15:continue
            extension=max((tlo-a['t'])*px,(b['t']-thi)*px)
            if extension>cfg.max_extension_m:continue
            score=extension+abs(wa-wb)+abs(a['center']-b['center'])*px+.1*angle
            options.append((score,dict(polygon=poly,axis=[aa.tolist(),bb.tolist()],width_m=width,
                length_m=float(np.linalg.norm(bb-aa)*px),host_widths_m=[wa,wb],direction_correction_degrees=float(np.degrees(angle)),
                abutments=[pa.tolist(),pb.tolist()],host_types=[a.get('host_type','aligned_wall_band'),b.get('host_type','aligned_wall_band')])))
    if not options:return None,'incompatible_wall_bands'
    return min(options,key=lambda o:o[0]),None


def fit_windows(prob,wall_result,wall_mask,pixel_m=.0025,prob_pixel_m=.02,config=None):
    cfg=config or Config();prob=np.asarray(prob,np.float32);mask=np.asarray(wall_mask,bool)
    factor=prob_pixel_m/pixel_m
    if prob.shape[0]!=8 or tuple(int(round(v*factor)) for v in prob.shape[1:])!=mask.shape:raise ValueError('Window/network/wall frames differ')
    labels=prob.argmax(0);seed_mask=(labels==4)&(prob[4]>=.30)
    n,cc,stats,_=cv2.connectedComponentsWithStats(seed_mask.astype('uint8'),8)
    walls=union_all([Polygon(c['polygon'],c.get('holes',[])) for c in wall_result['primitives']])
    door=mask_geometry(np.isin(labels,[3,5]));door=affinity.translate(affinity.scale(door,xfact=factor,yfact=factor,origin=(0,0)),(factor-1)/2,(factor-1)/2)
    windows=[];review=[];claimed=Polygon()
    for i in range(1,n):
        x,y,w,h,area=stats[i];component=cc[y:y+h,x:x+w]==i
        if area*prob_pixel_m**2<cfg.min_area_m2:continue
        yy,xx=np.nonzero(component);points=(np.c_[xx+x,yy+y]+.5)*factor-.5
        center=points.mean(0);_,s,v=np.linalg.svd(points-center,full_matrices=False);direction=v[0];aspect=float(s[0]/max(s[-1],1e-6))
        seed=mask_geometry(component,(x,y));seed=affinity.translate(affinity.scale(seed,xfact=factor,yfact=factor,origin=(0,0)),(factor-1)/2,(factor-1)/2)
        candidates=[];reasons=[]
        for u in _directions(wall_result,seed,direction,aspect,pixel_m):
            candidate,reason=_host(mask,seed,points,u,pixel_m,cfg)
            if candidate is None:reasons.append(reason);continue
            score,item=candidate
            if item['polygon'].intersection(door).area*pixel_m**2>pixel_m**2:reasons.append('door_or_opening_conflict');continue
            gap=item['polygon'].difference(walls).difference(door).difference(claimed)
            if gap.area*pixel_m**2<.002:reasons.append('no_wall_gap');continue
            if gap.area/item['polygon'].area<.15:reasons.append('mostly_existing_wall');continue
            pieces=sorted(regions(gap),key=lambda g:-g.area)
            major=[g for g in pieces if g.area>=max(.0005/pixel_m**2,pieces[0].area*.02)]
            if len(major)!=1:reasons.append('multiple_separate_gaps');continue
            gap=major[0]
            if gap.intersection(seed).area/max(1,gap.area)<.15:reasons.append('gap_away_from_seed');continue
            candidates.append((score,item,gap))
        metadata=dict(seed_component=i,seed_bbox_px=[int(x),int(y),int(w),int(h)],mean_window_probability=float(prob[4,y:y+h,x:x+w][component].mean()))
        if not candidates:
            review.append(dict(**metadata,reasons=sorted(set(reasons))));continue
        _,item,gap=min(candidates,key=lambda a:a[0]);claimed=claimed.union(gap)
        axis=np.asarray(item['axis']);u=axis[1]-axis[0];u/=np.linalg.norm(u)
        vertices=np.asarray(gap.exterior.coords);along=(vertices-axis[0])@u
        # The exported window ends at the actual gap, not deep inside its hosts.
        item['axis']=[(axis[0]+u*along.min()).tolist(),(axis[0]+u*along.max()).tolist()]
        item['length_m']=float(np.ptp(along)*pixel_m)
        geometry=[dict(polygon=np.asarray(g.exterior.coords)[:-1].tolist(),holes=[np.asarray(r.coords)[:-1].tolist() for r in g.interiors]) for g in regions(gap)]
        windows.append(dict(id=f'W{len(windows)+1}',kind='window',**metadata,
            axis=item['axis'],width_m=item['width_m'],length_m=item['length_m'],host_widths_m=item['host_widths_m'],host_types=item['host_types'],
            footprint_regions=geometry,bridge_polygon=np.asarray(item['polygon'].exterior.coords)[:-1].tolist(),
            host_region_ids=[[c['id'] for c in wall_result['primitives'] if Polygon(c['polygon'],c.get('holes',[])).distance(Point(point))*pixel_m<=.01] for point in item['abutments']],
            host_support_positions_px=item['abutments'],
            direction_correction_degrees=item['direction_correction_degrees'],placement='two_wall_abutment_gap_closer',
            source='class-4 mask for location; neighboring wall bands for alignment and width',height_status='not_estimated'))
    return dict(windows=windows,window_review=review,window_config=asdict(cfg),policy='separate window gap closers; wall geometry unchanged; doors remain excluded')
