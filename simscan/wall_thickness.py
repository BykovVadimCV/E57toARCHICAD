"""Robust per-run widths anchored to measured scan faces, with protected joints."""
from copy import deepcopy
import numpy as np
import cv2
from .decompose import Config,_local_mask,_connectivity_breaks,mask_metrics,_topology
from .wall_runs import extract_runs,_profile,_intervals,_sample,_boundary_distance,_union,_band
from .wall_footprint import mask_geometry,regions,primitive
from .wall_simplify import _fit_two_faces


def merge_face_runs(runs,barrier,pixel_m):
    """Join observed collinear face fragments before applying length thresholds."""
    groups=[[r] for r in runs]
    changed=True
    while changed:
        changed=False
        for i,group in enumerate(groups):
            anchor=max(group,key=lambda r:r['t1']-r['t0']);u=anchor['u'];v=anchor['normal']
            ends=np.array([r['u']*t+r['normal']*r['offset'] for r in group for t in [r['t0'],r['t1']]])
            amin,amax=np.min(ends@u),np.max(ends@u)
            for j in range(i+1,len(groups)):
                other=groups[j];b=max(other,key=lambda r:r['t1']-r['t0'])
                if abs(u@b['u'])<np.cos(np.radians(.8)):continue
                pts=np.array([r['u']*t+r['normal']*r['offset'] for r in other for t in [r['t0'],r['t1']]])
                cross=np.abs(pts@v-anchor['offset'])*pixel_m
                if cross.min()>.015 or cross.max()>.03:continue
                t=pts@u;gap=max(0.,amin-t.max(),t.min()-amax)*pixel_m
                if gap>.60:continue
                groups[i]+=other;groups.pop(j);changed=True;break
            if changed:break
    result=[]
    for group in groups:
        if len(group)==1:result+=group;continue
        samples=[]
        for r in group:
            corners=np.array([r['u']*t+r['normal']*(r['offset']+q/pixel_m) for t in [r['t0'],r['t1']] for q in [-.015,.015]])
            lo=np.maximum(0,np.floor(corners.min(0)-2).astype(int));hi=np.minimum(barrier.shape[::-1],np.ceil(corners.max(0)+3).astype(int))
            yy,xx=np.nonzero(barrier[lo[1]:hi[1],lo[0]:hi[0]]);pts=np.c_[xx+lo[0],yy+lo[1]].astype(float)
            t=pts@r['u'];keep=(t>=r['t0'])&(t<=r['t1'])&(np.abs(pts@r['normal']-r['offset'])*pixel_m<=.015)
            samples.append(pts[keep])
        pts=np.unique(np.concatenate(samples),axis=0)
        if len(pts)<20:result+=group;continue
        fit=cv2.fitLine(pts.astype('float32'),cv2.DIST_HUBER,0,.01,.001).ravel();u=fit[:2].astype(float);u/=np.linalg.norm(u)
        if u[0]<0:u=-u
        v=np.array([-u[1],u[0]]);q=pts@v;offset=float(np.median(q));scatter=float(np.percentile(np.abs(q-offset),90)*pixel_m)
        if scatter>.0125:result+=group;continue
        t=pts@u;result.append(dict(u=u,normal=v,offset=offset,t0=float(t.min()),t1=float(t.max()),scatter_m=scatter,merged_fragments=len(group)))
    return sorted(result,key=lambda r:r['t0']-r['t1'])


def stabilize_thickness(result,target,barrier,pixel_m=.0025,forbidden=None,config=None):
    cfg=config or Config();mask=result['mask'].copy();shape=mask.shape
    forbidden=np.zeros(shape,bool) if forbidden is None else np.asarray(forbidden,bool)
    raw_runs=extract_runs(mask,barrier,pixel_m,cfg,forbidden)
    runs=merge_face_runs(raw_runs,barrier,pixel_m)
    profiles=[_profile(r,mask,forbidden,pixel_m,cfg) for r in runs]
    records=[]
    for index,(run,pr) in enumerate(zip(runs,profiles)):
        for side in [-1,1]:
            widths=pr['width'][pr['valid']&(pr['side']==side)]
            if len(widths)<100:continue
            width=float(np.median(widths));center=run['normal']*(run['offset']+side*width/pixel_m/2)
            records.append(dict(index=index,side=side,width=width,values=widths,
                start=run['u']*run['t0']+center,end=run['u']*run['t1']+center,u=run['u']))
    groups=list(range(len(records)))
    def root(i):
        while groups[i]!=i:i=groups[i]
        return i
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            if abs(a['u']@b['u'])<np.cos(np.radians(1)) or abs(a['width']-b['width'])>.04:continue
            v=np.array([-a['u'][1],a['u'][0]]);t=np.array([b['start'],b['end']])@a['u'];ends=np.sort(np.array([a['start'],a['end']])@a['u'])
            gap=max(0.,ends[0]-max(t),min(t)-ends[1])*pixel_m
            if gap>.35:continue
            if max(abs((b['start']-a['start'])@v),abs((b['end']-a['start'])@v))*pixel_m>.035:continue
            groups[root(i)]=root(j)
    shared={}
    for i,rec in enumerate(records):
        values=np.concatenate([r['values'] for j,r in enumerate(records) if root(j)==root(i)])
        median=float(np.median(values));shared[rec['index'],rec['side']]=median
    edits=[];rejected={}
    for run_index,(run,initial) in enumerate(zip(runs,profiles)):
        pr=_profile(run,mask,forbidden,pixel_m,cfg)
        for side in [-1,1]:
            values=pr['width'][pr['valid']&(pr['side']==side)]
            if len(values)<100:continue
            median=shared.get((run_index,side),float(np.median(values)));valid=pr['valid']&(pr['side']==side)&(np.abs(pr['width']-median)<=.04)
            working_width=pr['width'].copy();bridged=np.zeros_like(valid)
            for left,right in _intervals(~valid):
                if left==0 or right==len(valid) or (right-left)*pr['step']*pixel_m>.20:continue
                if not valid[left-1] or not valid[right]:continue
                qs=np.linspace(0,median/pixel_m,max(3,int(np.ceil(median/pixel_m))+1))
                points=pr['points'][left:right,None,:]+side*qs[None,:,None]*run['normal']
                if _sample(forbidden,points.reshape(-1,2)).any():continue
                valid[left:right]=True;working_width[left:right]=median;bridged[left:right]=True
            # Preserve ends/aperture reveals and any continuous width-change region.
            for a,b in _intervals(valid):
                cap=max(1,int(round(.05/(pr['step']*pixel_m))));a+=cap;b-=cap
                if (b-a)*pr['step']*pixel_m<1.:continue
                widths=working_width[a:b];width=median
                if np.percentile(widths,98)-np.percentile(widths,2)<.005 and abs(float(np.median(widths))-width)<.005 and not bridged[a:b].any():continue
                t0=pr['t'][a];t1=pr['t'][b-1];u=run['u'];normal=run['normal']
                # A perpendicular observed wall identifies a protected junction.
                protected_t=[]
                for other in runs:
                    if abs(u@other['u'])>np.sin(np.radians(8)):continue
                    den=float(u@other['normal'])
                    if abs(den)<.5:continue
                    t=(other['offset']-run['offset']*(normal@other['normal']))/den
                    point=u*t+normal*run['offset'];along=point@other['u']
                    if other['t0']-.12/pixel_m<=along<=other['t1']+.12/pixel_m:protected_t.append(t)
                corners=np.array([u*t+normal*(run['offset']+side*q/pixel_m) for t in [t0,t1] for q in [-.025,float(widths.max())+.045]])
                lo=np.maximum(0,np.floor(corners.min(0)-5).astype(int));hi=np.minimum(shape[::-1],np.ceil(corners.max(0)+6).astype(int));sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
                band=_band(run,t0,t1,width,side,pixel_m)
                paired,pair_reason=_fit_two_faces([band],barrier,pixel_m,cfg)
                if bridged[a:b].any() and paired is None:continue
                old=mask[sl].copy();yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
                along=xx*u[0]+yy*u[1];q=(xx*normal[0]+yy*normal[1]-run['offset'])*side*pixel_m
                editable=(along>=t0)&(along<=t1)&(q>(-.02 if paired is not None else .015))&(q<float(widths.max())+.04)
                for t in protected_t:editable&=np.abs(along-t)*pixel_m>.12
                # Never straighten the independently observed opposite face.
                dense_band=cv2.dilate(barrier[sl].astype('uint8'),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))>0
                if paired is None:editable&=~dense_band
                desired=_local_mask(paired,lo,old.shape) if paired is not None else q<=width
                new=old.copy();new[editable]=desired[editable]
                if np.array_equal(old,new):continue
                if (new&forbidden[sl]).any():rejected['aperture']=rejected.get('aperture',0)+1;continue
                if _connectivity_breaks(old,new):rejected['connection']=rejected.get('connection',0)+1;continue
                depth=cv2.distanceTransform(np.pad(new.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
                if (new&~target[sl]&barrier[sl]&(depth>.005)).any():continue
                # Local moves are bounded to four centimetres from the old outline.
                distance=cv2.distanceTransform((~old).astype('uint8'),cv2.DIST_L2,5)*pixel_m
                if (new&~old&(distance>.04)).any():continue
                bd=_boundary_distance(old,pixel_m);ad=_boundary_distance(new,pixel_m);lost=False
                for profile in profiles:
                    pts=profile['points'];inside=profile['valid']&((pts>=lo+3)&(pts<hi-3)).all(1)
                    if inside.any() and ((_sample(bd,pts[inside]-lo)<=.015)&(_sample(ad,pts[inside]-lo)>.015)).any():lost=True;break
                if lost:rejected['measured_face']=rejected.get('measured_face',0)+1;continue
                mask[sl]=new
                after=_profile(run,mask,forbidden,pixel_m,cfg)
                before_spread=float(np.percentile(widths,90)-np.percentile(widths,10));after_widths=after['width'][a:b]
                after_spread=float(np.percentile(after_widths,90)-np.percentile(after_widths,10))
                if after_spread>before_spread+pixel_m:
                    mask[sl]=old;continue
                edits.append(dict(length_m=float((t1-t0)*pixel_m),width_m=width,
                    width_spread_before_m=before_spread,width_spread_after_m=after_spread,
                    changed_pixels=int((old!=new).sum()),mode='paired_parallel_faces' if paired is not None else 'inferred_back_face',merged_scan_fragments=run.get('merged_fragments',1),anchor='observed scan face',opposite_side='inferred except protected dense faces',
                    line=dict(u=u.tolist(),normal=normal.tolist(),offset=run['offset'],t0=float(t0),t1=float(t1))))
    from .wall_face_alignment import align_observed_faces
    mask,face_alignment=align_observed_faces(mask,target,barrier,forbidden,raw_runs,pixel_m,cfg)
    final=[primitive(g,i) for i,g in enumerate(regions(mask_geometry(mask)))]
    assert np.array_equal(_union(final,shape),mask)
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense','world_regions']})
    out.update(primitives=final,mask=mask,contacts=_topology(final,shape),metrics=mask_metrics(target,mask),
        observed_face_alignment=face_alignment,
        thickness_regularization=dict(edits=edits,rejections=rejected,raw_scan_runs=len(raw_runs),assembled_scan_runs=len(runs),max_adjustment_m=.04,min_run_m=1.,joint_protection_m=.12))
    return out
