"""Associate exposed wall edges with continuous dense scan faces, including short runs.

A single observed face can fix position and direction. It cannot measure the
unobserved wall back: retain the existing body's median cross-section width.
"""
import cv2
import numpy as np
from .wall_runs import _sample, _intervals, _boundary_distance
from .decompose import _connectivity_breaks


def align_observed_faces(mask, target, barrier, forbidden, runs, pixel_m, cfg):
    mask=mask.copy();edits=[];rejected={}
    def reject(reason):rejected[reason]=rejected.get(reason,0)+1
    # Keep independent raw faces as guards, rather than only merged anchors.
    guard_points=[]
    for r in runs:
        t=np.arange(r['t0'],r['t1'],max(1,.01/pixel_m))
        guard_points.append(t[:,None]*r['u']+r['offset']*r['normal'])
    guards=np.concatenate(guard_points) if guard_points else np.empty((0,2))
    for r in runs:
        u=r['u'];v=r['normal'];offset=r['offset']
        # Small endpoint extrapolations retain the old cap/junction extent;
        # every edited cross-section still needs measured scan support.
        step=max(1,int(round(.01/pixel_m)));ts=np.arange(r['t0']-.05/pixel_m,r['t1']+.05/pixel_m,step)
        points=ts[:,None]*u+offset*v
        qs=np.arange(-int(np.ceil(cfg.run_max_width_m/pixel_m)),int(np.ceil(cfg.run_max_width_m/pixel_m))+1)
        xy=points[:,None,:]+qs[None,:,None]*v
        bodies=_sample(mask,xy.reshape(-1,2)).reshape(len(ts),-1)
        semantics=_sample(target,xy.reshape(-1,2)).reshape(len(ts),-1)
        # A crossing cloud blob cannot own a wall face: require a continuous
        # near-line return in longitudinal bins, each bin contributing one vote.
        face_q=np.arange(-max(1,int(round(.0075/pixel_m))),max(1,int(round(.0075/pixel_m)))+1)
        face_xy=points[:,None,:]+face_q[None,:,None]*v
        support=_sample(barrier,face_xy.reshape(-1,2)).reshape(len(ts),-1).any(1)
        widths=np.zeros(len(ts));sides=np.zeros(len(ts));near=np.zeros(len(ts));valid=np.zeros(len(ts),bool)
        for i,(body,semantic) in enumerate(zip(bodies,semantics)):
            intervals=_intervals(body)
            if not intervals:continue
            a,b=min(intervals,key=lambda ab:min(abs(qs[ab[0]]),abs(qs[ab[1]-1])))
            low,high=qs[a]*pixel_m,qs[b-1]*pixel_m
            # The line must remain on one exposed edge, never through the body.
            side=1 if abs(low)<=abs(high) else -1
            edge=low if side==1 else high
            if min(abs(low),abs(high))>.06 or low<-.015 and high>.015:continue
            width=(b-a)*pixel_m
            if not cfg.min_width_m<=width<=cfg.run_max_width_m:continue
            # Local model evidence must belong to this same side of the scan.
            qbody=(qs*pixel_m*side>=.005)&(qs*pixel_m*side<=width)
            if semantic[qbody].mean()<.65:continue
            widths[i]=width;sides[i]=side;near[i]=edge;valid[i]=support[i]
        for side in [-1,1]:
            good=valid&(sides==side)
            for a,b in _intervals(good):
                length=(b-a)*step*pixel_m
                if length<cfg.run_min_m:continue
                width=float(np.median(widths[a:b]))
                if np.percentile(widths[a:b],90)-np.percentile(widths[a:b],10)>.02:continue
                if np.percentile(np.abs(near[a:b]),90)<=.0125:continue
                t0=ts[a];t1=ts[b-1]+step/2
                corners=np.array([u*t+v*(offset+side*q/pixel_m) for t in [t0,t1] for q in [-.065,width+.065]])
                lo=np.maximum(0,np.floor(corners.min(0)-3).astype(int));hi=np.minimum(mask.shape[::-1],np.ceil(corners.max(0)+4).astype(int));sl=(slice(lo[1],hi[1]),slice(lo[0],hi[0]))
                old=mask[sl].copy();yy,xx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
                along=xx*u[0]+yy*u[1];q=(xx*v[0]+yy*v[1]-offset)*side*pixel_m
                editable=(along>=t0)&(along<=t1)&(q>=-.06)&(q<=width+.06)
                # A robust line passes through the return band, whose thickness
                # includes sampling noise. Use its material-side surface rather
                # than burying that band behind the fitted boundary.
                face=barrier[sl]&(along>=t0)&(along<=t1)&(np.abs(q)<=.015)
                front=float(np.clip(np.percentile(q[face],95),-.01,.01)) if face.any() else 0.
                desired=(q>=front)&(q<=front+width)
                new=old.copy();new[editable]=(desired&~forbidden[sl])[editable]
                if np.array_equal(old,new):continue
                if (new&forbidden[sl]).any():reject('aperture');continue
                if _connectivity_breaks(old,new):reject('connection');continue
                dist=cv2.distanceTransform((~old).astype('uint8'),cv2.DIST_L2,5)*pixel_m
                if (new&~old&(dist>.06)).any():reject('shift');continue
                old_depth=cv2.distanceTransform(np.pad(old.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
                depth=cv2.distanceTransform(np.pad(new.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
                if (new&~target[sl]&barrier[sl]&(depth>cfg.barrier_contact_m)).any():reject('dense_crossing');continue
                buried=barrier[sl]&old&(old_depth<=cfg.run_face_tolerance_m)&(depth>cfg.run_face_tolerance_m)
                if buried.sum()>max(2,.001*barrier[sl].sum()):reject('buried_face');continue
                bd=_boundary_distance(old,pixel_m);ad=_boundary_distance(new,pixel_m)
                inside=((guards>=lo+2)&(guards<hi-2)).all(1);local=guards[inside]-lo
                if ((_sample(bd,local)<=cfg.run_face_tolerance_m)&(_sample(ad,local)>cfg.run_face_tolerance_m)).any():reject('measured_face');continue
                local=points[a:b]-lo
                before=_sample(bd,local);after=_sample(ad,local)
                if np.median(before-after)<.005 or np.percentile(after,90)>.0075:reject('no_face_gain');continue
                mask[sl]=new
                edits.append(dict(length_m=length,width_m=width,changed_pixels=int((old!=new).sum()),
                    face_distance_p90_before_m=float(np.percentile(before,90)),face_distance_p90_after_m=float(np.percentile(after,90)),
                    anchor='continuous dense scan face',opposite_side='existing median width; unobserved',
                    surface_inset_m=front,
                    line=dict(u=u.tolist(),normal=v.tolist(),offset=float(offset),t0=float(t0),t1=float(t1))))
    return mask,dict(edits=edits,rejections=rejected,max_face_shift_m=.06,min_run_m=cfg.run_min_m)
