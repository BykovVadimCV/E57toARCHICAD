"""Fit existing long polygon faces to continuous, density-qualified scan lines."""
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,Point,LineString
from shapely import union_all


def align_dense_faces(polygons, points, forbidden, *, max_shift=.05, min_length=.8,
                      sample_bin=.1, band=.0075, support=.70, max_width_change=.30,
                      min_face_aspect=3.0):
    """Shift straight boundary edges only when a long, consistent scan face supports it.

    Coordinates and distances are metres. Returns polygons and an audit ledger.
    """
    points=np.asarray(points,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2:raise ValueError('points must be Nx2')
    if not len(points):return polygons,[]
    def local_width(g,p,n):
        cross=LineString([p-n*.75,p+n*.75]);part=g.intersection(cross)
        pieces=[part] if part.geom_type=='LineString' else list(getattr(part,'geoms',[]))
        pieces=[q for q in pieces if q.geom_type=='LineString' and q.length>1e-5]
        if not pieces:return None
        closest=min(pieces,key=lambda q:q.distance(Point(p)))
        return float(closest.length) if closest.distance(Point(p))<.07 else None
    tree=cKDTree(points);out=list(polygons);ledger=[]
    for index,poly in enumerate(polygons):
        coords=np.asarray(poly.exterior.coords[:-1],float).copy();holes=[list(r.coords) for r in poly.interiors]
        for i in range(len(coords)):
            j=(i+1)%len(coords);a,b=coords[i].copy(),coords[j].copy();length=np.linalg.norm(b-a)
            if length<min_length:continue
            u=(b-a)/length;v=np.array([-u[1],u[0]])
            # Classify by local wall width, not by an absolute edge length:
            # a 60 cm end cap can still be the short side of a broad wall.
            widths=[local_width(poly,a+t*(b-a),v) for t in np.linspace(.15,.85,7)]
            widths=[w for w in widths if w is not None and w>=.04]
            if len(widths)<4 or length < min_face_aspect*np.median(widths):continue
            # Translate the long face, but keep its endpoints on the original
            # adjacent edge lines. Thus a cap remains in its original plane.
            before=coords[(i-1)%len(coords)];after=coords[(j+1)%len(coords)]
            v0=a-before;v1=after-b
            if np.linalg.norm(v0)<1e-8 or np.linalg.norm(v1)<1e-8:continue
            v0/=np.linalg.norm(v0);v1/=np.linalg.norm(v1)
            d0=float(v0@v);d1=float(v1@v)
            if abs(d0)<.4 or abs(d1)<.4:continue
            ids=tree.query_ball_point((a+b)/2, length/2+max_shift+.02)
            p=points[ids];local=p-a;t=local@u;q=local@v
            eligible=(t>=.05)&(t<=length-.05)&(np.abs(q)<=max_shift)
            t,q=t[eligible],q[eligible]
            if len(q)<10:continue
            n=max(1,int(np.floor((length-.1)/sample_bin)));bins=np.clip(((t-.05)/(length-.1)*n).astype(int),0,n-1)
            shifts=np.arange(-max_shift,max_shift+.001,.0025)
            scores=np.array([len(np.unique(bins[np.abs(q-x)<=band]))/n for x in shifts])
            current=float(scores[np.argmin(abs(shifts))]);best=int(np.argmax(scores));candidate=float(shifts[best]);score=float(scores[best])
            if score<support or score-current<.12 or abs(candidate)<.0075:continue
            selected=np.abs(q-candidate)<=band
            offset=float(np.median(q[selected]))
            if abs(offset)>max_shift or abs(offset)<.0075:continue
            updated=coords.copy();updated[i]+=v0*(offset/d0);updated[j]+=v1*(offset/d1)
            new=Polygon(updated,holes)
            if not new.is_valid or new.is_empty:continue
            # Apertures are inviolate and connectivity among components cannot change.
            if abs(new.area-poly.area)>length*max_shift+.005:continue
            # An unobserved back face is not evidence that the wall disappears.
            # Preserve local thickness at the centre of the adjusted run.
            before=[];after=[]
            for t0 in np.linspace(.15,.85,7):
                point=a+t0*(b-a);w0=local_width(poly,point,v);w1=local_width(new,point,v)
                if w0 is not None and w1 is not None and w0>=.04:before.append(w0);after.append(w1)
            if len(before)<4:continue
            ratios=np.asarray(after)/np.asarray(before)
            if np.any((ratios<1-max_width_change)|(ratios>1+max_width_change)):continue
            others=[g for k,g in enumerate(out) if k!=index]
            if others and any(poly.intersects(o)!=new.intersects(o) for o in others):continue
            coords=updated;poly=new;out[index]=new
            ledger.append(dict(component=index,edge=i,length_m=float(length),median_width_m=float(np.median(widths)),
                               shift_m=offset,old_support=current,new_support=score,adjacent_cap_lines_preserved=True))
    if not forbidden.is_empty:
        clipped=[g.difference(forbidden) for g in out]
        if any(g.geom_type!='Polygon' or not g.is_valid for g in clipped):
            raise ValueError('Face alignment would split a wall at an opening')
        out=clipped
    return out,ledger
