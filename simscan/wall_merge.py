"""Lossless analytic unions of compatible wall rectangles; no boundary snapping.

Original rectangles remain as parts for exact native rasterization and provenance.
The visible/exported boundary is their analytic polygon union, not a bounding box.
"""
from __future__ import annotations
from copy import deepcopy
import numpy as np
import cv2


def _cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]

def _area(p):return float(_cross(p,np.roll(p,-1,axis=0)).sum()/2)


def polygon_mask(polygon,holes,lo,shape):
    """Scan-convert unrounded vector boundaries at native pixel centres."""
    rings=[np.asarray(p,float) for p in [polygon]+list(holes)]
    a=np.vstack(rings);b=np.vstack([np.roll(p,-1,axis=0) for p in rings]);d=b-a
    output=np.zeros(shape,bool);eps=1e-6;x0,y0=lo
    def fill(row,left,right):
        start=max(0,int(np.ceil(left-x0-eps)));stop=min(shape[1],int(np.floor(right-x0+eps))+1)
        if stop>start:row[start:stop]=True
    for row_index in range(shape[0]):
        y=y0+row_index;row=output[row_index]
        active=((a[:,1]<=y)&(b[:,1]>y))|((b[:,1]<=y)&(a[:,1]>y))
        xs=np.sort(a[active,0]+(y-a[active,1])*d[active,0]/d[active,1])
        for left,right in zip(xs[::2],xs[1::2]):fill(row,left,right)
        horizontal=(np.abs(d[:,1])<eps)&(np.abs(a[:,1]-y)<=eps)
        for left,right in zip(np.minimum(a[horizontal,0],b[horizontal,0]),np.maximum(a[horizontal,0],b[horizontal,0])):fill(row,left,right)
        # End vertices and diagonal boundary pixels belong to the wall too.
        closed=(np.abs(d[:,1])>=eps)&(y>=np.minimum(a[:,1],b[:,1])-eps)&(y<=np.maximum(a[:,1],b[:,1])+eps)
        xx=a[closed,0]+(y-a[closed,1])*d[closed,0]/d[closed,1]
        for x in xx:
            pixel=int(round(x-x0))
            if 0<=pixel<shape[1] and abs(x-x0-pixel)<=eps:row[pixel]=True
    return output


def polygon_union(polygons):
    """Boundary arrangement of convex CCW polygons; return one outer ring + holes.

    Split edges at crossings and collinear endpoints; retain only edges with
    union interior on one side. Disconnected or point-touching unions are refused.
    """
    polygons=[np.asarray(p,float) for p in polygons]
    polygons=[p if _area(p)>0 else p[::-1] for p in polygons]
    edges=[(a,b) for p in polygons for a,b in zip(p,np.roll(p,-1,axis=0))]
    def contains(point):
        return any(np.all(_cross(np.roll(p,-1,axis=0)-p,point-p)>=-1e-9) for p in polygons)
    segments={};scale=1e7
    def key(p):return tuple(np.rint(p*scale).astype(np.int64))
    for a,b in edges:
        v=b-a;vv=float(v@v)
        if vv<1e-16:continue
        cuts=[0.,1.]
        for c,d in edges:
            w=d-c;den=float(_cross(v,w))
            if abs(den)>1e-10:
                t=float(_cross(c-a,w)/den);s=float(_cross(c-a,v)/den)
                if -1e-9<=t<=1+1e-9 and -1e-9<=s<=1+1e-9:cuts.append(float(np.clip(t,0,1)))
            elif abs(float(_cross(c-a,v)))<=1e-7*np.sqrt(vv):
                cuts.extend(float(np.clip((p-a)@v/vv,0,1)) for p in [c,d])
        cuts=sorted(set(round(t,12) for t in cuts));normal=np.array([-v[1],v[0]])/np.sqrt(vv)
        for t0,t1 in zip(cuts,cuts[1:]):
            if (t1-t0)*np.sqrt(vv)<1e-7:continue
            start=a+v*t0;end=a+v*t1;mid=(start+end)/2
            left=contains(mid+normal*1e-5);right=contains(mid-normal*1e-5)
            if left==right:continue
            if right:start,end=end,start
            segments[(key(start),key(end))]=(start,end)
    outgoing={}
    for k in segments:outgoing.setdefault(k[0],[]).append(k)
    # Multi-way vertices indicate disconnected/point-touching geometry. Keep
    # those pieces separate rather than inventing a bridge through the vertex.
    if any(len(v)!=1 for v in outgoing.values()):return None
    remaining=set(segments);rings=[]
    while remaining:
        first=min(remaining);edge=first;ring=[]
        for _ in range(len(segments)+1):
            if edge not in remaining:return None
            remaining.remove(edge);ring.append(segments[edge][0]);end=edge[1]
            if end==first[0]:break
            if end not in outgoing:return None
            edge=outgoing[end][0]
        else:return None
        p=np.array(ring)
        # Drop only analytically collinear vertices, not genuine small steps.
        changed=True
        while changed and len(p)>3:
            before=np.roll(p,1,axis=0);after=np.roll(p,-1,axis=0)
            keep=np.abs(_cross(p-before,after-p))>1e-7
            changed=not keep.all()
            if keep.sum()<3:break
            p=p[keep]
        rings.append(p)
    outer=[p for p in rings if _area(p)>1e-7];holes=[p for p in rings if _area(p)<-1e-7]
    if len(outer)!=1:return None
    return dict(polygon=outer[0].tolist(),holes=[p.tolist() for p in holes])


def _compatible(a,b,px,angle_tol,width_tol,offset_tol):
    da=abs((a['angle']-b['angle']+np.pi/2)%np.pi-np.pi/2)
    if da>np.radians(angle_tol) or abs(a['width']-b['width'])*px>width_tol:return False
    u=np.array([np.cos(a['angle']),np.sin(a['angle'])]);v=np.array([-u[1],u[0]])
    # Compare near the meeting ends, not at distant centers of rotated pieces.
    centers=np.array([a['center'],b['center']]);joint=centers.mean(0)
    normals=[]
    for c in [a,b]:
        n=np.array([-np.sin(c['angle']),np.cos(c['angle'])]);d=np.array([np.cos(c['angle']),np.sin(c['angle'])])
        t=np.clip((joint-np.array(c['center']))@d,-c['length']/2,c['length']/2)
        normals.append((np.array(c['center'])+d*t)@v)
    if abs(normals[0]-normals[1])*px>offset_tol:return False
    pa=np.array(a['polygon'],np.float32);pb=np.array(b['polygon'],np.float32)
    area,_=cv2.intersectConvexConvex(pa,pb)
    if area>1e-5:return True
    # Shared edges are valid; a small real gap or mere point contact is not.
    for x,y in zip(pa,np.roll(pa,-1,axis=0)):
        d=y-x;length=np.linalg.norm(d)
        if length<1e-7:continue
        n=np.array([-d[1],d[0]])/length
        for s,t in zip(pb,np.roll(pb,-1,axis=0)):
            if max(abs((s-x)@n),abs((t-x)@n))>1e-4:continue
            interval=np.sort(np.array([(s-x)@d,(t-x)@d])/length)
            if min(length,interval[1])-max(0,interval[0])>1e-4:return True
    return False


def merge_primitives(primitives,shape,pixel_m=.0025,angle_degrees=1.,width_tolerance_m=.005,offset_tolerance_m=.01):
    """Group same-run rectangles; keep width steps, junction branches and arcs."""
    from .decompose import _local_mask,_bounds
    if pixel_m<=0 or not 0<=angle_degrees<45 or width_tolerance_m<0 or offset_tolerance_m<0:raise ValueError('Invalid merge tolerances')
    parts=deepcopy(primitives);source_ids={id(p):p.get('id',i) for i,p in enumerate(parts)};groups=[[i] for i in range(len(parts))];merges=[]
    changed=True
    while changed:
        changed=False
        for gi,group in enumerate(groups):
            if any(parts[i]['kind']!='rectangle' for i in group):continue
            for gj in range(gi+1,len(groups)):
                other=groups[gj]
                if any(parts[i]['kind']!='rectangle' for i in other):continue
                all_ids=group+other;anchor=max(all_ids,key=lambda i:parts[i]['length'])
                if any(abs((parts[i]['angle']-parts[anchor]['angle']+np.pi/2)%np.pi-np.pi/2)>np.radians(angle_degrees) for i in all_ids):continue
                if (max(parts[i]['width'] for i in all_ids)-min(parts[i]['width'] for i in all_ids))*pixel_m>width_tolerance_m:continue
                if not any(_compatible(parts[i],parts[j],pixel_m,angle_degrees,width_tolerance_m,offset_tolerance_m) for i in group for j in other):continue
                boundary=polygon_union([parts[i]['polygon'] for i in all_ids])
                if boundary is None:continue
                c=dict(kind='wall_run',**boundary,parts=[parts[i] for i in all_ids])
                lo,hi=_bounds(c,shape);local_shape=tuple((hi-lo)[::-1]);original=np.zeros(local_shape,bool)
                for i in all_ids:original|=_local_mask(parts[i],lo,local_shape)
                # Independent polygon membership must match analytic parts at
                # native centers; cv2's rounded polygon filling is not used.
                # Validate output union through its authoritative analytic parts.
                if not np.array_equal(original,_local_mask(c,lo,local_shape)):continue
                groups[gi]=all_ids;groups.pop(gj);merges.append(all_ids.copy());changed=True;break
            if changed:break
    output=[]
    for group in groups:
        if len(group)==1:c=parts[group[0]]
        else:
            boundary=polygon_union([parts[i]['polygon'] for i in group]);anchor=max(group,key=lambda i:parts[i]['length']);angle=parts[anchor]['angle'];u=np.array([np.cos(angle),np.sin(angle)]);v=np.array([-u[1],u[0]])
            corners=np.vstack([parts[i]['polygon'] for i in group]);along=corners@u;cross=corners@v;center=u*(along.min()+along.max())/2+v*(cross.min()+cross.max())/2
            c=dict(kind='wall_run',**boundary,parts=[parts[i] for i in group],merged_from=[parts[i].get('id',i) for i in group],
                   center=center.tolist(),angle=angle,length=float(np.ptp(along)),width=float(np.median([parts[i]['width'] for i in group])),
                   width_range_m=[min(parts[i]['width'] for i in group)*pixel_m,max(parts[i]['width'] for i in group)*pixel_m],
                   axis=[(center-u*np.ptp(along)/2).tolist(),(center+u*np.ptp(along)/2).tolist()],axis_role='representative',
                   scan_verified=all(parts[i].get('scan_verified',False) for i in group))
        c['id']=len(output);output.append(c)
    # Small crosswise tiles can lie wholly inside an already merged wall.
    # Absorb those as provenance too; a genuine protruding T/L branch survives.
    absorbed=0
    for wall in list(output):
        if wall['kind']!='wall_run':continue
        boundary=np.array(wall['polygon']);lo,hi=_bounds(wall,shape);local_shape=tuple((hi-lo)[::-1]);body=_local_mask(wall,lo,local_shape)
        area=abs(_area(boundary))-sum(abs(_area(np.asarray(h))) for h in wall.get('holes',[]))
        for other in list(output):
            if other is wall or other['kind']!='rectangle':continue
            b=np.array(other['polygon'])
            if np.any(b.min(0)<boundary.min(0)-1e-7) or np.any(b.max(0)>boundary.max(0)+1e-7):continue
            tile=_local_mask(other,lo,local_shape)
            if (tile&~body).any():continue
            combined=polygon_union([p['polygon'] for p in wall['parts']]+[other['polygon']])
            if combined is None:continue
            combined_area=abs(_area(np.asarray(combined['polygon'])))-sum(abs(_area(np.asarray(h))) for h in combined['holes'])
            if abs(combined_area-area)>1e-5:continue
            original=deepcopy(other);original['component_role']='contained_tile'
            wall['parts'].append(original);wall['merged_from'].append(source_ids[id(other)])
            output.remove(other);absorbed+=1
    for i,c in enumerate(output):c['id']=i
    return output,dict(before=len(primitives),after=len(output),merged_runs=sum(c['kind']=='wall_run' for c in output),absorbed_redundant_tiles=absorbed,
                       removed_seams=len(primitives)-len(output),native_mask_policy='exact analytic component union; no snapping',
                       angle_degrees=angle_degrees,width_tolerance_m=width_tolerance_m,offset_tolerance_m=offset_tolerance_m)


def merge_result(result,pixel_m=.0025,angle_degrees=1.,width_tolerance_m=.005,offset_tolerance_m=.01):
    from .decompose import _topology,rectangle_overlap_metrics
    out=deepcopy({k:v for k,v in result.items() if k not in ['mask','dense']})
    shape=result['mask'].shape
    primitives,stats=merge_primitives(result['primitives'],shape,pixel_m,angle_degrees,width_tolerance_m,offset_tolerance_m)
    out.update(primitives=primitives,contacts=_topology(primitives,shape),merging=stats,mask=result['mask'])
    out['metrics']=dict(out.get('metrics',{}),pieces=len(primitives),kinds={k:sum(c['kind']==k for c in primitives) for k in ['rectangle','arc','junction','wall_run']})
    if 'dense' in result:out['dense']=result['dense']
    out.pop('world_regions',None)
    if 'pixel_to_world' in out:
        matrix=np.asarray(out['pixel_to_world'])
        def world(points):return (np.c_[points,np.ones(len(points))]@matrix.T).tolist()
        out['world_regions']=[dict(id=c['id'],kind=c['kind'],footprint=world(c['polygon']),holes=[world(h) for h in c.get('holes',[])],
                                  axis=world(c['axis']) if 'axis' in c else None,mask_width_m=c.get('width',0)*pixel_m) for c in primitives]
    return out
