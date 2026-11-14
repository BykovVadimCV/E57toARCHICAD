"""Render a native-resolution IFC/scan overlay and estimate wall-face proximity."""
from pathlib import Path
import argparse,csv,json
import cv2,ifcopenshell,numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from shapely.geometry import Polygon,Point
from shapely import union_all
from simscan.netinput import affine
from simscan.wall_constraints import dense_barriers


def run(ifc, geometry, image, counts_file, scan_file, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    doc=json.loads(Path(geometry).read_text());frame=doc['frame'];px=frame['pixel_m'];inv=np.linalg.inv(np.vstack([affine(frame),[0,0,1]]))
    f=ifcopenshell.open(str(ifc));canvas=np.array(Image.open(image).convert('RGB'));overlay=canvas.copy();walls=[];apertures=[]
    def polygon(entity):
        s=entity.Representation.Representations[0].Items[0];p=s.SweptArea
        return Polygon([v.Coordinates for v in p.OuterCurve.Points],[[v.Coordinates for v in h.Points] for h in getattr(p,'InnerCurves',[])])
    def ring_points(ring):
        a=np.c_[np.asarray(ring.coords),np.ones(len(ring.coords))];return np.rint((inv@a.T).T[:,:2]).astype('int32')
    def draw_poly(img,g,color,fill=False):
        rings=[g.exterior,*g.interiors]
        if fill:cv2.fillPoly(img,[ring_points(g.exterior)],color)
        else:
            for ring in rings:cv2.polylines(img,[ring_points(ring)],True,color,2,cv2.LINE_AA)
    for obj in f.by_type('IfcWall'):
        if obj.Name and obj.Name.startswith('Wall region'):
            g=polygon(obj);walls.append((obj.Name,g));draw_poly(overlay,g,(100,60,0),True)
    for kind in ['IfcDoor','IfcWindow']:
        for obj in f.by_type(kind):
            g=polygon(obj);apertures.append((kind,obj.Name,g));draw_poly(overlay,g,(255,85,70) if kind=='IfcDoor' else (0,200,255),True)
    canvas=cv2.addWeighted(overlay,.40,canvas,.60,0)
    for _,g in walls:draw_poly(canvas,g,(255,180,0))
    for kind,_,g in apertures:draw_poly(canvas,g,(255,95,80) if kind=='IfcDoor' else (0,220,255))
    legend=Image.fromarray(canvas);draw=ImageDraw.Draw(legend);draw.rectangle((10,10,720,85),fill=(8,8,8));draw.text((24,18),'IFC wall: gold     door: coral     window: cyan',fill='white');draw.text((24,46),'Native scan raster: 2.5 mm / pixel | all wall-height returns shown',fill='white')
    legend.save(out/'good_L1_ifc_cloud_overlay.png')
    counts=np.load(counts_file)['wall_height'];scan=np.load(scan_file)['input']
    barrier=dense_barriers(counts,scan);dense=(counts>=4)&barrier
    y,x=np.nonzero(dense);world=(np.asarray(affine(frame))@np.stack([x,y,np.ones(len(x))])).T
    tree=cKDTree(world);opening_union=union_all([g for _,_,g in apertures]);exclusion=opening_union.buffer(.05)
    distances=[];rows=[];excluded=0
    for wall_id,g in walls:
        for rid,ring in enumerate([g.exterior,*g.interiors]):
            coords=np.asarray(ring.coords)
            for seg,(a,b) in enumerate(zip(coords[:-1],coords[1:])):
                length=np.linalg.norm(b-a)
                if length<.10:continue
                n=max(1,int(np.ceil(length/.02)));steps=(np.arange(n)+.5)/n
                points=a[None,:]+steps[:,None]*(b-a)[None,:]
                valid=np.array([not exclusion.contains(Point(q)) for q in points]);excluded+=int((~valid).sum());points=points[valid]
                if not len(points):continue
                dist,_=tree.query(points);distances.extend(dist.tolist());matched=dist[dist<=.2]
                rows.append(dict(wall=wall_id,ring=rid,segment=seg,length_m=round(float(length),4),samples=len(dist),matched=len(matched),mean_matched_cm=round(float(matched.mean()*100),3) if len(matched) else '',unmatched_fraction=round(float((dist>.2).mean()),4)))
    values=np.asarray(distances);matched=values[values<=.2]
    report=dict(ifc=str(ifc),sample_spacing_m=.02,wall_segments_min_length_m=.1,dense_threshold='at least 4 returns per 2.5 mm pixel, equivalent to 256 per 2 cm pixel by area; vertical and connected-region gate',dense_points=len(world),scan_z_band='floor + 0.15 m to ceiling - 0.15 m',distance='2D Euclidean distance from sampled IFC wall footprint boundary to nearest qualifying occupied 2.5 mm scan bin center',opening_exclusion_m=.05,excluded_samples=excluded,wall_face_samples=len(values),search_range_m=.2,matched_samples=len(matched),matched_fraction=float(len(matched)/len(values)),mean_matched_cm=float(matched.mean()*100),median_matched_cm=float(np.median(matched)*100),p90_matched_cm=float(np.percentile(matched,90)*100),mean_all_samples_cm=float(values.mean()*100),median_all_samples_cm=float(np.median(values)*100),p90_all_samples_cm=float(np.percentile(values,90)*100),limitations='Nearest-return proximity, not signed perpendicular error or surveyed accuracy. One-sided walls, openings, occlusion, furniture and nearby parallel surfaces can affect the result. Samples farther than search range are reported separately, not dropped silently.')
    (out/'good_L1_ifc_cloud_metrics.json').write_text(json.dumps(report,indent=2));
    with (out/'good_L1_ifc_wall_segments.csv').open('w',newline='') as fp:
        writer=csv.DictWriter(fp,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ifc',required=True);p.add_argument('--geometry',required=True);p.add_argument('--image',required=True);p.add_argument('--counts',required=True);p.add_argument('--scan',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.ifc,a.geometry,a.image,a.counts,a.scan,a.out)
