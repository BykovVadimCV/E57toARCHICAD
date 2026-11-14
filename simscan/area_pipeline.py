"""Fit saved native wall masks with the reviewed staged area vectorizer.

python -m simscan.area_pipeline --mask wall.png --barrier dense.png \
  --forbidden apertures.png --frame frame.json --out result
All PNGs must use exactly the same metric frame. The barrier must already be
density/height qualified; a raw visualization is not a valid evidence mask.
"""
from pathlib import Path
import argparse,json
import cv2,numpy as np
from .decompose import Config
from .wall_pipeline import fit_walls
from .netinput import affine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['mask','barrier','forbidden','frame','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--coverage',type=float,default=.95)
    p.add_argument('--native-counts',help='optional NPZ counts raster for final long-face gravity')
    p.add_argument('--probabilities',help='optional coarse 8-class probability NPZ, same extent as native frame')
    p.add_argument('--prob-pixel-mm',type=float,default=20.)
    a=p.parse_args()
    def read(path):
        im=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        if im is None:raise ValueError('Cannot read '+path)
        return im>0
    target,barrier,forbidden=read(a.mask),read(a.barrier),read(a.forbidden)
    meta=json.loads(Path(a.frame).read_text(encoding='utf-8'));frame=meta.get('frame',meta)
    if target.shape!=(frame['height'],frame['width']):raise ValueError('Frame dimensions differ')
    result=fit_walls(target,frame['pixel_m'],barrier=barrier,forbidden=forbidden,config=Config(coverage_stop=a.coverage,max_primitives=220))
    counts=None
    if a.native_counts:
        with np.load(a.native_counts,allow_pickle=False) as z:
            counts=z['counts'] if 'counts' in z else z['wall_height']
        if counts.shape!=target.shape or not np.isfinite(counts).all() or np.any(counts<0):raise ValueError('Count frame differs')
        from shapely.geometry import Polygon
        from .dense_face_fit import align_dense_faces
        from .wall_footprint import mask_geometry,primitive
        from .wall_runs import _union
        from .decompose import mask_metrics,_topology,rectangle_overlap_metrics
        from .wall_pipeline import audit
        px=frame['pixel_m'];yy,xx=np.nonzero(barrier&(counts>=200*(px/.02)**2))
        polygons=[Polygon(np.asarray(c['polygon'])*px,[np.asarray(h)*px for h in c.get('holes',[])]) for c in result['primitives']]
        import shapely.affinity
        excluded=shapely.affinity.scale(mask_geometry(forbidden),xfact=px,yfact=px,origin=(0,0))
        fitted,edits=align_dense_faces(polygons,np.c_[xx,yy]*px,excluded)
        proposed=[primitive(shapely.affinity.scale(g,xfact=1/px,yfact=1/px,origin=(0,0)),i) for i,g in enumerate(fitted)]
        mask=_union(proposed,target.shape)
        try:audit(mask,target,barrier,forbidden,px)
        except ValueError as exc:result['final_face_gravity']=dict(accepted=False,reason=str(exc),edits=edits)
        else:
            result.update(primitives=proposed,mask=mask,metrics=mask_metrics(target,mask),
                          contacts=_topology(proposed,target.shape),
                          rectangle_overlap=rectangle_overlap_metrics(proposed,target.shape,px))
            result['final_face_gravity']=dict(accepted=True,edits=edits,short_cap_lines_preserved=True)
    if a.probabilities:
        from .window_bridge import fit_windows
        from .door_fit import fit_doors
        with np.load(a.probabilities,allow_pickle=False) as z:prob=z['prob']
        windows=fit_windows(prob,result,result['mask'],frame['pixel_m'],a.prob_pixel_mm/1000)
        result.update(windows)
        doors=fit_doors(prob,result,result['mask'],frame['pixel_m'],a.prob_pixel_mm/1000,
                        windows=windows['windows'],barrier=barrier,counts=counts)
        result.update(doors)
    M=np.asarray(affine(frame))
    def world(q):q=np.asarray(q);return (M@np.c_[q,np.ones(len(q))].T).T.tolist()
    result.update(frame=frame,world_regions=[dict(id=c['id'],kind=c['kind'],footprint=world(c['polygon']),holes=[world(h) for h in c.get('holes',[])]) for c in result['primitives']])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/'walls.png'),result['mask'].astype('uint8')*255)
    def cast(v):
        if isinstance(v,np.ndarray):return v.tolist()
        if isinstance(v,np.generic):return v.item()
        raise TypeError(type(v).__name__)
    (out/'geometry.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['mask','dense','review_mask']},default=cast,indent=2),encoding='utf-8')
    print(json.dumps(result['metrics']))

if __name__=='__main__':main()
