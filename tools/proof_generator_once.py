"""One bounded integration proof, using a NumPy triangle-ray test backend.

This does not benchmark Open3D. No production backend is changed on disk.
Run with --out NEW_DIRECTORY. Exactly one clean, single-room scene is generated.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simscan.config import load_config
from simscan.makedata import make_dataset,_done
from simscan.scene import SceneMesh
from simscan.scanner import ScanSimulator
from simscan.dataset_contract import complete_level
from simscan.torchdata import FloorplanSamples


def cast(self,mesh,origins,dirs):
    """Small exact Moller-Trumbore fixture; two-sided triangles, bounded rays."""
    o=np.broadcast_to(origins,dirs.shape)
    best=np.full(len(dirs),np.inf);ids=np.full(len(dirs),-1,dtype=np.int64)
    normal=np.zeros((len(dirs),3))
    for k,tri in enumerate(mesh.vertices[mesh.triangles].astype(float)):
        a,b,c=tri;e1=b-a;e2=c-a
        h=np.cross(dirs,e2);det=h@e1
        good=abs(det)>1e-10;inv=np.divide(1.,det,out=np.zeros_like(det),where=good)
        v=o-a;u=np.einsum('ij,ij->i',v,h)*inv
        q=np.cross(v,e1);w=np.einsum('ij,ij->i',dirs,q)*inv;t=(q@e2)*inv
        hit=good&(u>=-1e-8)&(w>=-1e-8)&(u+w<=1+1e-8)&(t>1e-6)&(t<best)
        best[hit]=t[hit];ids[hit]=k
        n=np.cross(e1,e2);normal[hit]=n/max(np.linalg.norm(n),1e-20)
    return best,ids,normal,np.isfinite(best)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('Proof output must be a new directory')
    cfg=load_config(overrides={
        'layout':{'source':'simscan','rooms':[1,1],'footprint_x_m':[5,5],'footprint_y_m':[4,4],
            'ceiling_height_m':[2.8,2.8],'p_l_shape':0,'p_entrance':0,'p_window_per_side':0,
            'p_suspended_ceiling':0},
        'interior':{'furniture':False,'radiators':False,'baseboards':False,'p_soffit':0,
            'p_column':0,'p_pilaster':0,'p_mirror_bath':0,'p_mirror_other':0,
            'p_curtains':0,'p_tulle':0,'p_risers':0,'lamps':False,'p_plants':0,
            'clutter_per_m2':0,'exterior_ground':False,'mess':[0,0]},
        'scanner':{'angular_step_deg':1.,'p_room_unscanned':0,'height_m':[1.5,1.5]},
        'effects':{'range_noise':False,'beam_model':False,'mixed_pixels':False,
            'grazing_dropout':False,'low_signal_dropout':False,'glass':False,
            'mirrors':False,'registration_error':False,'people':False},
        'realism':{'enabled':False},
        'export':{'world_yaw':False,'world_offset_m':[0,0],'world_z_offset_m':[0,0]}})
    old_scene=SceneMesh.to_raycasting_scene;old_cast=ScanSimulator._cast
    t=time.perf_counter()
    try:
        SceneMesh.to_raycasting_scene=lambda self:self
        ScanSimulator._cast=cast
        result=make_dataset(cfg,out,count=1,workers=1,seed=91,p_thin=0,p_shift=0)
    finally:
        SceneMesh.to_raycasting_scene=old_scene;ScanSimulator._cast=old_cast
    assert result['ok']==1 and not result['errors'],result
    assert _done(out,'scene_00000')
    level=out/'scenes/scene_00000/L1';assert complete_level(level)
    sc=json.loads((level/'scene.json').read_text())
    assert abs(sc['floor_z']-sc['floor_z_gt'])<.03
    assert abs(sc['ceiling_z']-sc['ceiling_z_gt'])<.03
    split=next(k for k,v in result['splits'].items() if v)
    sample=FloorplanSamples(out,split,crop=None,train=False,weighted_sampling=False)[0]
    assert (sample['sem'][~sample['supervision_valid']]==255).all()
    summary=dict(result=result,seconds=time.perf_counter()-t,
        ray_backend='NumPy triangle fixture; Open3D not tested',
        shape=list(sample['input'].shape),floor_error_m=sc['floor_z']-sc['floor_z_gt'],
        ceiling_error_m=sc['ceiling_z']-sc['ceiling_z_gt'],
        classes={str(k):int((sample['sem']==k).sum()) for k in np.unique(sample['sem'])})
    (out/'proof.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
