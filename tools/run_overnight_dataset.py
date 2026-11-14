"""Resumable overnight generation: dependency checks, one-scene preflight, final audit.

Run from the installed source folder. The same command resumes completed scenes.
"""
from pathlib import Path
import argparse,importlib,json,os,sys,time
from simscan.config import load_config
from simscan.makedata import make_dataset,_done
from simscan.dataset_contract import atomic_json
from audit_hd_dataset import audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--config',default='configs/customer_like.yaml');p.add_argument('--count',type=int,default=1000)
    p.add_argument('--seed',type=int,default=31111);p.add_argument('--start',type=int,default=0)
    p.add_argument('--workers',type=int,default=0,help='0 chooses up to four workers from available RAM/CPUs');p.add_argument('--variants',type=int,default=1)
    p.add_argument('--smoke',action='store_true',help='one small actual raycast; diagnostic profile, separate output')
    a=p.parse_args()
    if a.count<1 or a.workers<0 or a.variants<1 or a.start<0:raise ValueError('Invalid run size')
    # Default dataset generation rasterizes in-memory returns and does not
    # serialize E57; avoid requiring its native binding for this path.
    versions={}
    for module in ['open3d','numpy','cv2','shapely','scipy']:
        imported=importlib.import_module(module);versions[module]=getattr(imported,'__version__','unknown')
    if not a.workers:
        import psutil
        available=psutil.virtual_memory().available/1024**3
        a.workers=max(1,min(4,(os.cpu_count() or 2)//2,int(available//12)))
        print(f'Available RAM {available:.1f} GiB; using {a.workers} generator workers',flush=True)
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True);lock=out/'generation.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise RuntimeError('Run already locked. If the old process has stopped, remove '+str(lock))
    with os.fdopen(fd,'w') as f:f.write(str(os.getpid()))
    try:
        cfg=load_config(a.config)
        if a.smoke:
            cfg.layout.source='simscan';cfg.layout.rooms=(2,2);cfg.layout.p_duplex=0
            cfg.layout.footprint_x_m=(6,6);cfg.layout.footprint_y_m=(5,5)
            cfg.scanner.angular_step_deg=.6;cfg.interior.p_bare=0.;cfg.export.preview=False
            a.count=1;a.workers=1
        atomic_json(out/'launch.json',dict(command=sys.argv,python=sys.version,versions=versions,
                    started=time.time(),smoke=a.smoke,workers=a.workers,logical_cpus=os.cpu_count()))
        args=dict(cfg=cfg,out_root=out,seed=a.seed,start=a.start,variants=a.variants,
                  hd_masks=True,hd_pixel_mm=2.,hd_min_size=(2048,1152))
        t=time.monotonic()
        first=make_dataset(count=1,workers=1,**args)
        if first['errors'] or not _done(out,f'scene_{a.start:05d}',a.variants,True,2.,(2048,1152)):
            raise RuntimeError('First-scene preflight failed. Inspect make_log.jsonl before the overnight run.')
        print('PREFLIGHT PASSED; first scene elapsed %.1f seconds' % (time.monotonic()-t),flush=True)
        print('Rough remaining estimate from first scene: %.1f hours; scene complexity and retries vary' % ((time.monotonic()-t)*max(0,a.count-1)/a.workers/3600),flush=True)
        result=make_dataset(count=a.count,workers=a.workers,**args)
        if result['errors']:raise RuntimeError('Generation ended with errors; inspect make_log.jsonl and rerun to retry missing scenes')
        report=audit(out,deep=True)
        print('COMPLETE '+json.dumps(report),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
