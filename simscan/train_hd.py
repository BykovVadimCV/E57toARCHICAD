"""Train the native 15-input HD refiner on generated HD levels.

python -m simscan.train_hd --data dataset --out runs/hd --epochs 30
This is a new refinement stage, not a legacy PlanNet checkpoint continuation.
"""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--crop',type=int,default=512)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--seed',type=int,default=31)
    p.add_argument('--patches-per-level',type=int,default=4);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from .hddata import HDPatches
    from .hdnet import HDRefiner,loss
    torch.manual_seed(a.seed)
    tr=HDPatches(a.data,'train',a.crop,a.patches_per_level,a.seed)
    va=HDPatches(a.data,'val',a.crop,a.patches_per_level,a.seed)
    if not np.isclose(tr.pixel_m,va.pixel_m):raise ValueError('Train/val HD pixel pitches differ')
    h=hashlib.sha256()
    for name in ['train','val']:
        h.update((Path(a.data)/'splits'/(name+'.txt')).read_bytes())
        for item in (tr.items if name=='train' else va.items):
            h.update((Path(a.data)/'scenes'/item/'hd_complete.json').read_bytes())
    source=hashlib.sha256()
    for name in ['hddata.py','hdnet.py','train_hd.py']:source.update(Path(__file__).with_name(name).read_bytes())
    identity=dict(format='hd-refiner-v1',dataset=h.hexdigest(),source=source.hexdigest(),channels=15,classes=6,pixel_m=tr.pixel_m,crop=a.crop,
                  patches_per_level=a.patches_per_level,seed=a.seed,batch=a.batch,lr=a.lr)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=HDRefiner().to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.0001)
    start=0;best=-1.
    if a.resume:
        ck=torch.load(out/'last.pt',map_location=device,weights_only=False)
        if ck['identity']!=identity:raise ValueError('HD training data or settings changed; use a new run')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['opt']);start=ck['epoch']+1;best=ck['best']
        torch.set_rng_state(ck['torch_rng'].cpu())
        if device.type=='cuda' and ck.get('cuda_rng') is not None:torch.cuda.set_rng_state_all(ck['cuda_rng'])
    elif (out/'last.pt').exists():raise ValueError('Existing run: use --resume or a fresh output')
    def loader(ds,shuffle):
        # Shuffle levels, keep their crops adjacent: decompress each native
        # frame once for several patches instead of once per random crop.
        order=torch.randperm(len(ds.items)).tolist() if shuffle else range(len(ds.items))
        indices=[i*ds.patches+k for i in order for k in range(ds.patches)]
        return DataLoader(ds,batch_size=a.batch,sampler=indices,num_workers=a.workers,pin_memory=device.type=='cuda')
    print(f'HD refiner: {device}; train {len(tr.items)}, val {len(va.items)} levels',flush=True)
    def batch_to(b):return {k:v.to(device) for k,v in b.items()}
    for epoch in range(start,a.epochs):
        tr.epoch=epoch;model.train();total=0.;batches=0
        for b in loader(tr,True):
            b=batch_to(b);opt.zero_grad(set_to_none=True);l=loss(model(b['input']),b)
            if not torch.isfinite(l):raise ValueError('Nonfinite HD loss')
            l.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(l.detach());batches+=1
        model.eval();intersection=union=0
        with torch.inference_mode():
            for b in loader(va,False):
                b=batch_to(b);pred=model(b['input'])['sem'].argmax(1);valid=b['valid'];truth=b['sem']==1;wall=pred==1
                intersection+=int((wall&truth&valid).sum());union+=int(((wall|truth)&valid).sum())
        score=intersection/union if union else None
        improved=score is not None and score>best
        if improved:best=score
        row=dict(epoch=epoch,loss=total/max(1,batches),val_patch_wall_iou=score,
                 validation='fixed uniform native crops; not full-level or real-scan evaluation')
        ck=dict(identity=identity,epoch=epoch,best=best,model=model.state_dict(),opt=opt.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        for name in (['last.pt','best.pt'] if improved else ['last.pt']):
            temporary=out/(name+'.tmp');torch.save(ck,temporary);os.replace(temporary,out/name)
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)

if __name__=='__main__':main()
