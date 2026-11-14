"""Drive checkpoints -> three frozen real scan levels -> native wall review.

No generation, training, or E57 decoding. Run via the companion Colab notebook.
Historical vectors are a previous-review reference, not a controlled weights ablation.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, time, zipfile, gc
from dataclasses import asdict
import numpy as np
import cv2
from PIL import Image
from simscan.decompose import Config, decompose, rectangle_overlap_metrics
from simscan.mask_refine import Config as RefineConfig, refine, refine_high_resolution
from simscan.wall_constraints import wall_only, dense_barriers
from simscan.wall_runs import _union, extract_runs, _profile, face_metrics
from simscan.wall_pipeline import fit_walls, PIPELINE_VERSION, STAGE_ORDER
from simscan.netinput import affine
from simscan.window_bridge import fit_windows
from simscan.door_fit import fit_doors
import build_overlap_viewer as viewer

CASES = ['good/L1', 'medium/L1', 'medium/L2']
MODELS = {'new_wall': 'best_wall.pt', 'new_plan': 'best.pt'}
OLD_HASHES = {'3b1a081b78153dca0bec5d73584ed9dad7731b9637e381e41629bca6ea348ab1',
              'f950fb8dba258197b9f740d3cf8896d20fd709a26f2dfb16cc66fd97fa7b761a'}
REFINE = RefineConfig(smoothness=48., prior_weight=.08, boundary_attraction=.995,
    band_m=.16, influence_m=.10, boundary_locality_m=.05,
    boundary_ownership=True, preserve_topology=False)
DECOMP = Config(max_primitives=220, coverage_stop=.95)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def serial(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, np.generic): return v.item()
    raise TypeError(type(v).__name__)


def write(path, doc):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(doc, default=serial, indent=2), encoding='utf-8')
    tmp.replace(path)


def extract_checked(archive, dest):
    """Only confined, ordinary ZIP members; never extract absolute/traversal names."""
    dest = Path(dest).resolve(); dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for i in z.infolist():
            p = (dest/i.filename).resolve()
            if not p.is_relative_to(dest) or '\\' in i.filename:
                raise ValueError(f'Unsafe ZIP member: {i.filename}')
        z.extractall(dest)


def checkpoints(project, run_name, cache, allow_old=False):
    root = Path(project)/'runs'/run_name
    files = {k: root/v for k,v in MODELS.items()}
    if not all(p.is_file() for p in files.values()):
        archive = Path(project)/(run_name+'_results.zip')
        if not archive.is_file():
            raise FileNotFoundError(f'Need best_wall.pt and best.pt in {root}, or {archive}')
        with zipfile.ZipFile(archive) as z:
            for k, name in MODELS.items():
                if files[k].is_file(): continue
                matches = [n for n in z.namelist() if Path(n).name == name]
                if len(matches) != 1: raise ValueError(f'Ambiguous/missing {name} in {archive}')
                p = Path(cache)/name; p.parent.mkdir(parents=True, exist_ok=True)
                with z.open(matches[0]) as src, p.open('wb') as dst: shutil.copyfileobj(src, dst)
                files[k] = p
    hashes = {k: sha(p) for k,p in files.items()}
    if not allow_old and any(h in OLD_HASHES for h in hashes.values()):
        raise ValueError('An old review checkpoint was selected. Set TRAIN_RUN to the new run; use --allow-old only for deliberate regression tests.')
    # Snapshot weights locally: concurrent training cannot change the selected files mid-test.
    for k,p in list(files.items()):
        dst = Path(cache)/hashes[k]/p.name; dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst)
        if sha(dst) != hashes[k]: raise ValueError('Checkpoint changed during copy; retry after saving finishes')
        files[k] = dst
    return files, hashes


def verify_assets(root):
    manifest = json.loads((root/'assets_manifest.json').read_text())
    for rel, expected in manifest['sha256'].items():
        if sha(root/rel) != expected: raise ValueError(f'Input archive integrity error: {rel}')
    for key in CASES:
        x = np.load(root/'inputs'/key/'input.npy', mmap_mode='r')
        with np.load(root/'review/input_resolution_10mm'/key/'input_10mm.npz') as a: s = a['input'].shape
        with np.load(root/'review/wall_point_overlay'/key/'cloud_counts.npz') as a: c = a['wall_height'].shape
        scene = json.loads((root/'inputs'/key/'scene.json').read_text())
        assert x.shape[0] == s[0] == 8 and s[1:] == tuple(n*2 for n in x.shape[1:])
        assert c == tuple(n*8 for n in x.shape[1:]), (key, x.shape, s, c)
        assert np.isclose(scene['frame']['pixel_m'], .02)
    return manifest


def rgba(path, mask, color):
    a = np.zeros((*mask.shape,4), np.uint8); a[mask] = color
    Image.fromarray(a).save(path)


def metrics(mask, target):
    inter = int((mask & target).sum()); union = int((mask | target).sum())
    return dict(iou=inter/max(1,union), recall=inter/max(1,int(target.sum())),
                precision=inter/max(1,int(mask.sum())), area_m2=float(mask.sum())*.0025**2)


def process(root, out, key, model, prob, prepared_target=None):
    start = time.monotonic(); folder = out/key; folder.mkdir(parents=True, exist_ok=True)
    baseline = root/'baseline'/key
    with np.load(root/'review/input_resolution_10mm'/key/'input_10mm.npz') as a: scan = a['input']
    with np.load(root/'review/wall_point_overlay'/key/'cloud_counts.npz') as a: counts = a['wall_height']
    h,w = counts.shape; wall = wall_only(prob, include_windows=False)
    resize = lambda a, shape, method: cv2.resize(a, shape, interpolation=method)
    forbidden = resize(wall['forbidden'].astype('uint8'), (w,h), cv2.INTER_NEAREST)>0
    forbidden10 = resize(wall['forbidden'].astype('uint8'), scan.shape[:0:-1], cv2.INTER_NEAREST)>0
    barrier = dense_barriers(counts, scan)
    if prepared_target is None:
        seed = resize(wall['mask'].astype('uint8'), scan.shape[:0:-1], cv2.INTER_NEAREST)>0
        p10 = resize(wall['probability'], scan.shape[:0:-1], cv2.INTER_LINEAR)
        seed = refine(seed, p10, scan, forbidden=forbidden10)['mask']
        r = refine_high_resolution(seed, wall['probability'], scan, counts,
            config=REFINE, forbidden=forbidden, log=lambda s: print(key, model, s, flush=True))
        target = r['mask']; refinement_metrics = r['metrics']; del r, seed, p10
    else:
        target=np.array(Image.open(prepared_target))>0
        if target.shape != (h,w): raise ValueError('Saved refinement frame differs')
        refinement_metrics=dict(reused_refined_mask=str(prepared_target),sha256=sha(prepared_target))
    assert not (target & forbidden).any()
    del forbidden10, scan
    density = np.clip(np.log1p(counts)/np.log1p(4000),0,1).astype(np.float32); del counts
    p = resize(wall['probability'], (w,h), cv2.INTER_LINEAR)
    def save_stage(name, result, info):
        stage_folder=folder/'stages'/model/name
        stage_folder.mkdir(parents=True,exist_ok=True)
        Image.fromarray(result['mask'].astype('uint8')*255).save(stage_folder/'mask.png')
        if 'review_mask' in result:Image.fromarray(result['review_mask'].astype('uint8')*255).save(stage_folder/'review_mask.png')
        write(stage_folder/'geometry.json',{k:v for k,v in result.items() if k not in ['mask','dense','review_mask']})
        write(stage_folder/'metrics.json',info)
        print(key,model,name,'pieces',info['pieces'],'recall',round(info['metrics']['recall'],4),flush=True)
    accepted_reference=None
    accepted_folder=root/'accepted_reference'/key
    if (accepted_folder/(model+'_evidence.json')).exists():
        from simscan.wall_reference_guard import barrier_sha256
        evidence=json.loads((accepted_folder/(model+'_evidence.json')).read_text())
        rt=np.array(Image.open(accepted_folder/(model+'_mask_2p5mm.png')))>0
        rf=np.array(Image.open(accepted_folder/(model+'_excluded.png')))[...,3]>0
        if np.array_equal(rt,target) and np.array_equal(rf,forbidden) and evidence['barrier_sha256']==barrier_sha256(barrier):
            accepted=json.loads((accepted_folder/(model+'_decomposition.json')).read_text())
            accepted['mask']=np.array(Image.open(accepted_folder/(model+'_vector_mask.png')))>0
            rp=accepted_folder/(model+'_review.png')
            if rp.exists():accepted['review_mask']=np.array(Image.open(rp))[...,3]>0
            accepted_reference=dict(result=accepted,target=rt,forbidden=rf,pixel_m=.0025,barrier_sha256=evidence['barrier_sha256'],source='accepted same-input native fit')
    d = fit_walls(target, .0025, semantic_probability=p, density=density,
                  barrier=barrier, forbidden=forbidden, config=DECOMP,on_stage=save_stage,accepted_reference=accepted_reference)
    del density, p
    if 'review_mask' in d:rgba(folder/(model+'_review.png'),d['review_mask'],[255,110,20,255])
    mask = d['mask']
    prior_thickness=folder/'stages'/model/'11_native_reference_guard/mask.png'
    if prior_thickness.exists():
        prior_mask=np.array(Image.open(prior_thickness))>0
        rgba(folder/(model+'_small_feature_mask.png'),mask&~prior_mask,[30,240,100,255])
        del prior_mask
    assert np.array_equal(_union(d['primitives'], mask.shape), mask), 'Export geometry does not reconstruct the mask'
    depth = cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*.0025
    audit = dict(excluded_pixels_filled=int((mask & forbidden).sum()),
        dense_barrier_crossing_pixels=int((mask & ~target & barrier & (depth>.005)).sum()))
    del depth
    if any(audit.values()): raise AssertionError(audit)
    scene = json.loads((root/'inputs'/key/'scene.json').read_text())
    frame = dict(scene['frame'],pixel_m=.0025,width=w,height=h); M = np.asarray(affine(frame))
    def world(points):
        q=np.asarray(points); return (M@np.c_[q,np.ones(len(q))].T).T.tolist()
    d.update(frame=frame,pixel_to_world=M.tolist(),world_regions=[dict(id=c['id'],kind=c['kind'],
        footprint=world(c['polygon']),holes=[world(v) for v in c.get('holes',[])]) for c in d['primitives']])
    window_result=fit_windows(prob,d,mask)
    for window in window_result['windows']:
        window['world_axis']=world(window['axis'])
        window['world_regions']=[dict(footprint=world(g['polygon']),holes=[world(h) for h in g['holes']]) for g in window['footprint_regions']]
    d.update(window_result)
    write(folder/(model+'_windows.json'),dict(window_result,frame=frame))
    rgba(folder/(model+'_window_mask.png'),prob.argmax(0)==4,[30,180,255,230])
    with np.load(root/'review/wall_point_overlay'/key/'cloud_counts.npz') as source_counts:
        door_counts=source_counts['wall_height']
    door_result=fit_doors(prob,d,mask,windows=window_result['windows'],barrier=barrier,counts=door_counts)
    del door_counts
    for door in door_result['doors']:
        door['world_axis']=world(door['axis'])
        door['world_regions']=[dict(footprint=world(g['polygon']),holes=[world(h) for h in g['holes']]) for g in door['footprint_regions']]
    d.update(doors=door_result['doors'],door_review=door_result['door_review'],
        door_fitting={k:v for k,v in door_result.items() if k not in ['doors','door_review']})
    write(folder/(model+'_doors.json'),dict(door_result,frame=frame))
    rgba(folder/(model+'_door_mask.png'),prob.argmax(0)==3,[255,100,70,230])
    write(folder/(model+'_floorplan.json'),dict(frame=frame,wall_regions=d['world_regions'],
        windows=d['windows'],doors=d['doors'],window_review=d['window_review'],door_review=d['door_review']))
    d['metrics'].update(metrics(mask,target), pieces=len(d['primitives']), seconds=time.monotonic()-start)
    d['audit'] = audit
    write(folder/(model+'_decomposition.json'),{k:v for k,v in d.items() if k not in ['mask','dense','review_mask']})
    Image.fromarray(mask.astype('uint8')*255).save(folder/(model+'_vector_mask.png'))
    Image.fromarray(target.astype('uint8')*255).save(folder/(model+'_mask_2p5mm.png'))
    rgba(folder/(model+'_original.png'),wall['mask'],[255,190,0,255])
    rgba(folder/(model+'_refined.png'),target,[0,210,255,255])
    rgba(folder/(model+'_excluded.png'),forbidden,[210,70,255,255])
    rgba(folder/(model+'_barriers.png'),barrier,[255,255,255,255])
    err=np.zeros((*mask.shape,4),np.uint8);err[target&~mask]=[50,130,255,255];err[mask&~target]=[255,60,60,255]
    Image.fromarray(err).save(folder/(model+'_vector_errors.png')); del err
    # Fixed 0..1 P(wall), no per-case brightness rescaling.
    heat=cv2.applyColorMap((np.clip(prob[2],0,1)*255).astype('uint8'),cv2.COLORMAP_TURBO)
    Image.fromarray(cv2.cvtColor(heat,cv2.COLOR_BGR2RGB)).save(folder/(model+'_probability.png'))
    old=json.loads((baseline/(model+'_decomposition.json')).read_text())
    oldmask=np.array(Image.open(baseline/(model+'_vector_mask.png')))>0
    stats={}
    for state,geom in [('before',old),('after',d)]:
        stat,count=rectangle_overlap_metrics(geom['primitives'],mask.shape,.0025,return_counts=True);stats[state]=stat
        rgba(folder/(model+'_overlap_'+state+'.png'),count>1,[255,30,80,255]);del count
    # Same scan-derived reference for both versions, fixed to the historical target.
    fixed_target=np.array(Image.open(baseline/(model+'_mask_2p5mm.png')))>0
    fixed_forbid=np.array(Image.open(root/'baseline_evidence'/key/(model+'_forbidden.png')))>0
    runs=extract_runs(fixed_target,barrier,.0025,DECOMP,fixed_forbid)
    profiles=[_profile(run,fixed_target,fixed_forbid,.0025,DECOMP) for run in runs]
    profiles=[v for v in profiles if v is not None]
    face_comparison={v:face_metrics(m,profiles,.0025) for v,m in [('before',oldmask),('after',mask)]}
    disagreement=dict(changed_area_m2=float((mask!=oldmask).sum())*.0025**2,
                      new_vs_previous_vector_iou=metrics(mask,oldmask)['iou'])
    paths=[' '.join('M '+' L '.join(f'{x:.6f},{y:.6f}' for x,y in ring)+' Z' for ring in [c['polygon']]+c.get('holes',[])) for c in d['primitives']]
    cloud=viewer.data(root/'review/wall_point_overlay'/key/'wall_height.png')
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="black"/><image href="{cloud}" width="{w}" height="{h}"/>'+''.join(f'<path d="{v}" fill="#9be141" fill-opacity=".2" stroke="#ffb000" stroke-width="1" fill-rule="evenodd"/>' for v in paths)+'</svg>'
    (folder/(model+'_walls.svg')).write_text(svg,encoding='utf-8')
    return dict(case=key,model=model,decomposition=d['metrics'],audit=audit,
                refinement=refinement_metrics,pipeline=d['pipeline'],windows=len(d['windows']),window_review=len(d['window_review']),doors=len(d['doors']),door_review=len(d['door_review']),rectangle_overlap_comparison=stats,
                fixed_reference_scan_faces=face_comparison,comparison=disagreement)


def verify_historical(root, out):
    """Golden Good/L1 replay: exactly reproduce the reviewed staged footprint."""
    fixture=root/'parity/good/L1';dest=out/'historical_parity'
    identity=dict(pipeline_version=PIPELINE_VERSION,
        fixture={p.name:sha(p) for p in sorted(fixture.iterdir()) if p.is_file()},
        source={p.name:sha(p) for p in sorted((Path(__file__).resolve().parents[1]/'simscan').glob('*.py'))})
    cache=dest/'report.json'
    if cache.exists():
        doc=json.loads(cache.read_text())
        if doc.get('identity')==identity and doc.get('changed_pixels')==0:
            print('Historical staged replay: cached exact match',flush=True);return doc
    target=np.array(Image.open(fixture/'target.png'))>0
    forbidden=np.array(Image.open(fixture/'forbidden.png'))>0
    expected=np.array(Image.open(fixture/'expected.png'))>0
    with np.load(fixture/'probability.npz') as a:p=cv2.resize(a['probability'].astype('float32'),target.shape[::-1],interpolation=cv2.INTER_LINEAR)
    with np.load(fixture/'barrier.npz') as a:barrier=a['barrier']
    initial=json.loads((fixture/'initial.json').read_text())
    initial['mask']=np.array(Image.open(fixture/'initial.png'))>0
    def stage(name,result,info):
        print('HISTORICAL',name,'recall',round(info['metrics']['recall'],5),flush=True)
        dest.mkdir(parents=True,exist_ok=True)
        write(dest/(name+'.json'),info)
    result=fit_walls(target,semantic_probability=p,barrier=barrier,forbidden=forbidden,initial=initial,on_stage=stage,historical_replay=True)
    changed=int((expected!=result['mask']).sum())
    doc=dict(identity=identity,changed_pixels=changed,passed=changed==0,
             scope='Same saved area fit, mask and evidence: ordered postprocessing parity. This does not certify fresh area-fitting or training quality.')
    write(cache,doc)
    Image.fromarray(result['mask'].astype('uint8')*255).save(dest/'restored_mask.png')
    if changed:raise AssertionError(f'Historical wall regression: {changed} native pixels changed. New inference blocked.')
    print('Historical staged replay: EXACT MATCH, zero changed pixels',flush=True)
    return doc


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',required=True);ap.add_argument('--train-run',required=True)
    ap.add_argument('--assets',required=True);ap.add_argument('--out',required=True);ap.add_argument('--batch',type=int,default=4)
    ap.add_argument('--allow-old',action='store_true');ap.add_argument('--verify-only',action='store_true');args=ap.parse_args()
    root=Path(args.assets).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    assets=verify_assets(root)
    verify_historical(root,out)
    if args.verify_only:return
    import torch
    from simscan.infer import describe_checkpoint, load_model, predict
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime in Colab before running this test.')
    root=Path(args.assets).resolve();out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=True)
    files,hashes=checkpoints(args.project,args.train_run,out/'checkpoint_cache',args.allow_old)
    infos={k:describe_checkpoint(p) for k,p in files.items()}
    for k,info in infos.items():
        if info['format']!='plannet' or info['in_channels']!=8 or info['classes']!=8: raise ValueError((k,info))
        print('CHECKPOINT',k,hashes[k],info,flush=True)
    source=Path(__file__).resolve().parents[1]
    identity=dict(pipeline_version=PIPELINE_VERSION,stage_order=list(STAGE_ORDER),checkpoints=hashes,assets=assets,source={str(p.relative_to(source)):sha(p) for pattern in ['simscan/*.py','tools/*.py'] for p in sorted(source.glob(pattern))},
        inference=dict(tile=512,overlap=128,tta='none',batch=args.batch),refinement=asdict(REFINE),decomposition=asdict(DECOMP),
        mask_classes=[2],forbidden_classes=[3,4,5],pixel_m=.0025)
    contract=out/'test_contract.json'
    if contract.exists() and json.loads(contract.read_text())!=identity: raise ValueError('Run inputs/code/weights changed. Choose a new TEST_RUN folder.')
    write(contract,identity);write(out/'environment.json',dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()))
    records=[]
    for model,path in files.items():
        net,dev=load_model(path)
        for key in CASES:
            folder=out/key;folder.mkdir(parents=True,exist_ok=True);done=folder/(model+'_complete.json')
            if done.exists(): records.append(json.loads(done.read_text()));print('RESUME',key,model,flush=True);continue
            prediction=folder/(model+'_prediction.npz')
            if prediction.exists():
                with np.load(prediction) as a: prob=a['prob'].astype('float32')
            else:
                x=np.load(root/'inputs'/key/'input.npy').astype('float32')
                prediction_data=predict(net,dev,x,tile=512,overlap=128,batch=args.batch,tta='none')
                prob=prediction_data['prob']
                tmp=prediction.with_suffix('.tmp.npz')
                np.savez_compressed(tmp,**{k:v.astype('float16') for k,v in prediction_data.items()})
                tmp.replace(prediction)
                # Same stored probability precision on initial and resumed execution.
                prob=prob.astype('float16').astype('float32');del x,prediction_data
            row=process(root,out,key,model,prob);write(done,row);records.append(row);del prob;gc.collect()
            write(out/'summary.json',records);print('COMPLETE',key,model,row['audit'],flush=True)
        del net;torch.cuda.empty_cache();gc.collect()
    viewer.ROOT=root;viewer.OUT=out;viewer.OLD=root/'baseline';viewer.WINDOWS_INCLUDED=False
    viewer.BEFORE_LABEL='Previous weights / saved review';viewer.AFTER_LABEL='New weights / current pipeline';viewer.main()
    for name in ['wall_overlap_good.html','wall_overlap_barklaya_L1.html','wall_overlap_barklaya_L2.html']: shutil.copyfile(root/name,out/name)
    lines=['# Current pipeline / new weights review','',
        'Three real levels from two scans; KZN excluded. Wall masks exclude doors/windows/openings; windows and doors are fitted separately. Door leaves, hinge positions and swing directions are not inferred.',
        'Original model inference is 20mm; refinement and vector output are 2.5mm. Upsampling does not create new semantic information.',
        'Before uses the saved previous review. This comparison includes both weights and pipeline changes, not a controlled checkpoint-only ablation.',
        'Scan-face metrics reuse fitting evidence and the historical model-derived target; they are not surveyed wall accuracy.',
        'HTML: previous/current geometry and masks, original P(wall) heatmap, exclusions, barriers, and overlap. Zoom to 1:1 or closer.', '',
        '| Case | Weights | Mask recall | Mask IoU | Pieces | Fixed scan-face coverage before → after |',
        '|---|---|---:|---:|---:|---:|']
    for r in records:
        m=r['decomposition'];f=r['fixed_reference_scan_faces'];lines.append(f"| {r['case']} | {r['model']} | {m['recall']:.1%} | {m['iou']:.3f} | {m['pieces']} | {f['before']['coverage']:.1%} → {f['after']['coverage']:.1%} |")
    (out/'FINDINGS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    archive=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and 'checkpoint_cache' not in p.parts: z.write(p,p.relative_to(out))
    print('RESULTS',archive,flush=True)

if __name__=='__main__': main()
