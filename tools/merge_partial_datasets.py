"""Merge completed labels-5 levels from folders, ZIPs or Colab checkpoint folders.

Inputs are read only. No scan generation or Open3D dependency. Run --help for CLI.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile
import numpy as np
from PIL import Image

REQUIRED=('input.npy','sem.png','dist.npy','orient.npy','heights.npy',
          'supervision_valid.npy','semantic_truth.png','observation_support.npy',
          'vector.json','vector_physical.json','scene.json')


def read_json(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def file_hash(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def valid_level(d):
    try:
        manifest=read_json(d/'complete.json')['files']
        if any((d/n).stat().st_size!=manifest[n] for n in REQUIRED):return False
        x=np.load(d/'input.npy',mmap_mode='r',allow_pickle=False)
        if x.ndim!=3 or x.shape[0]!=8:return False
        shape=x.shape[-2:]
        for n,c in [('dist',1),('orient',2),('heights',2)]:
            if np.load(d/(n+'.npy'),mmap_mode='r',allow_pickle=False).shape!=(c,*shape):return False
        for n in ['supervision_valid','observation_support']:
            if np.load(d/(n+'.npy'),mmap_mode='r',allow_pickle=False).shape!=shape:return False
        for n in ['sem','semantic_truth']:
            with Image.open(d/(n+'.png')) as image:
                if np.asarray(image).shape!=shape:return False
        for n in ['scene','vector','vector_physical']:read_json(d/(n+'.json'))
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def unpack(archive,destination):
    sidecar=archive.with_suffix('.sha256.json')
    if sidecar.exists() and file_hash(archive)!=read_json(sidecar)['sha256']:
        raise ValueError(f'Checksum mismatch: {archive}')
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            if not (destination/item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe ZIP path')
            if (item.external_attr>>16)&0o170000==0o120000:raise ValueError('ZIP symlink refused')
        z.extractall(destination)


def roots_for(path,staging,issues=None):
    path=Path(path);issues=[] if issues is None else issues
    if path.is_dir() and (path/'meta.json').exists() and (path/'scenes').exists():return [path]
    archives=[path] if path.is_file() else sorted(path.glob('*.zip'))
    if not archives:raise ValueError(f'No dataset or ZIPs found at {path}')
    staging.mkdir(parents=True,exist_ok=True);roots=[]
    # Every archive stays isolated: shared Drive runs can reuse the same scene names.
    for i,archive in enumerate(archives):
        if archive.name.startswith('generator_source'):continue
        is_part=archive.name.startswith('part_')
        if is_part and not archive.with_suffix('.sha256.json').exists():
            issues.append({'archive':str(archive),'reason':'uncommitted checkpoint'});continue
        destination=staging/f'archive_{i:05d}';destination.mkdir()
        try:
            unpack(archive,destination)
        except (ValueError,OSError,zipfile.BadZipFile,EOFError) as exc:
            if not is_part:raise
            issues.append({'archive':str(archive),'reason':str(exc)})
            print('Skipping damaged shared checkpoint:',archive.name,flush=True);continue
        package=destination/'package_manifest.json'
        if package.exists():
            manifest=read_json(package)
            if manifest.get('format')!='colab-dataset-package-v1':raise ValueError('Unknown package format')
            for j,member in enumerate(manifest['archives']):
                payload=(destination/member).resolve()
                if not payload.is_relative_to(destination.resolve()):raise ValueError('Unsafe package member')
                roots.extend(roots_for(payload,destination/f'payload_{j:05d}',issues))
            continue
        roots.extend(p.parent for p in destination.rglob('meta.json') if (p.parent/'scenes').is_dir())
    if not roots:raise ValueError(f'No valid dataset root in {path}')
    return sorted(roots)


def contract(root):
    meta=read_json(root/'meta.json')
    if meta.get('version')!=3 or meta.get('labels')!='labels-5':
        raise ValueError(f'{root}: only labels-5 / metadata v3 supported; do not mix older targets')
    policy=read_json(root/'supervision_policy.json')['unseen_policy']
    keys=['version','labels','pixel_m','rasterizer','channels','classes','dist','orient','heights','exterior_wall']
    return {k:meta.get(k) for k in keys}|{'unseen_policy':policy}


def merge(inputs,out,zip_path=None):
    inputs=[Path(p).resolve() for p in inputs];out=Path(out).resolve()
    if out.exists():raise ValueError('Use a new output directory')
    if any(out.is_relative_to(p) or p.is_relative_to(out) for p in inputs):
        raise ValueError('Output and input paths must be separate')
    if zip_path and Path(zip_path).exists():raise ValueError('Output ZIP already exists')
    out.parent.mkdir(parents=True,exist_ok=True)
    report={'archive_issues':[],'sources':[],'kept':[],'duplicates':[],'skipped':[],'split_policy':'md5(layout_key), same layout grouped across all runs'}
    seen={};signature=None;first_meta=None;split={'train':[],'val':[],'test':[]};groups=set()
    with tempfile.TemporaryDirectory(prefix='e57_merge_',dir=out.parent) as temp:
        staging=Path(temp);result=staging/'merged';(result/'scenes').mkdir(parents=True)
        for source_id,path in enumerate(inputs,1):
            for root_id,root in enumerate(roots_for(path,staging/f'source_{source_id}',report['archive_issues']),1):
                sig=contract(root)
                if signature is None:signature=sig;first_meta=read_json(root/'meta.json')
                elif signature!=sig:raise ValueError(f'Incompatible label/input contract: {path}')
                provenance={'input':str(path),'root':root.name,'contract':sig}
                if (root/'build_contract.json').exists():provenance['build']=read_json(root/'build_contract.json')
                report['sources'].append(provenance)
                levels=sorted((root/'scenes').glob('*/L*'))
                for level in levels:
                    origin={'source':str(path),'level':level.relative_to(root).as_posix()}
                    if not valid_level(level):report['skipped'].append(origin|{'reason':'incomplete or invalid'});continue
                    scene=read_json(level/'scene.json');key=scene.get('layout_key')
                    if not isinstance(key,str) or not key or 'None' in key:raise ValueError(f'Missing layout identity: {level}')
                    if scene.get('labels')!='labels-5' or scene.get('unseen_policy')!=signature['unseen_policy']:
                        raise ValueError(f'Level supervision contract mismatch: {level}')
                    # Metadata differs between runs; learning targets + inputs decide exact duplicates.
                    h=hashlib.sha256()
                    for n in REQUIRED:
                        if n!='scene.json':h.update(n.encode());h.update(file_hash(level/n).encode())
                    token=h.hexdigest()
                    if token in seen:
                        previous=seen[token]
                        # Grouping differing keys would require reconciling every related variant.
                        if previous['layout_key']!=key:raise ValueError('Identical data has different layout keys; review before splitting')
                        report['duplicates'].append(origin|{'kept_as':previous['path']});continue
                    name=f'r{source_id:02d}_{root_id:02d}_{level.parent.name}'
                    dest=result/'scenes'/name/level.name
                    shutil.copytree(level,dest)
                    scene['original_scene_id']=scene.get('scene_id');scene['scene_id']=name
                    scene['merge_source']=str(path)
                    write_json(dest/'scene.json',scene)
                    write_json(dest/'complete.json',{'files':{n:(dest/n).stat().st_size for n in REQUIRED}})
                    entry=f'{name}/{level.name}'
                    bucket=int(hashlib.md5(key.encode()).hexdigest(),16)%10
                    split['train' if bucket<8 else ('val' if bucket==8 else 'test')].append(entry)
                    seen[token]={'layout_key':key,'path':entry};groups.add(key)
                    report['kept'].append(origin|{'output':entry,'layout_key':key})
                print(f'{path.name}: kept total {len(report["kept"])} levels',flush=True)
        if not report['kept']:raise ValueError('No completed compatible levels found')
        write_json(result/'meta.json',first_meta)
        write_json(result/'supervision_policy.json',{'unseen_policy':signature['unseen_policy'],'contract_version':1})
        (result/'splits').mkdir()
        for k,entries in split.items():(result/'splits'/(k+'.txt')).write_text(''.join(e+'\n' for e in sorted(entries)),encoding='utf-8')
        report['summary']={'levels':len(report['kept']),'layout_groups':len(groups),
            'duplicates_removed':len(report['duplicates']),'incomplete_levels_skipped':len(report['skipped']),'archives_skipped':len(report['archive_issues']),
            'splits':{k:len(v) for k,v in split.items()}}
        write_json(result/'merge_report.json',report)
        os.replace(result,out)
    if zip_path:
        zip_path=Path(zip_path);zip_path.parent.mkdir(parents=True,exist_ok=True)
        partial=zip_path.with_suffix(zip_path.suffix+'.partial')
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(out.rglob('*')):
                if p.is_file():z.write(p,Path(out.name)/p.relative_to(out))
        os.replace(partial,zip_path)
    print(json.dumps(report['summary'],indent=2),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True)
    p.add_argument('--out',required=True);p.add_argument('--zip')
    a=p.parse_args();merge(a.inputs,a.out,a.zip)

if __name__=='__main__':main()
