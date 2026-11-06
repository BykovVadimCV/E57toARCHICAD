"""Append a labels-5 batch while anchoring all existing sample/layout splits.

Streaming ZIP merge: only one new level is extracted at a time. Existing level
files and IDs are copied unchanged; outputs publish only after full validation.
"""
from pathlib import Path,PurePosixPath
from collections import Counter
import argparse,hashlib,json,os,shutil,tempfile,zipfile
import numpy as np
from PIL import Image
from merge_partial_datasets import REQUIRED,valid_level
SPLITS=('train','val','test');PAYLOAD=tuple(n for n in REQUIRED if n!='scene.json')


def digest(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()


def check_archive(z):
 names=z.namelist()
 if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entry names')
 for i in z.infolist():
  p=PurePosixPath(i.filename)
  if p.is_absolute() or '..' in p.parts or '\\' in i.filename or (i.external_attr>>16)&0o170000==0o120000:raise ValueError('Unsafe ZIP entry')
 roots=[n[:-len('meta.json')] for n in names if n.endswith('meta.json') and any(x.startswith(n[:-len('meta.json')]+'scenes/') for x in names)]
 if len(roots)!=1:raise ValueError('Expected one dataset root')
 return roots[0]


def layout_split(key,anchors):
 if key in anchors:return anchors[key]
 n=int(hashlib.md5(key.encode()).hexdigest(),16)%10
 return 'train' if n<8 else ('val' if n==8 else 'test')


def identity(hashes):
 h=hashlib.sha256()
 for n in PAYLOAD:h.update(n.encode());h.update(hashes[n].encode())
 return h.hexdigest()


def read_membership(z,prefix):
 lists={s:z.read(prefix+'splits/'+s+'.txt') for s in SPLITS};membership={};anchors={};scenes={}
 for split,raw in lists.items():
  entries=raw.decode('utf-8').split()
  if len(entries)!=len(set(entries)):raise ValueError('Duplicate split entries')
  for entry in entries:
   if entry in membership:raise ValueError('Old level spans multiple splits')
   sc=json.loads(z.read(prefix+'scenes/'+entry+'/scene.json'));key=sc.get('layout_key')
   if not isinstance(key,str) or not key:raise ValueError('Missing layout identity')
   if key in anchors and anchors[key]!=split:raise ValueError('Old dataset already leaks a layout across splits')
   anchors[key]=split;membership[entry]=split;scenes[entry]=sc
 actual={n[len(prefix+'scenes/'):-len('/scene.json')] for n in z.namelist() if n.startswith(prefix+'scenes/') and n.endswith('/scene.json')}
 if actual!=set(membership):raise ValueError('Old split lists do not cover all old levels')
 return lists,membership,anchors,scenes


def validate_finite(folder):
 if not valid_level(folder):raise ValueError('Invalid completion, shapes or metadata')
 for n in ['input','dist','orient','heights']:
  a=np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False)
  if not np.isfinite(a).all():raise ValueError('Nonfinite '+n)
 with Image.open(folder/'sem.png') as im:
  if not np.isin(np.asarray(im),[0,1,2,3,4,5,6,7,255]).all():raise ValueError('Invalid semantic class')
 aux=folder/'structure_aux.npz'
 if aux.exists():
  shape=np.load(folder/'input.npy',mmap_mode='r',allow_pickle=False).shape[-2:]
  try:
   with np.load(aux,allow_pickle=False) as data:
    for name in data.files:
     a=data[name]
     if a.shape!=shape or not np.isin(a,[0,1]).all():raise ValueError('Invalid auxiliary mask: '+name)
  except (OSError,EOFError,KeyError) as exc:raise ValueError('Invalid auxiliary archive') from exc


def merge(base_path,batch_path,out_path):
 base_path=Path(base_path);batch_path=Path(batch_path);out_path=Path(out_path)
 if out_path.exists():raise ValueError('Output already exists; use a fresh version name')
 out_path.parent.mkdir(parents=True,exist_ok=True);output_root=out_path.stem+'/'
 report={'policy':'existing sample IDs and layout split assignments are immutable; unseen layouts use original md5(layout_key) policy',
         'base':str(base_path.resolve()),'batch':str(batch_path.resolve()),'added':[],'duplicates':[],'skipped':[]}
 source_hash={};old_checks={};content={};seen_crc={};added_members={};new_entry_checks={};seen_names=set()
 with zipfile.ZipFile(base_path) as base,zipfile.ZipFile(batch_path) as batch:
  br=check_archive(base);nr=check_archive(batch)
  bm=json.loads(base.read(br+'meta.json'));nm=json.loads(batch.read(nr+'meta.json'))
  if bm!=nm or json.loads(base.read(br+'supervision_policy.json'))!=json.loads(batch.read(nr+'supervision_policy.json')):raise ValueError('Dataset contracts differ')
  if bm.get('labels')!='labels-5' or bm.get('version')!=3:raise ValueError('Expected labels-5 v3')
  split_bytes,old_members,anchors,old_scenes=read_membership(base,br)
  base_anchors=dict(anchors);report['old_splits']={s:sum(v==s for v in old_members.values()) for s in SPLITS}
  source_hash={'base_sha256':digest(base_path),'batch_sha256':digest(batch_path)};report.update(source_hash)
  # Archive CRC+sizes narrow exact-duplicate candidates; SHA256 confirms equality.
  for entry in old_members:
   directory=br+'scenes/'+entry+'/'
   sig=tuple((n,base.getinfo(directory+n).CRC,base.getinfo(directory+n).file_size) for n in PAYLOAD)
   seen_crc.setdefault(sig,[]).append(entry)
  with tempfile.TemporaryDirectory(prefix='split_preserving_merge_',dir=out_path.parent) as staging:
   staging=Path(staging);partial=staging/out_path.name;stage=staging/'level';stage.mkdir()
   with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as out:
    print('Copying original dataset unchanged:',len(old_members),'levels',flush=True)
    for i,item in enumerate(base.infolist()):
     if item.is_dir():continue
     rel=item.filename[len(br):]
     if rel.startswith('splits/') or rel=='merge_report.json':continue
     h=hashlib.sha256()
     with base.open(item) as src,out.open(output_root+rel,'w',force_zip64=True) as dst:
      for chunk in iter(lambda:src.read(4*1024**2),b''):h.update(chunk);dst.write(chunk)
     if rel.startswith('scenes/'):
      old_checks[rel]=h.hexdigest();parts=PurePosixPath(rel).parts
      entry='/'.join(parts[1:3]);name=parts[-1]
      if name in PAYLOAD:content.setdefault(entry,{})[name]=h.hexdigest()
     if i and i%2400==0:print('Copied',i,'base files',flush=True)
    for entry,hashes in content.items():
     token=identity(hashes);previous=source_hash.get(token)
     if previous and old_scenes[previous]['layout_key']!=old_scenes[entry]['layout_key']:raise ValueError('Identical old content has conflicting layout IDs')
     source_hash[token]=entry
    # Keep the original report and exact original split lists as provenance.
    if br+'merge_report.json' in base.namelist():out.writestr(output_root+'provenance/prior_merge_report.json',base.read(br+'merge_report.json'))
    for s,raw in split_bytes.items():out.writestr(output_root+'provenance/original_splits/'+s+'.txt',raw)
    if nr+'build_contract.json' in batch.namelist():out.writestr(output_root+'provenance/new_batch_build_contract.json',batch.read(nr+'build_contract.json'))
    levels=sorted(n.rsplit('/',1)[0] for n in batch.namelist() if n.startswith(nr+'scenes/') and n.endswith('/scene.json'))
    new_lists={s:[] for s in SPLITS};all_entries=set(old_members)
    for index,directory in enumerate(levels):
     origin=directory[len(nr+'scenes/'):];sc=json.loads(batch.read(directory+'/scene.json'));key=sc.get('layout_key')
     if not isinstance(key,str) or not key or 'None' in key:raise ValueError('Missing batch layout key')
     if sc.get('labels')!=bm['labels'] or sc.get('unseen_policy')!=json.loads(base.read(br+'supervision_policy.json'))['unseen_policy']:raise ValueError('Level supervision mismatch')
     if sc.get('frame',{}).get('pixel_m')!=bm['pixel_m']:raise ValueError('Level pixel size mismatch')
     try:
      marker=json.loads(batch.read(directory+'/complete.json'))['files']
      if any(batch.getinfo(directory+'/'+n).file_size!=marker[n] for n in REQUIRED):raise ValueError('Invalid completion sizes')
     except (KeyError,ValueError) as exc:
      report['skipped'].append({'level':origin,'reason':str(exc)});continue
     crc=tuple((n,batch.getinfo(directory+'/'+n).CRC,batch.getinfo(directory+'/'+n).file_size) for n in PAYLOAD)
     hashes={}
     if crc in seen_crc:
      for n in PAYLOAD:
       h=hashlib.sha256()
       with batch.open(directory+'/'+n) as src:
        for chunk in iter(lambda:src.read(4*1024**2),b''):h.update(chunk)
       hashes[n]=h.hexdigest()
      token=identity(hashes)
      if token in source_hash:
       previous=source_hash[token];prev_key=old_scenes[previous]['layout_key'] if previous in old_scenes else added_members[previous]['layout_key']
       if key!=prev_key:raise ValueError('Duplicate content has different layout keys; refusing possible leakage')
       report['duplicates'].append({'level':origin,'kept_as':previous,'layout_key':key,'auxiliary_from_duplicate_not_imported':directory+'/structure_aux.npz' in batch.NameToInfo});continue
     # Only one unique level is on disk at a time.
     for p in stage.iterdir():p.unlink()
     for n in (*REQUIRED,'complete.json'):
      h=hashlib.sha256()
      with batch.open(directory+'/'+n) as src,(stage/n).open('wb') as dst:
       for chunk in iter(lambda:src.read(4*1024**2),b''):h.update(chunk);dst.write(chunk)
      if n in PAYLOAD:hashes[n]=h.hexdigest()
     optional=directory+'/structure_aux.npz'
     if optional in batch.NameToInfo:
      with batch.open(optional) as src,(stage/'structure_aux.npz').open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
     try:validate_finite(stage)
     except ValueError as exc:
      report['skipped'].append({'level':origin,'reason':str(exc)});continue
     token=identity(hashes)
     if token in source_hash:
      previous=source_hash[token];prev_key=old_scenes[previous]['layout_key'] if previous in old_scenes else added_members[previous]['layout_key']
      if key!=prev_key:raise ValueError('Duplicate content has different layout keys')
      report['duplicates'].append({'level':origin,'kept_as':previous,'layout_key':key,'auxiliary_from_duplicate_not_imported':optional in batch.NameToInfo});continue
     scene_name='batch02_'+origin.split('/')[0];entry=scene_name+'/'+origin.split('/')[1]
     if entry in all_entries:raise ValueError('Output ID collision')
     split=layout_split(key,anchors);anchors[key]=split
     sc.update(original_scene_id=sc.get('scene_id'),scene_id=scene_name,merge_source=str(batch_path.resolve()))
     (stage/'scene.json').write_text(json.dumps(sc,ensure_ascii=False,indent=1),encoding='utf-8')
     (stage/'complete.json').write_text(json.dumps({'files':{n:(stage/n).stat().st_size for n in REQUIRED}},indent=1),encoding='utf-8')
     for p in sorted(stage.iterdir()):out.write(p,output_root+'scenes/'+entry+'/'+p.name)
     added_members[entry]={'layout_key':key,'split':split};new_lists[split].append(entry);all_entries.add(entry)
     source_hash[token]=entry;seen_crc.setdefault(crc,[]).append(entry)
     new_entry_checks[entry]=dict(hashes)
     report['added'].append({'source_level':origin,'output':entry,'layout_key':key,'split':split,'existing_layout':key in base_anchors})
     if len(report['added'])%100==0:print('Batch processed',index+1,'/',len(levels),'added',len(report['added']),'duplicates',len(report['duplicates']),flush=True)
    for s,raw in split_bytes.items():
     appended=''.join(e+'\n' for e in sorted(new_lists[s])).encode()
     out.writestr(output_root+'splits/'+s+'.txt',raw+(b'\n' if raw and not raw.endswith(b'\n') else b'')+appended)
    report['summary']={'old_levels':len(old_members),'new_batch_levels':len(levels),'added_levels':len(added_members),'duplicates_skipped':len(report['duplicates']),'invalid_skipped':len(report['skipped']),'total_levels':len(all_entries),'layout_groups':len(anchors),'splits':{s:report['old_splits'][s]+len(new_lists[s]) for s in SPLITS},'added_splits':{s:len(new_lists[s]) for s in SPLITS},'old_membership_changes':0}
    out.writestr(output_root+'merge_report.json',json.dumps(report,ensure_ascii=False,indent=2))
   print('Verifying every output file CRC and original level SHA256…',flush=True)
   with zipfile.ZipFile(partial) as out:
    op=check_archive(out);_,membership,final_anchors,scenes=read_membership(out,op)
    if any(membership.get(e)!=s for e,s in old_members.items()):raise ValueError('Old split membership changed')
    if len(membership)!=report['summary']['total_levels']:raise ValueError('Missing merged levels')
    for s in SPLITS:
     if out.read(op+'provenance/original_splits/'+s+'.txt')!=split_bytes[s]:raise ValueError('Original split bytes changed')
    original_verified=0
    for info in out.infolist():
     h=hashlib.sha256()
     with out.open(info) as f:
      for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
     rel=info.filename[len(op):]
     if rel in old_checks:
      if h.hexdigest()!=old_checks[rel]:raise ValueError('Original level bytes changed: '+rel)
      original_verified+=1
     elif rel.startswith('scenes/'):
      parts=PurePosixPath(rel).parts;entry='/'.join(parts[1:3]);n=parts[-1]
      if entry in new_entry_checks and n in PAYLOAD and h.hexdigest()!=new_entry_checks[entry][n]:raise ValueError('New payload bytes changed')
    verification={'old_level_files_byte_identical':original_verified,'old_levels_preserved':len(old_members),'old_split_changes':0,'layout_split_leakage':0,'all_output_crc':'passed','new_payload_sha256':'passed','output_sha256':digest(partial),'output_bytes':partial.stat().st_size,'summary':report['summary']}
   os.replace(partial,out_path)
  report_path=out_path.with_suffix('.merge_report.json');report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
  out_path.with_suffix('.verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
 print(json.dumps(verification,indent=2),flush=True);return report


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--batch',required=True);p.add_argument('--out',required=True);a=p.parse_args();merge(a.base,a.batch,a.out)

if __name__=='__main__':main()
