import json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from merge_dataset_preserve_splits import merge,layout_split,REQUIRED


def dataset(tmp,name,samples,splits=None,pixel=.02,aux=None):
 root=tmp/name;(root/'scenes').mkdir(parents=True)
 (root/'meta.json').write_text(json.dumps(dict(version=3,labels='labels-5',pixel_m=pixel)))
 (root/'supervision_policy.json').write_text(json.dumps(dict(unseen_policy='ignore')))
 for sid,key,value in samples:
  d=root/'scenes'/sid/'L1';d.mkdir(parents=True)
  np.save(d/'input.npy',np.full((8,4,5),value,np.float16))
  for n,c in [('dist',1),('orient',2),('heights',2)]:np.save(d/(n+'.npy'),np.zeros((c,4,5),np.float32))
  for n in ['supervision_valid','observation_support']:np.save(d/(n+'.npy'),np.ones((4,5),bool))
  for n in ['sem','semantic_truth']:Image.fromarray(np.full((4,5),2,np.uint8)).save(d/(n+'.png'))
  for n in ['vector','vector_physical']:(d/(n+'.json')).write_text('{}')
  if aux and sid in aux:np.savez_compressed(d/'structure_aux.npz',stair_footprint=aux[sid])
  (d/'scene.json').write_text(json.dumps(dict(scene_id=sid,layout_key=key,labels='labels-5',unseen_policy='ignore',frame={'pixel_m':pixel})))
  (d/'complete.json').write_text(json.dumps({'files':{n:(d/n).stat().st_size for n in REQUIRED}}))
 if splits is not None:
  (root/'splits').mkdir()
  for s in ['train','val','test']:(root/'splits'/(s+'.txt')).write_text(''.join(x+'/L1\n' for x in splits.get(s,[])))
 z=tmp/(name+'.zip')
 with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as archive:
  for p in root.rglob('*'):
   if p.is_file():archive.write(p,name+'/'+str(p.relative_to(root)))
 return z


def test_anchor_existing_splits_keep_ids_and_remove_duplicates(tmp_path):
 key=next('key'+str(i) for i in range(100) if layout_split('key'+str(i),{})=='train')
 base=dataset(tmp_path,'old',[('old_train','t',.1),('old_val',key,.2)],{'train':['old_train'],'val':['old_val']})
 batch=dataset(tmp_path,'batch',[('dup',key,.2),('variant',key,.3),('train_variant','t',.4),('new','fresh',.5)])
 out=tmp_path/'combined.zip';report=merge(base,batch,out)
 assert report['summary']['duplicates_skipped']==1
 with zipfile.ZipFile(out) as z,zipfile.ZipFile(base) as b:
  train=z.read('combined/splits/train.txt').decode().split();val=z.read('combined/splits/val.txt').decode().split()
  assert 'old_train/L1' in train and 'old_val/L1' in val
  assert 'batch02_variant/L1' in val and 'batch02_variant/L1' not in train
  assert z.read('combined/scenes/old_val/L1/scene.json')==b.read('old/scenes/old_val/L1/scene.json')
  assert z.read('combined/provenance/original_splits/val.txt')==b.read('old/splits/val.txt')
 assert report['summary']['old_membership_changes']==0


def test_old_leakage_refused(tmp_path):
 base=dataset(tmp_path,'old',[('a','same',.1),('b','same',.2)],{'train':['a'],'val':['b']})
 batch=dataset(tmp_path,'batch',[('new','different',.3)])
 with pytest.raises(ValueError,match='already leaks'):merge(base,batch,tmp_path/'out.zip')
 assert not (tmp_path/'out.zip').exists()


def test_incompatible_resolution_refused(tmp_path):
 base=dataset(tmp_path,'old',[('a','a',.1)],{'train':['a']})
 batch=dataset(tmp_path,'batch',[('new','b',.2)],pixel=.01)
 with pytest.raises(ValueError,match='contracts differ'):merge(base,batch,tmp_path/'out.zip')
 assert not (tmp_path/'out.zip').exists()


def test_auxiliary_annotations_preserved_and_invalid_masks_rejected(tmp_path):
 base=dataset(tmp_path,'old',[('a','a',.1)],{'train':['a']})
 batch=dataset(tmp_path,'batch',[('new','b',.2),('bad','c',.3)],aux={'new':np.ones((4,5),bool),'bad':np.ones((2,2),bool)})
 out=tmp_path/'out.zip';report=merge(base,batch,out)
 assert report['summary']['added_levels']==1 and report['summary']['invalid_skipped']==1
 with zipfile.ZipFile(out) as z:
  import io
  with np.load(io.BytesIO(z.read('out/scenes/batch02_new/L1/structure_aux.npz'))) as aux:
   assert aux['stair_footprint'].all()
  assert 'a/L1' in z.read('out/splits/train.txt').decode().split()
