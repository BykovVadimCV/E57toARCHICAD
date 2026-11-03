"""Immutable dataset contracts and validated completion markers."""
from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
import numpy as np

REQUIRED = ('input.npy','sem.png','dist.npy','orient.npy','heights.npy',
            'supervision_valid.npy','semantic_truth.png','observation_support.npy',
            'vector.json','vector_physical.json','scene.json')

def atomic_json(path, doc):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        f.write(json.dumps(doc,ensure_ascii=False,indent=1));tmp=f.name
    os.replace(tmp,path)

def fingerprint(config):
    from .config import config_to_dict
    folder=Path(__file__).parent
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('*.py'))}
    doc={'config':config_to_dict(config),'source':source}
    return hashlib.sha256(json.dumps(doc,sort_keys=True).encode()).hexdigest()

def validate_level(folder):
    import cv2
    folder=Path(folder)
    if any(not (folder/n).is_file() for n in REQUIRED):return False
    try:
        x=np.load(folder/'input.npy',mmap_mode='r',allow_pickle=False)
        if x.ndim!=3 or x.shape[0]!=8:return False
        shape=x.shape[-2:]
        for n,c in [('dist',1),('orient',2),('heights',2)]:
            if np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False).shape!=(c,*shape):return False
        for n in ['supervision_valid','observation_support']:
            if np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False).shape!=shape:return False
        for n in ['sem','semantic_truth']:
            im=cv2.imread(str(folder/(n+'.png')),cv2.IMREAD_UNCHANGED)
            if im is None or im.shape!=shape:return False
        for n in ['scene','vector','vector_physical']:json.loads((folder/(n+'.json')).read_text())
        return True
    except (ValueError,OSError,KeyError):return False

def finish_level(folder):
    folder=Path(folder)
    if not validate_level(folder):raise ValueError(f'Incomplete dataset level: {folder}')
    for n in ['input','dist','orient','heights']:
        a=np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False)
        if not np.isfinite(a).all():raise ValueError(f'Nonfinite {n} in {folder}')
    atomic_json(folder/'complete.json' ,{'files':{n:(folder/n).stat().st_size for n in REQUIRED}})

def complete_level(folder):
    folder=Path(folder)
    try:
        doc=json.loads((folder/'complete.json').read_text())
        return all((folder/n).stat().st_size==doc['files'][n] for n in REQUIRED) and validate_level(folder)
    except (OSError,ValueError,KeyError):return False
