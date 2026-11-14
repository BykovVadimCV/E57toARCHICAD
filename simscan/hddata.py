"""Aligned native HD patches: real scan bands/counts plus coarse *input* features.

Geometry targets never enter the network input. Coarse features are sampled by
world coordinates into each patch, avoiding full-resolution 8-channel upsampling.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from .hd_evidence import SCAN_HD_FILES
from .observation import OBSERVATION_CONTRACT
from .semantic_contract import (HD_MASK_FORMAT, HD_OPENING_VALUES,
                                HD_SEMANTIC_CLASSES, OPENING_SEMANTICS_VERSION)

INPUT_CHANNELS = 15
N_CLASSES = 6  # background, wall, door, window, passage, physical floor void


def load_patch(folder, top=0, left=0, size=512, cache=None):
    folder=Path(folder)
    if cache is None:cache={}
    if cache.get('_folder')!=str(folder):cache.clear();cache['_folder']=str(folder)
    meta=json.loads((folder/'wall_mask_hd.json').read_text(encoding='utf-8'))
    evidence=json.loads((folder/'scan_evidence_hd.json').read_text(encoding='utf-8'))
    if (meta.get('format')!=HD_MASK_FORMAT or
            meta.get('opening_semantics')!=OPENING_SEMANTICS_VERSION or
            meta.get('opening_values')!=dict(HD_OPENING_VALUES)):
        raise ValueError('Incompatible native HD opening semantics: regenerate labels with new contract')
    frame=meta['frame'];coarse_frame=evidence['low_res_frame']
    if frame!=evidence['frame']:raise ValueError('HD scan/target frames differ')
    if top<0 or left<0 or size<32:raise ValueError('Invalid HD crop')
    h,w=frame['height'],frame['width'];hh=max(0,min(size,h-top));ww=max(0,min(size,w-left))
    def crop(a):
        out=np.zeros((size,size),a.dtype)
        out[:hh,:ww]=a[top:top+hh,left:left+ww]
        return out
    def image(name):
        if name not in cache:cache[name]=cv2.imread(str(folder/name),cv2.IMREAD_UNCHANGED)
        a=cache[name]
        if a is None or a.shape!=(h,w):raise ValueError('Invalid HD image: '+name)
        return crop(a)
    inputs=[image(name).astype(np.float32)/255 for name in SCAN_HD_FILES]
    if 'counts' not in cache:
        with np.load(folder/'scan_counts_hd.npz',allow_pickle=False) as archive:cache['counts']=archive['counts']
    counts=cache['counts']
    if counts.shape!=(h,w) or counts.dtype!=np.uint32:raise ValueError('Invalid native count raster')
    inputs.append(np.clip(np.log1p(crop(counts))/np.log1p(256.),0,1).astype(np.float32))
    coarse=np.load(folder/'input.npy',mmap_mode='r',allow_pickle=False)
    if coarse.shape[0]!=8:raise ValueError('Expected eight coarse input channels')
    cp,hp=coarse_frame['pixel_m'],frame['pixel_m'];scale=hp/cp
    tx=(frame['origin_x']+hp/2-coarse_frame['origin_x'])/cp-.5+left*scale
    ty=coarse.shape[1]-.5-(frame['origin_y']+(h-.5)*hp-coarse_frame['origin_y'])/cp+top*scale
    matrix=np.array([[scale,0,tx],[0,scale,ty]],np.float64)
    for channel in coarse:
        inputs.append(cv2.warpAffine(channel.astype(np.float32),matrix,(size,size),
            flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_CONSTANT))
    wall=image('wall_mask_hd.png')>0;aperture=image('openings_hd.png');void=image('floor_void_hd.png')>0
    if not np.isin(aperture,[0,1,2,3]).all():raise ValueError('Invalid opening classes')
    semantic=np.zeros((size,size),np.int64);semantic[wall]=HD_SEMANTIC_CLASSES['wall']
    for kind in ('door','window','passage'):
        semantic[aperture==HD_OPENING_VALUES[kind]]=HD_SEMANTIC_CLASSES[kind]
    semantic[void&~wall&(aperture==0)]=HD_SEMANTIC_CLASSES['physical_floor_void']
    if meta.get("observation_contract") != OBSERVATION_CONTRACT or \
            evidence.get("observation", {}).get("format") != OBSERVATION_CONTRACT:
        raise ValueError("Missing native HD observation contract: regenerate HD data")
    valid=image('training_valid_hd.png')>0
    edges=(image('wall_edges_hd.png')>0)|(image('opening_edges_hd.png')>0)
    return dict(input=np.stack(inputs),sem=semantic,valid=valid,edge=edges.astype(np.float32)[None])


class HDPatches:
    def __init__(self,root,split='train',size=512,patches_per_level=4,seed=0):
        self.root=Path(root);self.split=split;self.size=size;self.patches=patches_per_level;self.seed=seed;self.epoch=0
        self.cache={}
        self.items=(self.root/'splits'/(split+'.txt')).read_text(encoding='utf-8').split()
        if not self.items or patches_per_level<1:raise ValueError('No HD levels or invalid patch count')
        from .hdmask import complete_hd_level
        self.pixel_m=None
        for item in self.items:
            path=self.root/'scenes'/item;meta=json.loads((path/'wall_mask_hd.json').read_text(encoding='utf-8'))
            pitch=float(meta['frame']['pixel_m'])
            if self.pixel_m is not None and not np.isclose(pitch,self.pixel_m):raise ValueError('Mixed native pixel pitches in training split')
            self.pixel_m=pitch
            if not complete_hd_level(path,meta['frame']['pixel_m']*1000,meta['min_size_px']):
                raise ValueError('Incomplete HD level: '+str(path))
    def __len__(self):return len(self.items)*self.patches
    def __getitem__(self,index):
        path=self.root/'scenes'/self.items[index//self.patches]
        rng=np.random.default_rng([self.seed,index,self.epoch if self.split=='train' else 0])
        meta=json.loads((path/'wall_mask_hd.json').read_text(encoding='utf-8'))['frame']
        top=int(rng.integers(max(1,meta['height']-self.size+1)));left=int(rng.integers(max(1,meta['width']-self.size+1)))
        # Bias training towards architecture; validation uses fixed uniform tiles.
        if self.split=='train' and rng.random()<.6:
            wall=cv2.imread(str(path/'wall_mask_hd.png'),cv2.IMREAD_GRAYSCALE)
            yy,xx=np.nonzero(wall)
            if len(yy):
                k=int(rng.integers(len(yy)));top=max(0,min(meta['height']-self.size,int(yy[k])-self.size//2));left=max(0,min(meta['width']-self.size,int(xx[k])-self.size//2))
        return load_patch(path,top,left,self.size,self.cache)
