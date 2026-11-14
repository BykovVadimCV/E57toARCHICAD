"""Wall/window masks and density/verticality barriers independent of semantic borders."""
import numpy as np
import cv2
from .vectorize import struct_mask


def wall_only(prob,pixel_m=.02,include_windows=False):
    prob=np.asarray(prob,np.float32)
    if prob.ndim!=3 or prob.shape[0]!=8:raise ValueError('Expected eight semantic probabilities')
    labels=prob.argmax(0);excluded=[3,5] if include_windows else [3,4,5]
    forbidden=np.isin(labels,excluded)
    structural=prob.copy();structural[excluded]=0
    mask=struct_mask(structural,pixel_m)&~forbidden
    return dict(mask=mask,probability=np.clip(prob[2]+(prob[4] if include_windows else 0),0,1),forbidden=forbidden,
                excluded_classes=['door','opening'] if include_windows else ['door','window','opening'])


def dense_barriers(counts,scan10,pixel_m=.0025,min_count_2cm=200.,vertical_min=.6):
    counts=np.asarray(counts);scan10=np.asarray(scan10)
    if counts.ndim!=2 or scan10.ndim!=3 or scan10.shape[0]!=8:raise ValueError('Invalid count/feature arrays')
    h,w=counts.shape;factor=.01/pixel_m
    if counts.shape!=(int(scan10.shape[1]*factor),int(scan10.shape[2]*factor)):raise ValueError('Frames must cover the same extent')
    gate=(scan10[6]>=vertical_min)&((scan10[:3]>.08).sum(0)>=2)
    gate=cv2.resize(gate.astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)>0
    dense=(counts>=min_count_2cm*(pixel_m/.02)**2)&gate
    # Close gaps at the sensor-tolerance scale; never use low-density returns as seeds.
    r=max(1,int(round(.005/pixel_m)))
    dense=cv2.morphologyEx(dense.astype(np.uint8),cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,)*2))
    n,labels,stats,_=cv2.connectedComponentsWithStats(dense,8)
    keep=stats[:,cv2.CC_STAT_AREA]>=max(3,int(round(.06/pixel_m)));keep[0]=False
    return keep[labels]
