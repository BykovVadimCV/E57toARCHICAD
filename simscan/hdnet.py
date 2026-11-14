"""Small native HD refinement model; independent of legacy PlanNet checkpoints."""
import torch
from torch import nn
import torch.nn.functional as F

class HDRefiner(nn.Module):
    def __init__(self,width=24):
        super().__init__()
        def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(4,b),nn.SiLU())
        self.a=block(15,width);self.b=block(width,width*2);self.c=block(width*2,width*4)
        self.upb=block(width*6,width*2);self.upa=block(width*3,width)
        self.sem=nn.Conv2d(width,6,1);self.edge=nn.Conv2d(width,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2))
        b=self.upb(torch.cat([F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        a=self.upa(torch.cat([F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1))
        return dict(sem=self.sem(a),edge=self.edge(a))

def loss(out,batch):
    valid=batch['valid'].bool();target=batch['sem'].long();logits=out['sem'].float()
    ce=F.cross_entropy(logits,target,weight=logits.new_tensor([1,4,4,4,4,3]),reduction='none')
    total=(ce*valid).sum()/valid.sum().clamp(min=1)
    prob=logits.softmax(1);truth=F.one_hot(target,6).permute(0,3,1,2).float();mask=valid[:,None]
    inter=(prob*truth*mask).sum((0,2,3));den=((prob+truth)*mask).sum((0,2,3))
    present=(truth*mask).sum((0,2,3))>0
    if present.any():total=total+(1-(2*inter+1)/(den+1))[present].mean()
    edge=F.binary_cross_entropy_with_logits(out['edge'].float(),batch['edge'].float(),reduction='none',pos_weight=logits.new_tensor(10.))
    return total+.25*(edge*mask).sum()/valid.sum().clamp(min=1)
