import numpy as np
import pytest

torch = pytest.importorskip('torch')
from simscan.net import PlanNet, losses
from simscan.infer import predict


def test_old_weights_initialize_auxiliary_head_and_backpropagate():
    old = PlanNet(width=8)
    model = PlanNet(width=8, aux_channels=5)
    status = model.load_state_dict(old.state_dict(), strict=False)
    assert set(status.missing_keys) == {'aux.weight', 'aux.bias'}
    assert not status.unexpected_keys
    x = torch.rand(2, 8, 32, 32)
    out = model(x)
    assert out['aux'].shape == (2, 5, 32, 32)
    b = dict(sem=torch.zeros((2,32,32),dtype=torch.long), dist=torch.zeros(2,1,32,32),
             orient=torch.zeros(2,2,32,32), heights=torch.zeros(2,2,32,32),
             mask_orient=torch.zeros(2,32,32), mask_heights=torch.zeros(2,32,32),
             aux=torch.zeros(2,5,32,32), aux_valid=torch.zeros(2,5,32,32,dtype=torch.bool))
    assert losses(out,b)['aux'].item() == 0
    b['aux_valid'][0,0] = True
    b['aux'][0,0,8:12,8:12] = 1
    result = losses(out,b)
    assert result['aux'].item() > 0
    result['total'].backward()
    assert model.aux.weight.grad is not None and torch.isfinite(model.aux.weight.grad).all()
    model.eval()
    pred = predict(model, torch.device('cpu'), np.zeros((8,32,32),np.float32),tile=32,tta='flip',batch=2)
    assert pred['aux_prob'].shape == (5,32,32)
    assert np.isfinite(pred['aux_prob']).all()
