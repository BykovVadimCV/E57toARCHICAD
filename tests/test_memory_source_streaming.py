import numpy as np
from types import SimpleNamespace
from simscan.netinput import MemorySource


def test_preallocated_memory_source_matches_previous_projection_and_order():
    rng=np.random.default_rng(4);scans=[];poses=[]
    for angle in [.3,-.7]:
        xyz=rng.normal(size=(19,3));valid=rng.random(19)>.25
        scans.append(SimpleNamespace(xyz_local=xyz,valid=valid))
        c,s=np.cos(angle),np.sin(angle);poses.append((np.array([[c,-s,0],[s,c,0],[0,0,1]]),np.array([42.,36.,3.])+rng.normal(size=3)))
    source=MemorySource(scans,poses,chunk=3)
    expected=np.concatenate([(s.xyz_local[s.valid].astype(np.float32).astype(float)@R.T+(t-source.center)).astype(np.float32) for s,(R,t) in zip(scans,poses)])
    np.testing.assert_array_equal(source.points,expected)
    assert source.total==len(expected)
    np.testing.assert_array_equal(np.concatenate(list(source.chunks())),expected.astype(float)+source.center)
