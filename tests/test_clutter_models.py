"""Smoke tests for eight raycastable procedural clutter families."""
import numpy as np
import pytest

from simscan.clutter import KINDS, make
from simscan.meshes import prim_mesh


@pytest.mark.parametrize("kind", KINDS)
def test_clutter_meshes_are_finite_and_have_volume(kind):
    rng = np.random.default_rng(127)
    boxes, prims, radius = make(kind, 2.0, 3.0, 0.45, rng, yaw=0.5)
    assert boxes or prims
    assert radius > 0
    for b in boxes:
        assert np.isfinite(np.asarray(b.center)).all()
        assert np.isfinite(np.asarray(b.size)).all()
        assert min(b.size) > 0
    for p in prims:
        v, triangles = prim_mesh(p)
        assert len(v) > 0 and len(triangles) > 0
        assert np.isfinite(v).all()


def test_unknown_clutter_kind_refused():
    with pytest.raises(ValueError):
        make("wall", 0, 0, .5, np.random.default_rng(5))
