import numpy as np
import pytest

from simscan.config import load_config
from simscan.interior import Furnisher
from simscan.layout import Layout, LayoutGenerator

SEEDS = range(60)


def make(seed, cfg=None):
    cfg = cfg or load_config()
    rng = np.random.default_rng(seed)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    Furnisher(cfg.interior, rng).furnish(lay)
    return lay


@pytest.mark.parametrize("seed", SEEDS)
def test_openings_inside_walls_and_disjoint(seed):
    lay = make(seed)
    for w in lay.walls:
        ops = sorted(lay.openings_on(w.id), key=lambda o: o.s)
        for o in ops:
            assert o.s - o.width / 2 >= -1e-6 and o.s + o.width / 2 <= w.length + 1e-6
            assert 0 <= o.z0 < o.z1 < lay.ceiling_height
            if o.kind == "window":
                assert w.kind == "exterior"
            if len(o.rooms) == 2:
                assert w.kind == "interior"
        for a, b in zip(ops, ops[1:]):
            assert b.s - b.width / 2 >= a.s + a.width / 2 + 0.2 - 1e-6


@pytest.mark.parametrize("seed", SEEDS)
def test_rooms_have_positive_clear_area(seed):
    lay = make(seed)
    for r in lay.rooms:
        assert r.area > 1.0


def test_rooms_mostly_connected():
    """Все помещения достижимы через двери почти всегда (исключение - нет участка под дверь)."""
    ok = 0
    for seed in SEEDS:
        lay = make(seed)
        adj = {r.id: set() for r in lay.rooms}
        for o in lay.openings:
            if len(o.rooms) == 2:
                a, b = o.rooms
                adj[a].add(b)
                adj[b].add(a)
        seen, stack = {0}, [0]
        while stack:
            for m in adj[stack.pop()] - seen:
                seen.add(m)
                stack.append(m)
        ok += len(seen) == len(lay.rooms)
    assert ok >= 0.95 * len(SEEDS)


def test_exterior_walls_close_outline():
    """Наружный контур замкнут: у каждого конца наружной стены есть другая наружная стена."""
    for seed in SEEDS:
        lay = make(seed)
        ext = [w for w in lay.walls if w.kind == "exterior"]
        for w in ext:
            for s in (-w.ext0, w.length + w.ext1):
                p = w.point(s)
                near = [v for v in ext if v.id != w.id and _dist_to_body(v, p) < 1e-6]
                assert near, f"seed {seed}: открытый конец наружной стены {w.id}"


def _dist_to_body(w, p):
    rel = np.asarray(p) - np.asarray(w.p0)
    s, off = float(rel @ w.u), float(rel @ w.n)
    ds = max(-w.ext0 - s, 0.0, s - (w.length + w.ext1))
    doff = max(abs(off) - w.thickness / 2, 0.0)
    return float(np.hypot(ds, doff))


def test_json_roundtrip(tmp_path):
    lay = make(3)
    path = tmp_path / "layout.json"
    lay.save_json(path)
    again = Layout.load_json(path)
    assert again.to_dict() == lay.to_dict()


def test_deterministic():
    assert make(11).to_dict() == make(11).to_dict()
