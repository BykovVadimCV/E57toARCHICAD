"""Генератор квартир (ProcTHOR-подобный рост комнат)."""
import numpy as np
import pytest

from simscan.apartment import ApartmentGenerator, _corners
from simscan.config import load_config
from simscan.groundtruth import build_masks, frame_for
from simscan.interior import Furnisher
from simscan.render import render_plan
from simscan.scene import build_scene

SEEDS = range(6)


@pytest.fixture(scope="module")
def plans():
    cfg = load_config()
    out = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        lay = ApartmentGenerator(cfg.layout, rng).generate()
        Furnisher(cfg.interior, rng).furnish(lay)
        out.append(lay)
    return out


def test_all_rooms_reachable_from_entrance(plans):
    for lay in plans:
        entrance = [o for o in lay.openings if o.kind == "door" and len(o.rooms) == 1]
        assert len(entrance) == 1
        adj = {r.id: set() for r in lay.rooms}
        for o in lay.openings:
            if o.kind in ("door", "passage") and len(o.rooms) == 2:
                a, b = o.rooms
                adj[a].add(b)
                adj[b].add(a)
        seen, stack = {entrance[0].rooms[0]}, [entrance[0].rooms[0]]
        while stack:
            for m in adj[stack.pop()] - seen:
                seen.add(m)
                stack.append(m)
        assert seen == {r.id for r in lay.rooms}, lay.meta["program"]


def test_windows_only_where_expected(plans):
    for lay in plans:
        for o in lay.openings:
            if o.kind != "window" or not o.rooms:
                continue
            kinds = {lay.rooms[r].kind for r in o.rooms}
            assert "bath" not in kinds and "corridor" not in kinds
        lit = {r for o in lay.openings if o.kind == "window" for r in o.rooms}
        for r in lay.rooms:
            if r.name in ("Кухня", "Гостиная", "Спальня", "Комната", "Кухня-гостиная"):
                assert r.id in lit, f"{r.name} без окна"
        party = [w for w in lay.walls if w.role == "party"]
        assert all(o.kind != "window" for o in lay.openings
                   if lay.wall(o.wall_id).role == "party")
        assert party


def test_masks_and_render(plans, tmp_path):
    lay = plans[0]
    _, solids = build_scene(lay)
    m = build_masks(lay, solids, frame_for(lay, 10, 0.5), 1.25)
    assert m["walls"].any() and m["doors"].any() and m["windows"].any()
    render_plan(lay, tmp_path / "plan.png")
    assert (tmp_path / "plan.png").stat().st_size > 10_000


def test_corner_count():
    m = np.zeros((6, 6), bool)
    m[1:5, 1:5] = True
    assert _corners(m) == 4
    m[1:3, 3:5] = False                                  # L-образная
    assert _corners(m) == 6
