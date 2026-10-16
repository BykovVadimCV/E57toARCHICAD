"""Привязка плана к облаку: сеть ошиблась на 2 см и 0,5° - грани по точкам возвращают стены
на место; рама окна снаружи и шкаф вплотную к стене не сбивают. Точки - синтетические, без E57."""
import math

import numpy as np

from simscan.cloudsnap import snap_plan

RNG = np.random.default_rng(0)


def _face(p0, p1, step=0.01, sigma=0.002):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.linalg.norm(p1 - p0) / step))
    t = np.linspace(0, 1, n)[:, None]
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    nrm = np.array([-d[1], d[0]])
    pts = p0 + (p1 - p0) * t
    return np.repeat(pts, 4, 0) + RNG.normal(0, sigma, (4 * n, 1)) * nrm


def _rot(p, deg, c=(3.5, 1.5)):
    a = math.radians(deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return ((np.asarray(p, float) - c) @ R.T + c).round(4).tolist()


def test_snap_recovers_walls():
    # правда: комнаты A (0..4 x 0..3) и B (4..7 x 0..3); перегородка x = 4, толщина 12 см;
    # наружные стены - внутренние грани на x = 0, 7 и y = 0, 3 (полоса 10 см наружу, как разметка)
    xy = np.vstack([
        _face([0, 0], [0, 0.5]), _face([0, 1.5], [0, 3]),          # x = 0, за шкафом грани не видно
        _face([0.03, 0.5], [0.03, 1.5], sigma=0.004),             # шкаф вплотную: 3 см от стены
        _face([7, 0], [7, 3]), _face([0, 0], [7, 0]), _face([0, 3], [7, 3]),
        _face([3.94, 0], [3.94, 3]), _face([4.06, 0], [4.06, 3]),  # перегородка, обе грани
        _face([1.0, 3.12], [2.0, 3.12]),                          # рама окна снаружи стены y = 3
    ])
    # «сеть»: всё сдвинуто на 2 см и повёрнуто на 0,5°
    sh = np.array([0.02, -0.02])
    walls = [("W1", [[-0.05, -0.05], [-0.05, 3.05]], 0.10, True), ("W2", [[7.05, -0.05], [7.05, 3.05]], 0.10, True),
             ("W3", [[-0.05, -0.05], [7.05, -0.05]], 0.10, True), ("W4", [[-0.05, 3.05], [7.05, 3.05]], 0.10, True),
             ("W5", [[4.0, 0.0], [4.0, 3.0]], 0.12, False)]
    plan = {"level": "L1", "floor_z": 0.0, "ceiling_z": 2.7,
            "walls": [{"id": i, "points": _rot(np.asarray(p) + sh, 0.5), "thickness": t, "is_outer": o, "rooms": [],
                       "length": 3.0} for i, p, t, o in walls],
            "openings": [], "rooms": [
                {"id": "R1", "name": "1", "points": _rot(np.array([[0, 0], [3.94, 0], [3.94, 3], [0, 3]]) + sh, 0.5),
                 "area": 11.8, "perimeter": 13.9},
                {"id": "R2", "name": "2", "points": _rot(np.array([[4.06, 0], [7, 0], [7, 3], [4.06, 3]]) + sh, 0.5),
                 "area": 8.8, "perimeter": 11.9}]}
    out = snap_plan(plan, xy, log=lambda *a: None)
    W = {w["id"]: (np.asarray(w["points"]), w["thickness"]) for w in out["walls"]}
    # перегородка: обе грани -> ось и толщина точно, направление - по граням
    p, t = W["W5"]
    assert np.abs(p[:, 0] - 4.0).max() < 0.004 and abs(t - 0.12) < 0.004, (p, t)
    # наружные: внутренняя грань - по облаку (шкаф и рама не мешают), стена параллельна правде
    p, t = W["W1"]
    assert abs((p[:, 0] + t / 2).mean() - 0.0) < 0.004 and abs(p[0, 0] - p[1, 0]) < 0.003, (p, t)
    p, t = W["W4"]
    assert abs((p[:, 1] - t / 2).mean() - 3.0) < 0.004 and abs(p[0, 1] - p[1, 1]) < 0.003, (p, t)
    # углы сведены: конец W1 - на оси W3
    a1 = np.asarray(W["W1"][0])
    assert min(abs(a1[:, 1] - W["W3"][0][:, 1].mean())) < 0.01
    # помещение - по новым граням: площадь A = 3,94 x 3
    areas = sorted(r["area"] for r in out["rooms"])
    assert abs(areas[1] - 3.94 * 3.0) < 0.05, areas
    assert out["metadata"]["snap"]["two_faces"] == 1
