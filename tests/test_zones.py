"""Зоны вне увиденного (simscan/zones.py): лоджия со ступенью и пониженным потолком, где стоял
сканер, против пространства, видного только через окно."""
import numpy as np

from simscan.zones import CELL_M, analyse_zone, find_zones

LEVEL = {"floor_z": 10.0, "ceiling_z": 12.7}


def _grid(w=3.0, d=2.0):
    m = np.ones((int(round(d / CELL_M)), int(round(w / CELL_M))), bool)
    return m, (0.0, 0.0)


def _plane(rng, z, n, w=3.0, d=2.0, hole=None):
    p = np.c_[rng.uniform(0, w, n), rng.uniform(0, d, n), z + rng.normal(0, 0.002, n)]
    if hole is not None:
        (cx, cy), r = hole
        p = p[np.hypot(p[:, 0] - cx, p[:, 1] - cy) > r]
    return p


def test_loggia_with_step_and_low_ceiling():
    rng = np.random.default_rng(0)
    walls = np.c_[rng.uniform(0, 3, 20_000), rng.choice([0.0, 2.0], 20_000), rng.uniform(10.06, 12.45, 20_000)]
    xyz = np.vstack([_plane(rng, 10.06, 60_000, hole=((1.5, 1.0), 0.5)),     # пол на 6 см выше, слепой круг
                     _plane(rng, 12.45, 60_000), walls])                      # потолок на 25 см ниже
    r = analyse_zone(xyz, _grid(), LEVEL)
    assert abs(r["floor"]["step_m"] - 0.06) < 0.02 and abs(r["ceiling"]["drop_m"] - 0.25) < 0.02, r
    assert r["seen_level"] < 0.1 and r["seen_own"] > 0.9                    # уровнем не видно, своими - да
    c = r["scanner_circle"]
    assert c is not None and abs(c["radius_m"] - 0.5) < 0.1 and np.hypot(c["center_xy"][0] - 1.5, c["center_xy"][1] - 1.0) < 0.1
    assert "лоджия" in r["verdict"]


def test_view_through_window():
    rng = np.random.default_rng(1)
    far = np.c_[rng.uniform(0, 3, 30_000), np.full(30_000, 2.0), rng.uniform(10.8, 12.2, 30_000)]   # дальняя стена
    patch = np.c_[rng.uniform(2.0, 3.0, 2_000), rng.uniform(1.5, 2.0, 2_000), 9.9 + rng.normal(0, 0.002, 2_000)]
    r = analyse_zone(np.vstack([far, patch]), _grid(), LEVEL)
    assert r["scanner_circle"] is None
    assert "лоджия" not in r["verdict"], r["verdict"]


def test_find_zones():
    """Квартира (виден пол) и рядом область с точками без пола и потолка уровня - одна зона;
    слепой круг под сканером внутри квартиры - не зона."""
    px = 0.02
    x = np.zeros((8, 200, 400), np.float32)
    x[3][20:180, 20:200] = 1
    x[3][90:110, 100:120] = 0                                   # слепой круг
    x[5][40:160, 230:380] = 0.5                                 # точки лоджии (пол выше уровня)
    lab, apt = find_zones(x, px)
    assert lab.max() == 1
    assert (lab[60:140, 250:360] == 1).all() and (lab[20:180, 20:200] == 0).all()
    assert apt[100, 110]                                        # круг - внутри квартиры


def test_open_balcony_and_suspended_ceiling():
    """Пол ниже на 22 см и потолка нет, точки до ограждения - открытый балкон; пол выше на 7 см и
    потолок ниже на 46 см - подвесной потолок (санузел, коридор), а не лоджия (реальный скан)."""
    rng = np.random.default_rng(2)
    rail = np.c_[rng.uniform(0, 3, 10_000), np.full(10_000, 1.97), rng.uniform(9.78, 10.9, 10_000)]
    r = analyse_zone(np.vstack([_plane(rng, 9.78, 60_000), rail]), _grid(), LEVEL)
    assert r["ceiling"] is None and "балкон или терраса" in r["verdict"] and "ограждение" in r["verdict"], r["verdict"]
    walls = np.c_[rng.uniform(0, 3, 20_000), rng.choice([0.0, 2.0], 20_000), rng.uniform(10.07, 12.24, 20_000)]
    r = analyse_zone(np.vstack([_plane(rng, 10.07, 60_000), _plane(rng, 12.24, 60_000), walls]), _grid(), LEVEL)
    assert "подвесным потолком" in r["verdict"] and "лоджия" not in r["verdict"].split(";")[0], r["verdict"]
