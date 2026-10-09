import numpy as np

from simscan.config import load_config
from simscan.interior import Furnisher
from simscan.labels import LABEL_ID
from simscan.layout import LayoutGenerator
from simscan.scanner import ScanSimulator, place_stations, ray_grid, sample_effect_params
from simscan.scene import build_scene

from conftest import box_room_config


def _setup(cfg, seed=0):
    rng = np.random.default_rng(seed)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    grids = Furnisher(cfg.interior, rng).furnish(lay)
    stations = place_stations(lay, grids, cfg.scanner, rng)
    mesh, _ = build_scene(lay)
    params = sample_effect_params(cfg.effects, rng)
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, params, rng)
    return lay, stations, sim, params


def _to_layout(scan, st):
    return scan.xyz_local[scan.valid] @ st.rotation().T + np.asarray(st.position)


def test_ray_grid_shape():
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5}})
    dirs, rows, cols, nrow, ncol = ray_grid(cfg.scanner)
    assert ncol == 720 and nrow == 300
    assert dirs.shape == (nrow * ncol, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
    assert rows.max() == nrow - 1 and cols.max() == ncol - 1


def test_clean_points_lie_on_room_surfaces():
    # минимальная дальность снята: иначе часть лучей к ближней стене законно теряется
    cfg = box_room_config(5.0, 3.0, scanner={"min_range_m": 0.1})
    lay, stations, sim, _ = _setup(cfg)
    st = stations[0]
    scan = sim.scan(st)
    p = _to_layout(scan, st)
    H = lay.ceiling_height
    d = np.min(np.abs(np.c_[p[:, 0], p[:, 0] - 5.0, p[:, 1], p[:, 1] - 3.0,
                            p[:, 2], p[:, 2] - H]), axis=1)
    assert scan.valid.mean() > 0.99
    assert d.max() < 1e-4
    assert set(np.unique(scan.label[scan.valid])) <= {LABEL_ID["wall"], LABEL_ID["floor"],
                                                      LABEL_ID["ceiling"]}


def test_range_noise_magnitude():
    cfg = box_room_config(5.0, 3.0, scanner={"min_range_m": 0.1},
                          effects={"range_noise": True, "range_sigma_mm": [2.0, 2.0],
                                             "range_sigma_per_m_mm": [0.0, 0.0]})
    lay, stations, sim, _ = _setup(cfg)
    st = stations[0]
    sim.mesh.tri_reflectance[:] = 0.5
    scan = sim.scan(st)
    p = _to_layout(scan, st)
    # точки торцевой стены x = 5 при почти нормальном падении
    o = np.asarray(st.position)
    v = p - o
    sel = (np.abs(p[:, 0] - 5.0) < 0.02) & (np.abs(v[:, 1]) < 0.2 * np.abs(v[:, 0])) \
        & (np.abs(v[:, 2]) < 0.2 * np.abs(v[:, 0]))
    resid = p[sel, 0] - 5.0
    assert sel.sum() > 200
    assert 1.6e-3 < resid.std() < 2.4e-3


def test_glass_passes_and_mirror_virtual():
    cfg = box_room_config(6.0, 4.0,
                          layout={"p_window_per_side": 1.0, "min_window_room_area_m2": 1.0},
                          interior={"p_mirror_other": 1.0},
                          effects={"glass": True, "p_glass_pass": [1.0, 1.0], "mirrors": True})
    lay, stations, sim, _ = _setup(cfg, seed=2)
    assert any(o.kind == "window" for o in lay.openings)
    scan = sim.scan(stations[0])
    assert not (scan.label[scan.valid] == LABEL_ID["glass"]).any()
    if any(it.kind == "mirror" for it in lay.items):
        assert scan.virtual.any()
        assert not (scan.label[scan.valid] == LABEL_ID["mirror"]).any()


def test_people_differ_between_stations():
    from simscan.scanner import people_for_station

    cfg = box_room_config(8.0, 6.0, scanner={"max_stations_per_room": 3,
                                             "stations_per_m2": 1.0})
    rng = np.random.default_rng(0)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    grids = Furnisher(cfg.interior, rng).furnish(lay)
    stations = place_stations(lay, grids, cfg.scanner, rng)
    assert len(stations) >= 2
    a = people_for_station(stations[0], grids, 2, rng, 100)
    b = people_for_station(stations[1], grids, 2, rng, 200)
    assert a and b and a[0].box.center != b[0].box.center
