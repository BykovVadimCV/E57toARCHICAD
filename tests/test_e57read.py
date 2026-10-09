"""Собственный читатель E57 против libE57 (pye57): те же значения во всех типах полей."""
import numpy as np
import pye57

from simscan.e57read import E57Reader

from conftest import box_room_config
from e57_scaled_writer import write_scaled


def _read_all(path, index, chunk):
    r = E57Reader(path)
    s = r.scans[index]
    parts = {k: [] for k in s.field_names}
    for c in r.iter_points(index, s.field_names, chunk):
        for k, v in c.items():
            parts[k].append(v)
    return r, {k: np.concatenate(v) for k, v in parts.items()}


def test_matches_pye57_on_float_file(tmp_path):
    from simscan.generate import generate_scene

    cfg = box_room_config(5.0, 3.0, layout={"source": "simscan"}, export={"free_space_input": False})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    path = tmp_path / "s" / "scan.e57"
    with pye57.E57(str(path)) as e:
        ref = e.read_scan_raw(0)
        rot, tr = e.get_header(0).rotation, e.get_header(0).translation
    r, mine = _read_all(path, 0, 7_777)                 # кусок не кратен пакету
    for k, v in ref.items():
        assert np.array_equal(mine[k].astype(np.float64), v.astype(np.float64)), k
    assert np.allclose(r.scans[0].rotation, rot) and np.allclose(r.scans[0].translation, tr)


def test_scaled_integer_color_and_odd_bit_widths(tmp_path):
    path = str(tmp_path / "scaled.e57")
    ref = write_scaled(path, n=120_000)
    r, mine = _read_all(path, 0, 50_000)
    widths = {f.name: f.bits for f in r.scans[0].fields}
    assert widths["cartesianX"] == 20 and widths["intensity"] == 11 and widths["returnIndex"] == 0
    assert widths["cartesianInvalidState"] == 2
    for k, v in ref.items():
        assert np.allclose(mine[k], v, atol=1e-6), k
    assert r.scans[0].tree["sensorSerialNumber"] == "<скрыто>"


def test_inspect_noise_estimate_is_unbiased(tmp_path):
    """Мерка шума inspect-e57 на синтетике с известной sigma (нормальное падение)."""
    from simscan.generate import generate_scene
    from simscan.inspect_e57 import aggregate, inspect_e57

    cfg = box_room_config(6.0, 4.0, layout={"source": "simscan"}, export={"free_space_input": False},
                          scanner={"angular_step_deg": 0.2, "min_range_m": 0.3},
                          effects={"range_noise": True, "range_sigma_mm": [1.0, 1.0],
                                   "range_sigma_per_m_mm": [0.0, 0.0], "incidence_noise_power": 0.0})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    import json

    doc = json.loads((tmp_path / "s" / "layout.json").read_text())
    refl = doc["materials"]["wall"]
    rep = inspect_e57(tmp_path / "s" / "scan.e57", tmp_path / "r", figures=False, sample=False,
                      zip_result=False, log=lambda *a: None)
    est = aggregate(rep)["noise_mm"][1][0]              # 2-4 м, 0-30°
    true = 1.0 * np.sqrt(0.5 / max(refl, 0.05))         # как в scanner.py
    assert abs(est / true - 1) < 0.2, (est, true)


def test_inspect_unified_cloud(tmp_path):
    """Сведённое облако (как у заказчика): станции из поз снимков, план по покрытию."""
    import json

    from simscan.generate import generate_scene
    from simscan.inspect_e57 import inspect_e57

    cfg = box_room_config(5.0, 3.0, layout={"source": "simscan"},
                          scanner={"angular_step_deg": 0.25, "min_range_m": 0.3, "max_stations_per_room": 2,
                                   "stations_per_m2": 0.5},
                          export={"free_space_input": False, "write_merged": True, "merged_spacing_mm": 10.0})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    doc = json.loads((tmp_path / "s" / "layout.json").read_text())
    rep = inspect_e57(tmp_path / "s" / "scan_merged.e57", tmp_path / "r", sample=False, log=lambda *a: None)
    u = rep["unified"]
    assert u["setups_from_images"] == len(doc["stations"]) >= 2
    assert abs(u["ceiling_height_m"] - doc["ceiling_height"]) < 0.03
    assert 5 < u["point_spacing_mm"] < 20
    assert rep["plan"]["method"] == "coverage"
    assert abs(rep["plan"]["free_area_m2"] - 15.0) < 1.5


def test_detect_levels_two_storeys():
    """Двухэтажная квартира: пол, короб, потолок, перекрытие, пол, потолок."""
    from simscan.rasterize import detect_levels

    rng = np.random.default_rng(0)

    def plane(z, x1, n):
        return np.c_[rng.uniform(0, x1, n), rng.uniform(0, 8, n), z + rng.normal(0, 0.002, n)]

    walls = np.c_[rng.uniform(0, 10, 40_000), rng.choice([0.0, 8.0], 40_000), rng.uniform(0, 5.7, 40_000)]
    pts = np.vstack([plane(0.0, 10, 30_000), plane(2.2, 3, 8_000),      # пол и короб 2,2 м
                     plane(2.7, 10, 30_000), plane(2.94, 8, 25_000),   # потолок и пол 2-го уровня
                     plane(5.7, 8, 25_000), walls])
    lv = detect_levels(pts)
    assert len(lv) == 2
    assert abs(lv[0]["floor_z"]) < 0.03 and abs(lv[0]["ceiling_z"] - 2.7) < 0.03
    assert abs(lv[1]["floor_z"] - 2.94) < 0.03 and abs(lv[1]["ceiling_z"] - 5.7) < 0.03
