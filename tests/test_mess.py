"""Беспорядок (мягкие вещи), отсканированные предметы, загрузчик моделей, вход сети."""
import dataclasses
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from simscan import furniture as fu
from simscan.config import load_config
from simscan.interior import Furnisher
from simscan.layout import Box, LayoutGenerator
from simscan.meshes import prim_mesh
from simscan.scene import SceneMesh, Solid, build_scene

from conftest import box_room_config


def _cast(mesh, origin, direction):
    import open3d as o3d

    sc = mesh.to_raycasting_scene()
    r = np.array([[*origin, *direction]], np.float32)
    return float(sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()[0])


def _cube_obj(size=0.2):
    v = [f"v {x * size} {y * size} {z * size}" for x in (0, 1) for y in (0, 1) for z in (0, 1)]
    f = ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2",
         "f 3 4 8", "f 3 8 7", "f 1 3 7", "f 1 7 5", "f 2 6 8", "f 2 8 4"]
    return "\n".join(v + f)


def test_drape_covers_support_and_hangs():
    rng = np.random.default_rng(0)
    box = Box((0.0, 0.0, 0.25), (0.5, 0.4, 0.5), 0.3)
    v, t = prim_mesh(fu.drape(box, rng, size=(0.9, 0.9)))
    assert len(t) > 100 and t.max() < len(v)
    assert v[:, 2].min() >= -1e-6                                   # не уходит под пол
    mesh = SceneMesh.from_solids([Solid(box, "clutter", 1, 0.5),
                                  Solid(box, "clutter", 2, 0.5, mesh=(v, t))])
    # сверху луч попадает в ткань (выше верха коробки), а не в коробку
    t_hit = _cast(mesh, (0.0, 0.0, 2.0), (0.0, 0.0, -1.0))
    assert 2.0 - t_hit > 0.5
    # свисающая часть: точки ниже верха опоры и за её габаритом
    hang = v[(v[:, 2] < 0.45) & (np.hypot(v[:, 0], v[:, 1]) > 0.2)]
    assert len(hang) > 10


def test_hanging_cloth_stays_in_front_of_wall():
    rng = np.random.default_rng(1)
    v, _ = prim_mesh(fu.hanging_cloth((0.0, 0.0, 1.7), (1, 0, 0), (0, 1, 0), rng))
    assert v[:, 1].min() > -0.03 and v[:, 1].max() < 0.25
    assert v[:, 2].max() <= 1.7 + 1e-6 and v[:, 2].min() > 0.6


def test_leaning_board_touches_wall_and_floor():
    rng = np.random.default_rng(2)
    v, _ = prim_mesh(fu.leaning_board((0.0, 0.0), (1, 0), (0, 1), rng))
    assert abs(v[:, 2].min()) < 0.02 and v[:, 1].min() > -0.035     # стоит на полу, у стены y = 0
    top = v[v[:, 2] > v[:, 2].max() - 0.05]
    assert top[:, 1].min() < 0.05                                   # верх опирается на стену


def _messy(mess, scan_dir="", seed=0):
    cfg = load_config(overrides={"interior": {"mess": [mess, mess], "scan_dir": scan_dir}})
    from simscan.apartment import ApartmentGenerator

    rng = np.random.default_rng(seed)
    lay = ApartmentGenerator(cfg.layout, rng).generate()
    Furnisher(cfg.interior, rng).furnish(lay)
    return lay


def test_mess_level_controls_soft_items():
    soft = ("cloth", "jacket", "bag", "box_stack", "board")
    calm = sum(it.kind in soft for s in range(4) for it in _messy(0.0, seed=s).items)
    messy = [it for s in range(4) for it in _messy(1.0, seed=s).items if it.kind in soft]
    assert calm == 0 and len(messy) >= 4
    assert all(it.label == "clutter" for it in messy)
    lay = _messy(1.0, seed=1)
    assert lay.meta["mess"] == 1.0
    mesh, _ = build_scene(lay)
    assert len(mesh.triangles) > 0


def test_scans_placed_at_native_size(tmp_path):
    for role in ("shoes", "tabletop", "floor"):
        (tmp_path / role).mkdir()
        (tmp_path / role / "cube.obj").write_text(_cube_obj(0.2))
    found = set()
    for seed in range(6):
        lay = _messy(1.0, str(tmp_path), seed)
        for it in lay.items:
            if not it.kind.startswith("scan:"):
                continue
            found.add(it.kind)
            for pr in it.prims:
                v, _ = prim_mesh(pr)
                ext = v.max(0) - v.min(0)
                assert abs(ext[2] - 0.2) < 1e-6                     # высота - натуральная
                assert abs(v[:, 2].min() - pr["center"][2]) < 1e-6   # стоит дном на опоре
    assert {"scan:shoes", "scan:floor"} <= found


def test_gso_fetch_offline(tmp_path, monkeypatch):
    """Загрузчик GSO без сети: манифест и архив подменены."""
    from simscan import assets

    manifest = {"assets": {
        "shoe_a": {"license": "CC BY-SA 4.0", "metadata": {"category": "Shoe", "description": "Shoe A"}},
        "thing": {"license": "CC BY-SA 4.0", "metadata": {"category": "None"}},
    }}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        data = _cube_obj(0.3).encode()
        info = tarfile.TarInfo("shoe_a/meshes/visual_geometry.obj")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def fake_get(url, timeout=120.0):
        if url.endswith("GSO.json"):
            return json.dumps(manifest).encode()
        assert url.endswith("shoe_a.tar.gz")
        return buf.getvalue()

    monkeypatch.setattr(assets, "_get", fake_get)
    counts = assets.fetch_gso(tmp_path, per_role=5, log=lambda *a: None)
    assert counts == {"shoes": 1}
    v, t = assets.read_obj((tmp_path / "shoes" / "shoe_a.obj").read_bytes())
    assert len(t) == 12 and np.allclose(v.max(0) - v.min(0), 0.3)
    rows = (tmp_path / "ATTRIBUTION.csv").read_text().splitlines()
    assert len(rows) == 2 and "CC BY-SA 4.0" in rows[1] and "shoes/shoe_a.obj" in rows[1]


def test_free_space_input_matches_box_room(tmp_path):
    """Пустая комната 5 x 3 с одной станцией: свободное пространство = комната, стены найдены."""
    from simscan.generate import generate_scene
    from simscan.rasterize import INPUT_VALUES, make_pair

    cfg = box_room_config(5.0, 3.0, export={"unet_mask": True, "free_space_input": False},
                          layout={"source": "simscan"})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    stats = make_pair(tmp_path / "s", figure=False)
    import cv2

    img = cv2.imread(str(tmp_path / "s" / "input" / "free_space.png"), cv2.IMREAD_GRAYSCALE)
    gt = json.loads((tmp_path / "s" / "gt" / "gt.json").read_text())
    px = gt["unet"]["frame"]["pixel_m"]
    # свободное пространство - белая связная область, в которой стоит станция
    from simscan.groundtruth import RasterFrame

    frame = RasterFrame(**gt["unet"]["frame"])
    n, lab = cv2.connectedComponents((img == INPUT_VALUES["free"]).astype(np.uint8), connectivity=4)
    i, j = frame.xy_to_ij(*gt["stations"][0]["position_layout_m"][:2])
    inner = (lab == lab[int(round(float(i))), int(round(float(j)))]).sum()
    assert abs(inner * px * px - 15.0) < 0.6
    assert stats["wall_recall"] > 0.9
    assert stats["stations"][0]["bins_empty"] == 0
    assert stats["wall_iou_scanned"] >= stats["wall_iou"] - 1e-9
    assert (tmp_path / "s" / "input" / "valid.png").exists()


def test_inspect_e57_recovers_scene_facts(tmp_path):
    """Анализатор реального E57 на синтетике: шаг, сетка, высота станции, потолок, архив."""
    from simscan.generate import generate_scene
    from simscan.inspect_e57 import inspect_e57

    cfg = box_room_config(5.0, 3.0, export={"free_space_input": False}, layout={"source": "simscan"},
                          scanner={"angular_step_deg": 0.5, "min_range_m": 0.1})
    generate_scene(cfg, tmp_path / "s", seed=0, index=0)
    doc = json.loads((tmp_path / "s" / "layout.json").read_text())
    rep = inspect_e57(tmp_path / "s" / "scan.e57", tmp_path / "rep", chunk=50_000, log=lambda *a: None)
    st, true = rep["scans"][0], doc["stations"][0]
    assert abs(st["azimuth_step_deg"] - 0.5) < 0.01 and abs(st["elevation_step_deg"] - 0.5) < 0.01
    assert st["grid_rows_cols"] == [doc["meta"]["scanner"]["nrow"], doc["meta"]["scanner"]["ncol"]]
    assert abs(-st["floor_rel_m"] - true["position_layout_m"][2]) < 0.02
    assert abs(st["ceiling_rel_m"] - st["floor_rel_m"] - doc["ceiling_height"]) < 0.02
    assert rep["file"]["points_total"] == sum(s["rays"] for s in doc["stations"])
    assert abs(rep["plan"]["free_area_m2"] - 15.0) < 1.5
    for name in ("report.json", "summary.txt", "header_tree.json", "plan.png", "sensor.png",
                 "sample.npz", "input.png"):
        assert (tmp_path / "rep" / name).exists(), name
    assert Path(rep["zip"]).exists()
    s = np.load(tmp_path / "rep" / "sample.npz")
    assert len(s["xyz"]) > 1000 and set(np.unique(s["station"])) <= set(range(len(doc["stations"])))
