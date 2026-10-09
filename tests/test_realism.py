"""Реалистичность: неидеальная геометрия (A) и физика сканера (B)."""
import dataclasses

import numpy as np
import pytest

from simscan import furniture as fu
from simscan.config import load_config
from simscan.groundtruth import build_masks, frame_for
from simscan.interior import Furnisher
from simscan.labels import LABEL_ID
from simscan.layout import Box, LayoutGenerator
from simscan.realism import apply_to_mesh, jitter_thickness, sample_params, warp_xy
from simscan.scanner import ScanSimulator, Station, sample_effect_params
from simscan.scene import SceneMesh, Solid, build_scene

from conftest import CLEAN_EFFECTS, box_room_config


def _room(seed=0, **real):
    cfg = box_room_config(5.0, 3.0, scanner={"min_range_m": 0.1})
    cfg = dataclasses.replace(cfg, realism=dataclasses.replace(
        cfg.realism, enabled=True, **real))
    rng = np.random.default_rng(seed)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    Furnisher(cfg.interior, rng).furnish(lay)
    params = sample_params(cfg.realism, rng)
    lay.meta["warp"] = params["warp"]
    mesh, solids = build_scene(lay, params["tessellation_m"])
    apply_to_mesh(mesh, params)
    return cfg, lay, mesh, solids, params, rng


def test_deformed_room_stays_closed():
    """Неровность, завал и сдвиг плана не открывают щелей: все лучи попадают в оболочку."""
    cfg, lay, mesh, _, params, rng = _room(wall_unevenness_mm=(4.0, 4.0), lean_mm_per_m=(3.0, 3.0),
                                           shear_deg_sigma=0.5)
    st = Station(0, 0, (*warp_xy(np.array([[2.5, 1.5]]), params)[0], 1.5), 30.0)
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, sample_effect_params(cfg.effects, rng), rng)
    scan = sim.scan(st)
    assert scan.valid.mean() > 0.999
    p = scan.xyz_local[scan.valid] @ st.rotation().T + np.asarray(st.position)
    # все точки в пределах номинальной коробки с учётом сдвига и деформации (< 3 см)
    q = p.copy()
    q[:, :2] = q[:, :2] @ np.linalg.inv(np.asarray(params["warp"])).T
    H = lay.ceiling_height
    assert q[:, 0].min() > -0.03 and q[:, 0].max() < 5.03
    assert q[:, 1].min() > -0.03 and q[:, 1].max() < 3.03
    assert q[:, 2].min() > -0.03 and q[:, 2].max() < H + 0.03
    # и поверхность действительно неровная (не плоскость)
    wall = q[np.abs(q[:, 0] - 5.0) < 0.03]
    assert wall[:, 0].std() > 0.0005


def test_warped_ground_truth_keeps_area():
    _, lay, _, solids, params, _ = _room(shear_deg_sigma=1.0)
    assert abs(params["shear_deg"]) > 0
    m = build_masks(lay, solids, frame_for(lay, 10, 0.5), 1.25)
    area = (m["rooms"] == 1).sum() * 1e-4
    assert abs(area - 15.0) < 0.15                           # сдвиг сохраняет площадь


def test_thickness_jitter_changes_walls():
    cfg = load_config()
    rng = np.random.default_rng(1)
    lay = LayoutGenerator(load_config(overrides={"layout": {"source": "simscan"}}).layout,
                          rng).generate()
    before = [w.thickness for w in lay.walls]
    jitter_thickness(lay, 5.0, rng)
    after = [w.thickness for w in lay.walls]
    assert before != after and max(abs(a - b) for a, b in zip(before, after)) < 0.03
    del cfg


def _step_scene(sep: float, divergence_mrad: float = 6.0):
    """Стена x = 5 и плита перед ней x = 4.85: перепад 15 см."""
    solids = [Solid(Box((5.05, 0.0, 1.5), (0.1, 10.0, 6.0)), "wall", 1, 0.6),
              Solid(Box((4.80, -1.0, 1.5), (0.1, 2.0, 6.0)), "furniture", 2, 0.6)]
    mesh = SceneMesh.from_solids(solids)
    cfg = load_config(overrides={
        "scanner": {"angular_step_deg": 0.25, "elevation_min_deg": -20.0,
                    "elevation_max_deg": 20.0, "min_range_m": 0.1},
        "effects": {**CLEAN_EFFECTS, "beam_model": True, "echo_separation_m": sep,
                    "beam_divergence_mrad": [divergence_mrad, divergence_mrad]}})
    rng = np.random.default_rng(0)
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, sample_effect_params(cfg.effects, rng), rng)
    st = Station(0, 0, (0.0, 0.0, 1.5), 0.0)
    scan = sim.scan(st)
    p = scan.xyz_local[scan.valid] @ st.rotation().T + np.asarray(st.position)
    return scan, p


def test_beam_mixes_close_echoes_only():
    # плита занимает x 4,75..4,85 (лицевая и боковая грани), стена - x 5,0;
    # точки в промежутке 4,86..4,99 бывают только от смешения эха плиты и стены
    scan, p = _step_scene(sep=0.4)
    between = (p[:, 0] > 4.86) & (p[:, 0] < 4.99)
    assert scan.mixed.any() and between.sum() > 0
    _, p2 = _step_scene(sep=0.05)                             # эхо разделимы - не смешиваются
    between2 = (p2[:, 0] > 4.86) & (p2[:, 0] < 4.99)
    assert between2.sum() == 0


def test_noise_grows_with_incidence():
    cfg = box_room_config(6.0, 6.0, scanner={"min_range_m": 0.1},
                          effects={"range_noise": True, "range_sigma_mm": [2.0, 2.0],
                                   "range_sigma_per_m_mm": [0.0, 0.0]})
    rng = np.random.default_rng(0)
    lay = LayoutGenerator(cfg.layout, rng).generate()
    mesh, _ = build_scene(lay)
    mesh.tri_reflectance[:] = 0.5
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, sample_effect_params(cfg.effects, rng), rng)
    sim.keep_diag = True
    scan = sim.scan(Station(0, 0, (3.0, 3.0, 1.5), 0.0))
    d = scan.diag
    err = np.linalg.norm(scan.xyz_local, axis=1) - d["t_true"]
    ok = scan.valid & np.isfinite(d["t_true"])
    low = ok & (d["cos_inc"] > 0.95)
    high = ok & (d["cos_inc"] < 0.35) & (d["cos_inc"] > 0.2)
    assert err[high].std() > 2.0 * err[low].std()


def _cast(mesh, origin, direction):
    import open3d as o3d

    sc = mesh.to_raycasting_scene()
    r = np.array([[*origin, *direction]], np.float32)
    return float(sc.cast_rays(o3d.core.Tensor(r))["t_hit"].numpy()[0])


def test_chair_is_see_through_below_seat():
    rng = np.random.default_rng(0)
    boxes = fu.chair(Box((0.0, 0.0, 0.45), (0.44, 0.42, 0.9), 0.0), rng)
    mesh = SceneMesh.from_solids([Solid(b, "furniture", 1, 0.5) for b in boxes])
    assert np.isinf(_cast(mesh, (-2.0, 0.0, 0.2), (1.0, 0.0, 0.0)))   # под сиденьем - насквозь
    assert np.isfinite(_cast(mesh, (0.0, 0.0, 2.0), (0.0, 0.0, -1.0)))  # сверху - сиденье


def test_tulle_lets_part_of_rays_through():
    from simscan.meshes import curtain_mesh

    v, t = curtain_mesh((2.0, -3.0), (2.0, 3.0), -2.0, 4.0, 0.02, 0.1)
    solids = [Solid(Box((5.05, 0.0, 1.0), (0.1, 10.0, 6.0)), "wall", 1, 0.6),
              Solid(Box((2.0, 0.0, 1.0), (0.1, 6.0, 6.0)), "curtain", 2, 0.6, mesh=(v, t),
                    transmit=0.5)]
    mesh = SceneMesh.from_solids(solids)
    cfg = load_config(overrides={
        "scanner": {"angular_step_deg": 0.5, "elevation_min_deg": -20.0, "elevation_max_deg": 20.0},
        "effects": dict(CLEAN_EFFECTS)})
    rng = np.random.default_rng(0)
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, sample_effect_params(cfg.effects, rng), rng)
    scan = sim.scan(Station(0, 0, (0.0, 0.0, 1.0), 0.0))
    front = scan.valid & np.isin(scan.label, [LABEL_ID["curtain"], LABEL_ID["wall"]])
    frac = (scan.label[front] == LABEL_ID["curtain"]).mean()
    assert 0.4 < frac < 0.6


def test_external_assets_replace_items(tmp_path):
    d = tmp_path / "assets" / "sofa"
    d.mkdir(parents=True)
    (d / "cube.obj").write_text("\n".join(
        [f"v {x} {y} {z}" for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        + ["f 1 2 4", "f 1 4 3", "f 5 7 8", "f 5 8 6", "f 1 5 6", "f 1 6 2",
           "f 3 4 8", "f 3 8 7", "f 1 3 7", "f 1 7 5", "f 2 6 8", "f 2 8 4"]))
    cfg = load_config(overrides={"interior": {"asset_dir": str(tmp_path / "assets"), "p_asset": 1.0,
                                              "furniture_per_m2": 0.6}})
    found = False
    for seed in range(8):
        rng = np.random.default_rng(seed)
        from simscan.apartment import ApartmentGenerator

        lay = ApartmentGenerator(cfg.layout, rng).generate()
        Furnisher(cfg.interior, rng).furnish(lay)
        sofas = [it for it in lay.items if it.kind == "sofa"]
        if sofas:
            assert all(it.prims and it.prims[0]["type"] == "mesh" for it in sofas)
            mesh, _ = build_scene(lay)
            assert len(mesh.triangles) > 0
            found = True
            break
    assert found


@pytest.mark.parametrize("seed", [2])
def test_debug_outputs(tmp_path, seed):
    from simscan.debug import debug_scene
    from simscan.generate import generate_scene

    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5},
                                 "export": {"preview": False}})
    generate_scene(cfg, tmp_path / "s", seed=seed, index=0)
    stats = debug_scene(tmp_path / "s")
    for name in ("plan", "panorama", "edge_closeup", "wall_flatness", "noise_vs_incidence",
                 "furniture", "mess", "mess_scan"):
        assert (tmp_path / "s" / "debug" / f"{name}.png").exists(), name
    assert stats["panorama"]["valid_fraction"] > 0.5
