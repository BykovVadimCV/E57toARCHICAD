"""Генерация одной сцены и набора сцен.

Каталог сцены:
    scan.e57              по скану на станцию, точки в СК станции, поза в заголовке
    scan_merged.e57       (опция) то же облако одним сканом - режим «без станций»
    labels/scan_NNN.npz   метки точек в порядке записи в E57 (см. README)
    gt/                   эталонные маски и gt.json; gt/unet_mask.png - цель сети
    input/free_space.png  вход сети: псевдочертёж по свободному пространству (rasterize.py)
    layout.json           планировка, станции, позы (истинные и записанные), параметры эффектов
    preview.png           плотностной срез из E57 поверх контуров эталона
    mesh.ply              (опция) сетка сцены с цветами классов
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, replace
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import __version__
from .config import SynthConfig, config_to_dict
from .e57io import write_e57, write_merged_e57
from .groundtruth import (UNET_VALUES, build_masks, describe, frame_for, save_gt, unet_frame,
                          unet_mask)
from .interior import Furnisher
from .layout import Box, LayoutGenerator
from .preview import render_preview
from .scanner import ScanSimulator, people_for_station, place_stations, sample_effect_params
from .scene import build_scene
from . import realism
from .transform import WorldTransform, small_rotation


def scene_rng(seed: int, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([seed, index]))


_DATASETGEN = {}


def make_layout(cfg: SynthConfig, rng: np.random.Generator, seed: int, index: int):
    lc = cfg.layout
    if lc.source == "simscan":
        return LayoutGenerator(lc, rng).generate()
    if lc.source == "apartment":
        from .apartment import ApartmentGenerator

        return ApartmentGenerator(lc, rng).generate()
    if lc.source == "datasetgen":
        from .datasetgen_adapter import extract_plan, layout_from_plan, load_datasetgen

        if not lc.datasetgen_path:
            raise ValueError("layout.datasetgen_path не задан")
        dg = _DATASETGEN.get(lc.datasetgen_path) or load_datasetgen(lc.datasetgen_path)
        _DATASETGEN[lc.datasetgen_path] = dg
        dg_seed = int(np.random.SeedSequence([seed, index, 1]).generate_state(1)[0] % 2**31)
        plan = extract_plan(dg, dg_seed, lc.datasetgen_strategies,
                            lc.datasetgen_balcony_probability)
        return layout_from_plan(plan, lc, rng)
    raise ValueError(f"неизвестный layout.source: {lc.source}")


def generate_scene(cfg: SynthConfig, out_dir: str | Path, seed: int = 0, index: int = 0,
                   write_scan: bool = True) -> dict:
    """write_scan=False - scan.e57 и метки точек не пишутся, сканы с записанными позами
    возвращаются в summary["_scans"] = (scans, poses) (make-dataset растеризует их из памяти)."""
    t_start = time.perf_counter()
    rng = scene_rng(seed, index)
    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    layout = make_layout(cfg, rng, seed, index)
    real = realism.sample_params(cfg.realism, rng)
    if real["enabled"]:
        realism.jitter_thickness(layout, cfg.realism.thickness_jitter_mm, rng)
        layout.meta["warp"] = real["warp"]
    duplex = None
    if cfg.layout.p_duplex > 0 and rng.random() < cfg.layout.p_duplex:
        from .duplex import make_duplex

        duplex = make_duplex(layout, cfg, rng)          # None - лестница не встала: один этаж
    grids = Furnisher(cfg.interior, rng).furnish(layout)
    stations = place_stations(layout, grids, cfg.scanner, rng)
    level_of = [0] * len(stations)
    if duplex:
        from .duplex import build_duplex_scene, upper_realism

        up, dz = duplex["upper"], duplex["dz"]
        m = float(layout.meta.get("mess", 0.5))
        icfg = replace(cfg.interior, exterior_ground=False, mess=(m, m),
                       p_bare=1.0 if layout.meta.get("bare") else 0.0)
        grids_up = Furnisher(icfg, rng).furnish(up)
        st_up = place_stations(up, grids_up, cfg.scanner, rng)
        for st in st_up:                                 # верхний этаж: свои помещения в grids, z от пола 1-го
            st.id = len(stations)
            st.room_id += 1000
            st.position = (st.position[0], st.position[1], st.position[2] + dz)
            stations.append(st)
            level_of.append(1)
        grids.update({k + 1000: g for k, g in grids_up.items()})
        mesh, solids, _ = build_duplex_scene(layout, up, dz, real, upper_realism(real, rng))
    else:
        mesh, solids = build_scene(layout, real.get("tessellation_m"))
        realism.apply_to_mesh(mesh, real)
    if not stations:
        raise RuntimeError("не удалось поставить ни одной станции")
    for st in stations:                     # станции живут в той же (сдвинутой) СК, что и сцена
        xy = realism.warp_xy(np.array(st.position[:2])[None], real)[0]
        st.position = (float(xy[0]), float(xy[1]), st.position[2])

    ex = cfg.export
    world = WorldTransform(
        float(rng.uniform(0, 360)) if ex.world_yaw else 0.0,
        (float(rng.uniform(*ex.world_offset_m)), float(rng.uniform(*ex.world_offset_m)),
         float(rng.uniform(*ex.world_z_offset_m))),
    )
    params = sample_effect_params(cfg.effects, rng)
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, params, rng)

    scans, poses, station_info = [], [], []
    for k, st in enumerate(stations):
        people = []
        if cfg.effects.people:
            lo, hi = cfg.effects.people_per_station
            people = people_for_station(st, grids, int(rng.integers(lo, hi + 1)), rng,
                                        instance0=1_000_000 + 100 * k)
            for person in people:                # люди стоят в сдвинутой СК, как и сцена
                cx, cy = realism.warp_xy(np.array(person.box.center[:2])[None], real)[0]
                cz = person.box.center[2] + (duplex["dz"] if level_of[k] else 0.0)
                person.box = Box((float(cx), float(cy), cz), person.box.size, person.box.yaw)
        scan = sim.scan(st, people)
        # истинная поза в СК объекта
        R_true = world.R @ st.rotation()
        t_true = world.to_world(np.array(st.position)[None])[0]
        # записанная поза: первая станция - опорная, остальные с ошибкой регистрации
        R_w, t_w = R_true, t_true
        if cfg.effects.registration_error and k > 0:
            R_w = small_rotation(params["registration_sigma_deg"], rng) @ R_true
            t_w = t_true + rng.normal(0, params["registration_sigma_m"], 3)
        scans.append(scan)
        poses.append((R_w, t_w))
        station_info.append({
            "id": st.id, "room_id": st.room_id % 1000, "level": level_of[k],
            "position_layout_m": list(st.position),
            "heading_deg": st.heading_deg, "tilt_deg": list(st.tilt_deg),
            "pose_true": {"R": R_true.tolist(), "t": t_true.tolist()},
            "pose_written": {"R": np.asarray(R_w).tolist(), "t": np.asarray(t_w).tolist()},
            "people": len(people), "rays": int(len(scan.valid)), "valid": scan.n_valid,
            "mixed": int(scan.mixed.sum()), "virtual": int(scan.virtual.sum()),
        })

    names = [f"Station_{s.id + 1:03d}" for s in stations]
    masks_written = write_e57(out / "scan.e57", scans, poses, names, cfg.effects.keep_invalid) \
        if write_scan else []
    for scan, written in zip(scans, masks_written):
        np.savez_compressed(
            out / "labels" / f"scan_{scan.station_id:03d}.npz",
            label=scan.label[written], instance=scan.instance[written],
            virtual=scan.virtual[written], mixed=scan.mixed[written],
            valid=scan.valid[written], row=scan.row[written], col=scan.col[written])

    if ex.write_merged and write_scan:
        pts, inten = [], []
        for scan, (R, t) in zip(scans, poses):
            v = scan.valid
            pts.append(scan.xyz_local[v] @ np.asarray(R).T + t)
            inten.append(scan.intensity[v])
        write_merged_e57(out / "scan_merged.e57", np.vstack(pts), np.concatenate(inten),
                         stations=[t for _, t in poses], spacing_m=ex.merged_spacing_mm / 1000)

    frame = frame_for(layout, ex.pixel_mm, ex.pad_m)
    masks = build_masks(layout, solids, frame, ex.cut_height_m)
    info = describe(layout, masks, frame, world, ex.cut_height_m)
    info["stations"] = station_info
    if ex.unet_mask:
        uframe = unet_frame(layout, ex.unet_target_wall_px, ex.pad_m)
        info["unet"] = {"frame": asdict(uframe), "target_wall_px": ex.unet_target_wall_px,
                        "values": UNET_VALUES, "file": "unet_mask.png"}
    save_gt(out / "gt", masks, info)
    if ex.unet_mask:
        import cv2

        cv2.imwrite(str(out / "gt" / "unet_mask.png"),
                    unet_mask(layout, solids, uframe, ex.cut_height_m))

    doc = layout.to_dict()
    doc["meta"].update({
        "simscan_version": __version__, "seed": seed, "index": index,
        "layout_to_world": world.to_dict(), "effects_sampled": params,
        "realism": real,
        "scanner": {"nrow": sim.nrow, "ncol": sim.ncol,
                    "angular_step_deg": cfg.scanner.angular_step_deg},
        "config": config_to_dict(cfg),
    })
    doc["stations"] = station_info
    if duplex:                                           # уровни: нижний - сам doc, верхний - отдельно
        doc["levels"] = [{"z0": 0.0, "ceiling_height": layout.ceiling_height},
                         {"z0": duplex["dz"], "ceiling_height": duplex["upper"].ceiling_height,
                          "layout": duplex["upper"].to_dict()}]
    (out / "layout.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    if ex.write_mesh:
        mesh.save_ply(out / "mesh.ply")
    summary = {"dir": str(out), "levels": 2 if duplex else 1, "rooms": len(layout.rooms), "stations": len(stations),
               "openings": len(layout.openings), "items": len(layout.items),
               "points_valid": int(sum(s.n_valid for s in scans))}
    if ex.preview and write_scan:
        summary.update(render_preview(out / "scan.e57", masks, frame, world, out / "preview.png"))
    if ex.free_space_input and ex.unet_mask and write_scan:
        from .rasterize import make_pair

        pair = make_pair(out, figure=ex.debug)
        summary["input_wall_iou"] = pair["wall_iou"]
    if ex.debug:
        from .debug import debug_scene

        debug_scene(out)
    summary["seconds"] = round(time.perf_counter() - t_start, 2)
    if not write_scan:
        summary["_scans"] = (scans, poses)
    return summary


def _job(args):
    cfg, out, seed, index = args
    try:
        return generate_scene(cfg, out, seed, index)
    except Exception as exc:  # одна плохая сцена не должна ронять весь набор
        return {"dir": str(out), "error": f"{type(exc).__name__}: {exc}"}


def generate_dataset(cfg: SynthConfig, out_root: str | Path, count: int, seed: int = 0,
                     start: int = 0, workers: int = 1) -> list[dict]:
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [(cfg, root / f"scene_{i:05d}", seed, i) for i in range(start, start + count)]
    if workers <= 1:
        results = []
        for j in jobs:
            r = _job(j)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = []
            for r in pool.map(_job, jobs):
                print(json.dumps(r, ensure_ascii=False), flush=True)
                results.append(r)
    with open(root / "index.jsonl", "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results

