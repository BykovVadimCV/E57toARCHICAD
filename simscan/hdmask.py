"""Native high-resolution architectural masks aligned to synthetic dataset world coordinates.

This renderer uses *procedural polygons*, never an upscaled sem.png or a noisy
point-cloud projection. The HD frame contains both the dataset frame and all
physical structural footprints; extra pixels are padding, not interpolation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .dataset_contract import atomic_json
from .netinput import SceneGT, _box_quad, _fill, affine

HD_FILES = ("wall_mask_hd.png", "wall_edges_hd.png", "openings_hd.png",
            "wall_mask_hd.json")


def parse_size(value: str | tuple[int, int]) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT in pixels, with an explicit upper bound."""
    if isinstance(value, str):
        try:
            w, h = (int(x) for x in value.lower().split("x"))
        except (ValueError, AttributeError) as exc:
            raise ValueError("Expected WIDTHxHEIGHT, for example 2048x1152") from exc
    else:
        w, h = value
    if not 1 <= w <= 32768 or not 1 <= h <= 32768:
        raise ValueError("HD mask dimensions must be between 1 and 32768")
    return int(w), int(h)


def _wall_polygons(gt: SceneGT) -> list[np.ndarray]:
    polys = []
    for wall in gt.layout.walls:
        if wall.role == "open_balcony":
            continue
        polys.append(gt.wall_quad(wall, -wall.ext0, wall.length + wall.ext1,
                                  -wall.thickness / 2, wall.thickness / 2))
    for item in gt.layout.items:
        if item.label == "column":
            for box in item.boxes:
                polys.append(gt.poly(_box_quad(box)))
    return polys


def _opening_polygons(gt: SceneGT) -> list[tuple[int, np.ndarray]]:
    """1=door, 2=window, 3=full-height opening."""
    out = []
    for op in gt.layout.openings:
        wall = gt.layout.wall(op.wall_id)
        if wall.role == "open_balcony":
            continue
        top = wall.height if wall.height is not None else gt.ceiling_h
        cls = 2 if op.kind == "window" else (3 if op.z1 >= top - 0.05 else 1)
        poly = gt.wall_quad(wall, op.s - op.width / 2, op.s + op.width / 2,
                            -wall.thickness / 2 - 0.01, wall.thickness / 2 + 0.01)
        out.append((cls, poly))
    return out


def make_hd_frame(base: dict, polygons: list[np.ndarray], pixel_mm: float = 2.0,
                  min_size: tuple[int, int] = (2048, 1152),
                  max_pixels: int = 64_000_000) -> dict:
    """Choose a native metric grid with at least min_size and no clipped walls."""
    if not math.isfinite(pixel_mm) or not 0.5 <= pixel_mm <= 100:
        raise ValueError("HD pixel size must be between 0.5 and 100 mm")
    mw, mh = parse_size(min_size)
    p = pixel_mm / 1000.0
    # HD exports follow the true architectural footprint, not the randomized
    # 0.5-2.5 m point-cloud frame padding. At 2 mm that padding can waste tens
    # of millions of pixels while contributing no architectural information.
    if polygons:
        verts = np.concatenate(polygons, axis=0)
        lo = verts.min(axis=0) - 0.10
        hi = verts.max(axis=0) + 0.10
    else:
        bp = float(base["pixel_m"])
        lo = np.array((base["origin_x"], base["origin_y"]), dtype=float)
        hi = lo + np.array((base["width"], base["height"]), dtype=float) * bp
    w = max(mw, int(math.ceil((hi[0] - lo[0]) / p)) + 2)
    h = max(mh, int(math.ceil((hi[1] - lo[1]) / p)) + 2)
    if w * h > max_pixels:
        raise ValueError(
            f"HD frame {w}x{h} exceeds {max_pixels:,} pixels at {pixel_mm:g} mm/px; "
            "reduce the generated footprint or choose a larger --hd-pixel-mm"
        )
    center = (lo + hi) / 2
    origin = np.floor((center - np.array((w, h)) * p / 2) / p) * p
    return {"origin_x": round(float(origin[0]), 6),
            "origin_y": round(float(origin[1]), 6),
            "pixel_m": p, "width": w, "height": h}


def render_hd_masks(gt: SceneGT, base_frame: dict, pixel_mm: float = 2.0,
                    min_size: tuple[int, int] = (2048, 1152)):
    """Rasterize wall bodies and openings directly, at the requested metric pitch."""
    walls_poly = _wall_polygons(gt)
    openings_poly = _opening_polygons(gt)
    frame = make_hd_frame(base_frame, walls_poly + [p for _, p in openings_poly],
                          pixel_mm, min_size)
    shape = (frame["height"], frame["width"])
    structure = np.zeros(shape, np.uint8)
    for poly in walls_poly:
        _fill(structure, poly, frame, 255)
    openings = np.zeros(shape, np.uint8)
    for cls, poly in openings_poly:
        _fill(openings, poly, frame, cls)
    # Opening labels must not erase structure outside a wall body.
    openings[structure == 0] = 0
    structure[openings != 0] = 0
    # Single-pixel geometry outlines, not a resized/blurred edge image.
    contours, _ = cv2.findContours(structure, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    edges = np.zeros(shape, np.uint8)
    cv2.drawContours(edges, contours, -1, 255, 1)
    return frame, structure, openings, edges


def write_hd_level(scene_dir, level_dir, pixel_mm: float = 2.0,
                   min_size: tuple[int, int] = (2048, 1152)) -> dict:
    """Write independent full-resolution outputs for one generated dataset level."""
    folder = Path(level_dir)
    meta = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    level = int(meta["level"][1:]) - 1
    gt = SceneGT(scene_dir, level)
    rot_deg = float(meta.get("aug", {}).get("rot_deg", 0.0))
    if rot_deg:
        stations = np.array([s["pose_true"]["t"] for s in gt.doc["stations"]], float)
        gt.rot = (rot_deg, float(stations[:, 0].mean()),
                  float(stations[:, 1].mean()))
    frame, wall, openings, edges = render_hd_masks(
        gt, meta["frame"], pixel_mm, min_size)
    data = {
        "format": "simscan-native-hd-masks-v1",
        "frame": frame,
        "pixel_to_world": affine(frame),
        "source": "physical layout polygons, not a resized raster",
        "wall_values": {"background": 0, "wall": 255},
        "opening_values": {"background": 0, "door": 1, "window": 2, "passage": 3},
        "level": meta["level"],
        "rot_deg": rot_deg,
        "min_size_px": list(parse_size(min_size)),
    }
    for name, image in (("wall_mask_hd.png", wall), ("openings_hd.png", openings),
                        ("wall_edges_hd.png", edges)):
        if not cv2.imwrite(str(folder / name), image):
            raise OSError(f"Could not write HD mask {folder / name}")
    atomic_json(folder / "wall_mask_hd.json", data)
    return data


def complete_hd_level(folder, pixel_mm: float, min_size: tuple[int, int]) -> bool:
    """Require the optional HD sidecars on resume, not merely a completed low-res level."""
    try:
        folder = Path(folder)
        if any(not (folder / name).is_file() for name in HD_FILES):
            return False
        doc = json.loads((folder / HD_FILES[-1]).read_text(encoding="utf-8"))
        frame = doc["frame"]
        size = (frame["width"], frame["height"])
        if size[0] < min_size[0] or size[1] < min_size[1]:
            return False
        if not math.isclose(frame["pixel_m"], pixel_mm / 1000, abs_tol=1e-10):
            return False
        for name in HD_FILES[:3]:
            image = cv2.imread(str(folder / name), cv2.IMREAD_UNCHANGED)
            if image is None or image.shape != (size[1], size[0]):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
