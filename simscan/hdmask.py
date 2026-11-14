"""Native high-resolution architectural masks aligned to synthetic dataset world coordinates.

This renderer uses *procedural polygons*, never an upscaled sem.png or a noisy
point-cloud projection. Its canvas tightly frames physical structural geometry
while the saved affine maps every HD pixel back into the original world coordinates.
"""
from __future__ import annotations

import json
import hashlib
import zipfile
import math
from pathlib import Path

import cv2
import numpy as np

from .dataset_contract import atomic_json
from .semantic_contract import (HD_MASK_FORMAT, HD_OPENING_VALUES,
                                OPENING_SEMANTICS_VERSION, hd_opening_value)
from .netinput import SceneGT, _box_quad, _fill, affine

# HD labels are genuinely rasterized from procedural geometry, while scan
# evidence is projected directly from simulated XYZ. Keeping them separate
# prevents a refinement model from accidentally learning ground-truth leakage.
HD_LABEL_FILES = ("wall_mask_hd.png", "wall_edges_hd.png", "openings_hd.png",
                  "opening_edges_hd.png", "floor_void_hd.png",
                  "stair_footprint_hd.png", "supervision_valid_hd.png")
from .hd_evidence import SCAN_HD_FILES, project_scan_bands, align_coarse_mask
from .observation import (OBSERVATION_CONTRACT, OBSERVATION_FILES,
                          build_observation_masks, audit_object_support)
HD_FILES = HD_LABEL_FILES + OBSERVATION_FILES + SCAN_HD_FILES + ("scan_counts_hd.npz", "wall_mask_hd.json", "scan_evidence_hd.json")


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
    """1=door leaf, 2=window, 3=leafless passage; independent of height."""
    out = []
    for op in gt.layout.openings:
        wall = gt.layout.wall(op.wall_id)
        if wall.role == "open_balcony":
            continue
        cls = hd_opening_value(op.kind)
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


def _rectangles(gt: SceneGT, frame: dict, rects) -> np.ndarray:
    """Native geometric footprint masks, rotated into the scan's world frame."""
    mask = np.zeros((frame["height"], frame["width"]), np.uint8)
    for x0, y0, x1, y1 in rects:
        poly = gt.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        _fill(mask, poly, frame, 255)
    return mask


def write_hd_level(scene_dir, level_dir, pixel_mm: float = 2.0,
                   min_size: tuple[int, int] = (2048, 1152),
                   source=None, ray_samples=None) -> dict:
    """Write high-res geometry, raw XYZ evidence, and physically correct voids.

    A live point source is required; otherwise native scan evidence cannot be
    exported and the sidecars would be incomplete.
    """
    if source is None:
        raise ValueError("HD refinement export requires the simulated point source")
    folder = Path(level_dir)
    (folder / 'hd_complete.json').unlink(missing_ok=True)
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
    # Aperture boundaries are native pixel edges, not upscaled 20-mm classes.
    aperture = (openings != 0).astype(np.uint8)
    contours, _ = cv2.findContours(aperture, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    opening_edges = np.zeros_like(aperture)
    cv2.drawContours(opening_edges, contours, -1, 255, 1)

    # The lower floor can contain *stairs*, but a stair footprint is not a
    # missing floor. Only upper-slab physical holes are class-7 floor voids.
    slab = gt.layout.meta.get("slab", {})
    true_floor_holes = slab.get("floor", {}).get("holes", [])
    floor_void = _rectangles(gt, frame, true_floor_holes)
    stair = gt.layout.meta.get("duplex", {}).get("stair", {})
    staircase = (_rectangles(gt, frame, [stair["footprint"]])
                 if level == 0 and stair.get("footprint") else
                 np.zeros_like(floor_void))

    # Validity comes from existing 20-mm scene supervision. It is *not*
    # claimed to provide millimetric scan visibility or target evidence.
    low_valid = np.load(folder / "supervision_valid.npy", allow_pickle=False)
    valid_hd = align_coarse_mask(low_valid, meta["frame"], frame) * 255

    aug = meta.get("aug", {})
    layers, stats, counts = project_scan_bands(
        source, frame, float(meta["floor_z"]), float(meta["ceiling_z"]),
        rotation=gt.rot, thin=float(aug.get("thin", 0.0)),
        sampling_seed=meta.get("raster_sample_seed"), return_counts=True)

    from .ray_evidence import render_direct_ray_bands, audit_passage_crossings
    from .observation import return_proximity_support
    ray_bands, ray_stats, kept_rays = render_direct_ray_bands(
        ray_samples, frame, float(meta["floor_z"]), float(meta["ceiling_z"]),
        rotation=gt.rot, thin=float(aug.get("thin", 0.0)),
        sampling_seed=meta.get("raster_sample_seed"))
    support = return_proximity_support(layers, frame)
    confirmed, aperture_stats = audit_passage_crossings(
        gt, kept_rays, frame, support, openings, float(meta["floor_z"]))
    observation_images, observation_stats = build_observation_masks(
        layers, frame, wall, openings, floor_void, valid_hd,
        ray_bands=ray_bands, ray_meta=ray_stats,
        confirmed_passages=confirmed, precomputed_support=support)

    observation_stats["objects"] = audit_object_support(
        gt, frame, support)
    observation_stats["objects"]["passage_ray_crossings"] = aperture_stats

    data = {
        "format": HD_MASK_FORMAT,
        "observation_contract": OBSERVATION_CONTRACT,
        "opening_semantics": OPENING_SEMANTICS_VERSION,
        "frame": frame,
        "pixel_to_world": affine(frame),
        "source": "physical layout polygons; not a resized semantic raster",
        "wall_values": {"background": 0, "wall": 255},
        "opening_values": dict(HD_OPENING_VALUES),
        "floor_void_values": {"background_or_floor": 0, "missing_upper_floor_slab": 255},
        "stair_footprint_values": {"background": 0, "lower_floor_stairs": 255},
        "level": meta["level"],
        "rot_deg": rot_deg,
        "min_size_px": list(parse_size(min_size)),
        "void_policy": "physical_missing_floor_only_v2",
    }
    images = (
        ("wall_mask_hd.png", wall), ("openings_hd.png", openings),
        ("wall_edges_hd.png", edges), ("opening_edges_hd.png", opening_edges),
        ("floor_void_hd.png", floor_void),
        ("stair_footprint_hd.png", staircase),
        ("supervision_valid_hd.png", valid_hd),
    )
    for name, image in (*images, *observation_images.items(), *layers.items()):
        if not cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
            raise OSError(f"Could not write HD mask/evidence {folder / name}")
    np.savez_compressed(folder / 'scan_counts_hd.npz', counts=counts)
    atomic_json(folder / "wall_mask_hd.json", data)
    atomic_json(folder / "scan_evidence_hd.json", {
        "format": "simscan-native-hd-scan-evidence-v3",
        "observation": observation_stats,
        "observation_files": list(OBSERVATION_FILES),
        "frame": frame, "pixel_to_world": affine(frame),
        "low_res_frame": meta["frame"],
        "supervision_valid_source": "nearest 20mm world-aligned supervision_valid.npy",
        "layers": list(SCAN_HD_FILES),
        "counts": {"file": "scan_counts_hd.npz", "key": "counts", "dtype": "uint32",
                   "z_band_m": stats['height_band_m']['low'][:1] + stats['height_band_m']['high'][1:],
                   "policy": "raw return multiplicity; not binary occupancy"},
        **stats,
    })
    files = {}
    for name in (*HD_FILES, 'input.npy'):
        path = folder / name
        h = hashlib.sha256()
        with path.open('rb') as src:
            for chunk in iter(lambda: src.read(4 * 1024**2), b''): h.update(chunk)
        files[name] = dict(bytes=path.stat().st_size, sha256=h.hexdigest())
    atomic_json(folder / 'hd_complete.json', {'version': 3, 'files': files})
    return data


def complete_hd_level(folder, pixel_mm: float, min_size: tuple[int, int]) -> bool:
    """Require *all* native labels and raw-return sidecars for safe resume."""
    try:
        folder = Path(folder)
        if any(not (folder / name).is_file() for name in HD_FILES):
            return False
        marker = json.loads((folder / 'hd_complete.json').read_text(encoding='utf-8'))
        if any((folder / name).stat().st_size != marker['files'][name]['bytes'] for name in HD_FILES):
            return False
        doc = json.loads((folder / "wall_mask_hd.json").read_text(encoding="utf-8"))
        evidence = json.loads((folder / "scan_evidence_hd.json").read_text(encoding="utf-8"))
        if (doc.get("format") != HD_MASK_FORMAT or
                doc.get("opening_semantics") != OPENING_SEMANTICS_VERSION or
                doc.get("opening_values") != dict(HD_OPENING_VALUES) or
                doc.get("observation_contract") != OBSERVATION_CONTRACT or
                evidence.get("format") != "simscan-native-hd-scan-evidence-v3" or
                evidence.get("observation", {}).get("format") != OBSERVATION_CONTRACT or
                marker.get("version") != 3):
            return False
        frame = doc["frame"]
        size = (frame["width"], frame["height"])
        if size[0] < min_size[0] or size[1] < min_size[1]:
            return False
        if not math.isclose(frame["pixel_m"], pixel_mm / 1000, abs_tol=1e-10):
            return False
        if evidence["frame"] != frame:
            return False
        from PIL import Image
        for name in HD_LABEL_FILES + OBSERVATION_FILES + SCAN_HD_FILES:
            with Image.open(folder / name) as image:
                if image.size != size: return False
        # Inspect the NPY header inside the compressed count archive without
        # allocating another full-resolution raster during resume checks.
        with zipfile.ZipFile(folder/'scan_counts_hd.npz') as archive:
            with archive.open('counts.npy') as stream:
                version=np.lib.format.read_magic(stream)
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,fortran,dtype=reader(stream)
                if shape!=(size[1],size[0]) or dtype!=np.dtype('uint32') or fortran:return False
        return True
    except (OSError, ValueError, KeyError, TypeError, EOFError, zipfile.BadZipFile):
        return False
