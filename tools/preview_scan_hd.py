"""Render actual simulated E57 point returns into a native metric top-down map.

This is a visualization, not ground-truth geometry. All visible pixels originate
from real simulated TLS point returns, with floor and ceiling excluded to expose
walls, furniture, occlusion, and clutter. No walls are drawn from layout masks.

Usage:
  python tools/preview_scan_hd.py --scene-dir dataset/_work/scene_00000 \\
    --level-dir dataset/scenes/scene_00000/L1 --preview-width 1800
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from simscan.e57io import read_e57_world

# These constants match the display defaults of the user's raw-pointcloud exporter.
EXTRACTOR_WIDTH = 3840
EXTRACTOR_HEIGHT = 2160
EXTRACTOR_MARGIN_PX = 55
EXTRACTOR_PADDING_M = 0.25
EXTRACTOR_MARKER_PX = 3


def extractor_grayscale(hist: np.ndarray, marker_px: int = 3) -> np.ndarray:
    """Render a point-count histogram using raw_pointcloud.py's log-density curve."""
    if hist.ndim != 2 or not np.issubdtype(hist.dtype, np.integer):
        raise ValueError("Expected a 2D integer count histogram")
    if marker_px < 1 or marker_px % 2 != 1:
        raise ValueError("marker_px must be a positive odd number")
    positive = hist[hist > 0]
    if not positive.size:
        raise ValueError("No projected scan points")
    peak = max(2.0, float(np.percentile(positive, 99)))
    gray = np.minimum(245, np.log1p(hist) / np.log1p(peak) * 185
                      + np.where(hist > 0, 50, 0)).astype(np.uint8)
    if marker_px > 1:
        gray = cv2.dilate(gray, np.ones((marker_px, marker_px), dtype=np.uint8))
    return gray


def render_extractor_view(point_sets: list[dict], folder: Path, floor: float,
                          ceil: float, width: int = EXTRACTOR_WIDTH,
                          height: int = EXTRACTOR_HEIGHT,
                          margin: int = EXTRACTOR_MARGIN_PX,
                          marker_px: int = EXTRACTOR_MARKER_PX,
                          padding_m: float = EXTRACTOR_PADDING_M) -> dict:
    """Display *actual scan returns* in the user's 4K log-density style.

    This is NOT a 2mm physical raster: display resolution depends on the crop.
    The parallel scan_topdown_2mm.png remains the native metric projection.
    """
    if width <= margin * 2 or height <= margin * 2 or margin < 0:
        raise ValueError("Extractor canvas must exceed twice the margin")
    if not np.isfinite(padding_m) or padding_m < 0:
        raise ValueError("Invalid crop padding")
    low, high = float(floor + .15), float(ceil - .15)
    parts = []
    for scan in point_sets:
        points = np.asarray(scan["xyz"])
        keep = np.isfinite(points).all(axis=1)
        keep &= (points[:, 2] >= low) & (points[:, 2] <= high)
        if np.any(keep):
            parts.append(points[keep, :2])
    if not parts:
        raise ValueError("No laser returns inside the selected floor slice")
    xy = np.concatenate(parts)
    if len(xy) < 30:
        raise ValueError("Too few points to estimate a scan viewport")
    # Match --bounds auto sampling: clip the isolated returns by 0.3/99.7%
    # percentiles, while retaining all points inside that final viewport.
    bounds_lo, bounds_hi = np.percentile(xy, [0.3, 99.7], axis=0)
    if np.any(bounds_hi - bounds_lo < 0.1):
        bounds_lo, bounds_hi = xy.min(axis=0), xy.max(axis=0)
    xmin, ymin = bounds_lo - padding_m
    xmax, ymax = bounds_hi + padding_m
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Degenerate scan viewport")
    scale = min((width - 2 * margin) / (xmax - xmin),
                (height - 2 * margin) / (ymax - ymin))
    offset_x = (width - (xmax - xmin) * scale) / 2
    offset_y = (height - (ymax - ymin) * scale) / 2
    hist = np.zeros((height, width), dtype=np.uint32)
    flat_hist = hist.ravel()
    rendered = 0
    for points in parts:
        ix = np.floor((points[:, 0] - xmin) * scale + offset_x).astype(np.int64)
        iy = np.floor((ymax - points[:, 1]) * scale + offset_y).astype(np.int64)
        inside = (ix >= 0) & (ix < width) & (iy >= 0) & (iy < height)
        if not np.any(inside):
            continue
        loc, counts = np.unique(iy[inside] * width + ix[inside], return_counts=True)
        flat_hist[loc] += counts.astype(np.uint32)
        rendered += int(inside.sum())
    image = extractor_grayscale(hist, marker_px=marker_px)
    output = folder / "scan_extractor_4k.png"
    if not cv2.imwrite(str(output), image, [cv2.IMWRITE_PNG_COMPRESSION, 5]):
        raise OSError(f"Could not write {output}")
    meta = {
        "source": "actual synthetic E57 laser points, no geometric overlays",
        "image": output.name,
        "pixel_dimensions": [width, height],
        "xy_bounds_m": [float(xmin), float(ymin), float(xmax), float(ymax)],
        "effective_visual_pixel_m": float(1 / scale),
        "z_range_m": [low, high],
        "points_rendered": rendered,
        "marker_px": marker_px,
        "render": "raw_xy_returns_log_density_grayscale",
        "crop_method": "scan_xy_percentiles_0.3_99.7",
        "overlays": "none",
    }
    (folder / "scan_extractor_4k.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta



def render_scan(e57_path: str | Path, level_dir: str | Path, preview_width: int = 1800):
    folder = Path(level_dir)
    info = json.loads((folder / "wall_mask_hd.json").read_text(encoding="utf-8"))
    scene = json.loads((folder / "scene.json").read_text(encoding="utf-8"))
    frame = info["frame"]
    p = float(frame["pixel_m"])
    h, w = int(frame["height"]), int(frame["width"])
    floor, ceil = float(scene["floor_z"]), float(scene["ceiling_z"])
    if not (0 < p < 1 and h > 0 and w > 0):
        raise ValueError("Invalid metric frame in HD mask metadata")
    if not (math.isfinite(floor) and math.isfinite(ceil) and ceil > floor + .5):
        raise ValueError("Invalid floor and ceiling levels")

    # Slice away the dominant floor/ceiling returns but retain walls, furniture
    # and other tall clutter. The physical wall mask is deliberately not used.
    low_z, high_z = floor + .18, ceil - .18
    out = np.zeros((h, w), dtype=np.uint8)
    seen = projected = 0
    point_sets = read_e57_world(e57_path)
    for station in point_sets:
        xyz = station["xyz"]
        seen += len(xyz)
        good = np.isfinite(xyz).all(axis=1)
        good &= (xyz[:, 2] >= low_z) & (xyz[:, 2] <= high_z)
        xy = xyz[good, :2]
        ix = np.floor((xy[:, 0] - float(frame["origin_x"])) / p).astype(np.int64)
        iy = h - 1 - np.floor((xy[:, 1] - float(frame["origin_y"])) / p).astype(np.int64)
        valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
        ix, iy = ix[valid], iy[valid]
        projected += len(ix)
        # One white pixel per occupied 2-mm cell, *no interpolation*.
        out[iy, ix] = 255

    if projected == 0:
        raise ValueError("No in-bounds scan points in selected vertical band")

    full = folder / "scan_topdown_2mm.png"
    if not cv2.imwrite(str(full), out):
        raise OSError(f"Could not save {full}")

    # Presentation-only preview: area downsample + contrast boost so sparse
    # actual samples remain visible on a normal display. Full image is exact.
    preview_width = min(int(preview_width), w)
    if preview_width < 256:
        raise ValueError("preview_width must be >=256")
    hh = max(1, int(round(preview_width * h / w)))
    small = cv2.resize(out, (preview_width, hh), interpolation=cv2.INTER_AREA)
    small = np.clip(np.sqrt(small.astype(np.float32) / 255.0) * 255, 0, 255).astype(np.uint8)
    small = cv2.dilate(small, np.ones((2, 2), dtype=np.uint8), iterations=1)
    preview = folder / "scan_topdown_preview.png"
    if not cv2.imwrite(str(preview), small):
        raise OSError(f"Could not save {preview}")

    stats = {"source": "simulated E57 laser returns", "pixel_m": p,
             "size_px": [w, h], "total_points": seen,
             "points_in_slice": projected, "occupied_pixels": int(np.count_nonzero(out)),
             "slice_world_z_m": [low_z, high_z],
             "preview_processing": "display-only downsample, gamma, 2px dilation"}
    extractor = render_extractor_view(point_sets, folder, floor, ceil)
    stats["extractor_style"] = extractor
    (folder / "scan_topdown_meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", required=True, type=Path, help="scene folder containing scan.e57")
    ap.add_argument("--level-dir", required=True, type=Path, help="dataset level with wall_mask_hd.json")
    ap.add_argument("--preview-width", type=int, default=1800)
    args = ap.parse_args()
    print(json.dumps(render_scan(args.scene_dir / "scan.e57", args.level_dir,
                                 args.preview_width), indent=2))


if __name__ == "__main__":
    main()
