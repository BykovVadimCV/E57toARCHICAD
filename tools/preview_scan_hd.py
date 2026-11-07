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
    for station in read_e57_world(e57_path):
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
