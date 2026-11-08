"""Native 2-mm laser-return evidence paired with geometric architectural masks.

All scan pixels originate in simulated scanner XYZ, never enlarged 20-mm
occupancy or ground-truth polygons. Designed for a second-stage refinement model.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

SCAN_HD_FILES = (
    "scan_low_hd.png", "scan_mid_hd.png", "scan_high_hd.png",
    "scan_floor_hd.png", "scan_ceiling_hd.png", "scan_below_hd.png",
)


def project_scan_bands(source, frame: dict, floor_z: float, ceiling_z: float,
                       rotation=None, thin: float = 0.0, sampling_seed: int | None = None,
                       log=lambda *_: None):
    """Project raw scan returns into five binary height bands at native metric pitch.

    With the same point order/seed, the optional dropout follows the low-res
    rasterizer: replay its level-detection sampling RNG pass, then its thin pass.
    Does not infer occupied or free space outside observed returns.
    """
    from .netinput import BAND_MARGIN_M, FLOOR_TOL_M, LOW_TOP, MID_TOP, BELOW_DEPTH_M, rotate_xy

    px = float(frame["pixel_m"])
    H, W = int(frame["height"]), int(frame["width"])
    if not 0 < px <= 0.1 or H <= 0 or W <= 0 or H * W > 64_000_000:
        raise ValueError("Invalid native evidence frame")
    if not (math.isfinite(floor_z) and math.isfinite(ceiling_z) and ceiling_z - floor_z > 0.5):
        raise ValueError("Invalid floor/ceiling elevations")
    if not 0.0 <= thin < 1.0:
        raise ValueError("thin must be in [0,1)")
    if thin and sampling_seed is None:
        raise ValueError("sampling_seed is required for matching thinned samples")

    layers = {name: np.zeros((H, W), np.uint8) for name in SCAN_HD_FILES}
    rng = np.random.default_rng(sampling_seed) if thin else None
    if rng is not None:
        # The 20-mm rasterizer draws one random sample per source return in
        # its initial level-detection pass before thinning the second pass.
        for points in source.chunks():
            rng.random(len(points))

    top = ceiling_z - floor_z
    ox, oy = float(frame["origin_x"]), float(frame["origin_y"])
    total = kept = projected = 0
    for points in source.chunks():
        total += len(points)
        if rng is not None:
            points = points[rng.random(len(points)) >= thin]
        kept += len(points)
        if not len(points):
            continue
        points = rotate_xy(np.asarray(points), rotation)
        points = points[np.isfinite(points).all(axis=1)]
        if not len(points):
            continue
        ix = np.floor((points[:, 0] - ox) / px).astype(np.int64)
        iy = H - 1 - np.floor((points[:, 1] - oy) / px).astype(np.int64)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        if not ok.any():
            continue
        h = points[ok, 2] - floor_z
        flat = iy[ok] * W + ix[ok]
        projected += len(flat)
        bands = (
            (h >= BAND_MARGIN_M) & (h < LOW_TOP),
            (h >= LOW_TOP) & (h < MID_TOP),
            (h >= MID_TOP) & (h <= top - BAND_MARGIN_M),
            np.abs(h) < FLOOR_TOL_M,
            np.abs(h - top) < FLOOR_TOL_M,
            (h < -BAND_MARGIN_M) & (h > -BELOW_DEPTH_M),
        )
        for name, mask in zip(SCAN_HD_FILES, bands):
            if mask.any():
                layers[name].ravel()[flat[mask]] = 255

    stats = {
        "source": "actual simulated scanner returns; no interpolation or polygon leakage",
        "total_source_points": int(total),
        "kept_after_thinning": int(kept),
        "in_frame_points_all_elevations": int(projected),
        "occupied_cells_by_band": {k: int(np.count_nonzero(v)) for k, v in layers.items()},
        "pixel_m": px, "shape_hw": [H, W], "thin": float(thin),
        "sampling_seed": sampling_seed, "floor_z": float(floor_z),
        "ceiling_z": float(ceiling_z),
        "height_band_m": {
            "low": [float(BAND_MARGIN_M), float(LOW_TOP)],
            "mid": [float(LOW_TOP), float(MID_TOP)],
            "high": [float(MID_TOP), float(top - BAND_MARGIN_M)],
            "below_floor": [-float(BELOW_DEPTH_M), -float(BAND_MARGIN_M)],
        },
    }
    log("2mm scan returns: " + str(stats["occupied_cells_by_band"]))
    return layers, stats


def align_coarse_mask(coarse: np.ndarray, coarse_frame: dict, target_frame: dict):
    """Map the existing *supervision validity* to HD by exact world coordinates.

    Coarse validity is not native laser evidence or high-resolution truth.
    OpenCV's inverse affine samples each target pixel at its world-space center.
    """
    cp, hp = float(coarse_frame["pixel_m"]), float(target_frame["pixel_m"])
    hc, _ = coarse.shape
    hh, wh = int(target_frame["height"]), int(target_frame["width"])
    if cp <= 0 or hp <= 0:
        raise ValueError("Pixel pitches must be positive")
    s = hp / cp
    tx = (float(target_frame["origin_x"]) + hp / 2 - float(coarse_frame["origin_x"])) / cp - .5
    ty = hc - .5 - (float(target_frame["origin_y"]) + (hh - .5) * hp -
                    float(coarse_frame["origin_y"])) / cp
    inverse = np.array([[s, 0, tx], [0, s, ty]], np.float64)
    return cv2.warpAffine(coarse.astype(np.uint8), inverse, (wh, hh),
                          flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
