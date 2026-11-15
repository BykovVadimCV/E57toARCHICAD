"""CAL-09.3 physical-wall multi-scale features on *observed* TLS return XYZ.

A fixed GT-oracle wall/face/cell association is reused from CAL-09.2. GT
envelope values are never passed to any feature estimator. Within the same
physical patch and exactly the same direct rays we evaluate:
- legacy 3D robust vertical-plane MAD (CAL-02/09)
- macro normal offset against robust plane fitted to the *whole visible face*
- robust dispersion of 3x4 subcell median signed residuals against a *local*
  vertical line. These are observational scores, not intrinsic scanner noise.

Only synthetic per-patch scores and GT-positive fractions are saved. There are
no raw point coordinates, station poses or raw triangle primitive identifiers.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from simscan.labels import LABEL_ID
from simscan.transform import WorldTransform
from simscan.realism import roughness_tail_envelope
from tools.calibrate_e57_surfaces import fit_vertical_plane
from tools.cal092_ray_observability import (
    ABLATIONS, MIN_PATCH_POINTS, barycentric_preimage, physical_keys,
)

FEATURES = ("legacy_mad", "macro_offset", "subcell_shape")
FORMAT = "cal093-multiscale-observed-scores-v1"
GT_FRACTION = .4
PATCH_MAD_DETECTED_MM = 6.
SUBCELL_BINS = (3, 4)  # 0.30m x 0.3125m within 0.90m x 1.25m
SUBCELL_MIN_POINTS = 10
SUBCELL_MIN_OCCUPIED = 4


def robust_vertical_line(points, *, max_iter=4, min_points=40):
    """TLS of observed XY coordinates with symmetric robust trimming.

    The fitted coefficient is not supplied by the truth mesh. Reflections on
    one known wall-face form a long narrow footprint. The returned residual is
    signed in metres, with arbitrary global sign convention.
    """
    a = np.asarray(points, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("Expected observed points of shape (N,3)")
    xy = a[np.isfinite(a).all(axis=1), :2]
    if len(xy) < min_points:
        return None
    fit = xy
    for _ in range(max_iter):
        if len(fit) < min_points:
            return None
        center = fit.mean(axis=0)
        cov = (fit - center).T @ (fit - center) / len(fit)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]
        if normal[0] < 0 or (abs(normal[0]) < 1e-10 and normal[1] < 0):
            normal = -normal
        bias = -float(normal @ center)
        residuals = xy @ normal + bias
        med = np.median(residuals)
        scale = 1.4826 * np.median(np.abs(residuals - med))
        trim = min(.060, max(.008, 3.5 * float(scale)))
        selected = xy[np.abs(residuals - med) <= trim]
        if len(selected) < min_points:
            break
        fit = selected
    center = fit.mean(axis=0)
    cov = (fit - center).T @ (fit - center) / len(fit)
    _, vectors = np.linalg.eigh(cov)
    normal = vectors[:, 0]
    if normal[0] < 0 or (abs(normal[0]) < 1e-10 and normal[1] < 0):
        normal = -normal
    return normal, -float(normal @ center)


def subcell_shape_mm(points, along, height, line, *, min_points=SUBCELL_MIN_POINTS):
    """1.4826*MAD of medians in locally fixed physical subcells, in mm.

    Binning uses the same oracle pre-deformation coordinates for all noise
    ablations. The signal comes solely from observed XY residuals to an
    observed local fitted vertical line, not from the GT envelope.
    """
    x = np.asarray(points, dtype=float)
    a = np.asarray(along, dtype=float)
    z = np.asarray(height, dtype=float)
    if line is None or len(x) != len(a) or len(a) != len(z):
        return None
    normal, bias = line
    signed = x[:, :2] @ normal + bias
    sub = np.column_stack((np.floor(a / (.9 / SUBCELL_BINS[0])).astype(np.int64),
                           np.floor(z / (1.25 / SUBCELL_BINS[1])).astype(np.int64)))
    _, inv = np.unique(sub, return_inverse=True, axis=0)
    levels = []
    for i in range(inv.max() + 1) if len(inv) else ():
        local = signed[inv == i]
        if len(local) >= min_points:
            levels.append(float(np.median(local)))
    if len(levels) < SUBCELL_MIN_OCCUPIED:
        return None
    vals = np.asarray(levels)
    return round(float(1.4826 * np.median(np.abs(vals - np.median(vals))) * 1000), 4)


def matched_patch_scores(sample, mesh, layout_doc, *, min_points=MIN_PATCH_POINTS):
    """Scores from exactly matched direct ray memberships in four ablations."""
    info = layout_doc["meta"]["layout_to_world"]
    world = WorldTransform(float(info["yaw_deg"]), tuple(info["offset_m"]))
    stations = layout_doc["stations"]
    count = len(sample["primitive"])
    transformed = {mode: np.empty((count, 3), dtype=np.float64)
                   for mode in ABLATIONS}
    for station_id in np.unique(sample["station"]):
        select = np.flatnonzero(sample["station"] == station_id)
        entry = stations[int(station_id)]
        for mode, point_source, pose in (
            ("ideal", "ideal_local", "pose_true"),
            ("range_only", "measured_local", "pose_true"),
            ("registration_only", "ideal_local", "pose_written"),
            ("combined", "measured_local", "pose_written"),
        ):
            p = entry[pose]
            R = np.asarray(p["R"], float)
            t = np.asarray(p["t"], float)
            transformed[mode][select] = world.to_layout(
                sample[point_source][select] @ R.T + t)

    prim = np.asarray(sample["primitive"], np.int64)
    pre, bary, normal, valid = barycentric_preimage(
        transformed["ideal"], prim, mesh)
    triangles = np.asarray(mesh["triangles"], np.int64)
    labels = np.asarray(mesh["tri_label"])
    valid &= labels[prim] == LABEL_ID["wall"]
    envelope = roughness_tail_envelope(
        np.asarray(mesh["vertices_pre"], float), mesh["realism"])
    truth_on_rays = np.einsum("ni,ni->n", bary, envelope[triangles[prim]])
    keys, along = physical_keys(pre, normal, np.asarray(mesh["tri_instance"])[prim])

    groups = defaultdict(list)
    faces = defaultdict(list)
    for idx in np.flatnonzero(valid):
        k = tuple(int(x) for x in keys[idx])
        groups[k].append(int(idx))
        faces[k[:3]].append(int(idx))

    # Observed face reference fits across physical cells, no truth scores.
    references = {
        mode: {face: robust_vertical_line(transformed[mode][idxs])
               for face, idxs in faces.items()}
        for mode in ABLATIONS
    }

    rows = []
    for key in sorted(groups):
        ids = np.asarray(groups[key], dtype=np.int64)
        if len(ids) < min_points:
            continue
        if (np.quantile(along[ids], .95) - np.quantile(along[ids], .05) < .4 or
            np.quantile(pre[ids, 2], .95) - np.quantile(pre[ids, 2], .05) < .65):
            continue

        fit = {mode: fit_vertical_plane(transformed[mode][ids])
               for mode in ABLATIONS}
        if any(x is None or x["inlier_fraction"] < .55 for x in fit.values()):
            continue
        scores = {}
        for mode in ABLATIONS:
            observed = transformed[mode][ids]
            local_line = robust_vertical_line(observed)
            face_line = references[mode].get(key[:3])
            macro = None
            if face_line is not None:
                n, b = face_line
                macro = round(float(abs(np.median(observed[:, :2] @ n + b))
                                    * 1000), 4)
            scores[mode] = {
                "legacy_mad": float(fit[mode]["observed_plane_mad_mm"]),
                "macro_offset": macro,
                "subcell_shape": subcell_shape_mm(
                    observed, along[ids], pre[ids, 2], local_line),
            }
        rows.append({
            "cell_key": list(key),
            "sampled_direct_rays": len(ids),
            "gt_visible_fraction": round(float(np.mean(truth_on_rays[ids] >= .5)), 6),
            "scores_mm": scores,
        })
    return {
        "format": FORMAT,
        "sampled_direct_wall_rays": count,
        "geometrically_valid_rays": int(valid.sum()),
        "n_eligible_cells": len(rows),
        "cells": rows,
        "oracle_patch_grouping": True,
        "same_rays_each_ablation": True,
    }
