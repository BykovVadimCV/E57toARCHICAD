"""CAL-09.6 blind wall geometry from XYZ only.

This module is intentionally independent of the synthetic truth mesh, triangle
IDs, wall IDs, semantic labels, scanner station poses and the production wall
pipeline. The sole detector input is an (N,3) array of measured XYZ points.

It extracts vertical planar support, splits disconnected coplanar components,
then infers collinear, corner and T-junction neighbours from the recovered
segments. It is a geometry proposal, not a production wall classifier.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, label as connected_components
from scipy.spatial import cKDTree

FORMAT = "cal096-blind-xyz-wall-topology-v1"

DEFAULTS = {
    "voxel_m": 0.025,
    "max_points": 200_000,
    "ransac_iterations": 256,
    "max_failed_plane_rounds": 3,
    "normal_neighbors": 20,
    "normal_planarity_ratio": 0.35,
    "normal_min_tangent_variance_m2": 0.0004,
    "normal_vertical_max_deg": 22.0,
    "normal_alignment_max_deg": 30.0,
    "normal_score_weight": 0.35,
    "ransac_sample_points": 16_000,
    "max_planes": 48,
    "plane_tolerance_m": 0.040,
    "min_pair_baseline_m": 0.60,
    "min_plane_support": 60,
    "grid_m": 0.08,
    "min_component_points": 20,
    "min_wall_length_m": 0.80,
    "min_wall_height_m": 1.80,
    "min_wall_density_per_m2": 8.0,
    "min_occupancy": 0.05,
    "collinear_angle_deg": 5.0,
    "perpendicular_angle_deg": 10.0,
    "collinear_offset_m": 0.08,
    "collinear_gap_m": 0.25,
    "junction_tolerance_m": 0.20,
    "min_vertical_overlap_m": 1.60,
    "parallel_face_min_offset_m": 0.06,
    "parallel_face_max_offset_m": 0.35,
    "parallel_face_min_overlap_m": 0.50,
}


def _unit(v):
    n = float(np.linalg.norm(v))
    if n <= 1e-12:
        raise ValueError("Degenerate direction")
    return np.asarray(v, float) / n


def _canonical_normal(n):
    n = _unit(n)
    # Choose the sign from the dominant axis. Using nx alone is unstable for
    # nearly axis-aligned Y-normal planes, where tiny noise flips the tangent.
    dominant = int(np.argmax(np.abs(n)))
    if n[dominant] < 0:
        n = -n
    return n


def voxel_sample(xyz, *, voxel_m=0.025, max_points=200_000):
    """Deterministic voxel representatives and their original input indices."""
    points = np.asarray(xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("XYZ must have shape (N,3)")
    if voxel_m <= 0 or max_points < 1:
        raise ValueError("Invalid voxel sampling settings")
    finite = np.isfinite(points).all(axis=1)
    original = np.flatnonzero(finite)
    if not len(original):
        return np.empty((0, 3), float), np.empty(0, np.int64)
    keys = np.floor(points[original] / voxel_m).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    first.sort()
    if len(first) > max_points:
        k = keys[first].astype(np.uint64, copy=False)
        # Stable integer hash gives spatially unbiased deterministic thinning.
        h = (k[:, 0] * np.uint64(73856093) ^
             k[:, 1] * np.uint64(19349663) ^
             k[:, 2] * np.uint64(83492791))
        keep = np.argpartition(h, max_points - 1)[:max_points]
        first = np.sort(first[keep])
    source = original[first]
    return points[source].copy(), source.astype(np.int64, copy=False)


def _estimate_vertical_normals(points, cfg):
    """Local PCA normals from XYZ only; no labels, instances or plan priors."""
    xyz = np.asarray(points, dtype=np.float64)
    if len(xyz) < 6:
        return np.zeros_like(xyz), np.zeros(len(xyz), dtype=bool), np.zeros(len(xyz), bool)
    k = min(int(cfg["normal_neighbors"]), len(xyz))
    _, indices = cKDTree(xyz).query(xyz, k=k, workers=1)
    neighbours = xyz[indices]
    centered = neighbours - neighbours.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / float(k)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    tangent_variance = eigenvalues[:, 1]
    ratio = eigenvalues[:, 0] / np.maximum(tangent_variance, 1e-12)
    vertical = np.abs(normals[:, 2]) <= math.sin(
        math.radians(float(cfg["normal_vertical_max_deg"])))
    planar = ((ratio <= float(cfg["normal_planarity_ratio"])) &
              (tangent_variance >= float(cfg["normal_min_tangent_variance_m2"])))
    return normals, vertical, vertical & planar


def _unit_xy_normals(normals):
    xy = np.asarray(normals, dtype=np.float64)[:, :2]
    lengths = np.linalg.norm(xy, axis=1, keepdims=True)
    return xy / np.maximum(lengths, 1e-12)


def _ransac_plane(points, active_ids, rng, cfg, normals=None,
                  normal_candidates=None):
    """Propose a vertical plane from XYZ; PCA normals softly rank hypotheses."""
    use_normal_gate = normals is not None and normal_candidates is not None
    candidate_ids = (active_ids[normal_candidates[active_ids]]
                     if use_normal_gate else active_ids)
    if len(candidate_ids) < cfg["min_plane_support"]:
        candidate_ids = active_ids
        use_normal_gate = False
    if len(candidate_ids) < cfg["min_plane_support"]:
        return None
    take = min(len(candidate_ids), cfg["ransac_sample_points"])
    local = rng.choice(candidate_ids, size=take, replace=False)
    sample = points[local]
    xy = sample[:, :2]
    sample_normals = (_unit_xy_normals(normals[local])
                      if use_normal_gate else None)
    best = None
    align_cos = math.cos(math.radians(float(cfg["normal_alignment_max_deg"])))
    weight = float(cfg["normal_score_weight"])
    for _ in range(cfg["ransac_iterations"]):
        a, b = rng.integers(0, take, size=2)
        if a == b:
            continue
        if sample_normals is not None:
            pair_agreement = abs(float(sample_normals[a] @ sample_normals[b]))
            if pair_agreement < align_cos:
                continue
        delta = xy[b] - xy[a]
        baseline = float(np.linalg.norm(delta))
        if baseline < cfg["min_pair_baseline_m"]:
            continue
        tangent = delta / baseline
        normal = np.array([-tangent[1], tangent[0]])
        if normal[0] < 0 or (abs(normal[0]) <= 1e-12 and normal[1] < 0):
            normal = -normal
        if sample_normals is not None:
            if (abs(float(sample_normals[a] @ normal)) < align_cos or
                    abs(float(sample_normals[b] @ normal)) < align_cos):
                continue
        bias = -float(normal @ xy[a])
        residual = np.abs(xy @ normal + bias)
        inside = residual <= cfg["plane_tolerance_m"]
        count = int(inside.sum())
        if count < cfg["min_plane_support"]:
            continue
        projected = xy[inside] @ tangent
        z = sample[inside, 2]
        span_s = float(np.quantile(projected, .95) - np.quantile(projected, .05))
        span_z = float(np.quantile(z, .95) - np.quantile(z, .05))
        if span_s < cfg["min_wall_length_m"] or span_z < cfg["min_wall_height_m"]:
            continue
        normal_factor = 1.0
        if sample_normals is not None:
            aligned_fraction = float(np.mean(
                np.abs(sample_normals[inside] @ normal) >= align_cos))
            normal_factor = 1.0 - weight + weight * aligned_fraction
        score = count * normal_factor * min(1., span_s / 2.0) * min(1., span_z / 2.2)
        if best is None or score > best[0]:
            best = (score, normal, bias)
    return best


def _refine_plane(points, active_ids, initial_normal, initial_bias, tolerance,
                  normals=None, normal_candidates=None, cfg=None):
    """Robust TLS refinement; local PCA selects fit points but does not gate final returns."""
    if normal_candidates is not None:
        fit_ids = active_ids[normal_candidates[active_ids]]
    else:
        fit_ids = active_ids
    if len(fit_ids) < max(3, int(cfg["min_plane_support"]) if cfg else 3):
        fit_ids = active_ids
    if len(fit_ids) < 3:
        return None
    xy = points[fit_ids, :2]
    residual = np.abs(xy @ initial_normal + initial_bias)
    selected = residual <= tolerance
    if int(selected.sum()) < 3:
        return None
    for _ in range(5):
        fit = xy[selected]
        center = fit.mean(axis=0)
        covariance = (fit - center).T @ (fit - center) / len(fit)
        _, vectors = np.linalg.eigh(covariance)
        normal = _canonical_normal(vectors[:, 0])
        bias = -float(normal @ center)
        residual = np.abs(xy @ normal + bias)
        current = residual[selected]
        med = float(np.median(current))
        scale = 1.4826 * float(np.median(np.abs(current - med)))
        cutoff = min(tolerance, max(0.012, med + 3.5 * scale))
        updated = residual <= cutoff
        if int(updated.sum()) < 3:
            break
        if np.array_equal(updated, selected):
            selected = updated
            break
        selected = updated
    normal = _canonical_normal(normal)
    bias = -float(normal @ center)
    return normal, bias, fit_ids[selected], np.abs(xy[selected] @ normal + bias)


def _component_segments(points, source_indices, plane_id, normal, bias,
                        inlier_ids, residuals, cfg):
    tangent = np.array([-normal[1], normal[0]])
    xy = points[inlier_ids, :2]
    along = xy @ tangent
    z = points[inlier_ids, 2]
    grid = cfg["grid_m"]
    s0 = math.floor(float(along.min()) / grid) * grid
    z0 = math.floor(float(z.min()) / grid) * grid
    si = np.floor((along - s0) / grid).astype(np.int64)
    zi = np.floor((z - z0) / grid).astype(np.int64)
    shape = (int(si.max()) + 1, int(zi.max()) + 1)
    occupancy = np.zeros(shape, dtype=bool)
    occupancy[si, zi] = True
    closed = binary_closing(occupancy, structure=np.ones((3, 3), bool), iterations=1)
    components, n_components = connected_components(
        closed, structure=np.ones((3, 3), np.uint8))
    point_component = components[si, zi]
    segments = []
    for component in range(1, int(n_components) + 1):
        local = np.flatnonzero(point_component == component)
        if len(local) < cfg["min_component_points"]:
            continue
        ids = inlier_ids[local]
        s = along[local]
        zz = z[local]
        s_lo, s_hi = (float(np.quantile(s, .02)), float(np.quantile(s, .98)))
        z_lo, z_hi = (float(np.quantile(zz, .02)), float(np.quantile(zz, .98)))
        length = max(0., s_hi - s_lo)
        height = max(0., z_hi - z_lo)
        cell_mask = components == component
        occupied = int(cell_mask.sum())
        coords = np.argwhere(cell_mask)
        if len(coords):
            box_area = int((coords[:, 0].max() - coords[:, 0].min() + 1) *
                           (coords[:, 1].max() - coords[:, 1].min() + 1))
        else:
            box_area = 0
        coverage = occupied / max(box_area, 1)
        density = len(ids) / max(length * height, 1e-9)
        r = np.abs(points[ids, :2] @ normal + bias)
        rms = float(np.sqrt(np.mean(r * r)))
        p90 = float(np.quantile(r, .90))
        wall_candidate = bool(
            length >= cfg["min_wall_length_m"] and
            height >= cfg["min_wall_height_m"] and
            len(ids) >= cfg["min_plane_support"] and
            density >= cfg["min_wall_density_per_m2"] and
            coverage >= cfg["min_occupancy"] and
            p90 <= cfg["plane_tolerance_m"])
        p0 = -bias * normal + s_lo * tangent
        p1 = -bias * normal + s_hi * tangent
        support_score = min(1., len(ids) / 500.) * min(1., coverage / .35)
        residual_score = math.exp(-rms / .02)
        extent_score = min(1., height / 2.2) * min(1., length / 2.0)
        confidence = float(np.clip((support_score * residual_score * extent_score) ** (1 / 3), 0, 1))
        segments.append({
            "id": len(segments),
            "plane_id": int(plane_id),
            "normal_xy": normal.tolist(),
            "plane_bias": float(bias),
            "tangent_xy": tangent.tolist(),
            "s_min_m": s_lo,
            "s_max_m": s_hi,
            "z_min_m": z_lo,
            "z_max_m": z_hi,
            "endpoints_xy": [p0.tolist(), p1.tolist()],
            "length_m": length,
            "height_m": height,
            "point_count": int(len(ids)),
            "density_per_m2": float(density),
            "occupancy_fraction": float(coverage),
            "plane_rms_m": rms,
            "plane_p90_m": p90,
            "wall_candidate": wall_candidate,
            "confidence": confidence,
            # Indices are in the voxelized array, not in the raw E57 input.
            "point_indices": ids.astype(np.int64).tolist(),
            "source_point_indices": source_indices[ids].astype(np.int64).tolist(),
        })
    return segments


def _interval_gap(a0, a1, b0, b1):
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def _vertical_overlap(a, b):
    return min(a["z_max_m"], b["z_max_m"]) - max(a["z_min_m"], b["z_min_m"])


def _line_intersection(a, b):
    na = np.asarray(a["normal_xy"], float)
    nb = np.asarray(b["normal_xy"], float)
    matrix = np.vstack((na, nb))
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        return None
    return np.linalg.solve(matrix, -np.array([a["plane_bias"], b["plane_bias"]]))


def infer_topology(segments, cfg=None):
    """Infer adjacency from recovered segment geometry; no labels or wall IDs."""
    c = dict(DEFAULTS)
    if cfg:
        c.update(cfg)
    nodes = [s for s in segments if s["wall_candidate"]]
    edges, parallel = [], []
    cos_col = math.cos(math.radians(c["collinear_angle_deg"]))
    sin_perp = math.sin(math.radians(c["perpendicular_angle_deg"]))
    tol = c["junction_tolerance_m"]
    for i, a in enumerate(nodes):
        na = np.asarray(a["normal_xy"], float)
        ta = np.asarray(a["tangent_xy"], float)
        for b in nodes[i + 1:]:
            nb = np.asarray(b["normal_xy"], float)
            tb = np.asarray(b["tangent_xy"], float)
            dot = abs(float(ta @ tb))
            overlap_z = _vertical_overlap(a, b)
            if overlap_z < c["min_vertical_overlap_m"]:
                continue
            if dot >= cos_col:
                offset = abs(float(na @ (np.asarray(b["endpoints_xy"][0]) -
                                         np.asarray(a["endpoints_xy"][0]))))
                gap = _interval_gap(a["s_min_m"], a["s_max_m"],
                                   b["s_min_m"], b["s_max_m"])
                if (offset <= c["collinear_offset_m"] and
                        1e-4 < gap <= c["collinear_gap_m"]):
                    edges.append({
                        "a": int(a["id"]), "b": int(b["id"]),
                        "type": "collinear_gap", "gap_m": float(gap),
                        "offset_m": float(offset), "vertical_overlap_m": float(overlap_z),
                        "confidence": float(math.exp(-gap / max(c["collinear_gap_m"], 1e-6)) *
                                            math.exp(-offset / max(c["collinear_offset_m"], 1e-6))),
                    })
                # Parallel planes are recorded as a hypothesis, never joined
                # into the same wall group: two nearby walls are ambiguous.
                plane_offset = abs(float(na @ np.asarray(b["endpoints_xy"][0]) +
                                         a["plane_bias"]))
                along_overlap = max(0., min(a["s_max_m"], b["s_max_m"]) -
                                    max(a["s_min_m"], b["s_min_m"]))
                if (c["parallel_face_min_offset_m"] <= plane_offset <=
                        c["parallel_face_max_offset_m"] and
                        along_overlap >= c["parallel_face_min_overlap_m"]):
                    parallel.append({
                        "a": int(a["id"]), "b": int(b["id"]),
                        "offset_m": float(plane_offset),
                        "overlap_m": float(along_overlap),
                        "vertical_overlap_m": float(overlap_z),
                        "relation": "parallel_face_candidate",
                    })
                continue
            if dot > sin_perp:
                continue
            intersection = _line_intersection(a, b)
            if intersection is None:
                continue
            sa = float(intersection @ ta)
            sb = float(intersection @ tb)
            da = min(abs(sa - a["s_min_m"]), abs(sa - a["s_max_m"]))
            db = min(abs(sb - b["s_min_m"]), abs(sb - b["s_max_m"]))
            inside_a = a["s_min_m"] - tol <= sa <= a["s_max_m"] + tol
            inside_b = b["s_min_m"] - tol <= sb <= b["s_max_m"] + tol
            if not (inside_a and inside_b):
                continue
            end_a, end_b = da <= tol, db <= tol
            if end_a and end_b:
                relation = "corner"
            elif end_a != end_b:
                relation = "tee"
            else:
                continue
            edges.append({
                "a": int(a["id"]), "b": int(b["id"]),
                "type": relation,
                "junction_xy": intersection.tolist(),
                "endpoint_distance_a_m": float(da),
                "endpoint_distance_b_m": float(db),
                "vertical_overlap_m": float(overlap_z),
                "confidence": float(math.exp(-(da + db) / max(2 * tol, 1e-6))),
            })
    adjacency = {int(s["id"]): set() for s in nodes}
    for edge in edges:
        adjacency[edge["a"]].add(edge["b"])
        adjacency[edge["b"]].add(edge["a"])
    groups, seen = [], set()
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack, group = [node], []
        seen.add(node)
        while stack:
            current = stack.pop()
            group.append(current)
            for near in sorted(adjacency[current]):
                if near not in seen:
                    seen.add(near)
                    stack.append(near)
        groups.append(sorted(group))
    return {
        "edges": edges,
        "parallel_face_candidates": parallel,
        "groups": groups,
        "candidate_node_count": len(nodes),
    }


def extract_wall_geometry(xyz, *, config=None, seed=9606):
    """Blind XYZ-only segmentation and topology inference.

    Output point indices refer to the deterministic voxel sample. The source
    indices map those representatives back to the caller's raw point array.
    """
    cfg = dict(DEFAULTS)
    if config:
        unknown = set(config) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"Unknown CAL-09.6 config keys: {sorted(unknown)}")
        cfg.update(config)
    points, source_indices = voxel_sample(
        xyz, voxel_m=cfg["voxel_m"], max_points=cfg["max_points"])
    if len(points) < cfg["min_plane_support"]:
        return {
            "format": FORMAT, "input_point_count": int(len(xyz)),
            "sampled_point_count": int(len(points)), "sampled_source_indices": source_indices.tolist(),
            "normal_candidate_count": 0,
            "segments": [], "topology": infer_topology([], cfg),
            "rejected_planes": [], "decision": "INSUFFICIENT_XYZ_SUPPORT",
            "detector_inputs": ["cartesianX", "cartesianY", "cartesianZ"],
            "uses_ground_truth": False,
        }
    normals, normal_vertical, normal_candidates = _estimate_vertical_normals(points, cfg)
    candidate_count = int(normal_candidates.sum())
    rng = np.random.default_rng(seed)
    active = np.ones(len(points), dtype=bool)
    segments, rejected = [], []
    failed_rounds = 0
    plane_id = 0
    while plane_id < cfg["max_planes"]:
        active_ids = np.flatnonzero(active)
        if len(active_ids) < cfg["min_plane_support"]:
            break
        proposal = _ransac_plane(points, active_ids, rng, cfg,
                                 normals=normals, normal_candidates=normal_candidates)
        if proposal is None:
            failed_rounds += 1
            if failed_rounds >= cfg["max_failed_plane_rounds"]:
                break
            continue
        _, initial_normal, initial_bias = proposal
        refined = _refine_plane(points, active_ids, initial_normal,
                                initial_bias, cfg["plane_tolerance_m"],
                                normals=normals, normal_candidates=normal_candidates,
                                cfg=cfg)
        if refined is None:
            failed_rounds += 1
            if failed_rounds >= cfg["max_failed_plane_rounds"]:
                break
            continue
        normal, bias, fit_inliers, fit_residuals = refined
        final_residual = np.abs(points[active_ids, :2] @ normal + bias)
        final_keep = final_residual <= cfg["plane_tolerance_m"]
        inliers = active_ids[final_keep]
        residuals = final_residual[final_keep]
        if len(inliers) < cfg["min_plane_support"]:
            failed_rounds += 1
            if failed_rounds >= cfg["max_failed_plane_rounds"]:
                break
            continue
        failed_rounds = 0
        tangent = np.array([-normal[1], normal[0]])
        along = points[inliers, :2] @ tangent
        z = points[inliers, 2]
        length = float(np.quantile(along, .95) - np.quantile(along, .05))
        height = float(np.quantile(z, .95) - np.quantile(z, .05))
        if length < cfg["min_wall_length_m"] or height < cfg["min_wall_height_m"]:
            rejected.append({
                "plane_id": int(plane_id), "point_count": int(len(inliers)),
                "length_m": length, "height_m": height,
                "reason": "insufficient_vertical_wall_extent",
            })
            active[inliers] = False
            plane_id += 1
            continue
        new_segments = _component_segments(
            points, source_indices, plane_id, normal, bias, inliers, residuals, cfg)
        if not new_segments:
            rejected.append({
                "plane_id": int(plane_id), "point_count": int(len(inliers)),
                "length_m": length, "height_m": height,
                "reason": "no_supported_connected_component",
            })
        else:
            for item in new_segments:
                item["id"] = len(segments)
                segments.append(item)
        active[inliers] = False
        plane_id += 1
    topology = infer_topology(segments, cfg)
    return {
        "format": FORMAT,
        "input_point_count": int(len(xyz)),
        "sampled_point_count": int(len(points)),
        "normal_candidate_count": candidate_count,
        "sampled_source_indices": source_indices.astype(np.int64).tolist(),
        "segments": segments,
        "topology": topology,
        "rejected_planes": rejected,
        "detector_inputs": ["cartesianX", "cartesianY", "cartesianZ"],
        "uses_ground_truth": False,
        "config": cfg,
        "decision": "BLIND_XYZ_GEOMETRY_PROPOSALS_READY" if segments
                    else "NO_WALL_GEOMETRY_PROPOSALS",
    }


def _quat_matrix(q):
    if q is None:
        return np.eye(3)
    w, x, y, z = np.asarray(q, float)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if norm < 1e-12:
        raise ValueError("Invalid E57 scan quaternion")
    w, x, y, z = w/norm, x/norm, y/norm, z/norm
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def read_e57_xyz(path, *, max_points=2_000_000, chunk_size=500_000, seed=9606):
    """Read valid Cartesian returns from every E57 scan into world coordinates.

    The loader exposes only XYZ. It deliberately discards intensity, RGB,
    station identifiers, semantic fields and all labels after pose application.
    A bounded deterministic reservoir prevents large E57 files exhausting RAM.
    """
    from simscan.e57read import E57Reader

    if max_points < 1 or chunk_size < 1:
        raise ValueError("Invalid E57 point limits")
    rng = np.random.default_rng(seed)
    reservoir = np.empty((0, 3), dtype=np.float64)
    priorities = np.empty(0, dtype=np.float64)
    total_valid = 0
    with E57Reader(path) as reader:
        for scan in reader.scans:
            fields = set(scan.field_names)
            required = {"cartesianX", "cartesianY", "cartesianZ"}
            if not required <= fields:
                raise ValueError(f"E57 scan {scan.index} lacks Cartesian XYZ")
            names = list(required)
            invalid_field = "cartesianInvalidState" if "cartesianInvalidState" in fields else None
            if invalid_field:
                names.append(invalid_field)
            rotation = _quat_matrix(scan.rotation)
            translation = np.zeros(3) if scan.translation is None else np.asarray(scan.translation, float)
            for block in reader.iter_points(scan.index, names, chunk=chunk_size):
                xyz = np.column_stack((block["cartesianX"], block["cartesianY"],
                                       block["cartesianZ"])).astype(np.float64)
                valid = np.isfinite(xyz).all(axis=1)
                if invalid_field:
                    valid &= np.asarray(block[invalid_field]) == 0
                xyz = xyz[valid]
                if not len(xyz):
                    continue
                world = xyz @ rotation.T + translation
                total_valid += len(world)
                scores = rng.random(len(world))
                if len(reservoir):
                    world = np.vstack((reservoir, world))
                    scores = np.concatenate((priorities, scores))
                if len(world) > max_points:
                    take = np.argpartition(scores, -max_points)[-max_points:]
                    reservoir = world[take]
                    priorities = scores[take]
                else:
                    reservoir = world
                    priorities = scores
    return reservoir, {
        "scans": len(reader.scans),
        "valid_points_seen": int(total_valid),
        "sampled_points": int(len(reservoir)),
        "sampling": "uniform_reservoir_without_replacement",
        "uses_station_ids_in_detector": False,
    }


def run_e57(input_path, output_path, *, max_points=2_000_000):
    xyz, info = read_e57_xyz(input_path, max_points=max_points)
    result = extract_wall_geometry(xyz)
    # CLI output is compact: omit point membership arrays and source-index maps.
    compact = dict(result)
    compact.pop("sampled_source_indices", None)
    compact["segments"] = [
        {k: v for k, v in s.items()
         if k not in ("point_indices", "source_point_indices")}
        for s in result["segments"]
    ]
    compact["e57_input"] = info
    compact["input_file"] = Path(input_path).name
    Path(output_path).write_text(
        json.dumps(compact, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_e57", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    args = parser.parse_args()
    report = run_e57(args.input_e57, args.out, max_points=args.max_points)
    print(json.dumps({
        "decision": report["decision"],
        "sampled_points": report["sampled_point_count"],
        "wall_candidates": sum(s["wall_candidate"] for s in report["segments"]),
        "topology_edges": len(report["topology"]["edges"]),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
