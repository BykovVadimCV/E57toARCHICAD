"""CAL-09.1 ground-truth geometry metrics, independent of scanner/MAD patches.

Mask is the piecewise-linear CAL-09 envelope sampled on *undeformed* wall mesh
vertices. A triangle is active when the mean of its three vertex values >=0.5.
Adjacency requires a shared entire geometric edge **and same wall instance**;
box-mesh seam duplicates are welded by metric coordinates (10 micrometres).
Only nearly vertical wall triangles count. Components never cross wall instances.
All areas are computed on the original mesh, before displacement or plan warp.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

MASK_THRESHOLD = 0.5
BINS_M = ((0.10, 0.35), (0.35, 0.70), (0.70, 1.20), (1.20, 2.00))


def _triangles(vertices, triangles, tri_instance, vert_deform=None, tri_label=None,
               wall_label=None):
    v = np.asarray(vertices, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    inst = np.asarray(tri_instance, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or t.ndim != 2 or t.shape[1] != 3:
        raise ValueError('Expected vertex (N,3) and triangle (M,3) arrays')
    if len(inst) != len(t) or (len(t) and (t.min() < 0 or t.max() >= len(v))):
        raise ValueError('Invalid mesh topology or triangle instances')
    xyz = v[t]
    area_vector = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    norms = np.linalg.norm(area_vector, axis=1)
    normals = np.divide(area_vector, norms[:, None],
                        out=np.zeros_like(area_vector), where=norms[:, None] > 0)
    keep = (norms > 1e-12) & (np.abs(normals[:, 2]) <= .2)
    if tri_label is not None:
        if wall_label is None:
            raise ValueError('wall_label must be supplied with tri_label')
        keep &= np.asarray(tri_label) == wall_label
    if vert_deform is not None:
        keep &= np.asarray(vert_deform, bool)[t].all(axis=1)
    return v, t[keep], inst[keep], normals[keep], norms[keep] / 2


def _edge_pairs(vertices, triangles, instances, tolerance=1e-5):
    """Triangle pairs sharing a whole welded edge, never merely a corner."""
    if len(triangles) == 0:
        return np.empty((0, 2), dtype=np.int64)
    corners = vertices[triangles].reshape(-1, 3)
    quantized = np.rint(corners / tolerance).astype(np.int64)
    keys = np.column_stack((np.repeat(instances, 3), quantized))
    _, ids = np.unique(keys, axis=0, return_inverse=True)
    tri_ids = ids.reshape(-1, 3)
    edges = np.stack((tri_ids[:, [0, 1]], tri_ids[:, [1, 2]],
                      tri_ids[:, [2, 0]]), axis=1).reshape(-1, 2)
    edges.sort(axis=1)
    owner = np.repeat(np.arange(len(triangles)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    e, o = edges[order], owner[order]
    same = np.all(e[:-1] == e[1:], axis=1)
    left, right = o[:-1][same], o[1:][same]
    cross = left != right
    return np.column_stack((left[cross], right[cross]))


def _correlations(centers, wall_ids, normals, field, selected, *, max_points=1600):
    """Distance-binned centered pair covariance; diagnostics, not fitted corr length."""
    choose = np.flatnonzero(selected & np.isfinite(field))
    if len(choose) < 12:
        return {f'{a:.2f}-{b:.2f}m': None for a, b in BINS_M}
    # Deterministic spatial sample, independent of the target field values.
    if len(choose) > max_points:
        rng = np.random.default_rng(19091)
        choose = np.sort(rng.choice(choose, max_points, replace=False))
    x = field[choose].astype(float)
    x -= x.mean()
    variance = float(np.mean(x * x))
    if variance < 1e-12:
        return {f'{a:.2f}-{b:.2f}m': None for a, b in BINS_M}
    ii, jj = np.asarray(list(cKDTree(centers[choose]).query_pairs(BINS_M[-1][1])), dtype=np.int64).reshape(-1, 2).T
    if len(ii) == 0:
        return {f'{a:.2f}-{b:.2f}m': None for a, b in BINS_M}
    valid = ((wall_ids[choose[ii]] == wall_ids[choose[jj]]) &
             (np.einsum('ij,ij->i', normals[choose[ii]], normals[choose[jj]]) > .98))
    distance = np.linalg.norm(centers[choose[ii]] - centers[choose[jj]], axis=1)
    out = {}
    for lo, hi in BINS_M:
        m = valid & (distance >= lo) & (distance < hi)
        out[f'{lo:.2f}-{hi:.2f}m'] = (round(float(np.mean(x[ii[m]] * x[jj[m]]) / variance), 4)
                                        if int(m.sum()) >= 15 else None)
    return out


def mesh_truth_metrics(vertices, triangles, tri_instance, envelope, residual_xy,
                       fine_texture=None, *, tri_label=None, vert_deform=None,
                       wall_label=None, threshold=MASK_THRESHOLD):
    """Assess exact mesh topology and CAL-09 injected displacement on wall faces.

    residual_xy is CAL-09 displacement minus paired probability-zero displacement
    evaluated at the *same* undeformed vertices (metres). This subtraction
    removes global base roughness and lean without using the scanner.
    fine_texture, when provided, is the unmasked CAL-09 RFF XY field (optional).
    """
    v, t, inst, normals, areas = _triangles(
        vertices, triangles, tri_instance, vert_deform, tri_label, wall_label)
    env = np.asarray(envelope, dtype=float)
    delta = np.asarray(residual_xy, dtype=float)
    if len(env) != len(v) or delta.shape != (len(v), 2):
        raise ValueError('Envelope/residual arrays do not match mesh vertices')
    if not (np.isfinite(env).all() and np.isfinite(delta).all() and
            np.all((env >= -1e-10) & (env <= 1 + 1e-10))):
        raise ValueError('Nonfinite or out-of-bounds ground truth fields')
    if not 0 <= threshold <= 1:
        raise ValueError('Invalid mask threshold')
    if len(t) == 0:
        raise ValueError('No vertical wall triangles available for analysis')
    e_tri = env[t].mean(axis=1)
    active = e_tri >= threshold
    neighbors = _edge_pairs(v, t, inst)
    active_idx = np.flatnonzero(active)
    active_area = float(areas[active].sum())
    total_area = float(areas.sum())
    comps = 0
    component_areas = np.zeros(0, dtype=float)
    if len(active_idx):
        local = np.full(len(t), -1, dtype=np.int64)
        local[active_idx] = np.arange(len(active_idx))
        pair = neighbors[active[neighbors[:, 0]] & active[neighbors[:, 1]]]
        a, b = local[pair[:, 0]], local[pair[:, 1]]
        graph = coo_matrix((np.ones(2 * len(pair), dtype=np.uint8),
                            (np.concatenate((a, b)), np.concatenate((b, a)))),
                           shape=(len(active_idx), len(active_idx))).tocsr()
        comps, labels = connected_components(graph, directed=False)
        component_areas = np.bincount(labels, weights=areas[active], minlength=comps)
    largest = float(component_areas.max()) if comps else 0.
    # Baseline equality strictly outside the support of the smooth envelope.
    used_vertex = np.unique(t.reshape(-1))
    outside = used_vertex[env[used_vertex] == 0]
    outside_rms = (float(np.sqrt(np.mean(np.sum(delta[outside] ** 2, axis=1)))) * 1000
                   if len(outside) else None)
    # The signed wall-normal residual is the actual piecewise-linear mesh field.
    nr = np.einsum('ij,ij->i', delta[t].mean(axis=1), normals[:, :2])
    inside_rms = (float(np.sqrt(np.average(nr[active] ** 2, weights=areas[active]))) * 1000
                  if np.any(active) else None)
    centers = v[t].mean(axis=1)
    # Analytic / actual texture samples only within nearly fully enabled zones.
    core = e_tri >= .95
    fine_corr = None
    if fine_texture is not None:
        fine = np.asarray(fine_texture, dtype=float)
        if fine.shape != (len(v), 2):
            raise ValueError('Fine texture must have shape (N,2)')
        fine_n = np.einsum('ij,ij->i', fine[t].mean(axis=1), normals[:, :2])
        fine_corr = _correlations(centers, inst, normals, fine_n, core)
    return {
        'vertical_wall_triangles': int(len(t)),
        'vertical_wall_area_m2': round(total_area, 5),
        'defect_area_m2': round(active_area, 5),
        'defect_area_fraction': round(active_area / total_area, 6),
        'connected_components': int(comps),
        'largest_component_area_m2': round(largest, 5),
        'largest_component_fraction_of_defects': round(largest / active_area, 5) if active_area else None,
        'fragmentation_index': round(1 - largest / active_area, 5) if active_area else None,
        'largest_equivalent_diameter_m': round(2 * math.sqrt(largest / math.pi), 4),
        'outside_support_vertex_count': int(len(outside)),
        'outside_support_rms_mm': round(outside_rms, 10) if outside_rms is not None else None,
        'inside_wall_normal_rms_mm': round(inside_rms, 5) if inside_rms is not None else None,
        'envelope_lag_correlation': _correlations(centers, inst, normals, e_tri,
                                                  np.ones(len(t), bool)),
        'fine_core_lag_correlation': fine_corr,
        'mask_threshold': float(threshold),
        'connectivity_definition': 'shared_welded_edge_within_wall_instance',
        'area_definition': 'pre_deformation_vertical_triangle_area',
    }
