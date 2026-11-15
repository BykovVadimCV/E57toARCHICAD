"""CAL-09.2: tie *direct* observed TLS returns to physical wall mesh triangles.

One and the same direct ray is reconstructed under a 2x2 counterfactual:
   measured vs pre-range-noise distance; true vs written registration pose.
Patch cells are determined from undeformed physical wall coordinates by
barycentric inverse of the deformed triangle; the measured fitted plane uses
ONLY return XYZ. GT is used for pairing and evaluation, never for fitting.

This is an ORACLE physical-patch diagnostic, not an automatic wall detector.
No raw XYZ, poses or primitive ids are exported in the JSON result.
"""
from __future__ import annotations

from collections import defaultdict
import numpy as np

from simscan.labels import LABEL_ID
from simscan.transform import WorldTransform
from tools.calibrate_e57_surfaces import fit_vertical_plane

PATCH_LENGTH_M = .9
PATCH_HEIGHT_M = 1.25
MIN_PATCH_POINTS = 110
MIN_GT_VISIBLE_FRACTION = .40
DETECTED_MAD_MM = 6.
ABLATIONS = ('ideal', 'range_only', 'registration_only', 'combined')


def confusion(pairs, threshold=DETECTED_MAD_MM, gt_fraction=MIN_GT_VISIBLE_FRACTION):
    """Patch-level confusion; undefined precision/recall represented by None."""
    tp = fp = tn = fn = 0
    for p in pairs:
        truth = float(p['gt_visible_fraction']) >= gt_fraction
        detected = float(p['mad_mm']) >= threshold
        if truth and detected:
            tp += 1
        elif truth:
            fn += 1
        elif detected:
            fp += 1
        else:
            tn += 1
    safe = lambda numerator, denom: round(numerator / denom, 4) if denom else None
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': safe(tp, tp + fp),
        'recall': safe(tp, tp + fn),
        'false_positive_rate': safe(fp, fp + tn),
        'n_patches': len(pairs),
        'n_gt_positive': tp + fn,
        'n_detected_positive': tp + fp,
    }


def barycentric_preimage(hit_xyz_post, prim, mesh):
    """Map true *noiseless* hit on post-warp triangle back to pre-warp mesh.

    Returns raw barycentrics and a conservative validity mask. Using the true
    geometrical hit and retaining primitive identity prevents noise/registration
    from changing the physical patch to which a ray belongs.
    """
    ids = np.asarray(prim, dtype=np.int64)
    points = np.asarray(hit_xyz_post, float)
    t = np.asarray(mesh['triangles'], dtype=np.int64)[ids]
    post = np.asarray(mesh['vertices_post'], dtype=float)[t]
    pre = np.asarray(mesh['vertices_pre'], dtype=float)[t]
    a = post[:, 0]
    e0, e1, q = post[:, 1] - a, post[:, 2] - a, points - a
    d00 = np.einsum('ij,ij->i', e0, e0)
    d01 = np.einsum('ij,ij->i', e0, e1)
    d11 = np.einsum('ij,ij->i', e1, e1)
    d20 = np.einsum('ij,ij->i', q, e0)
    d21 = np.einsum('ij,ij->i', q, e1)
    denom = d00 * d11 - d01 * d01
    good = np.isfinite(denom) & (np.abs(denom) > 1e-15)
    u = np.divide(d11 * d20 - d01 * d21, denom,
                  out=np.zeros(len(ids)), where=good)
    v = np.divide(d00 * d21 - d01 * d20, denom,
                  out=np.zeros(len(ids)), where=good)
    w = 1 - u - v
    good &= ((u >= -.02) & (v >= -.02) & (w >= -.02) &
             (u <= 1.02) & (v <= 1.02) & (w <= 1.02))
    uvw = np.column_stack((w, u, v))
    # Barycentric tolerance accommodates float32 raycasting and large yaw.
    uvw = np.maximum(uvw, 0.)
    uvw /= np.maximum(uvw.sum(axis=1, keepdims=True), 1e-12)
    pre_xyz = np.einsum('ni,nij->nj', uvw, pre)
    normal = np.cross(pre[:, 1] - pre[:, 0], pre[:, 2] - pre[:, 0])
    nnorm = np.linalg.norm(normal, axis=1)
    good &= nnorm > 1e-12
    normal = np.divide(normal, nnorm[:, None],
                       out=np.zeros_like(normal), where=nnorm[:, None] > 0)
    good &= np.abs(normal[:, 2]) <= .2
    return pre_xyz, uvw, normal, good


def physical_keys(pre_xyz, normal, instance):
    """One key per wall instance, oriented face, global wall-axis and height bin."""
    nxy = normal[:, :2]
    tangent = np.column_stack((-nxy[:, 1], nxy[:, 0]))
    # Canonical tangent shared by front/back faces, side remains in the key.
    flip = (tangent[:, 0] < -1e-9) | ((np.abs(tangent[:, 0]) < 1e-9) & (tangent[:, 1] < 0))
    tangent[flip] *= -1
    along = np.einsum('ij,ij->i', pre_xyz[:, :2], tangent)
    c = np.floor(along / PATCH_LENGTH_M).astype(np.int64)
    r = np.floor(pre_xyz[:, 2] / PATCH_HEIGHT_M).astype(np.int64)
    side = np.rint(nxy * 20).astype(np.int32)
    key = np.column_stack((instance, side[:, 0], side[:, 1], c, r))
    return key, along


def priority_sample(scans, *, max_sample=180000, seed=42):
    """Uniform global priority sample of valid DIRECT wall returns.

    Retains no noisy points that change identity at beam/mirror/glass edges.
    The same physical rays are reused across every ablation.
    """
    if max_sample < 1000:
        raise ValueError('max_sample must be >= 1000')
    rng = np.random.default_rng(seed)
    pieces = []
    n_direct_wall = 0
    for si, scan in enumerate(scans):
        prim = getattr(scan, 'primitive_id', None)
        truth = getattr(scan, 'true_range_m', None)
        if prim is None or truth is None:
            raise ValueError('CAL-09.2 requires capture_truth_hits=True')
        mask = (np.asarray(scan.straight_valid, bool) &
                (scan.label == LABEL_ID['wall']) &
                (prim >= 0) & np.isfinite(truth) & (truth > 0))
        ids = np.flatnonzero(mask)
        n_direct_wall += len(ids)
        if not len(ids):
            continue
        # Keeping the best K from each station is enough for global best K.
        pri = rng.random(len(ids))
        if len(ids) > max_sample:
            chosen = np.argpartition(pri, max_sample - 1)[:max_sample]
            ids, pri = ids[chosen], pri[chosen]
        measured = np.asarray(scan.xyz_local[ids], dtype=np.float64)
        tr = np.asarray(truth[ids], dtype=float)
        measured_range = np.linalg.norm(measured, axis=1)
        keep = measured_range > 0
        if not np.any(keep):
            continue
        measured, tr, pri, ids, measured_range = (a[keep] for a in
                                                  (measured, tr, pri, ids, measured_range))
        pieces.append({
            'priority': pri,
            'station': np.full(len(ids), si, np.int32),
            'primitive': np.asarray(prim[ids], np.int32),
            'measured_local': measured,
            'ideal_local': measured * (tr / measured_range)[:, None],
        })
    if not pieces:
        raise ValueError('No direct wall returns: cannot evaluate observability')
    all_p = np.concatenate([p['priority'] for p in pieces])
    selected = (np.argpartition(all_p, max_sample - 1)[:max_sample]
                if len(all_p) > max_sample else np.arange(len(all_p)))
    out = {k: np.concatenate([p[k] for p in pieces])[selected]
           for k in ('station', 'primitive', 'measured_local', 'ideal_local')}
    out['total_direct_wall_returns'] = int(n_direct_wall)
    out['sampled_direct_wall_returns'] = len(selected)
    return out


def evaluate_rays(sample, mesh, layout_doc, *, threshold=DETECTED_MAD_MM,
                  gt_fraction=MIN_GT_VISIBLE_FRACTION,
                  min_points=MIN_PATCH_POINTS):
    """Return matched-patch report. Does not persist point coordinates."""
    from simscan.realism import roughness_tail_envelope

    info = layout_doc['meta']['layout_to_world']
    world = WorldTransform(float(info['yaw_deg']), tuple(info['offset_m']))
    stations = layout_doc['stations']
    size = len(sample['primitive'])
    points = {name: np.empty((size, 3), float) for name in ABLATIONS}
    for si in np.unique(sample['station']):
        k = np.flatnonzero(sample['station'] == si)
        st = stations[int(si)]
        true = st['pose_true']
        written = st['pose_written']
        for name, local, pose in (
            ('ideal', sample['ideal_local'], true),
            ('range_only', sample['measured_local'], true),
            ('registration_only', sample['ideal_local'], written),
            ('combined', sample['measured_local'], written),
        ):
            R = np.asarray(pose['R'], float)
            t = np.asarray(pose['t'], float)
            points[name][k] = world.to_layout(local[k] @ R.T + t)
    pre_xyz, weights, normal, valid = barycentric_preimage(
        points['ideal'], sample['primitive'], mesh)
    prim = sample['primitive']
    valid &= np.asarray(mesh['tri_label'])[prim] == LABEL_ID['wall']
    tris = np.asarray(mesh['triangles'])[prim]
    v_env = roughness_tail_envelope(np.asarray(mesh['vertices_pre'], float),
                                     mesh['realism'])
    truth = np.einsum('ni,ni->n', weights, v_env[tris])
    keys, along = physical_keys(pre_xyz, normal, np.asarray(mesh['tri_instance'])[prim])

    # GT is for grouping / target only, never passed into the TLS plane fitter.
    groups = defaultdict(list)
    for i in np.flatnonzero(valid):
        groups[tuple(map(int, keys[i]))].append(int(i))
    records = []
    for key, indices in groups.items():
        ids = np.asarray(indices, np.int64)
        if len(ids) < min_points:
            continue
        if (np.quantile(along[ids], .95) - np.quantile(along[ids], .05) < .4 or
                np.quantile(pre_xyz[ids, 2], .95) - np.quantile(pre_xyz[ids, 2], .05) < .65):
            continue
        fits = {name: fit_vertical_plane(points[name][ids]) for name in ABLATIONS}
        # Preserve exact same physical cells and rays for all four conditions.
        if any(x is None or x['inlier_fraction'] < .55 for x in fits.values()):
            continue
        # Actual direct-visible fraction (sampling-weighted), not a wall area fraction.
        fraction = float(np.mean(truth[ids] >= .5))
        records.append({
            'wall_patch_key': list(key),
            'n_direct_rays': len(ids),
            'gt_visible_fraction': round(fraction, 5),
            'mean_envelope': round(float(np.mean(truth[ids])), 5),
            'mad_mm': {name: fits[name]['observed_plane_mad_mm'] for name in ABLATIONS},
        })
    results = {}
    for mode in ABLATIONS:
        matched = [{'gt_visible_fraction': row['gt_visible_fraction'],
                    'mad_mm': row['mad_mm'][mode]} for row in records]
        conf = confusion(matched, threshold=threshold, gt_fraction=gt_fraction)
        vals = [x['mad_mm'][mode] for x in records]
        pos = [x['mad_mm'][mode] for x in records if x['gt_visible_fraction'] >= gt_fraction]
        neg = [x['mad_mm'][mode] for x in records if x['gt_visible_fraction'] < gt_fraction]
        mean = lambda x: round(float(np.median(x)), 4) if x else None
        results[mode] = {
            **conf,
            'median_patch_mad_mm': mean(vals),
            'median_true_positive_region_mad_mm': mean(pos),
            'median_background_region_mad_mm': mean(neg),
        }
    paired = {}
    for mode in ABLATIONS[1:]:
        d = [p['mad_mm'][mode] - p['mad_mm']['ideal'] for p in records]
        paired[f'{mode}_minus_ideal_median_mad_mm'] = (
            round(float(np.median(d)), 4) if d else None)
    return {
        'n_sampled_direct_wall_rays': int(size),
        'n_geometrically_valid_rays': int(valid.sum()),
        'n_physical_wall_patches': len(records),
        'modes': results,
        'paired_ablation_shifts': paired,
        'threshold_mad_mm': threshold,
        'gt_visible_fraction_threshold': gt_fraction,
        'notes': ['Physical wall/face/cell oracle grouping; cannot be scored as blind segmentation.',
                  'Four ablations reuse identical sampled direct rays and fitted physical patches.',
                  'Ground truth from interpolated pre-deformation envelope on hit primitive.',
                  'Only direct wall rays; mixed, virtual, glass and beam-edge rays excluded.'],
    }
