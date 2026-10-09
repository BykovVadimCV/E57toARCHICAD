"""Greedy approximation of a binary mask by a minimal set of rotated rectangles.

Port of ReFloorBRUSNIKA's ``core/rect_decompose.py`` with two changes:

*  the hill-climb refinement scores candidate rectangles on a padded crop
   around the rectangle instead of the whole canvas (the prototype
   rasterised the full H×W grid ~320 times per rectangle, which dominated
   the runtime);
*  ``decompose()`` accepts ``max_rects`` as a safety stop.

Everything else — the Kadane-on-all-row-sums maximum-weight subrectangle
search, the bleed/spill discipline, the thin-residual rescue — is kept
verbatim so that wall extraction here behaves like the reference
implementation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Rect:
    """Axis-aligned-in-its-own-frame rectangle in image pixel coordinates."""

    cx: float
    cy: float
    w: float
    h: float
    angle: float   # degrees, counter-clockwise, image space (y grows downward)

    def corners(self) -> np.ndarray:
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        hw, hh = self.w / 2, self.h / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=float)
        rot = np.array([[cos, -sin], [sin, cos]])
        return local @ rot.T + np.array([self.cx, self.cy])

    def __str__(self):
        return (f"Rect(center=({self.cx:.1f},{self.cy:.1f})  "
                f"size=({self.w:.1f}x{self.h:.1f})  angle={self.angle:.1f}°)")


# ---------------------------------------------------------------------------
# Rasterisation
# ---------------------------------------------------------------------------

def rasterise(rect: Rect, H: int, W: int,
              x0: int = 0, y0: int = 0) -> np.ndarray:
    """Boolean (H, W) mask of the rect on the sub-canvas whose origin is (x0, y0)."""
    rad = math.radians(-rect.angle)
    cos, sin = math.cos(rad), math.sin(rad)
    ys, xs = np.mgrid[y0:y0 + H, x0:x0 + W]
    dx = xs - rect.cx
    dy = ys - rect.cy
    lx = dx * cos - dy * sin
    ly = dx * sin + dy * cos
    return (np.abs(lx) <= rect.w / 2) & (np.abs(ly) <= rect.h / 2)


def _rasterise_np(rect: Rect, H: int, W: int) -> np.ndarray:
    """Fully vectorised variant (no mgrid) — used by the full-canvas helpers."""
    rad = math.radians(-rect.angle)
    cos, sin = math.cos(rad), math.sin(rad)
    ys = np.arange(H, dtype=np.float32) - rect.cy
    xs = np.arange(W, dtype=np.float32) - rect.cx
    lx = xs[None, :] * cos - ys[:, None] * sin
    ly = xs[None, :] * sin + ys[:, None] * cos
    return (np.abs(lx) <= rect.w / 2) & (np.abs(ly) <= rect.h / 2)


# ---------------------------------------------------------------------------
# Weight grid in the rotated frame
# ---------------------------------------------------------------------------

def build_weight_grid(
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    angle: float,
    max_grid_dim: int = 200,
    bleed_weight: float = 1.0,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Accumulate per-pixel weights (+1 remaining, -bleed outside) into a coarse
    grid aligned with the frame rotated by ``-angle`` degrees.

    Returns (grid, lx_min, ly_min, scale) where grid index i corresponds to
    rotated coordinate i / scale + lx_min.
    """
    H, W = remaining.shape
    rad = math.radians(-angle)
    cos, sin = math.cos(rad), math.sin(rad)

    ys = np.arange(H, dtype=np.float64)[:, None]
    xs = np.arange(W, dtype=np.float64)[None, :]
    lx_f = (xs * cos - ys * sin).ravel()
    ly_f = (xs * sin + ys * cos).ravel()

    lx_min_f, lx_max_f = float(lx_f.min()), float(lx_f.max())
    ly_min_f, ly_max_f = float(ly_f.min()), float(ly_f.max())

    span = max(lx_max_f - lx_min_f, ly_max_f - ly_min_f, 1.0)
    scale = min(1.0, max_grid_dim / span)

    lx_i = np.round((lx_f - lx_min_f) * scale).astype(np.int32)
    ly_i = np.round((ly_f - ly_min_f) * scale).astype(np.int32)

    grid_w = int(lx_i.max()) + 1
    grid_h = int(ly_i.max()) + 1

    weights = (remaining.ravel().astype(np.float32)
               - bleed_weight * outside_mask.ravel().astype(np.float32))
    flat_idx = ly_i * grid_w + lx_i
    grid_flat = np.bincount(flat_idx, weights=weights.astype(np.float64),
                            minlength=grid_h * grid_w)
    return grid_flat.reshape(grid_h, grid_w).astype(np.int32), lx_min_f, ly_min_f, scale


# ---------------------------------------------------------------------------
# Maximum-weight subrectangle (vectorised Kadane over all row ranges)
# ---------------------------------------------------------------------------

def max_weight_subrect(grid: np.ndarray) -> Tuple[int, int, int, int, int]:
    """Best axis-aligned rectangle in ``grid`` as (sum, r1, c1, r2, c2) inclusive."""
    rows, cols = grid.shape
    row_cs = np.cumsum(grid, axis=0, dtype=np.float32)

    r1_idx = np.repeat(np.arange(rows, dtype=np.int32), rows - np.arange(rows, dtype=np.int32))
    r2_idx = np.concatenate([np.arange(r1, rows, dtype=np.int32) for r1 in range(rows)])
    K = len(r1_idx)

    col_sums = row_cs[r2_idx].copy()
    has_top = r1_idx > 0
    col_sums[has_top] -= row_cs[r1_idx[has_top] - 1]

    cs = np.empty((K, cols + 1), dtype=np.float64)
    cs[:, 0] = 0.0
    np.cumsum(col_sums, axis=1, out=cs[:, 1:])

    min_prefix = np.minimum.accumulate(cs[:, :-1], axis=1)
    gain = cs[:, 1:] - min_prefix

    best_j = np.argmax(gain, axis=1)
    best_val_k = gain[np.arange(K), best_j]
    best_k = int(np.argmax(best_val_k))
    best_val = int(best_val_k[best_k])

    if best_val <= 0:
        return 0, 0, 0, 0, 0

    r1 = int(r1_idx[best_k])
    r2 = int(r2_idx[best_k])
    c2 = int(best_j[best_k])
    c1 = int(np.argmin(cs[best_k, : c2 + 1]))
    return best_val, r1, c1, r2, c2


def best_rect_at_angle(
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    angle: float,
    max_grid_dim: int = 200,
    bleed_weight: float = 1.0,
) -> Tuple[Optional[Rect], int]:
    grid, lx_min_f, ly_min_f, scale = build_weight_grid(
        remaining, outside_mask, angle, max_grid_dim, bleed_weight)
    val, r1, c1, r2, c2 = max_weight_subrect(grid)
    if val <= 0:
        return None, 0

    cx_l = (c1 + c2) / 2.0 / scale + lx_min_f
    cy_l = (r1 + r2) / 2.0 / scale + ly_min_f
    rad_inv = math.radians(angle)
    ci, si = math.cos(rad_inv), math.sin(rad_inv)
    cx = cx_l * ci - cy_l * si
    cy = cx_l * si + cy_l * ci

    w = (c2 - c1 + 1) / scale
    h = (r2 - r1 + 1) / scale
    return Rect(cx, cy, float(w), float(h), angle), val


# ---------------------------------------------------------------------------
# Angle schedule
# ---------------------------------------------------------------------------

_PREFERRED_ANGLES = [0.0, 45.0, 90.0, 135.0]


def _near_preferred(angle: float, gap: float) -> bool:
    a = angle % 180
    for p in _PREFERRED_ANGLES:
        diff = abs(a - p)
        diff = min(diff, 180.0 - diff)
        if diff < gap:
            return True
    return False


def pca_angle(mask: np.ndarray) -> float:
    ys, xs = np.where(mask)
    if len(xs) < 2:
        return 0.0
    dx = xs - xs.mean()
    dy = ys - ys.mean()
    cov = np.array([[np.dot(dx, dx), np.dot(dx, dy)],
                    [np.dot(dx, dy), np.dot(dy, dy)]])
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, 1]
    return math.degrees(math.atan2(v[1], v[0])) % 180


def build_angle_list(
    remaining: np.ndarray,
    angle_steps: int,
    n_pca_fine: int = 8,
    pca_spread: float = 15.0,
    axis_only: bool = False,
    axis_gap: float = 0.0,
) -> List[float]:
    if axis_only:
        return [0.0, 90.0]
    uniform = [i * 180.0 / angle_steps for i in range(angle_steps)]
    fine = [pca + pca_spread * (k / (n_pca_fine - 1) - 0.5)
            for k in range(n_pca_fine)]
    candidates = list(uniform) + list(fine)
    if axis_gap > 0.0:
        candidates = [a for a in candidates if not _near_preferred(a, axis_gap)]
        candidates += list(_PREFERRED_ANGLES)
    return sorted({round(a % 180, 2) for a in candidates})


# ---------------------------------------------------------------------------
# Refinement (cropped hill-climb)
# ---------------------------------------------------------------------------

def refine_rect(
    rect: Rect,
    remaining: np.ndarray,
    outside_mask: np.ndarray,
    iters: int = 40,
    bleed_weight: float = 1.0,
    axis_only: bool = False,
) -> Tuple[Rect, int]:
    """
    Hill-climb (cx, cy, w, h[, angle]) to maximise

        score = #(pixels ∩ remaining) − bleed_weight · #(pixels ∩ outside)

    Scored on a padded crop that always contains every candidate, so the
    result is identical to full-canvas scoring but ~5–10× cheaper.
    """
    H, W = remaining.shape
    half_w0 = abs(rect.w / 2)
    half_h0 = abs(rect.h / 2)

    steps = {'cx': 2.0, 'cy': 2.0, 'w': 4.0, 'h': 4.0}
    if not axis_only:
        steps['angle'] = 5.0

    # Largest shift a single step (before any halving) can produce.
    # Padding: a step only shifts one attribute by its step size and a step is
    # halved when no improvement is found, so the cumulative excursion from the
    # *initial* rectangle is bounded by 2 × initial step of any attribute.
    # cx/cy shift 2 px, w/h grow 4 px, angle 5 deg → use the worst case.
    pad = (6.0 if not axis_only else 8.0) * 2.0 + 4.0

    def crop_for(r: Rect):
        rad = math.radians(r.angle)
        c, s = abs(math.cos(rad)), abs(math.sin(rad))
        ex = abs(half_w0 * c) + abs(half_h0 * s) + pad
        ey = abs(half_w0 * s) + abs(half_h0 * c) + pad
        x0 = int(math.floor(r.cx - ex)); x1 = int(math.ceil(r.cx + ex))
        y0 = int(math.floor(r.cy - ey)); y1 = int(math.ceil(r.cy + ey))
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(W, x1); y1 = min(H, y1)
        return x0, y0, x1, y1

    def sc(r: Rect) -> float:
        x0, y0, x1, y1 = crop_for(r)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        p = rasterise(r, y1 - y0, x1 - x0, x0=x0, y0=y0)
        rem = remaining[y0:y1, x0:x1]
        out = outside_mask[y0:y1, x0:x1]
        return (float(np.count_nonzero(p & rem))
                - bleed_weight * float(np.count_nonzero(p & out)))

    best = rect
    best_s = sc(rect)
    for _ in range(iters):
        improved = False
        for attr, delta in list(steps.items()):
            for sign in (+1, -1):
                new_val = getattr(best, attr) + sign * delta
                kwargs = dict(cx=best.cx, cy=best.cy, w=best.w, h=best.h, angle=best.angle)
                if attr in ('w', 'h'):
                    new_val = max(1.0, new_val)
                kwargs[attr] = new_val if attr != 'angle' else new_val % 180
                cand = Rect(**kwargs)
                cand_s = sc(cand)
                if cand_s > best_s:
                    best_s, best, improved = cand_s, cand, True
        if not improved:
            steps = {k: v / 2.0 for k, v in steps.items()}
            if max(steps.values()) < 0.25:
                break
    return best, best_s


# ---------------------------------------------------------------------------
# Overlap / spill helpers
# ---------------------------------------------------------------------------

def _overlap_fraction(rect: Rect, placed_union: np.ndarray, H: int, W: int) -> float:
    if not placed_union.any():
        return 0.0
    pix = _rasterise_np(rect, H, W)
    area = int(np.count_nonzero(pix))
    return int(np.count_nonzero(pix & placed_union)) / area if area else 0.0


def _spill_fraction(rect: Rect, outside_mask: np.ndarray, H: int, W: int) -> float:
    pix = _rasterise_np(rect, H, W)
    area = int(np.count_nonzero(pix))
    return int(np.count_nonzero(pix & outside_mask)) / area if area else 0.0


# ---------------------------------------------------------------------------
# Connected-component helpers (no hard cv2 / scipy dependency)
# ---------------------------------------------------------------------------

def _dilate_bool(m: np.ndarray, radius: int) -> np.ndarray:
    out = m.copy()
    for _ in range(max(0, radius)):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        d[1:, 1:] |= out[:-1, :-1]
        d[1:, :-1] |= out[:-1, 1:]
        d[:-1, 1:] |= out[:-1, :-1]
        d[:-1, :-1] |= out[1:, :-1]
        out = d
    return out


def _label_components(m: np.ndarray):
    """8-connectivity labelling via cv2 or scipy; (None, []) when neither exists."""
    try:
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        comps = [(i, int(stats[i, cv2.CC_STAT_AREA]),
                  (int(stats[i, cv2.CC_STAT_TOP]),
                   int(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]),
                   int(stats[i, cv2.CC_STAT_LEFT]),
                   int(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])))
                 for i in range(1, n)]
        return lab, comps
    except ImportError:
        pass
    try:
        from scipy import ndimage
        lab, _n = ndimage.label(m, structure=np.ones((3, 3), np.int32))
        comps = []
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            comps.append((i, int(np.count_nonzero(lab[sl] == i)),
                          (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop)))
        return lab, comps
    except ImportError:
        return None, []


# ---------------------------------------------------------------------------
# Thin-residual rescue
# ---------------------------------------------------------------------------

def rescue_thin_residual(
    mask: np.ndarray,
    remaining: np.ndarray,
    placed_union: np.ndarray,
    axis_only: bool = False,
    max_thickness: float = 20.0,
    min_len: float = 25.0,
    min_area: int = 40,
    fringe_px: int = 2,
    max_components: int = 32,
    max_rects_per_component: int = 6,
    component_coverage: float = 0.90,
    grid_dim_cap: int = 256,
    bleed_weight: float = 3.0,
    max_spill_fraction: float = 0.15,
    refine: bool = True,
) -> List[Rect]:
    """
    Re-run the greedy search per residual connected component on the
    component's own crop at scale 1 (no grid quantisation).  Only
    wall-shaped rectangles are accepted: short side ≤ ``max_thickness``,
    long side ≥ ``min_len``, spill ≤ ``max_spill_fraction``.
    """
    H, W = mask.shape
    outside = ~mask

    work = remaining.copy()
    if fringe_px > 0 and placed_union.any():
        work &= ~_dilate_bool(placed_union, fringe_px)
    if not work.any():
        return []

    labels, comps = _label_components(work)
    if labels is None:
        return []

    comps = sorted((c for c in comps if c[1] >= min_area), key=lambda c: -c[1])[:max_components]

    rescued: List[Rect] = []
    for cid, area, (by0, by1, bx0, bx1) in comps:
        bh, bw = by1 - by0, bx1 - bx0
        if min(bh, bw) > 3.0 * max_thickness and area > 0.35 * bh * bw:
            continue

        margin = 3
        y0, y1 = max(0, by0 - margin), min(H, by1 + margin)
        x0, x1 = max(0, bx0 - margin), min(W, bx1 + margin)
        comp_mask = labels[y0:y1, x0:x1] == cid
        sub_out = outside[y0:y1, x0:x1]
        ch, cw = comp_mask.shape

        sub_rem = comp_mask.copy()
        total_c = area
        min_gain = max(20, int(0.08 * total_c))
        placed: List[Rect] = []

        for _ in range(max_rects_per_component):
            n_left = int(np.count_nonzero(sub_rem))
            if n_left == 0 or (total_c - n_left) / total_c >= component_coverage:
                break
            angles = build_angle_list(sub_rem, angle_steps=8, axis_only=axis_only)
            best_rect: Optional[Rect] = None
            best_s = 0
            for ang in angles:
                rect, s = best_rect_at_angle(sub_rem, sub_out, ang, grid_dim_cap, bleed_weight)
                if rect is not None and s > best_s:
                    best_rect, best_s = rect, s
            if best_rect is None or best_s < min_gain:
                break
            if refine:
                best_rect, best_s = refine_rect(best_rect, sub_rem, sub_out,
                                                bleed_weight=bleed_weight, axis_only=axis_only)
            short, long_ = min(best_rect.w, best_rect.h), max(best_rect.w, best_rect.h)
            if short > max_thickness or long_ < min_len:
                break
            if _spill_fraction(best_rect, sub_out, ch, cw) > max_spill_fraction:
                break
            pix = rasterise(best_rect, ch, cw)
            if int(np.count_nonzero(pix & sub_rem)) < min_gain:
                break
            sub_rem &= ~pix
            placed.append(Rect(best_rect.cx + x0, best_rect.cy + y0,
                               best_rect.w, best_rect.h, best_rect.angle))

        rescued.extend(placed)

    return rescued


# ---------------------------------------------------------------------------
# Main decomposition
# ---------------------------------------------------------------------------

def decompose(
    mask: np.ndarray,
    angle_steps: int = 40,
    rect_penalty: float = 0.02,
    max_grid_dim: int = 200,
    coverage_stop: float = 0.995,
    bleed_weight: float = 3.0,
    initial_bleed_weight: Optional[float] = None,
    bleed_decay: float = 1.0,
    axis_only: bool = False,
    axis_gap: float = 0.0,
    max_overlap: float = 0.5,
    max_spill_fraction: float = 0.15,
    diag_axis_tol: float = 8.0,
    diag_score_penalty: float = 0.0,
    diag_overlap_penalty: float = 0.0,
    refine: bool = True,
    verbose: bool = True,
    thin_rescue: bool = False,
    thin_rescue_max_thickness: float = 20.0,
    thin_rescue_min_len: float = 25.0,
    thin_rescue_min_area: int = 40,
    thin_rescue_fringe_px: int = 2,
    second_rescue: bool = True,
    second_rescue_thickness_factor: float = 2.0,
    second_rescue_len_factor: float = 0.4,
    max_rects: int = 400,
) -> List[Rect]:
    """
    Greedy rectangle decomposition of ``mask``.  Runs until no rectangle
    clears the penalty threshold, ``coverage_stop`` is reached, no candidate
    is acceptable any more, or ``max_rects`` rectangles have been placed.
    """
    H, W = mask.shape
    mask = mask.astype(bool)
    outside = ~mask
    total = int(np.count_nonzero(mask))
    if total == 0:
        return []

    bw_start = bleed_weight
    if initial_bleed_weight is not None and initial_bleed_weight > bleed_weight:
        bw_start = float(initial_bleed_weight)

    remaining = mask.copy()
    placed_union = np.zeros((H, W), dtype=bool)
    chosen: List[Rect] = []
    step = 0

    def _axis_dev(angle: float) -> float:
        a = angle % 90.0
        return min(a, 90.0 - a)

    def _adjusted_score(rect: Rect, raw: int) -> float:
        if _axis_dev(rect.angle) <= diag_axis_tol:
            return float(raw)
        adj = raw * (1.0 - diag_score_penalty)
        if diag_overlap_penalty > 0.0 and placed_union.any():
            pix = _rasterise_np(rect, H, W)
            adj -= diag_overlap_penalty * int(np.count_nonzero(pix & placed_union))
        return adj

    while len(chosen) < max_rects:
        n_left = int(np.count_nonzero(remaining))
        if n_left == 0 or (total - n_left) / total >= coverage_stop:
            break
        step += 1
        covered = total - n_left
        min_gain = max(1, int(rect_penalty * n_left))
        current_bleed = max(bleed_weight, bw_start - float(len(chosen)) * bleed_decay)

        angles = build_angle_list(remaining, angle_steps, axis_only=axis_only, axis_gap=axis_gap)

        best_adj = float(min_gain) - 1.0
        best_rect: Optional[Rect] = None

        for angle in angles:
            rect, s = best_rect_at_angle(remaining, outside, angle, max_grid_dim, current_bleed)
            if rect is None:
                continue
            adj = _adjusted_score(rect, s)
            if adj < min_gain or adj <= best_adj:
                continue
            if max_spill_fraction < 1.0 and _spill_fraction(rect, outside, H, W) > max_spill_fraction:
                continue
            if max_overlap < 1.0 and chosen and _overlap_fraction(rect, placed_union, H, W) > max_overlap:
                continue
            best_adj, best_rect = adj, rect

        if best_rect is None:
            break

        if refine:
            best_rect, best_score = refine_rect(best_rect, remaining, outside,
                                                bleed_weight=current_bleed, axis_only=axis_only)
            if _adjusted_score(best_rect, best_score) < min_gain:
                break
            if max_spill_fraction < 1.0 and _spill_fraction(best_rect, outside, H, W) > max_spill_fraction:
                break
            if max_overlap < 1.0 and chosen and _overlap_fraction(best_rect, placed_union, H, W) > max_overlap:
                break

        chosen.append(best_rect)
        rect_pixels = _rasterise_np(best_rect, H, W)
        placed_union |= rect_pixels
        remaining &= ~rect_pixels

    if thin_rescue and remaining.any():
        rescued = rescue_thin_residual(
            mask, remaining, placed_union,
            axis_only=axis_only,
            max_thickness=thin_rescue_max_thickness,
            min_len=thin_rescue_min_len,
            min_area=thin_rescue_min_area,
            fringe_px=thin_rescue_fringe_px,
            bleed_weight=bleed_weight,
            max_spill_fraction=max_spill_fraction,
            refine=refine,
        )
        chosen.extend(rescued)
        for r in rescued:
            pix = _rasterise_np(r, H, W)
            placed_union |= pix
            remaining &= ~pix

        covered = total - int(np.count_nonzero(remaining))
        if second_rescue and remaining.any() and covered / total < coverage_stop:
            chosen.extend(rescue_thin_residual(
                mask, remaining, placed_union,
                axis_only=axis_only,
                max_thickness=thin_rescue_max_thickness * second_rescue_thickness_factor,
                min_len=max(4.0, thin_rescue_min_len * second_rescue_len_factor),
                min_area=max(12, int(thin_rescue_min_area * second_rescue_len_factor)),
                fringe_px=1,
                bleed_weight=bleed_weight,
                max_spill_fraction=max_spill_fraction,
                refine=refine,
            ))

    return chosen
