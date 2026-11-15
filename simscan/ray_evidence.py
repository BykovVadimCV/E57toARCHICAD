"""Sparse, station-linked *direct-return* 3D ray evidence for synthetic scans.

No claim is made for invalid/no-return beams or for mirror, glass-transmission,
beam-edge and mixed returns. A 2D free-pixel in a Z band means at least ONE
sampled direct ray traversed that pixel AT SOME height in that band. It does not
certify the full vertical column, nor does it become a background label.

Only physical scanner measurements and poses are used to build RaySamples.
GT is consulted later solely to audit whether a passage is independently
supported; it is never used to manufacture a ray.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

RAY_CONTRACT = 'station-direct-returns-height-bands-v1'
RAY_BANDS = ('ray_free_low_hd.png', 'ray_free_mid_hd.png', 'ray_free_high_hd.png')
MAX_DIRECT_RAYS_PER_STATION = 12_000
ENDPOINT_GUARD_M = 0.08
ORIGIN_GUARD_M = 0.03
MAX_PASSAGE_STRIPS = 3
MAX_PASSAGE_PROOF_FRACTION = 0.20
MAX_PASSAGE_STRIP_HALF_WIDTH_M = 0.012



@dataclass(frozen=True)
class RaySamples:
    """Small sampled subset of valid rays, indexed in valid-return source order."""
    origins: np.ndarray  # (N,3), actual scanner station pose in world coords
    endpoints: np.ndarray  # (N,3), measured direct return in same coords
    valid_indices: np.ndarray  # (N,) positions among all *valid* scene returns
    total_valid: int
    station_count: int

    def __post_init__(self):
        n = len(self.valid_indices)
        if (self.origins.shape != (n, 3) or self.endpoints.shape != (n, 3)
                or self.valid_indices.shape != (n,) or not isinstance(self.total_valid, int)
                or self.total_valid < 0 or self.station_count < 0):
            raise ValueError('Invalid station-linked 3D ray samples')
        if (not np.issubdtype(self.valid_indices.dtype, np.integer)
                or np.any(self.valid_indices < 0)
                or np.any(self.valid_indices >= self.total_valid)
                or np.any(np.diff(self.valid_indices) <= 0)
                or not np.isfinite(self.origins).all()
                or not np.isfinite(self.endpoints).all()):
            raise ValueError('Invalid physical ray provenance or geometry')


def capture_direct_returns(scans, poses, max_per_station=MAX_DIRECT_RAYS_PER_STATION) -> RaySamples:
    """Capture a bounded subset directly from ScanResult before it is freed.

    Non-direct ray flags are mandatory: older/manual scans without the flags
    cannot be used to assert physical free space. Indices mirror the valid-only
    iteration order of both E57Source and MemorySource for exact thinning.
    """
    if len(scans) != len(poses) or max_per_station < 1:
        raise ValueError('Scan poses/sampling budget mismatch')
    starts, ends, indices = [], [], []
    offset = 0
    for scan, pose in zip(scans, poses):
        valid = np.asarray(scan.valid, dtype=bool)
        count = int(np.count_nonzero(valid))
        direct = getattr(scan, 'straight_valid', None)
        if direct is None or np.asarray(direct).shape != valid.shape:
            raise ValueError('Scanner did not supply trustworthy straight-path flags')
        stride = max(1, math.ceil(count / max_per_station))
        # A temporary O(valid_count) index vector, not a persistent full beam grid.
        idx = np.flatnonzero(valid)[::stride]
        source_idx = offset + np.arange(0, count, stride, dtype=np.int64)
        assert len(idx) == len(source_idx)
        good = np.asarray(direct, dtype=bool)[idx] & ~scan.virtual[idx] & ~scan.mixed[idx]
        idx, source_idx = idx[good], source_idx[good]
        if len(idx):
            R, t = pose
            R = np.asarray(R, dtype=np.float64)
            t = np.asarray(t, dtype=np.float64)
            if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
                raise ValueError('Invalid station pose')
            # E57 single-precision quantization matches MemorySource/E57Source.
            e = np.asarray(scan.xyz_local[idx], dtype=np.float32).astype(np.float64) @ R.T + t
            starts.append(np.broadcast_to(t, e.shape).copy())
            ends.append(e)
            indices.append(source_idx)
        offset += count
    zeros = np.empty((0, 3), np.float64)
    return RaySamples(np.concatenate(starts) if starts else zeros,
                      np.concatenate(ends) if ends else zeros.copy(),
                      np.concatenate(indices) if indices else np.empty(0, np.int64),
                      offset, len(scans))


def thinning_keep_mask(samples: RaySamples, thin: float, sampling_seed: int | None,
                       chunk_size: int = 2_000_000) -> np.ndarray:
    """Replay the *exact* native/coarse RNG sampling stream for selected rays.

    Initial pass consumes one RNG value per valid XYZ point; second pass picks
    kept points. No large complete selection mask is allocated.
    """
    if not 0 <= thin < 1 or chunk_size < 1:
        raise ValueError('Invalid thin or chunk size')
    if thin == 0:
        return np.ones(len(samples.valid_indices), dtype=bool)
    if sampling_seed is None:
        raise ValueError('Thinned rays require the rasterizer seed')
    rng = np.random.default_rng(sampling_seed)
    for offset in range(0, samples.total_valid, chunk_size):
        rng.random(min(chunk_size, samples.total_valid-offset))
    selected = np.zeros(len(samples.valid_indices), dtype=bool)
    k = 0
    for offset in range(0, samples.total_valid, chunk_size):
        m = min(chunk_size, samples.total_valid-offset)
        random_values = rng.random(m)
        j = int(np.searchsorted(samples.valid_indices, offset + m))
        if j > k:
            selected[k:j] = random_values[samples.valid_indices[k:j] - offset] >= thin
        k = j
    return selected


def _frame_coordinates(p: np.ndarray, frame: dict):
    scale = float(frame['pixel_m'])
    x = (p[0] - float(frame['origin_x'])) / scale
    y = int(frame['height']) - 1 - (p[1] - float(frame['origin_y'])) / scale
    return int(math.floor(x)), int(math.floor(y))


def _clip_segment_to_z(origin, delta, start: float, stop: float, zmin: float, zmax: float):
    dz = float(delta[2])
    if abs(dz) < 1.e-10:
        return (start, stop) if zmin <= origin[2] < zmax else None
    t0, t1 = sorted(((zmin-origin[2])/dz, (zmax-origin[2])/dz))
    a, b = max(start, t0), min(stop, t1)
    return (a, b) if b > a else None


def render_direct_ray_bands(samples: RaySamples | None, frame: dict, floor_z: float,
                            ceiling_z: float, *, rotation=None, thin: float = 0.,
                            sampling_seed: int | None = None) -> tuple[dict, dict, RaySamples | None]:
    """Rasterize sampled 3D beam paths with height-band clipping and hit guards.

    The output is *weak free-path evidence*. A 2D XY projection cannot assert
    vacancy through an entire vertical slice. No negatives are learned from it.
    """
    from .netinput import BAND_MARGIN_M, LOW_TOP, MID_TOP, rotate_xy
    H, W = int(frame['height']), int(frame['width'])
    if H < 1 or W < 1 or H * W > 64_000_000:
        raise ValueError('Invalid ray HD frame')
    bands = {name: np.zeros((H, W), np.uint8) for name in RAY_BANDS}
    if samples is None:
        return bands, {'available': False, 'reason': 'station_linked_direct_ray_samples_unavailable',
                       'format': RAY_CONTRACT, 'ray_count': 0}, None
    if ceiling_z <= floor_z + MID_TOP + BAND_MARGIN_M:
        raise ValueError('Ceiling too low for the configured ray bands')
    keep = thinning_keep_mask(samples, thin, sampling_seed)
    good_origins, good_ends = samples.origins[keep], samples.endpoints[keep]
    if rotation is not None:
        good_origins = rotate_xy(good_origins, rotation)
        good_ends = rotate_xy(good_ends, rotation)
    kept = RaySamples(good_origins, good_ends,
                      samples.valid_indices[keep], samples.total_valid, samples.station_count)
    z_ranges = (
        (floor_z + BAND_MARGIN_M, floor_z + LOW_TOP),
        (floor_z + LOW_TOP, floor_z + MID_TOP),
        (floor_z + MID_TOP, ceiling_z - BAND_MARGIN_M),
    )
    counts = [0, 0, 0]
    for origin, endpoint in zip(good_origins, good_ends):
        delta = endpoint - origin
        length = float(np.linalg.norm(delta))
        if not math.isfinite(length) or length < ORIGIN_GUARD_M + ENDPOINT_GUARD_M + .02:
            continue
        start = ORIGIN_GUARD_M/length
        stop = 1. - ENDPOINT_GUARD_M/length
        for i, (lo, hi) in enumerate(z_ranges):
            interval = _clip_segment_to_z(origin, delta, start, stop, lo, hi)
            if interval is None:
                continue
            p0, p1 = origin + interval[0]*delta, origin + interval[1]*delta
            a, b = _frame_coordinates(p0, frame), _frame_coordinates(p1, frame)
            ok, a, b = cv2.clipLine((0, 0, W, H), a, b)
            if ok:
                cv2.line(bands[RAY_BANDS[i]], a, b, 255, 1, cv2.LINE_8)
                counts[i] += 1
    meta = {'format': RAY_CONTRACT, 'available': True, 'source':
            'sampled_valid_direct_station_returns_only',
            'ray_count_before_thin': len(samples.valid_indices),
            'ray_count': len(good_origins), 'station_count': samples.station_count,
            'total_valid_returns': samples.total_valid,
            'max_rays_per_station': MAX_DIRECT_RAYS_PER_STATION,
            'origin_guard_m': ORIGIN_GUARD_M, 'endpoint_guard_m': ENDPOINT_GUARD_M,
            'height_bands_m': [[float(a-floor_z), float(b-floor_z)] for a,b in z_ranges],
            'rasterized_segments_by_band': dict(zip(RAY_BANDS, counts)),
            'occupied_cells_by_band': {k:int(np.count_nonzero(v)) for k,v in bands.items()},
            'thin': float(thin), 'sampling_seed': sampling_seed,
            'policy': 'sparse 3d ray crossings; never whole-column free or automatic background labels',
            }
    return bands, meta, kept


def audit_passage_crossings(gt, rays: RaySamples | None, frame: dict,
                            support: np.ndarray, openings_raster: np.ndarray,
                            floor_z: float, *, min_crossings: int = 2,
                            edge_margin_m: float = .03) -> tuple[np.ndarray, dict]:
    """GT-localized passage proof from measured cross-wall direct ray segments.

    Uses GT only to decide if an existing passage target is supervised, never
    as a network input. Requires two observed jambs and >=2 direct rays through
    the *full wall thickness*, at heights inside the physical opening.
    """
    from .observation import audit_object_support
    from .semantic_contract import HD_OPENING_VALUES
    h, w = openings_raster.shape
    mask = np.zeros((h, w), np.uint8)
    if rays is None or len(rays.valid_indices) == 0:
        return mask, {'policy': 'no_passage_confirmation_without_valid_direct_rays', 'openings': []}
    if support.shape != mask.shape or min_crossings < 2:
        raise ValueError('Invalid passage audit evidence')
    jamb_audit = audit_object_support(gt, frame, support)
    jambs = {x['opening_id']:x['both_jambs_near_returns'] for x in jamb_audit['openings']}
    out = []
    for op in gt.layout.openings:
        if op.kind != 'passage':
            continue
        wall = gt.layout.wall(op.wall_id)
        if wall.role == 'open_balcony':
            continue
        centre = gt.xy(np.asarray([wall.point(float(op.s), 0.)]))[0]
        a = gt.xy(np.asarray([wall.point(max(0., float(op.s)-.025), 0.)]))[0]
        b = gt.xy(np.asarray([wall.point(min(wall.length, float(op.s)+.025), 0.)]))[0]
        tangent = b - a
        ln = float(np.linalg.norm(tangent))
        if ln < 1.e-8:
            continue
        tangent /= ln
        normal = np.array([-tangent[1], tangent[0]])
        d0 = (rays.origins[:, :2] - centre) @ normal
        d1 = (rays.endpoints[:, :2] - centre) @ normal
        # Require a return at least 8 cm past the far face so the wall-body
        # traversal is separated from an endpoint (door leaf, glass, etc.).
        offset = wall.thickness/2 + ENDPOINT_GUARD_M
        crosses = ((d0 < -offset) & (d1 > offset)) | ((d1 < -offset) & (d0 > offset))
        loc = np.nonzero(crosses)[0]
        good_s = np.empty(0, dtype=np.float64)
        if not len(loc):
            count = 0
        else:
            f = -d0[loc]/(d1[loc]-d0[loc])
            intersection = rays.origins[loc] + f[:, None]*(rays.endpoints[loc]-rays.origins[loc])
            s = (intersection[:, :2] - centre) @ tangent
            z = intersection[:, 2] - floor_z
            admissible = ((np.abs(s) < op.width/2-edge_margin_m)
                          & (z > op.z0+edge_margin_m)
                          & (z < op.z1-edge_margin_m))
            good_s = s[admissible]
            count = len(good_s)
        # Multiple almost-identical hits must not impersonate distinct paths
        # through an opening. Binning is conservative at 40 mm resolution.
        spatial_bins = len(np.unique(np.floor(good_s / .04).astype(int)))
        confirmed = bool(jambs.get(int(op.id), False)
                         and count >= min_crossings
                         and spatial_bins >= min_crossings)
        # Require localized, measured cross-sections, not all 20mm-quantized
        # rays. Nearby strips previously joined into a near-full GT aperture.
        # Select at most three separated physical ray positions for each
        # passage and cap *raster area*, not merely nominal strip widths.
        confirmed_pixels = 0
        domain_pixels = 0
        selected_offsets = []
        if confirmed:
            from .netinput import _fill
            domain = np.zeros_like(mask)
            full_poly = gt.wall_quad(wall, op.s-op.width/2, op.s+op.width/2,
                                     -wall.thickness/2-.01, wall.thickness/2+.01)
            _fill(domain, full_poly, frame, 255)
            domain = (domain != 0) & (openings_raster == HD_OPENING_VALUES['passage'])
            domain_pixels = int(np.count_nonzero(domain))
            unique_positions = np.unique(np.round(good_s / .02).astype(np.int32)) * .02
            # Keep extreme beams first to avoid clustering around the center;
            # an optional third beam may be chosen only if spatially distinct.
            if unique_positions.size >= 2:
                selected_offsets = [float(unique_positions[0]), float(unique_positions[-1])]
                centre_position = float(unique_positions[len(unique_positions)//2])
                if (len(unique_positions) > 2
                        and min(abs(centre_position-p) for p in selected_offsets) >= .06):
                    selected_offsets.append(centre_position)
            # A strip is only meaningful when supported by measured rays.
            # This width bound includes the physical pixel scale to allow
            # subcentimeter apertures to be represented without guessing.
            half = min(MAX_PASSAGE_STRIP_HALF_WIDTH_M,
                       .35 * op.width / max(1, 2*len(selected_offsets)))
            proof = None
            for attempt in range(5):
                candidate = np.zeros_like(mask)
                for s_offset in selected_offsets:
                    ss = float(op.s + s_offset)
                    poly = gt.wall_quad(wall, max(op.s-op.width/2, ss-half),
                                        min(op.s+op.width/2, ss+half),
                                        -wall.thickness/2-.01, wall.thickness/2+.01)
                    _fill(candidate, poly, frame, 255)
                candidate = (candidate != 0) & domain
                candidate_pixels = int(np.count_nonzero(candidate))
                if (domain_pixels > 0 and candidate_pixels > 0
                        and candidate_pixels <= MAX_PASSAGE_PROOF_FRACTION*domain_pixels):
                    proof = candidate
                    confirmed_pixels = candidate_pixels
                    break
                half *= .5
            # If no sufficiently narrow measured strips fit, do not mark the
            # full region valid. A validated crossing remains in the audit.
            if proof is None:
                confirmed = False
            else:
                mask[proof] = 255
        out.append({'opening_id': int(op.id), 'crossing_direct_ray_samples': count,
                    'distinct_crossing_bins': spatial_bins,
                    'both_jambs_near_returns': bool(jambs.get(int(op.id), False)),
                    'representative_ray_offsets_m': selected_offsets,
                    'physical_area_px': domain_pixels,
                    'confirmed_area_px': confirmed_pixels,
                    'confirmed_area_fraction': (round(confirmed_pixels/domain_pixels, 6)
                                                 if domain_pixels else 0.),
                    'max_area_fraction': MAX_PASSAGE_PROOF_FRACTION,
                    'confirmed_for_supervision': confirmed})
    return mask, {'policy': '2+ distinct cross-wall straight rays + two near-return jambs; <=3 separated measured strips and <=20pct GT per-opening area',
                  'openings': out, 'confirmed_count': sum(x['confirmed_for_supervision'] for x in out)}
