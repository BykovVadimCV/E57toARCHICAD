"""Conservative observation masks for native-HD synthetic supervision.

Only measured *returns* are available in the existing merged point-source
interface. In particular, missing returns do NOT establish a free ray, and a
nearby return does not prove its semantic class (it might be furniture).

These masks must never be appended to network input without updating its
contract. GT geometry is used only to decide which positive labels to withhold
from the loss, never to fabricate sensor observations.
"""
from __future__ import annotations

import math
import cv2
import numpy as np

from .hd_evidence import SCAN_HD_FILES

OBSERVATION_CONTRACT = "native-hd-observation-v1"
OBSERVATION_FILES = (
    "observation_occupied_hd.png",
    "observation_support_hd.png",
    "observation_free_hd.png",
    "observation_unknown_hd.png",
    "observation_uncertain_hd.png",
    "training_valid_hd.png",
)
SUPPORT_RADIUS_M = 0.012


def build_observation_masks(layers: dict[str, np.ndarray], frame: dict,
                            wall: np.ndarray, openings: np.ndarray,
                            floor_void: np.ndarray, legacy_valid: np.ndarray,
                            support_radius_m: float = SUPPORT_RADIUS_M,
                            free_rays: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Return 0/255 masks and a JSON-safe provenance summary.

    ``support`` means spatial proximity to a structural-height scan *return*,
    NOT an independently verified wall/door/window hit. This deliberately
    errs on the side of masking positive GT without return proximity.

    ``free_rays`` may be passed only when an upstream scanner explicitly
    supplies trustworthy, height-specific free-ray traces. Without it,
    ``observation_free_hd.png`` is all zero, and the sidecar says UNAVAILABLE.
    In this version, training background validity continues to follow the
    legacy supervision policy, preserving its exterior false-positive penalty.
    """
    h, w = int(frame["height"]), int(frame["width"])
    pitch = float(frame["pixel_m"])
    if not 0.0 < pitch <= 0.1 or not math.isfinite(pitch):
        raise ValueError("Invalid observation pixel pitch")
    if not math.isfinite(support_radius_m) or not 0 <= support_radius_m <= 0.05:
        raise ValueError("support_radius_m must be finite and in [0, 0.05]")
    shape = h, w
    for name in SCAN_HD_FILES:
        if name not in layers or layers[name].shape != shape or layers[name].dtype != np.uint8:
            raise ValueError("Missing or invalid scan return band: " + name)
    for name, a in (("wall", wall), ("openings", openings),
                    ("floor_void", floor_void), ("legacy_valid", legacy_valid)):
        if a.shape != shape or a.dtype != np.uint8:
            raise ValueError("Expected uint8 native raster: " + name)
    if free_rays is not None:
        if free_rays.shape != shape or free_rays.dtype != np.uint8:
            raise ValueError("Free-ray evidence must be a uint8 HD raster")
        if not np.isin(free_rays, [0, 255]).all():
            raise ValueError("Free-ray evidence must be binary")
        # Do not allow a free-space assertion at an observed return endpoint.
        # A 3D projection may legitimately overlap in XY, so full 3D
        # reconciliation belongs to the future ray-evidence contract.
        raise ValueError("Free-ray input requires validated 3D height-band metadata; not yet supported")

    # Purely sensor-derived return occupancy, never filled from polygons.
    occupied = np.zeros(shape, np.uint8)
    for name in SCAN_HD_FILES:
        cv2.bitwise_or(occupied, layers[name], dst=occupied)
    occupied[occupied != 0] = 255

    # Local *return proximity* only: floor/ceiling reflections do not provide
    # support for an otherwise unseen wall or opening. These bands include
    # furniture, therefore the summary explicitly names this a weak proxy.
    structural_height_returns = np.zeros(shape, np.uint8)
    for name in SCAN_HD_FILES[:3]:
        cv2.bitwise_or(structural_height_returns, layers[name], dst=structural_height_returns)
    radius_px = int(math.ceil(support_radius_m / pitch))
    if radius_px:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * radius_px + 1, 2 * radius_px + 1))
        support = cv2.dilate(structural_height_returns, kernel)
    else:
        support = structural_height_returns.copy()
    support[support != 0] = 255

    # A physical polygon is not automatically an observable positive target.
    positive = (wall != 0) | (openings != 0) | (floor_void != 0)
    unresolved = positive & (support == 0)
    candidate = (legacy_valid != 0) & (~positive | (support != 0))

    free = np.zeros(shape, np.uint8)  # no actual ray origins linked to returns
    images = {
        "observation_occupied_hd.png": occupied,
        "observation_support_hd.png": support,
        "observation_free_hd.png": free,
        "observation_unknown_hd.png": cv2.bitwise_not(occupied),
        "observation_uncertain_hd.png": unresolved.astype(np.uint8) * 255,
        "training_valid_hd.png": candidate.astype(np.uint8) * 255,
    }
    stats = {
        "format": OBSERVATION_CONTRACT,
        "occupancy_source": "measured_return_pixels_only",
        "support_source": "dilation_of_low_mid_high_return_pixels; not_semantic_evidence",
        "support_radius_m": float(support_radius_m),
        "free_rays_available": False,
        "free_rays_policy": "unknown_without_station_linked_3d_ray_paths",
        "unknown_policy": "missing_return_is_not_free_space",
        "negative_background_policy": "inherit_20mm_supervision_to_avoid_exterior_hallucinations",
        "training_valid_policy": "legacy_background_plus_return_proximity_gated_positive_gt_v1",
        "legacy_valid_pixels": int(np.count_nonzero(legacy_valid)),
        "observed_return_pixels": int(np.count_nonzero(occupied)),
        "unknown_status_pixels": int(occupied.size - np.count_nonzero(occupied)),
        "return_proximity_pixels": int(np.count_nonzero(support)),
        "physical_positive_pixels": int(np.count_nonzero(positive)),
        "supported_positive_pixels": int(np.count_nonzero(positive & (support != 0))),
        "unsupported_positive_pixels": int(np.count_nonzero(unresolved)),
        "training_valid_pixels": int(np.count_nonzero(candidate)),
    }
    return images, stats


def audit_object_support(gt, frame: dict, support: np.ndarray,
                         sample_step_m: float = 0.05) -> dict:
    """Diagnostic return proximity along individual wall faces and opening jambs.

    GT supplies *where to sample* only. A positive reading is spatially
    correlated with any structural-height return, not proof of object identity.
    The results are never used as model input or to fill missing scan pixels.
    """
    h, w = int(frame['height']), int(frame['width'])
    pitch = float(frame['pixel_m'])
    if support.shape != (h, w) or sample_step_m <= 0:
        raise ValueError('Invalid wall observation audit geometry')

    def sample(points):
        xy = gt.xy(np.asarray(points, dtype=float))
        x = np.floor((xy[:, 0] - frame['origin_x']) / pitch).astype(np.int64)
        y = h - 1 - np.floor((xy[:, 1] - frame['origin_y']) / pitch).astype(np.int64)
        good = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        found = np.zeros(len(points), bool)
        found[good] = support[y[good], x[good]] != 0
        return found

    walls = []
    for wall in gt.layout.walls:
        if wall.role == 'open_balcony':
            continue
        n = max(2, math.ceil(wall.length / sample_step_m) + 1)
        s = np.linspace(0, wall.length, n)
        faces = []
        for off in (-wall.thickness / 2, wall.thickness / 2):
            samples = sample([wall.point(float(k), off) for k in s])
            faces.append(round(float(samples.mean()), 4))
        walls.append({'wall_id': int(wall.id), 'kind': wall.kind,
                      'length_m': round(wall.length, 4),
                      'face_return_proximity_fraction': faces,
                      'both_faces_covered_fraction_upper_bound': min(faces)})

    apertures = []
    for op in gt.layout.openings:
        wall = gt.layout.wall(op.wall_id)
        if wall.role == 'open_balcony':
            continue
        jambs = []
        for side in (-1, 1):
            jamb_s = float(op.s + side * op.width / 2)
            # Sample the adjacent *wall*, not the invisible middle of a door.
            # 20--80 mm outside its head/edge on either face.
            positions = [wall.point(jamb_s + side * v, off)
                         for v in (0.02, 0.04, 0.08)
                         for off in (-wall.thickness / 2, wall.thickness / 2)]
            supported = sample(positions)
            jambs.append(bool(np.any(supported)))
        apertures.append({'opening_id': int(op.id), 'kind': op.kind,
                          'wall_id': int(op.wall_id),
                          'jamb_return_proximity': jambs,
                          'both_jambs_near_returns': bool(all(jambs))})
    return {'description': 'proximity only; non-structural clutter may produce false support',
            'wall_count': len(walls), 'opening_count': len(apertures),
            'walls': walls, 'openings': apertures}
