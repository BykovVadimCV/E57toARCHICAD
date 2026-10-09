"""Vectorization via rectangle decomposition with semantic opening extraction.

Replaces skeletonization with rect_decompose (greedy rectangle decomposition)
for robust wall extraction. Produces clean axis-aligned segments with accurate
thickness from rectangle dimensions.

Key improvements over skeleton-based approach:
- Direct mask decomposition → no 1px skeleton fragility
- Accurate thickness from rectangle short-side dimensions
- Proper mask post-processing (morphological cleanup, enclosed-region refinement)
- Semantic door/window extraction from U-Net classes (not proximity matching)
- Collinear segment merging and endpoint snapping
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from PIL import Image

from . import rect_decompose as rd


@dataclass
class WallSegment:
    """Сегмент стены."""
    id: str
    points: list[list[float]]
    thickness: float
    rooms: list[str]
    is_outer: bool
    length: float


@dataclass
class Opening:
    """Проём в стене."""
    id: str
    opening_type: Literal["door", "window", "opening"]
    wall_id: str
    a: list[float]
    b: list[float]
    width: float
    sill: float | None = None
    head: float | None = None


@dataclass
class Room:
    """Помещение."""
    id: str
    name: str
    points: list[list[float]]
    area: float
    perimeter: float
    label_xy: tuple[float, float] = (0.0, 0.0)
    is_enclosed: bool = False


@dataclass
class VectorPlan:
    """Векторизованный план уровня."""
    level: str
    floor_z: float
    ceiling_z: float
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    frame: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Coordinate utilities
# ----------------------------------------------------------------------

def px_to_world(ji: np.ndarray, frame: dict) -> np.ndarray:
    """Convert (col, row) pixel coords to (x, y) world coords."""
    px = frame["pixel_m"]
    H = frame["height"]
    j, i = ji[:, 0], ji[:, 1]
    x = frame["origin_x"] + (j + 0.5) * px
    y = frame["origin_y"] + (H - 1 - i + 0.5) * px
    return np.c_[x, y]


def world_to_px(xy: np.ndarray, frame: dict) -> np.ndarray:
    """Convert (x, y) world coords to (col, row) pixel coords."""
    px = frame["pixel_m"]
    H = frame["height"]
    j = (xy[:, 0] - frame["origin_x"]) / px - 0.5
    i = H - 1 - ((xy[:, 1] - frame["origin_y"]) / px - 0.5)
    return np.c_[j, i]


# ----------------------------------------------------------------------
# Mask pre-processing
# ----------------------------------------------------------------------

def preprocess_wall_mask(
    wall_mask: np.ndarray,
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    min_area_px: int = 50,
    morph_close_kernel: int = 3,
) -> np.ndarray:
    """
    Clean wall mask before decomposition:
    1. Remove small noise blobs (< min_area_px)
    2. Carve out doors and windows
    3. Light morphological closing to bridge 1-2px gaps
    """
    # Remove noise
    wall_clean = wall_mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(wall_clean.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area_px:
            wall_clean[labels == i] = 0

    # Carve openings
    wall_clean &= ~door_mask
    wall_clean &= ~window_mask

    # Light morphological close to bridge tiny gaps
    if morph_close_kernel > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close_kernel, morph_close_kernel))
        wall_clean = cv2.morphologyEx(wall_clean.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return wall_clean.astype(bool)


def refine_wall_mask_by_enclosed_regions(
    original_img: np.ndarray,
    wall_mask: np.ndarray,
    coverage_threshold: float = 0.25,
    black_threshold: int = 60,
    white_threshold: int = 220,
) -> np.ndarray:
    """
    Enclosed-region refinement (borrowed from ReFloorBRUSNIKA):
    1. Find white enclosed regions via flood-fill
    2. For each region: if wall coverage >= threshold → fill; else erase walls
    3. Fixes hollow walls and removes floating fragments
    """
    if original_img is None:
        return wall_mask

    # Convert to grayscale
    if len(original_img.shape) == 3:
        gray = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = original_img

    # Build enclosed regions: near-white pixels only, everything else is barrier
    H, W = gray.shape
    is_white = (gray >= white_threshold).astype(np.uint8)

    # Flood from borders → mark exterior
    flooded = is_white.copy()
    for i in range(H):
        if flooded[i, 0] > 0:
            cv2.floodFill(flooded, None, (0, i), 0)
        if flooded[i, W - 1] > 0:
            cv2.floodFill(flooded, None, (W - 1, i), 0)
    for j in range(W):
        if flooded[0, j] > 0:
            cv2.floodFill(flooded, None, (j, 0), 0)
        if flooded[H - 1, j] > 0:
            cv2.floodFill(flooded, None, (j, H - 1), 0)

    # Remaining white = enclosed interior
    interior_mask = flooded > 0

    # Label enclosed regions
    n, labels = cv2.connectedComponents(interior_mask.astype(np.uint8), connectivity=8)

    refined = wall_mask.copy()
    wall_bin = wall_mask > 0

    for label_id in range(1, n):
        region_mask = labels == label_id
        region_area = int(np.sum(region_mask))
        if region_area == 0:
            continue

        wall_in_region = int(np.sum(wall_bin & region_mask))
        coverage = wall_in_region / region_area

        if coverage >= coverage_threshold:
            refined[region_mask] = 255
        else:
            refined[region_mask] = 0

    return refined


# ----------------------------------------------------------------------
# Rectangle decomposition → wall segments
# ----------------------------------------------------------------------

def rect_to_wall_segment(
    rect: rd.Rect,
    idx: int,
    px_to_m: float,
) -> WallSegment:
    """Convert decomposed Rect to WallSegment with proper thickness."""
    # Thickness = short side of rectangle
    thickness_px = min(rect.w, rect.h)
    thickness_m = round(thickness_px * px_to_m, 3)

    # Centerline endpoints
    rad = math.radians(rect.angle)
    cos, sin = math.cos(rad), math.sin(rad)
    half_len = max(rect.w, rect.h) / 2.0
    dx, dy = half_len * cos, half_len * sin

    if rect.w < rect.h:  # vertical dominant → swap
        dx, dy = -half_len * sin, half_len * cos

    p1 = (rect.cx - dx, rect.cy - dy)
    p2 = (rect.cx + dx, rect.cy + dy)

    length_px = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    length_m = round(length_px * px_to_m, 2)

    return WallSegment(
        id=f"wall_{idx:03d}",
        points=[list(p1), list(p2)],
        thickness=thickness_m,
        rooms=[],
        is_outer=False,
        length=length_m,
    )


def merge_collinear_walls(
    walls: list[WallSegment],
    angle_tol: float = 5.0,
    dist_tol_m: float = 0.15,
    gap_tol_m: float = 0.20,
) -> list[WallSegment]:
    """
    Merge nearby collinear wall segments.

    Two segments merge if:
    - Angle difference < angle_tol
    - Perpendicular distance < dist_tol_m
    - Axial gap < gap_tol_m
    """
    if len(walls) <= 1:
        return walls

    def angle_of(w: WallSegment) -> float:
        p1, p2 = w.points[0], w.points[1]
        return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])) % 180

    def perp_dist(w1: WallSegment, w2: WallSegment) -> float:
        p1, p2 = w1.points[0], w1.points[1]
        q = w2.points[0]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        L = math.hypot(dx, dy)
        if L < 1e-9:
            return float('inf')
        return abs(dy * (q[0] - p1[0]) - dx * (q[1] - p1[1])) / L

    def axial_gap(w1: WallSegment, w2: WallSegment) -> float:
        # Project w2 endpoints onto w1 axis
        p1, p2 = np.array(w1.points[0]), np.array(w1.points[1])
        axis = p2 - p1
        L = np.linalg.norm(axis)
        if L < 1e-9:
            return float('inf')
        axis = axis / L

        def proj(pt):
            return np.dot(np.array(pt) - p1, axis)

        w1_min, w1_max = 0.0, L
        w2_min = min(proj(w2.points[0]), proj(w2.points[1]))
        w2_max = max(proj(w2.points[0]), proj(w2.points[1]))

        if w2_max < w1_min:
            return w1_min - w2_max
        if w2_min > w1_max:
            return w2_min - w1_max
        return 0.0

    merged = []
    used = set()

    for i, w1 in enumerate(walls):
        if i in used:
            continue

        group = [w1]
        used.add(i)

        for j, w2 in enumerate(walls):
            if j <= i or j in used:
                continue

            a1, a2 = angle_of(w1), angle_of(w2)
            angle_diff = min(abs(a1 - a2), 180 - abs(a1 - a2))

            if angle_diff > angle_tol:
                continue
            if perp_dist(w1, w2) > dist_tol_m:
                continue
            if axial_gap(w1, w2) > gap_tol_m:
                continue

            group.append(w2)
            used.add(j)

        if len(group) == 1:
            merged.append(w1)
        else:
            # Merge group: collect all points, fit line, extend to span
            all_pts = []
            for w in group:
                all_pts.extend(w.points)
            all_pts = np.array(all_pts)

            mean = all_pts.mean(axis=0)
            cov = np.cov(all_pts.T)
            _, vecs = np.linalg.eigh(cov)
            axis = vecs[:, 1]

            proj = (all_pts - mean) @ axis
            t_min, t_max = proj.min(), proj.max()

            p1 = mean + t_min * axis
            p2 = mean + t_max * axis

            avg_thickness = sum(w.thickness for w in group) / len(group)
            length = float(np.linalg.norm(p2 - p1))

            merged.append(WallSegment(
                id=group[0].id,
                points=[p1.tolist(), p2.tolist()],
                thickness=round(avg_thickness, 3),
                rooms=[],
                is_outer=False,
                length=round(length, 2),
            ))

    return merged


def snap_wall_endpoints(
    walls: list[WallSegment],
    snap_threshold_m: float = 0.10,
) -> list[WallSegment]:
    """
    Snap endpoints of walls that are close but not touching.
    Helps close T-junctions and corners.
    """
    if len(walls) <= 1:
        return walls

    endpoints = []
    for i, w in enumerate(walls):
        for end_idx in (0, -1):
            endpoints.append((i, end_idx, np.array(w.points[end_idx])))

    # For each endpoint, find nearest other endpoint
    for i, end_i, pt_i in endpoints:
        best_dist = snap_threshold_m
        best_j, best_end_j = None, None

        for j, end_j, pt_j in endpoints:
            if i == j:
                continue
            dist = float(np.linalg.norm(pt_i - pt_j))
            if dist < best_dist:
                best_dist = dist
                best_j, best_end_j = j, end_j

        if best_j is not None:
            # Snap to midpoint
            pt_j = np.array(walls[best_j].points[best_end_j])
            mid = (pt_i + pt_j) / 2.0
            walls[i].points[end_i] = mid.tolist()
            walls[best_j].points[best_end_j] = mid.tolist()

    return walls


# ----------------------------------------------------------------------
# Opening extraction from semantic classes
# ----------------------------------------------------------------------

def extract_openings_from_semantic(
    door_mask: np.ndarray,
    window_mask: np.ndarray,
    heights: np.ndarray,
    walls: list[WallSegment],
    frame: dict,
    ceiling_z: float,
    min_width_m: float = 0.40,
    max_wall_dist_m: float = 0.50,
) -> list[Opening]:
    """
    Extract openings from semantic door/window masks:
    1. Find connected components in each mask
    2. Compute bbox and associate with nearest wall
    3. Extract sill/head from heights tensor
    """
    px_m = frame["pixel_m"]
    openings = []

    def process_mask(mask: np.ndarray, opening_type: str):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )

        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 20:  # skip noise
                continue

            x1 = stats[i, cv2.CC_STAT_LEFT]
            y1 = stats[i, cv2.CC_STAT_TOP]
            x2 = x1 + stats[i, cv2.CC_STAT_WIDTH]
            y2 = y1 + stats[i, cv2.CC_STAT_HEIGHT]

            # Bbox in world coords
            corners_px = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            corners_world = px_to_world(corners_px, frame)

            # Width = min side of bbox
            w_px = min(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            width_m = w_px * px_m

            if width_m < min_width_m:
                continue

            # Centroid in world coords
            cx, cy = centroids[i]
            centroid_world = px_to_world(np.array([[cx, cy]]), frame)[0]

            # Find nearest wall
            best_wall = None
            best_dist = float('inf')

            for wall in walls:
                p1 = np.array(wall.points[0])
                p2 = np.array(wall.points[1])
                # Distance from centroid to wall centerline
                v = p2 - p1
                L = np.linalg.norm(v)
                if L < 1e-9:
                    continue
                t = np.clip(np.dot(centroid_world - p1, v) / (L * L), 0, 1)
                closest = p1 + t * v
                dist = float(np.linalg.norm(centroid_world - closest))

                if dist < best_dist:
                    best_dist = dist
                    best_wall = wall

            if best_wall is None or best_dist > max_wall_dist_m:
                continue

            # Extract sill/head from heights tensor
            blob = labels == i
            h_blob = heights[:, blob]
            if h_blob.shape[1] == 0:
                sill, head = None, None
            else:
                sill = float(np.median(h_blob[0]) * ceiling_z) if opening_type == "window" else 0.0
                head = float(np.median(h_blob[1]) * ceiling_z)

            # Opening endpoints: project corners onto wall axis
            p1 = np.array(best_wall.points[0])
            p2 = np.array(best_wall.points[1])
            axis = p2 - p1
            L = np.linalg.norm(axis)
            axis = axis / L

            proj = [(pt - p1) @ axis for pt in corners_world]
            t_min, t_max = min(proj), max(proj)

            a = p1 + t_min * axis
            b = p1 + t_max * axis

            openings.append(Opening(
                id=f"{opening_type}_{len(openings):03d}",
                opening_type=opening_type,
                wall_id=best_wall.id,
                a=a.tolist(),
                b=b.tolist(),
                width=round(width_m, 2),
                sill=round(sill, 2) if sill is not None else None,
                head=round(head, 2) if head is not None else None,
            ))

    process_mask(door_mask, "door")
    process_mask(window_mask, "window")

    return openings


# ----------------------------------------------------------------------
# Room extraction
# ----------------------------------------------------------------------

def extract_rooms(
    sem: np.ndarray,
    wall_mask: np.ndarray,
    frame: dict,
    min_area_m2: float = 1.0,
) -> list[Room]:
    """
    Extract rooms from interior (class 1) connected components.
    """
    px_m = frame["pixel_m"]
    interior = (sem == 1).astype(np.uint8)

    # Remove wall pixels from interior
    interior &= ~wall_mask

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(interior, connectivity=4)

    rooms = []
    for i in range(1, n):
        area_px = stats[i, cv2.CC_STAT_AREA]
        area_m2 = area_px * px_m * px_m

        if area_m2 < min_area_m2:
            continue

        # Extract contour
        room_mask = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(room_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        contour = contour.squeeze()

        if contour.ndim != 2 or len(contour) < 3:
            continue

        # Convert to world coords
        contour_world = px_to_world(contour, frame)

        # Simplify
        perimeter = float(cv2.arcLength(contour, True)) * px_m
        simplified = cv2.approxPolyDP(contour_world.astype(np.float32), 0.05, True)
        simplified = simplified.squeeze()

        if simplified.ndim != 2:
            simplified = contour_world

        # Label position
        cx, cy = centroids[i]
        label_world = px_to_world(np.array([[cx, cy]]), frame)[0]

        rooms.append(Room(
            id=f"room_{i:03d}",
            name=f"Room {i}",
            points=simplified.tolist(),
            area=round(area_m2, 2),
            perimeter=round(perimeter, 2),
            label_xy=(round(label_world[0], 2), round(label_world[1], 2)),
            is_enclosed=True,
        ))

    return rooms


# ----------------------------------------------------------------------
# Main vectorization
# ----------------------------------------------------------------------

def vectorize(pred_dir: Path, log=print) -> VectorPlan:
    """
    Vectorize neural network predictions into a structured floor plan.

    Reads:
        - sem.png: semantic segmentation (class 0-7)
        - dist.npy: distance to wall face (normalized [0,1])
        - heights.npy: sill/head heights (normalized [0,1])
        - scene.json: frame metadata (origin, pixel_m, ceiling_z)

    Classes (simscan/netinput.py CLASSES):
        0: outside, 1: interior, 2: wall, 3: door, 4: window, 5: opening, 6: clutter, 7: void
    """
    pred_dir = Path(pred_dir)

    # Load inputs
    sem = np.array(Image.open(pred_dir / "sem.png"))
    dist = np.load(pred_dir / "dist.npy").astype(np.float32)
    heights = np.load(pred_dir / "heights.npy").astype(np.float32)
    scene = json.loads((pred_dir / "scene.json").read_text())

    frame = scene["frame"]
    px_m = frame["pixel_m"]
    ceiling_z = scene.get("ceiling_z", 3.0)
    level = scene.get("level", "L1")
    floor_z = scene.get("floor_z", 0.0)

    H, W = sem.shape
    log(f"Vectorizing {level}: {H}x{W} px @ {px_m*100:.1f} cm/px")

    # Extract masks
    wall_mask = (sem == 2)
    door_mask = (sem == 3)
    window_mask = (sem == 4)

    log(f"  Wall pixels: {int(wall_mask.sum())} ({100*wall_mask.mean():.2f}%)")
    log(f"  Door pixels: {int(door_mask.sum())}")
    log(f"  Window pixels: {int(window_mask.sum())}")

    # Preprocess wall mask
    wall_clean = preprocess_wall_mask(
        wall_mask.astype(np.uint8) * 255,
        door_mask.astype(np.uint8) * 255,
        window_mask.astype(np.uint8) * 255,
        min_area_px=50,
        morph_close_kernel=3,
    )

    log(f"  Wall after preprocessing: {int(wall_clean.sum())} px")

    # Rect decomposition
    log("  Running rect decomposition...")
    rects = rd.decompose(
        wall_clean,
        angle_steps=12,
        rect_penalty=0.0,
        max_grid_dim=200,
        coverage_stop=0.93,
        bleed_weight=15.0,
        initial_bleed_weight=15.0,
        bleed_decay=0.0,
        axis_only=True,
        max_overlap=1.0,
        max_spill_fraction=0.15,
        refine=True,
        verbose=False,
        thin_rescue=True,
        thin_rescue_max_thickness=20.0,
        thin_rescue_min_len=25.0,
        max_rects=400,
    )

    log(f"  Decomposed into {len(rects)} rectangles")

    # Convert rects to wall segments
    walls = [rect_to_wall_segment(r, i, px_m) for i, r in enumerate(rects)]

    # Filter short walls
    walls = [w for w in walls if w.length >= 0.30]
    log(f"  Walls after length filter (≥0.30m): {len(walls)}")

    # Merge collinear segments
    walls = merge_collinear_walls(walls, angle_tol=5.0, dist_tol_m=0.15, gap_tol_m=0.20)
    log(f"  Walls after merging: {len(walls)}")

    # Snap endpoints
    walls = snap_wall_endpoints(walls, snap_threshold_m=0.10)

    # Transform to world coords
    for wall in walls:
        pts_px = np.array(wall.points)
        pts_world = px_to_world(pts_px, frame)
        wall.points = [[round(x, 2), round(y, 2)] for x, y in pts_world]
        wall.length = round(float(np.linalg.norm(pts_world[1] - pts_world[0])), 2)

    log(f"  Final walls: {len(walls)}")

    # Extract openings
    log("  Extracting openings...")
    openings = extract_openings_from_semantic(
        door_mask.astype(np.uint8),
        window_mask.astype(np.uint8),
        heights,
        walls,
        frame,
        ceiling_z,
        min_width_m=0.40,
        max_wall_dist_m=0.50,
    )

    log(f"  Openings: {len(openings)} ({sum(1 for o in openings if o.opening_type=='door')} doors, "
        f"{sum(1 for o in openings if o.opening_type=='window')} windows)")

    # Extract rooms
    log("  Extracting rooms...")
    rooms = extract_rooms(sem, wall_mask.astype(np.uint8), frame, min_area_m2=1.0)
    log(f"  Rooms: {len(rooms)}")

    return VectorPlan(
        level=level,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        walls=walls,
        openings=openings,
        rooms=rooms,
        frame=frame,
        metadata={
            "method": "rect_decompose",
            "n_rects": len(rects),
            "wall_px": int(wall_mask.sum()),
            "wall_clean_px": int(wall_clean.sum()),
        },
    )


# ----------------------------------------------------------------------
# Save utilities
# ----------------------------------------------------------------------

def save_json(plan: VectorPlan, path: Path):
    """Save plan as JSON."""
    data = asdict(plan)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_svg(plan: VectorPlan, path: Path):
    """Save plan as SVG."""
    from xml.etree import ElementTree as ET

    # Compute bounds
    all_pts = []
    for w in plan.walls:
        all_pts.extend(w.points)
    for r in plan.rooms:
        all_pts.extend(r.points)

    if not all_pts:
        path.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
        return

    all_pts = np.array(all_pts)
    x_min, y_min = all_pts.min(axis=0)
    x_max, y_max = all_pts.max(axis=0)

    margin = 1.0
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin

    width = x_max - x_min
    height = y_max - y_min

    svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg",
                     width=str(int(width * 100)), height=str(int(height * 100)),
                     viewBox=f"{x_min} {y_min} {width} {height}")

    # Flip Y for SVG (Y grows down)
    g = ET.SubElement(svg, "g", transform=f"scale(1,-1) translate(0,{-(y_min+y_max)})")

    # Draw rooms
    for room in plan.rooms:
        pts_str = " ".join(f"{x},{y}" for x, y in room.points)
        ET.SubElement(g, "polygon", points=pts_str, fill="#f0f0f0", stroke="#ccc", **{"stroke-width": "0.01"})

    # Draw walls
    for wall in plan.walls:
        x1, y1 = wall.points[0]
        x2, y2 = wall.points[1]
        ET.SubElement(g, "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                      stroke="#333", **{"stroke-width": str(wall.thickness)})

    # Draw openings
    for opening in plan.openings:
        x1, y1 = opening.a
        x2, y2 = opening.b
        color = "#e74c3c" if opening.opening_type == "door" else "#3498db"
        ET.SubElement(g, "line", x1=str(x1), y1=str(y1), x2=str(x2), y2=str(y2),
                      stroke=color, **{"stroke-width": "0.05"})

    path.write_text(ET.tostring(svg, encoding="unicode"), encoding="utf-8")


def save_png(plan: VectorPlan, path: Path, dpi: int = 100):
    """Save plan as PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('#f5f5f5')

    # Draw rooms
    for room in plan.rooms:
        poly = MplPolygon(room.points, closed=True, facecolor='#f0f0f0', edgecolor='#ccc', linewidth=0.5)
        ax.add_patch(poly)

    # Draw walls
    for wall in plan.walls:
        p1, p2 = wall.points
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=wall.thickness * 100, solid_capstyle='round')

    # Draw openings
    for opening in plan.openings:
        a, b = opening.a, opening.b
        color = '#e74c3c' if opening.opening_type == "door" else '#3498db'
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=2, solid_capstyle='round')

    # Set bounds
    all_pts = [p for w in plan.walls for p in w.points]
    if all_pts:
        all_pts = np.array(all_pts)
        m = 1.0
        ax.set_xlim(all_pts[:, 0].min() - m, all_pts[:, 0].max() + m)
        ax.set_ylim(all_pts[:, 1].min() - m, all_pts[:, 1].max() + m)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close()


def main_vectorize(pred_dir, out_dir, log=print):
    """Vectorize one level."""
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = vectorize(pred_dir, log=log)

    save_json(plan, out_dir / "plan.json")
    log(f"Saved: {out_dir / 'plan.json'}")

    save_svg(plan, out_dir / "plan.svg")
    log(f"Saved: {out_dir / 'plan.svg'}")

    save_png(plan, out_dir / "plan.png")
    log(f"Saved: {out_dir / 'plan.png'}")

    return plan
