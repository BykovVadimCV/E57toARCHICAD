"""Smart vectorizer - production version with robust connectivity handling."""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Literal
from PIL import Image

@dataclass
class WallSegment:
    id: str
    points: list[list[float]]
    thickness: float  # Simplified - uniform for now
    rooms: list[str]
    is_outer: bool
    length: float

@dataclass
class Opening:
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
    id: str
    name: str
    points: list[list[float]]
    area: float
    perimeter: float
    label_xy: tuple[float, float] = (0.0, 0.0)
    is_enclosed: bool = False

@dataclass
class VectorPlan:
    level: int
    floor_z: float
    ceiling_z: float
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    frame: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

def px_to_world(ji: np.ndarray, frame: dict) -> np.ndarray:
    ji = np.atleast_2d(np.asarray(ji, float))
    px, H = frame["pixel_m"], frame["height"]
    x = frame["origin_x"] + (ji[:, 0] + 0.5) * px
    y = frame["origin_y"] + (H - 1 - ji[:, 1] + 0.5) * px
    return np.c_[x, y]

def world_to_px(xy: np.ndarray, frame: dict) -> np.ndarray:
    xy = np.atleast_2d(np.asarray(xy, float))
    px, H = frame["pixel_m"], frame["height"]
    j = (xy[:, 0] - frame["origin_x"]) / px - 0.5
    i = H - 1 - ((xy[:, 1] - frame["origin_y"]) / px - 0.5)
    return np.c_[j, i]

# Copy skeletonization and basic functions from vectorize_fixed.py
from simscan.vectorize import (
    thin_walls, skeleton_to_segments, cleanup_wall_mask,
    simplify_polyline as simplify_polyline_rdp
)

def snap_to_axis(segment: np.ndarray, tolerance_deg: float = 5.0) -> np.ndarray:
    """Snap nearly-horizontal/vertical segments to true 0°/90°."""
    if len(segment) < 2:
        return segment
    
    start, end = segment[0], segment[-1]
    v = end - start
    angle = np.arctan2(v[1], v[0])
    length = np.linalg.norm(v)
    
    # Check if close to 0°, 90°, 180°, or 270°
    angles = [0, np.pi/2, np.pi, -np.pi/2]
    closest = min(angles, key=lambda a: abs(angle - a))
    
    if abs(angle - closest) < np.radians(tolerance_deg):
        # Snap to axis
        new_v = length * np.array([np.cos(closest), np.sin(closest)])
        new_end = start + new_v
        
        if len(segment) == 2:
            return np.array([start, new_end])
        else:
            # Interpolate intermediate points
            t = np.linspace(0, 1, len(segment))
            return start[None, :] + t[:, None] * new_v[None, :]
    
    return segment

def estimate_thickness(seg_px: np.ndarray, dist: np.ndarray) -> float:
    """Improved thickness estimation."""
    seg_px_int = seg_px.astype(int)
    valid = ((seg_px_int[:, 1] >= 0) & (seg_px_int[:, 1] < dist.shape[1]) &
             (seg_px_int[:, 0] >= 0) & (seg_px_int[:, 0] < dist.shape[2]))
    
    if valid.sum() == 0:
        return 0.20
    
    vals = dist[0, seg_px_int[valid, 1], seg_px_int[valid, 0]]
    valid_vals = vals[vals < 0.9]
    
    if len(valid_vals) < 3:
        return 0.20
    
    thick = float(np.median(valid_vals) * 2.0)
    return np.clip(thick, 0.08, 0.30)

def merge_nearby_segments(segments: list[np.ndarray], max_gap: float = 0.20, 
                         angle_tol: float = 10.0) -> list[np.ndarray]:
    """Merge segments that are close and roughly aligned."""
    if not segments:
        return []
    
    merged = []
    used = set()
    
    for i, seg1 in enumerate(segments):
        if i in used:
            continue
        
        current = [seg1]
        used.add(i)
        
        # Try to extend from both ends
        changed = True
        while changed:
            changed = False
            
            # Try extending from end
            end_pt = current[-1][-1]
            for j, seg2 in enumerate(segments):
                if j in used:
                    continue
                
                # Distance to start of seg2
                dist = np.linalg.norm(end_pt - seg2[0])
                if dist > max_gap:
                    continue
                
                # Check angle compatibility
                v1 = current[-1][-1] - current[-1][0]
                v2 = seg2[-1] - seg2[0]
                v1_norm, v2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
                
                if v1_norm > 1e-6 and v2_norm > 1e-6:
                    cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
                    if abs(cos_angle) > np.cos(np.radians(angle_tol)):
                        current.append(seg2)
                        used.add(j)
                        changed = True
                        break
        
        # Combine all segments
        if len(current) == 1:
            merged.append(current[0])
        else:
            all_pts = np.vstack(current)
            merged.append(all_pts)
    
    return merged

def extract_walls_robust(sem: np.ndarray, dist: np.ndarray, frame: dict,
                        min_length_m: float = 0.8) -> tuple[list[WallSegment], dict]:
    """Robust wall extraction."""
    
    metadata = {"raw": 0, "merged": 0, "final": 0}
    
    wall_mask = (sem == 2).astype(np.uint8)
    if wall_mask.sum() < 10:
        return [], metadata
    
    wall_mask = cleanup_wall_mask(wall_mask, min_area_px=50)
    skel = thin_walls(wall_mask)
    segments_px = skeleton_to_segments(skel)
    metadata["raw"] = len(segments_px)
    
    if not segments_px:
        return [], metadata
    
    # Convert to world
    segments_world = [px_to_world(seg, frame) for seg in segments_px]
    
    # Filter by length (permissive first pass)
    def seg_len(s):
        return float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))
    
    segments_world = [s for s, sp in zip(segments_world, segments_px) 
                     if seg_len(s) >= 0.3]
    segments_px = [sp for s, sp in zip(segments_world, segments_px) 
                  if seg_len(s) >= 0.3]
    
    # Merge nearby segments
    merged = merge_nearby_segments(segments_world, max_gap=0.25, angle_tol=10.0)
    metadata["merged"] = len(merged)
    
    # Axis alignment
    aligned = [snap_to_axis(seg, tolerance_deg=5.0) for seg in merged]
    
    # Final filtering and simplification
    walls = []
    for idx, (seg_world, seg_px) in enumerate(zip(aligned, segments_px)):
        length = seg_len(seg_world)
        if length < min_length_m:
            continue
        
        # Simplify
        seg_simp = simplify_polyline_rdp(seg_world, tol=0.05)
        if len(seg_simp) < 2:
            continue
        
        thickness = estimate_thickness(seg_px, dist)
        
        is_outer = any(int(y) < 5 or int(y) >= sem.shape[0] - 5 or
                      int(x) < 5 or int(x) >= sem.shape[1] - 5
                      for x, y in seg_px[:min(5, len(seg_px))])
        
        walls.append(WallSegment(
            id=f"wall_{idx:03d}",
            points=seg_simp.tolist(),
            thickness=thickness,
            rooms=[],
            is_outer=is_outer,
            length=seg_len(seg_simp)
        ))
    
    metadata["final"] = len(walls)
    return walls, metadata

def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labeled = np.zeros_like(mask, dtype=int)
    current_label = 0
    def flood_fill(i, j, label):
        stack = [(i, j)]
        while stack:
            ci, cj = stack.pop()
            if ci < 0 or ci >= mask.shape[0] or cj < 0 or cj >= mask.shape[1]:
                continue
            if labeled[ci, cj] != 0 or mask[ci, cj] == 0:
                continue
            labeled[ci, cj] = label
            stack.extend([(ci-1, cj), (ci+1, cj), (ci, cj-1), (ci, cj+1)])
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] and labeled[i, j] == 0:
                current_label += 1
                flood_fill(i, j, current_label)
    return labeled, current_label

def extract_rooms(sem: np.ndarray, frame: dict, min_area_m2: float = 1.0) -> list[Room]:
    interior = (sem == 1).astype(np.uint8)
    if interior.sum() < 10:
        return []
    
    try:
        from scipy import ndimage
        labeled, n = ndimage.label(interior)
    except:
        labeled, n = _label_components(interior)
    
    rooms = []
    px_area = frame["pixel_m"] ** 2
    
    for lid in range(1, n + 1):
        mask = (labeled == lid)
        area = float(mask.sum() * px_area)
        
        if area < min_area_m2:
            continue
        
        coords_y, coords_x = np.where(mask)
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(np.c_[coords_x, coords_y])
            hull_px = np.c_[coords_x, coords_y][hull.vertices]
        except:
            hull_px = np.array([[coords_x.min(), coords_y.min()],
                               [coords_x.max(), coords_y.min()],
                               [coords_x.max(), coords_y.max()],
                               [coords_x.min(), coords_y.max()]])
        
        contour = px_to_world(hull_px, frame)
        perim = float(np.sum(np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)))
        centroid = px_to_world(np.array([[coords_x.mean(), coords_y.mean()]]), frame)[0]
        
        rooms.append(Room(
            id=f"room_{lid:03d}",
            name=f"Room {lid}",
            points=contour.tolist(),
            area=area,
            perimeter=perim,
            label_xy=tuple(centroid.tolist()),
            is_enclosed=True
        ))
    
    return rooms

def extract_openings(sem: np.ndarray, heights: np.ndarray, walls: list[WallSegment],
                    frame: dict, ceiling_z: float) -> list[Opening]:
    openings = []
    for cls, typ in [(3, "door"), (4, "window"), (5, "opening")]:
        mask = (sem == cls)
        if mask.sum() < 5:
            continue
        try:
            from scipy import ndimage
            labeled, n = ndimage.label(mask.astype(np.uint8))
        except:
            labeled, n = _label_components(mask.astype(np.uint8))
        
        for lid in range(1, n + 1):
            if (labeled == lid).sum() < 5:
                continue
            coords_y, coords_x = np.where(labeled == lid)
            coords_world = px_to_world(np.c_[coords_x, coords_y], frame)
            center = coords_world.mean(axis=0)
            
            closest = None
            min_dist = 1.0
            for wall in walls:
                for k in range(len(wall.points) - 1):
                    a, b = np.array(wall.points[k]), np.array(wall.points[k+1])
                    v, w = b - a, center - a
                    c1, c2 = np.dot(w, v), np.dot(v, v)
                    if c2 < 1e-9:
                        d = np.linalg.norm(w)
                    elif c1 <= 0:
                        d = np.linalg.norm(w)
                    elif c1 >= c2:
                        d = np.linalg.norm(center - b)
                    else:
                        d = np.linalg.norm(center - (a + (c1/c2) * v))
                    if d < min_dist:
                        min_dist, closest = d, wall
            
            if closest:
                x_min, y_min = coords_world.min(axis=0)
                x_max, y_max = coords_world.max(axis=0)
                openings.append(Opening(
                    id=f"{typ}_{len([o for o in openings if o.opening_type == typ]):03d}",
                    opening_type=typ,
                    wall_id=closest.id,
                    a=[float(x_min), float(y_min)],
                    b=[float(x_max), float(y_max)],
                    width=float(max(x_max - x_min, y_max - y_min)),
                    sill=0.9 if typ == "window" else None,
                    head=2.1
                ))
    return openings

def vectorize(pred_dir: Path, log=print) -> VectorPlan:
    pred_dir = Path(pred_dir)
    sem = np.array(Image.open(pred_dir / "sem.png"))
    dist = np.load(pred_dir / "dist.npy")
    orient = np.load(pred_dir / "orient.npy")
    heights = np.load(pred_dir / "heights.npy")
    scene = json.loads((pred_dir / "scene.json").read_text(encoding="utf-8"))
    frame = scene["frame"]
    
    log(f"Умная векторизация {pred_dir.name}...")
    walls, meta = extract_walls_robust(sem, dist, frame, min_length_m=0.8)
    log(f"  Стен: {len(walls)} (сырых={meta['raw']}, после слияния={meta['merged']})")
    
    rooms = extract_rooms(sem, frame, min_area_m2=1.0)
    log(f"  Помещений: {len(rooms)}")
    
    openings = extract_openings(sem, heights, walls, frame, scene.get("ceiling_z", 3.0))
    log(f"  Проёмов: {len(openings)}")
    
    # Link rooms to walls
    for room in rooms:
        for wall in walls:
            for k in range(len(wall.points) - 1):
                mid = (np.array(wall.points[k]) + np.array(wall.points[k+1])) / 2
                if np.linalg.norm(np.array(room.label_xy) - mid) < 5.0:
                    if room.id not in wall.rooms:
                        wall.rooms.append(room.id)
    
    return VectorPlan(
        level=scene.get("level", 1),
        floor_z=scene.get("floor_z", 0.0),
        ceiling_z=scene.get("ceiling_z", 3.0),
        walls=walls, openings=openings, rooms=rooms,
        frame=frame, metadata=meta
    )

def save_json(plan: VectorPlan, path: Path):
    path.write_text(json.dumps(asdict(plan), indent=2, ensure_ascii=False), encoding="utf-8")

def save_svg(plan: VectorPlan, path: Path):
    from xml.etree import ElementTree as ET
    all_pts = [p for w in plan.walls for p in w.points] + [p for r in plan.rooms for p in r.points]
    if not all_pts:
        return
    all_pts = np.array(all_pts)
    x_min, y_min = all_pts.min(axis=0) - 1.0
    x_max, y_max = all_pts.max(axis=0) + 1.0
    w, h = x_max - x_min, y_max - y_min
    
    svg = ET.Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                             "viewBox": f"{x_min} {-y_max} {w} {h}",
                             "width": f"{w*100:.0f}", "height": f"{h*100:.0f}"})
    ET.SubElement(svg, "rect", {"x": str(x_min), "y": str(-y_max),
                                "width": str(w), "height": str(h), "fill": "#f5f5f5"})
    
    for room in plan.rooms:
        ET.SubElement(svg, "polygon", {
            "points": " ".join(f"{x},{-y}" for x, y in room.points),
            "fill": "#e8f4f8", "stroke": "none"})
    
    for wall in plan.walls:
        pts = np.array(wall.points)
        if len(pts) >= 2:
            ET.SubElement(svg, "polyline", {
                "points": " ".join(f"{x},{-y}" for x, y in pts),
                "fill": "none",
                "stroke": "#2a2a2a" if wall.is_outer else "#4a4a4a",
                "stroke-width": str(wall.thickness * 100),
                "stroke-linecap": "square"})
    
    for op in plan.openings:
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[op.opening_type]
        ET.SubElement(svg, "line", {
            "x1": str(op.a[0]), "y1": str(-op.a[1]),
            "x2": str(op.b[0]), "y2": str(-op.b[1]),
            "stroke": color, "stroke-width": "0.08", "stroke-linecap": "round"})
    
    for room in plan.rooms:
        text = ET.SubElement(svg, "text", {
            "x": str(room.label_xy[0]), "y": str(-room.label_xy[1]),
            "text-anchor": "middle", "font-size": "0.3",
            "font-family": "sans-serif", "fill": "#333"})
        text.text = f"{room.area:.1f} m²"
    
    ET.indent(ET.ElementTree(svg), space="  ")
    ET.ElementTree(svg).write(path, encoding="utf-8", xml_declaration=True)

def save_png(plan: VectorPlan, path: Path, dpi: int = 100):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')
    
    for room in plan.rooms:
        ax.add_patch(MplPolygon(room.points, facecolor='#e8f4f8', edgecolor='none'))
    
    for wall in plan.walls:
        pts = np.array(wall.points)
        if len(pts) >= 2:
            ax.plot(pts[:, 0], pts[:, 1],
                   color='#2a2a2a' if wall.is_outer else '#4a4a4a',
                   linewidth=wall.thickness * 100, solid_capstyle='projecting')
    
    for op in plan.openings:
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[op.opening_type]
        ax.plot([op.a[0], op.b[0]], [op.a[1], op.b[1]],
               color=color, linewidth=3, solid_capstyle='round')
    
    for room in plan.rooms:
        ax.text(room.label_xy[0], room.label_xy[1], f"{room.area:.1f} m²",
               ha='center', va='center', fontsize=8, color='#333')
    
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
    pred_dir, out_dir = Path(pred_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = vectorize(pred_dir, log=log)
    save_json(plan, out_dir / "plan.json")
    save_svg(plan, out_dir / "plan.svg")
    save_png(plan, out_dir / "plan.png")
    log(f"Сохранено в {out_dir}/")
    return plan
