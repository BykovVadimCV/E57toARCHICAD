"""Improved vectorization with noise filtering and realistic wall thickness.

Fixes:
- Aggressive minimum length filtering (walls >= 0.8m)
- Realistic thickness calculation and limits (0.08-0.30m)
- Pre-skeletonization morphological cleanup
- Segment merging for nearby collinear walls
- Better distance map interpretation
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


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


@dataclass
class VectorPlan:
    """Векторизованный план уровня."""
    level: int
    floor_z: float
    ceiling_z: float
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    frame: dict = field(default_factory=dict)


# ----------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------

def px_to_world(ji: np.ndarray, frame: dict) -> np.ndarray:
    """Пиксельные координаты (j, i) -> мировые (x, y)."""
    ji = np.atleast_2d(np.asarray(ji, float))
    px, H = frame["pixel_m"], frame["height"]
    x = frame["origin_x"] + (ji[:, 0] + 0.5) * px
    y = frame["origin_y"] + (H - 1 - ji[:, 1] + 0.5) * px
    return np.c_[x, y]


def world_to_px(xy: np.ndarray, frame: dict) -> np.ndarray:
    """Мировые координаты (x, y) -> пиксельные (j, i)."""
    xy = np.atleast_2d(np.asarray(xy, float))
    px, H = frame["pixel_m"], frame["height"]
    j = (xy[:, 0] - frame["origin_x"]) / px - 0.5
    i = H - 1 - ((xy[:, 1] - frame["origin_y"]) / px - 0.5)
    return np.c_[j, i]


def simplify_polyline(pts: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker упрощение полилинии."""
    if len(pts) <= 2:
        return pts

    start, end = pts[0], pts[-1]
    if len(pts) == 3:
        v1 = end - start
        v2 = start - pts[1]
        cross_mag = abs(v1[0] * v2[1] - v1[1] * v2[0])
        return pts if cross_mag / max(np.linalg.norm(v1), 1e-9) > tol else np.array([start, end])

    v = end - start
    norm_v = np.linalg.norm(v)
    if norm_v < 1e-9:
        return np.array([start, end])

    a = v[1]
    b = -v[0]
    c = v[0] * start[1] - v[1] * start[0]

    dist = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / norm_v
    idx_max = dist.argmax()

    if dist[idx_max] <= tol:
        return np.array([start, end])

    left = simplify_polyline(pts[:idx_max + 1], tol)
    right = simplify_polyline(pts[idx_max:], tol)
    return np.vstack([left[:-1], right])


# ----------------------------------------------------------------------
# Морфологическая очистка перед скелетонизацией
# ----------------------------------------------------------------------

def cleanup_wall_mask(mask: np.ndarray, min_area_px: int = 50) -> np.ndarray:
    """Удаление мелких шумовых компонент перед скелетонизацией."""
    try:
        import cv2
        # Удаление мелких дыр
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Удаление мелких компонент
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area_px:
                cleaned[labels == i] = 0
        return cleaned
    except ImportError:
        return mask


# ----------------------------------------------------------------------
# Скелетонизация (Zhang-Suen)
# ----------------------------------------------------------------------

def _thin_iter(img: np.ndarray, iter_num: int) -> tuple[np.ndarray, int]:
    """Одна итерация Zhang-Suen thinning."""
    marker = np.zeros_like(img, dtype=bool)
    H, W = img.shape

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if not img[i, j]:
                continue

            p = [img[i-1, j], img[i-1, j+1], img[i, j+1], img[i+1, j+1],
                 img[i+1, j], img[i+1, j-1], img[i, j-1], img[i-1, j-1]]

            n_black = sum(p)
            if not (2 <= n_black <= 6):
                continue

            transitions = sum(1 for k in range(8) if p[k] == 0 and p[(k + 1) % 8] == 1)
            if transitions != 1:
                continue

            if iter_num == 0:
                if p[0] * p[2] * p[4] != 0:
                    continue
                if p[2] * p[4] * p[6] != 0:
                    continue
            else:
                if p[0] * p[2] * p[6] != 0:
                    continue
                if p[0] * p[4] * p[6] != 0:
                    continue

            marker[i, j] = True

    img[marker] = 0
    return img, marker.sum()


def thin_walls(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Zhang-Suen скелетонизация бинарной маски стен."""
    skel = mask.astype(bool).copy()
    for _ in range(max_iter):
        skel, n1 = _thin_iter(skel, 0)
        skel, n2 = _thin_iter(skel, 1)
        if n1 + n2 == 0:
            break
    return skel.astype(np.uint8)


# ----------------------------------------------------------------------
# Граф скелета -> сегменты
# ----------------------------------------------------------------------

def _convolve_2d(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Simple 2D convolution."""
    kh, kw = kernel.shape
    h, w = img.shape
    result = np.zeros_like(img, dtype=float)

    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    for i in range(h):
        for j in range(w):
            window = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(window * kernel)

    return result


def skeleton_to_segments(skel: np.ndarray) -> list[np.ndarray]:
    """Скелет -> список полилиний."""
    try:
        from scipy import ndimage
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbors = ndimage.convolve(skel.astype(np.uint8), kernel, mode='constant')
    except ImportError:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
        neighbors = _convolve_2d(skel.astype(np.uint8), kernel).astype(np.uint8)

    nodes = (skel > 0) & ((neighbors != 2) | (neighbors == 0))

    visited = np.zeros_like(skel, dtype=bool)
    segments = []

    coords = np.argwhere(skel > 0)
    for i, j in coords:
        if visited[i, j] or not nodes[i, j]:
            continue

        for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ni, nj = i + di, j + dj
            if ni < 0 or ni >= skel.shape[0] or nj < 0 or nj >= skel.shape[1]:
                continue
            if not skel[ni, nj] or visited[ni, nj]:
                continue

            path = [(j, i)]
            ci, cj = ni, nj
            prev_i, prev_j = i, j

            while True:
                path.append((cj, ci))
                visited[ci, cj] = True

                if nodes[ci, cj]:
                    break

                found = False
                for di2, dj2 in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    ni2, nj2 = ci + di2, cj + dj2
                    if ni2 == prev_i and nj2 == prev_j:
                        continue
                    if ni2 < 0 or ni2 >= skel.shape[0] or nj2 < 0 or nj2 >= skel.shape[1]:
                        continue
                    if skel[ni2, nj2] and not visited[ni2, nj2]:
                        prev_i, prev_j = ci, cj
                        ci, cj = ni2, nj2
                        found = True
                        break

                if not found:
                    break

            if len(path) >= 2:
                segments.append(np.array(path, dtype=float))

    return segments


# ----------------------------------------------------------------------
# Улучшенная обработка толщины стен
# ----------------------------------------------------------------------

def estimate_wall_thickness(seg_px: np.ndarray, dist: np.ndarray, px: float) -> float:
    """Оценка толщины стены с учётом ограничений и реальных значений.

    dist - карта расстояний до границы стены (в метрах).
    Типичная толщина стен: несущие 0.20-0.30м, перегородки 0.08-0.15м.
    """
    seg_px_int = seg_px.astype(int)
    valid_idx = ((seg_px_int[:, 1] >= 0) & (seg_px_int[:, 1] < dist.shape[1]) &
                 (seg_px_int[:, 0] >= 0) & (seg_px_int[:, 0] < dist.shape[2]))

    if valid_idx.sum() == 0:
        return 0.20  # По умолчанию

    thickness_vals = dist[0, seg_px_int[valid_idx, 1], seg_px_int[valid_idx, 0]]

    # Фильтрация выбросов (удаление насыщенных значений)
    valid_thickness = thickness_vals[thickness_vals < 0.9]

    if len(valid_thickness) < 3:
        # Слишком мало значений или все насыщены - возврат к умолчанию
        return 0.20

    # Медианная толщина (половина расстояния до границы)
    # dist - это расстояние до края, полная толщина = 2 * dist
    median_half_thick = float(np.median(valid_thickness))
    full_thickness = median_half_thick * 2.0

    # Реалистичные пределы: 8см минимум (перегородка), 30см максимум (несущая)
    thickness = np.clip(full_thickness, 0.08, 0.30)

    return float(thickness)


# ----------------------------------------------------------------------
# Объединение коллинеарных сегментов
# ----------------------------------------------------------------------

def merge_collinear_segments(segments: list[np.ndarray], angle_tol: float = 5.0,
                             gap_tol: float = 0.05) -> list[np.ndarray]:
    """Объединение близких коллинеарных сегментов в один."""
    if not segments:
        return []

    merged = []
    used = set()

    for i, seg1 in enumerate(segments):
        if i in used:
            continue

        current = [seg1]
        used.add(i)

        # Направление первого сегмента
        v1 = seg1[-1] - seg1[0]
        len1 = np.linalg.norm(v1)
        if len1 < 1e-6:
            continue
        v1 = v1 / len1

        # Ищем продолжения
        changed = True
        while changed:
            changed = False
            for j, seg2 in enumerate(segments):
                if j in used:
                    continue

                v2 = seg2[-1] - seg2[0]
                len2 = np.linalg.norm(v2)
                if len2 < 1e-6:
                    continue
                v2 = v2 / len2

                # Проверка коллинеарности
                dot = abs(np.dot(v1, v2))
                if dot < np.cos(np.radians(angle_tol)):
                    continue

                # Проверка близости концов
                end_current = current[-1][-1]
                start_current = current[0][0]

                gaps = [
                    np.linalg.norm(end_current - seg2[0]),
                    np.linalg.norm(end_current - seg2[-1]),
                    np.linalg.norm(start_current - seg2[0]),
                    np.linalg.norm(start_current - seg2[-1])
                ]

                min_gap = min(gaps)
                if min_gap < gap_tol:
                    current.append(seg2)
                    used.add(j)
                    changed = True
                    break

        # Объединить все точки в один сегмент
        if len(current) == 1:
            merged.append(current[0])
        else:
            all_pts = np.vstack(current)
            merged.append(all_pts)

    return merged


# ----------------------------------------------------------------------
# Извлечение стен с фильтрацией
# ----------------------------------------------------------------------

def extract_walls(sem: np.ndarray, dist: np.ndarray, orient: np.ndarray, frame: dict,
                  min_length_m: float = 0.8, simplify_tol: float = 0.05) -> list[WallSegment]:
    """Извлечение стен с агрессивной фильтрацией шума."""
    # Маска стен (класс 2)
    wall_mask = (sem == 2).astype(np.uint8)

    if wall_mask.sum() < 10:
        return []

    # Предварительная очистка
    wall_mask = cleanup_wall_mask(wall_mask, min_area_px=100)

    # Скелет
    skel = thin_walls(wall_mask)

    # Сегменты
    segments_px = skeleton_to_segments(skel)

    # Объединение коллинеарных фрагментов
    segments_px = merge_collinear_segments(segments_px, angle_tol=5.0,
                                          gap_tol=0.1 / frame["pixel_m"])

    walls = []
    px = frame["pixel_m"]

    for idx, seg_px in enumerate(segments_px):
        if len(seg_px) < 2:
            continue

        # Конвертируем в метры
        seg_world = px_to_world(seg_px, frame)

        # Длина до упрощения
        length_raw = float(np.sum(np.linalg.norm(np.diff(seg_world, axis=0), axis=1)))

        # Фильтр: слишком короткие - шум
        if length_raw < min_length_m:
            continue

        # Упрощение (более агрессивное)
        seg_simplified = simplify_polyline(seg_world, simplify_tol)

        if len(seg_simplified) < 2:
            continue

        # Финальная длина
        length = float(np.sum(np.linalg.norm(np.diff(seg_simplified, axis=0), axis=1)))

        if length < min_length_m:
            continue

        # Толщина с улучшенной оценкой
        thickness = estimate_wall_thickness(seg_px, dist, px)

        # Наружная стена
        is_outer = False
        for x, y in seg_px[:min(5, len(seg_px))]:
            xi, yi = int(y), int(x)
            if xi < 5 or xi >= sem.shape[0] - 5 or yi < 5 or yi >= sem.shape[1] - 5:
                is_outer = True
                break

        wall = WallSegment(
            id=f"wall_{idx:03d}",
            points=seg_simplified.tolist(),
            thickness=thickness,
            rooms=[],
            is_outer=is_outer,
            length=length
        )
        walls.append(wall)

    return walls


# ----------------------------------------------------------------------
# Помещения
# ----------------------------------------------------------------------

def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Pure NumPy connected component labeling."""
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


def _center_of_mass(mask: np.ndarray) -> tuple[float, float]:
    """Pure NumPy center of mass."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return (0.0, 0.0)
    return tuple(coords.mean(axis=0))


def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Simple convex hull (Graham scan)."""
    if len(points) < 3:
        return points

    start = min(enumerate(points), key=lambda p: (p[1][1], p[1][0]))[0]
    hull_points = [points[start]]

    def angle(p):
        dx, dy = p - points[start]
        return np.arctan2(dy, dx)

    sorted_pts = sorted([p for i, p in enumerate(points) if i != start], key=angle)

    for pt in sorted_pts:
        while len(hull_points) >= 2:
            v1 = hull_points[-1] - hull_points[-2]
            v2 = pt - hull_points[-1]
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            if cross_product <= 0:
                hull_points.pop()
            else:
                break
        hull_points.append(pt)

    return np.array(hull_points)


def extract_rooms(sem: np.ndarray, frame: dict) -> list[Room]:
    """Извлечение помещений из семантической маски."""
    interior_mask = (sem == 1).astype(np.uint8)

    if interior_mask.sum() < 10:
        return []

    try:
        from scipy import ndimage
        labeled, n_labels = ndimage.label(interior_mask)
        use_scipy_com = True
    except ImportError:
        labeled, n_labels = _label_components(interior_mask)
        use_scipy_com = False

    rooms = []
    px_area = frame["pixel_m"] ** 2

    for label_id in range(1, n_labels + 1):
        room_mask = (labeled == label_id)
        area_px = room_mask.sum()

        if area_px < 25:
            continue

        area = float(area_px * px_area)

        coords_y, coords_x = np.where(room_mask)
        if len(coords_x) < 3:
            continue

        try:
            from scipy.spatial import ConvexHull
            points_px = np.c_[coords_x, coords_y]
            hull = ConvexHull(points_px)
            hull_px = points_px[hull.vertices]
        except ImportError:
            points_px = np.c_[coords_x, coords_y]
            hull_px = _convex_hull_2d(points_px)

        contour_world = px_to_world(hull_px, frame)

        perimeter = float(np.sum(np.linalg.norm(
            np.diff(np.vstack([contour_world, contour_world[:1]]), axis=0), axis=1)))

        if use_scipy_com:
            from scipy import ndimage
            yi, xi = ndimage.center_of_mass(room_mask)
        else:
            yi, xi = _center_of_mass(room_mask)
        centroid = px_to_world(np.array([[xi, yi]]), frame)[0]

        room = Room(
            id=f"room_{label_id:03d}",
            name=f"Room {label_id}",
            points=contour_world.tolist(),
            area=area,
            perimeter=perimeter,
            label_xy=tuple(centroid.tolist())
        )
        rooms.append(room)

    return rooms


# ----------------------------------------------------------------------
# Проёмы
# ----------------------------------------------------------------------

def extract_openings(sem: np.ndarray, heights: np.ndarray, walls: list[WallSegment],
                     frame: dict, ceiling_z: float) -> list[Opening]:
    """Извлечение проёмов (двери, окна) из семантической маски."""
    openings = []

    for opening_class, opening_type in [(3, "door"), (4, "window"), (5, "opening")]:
        mask = (sem == opening_class)
        if mask.sum() < 5:
            continue

        try:
            from scipy import ndimage
            labeled, n_labels = ndimage.label(mask.astype(np.uint8))
        except ImportError:
            labeled, n_labels = _label_components(mask.astype(np.uint8))

        for label_id in range(1, n_labels + 1):
            opening_mask = (labeled == label_id)
            if opening_mask.sum() < 5:
                continue

            coords_y, coords_x = np.where(opening_mask)
            coords_px = np.c_[coords_x, coords_y]
            coords_world = px_to_world(coords_px, frame)

            center = coords_world.mean(axis=0)

            min_dist = float('inf')
            closest_wall = None
            for wall in walls:
                wall_pts = np.array(wall.points)
                if len(wall_pts) < 2:
                    continue

                for k in range(len(wall_pts) - 1):
                    a, b = wall_pts[k], wall_pts[k + 1]
                    v = b - a
                    w = center - a
                    c1 = np.dot(w, v)
                    c2 = np.dot(v, v)

                    if c2 < 1e-9:
                        dist = np.linalg.norm(w)
                    elif c1 <= 0:
                        dist = np.linalg.norm(w)
                    elif c1 >= c2:
                        dist = np.linalg.norm(center - b)
                    else:
                        t = c1 / c2
                        proj = a + t * v
                        dist = np.linalg.norm(center - proj)

                    if dist < min_dist:
                        min_dist = dist
                        closest_wall = wall

            if closest_wall is None or min_dist > 1.0:
                continue

            x_min, y_min = coords_world.min(axis=0)
            x_max, y_max = coords_world.max(axis=0)
            width = max(x_max - x_min, y_max - y_min)

            h_vals = heights[:, coords_y, coords_x]
            if h_vals.size > 0:
                sill = float(np.median(h_vals[0]) * ceiling_z) if opening_type == "window" else None
                head = float(np.median(h_vals[1]) * ceiling_z)
            else:
                sill = 0.9 if opening_type == "window" else None
                head = 2.1

            opening = Opening(
                id=f"{opening_type}_{len([o for o in openings if o.opening_type == opening_type]):03d}",
                opening_type=opening_type,
                wall_id=closest_wall.id,
                a=[float(x_min), float(y_min)],
                b=[float(x_max), float(y_max)],
                width=float(width),
                sill=sill,
                head=head
            )
            openings.append(opening)

    return openings


# ----------------------------------------------------------------------
# Главная функция векторизации
# ----------------------------------------------------------------------

def vectorize(pred_dir: Path, log=print) -> VectorPlan:
    """Векторизация предсказаний сети в векторный план."""
    pred_dir = Path(pred_dir)

    sem = np.array(Image.open(pred_dir / "sem.png"))
    dist = np.load(pred_dir / "dist.npy")
    orient = np.load(pred_dir / "orient.npy")
    heights = np.load(pred_dir / "heights.npy")

    scene = json.loads((pred_dir / "scene.json").read_text(encoding="utf-8"))
    frame = scene["frame"]

    log(f"Векторизация {pred_dir.name} (улучшенная)...")

    walls = extract_walls(sem, dist, orient, frame, min_length_m=0.8, simplify_tol=0.05)
    log(f"  Стен: {len(walls)} (после фильтрации)")

    rooms = extract_rooms(sem, frame)
    log(f"  Помещений: {len(rooms)}")

    openings = extract_openings(sem, heights, walls, frame, scene.get("ceiling_z", 3.0))
    log(f"  Проёмов: {len(openings)}")

    # Связываем помещения и стены
    for room in rooms:
        room_poly = np.array(room.points)
        room_center = np.array(room.label_xy)

        for wall in walls:
            wall_pts = np.array(wall.points)
            if len(wall_pts) < 2:
                continue

            for k in range(len(wall_pts) - 1):
                a, b = wall_pts[k], wall_pts[k + 1]
                mid = (a + b) / 2

                dist_to_wall = np.linalg.norm(room_center - mid)
                if dist_to_wall < 5.0:
                    if room.id not in wall.rooms:
                        wall.rooms.append(room.id)

    plan = VectorPlan(
        level=scene.get("level", 1),
        floor_z=scene.get("floor_z", 0.0),
        ceiling_z=scene.get("ceiling_z", 3.0),
        walls=walls,
        openings=openings,
        rooms=rooms,
        frame=frame
    )

    return plan


# ----------------------------------------------------------------------
# Экспорт
# ----------------------------------------------------------------------

def save_json(plan: VectorPlan, path: Path) -> None:
    """Сохранение плана в JSON."""
    data = asdict(plan)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_svg(plan: VectorPlan, path: Path) -> None:
    """Сохранение плана в SVG."""
    from xml.etree import ElementTree as ET

    all_pts = []
    for wall in plan.walls:
        all_pts.extend(wall.points)
    for room in plan.rooms:
        all_pts.extend(room.points)

    if not all_pts:
        return

    all_pts = np.array(all_pts)
    x_min, y_min = all_pts.min(axis=0) - 1.0
    x_max, y_max = all_pts.max(axis=0) + 1.0
    width, height = x_max - x_min, y_max - y_min

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"{x_min} {-y_max} {width} {height}",
        "width": f"{width * 100:.0f}",
        "height": f"{height * 100:.0f}"
    })

    ET.SubElement(svg, "rect", {
        "x": str(x_min), "y": str(-y_max),
        "width": str(width), "height": str(height),
        "fill": "#f5f5f5"
    })

    for room in plan.rooms:
        pts = " ".join(f"{x},{-y}" for x, y in room.points)
        ET.SubElement(svg, "polygon", {
            "points": pts,
            "fill": "#e8f4f8",
            "stroke": "none"
        })

    for wall in plan.walls:
        pts = " ".join(f"{x},{-y}" for x, y in wall.points)
        color = "#2a2a2a" if wall.is_outer else "#4a4a4a"
        # Толщина в SVG - метры * 100 для перевода в см визуального отображения
        stroke_width = wall.thickness * 100
        ET.SubElement(svg, "polyline", {
            "points": pts,
            "fill": "none",
            "stroke": color,
            "stroke-width": str(stroke_width),
            "stroke-linecap": "square"
        })

    for opening in plan.openings:
        x1, y1 = opening.a
        x2, y2 = opening.b
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[opening.opening_type]
        ET.SubElement(svg, "line", {
            "x1": str(x1), "y1": str(-y1),
            "x2": str(x2), "y2": str(-y2),
            "stroke": color,
            "stroke-width": "0.08",
            "stroke-linecap": "round"
        })

    for room in plan.rooms:
        x, y = room.label_xy
        text = ET.SubElement(svg, "text", {
            "x": str(x), "y": str(-y),
            "text-anchor": "middle",
            "font-size": "0.3",
            "font-family": "sans-serif",
            "fill": "#333"
        })
        text.text = f"{room.area:.1f} m²"

    tree = ET.ElementTree(svg)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def save_png(plan: VectorPlan, path: Path, dpi: int = 100) -> None:
    """Сохранение плана в PNG."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
    ax.set_aspect('equal')
    ax.axis('off')

    for room in plan.rooms:
        poly = MplPolygon(room.points, facecolor='#e8f4f8', edgecolor='none')
        ax.add_patch(poly)

    for wall in plan.walls:
        pts = np.array(wall.points)
        if len(pts) < 2:
            continue
        color = '#2a2a2a' if wall.is_outer else '#4a4a4a'
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=wall.thickness * 100,
                solid_capstyle='projecting')

    for opening in plan.openings:
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[opening.opening_type]
        ax.plot([opening.a[0], opening.b[0]], [opening.a[1], opening.b[1]],
                color=color, linewidth=3, solid_capstyle='round')

    for room in plan.rooms:
        ax.text(room.label_xy[0], room.label_xy[1], f"{room.area:.1f} m²",
                ha='center', va='center', fontsize=8, color='#333')

    all_pts = []
    for wall in plan.walls:
        all_pts.extend(wall.points)
    if all_pts:
        all_pts = np.array(all_pts)
        margin = 1.0
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='#f5f5f5')
    plt.close()


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main_vectorize(pred_dir, out_dir, log=print):
    """Векторизация одного уровня."""
    pred_dir = Path(pred_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = vectorize(pred_dir, log=log)

    save_json(plan, out_dir / "plan.json")
    log(f"Сохранено: {out_dir / 'plan.json'}")

    save_svg(plan, out_dir / "plan.svg")
    log(f"Сохранено: {out_dir / 'plan.svg'}")

    save_png(plan, out_dir / "plan.png")
    log(f"Сохранено: {out_dir / 'plan.png'}")

    return plan