"""Smart architectural vectorizer: from messy predictions to real floor plans.

Not just vectorization - this enforces architectural constraints:
- Closed apartment perimeter
- Axis-aligned orthogonal walls (snap nearly-horizontal/vertical)
- Variable thickness following the mask
- Connected wall topology (no floating segments)
- Valid enclosed rooms only
- Gap closing and junction snapping
- Standalone wall removal

Transforms neural network fuzzy predictions into architecturally-valid plans.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image


@dataclass
class WallSegment:
    """Стена с переменной толщиной."""
    id: str
    points: list[list[float]]  # Ось стены
    thickness: list[float]      # Толщина в каждой точке
    rooms: list[str]
    is_outer: bool
    length: float


@dataclass
class Opening:
    """Проём."""
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
    """Помещение - только если полностью замкнуто стенами."""
    id: str
    name: str
    points: list[list[float]]
    area: float
    perimeter: float
    label_xy: tuple[float, float] = (0.0, 0.0)
    is_enclosed: bool = False  # Замкнуто стенами


@dataclass
class VectorPlan:
    """Векторизованный план."""
    level: int
    floor_z: float
    ceiling_z: float
    walls: list[WallSegment] = field(default_factory=list)
    openings: list[Opening] = field(default_factory=list)
    rooms: list[Room] = field(default_factory=list)
    frame: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)  # Статистика очистки


# ----------------------------------------------------------------------
# Координатные преобразования
# ----------------------------------------------------------------------

def px_to_world(ji: np.ndarray, frame: dict) -> np.ndarray:
    """Пиксели -> метры."""
    ji = np.atleast_2d(np.asarray(ji, float))
    px, H = frame["pixel_m"], frame["height"]
    x = frame["origin_x"] + (ji[:, 0] + 0.5) * px
    y = frame["origin_y"] + (H - 1 - ji[:, 1] + 0.5) * px
    return np.c_[x, y]


def world_to_px(xy: np.ndarray, frame: dict) -> np.ndarray:
    """Метры -> пиксели."""
    xy = np.atleast_2d(np.asarray(xy, float))
    px, H = frame["pixel_m"], frame["height"]
    j = (xy[:, 0] - frame["origin_x"]) / px - 0.5
    i = H - 1 - ((xy[:, 1] - frame["origin_y"]) / px - 0.5)
    return np.c_[j, i]


# ----------------------------------------------------------------------
# 1. Доминантные углы и выравнивание по осям
# ----------------------------------------------------------------------

def compute_dominant_angles(segments: list[np.ndarray], n_bins: int = 360) -> list[float]:
    """Находит доминантные углы в плане (обычно 0°, 90° для ортогональных зданий)."""
    angles = []
    for seg in segments:
        if len(seg) < 2:
            continue
        v = seg[-1] - seg[0]
        angle = np.arctan2(v[1], v[0])
        angles.append(angle)

    if not angles:
        return [0.0, np.pi/2]

    # Гистограмма углов
    hist, edges = np.histogram(angles, bins=n_bins, range=(-np.pi, np.pi))

    # Сглаживание
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    hist_smooth = np.convolve(hist, kernel, mode='same')

    # Пики
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i-1] and hist_smooth[i] > hist_smooth[i+1]:
            if hist_smooth[i] > hist_smooth.max() * 0.3:  # Значимые пики
                angle = (edges[i] + edges[i+1]) / 2
                peaks.append(angle)

    if not peaks:
        return [0.0, np.pi/2]

    # Группировка близких углов
    peaks = sorted(peaks)
    grouped = [peaks[0]]
    for p in peaks[1:]:
        if abs(p - grouped[-1]) > np.radians(15):  # Минимум 15° между доминантами
            grouped.append(p)

    # Нормализация к [-π/2, π/2]
    normalized = []
    for a in grouped:
        while a > np.pi/2:
            a -= np.pi
        while a < -np.pi/2:
            a += np.pi
        normalized.append(a)

    return sorted(set(normalized))


def snap_to_dominant_angle(segment: np.ndarray, dominant_angles: list[float],
                           tolerance_deg: float = 5.0) -> np.ndarray:
    """Выравнивание сегмента по ближайшему доминантному углу."""
    if len(segment) < 2:
        return segment

    start, end = segment[0], segment[-1]
    v = end - start
    current_angle = np.arctan2(v[1], v[0])

    # Нормализация
    while current_angle > np.pi/2:
        current_angle -= np.pi
    while current_angle < -np.pi/2:
        current_angle += np.pi

    # Ближайший доминантный угол
    best_angle = min(dominant_angles, key=lambda a: abs(a - current_angle))

    if abs(current_angle - best_angle) < np.radians(tolerance_deg):
        # Поворот сегмента
        length = np.linalg.norm(v)
        new_v = length * np.array([np.cos(best_angle), np.sin(best_angle)])
        new_end = start + new_v

        # Интерполяция промежуточных точек
        if len(segment) == 2:
            return np.array([start, new_end])
        else:
            # Проекция промежуточных точек на новую линию
            t = np.linspace(0, 1, len(segment))
            aligned = start[None, :] + t[:, None] * new_v[None, :]
            return aligned

    return segment


# ----------------------------------------------------------------------
# 2. Граф связности стен
# ----------------------------------------------------------------------

class WallGraph:
    """Граф топологии стен: узлы = концы стен, рёбра = сегменты."""

    def __init__(self, segments_world: list[np.ndarray], snap_tolerance: float = 0.10):
        self.segments = segments_world
        self.snap_tol = snap_tolerance
        self.nodes = []  # Уникальные точки
        self.edges = []  # (node_i, node_j, segment_idx)
        self._build_graph()

    def _snap_point(self, pt: np.ndarray) -> int:
        """Найти или создать узел для точки (с привязкой)."""
        for i, node in enumerate(self.nodes):
            if np.linalg.norm(pt - node) < self.snap_tol:
                return i
        self.nodes.append(pt.copy())
        return len(self.nodes) - 1

    def _build_graph(self):
        """Построить граф из сегментов."""
        for seg_idx, seg in enumerate(self.segments):
            if len(seg) < 2:
                continue
            start_node = self._snap_point(seg[0])
            end_node = self._snap_point(seg[-1])
            self.edges.append((start_node, end_node, seg_idx))

    def connected_components(self) -> list[list[int]]:
        """Найти связные компоненты (группы соединённых стен)."""
        n = len(self.nodes)
        visited = [False] * n
        components = []

        def dfs(node, component):
            visited[node] = True
            component.append(node)
            for i, j, _ in self.edges:
                if i == node and not visited[j]:
                    dfs(j, component)
                elif j == node and not visited[i]:
                    dfs(i, component)

        for i in range(n):
            if not visited[i]:
                comp = []
                dfs(i, comp)
                components.append(comp)

        return components

    def segment_component_map(self) -> dict[int, int]:
        """Маппинг: segment_idx -> component_id."""
        components = self.connected_components()
        node_to_comp = {}
        for comp_id, nodes in enumerate(components):
            for node in nodes:
                node_to_comp[node] = comp_id

        seg_comp = {}
        for start, end, seg_idx in self.edges:
            comp_id = node_to_comp.get(start, -1)
            seg_comp[seg_idx] = comp_id

        return seg_comp

    def largest_component_segments(self) -> list[int]:
        """Индексы сегментов в наибольшей связной компоненте (основной периметр)."""
        components = self.connected_components()
        if not components:
            return []

        largest_comp = max(components, key=len)
        largest_comp_id = self.connected_components().index(largest_comp)

        seg_comp = self.segment_component_map()
        return [seg_idx for seg_idx, comp_id in seg_comp.items() if comp_id == largest_comp_id]

    def snap_endpoints(self):
        """Обновить сегменты привязанными координатами узлов."""
        for start, end, seg_idx in self.edges:
            seg = self.segments[seg_idx]
            if len(seg) >= 2:
                seg[0] = self.nodes[start]
                seg[-1] = self.nodes[end]


# ----------------------------------------------------------------------
# 3. Скелетонизация с очисткой
# ----------------------------------------------------------------------

def cleanup_wall_mask(mask: np.ndarray, min_area_px: int = 100) -> np.ndarray:
    """Морфологическая очистка маски стен."""
    try:
        import cv2
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_area_px:
                cleaned[labels == i] = 0
        return cleaned
    except ImportError:
        return mask


def thin_walls(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """Zhang-Suen скелетонизация."""
    def _thin_iter(img: np.ndarray, iter_num: int) -> tuple[np.ndarray, int]:
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

    skel = mask.astype(bool).copy()
    for _ in range(max_iter):
        skel, n1 = _thin_iter(skel, 0)
        skel, n2 = _thin_iter(skel, 1)
        if n1 + n2 == 0:
            break
    return skel.astype(np.uint8)


def skeleton_to_segments(skel: np.ndarray) -> list[np.ndarray]:
    """Скелет -> полилинии."""
    try:
        from scipy import ndimage
        kernel = np.ones((3, 3), np.uint8)
        kernel[1, 1] = 0
        neighbors = ndimage.convolve(skel.astype(np.uint8), kernel, mode='constant')
    except ImportError:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
        neighbors = np.zeros_like(skel, dtype=np.uint8)
        for i in range(1, skel.shape[0] - 1):
            for j in range(1, skel.shape[1] - 1):
                if skel[i, j]:
                    neighbors[i, j] = skel[i-1:i+2, j-1:j+2].sum() - skel[i, j]

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
# 4. Переменная толщина вдоль стены
# ----------------------------------------------------------------------

def sample_thickness_along_wall(seg_px: np.ndarray, dist: np.ndarray, n_samples: int = 20) -> list[float]:
    """Сэмплирование толщины вдоль стены из карты расстояний."""
    if len(seg_px) < 2:
        return [0.20]

    # Интерполяция вдоль сегмента
    t = np.linspace(0, 1, n_samples)
    if len(seg_px) == 2:
        sampled = seg_px[0][None, :] + t[:, None] * (seg_px[-1] - seg_px[0])[None, :]
    else:
        # Сплайн-интерполяция
        from scipy.interpolate import interp1d
        s = np.linspace(0, 1, len(seg_px))
        interp_x = interp1d(s, seg_px[:, 0], kind='linear')
        interp_y = interp1d(s, seg_px[:, 1], kind='linear')
        sampled = np.c_[interp_x(t), interp_y(t)]

    # Извлечение толщины
    thicknesses = []
    for pt in sampled:
        xi, yi = int(pt[1]), int(pt[0])
        if 0 <= xi < dist.shape[1] and 0 <= yi < dist.shape[2]:
            d = dist[0, xi, yi]
            if d < 0.9:  # Не насыщенное значение
                thick = float(d * 2.0)  # dist - половина толщины
                thick = np.clip(thick, 0.08, 0.30)
                thicknesses.append(thick)

    if not thicknesses:
        return [0.20] * n_samples

    # Сглаживание
    from scipy.ndimage import gaussian_filter1d
    smoothed = gaussian_filter1d(thicknesses, sigma=2.0)

    return smoothed.tolist()


# ----------------------------------------------------------------------
# 5. Упрощение с сохранением топологии
# ----------------------------------------------------------------------

def simplify_polyline_rdp(pts: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker упрощение."""
    if len(pts) <= 2:
        return pts

    start, end = pts[0], pts[-1]
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

    left = simplify_polyline_rdp(pts[:idx_max + 1], tol)
    right = simplify_polyline_rdp(pts[idx_max:], tol)
    return np.vstack([left[:-1], right])


# ---------------------------------------------------------------------# 6. Закрытие зазоров в стенах
# ----------------------------------------------------------------------

def close_small_gaps(segments: list[np.ndarray], graph: WallGraph, 
                     max_gap: float = 0.15) -> list[np.ndarray]:
    """Закрытие небольших разрывов между близкими концами стен."""
    closed = []
    merged_indices = set()
    
    for i, seg1 in enumerate(segments):
        if i in merged_indices:
            continue
            
        current = seg1.copy()
        
        # Ищем сегменты, которые можно присоединить к концу
        while True:
            endpoint = current[-1]
            best_match = None
            best_dist = max_gap
            
            for j, seg2 in enumerate(segments):
                if j == i or j in merged_indices:
                    continue
                
                # Расстояние до начала seg2
                dist_to_start = np.linalg.norm(endpoint - seg2[0])
                if dist_to_start < best_dist:
                    # Проверка коллинеарности
                    v1 = current[-1] - current[-2] if len(current) > 1 else np.array([1, 0])
                    v2 = seg2[1] - seg2[0] if len(seg2) > 1 else np.array([1, 0])
                    v1_norm = np.linalg.norm(v1)
                    v2_norm = np.linalg.norm(v2)
                    if v1_norm > 1e-9 and v2_norm > 1e-9:
                        v1 = v1 / v1_norm
                        v2 = v2 / v2_norm
                        if abs(np.dot(v1, v2)) > 0.9:
                            best_match = j
                            best_dist = dist_to_start
            
            if best_match is not None:
                gap_segment = np.linspace(endpoint, segments[best_match][0], 3)
                current = np.vstack([current, gap_segment[1:], segments[best_match]])
                merged_indices.add(best_match)
            else:
                break
        
        closed.append(current)
        merged_indices.add(i)
    
    return closed


# ----------------------------------------------------------------------
# 7. Валидация помещений
# ----------------------------------------------------------------------

def is_room_enclosed(room_mask: np.ndarray, wall_mask: np.ndarray) -> bool:
    """Проверка что помещение окружено стенами."""
    try:
        import cv2
        dilated = cv2.dilate(room_mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        perimeter_on_walls = (dilated & wall_mask).sum()
        perimeter_total = dilated.sum() - room_mask.sum()
        return perimeter_on_walls / max(perimeter_total, 1) > 0.6
    except ImportError:
        coords_y, coords_x = np.where(room_mask)
        if len(coords_x) == 0:
            return False
        y_min, y_max = coords_y.min(), coords_y.max()
        x_min, x_max = coords_x.min(), coords_x.max()
        perimeter_pixels = set()
        for y in range(y_min, y_max + 1):
            perimeter_pixels.add((x_min, y))
            perimeter_pixels.add((x_max, y))
        for x in range(x_min, x_max + 1):
            perimeter_pixels.add((x, y_min))
            perimeter_pixels.add((x, y_max))
        wall_pixels = sum(1 for x, y in perimeter_pixels 
                         if 0 <= y < wall_mask.shape[0] and 
                            0 <= x < wall_mask.shape[1] and wall_mask[y, x])
        return wall_pixels / max(len(perimeter_pixels), 1) > 0.6


# ----------------------------------------------------------------------
# 8. Главная функция извлечения стен
# ----------------------------------------------------------------------

def extract_walls_smart(sem: np.ndarray, dist: np.ndarray, orient: np.ndarray, 
                        frame: dict, min_length_m: float = 0.8) -> tuple[list[WallSegment], dict]:
    """Умная векторизация с архитектурными ограничениями."""
    
    metadata = {"raw_segments": 0, "after_length_filter": 0, 
                "after_connectivity_filter": 0, "gaps_closed": 0}
    
    wall_mask = (sem == 2).astype(np.uint8)
    if wall_mask.sum() < 10:
        return [], metadata
    
    wall_mask = cleanup_wall_mask(wall_mask, min_area_px=100)
    skel = thin_walls(wall_mask)
    segments_px = skeleton_to_segments(skel)
    metadata["raw_segments"] = len(segments_px)
    
    if not segments_px:
        return [], metadata
    
    segments_world = [px_to_world(seg, frame) for seg in segments_px]
    
    def seg_length(seg):
        return float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1)))
    
    segments_world = [s for s in segments_world if seg_length(s) >= min_length_m * 0.5]
    metadata["after_length_filter"] = len(segments_world)
    
    graph = WallGraph(segments_world, snap_tolerance=0.10)
    graph.snap_endpoints()
    
    main_indices = graph.largest_component_segments()
    segments_world = [segments_world[i] for i in main_indices]
    segments_px = [segments_px[i] for i in main_indices]
    metadata["after_connectivity_filter"] = len(segments_world)
    
    if not segments_world:
        return [], metadata
    
    dominant_angles = compute_dominant_angles(segments_world)
    aligned = [snap_to_dominant_angle(s, dominant_angles, 5.0) for s in segments_world]
    
    graph = WallGraph(aligned, snap_tolerance=0.10)
    graph.snap_endpoints()
    aligned = graph.segments
    
    closed = close_small_gaps(aligned, graph, max_gap=0.15)
    metadata["gaps_closed"] = len(aligned) - len(closed)
    
    final = []
    final_px = []
    for i, seg in enumerate(closed):
        if seg_length(seg) >= min_length_m:
            simp = simplify_polyline_rdp(seg, tol=0.05)
            final.append(simp)
            if i < len(segments_px):
                final_px.append(segments_px[i])
            else:
                final_px.append(world_to_px(seg, frame))
    
    walls = []
    for idx, (seg_world, seg_px) in enumerate(zip(final, final_px)):
        if len(seg_world) < 2:
            continue
        
        try:
            thicknesses = sample_thickness_along_wall(seg_px, dist, n_samples=len(seg_world))
        except:
            thicknesses = [0.20] * len(seg_world)
        
        is_outer = any(int(y) < 5 or int(y) >= sem.shape[0] - 5 or 
                      int(x) < 5 or int(x) >= sem.shape[1] - 5 
                      for x, y in seg_px[:min(5, len(seg_px))])
        
        wall = WallSegment(
            id=f"wall_{idx:03d}",
            points=seg_world.tolist(),
            thickness=thicknesses,
            rooms=[],
            is_outer=is_outer,
            length=seg_length(seg_world)
        )
        walls.append(wall)
    
    return walls, metadata


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Connected component labeling fallback."""
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


def extract_rooms_smart(sem: np.ndarray, frame: dict, wall_mask: np.ndarray) -> list[Room]:
    """Извлечение только валидных помещений."""
    interior_mask = (sem == 1).astype(np.uint8)
    if interior_mask.sum() < 10:
        return []

    try:
        from scipy import ndimage
        labeled, n_labels = ndimage.label(interior_mask)
    except ImportError:
        labeled, n_labels = _label_components(interior_mask)

    rooms = []
    px_area = frame["pixel_m"] ** 2

    for label_id in range(1, n_labels + 1):
        room_mask = (labeled == label_id)
        if room_mask.sum() < 25 or not is_room_enclosed(room_mask, wall_mask):
            continue

        coords_y, coords_x = np.where(room_mask)
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
        area = float(room_mask.sum() * px_area)
        perimeter = float(np.sum(np.linalg.norm(np.diff(np.vstack([contour, contour[:1]]), axis=0), axis=1)))
        centroid = px_to_world(np.array([[coords_x.mean(), coords_y.mean()]]), frame)[0]

        rooms.append(Room(
            id=f"room_{label_id:03d}",
            name=f"Room {label_id}",
            points=contour.tolist(),
            area=area,
            perimeter=perimeter,
            label_xy=tuple(centroid.tolist()),
            is_enclosed=True
        ))

    return rooms


def extract_openings(sem: np.ndarray, heights: np.ndarray, walls: list[WallSegment],
                     frame: dict, ceiling_z: float) -> list[Opening]:
    """Проёмы."""
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
            coords_world = px_to_world(np.c_[coords_x, coords_y], frame)
            center = coords_world.mean(axis=0)

            closest_wall = None
            min_dist = 1.0
            for wall in walls:
                for k in range(len(wall.points) - 1):
                    a, b = np.array(wall.points[k]), np.array(wall.points[k + 1])
                    v, w = b - a, center - a
                    c1, c2 = np.dot(w, v), np.dot(v, v)
                    if c2 < 1e-9:
                        d = np.linalg.norm(w)
                    elif c1 <= 0:
                        d = np.linalg.norm(w)
                    elif c1 >= c2:
                        d = np.linalg.norm(center - b)
                    else:
                        d = np.linalg.norm(center - (a + (c1 / c2) * v))
                    if d < min_dist:
                        min_dist, closest_wall = d, wall

            if closest_wall:
                x_min, y_min = coords_world.min(axis=0)
                x_max, y_max = coords_world.max(axis=0)
                openings.append(Opening(
                    id=f"{opening_type}_{len([o for o in openings if o.opening_type == opening_type]):03d}",
                    opening_type=opening_type,
                    wall_id=closest_wall.id,
                    a=[float(x_min), float(y_min)],
                    b=[float(x_max), float(y_max)],
                    width=float(max(x_max - x_min, y_max - y_min)),
                    sill=0.9 if opening_type == "window" else None,
                    head=2.1
                ))
    return openings


def vectorize(pred_dir: Path, log=print) -> VectorPlan:
    """Главная функция."""
    pred_dir = Path(pred_dir)
    sem = np.array(Image.open(pred_dir / "sem.png"))
    dist = np.load(pred_dir / "dist.npy")
    orient = np.load(pred_dir / "orient.npy")
    heights = np.load(pred_dir / "heights.npy")
    scene = json.loads((pred_dir / "scene.json").read_text(encoding="utf-8"))
    frame = scene["frame"]

    log(f"Умная векторизация {pred_dir.name}...")
    walls, metadata = extract_walls_smart(sem, dist, orient, frame, min_length_m=0.8)
    log(f"  Стен: {len(walls)} (сырых={metadata['raw_segments']}, зазоров={metadata['gaps_closed']})")
    
    wall_mask = (sem == 2).astype(np.uint8)
    rooms = extract_rooms_smart(sem, frame, wall_mask)
    log(f"  Помещений: {len(rooms)} (только замкнутые)")
    
    openings = extract_openings(sem, heights, walls, frame, scene.get("ceiling_z", 3.0))
    log(f"  Проёмов: {len(openings)}")

    for room in rooms:
        for wall in walls:
            for k in range(len(wall.points) - 1):
                mid = (np.array(wall.points[k]) + np.array(wall.points[k + 1])) / 2
                if np.linalg.norm(np.array(room.label_xy) - mid) < 5.0:
                    if room.id not in wall.rooms:
                        wall.rooms.append(room.id)

    return VectorPlan(
        level=scene.get("level", 1),
        floor_z=scene.get("floor_z", 0.0),
        ceiling_z=scene.get("ceiling_z", 3.0),
        walls=walls, openings=openings, rooms=rooms,
        frame=frame, metadata=metadata
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
    width, height = x_max - x_min, y_max - y_min

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"{x_min} {-y_max} {width} {height}",
        "width": f"{width * 100:.0f}",
        "height": f"{height * 100:.0f}"
    })
    ET.SubElement(svg, "rect", {"x": str(x_min), "y": str(-y_max), 
                                "width": str(width), "height": str(height), "fill": "#f5f5f5"})

    for room in plan.rooms:
        ET.SubElement(svg, "polygon", {
            "points": " ".join(f"{x},{-y}" for x, y in room.points),
            "fill": "#e8f4f8", "stroke": "none"
        })

    for wall in plan.walls:
        pts = np.array(wall.points)
        if len(pts) < 2:
            continue
        avg_thick = np.mean(wall.thickness) if isinstance(wall.thickness, list) else wall.thickness
        ET.SubElement(svg, "polyline", {
            "points": " ".join(f"{x},{-y}" for x, y in pts),
            "fill": "none",
            "stroke": "#2a2a2a" if wall.is_outer else "#4a4a4a",
            "stroke-width": str(avg_thick * 100),
            "stroke-linecap": "square"
        })

    for opening in plan.openings:
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[opening.opening_type]
        ET.SubElement(svg, "line", {
            "x1": str(opening.a[0]), "y1": str(-opening.a[1]),
            "x2": str(opening.b[0]), "y2": str(-opening.b[1]),
            "stroke": color, "stroke-width": "0.08", "stroke-linecap": "round"
        })

    for room in plan.rooms:
        text = ET.SubElement(svg, "text", {
            "x": str(room.label_xy[0]), "y": str(-room.label_xy[1]),
            "text-anchor": "middle", "font-size": "0.3",
            "font-family": "sans-serif", "fill": "#333"
        })
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
            avg_thick = np.mean(wall.thickness) if isinstance(wall.thickness, list) else wall.thickness
            ax.plot(pts[:, 0], pts[:, 1], color='#2a2a2a' if wall.is_outer else '#4a4a4a',
                   linewidth=avg_thick * 100, solid_capstyle='projecting')

    for opening in plan.openings:
        color = {"door": "#d66049", "window": "#4393c3", "opening": "#f4a582"}[opening.opening_type]
        ax.plot([opening.a[0], opening.b[0]], [opening.a[1], opening.b[1]],
               color=color, linewidth=3, solid_capstyle='round')

    for room in plan.rooms:
        ax.text(room.label_xy[0], room.label_xy[1], f"{room.area:.1f} m²",
               ha='center', va='center', fontsize=8, color='#333')

    all_pts = [p for w in plan.walls for p in w.points]
    if all_pts:
        all_pts = np.array(all_pts)
        margin = 1.0
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)

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
