"""Эталон: 2D-маски плана в СК планировки и описание геометрии в мм.

Привязка пикселя - та же, что в конвертере (раздел 5.9 ТЗ):
    x = origin_x + (j + 0,5) * px,   y = origin_y + (H - 1 - i + 0,5) * px
Пиксель принадлежит объекту, если его центр внутри прямоугольника
(полуоткрытый интервал [min, max)) - так площадь в пикселях точно равна
площади в м2 / px2 для размеров, кратных пикселю.

Маски считаются на секущей плоскости cut_height_m:
    walls   - тела стен и колонн, пересекающие секущую плоскость;
    doors   - проёмы дверей и арок в толще стены;
    windows - проёмы окон в толще стены;
    rooms   - номер помещения + 1 (uint16), 0 - вне помещений и в стенах.
Мебель, радиаторы, короба на эталон не попадают.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .labels import PLAN_WALL_LABELS
from .layout import Box, Layout
from .scene import Solid
from .transform import WorldTransform

CLASSES = {0: "background", 1: "wall", 2: "door", 3: "window"}


@dataclass
class RasterFrame:
    origin_x: float
    origin_y: float
    pixel_m: float
    width: int
    height: int

    def xy_to_ij(self, x, y):
        """Метры -> дробные (строка, столбец) центра пикселя."""
        j = (np.asarray(x) - self.origin_x) / self.pixel_m - 0.5
        i = self.height - 1 - ((np.asarray(y) - self.origin_y) / self.pixel_m - 0.5)
        return i, j

    def ij_to_xy(self, i, j):
        x = self.origin_x + (np.asarray(j) + 0.5) * self.pixel_m
        y = self.origin_y + (self.height - 1 - np.asarray(i) + 0.5) * self.pixel_m
        return x, y


def layout_warp(layout: Layout):
    """Сдвиг плана из realism (матрица 2x2) или None, если его нет."""
    A = layout.meta.get("warp")
    if A is None or np.allclose(A, np.eye(2)):
        return None
    return np.asarray(A, float)


def warped_bbox(layout: Layout):
    x0, y0, x1, y1 = layout.bbox()
    A = layout_warp(layout)
    if A is None:
        return x0, y0, x1, y1
    c = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ A.T
    return (*c.min(0), *c.max(0))


def frame_for(layout: Layout, pixel_mm: float, pad_m: float) -> RasterFrame:
    p = pixel_mm / 1000.0
    x0, y0, x1, y1 = warped_bbox(layout)
    ox = math.floor((x0 - pad_m) / p) * p
    oy = math.floor((y0 - pad_m) / p) * p
    w = int(math.ceil((x1 + pad_m - ox) / p))
    h = int(math.ceil((y1 + pad_m - oy) / p))
    return RasterFrame(round(ox, 6), round(oy, 6), p, w, h)


def _span(lo: float, hi: float, origin: float, p: float, n: int) -> tuple[int, int]:
    """Индексы k с центром origin + (k + 0,5) p в [lo, hi)."""
    a = math.ceil((lo - origin) / p - 0.5 - 1e-9)
    b = math.ceil((hi - origin) / p - 0.5 - 1e-9)
    return max(a, 0), min(b, n)


def raster_rect(frame: RasterFrame, mask: np.ndarray, rect, value=True) -> None:
    """Осевой прямоугольник (xmin, ymin, xmax, ymax)."""
    xa, ya, xb, yb = rect
    j0, j1 = _span(xa, xb, frame.origin_x, frame.pixel_m, frame.width)
    k0, k1 = _span(ya, yb, frame.origin_y, frame.pixel_m, frame.height)
    if j1 > j0 and k1 > k0:
        mask[frame.height - k1:frame.height - k0, j0:j1] = value


def raster_quad(frame: RasterFrame, mask: np.ndarray, pts, value=True) -> None:
    """Выпуклый четырёхугольник (по центрам пикселей)."""
    pts = np.asarray(pts, float)
    j0, j1 = _span(pts[:, 0].min(), pts[:, 0].max() + frame.pixel_m, frame.origin_x,
                   frame.pixel_m, frame.width)
    k0, k1 = _span(pts[:, 1].min(), pts[:, 1].max() + frame.pixel_m, frame.origin_y,
                   frame.pixel_m, frame.height)
    if j1 <= j0 or k1 <= k0:
        return
    jj, kk = np.meshgrid(np.arange(j0, j1), np.arange(k0, k1))
    x = frame.origin_x + (jj + 0.5) * frame.pixel_m
    y = frame.origin_y + (kk + 0.5) * frame.pixel_m
    cross = []
    for a, b in zip(pts, np.roll(pts, -1, 0)):
        cross.append((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]))
    cross = np.stack(cross)
    inside = (cross >= 0).all(0) | (cross <= 0).all(0)
    mask[frame.height - 1 - kk[inside], jj[inside]] = value


def _box_corners(box: Box):
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    hx, hy = box.size[0] / 2, box.size[1] / 2
    return np.array([(box.center[0] + u * c - v * s, box.center[1] + u * s + v * c)
                     for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])


def raster_box(frame: RasterFrame, mask: np.ndarray, box: Box, value=True, warp=None) -> None:
    """След параллелепипеда на плане; осевые - точно, повёрнутые - по центрам пикселей.
    warp - сдвиг плана (2x2): след становится параллелограммом."""
    if warp is not None:
        raster_quad(frame, mask, _box_corners(box) @ warp.T, value)
        return
    cx, cy, _ = box.center
    sx, sy, _ = box.size
    yaw = box.yaw % (math.pi / 2)
    if min(yaw, math.pi / 2 - yaw) < 1e-9:
        quarter = round(box.yaw / (math.pi / 2)) % 2
        hx, hy = (sx / 2, sy / 2) if quarter == 0 else (sy / 2, sx / 2)
        raster_rect(frame, mask, (cx - hx, cy - hy, cx + hx, cy + hy), value)
        return
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    r = math.hypot(sx, sy) / 2
    j0, j1 = _span(cx - r, cx + r, frame.origin_x, frame.pixel_m, frame.width)
    k0, k1 = _span(cy - r, cy + r, frame.origin_y, frame.pixel_m, frame.height)
    if j1 <= j0 or k1 <= k0:
        return
    jj, kk = np.meshgrid(np.arange(j0, j1), np.arange(k0, k1))
    x = frame.origin_x + (jj + 0.5) * frame.pixel_m - cx
    y = frame.origin_y + (kk + 0.5) * frame.pixel_m - cy
    u, v = x * c + y * s, -x * s + y * c
    inside = (u >= -sx / 2) & (u < sx / 2) & (v >= -sy / 2) & (v < sy / 2)
    rows = frame.height - 1 - kk[inside]
    mask[rows, jj[inside]] = value


def opening_box(layout: Layout, o) -> Box:
    wall = layout.wall(o.wall_id)
    c = wall.point(o.s)
    return Box((c[0], c[1], (o.z0 + o.z1) / 2), (o.width, wall.thickness, o.z1 - o.z0), wall.yaw)


def build_masks(layout: Layout, solids: list[Solid], frame: RasterFrame,
                cut_z: float) -> dict[str, np.ndarray]:
    shape = (frame.height, frame.width)
    A = layout_warp(layout)
    walls = np.zeros(shape, bool)
    # ограждения балконов ниже секущей плоскости, но на планах рисуются стеной
    parapets = {f"wall:{w.id}" for w in layout.walls if w.kind == "parapet"}
    for s in solids:
        if s.label not in PLAN_WALL_LABELS:
            continue
        z0, z1 = s.box.center[2] - s.box.size[2] / 2, s.box.center[2] + s.box.size[2] / 2
        if z0 <= cut_z < z1 or (s.source in parapets and z0 <= 0.0):
            raster_box(frame, walls, s.box, warp=A)
    doors = np.zeros(shape, bool)
    windows = np.zeros(shape, bool)
    for o in layout.openings:
        raster_box(frame, windows if o.kind == "window" else doors, opening_box(layout, o), warp=A)
    doors &= ~walls
    windows &= ~walls
    rooms = np.zeros(shape, np.uint16)
    for r in layout.rooms:
        for rect in (r.region or [r.cell]):
            if A is None:
                raster_rect(frame, rooms, rect, r.id + 1)
            else:
                x0, y0, x1, y1 = rect
                raster_quad(frame, rooms, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ A.T,
                            r.id + 1)
    rooms[walls | doors | windows] = 0
    classes = np.zeros(shape, np.uint8)
    classes[walls], classes[doors], classes[windows] = 1, 2, 3
    return {"walls": walls, "doors": doors, "windows": windows, "rooms": rooms, "classes": classes}


def describe(layout: Layout, masks: dict, frame: RasterFrame, world: WorldTransform,
             cut_z: float) -> dict:
    """gt.json: привязка растра, размеры в мм и площади для отчёта о точности."""
    px2 = frame.pixel_m ** 2
    A = layout_warp(layout)
    Wp = (lambda p: np.asarray(p, float)) if A is None else (lambda p: np.asarray(p, float) @ A.T)
    walls = []
    for w in layout.walls:
        faces = []
        for side in (-1, 1):
            off = side * w.thickness / 2
            faces.append([Wp(w.point(-w.ext0, off)).round(4).tolist(),
                          Wp(w.point(w.length + w.ext1, off)).round(4).tolist()])
        walls.append({"id": w.id, "kind": w.kind, "thickness_mm": round(w.thickness * 1000, 1),
                      "axis": [Wp(w.p0).round(4).tolist(), Wp(w.p1).round(4).tolist()],
                      "faces": faces})
    openings = []
    for o in layout.openings:
        c = Wp(layout.wall(o.wall_id).point(o.s))
        openings.append({"id": o.id, "kind": o.kind, "wall_id": o.wall_id,
                         "width_mm": round(o.width * 1000, 1),
                         "z0_m": o.z0, "z1_m": o.z1, "rooms": list(o.rooms),
                         "closed": o.kind == "door" and o.angle_deg == 0.0,
                         "center_layout_m": c.round(4).tolist(),
                         "center_world_m": world.to_world([c[0], c[1], 0.0]).round(4).tolist()})
    rooms = []
    for r in layout.rooms:
        rooms.append({"id": r.id, "kind": r.kind, "scanned": r.scanned,
                      "area_px_m2": round(float((masks["rooms"] == r.id + 1).sum()) * px2, 4),
                      "ceiling_z_m": r.ceiling_z})
    return {
        "frame": asdict(frame),
        "pixel_mm": frame.pixel_m * 1000,
        "pixel_formula": "x = origin_x + (j + 0.5) * px; y = origin_y + (H - 1 - i + 0.5) * px "
                         "(СК планировки, затем layout_to_world)",
        "cut_height_m": cut_z,
        "warp": None if A is None else A.tolist(),
        "classes": CLASSES,
        "layout_to_world": world.to_dict(),
        "ceiling_height_m": layout.ceiling_height,
        "walls": walls,
        "openings": openings,
        "rooms": rooms,
    }


# Маска U-Net ReFloorBRUSNIKA (datasetgen UNetSemanticClasses): значение = класс * 64.
UNET_VALUES = {"background": 0, "wall": 64, "window": 128, "door": 192}


def unet_frame(layout: Layout, target_wall_px: float, pad_m: float) -> RasterFrame:
    """Квадратный кадр, в котором наружная стена занимает target_wall_px пикселей
    (так ReFloorBRUSNIKA нормирует масштаб перед U-Net, core/scale_norm.py)."""
    p = layout.meta["t_ext"] / target_wall_px
    x0, y0, x1, y1 = warped_bbox(layout)
    side = max(x1 - x0, y1 - y0) + 2 * pad_m
    n = int(math.ceil(side / p))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return RasterFrame(cx - n * p / 2, cy - n * p / 2, p, n, n)


def unet_mask(layout: Layout, solids: list[Solid], frame: RasterFrame, cut_z: float) -> np.ndarray:
    """Стены 64, окна 128, двери 192. К двери, как в datasetgen, относится и полотно
    (открытое полотно реально видно в скане); дуги открывания нет - в облаке её нет."""
    m = build_masks(layout, solids, frame, cut_z)
    out = np.zeros(m["walls"].shape, np.uint8)
    out[m["walls"]] = UNET_VALUES["wall"]
    out[m["windows"]] = UNET_VALUES["window"]
    out[m["doors"]] = UNET_VALUES["door"]
    leaves = np.zeros_like(m["walls"])
    A = layout_warp(layout)
    for s in solids:
        if s.label == "door" and s.source.endswith(":leaf"):
            raster_box(frame, leaves, s.box, warp=A)
    out[leaves & (out == 0)] = UNET_VALUES["door"]
    return out


def save_gt(out_dir: str | Path, masks: dict, info: dict) -> None:
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("walls", "doors", "windows"):
        cv2.imwrite(str(out / f"{name}.png"), masks[name].astype(np.uint8) * 255)
    cv2.imwrite(str(out / "rooms.png"), masks["rooms"])
    cv2.imwrite(str(out / "classes.png"), masks["classes"])
    # «чертёжный» вид: стены чёрные, окна серые, фон белый
    plan = np.full(masks["walls"].shape, 255, np.uint8)
    plan[masks["windows"]] = 160
    plan[masks["walls"]] = 0
    cv2.imwrite(str(out / "plan_gt.png"), plan)
    (out / "gt.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
