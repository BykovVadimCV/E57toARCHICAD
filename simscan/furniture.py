"""Процедурная мебель: предметы из частей, чтобы лучи проходили там же, где в жизни.

Система координат предмета (как в interior._local_box): du - вдоль стены, dv - от стены
в комнату, z - от пола. base - габарит предмета (Box, стоящий спиной к стене).
Функции возвращают (boxes, prims): параллелепипеды и не-параллелепипеды (цилиндры).
"""
from __future__ import annotations

import math

import numpy as np

from .layout import Box


def lb(base: Box, du: float, dv: float, z0: float, size) -> Box:
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    x = base.center[0] + du * c - dv * s
    y = base.center[1] + du * s + dv * c
    return Box((x, y, z0 + size[2] / 2), tuple(float(v) for v in size), base.yaw)


def _legs(base, w, d, h, t=0.04, inset=0.04):
    return [lb(base, su * (w / 2 - inset - t / 2), sv * (d / 2 - inset - t / 2), 0.0, (t, t, h))
            for su in (-1, 1) for sv in (-1, 1)]


def chair(base: Box, rng) -> list:
    """Стул: ножки, сиденье, спинка - под сиденьем и между ножками лучи проходят."""
    w, d = base.size[0], base.size[1]
    seat_z = float(rng.uniform(0.42, 0.47))
    boxes = _legs(base, w, d, seat_z - 0.04, t=0.035, inset=0.02)
    boxes.append(lb(base, 0, 0, seat_z - 0.04, (w, d, 0.04)))
    back_h = float(rng.uniform(0.35, 0.5))
    boxes.append(lb(base, 0, -d / 2 + 0.015, seat_z, (w, 0.03, back_h)))
    return boxes


def sofa(base: Box, rng) -> list:
    w, d, h = base.size
    gap = float(rng.uniform(0.04, 0.15))
    boxes = _legs(base, w, d, gap, t=0.05, inset=0.05)
    boxes.append(lb(base, 0, 0.1, gap, (w - 0.36, d - 0.2, 0.42 - gap)))       # сиденье
    boxes.append(lb(base, 0, -d / 2 + 0.1, gap, (w, 0.2, h - gap)))             # спинка
    for su in (-1, 1):                                                          # подлокотники
        boxes.append(lb(base, su * (w / 2 - 0.09), 0.1, gap, (0.18, d - 0.2, 0.62 - gap)))
    return boxes


def bed(base: Box, rng) -> list:
    w, d, h = base.size
    on_legs = rng.random() < 0.6
    gap = float(rng.uniform(0.18, 0.3)) if on_legs else 0.0
    boxes = _legs(base, w, d, gap, t=0.06) if on_legs else []
    boxes.append(lb(base, 0, 0, gap, (w, d, 0.15)))                             # рама
    boxes.append(lb(base, 0, 0.01, gap + 0.15, (w - 0.04, d - 0.06, max(0.12, h - gap - 0.15))))
    boxes.append(lb(base, 0, -d / 2 + 0.03, 0.0, (w, 0.06, float(rng.uniform(0.9, 1.2)))))
    return boxes


def shelf(base: Box, rng) -> list:
    """Открытый стеллаж: боковины, полки, книги; задняя стенка не всегда."""
    w, d, h = base.size
    t = 0.02
    boxes = [lb(base, su * (w / 2 - t / 2), 0, 0.0, (t, d, h)) for su in (-1, 1)]
    boxes.append(lb(base, 0, 0, h - t, (w - 2 * t, d, t)))
    boxes.append(lb(base, 0, 0.02, 0.0, (w - 2 * t, d - 0.04, 0.06)))          # цоколь
    if rng.random() < 0.5:
        boxes.append(lb(base, 0, -d / 2 + 0.003, 0.06, (w - 2 * t, 0.006, h - 0.06 - t)))
    step = float(rng.uniform(0.3, 0.4))
    z = 0.06
    while z + step < h - t:
        boxes.append(lb(base, 0, 0, z + step - t, (w - 2 * t, d, t)))
        # книги блоками: заполнение 30-90 % полки
        u = -w / 2 + t
        while u < w / 2 - t - 0.1:
            bw = float(rng.uniform(0.08, 0.35))
            if rng.random() < 0.65 and u + bw < w / 2 - t:
                bh = float(min(step - 0.04, rng.uniform(0.15, 0.3)))
                bd = float(min(d - 0.02, rng.uniform(0.15, 0.25)))
                boxes.append(lb(base, u + bw / 2, -d / 2 + bd / 2 + 0.01, z, (bw, bd, bh)))
            u += bw + float(rng.uniform(0.0, 0.1))
        z += step
    return boxes


def wardrobe(base: Box, rng) -> list:
    w, d, h = base.size
    return [lb(base, 0, -0.02, 0.0, (w - 0.02, d - 0.06, 0.08)),               # утопленный цоколь
            lb(base, 0, 0, 0.08, (w, d, h - 0.08))]


def tv_stand(base: Box, rng) -> list:
    w, d, h = base.size
    gap = 0.1
    boxes = _legs(base, w, d, gap, t=0.04)
    boxes.append(lb(base, 0, 0, gap, (w, d, h - gap)))
    if rng.random() < 0.6:
        tw = w * float(rng.uniform(0.5, 0.8))
        boxes.append(lb(base, 0, -d / 2 + 0.08, h, (tw, 0.05, tw * 0.58)))
    return boxes


def counter(base: Box, rng) -> list:
    """Кухонная тумба: утопленный цоколь, корпус, столешница со свесом."""
    w, d, h = base.size
    return [lb(base, 0, -0.04, 0.0, (w, d - 0.08, 0.1)),
            lb(base, 0, -0.01, 0.1, (w, d - 0.02, h - 0.14)),
            lb(base, 0, 0.01, h - 0.04, (w, d + 0.02, 0.04))]


def shoe_rack(base: Box, rng) -> list:
    w, d, h = base.size
    t = 0.02
    boxes = [lb(base, su * (w / 2 - t / 2), 0, 0.0, (t, d, h)) for su in (-1, 1)]
    n = max(2, int(h / 0.25))
    for k in range(n + 1):
        boxes.append(lb(base, 0, 0, min(h - t, k * h / n), (w - 2 * t, d, t)))
    return boxes


def table(base: Box, rng) -> list:
    w, d, h = base.size
    return [lb(base, 0, 0, h - 0.03, (w, d, 0.03)), *_legs(base, w, d, h - 0.03, t=0.05, inset=0.03)]


def plant(x: float, y: float, rng):
    """Растение: горшок, стебель и облако листьев - насквозь просвечивает."""
    r = float(rng.uniform(0.12, 0.2))
    ph = float(rng.uniform(0.25, 0.45))
    stem = float(rng.uniform(0.3, 0.9))
    prims = [{"type": "cylinder", "center": [x, y, ph / 2], "radius": r, "height": ph},
             {"type": "cylinder", "center": [x, y, ph + stem / 2], "radius": 0.015, "height": stem}]
    boxes = []
    crown_r = float(rng.uniform(0.2, 0.45))
    cz = ph + stem
    for _ in range(int(rng.integers(30, 70))):
        v = rng.normal(0, 1, 3)
        v /= np.linalg.norm(v)
        p = np.array([x, y, cz]) + v * crown_r * rng.uniform(0.2, 1.0) * np.array([1, 1, 0.8])
        s = rng.uniform(0.04, 0.12)
        boxes.append(Box(tuple(p), (float(s), float(s * 0.5), 0.01), float(rng.uniform(0, math.pi))))
    return boxes, prims


def coat_rack(x: float, y: float, rng):
    """Напольная вешалка: стойка на диске и 2-5 вещей на крючках (мягкие сетки)."""
    prims = [{"type": "cylinder", "center": [x, y, 0.015], "radius": 0.2, "height": 0.03},
             {"type": "cylinder", "center": [x, y, 0.9], "radius": 0.02, "height": 1.8}]
    for k in range(int(rng.integers(2, 6))):
        a = rng.uniform(0, 2 * math.pi)
        normal = np.array([math.cos(a), math.sin(a), 0.0])
        along = np.array([-math.sin(a), math.cos(a), 0.0])
        top = np.array([x, y, 1.75]) + 0.03 * normal
        prims.append(hanging_cloth(top, along, normal, rng))
    return [], prims


def radiator(wall, s: float, face: float, side: int, width: float, height: float, rng) -> list:
    """Радиатор: секционный (рёбра с зазорами) или панельный."""
    yaw = wall.yaw
    z0 = 0.1
    if rng.random() < 0.5:
        center = wall.point(s, face + side * 0.08)
        return [Box((*center, z0 + height / 2), (width, 0.08, height), yaw)]
    boxes = []
    n = max(3, int(width / 0.08))
    pitch = width / n
    for k in range(n):
        sk = s - width / 2 + (k + 0.5) * pitch
        c = wall.point(sk, face + side * 0.08)
        boxes.append(Box((*c, z0 + height / 2), (pitch - 0.012, 0.08, height), yaw))
    return boxes


# ----------------------------------------------------------------------
# Беспорядок: мягкие и неровные предметы (сетки задаются явно: type "trimesh")
# ----------------------------------------------------------------------

def _smooth_noise(rng, n_terms=6, scale=1.0):
    """Гладкая случайная функция двух переменных (сумма косинусов)."""
    w = rng.normal(0, scale, (n_terms, 2))
    ph = rng.uniform(0, 2 * math.pi, n_terms)
    a = rng.normal(0, 1, n_terms) / math.sqrt(n_terms)
    return lambda u, v: (a[None] * np.cos(np.stack([u, v], -1) @ w.T + ph)).sum(-1)


def drape(support: Box, rng, size=None, floor_z: float = 0.0) -> dict:
    """Ткань, наброшенная на опору (куртка на коробке или спинке стула, плед на диване).

    Над опорой ткань лежит на её верхней грани, за краями - свисает почти вертикально
    (чуть отходя от края) до пола или на свою длину; сверху - складки."""
    sx, sy, sz = support.size
    top = support.center[2] + sz / 2
    cw = float(size[0]) if size else float(rng.uniform(0.5, 1.0))     # размеры ткани
    cd = float(size[1]) if size else float(rng.uniform(0.6, 1.2))
    du, dv = rng.uniform(-0.25, 0.25, 2) * np.array([sx, sy])
    n = 16
    u = np.linspace(-cw / 2, cw / 2, n) + du
    v = np.linspace(-cd / 2, cd / 2, n) + dv
    uu, vv = np.meshgrid(u, v, indexing="ij")
    out_u = np.maximum(np.abs(uu) - sx / 2, 0)                      # насколько за краем опоры
    out_v = np.maximum(np.abs(vv) - sy / 2, 0)
    out = np.hypot(out_u, out_v)
    wr = _smooth_noise(rng, scale=8.0)
    wrinkle = 0.03 * wr(uu, vv)
    z = top + 0.01 + np.abs(wrinkle) * (out == 0)
    z = np.where(out > 0, top - out * 0.92 + wrinkle, z)
    pooled = np.maximum(floor_z + 0.005 - z, 0) / 0.92          # легло на пол - уходит вбок
    z = np.maximum(z, floor_z + 0.005)
    # свисающая часть отходит от грани опоры на 2-6 см (складки, толщина ткани)
    push = np.minimum(out, 0.04) + 0.02 + 0.02 * np.abs(wr(vv, uu)) + pooled
    gu = np.where(out_u > 0, np.sign(uu) * (sx / 2 + push), uu)
    gv = np.where(out_v > 0, np.sign(vv) * (sy / 2 + push), vv)
    c, s = math.cos(support.yaw), math.sin(support.yaw)
    x = support.center[0] + gu * c - gv * s
    y = support.center[1] + gu * s + gv * c
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    return {"type": "trimesh", "v": verts.round(4).tolist(), "f": _grid_faces(n, n)}


def hanging_cloth(top_center, along, normal, rng, width=None, length=None) -> dict:
    """Куртка или полотенце на крючке: сверху узко, книзу шире, с объёмом от стены."""
    top_center = np.asarray(top_center, float)
    along = np.asarray(along, float)
    normal = np.asarray(normal, float)
    w = float(width or rng.uniform(0.35, 0.55))
    L = float(length or rng.uniform(0.6, 0.95))
    n_u, n_v = 9, 10
    t = np.linspace(0, 1, n_v)                       # сверху вниз
    s = np.linspace(-0.5, 0.5, n_u)
    tt, ss = np.meshgrid(t, s, indexing="ij")
    half = w * (0.25 + 0.75 * np.sqrt(tt))           # плечи шире ворота
    bulge = 0.12 * np.sin(np.pi * np.clip(tt * 1.2, 0, 1)) * np.cos(np.pi * ss) + 0.02
    wr = _smooth_noise(rng, scale=6.0)
    wob = 0.02 * wr(tt * L, ss * w)
    pts = (top_center[None, None] + (ss * 2 * half)[..., None] * along
           + (bulge + wob)[..., None] * normal)
    pts = pts.astype(float)
    pts[..., 2] = top_center[2] - tt * L
    return {"type": "trimesh", "v": pts.reshape(-1, 3).round(4).tolist(), "f": _grid_faces(n_v, n_u)}


def blob(center_xy, size, rng, floor_z: float = 0.0) -> dict:
    """Неровная куча: сумка, рюкзак, бельё, мешок. Сфера с гладким радиальным шумом,
    приплюснутая и поставленная на пол."""
    sx, sy, sz = (float(v) for v in size)
    n_lat, n_lon = 9, 14
    th = np.linspace(0, np.pi, n_lat)
    ph = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    dirs = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], -1)
    wr = _smooth_noise(rng, n_terms=8, scale=2.5)
    r = 1.0 + 0.22 * wr(T * 1.3, P)
    pts = dirs * r[..., None] * np.array([sx / 2, sy / 2, sz / 2])
    pts[..., 2] = np.maximum(pts[..., 2], -sz / 2 * 0.85)          # плоское дно
    yaw = float(rng.uniform(0, 2 * np.pi))
    c, s = math.cos(yaw), math.sin(yaw)
    x = pts[..., 0] * c - pts[..., 1] * s + center_xy[0]
    y = pts[..., 0] * s + pts[..., 1] * c + center_xy[1]
    z = pts[..., 2] - pts[..., 2].min() + floor_z
    verts = np.stack([x, y, z], -1).reshape(-1, 3)
    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            a, b = i * n_lon + j, i * n_lon + (j + 1) % n_lon
            c2, d = a + n_lon, b + n_lon
            faces += [[a, b, d], [a, d, c2]]
    return {"type": "trimesh", "v": verts.round(4).tolist(), "f": faces}


def leaning_board(base_xy, wall_dir, out_normal, rng, floor_z: float = 0.0) -> dict:
    """Доска / лист гипсокартона, прислонённые к стене (наклон 70-82 градуса)."""
    L = float(rng.uniform(1.5, 2.6))
    w = float(rng.uniform(0.15, 1.2))
    t = float(rng.uniform(0.012, 0.03))
    tilt = math.radians(float(rng.uniform(70, 82)))
    up = np.array([0.0, 0.0, 1.0])
    n = np.r_[np.asarray(out_normal, float), 0.0]
    a = np.r_[np.asarray(wall_dir, float), 0.0]
    axis = math.sin(tilt) * up - math.cos(tilt) * n          # вдоль доски, верх к стене
    thick = np.cross(a, axis)
    if thick @ n < 0:                                        # толщина - от стены в комнату
        thick = -thick
    p0 = np.r_[np.asarray(base_xy, float), floor_z] + n * (L * math.cos(tilt))
    corners = []
    for k in range(8):
        cu = (k & 1) * w - w / 2
        cl = ((k >> 1) & 1) * L
        ct = ((k >> 2) & 1) * t
        corners.append(p0 + a * cu + axis * cl + thick * ct)
    f = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
         [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]]
    return {"type": "trimesh", "v": np.round(corners, 4).tolist(), "f": f}


def _grid_faces(n_rows: int, n_cols: int) -> list:
    f = []
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            a, b = i * n_cols + j, i * n_cols + j + 1
            c, d = a + n_cols, b + n_cols
            f += [[a, b, d], [a, d, c]]
    return f
