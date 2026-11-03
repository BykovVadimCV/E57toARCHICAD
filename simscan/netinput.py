"""Датасет для нейросети плана: многоканальный вход уровня + разметка + векторный эталон.

Главное правило: синтетический вход строится тем же кодом, что и реальный. Сцена ->
симуляция сканера (стоянки, шум, смешанные точки, стекло) -> облако E57 -> rasterize_e57.
Из меша вход не строится никогда: сеть выучила бы идеальные линии.

    python -m simscan dataset build out/scene_00000 out/scene_00001 ... --out dataset
    python -m simscan dataset real D:/scans/kvartira.e57 --out dataset [--id kvartira]

Раскладка:
    dataset/meta.json                     версия, пиксель, каналы с нормировками, классы
    dataset/splits/{train,val,test}.txt   по планировкам, а не по сканам
    dataset/scenes/<scene>/<level>/       input.npy  float16 [8, H, W]
                                          sem.png    uint8, классы 0-7 (255 - только в старых датасетах)
                                          dist.npy   float16 [1, H, W]
                                          orient.npy float16 [2, H, W]
                                          heights.npy float16 [2, H, W]
                                          vector.json, scene.json, preview.png
    dataset/real/<object>/<level>/        input.npy, scene.json, preview.png

Растр - в мировой СК облака (x вправо, y вверх; строка 0 - наибольший y), пиксель 2 см:
    x = origin_x + (j + 0,5) * px,   y = origin_y + (H - 1 - i + 0,5) * px.
Повороты для разнообразия - на облаке до растеризации (у генератора - случайный курс мира),
поворот готового растра дал бы интерполяцию, которой в реальных данных нет.
"""
from __future__ import annotations

import hashlib
import zlib
import json
import math
from pathlib import Path

import numpy as np

RASTERIZER_VERSION = "netinput-1"
LABELS_VERSION = "labels-5"           # labels-2: неотсканированные помещения - ignore (ignore_unseen);
                                      # labels-3: и любое помещение без стоянки - вне видимого;
                                      # labels-4: не ignore, а улица (unseen_to_outside): нет точек - нет помещения
PIXEL_M = 0.02
Z_BIN_M = 0.05                       # срез для долей занятости и вертикальности
BAND_MARGIN_M = 0.1                  # от пола и потолка
LOW_TOP, MID_TOP = 0.9, 2.0          # границы полос
FLOOR_TOL_M = 0.03                   # точки пола / потолка: |z - плоскость| < допуска
DENSITY_REF = 4000.0                 # точек в колонке 2x2 см, дающих 1,0 (BLK360 + Cyclone, ~4 мм)
BELOW_DEPTH_M = 3.0                  # «ниже пола»: от 0,1 до 3 м под полом уровня

CHANNELS = [
    {"name": "occ_low", "norm": "доля занятых срезов по 5 см в полосе 0,1-0,9 м"},
    {"name": "occ_mid", "norm": "то же, 0,9-2,0 м"},
    {"name": "occ_high", "norm": "то же, от 2,0 м до потолка минус 0,1 м"},
    {"name": "floor_vis", "norm": "0/1: в колонке есть точки пола (|z - пол| < 3 см)"},
    {"name": "ceil_vis", "norm": "0/1: в колонке есть точки потолка (|z - потолок| < 3 см)"},
    {"name": "density", "norm": f"log(1 + точек между полом и потолком) / log(1 + {DENSITY_REF:.0f}), обрезка до 1"},
    {"name": "vert_frac", "norm": "доля точек колонки в срезах по 5 см, у которых заняты и срез выше, "
                                  "и срез ниже (вертикальная поверхность); 0,1 м от пола - до 0,1 м от потолка"},
    {"name": "below_floor", "norm": "0/1: точки на 0,1-3 м ниже пола уровня (лестничный проём, второй свет)"},
]

CLASSES = {0: "outside", 1: "interior", 2: "wall", 3: "door", 4: "window", 5: "opening", 6: "clutter",
           7: "void", 255: "ignore"}
OUTSIDE, INTERIOR, WALL, DOOR, WINDOW, OPENING, CLUTTER, VOID, IGNORE = 0, 1, 2, 3, 4, 5, 6, 7, 255
EXTERIOR_BAND_M = 0.10               # наружная стена - полоса 10 см наружу от внутренней грани
DIST_MAX_M = 0.30                    # усечение расстояния до грани
CLUTTER_LABELS = ("furniture", "fixture", "clutter")
CLUTTER_SKIP_KINDS = ("baseboard",)  # плинтус - часть грани, не мусор


def meta_doc() -> dict:
    from . import __version__

    return {"version": 3, "labels": LABELS_VERSION, "rasterizer": RASTERIZER_VERSION, "generator": __version__,
            "pixel_m": PIXEL_M, "z_bin_m": Z_BIN_M, "bands_m": [BAND_MARGIN_M, LOW_TOP, MID_TOP],
            "channels": CHANNELS, "classes": {str(k): v for k, v in CLASSES.items()},
            "dist": {"max_m": DIST_MAX_M, "norm": "расстояние до ближайшей видимой грани стены / 0,3, обрезка до 1"},
            "orient": "(cos 2θ, sin 2θ) направления стены-хозяйки в мировой СК (x вправо, y вверх); "
                      "только на классах 2-5, иначе 0",
            "heights": "(низ проёма, верх проёма) / высота потолка; только на классах 3-5, иначе 0",
            "raster": "x = origin_x + (j + 0.5) * px; y = origin_y + (H - 1 - i + 0.5) * px (мировая СК облака)",
            "exterior_wall": f"полоса {EXTERIOR_BAND_M} м наружу от внутренней грани; истинная толщина - в vector.json",
            "ignore": "labels-5: unsupported physical regions ignored by default; legacy: неотснятое - улица (unseen_to_outside: помещение без стоянки или почти невидимое - "
                      "вне видимого); в labels-2/3 оно было ignore, в версии 1 - ещё и кольцо в 1 пиксель по границам классов",
            "frame_pad_m": "датасет: пустые поля 0,5-2,5 м с каждой стороны, случайно (вход для реальных - 0,5 м)"}


# ----------------------------------------------------------------------
# Вход: облако -> 8 каналов
# ----------------------------------------------------------------------

def _frame_from_points(xy: np.ndarray, px: float, pad_m=0.5, grow_m: float = 2.0,
                       cell_m: float = 0.1) -> dict:
    """Кадр уровня: ядро - 0,2-99,8 % точек (одиночные выбросы - вон); дальше - до grow_m по
    занятым ячейкам cell_m, связным с ядром (тонкая стена у края с долей точек меньше 0,2 % -
    не обрезается); плюс pad_m (число или (слева, снизу, справа, сверху))."""
    lo = np.percentile(xy, 0.2, axis=0)
    hi = np.percentile(xy, 99.8, axis=0)
    lo, hi = _grow_box(xy, lo, hi, grow_m, cell_m)
    pad = np.broadcast_to(np.asarray(pad_m, float), (4,)) if np.ndim(pad_m) else np.full(4, float(pad_m))
    lo, hi = lo - pad[:2], hi + pad[2:]
    ox = math.floor(lo[0] / px) * px
    oy = math.floor(lo[1] / px) * px
    W = int(math.ceil((hi[0] - ox) / px))
    H = int(math.ceil((hi[1] - oy) / px))
    return {"origin_x": round(ox, 6), "origin_y": round(oy, 6), "pixel_m": px, "width": W, "height": H}


def _grow_box(xy: np.ndarray, lo, hi, grow_m: float, cell_m: float):
    """Рамка [lo, hi] -> расширенная по ячейкам (от 2 точек), связным с ячейками внутри рамки."""
    import cv2

    a, b = lo - grow_m, hi + grow_m
    q = xy[(xy[:, 0] >= a[0]) & (xy[:, 0] < b[0]) & (xy[:, 1] >= a[1]) & (xy[:, 1] < b[1])]
    if len(q) < 100:
        return lo, hi
    n = np.ceil((b - a) / cell_m).astype(int) + 1
    ij = np.floor((q - a) / cell_m).astype(int)
    cnt = np.bincount(ij[:, 1] * n[0] + ij[:, 0], minlength=n[0] * n[1]).reshape(n[1], n[0])
    occ = (cnt >= 2).astype(np.uint8)
    k, lab = cv2.connectedComponents(occ, connectivity=8)
    c0, c1 = np.floor((lo - a) / cell_m).astype(int), np.floor((hi - a) / cell_m).astype(int)
    core = np.unique(lab[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1])
    keep = np.isin(lab, core[core > 0])
    if not keep.any():
        return lo, hi
    ys, xs = np.nonzero(keep)
    lo2 = a + np.array([xs.min(), ys.min()]) * cell_m
    hi2 = a + np.array([xs.max() + 1, ys.max() + 1]) * cell_m
    return np.minimum(lo, lo2), np.maximum(hi, hi2)


def affine(frame: dict) -> list:
    """Пиксель (j, i, 1) -> мир (x, y)."""
    px, H = frame["pixel_m"], frame["height"]
    return [[px, 0.0, frame["origin_x"] + 0.5 * px], [0.0, -px, frame["origin_y"] + (H - 0.5) * px]]


def world_to_px(xy: np.ndarray, frame: dict) -> np.ndarray:
    """Мир (x, y) -> дробные (столбец, строка) с центрами пикселей в целых."""
    xy = np.atleast_2d(np.asarray(xy, float))
    j = (xy[:, 0] - frame["origin_x"]) / frame["pixel_m"] - 0.5
    i = frame["height"] - 1 - ((xy[:, 1] - frame["origin_y"]) / frame["pixel_m"] - 0.5)
    return np.c_[j, i]


def station_positions(reader) -> np.ndarray:
    """Стоянки: позы сканов (E57 по станциям) или позы сферических снимков (сведённый
    файл); все в одной точке (поза облака) - пусто."""
    from .inspect_e57 import _image_stations

    st = np.array([s.translation for s in reader.scans if s.translation is not None], float).reshape(-1, 3)
    if len(st) < 2:
        st = _image_stations(reader.tree)
    if len(st) < 2 or float(np.ptp(st[:, :2], axis=0).max()) < 0.3:
        return np.zeros((0, 3))
    return st


def frame_points(points: np.ndarray, levels: list, stations: np.ndarray, reach_m: float = 20.0) -> np.ndarray:
    """Точки, по которым строится кадр: в пределах высот уровней (грунт под окнами - нет) и не
    дальше reach_m от стоянок (соседние дома через окна - нет)."""
    z = points[:, 2]
    m = np.zeros(len(points), bool)
    for lv in levels:
        m |= (z > lv["floor_z"] - 0.1) & (z < lv["ceiling_z"] + 0.1)
    if len(stations):
        d = np.full(len(points), np.inf)
        for sx, sy, _ in stations:
            d = np.minimum(d, np.hypot(points[:, 0] - sx, points[:, 1] - sy))
        m &= d < reach_m
    return m if m.sum() > 100 else np.ones(len(points), bool)


class E57Source:
    """Точки E57 (по станциям или сведённый) чанками в мировой СК + стоянки."""

    def __init__(self, path, chunk: int = 2_000_000):
        from .e57read import E57Reader

        self.reader = E57Reader(path)
        self.chunk = chunk
        self.total = sum(s.points for s in self.reader.scans)
        self.stations = station_positions(self.reader)

    def chunks(self):
        from .floorplan import _scan_points

        for i in range(len(self.reader.scans)):
            yield from _scan_points(self.reader, i, self.chunk)

    def close(self):
        self.reader.close()


class MemorySource:
    """Те же точки, что легли бы в scan.e57, без записи на диск: локальные координаты станции
    в float32 (как E57_SINGLE) и записанная поза (с ошибкой регистрации). Мировые координаты
    считаются один раз и хранятся (float32 от центра стоянок, 12 байт на точку) - для
    нескольких растров скана; от растра из файла отличаются долями пикселя на границах ячеек
    (округление до микрометров)."""

    def __init__(self, scans, poses, chunk: int = 2_000_000):
        self.chunk = chunk
        # float32 - относительно центра стоянок (мировые координаты до сотен метров
        # в float32 потеряли бы доли миллиметра)
        self.center = np.mean([np.asarray(t, float) for _, t in poses], axis=0) if poses else np.zeros(3)
        parts = []
        for s_, (R, t) in zip(scans, poses):
            xyz = s_.xyz_local[s_.valid].astype(np.float32).astype(float)
            parts.append((xyz @ np.asarray(R, float).T + (np.asarray(t, float) - self.center)).astype(np.float32))
        self.points = np.concatenate(parts) if parts else np.zeros((0, 3), np.float32)
        self.total = len(self.points)
        st = np.array([np.asarray(t, float) for _, t in poses]).reshape(-1, 3)
        self.stations = st if len(st) >= 2 and float(np.ptp(st[:, :2], axis=0).max()) >= 0.3 else np.zeros((0, 3))

    def chunks(self):
        for a in range(0, self.total, self.chunk):
            yield self.points[a:a + self.chunk].astype(float) + self.center

    def close(self):
        pass


def rotate_xy(p: np.ndarray, rot: tuple | None) -> np.ndarray:
    """Аугментация «поворот облака вокруг вертикали»: rot = (градусы, cx, cy)."""
    if not rot or not rot[0]:
        return p
    a = math.radians(rot[0])
    c, s_ = math.cos(a), math.sin(a)
    x, y = p[:, 0] - rot[1], p[:, 1] - rot[2]
    p = p.copy()
    p[:, 0], p[:, 1] = rot[1] + c * x - s_ * y, rot[2] + s_ * x + c * y
    return p


def rasterize_e57(path, px: float = PIXEL_M, levels: list | None = None, thin: float = 0.0,
                  z_shift: float = 0.0, seed: int = 0, chunk: int = 2_000_000,
                  sample_points: int = 3_000_000, log=print) -> list[dict]:
    """E57 (сведённый или по станциям) -> по уровню: {"level", "frame", "input" [8, H, W] float16}.
    levels - пол и потолок уровней (иначе detect_levels по выборке); thin - доля точек,
    выбрасываемых случайно; z_shift - сдвиг облака по z (аугментация «пол ±1 см»)."""
    src = E57Source(path, chunk)
    try:
        return rasterize_points(src, px, levels, thin, z_shift, seed, sample_points, log=log)
    finally:
        src.close()


def rasterize_points(src, px: float = PIXEL_M, levels: list | None = None, thin: float = 0.0,
                     z_shift: float = 0.0, seed: int = 0, sample_points: int = 3_000_000,
                     rot: tuple | None = None, pad_m=0.5, log=print, reference_shift: float = 0.) -> list[dict]:
    """Растеризатор входа сети по источнику точек (E57Source, MemorySource). Один и тот же
    код для реальных файлов и синтетики. rot - поворот облака (градусы, cx, cy); pad_m - поля
    кадра (датасет - случайные, см. build_scene)."""
    from .rasterize import detect_levels, floor_ceiling

    rng = np.random.default_rng(seed)
    p_keep = min(1.0, sample_points / max(src.total, 1))
    sample = []
    for p in src.chunks():
        sample.append(p[rng.random(len(p)) < p_keep].astype(np.float32))
    sample = rotate_xy(np.concatenate(sample).astype(float), rot)
    sample[:, 2] += z_shift
    st = rotate_xy(src.stations, rot) if len(src.stations) else src.stations
    if levels is None:
        levels = detect_levels(sample, stations_z=st[:, 2] + z_shift)
        if not levels:
            fl, ce = floor_ceiling(sample[:, 2])
            levels = [{"floor_z": fl, "ceiling_z": ce, "height_m": ce - fl}]
    levels=[dict(lv, floor_z=lv['floor_z']+reference_shift, ceiling_z=lv['ceiling_z']+reference_shift) for lv in levels]
    acc = []
    for lv in levels:
        top = lv["ceiling_z"] - lv["floor_z"]
        m = frame_points(sample, [lv], st)
        fr = _frame_from_points(sample[m, :2], px, pad_m)
        H, W = fr["height"], fr["width"]
        nz = max(1, int(math.floor((top - 2 * BAND_MARGIN_M) / Z_BIN_M)))
        acc.append({"level": lv, "frame": fr, "nz": nz, "top": top,
                    "vox": np.zeros(H * W * nz, np.uint16), "floor": np.zeros(H * W, bool),
                    "ceil": np.zeros(H * W, bool), "below": np.zeros(H * W, bool),
                    "count": np.zeros(H * W, np.int64)})
    log(f"уровней {len(levels)}: " + ", ".join(f"{a['frame']['width']}x{a['frame']['height']}" for a in acc))
    for p in src.chunks():
        if thin > 0:
            p = p[rng.random(len(p)) >= thin]
        p = rotate_xy(p.astype(float), rot)
        p[:, 2] += z_shift
        for a in acc:
            fr, lv = a["frame"], a["level"]
            H, W, nz = fr["height"], fr["width"], a["nz"]
            j = np.floor((p[:, 0] - fr["origin_x"]) / px).astype(np.int64)
            ii = H - 1 - np.floor((p[:, 1] - fr["origin_y"]) / px).astype(np.int64)
            ok = (ii >= 0) & (ii < H) & (j >= 0) & (j < W)
            h = p[:, 2] - lv["floor_z"]
            flat = ii * W + j
            a["floor"][flat[ok & (np.abs(h) < FLOOR_TOL_M)]] = True
            a["ceil"][flat[ok & (np.abs(h - a["top"]) < FLOOR_TOL_M)]] = True
            a["below"][flat[ok & (h < -BAND_MARGIN_M) & (h > -BELOW_DEPTH_M)]] = True
            inner = ok & (h > FLOOR_TOL_M) & (h < a["top"] - FLOOR_TOL_M)
            a["count"] += np.bincount(flat[inner], minlength=H * W)
            k = np.floor((h - BAND_MARGIN_M) / Z_BIN_M).astype(np.int64)
            band = ok & (k >= 0) & (k < nz)
            accumulate_voxels(a["vox"],flat[band]*nz+k[band])
    out = []
    for a in acc:
        fr, nz = a["frame"], a["nz"]
        H, W = fr["height"], fr["width"]
        vox = a["vox"].reshape(H, W, nz)
        occ = vox > 0
        zc = BAND_MARGIN_M + (np.arange(nz) + 0.5) * Z_BIN_M
        ch = np.zeros((8, H, W), np.float32)
        for c, m in enumerate((zc < LOW_TOP, (zc >= LOW_TOP) & (zc < MID_TOP), zc >= MID_TOP)):
            if m.any():
                ch[c] = occ[..., m].mean(-1)
        ch[3] = a["floor"].reshape(H, W)
        ch[4] = a["ceil"].reshape(H, W)
        ch[5] = np.clip(np.log1p(a["count"].reshape(H, W)) / math.log1p(DENSITY_REF), 0, 1)
        up = np.zeros_like(occ)
        dn = np.zeros_like(occ)
        up[..., :-1] = occ[..., 1:]
        dn[..., 1:] = occ[..., :-1]
        vert = occ & up & dn
        tot = vox.sum(-1, dtype=np.int64)
        ch[6] = np.where(tot > 0, (vox * vert).sum(-1, dtype=np.int64) / np.maximum(tot, 1), 0)
        ch[7] = a["below"].reshape(H, W)
        out.append({"level": a["level"], "frame": fr, "input": ch.astype(np.float16)})
    return out


# ----------------------------------------------------------------------
# Разметка из эталона генератора
# ----------------------------------------------------------------------

def _poly_px(poly_world: np.ndarray, frame: dict) -> np.ndarray:
    q = world_to_px(poly_world, frame)
    return np.round(q * 16).astype(np.int32)              # cv2 shift = 4: 1/16 пикселя


def _fill(mask: np.ndarray, poly_world, frame: dict, value=1) -> None:
    # Half-open scanline membership at actual pixel centers; no inclusive drawing.
    poly=np.asarray(poly_world,float)
    if len(poly)<3:return
    px=frame['pixel_m']; H,W=mask.shape
    i0=max(0,int(np.floor(H-(poly[:,1].max()-frame['origin_y'])/px-.5)))
    i1=min(H,int(np.ceil(H-(poly[:,1].min()-frame['origin_y'])/px-.5))+1)
    rows=np.arange(i0,i1); ys=frame['origin_y']+(H-rows-.5)*px
    aa=poly; bb=np.roll(poly,-1,axis=0)
    hit=(aa[:,1,None]>ys)!=(bb[:,1,None]>ys)
    dy=bb[:,1]-aa[:,1]
    xx=aa[:,0,None]+(ys-aa[:,1,None])*((bb[:,0]-aa[:,0])/np.where(dy!=0,dy,1))[:,None]
    xx=np.sort(np.where(hit,xx,np.inf),axis=0)
    for k in range(0,len(poly)-1,2):
        valid=np.isfinite(xx[k+1])
        for row,lo,hi in zip(rows[valid],xx[k,valid],xx[k+1,valid]):
            j0=max(0,int(np.ceil((lo-frame['origin_x'])/px-.5-1e-9)))
            j1=min(W,int(np.ceil((hi-frame['origin_x'])/px-.5-1e-9)))
            if j1>j0:mask[row,j0:j1]=value




class SceneGT:
    """Эталон сцены генератора в мировой СК облака: сдвиг realism (warp) и курс мира.
    level - этаж двухуровневой квартиры (layout.json "levels", duplex.py), 0 - нижний."""

    def __init__(self, scene_dir, level: int = 0):
        from .layout import Layout
        from .transform import WorldTransform

        self.dir = Path(scene_dir)
        self.doc = json.loads((self.dir / "layout.json").read_text(encoding="utf-8"))
        self.levels = self.doc.get("levels") or [{"z0": 0.0}]
        self.level = level
        lv = self.levels[level]
        self.layout = Layout.from_dict(lv["layout"] if "layout" in lv else self.doc)
        meta = self.doc.get("meta", {})
        w = meta.get("layout_to_world", {})
        self.world = WorldTransform(w.get("yaw_deg", 0.0), tuple(w.get("offset_m", (0.0, 0.0, 0.0))))
        from .planwarp import PlanWarp

        self.warp = PlanWarp(meta.get("warp"))            # сдвиг, наклоны линий стен, дуги
        self.A = self.warp.A
        self.floor_z = float(self.world.offset[2]) + float(lv.get("z0", 0.0))
        self.ceiling_h = float(self.layout.ceiling_height)
        self.rot = None                    # аугментация: облако повёрнуто (градусы, cx, cy)

    def xy(self, pts) -> np.ndarray:
        """План (x, y) -> мир (x, y) (точки)."""
        p = self.warp(pts)
        return rotate_xy(self.world.to_world(np.c_[p, np.zeros(len(p))])[:, :2], self.rot)

    def poly(self, pts, closed: bool = True) -> np.ndarray:
        """Контур плана -> мир; при нелинейной деформации стороны сгущаются (дуги, изломы)."""
        p = self.warp.poly(pts, 0.05, closed)
        return rotate_xy(self.world.to_world(np.c_[p, np.zeros(len(p))])[:, :2], self.rot)

    def dir2(self, w, s0: float, s1: float) -> tuple[float, float]:
        """(cos 2θ, sin 2θ) участка стены s0..s1 в мире."""
        d = np.diff(self.xy([w.point(s0), w.point(s1)]), axis=0)[0]
        t = math.atan2(d[1], d[0])
        return math.cos(2 * t), math.sin(2 * t)

    def stations(self) -> np.ndarray:
        """Стоянки этого этажа (мир)."""
        st = np.array([s["pose_true"]["t"] for s in self.doc.get("stations", [])
                       if s.get("level", 0) == self.level], float).reshape(-1, 3)
        return rotate_xy(st, self.rot) if len(st) else st

    # стена: прямоугольник по оси s и смещению поперёк (в СК плана) -> мир
    def wall_quad(self, w, s0, s1, o0, o1) -> np.ndarray:
        return self.poly([w.point(s0, o0), w.point(s1, o0), w.point(s1, o1), w.point(s0, o1)])

    def inner_offsets(self, w) -> tuple[float, float]:
        """Поперёк стены: (внутренняя грань, грань + 10 см наружу) - для наружной стены."""
        side = w.outward if w.outward else 1
        f = -side * w.thickness / 2
        return f, f + side * min(EXTERIOR_BAND_M, w.thickness)


def distance_to_faces(gt,frame):
    """Continuous XY distance to structural host faces, including column faces."""
    H,W=frame['height'],frame['width'];px=frame['pixel_m']
    out=np.full((H,W),DIST_MAX_M,np.float32);lines=[]
    for w in gt.layout.walls:
        if w.role=='open_balcony':continue
        offsets=[gt.inner_offsets(w)[0]] if w.kind=='exterior' else [-w.thickness/2,w.thickness/2]
        for off in offsets:lines.append(gt.poly([w.point(-w.ext0,off),w.point(w.length+w.ext1,off)],False))
    for it in gt.layout.items:
        if it.label=='column':
            for box in it.boxes:
                q=gt.poly(_box_quad(box));lines.append(np.vstack([q,q[:1]]))
    for line in lines:
        for a,b in zip(line[:-1],line[1:]):
            lo=np.minimum(a,b)-DIST_MAX_M;hi=np.maximum(a,b)+DIST_MAX_M
            j0=max(0,int(np.floor((lo[0]-frame['origin_x'])/px)))
            j1=min(W,int(np.ceil((hi[0]-frame['origin_x'])/px)))
            i0=max(0,H-int(np.ceil((hi[1]-frame['origin_y'])/px)))
            i1=min(H,H-int(np.floor((lo[1]-frame['origin_y'])/px)))
            if j1<=j0 or i1<=i0:continue
            yy,xx=np.mgrid[i0:i1,j0:j1]
            x=frame['origin_x']+(xx+.5)*px-a[0]
            y=frame['origin_y']+(H-yy-.5)*px-a[1]
            v=b-a;t=np.clip((x*v[0]+y*v[1])/max(float(v@v),1e-20),0,1)
            np.minimum(out[i0:i1,j0:j1],np.hypot(x-t*v[0],y-t*v[1]),out=out[i0:i1,j0:j1])
    return (out/DIST_MAX_M).astype(np.float16)[None]


def label_level(gt: SceneGT, frame: dict) -> dict:
    """Семантика, расстояние до грани, ориентация, высоты - в кадре frame."""
    import cv2

    H, W = frame["height"], frame["width"]
    lay = gt.layout
    rooms = np.zeros((H, W), np.uint8)
    room_id = np.zeros((H, W), np.uint16)                 # номер помещения + 1 (unseen_to_outside)
    for r in lay.rooms:
        if r.open_sky:                                    # открытый балкон - пока улица (с ограждением)
            continue
        for x0, y0, x1, y1 in r.rects():
            q = gt.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            _fill(rooms, q, frame)
            _fill(room_id, q, frame, r.id + 1)
    body = np.zeros((H, W), np.uint8)                     # истинные тела стен (для граней)
    sem = np.zeros((H, W), np.uint8)
    sem[rooms > 0] = INTERIOR
    ori = np.zeros((2, H, W), np.float32)
    hts = np.zeros((2, H, W), np.float32)

    curved = not gt.warp.linear
    piece_id = np.zeros((H, W), np.int32)                 # нелинейная деформация: кусок стены
    piece_dir = [(0.0, 0.0)]                              # -> его направление

    def wall_dir(w):
        return gt.dir2(w, 0.0, w.length)

    def set_dir(m, w):
        """Ориентация на пикселях m стены w: по оси или по кускам (наклоны, дуги)."""
        if not curved:
            c, s = wall_dir(w)
            ori[0][m], ori[1][m] = c, s
            return
        k = piece_id[m]
        dirs = np.asarray(piece_dir, np.float32)[k]
        own = k > 0
        if not own.all():                                  # кусков нет - направление всей стены
            dirs[~own] = wall_dir(w)
        ori[0][m], ori[1][m] = dirs[:, 0], dirs[:, 1]

    if curved:                                            # куски по 10 см со своим направлением
        for w in lay.walls:
            s0, s1 = -w.ext0, w.length + w.ext1
            n = max(1, int(math.ceil((s1 - s0) / 0.1)))
            for k in range(n):
                a, b = s0 + (s1 - s0) * k / n, s0 + (s1 - s0) * (k + 1) / n
                piece_dir.append(gt.dir2(w, a, b))
                _fill(piece_id, gt.wall_quad(w, a, b, -w.thickness / 2 - 0.01, w.thickness / 2 + 0.01), frame,
                      len(piece_dir) - 1)
    walls_px = {}
    open_rail = {w.id for w in lay.walls if w.role == "open_balcony"}
    for w in lay.walls:
        if w.id in open_rail:                             # ограждение открытого балкона - улица
            walls_px[w.id] = np.zeros((H, W), bool)
            continue
        s0, s1 = -w.ext0, w.length + w.ext1
        full = gt.wall_quad(w, s0, s1, -w.thickness / 2, w.thickness / 2)
        m = np.zeros((H, W), np.uint8)
        _fill(m, full, frame)
        body |= m
        if w.kind == "exterior":
            a, b = gt.inner_offsets(w)
            m = np.zeros((H, W), np.uint8)
            _fill(m, gt.wall_quad(w, s0, s1, min(a, b), max(a, b)), frame)
        walls_px[w.id] = m.astype(bool)
    # наружная полоса - только у помещений: за внутренним контуром не дальше 10 см
    inside = (rooms > 0).astype(np.uint8)
    for w in lay.walls:
        if w.kind != "exterior":
            inside |= walls_px[w.id].astype(np.uint8)
    d_in = cv2.distanceTransform((1 - inside).astype(np.uint8), cv2.DIST_L2, 5) * frame["pixel_m"]
    for w in lay.walls:
        if w.kind == "exterior":
            walls_px[w.id] &= d_in <= EXTERIOR_BAND_M + frame["pixel_m"]
    for w in lay.walls:
        m = walls_px[w.id]
        sem[m] = WALL
        set_dir(m, w)
    # колонны и пилястры - на плане стены
    for it in lay.items:
        if it.label == "column":
            for b in it.boxes:
                q = _box_quad(b)
                m = np.zeros((H, W), np.uint8)
                _fill(m, gt.poly(q), frame)
                sem[m > 0] = WALL
                # направление - по длинной стороне (пилястра вдоль стены)
                u = 0.0 if b.size[0] >= b.size[1] else math.pi / 2
                c0 = np.asarray(b.center[:2], float)
                e = 0.05 * np.array([math.cos(b.yaw + u), math.sin(b.yaw + u)])
                dv = np.diff(gt.xy([c0 - e, c0 + e]), axis=0)[0]
                t = math.atan2(dv[1], dv[0])
                ori[0][m > 0], ori[1][m > 0] = math.cos(2 * t), math.sin(2 * t)
    # проёмы: след в теле стены (той же глубины, что размеченная стена), по ширине проёма
    for o in lay.openings:
        w = lay.wall(o.wall_id)
        top_wall = w.height if w.height is not None else gt.ceiling_h
        if o.kind == "window":
            cls = WINDOW
        else:
            cls = OPENING if o.z1 >= top_wall - 0.05 else DOOR
        q = gt.wall_quad(w, o.s - o.width / 2, o.s + o.width / 2, -w.thickness / 2 - 0.01, w.thickness / 2 + 0.01)
        m = np.zeros((H, W), np.uint8)
        _fill(m, q, frame)
        m = (m > 0) & (walls_px[w.id] | (sem == WALL))
        sem[m] = cls
        set_dir(m, w)
        hts[0][m] = (0.0 if cls != WINDOW else o.z0) / gt.ceiling_h
        hts[1][m] = min(o.z1, top_wall) / gt.ceiling_h
    # мебель, радиаторы, короба, мусор - поверх свободного
    clutter = np.zeros((H, W), np.uint8)
    for it in lay.items:
        if it.label not in CLUTTER_LABELS or it.kind in CLUTTER_SKIP_KINDS:
            continue
        for b in it.boxes:
            _fill(clutter, gt.poly(_box_quad(b)), frame)
        for pr in it.prims:
            t = pr.get("type")
            from .meshes import prim_mesh
            vertices,triangles=prim_mesh(pr)
            for tri in triangles:
                _fill(clutter,gt.poly(vertices[tri,:2]),frame)
    sem[(clutter > 0) & (sem == INTERIOR)] = CLUTTER
    # проём в перекрытии (верхний этаж) и лестница под ним (нижний) - duplex.py
    for x0, y0, x1, y1 in lay.meta.get("void_rects", []):
        m = np.zeros((H, W), np.uint8)
        _fill(m, gt.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)]), frame)
        sem[(m > 0) & np.isin(sem, (INTERIOR, CLUTTER))] = VOID
    # расстояние до видимой грани: граница тела стены (с проёмами) с помещением
    dist=distance_to_faces(gt,frame)
    # границы классов размечены (пиксель - по своему центру): кольцо ignore по границам лишало
    # сеть сигнала ровно там, где грань стены и откос проёма (датасеты версии 1)
    sem_out = sem.copy()
    mask = np.isin(sem, (WALL, DOOR, WINDOW, OPENING))
    ori[:, ~mask] = 0
    hts[:, ~np.isin(sem, (DOOR, WINDOW, OPENING))] = 0
    return {"sem": sem_out, "sem_raw": sem, "dist": dist, "orient": ori.astype(np.float16),
            "heights": hts.astype(np.float16), "room_id": room_id}


def unseen_to_outside(lab: dict, inp: np.ndarray, px: float, min_seen: float = 0.15, near_m: float = 0.3,
                      no_station=None, policy: str = "outside") -> int:
    """Неотсканированное - улица. Помещение без стоянки (no_station: номера помещений + 1, как в
    room_id) или с видимым полом/потолком меньше чем на min_seen площади, и прилегающие к нему
    стены и проёмы - там, где точек нет ближе near_m, - класс «улица», расстояние до грани -
    «далеко» (1): план строится по облаку, нет точек - нет помещения (ТЗ). С ignore (labels-2,
    3) сеть ничему там не училась и на реальных сканах дорисовывала «помещения» в пустоте у
    края съёмки и обводила их выдуманными стенами. Без стоянки - при любой видимой доле:
    коридор, на четверть видный через дверь, - видимая четверть, а не угаданное целое. Видимые
    места такого помещения и слепые круги под стоянками и тени от мебели в отсканированных
    помещениях остаются размеченными - они окружены увиденным. Возвращает число пикселей,
    ставших улицей."""
    import cv2

    if policy not in ("outside", "ignore"):
        raise ValueError("unseen policy must be outside or ignore")
    x = np.asarray(inp, np.float32)
    ev = ((x[0] > 0) | (x[1] > 0) | (x[2] > 0) | (x[3] > 0) | (x[4] > 0) | (x[5] > 0)).astype(np.uint8)
    seen = (x[3] > 0) | (x[4] > 0)
    if policy == 'ignore':
        near=cv2.dilate(ev,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
            (2*max(1,int(round(near_m/px)))+1,)*2))>0
        unknown=(lab['sem_raw']!=OUTSIDE)&~near
        lab['sem'][unknown]=IGNORE
        return int(unknown.sum())
    rid = lab["room_id"]
    n = int(rid.max()) + 1
    area = np.bincount(rid.ravel(), minlength=n)
    vis = np.bincount(rid.ravel(), seen.ravel(), minlength=n)
    unseen = np.zeros(n, bool)
    unseen[1:] = (area[1:] > 0) & (vis[1:] < min_seen * np.maximum(area[1:], 1))
    for k in no_station or ():
        if 0 < k < n and area[k] > 0:
            unseen[k] = True
    if not unseen.any():
        return 0
    r = max(1, int(round(near_m / px)))
    near = cv2.dilate(ev, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    room_un = unseen[rid]
    adj = cv2.dilate(room_un.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (4 * r + 1, 4 * r + 1))) > 0
    sem = lab["sem"]
    struct = np.isin(lab["sem_raw"], (WALL, DOOR, WINDOW, OPENING))
    out = ~near & ((room_un & (rid > 0)) | (struct & adj)) & (sem != IGNORE) & (sem != OUTSIDE)
    sem[out] = IGNORE if policy == "ignore" else OUTSIDE
    lab["dist"][0][out] = 1
    return int(out.sum())


def _box_quad(b) -> np.ndarray:
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    hx, hy = b.size[0] / 2, b.size[1] / 2
    return np.array([(b.center[0] + u * c - v * s, b.center[1] + u * s + v * c)
                     for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))])


def vector_doc(gt: SceneGT) -> dict:
    """Векторный эталон в мировой СК (тот же формат, что и выход пайплайна)."""
    import cv2

    lay = gt.layout
    arcs = {int(a["wall_id"]): a for a in gt.warp.arcs}

    def along(w, s0: float, s1: float) -> float:
        """Длина образа участка оси стены (по дуге - по дуге)."""
        q = gt.poly([w.point(s0), w.point(s1)], closed=False)
        return float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())

    walls = []
    for w in lay.walls:
        if w.role == "open_balcony":                      # открытый балкон на плане пока не рисуется
            continue
        p0, p1 = gt.xy([w.p0])[0], gt.xy([w.p1])[0]
        geom = {"type": "segment", "p0": p0.round(4).tolist(), "p1": p1.round(4).tolist()}
        if w.id in arcs:                                  # дуговой пролёт: ломаная + окружность
            a = arcs[w.id]
            pts = gt.poly([w.p0, w.p1], closed=False)
            mid = gt.xy([w.point(sum(a["s"]) / 2)])[0]
            ends = gt.xy([w.point(a["s"][0]), w.point(a["s"][1])])
            geom = {"type": "polyline", "points": pts.round(4).tolist(),
                    "arc": {"a": ends[0].round(4).tolist(), "b": ends[1].round(4).tolist(),
                            "mid": mid.round(4).tolist(),
                            "radius": round(float(a["R"] * math.sqrt(abs(np.linalg.det(gt.A)))), 4)}}
        walls.append({"id": f"W{w.id}", "geometry": geom,
                      "thickness": round(float(w.thickness * math.sqrt(abs(np.linalg.det(gt.A)))), 4),
                      "is_exterior": w.kind == "exterior", "kind": w.kind})
    for it in lay.items:
        if it.label != 'column':continue
        for k,b in enumerate(it.boxes):
            q=gt.poly(_box_quad(b))
            for j,(a,bp) in enumerate(zip(q,np.roll(q,-1,axis=0))):
                walls.append({'id':f'C{it.id}_{k}_{j}',
                    'geometry':{'type':'segment','p0':a.tolist(),'p1':bp.tolist()},
                    'thickness':0.,'is_exterior':False,'kind':'column_face'})
    openings = []
    for o in lay.openings:
        w = lay.wall(o.wall_id)
        top_wall = w.height if w.height is not None else gt.ceiling_h
        typ = "window" if o.kind == "window" else ("opening" if o.z1 >= top_wall - 0.05 else "door")
        openings.append({"id": f"O{o.id}", "type": typ, "host_wall_id": f"W{w.id}",
                         "offset": round(along(w, 0.0, o.s - o.width / 2), 4),
                         "width": round(along(w, o.s - o.width / 2, o.s + o.width / 2), 4),
                         "sill_z": round(float(o.z0 if typ == "window" else 0.0), 4),
                         "top_z": round(float(min(o.z1, top_wall)), 4)})
    rooms = []
    for r in lay.rooms:
        if r.open_sky:
            continue
        rects = r.rects()
        x0 = min(q[0] for q in rects)
        y0 = min(q[1] for q in rects)
        x1 = max(q[2] for q in rects)
        y1 = max(q[3] for q in rects)
        step = 0.01
        Wn, Hn = int(math.ceil((x1 - x0) / step)) + 2, int(math.ceil((y1 - y0) / step)) + 2
        m = np.zeros((Hn, Wn), np.uint8)
        for a0, b0, a1, b1 in rects:
            m[int(round((b0 - y0) / step)) + 1:int(round((b1 - y0) / step)) + 1,
              int(round((a0 - x0) / step)) + 1:int(round((a1 - x0) / step)) + 1] = 1
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
        poly = np.c_[x0 + (cnt[:, 0] - 1) * step, y0 + (cnt[:, 1] - 1) * step]
        q = gt.poly(poly)
        area = r.area * abs(np.linalg.det(gt.A)) if gt.warp.linear else \
            0.5 * abs(float(q[:, 0] @ np.roll(q[:, 1], -1) - q[:, 1] @ np.roll(q[:, 0], -1)))
        rooms.append({"id": f"R{r.id}", "kind": r.kind, "name": r.name,
                      "polygon": q.round(4).tolist(), "area": round(float(area), 3)})
    return {"walls": walls, "openings": openings, "rooms": rooms}


# ----------------------------------------------------------------------
# Запись
# ----------------------------------------------------------------------

def _preview(inp: np.ndarray, sem: np.ndarray | None, path: Path) -> None:
    """Проверка совмещения: вход (низ - красный, верх - зелёный, пол - синий) и поверх -
    контуры классов разметки."""
    import cv2

    rgb = np.stack([inp[0], inp[2], inp[3] * 0.6], -1).astype(np.float32)
    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if sem is not None:
        colors = {WALL: (255, 255, 255), DOOR: (255, 140, 0), WINDOW: (0, 220, 220), OPENING: (180, 80, 255),
                  CLUTTER: (255, 230, 0), VOID: (255, 0, 160)}
        for c, col in colors.items():
            m = (sem == c).astype(np.uint8)
            edge = m - cv2.erode(m, np.ones((3, 3), np.uint8))
            img[edge > 0] = col
    cv2.imwrite(str(path), img[..., ::-1])


def _split_of(key: str) -> str:
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def write_meta(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    from .dataset_contract import atomic_json
    existing=root/'meta.json'
    if existing.exists() and (root/'scenes').exists() and any((root/'scenes').iterdir()):
        if json.loads(existing.read_text()).get('labels')!=LABELS_VERSION:
            raise ValueError('Label contract changed; use a new output dataset')
    atomic_json(existing,meta_doc())


def validate_supervision_policy(root, policy):
    """Refuse to resume a dataset using a different supervision contract."""
    if policy not in ("outside", "ignore"):
        raise ValueError("unseen policy must be outside or ignore")
    root = Path(root)
    marker = root / "supervision_policy.json"
    if marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))["unseen_policy"]
        if previous != policy:
            raise ValueError("Supervision policy changed; use a new output dataset")
        return
    elif (root / "scenes").exists() and any((root / "scenes").iterdir()) and policy != "outside":
        raise ValueError("Existing unversioned dataset; use a new output for ignore policy")
    root.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"unseen_policy": policy, "contract_version": 1}), encoding="utf-8")


def build_scene(scene_dir, root, px: float = PIXEL_M, use_gt_levels: bool = False, thin: float = 0.0,
                z_shift: float = 0.0, splits: bool = True, source=None, rot_deg: float = 0.0,
                variant: str = "", log=print, unseen_policy: str = "ignore",
                preview: bool = True, max_level_error_m: float = .03) -> list[str]:
    """Сцена генератора -> dataset/scenes/<scene><variant>/<level>/. Вход - тем же
    растеризатором, что и для реальных файлов (из scan.e57 или, без записи на диск, из
    source=MemorySource); уровни - detect_levels, как у реальных (или эталонные, use_gt_levels).
    thin, z_shift, rot_deg - аугментации облака; variant - суффикс каталога для нескольких
    растров одного скана. splits=False - не трогать splits/*.txt (их ведёт вызывающий).
    Возвращает записанные каталоги."""
    root = Path(root)
    validate_supervision_policy(root, unseen_policy)
    write_meta(root)
    gt0 = SceneGT(scene_dir)
    gts = [gt0] + [SceneGT(scene_dir, k) for k in range(1, len(gt0.levels))]
    levels = [{"floor_z": g.floor_z, "ceiling_z": g.floor_z + g.ceiling_h,
               "height_m": g.ceiling_h} for g in gts] if use_gt_levels else None
    meta = gt0.doc.get("meta", {})
    seed, index = meta.get("seed", 0), meta.get("index", 0)
    own = source is None
    if own:
        e57 = gt0.dir / "scan.e57"
        source = E57Source(e57 if e57.exists() else gt0.dir / "scan_merged.e57")
    rot = None
    if rot_deg:
        st = np.array([s_["pose_true"]["t"] for s_ in gt0.doc.get("stations", [])], float).reshape(-1, 3)
        rot = (float(rot_deg), float(st[:, 0].mean()), float(st[:, 1].mean()))
    for g in gts:
        g.rot = rot
    rs = int(seed) * 1000 + int(index) + zlib.crc32(variant.encode()) % 1000
    # пустые поля вокруг квартиры: сеть учится, что пустое без точек - улица, а не помещение
    # (на реальных сканах она дорисовывала помещения в пустоте рядом с домом)
    pad = np.random.default_rng(rs + 7).uniform(0.5, 2.5, 4)
    try:
        res = rasterize_points(source, px=px, levels=levels, thin=thin, reference_shift=z_shift,
                               seed=rs, rot=rot, pad_m=pad, log=log)
    finally:
        if own:
            source.close()
    # каждому этажу эталона - ближайший по полу найденный уровень (не дальше 0,5 м)
    pairs = []
    for g in gts:
        r = min(res, key=lambda r: abs(r["level"]["floor_z"] - g.floor_z))
        if abs(r["level"]["floor_z"] - g.floor_z) < 0.5 and all(r is not q for _, q in pairs):
            pairs.append((g, r))
    if len(pairs)!=len(gts):raise ValueError('Missing or duplicate detected level')
    for gt,r in pairs:
        # Validate detection independently of the intentional reference perturbation.
        fl=r['level']['floor_z']-z_shift
        ce=r['level']['ceiling_z']-z_shift
        if max(abs(fl-gt.floor_z),abs(ce-gt.floor_z-gt.ceiling_h))>max_level_error_m:
            raise ValueError(f'Level reference error exceeds {max_level_error_m} m')
    scene_id = gt0.dir.name + variant
    out_dirs = []
    for k, (gt, r) in enumerate(pairs):
        d = root / "scenes" / scene_id / f"L{gt.level + 1}"
        d.mkdir(parents=True, exist_ok=True)
        (d/'complete.json').unlink(missing_ok=True)
        lab = label_level(gt, r["frame"])
        # Heights are interpreted relative to the estimated level at inference.
        op=np.isin(lab['sem_raw'],(DOOR,WINDOW,OPENING))
        estimated_h=r['level']['ceiling_z']-r['level']['floor_z']
        lab['heights'][:,op]=np.clip((lab['heights'][:,op].astype(float)*gt.ceiling_h+
            gt.floor_z-r['level']['floor_z'])/estimated_h,0,1)
        physical_semantics = lab["sem"].copy()
        n_out = unseen_to_outside(lab, r["input"], px, no_station=[q.id + 1 for q in gt.layout.rooms if not q.scanned], policy=unseen_policy)
        np.save(d / "input.npy", r["input"])
        import cv2

        cv2.imwrite(str(d / "sem.png"), lab["sem"])
        np.save(d / "supervision_valid.npy", lab["sem"] != IGNORE)
        cv2.imwrite(str(d / "semantic_truth.png"), physical_semantics)
        np.save(d / "observation_support.npy", np.any(r["input"][:6] > 0, axis=0))
        np.save(d / "dist.npy", lab["dist"])
        np.save(d / "orient.npy", lab["orient"])
        np.save(d / "heights.npy", lab["heights"])
        (d / "vector.json").write_text(json.dumps(vector_doc(gt), indent=1, ensure_ascii=False), encoding="utf-8")
        sc = {"scene_id": scene_id, "level": f"L{gt.level + 1}", "levels_in_scene": len(gts), "layout_key": f"{meta.get('seed')}:{meta.get('index')}",
              "seed": seed, "index": index, "effects": meta.get("effects_sampled", {}),
              "realism": meta.get("realism", {}), "stations_world": gt.stations().round(4).tolist(),
              "floor_z": r["level"]["floor_z"], "ceiling_z": r["level"]["ceiling_z"],
              "floor_z_gt": gt.floor_z, "ceiling_z_gt": gt.floor_z + gt.ceiling_h,
              "frame": r["frame"], "pixel_to_world": affine(r["frame"]), "rasterizer": RASTERIZER_VERSION,
              "bare": meta.get("bare"), "aug": {"thin": thin, "z_shift": z_shift, "rot_deg": rot_deg},
              "labels": LABELS_VERSION, "unseen_policy": unseen_policy, "unseen_changed_px": n_out, "unseen_outside_px": n_out if unseen_policy == "outside" else 0}
        (d / "scene.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        from .dataset_contract import atomic_json,finish_level
        physical=vector_doc(gt)
        atomic_json(d/'vector_physical.json',physical)
        physical['evaluation_scope']={'frame':r['frame'],
            'valid_runs':mask_runs(lab['sem']!=IGNORE)}
        atomic_json(d/'vector.json',physical)
        if preview:_preview(r['input'].astype(np.float32),lab['sem_raw'],d/'preview.png')
        finish_level(d)
        out_dirs.append(str(d))
        if splits:
            _append_split(root, _split_of(sc["layout_key"]), f"{scene_id}/L{gt.level + 1}")
    return out_dirs


def _append_split(root: Path, split: str, entry: str) -> None:
    sd = root / "splits"
    sd.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        f = sd / f"{name}.txt"
        lines = f.read_text(encoding="utf-8").split() if f.exists() else []
        lines = [x for x in lines if x != entry]
        if name == split:
            lines.append(entry)
        f.write_text("\n".join(sorted(set(lines))) + ("\n" if lines else ""), encoding="utf-8")


def build_real(e57, root, object_id: str | None = None, px: float = PIXEL_M, log=print) -> list[str]:
    """Реальный E57 -> dataset/real/<object>/<level>/input.npy (без разметки)."""
    from .cellplan import image_stations

    root = Path(root)
    write_meta(root)
    e57 = Path(e57)
    oid = object_id or e57.stem
    res = rasterize_e57(e57, px=px, log=log)
    st = image_stations(e57)
    out = []
    for k, r in enumerate(res, 1):
        d = root / "real" / oid / f"L{k}"
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "input.npy", r["input"])
        sc = {"object_id": oid, "level": f"L{k}", "source": e57.name, "floor_z": r["level"]["floor_z"],
              "ceiling_z": r["level"]["ceiling_z"], "stations_world": st.round(4).tolist(), "frame": r["frame"],
              "pixel_to_world": affine(r["frame"]), "rasterizer": RASTERIZER_VERSION}
        (d / "scene.json").write_text(json.dumps(sc, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
        _preview(r["input"].astype(np.float32), None, d / "preview.png")
        out.append(str(d))
    return out



def mask_runs(mask):
    flat=np.r_[False,mask.ravel(),False].astype(np.int8)
    edges=np.flatnonzero(np.diff(flat))
    return edges.reshape(-1,2).tolist()


def accumulate_voxels(vox,indices):
    # Bound temporary dense counts to 16 MB. Sparse scenes retain sorted counting.
    if len(vox)<=2_000_000 and len(vox)<=4*len(indices):
        counts=np.bincount(indices,minlength=len(vox))
        vox[:]=np.minimum(vox.astype(np.uint32)+counts,65535).astype(np.uint16)
    else:
        u,c=np.unique(indices,return_counts=True)
        vox[u]=np.minimum(vox[u].astype(np.int64)+c,65535).astype(np.uint16)
