"""Вход сети из облака: псевдочертёж по свободному пространству (шаг 2 конвертера, раздел 5.5 ТЗ).

U-Net ReFloorBRUSNIKA обучена на чертежах: чёрные стены на белом. Облако приводится к такому
же виду без нейросети, только по лучам сканера:

  1. облако E57 переводится в СК планировки; пол и потолок каждой станции - пики гистограммы z;
  2. на станцию - полярные корзины по 0,5 градуса. Дальность корзины - k-я по величине
     горизонтальная дальность точек пола и потолка (k-я, а не максимум: одиночные выбросы,
     фасады за окном). Пол заслоняет мебель, потолок - почти никогда, поэтому берутся оба;
     но пол не дальше потолка + 0,6 м: за окном бывает земля на уровне пола, потолка там нет.
     Затем граница дотягивается до точек стены в пределах 0,25 м (пол у стены редкий).
     Пустые корзины до 3 градусов заполняются по соседям; многоугольник с отступом 2 см;
  3. объединение по станциям = свободное пространство F;
  4. здание B = закрытие F ядром 0,6 м (перегородки и несущие стены тоньше 0,6 м заливаются).
     Внутренние стены = B без F. Оставшиеся дыры шире 0,6 м - не стены, а то, куда лучи
     не попали (тень за углом, неотсканированная комната): «неизвестно». Наружные стены - кольцо 0,4 м вокруг B (за наружную стену сканер не видит,
     толщина принимается по умолчанию, как в ТЗ);
  5. признак окна: в корзине есть точки дальше границы (улица, соседний дом за стеклом).

Всё рисуется в кадре эталона U-Net (gt.json -> unet.frame), поэтому вход и маска совпадают
пиксель в пиксель. В конвертере кадр будет свой (оси по главным направлениям стен,
масштаб - стена 30 px), но вид входа тот же.

    python -m simscan pair D:/synth/scene_00000   ->  input/free_space.png, valid.png, pair.png

input/valid.png - где скан что-то видел (здание из свободного пространства + кольцо + 0,3 м).
Вне её эталон есть, а данных нет (неотсканированная комната) - это маска для функции потерь.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .groundtruth import RasterFrame
from .transform import WorldTransform

INPUT_VALUES = {"free": 255, "outside": 255, "unknown": 215, "window": 140, "wall": 0}


def _disk(radius_px: float):
    import cv2

    r = max(1, int(round(radius_px)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _plane_z(rel: np.ndarray, lo: float, hi: float) -> float | None:
    """Пик гистограммы высот (шаг 1 см) в окне [lo, hi] относительно станции."""
    m = (rel > lo) & (rel < hi)
    if m.sum() < 100:
        return None
    h, e = np.histogram(rel[m], bins=np.arange(lo, hi + 0.01, 0.01))
    k = int(h.argmax())
    return float((e[k] + e[k + 1]) / 2)


def station_polygon(c, pts: np.ndarray, floor_z: float, ceil_z: float | None, n_bins: int = 720,
                    k: int = 3, band: float = 0.04, inset: float = 0.02, max_gap: int = 6,
                    beyond: float = 0.3, ceil_margin: float = 0.6, wall_reach: float = 0.25):
    """Дальности корзин (n_bins,) и флаги «есть точки за границей» (n_bins,)."""
    d = pts[:, :2] - np.asarray(c[:2])
    r = np.hypot(d[:, 0], d[:, 1])
    b = ((np.arctan2(d[:, 1], d[:, 0]) + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins

    def kth(sel):
        rb = np.zeros(n_bins)
        cnt = np.bincount(b[sel], minlength=n_bins)
        if sel.any():
            order = np.lexsort((r[sel], b[sel]))
            ends = np.cumsum(cnt)
            ok = cnt >= k
            rb[ok] = r[sel][order][ends[ok] - k]
        return rb

    r_floor = kth(np.abs(pts[:, 2] - floor_z) < band)
    if ceil_z is None:
        rb = r_floor
    else:
        # пол дальше потолка более чем на ceil_margin - это земля за окном, а не комната
        # (снаружи нет потолка); запас - на короба и подвесные потолки у стен
        r_ceil = kth(np.abs(pts[:, 2] - ceil_z) < band)
        rb = np.where(r_ceil > 0, np.maximum(r_ceil, np.minimum(r_floor, r_ceil + ceil_margin)),
                      r_floor)
    # дотянуть до стены: при крупном шаге точки пола у стены редкие (на 4 м при шаге 0,5° -
    # через 10 см), а стена рядом с границей плотная. Дальше wall_reach от границы не берём
    reach = (rb[b] > 0) & (r <= rb[b] + wall_reach)
    rb = np.maximum(rb, kth(reach))
    # короткие пропуски - по соседям (минимум из двух сторон: осторожно)
    empty = rb <= 0
    if empty.any() and not empty.all():
        idx = np.arange(n_bins)
        filled = rb.copy()
        for i in idx[empty]:
            left = right = None
            for s in range(1, max_gap + 1):
                if left is None and rb[(i - s) % n_bins] > 0:
                    left = rb[(i - s) % n_bins]
                if right is None and rb[(i + s) % n_bins] > 0:
                    right = rb[(i + s) % n_bins]
            if left is not None and right is not None:
                filled[i] = min(left, right)
        rb = filled
    rb = np.maximum(rb - inset, 0)
    # точки дальше границы: стекло, открытое окно, зеркало (мнимые точки)
    far = r > rb[b] + beyond
    beyond_cnt = np.bincount(b[far], minlength=n_bins)
    return rb, beyond_cnt >= k


def _sector_poly(frame: RasterFrame, c, r0, r1, n_bins: int, bins) -> list:
    """Многоугольники секторов-колец [r0, r1] по корзинам (в пикселях, для cv2.fillPoly)."""
    polys = []
    step = 2 * math.pi / n_bins
    for i in bins:
        a0, a1 = -math.pi + i * step, -math.pi + (i + 1) * step
        pts = [(c[0] + r0[i] * math.cos(a0), c[1] + r0[i] * math.sin(a0)),
               (c[0] + r1[i] * math.cos(a0), c[1] + r1[i] * math.sin(a0)),
               (c[0] + r1[i] * math.cos(a1), c[1] + r1[i] * math.sin(a1)),
               (c[0] + r0[i] * math.cos(a1), c[1] + r0[i] * math.sin(a1))]
        ii, jj = frame.xy_to_ij(*np.array(pts).T)
        polys.append(np.round(np.c_[jj, ii] * 16).astype(np.int32))
    return polys


def free_space_input(stations: list[dict], frame: RasterFrame, n_bins: int = 720,
                     close_m: float = 0.6, ext_wall_m: float = 0.4) -> dict:
    """stations: [{"center": (x, y, z), "points": (n, 3)}] в СК планировки."""
    import cv2

    H, W = frame.height, frame.width
    px = frame.pixel_m
    free = np.zeros((H, W), np.uint8)
    window = np.zeros((H, W), np.uint8)
    info = []
    for st in stations:
        c, p = np.asarray(st["center"], float), st["points"]
        rel = p[:, 2] - c[2]
        fz, cz = _plane_z(rel, -2.5, -0.3), _plane_z(rel, 0.3, 4.0)
        if fz is None:
            continue
        rb, far = station_polygon(c, p, c[2] + fz, None if cz is None else c[2] + cz, n_bins)
        step = 2 * math.pi / n_bins
        ang = -math.pi + np.repeat(np.arange(n_bins), 2) * step + np.tile([0, step], n_bins)
        rr = np.repeat(rb, 2)
        ii, jj = frame.xy_to_ij(c[0] + rr * np.cos(ang), c[1] + rr * np.sin(ang))
        cv2.fillPoly(free, [np.round(np.c_[jj, ii] * 16).astype(np.int32)], 1, cv2.LINE_8, 4)
        if far.any():
            cv2.fillPoly(window, _sector_poly(frame, c, rb, rb + ext_wall_m + 0.05, n_bins,
                                              np.flatnonzero(far)), 1, cv2.LINE_8, 4)
        info.append({"floor_rel_m": round(fz, 3), "ceiling_rel_m": None if cz is None else round(cz, 3),
                     "bins_empty": int((rb <= 0).sum())})
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(0.03 / px))      # щели между секторами
    return _compose(free, window, px, close_m, ext_wall_m, info)


def _compose(free, window, px: float, close_m: float, ext_wall_m: float, info) -> dict:
    """Свободное пространство -> здание, стены, кольцо наружных стен, «неизвестно», вход."""
    import cv2

    H, W = free.shape
    building = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(close_m / 2 / px))
    # дыры, которые закрытие не залило (шире close_m) - «неизвестно»
    n, lab, stats, _ = cv2.connectedComponentsWithStats((building == 0).astype(np.uint8), 4)
    unknown = np.zeros_like(building)
    for k in range(1, n):
        x, y, w, h, _ = stats[k]
        if not (x == 0 or y == 0 or x + w >= W or y + h >= H):            # не снаружи здания
            unknown[lab == k] = 1
    building = building | unknown
    ring = cv2.dilate(building, _disk(ext_wall_m / px)) & (1 - building)
    walls = (building & (1 - free) & (1 - unknown)) | ring
    img = np.full((H, W), INPUT_VALUES["outside"], np.uint8)
    img[unknown.astype(bool)] = INPUT_VALUES["unknown"]
    img[walls.astype(bool)] = INPUT_VALUES["wall"]
    win = window.astype(bool) & walls.astype(bool)
    img[win] = INPUT_VALUES["window"]
    # зона, где скан что-то видел: здание с кольцом + 0,3 м. За ней эталон не проверяем
    # (неотсканированные комнаты), при обучении - маска потерь
    valid = cv2.dilate(building | ring, _disk(0.3 / px)) & (1 - unknown)
    return {"image": img, "free": free.astype(bool), "walls": walls.astype(bool),
            "window": win, "unknown": unknown.astype(bool), "valid": valid.astype(bool),
            "stations": info}


def floor_ceiling(z: np.ndarray) -> tuple[float | None, float | None]:
    """Пол и потолок сведённого облака: пики гистограммы z в нижней и верхней трети."""
    if len(z) < 1000:
        return None, None
    lo, hi = np.percentile(z, 0.5), np.percentile(z, 99.5)
    h, e = np.histogram(z, bins=np.arange(lo, hi + 0.01, 0.01))
    c = (e[:-1] + e[1:]) / 2
    third = (hi - lo) / 3
    fl = c[c < lo + third][h[c < lo + third].argmax()]
    ce = c[c > hi - third][h[c > hi - third].argmax()]
    return float(fl), float(ce)


def detect_levels(points: np.ndarray, min_share: float = 0.15, min_height: float = 2.0,
                  max_height: float = 4.5, stations_z=None) -> list[dict]:
    """Уровни сведённого облака: пары (пол, потолок) из горизонтальных плоскостей.

    Плоскость - пик гистограммы z (шаг 2 см), площадь - по ячейкам 25 см. Берутся пики
    с числом точек не меньше min_share от наибольшего, пики ближе 10 см сливаются. Снизу
    вверх: пол, затем самая плотная плоскость выше на min_height..max_height - потолок; следующая
    плоскость над потолком - пол следующего уровня (перекрытие)."""
    z = points[:, 2]
    if len(z) < 1000:
        return []
    # запас по краям: пол бывает самыми нижними точками облака, потолок - самыми верхними
    h, e = np.histogram(z, bins=np.arange(z.min() - 0.1, z.max() + 0.12, 0.02))
    c = (e[:-1] + e[1:]) / 2
    base = np.median(h[h > 0]) if (h > 0).any() else 0     # пустые столбцы (земля под окнами) не в счёт
    peaks = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1]
             and h[i] > 3 * base]
    planes = []
    for i in peaks:
        m = np.abs(z - c[i]) < 0.03
        cells = np.unique(np.floor(points[m, :2] / 0.25).astype(np.int64), axis=0)
        planes.append({"z": float(c[i]), "area": len(cells) * 0.0625, "n": int(h[i])})
    if not planes:
        return []
    if stations_z is not None and len(stations_z):
        # опора на стоянки: пол - самая плотная плоскость на 0,6-1,9 м ниже стоянки (грунт за
        # окнами бывает плотнее пола и иначе отсеивает его), потолок - самая плотная выше пола
        levels = []
        for sz in sorted(stations_z):
            fls = [p for p in planes if 0.6 <= sz - p["z"] <= 1.9]
            if not fls:
                continue
            fl = max(fls, key=lambda q: q["n"])
            ces = [q for q in planes if min_height <= q["z"] - fl["z"] <= max_height and q["z"] > sz]
            if not ces:
                continue
            ce = max(ces, key=lambda q: q["n"])
            if any(abs(lv["floor_z"] - fl["z"]) < 0.3 for lv in levels):
                continue
            levels.append({"floor_z": round(fl["z"], 3), "ceiling_z": round(ce["z"], 3),
                           "height_m": round(ce["z"] - fl["z"], 3), "floor_area_m2": round(fl["area"], 1)})
        if levels:
            return sorted(levels, key=lambda lv: lv["floor_z"])
    # отбор по числу точек, не по площади: земля за окнами занимает большую площадь,
    # но точек на ней мало (далеко и под острым углом)
    top = max(p["n"] for p in planes)
    planes = [p for p in planes if p["n"] >= min_share * top]
    merged = []
    for p in sorted(planes, key=lambda p: p["z"]):
        if merged and p["z"] - merged[-1]["z"] < 0.1:
            if p["n"] > merged[-1]["n"]:
                merged[-1] = p
            continue
        merged.append(p)
    levels, k = [], 0
    while k < len(merged) - 1:
        fl = merged[k]
        cand = [q for q in merged[k + 1:] if min_height <= q["z"] - fl["z"] <= max_height]
        # потолок - самая большая плоскость в диапазоне высот (не ближайшая: короба, верх проёмов)
        ce = max(cand, key=lambda q: q["n"]) if cand else None
        if ce is None:
            k += 1
            continue
        levels.append({"floor_z": round(fl["z"], 3), "ceiling_z": round(ce["z"], 3),
                       "height_m": round(ce["z"] - fl["z"], 3),
                       "floor_area_m2": round(fl["area"], 1)})
        k = merged.index(ce) + 1
    return levels


def coverage_input(points: np.ndarray, frame: RasterFrame, floor_z: float, ceil_z: float,
                   band: float = 0.04, close_m: float = 0.6, ext_wall_m: float = 0.4,
                   spacing_m: float = 0.015) -> dict:
    """Вход сети для сведённого облака (одно облако без станций и сетки, как экспорт
    Cyclone REGISTER 360): свободно - где есть точки пола или потолка. Потолок виден почти
    везде (мебель его не закрывает), над стенами и в толще перегородок точек потолка нет -
    там и получаются стены; пол добавляет дверные проёмы (под перемычкой потолка нет)."""
    import cv2

    H, W = frame.height, frame.width
    sel = (np.abs(points[:, 2] - floor_z) < band) | (np.abs(points[:, 2] - ceil_z) < band)
    i, j = frame.xy_to_ij(points[sel, 0], points[sel, 1])
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    free = np.zeros((H, W), np.uint8)
    free[i[ok], j[ok]] = 1
    # дырки покрытия закрываются на 2 шага точек, не больше: щель над перегородкой 8-10 см
    # должна остаться (при 3 см её заливало - тонкие перегородки пропадали)
    free = cv2.morphologyEx(free, cv2.MORPH_CLOSE, _disk(max(1.0, 2 * spacing_m / frame.pixel_m)))
    info = [{"floor_z": round(floor_z, 3), "ceiling_z": round(ceil_z, 3)}]
    return _compose(free, np.zeros_like(free), frame.pixel_m, close_m, ext_wall_m, info)


def load_stations(scene_dir: Path, world: WorldTransform, max_points: int = 3_000_000) -> list[dict]:
    from .e57io import read_e57_world

    out = []
    for s in read_e57_world(scene_dir / "scan.e57"):
        xyz = s["xyz"]
        c = world.to_layout(np.asarray(s["position"], float)[None])[0]
        p = world.to_layout(xyz[np.isfinite(xyz).all(1)])
        p = p[np.linalg.norm(p - c, axis=1) > 0.05]                     # недействительные - в нуле
        if len(p) > max_points:
            p = p[np.random.default_rng(0).choice(len(p), max_points, replace=False)]
        out.append({"center": c, "points": p})
    return out


def wall_scores(pred_walls: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None) -> dict:
    gt = (mask == 64) | (mask == 128)
    out = {}
    for suffix, m in (("", None), ("_scanned", valid)):
        if suffix and m is None:
            continue
        p, g = (pred_walls, gt) if m is None else (pred_walls & m, gt & m)
        inter, union = (p & g).sum(), (p | g).sum()
        out.update({f"wall_iou{suffix}": round(float(inter / max(union, 1)), 4),
                    f"wall_precision{suffix}": round(float(inter / max(p.sum(), 1)), 4),
                    f"wall_recall{suffix}": round(float(inter / max(g.sum(), 1)), 4)})
    return out


def pair_data(scene_dir) -> dict:
    """Вход, эталон и всё нужное для отрисовки; файлы input/* пишутся здесь."""
    import cv2

    scene_dir = Path(scene_dir)
    info = json.loads((scene_dir / "gt" / "gt.json").read_text(encoding="utf-8"))
    if "unet" not in info:
        raise ValueError("в сцене нет эталона U-Net (export.unet_mask)")
    frame = RasterFrame(**info["unet"]["frame"])
    w = info["layout_to_world"]
    world = WorldTransform(w["yaw_deg"], tuple(w["offset_m"]))
    stations = load_stations(scene_dir, world)
    res = free_space_input(stations, frame)
    out = scene_dir / "input"
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "free_space.png"), cv2.merge([res["image"]] * 3))
    cv2.imwrite(str(out / "valid.png"), res["valid"].astype(np.uint8) * 255)
    mask = cv2.imread(str(scene_dir / "gt" / info["unet"]["file"]), cv2.IMREAD_GRAYSCALE)
    stats = {"frame_px": [frame.height, frame.width], "pixel_mm": round(frame.pixel_m * 1000, 2),
             "stations": res["stations"], **wall_scores(res["walls"], mask, res["valid"])}
    (out / "input.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"name": scene_dir.name, "stations": stations, "frame": frame, "res": res,
            "mask": mask, "stats": stats}


def make_pair(scene_dir, figure: bool = True) -> dict:
    d = pair_data(scene_dir)
    if figure:
        pair_sheet([d], Path(scene_dir) / "input" / "pair.png")
    return d["stats"]


# ----------------------------------------------------------------------
# Отрисовка: строка на сцену, столбцы всегда одни и те же:
#   облако (срез 1,0-1,6 м) | вход | эталон | разница стен
# Вход и эталон в одной палитре: стена - чернила, окно - бирюза, дверь - оранжевый,
# не видно - светло-серый. Разница: совпало - серый, лишняя - красный, пропущена - синий.
# ----------------------------------------------------------------------

def _slice_density(stations, frame: RasterFrame) -> np.ndarray:
    """Плотность точек полосы 1,0-1,6 м над полом станции (пол - пик гистограммы)."""
    parts = []
    for st in stations:
        c, p = np.asarray(st["center"]), st["points"]
        fz = _plane_z(p[:, 2] - c[2], -2.5, -0.3)
        if fz is None:
            continue
        z = p[:, 2] - (c[2] + fz)
        parts.append(p[(z > 1.0) & (z < 1.6)])
    return density_image(np.concatenate(parts) if parts else np.zeros((0, 3)), frame)


def density_image(q: np.ndarray, frame: RasterFrame) -> np.ndarray:
    """Логарифм плотности точек в пикселях кадра, нормированный к 0..1."""
    import cv2

    H, W = frame.height, frame.width
    dens = np.zeros((H, W))
    i, j = frame.xy_to_ij(q[:, 0], q[:, 1])
    i, j = np.round(i).astype(int), np.round(j).astype(int)
    ok = (i >= 0) & (i < H) & (j >= 0) & (j < W)
    np.add.at(dens, (i[ok], j[ok]), 1)
    dens = cv2.dilate(np.log1p(dens), np.ones((3, 3)))
    top = np.percentile(dens[dens > 0], 95) if (dens > 0).any() else 1.0
    return np.clip(dens / max(top, 1e-9), 0, 1)


def _panels(d: dict):
    from matplotlib.colors import to_rgb

    from .debug import AQUA, BLUE, INK, ORANGE, RED, SURFACE

    rgb = lambda c: np.array(to_rgb(c))  # noqa: E731
    mask, res = d["mask"], d["res"]
    H, W = mask.shape
    blank = np.ones((H, W, 3)) * rgb(SURFACE)

    cloud = blank * (1 - _slice_density(d["stations"], d["frame"])[..., None] * 0.85)
    inp = blank.copy()
    inp[res["unknown"]] = rgb(UNKNOWN)
    inp[res["walls"]] = rgb(INK)
    inp[res["window"]] = rgb(AQUA)
    tgt = blank.copy()
    tgt[mask == 64] = rgb(INK)
    tgt[mask == 128] = rgb(AQUA)
    tgt[mask == 192] = rgb(ORANGE)
    gt = (mask == 64) | (mask == 128)
    pw = res["walls"]
    diff = blank.copy()
    diff[pw & gt] = rgb(MATCH)
    diff[pw & ~gt] = rgb(RED)
    diff[gt & ~pw] = rgb(BLUE)
    out = ~res["valid"]
    diff[out] = 0.4 * diff[out] + 0.6 * rgb(SURFACE)                    # вне скана - бледно

    # общая обрезка по зданию (эталон и вход) с полем 3 %
    ys, xs = np.nonzero(gt | (mask > 0) | pw)
    m = int(0.03 * max(H, W))
    crop = (slice(max(ys.min() - m, 0), min(ys.max() + m + 1, H)),
            slice(max(xs.min() - m, 0), min(xs.max() + m + 1, W)))
    return [x[crop] for x in (cloud, inp, tgt, diff)], crop


UNKNOWN, MATCH = "#dcdad5", "#a9a7a1"
COLUMNS = ("Облако 1,0–1,6 м", "Вход", "Эталон", "Разница стен")


def pair_sheet(items: list[dict], path: Path) -> None:
    """Сетка пар: строка на сцену, 4 одинаковых столбца, одна легенда внизу."""
    from matplotlib.patches import Patch

    from .debug import AQUA, BLUE, INK, INK2, ORANGE, RED, _plt

    plt = _plt()
    n = len(items)
    fig, axes = plt.subplots(n, 4, figsize=(15, 3.9 * n + 0.6), squeeze=False,
                             gridspec_kw={"wspace": 0.03, "hspace": 0.08})
    for r, d in enumerate(items):
        imgs, crop = _panels(d)
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img, interpolation="antialiased")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(COLUMNS[c], loc="left", fontsize=11, fontweight="normal", color=INK2)
        for st in d["stations"]:                                       # станции на облаке
            i, j = d["frame"].xy_to_ij(st["center"][0], st["center"][1])
            axes[r, 0].plot(j - crop[1].start, i - crop[0].start, "o", ms=5, mfc="white",
                            mec=INK, mew=1.0)
        s = d["stats"]
        axes[r, 0].set_ylabel(f"{d['name'].replace('scene_', '')}\nIoU {s['wall_iou_scanned']:.2f}", rotation=0,
                              ha="right", va="center", fontsize=10, color=INK2, labelpad=8)
    handles = [Patch(color=INK, label="стена"), Patch(color=AQUA, label="окно"),
               Patch(color=ORANGE, label="дверь"), Patch(color=UNKNOWN, label="не видно"),
               Patch(color=MATCH, label="совпало"), Patch(color=RED, label="лишняя"),
               Patch(color=BLUE, label="пропущена"),
               plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK, label="станция")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=10, handlelength=1.2, columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.06, right=0.995, top=1 - 0.35 / (3.9 * n + 0.6),
                        bottom=0.45 / (3.9 * n + 0.6))
    fig.savefig(path, dpi=90)
    plt.close(fig)
