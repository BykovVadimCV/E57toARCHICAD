"""Тепловая карта стен: наложение горизонтальных срезов облака с весами.

    python -m simscan wall-heatmap D:/scans/kvartira.e57 --out D:/heat
    -> heat.png (на уровень - heat_L1.png ...), slices.png, labels.png (стена / дверь / окно /
       шум), heat_full.png и labels_full.png (полное разрешение: пиксель = пиксель растра),
       heat.npy, labels.npy (0 пусто, 1 стена, 2 дверь, 3 окно, 4 шум), weights.json

Разрешение - --px (по умолчанию 1 см). Предел - шаг точек: у BLK360 после Cyclone ~4 мм, так
что 5 мм ещё имеет смысл; мельче - пиксели стены начинают пустеть.

Срез - точки в слое высоты (по умолчанию 10 см) над полом уровня, растр занятости. Стена
вертикальна: она есть почти во всех срезах на одном и том же месте. Шум и бред - нет: мебель
и радиаторы - в нескольких нижних срезах, светильники и короба - в верхних, пол и потолок
заливают срез целиком, окна и двери выпадают из средних срезов.

Вес среза - насколько его картина совпадает с устойчивой по высоте структурой:
  точность - доля занятого в срезе, что лежит на структуре (мало - шум, заливка);
  полнота  - доля структуры, что видна в срезе (мало - окна, проёмы, тень);
  качество = F1 (гармоническое среднее);
вес = априорный вес по высоте x качество. По высоте (по умолчанию): у пола 1 - там стена
сплошная, середина 1,0-2,0 м - 0,1 - там окна, верх - 0,6. Задаётся --prior.
Структура - пиксели, занятые хотя бы в 4 срезах (40 см
по высоте; с весами - на каждом проходе заново). Не «в половине срезов»: иначе стена под
окном и над ним выпадает из структуры и средние срезы, где окна, ничего не теряют.

Карта = сумма весов срезов, где пиксель занят, / сумма весов: 1 - стена во всю высоту,
~0,6 - стена с окном, ~0,3 - перемычка над дверью, около 0 - шум.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _frame(path, px: float, chunk: int, sample_points: int, log):
    """Уровни, поворот вдоль стен и кадр растра - по выборке точек."""
    from .e57read import E57Reader
    from .floorplan import _scan_points, dominant_angle
    from .rasterize import detect_levels, floor_ceiling

    with E57Reader(path) as r:
        total = sum(s.points for s in r.scans)
        p_keep = min(1.0, sample_points / max(total, 1))
        rng = np.random.default_rng(0)
        sample = []
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                sample.append(p[rng.random(len(p)) < p_keep].astype(np.float32))
    sample = np.concatenate(sample).astype(float)
    from .netinput import station_positions

    with E57Reader(path) as r:
        st = station_positions(r)
    levels = detect_levels(sample, stations_z=st[:, 2] if len(st) else None)
    if not levels:
        fl, ce = floor_ceiling(sample[:, 2])
        levels = [{"floor_z": fl, "ceiling_z": ce, "height_m": ce - fl}]
    from .netinput import frame_points

    hz = sample[:, 2] - levels[0]["floor_z"]
    near = frame_points(sample, levels, st)
    angle = dominant_angle(sample[near & (hz > 1.0) & (hz < 1.8), :2])
    c, s_ = math.cos(math.radians(-angle)), math.sin(math.radians(-angle))
    rot = np.array([[c, -s_], [s_, c]])
    q = sample[near, :2] @ rot.T
    lo = np.percentile(q, 0.2, axis=0) - 0.5
    hi = np.percentile(q, 99.8, axis=0) + 0.5
    px = max(px, float((hi - lo).max()) / 8000)
    shape = (int(math.ceil((hi[1] - lo[1]) / px)), int(math.ceil((hi[0] - lo[0]) / px)))
    log(f"уровней {len(levels)}, поворот {angle:.2f}°, растр {shape[1]}x{shape[0]} по {px * 1000:.0f} мм")
    return levels, angle, rot, lo, px, shape


def slice_counts(path, px: float = 0.01, slice_m: float = 0.1, margin_m: float = 0.1,
                 chunk: int = 2_000_000, sample_points: int = 3_000_000, log=print):
    """Второй проход: на каждый уровень - счётчики точек (срез, H, W), uint8 с насыщением.
    Срезы от margin_m над полом до margin_m под потолком."""
    from .e57read import E57Reader
    from .floorplan import _scan_points

    levels, angle, rot, lo, px, (H, W) = _frame(path, px, chunk, sample_points, log)
    out = []
    for lv in levels:
        top = lv["ceiling_z"] - lv["floor_z"]
        edges = np.arange(margin_m, top - margin_m + 1e-9, slice_m)
        out.append({"level": lv, "z0": edges[:-1], "slice_m": slice_m,
                    "counts": np.zeros((len(edges) - 1, H, W), np.uint8),
                    "free": np.zeros((H, W), bool),           # виден пол или потолок
                    "floor": np.zeros((H, W), bool), "ceil": np.zeros((H, W), bool),
                    "points": np.zeros((H, W), np.int32)})     # всего точек в срезах
    with E57Reader(path) as r:
        for i in range(len(r.scans)):
            for p in _scan_points(r, i, chunk):
                q = p[:, :2] @ rot.T
                jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
                flat = ii * W + jj
                for L in out:
                    if not len(L["z0"]):
                        continue
                    h = p[:, 2] - L["level"]["floor_z"]
                    top = L["level"]["ceiling_z"] - L["level"]["floor_z"]
                    fl = ok & (np.abs(h) < 0.04)
                    ce = ok & (np.abs(h - top) < 0.04)
                    L["floor"].reshape(-1)[flat[fl]] = True
                    L["ceil"].reshape(-1)[flat[ce]] = True
                    L["free"].reshape(-1)[flat[fl | ce]] = True
                    k = np.floor((h - L["z0"][0]) / slice_m).astype(np.int64)
                    m = ok & (k >= 0) & (k < len(L["z0"]))
                    if not m.any():
                        continue
                    key = k[m] * (H * W) + flat[m]
                    u, cnt = np.unique(key, return_counts=True)
                    c = L["counts"].reshape(-1)
                    c[u] = np.minimum(c[u].astype(np.int32) + cnt, 255).astype(np.uint8)
                    L["points"].reshape(-1)[:] += np.bincount(flat[m], minlength=H * W).astype(np.int32)
    return out, {"rotation_deg": angle, "origin": lo.tolist(), "px": px, "shape": [H, W]}


def to_raster(xy: np.ndarray, frame: dict) -> np.ndarray:
    """Точки файла (x, y) -> пиксели растра (j, i) в кадре slice_counts."""
    a = math.radians(-frame["rotation_deg"])
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    q = np.atleast_2d(xy)[:, :2] @ rot.T
    j = (q[:, 0] - frame["origin"][0]) / frame["px"]
    i = frame["shape"][0] - 1 - (q[:, 1] - frame["origin"][1]) / frame["px"]
    return np.c_[j, i]


DEFAULT_PRIOR = "0:1,0.8:1,1.0:0.1,2.0:0.1,2.2:0.6,9:0.6"


def height_prior(z: np.ndarray, spec: str = DEFAULT_PRIOR) -> np.ndarray:
    """Априорный вес среза по высоте над полом: ломаная «высота:вес,...». По умолчанию низ
    (до 0,8 м) - 1: там стена сплошная; середина 1,0-2,0 м - 0,1: там окна; верх - 0,6:
    перемычки есть, но и короба, светильники."""
    pts = sorted((float(a), float(b)) for a, b in (t.split(":") for t in spec.split(",")))
    return np.interp(z, [p[0] for p in pts], [p[1] for p in pts])


def _wsum(w, occ) -> np.ndarray:
    """Сумма срезов с весами без копии всего стека во float (на 5 мм стек - сотни МБ)."""
    out = np.zeros(occ.shape[1:], np.float32)
    for wk, o in zip(w, occ):
        if wk:
            out += np.float32(wk) * o
    return out


def slice_weights(occ: np.ndarray, prior: np.ndarray | None = None, iters: int = 3, tol_px: int = 1,
                  min_slices: int = 4) -> tuple[np.ndarray, np.ndarray, dict]:
    """occ (срез, H, W) bool -> (веса срезов, карта, разбор: точность, полнота)."""
    import cv2

    n = len(occ)
    k = np.ones((2 * tol_px + 1, 2 * tol_px + 1), np.uint8)
    dil = np.stack([cv2.dilate(o.view(np.uint8), k).view(bool) for o in occ])
    prior = np.ones(n) if prior is None else np.asarray(prior, float)
    w = prior.copy()
    f1 = np.ones(n)
    support = _wsum(w, occ)                          # во скольких срезах занят (с весами)
    prec = rec = np.zeros(n)
    for _ in range(iters):
        # структура - занято хотя бы в min_slices срезах (40 см по высоте), а не в половине:
        # стена под окном и над ним тоже структура, срезы через окна её не видят - полнота ниже
        ref = support >= min_slices * w.mean()
        ref_d = cv2.dilate(ref.astype(np.uint8), k).astype(bool)
        nr = max(int(ref.sum()), 1)
        prec = np.array([(o & ref_d).sum() / max(int(o.sum()), 1) for o in occ])
        rec = np.array([(ref & d).sum() / nr for d in dil])
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-9), 0.0)
        w = prior * f1                               # вес = высота (априори) x качество среза
        support = _wsum(w, occ)
    heat = support / max(w.sum(), 1e-9)
    return w, heat, {"precision": prec, "recall": rec, "f1": f1, "prior": prior}


def analyse_slices(L: dict, px: float, prior: str = DEFAULT_PRIOR) -> dict:
    """Срезы уровня -> веса, карта, доли по полосам (низ < 0,9 м, верх >= 2 м) и разметка."""
    import cv2

    occ = L["counts"] > 0
    zc = L["z0"] + L["slice_m"] / 2
    w, heat, info = slice_weights(occ, height_prior(zc, prior))
    # для разметки - с допуском в пиксель: грань дрожит между соседними пикселями от среза
    # к срезу, и на редком облаке стена иначе рвётся
    k3 = np.ones((3, 3), np.uint8)
    bands = {"all": np.ones(len(w), bool), "low": zc < 0.9, "top": zc >= 2.0}
    acc = {b: np.zeros(heat.shape, np.float32) for b in bands}
    for k, (wk, o) in enumerate(zip(w, occ)):
        d = cv2.dilate(o.view(np.uint8), k3).astype(np.float32) * np.float32(wk)
        for b, m in bands.items():
            if m[k]:
                acc[b] += d
    for b, m in bands.items():
        acc[b] /= max(float(w[m].sum()), 1e-9)
    lab, linfo = label_heatmap(acc["all"], px, low=acc["low"], top=acc["top"], free=L["free"])
    return {"occ": occ, "zc": zc, "w": w, "heat": heat, "info": info, "bands": acc, "lab": lab, "linfo": linfo}


def wall_heatmap(path, out_dir, px: float = 0.01, slice_m: float = 0.1, prior: str = DEFAULT_PRIOR,
                 log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, frame = slice_counts(path, px=px, slice_m=slice_m, log=log)
    result = {"file": Path(path).name, **frame, "prior": prior, "levels": []}
    for li, L in enumerate(data, 1):
        A = analyse_slices(L, frame["px"], prior)
        occ, w, heat, info, lab, linfo = A["occ"], A["w"], A["heat"], A["info"], A["lab"], A["linfo"]
        sfx = "" if len(data) == 1 else f"_L{li}"
        np.save(out / f"heat{sfx}.npy", heat.astype(np.float16))
        _plot_heat(heat, L, w, info, frame["px"], out / f"heat{sfx}.png")
        _plot_slices(occ, L, w, out / f"slices{sfx}.png")
        np.save(out / f"labels{sfx}.npy", lab)
        _plot_labels(lab, linfo, frame["px"], out / f"labels{sfx}.png")
        _save_full(heat, lab, out / f"heat_full{sfx}.png", out / f"labels_full{sfx}.png")
        H_ = heat.shape[0]
        to_m = lambda q: [round(frame["origin"][0] + float(q[0]) * frame["px"], 3),          # noqa: E731
                          round(frame["origin"][1] + (H_ - 1 - float(q[1])) * frame["px"], 3)]
        result["levels"].append({
            "floor_z": round(L["level"]["floor_z"], 3), "ceiling_z": round(L["level"]["ceiling_z"], 3),
            "slices": [{"z": round(float(z), 2), "weight": round(float(a), 3), "prior": round(float(pr), 3),
                        "f1": round(float(f), 3), "precision": round(float(p), 3), "recall": round(float(r), 3)}
                       for z, a, pr, f, p, r in zip(L["z0"], w, info["prior"], info["f1"], info["precision"],
                                                    info["recall"])],
            "doors": [{"a": to_m(d["a"]), "b": to_m(d["b"]), "width": round(d["width_px"] * frame["px"], 3)}
                      for d in linfo["doors"]],
            "windows": [{"a": to_m(d["a"]), "b": to_m(d["b"]), "width": round(d["width_px"] * frame["px"], 3),
                         "depth": round(d["depth_px"] * frame["px"], 3)} for d in linfo["windows"]],
            "labels": {LABEL_NAMES[k]: round(float((lab == k).sum()) * frame["px"] ** 2, 3)
                       for k in (WALL, DOOR, WINDOW, NOISE)}})
        log(f"уровень {li}: срезов {len(w)}, дверей {len(linfo['doors'])}, окон {len(linfo['windows'])}")
    (out / "weights.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    return result


# ----------------------------------------------------------------------

def _crop(heat: np.ndarray, pad: int = 20):
    ys, xs = np.nonzero(heat > 0.05)
    if not len(ys):
        return slice(None), slice(None)
    return (slice(max(0, ys.min() - pad), ys.max() + pad), slice(max(0, xs.min() - pad), xs.max() + pad))


def _pool(a: np.ndarray, max_px: int = 1800, rank: np.ndarray | None = None) -> np.ndarray:
    """Уменьшение для рисунка максимумом по блоку: линия в пиксель не пропадает. rank -
    порядок важности меток (больше - важнее) для карты меток."""
    f = int(math.ceil(max(a.shape) / max_px))
    if f <= 1:
        return a
    H, W = a.shape
    pad = np.pad(a if rank is None else rank[a], ((0, -H % f), (0, -W % f)))
    m = pad.reshape(pad.shape[0] // f, f, pad.shape[1] // f, f).max(axis=(1, 3))
    return m if rank is None else np.argsort(rank)[m]


def _plot_heat(heat, L, w, info, px, path):
    from .debug import INK, INK2, ORANGE, _plt

    plt = _plt()
    sl = _crop(heat)
    h = heat[sl]
    H, W = h.shape
    fig = plt.figure(figsize=(12 * W / max(H, W) + 3.6, 12 * H / max(H, W) + 0.6))
    ax = fig.add_axes([0.01, 0.03, 0.74, 0.92])
    im = ax.imshow(_pool(h), cmap="magma_r", vmin=0, vmax=1, interpolation="nearest",
                   extent=[-0.5, W - 0.5, H - 0.5, -0.5])
    ax.plot([20, 20 + 1 / px], [H - 20] * 2, color=INK, lw=2)
    ax.text(20 + 0.5 / px, H - 28, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title("Стены: взвешенная доля срезов, где пиксель занят", loc="left", fontsize=10, color=INK)
    ax.axis("off")
    cax = fig.add_axes([0.03, 0.01, 0.25, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=7, colors=INK2)
    bx = fig.add_axes([0.80, 0.14, 0.17, 0.79])
    z = L["z0"] + L["slice_m"] / 2
    bx.barh(z, w, height=L["slice_m"] * 0.8, color="#3b6fb6", label="вес")
    bx.plot(info["prior"], z, color=INK, lw=1.2, ls=":", label="по высоте")
    bx.plot(info["f1"], z, color=ORANGE, lw=1.4, label="качество (F1)")
    bx.plot(info["precision"], z, color=ORANGE, lw=0.7, alpha=0.6, label="точность")
    bx.plot(info["recall"], z, color=INK2, lw=1, ls="--", label="полнота")
    bx.set_xlim(0, 1)
    bx.set_ylabel("высота над полом, м", fontsize=8, color=INK2)
    bx.set_title("Вес среза", loc="left", fontsize=10, color=INK)
    bx.tick_params(labelsize=7, colors=INK2)
    bx.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.04), ncol=2)
    # пиксель карты - не меньше пикселя картинки
    fig.savefig(path, dpi=max(110, _pool(h).shape[1] / (fig.get_size_inches()[0] * 0.74)))
    plt.close(fig)


def _plot_slices(occ, L, w, path, cols: int = 6):
    from .debug import INK, _plt

    plt = _plt()
    sl = _crop(occ.mean(0))
    n = len(occ)
    rows = math.ceil(n / cols)
    H, W = occ[0][sl].shape
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6 * H / max(W, 1) + 0.3), squeeze=False)
    for ax, k in zip(axs.flat, range(n)):
        ax.imshow(occ[k][sl], cmap="Greys", vmin=0, vmax=1, interpolation="nearest")
        z = L["z0"][k]
        ax.set_title(f"{z:.1f}-{z + L['slice_m']:.1f} м  вес {w[k]:.2f}".replace(".", ","), loc="left",
                     fontsize=7, color=INK)
        ax.axis("off")
    for ax in list(axs.flat)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


# ----------------------------------------------------------------------
# Разметка карты: стена / дверь / окно / шум
# ----------------------------------------------------------------------

EMPTY, WALL, DOOR, WINDOW, NOISE = 0, 1, 2, 3, 4
LABEL_NAMES = {WALL: "стена", DOOR: "дверь", WINDOW: "окно", NOISE: "шум"}


def _axis_extent(ys, xs):
    """Главная ось компоненты: длина вдоль, ширина поперёк, концы (x, y), центр."""
    p = np.c_[xs, ys].astype(float)
    c = p.mean(0)
    if len(p) < 2:
        return 0.0, 0.0, c, c, c
    _, _, vt = np.linalg.svd(p - c, full_matrices=False)
    u = p @ vt[0]
    v = p @ vt[1]
    return float(u.max() - u.min() + 1), float(v.max() - v.min() + 1), p[u.argmin()], p[u.argmax()], c


def label_heatmap(heat: np.ndarray, px: float, low: np.ndarray | None = None, top: np.ndarray | None = None,
                  free: np.ndarray | None = None, wall_min: float = 0.7, door_min: float = 0.2, floor_min: float = 0.04
                  ) -> tuple[np.ndarray, dict]:
    """Правила по тепловой карте (вес по высоте: низ 1, середина 0,1, верх 0,6):
      стена - тепло >= wall_min или низ (срезы до 0,9 м) видит >= 50 %: под окном стена есть,
              окно контур не рвёт; линия от 30 см;
      дверь - низа нет (< 20 %), верх (от 2 м) есть (>= 40 %) - только перемычка; линия
              0,5-1,5 м, оба конца у стены (без low/top - по теплу door_min..wall_min);
      окно  - площадное пятно снаружи контура (откос, стекло, улица) у наружной стены, глубиной
              до 0,7 м; в нише и за стеклом нет ни пола, ни потолка (free) - иначе это
              помещение (коридор, светильник в комнате), а не окно;
      шум   - всё остальное занятое.
    -> (метки uint8, {"doors": [...], "windows": [...]} в пикселях)."""
    import cv2

    P = lambda m: max(1, int(round(m / px)))                     # noqa: E731
    H, W = heat.shape
    occ = heat >= floor_min
    lab = np.zeros((H, W), np.uint8)
    lab[occ] = NOISE

    # --- стены ----------------------------------------------------------------------------
    wall = (heat >= wall_min) if low is None else ((heat >= wall_min) | ((low >= 0.5) & occ))
    wall = wall.astype(np.uint8)
    wall = cv2.morphologyEx(wall, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, cl, st, _ = cv2.connectedComponentsWithStats(wall, connectivity=8)
    for c in range(1, n):
        if math.hypot(st[c, 2], st[c, 3]) < P(0.3):
            wall[cl == c] = 0
    lab[wall > 0] = WALL
    wall_near = cv2.dilate(wall, np.ones((2 * P(0.08) + 1,) * 2, np.uint8)) > 0

    # --- двери: средняя линия, концами в стену ---------------------------------------------
    if low is None or top is None:
        mid = (heat >= door_min) & (heat < wall_min)
    else:
        mid = (low < 0.2) & (top >= 0.4)
    mid = (mid & (wall == 0)).astype(np.uint8)
    mid = cv2.morphologyEx(mid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, cl, st, _ = cv2.connectedComponentsWithStats(mid, connectivity=8)
    doors = []
    for c in range(1, n):
        ys, xs = np.nonzero(cl == c)
        length, width, a, b, ctr = _axis_extent(ys, xs)
        if not (P(0.5) <= length <= P(1.5)) or width > P(0.25) or width > 0.35 * length:
            continue
        ends_ok = all(wall_near[min(H - 1, int(round(e[1]))), min(W - 1, int(round(e[0])))] for e in (a, b))
        if not ends_ok:
            continue
        lab[cl == c] = DOOR
        doors.append({"a": a, "b": b, "center": ctr, "width_px": length})

    # --- снаружи: заливка от края, барьер - стены и двери -------------------------------------
    barrier = cv2.morphologyEx(((lab == WALL) | (lab == DOOR)).astype(np.uint8), cv2.MORPH_CLOSE,
                               np.ones((P(0.1),) * 2, np.uint8))
    ff = (1 - barrier).astype(np.uint8)
    pad = np.pad(ff, 1, constant_values=1)
    mask = np.zeros((H + 4, W + 4), np.uint8)
    cv2.floodFill(pad, mask, (0, 0), 2)
    outside = pad[1:-1, 1:-1] == 2

    # --- окна: площадное пятно снаружи, вплотную к наружной стене ------------------------------
    outer_wall = (lab == WALL) & cv2.dilate(outside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    near_outer = cv2.dilate(outer_wall.astype(np.uint8), np.ones((2 * P(0.6) + 1,) * 2, np.uint8)) > 0
    cand = (occ & outside & near_outer & (lab == NOISE)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((P(0.1),) * 2, np.uint8))
    n, cl, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    windows = []
    free_d = None if free is None else cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE,
                                                        np.ones((5, 5), np.uint8)).astype(bool)
    touch = cv2.dilate(outer_wall.astype(np.uint8), np.ones((2 * P(0.06) + 1,) * 2, np.uint8)) > 0
    for c in range(1, n):
        comp = cl == c
        if st[c, 4] * px * px < 0.04 or not (comp & touch).any():
            continue
        ys, xs = np.nonzero(comp)
        # ширина - вдоль стены: по касанию со стеной
        ty, tx = np.nonzero(comp & touch)
        width, _, a, b, ctr = _axis_extent(ty, tx)
        _, depth, _, _, _ = _axis_extent(ys, xs)
        if not (P(0.4) <= width <= P(2.6)) or depth > P(0.7):
            continue
        if free_d is not None and free_d[comp].mean() > 0.3:
            continue
        lab[comp & occ & (lab == NOISE)] = WINDOW
        windows.append({"a": a, "b": b, "center": np.c_[xs, ys].mean(0), "width_px": width, "depth_px": depth})
    return lab, {"doors": doors, "windows": windows, "outside": outside}


def _plot_labels(lab, info, px, path):
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from .debug import AQUA, INK, INK2, ORANGE, SURFACE, _plt

    plt = _plt()
    sl = _crop((lab > 0).astype(float))
    lb = lab[sl]
    H, W = lb.shape
    oy, ox = sl[0].start or 0, sl[1].start or 0
    colors = [SURFACE, INK, ORANGE, AQUA, "#c9c7c0"]
    fig = plt.figure(figsize=(12 * W / max(H, W) + 0.4, 12 * H / max(H, W) + 0.9))
    ax = fig.add_axes([0.01, 0.06, 0.98, 0.88])
    rank = np.array([0, 4, 3, 2, 1])                 # стена > дверь > окно > шум > пусто
    ax.imshow(_pool(lb, rank=rank), cmap=ListedColormap(colors), vmin=0, vmax=4, interpolation="nearest",
              extent=[-0.5, W - 0.5, H - 0.5, -0.5])
    for d in info["doors"]:
        x, y = d["center"][0] - ox, d["center"][1] - oy
        ax.text(x, y - 12, f"{d['width_px'] * px * 1000:.0f}", ha="center", fontsize=7, color=ORANGE)
    for w in info["windows"]:
        x, y = w["center"][0] - ox, w["center"][1] - oy
        ax.text(x, y, f"{w['width_px'] * px * 1000:.0f}", ha="center", va="center", fontsize=7, color=INK2,
                rotation=90 if abs(w["b"][1] - w["a"][1]) > abs(w["b"][0] - w["a"][0]) else 0)
    ax.plot([20, 20 + 1 / px], [H - 20] * 2, color=INK, lw=2)
    ax.text(20 + 0.5 / px, H - 28, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(f"Разметка: дверей {len(info['doors'])}, окон {len(info['windows'])}", loc="left",
                 fontsize=10, color=INK)
    ax.axis("off")
    fig.legend(handles=[Patch(color=c, label=LABEL_NAMES[k]) for k, c in zip((WALL, DOOR, WINDOW, NOISE), colors[1:])],
               loc="lower left", ncol=4, frameon=False, fontsize=9)
    fig.savefig(path, dpi=max(110, _pool(lb).shape[1] / (fig.get_size_inches()[0] * 0.98)))
    plt.close(fig)


def _save_full(heat, lab, heat_path, lab_path):
    """Карта и разметка в полном разрешении: пиксель картинки = пиксель растра."""
    import cv2
    from matplotlib import colormaps

    from .debug import AQUA, INK, ORANGE, SURFACE

    sl = _crop(heat)
    rgb = (colormaps["magma_r"](np.clip(heat[sl], 0, 1))[..., :3] * 255).astype(np.uint8)
    cv2.imwrite(str(heat_path), rgb[..., ::-1])
    lut = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in (SURFACE, INK, ORANGE, AQUA, "#c9c7c0")],
                   np.uint8)
    cv2.imwrite(str(lab_path), lut[lab[sl]][..., ::-1])


# ----------------------------------------------------------------------
# Тонкий срез у пола (под радиаторами)
# ----------------------------------------------------------------------

def slice_map(path, out_dir, z0: float = 0.03, z1: float = 0.08, px: float = 0.01, floor_cell_m: float = 0.25,
              chunk: int = 2_000_000, sample_points: int = 3_000_000, log=print) -> dict:
    """Карта точек в тонком срезе z0..z1 над полом - под радиаторами (их низ ~10-15 см) видны
    только стены, плинтусы и стоящее на полу. Высота - от местного пола: медиана точек пола в
    клетках floor_cell_m (пол у реальных объектов неровный на 1-2 см, срез в 5 см иначе
    задевает сам пол). На уровень: slice.png (полное разрешение, пиксель = px), slice_view.png
    (с масштабом), slice.npy (точек в пикселе, float32), floor.npy (местный пол, м)."""
    import cv2

    from .e57read import E57Reader
    from .floorplan import _scan_points

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels, angle, rot, lo, px, (H, W) = _frame(path, px, chunk, sample_points, log)
    fc = max(1, int(round(floor_cell_m / px)))
    Hc, Wc = H // fc + 1, W // fc + 1
    res = []
    for li, lv in enumerate(levels, 1):
        # проход 1: местный пол по клеткам (точки в 6 см от плоскости пола уровня)
        acc_z = np.zeros(Hc * Wc)
        acc_n = np.zeros(Hc * Wc)
        acc_z2 = np.zeros(Hc * Wc)
        with E57Reader(path) as r:
            for i in range(len(r.scans)):
                for p in _scan_points(r, i, chunk):
                    q = p[:, :2] @ rot.T
                    jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                    ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                    ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W) & (np.abs(p[:, 2] - lv["floor_z"]) < 0.06)
                    c = (ii[ok] // fc) * Wc + jj[ok] // fc
                    acc_n += np.bincount(c, minlength=Hc * Wc)
                    acc_z += np.bincount(c, p[ok, 2], minlength=Hc * Wc)
                    acc_z2 += np.bincount(c, p[ok, 2] ** 2, minlength=Hc * Wc)
        # среднее с отсевом: клетки, где точек мало или разброс велик (мусор на полу) - соседями
        mean = np.where(acc_n > 20, acc_z / np.maximum(acc_n, 1), np.nan)
        sd = np.sqrt(np.maximum(acc_z2 / np.maximum(acc_n, 1) - np.nan_to_num(mean) ** 2, 0))
        mean[sd > 0.015] = np.nan
        fl = mean.reshape(Hc, Wc)
        known = np.isfinite(fl)
        if known.any():                                  # пустые клетки - значение ближайшей известной
            idx = cv2.distanceTransformWithLabels((~known).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)[1]
            lut = np.zeros(idx.max() + 1)
            kk = np.nonzero(known)
            lut[idx[kk]] = fl[kk]
            fl = lut[idx]
        else:
            fl = np.full((Hc, Wc), lv["floor_z"])
        # проход 2: точки в срезе над местным полом
        cnt = np.zeros(H * W, np.int64)
        with E57Reader(path) as r:
            for i in range(len(r.scans)):
                for p in _scan_points(r, i, chunk):
                    q = p[:, :2] @ rot.T
                    jj = ((q[:, 0] - lo[0]) / px).astype(np.int64)
                    ii = H - 1 - ((q[:, 1] - lo[1]) / px).astype(np.int64)
                    ok = (ii >= 0) & (ii < H) & (jj >= 0) & (jj < W)
                    h = p[ok, 2] - fl[ii[ok] // fc, jj[ok] // fc]
                    m = (h >= z0) & (h < z1)
                    cnt += np.bincount((ii[ok] * W + jj[ok])[m], minlength=H * W)
        cnt = cnt.reshape(H, W).astype(np.float32)
        sfx = "" if len(levels) == 1 else f"_L{li}"
        np.save(out / f"slice{sfx}.npy", cnt)
        np.save(out / f"floor{sfx}.npy", np.repeat(np.repeat(fl, fc, 0), fc, 1)[:H, :W].astype(np.float32))
        nz = cnt[cnt > 0]
        ref = float(np.percentile(nz, 95)) if len(nz) else 1.0
        v = np.clip(np.log1p(cnt) / math.log1p(ref), 0, 1)
        _save_slice(v, out / f"slice{sfx}.png", out / f"slice_view{sfx}.png", px, z0, z1, lv)
        flat = float(np.nanstd(mean)) if np.isfinite(mean).any() else 0.0
        log(f"уровень {li}: срез {z0 * 100:.0f}-{z1 * 100:.0f} см над местным полом, точек {int(cnt.sum())}, "
            f"разброс пола по клеткам {flat * 1000:.0f} мм")
        res.append({"level": lv, "points": int(cnt.sum()), "floor_cells_sd_mm": round(flat * 1000, 1)})
    (out / "slice.json").write_text(json.dumps({"file": Path(path).name, "z_m": [z0, z1], "pixel_m": px,
                                                "rotation_deg": angle, "origin": list(map(float, lo)),
                                                "levels": res}, indent=1, ensure_ascii=False, default=float),
                                    encoding="utf-8")
    return {"levels": res}


def _save_slice(v: np.ndarray, full_path: Path, view_path: Path, px: float, z0: float, z1: float, lv: dict) -> None:
    import cv2
    from matplotlib import colormaps

    from .debug import INK, INK2, _plt

    sl = _crop(v, pad=30)
    vv = v[sl]
    rgb = (colormaps["magma_r"](vv)[..., :3] * 255).astype(np.uint8)
    cv2.imwrite(str(full_path), rgb[..., ::-1])
    plt = _plt()
    Hh, Ww = vv.shape
    fig = plt.figure(figsize=(14 * Ww / max(Hh, Ww) + 0.4, 14 * Hh / max(Hh, Ww) + 0.8))
    ax = fig.add_axes([0.01, 0.05, 0.98, 0.9])
    im = ax.imshow(_pool(vv), cmap="magma_r", vmin=0, vmax=1, interpolation="nearest",
                   extent=[-0.5, Ww - 0.5, Hh - 0.5, -0.5])
    ax.plot([20, 20 + 1 / px], [Hh - 20] * 2, color=INK, lw=2)
    ax.text(20 + 0.5 / px, Hh - 28, "1 м", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_title(f"Срез {z0 * 100:.0f}-{z1 * 100:.0f} см над местным полом (под радиаторами), пиксель "
                 f"{px * 1000:.0f} мм".replace(".", ","), loc="left", fontsize=11, color=INK)
    ax.axis("off")
    cax = fig.add_axes([0.02, 0.02, 0.25, 0.012])
    fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(labelsize=7, colors=INK2)
    fig.savefig(view_path, dpi=max(110, _pool(vv).shape[1] / (fig.get_size_inches()[0] * 0.98)))
    plt.close(fig)
