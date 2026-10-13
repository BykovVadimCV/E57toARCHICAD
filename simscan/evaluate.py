"""Подробная оценка сети плана на части датасета (val / test).

    python -m simscan evaluate runs/r1/best.pt --data dataset --split test --out runs/r1/report [--tta]

Уровень предсказывается целиком (окнами, как в infer) и сравнивается с разметкой:

  классы        матрица ошибок, IoU / точность / полнота по классам, mIoU стены и проёмов
                (как best.pt), калибровка (ECE) уверенности;
  геометрия     «конструкция» = стена + дверь + окно + проём: F1 с допуском 2 px (4 см) -
                положение стен без придирки к толщине; полнота и точность по скелету -
                связность контура (разрыв стены рвёт помещение в cellplan); число кусков;
  проёмы        как объекты: компонента разметки двери / окна / проёма найдена, если её
                перекрывает предсказание того же класса (IoU >= 0.2); точность, полнота,
                ошибка ширины (см), с чем путаются (другой тип, стена, ничего);
  головы        расстояние до грани (MAE, см), ориентация (ошибка в градусах) отдельно для
                стен по осям сцены, наклонных на 1-6° и прочих (дуги, 45°), высоты проёмов
                (низ и верх, см);
  разрезы       те же числа по группам уровней: с мебелью / под отделку, с дугой, с
                наклонами, двухуровневые, повёрнутое / прореженное облако (аугментации);
  худшие        уровни с наименьшим IoU конструкции: вход, разметка, сеть, карта ошибок.

Выход (--out): report.json (всё числами), levels.csv (строка на уровень), *.png (графики).
"""
from __future__ import annotations

import csv
import heapq
import json
import math
from pathlib import Path

import numpy as np

NAMES = ["улица", "помещение", "стена", "дверь", "окно", "проём", "мусор", "пустота"]
STRUCT = (2, 3, 4, 5)
OPENINGS = (3, 4, 5)
SCORED = (2, 3, 4, 5)
IGNORE = 255
TOL_PX = 2
DIST_MAX_CM = 30.0
OPEN_MIN_PX = 12                    # компонента проёма меньше - шум (12 px = 48 см² при 2 см)
OPEN_MATCH_IOU = 0.2
ORI_BUCKETS = ("по осям", "наклон 1-6°", "прочие")
ERR_BINS = np.linspace(0, 90, 181)  # гистограммы ошибок ориентации, шаг 0,5°
ECE_BINS = 10


# ----------------------------------------------------------------------
# Вспомогательное
# ----------------------------------------------------------------------

def _skeleton(m: np.ndarray) -> np.ndarray:
    """Скелет маски (морфологический, Lantuéjoul): достаточно для связности стен 2-30 px."""
    import cv2

    m = m.astype(np.uint8)
    skel = np.zeros_like(m)
    k = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while m.any():
        er = cv2.erode(m, k)
        skel |= m & ~cv2.dilate(er, k)
        m = er
    return skel.astype(bool)


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    import cv2

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k).astype(bool)


def _components(m: np.ndarray, min_px: int):
    import cv2

    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return [lab == i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_px]


def _length_px(m: np.ndarray) -> float:
    """Длинная сторона описанного прямоугольника компоненты - ширина проёма."""
    import cv2

    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return float(len(xs))
    (_, _), (w, h), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    return float(max(w, h))


def _angle_err(p: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Ошибка направления по (cos 2θ, sin 2θ): градусы 0..90."""
    d = np.arctan2(p[1], p[0]) - np.arctan2(g[1], g[0])
    return np.degrees(np.abs(np.angle(np.exp(1j * d)))) / 2


def _buckets(g: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Номер группы ориентации эталона: 0 - по осям сцены, 1 - наклон 1-6°, 2 - прочие.
    Оси сцены - преобладающее направление стен уровня (по 4θ, с точностью до 90°)."""
    out = np.full(m.shape, -1, np.int8)
    if not m.any():
        return out
    th = np.arctan2(g[1][m], g[0][m]) / 2                      # θ стены, -90..90°
    z = np.exp(4j * th).sum()
    dom = np.angle(z) / 4
    dev = np.degrees(np.abs(np.angle(np.exp(4j * (th - dom))))) / 4   # 0..45° до ближайшей оси
    b = np.where(dev < 1.0, 0, np.where(dev < 6.0, 1, 2)).astype(np.int8)
    out[m] = b
    return out


# ----------------------------------------------------------------------
# Один уровень
# ----------------------------------------------------------------------

def level_stats(gt: dict, pr: dict, ceiling_m: float = 2.7) -> dict:
    """Суммируемые величины по уровню. gt: sem (255 - ignore), dist [1], orient [2], heights [2];
    pr: prob [8] (или sem), dist, orient, heights - как из infer.predict."""
    sem = np.asarray(gt["sem"])
    prob = pr.get("prob")
    pred = np.asarray(pr["sem"] if prob is None else np.asarray(prob).argmax(0)).astype(np.int64)
    valid = sem != IGNORE
    n = len(NAMES)
    cm = np.bincount(sem[valid].astype(np.int64) * n + pred[valid], minlength=n * n).reshape(n, n)
    s = {"cm": cm}

    # калибровка: уверенность (макс. вероятность) против правильности
    if prob is not None:
        conf = np.asarray(prob, np.float32).max(0)[valid]
        corr = (pred[valid] == sem[valid])
        b = np.clip((conf * ECE_BINS).astype(int), 0, ECE_BINS - 1)
        s["ece_n"] = np.bincount(b, minlength=ECE_BINS)
        s["ece_conf"] = np.bincount(b, conf, minlength=ECE_BINS)
        s["ece_acc"] = np.bincount(b, corr, minlength=ECE_BINS)

    # конструкция: допуск и скелет
    g_st = np.isin(sem, STRUCT)
    p_st = np.isin(pred, STRUCT) & (valid | _dilate(g_st, 1))      # кольцо ignore у стен не в счёт
    g_d, p_d = _dilate(g_st, TOL_PX), _dilate(p_st, TOL_PX)
    s["tol"] = np.array([(p_st & g_d).sum(), p_st.sum(), (g_st & p_d).sum(), g_st.sum()], np.int64)
    gk, pk = _skeleton(g_st), _skeleton(p_st)
    s["skel"] = np.array([(pk & g_d).sum(), pk.sum(), (gk & p_d).sum(), gk.sum()], np.int64)
    s["pieces"] = np.array([len(_components(g_st, 20)), len(_components(p_st, 20))], np.int64)

    # проёмы как объекты
    op = {}
    conf_types = np.zeros((3, 5), np.int64)        # эталон дверь/окно/проём -> то же, другой проём, стена, ничего, -
    width_err = []
    for ci, c in enumerate(OPENINGS):
        gcs = _components(sem == c, OPEN_MIN_PX)
        pcs = _components(pred == c, OPEN_MIN_PX)
        used = set()
        tp = 0
        for g in gcs:
            best, bi = 0.0, -1
            for j, p in enumerate(pcs):
                if j in used:
                    continue
                inter = (g & p).sum()
                if inter:
                    iou = inter / (g | p).sum()
                    if iou > best:
                        best, bi = iou, j
            if best >= OPEN_MATCH_IOU:
                used.add(bi)
                tp += 1
                conf_types[ci, 0] += 1
                width_err.append((_length_px(pcs[bi]) - _length_px(g)))
            else:                                   # что сеть поставила вместо проёма
                lab = pred[g]
                other = np.isin(lab, [o for o in OPENINGS if o != c]).mean()
                wall = (lab == 2).mean()
                if other >= 0.3:
                    conf_types[ci, 1] += 1
                elif wall >= 0.5:
                    conf_types[ci, 2] += 1
                else:
                    conf_types[ci, 3] += 1
        op[c] = np.array([tp, len(pcs) - len(used), len(gcs) - tp], np.int64)   # tp, fp, fn
    s["open"] = np.stack([op[c] for c in OPENINGS])
    s["open_conf"] = conf_types
    s["width_err_px"] = np.asarray(width_err, np.float64)

    # головы
    if "dist" in pr and "dist" in gt:
        gdist = np.asarray(gt["dist"], np.float32)[0]
        m = valid & (gdist < 0.999)
        e = np.abs(np.asarray(pr["dist"], np.float32)[0][m] - gdist[m]) * DIST_MAX_CM
        s["dist"] = np.array([e.sum(), len(e)], np.float64)
    if "orient" in pr and "orient" in gt:
        go = np.asarray(gt["orient"], np.float32)
        m = valid & np.isin(sem, STRUCT) & (np.hypot(go[0], go[1]) > 0.5)
        b = _buckets(go, m)
        err = _angle_err(np.asarray(pr["orient"], np.float32)[:, m], go[:, m])
        bb = b[m]
        s["ori_hist"] = np.stack([np.histogram(err[bb == k], ERR_BINS)[0] for k in range(3)]).astype(np.int64)
    if "heights" in pr and "heights" in gt:
        gh = np.asarray(gt["heights"], np.float32)
        ph = np.asarray(pr["heights"], np.float32)
        out = []
        for k, cls in ((0, (4,)), (1, OPENINGS)):            # низ - только окна, верх - все проёмы
            m = valid & np.isin(sem, cls) & np.isin(pred, cls)
            e = np.abs(ph[k][m] - gh[k][m]) * ceiling_m * 100
            out += [e.sum(), len(e)]
        s["heights"] = np.array(out, np.float64)
    return s


def _iou(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    den = cm.sum(0) + cm.sum(1) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, tp / den, np.nan)


def level_summary(s: dict) -> dict:
    """Короткая строка уровня для таблицы и выбора худших."""
    cm = s["cm"]
    st = list(STRUCT)
    tp = cm[np.ix_(st, st)].sum()
    den = cm[st].sum() + cm[:, st].sum() - tp
    iou = _iou(cm)
    t = s["tol"]
    f1 = _f1(t[0] / max(t[1], 1), t[2] / max(t[3], 1))
    o = s["open"]
    return {"struct_iou": round(float(tp / den), 4) if den else None,
            "wall_iou": None if np.isnan(iou[2]) else round(float(iou[2]), 4),
            "struct_f1_tol": round(f1, 4),
            "skel_recall": round(float(s["skel"][2] / max(s["skel"][3], 1)), 4),
            "openings_gt": int(o[:, 0].sum() + o[:, 2].sum()), "openings_found": int(o[:, 0].sum()),
            "openings_false": int(o[:, 1].sum()),
            "pieces_gt": int(s["pieces"][0]), "pieces_pred": int(s["pieces"][1])}


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


# ----------------------------------------------------------------------
# Сводка
# ----------------------------------------------------------------------

def _pct(hist: np.ndarray, q: float) -> float | None:
    n = hist.sum()
    if n == 0:
        return None
    c = np.cumsum(hist) / n
    return float(ERR_BINS[1:][np.searchsorted(c, q)])


def aggregate(stats: list[dict]) -> dict:
    """Сумма по уровням -> числа отчёта."""
    if not stats:
        return {}
    tot = {}
    widths = []
    for s in stats:
        for k, v in s.items():
            if k == "width_err_px":
                widths.append(v)
            else:
                tot[k] = tot[k] + v if k in tot else np.array(v, copy=True)
    cm = tot["cm"]
    iou = _iou(cm)
    with np.errstate(invalid="ignore", divide="ignore"):       # нет предсказаний / эталона - не определено
        prec = np.where(cm.sum(0) > 0, np.diag(cm) / cm.sum(0), np.nan)
        rec = np.where(cm.sum(1) > 0, np.diag(cm) / cm.sum(1), np.nan)
    rep = {"levels": len(stats),
           "classes": {NAMES[i]: {"iou": _r(iou[i]), "precision": _r(prec[i]), "recall": _r(rec[i]),
                                  "gt_share": _r(cm[i].sum() / max(cm.sum(), 1))} for i in range(len(NAMES))},
           "miou_walls": _r(np.nanmean([iou[i] for i in SCORED]) if not all(np.isnan(iou[i]) for i in SCORED)
                            else np.nan),
           "confusion": cm.tolist()}
    st = list(STRUCT)
    tp = cm[np.ix_(st, st)].sum()
    rep["struct_iou"] = _r(tp / max(cm[st].sum() + cm[:, st].sum() - tp, 1))
    t, k = tot["tol"], tot["skel"]
    p, r = t[0] / max(t[1], 1), t[2] / max(t[3], 1)
    rep["struct_tol"] = {"tol_cm": TOL_PX * 2, "precision": _r(p), "recall": _r(r), "f1": _r(_f1(p, r))}
    p, r = k[0] / max(k[1], 1), k[2] / max(k[3], 1)
    rep["skeleton"] = {"precision": _r(p), "recall": _r(r), "f1": _r(_f1(p, r)),
                       "pieces_gt": int(tot["pieces"][0]), "pieces_pred": int(tot["pieces"][1])}
    w = np.concatenate(widths) if widths else np.zeros(0)
    ops = {}
    for i, c in enumerate(OPENINGS):
        tp_, fp, fn = (int(x) for x in tot["open"][i])
        p = tp_ / (tp_ + fp) if tp_ + fp else None            # сеть таких не ставила - не определено
        r = tp_ / (tp_ + fn) if tp_ + fn else None            # в эталоне таких нет - не определено
        cf = tot["open_conf"][i]
        ops[NAMES[c]] = {"gt": tp_ + fn, "found": tp_, "false": fp, "precision": _r(p), "recall": _r(r),
                         "f1": _r(_f1(p or 0.0, r or 0.0)) if r is not None else None,
                         "missed_as": {"другой проём": int(cf[1]), "стена": int(cf[2]), "ничего/помещение": int(cf[3])}}
    rep["openings"] = ops
    rep["opening_width_err_cm"] = {"mae": _r(np.abs(w).mean() * 2) if len(w) else None,
                                   "bias": _r(w.mean() * 2) if len(w) else None, "n": int(len(w))}
    if "dist" in tot:
        rep["dist_mae_cm"] = _r(tot["dist"][0] / max(tot["dist"][1], 1))
    if "ori_hist" in tot:
        h = tot["ori_hist"]
        rep["orientation_err_deg"] = {
            ORI_BUCKETS[k]: {"px": int(h[k].sum()), "median": _pct(h[k], 0.5), "p90": _pct(h[k], 0.9),
                             "share_over_5": _r(h[k][ERR_BINS[1:] > 5].sum() / max(h[k].sum(), 1))}
            for k in range(3)}
    if "heights" in tot:
        hh = tot["heights"]
        rep["heights_mae_cm"] = {"sill_window": _r(hh[0] / max(hh[1], 1)), "head": _r(hh[2] / max(hh[3], 1))}
    if "ece_n" in tot:
        n_, c_, a_ = tot["ece_n"], tot["ece_conf"], tot["ece_acc"]
        rep["calibration"] = {"ece": _r(np.sum(np.abs(c_ - a_)) / max(n_.sum(), 1)),
                              "bins": [{"conf": _r(c / max(m, 1)), "acc": _r(a / max(m, 1)), "n": int(m)}
                                       for c, a, m in zip(c_, a_, n_)]}
    return rep


def _r(x, nd: int = 4):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else round(x, nd)


# ----------------------------------------------------------------------
# Признаки уровня (для разрезов)
# ----------------------------------------------------------------------

def level_tags(scene: dict) -> dict:
    w = (scene.get("realism") or {}).get("warp") or {}
    old = not isinstance(w, dict)                      # матрица 2x2 - генератор до planwarp
    if old:
        w = {}
    aug = scene.get("aug") or {}
    gen = "нет данных" if "realism" not in scene else \
        ("старый: без наклонов и дуг" if old else "с наклонами и дугами")
    return {"генератор": gen,
            "мебель": "под отделку" if scene.get("bare") else "с мебелью",
            "дуга": "есть" if w.get("arcs") else "нет",
            "наклоны стен": "есть" if w.get("grid") else "нет",
            "двухуровневая": "да" if scene.get("levels_in_scene", 1) > 1 else "нет",
            "облако": " и ".join(x for x, on in (("повёрнуто", aug.get("rot_deg")), ("прорежено", aug.get("thin")))
                                 if on) or "как есть"}


# ----------------------------------------------------------------------
# Прогон
# ----------------------------------------------------------------------

def _rgb(x):
    x = np.asarray(x, np.float32)
    return np.clip(np.stack([x[0], x[3] * 0.6 + x[6] * 0.4, x[2]], -1), 0, 1)


def _sem_rgb(s):
    from .infer import COLORS

    s = np.asarray(s).copy()
    ign = s == IGNORE
    s[ign] = 0
    im = COLORS[s].copy()
    im[ign] = (255, 255, 255)
    return im


def _err_rgb(sem, pred):
    """Карта ошибок: серое - верно, красное - лишняя конструкция, синее - пропущенная,
    оранжевое - конструкция не того типа (стена вместо двери и т.п.)."""
    im = np.full(sem.shape + (3,), 245, np.uint8)
    v = sem != IGNORE
    g, p = np.isin(sem, STRUCT) & v, np.isin(pred, STRUCT) & v
    im[g & p & (sem == pred)] = (90, 90, 90)
    im[g & p & (sem != pred)] = (240, 150, 0)
    im[p & ~g] = (220, 40, 40)
    im[g & ~p] = (40, 90, 230)
    return im


def evaluate(ckpt, data, split: str = "test", out=None, tta: bool = False, tile: int = 512,
             limit: int | None = None, worst: int = 6, log=print) -> dict:
    from .infer import load_model, predict
    from .torchdata import load_level

    data = Path(data)
    ids = (data / "splits" / f"{split}.txt").read_text(encoding="utf-8").split()
    if limit:
        ids = ids[:limit]
    model, dev = load_model(ckpt)
    stats, rows, heap = [], [], []
    groups: dict = {}
    for k, i in enumerate(ids):
        d = data / "scenes" / i
        s = load_level(d)
        scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
        ceil = float(scene.get("ceiling_z_gt", 2.7) - scene.get("floor_z_gt", 0.0)) or 2.7
        r = predict(model, dev, s["input"], tile=tile, tta=tta)
        st = level_stats(s, r, ceil)
        stats.append(st)
        row = {"id": i, **level_tags(scene), **level_summary(st)}
        rows.append(row)
        for g, v in level_tags(scene).items():
            groups.setdefault((g, v), []).append(st)
        key = row["struct_iou"] if row["struct_iou"] is not None else 1.0
        item = (-key, k, i, _rgb(s["input"]), s["sem"].astype(np.uint8),
                r["prob"].argmax(0).astype(np.uint8))
        if len(heap) < worst:
            heapq.heappush(heap, item)
        elif -key > heap[0][0]:
            heapq.heapreplace(heap, item)
        if (k + 1) % 10 == 0 or k + 1 == len(ids):
            log(f"  {k + 1}/{len(ids)}")
    rep = {"ckpt": str(ckpt), "split": split, "tta": tta, **aggregate(stats)}
    rep["groups"] = {}
    for (g, v), ss in sorted(groups.items()):
        a = aggregate(ss)
        rep["groups"].setdefault(g, {})[v] = {
            "levels": len(ss), "struct_iou": a["struct_iou"], "miou_walls": a["miou_walls"],
            "struct_f1_tol": a["struct_tol"]["f1"], "skeleton_recall": a["skeleton"]["recall"],
            "openings_f1": {n: o["f1"] for n, o in a["openings"].items()},
            "orient_median_deg": {b: o["median"] for b, o in a.get("orientation_err_deg", {}).items()}}
    worst_items = sorted(heap, key=lambda t: t[0], reverse=True)        # худший первым
    rep["worst"] = [{"id": t[2], "struct_iou": -t[0]} for t in sorted(heap, key=lambda t: t[0], reverse=True)]
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
        with open(out / "levels.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        rep["figures"] = plot_report(rep, rows, worst_items, out)
    return rep


# ----------------------------------------------------------------------
# Картинки
# ----------------------------------------------------------------------

def plot_report(rep: dict, rows: list, worst_items: list, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []

    def save(fig, name):
        p = out / name
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        files.append(str(p))

    # 1. матрица ошибок (доли по строкам) и IoU / точность / полнота
    cm = np.asarray(rep["confusion"], float)
    rn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1, 1.2]})
    ax[0].imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for a in range(8):
        for b in range(8):
            if rn[a, b] >= 0.005:
                ax[0].text(b, a, f"{rn[a, b]:.2f}", ha="center", va="center", fontsize=7,
                           color="white" if rn[a, b] > 0.5 else "black")
    ax[0].set_xticks(range(8), NAMES, rotation=45, ha="right")
    ax[0].set_yticks(range(8), NAMES)
    ax[0].set_xlabel("сеть")
    ax[0].set_ylabel("разметка")
    ax[0].set_title("Матрица ошибок (доля строки)")
    x = np.arange(8)
    for j, (k, lab) in enumerate((("iou", "IoU"), ("precision", "точность"), ("recall", "полнота"))):
        v = [rep["classes"][n][k] or 0 for n in NAMES]
        ax[1].bar(x + (j - 1) * 0.27, v, 0.27, label=lab)
    ax[1].set_xticks(x, NAMES, rotation=45, ha="right")
    ax[1].set_ylim(0, 1)
    ax[1].grid(axis="y", alpha=0.3)
    ax[1].legend()
    ax[1].set_title(f"По классам; mIoU стены и проёмов {rep['miou_walls']}")
    save(fig, "classes.png")

    # 2. проёмы как объекты и ориентация
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    ops = rep["openings"]
    names = list(ops)
    x = np.arange(len(names))
    for j, (k, lab) in enumerate((("precision", "точность"), ("recall", "полнота"), ("f1", "F1"))):
        ax[0].bar(x + (j - 1) * 0.27, [ops[n][k] or 0 for n in names], 0.27, label=lab)
    for xi, n in zip(x, names):
        ax[0].text(xi, 1.02, f"{ops[n]['found']}/{ops[n]['gt']}", ha="center", fontsize=8)
    ax[0].set_xticks(x, names)
    ax[0].set_ylim(0, 1.1)
    ax[0].legend()
    we = rep["opening_width_err_cm"]
    ax[0].set_title(f"Проёмы как объекты (найдено/всего); ошибка ширины {we['mae']} см, смещение {we['bias']} см")
    if "orientation_err_deg" in rep:
        for b, o in rep["orientation_err_deg"].items():
            if o["px"]:
                ax[1].bar(b, o["median"] or 0, color="#4c78a8")
                ax[1].text(b, (o["median"] or 0) + 0.05, f"медиана {o['median']}°\np90 {o['p90']}°\n{o['px']} px",
                           ha="center", fontsize=8)
        ax[1].set_ylabel("градусы")
        ax[1].set_title("Ошибка ориентации стен по группам (эталон)")
    save(fig, "openings_orientation.png")

    # 3. разрезы
    g = rep["groups"]
    keys = [(k, v) for k in g for v in g[k]]
    fig, ax = plt.subplots(figsize=(15, 4.8))
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [g[k][v]["struct_iou"] or 0 for k, v in keys], 0.4, label="IoU конструкции")
    ax.bar(x + 0.2, [g[k][v]["skeleton_recall"] or 0 for k, v in keys], 0.4, label="полнота скелета")
    ax.set_xticks(x, [f"{k}: {v}\n({g[k][v]['levels']} ур.)" for k, v in keys], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_title("Качество по группам уровней")
    save(fig, "groups.png")

    # 4. распределение IoU по уровням
    v = [r["struct_iou"] for r in rows if r["struct_iou"] is not None]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(v, bins=20, range=(0, 1), color="#4c78a8")
    ax.set_xlabel("IoU конструкции на уровне")
    ax.set_ylabel("уровней")
    ax.set_title(f"По уровням: медиана {np.median(v):.3f}, худший {min(v):.3f}" if v else "нет уровней")
    save(fig, "levels_hist.png")

    # 5. худшие уровни
    if worst_items:
        n = len(worst_items)
        fig, ax = plt.subplots(n, 4, figsize=(20, 4.6 * n), squeeze=False)
        for r_, t in enumerate(worst_items):
            _, _, i, rgb, sem, pred = t
            ax[r_, 0].imshow(rgb)
            ax[r_, 0].set_title(f"{i}  IoU {-t[0]:.3f}", fontsize=9)
            ax[r_, 1].imshow(_sem_rgb(sem))
            ax[r_, 1].set_title("разметка")
            ax[r_, 2].imshow(_sem_rgb(pred))
            ax[r_, 2].set_title("сеть")
            ax[r_, 3].imshow(_err_rgb(sem, pred))
            ax[r_, 3].set_title("красное - лишнее, синее - пропуск, оранжевое - не тот тип", fontsize=8)
        for a in ax.ravel():
            a.axis("off")
        save(fig, "worst.png")
    return files


def compare(a: dict, b: dict, names=("A", "B")) -> list[tuple]:
    """Главные числа двух отчётов (например, без TTA и с TTA): [(метрика, a, b, b - a)]."""
    def pick(r):
        out = {"mIoU стены и проёмов": r["miou_walls"], "IoU конструкции": r["struct_iou"],
               "F1 конструкции (±4 см)": r["struct_tol"]["f1"], "полнота скелета": r["skeleton"]["recall"]}
        for n, o in r["openings"].items():
            out[f"F1 {n}"] = o["f1"]
        for bk, o in r.get("orientation_err_deg", {}).items():
            out[f"ориентация {bk}, медиана °"] = o["median"]
        out["расстояние до грани, MAE см"] = r.get("dist_mae_cm")
        return out
    pa, pb = pick(a), pick(b)
    return [(k, pa[k], pb.get(k), None if pa[k] is None or pb.get(k) is None else round(pb[k] - pa[k], 4))
            for k in pa]


# ----------------------------------------------------------------------
# Векторный план против векторного эталона (vector.json датасета)
# ----------------------------------------------------------------------

VEC_TOL_M = 0.06                    # ось стены найдена, если ближе 6 см
VEC_STEP_M = 0.05                   # шаг выборки по осям
OPEN_MATCH_M = 0.4                  # проём найден, если центр ближе 40 см и тип тот же


def _polyline(geom) -> np.ndarray:
    if geom["type"] == "segment":
        return np.array([geom["p0"], geom["p1"]], float)
    return np.asarray(geom["points"], float)


def _sample(poly: np.ndarray, step: float):
    """Точки вдоль ломаной через step и направление участка в каждой."""
    pts, dirs = [], []
    for a, b in zip(poly[:-1], poly[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            continue
        n = max(1, int(round(L / step)))
        t = (np.arange(n) + 0.5) / n
        pts.append(a + (b - a) * t[:, None])
        dirs.append(np.repeat(((b - a) / L)[None], n, 0))
    if not pts:
        return np.zeros((0, 2)), np.zeros((0, 2))
    return np.vstack(pts), np.vstack(dirs)


def _seg_dist(p: np.ndarray, segs: np.ndarray):
    """Расстояние от точек p [n,2] до отрезков segs [m,2,2]: (минимум, индекс отрезка)."""
    if len(segs) == 0 or len(p) == 0:
        return np.full(len(p), np.inf), np.zeros(len(p), int)
    a, b = segs[:, 0], segs[:, 1]
    d = b - a
    L2 = np.maximum((d * d).sum(1), 1e-12)
    out_d = np.full(len(p), np.inf)
    out_i = np.zeros(len(p), int)
    for k in range(0, len(p), 2000):
        q = p[k:k + 2000, None, :]
        t = np.clip(((q - a) * d).sum(-1) / L2, 0, 1)
        dist = np.linalg.norm(q - (a + t[..., None] * d), axis=-1)
        out_i[k:k + 2000] = dist.argmin(1)
        out_d[k:k + 2000] = dist.min(1)
    return out_d, out_i


def _inside(pt, polys) -> bool:
    import cv2

    return any(cv2.pointPolygonTest(np.asarray(p, np.float32), (float(pt[0]), float(pt[1])), False) >= 0
               for p in polys)


def label_walls(gt_vec: dict, band: float = 0.10) -> list[dict]:
    """Стены эталона так, как их учит сеть: наружная - полоса band от внутренней грани (сторона
    помещения), внутренняя - во всю толщину. [{poly, t, exterior}]."""
    rooms = [r["polygon"] for r in gt_vec.get("rooms", [])]
    out = []
    for w in gt_vec["walls"]:
        poly = _polyline(w["geometry"])
        t = float(w["thickness"])
        if w.get("is_exterior") and t > band:
            pts, dirs = _sample(poly, 0.5)
            if len(pts):
                nrm = np.c_[-dirs[:, 1], dirs[:, 0]]
                side = 0
                for sgn in (1, -1):
                    side += sgn * sum(_inside(p + sgn * nrm[i] * (t / 2 + 0.25), rooms) for i, p in enumerate(pts))
                sgn = 1 if side >= 0 else -1
                d = poly[1:] - poly[:-1]
                n_seg = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
                n_v = np.vstack([n_seg[:1], (n_seg[:-1] + n_seg[1:]) / 2, n_seg[-1:]]) if len(n_seg) > 1 \
                    else np.vstack([n_seg, n_seg])
                n_v /= np.maximum(np.linalg.norm(n_v, axis=1, keepdims=True), 1e-12)
                poly = poly + sgn * n_v * (t / 2 - band / 2)
                t = band
        out.append({"poly": poly, "t": t, "exterior": bool(w.get("is_exterior"))})
    return out


def _opening_centers_gt(gt_vec: dict) -> list[dict]:
    walls = {w["id"]: _polyline(w["geometry"]) for w in gt_vec["walls"]}
    out = []
    for o in gt_vec.get("openings", []):
        poly = walls.get(o["host_wall_id"])
        if poly is None:
            continue
        s = float(o["offset"]) + float(o["width"]) / 2
        seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
        k = int(np.searchsorted(np.cumsum(seg), s))
        k = min(k, len(seg) - 1)
        s0 = s - (np.cumsum(seg)[k] - seg[k])
        c = poly[k] + (poly[k + 1] - poly[k]) * (s0 / max(seg[k], 1e-12))
        out.append({"type": o["type"], "c": c, "width": float(o["width"])})
    return out


def _room_iou(gt_rooms, pr_rooms, px: float = 0.05) -> tuple[list, int]:
    import cv2

    polys = [np.asarray(r, float) for r in gt_rooms + pr_rooms if len(r) >= 3]
    if not polys:
        return [], 0
    allp = np.vstack(polys)
    lo = allp.min(0) - px
    H, W = (np.ceil((allp.max(0) - lo) / px).astype(int) + 2)[::-1]

    def mask(poly):
        m = np.zeros((H, W), np.uint8)
        q = np.round((np.asarray(poly, float) - lo) / px).astype(np.int32)
        cv2.fillPoly(m, [q], 1)
        return m.astype(bool)

    gm = [mask(r) for r in gt_rooms if len(r) >= 3]
    pm = [mask(r) for r in pr_rooms if len(r) >= 3]
    ious = []
    used = set()
    for g in gm:
        best, bj = 0.0, -1
        for j, p in enumerate(pm):
            if j in used:
                continue
            inter = (g & p).sum()
            if inter:
                iou = inter / (g | p).sum()
                if iou > best:
                    best, bj = iou, j
        if bj >= 0 and best >= 0.3:
            used.add(bj)
        ious.append(best)
    return ious, len(pm) - len(used)


def _in_raster(xy: np.ndarray, plan: dict) -> np.ndarray:
    """Точки внутри кадра растра плана (plan["pixel_to_world"], plan["frame"]); без кадра - все."""
    M, fr = plan.get("pixel_to_world"), plan.get("frame") or {}
    if M is None or not fr.get("width") or len(xy) == 0:
        return np.ones(len(xy), bool)
    M = np.vstack([np.asarray(M, float), [0, 0, 1]])
    ji = (np.linalg.inv(M) @ np.c_[xy, np.ones(len(xy))].T).T
    return (ji[:, 0] >= 0) & (ji[:, 0] < fr["width"]) & (ji[:, 1] >= 0) & (ji[:, 1] < fr["height"])


def _structure_quality(gw, pw, owner_g, d_g, i_g, pseg, tol) -> dict:
    """Чистота плана, а не только совпадение:
      pieces_per_gt_wall - на сколько стен сети разбита прямая стена эталона (1 - одна прямая);
      axis_offset_cm_p90 - разброс осей кусков, покрывающих одну стену эталона, поперёк неё
                           (0 - все куски на одной прямой);
      dangling_ends      - концы стен сети, не упирающиеся в другую стену (щели на стыках,
                           обрывы) - на стену эталона;
      standalone_walls   - стены сети, не касающиеся ни одной другой (пятна шума)."""
    out = {}
    seg_owner = np.array([k for k, w in enumerate(pw) for _ in range(len(w["poly"]) - 1)])
    pieces, spread = [], []
    if len(seg_owner):
        for k, w in enumerate(gw):
            if len(w["poly"]) != 2:
                continue
            m = (owner_g == k) & (d_g <= tol)
            if m.sum() < 4:
                continue
            a, b = w["poly"]
            u = (b - a) / max(np.linalg.norm(b - a), 1e-9)
            nrm = np.array([-u[1], u[0]])
            sd = pseg[i_g[m], 1] - pseg[i_g[m], 0]
            par = np.abs((sd / np.maximum(np.linalg.norm(sd, axis=1, keepdims=True), 1e-9)) @ u) > 0.98
            own, cnt = np.unique(seg_owner[i_g[m]][par], return_counts=True)
            own = own[(cnt >= 3) & np.array([len(pw[o]["poly"]) == 2 for o in own], bool)]   # прямые от 15 см
            if not len(own):
                continue
            pieces.append(len(own))
            offs = []
            for o in own:                                    # смещение куска - по его середине
                q = pw[o]["poly"]
                offs.append(float(((q[0] + q[-1]) / 2 - a) @ nrm))
            spread.append((max(offs) - min(offs)) * 100 if len(offs) > 1 else 0.0)
    out["pieces_per_gt_wall"] = _r(np.mean(pieces), 2) if pieces else None
    out["axis_offset_cm_p90"] = _r(np.percentile(spread, 90), 1) if spread else None
    # концы и одиночки
    segs = [(k, a, b) for k, w in enumerate(pw) for a, b in zip(w["poly"][:-1], w["poly"][1:])]
    dang, alone = 0, 0
    for k, w in enumerate(pw):
        other = np.array([[a, b] for o, a, b in segs if o != k]).reshape(-1, 2, 2)
        reach = max(w["t"], 0.05) / 2 + 0.05
        ends = np.array([w["poly"][0], w["poly"][-1]])
        d, _ = _seg_dist(ends, other)
        free = d > reach + 0.15
        dang += int(free.sum())
        if free.all():
            pts, _ = _sample(w["poly"], 0.05)
            dd, _ = _seg_dist(pts, other)
            alone += int((dd > reach + 0.15).all())
    out["dangling_ends_per_gt_wall"] = _r(dang / max(len(gw), 1), 3)
    out["standalone_walls"] = alone
    return out


def vector_metrics(plan: dict, gt_vec: dict, tol: float = VEC_TOL_M) -> dict:
    """plan.json (vectorize / cellplan, мировая СК) против vector.json датасета.

    стены: полнота / точность осей с допуском tol (оси эталона - как учится сеть, см.
           label_walls), ошибка направления (°) и толщины (см) на совпавших участках, число
           стен на стену эталона (дробление);
    проёмы: центр ближе OPEN_MATCH_M и тот же тип - найден; ошибка ширины;
    помещения: IoU с лучшим совпавшим, лишние."""
    gw = label_walls(gt_vec)
    pw = [{"poly": np.asarray(w["points"], float), "t": float(w.get("thickness") or 0.0)}
          for w in plan.get("walls", []) if len(w.get("points", [])) >= 2]
    gseg = np.array([[a, b] for w in gw for a, b in zip(w["poly"][:-1], w["poly"][1:])]).reshape(-1, 2, 2)
    gseg_t = np.array([w["t"] for w in gw for _ in range(len(w["poly"]) - 1)])
    pseg = np.array([[a, b] for w in pw for a, b in zip(w["poly"][:-1], w["poly"][1:])]).reshape(-1, 2, 2)
    pseg_t = np.array([w["t"] for w in pw for _ in range(len(w["poly"]) - 1)])
    gs = [_sample(w["poly"], VEC_STEP_M) for w in gw]
    gp = np.vstack([s[0] for s in gs]) if gs else np.zeros((0, 2))
    gd = np.vstack([s[1] for s in gs]) if gs else np.zeros((0, 2))
    owner_g = np.concatenate([np.full(len(s[0]), k) for k, s in enumerate(gs)]) if gs else np.zeros(0, int)
    inside = _in_raster(gp, plan)                               # за краем растра сеть не видит
    gp, gd, owner_g = gp[inside], gd[inside], owner_g[inside]
    ps = [_sample(w["poly"], VEC_STEP_M) for w in pw]
    pp = np.vstack([s[0] for s in ps]) if ps else np.zeros((0, 2))
    d_g, i_g = _seg_dist(gp, pseg)
    d_p, _ = _seg_dist(pp, gseg)
    hit = d_g <= tol
    res = {"walls_gt": len(gw), "walls_pred": len(pw),
           "axis_recall": _r(hit.mean()) if len(gp) else None,
           "axis_precision": _r((d_p <= tol).mean()) if len(pp) else None}
    if hit.any():
        pd_ = pseg[i_g[hit], 1] - pseg[i_g[hit], 0]
        pd_ /= np.maximum(np.linalg.norm(pd_, axis=1, keepdims=True), 1e-12)
        ang = np.degrees(np.arccos(np.clip(np.abs((pd_ * gd[hit]).sum(1)), 0, 1)))
        _, i_t = _seg_dist(gp[hit], gseg)
        terr = np.abs(pseg_t[i_g[hit]] - gseg_t[i_t]) * 100
        res.update({"angle_err_deg_median": _r(np.median(ang), 2), "angle_err_deg_p90": _r(np.percentile(ang, 90), 2),
                    "thickness_err_cm_median": _r(np.median(terr), 1)})
    res["axis_f1"] = _r(_f1(res["axis_recall"] or 0, res["axis_precision"] or 0))
    res.update(_structure_quality(gw, pw, owner_g, d_g, i_g, pseg, tol))
    # проёмы
    go = [g for g in _opening_centers_gt(gt_vec) if _in_raster(np.asarray(g["c"])[None], plan)[0]]
    po = [{"type": o.get("opening_type"), "c": (np.asarray(o["a"], float) + np.asarray(o["b"], float)) / 2,
           "width": float(o.get("width") or np.linalg.norm(np.subtract(o["a"], o["b"])))}
          for o in plan.get("openings", [])]
    used, tp, werr = set(), 0, []
    for g in go:
        cand = [(np.linalg.norm(p["c"] - g["c"]), j) for j, p in enumerate(po)
                if j not in used and p["type"] == g["type"]]
        if cand:
            dmin, j = min(cand)
            if dmin <= OPEN_MATCH_M:
                used.add(j)
                tp += 1
                werr.append(abs(po[j]["width"] - g["width"]) * 100)
    res.update({"openings_gt": len(go), "openings_pred": len(po), "openings_found": tp,
                "openings_f1": _r(_f1(tp / max(len(po), 1), tp / max(len(go), 1))) if go or po else None,
                "opening_width_err_cm": _r(np.median(werr), 1) if werr else None})
    # помещения
    ious, extra = _room_iou([r["polygon"] for r in gt_vec.get("rooms", [])],
                            [r["points"] for r in plan.get("rooms", [])])
    res.update({"rooms_gt": len(ious), "rooms_pred": len(plan.get("rooms", [])), "rooms_extra": extra,
                "room_iou_mean": _r(np.mean(ious)) if ious else None})
    return res
