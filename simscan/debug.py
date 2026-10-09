"""Отладочные визуализации сцены: что именно делает синтезатор с геометрией и лучами.

    python -m simscan debug D:/synth/scene_00001

Пересобирает сцену из layout.json (те же параметры неровности и сдвига), заново
сканирует станцию 0 с диагностикой - с моделью пятна и без неё - и пишет в debug/:

    plan.png                план с мебелью, шторами, стояками (render.py)
    panorama.png            панорама станции 0: дальность, классы, кромки/смешанные/пропуски,
                            интенсивность
    edge_closeup.png        срез у дверного проёма: центральный луч против модели пятна
    wall_flatness.png       отклонение точек одной стены от плоскости: неровность и шум
    noise_vs_incidence.png  шум дальности и доля пропусков от угла падения
    furniture.png           галерея процедурной мебели
    mess.png                беспорядок: ткань на опорах, куртки, сумки, стопки, доски, сканы
    mess_scan.png           тот же беспорядок глазами сканера (облако точек по классам)
    stats.json              сводные числа
"""
from __future__ import annotations

import dataclasses
import math
import json
from pathlib import Path

import numpy as np

# Палитра (dataviz reference palette, проверена validate_palette.js, светлая тема)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
GROUPS = [  # классы точек -> 8 групп (порядок слотов фиксирован)
    ("стены, колонны", BLUE, ("wall", "column")),
    ("пол, потолок", AQUA, ("floor", "ceiling")),
    ("двери, окна, стекло", ORANGE, ("door", "window", "glass")),
    ("мебель", YELLOW, ("furniture",)),
    ("инженерия", GREEN, ("fixture",)),
    ("шторы, тюль", VIOLET, ("curtain",)),
    ("люди, коробки", MAGENTA, ("clutter",)),
    ("улица, зеркало", RED, ("exterior", "mirror")),
]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "normal", "axes.titlelocation": "left",
        "axes.titlecolor": INK2, "legend.frameon": False, "legend.fontsize": 9,
    })
    return plt


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", RED])


# ----------------------------------------------------------------------
# пересборка сцены и сканирование станции с диагностикой
# ----------------------------------------------------------------------

def _rebuild(scene_dir: Path):
    from .config import load_config
    from .layout import Layout
    from .realism import apply_to_mesh
    from .scanner import ScanSimulator, Station
    from .scene import build_scene

    doc = json.loads((scene_dir / "layout.json").read_text(encoding="utf-8"))
    cfg = load_config(overrides=doc["meta"]["config"])
    lay = Layout.from_dict(doc)
    real = doc["meta"].get("realism", {"enabled": False})
    mesh, solids = build_scene(lay, real.get("tessellation_m") if real.get("enabled") else None)
    apply_to_mesh(mesh, real)
    params = doc["meta"]["effects_sampled"]
    # станция для отладки - в жилой комнате или кухне с наибольшим числом предметов
    n_items = {}
    for it in lay.items:
        if it.room_id is not None:
            n_items[it.room_id] = n_items.get(it.room_id, 0) + 1
    kinds = {r.id: r.kind for r in lay.rooms}
    st = max(doc["stations"], key=lambda q: (kinds.get(q["room_id"]) in ("room", "kitchen"),
                                             n_items.get(q["room_id"], 0)))
    station = Station(st["id"], st["room_id"], tuple(st["position_layout_m"]), st["heading_deg"],
                      tuple(st["tilt_deg"]))

    def scan(beam: bool, seed: int = 0):
        e = dataclasses.replace(cfg.effects, beam_model=beam)
        p = dict(params, beam_divergence_rad=params.get("beam_divergence_rad", 0.0) if beam else 0.0)
        sim = ScanSimulator(mesh, cfg.scanner, e, p, np.random.default_rng(seed))
        sim.keep_diag = True
        return sim, sim.scan(station)

    return doc, cfg, lay, mesh, solids, station, scan


def _to_layout(scan, station, t=None):
    """Точки скана (или дальности t по тем же лучам) в СК сцены."""
    xyz = scan.xyz_local
    if t is not None:
        d = xyz / np.maximum(np.linalg.norm(xyz, axis=1, keepdims=True), 1e-12)
        xyz = d * np.where(np.isfinite(t), t, 0.0)[:, None]
    return xyz @ station.rotation().T + np.asarray(station.position)


# ----------------------------------------------------------------------
# картинки
# ----------------------------------------------------------------------

def panorama(scan, out: Path) -> dict:
    from matplotlib.colors import ListedColormap

    from .labels import LABELS

    plt = _plt()
    R, C = scan.nrow, scan.ncol
    t = np.where(scan.valid, np.linalg.norm(scan.xyz_local, axis=1), np.nan).reshape(R, C)
    grp = np.zeros(len(scan.label), int)
    for k, (_, _, names) in enumerate(GROUPS, 1):
        ids = [i for i, n in LABELS.items() if n in names]
        grp[np.isin(scan.label, ids) & scan.valid] = k
    grp = grp.reshape(R, C)
    d = scan.diag
    central_hit = np.isfinite(d["t_true"])
    dropped = central_hit & ~scan.valid & ~d["edge"]
    over = np.zeros((R, C))
    over[d["edge"].reshape(R, C)] = 1
    over[(scan.mixed).reshape(R, C)] = 2
    over[dropped.reshape(R, C)] = 3

    fig, axes = plt.subplots(4, 1, figsize=(15, 13), constrained_layout=True)
    ax = axes[0]
    im = ax.imshow(t, cmap="Blues_r", aspect="auto", interpolation="nearest")
    ax.set_title("Дальность, м")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    ax = axes[1]
    cmap = ListedColormap([SURFACE] + [c for _, c, _ in GROUPS])
    ax.imshow(grp, cmap=cmap, vmin=-0.5, vmax=len(GROUPS) + 0.5, aspect="auto", interpolation="nearest")
    ax.set_title("Классы")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, c, _ in GROUPS],
              labels=[n for n, _, _ in GROUPS], loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    ax = axes[2]
    base = np.where(np.isfinite(t), 0.85, 1.0)
    ax.imshow(base, cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    cmap2 = ListedColormap([(0, 0, 0, 0), AQUA, ORANGE, RED])
    ax.imshow(over, cmap=cmap2, vmin=-0.5, vmax=3.5, aspect="auto", interpolation="nearest")
    ax.set_title("Кромки, смешанные эхо, пропуски")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in (AQUA, ORANGE, RED)],
              labels=["кромка", "смешанное эхо", "пропуск"],
              loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=9)
    ax = axes[3]
    im = ax.imshow(np.where(scan.valid, scan.intensity, np.nan).reshape(R, C), cmap="gray",
                   aspect="auto", interpolation="nearest", vmin=0, vmax=np.nanpercentile(
                       np.where(scan.valid, scan.intensity, np.nan), 99))
    ax.set_title("Интенсивность")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    for ax in axes:
        ax.grid(False)
        ax.set_xticks(np.linspace(0, C - 1, 9))
        ax.set_xticklabels([f"{a:.0f}°" for a in np.linspace(0, 360, 9)])
        ax.set_yticks([0, R - 1])
        ax.set_yticklabels(["верх", "низ"])
    fig.savefig(out, dpi=90)
    plt.close(fig)
    valid = scan.valid
    return {"edge_fraction": float(d["edge"].mean()), "mixed_fraction_of_valid":
            float(scan.mixed[valid].mean()), "valid_fraction": float(valid.mean()),
            "dropout_fraction_of_central_hits": float(dropped.sum() / max(central_hit.sum(), 1))}


def edge_closeup(lay, station, scan_off, scan_on, out: Path, solids=None) -> dict:
    """Горизонтальный срез у проёма двери, ближайшей к станции: что даёт модель пятна."""
    plt = _plt()
    from .groundtruth import _box_corners, layout_warp
    A = layout_warp(lay)
    W = (lambda p: p) if A is None else (lambda p: np.asarray(p) @ A.T)
    pos = np.asarray(station.position[:2])
    p_on = _to_layout(scan_on, station)
    m = scan_on.valid & scan_on.mixed & (p_on[:, 2] > 0.3) & (p_on[:, 2] < 2.0)
    if m.sum() < 10:
        return {}
    cell = 0.25
    key = np.floor(p_on[m, :2] / cell).astype(int)
    uniq, cnt = np.unique(key, axis=0, return_counts=True)
    c = (uniq[np.argmax(cnt)] + 0.5) * cell
    r = 0.6
    best = (float(np.linalg.norm(c - pos)), c)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), constrained_layout=True, sharex=True, sharey=True)
    stats = {}
    for ax, (name, sc) in zip(axes, (("Центральный луч", scan_off), ("Модель пятна", scan_on))):
        p = _to_layout(sc, station)
        sel = sc.valid & (p[:, 2] > 0.3) & (p[:, 2] < 2.0) & (np.abs(p[:, 0] - c[0]) < r) & \
            (np.abs(p[:, 1] - c[1]) < r)
        for sol in solids or []:                      # тела стен на высоте 1,25 м (с проёмами)
            z0 = sol.box.center[2] - sol.box.size[2] / 2
            if sol.label in ("wall", "column") and sol.mesh is None and \
                    z0 <= 1.25 < z0 + sol.box.size[2]:
                q = W(_box_corners(sol.box))
                ax.fill(q[:, 0], q[:, 1], color="#ecebe6", zorder=0, lw=0)
        m = sc.mixed[sel]
        ax.scatter(p[sel][~m, 0], p[sel][~m, 1], s=1, color=BLUE, alpha=0.5, label="точки",
                   zorder=2)
        ax.scatter(p[sel][m, 0], p[sel][m, 1], s=5, color=ORANGE, label="смешанное эхо", zorder=3)
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_xlabel("x, м")
        stats[name] = {"points": int(sel.sum()), "mixed": int(m.sum())}
    axes[0].set_ylabel("y, м")
    axes[1].legend(loc="upper right", markerscale=2)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return stats


def wall_flatness(scan, station, solids, out: Path) -> dict:
    """Точки одной стены относительно наилучшей плоскости: истинная поверхность и измерение."""
    plt = _plt()
    from .labels import LABEL_ID

    wall_ids = {s.instance for s in solids if s.label == "wall"}
    sel0 = scan.valid & (scan.label == LABEL_ID["wall"]) & ~scan.diag["edge"]
    if not sel0.any():
        return {}
    inst = np.unique(scan.instance[sel0])
    inst = inst[np.isin(inst, list(wall_ids))]
    if len(inst) == 0:
        return {}
    counts = dict(zip(*np.unique(scan.instance[sel0], return_counts=True)))
    best_inst = max(inst, key=lambda i: counts[i])
    sel = sel0 & (scan.instance == best_inst)
    p_meas = _to_layout(scan, station)[sel]
    p_true = _to_layout(scan, station, scan.diag["t_true"])[sel]
    # плоскость по истинной поверхности: SVD + отсев (у стены видна в основном одна грань)
    keep = np.ones(len(p_true), bool)
    for _ in range(3):
        ctr = p_true[keep].mean(0)
        _, _, vt = np.linalg.svd(p_true[keep] - ctr, full_matrices=False)
        nrm = vt[2]
        dist = (p_true - ctr) @ nrm
        keep = np.abs(dist) < 0.03
    u = np.cross(nrm, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    s = (p_true - ctr) @ u
    res_true = ((p_true - ctr) @ nrm)[keep] * 1000
    res_meas = ((p_meas - ctr) @ nrm)[keep] * 1000
    s, z = s[keep], p_true[keep, 2]
    lim = float(np.percentile(np.abs(res_meas), 99))
    fig = plt.figure(figsize=(14, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.55])
    for k, (name, res) in enumerate((("Поверхность", res_true), ("Измерение", res_meas))):
        ax = fig.add_subplot(gs[k, :2])
        sc = ax.scatter(s, z, c=res, s=2, cmap=_diverging(), vmin=-lim, vmax=lim)
        ax.set_title(name)
        ax.set_xlabel("вдоль стены, м")
        ax.set_ylabel("высота, м")
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.01, label="мм")
    ax = fig.add_subplot(gs[:, 2])
    bins = np.linspace(-lim, lim, 50)
    ax.hist(res_meas, bins=bins, color=BLUE, alpha=0.9, label="измерение")
    ax.hist(res_true, bins=bins, histtype="step", color=ORANGE, lw=2, label="поверхность")
    ax.set_xlabel("отклонение, мм")
    ax.set_ylabel("точек")
    ax.legend(loc="upper right")
    ax.set_title("Распределение")
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"wall_points": int(keep.sum()), "surface_rms_mm": float(res_true.std()),
            "measured_rms_mm": float(res_meas.std())}


def noise_vs_incidence(scan, out: Path) -> dict:
    plt = _plt()
    d = scan.diag
    ang = np.degrees(np.arccos(np.clip(d["cos_inc"], 0, 1)))
    central = np.isfinite(d["t_true"]) & ~d["edge"] & ~scan.virtual
    ok = central & scan.valid
    err = (np.linalg.norm(scan.xyz_local, axis=1) - d["t_true"]) * 1000
    bins = np.arange(0, 90, 5)
    mid, meas, model, drop = [], [], [], []
    for a in bins:
        b = (ang >= a) & (ang < a + 5)
        if (ok & b).sum() < 200:
            continue
        mid.append(a + 2.5)
        meas.append(float(np.std(err[ok & b])))
        model.append(float(np.sqrt(np.mean(d["sigma"][ok & b] ** 2)) * 1000))
        drop.append(float(100 * (central & b & ~scan.valid).sum() / max((central & b).sum(), 1)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.plot(mid, meas, color=BLUE, lw=2, marker="o", ms=5, label="симуляция")
    ax.plot(mid, model, color=ORANGE, lw=2, ls="--", label="модель σ")
    ax.set_xlabel("угол падения, °")
    ax.set_ylabel("СКО, мм")
    ax.set_title("Шум дальности")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.plot(mid, drop, color=BLUE, lw=2, marker="o", ms=5)
    ax.set_xlabel("угол падения, °")
    ax.set_ylabel("% лучей")
    ax.set_title("Пропуски")
    ax.set_ylim(bottom=0)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return {"incidence_deg": mid, "noise_mm": meas, "model_mm": model, "dropout_pct": drop}


def furniture_gallery(out: Path, seed: int = 3) -> None:
    """3D-галерея процедурных предметов (каждый - как его видит сканер: из частей)."""
    plt = _plt()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from . import furniture as fu
    from .layout import Box
    from .meshes import box_mesh, curtain_mesh, prim_mesh

    rng = np.random.default_rng(seed)
    base = lambda w, d, h: Box((0.0, 0.0, h / 2), (w, d, h), 0.0)  # noqa: E731
    items = [
        ("Стул", fu.chair(base(0.44, 0.42, 0.9), rng), []),
        ("Диван", fu.sofa(base(2.0, 0.9, 0.85), rng), []),
        ("Кровать", fu.bed(base(1.6, 2.0, 0.5), rng), []),
        ("Стеллаж", fu.shelf(base(1.0, 0.35, 1.9), rng), []),
        ("Шкаф", fu.wardrobe(base(1.4, 0.6, 2.2), rng), []),
        ("Тумба ТВ", fu.tv_stand(base(1.6, 0.4, 0.5), rng), []),
        ("Кухня", fu.counter(base(2.0, 0.6, 0.9), rng), []),
        ("Стол", fu.table(base(1.4, 0.85, 0.75), rng), []),
        ("Растение", *fu.plant(0.0, 0.0, rng)),
        ("Вешалка", *fu.coat_rack(0.0, 0.0, rng)),
    ]
    from .layout import Wall

    wall = Wall(0, (-1.0, 0.0), (1.0, 0.0), 0.1, "interior")
    items.append(("Радиатор", fu.radiator(wall, 0.0, 0.05, 1, 1.0, 0.5,
                                                   np.random.default_rng(1)), []))
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    for k, (name, boxes, prims) in enumerate(items):
        ax = fig.add_subplot(3, 4, k + 1, projection="3d")
        polys = []
        for b in boxes:
            v, t = box_mesh(b, 10.0)
            polys += [v[tri] for tri in t]
        for pr in prims:
            v, t = prim_mesh(pr)
            polys += [v[tri] for tri in t]
        _draw3d(ax, polys, Poly3DCollection)
        ax.set_title(name)
    ax = fig.add_subplot(3, 4, 12, projection="3d")
    polys = []
    for p0, p1, color in (((-1.0, 0.0), (-0.3, 0.0), VIOLET), ((0.4, 0.0), (1.0, 0.0), VIOLET)):
        v, t = curtain_mesh(p0, p1, 0.02, 2.4, 0.05, 0.16)
        polys += [v[tri] for tri in t]
    _draw3d(ax, polys, Poly3DCollection, color=VIOLET)
    v, t = curtain_mesh((-1.0, 0.1), (1.0, 0.1), 0.02, 2.4, 0.025, 0.1)
    _draw3d(ax, [v[tri] for tri in t], Poly3DCollection, color="#cfc9ec", alpha=0.5, fit=False)
    ax.set_title("Шторы и тюль")
    fig.savefig(out, dpi=80)
    plt.close(fig)


def _mess_scene(rng, scan_dir: str = ""):
    """Набор «неудобных» предметов на одной площадке 6 x 4 м (для галереи и скана)."""
    from . import furniture as fu
    from .interior import Furnisher
    from .layout import Box

    def base(x, y, w, d, h, yaw=0.0):
        return Box((x, y, h / 2), (w, d, h), yaw)

    groups = []                                    # (подпись, опоры [Box], мягкое [prim])
    ch = fu.chair(base(0.6, 0.6, 0.44, 0.42, 0.9, 0.3), rng)
    groups.append(("Стул + куртка", ch,
                   [fu.drape(max(ch, key=lambda b: b.center[2] + b.size[2] / 2), rng,
                             size=(0.5, 0.9))]))
    bx = Box((2.0, 0.6, 0.25), (0.5, 0.4, 0.5), 0.4)
    groups.append(("Коробка + куртка", [bx], [fu.drape(bx, rng, size=(0.6, 1.0))]))
    so = fu.sofa(base(4.2, 0.55, 2.0, 0.9, 0.85, math.pi), rng)
    seat = so[4]
    groups.append(("Диван + плед", so, [fu.drape(seat, rng, size=(1.2, 1.4))]))
    wall = Box((0.9, 3.95, 1.0), (1.4, 0.1, 2.0), 0.0)                # кусок стены
    jackets = [fu.hanging_cloth((x, 3.88, 1.75), (1, 0, 0), (0, -1, 0), rng) for x in (0.6, 1.15)]
    groups.append(("Крючки", [wall], jackets))
    groups.append(("Сумки, бельё", [], [fu.blob((0.7, 2.2), (0.45, 0.3, 0.35), rng),
                                        fu.blob((1.4, 2.0), (0.6, 0.5, 0.2), rng)]))
    stack, z = [], 0.0
    for k in range(3):
        h = float(rng.uniform(0.25, 0.4))
        stack.append(Box((2.6 + rng.normal(0, 0.03), 2.1 + rng.normal(0, 0.03), z + h / 2),
                         (0.5 * rng.uniform(0.8, 1), 0.4 * rng.uniform(0.8, 1), h),
                         float(rng.normal(0, 0.15))))
        z += h
    groups.append(("Стопка коробок", stack, []))
    groups.append(("Доска", [Box((4.0, 3.95, 1.0), (1.6, 0.1, 2.0), 0.0)],
                   [fu.leaning_board((4.0, 3.9), (1, 0), (0, -1), rng)]))
    scans = []
    if scan_dir:
        f = Furnisher.__new__(Furnisher)
        f.cfg, f.rng = type("C", (), {"scan_dir": scan_dir})(), rng
        tbl = fu.table(base(5.0, 2.3, 1.0, 0.7, 0.75), rng)
        top = max(tbl, key=lambda b: b.center[2])
        zt = top.center[2] + top.size[2] / 2
        for role, x, y, z in (("shoes", 4.3, 2.0, 0.0), ("shoes", 4.3, 2.4, 0.0),
                              ("tabletop", 4.75, 2.2, zt), ("tabletop", 5.0, 2.45, zt),
                              ("tabletop", 5.25, 2.15, zt), ("floor", 5.0, 1.65, 0.0)):
            pr = f._scan_prim(role, x, y, z)
            if pr is not None:
                scans.append(pr)
        groups.append(("Сканы GSO", tbl, scans))
    return groups


def mess_gallery(out: Path, scan_dir: str = "", seed: int = 5) -> None:
    """Сетки беспорядка: опоры - синим, мягкое и сканы - оранжевым."""
    plt = _plt()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from .meshes import box_mesh, prim_mesh

    groups = _mess_scene(np.random.default_rng(seed), scan_dir)
    n = len(groups)
    cols = 4
    fig = plt.figure(figsize=(16, 4.2 * ((n + cols - 1) // cols)), constrained_layout=True)
    for k, (name, boxes, prims) in enumerate(groups):
        ax = fig.add_subplot((n + cols - 1) // cols, cols, k + 1, projection="3d")
        hard = [v[tri] for b in boxes for v, t in [box_mesh(b, 0.08)] for tri in t]
        soft = [v[tri] for pr in prims for v, t in [prim_mesh(pr)] for tri in t]
        pts = np.concatenate(hard + soft)
        # одной коллекцией: matplotlib сортирует по глубине только внутри коллекции
        _draw3d(ax, hard + soft, Poly3DCollection, color=[BLUE] * len(hard) + [ORANGE] * len(soft),
                fit=False)
        lo, hi = pts.min(0), pts.max(0)
        c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.05
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(max(0, c[2] - r), c[2] + r)
        ax.set_title(name)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=BLUE, label="опора"), Patch(color=ORANGE, label="мягкое, скан")],
               loc="lower center", ncol=2)
    fig.savefig(out, dpi=80)
    plt.close(fig)


def mess_scan(out: Path, scan_dir: str = "", seed: int = 5) -> dict:
    """Та же площадка, отсканированная с одной станции: точки по классам, вид сверху и сбоку."""
    plt = _plt()
    from .config import load_config
    from .labels import LABEL_ID
    from .layout import Box
    from .meshes import prim_mesh
    from .scanner import ScanSimulator, Station, sample_effect_params
    from .scene import SceneMesh, Solid

    rng = np.random.default_rng(seed)
    solids = [Solid(Box((3.0, 2.0, -0.05), (6.4, 4.4, 0.1), 0.0), "floor", 0, 0.4),
              Solid(Box((3.0, 3.95, 1.3), (6.4, 0.1, 2.6), 0.0), "wall", 1, 0.7)]
    inst = 2
    for name, boxes, prims in _mess_scene(np.random.default_rng(seed), scan_dir):
        for b in boxes:
            if b.center[1] > 3.9:                               # куски стены из галереи - уже есть
                continue
            lab = "clutter" if "короб" in name else "furniture"
            solids.append(Solid(b, lab, inst, 0.5))
        for pr in prims:
            v, t = prim_mesh(pr)
            lo, hi = v.min(0), v.max(0)
            solids.append(Solid(Box(tuple((lo + hi) / 2), tuple(hi - lo + 1e-3), 0.0), "clutter",
                                inst, 0.5, mesh=(v, t)))
        inst += 1
    mesh = SceneMesh.from_solids(solids)
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.12, "min_range_m": 0.3}})
    sim = ScanSimulator(mesh, cfg.scanner, cfg.effects, sample_effect_params(cfg.effects, rng), rng)
    st = Station(0, 0, (3.0, -1.5, 1.6), 90.0)
    scan = sim.scan(st)
    p = scan.xyz_local[scan.valid] @ st.rotation().T + np.asarray(st.position)
    lab = scan.label[scan.valid]
    keep = (lab != LABEL_ID["floor"]) | (rng.random(len(lab)) < 0.15)
    p, lab = p[keep], lab[keep]
    colors = {LABEL_ID["wall"]: BLUE, LABEL_ID["floor"]: "#c9c7c0", LABEL_ID["furniture"]: YELLOW,
              LABEL_ID["clutter"]: MAGENTA}
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)
    for ax, (i, j, xl, yl, title) in zip(axes, ((0, 1, "x, м", "y, м", "Сверху"),
                                                (0, 2, "x, м", "z, м", "Спереди"))):
        for lid, col in colors.items():
            m = lab == lid
            ax.scatter(p[m, i], p[m, j], s=0.4, c=col, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.set_xlim(-0.2, 6.2)
    axes[0].set_ylim(-0.2, 4.2)
    axes[1].set_ylim(-0.05, 2.7)
    from matplotlib.lines import Line2D

    names = {LABEL_ID["wall"]: "стена", LABEL_ID["floor"]: "пол",
             LABEL_ID["furniture"]: "мебель", LABEL_ID["clutter"]: "беспорядок"}
    axes[0].legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=c, label=names[k])
                            for k, c in colors.items()], loc="upper right", fontsize=9)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return {"points": int(len(p)), "clutter_points": int((lab == LABEL_ID["clutter"]).sum())}


def _draw3d(ax, polys, Poly3DCollection, color=BLUE, alpha=0.95, fit=True):
    """Грани с простым освещением (Ламберт), чтобы читалась форма."""
    if not polys:
        return
    from matplotlib.colors import to_rgb

    P = np.array([p[:3] for p in polys])
    nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.45, -0.6, 0.66])
    light /= np.linalg.norm(light)
    shade = 0.45 + 0.55 * np.abs(nrm @ light)
    if isinstance(color, (list, tuple)) and len(color) == len(polys):   # цвет на грань
        base = np.array([to_rgb(c) for c in color])
    else:
        base = np.array(to_rgb(color))[None]
    fc = np.clip(base * shade[:, None] + (1 - shade[:, None]) * 0.12, 0, 1)
    pc = Poly3DCollection(polys, facecolors=np.c_[fc, np.full(len(fc), alpha)], edgecolor="none")
    ax.add_collection3d(pc)
    if fit:
        pts = np.concatenate(polys)
        lo, hi = pts.min(0), pts.max(0)
        c, r = (lo + hi) / 2, max(hi - lo) / 2 + 0.05
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(max(0, c[2] - r), c[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    ax.set_axis_off()


# ----------------------------------------------------------------------

def debug_scene(scene_dir: str | Path) -> dict:
    from .render import render_plan

    scene_dir = Path(scene_dir)
    out = scene_dir / "debug"
    out.mkdir(exist_ok=True)
    doc, cfg, lay, mesh, solids, station, scan = _rebuild(scene_dir)
    render_plan(lay, out / "plan.png")
    _, s_on = scan(True)
    _, s_off = scan(False)
    stats = {"scene": str(scene_dir), "triangles": int(len(mesh.triangles))}
    stats["panorama"] = panorama(s_on, out / "panorama.png")
    stats["edge_closeup"] = edge_closeup(lay, station, s_off, s_on, out / "edge_closeup.png", solids)
    stats["wall_flatness"] = wall_flatness(s_on, station, solids, out / "wall_flatness.png")
    stats["noise_vs_incidence"] = noise_vs_incidence(s_on, out / "noise_vs_incidence.png")
    furniture_gallery(out / "furniture.png")
    mess_gallery(out / "mess.png", cfg.interior.scan_dir)
    stats["mess_scan"] = mess_scan(out / "mess_scan.png", cfg.interior.scan_dir)
    stats["mess"] = doc["meta"].get("mess")
    stats["realism"] = {k: v for k, v in doc["meta"].get("realism", {}).items()
                        if k not in ("w", "phi", "w_lean", "phi_lean")}
    (out / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    return stats


__all__ = ["debug_scene"]
