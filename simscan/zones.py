"""Зоны вне увиденного: что за пространство там, где в облаке есть точки, а пола и потолка уровня
нет (лоджия со ступенью и пониженным потолком? комната за окном? улица?).

    python -m simscan zones объект.e57 [--out D] [--box x0,y0,x1,y1] [--min-area 1]

Растр - тот же, что у сети (netinput.rasterize_points). Помещения уровня - где видны его пол
или потолок (расширено на 30 см, закрытые дыры - тоже); зона - связная область с точками
(любые каналы, включая плотность и точки ниже пола) вне них, площадью от min_area м²; или
прямоугольник --box в мировых метрах. По сырым точкам каждой зоны (второй проход по файлу):
- профиль покрытия по высоте: доля клеток 5 см зоны, в которых есть точки на этой высоте (±1,5
  см); свой пол - самая покрытая горизонталь в -0,6..+0,35 м от пола уровня, свой потолок - в
  1,8 м..потолок+0,6; ступень пола и опускание потолка относительно уровня;
- увиденное по своим полу и потолку (как floor_vis/ceil_vis, но на их высотах);
- слепой круг: наибольшая закрытая дыра в своём полу - круг радиусом 0,3-1,2 м значит, что
  сканер стоял в зоне (снаружи через окно пол виден клочками, без такого круга);
- плотность точек на м² против помещений уровня, доля границы зоны у помещений.
Вывод - <файл>_zones/: zones.json, overview_L*.png (вход сети с зонами), zone_L*_N.png
(профиль по высоте и карта высоты пола) и zip рядом.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np

ZONE_MIN_M2 = 1.0
CELL_M = 0.05                          # клетка покрытия
SLAB_M = 0.015                         # полутолщина горизонтального слоя
PLANE_COVER = 0.15                     # горизонталь - плоскость, если покрывает столько зоны
STEP_M = 0.03                          # ступень пола / опускание потолка заметны от
DISC_R = (0.3, 1.2)                    # слепой круг сканера, м (высота 1-1,7 м, обзор до -60°)
MAX_ZONE_POINTS = 600_000


def apartment_mask(x: np.ndarray, px: float, reach_m: float = 0.3) -> np.ndarray:
    """Увиденное уровня: пол или потолок, расширено на reach_m, закрытые дыры заполнены."""
    import cv2

    seen = ((np.asarray(x[3], np.float32) > 0) | (np.asarray(x[4], np.float32) > 0)).astype(np.uint8)
    r = max(1, int(round(reach_m / px)))
    ev = cv2.dilate(seen, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return _fill_holes(ev > 0)


def _fill_holes(m: np.ndarray) -> np.ndarray:
    import cv2

    n, comp = cv2.connectedComponents((~m).astype(np.uint8), connectivity=4)
    border = np.unique(np.concatenate([comp[0], comp[-1], comp[:, 0], comp[:, -1]]))
    return m | ((comp > 0) & ~np.isin(comp, border))


def find_zones(x: np.ndarray, px: float, min_m2: float = ZONE_MIN_M2) -> tuple[np.ndarray, np.ndarray]:
    """-> (метки зон [H, W] int32, 0 - не зона; маска помещений уровня). Зона - точки вне
    помещений, сомкнутые на 0,5 м (редкие точки дальней стены за окном - тоже одна зона)."""
    import cv2

    x = np.asarray(x, np.float32)
    apt = apartment_mask(x, px)
    pts = ((x[0] > 0) | (x[1] > 0) | (x[2] > 0) | (x[5] > 0) | (x[7] > 0)) & ~apt
    r = max(1, int(round(0.25 / px)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)
    z = _fill_holes(cv2.morphologyEx(pts.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0) & ~apt
    n, lab = cv2.connectedComponents(z.astype(np.uint8), connectivity=8)
    area = np.bincount(lab.ravel(), minlength=n) * px * px
    keep = area >= min_m2
    keep[0] = False
    out = np.zeros_like(lab)
    for new, old in enumerate(np.flatnonzero(keep), 1):
        out[lab == old] = new
    return out.astype(np.int32), apt


def _cells(xy: np.ndarray, origin: np.ndarray, shape: tuple) -> np.ndarray:
    ij = np.floor((xy - origin) / CELL_M).astype(np.int64)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < shape[0])
    return np.where(ok, ij[:, 1] * shape[1] + ij[:, 0], -1)


def analyse_zone(xyz: np.ndarray, zone_xy_mask, level: dict, n_points: int | None = None) -> dict:
    """Сырые точки зоны (мир) и её маска на сетке CELL_M -> числа. zone_xy_mask = (маска [h, w]
    bool, начало (x0, y0) левого нижнего угла клетки [0, 0]); строки маски - по y вверх."""
    import cv2

    mask, origin = zone_xy_mask
    origin = np.asarray(origin, float)
    floor_z, ceil_z = float(level["floor_z"]), float(level["ceiling_z"])
    top = ceil_z - floor_z
    n_cells = int(mask.sum())
    res = {"area_m2": round(n_cells * CELL_M ** 2, 2), "points": int(n_points if n_points is not None else len(xyz))}
    if n_cells == 0 or len(xyz) == 0:
        return {**res, "verdict": "нет точек"}
    cell = _cells(xyz[:, :2], origin, mask.shape)
    inz = cell >= 0
    inz[inz] = mask.ravel()[cell[inz]]
    xyz, cell = xyz[inz], cell[inz]
    h = xyz[:, 2] - floor_z
    # профиль покрытия по высоте: шаг 1 см, слой ±1,5 см
    hs = np.round(np.arange(-1.0, top + 0.8, 0.01), 2)
    hb = np.floor((h + 1.0) / 0.01).astype(np.int64)
    okb = (hb >= 0) & (hb < len(hs))
    occ = np.zeros((len(hs), mask.size), bool)
    occ[hb[okb], cell[okb]] = True
    s = occ.copy()                                        # ±1 бин: слой ±1,5 см
    s[1:] |= occ[:-1]
    s[:-1] |= occ[1:]
    slab = s.sum(1) / n_cells
    res["profile"] = {"h_m": hs.tolist(), "cover": np.round(slab, 3).tolist()}

    def plane(lo, hi):
        sel = (hs >= lo) & (hs <= hi)
        if not sel.any() or slab[sel].max() < PLANE_COVER:
            return None, 0.0
        b = np.flatnonzero(sel)[int(np.argmax(slab[sel]))]
        near = h[np.abs(h - hs[b]) < 0.025]                 # слой насыщает соседние бины - высота по точкам
        return float(np.median(near)) if len(near) else float(hs[b]), float(slab[b])

    fh, fc = plane(-0.6, 0.35)                            # выше - ванна, стол, подоконник, не пол
    ch, cc = plane(1.8, top + 0.6)
    res["floor"] = None if fh is None else {"h_m": round(fh, 3), "z_m": round(floor_z + fh, 3), "cover": round(fc, 3),
                                            "step_m": round(fh, 3)}
    res["ceiling"] = None if ch is None else {"h_m": round(ch, 3), "z_m": round(floor_z + ch, 3), "cover": round(cc, 3),
                                              "drop_m": round(top - ch, 3)}
    # увиденное по своим полу и потолку (а по уровню - как у сети)
    def seen_at(hh):
        return np.zeros(mask.size, bool) if hh is None else np.bincount(cell[np.abs(h - hh) < 0.03], minlength=mask.size) > 0

    own_floor, own_ceil = seen_at(fh), seen_at(ch)
    lvl = seen_at(0.0) | seen_at(top)
    m1 = mask.ravel()
    res["seen_own"] = round(float((own_floor | own_ceil)[m1].mean()), 3)
    res["seen_level"] = round(float(lvl[m1].mean()), 3)
    res["floor_points_share"] = round(float(own_floor[m1].mean()), 3)
    # слепой круг в своём полу
    disc = None
    if fh is not None:
        fl = own_floor.reshape(mask.shape) & mask
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fl = cv2.morphologyEx(fl.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
        holes = _fill_holes(fl) & ~fl
        n, lab = cv2.connectedComponents(holes.astype(np.uint8), connectivity=4)
        best = None
        for q in range(1, n):
            m = (lab == q).astype(np.uint8)
            a = float(m.sum()) * CELL_M ** 2
            r_eq = math.sqrt(a / math.pi)
            if not DISC_R[0] <= r_eq <= DISC_R[1]:
                continue
            cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            per = sum(cv2.arcLength(c, True) for c in cnt) * CELL_M
            circ = 4 * math.pi * a / max(per * per, 1e-9)
            if circ >= 0.6 and (best is None or a > best["area_m2"]):
                ys, xs = np.nonzero(m)
                best = {"area_m2": round(a, 3), "radius_m": round(r_eq, 2), "circularity": round(circ, 2),
                        "center_xy": [round(float(origin[0] + (xs.mean() + 0.5) * CELL_M), 3),
                                      round(float(origin[1] + (ys.mean() + 0.5) * CELL_M), 3)]}
        disc = best
    res["scanner_circle"] = disc
    res["points_per_m2"] = round(res["points"] / max(res["area_m2"], 1e-9))
    res["verdict"] = verdict(res)
    return res


def verdict(r: dict) -> str:
    f, c = r.get("floor"), r.get("ceiling")
    scanned = r.get("scanner_circle") is not None or r.get("floor_points_share", 0) >= 0.5
    if f is None and c is None:
        return "ни пола, ни потолка: стены и предметы, видные через окно или дверь (не отснято)"
    if f is not None and f["step_m"] < -0.6 + 0.05:
        return "пол ниже уровня больше чем на 0,5 м: улица, двор или лестница вниз"
    steps = []
    if f is not None and abs(f["step_m"]) >= STEP_M:
        steps.append(f"пол {'выше' if f['step_m'] > 0 else 'ниже'} уровня на {abs(f['step_m']) * 100:.0f} см")
    if c is not None and abs(c["drop_m"]) >= STEP_M:
        steps.append(f"потолок {'ниже' if c['drop_m'] > 0 else 'выше'} уровня на {abs(c['drop_m']) * 100:.0f} см")
    if scanned and steps:
        return ("отснятое помещение со своими полом/потолком (" + ", ".join(steps) +
                "): лоджия, балкон, санузел на подиуме - сеть по полу/потолку уровня его не видит")
    if scanned:
        return "отснятое помещение на высотах уровня - проверить, почему пол/потолок не попали в вход сети"
    return ("пол или потолок видны клочками (" + (", ".join(steps) or "на высотах уровня") +
            "): пространство, видное через проём, сканер в нём не стоял")


def _zone_grid(lab_px: np.ndarray, k: int, frame: dict) -> tuple[np.ndarray, tuple]:
    """Метка зоны в растре сети -> маска на сетке CELL_M (строки по y вверх) и её начало."""
    import cv2

    px = frame["pixel_m"]
    ii, jj = np.nonzero(lab_px == k)
    i0, i1, j0, j1 = ii.min(), ii.max() + 1, jj.min(), jj.max() + 1
    sub = (lab_px[i0:i1, j0:j1] == k).astype(np.uint8)[::-1]          # строки снизу вверх
    s = px / CELL_M
    h, w = max(1, int(round(sub.shape[0] * s))), max(1, int(round(sub.shape[1] * s)))
    m = cv2.resize(sub, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    H = frame["height"]
    x0 = frame["origin_x"] + j0 * px
    y0 = frame["origin_y"] + (H - i1) * px
    return m, (x0, y0)


def _box_zone(box, frame: dict) -> np.ndarray:
    from .netinput import world_to_px

    x0, y0, x1, y1 = box
    lab = np.zeros((frame["height"], frame["width"]), np.int32)
    q = world_to_px(np.array([[x0, y0], [x1, y1]]), frame)
    ja, jb = sorted(np.clip(np.round(q[:, 0]).astype(int), 0, frame["width"] - 1))
    ia, ib = sorted(np.clip(np.round(q[:, 1]).astype(int), 0, frame["height"] - 1))
    lab[ia:ib + 1, ja:jb + 1] = 1
    return lab


def _figures(r: dict, level: dict, xyz: np.ndarray, grid, path: Path, title: str) -> None:
    from .debug import BLUE, INK2, ORANGE, _plt

    plt = _plt()
    fig, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.4]})
    pr = r.get("profile")
    if pr:
        ax[0].plot(pr["cover"], pr["h_m"], color=BLUE)
        top = level["ceiling_z"] - level["floor_z"]
        for hh, nm, col in ((0.0, "пол уровня", ORANGE), (top, "потолок уровня", INK2)):
            ax[0].axhline(hh, color=col, lw=1, ls="--")
            ax[0].annotate(nm, (1, hh), xycoords=("axes fraction", "data"), ha="right", va="bottom", fontsize=8, color=col)
        for key, nm in (("floor", "свой пол"), ("ceiling", "свой потолок")):
            if r.get(key):
                ax[0].axhline(r[key]["h_m"], color="k", lw=1)
                ax[0].annotate(f"{nm} {r[key]['h_m']:+.2f}", (0.02, r[key]["h_m"]), xycoords=("axes fraction", "data"),
                               va="bottom", fontsize=8)
        ax[0].axvline(PLANE_COVER, color="0.7", lw=0.8, ls=":")
        ax[0].set_xlabel("доля зоны с точками на этой высоте (слой ±1,5 см)")
        ax[0].set_ylabel("высота от пола уровня, м")
    m, origin = grid
    if len(xyz):
        h = xyz[:, 2] - level["floor_z"]
        low = h < 1.0                                     # нижняя поверхность: пол, ступень, подоконник
        cell = _cells(xyz[low, :2], np.asarray(origin, float), m.shape)
        ok = cell >= 0
        zmin = np.full(m.size, np.nan)
        np.fmin.at(zmin, cell[ok], h[low][ok])
        img = np.where(m.ravel(), zmin, np.nan).reshape(m.shape)
        ext = [origin[0], origin[0] + m.shape[1] * CELL_M, origin[1], origin[1] + m.shape[0] * CELL_M]
        im = ax[1].imshow(img, origin="lower", extent=ext, cmap="coolwarm", vmin=-0.3, vmax=0.3)
        fig.colorbar(im, ax=ax[1], label="нижняя точка в клетке от пола уровня, м")
        if r.get("scanner_circle"):
            c = r["scanner_circle"]
            ax[1].add_patch(plt.Circle(c["center_xy"], c["radius_m"], fill=False, color="k", lw=1.2))
        ax[1].set_aspect("equal")
        ax[1].set_title("карта высоты пола (кружок - слепой круг сканера)", fontsize=9)
    fig.suptitle(title + "\n" + r.get("verdict", ""), fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _overview(x: np.ndarray, lab: np.ndarray, apt: np.ndarray, path: Path) -> None:
    import cv2

    from .evaluate import _rgb

    im = (_rgb(x) * 255).astype(np.uint8)[..., ::-1].copy()
    cnt, _ = cv2.findContours(apt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(im, cnt, -1, (120, 120, 120), 1)
    for k in range(1, int(lab.max()) + 1):
        m = (lab == k).astype(np.uint8)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(im, cnt, -1, (0, 200, 255), 2)
        ys, xs = np.nonzero(m)
        cv2.putText(im, str(k), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(im, str(k), (int(xs.mean()) - 8, int(ys.mean()) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.imwrite(str(path), im)


def zones(e57, out=None, box=None, min_m2: float = ZONE_MIN_M2, log=print) -> Path:
    """E57 -> <out>/<имя>_zones/ (zones.json, картинки) и zip; возвращает путь к zip."""
    from .netinput import E57Source, rasterize_points

    e57 = Path(e57)
    root = Path(out or e57.parent) / f"{e57.stem}_zones"
    if root.exists():
        if not (root / "zones.json").exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} не пуст и не похож на прошлый разбор зон - не трогаю")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    src = E57Source(e57)
    try:
        rasters = rasterize_points(src, log=log)
        jobs = []                                     # (уровень, номер зоны, маска на сетке, начало)
        doc = {"file": e57.name, "levels": [], "params": {"min_area_m2": min_m2, "cell_m": CELL_M,
                                                         "plane_cover": PLANE_COVER, "box": box}}
        for li, r in enumerate(rasters, 1):
            x, fr = np.asarray(r["input"], np.float32), r["frame"]
            lab, apt = find_zones(x, fr["pixel_m"], min_m2)
            if box is not None:
                lab = _box_zone(box, fr)
            _overview(x, lab, apt, root / f"overview_L{li}.png")
            lv = {"level": f"L{li}", **{k: round(float(v), 3) for k, v in r["level"].items() if k in ("floor_z", "ceiling_z")},
                  "apartment_m2": round(float(apt.sum()) * fr["pixel_m"] ** 2, 1), "zones": []}
            doc["levels"].append(lv)
            for k in range(1, int(lab.max()) + 1):
                if (lab == k).any():
                    jobs.append((li, k, *_zone_grid(lab, k, fr), r["level"], lab, apt, fr))
        log(f"зон: {len(jobs)}; собираю их точки")
        bufs = [[] for _ in jobs]
        counts = [0] * len(jobs)
        rng = np.random.default_rng(0)
        for p in src.chunks():
            for q, (li, k, m, origin, level, *_rest) in enumerate(jobs):
                sel = (p[:, 2] > level["floor_z"] - 1.0) & (p[:, 2] < level["ceiling_z"] + 0.8)
                c = _cells(p[sel, :2], np.asarray(origin, float), m.shape)
                ok = c >= 0
                ok[ok] = m.ravel()[c[ok]]
                pts = p[sel][ok]
                counts[q] += len(pts)
                if len(pts) > MAX_ZONE_POINTS // 4:
                    pts = pts[rng.random(len(pts)) < (MAX_ZONE_POINTS // 4) / len(pts)]
                bufs[q].append(pts.astype(np.float32))
        # плотность в помещениях уровня - для сравнения
        for q, (li, k, m, origin, level, lab, apt, fr) in enumerate(jobs):
            xyz = np.concatenate(bufs[q]).astype(float) if bufs[q] else np.zeros((0, 3))
            if len(xyz) > MAX_ZONE_POINTS:
                xyz = xyz[rng.choice(len(xyz), MAX_ZONE_POINTS, replace=False)]
            res = analyse_zone(xyz, (m, origin), level, n_points=counts[q])
            px = fr["pixel_m"]
            res["touches_apartment"] = round(_touch_share(lab == k, apt, px), 3)
            res["bbox_world"] = [round(float(origin[0]), 3), round(float(origin[1]), 3),
                                 round(float(origin[0] + m.shape[1] * CELL_M), 3), round(float(origin[1] + m.shape[0] * CELL_M), 3)]
            name = f"zone_L{li}_{k}.png"
            _figures(res, level, xyz, (m, origin), root / name, f"L{li}, зона {k}: {res['area_m2']} м²")
            res = {"zone": k, "figure": name, **res}
            prof = res.pop("profile", None)
            doc["levels"][li - 1]["zones"].append(res)
            if prof:
                (root / f"zone_L{li}_{k}_profile.json").write_text(json.dumps(prof), encoding="utf-8")
            log(f"L{li} зона {k}: {res['area_m2']} м², пол {res.get('floor')}, потолок {res.get('ceiling')}, "
                f"круг сканера {res.get('scanner_circle')} -> {res['verdict']}")
    finally:
        src.close()
    (root / "zones.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    dst = root.with_suffix(".zip")
    shutil.make_archive(str(root), "zip", root.parent, root.name)
    return dst


def _touch_share(zone: np.ndarray, apt: np.ndarray, px: float) -> float:
    """Доля границы зоны ближе 0,3 м к помещениям уровня (окно, дверь, общая стена)."""
    import cv2

    z = zone.astype(np.uint8)
    ring = (cv2.dilate(z, np.ones((3, 3), np.uint8)) > 0) & ~zone
    r = max(1, int(round(0.3 / px)))
    near = cv2.dilate(apt.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)) > 0
    return float(near[ring].mean()) if ring.any() else 0.0
