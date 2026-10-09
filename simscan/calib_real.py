"""Отчёт для подгонки генератора под реальные объекты: всё, что нужно, из E57 в один zip.

    python -m simscan calib-real D:/scans/test.e57 D:/scans/test_lvl.e57 --out D:/calib
    -> D:/calib.zip (и каталог D:/calib)

Тот же код работает на синтетике (scan.e57 сцены) - реальное и синтетическое сравниваются
одной меркой. На каждый файл и уровень (<файл>/L1, L2 ...):
    input.npz      вход сети: 8 каналов, пиксель 2 см (netinput.rasterize_e57), float16
    slices.npz     занятость срезов по 10 см (пиксель 2 см, упакованные биты), z срезов,
                   пол, потолок, точек в колонке - по ним веса и карту можно пересчитать без файла
    heat.npz       тепловая карта и разметка стена / дверь / окно / шум (пиксель 1 см)
    low.npz        срез 3-8 см над местным полом (пиксель 1 см) и карта местного пола
    stats.json     сводка: профиль весов и качества срезов по высоте, площади классов, толщина
                   линий стен, вытянутые полосы шума внутри помещений (радиальные от стоянок),
                   оконные зоны, статистика каналов входа, пол, стоянки
    *.png          картинки для глаза
Плюс <файл>/info.json: размер, точки, стоянки, уровни, время.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

VERSION = 1


def _js(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def _wall_width_m(wall: np.ndarray, px: float) -> float | None:
    """Средняя толщина линий стен: площадь / длина (длина - половина периметра контуров)."""
    import cv2

    m = wall.astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    per = sum(cv2.arcLength(c, True) for c in cnts)
    return round(float(m.sum() / max(per / 2, 1) * px), 4) if per > 0 else None


def _streaks(noise: np.ndarray, free: np.ndarray, px: float) -> dict:
    """Вытянутые полосы шума внутри помещений (смешанные точки вдоль луча и т. п.):
    связные куски шума в свободном, длина >= 0,5 м, длина / ширина >= 8."""
    import cv2

    k = np.ones((3, 3), np.uint8)
    inside = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = (noise & (inside > 0)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    lens = []
    for c in range(1, n):
        if st[c, 4] < 10:
            continue
        ys, xs = np.nonzero(lab == c)
        p = np.c_[xs, ys].astype(float)
        p -= p.mean(0)
        ev = np.linalg.eigvalsh(np.cov(p.T)) if len(p) > 2 else np.array([0, 0])
        L = 4 * math.sqrt(max(ev[-1], 0)) * px
        Wd = 4 * math.sqrt(max(ev[0], 1e-6)) * px
        if L >= 0.5 and L / max(Wd, px) >= 8:
            lens.append(L)
    return {"count": len(lens), "total_m": round(float(sum(lens)), 2),
            "lengths_m": sorted(round(x, 2) for x in lens)[-30:]}


def _channel_stats(x: np.ndarray) -> dict:
    """Каналы входа по областям: грани (вертикально и занят низ), помещение (виден пол),
    прочее занятое. Для каждой - доля пикселей и квантили каналов."""
    from .netinput import CHANNELS

    x = x.astype(np.float32)
    face = (x[6] > 0.5) & ((x[0] > 0.5) | (x[2] > 0.5))
    room = (x[3] > 0) & ~face
    occ = (x[0] + x[1] + x[2] > 0) & ~face & ~room
    out = {}
    for name, m in (("faces", face), ("room", room), ("other_occupied", occ)):
        d = {"share": round(float(m.mean()), 5)}
        if m.any():
            for c, ch in enumerate(CHANNELS):
                v = x[c][m]
                d[ch["name"]] = [round(float(q), 3) for q in np.percentile(v, (10, 50, 90))] + \
                    [round(float(v.mean()), 3)]
        out[name] = d
    return out


def calib_file(path, out_dir, log=print) -> dict:
    import cv2

    from .cellplan import image_stations
    from .e57read import E57Reader
    from .netinput import rasterize_e57
    from .wallheat import (DOOR, LABEL_NAMES, NOISE, WALL, WINDOW, _plot_heat, _plot_labels, _plot_slices,
                           analyse_slices, slice_counts, slice_map)

    t0 = time.perf_counter()
    path = Path(path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with E57Reader(path) as r:
        n_pts = int(sum(s.points for s in r.scans))
        n_scans = len(r.scans)
    st = image_stations(path)
    info = {"file": path.name, "bytes": path.stat().st_size, "points": n_pts, "scans": n_scans,
            "stations_world": st.round(3).tolist(), "calib_version": VERSION}

    # 1) срезы 10 см, пиксель 1 см: карта, разметка, веса
    log(f"{path.name}: срезы и тепловая карта (1 см)")
    data, frame = slice_counts(path, px=0.01, log=log)
    info.update({"rotation_deg": frame["rotation_deg"], "heat_px": frame["px"],
                 "levels": [{k: d["level"][k] for k in ("floor_z", "ceiling_z")} for d in data]})
    stats_levels = []
    for li, L in enumerate(data, 1):
        d = out / f"L{li}"
        d.mkdir(exist_ok=True)
        A = analyse_slices(L, frame["px"])
        lab, linfo, heat = A["lab"], A["linfo"], A["heat"]
        np.savez_compressed(d / "heat.npz", heat=heat.astype(np.float16), labels=lab,
                            low=A["bands"]["low"].astype(np.float16), top=A["bands"]["top"].astype(np.float16),
                            free=L["free"], weights=A["w"].astype(np.float32), z0=L["z0"])
        _plot_heat(heat, L, A["w"], A["info"], frame["px"], d / "heat.png")
        _plot_slices(A["occ"], L, A["w"], d / "slices.png")
        _plot_labels(lab, linfo, frame["px"], d / "labels.png")
        # занятость срезов на 2 см (ИЛИ по 2x2) - компактно, для пересчёта весов
        occ = A["occ"]
        S, H, W = occ.shape
        o2 = occ[:, :H // 2 * 2, :W // 2 * 2].reshape(S, H // 2, 2, W // 2, 2).any(axis=(2, 4))
        np.savez_compressed(d / "slices.npz", occ_bits=np.packbits(o2, axis=-1), shape=np.array(o2.shape),
                            z0=L["z0"], slice_m=L["slice_m"],
                            floor=L["floor"][::2, ::2], ceil=L["ceil"][::2, ::2],
                            points=np.minimum(L["points"][::2, ::2], 65535).astype(np.uint16))
        px = frame["px"]
        info_w = A["info"]
        areas = {LABEL_NAMES[k]: round(float((lab == k).sum()) * px * px, 3) for k in (WALL, DOOR, WINDOW, NOISE)}
        occupied = heat > 0.04
        s = {"level": li, "floor_z": L["level"]["floor_z"], "ceiling_z": L["level"]["ceiling_z"],
             "slices": [{"z": round(float(z), 2), "weight": round(float(a), 3), "f1": round(float(f), 3),
                         "precision": round(float(p), 3), "recall": round(float(q), 3)}
                        for z, a, f, p, q in zip(L["z0"], A["w"], info_w["f1"], info_w["precision"],
                                                 info_w["recall"])],
             "label_areas_m2": areas,
             "noise_share_of_labelled": round(float((lab == NOISE).sum() / max((lab > 0).sum(), 1)), 4),
             "wall_line_width_m": _wall_width_m(lab == WALL, px),
             "streaks_in_rooms": _streaks(lab == NOISE, L["free"], px),
             "windows": [{"width_m": round(w["width_px"] * px, 3), "depth_m": round(w["depth_px"] * px, 3)}
                         for w in linfo["windows"]],
             "doors": [{"width_m": round(dd["width_px"] * px, 3)} for dd in linfo["doors"]],
             "heat_quantiles_occupied": [round(float(q), 3) for q in np.percentile(heat[occupied], (10, 50, 90))]
             if occupied.any() else None}
        stats_levels.append(s)
        log(f"  уровень {li}: стены {areas['стена']} м², шум {areas['шум']} м², полос {s['streaks_in_rooms']['count']}")

    # 2) вход сети (2 см, мировая СК) - тем же растеризатором, что датасет
    log(f"{path.name}: вход сети (2 см)")
    for li, r in enumerate(rasterize_e57(path, log=log), 1):
        d = out / f"L{li}"
        d.mkdir(exist_ok=True)
        np.savez_compressed(d / "input.npz", input=r["input"], frame=json.dumps(r["frame"]),
                            level=json.dumps(r["level"]))
        x = r["input"].astype(np.float32)
        img = (np.clip(np.stack([x[0], x[2], x[3] * 0.6], -1), 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(str(d / "input.png"), img[..., ::-1])
        if li <= len(stats_levels):
            stats_levels[li - 1]["input_channels"] = _channel_stats(r["input"])

    # 3) срез 3-8 см над местным полом
    log(f"{path.name}: срез у пола")
    tmp = out / "_low"
    res = slice_map(path, tmp, log=log)
    for li, lv in enumerate(res["levels"], 1):
        sfx = "" if len(res["levels"]) == 1 else f"_L{li}"
        d = out / f"L{li}"
        d.mkdir(exist_ok=True)
        cnt = np.load(tmp / f"slice{sfx}.npy")
        fl = np.load(tmp / f"floor{sfx}.npy")
        np.savez_compressed(d / "low.npz", count=np.minimum(cnt, 65535).astype(np.uint16), floor=fl.astype(np.float32))
        shutil.copy(tmp / f"slice_view{sfx}.png", d / "low.png")
        if li <= len(stats_levels):
            stats_levels[li - 1]["low_slice"] = {"points": lv["points"], "floor_cells_sd_mm": lv["floor_cells_sd_mm"]}
    shutil.rmtree(tmp, ignore_errors=True)

    for li, s in enumerate(stats_levels, 1):
        (out / f"L{li}" / "stats.json").write_text(json.dumps(s, indent=1, ensure_ascii=False, default=_js),
                                                   encoding="utf-8")
    info["seconds"] = round(time.perf_counter() - t0, 1)
    (out / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=_js), encoding="utf-8")
    log(f"{path.name}: готово за {info['seconds']} с")
    return {"info": info, "levels": stats_levels}


def calib_real(paths, out_dir, log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"calib_version": VERSION, "files": []}
    for p in paths:
        p = Path(p)
        name = p.stem if p.stem not in ("scan", "sample") else f"{p.parent.name}_{p.stem}"
        try:
            r = calib_file(p, out / name, log=log)
            summary["files"].append({"name": name, "points": r["info"]["points"], "levels": len(r["levels"]),
                                     "seconds": r["info"]["seconds"]})
        except Exception as e:                        # один плохой файл не губит отчёт по остальным
            import traceback

            (out / f"{name}_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            summary["files"].append({"name": name, "error": repr(e)})
            log(f"{p.name}: ошибка {e!r} (подробности в {name}_error.txt)")
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    z = shutil.make_archive(str(out), "zip", root_dir=out)
    log(f"архив: {z} ({Path(z).stat().st_size / 1e6:.1f} МБ)")
    return Path(z)


# ----------------------------------------------------------------------
# Сравнение отчётов (реальные против синтетики)
# ----------------------------------------------------------------------

def _level_metrics(st: dict) -> dict:
    """Числа уровня, которые сравниваются между реальным и синтетикой (на м² пола - чтобы
    квартиры разного размера были сравнимы)."""
    out = {}
    for band, (a, b) in {"низ": (0, 0.9), "середина": (0.9, 2.0), "верх": (2.0, 99)}.items():
        s = [x for x in st["slices"] if a <= x["z"] < b]
        for k in ("f1", "precision", "recall"):
            out[f"{k} {band}"] = float(np.mean([x[k] for x in s])) if s else np.nan
    areas = st["label_areas_m2"]
    wall = max(areas.get("стена", 0), 1e-6)
    out["шум / стены (площадь)"] = areas.get("шум", 0) / wall
    out["доля шума в размеченном"] = st["noise_share_of_labelled"]
    out["толщина линии стены, см"] = (st["wall_line_width_m"] or np.nan) * 100
    wall_len = wall / max(st["wall_line_width_m"] or 0.035, 1e-3)          # м линий стен
    out["полос шума на 10 м стен"] = st["streaks_in_rooms"]["count"] / max(wall_len / 10, 1e-6)
    out["длина полос на 10 м стен, м"] = st["streaks_in_rooms"]["total_m"] / max(wall_len / 10, 1e-6)
    out["шум на 10 м стен, м²"] = areas.get("шум", 0) / max(wall_len / 10, 1e-6)
    out["окон: глубина зоны, м"] = float(np.mean([w["depth_m"] for w in st["windows"]])) if st["windows"] else np.nan
    ic = st.get("input_channels", {})
    for reg, nm in (("faces", "грани"), ("other_occupied", "прочее занятое")):
        if reg in ic and "occ_low" in ic[reg]:
            out[f"{nm}: доля пикселей"] = ic[reg]["share"]
            out[f"{nm}: occ_low (ср.)"] = ic[reg]["occ_low"][3]
            out[f"{nm}: occ_high (ср.)"] = ic[reg]["occ_high"][3]
            out[f"{nm}: density (медиана)"] = ic[reg]["density"][1]
            out[f"{nm}: vert_frac (ср.)"] = ic[reg]["vert_frac"][3]
    out["пол неровен, мм"] = st.get("low_slice", {}).get("floor_cells_sd_mm", np.nan)
    out["точек в срезе 3-8 см на м стен"] = st.get("low_slice", {}).get("points", 0) / max(wall_len, 1e-6)
    return out


def compare_calib(groups: dict, out_png=None) -> dict:
    """groups: {"реальные": [каталоги отчётов], "синтетика": [...]} -> {группа: {метрика: (ср, мин, макс)}}."""
    res = {}
    for g, roots in groups.items():
        rows = []
        for r in roots:
            for f in sorted(Path(r).rglob("L*/stats.json")):
                rows.append(_level_metrics(json.loads(f.read_text(encoding="utf-8"))))
        keys = rows[0].keys() if rows else []
        res[g] = {k: (float(np.nanmean([x.get(k, np.nan) for x in rows])),
                      float(np.nanmin([x.get(k, np.nan) for x in rows])),
                      float(np.nanmax([x.get(k, np.nan) for x in rows]))) for k in keys}
        res[g]["_n"] = len(rows)
    if out_png is not None:
        from .debug import INK, _plt

        plt = _plt()
        names = list(groups)
        keys = [k for k in res[names[0]] if not k.startswith("_")]
        fig, ax = plt.subplots(figsize=(11, 0.42 * len(keys) + 1))
        y = np.arange(len(keys))
        colors = ["#2a6fb0", "#e0702b", "#3a9a5b"]
        for gi, g in enumerate(names):
            ref = np.array([max(abs(res[names[0]][k][0]), 1e-9) for k in keys])
            m = np.array([res[g][k][0] for k in keys]) / ref
            lo = np.array([res[g][k][1] for k in keys]) / ref
            hi = np.array([res[g][k][2] for k in keys]) / ref
            yy = y + (gi - (len(names) - 1) / 2) * 0.25
            ax.errorbar(m, yy, xerr=[m - lo, hi - m], fmt="o", color=colors[gi % 3], label=f"{g} (уровней {res[g]['_n']})",
                        capsize=2, ms=5)
        ax.axvline(1, color="#999", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(keys, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel(f"отношение к «{names[0]}» (1 - совпадает; разброс - мин..макс по уровням)", fontsize=8)
        ax.set_title("Сверка синтетики с реальными объектами одной меркой (calib-real)", loc="left", fontsize=10,
                     color=INK)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(out_png, dpi=110)
        plt.close(fig)
    return res
