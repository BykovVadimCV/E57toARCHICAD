"""Анализ реального E57: всё, что нужно для калибровки синтетики и конвертера, в маленьком архиве.

    python -m simscan inspect-e57 D:/scans/kvartira.e57 --out D:/e57_report
    -> D:/e57_report.zip  (несколько МБ; его и пересылать)

Файл читается потоково, кусками по --chunk точек: память не зависит от размера файла
(на 20 ГБ - те же сотни МБ). Что извлекается:

  формат       версия, поля точек и их типы (float32/64, scaled integer и шаг), есть ли
               RGB / недействительные лучи / сетка row-column, встроенные снимки (только
               число и размер - сами снимки не копируются), байт на точку;
  станции      число точек, сетка строк и столбцов, шаг по азимуту и углу места, поза,
               наклон, высота над полом, пол и потолок, доля лучей без отклика;
  сенсор       распределения дальности и интенсивности; шум дальности по дальности и углу
               падения (отклонение точки от плоскости соседей по сетке); смешанные пиксели
               на перепадах (доля точек «между» поверхностями); скорость чтения и память;
  план         срез 1,0-1,6 м и вход сети (псевдочертёж по свободному пространству,
               rasterize.py) в СК файла;
  sample.npz   (по умолчанию; --no-sample - без него) прореженное облако: по станции не
               больше 200 тыс. точек в вокселях 4 см, xyz + интенсивность, без цвета.

Что в архив не попадает: цвет точек, фотографии, серийные номера (вычищаются из дерева
заголовка), полное облако.
"""
from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

SUPPORTED = ("cartesianX", "cartesianY", "cartesianZ", "sphericalRange", "sphericalAzimuth",
             "sphericalElevation", "intensity", "rowIndex", "columnIndex", "cartesianInvalidState",
             "sphericalInvalidState")
RANGE_BINS = (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 1e9)
INC_BINS = (0.0, 30.0, 50.0, 65.0, 75.0, 85.0)


# ----------------------------------------------------------------------
# Заголовок и чтение - e57read.py (чистый numpy, без pye57)
# ----------------------------------------------------------------------

def _blob_bytes(tree) -> int:
    if isinstance(tree, dict):
        return int(tree.get("blob_bytes", 0)) + sum(_blob_bytes(v) for v in tree.values())
    if isinstance(tree, list):
        return sum(_blob_bytes(v) for v in tree)
    return 0


# ----------------------------------------------------------------------
# Потоковое чтение
# ----------------------------------------------------------------------

def _quat_R(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ScanStats:
    """Накопители одной станции: всё в фиксированной памяти."""

    def __init__(self, n_points: int, rows: int | None, cols: int | None, keep_points: int,
                 keep_pairs: int, rng):
        self.rng = rng
        self.n = 0
        self.valid = 0
        self.p_keep = min(1.0, keep_points / max(n_points, 1))
        self.p_pair = min(1.0, 2 * keep_pairs / max(n_points, 1))
        self.keep_pairs = keep_pairs
        self.kept = []                                   # (xyz local float32, intensity)
        self.noise = []                                  # (range, incidence°, отклонение по нормали)
        self.jumps = [0, 0]                              # перепадов, из них «между»
        self.jump_pos = np.zeros(20, np.int64)
        self.az_step, self.el_step = [], []
        self.rows, self.cols = rows, cols
        self.pano = None
        if rows and cols:
            k = max(1, int(math.ceil(max(rows / 400, cols / 1600))))
            self.k = k
            shape = (rows // k + 1, cols // k + 1)
            self.pano = {"r": np.zeros(shape), "i": np.zeros(shape), "n": np.zeros(shape),
                         "all": np.zeros(shape)}

    def add(self, xyz: np.ndarray, inten, row, col, valid: np.ndarray) -> None:
        self.n += len(xyz)
        self.valid += int(valid.sum())
        if self.pano is not None and row is not None:
            P = self.pano
            shape = P["n"].shape
            flat = (row.astype(np.int64) // self.k) * shape[1] + col.astype(np.int64) // self.k
            size = shape[0] * shape[1]
            fv = flat[valid]
            xv = xyz[valid]
            r = np.sqrt(np.einsum("ij,ij->i", xv, xv))
            P["all"] += np.bincount(flat, minlength=size).reshape(shape)
            P["n"] += np.bincount(fv, minlength=size).reshape(shape)
            P["r"] += np.bincount(fv, weights=r, minlength=size).reshape(shape)
            if inten is not None:
                P["i"] += np.bincount(fv, weights=inten[valid], minlength=size).reshape(shape)
        keep = valid & (self.rng.random(len(xyz)) < self.p_keep)
        self.kept.append((xyz[keep].astype(np.float32),
                          None if inten is None else inten[keep].astype(np.float32)))
        if row is not None and col is not None:
            self._grid(xyz, row.astype(np.int64), col.astype(np.int64), valid)

    def _grid(self, xyz, row, col, valid) -> None:
        """Соседи по сетке внутри куска: шаг углов, шум относительно плоскости, перепады."""
        R = int(row.max()) + 2
        idx = np.flatnonzero(valid)
        if len(idx) < 100:
            return
        key = col[idx] * R + row[idx]
        order = np.argsort(key, kind="stable")
        ks, ids = key[order], idx[order]

        # центры - случайное подмножество: соседи ищутся по всем точкам куска, но считаем
        # только для центров (статистике хватает сотен тысяч, а время падает в разы)
        cen = np.flatnonzero(self.rng.random(len(ks)) < self.p_pair)
        ks_all = ks
        ks, ids = ks[cen], ids[cen]

        def nb(dk):                                      # сосед по всем точкам куска
            t = ks + dk
            j = np.clip(np.searchsorted(ks_all, t), 0, len(ks_all) - 1)
            return np.where(ks_all[j] == t, ids_all[j], -1)

        ids_all = idx[order]
        up, dn, lf, rt = nb(-1), nb(1), nb(-R), nb(R)
        up2, dn2, lf2, rt2 = nb(-2), nb(2), nb(-2 * R), nb(2 * R)
        p = xyz[ids]
        r = np.linalg.norm(p, axis=1)
        # шаг углов
        az = np.arctan2(p[:, 1], p[:, 0])
        el = np.arctan2(p[:, 2], np.hypot(p[:, 0], p[:, 1]))
        m = rt >= 0
        if m.any():
            pr = xyz[rt[m]]
            daz = np.abs(np.angle(np.exp(1j * (np.arctan2(pr[:, 1], pr[:, 0]) - az[m]))))
            self.az_step.append(np.median(np.degrees(daz)))
        m = dn >= 0
        if m.any():
            pd = xyz[dn[m]]
            del_ = np.abs(np.arctan2(pd[:, 2], np.hypot(pd[:, 0], pd[:, 1])) - el[m])
            self.el_step.append(np.median(np.degrees(del_)))
        # шум: отклонение точки от плоскости через 4 соседей, вдоль нормали
        m4 = (up >= 0) & (dn >= 0) & (lf >= 0) & (rt >= 0)
        sel = np.flatnonzero(m4)
        if len(sel):
            pu, pd, pl, pr = xyz[up[sel]], xyz[dn[sel]], xyz[lf[sel]], xyz[rt[sel]]
            c = p[sel]
            nrm = np.cross(pd - pu, pr - pl)
            nn = np.linalg.norm(nrm, axis=1)
            ok = nn > 1e-12
            nrm = nrm[ok] / nn[ok, None]
            mean4 = ((pu + pd + pl + pr) / 4)[ok]
            dev = np.abs(np.einsum("ij,ij->i", c[ok] - mean4, nrm))
            ray = c[ok] / np.maximum(r[sel][ok, None], 1e-9)
            cosi = np.abs(np.einsum("ij,ij->i", ray, nrm))
            # перепады больше 2 см у соседей - кромки, не шум
            span = np.maximum.reduce([np.abs(np.linalg.norm(q, axis=1) - r[sel])
                                      for q in (pu, pd, pl, pr)])[ok]
            flat = span < 0.05 + 0.02 * r[sel][ok]
            self.noise.append(np.c_[r[sel][ok][flat], np.degrees(np.arccos(np.clip(cosi[flat], 0, 1))),
                                    dev[flat]].astype(np.float32))
        # перепады: a2 a | c | b b2. Перепад - это |b - a| > 15 см при ровных поверхностях
        # с обеих сторон (|a - a2| и |b - b2| меньше 1/5 перепада). Без этого условия в счёт
        # шли скользящие поверхности (дальний пол), где средняя точка честно лежит посередине.
        for a, b, a2, b2 in ((up, dn, up2, dn2), (lf, rt, lf2, rt2)):
            m = (a >= 0) & (b >= 0) & (a2 >= 0) & (b2 >= 0)
            nrm_ = lambda q: np.linalg.norm(xyz[q[m]], axis=1)  # noqa: E731
            ra, rb, ra2, rb2, rc = nrm_(a), nrm_(b), nrm_(a2), nrm_(b2), r[m]
            lo, hi = np.minimum(ra, rb), np.maximum(ra, rb)
            jump = (hi - lo > 0.15) & (np.maximum(np.abs(ra - ra2), np.abs(rb - rb2)) < 0.2 * (hi - lo))
            between = jump & (rc > lo + 0.03) & (rc < hi - 0.03)
            self.jumps[0] += int(jump.sum())
            self.jumps[1] += int(between.sum())
            if between.any():
                # 0 - точка у той поверхности, что ближе к сканеру
                t = (rc - lo)[between] / (hi - lo)[between]
                self.jump_pos += np.bincount(np.clip((t * 20).astype(int), 0, 19), minlength=20)

    def points(self):
        xyz = np.concatenate([k[0] for k in self.kept]) if self.kept else np.zeros((0, 3), np.float32)
        it = None
        if self.kept and self.kept[0][1] is not None:
            it = np.concatenate([k[1] for k in self.kept])
        return xyz, it

    def noise_table(self):
        if not self.noise:
            return None
        a = np.concatenate(self.noise)
        if len(a) > self.keep_pairs:
            a = a[self.rng.choice(len(a), self.keep_pairs, replace=False)]
        return a


def _robust_sigma(x: np.ndarray) -> float:
    """Устойчивое СКО по модулям отклонений (|N(0, s)|: медиана = 0,6745 s)."""
    return float(1.4826 * np.median(np.abs(x))) if len(x) else float("nan")


def noise_summary(tab: np.ndarray) -> dict:
    """СКО дальности по дальности и углу падения, мм. Отклонение от плоскости соседей
    пересчитывается в дальность: делим на cos(угла) и на sqrt(1,25) (шум четырёх соседей)."""
    out = {"range_bins_m": list(RANGE_BINS[:-1]), "incidence_bins_deg": list(INC_BINS[:-1]),
           "sigma_mm": [], "count": []}
    for r0, r1 in zip(RANGE_BINS[:-1], RANGE_BINS[1:]):
        row, cnt = [], []
        for a0, a1 in zip(INC_BINS[:-1], INC_BINS[1:]):
            m = (tab[:, 0] >= r0) & (tab[:, 0] < r1) & (tab[:, 1] >= a0) & (tab[:, 1] < a1)
            if m.sum() < 200:
                row.append(None)
            else:
                cos = np.cos(np.radians(tab[m, 1]))
                row.append(round(_robust_sigma(tab[m, 2] / np.maximum(cos, 0.1)) / math.sqrt(1.25) * 1000, 3))
            cnt.append(int(m.sum()))
        out["sigma_mm"].append(row)
        out["count"].append(cnt)
    return out


def _memory_mb() -> float | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:                                   # noqa: BLE001
        pass
    try:
        import resource
        import sys

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e6 if sys.platform == "darwin" else r / 1e3
    except Exception:                                   # noqa: BLE001
        return None


# ----------------------------------------------------------------------

def inspect_e57(path, out_dir, chunk: int = 2_000_000, keep_points: int = 1_500_000,
                keep_total: int = 12_000_000,
                keep_pairs: int = 300_000, sample: bool = True, sample_voxel_m: float = 0.04,
                sample_per_scan: int = 200_000, figures: bool = True, zip_result: bool = True,
                log=print) -> dict:
    from .e57read import E57Reader

    path, out = Path(path), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    t_all = time.perf_counter()
    report = {"file": {"name": path.name, "bytes": path.stat().st_size}, "scans": []}
    stations, samples = [], []
    with E57Reader(path) as e57:
        tree = e57.tree
        hdr = {k: tree.get(k) for k in ("formatName", "versionMajor", "versionMinor",
                                         "e57LibraryVersion", "coordinateMetadata")
               if isinstance(tree, dict) and k in tree}
        images = tree.get("images2D", []) if isinstance(tree, dict) else []
        report["file"].update(hdr)
        report["file"]["page_size"] = e57.page
        report["images2D"] = {"count": len(images) if isinstance(images, list) else None,
                              "bytes": _blob_bytes(images)}
        total_points = 0
        for i, h in enumerate(e57.scans):
            fields = [f for f in h.field_names if f in SUPPORTED]
            cart = all(f in fields for f in ("cartesianX", "cartesianY", "cartesianZ"))
            if not cart:
                fields = [f for f in fields if not f.startswith("cartesian")]
            n_pts = h.points
            total_points += n_pts
            scan_tree = h.tree
            rows = cols = None
            ib = h.index_bounds
            if isinstance(ib, dict) and "rowMaximum" in ib and "columnMaximum" in ib:
                rows = int(ib["rowMaximum"]) - int(ib.get("rowMinimum", 0)) + 1
                cols = int(ib["columnMaximum"]) - int(ib.get("columnMinimum", 0)) + 1
            if rows is None and "rowIndex" in fields and "columnIndex" in fields:
                fs = {f.name: f for f in h.fields}            # нет indexBounds - по прототипу
                rows = fs["rowIndex"].maximum - fs["rowIndex"].minimum + 1
                cols = fs["columnIndex"].maximum - fs["columnIndex"].minimum + 1
            if "rowIndex" not in fields or "columnIndex" not in fields:
                rows = cols = None
            has_pose = h.rotation is not None or h.translation is not None
            R = _quat_R(h.rotation) if h.rotation is not None else np.eye(3)
            t = np.asarray(h.translation, float) if h.translation is not None else np.zeros(3)
            # выборка на станцию - не больше общего бюджета / число станций (память не растёт)
            keep = min(keep_points, max(200_000, keep_total // max(len(e57.scans), 1)))
            if len(e57.scans) == 1:                     # сведённое облако: плану нужна плотность
                keep = keep_total
            st = ScanStats(n_pts, rows, cols, keep, keep_pairs, rng)
            t0 = time.perf_counter()
            for d in e57.iter_points(i, fields, chunk):
                if cart:
                    xyz = np.c_[d["cartesianX"], d["cartesianY"], d["cartesianZ"]]
                    inv = d.get("cartesianInvalidState")
                else:
                    rr, az, el = d["sphericalRange"], d["sphericalAzimuth"], d["sphericalElevation"]
                    xyz = np.c_[rr * np.cos(el) * np.cos(az), rr * np.cos(el) * np.sin(az), rr * np.sin(el)]
                    inv = d.get("sphericalInvalidState")
                valid = np.ones(len(xyz), bool) if inv is None else (inv == 0)
                valid &= np.isfinite(xyz).all(1) & (np.abs(xyz).sum(1) > 0)
                st.add(xyz, d.get("intensity"), d.get("rowIndex"), d.get("columnIndex"), valid)
            secs = time.perf_counter() - t0
            xyz_l, inten = st.points()
            r_l = np.linalg.norm(xyz_l, axis=1)
            world_coords = bool(len(r_l) and not has_pose and np.median(r_l) > 50)
            xyz_w = (xyz_l.astype(np.float64) @ R.T + t).astype(np.float32)
            stations.append({"center": t, "points": xyz_w})
            tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
            info = {
                "index": i, "name": h.name,
                "points": int(n_pts), "valid": int(st.valid),
                "valid_fraction": round(st.valid / max(st.n, 1), 4),
                "fields": {f.name: f.describe() for f in h.fields},
                "has_color": "colorRed" in h.field_names,
                "grid_rows_cols": [rows, cols] if rows else None,
                "grid_fill": round(n_pts / (rows * cols), 4) if rows else None,
                "azimuth_step_deg": round(float(np.median(st.az_step)), 5) if st.az_step else None,
                "elevation_step_deg": round(float(np.median(st.el_step)), 5) if st.el_step else None,
                "pose": {"translation": t.round(4).tolist(),
                         "rotation_wxyz": None if h.rotation is None else
                         [round(float(v), 6) for v in h.rotation]} if has_pose else None,
                "tilt_deg": round(tilt, 3),
                "points_in_world_coords": world_coords,
                "range_m": {"min": round(float(r_l.min()), 3) if len(r_l) else None,
                            "p50": round(float(np.median(r_l)), 3) if len(r_l) else None,
                            "p99": round(float(np.percentile(r_l, 99)), 3) if len(r_l) else None,
                            "max": round(float(r_l.max()), 3) if len(r_l) else None},
                "intensity": None if inten is None or not len(inten) else {
                    "min": float(inten.min()), "p1": float(np.percentile(inten, 1)),
                    "p50": float(np.median(inten)), "p99": float(np.percentile(inten, 99)),
                    "max": float(inten.max())},
                "jumps": {"count": st.jumps[0], "between_fraction":
                          round(st.jumps[1] / max(st.jumps[0], 1), 4),
                          "between_position_hist": st.jump_pos.tolist()},
                "read_seconds": round(secs, 2), "read_points_per_s": int(n_pts / max(secs, 1e-9)),
            }
            from .rasterize import _plane_z

            rel = xyz_w[:, 2] - t[2]
            fz, cz = _plane_z(rel, -2.5, -0.3), _plane_z(rel, 0.3, 4.0)
            info["floor_rel_m"] = None if fz is None else round(fz, 3)
            info["ceiling_rel_m"] = None if cz is None else round(cz, 3)
            if fz is not None:
                fl = np.abs(rel - fz) < 0.05
                info["floor_flatness_mm"] = round(float(np.std(rel[fl] - fz)) * 1000, 2) if fl.any() else None
            tab = st.noise_table()
            if tab is not None:
                info["noise"] = noise_summary(tab)
                st.noise_tab = tab
            report["scans"].append(info)
            st.kept = []                                # точки уже в stations
            samples.append((st, inten))
            log(f"станция {i + 1}/{len(e57.scans)}: {n_pts / 1e6:.1f} млн точек, {secs:.0f} с")
    total_s = time.perf_counter() - t_all
    img = report["images2D"]["bytes"] or 0
    report["file"]["points_total"] = int(total_points)
    report["file"]["bytes_per_point"] = round((report["file"]["bytes"] - img) / max(total_points, 1), 2)
    report["performance"] = {"seconds_total": round(total_s, 1),
                             "points_per_s": int(total_points / max(total_s, 1e-9)),
                             "peak_memory_mb": _memory_mb(), "chunk_points": chunk}
    floors = [c["center"][2] + s["floor_rel_m"] for c, s in zip(stations, report["scans"])
              if s["floor_rel_m"] is not None]
    report["levelling"] = {"floor_z_std_mm": round(float(np.std(floors)) * 1000, 2) if floors else None,
                           "station_height_m": [s["floor_rel_m"] and round(-s["floor_rel_m"], 3)
                                                for s in report["scans"]]}
    # сведённое облако (Cyclone REGISTER 360 «unified»): один скан без сетки; станции -
    # только в позах снимков. Лучи от станций восстановить нельзя - план по покрытию
    img_st = _image_stations(tree)
    unified = len(report["scans"]) == 1 and report["scans"][0]["grid_rows_cols"] is None \
        and len(img_st) >= 2
    if unified:
        report["unified"] = _unified_info(stations[0]["points"], img_st, samples[0][0].p_keep)
        plan = _plan_coverage(stations[0]["points"], report["unified"], img_st, out)
    else:
        plan = _plan(stations, out) if stations else None
    if plan:
        report["plan"] = plan["info"]
    if figures:
        _figures(report, stations, samples, plan, out)
    if sample:
        _write_sample(stations, samples, out / "sample.npz", sample_voxel_m, sample_per_scan, rng)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=_js),
                                     encoding="utf-8")
    (out / "header_tree.json").write_text(json.dumps(tree, indent=1, ensure_ascii=False, default=_js),
                                          encoding="utf-8")
    (out / "summary.txt").write_text(_summary(report), encoding="utf-8")
    if zip_result:
        report["zip"] = shutil.make_archive(str(out), "zip", root_dir=out)
    return report


def _js(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _plan(stations, out: Path) -> dict | None:
    """Псевдочертёж в СК файла: кадр по 1-99 % точек, пиксель 1 см (не больше 2048 px)."""
    import cv2

    from .groundtruth import RasterFrame
    from .rasterize import free_space_input

    allp = np.concatenate([s["points"][:: max(1, len(s["points"]) // 200_000)] for s in stations])
    if len(allp) < 1000:
        return None
    lo = np.percentile(allp[:, :2], 1, axis=0) - 1.0
    hi = np.percentile(allp[:, :2], 99, axis=0) + 1.0
    px = max(0.01, float((hi - lo).max()) / 2048)
    w, h = int(math.ceil((hi[0] - lo[0]) / px)), int(math.ceil((hi[1] - lo[1]) / px))
    frame = RasterFrame(float(lo[0]), float(lo[1]), px, w, h)
    res = free_space_input(stations, frame)
    cv2.imwrite(str(out / "input.png"), res["image"])
    free_m2 = float(res["free"].sum()) * px * px
    return {"frame": frame, "res": res,
            "info": {"pixel_mm": round(px * 1000, 2), "size_px": [h, w],
                     "free_area_m2": round(free_m2, 2), "stations": res["stations"]}}


def _image_stations(tree) -> np.ndarray:
    """Положения станций по позам встроенных снимков (по 6 граней куба на станцию)."""
    pos = set()
    for im in tree.get("images2D", []) if isinstance(tree, dict) else []:
        t = (im.get("pose") or {}).get("translation") if isinstance(im, dict) else None
        if isinstance(t, dict):
            pos.add(tuple(round(float(t.get(k, 0.0)), 3) for k in "xyz"))
    return np.array(sorted(pos)) if pos else np.zeros((0, 3))


def _unified_info(points: np.ndarray, img_st: np.ndarray, p_keep: float) -> dict:
    from .rasterize import detect_levels, floor_ceiling

    levels = detect_levels(points)
    if levels:
        fl, ce = levels[0]["floor_z"], levels[0]["ceiling_z"]
    else:
        fl, ce = floor_ceiling(points[:, 2])
    # у некоторых экспортов позы снимков - поза всего облака, а не станций (все в одной точке)
    spread = float(np.ptp(img_st[:, :2], axis=0).max()) if len(img_st) else 0.0
    stations_known = len(img_st) >= 2 and spread > 0.3
    out = {"setups_from_images": int(len(img_st)) if stations_known else None,
           "image_poses_spread_m": round(spread, 3),
           "levels": levels, "floor_z": fl, "ceiling_z": ce,
           "sample_fraction": round(float(min(p_keep, 1.0)), 4)}
    if fl is None:
        return out
    out["ceiling_height_m"] = round(ce - fl, 3)
    if stations_known:
        out["station_height_m"] = [round(float(z - fl), 3) for z in img_st[:, 2]]
    # шаг точек по потолку нижнего уровня: число точек / площадь покрытия (ячейки 5 см)
    c = points[np.abs(points[:, 2] - ce) < 0.03]
    if len(c) > 1000:
        cells = np.unique(np.floor(c[:, :2] / 0.05).astype(np.int64), axis=0)
        dens = len(c) / max(p_keep, 1e-9) / (len(cells) * 0.0025)
        out["ceiling_points_per_m2"] = int(dens)
        out["point_spacing_mm"] = round(1000 / math.sqrt(dens), 1)
        # шероховатость потолка: СКО по z в ячейках 10 см (шум + неровность при нормальном падении)
        key = np.floor(c[:, :2] / 0.1).astype(np.int64)
        _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        mean = np.bincount(inv, c[:, 2]) / cnt
        r = c[:, 2] - mean[inv]
        ok = cnt[inv] >= 5
        out["ceiling_roughness_mm"] = round(float(np.std(r[ok]) * 1000), 2) if ok.any() else None
    return out


def _plan_coverage(points, uinfo, img_st, out: Path) -> dict | None:
    """План по покрытию - на каждый уровень свой (input_L1.png, input_L2.png ...);
    в общий отчёт идёт нижний уровень."""
    import cv2

    from .groundtruth import RasterFrame
    from .rasterize import coverage_input

    levels = uinfo.get("levels") or ([{"floor_z": uinfo["floor_z"], "ceiling_z": uinfo["ceiling_z"]}]
                                     if uinfo.get("floor_z") is not None else [])
    if not levels:
        return None
    lo = np.percentile(points[:, :2], 0.5, axis=0) - 1.0
    hi = np.percentile(points[:, :2], 99.5, axis=0) + 1.0
    px = max(0.01, float((hi - lo).max()) / 2048)
    w, h = int(math.ceil((hi[0] - lo[0]) / px)), int(math.ceil((hi[1] - lo[1]) / px))
    frame = RasterFrame(float(lo[0]), float(lo[1]), px, w, h)
    spacing = (uinfo.get("point_spacing_mm") or 15.0) / 1000
    # в выборке точек меньше, чем в файле: шаг выборки больше в 1/sqrt(доли) раз
    spacing *= 1 / math.sqrt(max(uinfo.get("sample_fraction", 1.0), 1e-6))
    markers = img_st if uinfo.get("setups_from_images") else np.zeros((0, 3))
    plans = []
    for k, lv in enumerate(levels):
        m = (points[:, 2] > lv["floor_z"] - 0.1) & (points[:, 2] < lv["ceiling_z"] + 0.1)
        res = coverage_input(points[m], frame, lv["floor_z"], lv["ceiling_z"], spacing_m=spacing)
        name = "input.png" if len(levels) == 1 else f"input_L{k + 1}.png"
        cv2.imwrite(str(out / name), res["image"])
        lv["free_area_m2"] = round(float(res["free"].sum()) * px * px, 2)
        plans.append({"frame": frame, "res": res, "markers": markers, "floor_z": lv["floor_z"],
                      "level": k + 1})
    first = dict(plans[0])
    first["levels"] = plans
    first["info"] = {"method": "coverage", "pixel_mm": round(px * 1000, 2), "size_px": [h, w],
                     "levels": len(levels), "free_area_m2": levels[0]["free_area_m2"]}
    return first


def _write_sample(stations, samples, path: Path, voxel: float, per_scan: int, rng) -> None:
    xs, its, sid = [], [], []
    for k, (s, (_, inten)) in enumerate(zip(stations, samples)):
        p = s["points"]
        key = np.floor(p / voxel).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        if len(first) > per_scan:
            first = rng.choice(first, per_scan, replace=False)
        xs.append(p[first].astype(np.float32))
        its.append(np.zeros(len(first), np.float32) if inten is None else inten[first])
        sid.append(np.full(len(first), k, np.uint8))
    np.savez_compressed(path, xyz=np.concatenate(xs), intensity=np.concatenate(its).astype(np.float16),
                        station=np.concatenate(sid),
                        station_xyz=np.array([s["center"] for s in stations], np.float64))


def _summary(r: dict) -> str:
    f = r["file"]
    lines = [f"{f['name']}: {f['bytes'] / 1e9:.2f} ГБ, {f['points_total'] / 1e6:.1f} млн точек, "
             f"{f['bytes_per_point']} Б/точку, станций {len(r['scans'])}, "
             f"снимков {r['images2D']['count']} ({r['images2D']['bytes'] / 1e6:.0f} МБ)",
             f"чтение {r['performance']['seconds_total']} с ({r['performance']['points_per_s'] / 1e6:.2f} "
             f"млн точек/с), память {r['performance']['peak_memory_mb']} МБ"]
    for s in r["scans"]:
        lines.append(
            f"  {s['index'] + 1:>2} {s['name'] or ''}: {s['points'] / 1e6:.1f} млн, сетка {s['grid_rows_cols']}, "
            f"шаг {s['azimuth_step_deg']}°/{s['elevation_step_deg']}°, без отклика "
            f"{(1 - s['valid_fraction']) * 100:.1f} %, наклон {s['tilt_deg']}°, "
            f"пол {s['floor_rel_m']} м, потолок {s['ceiling_rel_m']} м, "
            f"перепады «между» {s['jumps']['between_fraction'] * 100:.1f} %")
    lines.append(f"поля: {r['scans'][0]['fields'] if r['scans'] else '-'}")
    return "\n".join(lines) + "\n"


def _figures(report, stations, samples, plan, out: Path) -> None:
    from .debug import AQUA, BLUE, INK, INK2, ORANGE, RED, SURFACE, YELLOW, _plt

    plt = _plt()
    # план: строка на уровень, срез | вход
    if plan:
        from .rasterize import _slice_density, density_image

        rows = plan.get("levels") or [plan]
        fig, axes = plt.subplots(len(rows), 2, figsize=(14, 6.4 * len(rows) + 0.4), squeeze=False)
        for r, pl in enumerate(rows):
            fr, res = pl["frame"], pl["res"]
            markers = pl.get("markers")
            if markers is None:
                markers = np.array([s["center"] for s in stations])
                dens = _slice_density(stations, fr)
            else:                                       # сведённое облако: срез от пола уровня
                p = stations[0]["points"]
                z = p[:, 2] - pl["floor_z"]
                dens = density_image(p[(z > 1.0) & (z < 1.6)], fr)
            prefix = f"Уровень {pl['level']}: " if len(rows) > 1 else ""
            for ax, img, vmax, title in ((axes[r, 0], 1 - dens * 0.85, 1, prefix + "срез 1,0–1,6 м"),
                                         (axes[r, 1], res["image"], 255, "Вход")):
                ax.imshow(img, cmap="gray", vmin=0, vmax=vmax, interpolation="antialiased")
                for k, c in enumerate(markers):
                    i, j = fr.xy_to_ij(c[0], c[1])
                    ax.plot(j, i, "o", ms=5, mfc="white", mec=INK, mew=1)
                    ax.annotate(str(k + 1), (j, i), xytext=(4, 4), textcoords="offset points",
                                fontsize=8, color=INK2)
                ax.set_title(title)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                for sp in ax.spines.values():
                    sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "plan.png", dpi=90)
        plt.close(fig)
    # панорамы: строка на станцию, дальность | интенсивность | без отклика
    pans = [(k, st) for k, (st, _) in enumerate(samples) if st.pano is not None]
    for page in range(0, len(pans), 8):
        part = pans[page:page + 8]
        fig, axes = plt.subplots(len(part), 3, figsize=(16, 1.9 * len(part) + 0.4), squeeze=False)
        for row, (k, st) in enumerate(part):
            P = st.pano
            n = np.maximum(P["n"], 1)
            rng_img = np.where(P["n"] > 0, P["r"] / n, np.nan)
            int_img = np.where(P["n"] > 0, P["i"] / n, np.nan)
            miss = np.where(P["all"] > 0, 1 - P["n"] / np.maximum(P["all"], 1), np.nan)
            for c, (img, cmap, title) in enumerate(((rng_img, "Blues_r", "Дальность"),
                                                    (int_img, "gray", "Интенсивность"),
                                                    (miss, "Reds", "Без отклика"))):
                ax = axes[row, c]
                vmax = np.nanpercentile(img, 99) if np.isfinite(img).any() else 1
                ax.imshow(img, cmap=cmap, aspect="auto", interpolation="antialiased", vmin=0,
                          vmax=vmax if c < 2 else 1)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                if row == 0:
                    ax.set_title(title)
            axes[row, 0].set_ylabel(str(k + 1), rotation=0, ha="right", va="center", color=INK2)
        fig.tight_layout()
        fig.savefig(out / f"panoramas_{page // 8 + 1}.png", dpi=80)
        plt.close(fig)
    # сенсор: дальности, интенсивность от дальности, шум от угла, перепады
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    colors = (BLUE, ORANGE, AQUA, YELLOW, RED, INK)
    rmax = max(float(np.percentile(np.linalg.norm(s["points"] - s["center"], axis=1), 99.5))
               for s in stations[:6] if len(s["points"]))
    step = max(0.05, rmax / 60)
    for k, s in enumerate(stations[:6]):
        r = np.linalg.norm(s["points"] - s["center"], axis=1)
        axes[0].hist(r, bins=np.arange(0, rmax + step, step), histtype="step", color=colors[k % 6],
                     label=str(k + 1))
        it = samples[k][1]
        if it is not None and len(it):
            b = np.arange(0, rmax + 2 * step, 2 * step)
            idx = np.digitize(r, b)
            med = [np.median(it[idx == j]) if (idx == j).sum() > 200 else np.nan for j in range(1, len(b))]
            axes[1].plot(b[:-1] + step, med, color=colors[k % 6], lw=1.5)
    axes[0].set_title("Дальность")
    axes[0].set_xlabel("м")
    axes[0].legend(title="станция", fontsize=8)
    axes[1].set_title("Интенсивность (медиана)")
    axes[1].set_xlabel("дальность, м")
    tabs = [st.noise_tab for st, _ in samples if hasattr(st, "noise_tab")]
    if tabs:
        tab = np.concatenate(tabs)
        for (r0, r1), col in zip(((0, 3), (3, 6), (6, 12), (12, 40)), (BLUE, ORANGE, AQUA, RED)):
            xs, ys = [], []
            for a0 in range(0, 85, 5):
                m = (tab[:, 0] >= r0) & (tab[:, 0] < r1) & (tab[:, 1] >= a0) & (tab[:, 1] < a0 + 5)
                if m.sum() > 300:
                    cos = np.cos(np.radians(tab[m, 1]))
                    xs.append(a0 + 2.5)
                    ys.append(_robust_sigma(tab[m, 2] / np.maximum(cos, 0.1)) / math.sqrt(1.25) * 1000)
            if xs:
                axes[2].plot(xs, ys, "o-", color=col, ms=3, lw=1.5, label=f"{r0}–{r1} м")
        axes[2].set_ylim(bottom=0)
        axes[2].legend(fontsize=8)
    axes[2].set_title("Шум дальности, мм")
    axes[2].set_xlabel("угол падения, °")
    pos = np.sum([s["jumps"]["between_position_hist"] for s in report["scans"]], axis=0)
    axes[3].bar(np.arange(20) / 20 + 0.025, pos, width=0.045, color=BLUE)
    axes[3].set_title("Точки на перепадах")
    axes[3].set_xlabel("0 - ближняя поверхность, 1 - дальняя")
    fig.tight_layout()
    fig.savefig(out / "sensor.png", dpi=90)
    plt.close(fig)
    del SURFACE


# ----------------------------------------------------------------------
# Сравнение двух отчётов (реальный скан против синтетики): одна мерка для обоих
# ----------------------------------------------------------------------

def aggregate(report: dict) -> dict:
    """Сводка отчёта: шум (взвешенно по станциям), перепады, скаляры."""
    S = report["scans"]
    tabs = [s["noise"] for s in S if s.get("noise")]
    out = {}
    if tabs:
        sig = np.array([[[np.nan if v is None else v for v in row] for row in t["sigma_mm"]] for t in tabs])
        cnt = np.array([t["count"] for t in tabs], float)
        w = np.where(np.isnan(sig), 0, cnt)
        tot = w.sum(0)
        out["noise_mm"] = np.where(tot > 2000, np.nansum(np.nan_to_num(sig) * w, 0) / np.maximum(tot, 1), np.nan)
        out["noise_count"] = tot
        out["range_bins_m"] = tabs[0]["range_bins_m"]
        out["incidence_bins_deg"] = tabs[0]["incidence_bins_deg"]
    hist = np.sum([s["jumps"]["between_position_hist"] for s in S], 0)
    jumps = sum(s["jumps"]["count"] for s in S)
    med = lambda k: float(np.median([s[k] for s in S if s.get(k) is not None]))  # noqa: E731
    out.update({
        "jump_hist": hist / max(hist.sum(), 1), "jumps": int(jumps),
        "between_fraction": float(hist.sum() / max(jumps, 1)),
        "step_deg": med("azimuth_step_deg"),
        "station_height_m": -med("floor_rel_m"),
        "valid_fraction": med("valid_fraction"),
        "intensity_p50": float(np.median([s["intensity"]["p50"] for s in S if s.get("intensity")])),
        "floor_flatness_mm": med("floor_flatness_mm"),
        "range_p50_m": float(np.median([s["range_m"]["p50"] for s in S])),
        "bytes_per_point": report["file"].get("bytes_per_point"),
    })
    return out


def compare_reports(paths: list, names: list, out: Path) -> str:
    """Картинка (шум от угла по дальностям, точки на перепадах) и таблица скаляров."""
    from .debug import AQUA, BLUE, INK2, ORANGE, RED, _plt

    reps = [aggregate(json.loads(Path(p).read_text(encoding="utf-8"))) for p in paths]
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    styles = ("-", "--", ":", "-.")
    colors = (BLUE, ORANGE, AQUA, RED)
    for k, (rep, name) in enumerate(zip(reps, names)):
        if "noise_mm" in rep:
            inc = np.array(rep["incidence_bins_deg"] + [85.0])
            mid = (inc[:-1] + inc[1:]) / 2
            rb = rep["range_bins_m"] + [None]
            for j, col in enumerate(colors):
                if j >= len(rep["noise_mm"]):
                    break
                y = rep["noise_mm"][j]
                if np.isfinite(y).any():
                    label = f"{rb[j]:g}–{rb[j + 1]:g} м" if k == 0 and rb[j + 1] else None
                    axes[0].plot(mid, y, styles[k % 4], color=col, marker="o", ms=3, lw=1.5, label=label)
        axes[1].plot(np.arange(20) / 20 + 0.025, rep["jump_hist"], styles[k % 4], color=INK2, lw=1.8,
                     label=f"{name}: {rep['between_fraction'] * 100:.0f} %")
    axes[0].set_title("Шум дальности, мм")
    axes[0].set_xlabel("угол падения, °")
    axes[0].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    for k, name in enumerate(names):
        handles.append(plt.Line2D([], [], ls=styles[k % 4], color=INK2))
        labels.append(name)
    axes[0].legend(handles, labels, fontsize=8)
    axes[1].set_title("Точки на перепадах (доля «между»)")
    axes[1].set_xlabel("0 - ближняя поверхность, 1 - дальняя")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    keys = [("step_deg", "шаг, °", "{:.4f}"), ("station_height_m", "высота станции, м", "{:.3f}"),
            ("valid_fraction", "с откликом", "{:.3f}"), ("intensity_p50", "интенсивность, медиана", "{:.3f}"),
            ("floor_flatness_mm", "пол, СКО мм", "{:.1f}"), ("range_p50_m", "дальность, медиана м", "{:.2f}"),
            ("between_fraction", "перепады «между»", "{:.3f}"), ("bytes_per_point", "Б/точку", "{}")]
    lines = ["".ljust(26) + "".join(n.ljust(14) for n in names)]
    for k, label, fmt in keys:
        lines.append(label.ljust(26) + "".join(fmt.format(r.get(k)).ljust(14) for r in reps))
    return "\n".join(lines)
