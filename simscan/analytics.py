"""Аналитика одного E57 -> один zip: всё, чтобы понять, что конвертер сделал с файлом и почему.
Архив и присылать разработчику вместо самого E57 (десятки МБ вместо гигабайт).

    python -m simscan analytics объект.e57 [--model best.pt | --pred <каталог infer этого файла>]
                                [--out D:/an] [--tta none|flip|d4] [--no-cellplan] [--panoramas]
    -> D:/an/<имя>_analytics.zip  (рядом - та же папка распакованной)

Каждый раздел - отдельный шаг: упавший шаг не роняет остальные (трассировка - в errors.txt и
run.json), архив собирается из того, что получилось. Содержимое:

  SUMMARY.md          главное одним листом: числа по шагам, предупреждения, ошибки, оглавление
  run.json, log.txt   версии (python, numpy, opencv, torch), коммит, аргументы, модель (sha256),
                      время шагов, предупреждения, ошибки; весь журнал
  file/               inspect_e57: заголовок, станции (сетка, шаг, наклон, высота, пол/потолок),
                      шум сенсора, перепады, панорамы дальности, срез и вход, sample.npz (облако 4 см)
  levels/             гистограмма высот точек с найденными уровнями, levels.json
  images/             cameras.json: параметры и позы снимков, стоянки, соглашение граней куба;
                      с --panoramas - ещё панорамы стоянок 2048 px (без него фото не попадают)
  L<k>/input.*        вход сети: 8 каналов (input.npz float16, листом input.png), покрытие каналов
  L<k>/net/           prob.npz, sem.png, pred.png, classes.png (вероятности по классам),
                      uncertainty.png (энтропия), net.json (площади классов, уверенность)
  L<k>/vectorize/     plan.json/png/svg; overlay.png - план поверх классов сети;
                      candidates.png/json - все кандидаты в помещения со счётом замкнутости, принят
                      или почему нет; stages.json - стен после каждого шага; walls.csv (длина,
                      толщина, помещения, опора на сеть), openings.csv (тип, ширина, вероятность)
  L<k>/snap/          план после привязки к облаку; walls.csv - по каждой стене: найдено граней,
                      сдвиг, толщина сети и облака, СКО грани
  cellplan/           второй построитель (облако + сеть): планы, сверка с картой стен, elements.json

Не попадают: цвет точек, фотографии (без --panoramas), полное облако, файлы больше
--max-file-mb (список - в SUMMARY.md).
"""
from __future__ import annotations

import contextlib
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np

CLASS_NAMES = ["outside", "interior", "wall", "door", "window", "opening", "clutter", "void"]
LOW_CONF = 0.6                       # уверенность сети ниже - «неуверенный» пиксель
NEAR_MISS = 0.15                     # отклонённый кандидат в пределах порог - 0,15: предупреждение
WEAK_WALL = 0.3                      # стена, вдоль оси которой сеть видит конструкцию слабее, - слабая
PANO_W = 2048


class Run:
    """Журнал, время шагов, ошибки и предупреждения одного прогона."""

    def __init__(self, log):
        self._log = log
        self.lines, self.errors, self.warnings = [], [], []
        self.timings: dict[str, float] = {}
        self.facts: dict = {}

    def log(self, *a) -> None:
        msg = " ".join(str(x) for x in a)
        self.lines.append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        self._log(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.log(f"ВНИМАНИЕ: {msg}")

    @contextlib.contextmanager
    def step(self, name: str):
        t = time.perf_counter()
        self.log(f"== {name}")
        try:
            yield
        except Exception as e:                           # noqa: BLE001 - шаг упал, остальные идут
            self.errors.append({"step": name, "error": f"{type(e).__name__}: {e}",
                                "traceback": traceback.format_exc()})
            self.log(f"!! {name}: {type(e).__name__}: {e}")
        finally:
            self.timings[name] = round(time.perf_counter() - t, 1)


def _js(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_js), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)


def _to_px(M: np.ndarray, xy) -> np.ndarray:
    """Мировые x, y -> пиксели сети (j, i) по pixel_to_world."""
    Mi = np.linalg.inv(np.vstack([np.asarray(M, float), [0, 0, 1]]))[:2]
    xy = np.atleast_2d(np.asarray(xy, float))
    return (Mi @ np.c_[xy, np.ones(len(xy))].T).T


def _along(arr: np.ndarray, pts_px: np.ndarray, step: float = 1.0) -> np.ndarray:
    """Значения растра вдоль ломаной (пиксели), через ~step px."""
    out = []
    for a, b in zip(pts_px[:-1], pts_px[1:]):
        n = max(2, int(np.linalg.norm(b - a) / step))
        q = np.round(a + (b - a) * np.linspace(0, 1, n)[:, None]).astype(int)
        q[:, 0] = np.clip(q[:, 0], 0, arr.shape[1] - 1)
        q[:, 1] = np.clip(q[:, 1], 0, arr.shape[0] - 1)
        out.append(arr[q[:, 1], q[:, 0]])
    return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------------
# Окружение
# ----------------------------------------------------------------------

def environment(model=None) -> dict:
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    for mod in ("cv2", "torch", "pye57", "matplotlib", "ifcopenshell"):
        try:
            v = getattr(__import__(mod), "__version__", None)
            env[mod] = v if isinstance(v, str) or v is None else str(getattr(v, "__version__", v))
        except Exception:                                # noqa: BLE001
            env[mod] = None
    try:
        import torch

        env["cuda"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:                                    # noqa: BLE001
        pass
    repo = Path(__file__).resolve().parents[1]
    try:
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,  # noqa: E731
                                        timeout=10, check=False).stdout.strip()
        env["commit"] = git("rev-parse", "HEAD") or None
        env["branch"] = git("rev-parse", "--abbrev-ref", "HEAD") or None
        env["dirty"] = bool(git("status", "--porcelain", "--untracked-files=no"))
    except Exception:                                    # noqa: BLE001
        env["commit"] = None
    if model and not Path(model).is_file():
        env["model"] = {"name": Path(model).name, "missing": True}
    elif model:
        p = Path(model)
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        env["model"] = {"name": p.name, "bytes": p.stat().st_size, "sha256": h.hexdigest()}
    return env


# ----------------------------------------------------------------------
# Разделы
# ----------------------------------------------------------------------

def file_section(run: Run, e57: Path, d: Path) -> dict:
    """inspect_e57 без архива; выборка облака - не больше ~3 млн точек на файл."""
    from .e57read import E57Reader
    from .inspect_e57 import inspect_e57

    with E57Reader(e57) as r:
        n_scans = max(1, len(r.scans))
    per_scan = max(20_000, min(200_000, 3_000_000 // n_scans))
    rep = inspect_e57(e57, d, sample_per_scan=per_scan, zip_result=False, log=run.log)
    f = rep["file"]
    run.facts["file"] = {"name": f["name"], "gb": round(f["bytes"] / 1e9, 3), "points_m": round(f["points_total"] / 1e6, 2),
                         "scans": len(rep["scans"]), "unified": "unified" in rep,
                         "images": rep["images2D"]["count"], "has_color": any(s["has_color"] for s in rep["scans"]),
                         "read_s": rep["performance"]["seconds_total"]}
    if "unified" in rep:
        u = rep["unified"]
        run.facts["file"].update(setups_from_images=u.get("setups_from_images"),
                                 point_spacing_mm=u.get("point_spacing_mm"))
    tilts = [s["tilt_deg"] for s in rep["scans"] if s.get("tilt_deg") is not None]
    if tilts and max(tilts) > 1.0:
        run.warn(f"станция наклонена на {max(tilts):.1f}° - горизонтальные срезы смазаны")
    return rep


def levels_section(run: Run, sample: Path, levels: list[dict], d: Path) -> None:
    """Гистограмма высот (выборка облака из file/, воксели 4 см) с уровнями растеризатора."""
    from .debug import BLUE, INK2, ORANGE, _plt

    d.mkdir(parents=True, exist_ok=True)
    doc = {"levels": levels}
    if sample.exists():
        z = np.load(sample)["xyz"][:, 2].astype(float)
        lo, hi = np.percentile(z, 0.2) - 0.3, np.percentile(z, 99.8) + 0.3
        cnt, edges = np.histogram(z, bins=np.arange(lo, hi + 0.04, 0.04))      # выборка - вокселями 4 см
        doc["z_hist"] = {"bin_m": 0.04, "from_m": round(float(lo), 3), "count": cnt.tolist()}
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 7))
        ax.barh(edges[:-1] + 0.02, cnt, height=0.04, color=BLUE)
        ax.set_xscale("log")
        for k, lv in enumerate(levels, 1):
            for key, col in (("floor_z", ORANGE), ("ceiling_z", INK2)):
                if lv.get(key) is not None:
                    ax.axhline(lv[key], color=col, lw=1.2, ls="-" if key == "floor_z" else "--")
                    ax.annotate(f"L{k} {'пол' if key == 'floor_z' else 'потолок'} {lv[key]:.2f}",
                                (1, lv[key]), xycoords=("axes fraction", "data"), ha="right", va="bottom",
                                fontsize=8, color=col)
        ax.set_title("Точек выборки на 4 см высоты")
        ax.set_ylabel("z, м")
        fig.tight_layout()
        fig.savefig(d / "z_hist.png", dpi=90)
        plt.close(fig)
    _write_json(d / "levels.json", doc)
    run.facts["levels"] = [{"floor_z": round(lv["floor_z"], 3), "ceiling_z": round(lv["ceiling_z"], 3),
                            "height_m": round(lv["ceiling_z"] - lv["floor_z"], 3)} for lv in levels]
    for k, lv in enumerate(levels, 1):
        h = lv["ceiling_z"] - lv["floor_z"]
        if not 2.2 <= h <= 4.5:
            run.warn(f"L{k}: высота уровня {h:.2f} м - вне 2,2-4,5 м, пол или потолок найдены неверно")


def images_section(run: Run, e57: Path, d: Path, panoramas: bool) -> None:
    """Параметры и позы снимков; панорамы стоянок - только по флагу (это фото объекта)."""
    from .e57images import extract, read_images, seam_check, stations
    from .e57read import E57Reader

    d.mkdir(parents=True, exist_ok=True)
    if panoramas:
        import cv2

        tmp = d / "_extract"
        doc = extract(e57, tmp, panoramas=True, pano_width=PANO_W, log=run.log)
        for s in doc["stations"]:
            if s.get("panorama"):
                (d / "panoramas").mkdir(exist_ok=True)
                im = cv2.imread(str(tmp / s["panorama"]))
                cv2.imwrite(str(d / s["panorama"]), im, [cv2.IMWRITE_JPEG_QUALITY, 85])
        shutil.rmtree(tmp, ignore_errors=True)
        for c in doc["cameras"]:
            c.pop("file", None)
    else:
        with E57Reader(e57) as r:
            ims = read_images(r)
            groups = stations(ims)
            conv = seam_check(r, ims, groups, log=run.log) if ims else {}
            cams = [{"index": im.index, "name": im.name, "guid": im.guid, "scan_guid": im.scan_guid,
                     "representation": im.representation, **im.camera.to_json()}
                    for im in ims if im.camera is not None]
            st = [{"station": s + 1, "center": np.round(ims[g[0]].camera.center, 4).tolist(), "images": g}
                  for s, g in enumerate(groups) if ims[g[0]].camera is not None]
        doc = {"file": e57.name, "pinhole_convention": conv, "stations": st, "cameras": cams}
    _write_json(d / "cameras.json", doc)
    kinds = {}
    for c in doc["cameras"]:
        kinds[c["representation"]] = kinds.get(c["representation"], 0) + 1
    conv = doc.get("pinhole_convention") or {}
    run.facts["images"] = {"cameras": len(doc["cameras"]), "kinds": kinds, "stations": len(doc["stations"]),
                           "pinhole_convention": conv.get("conv"), "convention_sure": conv.get("sure")}
    if kinds.get("pinhole") and conv and not conv.get("sure"):
        run.warn("соглашение осей граней куба не определилось по швам - панорамы могут быть перевёрнуты")


def input_section(run: Run, r: dict, d: Path) -> dict:
    """Вход сети уровня: каналы листом, доли покрытия."""
    from .debug import _plt
    from .netinput import CHANNELS

    d.mkdir(parents=True, exist_ok=True)
    x = r["input"]
    np.savez_compressed(d / "input.npz", input=x.astype(np.float16))
    fr = r["frame"]
    xf = x.astype(np.float32)
    stats = {"frame": fr, "level": r["level"], "channels": []}
    for c, ch in enumerate(CHANNELS):
        nz = xf[c] > 0
        stats["channels"].append({"name": ch["name"], "nonzero_share": round(float(nz.mean()), 4),
                                  "mean_nonzero": round(float(xf[c][nz].mean()), 4) if nz.any() else None})
    covered = (xf[[0, 1, 2, 3, 4, 5]].max(0) > 0)
    stats["covered_m2"] = round(float(covered.sum()) * float(fr["pixel_m"]) ** 2, 1)
    stats["floor_seen_of_covered"] = round(float((xf[3] > 0)[covered].mean()), 4) if covered.any() else None
    _write_json(d / "input.json", stats)
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4 * x.shape[1] / max(x.shape[2], 1) + 1))
    for c, ax in enumerate(axes.ravel()):
        ax.imshow(xf[c], cmap="gray_r", vmin=0, vmax=1, interpolation="antialiased")
        ax.set_title(CHANNELS[c]["name"])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(d / "input.png", dpi=80)
    plt.close(fig)
    return stats


def net_section(run: Run, lvl: str, d: Path) -> dict:
    """Статистика выхода сети уровня (d - каталог infer уровня)."""
    import cv2

    from .debug import _plt

    prob = np.load(d / "prob.npz")["prob"].astype(np.float32)
    scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    px = abs(float(np.asarray(scene["pixel_to_world"])[0][0]))
    sem = prob.argmax(0)
    pmax = prob.max(0)
    ent = -(prob * np.log(np.clip(prob, 1e-6, 1))).sum(0) / math.log(len(prob))
    seen = sem != 0
    st = {"level": lvl, "pixel_m": px, "size_px": list(sem.shape), "classes": {}}
    for k, name in enumerate(CLASS_NAMES):
        m = sem == k
        st["classes"][name] = {"area_m2": round(float(m.sum()) * px * px, 2),
                               "mean_conf": round(float(pmax[m].mean()), 3) if m.any() else None}
    st["low_conf_share"] = round(float(((pmax < LOW_CONF) & seen).sum() / max(seen.sum(), 1)), 4)
    st["conf_hist"] = np.histogram(pmax[seen], bins=10, range=(0, 1))[0].tolist() if seen.any() else []
    st["entropy_mean"] = round(float(ent[seen].mean()), 4) if seen.any() else None
    if (d / "heights.npy").exists():
        h = np.load(d / "heights.npy").astype(np.float32)
        for k in (3, 4):
            m = sem == k
            if m.any() and h.ndim == 3:
                st[f"{CLASS_NAMES[k]}_heights_median"] = [round(float(np.median(h[c][m])), 3) for c in range(len(h))]
    _write_json(d / "net.json", st)
    col = (np.clip(ent, 0, 1) * 255).astype(np.uint8)
    cv2.imwrite(str(d / "uncertainty.png"), cv2.applyColorMap(col, cv2.COLORMAP_INFERNO))
    plt = _plt()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.4 * sem.shape[0] / max(sem.shape[1], 1) + 1))
    for k, ax in enumerate(axes.ravel()):
        ax.imshow(prob[k], cmap="magma", vmin=0, vmax=1, interpolation="antialiased")
        ax.set_title(CLASS_NAMES[k])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(d / "classes.png", dpi=80)
    plt.close(fig)
    run.facts.setdefault("net", {})[lvl] = {
        "interior_m2": st["classes"]["interior"]["area_m2"], "wall_m2": st["classes"]["wall"]["area_m2"],
        "low_conf_share": st["low_conf_share"]}
    if st["low_conf_share"] > 0.2:
        run.warn(f"{lvl}: уверенность сети ниже {LOW_CONF:.0%} в {st['low_conf_share']:.0%} пикселей помещения")
    return st


def _overlay(prob: np.ndarray, plan: dict, path: Path, cands: list | None = None, scale: int = 2) -> None:
    """Классы сети (как infer.COLORS, светлее) + план: стены, проёмы, помещения; отклонённые
    кандидаты в помещения - красным номером."""
    import cv2

    from .infer import COLORS

    H, W = prob.shape[1:]
    S = scale
    base = (0.55 * COLORS[prob.argmax(0)] + 0.45 * 255).astype(np.uint8)
    img = cv2.resize(base, (W * S, H * S), interpolation=cv2.INTER_NEAREST)[..., ::-1].copy()
    M = plan["pixel_to_world"]
    P = lambda xy: np.round((_to_px(M, xy) + 0.5) * S - 0.5).astype(np.int32)   # noqa: E731
    for r in plan["rooms"]:
        cv2.polylines(img, [P(r["points"])], True, (40, 150, 40), 2)
        c = P([r["label_xy"]])[0] if r.get("label_xy") else P(r["points"]).mean(0).astype(int)
        cv2.putText(img, r["id"].split("-")[-1], tuple(int(v) for v in c), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (30, 110, 30), 2)
    for w in plan["walls"]:
        q = P(w["points"])
        cv2.polylines(img, [q], False, (0, 0, 0), 2)
        for e in (q[0], q[-1]):
            cv2.circle(img, tuple(int(v) for v in e), 3, (0, 0, 220), -1)
    for o in plan["openings"]:
        a, b = P([o["a"], o["b"]])
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b),
                 (0, 0, 200) if o["opening_type"] == "door" else (200, 80, 0), 4)
    for c in cands or []:
        if not c["ok"] and c["area_m2"] >= 1.0:
            x, y = (np.asarray(c["centre"]) + 0.5) * S - 0.5
            cv2.putText(img, f"x{c['score']:.2f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(str(path), img)


def _candidates_png(lab: np.ndarray, cands: list, path: Path, scale: int = 2) -> None:
    import cv2

    H, W = lab.shape
    img = np.full((H, W, 3), 255, np.uint8)
    rng = np.random.default_rng(1)
    for c in cands:
        m = lab == c["label"]
        col = rng.integers(120, 235, 3) if c["ok"] else np.array([200, 200, 235])
        img[m] = col
    img = cv2.resize(img, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    for c in cands:
        if c["area_m2"] < 0.5:
            continue
        x, y = (np.asarray(c["centre"]) + 0.5) * scale - 0.5
        txt = f"{c['label']}: {c['score']:.2f} {'OK' if c['ok'] else 'X'}"
        cv2.putText(img, txt, (int(x) - 30, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20) if c["ok"] else (0, 0, 190), 1)
    cv2.imwrite(str(path), img)


def vectorize_section(run: Run, lvl: str, net_dir: Path, d: Path) -> dict:
    """Векторизатор с трассой: план, кандидаты в помещения, шаги, стены и проёмы с опорой на сеть."""
    from . import vectorize as V

    d.mkdir(parents=True, exist_ok=True)
    prob, orient, heights, scene = V._load(net_dir)
    trace: dict = {}
    plan = V.vectorize_arrays(prob, orient, heights, scene, log=run.log, trace=trace)
    V.save_json(plan, d / "plan.json")
    V.save_png(plan, d / "plan.png")
    V.save_svg(plan, d / "plan.svg")
    M = np.asarray(plan["pixel_to_world"], float)
    cands = trace.get("candidates", [])
    for c in cands:
        c["centre_xy"] = np.round(M @ [*c["centre"], 1.0], 3).tolist()
        c["verdict"] = ("принят" if c["ok"] else
                        f"мал (< {V.MIN_ROOM_M2} м²)" if c["area_m2"] < V.MIN_ROOM_M2 else
                        f"не замкнут: стены {c['walls']:.2f} - улица {c['street']:.2f} < {V.MIN_ENCLOSED}")
    _write_json(d / "candidates.json", {"min_enclosed": V.MIN_ENCLOSED, "min_room_m2": V.MIN_ROOM_M2,
                                        "candidates": cands})
    if "candidate_labels" in trace:
        _candidates_png(trace["candidate_labels"], cands, d / "candidates.png")
    _write_json(d / "stages.json", {"stages": trace.get("stages", []), "rooms": len(plan["rooms"]),
                                    "openings": len(plan["openings"])})
    _overlay(prob, plan, d / "overlay.png", cands)
    p_struct = prob[list(V.STRUCT)].sum(0)
    walls = []
    for w in plan["walls"]:
        v = _along(p_struct, _to_px(M, w["points"]))
        walls.append({"id": w["id"], "length_m": w["length"], "thickness_cm": round(w["thickness"] * 100, 1),
                      "rooms": len(w["rooms"]), "is_outer": w["is_outer"], "arc": "arc" in w,
                      "net_support": round(float(v.mean()), 3) if len(v) else None,
                      "net_wall_share": round(float((v > 0.5).mean()), 3) if len(v) else None})
    _write_csv(d / "walls.csv", walls)
    ops = []
    for o in plan["openings"]:
        k = CLASS_NAMES.index(o["opening_type"]) if o["opening_type"] in CLASS_NAMES else None
        v = _along(prob[k], _to_px(M, [o["a"], o["b"]])) if k is not None else np.zeros(0)
        ops.append({"id": o["id"], "type": o["opening_type"], "wall_id": o["wall_id"], "width_m": o["width"],
                    "sill_m": o.get("sill"), "head_m": o.get("head"),
                    "net_prob": round(float(v.mean()), 3) if len(v) else None})
    _write_csv(d / "openings.csv", ops)
    weak = [w["id"] for w in walls if w["net_support"] is not None and w["net_support"] < WEAK_WALL]
    near = [c for c in cands if not c["ok"] and c["area_m2"] >= 4.0
            and c["score"] >= V.MIN_ENCLOSED - NEAR_MISS]
    run.facts.setdefault("vectorize", {})[lvl] = {
        "rooms": len(plan["rooms"]), "rooms_m2": round(sum(r["area"] for r in plan["rooms"]), 1),
        "walls": len(plan["walls"]), "doors": sum(o["opening_type"] == "door" for o in plan["openings"]),
        "windows": sum(o["opening_type"] == "window" for o in plan["openings"]),
        "openings": sum(o["opening_type"] == "opening" for o in plan["openings"]),
        "candidates_rejected": sum(not c["ok"] for c in cands if c["area_m2"] >= V.MIN_ROOM_M2),
        "weak_walls": len(weak)}
    if not plan["rooms"]:
        run.warn(f"{lvl}: векторизатор не нашёл ни одного помещения")
    for c in near:
        run.warn(f"{lvl}: кандидат {c['label']} ({c['area_m2']} м², счёт {c['score']}) отклонён у самого порога "
                 f"{V.MIN_ENCLOSED} - смотри vectorize/candidates.png")
    if weak:
        run.warn(f"{lvl}: стен без опоры на сеть (< {WEAK_WALL}): {len(weak)} - {', '.join(weak[:8])}")
    return plan


def snap_section(run: Run, e57: Path, plans: dict, root: Path) -> None:
    """Привязка всех планов к облаку (один проход по файлу) и разбор по стенам."""
    from .cloudsnap import snap_e57
    from .vectorize import save_json, save_png, save_svg

    names = list(plans)
    for lvl, snapped in zip(names, snap_e57([plans[n] for n in names], e57, log=run.log)):
        d = root / lvl / "snap"
        d.mkdir(parents=True, exist_ok=True)
        rep = snapped["metadata"]["snap"]
        save_json(snapped, d / "plan.json")
        save_png(snapped, d / "plan.png")
        save_svg(snapped, d / "plan.svg")
        _write_csv(d / "walls.csv", rep.get("per_wall", []))
        straight = [w for w in rep.get("per_wall", []) if w["status"] != "not_straight"]
        done = sum(w["status"] == "snapped" for w in straight)
        run.facts.setdefault("snap", {})[lvl] = {"walls": len(straight), "snapped": done,
                                                 "two_faces": rep["two_faces"], "one_face": rep["one_face"],
                                                 "move_cm_median": rep["move_cm_median"],
                                                 "move_cm_max": rep["move_cm_max"], "points": rep["points"]}
        if straight and done < 0.5 * len(straight):
            run.warn(f"{lvl}: к облаку привязано {done} из {len(straight)} прямых стен - план и облако "
                     "расходятся (уровень, кадр) или стены заставлены")


def cellplan_section(run: Run, e57: Path, pred: Path | None, d: Path) -> None:
    from .cellplan import cellplan

    res = cellplan(e57, d, pred=pred, log=run.log)
    out = {}
    for k, lv in enumerate(res["levels"], 1):
        chk = lv.get("check", {})
        out[f"L{k}"] = {"rooms": len(lv.get("rooms", [])), "rooms_m2": round(sum(r["area"] for r in lv.get("rooms", [])), 1),
                        "walls": len(lv.get("walls", [])), "openings": len(lv.get("openings", [])),
                        "wall_recall": chk.get("wall_recall"), "wall_precision": chk.get("wall_precision")}
        if chk.get("wall_recall") is not None and chk["wall_recall"] < 0.5:
            run.warn(f"cellplan L{k}: найдено {chk['wall_recall']:.0%} стен карты - план неполный")
    run.facts["cellplan"] = out


# ----------------------------------------------------------------------
# Сводка и архив
# ----------------------------------------------------------------------

def _table(rows: list[dict]) -> list[str]:
    if not rows:
        return []
    keys = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k)) for k in keys) + " |")
    return out


def summary_md(run: Run, args: dict, skipped: list) -> str:
    F = run.facts
    L = [f"# Аналитика {F.get('file', {}).get('name', '')}", ""]
    if F.get("file"):
        L += ["## Файл", "", *_table([F["file"]]), ""]
    if F.get("levels"):
        L += ["## Уровни", "", *_table([{"level": f"L{k}", **v} for k, v in enumerate(F["levels"], 1)]), ""]
    if F.get("images"):
        L += ["## Снимки", "", *_table([{**F["images"], "kinds": json.dumps(F["images"]["kinds"])}]), ""]
    for key, title in (("input", "Вход сети"), ("net", "Сеть"), ("vectorize", "Векторизатор (vectorize)"),
                       ("snap", "Привязка к облаку"), ("cellplan", "cellplan")):
        if F.get(key):
            L += [f"## {title}", "", *_table([{"level": k, **v} for k, v in F[key].items()]), ""]
    L += ["## Предупреждения", ""] + ([f"- {w}" for w in run.warnings] or ["- нет"]) + [""]
    L += ["## Ошибки шагов", ""] + ([f"- **{e['step']}**: {e['error']}" for e in run.errors] or ["- нет"])
    L += ["", "## Время шагов, с", "", *_table([run.timings]), ""]
    L += ["## Запуск", "", "```", json.dumps(args, ensure_ascii=False, default=_js), "```", ""]
    if skipped:
        L += ["## Не вошло в архив (больше --max-file-mb)", ""] + [f"- {p} ({mb:.1f} МБ)" for p, mb in skipped] + [""]
    L += ["## Где что", "",
          "- `file/` - заголовок, станции, сенсор (summary.txt, report.json, plan.png, sensor.png, panoramas_*.png)",
          "- `levels/` - гистограмма высот и уровни",
          "- `images/cameras.json` - снимки и стоянки",
          "- `L<k>/input.png` - вход сети, `L<k>/net/` - выход сети (pred.png, classes.png, uncertainty.png)",
          ("- `L<k>/vectorize/` - overlay.png (план на сети), candidates.png (кандидаты в помещения со счётом), "
           "stages.json, walls.csv, openings.csv"),
          "- `L<k>/snap/` - план после привязки к облаку, walls.csv по стенам",
          "- `cellplan/` - второй построитель", "- `errors.txt`, `log.txt`, `run.json`", ""]
    return "\n".join(L)


def _zip(root: Path, max_mb: float) -> tuple[Path, list]:
    dst = root.with_suffix(".zip")
    skipped = []
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(root.rglob("*")):
            if not f.is_file():
                continue
            mb = f.stat().st_size / 1e6
            if mb > max_mb:
                skipped.append((str(f.relative_to(root)), mb))
                continue
            z.write(f, Path(root.name) / f.relative_to(root))
    return dst, skipped


def _pred_levels(pred: Path) -> list[Path]:
    return sorted({sc.parent for sc in Path(pred).rglob("scene.json") if (sc.parent / "prob.npz").exists()})


def analytics(e57, out=None, model=None, pred=None, tta: str = "none", cellplan: bool = True,
              panoramas: bool = False, max_file_mb: float = 40.0, log=print) -> Path:
    """E57 -> <out>/<имя>_analytics/ и <out>/<имя>_analytics.zip; возвращает путь к zip.
    model - чекпойнт сети (прогнать её); pred - готовый каталог infer этого файла; без обоих
    разделы сети, векторизатора и привязки пропускаются."""
    e57 = Path(e57)
    root = Path(out or e57.parent) / f"{e57.stem}_analytics"
    if root.exists():
        if not (root / "run.json").exists() and any(root.iterdir()):
            raise FileExistsError(f"{root} не пуст и не похож на прошлую аналитику - не трогаю")
        shutil.rmtree(root)
    root.mkdir(parents=True)
    run = Run(log)
    args = {"e57": str(e57), "model": str(model) if model else None, "pred": str(pred) if pred else None,
            "tta": tta, "cellplan": cellplan, "panoramas": panoramas, "max_file_mb": max_file_mb}
    t0 = time.perf_counter()
    env = None
    with run.step("environment"):
        env = environment(model)
    with run.step("file"):
        file_section(run, e57, root / "file")
    with run.step("images"):
        images_section(run, e57, root / "images", panoramas)
    rasters = []
    with run.step("input"):
        from .netinput import affine, rasterize_e57

        rasters = rasterize_e57(e57, log=run.log)
        for k, r in enumerate(rasters, 1):
            r["scene"] = {"source": e57.name, "level": f"L{k}", "floor_z": r["level"]["floor_z"],
                          "ceiling_z": r["level"]["ceiling_z"], "frame": r["frame"],
                          "pixel_to_world": affine(r["frame"])}
            st = input_section(run, r, root / f"L{k}")
            run.facts.setdefault("input", {})[f"L{k}"] = {
                "size_px": f"{r['frame']['width']}x{r['frame']['height']}",
                **{c["name"]: c["nonzero_share"] for c in st["channels"] if c["name"] in ("floor_vis", "ceil_vis",
                                                                                            "occ_mid")}}
    with run.step("levels"):
        levels_section(run, root / "file" / "sample.npz", [r["level"] for r in rasters], root / "levels")
    net_dirs: dict[str, Path] = {}
    if model:
        with run.step("net"):
            from .infer import infer_input, load_model

            net, dev = load_model(model)
            for k, r in enumerate(rasters, 1):
                d = infer_input(net, dev, r["input"], root / f"L{k}" / "net", r["scene"], tta=tta, log=run.log)
                net_dirs[f"L{k}"] = d
    elif pred:
        with run.step("net"):
            for k, src in enumerate(_pred_levels(Path(pred)), 1):
                sc = json.loads((src / "scene.json").read_text(encoding="utf-8"))
                lvl = str(sc.get("level") or f"L{k}")
                d = root / lvl / "net"
                d.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    if f.is_file() and f.suffix in (".npz", ".npy", ".json", ".png") and not f.name.startswith("plan"):
                        shutil.copy2(f, d / f.name)
                net_dirs[lvl] = d
            run.log(f"предсказания из {pred}: уровней {len(net_dirs)}")
    plans = {}
    for lvl, d in net_dirs.items():
        with run.step(f"{lvl}: net"):
            net_section(run, lvl, d)
        with run.step(f"{lvl}: vectorize"):
            plans[lvl] = vectorize_section(run, lvl, d, root / lvl / "vectorize")
    if plans:
        with run.step("snap"):
            snap_section(run, e57, plans, root)
    if cellplan:
        with run.step("cellplan"):
            cellplan_section(run, e57, root if net_dirs else None, root / "cellplan")
    run.timings["total"] = round(time.perf_counter() - t0, 1)
    doc = {"args": args, "environment": env, "timings": run.timings,
           "facts": run.facts, "warnings": run.warnings, "errors": run.errors}
    _write_json(root / "run.json", doc)
    (root / "log.txt").write_text("\n".join(run.lines) + "\n", encoding="utf-8")
    (root / "errors.txt").write_text("\n\n".join(f"[{e['step']}]\n{e['traceback']}" for e in run.errors) or "нет\n",
                                     encoding="utf-8")
    (root / "SUMMARY.md").write_text(summary_md(run, args, []), encoding="utf-8")
    dst, skipped = _zip(root, max_file_mb)
    if skipped:                                          # список пропущенного - в сводку, архив заново
        (root / "SUMMARY.md").write_text(summary_md(run, args, skipped), encoding="utf-8")
        dst, skipped = _zip(root, max_file_mb)
    log(f"архив: {dst} ({dst.stat().st_size / 1e6:.1f} МБ); предупреждений {len(run.warnings)}, "
        f"ошибок шагов {len(run.errors)}")
    return dst
