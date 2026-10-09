"""Массовая сборка датасета для сети плана: сцена генератора -> растр -> удаление скана.

    python -m simscan make-dataset --config configs/customer_like.yaml --count 3000 \
        --workers 8 --out dataset
    python -m simscan dataset pack dataset --out dataset.zip       # один архив для Google Drive

Скан E57 сцены весит ~0,5 ГБ, готовый уровень - ~5 МБ: каждый рабочий процесс генерирует
сцену во временный каталог, растеризует её в dataset/scenes/<scene>/L1 и удаляет скан.
Повторный запуск с теми же --seed/--start продолжает с места обрыва (готовые сцены
пропускаются). Несколько машин: одинаковый --seed, разные --start (непересекающиеся номера),
потом каталоги scenes/ слить и выполнить `dataset splits`.

Аугментации облака (до растеризации): прореживание и сдвиг по z - на доле сцен.
"""
from __future__ import annotations

import json
import shutil
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .config import SynthConfig


def _vid(k: int) -> str:
    return "" if k == 0 else f"_v{k}"


def _done(root: Path, scene_id: str, variants: int = 1) -> bool:
    return (root / "scenes" / (scene_id + _vid(variants - 1)) / "L1" / "scene.json").exists()


def _aug(seed: int, index: int, k: int, p_thin: float, p_shift: float) -> dict:
    """Аугментации облака для k-го растра скана: k = 0 - как есть (кроме доли сцен с
    прореживанием и сдвигом пола), k > 0 - всегда поворот на случайный угол и прореживание."""
    rng = np.random.default_rng([seed, index, 7, k])
    thin = float(rng.uniform(0.2, 0.6)) if (k > 0 or rng.random() < p_thin) else 0.0
    z_shift = float(rng.normal(0, 0.01)) if rng.random() < p_shift else 0.0
    rot = float(rng.uniform(0, 360)) if k > 0 else 0.0
    return {"thin": round(thin, 3), "z_shift": round(z_shift, 4), "rot_deg": round(rot, 2)}


def _one(args) -> dict:
    cfg, root, work, seed, index, keep_scan, p_thin, p_shift, variants = args
    from .generate import generate_scene
    from .netinput import E57Source, MemorySource, build_scene

    scene_id = f"scene_{index:05d}"
    tmp = Path(work) / scene_id
    t0 = time.time()
    try:
        shutil.rmtree(tmp, ignore_errors=True)
        # скан не пишется на диск (~0,5-1 ГБ записи и чтения на сцену): растр - из памяти,
        # теми же точками, что легли бы в scan.e57; --keep-scans - по-старому, через файл
        r = generate_scene(cfg, tmp, seed, index, write_scan=keep_scan)
        if isinstance(r, dict) and r.get("error"):
            raise RuntimeError(r["error"])
        t_gen = time.time() - t0
        src = E57Source(tmp / "scan.e57") if keep_scan else MemorySource(*r.pop("_scans"))
        dirs = []
        try:
            for k in range(variants):
                dirs += build_scene(tmp, root, splits=False, source=src, variant=_vid(k),
                                    log=lambda *a, **kw: None, **_aug(seed, index, k, p_thin, p_shift))
        finally:
            src.close()
        return {"scene": scene_id, "levels": [str(Path(d).relative_to(Path(root) / "scenes")) for d in dirs],
                "s": round(time.time() - t0, 1), "s_gen": round(t_gen, 1)}
    except Exception as exc:  # одна плохая сцена не роняет набор
        return {"scene": scene_id, "error": f"{type(exc).__name__}: {exc}", "s": round(time.time() - t0, 1)}
    finally:
        if not keep_scan:
            shutil.rmtree(tmp, ignore_errors=True)


def rebuild_splits(root) -> dict:
    """splits/*.txt заново по всем scenes/*/L*/scene.json (после слияния машин или сбоя)."""
    from .netinput import _split_of

    root = Path(root)
    sp = {"train": [], "val": [], "test": []}
    for f in sorted((root / "scenes").glob("*/L*/scene.json")):
        sc = json.loads(f.read_text(encoding="utf-8"))
        sp[_split_of(sc["layout_key"])].append(f"{f.parent.parent.name}/{f.parent.name}")
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for k, v in sp.items():
        (root / "splits" / f"{k}.txt").write_text("".join(x + "\n" for x in v), encoding="utf-8")
    return {k: len(v) for k, v in sp.items()}


def make_dataset(cfg: SynthConfig, out_root, count: int, seed: int = 0, start: int = 0, workers: int = 1,
                 work_dir=None, keep_scans: bool = False, p_thin: float = 0.3, p_shift: float = 0.5,
                 variants: int = 1, log=print) -> dict:
    """variants - растров на один скан: первый как есть, остальные с поворотом облака на
    случайный угол и прореживанием (растр стоит ~5-10 с против ~45 с симуляции скана)."""
    from .netinput import write_meta

    root = Path(out_root)
    write_meta(root)
    # лишнее для датасета сети не пишем: превью, маски datasetgen, отладку
    cfg.export.preview = cfg.export.unet_mask = cfg.export.debug = cfg.export.free_space_input = False
    cfg.export.write_mesh = False
    work = Path(work_dir) if work_dir else root / "_work"
    work.mkdir(parents=True, exist_ok=True)
    todo = [i for i in range(start, start + count) if not _done(root, f"scene_{i:05d}", variants)]
    log(f"сцен {count}, готово {count - len(todo)}, осталось {len(todo)}")
    jobs = [(cfg, str(root), str(work), seed, i, keep_scans, p_thin, p_shift, variants) for i in todo]
    ok = err = 0
    t0 = time.time()
    with open(root / "make_log.jsonl", "a", encoding="utf-8") as flog:
        def take(r):
            nonlocal ok, err
            flog.write(json.dumps(r, ensure_ascii=False) + "\n")
            flog.flush()
            if r.get("error"):
                err += 1
            else:
                ok += 1
            n = ok + err
            eta = (time.time() - t0) / n * (len(todo) - n)
            log(f"[{n}/{len(todo)}] {r['scene']} {'ОШИБКА ' + r['error'][:120] if r.get('error') else 'ок'}"
                f" ({r['s']} с; осталось ~{eta / 3600:.1f} ч)")

        if workers <= 1:
            for j in jobs:
                take(_one(j))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for fut in as_completed([pool.submit(_one, j) for j in jobs]):
                    take(fut.result())
    if not keep_scans:
        shutil.rmtree(work, ignore_errors=True)
    splits = rebuild_splits(root)
    log(f"готово: {ok}, ошибок: {err}; splits {splits}")
    return {"ok": ok, "errors": err, "splits": splits}


def pack(root, out_zip, include_previews: bool = False) -> Path:
    """Датасет -> один zip (deflate; входы и разметка в основном нули и сжимаются в разы)."""
    root, out_zip = Path(root), Path(out_zip)
    skip = {"_work"}
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if f.is_dir() or rel.parts[0] in skip or (f.name == "preview.png" and not include_previews):
                continue
            z.write(f, Path(root.name) / rel)
    return out_zip
