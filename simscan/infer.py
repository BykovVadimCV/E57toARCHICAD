"""Применение сети плана к реальному E57 (или к уже растеризованному input.npy).

    python -m simscan infer best.pt объект.e57 --out pred [--tta flip|d4] [--plan]

На уровень: pred/<файл>/L<k>/
    sem.png        классы 0..7 (netinput.CLASSES), тот же растр 2 см, что вход
    prob.npz       вероятности классов float16 [8, H, W]
    dist.npy, orient.npy, heights.npy  - регрессионные головы (как разметка датасета)
    scene.json     кадр растра и pixel_to_world (пиксель -> мировые x, y)
    pred.png       цветная карта классов поверх входа
    plan.json, plan.png, plan.svg - векторный план уровня (с --plan, simscan/vectorize.py)

Растр предсказывается окнами tile x tile с перекрытием, пачками по batch окон; вклад окна
взвешен к центру, чтобы на стыках не было швов. Усредняются логиты, не вероятности.

TTA (--tta): flip - 4 варианта (отражения), d4 - 8 (повороты на 90° x отражение). Выходы
возвращаются обратным преобразованием по тем же правилам, что аугментации обучения
(torchdata): отражение меняет знак sin 2θ, поворот на 90° - знак всего (cos 2θ, sin 2θ).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COLORS = np.array([[250, 250, 250], [222, 235, 247], [30, 30, 30], [214, 96, 77], [67, 147, 195],
                   [244, 165, 130], [180, 180, 120], [150, 90, 170]], np.uint8)


def load_model(ckpt, device=None):
    import torch

    from .net import PlanNet

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    m = PlanNet(**ck.get("net", {})).to(dev).eval()
    m.load_state_dict(ck["model"])
    return m, dev


TTA_SETS = {"none": [(0, False)],
            "flip": [(0, False), (0, True), (2, False), (2, True)],
            "d4": [(k, f) for k in range(4) for f in (False, True)]}


def _forward(a: np.ndarray, k: int, flip: bool) -> np.ndarray:
    if flip:
        a = a[..., ::-1]
    return np.rot90(a, k, axes=(-2, -1))


def _inverse(r: dict, k: int, flip: bool) -> dict:
    """Выходы сети на преобразованном входе -> в кадр исходного (см. torchdata._flip_x/_rot90)."""
    out = {key: np.rot90(v, -k, axes=(-2, -1)) for key, v in r.items()}
    if flip:
        out = {key: v[..., ::-1] for key, v in out.items()}
    out = {key: np.ascontiguousarray(v) for key, v in out.items()}
    if k % 2:
        out["orient"] = -out["orient"]
    if flip:
        out["orient"][1] = -out["orient"][1]
    return out


def predict(model, dev, x: np.ndarray, tile: int = 512, overlap: int = 128, tta=False, batch: int = 4) -> dict:
    """x [8, H, W] -> {"prob" [8,H,W], "dist" [1,H,W], "orient" [2,H,W], "heights" [2,H,W]}.

    tta: False/"none", True/"flip" (4 отражения) или "d4" (8: повороты x отражение) - логиты и
    регрессии усредняются по вариантам, возвращённым в кадр входа."""
    mode = {False: "none", True: "flip", None: "none"}.get(tta, tta)
    variants = TTA_SETS[mode]
    acc = None
    for k, flip in variants:
        r = _inverse(_predict_single(model, dev, np.ascontiguousarray(_forward(x, k, flip)), tile, overlap,
                                     return_logits=True, batch=batch), k, flip)
        acc = r if acc is None else {key: acc[key] + r[key] for key in acc}
    res = {key: v / len(variants) for key, v in acc.items()}
    lg = res.pop("logits")
    e = np.exp(lg - lg.max(0))
    res["prob"] = e / e.sum(0)
    n = np.linalg.norm(res["orient"], axis=0, keepdims=True)
    res["orient"] = res["orient"] / np.maximum(n, 1e-6)
    return res


def _predict_single(model, dev, x: np.ndarray, tile: int = 512, overlap: int = 128, return_logits: bool = False,
                    batch: int = 4) -> dict:
    """Одно положение входа: окна tile x tile с перекрытием, пачками по batch."""
    import torch

    C, H, W = x.shape
    Hp, Wp = max(H, tile), max(W, tile)
    xp = np.zeros((C, Hp, Wp), np.float32)
    xp[:, :H, :W] = x
    step = tile - overlap
    ys = list(range(0, max(Hp - tile, 0) + 1, step))
    xs = list(range(0, max(Wp - tile, 0) + 1, step))
    if ys[-1] != Hp - tile:
        ys.append(Hp - tile)
    if xs[-1] != Wp - tile:
        xs.append(Wp - tile)
    r = np.minimum(np.arange(tile) + 1, tile - np.arange(tile)).astype(np.float32)
    wt = np.clip(np.minimum.outer(r, r) / max(overlap / 2, 1), 0.05, 1.0)
    acc = {"logits": np.zeros((8, Hp, Wp), np.float32), "dist": np.zeros((1, Hp, Wp), np.float32),
           "orient": np.zeros((2, Hp, Wp), np.float32), "heights": np.zeros((2, Hp, Wp), np.float32)}
    wsum = np.zeros((Hp, Wp), np.float32)
    pos = [(y, xx) for y in ys for xx in xs]
    with torch.inference_mode():
        for b0 in range(0, len(pos), max(1, batch)):
            chunk = pos[b0:b0 + max(1, batch)]
            t = torch.from_numpy(np.stack([xp[:, y:y + tile, xx:xx + tile] for y, xx in chunk])).to(dev)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = model(t)
            o = {k: v.float().cpu().numpy() for k, v in o.items()}
            for n_, (y, xx) in enumerate(chunk):
                acc["logits"][:, y:y + tile, xx:xx + tile] += o["sem"][n_] * wt
                for k in ("dist", "orient", "heights"):
                    acc[k][:, y:y + tile, xx:xx + tile] += o[k][n_] * wt
                wsum[y:y + tile, xx:xx + tile] += wt
    res = {k: (v / wsum)[:, :H, :W] for k, v in acc.items()}
    if return_logits:
        return res
    lg = res.pop("logits")
    e = np.exp(lg - lg.max(0))
    res["prob"] = e / e.sum(0)
    n = np.linalg.norm(res["orient"], axis=0, keepdims=True)
    res["orient"] = res["orient"] / np.maximum(n, 1e-6)
    return res


def _overlay(x: np.ndarray, sem: np.ndarray, path: Path) -> None:
    import cv2

    occ = np.clip(x[0].astype(np.float32) + x[2].astype(np.float32), 0, 1)
    base = (255 - 120 * occ).astype(np.uint8)
    img = COLORS[sem].astype(np.float32)
    img = 0.75 * img + 0.25 * base[..., None]
    cv2.imwrite(str(path), img.astype(np.uint8)[..., ::-1])


def infer_input(model, dev, x: np.ndarray, out_dir, scene: dict | None = None, tile: int = 512, tta=False,
                plan: bool = False, batch: int = 4, log=print) -> Path:
    import cv2

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    r = predict(model, dev, x.astype(np.float32), tile=tile, tta=tta, batch=batch)
    sem = r["prob"].argmax(0).astype(np.uint8)
    cv2.imwrite(str(d / "sem.png"), sem)
    np.savez_compressed(d / "prob.npz", prob=r["prob"].astype(np.float16))
    for k in ("dist", "orient", "heights"):
        np.save(d / f"{k}.npy", r[k].astype(np.float16))
    if scene is not None:
        (d / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    _overlay(x, sem, d / "pred.png")
    if plan and scene is not None:
        from .vectorize import main_vectorize

        main_vectorize(d, d, log=log)
    return d


def infer(ckpt, paths, out, tile: int = 512, tta=False, plan: bool = False, batch: int = 4, log=print) -> list[str]:
    """paths: файлы E57 или каталоги уровней с input.npy (dataset real/... или scenes/...)."""
    from .netinput import affine, rasterize_e57

    model, dev = load_model(ckpt)
    done = []
    for p in map(Path, paths):
        if p.is_dir():
            for f in sorted(p.rglob("input.npy")):
                rel = f.parent.relative_to(p) if f.parent != p else Path(p.name)
                sc = f.parent / "scene.json"
                scene = json.loads(sc.read_text(encoding="utf-8")) if sc.exists() else None
                done.append(str(infer_input(model, dev, np.load(f), Path(out) / rel, scene, tile, tta, plan, batch,
                                            log)))
                log(done[-1])
            continue
        for k, r in enumerate(rasterize_e57(p, log=log), 1):
            scene = {"source": p.name, "level": f"L{k}", "floor_z": r["level"]["floor_z"],
                     "ceiling_z": r["level"]["ceiling_z"], "frame": r["frame"], "pixel_to_world": affine(r["frame"])}
            done.append(str(infer_input(model, dev, r["input"], Path(out) / p.stem / f"L{k}", scene, tile, tta, plan,
                                        batch, log)))
            log(done[-1])
    return done
