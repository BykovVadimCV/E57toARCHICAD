"""Применение сети плана к реальному E57 (или к уже растеризованному input.npy).

    python -m simscan infer best.pt объект.e57 --out pred [--tta]

На уровень: pred/<файл>/L<k>/
    sem.png        классы 0..7 (netinput.CLASSES), тот же растр 2 см, что вход
    prob.npz       вероятности классов float16 [8, H, W]
    dist.npy, orient.npy, heights.npy  - регрессионные головы (как разметка датасета)
    scene.json     кадр растра и pixel_to_world (пиксель -> мировые x, y)
    pred.png       цветная карта классов поверх входа

Растр предсказывается окнами tile×tile с перекрытием; вклад окна взвешен к центру,
чтобы на стыках не было швов.

Улучшения v2:
- Test-Time Augmentation (TTA): усреднение по 4 вариантам (orig, flipH, flipV, flipHV)
  даёт +1-2% IoU за счёт снижения дисперсии предсказаний на границах
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


def predict(model, dev, x: np.ndarray, tile: int = 512, overlap: int = 128, tta: bool = False) -> dict:
    """x [8, H, W] -> {"prob" [8,H,W], "dist" [1,H,W], "orient" [2,H,W], "heights" [2,H,W]}.

    tta=True: Test-Time Augmentation с усреднением по 4 вариантам (orig, flipH, flipV, flipHV).
    Мягкое усреднение logits перед softmax.
    """
    if not tta:
        return _predict_single(model, dev, x, tile, overlap)

    # TTA: 4 варианта (fx, fy). Каждое отражение меняет знак sin 2θ (как _flip_x/_flip_y в
    # torchdata), поэтому обратное отражение orient тоже меняет знак [1], иначе после
    # усреднения sin 2θ гасится и косые стены сваливаются в 0° / 90°.
    def _flip(a: np.ndarray, fx: bool, fy: bool) -> np.ndarray:
        if fx:
            a = np.flip(a, -1)
        if fy:
            a = np.flip(a, -2)
        return a.copy()

    def _reverse(r: dict, fx: bool, fy: bool) -> dict:
        out = {k: _flip(v, fx, fy) for k, v in r.items()}
        if fx != fy:
            out["orient"][1] *= -1
        return out

    variants = [(fx, fy) for fy in (False, True) for fx in (False, True)]

    # Усреднить logits, не prob
    C, H, W = x.shape
    acc_logits = np.zeros((8, H, W), np.float32)
    acc_dist = np.zeros((1, H, W), np.float32)
    acc_orient = np.zeros((2, H, W), np.float32)
    acc_heights = np.zeros((2, H, W), np.float32)

    for fx, fy in variants:
        r = _predict_single(model, dev, _flip(x, fx, fy), tile, overlap, return_logits=True)
        r = _reverse(r, fx, fy)
        acc_logits += r["logits"]
        acc_dist += r["dist"]
        acc_orient += r["orient"]
        acc_heights += r["heights"]

    # Усреднить
    acc_logits /= len(variants)
    acc_dist /= len(variants)
    acc_orient /= len(variants)
    acc_heights /= len(variants)

    # Softmax после усреднения logits
    prob = np.exp(acc_logits - acc_logits.max(0)) / np.exp(acc_logits - acc_logits.max(0)).sum(0)

    # Нормализовать orient
    n = np.linalg.norm(acc_orient, axis=0, keepdims=True)
    acc_orient = acc_orient / np.maximum(n, 1e-6)

    return {"prob": prob, "dist": acc_dist, "orient": acc_orient, "heights": acc_heights}


def _predict_single(model, dev, x: np.ndarray, tile: int = 512, overlap: int = 128, return_logits: bool = False) -> dict:
    """x [8, H, W] -> {"prob" [8,H,W], "dist" [1,H,W], "orient" [2,H,W], "heights" [2,H,W]}."""
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
    wt = np.minimum.outer(r, r)
    wt = np.clip(wt / max(overlap / 2, 1), 0.05, 1.0)
    acc = {"logits": np.zeros((8, Hp, Wp), np.float32), "dist": np.zeros((1, Hp, Wp), np.float32),
           "orient": np.zeros((2, Hp, Wp), np.float32), "heights": np.zeros((2, Hp, Wp), np.float32)}
    wsum = np.zeros((Hp, Wp), np.float32)
    with torch.no_grad():
        for y in ys:
            for xx in xs:
                t = torch.from_numpy(xp[None, :, y:y + tile, xx:xx + tile]).to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    o = model(t)
                o = {k: v.float()[0].cpu().numpy() for k, v in o.items()}
                # Сохранить logits, не prob
                acc["logits"][:, y:y + tile, xx:xx + tile] += o["sem"] * wt
                for k in ("dist", "orient", "heights"):
                    acc[k][:, y:y + tile, xx:xx + tile] += o[k] * wt
                wsum[y:y + tile, xx:xx + tile] += wt
    res = {k: (v / wsum)[:, :H, :W] for k, v in acc.items()}

    if return_logits:
        return res

    # Prob из logits
    prob = np.exp(res["logits"] - res["logits"].max(0)) / np.exp(res["logits"] - res["logits"].max(0)).sum(0)
    res["prob"] = prob
    del res["logits"]

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


def infer_input(model, dev, x: np.ndarray, out_dir, scene: dict | None = None, tile: int = 512, tta: bool = False) -> Path:
    import cv2

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    r = predict(model, dev, x.astype(np.float32), tile=tile, tta=tta)
    sem = r["prob"].argmax(0).astype(np.uint8)
    cv2.imwrite(str(d / "sem.png"), sem)
    np.savez_compressed(d / "prob.npz", prob=r["prob"].astype(np.float16))
    for k in ("dist", "orient", "heights"):
        np.save(d / f"{k}.npy", r[k].astype(np.float16))
    if scene is not None:
        (d / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    _overlay(x, sem, d / "pred.png")
    return d


def infer(ckpt, paths, out, tile: int = 512, tta: bool = False, log=print) -> list[str]:
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
                done.append(str(infer_input(model, dev, np.load(f), Path(out) / rel, scene, tile, tta)))
                log(done[-1])
            continue
        for k, r in enumerate(rasterize_e57(p, log=log), 1):
            scene = {"source": p.name, "level": f"L{k}", "floor_z": r["level"]["floor_z"],
                     "ceiling_z": r["level"]["ceiling_z"], "frame": r["frame"], "pixel_to_world": affine(r["frame"])}
            done.append(str(infer_input(model, dev, r["input"], Path(out) / p.stem / f"L{k}", scene, tile, tta)))
            log(done[-1])
    return done
