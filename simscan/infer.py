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


def _state_dict(ck) -> dict:
    """Веса из чекпойнта любого из обычных видов: {"model": ...}, {"state_dict": ...} (Lightning,
    префикс "model."), {"model_state_dict": ...} или сам state_dict; префикс "module." (DataParallel) - вон."""
    sd = ck
    if isinstance(ck, dict):
        for key in ("model", "state_dict", "model_state_dict", "net_state_dict"):
            if isinstance(ck.get(key), dict) and ck[key] and all(hasattr(v, "shape") for v in ck[key].values()):
                sd = ck[key]
                break
    if not isinstance(sd, dict) or not sd or not all(hasattr(v, "shape") for v in sd.values()):
        raise ValueError("в файле нет весов сети (ожидался словарь тензоров или {'model': ...})")
    for pre in ("module.", "model.", "_orig_mod."):
        if all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    return sd


def describe_checkpoint(ckpt) -> dict:
    """Что за сеть в чекпойнте (без загрузки модели): {"format", "in_channels", "classes", "text"}."""
    import torch

    from .smpunet import is_smp_unet

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = _state_dict(ck)
    if "stem.0.weight" in sd and "sem.weight" in sd:
        return {"format": "plannet", "in_channels": int(sd["stem.0.weight"].shape[1]),
                "classes": int(sd["sem.weight"].shape[0]), "text": "PlanNet (simscan train)"}
    if is_smp_unet(sd):
        from .smpunet import SmpUnet

        m = SmpUnet(sd)
        return {"format": "smp_unet", "in_channels": m.in_channels, "classes": m.classes, "text": m.describe()}
    heads = sorted({k.split(".")[0] for k in sd})
    return {"format": "unknown", "text": f"неизвестная архитектура, верхние модули: {', '.join(heads[:12])}"}


def load_model(ckpt, device=None, log=print):
    """Чекпойнт -> (модель, устройство). Понимает PlanNet (simscan train) и smp.Unet с кодировщиком
    ResNet (simscan/smpunet.py, без smp); у последней только классы, если выход 8 каналов."""
    import torch

    from .net import PlanNet
    from .smpunet import N_CLASSES, SmpUnet, is_smp_unet

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = _state_dict(ck)
    if is_smp_unet(sd):
        m = SmpUnet(sd)
        if m.in_channels != 8 or m.classes not in (N_CLASSES, N_CLASSES + 5):
            raise ValueError(f"{Path(ckpt).name}: {m.describe()} - конвейеру нужен вход 8 каналов (netinput.CHANNELS) "
                             f"и выход {N_CLASSES} классов (или {N_CLASSES} + 5 регрессий)")
        bad = m.load_state_dict(sd, strict=False)
        missing = [k for k in bad.missing_keys if not k.endswith("num_batches_tracked")]
        if missing or bad.unexpected_keys:
            raise ValueError(f"{Path(ckpt).name}: веса smp Unet не легли: нет {missing[:5]}, лишние {bad.unexpected_keys[:5]}")
        log(f"модель: {m.describe()}" + ("" if m.classes > N_CLASSES else
                                          "; регрессионных голов нет - ориентация стен и высоты проёмов не используются"))
        return m.to(dev).eval(), dev
    if not ("stem.0.weight" in sd and "sem.weight" in sd):
        raise ValueError(f"{Path(ckpt).name}: {describe_checkpoint(ckpt)['text']} - это не PlanNet и не smp Unet(ResNet)")
    m = PlanNet(**(ck.get("net", {}) if isinstance(ck, dict) else {})).to(dev).eval()
    m.load_state_dict(sd)
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
    if "orient" in out and k % 2:
        out["orient"] = -out["orient"]
    if "orient" in out and flip:
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
    if "orient" in res:
        n = np.linalg.norm(res["orient"], axis=0, keepdims=True)
        res["orient"] = res["orient"] / np.maximum(n, 1e-6)
    return res


def _predict_single(model, dev, x: np.ndarray, tile: int = 512, overlap: int = 128, return_logits: bool = False,
                    batch: int = 4) -> dict:
    """Одно положение входа: окна tile x tile с перекрытием, пачками по batch."""
    import torch

    C, H, W = x.shape
    overlap = min(overlap, tile // 4)                      # при малом окне шаг не обнуляется (512 -> 128 как было)
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
    acc = {"logits": np.zeros((8, Hp, Wp), np.float32)}           # регрессии - если сеть их выдаёт
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
                    if k in o:
                        acc.setdefault(k, np.zeros((o[k].shape[1], Hp, Wp), np.float32))
                        acc[k][:, y:y + tile, xx:xx + tile] += o[k][n_] * wt
                wsum[y:y + tile, xx:xx + tile] += wt
    res = {k: (v / wsum)[:, :H, :W] for k, v in acc.items()}
    if return_logits:
        return res
    lg = res.pop("logits")
    e = np.exp(lg - lg.max(0))
    res["prob"] = e / e.sum(0)
    if "orient" in res:
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

    from .vectorize import seen_from_input

    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    r = predict(model, dev, x.astype(np.float32), tile=tile, tta=tta, batch=batch)
    sem = r["prob"].argmax(0).astype(np.uint8)
    cv2.imwrite(str(d / "sem.png"), sem)
    cv2.imwrite(str(d / "seen.png"), (seen_from_input(x) * 255).astype(np.uint8))   # видны пол или потолок
    np.savez_compressed(d / "prob.npz", prob=r["prob"].astype(np.float16))
    for k in ("dist", "orient", "heights"):
        if k in r:
            np.save(d / f"{k}.npy", r[k].astype(np.float16))
        else:
            (d / f"{k}.npy").unlink(missing_ok=True)      # от прошлого прогона другой сетью
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

    model, dev = load_model(ckpt, log=log)
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
        lv_dirs = []
        for k, r in enumerate(rasterize_e57(p, log=log), 1):
            scene = {"source": p.name, "level": f"L{k}", "floor_z": r["level"]["floor_z"],
                     "ceiling_z": r["level"]["ceiling_z"], "frame": r["frame"], "pixel_to_world": affine(r["frame"])}
            lv_dirs.append(infer_input(model, dev, r["input"], Path(out) / p.stem / f"L{k}", scene, tile, tta, plan,
                                       batch, log))
            done.append(str(lv_dirs[-1]))
            log(done[-1])
        plans = [d / "plan.json" for d in lv_dirs if (d / "plan.json").exists()]
        if plan and plans:                                 # грани стен - по облаку этого же E57
            from .cloudsnap import main_snap

            main_snap(plans, p, log=log)
    return done
