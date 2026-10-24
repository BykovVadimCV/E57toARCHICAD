"""Обучение сети плана (simscan/net.py) на датасете simscan/netinput.py.

    python -m simscan train --data dataset --out runs/r1 --epochs 60 --batch 8

Каждую эпоху: last.pt (для продолжения: тот же --out, флаг --resume), history.jsonl.
best.pt - по плану, а не по пикселям: каждые --vec-every эпох (и на последней) сеть
применяется к --vec-levels уровням val целиком, план строит vectorize, vector_metrics сверяет
его с vector.json, evaluate.geo_score сводит в одно число (стены 0,5, двери 0,3, окна 0,2).
best_pixel.pt - по прежнему mIoU стены и проёмов. --vec-every 0 - только по mIoU (best.pt).
На Colab --out лучше держать на Google Drive: сессия обрывается, а last.pt переживает обрыв.

Продолжение с другим числом эпох (--epochs больше, чем было): расписание OneCycle уже прошло,
поэтому на оставшиеся эпохи - новое (пик 0,3 lr). --init ckpt - новый прогон с весами готовой
модели (оптимизатор и расписание - с нуля): дообучение.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

CLASS_WEIGHTS = (0.3, 0.5, 2.0, 4.0, 4.0, 3.0, 1.0, 2.0)   # улица, помещение, стена, дверь, окно, проём, мусор, пустота
SCORED = (2, 3, 4, 5)                                     # mIoU, по которому выбирается best.pt


def _loaders(data, crop, batch, workers, seed, weighted_sampling: bool = False):
    import torch
    from torch.utils.data import DataLoader

    from .torchdata import torch_dataset, make_weighted_sampler

    tr = torch_dataset(data, "train", crop=crop, train=True, seed=seed, weighted_sampling=weighted_sampling)
    va = torch_dataset(data, "val", crop=None, train=False)          # val - уровни целиком

    def init(wid):                     # у каждого процесса загрузчика своя случайность аугментаций
        info = torch.utils.data.get_worker_info()
        info.dataset.base.rng = np.random.default_rng([seed, wid, int(torch.initial_seed() % 2**31)])

    kw = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)

    # Weighted sampling для баланса классов
    sampler = make_weighted_sampler(tr, seed) if weighted_sampling else None
    if sampler is not None:
        tl = DataLoader(tr, batch_size=batch, sampler=sampler, drop_last=len(tr) > batch,
                       worker_init_fn=init, **kw)
    else:
        tl = DataLoader(tr, batch_size=batch, shuffle=True, drop_last=len(tr) > batch,
                       worker_init_fn=init, **kw)

    vl = DataLoader(va, batch_size=1, shuffle=False, **kw)          # уровни разного размера
    return tl, vl


def _to(batch, dev):
    return {k: (v.to(dev, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}


def evaluate(model, loader, dev, amp: bool = True) -> dict:
    import torch

    from .net import confusion, iou_from_confusion, losses, N_CLASSES

    model.eval()
    cm = torch.zeros(N_CLASSES, N_CLASSES, dtype=torch.long, device=dev)
    tot, n = {}, 0
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp and dev.type == "cuda"):
        for b in loader:
            b = _to(b, dev)
            out = model(b["input"])
            out = {k: v.float() for k, v in out.items()}
            for k, v in losses(out, b).items():
                tot[k] = tot.get(k, 0.0) + float(v)
            n += 1
            cm += confusion(out["sem"].argmax(1), b["sem"])
    iou = iou_from_confusion(cm.cpu())
    res = {f"val_{k}": v / max(n, 1) for k, v in tot.items()}
    res["iou"] = [None if math.isnan(x) else round(x, 4) for x in iou]
    sc = [iou[i] for i in SCORED if not math.isnan(iou[i])]
    res["miou_walls"] = float(np.mean(sc)) if sc else 0.0
    return res


def _save(obj, path: Path) -> None:
    """torch.save через временный файл и переименование: обрыв не оставит битый или пустой файл."""
    import os

    import torch

    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def vector_eval(model, dev, data, ids: list[str], tile: int = 512) -> dict:
    """Уровни val целиком -> сеть -> vectorize -> vector_metrics против vector.json; средние и
    geo_score. Упавший уровень - нулевые метрики (план не построился)."""
    from .evaluate import aggregate_vector, geo_score, vector_metrics
    from .infer import predict
    from .torchdata import load_level
    from .vectorize import seen_from_input, vectorize_arrays

    model.eval()
    rows = []
    for i in ids:
        d = Path(data) / "scenes" / i
        s = load_level(d)
        scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
        gt = json.loads((d / "vector.json").read_text(encoding="utf-8"))
        r = predict(model, dev, s["input"], tile=tile)
        try:
            plan = vectorize_arrays(r["prob"], r.get("orient"), r.get("heights"), scene, log=lambda *a: None,
                                    seen=seen_from_input(s["input"]))
            rows.append(vector_metrics(plan, gt))
        except Exception:                                  # noqa: BLE001 - уровень без плана
            rows.append({"axis_f1": 0.0, "door_f1": 0.0, "window_f1": 0.0})
    agg = aggregate_vector(rows)
    agg["geo_score"] = geo_score(agg)
    return agg


def train(data, out, epochs: int = 60, batch: int = 8, crop: int = 512, lr: float = 1e-3, workers: int = 2,
          resume: bool = False, seed: int = 0, amp: bool = True, max_steps: int | None = None,
          accum_steps: int = 1, weighted_sampling: bool = False, use_focal: bool = False,
          init: str | None = None, vec_every: int = 5, vec_levels: int = 48, log=print) -> dict:
    import torch

    from .net import PlanNet, losses

    torch.manual_seed(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl, vl = _loaders(data, crop, batch, workers, seed, weighted_sampling)
    log(f"устройство {dev}; train {len(tl.dataset)}, val {len(vl.dataset)} уровней")
    if weighted_sampling:
        log("weighted sampling включён для баланса классов")
    if use_focal:
        log("Focal Loss включён (gamma=2.0)")
    if accum_steps > 1:
        log(f"gradient accumulation: {accum_steps} шагов (эффективный batch={batch * accum_steps})")
    model = PlanNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # шаг планировщика - на шаг оптимизатора, то есть раз в accum_steps батчей
    steps = epochs * max(len(tl) // max(accum_steps, 1), 1)
    steps = max(steps, 10)                        # разогрев - хотя бы 2 шага (иначе деление на ноль)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=min(0.3, max(0.05, 2.5 / steps)))
    scaler = torch.amp.GradScaler(enabled=amp and dev.type == "cuda")
    cw = torch.tensor(CLASS_WEIGHTS, device=dev)
    start, best, best_geo = 0, -1.0, -1.0
    per_epoch = max(len(tl) // max(accum_steps, 1), 1)
    if resume and (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, best, best_geo = ck["epoch"] + 1, ck["best"], ck.get("best_geo", -1.0)
        # лучших весов на диске нет (потерялись при обрыве) - рекорд не в счёт, следующая проверка их запишет
        if not (out / "best.pt").exists():
            best_geo = -1.0
        if not (out / ("best_pixel.pt" if vec_every > 0 else "best.pt")).exists():
            best = -1.0
        if ck["sched"].get("total_steps") == steps:
            sched.load_state_dict(ck["sched"])
        else:                                          # другое число эпох или другой датасет
            rest = max((epochs - start) * per_epoch, 10)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.3 * lr, total_steps=rest,
                                                        pct_start=min(0.1, max(0.05, 2.5 / rest)))
            log(f"расписание было на {ck['sched'].get('total_steps')} шагов, теперь {steps}: на оставшиеся "
                f"{epochs - start} эпох - новое, пик {0.3 * lr:g}")
        log(f"продолжение с эпохи {start}, лучший mIoU {best:.3f}" +
            (f", лучший счёт плана {best_geo:.3f}" if best_geo >= 0 else ""))
    elif init:
        from .infer import _state_dict

        ck0 = torch.load(init, map_location=dev, weights_only=False)
        model.load_state_dict(_state_dict(ck0))
        log(f"веса из {init} (эпоха {ck0.get('epoch', '?') if isinstance(ck0, dict) else '?'}); оптимизатор и "
            "расписание - с нуля")
    val_ids = list(getattr(vl.dataset, "base", vl.dataset).items) if len(vl.dataset) else []
    vec_ids = [val_ids[k] for k in np.linspace(0, len(val_ids) - 1, min(vec_levels, len(val_ids))).astype(int)] \
        if val_ids and vec_every > 0 else []
    meta = json.loads((Path(data) / "meta.json").read_text(encoding="utf-8"))
    step = 0
    accum_step = 0
    for ep in range(start, epochs):
        model.train()
        t0, agg, n = time.time(), {}, 0
        for b in tl:
            b = _to(b, dev)
            with torch.autocast(dev.type, enabled=amp and dev.type == "cuda"):
                o = model(b["input"])
            o = {k: v.float() for k, v in o.items()}
            parts = losses(o, b, class_weights=cw, use_focal=use_focal)
            # Gradient accumulation: нормализовать loss
            loss = parts["total"] / accum_steps
            scaler.scale(loss).backward()

            accum_step += 1
            if accum_step >= accum_steps:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                accum_step = 0

            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
            step += 1
            if max_steps and step >= max_steps:
                break
        rec = {"epoch": ep, **{k: round(v / max(n, 1), 4) for k, v in agg.items()},
               "lr": opt.param_groups[0]["lr"], "s": round(time.time() - t0, 1)}
        if len(vl.dataset):
            rec.update(evaluate(model, vl, dev, amp))
        last_ep = ep == epochs - 1 or bool(max_steps and step >= max_steps)
        if vec_ids and ((ep + 1) % vec_every == 0 or last_ep):
            t1 = time.time()
            vm = vector_eval(model, dev, data, vec_ids, tile=crop)
            rec.update({f"vec_{k}": v for k, v in vm.items() if not k.endswith("_total")})
            log(f"  план на {len(vec_ids)} уровнях val: счёт {vm['geo_score']:.3f} (оси F1 {vm.get('axis_f1')}, "
                f"p90 оси {vm.get('axis_err_cm_p90')} см, двери F1 {vm.get('door_f1')}, окна F1 "
                f"{vm.get('window_f1')}) за {time.time() - t1:.0f} с")
        score = rec.get("miou_walls", -rec["total"])
        net_cfg = {"in_channels": 8, "n_classes": 8, "width": 64}
        geo = rec.get("vec_geo_score")
        # сначала лучшие, потом last.pt: обрыв между ними (Colab) не оставит в last.pt рекорд без его best.pt
        snap = {"model": model.state_dict(), "meta": meta, "net": net_cfg, "epoch": ep, "score": score,
                "geo_score": geo}
        if score > best:
            best = score
            _save(snap, out / ("best_pixel.pt" if vec_ids else "best.pt"))
        if geo is not None and geo > best_geo:
            best_geo = geo
            _save(snap, out / "best.pt")
            log(f"  best.pt <- эпоха {ep} (счёт плана {geo:.3f})")
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
              "scaler": scaler.state_dict(), "epoch": ep, "best": best, "meta": meta, "net": net_cfg,
              "best_geo": best_geo}
        _save(ck, out / "last.pt")
        with open(out / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log(f"эпоха {ep}: loss {rec['total']:.3f}" + (f", val mIoU стен {rec['miou_walls']:.3f}"
                                                      if "miou_walls" in rec else "") + f" ({rec['s']} с)")
        if max_steps and step >= max_steps:
            break
    return {"best": best, "best_geo": best_geo, "out": str(out)}
