"""«Предсказание сети» из разметки уровня датасета: идеальное или испорченное (как у настоящей
сети: размытые границы, шум, разрывы стен, ложные пятна, шум ориентации). Для проверки
векторизации без обученной сети."""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def _fill_ignore(sem: np.ndarray) -> np.ndarray:
    """255 (кольцо ignore) -> класс ближайшего размеченного пикселя."""
    ign = (sem == 255).astype(np.uint8)
    if not ign.any():
        return sem.copy()
    _, lab = cv2.distanceTransformWithLabels(ign, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    src = np.zeros(lab.max() + 1, np.uint8)
    ys, xs = np.nonzero(ign == 0)
    src[lab[ys, xs]] = sem[ys, xs]
    return src[lab]


def make_pred(level_dir, out_dir, degrade: float = 0.0, seed: int = 0) -> Path:
    """level_dir - уровень датасета (sem.png, dist/orient/heights.npy, scene.json) -> каталог
    как у infer: sem.png, prob.npz, dist/orient/heights.npy, scene.json. degrade 0..1."""
    rng = np.random.default_rng(seed)
    d, o = Path(level_dir), Path(out_dir)
    o.mkdir(parents=True, exist_ok=True)
    sem = _fill_ignore(cv2.imread(str(d / "sem.png"), cv2.IMREAD_UNCHANGED))
    H, W = sem.shape
    logit = np.stack([(sem == k).astype(np.float32) for k in range(8)]) * 6.0
    orient = np.load(d / "orient.npy").astype(np.float32)
    if degrade > 0:
        logit = np.stack([cv2.GaussianBlur(c, (0, 0), 1.0 + 1.5 * degrade) for c in logit])
        for c in range(8):                               # гладкий шум
            n = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3.0)
            logit[c] += n * 8.0 * degrade
        struct = np.isin(sem, (2, 3, 4, 5))
        ys, xs = np.nonzero(struct)
        for _ in range(int(12 * degrade)):               # разрывы стен 8-20 см
            k = rng.integers(len(ys))
            r = int(rng.integers(4, 10))
            cv2.circle(logit[1], (int(xs[k]), int(ys[k])), r, 12.0, -1)
        ys, xs = np.nonzero(sem == 1)
        for _ in range(int(10 * degrade)):               # ложные пятна «стена» в помещении
            k = rng.integers(len(ys))
            cv2.ellipse(logit[2], (int(xs[k]), int(ys[k])), (int(rng.integers(3, 12)), int(rng.integers(2, 4))),
                        float(rng.uniform(0, 180)), 0, 360, 12.0, -1)
        a = np.arctan2(orient[1], orient[0]) + np.radians(rng.normal(0, 4.0 * degrade, (H, W)))
        m = np.hypot(orient[0], orient[1]) > 0.5
        orient = np.where(m, np.stack([np.cos(a), np.sin(a)]), orient).astype(np.float32)
    prob = np.exp(logit - logit.max(0))
    prob /= prob.sum(0, keepdims=True)
    cv2.imwrite(str(o / "sem.png"), prob.argmax(0).astype(np.uint8))
    np.savez_compressed(o / "prob.npz", prob=prob.astype(np.float16))
    np.save(o / "orient.npy", orient.astype(np.float16))
    for k in ("dist", "heights"):
        shutil.copy(d / f"{k}.npy", o / f"{k}.npy")
    sc = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    (o / "scene.json").write_text(json.dumps(sc), encoding="utf-8")
    return o
