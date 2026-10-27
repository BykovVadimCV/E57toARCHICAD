"""Загрузчик датасета плана (раскладка simscan/netinput.py) для обучения.

    from simscan.torchdata import FloorplanSamples, torch_dataset
    ds = torch_dataset("dataset", "train", crop=512)      # нужен torch
    s = FloorplanSamples("dataset", "train")[0]           # то же на numpy, без torch

Образец: input [8, h, w] float32; sem [h, w] int64 (255 - ignore, только поля кропа); dist [1, h, w];
orient [2, h, w]; heights [2, h, w]; mask_orient, mask_heights [h, w] bool.

Аугментации на растре - только точные: случайный кроп, отражения и повороты на 90°
(без интерполяции), выпадение каналов. Повороты на произвольный угол, прореживание,
сдвиг пола - на облаке до растеризации (netinput.rasterize_e57: thin, z_shift), иначе
появятся артефакты, которых нет в реальных данных. Ориентация (cos 2θ, sin 2θ) пересчитывается
вместе с отражением и поворотом.

Улучшения v2:
- Multiplicative/additive noise на входные каналы
- Random erasing (cutout): стёртая область во входе = улица в разметке (расстояние до грани -
  «далеко»). Сеть не должна достраивать то, чего скан не видел (ТЗ): нет точек - нет помещения;
  с ignore она там ничему не училась и дорисовывала помещения в пустоте реальных сканов. Тени
  от мебели и неотсканированные зоны моделируются на облаке (симуляция сканера), а не здесь.
- unseen_outside: в датасетах labels-2/3 неотснятое помечено 255 (ignore) - читается как улица,
  как в labels-4 (netinput.unseen_to_outside); у датасетов версии 1 255 - ещё и кольцо по
  границам классов, их это не касается.
- Weighted sampling: oversample сцены с проёмами (door/window малочисленны)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

IGNORE = 255
OUTSIDE = 0
WALLISH = (2, 3, 4, 5)
OPENINGS = (3, 4, 5)


def load_level(d: Path) -> dict:
    import cv2

    d = Path(d)
    out = {"input": np.load(d / "input.npy").astype(np.float32)}
    if (d / "sem.png").exists():
        out["sem"] = cv2.imread(str(d / "sem.png"), cv2.IMREAD_UNCHANGED).astype(np.int64)
        out["dist"] = np.load(d / "dist.npy").astype(np.float32)
        out["orient"] = np.load(d / "orient.npy").astype(np.float32)
        out["heights"] = np.load(d / "heights.npy").astype(np.float32)
    return out


def _flip_x(s: dict) -> dict:                      # столбцы: x -> -x, угол θ -> π - θ
    out = {k: (v[..., ::-1].copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"][1] *= -1
    return out


def _flip_y(s: dict) -> dict:                      # строки: y -> -y, θ -> -θ
    out = {k: (v[..., ::-1, :].copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"][1] *= -1
    return out


def _rot90(s: dict) -> dict:                       # против часовой на 90°: θ -> θ + 90°, 2θ -> 2θ + 180°
    out = {k: (np.rot90(v, 1, axes=(-2, -1)).copy() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
    if "orient" in out:
        out["orient"] *= -1
    return out


class FloorplanSamples:
    """Образцы на numpy. crop - сторона кропа (None - уровень целиком); train - аугментации."""

    def __init__(self, root, split: str = "train", crop: int | None = 512, train: bool = True,
                 channel_dropout: float = 0.1, multiplicative_noise: float = 0.15,
                 additive_noise: float = 0.05, random_erasing: float = 0.3,
                 weighted_sampling: bool = True, seed: int = 0, unseen_outside: bool = True):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text(encoding="utf-8"))
        self.unseen_outside = unseen_outside and int(self.meta.get("version", 1)) >= 2
        f = self.root / "splits" / f"{split}.txt"
        self.items = [x for x in f.read_text(encoding="utf-8").split() if x] if f.exists() else []
        self.crop, self.train, self.p_drop = crop, train, channel_dropout
        self.mult_noise, self.add_noise = multiplicative_noise, additive_noise
        self.p_erase = random_erasing
        self.weighted = weighted_sampling and train
        self.rng = np.random.default_rng(seed)

        # Вычислить веса для weighted sampling
        self.weights = None
        if self.weighted:
            self.weights = self._compute_sample_weights()

    def __len__(self) -> int:
        return len(self.items)

    def _compute_sample_weights(self) -> np.ndarray:
        """Вычислить веса для weighted sampling: oversample сцены с проёмами."""
        import cv2

        weights = np.ones(len(self.items), dtype=np.float32)
        for i, item in enumerate(self.items):
            sem_path = self.root / "scenes" / item / "sem.png"
            if sem_path.exists():
                sem = cv2.imread(str(sem_path), cv2.IMREAD_UNCHANGED)
                if sem is not None:
                    # Доля проёмов (door=3, window=4, opening=5)
                    opening_mask = np.isin(sem, OPENINGS)
                    valid_mask = sem != IGNORE
                    opening_ratio = opening_mask.sum() / max(valid_mask.sum(), 1)
                    # Oversample если проёмов > 3%: weight = 1 + 2 * (ratio / 0.05)
                    if opening_ratio > 0.03:
                        weights[i] = 1.0 + 2.0 * (opening_ratio / 0.05)
        return weights

    def _crop(self, s: dict) -> dict:
        C = self.crop
        H, W = s["input"].shape[-2:]
        ph, pw = max(0, C - H), max(0, C - W)
        if ph or pw:                                   # мал - дополнить: вход 0, разметка ignore
            pad = lambda a, v: np.pad(a, [(0, 0)] * (a.ndim - 2) + [(0, ph), (0, pw)], constant_values=v)  # noqa: E731
            s = {k: pad(v, IGNORE if k == "sem" else 0) for k, v in s.items()}
            H, W = H + ph, W + pw
        if self.train:
            # кроп чаще с размеченным (не только улицей)
            for _ in range(10):
                y = int(self.rng.integers(0, H - C + 1))
                x = int(self.rng.integers(0, W - C + 1))
                if "sem" not in s or ((s["sem"][y:y + C, x:x + C] % 255) > 0).mean() > 0.05:
                    break
        else:
            y, x = (H - C) // 2, (W - C) // 2
        return {k: v[..., y:y + C, x:x + C] for k, v in s.items()}

    def _augment(self, s: dict) -> dict:
        """Шум на входе и cutout. Стёртое во входе - ignore в разметке (сеть не учится
        достраивать невидимое). Каналы местами не меняются: occ_low / occ_mid - ровно тот
        признак, по которому окно отличается от двери."""
        inp = s["input"]
        C, H, W = inp.shape

        # 1. Multiplicative noise на каналы density и occ_* (0,1,2,5 - density тоже)
        if self.mult_noise > 0:
            noise_channels = [0, 1, 2, 5]  # occ_low, occ_mid, occ_high, density
            for c in noise_channels:
                if c < C:
                    factor = 1.0 + self.rng.uniform(-self.mult_noise, self.mult_noise)
                    inp[c] = np.clip(inp[c] * factor, 0, 1)

        # 2. Additive noise на все каналы
        if self.add_noise > 0:
            noise = self.rng.uniform(-self.add_noise, self.add_noise, (C, H, W)).astype(np.float32)
            inp = np.clip(inp + noise, 0, 1)

        # 3. Random erasing (cutout): 1-2 прямоугольника, в разметке - ignore
        if self.p_erase > 0 and self.rng.random() < self.p_erase:
            n_holes = int(self.rng.integers(1, 3))
            for _ in range(n_holes):
                # Размер дыры: 10-30% площади
                area = H * W
                hole_area = self.rng.uniform(0.1, 0.3) * area
                aspect = self.rng.uniform(0.5, 2.0)
                hole_h = int(np.sqrt(hole_area * aspect))
                hole_w = int(np.sqrt(hole_area / aspect))
                hole_h = min(hole_h, H)
                hole_w = min(hole_w, W)
                y = int(self.rng.integers(0, H - hole_h + 1)) if hole_h < H else 0
                x = int(self.rng.integers(0, W - hole_w + 1)) if hole_w < W else 0
                inp[:, y:y + hole_h, x:x + hole_w] = 0.0
                if "sem" in s:
                    hole = s["sem"][y:y + hole_h, x:x + hole_w]
                    if self.unseen_outside:                 # нет точек - нет помещения; поля кропа - как были
                        s["dist"][:, y:y + hole_h, x:x + hole_w][:, hole != IGNORE] = 1.0
                        hole[hole != IGNORE] = OUTSIDE
                    else:
                        hole[:] = IGNORE

        s["input"] = inp
        return s

    def __getitem__(self, idx: int) -> dict:
        s = load_level(self.root / "scenes" / self.items[idx])
        if self.unseen_outside and "sem" in s:
            m = s["sem"] == IGNORE
            s["sem"][m] = OUTSIDE
            s["dist"][0][m] = 1.0
        if self.crop:
            s = self._crop(s)
        if self.train:
            if self.rng.random() < 0.5:
                s = _flip_x(s)
            if self.rng.random() < 0.5:
                s = _flip_y(s)
            for _ in range(int(self.rng.integers(0, 4))):
                s = _rot90(s)
            # Channel dropout (старая аугментация)
            drop = self.rng.random(s["input"].shape[0]) < self.p_drop
            s["input"][drop] = 0.0
            s = self._augment(s)
        if "sem" in s:
            sem = s["sem"]
            s["mask_orient"] = np.isin(sem, WALLISH)
            s["mask_heights"] = np.isin(sem, OPENINGS)
        s["id"] = self.items[idx]
        return s


def torch_dataset(root, split: str = "train", **kw):
    """Обёртка torch.utils.data.Dataset над FloorplanSamples (нужен torch)."""
    import torch
    from torch.utils.data import Dataset

    base = FloorplanSamples(root, split, **kw)

    class _DS(Dataset):
        def __init__(self):
            self.base = base               # для worker_init_fn: своя случайность в каждом процессе

        def __len__(self):
            return len(self.base)

        def __getitem__(self, i):
            s = self.base[i]
            return {k: (torch.from_numpy(np.ascontiguousarray(v)) if isinstance(v, np.ndarray) else v)
                    for k, v in s.items()}

    return _DS()


def make_weighted_sampler(dataset, seed: int = 0):
    """Создать WeightedRandomSampler если weighted sampling включен."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    if hasattr(dataset, 'base') and dataset.base.weights is not None:
        weights = torch.from_numpy(dataset.base.weights)
        return WeightedRandomSampler(weights, len(weights), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    return None
