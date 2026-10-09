"""Датасет сети плана: вход тем же растеризатором, что и реальный E57, разметка совмещена."""
import json

import cv2
import numpy as np
import pytest

from simscan.config import load_config
from simscan.generate import generate_scene
from simscan.netinput import CHANNELS, build_scene
from simscan.torchdata import FloorplanSamples, _flip_x, _rot90


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ds")
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.4},
                                 "export": {"preview": False, "write_merged": False, "free_space_input": False,
                                            "unet_mask": False}})
    generate_scene(cfg, tmp / "scene_00000", seed=3, index=0)
    dirs = build_scene(tmp / "scene_00000", tmp / "dataset", log=lambda *a: None)
    return tmp / "dataset", dirs


def test_format_and_alignment(built):
    root, dirs = built
    d = root / "scenes" / "scene_00000" / "L1"
    assert [str(d)] == dirs
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta["pixel_m"] == 0.02 and len(meta["channels"]) == len(CHANNELS) == 8
    x = np.load(d / "input.npy")
    assert x.dtype == np.float16 and x.shape[0] == 8
    assert float(x.min()) >= 0 and float(x.max()) <= 1
    sem = cv2.imread(str(d / "sem.png"), cv2.IMREAD_UNCHANGED)
    assert sem.dtype == np.uint8 and sem.shape == x.shape[1:]
    cls = set(np.unique(sem).tolist())
    assert {0, 1, 2, 255} <= cls and cls & {3, 4}, cls
    # совмещение: грани в облаке (вертикальные, занятые внизу или вверху) - на размеченных стенах
    xf = x.astype(np.float32)
    face = (xf[6] > 0.5) & ((xf[0] > 0.5) | (xf[2] > 0.5))
    body = np.isin(sem, (2, 3, 4, 5, 255))
    dt = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
    assert face.sum() > 500 and (dt[face] <= 1.0).mean() > 0.8
    # головы: ориентация - на стенах и проёмах, высоты - на проёмах, расстояние в [0, 1]
    ori = np.load(d / "orient.npy").astype(np.float32)
    hts = np.load(d / "heights.npy").astype(np.float32)
    dist = np.load(d / "dist.npy").astype(np.float32)
    assert ori.shape == (2,) + sem.shape and hts.shape == (2,) + sem.shape and dist.shape == (1,) + sem.shape
    wall = np.isin(sem, (2, 3, 4, 5))
    assert np.allclose(np.hypot(ori[0], ori[1])[wall], 1, atol=0.01)
    assert (np.abs(ori[:, ~wall & (sem != 255)]) < 1e-6).all()
    op = np.isin(sem, (3, 4, 5))
    assert (hts[1][op] > 0.5).all() and (hts[1][op] <= 1.0).all()
    assert 0 <= dist.min() and dist.max() <= 1 and dist[0][wall].mean() < 0.3
    vec = json.loads((d / "vector.json").read_text(encoding="utf-8"))
    assert vec["walls"] and vec["rooms"] and vec["openings"]
    sc = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert abs(sc["floor_z"] - sc["floor_z_gt"]) < 0.03          # уровень найден тем же кодом, что у реального
    split = [s for s in ("train", "val", "test") if "scene_00000/L1" in (root / "splits" / f"{s}.txt").read_text()]
    assert len(split) == 1


def test_loader_and_exact_augmentations(built):
    root, _ = built
    split = [s for s in ("train", "val", "test") if (root / "splits" / f"{s}.txt").read_text().strip()][0]
    ds = FloorplanSamples(root, split, crop=256, train=True, seed=1)
    s = ds[0]
    assert s["input"].shape == (8, 256, 256) and s["sem"].shape == (256, 256)
    assert s["mask_orient"].dtype == bool
    # поворот на 90° четыре раза - тождество; отражение меняет знак sin 2θ
    base = FloorplanSamples(root, split, crop=None, train=False)[0]
    r = base
    for _ in range(4):
        r = _rot90(r)
    assert np.array_equal(r["sem"], base["sem"]) and np.allclose(r["orient"], base["orient"])
    f = _flip_x(base)
    assert np.allclose(f["orient"][1][:, ::-1], -base["orient"][1]) and np.allclose(f["orient"][0][:, ::-1], base["orient"][0])
