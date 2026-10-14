"""TTA в infer.predict сохраняет ориентацию косых стен; аугментации не портят разметку."""
import json

import cv2
import numpy as np

from simscan import infer
from simscan.torchdata import IGNORE, FloorplanSamples


def _fake_single(model, dev, x, tile=512, overlap=128, return_logits=False, **kw):
    """Эквивариантная «сеть»: (cos 2θ, sin 2θ) направления линии по тензору структуры x[0]."""
    gy, gx = np.gradient(x[0].astype(np.float64))
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    phi2 = np.arctan2(2 * jxy, jxx - jyy) + np.pi          # 2θ линии = 2θ градиента + 180°
    H, W = x.shape[-2:]
    orient = np.stack([np.full((H, W), np.cos(phi2)), np.full((H, W), np.sin(phi2))]).astype(np.float32)
    z = lambda c: np.zeros((c, H, W), np.float32)  # noqa: E731
    return {"logits": z(8), "dist": z(1), "orient": orient, "heights": z(2)}


def _line(deg, n=96):
    img = np.zeros((n, n), np.uint8)
    t = np.deg2rad(deg)
    c, d = np.array([n / 2, n / 2]), np.array([np.cos(t), -np.sin(t)]) * n
    cv2.line(img, tuple(int(v) for v in c - d), tuple(int(v) for v in c + d), 255, 3, cv2.LINE_AA)
    x = np.zeros((8, n, n), np.float32)
    x[0] = cv2.GaussianBlur(img, (0, 0), 1.5) / 255.0
    return x


def test_tta_keeps_oblique_orientation(monkeypatch):
    monkeypatch.setattr(infer, "_predict_single", _fake_single)
    for deg in (30, 60, 88, 93, 120):
        x = _line(deg)
        plain = infer.predict(None, None, x, tta=False)["orient"][:, 48, 48]
        assert abs(plain[1]) > 0.05, deg                    # sin 2θ не ноль - стена косая
        for mode in (True, "flip", "d4"):                   # отражения и повороты на 90°
            tta = infer.predict(None, None, x, tta=mode)["orient"][:, 48, 48]
            assert np.allclose(plain, tta, atol=1e-4), (deg, mode, plain, tta)


def _dataset(tmp_path, n=64):
    d = tmp_path / "scenes" / "s0"
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    np.save(d / "input.npy", rng.random((8, n, n)).astype(np.float16))
    sem = np.ones((n, n), np.uint8)
    sem[:, 30:34] = 2
    sem[20:28, 30:34] = 4
    cv2.imwrite(str(d / "sem.png"), sem)
    np.save(d / "dist.npy", rng.random((1, n, n)).astype(np.float16))
    np.save(d / "orient.npy", np.stack([np.ones((n, n)), np.zeros((n, n))]).astype(np.float16))
    np.save(d / "heights.npy", rng.random((2, n, n)).astype(np.float16))
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "train.txt").write_text("s0\n", encoding="utf-8")
    (tmp_path / "meta.json").write_text(json.dumps({}), encoding="utf-8")


def test_erasing_marks_labels_ignore(tmp_path):
    _dataset(tmp_path)
    ds = FloorplanSamples(tmp_path, "train", crop=None, channel_dropout=0.0, multiplicative_noise=0.0,
                          additive_noise=0.0, random_erasing=1.0, weighted_sampling=False, seed=3)
    for _ in range(20):
        s = ds[0]
        erased = (s["input"] == 0).all(0)
        assert erased.any()
        assert (s["sem"][erased] == IGNORE).all()            # стёртое не размечено
        assert not s["mask_orient"][erased].any() and not s["mask_heights"][erased].any()
        assert (s["sem"][~erased] != IGNORE).all()
