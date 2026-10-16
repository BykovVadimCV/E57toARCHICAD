"""Снимки E57: блобы читаются как есть, параметры и позы верны, соглашение граней куба
определяется по швам, панорама стоянки - по осям мира. Тестовый узор - функция направления
(азимут, угол места), грани и сфера нарезаются из него и пишутся в E57 через libE57."""
import json
import math

import numpy as np
import pytest

from simscan.e57images import Camera, conventions, extract

pye57 = pytest.importorskip("pye57")


def pattern(d: np.ndarray) -> np.ndarray:
    """Цвет по направлению в мире (несимметричный: ошибки поворота и зеркала видны)."""
    a = np.arctan2(d[:, 1], d[:, 0])
    e = np.arcsin(np.clip(d[:, 2], -1, 1))
    r = 127 + 120 * np.sin(2 * a)
    g = 127 + 120 * np.sin(3 * e + a)
    b = np.where((np.floor(a * 4 / math.pi) + np.floor(e * 6 / math.pi)) % 2 == 0, 220, 40)
    return np.c_[r, g, b].clip(0, 255).astype(np.uint8)


def render(cam: Camera) -> np.ndarray:
    jj, ii = np.meshgrid(np.arange(cam.width) + 0.5, np.arange(cam.height) + 0.5)
    return pattern(cam.rays(jj.ravel(), ii.ravel())).reshape(cam.height, cam.width, 3)


def cube_faces(center, size, conv):
    """6 граней (90°) в соглашении conv: оси снимка = оси OpenCV-грани, умноженные на знаки."""
    faces = []
    for fwd, down in (((1, 0, 0), (0, 0, -1)), ((-1, 0, 0), (0, 0, -1)), ((0, 1, 0), (0, 0, -1)),
                      ((0, -1, 0), (0, 0, -1)), ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, 1, 0))):
        z, y = np.array(fwd, float), np.array(down, float)
        R_cv = np.c_[np.cross(y, z), y, z]                 # x вправо, y вниз, z вперёд
        R = R_cv @ np.diag(conv)                           # оси снимка: (sx x, sy y, fz z)
        faces.append(Camera("pinhole", size, size, R, np.asarray(center, float), f_px=size / 2, f_py=size / 2,
                            cx=size / 2, cy=size / 2, conv=tuple(conv)))
    return faces


def write(path, frames):
    """E57: маленькое облако + снимки [(Camera, RGB)] со своими позами."""
    import cv2
    from pye57 import libe57

    from simscan.transform import quat_from_matrix

    pts = np.random.default_rng(0).random((100, 3))
    blobs = []
    with pye57.E57(str(path), mode="w") as e:
        e.write_scan_raw({"cartesianX": pts[:, 0], "cartesianY": pts[:, 1], "cartesianZ": pts[:, 2]}, name="s",
                         rotation=np.array([1.0, 0, 0, 0]), translation=np.zeros(3))
        imf = e.image_file
        images = e.root["images2D"]
        images = libe57.VectorNode(images) if not isinstance(images, libe57.VectorNode) else images
        for k, (cam, rgb) in enumerate(frames):
            data = cv2.imencode(".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            blobs.append(data)
            im = libe57.StructureNode(imf)
            im.set("guid", libe57.StringNode(imf, f"img-{k}"))
            im.set("name", libe57.StringNode(imf, f"Setup face {k}"))
            pose, rot, tr = libe57.StructureNode(imf), libe57.StructureNode(imf), libe57.StructureNode(imf)
            for key, v in zip("wxyz", quat_from_matrix(cam.R)):
                rot.set(key, libe57.FloatNode(imf, float(v)))
            for key, v in zip("xyz", cam.center):
                tr.set(key, libe57.FloatNode(imf, float(v)))
            pose.set("rotation", rot)
            pose.set("translation", tr)
            im.set("pose", pose)
            rep = libe57.StructureNode(imf)
            rep.set("imageWidth", libe57.IntegerNode(imf, cam.width))
            rep.set("imageHeight", libe57.IntegerNode(imf, cam.height))
            if cam.kind == "pinhole":
                f = 0.01
                for key, v in (("focalLength", f), ("pixelWidth", f / cam.f_px), ("pixelHeight", f / cam.f_py),
                               ("principalPointX", cam.cx), ("principalPointY", cam.cy)):
                    rep.set(key, libe57.FloatNode(imf, float(v)))
            else:
                rep.set("pixelWidth", libe57.FloatNode(imf, cam.pw))
                rep.set("pixelHeight", libe57.FloatNode(imf, cam.ph))
            blob = libe57.BlobNode(imf, len(data))
            rep.set("jpegImage", blob)
            im.set(f"{cam.kind}Representation", rep)
            images.append(im)
            blob.write(np.frombuffer(data, np.uint8), 0, len(data))
    return blobs


def _pano_error(out, W):
    import cv2

    pano = cv2.imread(str(out / "panoramas" / "station_01.jpg"))[:, :, ::-1]
    assert pano.shape == (W // 2, W, 3)
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(W // 2) + 0.5)
    a = (W / 2 - jj) * 2 * math.pi / W
    e = (W / 4 - ii) * math.pi / (W / 2)
    d = np.c_[(np.cos(e) * np.cos(a)).ravel(), (np.cos(e) * np.sin(a)).ravel(), np.sin(e).ravel()]
    ref = pattern(d).reshape(pano.shape)
    band = slice(W // 16, W // 2 - W // 16)                # без полюсов
    return float(np.abs(pano[band].astype(float) - ref[band]).mean())


@pytest.mark.parametrize("conv", [conventions("pinhole")[0], conventions("pinhole")[1]])
def test_cube_faces(tmp_path, conv):
    center = np.array([3.0, -2.0, 1.5])
    frames = [(c, render(c)) for c in cube_faces(center, 256, conv)]
    src = tmp_path / "cube.e57"
    blobs = write(src, frames)
    out = tmp_path / "out"
    doc = extract(src, out, pano_width=1024, log=lambda *a: None)
    cams = doc["cameras"]
    assert len(cams) == 6 and len(doc["stations"]) == 1
    for c, data in zip(cams, blobs):                       # снимки - байт в байт
        assert (out / c["file"]).read_bytes() == data
        assert c["width"] == 256 and abs(c["f_px"] - 128) < 1e-6 and np.allclose(c["t"], center)
    assert doc["pinhole_convention"]["conv"] == list(conv) and doc["pinhole_convention"]["sure"]
    assert _pano_error(out, 1024) < 12.0                   # панорама - тот же узор по осям мира
    assert json.loads((out / "cameras.json").read_text(encoding="utf-8"))["stations"][0]["panorama"]


def test_spherical_rotated(tmp_path):
    """Сферический снимок с повёрнутой позой -> панорама по осям мира."""
    th = math.radians(30)
    R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
    cam = Camera("spherical", 1024, 512, R, np.array([1.0, 1.0, 1.4]), pw=2 * math.pi / 1024, ph=math.pi / 512)
    src = tmp_path / "sph.e57"
    write(src, [(cam, render(cam))])
    out = tmp_path / "out"
    doc = extract(src, out, pano_width=1024, log=lambda *a: None)
    assert doc["cameras"][0]["representation"] == "spherical" and doc["pinhole_convention"] == {}
    assert _pano_error(out, 1024) < 12.0
