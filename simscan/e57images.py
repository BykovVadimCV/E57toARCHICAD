"""Снимки E57 (images2D): извлечение исходных снимков, параметров камер и панорам стоянок.

    python -m simscan images kvartira.e57 --out images/
    -> images/NNN_<имя>.jpg|png   снимки как есть (байты из файла, без перекодирования)
       panoramas/station_NN.jpg   панорама стоянки: равнопромежуточная 2:1, по осям мира
       cameras.json               позы и параметры всех снимков, стоянки, формула панорам

Панорама стоянки (для разметки окон и дверей и обучения): пиксель (u, v), ширина W, высота
H = W / 2 -> луч из центра стоянки center в мировых осях:
    азимут     a = (W / 2 - u) * 2 pi / W   (0 - мировая +X в середине кадра, против часовой - влево)
    угол места e = (H / 2 - v) * pi / H     (верх кадра - зенит)
    луч = (cos e cos a, cos e sin a, sin e)
По нему результат детектора на панораме ложится на план сети и облако.

Формат (ASTM E2807, «E57 best coordinates»): images2D - вектор структур: guid, name,
associatedData3DGuid, pose (rotation - кватернион w, x, y, z; translation; мир = R p + t) и одно
из представлений:
    pinholeRepresentation   jpegImage | pngImage (Blob), imageMask, imageWidth, imageHeight,
                            focalLength (м), pixelWidth, pixelHeight (м), principalPointX, Y (пикс.)
    sphericalRepresentation jpegImage | pngImage, imageWidth, imageHeight, pixelWidth,
                            pixelHeight (рад на пиксель)
    cylindricalRepresentation  + radius, principalPointY
Проекция по стандарту: pinhole - перед камерой z < 0, u = cx - (x / z) f / pw,
v = cy - (y / z) f / ph; сферическая - u = W / 2 - azimuth / pw, v = H / 2 - elevation / ph,
азимут против часовой от +X, (0, 0) - левый верхний угол. Направления осей x, y у pinhole
стандарт явно не задаёт (буквально формула даёт левую тройку, а поза - поворот), программы
трактуют по-разному. Поэтому для граней куба соглашение выбирается по швам: при верном
соседние грани на общих краях совпадают (seam_check). Блоб в файле: секция - 16 байт
заголовка (id 0, резерв, длина uint64), затем байты JPEG/PNG.
Cyclone REGISTER 360 по умолчанию пишет 6 граней куба (pinhole, 90°, до 5120 x 5120) на стоянку.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .e57read import E57Reader, _local


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def conventions(kind: str) -> list[tuple]:
    """Варианты знаков; первый - по умолчанию. Pinhole (sx, sy, fz): u растёт вдоль sx * x,
    v (вниз) - вдоль sy * y, камера смотрит вдоль fz * z; тройка правая (sx sy fz = 1), иначе
    поза-поворот её не выразит. По умолчанию OpenGL: x вправо, y вверх, смотрит вдоль -z.
    Сфера (s, a0): u = W a0 - s * azimuth / pw."""
    if kind == "pinhole":
        return [(1, -1, -1), (1, 1, 1), (-1, 1, -1), (-1, -1, 1)]
    return [(1, 0.5), (-1, 0.5), (1, 0.0), (-1, 0.0)]


@dataclass
class Camera:
    """kind - pinhole | spherical | cylindrical; R, t - поза (мир = R p + t); conv - знаки."""
    kind: str
    width: int
    height: int
    R: np.ndarray
    t: np.ndarray
    f_px: float = 0.0                 # фокус в пикселях (pinhole)
    f_py: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    pw: float = 0.0                   # рад на пиксель (сфера, цилиндр)
    ph: float = 0.0
    radius: float = 0.0
    conv: tuple = ()

    def __post_init__(self):
        if not self.conv:
            self.conv = conventions(self.kind)[0]

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.t, float)

    def project(self, pts_world: np.ndarray):
        """Мировые точки -> (u, v, глубина, попала ли в кадр)."""
        p = (np.asarray(pts_world, float) - self.t) @ self.R
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        if self.kind == "pinhole":
            sx, sy, fz = self.conv
            depth = fz * z
            dz = np.where(np.abs(depth) > 1e-12, depth, 1e-12)
            u = self.cx + sx * (x / dz) * self.f_px
            v = self.cy + sy * (y / dz) * self.f_py
            ok = depth > 1e-9
        else:
            s, a0 = self.conv
            th, rho = np.arctan2(y, x), np.hypot(x, y)
            u = (self.width * a0 - s * th / self.pw) % self.width
            if self.kind == "spherical":
                v = self.height / 2 - np.arctan2(z, rho) / self.ph
            else:
                v = self.cy - z * (self.radius / self.ph) / np.maximum(rho, 1e-12)
            depth = np.sqrt(x * x + y * y + z * z)
            ok = depth > 1e-9
        ok &= (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return u, v, depth, ok

    def rays(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Пиксели (u, v; центр пикселя i - i + 0,5) -> единичные лучи в мире."""
        u, v = np.asarray(u, float), np.asarray(v, float)
        if self.kind == "pinhole":
            sx, sy, fz = self.conv
            d = np.c_[sx * (u - self.cx) / self.f_px, sy * (v - self.cy) / self.f_py, np.full(len(u), float(fz))]
        else:
            s, a0 = self.conv
            th = s * (self.width * a0 - u) * self.pw
            el = (self.height / 2 - v) * self.ph if self.kind == "spherical" else \
                np.arctan2((self.cy - v) * self.ph / self.radius, 1.0)
            d = np.c_[np.cos(el) * np.cos(th), np.cos(el) * np.sin(th), np.sin(el)]
        d = d / np.linalg.norm(d, axis=1, keepdims=True)
        return d @ self.R.T

    def to_json(self) -> dict:
        return {"kind": self.kind, "width": self.width, "height": self.height, "R": np.round(self.R, 9).tolist(),
                "t": np.round(np.asarray(self.t, float), 6).tolist(), "f_px": self.f_px, "f_py": self.f_py,
                "cx": self.cx, "cy": self.cy, "pw": self.pw, "ph": self.ph, "radius": self.radius,
                "conv": list(self.conv)}

    @staticmethod
    def from_json(d: dict) -> "Camera":
        d = {k: v for k, v in d.items() if k in Camera.__dataclass_fields__}
        d["R"], d["t"], d["conv"] = np.asarray(d["R"], float), np.asarray(d["t"], float), tuple(d["conv"])
        return Camera(**d)


@dataclass
class Image2D:
    index: int
    guid: str
    name: str
    scan_guid: str
    camera: Camera | None
    blob: tuple | None                # (физ. смещение, длина, jpeg | png)
    mask_blob: tuple | None = None
    representation: str = ""


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _num(el, name, default=None):
    c = _child(el, name)
    txt = (c.text or "").strip() if c is not None else ""
    return float(txt) if txt else default


def _text(el, name, default=""):
    c = _child(el, name)
    return (c.text or "").strip() if c is not None and c.text else default


def read_images(reader: E57Reader) -> list[Image2D]:
    """Описания снимков (картинки не читаются)."""
    out = []
    for k, el in enumerate(reader.images):
        R, t = np.eye(3), np.zeros(3)
        pose = _child(el, "pose")
        if pose is not None:
            rot, tr = _child(pose, "rotation"), _child(pose, "translation")
            if rot is not None:
                R = quat_to_R([_num(rot, c, 0.0) for c in "wxyz"])
            if tr is not None:
                t = np.array([_num(tr, c, 0.0) for c in "xyz"])
        cam = blob = mblob = None
        rep_name = ""
        for kind in ("pinhole", "spherical", "cylindrical", "visualReference"):
            rep = _child(el, kind + "Representation")
            if rep is None:
                continue
            rep_name = kind
            for fmt in ("jpeg", "png"):
                b = _child(rep, fmt + "Image")
                if b is not None:
                    blob = (int(b.get("fileOffset")), int(b.get("length")), fmt)
            m = _child(rep, "imageMask")
            if m is not None:
                mblob = (int(m.get("fileOffset")), int(m.get("length")), "png")
            W, H = int(_num(rep, "imageWidth", 0)), int(_num(rep, "imageHeight", 0))
            pw_, ph_ = _num(rep, "pixelWidth", 0.0), _num(rep, "pixelHeight", 0.0)
            if kind == "pinhole":
                f = _num(rep, "focalLength", 0.0)
                cam = Camera("pinhole", W, H, R, t, f_px=f / pw_ if pw_ else 0.0, f_py=f / ph_ if ph_ else 0.0,
                             cx=_num(rep, "principalPointX", W / 2), cy=_num(rep, "principalPointY", H / 2))
            elif kind == "spherical":
                cam = Camera("spherical", W, H, R, t, pw=pw_, ph=ph_)
            elif kind == "cylindrical":
                cam = Camera("cylindrical", W, H, R, t, pw=pw_, ph=ph_, radius=_num(rep, "radius", 1.0),
                             cy=_num(rep, "principalPointY", H / 2))
            break
        out.append(Image2D(k, _text(el, "guid"), _text(el, "name", f"image_{k}"), _text(el, "associatedData3DGuid"),
                           cam, blob, mblob, rep_name))
    return out


def read_blob(reader: E57Reader, blob: tuple) -> bytes:
    """Байты картинки: секция блоба - 16 байт заголовка, затем данные."""
    off, n, _ = blob
    return reader._read_logical(off, 16 + n)[16:16 + n]


def decode(data: bytes) -> np.ndarray:
    """JPEG/PNG -> RGB uint8."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("снимок не читается (не JPEG/PNG)")
    return img[:, :, ::-1].copy()


def stations(images: list[Image2D], tol_m: float = 0.05) -> list[list[int]]:
    """Снимки -> группы по положению камеры (грани куба одной стоянки - одна группа)."""
    groups, centers = [], []
    for im in images:
        if im.camera is None:
            continue
        c = im.camera.center
        for g, cc in zip(groups, centers):
            if np.linalg.norm(c - cc) <= tol_m:
                g.append(im.index)
                break
        else:
            groups.append([im.index])
            centers.append(c)
    return groups


# ----------------------------------------------------------------------
# Соглашение граней куба - по швам
# ----------------------------------------------------------------------

def seam_error(cams: list[Camera], rgbs: list[np.ndarray], step: int = 4) -> float:
    """Расхождение цвета на общих краях граней: лучи по краю каждой грани (на полпикселя
    внутрь) -> в соседние грани (допуск 2 пикселя за краем). Меньше - лучше; nan - краёв нет."""
    diffs = []
    for i, (ca, ia) in enumerate(zip(cams, rgbs)):
        W, H = ca.width, ca.height
        s = np.arange(0.5, W, step)
        t = np.arange(0.5, H, step)
        u = np.r_[s, s, np.full(len(t), 0.5), np.full(len(t), W - 0.5)]
        v = np.r_[np.full(len(s), 0.5), np.full(len(s), H - 0.5), t, t]
        col_a = ia[v.astype(int), u.astype(int)].astype(float)
        P = ca.center + ca.rays(u, v)
        for j, (cb, ib) in enumerate(zip(cams, rgbs)):
            if j == i:
                continue
            p = (P - cb.t) @ cb.R
            sx, sy, fz = cb.conv
            depth = fz * p[:, 2]
            front = depth > 1e-9
            dz = np.where(front, depth, 1.0)
            ub = cb.cx + sx * (p[:, 0] / dz) * cb.f_px
            vb = cb.cy + sy * (p[:, 1] / dz) * cb.f_py
            ok = front & (ub > -2) & (ub < cb.width + 2) & (vb > -2) & (vb < cb.height + 2)
            if not ok.any():
                continue
            ui = np.clip(ub[ok], 0, cb.width - 1).astype(int)
            vi = np.clip(vb[ok], 0, cb.height - 1).astype(int)
            diffs.append(np.abs(ib[vi, ui].astype(float) - col_a[ok]).mean(1))
    return float(np.median(np.concatenate(diffs))) if diffs else float("nan")


def seam_check(reader: E57Reader, images: list[Image2D], groups: list[list[int]], max_groups: int = 3,
               log=print) -> dict:
    """Соглашение pinhole по швам граней куба (несколько стоянок); проставляется всем pinhole.
    Если швы не различают варианты (меньше чем в 1,5 раза) - вариант по умолчанию."""
    import cv2

    cubes = [g for g in groups if sum(images[i].camera.kind == "pinhole" and images[i].blob is not None
                                      for i in g) >= 2][:max_groups]
    if not cubes:
        return {}
    small = {}
    for g in cubes:
        for i in g:
            rgb = decode(read_blob(reader, images[i].blob))
            k = max(1, rgb.shape[1] // 512)
            small[i] = (cv2.resize(cv2.GaussianBlur(rgb, (0, 0), k), (rgb.shape[1] // k, rgb.shape[0] // k),
                                   interpolation=cv2.INTER_AREA), k)
    scores = {}
    for conv in conventions("pinhole"):
        errs = []
        for g in cubes:
            cams, rgbs = [], []
            for i in g:
                c = images[i].camera
                rgb, k = small[i]
                cams.append(Camera("pinhole", rgb.shape[1], rgb.shape[0], c.R, c.t, f_px=c.f_px / k, f_py=c.f_py / k,
                                   cx=c.cx / k, cy=c.cy / k, conv=conv))
                rgbs.append(rgb)
            errs.append(seam_error(cams, rgbs))
        errs = [e for e in errs if not math.isnan(e)]
        if errs:
            scores[conv] = float(np.mean(errs))
    if not scores:
        return {}
    ranked = sorted((v, k) for k, v in scores.items())
    best = ranked[0][1]
    second = ranked[1][0] if len(ranked) > 1 else float("inf")
    sure = ranked[0][0] * 1.5 < second
    conv = best if sure else conventions("pinhole")[0]
    for im in images:
        if im.camera is not None and im.camera.kind == "pinhole":
            im.camera.conv = conv
    log(f"  соглашение граней: {list(conv)} ({'по швам' if sure else 'по умолчанию - швы не решили'}; "
        f"расхождение на швах {ranked[0][0]:.1f} против {second:.1f})")
    return {"conv": list(conv), "sure": bool(sure), "seam_error": {str(list(k)): round(v, 2) for k, v in scores.items()}}


# ----------------------------------------------------------------------
# Панорама стоянки
# ----------------------------------------------------------------------

def pano_rays(width: int, rows: slice) -> np.ndarray:
    """Лучи (мировые оси) строк rows равнопромежуточной панорамы ширины width (см. шапку)."""
    H = width // 2
    v = np.arange(H)[rows] + 0.5
    u = np.arange(width) + 0.5
    a = (width / 2 - u) * 2 * math.pi / width
    e = (H / 2 - v) * math.pi / H
    ee, aa = np.meshgrid(e, a, indexing="ij")
    return np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], -1)


def panorama(sources: list[tuple[np.ndarray, Camera]], width: int, band: int = 256) -> np.ndarray:
    """Снимки одной стоянки (RGB, камера) -> равнопромежуточная панорама по осям мира; полосами
    по band строк (грани 5120 x 5120 не держат всю сетку лучей в памяти)."""
    import cv2

    H = width // 2
    out = np.zeros((H, width, 3), np.uint8)
    filled = np.zeros((H, width), bool)
    for r0 in range(0, H, band):
        rows = slice(r0, min(H, r0 + band))
        D = pano_rays(width, rows).reshape(-1, 3)
        h = rows.stop - rows.start
        for rgb, cam in sources:
            u, v, _, ok = cam.project(cam.center + D)
            if not ok.any():
                continue
            mx = np.where(ok, u - 0.5, -1).reshape(h, width).astype(np.float32)
            my = np.where(ok, v - 0.5, -1).reshape(h, width).astype(np.float32)
            part = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m = ok.reshape(h, width) & ~filled[rows]
            out[rows][m] = part[m]
            filled[rows] |= m
    return out


def extract(e57, out_dir, panoramas: bool = True, pano_width: int | None = None, log=print) -> dict:
    """Все снимки E57 -> out/images (как есть), out/panoramas/station_NN.jpg, out/cameras.json."""
    import cv2

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    with E57Reader(e57) as r:
        ims = read_images(r)
        kinds = {k: sum(1 for i in ims if i.representation == k) for k in ("pinhole", "spherical", "cylindrical")}
        log(f"снимков {len(ims)}: " + ", ".join(f"{k} {n}" for k, n in kinds.items() if n))
        groups = stations(ims)
        conv = seam_check(r, ims, groups, log=log)
        cams = []
        for im in ims:
            if im.blob is None or im.camera is None:
                continue
            safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in im.name)[:60]
            fn = f"images/{im.index:03d}_{safe}.{'jpg' if im.blob[2] == 'jpeg' else 'png'}"
            (out / fn).write_bytes(read_blob(r, im.blob))
            cams.append({"index": im.index, "file": fn, "name": im.name, "guid": im.guid, "scan_guid": im.scan_guid,
                         "representation": im.representation, **im.camera.to_json()})
        st_doc = []
        if panoramas:
            (out / "panoramas").mkdir(exist_ok=True)
        for s, g in enumerate(groups):
            src = [im for im in (ims[i] for i in g) if im.blob is not None and im.camera is not None]
            rec = {"station": s + 1, "center": np.round(ims[g[0]].camera.center, 4).tolist(), "images": g}
            if panoramas and src:
                c = src[0].camera
                W = int(pano_width) if pano_width else (4 * c.width if c.kind == "pinhole" else c.width)
                W = int(min(8192, max(1024, W))) // 2 * 2          # как у снимков, не больше 8192
                pano = panorama([(decode(read_blob(r, im.blob)), im.camera) for im in src], W)
                fn = f"panoramas/station_{s + 1:02d}.jpg"
                cv2.imwrite(str(out / fn), pano[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
                rec.update({"panorama": fn, "width": W, "height": W // 2})
            st_doc.append(rec)
        doc = {"file": Path(e57).name, "pinhole_convention": conv,
               "panorama_projection": "equirectangular, world axes: azimuth = (W/2 - u) * 2pi/W (0 = world +X at "
                                      "centre, counter-clockwise to the left), elevation = (H/2 - v) * pi/H; "
                                      "ray = (cos e cos a, cos e sin a, sin e) from station centre",
               "stations": st_doc, "cameras": cams}
        (out / "cameras.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"записано: {out / 'cameras.json'}; снимков {len(cams)}, стоянок {len(groups)}")
    return doc
