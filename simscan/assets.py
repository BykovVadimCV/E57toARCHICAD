"""Загрузка реальных 3D-моделей для сцен.

Два вида каталогов:

  scan_dir/<роль>/*.obj   - отсканированные предметы в натуральную величину (метры, Z вверх).
                            Ставятся как есть: обувь у входа, вещи на столах, сумки на полу.
                            Источник - Google Scanned Objects (1033 предмета, CC BY-SA 4.0).
  asset_dir/<вид>/*.obj   - модели мебели, вписываемые в габарит предмета (оси glTF, Y вверх).
                            Источники - Poly Haven (CC0) и Objaverse (фильтр по лицензии).

Каждая загрузка дописывает ATTRIBUTION.csv в корень каталога: CC BY / CC BY-SA требуют
указывать авторство, а BY-SA - распространять производные модели на тех же условиях
(обучение на облаках точек из них производной моделью обычно не считается, но сами
упрощённые OBJ - считаются; не выкладывайте их без файла атрибуции).

Сетки упрощаются до max_triangles (рейкасту лишние треугольники только мешают).
"""
from __future__ import annotations

import csv
import io
import json
import tarfile
import tempfile
import urllib.request
from pathlib import Path

import numpy as np

GSO_ROOT = "https://storage.googleapis.com/kubric-public/assets/GSO"
GSO_LICENSE = "CC BY-SA 4.0"

# Категории GSO -> роль в сцене. Категория "None" (216 предметов) не размечена - пропускаем.
GSO_ROLES = {
    "Shoe": "shoes",
    "Bag": "floor",
    "Toys": "floor",
    "Legos": "floor",
    "Stuffed Toys": "floor",
    "Consumer Goods": "tabletop",
    "Bottles and Cans and Cups": "tabletop",
    "Media Cases": "tabletop",
    "Board Games": "tabletop",
    "Action Figures": "tabletop",
    "Keyboard": "tabletop",
    "Mouse": "tabletop",
    "Headphones": "tabletop",
    "Camera": "tabletop",
    "Hat": "tabletop",
}

# Poly Haven: категории моделей -> вид предмета в asset_dir
POLYHAVEN_KINDS = {
    "seating": "chair", "sofa": "sofa", "table": "table", "bed": "bed",
    "shelves": "shelf", "storage": "wardrobe", "cabinet": "wardrobe", "lighting": "lamp",
}

# Objaverse: запросы по названию -> вид предмета; допустимые лицензии
OBJAVERSE_QUERIES = {
    "sofa": ("sofa", "couch"), "chair": ("chair",), "table": ("table",), "bed": ("bed",),
    "wardrobe": ("wardrobe", "cabinet"), "shelf": ("bookshelf", "shelf"),
}
OBJAVERSE_LICENSES = ("cc0", "by", "by-sa")


def _get(url: str, timeout: float = 120.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "simscan/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def simplify(v: np.ndarray, t: np.ndarray, max_triangles: int):
    """Упростить сетку (квадрики Гарланда) и убрать вырожденные треугольники."""
    import open3d as o3d

    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3iVector(t))
    m.remove_duplicated_vertices()
    m.remove_degenerate_triangles()
    if len(m.triangles) > max_triangles:
        m = m.simplify_quadric_decimation(max_triangles)
    m.remove_unreferenced_vertices()
    return np.asarray(m.vertices), np.asarray(m.triangles)


def write_obj(path: Path, v: np.ndarray, t: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    np.savetxt(buf, v, fmt="v %.5f %.5f %.5f")
    np.savetxt(buf, t + 1, fmt="f %d %d %d")
    path.write_text(buf.getvalue())


def read_obj(data: bytes):
    """Только вершины и треугольники (многоугольники - веером)."""
    vs, fs = [], []
    for line in data.decode("utf-8", "replace").splitlines():
        if line.startswith("v "):
            vs.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "):
            idx = [int(p.split("/")[0]) for p in line.split()[1:]]
            idx = [i - 1 if i > 0 else len(vs) + i for i in idx]
            fs += [[idx[0], idx[k], idx[k + 1]] for k in range(1, len(idx) - 1)]
    return np.asarray(vs, float), np.asarray(fs, np.int64)


class Attribution:
    FIELDS = ("file", "source", "id", "title", "author", "license", "url")

    def __init__(self, root: Path):
        self.path = Path(root) / "ATTRIBUTION.csv"
        self.rows = {}
        if self.path.exists():
            with self.path.open(newline="") as f:
                self.rows = {r["file"]: r for r in csv.DictReader(f)}

    def add(self, **row) -> None:
        self.rows[row["file"]] = {k: row.get(k, "") for k in self.FIELDS}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as f:
            w = csv.DictWriter(f, self.FIELDS)
            w.writeheader()
            w.writerows(sorted(self.rows.values(), key=lambda r: r["file"]))


# ----------------------------------------------------------------------
# Google Scanned Objects (бакет Kubric: те же сетки, что в оригинальном датасете Google)
# ----------------------------------------------------------------------

def gso_manifest(cache: Path | None = None) -> dict:
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text())["assets"]
    data = _get(f"{GSO_ROOT}/GSO.json")
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_bytes(data)
    return json.loads(data)["assets"]


def _gso_mesh(asset_id: str):
    """Скачать архив предмета и вернуть visual_geometry.obj (архив не распаковывается на диск)."""
    data = _get(f"{GSO_ROOT}/{asset_id}.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.isfile() and Path(member.name).name == "visual_geometry.obj":
                return read_obj(tar.extractfile(member).read())
    raise FileNotFoundError(f"{asset_id}: нет visual_geometry.obj")


def fetch_gso(out_dir, per_role: int = 20, roles=None, max_triangles: int = 3000, seed: int = 0,
              manifest_cache=None, log=print) -> dict:
    """Скачать по per_role предметов на роль в out_dir/<роль>/<id>.obj. Возвращает счётчики."""
    out = Path(out_dir)
    assets = gso_manifest(manifest_cache)
    by_role: dict[str, list[str]] = {}
    for aid, a in sorted(assets.items()):
        role = GSO_ROLES.get(a.get("metadata", {}).get("category"))
        if role and (roles is None or role in roles):
            by_role.setdefault(role, []).append(aid)
    rng = np.random.default_rng(seed)
    attr = Attribution(out)
    counts = {}
    for role, ids in sorted(by_role.items()):
        pick = rng.permutation(ids)[:per_role]
        n = 0
        for aid in pick:
            path = out / role / f"{aid}.obj"
            if not path.exists():
                try:
                    v, t = _gso_mesh(aid)
                except Exception as e:                       # noqa: BLE001 - сеть, битый архив
                    log(f"  {aid}: {e}")
                    continue
                v, t = simplify(v, t, max_triangles)
                write_obj(path, v, t)
            a = assets[aid]
            attr.add(file=str(path.relative_to(out)), source="Google Scanned Objects", id=aid,
                     title=a["metadata"].get("description", "").splitlines()[0] if
                     a["metadata"].get("description") else aid,
                     author="Google LLC", license=a.get("license", GSO_LICENSE),
                     url=f"{GSO_ROOT}/{aid}.tar.gz")
            n += 1
        counts[role] = n
        log(f"{role}: {n} предметов")
    attr.save()
    return counts


# ----------------------------------------------------------------------
# Poly Haven (CC0). Не проверено в среде разработки: сайт закрыт прокси.
# ----------------------------------------------------------------------

def fetch_polyhaven(out_dir, max_per_kind: int = 10, max_triangles: int = 6000, log=print) -> dict:
    """Модели мебели Poly Haven -> out_dir/<вид>/<id>.obj (оси glTF, Y вверх)."""
    import open3d as o3d

    out = Path(out_dir)
    models = json.loads(_get("https://api.polyhaven.com/assets?t=models"))
    attr = Attribution(out)
    counts: dict[str, int] = {}
    for mid, info in sorted(models.items()):
        cats = set(info.get("categories", []))
        kind = next((k for c, k in POLYHAVEN_KINDS.items() if c in cats), None)
        if kind is None or counts.get(kind, 0) >= max_per_kind:
            continue
        files = json.loads(_get(f"https://api.polyhaven.com/files/{mid}"))
        gltf = files.get("gltf", {}).get("1k", {}).get("gltf")
        if not gltf:
            continue
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / Path(gltf["url"]).name
            main.write_bytes(_get(gltf["url"]))
            for rel, inc in gltf.get("include", {}).items():
                if rel.endswith(".bin"):                       # текстуры не нужны
                    p = Path(tmp) / rel
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(_get(inc["url"]))
            m = o3d.io.read_triangle_mesh(str(main))
        v, t = np.asarray(m.vertices), np.asarray(m.triangles)
        if len(t) == 0:
            continue
        v, t = simplify(v, t, max_triangles)
        path = out / kind / f"{mid}.obj"
        write_obj(path, v, t)
        attr.add(file=str(path.relative_to(out)), source="Poly Haven", id=mid,
                 title=info.get("name", mid), author=", ".join(info.get("authors", {})),
                 license="CC0", url=f"https://polyhaven.com/a/{mid}")
        counts[kind] = counts.get(kind, 0) + 1
        log(f"{kind}: {mid}")
    attr.save()
    return counts


# ----------------------------------------------------------------------
# Objaverse (нужен pip install objaverse). Не проверено: HuggingFace закрыт прокси.
# ----------------------------------------------------------------------

def fetch_objaverse(out_dir, max_per_kind: int = 20, max_triangles: int = 6000,
                    licenses=OBJAVERSE_LICENSES, log=print) -> dict:
    """Мебель из Objaverse по названию и лицензии -> out_dir/<вид>/<uid>.obj (Y вверх).

    Масштаб у Objaverse произвольный - модели вписываются в габарит предмета, поэтому
    годятся только для asset_dir. Качество разное: отбирайте глазами (simscan render)."""
    import objaverse
    import open3d as o3d

    out = Path(out_dir)
    ann = objaverse.load_annotations()
    attr = Attribution(out)
    counts: dict[str, int] = {}
    for kind, words in OBJAVERSE_QUERIES.items():
        uids = [u for u, a in ann.items()
                if a.get("license") in licenses
                and any(w in (a.get("name") or "").lower() for w in words)
                and 500 < (a.get("faceCount") or 0) < 200_000][:max_per_kind * 3]
        paths = objaverse.load_objects(uids=uids)
        for uid, glb in paths.items():
            if counts.get(kind, 0) >= max_per_kind:
                break
            m = o3d.io.read_triangle_mesh(glb)
            v, t = np.asarray(m.vertices), np.asarray(m.triangles)
            if len(t) == 0:
                continue
            v, t = simplify(v, t, max_triangles)
            path = out / kind / f"{uid}.obj"
            write_obj(path, v, t)
            a = ann[uid]
            attr.add(file=str(path.relative_to(out)), source="Objaverse", id=uid,
                     title=a.get("name", ""), author=(a.get("user") or {}).get("username", ""),
                     license=a.get("license", ""), url=a.get("viewerUrl", ""))
            counts[kind] = counts.get(kind, 0) + 1
    attr.save()
    for k, n in counts.items():
        log(f"{k}: {n}")
    return counts
