"""Сцена: планировка -> треугольная сетка с метками класса, экземпляра и отражательной способности.

Вся геометрия - ориентированные параллелепипеды. Стена режется проёмами на куски:
простенки на всю высоту, под окном - до подоконника, над проёмом - перемычка.
Пересечения тел на стыках стен допустимы: лучу видна только внешняя оболочка.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .labels import LABEL_ID
from .layout import Box, Layout, Opening, Wall

# Вершины единичного параллелепипеда: бит 0 - x, бит 1 - y, бит 2 - z.
_CORNERS = np.array([[(i & 1) * 2 - 1, ((i >> 1) & 1) * 2 - 1, ((i >> 2) & 1) * 2 - 1]
                     for i in range(8)], float) * 0.5
# Грани с внешней нормалью (обход против часовой стрелки снаружи).
_TRIS = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]], np.uint32)

SLAB_M = 0.2  # толщина перекрытий в сцене


@dataclass
class Solid:
    """Тело с семантикой; общий вход и для сетки, и для эталонных масок.

    Обычно - параллелепипед box. Если задан mesh, геометрия берётся из него,
    а box - только габарит (в эталон такие тела не попадают)."""
    box: Box
    label: str
    instance: int
    reflectance: float
    source: str = ""          # wall:<id>, opening:<id>:leaf, item:<id>:<вид>, slab:<k>
    mesh: tuple | None = None # (vertices, triangles)
    transmit: float = 0.0     # доля лучей насквозь (тюль); стекло задаётся отдельно
    deform: bool = False      # архитектура: разбивается на сетку и получает неровность


@dataclass
class SceneMesh:
    vertices: np.ndarray                      # (N, 3) float32
    triangles: np.ndarray                     # (M, 3) uint32
    tri_label: np.ndarray                     # (M,) uint8
    tri_instance: np.ndarray                  # (M,) int32
    tri_reflectance: np.ndarray               # (M,) float32
    instances: list[dict] = field(default_factory=list)
    tri_transmit: np.ndarray | None = None    # (M,) float32
    vert_deform: np.ndarray | None = None     # (N,) bool - вершины архитектуры

    def __post_init__(self):
        if self.tri_transmit is None:
            self.tri_transmit = np.zeros(len(self.triangles), np.float32)
        if self.vert_deform is None:
            self.vert_deform = np.zeros(len(self.vertices), bool)

    @classmethod
    def from_solids(cls, solids: list[Solid], tessellation: float | None = None) -> "SceneMesh":
        """tessellation - шаг сетки для тел с deform=True (None - без разбиения)."""
        plain = [s for s in solids if s.mesh is None and not (tessellation and s.deform)]
        other = [s for s in solids if not (s.mesh is None and not (tessellation and s.deform))]
        out = cls._from_boxes(plain)
        for s in other:
            v, t = s.mesh if s.mesh is not None else _box_mesh(s.box, tessellation)
            n = len(t)
            part = cls(np.asarray(v, np.float32), np.asarray(t, np.uint32),
                       np.full(n, LABEL_ID[s.label], np.uint8), np.full(n, s.instance, np.int32),
                       np.full(n, s.reflectance, np.float32),
                       [{"instance": s.instance, "label": s.label, "source": s.source}],
                       np.full(n, s.transmit, np.float32),
                       np.full(len(v), bool(s.deform and s.mesh is None), bool))
            out = out.concatenate(part)
        return out

    @classmethod
    def _from_boxes(cls, solids: list[Solid]) -> "SceneMesh":
        if not solids:
            return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32),
                       np.zeros(0, np.uint8), np.zeros(0, np.int32), np.zeros(0, np.float32))
        centers = np.array([s.box.center for s in solids], float)
        sizes = np.array([s.box.size for s in solids], float)
        yaws = np.array([s.box.yaw for s in solids], float)
        local = _CORNERS[None] * sizes[:, None, :]                      # (K, 8, 3)
        c, s_ = np.cos(yaws)[:, None], np.sin(yaws)[:, None]
        x = local[..., 0] * c - local[..., 1] * s_
        y = local[..., 0] * s_ + local[..., 1] * c
        verts = np.stack([x, y, local[..., 2]], -1) + centers[:, None, :]
        k = len(solids)
        tris = (_TRIS[None] + (np.arange(k, dtype=np.uint32) * 8)[:, None, None]).reshape(-1, 3)
        rep = np.repeat
        return cls(
            verts.reshape(-1, 3).astype(np.float32), tris,
            rep(np.array([LABEL_ID[s.label] for s in solids], np.uint8), 12),
            rep(np.array([s.instance for s in solids], np.int32), 12),
            rep(np.array([s.reflectance for s in solids], np.float32), 12),
            [{"instance": s.instance, "label": s.label, "source": s.source} for s in solids],
            rep(np.array([s.transmit for s in solids], np.float32), 12),
            rep(np.array([s.deform for s in solids], bool), 8),
        )

    def concatenate(self, other: "SceneMesh") -> "SceneMesh":
        return SceneMesh(
            np.vstack([self.vertices, other.vertices]),
            np.vstack([self.triangles, other.triangles + np.uint32(len(self.vertices))]),
            np.concatenate([self.tri_label, other.tri_label]),
            np.concatenate([self.tri_instance, other.tri_instance]),
            np.concatenate([self.tri_reflectance, other.tri_reflectance]),
            self.instances + other.instances,
            np.concatenate([self.tri_transmit, other.tri_transmit]),
            np.concatenate([self.vert_deform, other.vert_deform]),
        )

    def to_raycasting_scene(self):
        import open3d as o3d

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.core.Tensor(np.ascontiguousarray(self.vertices)),
                            o3d.core.Tensor(np.ascontiguousarray(self.triangles)))
        return scene

    def save_ply(self, path) -> None:
        """Сетка с цветом по классу - для просмотра в CloudCompare / MeshLab."""
        palette = _palette()
        colors = palette[self.tri_label]
        n = len(self.triangles)
        verts = self.vertices[self.triangles.reshape(-1)]
        vcol = np.repeat(colors, 3, axis=0)
        with open(path, "wb") as f:
            header = ("ply\nformat binary_little_endian 1.0\n"
                      f"element vertex {3 * n}\nproperty float x\nproperty float y\nproperty float z\n"
                      "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                      f"element face {n}\nproperty list uchar int vertex_indices\nend_header\n")
            f.write(header.encode())
            vdt = np.dtype([("p", "<f4", 3), ("c", "u1", 3)])
            vbuf = np.empty(3 * n, vdt)
            vbuf["p"], vbuf["c"] = verts, vcol
            f.write(vbuf.tobytes())
            fdt = np.dtype([("n", "u1"), ("i", "<i4", 3)])
            fbuf = np.empty(n, fdt)
            fbuf["n"], fbuf["i"] = 3, np.arange(3 * n, dtype=np.int32).reshape(-1, 3)
            f.write(fbuf.tobytes())


def _box_mesh(box, cell):
    from .meshes import box_mesh

    return box_mesh(box, cell)


def _palette() -> np.ndarray:
    p = np.full((256, 3), 128, np.uint8)
    p[13] = [170, 110, 200]
    p[:13] = [[0, 0, 0], [150, 120, 90], [220, 220, 220], [200, 60, 60], [160, 40, 40],
              [60, 160, 60], [60, 60, 200], [120, 200, 255], [255, 0, 255], [230, 160, 40],
              [255, 230, 80], [0, 200, 200], [90, 90, 90]]
    return p


# ======================================================================

class SceneBuilder:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.solids: list[Solid] = []
        self._next_instance = 1

    def _new_instance(self) -> int:
        k = self._next_instance
        self._next_instance += 1
        return k

    def _add(self, box: Box, label: str, refl: float, source: str, instance: int | None = None,
             deform: bool = False):
        inst = self._new_instance() if instance is None else instance
        self.solids.append(Solid(box, label, inst, float(refl), source, deform=deform))
        return inst

    def build(self) -> list[Solid]:
        lay = self.layout
        mat = lay.materials or {}
        for w in lay.walls:
            self._wall(w, mat.get("wall", 0.7))
        for o in lay.openings:
            wall = lay.wall(o.wall_id)
            if o.kind == "door":
                self._door_leaf(wall, o, mat.get("door", 0.5))
            elif o.kind == "window":
                self._window(wall, o, mat)
        self._slabs(mat)
        from .meshes import prim_mesh

        for it in lay.items:
            inst = self._new_instance()
            src = f"item:{it.id}:{it.kind}"
            for b in it.boxes:
                self._add(b, it.label, it.reflectance, src, inst, deform=it.label == "column")
                self.solids[-1].transmit = it.transmit
            for prim in it.prims:
                v, t = prim_mesh(prim)
                if len(t) == 0:
                    continue
                lo, hi = v.min(0), v.max(0)
                bbox = Box(tuple((lo + hi) / 2), tuple(hi - lo), 0.0)
                self.solids.append(Solid(bbox, it.label, inst, float(it.reflectance), src,
                                         mesh=(v, t), transmit=it.transmit))
        return self.solids

    # --- стены -------------------------------------------------------------
    def wall_pieces(self, wall: Wall) -> list[tuple[float, float, float, float]]:
        """Куски тела стены (s_a, s_b, z_a, z_b) после вырезания проёмов."""
        H = wall.height if wall.height is not None else self.layout.ceiling_height
        ops = sorted(self.layout.openings_on(wall.id), key=lambda o: o.s)
        pieces = []
        cur = -wall.ext0
        for o in ops:
            sa, sb = o.s - o.width / 2, o.s + o.width / 2
            if sa - cur > 1e-6:
                pieces.append((cur, sa, 0.0, H))
            if o.z0 > 1e-6:
                pieces.append((sa, sb, 0.0, o.z0))
            if o.z1 < H - 1e-6:
                pieces.append((sa, sb, o.z1, H))
            cur = max(cur, sb)
        end = wall.length + wall.ext1
        if end - cur > 1e-6:
            pieces.append((cur, end, 0.0, H))
        return pieces

    def _wall(self, wall: Wall, refl: float) -> None:
        inst = self._new_instance()
        for sa, sb, za, zb in self.wall_pieces(wall):
            c = wall.point((sa + sb) / 2)
            box = Box((c[0], c[1], (za + zb) / 2), (sb - sa, wall.thickness, zb - za), wall.yaw)
            self._add(box, "wall", refl, f"wall:{wall.id}", inst, deform=True)
        if wall.kind == "exterior":
            # торцы перекрытий: закрывают щель между плитой и наружной стеной
            H = self.layout.ceiling_height
            sa, sb = -wall.ext0, wall.length + wall.ext1
            c = wall.point((sa + sb) / 2)
            for za, zb in ((-SLAB_M, 0.0), (H, H + SLAB_M)):
                box = Box((c[0], c[1], (za + zb) / 2), (sb - sa, wall.thickness, zb - za), wall.yaw)
                self._add(box, "wall", refl, f"wall:{wall.id}", inst, deform=True)

    def _door_leaf(self, wall: Wall, o: Opening, refl: float) -> None:
        lt = o.leaf_thickness
        leaf_w = o.width - 0.01
        leaf_h = o.z1 - 0.01
        # петля - у грани стены со стороны открывания, внутри толщины стены
        off = o.swing * (wall.thickness / 2 - lt / 2)
        hinge_s = o.s + o.hinge * (o.width / 2 - 0.005)
        hinge = wall.point(hinge_s, off)
        dir0 = -o.hinge * wall.u                       # от петли к другому косяку
        th = math.radians(o.angle_deg)
        d = math.cos(th) * dir0 + math.sin(th) * o.swing * wall.n
        c = hinge + d * leaf_w / 2
        yaw = math.atan2(d[1], d[0])
        box = Box((c[0], c[1], 0.005 + leaf_h / 2), (leaf_w, lt, leaf_h), yaw)
        self._add(box, "door", refl, f"opening:{o.id}:leaf")

    def _window(self, wall: Wall, o: Opening, mat: dict) -> None:
        t = wall.thickness
        out = wall.outward or 1
        g = out * (t / 2 - min(0.12, 0.35 * t))         # плоскость остекления
        fw, fd = 0.05, 0.07                              # ширина и глубина профиля
        sa, sb = o.s - o.width / 2, o.s + o.width / 2
        frame = self._new_instance()

        def piece(s0, s1, off0, off1, z0, z1, label, inst):
            c = wall.point((s0 + s1) / 2, (off0 + off1) / 2)
            box = Box((c[0], c[1], (z0 + z1) / 2), (s1 - s0, abs(off1 - off0), z1 - z0), wall.yaw)
            self._add(box, label, mat.get("window" if label == "window" else "glass", 0.5),
                      f"opening:{o.id}", inst)

        fa, fb = g - fd / 2, g + fd / 2
        piece(sa, sa + fw, fa, fb, o.z0, o.z1, "window", frame)
        piece(sb - fw, sb, fa, fb, o.z0, o.z1, "window", frame)
        piece(sa + fw, sb - fw, fa, fb, o.z0, o.z0 + fw, "window", frame)
        piece(sa + fw, sb - fw, fa, fb, o.z1 - fw, o.z1, "window", frame)
        panes = [(sa + fw, sb - fw)]
        if o.width > 1.1:                                # импост
            m = o.s
            piece(m - fw / 2, m + fw / 2, fa, fb, o.z0 + fw, o.z1 - fw, "window", frame)
            panes = [(sa + fw, m - fw / 2), (m + fw / 2, sb - fw)]
        glass = self._new_instance()
        for p0, p1 in panes:
            piece(p0, p1, g - 0.004, g + 0.004, o.z0 + fw, o.z1 - fw, "glass", glass)
        self._sill(wall, o, g, fd, out, frame, mat)

    def _sill(self, wall, o, g, fd, out, inst, mat) -> None:
        """Подоконная доска: от рамы до комнатной грани стены плюс свес 6 см."""
        inner_face = -out * wall.thickness / 2
        frame_inner = g - out * fd / 2
        a, b = sorted((frame_inner, inner_face - out * 0.06))
        c = wall.point(o.s, (a + b) / 2)
        box = Box((c[0], c[1], o.z0 - 0.015), (o.width + 0.1, b - a, 0.03), wall.yaw)
        self._add(box, "window", mat.get("window", 0.8), f"opening:{o.id}", inst)

    # --- перекрытия --------------------------------------------------------
    def _slabs(self, mat: dict) -> None:
        """Пол и перекрытие. layout.meta["slab"] (двухуровневые квартиры, duplex.py):
        {"floor"|"ceiling": {"holes": [прямоугольники], "thickness": м}} - проём и толщина слоя."""
        H = self.layout.ceiling_height
        cfg = self.layout.meta.get("slab", {})
        floor_i, ceil_i = self._new_instance(), self._new_instance()
        for part, label, inst, refl in (("floor", "floor", floor_i, mat.get("floor", 0.4)),
                                        ("ceiling", "ceiling", ceil_i, mat.get("ceiling", 0.8))):
            c = cfg.get(part, {})
            t = float(c.get("thickness", SLAB_M))
            rects = list(self.layout.slab_rects())
            for h in c.get("holes", []):
                from .duplex import subtract_rect

                rects = [q for r in rects for q in subtract_rect(r, h)]
            zc = -t / 2 if part == "floor" else H + t / 2
            for k, (x0, y0, x1, y1) in enumerate(rects):
                self._add(Box(((x0 + x1) / 2, (y0 + y1) / 2, zc), (x1 - x0, y1 - y0, t)), label, refl,
                          f"slab:{k}", inst, deform=True)
        for r in self.layout.rooms:
            if r.ceiling_z < H - 1e-6:
                inst = self._new_instance()
                for a, b, c, d in r.rects():
                    box = Box(((a + c) / 2, (b + d) / 2, r.ceiling_z + 0.01), (c - a, d - b, 0.02))
                    self._add(box, "fixture", mat.get("ceiling", 0.8), f"suspended:{r.id}", inst)


def build_scene(layout: Layout, tessellation: float | None = None) -> tuple[SceneMesh, list[Solid]]:
    solids = SceneBuilder(layout).build()
    return SceneMesh.from_solids(solids, tessellation), solids


def surface_distance(points: np.ndarray, solids: list[Solid], chunk: int = 4096) -> np.ndarray:
    """Точное расстояние от точек до ближайшей грани параллелепипедов сцены.

    Для проверок на уровне миллиметров: RaycastingScene.compute_distance в Open3D 0.20
    на длинных тонких треугольниках (плинтус 8 м x 7 см) ошибается до нескольких мм.
    """
    pts = np.asarray(points, float)
    solids = [s for s in solids if s.mesh is None]
    c = np.array([s.box.center for s in solids])                 # (K, 3)
    h = np.array([s.box.size for s in solids]) / 2
    yaw = np.array([s.box.yaw for s in solids])
    cos, sin = np.cos(yaw), np.sin(yaw)
    out = np.empty(len(pts))
    for a in range(0, len(pts), chunk):
        d = pts[a:a + chunk, None, :] - c[None]                     # (n, K, 3)
        u = d[..., 0] * cos + d[..., 1] * sin
        v = -d[..., 0] * sin + d[..., 1] * cos
        q = np.abs(np.stack([u, v, d[..., 2]], -1)) - h
        outside = np.linalg.norm(np.maximum(q, 0), axis=-1)
        inside = np.minimum(q.max(-1), 0)
        out[a:a + chunk] = np.abs(outside + inside).min(1)
    return out
