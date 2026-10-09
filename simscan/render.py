"""«Архитектурный» рендер планировки: стены, окна, двери с дугами, подписи и площади.

Для глаз (проверить, что генератор строит правдоподобные квартиры), не для обучения.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .layout import Box, Layout, Opening, Wall

FILL = {"room": (245, 236, 214), "kitchen": (250, 240, 190), "bath": (210, 232, 245),
        "corridor": (232, 232, 232), "balcony": (220, 238, 214)}
WALL = {"exterior": (25, 25, 25), "interior": (70, 70, 70), "parapet": (120, 120, 120)}
FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")


def _font(size: int):
    from PIL import ImageFont

    for f in FONTS:
        if Path(f).exists():
            return ImageFont.truetype(f, size)
    try:
        import matplotlib

        return ImageFont.truetype(str(Path(matplotlib.__file__).parent /
                                      "mpl-data/fonts/ttf/DejaVuSans.ttf"), size)
    except Exception:
        return ImageFont.load_default()


class _Canvas:
    def __init__(self, layout: Layout, px_per_m: float, margin_m: float, ss: int):
        from PIL import Image, ImageDraw

        x0, y0, x1, y1 = layout.bbox()
        self.k = px_per_m * ss
        self.ox, self.oy = x0 - margin_m, y1 + margin_m
        w = int((x1 - x0 + 2 * margin_m) * self.k)
        h = int((y1 - y0 + 2 * margin_m) * self.k)
        self.img = Image.new("RGB", (w, h), (255, 255, 255))
        self.d = ImageDraw.Draw(self.img)
        self.ss = ss

    def p(self, x, y):
        return ((x - self.ox) * self.k, (self.oy - y) * self.k)

    def poly(self, pts, fill=None, outline=None, width=1):
        self.d.polygon([self.p(*q) for q in pts], fill=fill, outline=outline,
                       width=max(1, int(width * self.ss)) if outline else 0)

    def line(self, a, b, fill, width=1.0):
        self.d.line([self.p(*a), self.p(*b)], fill=fill, width=max(1, int(round(width * self.ss))))


def _rect(wall: Wall, s0, s1, o0, o1):
    return [tuple(wall.point(s, o)) for s, o in ((s0, o0), (s1, o0), (s1, o1), (s0, o1))]


def _footprint(prim):
    """Выпуклая оболочка проекции сетки ниже 1,4 м (или None)."""
    from .meshes import prim_mesh

    v = prim_mesh(prim)[0]
    v = v[v[:, 2] < 1.4][:, :2]
    if len(v) < 3:
        return None
    pts = sorted(set(map(tuple, np.round(v, 3))))
    if len(pts) < 3:
        return None

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and ((h[-1][0] - h[-2][0]) * (p[1] - h[-2][1])
                                   - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0])) <= 0:
                h.pop()
            h.append(p)
        return h[:-1]

    return half(pts) + half(pts[::-1])


def _box_pts(b: Box):
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    hx, hy = b.size[0] / 2, b.size[1] / 2
    return [(b.center[0] + u * c - v * s, b.center[1] + u * s + v * c)
            for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]


def _label_point(room):
    rects = room.rects()
    a = max(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
    return (a[0] + a[2]) / 2, (a[1] + a[3]) / 2


def render_plan(layout: Layout, out_png: str | Path, px_per_m: float = 60.0,
                furniture: bool = True, title: str | None = None) -> None:
    ss = 3
    cv = _Canvas(layout, px_per_m, 1.2, ss)
    k = cv.k

    # помещения
    for r in layout.rooms:
        for x0, y0, x1, y1 in r.rects():
            cv.poly([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], fill=FILL.get(r.kind, (240, 240, 240)))
    # мебель - тонким контуром
    if furniture:
        for it in layout.items:
            if it.label not in ("furniture", "clutter", "fixture", "curtain") or \
                    it.kind in ("baseboard", "lamp", "soffit"):
                continue
            for b in sorted(it.boxes, key=lambda b: b.center[2]):
                if b.center[2] - b.size[2] / 2 > 1.4:           # навесное - не рисуем
                    continue
                cv.poly(_box_pts(b), fill=(255, 255, 255), outline=(150, 150, 150), width=1)
            for pr in it.prims:
                if pr["type"] == "cylinder" and pr["center"][2] - pr["height"] / 2 < 1.4:
                    x, y = cv.p(pr["center"][0], pr["center"][1])
                    r = pr["radius"] * cv.k
                    cv.d.ellipse([x - r, y - r, x + r, y + r], fill=(255, 255, 255),
                                 outline=(120, 120, 120), width=cv.ss)
                elif pr["type"] == "curtain":
                    _curtain(cv, pr, (150, 90, 190) if it.kind == "curtains" else (190, 160, 215))
                elif pr["type"] == "mesh" and not pr.get("native"):
                    b = Box(tuple(pr["center"]), tuple(pr["size"]), pr.get("yaw", 0.0))
                    cv.poly(_box_pts(b), fill=(235, 240, 250), outline=(90, 110, 160), width=1)
                elif pr["type"] in ("mesh", "trimesh"):                 # сканы и мягкие вещи
                    hull = _footprint(pr)
                    if hull is not None:
                        cv.poly(hull, fill=(250, 238, 225), outline=(200, 120, 70), width=1)
    for it in layout.items:
        if it.label == "column":
            for b in it.boxes:
                cv.poly(_box_pts(b), fill=WALL["exterior"])

    # стены
    for w in layout.walls:
        color = WALL.get(w.kind, (40, 40, 40))
        if w.role in ("party", "bearing"):
            color = (45, 45, 45)
        if w.role == "loggia_glazing":
            color = (110, 110, 110)
        t = w.thickness / 2
        cv.poly(_rect(w, -w.ext0, w.length + w.ext1, -t, t), fill=color)

    # проёмы
    for o in layout.openings:
        wall = layout.wall(o.wall_id)
        t = wall.thickness / 2
        sa, sb = o.s - o.width / 2, o.s + o.width / 2
        sides = [layout.room_at(*wall.point(o.s, sd * (t + 0.05))) for sd in (1, -1)]
        bg = [FILL.get(layout.rooms[r].kind) if r is not None else (255, 255, 255) for r in sides]
        cv.poly(_rect(wall, sa, sb, -t, 0), fill=bg[1])
        cv.poly(_rect(wall, sa, sb, 0, t), fill=bg[0])
        if o.kind == "window":
            for off in (-t, 0.0, t):
                cv.line(tuple(wall.point(sa, off)), tuple(wall.point(sb, off)), (60, 60, 60), 1)
            for s_ in (sa, sb):
                cv.line(tuple(wall.point(s_, -t)), tuple(wall.point(s_, t)), (60, 60, 60), 1)
        elif o.kind == "door":
            _door(cv, wall, o)
        else:                                                      # арка
            for off in (-t, t):
                cv.line(tuple(wall.point(sa, off)), tuple(wall.point(sb, off)), (140, 140, 140), 1)

    # подписи
    f_name, f_area = _font(int(0.22 * k)), _font(int(0.2 * k))
    for r in layout.rooms:
        x, y = _label_point(r)
        name = r.name or {"room": "Комната", "kitchen": "Кухня", "bath": "Санузел",
                          "corridor": "Коридор", "balcony": "Балкон"}.get(r.kind, r.kind)
        area = f"{r.area:.1f}".replace(".", ",") + " м²"
        px, py = cv.p(x, y)
        cv.d.text((px, py - 0.14 * k), name, fill=(30, 30, 30), font=f_name, anchor="mm")
        cv.d.text((px, py + 0.16 * k), area, fill=(80, 80, 80), font=f_area, anchor="mm")

    # габаритные размеры (мм) и масштабная линейка
    x0, y0, x1, y1 = layout.bbox()
    f_dim = _font(int(0.2 * k))
    dim_c = (90, 90, 90)
    yb = y0 - 0.45
    cv.line((x0, yb), (x1, yb), dim_c, 1)
    for x in (x0, x1):
        cv.line((x, yb - 0.1), (x, yb + 0.1), dim_c, 1)
    cv.d.text(cv.p((x0 + x1) / 2, yb - 0.25), f"{round((x1 - x0) * 1000):d}", fill=dim_c,
              font=f_dim, anchor="mm")
    xl = x0 - 0.45
    cv.line((xl, y0), (xl, y1), dim_c, 1)
    for y in (y0, y1):
        cv.line((xl - 0.1, y), (xl + 0.1, y), dim_c, 1)
    _vtext(cv, f"{round((y1 - y0) * 1000):d}", cv.p(xl - 0.25, (y0 + y1) / 2), f_dim, dim_c)
    sx, sy = x1 - 1.0, y0 - 0.9
    cv.line((sx, sy), (sx + 1.0, sy), (0, 0, 0), 2)
    cv.d.text(cv.p(sx + 0.5, sy - 0.18), "1 м", fill=(0, 0, 0), font=f_dim, anchor="mm")

    total = sum(r.area for r in layout.rooms if r.kind != "balcony")
    head = title or _title(layout, total)
    cv.d.text(cv.p(x0, y1 + 0.75), head, fill=(0, 0, 0), font=_font(int(0.26 * k)), anchor="lm")

    from PIL import Image

    img = cv.img.resize((cv.img.width // ss, cv.img.height // ss), Image.LANCZOS)
    img.save(out_png)


def _vtext(cv, text, xy, font, fill):
    from PIL import Image, ImageDraw

    l, t, r, b = font.getbbox(text)
    tmp = Image.new("RGBA", (r - l + 4, b - t + 4), (255, 255, 255, 0))
    ImageDraw.Draw(tmp).text((2 - l, 2 - t), text, font=font, fill=fill)
    tmp = tmp.rotate(90, expand=True)
    cv.img.paste(tmp, (int(xy[0] - tmp.width / 2), int(xy[1] - tmp.height / 2)), tmp)


def _title(layout: Layout, total: float) -> str:
    m = layout.meta
    names = {"studio": "Студия", "1k": "1-комнатная", "2k": "2-комнатная", "3k": "3-комнатная",
             "4k": "4-комнатная"}
    shapes = {"single": "один фасад", "through": "сквозная", "corner": "угловая"}
    if m.get("source") == "apartment":
        parts = [names.get(m.get("program"), m.get("program")), shapes.get(m.get("shape"), "")]
    else:
        parts = [m.get("source", "simscan")]
    parts.append(f"общая {total:.1f}".replace(".", ",") + " м²")
    return ", ".join(p for p in parts if p)


def _curtain(cv: _Canvas, pr: dict, color) -> None:
    p0, p1 = np.asarray(pr["p0"], float), np.asarray(pr["p1"], float)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-6:
        return
    u = (p1 - p0) / L
    n = np.array([-u[1], u[0]])
    s = np.linspace(0, L, max(8, int(L / (pr["period"] / 6))))
    pts = p0[None] + s[:, None] * u + (pr["amp"] * np.sin(2 * math.pi * s / pr["period"]))[:, None] * n
    cv.d.line([cv.p(*q) for q in pts], fill=color, width=max(1, int(1.5 * cv.ss)))


def _door(cv: _Canvas, wall: Wall, o: Opening) -> None:
    """Полотно в положении 90 градусов и дуга открывания - как на чертеже."""
    t = wall.thickness / 2
    hinge_s = o.s + o.hinge * o.width / 2
    hinge = wall.point(hinge_s, o.swing * t)
    closed = -o.hinge * wall.u * o.width
    opened = o.swing * wall.n * o.width
    cv.line(tuple(hinge), tuple(hinge + opened), (40, 40, 40), 2)
    hx, hy = cv.p(*hinge)
    r = o.width * cv.k

    def ang(v):                               # угол в координатах картинки (y вниз)
        return math.degrees(math.atan2(-v[1], v[0])) % 360

    a1, a2 = ang(closed), ang(opened)
    start, end = (a1, a2) if (a2 - a1) % 360 <= 180 else (a2, a1)
    cv.d.arc([hx - r, hy - r, hx + r, hy + r], start, end, fill=(120, 120, 120),
             width=max(1, cv.ss))


def render_many(layouts: list, out_png: str | Path, cols: int = 3, px_per_m: float = 45.0) -> None:
    """Несколько планов на одном листе."""
    import tempfile

    from PIL import Image

    tiles = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, lay in enumerate(layouts):
            p = Path(tmp) / f"{k}.png"
            render_plan(lay, p, px_per_m)
            tiles.append(Image.open(p).copy())
    w = max(t.width for t in tiles)
    h = max(t.height for t in tiles)
    rows = math.ceil(len(tiles) / cols)
    sheet = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * w + (w - t.width) // 2, (k // cols) * h + (h - t.height) // 2))
    sheet.save(out_png)


__all__ = ["render_plan", "render_many"]
