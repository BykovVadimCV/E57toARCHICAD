"""Procedural raycastable clutter for synthetic indoor scan generation.

Each shape is built from small oriented solids and cylinders in metric coordinates,
not raster decorations. All pieces remain in the clutter semantic class.
"""
from __future__ import annotations

import math
import numpy as np
from .layout import Box

KINDS = ("moving_box", "suitcase", "bucket", "vacuum", "folded_ladder",
         "laundry_basket", "stool", "paint_can")


def _box(x, y, z, w, d, h, yaw):
    return Box((float(x), float(y), float(z)), (float(w), float(d), float(h)), float(yaw))


def _local(base, du, dv, z, w, d, h, angle=0.0):
    c, s = math.cos(base.yaw), math.sin(base.yaw)
    x = base.center[0] + du*c - dv*s
    y = base.center[1] + du*s + dv*c
    return _box(x, y, z, w, d, h, base.yaw + angle)


def make(kind: str, x: float, y: float, size: float, rng, yaw: float = 0.0):
    """Return (boxes, cylindrical primitives, footprint radius).

    Coordinates are metric and ground-referenced. The footprint radius includes
    any projecting handles or rods. No external assets or downloads are required.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown clutter kind: {kind}")
    s = float(size)
    base = _box(x, y, 0.0, s, s, s, yaw)
    boxes, prims = [], []
    cyl = lambda du, dv, z, r, h: prims.append({
        "type": "cylinder",
        "center": list(_local(base, du, dv, z, 0.01, 0.01, 0.01).center),
        "radius": float(r), "height": float(h)})
    if kind == "moving_box":
        h = s * float(rng.uniform(.65, 1.2))
        boxes.append(_local(base, 0, 0, h/2, s, s*.78, h))
        # Flat open carton flaps (not a solid closed cube).
        if rng.random() < .6:
            for v in (-1, 1):
                boxes.append(_local(base, 0, v*s*.47, h, s*.96, s*.22, .012, v*.32))
    elif kind == "suitcase":
        h=s*1.25
        boxes.append(_local(base, 0, 0, h/2+.04, s*.78, s*.42, h))
        for u in (-.26, .26):
            cyl(u*s, 0, .045, s*.075, .09)
        boxes.append(_local(base, 0, 0, h+.12, s*.24, .028, .025))
        for u in (-.105,.105):
            boxes.append(_local(base, u*s, 0, h+.065, .023, .023, .14))
    elif kind == "bucket" or kind == "paint_can":
        r=s*(.37 if kind=="bucket" else .32)
        h=s*(.85 if kind=="bucket" else .7)
        cyl(0, 0, h/2, r, h)
        if kind == "paint_can":
            cyl(0, 0, h+.018, r*.98, .036)
        for u in (-r*.9,r*.9):
            boxes.append(_local(base, u, 0, h+.12, .019, .022, .20))
        boxes.append(_local(base, 0, 0, h+.22, r*1.8, .02, .022))
    elif kind == "vacuum":
        boxes.append(_local(base, 0, 0, s*.24, s*.72, s*.42, s*.36))
        for u in (-.26,.26):
            cyl(u*s, -s*.18, s*.09, s*.075, s*.18)
        boxes.append(_local(base, s*.19, 0, s*.63, .045, .045, s*.87,
                            float(rng.uniform(-.18,.18))))
        boxes.append(_local(base, s*.30, 0, s*1.10, s*.27, .043, .038))
    elif kind == "folded_ladder":
        h=s*2.4
        for u in (-s*.31,s*.31):
            boxes.append(_local(base, u, 0, h*.5, .035, .048, h, .15))
        for z in np.arange(s*.35, h, s*.30):
            boxes.append(_local(base, 0, 0, float(z), s*.65, .055, .038))
        boxes.append(_local(base, 0, -.04, h*.50, s*.58, .045, h*.95, -.13))
    elif kind == "laundry_basket":
        h=s*.65
        # The interior is empty, unlike a cuboid approximation.
        boxes.append(_local(base, 0, 0, .018, s*.95, s*.7, .036))
        for v in (-1,1):
            boxes.append(_local(base, 0, v*s*.34, h/2, s*.95, .035, h))
        for u in (-1,1):
            boxes.append(_local(base, u*s*.46, 0, h/2, .035, s*.65, h))
    elif kind == "stool":
        h=s*.94
        boxes.append(_local(base, 0, 0, h, s*.78, s*.7, .06))
        for u in (-.29,.29):
            for v in (-.26,.26):
                boxes.append(_local(base, u*s, v*s, h/2, .055, .055, h))
    elif kind == "paint_can":
        raise AssertionError("unreachable")
    return boxes, prims, s*.85
