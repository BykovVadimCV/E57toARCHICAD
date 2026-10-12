"""Экспорт векторного плана в IFC 4 для ArchiCAD (на ifcopenshell).

    python -m simscan export-ifc vectors/L1/plan.json --out plan.ifc
    python -m simscan export-ifc vectors/L1/plan.json vectors/L2/plan.json --out plan.ifc   # этажи

Что пишется:
  - IfcProject / IfcSite / IfcBuilding / IfcBuildingStorey на отметке floor_z уровня;
  - IfcWall на каждый прямой участок оси стены: размещение в начале участка, ось X - вдоль
    участка; представления Axis (ось) и Body (SweptSolid, симметрично оси), материал -
    IfcMaterialLayerSetUsage со смещением -t/2 (ось по середине слоя);
  - проём - IfcOpeningElement, вырезающий стену (IfcRelVoidsElement), и IfcDoor / IfcWindow,
    заполняющие проём (IfcRelFillsElement). Низ и верх проёма - из plan.json (sill, head - от
    пола уровня), по умолчанию дверь 0-2,1 м, окно 0,9-2,1 м, проём во всю высоту - до потолка;
  - IfcSpace с контуром помещения и Qto_SpaceBaseQuantities (NetFloorArea, GrossPerimeter, Height).

Высота стен и помещений - ceiling_z - floor_z уровня. Координаты plan.json (м) пишутся как есть.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

DEFAULT_STOREY_H = 2.7
DOOR_HEAD = 2.1
WINDOW_SILL, WINDOW_HEAD = 0.9, 2.1
MIN_SEGMENT = 0.01          # участки оси короче - вырожденные
MIN_OPENING = 0.2           # проёмы уже - не вырезаются
OPENING_EPS = 0.02          # проём толще стены на столько с каждой стороны - чистый вырез
FILL_THICKNESS = 0.05       # толщина полотна двери / рамы окна


def _new_file(project_name: str):
    import ifcopenshell.api as api

    f = api.run("project.create_file", version="IFC4")
    project = api.run("root.create_entity", f, ifc_class="IfcProject", name=project_name)
    si = lambda raw: {"is_metric": True, "raw": raw}  # noqa: E731  (по умолчанию было бы мм)
    api.run("unit.assign_unit", f, length=si("METERS"), area=si("SQUARE_METERS"), volume=si("CUBIC_METERS"))
    model = api.run("context.add_context", f, context_type="Model")
    ctx = {
        "body": api.run("context.add_context", f, context_type="Model", context_identifier="Body",
                        target_view="MODEL_VIEW", parent=model),
        "axis": api.run("context.add_context", f, context_type="Model", context_identifier="Axis",
                        target_view="GRAPH_VIEW", parent=model),
    }
    site = api.run("root.create_entity", f, ifc_class="IfcSite", name="Site")
    building = api.run("root.create_entity", f, ifc_class="IfcBuilding", name="Building")
    api.run("aggregate.assign_object", f, products=[site], relating_object=project)
    api.run("aggregate.assign_object", f, products=[building], relating_object=site)
    for el in (site, building):
        el.ObjectPlacement = _placement(f, None)
    return f, ctx, building


def _placement(f, rel_to, origin=(0.0, 0.0, 0.0), x_dir=(1.0, 0.0)):
    """IfcLocalPlacement относительно rel_to: начало origin, ось X в плане - x_dir."""
    ax = f.createIfcAxis2Placement3D(
        f.createIfcCartesianPoint([float(v) for v in origin]),
        f.createIfcDirection([0.0, 0.0, 1.0]),
        f.createIfcDirection([float(x_dir[0]), float(x_dir[1]), 0.0]))
    return f.createIfcLocalPlacement(rel_to, ax)


def _box(f, ctx, x0, y0, z0, dx, dy, dz):
    """Тело-параллелепипед [x0, x0+dx] x [y0, y0+dy] x [z0, z0+dz] в СК объекта."""
    prof = f.createIfcRectangleProfileDef(
        "AREA", None,
        f.createIfcAxis2Placement2D(f.createIfcCartesianPoint([x0 + dx / 2, y0 + dy / 2]), None),
        float(dx), float(dy))
    solid = f.createIfcExtrudedAreaSolid(
        prof, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, 0.0, float(z0)]), None, None),
        f.createIfcDirection([0.0, 0.0, 1.0]), float(dz))
    return f.createIfcShapeRepresentation(ctx, "Body", "SweptSolid", [solid])


def _layer_set(f, cache: dict, thickness: float):
    import ifcopenshell.api as api

    key = round(thickness, 3)
    if key not in cache:
        if "material" not in cache:
            cache["material"] = api.run("material.add_material", f, name="Wall")
        ls = api.run("material.add_material_set", f, name=f"Wall {key * 1000:.0f} mm",
                     set_type="IfcMaterialLayerSet")
        layer = api.run("material.add_layer", f, layer_set=ls, material=cache["material"])
        layer.LayerThickness = key
        cache[key] = ls
    return cache[key]


def _segments(points) -> list[tuple[np.ndarray, np.ndarray]]:
    pts = [np.asarray(p, float) for p in points]
    return [(a, b) for a, b in zip(pts, pts[1:]) if np.linalg.norm(b - a) >= MIN_SEGMENT]


def _add_wall(f, ctx, storey, mats, wall: dict, a: np.ndarray, b: np.ndarray, height: float, name: str):
    import ifcopenshell.api as api
    import ifcopenshell.util.element

    t = float(wall["thickness"])
    L = float(np.linalg.norm(b - a))
    u = (b - a) / L
    w = api.run("root.create_entity", f, ifc_class="IfcWall", predefined_type="STANDARD", name=name)
    w.ObjectPlacement = _placement(f, storey.ObjectPlacement, (a[0], a[1], 0.0), u)
    axis = f.createIfcShapeRepresentation(
        ctx["axis"], "Axis", "Curve2D",
        [f.createIfcPolyline([f.createIfcCartesianPoint([0.0, 0.0]), f.createIfcCartesianPoint([L, 0.0])])])
    w.Representation = f.createIfcProductDefinitionShape(None, None, [axis, _box(f, ctx["body"], 0, -t / 2, 0, L, t, height)])
    api.run("material.assign_material", f, products=[w], type="IfcMaterialLayerSetUsage",
            material=_layer_set(f, mats, t))
    ifcopenshell.util.element.get_material(w).OffsetFromReferenceLine = -t / 2
    api.run("spatial.assign_container", f, products=[w], relating_structure=storey)
    pset = api.run("pset.add_pset", f, product=w, name="Pset_WallCommon")
    api.run("pset.edit_pset", f, pset=pset, properties={"IsExternal": bool(wall.get("is_outer", False))})
    return {"wall": w, "a": a, "u": u, "L": L, "t": t}


def _host(segs: list[dict], by_id: dict, op: dict) -> dict | None:
    """Участок стены под проёмом: тот, на который проём ложится наибольшей длиной (участки
    wall_id - первыми). Ближе к середине - при равенстве и если не перекрывает ни один."""
    a, b = np.asarray(op["a"], float), np.asarray(op["b"], float)
    m = (a + b) / 2
    w = max(float(op.get("width") or 0.0), float(np.linalg.norm(b - a)))

    def score(s):
        r = m - s["a"]
        c = float(r @ s["u"])
        d = float(np.linalg.norm(r - c * s["u"]))
        overlap = max(0.0, min(s["L"], c + w / 2) - max(0.0, c - w / 2))
        near = d <= s["t"] / 2 + 0.3
        return (near and overlap >= MIN_OPENING, s in own, overlap if near else 0.0, -d)

    own = by_id.get(op.get("wall_id")) or []
    return max(segs, key=score) if segs else None


def _opening_heights(op: dict, height: float) -> tuple[float, float]:
    kind = op.get("opening_type", "opening")
    sill, head = op.get("sill"), op.get("head")
    if kind == "window":
        sill = WINDOW_SILL if sill is None else sill
        head = WINDOW_HEAD if head is None else head
    else:
        sill = 0.0
        head = (DOOR_HEAD if kind == "door" else height) if head is None else head
    sill = float(np.clip(sill, 0.0, height))
    head = float(np.clip(head, 0.0, height))
    if head - sill < 0.1:                                    # высоты сети вырождены - по умолчанию
        sill, head = (WINDOW_SILL, min(WINDOW_HEAD, height)) if kind == "window" else (0.0, min(DOOR_HEAD, height))
    return sill, head


def _add_opening(f, ctx, storey, seg: dict, op: dict, height: float, log) -> bool:
    import ifcopenshell.api as api

    kind = op.get("opening_type", "opening")
    a, b = np.asarray(op["a"], float), np.asarray(op["b"], float)
    s = sorted(((a - seg["a"]) @ seg["u"], (b - seg["a"]) @ seg["u"]))
    width = float(op.get("width") or 0.0)
    if s[1] - s[0] < MIN_OPENING and width > 0:              # концы не по оси - ширина по центру
        c = (s[0] + s[1]) / 2
        s = [c - width / 2, c + width / 2]
    s0, s1 = max(0.0, s[0]), min(seg["L"], s[1])
    if s1 - s0 < MIN_OPENING:
        log(f"  проём {op.get('id')} пропущен: ширина на стене {max(s1 - s0, 0):.2f} м")
        return False
    w = s1 - s0
    sill, head = _opening_heights(op, height)
    t = seg["t"] + 2 * OPENING_EPS

    opening = api.run("root.create_entity", f, ifc_class="IfcOpeningElement", predefined_type="OPENING",
                      name=f"{op.get('id', kind)} opening")
    opening.ObjectPlacement = _placement(f, seg["wall"].ObjectPlacement, (s0, 0.0, 0.0))
    opening.Representation = f.createIfcProductDefinitionShape(
        None, None, [_box(f, ctx["body"], 0, -t / 2, sill, w, t, head - sill)])
    api.run("feature.add_feature", f, feature=opening, element=seg["wall"])

    if kind not in ("door", "window"):
        return True
    cls = "IfcDoor" if kind == "door" else "IfcWindow"
    fill = api.run("root.create_entity", f, ifc_class=cls, predefined_type=kind.upper(), name=op.get("id", kind))
    fill.ObjectPlacement = _placement(f, opening.ObjectPlacement)
    fill.Representation = f.createIfcProductDefinitionShape(
        None, None, [_box(f, ctx["body"], 0, -FILL_THICKNESS / 2, sill, w, FILL_THICKNESS, head - sill)])
    fill.OverallWidth, fill.OverallHeight = w, head - sill
    api.run("feature.add_filling", f, opening=opening, element=fill)
    api.run("spatial.assign_container", f, products=[fill], relating_structure=storey)
    return True


def _add_space(f, ctx, storey, room: dict, height: float) -> bool:
    import ifcopenshell.api as api

    pts = [tuple(map(float, p)) for p in room["points"]]
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return False
    sp = api.run("root.create_entity", f, ifc_class="IfcSpace", predefined_type="INTERNAL",
                 name=room.get("name") or room.get("id"))
    sp.LongName = room.get("name")
    sp.ObjectPlacement = _placement(f, storey.ObjectPlacement)
    prof = f.createIfcArbitraryClosedProfileDef(
        "AREA", None, f.createIfcPolyline([f.createIfcCartesianPoint(p) for p in pts + pts[:1]]))
    solid = f.createIfcExtrudedAreaSolid(
        prof, f.createIfcAxis2Placement3D(f.createIfcCartesianPoint([0.0, 0.0, 0.0]), None, None),
        f.createIfcDirection([0.0, 0.0, 1.0]), float(height))
    sp.Representation = f.createIfcProductDefinitionShape(
        None, None, [f.createIfcShapeRepresentation(ctx["body"], "Body", "SweptSolid", [solid])])
    api.run("aggregate.assign_object", f, products=[sp], relating_object=storey)
    area = room.get("area")
    if area is None:
        x, y = np.array(pts).T
        area = 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
    perim = room.get("perimeter") or float(sum(math.dist(p, q) for p, q in zip(pts, pts[1:] + pts[:1])))
    qto = api.run("pset.add_qto", f, product=sp, name="Qto_SpaceBaseQuantities")
    api.run("pset.edit_qto", f, qto=qto, properties={"NetFloorArea": float(area), "GrossPerimeter": float(perim),
                                                     "Height": float(height)})
    return True


def export_plans(plans: list[dict], output_ifc: Path, project_name: str = "Apartment scan", log=print):
    """Векторные планы уровней (словари plan.json) -> один IFC, этаж на уровень."""
    import ifcopenshell.api as api

    f, ctx, building = _new_file(project_name)
    mats: dict = {}
    for i, plan in enumerate(plans):
        level = str(plan.get("level", f"L{i + 1}"))
        floor_z = float(plan.get("floor_z") or 0.0)
        ceil_z = plan.get("ceiling_z")
        height = float(ceil_z) - floor_z if ceil_z is not None else DEFAULT_STOREY_H
        if not 1.5 <= height <= 10.0:
            log(f"  {level}: высота {height:.2f} м вне 1,5-10 м, взято {DEFAULT_STOREY_H} м")
            height = DEFAULT_STOREY_H
        storey = api.run("root.create_entity", f, ifc_class="IfcBuildingStorey", name=level)
        storey.Elevation = floor_z
        storey.ObjectPlacement = _placement(f, building.ObjectPlacement, (0.0, 0.0, floor_z))
        api.run("aggregate.assign_object", f, products=[storey], relating_object=building)

        segs, by_id = [], {}
        for wall in plan.get("walls", []):
            parts = _segments(wall.get("points", []))
            for k, (a, b) in enumerate(parts):
                name = wall.get("id", "wall") + (f".{k}" if len(parts) > 1 else "")
                seg = _add_wall(f, ctx, storey, mats, wall, a, b, height, name)
                segs.append(seg)
                by_id.setdefault(wall.get("id"), []).append(seg)
        n_op = 0
        for op in plan.get("openings", []):
            seg = _host(segs, by_id, op)
            if seg is None:
                log(f"  проём {op.get('id')} пропущен: нет стен")
                continue
            n_op += _add_opening(f, ctx, storey, seg, op, height, log)
        n_sp = sum(_add_space(f, ctx, storey, r, height) for r in plan.get("rooms", []))
        log(f"{level}: отметка {floor_z:.3f} м, высота {height:.2f} м; стен {len(segs)}, "
            f"проёмов {n_op} из {len(plan.get('openings', []))}, помещений {n_sp}")
    output_ifc = Path(output_ifc)
    output_ifc.parent.mkdir(parents=True, exist_ok=True)
    f.write(str(output_ifc))
    log(f"Записано: {output_ifc}")
    return f


def export_to_ifc(plan_json, output_ifc, log=print):
    """plan.json (один путь или список) -> IFC."""
    paths = [plan_json] if isinstance(plan_json, (str, Path)) else list(plan_json)
    plans = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    return export_plans(plans, Path(output_ifc), log=log)


def main_export_ifc(plan_json, output_ifc, log=print):
    """CLI для экспорта в IFC."""
    return export_to_ifc(plan_json, output_ifc, log=log)
