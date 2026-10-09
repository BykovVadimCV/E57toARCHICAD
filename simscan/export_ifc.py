"""Экспорт векторного плана в IFC для ArchiCAD.

    python -m simscan export-ifc vectors/plan.json --out plan.ifc

Создаёт IFC 4 файл с:
  - IfcBuilding, IfcBuildingStorey (этажи)
  - IfcWall для каждой стены (IfcMaterialLayerSetUsage для толщины)
  - IfcDoor, IfcWindow для проёмов
  - IfcSpace для помещений с площадями
  - 2D геометрия (IfcArbitraryClosedProfileDef, IfcPolyline)
  - Координатная система проекта

ArchiCAD читает IFC4 начиная с версии 21. Файл совместим с:
  - ArchiCAD 21+
  - Revit 2020+ (с IFC импортом)
  - BricsCAD BIM
  - Открытыми просмотрщиками (BIMVision, Solibri Anywhere)
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


# ----------------------------------------------------------------------
# IFC генератор (без ifcopenshell - чистый текстовый формат)
# ----------------------------------------------------------------------

class IFCWriter:
    """Минималистичный генератор IFC 4 ADD2 TC1."""

    def __init__(self):
        self.entities: list[str] = []
        self.next_id = 1
        self.refs: dict[str, int] = {}

    def new_id(self, label: str | None = None) -> int:
        """Выделить новый #ID."""
        eid = self.next_id
        self.next_id += 1
        if label:
            self.refs[label] = eid
        return eid

    def ref(self, label: str) -> str:
        """Ссылка на ранее созданную сущность."""
        return f"#{self.refs[label]}"

    def add(self, entity_type: str, *args, label: str | None = None) -> int:
        """Добавить сущность IFC."""
        eid = self.new_id(label)
        formatted_args = []
        for arg in args:
            formatted_args.append(self._format_value(arg))
        line = f"#{eid}={entity_type}({','.join(formatted_args)});"
        self.entities.append(line)
        return eid

    def _format_value(self, val: Any) -> str:
        """Форматирование значения для IFC."""
        if val is None or val == "$":
            return "$"
        elif isinstance(val, str):
            if val.startswith("#"):
                return val  # ссылка
            return f"'{val}'"
        elif isinstance(val, bool):
            return ".T." if val else ".F."
        elif isinstance(val, (int, float)):
            if isinstance(val, float):
                return f"{val:.6f}".rstrip('0').rstrip('.')
            return str(val)
        elif isinstance(val, (list, tuple)):
            if len(val) == 0:
                return "()"
            return f"({','.join(self._format_value(v) for v in val)})"
        elif hasattr(val, "__iter__") and not isinstance(val, str):
            return f"({','.join(self._format_value(v) for v in val)})"
        else:
            return str(val)

    def write(self, path: Path, description: str = "Floor plan from E57 scan") -> None:
        """Записать IFC файл."""
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")

        header = f"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('{description}'),'2;1');
FILE_NAME('{path.name}','{timestamp}',('simscan'),('DotCloudCONVERT'),'','','');
FILE_SCHEMA(('IFC4'));
ENDSEC;

DATA;
"""

        footer = """ENDSEC;
END-ISO-10303-21;
"""

        content = header + "\n".join(self.entities) + "\n" + footer
        path.write_text(content, encoding="utf-8")


# ----------------------------------------------------------------------
# IFC построители
# ----------------------------------------------------------------------

def create_ifc_header(ifc: IFCWriter, project_name: str = "Apartment") -> None:
    """Создать заголовок IFC: проект, контекст, единицы."""

    # Единицы измерения
    ifc.add("IFCSIUNIT", "*", ".LENGTHUNIT.", "$", ".METRE.", label="unit_m")
    ifc.add("IFCSIUNIT", "*", ".AREAUNIT.", "$", ".SQUARE_METRE.", label="unit_m2")
    ifc.add("IFCSIUNIT", "*", ".VOLUMEUNIT.", "$", ".CUBIC_METRE.", label="unit_m3")
    ifc.add("IFCSIUNIT", "*", ".PLANEANGLEUNIT.", "$", ".RADIAN.", label="unit_rad")

    units_id = ifc.add("IFCUNITASSIGNMENT", [ifc.ref("unit_m"), ifc.ref("unit_m2"),
                                              ifc.ref("unit_m3"), ifc.ref("unit_rad")],
                       label="units")

    # Контекст геометрии
    ifc.add("IFCAXIS2PLACEMENT3D", ifc.ref("origin"), "$", "$", label="world_placement")

    ifc.add("IFCGEOMETRICREPRESENTATIONCONTEXT", "$", "Model", 3, 1.0e-5,
            ifc.ref("world_placement"), "$", label="geom_context")

    ifc.add("IFCGEOMETRICREPRESENTATIONSUBCONTEXT", "Axis", "Model", "*", "*", "*",
            ifc.ref("geom_context"), "$", ".GRAPH_VIEW.", "$", label="axis_context")

    ifc.add("IFCGEOMETRICREPRESENTATIONSUBCONTEXT", "Body", "Model", "*", "*", "*",
            ifc.ref("geom_context"), "$", ".MODEL_VIEW.", "$", label="body_context")

    # Проект
    ifc.add("IFCPROJECT", _guid(), ifc.ref("owner_history"), project_name,
            project_name + " floor plan", "$", "$", "$",
            [ifc.ref("geom_context")], ifc.ref("units"), label="project")

    # Локальное размещение для здания
    ifc.add("IFCLOCALPLACEMENT", "$", ifc.ref("world_placement"), label="building_placement")

    # Здание
    ifc.add("IFCBUILDING", _guid(), ifc.ref("owner_history"), "Building",
            "Residential building", "$", ifc.ref("building_placement"), "$",
            ".ELEMENT.", "$", "$", "$", label="building")

    ifc.add("IFCRELAGGREGATES", _guid(), ifc.ref("owner_history"), "BuildingContainer",
            "$", ifc.ref("project"), [ifc.ref("building")])


def create_origin(ifc: IFCWriter) -> None:
    """Создать начало координат."""
    ifc.add("IFCCARTESIANPOINT", [0.0, 0.0, 0.0], label="origin")
    ifc.add("IFCDIRECTION", [0.0, 0.0, 1.0], label="axis_z")
    ifc.add("IFCDIRECTION", [1.0, 0.0, 0.0], label="axis_x")
    ifc.add("IFCDIRECTION", [0.0, 1.0, 0.0], label="axis_y")


def create_owner_history(ifc: IFCWriter) -> None:
    """Создать OwnerHistory."""
    ifc.add("IFCPERSON", "$", "DotCloudCONVERT", "$", "$", "$", "$", "$", label="person")
    ifc.add("IFCORGANIZATION", "$", "simscan", "$", "$", "$", label="organization")
    ifc.add("IFCPERSONANDORGANIZATION", ifc.ref("person"), ifc.ref("organization"), "$",
            label="person_org")
    ifc.add("IFCAPPLICATION", ifc.ref("organization"), "0.1.0", "simscan", "DotCloudCONVERT",
            label="application")
    ifc.add("IFCOWNERHISTORY", ifc.ref("person_org"), ifc.ref("application"), "$",
            ".ADDED.", "$", ifc.ref("person_org"), ifc.ref("application"), int(time.time()),
            label="owner_history")


def create_local_placement(ifc: IFCWriter, parent_ref: str | None = None,
                           z_offset: float = 0.0, label: str = "local_placement") -> None:
    """Создать локальную систему координат."""
    if z_offset != 0.0:
        ifc.add("IFCCARTESIANPOINT", [0.0, 0.0, z_offset], label=f"{label}_origin")
        ifc.add("IFCAXIS2PLACEMENT3D", ifc.ref(f"{label}_origin"), "$", "$",
                label=f"{label}_axis")
    else:
        ifc.add("IFCAXIS2PLACEMENT3D", ifc.ref("origin"), "$", "$",
                label=f"{label}_axis")

    parent = ifc.ref(parent_ref) if parent_ref else "$"
    ifc.add("IFCLOCALPLACEMENT", parent, ifc.ref(f"{label}_axis"), label=label)


def create_storey(ifc: IFCWriter, name: str, elevation: float, label: str) -> None:
    """Создать этаж."""
    create_local_placement(ifc, "local_placement", elevation, f"{label}_placement")

    ifc.add("IFCBUILDINGSTOREY", _guid(), ifc.ref("owner_history"), name,
            f"Floor level at {elevation:.2f}m", "$", ifc.ref(f"{label}_placement"),
            "$", ".ELEMENT.", elevation, label=label)

    ifc.add("IFCRELAGGREGATES", _guid(), ifc.ref("owner_history"),
            f"{name}Container", "$", ifc.ref("building"), [ifc.ref(label)])


# ----------------------------------------------------------------------
# Экспорт стен, проёмов, помещений
# ----------------------------------------------------------------------

def export_wall(ifc: IFCWriter, wall: dict, storey_ref: str, idx: int) -> None:
    """Экспорт стены в IFC."""
    wall_label = f"wall_{idx}"
    points = wall["points"]
    thickness = wall["thickness"]

    if len(points) < 2:
        return

    # Геометрия: полилиния как ось стены
    pt_refs = []
    for i, (x, y) in enumerate(points):
        pt_label = f"{wall_label}_pt{i}"
        ifc.add("IFCCARTESIANPOINT", [x, y], label=pt_label)
        pt_refs.append(ifc.ref(pt_label))

    ifc.add("IFCPOLYLINE", pt_refs, label=f"{wall_label}_axis_geom")

    # Представление оси
    ifc.add("IFCSHAPEREPRESENTATION", ifc.ref("axis_context"), "Axis", "Curve2D",
            [ifc.ref(f"{wall_label}_axis_geom")], label=f"{wall_label}_axis_rep")

    # Материал стены
    ifc.add("IFCMATERIAL", "Concrete", "$", "$", label=f"{wall_label}_material")
    ifc.add("IFCMATERIALLAYER", ifc.ref(f"{wall_label}_material"), thickness, "$",
            label=f"{wall_label}_mat_layer")
    ifc.add("IFCMATERIALLAYERSET", [ifc.ref(f"{wall_label}_mat_layer")],
            f"Wall {thickness*1000:.0f}mm", label=f"{wall_label}_mat_layerset")

    # Профиль стены (прямоугольник толщины по оси)
    ifc.add("IFCRECTANGLEPROFILEDEF", ".AREA.", f"Wall_{idx}", "$",
            thickness, sum(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1)),
            label=f"{wall_label}_profile")

    # Выдавливание вдоль оси
    ifc.add("IFCDIRECTION", [0.0, 0.0, 1.0], label=f"{wall_label}_extrude_dir")
    ifc.add("IFCEXTRUDEDAREASOLID", ifc.ref(f"{wall_label}_profile"), "$",
            ifc.ref(f"{wall_label}_extrude_dir"), 3.0, label=f"{wall_label}_body_geom")

    # Представление тела
    ifc.add("IFCSHAPEREPRESENTATION", ifc.ref("body_context"), "Body", "SweptSolid",
            [ifc.ref(f"{wall_label}_body_geom")], label=f"{wall_label}_body_rep")

    # Определение продукта
    ifc.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$",
            [ifc.ref(f"{wall_label}_axis_rep"), ifc.ref(f"{wall_label}_body_rep")],
            label=f"{wall_label}_shape")

    # Размещение стены
    create_local_placement(ifc, f"{storey_ref}_placement", 0.0, f"{wall_label}_placement")

    # Стена
    is_external = wall.get("is_outer", False)
    ifc.add("IFCWALL", _guid(), ifc.ref("owner_history"), wall.get("id", f"Wall {idx}"),
            f"Length {wall.get('length', 0):.2f}m", "$", ifc.ref(f"{wall_label}_placement"),
            ifc.ref(f"{wall_label}_shape"), "$", ".STANDARD.", label=wall_label)

    # Связь с этажом
    ifc.add("IFCRELCONTAINEDINSPATIALSTRUCTURE", _guid(), ifc.ref("owner_history"),
            f"{wall_label}_containment", "$", [ifc.ref(wall_label)], ifc.ref(storey_ref))

    # Связь с материалом
    ifc.add("IFCRELASSOCIATESMATERIAL", _guid(), ifc.ref("owner_history"),
            f"{wall_label}_material_rel", "$", [ifc.ref(wall_label)],
            ifc.ref(f"{wall_label}_mat_layerset"))


def export_opening(ifc: IFCWriter, opening: dict, storey_ref: str, idx: int) -> None:
    """Экспорт проёма (дверь/окно) в IFC."""
    opening_type = opening["opening_type"]
    opening_label = f"{opening_type}_{idx}"

    a = opening["a"]
    b = opening["b"]
    width = opening["width"]
    height = opening.get("head", 2.1) - opening.get("sill", 0.0) if opening_type == "window" else 2.1
    sill = opening.get("sill", 0.0)

    # Размещение
    x, y = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    ifc.add("IFCCARTESIANPOINT", [x, y, sill], label=f"{opening_label}_origin")
    ifc.add("IFCAXIS2PLACEMENT3D", ifc.ref(f"{opening_label}_origin"), "$", "$",
            label=f"{opening_label}_axis")
    ifc.add("IFCLOCALPLACEMENT", ifc.ref(f"{storey_ref}_placement"),
            ifc.ref(f"{opening_label}_axis"), label=f"{opening_label}_placement")

    # Профиль (прямоугольник)
    ifc.add("IFCRECTANGLEPROFILEDEF", ".AREA.", opening_label, "$",
            width, height, label=f"{opening_label}_profile")

    # Выдавливание
    ifc.add("IFCDIRECTION", [0.0, 0.0, 1.0], label=f"{opening_label}_dir")
    ifc.add("IFCEXTRUDEDAREASOLID", ifc.ref(f"{opening_label}_profile"), "$",
            ifc.ref(f"{opening_label}_dir"), 0.1, label=f"{opening_label}_geom")

    # Представление
    ifc.add("IFCSHAPEREPRESENTATION", ifc.ref("body_context"), "Body", "SweptSolid",
            [ifc.ref(f"{opening_label}_geom")], label=f"{opening_label}_rep")
    ifc.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", [ifc.ref(f"{opening_label}_rep")],
            label=f"{opening_label}_shape")

    # Элемент
    if opening_type == "door":
        entity_type = "IFCDOOR"
    elif opening_type == "window":
        entity_type = "IFCWINDOW"
    else:
        entity_type = "IFCOPENINGELEMENT"

    ifc.add(entity_type, _guid(), ifc.ref("owner_history"), opening.get("id", opening_label),
            f"{width:.2f}m x {height:.2f}m", "$", ifc.ref(f"{opening_label}_placement"),
            ifc.ref(f"{opening_label}_shape"), "$",
            height if opening_type in ["door", "window"] else "$",
            width if opening_type in ["door", "window"] else "$",
            label=opening_label)

    # Связь с этажом
    ifc.add("IFCRELCONTAINEDINSPATIALSTRUCTURE", _guid(), ifc.ref("owner_history"),
            f"{opening_label}_containment", "$", [ifc.ref(opening_label)], ifc.ref(storey_ref))


def export_room(ifc: IFCWriter, room: dict, storey_ref: str, idx: int) -> None:
    """Экспорт помещения (IfcSpace) в IFC."""
    room_label = f"space_{idx}"
    points = room["points"]
    area = room["area"]

    if len(points) < 3:
        return

    # Геометрия: замкнутый контур
    pt_refs = []
    for i, (x, y) in enumerate(points):
        pt_label = f"{room_label}_pt{i}"
        ifc.add("IFCCARTESIANPOINT", [x, y], label=pt_label)
        pt_refs.append(ifc.ref(pt_label))

    # Замыкаем контур
    pt_refs.append(pt_refs[0])

    ifc.add("IFCPOLYLINE", pt_refs, label=f"{room_label}_boundary")
    ifc.add("IFCARBITRARYCLOSEDPROFILEDEF", ".AREA.", room.get("name", f"Room {idx}"),
            ifc.ref(f"{room_label}_boundary"), label=f"{room_label}_profile")

    # Выдавливание на высоту этажа
    ifc.add("IFCDIRECTION", [0.0, 0.0, 1.0], label=f"{room_label}_dir")
    ifc.add("IFCEXTRUDEDAREASOLID", ifc.ref(f"{room_label}_profile"), "$",
            ifc.ref(f"{room_label}_dir"), 3.0, label=f"{room_label}_geom")

    # Представление
    ifc.add("IFCSHAPEREPRESENTATION", ifc.ref("body_context"), "Body", "SweptSolid",
            [ifc.ref(f"{room_label}_geom")], label=f"{room_label}_rep")
    ifc.add("IFCPRODUCTDEFINITIONSHAPE", "$", "$", [ifc.ref(f"{room_label}_rep")],
            label=f"{room_label}_shape")

    # Размещение
    create_local_placement(ifc, f"{storey_ref}_placement", 0.0, f"{room_label}_placement")

    # Помещение
    ifc.add("IFCSPACE", _guid(), ifc.ref("owner_history"), room.get("name", f"Room {idx}"),
            f"Area: {area:.2f} m²", "$", ifc.ref(f"{room_label}_placement"),
            ifc.ref(f"{room_label}_shape"), "$", ".INTERNAL.", "$", "$", "$",
            label=room_label)

    # Связь с этажом
    ifc.add("IFCRELCONTAINEDINSPATIALSTRUCTURE", _guid(), ifc.ref("owner_history"),
            f"{room_label}_containment", "$", [ifc.ref(room_label)], ifc.ref(storey_ref))

    # Свойства площади
    ifc.add("IFCPROPERTYSINGLEVALUE", "NetFloorArea", "$",
            ifc.add("IFCAREAMEASURE", area), "$", label=f"{room_label}_area_prop")

    ifc.add("IFCPROPERTYSET", _guid(), ifc.ref("owner_history"), "Pset_SpaceCommon",
            "$", [ifc.ref(f"{room_label}_area_prop")], label=f"{room_label}_pset")

    ifc.add("IFCRELDEFINESBYPROPERTIES", _guid(), ifc.ref("owner_history"),
            f"{room_label}_props", "$", [ifc.ref(room_label)],
            ifc.ref(f"{room_label}_pset"))


# ----------------------------------------------------------------------
# Главная функция экспорта
# ----------------------------------------------------------------------

def export_to_ifc(plan_json: Path, output_ifc: Path, log=print) -> None:
    """Экспорт векторного плана из JSON в IFC."""
    log(f"Загрузка плана из {plan_json}...")
    plan_data = json.loads(plan_json.read_text(encoding="utf-8"))

    ifc = IFCWriter()

    log("Создание структуры IFC...")

    # Базовые элементы
    create_origin(ifc)
    create_owner_history(ifc)
    create_ifc_header(ifc, project_name="Apartment Scan")
    create_local_placement(ifc)

    # Этаж
    floor_z = plan_data.get("floor_z", 0.0)
    storey_label = f"storey_L{plan_data.get('level', 1)}"
    create_storey(ifc, f"Level {plan_data.get('level', 1)}", floor_z, storey_label)

    # Стены
    log(f"Экспорт стен: {len(plan_data.get('walls', []))}...")
    for idx, wall in enumerate(plan_data.get("walls", [])):
        export_wall(ifc, wall, storey_label, idx)

    # Проёмы
    log(f"Экспорт проёмов: {len(plan_data.get('openings', []))}...")
    for idx, opening in enumerate(plan_data.get("openings", [])):
        export_opening(ifc, opening, storey_label, idx)

    # Помещения
    log(f"Экспорт помещений: {len(plan_data.get('rooms', []))}...")
    for idx, room in enumerate(plan_data.get("rooms", [])):
        export_room(ifc, room, storey_label, idx)

    # Запись файла
    log(f"Запись IFC в {output_ifc}...")
    ifc.write(output_ifc)
    log(f"Готово. Экспортировано {len(ifc.entities)} сущностей.")


def _guid() -> str:
    """Генерация IFC GUID (base64 кодирование UUID)."""
    uid = uuid.uuid4()
    b = uid.bytes
    # IFC использует специальную base64 кодировку
    chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
    result = []
    for i in range(0, 16, 3):
        chunk = (b[i] << 16) | (b[i+1] << 8) | b[i+2] if i+2 < 16 else (b[i] << 16) | (b[i+1] << 8) if i+1 < 16 else b[i] << 16
        for _ in range(4):
            result.append(chars[chunk & 0x3F])
            chunk >>= 6
    return ''.join(result[:22])


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main_export_ifc(plan_json, output_ifc, log=print):
    """CLI для экспорта в IFC."""
    export_to_ifc(Path(plan_json), Path(output_ifc), log=log)
