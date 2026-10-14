"""План через разбиение на ячейки и совместную разметку: две комнаты, дверь, окно."""
import math

import numpy as np
import pytest

from simscan.cellplan import cellplan
from simscan.e57io import write_merged_e57

from test_floorplan import _apartment


def test_cellplan_rooms_door_window(tmp_path):
    p = _apartment()
    a = math.radians(12)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    stations = np.array([[2.0, 2.0, 1.1], [6.0, 2.0, 1.1]]) @ R.T + [3.0, -2.0, 10.0]
    path = tmp_path / "apt.e57"
    write_merged_e57(path, p, np.full(len(p), 0.5, np.float32), stations=list(stations))
    res = cellplan(path, tmp_path / "out", px=0.02, log=lambda *a: None)
    lv = res["levels"][0]
    areas = sorted(r["area"] for r in lv["rooms"])
    assert len(areas) == 2 and all(abs(x - 15.8) < 1.2 for x in areas), areas
    doors = [o for o in lv["openings"] if o["opening_type"] == "door"]
    windows = [o for o in lv["openings"] if o["opening_type"] == "window"]
    assert len(doors) == 1 and abs(doors[0]["width"] - 0.9) < 0.15 and abs(doors[0]["head"] - 2.1) < 0.15, doors
    # окно правилами не находится стабильно (подход отложен в пользу сети) - не проверяется
    assert len(windows) <= 1, windows
    for f in ("plan.png", "cells.png", "features.png", "walls.png", "elements.json", "features.npz"):
        assert (tmp_path / "out" / f).exists(), f


def _scan(tmp_path):
    p = _apartment()
    a = math.radians(12)
    R = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    stations = np.array([[2.0, 2.0, 1.1], [6.0, 2.0, 1.1]]) @ R.T + [3.0, -2.0, 10.0]
    path = tmp_path / "apt.e57"
    write_merged_e57(path, p, np.full(len(p), 0.5, np.float32), stations=list(stations))
    return path


def test_cellplan_with_net_finds_window_and_stops_leak(tmp_path):
    """Вероятности сети: окно найдено, помещение не протекает наружу через окно, стены - оси
    с толщиной, план уходит в IFC с вырезанными проёмами."""
    import json

    from netfake import write_pred

    path = _scan(tmp_path)
    pred = write_pred(tmp_path / "pred")
    res = cellplan(path, tmp_path / "out", px=0.02, pred=pred, log=lambda *a: None)
    lv = res["levels"][0]
    assert lv["net"]
    areas = sorted(r["area"] for r in lv["rooms"])
    assert len(areas) == 2 and all(abs(x - 15.8) < 0.8 for x in areas), areas
    plan = json.loads((tmp_path / "out" / "plan_L1.json").read_text(encoding="utf-8"))
    ops = {o["opening_type"]: o for o in plan["openings"]}
    assert len(plan["openings"]) == 2, plan["openings"]
    assert abs(ops["window"]["width"] - 1.4) < 0.15 and abs(ops["window"]["sill"] - 0.9) < 0.15
    assert abs(ops["door"]["width"] - 0.9) < 0.15 and abs(ops["door"]["head"] - 2.1) < 0.15
    inner = [w for w in plan["walls"] if not w["is_outer"]]
    outer = [w for w in plan["walls"] if w["is_outer"]]
    assert len(inner) == 1 and len(inner[0]["rooms"]) == 2 and abs(inner[0]["thickness"] - 0.1) < 0.08, inner
    assert ops["door"]["wall_id"] == inner[0]["id"]
    # наружный контур - 4 оси, сходящиеся в углах (стена, разрезанная перегородкой, - одна);
    # 8 x 4 м по внутренним граням плюс толщина
    assert len(outer) == 4, outer
    ends = [tuple(p) for w in outer for p in w["points"]]
    assert all(sum(math.dist(e, q) < 0.01 for q in ends) == 2 for e in ends), ends
    xs, ys = [e[0] for e in ends], [e[1] for e in ends]
    assert abs((max(xs) - min(xs)) - 8.1) < 0.2 and abs((max(ys) - min(ys)) - 4.1) < 0.2, (xs, ys)
    # перегородка доведена до осей наружных стен
    assert {round(p[1], 3) for p in inner[0]["points"]} == {round(min(ys), 3), round(max(ys), 3)}

    ifcopenshell = pytest.importorskip("ifcopenshell")
    from simscan.export_ifc import export_to_ifc

    export_to_ifc(tmp_path / "out" / "plan_L1.json", tmp_path / "apt.ifc", log=lambda *a: None)
    f = ifcopenshell.open(str(tmp_path / "apt.ifc"))
    assert len(f.by_type("IfcWall")) == len(plan["walls"])
    assert len(f.by_type("IfcRelFillsElement")) == 2 and len(f.by_type("IfcSpace")) == 2


def test_cellplan_net_level_mismatch_falls_back_to_rules(tmp_path):
    from netfake import write_pred

    path = _scan(tmp_path)
    pred = write_pred(tmp_path / "pred", floor_z=20.0, ceiling_z=22.8)       # другой уровень
    res = cellplan(path, tmp_path / "out", px=0.02, pred=pred, log=lambda *a: None)
    assert not res["levels"][0]["net"] and len(res["levels"][0]["rooms"]) == 2
