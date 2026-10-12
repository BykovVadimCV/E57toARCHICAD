"""export-ifc: валидный IFC4, стены на своих местах, проёмы вырезают стены, высоты из плана."""
import math

import numpy as np
import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402
import ifcopenshell.guid  # noqa: E402
import ifcopenshell.validate  # noqa: E402

from simscan.export_ifc import export_plans  # noqa: E402

T = 0.2


def _plan(level="L1", floor_z=0.0):
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    return {
        "level": level, "floor_z": floor_z, "ceiling_z": floor_z + 2.7,
        "walls": [
            {"id": "w0", "points": [[0, 0], [4, 0], [4, 3]], "thickness": T, "is_outer": True},
            {"id": "w1", "points": [[0, 3], [3 * c, 3 + 3 * s]], "thickness": 0.12},   # 30°, не по осям
        ],
        "openings": [
            {"id": "d0", "opening_type": "door", "wall_id": "w0", "a": [1.0, 0], "b": [1.9, 0],
             "width": 0.9, "sill": 0.0, "head": 2.3},
            {"id": "win0", "opening_type": "window", "wall_id": "w0", "a": [4, 1.0], "b": [4, 2.2],
             "width": 1.2, "sill": 0.8, "head": 2.0},
        ],
        "rooms": [{"id": "r0", "name": "Room 1", "points": [[0, 0], [4, 0], [4, 3], [0, 3]], "area": 12.0}],
    }


def _shapes(f, storey="L1"):
    import ifcopenshell.util.element as ue

    st = ifcopenshell.geom.settings()
    st.set("use-world-coords", True)
    out = {}
    for el in f.by_type("IfcWall") + f.by_type("IfcDoor") + f.by_type("IfcWindow"):
        if ue.get_container(el).Name != storey:
            continue
        body = next(r for r in el.Representation.Representations if r.RepresentationIdentifier == "Body")
        sh = ifcopenshell.geom.create_shape(st, el, body)
        v = np.array(sh.geometry.verts).reshape(-1, 3)
        tri = np.array(sh.geometry.faces).reshape(-1, 3)
        a, b, c = v[tri[:, 0]], v[tri[:, 1]], v[tri[:, 2]]
        vol = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6
        out[el.Name] = (v.min(0), v.max(0), vol)
    return out


@pytest.fixture(scope="module")
def ifc(tmp_path_factory):
    path = tmp_path_factory.mktemp("ifc") / "plan.ifc"
    export_plans([_plan(), _plan("L2", 3.0)], path, log=lambda *_: None)
    return ifcopenshell.open(str(path)), path


def test_schema_valid_and_guids(ifc):
    f, path = ifc
    logger = ifcopenshell.validate.json_logger()
    ifcopenshell.validate.validate(f, logger, express_rules=True)
    assert not logger.statements, logger.statements[:5]
    for e in f.by_type("IfcRoot"):
        assert e.GlobalId[0] in "0123" and len(ifcopenshell.guid.expand(e.GlobalId)) == 32
    assert "'.AREA.'" not in path.read_text()
    units = {u.UnitType: (u.Prefix, u.Name) for u in f.by_type("IfcSIUnit")}
    assert units["LENGTHUNIT"] == (None, "METRE")


def test_walls_in_place_and_cut(ifc):
    f, _ = ifc
    assert [s.Elevation for s in f.by_type("IfcBuildingStorey")] == [0.0, 3.0]
    sh = _shapes(f)
    lo, hi, vol = sh["w0.0"]                                   # (0,0)-(4,0), дверь 0,9 x 2,3
    assert np.allclose(lo, [0, -T / 2, 0], atol=1e-6) and np.allclose(hi, [4, T / 2, 2.7], atol=1e-6)
    assert vol == pytest.approx(4 * T * 2.7 - 0.9 * T * 2.3, rel=1e-4)
    lo, hi, vol = sh["w0.1"]                                   # (4,0)-(4,3), окно 1,2 x (0,8-2,0)
    assert np.allclose(lo[:2], [4 - T / 2, 0], atol=1e-6) and np.allclose(hi[:2], [4 + T / 2, 3], atol=1e-6)
    assert vol == pytest.approx(3 * T * 2.7 - 1.2 * T * 1.2, rel=1e-4)
    lo, hi, _ = sh["w1"]                                       # косая стена под 30°
    assert hi[0] == pytest.approx(3 * math.cos(math.radians(30)) + 0.06 * math.sin(math.radians(30)), abs=1e-6)
    assert hi[1] > 4.4
    lo, hi, _ = sh["d0"]
    assert lo[2] == pytest.approx(0.0, abs=1e-6) and hi[2] == pytest.approx(2.3, abs=1e-6)
    lo, hi, _ = sh["win0"]
    assert lo[2] == pytest.approx(0.8, abs=1e-6) and hi[2] == pytest.approx(2.0, abs=1e-6)
    lo, hi, _ = _shapes(f, "L2")["w0.0"]                      # второй этаж на отметке 3 м
    assert lo[2] == pytest.approx(3.0, abs=1e-6) and hi[2] == pytest.approx(5.7, abs=1e-6)


def test_relations(ifc):
    f, _ = ifc
    voids = f.by_type("IfcRelVoidsElement")
    fills = f.by_type("IfcRelFillsElement")
    assert len(voids) == len(fills) == 4                       # 2 проёма x 2 этажа
    assert {v.RelatingBuildingElement.Name for v in voids} == {"w0.0", "w0.1"}
    assert {r.RelatedBuildingElement.is_a() for r in fills} == {"IfcDoor", "IfcWindow"}
    usage = f.by_type("IfcMaterialLayerSetUsage")
    assert all(u.OffsetFromReferenceLine == pytest.approx(-u.ForLayerSet.MaterialLayers[0].LayerThickness / 2)
               for u in usage)
    qto = [q for q in f.by_type("IfcElementQuantity") if q.Name == "Qto_SpaceBaseQuantities"]
    area = {q.Name: q for q in qto[0].Quantities}["NetFloorArea"]
    assert len(qto) == 2 and area.AreaValue == pytest.approx(12.0)
