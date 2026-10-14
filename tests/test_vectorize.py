"""Векторизация вероятностей сети: прямые - прямые, ось по центру, контур замкнут, дуга -
дуга, проёмы на стенах, шум отфильтрован. Синтетический растр, без сети."""
import json
import math

import cv2
import numpy as np
import pytest

from simscan.vectorize import vectorize

PX = 0.02
ROT = math.radians(17)


def _scene(tmp_path, noisy: bool):
    """Квартира 6 x 4 м, повёрнута на 17°: наружная стена - полоса 10 см внутрь от грани
    (как в разметке датасета), левая стена - дуга наружу (стрела 0,4 м), перегородка 12 см
    с дверью 0,9 м, окно 1,4 м в верхней стене."""
    R = np.array([[math.cos(ROT), -math.sin(ROT)], [math.sin(ROT), math.cos(ROT)]])
    W, H = 460, 400
    o = np.array([1.5, 1.5])

    def to_px(xy):
        w = np.asarray(xy, float) @ R.T + o
        return np.c_[w[:, 0] / PX, H - w[:, 1] / PX]

    sag, L = 0.4, 4.0
    r = (L * L / 4 + sag * sag) / (2 * sag)
    t = np.linspace(-1, 1, 41)
    arc_y = 2.0 + t * 2.0
    arc_x = -(np.sqrt(r * r - (arc_y - 2.0) ** 2) - (r - sag))       # левая стена - дуга наружу (x < 0)
    inner = np.vstack([np.c_[arc_x + 0.05, arc_y][::-1], [[6.0 - 0.05, 4.0 - 0.05], [6.0 - 0.05, 0.05]]])
    inner[0, 1], inner[-1, 1] = 3.95, 0.05
    sem = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sem, [np.round(to_px(inner)).astype(np.int32)], 1)
    band = 5                                                        # 10 см
    outer = np.vstack([np.c_[arc_x, arc_y], [[6.0, 4.0], [6.0, 0.0]]])
    cv2.polylines(sem, [np.round(to_px(np.vstack([outer, outer[:1]])) + 0).astype(np.int32)], False, 2, band)
    cv2.line(sem, *[tuple(np.round(p).astype(int)) for p in to_px([[3.5, 0.05], [3.5, 3.95]])], 2, 6)
    cv2.line(sem, *[tuple(np.round(p).astype(int)) for p in to_px([[3.5, 1.5], [3.5, 2.4]])], 3, 6)        # дверь
    cv2.line(sem, *[tuple(np.round(p).astype(int)) for p in to_px([[4.2, 3.97], [5.6, 3.97]])], 4, band)   # окно
    logit = np.stack([(sem == k).astype(np.float32) for k in range(8)]) * 6
    if noisy:
        rng = np.random.default_rng(0)
        logit = np.stack([cv2.GaussianBlur(c, (0, 0), 1.5) + cv2.GaussianBlur(
            rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3) * 3 for c in logit])
        c = np.round(to_px([[5.0, 0.03]])[0]).astype(int)
        cv2.circle(logit[1], tuple(c), 7, 12.0, -1)                  # дыра 30 см в нижней стене
        c = np.round(to_px([[1.5, 2.0]])[0]).astype(int)
        cv2.ellipse(logit[2], tuple(c), (8, 3), 30, 0, 360, 12.0, -1)  # пятно «стены» в комнате
    prob = np.exp(logit - logit.max(0))
    prob /= prob.sum(0, keepdims=True)
    d = tmp_path / ("noisy" if noisy else "clean")
    d.mkdir()
    np.savez_compressed(d / "prob.npz", prob=prob.astype(np.float16))
    h = np.zeros((2, H, W), np.float32)
    h[0][sem == 4], h[1][np.isin(sem, (3, 4))] = 0.9 / 2.7, 2.1 / 2.7
    np.save(d / "heights.npy", h.astype(np.float16))
    sc = {"level": "L1", "floor_z": 0.0, "ceiling_z": 2.7, "frame": {"width": W, "height": H},
          "pixel_to_world": [[PX, 0.0, 0.5 * PX], [0.0, -PX, (H - 0.5) * PX]]}
    (d / "scene.json").write_text(json.dumps(sc), encoding="utf-8")
    return d, R, o


def _angle_mod90(v, R):
    """Отклонение направления от осей квартиры (с учётом поворота), градусы."""
    a = math.degrees(math.atan2(v[1], v[0])) - math.degrees(ROT)
    return abs((a + 45) % 90 - 45)


@pytest.mark.parametrize("noisy", [False, True])
def test_plan_is_clean(tmp_path, noisy):
    d, R, o = _scene(tmp_path, noisy)
    plan = vectorize(d, log=lambda *a: None)
    walls = plan["walls"]
    arcs = [w for w in walls if "arc" in w]
    lines = [w for w in walls if "arc" not in w]
    # дуга - одна, с правильным радиусом (ось полосы - на 5 см внутри грани)
    assert len(arcs) == 1, [w["points"][:2] for w in arcs]
    r_axis = (4.0 ** 2 / 4 + 0.4 ** 2) / (2 * 0.4) - 0.05
    assert abs(arcs[0]["arc"]["radius"] - r_axis) / r_axis < 0.06
    # прямые: верх, право, низ, перегородка - по одной прямой, вдоль осей квартиры
    assert len(lines) == 4, [np.round(w["points"], 2).tolist() for w in lines]
    for w in lines:
        p = np.asarray(w["points"])
        assert len(p) == 2 and _angle_mod90(p[1] - p[0], R) < 0.5
    # ось по центру: перегородка - на x = 3,5 м в СК квартиры, толщина 12 см
    part = [w for w in lines if abs((np.asarray(w["points"]) - o) @ R[:, 0] - 3.5).max() < 0.03]
    assert len(part) == 1 and abs(part[0]["thickness"] - 0.12) < 0.03
    # контур замкнут: каждый конец прямой упирается в другую стену
    segs = [(np.asarray(w["points"])[k], np.asarray(w["points"])[k + 1]) for w in walls
            for k in range(len(w["points"]) - 1)]
    for w in lines:
        for e in np.asarray(w["points"]):
            dist = min(np.linalg.norm(e - (a + np.clip((e - a) @ (b - a) / max((b - a) @ (b - a), 1e-9), 0, 1) * (b - a)))
                       for a, b in segs if not (np.allclose(a, w["points"][0]) and np.allclose(b, w["points"][1])))
            assert dist < 0.1, (w["id"], e, dist)
    # проёмы: дверь в перегородке, окно в верхней стене, с высотами
    kinds = sorted(o_["opening_type"] for o_ in plan["openings"])
    assert kinds == ["door", "window"], kinds
    win = next(o_ for o_ in plan["openings"] if o_["opening_type"] == "window")
    assert abs(win["width"] - 1.4) < 0.12 and abs(win["sill"] - 0.9) < 0.05
    door = next(o_ for o_ in plan["openings"] if o_["opening_type"] == "door")
    assert door["wall_id"] == part[0]["id"] and abs(door["width"] - 0.9) < 0.12
    # помещения: два, слева больше (дуга добавляет площадь)
    areas = sorted(r["area"] for r in plan["rooms"])
    assert len(areas) == 2 and areas[1] > areas[0] * 1.2, areas
