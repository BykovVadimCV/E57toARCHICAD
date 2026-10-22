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
    inner = np.vstack([np.c_[arc_x + 0.05, arc_y], [[6.0 - 0.05, 4.0 - 0.05], [6.0 - 0.05, 0.05]]])
    inner[0, 1], inner[len(arc_y) - 1, 1] = 0.05, 3.95       # обход как у outer: дуга снизу вверх
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


def test_kinked_wall_window_and_open_area(tmp_path):
    """Наружная стена с изломом 14° и окном через излом: окно целиком на одной стене, излом не
    выпрямлен (стена - ломаная или дуга); пятно «помещения» снаружи, не замкнутое стенами
    (площадка за дверью), - не помещение."""
    W, H = 300, 260
    sem = np.zeros((H, W), np.uint8)
    room = np.array([[40, 40], [200, 40], [210, 120], [200, 200], [40, 200]])   # правая стена с изломом 14°
    cv2.fillPoly(sem, [room], 1)
    cv2.polylines(sem, [room], True, 2, 6)
    win = np.array([[205, 80], [210, 120], [205, 160]]) * 4      # окно 1,6 м по стене, через излом
    cv2.polylines(sem, [np.round(win).astype(np.int32)], False, 4, 6, cv2.LINE_8, 2)
    cv2.ellipse(sem, (120, 235), (60, 15), 0, 0, 360, 1, -1)    # площадка снаружи
    logit = np.stack([(sem == k).astype(np.float32) for k in range(8)]) * 6
    prob = np.exp(logit - logit.max(0))
    prob /= prob.sum(0, keepdims=True)
    np.savez_compressed(tmp_path / "prob.npz", prob=prob.astype(np.float16))
    sc = {"level": "L1", "floor_z": 0.0, "ceiling_z": 2.7, "frame": {"width": W, "height": H},
          "pixel_to_world": [[PX, 0.0, 0.5 * PX], [0.0, -PX, (H - 0.5) * PX]]}
    (tmp_path / "scene.json").write_text(json.dumps(sc), encoding="utf-8")
    plan = vectorize(tmp_path, log=lambda *a: None)
    assert len(plan["rooms"]) == 1, [r["area"] for r in plan["rooms"]]
    r = plan["rooms"][0]
    assert abs(r["area"] - 3.0 * 3.1) < 0.6, r
    win = [o for o in plan["openings"] if o["opening_type"] == "window"]
    assert len(win) == 1 and abs(win[0]["width"] - 1.6) < 0.12, win
    host = next(w for w in plan["walls"] if w["id"] == win[0]["wall_id"])
    assert len(host["points"]) >= 3                             # излом сохранён


def test_chain_at_openings():
    """Две прямые с изломом 8°, окно сети через стык -> одна стена-ломаная с тем же изломом."""
    from simscan.vectorize import chain_at_openings

    H, W = 120, 220
    J = np.array([110.0, 60.0])
    a = J + np.array([-100.0, -100.0 * math.tan(math.radians(8))])
    b = J + np.array([100.0, 0.0])
    items = [{"pts": np.array([a, J]), "t": 6.0}, {"pts": np.array([J, b]), "t": 6.0}]
    prob = np.zeros((8, H, W), np.float32)
    prob[0] = 1
    win = np.zeros((H, W), np.uint8)
    cv2.polylines(win, [np.round(np.array([a + (J - a) * 0.6, J, J + (b - J) * 0.4])).astype(np.int32)], False, 1, 6)
    prob[4][win > 0], prob[0][win > 0] = 1, 0
    out = chain_at_openings(prob, items, PX)
    assert len(out) == 1 and len(out[0]["pts"]) == 3
    assert np.allclose(out[0]["pts"][0], a) and np.allclose(out[0]["pts"][1], J) and np.allclose(out[0]["pts"][2], b)
    far = np.zeros((H, W), np.uint8)                           # окно целиком на одной прямой - не трогать
    cv2.line(far, tuple(np.round(J + (b - J) * 0.3).astype(int)), tuple(np.round(J + (b - J) * 0.8).astype(int)), 1, 6)
    prob[4], prob[0] = far.astype(np.float32), 1 - far
    assert len(chain_at_openings(prob, [dict(w) for w in items], PX)) == 2


def _niche_scene(d, noisy: bool, rot_deg: float = 7.0):
    """Комната 6 x 4 м (повёрнута): в верхней стене две ниши-эркера 0,8 x 0,4 м с окном в
    задней стенке, пилястр 0,3 x 0,4 м у левой стены, снаружи слева - терраса со своей
    стеной, не ограничивающей помещений."""
    W, H = 520, 340
    a = math.radians(rot_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    c0 = np.array([290.0, 180.0])

    def q(pts):
        p = ((np.asarray(pts, float) - [3, 2]) / PX) @ R.T
        return np.round(np.c_[c0[0] + p[:, 0], c0[1] - p[:, 1]] * 16).astype(np.int32)

    t = 0.15
    inner = [[0, 0], [6, 0], [6, 4], [4.6, 4], [4.6, 4.4], [3.8, 4.4], [3.8, 4], [2.2, 4], [2.2, 4.4], [1.4, 4.4],
             [1.4, 4], [0, 4]]
    outer = [[-t, -t], [6 + t, -t], [6 + t, 4 + t], [4.6 + t, 4 + t], [4.6 + t, 4.4 + t], [3.8 - t, 4.4 + t],
             [3.8 - t, 4 + t], [2.2 + t, 4 + t], [2.2 + t, 4.4 + t], [1.4 - t, 4.4 + t], [1.4 - t, 4 + t], [-t, 4 + t]]
    sem = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sem, [q(outer)], 2, cv2.LINE_8, 4)
    cv2.fillPoly(sem, [q(inner)], 1, cv2.LINE_8, 4)
    for x0, x1 in ((1.45, 2.15), (3.85, 4.55)):
        cv2.fillPoly(sem, [q([[x0, 4.4], [x1, 4.4], [x1, 4.4 + t], [x0, 4.4 + t]])], 4, cv2.LINE_8, 4)
    cv2.fillPoly(sem, [q([[0, 1.5], [0.3, 1.5], [0.3, 1.9], [0, 1.9]])], 2, cv2.LINE_8, 4)
    cv2.fillPoly(sem, [q([[-1.5, 1.0], [-1.5 + t, 1.0], [-1.5 + t, 3.5], [-1.5, 3.5]])], 2, cv2.LINE_8, 4)
    cv2.fillPoly(sem, [q([[-1.5 + t, 1.0], [-t, 1.0], [-t, 3.5], [-1.5 + t, 3.5]])], 1, cv2.LINE_8, 4)
    logit = np.stack([(sem == k).astype(np.float32) for k in range(8)]) * 6
    if noisy:
        rng = np.random.default_rng(0)
        logit = np.stack([cv2.GaussianBlur(c, (0, 0), 1.2) + cv2.GaussianBlur(
            rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 3) * 2.5 for c in logit])
    prob = np.exp(logit - logit.max(0))
    prob /= prob.sum(0, keepdims=True)
    np.savez_compressed(d / "prob.npz", prob=prob.astype(np.float16))
    sc = {"level": "L1", "floor_z": 0.0, "ceiling_z": 2.7, "frame": {"width": W, "height": H},
          "pixel_to_world": [[PX, 0.0, 0.5 * PX], [0.0, -PX, (H - 0.5) * PX]]}
    (d / "scene.json").write_text(json.dumps(sc), encoding="utf-8")
    return rot_deg


@pytest.mark.parametrize("noisy", [False, True])
def test_niches_square_and_stray_walls_dropped(tmp_path, noisy):
    """Боковины ниш - поперёк стены, а не диагонали; стена террасы (не ограничивает помещений)
    - вон; два окна в нишах; одно помещение."""
    rot = _niche_scene(tmp_path, noisy)
    plan = vectorize(tmp_path, log=lambda *a: None)
    assert len(plan["rooms"]) == 1
    for w in plan["walls"]:
        p = np.asarray(w["points"])
        assert w["rooms"] or w["length"] <= 1.0, w                 # ничейных длинных нет
        for a, b in zip(p[:-1], p[1:]):
            ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) - rot
            assert abs((ang + 45) % 90 - 45) < 3.0, (w["id"], p.round(2).tolist())   # всё по осям комнаты
    win = [o for o in plan["openings"] if o["opening_type"] == "window"]
    assert len(win) == 2 and all(abs(o["width"] - 0.7) < 0.15 for o in win), win


def test_unseen_candidates():
    """Сеть видит «помещение» там, где сканер не видел ни пола, ни потолка: не замкнутое стенами
    до конца - не помещение (терраса за окном), замкнутое - остаётся (комната за закрытой дверью)."""
    from simscan.vectorize import ENCLOSED_UNSEEN, MIN_ENCLOSED, room_candidates, struct_mask

    def scene(open_side: bool):
        prob = np.zeros((8, 200, 200), np.float32)
        prob[0] = 1
        sem = np.zeros((200, 200), np.uint8)
        sem[40:160, 40:160] = 2
        sem[46:154, 46:154] = 1
        prob[:] = 0
        for k in range(8):
            prob[k][sem == k] = 1
        if open_side:                                     # нижняя сторона - неуверенно (не стена, не улица)
            prob[:, 150:170, 46:154] = 0
            prob[0, 150:170, 46:154], prob[1, 150:170, 46:154], prob[2, 150:170, 46:154] = 0.4, 0.4, 0.2
        return prob

    for open_side in (True, False):
        prob = scene(open_side)
        mask = struct_mask(prob, PX)
        lab, c = room_candidates(prob, [], mask, PX)
        room = max(c, key=lambda r: r["area_m2"])
        assert room["ok"] and room["seen"] is None
        assert (MIN_ENCLOSED <= room["score"] < ENCLOSED_UNSEEN) == open_side, room["score"]
        _, c0 = room_candidates(prob, [], mask, PX, seen=np.zeros((200, 200), bool))
        assert max(c0, key=lambda r: r["area_m2"])["ok"] == (not open_side)
        _, c1 = room_candidates(prob, [], mask, PX, seen=np.ones((200, 200), bool))
        assert max(c1, key=lambda r: r["area_m2"])["ok"]
