"""Деформация плана (planwarp.py): углы не 90°, дуговые стены; сетка и эталон совпадают."""
import json
import math
from collections import defaultdict
from dataclasses import replace

import cv2
import numpy as np
import pytest

from simscan.config import SynthConfig, load_config
from simscan.generate import generate_scene, make_layout, scene_rng
from simscan.planwarp import PlanWarp, sample_warp


def _layouts(n=12):
    cfg = SynthConfig()
    rc = replace(cfg.realism, p_angle_jitter=1.0, p_arc=1.0)
    for idx in range(n):
        rng = scene_rng(5, idx)
        lay = make_layout(cfg, rng, 5, idx)
        yield lay, sample_warp([lay], np.eye(2), rc, rng)


def _thickness(W, w, s0, s1):
    """Расстояние от точек внешней грани до образа внутренней (ломаной)."""
    inner = W(np.array([w.point(x, -w.thickness / 2) for x in np.linspace(s0, s1, 800)]))
    outer = W(np.array([w.point(x, w.thickness / 2) for x in np.linspace(s0, s1, 12)[1:-1]]))
    return np.array([np.linalg.norm(inner - p, axis=1).min() for p in outer])


def test_warp_geometry():
    angles, n_arcs, n_grid = [], 0, 0
    for lay, spec in _layouts():
        W = PlanWarp(spec)
        n_grid += "grid" in spec
        arcs = {a["wall_id"]: a for a in spec.get("arcs", [])}
        n_arcs += len(arcs)
        for w in lay.walls:
            a = arcs.get(w.id)
            spans = [(-w.ext0, w.length + w.ext1)] if a is None else \
                [(-w.ext0, a["s"][0]), (a["s"][1], w.length + w.ext1)]
            for s0, s1 in spans:
                if s1 - s0 < 0.3:
                    continue
                q = W(np.array([w.point(x) for x in np.linspace(s0, s1, 30)]))
                d = q[-1] - q[0]
                nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                assert np.abs((q - q[0]) @ nrm).max() < 0.01          # прямая остаётся прямой (до см)
                assert np.abs(_thickness(W, w, s0, s1) - w.thickness).max() < 0.01   # грани параллельны
            if a is not None:                                          # дуга: окружность, толщина та же
                s0, s1 = a["s"]
                mid = np.linspace(s0 + 0.25 * (s1 - s0), s1 - 0.25 * (s1 - s0), 20)
                q = W(np.array([w.point(x) for x in mid]))
                p0, p1 = np.array(a["p0"]), np.array(a["p1"])
                c = (p0 + p1) / 2 + np.array(a["out"]) * (a["h"] - a["R"])
                assert np.abs(np.linalg.norm(q - c, axis=1) - a["R"]).max() < 0.005
                assert np.abs(_thickness(W, w, mid[0], mid[-1]) - w.thickness).max() < 0.006
        nodes = defaultdict(list)                                       # углы в узлах прямых стен
        for w in lay.walls:
            if w.id not in arcs:
                nodes[tuple(np.round(w.p0, 4))].append((w.p0, w.p1))
                nodes[tuple(np.round(w.p1, 4))].append((w.p1, w.p0))
        for lst in nodes.values():
            dirs = []
            for p, q in lst:
                p, q = np.asarray(p, float), np.asarray(q, float)
                ab = W(np.array([p, p + (q - p) / np.linalg.norm(q - p) * 0.3]))
                dirs.append((ab[1] - ab[0]) / np.linalg.norm(ab[1] - ab[0]))
            for i in range(len(dirs)):
                for j in range(i + 1, len(dirs)):
                    cosv = abs(dirs[i] @ dirs[j])
                    if cosv < 0.5:
                        angles.append(math.degrees(math.acos(cosv)))
        x0, y0, x1, y1 = lay.bbox()                                     # обратимость: без складок
        P = np.c_[[v.ravel() for v in np.meshgrid(np.linspace(x0 - 1, x1 + 1, 90), np.linspace(y0 - 1, y1 + 1, 90))]].T
        e = 1e-4
        jx = (W(P + [e, 0]) - W(P - [e, 0])) / (2 * e)
        jy = (W(P + [0, e]) - W(P - [0, e])) / (2 * e)
        assert (jx[:, 0] * jy[:, 1] - jx[:, 1] * jy[:, 0]).min() > 0.3
    dev = 90 - np.array(angles)
    assert n_grid >= 10 and n_arcs >= 8, (n_grid, n_arcs)
    assert dev.max() <= 5.0 + 1e-6 and (dev > 0.5).mean() > 0.3, (dev.max(), (dev > 0.5).mean())


def test_old_matrix_warp_is_affine():
    A = [[1.0, 0.01], [0.0, 1.0]]
    W = PlanWarp(A)
    p = np.array([[1.0, 2.0], [3.0, -4.0]])
    assert W.linear and np.allclose(W(p), p @ np.asarray(A).T)
    assert PlanWarp(None).identity and PlanWarp({"A": np.eye(2).tolist()}).identity


@pytest.fixture(scope="module")
def warped_scene(tmp_path_factory):
    from simscan.netinput import build_scene

    tmp = tmp_path_factory.mktemp("warp")
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.4},
                                 "realism": {"p_angle_jitter": 1.0, "p_arc": 1.0},
                                 "export": {"preview": False, "write_merged": False, "free_space_input": False,
                                            "unet_mask": False}})
    for idx in range(6):                                                # сцена с наклонами и дугой
        generate_scene(cfg, tmp / "scene", seed=7, index=idx)
        w = json.loads((tmp / "scene" / "layout.json").read_text(encoding="utf-8"))["meta"]["warp"]
        if "grid" in w and w.get("arcs"):
            break
    assert "grid" in w and w.get("arcs")
    return tmp, w, build_scene(tmp / "scene", tmp / "dataset", log=lambda *a: None)


def test_scan_and_labels_agree(warped_scene):
    tmp, warp, dirs = warped_scene
    d = dirs[0]
    x = np.load(f"{d}/input.npy").astype(np.float32)
    sem = cv2.imread(f"{d}/sem.png", cv2.IMREAD_UNCHANGED)
    ori = np.load(f"{d}/orient.npy").astype(np.float32)
    # грани облака лежат на размеченных стенах (как в test_netinput, но с деформацией)
    face = (x[6] > 0.5) & ((x[0] > 0.5) | (x[2] > 0.5))
    body = np.isin(sem, (2, 3, 4, 5, 255))
    dt = cv2.distanceTransform((~body).astype(np.uint8), cv2.DIST_L2, 5)
    assert face.sum() > 500 and (dt[face] <= 1.0).mean() > 0.8
    # ориентация разметки - по направлению граней облака (тензор структуры), а не по осям
    f = cv2.GaussianBlur(face.astype(np.float32), (0, 0), 1.5)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), -cv2.Sobel(f, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(v, (0, 0), 3) for v in (gx * gx, gy * gy, gx * gy))
    coh = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2) / (jxx + jyy + 1e-9)
    cloud = 0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2
    m = (sem == 2) & face & (coh > 0.8)
    lab = 0.5 * np.arctan2(ori[1], ori[0])
    err = np.degrees(np.abs(np.angle(np.exp(2j * (lab[m] - cloud[m]))))) / 2
    assert m.sum() > 200 and np.median(err) < 1.5, np.median(err)
    wall = np.isin(sem, (2, 3, 4, 5))
    assert np.allclose(np.hypot(ori[0], ori[1])[wall], 1, atol=0.01)
    # стены не только по двум осям
    a = np.round(np.degrees(lab[sem == 2]) % 180)
    assert len(np.unique(a)) >= 6
    # векторный эталон: дуговая стена - ломаная с окружностью
    vec = json.loads(open(f"{d}/vector.json", encoding="utf-8").read())
    arcs = [w for w in vec["walls"] if w["geometry"]["type"] == "polyline"]
    assert len(arcs) == len(warp["arcs"]) and all(w["geometry"]["arc"]["radius"] > 1.5 for w in arcs)


def test_gt_masks_follow_arc(warped_scene):
    """Маски эталона (стена или окно на секущей) повторяют дугу: середина пролёта сдвинута
    наружу на стрелу."""
    from simscan.groundtruth import RasterFrame
    from simscan.layout import Layout

    tmp, warp, _ = warped_scene
    lay = Layout.load_json(tmp / "scene" / "layout.json")
    info = json.loads((tmp / "scene" / "gt" / "gt.json").read_text(encoding="utf-8"))
    fr = RasterFrame(**info["frame"])
    walls = cv2.imread(str(tmp / "scene" / "gt" / "classes.png"), cv2.IMREAD_UNCHANGED) > 0
    W = PlanWarp(warp)
    for a in warp["arcs"]:
        w = lay.wall(a["wall_id"])
        mid = w.point(sum(a["s"]) / 2)
        for off, expect in ((0.0, True), (w.thickness / 2 + 0.05, False)):
            q = W(np.asarray(mid) + w.n * w.outward * off)[0]          # ось на дуге - стена
            i, j = fr.xy_to_ij(q[0], q[1])
            assert walls[int(round(float(i))), int(round(float(j)))] == expect
        q = W(np.asarray(mid))[0]                                       # и это не хорда
        p0 = np.array(a["p0"])
        assert abs((q - p0) @ np.array(a["out"]) - a["h"]) < 0.01
