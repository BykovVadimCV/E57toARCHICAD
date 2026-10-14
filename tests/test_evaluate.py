"""Разбор результатов (simscan/evaluate.py) на искусственном уровне: числа совпадают с ожидаемыми."""
import numpy as np

from simscan.evaluate import aggregate, level_stats, level_summary


def _level():
    sem = np.ones((80, 80), np.uint8)              # помещение
    sem[:, 38:42] = 2                              # стена
    sem[30:45, 38:42] = 3                          # дверь
    sem[0:2, :] = 255
    orient = np.zeros((2, 80, 80), np.float32)
    orient[0][np.isin(sem, (2, 3))] = -1.0         # вертикальная стена: θ = 90°, (cos 2θ, sin 2θ) = (-1, 0)
    return {"sem": sem, "dist": np.ones((1, 80, 80), np.float32), "orient": orient,
            "heights": np.zeros((2, 80, 80), np.float32)}


def _prob(sem):
    p = np.zeros((8,) + sem.shape, np.float32)
    s = np.where(sem == 255, 1, sem)
    np.put_along_axis(p, s[None].astype(np.int64), 1.0, 0)
    return p


def test_perfect_prediction():
    g = _level()
    rep = aggregate([level_stats(g, {"prob": _prob(g["sem"]), "dist": g["dist"], "orient": g["orient"],
                                     "heights": g["heights"]})])
    assert rep["classes"]["стена"]["iou"] == 1.0 and rep["struct_iou"] == 1.0
    assert rep["openings"]["дверь"] == {**rep["openings"]["дверь"], "found": 1, "gt": 1, "false": 0, "f1": 1.0}
    assert rep["openings"]["окно"]["precision"] is None and rep["openings"]["окно"]["f1"] is None
    assert rep["orientation_err_deg"]["по осям"]["median"] <= 0.5
    assert rep["skeleton"]["recall"] == 1.0 and rep["calibration"]["ece"] == 0.0


def test_missed_door_and_broken_wall():
    g = _level()
    pred = g["sem"].copy()
    pred[30:45, 38:42] = 2                         # дверь принята за стену
    pred[60:70, 38:42] = 1                         # разрыв стены
    s = level_stats(g, {"prob": _prob(pred)})
    rep = aggregate([s])
    d = rep["openings"]["дверь"]
    assert d["found"] == 0 and d["missed_as"]["стена"] == 1
    assert rep["skeleton"]["recall"] < 0.95 and rep["skeleton"]["pieces_pred"] == 2   # 10 px разрыва минус допуск
    assert level_summary(s)["struct_iou"] < 1.0
