"""Неотсканированное помещение -> улица (расстояние - «далеко»), отсканированное и стены с точками - размечены."""
import numpy as np

from simscan.netinput import INTERIOR, OUTSIDE, WALL, unseen_to_outside

PX = 0.02


def test_unseen_to_outside():
    H, W = 100, 200
    sem = np.zeros((H, W), np.uint8)
    rid = np.zeros((H, W), np.uint16)
    sem[10:90, 10:190] = INTERIOR
    rid[10:90, 10:97], rid[10:90, 103:190] = 1, 2
    sem[10:90, 97:103] = WALL                                  # перегородка между комнатами
    sem[5:10, 5:195] = WALL                                    # наружная стена сверху
    inp = np.zeros((8, H, W), np.float32)
    inp[3][10:90, 10:97] = 1                                   # пол комнаты 1 виден
    inp[0][10:90, 96:98] = 1                                   # грань перегородки - из комнаты 1
    inp[3][40:50, 103:110] = 1                                 # в комнату 2 видно чуть-чуть
    lab = {"sem": sem.copy(), "sem_raw": sem.copy(), "room_id": rid, "dist": np.zeros((1,) + sem.shape, np.float16)}
    n = unseen_to_outside(lab, inp, PX)
    out = lab["sem"]
    assert n > 0
    assert (out[10:90, 10:97] == INTERIOR).all()               # комната 1 не тронута
    assert (out[10:90, 97:103] == WALL).all()                  # перегородка с точками - стена
    assert (out[70:85, 150:185] == OUTSIDE).all()              # глубина комнаты 2 - улица
    assert (out[42:48, 104:108] == INTERIOR).all()             # видимое место комнаты 2 - осталось
    assert (out[5:10, 160:190] == OUTSIDE).all()               # стена комнаты 2 без точек - улица
    assert (lab["dist"][0][70:85, 150:185] == 1).all()         # невидимой грани нет - «далеко»
    assert (out[5:10, 20:80] == WALL).all()                    # дальняя от комнаты 2 стена - нет


def test_room_without_station_outside_beyond_seen():
    """Комната без стоянки, видная через дверь на треть: видимое размечено, остальное - улица."""
    H, W = 100, 200
    sem = np.zeros((H, W), np.uint8)
    rid = np.zeros((H, W), np.uint16)
    sem[10:90, 10:190] = INTERIOR
    rid[10:90, 10:97], rid[10:90, 103:190] = 1, 2
    sem[10:90, 97:103] = WALL
    inp = np.zeros((8, H, W), np.float32)
    inp[3][10:90, 10:97] = 1
    inp[3][10:90, 103:135] = 1                                 # треть комнаты 2 видна - больше min_seen
    lab = {"sem": sem.copy(), "sem_raw": sem.copy(), "room_id": rid, "dist": np.zeros((1,) + sem.shape, np.float16)}
    assert unseen_to_outside(lab, inp, PX) == 0                    # со стоянкой: размечено целиком
    lab = {"sem": sem.copy(), "sem_raw": sem.copy(), "room_id": rid, "dist": np.zeros((1,) + sem.shape, np.float16)}
    assert unseen_to_outside(lab, inp, PX, no_station=[2]) > 0
    out = lab["sem"]
    assert (out[10:90, 10:97] == INTERIOR).all()
    assert (out[15:85, 105:135] == INTERIOR).all()             # видимая часть
    assert (out[15:85, 160:185] == OUTSIDE).all()               # невидимая глубина - улица
