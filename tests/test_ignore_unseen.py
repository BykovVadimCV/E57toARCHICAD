"""Неотсканированное помещение -> ignore, отсканированное и стены с точками - размечены."""
import numpy as np

from simscan.netinput import IGNORE, INTERIOR, WALL, ignore_unseen

PX = 0.02


def test_ignore_unseen():
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
    lab = {"sem": sem.copy(), "sem_raw": sem.copy(), "room_id": rid}
    n = ignore_unseen(lab, inp, PX)
    out = lab["sem"]
    assert n > 0
    assert (out[10:90, 10:97] == INTERIOR).all()               # комната 1 не тронута
    assert (out[10:90, 97:103] == WALL).all()                  # перегородка с точками - стена
    assert (out[70:85, 150:185] == IGNORE).all()               # глубина комнаты 2 - ignore
    assert (out[42:48, 104:108] == INTERIOR).all()             # видимое место комнаты 2 - осталось
    assert (out[5:10, 160:190] == IGNORE).all()                # стена комнаты 2 без точек - ignore
    assert (out[5:10, 20:80] == WALL).all()                    # дальняя от комнаты 2 стена - нет
