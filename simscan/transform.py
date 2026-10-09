"""Повороты, кватернионы и преобразование «планировка -> СК объекта (мир)»."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def quat_from_matrix(R: np.ndarray) -> np.ndarray:
    """Кватернион (w, x, y, z) по матрице поворота (метод Шеппарда), w >= 0."""
    m = np.asarray(R, float)
    tr = np.trace(m)
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def small_rotation(sigma_deg: float, rng: np.random.Generator) -> np.ndarray:
    """Случайный малый поворот: ошибка регистрации в основном по азимуту, наклон меньше."""
    if sigma_deg <= 0:
        return np.eye(3)
    rz, rx, ry = np.radians(rng.normal(0, sigma_deg, 3) * np.array([1.0, 0.3, 0.3]))
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


@dataclass
class WorldTransform:
    """world = Rz(yaw) @ layout + offset."""
    yaw_deg: float = 0.0
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def R(self) -> np.ndarray:
        return rot_z(math.radians(self.yaw_deg))

    def matrix(self) -> np.ndarray:
        M = np.eye(4)
        M[:3, :3] = self.R
        M[:3, 3] = self.offset
        return M

    def to_world(self, pts: np.ndarray) -> np.ndarray:
        return np.asarray(pts, float) @ self.R.T + np.asarray(self.offset)

    def to_layout(self, pts: np.ndarray) -> np.ndarray:
        return (np.asarray(pts, float) - np.asarray(self.offset)) @ self.R

    def to_dict(self) -> dict:
        return {"yaw_deg": self.yaw_deg, "offset_m": list(self.offset),
                "matrix_layout_to_world": self.matrix().tolist(),
                "formula": "world = Rz(yaw) * layout + offset"}
