"""Оси стен: общие шаги сборки плана для векторизации (vectorize.py) и cellplan.py.

Стена - словарь {"pts": ломаная оси [n, 2], "t": толщина, ...}; прямая - две точки, дуга -
ломаная по дуге (и "arc": {"c", "r"}). Единицы - любые (пиксели или метры): пороги в
метрах переводятся функцией P(м) -> единицы.

  regularize       направления прямых кластеризуются (по длине, с точностью tol_deg) и
                   прямые поворачиваются к направлению кластера: снимает дрожание
                   оценки, но угол 85-95° остаётся своим кластером;
  merge_collinear  соосные прямые (одно направление, ось ближе части толщины) с
                   перекрытием или зазором до gap_m - одна стена;
  join             конец прямой продлевается (или укорачивается) до пересечения с осью
                   другой стены - прямой или дуги: L- и T-стыки. Свободный конец (стена
                   оборвалась - пропуск сети, тень) достаёт дальше, до free_reach_m.
"""
from __future__ import annotations

import math

import numpy as np


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _is_line(w) -> bool:
    """Прямая ненулевой длины (вырожденная - без направления, её не сливать и не стыковать)."""
    return len(w["pts"]) == 2 and not w.get("arc") and float(np.linalg.norm(w["pts"][1] - w["pts"][0])) > 1e-6


def _rotate(w, th: float) -> None:
    """Прямая поворачивается вокруг середины к направлению th (длина та же)."""
    d = np.array([math.cos(th), math.sin(th)])
    a, b = w["pts"]
    m = (a + b) / 2
    h = abs((b - a) @ d) / 2 if abs((b - a) @ d) > 0 else np.linalg.norm(b - a) / 2
    d = d if (b - a) @ d >= 0 else -d
    w["pts"] = np.array([m - d * h, m + d * h])


def regularize(items: list, tol_deg: float = 1.0, P=None, long_m: float = 1.0, snap_deg: float = 5.0) -> None:
    """Направления прямых -> кластеры по длине (циклически по 180°), прямая поворачивается
    вокруг середины к среднему направлению кластера. С P: кластеры - только по длинным
    (от long_m) прямым, короткая прямая притягивается к ближайшему кластеру ближе snap_deg
    (её направление по немногим пикселям шумное) или остаётся как есть."""
    lines = [w for w in items if _is_line(w)]
    if len(lines) < 2:
        return
    if P is not None:
        long_ = [w for w in lines if np.linalg.norm(w["pts"][1] - w["pts"][0]) >= P(long_m)]
        if len(long_) >= 2:
            regularize(long_, tol_deg)
            th_long = np.array([math.atan2(*(w["pts"][1] - w["pts"][0])[::-1]) % math.pi for w in long_])
            for w in lines:
                if any(w is x for x in long_):
                    continue
                th = math.atan2(*(w["pts"][1] - w["pts"][0])[::-1]) % math.pi
                diff = np.abs((th_long - th + math.pi / 2) % math.pi - math.pi / 2)
                k = int(diff.argmin())
                if diff[k] <= math.radians(snap_deg):
                    _rotate(w, th_long[k])
            return
    ang = np.array([math.atan2(*(w["pts"][1] - w["pts"][0])[::-1]) % math.pi for w in lines])
    L = np.array([np.linalg.norm(w["pts"][1] - w["pts"][0]) for w in lines])
    order = np.argsort(ang)
    tol = math.radians(tol_deg)
    groups = [[order[0]]]
    for k in order[1:]:
        if ang[k] - ang[groups[-1][-1]] <= tol:
            groups[-1].append(k)
        else:
            groups.append([k])
    if len(groups) > 1 and ang[groups[0][0]] + math.pi - ang[groups[-1][-1]] <= tol:
        groups[0] = groups.pop() + groups[0]                    # через 0/180°
    for g in groups:
        z = np.sum(L[g] * np.exp(2j * ang[g]))                  # среднее по 2θ - без скачка у 180°
        th = np.angle(z) / 2
        for k in g:
            _rotate(lines[k], th)


def merge_collinear(items: list, P, gap_m: float = 0.3, off_m: float = 0.04, sin_tol: float = 0.05,
                    keep_outer_apart: bool = True, gap_max_m: float = 0.0, bridge_ok=None) -> list:
    """Соосные прямые с перекрытием или зазором до gap_m -> одна (ось - средняя по длине,
    толщина - средняя по длине; списки rooms/src объединяются). keep_outer_apart: наружная и
    внутренняя стена без перекрытия не сливаются (разные стены на одной линии). Зазор до
    gap_max_m закрывается, если bridge_ok(точка, точка) подтверждает (например, в щели сеть
    видит конструкцию)."""
    alive = [True] * len(items)
    changed = True
    while changed:
        changed = False
        for i, A in enumerate(items):
            if not alive[i] or not _is_line(A):
                continue
            a0, a1 = A["pts"]
            u = _unit(a1 - a0)
            nrm = np.array([-u[1], u[0]])
            for j in range(i + 1, len(items)):
                B = items[j]
                if not alive[j] or not _is_line(B):
                    continue
                b0, b1 = B["pts"]
                v = _unit(b1 - b0)
                if abs(u[0] * v[1] - u[1] * v[0]) > sin_tol:
                    continue
                off = ((b0 + b1) / 2 - a0) @ nrm
                if abs(off) > max(P(off_m), 0.25 * max(A["t"], B["t"])):
                    continue
                sa = sorted([0.0, (a1 - a0) @ u])
                sb = sorted([(b0 - a0) @ u, (b1 - a0) @ u])
                overlap = min(sa[1], sb[1]) - max(sa[0], sb[0])
                if overlap < -P(gap_m):
                    if overlap < -P(gap_max_m) or bridge_ok is None:
                        continue
                    g0, g1 = (sa[1], sb[0]) if sa[1] <= sb[0] else (sb[1], sa[0])
                    if not bridge_ok(a0 + u * g0 + nrm * off / 2, a0 + u * g1 + nrm * off / 2):
                        continue
                if keep_outer_apart and A.get("outer", False) != B.get("outer", False) and overlap <= 0:
                    continue
                la, lb = sa[1] - sa[0], sb[1] - sb[0]
                base = a0 + nrm * off * lb / max(la + lb, 1e-12)
                lo, hi = min(sa[0], sb[0]), max(sa[1], sb[1])
                fa, fb = A.get("free", (False, False)), B.get("free", (False, False))
                ends = [(0.0, fa[0]), ((a1 - a0) @ u, fa[1]), ((b0 - a0) @ u, fb[0]), ((b1 - a0) @ u, fb[1])]
                A["free"] = (min(ends)[1], max(ends)[1])               # свободны ли крайние концы
                A["pts"] = np.array([base + u * lo, base + u * hi])
                A["t"] = (A["t"] * la + B["t"] * lb) / max(la + lb, 1e-12)
                A["outer"] = A.get("outer", False) or B.get("outer", False)
                for key in ("rooms", "src"):
                    if key in A or key in B:
                        A[key] = sorted(set(A.get(key, [])) | set(B.get(key, []))) if key == "rooms" \
                            else A.get(key, []) + B.get(key, [])
                alive[j] = False
                changed = True
                a0, a1 = A["pts"]
    return [x for x, ok in zip(items, alive) if ok]


def join(items: list, P, tol_m: float = 0.05, free_reach_m: float = 0.4) -> None:
    """Конец прямой стены у оси другой стены (не параллельной; прямой или любого участка дуги)
    продлевается до пересечения осей: дотягивается на свою и чужую толщину (+ tol_m), а
    свободный конец (w["free"][e]) - до free_reach_m."""
    segs = []                                                    # (стена, a, b, концы ломаной)
    for w in items:
        pts = w["pts"]
        for k in range(len(pts) - 1):
            segs.append((w, pts[k], pts[k + 1], k == 0, k == len(pts) - 2))
    for A in items:
        if not _is_line(A):
            continue
        for e in (0, 1):
            p, q = A["pts"][e], A["pts"][1 - e]
            u = _unit(p - q)
            L_a = float(np.linalg.norm(p - q))
            best = None
            for B, b0, b1, first, last in segs:
                if B is A:
                    continue
                Lb = float(np.linalg.norm(b1 - b0))
                if Lb < 1e-9:
                    continue
                v = (b1 - b0) / Lb
                den = u[0] * v[1] - u[1] * v[0]
                if abs(den) < 0.3:                               # почти параллельны
                    continue
                r = b0 - p
                s_a = (r[0] * v[1] - r[1] * v[0]) / den          # p + s_a u = b0 + s_b v
                s_b = (r[0] * u[1] - r[1] * u[0]) / den
                reach = max(A["t"], B["t"]) + P(tol_m)
                if A.get("free", (False, False))[e]:
                    reach = max(reach, P(free_reach_m))
                lo = -reach if first else 0.0                    # дальше конца ломаной - только на её концах
                hi = Lb + reach if last else Lb
                if not (-max(A["t"], B["t"]) - P(tol_m) <= s_a <= reach and lo <= s_b <= hi):
                    continue
                if s_a < 0 and -s_a > 0.5 * L_a:                 # укорачивать больше половины - не стык
                    continue
                if best is None or abs(s_a) < abs(best):
                    best = s_a
            if best is not None:
                A["pts"][e] = p + u * best


def _dist_to_wall(p: np.ndarray, w: dict) -> float:
    best = math.inf
    pts = w["pts"]
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a
        L2 = float(d @ d)
        t = 0.0 if L2 < 1e-12 else min(max(float((p - a) @ d) / L2, 0.0), 1.0)
        best = min(best, float(np.linalg.norm(p - (a + t * d))))
    return best


def connected_ends(items: list, P, tol_m: float = 0.05) -> list[tuple[bool, bool]]:
    """Упирается ли каждый конец стены в ось другой стены (ближе полутолщин + tol_m)."""
    out = []
    for i, A in enumerate(items):
        ends = []
        for p in (A["pts"][0], A["pts"][-1]):
            ok = False
            for j, B in enumerate(items):
                if i != j and _dist_to_wall(p, B) <= (A["t"] + B["t"]) / 2 + P(tol_m):
                    ok = True
                    break
            ends.append(ok)
        out.append(tuple(ends))
    return out


def _shadow(A: dict, items: list, P, max_m: float = 1.0) -> bool:
    """Короткая прямая (до max_m), целиком лежащая вдоль более длинной параллельной стены
    вплотную (ближе полутолщин + 15 см) - тень той стены (кусок окна, подоконника, шум)."""
    if not _is_line(A):
        return False
    a, b = A["pts"]
    L = float(np.linalg.norm(b - a))
    if L >= P(max_m):
        return False
    u = (b - a) / L
    samples = a + (b - a) * np.linspace(0, 1, 7)[:, None]
    for B in items:
        if B is A or not _is_line(B) or float(np.linalg.norm(B["pts"][1] - B["pts"][0])) <= L:
            continue
        v = _unit(B["pts"][1] - B["pts"][0])
        if abs(u[0] * v[1] - u[1] * v[0]) > 0.1:
            continue
        lim = (A["t"] + B["t"]) / 2 + P(0.15)
        if all(_dist_to_wall(q, B) <= lim for q in samples):
            return True
    return False


def _bevel(A: dict, items: list, P, max_m: float = 0.5) -> bool:
    """Скос угла: короткая прямая, оба конца которой упираются в две непараллельные стены,
    чьи оси пересекаются не дальше её длины от неё (утончение срезает прямой угол
    диагональю). Ниша - короткая стенка между параллельными стенами - не скос."""
    if not _is_line(A):
        return False
    a, b = A["pts"]
    L = float(np.linalg.norm(b - a))
    if L >= P(max_m):
        return False
    nb = []
    for p in (a, b):
        best = None
        for B in items:
            if B is A or not _is_line(B):
                continue
            d = _dist_to_wall(p, B)
            if d <= (A["t"] + B["t"]) / 2 + P(0.05) and (best is None or d < best[0]):
                best = (d, B)
        if best is None:
            return False
        nb.append(best[1])
    B, C = nb
    if B is C:
        return False
    u, v = _unit(B["pts"][1] - B["pts"][0]), _unit(C["pts"][1] - C["pts"][0])
    den = u[0] * v[1] - u[1] * v[0]
    if abs(den) < 0.3:                                      # параллельные - ниша
        return False
    r = C["pts"][0] - B["pts"][0]
    x = B["pts"][0] + u * (r[0] * v[1] - r[1] * v[0]) / den
    return float(np.linalg.norm(x - (a + b) / 2)) <= L


def tidy(items: list, P, reach_m: float = 0.5, min_alone_m: float = 1.0, min_stub_m: float = 0.5,
         max_bridge_m: float = 0.25, rounds: int = 3) -> list:
    """Чистка после сборки: висящий конец ещё раз тянется к соседней стене (до reach_m);
    вон - одиночная стена (оба конца ни во что не упираются) короче min_alone_m (шум),
    огрызок с висящим концом короче min_stub_m, перемычка короче max_bridge_m между двумя
    стенами (остаток утончения на стыке), скос угла (_bevel; стены потом сходятся в угол на
    следующем круге) и тень (_shadow). Повторяется, пока что-то меняется."""
    def length(w):
        return float(np.linalg.norm(np.diff(w["pts"], axis=0), axis=1).sum())

    for _ in range(rounds):
        con = connected_ends(items, P)
        for w, c in zip(items, con):
            w["free"] = (not c[0], not c[1])
        join(items, P, free_reach_m=reach_m)
        con = connected_ends(items, P)
        keep = []
        for w, c in zip(items, con):
            L = length(w)
            if not any(c) and L < P(min_alone_m):
                continue
            if sum(c) == 1 and L < P(min_stub_m):
                continue
            if all(c) and L < P(max_bridge_m) and _is_line(w):
                continue
            if all(c) and _bevel(w, items, P):
                continue
            if _shadow(w, items, P):
                continue
            keep.append(w)
        if len(keep) == len(items):
            break
        items = keep
    return items
