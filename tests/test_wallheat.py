"""Тепловая карта стен: шумные срезы (стол, светильник) получают меньший вес."""
import numpy as np

from simscan.e57io import write_merged_e57
from simscan.wallheat import wall_heatmap

from test_floorplan import _apartment, _grid


def test_noisy_slices_get_lower_weight(tmp_path):
    p = _apartment()
    # стол 1,2 x 0,8 м на высоте 0,75 и плоский светильник 0,6 x 0,6 на 2,45 - в мировой СК
    x, y = _grid(1.0, 2.2, 1.0, 1.8)
    table = np.c_[x, y, np.full(len(x), 0.75)]
    x, y = _grid(5.5, 6.1, 1.5, 2.1)
    lamp = np.c_[x, y, np.full(len(x), 2.45)]
    a = np.radians(12)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    extra = np.vstack([table, lamp]) @ R.T + [3.0, -2.0, 10.0]
    pts = np.vstack([p, extra])
    path = tmp_path / "apt.e57"
    write_merged_e57(path, pts, np.full(len(pts), 0.5, np.float32))
    res = wall_heatmap(path, tmp_path / "out", px=0.02, log=lambda *a: None)
    sl = {round(s["z"], 1): s for s in res["levels"][0]["slices"]}
    # качество: срезы со столом и светильником хуже чистых
    clean = np.median([sl[z]["f1"] for z in (0.3, 0.4, 1.2, 1.3, 2.0)])
    assert sl[0.7]["f1"] < clean - 0.05 and sl[2.4]["f1"] < clean - 0.05, sl
    # по высоте: у пола ценится выше, середина почти нет
    assert sl[0.3]["weight"] > 5 * sl[1.5]["weight"], sl
    heat = np.load(tmp_path / "out" / "heat.npy").astype(float)
    assert heat.max() > 0.9                                   # стена во всю высоту
    assert (heat > 0.5).mean() < 0.1                          # стол и светильник не стали стенами
    for f in ("heat.png", "slices.png", "labels.png", "heat_full.png", "labels_full.png", "labels.npy"):
        assert (tmp_path / "out" / f).exists(), f
    # разметка: дверь в перегородке - по перемычке (только верхние срезы), ширина 0,9 м
    doors = res["levels"][0]["doors"]
    assert len(doors) == 1 and abs(doors[0]["width"] - 0.9) < 0.12, doors
    lab = np.load(tmp_path / "out" / "labels.npy")
    assert (lab == 1).sum() * 0.02 ** 2 > 0.3                  # стены размечены
