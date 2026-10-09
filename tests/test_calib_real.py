"""calib-real: из E57 - вход сети, срезы, карта, срез у пола и сводка в одном zip."""
import json
import zipfile

import numpy as np

from simscan.calib_real import calib_real, compare_calib
from simscan.e57io import write_merged_e57

from test_floorplan import _apartment


def test_calib_real_zip(tmp_path):
    p = _apartment()
    e57 = tmp_path / "apt.e57"
    write_merged_e57(e57, p, np.full(len(p), 0.5, np.float32))
    z = calib_real([e57], tmp_path / "calib", log=lambda *a: None)
    names = set(zipfile.ZipFile(z).namelist())
    for f in ("summary.json", "apt/info.json", "apt/L1/input.npz", "apt/L1/slices.npz", "apt/L1/heat.npz",
              "apt/L1/low.npz", "apt/L1/stats.json", "apt/L1/heat.png"):
        assert f in names, f
    d = tmp_path / "calib" / "apt" / "L1"
    x = np.load(d / "input.npz")["input"]
    assert x.shape[0] == 8 and x.dtype == np.float16
    sl = np.load(d / "slices.npz")
    occ = np.unpackbits(sl["occ_bits"], axis=-1)[..., :sl["shape"][2]]
    assert occ.shape == tuple(sl["shape"]) and occ.any()
    st = json.loads((d / "stats.json").read_text(encoding="utf-8"))
    assert st["slices"] and 0 <= st["noise_share_of_labelled"] <= 1 and st["wall_line_width_m"] > 0
    assert "input_channels" in st and "low_slice" in st
    res = compare_calib({"a": [tmp_path / "calib"], "b": [tmp_path / "calib"]}, tmp_path / "cmp.png")
    assert res["a"]["_n"] == 1 and (tmp_path / "cmp.png").exists()
    assert res["a"]["f1 низ"][0] == res["b"]["f1 низ"][0]
