"""make-dataset: сцена -> датасет без скана, продолжение с места обрыва, режим «под отделку», zip."""
import json
import zipfile

from simscan.config import load_config
from simscan.makedata import make_dataset, pack, rebuild_splits


def test_make_dataset_resume_and_pack(tmp_path):
    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5}, "interior": {"p_bare": 1.0}})
    root = tmp_path / "ds"
    r = make_dataset(cfg, root, count=1, seed=5, log=lambda *a: None)
    assert r["ok"] == 1 and r["errors"] == 0 and sum(r["splits"].values()) == 1
    d = root / "scenes" / "scene_00000" / "L1"
    for f in ("input.npy", "sem.png", "dist.npy", "orient.npy", "heights.npy", "scene.json"):
        assert (d / f).exists(), f
    assert not (root / "_work").exists()                       # скан удалён
    sc = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    assert sc["bare"] is True and "thin" in sc["aug"]
    r2 = make_dataset(cfg, root, count=1, seed=5, log=lambda *a: None)
    assert r2["ok"] == 0                                        # готовая сцена пропущена
    assert rebuild_splits(root) == r["splits"]
    names = zipfile.ZipFile(pack(root, tmp_path / "ds.zip")).namelist()
    assert "ds/meta.json" in names and "ds/scenes/scene_00000/L1/input.npy" in names
    assert not any(n.endswith("preview.png") for n in names)


def test_memory_source_equals_e57(tmp_path):
    """Растр из памяти (без scan.e57) - тот же, что из файла (с точностью до округления)."""
    import numpy as np

    from simscan.generate import generate_scene
    from simscan.netinput import MemorySource, build_scene

    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5},
                                 "export": {"preview": False, "free_space_input": False, "unet_mask": False}})
    generate_scene(cfg, tmp_path / "a", seed=2, index=0)
    r = generate_scene(cfg, tmp_path / "b", seed=2, index=0, write_scan=False)
    assert not (tmp_path / "b" / "scan.e57").exists()
    da = build_scene(tmp_path / "a", tmp_path / "dsa", log=lambda *a: None)
    db = build_scene(tmp_path / "b", tmp_path / "dsb", source=MemorySource(*r["_scans"]), log=lambda *a: None)
    xa, xb = (np.load(f"{d[0]}/input.npy").astype(np.float32) for d in (da, db))
    assert xa.shape == xb.shape
    diff = np.abs(xa - xb).max(0) > 1e-3                 # округление на границах ячеек - единицы пикселей
    assert diff.mean() < 1e-3, diff.sum()


def test_variants(tmp_path):
    import json

    cfg = load_config(overrides={"scanner": {"angular_step_deg": 0.5}})
    r = make_dataset(cfg, tmp_path / "ds", count=1, seed=5, variants=2, log=lambda *a: None)
    assert r["ok"] == 1 and sum(r["splits"].values()) == 2
    sc = json.loads((tmp_path / "ds" / "scenes" / "scene_00000_v1" / "L1" / "scene.json").read_text(encoding="utf-8"))
    assert sc["aug"]["rot_deg"] > 0 and sc["layout_key"] == "5:0"
    assert make_dataset(cfg, tmp_path / "ds", count=1, seed=5, variants=2, log=lambda *a: None)["ok"] == 0
