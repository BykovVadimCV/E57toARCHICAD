"""Аналитика E57: все шаги проходят без ошибок, архив содержит разделы, числа сходятся с
квартирой (две комнаты, дверь, окно); упавший шаг не роняет остальные."""
import json
import zipfile

import numpy as np

from netfake import write_pred
from simscan.analytics import analytics
from simscan.e57io import write_merged_e57
from test_floorplan import _apartment


def _e57(tmp_path):
    p = _apartment()
    path = tmp_path / "apt.e57"
    write_merged_e57(path, p, np.full(len(p), 0.5, np.float32))
    return path


def test_analytics_archive(tmp_path):
    e57 = _e57(tmp_path)
    z = analytics(e57, tmp_path / "an", pred=write_pred(tmp_path / "pred"), cellplan=False, log=lambda *a: None)
    names = set(zipfile.ZipFile(z).namelist())
    for f in ("SUMMARY.md", "run.json", "log.txt", "file/report.json", "file/sample.npz", "levels/z_hist.png",
              "images/cameras.json", "L1/input.npz", "L1/input.png", "L1/net/net.json", "L1/net/uncertainty.png",
              "L1/vectorize/plan.json", "L1/vectorize/overlay.png", "L1/vectorize/candidates.json",
              "L1/vectorize/stages.json", "L1/vectorize/walls.csv", "L1/vectorize/openings.csv",
              "L1/snap/plan.json", "L1/snap/walls.csv"):
        assert f"apt_analytics/{f}" in names, f
    run = json.loads((tmp_path / "an" / "apt_analytics" / "run.json").read_text(encoding="utf-8"))
    assert run["errors"] == [], run["errors"]
    F = run["facts"]
    assert F["vectorize"]["L1"]["rooms"] == 2 and F["vectorize"]["L1"]["doors"] == 1
    assert F["snap"]["L1"]["snapped"] >= 4
    assert abs(F["levels"][0]["floor_z"] - 10.0) < 0.05
    cands = json.loads((tmp_path / "an" / "apt_analytics" / "L1" / "vectorize" / "candidates.json").read_text(
        encoding="utf-8"))["candidates"]
    assert sum(c["ok"] for c in cands) == 2 and all("verdict" in c for c in cands)
    stages = json.loads((tmp_path / "an" / "apt_analytics" / "L1" / "vectorize" / "stages.json").read_text(
        encoding="utf-8"))["stages"]
    assert stages[0]["stage"] == "skeleton" and stages[-1]["walls"] == F["vectorize"]["L1"]["walls"]


def test_failed_step_does_not_stop_others(tmp_path):
    """Нет чекпойнта модели: шаг сети падает, файл, вход и уровни - в архиве, ошибка - в run.json."""
    e57 = _e57(tmp_path)
    z = analytics(e57, tmp_path / "an", model=tmp_path / "missing.pt", cellplan=False, log=lambda *a: None)
    names = set(zipfile.ZipFile(z).namelist())
    assert "apt_analytics/L1/input.npz" in names and "apt_analytics/file/report.json" in names
    run = json.loads((tmp_path / "an" / "apt_analytics" / "run.json").read_text(encoding="utf-8"))
    assert [e["step"] for e in run["errors"]] == ["net"], run["errors"]
    assert run["environment"]["model"]["missing"]
