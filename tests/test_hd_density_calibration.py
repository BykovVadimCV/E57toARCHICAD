"""Density calibration must not silently use GT or invented free-ray evidence."""
from __future__ import annotations

import numpy as np
import pytest

from tools.calibrate_hd_density import evidence_metrics, distribution, compare, make_report


def _dummy(c, kind="synthetic", source_root="angle_0p1", thin=0.0):
    frame = {"width": c.shape[1], "height": c.shape[0], "pixel_m": 0.002}
    v = evidence_metrics(c, frame, {"thin": thin}, "fixture", kind)
    v["source_root"] = source_root
    v["station_count"] = 1
    return v


def test_native_counts_and_20mm_aggregation_are_metric():
    c = np.zeros((20, 30), np.uint32)
    c[2, 4] = 2
    c[9, 8] = 1
    c[14, 14] = 1
    v = _dummy(c)
    assert v["structural_returns"] == 4
    assert v["occupied_2mm_cells"] == 3
    assert v["occupied_20mm_fraction"] == pytest.approx(2 / 6)
    assert v["multi_return_fraction_among_occupied_2mm"] == pytest.approx(1 / 3)
    assert v["occupied_2mm_cells_per_frame_m2"] == pytest.approx(3 / (20 * 30 * .002**2), abs=.001)


def test_only_unthinned_rows_are_compared():
    a = _dummy(np.ones((10, 10), dtype=np.uint32))
    b = _dummy(np.zeros((10, 10), dtype=np.uint32), thin=0.3)
    assert distribution([a, b])["n_levels"] == 1
    report = compare([a], [a, b])
    assert report["candidates"]["angle_0p1"]["density"]["n_levels"] == 1
    assert "Реальные E57" in make_report(report)


def test_invalid_pitch_count_or_array_rejected():
    frame = {"width": 10, "height": 10, "pixel_m": .003}
    with pytest.raises(ValueError):
        evidence_metrics(np.ones((10, 10), np.uint32), frame, {}, "x", "real")
    frame["pixel_m"] = .002
    with pytest.raises(ValueError):
        evidence_metrics(np.ones((10, 9), np.uint32), frame, {}, "x", "real")
    with pytest.raises(ValueError):
        evidence_metrics(np.ones((10, 10), np.float32), frame, {}, "x", "real")


def test_merged_scan_does_not_identify_angular_step():
    a = _dummy(np.ones((10, 10), dtype=np.uint32))
    report = compare([a], [a])
    assert any("NOT per-station angular" in x for x in report["limitations"])
    assert "angular_step_deg" not in report


def test_real_references_without_synthetic_thinning_are_accepted():
    reference = _dummy(np.ones((10, 10), dtype=np.uint32), kind="real")
    del reference["synthetic_thinning"]
    candidate = _dummy(np.ones((10, 10), dtype=np.uint32))
    assert compare([reference], [candidate])["real"]["n_levels"] == 1


def test_repository_real_reference_is_loadable_and_has_all_metrics():
    from pathlib import Path
    import json
    file = Path(__file__).parents[1] / "configs" / "calibration_real_e57_observed.json"
    levels = json.loads(file.read_text(encoding="utf-8"))["levels"]
    assert len(levels) == 3
    assert distribution(levels)["metrics"]["occupied_20mm_fraction"]["median"] > 0


def test_angular_only_calibration_gate_requires_both_metrics():
    base = _dummy(np.ones((10, 10), dtype=np.uint32))
    candidate = _dummy(np.full((10, 10), 5, dtype=np.uint32))
    result = compare([base], [candidate])
    assert result["any_angle_matches_both_primary_density_metrics"] is False
