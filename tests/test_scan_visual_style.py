"""The synthetic point-cloud preview must preserve the extractor's display curve."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.preview_scan_hd import extractor_grayscale, render_extractor_view


def test_log_density_and_marker_match_raw_extractor():
    counts = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 0, 4, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint32)
    positive = counts[counts > 0]
    peak = max(2.0, float(np.percentile(positive, 99)))
    expected = np.minimum(
        245, np.log1p(counts) / np.log1p(peak) * 185
        + np.where(counts > 0, 50, 0)
    ).astype("uint8")
    np.testing.assert_array_equal(extractor_grayscale(counts, 1), expected)
    np.testing.assert_array_equal(
        extractor_grayscale(counts, 3),
        cv2.dilate(expected, np.ones((3, 3), dtype=np.uint8))
    )


def test_framed_density_map_uses_true_scan_xyz(tmp_path: Path):
    # Dense return cluster, sparse wall-like line, outlier and floor/ceiling.
    xyz = np.array(
        [[x, 0., 1.] for x in np.linspace(0., 5., 200)]
        + [[2., 2., 1.]] * 30
        + [[1., 1., 0.]] * 10
        + [[3., 1., 3.]] * 10
        + [[20., 20., 1.]],
        dtype=float,
    )
    result = render_extractor_view(
        [{"xyz": xyz}], tmp_path, floor=0., ceil=3.,
        width=320, height=180, margin=10, marker_px=3
    )
    image = cv2.imread(str(tmp_path / "scan_extractor_4k.png"), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (180, 320)
    assert result["points_rendered"] >= 200
    assert result["marker_px"] == 3
    assert result["z_range_m"] == pytest.approx([0.15, 2.85])
    assert result["effective_visual_pixel_m"] > 0
    assert image.max() > 0
    assert np.count_nonzero(image) > 100


def test_empty_histogram_fails():
    with pytest.raises(ValueError, match="No projected"):
        extractor_grayscale(np.zeros((5, 5), dtype=np.uint32))
