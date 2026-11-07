"""Regression checks for native metric HD mask frames and boundaries."""
import numpy as np
import pytest

from simscan.hdmask import make_hd_frame, parse_size, render_hd_masks
from simscan.layout import Layout, Wall, Opening


def test_requested_pitch_and_minimum_canvas():
    base = {"origin_x": 0.0, "origin_y": 0.0, "pixel_m": .02, "width": 800, "height": 600}
    poly = [np.array([[1., 1.], [11., 1.], [11., 7.], [1., 7.]])]
    frame = make_hd_frame(base, poly, 2.0, (2048, 1152))
    assert frame["pixel_m"] == pytest.approx(.002)
    assert frame["width"] >= 5100
    assert frame["height"] >= 3100
    assert frame["width"] * frame["height"] <= 64_000_000


def test_hd_frame_does_not_inherit_cloud_padding():
    base = {"origin_x": -3., "origin_y": -3., "pixel_m": .02, "width": 900, "height": 900}
    poly = [np.array([[0., 0.], [6., 0.], [6., 4.], [0., 4.]])]
    frame = make_hd_frame(base, poly, 2.0)
    assert frame["width"] < 3300
    assert frame["height"] < 2300


def test_hd_frame_refuses_unbounded_memory():
    base = {"origin_x": 0., "origin_y": 0., "pixel_m": .02, "width": 100, "height": 100}
    poly = [np.array([[0., 0.], [40., 0.], [40., 40.], [0., 40.]])]
    with pytest.raises(ValueError, match="exceeds"):
        make_hd_frame(base, poly, 2.0)


def test_invalid_pixel_size_and_dimensions():
    base = {"origin_x": 0., "origin_y": 0., "pixel_m": .02, "width": 50, "height": 50}
    for value in ("1920", "0x1080", "60000x200"):
        with pytest.raises(ValueError):
            parse_size(value)
    with pytest.raises(ValueError):
        make_hd_frame(base, [], 0.0)


def test_native_opening_cut_from_wall_without_resize():
    class GT:
        def __init__(self):
            self.layout = Layout(2.7, [], [
                Wall(0, (0., 0.), (3., 0.), .20, "interior")
            ])
            self.layout.openings = [
                Opening(0, 0, "door", 1.5, .9, 0., 2.1)
            ]
            self.ceiling_h = 2.7
            self.layout.items = []
        def wall_quad(self, wall, s0, s1, o0, o1):
            return np.asarray([wall.point(s0, o0), wall.point(s1, o0),
                               wall.point(s1, o1), wall.point(s0, o1)])
    gt = GT()
    base = {"origin_x": -.5, "origin_y": -.5, "pixel_m": .02, "width": 200, "height": 200}
    frame, wall, openings, edges = render_hd_masks(gt, base, 2., (128, 128))
    assert wall.shape == openings.shape == edges.shape
    assert np.any(wall == 255)
    assert np.any(openings == 1)
    assert np.all(wall[openings != 0] == 0)
