"""Regression tests for physically supervised voids and native 2-mm evidence."""
import json

import numpy as np

from simscan.hd_evidence import project_scan_bands, align_coarse_mask
from simscan.layout import Layout, Room
from simscan.netinput import SceneGT, label_level, VOID, INTERIOR


class Source:
    def __init__(self, chunks):
        self._chunks = [np.asarray(c, dtype=np.float64) for c in chunks]

    def chunks(self):
        for c in self._chunks:
            yield c


def frame(pixel=.002, width=100, height=100):
    return dict(origin_x=0., origin_y=0., pixel_m=pixel,
                width=width, height=height)


def test_native_returns_are_separate_height_bands_without_invented_points():
    # Five returns at one XY, with one at a second XY.
    p = np.array([[.012, .012, 0.], [.012, .012, .35],
                  [.012, .012, 1.3], [.012, .012, 2.4],
                  [.012, .012, 2.7], [.038, .038, 1.3]])
    bands, stats = project_scan_bands(Source([p[:3], p[3:]]), frame(), 0., 2.7)
    assert stats["total_source_points"] == 6
    assert stats["in_frame_points_all_elevations"] == 6
    assert [np.count_nonzero(a) for a in bands.values()] == [1, 2, 1, 1, 1]
    # Every positive cell must originate from at least one observed return.
    h = frame()["height"]
    observed = np.zeros((h, h), dtype=bool)
    ix = np.floor(p[:, 0] / .002).astype(int)
    iy = h - 1 - np.floor(p[:, 1] / .002).astype(int)
    observed[iy, ix] = True
    assert all(not np.any((v > 0) & ~observed) for v in bands.values())


def test_native_projection_replays_same_thinning_as_low_res():
    rng_seed = 123
    p = np.array([[.002 * (i + .5), .005, 1.1] for i in range(25)])
    source = Source([p[:11], p[11:]])
    bands, stats = project_scan_bands(source, frame(), 0., 2.7,
                                      thin=.45, sampling_seed=rng_seed)
    rng = np.random.default_rng(rng_seed)
    rng.random(11)
    rng.random(14)  # low-res sampling pass
    keep = np.r_[rng.random(11) >= .45, rng.random(14) >= .45]
    assert stats["kept_after_thinning"] == int(keep.sum())
    predicted = np.where(bands["scan_mid_hd.png"][-3] > 0)[0]
    np.testing.assert_array_equal(predicted, np.flatnonzero(keep))


def test_coarse_validity_uses_metric_world_mapping_not_stretched_resize():
    coarse = np.zeros((10, 10), np.uint8)
    coarse[2, 3] = 1
    aligned = align_coarse_mask(coarse, frame(.02, 10, 10),
                                frame(.002, 100, 100))
    assert aligned.shape == (100, 100)
    assert np.count_nonzero(aligned) == 100
    assert aligned[24, 34] == 1
    assert aligned[40, 40] == 0
    # A translated HD origin must translate the valid block exactly.
    translated = dict(frame(.002, 100, 100), origin_x=.02)
    shifted = align_coarse_mask(coarse, frame(.02, 10, 10), translated)
    assert shifted[24, 24] == 1
    assert shifted[24, 34] == 0


def test_solid_stair_is_not_semantic_void_but_upper_slab_hole_is(tmp_path):
    lower = Layout(2.8, [Room(0, (0, 0, 4, 4), (0, 0, 4, 4), ceiling_z=2.8)], [])
    upper = Layout.from_dict(lower.to_dict())
    rect = [1., 1., 2., 2.]
    stair = dict(footprint=rect, hole=rect)
    lower.meta = dict(duplex=dict(stair=stair), slab=dict(ceiling=dict(holes=[rect])))
    upper.meta = dict(duplex=dict(stair=stair), slab=dict(floor=dict(holes=[rect])))
    (tmp_path / "layout.json").write_text(json.dumps({
        "levels": [{"z0": 0, "layout": lower.to_dict()},
                   {"z0": 3.0, "layout": upper.to_dict()}],
    }))
    raster = frame(.02, 200, 200)
    lab0 = label_level(SceneGT(tmp_path, 0), raster)
    lab1 = label_level(SceneGT(tmp_path, 1), raster)
    assert np.count_nonzero(lab0["sem"] == VOID) == 0
    assert np.count_nonzero(lab0["aux"]["stair_footprint"]) > 0
    assert lab0["sem"][125, 75] == INTERIOR
    assert np.count_nonzero(lab1["sem"] == VOID) > 2000
    assert np.count_nonzero(lab1["aux"]["floor_opening"]) > 0
