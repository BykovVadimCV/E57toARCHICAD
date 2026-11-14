"""3D direct-ray provenance: no phantom vacancy, and no GT-derived beam paths."""
import json
from types import SimpleNamespace

import numpy as np
import cv2
import pytest

from simscan.ray_evidence import (
    RaySamples, capture_direct_returns, thinning_keep_mask,
    render_direct_ray_bands, audit_passage_crossings, RAY_BANDS)
from simscan.observation import build_observation_masks
from simscan.hd_evidence import SCAN_HD_FILES


def _rays(origins, ends, total=None):
    origins = np.asarray(origins, float).reshape(-1, 3)
    ends = np.asarray(ends, float).reshape(-1, 3)
    count = len(origins)
    return RaySamples(origins, ends, np.arange(count, dtype=np.int64),
                      count if total is None else total, 1)


def _frame():
    return dict(origin_x=0., origin_y=0., pixel_m=.01, width=250, height=250)


def _fake_scan(valid, direct, xyz=None):
    valid, direct = np.asarray(valid, bool), np.asarray(direct, bool)
    if xyz is None:
        xyz = np.array([[float(i + 1), 0.2, 1.] for i in range(len(valid))])
    return SimpleNamespace(valid=valid, straight_valid=direct,
                           virtual=np.zeros(len(valid), bool),
                           mixed=np.zeros(len(valid), bool),
                           xyz_local=np.asarray(xyz, float))


def test_capture_rejects_missing_provenance_and_excludes_non_direct():
    scan = _fake_scan([1, 1, 0, 1, 1], [1, 0, 1, 1, 0])
    pose = (np.eye(3), np.array([10., 11., 12.]))
    rays = capture_direct_returns([scan], [pose], max_per_station=6)
    assert rays.total_valid == 4
    assert rays.valid_indices.tolist() == [0, 2]
    np.testing.assert_allclose(rays.origins, [[10., 11., 12.]] * 2)
    np.testing.assert_allclose(rays.endpoints[:, 0], [11., 14.])
    scan.straight_valid = None
    with pytest.raises(ValueError, match='straight-path'):
        capture_direct_returns([scan], [pose])


def test_ray_thinning_replays_the_same_return_index_rng_stream():
    n = 31
    rays = _rays([[0, 0, 0]] * n, [[1, 0, 1]] * n)
    rng = np.random.default_rng(41)
    rng.random(n)  # first rasterizer level-estimate pass
    expected = rng.random(n) >= 0.4
    np.testing.assert_array_equal(thinning_keep_mask(rays, 0.4, 41, chunk_size=7), expected)
    assert thinning_keep_mask(rays, 0, None).all()
    with pytest.raises(ValueError, match='seed'):
        thinning_keep_mask(rays, 0.2, None)


def test_ray_clipping_by_3d_height_and_endpoint_guard():
    rays = _rays([[.2, .5, 1.1]], [[2., .5, 1.1]])
    frame = _frame()
    images, meta, kept = render_direct_ray_bands(rays, frame, 0., 2.7)
    assert meta['available'] and kept is not None
    assert not images[RAY_BANDS[0]].any()
    assert images[RAY_BANDS[1]][199, 100] == 255  # ~1m, 1.1m altitude
    assert not images[RAY_BANDS[2]].any()
    assert images[RAY_BANDS[1]][199, 196] == 0  # guard before measured return
    assert images[RAY_BANDS[1]][199, 210] == 0  # past return


def test_ray_rotation_moves_ray_and_preserves_height():
    rays = _rays([[1., 1., 1.]], [[1.8, 1., 1.]])
    frame = _frame()
    images, _, _ = render_direct_ray_bands(rays, frame, 0., 2.7,
                                            rotation=(90., 1., 1.))
    assert images[RAY_BANDS[1]][100, 100] == 255  # 0.5m near y=1.5 on rotated line
    assert images[RAY_BANDS[1]][149, 150] == 0


def test_no_return_no_rays_does_not_invent_free_space():
    scan = _fake_scan([0, 0, 0], [0, 0, 0])
    rays = capture_direct_returns([scan], [(np.eye(3), np.zeros(3))])
    bands, info, _ = render_direct_ray_bands(rays, _frame(), 0., 2.7)
    assert info['available']
    assert info['ray_count'] == 0
    assert all(not arr.any() for arr in bands.values())
    zeros, absent, _ = render_direct_ray_bands(None, _frame(), 0., 2.7)
    assert not absent['available']
    assert all(not x.any() for x in zeros.values())


def test_3d_free_path_does_not_make_background_or_confirm_invisible_wall():
    frame = _frame()
    sh = (frame['height'], frame['width'])
    layers = {name:np.zeros(sh, np.uint8) for name in SCAN_HD_FILES}
    wall = np.zeros(sh, np.uint8)
    wall[120, 80:150] = 255
    zeros = np.zeros(sh, np.uint8)
    valid = np.full(sh, 255, np.uint8)
    ray = _rays([[.2, .5, 1.1]], [[2., .5, 1.1]])
    bands, meta, _ = render_direct_ray_bands(ray, frame, 0., 2.7)
    images, summary = build_observation_masks(
        layers, frame, wall, zeros, zeros, valid, ray_bands=bands, ray_meta=meta)
    assert images['observation_free_hd.png'][199, 100] == 255
    assert images['observation_unknown_hd.png'][199, 100] == 0
    assert images['training_valid_hd.png'][120, 100] == 0
    assert images['training_valid_hd.png'][199, 100] == 255  # original background only
    assert summary['free_rays_available']


def test_provenance_rejects_unbacked_free_band():
    frame = _frame();sh = (frame['height'], frame['width'])
    layers = {name:np.zeros(sh, np.uint8) for name in SCAN_HD_FILES}
    zeros=np.zeros(sh, np.uint8)
    bands={name:zeros.copy() for name in RAY_BANDS};bands[RAY_BANDS[1]][120,120]=255
    with pytest.raises(ValueError, match='provenance'):
        build_observation_masks(layers,frame,zeros,zeros,zeros,zeros,
            ray_bands=bands,ray_meta={'format':'station-direct-returns-height-bands-v1','available':False})


def test_passage_confirmation_requires_two_jambs_and_cross_wall_rays(tmp_path):
    from simscan.layout import Layout, Room, Wall, Opening
    from simscan.netinput import SceneGT
    from simscan.hdmask import render_hd_masks
    scene=tmp_path/'scene';scene.mkdir()
    layout=Layout(2.7,[Room(0,(0.,0.,2.,2.),(0.,0.,2.,2.),ceiling_z=2.7)],
                  [Wall(0,(.1,1.),(1.9,1.),.2,'interior')],
                  [Opening(0,0,'passage',.9,.5,0.,2.4)])
    (scene/'layout.json').write_text(json.dumps(layout.to_dict()))
    gt=SceneGT(scene,0)
    base=dict(origin_x=0.,origin_y=0.,pixel_m=.02,width=100,height=100)
    frame,_,openings,_=render_hd_masks(gt,base,10.,(100,100))
    support=np.zeros_like(openings)
    wall=gt.layout.walls[0];op=gt.layout.openings[0]
    for sign in (-1,1):
        for off in (-wall.thickness/2,wall.thickness/2):
            world=gt.xy(np.asarray([wall.point(op.s + sign*(op.width/2+.04),off)]))[0]
            x=int((world[0]-frame['origin_x'])/frame['pixel_m'])
            y=frame['height']-1-int((world[1]-frame['origin_y'])/frame['pixel_m'])
            cv2.circle(support,(x,y),3,255,-1)
    # Full wall traversal: measured endpoints beyond the opposite face.
    rays=_rays([[x,.65,1.1] for x in (.88,.99,1.1)],
               [[x,1.4,1.1] for x in (.88,.99,1.1)])
    confirmed,stats=audit_passage_crossings(gt,rays,frame,support,openings,0.)
    assert stats['confirmed_count'] == 1
    assert stats['openings'][0]['crossing_direct_ray_samples'] == 3
    assert confirmed.any() and np.all(openings[confirmed>0]==3)
    assert np.count_nonzero(confirmed) < np.count_nonzero(openings == 3)  # no entire-GT leakage
    no_jamb,stats=audit_passage_crossings(gt,rays,frame,np.zeros_like(support),openings,0.)
    assert stats['confirmed_count']==0 and not no_jamb.any()
    above=_rays([[x,.65,2.55] for x in (.88,.99,1.1)],
                [[x,1.4,2.55] for x in (.88,.99,1.1)])
    high,stats=audit_passage_crossings(gt,above,frame,support,openings,0.)
    assert stats['confirmed_count']==0 and not high.any()
    duplicates=_rays([[.99,.65,1.1]] * 3, [[.99,1.4,1.1]] * 3)
    dup,stats=audit_passage_crossings(gt,duplicates,frame,support,openings,0.)
    assert stats['confirmed_count']==0 and not dup.any()
    stopped=_rays([[x,.65,1.1] for x in (.88,.99,1.1)],
                  [[x,.86,1.1] for x in (.88,.99,1.1)])
    blocked,stats=audit_passage_crossings(gt,stopped,frame,support,openings,0.)
    assert stats['confirmed_count']==0 and not blocked.any()
