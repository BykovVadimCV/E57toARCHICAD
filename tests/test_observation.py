"""Guard against teaching the network to invent unobserved geometry."""
import json
import numpy as np
import cv2
import pytest

from simscan.observation import (build_observation_masks, OBSERVATION_CONTRACT,
                                OBSERVATION_FILES)
from simscan.hd_evidence import SCAN_HD_FILES


def scene(shape=(60, 90)):
    layers = {x: np.zeros(shape, np.uint8) for x in SCAN_HD_FILES}
    frame = dict(origin_x=0., origin_y=0., width=shape[1], height=shape[0], pixel_m=.002)
    wall = np.zeros(shape, np.uint8)
    wall[10:21, 10:65] = 255
    openings = np.zeros(shape, np.uint8)
    openings[13:18, 33:42] = 1
    wall[openings != 0] = 0
    void = np.zeros(shape, np.uint8)
    valid = np.full(shape, 255, np.uint8)
    return layers, frame, wall, openings, void, valid


def test_unobserved_wall_is_not_training_positive_even_if_gt_exists():
    images, meta = build_observation_masks(*scene())
    assert meta['format'] == OBSERVATION_CONTRACT
    assert meta['physical_positive_pixels'] > 0
    assert meta['supported_positive_pixels'] == 0
    assert not images['training_valid_hd.png'][15, 12]  # hidden wall
    assert not images['training_valid_hd.png'][15, 35]  # hidden opening
    assert images['training_valid_hd.png'][25, 75] == 255  # retains old outside/background penalty
    assert not images['observation_occupied_hd.png'].any()
    assert not images['observation_support_hd.png'].any()
    assert not images['observation_free_hd.png'].any()  # no invented free rays
    assert images['observation_unknown_hd.png'].all()  # no evidence anywhere
    assert images['observation_uncertain_hd.png'][15, 12] == 255
    assert meta['free_rays_available'] is False


def test_raw_return_support_only_near_observed_pixels():
    layers, frame, wall, openings, void, valid = scene()
    layers['scan_mid_hd.png'][15, 15] = 255
    images, _ = build_observation_masks(layers, frame, wall, openings, void, valid)
    assert images['observation_occupied_hd.png'][15, 15] == 255
    assert images['observation_unknown_hd.png'][15, 15] == 0
    assert images['observation_unknown_hd.png'][15, 18] == 255
    assert images['observation_occupied_hd.png'][15, 18] == 0  # no fabricated scan returns
    assert images['observation_support_hd.png'][15, 18] == 255
    assert images['training_valid_hd.png'][15, 18] == 255
    assert not images['training_valid_hd.png'][15, 57]
    assert not images['training_valid_hd.png'][15, 37]  # door without evidence
    assert images['observation_free_hd.png'].sum() == 0


def test_floor_or_ceiling_return_does_not_confirm_hidden_wall():
    layers, frame, wall, openings, void, valid = scene()
    layers['scan_floor_hd.png'][15, 15] = 255
    images, _ = build_observation_masks(layers, frame, wall, openings, void, valid)
    assert images['observation_occupied_hd.png'][15, 15] == 255
    assert images['observation_unknown_hd.png'][15, 15] == 0
    assert images['observation_unknown_hd.png'][15, 18] == 255
    assert images['observation_support_hd.png'][15, 15] == 0
    assert images['training_valid_hd.png'][15, 15] == 0


def test_observation_masks_do_not_erase_existing_background_ignore():
    layers, frame, wall, openings, void, valid = scene()
    valid[:, 40:] = 0
    layers['scan_mid_hd.png'][15, 50] = 255
    images, _ = build_observation_masks(layers, frame, wall, openings, void, valid)
    assert images['training_valid_hd.png'][15, 50] == 0
    assert images['training_valid_hd.png'][25, 50] == 0
    assert images['training_valid_hd.png'][25, 5] == 255


def test_missing_ray_origins_cannot_be_replaced_with_synthetic_free_space():
    args = scene()
    with pytest.raises(ValueError, match='3D height-band'):
        build_observation_masks(*args, free_rays=np.zeros(args[2].shape, np.uint8))
    with pytest.raises(ValueError, match='support_radius'):
        build_observation_masks(*args, support_radius_m=.08)


def test_output_keys_and_binary_dtypes_are_stable():
    images, meta = build_observation_masks(*scene())
    assert tuple(images) == OBSERVATION_FILES
    for name, arr in images.items():
        assert arr.shape == (60, 90) and arr.dtype == np.uint8
        assert np.isin(arr, [0, 255]).all()
    assert json.loads(json.dumps(meta))['training_valid_policy'].startswith('legacy_background')


def test_layout_support_reports_each_wall_face_and_both_jambs(tmp_path):
    from simscan.layout import Layout, Room, Wall, Opening
    from simscan.netinput import SceneGT
    scene = tmp_path / 'scene'
    scene.mkdir()
    layout = Layout(2.7, [Room(0, (0, 0, 1.0, 1.0), (0, 0, 1.0, 1.0), ceiling_z=2.7)],
                    [Wall(0, (0.1, 0.1), (0.9, 0.1), 0.1, 'interior')],
                    [Opening(0, 0, 'passage', 0.4, 0.2, 0., 2.4)])
    (scene / 'layout.json').write_text(json.dumps(layout.to_dict()))
    gt = SceneGT(scene, 0)
    frame = dict(origin_x=0., origin_y=0., width=1000, height=500, pixel_m=.002)
    support = np.zeros((500, 1000), np.uint8)
    from simscan.observation import audit_object_support
    empty = audit_object_support(gt, frame, support)
    assert empty['wall_count'] == 1 and empty['opening_count'] == 1
    assert empty['walls'][0]['face_return_proximity_fraction'] == [0., 0.]
    assert empty['openings'][0]['both_jambs_near_returns'] is False
    # One side of the wall is supported, the other not: don't infer thickness.
    row = frame['height'] - 1 - int((.1 - .05) / .002)
    support[row, 60:460] = 255
    a = audit_object_support(gt, frame, support)
    assert a['walls'][0]['face_return_proximity_fraction'][0] > .7
    assert a['walls'][0]['face_return_proximity_fraction'][1] < .1


def test_dataset_metadata_preserves_native_observation_identity(tmp_path):
    from simscan.netinput import write_meta
    from simscan.observation import OBSERVATION_CONTRACT
    root = tmp_path / 'generated'
    write_meta(root, hd_masks=True)
    doc = json.loads((root / 'meta.json').read_text())
    assert doc['hd_observation_contract'] == OBSERVATION_CONTRACT
    write_meta(root)  # build_scene must not erase the HD identity
    assert json.loads((root / 'meta.json').read_text()) == doc
    (root / 'scenes' / 'a').mkdir(parents=True)
    with pytest.raises(ValueError, match='observation contract changed'):
        write_meta(root, hd_masks=False)
