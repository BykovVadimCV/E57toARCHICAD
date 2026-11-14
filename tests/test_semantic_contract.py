"""PR-01: one physical opening meaning in layout, coarse labels, HD and vectors."""
import json

import numpy as np
import pytest

from simscan.hddata import load_patch
from simscan.hdmask import render_hd_masks
from simscan.layout import Layout, Opening, Room, Wall
from simscan.netinput import (DOOR, OPENING, WINDOW, SceneGT, label_level,
                              meta_doc, vector_doc, world_to_px, write_meta)
from simscan.semantic_contract import (
    COARSE_LABELS_VERSION, COARSE_META_VERSION, HD_MASK_FORMAT,
    HD_OPENING_VALUES, HD_SEMANTIC_CLASSES, OPENING_SEMANTICS_VERSION,
    coarse_opening_class, hd_opening_value, vector_opening_type,
)


def geometry(tmp_path, kind, z1):
    lay = Layout(2.7,
                 [Room(0, (0, 0, 5, 5), (0, 0, 5, 5), ceiling_z=2.7)],
                 [Wall(0, (1, 2), (4, 2), .2, 'interior')],
                 [Opening(0, 0, kind, 2.5, .8, 0.8 if kind == 'window' else 0.0, z1)])
    lay.save_json(tmp_path / 'layout.json')
    return SceneGT(tmp_path)


@pytest.mark.parametrize('kind,expected_coarse,expected_hd', [
    ('door', DOOR, HD_OPENING_VALUES['door']),
    ('window', WINDOW, HD_OPENING_VALUES['window']),
    ('passage', OPENING, HD_OPENING_VALUES['passage']),
])
@pytest.mark.parametrize('head', [2.1, 2.68])
def test_physical_kind_not_height_controls_both_rasters_and_vector(tmp_path, kind, expected_coarse, expected_hd, head):
    gt = geometry(tmp_path, kind, head)
    low = {'origin_x': 0., 'origin_y': 0., 'pixel_m': .02,
           'width': 250, 'height': 250}
    lab = label_level(gt, low)
    j, i = np.rint(world_to_px([[3.5, 2.0]], low)[0]).astype(int)
    assert lab['sem'][i, j] == expected_coarse
    assert lab['heights'][1, i, j] == pytest.approx(head / 2.7, abs=.001)

    frame, wall, opening, _ = render_hd_masks(gt, low, 10.0, (32, 32))
    jh, ih = np.rint(world_to_px([[3.5, 2.0]], frame)[0]).astype(int)
    assert opening[ih, jh] == expected_hd
    assert wall[ih, jh] == 0
    assert HD_SEMANTIC_CLASSES[kind] == expected_hd + 1

    entry = vector_doc(gt)['openings'][0]
    assert entry['physical_kind'] == kind
    assert entry['type'] == ('opening' if kind == 'passage' else kind)
    assert entry['top_z'] == pytest.approx(head)


def test_closed_door_and_leafless_passage_same_height_get_distinct_classes(tmp_path):
    door = geometry(tmp_path, 'door', 2.1)
    original = label_level(door, {'origin_x': 0., 'origin_y': 0., 'pixel_m': .02,
                                  'width': 250, 'height': 250})['sem']
    passage = geometry(tmp_path, 'passage', 2.1)
    changed = label_level(passage, {'origin_x': 0., 'origin_y': 0., 'pixel_m': .02,
                                    'width': 250, 'height': 250})['sem']
    assert np.count_nonzero(original == DOOR) > 0
    assert np.count_nonzero(changed == OPENING) > 0
    assert np.count_nonzero(changed == DOOR) == 0


@pytest.mark.parametrize('value', ['archway', '', 'DOOR', None])
def test_unknown_physical_kinds_refused(value):
    for f in (coarse_opening_class, hd_opening_value, vector_opening_type):
        with pytest.raises(ValueError, match='Unknown physical opening kind'):
            f(value)


def test_new_metadata_does_not_silently_mix_with_legacy(tmp_path):
    new = meta_doc()
    assert new['labels'] == COARSE_LABELS_VERSION
    assert new['version'] == COARSE_META_VERSION
    assert new['opening_semantics'] == OPENING_SEMANTICS_VERSION
    out = tmp_path / 'data'
    (out / 'scenes' / 'existing').mkdir(parents=True)
    (out / 'meta.json').write_text(json.dumps(dict(version=3, labels='labels-5',
                                                   floor_void_policy=new['floor_void_policy'])))
    with pytest.raises(ValueError, match='Label/physical floor-void contract changed'):
        write_meta(out)


def test_hd_loader_rejects_pre_fix_classification(tmp_path):
    from test_hd_training_data import tiny_level
    tiny_level(tmp_path)
    m = tmp_path / 'wall_mask_hd.json'
    data = json.loads(m.read_text())
    data['format'] = 'simscan-native-hd-masks-v2'
    m.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Incompatible native HD opening semantics'):
        load_patch(tmp_path, 0, 0, 64)


def test_open_door_remains_door_when_leaf_rotates(tmp_path):
    fr = {'origin_x': 0., 'origin_y': 0., 'pixel_m': .02,
          'width': 250, 'height': 250}
    gt = geometry(tmp_path, 'door', 2.1)
    gt.layout.openings[0].angle_deg = 80.0
    i, j = 150, 175  # midpoint near world x=3.5 m, y=2.0 m
    assert label_level(gt, fr)['sem'][i, j] == DOOR


def test_partial_merger_reads_both_versions_without_mixing(tmp_path):
    from tools.merge_partial_datasets import contract
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'supervision_policy.json').write_text(json.dumps({'unseen_policy': 'ignore'}))
    new = meta_doc()
    (root / 'meta.json').write_text(json.dumps(new))
    new_contract = contract(root)
    assert new_contract['opening_semantics'] == OPENING_SEMANTICS_VERSION
    legacy = dict(new, version=3, labels='labels-5')
    del legacy['opening_semantics']
    (root / 'meta.json').write_text(json.dumps(legacy))
    assert contract(root)['opening_semantics'] is None
    assert contract(root) != new_contract
