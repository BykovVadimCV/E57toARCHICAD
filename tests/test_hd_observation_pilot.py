"""Pilot audit must reject fabricated support and report class retention."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'tools' / 'audit_hd_observation_pilot.py'
spec = importlib.util.spec_from_file_location('audit_hd_observation_pilot', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _level(tmp_path):
    d = tmp_path / 'scene_00000' / 'L1'
    d.mkdir(parents=True)
    shape = (16, 20)
    blank = np.zeros(shape, np.uint8)
    wall = blank.copy(); wall[3:6, 2:7] = 255
    openings = blank.copy(); openings[7:9, 3:7] = 3  # physical passage
    void = blank.copy()
    positive = (wall != 0) | (openings != 0)
    legacy = np.full(shape, 255, np.uint8)
    support = blank.copy(); support[3:6,2:5] = 255
    confirmed = blank.copy(); confirmed[7, 3] = 255
    face=wall.copy(); observed=np.bitwise_and(face,support)
    unresolved = (positive & ~((observed > 0) | (confirmed > 0))).astype(np.uint8) * 255
    training = ((legacy > 0) & ~(unresolved > 0)).astype(np.uint8) * 255
    occ = blank.copy(); occ[2:4, 2:5]=255
    free = blank.copy(); free[10:13, 2:6]=255
    unknown = (255 - np.bitwise_or(occ, free)).astype(np.uint8)
    images={
        'wall_mask_hd.png':wall, 'openings_hd.png':openings, 'floor_void_hd.png':void,
        'supervision_valid_hd.png':legacy, 'training_valid_hd.png':training,
        'observation_support_hd.png':support, 'observation_occupied_hd.png':occ,
        'observation_free_hd.png':free, 'observation_unknown_hd.png':unknown,
        'observation_uncertain_hd.png':unresolved,
        'observation_confirmed_passage_hd.png':confirmed,
        'wall_face_candidates_hd.png':face, 'observed_wall_surface_hd.png':observed,
    }
    for name,arr in images.items():
        assert cv2.imwrite(str(d/name),arr)
    (d/'wall_mask_hd.json').write_text(json.dumps({'frame':{'width':20,'height':16,'pixel_m':.002}}))
    obs={'format':'native-hd-observation-v3', 'ray_provenance':{'available':True,'ray_count':2},
         'observed_return_pixels':int((occ!=0).sum()),
         'return_proximity_pixels':int((support!=0).sum()),
         'unsupported_positive_pixels':int((unresolved!=0).sum()),
         'training_valid_pixels':int((training!=0).sum()),
         'confirmed_passage_pixels':int((confirmed!=0).sum()),
         'wall_face_candidate_pixels':int((face!=0).sum()),
         'observed_wall_surface_pixels':int((observed!=0).sum()),
         'free_ray_pixels':int((free!=0).sum()),
         'objects':{'wall_count':0,'opening_count':0,'walls':[],'openings':[],
                    'passage_ray_crossings':{'confirmed_count':1,
                     'openings':[{'confirmed_for_supervision':True, 'physical_area_px':8,
                                  'confirmed_area_px':1,'max_area_fraction':.20,
                                  'representative_ray_offsets_m':[0.,.04]}]}}}
    (d/'scan_evidence_hd.json').write_text(json.dumps({'observation':obs}))
    return d


def test_audit_reports_retention_and_confirmed_passage(tmp_path):
    d=_level(tmp_path)
    row=module.inspect_level(d)
    assert row['classes']['wall']['physical_px']==15
    assert row['classes']['wall']['trained_px']==9
    assert row['classes']['passage']['physical_px']==8
    assert row['classes']['passage']['trained_px']==1
    assert row['metrics']['unsupported_positive']==13
    assert row['metrics']['negative_kept']==row['metrics']['negative_legacy']
    assert row['passage_objects_confirmed']==1
    s=module.summarize([row],expected_scenes=1)
    assert s['scene_count']==1
    assert s['class_retention']['passage']['trained_fraction']==0.125
    assert s['wall_face_training_coverage']==0.6
    assert s['negative_background_retention']==1.0


def test_audit_rejects_unsupported_targets_used_for_training(tmp_path):
    d=_level(tmp_path)
    img=cv2.imread(str(d/'training_valid_hd.png'),cv2.IMREAD_UNCHANGED)
    img[3,6]=255
    cv2.imwrite(str(d/'training_valid_hd.png'),img)
    with pytest.raises(ValueError,match='Unsupported positive'):
        module.inspect_level(d)


def test_audit_rejects_fabricated_passage_proof(tmp_path):
    d=_level(tmp_path)
    img=cv2.imread(str(d/'observation_confirmed_passage_hd.png'),cv2.IMREAD_UNCHANGED)
    img[1,1]=255
    cv2.imwrite(str(d/'observation_confirmed_passage_hd.png'),img)
    with pytest.raises(ValueError,match='Confirmed passage'):
        module.inspect_level(d)


def test_audit_refuses_unknown_equals_free(tmp_path):
    d=_level(tmp_path)
    img=cv2.imread(str(d/'observation_unknown_hd.png'),cv2.IMREAD_UNCHANGED)
    img[:]=0
    cv2.imwrite(str(d/'observation_unknown_hd.png'),img)
    with pytest.raises(ValueError,match='Unknown mask'):
        module.inspect_level(d)


def test_quality_gate_flags_wall_starvation_and_passage_overcoverage(tmp_path):
    row=module.inspect_level(_level(tmp_path))
    row['metrics']['positive']=100
    row['metrics']['positive_kept']=6
    row['metrics']['confirmed_passage']=9
    row['classes']['wall']['physical_px']=90
    row['classes']['wall']['face_candidate_legacy_valid_px']=90
    row['classes']['wall']['surface_trained_px']=5
    row['classes']['wall']['trained_px']=5
    row['classes']['passage']['physical_px']=10
    summary=module.summarize([row],expected_scenes=1)
    assert summary['quality_gate']['approved_for_training'] is False
    assert summary['confirmed_passage_area_fraction']==0.9
    assert any('Low measured wall-face coverage' in msg for msg in summary['warnings'])
    assert any('Passage ray-confirmed pixels cover' in msg for msg in summary['warnings'])
    assert any('No floor-void targets' in msg for msg in summary['warnings'])


def test_audit_rejects_oversized_per_opening_confirmation_even_if_raster_small(tmp_path):
    d=_level(tmp_path)
    evidence=json.loads((d/'scan_evidence_hd.json').read_text())
    p=evidence['observation']['objects']['passage_ray_crossings']['openings'][0]
    p['confirmed_area_px']=5  # 5 / 8 exceeds strict per-opening cap
    (d/'scan_evidence_hd.json').write_text(json.dumps(evidence))
    with pytest.raises(ValueError,match='per-opening proof area'):
        module.inspect_level(d)


def test_audit_rejects_wall_surface_fabricated_without_returns(tmp_path):
    d=_level(tmp_path)
    surf=cv2.imread(str(d/'observed_wall_surface_hd.png'),cv2.IMREAD_UNCHANGED)
    surf[3,6]=255  # no return proximity at this face
    cv2.imwrite(str(d/'observed_wall_surface_hd.png'),surf)
    with pytest.raises(ValueError,match='Fabricated wall surface'):
        module.inspect_level(d)
