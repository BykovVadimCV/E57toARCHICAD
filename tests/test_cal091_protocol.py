"""Predeclared paired-design gates for CAL-09.1 (no scanner generation)."""
import pytest
from tools.validate_cal091_ground_truth import (SEEDS, VARIANTS, RESOLUTIONS_M, summarize)


def fixture_rows():
    rows = []
    for seed in SEEDS:
        for step in RESOLUTIONS_M:
            for variant in VARIANTS:
                zone = 'zone08' if 'zone08' in variant else 'zone18'
                fine = 'fine35' if 'fine35' in variant else 'fine65'
                rows.append({
                    'seed':seed, 'variant': variant, 'tessellation_m': step,
                    'layout_hash':f'{seed}', 'mesh_vertex_hash':f'{seed}_{step}',
                    'envelope_hash': 'zero' if variant=='uniform_2_4' else f'{seed}_{step}_{zone}',
                    'fine_field_hash': None if variant=='uniform_2_4' else f'{seed}_{step}_{fine}',
                    'metrics':{'outside_support_rms_mm':0.,
                               'defect_area_fraction':0. if variant=='uniform_2_4' else .10,
                               'vertical_wall_area_m2':100.,
                               'largest_component_area_m2':0. if variant=='uniform_2_4' else 1.,
                               'fragmentation_index':None if variant=='uniform_2_4' else .25}
                })
    return rows


def test_complete_fixed_design_can_pass_algorithmic_gates():
    d=summarize(fixture_rows())
    assert d['algorithmic_gates_pass']
    assert d['auto_update_production_config'] is False
    assert len(d['scene_metrics'])==120


def test_detects_independence_and_geometry_violation():
    rows=fixture_rows()
    for row in rows:
        if row['variant']=='zone08_fine65' and row['seed']==SEEDS[0] and row['tessellation_m']==.1:
            row['envelope_hash']='changed'
    assert not summarize(rows)['checks']['fine_does_not_move_coarse_mask']
    rows=fixture_rows()
    rows[0]['mesh_vertex_hash']='mismatch'
    assert not summarize(rows)['checks']['all_paired_geometry_hashes_match']


def test_detects_bad_resolution_convergence_or_missing_row():
    rows=fixture_rows()
    for row in rows:
        if row['tessellation_m']==.15 and row['variant']=='zone18_fine65':
            row['metrics']['defect_area_fraction']=.21
    assert not summarize(rows)['checks']['area_convergence_6_of_8_per_variant']
    with pytest.raises(ValueError):summarize(rows[:-1])
