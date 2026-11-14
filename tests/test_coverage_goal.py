import numpy as np
from simscan.decompose import Config,decompose


def test_higher_coverage_target_includes_an_eligible_tail_beyond_99p5():
    target=np.zeros((150,500),bool);target[20:70,10:410]=True;target[100:104,430:450]=True
    a=decompose(target,config=Config(coverage_stop=.995,axis_weight=0))
    b=decompose(target,config=Config(coverage_stop=.999,axis_weight=0))
    assert .995<=a['metrics']['recall']<.999
    assert b['metrics']['recall']>=.999
    assert b['mask'][100:104,430:450].all()
    assert b['coverage_search']['requested']==.999
    assert b['coverage_search']['target_met']
    assert b['coverage_search']['stop_reason']=='target_reached'


def test_unattainable_target_is_reported_without_painting_ineligible_residuals():
    target=np.zeros((100,180),bool);target[20:30,10:150]=True;target[70:72,160:162]=True
    d=decompose(target,config=Config(coverage_stop=.999,axis_weight=0))
    assert not d['mask'][70:72,160:162].any()
    assert not d['coverage_search']['target_met']
    assert d['coverage_search']['requested']==.999
    assert d['coverage_search']['remaining_pixels']>0
    assert d['coverage_search']['achieved']==d['metrics']['recall']
