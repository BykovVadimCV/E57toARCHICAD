"""Ordered wall fitting used by both saved-mask replay and new inference.

Keep stages explicit: running all structure flags once does not reproduce the
continuity -> confidence -> shared-face passes used by the reviewed pipeline.
"""
from dataclasses import replace, asdict
import time
import numpy as np
import cv2
from .decompose import Config, decompose, mask_metrics, rectangle_overlap_metrics
from .wall_runs import refine_structure, _union, extract_runs, _profile, _point_distances
from .wall_merge import merge_result
from .wall_simplify import simplify_result
from .wall_face_recovery import recover_scan_faces

PIPELINE_VERSION = 'guarded-short-piers-v13'
STAGE_ORDER = ('00_area_fit', '01_continuity', '02_confidence', '03_shared_faces',
               '04_merge', '05_two_face_simplification', '06_reference_guard',
               '07_scan_face_recovery', '08_final_merge', '09_nonoverlapping_footprints', '10_consistent_thickness', '11_native_reference_guard', '12_short_scan_piers')


def audit(mask, target, barrier, forbidden, pixel_m, contact_m=.005):
    depth=cv2.distanceTransform(np.pad(mask.astype('uint8'),1),cv2.DIST_L2,5)[1:-1,1:-1]*pixel_m
    checks=dict(excluded_pixels_filled=int((mask&forbidden).sum()) if forbidden is not None else 0,
                dense_barrier_crossing_pixels=int((mask&~target&barrier&(depth>contact_m)).sum()))
    if any(checks.values()): raise ValueError(f'Wall pipeline constraint violation: {checks}')
    return checks


def finish_walls(result, area, target, barrier, pixel_m=.0025, *, probability=None,
                 forbidden=None, config=None, historical_replay=False, on_stage=None, reference_result=None, density=None, accepted_reference=None):
    """Guard the staged fit against a same-input combined reference, then recover faces."""
    cfg=config or Config(max_primitives=220)
    start=time.monotonic()
    if historical_replay:
        result['regression_guard']=dict(enabled=False,reason='exact historical postprocessing replay')
    else:
        # Preserve the complete previous entry point, including its pre-area
        # confidence filtering. Reusing the unfiltered staged area fit here is
        # not equivalent and can miss regressions caused by proposal changes.
        reference=reference_result
        if reference is None:
            reference=decompose(target,pixel_m,semantic_probability=probability,
                density=density,barrier=barrier,forbidden=forbidden,config=cfg)
        if np.asarray(reference['mask']).shape!=target.shape:raise ValueError('Reference frame differs')
        audit(reference['mask'],target,barrier,forbidden,pixel_m,cfg.barrier_contact_m)
        profiles=[_profile(r,target,forbidden,pixel_m,cfg) for r in extract_runs(target,barrier,pixel_m,cfg,forbidden)]
        lost=0
        for pr in profiles:
            before=_point_distances(reference['mask'],pr['points'],pixel_m)
            after=_point_distances(result['mask'],pr['points'],pixel_m)
            lost+=int((pr['valid']&(before<=cfg.run_face_tolerance_m)&(after>cfg.run_face_tolerance_m)).sum())
        checks=dict(enabled=True,lost_reference_samples=lost,
            selected='combined_reference' if lost else 'staged',
            reference_scope='previous complete combined pipeline on the same target, probability, density, exclusions and scan evidence')
        if lost:result=reference
        result['regression_guard']=checks
    if on_stage:on_stage('06_reference_guard',result,start)
    start=time.monotonic()
    if not historical_replay:
        result=recover_scan_faces(result,target,barrier,pixel_m,probability,forbidden,cfg)
    if on_stage:on_stage('07_scan_face_recovery',result,start)
    start=time.monotonic();before=result['mask'];result=merge_result(result,pixel_m)
    if not np.array_equal(before,result['mask']):raise ValueError('Final merging changed footprint')
    if on_stage:on_stage('08_final_merge',result,start)
    start=time.monotonic()
    if not historical_replay:
        from .wall_footprint import clean_footprint
        result=clean_footprint(result,target,barrier,pixel_m,probability,forbidden,cfg)
    if on_stage:on_stage('09_nonoverlapping_footprints',result,start)
    start=time.monotonic()
    if not historical_replay:
        from .wall_thickness import stabilize_thickness
        result=stabilize_thickness(result,target,barrier,pixel_m,forbidden,cfg)
    if on_stage:on_stage('10_consistent_thickness',result,start)
    start=time.monotonic()
    if not historical_replay:
        from .wall_reference_guard import select_native_fit
        result=select_native_fit(result,accepted_reference,target,barrier,forbidden,pixel_m,cfg)
    if on_stage:on_stage('11_native_reference_guard',result,start)
    start=time.monotonic()
    if not historical_replay:
        from .wall_small_features import recover_small_features
        result=recover_small_features(result,target,barrier,pixel_m,forbidden,cfg)
    if on_stage:on_stage('12_short_scan_piers',result,start)
    return result


def fit_walls(target, pixel_m=.0025, *, semantic_probability=None, density=None,
              barrier=None, forbidden=None, config=None, initial=None, on_stage=None, historical_replay=False, accepted_reference=None):
    """Area fit, three structural passes, merge, simplify, final exact merge.

    ``initial`` replays a saved area fit on the SAME raster/evidence for regression
    checks. It is never used as geometry for a different model or scan. Inputs and
    intermediate stages are recorded; callbacks must not mutate the result.
    """
    target=np.asarray(target,bool)
    barrier=np.zeros_like(target) if barrier is None else np.asarray(barrier,bool)
    forbidden=np.zeros_like(target) if forbidden is None else np.asarray(forbidden,bool)
    if target.ndim!=2 or barrier.shape!=target.shape or forbidden.shape!=target.shape or pixel_m<=0:
        raise ValueError('Wall pipeline raster frames differ')
    if (target&forbidden).any(): raise ValueError('Target includes forbidden apertures')
    if historical_replay and initial is None:raise ValueError('Historical replay needs a saved area fit')
    cfg=config or Config(max_primitives=220)
    base=replace(cfg,structure_refine=False,structure_filter=False,structure_joint=False,
                 merge_wall_runs=False,simplify_wall_runs=False)
    history=[]
    def record(name,result,start):
        reconstructed=_union(result['primitives'],target.shape)
        if not np.array_equal(reconstructed,result['mask']):
            raise ValueError(f'{name}: exported footprints differ from mask')
        checks=audit(result['mask'],target,barrier,forbidden,pixel_m,cfg.barrier_contact_m)
        item=dict(stage=name,seconds=time.monotonic()-start,pieces=len(result['primitives']),
                  metrics=mask_metrics(target,result['mask']),audit=checks)
        if name=='00_area_fit':item['coverage_search']=result.get('coverage_search',{})
        if 'structure' in result and name in STAGE_ORDER[1:4]:
            item['structure']=result['structure']
        if name=='05_two_face_simplification':item['simplification']=result.get('simplification',{})
        if name in ['04_merge','08_final_merge']:item['merging']=result.get('merging',{})
        if name=='06_reference_guard':item['regression_guard']=result.get('regression_guard',{})
        if name=='07_scan_face_recovery':item['scan_face_recovery']=result.get('scan_face_recovery',{})
        if name=='09_nonoverlapping_footprints':item['footprint_cleanup']=result.get('footprint_cleanup',{})
        if name=='10_consistent_thickness':item['thickness_regularization']=result.get('thickness_regularization',{})
        if name=='11_native_reference_guard':item['native_reference_guard']=result.get('native_reference_guard',{})
        if name=='12_short_scan_piers':item['small_feature_recovery']=result.get('small_feature_recovery',{})
        history.append(item)
        if on_stage is not None:on_stage(name,result,item)
    start=time.monotonic()
    if initial is None:
        result=decompose(target,pixel_m,semantic_probability=semantic_probability,
            density=density,barrier=barrier,forbidden=forbidden,config=base)
    else:
        if np.asarray(initial['mask']).shape!=target.shape:raise ValueError('Initial area-fit frame differs')
        result=initial
    record(STAGE_ORDER[0],result,start)
    area=result
    # These are the historical structural pass settings; the 180 cap does not
    # re-fit area geometry, which uses the separate 220-candidate-piece cap.
    structural=replace(cfg,max_primitives=180)
    for name,stage_cfg in [
        ('01_continuity',replace(structural,structure_filter=False,structure_joint=False)),
        ('02_confidence',replace(structural,structure_joint=False)),
        ('03_shared_faces',structural)]:
        start=time.monotonic()
        result=refine_structure(result,target,barrier,pixel_m,semantic_probability,forbidden,stage_cfg)
        record(name,result,start)
    for name in STAGE_ORDER[4:6]:
        start=time.monotonic()
        if name=='05_two_face_simplification':
            result=simplify_result(result,target,barrier,pixel_m,semantic_probability,forbidden,structural)
        else:
            before=result['mask'];result=merge_result(result,pixel_m)
            if not np.array_equal(before,result['mask']):raise ValueError('Exact merging changed wall footprint')
        record(name,result,start)
    result=finish_walls(result,area,target,barrier,pixel_m,probability=semantic_probability,
        forbidden=forbidden,config=cfg,historical_replay=historical_replay,on_stage=record,density=density,accepted_reference=accepted_reference)
    result['pipeline']=dict(version=PIPELINE_VERSION,stages=history,area_config=asdict(base),
        structural_config=asdict(structural),initial_source='saved_area_fit' if initial is not None else 'fresh_area_fit')
    result['metrics']=mask_metrics(target,result['mask'])
    result['rectangle_overlap']=rectangle_overlap_metrics(result['primitives'],target.shape,pixel_m)
    return result
