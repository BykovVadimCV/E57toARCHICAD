# CAL-09.4: spatial coherence against false positives

Status: **experimental diagnostic**, no production changes.

## Motivation

CAL-09.3 selected subcell_shape >= 4mm on development. Independent 471xx
holdout at 0.10m mesh yielded recall 47.0%, cluster recall 53.2%,
FPR 13.2% (target <=10%). CAL-09 legacy MAD>=6mm recall was 15.1%.
All those seeds have been viewed, so none may tune CAL-09.4.

## New frozen experiment design

- NEW development seeds: 48101, 48102, 48105, 48106.
- NEW independent holdout: 49101, 49102, 49105, 49106.
- Three variants: uniform_2_4, zone08_fine35, zone18_fine35.
- Two mesh resolutions: 0.30 and 0.10 metres.
- Fixed scanner angular step: 0.10 degrees; up to 180,000
  uniform-priority sampled DIRECT wall returns per scene.
- Total: 8 seeds x 3 variants x 2 meshes = **48 complete scans**.

The physical cell definitions, scene config and scanner ray settings
are reused from CAL-09.3. All changes in CAL-09.4 are in the detector
and analysis; wall-defect amplitude and probability do not change.

## Frozen topological rules

On each physical wall side, evaluate only observed subcell_shape scores.
A weak cell has score >= weak_mm; a strong cell >= strong_mm.
Build 4-neighbour connected components on the physical wall/face grid.
Keep a weak component only when:
1. Its size is at least the candidate's min_component_size (>=2).
2. It contains at least one strong cell.

No diagonal or cross-wall connection is accepted. Missing scores
are never positive. Ground-truth labels are NEVER consulted to
decide whether a cell is predicted positive. All true labels
are used solely for scoring.

16 candidates are explicitly enumerated in the frozen
CANDIDATES tuple in tools/cal094_spatial_selection.py:
shape3/3.5/4/4.5 with component sizes 2,3,4; and
hysteresis with weak 3/3.5/4 and strong 4/5/6,
requiring at least 2 or 3 adjacent cells.

Two baselines are fixed and never considered for tuning:
- CAL-09: legacy_mad >=6 mm;
- CAL-09.3: subcell_shape >=4 mm without spatial filtering.

## New development selection rules

Per seed and mesh, confusion counts aggregate three independent
geometry variants but the physical topology **never** joins
different scans, even where key coordinates happen to match.
FPR uses all negative physical patches, including uniform controls.

Eligibility on NEW development seeds:
- Mean FPR over 8 seed×mesh cases <=8%.
- Worst individual seed×mesh FPR <=15%.

Among eligible rules, maximize mean recall, then connected-GT
component recall, then minimize mean FPR, then use frozen
tuple-order to break ties. A failure to find an eligible candidate
results in no selection. The new holdout data are never consulted
until the selection is frozen.

Holdout contains only chosen rule plus the two fixed baselines.
The chosen rule passes diagnostic gates at the 0.10m mesh only if:
- Mean recall >=40%.
- Mean FPR <=10%.
- Mean connected-cluster recall >=35%.
- Mean recall improves over MAD6 by at least 15 percentage points.

A true GT cluster contains >=2 adjacent positive physical patches.
It counts as recovered only if >=2 adjacent true-positive patches
in it are detected. A false positive always contributes to FPR,
even if its geometric component has more than one cell.

## Reproducible commands (repository root)

    PYTHONPATH=. python -m pytest -q tests/test_cal094_spatial.py

Two-case smoke (not a canonical scanner measurement):

    PYTHONPATH=. python tools/validate_cal094_spatial.py \
      --out /tmp/cal094_smoke --seeds 48101 \
      --variants uniform_2_4 zone18_fine35 \
      --resolutions 0.30 --angular-step-deg 0.50 --max-sample 20000

Full 48-scene feature collection:

    PYTHONPATH=. python tools/validate_cal094_spatial.py \
      --out results/CAL094_features --angular-step-deg 0.10 \
      --max-sample 180000

The CI workflow cal094-spatial-coherence.yml distributes work
across eight seed jobs. Aggregate only all eight complete shards:

    PYTHONPATH=. python tools/aggregate_cal094.py \
      --out results/CAL094_final /path/cal094-seed-*/cal094_scores.json

Output: cal094_scores.json, cal094_progress.json,
cal094_holdout_results.json, CAL094_REPORT.md.
Caching is protected by source YAML SHA, protocol version,
max sampled ray count and angular scan step.

## Strict limitations

This is an **oracle grouped physical-patch diagnostic** on synthetic
rays, NOT blind wall detection. Predicted labels only use observed
scores but the wall/face/physical-cell grouping itself comes from
the synthetic ground truth. No real E57 with per-return physical
wall ground truth has been used to validate the detector.

Both 0.30 and 0.10 mesh scans have the same layout and station
configuration but different geometric ray intersections; they
are not ray-for-ray identical. Four new holdout seeds form only
a small exploratory sample.

Even a passing oracle metric cannot override the original failed
CAL-09 blind top-24 observed-patch gates. Production YAML, original
real E57, GT masks and model weights are not updated.


## Final results (frozen new-seed holdout)

[Successful GitHub Actions run](https://github.com/BykovVadimCV/E57toARCHICAD/actions/runs/43741721999)

[Canonical JSON and report artifact](https://github.com/BykovVadimCV/E57toARCHICAD/actions/runs/43741721999/artifacts/14172623288)

All 48 scanner simulations completed; 60 regression tests and actual raycast
smoke passed. Selection on development only: **shape4p5_cc2**, i.e.
observed subcell_shape >=4.5mm and 4-connected component size >=2.
Only this and shape4p5_cc3 met both development FPR restrictions.

| Rule | Mesh | Independent mean recall | Mean FPR | GT cluster recall |
|---|---:|---:|---:|---:|
| legacy MAD6 | 0.30 | 3.7% | 6.8% | 1.6% |
| legacy MAD6 | 0.10 | 15.2% | 6.2% | 3.1% |
| raw CAL-09.3 shape4 | 0.30 | 51.1% | 20.4% | 38.0% |
| raw CAL-09.3 shape4 | 0.10 | 47.6% | 22.2% | 29.2% |
| selected shape4p5_cc2 | 0.30 | 15.7% | 9.3% | 19.8% |
| selected shape4p5_cc2 | 0.10 | 21.1% | 10.6% | 26.0% |

**Decision: CAL094_ORACLE_SPATIAL_DIAGNOSTIC_NOT_ESTABLISHED**.

All four engineering criteria on the 0.10m mesh **failed**:
- Recall 21.1% <40%.
- FPR 10.6% >10%.
- Cluster recall 26.0% <35%.
- Gain over MAD6 +5.9 percentage points <15.

Hard spatial filtering almost halved *this holdout's* false-positive rate
relative to unfiltered shape4, but eliminated many true detections.
The earlier CAL-09.3 FPR=13.2% came from a **different seed holdout**,
so that contrast is not a paired effect. Neither set may now be reused
to tune follow-on rules without invalidating independent assessment.

Do not promote this detector or modify production. A next study must
use fresh seeds and should investigate probabilistic spatial evidence
or restore blind physical-wall extraction, instead of post-hoc adjusting
these thresholds on the used holdout.
