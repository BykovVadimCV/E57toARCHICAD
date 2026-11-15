# CAL-09.5: Soft spatial confidence / independent synthetic holdout

**Experimental only. Original CAL-09 blind top-24 validation remains FAIL.**

## Goal and stopping rule

CAL-09.3 unfiltered subcell-shape detection increased oracle-patch recall
but generated too many false positives. CAL-09.4 removed small connected
components but also removed too many true defects. CAL-09.5 tests continuous
local-plus-neighbour evidence using a lightweight logistic score rather
than irreversible binary morphological filtering.

If it fails the preregistered fresh holdout gates, do **not** tune again
on 511xx. Continue with CAL-09.6 blind cloud geometry and collecting
real-E57 truth rather than repeated synthetic threshold fishing.

## Strictly new seeds

All 461xx, 471xx, 481xx and 491xx have already been consulted and cannot
be reused as untouched data.

Development only: 50101, 50102, 50105, 50106.
Independent holdout only: 51101, 51102, 51105, 51106.

8 seeds x 3 fixed variants (uniform_2_4, zone08_fine35, zone18_fine35)
at **0.10 m wall mesh** = **24 full scanner scenes**, fixed 0.1 degree
angular step and 180000 direct wall returns per scene.

Only the detector changes. Geometry amplitude, CAL-09 defect probability,
scanner noise, station registration model and source YAML remain fixed.

## Frozen model and candidate grid

Oracle-assigned wall/face/0.90x1.25m physical cells are computed with
the unchanged CAL-09.3 feature extractor. The detection model sees
only the measured combined-mode scores, not GT envelope fractions.

Candidate feature types:
- local: log1p of measured subcell_shape, local MAD, macro offset,
  and a binary missing-shape indicator.
- shape_context: local plus log1p(mean shape of immediately adjacent
  valid same-wall-face cells) and fraction of four present neighbours.
- full_context: shape_context plus log1p(max neighbouring shape).

No edge, vertex, or cell belonging to another wall or different
geometry variant is used as a neighbour. Neighbour signals are not
binarized, so strong isolated evidence can survive.

Regularised logistic regression is fit using deterministic SciPy
L-BFGS-B, no arbitrary neural network initialization. Candidate grid:
3 feature sets x L2 in [0.01,0.1] x 7 negative-score quantiles
[0.80,0.85,0.90,0.925,0.95,0.975,0.99] = **42 predefined choices**.

Development score selection uses leave-one-**SEED**-out CV (4 folds).
All three geometry variants and all patches of each held-out seed
remain in its validation fold; only other seeds train its model.
Within development, candidate mean FPR must be <=8% and worst
single-seed FPR <=15%. Select highest mean recall, then cluster recall,
then lowest FPR and stable tuple order. No candidate means no
selection, and this is a legitimate failed result.

Only after selection is frozen, a model is fitted on all development
scenes and evaluated at the same OOF-selected threshold on the
independent 511xx holdout. The Brier score and 5-bin expected
calibration error are reported: logistic score is not assumed
accurately calibrated before checking it.

## Predetermined acceptance gates (oracle diagnostic only)

At 0.10m mesh, ALL required:
- Mean recall >= 40%.
- Mean FPR <= 10%.
- Mean connected GT-cluster recall >= 35%.
- Recall gain vs the predeclared legacy MAD >=6mm >=15 percentage points.

Holdout compares the single development winner to two fixed baselines:
legacy MAD >=6mm and raw CAL-09.3 shape >=4mm. Holdout scores never
influence hyperparameter, threshold or tie breaking.

Cluster recall is calculated for true connected components of at least
two cells within a physical wall face. Such a cluster is detected only
if at least two true-positive predicted cells are themselves adjacent.
Confusion counts are pooled across three variants only *after*
independent within-variant topology calculations.

## Reproduce (repository root)

Install the existing requirements. Then run tests:

    PYTHONPATH=. python -m pytest -q tests/test_cal095_soft_confidence.py

Run noncanonical two-variant smoke:

    PYTHONPATH=. python tools/validate_cal095_confidence.py \
      --out /tmp/cal095_smoke --seeds 50101 \
      --variants uniform_2_4 zone18_fine35 \
      --angular-step-deg 0.5 --max-sample 20000

Canonical full experiment:

    PYTHONPATH=. python tools/validate_cal095_confidence.py \
      --out results/CAL095 --angular-step-deg 0.1 --max-sample 180000

The GitHub Actions workflow named cal095-soft-confidence.yml performs
one seed per job and aggregates the exact eight JSON shards:

    PYTHONPATH=. python tools/aggregate_cal095.py \
      --out results/CAL095_final \
      /path/cal095-seed-*/cal095_scores.json

Output: cal095_scores.json, cal095_progress.json,
cal095_holdout_results.json and CAL095_REPORT.md.
Per-case caches are guarded by source YAML SHA-256, scanner step,
protocol version and sampled direct-ray count. Invalid or incomplete
shards cannot yield an accepted report.

## Explicit limitations

- True physical wall/face grouping comes from the synthetic raycaster.
  This is an ORACLE physical-patch test, NOT blind E57 segmentation.
- The supervised logistic model uses development GT labels only.
  Holdout truth is used once, exclusively for final evaluation.
- Four development and four holdout seeds form a small experiment,
  not a production-grade real-world statistical guarantee.
- True defects are defined by fractions of visible direct ray hits,
  not surveyed actual square metres of damaged walls.
- Calibration shift between cross-fit development and final refit
  may reduce performance on untouched data.
- Original real E57 sources still lack known per-return station
  provenance and independently surveyed defect masks.
- Even a synthetic PASS cannot override the original CAL-09 blind
  top-24 failures.

Production YAML, original E57, GT annotations and trained weights
are not modified or automatically promoted.


## Verified final result

[Successful canonical full run](https://github.com/BykovVadimCV/E57toARCHICAD/actions/runs/43743465844)

[Independent holdout JSON and Markdown artifact](https://github.com/BykovVadimCV/E57toARCHICAD/actions/runs/43743465844/artifacts/14173890037)

All 24/24 canonical raycast scenes, 67 targeted regression tests and
the real-raycast smoke passed. The winner selected *only* from four
501xx development seeds was **local** (not a spatial feature set),
L2=0.01, cross-fitted 95th negative-score percentile, score threshold
0.27284485. Crossfit development recall 27.2%, mean FPR 5.6%,
worst-seed FPR 11.3%, cluster recall 24.5%.

Fully untouched 511xx holdout (mean across four seeds, same mesh):

| Rule | Mean recall | Mean FPR | Mean GT cluster recall |
|---|---:|---:|---:|
| MAD>=6mm | 12.2% | 2.7% | 12.5% |
| Raw shape>=4mm | 47.3% | 13.8% | 53.8% |
| **Selected logistic confidence** | **13.7%** | **2.2%** | **11.8%** |

Brier=0.111; ECE-5=0.040. The dev-selected threshold trades most
sensitivity for low FPR, and spatial model candidates did not win.

**Final decision: CAL095_ORACLE_SOFT_CONFIDENCE_NOT_ESTABLISHED.**

Three of four holdout gates fail: recall <40%, connected-cluster
recall <35%, and gain over MAD6 <15 percentage points. FPR <=10%
alone passes. No 511xx threshold adjustment is permitted.

**Stopping rule triggered:** Discontinue iterative synthetic oracle
score/threshold trials. The next research work should validate blind
wall topology from raw point clouds (CAL-09.6), followed by independently
surveyed real E57 defect truth (CAL-09.7). Neither synthetic
study can substitute for real-world target truth, and production remains
blocked on those gates and the final regression/performance review.
