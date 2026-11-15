# CAL-09.6: blind wall geometry and neighbour topology from XYZ

**Scope:** infer vertical planar wall candidates and their corner, T-junction and
collinear-gap relationships directly from measured XYZ. This stage is not a
snow/defect classifier and does not change the production wall pipeline.

## Why this stage exists

CAL-09.4's hard connected-component filter and CAL-09.5's soft score both
operated on synthetic physical cells whose wall/face/cell association was
supplied by the truth mesh. CAL-09.5's untouched 511xx holdout did not establish
the required recall/FPR trade-off. More oracle-cell threshold tuning is stopped.

CAL-09.6 removes that association from the detector itself. The detector API is
extract_wall_geometry(xyz) and accepts exactly one positional measurement
array with shape (N,3). It cannot receive wall IDs, triangle IDs, labels,
station IDs, poses, a plan, a layout, or synthetic truth.

## Blind inference implementation

tools/cal096_blind_geometry.py implements a deterministic, geometry-only
baseline:

1. Reject non-finite returns; voxel-thin to 25mm representatives, capped at
   200,000 points. Preserve a source-index map for evaluation and E57 traceability.
2. Propose vertical planes with seeded RANSAC in XY. A candidate needs long
   horizontal support and at least 1.8m vertical extent.
3. Refine the line by robust total least squares and a fixed 40mm inlier bound.
4. Project inliers to along-wall distance and height. Split disconnected
   components on an 80mm occupancy grid with one 8-connected closing pass.
5. Keep wall candidates only when fixed support, length, height, density,
   occupancy and plane-residual gates pass.
6. Infer graph edges from segment geometry alone:
   - corner: both finite segments reach the same perpendicular line intersection within the fixed junction tolerance;
   - tee: one segment end reaches the interior of a perpendicular segment at their line intersection;
   - collinear_gap: parallel, near-coincident segments separated by at most
     250mm with sufficient shared height.
7. Report parallel offset faces separately as ambiguous parallel_face_candidate
   hypotheses. They are not automatically merged into one wall or one graph edge.

The fixed parameter set is embedded in DEFAULTS and written to each case
report. There is no trained model or threshold selection in CAL-09.6. Any
parameter change after viewing development metrics requires rerunning the
development phase; holdout is not to be inspected until the detector is frozen.

## E57 input path

The same module provides a streaming E57Reader adapter. It reads Cartesian
XYZ from every scan, applies each E57 scan pose, removes non-finite and invalid
returns, and uses a bounded deterministic reservoir if the file exceeds the
input budget. It discards intensity, colour, station IDs and all semantic data.
Example:

    PYTHONPATH=. python tools/cal096_blind_geometry.py customer.e57 \
      --out /tmp/customer_blind_geometry.json

The repository currently contains no real .e57, .las, .laz, .ply or .pcd
point-cloud dataset. The E57 path can be unit-tested with generated E57,
but a customer-data acceptance result cannot be produced from repository
fixtures alone.

## Synthetic scanner evaluation, with a strict information boundary

The experiment generator reuses the existing scene and scanner implementation
with 0.10m mesh tessellation, 0.1-degree angular step and the three fixed CAL-09
geometry variants (uniform_2_4, zone08_fine35, zone18_fine35).

For each scene, all valid scanner returns are transformed using the written
station poses and concatenated. Mixed returns, virtual returns, furniture and
other clutter are not removed by semantic labels. Only the XYZ array is passed
to extract_wall_geometry.

A separate evaluation function, called only after inference, receives
synthetic point labels/instances and layout wall axes. It computes:
- point precision, recall, FPR and F1 on the same deterministic voxel sample;
- candidate-segment precision and visible wall-instance recall;
- topology edge precision/recall/F1 and edge-type accuracy against visible
  synthetic layout-wall relationships.

The evaluator's synthetic wall-instance mapping follows SceneBuilder's
wall creation order. This is evaluation-only; the detector never receives
or uses this mapping. The experiment records detector_uses_ground_truth=false
and the exact XYZ-only input contract in every case.

## Preregistered seeds and split

Previously observed CAL-09 seeds (461xx through 511xx) are not reused.

- Development: **60101, 60102, 60105, 60106**.
- Untouched holdout: **61101, 61102, 61105, 61106**.
- Three geometry variants per seed.
- 0.10m tessellation, 0.1° scanner, 200,000-point detector cap.
- 24 full scanner scenes total, split into two phases of 12 scenes each.

Development runs first. Inspect only 601xx metrics, fix bugs or tune the fixed
geometry parameters if necessary, and rerun development until the detector is
frozen. The 611xx holdout workflow is added/run only after the final detector
and evaluator source hashes are recorded. The holdout aggregator rejects a
fingerprint that differs from the frozen development fingerprint.

## Holdout engineering gates

All gates must pass on the untouched 611xx holdout, using the mean of four
seed-level means across three variants. Worst-seed point FPR is also checked.

| Metric | Required |
|---|---:|
| Point-level precision | >= 0.90 |
| Point-level recall | >= 0.70 |
| Point-level FPR | <= 0.10 |
| Worst seed point-level FPR | <= 0.15 |
| Visible wall-instance recall | >= 0.80 |
| Topology edge precision | >= 0.80 |
| Topology edge recall | >= 0.70 |
| Topology edge-type accuracy | >= 0.80 |

A PASS is only a synthetic blind-geometry engineering diagnostic. It does
not establish defect detection quality, robustness to all real E57 scans, or
production readiness. A FAIL stops promotion; do not retune against 611xx.

## Reproduction

Targeted tests:

    PYTHONPATH=. python -m pytest -q tests/test_cal096_blind_geometry.py

Small raycaster smoke:

    PYTHONPATH=. python tools/validate_cal096_blind_geometry.py \
      --out /tmp/CAL096_smoke --seed 60101 \
      --variants uniform_2_4 zone18_fine35 \
      --angular-step-deg 0.50 --max-points 50000

Full one-seed development shard:

    PYTHONPATH=. python tools/validate_cal096_blind_geometry.py \
      --out results/CAL096_dev --seed 60101 \
      --angular-step-deg 0.10 --max-points 200000

Aggregate four development shards:

    PYTHONPATH=. python tools/aggregate_cal096.py \
      --phase development --out results/CAL096_dev_final \
      /path/cal096_seed_60101.json /path/cal096_seed_60102.json \
      /path/cal096_seed_60105.json /path/cal096_seed_60106.json

Holdout aggregation requires the exact frozen development fingerprint:

    PYTHONPATH=. python tools/aggregate_cal096.py \
      --phase holdout --expected-fingerprint FROZEN_DEV_FINGERPRINT \
      --out results/CAL096_holdout_final \
      /path/cal096_seed_61101.json /path/cal096_seed_61102.json \
      /path/cal096_seed_61105.json /path/cal096_seed_61106.json

The per-seed cache fingerprint includes source YAML SHA-256, detector source
SHA-256, evaluator source SHA-256, mesh resolution, scanner angular step and
point budget. A code change invalidates cached scenes.

## Known limits / what CAL-09.7 still needs

- This is a vertical-plane baseline; curved, strongly sloped or heavily
  occluded wall faces are not yet reconstructed.
- A flush cabinet or other coplanar object can be indistinguishable from a
  wall using XYZ alone. Additional context or semantic evidence is required.
- One 80mm occupancy closing pass can bridge small holes but deliberately does
  not close door-sized gaps. It is not a substitute for opening recognition.
- Parallel offset planes are ambiguous: they may be opposite faces of one wall
  or separate nearby walls.
- Synthetic point labels/layout are used only by the evaluator. The blind
  detector is not allowed to use them.
- There are no real E57 scans in the repository to establish real-world
  precision, recall, wall topology, registration behaviour or failure modes.

CAL-09.7 therefore still requires real customer E57 files, station/registration
provenance where available, and independently surveyed wall geometry and
defect/opening annotations. Keep this branch experimental. Production YAML,
source E57, training labels, trained weights and production wall code remain
untouched.
