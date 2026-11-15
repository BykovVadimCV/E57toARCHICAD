# CAL-02: audit of original E57 station metadata and locally observed wall-plane residuals

## Purpose

CAL-01 found that changing only the synthetic scanner angular step does not match two real-world 2-mm occupancy metrics simultaneously. CAL-02 is a targeted follow-up. It first checks what the *original E57 export actually contains*; then reads a bounded uniform sample of raw XYZ and estimates local planar-wall residual distributions.

It **does not fit** `effects.range_sigma_mm`, `effects.registration_sigma_mm`, `incidence_noise_power`, or `scanner.angular_step_deg`. Any such fit would require additional evidence of independent stations, known scan ranges, incidence angles, and adequately planar reference surfaces. A residual from a real wall also reflects real surface roughness and potentially multiple registered point-cloud passes.

The Stage 0 verification report listed two real E57 objects, with **one `data3D` node per file**. This is not proof of one physical scanner station. If the E57 export merged stations, the original number of stations and scan-ray origins may no longer be recoverable. `rowIndex`, `columnIndex`, index bounds, and pose *availability* are reported explicitly. A pose belonging to a single merged scan node cannot automatically be treated as the original scanner position.

## Reproducible notebook

`notebooks/colab_cal02_real_e57_surfaces.ipynb` downloads the user's two shared Drive originals only into the Colab runtime (or uses existing local/Drive copies), validates their sizes and SHA-256 values from Stage 0, runs the header-only inspection and then the streaming surface audit. It saves JSON/Markdown results and offers a small downloadable ZIP. The big E57 files and any raw point samples are **not** uploaded to GitHub.

## CLI

```bash
PYTHONPATH=. python tools/calibrate_e57_surfaces.py \
  --e57 path/4012_Mikluhi-Maklaya-3-793.e57 path/Barklaya_14_42.e57 \
  --out results/cal02_header --metadata-only

PYTHONPATH=. python tools/calibrate_e57_surfaces.py \
  --e57 path/4012_Mikluhi-Maklaya-3-793.e57 path/Barklaya_14_42.e57 \
  --out results/cal02_surfaces --max-sample 250000 --chunk 250000
```

The stream reader reads Cartesian fields only, filtering non-finite and `cartesianInvalidState` records when available. For each `data3D` node, uniform-priority reservoir sampling is limited to `--max-sample` points and is repeatable for a fixed seed. Wall hypotheses are obtained by XY line RANSAC, followed by spatially local 3D trimmed plane fits. Candidates need longitudinal and vertical extent, which rejects most purely horizontal floors/ceilings. These are heuristic hypotheses, **not authoritative wall labels**.

Output:

- `cal02_surface_calibration.json`: source metadata, field/index/pose availability, number of sampled and valid points, per-patch observed MAD/RMS/p95 thickness in mm, calibration limits.
- `CAL02_REPORT.md`: readable summary.

`--sample-npz` can be used for regression fixtures containing an `xyz` array. Synthetic fixtures serve to verify the estimator, not to infer real equipment performance.

## Interpretation and quality gates

- `observed_plane_mad_mm`: 1.4826 × median absolute deviation from a locally fitted measured plane. **Not intrinsic range sigma.**
- `observed_plane_rms_mm`: RMS local 3D perpendicular residual, after explicit iterative trimming. **Not a sensor setting.**
- `row_and_column_indices_available`: field names in the compressed E57 vector. Presence may indicate an organized export but does not prove independent original stations.
- `one_data3d_node_is_one_physical_station = unverified`: essential provenance rule.
- Missing/insufficient wall patches must remain `null` or missing, not be backfilled with guessed values.
- No full building geometry, point coordinates, room reconstruction or scanner identification is published by the audit.

## Limitations

1. Reservoir sampling removes absolute point-density comparability; CAL-01's *full* count images remain the appropriate density reference.
2. Plane heuristics can select glass, partitions, surfaces with finish, parallel walls, or mixtures of registered scans. They do not recover physical ground-truth walls.
3. Camera/laser incidence cannot be determined without actual sensor origins and/or indexed scan rays.
4. If the raw E57 does not expose independent stations or grid indices, report the lack of identifiability. Do **not** invent stations based on the single `data3D` count.
5. The full source E57s are over the 256-MB connector transfer limit, so this computation runs next to the files in Colab; GitHub CI tests only the numerical estimator using controlled fixtures.

Next decision after real reports: compare **matched local planar patches** from actual and synthetic scan returns using identical estimator settings and known/unknown station provenance. No direct tuning of `customer_like.yaml` before that comparison.
