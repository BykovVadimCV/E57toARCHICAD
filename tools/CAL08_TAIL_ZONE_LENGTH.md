# CAL-08: spatial tail-zone correlation length (preregistered)

## Motivation and reference

CAL-07 concluded `NO_SPATIAL_TAIL_VARIANT_PASSES`: even the 20% nominal 0.8-m envelope produced only one scene with a top-24 adjacent heavy-tail patch cluster (1/8), although its synthetic scene p90 increased to 4.6071 mm. The two measured CAL-02 E57 sites have top-24 patch MAD p90 1.7205 and 10.0135 mm, respectively. The original physical stations, angles and raw scan-to-ray provenance are unavailable. Residuals combine wall surface, sensor noise, registration, occlusion and patch selection.

The CAL-08 hypothesis is that **enlarging only the correlation length of the local roughness envelope** increases the occurrence of *adjacent* locally rough wall patches, without broadly thickening all observed wall surfaces. Longer-envelope patches still contain shorter-frequency roughness from the unchanged base Fourier field; a perfectly flat offset is not expected to increase a local-plane fit residual by itself.

## Frozen design before running new seeds

- **New seeds:** `45101 ... 45108`, disjoint from all CAL-03, CAL-04, CAL-05, CAL-06 and CAL-07 seeds. The first four use the original 3-room footprint, the second four larger footprints. The scanner remains at 0.1 degrees.
- **Exactly four variants per seed (32 full scans):**
  - `uniform_2_4`: calibrated-candidate uniform roughness 2–4 mm, no heavy-tail zone.
  - `tail20_corr08`: CAL-07 20% heavy-tail control, 0.8-m envelope correlation.
  - `tail20_corr18`: 20% heavy-tail, 1.8-m envelope correlation.
  - `tail20_corr30`: 20% heavy-tail, 3.0-m envelope correlation.
- **Locked factors:** 2–4 mm base roughness, 8–12 mm local tail amplitude, 20% nominal Gaussian threshold for active tail variants, and 0.15-sigma smooth transition. Scanner noise, registration, layout and all other parameters remain unchanged. The smooth envelope uses the old random Fourier basis without additional random numbers, so the true room geometry and physical station poses should match per-seed.
- **Estimator:** exact CAL-02/CAL-04 observed-XYZ vertical local-plane MAD. Primary p50/p90/p95 from the same top-24 supported patches per scene as CAL-02. Additionally report clustering on the top-72 supported synthetic-only patches, to check selection sensitivity. These 72 are **not** comparable to the real site top-24 reference.
- **Spatial clusters:** two or more adjacent cells with MAD >= 6 mm on the same fitted wall-line candidate's *local* (length,height) patch grid. These grids do **not** identify physical walls or provide world coordinates. Clusters are descriptive synthetic evidence only, not ground truth defects.

## Locked acceptance criteria

For either longer-correlation candidate to earn provisional support, **all** must hold:

1. All eight pairs have >=8 viable primary patches; retain >=75% of baseline patches; candidate and short-correlation control both retain 70–130% of valid scan returns. Identical true geometry and stations are mandatory.
2. Candidate median observed-plane MAD differs by <=0.5 mm from the uniform 2–4 mm control on >=6/8 scenes.
3. >=6/8 scene median MADs fall between the two real-site median references expanded by 0.5 mm (exploratory engineering bracket).
4. Candidate top-24 patch p90 grows by >=20% against uniform on >=6/8 scenes, including >=3/4 in each footprint cohort.
5. Median over scene top-24 p90 lies between 5 mm and upper real-site p90 +1 mm (11.0135 mm). >=6/8 individual scene p90 values must be no more than 11.0135 mm.
6. >=4/8 scenes exhibit an adjacent heavy-tail patch cluster within top-24, and >=4/8 exhibit one in expanded top-72 (sensitivity diagnostic).
7. At least 4/8 scenes have a strictly larger top-24 tail cluster than the same-seed short-correlation 0.8-m control.

If both 1.8 and 3.0 m pass, select 1.8 m to minimize spatial intervention. If neither passes, keep the 2–4 mm base as **candidate only** and do not change production YAML. Thresholds are engineering gates, not confidence intervals or evidence that 20% of actual walls are defective.

## Reproducibility and privacy

```bash
python -m pytest -q tests/test_cal08_tail_zones.py tests/test_cal07_spatial_tail.py tests/test_cal06_holdout.py
python tools/validate_cal08_tail_zones.py --out /tmp/cal08 --seeds 45101 45102 45103 45104 45105 45106 45107 45108 --max-sample 250000
```

A partial JSON checkpoint contains only aggregate rows. Final artifacts contain per-seed scalar diagnostics and gates, never XYZ, scans, true/registered station poses, or real E57 files. `configs/customer_like.yaml`, trained weights and GT masks are not modified. No automatic model retraining or physical scanner calibration is authorized.
