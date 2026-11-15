# CAL-04: multi-room local wall-plane evidence

## Purpose

Extend CAL-02/CAL-03 from a single small room to three independent **three-room** synthetic scenes, at fixed **0.1°**, without modifying `configs/customer_like.yaml`. Study observed planar residuals against measured ray distance, estimated incidence, and the number of individual scanner stations supporting the same wall patch.

## What is and is not measurable

Synthetic scans include station-specific XYZ and registered `pose_written`; this supports per-return measured range, ray direction and physical station identity. For an estimated vertical wall candidate, incidence is computed against the locally fitted surface normal, not a perfect GT normal. Patches and residuals come from the **same CAL-02 plane fitter, RANSAC threshold, 0.9 m × 1.25 m windows, and top-24 selection**. The scan class label and procedural wall geometry are never used to select patches.

The two original E57 files expose only one data3D cloud each, without row/column indices or the original physical station mapping. Thus *real per-station range, incidence and overlap metrics are unknown*. Cross-domain comparison is limited to **overall local plane MAD/RMS**, with strong selection bias. Do not infer a sensor's intrinsic range sigma, registration sigma, number of physical stations or angular step from these results.

## Method

1. Generate three independent 3-room scenes at seed 42121, 42122 and 42123, with at most two stations per room and no point thinning or dropped rooms.
2. Read valid return XYZ from each scan and transform it using the **written** station pose. Keep a single priority reservoir of at most 250,000 returns over all stations.
3. Detect wall candidates with the CAL-02 estimator using only observed Cartesian points. Select at most the 24 largest locally supported eligible patches per scene.
4. Calculate the median station-local range, estimated incidence angle and the number of contributing stations to each local wall patch. Multi-station evidence requires >=25 direct returns and >=10% direct-return share **per contributing station**, for at least two stations.
5. Record MAD/RMS by patch and group into non-overlapping range/angle bins and the single/multi-station coverage cohorts. Mark groups with fewer than five patches or fewer than three independent scenes as insufficient for comparison.
6. Keep per-scene medians separate; a large number of patches within one scene does not create independent sites. Save only summaries and patch statistics without sensor poses, raw points or coordinates.

## Non-goals and safety boundaries

- The observed plane residual combines wall roughness, incidence, station registration and reflective materials. Its trend is *not* a uniquely identifiable physical sensor law.
- The angle, range and station overlap are correlated in the geometry. A difference in MAD across bins does not prove causality.
- Real source data contain no recovered per-ray station trace. It would be misleading to compare synthetic range bins directly against these real clouds.
- The real CAL-02 estimate is from two buildings, with patches selected by density, and cannot support production parameter optimization.
- **Do not change** `scanner.angular_step_deg`, `effects.range_sigma_mm`, `effects.registration_sigma_mm` or training data until a larger independent calibration/validation experiment is completed.

## Run

```bash
PYTHONPATH=.:tools python tools/calibrate_multiroom_surfaces.py \
  --out /tmp/cal04 --seeds 42121 42122 42123
```

Results: `cal04_comparison.json`, `CAL04_REPORT.md`, plus an Actions artifact with only those files. The short-lived generated scenes are deleted. The CI job intentionally does not upload any E57 or XYZ file.
