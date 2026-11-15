# CAL-03: matching local real and synthetic E57 wall-surface measurement

## The experiment

One robust CAL-02 plane-residual estimator runs on both streams: the original per-file real references (`configs/calibration_real_e57_surfaces.json`) and registered synthetic direct returns (without architectural ground truth). The synthetic stream is built from `ScanSimulator` valid XYZ and **written station poses**, using the same `PointReservoir(capacity=250000, seed=42)` and `assess_wall_patches(max_lines=5,max_patches=24,seed=42)` used for CAL-02 real E57 scans.

The real input reference contains 24 selected patches from each of two independently SHA256-verified E57 files, with MAD/RMS recorded by CAL-02. It does not contain raw coordinates or station ray trajectories. The comparison uses per-file reported medians and candidate per-patch distributions. No real point cloud is uploaded to GitHub in this stage.

Two small, one-room synthetic configurations, with the same layout seeds across all candidates, isolate two sensitivity factors while holding the angular scan step at **0.1°**. Five named configurations run for both seeds:

- `baseline`: unmodified scanner effects and surface roughness from `configs/customer_like.yaml`.
- `low_unevenness`: only change `realism.wall_unevenness_mm` to `[2,4]`.
- `high_unevenness`: only change `realism.wall_unevenness_mm` to `[12,20]`.
- `low_registration`: only change `effects.registration_sigma_mm` to `[0.5,1]`.
- `high_registration`: only change `effects.registration_sigma_mm` to `[7,11]`.

Configuration changes are **ephemeral**: source production YAML, baseline checkpoints and datasets are never changed. Only aggregate statistics are uploaded. The workflow deletes ephemeral scratch scans.

## What the result means and does not mean

The observed residual is a **mixture** of wall roughness, registration, noise, grazing angle, ray/beam effects, material and patch selection. A good numerical match is not sufficient to say a scanner setting is calibrated. There are two real properties, both exported as merged `data3D` with unknown original physical stations, and only two synthetic seeds, with different spatial layouts and material mix. The CAL-02 estimator preferentially selects 24 well-populated patches per scan; it is not representative wall sampling. Avoid mean and worst-case extrapolation to full apartment data.

If the baseline sits far outside both real sites, this identifies a **synthetic-vs-real observation mismatch**. If changing one factor has a consistent directional effect across seeds, it identifies a **parameter sensitivity**, not the correct physical value. A change to production settings requires matching in distance, incidence angle, station overlap and wall material and testing several complete apartments.

## Reproduction

```
PYTHONPATH=.:tools python -m pytest -q tests/test_cal03_surface_comparison.py
PYTHONPATH=.:tools python tools/calibrate_real_synthetic_surfaces.py \
 --out /tmp/cal03 --real-reference configs/calibration_real_e57_surfaces.json \
 --seeds 42111 42112
```

Official GitHub workflow: `.github/workflows/cal03-real-vs-synthetic.yml`. Artifacts: `CAL03_REPORT.md` and `cal03_comparison.json`, no raw 3D coordinates.
