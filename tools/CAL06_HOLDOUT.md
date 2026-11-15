# CAL-06: independent roughness holdout

## Locked hypothesis

CAL-05 used four synthetic seeds `42121–42124` to select `realism.wall_unevenness_mm: [2, 4]` for further study. This candidate has **not** been approved for the production `customer_like.yaml`.

CAL-06 freezes two variants only: the original YAML `6–14 mm` versus the selected candidate `2–4 mm`, with every other setting unchanged, `angular_step_deg=0.1`. It evaluates eight **new** seeds `43101–43108` and compares both options on identical room geometry and true scanner stations within a pair. The last four seeds have a larger footprint than CAL-05. A scene may be counted only when both variants have sufficient observed wall patches and retain scan returns.

Expected 16 synthetic scans. No reused or searched holdout seed is permitted. The only reference statistic from original E57 is the CAL-02 local-plane robust MAD distribution: site medians 1.4731 mm and 2.7332 mm; original station/ray provenance is not available.

## Predeclared acceptance gates (before first CAL-06 run)

- All 8 pairs must have ≥8 eligible wall patches per scan; the candidate must retain ≥75% of baseline patches and between 70% and 130% of valid scan returns.
- At least 7/8 pairs must improve median local-plane MAD and median paired improvement must be ≥1.0 mm.
- At least 6/8 candidate scene medians must lie inside the **exploratory** expanded interval bounded by two real-site medians ±0.5 mm.
- Both footprint cohorts need improvement in at least 3/4 scenes.
- Failure of any gate rejects provisional support; passing all gates allows **provisional metric-level support only**, not automatic production setting changes.

These gates are pragmatic engineering thresholds, **not confidence intervals** and not physical validation of a particular instrument. CAL-02 selects the densest local patches and real E57 scans lack original station attribution.

## Provenance and non-leakage

- The same CAL-02 plane fitter and CAL-04 tagged return sampler are reused, and synthetic points are registered with the **written** (potentially noisy) poses.
- Same true station locations and room layout per pair verified by signature. Different noisy returns and local fit patch selection can still occur; matched observed physical patches are **not** claimed.
- No raw E57 points, simulated XYZ, true station coordinates, or original site photos are published as artifacts. Only non-coordinate aggregate results, quantiles and signed reductions.
- Dataset masks, neural network weights and production `configs/customer_like.yaml` remain unchanged.

## Reproduce

```bash
python tools/validate_cal06_holdout.py --out /tmp/cal06 --seeds 43101 43102 43103 43104 43105 43106 43107 43108 --max-sample 250000
```

Full scan requires Open3D, pye57 and Python 3.11 CI dependencies. `cal06_partial_progress.json` is a partial checkpoint **not a calibration conclusion**. Final `cal06_results.json` and `CAL06_REPORT.md` are emitted only after all 16 scans.
