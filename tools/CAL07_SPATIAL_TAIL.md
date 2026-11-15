# CAL-07: spatially clustered heavy-tail roughness

## Hypothesis

CAL-06 supports a uniform base wall-unevenness range of 2–4 mm for typical observed planes. CAL-02 also shows that real-site patch distributions are not homogeneous: Mikluhi–Maklaya has patch MAD p90 1.7205 mm, while Barklaya has p90 10.0135 mm (24 densest eligible patches per site). CAL-07 tests whether a *small, spatially clustered fraction* of rougher surface zones can represent this upper tail without shifting the typical-surface median.

The real tail cannot be attributed uniquely to plaster, range noise, registration, edge contamination or material effects. The experiment therefore tests an observed-surface morphology option, not physical scanner noise.

## Synthetic model

- Base wall-unevenness remains 2–4 mm.
- Variants: uniform control; nominal 10% and 20% spatial tail zones.
- Tail-zone horizontal displacement amplitude is mapped smoothly to 8–12 mm from the already sampled base amplitude.
- Tail zones use a smooth 3-D Gaussian/RFF envelope with 0.8 m correlation length and 0.15-sigma transition. There are no discontinuous masks or mesh seams.
- The envelope reuses the existing random Fourier basis. It consumes **no additional random numbers**, so layout, true station positions, thickness jitter and the historical random stream remain paired.
- Only XY displacement is tail-amplified; vertical displacement and all scanner settings remain at the CAL-06 base profile.

The probability values describe the nominal Gaussian-field upper tail. The smooth transition means the fraction with any nonzero amplification is somewhat larger than the nominal probability.

## Frozen experiment

- Eight new seeds 44101–44108, disjoint from CAL-03–CAL-06.
- Two footprint cohorts, four scenes each; three rooms and 0.1° angular step.
- Three variants per seed, 24 full scans. Geometry and true station poses must hash identically within each seed.
- Same CAL-02/CAL-04 local-plane estimator and 24 densest eligible patches per scan.
- Patch-level MAD p50/p90/p95, tail-patch count (MAD >= 6 mm), local grid connected-component size, valid-return count and patch retention are reported.
- Local patch grid indices are internal to each fitted wall candidate; no XYZ, world coordinates or station poses are exported.

## Predeclared engineering gates

A tail variant is supported only if all conditions hold:
1. All eight paired scenes retain at least eight valid patches, at least 75% patch retention and 70–130% valid-return count.
2. Typical median changes by no more than ±0.5 mm in at least six of eight scenes.
3. At least six of eight scene medians remain in the exploratory real-site median bracket (two site medians ±0.5 mm).
4. Patch p90 rises by at least 20% versus uniform 2–4 mm in at least six of eight scenes.
5. Median scene p90 is between 5 mm and the higher real-site p90 plus 1 mm.
6. At least four of eight scenes contain two or more adjacent tail patches on the same local wall grid.
7. Each footprint cohort has p90 improvement in at least three of four scenes.

If both variants pass, prefer the lower 10% nominal tail probability. Passing is provisional support for a synthetic heterogeneity option only; no production YAML update or model retraining is automatic.

## Limitations

- Only two real E57 sites are available; CAL-02 selected the 24 densest eligible wall patches rather than a random sample of all wall surface.
- Original per-station ray provenance is unavailable in both real E57 files.
- Plane residuals combine surface roughness, scanner noise, registration, material effects and patch selection.
- The local patch-grid clustering metric is synthetic-only and cannot be matched to real wall coordinates.
- The smooth tail envelope is a modeling choice, not a measured prevalence distribution.

## Reproduce

~~~bash
python -m pytest -q tests/test_cal07_spatial_tail.py tests/test_realism.py
python tools/validate_cal07_spatial_tail.py --out /tmp/cal07 --seeds 44101 44102 44103 44104 44105 44106 44107 44108 --max-sample 250000
~~~

The workflow uploads only derived JSON/Markdown reports, never raw point clouds or XYZ.
