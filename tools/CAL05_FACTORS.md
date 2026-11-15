# CAL-05: paired sensitivity of physical surface residuals

## Purpose

Estimate how synthetic **observed planar MAD/RMS** changes when exactly one of
wall roughness, station registration translation error, or scanner range noise
is changed, on fixed scene seeds, room geometry and true station poses. This is
**not** identifying physical scanner sigma from two merged real E57 clouds.

## Experimental design

Four predefined independent three-room scenes (`42121..42124`), 0.1° ray grid,
at most two stations per room, no augmentation/thinning. Seven variants per
seed: baseline; roughness low/high (2–4 / 12–20 mm); translation registration
low/high (0.5–1 / 7–11 mm); range noise low/high (0.1–0.2 / 1.2–2 mm).

Each factor varies on its own. The original `customer_like.yaml` remains
unchanged. Before accepting a paired variant, hash the *true* station poses,
room boundaries, walls, openings and layout transform and require an exact
match to the baseline for that seed. Written noisy poses deliberately differ
for registration variants. This check prevents changing the camera placement
and calling the result a one-factor test.

All registered synthetic returns are sampled uniformly, and all surface
patches are fitted with the CAL-02 method. CAL-04 provides synthetic-only
range, observed incidence and station overlap strata. Per-seed MAD deltas
avoid treating many wall patches from one scene as independent samples.

## Trust boundaries and reporting

- Pairing holds the **physical scene and station locations**, but cannot
  force exactly the same noisy return realization; patch selection changes
  with distortion and may alter which surfaces are compared.
- The real E57 exports do not contain row/column or physical station identity.
  Their measured 1.4731 and 2.7332 mm MAD values combine acquisition noise,
  wall material, surface shape and any previous registration error.
- The 24 most-supported wall patches selected by CAL-02 are biased; four
  synthetic layouts and two real sites do not permit device calibration.
- The data do not separately identify registration **rotation** error,
  incidence dependence, mirror effects or intrinsic scanner range sigma.
- All reported range/incidence/overlap groups are synthetic-only. Scarce
  far and grazing samples are flagged, not used to tune production settings.
- No raw points, coordinates or poses enter GitHub artifacts. Production
  scanner settings and `best.pt` / `last.pt` are not modified.

## Reproduce in GitHub Actions

The workflow `cal05-paired-factors.yml` runs the numerical unit tests and 28
fixed-layout scan comparisons. Reports: `cal05_results.json` and
`CAL05_REPORT.md`. Test failures or a changed true-pose/geometry signature
must block calibration conclusions.

Next: CAL-06 tests a held-out set of scene seeds and a fixed, explicitly
reviewed candidate *without* selecting settings from those held-out scenes.
