# HD scanning density calibration

## Source of truth

The reference is three native 2 mm **scan count** projections of two real E57 clouds, not a neural network's predicted masks. The primary metrics are the number of structural laser returns per frame m², distinct occupied 2 mm cells per frame m², and the fraction of occupied 20 mm cells. Real cloud hashes and independent Stage 0 scan counts were checked earlier.

These three levels contain 21,871,717 and 36,002,688 source points, respectively. One E57 data3D scan is recorded in each. Their point density alone **cannot identify** the instrument's horizontal/vertical angular step: the E57 data3D node may be a merged cloud. Also, point density, range noise, beam effects, registration error, and furniture occlusion are different physical parameters.

## Design of experiment

The GitHub workflow creates the *same small scene* at 0.15°, 0.1° and 0.0667°, with the same station placement rules, two maximum stations and no thinning. Every run writes native 2 mm counts. The comparison computes identical observable statistics for real and synthetic data.

This is a *sensitivity sweep*, not an absolute optimization of the BLK360 scanner: the real E57 images came from much larger multiroom environments with unknown counts of original stations. The procedure cannot guarantee equivalence and **does not automatically change** `customer_like.yaml` or retrain.

## Further calibration still required

1. Inspect `rowIndex`, `columnIndex` and station pose/grid data in the original E57, if present. If a file is a merged cloud, the original station angular step is not observable from those points alone.
2. Obtain or infer station count and overlaps, and compare the same statistic within matched local wall patches rather than whole-image averages.
3. Analyze real planar wall patches (RMS thickness, grazing-angle behavior, spatial noise), then separately tune `registration_sigma_mm`, `range_sigma_mm`, `incidence_noise_power` and beam parameters.
4. Repeat with several random seeds, realistic scene footprint/station density and full 2 mm output. No single-scene sweep can justify production tuning.
5. Preserve current external-geometry and visibility guards. More density must **not** transform unobserved GT into positive labels.

No scanner models, angular steps, wall RMS or device settings may be claimed from this comparison without a measurement supporting them.
