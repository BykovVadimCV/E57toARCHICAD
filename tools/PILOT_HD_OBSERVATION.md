# Pilot HD observation audit: 10 synthetic scenes

## What this validates

The pilot checks what fraction of the **measured wall-face candidates** is retained as
wall training targets (and separately how much physical polygon GT is excluded) under the *observability-v3* masks, and whether ray-crossing
proof can support label-free passages. It checks the actual 2 mm image masks,
validates complete-scene hashes and level splits, and quantifies retention
separately for wall, door, window, passage, and physical upper-floor void.

**This does not measure network hallucination, true structural hit identity,
real-world E57 accuracy, or reconstruction quality.** A nearby sensor return
might be clutter. A path through one z-band does not make a whole 2D XY column
free. Unobserved XY does not mean non-wall.

## Two different cohorts, never merged for training

1. Eight *baseline* scenes use the active source profile (`customer_like.yaml`),
   but the default **diagnostic** mode deliberately switches to simple 2–4 room
   layouts, 6–8.5 m footprint, no duplex, 0.45° angular step for cheap CI.
2. Two *passage_enriched* scenes use the same diagnostic acquisition, but
   increase `layout.p_passage` to 0.75 to exercise independent jamb/ray proof.

Both cohorts retain `require_all_rooms_scanned: true`, `p_room_unscanned: 0`,
original station/return coverage minimums, thinning setting (30% chance), and
**2 mm** rasterization. Changes are explicit in `pilot_setup.json` and separate
`build_contract.json` files. `mode=full` retains the unmodified production
scanner and floorplan configuration in the baseline cohort; passage enrichment
remains deliberately distinct. Full mode may require substantial CPU, memory,
and disk space: do not run it accidentally on a small CI machine.

## Commands

From repository root with Open3D installed:

```bash
PYTHONPATH=. python tools/run_hd_pilot.py --out /tmp/e57_pilot10 --seed 31111 --mode diagnostic --workers 1
PYTHONPATH=. python tools/audit_hd_observation_pilot.py \
  /tmp/e57_pilot10/baseline /tmp/e57_pilot10/passage_enriched \
  --expected-scenes 10 --out /tmp/e57_pilot10/reports
```

GitHub workflow `.github/workflows/pilot-hd-10.yml` automatically runs once
when first added to `fix/native-hd-v4`. It uploads complete diagnostic output
as a short-lived artifact (7 days), and an audit report (14 days).

## Principal quantitative indicators

- Physical class area at native 2 mm, train-valid pixels, **wall-face candidate area**,
  **observed wall-face training coverage**, and physical-GT retention as separate metrics.
- Total unsupported-positive fraction. It is allowed for actual physical GT to
  have unsupported areas; *only their inclusion in training* is a hard error.
- Per-passage `confirmed_for_supervision`, <=3 separated ray strips, and strict
  **20% per-opening raster-area cap** checked against measured-ray metadata.
- Mean return proximity per wall's *two faces* (upper bound, not identity).
- Per-opening fraction with nearby returns at *both* jambs.
- Retention of existing negative background supervision. Current rule must
  keep 100% of legacy negative validity and not infer new negatives from rays.
- Integrality, shape, provenance, pixel-mask invariants; scene count and
  train/validation/test grouping; SHA-256 of generated files.

### Interpretation gates

- Hard failure: inconsistent masks, unsupported positive supervised, passage
  proof outside the passage, missing ray provenance, changed background
  validity, incomplete hash or layout leakage, fewer than 10 scenes.
- Warning (not automatic failure): zero passage examples, zero proven passage
  objects, any class with very poor valid-pixel retention, unexpectedly sparse
  ray paths, or a baseline without the intended class types.
- No model retraining until the diagnostic results are manually examined and
  representative **full 0.1° acquisition** data has been audited.
