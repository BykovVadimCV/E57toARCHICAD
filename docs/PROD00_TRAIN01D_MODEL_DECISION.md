# PROD-00 decision: TRAIN-01D checkpoint selection (2026-11-16)

## Source and verification

Source: user-provided `TRAIN01D_model_bundle.zip`. Archive ZIP integrity checked. Both PyTorch checkpoints loaded with `torch.load(..., weights_only=True)` on CPU. All floating-point state tensors are finite, and both have matching 7-channel/3-class architecture metadata.

The two checkpoint files are **not included in this PR**; they require an explicitly versioned model-asset publication process with checksum verification.

| Checkpoint | SHA-256 | Best epoch | Best validation score |
| --- | --- | ---: | ---: |
| `train01d_best_50mm.pt` | `8197030075735aaf63c695493315d698a91738a7dcbdc873adc765bee76ea45a` | 4 | 0.772208 |
| `train01d_best_25mm.pt` | `825326fb734b0fb12d1ac8b7b8cbe9f83c5eaab0b793d71a15045797f66d835d` | 5 | 0.672193 |

Metadata:
- `schema_version=1`
- `base_channels=12`
- Input: seven wall-aligned XYZ-derived features: `log_near_returns`, `log_recessed_returns`, `log_foreground_returns`, `log_deep_returns`, `mean_signed_depth`, `depth_std`, `relative_height`.
- Output: `0=wall`, `1=door_opening`, `2=window_opening`.
- Model family: independent 2D wall-local U-Net with auxiliary edge head; **not** directly loadable by `simscan.infer.load_model` (which currently accepts PlanNet/SMP raster checkpoints).

## Dataset and test limitations

TRAIN-01D smoke data: 12 synthetic procedural scenes (8 train, 2 val, 2 test); test consists of **16 complete wall maps from only 2 scenes**. The 25 and 50 mm tests use the same scene IDs and walls, but different cell grids. Synthetic wall planes are known exactly at generation time, and synthetic scanner coverage defines `unknown=255`. Neither condition has yet been reproduced on real E57 data. These are not independently confirmed real-cloud ground truths.

Test metrics (entire wall maps; unknown excluded):

| Test metric | 50 mm | 25 mm |
| --- | ---: | ---: |
| Overall mIoU | 0.775511 | 0.715890 |
| Opening mIoU | 0.672718 | 0.577211 |
| Door IoU | 0.822241 | 0.660550 |
| Door precision | 0.867477 | 0.906408 |
| Door recall | 0.940362 | 0.708902 |
| Window IoU | 0.523194 | 0.493872 |
| Window precision | **0.528616** | **0.496784** |
| Window recall | 0.980769 | 0.988270 |

50 mm test confusion matrix (GT rows, prediction columns in `wall / door_opening / window_opening` order):

```text
wall            82113    436    994
door_opening      129   2854     52
window_opening     23      0   1173
```

Therefore 50 mm predicts 1,046 false-positive window cells (including 994 from actual wall), with about 52.9% window precision. These counts are **cells, not distinct opening objects**. Object-level false-positive opening counts and geometric tolerances remain unmeasured.

## Integration decision

**Preferred exploratory opening-proposal checkpoint: 50 mm**, selected on the synthetic results. Keep 25 mm only as a secondary diagnostic/ablation.

**Not approved** as an unattended IFC opening cutter. Proposed use in PROD-03:
1. Obtain a blind, provenance-bearing wall-plane hypothesis from real XYZ and infer its floor datum.
2. Compute the exact same seven feature channels on a 50 mm wall-local grid.
3. Load the above checkpoint by its SHA-256 and strict schema/class-order/scale contract in an independent, optional adapter; DO NOT feed it to existing PlanNet checkpoint loader.
4. Generate diagnostic opening hypotheses with `review_required` status and geometry/support metadata.
5. Do **not** write `IfcOpeningElement`, `IfcDoor` or `IfcWindow` from this model until the proposal has an independent confirmation and the real acceptance gate is passed.
6. Record `unknown/unobserved` separately from `wall without opening`. Never interpret empty returns alone as evidence of an opening.
7. Run current E57→plan→IFC pipeline first (PROD-01/02); no more TRAIN-01E or unbounded synthetic experiments unless a specific integration defect requires one.

**Release gate:** reliable blind wall planes/topology on real scans, independently verified real opening ground truth, opening-object false-positive rate, plausible sill/head/width geometry, reproducible Archicad IFC import, and rollback to legacy behavior. No such real-world acceptance is claimed here.
