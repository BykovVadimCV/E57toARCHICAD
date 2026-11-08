# Native HD masks and procedural clutter

The `make-dataset` generator can optionally export full-resolution architectural
wall masks and door/window masks directly from the procedural layout. This is an
**additional geometric annotation**; the normal eight-channel point-cloud input
continues to be rasterized at 20 mm/pixel for compatibility with existing networks.

## Generate at 2 mm/pixel

```bash
python -m simscan make-dataset \
  --config configs/customer_like.yaml \
  --count 100 \
  --variants 2 \
  --workers 2 \
  --hd-masks \
  --hd-pixel-mm 2 \
  --hd-min-size 2048x1152 \
  --out dataset_native_2mm
```

**The default HD pixel size is 2 mm.** `--hd-pixel-mm 2` is shown explicitly
for clarity. Without `--hd-masks`, existing sample generation is unchanged.

Every generated level now contains **paired native high-resolution targets and
point-cloud evidence**:

| File | Contents |
| --- | --- |
| `wall_mask_hd.png` | Architectural wall body, uint8 0/255, aperture pixels cut out |
| `wall_edges_hd.png` | Native single-pixel wall-body edge |
| `openings_hd.png` | Native host-wall aperture: 0 background, 1 door, 2 window, 3 passage |
| `opening_edges_hd.png` | Native single-pixel aperture edge (uint8 0/255) |
| `floor_void_hd.png` | **True missing floor slab only**, uint8 0/255 |
| `stair_footprint_hd.png` | Lower-floor stairs as their **own** target, NOT floor void |
| `supervision_valid_hd.png` | 20 mm supervision-valid regions mapped to the HD frame; 0 = ignore |
| `scan_low_hd.png` | Actual E57 XYZ returns at low elevations, 0/255 occupancy |
| `scan_mid_hd.png` | Actual XYZ returns at middle elevations, 0/255 occupancy |
| `scan_high_hd.png` | Actual XYZ returns at upper elevations, 0/255 occupancy |
| `scan_floor_hd.png` | Actual scan returns within 3 cm of floor |
| `scan_ceiling_hd.png` | Actual scan returns within 3 cm of ceiling |
| `scan_below_hd.png` | Actual returns 0.1–3 m **below** selected floor, helpful for floor-hole detection |
| `wall_mask_hd.json` | Native frame/affine, mask legends, format v2 and floor-void semantics |
| `scan_evidence_hd.json` | Native frame/affine, band ranges, augmentation and hit-count diagnostics |

The masks are created from **original procedural polygons**, never enlarged
`sem.png`. The six scan layers are projected from **actual simulated scanner
returns** at their measured XY locations, with no invented returns or
morphological filling. They are kept separate from the ground-truth geometry.
All sidecars use the **same pixel-to-world affine and 2 mm grid**.

When training a 2 mm refinement network, use the scan layers and coarse
network probabilities as inputs; use the wall, aperture, edge and physical
floor-hole masks as targets, with `supervision_valid_hd.png` as the ignore mask.
Closed door leaves may generate returns in the aperture: this is expected,
because an architectural doorway is not synonymous with empty scan space.

Canvas dimensions grow with the physical footprint: an approximately
12 × 8 m plan takes roughly 6,100 × 4,100 pixels including a small
architectural margin. Minimum pixel dimensions are not a fixed image
aspect ratio. Pixels always represent their requested **metric size**
without stretching the plan.

To avoid accidentally allocating huge images on highly atypical layouts,
the HD renderer refuses frames larger than 64 million pixels and reports
the computed size. Use a smaller generated footprint or increase
`--hd-pixel-mm` for such cases. At 2 mm, each 64-million-pixel uint8 mask
takes 64 MB uncompressed, and rendering requires several mask buffers.
Adjust `--workers` for available RAM.

**Changed class-7 semantics:** `void` now means a physically missing floor
slab, rather than the lower-floor footprint of a stair. That mistaken
stair-as-void supervision could encourage false floor holes in scanned rooms.
The new `meta.json` records
`floor_void_policy: physical_missing_floor_only_v2`.
Do **not** silently merge older labels-5 datasets (without this policy) with
newly generated examples. The `merge_partial_datasets.py` merger now checks
this policy, and building into an old dataset root is rejected. Regenerate
training data in a **new dataset folder** and retrain/fine-tune the segmentation
model before expecting the existing 20 mm model's void predictions to improve.
A high-resolution mask cannot retroactively change an already trained model.

**Reproducibility:** HD settings are saved in `build_contract.json`.
Resume now checks all 2 mm scan and supervision sidecars, their dimensions and
their versioned metadata. Missing or old sidecars cause regeneration, not an
apparently complete dataset. Use a new output directory for a changed generator
source, semantic policy, pixel size or other settings.

## Detailed clutter

`simscan/clutter.py` creates raycastable, geometry-based props: moving
cartons (including flaps), suitcases, buckets, paint cans, vacuum cleaners,
folded ladders, laundry baskets, and stools. In furnished scenes, the
`interior.p_detailed_clutter` configuration controls what fraction of
ordinary floor clutter is replaced with those multipart objects (default:
0.75). All new props are labelled `clutter`, not `wall`, and participate
in the standard scanner simulation.

Example for a clutter-heavy dataset:

```bash
python -m simscan make-dataset \
  --config configs/customer_like.yaml \
  --set interior.p_bare=0.0 \
  --set interior.clutter_per_m2=0.18 \
  --set interior.p_detailed_clutter=1.0 \
  --hd-masks --hd-pixel-mm 2 \
  --count 50 --workers 2 --out clutter_hd_2mm
```

Existing optional external models remain available via
`python -m simscan assets --source gso --out ...` and
`--set interior.scan_dir=/path/to/assets`. External assets are downloaded
at generation-site setup time and are not bundled into this repository;
keep their `ATTRIBUTION.csv` alongside any redistributed model files.
GSO category mapping has also been broadened to recognize a few additional
common clutter categories when those names appear in its manifest.

The new HD masks are not neural-network inputs. Training a model with
actual 2 mm point-cloud tensors requires a separate multiresolution input
and model-training change.

## Complete-apartment acquisition and closed doors

New synthetic scenes require every room on **every level** to have a scanner
station. The scanner validates actual raycast evidence per room before accepting
a scene: by default at least 100 valid returns and 30 wall/column/door/window
returns aggregated from that room's station(s). The expected totals and measured
returns are saved under `layout.json -> meta.coverage.rooms`. A room with
no safe scanner position or insufficient raycast returns causes a hard
generation failure, **never an invisible missing room**.

The generator makes up to four deterministic attempts using distinct random
seeds for a scene in complete-coverage mode; failed partial scenes are cleaned
up. If none passes, the scene is reported as failed rather than being accepted
with missing coverage. Configure via:

```yaml
scanner:
  require_all_rooms_scanned: true
  complete_scene_attempts: 4
  min_valid_returns_per_room: 100
  min_architectural_returns_per_room: 30
  p_room_unscanned: 0.0

interior:
  p_closed_door: 0.45
  min_closed_interior_doors: 1
```

Any level containing interior doors will have at least one with
`angle_deg: 0.0`. The leaf remains a real 3D surface and occludes simulated
laser rays. Each room has its own station even if its doorway is closed,
like a multi-position TLS acquisition where the operator visits every room.
The wall mask continues to identify the underlying architectural opening,
while the scanner observes a solid closed door leaf.

This is a **per-room station and return requirement**, not a guarantee that
100% of every wall surface has line-of-sight coverage. Furniture, closed doors
and scan shadows remain deliberately visible in the simulated point cloud.

Opt out for deliberate incomplete-scan augmentation by setting
`scanner.require_all_rooms_scanned=false`, which restores the older
`p_room_unscanned` behavior and disables the strict acceptance check.
