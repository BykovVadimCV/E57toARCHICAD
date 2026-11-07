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

Every generated level receives four additional files:

- `wall_mask_hd.png`: grayscale uint8, wall = 255, background = 0;
  door/window openings are removed from the solid-wall mask.
- `openings_hd.png`: grayscale uint8, background = 0,
  door = 1, window = 2, full-height passage = 3.
- `wall_edges_hd.png`: single-pixel boundary of the solid-wall mask.
- `wall_mask_hd.json`: world-coordinate frame, `pixel_to_world` affine,
  resolution, legend, floor level and augmentation rotation.

The masks are created from the **original procedurally generated geometry**.
They do not resample `sem.png`, or amplify noise in point-cloud rasters.
Wall and opening annotations share exactly the same physical frame.

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

**Reproducibility:** HD settings are saved in `build_contract.json`,
and the generator checks that all enabled HD sidecars exist and have
the correct dimensions when resuming. Use a new output directory for
a changed pixel size or any change to generator source/settings.

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
