# PR-01: physical opening semantics and dataset migration

The generator's `Layout.Opening.kind` is the authoritative physical type.  A
`passage` is a wall opening **without a door leaf**; a `door` has a door leaf;
a `window` is an elevated window. `z0`/`z1` specify geometry, not type.
A passage may have a lintel. Conversely a high door remains a door.

`simscan/semantic_contract.py` centralizes numerical IDs. Neither output head
size nor channel order changes:

| Physical kind | Coarse `sem.png` | HD `openings_hd.png` | HD network class |
|---|---:|---:|---:|
| door | 3 | 1 | 2 |
| window | 4 | 2 | 3 |
| passage | 5 | 3 | 4 |

The coarse class-5 name `opening` is retained as a *legacy downstream API
alias*. Vector reference entries keep `type: "opening"` for passages so
existing IFC/vector consumers still work, and gain `physical_kind: "passage"`.
Unknown kinds now raise an exception rather than silently becoming doors.

## Version break

- Coarse metadata `version: 4`, `labels: "labels-6"`,
  `opening_semantics: "physical-kind-v1"`. Scene metadata uses the same
  version and opening semantics.
- Native-HD `wall_mask_hd.json` uses
  `format: "simscan-native-hd-masks-v3"`, the semantic identifier, and
  an explicit legend. `complete_hd_level` and `load_patch` reject old-format
  HD labels for *new* training.
- The two dataset merge scripts can still merge old labels-5 datasets with
  each other and new labels-6 datasets with each other, but **not across
  versions**. In-place resuming an old dataset with new code is refused.
- No numerical channel/class ID has changed. Old checkpoints and old data
  are left intact for baseline comparison. However a model trained using
  historical class-5 targets cannot be reinterpreted as trained passage
  detection. Train a new checkpoint on regenerated labels-6 data.

Regenerate labels from original procedural scenes (not from a saved mask).
Do not assign physical type by inferring it from height on old raster masks.

## Scope

PR-01 only fixes semantic IDs, rasterization contracts, metadata validation and
regression tests. It does **not** claim to solve occlusion validity, unsupported
positive target supervision, or real-scan segmentation. Those are subsequent
stages and need separate experiments.

`source_manifest.json` remains a historical overnight-ZIP provenance manifest; its
hashes describe the Stage-0 archive, not this edited Git branch.
