# Multi-level generator and auxiliary training

The `configs/customer_like.yaml` profile varies duplex slab thickness (0.12–0.35 m), permits shorter wall returns around openings, increases leaning-board clutter in eligible furnished rooms, and mixes random and farthest-point scanner positions. Where space permits, duplexes can include a two-flight U-shaped stair with a landing; existing straight and spiral stairs remain available. The turning-stair setting is a probability per stair-planning attempt, not a guaranteed dataset proportion. The rectangular slab opening is an approximation that should be inspected in generated scans. Scanner variation changes station selection between generated scenes; it does not create multiple independently rescanned versions of the same layout.

New synthetic levels contain `structure_aux.npz`, aligned with `sem.png`, with five binary masks in this order: `stair_footprint`, `floor_opening`, `ceiling_opening`, `railing`, and `landing`. The original eight semantic classes are unchanged. Slab and duplex geometry are recorded in `scene.json`. Existing datasets have no new annotations unless regenerated. The split-preserving merge carries these files on new unique levels and validates their shapes; existing levels and splits remain untouched. The merge report flags duplicate levels whose auxiliary files are not imported.

Optional auxiliary training is enabled with `--use-aux`. Missing files and missing channels are masked from the auxiliary loss rather than treated as negative examples. To start a new training stage from an existing eight-class checkpoint, for example:

```sh
python -m simscan train --data /path/to/merged_dataset --out /path/to/new_run --init /path/to/best.pt --use-aux
```

This initializes shared weights and a new auxiliary head; it starts a new optimizer and schedule, so use a new output directory. To resume that new stage later, use the same output directory with `--resume --use-aux` and omit `--init`. Inference with an auxiliary checkpoint saves `aux_prob.npy` in the channel order above; the standard prediction and vectorizer are unchanged. At least some training levels must have the new sidecars for the auxiliary head to learn; ensure the validation split also contains representative annotated levels for meaningful evaluation.

The mask/geometry and merge checks run without a simulated scan. A complete scanner proof requires Open3D; model training and checkpoint runtime checks require PyTorch. Validate both in the generation and training environments before a large run.

## Hard-scene additions

The customer-like profile can create an upper open-air terrace when an eligible near-rectangular room has a complete, exclusively adjacent exterior wall segment. That conservative requirement avoids turning a neighbouring room's facade into a parapet, so the option can legitimately produce no terrace in a particular duplex. The terrace retains the existing open-balcony policy: it is labelled outside, has its own lowered floor, no ceiling, and can contain non-structural furniture. Shared facade segments are not split. This adds variation in enclosed upper-floor area; it does **not** create a true double-height void or independently replan upper-floor partitions.

Turning stair candidates now reserve a 0.9 m clear approach on both levels, and sample mirrored directions and sides when space permits. Their slab opening is still rectangular; no code-compliance or physical-accessibility certification is implied. Furnished indoor scenes can include groups of upright movable panels, labelled as clutter, and a fraction of acquisitions limit the scanner's elevation field of view. Room-level station omission is also increased in this profile. These are controlled synthetic variations, not measured reconstructions of KZN. The maximum sampled scanner elevation is recorded in the layout metadata.

The slab thickness range is broadened to 0.12–0.55 m as a domain-randomization hypothesis. The separation between detected KZN floor and ceiling planes is not a measurement of slab thickness. Independent upper-level replanning, double-height voids with safe guardrails, true per-station field-of-view changes, and simulated scanner rescans of the same layout remain future work. Reconstruction/solver failures in the KZN case remain separate from the generator changes.
