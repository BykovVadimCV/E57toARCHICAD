# PROD-00: controlled transition from research to the E57 → IFC pipeline

Status: **INTEGRATION PLAN / NOT YET APPROVED FOR UNATTENDED PRODUCTION**. Research feature work is frozen after TRAIN-01D until a real end-to-end gate has been demonstrated.

## Reuse the working code

- E57 ingestion: `simscan/e57read.py`, `simscan/e57io.py`.
- Existing plan model interface: `simscan/infer.py` and `simscan/net.py` / `simscan/smpunet.py`.
- Geometry: `simscan/wall_pipeline.py`, `simscan/vectorize.py`, `simscan/cellplan.py`.
- IFC: `simscan/export_ifc.py`. Existing CLI: `python -m simscan` and `e57_to_ifc.py`.
- Keep the current converter as the baseline and preserve rollback. Do not merge the failed CAL-09.6 experimental detector as a replacement.

## Explicit model contracts

These model checkpoints **are not interchangeable**:

| Family | Inference representation | Classes | Production status |
| --- | --- | --- | --- |
| Existing PlanNet / compatible SMP U-Net | 8-channel level raster | 8 semantic classes (or 13 outputs with regressions) | Existing pipeline contract; end-to-end acceptance still required |
| TRAIN-01B / TRAIN-01C | 6-channel local 3D voxel blocks | wall/floor/ceiling/door/window/other | Synthetic-only research baseline |
| TRAIN-01D | 7-channel wall-aligned depth/occupancy maps | wall region / door opening / window opening | Auxiliary, experimental, depends on proposed wall planes |

Do **not** copy TRAIN-01C or TRAIN-01D checkpoint to `models/best.pt`: `simscan.infer.load_model` does not support them. Any future adapter needs versioned preprocessing and a separate typed checkpoint loader.

## First shippable slice: review-assisted IFC export

Scope: **E57 → candidate wall geometry + floor/ceiling + plan JSON → IFC4** for review in Archicad.

1. Keep the existing E57→plan→IFC route operational. Record tool versions, input scan count, transforms, coordinate bounds and run options.
2. Represent each candidate as `accepted`, `review_required` or `rejected`, with provenance and supporting scan evidence. Never silently upgrade a weak hypothesis to accepted.
3. Wall topology: verify continuity, L/T junctions and closed contours against scan evidence. If geometry is ambiguous, report the gap and keep `review_required`; do not invent connections.
4. Openings: **default to review_required** unless confirmed by multiple independent cues. Do not cut holes from an IFC wall based on empty voxels alone. Unconfirmed door/window proposals must remain in a diagnostic overlay/report, not as `IfcOpeningElement` / `IfcDoor` / `IfcWindow`.
5. Export accepted geometry using existing IFC routines and reopen/validate IFC entities, placements, units, void/fill relations and Archicad import. Export provenance and rejected/review-required proposals separately.
6. Output a machine-readable acceptance report, QA imagery and a reproducible run manifest next to `plan.json` and `*.ifc`. Distinguish not observed from confirmed absent.

## Experimental module integration behind an explicit flag

- TRAIN-01D may add **proposal scores** to an already identified wall: raw XYZ + *blind* wall-plane hypothesis → wall-local 7-channel features → opening probabilities → geometric verification.
- No GT plane, opening rectangle, scene seed, or synthetic coverage mask may be consumed at real inference.
- Calibrate/abstain on low coverage, conflicting scales (25 vs 50 mm), uncertain frame geometry or unstable wall-plane hypotheses.
- Train/Eval remain on Colab. Production requirements must pin checkpoint hash, schema version, voxel/pixel scales, dependencies, model signature and preprocessing code.
- Keep all new functionality disabled by default; existing PlanNet routing and IFC export are rollback.

## Acceptance gates for promotion

**Gate A: operational E2E**. A clean environment can run a real E57 from start to finish; produces nonempty plan+IFC+QA report; no coordinate/units drift; Archicad opens the IFC.

**Gate B: conservative geometry**. On independently checked real regions, compare wall support, L/T junctions, topological connectivity, opening false positives, and geometric offsets. A successful synthetic test is not evidence for this gate. The available single compact E57 is **not** independent GT and lacks detail below ~4 cm.

**Gate C: release safety**. No silent false-positive openings; ambiguous hypotheses remain review-required; audit trail to raw scan; regression tests, pinned releases, feature flag and rollback.

Promotion decision: do not call the converter **unattended production-ready** until A/B/C pass. A review-assisted preview may ship earlier with its limitations prominent.

## Execution order (no further open-ended research rounds)

1. PROD-01: wire the review-state/provenance/reporting contract into the current CLI without replacing the current model.
2. PROD-02: run current E57 through the existing plan/IFC pipeline, inspect Archicad output and fix concrete failures.
3. PROD-03: integrate the TRAIN-01D opening proposal adapter behind a feature flag; ablate against geometry-only proposals.
4. PROD-04: freeze an independently checked real-world acceptance set and run release gates A/B/C. Only promote passing features.

Current hard blocker: the blind CAL-09.6 wall-topology prototype failed development gates (point recall ~53%, edge recall ~12.5% on its tested dev variant). It remains **Draft / do not merge**. It is not safe to silently promote its output as finished geometry.

## Scope freeze

No TRAIN-01E, no more generator or loss-function exploration until the above operational integration work produces a specific blocking defect requiring research. A Colab improvement is not itself a release.
