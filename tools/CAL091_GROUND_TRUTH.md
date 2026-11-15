# CAL-09.1: ground-truth spatial validation of CAL-09

This is a **diagnostic experiment**, not physical scanner calibration and not
an acceptance of CAL-09's failed registered-E57 patch criteria.

CAL-09 (uploaded Colab result: 40/40 scans, 30/30 regression tests) returned
\`NO_CAL09_VARIANT_PASSED\`. Highest reported median p90 values were 3.56mm
(zone08/fine35) and 3.34mm (zone18/fine35); no variant met the four-of-eight
connected-patch gate. Those outcomes remain unchanged.

## Reproducible experiment

Run from the repository root, on this branch:

\`\`\`bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_cal091_mesh_truth.py tests/test_cal09_two_scale_defects.py \
    tests/test_cal08_tail_zones.py tests/test_cal07_spatial_tail.py
PYTHONPATH=. python tools/validate_cal091_ground_truth.py \
    --out results/CAL091 --save-truth one
\`\`\`

The full experiment contains **8 CAL-09 seeds × 5 variants × 3 wall tessellation
resolutions (0.30, 0.15, 0.10m)**. It **does not simulate scanner rays**. The
initial layout and wall thickness jitter are regenerated with the same scene
RNG for each pair. Only wall geometry is extracted from the actual
\`SceneBuilder\` / \`SceneMesh\` classes, keeping scanner calibration fixed.

Completed seed-variant-resolution rows are cached with an SHA-256 of the source
YAML. Re-running the same command resumes completed work; changing the source
YAML invalidates cached records. Results are written to
\`cal091_results.json\`, \`CAL091_REPORT.md\`, and
\`cal091_progress.json\`. The default \`--save-truth one\` saves an example
\`mesh_truth/*.npz\` for the first seed and finest resolution.

## Ground-truth definition

- Evaluate envelope at undeformed mesh vertices, as the real deformation
  function does.
- Exclude horizontal faces and non-wall triangles.
- Mark an original triangle as defective when mean vertex envelope >= 0.5.
- Calculate areas on the pre-deformation triangle geometry in square metres.
- Weld duplicated box-face vertices by quantized XYZ within 10µm, but only
  within a common wall instance. Adjacency requires a whole shared edge
  rather than a single vertex.
- Report component count, largest component area, fraction of area in the
  largest component, fragmentation \`1 - largest_area / defect_area\`, and
  equivalent circular diameter. The diameter is an *area-derived equivalent*,
  not the true maximum linear extent.
- Report distance-binned normalized pair correlations for coarse envelope
  and masked fine texture, with null for poorly sampled distance bins.
- Compute *paired additive residual* as displacement with CAL-09 minus
  displacement with tail probability zero using the same \`w\`, phases and
  lean. Report RMS of the residual outside the envelope's exact support.

The truth mask is an **interpolated mesh-envelope mask**, and hence can vary
with tessellation. This is precisely what CAL-09.1 is intended to assess.

## Predetermined algorithmic gates

1. Eight seeds present.
2. All three resolution levels present.
3. Paired layout hashes and original mesh vertex hashes match within seed
   and resolution.
4. Changing fine scale does not move the envelope.
5. Changing zone scale does not change the fine RFF texture.
6. Outside-support additive RMS <= 1e-8 mm.
7. Area fraction changes by no more than 0.02 and total original wall area by
   no more than 0.5% between 0.15m and 0.10m, in >=6/8 seeds **per variant**.

The 0.30m metrics are reported for comparison. These are *engineering
diagnostics*, not preregistered real-world efficacy gates; passing does not
mean defect realism has been validated against real E57.

## Current limitations

- Ground truth is restricted to vertical wall faces, not openings and slabs.
- Mesh seam welding does not bridge different wall IDs; fragmentation can
  therefore include architectural splits.
- The geometry-only replay omits Furnisher and all scanner ray effects.
- Fine-scale autocorrelation is measured only in envelope core >=0.95, and
  may be missing for small zones.
- Real E57 patch samples remain biased toward top-24 dense candidate patches;
  the number and provenance of physical stations are unknown.
- The eight seeds are **reused** from CAL-09, not independent holdout data.
- Production YAML, original real E57, GT masks, and model weights are untouched.

Use the accompanying Colab notebook for the complete run. Do not merge this
experimental work into production before reviewing the saved JSON, passing
tests, and planning an independently seeded physical-E57 holdout.
