# Generator fixes and performance changes

The new output contract is dataset metadata version 3, labels-5, generator 0.3.0.
Use a **new output directory**. Existing datasets are read without being rewritten;
generation refuses to resume across changed source, settings, seed or augmentation
parameters. The root `build_contract.json` retains the generation configuration and
source/configuration fingerprint.

## Correctness

- Unknown physical regions use ignore supervision by default, determined by local
  observation support within 30 cm. Room station flags no longer change that policy.
  Known exterior remains a negative example. Physical semantic truth and observation
  support remain separate files. Legacy outside labeling requires an explicit option.
- Training erasing defaults to ignore. Old ignore labels are preserved by default;
  deliberate legacy conversion remains available through the Python loader.
- Polygon labels use pixel-center scanline membership, with a half-open boundary
  convention. Distance targets use continuous structural host faces, including columns,
  rather than raster body edges. A host face across a door is an architectural reference,
  not a claim of a measured solid surface at that height.
- Procedural and external clutter meshes use their projected triangles; no blanket
  bounding-box label is added for hollow mesh interiors.
- Columns/pilasters receive structural face references. `vector_physical.json` retains
  complete physical geometry. `vector.json` retains that geometry plus an explicit
  observation evaluation scope. Wall, opening and room evaluation applies the scope
  to both predictions and references. Hidden physical geometry is retained for analysis
  and is excluded from observed reconstruction scoring, rather than mislabeled outside.
- Floor/ceiling matches beyond 3 cm are rejected before publishing a level. This gate
  is an initial dataset quality setting (`build_scene(max_level_error_m=...)`), not a
  surveying certification. Expected levels must all match distinct detected levels.
  Upper-floor selection prioritizes covered area over return count, and an empty
  candidate set no longer crashes selection.
- Dataset z-shift augmentation now perturbs estimated level references rather than
  translating cloud and levels together. Opening heights are transformed to the
  estimated reference. Detection quality is checked before that intentional perturbation.
- Level completion is atomic and checks required files, shapes and readable arrays;
  finite numerical targets are checked before publishing the marker. Scene completion
  requires all requested variants/levels. Split rebuilding admits completed valid
  levels only, and managed builds require a completed scene manifest as well.

## Performance and resource use

- Training-only generation skips unused high-resolution mask/export construction.
  Standard synthetic E57 exports retain the full export path.
- Mesh assembly concatenates parts once, avoiding repeated copying of the growing mesh.
- Voxel counting chooses bounded dense counting for suitable small buffers and retains
  sorted sparse counting otherwise. Counts still saturate at 65,535; the main buffers
  remain uint16. The dense count temporary is capped at 16 MB.
- Scanner direction grids are cached per worker, with read-only arrays.
- Worker initialization limits competing OpenCV/BLAS/OpenMP thread pools. Workers stay
  explicitly user-selected; the default remains one. Temporary cleanup removes only
  the scene's own files and the empty work directory.
- Dataset builds skip preview rendering. Logs report layout, scene-build, scan and
  export times plus raster/export time, so future optimization can target measured cost.

These changes do not reduce angular resolution, station count, noise effects or beam
model fidelity in a production configuration. A small isolated kernel measurement is
not a production throughput guarantee. Full 1,000-plan throughput was intentionally
not measured on this laptop.

## Run on the generation machine

Use your actual production configuration and a fresh output directory:

```sh
python -m simscan make-dataset --config YOUR_CONFIG.yaml --count 1000 --variants 3 --workers 4 --out dataset_labels5
python -m simscan dataset pack dataset_labels5 --out dataset_labels5.zip
```

Four workers is an example, not a measured optimum. Choose concurrency according to
available RAM; dense raycasting scenes and station clouds still consume substantial
memory. Reusing three variants is cheaper than simulating three independent scans,
but creates three correlated samples, not three independent apartment layouts.

## Local verification

Focused tests cover raster membership under rotation/translation, continuous face
distance, primitive clutter, column vectors, reference perturbation, floor selection,
unknown supervision, symmetric evaluation scope, resume contracts and equivalent mesh
attributes. No test batch-generates apartment plans.

`proof_generator_once.py` generates exactly one clean single-room fixture through
the production scene, scanner-effects, raster, label, manifest, split and loader paths.
Because Open3D is unavailable locally, the script temporarily substitutes a small
NumPy triangle-ray test backend in memory. It writes the backend limitation into
`proof.json`. It does not validate Open3D installation, full realism or production
throughput, and refuses an existing output directory.

```sh
python tools/proof_generator_once.py --out NEW_PROOF_DIRECTORY
```

Before a full production build, run one scene with the actual Open3D backend and
production configuration on that machine. Review its masks and the new timing log.
