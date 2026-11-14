# Overnight integration verification — 2026-11-11

Base: GitHub main 091fb4f3ac4300d7df4d48fa7ae367d42e7e2d07. Local integration branch: integration/overnight-hd. Existing repo/ and publish-generator/ work was preserved; this checkout contains the updated generator and reviewed local geometry modules together.

Focused suite: 178 passed, 2 skipped. The suite covers HD world alignment, real-return projection with retained multiplicity, native output completion, geometry-target/input separation, split-preserving HD merges, duplicate identity, process exit status, empty completion markers, chunked point-source equivalence, clutter, doors/windows, staged wall fitting and protected long-face gravity. Compilation and git whitespace checks passed. After adding generator runtime-version binding, the 18 contract/CLI checks passed again.

The two optional local checks require Open3D and Torch. An isolated Python 3.11 dependency install was attempted; the native E57 binding cannot build on this Intel Mac, and the separate raycasting/model dependency download then exhausted available disk space. All temporary installer/runtime files and its Python alias were removed. No dataset batch was generated. The actual native file writer was exercised with constructed point returns; this is not claimed as a physical scanner simulation.

A full raycast/GPU training proof was not completed locally. The Windows runner therefore performs dependency checks and one complete production-setting scene before proceeding to the requested count; it rejects any generation errors and performs a deep final hash/split audit. The CI workflow also covers Open3D occlusion and a small actual 2 mm scene, plus a CPU refiner forward/backward check, when this branch is pushed. Those remote CI results are not yet verified.

The separate HD refinement architecture is experimental and has not demonstrated improved real-scan accuracy. Existing PlanNet checkpoints can continue through the legacy trainer, but cannot directly initialize the new 15-input HD model. Corrected physical-void labels cannot be silently mixed with historical stair-as-void datasets.

The Windows WMI launcher is provided for the server/SSH environment; it could not be executed on macOS. Wait for PREFLIGHT PASSED in its UTF-8 log. A process ID alone is not confirmation that generation has passed preflight. The original source and datasets remain available for rollback.
