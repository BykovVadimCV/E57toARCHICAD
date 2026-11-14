# Native HD overnight build

Use Python 3.11 or 3.12 on the Windows server (CI uses 3.11; the server’s earlier logs showed 3.12). Unpack the supplied source bundle into a fresh directory. Keep the old generator checkout and datasets until the new run is audited.

Windows PowerShell, in the extracted source folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=pye57 -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q tests/test_hdmask.py tests/test_hd_evidence.py tests/test_hd_training_data.py tests/test_merge_preserve_splits.py tests/test_full_scan_and_doors.py
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\start_overnight_windows.ps1 -Output D:\remote_e57\dataset_native_hd_v3 -Count 1000 -Workers 0
Get-Content .\logs\overnight_hd.log -Encoding UTF8 -Tail 30 -Wait
```

The launcher uses a local WMI-created process to detach from SSH. It prints the process ID; monitor the log before disconnecting. If WMI returns a permission error, the launcher reports it rather than pretending the job started. Do not launch a second copy against the same output folder. After a killed/crashed run, verify the prior process has stopped before removing `generation.lock` and rerunning.

Linux:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
mkdir -p logs
nohup .venv/bin/python -u tools/run_overnight_dataset.py --out /path/to/dataset_native_hd_v3 --count 1000 --workers 0 --seed 31111 > logs/overnight_hd.log 2>&1 < /dev/null &
```

The runner checks imports, generates the first scene at the production settings, verifies its native outputs, then proceeds to the requested count. It audits SHA256 hashes and split grouping after generation. Restart the same command to skip finished scenes and retry missing ones. The last log line must be `COMPLETE`; `make_log.jsonl` records per-scene failures. With `--workers 0`, the runner chooses up to four workers from CPU count and currently available RAM, budgeting 12 GiB per worker. This is an estimate: use `--workers 1` if memory is tight, or specify a tested count. The scanner stays at the customer-like 0.1 degree resolution; native raster pitch is 2 mm. No overnight runtime or completion count is guaranteed.

For a small actual scanner proof, use a **different output folder**:

```sh
python tools/run_overnight_dataset.py --out /path/to/hd_smoke --smoke
```

The smoke profile uses a coarse angular step for speed; it is not the production dataset profile. Do not use `--keep-scans` for an overnight build unless you have disk space for all raw E57 files. The normal runner retains compressed learning data and removes temporary scans.

Each level contains the legacy 20 mm input/targets, independent native 2 mm labels and six raw-return bands, exact uint32 return counts, and an HD completion manifest. HD sidecars survive both merge tools. Duplicates include auxiliary and HD-manifest identity; existing split assignments remain anchored with `merge_dataset_preserve_splits.py`. Historical stair-as-void datasets are incompatible with the corrected void contract: train on a fresh dataset or regenerate them before merging.

HD training is now separate from legacy PlanNet training:

```sh
python -m pip install -e '.[train]'
python -m simscan.train_hd --data /path/to/dataset_native_hd_v3 --out runs/native_hd --epochs 30 --crop 512 --batch 4
```

The refiner reads six native scan bands, log raw counts and eight coarse **input** features sampled into each crop by world coordinates. Geometry labels never enter the input. It predicts background/wall/door/window/passage/physical-floor-void and boundary targets. It starts a new model; legacy PlanNet weights cannot initialize its architecture. `--resume` restores the same HD run and rejects a changed dataset/settings identity. Its validation metric uses fixed uniformly sampled crops, not full-level or real-scan accuracy. Validate on the real benchmarks before promoting the new model. At 2 mm, a 512-pixel patch covers about one metre; full-plan context and real-domain validation remain important.

The recent local decomposition and IFC modules and their tests are included as a separate geometry path. They do not replace the current generator, alter synthetic labels, or silently switch the legacy `vectorize` command. Exported opening heights remain provisional unless supplied by measured data.

The reviewed area-fitting path is available for saved native masks as:

```sh
python -m simscan.area_pipeline --mask wall.png --barrier dense.png --forbidden apertures.png --frame frame.json --out fitted
```

Use density-qualified barriers and identical raster frames. This entry point exports walls and stage audits; doors/windows can then be fitted from coarse model probabilities by the included fitting functions. It does not assume HD target masks are predictions or auto-run a model during generation.
