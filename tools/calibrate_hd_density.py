"""Compare actual 2-mm laser-return density with synthetic native HD evidence.

Only *observed scan returns* are used. These measurements cannot determine
scanner angular resolution from merged XYZ or separate registration/noise effects.

Examples:
    python tools/calibrate_hd_density.py --real-zip hd_inference.zip \
        --synthetic-roots datasetA datasetB --out calibration/
    python tools/calibrate_hd_density.py --reference real_evidence_reference.json \
        --synthetic-roots sweep/angle_0p10 --out calibration/
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import zipfile
from pathlib import Path

import numpy as np

VERSION = "native-hd-density-calibration-v1"
COARSE_PITCH_M = 0.02


def evidence_metrics(counts: np.ndarray, frame: dict, evidence: dict, label: str,
                     kind: str, floor_area_m2: float | None = None) -> dict:
    """Physical metric normalizations, not trained semantic predictions.

    Some real exports repeat *all* points per floor. Metrics use the actual
    height-limited structural count channel; do not interpret as scanner rays.
    """
    px = float(frame["pixel_m"])
    height, width = int(frame["height"]), int(frame["width"])
    if (not 0 < px <= COARSE_PITCH_M or counts.shape != (height, width)
            or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError("Invalid native raster count/frame contract")
    block = round(COARSE_PITCH_M / px)
    if abs(block * px - COARSE_PITCH_M) > 1e-8:
        raise ValueError("The coarse 20-mm comparison requires integer pixel aggregation")
    occupied = counts > 0
    n_occupied = int(np.count_nonzero(occupied))
    n_returns = int(counts.sum(dtype=np.uint64))
    grid_area = px * px * height * width
    if height < block or width < block:
        raise ValueError("HD frame too small for coarse comparison")
    H, W = height // block, width // block
    occ20 = occupied[:H*block, :W*block].reshape(H, block, W, block).any(axis=(1, 3))
    bands = evidence.get("occupied_cells_by_band", {})
    out = {
        "label": label, "kind": kind,
        "pixel_m": px, "frame_area_m2": round(grid_area, 6),
        "pixel_count": height * width, "structural_returns": n_returns,
        "occupied_2mm_cells": n_occupied,
        "returns_per_occupied_2mm_cell": round(n_returns / max(n_occupied, 1), 3),
        "structural_returns_per_frame_m2": round(n_returns / grid_area, 3),
        "occupied_2mm_cells_per_frame_m2": round(n_occupied / grid_area, 3),
        "occupied_2mm_fraction": round(n_occupied / (height * width), 7),
        "occupied_20mm_fraction": round(float(occ20.mean()), 7),
        "multi_return_fraction_among_occupied_2mm": round(
            float(np.count_nonzero(counts > 1)) / max(n_occupied, 1), 7),
        "synthetic_thinning": float(evidence.get("thin", 0.0)),
        "band_occupied_cells": {k: int(v) for k, v in bands.items()},
    }
    if floor_area_m2 and float(floor_area_m2) > 0:
        out["floor_area_m2"] = float(floor_area_m2)
        out["structural_returns_per_floor_area_m2"] = round(n_returns / float(floor_area_m2), 3)
    return out


def from_real_zip(path: Path) -> list[dict]:
    """Read archived *raw scan counts* from real E57 inference, not model masks.

    Only frame.json and scan_counts_hd.npz are trusted. No model predictions are
    used when assessing the scan acquisition density.
    """
    result = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith("/frame.json") or name.startswith("__MACOSX/"):
                continue
            prefix = name[:-len("frame.json")]
            fn = prefix + "scan_counts_hd.npz"
            if fn not in z.namelist():
                raise ValueError(f"Missing source returns for {name}")
            info = json.loads(z.read(name))
            with np.load(io.BytesIO(z.read(fn)), allow_pickle=False) as data:
                val = evidence_metrics(data["counts"], info["frame"], info["evidence"],
                                       f"real_level_{len(result)+1}", "real",
                                       info.get("level", {}).get("floor_area_m2"))
            result.append(val)
    if not result:
        raise ValueError("No measured E57 HD evidence in archive")
    return result


def from_synthetic_roots(paths: list[Path]) -> list[dict]:
    result = []
    for root in paths:
        if not root.exists():
            raise FileNotFoundError(root)
        for file in sorted(root.glob("**/L*/scan_evidence_hd.json")):
            info = json.loads(file.read_text(encoding="utf-8"))
            countfile = file.parent / "scan_counts_hd.npz"
            with np.load(countfile, allow_pickle=False) as data:
                val = evidence_metrics(data["counts"], info["frame"], info,
                                       file.parent.relative_to(root).as_posix(),
                                       "synthetic")
            scene = json.loads((file.parent / "scene.json").read_text(encoding="utf-8"))
            val["station_count"] = len(scene.get("stations_world") or [])
            val["source_root"] = root.name
            result.append(val)
    if not result:
        raise ValueError("No synthetic HD evidence found in requested roots")
    return result


def distribution(rows: list[dict]) -> dict:
    """Use unthinned scenes only, and report number of levels for context."""
    rows = [r for r in rows if float(r.get("synthetic_thinning", 0.0)) == 0.0]
    if not rows:
        raise ValueError("No unthinned evidence available")
    fields = ("structural_returns_per_frame_m2", "occupied_2mm_cells_per_frame_m2",
              "occupied_20mm_fraction", "multi_return_fraction_among_occupied_2mm",
              "returns_per_occupied_2mm_cell")
    return {"n_levels": len(rows),
            "metrics": {k: {"median": round(float(statistics.median(r[k] for r in rows)), 6),
                            "min": round(float(min(r[k] for r in rows)), 6),
                            "max": round(float(max(r[k] for r in rows)), 6)}
                        for k in fields}}


def compare(real: list[dict], candidates: list[dict]) -> dict:
    """Comparison of observables. Never fit/declare a sensor from a merged cloud."""
    rb = distribution(real)
    out = {"format": VERSION, "real_levels": len(real), "synthetic_levels": len(candidates),
           "real": rb, "candidates": {},
           "limitations": [
               "Native HD evidence measures projected returns, NOT per-station angular resolution.",
               "E57 single merged scan may have no row/column indexing or original station ray paths.",
               "Different station counts, layout area, clutter and overlapping scans confound density.",
               "Calibration candidates are diagnostic: no automatic overwrite of customer_like.yaml.",
               "Range sigma, incidence dependence and beam artifacts require original organized scans or validated planar patches.",
               "These diagnostics do NOT measure model quality or false-positive architectural geometry.",
           ]}
    groups = {}
    for r in candidates:
        if float(r.get("synthetic_thinning", 0.0)) != 0.0:
            continue
        groups.setdefault(r.get("source_root", "synthetic"), []).append(r)
    if not groups:
        raise ValueError("Synthetic calibration contains no unthinned levels")
    for name, rows in sorted(groups.items()):
        x = distribution(rows)
        ratios = {}
        for key, val in x["metrics"].items():
            target = rb["metrics"][key]["median"]
            ratios[key] = round(val["median"] / max(target, 1e-12), 5)
        within_30pct = all(0.7 <= ratios[key] <= 1.3 for key in
                         ("structural_returns_per_frame_m2", "occupied_2mm_cells_per_frame_m2"))
        out["candidates"][name] = {"density": x, "synthetic_over_real": ratios,
                                     "two_primary_metrics_within_30pct": within_30pct,
                                     "station_counts": sorted({r.get("station_count") for r in rows})}
    out["any_angle_matches_both_primary_density_metrics"] = any(
        c["two_primary_metrics_within_30pct"] for c in out["candidates"].values())
    return out


def make_report(report: dict) -> str:
    lines = ["# Калибровка плотности 2-мм HD-признаков", "",
             "Измерено по *фактическим лазерным отражениям*, не по предсказаниям модели.",
             f"Реальные уровни: {report['real_levels']}. Синтетические: {report['synthetic_levels']}.", "",
             "| Группа | Отражения/м² кадра | Занятые 2-мм ячейки/м² | Отражения на занятую ячейку | Занято ячеек 20 мм |",
             "|---|---:|---:|---:|---:|"]
    def row(name, x):
        m = x['metrics']; return (f"| {name} | {m['structural_returns_per_frame_m2']['median']:,.0f} | "
            f"{m['occupied_2mm_cells_per_frame_m2']['median']:,.0f} | "
            f"{m['returns_per_occupied_2mm_cell']['median']:.1f} | "
            f"{100*m['occupied_20mm_fraction']['median']:.2f}% |")
    lines.append(row("Реальные E57", report["real"]))
    for name, item in report["candidates"].items():
        lines.append(row(name, item["density"]))
    if not report.get("any_angle_matches_both_primary_density_metrics"):
        lines.extend(["", "**Ни один кандидат не приблизился одновременно к обеим основным наблюдаемым метрикам",
                      "в пределах диагностического допуска ±30%. Значит, изменение только углового шага",
                      "не является обоснованной калибровкой этого эксперимента.**", ""])
    lines.extend(["", "## Ограничения интерпретации", ""])
    lines.extend(f"- {x}" for x in report["limitations"])
    lines.extend(["", "Вывод о выборе конкретного углового шага нельзя делать только по этой таблице.",
                  "Нужна проверка сетки исходных сканов, числа станций и толщины измеренных поверхностей."])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--real-zip", type=Path, help="Archived E57 inference frame.json + scan_counts_hd.npz")
    p.add_argument("--reference", type=Path, help="Previously measured aggregate real reference")
    p.add_argument("--synthetic-roots", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if bool(a.real_zip) == bool(a.reference):
        p.error("Exactly one of --real-zip or --reference must be supplied")
    real = from_real_zip(a.real_zip) if a.real_zip else json.loads(a.reference.read_text(encoding="utf-8"))["levels"]
    syn = from_synthetic_roots(a.synthetic_roots)
    result = compare(real, syn)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "density_calibration.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (a.out / "density_calibration.md").write_text(make_report(result), encoding="utf-8")
    print(make_report(result))


if __name__ == "__main__":
    main()
