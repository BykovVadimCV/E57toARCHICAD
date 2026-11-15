"""Aggregate exactly eight CAL-09.4 fresh-seed scanner shard files.

The final scorer freezes the spatial rule using 481xx only. It then reads
the previously unconsulted 491xx holdout once; the old 461xx/471xx seeds
never enter selection. No raw point clouds are written to results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cal094_spatial_selection import (
    FORMAT, MESHES, SEED_DEV, SEED_HOLDOUT, SEEDS, VARIANTS,
    evaluate_protocol,
)
from tools.validate_cal094_spatial import RUN_FORMAT


def aggregate(shards):
    if len(shards) != len(SEEDS):
        raise ValueError("Exactly eight distinct CAL-09.4 seed shards are required")
    fingerprints, source_shas = set(), set()
    seen = []
    cases = []
    for path in shards:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if (doc.get("format") != RUN_FORMAT or
                doc.get("auto_update_production_config") is not False):
            raise ValueError("Unrecognized CAL-09.4 input shard")
        if doc.get("variants") != list(VARIANTS) or doc.get("resolutions_m") != list(MESHES):
            raise ValueError("Shard has a different factorial design")
        if doc.get("angular_step_deg") != .1 or doc.get("max_sample") != 180000:
            raise ValueError("Noncanonical scanner settings")
        if len(doc["seeds"]) != 1 or doc["n_cases"] != len(VARIANTS) * len(MESHES):
            raise ValueError("One-seed shard must contain six complete scans")
        if len(doc["cases"]) != doc["n_cases"]:
            raise ValueError("Shard row count does not match")
        seen.extend(doc["seeds"])
        if set(x["seed"] for x in doc["cases"]) != set(doc["seeds"]):
            raise ValueError("Cross-seed row contamination")
        fingerprints.add(doc["run_fingerprint"])
        source_shas.add(doc["source_config_sha256"])
        cases.extend(doc["cases"])
    if (set(seen) != set(SEEDS) or len(set(seen)) != len(SEEDS) or
            len(fingerprints) != 1 or len(source_shas) != 1):
        raise ValueError("Missing/repeated seed, inconsistent source or scanner config")
    result = evaluate_protocol(cases)
    result.update({
        "n_complete_scans": len(cases),
        "source_config_sha256": next(iter(source_shas)),
        "run_fingerprint": next(iter(fingerprints)),
    })
    return result


def markdown(doc):
    lines = [
        "# CAL-09.4: пространственная согласованность детекции дефектов",
        "",
        "**Данные:** 48 полных синтетических сканов. Старые CAL-09 / CAL-09.3 "
        "seed исключены из нового выбора правил.",
        f"Development: {doc['design']['development_seeds']}. "
        f"Независимый holdout: {doc['design']['independent_holdout_seeds']}.",
        "Все ячейки сопоставлены через GT-oracle физическую стену, "
        "детектор использует только измеренные score и соседство.",
        "",
        "## Зафиксированный development-поиск",
        "",
        "| Метод | Recall | FPR (средний) | FPR (макс.) | Cluster recall | Допущен |",
        "|---|---:|---:|---:|---:|---|",
    ]
    fmt = lambda x: "Н/Д" if x is None else f"{x:.3f}"
    for row in doc["development_selection"]:
        lines.append(
            f"| {row['name']} | {fmt(row['mean_recall'])} | "
            f"{fmt(row['mean_fpr'])} | {fmt(row['worst_fpr'])} | "
            f"{fmt(row['mean_cluster_recall'])} | "
            f"{'да' if row['eligible'] else 'нет'} |")
    lines.extend([
        "",
        "**Правило, выбранное только по development:** " +
        str(doc["selected_on_development_only"]),
        "",
        "## Независимый holdout 491xx",
        "",
        "| Метод | Mesh, м | Recall | FPR | Cluster recall | TP | FP | GT clusters |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for rule, entry in doc["independent_holdout"].items():
        for mesh, m in entry["by_mesh"].items():
            lines.append(
                f"| {rule} | {mesh} | {fmt(m['mean_recall'])} | "
                f"{fmt(m['mean_fpr'])} | {fmt(m['mean_cluster_recall'])} | "
                f"{m['pooled']['tp']} | {m['pooled']['fp']} | "
                f"{m['pooled']['gt_clusters_ge_2']} |")
    lines.extend([
        "", "## Решение",
        "**" + doc["decision"] + "**",
        "",
        "Критерии: recall >= 0.40, FPR <= 0.10, кластерный recall >= 0.35, "
        "прирост recall к MAD6 >= 0.15. Выполняются все одновременно.",
        "Даже PASS относится только к **oracle-патчам**: это не принятый CAL-09 "
        "и не проверка на реальных E57.",
        "Production YAML, исходные E57/GT и веса моделей не менялись.",
        "",
        "## Ограничения",
    ])
    lines.extend("- " + x for x in doc["limitations"])
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("shards", type=Path, nargs="+")
    args = p.parse_args()
    result = aggregate(args.shards)
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "cal094_holdout_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.out / "CAL094_REPORT.md").write_text(markdown(result), encoding="utf-8")
    print(markdown(result), flush=True)


if __name__ == "__main__":
    main()
