"""Aggregate CAL-09.3 8 independent seed shards and freeze holdout evaluation.

Only after receiving EXACTLY 8 matching-config shards do we select a score
using development seeds, then evaluate the predeclared new-seed holdout once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cal093_threshold_selection import (
    DEV_SEEDS, HOLDOUT_SEEDS, RESOLUTIONS_M, VARIANTS,
    analyze_holdout,
)
from tools.validate_cal093_multiscale import RUN_FORMAT


def aggregate(shards):
    if len(shards) != len(DEV_SEEDS + HOLDOUT_SEEDS):
        raise ValueError("Exactly eight per-seed shard files required")
    cases = []
    fingerprints = set()
    config_shas = set()
    seeds_seen = []
    for path in shards:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("format") != RUN_FORMAT or d.get("auto_update_production_config") is not False:
            raise ValueError("Unrecognized or unsafe experiment shard")
        if d.get("variants") != list(VARIANTS) or d.get("resolutions_m") != list(RESOLUTIONS_M):
            raise ValueError("Shard design differs from predeclared factorial")
        if d.get("angular_step_deg") != .1 or d.get("max_sample") != 180000:
            raise ValueError("Only frozen 0.1 degree, 180000-return canonical shards allowed")
        if d.get("n_cases") != len(VARIANTS) * len(RESOLUTIONS_M) or len(d["seeds"]) != 1:
            raise ValueError("Shard must contain exactly one seed and six cases")
        seeds_seen.extend(d["seeds"])
        fingerprints.add(d["run_fingerprint"])
        config_shas.add(d["source_config_sha256"])
        cases.extend(d["cases"])
    if (set(seeds_seen) != set(DEV_SEEDS + HOLDOUT_SEEDS) or
            len(set(seeds_seen)) != 8 or len(fingerprints) != 1 or
            len(config_shas) != 1):
        raise ValueError("Duplicated/missing seed or source-config mismatch")
    result = analyze_holdout(cases)
    result["source_config_sha256"] = next(iter(config_shas))
    result["run_fingerprint"] = next(iter(fingerprints))
    result["n_paired_scanner_simulations"] = len(cases)
    return result


def report(doc):
    lines = [
        "# CAL-09.3: наблюдаемость пространственных дефектов",
        "",
        "**Верификация:** независимый holdout, одни и те же прямые отражения для "
        "четырёх контрфактических режимов в каждом скане.",
        "Первоначальный CAL-09 не прошёл обнаружение на слепых top-24 патчах.",
        "",
        "## Дизайн",
        f"- Прогонов: {doc['n_paired_scanner_simulations']}; "
        f"development seed: {doc['design']['development_seeds']}; "
        f"holdout seed: {doc['design']['independent_holdout_seeds']}.",
        "- Порог и метод выбираются только по development (без чтения holdout).",
        "- Контроль uniform включён в расчёт FPR. Группировка по физическим "
        "стенам использует GT; это не blind detection.",
        "",
        "## Диагностический поиск на development",
        "",
        "| Правило | Recall (mean) | FPR (mean) | FPR (max) | Кластерный recall | Допустимо |",
        "|---|---:|---:|---:|---:|---|",
    ]
    fmt = lambda a: "н/д" if a is None else f"{a:.3f}"
    for x in doc["development_threshold_sweep"]:
        lines.append(
            f"| {x['name']} | {fmt(x['mean_recall'])} | {fmt(x['mean_fpr'])} | "
            f"{fmt(x['worst_seed_resolution_fpr'])} | "
            f"{fmt(x['mean_cluster_recall'])} | "
            f"{'да' if x['permitted'] else 'нет'} |")
    lines.extend([
        "",
        "**Выбрано только по development:** " + str(doc["selected_using_development_only"]),
        "",
        "## Независимый holdout",
        "",
        "| Метод | Mesh | Mean recall | Mean FPR | Mean cluster recall | GT-positive patches |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, groups in doc["holdout_selected_vs_fixed_mad6"].items():
        for step, d in groups.items():
            lines.append(
                f"| {name} | {step} | {fmt(d['mean_recall'])} | "
                f"{fmt(d['mean_fpr'])} | {fmt(d['mean_cluster_recall'])} | "
                f"{d['gt_positive_patches']} |")
    lines.extend([
        "", "## Решение",
        "**" + doc["decision"] + "**",
        "",
        "Это диагностический результат на GT-oracle физических патчах, "
        "**НЕ** принятие CAL-09, НЕ физическая калибровка BLK360.",
        "Production YAML, E57, GT и веса моделей не обновлены.",
        "", "## Ограничения", "",
    ])
    lines.extend("- " + s for s in doc["limitations"])
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("shards", nargs="+", type=Path)
    args = ap.parse_args()
    data = aggregate(args.shards)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cal093_holdout_results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (args.out / "CAL093_REPORT.md").write_text(report(data), encoding="utf-8")
    print(report(data), flush=True)


if __name__ == "__main__":
    main()
