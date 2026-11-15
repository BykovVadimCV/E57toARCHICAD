"""CAL-09.5: exact 8-shard aggregation with dev-only group-CV soft scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cal095_soft_confidence import (
    FORMAT, SEEDS, VARIANTS, DEV, HOLDOUT, MESH_M, evaluate_protocol,
)
from tools.validate_cal095_confidence import FORMAT as SHARD_FORMAT


def aggregate(paths):
    if len(paths) != len(SEEDS):
        raise ValueError("Expected precisely eight fresh seed shards")
    cases, fingerprints, configs, seen = [], set(), set(), []
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("format") != SHARD_FORMAT or doc.get("auto_update_production_config") is not False:
            raise ValueError("Not a CAL-09.5 shard")
        if (doc.get("variants") != list(VARIANTS) or
                doc.get("mesh_m") != MESH_M or
                doc.get("angular_step_deg") != .1 or
                doc.get("max_sample") != 180000 or
                doc.get("n_scans") != len(VARIANTS) or len(doc.get("seeds", [])) != 1):
            raise ValueError("Noncanonical scanner shard")
        if len(doc["cases"]) != doc["n_scans"] or set(
                c["seed"] for c in doc["cases"]) != set(doc["seeds"]):
            raise ValueError("Case count/seed mismatch")
        seen.extend(doc["seeds"])
        fingerprints.add(doc["run_fingerprint"])
        configs.add(doc["source_config_sha256"])
        cases.extend(doc["cases"])
    if (set(seen) != set(SEEDS) or len(set(seen)) != len(seen) or
            len(fingerprints) != 1 or len(configs) != 1):
        raise ValueError("Missing or duplicate seed / mixed code and config")
    result = evaluate_protocol(cases)
    result["n_complete_scanner_scenes"] = len(cases)
    result["source_config_sha256"] = next(iter(configs))
    result["run_fingerprint"] = next(iter(fingerprints))
    return result


def markdown(doc):
    fmt = lambda n: "н/д" if n is None else f"{n:.3f}"
    lines = [
        "# CAL-09.5: мягкая пространственная уверенность",
        "",
        "24 новых полноценных сканирования при шаге 0,1° и сетке 0,10 м. "
        "Скоринг использует только наблюдаемые признаки, но физические "
        "патчи сопоставлены по synthetic GT-oracle.",
        f"Development seed: {doc['design']['development_seeds']}.",
        f"Неиспользованные при подборе holdout seed: {doc['design']['untouched_holdout_seeds']}.",
        "",
        "## Выбор по leave-one-seed-out development",
        "",
        "| Признаки | L2 | Квантиль отрицательных | CV recall | CV FPR | Worst FPR | Cluster recall | Допущено |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for x in doc["development_crossfit_selection"]:
        lines.append(
            f"| {x['feature_set']} | {x['l2']:.3f} | "
            f"{x['negative_quantile']:.3f} | {fmt(x['mean_recall'])} | "
            f"{fmt(x['mean_fpr'])} | {fmt(x['worst_seed_fpr'])} | "
            f"{fmt(x['mean_cluster_recall'])} | "
            f"{'да' if x['eligible'] else 'нет'} |")
    selected = doc["selected_on_development_only"]
    lines += ["", "**Выбрано исключительно по development:** " +
              (json.dumps(selected, ensure_ascii=False) if selected else "нет кандидата"),
              "", "## Новый независимый holdout", "",
              "| Правило | Mean recall | Mean FPR | Mean cluster recall | TP | FP |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, item in doc["holdout_predeclared_comparators"].items():
        lines.append(
            f"| {name} | {fmt(item['mean_recall'])} | "
            f"{fmt(item['mean_fpr'])} | {fmt(item['mean_cluster_recall'])} | "
            f"{item['pooled']['tp']} | {item['pooled']['fp']} |")
    if "cal095_selected" in doc["holdout_predeclared_comparators"]:
        model = doc["holdout_predeclared_comparators"]["cal095_selected"]
        lines += ["", f"Диагностика вероятностей: Brier={fmt(model.get('brier_score'))}; "
                  f"ECE-5={fmt(model.get('ece_5bins'))}."]
    lines += ["", "## Решение",
              "**" + doc["decision"] + "**",
              "",
              "PASS возможен только при recall >=40%, FPR <=10%, "
              "кластерном recall >=35% и приросте recall >=15 п.п. "
              "против MAD6 на новых holdout seed.",
              "Даже PASS здесь не заменяет blind-проверку CAL-09 или реальные E57.",
              "",
              "## Ограничения"]
    lines += ["- " + msg for msg in doc["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    args = p.parse_args()
    summary = aggregate(args.shards)
    args.out.mkdir(exist_ok=True, parents=True)
    (args.out / "cal095_holdout_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8")
    (args.out / "CAL095_REPORT.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary), flush=True)


if __name__ == "__main__":
    main()
