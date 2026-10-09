#!/usr/bin/env python3
"""
Пример использования модулей векторизации и экспорта в ArchiCAD.

Демонстрирует полный пайплайн:
  1. Векторизация предсказаний нейросети
  2. Сохранение промежуточных результатов (JSON, SVG, PNG)
  3. Экспорт в IFC для ArchiCAD
"""

from pathlib import Path
import json
from simscan.vectorize import vectorize
from simscan.export_ifc import export_to_ifc


def main():
    # Пример 1: Полный пайплайн с промежуточными файлами
    print("=== Пример 1: Полный пайплайн ===\n")

    pred_dir = Path("pred")  # Каталог с предсказаниями сети
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Шаг 1: Векторизация
    print(f"Векторизация {pred_dir}...")
    plan = vectorize(
        pred_dir,
        simplify_tol=0.03,  # Упрощение ломаных: порог 3 см
        min_wall_len=0.15,  # Минимальная длина стены: 15 см
    )

    # Статистика
    print(f"  Стен: {len(plan['walls'])}")
    print(f"  Помещений: {len(plan['rooms'])}")
    print(f"  Проёмов: {len(plan['openings'])}")

    # Шаг 2: Сохранение JSON
    json_path = output_dir / "plan.json"
    print(f"\nСохранение {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    # Шаг 3: Экспорт в IFC
    ifc_path = output_dir / "apartment.ifc"
    print(f"\nЭкспорт в {ifc_path}...")
    stats = export_to_ifc(
        json_path,
        ifc_path,
        wall_height=2.8,      # Высота стен: 2.8 м
        slab_thickness=0.2,   # Толщина плиты: 20 см
    )

    print(f"\nГотово!")
    print(f"  Сущностей в IFC: {stats['total_entities']}")
    print(f"  Размер файла: {ifc_path.stat().st_size / 1024:.1f} КБ")

    # Пример 2: Прямая работа с dict (без промежуточных файлов)
    print("\n\n=== Пример 2: Прямой экспорт ===\n")

    plan2 = vectorize("pred", simplify_tol=0.05)  # Более грубое упрощение

    stats2 = export_to_ifc(
        plan2,  # Передаём dict напрямую, без JSON
        "output/apartment_simplified.ifc",
        wall_height=3.0,
    )

    print(f"Упрощённый план: {stats2['total_entities']} сущностей")

    # Пример 3: Анализ результата векторизации
    print("\n\n=== Пример 3: Анализ плана ===\n")

    # Общая длина стен
    total_wall_length = sum(w["length"] for w in plan["walls"])
    print(f"Общая длина стен: {total_wall_length:.2f} м")

    # Общая площадь помещений
    total_area = sum(r["area"] for r in plan["rooms"])
    print(f"Общая площадь: {total_area:.2f} м²")

    # Средняя толщина стен
    avg_thickness = sum(w["thickness"] for w in plan["walls"]) / len(plan["walls"])
    print(f"Средняя толщина стен: {avg_thickness * 100:.1f} см")

    # Распределение проёмов по типам
    opening_types = {}
    for op in plan["openings"]:
        typ = op["type"]
        opening_types[typ] = opening_types.get(typ, 0) + 1

    print("\nПроёмы:")
    for typ, count in opening_types.items():
        print(f"  {typ}: {count}")

    # Пример 4: Работа с метаданными
    print("\n\n=== Пример 4: Метаданные ===\n")

    meta = plan["metadata"]
    print(f"Источник: {meta['source']}")
    print(f"Разрешение растра: {meta['pixel_m'] * 100:.1f} см/пиксель")

    frame = meta["frame"]
    print(f"Область плана:")
    print(f"  X: [{frame['x_min']:.2f}, {frame['x_max']:.2f}] м")
    print(f"  Y: [{frame['y_min']:.2f}, {frame['y_max']:.2f}] м")
    print(f"  Размер: {frame['x_max'] - frame['x_min']:.2f} × "
          f"{frame['y_max'] - frame['y_min']:.2f} м")


if __name__ == "__main__":
    main()
