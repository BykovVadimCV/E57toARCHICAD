#!/usr/bin/env python3
"""Тест модулей векторизации и экспорта IFC.

Проверяет работу векторизации и экспорта без запуска полной сети.
Создаёт минимальные тестовые данные вручную.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image


def create_test_prediction(out_dir: Path) -> None:
    """Создать минимальные тестовые данные предсказания."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Семантическая маска: простая комната со стенами
    H, W = 256, 256
    sem = np.zeros((H, W), dtype=np.uint8)

    # Стены (класс 2) - рамка
    sem[10:20, 10:W-10] = 2  # верхняя стена
    sem[H-20:H-10, 10:W-10] = 2  # нижняя стена
    sem[10:H-10, 10:20] = 2  # левая стена
    sem[10:H-10, W-20:W-10] = 2  # правая стена

    # Помещение (класс 1) - внутри
    sem[25:H-25, 25:W-25] = 1

    # Дверь (класс 3) - проём в нижней стене
    sem[H-20:H-10, 100:120] = 3

    # Окно (класс 4) - проём в верхней стене
    sem[10:20, 180:210] = 4

    Image.fromarray(sem).save(out_dir / "sem.png")

    # Расстояние до стен (1 канал)
    dist = np.ones((1, H, W), dtype=np.float16) * 0.1  # 10 см до грани
    np.save(out_dir / "dist.npy", dist)

    # Ориентация (2 канала: cos 2θ, sin 2θ)
    orient = np.zeros((2, H, W), dtype=np.float16)
    orient[0] = 1.0  # cos 2θ = 1 → θ = 0° (горизонтальные стены)
    np.save(out_dir / "orient.npy", orient)

    # Высоты проёмов (2 канала: sill, head)
    heights = np.zeros((2, H, W), dtype=np.float16)
    heights[0] = 0.3  # подоконник 30% высоты (0.9м при 3м потолке)
    heights[1] = 0.7  # перемычка 70% высоты (2.1м)
    np.save(out_dir / "heights.npy", heights)

    # Кадр и метаданные
    scene = {
        "source": "test",
        "level": "L1",
        "floor_z": 0.0,
        "ceiling_z": 3.0,
        "frame": {
            "origin_x": 0.0,
            "origin_y": 0.0,
            "pixel_m": 0.02,  # 2 см на пиксель
            "width": W,
            "height": H
        },
        "pixel_to_world": [[0.02, 0.0, 0.01], [0.0, -0.02, 5.11]]
    }

    (out_dir / "scene.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")

    # Превью (опционально)
    preview = np.zeros((H, W, 3), dtype=np.uint8)
    preview[sem == 1] = [232, 244, 248]  # интерьер - голубой
    preview[sem == 2] = [42, 42, 42]  # стены - тёмный
    preview[sem == 3] = [214, 96, 77]  # двери - красный
    preview[sem == 4] = [67, 147, 195]  # окна - синий
    Image.fromarray(preview).save(out_dir / "pred.png")

    print(f"Тестовые данные созданы: {out_dir}")
    print(f"  - sem.png: {H}×{W}, классы 0-4")
    print(f"  - dist.npy, orient.npy, heights.npy")
    print(f"  - scene.json (пиксель 2 см, кадр 5.12×5.12 м)")


def test_vectorization(test_dir: Path) -> None:
    """Тест векторизации."""
    print("\n=== Тест векторизации ===")

    from simscan.vectorize import vectorize

    pred_dir = test_dir / "pred"
    vectors_dir = test_dir / "vectors"

    if not pred_dir.exists():
        print("Создание тестовых данных...")
        create_test_prediction(pred_dir)

    print(f"Векторизация {pred_dir}...")
    plan = vectorize(pred_dir)

    print(f"Результат:")
    print(f"  Стены: {len(plan.walls)}")
    for w in plan.walls[:3]:
        print(f"    {w.id}: {len(w.points)} точек, {w.thickness*100:.1f} см, {w.length:.2f} м")

    print(f"  Помещения: {len(plan.rooms)}")
    for r in plan.rooms:
        print(f"    {r.id}: {r.area:.2f} м², {r.perimeter:.2f} м")

    print(f"  Проёмы: {len(plan.openings)}")
    for o in plan.openings:
        print(f"    {o.id}: {o.opening_type}, {o.width:.2f} м")

    # Сохранение
    from simscan.vectorize import save_json, save_svg, save_png

    vectors_dir.mkdir(parents=True, exist_ok=True)
    save_json(plan, vectors_dir / "plan.json")
    save_svg(plan, vectors_dir / "plan.svg")
    save_png(plan, vectors_dir / "plan.png")

    print(f"\nСохранено:")
    print(f"  {vectors_dir / 'plan.json'}")
    print(f"  {vectors_dir / 'plan.svg'}")
    print(f"  {vectors_dir / 'plan.png'}")

    return plan


def test_ifc_export(test_dir: Path) -> None:
    """Тест экспорта IFC."""
    print("\n=== Тест экспорта IFC ===")

    from simscan.export_ifc import export_to_ifc

    plan_json = test_dir / "vectors" / "plan.json"
    output_ifc = test_dir / "test.ifc"

    if not plan_json.exists():
        print("Сначала запустите векторизацию")
        return

    print(f"Экспорт {plan_json} → {output_ifc}...")
    export_to_ifc(plan_json, output_ifc)

    # Проверка результата
    if output_ifc.exists():
        size_kb = output_ifc.stat().st_size / 1024
        lines = len(output_ifc.read_text(encoding="utf-8").splitlines())
        print(f"\nIFC файл создан:")
        print(f"  Размер: {size_kb:.1f} КБ")
        print(f"  Строк: {lines}")
        print(f"  Путь: {output_ifc}")

        # Базовая валидация
        content = output_ifc.read_text(encoding="utf-8")
        if "ISO-10303-21" in content and "ENDSEC" in content:
            print("  ✓ Формат IFC корректен")

        entity_types = ["IFCPROJECT", "IFCBUILDING", "IFCBUILDINGSTOREY",
                       "IFCWALL", "IFCSPACE"]
        for etype in entity_types:
            count = content.count(etype)
            if count > 0:
                print(f"  ✓ {etype}: {count}")
    else:
        print("ОШИБКА: IFC файл не создан")


def main():
    """Запуск всех тестов."""
    test_dir = Path("test_archicad_export")

    print("Тестирование модулей векторизации и экспорта IFC")
    print("=" * 60)

    try:
        # Тест 1: Векторизация
        plan = test_vectorization(test_dir)

        # Тест 2: Экспорт IFC
        test_ifc_export(test_dir)

        print("\n" + "=" * 60)
        print("✓ Все тесты пройдены успешно")
        print(f"\nРезультаты в каталоге: {test_dir.absolute()}")
        print("\nДля просмотра IFC:")
        print("  - ArchiCAD: File → Import → Import IFC...")
        print("  - BIMVision (бесплатно): https://bimvision.eu/download/")
        print("  - Онлайн: https://view.ifcengine.com/")

    except Exception as e:
        print(f"\n✗ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
