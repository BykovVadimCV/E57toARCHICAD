#!/usr/bin/env python3
"""Интерактивное меню для конвертации E57 → IFC с использованием обученной модели."""
from __future__ import annotations

import os
import sys
from pathlib import Path


class Colors:
    """ANSI цвета для консоли."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def clear_screen():
    """Очистка экрана."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header(text: str):
    """Печать заголовка."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(70)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.ENDC}\n")


def print_success(text: str):
    """Печать успеха."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_error(text: str):
    """Печать ошибки."""
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    """Печать информации."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_warning(text: str):
    """Печать предупреждения."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def get_input(prompt: str, default: str = "") -> str:
    """Получить ввод с дефолтным значением."""
    if default:
        full_prompt = f"{Colors.OKCYAN}{prompt} [{default}]{Colors.ENDC}: "
    else:
        full_prompt = f"{Colors.OKCYAN}{prompt}{Colors.ENDC}: "
    
    value = input(full_prompt).strip()
    return value if value else default


def get_model_path() -> Path | None:
    """Определить путь к файлу модели."""
    default_locations = [
        Path("models/best.pt"),
        Path("runs/best.pt"),
        Path("best.pt"),
        Path.home() / ".simscan" / "models" / "best.pt",
    ]
    
    for loc in default_locations:
        if loc.exists():
            return loc
    
    return None


def ensure_model_exists() -> Path:
    """Проверить наличие модели или запросить путь."""
    model_path = get_model_path()
    
    if model_path:
        print_success(f"Найдена модель: {model_path}")
        use_it = get_input("Использовать эту модель? (y/n)", "y").lower()
        if use_it == 'y':
            return model_path
    
    while True:
        path_str = get_input("Введите путь к файлу модели (best.pt)")
        if not path_str:
            print_error("Путь не может быть пустым")
            continue
        
        path = Path(path_str)
        if not path.exists():
            print_error(f"Файл не найден: {path}")
            retry = get_input("Попробовать снова? (y/n)", "y").lower()
            if retry != 'y':
                sys.exit(1)
            continue
        
        return path


def get_e57_file() -> Path:
    """Получить путь к E57 файлу."""
    while True:
        path_str = get_input("Введите путь к E57 файлу (или перетащите файл)")
        
        # Убрать кавычки если файл был перетащен
        path_str = path_str.strip().strip("'").strip('"')
        
        if not path_str:
            print_error("Путь не может быть пустым")
            continue
        
        path = Path(path_str)
        if not path.exists():
            print_error(f"Файл не найден: {path}")
            retry = get_input("Попробовать снова? (y/n)", "y").lower()
            if retry != 'y':
                sys.exit(1)
            continue
        
        if path.suffix.lower() != '.e57':
            print_warning(f"Файл не имеет расширения .e57: {path.suffix}")
            proceed = get_input("Продолжить? (y/n)", "n").lower()
            if proceed != 'y':
                continue
        
        return path


BUILDERS = {"1": "vectorize", "2": "cellplan", "3": "both"}
TTAS = {"1": "none", "2": "flip", "3": "d4"}


def ask_options() -> dict:
    """Построитель плана и TTA."""
    print(f"\n{Colors.BOLD}Построитель плана:{Colors.ENDC}")
    print("  1. vectorize - по выходу сети, грани стен - по облаку E57 (точность - мм, а не пиксель 2 см)")
    print("  2. cellplan  - облако + вероятности сети (медленнее, читает E57 ещё раз)")
    print("  3. оба       - два IFC рядом, для сравнения")
    builder = BUILDERS.get(get_input("Выбор (1-3)", "3"), "both")
    print(f"\n{Colors.BOLD}Test-Time Augmentation (точнее, но дольше шаг сети):{Colors.ENDC}")
    print("  1. none - без TTA")
    print("  2. flip - 4 отражения (x4 по времени)")
    print("  3. d4   - отражения и повороты на 90° (x8 по времени)")
    tta = TTAS.get(get_input("Выбор (1-3)", "2"), "flip")
    return {"builder": builder, "tta": tta}


def run_inference(model_path: Path, e57_path: Path, output_dir: Path, opts: dict) -> list[Path]:
    """Вывод модели; при builder vectorize/both - сразу векторный план на каждый уровень.
    Возвращает каталоги уровней ([] - ошибка)."""
    from simscan.infer import infer

    print_header("ШАГ 1/3: Вывод модели (Inference)")
    print_info(f"Модель: {model_path}")
    print_info(f"E57 файл: {e57_path}")
    print_info(f"Выходная директория: {output_dir}")
    print_info(f"TTA: {opts['tta']}; векторный план: {'да' if opts['builder'] != 'cellplan' else 'нет'}")

    try:
        results = infer(
            str(model_path),
            [str(e57_path)],
            str(output_dir),
            tile=512,
            tta=False if opts["tta"] == "none" else opts["tta"],
            plan=opts["builder"] in ("vectorize", "both"),
            log=lambda x: print(f"  {Colors.OKBLUE}→{Colors.ENDC} {x}")
        )

        if results:
            print_success(f"Inference завершен. Создано {len(results)} уровней:")
            for r in results:
                print(f"  • {r}")
            return [Path(r) for r in results]
        print_error("Не удалось выполнить inference")
        return []

    except Exception as e:
        print_error(f"Ошибка при inference: {e}")
        import traceback
        traceback.print_exc()
        return []


def vector_plans(levels: list[Path]) -> list[Path]:
    """plan.json векторизатора по уровням (их пишет infer с plan=True)."""
    plans = [d / "plan.json" for d in levels if (d / "plan.json").exists()]
    if plans:
        print_success(f"Векторный план готов: {len(plans)} уровней")
        for d in levels:
            if (d / "plan.png").exists():
                print(f"  • {d / 'plan.png'}")
    else:
        print_error("Векторный план не построен")
    return plans


def build_and_export(e57_path: Path, levels: list[Path], pred_dir: Path, out_dir: Path, opts: dict) -> list[Path]:
    """Шаги 2-3: план выбранным построителем -> IFC. Возвращает записанные IFC."""
    stem = e57_path.stem
    both = opts["builder"] == "both"
    done = []
    if opts["builder"] in ("vectorize", "both"):
        print_header("ШАГ 2/3: План (vectorize - по выходу сети)")
        plans = vector_plans(levels)
        ifc = out_dir / (f"{stem}_vectorize.ifc" if both else f"{stem}.ifc")
        if plans and run_ifc_export(plans, ifc):
            done.append(ifc)
    if opts["builder"] in ("cellplan", "both"):
        plans = run_cellplan(e57_path, pred_dir, out_dir / "plan")
        ifc = out_dir / (f"{stem}_cellplan.ifc" if both else f"{stem}.ifc")
        if plans and run_ifc_export(plans, ifc):
            done.append(ifc)
    return done


def run_cellplan(e57_path: Path, pred_dir: Path, plan_dir: Path) -> list[Path]:
    """План: разбиение на ячейки и разметка по облаку и вероятностям сети."""
    from simscan.cellplan import cellplan

    print_header("ШАГ 2/3: План (cellplan - облако + вероятности сети)")
    print_info(f"Предсказания: {pred_dir}")
    print_info(f"Выходная директория: {plan_dir}")

    try:
        cellplan(e57_path, plan_dir, pred=pred_dir,
                 log=lambda x: print(f"  {Colors.OKBLUE}→{Colors.ENDC} {x}"))
        plans = sorted(plan_dir.glob("plan_L*.json"))
        print_success(f"План готов: {len(plans)} уровней")
        return plans

    except Exception as e:
        print_error(f"Ошибка при построении плана: {e}")
        import traceback
        traceback.print_exc()
        return []


def run_ifc_export(plan_json, ifc_path: Path) -> bool:
    """Запустить экспорт в IFC."""
    from simscan.export_ifc import main_export_ifc
    
    print_header("ШАГ 3/3: Экспорт в IFC")
    print_info(f"План: {', '.join(map(str, plan_json)) if isinstance(plan_json, list) else plan_json}")
    print_info(f"Выходной IFC: {ifc_path}")
    
    try:
        main_export_ifc(
            plan_json,
            str(ifc_path),
            log=lambda x: print(f"  {Colors.OKBLUE}→{Colors.ENDC} {x}")
        )
        print_success(f"IFC экспорт завершен: {ifc_path}")
        return True
    
    except Exception as e:
        print_error(f"Ошибка при экспорте IFC: {e}")
        import traceback
        traceback.print_exc()
        return False


def full_pipeline():
    """Полный пайплайн E57 → IFC."""
    clear_screen()
    print_header("E57 → IFC КОНВЕРТЕР")
    print_info("Полный пайплайн: Inference → план (vectorize / cellplan) → IFC\n")
    
    # 1. Проверка модели
    model_path = ensure_model_exists()
    
    # 2. Получить E57 файл
    e57_path = get_e57_file()
    
    # 3. Настроить выходные пути
    base_name = e57_path.stem
    default_output = e57_path.parent / f"{base_name}_output"
    
    output_str = get_input(
        f"Выходная директория", 
        str(default_output)
    )
    output_dir = Path(output_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = ask_options()

    pred_dir = output_dir / "predictions"
    print(f"\n{Colors.BOLD}Будут созданы:{Colors.ENDC}")
    print(f"  Предсказания: {pred_dir}")
    if opts["builder"] in ("vectorize", "both"):
        print(f"  Векторный план: {pred_dir / base_name}/L*/plan.json, plan.png, plan.svg")
    if opts["builder"] in ("cellplan", "both"):
        print(f"  План (cellplan): {output_dir / 'plan'}")
    print(f"  IFC: {output_dir}/{base_name}*.ifc\n")
    
    proceed = get_input("Начать обработку? (y/n)", "y").lower()
    if proceed != 'y':
        print_warning("Отменено пользователем")
        return
    
    # Шаг 1: Inference (+ векторный план)
    levels = run_inference(model_path, e57_path, pred_dir, opts)
    if not levels:
        print_error("Не удалось выполнить inference. Прерывание.")
        return

    # Шаги 2-3: план -> один IFC, этаж на уровень
    ifcs = build_and_export(e57_path, levels, pred_dir / base_name, output_dir, opts)
    if not ifcs:
        print_error("IFC не записан.")
        return

    print_header("ГОТОВО!")
    print_success(f"Результаты сохранены в: {output_dir}")
    print(f"\n{Colors.BOLD}IFC файлы:{Colors.ENDC}")
    for ifc_file in ifcs:
        print(f"  • {ifc_file}")


def inspect_e57():
    """Инспекция E57 файла."""
    clear_screen()
    print_header("ИНСПЕКЦИЯ E57")
    
    e57_path = get_e57_file()
    
    from simscan.inspect_e57 import inspect_e57
    
    output_dir = e57_path.parent / f"{e57_path.stem}_report"
    
    print_info("Анализ E57 файла...")
    try:
        result = inspect_e57(e57_path, output_dir)
        print_success("Отчет создан!")
        
        summary = output_dir / "summary.txt"
        if summary.exists():
            print(f"\n{Colors.BOLD}КРАТКИЙ ОТЧЕТ:{Colors.ENDC}")
            print(summary.read_text(encoding='utf-8'))
        
        zip_path = result.get("zip")
        if zip_path:
            print_info(f"Архив отчета: {zip_path}")
    
    except Exception as e:
        print_error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


def analytics_e57():
    """Весь конвейер на одном E57 с диагностикой каждого шага -> один zip для разбора."""
    from simscan.analytics import analytics

    clear_screen()
    print_header("АНАЛИТИКА E57 (архив для разбора)")
    e57_path = get_e57_file()
    model_path = None
    if get_input("Прогнать сеть? (y/n)", "y").lower() == "y":
        model_path = ensure_model_exists()
    tta = get_input("TTA для сети (none / flip / d4)", "none").lower()
    if tta not in ("none", "flip", "d4"):
        print_warning(f"Неизвестный TTA «{tta}» - без TTA")
        tta = "none"
    cell = get_input("Строить и cellplan (дольше)? (y/n)", "y").lower() == "y"
    pano = get_input("Добавить панорамы стоянок (это фото объекта)? (y/n)", "n").lower() == "y"
    try:
        zip_path = analytics(e57_path, e57_path.parent, model=model_path, tta=tta, cellplan=cell, panoramas=pano,
                             log=lambda x: print(f"  {Colors.OKBLUE}→{Colors.ENDC} {x}"))
        summary = zip_path.with_suffix("") / "SUMMARY.md"
        if summary.exists():
            print(f"\n{Colors.BOLD}СВОДКА:{Colors.ENDC}")
            print(summary.read_text(encoding="utf-8"))
        print_success(f"Архив для разбора: {zip_path}")
    except Exception as e:
        print_error(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()


def batch_convert():
    """Пакетная конвертация нескольких E57 файлов."""
    clear_screen()
    print_header("ПАКЕТНАЯ КОНВЕРТАЦИЯ")
    
    # Получить директорию с E57 файлами
    dir_str = get_input("Введите путь к директории с E57 файлами")
    input_dir = Path(dir_str)
    
    if not input_dir.exists() or not input_dir.is_dir():
        print_error(f"Директория не найдена: {input_dir}")
        return
    
    # Найти все E57 файлы
    e57_files = list(input_dir.glob("*.e57")) + list(input_dir.glob("*.E57"))
    
    if not e57_files:
        print_error("E57 файлы не найдены")
        return
    
    print_info(f"Найдено {len(e57_files)} E57 файлов:")
    for f in e57_files:
        print(f"  • {f.name}")
    
    proceed = get_input("\nОбработать все файлы? (y/n)", "y").lower()
    if proceed != 'y':
        return
    
    model_path = ensure_model_exists()
    
    output_dir = Path(get_input("Выходная директория", str(input_dir / "output")))
    output_dir.mkdir(parents=True, exist_ok=True)
    opts = ask_options()
    
    # Обработать каждый файл
    for idx, e57_path in enumerate(e57_files, 1):
        print(f"\n{Colors.BOLD}[{idx}/{len(e57_files)}] {e57_path.name}{Colors.ENDC}")
        
        file_output = output_dir / e57_path.stem
        file_output.mkdir(parents=True, exist_ok=True)
        
        pred_dir = file_output / "predictions"
        
        # Inference (+ векторный план)
        levels = run_inference(model_path, e57_path, pred_dir, opts)
        if not levels:
            print_warning(f"Пропуск {e57_path.name}")
            continue

        build_and_export(e57_path, levels, pred_dir / e57_path.stem, file_output, opts)
    
    print_header("ПАКЕТНАЯ ОБРАБОТКА ЗАВЕРШЕНА")
    print_success(f"Результаты в: {output_dir}")


def configure_model():
    """Настройка пути к модели."""
    clear_screen()
    print_header("НАСТРОЙКА МОДЕЛИ")
    
    current = get_model_path()
    if current:
        print_info(f"Текущая модель: {current}")
    else:
        print_warning("Модель не найдена в стандартных местах")
    
    print(f"\n{Colors.BOLD}Стандартные места поиска:{Colors.ENDC}")
    print("  • models/best.pt")
    print("  • runs/best.pt")
    print("  • best.pt")
    print(f"  • {Path.home() / '.simscan' / 'models' / 'best.pt'}")
    
    new_path_str = get_input("\nВведите новый путь (или Enter для пропуска)")
    if not new_path_str:
        return
    
    new_path = Path(new_path_str)
    if not new_path.exists():
        print_error(f"Файл не найден: {new_path}")
        return
    
    # Создать стандартную директорию и скопировать модель
    default_model_dir = Path.home() / ".simscan" / "models"
    default_model_dir.mkdir(parents=True, exist_ok=True)
    
    import shutil
    target = default_model_dir / "best.pt"
    
    copy = get_input(f"Скопировать модель в {target}? (y/n)", "y").lower()
    if copy == 'y':
        shutil.copy(new_path, target)
        print_success(f"Модель скопирована: {target}")


def show_menu():
    """Показать главное меню."""
    while True:
        clear_screen()
        print_header("E57 → IFC КОНВЕРТЕР")
        
        model_path = get_model_path()
        if model_path:
            print_success(f"Модель: {model_path}")
        else:
            print_warning("Модель не настроена")
        
        print(f"\n{Colors.BOLD}ГЛАВНОЕ МЕНЮ:{Colors.ENDC}\n")
        print(f"  {Colors.OKCYAN}1.{Colors.ENDC} Полная конвертация E57 → IFC")
        print(f"  {Colors.OKCYAN}2.{Colors.ENDC} Инспекция E57 файла")
        print(f"  {Colors.OKCYAN}3.{Colors.ENDC} Пакетная конвертация")
        print(f"  {Colors.OKCYAN}4.{Colors.ENDC} Настройка модели")
        print(f"  {Colors.OKCYAN}5.{Colors.ENDC} О программе")
        print(f"  {Colors.OKCYAN}6.{Colors.ENDC} Аналитика E57 → архив для разбора")
        print(f"  {Colors.OKCYAN}0.{Colors.ENDC} Выход")
        
        choice = get_input("\nВыберите опцию (0-6)", "1")
        
        if choice == '1':
            full_pipeline()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '2':
            inspect_e57()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '3':
            batch_convert()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '4':
            configure_model()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '5':
            show_about()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '6':
            analytics_e57()
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")
        elif choice == '0':
            print_info("До свидания!")
            sys.exit(0)
        else:
            print_warning("Неверный выбор")
            input(f"\n{Colors.OKCYAN}Нажмите Enter для продолжения...{Colors.ENDC}")


def show_about():
    """Показать информацию о программе."""
    clear_screen()
    print_header("О ПРОГРАММЕ")
    
    print(f"""
{Colors.BOLD}DotCloudCONVERT{Colors.ENDC} - E57 to IFC Converter
    
Инструмент для автоматической конвертации облаков точек (E57) в модели IFC
с использованием нейронных сетей для извлечения планов помещений.

{Colors.BOLD}Возможности:{Colors.ENDC}
  • Автоматическое извлечение планов из облаков точек
  • Распознавание стен, дверей, окон, помещений
  • План: vectorize (по выходу сети: прямые стены, ось по центру, дуги, чистка мусора)
    или cellplan (облако + вероятности сети), или оба для сравнения
  • Test-Time Augmentation: flip (4 отражения) или d4 (8 вариантов)
  • Экспорт в формат IFC для ArchiCAD и других BIM систем
  • Пакетная обработка нескольких файлов
  • Детальная инспекция E57 файлов

{Colors.BOLD}Пайплайн обработки:{Colors.ENDC}
  1. Inference: нейросеть извлекает план из облака точек
  2. План: vectorize и/или cellplan → стены (оси, толщины, дуги), проёмы, помещения
  3. IFC Export: все уровни → один IFC, этаж на уровень

{Colors.BOLD}Требования:{Colors.ENDC}
  • Python 3.10-3.14
  • Обученная модель (best.pt)
  • E57 файлы с облаками точек

{Colors.BOLD}Разработчик:{Colors.ENDC} DotCloudCONVERT Team
{Colors.BOLD}Версия:{Colors.ENDC} 1.0.0
    """)


if __name__ == "__main__":
    try:
        show_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Прервано пользователем{Colors.ENDC}")
        sys.exit(0)
    except Exception as e:
        print_error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
