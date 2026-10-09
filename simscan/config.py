"""Конфигурация синтезатора. Все пороги и диапазоны живут только здесь.

Диапазон (lo, hi) означает: значение разыгрывается равномерно для каждой сцены
(или для каждого объекта, если сказано в комментарии). Это рандомизация домена:
чем шире диапазоны, тем меньше модель привязывается к одной «идеальной» синтетике.

Числа - стартовые. Параметры сканера и шума НЕ сверены с паспортами
конкретных сканеров; подбирать по статистике реального скана
(плотность от дальности, доля пропусков, RMS на плоской грани).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

Range = tuple[float, float]


@dataclass
class LayoutConfig:
    # "apartment" - квартиры (ProcTHOR-подобный рост комнат, simscan/apartment.py);
    # "simscan" - простое BSP-разбиение; "datasetgen" - планировки ReFloorBRUSNIKA
    source: str = "apartment"
    datasetgen_path: str = ""           # путь к datasetgen.py или к его каталогу
    datasetgen_strategies: tuple[str, ...] = ("central", "linear", "radial", "graph", "open_plan",
                                              "asymmetric", "courtyard", "spine")
    datasetgen_long_side_m: Range = (8.0, 16.0)   # масштаб px -> м по длинной стороне здания
    datasetgen_min_room_m2: float = 0.5           # меньше - пустота между стенами, не помещение
    datasetgen_balcony_probability: float = 0.7
    parapet_height_m: Range = (1.0, 1.1)          # ограждение балкона
    # source = "apartment": квартиры в секционном доме (simscan/apartment.py)
    apartment_cell_m: float = 0.3
    apartment_candidates: int = 40                # кандидатов на контур, берётся лучший
    apartment_shell_trials: int = 30
    apartment_zone_trials: int = 12               # нарезок каждой зоны, лучшая из 3 удачных
    apartment_keep_best_of: int = 5               # удачных планов на контур, берётся лучший
    apartment_facade_weights: tuple[float, float, float] = (0.5, 0.25, 0.25)  # один фасад / сквозная / угловая
    apartment_p_cut: float = 0.3                  # угловой вырез контура у входа
    apartment_p_loggia: float = 0.6
    apartment_p_glazed_loggia: float = 0.5
    apartment_party_wall_m: tuple[float, ...] = (0.16, 0.18, 0.2, 0.24)
    apartment_p_bearing_wall: float = 0.5
    parapet_thickness_m: float = 0.12
    footprint_x_m: Range = (6.0, 16.0)
    footprint_y_m: Range = (5.0, 12.0)
    rooms: tuple[int, int] = (3, 9)
    min_room_side_m: float = 1.6
    snap_m: float = 0.05
    p_l_shape: float = 0.35
    ceiling_height_m: Range = (2.5, 3.3)
    exterior_wall_m: tuple[float, ...] = (0.3, 0.38, 0.4, 0.5, 0.6)
    interior_wall_m: tuple[float, ...] = (0.08, 0.1, 0.12, 0.12, 0.16, 0.2, 0.25)
    door_width_m: Range = (0.7, 0.9)
    door_head_m: Range = (2.0, 2.1)
    door_margin_m: float = 0.3          # от концов общего участка стены до проёма
    p_extra_door: float = 0.25          # дополнительные двери сверх остовного дерева
    p_passage: float = 0.1              # проём без полотна (арка)
    passage_width_m: Range = (0.8, 1.6)
    passage_head_m: Range = (2.0, 2.4)
    p_entrance: float = 1.0
    p_duplex: float = 0.0               # двухуровневая квартира: второй этаж, лестница, проём (duplex.py)
    duplex_p_spiral: float = 0.3        # доля винтовых лестниц (иначе прямой марш у глухой стены)
    entrance_width_m: Range = (0.9, 1.0)
    window_width_m: Range = (0.6, 1.8)
    window_sill_m: Range = (0.6, 0.9)   # на здание
    window_head_m: Range = (2.0, 2.3)   # на здание, не выше потолка - 0,15
    window_margin_m: float = 0.4
    p_window_per_side: float = 0.75
    min_window_room_area_m2: float = 5.0
    p_suspended_ceiling: float = 0.15   # на помещение
    suspended_drop_m: Range = (0.1, 0.35)


@dataclass
class InteriorConfig:
    furniture: bool = True              # False - без мебели вовсе (и без обязательной кухни/санузла)
    p_bare: float = 0.0                 # доля квартир «под отделку»: только сантехника и патроны на проводе,
                                        # без мебели, штор, вещей и плинтусов, строительного мусора мало
    furniture_per_m2: float = 0.12
    p_dark_furniture: float = 0.2       # отражательная способность 0,03-0,1
    p_wardrobe_to_ceiling: float = 0.3
    p_closed_door: float = 0.2
    door_open_deg: Range = (20.0, 110.0)
    radiators: bool = True
    baseboards: bool = True
    p_soffit: float = 0.12              # короб под потолком вдоль стены
    p_column: float = 0.25              # колонна в помещении площадью > column_min_area
    column_min_area_m2: float = 15.0
    p_pilaster: float = 0.15
    p_mirror_bath: float = 0.7
    p_mirror_other: float = 0.05
    clutter_per_m2: float = 0.05
    procedural: bool = True             # мебель из частей (ножки, полки, спинки) вместо цельных коробок
    p_curtains: float = 0.6             # шторы на окне жилой комнаты
    p_tulle: float = 0.5                # тюль (пропускает часть лучей)
    tulle_transmit: Range = (0.4, 0.8)
    door_frames: bool = True            # коробки и наличники дверей
    p_risers: float = 0.8               # стояки в санузле (трубы или короб)
    p_plants: float = 0.3               # растение в комнате
    p_wall_decor: float = 0.4           # телевизор / картина на стене
    lamps: bool = True
    asset_dir: str = ""                 # каталог внешних моделей: <вид>/*.obj|ply|glb|stl
    p_asset: float = 0.5                # доля предметов, заменяемых внешней моделью
    mess: Range = (0.0, 1.0)            # уровень беспорядка на сцену: ткань, куртки, сумки, стопки
    scan_dir: str = ""                  # отсканированные предметы: <роль>/*.obj в метрах (simscan assets)
    scans_per_m2: float = 0.25          # при уровне беспорядка 1
    exterior_ground: bool = True
    ground_drop_m: Range = (0.0, 20.0)  # этаж над землёй
    neighbour_buildings: tuple[int, int] = (0, 4)


@dataclass
class RealismConfig:
    """Неидеальная геометрия здания. Диапазоны - на сцену."""
    enabled: bool = True
    tessellation_m: float = 0.3                  # шаг сетки на стенах и перекрытиях
    wall_unevenness_mm: Range = (1.0, 4.0)       # СКО неровности поверхности
    unevenness_corr_m: Range = (0.4, 1.2)        # длина корреляции неровности
    lean_mm_per_m: Range = (0.0, 3.0)            # отклонение от вертикали
    shear_deg_sigma: float = 0.25                # отклонение углов от 90 градусов (сдвиг плана)
    thickness_jitter_mm: float = 5.0             # фактическая толщина против номинальной


@dataclass
class ScannerConfig:
    angular_step_deg: float = 0.12      # и по азимуту, и по углу места
    elevation_min_deg: float = -60.0
    elevation_max_deg: float = 90.0
    min_range_m: float = 0.6
    max_range_m: float = 60.0
    height_m: Range = (1.3, 1.7)        # над полом
    wall_clearance_m: float = 0.5
    stations_per_m2: float = 1.0 / 18.0
    max_stations_per_room: int = 3
    p_room_unscanned: float = 0.05      # закрытое неотсканированное помещение
    tilt_sigma_deg: float = 0.0         # остаточный наклон после компенсатора
    chunk_rays: int = 2_000_000


@dataclass
class EffectsConfig:
    """Каждый эффект включается отдельным флагом."""
    range_noise: bool = True
    range_sigma_mm: Range = (0.5, 2.5)          # на сцену
    range_sigma_per_m_mm: Range = (0.0, 0.1)
    mixed_pixels: bool = False                  # эвристика; при beam_model не нужна
    mixed_jump_m: float = 0.05
    p_mixed: Range = (0.2, 0.6)
    grazing_dropout: bool = True
    grazing_start_deg: float = 75.0
    grazing_full_deg: float = 89.0
    low_signal_dropout: bool = True
    low_signal_snr0: float = 0.02
    glass: bool = True
    p_glass_pass: Range = (0.7, 0.95)
    p_glass_return: float = 0.3                 # если не прошёл: доля откликов от стекла
    mirrors: bool = True
    registration_error: bool = True
    registration_sigma_mm: Range = (1.0, 4.0)
    registration_sigma_deg: Range = (0.002, 0.02)
    people: bool = True
    people_per_station: tuple[int, int] = (0, 2)
    keep_invalid: bool = True                   # писать лучи без отклика (InvalidState = 1)
    # Модель пятна луча: на кромках луч разбивается на подлучи по гауссову профилю,
    # отклик - эхо с наибольшей энергией, дальность - средневзвешенная внутри эха.
    beam_model: bool = True
    beam_divergence_mrad: Range = (0.3, 0.6)    # НЕ ПРОВЕРЕНО для BLK360
    beam_exit_mm: float = 3.0
    echo_separation_m: float = 0.4              # ближе - одно (смешанное) эхо
    edge_jump_m: float = 0.01                   # перепад, при котором пиксель считается кромкой
    incidence_noise_power: float = 1.0          # sigma ~ sec(угла падения)^p (Soudarissanane 2011)
    noise_cap_x_base: float = 6.0               # потолок шума: столько базовых sigma (реальный
    #                                             BLK360 на скользящих углах - до 15-25)


@dataclass
class ExportConfig:
    pixel_mm: float = 10.0
    pad_m: float = 0.5
    cut_height_m: float = 1.25                  # секущая плоскость эталонного плана
    world_yaw: bool = True                      # случайный поворот здания в СК объекта
    world_offset_m: Range = (-500.0, 500.0)     # сдвиг XY
    world_z_offset_m: Range = (-50.0, 150.0)    # абсолютная отметка пола
    write_merged: bool = False                  # scan_merged.e57: одно сведённое облако, как
    #                                             экспорт заказчика (станции - в позах снимков)
    merged_spacing_mm: float = 0.0              # прореживание сведённого облака (0 - без)
    write_mesh: bool = False                    # mesh.ply с цветами классов
    unet_mask: bool = True                      # gt/unet_mask.png в формате datasetgen (ReFloorBRUSNIKA)
    unet_target_wall_px: float = 30.0           # наружная стена в px, как core/scale_norm.py
    preview: bool = True
    debug: bool = False                         # debug/*.png: панорамы, кромки, неровность, шум
    free_space_input: bool = True               # input/free_space.png - вход сети (rasterize.py)


@dataclass
class SynthConfig:
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    interior: InteriorConfig = field(default_factory=InteriorConfig)
    realism: RealismConfig = field(default_factory=RealismConfig)
    scanner: ScannerConfig = field(default_factory=ScannerConfig)
    effects: EffectsConfig = field(default_factory=EffectsConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def _merge(obj: Any, data: dict, path: str = "") -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"{path or 'config'}: ожидался словарь, получено {type(data).__name__}")
    known = {f.name: f for f in dataclasses.fields(obj)}
    changes = {}
    for key, value in data.items():
        if key not in known:
            raise KeyError(f"неизвестный параметр конфигурации: {path}{key}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current):
            changes[key] = _merge(current, value, f"{path}{key}.")
        elif isinstance(current, tuple):
            changes[key] = tuple(value)
        else:
            changes[key] = value
    return dataclasses.replace(obj, **changes)


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> SynthConfig:
    cfg = SynthConfig()
    if path is not None:
        with open(path, encoding="utf-8") as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg


def config_to_dict(cfg: SynthConfig) -> dict:
    return dataclasses.asdict(cfg)
