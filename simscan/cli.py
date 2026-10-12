"""Командная строка.

    python -m simscan generate --out D:/synth --count 200 --seed 0 [--config cfg.yaml] [--workers 2]
    python -m simscan generate --out D:/synth --count 1 --set scanner.angular_step_deg=0.3
    python -m simscan dump-config > cfg.yaml
    python -m simscan inspect-e57 D:/scans/kvartira.e57    # отчёт по реальному E57 -> _report.zip
    python -m simscan assets --source gso --out D:/scans --per-role 40
    python -m simscan pair D:/synth/scene_00000 D:/synth/scene_00001 --out pairs.png
"""
from __future__ import annotations

import argparse
import sys

# тяжёлые зависимости (Open3D, pye57) импортируются внутри команд: inspect-e57 работает без них


def _parse_set(items: list[str]) -> dict:
    """--set a.b=1 --set c.d=[1,2] -> вложенный словарь (значения разбираются как YAML)."""
    import yaml

    out: dict = {}
    for item in items:
        key, _, value = item.partition("=")
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)
    return out


def _plans(args) -> None:
    from pathlib import Path

    from .generate import make_layout, scene_rng
    from .interior import Furnisher
    from .render import render_many, render_plan

    from .config import load_config

    cfg = load_config(args.config, _parse_set(args.set))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    layouts = []
    for i in range(args.count):
        rng = scene_rng(args.seed, i)
        lay = make_layout(cfg, rng, args.seed, i)
        Furnisher(cfg.interior, rng).furnish(lay)
        lay.save_json(out / f"plan_{i:04d}.json")
        render_plan(lay, out / f"plan_{i:04d}.png")
        layouts.append(lay)
    render_many(layouts, out / "sheet.png")
    print(out / "sheet.png")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="simscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="сгенерировать сцены")
    g.add_argument("--out", required=True)
    g.add_argument("--count", type=int, default=1)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--start", type=int, default=0, help="номер первой сцены (для дозапуска)")
    g.add_argument("--workers", type=int, default=1)
    g.add_argument("--config")
    g.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    md = sub.add_parser("make-dataset", help="сцены -> сразу датасет сети плана (сканы удаляются)")
    md.add_argument("--out", required=True, help="корень датасета")
    md.add_argument("--count", type=int, default=100)
    md.add_argument("--seed", type=int, default=0)
    md.add_argument("--start", type=int, default=0, help="номер первой сцены (разные машины - разные диапазоны)")
    md.add_argument("--workers", type=int, default=1)
    md.add_argument("--work-dir", help="временный каталог сканов (по умолчанию <out>/_work)")
    md.add_argument("--keep-scans", action="store_true", help="писать scan.e57 и не удалять сцены (медленнее)")
    md.add_argument("--variants", type=int, default=1,
                    help="растров на скан: 2-3 - дешёвая аугментация (поворот облака, прореживание)")
    md.add_argument("--config")
    md.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    tr = sub.add_parser("train", help="обучить сеть плана на датасете (нужен torch)")
    tr.add_argument("--data", required=True, help="корень датасета")
    tr.add_argument("--out", required=True, help="каталог прогона: last.pt, best.pt, history.jsonl")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--crop", type=int, default=512)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--workers", type=int, default=2)
    tr.add_argument("--resume", action="store_true", help="продолжить с last.pt в --out")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--max-steps", type=int, help="остановиться после стольких шагов (проверка)")
    tr.add_argument("--accum-steps", type=int, default=1, help="gradient accumulation steps (эффективный batch = batch * accum-steps)")
    tr.add_argument("--weighted-sampling", action="store_true", help="weighted sampling для баланса классов")
    tr.add_argument("--use-focal", action="store_true", help="использовать Focal Loss вместо обычного CE")
    inf = sub.add_parser("infer", help="применить сеть плана к E57 или к каталогам с input.npy")
    inf.add_argument("ckpt", help="best.pt")
    inf.add_argument("paths", nargs="+")
    inf.add_argument("--out", required=True)
    inf.add_argument("--tile", type=int, default=512)
    inf.add_argument("--tta", action="store_true", help="Test-Time Augmentation: усреднение по 4 вариантам (flipH/V/HV), +1-2% IoU")
    d = sub.add_parser("dump-config", help="напечатать конфигурацию по умолчанию (YAML)")
    d.add_argument("--config")
    pl = sub.add_parser("plans", help="только планировки + рендер (без сканирования)")
    pl.add_argument("--out", required=True)
    pl.add_argument("--count", type=int, default=6)
    pl.add_argument("--seed", type=int, default=0)
    pl.add_argument("--config")
    pl.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    r = sub.add_parser("render", help="отрисовать layout.json сцены")
    r.add_argument("scene_dir")
    dbg = sub.add_parser("debug", help="отладочные визуализации сцены в <сцена>/debug")
    dbg.add_argument("scene_dir")
    a = sub.add_parser("assets", help="скачать реальные 3D-модели (сканы предметов, мебель)")
    a.add_argument("--source", choices=("gso", "polyhaven", "objaverse"), default="gso")
    a.add_argument("--out", required=True,
                   help="gso -> interior.scan_dir; polyhaven/objaverse -> interior.asset_dir")
    a.add_argument("--per-role", type=int, default=20)
    a.add_argument("--max-triangles", type=int, default=3000)
    a.add_argument("--seed", type=int, default=0)
    ie = sub.add_parser("inspect-e57", help="отчёт по реальному E57 для калибровки (маленький zip)")
    ie.add_argument("e57")
    ie.add_argument("--out", help="каталог отчёта (по умолчанию <файл>_report рядом с файлом)")
    ie.add_argument("--no-sample", action="store_true", help="без прореженного облака sample.npz")
    ie.add_argument("--chunk", type=int, default=2_000_000, help="точек за одно чтение")
    fp = sub.add_parser("floorplan", help="план из сведённого облака: стены, окна, двери, помещения")
    fp.add_argument("e57")
    fp.add_argument("--out", help="каталог (по умолчанию <файл>_plan рядом с файлом)")
    fp.add_argument("--px", type=float, default=0.01, help="размер пикселя растров, м")
    ds = sub.add_parser("dataset", help="датасет для сети плана: вход 8 каналов + разметка")
    ds.add_argument("action", choices=["build", "real", "splits", "pack"],
                    help="build - сцены генератора, real - реальные E57, splits - пересобрать splits/*.txt "
                         "по каталогу датасета, pack - датасет в один zip (--out архив)")
    ds.add_argument("paths", nargs="+", help="каталоги сцен (build) или файлы E57 (real)")
    ds.add_argument("--out", required=True, help="корень датасета")
    ds.add_argument("--id", help="real: имя объекта (по умолчанию имя файла)")
    ds.add_argument("--px", type=float, default=0.02, help="пиксель, м")
    ds.add_argument("--gt-levels", action="store_true", help="build: уровни из эталона, а не detect_levels")
    cp = sub.add_parser("cellplan", help="план: разбиение на ячейки по кандидатам стен и глобальная разметка")
    cp.add_argument("e57")
    cp.add_argument("--out", help="каталог (по умолчанию <файл>_cellplan рядом с файлом)")
    cp.add_argument("--px", type=float, default=0.01, help="размер пикселя, м")
    cp.add_argument("--pred", help="каталог infer для этого файла (prob.npz + scene.json по уровням)")
    cp.add_argument("--net-w", type=float, default=1.0, help="вес вероятностей сети в задаче")
    cr = sub.add_parser("calib-real", help="всё для подгонки генератора из E57 в один zip")
    cr.add_argument("e57", nargs="+", help="файлы E57 (реальные или scan.e57 сцен)")
    cr.add_argument("--out", required=True, help="каталог отчёта; рядом пишется <каталог>.zip")
    cc = sub.add_parser("calib-compare", help="сверка отчётов calib-real: реальные против синтетики")
    cc.add_argument("--real", nargs="+", required=True, help="каталоги отчётов calib-real реальных объектов")
    cc.add_argument("--syn", nargs="+", required=True, help="каталоги отчётов calib-real синтетики")
    cc.add_argument("--out", default="calib_compare.png", help="рисунок сверки")
    sm = sub.add_parser("slice-map", help="тонкий срез у пола (под радиаторами) от местного пола")
    sm.add_argument("e57")
    sm.add_argument("--out", help="каталог (по умолчанию <файл>_slice рядом с файлом)")
    sm.add_argument("--z0", type=float, default=0.03, help="низ среза над полом, м")
    sm.add_argument("--z1", type=float, default=0.08, help="верх среза над полом, м")
    sm.add_argument("--px", type=float, default=0.01, help="пиксель, м")
    wh = sub.add_parser("wall-heatmap", help="тепловая карта стен: срезы по высоте с весами")
    wh.add_argument("e57")
    wh.add_argument("--out", help="каталог (по умолчанию <файл>_heat рядом с файлом)")
    wh.add_argument("--px", type=float, default=0.01, help="размер пикселя, м")
    wh.add_argument("--slice", type=float, default=0.1, help="толщина среза, м")
    wh.add_argument("--prior", default="0:1,0.8:1,1.0:0.1,2.0:0.1,2.2:0.6,9:0.6",
                    help="вес среза по высоте над полом: «высота:вес,...» (ломаная)")
    cm = sub.add_parser("compare-e57", help="сравнить отчёты inspect-e57 (реальный скан и синтетика)")
    cm.add_argument("reports", nargs="+", help="report.json")
    cm.add_argument("--names", nargs="+")
    cm.add_argument("--out", default="compare.png")
    pr = sub.add_parser("pair", help="обучающая пара сцены: растр свободного пространства + эталон")
    pr.add_argument("scene_dirs", nargs="+")
    pr.add_argument("--out", help="общая картинка (по умолчанию <сцена>/input/pair.png)")
    vec = sub.add_parser("vectorize", help="векторизация предсказаний сети в векторный план")
    vec.add_argument("pred_dir", help="каталог предсказаний (из infer): sem.png, dist.npy, ...")
    vec.add_argument("--out", required=True, help="каталог для векторного плана")
    exp = sub.add_parser("export-ifc", help="экспорт векторного плана в IFC для ArchiCAD")
    exp.add_argument("plan_json", nargs="+", help="plan.json из vectorize (несколько - этажи одного IFC)")
    exp.add_argument("--out", required=True, help="выходной IFC файл (.ifc)")
    args = ap.parse_args(argv)

    if args.cmd == "assets":
        from . import assets

        fn = {"gso": assets.fetch_gso, "polyhaven": assets.fetch_polyhaven,
              "objaverse": assets.fetch_objaverse}[args.source]
        kw = {"per_role": args.per_role, "seed": args.seed} if args.source == "gso" \
            else {"max_per_kind": args.per_role}
        print(fn(args.out, max_triangles=args.max_triangles, **kw))
        return
    if args.cmd == "floorplan":
        from pathlib import Path

        from .floorplan import floorplan

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_plan")
        floorplan(src, out, px=args.px)
        print(out)
        return
    if args.cmd == "train":
        from .train import train

        print(train(args.data, args.out, epochs=args.epochs, batch=args.batch, crop=args.crop, lr=args.lr,
                    workers=args.workers, resume=args.resume, seed=args.seed, max_steps=args.max_steps,
                    accum_steps=args.accum_steps, weighted_sampling=args.weighted_sampling,
                    use_focal=args.use_focal))
        return
    if args.cmd == "infer":
        from .infer import infer

        infer(args.ckpt, args.paths, args.out, tile=args.tile, tta=args.tta)
        return
    if args.cmd == "make-dataset":
        from .config import load_config
        from .makedata import make_dataset

        cfg = load_config(args.config, _parse_set(args.set))
        make_dataset(cfg, args.out, args.count, args.seed, args.start, args.workers, work_dir=args.work_dir,
                     keep_scans=args.keep_scans, variants=args.variants)
        return
    if args.cmd == "dataset":
        from .makedata import pack, rebuild_splits
        from .netinput import build_real, build_scene

        if args.action == "splits":
            for p in args.paths:
                print(p, rebuild_splits(p))
            return
        if args.action == "pack":
            print(pack(args.paths[0], args.out))
            return
        for p in args.paths:
            if args.action == "build":
                print("\n".join(build_scene(p, args.out, px=args.px, use_gt_levels=args.gt_levels)))
            else:
                print("\n".join(build_real(p, args.out, object_id=args.id, px=args.px)))
        return
    if args.cmd == "cellplan":
        from pathlib import Path

        from .cellplan import cellplan

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_cellplan")
        cellplan(src, out, px=args.px, pred=args.pred, net_w=args.net_w)
        print(out)
        return
    if args.cmd == "calib-real":
        from .calib_real import calib_real

        print(calib_real(args.e57, args.out))
        return
    if args.cmd == "calib-compare":
        from .calib_real import compare_calib

        res = compare_calib({"реальные": args.real, "синтетика": args.syn}, args.out)
        r, y = res["реальные"], res["синтетика"]
        for k in r:
            if not k.startswith("_"):
                print(f"{k:34s} {r[k][0]:9.3f} {y[k][0]:9.3f}  x{y[k][0] / r[k][0] if r[k][0] else float('nan'):.2f}")
        print(args.out)
        return
    if args.cmd == "slice-map":
        from pathlib import Path

        from .wallheat import slice_map

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_slice")
        slice_map(src, out, z0=args.z0, z1=args.z1, px=args.px)
        print(out)
        return
    if args.cmd == "wall-heatmap":
        from pathlib import Path

        from .wallheat import wall_heatmap

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_heat")
        wall_heatmap(src, out, px=args.px, slice_m=args.slice, prior=args.prior)
        print(out)
        return
    if args.cmd == "compare-e57":
        from pathlib import Path

        from .inspect_e57 import compare_reports

        names = args.names or [Path(p).parent.name for p in args.reports]
        print(compare_reports(args.reports, names, Path(args.out)))
        print(args.out)
        return
    if args.cmd == "inspect-e57":
        from pathlib import Path

        from .inspect_e57 import inspect_e57

        src = Path(args.e57)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_report")
        rep = inspect_e57(src, out, chunk=args.chunk, sample=not args.no_sample)
        print((out / "summary.txt").read_text(encoding="utf-8"))
        print("отправить:", rep.get("zip"))
        return
    if args.cmd == "pair":
        import json
        from pathlib import Path

        from .rasterize import pair_data, pair_sheet

        items = [pair_data(d) for d in args.scene_dirs]
        out = Path(args.out) if args.out else Path(args.scene_dirs[0]) / "input" / "pair.png"
        pair_sheet(items, out)
        print(json.dumps({d["name"]: {k: v for k, v in d["stats"].items() if k.startswith("wall")}
                          for d in items}, ensure_ascii=False, indent=1))
        print(out)
        return

    if args.cmd == "vectorize":
        from pathlib import Path

        from .vectorize import main_vectorize

        main_vectorize(args.pred_dir, args.out)
        return

    if args.cmd == "export-ifc":
        from pathlib import Path

        from .export_ifc import main_export_ifc

        main_export_ifc(args.plan_json, args.out)
        return

    if args.cmd == "debug":
        import json

        from .debug import debug_scene

        print(json.dumps(debug_scene(args.scene_dir), ensure_ascii=False, indent=1)[:2000])
        return

    if args.cmd == "plans":
        _plans(args)
        return
    if args.cmd == "render":
        from pathlib import Path

        from .layout import Layout
        from .render import render_plan

        d = Path(args.scene_dir)
        render_plan(Layout.load_json(d / "layout.json"), d / "plan.png")
        print(d / "plan.png")
        return

    if args.cmd == "dump-config":
        import yaml

        from .config import config_to_dict, load_config

        cfg = load_config(args.config)
        yaml.safe_dump(config_to_dict(cfg), sys.stdout, allow_unicode=True, sort_keys=False)
        return
    from .config import load_config
    from .generate import generate_dataset

    cfg = load_config(args.config, _parse_set(args.set))
    results = generate_dataset(cfg, args.out, args.count, args.seed, args.start, args.workers)
    failed = [r for r in results if "error" in r]
    if failed:
        print(f"ошибок: {len(failed)} из {len(results)}", file=sys.stderr)
        sys.exit(1)
