"""Чтение E57 на чистом numpy (ASTM E2807), без libE57/pye57.

Зачем: pye57 собран не под все платформы (на PyPI нет сборок для Intel-Mac), а анализатору
реального файла нужна только выборка точек. Читатель поддерживает то, что пишут сканеры и
Cyclone/REGISTER 360: страницы с CRC, XML-заголовок, сжатые векторы с кодеком bitPack
(float32/float64, integer и scaled integer любой разрядности до 56 бит), блобы (только размер).

    with E57Reader(path) as f:
        f.scans[i]          # имя, число точек, поля и их типы, поза, границы сетки
        for chunk in f.iter_points(i, ["cartesianX", "rowIndex"], chunk=2_000_000): ...

Формат (кратко): файл - страницы по page_size (1024) байт, последние 4 байта страницы - CRC32C;
«логический» поток - страницы без CRC. Заголовок 48 байт: "ASTM-E57", версия, длина,
смещение и длина XML. Сжатый вектор: секция (id=1, длина, смещение данных) из пакетов
данных (тип 1: число потоков, длины буферов, буферы), индексных (0) и пустых (2). Поток на
каждое поле прототипа; значения в потоке идут подряд через границы пакетов, целые упакованы
по (max-min) битам, младший бит первый.
"""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1].split(":", 1)[-1]


@dataclass
class FieldSpec:
    name: str
    kind: str                       # Float | Integer | ScaledInteger
    double: bool = False
    minimum: int = 0
    maximum: int = 0
    scale: float = 1.0
    offset: float = 0.0

    @property
    def bits(self) -> int:
        if self.kind == "Float":
            return 64 if self.double else 32
        span = self.maximum - self.minimum
        return int(span).bit_length() if span > 0 else 0

    def describe(self) -> str:
        if self.kind == "Float":
            return "float64" if self.double else "float32"
        if self.kind == "ScaledInteger":
            return f"scaled_integer(scale={self.scale:g}, bits={self.bits})"
        return f"integer[{self.minimum}..{self.maximum}]"


@dataclass
class ScanInfo:
    index: int
    name: str | None
    points: int
    file_offset: int
    fields: list[FieldSpec]
    rotation: list[float] | None = None          # w, x, y, z
    translation: list[float] | None = None
    index_bounds: dict = field(default_factory=dict)
    tree: dict = field(default_factory=dict)

    @property
    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


class E57Reader:
    def __init__(self, path):
        self.path = Path(path)
        self.f = open(self.path, "rb")
        head = self.f.read(48)
        if head[:8] != b"ASTM-E57":
            raise ValueError(f"{self.path}: не E57 (нет сигнатуры ASTM-E57)")
        (self.major, self.minor, self.phys_length, xml_off, xml_len,
         self.page) = struct.unpack("<IIQQQQ", head[8:48])
        self.logical_page = self.page - 4
        xml = self._read_logical(xml_off, xml_len)
        self.root_xml = ET.fromstring(xml)
        self.tree = self._to_tree(self.root_xml)
        self.scans = [self._scan(i, el) for i, el in enumerate(self._children(self.root_xml, "data3D"))]
        self.images = list(self._children(self.root_xml, "images2D"))

    # --- служебное ---------------------------------------------------------
    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _read_logical(self, phys_off: int, n: int) -> bytes:
        """n логических байт начиная с физического смещения (CRC страниц пропускаются)."""
        out = bytearray()
        pos = phys_off
        while len(out) < n:
            in_page = pos % self.page
            take = min(self.logical_page - in_page, n - len(out))
            if take <= 0:                          # попали на CRC - на следующую страницу
                pos += self.page - in_page
                continue
            self.f.seek(pos)
            out += self.f.read(take)
            pos += take
            if pos % self.page == self.logical_page:
                pos += 4
        return bytes(out)

    def _stream(self, phys_off: int, block_pages: int = 4096):
        """Логические байты от физического смещения до конца файла, блоками."""
        pos = phys_off
        in_page = pos % self.page
        if in_page:
            self.f.seek(pos)
            yield self.f.read(max(self.logical_page - in_page, 0))
            pos += self.page - in_page
        while pos < self.phys_length:
            self.f.seek(pos)
            raw = self.f.read(self.page * block_pages)
            if not raw:
                return
            full = len(raw) // self.page
            if full:
                pages = np.frombuffer(raw[:full * self.page], np.uint8).reshape(full, self.page)
                yield pages[:, :self.logical_page].tobytes()
            rest = raw[full * self.page:]
            if rest:
                yield rest[:self.logical_page]
            pos += len(raw)

    @staticmethod
    def _children(root, name):
        for el in root:
            if _local(el.tag) == name:
                return list(el)
        return []

    def _to_tree(self, el):
        t = el.get("type")
        if t in ("Structure", None) and len(el):
            return {_local(c.tag): self._to_tree(c) for c in el}
        if t == "Vector":
            return [self._to_tree(c) for c in el]
        if t == "CompressedVector":
            return {"records": int(el.get("recordCount", 0))}
        if t == "Blob":
            return {"blob_bytes": int(el.get("length", 0))}
        text = (el.text or "").strip()
        if t == "Integer":
            return int(text) if text else 0
        if t in ("Float", "ScaledInteger"):
            return float(text) if text else 0.0
        if "serial" in _local(el.tag).lower():
            return "<скрыто>"
        return text

    @staticmethod
    def _field(el) -> FieldSpec:
        t = el.get("type")
        name = _local(el.tag)
        if t == "Float":
            return FieldSpec(name, "Float", double=el.get("precision", "double") != "single")
        lo, hi = int(el.get("minimum", 0)), int(el.get("maximum", 0))
        if t == "ScaledInteger":
            return FieldSpec(name, "ScaledInteger", minimum=lo, maximum=hi,
                             scale=float(el.get("scale", 1)), offset=float(el.get("offset", 0)))
        if t == "Integer":
            return FieldSpec(name, "Integer", minimum=lo, maximum=hi)
        raise ValueError(f"поле {name}: тип {t} в точках не поддерживается")

    def _scan(self, i: int, el) -> ScanInfo:
        kids = {_local(c.tag): c for c in el}
        pts = kids["points"]
        proto = next(c for c in pts if _local(c.tag) == "prototype")
        codecs = [c for c in pts if _local(c.tag) == "codecs"]
        if codecs and len(codecs[0]):
            raise ValueError("сжатый вектор с нестандартным кодеком - поддерживается только bitPack")
        tree = self._to_tree(el)
        pose = tree.get("pose", {}) if isinstance(tree, dict) else {}
        rot = pose.get("rotation")
        tr = pose.get("translation")
        return ScanInfo(
            index=i, name=tree.get("name") if isinstance(tree, dict) else None,
            points=int(pts.get("recordCount", 0)), file_offset=int(pts.get("fileOffset", 0)),
            fields=[self._field(c) for c in proto],
            rotation=[rot.get(k, 0.0) for k in ("w", "x", "y", "z")] if isinstance(rot, dict) else None,
            translation=[tr.get(k, 0.0) for k in ("x", "y", "z")] if isinstance(tr, dict) else None,
            index_bounds=tree.get("indexBounds", {}) if isinstance(tree, dict) else {},
            tree=tree)

    # --- точки ----------------------------------------------------------------
    def iter_points(self, index: int, names: list[str], chunk: int = 2_000_000):
        """Куски {поле: массив} по chunk записей (последний - короче). Целые - int64,
        scaled integer - float64 (value * scale + offset), float - как в файле."""
        scan = self.scans[index]
        want = [k for k, f in enumerate(scan.fields) if f.name in names]
        dec = {k: _Decoder(scan.fields[k]) for k in want}
        n_streams = len(scan.fields)
        total = scan.points
        done = 0
        # заголовок секции: id(1) + 7 резерв, длина секции, смещение данных, смещение индекса
        sec = self._read_logical(scan.file_offset, 32)
        sec_id, = struct.unpack("<B", sec[:1])
        if sec_id != 1:
            raise ValueError(f"скан {index}: неверная секция сжатого вектора ({sec_id})")
        _, data_off, _ = struct.unpack("<QQQ", sec[8:32])
        buf = bytearray()
        src = self._stream(data_off)
        pos = 0

        def need(n):
            nonlocal buf, pos
            while len(buf) - pos < n:
                try:
                    blk = next(src)
                except StopIteration:
                    raise ValueError(f"скан {index}: файл кончился внутри сжатого вектора")
                if pos > (1 << 20):
                    del buf[:pos]
                    pos = 0
                buf += blk

        while done < total:
            need(4)
            ptype = buf[pos]
            length = struct.unpack_from("<H", buf, pos + 2)[0] + 1
            need(length)
            if ptype == 1:
                count = struct.unpack_from("<H", buf, pos + 4)[0]
                if count != n_streams:
                    raise ValueError(f"скан {index}: потоков {count}, полей {n_streams}")
                lens = struct.unpack_from(f"<{count}H", buf, pos + 6)
                off = pos + 6 + 2 * count
                for k, ln in enumerate(lens):
                    if k in dec and ln:
                        dec[k].feed(bytes(buf[off:off + ln]))
                    off += ln
            elif ptype not in (0, 2):
                raise ValueError(f"скан {index}: неизвестный тип пакета {ptype}")
            pos += length
            ready = min(dec[k].available for k in want) if want else total - done
            ready = min(ready, total - done)
            while ready >= chunk or (ready > 0 and done + ready >= total):
                n = min(chunk, ready)
                yield {scan.fields[k].name: dec[k].take(n) for k in want}
                done += n
                ready -= n
            if not want and ready:
                done = total


class _Decoder:
    """Поток одного поля: принимает байты пакетов, отдаёт значения."""

    def __init__(self, spec: FieldSpec):
        self.spec = spec
        self.bits = spec.bits
        if spec.kind != "Float" and self.bits > 56:
            raise ValueError(f"поле {spec.name}: {self.bits} бит не поддерживается")
        self.raw = bytearray()
        self.bitpos = 0                              # позиция первого необработанного бита
        self.vals: list[np.ndarray] = []
        self.n_vals = 0
        self.const = self.bits == 0
        self.pending_const = 0

    @property
    def available(self) -> int:
        return 10 ** 18 if self.const else self.n_vals

    def feed(self, data: bytes) -> None:
        if self.const:
            return
        self.raw += data
        if self.spec.kind == "Float":
            w = 8 if self.spec.double else 4
            n = len(self.raw) // w
            if n:
                arr = np.frombuffer(bytes(self.raw[:n * w]), "<f8" if w == 8 else "<f4").copy()
                del self.raw[:n * w]
                self._push(arr)
            return
        b = self.bits
        n = (len(self.raw) * 8 - self.bitpos) // b
        if n == 0:
            return
        start = self.bitpos + np.arange(n, dtype=np.int64) * b
        byte = start >> 3
        shift = (start & 7).astype(np.uint64)
        src = np.frombuffer(bytes(self.raw) + b"\0" * 8, np.uint8)
        win = np.zeros(n, np.uint64)
        nbytes = (b + 7 + 7) // 8                    # байт, которые может задеть значение
        for j in range(min(nbytes, 8)):
            win |= src[byte + j].astype(np.uint64) << np.uint64(8 * j)
        v = (win >> shift) & np.uint64((1 << b) - 1)
        end = self.bitpos + n * b
        drop = end >> 3
        del self.raw[:drop]
        self.bitpos = end & 7
        self._push(self._value(v.astype(np.int64)))

    def _value(self, v: np.ndarray) -> np.ndarray:
        v = v + self.spec.minimum
        if self.spec.kind == "ScaledInteger":
            return v * self.spec.scale + self.spec.offset
        return v

    def _push(self, arr: np.ndarray) -> None:
        self.vals.append(arr)
        self.n_vals += len(arr)

    def take(self, n: int) -> np.ndarray:
        if self.const:
            v = np.full(n, self.spec.minimum, np.int64)
            return self._value(v - self.spec.minimum)
        out, need = [], n
        while need:
            a = self.vals[0]
            if len(a) <= need:
                out.append(a)
                self.vals.pop(0)
                need -= len(a)
            else:
                out.append(a[:need])
                self.vals[0] = a[need:]
                need = 0
        self.n_vals -= n
        return np.concatenate(out) if len(out) > 1 else out[0]
