"""Тестовый E57 как у Leica: scaled integer xyz, целые интенсивность и цвет, поле-константа."""
import sys
import numpy as np
from pye57 import libe57

def write_scaled(path, n=300_000, seed=0):
    rng = np.random.default_rng(seed)
    imf = libe57.ImageFile(path, "w")
    root = imf.root()
    root.set("formatName", libe57.StringNode(imf, "ASTM E57 3D Imaging Data File"))
    root.set("guid", libe57.StringNode(imf, "{test}"))
    root.set("versionMajor", libe57.IntegerNode(imf, 1))
    root.set("versionMinor", libe57.IntegerNode(imf, 0))
    data3d = libe57.VectorNode(imf, True)
    root.set("data3D", data3d)
    root.set("images2D", libe57.VectorNode(imf, True))
    scan = libe57.StructureNode(imf)
    scan.set("guid", libe57.StringNode(imf, "{scan}"))
    scan.set("name", libe57.StringNode(imf, "Setup 1"))
    scan.set("sensorSerialNumber", libe57.StringNode(imf, "SN-123"))
    pose = libe57.StructureNode(imf)
    rot = libe57.StructureNode(imf)
    for k, v in zip("wxyz", (0.9238795, 0.0, 0.0, 0.3826834)):
        rot.set(k, libe57.FloatNode(imf, v))
    tr = libe57.StructureNode(imf)
    for k, v in zip("xyz", (10.0, -5.0, 1.5)):
        tr.set(k, libe57.FloatNode(imf, v))
    pose.set("rotation", rot)
    pose.set("translation", tr)
    scan.set("pose", pose)
    proto = libe57.StructureNode(imf)
    for k in ("cartesianX", "cartesianY", "cartesianZ"):
        proto.set(k, libe57.ScaledIntegerNode(imf, 0, -500000, 500000, 0.0001, 0.0))
    proto.set("intensity", libe57.IntegerNode(imf, 0, 0, 2047))
    for k in ("colorRed", "colorGreen", "colorBlue"):
        proto.set(k, libe57.IntegerNode(imf, 0, 0, 255))
    proto.set("rowIndex", libe57.IntegerNode(imf, 0, 0, 1249))
    proto.set("columnIndex", libe57.IntegerNode(imf, 0, 0, 2999))
    proto.set("returnIndex", libe57.IntegerNode(imf, 0, 0, 0))
    proto.set("cartesianInvalidState", libe57.IntegerNode(imf, 0, 0, 2))
    codecs = libe57.VectorNode(imf, True)
    points = libe57.CompressedVectorNode(imf, proto, codecs)
    scan.set("points", points)
    data3d.append(scan)
    vals = {
        "cartesianX": np.round(rng.uniform(-49, 49, n), 4), "cartesianY": np.round(rng.uniform(-49, 49, n), 4),
        "cartesianZ": np.round(rng.uniform(-3, 3, n), 4),
        "intensity": rng.integers(0, 2048, n).astype(np.uint16),
        "colorRed": rng.integers(0, 256, n).astype(np.uint8), "colorGreen": rng.integers(0, 256, n).astype(np.uint8),
        "colorBlue": rng.integers(0, 256, n).astype(np.uint8),
        "rowIndex": rng.integers(0, 1250, n).astype(np.uint16), "columnIndex": rng.integers(0, 3000, n).astype(np.uint16),
        "returnIndex": np.zeros(n, np.uint8), "cartesianInvalidState": rng.integers(0, 3, n).astype(np.int8),
    }
    bufs = libe57.VectorSourceDestBuffer()
    for k, a in vals.items():
        a = np.ascontiguousarray(a)
        vals[k] = a
        bufs.append(libe57.SourceDestBuffer(imf, k, a, n, True, True))
    w = points.writer(bufs)
    w.write(n)
    w.close()
    imf.close()
    return vals

