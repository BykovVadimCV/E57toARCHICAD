"""One semantic definition for generated architectural openings.

Physical ``Layout.Opening.kind`` is independent of ``z1`` and wall height:
``door`` has a door leaf, ``passage`` has no leaf, ``window`` is an elevated
window aperture. Height remains geometry metadata, never a class selector.

The legacy eight-class PlanNet raster calls passage class 5 ``opening``.
That name and numeric IDs stay unchanged for downstream vectorization/IFC;
new *training data* have a different version because the meaning changed.
"""
from __future__ import annotations

from types import MappingProxyType

OPENING_SEMANTICS_VERSION = "physical-kind-v1"
COARSE_LABELS_VERSION = "labels-6"
COARSE_META_VERSION = 4
HD_MASK_FORMAT = "simscan-native-hd-masks-v3"

# Public, immutable maps. Numerical IDs must stay stable for both architectures.
COARSE_OPENING_CLASSES = MappingProxyType({"door": 3, "window": 4, "passage": 5})
HD_OPENING_VALUES = MappingProxyType({"background": 0, "door": 1, "window": 2, "passage": 3})
HD_SEMANTIC_CLASSES = MappingProxyType({
    "background": 0, "wall": 1, "door": 2, "window": 3,
    "passage": 4, "physical_floor_void": 5,
})


def coarse_opening_class(kind: str) -> int:
    """Convert physical opening kind to stable eight-class PlanNet label."""
    try:
        return COARSE_OPENING_CLASSES[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown physical opening kind: {kind!r}") from exc


def hd_opening_value(kind: str) -> int:
    """Convert physical opening kind to stored native-HD openings.png value."""
    try:
        return HD_OPENING_VALUES[kind]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown physical opening kind: {kind!r}") from exc


def vector_opening_type(kind: str) -> str:
    """Preserve legacy vector ``opening`` schema while recording physical kind."""
    coarse_opening_class(kind)  # validate kind; do not silently turn unknowns into doors
    return "opening" if kind == "passage" else kind
