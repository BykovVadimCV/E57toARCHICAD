"""Reject synthetic scenes whose acquisition does not cover every room.

Full-coverage mode is an acquisition contract, not merely a semantic-label
change: a generated floor is accepted only when each room has a valid scanner
station with enough real raycast returns from architectural surfaces.
"""
from __future__ import annotations

from collections import defaultdict


def verify_room_scans(floors, stations: list[dict], cfg) -> dict:
    """Validate per-room scan evidence and return JSON-ready coverage metadata.

    'architectural_valid' is measured from *actual raycast labels* (wall,
    column, door, window), not from the layout polygons or rendered ground truth.
    Room station membership is recorded before registration noise.
    """
    required = bool(cfg.require_all_rooms_scanned)
    minimum_valid = int(cfg.min_valid_returns_per_room)
    minimum_arch = int(cfg.min_architectural_returns_per_room)
    if minimum_valid < 0 or minimum_arch < 0:
        raise ValueError("Room return thresholds cannot be negative")
    counts = defaultdict(lambda: {"stations": 0, "valid_returns": 0,
                                  "architectural_returns": 0})
    for st in stations:
        key = (int(st["level"]), int(st["room_id"]))
        counts[key]["stations"] += 1
        counts[key]["valid_returns"] += int(st["valid"])
        counts[key]["architectural_returns"] += int(st["architectural_valid"])

    report = []
    failures = []
    for level, layout in enumerate(floors):
        for room in layout.rooms:
            key = (level, room.id)
            n = counts[key]
            has_station = n["stations"] > 0
            sufficient = (has_station and n["valid_returns"] >= minimum_valid
                          and n["architectural_returns"] >= minimum_arch)
            if required and (not room.scanned or not sufficient):
                failures.append(
                    f"L{level + 1}/R{room.id} ({room.kind}): "
                    f"{n['stations']} stations, {n['valid_returns']} valid returns, "
                    f"{n['architectural_returns']} architectural returns"
                )
            report.append({"level": level + 1, "room_id": room.id, "kind": room.kind,
                           "station_count": n["stations"],
                           "valid_returns": n["valid_returns"],
                           "architectural_returns": n["architectural_returns"],
                           "verified": sufficient,
                           "room_scanned_flag": bool(room.scanned)})
    if failures:
        raise RuntimeError("Incomplete apartment scan; reject scene: " + "; ".join(failures))
    return {"required": required, "total_rooms": len(report),
            "verified_rooms": sum(int(row["verified"]) for row in report),
            "min_valid_returns_per_room": minimum_valid,
            "min_architectural_returns_per_room": minimum_arch,
            "rooms": report}
