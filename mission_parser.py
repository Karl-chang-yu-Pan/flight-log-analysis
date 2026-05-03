from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


COMMAND_NAMES = {
    16: "MAV_CMD_NAV_WAYPOINT",
    17: "MAV_CMD_NAV_LOITER_UNLIM",
    18: "MAV_CMD_NAV_LOITER_TURNS",
    19: "MAV_CMD_NAV_LOITER_TIME",
    20: "MAV_CMD_NAV_RETURN_TO_LAUNCH",
    21: "MAV_CMD_NAV_LAND",
    22: "MAV_CMD_NAV_TAKEOFF",
    84: "MAV_CMD_NAV_VTOL_TAKEOFF",
    85: "MAV_CMD_NAV_VTOL_LAND",
    177: "MAV_CMD_DO_JUMP",
    178: "MAV_CMD_DO_CHANGE_SPEED",
    181: "MAV_CMD_DO_SET_RELAY",
    183: "MAV_CMD_DO_SET_SERVO",
}

FRAME_NAMES = {
    0: "MAV_FRAME_GLOBAL",
    3: "MAV_FRAME_GLOBAL_RELATIVE_ALT",
    6: "MAV_FRAME_GLOBAL_RELATIVE_ALT_INT",
    10: "MAV_FRAME_GLOBAL_TERRAIN_ALT",
    11: "MAV_FRAME_GLOBAL_TERRAIN_ALT_INT",
}


def parse_mission_file(mission_path: Optional[Path]) -> Optional[dict]:
    if mission_path is None:
        return None

    summary = _empty_summary(mission_path)

    if not mission_path.exists():
        summary["warnings"].append(f"mission file does not exist: {mission_path}")
        return summary

    try:
        text = mission_path.read_text(encoding="utf-8-sig")
    except Exception as exc:
        summary["warnings"].append(f"failed to read mission file: {exc}")
        return summary

    stripped = text.lstrip()
    if not stripped:
        summary["warnings"].append("mission file is empty")
        return summary

    if stripped.startswith("{"):
        return _parse_plan_json(mission_path, text)

    if stripped.startswith("QGC WPL"):
        return _parse_qgc_wpl(mission_path, text)

    summary["warnings"].append("unsupported mission file format")
    return summary


def _empty_summary(mission_path: Path) -> dict:
    return {
        "mission_file": str(mission_path),
        "format": None,
        "version": None,
        "ground_station": None,
        "planned_home_position": None,
        "vehicle_type": None,
        "firmware_type": None,
        "cruise_speed": None,
        "hover_speed": None,
        "items": [],
        "warnings": [],
    }


def _parse_plan_json(mission_path: Path, text: str) -> dict:
    summary = _empty_summary(mission_path)
    summary["format"] = "qgroundcontrol_plan"

    try:
        plan = json.loads(text)
    except json.JSONDecodeError as exc:
        summary["warnings"].append(f"failed to parse JSON mission file: {exc}")
        return summary

    mission = plan.get("mission")
    if not isinstance(mission, dict):
        summary["warnings"].append("QGroundControl plan has no mission object")
        return summary

    summary.update({
        "version": _json_safe_value(plan.get("version") or mission.get("version")),
        "ground_station": _json_safe_value(plan.get("groundStation")),
        "planned_home_position": _json_safe_value(mission.get("plannedHomePosition")),
        "vehicle_type": _json_safe_value(mission.get("vehicleType")),
        "firmware_type": _json_safe_value(mission.get("firmwareType")),
        "cruise_speed": _json_safe_value(mission.get("cruiseSpeed")),
        "hover_speed": _json_safe_value(mission.get("hoverSpeed")),
    })

    items = mission.get("items", [])
    if not isinstance(items, list):
        summary["warnings"].append("QGroundControl mission items field is not a list")
        return summary

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            summary["warnings"].append(f"skipped non-object mission item at index {index}")
            continue

        summary["items"].append(_parse_plan_item(index, item))

    return summary


def _parse_plan_item(index: int, item: dict) -> dict:
    item_type = item.get("type")
    if item_type == "ComplexItem":
        return {
            "sequence": _json_safe_value(item.get("doJumpId", index)),
            "type": "ComplexItem",
            "complex_item_type": _json_safe_value(item.get("complexItemType")),
            "command": None,
            "command_name": None,
            "frame": None,
            "frame_name": None,
            "auto_continue": None,
            "params": [],
            "latitude": None,
            "longitude": None,
            "altitude": _json_safe_value(item.get("Altitude")),
        }

    params = _normalize_params(item.get("params", []))
    command = _safe_int(item.get("command"))
    frame = _safe_int(item.get("frame"))

    return {
        "sequence": _json_safe_value(item.get("doJumpId", index)),
        "type": _json_safe_value(item_type),
        "command": command,
        "command_name": _command_name(command),
        "frame": frame,
        "frame_name": _frame_name(frame),
        "auto_continue": _json_safe_value(item.get("autoContinue")),
        "params": params,
        "latitude": _param(params, 4),
        "longitude": _param(params, 5),
        "altitude": _param(params, 6, fallback=item.get("Altitude")),
    }


def _parse_qgc_wpl(mission_path: Path, text: str) -> dict:
    summary = _empty_summary(mission_path)
    summary["format"] = "qgc_wpl"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = lines[0]
    summary["version"] = header.removeprefix("QGC WPL").strip() or None

    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) != 12:
            summary["warnings"].append(
                f"skipped malformed QGC WPL line {line_number}: expected 12 fields"
            )
            continue

        try:
            sequence = int(fields[0])
            current = bool(int(fields[1]))
            frame = int(fields[2])
            command = int(fields[3])
            params = [_safe_number(value) for value in fields[4:11]]
            auto_continue = bool(int(fields[11]))
        except ValueError as exc:
            summary["warnings"].append(
                f"skipped malformed QGC WPL line {line_number}: {exc}"
            )
            continue

        summary["items"].append({
            "sequence": sequence,
            "current": current,
            "type": "SimpleItem",
            "command": command,
            "command_name": _command_name(command),
            "frame": frame,
            "frame_name": _frame_name(frame),
            "auto_continue": auto_continue,
            "params": params,
            "latitude": _param(params, 4),
            "longitude": _param(params, 5),
            "altitude": _param(params, 6),
        })

    return summary


def _normalize_params(params: Any) -> list[Any]:
    if not isinstance(params, list):
        return []

    return [_json_safe_value(value) for value in params]


def _param(params: list[Any], index: int, fallback: Any = None) -> Any:
    if index < len(params) and params[index] is not None:
        return params[index]

    return _json_safe_value(fallback)


def _command_name(command: Optional[int]) -> Optional[str]:
    if command is None:
        return None

    return COMMAND_NAMES.get(command, f"MAV_CMD_{command}")


def _frame_name(frame: Optional[int]) -> Optional[str]:
    if frame is None:
        return None

    return FRAME_NAMES.get(frame, f"MAV_FRAME_{frame}")


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_number(value: str) -> int | float:
    parsed = float(value)
    if parsed.is_integer():
        return int(parsed)

    return parsed


def _json_safe_value(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()

    return value
