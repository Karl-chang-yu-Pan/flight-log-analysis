from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional


FRAME_NAMES = {
    0: "MAV_FRAME_GLOBAL",
    3: "MAV_FRAME_GLOBAL_RELATIVE_ALT",
    6: "MAV_FRAME_GLOBAL_RELATIVE_ALT_INT",
    10: "MAV_FRAME_GLOBAL_TERRAIN_ALT",
    11: "MAV_FRAME_GLOBAL_TERRAIN_ALT_INT",
}


def parse_mission_file(
    mission_path: Optional[Path],
    source_path: Optional[Path] = None,
) -> Optional[dict]:
    if mission_path is None:
        return None

    summary = _empty_summary(mission_path)
    command_names = load_mavlink_command_names(source_path)

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
        return _parse_plan_json(mission_path, text, command_names)

    if stripped.startswith("QGC WPL"):
        return _parse_qgc_wpl(mission_path, text, command_names)

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


def _parse_plan_json(mission_path: Path, text: str, command_names: dict[int, str]) -> dict:
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

        summary["items"].append(_parse_plan_item(index, item, command_names))

    return summary


def _parse_plan_item(index: int, item: dict, command_names: dict[int, str]) -> dict:
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
        "command_name": _command_name(command, command_names),
        "frame": frame,
        "frame_name": _frame_name(frame),
        "auto_continue": _json_safe_value(item.get("autoContinue")),
        "params": params,
        "latitude": _param(params, 4),
        "longitude": _param(params, 5),
        "altitude": _param(params, 6, fallback=item.get("Altitude")),
    }


def _parse_qgc_wpl(mission_path: Path, text: str, command_names: dict[int, str]) -> dict:
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
            "command_name": _command_name(command, command_names),
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


def _command_name(command: Optional[int], command_names: dict[int, str]) -> Optional[str]:
    if command is None:
        return None

    return command_names.get(command, f"MAV_CMD_{command}")


def load_mavlink_command_names(source_path: Optional[Path]) -> dict[int, str]:
    if source_path is None:
        return {}

    root = Path(source_path)
    if not root.exists():
        return {}

    for path in _candidate_mavlink_xml_paths(root):
        names = _load_mavlink_command_names_from_xml(path)
        if names:
            return names

    for path in _candidate_mavlink_header_paths(root):
        names = _load_mavlink_command_names_from_header(path)
        if names:
            return names

    return {}


def _candidate_mavlink_xml_paths(root: Path) -> list[Path]:
    candidates = [
        root / "mavlink" / "message_definitions" / "v1.0" / "common.xml",
        root / "src" / "modules" / "mavlink" / "mavlink" / "message_definitions" / "v1.0" / "common.xml",
        root / "src" / "modules" / "mavlink" / "mavlink" / "message_definitions" / "v1.0" / "minimal.xml",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing
    return list(root.glob("**/message_definitions/v1.0/common.xml"))[:8]


def _candidate_mavlink_header_paths(root: Path) -> list[Path]:
    candidates = [
        root / "build" / "px4_sitl_default" / "mavlink" / "common" / "mavlink.h",
        root / "mavlink" / "include" / "mavlink" / "v2.0" / "common" / "mavlink.h",
    ]
    existing = [path for path in candidates if path.exists()]
    if existing:
        return existing
    return list(root.glob("**/mavlink/v*/common/mavlink.h"))[:8]


def _load_mavlink_command_names_from_xml(path: Path) -> dict[int, str]:
    try:
        tree = ET.parse(path)
    except Exception:
        return {}

    names: dict[int, str] = {}
    for enum in tree.findall(".//enum"):
        if enum.attrib.get("name") != "MAV_CMD":
            continue
        for entry in enum.findall("entry"):
            value = _safe_int(entry.attrib.get("value"))
            name = entry.attrib.get("name")
            if value is not None and name:
                names[value] = name
    return names


def _load_mavlink_command_names_from_header(path: Path) -> dict[int, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return {}

    match = re.search(r"typedef\s+enum\s+MAV_CMD\s*\{(?P<body>.*?)\}\s*MAV_CMD\s*;", text, re.S)
    if not match:
        return {}

    names: dict[int, str] = {}
    for item in match.group("body").split(","):
        entry = re.search(r"\b(?P<name>MAV_CMD_[A-Z0-9_]+)\s*=\s*(?P<value>\d+)\b", item)
        if entry:
            names[int(entry.group("value"))] = entry.group("name")
    return names


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
