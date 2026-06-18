from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Optional

from pyulog import ULog

from flight_log_agent.px4.source_snapshot import SourceRepository, SourceSnapshot
from flight_log_agent.ulog.control_surface import infer_control_surface
from flight_log_agent.ulog.inventory import enrich_inventory_from_source, parse_ulog_inventory
from flight_log_agent.ulog.timeline import build_basic_timeline
from flight_log_agent.mission.parser import parse_mission_file
from flight_log_agent.source_path import SOURCE_UNAVAILABLE, resolve_source_path
from flight_log_agent.utils import json_safe_value as _json_safe_value


FLOAT_TOLERANCE = 0.00001


def build_preparse_payload(
    log_path: str | Path,
    *,
    mission_path: str | Path | None = None,
    source_path: str | Path | None = None,
    parameters_xml_path: str | Path | None = None,
) -> dict[str, Any]:
    log_path_obj = Path(log_path)
    mission_path_obj = Path(mission_path) if mission_path else None
    source_path_obj = resolve_source_path(source_path)
    parameters_xml_path_obj = Path(parameters_xml_path) if parameters_xml_path else None

    inventory = parse_ulog_inventory(log_path_obj, SOURCE_UNAVAILABLE)
    logged_px4_git_hash = (
        inventory.get("git_hash")
        or inventory.get("px4_git_hash")
        or inventory.get("firmware_git_hash")
    )
    source_snapshot = None
    if source_path_obj is not None and logged_px4_git_hash:
        try:
            source_snapshot = SourceRepository(source_path_obj).resolve_snapshot(logged_px4_git_hash)
        except Exception as exc:
            status = getattr(exc, "status", "repository_unavailable")
            inventory.setdefault("warnings", []).append(
                f"Exact PX4 source is unavailable ({status}): {exc}"
            )
    if isinstance(source_snapshot, SourceSnapshot):
        enrich_inventory_from_source(inventory, source_snapshot)

    timeline = build_basic_timeline(log_path_obj)
    exact_source = source_snapshot or SOURCE_UNAVAILABLE
    assumptions = infer_control_surface(log_path_obj, exact_source)
    mission = parse_mission_file(mission_path_obj, source_path=exact_source)
    parameter_metadata = load_parameter_metadata(parameters_xml_path_obj)
    parameter_payload = build_parameter_payload(log_path_obj, parameter_metadata)

    return {
        "inputs": {
            "log_path": str(log_path_obj),
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": str(source_path_obj) if source_path_obj else None,
            "parameters_xml_path": (
                str(parameters_xml_path_obj) if parameters_xml_path_obj else None
            ),
        },
        "inventory": inventory,
        "timeline": timeline,
        "assumptions": assumptions,
        "mission": mission,
        "parameters": parameter_payload,
        "topics": build_topic_rows(inventory),
    }


def build_parameter_payload(
    log_path: Path,
    metadata: Optional[dict[str, dict[str, Any]]] = None,
) -> dict[str, Any]:
    metadata = metadata or {}

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return {
            "rows": [],
            "changed": [],
            "summary": {
                "total": 0,
                "default": 0,
                "non_default": 0,
                "unknown": 0,
                "has_embedded_defaults": False,
            },
            "warnings": [f"failed to parse ULog parameters: {exc}"],
        }

    rows = build_parameter_rows(ulog, metadata)
    changed = build_changed_parameter_rows(ulog)

    return {
        "rows": rows,
        "changed": changed,
        "summary": summarize_parameter_rows(
            rows,
            has_embedded_defaults=bool(getattr(ulog, "has_default_parameters", False)),
        ),
        "warnings": [],
    }


def build_parameter_rows(
    ulog: Any,
    metadata: Optional[dict[str, dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    metadata = metadata or {}
    initial_parameters = getattr(ulog, "initial_parameters", {}) or {}
    system_defaults, airframe_defaults = _embedded_defaults(ulog)
    changed_names = {
        str(name)
        for _, name, _ in (getattr(ulog, "changed_parameters", []) or [])
    }

    rows = []
    for name in sorted(initial_parameters):
        value = _json_safe_value(initial_parameters[name])
        param_metadata = metadata.get(name, {})
        default_value, default_source = _select_default(
            name,
            airframe_defaults,
            system_defaults,
            param_metadata,
        )
        default_status = _default_status(value, default_value, param_metadata)

        rows.append({
            "name": name,
            "value": _display_value(value, param_metadata),
            "raw_value": value,
            "default": _display_value(default_value, param_metadata),
            "raw_default": _json_safe_value(default_value),
            "default_source": default_source,
            "default_status": default_status,
            "min": param_metadata.get("min", ""),
            "max": param_metadata.get("max", ""),
            "description": param_metadata.get("short_desc", ""),
            "long_description": param_metadata.get("long_desc", ""),
            "group": param_metadata.get("group_name", ""),
            "type": param_metadata.get("type", _infer_type_label(value)),
            "is_rc_or_cal": name.startswith("RC") or name.startswith("CAL_"),
            "changed_during_log": name in changed_names,
        })

    return rows


def build_changed_parameter_rows(ulog: Any) -> list[dict[str, Any]]:
    rows = []
    for timestamp, name, value in (getattr(ulog, "changed_parameters", []) or []):
        rows.append({
            "time_s": round(_json_safe_value(timestamp) / 1_000_000, 3),
            "name": str(name),
            "value": _json_safe_value(value),
        })
    return rows


def summarize_parameter_rows(
    rows: list[dict[str, Any]],
    *,
    has_embedded_defaults: bool,
) -> dict[str, Any]:
    counts = {
        "total": len(rows),
        "default": 0,
        "non_default": 0,
        "unknown": 0,
        "has_embedded_defaults": has_embedded_defaults,
    }
    for row in rows:
        status = row.get("default_status")
        if status in {"default", "non_default", "unknown"}:
            counts[status] += 1
    return counts


def build_topic_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    available_topics = set(inventory.get("available_topics") or [])
    topic_fields = inventory.get("topic_fields") or {}
    missing_topics = set(inventory.get("missing_topics") or [])
    topic_names = sorted(available_topics | set(topic_fields) | missing_topics)

    rows = []
    for topic in topic_names:
        fields = topic_fields.get(topic, [])
        rows.append({
            "name": topic,
            "fields": fields,
            "field_count": len(fields),
            "missing_expected": topic in missing_topics and topic not in available_topics,
            "has_derived_attitude_fields": any(
                field in fields for field in ("roll", "pitch", "yaw", "roll_d", "pitch_d", "yaw_d")
            ),
        })
    return rows


def load_parameter_metadata(
    parameters_xml_path: Optional[Path],
) -> dict[str, dict[str, Any]]:
    if parameters_xml_path is None or not parameters_xml_path.is_file():
        return {}

    metadata: dict[str, dict[str, Any]] = {}
    try:
        root = ET.parse(parameters_xml_path).getroot()
    except Exception:
        return {}

    for group in root.findall("group"):
        group_name = group.get("name", "")
        for param in group.findall("parameter"):
            name = param.get("name")
            if not name:
                continue
            item = {
                "default": param.get("default"),
                "type": param.get("type"),
                "group_name": group_name,
            }
            for child_name in ("min", "max", "short_desc", "long_desc", "decimal"):
                child = param.find(child_name)
                if child is not None and child.text is not None:
                    item[child_name] = child.text
            metadata[name] = item

    return metadata


def _embedded_defaults(ulog: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    if not bool(getattr(ulog, "has_default_parameters", False)):
        return {}, {}

    return (
        getattr(ulog, "get_default_parameters")(0) or {},
        getattr(ulog, "get_default_parameters")(1) or {},
    )


def _select_default(
    name: str,
    airframe_defaults: dict[str, Any],
    system_defaults: dict[str, Any],
    metadata: dict[str, Any],
) -> tuple[Any, Optional[str]]:
    if name in airframe_defaults:
        return airframe_defaults[name], "ulog_airframe"
    if name in system_defaults:
        return system_defaults[name], "ulog_system"
    if "default" in metadata and metadata["default"] is not None:
        return metadata["default"], "metadata_xml"
    return None, None


def _default_status(value: Any, default_value: Any, metadata: dict[str, Any]) -> str:
    if default_value is None:
        return "unknown"
    return "default" if _values_equal(value, default_value, metadata) else "non_default"


def _values_equal(value: Any, default_value: Any, metadata: dict[str, Any]) -> bool:
    param_type = str(metadata.get("type") or "").upper()
    if param_type == "FLOAT" or _is_float_like(value) or _is_float_like(default_value):
        try:
            return abs(float(value) - float(default_value)) < FLOAT_TOLERANCE
        except (TypeError, ValueError):
            return str(value) == str(default_value)

    try:
        return int(value) == int(default_value)
    except (TypeError, ValueError):
        return str(value) == str(default_value)


def _display_value(value: Any, metadata: dict[str, Any]) -> Any:
    if value is None:
        return ""
    decimal = metadata.get("decimal")
    if decimal is not None:
        try:
            return round(float(value), int(decimal))
        except (TypeError, ValueError):
            return _json_safe_value(value)
    return _json_safe_value(value)


def _infer_type_label(value: Any) -> str:
    if isinstance(value, bool):
        return "BOOL"
    if isinstance(value, int):
        return "INT"
    if isinstance(value, float):
        return "FLOAT"
    return type(value).__name__.upper()


def _is_float_like(value: Any) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, str):
        return "." in value or "e" in value.lower()
    return False


