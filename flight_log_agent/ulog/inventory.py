from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pyulog import ULog
from pyulog.px4_events import PX4Events

from flight_log_agent.px4.source_snapshot import SourceHandle, SourceInput, SourceSnapshot, source_handle
from flight_log_agent.utils import json_safe_value as _json_safe_value
from flight_log_agent.utils import timestamp_to_seconds


EXPECTED_TIMELINE_TOPICS = [
    "vehicle_status",
    "vehicle_type",
    "vtol_vehicle_status",
    "mission_result",
]

def parse_ulog_inventory(
    log_path: Path,
    source_path: SourceInput = None,
) -> dict:
    inventory = _empty_inventory()
    source = source_handle(source_path)

    try:
        ulog = ULog(str(log_path), None, disable_str_exceptions=True)
    except Exception as exc:
        inventory["warnings"].append(f"failed to parse ULog: {exc}")
        inventory["missing_topics"] = EXPECTED_TIMELINE_TOPICS.copy()
        return inventory

    info = getattr(ulog, "msg_info_dict", {}) or {}
    available_topics = _extract_topic_names(ulog)
    git_hash = _json_safe_value(info.get("ver_sw"))

    inventory["firmware_version"] = _extract_firmware_version(ulog)
    inventory["firmware_branch"] = _json_safe_value(info.get("ver_sw_branch"))
    inventory["git_hash"] = git_hash
    inventory["airframe"] = _extract_airframe(ulog, source, git_hash)
    inventory["duration_s"] = _extract_duration_s(ulog)
    inventory["parameters"] = _extract_parameters(ulog)
    inventory["source_path"] = (
        str(source.repository_path)
        if isinstance(source, SourceSnapshot)
        else source.identity if source else None
    )
    if isinstance(source, SourceSnapshot):
        inventory["source_commit"] = source.commit_sha
    inventory["available_topics"] = available_topics
    inventory["topic_fields"] = _extract_topic_fields(ulog)
    inventory["topic_instances"] = _extract_topic_instances(ulog)
    inventory["logged_messages"] = _extract_logged_messages(ulog)
    inventory["warnings"] = _extract_logged_warnings(ulog, inventory["logged_messages"])
    inventory["missing_topics"] = [
        topic for topic in EXPECTED_TIMELINE_TOPICS if topic not in available_topics
    ]

    return inventory


def _empty_inventory() -> dict:
    return {
        "firmware_version": None,
        "firmware_branch": None,
        "git_hash": None,
        "airframe": None,
        "duration_s": None,
        "parameters": {},
        "source_path": None,
        "available_topics": [],
        "topic_fields": {},
        "topic_instances": {},
        "logged_messages": [],
        "warnings": [],
        "missing_topics": [],
    }


def observed_signals_from_inventory(inventory: dict[str, Any] | None) -> set[str]:
    """Return unambiguous signal references actually present in one ULog.

    ``topic_fields`` intentionally contains a few UI-derived fields and merges
    all multi instances, so it cannot be the canonical evidence inventory.
    ``topic_instances`` is lossless, so every reference retains its
    ``topic[multi_id].field`` identity. Consumers may resolve an unqualified
    source reference only when exactly one observed instance is compatible.

    The ``topic_fields`` fallback supports older serialized inventories that
    predate ``topic_instances``. It is deliberately used only when no instance
    inventory is available.
    """
    payload = inventory or {}
    topic_instances = payload.get("topic_instances") or {}
    observed: set[str] = set()
    if isinstance(topic_instances, dict) and topic_instances:
        for topic, raw_instances in topic_instances.items():
            if not isinstance(topic, str) or not topic:
                continue
            instances = [item for item in (raw_instances or []) if isinstance(item, dict)]
            if not instances:
                continue
            for instance in instances:
                try:
                    multi_id = int(instance.get("multi_id", 0) or 0)
                except (TypeError, ValueError):
                    continue
                prefix = f"{topic}[{multi_id}]"
                for field in instance.get("fields") or []:
                    if isinstance(field, str) and field and field != "timestamp":
                        observed.add(f"{prefix}.{field}")
        return observed

    for topic, fields in (payload.get("topic_fields") or {}).items():
        if not isinstance(topic, str) or not topic:
            continue
        for field in fields or []:
            if isinstance(field, str) and field and field != "timestamp":
                observed.add(f"{topic}.{field}")
    return observed




def _extract_topic_names(ulog: Any) -> list[str]:
    topics = set()

    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if name:
            topics.add(str(name))

    return sorted(topics)


def _extract_topic_fields(ulog: Any) -> dict[str, list[str]]:
    fields_by_topic: dict[str, set[str]] = {}

    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if not name:
            continue

        fields = fields_by_topic.setdefault(str(name), set())
        for field in getattr(data, "field_data", []) or []:
            field_name = getattr(field, "field_name", None)
            if field_name:
                fields.add(str(field_name))

        fields.update(str(field) for field in (getattr(data, "data", {}) or {}).keys())

    _add_derived_attitude_fields(fields_by_topic)

    return {
        topic: sorted(fields)
        for topic, fields in sorted(fields_by_topic.items())
    }


def _extract_topic_instances(ulog: Any) -> dict[str, list[dict[str, Any]]]:
    instances_by_topic: dict[str, list[dict[str, Any]]] = {}

    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if not name:
            continue

        fields = set()
        for field in getattr(data, "field_data", []) or []:
            field_name = getattr(field, "field_name", None)
            if field_name:
                fields.add(str(field_name))

        data_dict = getattr(data, "data", {}) or {}
        fields.update(str(field) for field in data_dict.keys())

        timestamp_values = data_dict.get("timestamp")
        sample_count = len(timestamp_values) if timestamp_values is not None else None

        instances_by_topic.setdefault(str(name), []).append(
            {
                "multi_id": _json_safe_value(getattr(data, "multi_id", 0)),
                "fields": sorted(fields),
                "sample_count": sample_count,
            }
        )

    return {
        topic: sorted(instances, key=lambda item: item["multi_id"])
        for topic, instances in sorted(instances_by_topic.items())
    }


def _add_derived_attitude_fields(fields_by_topic: dict[str, set[str]]) -> None:
    attitude_fields = fields_by_topic.get("vehicle_attitude")
    if attitude_fields and all(f"q[{index}]" in attitude_fields for index in range(4)):
        attitude_fields.update({"roll", "pitch", "yaw"})

    attitude_setpoint_fields = fields_by_topic.get("vehicle_attitude_setpoint")
    if attitude_setpoint_fields and all(
        f"q_d[{index}]" in attitude_setpoint_fields for index in range(4)
    ):
        attitude_setpoint_fields.update({"roll_d", "pitch_d", "yaw_d"})


def _extract_duration_s(ulog: Any) -> Optional[float]:
    start_timestamp = getattr(ulog, "start_timestamp", None)
    last_timestamp = getattr(ulog, "last_timestamp", None)

    if start_timestamp is not None and last_timestamp is not None:
        return round((last_timestamp - start_timestamp) / 1_000_000, 3)

    timestamps = []
    for data in getattr(ulog, "data_list", []) or []:
        data_dict = getattr(data, "data", {}) or {}
        topic_timestamps = data_dict.get("timestamp")
        if topic_timestamps is not None and len(topic_timestamps) > 0:
            timestamps.extend([topic_timestamps[0], topic_timestamps[-1]])

    if not timestamps:
        return None

    return round((max(timestamps) - min(timestamps)) / 1_000_000, 3)


def _extract_firmware_version(ulog: Any) -> Optional[str]:
    get_version_info_str = getattr(ulog, "get_version_info_str", None)
    if callable(get_version_info_str):
        try:
            version = get_version_info_str()
        except Exception:
            version = None
        if version:
            return str(version)

    get_version_info = getattr(ulog, "get_version_info", None)
    if not callable(get_version_info):
        return None

    try:
        version_info = get_version_info()
    except Exception:
        return None
    if not version_info or len(version_info) < 4:
        return None

    major, minor, patch, release_type = version_info[:4]
    suffix = _release_type_suffix(release_type)
    return f"v{major}.{minor}.{patch}{suffix}"


def _release_type_suffix(release_type: Any) -> str:
    try:
        release_type_int = int(release_type)
    except (TypeError, ValueError):
        return ""

    if release_type_int < 64:
        return " (dev)"
    if release_type_int < 128:
        return " (alpha)"
    if release_type_int < 192:
        return " (beta)"
    if release_type_int < 255:
        return " (RC)"
    return ""


def _extract_parameters(ulog: Any) -> dict:
    parameters = getattr(ulog, "initial_parameters", {}) or {}

    return {
        name: _json_safe_value(value)
        for name, value in sorted(parameters.items())
    }


def _extract_logged_messages(ulog: Any) -> list[dict[str, Any]]:
    messages = []

    for timestamp, level, text in _extract_logged_events(ulog):
        messages.append(
            {
                "timestamp": _json_safe_value(timestamp),
                "time_s": _timestamp_to_seconds(timestamp),
                "level": str(level),
                "message": str(text),
                "source": "event",
            }
        )

    for message in getattr(ulog, "logged_messages", []) or []:
        text = str(getattr(message, "message", "") or "")
        if text.endswith("\t"):
            continue

        timestamp = getattr(message, "timestamp", None)
        messages.append(
            {
                "timestamp": _json_safe_value(timestamp),
                "time_s": _timestamp_to_seconds(timestamp),
                "level": _logged_message_level(message),
                "message": text,
                "source": "logged_message",
            }
        )

    return sorted(
        messages,
        key=lambda item: (
            item["timestamp"] is None,
            item["timestamp"] if item["timestamp"] is not None else 0,
        ),
    )


def _extract_logged_warnings(ulog: Any, logged_messages: list[dict[str, Any]]) -> list[str]:
    warnings = []

    for message in logged_messages:
        level = str(message.get("level") or "")
        text = str(message.get("message") or "")

        if _is_warning_or_worse(level):
            warnings.append(f"{level.lower()}: {text}" if level else text)

    for dropout in getattr(ulog, "dropouts", []) or []:
        duration_ms = getattr(dropout, "duration", None)
        if duration_ms is not None:
            warnings.append(f"dropout: {duration_ms} ms")

    return warnings


def _extract_logged_events(ulog: Any) -> list[tuple[Any, str, str]]:
    try:
        parser = PX4Events()
        parser.set_default_json_definitions_cb(lambda already_has_default_parser: None)
        return parser.get_logged_events(ulog)
    except Exception:
        return []


def _logged_message_level(message: Any) -> str:
    level = getattr(message, "log_level_str", None)
    if callable(level):
        try:
            return str(level())
        except Exception:
            pass
    if level is not None:
        return str(level)

    raw_level = getattr(message, "log_level", None)
    if raw_level is not None:
        return {
            ord("0"): "EMERGENCY",
            ord("1"): "ALERT",
            ord("2"): "CRITICAL",
            ord("3"): "ERROR",
            ord("4"): "WARNING",
            ord("5"): "NOTICE",
            ord("6"): "INFO",
            ord("7"): "DEBUG",
        }.get(raw_level, "UNKNOWN")

    return "UNKNOWN"


def _is_warning_or_worse(level: str) -> bool:
    return level.upper() in {"EMERGENCY", "ALERT", "CRITICAL", "ERROR", "WARNING"}


def _timestamp_to_seconds(timestamp: Any) -> Optional[float]:
    if timestamp is None:
        return None
    try:
        return timestamp_to_seconds(timestamp)
    except (TypeError, ValueError):
        return None


def _extract_airframe(
    ulog: Any,
    source_path: Optional[SourceHandle],
    git_hash: Any,
) -> Optional[dict[str, Any]]:
    parameters = getattr(ulog, "initial_parameters", {}) or {}
    if "SYS_AUTOSTART" not in parameters:
        return None

    airframe_id = _json_safe_value(parameters["SYS_AUTOSTART"])
    airframe = {"id": airframe_id}

    metadata = _resolve_airframe_metadata(airframe_id, source_path, git_hash)
    if metadata:
        airframe.update(metadata)

    return airframe


def _resolve_airframe_metadata(
    airframe_id: Any,
    source_path: Optional[SourceHandle],
    git_hash: Any,
) -> Optional[dict[str, Any]]:
    if source_path is None:
        return None
    return _airframe_metadata_from_source(source_path, airframe_id)


def enrich_inventory_from_source(inventory: dict[str, Any], source_snapshot: SourceSnapshot) -> dict[str, Any]:
    inventory["source_path"] = str(source_snapshot.repository_path)
    inventory["source_commit"] = source_snapshot.commit_sha
    parameters = inventory.get("parameters") or {}
    airframe_id = parameters.get("SYS_AUTOSTART")
    if airframe_id is not None:
        airframe = {"id": _json_safe_value(airframe_id)}
        metadata = _airframe_metadata_from_source(source_snapshot, airframe_id)
        if metadata:
            airframe.update(metadata)
        inventory["airframe"] = airframe
    return inventory


def _airframe_metadata_from_snapshot(
    source_snapshot: SourceSnapshot,
    airframe_id: Any,
) -> Optional[dict[str, Any]]:
    return _airframe_metadata_from_source(source_snapshot, airframe_id)


def _airframe_metadata_from_source(
    source_snapshot: SourceHandle,
    airframe_id: Any,
) -> Optional[dict[str, Any]]:
    prefix = f"{airframe_id}_"
    for airframe_dir in _airframe_dirs():
        for file_path in source_snapshot.list_files(airframe_dir):
            if not Path(file_path).name.startswith(prefix):
                continue
            metadata = _parse_airframe_script_metadata(source_snapshot.read_text(file_path))
            metadata["source"] = source_snapshot.identity
            metadata["file"] = file_path
            return metadata
    return None


def _parse_airframe_script_metadata(text: str) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("# @"):
            continue

        tag, _, value = stripped[3:].partition(" ")
        tag = tag.strip().lower()
        value = value.strip()
        if tag in {"name", "type", "class"} and value:
            metadata[tag] = value
    return metadata


def _airframe_dirs() -> tuple[str, ...]:
    return (
        "ROMFS/px4fmu_common/init.d/airframes",
        "ROMFS/px4fmu_common/init.d-posix/airframes",
    )


def _clean_string(value: Any) -> Optional[str]:
    value = _json_safe_value(value)
    if value is None:
        return None
    text = str(value).strip().rstrip("\x00")
    return text or None
