from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from pyulog import ULog


EXPECTED_TIMELINE_TOPICS = [
    "vehicle_status",
    "vehicle_type",
    "vtol_vehicle_status",
    "mission_result",
]

IMPORTANT_PARAMETER_PREFIXES = (
    "SYS_",
    "COM_",
    "NAV_",
    "FW_",
    "MC_",
    "MPC_",
    "VT_",
    "CA_",
    "PWM_",
    "UAVCAN_",
    "CBRK_",
    "EKF2_",
    "SENS_",
)


def parse_ulog_inventory(log_path: Path) -> dict:
    inventory = _empty_inventory()

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        inventory["warnings"].append(f"failed to parse ULog: {exc}")
        inventory["missing_topics"] = EXPECTED_TIMELINE_TOPICS.copy()
        return inventory

    info = getattr(ulog, "msg_info_dict", {}) or {}
    available_topics = _extract_topic_names(ulog)

    inventory["firmware_version"] = _json_safe_value(
        info.get("ver_sw_release") or info.get("sys_name")
    )
    inventory["git_hash"] = _json_safe_value(info.get("ver_sw"))
    inventory["duration_s"] = _extract_duration_s(ulog)
    inventory["important_parameters"] = _extract_important_parameters(ulog)
    inventory["available_topics"] = available_topics
    inventory["warnings"] = _extract_logged_warnings(ulog)
    inventory["missing_topics"] = [
        topic for topic in EXPECTED_TIMELINE_TOPICS if topic not in available_topics
    ]

    return inventory


def _empty_inventory() -> dict:
    return {
        "firmware_version": None,
        "git_hash": None,
        "duration_s": None,
        "important_parameters": {},
        "available_topics": [],
        "warnings": [],
        "missing_topics": [],
    }


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")

    if hasattr(value, "item"):
        return value.item()

    return value


def _extract_topic_names(ulog: Any) -> list[str]:
    topics = set()

    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if name:
            topics.add(str(name))

    return sorted(topics)


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


def _extract_important_parameters(ulog: Any) -> dict:
    parameters = getattr(ulog, "initial_parameters", {}) or {}

    return {
        name: _json_safe_value(value)
        for name, value in sorted(parameters.items())
        if name.startswith(IMPORTANT_PARAMETER_PREFIXES)
    }


def _extract_logged_warnings(ulog: Any) -> list[str]:
    warnings = []

    for message in getattr(ulog, "logged_messages", []) or []:
        level = str(getattr(message, "log_level_str", "") or "").lower()
        text = str(getattr(message, "message", "") or "")

        if level in {"warning", "warn", "error", "critical"}:
            warnings.append(f"{level}: {text}" if level else text)

    for dropout in getattr(ulog, "dropouts", []) or []:
        duration_ms = getattr(dropout, "duration", None)
        if duration_ms is not None:
            warnings.append(f"dropout: {duration_ms} ms")

    return warnings
