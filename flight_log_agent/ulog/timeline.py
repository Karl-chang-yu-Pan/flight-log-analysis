from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pyulog import ULog


TIMELINE_FIELDS_BY_TOPIC = {
    "vehicle_status": [
        "arming_state",
        "nav_state",
        "hil_state",
        "failsafe",
        "failure_detector_status",
        "vehicle_type",
    ],
    "vehicle_type": [
        "fixed_wing",
        "rotary_wing",
        "vtol",
    ],
    "vtol_vehicle_status": [
        "vehicle_vtol_state",
        "fixed_wing_system_failure",
        "transition_failsafe",
    ],
    "mission_result": [
        "seq_current",
        "seq_reached",
        "mission_finished",
        "item_changed",
        "item_do_jump_changed",
        "warning",
        "failure",
    ],
}


def build_basic_timeline(log_path: Path) -> list[dict]:
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return [{
            "time_s": None,
            "event": "timeline_unavailable",
            "details": f"failed to parse ULog: {exc}",
        }]

    events = []

    for data in getattr(ulog, "data_list", []) or []:
        topic = getattr(data, "name", None)
        if topic not in TIMELINE_FIELDS_BY_TOPIC:
            continue

        topic_data = getattr(data, "data", {}) or {}
        timestamps = topic_data.get("timestamp")
        if timestamps is None:
            continue

        for field in TIMELINE_FIELDS_BY_TOPIC[topic]:
            values = topic_data.get(field)
            if values is None:
                continue

            events.extend(_field_change_events(topic, field, timestamps, values))

    return sorted(events, key=lambda event: _sort_key(event["time_s"]))


def _field_change_events(
    topic: str,
    field: str,
    timestamps: Iterable[Any],
    values: Iterable[Any],
) -> list[dict]:
    events = []
    unset = object()
    previous_value = unset

    for timestamp, value in zip(timestamps, values):
        value = _json_safe_value(value)
        if value == previous_value:
            continue

        events.append({
            "time_s": _timestamp_to_seconds(timestamp),
            "event": "initial_value" if previous_value is unset else "value_changed",
            "topic": topic,
            "field": field,
            "value": value,
        })
        previous_value = value

    return events


def _timestamp_to_seconds(timestamp: Any) -> float:
    return round(_json_safe_value(timestamp) / 1_000_000, 3)


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")

    if hasattr(value, "item"):
        return value.item()

    return value


def _sort_key(time_s: Any) -> tuple[int, float]:
    if time_s is None:
        return (1, 0.0)

    return (0, float(time_s))
