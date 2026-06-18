from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from pyulog import ULog

from flight_log_agent.utils import json_safe_value as _json_safe_value
from flight_log_agent.utils import timestamp_to_seconds


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
    return build_signal_timeline(
        log_path,
        [
            f"{topic}.{field}"
            for topic, fields in TIMELINE_FIELDS_BY_TOPIC.items()
            for field in fields
        ],
    )


def build_signal_timeline(log_path: Path, signals: Iterable[str]) -> list[dict]:
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return [{
            "time_s": None,
            "event": "timeline_unavailable",
            "details": f"failed to parse ULog: {exc}",
        }]

    fields_by_topic: dict[str, list[str]] = {}
    for signal in signals:
        topic, separator, field = str(signal or "").partition(".")
        if separator and topic and field:
            fields = fields_by_topic.setdefault(topic, [])
            if field not in fields:
                fields.append(field)

    events = []
    for data in getattr(ulog, "data_list", []) or []:
        topic = getattr(data, "name", None)
        if topic not in fields_by_topic:
            continue

        topic_data = getattr(data, "data", {}) or {}
        timestamps = topic_data.get("timestamp")
        if timestamps is None:
            continue

        for field in fields_by_topic[topic]:
            values = topic_data.get(field)
            if values is None:
                continue

            events.extend(_field_change_events(topic, field, timestamps, values))

    return sorted(events, key=lambda event: _sort_key(event["time_s"]))


def merge_timeline_events(*timelines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    events: dict[tuple[Any, ...], dict[str, Any]] = {}
    for timeline in timelines:
        for event in timeline or []:
            if not isinstance(event, dict):
                continue
            key = (
                event.get("time_s"),
                event.get("event"),
                event.get("topic"),
                event.get("field"),
                repr(event.get("value")),
                event.get("details"),
            )
            events[key] = event
    return sorted(events.values(), key=lambda event: _sort_key(event.get("time_s")))


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
    return round(timestamp_to_seconds(timestamp), 3)


def _sort_key(time_s: Any) -> tuple[int, float]:
    if time_s is None:
        return (1, 0.0)

    return (0, float(time_s))
