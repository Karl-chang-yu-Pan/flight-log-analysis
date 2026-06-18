from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from pyulog import ULog

from flight_log_agent.symbols import parse_simple_signal as _parse_signal
from flight_log_agent.utils import (
    is_number as _is_number,
    json_safe_value as _json_safe_value,
    round_float as _round_float,
    timestamp_to_seconds as _timestamp_to_seconds,
)


DISCRETE_UNIQUE_LIMIT = 20

UNIT_HINTS = {
    "lat": "degE7",
    "lon": "degE7",
    "alt": "m",
    "alt_ellipsoid": "m",
    "dist_bottom": "m",
    "vx": "m/s",
    "vy": "m/s",
    "vz": "m/s",
    "roll": "rad",
    "pitch": "rad",
    "yaw": "rad",
    "rollspeed": "rad/s",
    "pitchspeed": "rad/s",
    "yawspeed": "rad/s",
    "airspeed": "m/s",
    "groundspeed": "m/s",
    "seq_current": "index",
    "seq_reached": "index",
}


def compute_log_metrics(log_path: Path, start_s: float, end_s: float, signals: list[str]) -> dict:
    if end_s < start_s:
        return {
            "window_s": [start_s, end_s],
            "signals": {},
            "missing_signals": list(signals),
            "warnings": ["end_s must be greater than or equal to start_s."],
        }

    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return {
            "window_s": [start_s, end_s],
            "signals": {},
            "missing_signals": list(signals),
            "warnings": [f"failed to parse ULog: {exc}"],
        }

    topics = _topics_by_name(ulog)
    metrics = {}
    missing_signals = []
    warnings = []

    for signal in signals:
        signal_result = _compute_signal_metrics(topics, signal, start_s, end_s)
        if "metrics" in signal_result:
            metrics[signal] = signal_result["metrics"]
        else:
            missing_signals.append(signal)
            warnings.append(signal_result["warning"])

    return {
        "window_s": [start_s, end_s],
        "signals": metrics,
        "missing_signals": missing_signals,
        "warnings": warnings,
    }


def _topics_by_name(ulog: Any) -> dict[str, Any]:
    topics = {}

    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if name and name not in topics:
            topics[str(name)] = data

    return topics


def _compute_signal_metrics(
    topics: dict[str, Any],
    signal: str,
    start_s: float,
    end_s: float,
) -> dict:
    parsed_signal = _parse_signal(signal)
    if parsed_signal is None:
        return {"warning": f"invalid signal '{signal}'; expected 'topic.field'."}

    topic_name, field_name = parsed_signal
    topic = topics.get(topic_name)
    if topic is None:
        return {"warning": f"missing topic for signal '{signal}': {topic_name}"}

    topic_data = getattr(topic, "data", {}) or {}
    timestamps = topic_data.get("timestamp")
    if timestamps is None:
        return {"warning": f"missing timestamp field for topic '{topic_name}'."}

    values = topic_data.get(field_name)
    if values is None:
        return {"warning": f"missing field for signal '{signal}': {field_name}"}

    samples = _window_samples(timestamps, values, start_s, end_s)
    if not samples:
        return {"warning": f"no samples for signal '{signal}' in window {start_s}-{end_s}s."}

    return {"metrics": _summarize_samples(field_name, samples)}


def _window_samples(
    timestamps: Any,
    values: Any,
    start_s: float,
    end_s: float,
) -> list[tuple[float, Any]]:
    samples = []

    for timestamp, value in zip(timestamps, values):
        time_s = _timestamp_to_seconds(timestamp)
        if start_s <= time_s <= end_s:
            samples.append((time_s, _json_safe_value(value)))

    return samples


def _summarize_samples(field_name: str, samples: list[tuple[float, Any]]) -> dict:
    times = [time_s for time_s, _ in samples]
    values = [value for _, value in samples]
    summary = {
        "count": len(samples),
        "first_time_s": _round_float(times[0]),
        "last_time_s": _round_float(times[-1]),
        "sample_rate_hz": _sample_rate_hz(times),
        "start": values[0],
        "end": values[-1],
        "unit": _unit_hint(field_name),
    }

    if _is_continuous_numeric(values):
        numeric_values = [float(value) for value in values]
        summary.update({
            "min": _round_float(min(numeric_values)),
            "max": _round_float(max(numeric_values)),
            "mean": _round_float(mean(numeric_values)),
            "median": _round_float(median(numeric_values)),
            "std": _round_float(pstdev(numeric_values)) if len(numeric_values) > 1 else 0.0,
            "delta": _round_float(numeric_values[-1] - numeric_values[0]),
        })
    else:
        summary.update({
            "value_counts": _value_counts(values),
            "transitions": _transitions(samples),
        })

    return summary


def _is_continuous_numeric(values: list[Any]) -> bool:
    if not all(_is_number(value) for value in values):
        return False

    if all(isinstance(value, bool) for value in values):
        return False

    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return len(set(values)) > DISCRETE_UNIQUE_LIMIT

    return True


def _value_counts(values: list[Any]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items(), key=lambda item: str(item[0]))
    }


def _transitions(samples: list[tuple[float, Any]]) -> list[dict]:
    transitions = []
    previous = object()

    for time_s, value in samples:
        if value == previous:
            continue

        transitions.append({
            "time_s": _round_float(time_s),
            "value": value,
        })
        previous = value

    return transitions


def _sample_rate_hz(times: list[float]) -> float | None:
    if len(times) < 2:
        return None

    duration_s = times[-1] - times[0]
    if duration_s <= 0:
        return None

    return _round_float((len(times) - 1) / duration_s)


def _unit_hint(field_name: str) -> str | None:
    normalized = field_name.split("[", 1)[0]
    return UNIT_HINTS.get(normalized)
