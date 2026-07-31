from __future__ import annotations

import math
import re
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from flight_log_agent.ulog.plots import (
    _json_safe_value,
    _normalize_signal_value,
    _signal_axis_label,
    _signal_unit,
    _timestamp_to_seconds,
)


NumericTransform = Callable[[float], float]


def dataset_instance(dataset: Any) -> int:
    try:
        return int(_json_safe_value(getattr(dataset, "multi_id", 0)) or 0)
    except (TypeError, ValueError):
        return 0


def iter_topic_datasets(ulog: Any, topic_name: str) -> list[Any]:
    datasets = [
        dataset
        for dataset in (getattr(ulog, "data_list", []) or [])
        if getattr(dataset, "name", None) == topic_name
    ]
    return sorted(datasets, key=dataset_instance)


def find_dataset(ulog: Any, topic_name: str, topic_instance: int = 0) -> Any:
    for dataset in iter_topic_datasets(ulog, topic_name):
        if dataset_instance(dataset) == topic_instance:
            return dataset
    return None


def indexed_fields(
    dataset: Any,
    prefix: str,
    *,
    suffix: str = "",
) -> list[tuple[int, str]]:
    data = getattr(dataset, "data", {}) or {}
    expression = re.compile(
        rf"^{re.escape(prefix)}\[(?P<index>\d+)\]{re.escape(suffix)}$"
    )
    matches = []
    for field_name in data:
        match = expression.fullmatch(str(field_name))
        if match:
            matches.append((int(match.group("index")), str(field_name)))
    return sorted(matches)


def field_type(dataset: Any, field_name: str) -> str | None:
    for field in getattr(dataset, "field_data", []) or []:
        if getattr(field, "field_name", None) == field_name:
            value = getattr(field, "type_str", None)
            return str(value) if value is not None else None
    return None


def boolean_fields(dataset: Any) -> list[str]:
    typed = [
        str(getattr(field, "field_name"))
        for field in (getattr(dataset, "field_data", []) or [])
        if getattr(field, "type_str", None) == "bool"
        and getattr(field, "field_name", None)
    ]
    if typed:
        return sorted(typed)

    data = getattr(dataset, "data", {}) or {}
    result = []
    for field_name, values in data.items():
        if field_name in {"timestamp", "timestamp_sample"}:
            continue
        array = _numeric_array(values)
        if array is None or array.size == 0:
            continue
        finite = array[np.isfinite(array)]
        if finite.size and set(np.unique(finite)).issubset({0.0, 1.0}):
            result.append(str(field_name))
    return sorted(result)


def finite_field_values(dataset: Any, field_name: str) -> np.ndarray | None:
    data = getattr(dataset, "data", {}) or {}
    return _numeric_array(data.get(field_name))


def field_has_finite_values(dataset: Any, field_name: str) -> bool:
    values = finite_field_values(dataset, field_name)
    return bool(values is not None and np.isfinite(values).any())


def field_has_nonzero_values(
    dataset: Any,
    field_name: str,
    *,
    threshold: float = 0.0,
) -> bool:
    values = finite_field_values(dataset, field_name)
    if values is None:
        return False
    finite = np.abs(values[np.isfinite(values)])
    return bool(finite.size and np.max(finite) > threshold)


def series_from_field(
    dataset: Any,
    field_name: str,
    *,
    key: str,
    label: str,
    color: str,
    start_s: float,
    end_s: float,
    max_points: int,
    timestamp_field: str = "timestamp",
    transform: NumericTransform | None = None,
    normalize: bool = True,
    interpolation: str = "linear",
    signal: str | None = None,
    unit: str | None = None,
    axis_label: str | None = None,
) -> dict[str, Any] | None:
    if dataset is None:
        return None
    data = getattr(dataset, "data", {}) or {}
    timestamps = data.get(timestamp_field)
    values = data.get(field_name)
    if timestamps is None or values is None:
        return None

    topic_name = str(getattr(dataset, "name", "") or "")
    times_s: list[float] = []
    numeric_values: list[float] = []
    for timestamp, raw_value in zip(timestamps, values):
        try:
            time_s = _timestamp_to_seconds(timestamp)
            value = float(_json_safe_value(raw_value))
        except (TypeError, ValueError):
            continue
        if time_s < start_s or time_s > end_s or not math.isfinite(value):
            continue
        if normalize:
            value = _normalize_signal_value(topic_name, field_name, value)
        if transform is not None:
            value = float(transform(value))
        if not math.isfinite(value):
            continue
        times_s.append(time_s)
        numeric_values.append(value)

    if not times_s:
        return None

    times_s, numeric_values = downsample_pair(
        times_s,
        numeric_values,
        max_points,
        preserve_steps=interpolation == "step_after",
    )
    resolved_signal = signal or _format_signal(dataset, field_name)
    resolved_unit = unit
    if resolved_unit is None and normalize:
        resolved_unit = _signal_unit(topic_name, field_name)
    resolved_axis_label = axis_label
    if resolved_axis_label is None and normalize:
        resolved_axis_label = _signal_axis_label(topic_name, field_name)

    result = {
        "key": key,
        "label": label,
        "signal": resolved_signal,
        "unit": resolved_unit,
        "axis_label": resolved_axis_label,
        "color": color,
        "time_s": times_s,
        "values": numeric_values,
    }
    if interpolation != "linear":
        result["interpolation"] = interpolation
    return result


def series_from_arrays(
    timestamps: Iterable[Any],
    values: Iterable[Any],
    *,
    key: str,
    label: str,
    signal: str,
    color: str,
    start_s: float,
    end_s: float,
    max_points: int,
    timestamps_are_seconds: bool = False,
    transform: NumericTransform | None = None,
    interpolation: str = "linear",
    unit: str | None = None,
    axis_label: str | None = None,
) -> dict[str, Any] | None:
    times_s: list[float] = []
    numeric_values: list[float] = []
    for raw_time, raw_value in zip(timestamps, values):
        try:
            time_s = (
                float(_json_safe_value(raw_time))
                if timestamps_are_seconds
                else _timestamp_to_seconds(raw_time)
            )
            value = float(_json_safe_value(raw_value))
        except (TypeError, ValueError):
            continue
        if time_s < start_s or time_s > end_s or not math.isfinite(value):
            continue
        if transform is not None:
            value = float(transform(value))
        if not math.isfinite(value):
            continue
        times_s.append(time_s)
        numeric_values.append(value)

    if not times_s:
        return None

    times_s, numeric_values = downsample_pair(
        times_s,
        numeric_values,
        max_points,
        preserve_steps=interpolation == "step_after",
    )
    result = {
        "key": key,
        "label": label,
        "signal": signal,
        "unit": unit,
        "axis_label": axis_label,
        "color": color,
        "time_s": times_s,
        "values": numeric_values,
    }
    if interpolation != "linear":
        result["interpolation"] = interpolation
    return result


def downsample_pair(
    times: Sequence[float],
    values: Sequence[float],
    max_points: int,
    *,
    preserve_steps: bool = False,
) -> tuple[list[float], list[float]]:
    count = min(len(times), len(values))
    if count == 0:
        return [], []
    limit = max(2, int(max_points))
    if count <= limit:
        indices = list(range(count))
    elif preserve_steps:
        indices = _step_preserving_indices(values[:count], limit)
    else:
        indices = _extrema_preserving_indices(values[:count], limit)
    return (
        [round(float(times[index]), 6) for index in indices],
        [round(float(values[index]), 6) for index in indices],
    )


def timeseries_plot(
    plot_id: str,
    title: str,
    series: list[dict[str, Any]],
    *,
    overlays: list[dict[str, Any]],
    time_range_s: Sequence[float],
    y_range: Sequence[float] | None = None,
    horizontal_bands: list[dict[str, Any]] | None = None,
    horizontal_spans: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    if not series:
        return None
    result: dict[str, Any] = {
        "id": plot_id,
        "title": title,
        "kind": "timeseries",
        "time_range_s": [round(float(time_range_s[0]), 6), round(float(time_range_s[1]), 6)],
        "y_range": list(y_range) if y_range is not None else None,
        "series": series,
        "overlays": overlays,
        "warnings": [],
    }
    if horizontal_bands:
        result["horizontal_bands"] = horizontal_bands
    if horizontal_spans:
        result["horizontal_spans"] = horizontal_spans
    return result


def _format_signal(dataset: Any, field_name: str) -> str:
    topic_name = str(getattr(dataset, "name", "") or "")
    instance = dataset_instance(dataset)
    if instance:
        return f"{topic_name}[{instance}].{field_name}"
    return f"{topic_name}.{field_name}"


def _numeric_array(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        return np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None


def _step_preserving_indices(values: Sequence[float], max_points: int) -> list[int]:
    count = len(values)
    important = {0, count - 1}
    for index in range(1, count):
        if values[index] != values[index - 1]:
            important.update({index - 1, index})
    return _bounded_indices(sorted(important), count, max_points)


def _extrema_preserving_indices(values: Sequence[float], max_points: int) -> list[int]:
    count = len(values)
    if max_points <= 2:
        return [0, count - 1]
    bucket_count = max(1, (max_points - 2) // 2)
    boundaries = np.linspace(1, count - 1, bucket_count + 1, dtype=int)
    selected = {0, count - 1}
    numeric = np.asarray(values, dtype=float)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop <= start:
            continue
        bucket = numeric[start:stop]
        finite_positions = np.flatnonzero(np.isfinite(bucket))
        if finite_positions.size == 0:
            continue
        finite_values = bucket[finite_positions]
        selected.add(start + int(finite_positions[int(np.argmin(finite_values))]))
        selected.add(start + int(finite_positions[int(np.argmax(finite_values))]))
    return _bounded_indices(sorted(selected), count, max_points)


def _bounded_indices(
    preferred: Sequence[int],
    count: int,
    max_points: int,
) -> list[int]:
    selected = sorted({int(index) for index in preferred if 0 <= int(index) < count})
    if len(selected) > max_points:
        positions = np.linspace(0, len(selected) - 1, max_points, dtype=int)
        selected = [selected[int(position)] for position in positions]
    elif len(selected) < max_points:
        for index in np.linspace(0, count - 1, max_points, dtype=int):
            selected.append(int(index))
            if len(set(selected)) >= max_points:
                break
    return sorted(set(selected))
