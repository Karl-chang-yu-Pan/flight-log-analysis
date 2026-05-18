from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from pyulog import ULog


def evaluate_log_signature(
    log_path: Path,
    mechanism_title: str,
    expected_signature: list[dict],
    candidate_windows: list[dict],
    required_signals: list[str],
    derived_signals: list[dict],
    events: list[dict],
    supporting_checks: list[dict],
    exclusion_checks: list[dict],
    numeric_checks: list[dict],
) -> dict:
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return {
            "mechanism_title": mechanism_title,
            "expected_signature": expected_signature,
            "required_signals": {"present": [], "missing": list(required_signals)},
            "missing_required_signals": list(required_signals),
            "window_results": [],
            "check_results": [],
            "evidence": [],
            "contradictions": [],
            "warnings": [f"failed to parse ULog: {exc}"],
            "verdict": "unresolved",
            "confidence_ceiling": "unresolved",
            "summary": "Signature evaluator could not inspect the log.",
        }

    topics = _topics_by_name(ulog)
    windows = _normalize_windows(candidate_windows)
    event_index = _normalize_events(events)
    present, missing = _required_signal_status(topics, required_signals)
    warnings = _derived_signal_warnings(derived_signals)

    check_results = []
    for category, checks in (
        ("supporting", supporting_checks),
        ("numeric", numeric_checks),
        ("exclusion", exclusion_checks),
    ):
        for check in checks or []:
            check_results.append(_run_check(topics, windows, event_index, check, category))

    evidence = [
        result["message"]
        for result in check_results
        if result["status"] == "passed"
    ]
    contradictions = [
        result["message"]
        for result in check_results
        if result["status"] == "failed"
    ]
    unresolved = [
        result["message"]
        for result in check_results
        if result["status"] == "unresolved"
    ]

    for signal in missing:
        unresolved.append(f"required signal is missing: {signal}")

    verdict, ceiling = _verdict(evidence, contradictions, unresolved, missing)
    return {
        "mechanism_title": mechanism_title,
        "expected_signature": expected_signature,
        "required_signals": {
            "present": present,
            "missing": missing,
        },
        "missing_required_signals": missing,
        "window_results": list(windows.values()),
        "event_results": list(event_index.values()),
        "check_results": check_results,
        "evidence": evidence,
        "contradictions": contradictions,
        "unresolved": unresolved,
        "warnings": warnings,
        "verdict": verdict,
        "confidence_ceiling": ceiling,
        "summary": (
            f"Signature verdict is {verdict}: {len(evidence)} supporting checks, "
            f"{len(contradictions)} contradicting checks, {len(unresolved)} unresolved checks."
        ),
    }


def _run_check(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    check_type = str(check.get("type") or "").strip()
    handlers = {
        "compare": _check_compare,
        "tracking_error": _check_tracking_error,
        "setpoint_actual_separation": _check_setpoint_actual_separation,
        "before_after_delta": _check_before_after_delta,
        "threshold_fraction": _check_threshold_fraction,
        "saturation": _check_saturation,
        "event_alignment": _check_event_alignment,
        "rate_of_change": _check_rate_of_change,
        "correlation": _check_correlation,
        "lagged_correlation": _check_correlation,
        "monotonic_change": _check_monotonic_change,
        "missing_signal": _check_missing_signal,
    }
    handler = handlers.get(check_type)
    if handler is None:
        return _check_result(
            check,
            category,
            "unresolved",
            f"unsupported {category} check type: {check_type or 'missing'}",
        )

    try:
        return handler(topics, windows, events, check, category)
    except Exception as exc:
        return _check_result(
            check,
            category,
            "unresolved",
            f"{check_type} check could not be evaluated: {exc}",
        )


def _check_compare(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    left_signal = str(check.get("left") or check.get("signal") or "")
    left = _metric_value(topics, windows, check, left_signal)
    if "result" in left:
        return left["result"]

    right_signal = check.get("right")
    if right_signal:
        right = _metric_value(topics, windows, check, str(right_signal))
        if "result" in right:
            return right["result"]
        expected = right["value"]
    else:
        expected = _safe_float(check.get("value"))

    if expected is None:
        return _check_result(check, category, "unresolved", "compare value is missing")

    op = str(check.get("op") or ">=").strip()
    passed = _compare(left["value"], op, expected, check)
    return _checked(
        check,
        category,
        passed,
        f"{left_signal} {left['metric']} {left['value']} {op} {expected}",
        {
            "left": left_signal,
            "metric": left["metric"],
            "actual": left["value"],
            "op": op,
            "expected": expected,
        },
    )


def _check_tracking_error(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    error_result = _aligned_errors(topics, windows, check)
    if "result" in error_result:
        return error_result["result"]

    max_error = _safe_float(check.get("max_error"))
    if max_error is None:
        return _check_result(check, category, "unresolved", "max_error is missing")

    max_abs_error = max(abs(error) for error in error_result["errors"])
    passed = max_abs_error <= max_error
    return _checked(
        check,
        category,
        passed,
        f"max tracking error {max_abs_error} <= {max_error}",
        {"max_abs_error": _round_float(max_abs_error), "max_error": max_error},
    )


def _check_setpoint_actual_separation(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    error_result = _aligned_errors(topics, windows, check)
    if "result" in error_result:
        return error_result["result"]

    threshold = _safe_float(check.get("min_delta"))
    if threshold is None:
        threshold = _safe_float(check.get("value"))
    if threshold is None:
        return _check_result(check, category, "unresolved", "separation threshold is missing")

    max_abs_error = max(abs(error) for error in error_result["errors"])
    passed = max_abs_error >= threshold
    return _checked(
        check,
        category,
        passed,
        f"max setpoint/actual separation {max_abs_error} >= {threshold}",
        {"max_abs_error": _round_float(max_abs_error), "threshold": threshold},
    )


def _check_before_after_delta(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or check.get("left") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    samples = [(t, float(v)) for t, v in samples_result["samples"] if _is_number(v)]
    if len(samples) < 2:
        return _check_result(check, category, "unresolved", f"not enough numeric samples for {signal}")

    split_time = _event_time(events, check.get("event"))
    if split_time is None:
        window = samples_result["window"]
        split_time = (window["start_s"] + window["end_s"]) / 2.0

    before = [value for time_s, value in samples if time_s <= split_time]
    after = [value for time_s, value in samples if time_s > split_time]
    if not before or not after:
        return _check_result(check, category, "unresolved", f"cannot split {signal} before/after event")

    delta = mean(after) - mean(before)
    threshold = _safe_float(check.get("min_delta"))
    if threshold is None:
        threshold = _safe_float(check.get("value")) or 0.0
    op = str(check.get("op") or ">=").strip()
    passed = _compare(delta, op, threshold, check)
    return _checked(
        check,
        category,
        passed,
        f"{signal} before/after delta {delta} {op} {threshold}",
        {"delta": _round_float(delta), "op": op, "threshold": threshold},
    )


def _check_threshold_fraction(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [float(value) for _, value in samples_result["samples"] if _is_number(value)]
    threshold = _safe_float(check.get("value"))
    if threshold is None:
        return _check_result(check, category, "unresolved", "threshold_fraction value is missing")
    if not values:
        return _check_result(check, category, "unresolved", f"no numeric samples for {signal}")

    op = str(check.get("op") or ">=").strip()
    fraction = sum(1 for value in values if _compare(value, op, threshold, check)) / len(values)
    required_fraction = _safe_float(check.get("lower"))
    if required_fraction is None:
        required_fraction = 0.5
    passed = fraction >= required_fraction
    return _checked(
        check,
        category,
        passed,
        f"{signal} fraction {fraction} satisfying {op} {threshold} >= {required_fraction}",
        {"fraction": _round_float(fraction), "required_fraction": required_fraction},
    )


def _check_saturation(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [float(value) for _, value in samples_result["samples"] if _is_number(value)]
    if not values:
        return _check_result(check, category, "unresolved", f"no numeric samples for {signal}")

    lower = _safe_float(check.get("lower"))
    upper = _safe_float(check.get("upper"))
    if lower is None and upper is None:
        return _check_result(check, category, "unresolved", "saturation lower or upper bound is missing")

    saturated = []
    for value in values:
        saturated.append((lower is not None and value <= lower) or (upper is not None and value >= upper))

    fraction = sum(1 for value in saturated if value) / len(saturated)
    required_fraction = _safe_float(check.get("value")) or 0.1
    passed = fraction >= required_fraction
    return _checked(
        check,
        category,
        passed,
        f"{signal} saturation fraction {fraction} >= {required_fraction}",
        {"fraction": _round_float(fraction), "required_fraction": required_fraction},
    )


def _check_event_alignment(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, dict],
    check: dict,
    category: str,
) -> dict:
    event_name = check.get("event")
    event_time = _event_time(events, event_name)
    if event_time is None:
        return _check_result(check, category, "unresolved", f"unknown event: {event_name}")

    window = _window_for_check(windows, check)
    if window is None:
        return _check_result(check, category, "unresolved", f"unknown window: {check.get('window')}")

    tolerance = _safe_float(check.get("value")) or 0.0
    passed = window["start_s"] - tolerance <= event_time <= window["end_s"] + tolerance
    return _checked(
        check,
        category,
        passed,
        f"event {event_name} at {event_time} aligns with window {window['name']}",
        {"event_time_s": event_time, "window": window["name"], "tolerance_s": tolerance},
    )


def _check_rate_of_change(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, Any],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or check.get("left") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    samples = [(time_s, float(value)) for time_s, value in samples_result["samples"] if _is_number(value)]
    if len(samples) < 2:
        return _check_result(check, category, "unresolved", f"not enough numeric samples for {signal}")

    duration = samples[-1][0] - samples[0][0]
    if duration <= 0:
        return _check_result(check, category, "unresolved", f"invalid duration for {signal}")

    rate = (samples[-1][1] - samples[0][1]) / duration
    threshold = _safe_float(check.get("value"))
    if threshold is None:
        return _check_result(check, category, "unresolved", "rate_of_change value is missing")
    op = str(check.get("op") or ">=").strip()
    passed = _compare(rate, op, threshold, check)
    return _checked(
        check,
        category,
        passed,
        f"{signal} rate {rate} {op} {threshold}",
        {"rate": _round_float(rate), "op": op, "threshold": threshold},
    )


def _check_correlation(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, Any],
    check: dict,
    category: str,
) -> dict:
    left_signal = str(check.get("left") or check.get("actual") or "")
    right_signal = str(check.get("right") or check.get("setpoint") or "")
    aligned = _aligned_pairs(topics, windows, check, left_signal, right_signal)
    if "result" in aligned:
        return aligned["result"]

    left = aligned["left"]
    right = aligned["right"]
    if len(left) < 2:
        return _check_result(check, category, "unresolved", "not enough aligned samples for correlation")

    corr = _pearson(left, right)
    threshold = _safe_float(check.get("value"))
    if threshold is None:
        threshold = 0.5
    op = str(check.get("op") or ">=").strip()
    passed = _compare(corr, op, threshold, check)
    return _checked(
        check,
        category,
        passed,
        f"correlation {left_signal} vs {right_signal} {corr} {op} {threshold}",
        {"correlation": _round_float(corr), "op": op, "threshold": threshold},
    )


def _check_monotonic_change(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, Any],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or check.get("left") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [float(value) for _, value in samples_result["samples"] if _is_number(value)]
    if len(values) < 2:
        return _check_result(check, category, "unresolved", f"not enough numeric samples for {signal}")

    delta = values[-1] - values[0]
    min_delta = _safe_float(check.get("min_delta")) or _safe_float(check.get("value")) or 0.0
    direction = str(check.get("direction") or "increase")
    passed = delta >= min_delta if direction != "decrease" else delta <= -min_delta
    return _checked(
        check,
        category,
        passed,
        f"{signal} delta {delta} direction {direction}",
        {"delta": _round_float(delta), "direction": direction, "min_delta": min_delta},
    )


def _check_missing_signal(
    topics: dict[str, Any],
    windows: dict[str, dict],
    events: dict[str, Any],
    check: dict,
    category: str,
) -> dict:
    signal = str(check.get("signal") or "")
    parsed = _parse_signal(signal)
    missing = parsed is None
    if parsed is not None:
        topic_name, field_name = parsed
        topic = topics.get(topic_name)
        data = getattr(topic, "data", {}) or {} if topic is not None else {}
        missing = topic is None or field_name not in data

    return _checked(
        check,
        category,
        missing,
        f"signal missing: {signal}",
        {"signal": signal, "missing": missing},
    )


def _metric_value(topics: dict[str, Any], windows: dict[str, dict], check: dict, signal: str) -> dict:
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result

    metric = str(check.get("metric") or "mean")
    metrics = _sample_metrics(samples_result["samples"])
    if metric not in metrics:
        return {"result": _check_result(check, "numeric", "unresolved", f"unsupported metric '{metric}' for {signal}")}
    return {"metric": metric, "value": metrics[metric]}


def _aligned_errors(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    actual = str(check.get("actual") or check.get("left") or "")
    setpoint = str(check.get("setpoint") or check.get("right") or "")
    aligned = _aligned_pairs(topics, windows, check, actual, setpoint)
    if "result" in aligned:
        return aligned
    return {"errors": [actual_value - setpoint_value for actual_value, setpoint_value in zip(aligned["left"], aligned["right"])]}


def _aligned_pairs(
    topics: dict[str, Any],
    windows: dict[str, dict],
    check: dict,
    left_signal: str,
    right_signal: str,
) -> dict:
    left_result = _check_samples(topics, windows, check, left_signal)
    if "result" in left_result:
        return left_result
    right_result = _check_samples(topics, windows, check, right_signal)
    if "result" in right_result:
        return right_result

    right_times = [time_s for time_s, value in right_result["samples"] if _is_number(value)]
    right_values = [float(value) for _, value in right_result["samples"] if _is_number(value)]
    if not right_times:
        return {"result": _check_result(check, "numeric", "unresolved", f"no numeric samples for {right_signal}")}

    left = []
    right = []
    for time_s, value in left_result["samples"]:
        if not _is_number(value) or time_s < right_times[0] or time_s > right_times[-1]:
            continue
        interpolated = _interpolated_value(right_times, right_values, time_s)
        if interpolated is not None:
            left.append(float(value))
            right.append(interpolated)

    if not left:
        return {"result": _check_result(check, "numeric", "unresolved", f"no overlapping samples for {left_signal} and {right_signal}")}
    return {"left": left, "right": right}


def _check_samples(topics: dict[str, Any], windows: dict[str, dict], check: dict, signal: str) -> dict:
    if not signal:
        return {"result": _check_result(check, "numeric", "unresolved", "signal is missing")}

    parsed = _parse_signal(signal)
    if parsed is None:
        return {"result": _check_result(check, "numeric", "unresolved", f"invalid signal: {signal}")}

    window = _window_for_check(windows, check)
    if window is None:
        return {"result": _check_result(check, "numeric", "unresolved", f"unknown window: {check.get('window')}")}

    topic_name, field_name = parsed
    topic = topics.get(topic_name)
    if topic is None:
        return {"result": _check_result(check, "numeric", "unresolved", f"missing topic for signal: {signal}")}

    data = getattr(topic, "data", {}) or {}
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return {"result": _check_result(check, "numeric", "unresolved", f"missing field for signal: {signal}")}

    samples = _window_samples(timestamps, values, window["start_s"], window["end_s"])
    if not samples:
        return {"result": _check_result(check, "numeric", "unresolved", f"no samples for {signal} in window {window['name']}")}
    return {"samples": samples, "window": window}


def _topics_by_name(ulog: Any) -> dict[str, Any]:
    topics = {}
    for data in getattr(ulog, "data_list", []) or []:
        name = getattr(data, "name", None)
        if name and name not in topics:
            topics[str(name)] = data
    return topics


def _normalize_windows(candidate_windows: list[dict]) -> dict[str, dict]:
    windows = {}
    for index, window in enumerate(candidate_windows or []):
        start_s = _safe_float(window.get("start_s"))
        end_s = _safe_float(window.get("end_s"))
        if start_s is None or end_s is None or end_s < start_s:
            continue
        name = str(window.get("name") or f"window_{index + 1}")
        windows[name] = {"name": name, "start_s": start_s, "end_s": end_s}
    return windows


def _normalize_events(events: list[dict]) -> dict[str, dict]:
    normalized = {}
    for event in events or []:
        name = str(event.get("name") or "")
        if not name:
            continue
        normalized[name] = {key: value for key, value in event.items() if value is not None}
    return normalized


def _derived_signal_warnings(derived_signals: list[dict]) -> list[str]:
    if not derived_signals:
        return []
    return [
        "Derived signal declarations were recorded, but this evaluator currently supports direct logged signals only."
    ]


def _required_signal_status(topics: dict[str, Any], required_signals: list[str]) -> tuple[list[str], list[str]]:
    present = []
    missing = []
    for signal in required_signals:
        parsed = _parse_signal(signal)
        if parsed is None:
            missing.append(signal)
            continue
        topic_name, field_name = parsed
        topic = topics.get(topic_name)
        data = getattr(topic, "data", {}) or {} if topic is not None else {}
        if topic is not None and field_name in data and "timestamp" in data:
            present.append(signal)
        else:
            missing.append(signal)
    return present, missing


def _window_for_check(windows: dict[str, dict], check: dict) -> dict | None:
    if not windows:
        return None
    window_name = check.get("window")
    if window_name:
        return windows.get(str(window_name))
    return next(iter(windows.values()))


def _event_time(events: dict[str, dict], event_name: Any) -> float | None:
    if not event_name:
        return None
    event = events.get(str(event_name))
    if event is None:
        return None
    return _safe_float(event.get("time_s"))


def _parse_signal(signal: str) -> tuple[str, str] | None:
    if "." not in signal:
        return None
    topic, field = signal.split(".", 1)
    if not topic or not field:
        return None
    return topic, field


def _window_samples(timestamps: Any, values: Any, start_s: float, end_s: float) -> list[tuple[float, Any]]:
    samples = []
    for timestamp, value in zip(timestamps, values):
        time_s = _timestamp_to_seconds(timestamp)
        if start_s <= time_s <= end_s:
            samples.append((time_s, _json_safe_value(value)))
    return samples


def _sample_metrics(samples: list[tuple[float, Any]]) -> dict[str, Any]:
    values = [float(value) for _, value in samples if _is_number(value)]
    if not values:
        return {"count": len(samples)}
    return {
        "count": len(values),
        "min": _round_float(min(values)),
        "max": _round_float(max(values)),
        "mean": _round_float(mean(values)),
        "median": _round_float(median(values)),
        "std": _round_float(pstdev(values)) if len(values) > 1 else 0.0,
        "start": _round_float(values[0]),
        "end": _round_float(values[-1]),
        "delta": _round_float(values[-1] - values[0]),
    }


def _compare(actual: float, op: str, expected: float, check: dict) -> bool:
    if op == ">":
        return actual > expected
    if op == ">=":
        return actual >= expected
    if op == "<":
        return actual < expected
    if op == "<=":
        return actual <= expected
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    if op == "between":
        lower = _safe_float(check.get("lower"))
        upper = _safe_float(check.get("upper"))
        return lower is not None and upper is not None and lower <= actual <= upper
    if op == "outside":
        lower = _safe_float(check.get("lower"))
        upper = _safe_float(check.get("upper"))
        return lower is not None and upper is not None and not lower <= actual <= upper
    raise ValueError(f"unsupported operator: {op}")


def _checked(check: dict, category: str, passed: bool, fallback: str, value: Any) -> dict:
    return _check_result(
        check,
        category,
        "passed" if passed else "failed",
        _message(check, passed, fallback),
        value,
    )


def _check_result(
    check: dict,
    category: str,
    status: str,
    message: str,
    value: Any = None,
) -> dict:
    result = {
        "type": check.get("type"),
        "category": category,
        "window": check.get("window"),
        "status": status,
        "message": message,
    }
    if value is not None:
        result["value"] = value
    return result


def _message(check: dict, passed: bool, fallback: str) -> str:
    key = "supports" if passed else "contradicts"
    return str(check.get(key) or check.get("description") or fallback)


def _verdict(
    evidence: list[str],
    contradictions: list[str],
    unresolved: list[str],
    missing: list[str],
) -> tuple[str, str]:
    if contradictions and evidence:
        verdict = "mixed"
    elif contradictions:
        verdict = "contradicted"
    elif evidence and not missing:
        verdict = "supported"
    else:
        verdict = "unresolved"

    if missing:
        return verdict, "low"
    if verdict == "supported":
        return verdict, "high" if not unresolved else "medium"
    if verdict == "mixed":
        return verdict, "medium"
    if verdict == "contradicted":
        return verdict, "low"
    return verdict, "unresolved"


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_den = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_den = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_den == 0 or right_den == 0:
        return 0.0
    return numerator / (left_den * right_den)


def _interpolated_value(times: list[float], values: list[float], target: float) -> float | None:
    if not times:
        return None
    if target <= times[0]:
        return values[0]
    if target >= times[-1]:
        return values[-1]
    for index in range(1, len(times)):
        if times[index] < target:
            continue
        before_t = times[index - 1]
        after_t = times[index]
        before_v = values[index - 1]
        after_v = values[index]
        if after_t == before_t:
            return after_v
        ratio = (target - before_t) / (after_t - before_t)
        return before_v + ratio * (after_v - before_v)
    return None


def _timestamp_to_seconds(timestamp: Any) -> float:
    return float(_json_safe_value(timestamp)) / 1_000_000


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").rstrip("\x00")
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _is_number(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
