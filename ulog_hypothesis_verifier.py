from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from pyulog import ULog


NUMERIC_METRICS = {"min", "max", "mean", "median", "std", "start", "end", "delta", "count"}


def verify_hypothesis_against_log(
    log_path: Path,
    mechanism: str,
    expected_signature: dict,
    candidate_windows: list[dict],
    required_signals: list[str],
    exclusion_checks: list[dict],
    numeric_checks: list[dict],
) -> dict:
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return {
            "mechanism": mechanism,
            "expected_signature": expected_signature,
            "required_signals": {"present": [], "missing": list(required_signals)},
            "window_results": [],
            "exclusion_checks": [],
            "numeric_checks": [],
            "evidence": [],
            "contradicting_evidence": [],
            "unresolved": [f"failed to parse ULog: {exc}"],
            "confidence": "low",
            "confidence_score": 0.0,
            "summary": "Verifier could not inspect the log.",
        }

    topics = _topics_by_name(ulog)
    windows = _normalize_windows(candidate_windows)
    present, missing = _required_signal_status(topics, required_signals)
    window_results = [
        {
            "name": window["name"],
            "start_s": window["start_s"],
            "end_s": window["end_s"],
        }
        for window in windows.values()
    ]

    numeric_results = [
        _run_check(topics, windows, check, category="numeric")
        for check in numeric_checks
    ]
    exclusion_results = [
        _run_check(topics, windows, check, category="exclusion")
        for check in exclusion_checks
    ]

    evidence = []
    contradictions = []
    unresolved = []
    for result in [*numeric_results, *exclusion_results]:
        if result["status"] == "passed":
            evidence.append(result["message"])
        elif result["status"] == "failed":
            contradictions.append(result["message"])
        else:
            unresolved.append(result["message"])

    for signal in missing:
        unresolved.append(f"required signal is missing: {signal}")

    score, confidence = _confidence(numeric_results, exclusion_results, missing)
    summary = _summary(confidence, evidence, contradictions, unresolved)

    return {
        "mechanism": mechanism,
        "expected_signature": expected_signature,
        "required_signals": {
            "present": present,
            "missing": missing,
        },
        "window_results": window_results,
        "exclusion_checks": exclusion_results,
        "numeric_checks": numeric_results,
        "evidence": evidence,
        "contradicting_evidence": contradictions,
        "unresolved": unresolved,
        "confidence": confidence,
        "confidence_score": score,
        "summary": summary,
    }


def _run_check(
    topics: dict[str, Any],
    windows: dict[str, dict],
    check: dict,
    *,
    category: str,
) -> dict:
    check_type = str(check.get("type") or "").strip()
    handlers = {
        "threshold": _check_threshold,
        "transition_occurs": _check_transition_occurs,
        "no_transition": _check_no_transition,
        "state_equals": _check_state_equals,
        "state_not_equals": _check_state_not_equals,
        "tracks_setpoint": _check_tracks_setpoint,
        "diverges_from_setpoint": _check_diverges_from_setpoint,
        "monotonic_change": _check_monotonic_change,
        "same_direction_change": _check_same_direction_change,
    }
    handler = handlers.get(check_type)
    if handler is None:
        return _check_result(
            check,
            status="unresolved",
            message=f"unsupported {category} check type: {check_type or 'missing'}",
        )

    try:
        return handler(topics, windows, check)
    except Exception as exc:
        return _check_result(
            check,
            status="unresolved",
            message=f"{check_type} check could not be evaluated: {exc}",
        )


def _check_threshold(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    samples = samples_result["samples"]
    metric_name = str(check.get("metric") or "mean")
    metrics = _sample_metrics(samples)
    if metric_name not in metrics:
        return _check_result(
            check,
            status="unresolved",
            message=f"unsupported threshold metric '{metric_name}' for {signal}",
        )

    actual = metrics[metric_name]
    expected = _safe_float(check.get("value"))
    op = str(check.get("op") or ">=").strip()
    if expected is None:
        return _check_result(check, status="unresolved", message="threshold value is missing")

    passed = _compare(actual, op, expected)
    message = _message(
        check,
        passed,
        f"{signal} {metric_name} {actual} {op} {expected}",
    )
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"metric": metric_name, "actual": actual, "op": op, "expected": expected},
    )


def _check_transition_occurs(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    transitions = _transitions(samples_result["samples"])
    expected_from = check.get("from")
    expected_to = check.get("to")
    passed = any(
        (expected_from is None or transition.get("from") == expected_from)
        and (expected_to is None or transition.get("to") == expected_to)
        for transition in transitions
    )
    message = _message(check, passed, f"{signal} transitions: {transitions}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"transitions": transitions},
    )


def _check_no_transition(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    transitions = _transitions(samples_result["samples"])
    passed = not transitions
    message = _message(check, passed, f"{signal} transitions: {transitions}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"transitions": transitions},
    )


def _check_state_equals(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    target = check.get("value")
    values = [value for _, value in samples_result["samples"]]
    mode = str(check.get("mode") or "any")
    passed = all(value == target for value in values) if mode == "all" else any(value == target for value in values)
    message = _message(check, passed, f"{signal} values include {dict(Counter(values))}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"target": target, "mode": mode, "counts": dict(Counter(values))},
    )


def _check_state_not_equals(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    target = check.get("value")
    values = [value for _, value in samples_result["samples"]]
    passed = all(value != target for value in values)
    message = _message(check, passed, f"{signal} values include {dict(Counter(values))}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"target": target, "counts": dict(Counter(values))},
    )


def _check_tracks_setpoint(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    error_result = _aligned_error(topics, windows, check)
    if "result" in error_result:
        return error_result["result"]

    max_error = _safe_float(check.get("max_error"))
    if max_error is None:
        return _check_result(check, status="unresolved", message="max_error is missing")

    abs_errors = [abs(error) for error in error_result["errors"]]
    actual_max = max(abs_errors)
    passed = actual_max <= max_error
    message = _message(check, passed, f"max absolute tracking error {actual_max} <= {max_error}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"max_abs_error": _round_float(actual_max), "max_error": max_error},
    )


def _check_diverges_from_setpoint(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    error_result = _aligned_error(topics, windows, check)
    if "result" in error_result:
        return error_result["result"]

    min_error = _safe_float(check.get("min_error"))
    if min_error is None:
        return _check_result(check, status="unresolved", message="min_error is missing")

    direction = str(check.get("direction") or "absolute")
    errors = error_result["errors"]
    if direction == "below":
        relevant_errors = [-error for error in errors]
    elif direction == "above":
        relevant_errors = errors
    else:
        relevant_errors = [abs(error) for error in errors]

    actual_max = max(relevant_errors)
    passed = actual_max >= min_error
    message = _message(check, passed, f"setpoint divergence {actual_max} >= {min_error}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={
            "direction": direction,
            "max_relevant_error": _round_float(actual_max),
            "min_error": min_error,
        },
    )


def _check_monotonic_change(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [float(value) for _, value in samples_result["samples"] if _is_number(value)]
    if len(values) < 2:
        return _check_result(check, status="unresolved", message=f"not enough numeric samples for {signal}")

    direction = str(check.get("direction") or "increase")
    min_delta = _safe_float(check.get("min_delta")) or 0.0
    delta = values[-1] - values[0]
    passed = delta >= min_delta if direction == "increase" else delta <= -min_delta
    message = _message(check, passed, f"{signal} delta {delta} direction {direction}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"delta": _round_float(delta), "direction": direction, "min_delta": min_delta},
    )


def _check_same_direction_change(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    first = str(check.get("first") or "")
    second = str(check.get("second") or "")
    first_result = _check_samples(topics, windows, check, first)
    if "result" in first_result:
        return first_result["result"]
    second_result = _check_samples(topics, windows, check, second)
    if "result" in second_result:
        return second_result["result"]

    first_delta = _numeric_delta(first_result["samples"])
    second_delta = _numeric_delta(second_result["samples"])
    min_delta = _safe_float(check.get("min_delta")) or 0.0
    if first_delta is None or second_delta is None:
        return _check_result(check, status="unresolved", message="same_direction_change needs numeric samples")

    passed = abs(first_delta) >= min_delta and abs(second_delta) >= min_delta and first_delta * second_delta > 0
    message = _message(check, passed, f"{first} delta {first_delta}; {second} delta {second_delta}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={
            "first_delta": _round_float(first_delta),
            "second_delta": _round_float(second_delta),
            "min_delta": min_delta,
        },
    )


def _aligned_error(topics: dict[str, Any], windows: dict[str, dict], check: dict) -> dict:
    actual = str(check.get("actual") or "")
    setpoint = str(check.get("setpoint") or "")
    actual_result = _check_samples(topics, windows, check, actual)
    if "result" in actual_result:
        return actual_result
    setpoint_result = _check_samples(topics, windows, check, setpoint)
    if "result" in setpoint_result:
        return setpoint_result

    setpoint_times = [time_s for time_s, value in setpoint_result["samples"] if _is_number(value)]
    setpoint_values = [float(value) for _, value in setpoint_result["samples"] if _is_number(value)]
    if not setpoint_times:
        return {"result": _check_result(check, status="unresolved", message=f"no numeric setpoint samples for {setpoint}")}

    errors = []
    for time_s, value in actual_result["samples"]:
        if not _is_number(value) or time_s < setpoint_times[0] or time_s > setpoint_times[-1]:
            continue
        setpoint_value = _interpolated_value(setpoint_times, setpoint_values, time_s)
        if setpoint_value is not None:
            errors.append(float(value) - setpoint_value)

    if not errors:
        return {"result": _check_result(check, status="unresolved", message=f"no overlapping samples for {actual} and {setpoint}")}

    return {"errors": errors}


def _check_samples(
    topics: dict[str, Any],
    windows: dict[str, dict],
    check: dict,
    signal: str,
) -> dict:
    if not signal:
        return {"result": _check_result(check, status="unresolved", message="signal is missing")}

    window = _window_for_check(windows, check)
    if window is None:
        return {"result": _check_result(check, status="unresolved", message=f"unknown window: {check.get('window')}")}

    parsed = _parse_signal(signal)
    if parsed is None:
        return {"result": _check_result(check, status="unresolved", message=f"invalid signal: {signal}")}

    topic_name, field_name = parsed
    topic = topics.get(topic_name)
    if topic is None:
        return {"result": _check_result(check, status="unresolved", message=f"missing topic for signal: {signal}")}

    data = getattr(topic, "data", {}) or {}
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return {"result": _check_result(check, status="unresolved", message=f"missing field for signal: {signal}")}

    samples = _window_samples(timestamps, values, window["start_s"], window["end_s"])
    if not samples:
        return {"result": _check_result(check, status="unresolved", message=f"no samples for {signal} in window {window['name']}")}

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


def _transitions(samples: list[tuple[float, Any]]) -> list[dict]:
    transitions = []
    if not samples:
        return transitions
    previous = samples[0][1]
    for time_s, value in samples[1:]:
        if value == previous:
            continue
        transitions.append({"time_s": _round_float(time_s), "from": previous, "to": value})
        previous = value
    return transitions


def _numeric_delta(samples: list[tuple[float, Any]]) -> float | None:
    values = [float(value) for _, value in samples if _is_number(value)]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _compare(actual: float, op: str, expected: float) -> bool:
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
    raise ValueError(f"unsupported operator: {op}")


def _message(check: dict, passed: bool, fallback: str) -> str:
    key = "supports" if passed else "contradicts"
    return str(check.get(key) or check.get("description") or fallback)


def _check_result(
    check: dict,
    *,
    status: str,
    message: str,
    value: Any = None,
) -> dict:
    result = {
        "type": check.get("type"),
        "window": check.get("window"),
        "status": status,
        "message": message,
    }
    if value is not None:
        result["value"] = value
    return result


def _confidence(
    numeric_results: list[dict],
    exclusion_results: list[dict],
    missing_signals: list[str],
) -> tuple[float, str]:
    results = [*numeric_results, *exclusion_results]
    passed = sum(1 for result in results if result["status"] == "passed")
    failed = sum(1 for result in results if result["status"] == "failed")
    unresolved = sum(1 for result in results if result["status"] == "unresolved")
    resolved = passed + failed
    if resolved == 0:
        return 0.0, "low"

    score = passed / resolved
    if missing_signals:
        score *= 0.7
    if unresolved:
        score *= max(0.5, 1.0 - unresolved * 0.1)
    score = _round_float(score)

    if score >= 0.8 and not missing_signals:
        return score, "high"
    if score >= 0.6:
        return score, "medium"
    if score >= 0.4:
        return score, "medium-low"
    return score, "low"


def _summary(confidence: str, evidence: list[str], contradictions: list[str], unresolved: list[str]) -> str:
    return (
        f"Log-evidence confidence is {confidence}: "
        f"{len(evidence)} supporting checks, "
        f"{len(contradictions)} contradicting checks, "
        f"{len(unresolved)} unresolved checks."
    )


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


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not (
        isinstance(value, float) and math.isnan(value)
    )


def _round_float(value: float) -> float:
    return round(float(value), 6)


def _interpolated_value(times: list[float], values: list[float], target_time: float) -> float | None:
    if not times:
        return None
    if target_time <= times[0]:
        return values[0]
    if target_time >= times[-1]:
        return values[-1]
    for index in range(1, len(times)):
        if times[index] < target_time:
            continue
        t0 = times[index - 1]
        t1 = times[index]
        v0 = values[index - 1]
        v1 = values[index]
        if t1 == t0:
            return v0
        ratio = (target_time - t0) / (t1 - t0)
        return v0 + ratio * (v1 - v0)
    return None
