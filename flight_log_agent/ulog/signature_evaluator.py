from __future__ import annotations

import ast
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from pyulog import ULog

from flight_log_agent.analysis.helper_resolution import HelperRegistry
from flight_log_agent.analysis.parameter_lookup import get_parameter as _get_parameter
from flight_log_agent.symbols import parse_simple_signal as _parse_signal
from flight_log_agent.utils import (
    compare as _compare,
    is_number as _is_number,
    json_safe_value as _json_safe_value,
    round_float as _round_float,
    safe_float as _safe_float,
    safe_float as _number,
    timestamp_to_seconds as _timestamp_to_seconds,
)
from flight_log_agent.analysis.source_expression import (
    alias_dotted_names,
    normalize_source_expression,
    source_expression_names,
)
from flight_log_agent.analysis.verdict import ceiling_for, verdict_from_counts
from flight_log_agent.expression_math import SAFE_MATH_FUNCTIONS, normalize_expression_function_names
from flight_log_agent.px4.msg_schema import field_or_flattened_prefix_present, normalize_px4_enum_value
from flight_log_agent.utils import dedupe_keep_order


NUMERIC_METRICS = {"min", "max", "mean", "median", "std", "start", "end", "delta", "count"}


def evaluate_log_signature(
    log_path: Path,
    mechanism: str,
    expected_signature: list[dict],
    candidate_windows: list[dict],
    required_signals: list[str],
    exclusion_checks: list[dict],
    numeric_checks: list[dict],
    *,
    source_path: Path | None = None,
    helper_expressions: Iterable[dict[str, Any]] = (),
) -> dict:
    helper_registry = HelperRegistry(list(helper_expressions))
    try:
        ulog = ULog(str(log_path))
    except Exception as exc:
        return _evaluation_result(
            mechanism=mechanism,
            expected_signature=expected_signature,
            present=[],
            missing=list(required_signals),
            window_results=[],
            numeric_results=[],
            exclusion_results=[],
            evidence=[],
            contradictions=[],
            unresolved=[f"failed to parse ULog: {exc}"],
            warnings=[f"failed to parse ULog: {exc}"],
        )

    topics = _topics_by_name(ulog)
    parameters = getattr(ulog, "initial_parameters", {}) or {}
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
        _run_check(topics, windows, parameters, check, category="numeric", source_path=source_path, helper_registry=helper_registry)
        for check in numeric_checks
    ]
    exclusion_results = [
        _run_check(topics, windows, parameters, check, category="exclusion", source_path=source_path, helper_registry=helper_registry)
        for check in exclusion_checks
    ]

    evidence = []
    contradictions = []
    unresolved = []
    for result in [*numeric_results, *exclusion_results]:
        if result["status"] == "passed":
            if result.get("claim_effect") == "contradicts":
                contradictions.append(result["message"])
            else:
                evidence.append(result["message"])
        elif result["status"] == "failed":
            if result.get("claim_effect") == "supports":
                evidence.append(result["message"])
            else:
                contradictions.append(result["message"])
        else:
            unresolved.append(result["message"])

    for signal in missing:
        unresolved.append(f"required signal is missing: {signal}")

    return _evaluation_result(
        mechanism=mechanism,
        expected_signature=expected_signature,
        present=present,
        missing=missing,
        window_results=window_results,
        numeric_results=numeric_results,
        exclusion_results=exclusion_results,
        evidence=evidence,
        contradictions=contradictions,
        unresolved=unresolved,
        warnings=[],
    )


def _run_check(
    topics: dict[str, Any],
    windows: dict[str, dict],
    parameters: dict[str, Any],
    check: dict,
    *,
    category: str,
    source_path: Path | None = None,
    helper_registry: HelperRegistry | None = None,
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
        "parameter_equals": _check_parameter_equals,
        "branch_parameter_satisfied": _check_branch_parameter_satisfied,
        "tracks_parameter_value": _check_tracks_parameter_value,
        "topic_field_present": _check_topic_field_present,
        "derived_expression": _check_derived_expression,
    }
    handler = handlers.get(check_type)
    if handler is None:
        return _check_result(
            check,
            status="unresolved",
            message=f"unsupported {category} check type: {check_type or 'missing'}",
        )

    try:
        check = {**check, "_source_path": source_path, "_helper_registry": helper_registry}
        result = handler(topics, windows, parameters, check)
    except Exception as exc:
        result = _check_result(
            check,
            status="unresolved",
            message=f"{check_type} check could not be evaluated: {exc}",
        )
    result["category"] = category
    claim_effect = _claim_effect(check, str(result.get("status") or ""))
    if claim_effect is not None:
        result["claim_effect"] = claim_effect
    return result


def _check_threshold(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
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

    op = str(check.get("op") or ">=").strip()
    actual = metrics[metric_name]
    expected = _safe_float(check.get("value"))
    if op in ("between", "outside"):
        lower = _safe_float(check.get("lower"))
        upper = _safe_float(check.get("upper"))
        if lower is None or upper is None:
            return _check_result(check, status="unresolved", message=f"{op} threshold requires lower and upper")
        passed = lower <= actual <= upper
        if op == "outside":
            passed = not passed
        expected_label = f"{lower}..{upper}"
    else:
        if expected is None:
            return _check_result(check, status="unresolved", message="threshold value is missing")
        passed = _compare(actual, op, expected)
        expected_label = str(expected)
    message = _message(
        check,
        passed,
        f"{signal} {metric_name} {actual} {op} {expected_label}",
    )
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"metric": metric_name, "actual": actual, "op": op, "expected": expected_label},
    )


def _check_transition_occurs(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    normalized_samples = _normalize_state_samples(signal, samples_result["samples"], check.get("_source_path"))
    transitions = _transitions(normalized_samples)
    expected_from = check.get("from")
    if expected_from is None:
        expected_from = check.get("from_value")
    expected_to = check.get("to")
    if expected_to is None:
        expected_to = check.get("to_value")
    expected_from = _normalize_state_value(signal, expected_from, check.get("_source_path"))
    expected_to = _normalize_state_value(signal, expected_to, check.get("_source_path"))
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


def _check_no_transition(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    transitions = _transitions(_normalize_state_samples(signal, samples_result["samples"], check.get("_source_path")))
    passed = not transitions
    message = _message(check, passed, f"{signal} transitions: {transitions}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"transitions": transitions},
    )


def _check_state_equals(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    target = _normalize_state_value(signal, check.get("value"), check.get("_source_path"))
    values = [value for _, value in _normalize_state_samples(signal, samples_result["samples"], check.get("_source_path"))]
    mode = str(check.get("mode") or "any")
    passed = all(value == target for value in values) if mode == "all" else any(value == target for value in values)
    message = _message(check, passed, f"{signal} values include {dict(Counter(values))}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"target": target, "mode": mode, "counts": dict(Counter(values))},
    )


def _check_state_not_equals(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    target = _normalize_state_value(signal, check.get("value"), check.get("_source_path"))
    values = [value for _, value in _normalize_state_samples(signal, samples_result["samples"], check.get("_source_path"))]
    passed = all(value != target for value in values)
    message = _message(check, passed, f"{signal} values include {dict(Counter(values))}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"target": target, "counts": dict(Counter(values))},
    )


def _check_tracks_setpoint(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
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


def _check_diverges_from_setpoint(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
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


def _check_monotonic_change(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [_number(value) for _, value in samples_result["samples"]]
    values = [value for value in values if value is not None]
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


def _check_same_direction_change(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
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


def _check_parameter_equals(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    parameter = str(check.get("parameter") or "")
    if not parameter:
        return _check_result(check, status="unresolved", message="parameter is missing")
    raw_actual, missing_reason = _get_parameter(parameters, parameter)
    if missing_reason:
        return _check_result(check, status="unresolved", message=missing_reason)

    op = str(check.get("op") or "==").strip()
    expected = check.get("value")
    if expected is None:
        return _check_result(check, status="unresolved", message=f"expected value is missing for {parameter}")

    actual = _json_safe_value(raw_actual)
    tolerance = _safe_float(check.get("max_error"))
    passed = _compare_literal(actual, op, expected, tolerance=tolerance)
    message = _message(check, passed, f"{parameter} actual {actual} {op} expected {expected}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"parameter": parameter, "actual": actual, "op": op, "expected": expected},
    )


def _check_branch_parameter_satisfied(
    topics: dict[str, Any],
    windows: dict[str, dict],
    parameters: dict[str, Any],
    check: dict,
) -> dict:
    normalized = dict(check)
    if normalized.get("value") is None or not normalized.get("op"):
        parsed = _parse_parameter_predicate(
            str(normalized.get("source_predicate") or ""),
            str(normalized.get("parameter") or ""),
        )
        if parsed is not None:
            _, op, expected = parsed
            if normalized.get("op") is None:
                normalized["op"] = op
            if normalized.get("value") is None:
                normalized["value"] = expected
    result = _check_parameter_equals(topics, windows, parameters, normalized)
    result["type"] = check.get("type")
    return result


def _check_tracks_parameter_value(
    topics: dict[str, Any],
    windows: dict[str, dict],
    parameters: dict[str, Any],
    check: dict,
) -> dict:
    parameter = str(check.get("parameter") or "")
    signal = str(check.get("signal") or "")
    if not parameter:
        return _check_result(check, status="unresolved", message="parameter is missing")
    expected, missing_reason = _get_parameter(parameters, parameter, kind="float")
    if missing_reason:
        return _check_result(check, status="unresolved", message=missing_reason)

    samples_result = _check_samples(topics, windows, check, signal)
    if "result" in samples_result:
        return samples_result["result"]

    values = [_number(value) for _, value in samples_result["samples"]]
    values = [value for value in values if value is not None]
    if not values:
        return _check_result(check, status="unresolved", message=f"no numeric samples for {signal}")

    max_error = _safe_float(check.get("max_error"))
    if max_error is None:
        max_error = 0.0

    actual_max = max(abs(value - expected) for value in values)
    passed = actual_max <= max_error
    message = _message(check, passed, f"{signal} max error from {parameter} {actual_max} <= {max_error}")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={
            "signal": signal,
            "parameter": parameter,
            "parameter_value": _round_float(expected),
            "max_abs_error": _round_float(actual_max),
            "max_error": max_error,
        },
    )


def _check_derived_expression(
    topics: dict[str, Any],
    windows: dict[str, dict],
    parameters: dict[str, Any],
    check: dict,
) -> dict:
    expression = str(check.get("expression") or "").strip()
    if not expression:
        return _check_result(check, status="unresolved", message="derived expression is missing")

    window = _window_for_check(windows, check)
    if window is None:
        return _check_result(check, status="unresolved", message=f"unknown window: {check.get('window')}")

    expected_expression = str(check.get("expected_expression") or "").strip()

    # Helper substitution already ran at compile time inside
    # compile_check_plan against helpers whose lowered_return_expression
    # was replaced with the BindingIndex-materialized resolved form. The
    # registry stays available for the dependency-resolution check
    # below; the runtime substitute_helpers walk is redundant.
    helper_registry = check.get("_helper_registry")

    helper_dependency = _first_unresolved_helper_dependency(check)
    if helper_dependency is not None and not _helper_dependency_resolved(helper_dependency, helper_registry):
        helper_name = helper_dependency["name"]
        reason = helper_dependency.get("unresolved_reason") or "helper has not been translated into safe expression IR"
        return _check_result(
            check,
            status="unresolved",
            message=f"cannot evaluate helper {helper_name}: {reason}",
            value={
                "expression": expression,
                "helper_dependency": helper_dependency,
            },
        )

    context = _expression_context(
        topics,
        parameters,
        check,
        expression,
        window,
        expected_expression=expected_expression,
    )
    if context["missing"] and not context.get("defer_missing"):
        missing = ", ".join(context["missing"])
        return _check_result(
            check,
            status="unresolved",
            message=f"cannot evaluate expression '{expression}': missing input {missing}",
            value={"expression": expression, "missing_inputs": context["missing"]},
        )

    try:
        results = _evaluate_expression_over_context(expression, context)
    except ExpressionEvaluationError as exc:
        return _check_result(
            check,
            status="unresolved",
            message=f"cannot evaluate expression '{expression}': {exc}",
            value={"expression": expression},
        )

    if not results:
        return _check_result(check, status="unresolved", message=f"no expression samples for {expression}")

    op = str(check.get("op") or "").strip()
    expected_literal = check.get("value")
    comparisons: list[bool] = []
    expected_values: list[Any] = []
    try:
        if expected_expression:
            expected_results = _evaluate_expression_over_context(expected_expression, context)
            expected_values = [value for _, value in expected_results]
            if len(expected_values) != len(results):
                raise ExpressionEvaluationError("expected expression produced mismatched samples")
            op = op or "=="
            tolerance = _safe_float(check.get("max_error"))
            comparisons = [
                _compare_literal(actual, op, expected, tolerance=tolerance)
                for (_, actual), expected in zip(results, expected_values)
            ]
        elif op:
            comparisons = [
                _compare_literal(actual, op, expected_literal)
                for _, actual in results
            ]
            expected_values = [expected_literal]
        else:
            comparisons = [bool(actual) for _, actual in results]
    except (ExpressionEvaluationError, ValueError, TypeError) as exc:
        return _check_result(
            check,
            status="unresolved",
            message=f"cannot evaluate expression '{expression}': {exc}",
            value={"expression": expression},
        )

    mode = str(check.get("mode") or "any")
    passed = all(comparisons) if mode == "all" else any(comparisons)
    values = [_expression_display_value(value) for _, value in results]
    message = _message(
        check,
        passed,
        f"{expression} values include {dict(Counter(values))}",
    )
    value: dict[str, Any] = {
        "expression": expression,
        "mode": mode,
        "values": values[:20],
    }
    if op:
        value["op"] = op
        value["expected"] = expected_expression or expected_literal
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value=value,
    )


def _check_topic_field_present(topics: dict[str, Any], windows: dict[str, dict], parameters: dict[str, Any], check: dict) -> dict:
    signal = str(check.get("signal") or "")
    parsed = _parse_signal(signal)
    if parsed is None:
        return _check_result(check, status="unresolved", message=f"invalid signal: {signal or 'missing'}")

    topic_name, field_name = parsed
    topic = topics.get(topic_name)
    data = getattr(topic, "data", {}) or {} if topic is not None else {}
    field_present = field_or_flattened_prefix_present(field_name, data)
    passed = topic is not None and field_present and "timestamp" in data
    message = _message(check, passed, f"{signal} is logged" if passed else f"{signal} is not logged")
    return _check_result(
        check,
        status="passed" if passed else "failed",
        message=message,
        value={"signal": signal, "topic_present": topic is not None, "field_present": field_present},
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

    setpoint_samples = [
        (time_s, number)
        for time_s, value in setpoint_result["samples"]
        if (number := _number(value)) is not None
    ]
    setpoint_times = [time_s for time_s, _ in setpoint_samples]
    setpoint_values = [value for _, value in setpoint_samples]
    if not setpoint_times:
        return {"result": _check_result(check, status="unresolved", message=f"no numeric setpoint samples for {setpoint}")}

    errors = []
    for time_s, value in actual_result["samples"]:
        actual_value = _number(value)
        if actual_value is None or time_s < setpoint_times[0] or time_s > setpoint_times[-1]:
            continue
        setpoint_value = _interpolated_value(setpoint_times, setpoint_values, time_s)
        if setpoint_value is not None:
            errors.append(actual_value - setpoint_value)

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


class ExpressionEvaluationError(ValueError):
    pass


ALLOWED_EXPRESSION_FUNCTIONS = SAFE_MATH_FUNCTIONS


def _expression_context(
    topics: dict[str, Any],
    parameters: dict[str, Any],
    check: dict,
    expression: str,
    window: dict[str, Any],
    *,
    expected_expression: str = "",
) -> dict[str, Any]:
    variables = _expression_variables_map(check.get("variables") or {})
    names = source_expression_names(expression)
    if expected_expression:
        names.extend(source_expression_names(expected_expression))
    names = dedupe_keep_order(names)
    defer_missing = _expression_has_conditional(expression) or _expression_has_conditional(expected_expression)

    scalars: dict[str, Any] = {}
    series: dict[str, list[tuple[float, Any]]] = {}
    missing: list[str] = []

    for name in names:
        if name in ALLOWED_EXPRESSION_FUNCTIONS:
            continue
        source = variables.get(name)
        if source is None and name in parameters:
            scalars[name] = _json_safe_value(parameters[name])
            continue
        if source is None:
            missing.append(name)
            continue

        source_text = str(source)
        if source_text in parameters:
            scalars[name] = _json_safe_value(parameters[source_text])
            continue
        literal = _number(source_text)
        if literal is not None:
            scalars[name] = literal
            continue
        signal_samples = _signal_window_samples(topics, source_text, window)
        if "missing" in signal_samples:
            missing.append(f"{name} ({signal_samples['missing']})")
            continue
        series[name] = signal_samples["samples"]

    times: list[float] = []
    if series:
        first_series = next(iter(series.values()))
        times = [time_s for time_s, _ in first_series]

    return {
        "scalars": scalars,
        "series": series,
        "times": times,
        "missing": missing,
        "defer_missing": defer_missing,
    }


def _expression_has_conditional(expression: str) -> bool:
    if not expression:
        return False
    try:
        tree = ast.parse(normalize_expression_function_names(expression), mode="eval")
    except SyntaxError:
        return False
    return any(isinstance(node, ast.IfExp) for node in ast.walk(tree))


def _signal_window_samples(topics: dict[str, Any], signal: str, window: dict[str, Any]) -> dict[str, Any]:
    parsed = _parse_signal(signal)
    if parsed is None:
        return {"missing": signal}
    topic_name, field_name = parsed
    topic = topics.get(topic_name)
    if topic is None:
        return {"missing": signal}
    data = getattr(topic, "data", {}) or {}
    timestamps = data.get("timestamp")
    values = data.get(field_name)
    if timestamps is None or values is None:
        return {"missing": signal}
    samples = _window_samples(timestamps, values, window["start_s"], window["end_s"])
    if not samples:
        return {"missing": signal}
    return {"samples": samples}


def _expression_variables_map(raw_variables: Any) -> dict[str, str]:
    if isinstance(raw_variables, dict):
        return {str(name): str(source) for name, source in raw_variables.items()}
    if not isinstance(raw_variables, list):
        return {}
    variables: dict[str, str] = {}
    for item in raw_variables:
        if isinstance(item, dict):
            name = item.get("name")
            source = item.get("source")
        else:
            name = getattr(item, "name", None)
            source = getattr(item, "source", None)
        if name and source:
            variables[str(name)] = str(source)
    return variables


def _helper_dependency_resolved(
    dependency: dict[str, Any] | None,
    registry: Any,
) -> bool:
    """Whether a helper_dependency's unresolved_reason can be cleared.

    True when a HelperRegistry is in scope AND the registry has a record
    for the dependency's name with no unresolved_reason of its own — i.e.
    the profiler successfully lowered the helper from source, regardless
    of what the LLM wrote into the dependency's unresolved_reason string.
    """
    if dependency is None:
        return False
    if not isinstance(registry, HelperRegistry):
        return False
    name = str(dependency.get("name") or "").strip()
    if not name:
        return False
    ref = registry.get(name)
    if ref is None:
        return False
    return not ref.get("unresolved_reason")


def _first_unresolved_helper_dependency(check: dict) -> dict[str, Any] | None:
    dependencies = check.get("helper_dependencies") or []
    if not isinstance(dependencies, list):
        return None
    for item in dependencies:
        if isinstance(item, dict):
            dependency = item
        else:
            dependency = {
                "name": getattr(item, "name", None),
                "args": getattr(item, "args", []),
                "source_file": getattr(item, "source_file", None),
                "source_line": getattr(item, "source_line", None),
                "unresolved_reason": getattr(item, "unresolved_reason", None),
            }
        if dependency.get("unresolved_reason"):
            return {key: value for key, value in dependency.items() if value is not None}
    return None


def _evaluate_expression_over_context(expression: str, context: dict[str, Any]) -> list[tuple[float | None, Any]]:
    series = context["series"]
    scalars = context["scalars"]
    env_keys = list(series.keys()) + list(scalars.keys())
    rewritten, alias_to_name = alias_dotted_names(normalize_source_expression(expression), env_keys)
    tree = _parse_expression(rewritten)
    name_to_alias = {name: alias for alias, name in alias_to_name.items()}

    def _bind(env: dict[str, Any]) -> dict[str, Any]:
        return {name_to_alias.get(name, name): value for name, value in env.items()}

    if not series:
        return [(None, _eval_expression_node(tree, _bind(scalars)))]

    results: list[tuple[float | None, Any]] = []
    for time_s in context["times"]:
        env = dict(scalars)
        for name, samples in series.items():
            value = _series_value_at(samples, time_s)
            if value is None:
                raise ExpressionEvaluationError(f"missing sample for {name} at {time_s}")
            env[name] = value
        results.append((time_s, _eval_expression_node(tree, _bind(env))))
    return results


def _parse_expression(expression: str) -> ast.Expression:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ExpressionEvaluationError("invalid expression syntax") from exc
    return tree


def _eval_expression_node(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_expression_node(node.body, env)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        raise ExpressionEvaluationError("string literals are not supported")
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExpressionEvaluationError(f"missing runtime variable {node.id}")
        return env[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _numeric_expression_value(_eval_expression_node(node.operand, env))
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_expression_node(node.operand, env))
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
        return all(bool(_eval_expression_node(value, env)) for value in node.values)
    if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
        return any(bool(_eval_expression_node(value, env)) for value in node.values)
    if isinstance(node, ast.IfExp):
        branch = node.body if bool(_eval_expression_node(node.test, env)) else node.orelse
        return _eval_expression_node(branch, env)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod)):
        left = _numeric_expression_value(_eval_expression_node(node.left, env))
        right = _numeric_expression_value(_eval_expression_node(node.right, env))
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if right == 0:
            raise ExpressionEvaluationError("division by zero")
        if isinstance(node.op, ast.Mod):
            return left % right
        return left / right
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_EXPRESSION_FUNCTIONS:
            raise ExpressionEvaluationError("unsupported function")
        if node.keywords:
            raise ExpressionEvaluationError("keyword arguments are not supported")
        args = [_numeric_expression_value(_eval_expression_node(arg, env)) for arg in node.args]
        if not args:
            raise ExpressionEvaluationError("function requires at least one argument")
        return ALLOWED_EXPRESSION_FUNCTIONS[node.func.id](*args)
    if isinstance(node, ast.Compare):
        left = _eval_expression_node(node.left, env)
        for operator, comparator in zip(node.ops, node.comparators):
            right = _eval_expression_node(comparator, env)
            op = _comparison_operator(operator)
            if not _compare_literal(left, op, right):
                return False
            left = right
        return True
    raise ExpressionEvaluationError("unsupported expression syntax")


def _comparison_operator(operator: ast.cmpop) -> str:
    if isinstance(operator, ast.Gt):
        return ">"
    if isinstance(operator, ast.GtE):
        return ">="
    if isinstance(operator, ast.Lt):
        return "<"
    if isinstance(operator, ast.LtE):
        return "<="
    if isinstance(operator, ast.Eq):
        return "=="
    if isinstance(operator, ast.NotEq):
        return "!="
    raise ExpressionEvaluationError("unsupported comparison operator")


def _numeric_expression_value(value: Any) -> float:
    number = _number(value)
    if number is None:
        raise ExpressionEvaluationError(f"non-numeric expression value: {value}")
    return number


def _series_value_at(samples: list[tuple[float, Any]], target_time: float) -> Any:
    numeric_samples = [
        (time_s, number)
        for time_s, value in samples
        if (number := _number(value)) is not None
    ]
    if len(numeric_samples) == len(samples):
        return _interpolated_value(
            [time_s for time_s, _ in numeric_samples],
            [value for _, value in numeric_samples],
            target_time,
        )
    for time_s, value in samples:
        if abs(time_s - target_time) < 1e-9:
            return value
    return None


def _expression_display_value(value: Any) -> Any:
    number = _number(value)
    if number is not None:
        return _round_float(number)
    return _json_safe_value(value)




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
        if topic is not None and field_or_flattened_prefix_present(field_name, data) and "timestamp" in data:
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


def _window_samples(timestamps: Any, values: Any, start_s: float, end_s: float) -> list[tuple[float, Any]]:
    samples = []
    for timestamp, value in zip(timestamps, values):
        time_s = _timestamp_to_seconds(timestamp)
        if start_s <= time_s <= end_s:
            samples.append((time_s, _json_safe_value(value)))
    return samples


def _sample_metrics(samples: list[tuple[float, Any]]) -> dict[str, Any]:
    values = [_number(value) for _, value in samples]
    values = [value for value in values if value is not None]
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


def _normalize_state_samples(
    signal: str,
    samples: list[tuple[float, Any]],
    source_path: Path | None = None,
) -> list[tuple[float, Any]]:
    return [(time_s, _normalize_state_value(signal, value, source_path)) for time_s, value in samples]


def _normalize_state_value(signal: str, value: Any, source_path: Path | None = None) -> Any:
    if source_path is None:
        return normalize_px4_enum_value(signal, value)
    return normalize_px4_enum_value(signal, value, source_path)


def _numeric_delta(samples: list[tuple[float, Any]]) -> float | None:
    values = [_number(value) for _, value in samples]
    values = [value for value in values if value is not None]
    if len(values) < 2:
        return None
    return values[-1] - values[0]


def _compare_literal(actual: Any, op: str, expected: Any, *, tolerance: float | None = None) -> bool:
    actual_number = _number(actual)
    expected_number = _number(expected)
    if actual_number is not None and expected_number is not None:
        if op == "==" and tolerance is not None:
            return abs(actual_number - expected_number) <= tolerance
        if op == "!=" and tolerance is not None:
            return abs(actual_number - expected_number) > tolerance
        return _compare(actual_number, op, expected_number)

    if op == "==":
        return _normalized_literal(actual) == _normalized_literal(expected)
    if op == "!=":
        return _normalized_literal(actual) != _normalized_literal(expected)
    raise ValueError(f"operator {op} requires numeric values")


def _normalized_literal(value: Any) -> Any:
    value = _json_safe_value(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return value.strip()
    return value


def _parse_parameter_predicate(predicate: str, parameter: str) -> tuple[str, str, Any] | None:
    if not predicate:
        return None
    pattern = re.compile(
        r"(?P<left>[A-Za-z_][A-Za-z0-9_.:]*\s*(?:\.\s*get\s*\(\s*\))?)\s*"
        r"(?P<op>>=|<=|==|!=|>|<)\s*"
        r"(?P<right>-?[A-Za-z_][A-Za-z0-9_:]*|-?\d+(?:\.\d+)?|true|false)"
    )
    for match in pattern.finditer(predicate):
        left = match.group("left").replace(" ", "")
        if parameter and parameter not in left and ".get()" not in left:
            continue
        return parameter, match.group("op"), _parse_literal(match.group("right"))
    return None


def _parse_literal(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    number = _number(value)
    if number is not None:
        if float(number).is_integer() and "." not in value:
            return int(number)
        return number
    return value


def _message(check: dict, passed: bool, fallback: str) -> str:
    effect = _claim_effect(check, "passed" if passed else "failed")
    if effect == "supports" and check.get("supports"):
        return str(check.get("supports"))
    if effect == "contradicts" and check.get("contradicts"):
        return str(check.get("contradicts"))
    key = "supports" if passed else "contradicts"
    return str(check.get(key) or check.get("description") or fallback)


def _claim_effect(check: dict, status: str) -> str | None:
    if status == "unresolved":
        return None
    if status == "passed":
        if check.get("supports"):
            return "supports"
        if check.get("contradicts"):
            return "contradicts"
        return "supports"
    if status == "failed":
        if check.get("contradicts"):
            return "contradicts"
        return "contradicts"
    return None


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
    for key in ("check_id", "branch_id", "role"):
        if check.get(key) is not None:
            result[key] = check.get(key)
    if value is not None:
        result["value"] = value
    return result


def _evaluation_result(
    *,
    mechanism: str,
    expected_signature: list[dict],
    present: list[str],
    missing: list[str],
    window_results: list[dict],
    numeric_results: list[dict],
    exclusion_results: list[dict],
    evidence: list[str],
    contradictions: list[str],
    unresolved: list[str],
    warnings: list[str],
) -> dict:
    score, confidence = _confidence(numeric_results, exclusion_results, missing)
    verdict, confidence_ceiling = _verdict(evidence, contradictions, unresolved, missing)
    check_results = [*numeric_results, *exclusion_results]
    summary = _summary(confidence, evidence, contradictions, unresolved)

    return {
        "mechanism": mechanism,
        "expected_signature": expected_signature,
        "required_signals": {
            "present": present,
            "missing": missing,
        },
        "missing_required_signals": missing,
        "window_results": window_results,
        "exclusion_checks": exclusion_results,
        "numeric_checks": numeric_results,
        "check_results": check_results,
        "evidence": evidence,
        "contradictions": contradictions,
        "contradicting_evidence": contradictions,
        "unresolved": unresolved,
        "warnings": warnings,
        "verdict": verdict,
        "confidence": confidence,
        "confidence_ceiling": confidence_ceiling,
        "confidence_score": score,
        "summary": summary,
    }


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
    return score, "low"


def _verdict(
    evidence: list[str],
    contradictions: list[str],
    unresolved: list[str],
    missing: list[str],
) -> tuple[str, str]:
    verdict = verdict_from_counts(
        supported=len(evidence),
        contradicted=len(contradictions),
    )
    ceiling = ceiling_for(
        verdict,
        has_unresolved_defining=bool(unresolved),
        missing_required_signals=bool(missing),
    )
    return verdict, ceiling


def _summary(confidence: str, evidence: list[str], contradictions: list[str], unresolved: list[str]) -> str:
    return (
        f"Log-evidence confidence is {confidence}: "
        f"{len(evidence)} supporting checks, "
        f"{len(contradictions)} contradicting checks, "
        f"{len(unresolved)} unresolved checks."
    )


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
