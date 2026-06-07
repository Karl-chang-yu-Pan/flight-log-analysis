from __future__ import annotations

import re
from typing import Any, Optional

from flight_log_agent.models import ApplicabilityResult, MechanismCandidate, VerificationPlan, WindowSpec
from flight_log_agent.px4.msg_schema import (
    field_or_flattened_prefix_present,
    load_px4_msg_schema,
    normalize_px4_enum_value,
)


def evaluate_candidate_applicability(
    candidate: MechanismCandidate,
    inventory: dict,
    timeline: list[dict],
    mission: Optional[dict],
    verification_plan: Optional[VerificationPlan] = None,
) -> ApplicabilityResult:
    """
    Use actual parameters/timeline/mission/topic availability to eliminate mechanisms.
    This is where parameters and topics enter the workflow.
    """
    if verification_plan is not None:
        from flight_log_agent.analysis.verification_plan import applicability_from_verification_plan

        return applicability_from_verification_plan(candidate, verification_plan, inventory)

    params = inventory.get("parameters") or {}
    topic_fields = inventory.get("topic_fields") or {}
    schema_topic_fields = load_px4_msg_schema(inventory.get("source_path"))
    available_topics = set(inventory.get("available_topics") or topic_fields.keys())

    relevant_parameters = {
        name: params.get(name)
        for name in candidate.required_parameters
        if name in params
    }

    supported: list[str] = []
    excluded: list[str] = []
    unresolved: list[str] = []

    for param_name in candidate.required_parameters:
        if param_name in params:
            supported.append(f"Parameter present: {param_name}={params.get(param_name)}")
        else:
            unresolved.append(f"Required/candidate parameter not found in log: {param_name}")

    available_required_signals, missing_required_signals = check_required_signals(
        candidate.required_signals,
        available_topics,
        topic_fields,
        schema_topic_fields,
    )

    if missing_required_signals:
        unresolved.append(f"Missing required signals: {missing_required_signals}")

    candidate_windows = derive_candidate_windows(candidate, timeline, mission, inventory=inventory)
    if not candidate_windows:
        if candidate_has_logged_predicates(candidate):
            excluded.append("No timeline window satisfied the candidate's logged source predicates.")
        else:
            unresolved.append("No candidate verification window could be derived from timeline/mission.")

    return ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=len(excluded) == 0,
        supported_conditions=supported,
        excluded_by=excluded,
        unresolved_conditions=unresolved,
        relevant_parameters=relevant_parameters,
        candidate_windows=candidate_windows,
        available_required_signals=available_required_signals,
        missing_required_signals=missing_required_signals,
    )


def derive_candidate_windows(
    candidate: MechanismCandidate,
    timeline: list[dict],
    mission: Optional[dict],
    inventory: Optional[dict] = None,
) -> list[WindowSpec]:
    windows: list[WindowSpec] = []
    for plot in candidate.plot_requests:
        if plot.start_s is not None and plot.end_s is not None:
            windows.append(
                WindowSpec(
                    name=plot.title.lower().replace(" ", "_"),
                    start_s=float(plot.start_s),
                    end_s=float(plot.end_s),
                    reason=plot.purpose,
                )
            )
    if windows:
        return windows

    if candidate_has_logged_predicates(candidate):
        return derive_logged_predicate_windows(
            candidate,
            timeline,
            source_path=(inventory or {}).get("source_path"),
        )

    span = timeline_time_span(timeline)
    if span is None:
        return windows

    start_s, end_s = span
    for name in check_window_names(candidate):
        windows.append(
            WindowSpec(
                name=name,
                start_s=start_s,
                end_s=end_s,
                reason=f"Fallback verification window from full timeline span for check window '{name}'.",
            )
        )
    return windows


def candidate_has_logged_predicates(candidate: MechanismCandidate) -> bool:
    return any(
        parse_logged_predicate(predicate) is not None
        for group in candidate_source_predicate_groups(candidate)
        for predicate in group
    )


def derive_logged_predicate_windows(
    candidate: MechanismCandidate,
    timeline: list[dict],
    *,
    source_path: Optional[str] = None,
) -> list[WindowSpec]:
    span = timeline_time_span(timeline)
    if span is None:
        return []

    windows: list[WindowSpec] = []
    for index, predicates in enumerate(candidate_source_predicate_groups(candidate), start=1):
        parsed = [
            parsed for parsed in (
                parse_logged_predicate(predicate, source_path=source_path)
                for predicate in predicates
            )
            if parsed is not None
        ]
        if not parsed:
            continue
        intervals = [span]
        for predicate in parsed:
            intervals = intersect_interval_sets(
                intervals,
                timeline_predicate_intervals(timeline, predicate, span),
            )
            if not intervals:
                break
        for start_s, end_s in intervals:
            windows.append(
                WindowSpec(
                    name=f"source_predicate_window_{index}",
                    start_s=start_s,
                    end_s=end_s,
                    reason="Timeline window satisfying logged source predicates: "
                    + "; ".join(predicate["text"] for predicate in parsed),
                )
            )
    return windows


def candidate_source_predicate_groups(candidate: MechanismCandidate) -> list[list[str]]:
    groups: list[list[str]] = []
    if candidate.mode_state_gates:
        groups.append(list(candidate.mode_state_gates))
    for group in getattr(candidate, "branch_groups", []) or []:
        if group.source_predicates:
            groups.append(list(group.source_predicates))
    return groups or [[]]


def parse_logged_predicate(predicate: str, *, source_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    match = re.search(
        r"(?P<signal>[a-z][a-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s*"
        r"(?P<op>==|!=)\s*"
        r"(?P<value>[A-Za-z_][A-Za-z0-9_:]*|-?\d+(?:\.\d+)?|true|false)",
        predicate or "",
    )
    if not match:
        return None
    signal = match.group("signal")
    raw_value = match.group("value")
    value = normalize_px4_enum_value(signal, parse_predicate_literal(raw_value), source_path)
    return {
        "signal": signal,
        "op": match.group("op"),
        "value": value,
        "text": f"{signal} {match.group('op')} {raw_value}",
    }


def parse_predicate_literal(value: str) -> Any:
    lowered = str(value).lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if number.is_integer() and "." not in str(value):
        return int(number)
    return number


def timeline_predicate_intervals(
    timeline: list[dict],
    predicate: dict[str, Any],
    span: tuple[float, float],
) -> list[tuple[float, float]]:
    signal = predicate["signal"]
    topic, field = signal.split(".", 1)
    events = sorted(
        (
            (float(event["time_s"]), event.get("value"))
            for event in timeline or []
            if isinstance(event, dict)
            and event.get("topic") == topic
            and event.get("field") == field
            and isinstance(event.get("time_s"), (int, float))
        ),
        key=lambda item: item[0],
    )
    if not events:
        return []
    intervals: list[tuple[float, float]] = []
    start_s, end_s = span
    for index, (event_time, value) in enumerate(events):
        interval_start = max(event_time, start_s)
        interval_end = min(events[index + 1][0], end_s) if index + 1 < len(events) else end_s
        if interval_end <= interval_start:
            continue
        if predicate_matches(value, predicate["op"], predicate["value"]):
            intervals.append((interval_start, interval_end))
    return intervals


def predicate_matches(actual: Any, op: str, expected: Any) -> bool:
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    return False


def intersect_interval_sets(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    intersections: list[tuple[float, float]] = []
    for left_start, left_end in left:
        for right_start, right_end in right:
            start_s = max(left_start, right_start)
            end_s = min(left_end, right_end)
            if end_s > start_s:
                intersections.append((start_s, end_s))
    return intersections


def check_window_names(candidate: MechanismCandidate) -> list[str]:
    names = []
    for check in list(candidate.numeric_checks or []) + list(candidate.exclusion_checks or []):
        window = getattr(check, "window", None)
        if window and window not in names:
            names.append(str(window))
    return names


def timeline_time_span(timeline: list[dict]) -> Optional[tuple[float, float]]:
    times = []
    for event in timeline or []:
        time_s = event.get("time_s") if isinstance(event, dict) else None
        if isinstance(time_s, (int, float)):
            times.append(float(time_s))
    if not times:
        return None
    start_s = min(times)
    end_s = max(times)
    if end_s < start_s:
        return None
    if end_s == start_s:
        end_s = start_s + 0.001
    return start_s, end_s


def check_required_signals(
    required_signals: list[str],
    available_topics: set[str],
    topic_fields: dict[str, Any],
    schema_topic_fields: Optional[dict[str, list[str]]] = None,
) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []

    for signal in required_signals:
        if "." not in signal:
            if signal in available_topics:
                available.append(signal)
            else:
                missing.append(signal)
            continue

        topic, field = signal.split(".", 1)
        fields = topic_fields.get(topic)
        schema_fields = (schema_topic_fields or {}).get(topic)
        field_known = (
            topic not in topic_fields
            or field_or_flattened_prefix_present(field, fields or [])
            or field_or_flattened_prefix_present(field, schema_fields or [])
        )
        if topic in available_topics and field_known:
            available.append(signal)
        else:
            missing.append(signal)

    return available, missing
