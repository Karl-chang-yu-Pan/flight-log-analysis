from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import re
from typing import Any, Iterable, Optional

from flight_log_agent.expression_math import SAFE_MATH_FUNCTIONS, normalize_expression_function_names
from flight_log_agent.models import (
    ApplicabilityResult,
    MechanismBranchGroup,
    MechanismCandidate,
    RelationshipCheckSpec,
    VerificationBranchPlan,
    VerificationCheckPlan,
    VerificationPlan,
    VerificationSignalResolution,
    WindowSpec,
)
from flight_log_agent.px4.msg_schema import (
    field_or_flattened_prefix_present,
    load_px4_msg_schema,
    normalize_px4_enum_value,
)
from flight_log_agent.px4.source_snapshot import source_from_inventory


CHECK_SIGNAL_FIELDS = ("signal", "first", "second", "actual", "setpoint")
ALLOWED_EXPRESSION_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.IfExp,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Call,
    ast.Compare,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
)


def compile_verification_plan(
    candidate: MechanismCandidate,
    inventory: dict[str, Any],
    timeline: list[dict[str, Any]],
    mission: Optional[dict[str, Any]],
    output_bindings: Iterable[Any] = (),
) -> VerificationPlan:
    resolver = SignalResolver(inventory, output_bindings)
    mechanism_id = stable_id("mechanism", {
        "name": candidate.name,
        "source_refs": [_model_dump(ref) for ref in candidate.source_refs],
    })
    shared_numeric_checks = list(candidate.numeric_checks)
    if candidate.branch_groups:
        branch_inputs = [
            (
                group.name,
                [*candidate.source_refs, *group.source_refs],
                [*candidate.mode_state_gates, *group.source_predicates],
                dedupe([*candidate.required_parameters, *group.required_parameters]),
                dedupe([*candidate.required_signals, *group.required_signals]),
                [
                    *checks_for_branch(shared_numeric_checks, group.name),
                    *group.numeric_checks,
                ],
                [
                    *checks_for_branch(candidate.exclusion_checks, group.name),
                    *group.exclusion_checks,
                ],
            )
            for group in candidate.branch_groups
        ]
    else:
        branch_names = dedupe(
            check.branch_group
            for check in [*shared_numeric_checks, *candidate.exclusion_checks]
            if check.branch_group
        )
        branch_inputs = [
            (
                name,
                candidate.source_refs,
                candidate.mode_state_gates,
                candidate.required_parameters,
                candidate.required_signals,
                checks_for_branch(shared_numeric_checks, name),
                checks_for_branch(candidate.exclusion_checks, name),
            )
            for name in (branch_names or ["candidate"])
        ]
    branch_inputs = expand_disjunctive_branch_inputs(branch_inputs)

    branches = [
        compile_branch_plan(
            mechanism_id=mechanism_id,
            name=name,
            source_refs=source_refs,
            source_predicates=source_predicates,
            required_parameters=required_parameters,
            required_signals=required_signals,
            numeric_checks=numeric_checks,
            exclusion_checks=exclusion_checks,
            candidate=candidate,
            inventory=inventory,
            timeline=timeline,
            mission=mission,
            resolver=resolver,
        )
        for (
            name,
            source_refs,
            source_predicates,
            required_parameters,
            required_signals,
            numeric_checks,
            exclusion_checks,
        ) in branch_inputs
    ]
    return VerificationPlan(
        mechanism_id=mechanism_id,
        candidate_name=candidate.name,
        branches=branches,
    )


def resolved_candidate_predicate_signals(
    candidates: Iterable[MechanismCandidate],
    inventory: dict[str, Any],
    output_bindings: Iterable[Any] = (),
) -> list[str]:
    resolver = SignalResolver(inventory, output_bindings)
    signals: list[str] = []
    for candidate in candidates:
        for predicate_group in candidate_source_predicate_groups(candidate):
            for predicate in predicate_group:
                for parsed in parse_and_resolve_predicates(predicate, resolver, inventory):
                    signal = parsed.get("signal")
                    if parsed.get("resolved") and parsed.get("kind") == "logged_signal" and signal:
                        signals.append(str(signal))
    return dedupe(signals)


def candidate_source_predicate_groups(candidate: MechanismCandidate) -> list[list[str]]:
    groups: list[list[str]] = []
    if candidate.mode_state_gates:
        groups.append(list(candidate.mode_state_gates))
    for group in getattr(candidate, "branch_groups", []) or []:
        if group.source_predicates:
            groups.append(list(group.source_predicates))
    return groups or [[]]


def applicability_from_verification_plan(
    candidate: MechanismCandidate,
    plan: VerificationPlan,
    inventory: dict[str, Any],
) -> ApplicabilityResult:
    parameters = inventory.get("parameters") or {}
    topic_fields = inventory.get("topic_fields") or {}
    schema = load_px4_msg_schema(source_from_inventory(inventory))
    available_topics = set(inventory.get("available_topics") or topic_fields.keys())
    applicable_branches = [branch for branch in plan.branches if branch.applicable and branch.windows]
    required_parameters = dedupe(
        parameter
        for branch in applicable_branches
        for parameter in branch.required_parameters
    )
    required_signals = dedupe(
        signal
        for branch in applicable_branches
        for signal in branch.required_signals
    )
    available_signals, missing_signals = check_signal_availability(
        required_signals,
        available_topics,
        topic_fields,
        schema,
    )
    unresolved = dedupe(
        dependency
        for branch in applicable_branches
        for dependency in branch.unresolved_dependencies
    )
    unresolved.extend(f"Missing required signals: {missing_signals}" for _ in [0] if missing_signals)
    excluded = []
    if not applicable_branches:
        excluded.append("No verification branch has a resolved applicable log-data window.")
    return ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=bool(applicable_branches),
        supported_conditions=[
            f"Parameter present: {parameter}={parameters[parameter]}"
            for parameter in required_parameters
            if parameter in parameters
        ],
        excluded_by=excluded,
        unresolved_conditions=dedupe(unresolved),
        relevant_parameters={
            parameter: parameters[parameter]
            for parameter in required_parameters
            if parameter in parameters
        },
        candidate_windows=[
            window
            for branch in applicable_branches
            for window in branch.windows
        ],
        available_required_signals=available_signals,
        missing_required_signals=missing_signals,
    )


def compile_branch_plan(
    *,
    mechanism_id: str,
    name: str,
    source_refs: list[Any],
    source_predicates: list[str],
    required_parameters: list[str],
    required_signals: list[str],
    numeric_checks: list[RelationshipCheckSpec],
    exclusion_checks: list[RelationshipCheckSpec],
    candidate: MechanismCandidate,
    inventory: dict[str, Any],
    timeline: list[dict[str, Any]],
    mission: Optional[dict[str, Any]],
    resolver: "SignalResolver",
) -> VerificationBranchPlan:
    branch_id = stable_id("branch", {
        "mechanism_id": mechanism_id,
        "name": name,
        "source_refs": [_model_dump(ref) for ref in source_refs],
        "predicates": source_predicates,
    })
    signal_resolutions = [resolver.resolve(signal) for signal in required_signals]
    resolved_required_signals = [
        resolution.resolved
        for resolution in signal_resolutions
        if resolution.status == "resolved" and resolution.resolved
    ]
    unresolved = [
        f"{resolution.original}: {resolution.reason or resolution.status}"
        for resolution in signal_resolutions
        if resolution.status != "resolved"
    ]
    resolved_predicates: list[str] = []
    parsed_predicates: list[dict[str, Any]] = []
    excluded_by = []
    for predicate in source_predicates:
        parsed_parts = parse_and_resolve_predicates(predicate, resolver, inventory)
        unresolved.extend(
            parsed["reason"]
            for parsed in parsed_parts
            if not parsed.get("resolved")
        )
        if not parsed_parts or not all(parsed.get("resolved") for parsed in parsed_parts):
            continue
        resolved_predicates.extend(parsed["text"] for parsed in parsed_parts)
        contradicted = [
            parsed
            for parsed in parsed_parts
            if parsed.get("kind") == "parameter" and not parsed.get("satisfied", False)
        ]
        if contradicted:
            excluded_by.append(
                "Fully resolved source applicability predicate is contradicted: "
                + predicate
            )
            continue
        for parsed in parsed_parts:
            if parsed.get("kind") != "logged_signal":
                continue
            if timeline_has_signal(timeline, parsed["signal"]):
                parsed_predicates.append(parsed)
            else:
                unresolved.append(
                    f"Predicate signal {parsed['signal']} has no samples in the available timeline."
                )

    windows = resolve_branch_windows(candidate, branch_id, parsed_predicates, timeline, inventory)
    if parsed_predicates and not windows and not excluded_by:
        excluded_by.append("No available log-data window satisfied the branch source predicates.")
    if source_predicates and not resolved_predicates:
        unresolved.append("No source predicate could be resolved to a logged signal.")

    checks: list[VerificationCheckPlan] = []
    for category, source_checks in (("numeric", numeric_checks), ("exclusion", exclusion_checks)):
        for check in source_checks:
            if (
                check.type == "topic_field_present"
                and check.signal
                and not resolver.is_known_signal_reference(check.signal)
            ):
                continue
            checks.append(compile_check_plan(check, category, branch_id, resolver, inventory))
    check_signals = dedupe(
        signal
        for planned in checks
        if planned.executable
        for signal in check_signal_references(planned.check)
    )
    resolved_required_signals = dedupe([*resolved_required_signals, *check_signals])

    return VerificationBranchPlan(
        branch_id=branch_id,
        name=name,
        source_refs=source_refs,
        source_predicates=source_predicates,
        resolved_predicates=resolved_predicates,
        required_parameters=required_parameters,
        required_signals=resolved_required_signals,
        signal_resolutions=signal_resolutions,
        windows=windows,
        checks=checks,
        applicable=not excluded_by,
        excluded_by=excluded_by,
        unresolved_dependencies=dedupe(unresolved),
    )


def compile_check_plan(
    check: RelationshipCheckSpec,
    category: str,
    branch_id: str,
    resolver: "SignalResolver",
    inventory: dict[str, Any],
) -> VerificationCheckPlan:
    role = check_role(check)
    data = _model_dump(check)
    unresolved: list[str] = []
    for field in CHECK_SIGNAL_FIELDS:
        value = data.get(field)
        if not value or not looks_like_signal_reference(str(value)):
            continue
        resolution = resolver.resolve(str(value))
        if resolution.status == "resolved":
            data[field] = resolution.resolved
        else:
            unresolved.append(f"{field} {value} is {resolution.status}: {resolution.reason or resolution.status}")

    variables = []
    for variable in data.get("variables") or []:
        variable_data = _model_dump(variable)
        source = str(variable_data.get("source") or "")
        if looks_like_signal_reference(source):
            resolution = resolver.resolve(source)
            if resolution.status == "resolved":
                variable_data["source"] = resolution.resolved
            else:
                unresolved.append(
                    f"variable {variable_data.get('name')} ({source}) is {resolution.status}: "
                    f"{resolution.reason or resolution.status}"
                )
        variables.append(variable_data)
    data["variables"] = variables

    helper_dependencies = data.get("helper_dependencies") or []
    unresolved.extend(
        f"helper {dependency.get('name')}: {dependency.get('unresolved_reason')}"
        for dependency in helper_dependencies
        if dependency.get("unresolved_reason")
    )
    if check.type == "derived_expression":
        unresolved.extend(validate_derived_expression(data, inventory))

    compiled_check = RelationshipCheckSpec(**data)
    check_id = stable_id("check", {
        "branch_id": branch_id,
        "role": role,
        "category": category,
        "check": data,
    })
    return VerificationCheckPlan(
        check_id=check_id,
        branch_id=branch_id,
        role=role,
        category=category,
        check=compiled_check,
        executable=not unresolved and check.type != "custom",
        unresolved_dependencies=dedupe(unresolved or (
            ["custom semantic requirement is not executable"] if check.type == "custom" else []
        )),
    )


def validate_derived_expression(check: dict[str, Any], inventory: dict[str, Any]) -> list[str]:
    unresolved: list[str] = []
    variable_sources = {
        str(item.get("name")): str(item.get("source"))
        for item in check.get("variables") or []
        if item.get("name") and item.get("source")
    }
    parameters = set((inventory.get("parameters") or {}).keys())
    for field in ("expression", "expected_expression"):
        expression = str(check.get(field) or "").strip()
        if not expression:
            if field == "expression":
                unresolved.append("derived expression is missing")
            continue
        try:
            tree = ast.parse(normalize_expression_function_names(expression), mode="eval")
        except SyntaxError:
            unresolved.append(f"{field} has invalid expression syntax")
            continue
        unsupported = [node.__class__.__name__ for node in ast.walk(tree) if not isinstance(node, ALLOWED_EXPRESSION_NODES)]
        if unsupported:
            unresolved.append(f"{field} uses unsupported expression syntax: {dedupe(unsupported)}")
            continue
        names = {
            node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name)
        }
        allowed_names = set(variable_sources) | parameters | set(SAFE_MATH_FUNCTIONS) | {"True", "False"}
        missing = sorted(names - allowed_names)
        if missing:
            unresolved.append(f"{field} has unresolved variables: {missing}")
    return unresolved


def check_role(check: RelationshipCheckSpec) -> str:
    if check.type == "custom":
        return "advisory"
    if check.type == "topic_field_present":
        return "evidence_availability"
    if check.type in {"parameter_equals", "branch_parameter_satisfied"}:
        return "branch_applicability"
    return "mechanism_defining"


def checks_for_branch(
    checks: list[RelationshipCheckSpec],
    branch_name: str,
) -> list[RelationshipCheckSpec]:
    return [
        check
        for check in checks
        if not check.branch_group or check.branch_group == branch_name
    ]


def expand_disjunctive_branch_inputs(branch_inputs: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    expanded: list[tuple[Any, ...]] = []
    for branch_input in branch_inputs:
        name, source_refs, source_predicates, *rest = branch_input
        alternatives = [safe_disjunction_alternatives(predicate) for predicate in source_predicates]
        combinations = list(itertools.product(*alternatives)) if alternatives else [()]
        for index, predicates in enumerate(combinations, start=1):
            expanded_name = name if len(combinations) == 1 else f"{name}:alternative-{index}"
            expanded.append((expanded_name, source_refs, list(predicates), *rest))
    return expanded


def safe_disjunction_alternatives(predicate: str) -> list[str]:
    expression = strip_outer_parentheses(predicate)
    parts = split_top_level(expression, "||")
    if len(parts) <= 1 or any(not part.strip() for part in parts):
        return [predicate]
    return [strip_outer_parentheses(part.strip()) for part in parts]


def parse_and_resolve_predicates(
    predicate: str,
    resolver: "SignalResolver",
    inventory: dict[str, Any],
) -> list[dict[str, Any]]:
    expression = strip_outer_parentheses(predicate)
    if len(split_top_level(expression, "||")) > 1:
        return [{"resolved": False, "reason": f"Disjunctive source predicate was not expanded safely: {predicate}"}]
    parts = [
        strip_outer_parentheses(part.strip())
        for part in split_top_level(expression, "&&")
        if part.strip()
    ]
    return [
        parse_and_resolve_predicate_part(part, resolver, inventory)
        for part in parts
    ]


def parse_and_resolve_predicate_part(
    predicate: str,
    resolver: "SignalResolver",
    inventory: dict[str, Any],
) -> dict[str, Any]:
    bitmask_match = re.fullmatch(
        r"\(?\s*(?P<left>[_A-Za-z][_A-Za-z0-9]*(?:\s*\.\s*get\s*\(\s*\)|(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*))"
        r"\s*&\s*(?P<mask>[A-Za-z_][A-Za-z0-9_:]*|\d+)\s*\)?\s*(?P<op>==|!=)\s*(?P<value>\d+)",
        predicate.strip(),
    )
    if bitmask_match:
        parameter = parameter_name_for_accessor(bitmask_match.group("left"), inventory)
        mask = parse_literal(bitmask_match.group("mask"))
        expected = parse_literal(bitmask_match.group("value"))
        if not isinstance(mask, int):
            return {"resolved": False, "reason": f"Bitmask constant is unresolved: {bitmask_match.group('mask')}"}
        bit_op = "bit_set" if (
            bitmask_match.group("op") == "!=" and expected == 0
        ) or (
            bitmask_match.group("op") == "==" and expected == mask
        ) else "bit_clear"
        if parameter:
            parameters = inventory.get("parameters") or {}
            if parameter not in parameters:
                return {"resolved": False, "reason": f"Predicate parameter {parameter} is missing from the log."}
            return {
                "resolved": True,
                "kind": "parameter",
                "satisfied": predicate_matches(parameters[parameter], bit_op, mask),
                "text": predicate,
            }
        resolution = resolver.resolve(bitmask_match.group("left"))
        if resolution.status != "resolved" or not resolution.resolved:
            return unresolved_predicate_signal(bitmask_match.group("left"), resolution)
        return {
            "resolved": True,
            "kind": "logged_signal",
            "signal": resolution.resolved,
            "op": bit_op,
            "value": mask,
            "text": predicate,
        }

    finite_match = re.fullmatch(r"PX4_ISFINITE\s*\(\s*(?P<signal>[^()]+)\s*\)", predicate.strip())
    if finite_match:
        resolution = resolver.resolve(finite_match.group("signal"))
        if resolution.status != "resolved" or not resolution.resolved:
            return unresolved_predicate_signal(finite_match.group("signal"), resolution)
        return {
            "resolved": True,
            "kind": "logged_signal",
            "signal": resolution.resolved,
            "op": "isfinite",
            "value": True,
            "text": predicate,
        }

    comparison = re.fullmatch(
        r"(?P<left>[_A-Za-z][_A-Za-z0-9]*(?:\s*\.\s*get\s*\(\s*\)|(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*)?)\s*"
        r"(?P<op>==|!=|>=|<=|>|<)\s*"
        r"(?P<value>[A-Za-z_][A-Za-z0-9_:]*|-?\d+(?:\.\d+)?|true|false)",
        predicate.strip(),
    )
    if comparison:
        parameter = parameter_name_for_accessor(comparison.group("left"), inventory)
        if parameter:
            parameters = inventory.get("parameters") or {}
            if parameter not in parameters:
                return {"resolved": False, "reason": f"Predicate parameter {parameter} is missing from the log."}
            expected = parse_literal(comparison.group("value"))
            op = comparison.group("op")
            return {
                "resolved": True,
                "kind": "parameter",
                "satisfied": predicate_matches(parameters[parameter], op, expected),
                "text": f"{parameter} {op} {comparison.group('value')}",
            }
        resolution = resolver.resolve(comparison.group("left"))
        if resolution.status != "resolved" or not resolution.resolved:
            return unresolved_predicate_signal(comparison.group("left"), resolution)
        raw_value = comparison.group("value")
        value = normalize_px4_enum_value(resolution.resolved, parse_literal(raw_value), source_from_inventory(inventory))
        return {
            "resolved": True,
            "kind": "logged_signal",
            "signal": resolution.resolved,
            "op": comparison.group("op"),
            "value": value,
            "text": f"{resolution.resolved} {comparison.group('op')} {raw_value}",
        }

    boolean_match = re.fullmatch(
        r"(?P<negated>!)?\s*(?P<signal>_?[A-Za-z][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)*)",
        predicate.strip(),
    )
    if boolean_match:
        parameter = parameter_name_for_accessor(boolean_match.group("signal"), inventory)
        if parameter:
            parameters = inventory.get("parameters") or {}
            if parameter not in parameters:
                return {"resolved": False, "reason": f"Predicate parameter {parameter} is missing from the log."}
            expected = not bool(boolean_match.group("negated"))
            return {
                "resolved": True,
                "kind": "parameter",
                "satisfied": bool(parameters[parameter]) == expected,
                "text": f"{parameter} == {str(expected).lower()}",
            }
        resolution = resolver.resolve(boolean_match.group("signal"))
        if resolution.status != "resolved" or not resolution.resolved:
            return unresolved_predicate_signal(boolean_match.group("signal"), resolution)
        expected = not bool(boolean_match.group("negated"))
        return {
            "resolved": True,
            "kind": "logged_signal",
            "signal": resolution.resolved,
            "op": "==",
            "value": expected,
            "text": f"{resolution.resolved} == {str(expected).lower()}",
        }
    return {"resolved": False, "reason": f"Unsupported source predicate: {predicate}"}


def parameter_name_for_accessor(value: str, inventory: dict[str, Any]) -> Optional[str]:
    compact = re.sub(r"\s+", "", value)
    compact = re.sub(r"\.get\(\)$", "", compact)
    parameters = inventory.get("parameters") or {}
    if compact in parameters:
        return compact
    normalized = compact.lstrip("_")
    if normalized.startswith("param_"):
        candidate = normalized[len("param_"):].upper()
        if candidate in parameters:
            return candidate
    if normalized.upper() in parameters and normalized.upper() == normalized:
        return normalized.upper()
    return None


def unresolved_predicate_signal(reference: str, resolution: VerificationSignalResolution) -> dict[str, Any]:
    return {
        "resolved": False,
        "reason": f"Predicate signal {reference} is {resolution.status}: {resolution.reason or resolution.candidates}",
    }


def split_top_level(expression: str, operator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    start = 0
    index = 0
    while index < len(expression):
        char = expression[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        elif depth == 0 and expression.startswith(operator, index):
            parts.append(expression[start:index])
            start = index + len(operator)
            index += len(operator) - 1
        index += 1
    parts.append(expression[start:])
    return parts


def strip_outer_parentheses(value: str) -> str:
    stripped = value.strip()
    while stripped.startswith("(") and stripped.endswith(")"):
        depth = 0
        balanced = True
        for index, char in enumerate(stripped):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0 and index != len(stripped) - 1:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        stripped = stripped[1:-1].strip()
    return stripped


def timeline_has_signal(timeline: list[dict[str, Any]], signal: str) -> bool:
    if "." not in signal:
        return False
    topic, field = signal.split(".", 1)
    return any(
        isinstance(event, dict)
        and event.get("topic") == topic
        and event.get("field") == field
        and isinstance(event.get("time_s"), (int, float))
        for event in timeline or []
    )


def resolve_branch_windows(
    candidate: MechanismCandidate,
    branch_id: str,
    predicates: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    inventory: dict[str, Any],
) -> list[WindowSpec]:
    available = available_log_intervals(timeline, inventory)
    if not available:
        return []
    requested = [
        (float(plot.start_s), float(plot.end_s))
        for plot in candidate.plot_requests
        if plot.start_s is not None and plot.end_s is not None and float(plot.end_s) > float(plot.start_s)
    ] or available
    intervals = intersect_intervals(available, requested)
    for predicate in predicates:
        intervals = intersect_intervals(intervals, timeline_predicate_intervals(timeline, predicate, available))
        if not intervals:
            break
    reason_parts = ["available log-data window"]
    if requested != available:
        reason_parts.append("requested window")
    if predicates:
        reason_parts.append("resolved source predicates")
    return [
        WindowSpec(
            name=f"{branch_id}_window_{index}",
            start_s=start,
            end_s=end,
            reason="Intersection of " + ", ".join(reason_parts) + ".",
        )
        for index, (start, end) in enumerate(intervals, start=1)
    ]


def available_log_intervals(timeline: list[dict[str, Any]], inventory: dict[str, Any]) -> list[tuple[float, float]]:
    times = [
        float(event["time_s"])
        for event in timeline or []
        if isinstance(event, dict) and isinstance(event.get("time_s"), (int, float))
    ]
    if times:
        start, end = min(times), max(times)
        return [(start, end if end > start else start + 0.001)]
    duration = inventory.get("duration_s")
    if isinstance(duration, (int, float)) and duration > 0:
        return [(0.0, float(duration))]
    return []


def timeline_predicate_intervals(
    timeline: list[dict[str, Any]],
    predicate: dict[str, Any],
    available: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    topic, field = predicate["signal"].split(".", 1)
    events = sorted(
        (
            float(event["time_s"]),
            event.get("value"),
        )
        for event in timeline or []
        if isinstance(event, dict)
        and event.get("topic") == topic
        and event.get("field") == field
        and isinstance(event.get("time_s"), (int, float))
    )
    if not events:
        return []
    result: list[tuple[float, float]] = []
    for available_start, available_end in available:
        current_value = None
        current_time = available_start
        for event_time, value in events:
            if event_time <= available_start:
                current_value = value
                continue
            if event_time >= available_end:
                break
            if current_value is not None and predicate_matches(current_value, predicate["op"], predicate["value"]):
                result.append((current_time, event_time))
            current_value = value
            current_time = event_time
        if current_value is not None and predicate_matches(current_value, predicate["op"], predicate["value"]):
            result.append((current_time, available_end))
    return result


def intersect_intervals(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    return [
        (max(left_start, right_start), min(left_end, right_end))
        for left_start, left_end in left
        for right_start, right_end in right
        if min(left_end, right_end) > max(left_start, right_start)
    ]


def predicate_matches(actual: Any, op: str, expected: Any) -> bool:
    if op == "isfinite":
        try:
            return math.isfinite(float(actual))
        except (TypeError, ValueError):
            return False
    if op == "bit_set":
        try:
            return int(actual) & int(expected) != 0
        except (TypeError, ValueError):
            return False
    if op == "bit_clear":
        try:
            return int(actual) & int(expected) == 0
        except (TypeError, ValueError):
            return False
    if op == "==":
        return actual == expected
    if op == "!=":
        return actual != expected
    try:
        if op == ">":
            return actual > expected
        if op == ">=":
            return actual >= expected
        if op == "<":
            return actual < expected
        if op == "<=":
            return actual <= expected
    except TypeError:
        return False
    return False


class SignalResolver:
    def __init__(self, inventory: dict[str, Any], output_bindings: Iterable[Any]) -> None:
        schema = load_px4_msg_schema(source_from_inventory(inventory))
        topic_fields = inventory.get("topic_fields") or {}
        available_topics = set(inventory.get("available_topics") or topic_fields)
        self.logged_signals = {
            f"{topic}.{field}"
            for topic, fields in topic_fields.items()
            for field in fields
        }
        self.schema_signals = {
            f"{topic}.{field}"
            for topic, fields in schema.items()
            for field in fields
        }
        aliases: dict[str, set[str]] = {}
        binding_suffix_aliases: dict[str, set[str]] = {}
        unavailable_aliases: dict[str, set[str]] = {}
        for signal in self.logged_signals:
            aliases.setdefault(normalize_symbol(signal), set()).add(signal)
        for binding in output_bindings:
            logged_signal = str(_get(binding, "logged_signal") or "")
            if not logged_signal:
                continue
            topic = logged_signal.split(".", 1)[0]
            binding_is_logged = logged_signal in self.logged_signals or (
                topic in available_topics and topic not in topic_fields
            )
            target_aliases = aliases if binding_is_logged else unavailable_aliases
            for value in (
                logged_signal,
                _get(binding, "source_symbol"),
                _get(binding, "target_symbol"),
                *assignment_path_symbols(_get(binding, "assignment_path") or []),
            ):
                normalized = normalize_symbol(str(value or ""))
                if normalized:
                    target_aliases.setdefault(normalized, set()).add(logged_signal)
                    parts = normalized.split(".")
                    for index in range(1, len(parts)):
                        binding_suffix_aliases.setdefault(".".join(parts[index:]), set()).add(logged_signal)
        self.aliases = aliases
        self.binding_suffix_aliases = binding_suffix_aliases
        self.unavailable_aliases = unavailable_aliases

    def is_known_signal_reference(self, reference: str) -> bool:
        normalized = normalize_symbol(reference)
        suffix = normalized.split(".", 1)[1] if "." in normalized else ""
        return bool(
            normalized in self.logged_signals
            or normalized in self.schema_signals
            or normalized in self.aliases
            or normalized in self.unavailable_aliases
            or suffix in self.binding_suffix_aliases
        )

    def resolve(self, reference: str) -> VerificationSignalResolution:
        normalized = normalize_symbol(reference)
        candidates = set(self.aliases.get(normalized, set()))
        if normalized in self.logged_signals:
            candidates.add(normalized)
        if not candidates and "." in normalized:
            suffix = normalized.split(".", 1)[1]
            candidates.update(self.binding_suffix_aliases.get(suffix, set()))
        ordered = sorted(candidates)
        if len(ordered) == 1:
            return VerificationSignalResolution(original=reference, status="resolved", resolved=ordered[0], candidates=ordered)
        if len(ordered) > 1:
            return VerificationSignalResolution(
                original=reference,
                status="ambiguous",
                candidates=ordered,
                reason="multiple deterministic logged-signal matches",
            )
        unavailable = sorted(self.unavailable_aliases.get(normalized, set()))
        if normalized in self.schema_signals or unavailable:
            return VerificationSignalResolution(
                original=reference,
                status="unresolved",
                candidates=unavailable or [normalized],
                reason="deterministic source/schema match is not present in the log",
            )
        return VerificationSignalResolution(
            original=reference,
            status="unresolved",
            reason="no deterministic logged output-binding match",
        )


def assignment_path_symbols(path: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for step in path:
        for key in ("source", "target", "source_symbol", "target_symbol", "expression"):
            value = step.get(key)
            if isinstance(value, str) and re.fullmatch(r"[_A-Za-z][_A-Za-z0-9]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)+", value):
                values.append(value)
    return values


def check_signal_availability(
    signals: list[str],
    available_topics: set[str],
    topic_fields: dict[str, Any],
    schema: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    available: list[str] = []
    missing: list[str] = []
    for signal in signals:
        if "." not in signal:
            (available if signal in available_topics else missing).append(signal)
            continue
        topic, field = signal.split(".", 1)
        fields = [*(topic_fields.get(topic) or []), *(schema.get(topic) or [])]
        if topic in available_topics and (
            topic not in topic_fields or field_or_flattened_prefix_present(field, fields)
        ):
            available.append(signal)
        else:
            missing.append(signal)
    return available, missing


def normalize_symbol(value: str) -> str:
    normalized = value.strip().replace("->", ".").replace("::", ".").replace(" ", "").strip("&*")
    normalized = re.sub(r"\[[^\]]+\]", "", normalized)
    parts = normalized.split(".")
    if parts and parts[0].startswith("_"):
        parts[0] = parts[0][1:]
    return ".".join(parts)


def looks_like_signal_reference(value: str) -> bool:
    return bool(re.fullmatch(r"_?[A-Za-z][A-Za-z0-9_]*(?:(?:\.|->)[A-Za-z_][A-Za-z0-9_]*)+", value))


def check_signal_references(check: RelationshipCheckSpec) -> list[str]:
    signals = [
        str(value)
        for value in (getattr(check, field, None) for field in CHECK_SIGNAL_FIELDS)
        if value and looks_like_signal_reference(str(value))
    ]
    for variable in check.variables:
        source = variable.get("source") if isinstance(variable, dict) else getattr(variable, "source", None)
        if source and looks_like_signal_reference(str(source)):
            signals.append(str(source))
    return dedupe(signals)


def stable_id(prefix: str, value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def parse_literal(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() and "." not in value else number


def dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    return dict(value) if isinstance(value, dict) else dict(vars(value))


def _get(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)
