from __future__ import annotations

from typing import Any, Optional

from flight_log_agent.models import AirframeContext
from flight_log_agent.px4.mechanism_source_profiler import ParameterPredicateRef
from flight_log_agent.px4.source_mechanism_models import (
    ParameterRequirement,
    SourceDiscoveryLogContext,
)


def build_source_discovery_log_context(
    inventory: dict[str, Any],
    airframe_context: Optional[AirframeContext] = None,
    mode_state_constraints: Optional[dict[str, Any]] = None,
) -> SourceDiscoveryLogContext:
    return SourceDiscoveryLogContext(
        parameters=dict(inventory.get("parameters") or {}),
        topic_fields={
            str(topic): [str(field) for field in fields]
            for topic, fields in (inventory.get("topic_fields") or {}).items()
        },
        available_topics=[str(topic) for topic in (inventory.get("available_topics") or [])],
        vehicle_type=getattr(airframe_context, "vehicle_type", None) if airframe_context else None,
        mode_state_constraints=mode_state_constraints or {},
    )


class ParameterFeasibilityGate:
    """
    Conservative gate for source-discovered parameter predicates.

    This gate only compares static parameter values against source predicates.
    It does not inspect signal time series, timestamps, plots, or final root
    cause confidence.
    """

    def evaluate(
        self,
        predicates: list[ParameterPredicateRef],
        log_context: SourceDiscoveryLogContext,
    ) -> list[ParameterRequirement]:
        requirements: list[ParameterRequirement] = []
        for predicate in predicates:
            if not predicate.name:
                requirements.append(self._unknown_requirement(predicate, "Parameter name could not be resolved."))
                continue

            actual_value = log_context.parameters.get(predicate.name)
            role = self._classify_parameter_role(predicate.name, predicate)
            gate_result = self._evaluate_predicate(predicate, actual_value, role)
            requirements.append(
                ParameterRequirement(
                    name=predicate.name,
                    role=role,
                    source_predicate=predicate.predicate,
                    actual_value=actual_value,
                    gate_result=gate_result,
                    effect=self._effect_text(predicate.name, role, gate_result),
                    source_file=predicate.file,
                    source_line=predicate.line,
                )
            )
        return requirements

    def _unknown_requirement(
        self,
        predicate: ParameterPredicateRef,
        effect: str,
    ) -> ParameterRequirement:
        return ParameterRequirement(
            name=predicate.name or "unknown",
            role="unknown",
            source_predicate=predicate.predicate,
            actual_value=None,
            gate_result="unknown",
            effect=effect,
            source_file=predicate.file,
            source_line=predicate.line,
        )

    def _evaluate_predicate(
        self,
        predicate: ParameterPredicateRef,
        actual_value: Any,
        role: str,
    ) -> str:
        if actual_value is None:
            return "unknown"
        if role != "branch_selector":
            return "verification_required"
        if not predicate.operator or predicate.compared_value is None:
            return "unknown"

        expected = _coerce_literal(predicate.compared_value)
        actual = _coerce_literal(actual_value)
        if expected is None or actual is None:
            return "unknown"

        try:
            satisfied = _compare_values(actual, predicate.operator, expected)
        except TypeError:
            return "unknown"
        return "satisfied" if satisfied else "contradicted"

    def _classify_parameter_role(
        self,
        name: str,
        predicate: ParameterPredicateRef,
    ) -> str:
        name_l = name.lower()
        predicate_l = predicate.predicate.lower()
        if any(token in name_l for token in ("enable", "circuit", "type", "mode", "sel", "switch")):
            return "branch_selector"
        if any(token in predicate_l for token in ("if", "switch", "case")):
            return "branch_selector"
        if any(token in name_l for token in ("rad", "alt", "dist", "speed", "min", "max", "lim", "thr")):
            return "threshold"
        if any(token in name_l for token in ("gain", "p_", "i_", "d_", "tc", "damp")):
            return "tuning_or_shaping"
        return "unknown"

    def _effect_text(
        self,
        name: str,
        role: str,
        gate_result: str,
    ) -> str:
        if gate_result == "contradicted":
            return f"{name} contradicts a source branch predicate and should eliminate or downrank that path."
        if gate_result == "satisfied":
            return f"{name} satisfies a source branch predicate for this path."
        if gate_result == "verification_required":
            return f"{name} affects source behavior, but dynamic log verification must evaluate the numeric effect."
        return f"{name} could not be resolved against the available static log parameters."


def _coerce_literal(value: Any) -> Any:
    if isinstance(value, (bool, int, float)):
        return value
    text = str(value).strip()
    if text.lower() == "true":
        return True
    if text.lower() == "false":
        return False
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text.strip('"')


def _compare_values(actual: Any, operator: str, expected: Any) -> bool:
    if operator == "==":
        return actual == expected
    if operator == "!=":
        return actual != expected
    if operator == ">":
        return actual > expected
    if operator == ">=":
        return actual >= expected
    if operator == "<":
        return actual < expected
    if operator == "<=":
        return actual <= expected
    raise TypeError(f"unsupported operator: {operator}")
