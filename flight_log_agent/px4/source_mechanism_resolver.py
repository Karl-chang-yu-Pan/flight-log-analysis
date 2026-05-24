from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from flight_log_agent.models import AirframeContext, CodeRef
from flight_log_agent.px4.mechanism_source_profiler import (
    BranchConditionRef,
    FieldRef,
    FunctionCallRef,
    MechanismSourceProfiler,
    ParameterPredicateRef,
    ParameterRef,
    SourceFileHit,
    TopicRef,
)
from flight_log_agent.px4.source_mechanism_models import (
    ParameterRequirement,
    SourceDiscoveryLogContext,
    SourceFieldRef,
    SourceMechanismCandidate,
    SourceMechanismCandidateSet,
    TopicFieldRef,
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


class SourceMechanismResolver:
    """
    Deterministic scaffold for iterative source-mechanism discovery.

    The resolver may use static log facts such as parameter values and topic
    inventory to narrow source branches, but it does not inspect dynamic
    time-series samples or produce final root-cause confidence.
    """

    def __init__(
        self,
        source_path: str | Path,
        *,
        profiler: Optional[MechanismSourceProfiler] = None,
        parameter_gate: Optional["ParameterFeasibilityGate"] = None,
    ) -> None:
        self.source_path = Path(source_path)
        self.profiler = profiler or MechanismSourceProfiler(self.source_path)
        self.parameter_gate = parameter_gate or ParameterFeasibilityGate()

    def discover(
        self,
        user_question: str,
        log_context: SourceDiscoveryLogContext,
        *,
        seed_queries: Optional[list[str]] = None,
        max_depth: int = 2,
        max_files_per_query: int = 8,
        max_total_files: int = 18,
        max_expansion_queries: int = 32,
    ) -> SourceMechanismCandidateSet:
        active_queries = dedupe_keep_order(seed_queries or self._seed_queries(user_question))
        all_queries = list(active_queries)
        visited_files: list[str] = []
        hits_by_file: dict[str, SourceFileHit] = {}
        published_topics: list[TopicRef] = []
        subscribed_topics: list[TopicRef] = []
        parameter_refs: list[ParameterRef] = []
        assigned_fields: list[FieldRef] = []
        read_fields: list[FieldRef] = []
        function_calls: list[FunctionCallRef] = []
        branch_conditions: list[BranchConditionRef] = []
        parameter_predicates: list[ParameterPredicateRef] = []

        for _depth in range(max(max_depth, 1)):
            if not active_queries or len(visited_files) >= max_total_files:
                break

            hits = self.profiler.search_related_source_files(
                active_queries,
                max_files=max_files_per_query,
            )
            new_files = []
            for hit in hits:
                existing = hits_by_file.get(hit.file)
                if existing is None or hit.score > existing.score:
                    hits_by_file[hit.file] = hit
                if hit.file not in visited_files and len(visited_files) < max_total_files:
                    visited_files.append(hit.file)
                    new_files.append(hit.file)

            if not new_files:
                break

            uorb = self.profiler.extract_uorb_io_from_source(new_files)
            published_topics.extend(uorb["published_topics"])
            subscribed_topics.extend(uorb["subscribed_topics"])
            parameter_refs.extend(self.profiler.extract_params_from_source(new_files))
            assigned_fields.extend(self.profiler.extract_assigned_fields_from_source(new_files))
            read_fields.extend(self.profiler.extract_read_fields_from_source(new_files))
            function_calls.extend(self.profiler.extract_function_calls_from_source(new_files))
            branch_conditions.extend(self.profiler.extract_branch_conditions_from_source(new_files))
            parameter_predicates.extend(self.profiler.extract_parameter_predicates_from_source(new_files))

            expansions = self._build_expansion_queries(
                parameter_refs,
                published_topics + subscribed_topics,
                assigned_fields + read_fields,
                function_calls,
                max_queries=max_expansion_queries,
            )
            active_queries = [
                query for query in expansions
                if query not in all_queries
            ]
            all_queries.extend(active_queries)

        if not visited_files:
            return SourceMechanismCandidateSet(
                candidates=[],
                expansion_queries=all_queries,
                unresolved_questions=["No PX4 source files matched the discovery seed queries."],
            )

        parameter_requirements = self.parameter_gate.evaluate(
            dedupe_parameter_predicates(parameter_predicates),
            log_context,
        )
        candidate = self._build_candidate(
            user_question=user_question,
            source_files=visited_files,
            hits=list(hits_by_file.values()),
            parameter_requirements=parameter_requirements,
            published_topics=dedupe_topic_refs(published_topics),
            subscribed_topics=dedupe_topic_refs(subscribed_topics),
            fields=dedupe_field_refs(assigned_fields + read_fields),
            branch_conditions=dedupe_branch_conditions(branch_conditions),
            log_context=log_context,
        )
        return SourceMechanismCandidateSet(
            candidates=[candidate],
            expansion_queries=all_queries,
            unresolved_questions=[],
        )

    def _seed_queries(self, user_question: str) -> list[str]:
        queries = [user_question.strip()]
        queries.extend(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", user_question))
        for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{5,}\b", user_question):
            if token.lower() not in {"explain", "source", "mechanism"}:
                queries.append(token)
        return dedupe_keep_order([query for query in queries if query])

    def _build_expansion_queries(
        self,
        parameters: list[ParameterRef],
        topics: list[TopicRef],
        fields: list[FieldRef],
        function_calls: list[FunctionCallRef],
        *,
        max_queries: int,
    ) -> list[str]:
        queries: list[str] = []
        queries.extend(ref.name for ref in parameters if ref.name)
        queries.extend(ref.topic for ref in topics if ref.topic)
        queries.extend(ref.field for ref in fields if ref.field)
        queries.extend(ref.name for ref in function_calls if self._is_relevant_function_name(ref.name))
        return dedupe_keep_order([query for query in queries if query])[:max_queries]

    def _build_candidate(
        self,
        *,
        user_question: str,
        source_files: list[str],
        hits: list[SourceFileHit],
        parameter_requirements: list[ParameterRequirement],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        fields: list[FieldRef],
        branch_conditions: list[BranchConditionRef],
        log_context: SourceDiscoveryLogContext,
    ) -> SourceMechanismCandidate:
        source_chain = self._source_chain_from_hits(hits)
        relevant_fields = [
            SourceFieldRef(
                field=field.field,
                topic=field.topic,
                variable=field.variable,
                source_file=field.file,
                source_line=field.line,
            )
            for field in fields
        ]
        expected_log_signature = self._expected_log_signature(fields, published_topics, subscribed_topics)
        required_log_evidence = self._required_log_evidence(fields, published_topics, subscribed_topics, log_context)
        contradiction_checks = [
            requirement.effect
            for requirement in parameter_requirements
            if requirement.gate_result == "contradicted"
        ]
        title = self._candidate_title(user_question, source_files)
        return SourceMechanismCandidate(
            title=title,
            source_mechanism=(
                "Source discovery found related PX4 files, static branches, uORB interfaces, "
                "and parameter predicates. Dynamic log verification is still required."
            ),
            source_chain=source_chain,
            source_files=source_files,
            controlling_parameters=parameter_requirements,
            published_topics=[
                TopicFieldRef(topic=ref.topic, source_file=ref.file, source_line=ref.line)
                for ref in published_topics
            ],
            subscribed_topics=[
                TopicFieldRef(topic=ref.topic, source_file=ref.file, source_line=ref.line)
                for ref in subscribed_topics
            ],
            relevant_fields=relevant_fields,
            branch_conditions=[ref.condition for ref in branch_conditions],
            expected_log_signature=expected_log_signature,
            required_log_evidence=required_log_evidence,
            contradiction_checks=contradiction_checks,
            source_confidence=self._source_confidence(source_files, source_chain, parameter_requirements),
            resolver_notes=[
                "Source resolver used only static source facts and static log inventory/parameters.",
                "No time-series signal comparison, event timestamp proof, plots, or final root-cause confidence were computed.",
            ],
        )

    def _source_chain_from_hits(self, hits: list[SourceFileHit]) -> list[CodeRef]:
        refs: list[CodeRef] = []
        for hit in sorted(hits, key=lambda item: (-item.score, item.file)):
            if not hit.matches:
                refs.append(CodeRef(file=hit.file, explanation="Related source file selected by source search."))
                continue
            for match in hit.matches[:2]:
                refs.append(
                    CodeRef(
                        file=match.file,
                        start_line=match.line,
                        snippet=match.text,
                        explanation=f"Matched source discovery query: {match.query}",
                    )
                )
        return refs[:12]

    def _expected_log_signature(
        self,
        fields: list[FieldRef],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
    ) -> list[str]:
        signatures = []
        for field in fields:
            if field.topic:
                signatures.append(f"Verify later whether {field.topic}.{field.field} follows the source-expected behavior.")
        for topic in published_topics:
            signatures.append(f"Verify later whether published topic {topic.topic} is present and behaviorally relevant.")
        for topic in subscribed_topics:
            signatures.append(f"Verify later whether subscribed topic {topic.topic} provides the required source inputs.")
        return dedupe_keep_order(signatures)[:24]

    def _required_log_evidence(
        self,
        fields: list[FieldRef],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        log_context: SourceDiscoveryLogContext,
    ) -> list[str]:
        evidence = []
        for field in fields:
            if field.topic and field.field in log_context.topic_fields.get(field.topic, []):
                evidence.append(f"Fetch time-series for {field.topic}.{field.field}.")
            elif field.topic:
                evidence.append(f"Check whether {field.topic}.{field.field} is logged before verification.")
        for topic in published_topics + subscribed_topics:
            if topic.topic in log_context.available_topics:
                evidence.append(f"Use logged topic {topic.topic} for later dynamic verification.")
            else:
                evidence.append(f"Topic {topic.topic} is source-relevant but was not listed in the log inventory.")
        return dedupe_keep_order(evidence)[:24]

    def _candidate_title(self, user_question: str, source_files: list[str]) -> str:
        if source_files:
            return f"Source mechanism path for {Path(source_files[0]).stem}: {user_question}"
        return f"Source mechanism path: {user_question}"

    def _source_confidence(
        self,
        source_files: list[str],
        source_chain: list[CodeRef],
        parameter_requirements: list[ParameterRequirement],
    ) -> str:
        if len(source_files) >= 2 and source_chain and parameter_requirements:
            return "medium"
        if source_chain:
            return "low"
        return "low"

    @staticmethod
    def _is_relevant_function_name(name: str) -> bool:
        lowered = name.lower()
        return not (
            lowered.startswith("orb_")
            or lowered in {"get", "update", "publish", "copy", "min", "max", "abs"}
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


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def dedupe_topic_refs(refs: list[TopicRef]) -> list[TopicRef]:
    seen = set()
    out: list[TopicRef] = []
    for ref in refs:
        key = (ref.topic, ref.direction, ref.file, ref.line, ref.variable)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def dedupe_field_refs(refs: list[FieldRef]) -> list[FieldRef]:
    seen = set()
    out: list[FieldRef] = []
    for ref in refs:
        key = (ref.topic, ref.variable, ref.field, ref.file, ref.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def dedupe_branch_conditions(refs: list[BranchConditionRef]) -> list[BranchConditionRef]:
    seen = set()
    out: list[BranchConditionRef] = []
    for ref in refs:
        key = (ref.kind, ref.condition, ref.file, ref.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def dedupe_parameter_predicates(refs: list[ParameterPredicateRef]) -> list[ParameterPredicateRef]:
    seen = set()
    out: list[ParameterPredicateRef] = []
    for ref in refs:
        key = (ref.name, ref.member, ref.predicate, ref.file, ref.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out
