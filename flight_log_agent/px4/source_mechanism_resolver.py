from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

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
    SourceDiscoveryCandidateDraft,
    SourceDiscoveryDecision,
    SourceDiscoveryIterationPacket,
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

    async def discover(
        self,
        user_question: str,
        log_context: SourceDiscoveryLogContext,
        *,
        seed_queries: Optional[list[str]] = None,
        decide: Optional[Callable[[SourceDiscoveryIterationPacket], Awaitable[SourceDiscoveryDecision]]] = None,
        max_depth: int = 2,
        max_files_per_query: int = 8,
        max_total_files: int = 18,
        max_expansion_queries: int = 32,
        max_profile_files_per_iteration: int = 3,
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
        decision_notes: list[str] = []
        candidate_drafts: list[SourceDiscoveryCandidateDraft] = []
        relevant_files: list[str] = []

        for depth in range(max(max_depth, 1)):
            if not active_queries or len(visited_files) >= max_total_files:
                break

            hits = self.profiler.search_related_source_files(
                active_queries,
                max_files=max_files_per_query,
            )
            for hit in hits:
                existing = hits_by_file.get(hit.file)
                if existing is None or hit.score > existing.score:
                    hits_by_file[hit.file] = hit

            candidate_files = [
                hit.file for hit in hits
                if hit.file not in visited_files
            ]
            if not candidate_files:
                break

            selected_files = candidate_files[:max_profile_files_per_iteration]
            if decide is not None:
                search_decision = await decide(
                    self._build_search_iteration_packet(
                        user_question=user_question,
                        depth=depth,
                        active_queries=active_queries,
                        visited_files=visited_files,
                        candidate_hits=hits,
                        log_context=log_context,
                        prior_decision_notes=decision_notes,
                    )
                )
                decision_notes.extend(search_decision.notes)
                if search_decision.candidate_drafts:
                    candidate_drafts.extend(
                        self._filter_candidate_drafts_by_parameter_gate(
                            search_decision.candidate_drafts,
                            [],
                        )
                    )
                requested_files = [
                    path for path in search_decision.relevant_files
                    if path in candidate_files
                ]
                if requested_files:
                    selected_files = requested_files[:max_profile_files_per_iteration]
                if search_decision.expansion_queries:
                    active_queries = [
                        query for query in search_decision.expansion_queries
                        if query not in all_queries
                    ]
                    all_queries.extend(active_queries)
                    if search_decision.stop:
                        break
                    if not selected_files:
                        continue
                if search_decision.stop and not selected_files:
                    break

            new_files = []
            for file_path in selected_files:
                if file_path not in visited_files and len(visited_files) < max_total_files:
                    visited_files.append(file_path)
                    new_files.append(file_path)

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
            parameter_requirements = self.parameter_gate.evaluate(
                dedupe_parameter_predicates(parameter_predicates),
                log_context,
            )
            eliminated_parameter_paths = eliminated_branch_selector_requirements(parameter_requirements)

            expansions = self._build_expansion_queries(
                parameter_refs,
                published_topics + subscribed_topics,
                assigned_fields + read_fields,
                function_calls,
                max_queries=max_expansion_queries,
            )
            if decide is not None:
                decision = await decide(
                    self._build_iteration_packet(
                        user_question=user_question,
                        depth=depth,
                        active_queries=active_queries,
                        visited_files=visited_files,
                        new_files=new_files,
                        hits=list(hits_by_file.values()),
                        parameter_refs=parameter_refs,
                        published_topics=published_topics,
                        subscribed_topics=subscribed_topics,
                        assigned_fields=assigned_fields,
                        read_fields=read_fields,
                        function_calls=function_calls,
                        branch_conditions=branch_conditions,
                        parameter_predicates=parameter_predicates,
                        parameter_requirements=parameter_requirements,
                        eliminated_parameter_paths=eliminated_parameter_paths,
                        log_context=log_context,
                        prior_decision_notes=decision_notes,
                    )
                )
                relevant_files.extend([
                    path for path in decision.relevant_files
                    if path in visited_files
                ])
                candidate_drafts.extend(
                    self._filter_candidate_drafts_by_parameter_gate(
                        decision.candidate_drafts,
                        parameter_requirements,
                    )
                )
                decision_notes.extend(decision.notes)
                expansions = dedupe_keep_order([
                    *decision.expansion_queries,
                    *expansions,
                ])[:max_expansion_queries]
                if decision.stop:
                    all_queries.extend([
                        query for query in expansions
                        if query not in all_queries
                    ])
                    break

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
        surviving_parameter_requirements = [
            requirement for requirement in parameter_requirements
            if not is_contradicted_branch_selector(requirement)
        ]
        selected_files = dedupe_keep_order(relevant_files) or visited_files
        candidates = self._build_candidates(
            user_question=user_question,
            source_files=selected_files,
            hits=list(hits_by_file.values()),
            parameter_requirements=surviving_parameter_requirements,
            published_topics=dedupe_topic_refs(published_topics),
            subscribed_topics=dedupe_topic_refs(subscribed_topics),
            fields=dedupe_field_refs(assigned_fields + read_fields),
            branch_conditions=dedupe_branch_conditions(branch_conditions),
            log_context=log_context,
            candidate_drafts=candidate_drafts,
            decision_notes=decision_notes,
        )
        return SourceMechanismCandidateSet(
            candidates=candidates,
            expansion_queries=all_queries,
            unresolved_questions=[],
        )

    def _build_iteration_packet(
        self,
        *,
        user_question: str,
        depth: int,
        active_queries: list[str],
        visited_files: list[str],
        new_files: list[str],
        hits: list[SourceFileHit],
        parameter_refs: list[ParameterRef],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        assigned_fields: list[FieldRef],
        read_fields: list[FieldRef],
        function_calls: list[FunctionCallRef],
        branch_conditions: list[BranchConditionRef],
        parameter_predicates: list[ParameterPredicateRef],
        parameter_requirements: list[ParameterRequirement],
        eliminated_parameter_paths: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        prior_decision_notes: list[str],
    ) -> SourceDiscoveryIterationPacket:
        discovered_topics = dedupe_keep_order([
            ref.topic
            for ref in published_topics + subscribed_topics
            if ref.topic
        ])
        discovered_topics.extend([
            ref.topic
            for ref in assigned_fields + read_fields
            if ref.topic and ref.topic not in discovered_topics
        ])
        discovered_parameter_names = dedupe_keep_order([
            ref.name for ref in parameter_refs if ref.name
        ])
        discovered_parameter_names.extend([
            requirement.name
            for requirement in parameter_requirements
            if requirement.name and requirement.name != "unknown" and requirement.name not in discovered_parameter_names
        ])
        return SourceDiscoveryIterationPacket(
            user_question=user_question,
            depth=depth,
            active_queries=active_queries,
            visited_files=visited_files,
            new_files=new_files,
            source_profile={
                "related_files": compact_source_hits(hits, limit=8, max_matches_per_file=3),
                "referenced_parameters": compact_refs(dedupe_parameter_refs(parameter_refs), limit=40),
                "published_topics": compact_refs(dedupe_topic_refs(published_topics), limit=30),
                "subscribed_topics": compact_refs(dedupe_topic_refs(subscribed_topics), limit=30),
                "assigned_fields": compact_refs(dedupe_field_refs(assigned_fields), limit=60),
                "read_fields": compact_refs(dedupe_field_refs(read_fields), limit=60),
                "function_calls": compact_refs(dedupe_function_call_refs(function_calls), limit=80),
                "branch_conditions": compact_refs(dedupe_branch_conditions(branch_conditions), limit=80),
                "parameter_predicates": [
                    compact_ref(ref) for ref in dedupe_parameter_predicates(parameter_predicates)[:40]
                ],
            },
            parameter_requirements=parameter_requirements,
            static_log_context={
                "vehicle_type": log_context.vehicle_type,
                "mode_state_constraints": log_context.mode_state_constraints,
                "discovered_parameter_values": {
                    name: log_context.parameters.get(name)
                    for name in discovered_parameter_names
                    if name in log_context.parameters
                },
                "discovered_topic_fields": {
                    topic: log_context.topic_fields.get(topic, [])
                    for topic in discovered_topics
                    if topic in log_context.topic_fields
                },
                "available_discovered_topics": [
                    topic for topic in discovered_topics
                    if topic in log_context.available_topics
                ],
                "eliminated_parameter_paths": [
                    _safe_model_dump(requirement)
                    for requirement in eliminated_parameter_paths
                ],
            },
            prior_decision_notes=prior_decision_notes,
        )

    def _filter_candidate_drafts_by_parameter_gate(
        self,
        drafts: list[SourceDiscoveryCandidateDraft],
        parameter_requirements: list[ParameterRequirement],
    ) -> list[SourceDiscoveryCandidateDraft]:
        contradicted = {
            requirement.name
            for requirement in parameter_requirements
            if is_contradicted_branch_selector(requirement)
        }
        if not contradicted:
            return drafts
        return [
            draft for draft in drafts
            if not any(name in contradicted for name in draft.controlling_parameter_names)
        ]

    def _build_search_iteration_packet(
        self,
        *,
        user_question: str,
        depth: int,
        active_queries: list[str],
        visited_files: list[str],
        candidate_hits: list[SourceFileHit],
        log_context: SourceDiscoveryLogContext,
        prior_decision_notes: list[str],
    ) -> SourceDiscoveryIterationPacket:
        return SourceDiscoveryIterationPacket(
            user_question=user_question,
            depth=depth,
            active_queries=active_queries,
            visited_files=visited_files,
            new_files=[],
            source_profile={
                "stage": "search_hits_only",
                "related_files": compact_source_hits(candidate_hits, limit=12, max_matches_per_file=4),
            },
            parameter_requirements=[],
            static_log_context={
                "vehicle_type": log_context.vehicle_type,
                "mode_state_constraints": compact_mode_state_constraints(log_context.mode_state_constraints),
                "discovered_parameter_values": {},
                "discovered_topic_fields": {},
                "available_discovered_topics": [],
            },
            prior_decision_notes=prior_decision_notes,
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

    def _build_candidates(
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
        candidate_drafts: list[SourceDiscoveryCandidateDraft],
        decision_notes: list[str],
    ) -> list[SourceMechanismCandidate]:
        if candidate_drafts:
            return [
                self._build_candidate_from_draft(
                    draft,
                    fallback_user_question=user_question,
                    fallback_source_files=source_files,
                    fallback_hits=hits,
                    parameter_requirements=parameter_requirements,
                    published_topics=published_topics,
                    subscribed_topics=subscribed_topics,
                    fields=fields,
                    branch_conditions=branch_conditions,
                    log_context=log_context,
                    decision_notes=decision_notes,
                )
                for draft in candidate_drafts
            ]
        return [
            self._build_candidate(
                user_question=user_question,
                source_files=source_files,
                hits=hits,
                parameter_requirements=parameter_requirements,
                published_topics=published_topics,
                subscribed_topics=subscribed_topics,
                fields=fields,
                branch_conditions=branch_conditions,
                log_context=log_context,
                decision_notes=decision_notes,
            )
        ]

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
        decision_notes: list[str],
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
                *decision_notes,
            ],
        )

    def _build_candidate_from_draft(
        self,
        draft: SourceDiscoveryCandidateDraft,
        *,
        fallback_user_question: str,
        fallback_source_files: list[str],
        fallback_hits: list[SourceFileHit],
        parameter_requirements: list[ParameterRequirement],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        fields: list[FieldRef],
        branch_conditions: list[BranchConditionRef],
        log_context: SourceDiscoveryLogContext,
        decision_notes: list[str],
    ) -> SourceMechanismCandidate:
        source_files = draft.source_files or fallback_source_files
        required_parameters = set(draft.controlling_parameter_names)
        controlling_parameters = [
            requirement for requirement in parameter_requirements
            if not required_parameters or requirement.name in required_parameters
        ]
        relevant_fields = [
            SourceFieldRef(
                field=field.field,
                topic=field.topic,
                variable=field.variable,
                source_file=field.file,
                source_line=field.line,
            )
            for field in fields
            if not draft.relevant_signals
            or (field.topic and f"{field.topic}.{field.field}" in draft.relevant_signals)
        ]
        fallback = self._build_candidate(
            user_question=fallback_user_question,
            source_files=source_files,
            hits=fallback_hits,
            parameter_requirements=controlling_parameters,
            published_topics=published_topics,
            subscribed_topics=subscribed_topics,
            fields=fields,
            branch_conditions=branch_conditions,
            log_context=log_context,
            decision_notes=decision_notes,
        )
        return SourceMechanismCandidate(
            title=draft.title or fallback.title,
            source_mechanism=draft.source_mechanism or fallback.source_mechanism,
            source_chain=draft.source_chain or fallback.source_chain,
            source_files=source_files,
            controlling_parameters=controlling_parameters,
            published_topics=fallback.published_topics,
            subscribed_topics=fallback.subscribed_topics,
            relevant_fields=relevant_fields or fallback.relevant_fields,
            branch_conditions=draft.branch_conditions or fallback.branch_conditions,
            expected_log_signature=draft.expected_log_signature or fallback.expected_log_signature,
            required_log_evidence=draft.required_log_evidence or fallback.required_log_evidence,
            contradiction_checks=draft.contradiction_checks or fallback.contradiction_checks,
            source_confidence=draft.source_confidence,
            resolver_notes=[
                "Source candidate was drafted by the source-discovery decision agent.",
                "No time-series signal comparison, event timestamp proof, plots, or final root-cause confidence were computed.",
                *draft.resolver_notes,
                *decision_notes,
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


def dedupe_parameter_refs(refs: list[ParameterRef]) -> list[ParameterRef]:
    seen = set()
    out: list[ParameterRef] = []
    for ref in refs:
        key = (ref.name, ref.member, ref.file, ref.line, ref.access_pattern)
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


def dedupe_function_call_refs(refs: list[FunctionCallRef]) -> list[FunctionCallRef]:
    seen = set()
    out: list[FunctionCallRef] = []
    for ref in refs:
        key = (ref.name, ref.receiver, ref.file, ref.line)
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


def is_contradicted_branch_selector(requirement: ParameterRequirement) -> bool:
    return requirement.role == "branch_selector" and requirement.gate_result == "contradicted"


def eliminated_branch_selector_requirements(
    requirements: list[ParameterRequirement],
) -> list[ParameterRequirement]:
    return [
        requirement for requirement in requirements
        if is_contradicted_branch_selector(requirement)
    ]


def compact_source_hits(
    hits: list[SourceFileHit],
    *,
    limit: int,
    max_matches_per_file: int,
) -> list[dict[str, Any]]:
    compact = []
    for hit in sorted(hits, key=lambda item: (-item.score, item.file))[:limit]:
        compact.append({
            "file": hit.file,
            "score": round(hit.score, 3),
            "matched_queries": list(hit.matched_queries)[:8],
            "matches": [
                {
                    "file": match.file,
                    "line": match.line,
                    "query": match.query,
                    "text": truncate_text(match.text, 240),
                }
                for match in hit.matches[:max_matches_per_file]
            ],
        })
    return compact


def compact_refs(refs: list[Any], *, limit: int) -> list[dict[str, Any]]:
    return [compact_ref(ref) for ref in refs[:limit]]


def compact_ref(ref: Any) -> dict[str, Any]:
    data = _safe_model_dump(ref)
    if not isinstance(data, dict) and hasattr(ref, "__dict__"):
        data = dict(ref.__dict__)
    if not isinstance(data, dict):
        return {"value": truncate_text(str(data), 240)}
    compact = {}
    for key, value in data.items():
        if value is None:
            continue
        if key == "evidence":
            compact[key] = truncate_text(str(value), 240)
        elif isinstance(value, str):
            compact[key] = truncate_text(value, 240)
        else:
            compact[key] = value
    return compact


def compact_mode_state_constraints(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        "observed_values": value.get("observed_values", {}),
        "omitted_event_count": value.get("omitted_event_count", 0),
    }


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _safe_model_dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_safe_model_dump(item) for item in value]
    if isinstance(value, dict):
        return {key: _safe_model_dump(item) for key, item in value.items()}
    return value


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
