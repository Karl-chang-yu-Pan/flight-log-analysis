from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from flight_log_agent.models import (
    AirframeContext,
    CodeRef,
    RelationshipCheckSpec,
    SourceOutputBindingRecord,
)
from flight_log_agent.utils import compare
from flight_log_agent.px4.mechanism_source_profiler import (
    BranchConditionRef,
    FieldRef,
    FunctionCallRef,
    HelperExpressionRef,
    MechanismSourceProfiler,
    ParameterPredicateRef,
    ParameterRef,
    SourceAssignmentRef,
    SourceFileHit,
    TopicRef,
    split_top_level_args,
    substitute_expression_symbols,
)
from flight_log_agent.px4.msg_schema import is_valid_topic_field, load_px4_msg_schema, normalize_px4_enum_value
from flight_log_agent.px4 import discovery_frontier as df_mod
from flight_log_agent.px4.source_mechanism_models import (
    ParameterRequirement,
    SourceBackedParameterPredicate,
    SourceBackedVerificationCheck,
    SourceDiscoveryCandidateDraft,
    SourceDiscoveryDecision,
    SourceDiscoveryIterationPacket,
    SourceDiscoveryLogContext,
    SourceFieldRef,
    SourceMechanismBranchGroup,
    SourceMechanismCandidate,
    SourceMechanismCandidateSet,
    SourceSnippet,
    TopicFieldRef,
)
from flight_log_agent.px4.source_snapshot import SourceHandle, SourceInput, source_handle


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
        source_path: SourceInput,
        *,
        profiler: Optional[MechanismSourceProfiler] = None,
        parameter_gate: Optional["ParameterFeasibilityGate"] = None,
    ) -> None:
        source = source_handle(source_path)
        if source is None:
            raise FileNotFoundError("PX4 source is unavailable.")
        self.source: SourceHandle = source
        self.profiler = profiler or MechanismSourceProfiler(source)
        self.parameter_gate = parameter_gate or ParameterFeasibilityGate()

    async def discover(
        self,
        user_question: str,
        log_context: SourceDiscoveryLogContext,
        *,
        seed_queries: Optional[list[str]] = None,
        cached_mechanism_seeds: Optional[list[dict[str, Any]]] = None,
        decide: Optional[Callable[[SourceDiscoveryIterationPacket], Awaitable[SourceDiscoveryDecision]]] = None,
        max_depth: int = 2,
        max_files_per_query: int = 8,
        max_total_files: int = 18,
        max_expansion_queries: int = 32,
        max_profile_files_per_iteration: int = 3,
    ) -> SourceMechanismCandidateSet:
        compact_seed_candidates = compact_cached_mechanism_seeds(cached_mechanism_seeds or [])
        active_queries = dedupe_keep_order([
            *(seed_queries or self._seed_queries(user_question)),
            *cached_mechanism_seed_queries(compact_seed_candidates),
        ])
        all_queries = list(active_queries)
        visited_files: list[str] = []
        hits_by_file: dict[str, SourceFileHit] = {}
        published_topics: list[TopicRef] = []
        subscribed_topics: list[TopicRef] = []
        parameter_refs: list[ParameterRef] = []
        assigned_fields: list[FieldRef] = []
        read_fields: list[FieldRef] = []
        source_assignments: list[SourceAssignmentRef] = []
        function_calls: list[FunctionCallRef] = []
        helper_expressions: list[HelperExpressionRef] = []
        branch_conditions: list[BranchConditionRef] = []
        parameter_predicates: list[ParameterPredicateRef] = []
        decision_notes: list[str] = []
        candidate_drafts: list[SourceDiscoveryCandidateDraft] = []
        relevant_files: list[str] = []
        # Bounded-frontier run bookkeeping (spec §8-§10, §16, §17).
        # Canonical store + novelty baseline + fallback/mass
        # accounting. Traversal state stays in the loop locals below.
        frontier_state = df_mod.DiscoveryRoundState()
        frontier_accounting: dict[str, Any] = {
            "rounds": 0,
            "bounded_rounds": 0,
            "fallback_count": 0,
            "fallback_bytes": 0,
            "lookup_complete": 0,
            "lookup_dropped_unknown": 0,
            "lookup_unavailable": 0,
            "lookup_exhausted": 0,
            "iteration_dropped_files": 0,
            "search_dropped_files": 0,
            "total_avoided_bytes": 0,
            "unavailable_rounds": 0,
        }

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
                requested_file for requested_file in self._requested_source_files(active_queries)
                if requested_file not in visited_files
            ]
            candidate_files.extend([
                hit.file for hit in hits
                if hit.file not in visited_files and hit.file not in candidate_files
            ])
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
                        cached_mechanism_seeds=compact_seed_candidates,
                    )
                )
                decision_notes.extend(search_decision.notes)
                if search_decision.candidate_drafts:
                    candidate_drafts.extend(
                        self._filter_candidate_drafts_by_parameter_gate(
                            search_decision.candidate_drafts,
                            [],
                            log_context,
                        )
                    )
                requested_files, search_dropped = df_mod.validate_decision_files(
                    search_decision.relevant_files,
                    eligible_files=candidate_files,
                )
                frontier_accounting["search_dropped_files"] = (
                    frontier_accounting.get("search_dropped_files", 0) + search_dropped
                )
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
            relevance_terms = dedupe_keep_order([user_question, *all_queries, *selected_files])
            assigned_fields.extend(
                self.profiler.extract_assigned_fields_from_source(
                    new_files,
                    relevance_terms=relevance_terms,
                )
            )
            read_fields.extend(
                self.profiler.extract_read_fields_from_source(
                    new_files,
                    relevance_terms=relevance_terms,
                )
            )
            source_assignments.extend(self.profiler.extract_source_assignments_from_source(new_files))
            function_calls.extend(self.profiler.extract_function_calls_from_source(new_files))
            helper_expressions.extend(
                self.profiler.extract_helper_expressions_recursive(
                    new_files,
                    helper_names=[ref.name for ref in function_calls],
                )
            )
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
                branch_conditions,
                max_queries=max_expansion_queries,
            )
            if decide is not None:
                frontier_accounting["rounds"] = (
                    frontier_accounting.get("rounds", 0) + 1
                )
                observed = df_mod.observe_discovery_round(
                    frontier_state,
                    round_no=depth,
                    assignments=source_assignments,
                    calls=function_calls,
                    helpers=helper_expressions,
                    branches=branch_conditions,
                    predicates=parameter_predicates,
                    topics=published_topics + subscribed_topics,
                    fields=assigned_fields + read_fields,
                    parameter_refs=parameter_refs,
                    requirements=parameter_requirements,
                    files=new_files,
                )
                full_packet = self._build_iteration_packet(
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
                    source_assignments=source_assignments,
                    function_calls=function_calls,
                    helper_expressions=helper_expressions,
                    branch_conditions=branch_conditions,
                    parameter_predicates=parameter_predicates,
                    parameter_requirements=parameter_requirements,
                    eliminated_parameter_paths=eliminated_parameter_paths,
                    log_context=log_context,
                    prior_decision_notes=decision_notes,
                    cached_mechanism_seeds=compact_seed_candidates,
                )
                try:
                    bounded_packet, round_acct = self._build_bounded_round_packet(
                        full_packet,
                        state=frontier_state,
                        observed=observed,
                        depth=depth,
                        max_depth=max_depth,
                        max_total_files=max_total_files,
                        active_queries=active_queries,
                        visited_files=visited_files,
                        new_files=new_files,
                        hits=list(hits_by_file.values()),
                        parameter_refs=parameter_refs,
                        published_topics=published_topics,
                        subscribed_topics=subscribed_topics,
                        assigned_fields=assigned_fields,
                        read_fields=read_fields,
                        source_assignments=source_assignments,
                        function_calls=function_calls,
                        helper_expressions=helper_expressions,
                        branch_conditions=branch_conditions,
                        parameter_predicates=parameter_predicates,
                        parameter_requirements=parameter_requirements,
                        log_context=log_context,
                        prior_decision_notes=decision_notes,
                        cached_mechanism_seeds=compact_seed_candidates,
                    )
                    packet_to_send = bounded_packet
                    if round_acct.get("unavailable"):
                        frontier_accounting["unavailable_rounds"] = (
                            frontier_accounting.get("unavailable_rounds", 0) + 1
                        )
                    else:
                        frontier_accounting["bounded_rounds"] = (
                            frontier_accounting.get("bounded_rounds", 0) + 1
                        )
                    frontier_accounting["total_avoided_bytes"] = (
                        frontier_accounting.get("total_avoided_bytes", 0)
                        + round_acct["mass"]["retransmission_avoided_bytes"]
                    )
                except df_mod.BoundedPacketError as exc:
                    fell_back = self._apply_round_fallback(
                        full_packet,
                        state=frontier_state,
                        accounting=frontier_accounting,
                        reason=exc.reason,
                    )
                    packet_to_send = fell_back
                decision = await decide(packet_to_send)
                expansions, break_loop = await self._consume_decision_with_lookup(
                    decision,
                    packet_to_send=packet_to_send,
                    state=frontier_state,
                    accounting=frontier_accounting,
                    decide=decide,
                    visited_files=visited_files,
                    parameter_requirements=parameter_requirements,
                    log_context=log_context,
                    decision_notes=decision_notes,
                    candidate_drafts=candidate_drafts,
                    relevant_files=relevant_files,
                    expansions=expansions,
                    max_expansion_queries=max_expansion_queries,
                    all_queries=all_queries,
                    depth=depth,
                )
                if break_loop:
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
                frontier_accounting=dict(frontier_accounting),
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
        output_bindings = source_output_binding_records(
            dedupe_source_assignment_refs(source_assignments),
            dedupe_function_call_refs(function_calls),
        )
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
            helper_expressions=dedupe_helper_expression_refs(helper_expressions),
            source_assignments=dedupe_source_assignment_refs(source_assignments),
            function_calls=dedupe_function_call_refs(function_calls),
            output_bindings=output_bindings,
        )
        return SourceMechanismCandidateSet(
            candidates=candidates,
            expansion_queries=all_queries,
            unresolved_questions=[],
            output_bindings=output_bindings,
            helper_expressions=[
                ref.model_dump(exclude_none=True)
                for ref in dedupe_helper_expression_refs(helper_expressions)
            ],
            source_assignments=[
                ref.model_dump(exclude_none=True)
                for ref in dedupe_source_assignment_refs(source_assignments)
            ],
            frontier_accounting={
                **frontier_accounting,
                "p0_mass": df_mod.p0_mass_fragment(
                    frontier_state.mass_log[-1]
                    if frontier_state.mass_log else {}
                ),
            },
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
        source_assignments: list[SourceAssignmentRef],
        function_calls: list[FunctionCallRef],
        helper_expressions: list[HelperExpressionRef],
        branch_conditions: list[BranchConditionRef],
        parameter_predicates: list[ParameterPredicateRef],
        parameter_requirements: list[ParameterRequirement],
        eliminated_parameter_paths: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        prior_decision_notes: list[str],
        cached_mechanism_seeds: list[dict[str, Any]],
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
        requirement_parameter_names = {
            requirement.name
            for requirement in parameter_requirements
            if requirement.name and requirement.name != "unknown"
        }
        published_topic_names = dedupe_keep_order([
            ref.topic for ref in published_topics if ref.topic
        ])
        subscribed_topic_names = dedupe_keep_order([
            ref.topic for ref in subscribed_topics if ref.topic
        ])
        source_profile = {
            "related_files": compact_source_hits(hits, limit=8, max_matches_per_file=3),
            "referenced_parameters": compact_refs(dedupe_parameter_refs(parameter_refs), limit=40),
            "published_topic_names": published_topic_names[:40],
            "subscribed_topic_names": subscribed_topic_names[:40],
            "published_topics": compact_refs(dedupe_topic_refs(published_topics), limit=30),
            "subscribed_topics": compact_refs(dedupe_topic_refs(subscribed_topics), limit=30),
            "assigned_fields": compact_refs(dedupe_field_refs(assigned_fields), limit=60),
            "read_fields": compact_refs(dedupe_field_refs(read_fields), limit=60),
            "source_assignments": compact_refs(dedupe_source_assignment_refs(source_assignments), limit=100),
            "function_calls": compact_refs(dedupe_function_call_refs(function_calls), limit=80),
            "helper_expressions": compact_refs(dedupe_helper_expression_refs(helper_expressions), limit=40),
            "expression_verification_candidates": source_discovery_expression_verification_candidates(
                dedupe_helper_expression_refs(helper_expressions),
                source_assignments=dedupe_source_assignment_refs(source_assignments),
                function_calls=dedupe_function_call_refs(function_calls),
                source_path=self.source,
                limit=40,
            ),
            "branch_conditions": compact_refs(dedupe_branch_conditions(branch_conditions), limit=80),
            "parameter_predicates": [
                compact_ref(ref) for ref in dedupe_parameter_predicates(parameter_predicates)[:40]
            ],
            "source_snippets": [
                _safe_model_dump(snippet)
                for snippet in self._source_snippets_for_files(
                    new_files,
                    active_queries=active_queries,
                    hits=hits,
                )
            ],
        }
        if cached_mechanism_seeds:
            source_profile["cached_mechanism_seeds"] = cached_mechanism_seeds

        return SourceDiscoveryIterationPacket(
            user_question=user_question,
            depth=depth,
            active_queries=active_queries,
            visited_files=visited_files,
            new_files=new_files,
            source_profile=source_profile,
            parameter_requirements=parameter_requirements,
            static_log_context={
                "vehicle_type": log_context.vehicle_type,
                "mode_state_constraints": log_context.mode_state_constraints,
                "discovered_parameter_values": {
                    name: log_context.parameters.get(name)
                    for name in discovered_parameter_names
                    if name in log_context.parameters and name not in requirement_parameter_names
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
        log_context: SourceDiscoveryLogContext,
    ) -> list[SourceDiscoveryCandidateDraft]:
        cited_drafts = [self._drop_uncited_agent_facts(draft) for draft in drafts]
        contradicted = {
            requirement.name
            for requirement in parameter_requirements
            if is_contradicted_branch_selector(requirement)
        }
        for draft in cited_drafts:
            agent_requirements = self.parameter_gate.evaluate_source_predicates(
                draft.interpreted_parameter_predicates,
                log_context,
            )
            contradicted.update(
                requirement.name
                for requirement in agent_requirements
                if is_contradicted_branch_selector(requirement)
            )
        if not contradicted:
            return cited_drafts
        return [
            self._prune_contradicted_branch_facts(draft, contradicted)
            for draft in cited_drafts
        ]

    def _prune_contradicted_branch_facts(
        self,
        draft: SourceDiscoveryCandidateDraft,
        contradicted: set[str],
    ) -> SourceDiscoveryCandidateDraft:
        pruned_names = [
            name for name in draft.controlling_parameter_names
            if name in contradicted
        ]
        kept_predicates = [
            predicate for predicate in draft.interpreted_parameter_predicates
            if predicate.name not in contradicted
        ]
        kept_checks = [
            check for check in draft.verification_checks
            if (check.check.parameter or "") not in contradicted
        ]
        kept_groups = [
            self._prune_contradicted_branch_group(group, contradicted)
            for group in draft.branch_groups
            if not any(name in contradicted for name in group.controlling_parameter_names)
            and not any(
                predicate.name in contradicted
                for predicate in group.interpreted_parameter_predicates
            )
        ]
        notes = list(draft.resolver_notes)
        if pruned_names or len(kept_predicates) != len(draft.interpreted_parameter_predicates):
            notes.append(
                "Resolver pruned contradicted branch-selector facts without discarding the full mechanism: "
                + ", ".join(sorted(set(pruned_names) | {
                    predicate.name
                    for predicate in draft.interpreted_parameter_predicates
                    if predicate.name in contradicted
                }))
                + "."
            )
        return draft.model_copy(
            update={
                "controlling_parameter_names": [
                    name for name in draft.controlling_parameter_names
                    if name not in contradicted
                ],
                "interpreted_parameter_predicates": kept_predicates,
                "verification_checks": kept_checks,
                "branch_groups": kept_groups,
                "resolver_notes": notes,
            }
        )

    @staticmethod
    def _prune_contradicted_branch_group(
        group: SourceMechanismBranchGroup,
        contradicted: set[str],
    ) -> SourceMechanismBranchGroup:
        return group.model_copy(
            update={
                "interpreted_parameter_predicates": [
                    predicate for predicate in group.interpreted_parameter_predicates
                    if predicate.name not in contradicted
                ],
                "verification_checks": [
                    check for check in group.verification_checks
                    if (check.check.parameter or "") not in contradicted
                ],
            }
        )

    def _drop_uncited_agent_facts(
        self,
        draft: SourceDiscoveryCandidateDraft,
    ) -> SourceDiscoveryCandidateDraft:
        return draft.model_copy(
            update={
                "interpreted_parameter_predicates": [
                    predicate for predicate in draft.interpreted_parameter_predicates
                    if predicate.source_file and predicate.source_line is not None
                ],
                "verification_checks": [
                    check for check in draft.verification_checks
                    if check.source_file and check.source_line is not None
                ],
            }
        )

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
        cached_mechanism_seeds: list[dict[str, Any]],
    ) -> SourceDiscoveryIterationPacket:
        source_profile = {
            "stage": "search_hits_only",
            "related_files": compact_source_hits(candidate_hits, limit=12, max_matches_per_file=4),
        }
        if cached_mechanism_seeds:
            source_profile["cached_mechanism_seeds"] = cached_mechanism_seeds
        return SourceDiscoveryIterationPacket(
            user_question=user_question,
            depth=depth,
            active_queries=active_queries,
            visited_files=visited_files,
            new_files=[],
            source_profile=source_profile,
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

    def _consume_iteration_decision(
        self,
        decision: SourceDiscoveryDecision,
        *,
        visited_files: list[str],
        parameter_requirements: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        decision_notes: list[str],
        candidate_drafts: list[SourceDiscoveryCandidateDraft],
        relevant_files: list[str],
        expansions: list[str],
        max_expansion_queries: int,
        all_queries: list[str],
        accounting: Optional[dict[str, Any]] = None,
    ) -> tuple[list[str], bool, list[str]]:
        """Validate and consume one iteration-phase decision (spec §15).

        Semantics identical to the previously inline block: relevant
        files survive only when deterministically eligible (visited),
        drafts pass the parameter gate, notes accumulate,
        deterministic expansions merge with advisory queries under the
        cap, and the stop hint breaks only with remaining-work
        accounting. Returns (expansions, break_loop, all_queries_add).
        """
        kept_files, dropped_files = df_mod.validate_decision_files(
            decision.relevant_files, eligible_files=visited_files,
        )
        if accounting is not None:
            accounting["iteration_dropped_files"] = (
                accounting.get("iteration_dropped_files", 0) + dropped_files
            )
        relevant_files.extend(kept_files)
        candidate_drafts.extend(
            self._filter_candidate_drafts_by_parameter_gate(
                decision.candidate_drafts,
                parameter_requirements,
                log_context,
            )
        )
        decision_notes.extend(decision.notes)
        expansions = dedupe_keep_order([
            *decision.expansion_queries,
            *expansions,
        ])[:max_expansion_queries]
        if decision.stop:
            return expansions, True, [
                query for query in expansions if query not in all_queries
            ]
        return expansions, False, []

    def _apply_round_fallback(
        self,
        full_packet: SourceDiscoveryIterationPacket,
        *,
        state: df_mod.DiscoveryRoundState,
        accounting: dict[str, Any],
        reason: str,
    ) -> SourceDiscoveryIterationPacket:
        """Fail closed to the current full packet for one round (spec §16).

        Marks fallback use with the committed reason, measures mass
        with fallback fields set, counts monotonically, and treats
        the full send as delivering every observed identity. Never
        counts as bounded-frontier success.
        """
        full_bytes = len(df_mod.serialize_packet(
            full_packet.model_dump()).encode("utf-8"))
        state.fallback_count += 1
        state.fallback_bytes_total += full_bytes
        full_packet.fallback = {"used": True, "reason": reason}
        full_packet.packet_mass = df_mod.measure_packet_mass(
            sections={
                "source_profile": df_mod.serialize_packet(full_packet.source_profile),
                "parameter_requirements": df_mod.serialize_packet([
                    _safe_model_dump(requirement)
                    for requirement in full_packet.parameter_requirements
                ]),
                "static_log_context": df_mod.serialize_packet(full_packet.static_log_context),
            },
            section_item_counts={
                "parameter_requirements": len(full_packet.parameter_requirements),
            },
            new_item_count=0,
            summary_bytes=0,
            frontier_bytes=0,
            full_equivalent_bytes=full_bytes,
            fallback_used=True,
            fallback_reason=reason,
            fallback_count=state.fallback_count,
            fallback_bytes=full_bytes,
        )
        state.prior_sent_identities |= set(state.all_observed_identities)
        state.mass_log.append(dict(full_packet.packet_mass))
        accounting["fallback_count"] = state.fallback_count
        accounting["fallback_bytes"] = state.fallback_bytes_total
        return full_packet

    async def _consume_decision_with_lookup(
        self,
        decision: SourceDiscoveryDecision,
        *,
        packet_to_send: SourceDiscoveryIterationPacket,
        state: df_mod.DiscoveryRoundState,
        accounting: dict[str, Any],
        decide: Any,
        visited_files: list[str],
        parameter_requirements: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        decision_notes: list[str],
        candidate_drafts: list[SourceDiscoveryCandidateDraft],
        relevant_files: list[str],
        expansions: list[str],
        max_expansion_queries: int,
        all_queries: list[str],
        depth: int,
    ) -> tuple[list[str], bool]:
        """Consume one decision plus at most one bounded lookup round.

        Processes advisory Mode-B lookup requests deterministically:
        validated identities retrieve canonical content for exactly
        one reconstructed follow-up decision call; unknown identities
        count as unavailable without extra calls, and budget
        exhaustion fails closed. Every consumed decision (first and
        follow-up) passes through the same validated traversal
        consumption, so model output can never directly mutate
        visited/exhaustion state (spec §15, §27).
        """
        current = decision
        current_packet = packet_to_send
        extra_allowed = (
            current_packet.bounded
            and not current_packet.fallback.get("used", False)
        )
        # One shared lookup budget per discovery round (spec §27
        # default: maximum 1 lookup round). The initial and follow-up
        # decisions validate against the SAME usage object, so a
        # second request deterministically exhausts, counts, and
        # retrieves nothing. Never per-decision, never global.
        usage = df_mod.LookupUsage()
        while True:
            for draft in current.candidate_drafts or ():
                title = str(getattr(draft, "title", "") or "")
                if title and all(
                    seen != title for seen, _ in state.prior_draft_titles
                ):
                    state.prior_draft_titles.append((title, depth))
            if not current.lookup_requests:
                break
            result = df_mod.process_lookup_requests(
                current.lookup_requests,
                store=state.store,
                budget=df_mod.LookupBudget(),
                usage=usage,
            )
            if result.status == df_mod.LOOKUP_COMPLETE:
                accounting["lookup_complete"] = accounting.get("lookup_complete", 0) + 1
            elif result.status == df_mod.LOOKUP_UNAVAILABLE:
                accounting["lookup_unavailable"] = accounting.get("lookup_unavailable", 0) + 1
            else:
                accounting["lookup_exhausted"] = accounting.get("lookup_exhausted", 0) + 1
            accounting["lookup_dropped_unknown"] = (
                accounting.get("lookup_dropped_unknown", 0)
                + result.dropped_unknown_count
            )
            if (
                result.status != df_mod.LOOKUP_COMPLETE
                or not result.retrieved
                or not extra_allowed
            ):
                break
            extra_allowed = False
            follow = current_packet.model_copy(deep=True)
            frontier = dict(follow.decision_frontier)
            frontier["lookup_retrieved"] = {
                repr(identity): content
                for identity, content in result.retrieved.items()
            }
            follow.decision_frontier = frontier
            mass = dict(follow.packet_mass)
            mass.update(usage.mass_fragment(
                prefetched_item_count=0, prefetched_bytes=0,
            ))
            follow.packet_mass = mass
            current = await decide(follow)
            current_packet = follow
        expansions, break_loop, all_add = self._consume_iteration_decision(
            current,
            visited_files=visited_files,
            parameter_requirements=parameter_requirements,
            log_context=log_context,
            decision_notes=decision_notes,
            candidate_drafts=candidate_drafts,
            relevant_files=relevant_files,
            expansions=expansions,
            max_expansion_queries=max_expansion_queries,
            all_queries=all_queries,
            accounting=accounting,
        )
        all_queries.extend(all_add)
        return expansions, break_loop

    def _build_bounded_round_packet(
        self,
        full_packet: SourceDiscoveryIterationPacket,
        *,
        state: df_mod.DiscoveryRoundState,
        observed: dict[str, Any],
        depth: int,
        max_depth: int,
        max_total_files: int,
        active_queries: list[str],
        visited_files: list[str],
        new_files: list[str],
        hits: list[SourceFileHit],
        parameter_refs: list[ParameterRef],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        assigned_fields: list[FieldRef],
        read_fields: list[FieldRef],
        source_assignments: list[SourceAssignmentRef],
        function_calls: list[FunctionCallRef],
        helper_expressions: list[HelperExpressionRef],
        branch_conditions: list[BranchConditionRef],
        parameter_predicates: list[ParameterPredicateRef],
        parameter_requirements: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        prior_decision_notes: list[str],
        cached_mechanism_seeds: list[dict[str, Any]],
        max_frontier_bytes: Optional[int] = None,
    ) -> tuple[SourceDiscoveryIterationPacket, dict[str, Any]]:
        """Derive the bounded decision packet for one round (spec §14).

        Narrows the accumulated corpus to new/changed/prefetched
        material plus the carry-forward projection, reusing the
        existing packet builder so round-1 bounded packets carry
        identical content to full packets. Raises
        BoundedPacketError (summary-failure | invariant-violation)
        when narrowing cannot preserve the decision; the caller
        fails closed to the full packet.
        """
        try:
            return self._narrow_to_bounded(
                full_packet,
                state=state,
                observed=observed,
                depth=depth,
                max_depth=max_depth,
                max_total_files=max_total_files,
                active_queries=active_queries,
                visited_files=visited_files,
                new_files=new_files,
                hits=hits,
                parameter_refs=parameter_refs,
                published_topics=published_topics,
                subscribed_topics=subscribed_topics,
                assigned_fields=assigned_fields,
                read_fields=read_fields,
                source_assignments=source_assignments,
                function_calls=function_calls,
                helper_expressions=helper_expressions,
                branch_conditions=branch_conditions,
                parameter_predicates=parameter_predicates,
                parameter_requirements=parameter_requirements,
                log_context=log_context,
                prior_decision_notes=prior_decision_notes,
                cached_mechanism_seeds=cached_mechanism_seeds,
                max_frontier_bytes=max_frontier_bytes,
            )
        except df_mod.BoundedPacketError:
            raise
        except Exception as exc:
            raise df_mod.BoundedPacketError("summary-failure") from exc

    def _narrow_to_bounded(
        self,
        full_packet: SourceDiscoveryIterationPacket,
        *,
        state: df_mod.DiscoveryRoundState,
        observed: dict[str, Any],
        depth: int,
        max_depth: int,
        max_total_files: int,
        active_queries: list[str],
        visited_files: list[str],
        new_files: list[str],
        hits: list[SourceFileHit],
        parameter_refs: list[ParameterRef],
        published_topics: list[TopicRef],
        subscribed_topics: list[TopicRef],
        assigned_fields: list[FieldRef],
        read_fields: list[FieldRef],
        source_assignments: list[SourceAssignmentRef],
        function_calls: list[FunctionCallRef],
        helper_expressions: list[HelperExpressionRef],
        branch_conditions: list[BranchConditionRef],
        parameter_predicates: list[ParameterPredicateRef],
        parameter_requirements: list[ParameterRequirement],
        log_context: SourceDiscoveryLogContext,
        prior_decision_notes: list[str],
        cached_mechanism_seeds: list[dict[str, Any]],
        max_frontier_bytes: Optional[int] = None,
    ) -> tuple[SourceDiscoveryIterationPacket, dict[str, Any]]:
        store = state.store
        by_category = observed["by_category"]
        # Reactivation BEFORE frontier construction (spec §10A, §27).
        # Discovery-round trigger modeling covers changed same-identity
        # requirements and new contradictions touching retained
        # predecessors by variant base key. Hypothesis/candidate/window
        # relation triggers are not modeled at discovery granularity;
        # evaluation runs regardless so future trigger wiring composes.
        contradicted = list(observed["contradicted"])
        base_keys = {
            df_mod.requirement_base_key(identity) for identity in contradicted
        }
        support = set(contradicted)
        for record in store.all_current():
            if (
                record.kind == "requirement"
                and df_mod.requirement_base_key(record.identity) in base_keys
            ):
                support.add(record.identity)
        fired = df_mod.evaluate_reactivation(
            store,
            round_no=depth,
            new_contradictions=[support] if support else [],
            changed_requirements=observed["changed_requirements"],
        )
        reactivated_ids = {record.identity for record in fired}
        send = set(observed["new_all"]) | set(observed["changed_all"]) | reactivated_ids

        def keep(category: str, refs: list, idfn: Any) -> list:
            return [ref for ref in refs if idfn(ref) in send]

        narrowed_assignments = keep("assignment", source_assignments, df_mod.identity_for_assignment)
        narrowed_calls = keep("call", function_calls, df_mod.identity_for_call)
        narrowed_helpers = keep("helper", helper_expressions, df_mod.identity_for_helper)
        narrowed_branches = keep("branch", branch_conditions, df_mod.identity_for_branch)
        narrowed_predicates = keep("predicate", parameter_predicates, df_mod.identity_for_predicate)
        narrowed_published = keep("topic", published_topics, df_mod.identity_for_topic)
        narrowed_subscribed = keep("topic", subscribed_topics, df_mod.identity_for_topic)
        narrowed_assigned = keep("field", assigned_fields, df_mod.identity_for_field)
        narrowed_read = keep("field", read_fields, df_mod.identity_for_field)
        narrowed_param_refs = keep("paramref", parameter_refs, df_mod.identity_for_parameter_ref)
        narrowed_requirements = keep("requirement", parameter_requirements, df_mod.identity_for_requirement)
        narrowed_hits = [hit for hit in hits if hit.file in new_files]
        narrowed_eliminated = eliminated_branch_selector_requirements(narrowed_requirements)

        new_branch_ids = [identity for identity in by_category["branch"] if identity in send]
        open_reqs = list(observed["open_requirements"])
        # Fail closed on unexpected prefetch gaps: every closure
        # member is store-observed by construction, so a missing
        # record means the bounded representation cannot preserve
        # the decision (spec §13).
        found = df_mod.closed_prefetch(
            store,
            active_candidates=sorted(observed["new_all"], key=repr),
            open_claims=open_reqs,
            new_contradictions=contradicted,
            new_branches=new_branch_ids,
            reactivated=sorted(reactivated_ids, key=repr),
            unresolved_requirements=open_reqs,
        )

        file_counts: dict[str, dict[str, int]] = {}
        for path in visited_files:
            file_counts[path] = {
                "assignments": 0, "calls": 0, "helpers": 0, "branches": 0,
            }
        for ref in source_assignments:
            if ref.file in file_counts:
                file_counts[ref.file]["assignments"] += 1
        for ref in function_calls:
            if ref.file in file_counts:
                file_counts[ref.file]["calls"] += 1
        for ref in helper_expressions:
            if ref.file in file_counts:
                file_counts[ref.file]["helpers"] += 1
        for ref in branch_conditions:
            if ref.file in file_counts:
                file_counts[ref.file]["branches"] += 1
        coverage_map = {
            "files_seen": len(visited_files),
            "files": file_counts,
            "rounds": depth,
        }
        standings = [
            {
                "candidate": title,
                "rounds_in_contention": max(0, depth - first_round),
            }
            for title, first_round in state.prior_draft_titles
        ]
        summary = df_mod.build_summary(
            store,
            coverage_map=coverage_map,
            gate_tally=observed["gate_tally"],
            candidate_standings=standings,
            round_no=depth,
        )

        def compact_map(category: str) -> dict[str, Any]:
            # New/changed items render from current refs; reactivated
            # items that were not re-extracted render from the
            # canonical record with full prior verdict history (§10A).
            # Category gating via the record kind keeps reactivated
            # identities in their own section only.
            wanted = {
                identity for identity in send
                if identity in by_category.get(category, {})
            }
            for identity in reactivated_ids:
                record = store.get(identity)
                if record is not None and record.kind == category:
                    wanted.add(identity)
            return {
                identity: _frontier_item_payload(
                    store, by_category[category].get(identity), identity,
                )
                for identity in sorted(wanted, key=repr)
            }

        unresolved_entries: dict[Any, Any] = {}
        for identity in sorted(open_reqs, key=repr):
            record = found.get(identity)
            if record is None:
                continue
            unresolved_entries[identity] = {
                "lifecycle": record.lifecycle,
                "rounds_waiting": max(0, depth - record.first_seen_round),
                "value_revision": record.value_revision,
                "changed_this_round": record.last_changed_round == depth,
                "full_content": dict(record.content),
            }
        contradiction_entries: dict[Any, Any] = {}
        for identity in sorted(set(contradicted), key=repr):
            record = store.get(identity)
            if record is None:
                continue
            if identity in state.prior_sent_identities and record.last_changed_round != depth:
                continue
            contradiction_entries[identity] = {
                "refutations": [list(verdict) for verdict in record.prior_verdicts],
                "window_identity": record.window_identity,
            }
        grouped = df_mod.group_requirements(parameter_requirements)
        frontier = df_mod.DiscoveryFrontier(
            new_candidates={
                df_mod.identity_for_file(path): {
                    "hits": sum(1 for hit in narrowed_hits if hit.file == path),
                }
                for path in new_files
            },
            changed_assignments=compact_map("assignment"),
            changed_calls=compact_map("call"),
            changed_branches=compact_map("branch"),
            new_helper_relationships=compact_map("helper"),
            new_requirements={
                identity: _requirement_variant_descriptor(
                    store, by_category["requirement"].get(identity), identity,
                )
                for identity in sorted(by_category["requirement"].keys(), key=repr)
                if identity in send
            } | {
                identity: _frontier_item_payload(store, None, identity)
                for identity in sorted(reactivated_ids, key=repr)
                if (store.get(identity) is not None
                    and store.get(identity).kind == "requirement"
                    and identity not in by_category["requirement"])
            },
            new_verification_candidates={},
            unresolved_questions=unresolved_entries,
            contradictions=contradiction_entries,
            work_state={
                "queries_remaining": len(active_queries),
                "files_remaining_under_cap": max(0, max_total_files - len(visited_files)),
                "depth_remaining": max(0, max_depth - depth),
            },
        )
        frontier_dict = frontier.to_packet_dict()
        frontier_dict["requirement_groups"] = {
            name: [repr(df_mod.identity_for_requirement(variant)) for variant in variants]
            for name, variants in grouped["groups"].items()
        }
        summary_dict = summary.to_packet_dict()

        narrowed_packet = self._build_iteration_packet(
            user_question=full_packet.user_question,
            depth=depth,
            active_queries=active_queries,
            visited_files=visited_files,
            new_files=new_files,
            hits=narrowed_hits,
            parameter_refs=narrowed_param_refs,
            published_topics=narrowed_published,
            subscribed_topics=narrowed_subscribed,
            assigned_fields=narrowed_assigned,
            read_fields=narrowed_read,
            source_assignments=narrowed_assignments,
            function_calls=narrowed_calls,
            helper_expressions=narrowed_helpers,
            branch_conditions=narrowed_branches,
            parameter_predicates=narrowed_predicates,
            parameter_requirements=narrowed_requirements,
            eliminated_parameter_paths=narrowed_eliminated,
            log_context=log_context,
            prior_decision_notes=prior_decision_notes,
            cached_mechanism_seeds=cached_mechanism_seeds,
        )
        verification = narrowed_packet.source_profile.get(
            "expression_verification_candidates", [],
        ) or []
        verification_ids: dict[Any, Any] = {}
        for candidate in verification:
            identity = df_mod.identity_for_verification(
                candidate.get("name"), candidate.get("source_file"),
                candidate.get("source_line"),
            )
            verification_ids[identity] = candidate
            if identity not in state.prior_sent_identities:
                state.store.observe(
                    identity, kind="verification",
                    content={"semantic_effect": candidate.get("name")},
                    round_no=depth,
                )
                state.all_observed_identities.add(identity)
        prior_sent_reprs = {
            repr(identity) for identity in state.prior_sent_identities
        }
        frontier_dict["new_verification_candidates"] = {
            identity: candidate
            for identity, candidate in verification_ids.items()
            if repr(identity) not in prior_sent_reprs
        }

        frontier_ids: set = set()
        for section in (
            "new_candidates", "changed_assignments", "changed_calls",
            "changed_branches", "new_helper_relationships",
            "new_requirements", "unresolved_questions", "contradictions",
        ):
            frontier_ids.update(frontier_dict.get(section, {}).keys())
        frontier_ids.update(repr(identity) for identity in verification_ids.keys())
        summary_ids: set = set()
        for entry in summary.resolved_claims + summary.open_claims + summary.contradiction_ledger:
            identity = entry.get("identity")
            # Summary entries serialize identities as lists; normalize
            # back to tuples so reprs match the frontier side exactly.
            if isinstance(identity, list):
                identity = tuple(identity)
            summary_ids.add(identity if isinstance(identity, str) else repr(identity))
        summary_ids.update(repr(df_mod.identity_for_file(path)) for path in coverage_map["files"])
        verdict_ids = {
            repr(record.identity)
            for record in store.all_current()
            if record.lifecycle in {df_mod.RESOLVED, df_mod.CONTRADICTED}
        }
        full_check = {
            identity
            for identity in state.all_observed_identities
            if (store.get(identity) is None
                or store.get(identity).lifecycle != df_mod.IRRELEVANT)
        }
        ok, report = df_mod.check_round_trip(
            full_identities={repr(identity) for identity in full_check},
            frontier_identities=set(frontier_ids),
            summary_identities=set(summary_ids),
            verdict_identities=set(verdict_ids),
        )
        if not ok:
            raise df_mod.BoundedPacketError("invariant-violation")

        narrowed_packet.bounded = True
        narrowed_packet.decision_frontier = frontier_dict
        narrowed_packet.carry_forward_summary = summary_dict
        narrowed_packet.fallback = {"used": False, "reason": ""}
        sections = {
            "decision_frontier": df_mod.serialize_packet(frontier_dict),
            "carry_forward_summary": df_mod.serialize_packet(summary_dict),
            "source_profile": df_mod.serialize_packet(narrowed_packet.source_profile),
            "parameter_requirements": df_mod.serialize_packet([
                _safe_model_dump(requirement) for requirement in narrowed_requirements
            ]),
            "static_log_context": df_mod.serialize_packet(narrowed_packet.static_log_context),
        }
        full_sections = {
            "source_profile": df_mod.serialize_packet(full_packet.source_profile),
            "parameter_requirements": df_mod.serialize_packet([
                _safe_model_dump(requirement) for requirement in full_packet.parameter_requirements
            ]),
            "static_log_context": df_mod.serialize_packet(full_packet.static_log_context),
        }
        full_bytes = sum(len(text.encode("utf-8")) for text in full_sections.values())
        frontier_bytes = len(sections["decision_frontier"].encode("utf-8"))
        summary_bytes = len(sections["carry_forward_summary"].encode("utf-8"))
        mass = df_mod.measure_packet_mass(
            sections=sections,
            section_item_counts={
                "decision_frontier": len(frontier_ids),
                "carry_forward_summary": len(summary_ids),
                "parameter_requirements": len(narrowed_requirements),
            },
            new_item_count=len(observed["new_all"]),
            summary_bytes=summary_bytes,
            frontier_bytes=frontier_bytes,
            full_equivalent_bytes=full_bytes,
        )
        narrowed_packet.packet_mass = mass
        if max_frontier_bytes is not None:
            over = df_mod.check_frontier_budget(
                sections["decision_frontier"], max_bytes=max_frontier_bytes,
            )
            if over is not None:
                unavailable = full_packet.model_copy(deep=True)
                unavailable.bounded = True
                unavailable.source_profile = {
                    "stage": "bounded-unavailable",
                    "unavailable": df_mod.FRONTIER_TOO_LARGE,
                }
                unavailable.decision_frontier = {
                    "unavailable": df_mod.FRONTIER_TOO_LARGE,
                    "work_state": frontier_dict["work_state"],
                }
                unavailable.carry_forward_summary = summary_dict
                unavailable.packet_mass = mass
                unavailable.fallback = {"used": False, "reason": ""}
                return unavailable, {
                    "bounded_ok": False, "unavailable": True,
                    "frontier_ids": set(), "summary_ids": set(summary_ids),
                    "mass": mass, "new_count": 0,
                }
        sent_ids = {
            identity for identity in state.all_observed_identities
            if repr(identity) in frontier_ids or repr(identity) in summary_ids
        }
        state.prior_sent_identities |= sent_ids
        state.mass_log.append(mass)
        return narrowed_packet, {
            "bounded_ok": True, "unavailable": False,
            "frontier_ids": set(frontier_ids), "summary_ids": set(summary_ids),
            "mass": mass, "new_count": len(observed["new_all"]),
        }

    def _source_snippets_for_files(
        self,
        files: list[str],
        *,
        active_queries: Optional[list[str]] = None,
        hits: Optional[list[SourceFileHit]] = None,
        default_lines_per_file: int = 220,
        context_lines: int = 12,
        max_lines_per_snippet: int = 180,
        max_chars_per_snippet: int = 12_000,
        max_snippets_per_file: int = 4,
    ) -> list[SourceSnippet]:
        snippets: list[SourceSnippet] = []
        query_targets = self._snippet_targets_from_queries(active_queries or [])
        hit_targets = self._snippet_targets_from_hits(hits or [])

        for file in files:
            try:
                text = self.source.read_text(file)
            except Exception:
                continue

            lines = text.splitlines()
            targets = [
                *query_targets.get(file, []),
                *self._function_targets_for_file(file, lines, active_queries or []),
                *hit_targets.get(file, []),
            ]
            ranges = self._snippet_ranges(
                len(lines),
                targets,
                context_lines=context_lines,
                max_lines_per_snippet=max_lines_per_snippet,
                max_snippets=max_snippets_per_file,
            )
            if not ranges:
                ranges = [(1, min(len(lines), default_lines_per_file))]

            for start_line, end_line in ranges:
                selected = lines[start_line - 1:end_line]
                snippet_text = "\n".join(selected)
                if len(snippet_text) > max_chars_per_snippet:
                    snippet_text = snippet_text[: max_chars_per_snippet - 3] + "..."
                snippets.append(
                    SourceSnippet(
                        file=file,
                        start_line=start_line,
                        end_line=end_line,
                        text=snippet_text,
                    )
                )
        return snippets

    def _requested_source_files(self, queries: list[str]) -> list[str]:
        files: list[str] = []
        for query in queries:
            for file in extract_source_file_paths(query):
                if self.source.file_exists(file):
                    files.append(file)
        return dedupe_keep_order(files)

    def _snippet_targets_from_queries(self, queries: list[str]) -> dict[str, list[tuple[int, int]]]:
        targets: dict[str, list[tuple[int, int]]] = {}
        for query in queries:
            files = extract_source_file_paths(query)
            if not files:
                continue
            ranges = extract_line_ranges(query)
            if not ranges:
                continue
            for file in files:
                targets.setdefault(file, []).extend(ranges)
        return targets

    def _snippet_targets_from_hits(self, hits: list[SourceFileHit]) -> dict[str, list[tuple[int, int]]]:
        targets: dict[str, list[tuple[int, int]]] = {}
        for hit in hits:
            for match in hit.matches[:2]:
                targets.setdefault(hit.file, []).append((match.line, match.line))
        return targets

    def _function_targets_for_file(
        self,
        file: str,
        lines: list[str],
        queries: list[str],
    ) -> list[tuple[int, int]]:
        function_names: list[str] = []
        for query in queries:
            files = extract_source_file_paths(query)
            if files and file not in files:
                continue
            function_names.extend(extract_requested_function_names(query))

        targets: list[tuple[int, int]] = []
        for function_name in dedupe_keep_order(function_names):
            short_name = function_name.split("::")[-1]
            pattern = re.compile(
                rf"(?:\b{re.escape(function_name)}\s*\(|\b{re.escape(short_name)}\s*\()"
            )
            for index, line in enumerate(lines, start=1):
                if pattern.search(line):
                    targets.append((index, min(index + 160, len(lines))))
                    break
        return targets

    @staticmethod
    def _snippet_ranges(
        line_count: int,
        targets: list[tuple[int, int]],
        *,
        context_lines: int,
        max_lines_per_snippet: int,
        max_snippets: int,
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        for start, end in targets:
            if line_count <= 0:
                continue
            start = max(1, min(int(start), line_count))
            end = max(start, min(int(end), line_count))
            start = max(1, start - context_lines)
            end = min(line_count, end + context_lines)
            if end - start + 1 > max_lines_per_snippet:
                end = start + max_lines_per_snippet - 1
            ranges.append((start, end))

        merged: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if not merged or start > merged[-1][1] + 1:
                merged.append((start, end))
            else:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
        return merged[:max_snippets]

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
        branch_conditions: list[BranchConditionRef],
        *,
        max_queries: int,
    ) -> list[str]:
        queries: list[str] = []
        queries.extend(ref.name for ref in parameters if ref.name)
        queries.extend(self._helper_call_queries(function_calls, branch_conditions))
        queries.extend(ref.topic for ref in topics if ref.topic)
        queries.extend(ref.field for ref in fields if ref.field)
        return dedupe_keep_order([query for query in queries if query])[:max_queries]

    def _helper_call_queries(
        self,
        function_calls: list[FunctionCallRef],
        branch_conditions: list[BranchConditionRef],
    ) -> list[str]:
        queries: list[str] = []
        for ref in function_calls:
            if self._is_relevant_function_name(ref.name):
                queries.append(ref.name)
        for ref in branch_conditions:
            queries.extend(
                name
                for name in extract_call_names(ref.condition)
                if self._is_relevant_function_name(name)
            )
        return dedupe_keep_order(queries)

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
        helper_expressions: list[HelperExpressionRef],
        source_assignments: list[SourceAssignmentRef],
        function_calls: list[FunctionCallRef],
        output_bindings: list[SourceOutputBindingRecord],
    ) -> list[SourceMechanismCandidate]:
        deterministic_checks = source_backed_derived_expression_checks(
            helper_expressions,
            source_assignments=source_assignments,
            function_calls=function_calls,
            source_path=self.source,
            limit=40,
        )
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
                    deterministic_checks=deterministic_checks,
                    output_bindings=output_bindings,
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
                deterministic_checks=deterministic_checks,
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
        deterministic_checks: list[SourceBackedVerificationCheck],
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
        scoped_deterministic_checks = scope_deterministic_checks_to_candidate(
            deterministic_checks,
            source_files=source_files,
            relevant_signals=[
                f"{field.topic}.{field.field}"
                for field in relevant_fields
                if field.topic and field.field
            ],
        )
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
                for ref in dedupe_topic_refs_by_topic(published_topics)
            ],
            subscribed_topics=[
                TopicFieldRef(topic=ref.topic, source_file=ref.file, source_line=ref.line)
                for ref in dedupe_topic_refs_by_topic(subscribed_topics)
            ],
            relevant_fields=relevant_fields,
            branch_conditions=[ref.condition for ref in branch_conditions],
            expected_log_signature=expected_log_signature,
            required_log_evidence=required_log_evidence,
            verification_checks=scoped_deterministic_checks,
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
        deterministic_checks: list[SourceBackedVerificationCheck],
        output_bindings: Optional[list[SourceOutputBindingRecord]] = None,
    ) -> SourceMechanismCandidate:
        source_files = draft.source_files or fallback_source_files
        required_parameters = set(draft.controlling_parameter_names)
        controlling_parameters = [
            requirement for requirement in parameter_requirements
            if (not required_parameters or requirement.name in required_parameters)
            and not is_contradicted_branch_selector(requirement)
        ]
        controlling_parameters = dedupe_parameter_requirements([
            *controlling_parameters,
            *self.parameter_gate.evaluate_source_predicates(
                draft.interpreted_parameter_predicates,
                log_context,
            ),
        ])
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
            deterministic_checks=deterministic_checks,
        )
        scoped_deterministic_checks = scope_deterministic_checks_to_candidate(
            deterministic_checks if draft.relevant_signals and not draft.branch_groups else [],
            source_files=source_files,
            relevant_signals=draft.relevant_signals,
        )
        verification_checks = dedupe_source_backed_verification_checks([
            *draft.verification_checks,
            *scoped_deterministic_checks,
        ])
        branch_groups = [
            self._build_source_candidate_branch_group(
                group,
                deterministic_checks=deterministic_checks,
            )
            for group in draft.branch_groups
        ]
        source_chain = draft.source_chain or fallback.source_chain
        return SourceMechanismCandidate(
            title=draft.title or fallback.title,
            source_mechanism=draft.source_mechanism or fallback.source_mechanism,
            source_chain=source_chain,
            source_files=source_files,
            controlling_parameters=controlling_parameters,
            published_topics=fallback.published_topics,
            subscribed_topics=fallback.subscribed_topics,
            relevant_fields=relevant_fields or fallback.relevant_fields,
            primary_output_signals=source_backed_primary_output_signals(
                draft.primary_output_signals,
                output_bindings or [],
                source_chain,
            ),
            branch_conditions=draft.branch_conditions or fallback.branch_conditions,
            expected_log_signature=draft.expected_log_signature or fallback.expected_log_signature,
            required_log_evidence=draft.required_log_evidence or fallback.required_log_evidence,
            interpreted_parameter_predicates=draft.interpreted_parameter_predicates,
            verification_checks=verification_checks,
            contradiction_checks=draft.contradiction_checks or fallback.contradiction_checks,
            branch_groups=branch_groups,
            source_confidence=draft.source_confidence,
            resolver_notes=[
                "Source candidate was drafted by the source-discovery decision agent.",
                "No time-series signal comparison, event timestamp proof, plots, or final root-cause confidence were computed.",
                *draft.resolver_notes,
                *decision_notes,
            ],
        )

    def _build_source_candidate_branch_group(
        self,
        group: SourceMechanismBranchGroup,
        *,
        deterministic_checks: list[SourceBackedVerificationCheck],
    ) -> SourceMechanismBranchGroup:
        scoped_deterministic_checks = scope_deterministic_checks_to_candidate(
            deterministic_checks if group.relevant_signals else [],
            source_files=group.source_files,
            relevant_signals=group.relevant_signals,
        )
        return group.model_copy(
            update={
                "verification_checks": dedupe_source_backed_verification_checks([
                    *group.verification_checks,
                    *scoped_deterministic_checks,
                ])
            }
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
            or lowered in {"get", "set", "update", "publish", "copy", "min", "max", "abs"}
            or lowered in {"if", "for", "while", "switch", "return", "sizeof"}
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

    def evaluate_source_predicates(
        self,
        predicates: list[SourceBackedParameterPredicate],
        log_context: SourceDiscoveryLogContext,
    ) -> list[ParameterRequirement]:
        requirements: list[ParameterRequirement] = []
        for predicate in predicates:
            actual_value = log_context.parameters.get(predicate.name)
            gate_result = self._evaluate_source_predicate(predicate, actual_value)
            requirements.append(
                ParameterRequirement(
                    name=predicate.name,
                    role=predicate.role,
                    source_predicate=predicate.predicate,
                    actual_value=actual_value,
                    gate_result=gate_result,
                    effect=predicate.effect or self._effect_text(predicate.name, predicate.role, gate_result),
                    source_file=predicate.source_file or "unknown",
                    source_line=predicate.source_line,
                )
            )
        return requirements

    def _evaluate_source_predicate(
        self,
        predicate: SourceBackedParameterPredicate,
        actual_value: Any,
    ) -> str:
        if actual_value is None:
            return "unknown"
        if predicate.role != "branch_selector":
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
    try:
        return compare(actual, operator, expected)
    except ValueError as exc:
        raise TypeError(str(exc)) from exc


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_call_names(text: str) -> list[str]:
    names = re.findall(
        r"(?<![#A-Za-z0-9_])((?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)\s*\(",
        text or "",
    )
    ignored = {"if", "for", "while", "switch", "return", "sizeof"}
    return [name for name in names if name not in ignored]


def extract_source_file_paths(text: str) -> list[str]:
    pattern = re.compile(
        r"\b(?P<file>(?:src|platforms|boards|ROMFS|msg|test)/"
        r"[A-Za-z0-9_./+\-]+"
        r"\.(?:c|cc|cpp|cxx|h|hpp|hh|hxx|cuh|cu))\b"
    )
    return dedupe_keep_order([
        match.group("file").rstrip(".,;:")
        for match in pattern.finditer(text or "")
    ])


def extract_line_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(
        r"\b(?:lines?|around(?:\s+lines?)?)\s+"
        r"(?P<start>\d{1,6})\s*(?:-|to|:)\s*(?P<end>\d{1,6})\b",
        text or "",
        flags=re.IGNORECASE,
    ):
        start = int(match.group("start"))
        end = int(match.group("end"))
        ranges.append((min(start, end), max(start, end)))
    for match in re.finditer(
        r"\bline\s+(?P<line>\d{1,6})\b",
        text or "",
        flags=re.IGNORECASE,
    ):
        line = int(match.group("line"))
        ranges.append((line, line))
    return dedupe_line_ranges(ranges)


def extract_requested_function_names(text: str) -> list[str]:
    names = re.findall(
        r"\b([A-Za-z_][A-Za-z0-9_~]*(?:::[A-Za-z_][A-Za-z0-9_~]*)?)\s*\(\)",
        text or "",
    )
    ignored = {
        "around",
        "profile",
        "including",
        "function",
        "functions",
        "helper",
        "helpers",
    }
    return dedupe_keep_order([
        name for name in names
        if name.lower() not in ignored
    ])


def dedupe_line_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for start, end in ranges:
        key = (start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
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


def dedupe_topic_refs_by_topic(refs: list[TopicRef]) -> list[TopicRef]:
    seen = set()
    out: list[TopicRef] = []
    for ref in refs:
        if ref.topic in seen:
            continue
        seen.add(ref.topic)
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


def dedupe_source_assignment_refs(refs: list[SourceAssignmentRef]) -> list[SourceAssignmentRef]:
    seen = set()
    out: list[SourceAssignmentRef] = []
    for ref in refs:
        key = (ref.target, ref.expression, ref.function, ref.file, ref.line)
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


def dedupe_helper_expression_refs(refs: list[HelperExpressionRef]) -> list[HelperExpressionRef]:
    seen = set()
    out: list[HelperExpressionRef] = []
    for ref in refs:
        key = (ref.name, ref.file, ref.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(ref)
    return out


def helper_expression_verification_candidates(
    refs: list[HelperExpressionRef],
    *,
    source_assignments: list[SourceAssignmentRef] | None = None,
    function_calls: list[FunctionCallRef] | None = None,
    source_path: str | Path | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    source_assignments = source_assignments or []
    function_calls = function_calls or []
    output_bindings = source_output_binding_candidates(
        source_assignments,
        function_calls,
    )
    output_checks = source_output_derived_expression_candidates(
        refs,
        source_assignments,
        function_calls,
        source_path=source_path,
        limit=limit,
    )
    helper_symbol_bindings = combined_helper_symbol_bindings(refs)
    candidates: list[dict[str, Any]] = []
    for ref in refs:
        if not ref.lowered_return_expression:
            continue
        unresolved_calls = [
            call
            for call in ref.call_resolutions
            if call.get("kind") in {"source_helper_candidate", "unresolved_runtime_call"}
        ]
        lowered = lower_source_expression_for_evaluator(
            ref.lowered_return_expression,
            {**helper_symbol_bindings, **ref.symbol_bindings},
            source_path=source_path,
            extra_signal_bindings={
                **source_assignment_signal_bindings(source_assignments, source_path=source_path),
                **output_signal_bindings(output_bindings),
            },
        )
        candidates.append(
            {
                "name": ref.name,
                "source_file": ref.file,
                "source_line": ref.line,
                "lowered_return_expression": ref.lowered_return_expression,
                "evaluator_expression": lowered["expression"],
                "evaluator_variables": lowered["variables"],
                "unresolved_symbols": lowered["unresolved_symbols"],
                "symbol_bindings": dict(ref.symbol_bindings),
                "unresolved_calls": unresolved_calls,
                "output_binding_candidates": [
                    binding for binding in output_bindings
                    if expression_mentions_symbol(ref.lowered_return_expression, binding.get("source_symbol", ""))
                ],
                "derived_expression_checks": [
                    check for check in output_checks
                    if check.get("source_helper") == ref.name
                ],
                "output_binding_required": True,
            }
        )
        if len(candidates) >= limit:
            break
    return candidates


def source_discovery_expression_verification_candidates(
    refs: list[HelperExpressionRef],
    *,
    source_assignments: list[SourceAssignmentRef] | None = None,
    function_calls: list[FunctionCallRef] | None = None,
    source_path: str | Path | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    return [
        source_discovery_expression_candidate(candidate)
        for candidate in helper_expression_verification_candidates(
            refs,
            source_assignments=source_assignments,
            function_calls=function_calls,
            source_path=source_path,
            limit=limit,
        )
    ]


def source_discovery_expression_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(candidate)
    sanitized.pop("output_binding_candidates", None)
    sanitized.pop("derived_expression_checks", None)
    sanitized.pop("output_binding_required", None)
    return sanitized


def source_backed_derived_expression_checks(
    refs: list[HelperExpressionRef],
    *,
    source_assignments: list[SourceAssignmentRef] | None = None,
    function_calls: list[FunctionCallRef] | None = None,
    source_path: str | Path | None = None,
    limit: int,
) -> list[SourceBackedVerificationCheck]:
    source_assignments = source_assignments or []
    function_calls = function_calls or []
    helper_refs = {
        ref.name: ref
        for ref in refs
        if ref.name
    }
    checks: list[SourceBackedVerificationCheck] = []
    for raw_check in source_output_derived_expression_candidates(
        refs,
        source_assignments,
        function_calls,
        source_path=source_path,
        limit=limit,
    ):
        source_check = source_backed_derived_expression_check(raw_check, helper_refs)
        if source_check is not None:
            checks.append(source_check)
    return dedupe_source_backed_verification_checks(checks)


def source_backed_derived_expression_check(
    raw_check: dict[str, Any],
    helper_refs: dict[str, HelperExpressionRef],
) -> SourceBackedVerificationCheck | None:
    source_file = None
    source_line = None
    helper_name = str(raw_check.get("source_helper") or "")
    helper_ref = helper_refs.get(helper_name)
    if helper_ref is not None:
        source_file = helper_ref.file
        source_line = helper_ref.line
    if source_file is None:
        for step in raw_check.get("assignment_path") or []:
            if step.get("file"):
                source_file = step.get("file")
                source_line = step.get("line")
                break
    if source_file is None or source_line is None:
        return None

    variables = [
        {"name": str(name), "source": str(source)}
        for name, source in (raw_check.get("variables") or {}).items()
        if name and source is not None
    ]
    check = RelationshipCheckSpec(
        type="derived_expression",
        expression=raw_check.get("expression"),
        expected_expression=raw_check.get("expected_expression"),
        variables=variables,
        op=raw_check.get("op"),
        max_error=raw_check.get("max_error"),
        mode=raw_check.get("mode"),
        supports=(
            f"Source-derived expression for {raw_check.get('source_output')} "
            f"matches {raw_check.get('expected_expression')}."
        ),
        contradicts=(
            f"Source-derived expression for {raw_check.get('source_output')} "
            "does not match the logged output."
        ),
        description=f"Check source-derived expression for {raw_check.get('source_output')}.",
    )
    return SourceBackedVerificationCheck(
        check=check,
        source_file=str(source_file),
        source_line=int(source_line),
        rationale="Deterministically derived from source helper expression and output binding.",
    )


def dedupe_source_backed_verification_checks(
    checks: list[SourceBackedVerificationCheck],
) -> list[SourceBackedVerificationCheck]:
    seen = set()
    out: list[SourceBackedVerificationCheck] = []
    for source_check in checks:
        dumped = source_check.check.model_dump(mode="json")
        key = (
            source_check.source_file,
            source_check.source_line,
            repr(sorted(dumped.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(source_check)
    return out


def compact_cached_mechanism_seeds(seeds: list[dict[str, Any]], *, limit: int = 5) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for seed in seeds[:limit]:
        compact.append({
            "name": seed.get("name"),
            "summary": seed.get("summary"),
            "source_refs": [
                {
                    "file": ref.get("file"),
                    "function": ref.get("function"),
                    "start_line": ref.get("start_line"),
                    "explanation": ref.get("explanation"),
                }
                for ref in seed.get("source_refs", [])[:8]
                if isinstance(ref, dict)
            ],
            "required_parameters": list(seed.get("required_parameters") or [])[:12],
            "required_signals": list(seed.get("required_signals") or [])[:12],
            "source_relevant_fields": list(seed.get("source_relevant_fields") or [])[:12],
            "numeric_check_count": len(seed.get("numeric_checks") or []),
            "exclusion_check_count": len(seed.get("exclusion_checks") or []),
        })
    return [
        {
            key: value
            for key, value in item.items()
            if value not in (None, "", [], {})
        }
        for item in compact
    ]


def cached_mechanism_seed_queries(seeds: list[dict[str, Any]]) -> list[str]:
    queries: list[str] = []
    for seed in seeds:
        if seed.get("name"):
            queries.append(str(seed["name"]))
        for ref in seed.get("source_refs", [])[:4]:
            file = ref.get("file")
            function = ref.get("function")
            if file:
                queries.append(str(file))
            if function:
                queries.append(str(function))
        queries.extend(str(param) for param in seed.get("required_parameters", [])[:6])
    return dedupe_keep_order(queries)


def scope_deterministic_checks_to_candidate(
    checks: list[SourceBackedVerificationCheck],
    *,
    source_files: list[str],
    relevant_signals: list[str],
) -> list[SourceBackedVerificationCheck]:
    normalized_files = {str(file) for file in source_files if file}
    normalized_signals = {str(signal) for signal in relevant_signals if signal}
    scoped: list[SourceBackedVerificationCheck] = []
    for source_check in checks:
        check_signals = set(relationship_check_signal_references(source_check.check))
        if normalized_signals:
            if check_signals & normalized_signals:
                scoped.append(source_check)
            continue
        if source_check.source_file and str(source_check.source_file) in normalized_files:
            scoped.append(source_check)
    return dedupe_source_backed_verification_checks(scoped)


def relationship_check_signal_references(check: RelationshipCheckSpec) -> list[str]:
    signals: list[str] = []
    for value in (check.signal, check.actual, check.setpoint, check.first, check.second):
        if value and looks_like_logged_signal(str(value)):
            signals.append(str(value))
    for variable in check.variables:
        source = getattr(variable, "source", None)
        if source and looks_like_logged_signal(str(source)):
            signals.append(str(source))
    return dedupe_keep_order(signals)


def looks_like_logged_signal(value: str) -> bool:
    return bool(re.fullmatch(r"[a-z][a-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*(?:[.\[][^ ]*)*", value))


def source_output_derived_expression_candidates(
    refs: list[HelperExpressionRef],
    source_assignments: list[SourceAssignmentRef],
    function_calls: list[FunctionCallRef],
    *,
    source_path: str | Path | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    output_bindings = [
        binding for binding in source_output_binding_candidates(source_assignments, function_calls)
        if binding.get("logged_signal")
    ]
    target_to_sources = assignment_sources_by_target(source_assignments)
    for edge in call_bound_assignment_edges(source_assignments, function_calls):
        source = str(edge.get("source_symbol") or "")
        target = str(edge.get("target_symbol") or "")
        if source and target:
            target_to_sources.setdefault(target, [])
            if source not in target_to_sources[target]:
                target_to_sources[target].append(source)
    helper_index = composable_helper_expression_index(refs)
    helper_symbol_bindings = combined_helper_symbol_bindings(refs)
    extra_signal_bindings = {
        **source_assignment_signal_bindings(source_assignments, source_path=source_path),
        **output_signal_bindings(output_bindings),
    }
    output_bindings = sorted(
        output_bindings,
        key=lambda binding: output_binding_sort_key(binding, helper_index),
    )

    checks: list[dict[str, Any]] = []
    seen = set()
    for binding in output_bindings:
        logged_signal = str(binding.get("logged_signal") or "")
        source_expression = str(binding.get("source_symbol") or "")
        if not logged_signal or not source_expression:
            continue
        for expression in expand_source_expression_variants(
            source_expression,
            target_to_sources=target_to_sources,
            helper_index=helper_index,
            max_depth=5,
            max_variants=80,
        ):
            lowered = lower_source_expression_for_evaluator(
                expression,
                helper_symbol_bindings,
                source_path=source_path,
                extra_signal_bindings=extra_signal_bindings,
            )
            if lowered["unresolved_symbols"] and " if " not in lowered["expression"]:
                continue
            if not evaluator_expression_syntax_valid(lowered["expression"]):
                continue
            helper_name = source_helper_name_for_expression(source_expression, helper_index)
            key = (logged_signal, lowered["expression"])
            if key in seen:
                continue
            seen.add(key)
            checks.append(
                {
                    "type": "derived_expression",
                    "source_helper": helper_name,
                    "source_expression": expression,
                    "expression": "actual",
                    "expected_expression": lowered["expression"],
                    "variables": {"actual": logged_signal, **lowered["variables"]},
                    "op": "==",
                    "max_error": 1.0e-3,
                    "mode": "all",
                    "source_output": logged_signal,
                    "assignment_path": binding.get("assignment_path") or [],
                }
            )
            if len(checks) >= limit:
                return checks
    return checks


def assignment_sources_by_target(source_assignments: list[SourceAssignmentRef]) -> dict[str, list[str]]:
    by_target: dict[str, list[str]] = {}
    for ref in source_assignments:
        if ref.target and ref.expression and ref.expression != ref.target:
            by_target.setdefault(ref.target, []).append(ref.expression)
    return {target: dedupe_keep_order(sources) for target, sources in by_target.items()}


def composable_helper_expression_index(refs: list[HelperExpressionRef]) -> dict[str, HelperExpressionRef]:
    full_names: dict[str, HelperExpressionRef] = {}
    short_names: dict[str, list[HelperExpressionRef]] = {}
    for ref in refs:
        if ref.unresolved_reason or not (ref.lowered_return_expression or ref.return_expression):
            continue
        full_names[ref.name] = ref
        short_names.setdefault(ref.name.split("::")[-1], []).append(ref)
    out = dict(full_names)
    for short_name, matches in short_names.items():
        if len(matches) == 1:
            out[short_name] = matches[0]
    return out


def combined_helper_symbol_bindings(refs: list[HelperExpressionRef]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for ref in refs:
        bindings.update(ref.symbol_bindings)
    return bindings


def expand_source_expression_variants(
    expression: str,
    *,
    target_to_sources: dict[str, list[str]],
    helper_index: dict[str, HelperExpressionRef],
    max_depth: int,
    max_variants: int,
) -> list[str]:
    variants = dedupe_keep_order([inline_source_helper_calls(expression, helper_index)])
    frontier = list(variants)
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for item in frontier:
            for target in sorted(target_to_sources, key=len, reverse=True):
                if not expression_mentions_symbol(item, target):
                    continue
                for source in target_to_sources[target][:4]:
                    if source == target:
                        continue
                    expanded = replace_source_symbol(item, target, f"({source})")
                    expanded = inline_source_helper_calls(expanded, helper_index)
                    if expanded not in variants and expanded not in next_frontier:
                        next_frontier.append(expanded)
                        if len(variants) + len(next_frontier) >= max_variants:
                            break
                if len(variants) + len(next_frontier) >= max_variants:
                    break
            if len(variants) + len(next_frontier) >= max_variants:
                break
        if not next_frontier:
            break
        variants.extend(next_frontier)
        frontier = next_frontier
    return variants


def inline_source_helper_calls(expression: str, helper_index: dict[str, HelperExpressionRef]) -> str:
    changed = True
    out = expression
    depth = 0
    while changed and depth < 4:
        changed = False
        depth += 1
        pieces: list[str] = []
        cursor = 0
        for match in re.finditer(
            r"(?<![#A-Za-z0-9_])(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*)\s*\(",
            out,
        ):
            name = match.group("name")
            callee = helper_index.get(name) or helper_index.get(name.split("::")[-1])
            if callee is None:
                continue
            close_paren = matching_delimiter(out, match.end() - 1, "(", ")")
            if close_paren is None:
                continue
            callee_expression = callee.lowered_return_expression or callee.return_expression
            if not callee_expression:
                continue
            args = split_top_level_args(out[match.end():close_paren])
            inlined = substitute_expression_symbols(callee_expression, callee.parameters, args)
            pieces.append(out[cursor:match.start()])
            pieces.append(f"({inlined})")
            cursor = close_paren + 1
            changed = True
        pieces.append(out[cursor:])
        out = "".join(pieces)
    return out


def matching_delimiter(text: str, open_index: int, open_char: str, close_char: str) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index
    return None


def output_signal_bindings(output_bindings: list[dict[str, Any]]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for binding in sorted(output_bindings, key=lambda item: binding_evidence_score(item)):
        source = str(binding.get("source_symbol") or "")
        logged_signal = str(binding.get("logged_signal") or "")
        if source and logged_signal and re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$", source):
            bindings.setdefault(source, logged_signal)
    return bindings


def source_assignment_signal_bindings(
    source_assignments: list[SourceAssignmentRef],
    *,
    source_path: str | Path | None = None,
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    schema = load_px4_msg_schema(source_path)
    for ref in source_assignments:
        if not ref.target_topic or ref.target_field:
            continue
        for field in schema.get(ref.target_topic, []):
            bindings[f"{ref.target}.{field}"] = f"{ref.target_topic}.{field}"
    return bindings


def output_binding_sort_key(
    binding: dict[str, Any],
    helper_index: dict[str, HelperExpressionRef],
) -> tuple[bool, int, int, str]:
    source_expression = str(binding.get("source_symbol") or "")
    return (
        source_helper_name_for_expression(source_expression, helper_index) is None,
        binding_evidence_score(binding),
        len(source_expression),
        str(binding.get("logged_signal") or ""),
    )


def binding_evidence_score(binding: dict[str, Any]) -> int:
    path = binding.get("assignment_path") or []
    return len(path) if isinstance(path, list) else 0


def source_helper_name_for_expression(
    expression: str,
    helper_index: dict[str, HelperExpressionRef],
) -> str | None:
    match = re.match(r"\s*(?P<name>[A-Za-z_][A-Za-z0-9_:]*)\s*\(", expression)
    if not match:
        return None
    ref = helper_index.get(match.group("name")) or helper_index.get(match.group("name").split("::")[-1])
    return ref.name if ref else match.group("name")


def lower_source_expression_for_evaluator(
    expression: str,
    symbol_bindings: dict[str, str],
    *,
    source_path: str | Path | None = None,
    extra_signal_bindings: dict[str, str] | None = None,
) -> dict[str, Any]:
    bindings = dict(extra_signal_bindings or {})
    bindings.update(symbol_bindings)
    lowered = expression
    lowered = re.sub(r"\btrue\b", "True", lowered)
    lowered = re.sub(r"\bfalse\b", "False", lowered)
    lowered = re.sub(r"\bPX4_ISFINITE\s*\(", "isfinite(", lowered)

    for constant, value in numeric_source_constants(source_path).items():
        lowered = re.sub(rf"(?<![A-Za-z0-9_:]){re.escape(constant)}(?![A-Za-z0-9_:])", str(value), lowered)

    for token in sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*::[A-Z][A-Z0-9_]*\b", lowered)), key=len, reverse=True):
        value = enum_constant_value_for_expression(token, bindings.values(), source_path)
        if isinstance(value, int):
            lowered = re.sub(rf"(?<![A-Za-z0-9_:]){re.escape(token)}(?![A-Za-z0-9_:])", str(value), lowered)

    variables: dict[str, str] = {}
    used_names: set[str] = set()
    for source, signal in sorted(bindings.items(), key=lambda item: len(item[0]), reverse=True):
        if source not in lowered:
            continue
        name = unique_expression_variable_name(signal, used_names)
        lowered = replace_source_symbol(lowered, source, name)
        variables[name] = signal

    for token in sorted(set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*\b", lowered)), key=len, reverse=True):
        if not is_valid_topic_field(token, source_path):
            continue
        name = unique_expression_variable_name(token, used_names)
        lowered = replace_source_symbol(lowered, token, name)
        variables[name] = token

    unresolved_symbols = unresolved_source_symbols(lowered)
    return {
        "expression": lowered,
        "variables": variables,
        "unresolved_symbols": unresolved_symbols,
    }


def enum_constant_value_for_expression(
    token: str,
    signals: Any,
    source_path: str | Path | None,
) -> Any:
    for signal in signals:
        value = normalize_px4_enum_value(str(signal), token, source_path)
        if isinstance(value, int):
            return value
    return token


def numeric_source_constants(source_path: SourceInput) -> dict[str, float | int]:
    constants: dict[str, float | int] = {}
    constant_re = re.compile(
        r"\b(?:static\s+)?constexpr\s+(?:float|double|int|uint\d+_t|int\d+_t)\s+"
        r"(?P<name>[A-Z][A-Z0-9_]*)\s*(?:=|\{)\s*(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[fF]?"
    )
    define_re = re.compile(
        r"^\s*#\s*define\s+(?P<name>[A-Z][A-Z0-9_]*)\s+"
        r"(?P<value>-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)[fF]?\b"
    )
    source = source_handle(source_path)
    if source is None:
        return {}
    texts = [
        source.read_text(path)
        for path in source.list_files("src/lib", patterns=["*.h", "*.hpp"])
    ]
    for text in texts:
        for match in constant_re.finditer(text):
            constants[match.group("name")] = parse_numeric_literal(match.group("value"))
        for line in text.splitlines():
            match = define_re.match(line)
            if match:
                constants[match.group("name")] = parse_numeric_literal(match.group("value"))
    return constants


def parse_numeric_literal(value: str) -> float | int:
    number = float(value)
    return int(number) if number.is_integer() else number


def replace_source_symbol(expression: str, symbol: str, replacement: str) -> str:
    return re.sub(
        rf"(?<![A-Za-z0-9_\.]){re.escape(symbol)}(?![A-Za-z0-9_\.])",
        replacement,
        expression,
    )


def unique_expression_variable_name(signal: str, used_names: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9_]", "_", signal).strip("_")
    if not base:
        base = "value"
    if base[0].isdigit():
        base = f"v_{base}"
    name = base
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def unresolved_source_symbols(expression: str) -> list[str]:
    unresolved = []
    for token in re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*(?:::[A-Za-z_][A-Za-z0-9_]*)?(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b", expression):
        if token not in unresolved:
            unresolved.append(token)
    return unresolved


def evaluator_expression_syntax_valid(expression: str) -> bool:
    try:
        ast.parse(expression, mode="eval")
    except SyntaxError:
        return False
    return True


def source_output_binding_candidates(
    source_assignments: list[SourceAssignmentRef],
    function_calls: list[FunctionCallRef],
) -> list[dict[str, Any]]:
    direct_edges = [
        assignment_edge(ref)
        for ref in source_assignments
        if ref.target and ref.expression
    ]
    bound_edges = call_bound_assignment_edges(source_assignments, function_calls)
    return dedupe_binding_candidates([
        *direct_edges,
        *bound_edges,
        *transitive_binding_edges([*direct_edges, *bound_edges]),
    ])


def source_output_binding_records(
    source_assignments: list[SourceAssignmentRef],
    function_calls: list[FunctionCallRef],
) -> list[SourceOutputBindingRecord]:
    return [
        SourceOutputBindingRecord(
            binding_id=source_output_binding_id(binding),
            source_symbol=str(binding.get("source_symbol") or ""),
            target_symbol=str(binding.get("target_symbol") or ""),
            logged_signal=binding.get("logged_signal"),
            assignment_path=list(binding.get("assignment_path") or []),
            control_predicates=list(binding.get("control_predicates") or []),
            symbol_bindings=dict(binding.get("symbol_bindings") or {}),
        )
        for binding in source_output_binding_candidates(source_assignments, function_calls)
    ]


def source_output_binding_id(binding: dict[str, Any]) -> str:
    text = "|".join([
        str(binding.get("source_symbol") or ""),
        str(binding.get("target_symbol") or ""),
        str(binding.get("logged_signal") or ""),
        *[str(item) for item in binding.get("control_predicates") or []],
    ])
    return f"bind_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:12]}"


def source_backed_primary_output_signals(
    requested_signals: list[str],
    output_bindings: list[SourceOutputBindingRecord],
    source_chain: list[CodeRef],
) -> list[str]:
    supported = []
    for signal in requested_signals:
        matching_bindings = [
            binding
            for binding in output_bindings
            if binding.logged_signal == signal
            and assignment_path_is_cited(binding.assignment_path, source_chain)
        ]
        if matching_bindings:
            supported.append(signal)
    return dedupe_keep_order(supported)


def assignment_path_is_cited(path: list[dict[str, Any]], source_chain: list[CodeRef]) -> bool:
    for step in path:
        file = str(step.get("file") or "")
        line = step.get("line")
        function = str(step.get("function") or "")
        for ref in source_chain:
            if not file or ref.file != file:
                continue
            if line is not None and ref.start_line is not None and ref.end_line is not None:
                if int(ref.start_line) <= int(line) <= int(ref.end_line):
                    return True
                continue
            if function and ref.function and function == ref.function:
                return True
    return False


def assignment_edge(ref: SourceAssignmentRef) -> dict[str, Any]:
    logged_signal = f"{ref.target_topic}.{ref.target_field}" if ref.target_topic and ref.target_field else None
    return {
        "source_symbol": ref.expression,
        "target_symbol": ref.target,
        "logged_signal": logged_signal,
        "control_predicates": list(ref.control_predicates),
        "symbol_bindings": dict(ref.symbol_bindings),
        "assignment_path": [
            {
                "file": ref.file,
                "line": ref.line,
                "function": ref.function,
                "evidence": ref.evidence,
            }
        ],
    }


def call_bound_assignment_edges(
    source_assignments: list[SourceAssignmentRef],
    function_calls: list[FunctionCallRef],
) -> list[dict[str, Any]]:
    by_function: dict[str, list[SourceAssignmentRef]] = {}
    for ref in source_assignments:
        if ref.function:
            by_function.setdefault(ref.function.split("::")[-1], []).append(ref)

    out: list[dict[str, Any]] = []
    for call in function_calls:
        callee_assignments = by_function.get(call.name.split("::")[-1], [])
        if not callee_assignments or not call.args:
            continue
        for assignment in callee_assignments:
            target = substitute_call_args(assignment.target, assignment.function_parameters, call.args)
            if call.receiver and "." not in target:
                target = f"{call.receiver}.{target}"
            expression = substitute_call_args(assignment.expression, assignment.function_parameters, call.args)
            control_predicates = [
                substitute_call_args(predicate, assignment.function_parameters, call.args)
                for predicate in assignment.control_predicates
            ]
            symbol_bindings = {
                substitute_call_args(source, assignment.function_parameters, call.args): signal
                for source, signal in assignment.symbol_bindings.items()
            }
            symbol_bindings.update(call.symbol_bindings)
            logged_signal = logged_signal_for_bound_target(target, call.argument_topics)
            out.append(
                {
                    "source_symbol": expression,
                    "target_symbol": target,
                    "logged_signal": logged_signal,
                    "control_predicates": dedupe_keep_order([
                        *call.control_predicates,
                        *control_predicates,
                    ]),
                    "symbol_bindings": symbol_bindings,
                    "assignment_path": [
                        {
                            "file": assignment.file,
                            "line": assignment.line,
                            "function": assignment.function,
                            "evidence": assignment.evidence,
                        },
                        {
                            "file": call.file,
                            "line": call.line,
                            "function": None,
                            "evidence": call.evidence,
                        },
                    ],
                }
            )
    return out


def substitute_call_args(expression: str, params: list[str], args: list[str]) -> str:
    if not params:
        return expression
    substituted = expression
    for root, arg in zip(params, args):
        substituted = re.sub(
            rf"(?<![A-Za-z0-9_\.]){re.escape(root)}(?![A-Za-z0-9_])",
            arg,
            substituted,
        )
    return substituted


def logged_signal_for_bound_target(target: str, argument_topics: dict[str, str]) -> str | None:
    for arg, topic in sorted(argument_topics.items(), key=lambda item: len(item[0]), reverse=True):
        arg_root, _, arg_field = arg.partition(".")
        logged_prefix = f"{topic}.{arg_field}" if arg_field else topic
        if target == arg:
            return logged_prefix
        if target.startswith(arg + "."):
            return f"{logged_prefix}.{target[len(arg) + 1:]}"
    return None


def expression_mentions_symbol(expression: str, symbol: str) -> bool:
    if not symbol:
        return False
    return symbol in expression or symbol.split(".")[-1] in expression


def dedupe_binding_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate.get("source_symbol"),
            candidate.get("target_symbol"),
            candidate.get("logged_signal"),
            tuple(candidate.get("control_predicates") or []),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out


def transitive_binding_edges(edges: list[dict[str, Any]], *, max_depth: int = 4) -> list[dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        target = edge.get("target_symbol")
        if isinstance(target, str) and target:
            by_target.setdefault(target, []).append(edge)

    expanded: list[dict[str, Any]] = []
    frontier = list(edges)
    for _ in range(max_depth):
        next_frontier: list[dict[str, Any]] = []
        for edge in frontier:
            source = edge.get("source_symbol")
            if not isinstance(source, str):
                continue
            for upstream in by_target.get(source, []):
                candidate = {
                    "source_symbol": upstream.get("source_symbol"),
                    "target_symbol": edge.get("target_symbol"),
                    "logged_signal": edge.get("logged_signal"),
                    "control_predicates": dedupe_keep_order([
                        *(upstream.get("control_predicates") or []),
                        *(edge.get("control_predicates") or []),
                    ]),
                    "symbol_bindings": {
                        **dict(upstream.get("symbol_bindings") or {}),
                        **dict(edge.get("symbol_bindings") or {}),
                    },
                    "assignment_path": [
                        *(upstream.get("assignment_path") or []),
                        *(edge.get("assignment_path") or []),
                    ],
                }
                expanded.append(candidate)
                next_frontier.append(candidate)
        frontier = next_frontier
        if not frontier:
            break
    return expanded


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


def dedupe_parameter_requirements(refs: list[ParameterRequirement]) -> list[ParameterRequirement]:
    seen = set()
    out: list[ParameterRequirement] = []
    for ref in refs:
        key = (ref.name, ref.source_predicate, ref.source_file, ref.source_line)
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


def _frontier_item_payload(store: Any, ref: Any, identity: Any) -> dict[str, Any]:
    """Render one frontier item from its current ref, or from the
    canonical record when the item reactivated without
    re-extraction. Reactivated payloads carry the full prior verdict
    history, never a blank item (spec §10A)."""
    if ref is not None:
        return compact_ref(ref)
    record = store.get(identity) if store is not None else None
    if record is None:
        return {"identity": repr(identity), "reactivated": True}
    return {
        "identity": repr(identity),
        "reactivated": True,
        "lifecycle": record.lifecycle,
        "value_revision": record.value_revision,
        "prior_verdicts": [list(verdict) for verdict in record.prior_verdicts],
    }


def _requirement_variant_descriptor(store: Any, req: Any, identity: Any) -> dict[str, Any]:
    """Lightweight decision-relevant descriptor for one requirement
    variant in the frontier (spec §8, §12)."""
    if req is not None:
        record = store.get(identity) if store is not None else None
        return {
            "name": getattr(req, "name", None),
            "gate_result": getattr(req, "gate_result", None),
            "role": getattr(req, "role", None),
            "site": f"{getattr(req, 'source_file', '')}:{getattr(req, 'source_line', '')}",
            "value_revision": record.value_revision if record is not None else 0,
        }
    return _frontier_item_payload(store, None, identity)


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
