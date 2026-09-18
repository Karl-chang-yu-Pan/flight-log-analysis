"""Checkpoint-driven source discovery over the one mechanism DAG.

No question is reinterpreted here. An exact observed question target, or the
explicit discovery terminal when no comparison is supplied, defines scope.
A verified stop closes that source investigation; the judge still decides
whether the established mechanism answers the user's question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Sequence

from flight_log_agent.analysis.dag_checkpoint import assess_checkpoint, dependency_view
from flight_log_agent.analysis.dag_replay import EvaluationScope, observed_checkpoint_roots
from flight_log_agent.analysis.dag_value import DAGValueProgram
from flight_log_agent.analysis.dag_observation import evaluate_local_observed_equations
from flight_log_agent.analysis.mechanism_dag import (
    ConstructionDemand, MechanismDAG, evaluate_feasibility, prepare_signal_series, sample_prepared_signal,
)
from flight_log_agent.analysis.source_expansion import UnresolvedSourceReference


@dataclass(frozen=True)
class CheckpointProofObservation:
    """Record-only coverage observation for one checkpoint round.

    Diagnostic only: relevance and coverage bookkeeping with zero
    scheduling, retirement, applicability, or stop meaning. Key
    namespaces are visit keys (scheduling identity, computable from the
    reference alone) except `covered_semantic_keys`, which retains the
    proven semantic obligation keys of the covering certificates for
    later applicability work.

    A certificate is observable only when its scheduling key belongs to
    the relevance set, its version equals the supplied proof version,
    and its obligation/boundary fields are intact. Anything else is
    excluded (unrelated scope) or reported as stale (present but not
    current). Filtering never removes relevance: an obligation filtered
    from current scheduling stays relevant and, without a current
    certificate, uncovered.
    """

    relevant_obligation_keys: tuple = ()
    covered_obligation_keys: tuple = ()
    uncovered_obligation_keys: tuple = ()
    filtered_obligation_keys: tuple = ()
    stale_certificate_keys: tuple = ()
    covered_semantic_keys: tuple = ()
    proof_version: Any = None
    scope_degenerate: bool = False
    relevant_empty: bool = True


def observe_checkpoint_coverage(
    references: Any,
    *,
    certificates: Any = (),
    proof_version: Any = None,
    searchable_keys: Any = None,
    scope_degenerate: bool = False,
) -> CheckpointProofObservation:
    """Observe current coverage proof without deciding anything.

    Pure function over explicit inputs: the pre-filter relevance base
    (references entering the round, before exhausted/queue filtering),
    the certificates to consider, the current proof version, and the
    visit keys schedulable through any current path (ordinary searchable
    plus local-request scheduling). Reads no session, scheduler, or
    checkpoint state. Never authorizes, retires, or filters. Relevance
    entries are deduplicated by visit key, references first.
    """
    seen: set = set()
    relevant: list = []
    for reference in references or ():
        key = reference.visit_key()
        if key not in seen:
            seen.add(key)
            relevant.append(key)
    relevant_set = set(relevant)
    if searchable_keys is None:
        searchable = set(relevant)
    else:
        searchable = set(searchable_keys)
    covered: list = []
    covered_semantic: list = []
    stale: list = []
    for certificate in certificates or ():
        scheduling_key = tuple(
            getattr(certificate, "scheduling_key", None) or ())
        if scheduling_key not in relevant_set:
            # Outside the relevance scope: excluded entirely, not even
            # reported as stale. A foreign certificate must never appear
            # associated with an unrelated obligation.
            continue
        intact = bool(getattr(certificate, "obligation_key", None)) and bool(
            getattr(certificate, "boundary", None))
        current = (
            proof_version is not None
            and getattr(certificate, "version", None) == proof_version
            and intact
        )
        if current:
            if scheduling_key not in covered:
                covered.append(scheduling_key)
                covered_semantic.append(
                    tuple(certificate.obligation_key))
        elif scheduling_key not in stale:
            stale.append(scheduling_key)
    covered_set = set(covered)
    return CheckpointProofObservation(
        relevant_obligation_keys=tuple(relevant),
        covered_obligation_keys=tuple(covered),
        uncovered_obligation_keys=tuple(
            key for key in relevant if key not in covered_set),
        filtered_obligation_keys=tuple(
            key for key in relevant if key not in searchable),
        stale_certificate_keys=tuple(stale),
        covered_semantic_keys=tuple(covered_semantic),
        proof_version=proof_version,
        scope_degenerate=bool(scope_degenerate),
        relevant_empty=not relevant,
    )


@dataclass(frozen=True)
class ProofAuthority:
    """Stop-authority verdict for one checkpoint round (T6B).

    Pure diagnostic value computed by `evaluate_proof_authority`:
    `authorizes_stop` conjoins the legacy verified decision (computed
    unchanged by the round) with proof conditions. Omitted proof
    inputs (`gate_active` false, i.e. no proof version threaded) are
    required proof absent, never a disengaged gate: non-empty
    relevance vetoes stop, while an empty relevance set under a
    non-degenerate scope follows the legacy verdict (the
    genuinely-no-work path) with both positive verifications false,
    so legacy callers observe zero change on that path only.
    """

    gate_active: bool
    legacy_verified: bool
    coverage_ok: bool
    applicability_ok: bool
    non_vacuous_ok: bool
    writer_coverage_verified: bool
    applicability_verified: bool
    authorizes_stop: bool
    uncovered_relevant: tuple = ()
    missing_applicability_uses: tuple = ()
    conflicting_applicability_uses: tuple = ()


def _is_current_proof_certificate(
    certificate: Any,
    proof_version: Any,
) -> bool:
    """Whether a certificate is well-formed and current-versioned.

    Mirrors the currency/intactness rule `observe_checkpoint_coverage`
    applies when associating coverage (same rule, pointed here rather
    than refactored, so T4 observation behavior stays frozen): a
    version match plus non-empty obligation and boundary identity.
    Scheduling-key association happens at the call site.
    """
    return (
        getattr(certificate, "version", None) == proof_version
        and bool(getattr(certificate, "obligation_key", None))
        and bool(getattr(certificate, "boundary", None))
    )


def collect_applicability_uses(
    references: Any,
    local_needs: Any,
) -> tuple:
    """Concrete (use, obligation) pairs for applicability matching.

    A use is an (origin vertex, operand) pair; its obligation is the
    scheduling visit key it was derived alongside. Reference origins
    and operands accumulate independently upstream, so the pairing is
    the conservative cartesian product (extra pairs fail closed: an
    unprovable pair vetoes). Local needs contribute exact
    (vertex, operand) pairs bound to each attached request. Empty
    origins/operands contribute nothing: a use without a concrete
    vertex and operand cannot govern a value. Order-preserving,
    deduplicated.
    """
    uses: list = []

    def _add(use: Any, obligation: Any) -> None:
        entry = (tuple(use), tuple(obligation))
        if entry not in uses:
            uses.append(entry)

    for reference in references or ():
        try:
            visit = tuple(reference.visit_key())
        except (AttributeError, TypeError, ValueError):
            continue
        origins = list(getattr(reference, "origin_vertex_ids", None) or ())
        operands = list(getattr(reference, "origin_operands", None) or ())
        for origin in origins:
            for operand in operands:
                if origin and operand:
                    _add((origin, operand), visit)
    for need in local_needs or ():
        if not isinstance(need, dict):
            continue
        vertex = need.get("vertex_id") or ""
        operand = need.get("operand") or ""
        if not vertex or not operand:
            continue
        for raw in need.get("source_requests", None) or ():
            try:
                visit = UnresolvedSourceReference.model_validate(
                    raw).visit_key()
            except (AttributeError, TypeError, ValueError):
                continue
            _add((vertex, operand), visit)
    return tuple(uses)


def evaluate_proof_authority(
    *,
    observation: Any,
    certificates: Any = (),
    applicability_proofs: Any = (),
    proof_version: Any = None,
    legacy_verified: bool = False,
    applicability_uses: Any = (),
) -> ProofAuthority:
    """Decide stop authority from legacy verdict plus proof state.

    Pure function: `stop = legacy AND coverage AND applicability AND
    non-vacuity`. Coverage requires every relevant obligation
    currently covered; applicability requires every use of a covered
    obligation to hold a current proof bound to the same use,
    scheduling visit, obligation, and writer set; non-vacuity requires
    a non-empty relevance set, or an empty one only under a
    non-degenerate scope (the genuinely-no-work path, which defers to
    legacy). Omitted proof inputs (`proof_version` None) are required
    proof absent, never a disengaged gate: non-empty relevance vetoes,
    empty non-degenerate scope follows legacy, degenerate scope vetoes.
    """
    legacy = bool(legacy_verified)
    gate_active = proof_version is not None
    relevant = list(getattr(observation, "relevant_obligation_keys", None)
                    or ())
    observed_covered = set(
        getattr(observation, "covered_obligation_keys", None) or ())
    # Without a proof version no certificate can be current: coverage
    # claims from another version never carry authority.
    covered = observed_covered if proof_version is not None else set()
    uncovered = [key for key in relevant if key not in covered]
    degenerate = bool(getattr(observation, "scope_degenerate", False))
    non_vacuous_ok = bool(relevant) or not degenerate
    coverage_ok = not uncovered
    writer_coverage_verified = bool(relevant) and coverage_ok
    if proof_version is None:
        current_certs: list = []
        current_proofs: list = []
    else:
        current_certs = [
            certificate for certificate in certificates or ()
            if _is_current_proof_certificate(certificate, proof_version)
        ]
        current_proofs = [
            proof for proof in applicability_proofs or ()
            if getattr(proof, "version", None) == proof_version
        ]

    def _certs_for(obligation_visit: Any) -> list:
        try:
            wanted = tuple(obligation_visit)
        except TypeError:
            return []
        return [
            certificate for certificate in current_certs
            if tuple(getattr(certificate, "scheduling_key", None) or ())
            == wanted
        ]

    def _proofs_for(use: Any, obligation_visit: Any) -> list:
        """Candidate proofs for one concrete scheduled use.

        Coverage may be shared across visits; applicability may not:
        the proof's scheduling key must equal the required visit (same
        call scope), alongside use, version, obligation, and writer
        bindings checked by `_matched_pairs`. Basis labels play no
        role: distinct positive bases corroborate one claim.
        """
        try:
            visit = tuple(obligation_visit)
        except TypeError:
            return []
        return [
            proof for proof in current_proofs
            if tuple(getattr(proof, "use_key", None) or ()) == tuple(use)
            and tuple(getattr(proof, "scheduling_key", None) or ())
            == visit
        ]

    def _matched_pairs(use: Any, obligation_visit: Any) -> list:
        """Distinct (obligation, writers) pairs bound to current proof.

        A pair counts only when some current certificate for the visit
        carries the same obligation with the same writer set. Proofs
        that mismatch version, scheduling, obligation, or writers are
        irrelevant to this claim and ignored — they neither satisfy
        nor conflict.
        """
        try:
            visit = tuple(obligation_visit)
        except TypeError:
            return []
        matched: list = []
        for proof in _proofs_for(use, visit):
            pair = (
                tuple(getattr(proof, "obligation_key", None) or ()),
                tuple(getattr(proof, "writers", None) or ()),
            )
            if pair in matched:
                continue
            if any(
                tuple(getattr(certificate, "obligation_key", None) or ())
                == pair[0]
                and set(getattr(certificate, "writers", None) or ())
                == set(pair[1])
                and tuple(
                    getattr(certificate, "scheduling_key", None) or ())
                == visit
                for certificate in _certs_for(visit)
            ):
                matched.append(pair)
        return matched

    def _use_proven(use: Any, obligation_visit: Any) -> bool:
        return bool(_matched_pairs(use, obligation_visit))

    required = [use for use, visit in (applicability_uses or ())
                if tuple(visit) in covered]
    missing = [
        use for use, visit in (applicability_uses or ())
        if tuple(visit) in covered
        and not _use_proven(use, visit)
    ]
    conflicting: list = []
    for use, visit in (applicability_uses or ()):
        # Only otherwise-binding proofs participate: distinct valid
        # (obligation, writers) claims for one scheduled use are
        # genuinely ambiguous and veto. Irrelevant proofs never reach
        # `_matched_pairs`, and basis labels are not compared.
        if len(_matched_pairs(use, visit)) > 1 \
                and tuple(use) not in conflicting:
            conflicting.append(tuple(use))
    applicability_ok = not missing and not conflicting
    applicability_verified = bool(required) and applicability_ok
    authorizes_stop = bool(
        legacy and coverage_ok and applicability_ok and non_vacuous_ok)
    return ProofAuthority(
        gate_active=bool(gate_active),
        legacy_verified=legacy,
        coverage_ok=bool(coverage_ok),
        applicability_ok=bool(applicability_ok),
        non_vacuous_ok=bool(non_vacuous_ok),
        writer_coverage_verified=bool(writer_coverage_verified),
        applicability_verified=bool(applicability_verified),
        authorizes_stop=bool(authorizes_stop),
        uncovered_relevant=tuple(uncovered),
        missing_applicability_uses=tuple(missing),
        conflicting_applicability_uses=tuple(conflicting),
    )


@dataclass
class CheckpointRound:
    action: Literal["continue", "verified", "unresolved"]
    annotated: MechanismDAG
    references: list[UnresolvedSourceReference]
    summary: dict[str, Any]
    construction: ConstructionDemand = ConstructionDemand()
    # Record-only coverage observation (T4, diagnostic-only). Computed
    # from the pre-filter relevance base on every round, with or without
    # threaded proof state. Never influences scheduling, requirements,
    # flags, or stop. Kept off `summary` so no report/schema field moves.
    proof_observation: Optional[CheckpointProofObservation] = None
    # Proof-authority verdict (T6B). Always attached; kept off `summary`
    # like proof observation. Only this field (via the gated `verified`
    # decision) may authorize stop, and only under the full conjunction.
    proof_authority: Optional[ProofAuthority] = None


def evaluate_checkpoint_round(
    dag: MechanismDAG,
    *,
    parameter_values: dict[str, Any],
    observed_signals: set[str],
    signal_policies: dict[str, Any],
    load_samples: Callable[[MechanismDAG, tuple[str, ...]], dict[str, list[tuple[float, Any]]]],
    scope: Optional[EvaluationScope] = None,
    question_target: Optional[str] = None,
    coverage_certificates: Sequence[Any] = (),
    proof_version: Any = None,
    applicability_proofs: Sequence[Any] = (),
) -> CheckpointRound:
    """Preflight first, evaluate ready gates, replay, then select exact needs.

    No file/round budget, guessed binding, or numerical-only stop. An unresolved
    source-writer request remains a proof obligation even after a local match.

    `coverage_certificates` with `proof_version` threads T3 proof state for
    record-only observation: relevance and coverage bookkeeping are computed
    diagnostically and cannot change scheduling, requirements, flags, or
    stop. Omitting them computes the same relevance with empty coverage.
    `applicability_proofs` threads T5 proofs for the stop-authority gate:
    stop additionally requires every required concrete use to hold a
    current proof bound to the same use, visit, obligation, and writer
    set. Omitted proof inputs are required proof absent: non-empty
    relevance vetoes stop, while empty non-degenerate scope follows
    the legacy verdict.
    """
    all_groups = observed_checkpoint_roots(dag)
    groups = all_groups
    terminal_ids = {v.id for v in dag.vertices if v.metadata.get("is_terminal")}
    if question_target is not None:
        groups = {signal: roots for signal, roots in groups.items() if signal == question_target}
    else:
        groups = {signal: roots for signal, roots in groups.items() if terminal_ids.intersection(roots)}
    targets = groups or {None: tuple(sorted(terminal_ids))}
    roots = {root for ids in targets.values() for root in ids}
    view = dependency_view(dag, roots)
    target_dependencies = {v.id for v in view.vertices}
    observed_dependencies = {item["signal"] for item in dag.observation_witnesses
                             if item["value_id"] in target_dependencies}
    intermediate_groups = {
        signal: ids for signal, ids in all_groups.items()
        if signal not in groups and (target_dependencies.intersection(ids) or signal in observed_dependencies)
    }
    # A local comparison must include alternative writers of that same
    # publication, not merely whichever writer the terminal walk encountered.
    view = dependency_view(dag, roots | {root for ids in intermediate_groups.values() for root in ids})
    samples = load_samples(view, tuple(s for s in (*groups, *intermediate_groups) if s in observed_signals))
    prepared = prepare_signal_series(samples, signal_policies)
    program = DAGValueProgram(view)
    session = program.bind(
        parameter_values=parameter_values,
        sample_resolver=lambda signal, timestamp: sample_prepared_signal(prepared, signal, timestamp),
    )
    kwargs = dict(
        source_dag=dag, parameter_values=parameter_values,
        observed_signals=observed_signals, signal_samples=samples,
        signal_policies=signal_policies, prepared_signal_series=prepared,
        value_program=program, value_session=session, scope=scope,
    )
    preflight = {signal: assess_checkpoint(view, ids, signal, attempt_replay=False, **kwargs)
                 for signal, ids in targets.items()}

    # Only graph-complete gate inputs can justify dynamic scheduling. Opaque
    # dependencies do not consume a whole log merely to fail at each timestamp.
    blocked: set[str] = set()
    unscoped_frontier = False
    for checkpoint in preflight.values():
        for requirement in checkpoint["analysis_requirements"]:
            if requirement["reason"] == "gate has no complete evaluation over the comparison domain":
                continue
            if requirement["kind"] == "observation_binding" and not requirement.get("vertex_id"):
                continue
            vertex_id = requirement.get("vertex_id")
            blocked.update(requirement.get("origin_vertex_ids") or ())
            if vertex_id:
                blocked.add(vertex_id)
            elif requirement.get("source_reference"):
                unscoped_frontier = True
    ready: set[str] = set()
    if not unscoped_frontier:
        for vertex in view.vertices:
            if vertex.kind != "branch":
                continue
            dependencies = dependency_view(view, [vertex.id])
            if not blocked.intersection(v.id for v in dependencies.vertices):
                ready.add(vertex.id)
    annotated_view = evaluate_feasibility(
        view, parameter_values=parameter_values, signal_samples=samples,
        signal_policies=signal_policies, prepared_signal_series=prepared,
        value_program=program, value_session=session, prune_dead=False,
        dynamic_branch_ids=ready, allow_assumptions=False, stream_timestamps=True,
    )
    checkpoints = {signal: assess_checkpoint(annotated_view, ids, signal, **kwargs)
                   for signal, ids in targets.items()}
    intermediates = {signal: assess_checkpoint(annotated_view, ids, signal, **kwargs)
                     for signal, ids in intermediate_groups.items()}
    local_equations = evaluate_local_observed_equations(
        view, program, signal_samples=samples, parameter_values=parameter_values,
        signal_policies=signal_policies, scope=scope, relevant_ids=target_dependencies,
        prepared_signal_series=prepared,
    )
    selected = next(iter(checkpoints.values())) if len(checkpoints) == 1 else None
    legacy_verified = bool(
        selected and selected["observed"] and selected["status"] == "matched"
        and selected["complete"] and not selected["analysis_requirements"]
        and not selected["source_requests"]
        and not any(item["status"] == "mismatched" for item in intermediates.values())
    )
    wanted = {UnresolvedSourceReference.model_validate(raw).visit_key()
              for checkpoint in checkpoints.values() for raw in checkpoint["source_requests"]}
    references = [r for r in dag.unresolved_references if r.visit_key() in wanted]
    inactive = {vertex_id for checkpoint in checkpoints.values()
                for vertex_id in checkpoint["inactive_writer_ids"]}
    required = {requirement["vertex_id"] for checkpoint in checkpoints.values()
                for requirement in checkpoint["analysis_requirements"]
                if requirement["kind"] == "construction"}
    relevant = {vertex_id for checkpoint in checkpoints.values()
                for vertex_id in checkpoint["dependency_vertex_ids"]}
    gate_roots = [v.id for v in annotated_view.vertices if v.id in relevant and v.kind == "branch"
                  and v.feasibility_verdict == "unknown"]
    gate_inputs = {v.id for v in dependency_view(annotated_view, gate_roots).vertices}
    searchable = [reference for reference in references
                  if reference.visit_key() not in dag.exhausted_source_requests]
    gate_requests = [reference for reference in searchable
                     if gate_inputs.intersection(reference.origin_vertex_ids)]
    gate_work = required & gate_inputs
    local_needs = [need for check in local_equations
                   for need in check.get("input_requirements", [])]
    local_work = {need["vertex_id"] for need in local_needs if need["kind"] == "construction"}
    local_work &= set(dag.pending_construction) - inactive
    local_keys = {UnresolvedSourceReference.model_validate(raw).visit_key()
                  for need in local_needs for raw in need["source_requests"]
                  if need["vertex_id"] not in inactive}
    local_requests = [reference for reference in dag.unresolved_references
                      if reference.visit_key() in local_keys
                      and reference.visit_key() not in dag.exhausted_source_requests]
    # A missing guard helper is source work, not permission to expand its
    # guarded values. If that exact search is exhausted, retain the unknown
    # obligation but allow other useful work to proceed.
    # Conditional equation construction is not an applicability decision.
    # Its exact value prerequisites may proceed while unrelated guard source
    # is unknown. Only proven inactivity can discharge either kind of work.
    if local_work:
        materialize, next_references, kind = local_work, [], "local_calculation_construction"
    elif local_requests:
        materialize, next_references, kind = set(), local_requests, "local_calculation_source"
    elif gate_work:
        materialize, next_references, kind = gate_work, [], "guard_construction"
    elif gate_requests:
        materialize, next_references, kind = set(), gate_requests, "guard_source"
    else:
        local_inputs = {vertex_id for item in intermediates.values()
                        for vertex_id in item["dependency_vertex_ids"]}
        local_work = required & local_inputs
        materialize = local_work or required
        next_references = [] if materialize else searchable
        kind = "calculation_construction" if materialize else "source" if searchable else "blocked"
    construction = ConstructionDemand(
        materialize=frozenset(materialize),
        inactive=frozenset(inactive),
    )
    # Record-only proof observation (T4 F1), computed here (before the
    # stop decision) so the authority gate can consume it: relevance
    # comes from the pre-filter `references` base above UNION the active
    # local-request obligations below — never from the exhausted- or
    # queue-filtered views — so filtering cannot erase relevance. Local
    # needs attach obligations (e.g. by exact declaration identity) that
    # the checkpoint-wanted set may omit; those obligations still
    # participate in this round's analysis and must stay visible to
    # proof. Reads only; scheduling, requirements, flags, and stop are
    # already decided and untouched below.
    relevance_references = list(references)
    relevance_keys = {reference.visit_key() for reference in references}
    for reference in dag.unresolved_references:
        if (reference.visit_key() in local_keys
                and reference.visit_key() not in relevance_keys):
            relevance_keys.add(reference.visit_key())
            relevance_references.append(reference)
    proof_observation = observe_checkpoint_coverage(
        relevance_references,
        certificates=coverage_certificates,
        proof_version=proof_version,
        searchable_keys=(
            {reference.visit_key() for reference in searchable}
            | {reference.visit_key() for reference in local_requests}
        ),
        scope_degenerate=not roots,
    )
    # Proof-gated stop authority (T6B): the legacy verified decision
    # above is necessary but no longer sufficient on its own. Stop
    # additionally requires the proof conjunction; omitted proof inputs
    # count as required proof absent (non-empty relevance vetoes, empty
    # non-degenerate scope follows legacy).
    proof_authority = evaluate_proof_authority(
        observation=proof_observation,
        certificates=coverage_certificates,
        applicability_proofs=applicability_proofs,
        proof_version=proof_version,
        legacy_verified=legacy_verified,
        applicability_uses=collect_applicability_uses(
            relevance_references, local_needs),
    )
    verified = legacy_verified and proof_authority.authorizes_stop
    if verified:
        selected["authorizes_discovery_stop"] = True
        selected["verification_scope"] = "questioned_signal" if question_target is not None else "terminal"
    action = "verified" if verified else "continue" if next_references or materialize else "unresolved"
    updates = {v.id: v for v in annotated_view.vertices}
    annotated = dag.model_copy(update={"vertices": [updates.get(v.id, v) for v in dag.vertices]})
    return CheckpointRound(action, annotated, next_references, {
        "action": action, "question_target": question_target,
        "checkpoints": {signal: value for signal, value in checkpoints.items() if signal is not None},
        "terminal_checkpoint": checkpoints.get(None),
        "selected_checkpoint": selected,
        "intermediate_checkpoints": intermediates,
        "local_equation_checks": local_equations,
        "next_analysis": {
            "kind": kind, "operation_ids": sorted(materialize),
            "guard_ids": gate_roots,
            "source_requests": [reference.model_dump(mode="json") for reference in next_references],
        },
        "preflight": {signal or "internal_terminal": value for signal, value in preflight.items()},
        "dynamic_gate_count": len(ready),
        "pending_construction_count": len(dag.pending_construction),
        "reason": "source-backed checkpoint verified" if verified else "checkpoint has outstanding analysis requirements",
    }, construction=construction, proof_observation=proof_observation,
        proof_authority=proof_authority)
