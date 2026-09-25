"""DAG composition-closure regression tests (spec: docs/dag_composition_closure_test_spec.md).

Deterministic synthetic coverage pinning known conservative composition
behavior: C4 merged-scope retention, per-window acceptance pattern,
replay-neutrality, history/persistence refusal, contradiction scoping,
writer/mechanism separation, unknown-state handling, None semantics,
and retention-vs-authority polarity.

Every test uses generic synthetic identities (writer_a/writer_b,
window_1/window_2, candidate_alpha/candidate_beta). No benchmark
literals appear anywhere in this module.

Classification labels follow spec section 11:
- PASS CURRENT PRODUCTION
- PASS ACCEPTANCE-SIDE HELPER
- EXPECTED CONSERVATIVE LIMITATION
- WOULD REQUIRE PRECISION IMPLEMENTATION (reported, never implemented here)
"""

from __future__ import annotations

from types import SimpleNamespace

WINDOW_1 = (10.0, 20.0)
WINDOW_2 = (30.0, 40.0)


def _candidates_w1_w2(*, replay_a=None, replay_b=None):
    """Writer A overlaps window 1 only; writer B overlaps window 2 only."""
    return [
        {"key": "writer_a", "domain": [WINDOW_1], "replay_match": replay_a},
        {"key": "writer_b", "domain": [WINDOW_2], "replay_match": replay_b},
    ]


# ----------------------------------------------------------------------
# GROUP A — C4 conservative merged-scope behavior (spec section 4)
# ----------------------------------------------------------------------


def test_a1_merged_scope_retains_both_writers_without_unique_claim():
    """EXPECTED CONSERVATIVE LIMITATION: A overlaps W1 only and B
    overlaps W2 only; under the merged scope production must NOT claim
    one writer uniquely applies across the combined scope."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        _candidates_w1_w2(), [WINDOW_1, WINDOW_2]
    )
    assert outcome["eligible"] == ["writer_a", "writer_b"]
    assert outcome["unique"] is None
    assert outcome["unresolved"] is not None


def test_a2_both_writers_eligible_under_merged_scope():
    """PASS CURRENT PRODUCTION: each writer is individually eligible
    under the merged scope, so retention (not exclusion) is the
    conservative outcome."""
    from flight_log_agent.analysis.temporal_selection import (
        temporally_eligible,
    )

    merged = [WINDOW_1, WINDOW_2]
    assert temporally_eligible([WINDOW_1], merged) is True
    assert temporally_eligible([WINDOW_2], merged) is True
    assert temporally_eligible([(50.0, 60.0)], merged) is False


# ----------------------------------------------------------------------
# GROUP B — generic per-window qualification (spec section 5)
# ----------------------------------------------------------------------


def test_b1_per_window_loop_derives_w1_to_a_and_w2_to_b():
    """PASS ACCEPTANCE-SIDE HELPER: applying the existing generic
    per-window qualification independently per window derives
    W1 -> writer A and W2 -> writer B. This composition lives in the
    test only; production behavior (test A1) is unchanged."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    candidates = _candidates_w1_w2()
    per_window = {
        "window_1": discriminate_candidates(candidates, [WINDOW_1]),
        "window_2": discriminate_candidates(candidates, [WINDOW_2]),
    }
    assert per_window["window_1"]["unique"] == "writer_a"
    assert per_window["window_2"]["unique"] == "writer_b"
    merged = discriminate_candidates(candidates, [WINDOW_1, WINDOW_2])
    assert merged["unique"] is None


# ----------------------------------------------------------------------
# GROUP C — replay unavailable stays neutral (spec section 8)
# ----------------------------------------------------------------------


def test_c1_missing_windows_mean_unresolved_not_all_time():
    """PASS CURRENT PRODUCTION: EvaluationScope.from_result with no
    windows resolves to None windows plus an error, never to an
    all-time domain."""
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    scope = EvaluationScope.from_result({})
    assert scope.windows is None
    assert scope.error == "questioned condition is unresolved"


def test_c2_unresolved_scope_replay_is_not_attempted():
    """PASS CURRENT PRODUCTION: an unresolved scope yields
    not_attempted (neutral), which is distinct from mismatch."""
    from flight_log_agent.analysis.dag_replay import (
        EvaluationScope,
        replay_dag_roots,
    )

    result = replay_dag_roots(
        None,
        [],
        "generic_signal.field",
        signal_samples={},
        scope=EvaluationScope(
            windows=None, error="questioned condition is unresolved"
        ),
    )
    assert result["status"] == "not_attempted"
    assert result["complete"] is False


def test_c3_empty_window_list_replay_is_not_attempted():
    """PASS CURRENT PRODUCTION: a scope with no evaluation windows
    cannot produce a comparison verdict."""
    from flight_log_agent.analysis.dag_replay import (
        EvaluationScope,
        replay_dag_roots,
    )

    result = replay_dag_roots(
        None,
        [],
        "generic_signal.field",
        signal_samples={},
        scope=EvaluationScope(windows=()),
    )
    assert result["status"] == "not_attempted"
    assert result["complete"] is False


def test_c4_not_attempted_is_neutral_never_a_mismatch():
    """PASS CURRENT PRODUCTION: replay not_attempted never reads as
    matched or mismatched evidence (not_attempted != mismatch)."""
    from flight_log_agent.analysis.dag_replay import (
        EvaluationScope,
        replay_dag_roots,
    )

    for scope in (
        EvaluationScope(windows=None, error="questioned condition is unresolved"),
        EvaluationScope(windows=()),
    ):
        result = replay_dag_roots(
            None, [], "generic_signal.field",
            signal_samples={}, scope=scope,
        )
        assert result["status"] not in ("matched", "mismatched")
        assert result["complete"] is False


# ----------------------------------------------------------------------
# GROUP D — history/persistence insufficient evidence (spec section 7)
# ----------------------------------------------------------------------


def test_d1_ordering_fact_carries_no_persistence_claim():
    """EXPECTED CONSERVATIVE LIMITATION: a matched transition (value
    observed at T1 and again at T2 with known ordering) yields only
    the ordering fact itself; the payload carries no persistence or
    continuity claim."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    spec = SimpleNamespace(
        transition_signal="generic_signal.field",
        from_value=5.0,
        to_value=5.0,
    )
    events = derive_transition_events(
        spec, {"generic_signal.field": [(1.0, 5.0), (2.0, 5.0)]}
    )
    assert events == [{"time": 2.0, "from_value": 5.0, "to_value": 5.0}]
    assert set(events[0].keys()) == {"time", "from_value", "to_value"}
    assert "persist" not in repr(events).lower()
    assert "continu" not in repr(events).lower()


def test_d2_hidden_interval_change_breaks_endpoint_matching():
    """EXPECTED CONSERVATIVE LIMITATION: equal endpoints with a changed
    value in the hidden interval do not even match the transition, let
    alone prove persistence."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    spec = SimpleNamespace(
        transition_signal="generic_signal.field",
        from_value=5.0,
        to_value=5.0,
    )
    events = derive_transition_events(
        spec,
        {"generic_signal.field": [(1.0, 5.0), (1.5, 9.0), (2.0, 5.0)]},
    )
    assert events == []


# ----------------------------------------------------------------------
# GROUP E — retained evidence vs runtime persistence (spec section 7)
# ----------------------------------------------------------------------


def test_e1_retained_candidacy_does_not_authorize_stop():
    """PASS CURRENT PRODUCTION: evidence retention (both writers
    eligible in the frontier) combined with absent proof state must
    not manufacture stop authority. Retention != persistence, and
    retention != authority."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_proof_authority,
    )
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        _candidates_w1_w2(), [WINDOW_1, WINDOW_2]
    )
    assert outcome["eligible"] == ["writer_a", "writer_b"]
    observation = SimpleNamespace(
        relevant_obligation_keys=(("obligation_alpha", "visit_1"),),
        covered_obligation_keys=(),
        scope_degenerate=False,
    )
    authority = evaluate_proof_authority(
        observation=observation, legacy_verified=True
    )
    assert authority.writer_coverage_verified is False
    assert authority.authorizes_stop is False
    assert authority.uncovered_relevant == (("obligation_alpha", "visit_1"),)


# ----------------------------------------------------------------------
# GROUP F — contradiction isolation across windows (spec section 10.3)
# ----------------------------------------------------------------------


def test_f1_window_contradiction_composes_to_mixed_not_global_denial():
    """PASS CURRENT PRODUCTION: a contradicted window-1 branch and a
    supported window-2 branch compose to mixed at the shared verdict
    seam, so the window-2 support survives instead of being
    contaminated by the window-1 contradiction."""
    from flight_log_agent.analysis.verdict import branch_result, ceiling_for

    branch = SimpleNamespace(
        branch_id="branch_alpha",
        name="candidate_alpha",
        unresolved_dependencies=[],
        checks=[SimpleNamespace(role="mechanism_defining")],
    )
    result = branch_result(
        branch,
        [{"verdict": "contradicted"}, {"verdict": "supported"}],
        [],
    )
    assert result["verdict"] == "mixed"
    assert result["verdict"] != "contradicted"
    assert ceiling_for(result["verdict"]) == "medium"


def test_f2_per_window_view_preserves_unaffected_support():
    """PASS ACCEPTANCE-SIDE HELPER: the per-window view keeps the
    unaffected window's support intact while the composed view stays
    conservatively mixed. This separation is test-side composition of
    existing primitives; current production merged-scope behavior is
    NOT required to provide this per-window precision automatically."""
    from flight_log_agent.analysis.verdict import branch_result

    def branch_for(window):
        return SimpleNamespace(
            branch_id=f"branch_{window}",
            name=f"candidate_{window}",
            unresolved_dependencies=[],
            checks=[SimpleNamespace(role="mechanism_defining")],
        )

    window_1 = branch_result(
        branch_for("window_1"), [{"verdict": "contradicted"}], []
    )
    window_2 = branch_result(
        branch_for("window_2"), [{"verdict": "supported"}], []
    )
    assert window_1["verdict"] == "contradicted"
    assert window_2["verdict"] == "supported"
    composed = branch_result(
        branch_for("composed"),
        [{"verdict": window_1["verdict"]},
         {"verdict": window_2["verdict"]}],
        [],
    )
    assert composed["verdict"] == "mixed"


# ----------------------------------------------------------------------
# GROUP G — equivalent writers across windows (spec section 10.4)
# ----------------------------------------------------------------------


def test_g1_mechanism_conclusion_stands_while_writer_unresolved():
    """PASS CURRENT PRODUCTION: a mechanism-level supported verdict
    survives an unresolved exact-writer question, while writer
    selection itself stays unresolved (no invented writer
    equivalence authority)."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )
    from flight_log_agent.analysis.verdict import combine_verdicts

    mechanism = combine_verdicts("supported", "unresolved")
    assert mechanism == "supported"
    selection = discriminate_candidates(
        _candidates_w1_w2(), [WINDOW_1, WINDOW_2]
    )
    assert selection["unique"] is None
    assert selection["unresolved"] is not None


# ----------------------------------------------------------------------
# GROUP H — unknown -> mixed pinning (spec section 8)
# ----------------------------------------------------------------------


def test_h1_graph_unknown_reduces_to_mixed():
    """PASS CURRENT PRODUCTION: _from_graph unknown verdicts reduce to
    mixed (a mixed verdict authorizes no stronger conclusion than
    unknown)."""
    from flight_log_agent.analysis.verdict import aggregate

    assert (
        aggregate(
            [{"verdict": "unknown"}, {"verdict": "unknown"}],
            level="graph",
        )
        == "mixed"
    )


def test_h2_single_unknown_reduces_to_mixed():
    """PASS CURRENT PRODUCTION: one unknown graph verdict alone is
    mixed, never silently satisfiable."""
    from flight_log_agent.analysis.verdict import aggregate

    assert aggregate([{"verdict": "unknown"}], level="graph") == "mixed"


def test_h3_mixed_from_unknown_caps_confidence_below_high():
    """PASS CURRENT PRODUCTION: the unknown-derived mixed verdict caps
    at medium confidence; mixed can never authorize a high-confidence
    conclusion."""
    from flight_log_agent.analysis.verdict import aggregate, ceiling_for

    verdict = aggregate([{"verdict": "unknown"}], level="graph")
    assert ceiling_for(verdict) == "medium"
    assert ceiling_for(verdict) != "high"


def test_h4_graph_mixed_with_unknown_stays_mixed():
    """PASS CURRENT PRODUCTION: mixing an already-mixed graph result
    with unknown evidence stays mixed."""
    from flight_log_agent.analysis.verdict import aggregate

    assert (
        aggregate(
            [{"verdict": "mixed"}, {"verdict": "unknown"}],
            level="graph",
        )
        == "mixed"
    )


# ----------------------------------------------------------------------
# GROUP I — dag_observation / scope None semantics (spec section 9)
# ----------------------------------------------------------------------


def test_i1_from_result_without_windows_is_unresolved():
    """PASS CURRENT PRODUCTION: absent windows parse to None windows
    with an error (unresolved), never to an all-time domain."""
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    scope = EvaluationScope.from_result({})
    assert scope.windows is None
    assert scope.error == "questioned condition is unresolved"


def test_i2_absent_scope_keeps_selection_inactive():
    """PASS CURRENT PRODUCTION: a None scope keeps temporal selection
    inactive (returns None) so callers preserve existing behavior
    exactly."""
    from flight_log_agent.analysis.dag_pipeline import (
        _temporal_report_selection,
    )

    assert _temporal_report_selection(None, None, None) is None


def test_i3_none_windows_fail_closed_with_notes():
    """PASS CURRENT PRODUCTION: a scope with None windows fails closed
    to explanatory notes with no filtering and no unique claim,
    preserving broader evidence."""
    from flight_log_agent.analysis.dag_pipeline import (
        _temporal_report_selection,
    )
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    scope = EvaluationScope(
        windows=None, error="questioned condition is unresolved"
    )
    outcome = _temporal_report_selection(None, scope, None)
    assert outcome["unique_terminal_id"] is None
    assert outcome["filter_active"] is False
    assert any("no temporally-selected causal claim" in note
               for note in outcome["notes"])


def test_i4_vertex_without_correspondence_yields_no_check():
    """PASS CURRENT PRODUCTION: with no observation correspondences,
    local equation evaluation yields no checks at all — a vertex with
    no correspondence is never silently satisfiable."""
    from flight_log_agent.analysis.dag_observation import (
        evaluate_local_observed_equations,
        observation_correspondences,
    )

    dag = SimpleNamespace(observation_witnesses=[])
    program = SimpleNamespace(vertices={}, compiled_vertices={})
    assert observation_correspondences(dag, program) == []
    checks = evaluate_local_observed_equations(
        dag,
        program,
        signal_samples={},
        parameter_values={},
        signal_policies={},
        scope=None,
        relevant_ids=set(),
    )
    assert checks == []
    # Zero correspondences means no replay checks are produced, so no
    # matched/mismatched state can be inferred from this path.


# ----------------------------------------------------------------------
# GROUP J — retention polarity vs authority polarity (spec section 9)
# ----------------------------------------------------------------------


def test_j1_unknown_temporal_eligibility_retains_candidate():
    """PASS CURRENT PRODUCTION: unknown feasibility without windows
    constrains nothing, so the candidate stays eligible (retention is
    not a claim)."""
    from flight_log_agent.analysis.temporal_selection import writer_domain

    domain = writer_domain(
        replay_domain=None,
        branch_gates=[("unknown", [])],
        scope_windows=[WINDOW_1, WINDOW_2],
    )
    assert domain == [WINDOW_1, WINDOW_2]


def test_j2_unknown_proof_state_blocks_stop_authority():
    """PASS CURRENT PRODUCTION: with an active proof gate but no
    current coverage, authority stays blocked even when legacy
    verification passes. Candidate retention does not equal
    applicability satisfaction or stop authority."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_proof_authority,
    )

    observation = SimpleNamespace(
        relevant_obligation_keys=(("obligation_alpha", "visit_1"),),
        covered_obligation_keys=(),
        scope_degenerate=False,
    )
    authority = evaluate_proof_authority(
        observation=observation,
        proof_version="version_9",
        legacy_verified=True,
    )
    assert authority.gate_active is True
    assert authority.coverage_ok is False
    assert authority.applicability_ok is True
    assert authority.authorizes_stop is False


def test_j3_vacuous_scope_cannot_authorize_stop():
    """PASS CURRENT PRODUCTION: a vacuous (degenerate, empty-relevance)
    scope cannot authorize stop even when legacy verification passes.
    Legacy verified != authority when proof is vacuous."""
    from flight_log_agent.analysis.checkpoint_discovery import (
        evaluate_proof_authority,
    )

    observation = SimpleNamespace(
        relevant_obligation_keys=(),
        covered_obligation_keys=(),
        scope_degenerate=True,
    )
    authority = evaluate_proof_authority(
        observation=observation,
        proof_version="version_9",
        legacy_verified=True,
    )
    assert authority.non_vacuous_ok is False
    assert authority.authorizes_stop is False


# ----------------------------------------------------------------------
# GROUP K — combined synthetic cases (spec section 10)
# ----------------------------------------------------------------------


def test_k1_writers_plus_windows_plus_unavailable_replay():
    """PASS CURRENT PRODUCTION (matrix item 1): multiple writers with
    per-window applicability and unavailable replay produce no false
    exact-writer claim and no authority upgrade."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        _candidates_w1_w2(replay_a=False, replay_b=None),
        [WINDOW_1, WINDOW_2],
    )
    assert outcome["eligible"] == ["writer_a", "writer_b"]
    assert outcome["unique"] is None
    assert outcome["unresolved"] is not None


def test_k2_retained_evidence_plus_later_window_without_proof():
    """EXPECTED CONSERVATIVE LIMITATION (matrix item 2): evidence
    stays available in a later window while the persistence question
    stays unresolved; the outcome carries no manufactured continuity
    claim."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        _candidates_w1_w2(), [WINDOW_2]
    )
    assert outcome["eligible"] == ["writer_b"]
    assert set(outcome.keys()) == {"eligible", "unique", "unresolved"}
    assert "persist" not in repr(outcome).lower()
    assert "continu" not in repr(outcome).lower()


def test_k3_per_window_discrimination_plus_one_window_contradiction():
    """PASS CURRENT PRODUCTION (matrix item 3): per-window
    discrimination (W1 -> A, W2 -> B) composed with a window-1
    contradiction yields a conservatively mixed global view while the
    unaffected window-2 support is preserved in the per-window view.
    The per-window separation here is test-side composition of
    existing primitives; current production merged-scope behavior is
    NOT required to provide this per-window precision automatically."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )
    from flight_log_agent.analysis.verdict import branch_result

    candidates = _candidates_w1_w2()
    assert discriminate_candidates(candidates, [WINDOW_1])["unique"] == "writer_a"
    assert discriminate_candidates(candidates, [WINDOW_2])["unique"] == "writer_b"

    def branch_for(window):
        return SimpleNamespace(
            branch_id=f"branch_{window}",
            name=f"candidate_{window}",
            unresolved_dependencies=[],
            checks=[SimpleNamespace(role="mechanism_defining")],
        )

    composed = branch_result(
        branch_for("composed"),
        [{"verdict": "contradicted"}, {"verdict": "supported"}],
        [],
    )
    assert composed["verdict"] == "mixed"
    window_2 = branch_result(
        branch_for("window_2"), [{"verdict": "supported"}], []
    )
    assert window_2["verdict"] == "supported"


def test_k4_equivalent_writers_mechanism_stands_writer_open():
    """PASS CURRENT PRODUCTION (matrix item 4): mechanism-level
    reasoning may proceed on a supported verdict while the exact
    execution writer across windows stays unresolved."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )
    from flight_log_agent.analysis.verdict import combine_verdicts

    mechanism = combine_verdicts("supported", "unresolved")
    assert mechanism == "supported"
    selection = discriminate_candidates(
        [
            {"key": "writer_a", "domain": [WINDOW_1], "replay_match": None},
            {"key": "writer_b", "domain": [WINDOW_1], "replay_match": None},
        ],
        [WINDOW_1],
    )
    assert selection["eligible"] == ["writer_a", "writer_b"]
    assert selection["unique"] is None
