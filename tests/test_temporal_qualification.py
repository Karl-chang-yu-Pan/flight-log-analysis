"""W3A temporal qualification unit tests (slices 1-8, T1-T13).

Pure derivation/selection behavior first; live-log TECS acceptance
(T14-T19) lives in tests/test_acceptance_tecs.py once STOP G is clear.
"""

from __future__ import annotations


def _tecs_like_intent(**overrides):
    """TECS-shaped transition intent; case values are test data only."""
    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    fields = {
        "transition_signal": "vehicle_status.nav_state",
        "from_value": 15,
        "to_value": 4,
        "event_selection": "first",
        "relation": "after",
        "first_sample_of": "tecs_status.height_rate_setpoint",
    }
    fields.update(overrides)
    return TransitionEventSpec(**fields)


def test_t1_questioned_condition_without_transition_spec_unchanged():
    """T1: no temporal seed → QuestionedCondition behaves exactly as before."""
    from flight_log_agent.analysis.mechanism_judge import QuestionedCondition

    condition = QuestionedCondition(
        signal_hint="position_setpoint_triplet.current.alt",
        op=">",
        reference="home_position.alt + RTL_RETURN_ALT",
        units="signal: meters; reference: m",
        frame="AMSL altitude",
    )
    assert condition.transition is None
    payload = condition.model_dump()
    assert payload["transition"] is None
    revived = QuestionedCondition.model_validate(payload)
    assert revived.transition is None
    assert revived == condition


def test_t1_transition_spec_round_trips_on_questioned_condition():
    """T1: additive optional intent parses, validates, and round-trips."""
    from flight_log_agent.analysis.mechanism_judge import QuestionedCondition

    condition = QuestionedCondition(
        signal_hint="tecs_status.height_rate_setpoint",
        op=">",
        reference="1.0",
        units="signal: m/s; reference: m/s",
        frame="test frame",
        transition=_tecs_like_intent(),
    )
    assert condition.transition is not None
    assert condition.transition.transition_signal == "vehicle_status.nav_state"
    revived = QuestionedCondition.model_validate(condition.model_dump())
    assert revived == condition


def test_transition_spec_rejects_unknown_relation_and_selection():
    """Intent outside the minimal schema fails validation, not silently."""
    import pytest

    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    with pytest.raises(Exception):
        TransitionEventSpec(
            transition_signal="vehicle_status.nav_state",
            relation="sometime",
        )
    with pytest.raises(Exception):
        TransitionEventSpec(
            transition_signal="vehicle_status.nav_state",
            relation="after",
            duration_s=1.0,
            event_selection="middle",
        )


def test_transition_spec_requires_exactly_one_extent_form():
    """duration_s and first_sample_of are alternatives, never combined or absent."""
    import pytest

    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    with pytest.raises(Exception):
        TransitionEventSpec(
            transition_signal="vehicle_status.nav_state",
            relation="after",
        )
    with pytest.raises(Exception):
        TransitionEventSpec(
            transition_signal="vehicle_status.nav_state",
            relation="after",
            duration_s=1.0,
            first_sample_of="tecs_status.height_rate_setpoint",
        )
    ok = TransitionEventSpec(
        transition_signal="vehicle_status.nav_state",
        relation="around",
        duration_s=2.0,
    )
    assert ok.duration_s == 2.0
    assert ok.first_sample_of is None


def _nav_samples():
    """Synthetic mode-state series with one 15→4 transition."""
    return {
        "vehicle_status.nav_state": [
            (496.0, 15),
            (496.7811, 15),
            (496.8814, 4),
            (497.0, 4),
        ]
    }


def test_t2_transition_event_derivation_matches_logged_change():
    """T2: generic intent + logged samples → deterministic matched event."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(event_selection=None), _nav_samples()
    )
    assert events == [
        {"time": 496.8814, "from_value": 15, "to_value": 4},
    ]


def test_t3_zero_matching_events_yields_no_event():
    """T3: no matching transition → empty derivation (fail closed downstream)."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(event_selection=None),
        {"vehicle_status.nav_state": [(496.0, 4), (497.0, 4)]},
    )
    assert events == []


def test_t4_ambiguous_events_require_explicit_selection():
    """T4: multiple matches without selection → unresolved; explicit
    first/last chooses deterministically."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
        select_transition_event,
    )

    samples = {
        "vehicle_status.nav_state": [
            (1.0, 15),
            (2.0, 4),
            (3.0, 15),
            (4.0, 4),
        ]
    }
    events = derive_transition_events(
        _tecs_like_intent(event_selection=None), samples
    )
    assert [event["time"] for event in events] == [2.0, 4.0]
    assert select_transition_event(events, None) is None
    assert select_transition_event(events, "first")["time"] == 2.0
    assert select_transition_event(events, "last")["time"] == 4.0
    single = derive_transition_events(
        _tecs_like_intent(event_selection=None), _nav_samples()
    )
    assert select_transition_event(single, None)["time"] == 496.8814


def test_n1_repeated_nan_emits_no_transition():
    """N1 (F1): NaN → NaN must not invent an event. Python's
    ``nan != nan`` would otherwise fabricate a transition out of
    repeated missing/non-finite observations."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    nan = float("nan")
    events = derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=None),
        {"vehicle_status.nav_state": [(496.0, nan), (497.0, nan)]},
    )
    assert events == []


def test_n2_finite_to_nan_matches_nothing():
    """N2 (F1): 15 → NaN satisfies no concrete ``to_value``."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=4),
        {"vehicle_status.nav_state": [(496.0, 15), (497.0, float("nan"))]},
    )
    assert events == []


def test_n3_nan_to_finite_matches_no_spec_value():
    """N3 (F1): NaN → 4 satisfies neither a concrete ``from_value``
    nor a concrete spec value in general."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(from_value=15, to_value=4),
        {"vehicle_status.nav_state": [(496.0, float("nan")), (497.0, 4)]},
    )
    assert events == []


def test_n4_finite_transition_still_detected():
    """N4 (F1): ordinary finite transitions are unaffected."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=None),
        {"vehicle_status.nav_state": [(496.0, 15), (497.0, 4)]},
    )
    assert events == [
        {"time": 497.0, "from_value": 15.0, "to_value": 4.0}
    ]


def test_n5_int_float_coercion_unchanged():
    """N5 (F1): spec ``15`` continues to match logged ``15.0``."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    events = derive_transition_events(
        _tecs_like_intent(from_value=15, to_value=4),
        {"vehicle_status.nav_state": [(496.0, 15.0), (497.0, 4.0)]},
    )
    assert [(event["time"], event["from_value"], event["to_value"])
            for event in events] == [(497.0, 15.0, 4.0)]


def test_nan_fix_preserves_bool_and_infinity_semantics():
    """F1 boundary pins: booleans keep strict identity (no int
    broadening), and identical infinities remain stable without
    special handling since ``inf == inf`` already holds."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_transition_events,
    )

    assert derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=None),
        {"vehicle_status.nav_state": [(496.0, True), (497.0, True)]},
    ) == []
    assert derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=None),
        {"vehicle_status.nav_state": [(496.0, True), (497.0, False)]},
    ) == [{"time": 497.0, "from_value": True, "to_value": False}]
    assert derive_transition_events(
        _tecs_like_intent(from_value=None, to_value=None),
        {"vehicle_status.nav_state": [
            (496.0, float("inf")), (497.0, float("inf"))]},
    ) == []


def _tecs_event():
    return {"time": 496.8814, "from_value": 15, "to_value": 4}


def test_t5_transition_relative_diagnostic_windows():
    """T5: matched event + relation/duration → concrete scope windows."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_diagnostic_windows,
    )

    assert derive_diagnostic_windows(
        _tecs_like_intent(
            relation="after", duration_s=2.0, first_sample_of=None
        ),
        _tecs_event(),
        {},
    ) == [(496.8814, 498.8814)]
    assert derive_diagnostic_windows(
        _tecs_like_intent(
            relation="around", duration_s=1.0, first_sample_of=None
        ),
        _tecs_event(),
        {},
    ) == [(495.8814, 497.8814)]
    assert derive_diagnostic_windows(
        _tecs_like_intent(
            relation="before", duration_s=1.0, first_sample_of=None
        ),
        _tecs_event(),
        {},
    ) == [(495.8814, 496.8814)]
    assert (
        derive_diagnostic_windows(
            _tecs_like_intent(
                relation="after", duration_s=1.0, first_sample_of=None
            ),
            None,
            {},
        )
        == []
    )


def test_t6_first_sample_after_event_uses_exact_timestamps():
    """T6: first prepared target sample at/after event time selects
    the window; degenerate point windows stay valid for selection."""
    from flight_log_agent.analysis.temporal_selection import (
        derive_diagnostic_windows,
    )

    samples = {
        "tecs_status.height_rate_setpoint": [
            (496.8993, 1.012127),
            (496.9187, 0.978286),
        ]
    }
    assert derive_diagnostic_windows(
        _tecs_like_intent(), _tecs_event(), samples
    ) == [(496.8814, 496.8993)]
    exact = {
        "tecs_status.height_rate_setpoint": [(496.8814, 1.0)],
    }
    assert derive_diagnostic_windows(
        _tecs_like_intent(), _tecs_event(), exact
    ) == [(496.8814, 496.8814)]
    assert (
        derive_diagnostic_windows(_tecs_like_intent(), _tecs_event(), {}) == []
    )
    assert (
        derive_diagnostic_windows(
            _tecs_like_intent(),
            _tecs_event(),
            {"tecs_status.height_rate_setpoint": [(496.0, 1.0)]},
        )
        == []
    )


def test_scope_construction_preserves_multiple_windows():
    """W3C safety: EvaluationScope carries plural windows unchanged."""
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    scope = EvaluationScope.from_result(
        {"windows": [(1.0, 2.0), (5.0, 6.0)]}
    )
    assert scope.windows == ((1.0, 2.0), (5.0, 6.0))
    assert scope.error == ""


def test_intersect_diagnostic_windows_is_deterministic():
    """Stage composition: predicate ∩ transition windows, ordered."""
    from flight_log_agent.analysis.temporal_selection import (
        intersect_diagnostic_windows,
    )

    assert intersect_diagnostic_windows(
        [(496.0, 497.0)], [(496.8814, 496.8993)]
    ) == [(496.8814, 496.8993)]
    assert intersect_diagnostic_windows(
        [(1.0, 2.0)], [(5.0, 6.0)]
    ) == []


def test_t7_temporal_eligibility_is_closed_window_overlap():
    """T7: domain overlap → eligible; disjoint domain → excluded."""
    from flight_log_agent.analysis.temporal_selection import (
        temporally_eligible,
        writer_domain,
    )

    window = [(496.8814, 496.8993)]
    assert temporally_eligible(
        writer_domain(
            replay_domain=[(496.8814, 496.8993)],
            branch_gates=[],
            scope_windows=[(496.0, 497.0)],
        ),
        window,
    ) is True
    assert temporally_eligible(
        writer_domain(
            replay_domain=[(497.5, 498.0)],
            branch_gates=[],
            scope_windows=[(496.0, 499.0)],
        ),
        window,
    ) is False
    # Closed intervals: endpoint touch counts as overlap.
    assert temporally_eligible([(496.8993, 497.0)], window) is True
    assert temporally_eligible([(496.0, 496.8814)], window) is True


def test_candidate_domain_prefers_replay_then_branch_windows():
    """Domain priority: replay result first, then gating branches."""
    from flight_log_agent.analysis.temporal_selection import writer_domain

    scope = [(496.0, 499.0)]
    assert writer_domain(
        replay_domain=[(496.8814, 496.8993)],
        branch_gates=[("always_true", [])],
        scope_windows=scope,
    ) == [(496.8814, 496.8993)]
    assert writer_domain(
        replay_domain=None,
        branch_gates=[("unknown", [(496.5, 497.5)])],
        scope_windows=scope,
    ) == [(496.5, 497.5)]
    assert writer_domain(
        replay_domain=None,
        branch_gates=[("always_true", [])],
        scope_windows=scope,
    ) == [(496.0, 499.0)]
    assert writer_domain(
        replay_domain=None,
        branch_gates=[("always_false", [(496.0, 499.0)])],
        scope_windows=scope,
    ) == []
    # Unknown feasibility without windows constrains nothing.
    assert writer_domain(
        replay_domain=None,
        branch_gates=[("unknown", [])],
        scope_windows=scope,
    ) == [(496.0, 499.0)]


def test_t8_multiple_eligible_writers_stay_unresolved():
    """T8: two overlapping writers, no discriminator → both retained,
    no unique selection, unresolved surfaced."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        [
            {"key": "writer_a", "domain": [(496.8814, 496.8993)],
             "replay_match": None},
            {"key": "writer_b", "domain": [(496.8814, 496.8993)],
             "replay_match": None},
        ],
        [(496.8814, 496.8993)],
    )
    assert outcome["eligible"] == ["writer_a", "writer_b"]
    assert outcome["unique"] is None
    assert outcome["unresolved"]


def test_t9_single_overlapping_writer_is_uniquely_selected():
    """T9: exactly one overlapping writer → unique selection."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        [
            {"key": "writer_a", "domain": [(496.8814, 496.8993)],
             "replay_match": None},
            {"key": "writer_b", "domain": [(497.5, 498.0)],
             "replay_match": None},
        ],
        [(496.8814, 496.8993)],
    )
    assert outcome["eligible"] == ["writer_a"]
    assert outcome["unique"] == "writer_a"
    assert outcome["unresolved"] is None


def test_t10_replay_support_may_discriminate_when_available():
    """T10: existing in-window replay support for one rival may select
    it; replay semantics themselves are untouched."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    outcome = discriminate_candidates(
        [
            {"key": "writer_a", "domain": [(496.8814, 496.8993)],
             "replay_match": True},
            {"key": "writer_b", "domain": [(496.8814, 496.8993)],
             "replay_match": False},
        ],
        [(496.8814, 496.8993)],
    )
    assert outcome["eligible"] == ["writer_a", "writer_b"]
    assert outcome["unique"] == "writer_a"
    tied = discriminate_candidates(
        [
            {"key": "writer_a", "domain": [(496.8814, 496.8993)],
             "replay_match": True},
            {"key": "writer_b", "domain": [(496.8814, 496.8993)],
             "replay_match": True},
        ],
        [(496.8814, 496.8993)],
    )
    assert tied["unique"] is None
    assert tied["unresolved"]


def test_t11_selection_is_order_invariant():
    """T11: candidate input permutation changes nothing."""
    from flight_log_agent.analysis.temporal_selection import (
        discriminate_candidates,
    )

    first = [
        {"key": "writer_a", "domain": [(496.8814, 496.8993)],
         "replay_match": None},
        {"key": "writer_b", "domain": [(497.5, 498.0)],
         "replay_match": None},
    ]
    second = list(reversed(first))
    window = [(496.8814, 496.8993)]
    assert discriminate_candidates(first, window) == discriminate_candidates(
        second, window
    )


def _temporal_dag(*, steady_windows):
    """Two branch-gated terminal writers plus one helper return.

    writer_init is feasible only in the TECS-like diagnostic window;
    writer_steady only in ``steady_windows``; helper h1 feeds
    writer_init through a call edge.
    """
    from flight_log_agent.analysis.mechanism_dag import DAGEdge, DAGVertex

    def branch(vid, verdict, windows):
        return DAGVertex(
            id=vid, kind="branch",
            predicate_raw="gate", predicate_lowered="gate",
            feasibility_verdict=verdict, active_windows=list(windows),
            metadata={},
        )

    def writer(vid, variable, expression, file, line, gate):
        vertex = DAGVertex(
            id=vid, kind="operation", variable=variable,
            expression=expression, file=file, line=line,
            metadata={"is_terminal": True},
        )
        edge = DAGEdge(
            id=f"{gate}> {vid}", source_id=gate, target_id=vid,
            kind="control",
        )
        return vertex, edge

    leaf = DAGVertex(
        id="leaf", kind="evidence", sub_kind="logged_signal",
        signal_name="x.y", metadata={"observation": "observed"})
    gate_init = branch("gate_init", "unknown", [(496.8814, 496.8993)])
    gate_steady = branch("gate_steady", "unknown", list(steady_windows))
    init, edge_init = writer(
        "writer_init", "_out", "transient_value", "src/a.cpp", 10,
        "gate_init")
    steady, edge_steady = writer(
        "writer_steady", "_out", "steady_value", "src/a.cpp", 20,
        "gate_steady")
    helper = DAGVertex(
        id="h1", kind="operation", variable="__return__",
        expression="return_value", file="src/h.cpp", line=20,
        metadata={
            "synthetic_helper_return_binding": True,
            "call_site_id": "src/c.cpp:10:3:legacy_call",
            "call_instance_scope": "scale::@call:aaa",
            "target_identity": {
                "declaration_id": "src/h.cpp:20:scale:value:return",
                "declaration_proven": True,
            },
        })
    call_edge = DAGEdge(
        id="h1>writer_init", source_id="h1", target_id="writer_init",
        kind="data", role="call:scale")
    vertices = [leaf, gate_init, gate_steady, init, steady, helper]
    edges = [edge_init, edge_steady, call_edge]
    return vertices, edges


def _temporal_report(vertices, edges, scope):
    from flight_log_agent.analysis.dag_pipeline import build_report_from_dag
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds,
        DiscoveryVerdict,
        JudgedDiscovery,
    )

    dag = MechanismDAG(
        dag_id="dag-temporal", terminal="_out",
        vertices=list(vertices), edges=list(edges),
        assumed_pruned_provenance=[])
    judged = JudgedDiscovery(
        seeds=DiscoverySeeds(seeds=[], candidate_terminals=[]),
        verdict=DiscoveryVerdict(
            sufficient=True, selected_terminal="_out",
            explaining_branches=[], reasoning="temporal"),
        results={}, selected=None)
    return build_report_from_dag(
        "why?", judged, dag, replay=None, scope=scope)


def _window_scope(windows):
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    return EvaluationScope(windows=tuple(windows))


def _ref_locations(report):
    return sorted(
        (ref.file, ref.start_line)
        for ref in report.ranked_hypotheses[0].source_refs
    )


def test_t9_report_selects_only_overlapping_writer():
    """T9 report level: unique overlapping writer selected; the
    disjoint writer's ref is absent."""
    vertices, edges = _temporal_dag(steady_windows=[(499.0, 501.0)])
    report = _temporal_report(
        vertices, edges, _window_scope([(496.8814, 496.8993)]))
    locations = _ref_locations(report)
    assert ("src/a.cpp", 10) in locations
    assert ("src/a.cpp", 20) not in locations


def test_t8_report_retains_tied_writers_with_unresolved_note():
    """T8 report level: tied writers both retained plus a
    deterministic unresolved note; no unique claim."""
    vertices, edges = _temporal_dag(
        steady_windows=[(496.8814, 496.8993)])
    report = _temporal_report(
        vertices, edges, _window_scope([(496.8814, 496.8993)]))
    locations = _ref_locations(report)
    assert ("src/a.cpp", 10) in locations
    assert ("src/a.cpp", 20) in locations
    unresolved = report.ranked_hypotheses[0].unresolved_evidence
    assert any(
        "temporally eligible writers remain indistinguishable" in item
        for item in unresolved
    )


def test_t12_temporal_filtering_preserves_source_identities():
    """T12: an all-covering scope leaves every source ref identical,
    including the helper representative identity."""
    vertices, edges = _temporal_dag(
        steady_windows=[(496.8814, 496.8993)])
    plain = _temporal_report(vertices, edges, None)
    scoped = _temporal_report(
        vertices, edges, _window_scope([(496.0, 502.0)]))
    plain_refs = [
        (ref.file, ref.start_line, ref.explanation)
        for ref in plain.ranked_hypotheses[0].source_refs
    ]
    scoped_refs = [
        (ref.file, ref.start_line, ref.explanation)
        for ref in scoped.ranked_hypotheses[0].source_refs
    ]
    assert scoped_refs == plain_refs
    assert any(
        explanation.startswith("upstream helper contribution:")
        for _, _, explanation in scoped_refs
    )


def test_t13_temporal_selection_leaves_status_and_proof_untouched():
    """T13: confidence, confirmation, and branch verification are
    identical with temporal selection active or absent."""
    vertices, edges = _temporal_dag(steady_windows=[(499.0, 501.0)])
    plain = _temporal_report(vertices, edges, None)
    scoped = _temporal_report(
        vertices, edges, _window_scope([(496.8814, 496.8993)]))
    for report in (plain, scoped):
        hypothesis = report.ranked_hypotheses[0]
        assert report.confirmed == []
        assert hypothesis.confidence == plain.ranked_hypotheses[0].confidence


def _stub_index_for(monkeypatch, table):
    from flight_log_agent.analysis import dag_pipeline

    class _Sample:
        def __init__(self, t, v):
            self.time_s = t
            self.value = v

    class _Series:
        def __init__(self, samples):
            self.samples = [_Sample(t, v) for t, v in samples]

    class _Resolution:
        def __init__(self, series):
            self.status = "observed" if series else "missing"
            self.series = series

    class _StubIndex:
        @classmethod
        def from_path(cls, _path, _references):
            return cls()

        def resolve_signal(self, name):
            series = table.get(name)
            return _Resolution(_Series(series) if series else None)

    monkeypatch.setattr(dag_pipeline, "ULogEvidenceIndex", _StubIndex)


def test_transition_windows_derive_from_logged_samples(tmp_path, monkeypatch):
    """Transition spec + logged samples → concrete diagnostic windows."""
    from flight_log_agent.analysis.dag_pipeline import (
        evaluate_transition_windows,
    )

    _stub_index_for(monkeypatch, {
        "vehicle_status.nav_state": [
            (496.0, 15), (496.8814, 4), (497.0, 4),
        ],
        "tecs_status.height_rate_setpoint": [
            (496.8993, 1.012127), (496.9187, 0.978286),
        ],
    })
    result = evaluate_transition_windows(
        _tecs_like_intent(),
        logged_set={
            "vehicle_status.nav_state",
            "tecs_status.height_rate_setpoint",
        },
        log_path=tmp_path / "flight.ulg",
        signal_policies={},
    )
    assert "error" not in result
    assert result["windows"] == [(496.8814, 496.8993)]


def test_transition_windows_fail_closed_without_event(tmp_path, monkeypatch):
    """Zero matching events → no windows with an explicit error."""
    from flight_log_agent.analysis.dag_pipeline import (
        evaluate_transition_windows,
    )

    _stub_index_for(monkeypatch, {
        "vehicle_status.nav_state": [(496.0, 4), (497.0, 4)],
        "tecs_status.height_rate_setpoint": [(496.8993, 1.012127)],
    })
    result = evaluate_transition_windows(
        _tecs_like_intent(),
        logged_set={
            "vehicle_status.nav_state",
            "tecs_status.height_rate_setpoint",
        },
        log_path=tmp_path / "flight.ulg",
        signal_policies={},
    )
    assert result["windows"] is None
    assert result["error"]


def test_replay_terminal_expressions_accepts_scope(tmp_path, monkeypatch):
    """Scope threading: terminal replay restricts to supplied windows."""
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.dag_pipeline import replay_terminal_expressions
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    _stub_index_for(monkeypatch, {
        "topic_in.value": [(0.0, 5.0), (10.0, 5.0)],
        "topic_out.value": [(0.0, 5.0), (10.0, 5.0)],
    })
    dag = build_mechanism_dag(
        [
            {
                "target_symbol": "input_value",
                "source_symbol": "topic_in.value",
                "assignment_path": [{"file": "a.cpp", "line": 1,
                                     "expression": "topic_in.value"}],
                "expression_ref": {
                    "text": "topic_in.value",
                    "lowered_text": "topic_in.value",
                    "input_symbols": ["topic_in.value"],
                    "input_identities": {},
                    "call_results": [],
                    "direct_storage": "",
                    "exact": True,
                },
                "external_source_signal": True,
                "synthetic_boundary_transfer": True,
                "boundary_direction": "subscribe",
                "control_predicates": [],
                "function": "A::run",
            },
            {
                "target_symbol": "topic_out.value",
                "source_symbol": "input_value",
                "assignment_path": [{"file": "a.cpp", "line": 2,
                                     "expression": "input_value"}],
                "expression_ref": {
                    "text": "input_value",
                    "lowered_text": "input_value",
                    "input_symbols": ["input_value"],
                    "input_identities": {},
                    "call_results": [],
                    "direct_storage": "",
                    "exact": True,
                },
                "external_target_signal": True,
                "synthetic_boundary_transfer": True,
                "boundary_direction": "publish",
                "control_predicates": [],
                "function": "A::run",
            },
        ],
        "topic_out.value",
        logged_signals={"topic_in.value", "topic_out.value"},
    )
    policies = {
        "topic_in.value": {"method": "linear"},
        "topic_out.value": {"method": "linear"},
    }
    scoped = replay_terminal_expressions(
        dag,
        tmp_path / "stubbed.ulg",
        {},
        {"topic_in.value", "topic_out.value"},
        signal_policies=policies,
        scope=EvaluationScope(((2.0, 4.0),)),
    )
    assert scoped["evaluation_windows"] == [(2.0, 4.0)]
    unscoped = replay_terminal_expressions(
        dag,
        tmp_path / "stubbed.ulg",
        {},
        {"topic_in.value", "topic_out.value"},
        signal_policies=policies,
    )
    assert unscoped["evaluation_windows"] == [(0.0, 10.0)]
