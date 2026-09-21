from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from flight_log_agent.analysis.dag_pipeline import (
    build_report_from_dag,
    evaluate_questioned_condition_windows,
    layer4_cache_path,
    read_seeds_from_cache,
    run_dag_discovery_stage,
    write_seeds_to_cache,
)
from flight_log_agent.analysis.mechanism_judge import (
    DiscoverySeeds,
    DiscoveryVerdict,
    QuestionedCondition,
    TerminalCandidate,
    judge_agent,
    seeder_agent,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler

TREE = {
    "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    orb_copy(ORB_ID(gspeed), _gspeed_sub, &gspeed);
    _dest_val = gspeed;
    if (_param_rtl_type.get() == 1) {
        _final_out = _dest_val + 1.0f;
    }
}
""",
}


def _exact_expression(text: str, *inputs: str) -> dict:
    return {
        "text": text,
        "lowered_text": text,
        "input_symbols": list(inputs),
        "input_identities": {},
        "call_results": [],
        "direct_storage": inputs[0] if len(inputs) == 1 and text == inputs[0] else "",
        "exact": True,
    }


def _logged_input(local: str, signal: str, line: int) -> dict:
    return {
        "target_symbol": local,
        "source_symbol": signal,
        "assignment_path": [
            {"file": "a.cpp", "line": line, "expression": signal}
        ],
        "expression_ref": _exact_expression(signal, signal),
        "external_source_signal": True,
        "synthetic_boundary_transfer": True,
        "boundary_direction": "subscribe",
        "control_predicates": [],
        "function": "A::run",
    }


@pytest.fixture(params=["terminal", "checkpoint"])
def replay_engine(request):
    """Exercise one numerical contract through both public replay boundaries."""
    from flight_log_agent.analysis.dag_pipeline import replay_terminal_expressions
    from flight_log_agent.analysis.dag_replay import observed_checkpoint_roots, replay_dag_roots

    def replay(dag, path, parameters, logged, **kwargs):
        if request.param == "terminal":
            return replay_terminal_expressions(dag, path, parameters, logged, **kwargs)
        groups = observed_checkpoint_roots(dag)
        assert len(groups) == 1
        observed, roots = next(iter(groups.items()))
        assert observed in logged
        return replay_dag_roots(
            dag, roots, observed, parameter_values=parameters, **kwargs
        )

    return replay


@pytest.fixture(autouse=True)
def _stub_stage_signal_samples(monkeypatch):
    from flight_log_agent.analysis import dag_pipeline

    class _Sample:
        def __init__(self, timestamp, value):
            self.time_s = timestamp
            self.value = value

    class _Series:
        samples = [_Sample(0.0, 5.0), _Sample(10.0, 5.0)]

    class _Resolution:
        status = "observed"
        series = _Series()

    class _StubIndex:
        @classmethod
        def from_path(cls, _path, _references):
            return cls()

        def resolve_signal(self, _name):
            return _Resolution()

    monkeypatch.setattr(dag_pipeline, "ULogEvidenceIndex", _StubIndex)


def _mini_tree(tmp_path) -> MechanismSourceProfiler:
    root = tmp_path / "PX4-Autopilot"
    for rel, text in TREE.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return MechanismSourceProfiler(root, rg_path="missing-rg")


def _stub_runner(calls: list[str]):
    async def run(agent, payload):
        calls.append(agent.name)
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        assert agent is judge_agent
        branch_id = payload["candidates"]["_final_out"]["branches"][0]["id"]
        return DiscoveryVerdict(
            sufficient=True,
            selected_terminal="_final_out",
            explaining_branches=[branch_id],
            reasoning="terminal grounded in constant and parameter gate",
        )

    return run


def test_layer4_roundtrip_and_miss(tmp_path):
    path = layer4_cache_path(tmp_path, "Why did RTL climb so high?")
    assert read_seeds_from_cache(path) is None

    seeds = DiscoverySeeds(
        seeds=["RTL_RETURN_ALT"],
        candidate_terminals=[TerminalCandidate(terminal="_rtl_alt")],
    )
    write_seeds_to_cache(seeds, path)
    loaded = read_seeds_from_cache(path)
    assert loaded is not None
    assert loaded.seeds == ["RTL_RETURN_ALT"]

    # same question maps to the same path; different question does not
    assert layer4_cache_path(tmp_path, "Why did RTL climb so high?") == path
    assert layer4_cache_path(tmp_path, "Why did the motor stop?") != path

    path.write_text("{not json", encoding="utf-8")
    assert read_seeds_from_cache(path) is None


def test_stage_bypasses_all_dag_cache_layers(tmp_path):
    profiler = _mini_tree(tmp_path)
    cache_root = tmp_path / "cache"
    calls: list[str] = []
    kwargs = dict(
        inventory={"parameters": {"RTL_TYPE": 1}},
        ulog_hash="ulog0",
        run_agent=_stub_runner(calls),
        logged_signals={"gspeed"},
    )

    stage = asyncio.run(
        run_dag_discovery_stage(
            profiler,
            cache_root,
            "why did final_out increase?",
            "srchash",
            Path("/nonexistent.ulg"),
            **kwargs,
        )
    )

    assert stage.layer4_hit is False
    assert not cache_root.exists()
    assert read_seeds_from_cache(
        layer4_cache_path(cache_root, "why did final_out increase?")
    ) is None
    assert stage.report.ranked_hypotheses[0].confidence == "low"
    assert stage.report.confirmed == []

    # The second run performs fresh seeding and discovery as well.
    again = asyncio.run(
        run_dag_discovery_stage(
            profiler,
            cache_root,
            "why did final_out increase?",
            "srchash",
            Path("/nonexistent.ulg"),
            **kwargs,
        )
    )
    assert again.layer4_hit is False
    assert calls == [
        seeder_agent.name,
        judge_agent.name,
        seeder_agent.name,
        judge_agent.name,
    ]
    assert not cache_root.exists()


def test_stage_report_maps_feasibility_into_applicability(tmp_path):
    profiler = _mini_tree(tmp_path)
    calls: list[str] = []

    stage = asyncio.run(
        run_dag_discovery_stage(
            profiler,
            tmp_path / "cache",
            "why did final_out increase?",
            "srchash",
            Path("/nonexistent.ulg"),
            inventory={"parameters": {"RTL_TYPE": 1}},
            ulog_hash="ulog0",
            run_agent=_stub_runner(calls),
            logged_signals={"gspeed"},
        )
    )

    hypothesis = stage.report.ranked_hypotheses[0]
    assert hypothesis.known_px4_mechanism == "_final_out"
    assert hypothesis.mechanism.startswith("terminal grounded")
    assert hypothesis.confidence == "low"
    assert stage.report.confirmed == []
    assert stage.report.unconfirmed == [hypothesis.title]
    conditions = (
        hypothesis.applicability.supported_conditions
        + hypothesis.applicability.unresolved_conditions
        + hypothesis.applicability.excluded_by
    )
    assert any("_param_rtl_type" in c for c in conditions)


def test_insufficient_verdict_yields_unresolved_report(tmp_path):
    profiler = _mini_tree(tmp_path)

    async def run(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        return DiscoveryVerdict(sufficient=False, selected_terminal="_final_out")

    stage = asyncio.run(
        run_dag_discovery_stage(
            profiler,
            tmp_path / "cache",
            "why did final_out increase?",
            "srchash",
            Path("/nonexistent.ulg"),
            ulog_hash="ulog0",
            run_agent=run,
            logged_signals={"gspeed"},
        )
    )

    hypothesis = stage.report.ranked_hypotheses[0]
    assert hypothesis.confidence == "unresolved"
    assert stage.report.unconfirmed == [hypothesis.title]
    assert stage.report.confirmed == []


def _run_with_verdict(tmp_path, verdict_kwargs, *, use_valid_branch=False):
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds as Seeds,
        DiscoveryVerdict as Verdict,
        TerminalCandidate as Cand,
        seeder_agent as seeder,
    )

    profiler = _mini_tree(tmp_path)

    async def run(agent, payload):
        if agent is seeder:
            return Seeds(seeds=["pick_altitude"],
                         candidate_terminals=[Cand(terminal="_final_out")])
        kwargs = dict(verdict_kwargs)
        if use_valid_branch:
            kwargs["explaining_branches"] = [
                payload["candidates"]["_final_out"]["branches"][0]["id"]
            ]
        return Verdict(sufficient=True, selected_terminal="_final_out",
                       **kwargs)

    return asyncio.run(
        run_dag_discovery_stage(
            profiler, tmp_path / "cache", "why?", "srchash",
            Path("/nonexistent.ulg"), ulog_hash="u", run_agent=run,
            logged_signals={"gspeed"},
        )
    )


def test_sufficient_with_unmatched_branch_is_downgraded(tmp_path):
    stage = _run_with_verdict(
        tmp_path, {"explaining_branches": ["some_bogus_predicate > 99"]}
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert h.applicability.applicable is False
    assert stage.report.confirmed == []
    assert any("feasibility-dead" in u or "absent" in u for u in h.unresolved_evidence)


def test_sufficient_without_named_branch_is_downgraded(tmp_path):
    stage = _run_with_verdict(tmp_path, {})
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert stage.report.confirmed == []
    assert any("without naming" in u for u in h.unresolved_evidence)


def test_sufficient_with_valid_branch_stays_unconfirmed_without_replay(tmp_path):
    stage = _run_with_verdict(tmp_path, {}, use_valid_branch=True)
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert stage.report.confirmed == []


def test_layer4_helpers_remain_available_but_stage_does_not_prune_or_use_them(tmp_path):
    from flight_log_agent.analysis.dag_pipeline import seeder_fingerprint

    path = layer4_cache_path(tmp_path, "why?")
    assert path.parent.name == seeder_fingerprint()
    assert path.parent.parent.name == "intent"

    stale = tmp_path / "cache" / "intent" / "0123456789abcdef"
    stale.mkdir(parents=True)
    profiler = _mini_tree(tmp_path)
    calls: list[str] = []
    asyncio.run(
        run_dag_discovery_stage(
            profiler, tmp_path / "cache", "why?", "srchash",
            Path("/nonexistent.ulg"), ulog_hash="u", run_agent=_stub_runner(calls),
        )
    )
    assert stale.exists()


def test_explaining_predicate_text_is_not_accepted_as_branch_identity(tmp_path):
    stage = _run_with_verdict(
        tmp_path,
        {"explaining_branches": ["_param_rtl_type.get() == 1"]},
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert h.applicability.applicable is False
    assert stage.report.confirmed == []


def test_replay_status_gates_the_confidence_upgrade(tmp_path):
    """Only a complete ``matched`` replay upgrades to high; ``partial``
    is unresolved evidence and never contradiction — the confirmation
    and confidence stay at the structural level."""
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds as Seeds, DiscoveryVerdict as Verdict,
        TerminalCandidate as Cand,
    )
    from flight_log_agent.analysis.dag_pipeline import build_report_from_dag
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path)
    result = discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        ["pick_altitude"],
        "_final_out",
        "hash",
        logged_signals={"gspeed"},
    )
    branch_id = next(v.id for v in result.dag.vertices if v.kind == "branch")
    from flight_log_agent.analysis.mechanism_judge import JudgedDiscovery
    judged = JudgedDiscovery(
        seeds=Seeds(seeds=[], candidate_terminals=[Cand(terminal="_final_out")]),
        verdict=Verdict(sufficient=True, selected_terminal="_final_out",
                        explaining_branches=[branch_id]),
        results={"_final_out": result}, selected=result,
    )
    matched = {"status": "matched", "complete": True, "observed": "x.y",
               "results": [{"grounded": "a+b", "evaluable": True, "match_fraction": 0.9}]}
    report = build_report_from_dag("why?", judged, result.dag, replay=matched)
    assert report.ranked_hypotheses[0].confidence == "high"
    assert any("expression replay [matched]" in e
               for e in report.ranked_hypotheses[0].evidence)

    partial = {"status": "partial", "complete": False, "observed": "x.y",
               "results": [{"grounded": "a+b", "evaluable": True, "match_fraction": 0.9}]}
    unresolved = build_report_from_dag("why?", judged, result.dag, replay=partial)
    assert unresolved.ranked_hypotheses[0].confidence == "medium"
    assert unresolved.ranked_hypotheses[0].contradicting_evidence == []

    for numeric in (partial, matched):
        incomplete_checkpoint = dict(numeric, authorizes_discovery_stop=False,
                                    analysis_requirements=[{"reason": "missing source writer"}])
        blocked = build_report_from_dag("why?", judged, result.dag, replay=incomplete_checkpoint)
        assert blocked.confirmed == []
        assert blocked.ranked_hypotheses[0].confidence == "unresolved"
        assert not blocked.ranked_hypotheses[0].applicability.applicable
        assert "missing source writer" in blocked.ranked_hypotheses[0].unresolved_evidence

    skipped = build_report_from_dag(
        "why?", judged, result.dag,
        replay={"status": "not_attempted", "complete": False, "reason": "r"})
    assert skipped.ranked_hypotheses[0].confidence == "low"
    assert not any("expression replay" in e
                   for e in skipped.ranked_hypotheses[0].evidence)

    mismatched = {
        "status": "mismatched",
        "complete": True,
        "observed": "x.y",
        "results": [
            {"grounded": "a+b", "evaluable": True, "match_fraction": 0.1}
        ],
    }
    contradicted = build_report_from_dag(
        "why?", judged, result.dag, replay=mismatched
    )
    item = contradicted.ranked_hypotheses[0]
    assert item.confidence == "unresolved"
    assert item.applicability.applicable is False
    assert item.contradicting_evidence
    assert contradicted.confirmed == []


def test_seeds_not_cached_when_selected_slice_is_empty(tmp_path):
    """A seeder sample whose discovery produced an empty slice must not
    be cached as the question's answer — the next run retries the
    seeder instead of replaying the bad sample."""
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds as Seeds, DiscoveryVerdict as Verdict,
        TerminalCandidate as Cand, seeder_agent as seeder,
    )

    profiler = _mini_tree(tmp_path)

    async def run(agent, payload):
        if agent is seeder:
            return Seeds(seeds=["nonexistent_marker"],
                         candidate_terminals=[Cand(terminal="_ghost_var")])
        return Verdict(sufficient=False, selected_terminal="_ghost_var")

    stage = asyncio.run(
        run_dag_discovery_stage(
            profiler, tmp_path / "cache", "why?", "srchash",
            Path("/nonexistent.ulg"), ulog_hash="u", run_agent=run,
        )
    )

    assert read_seeds_from_cache(
        layer4_cache_path(tmp_path / "cache", "why?")
    ) is None
    assert stage.report.confirmed == []


def test_questioned_signal_resolves_via_slice_not_string_fuzz(tmp_path):
    """A renamed field (hint tecs_status.airspeed_sp vs actual
    true_airspeed_sp) resolves through the slice's logged leaves for the
    hinted topic; ambiguity and misses return honest errors."""
    from flight_log_agent.analysis.dag_pipeline import resolve_questioned_signal
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    logged = {"tecs_status.true_airspeed_sp", "tecs_status.height_rate",
              "vehicle_status.nav_state"}
    dag = build_mechanism_dag(
        [
            {
                "target_symbol": "airspeed_sp",
                "source_symbol": "tecs_status.true_airspeed_sp",
                "assignment_path": [{"file": "a.cpp", "line": 1,
                                     "expression": "tecs_status.true_airspeed_sp"}],
                "control_predicates": [],
                "function": "A::run",
                "external_source_signal": True,
                "synthetic_boundary_transfer": True,
                "boundary_direction": "subscribe",
            },
            {
                "target_symbol": "_x",
                "source_symbol": "airspeed_sp",
                "assignment_path": [{"file": "a.cpp", "line": 2,
                                     "expression": "airspeed_sp"}],
                "control_predicates": [],
                "function": "A::run",
            },
        ],
        "_x", logged_signals=logged,
    )

    signal, error, cands = resolve_questioned_signal(
        "tecs_status.airspeed_sp", logged, [dag])
    assert signal == "tecs_status.true_airspeed_sp" and error is None

    exact, error, _ = resolve_questioned_signal(
        "vehicle_status.nav_state", logged, [dag])
    assert exact == "vehicle_status.nav_state"

    missing, error, cands = resolve_questioned_signal(
        "unknown_topic.field", logged, [dag])
    assert missing is None and "did not resolve" in error


def test_questioned_hint_never_resolves_by_containment():
    """A hint whose topic exists in the schema but not in any slice
    returns the topic's fields as CANDIDATES — even a unique substring
    match is name guessing, not provenance."""
    from flight_log_agent.analysis.dag_pipeline import resolve_questioned_signal

    logged = {"topic_a.true_value_sp", "topic_a.mode"}
    signal, error, cands = resolve_questioned_signal("topic_a.value_sp", logged, [])
    assert signal is None
    assert "did not resolve exactly" in error
    assert cands == ["topic_a.mode", "topic_a.true_value_sp"]


def test_questioned_hint_resolves_only_one_exact_topic_instance():
    from flight_log_agent.analysis.dag_pipeline import resolve_questioned_signal

    unique = {"topic_a[3].value"}
    signal, error, candidates = resolve_questioned_signal(
        "topic_a.value", unique, []
    )
    assert signal == "topic_a[3].value"
    assert error is None
    assert candidates == []

    ambiguous = {"topic_a[0].value", "topic_a[3].value"}
    signal, error, candidates = resolve_questioned_signal(
        "topic_a.value", ambiguous, []
    )
    assert signal is None
    assert "ambiguous" in error
    assert candidates == ["topic_a[0].value", "topic_a[3].value"]


def test_questioned_condition_evaluates_signal_plus_parameter_reference(
    monkeypatch, tmp_path
):
    from flight_log_agent.analysis import dag_pipeline

    values = {
        "position_setpoint_triplet[0].current.alt": [(0.0, 110.0), (10.0, 130.0)],
        "home_position[0].alt": [(0.0, 100.0), (10.0, 100.0)],
    }

    class Sample:
        def __init__(self, time_s, value):
            self.time_s = time_s
            self.value = value

    class Resolution:
        status = "observed"

        def __init__(self, samples):
            self.series = type(
                "Series", (), {"samples": [Sample(*sample) for sample in samples]}
            )()

    class Index:
        @classmethod
        def from_path(cls, _path, references):
            assert set(references) == set(values)
            return cls()

        def resolve_signal(self, name):
            return Resolution(values[name])

    monkeypatch.setattr(dag_pipeline, "ULogEvidenceIndex", Index)
    condition = QuestionedCondition(
        signal_hint="position_setpoint_triplet.current.alt",
        op=">",
        reference="home_position.alt + RTL_RETURN_ALT",
        units="signal: meters; reference: m",
        frame="AMSL altitude",
    )
    policies = {
        "position_setpoint_triplet.current.alt": {
            "method": "linear", "unit": "m", "confidence": "high"
        },
        "home_position.alt": {
            "method": "linear", "unit": "m", "confidence": "high"
        },
    }

    result = evaluate_questioned_condition_windows(
        condition,
        candidates={},
        logged_set=set(values),
        log_path=tmp_path / "flight.ulg",
        parameter_values={"RTL_RETURN_ALT": 20.0},
        signal_policies=policies,
    )

    assert "error" not in result
    assert result["signal"] == "position_setpoint_triplet[0].current.alt"
    assert result["reference"] == "home_position[0].alt + RTL_RETURN_ALT"
    assert result["units"] == "signal: meters; reference: m"
    assert result["resolved_units"] == {"signal": "m", "reference": "m"}
    assert result["windows"] == [(10.0, 10.0)]


def test_questioned_condition_rejects_unresolved_expression_operand(tmp_path):
    condition = QuestionedCondition(
        signal_hint="position_setpoint_triplet.current.alt",
        op=">",
        reference="home_position.alt + UNKNOWN_OFFSET",
        units="m",
        frame="AMSL altitude",
    )
    result = evaluate_questioned_condition_windows(
        condition,
        candidates={},
        logged_set={
            "position_setpoint_triplet.current.alt",
            "home_position.alt",
        },
        log_path=tmp_path / "flight.ulg",
        parameter_values={},
        signal_policies={
            "position_setpoint_triplet.current.alt": {"method": "linear", "unit": "m"},
            "home_position.alt": {"method": "linear", "unit": "m"},
        },
    )
    assert result["windows"] is None
    assert "UNKNOWN_OFFSET" in result["error"]


def test_replay_requires_source_proven_terminal_publication():
    from flight_log_agent.analysis.dag_pipeline import replay_terminal_expressions
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    dag = build_mechanism_dag(
        [{"target_symbol": "_x", "source_symbol": "topic_b.alt + 1.0f",
          "assignment_path": [{"file": "a.cpp", "line": 1,
                               "expression": "topic_b.alt + 1.0f"}],
          "logged_signal": "", "control_predicates": [], "function": "A::run"}],
        "_x", logged_signals={"topic_b.alt"},
    )
    replay = replay_terminal_expressions(
        dag, Path("/nonexistent.ulg"), {}, {"topic_b.alt"},
        observed_hint="wrong_topic.alt",
    )
    assert replay["status"] == "not_attempted"
    assert "publication provenance" in replay["reason"]


def test_complete_replay_distinguishes_match_mismatch_and_missing_policy(monkeypatch):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    class _Sample:
        def __init__(self, t, v):
            self.time_s = t
            self.value = v

    class _Series:
        def __init__(self, samples):
            self.samples = samples

    class _Resolution:
        def __init__(self, series):
            self.status = "observed" if series else "missing"
            self.series = series

    class _StubIndex:
        _table = {
            "topic_in.value": [_Sample(0.0, 5.0), _Sample(10.0, 5.0)],
            "topic_out.value": [_Sample(0.0, 5.0), _Sample(10.0, 5.0)],
        }

        @classmethod
        def from_path(cls, path, references):
            return cls()

        def resolve_signal(self, name):
            samples = self._table.get(name)
            return _Resolution(_Series(samples) if samples else None)

    monkeypatch.setattr(dag_pipeline, "ULogEvidenceIndex", _StubIndex)

    dag = build_mechanism_dag(
        [
            _logged_input("input_value", "topic_in.value", 1),
            {
                "target_symbol": "topic_out.value",
                "source_symbol": "input_value",
                "assignment_path": [{"file": "a.cpp", "line": 2,
                                     "expression": "input_value"}],
                "expression_ref": _exact_expression("input_value", "input_value"),
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
    matched = dag_pipeline.replay_terminal_expressions(
        dag,
        Path("/stubbed.ulg"),
        {},
        {"topic_in.value", "topic_out.value"},
        signal_policies=policies,
    )
    assert matched["status"] == "matched"
    assert matched["complete"] is True

    _StubIndex._table["topic_out.value"] = [
        _Sample(0.0, 8.0),
        _Sample(10.0, 8.0),
    ]
    mismatched = dag_pipeline.replay_terminal_expressions(
        dag,
        Path("/stubbed.ulg"),
        {},
        {"topic_in.value", "topic_out.value"},
        signal_policies=policies,
    )
    assert mismatched["status"] == "mismatched"
    assert mismatched["complete"] is True

    partial = dag_pipeline.replay_terminal_expressions(
        dag,
        Path("/stubbed.ulg"),
        {},
        {"topic_in.value", "topic_out.value"},
    )
    assert partial["status"] == "partial"
    assert partial["complete"] is False


def test_replay_reuses_supplied_dag_signal_data(monkeypatch, replay_engine):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import (
        build_mechanism_dag,
        prepare_signal_series,
    )

    dag = build_mechanism_dag(
        [
            _logged_input("input_value", "topic_in.value", 1),
            {
                "target_symbol": "topic_out.value",
                "source_symbol": "input_value",
                "assignment_path": [{
                    "file": "a.cpp",
                    "line": 2,
                    "expression": "input_value",
                }],
                "expression_ref": _exact_expression("input_value", "input_value"),
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
    samples = {
        "topic_in.value": [(10.0, 5.0), (0.0, 5.0)],
        "topic_out.value": [(10.0, 5.0), (0.0, 5.0)],
    }
    policies = {
        "topic_in.value": {"method": "linear"},
        "topic_out.value": {"method": "linear"},
    }
    prepared = prepare_signal_series(samples, policies)

    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("replay must reuse the selected DAG's signal data")

    monkeypatch.setattr(dag_pipeline, "_signal_samples_for_dag", unexpected_load)
    replay = replay_engine(
        dag,
        Path("/nonexistent.ulg"),
        {},
        {"topic_in.value", "topic_out.value"},
        signal_policies=policies,
        signal_samples=samples,
        prepared_signal_series=prepared,
    )
    assert replay["status"] == "matched"
    assert replay["complete"] is True


def test_replay_combines_mutually_exclusive_writers_piecewise(monkeypatch):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import (
        build_mechanism_dag,
        evaluate_feasibility,
    )

    class _Sample:
        def __init__(self, timestamp, value):
            self.time_s = timestamp
            self.value = value

    class _Series:
        def __init__(self, samples):
            self.samples = samples

    class _Resolution:
        def __init__(self, samples):
            self.status = "observed" if samples else "unavailable"
            self.series = _Series(samples) if samples else None

    class _StubIndex:
        table = {
            "mode.state": [
                _Sample(0.0, 0),
                _Sample(5.0, 1),
                _Sample(10.0, 1),
            ],
            "input.first": [_Sample(0.0, 1.0), _Sample(10.0, 1.0)],
            "input.second": [_Sample(0.0, 2.0), _Sample(10.0, 2.0)],
            "output.value": [
                _Sample(0.0, 1.0),
                _Sample(5.0, 2.0),
                _Sample(10.0, 2.0),
            ],
        }

        @classmethod
        def from_path(cls, _path, _references):
            return cls()

        def resolve_signal(self, name):
            return _Resolution(self.table.get(name))

    monkeypatch.setattr(dag_pipeline, "ULogEvidenceIndex", _StubIndex)
    logged = set(_StubIndex.table)
    policies = {
        "mode.state": {"method": "discrete_hold"},
        "input.first": {"method": "linear"},
        "input.second": {"method": "linear"},
        "output.value": {"method": "linear"},
    }
    bindings = [
        _logged_input("mode_state", "mode.state", 1),
        _logged_input("first_value", "input.first", 2),
        _logged_input("second_value", "input.second", 3),
        {
            "target_symbol": "output.value",
            "source_symbol": "first_value",
            "assignment_path": [
                {"file": "a.cpp", "line": 10, "expression": "first_value"}
            ],
            "expression_ref": _exact_expression("first_value", "first_value"),
            "external_target_signal": True,
            "synthetic_boundary_transfer": True,
            "boundary_direction": "publish",
            "control_predicates": ["mode_state == 0"],
            "control_expression_refs": [
                _exact_expression("mode_state == 0", "mode_state")
            ],
            "control_predicate_lines": [9],
            "function": "A::run",
        },
        {
            "target_symbol": "output.value",
            "source_symbol": "second_value",
            "assignment_path": [
                {"file": "a.cpp", "line": 12, "expression": "second_value"}
            ],
            "expression_ref": _exact_expression("second_value", "second_value"),
            "external_target_signal": True,
            "synthetic_boundary_transfer": True,
            "boundary_direction": "publish",
            "control_predicates": ["mode_state == 1"],
            "control_expression_refs": [
                _exact_expression("mode_state == 1", "mode_state")
            ],
            "control_predicate_lines": [11],
            "function": "A::run",
        },
    ]
    dag = build_mechanism_dag(
        bindings, "output.value", logged_signals=logged
    )
    samples = {
        name: [(sample.time_s, sample.value) for sample in values]
        for name, values in _StubIndex.table.items()
    }
    annotated = evaluate_feasibility(
        dag,
        signal_samples=samples,
        signal_policies=policies,
        prune_dead=False,
    )

    replay = dag_pipeline.replay_terminal_expressions(
        annotated,
        Path("/stubbed.ulg"),
        {},
        logged,
        signal_policies=policies,
    )

    assert replay["status"] == "matched"
    assert replay["complete"] is True
    assert len(replay["results"]) == 2
    assert all(result["match_fraction"] == 1.0 for result in replay["results"])


def test_replay_reconstructs_nested_internal_flow_from_dag_edges(replay_engine):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    bindings = [
        _logged_input("input_value", "input.value", 1),
        {
            "target_symbol": "internal",
            "source_symbol": "input_value * 2.0",
            "assignment_path": [
                {"file": "a.cpp", "line": 10, "expression": "input_value * 2.0"}
            ],
            "expression_ref": _exact_expression(
                "input_value * 2.0", "input_value"
            ),
            "control_predicates": [],
            "function": "A::run",
        },
        {
            "target_symbol": "output.value",
            "source_symbol": "max(internal, 3.0)",
            "assignment_path": [
                {"file": "a.cpp", "line": 20, "expression": "max(internal, 3.0)"}
            ],
            "expression_ref": _exact_expression(
                "max(internal, 3.0)", "internal"
            ),
            "external_target_signal": True,
            "synthetic_boundary_transfer": True,
            "boundary_direction": "publish",
            "control_predicates": [],
            "function": "A::run",
        },
    ]
    samples = {
        "input.value": [(0.0, 1.0), (10.0, 2.0)],
        "output.value": [(0.0, 3.0), (10.0, 4.0)],
    }
    policies = {
        "input.value": {"method": "linear"},
        "output.value": {"method": "linear"},
    }
    dag = build_mechanism_dag(
        bindings,
        "output.value",
        logged_signals=set(samples),
    )

    replay = replay_engine(
        dag,
        Path("/unused.ulg"),
        {},
        set(samples),
        signal_policies=policies,
        signal_samples=samples,
    )

    assert replay["status"] == "matched"
    assert replay["complete"] is True
    assert replay["results"][0]["evaluation_mode"] == "dag_value_plan"
    assert replay["results"][0]["grounded"] is None


def test_replay_selects_mutually_exclusive_internal_writers_from_dag_edges(replay_engine):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import (
        build_mechanism_dag,
        evaluate_feasibility,
    )

    def binding(target, expression, line, *, controls=None, logged_signal=""):
        binding = {
            "target_symbol": logged_signal or target,
            "source_symbol": expression,
            "assignment_path": [
                {"file": "a.cpp", "line": line, "expression": expression}
            ],
            "control_predicates": controls or [],
            "control_predicate_lines": [line - 1 for _ in controls or []],
            "control_expression_refs": [
                _exact_expression(predicate, "mode_state")
                for predicate in controls or []
            ],
            "expression_ref": _exact_expression(expression, expression),
            "function": "A::run",
        }
        if logged_signal:
            binding.update(
                {
                    "external_target_signal": True,
                    "synthetic_boundary_transfer": True,
                    "boundary_direction": "publish",
                }
            )
        return binding

    bindings = [
        _logged_input("mode_state", "mode.state", 1),
        _logged_input("first_value", "input.first", 2),
        _logged_input("second_value", "input.second", 3),
        binding("selected", "first_value", 10, controls=["mode_state == 0"]),
        binding("selected", "second_value", 12, controls=["mode_state == 1"]),
        binding("output", "selected", 20, logged_signal="output.value"),
    ]
    samples = {
        "mode.state": [(0.0, 0), (5.0, 1), (10.0, 1)],
        "input.first": [(0.0, 1.0), (10.0, 1.0)],
        "input.second": [(0.0, 2.0), (10.0, 2.0)],
        "output.value": [(0.0, 1.0), (5.0, 2.0), (10.0, 2.0)],
    }
    policies = {
        "mode.state": {"method": "discrete_hold"},
        "input.first": {"method": "linear"},
        "input.second": {"method": "linear"},
        "output.value": {"method": "linear"},
    }
    dag = build_mechanism_dag(
        bindings,
        "output.value",
        logged_signals=set(samples),
    )
    annotated = evaluate_feasibility(
        dag,
        signal_samples=samples,
        signal_policies=policies,
        prune_dead=False,
    )

    replay = replay_engine(
        annotated,
        Path("/unused.ulg"),
        {},
        set(samples),
        signal_policies=policies,
        signal_samples=samples,
    )

    assert replay["status"] == "matched"
    assert replay["complete"] is True
    assert replay["results"][0]["match_fraction"] == 1.0


def _checkpoint_fixture(expression="input_value", input_signal="input.value"):
    from flight_log_agent.analysis.mechanism_dag import build_mechanism_dag

    dag = build_mechanism_dag(
        [
            _logged_input("input_value", input_signal, 1),
            {
                "target_symbol": "output.value",
                "source_symbol": expression,
                "assignment_path": [{"file": "a.cpp", "line": 2, "expression": expression}],
                "expression_ref": _exact_expression(expression, "input_value"),
                "external_target_signal": True,
                "synthetic_boundary_transfer": True,
                "boundary_direction": "publish",
                "control_predicates": [],
                "function": "A::run",
            },
        ],
        "output.value", logged_signals={input_signal, "output.value"},
    )
    samples = {
        input_signal: [(0.0, 5.0), (2.0, 5.0), (4.0, 5.0), (6.0, 5.0), (10.0, 5.0)],
        "output.value": [(0.0, 8.0), (2.0, 5.0), (4.0, 5.0), (6.0, 8.0), (10.0, 8.0)],
    }
    policies = {signal: {"method": "linear"} for signal in samples}
    return dag, samples, policies


def _checkpoint_replay(dag, samples, policies, scope=None, **kwargs):
    from flight_log_agent.analysis.dag_replay import observed_checkpoint_roots, replay_dag_roots

    return replay_dag_roots(
        dag, observed_checkpoint_roots(dag)["output.value"], "output.value",
        signal_samples=samples, signal_policies=policies, scope=scope, **kwargs,
    )


@pytest.mark.parametrize("windows", [((2.0, 4.0),), ((2.0, 2.5), (3.5, 4.0))])
def test_checkpoint_scope_uses_requested_domain_not_whole_log(windows):
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    dag, samples, policies = _checkpoint_fixture()
    assert _checkpoint_replay(dag, samples, policies)["status"] == "mismatched"
    replay = _checkpoint_replay(dag, samples, policies, EvaluationScope(windows))
    assert replay["status"] == "matched"
    assert replay["evaluation_windows"] == list(windows)
    assert replay["results"][0]["max_abs_error"] == 0


@pytest.mark.parametrize("result", [
    {"windows": None, "error": "unknown frame"},
    {"windows": []},
    {"windows": [[4, 2]]},
    {"windows": [[float("nan"), 5]]},
    {"windows": [[20, 30]]},
])
def test_invalid_empty_or_unobserved_scope_never_counts_as_match(result):
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    dag, samples, policies = _checkpoint_fixture()
    replay = _checkpoint_replay(dag, samples, policies, EvaluationScope.from_result(result))
    assert replay["status"] == "not_attempted"
    assert replay["complete"] is False


def test_scope_outside_observation_coverage_remains_partial():
    from flight_log_agent.analysis.dag_replay import EvaluationScope

    dag, samples, policies = _checkpoint_fixture()
    replay = _checkpoint_replay(dag, samples, policies, EvaluationScope(((-1, 4),)))
    assert replay["status"] == "partial"
    assert replay["complete"] is False
    assert replay["results"][0]["mismatched_samples"] > 0


def test_checkpoint_cannot_hide_another_known_writer():
    from flight_log_agent.analysis.dag_replay import EvaluationScope, observed_checkpoint_roots, replay_dag_roots

    dag, samples, policies = _checkpoint_fixture()
    roots = observed_checkpoint_roots(dag)["output.value"]
    writer = next(v for v in dag.vertices if v.id == roots[0])
    dag.vertices.append(writer.model_copy(update={"id": "another_writer"}))
    result = replay_dag_roots(
        dag, roots, "output.value", signal_samples=samples, signal_policies=policies,
        scope=EvaluationScope(((2.0, 4.0),)),
    )
    assert result["status"] == "partial"
    assert result["missing_writer_ids"] == ["another_writer"]


def test_observed_output_cannot_verify_itself(replay_engine):
    dag, samples, policies = _checkpoint_fixture(input_signal="output.value")
    replay = replay_engine(
        dag, Path("/unused.ulg"), {}, set(samples),
        signal_samples=samples, signal_policies=policies,
    )
    assert replay["status"] == "not_attempted"
    assert "also an input" in replay["reason"]
    assert replay["blocking_vertex_ids"]


def test_type_only_observation_is_not_checkpoint_proof(replay_engine):
    dag, samples, policies = _checkpoint_fixture()
    leaf = next(v for v in dag.vertices if v.kind == "evidence" and v.signal_name == "input.value")
    leaf.metadata = {"grounded_via": "declared_type"}
    replay = replay_engine(
        dag, Path("/unused.ulg"), {}, set(samples),
        signal_samples=samples, signal_policies=policies,
    )
    assert replay["status"] == "not_attempted"
    assert replay["blocking_vertex_ids"] == [leaf.id]


@pytest.mark.parametrize("expression,input_value,observed", [
    pytest.param("max(10.0, 2.0 * input_value)", 10.0, 20.0, id="rtl-floor"),
    pytest.param("19.0 * sqrt(1.0 / cos(input_value))", 0.8726646259971648,
                 23.698446, id="airspeed-load-factor"),
])
def test_successful_run_numeric_checkpoints(replay_engine, expression, input_value, observed):
    """Archived numeric checkpoints, not proof of the full cone or slew behavior."""
    dag, samples, policies = _checkpoint_fixture(expression)
    samples = {
        "input.value": [(0.0, input_value), (10.0, input_value)],
        "output.value": [(0.0, observed), (10.0, observed)],
    }
    result = replay_engine(
        dag, Path("/unused.ulg"), {}, set(samples),
        signal_samples=samples, signal_policies=policies,
    )
    assert result["status"] == "matched"
    assert result["results"][0]["max_abs_error"] < 0.00001


def test_checkpoint_program_and_prepared_samples_are_reused(monkeypatch):
    from flight_log_agent.analysis import dag_replay
    from flight_log_agent.analysis.dag_value import DAGValueProgram
    from flight_log_agent.analysis.mechanism_dag import prepare_signal_series, sample_prepared_signal

    dag, samples, policies = _checkpoint_fixture()
    prepared = prepare_signal_series(samples, policies)
    program = DAGValueProgram(dag)
    session = program.bind(sample_resolver=lambda s, t: sample_prepared_signal(prepared, s, t))

    def unexpected(*_args, **_kwargs):
        raise AssertionError("checkpoint must reuse its round's compiled program and samples")

    monkeypatch.setattr(dag_replay, "DAGValueProgram", unexpected)
    monkeypatch.setattr(dag_replay, "prepare_signal_series", unexpected)
    replay = _checkpoint_replay(
        dag, samples, policies, prepared_signal_series=prepared,
        value_program=program, value_session=session,
    )
    assert replay["status"] == "mismatched"


def _run_checkpoint_trial(tmp_path, monkeypatch, backend, *, observe_input=True, checkpoint_control=False):
    from flight_log_agent.analysis import dag_pipeline

    root = tmp_path / "source"
    root.mkdir()
    path = root / "sample.cpp"
    path.write_text("""
void A::run()
{
    topic_in_s input;
    orb_copy(ORB_ID(topic_in), _sub, &input);
    topic_out_s output;
    output.value = input.value * 2.0f;
    orb_publish(ORB_ID(topic_out), _pub, &output);
}
""", encoding="utf-8")
    profiler = MechanismSourceProfiler(root, rg_path="missing-rg", source_parser_backend=backend)
    samples = {
        "topic_in.value": [(0.0, 5.0), (10.0, 5.0)],
        "topic_out.value": [(0.0, 10.0), (10.0, 10.0)],
    }
    policies = {signal: {"method": "linear", "unit": "m"} for signal in samples}
    schema_signals = set(samples)
    if not observe_input:
        samples.pop("topic_in.value")
    scopes_evaluated = []
    original_windows = dag_pipeline.evaluate_questioned_condition_windows

    def windows(*args, **kwargs):
        scopes_evaluated.append(kwargs.get("candidates"))
        return original_windows(*args, **kwargs)

    monkeypatch.setattr(dag_pipeline, "evaluate_questioned_condition_windows", windows)
    monkeypatch.setattr(dag_pipeline, "_signal_samples_for_dag", lambda *_args, **_kwargs: samples)
    payloads = []
    events = []

    async def runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["A::run"],
                candidate_terminals=[TerminalCandidate(terminal="output.value", terminal_file="sample.cpp")],
                questioned_condition=QuestionedCondition(
                    signal_hint="topic_out.value", op=">", reference="1",
                    units="m", frame="test frame",
                ),
            )
        payloads.append(payload)
        return DiscoveryVerdict(sufficient=False, selected_terminal=next(iter(payload["candidates"])))

    def observe(summary):
        assert scopes_evaluated and scopes_evaluated[0] is None
        assert not payloads, "checkpoint must run before the judge"
        events.append(summary)

    kwargs = dict(
        inventory={"parameters": {}}, logged_signals=set(samples), schema_signals=schema_signals,
        signal_policies=policies, run_agent=runner,
    )
    trial = asyncio.run(run_dag_discovery_stage(
        profiler, tmp_path / "cache", "why?", "source", Path("/stubbed.ulg"),
        checkpoint_diagnostics=True, checkpoint_discovery=checkpoint_control, checkpoint_observer=observe, **kwargs,
    ))
    assert events == trial.checkpoint_rounds
    assert events
    assert events[0]["scope"]["windows"] == ((0.0, 10.0),), events[0]["scope"]
    checkpoint = next(event["checkpoints"]["topic_out.value"] for event in reversed(events)
                      if "topic_out.value" in event.get("checkpoints", {})
                      and not event.get("pending_construction_count"))
    assert events[0]["resources"]["checkpoint_wall_s"] >= 0
    control = asyncio.run(run_dag_discovery_stage(
        profiler, tmp_path / "cache", "why?", "source", Path("/stubbed.ulg"), **kwargs,
    ))
    assert control.checkpoint_rounds == []
    if not checkpoint_control:
        assert trial.report.model_dump() == control.report.model_dump()
    else:
        assert not trial.report.confirmed
        assert not control.report.confirmed
    assert trial.judged.selected.files_loaded == control.judged.selected.files_loaded
    assert trial.judged.selected.dag.model_dump() == control.judged.selected.dag.model_dump()
    if checkpoint_control:
        for candidate in payloads[0]["candidates"].values():
            assert "discovery_checkpoint" in candidate
            candidate.pop("discovery_checkpoint")
        # The internal-value candidate now retains its source-proven downstream
        # publication as an observation witness. Its causal render is unchanged;
        # assert the exact diagnostic extension before comparing everything else.
        candidate = payloads[0]["candidates"]["output.value"]
        assert candidate["other_writes_not_reaching_terminal"] == [{
            "target": "topic_out.value", "expression": "output.value",
            "file": "sample.cpp", "line": 8, "reaches_terminal": False,
        }]
        candidate["other_writes_not_reaching_terminal"] = []
        candidate["vertices"] -= 1
        candidate["edges"] -= 1
        for entry in candidate["rounds"]:
            entry["vertices"] -= 1
            entry["edges"] -= 1
    assert payloads[0] == payloads[1]
    return checkpoint


@pytest.mark.parametrize("backend", ["legacy", "tree_sitter"])
def test_checkpoint_trial_does_not_change_discovery_or_judge(tmp_path, monkeypatch, backend):
    _run_checkpoint_trial(tmp_path, monkeypatch, backend)


@pytest.mark.parametrize("backend", ["legacy", "tree_sitter"])
def test_checkpoint_source_replay_requires_receiver_evidence(tmp_path, monkeypatch, backend):
    checkpoint = _run_checkpoint_trial(tmp_path, monkeypatch, backend)
    assert checkpoint["status"] == "not_attempted", str(checkpoint)
    assert checkpoint["analysis_requirements"]
    assert checkpoint["authorizes_discovery_stop"] is False


@pytest.mark.parametrize("backend", ["legacy", "tree_sitter"])
def test_source_checkpoint_requests_missing_observation_through_shared_pipeline(tmp_path, monkeypatch, backend):
    checkpoint = _run_checkpoint_trial(tmp_path, monkeypatch, backend, observe_input=False)
    assert checkpoint["complete"] is False
    assert checkpoint["status"] == "not_attempted"
    assert any(requirement["kind"] == "observation_data"
               and requirement.get("signal") == "topic_in.value"
               for requirement in checkpoint["analysis_requirements"])


@pytest.mark.parametrize("backend", ["legacy", "tree_sitter"])
def test_checkpoint_control_uses_shared_source_pipeline(tmp_path, monkeypatch, backend):
    _run_checkpoint_trial(tmp_path, monkeypatch, backend, checkpoint_control=True)


@pytest.mark.parametrize("backend", ["legacy", "tree_sitter"])
def test_checkpoint_control_cannot_verify_from_topic_presence(tmp_path, monkeypatch, backend):
    checkpoint = _run_checkpoint_trial(tmp_path, monkeypatch, backend, checkpoint_control=True)
    assert checkpoint["status"] == "not_attempted"
    assert checkpoint["authorizes_discovery_stop"] is False


def test_validation_downgrade_resynchronizes_confirmation_lists():
    from flight_log_agent.analysis.report_validation import (
        enforce_validation_downgrades,
    )
    from flight_log_agent.models import (
        ApplicabilityReport,
        CodeRef,
        ExpectedSignatureItem,
        FlightLogReport,
        HypothesisReportItem,
        RelationshipCheckSpec,
        ValidationIssue,
        ValidationResult,
    )

    title = "Mechanism slice"
    hypothesis = HypothesisReportItem(
        title=title,
        known_px4_mechanism="terminal",
        mechanism="source and numeric evidence",
        source_refs=[CodeRef(file="a.cpp")],
        expected_logged_signature=[
            ExpectedSignatureItem(name="output.value", description="output")
        ],
        applicability=ApplicabilityReport(applicable=True),
        evidence=[],
        contradicting_evidence=[],
        exclusion_checks=[],
        numeric_checks=[RelationshipCheckSpec(type="derived_expression")],
        confidence="medium",
    )
    report = FlightLogReport(
        airframe_summary="",
        question_intent_summary="question",
        ranked_hypotheses=[hypothesis],
        excluded_mechanisms=[],
        confirmed=[title],
        unconfirmed=[],
        final_summary="",
    )
    validation = ValidationResult(
        passed=False,
        issues=[
            ValidationIssue(
                severity="error",
                path="ranked_hypotheses[0].numeric_checks",
                message="numeric evidence invalid",
            )
        ],
    )

    downgraded = enforce_validation_downgrades(report, validation)

    assert downgraded.ranked_hypotheses[0].confidence == "low"
    assert downgraded.confirmed == []
    assert downgraded.unconfirmed == [title]


def test_proof_snapshot_args_contract():
    """P3-H: snapshot maps onto exact T6B input names; None preserves
    legacy omission (no empty-proof semantic change)."""
    from flight_log_agent.analysis.dag_pipeline import _proof_snapshot_args
    from flight_log_agent.analysis.coverage import ProofSnapshot
    assert _proof_snapshot_args(None) == {}
    snapshot = ProofSnapshot(
        version=3, certificates=("c1",), applicability_proofs=("p1", "p2"))
    assert _proof_snapshot_args(snapshot) == {
        "proof_version": 3,
        "coverage_certificates": ("c1",),
        "applicability_proofs": ("p1", "p2"),
    }
    empty = ProofSnapshot(version=0)
    assert _proof_snapshot_args(empty) == {
        "proof_version": 0,
        "coverage_certificates": (),
        "applicability_proofs": (),
    }


def test_construction_adapter_forwards_proof_snapshot():
    """P3 construction fix: the production construction adapter
    accepts and forwards the current proof snapshot to control_round
    (previously dropped, silently keeping construction proof-blind)."""
    from flight_log_agent.analysis.dag_pipeline import (
        _construction_evaluator_for,
    )
    from flight_log_agent.analysis.coverage import ProofSnapshot
    seen = []

    def control_round(dag, index, during_construction=False,
                      proof_snapshot=None):
        seen.append((dag, index, during_construction, proof_snapshot))
        return "round"

    adapter = _construction_evaluator_for(control_round)
    snapshot = ProofSnapshot(version=2, certificates=("c",))
    assert adapter("dag", 7, proof_snapshot=snapshot) == "round"
    assert seen == [("dag", 7, True, snapshot)]
    assert adapter("dag", 8) == "round"
    assert seen[-1] == ("dag", 8, True, None)


def test_diagnostic_observer_checkpoints_stay_replay_free(tmp_path, monkeypatch):
    """R1 freeze (M/N): pipeline diagnostic observer summaries keep
    replay disabled even where production checkpoint rounds may now
    attempt it."""
    import flight_log_agent.analysis.dag_pipeline as pipeline
    from unittest import mock
    samples = {'topic_in.value': [(0.0, 5.0), (10.0, 5.0)],
               'topic_out.value': [(0.0, 10.0), (10.0, 10.0)]}
    with mock.patch.object(pipeline, '_signal_samples_for_dag',
                           lambda *_a, **_k: dict(samples)):
        profiler = _mini_tree(tmp_path)
        calls: list[str] = []
        events: list[dict] = []
        asyncio.run(pipeline.run_dag_discovery_stage(
            profiler, tmp_path / 'cache', 'why?', 'srchash',
            Path('/nonexistent.ulg'), ulog_hash='u',
            run_agent=_stub_runner(calls), checkpoint_diagnostics=True,
            checkpoint_observer=events.append))
    assert events, "diagnostic observer never ran"
    seen = 0
    for event in events:
        for name, check in list((event.get('checkpoints') or {}).items()):
            seen += 1
            assert check.get('status') == 'not_attempted', name
        terminal = event.get('terminal_checkpoint')
        if terminal is not None:
            seen += 1
            assert terminal.get('status') == 'not_attempted'
    assert seen, "no diagnostic checkpoint entries observed"


def test_matched_unverified_report_stays_unresolved(tmp_path):
    """R1-O: a numerically matched replay without verified authority
    keeps downstream confidence unresolved/non-authoritative."""
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds as Seeds, DiscoveryVerdict as Verdict,
        TerminalCandidate as Cand,
    )
    from flight_log_agent.analysis.dag_pipeline import build_report_from_dag
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag
    profiler = _mini_tree(tmp_path)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", ["pick_altitude"], "_final_out",
        "hash", logged_signals={"gspeed"})
    from flight_log_agent.analysis.mechanism_judge import JudgedDiscovery
    judged = JudgedDiscovery(
        seeds=Seeds(seeds=[], candidate_terminals=[Cand(terminal="_final_out")]),
        verdict=Verdict(sufficient=True, selected_terminal="_final_out",
                        explaining_branches=[]),
        results={"_final_out": result}, selected=result)
    matched = {"status": "matched", "complete": True, "observed": "x.y",
               "authorizes_discovery_stop": False,
               "results": [{"grounded": "a+b", "evaluable": True,
                            "match_fraction": 0.9}]}
    report = build_report_from_dag("why?", judged, result.dag, replay=matched)
    assert report.ranked_hypotheses[0].confidence != "high"
    assert report.confirmed == []


def _report_vertex(*, vid, variable, expression, file, line):
    from flight_log_agent.analysis.mechanism_dag import DAGVertex
    return DAGVertex(
        id=vid, kind="operation", variable=variable,
        expression=expression, file=file, line=line,
        metadata={"is_terminal": True})


def _report_observed_leaf(*, vid, signal):
    from flight_log_agent.analysis.mechanism_dag import DAGVertex
    return DAGVertex(
        id=vid, kind="evidence", sub_kind="logged_signal",
        signal_name=signal, metadata={"observation": "observed"})


def _report_candidate_record(*, file, line, variable="stored",
                             expression="input * 3.0"):
    return {
        "file": file, "line": line, "end_line": None,
        "variable": variable, "expression": expression,
        "callable": None, "source_site_id": None,
        "terminal": True,
        "control_predicates": ["(hrt_absolute_time() - _t) > 1"],
        "verdict": "always_false", "windows": [],
        "assumed": True,
        "assumption_category": "assumed_feasibility",
        "assumption_reason": ["temporary: uORB freshness gate assumed fresh"],
    }


def _report_for(vertices, retained=(), edges=()):
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    from flight_log_agent.analysis.mechanism_judge import JudgedDiscovery
    dag = MechanismDAG(
        dag_id="dag-acceptance", terminal="_final_out",
        vertices=list(vertices), edges=list(edges),
        assumed_pruned_provenance=list(retained))
    judged = JudgedDiscovery(
        seeds=DiscoverySeeds(seeds=[], candidate_terminals=[]),
        verdict=DiscoveryVerdict(
            sufficient=True, selected_terminal="_final_out",
            explaining_branches=[], reasoning="acceptance"),
        results={}, selected=None)
    return build_report_from_dag("why?", judged, dag, replay=None)


def _writer_source(*, file="src/a.cpp", line=5, end_line=None,
                   variable="x", expression="input * 3.0",
                   extra=None):
    source = {
        "file": file, "line": line, "end_line": end_line,
        "variable": variable, "expression": expression,
    }
    if extra:
        source.update(extra)
    return source


def test_writer_identity_collapses_repeat_instances():
    """N1a: same source writer instantiated with differing
    instance-only metadata shares one stable identity."""
    from flight_log_agent.analysis.dag_pipeline import (
        _source_writer_identity,
    )
    first = _writer_source(extra={
        "call_instance_scope": "fn::@call:aaa",
        "source_order": 10,
    })
    second = _writer_source(extra={
        "call_instance_scope": "fn::@call:bbb",
        "source_order": 10,
    })
    assert (_source_writer_identity(first)
            == _source_writer_identity(second))


def test_writer_identity_separates_distinct_assignments():
    """N1b: same symbol with a different line or expression is a
    different writer identity."""
    from flight_log_agent.analysis.dag_pipeline import (
        _source_writer_identity,
    )
    base = _writer_source()
    assert (_source_writer_identity(base)
            != _source_writer_identity(_writer_source(line=9)))
    assert (_source_writer_identity(base)
            != _source_writer_identity(
                _writer_source(expression="input * 4.0")))
    assert (_source_writer_identity(base)
            != _source_writer_identity(
                _writer_source(variable="y")))


def test_writer_identity_tolerates_missing_provenance():
    """N1c: absent optional provenance still yields a deterministic
    identity instead of crashing."""
    from flight_log_agent.analysis.dag_pipeline import (
        _source_writer_identity,
    )
    sparse = {"variable": "x"}
    assert (_source_writer_identity(sparse)
            == _source_writer_identity({"variable": "x"}))
    assert isinstance(_source_writer_identity({}), tuple)


def test_duplicate_surviving_terminals_collapse_to_one_ref():
    """N2: many terminal graph instances of one publication source
    site collapse to a single semantic CodeRef."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    first = _report_vertex(
        vid="a", variable="out", expression="input * 2.0",
        file="src/pub.cpp", line=44)
    second = _report_vertex(
        vid="b", variable="out", expression="input * 2.0",
        file="src/pub.cpp", line=44)
    report = _report_for([leaf, first, second])
    refs = report.ranked_hypotheses[0].source_refs
    assert [(ref.file, ref.start_line) for ref in refs] == [
        ("src/pub.cpp", 44)]


def test_distinct_writers_stay_distinct():
    """N2: same terminal symbol with distinct source assignment
    identity keeps separate CodeRefs."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    first = _report_vertex(
        vid="a", variable="out", expression="input * 2.0",
        file="src/pub.cpp", line=44)
    second = _report_vertex(
        vid="b", variable="out", expression="input * 3.0",
        file="src/pub.cpp", line=44)
    report = _report_for([leaf, first, second])
    refs = report.ranked_hypotheses[0].source_refs
    assert len(refs) == 2


def _ref_signature(report):
    return [(ref.file, ref.start_line, ref.explanation)
            for ref in report.ranked_hypotheses[0].source_refs]


def test_normalized_order_ignores_input_order():
    """N6: semantically identical eligible refs in different input
    orders normalize to the same CodeRef order."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    calc = _report_vertex(
        vid="calc", variable="out", expression="input * 3.0",
        file="src/calc.cpp", line=10)
    anchor = _report_vertex(
        vid="decl", variable="out", expression="0.0f",
        file="include/decl.h", line=3)
    retained = [_report_candidate_record(
        file="src/old.cpp", line=7, variable="out")]
    first = _report_for([leaf, anchor, calc], retained=retained)
    second = _report_for([leaf, calc, anchor],
                         retained=list(reversed(retained)))
    assert _ref_signature(first) == _ref_signature(second)


def test_auto_heal_prefers_surviving_over_candidate():
    """N8: a retained ASSUMED candidate whose equivalent source
    writer survives normally yields exactly one ordinary ref —
    no duplicate slot, no candidate wording for that writer."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    surviving = _report_vertex(
        vid="live", variable="out", expression="input * 3.0",
        file="src/a.cpp", line=5)
    retained = [_report_candidate_record(
        file="src/a.cpp", line=5, variable="out",
        expression="input * 3.0")]
    report = _report_for([leaf, surviving], retained=retained)
    refs = [ref for ref in report.ranked_hypotheses[0].source_refs
            if (ref.file, ref.start_line) == ("src/a.cpp", 5)]
    assert len(refs) == 1
    assert refs[0].explanation.startswith("terminal write:")


def test_report_orders_runtime_before_declaration_anchor():
    """D3/T3: a runtime terminal writer precedes a supporting
    declaration anchor even when the anchor vertex comes first;
    a lone declaration may still lead."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    anchor = _report_vertex(
        vid="decl", variable="_final_out", expression="0.0f",
        file="include/decl.h", line=3)
    computation = _report_vertex(
        vid="calc", variable="_final_out", expression="input * 3.0",
        file="src/calc.cpp", line=10)
    report = _report_for([leaf, anchor, computation])
    refs = report.ranked_hypotheses[0].source_refs
    assert [ref.file for ref in refs] == ["src/calc.cpp", "include/decl.h"]
    assert refs[0].explanation.startswith("terminal write:")

    lone = _report_for([leaf, anchor])
    assert [ref.file for ref in lone.ranked_hypotheses[0].source_refs] == [
        "include/decl.h"]


def test_report_dedups_retained_writer_instances():
    """D3/T4: duplicate retained records of one source writer
    collapse; a genuinely different assignment stays distinct."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    retained = [
        _report_candidate_record(
            file="src/a.cpp", line=5, variable="_final_out"),
        _report_candidate_record(
            file="src/a.cpp", line=5, variable="_final_out"),
        _report_candidate_record(
            file="src/a.cpp", line=9, variable="_final_out",
            expression="input * 4.0"),
    ]
    report = _report_for([leaf], retained=retained)
    refs = report.ranked_hypotheses[0].source_refs
    assert sorted((ref.file, ref.start_line) for ref in refs) == [
        ("src/a.cpp", 5), ("src/a.cpp", 9)]
    assert all(ref.explanation.startswith(
        "candidate terminal write excluded by assumed feasibility "
        "condition:") for ref in refs)


def _helper_return_vertex(*, vid, variable="__return__",
                           file="src/h.cpp", line=20,
                           callable_id="src/h.cpp:20:scale:value",
                           call_site_id="src/c.cpp:10:3:legacy_call"):
    from flight_log_agent.analysis.mechanism_dag import DAGVertex
    return DAGVertex(
        id=vid, kind="operation", variable=variable,
        expression="return_value", file=file, line=line,
        metadata={
            "synthetic_helper_return_binding": True,
            "call_site_id": call_site_id,
            "call_instance_scope": "scale::@call:aaa",
            "target_identity": {
                "declaration_id": f"{callable_id}:return",
                "declaration_proven": True,
            },
        })


def test_helper_identity_single_return():
    """H1: represented single-return helper yields
    (callable, call-site, empty-discriminator) identity."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )
    vertex = _helper_return_vertex(vid="h1")
    assert (_helper_representative_identity(vertex)
            == ("src/h.cpp:20:scale:value",
                "src/c.cpp:10:3:legacy_call", ""))


def test_helper_identity_missing_without_callable():
    """H1: no stable callable identity means no helper
    representative — never a synthesized identifier."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )
    from flight_log_agent.analysis.mechanism_dag import DAGVertex
    vertex = DAGVertex(
        id="h9", kind="operation", variable="__return__",
        expression="return_value", file="src/h.cpp", line=20,
        metadata={"synthetic_helper_return_binding": True})
    assert _helper_representative_identity(vertex) is None


def test_helper_identity_ignores_call_instance_scope():
    """H2 (identity part): different call instances of one site
    share one helper identity."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )
    first = _helper_return_vertex(vid="h1")
    second = _helper_return_vertex(vid="h2")
    second.metadata = dict(second.metadata,
                           call_instance_scope="scale::@call:bbb")
    assert (_helper_representative_identity(first)
            == _helper_representative_identity(second))


def test_helper_identity_separates_result_paths():
    """H4: one callable+site with two represented result paths
    yields two helper identities."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )
    plain = _helper_return_vertex(vid="h1", variable="__return__")
    pathed = _helper_return_vertex(
        vid="h2", variable="__return__.altitude")
    assert (_helper_representative_identity(plain)
            != _helper_representative_identity(pathed))
    assert _helper_representative_identity(pathed)[2] == "altitude"


def _wired_helper_vertex(*, vid, variable="__return__", **kwargs):
    return _helper_return_vertex(vid=vid, variable=variable, **kwargs)


def _data_edge(*, eid, source, target, role="call:calc", via=None):
    from flight_log_agent.analysis.mechanism_dag import DAGEdge
    return DAGEdge(id=eid, source_id=source, target_id=target,
                   kind="data", role=role, via=via)


_STABLE_CALL_SITE = "src/c.cpp:120:140:call_expression"


def _consumer_with_call_site(*, vid="term", variable="_final_out",
                             call_site=_STABLE_CALL_SITE,
                             result_path="", call_text="scale(x)"):
    """Terminal consumer whose expression provably contains one call.

    The structured ``call_results`` entry carries the stable SOURCE
    call-site identity, exactly as production
    ``_source_expression_metadata`` records it."""
    term = _terminal_consumer(vid=vid, variable=variable)
    term.metadata = dict(term.metadata, source_expression_ref={
        "text": term.expression,
        "lowered_text": term.expression,
        "input_symbols": [],
        "input_identities": {},
        "call_results": [{
            "call_source_site_id": call_site,
            "result_path": result_path,
            "text": call_text,
        }],
        "exact": True,
    })
    return term


def _terminal_consumer(*, vid="term", variable="_final_out"):
    return _report_vertex(
        vid=vid, variable=variable, expression="helper_value * 2.0",
        file="src/main.cpp", line=30)


def test_helper_qualifies_through_call_data_edge():
    """H5: helper-return supplying terminal value through a
    call/data edge yields one helper ref with helper wording."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(vid="h1")
    term = _terminal_consumer()
    report = _report_for(
        [leaf, helper, term],
        edges=[_data_edge(eid="e1", source="h1", target="term")])
    refs = report.ranked_hypotheses[0].source_refs
    helpers = [ref for ref in refs
               if ref.explanation.startswith("upstream helper contribution:")]
    assert len(helpers) == 1
    assert (helpers[0].file, helpers[0].start_line) == ("src/h.cpp", 20)


def test_helper_control_only_does_not_qualify():
    """H5/H7: helper reachable only through a control edge
    contributes no value and yields no helper ref."""
    from flight_log_agent.analysis.mechanism_dag import DAGEdge
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(vid="h1")
    term = _terminal_consumer()
    report = _report_for(
        [leaf, helper, term],
        edges=[DAGEdge(id="e1", source_id="h1", target_id="term",
                       kind="control")])
    refs = report.ranked_hypotheses[0].source_refs
    assert not any(ref.explanation.startswith("upstream helper contribution:")
                   for ref in refs)


def test_helper_instances_collapse_to_one_ref():
    """H2/S1: same callable + same SOURCE site + same path across
    graph instances yields one helper ref — even when the raw
    per-instance sites are distinct invocation_<hash> values."""
    from flight_log_agent.analysis.dag_pipeline import (
        _qualified_helper_entries,
    )
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    first = _wired_helper_vertex(
        vid="h1", call_site_id="invocation_000aaa111bbb")
    first.metadata = dict(first.metadata,
                          call_instance_scope="scale::@call:aaa")
    second = _wired_helper_vertex(
        vid="h2", call_site_id="invocation_000bbb222ccc")
    second.metadata = dict(second.metadata,
                           call_instance_scope="scale::@call:bbb")
    term = _consumer_with_call_site()
    vertices = [leaf, first, second, term]
    edges = [_data_edge(eid="e1", source="h1", target="term",
                        via="invocation_000aaa111bbb"),
             _data_edge(eid="e2", source="h2", target="term",
                        via="invocation_000bbb222ccc")]
    # Selection-layer collapse: one stable identity, not two
    # per-instance identities. Anchor-location overlap must not be
    # what saves this test.
    dag = MechanismDAG(
        dag_id="dag-acceptance", terminal="_final_out",
        vertices=vertices, edges=edges)
    qualified = _qualified_helper_entries(dag)
    assert len(qualified) == 1
    assert qualified[0][1] == (
        "src/h.cpp:20:scale:value", _STABLE_CALL_SITE, "")
    report = _report_for(vertices, edges=edges)
    refs = report.ranked_hypotheses[0].source_refs
    helpers = [ref for ref in refs
               if ref.explanation.startswith("upstream helper contribution:")]
    assert len(helpers) == 1


def test_helper_identity_resolves_stable_source_site():
    """S1 (identity part): distinct invocation_<hash> raw sites from
    one source call resolve to one stable SOURCE-site identity."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    first = _wired_helper_vertex(
        vid="h1", call_site_id="invocation_000aaa111bbb")
    second = _wired_helper_vertex(
        vid="h2", call_site_id="invocation_000bbb222ccc")
    term = _consumer_with_call_site()
    dag = MechanismDAG(
        dag_id="dag-acceptance", terminal="_final_out",
        vertices=[leaf, first, second, term],
        edges=[_data_edge(eid="e1", source="h1", target="term",
                          via="invocation_000aaa111bbb"),
               _data_edge(eid="e2", source="h2", target="term",
                          via="invocation_000bbb222ccc")])
    assert (_helper_representative_identity(first, dag)
            == ("src/h.cpp:20:scale:value", _STABLE_CALL_SITE, ""))
    assert (_helper_representative_identity(first, dag)
            == _helper_representative_identity(second, dag))


def test_helper_identity_none_without_stable_source_site():
    """S3: only per-instance invocation hashes available (no stable
    consumer site, raw site, via, or roles key) → no helper
    identity. Never fabricate one from the invocation hash."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
        _qualified_helper_entries,
    )
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(
        vid="h1", call_site_id="invocation_000aaa111bbb")
    term = _terminal_consumer()
    vertices = [leaf, helper, term]
    edges = [_data_edge(eid="e1", source="h1", target="term",
                        via="invocation_000aaa111bbb")]
    dag = MechanismDAG(
        dag_id="dag-acceptance", terminal="_final_out",
        vertices=vertices, edges=edges)
    assert _helper_representative_identity(helper, dag) is None
    assert _qualified_helper_entries(dag) == []
    report = _report_for(vertices, edges=edges)
    refs = report.ranked_hypotheses[0].source_refs
    assert not any(ref.explanation.startswith("upstream helper contribution:")
                   for ref in refs)


def test_helper_pruned_consumer_yields_no_helper_ref():
    """H12: helper vertex exists but its qualifying data consumer is
    absent from the surviving DAG — the selected terminal has no
    data path from the helper → no helper representative."""
    from flight_log_agent.analysis.mechanism_dag import DAGVertex
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(vid="h1")
    dead = DAGVertex(
        id="dead", kind="operation", variable="stale_val",
        expression="helper_value + 1.0", file="src/main.cpp", line=25,
        metadata={})
    term = _terminal_consumer()
    report = _report_for(
        [leaf, helper, dead, term],
        edges=[_data_edge(eid="e1", source="h1", target="dead")])
    refs = report.ranked_hypotheses[0].source_refs
    assert not any(ref.explanation.startswith("upstream helper contribution:")
                   for ref in refs)
    assert any(ref.explanation.startswith("terminal write:")
               for ref in refs)


def test_helper_distinct_sites_preserved_as_identities():
    """H10: same callable at two caller sites yields two helper
    identities (selection layer); both anchors coincide, so report
    emission still collapses to one ref (see overlap test)."""
    from flight_log_agent.analysis.dag_pipeline import (
        _qualified_helper_entries,
    )
    from flight_log_agent.analysis.mechanism_dag import MechanismDAG
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    first = _wired_helper_vertex(vid="h1")
    second = _helper_return_vertex(
        vid="h2", call_site_id="src/other.cpp:5:1:legacy_call")
    term = _terminal_consumer()
    dag = MechanismDAG(
        dag_id="dag-acceptance", terminal="_final_out",
        vertices=[leaf, first, second, term],
        edges=[_data_edge(eid="e1", source="h1", target="term"),
               _data_edge(eid="e2", source="h2", target="term")])
    qualified = _qualified_helper_entries(dag)
    assert len(qualified) == 2
    assert qualified[0][1] != qualified[1][1]
    report = _report_for(
        [leaf, first, second, term],
        edges=[_data_edge(eid="e1", source="h1", target="term"),
               _data_edge(eid="e2", source="h2", target="term")])
    refs = report.ranked_hypotheses[0].source_refs
    helpers = [ref for ref in refs
               if ref.explanation.startswith(
                   "upstream helper contribution:")]
    assert len(helpers) == 1


def test_helper_cross_path_dedup():
    """H11/H6: one helper reachable directly and via an
    intermediate op still yields one helper ref."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(vid="h1")
    mid = _report_vertex(
        vid="mid", variable="mid_val", expression="helper_value + 1.0",
        file="src/main.cpp", line=25)
    mid.metadata = {}
    term = _terminal_consumer()
    report = _report_for(
        [leaf, helper, mid, term],
        edges=[_data_edge(eid="e1", source="h1", target="term"),
               _data_edge(eid="e2", source="h1", target="mid"),
               _data_edge(eid="e3", source="mid", target="term",
                          role="data")])
    refs = report.ranked_hypotheses[0].source_refs
    helpers = [ref for ref in refs
               if ref.explanation.startswith("upstream helper contribution:")]
    assert len(helpers) == 1


def test_helper_terminal_overlap_single_terminal_ref():
    """H12/H7: helper anchor coinciding with a terminal write
    yields one ref with terminal wording."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    helper = _wired_helper_vertex(
        vid="h1", file="src/main.cpp", line=30)
    term = _terminal_consumer()
    report = _report_for(
        [leaf, helper, term],
        edges=[_data_edge(eid="e1", source="h1", target="term")])
    refs = [ref for ref in report.ranked_hypotheses[0].source_refs
            if (ref.file, ref.start_line) == ("src/main.cpp", 30)]
    assert len(refs) == 1
    assert refs[0].explanation.startswith("terminal write:")


def test_helper_tier_between_terminal_and_anchor():
    """H8: ordering is terminal, then helper, then anchor."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    anchor = _report_vertex(
        vid="decl", variable="_final_out", expression="0.0",
        file="include/decl.h", line=3)
    helper = _wired_helper_vertex(vid="h1")
    term = _terminal_consumer()
    report = _report_for(
        [leaf, anchor, helper, term],
        edges=[_data_edge(eid="e1", source="h1", target="term")])
    refs = report.ranked_hypotheses[0].source_refs
    kinds = [("helper" if ref.explanation.startswith(
        "upstream helper contribution:")
        else "anchor" if ref.file.endswith("decl.h")
        else "terminal") for ref in refs]
    assert kinds == ["terminal", "helper", "anchor"]


def test_helper_confidence_isolation():
    """H11: helper presence does not change confidence or
    confirmation outcomes."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    term = _terminal_consumer()
    vertices = [leaf, term]
    plain = _report_for(vertices)
    helper = _wired_helper_vertex(vid="h1")
    with_helper = _report_for(
        [leaf, helper, term],
        edges=[_data_edge(eid="e1", source="h1", target="term")])
    assert (with_helper.ranked_hypotheses[0].confidence
            == plain.ranked_hypotheses[0].confidence)
    assert with_helper.confirmed == plain.confirmed


def _runtime_terminal(*, vid, variable, file="src/ops.cpp", line=10):
    return _report_vertex(
        vid=vid, variable=variable, expression=f"{variable}_in * 2.0",
        file=file, line=line)


def test_helper_omitted_when_terminals_saturate_budget():
    """H15 saturated: 8 distinct causal terminal refs leave no
    room; the qualified helper is omitted, terminals intact."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    terms = [_runtime_terminal(vid=f"t{i}", variable=f"out_{i}",
                              line=10 + i) for i in range(8)]
    helper = _wired_helper_vertex(vid="h1")
    edges = [_data_edge(eid=f"e{i}", source="h1", target=f"t{i}")
             for i in range(8)]
    report = _report_for([leaf, helper, *terms], edges=edges)
    refs = report.ranked_hypotheses[0].source_refs
    assert len(refs) == 8
    assert not any(ref.explanation.startswith("upstream helper contribution:")
                   for ref in refs)


def test_helper_precedes_anchors_with_remaining_capacity():
    """H15 remaining capacity: 7 terminals + helper + anchors
    keeps the helper before anchors."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    terms = [_runtime_terminal(vid=f"t{i}", variable=f"out_{i}",
                              line=10 + i) for i in range(7)]
    anchors = [
        _report_vertex(
            vid=f"d{i}", variable=f"out_{i}", expression="0.0",
            file="include/decl.h", line=20 + i)
        for i in range(7)]
    helper = _wired_helper_vertex(vid="h1")
    edges = [_data_edge(eid=f"e{i}", source="h1", target=f"t{i}")
             for i in range(7)]
    report = _report_for([leaf, helper, *terms, *anchors], edges=edges)
    refs = report.ranked_hypotheses[0].source_refs
    assert len(refs) == 8
    kinds = [("helper" if ref.explanation.startswith(
        "upstream helper contribution:")
        else "terminal") for ref in refs]
    assert kinds == ["terminal"] * 7 + ["helper"]


def test_many_helpers_order_deterministically():
    """H16: qualifying helpers order by source-derived keys,
    independent of input order; no semantic ranking invented."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    term = _terminal_consumer()
    first = _helper_return_vertex(
        vid="h1", file="src/h.cpp", line=20,
        callable_id="src/h.cpp:20:aaa:value",
        call_site_id="src/c.cpp:10:3:legacy_call")
    second = _helper_return_vertex(
        vid="h2", file="src/i.cpp", line=30,
        callable_id="src/i.cpp:30:bbb:value",
        call_site_id="src/c.cpp:11:3:legacy_call")
    edges = [_data_edge(eid="e1", source="h1", target="term"),
             _data_edge(eid="e2", source="h2", target="term")]
    forward = _report_for([leaf, first, second, term], edges=edges)
    backward = _report_for([leaf, term, second, first], edges=edges)
    for report in (forward, backward):
        helpers = [ref for ref in report.ranked_hypotheses[0].source_refs
                   if ref.explanation.startswith(
                       "upstream helper contribution:")]
        assert [(ref.file, ref.start_line) for ref in helpers] == [
            ("src/h.cpp", 20), ("src/i.cpp", 30)]
    assert ([(ref.file, ref.start_line) for ref in forward
             .ranked_hypotheses[0].source_refs]
            == [(ref.file, ref.start_line) for ref in backward
                .ranked_hypotheses[0].source_refs])


def test_report_budget_keeps_causal_runtime_ref():
    """D3/T5: with more refs than budget, an eligible causal
    runtime ref survives selection without vertex-ID ordering."""
    leaf = _report_observed_leaf(vid="leaf", signal="x.y")
    anchors = [
        _report_vertex(
            vid=f"decl-{line}", variable="_final_out", expression="0.0",
            file="include/decl.h", line=line)
        for line in range(20, 28)
    ]
    computation = _report_vertex(
        vid="calc", variable="_final_out", expression="input * 3.0",
        file="src/calc.cpp", line=10)
    report = _report_for([leaf, *anchors, computation])
    refs = report.ranked_hypotheses[0].source_refs
    assert len(refs) <= 8
    assert refs[0].file == "src/calc.cpp"
