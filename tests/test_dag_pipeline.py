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
    _dest_val = gspeed;
    if (_param_rtl_type.get() == 1) {
        _final_out = _dest_val + 1.0f;
    }
}
""",
}


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
        [{"target_symbol": "_x", "source_symbol": "tecs_status.true_airspeed_sp",
          "assignment_path": [{"file": "a.cpp", "line": 1,
                               "expression": "tecs_status.true_airspeed_sp"}],
          "logged_signal": "", "control_predicates": [], "function": "A::run"}],
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
        [{"target_symbol": "_x", "source_symbol": "topic_in.value",
          "assignment_path": [{"file": "a.cpp", "line": 1,
                               "expression": "topic_in.value"}],
          "logged_signal": "topic_out.value", "control_predicates": [],
          "function": "A::run"}],
        "_x", logged_signals={"topic_in.value", "topic_out.value"},
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


def test_replay_reuses_supplied_dag_signal_data(monkeypatch):
    from flight_log_agent.analysis import dag_pipeline
    from flight_log_agent.analysis.mechanism_dag import (
        build_mechanism_dag,
        prepare_signal_series,
    )

    dag = build_mechanism_dag(
        [{
            "target_symbol": "output",
            "source_symbol": "topic_in.value",
            "assignment_path": [{
                "file": "a.cpp",
                "line": 1,
                "expression": "topic_in.value",
            }],
            "logged_signal": "topic_out.value",
            "control_predicates": [],
            "function": "A::run",
        }],
        "output",
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
    replay = dag_pipeline.replay_terminal_expressions(
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
        {
            "target_symbol": "_out",
            "source_symbol": "input.first",
            "assignment_path": [
                {"file": "a.cpp", "line": 10, "expression": "input.first"}
            ],
            "logged_signal": "output.value",
            "control_predicates": ["mode.state == 0"],
            "control_predicate_lines": [9],
            "function": "A::run",
        },
        {
            "target_symbol": "_out",
            "source_symbol": "input.second",
            "assignment_path": [
                {"file": "a.cpp", "line": 12, "expression": "input.second"}
            ],
            "logged_signal": "output.value",
            "control_predicates": ["mode.state == 1"],
            "control_predicate_lines": [11],
            "function": "A::run",
        },
    ]
    dag = build_mechanism_dag(bindings, "_out", logged_signals=logged)
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
