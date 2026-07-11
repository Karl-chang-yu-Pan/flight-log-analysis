from __future__ import annotations

import asyncio
from pathlib import Path

from flight_log_agent.analysis.dag_pipeline import (
    build_report_from_dag,
    layer4_cache_path,
    read_seeds_from_cache,
    run_dag_discovery_stage,
    write_seeds_to_cache,
)
from flight_log_agent.analysis.mechanism_dag import layer2_cache_path, layer3_cache_path
from flight_log_agent.analysis.mechanism_judge import (
    DiscoverySeeds,
    DiscoveryVerdict,
    TerminalCandidate,
    judge_agent,
    seeder_agent,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler

TREE = {
    "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    if (_param_rtl_type.get() == 1) {
        _final_out = _dest_val + 1.0f;
    }
    _dest_val = 5.0f;
}
""",
}


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
        return DiscoveryVerdict(
            sufficient=True,
            selected_terminal="_final_out",
            explaining_branches=["_param_rtl_type.get() == 1"],
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


def test_stage_writes_layers_and_reuses_seeds(tmp_path):
    profiler = _mini_tree(tmp_path)
    cache_root = tmp_path / "cache"
    calls: list[str] = []
    kwargs = dict(
        inventory={"parameters": {"RTL_TYPE": 1}},
        ulog_hash="ulog0",
        run_agent=_stub_runner(calls),
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
    assert layer2_cache_path(cache_root, "srchash", "_final_out").exists()
    assert layer3_cache_path(cache_root, "srchash", "ulog0", "_final_out").exists()
    assert read_seeds_from_cache(
        layer4_cache_path(cache_root, "why did final_out increase?")
    ) is not None
    assert stage.report.ranked_hypotheses[0].confidence == "medium"

    # second run: Layer 4 hit skips the seeder; judge still consulted
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
    assert again.layer4_hit is True
    assert calls == [seeder_agent.name, judge_agent.name, judge_agent.name]


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
        )
    )

    hypothesis = stage.report.ranked_hypotheses[0]
    assert hypothesis.known_px4_mechanism == "_final_out"
    assert hypothesis.mechanism.startswith("terminal grounded")
    assert stage.report.confirmed == [hypothesis.title]
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
        )
    )

    hypothesis = stage.report.ranked_hypotheses[0]
    assert hypothesis.confidence == "unresolved"
    assert stage.report.unconfirmed == [hypothesis.title]
    assert stage.report.confirmed == []


def _run_with_verdict(tmp_path, verdict_kwargs):
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
        return Verdict(sufficient=True, selected_terminal="_final_out",
                       **verdict_kwargs)

    return asyncio.run(
        run_dag_discovery_stage(
            profiler, tmp_path / "cache", "why?", "srchash",
            Path("/nonexistent.ulg"), ulog_hash="u", run_agent=run,
        )
    )


def test_sufficient_with_unmatched_branch_is_downgraded(tmp_path):
    stage = _run_with_verdict(
        tmp_path, {"explaining_branches": ["some_bogus_predicate > 99"]}
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert stage.report.confirmed == []
    assert any("feasibility-dead" in u or "absent" in u for u in h.unresolved_evidence)


def test_sufficient_without_named_branch_is_downgraded(tmp_path):
    stage = _run_with_verdict(tmp_path, {})
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "low"
    assert stage.report.confirmed == []
    assert any("without naming" in u for u in h.unresolved_evidence)


def test_sufficient_with_live_matching_branch_stays_confirmed(tmp_path):
    stage = _run_with_verdict(
        tmp_path, {"explaining_branches": ["_param_rtl_type.get() == 1"]}
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "medium"
    assert stage.report.confirmed == [h.title]


def test_layer4_path_includes_seeder_fingerprint_and_prunes_stale(tmp_path):
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
    assert not stale.exists()


def test_explaining_branch_matches_despite_render_suffix_and_truncation(tmp_path):
    """The judge copies branch entries verbatim from the rendering,
    including the trailing feasibility tag and truncation ellipsis."""
    stage = _run_with_verdict(
        tmp_path,
        {"explaining_branches": ["_param_rtl_type.get() ==… [unknown]"]},
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "medium"
    assert stage.report.confirmed == [h.title]


def test_explaining_branch_matches_despite_windowed_tag(tmp_path):
    stage = _run_with_verdict(
        tmp_path,
        {"explaining_branches": ["_param_rtl_type.get() == 1 [unknown; active 2w 1.0-2.0s]"]},
    )
    h = stage.report.ranked_hypotheses[0]
    assert h.confidence == "medium"
    assert stage.report.confirmed == [h.title]


def test_replay_verified_upgrades_confidence_to_high(tmp_path):
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds as Seeds, DiscoveryVerdict as Verdict,
        TerminalCandidate as Cand, seeder_agent as seeder,
    )
    from flight_log_agent.analysis.dag_pipeline import build_report_from_dag
    from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag

    profiler = _mini_tree(tmp_path)
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", ["pick_altitude"], "_final_out", "hash")
    from flight_log_agent.analysis.mechanism_judge import JudgedDiscovery
    judged = JudgedDiscovery(
        seeds=Seeds(seeds=[], candidate_terminals=[Cand(terminal="_final_out")]),
        verdict=Verdict(sufficient=True, selected_terminal="_final_out",
                        explaining_branches=["_param_rtl_type.get() == 1"]),
        results={"_final_out": result}, selected=result,
    )
    replay = {"observed": "x.y", "verified": True,
              "results": [{"grounded": "a+b", "evaluable": True, "match_fraction": 0.9}]}
    report = build_report_from_dag("why?", judged, result.dag, replay=replay)
    assert report.ranked_hypotheses[0].confidence == "high"
    assert any("expression replay" in e for e in report.ranked_hypotheses[0].evidence)

    unverified = build_report_from_dag("why?", judged, result.dag,
                                       replay={"verified": False, "results": []})
    assert unverified.ranked_hypotheses[0].confidence == "medium"


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
