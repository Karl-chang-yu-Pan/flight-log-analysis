from __future__ import annotations

import asyncio

from flight_log_agent.analysis.mechanism_discovery import discover_mechanism_dag
from flight_log_agent.analysis.mechanism_judge import (
    DiscoverySeeds,
    DiscoveryVerdict,
    TerminalCandidate,
    discover_with_judge,
    judge_agent,
    render_discovery_compact,
    seeder_agent,
)
from flight_log_agent.px4.mechanism_source_profiler import MechanismSourceProfiler

TWO_FILE_TREE = {
    "src/modules/example/rtl.cpp": """
void Rtl::pick_altitude()
{
    if (_param_rtl_type.get() == 1) {
        _final_out = _dest_val + 1.0f;
    }
}
""",
    "src/modules/example/dest.cpp": """
void Rtl::update()
{
    _dest_val = gspeed;
}
""",
}


def _mini_tree(tmp_path, files: dict[str, str]) -> MechanismSourceProfiler:
    root = tmp_path / "PX4-Autopilot"
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return MechanismSourceProfiler(root, rg_path="missing-rg")


def _discover(tmp_path, profiler, **kwargs):
    return discover_mechanism_dag(
        profiler,
        tmp_path / "cache",
        seeds=["pick_altitude"],
        terminal="_final_out",
        source_hash="hash",
        logged_signals={"gspeed"},
        **kwargs,
    )


def test_render_discovery_compact_carries_the_facts(tmp_path):
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    result = _discover(tmp_path, profiler)

    render = render_discovery_compact(result)

    assert render["terminal"] == "_final_out"
    assert any(
        op.startswith("_final_out <- _dest_val + 1.0 @ src/modules/example/rtl.cpp:")
        for op in render["operations"]
    )
    assert any("_param_rtl_type" in b for b in render["branches"])
    assert "gspeed" in render["evidence"]["logged_signal"]
    # No DEFINE_PARAMETERS block in the fixture, so the param member has
    # no canonical name and honestly renders as unresolved.
    assert render["unresolved_symbols"] == ["_param_rtl_type.get"]
    assert [r["index"] for r in render["rounds"]] == [0, 1]
    assert render["files_loaded"][0] == "src/modules/example/rtl.cpp"


def test_render_discovery_compact_caps_lists(tmp_path):
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    result = _discover(tmp_path, profiler)

    render = render_discovery_compact(result, max_operations=1)

    assert len(render["operations"]) == 2
    assert render["operations"][1].startswith("… +")


def test_discover_with_judge_runs_seeder_discovery_judge(tmp_path):
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    calls: list[tuple[str, dict]] = []

    async def stub_runner(agent, payload):
        calls.append((agent.name, payload))
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[
                    TerminalCandidate(
                        terminal="_final_out",
                        terminal_file="src/modules/example/rtl.cpp",
                    )
                ],
            )
        assert agent is judge_agent
        return DiscoveryVerdict(sufficient=True, selected_terminal="_final_out")

    judged = asyncio.run(
        discover_with_judge(
            profiler,
            tmp_path / "cache",
            "why did final_out increase?",
            "hash",
            run_agent=stub_runner,
            logged_signals={"gspeed"},
        )
    )

    assert [name for name, _ in calls] == [seeder_agent.name, judge_agent.name]
    judge_payload = calls[1][1]
    assert "_final_out" in judge_payload["candidates"]
    assert judge_payload["candidates"]["_final_out"]["operations"]
    assert judged.selected is judged.results["_final_out"]
    assert "_dest_val" not in judged.selected.dag.unresolved_symbols
    assert judged.bonus_round_used is False


def test_judge_grants_one_bonus_round_for_essential_gaps(tmp_path):
    """max_rounds=1 leaves _dest_val unresolved; the judge names it as an
    essential gap and discovery reruns ONCE with the gap as a seed, which
    pulls dest.cpp and completes the slice."""
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        return DiscoveryVerdict(
            sufficient=False,
            selected_terminal="_final_out",
            essential_gaps=["_dest_val"],
        )

    judged = asyncio.run(
        discover_with_judge(
            profiler,
            tmp_path / "cache",
            "why did final_out increase?",
            "hash",
            run_agent=stub_runner,
            logged_signals={"gspeed"},
            max_rounds=1,
        )
    )

    assert judged.bonus_round_used is True
    assert "_dest_val" not in judged.selected.dag.unresolved_symbols
    op_targets = {
        v.variable for v in judged.selected.dag.vertices if v.kind == "operation"
    }
    assert "_dest_val" in op_targets


def test_render_lists_unexpanded_calls(tmp_path):
    profiler = _mini_tree(tmp_path, {
        "src/modules/example/fw.cpp": """
void Fw::run()
{
    _airspeed_sp = adapt_airspeed_setpoint(_base_sp);
}
""",
    })
    result = discover_mechanism_dag(
        profiler, tmp_path / "cache", seeds=["Fw::run"],
        terminal="_airspeed_sp", source_hash="hash",
    )
    render = render_discovery_compact(result)
    assert "adapt_airspeed_setpoint" in render["unexpanded_calls"]


def test_judge_next_terminals_reterminal_bonus_round(tmp_path):
    """An insufficient verdict naming next_terminals re-slices ONCE from
    the judge-proposed decision-site variable; the re-judge then rules
    on the new terminal's graph."""
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    judge_calls: list[dict] = []

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        judge_calls.append(payload)
        if len(judge_calls) == 1:
            return DiscoveryVerdict(
                sufficient=False,
                selected_terminal="_final_out",
                next_terminals=[TerminalCandidate(terminal="_dest_val")],
            )
        return DiscoveryVerdict(sufficient=True, selected_terminal="_dest_val")

    judged = asyncio.run(
        discover_with_judge(
            profiler, tmp_path / "cache", "why?", "hash",
            run_agent=stub_runner, logged_signals={"gspeed"},
        )
    )
    assert judged.bonus_round_used is True
    assert judged.selected.dag.terminal == "_dest_val"
    assert "_dest_val" in judged.results


def test_empty_candidates_are_excluded_from_judge_choices(tmp_path):
    """A candidate whose slice found nothing must not be offered to the
    judge as a selectable choice; it is surfaced under empty_candidates."""
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    payloads: list[dict] = []

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[
                    TerminalCandidate(terminal="_ghost_var"),
                    TerminalCandidate(terminal="_final_out"),
                ],
            )
        payloads.append(payload)
        return DiscoveryVerdict(sufficient=True, selected_terminal="_final_out")

    judged = asyncio.run(
        discover_with_judge(
            profiler, tmp_path / "cache", "why?", "hash",
            run_agent=stub_runner, logged_signals={"gspeed"},
        )
    )

    assert list(payloads[0]["candidates"]) == ["_final_out"]
    assert payloads[0]["empty_candidates"] == ["_ghost_var"]
    # Validation tells the judge WHY the candidate is empty, so its
    # replacement proposal can carry a usable terminal_file.
    assert payloads[0]["rejected_terminals"] == {
        "_ghost_var": "no write target in loaded facts"
    }
    assert judged.selected is judged.results["_final_out"]


def test_prose_gap_entries_are_dropped_from_bonus_seeds(tmp_path):
    """Gap entries are verbatim search queries — a sentence greps nothing
    and must not poison the bonus round; symbol-shaped entries survive."""
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        return DiscoveryVerdict(
            sufficient=False,
            selected_terminal="_final_out",
            essential_gaps=[
                "_dest_val",
                "the DAG does not show how the setpoint is computed",
            ],
        )

    judged = asyncio.run(
        discover_with_judge(
            profiler, tmp_path / "cache", "why?", "hash",
            run_agent=stub_runner, logged_signals={"gspeed"},
            max_rounds=1,
        )
    )

    assert judged.bonus_round_used is True
    op_targets = {
        v.variable for v in judged.selected.dag.vertices if v.kind == "operation"
    }
    assert "_dest_val" in op_targets


def test_judge_sees_annotated_render_and_selected_annotated_returned(tmp_path):
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    payloads: list[dict] = []
    annotated_dags: list = []

    def fake_annotate(result):
        vertices = [
            v.model_copy(update={"feasibility_verdict": "always_true",
                                 "active_windows": [(10.0, 50.0)]})
            if v.kind == "branch" else v
            for v in result.dag.vertices
        ]
        annotated = result.dag.model_copy(update={"vertices": vertices})
        annotated_dags.append(annotated)
        return annotated

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        payloads.append(payload)
        return DiscoveryVerdict(sufficient=True, selected_terminal="_final_out")

    judged = asyncio.run(
        discover_with_judge(
            profiler, tmp_path / "cache", "why?", "hash",
            run_agent=stub_runner, logged_signals={"gspeed"},
            annotate=fake_annotate,
        )
    )

    branches = payloads[0]["candidates"]["_final_out"]["branches"]
    assert any("always_true; active 1w 10.0-50.0s" in b for b in branches)
    assert judged.selected_annotated is annotated_dags[-1]


def test_bonus_round_triggers_single_rejudge(tmp_path):
    """After the bonus round the improved graph is judged once more; the
    re-judge's own steering is not acted on (hard cap of one round)."""
    profiler = _mini_tree(tmp_path, TWO_FILE_TREE)
    judge_calls: list[dict] = []

    async def stub_runner(agent, payload):
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["pick_altitude"],
                candidate_terminals=[TerminalCandidate(terminal="_final_out")],
            )
        judge_calls.append(payload)
        if len(judge_calls) == 1:
            return DiscoveryVerdict(
                sufficient=False, selected_terminal="_final_out",
                essential_gaps=["_dest_val"],
            )
        return DiscoveryVerdict(
            sufficient=False, selected_terminal="_final_out",
            essential_gaps=["_never_acted_on"],
        )

    judged = asyncio.run(
        discover_with_judge(
            profiler, tmp_path / "cache", "why?", "hash",
            run_agent=stub_runner, logged_signals={"gspeed"},
            max_rounds=1,
        )
    )

    assert len(judge_calls) == 2
    assert judged.bonus_round_used is True
    op_targets = {v.variable for v in judged.selected.dag.vertices
                  if v.kind == "operation"}
    assert "_dest_val" in op_targets
    assert judged.verdict.essential_gaps == ["_never_acted_on"]
