"""Deterministic full-system acceptance: TECS restart transient.

W3B case driven jointly with the W3A temporal capability. The real
log carries a mode transition immediately followed by a ~1.012 m/s
height-rate setpoint that later reaches ~5 m/s: a bumpless/restart
transient, not a persistent ~1 m/s limit. Exact writer execution
(init vs update) is unavailable/NON_DECISIVE by design; the
mechanism-level conclusion does not depend on it.
"""
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TECS_LOG = REPO_ROOT / (
    "uploads/tecs-switch-moded-hrate/switch-moded-hrate.ulg")
SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
PINNED_COMMIT = "1dacb4cdef2d7145754fc788fa8dc482eed74b40"

TECS_QUESTION = ("Why did the height-rate setpoint read about "
                 "1.012 m/s immediately after the mode transition, "
                 "rather than holding a steady climb-rate limit?")


def _tecs_transition_intent():
    """Case-specific transition intent (test data, never production)."""
    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    return TransitionEventSpec(
        transition_signal="vehicle_status.nav_state",
        from_value=15,
        to_value=4,
        event_selection="first",
        relation="after",
        first_sample_of="tecs_status.height_rate_setpoint",
    )


def _tecs_frame():
    """Transition/first-sample frame derived from raw log columns.

    Independent ground truth for the production derivation: raw
    pyulog samples, no pipeline structures involved.
    """
    from pyulog import ULog

    log = ULog(str(TECS_LOG))
    nav = next(m.data for m in log.data_list if m.name == "vehicle_status")
    stamps = [t / 1e6 for t in nav["timestamp"]]
    states = [int(v) for v in nav["nav_state"]]
    transition = next(
        stamps[i + 1] for i in range(len(stamps) - 1)
        if states[i] == 15 and states[i + 1] == 4
    )
    tecs = next(m.data for m in log.data_list if m.name == "tecs_status")
    pairs = sorted(
        (t / 1e6, float(v))
        for t, v in zip(tecs["timestamp"], tecs["height_rate_setpoint"])
    )
    first_time, first_value = next(
        (t, v) for t, v in pairs if t >= transition
    )
    reference = float(next(
        v for t, v in sorted(
            (t / 1e6, float(v))
            for t, v in zip(tecs["timestamp"], tecs["height_rate_reference"])
        ) if t >= transition
    ))
    later = [v for t, v in pairs if t >= transition]
    return {
        "transition": transition,
        "first_time": first_time,
        "first_value": first_value,
        "reference": reference,
        "later": later,
    }


def test_t1_tecs_fixture_loads_and_calibrates():
    """T1-analog: real .ulg parses via production inventory; pinned
    source snapshot verified; TECS/mode signals calibrated."""
    if not TECS_LOG.exists():
        raise AssertionError(
            "TECS acceptance log missing: "
            f"{TECS_LOG} (gitignored fixture; provision the "
            "switch-moded-hrate.ulg at this exact path)")
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(TECS_LOG, SOURCE_ROOT)
    assert inventory, "production inventory is empty"

    head = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_COMMIT, f"source snapshot {head} != pin"

    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
    )
    observed = set(observed_signals_from_inventory(inventory))
    for signal in (
            "vehicle_status[0].nav_state",
            "tecs_status[0].height_rate_setpoint",
            "tecs_status[0].height_rate_reference"):
        assert signal in observed, f"signal not surfaced: {signal}"


def test_t14_tecs_transition_window_derived_from_real_log(tmp_path):
    """T14: W3A derives the transition-relative diagnostic window
    from the real log, matching the independently parsed frame."""
    from flight_log_agent.analysis.dag_pipeline import (
        evaluate_transition_windows,
    )
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    inventory = parse_ulog_inventory(TECS_LOG, SOURCE_ROOT)
    logged = set(observed_signals_from_inventory(inventory))
    result = evaluate_transition_windows(
        _tecs_transition_intent(),
        logged_set=logged,
        log_path=TECS_LOG,
        signal_policies={},
    )
    assert result["windows"] is not None, result.get("error")
    frame = _tecs_frame()
    (start, end), = result["windows"]
    assert start == frame["transition"]
    assert end == frame["first_time"]
    print(f"\nT14 transition={start:.4f} first={end:.4f} "
          f"delta={(end - start) * 1000:.1f}ms")


def _tecs_stub_runner(calls: list):
    """Deterministic stand-in for LLM-owned seeder/judge decisions.

    Fixed TECS seeds (including the transition-event intent),
    fixed terminal, fixed verdict. Fails loudly on unexpected
    invocation; injects no mechanism evidence.
    """
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds,
        DiscoveryVerdict,
        QuestionedCondition,
        TerminalCandidate,
        judge_agent,
        seeder_agent,
    )

    async def run(agent, payload):
        calls.append(agent.name)
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["TECS", "altitude_rate_control",
                       "tecs_status", "_reinitialize_tecs"],
                candidate_terminals=[TerminalCandidate(
                    terminal="_debug_output.altitude_rate_control",
                    terminal_file="src/lib/tecs/TECS.cpp")],
                questioned_condition=QuestionedCondition(
                    signal_hint="tecs_status.height_rate_setpoint",
                    op=">",
                    reference="1.0",
                    units="signal: m/s; reference: m/s",
                    frame="test frame",
                    transition=_tecs_transition_intent(),
                ),
            )
        if agent is judge_agent:
            return DiscoveryVerdict(
                sufficient=True,
                selected_terminal="_debug_output.altitude_rate_control",
                explaining_branches=[],
                reasoning="acceptance stub selects the grounded slice",
            )
        raise AssertionError(
            f"unexpected LLM-owned invocation: {agent.name}")

    return run


async def _run_tecs_stage(tmp_path, calls: list):
    """Shared T15/T16/T18 driver: real log + pinned source +
    stubbed decisions through the production DAG stage."""
    from flight_log_agent.analysis.dag_pipeline import (
        run_dag_discovery_stage,
    )
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    from flight_log_agent.px4.msg_schema import load_px4_msg_schema
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )
    inventory = parse_ulog_inventory(TECS_LOG, SOURCE_ROOT)
    schema = load_px4_msg_schema(SOURCE_ROOT)
    schema_signals = sorted(
        f"{topic}.{field}" for topic, fields in schema.items()
        for field in fields)
    profiler = MechanismSourceProfiler(
        SOURCE_ROOT, source_parser_backend="tree_sitter")
    return await run_dag_discovery_stage(
        profiler, tmp_path / "cache", TECS_QUESTION, "acceptance",
        TECS_LOG, inventory=inventory,
        run_agent=_tecs_stub_runner(calls),
        logged_signals=set(observed_signals_from_inventory(inventory)),
        schema_signals=set(schema_signals))


import pytest


@pytest.fixture(scope="module")
def _tecs_stage(tmp_path_factory):
    """One shared production stage run for T15/T16/T18."""
    import asyncio
    import time

    tmp_path = tmp_path_factory.mktemp("tecs_stage")
    calls: list = []
    started = time.perf_counter()
    result = asyncio.run(_run_tecs_stage(tmp_path, calls))
    return result, calls, time.perf_counter() - started


def _tecs_terminal_writers(dag):
    """Real terminal writers for the setpoint member, keyed by line."""
    writers = {}
    for vertex in dag.vertices:
        if vertex.kind != "operation":
            continue
        if not (vertex.metadata or {}).get("is_terminal"):
            continue
        if vertex.variable != "_debug_output.altitude_rate_control":
            continue
        writers[vertex.line] = vertex
    return writers


def _data_ancestors(dag, roots):
    by_id = {vertex.id: vertex for vertex in dag.vertices}
    incoming = {}
    for edge in dag.edges:
        if edge.kind == "data":
            incoming.setdefault(edge.target_id, []).append(edge.source_id)
    seen = set(roots)
    stack = list(roots)
    while stack:
        for source in incoming.get(stack.pop(), ()):
            if source not in seen:
                seen.add(source)
                stack.append(source)
    return [by_id[vid] for vid in seen if vid in by_id]


def test_t15_tecs_candidate_anchors_in_source_and_dag(_tecs_stage):
    """T15: both writer candidates exist at the pinned source sites
    and are represented as terminal writers. Anchors only — no
    execution attribution."""
    result, _calls, _elapsed = _tecs_stage
    lines = (SOURCE_ROOT / "src/lib/tecs/TECS.cpp").read_text().splitlines()
    for number in (262, 296):
        text = lines[number - 1].strip()
        assert text.startswith("_debug_output.altitude_rate_control ="), \
            f"pinned source line {number} moved: {text!r}"
    dag = result.annotated_dag
    assert dag is not None and dag.vertices, "no DAG built"
    writers = _tecs_terminal_writers(dag)
    assert set(writers) == {262, 296}, \
        f"expected exactly the two writer candidates, found lines {sorted(writers)}"
    assert len(writers) == 2


def test_t16_tecs_writers_share_downstream_mechanism(_tecs_stage):
    """T16 (STOP G gate): both retained writers feed the same
    downstream altitude-rate-control mechanism for THIS claim.
    Proven structurally — same helper callable reached through
    distinct call instances in each writer cone — never by
    same-text alone. Failure here is STOP G."""
    from flight_log_agent.analysis.dag_pipeline import (
        _helper_representative_identity,
    )

    result, _calls, _elapsed = _tecs_stage
    dag = result.annotated_dag
    writers = _tecs_terminal_writers(dag)
    assert set(writers) == {262, 296}
    assert all(
        vertex.variable == "_debug_output.altitude_rate_control"
        for vertex in writers.values()
    )
    by_id = {vertex.id: vertex for vertex in dag.vertices}
    callables = set()
    sites = set()
    for line, writer in writers.items():
        cone = {vertex.id for vertex in _data_ancestors(dag, [writer.id])}
        cone.add(writer.id)
        hits = [
            (edge.source_id, edge.target_id)
            for edge in dag.edges
            if edge.kind == "data"
            and edge.role == "call:_calcAltitudeControlOutput"
            and edge.target_id in cone
        ]
        assert hits, \
            f"writer at line {line} has no altitude-rate-control call: STOP G"
        for source_id, _target_id in hits:
            identity = _helper_representative_identity(
                by_id[source_id], dag)
            assert identity is not None, \
                f"unidentifiable helper at line {line}: STOP G"
            callables.add(identity[0])
            sites.add(identity[1])
    # One shared helper definition across both cones (same callable),
    # reached through distinct call instances (distinct sites): the
    # mechanism-equivalence structure. Same text alone proves nothing;
    # same callable + distinct instances does.
    assert len(callables) == 1, \
        f"writer cones use different helpers {callables}: STOP G"
    assert len(sites) > 1, \
        "expected distinct call instances, not one shared vertex"
    assert callables.pop().endswith("_calcAltitudeControlOutput:setpoint,input,param")
    # Conditions 1-4 (spec §17a): shared helper + same destination
    # member leave no rival mechanism supported by either retained
    # candidate (1-2); nothing here names an executing writer (3);
    # independent timing/numeric evidence is established by
    # T14/T17/T18, not by this test (4).
    print(f"\nT16 shared helper reached from {sorted(sites)}")


def test_t17_tecs_first_sample_timing_and_numeric():
    """T17: exact transition/first-sample timing plus the accepted
    0.3 × reference relation, all from the real log. Pins integer
    microsecond timestamps — never rounded reminders."""
    frame = _tecs_frame()
    assert round(frame["transition"] * 1e6) == 496881397
    assert round(frame["first_time"] * 1e6) == 496899343
    delta_ms = (frame["first_time"] - frame["transition"]) * 1000.0
    assert 0.0 < delta_ms < 100.0, \
        f"first sample must shortly follow transition, delta={delta_ms}ms"
    print(f"\nT17 delta={delta_ms:.3f}ms")
    assert frame["reference"] == 3.3737552165985107
    assert frame["first_value"] == 1.0121265649795532
    assert abs(frame["first_value"] - 0.3 * frame["reference"]) <= 1e-9, \
        "first setpoint must equal 0.3 × reference"
    print(f"T17 0.3*{frame['reference']}={0.3 * frame['reference']:.6f} "
          f"vs logged {frame['first_value']}")


def test_t18_tecs_later_behavior_and_rival_exclusion(_tecs_stage):
    """T18: later setpoint reaches ~5 m/s, rejecting the persistent
    ~1 m/s limit rival; mechanism grounded in represented DAG
    structure outside report CodeRefs (no fabrication)."""
    result, _calls, _elapsed = _tecs_stage
    frame = _tecs_frame()
    assert max(frame["later"]) == 5.0
    assert frame["first_value"] < 1.5 < max(frame["later"]), \
        "early transient plus later normal behavior required"
    dag = result.annotated_dag
    by_id = {vertex.id: vertex for vertex in dag.vertices}
    mechanism_vertices = [
        vertex for vertex in dag.vertices
        if str(vertex.expression or "").startswith(
            "_calcAltitudeControlOutput(")
    ]
    assert mechanism_vertices, \
        "shared altitude-rate-control equation not represented"
    assert {vertex.file for vertex in mechanism_vertices} == {
        "src/lib/tecs/TECS.cpp"}, \
        "shared mechanism must live at the pinned source site"


SIDECAR = REPO_ROOT / "tests/acceptance/tecs_restart_transient.json"


def test_t19_tecs_sidecar_strength_and_availability(_tecs_stage):
    """T19: sidecar preserves PROVEN strength, records exact writer
    execution as unavailable/NON_DECISIVE, and forbids execution
    claims and confidence forcing."""
    import json

    assert SIDECAR.exists(), "semantic sidecar missing"
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    for key in ("scenario_id", "question", "inputs", "expected",
                "strength", "unavailable", "forbidden",
                "normalization"):
        assert key in sidecar, f"sidecar missing category: {key}"
    assert sidecar["strength"] == "PROVEN"
    assert sidecar["inputs"]["log"] == str(TECS_LOG.relative_to(REPO_ROOT))
    assert sidecar["inputs"]["source_snapshot"] == PINNED_COMMIT

    expected = sidecar["expected"]
    frame = _tecs_frame()
    tolerance = float(expected["tolerance"])
    assert abs(frame["transition"] - expected["transition_time_s"]) <= tolerance
    assert abs(frame["first_time"] - expected["first_sample_time_s"]) <= tolerance
    assert abs(frame["reference"] - expected["reference"]) <= tolerance
    assert abs(frame["first_value"] - expected["first_setpoint"]) <= tolerance
    assert abs(max(frame["later"]) - expected["later_setpoint"]) <= tolerance
    assert abs(0.3 * expected["reference"] - expected["first_setpoint"]) <= tolerance

    unavailable = " ".join(sidecar["unavailable"])
    assert "initialize-vs-update" in unavailable
    assert "NON_DECISIVE" in unavailable
    forbidden = " ".join(sidecar["forbidden"])
    assert "execution" in forbidden

    result, _calls, _elapsed = _tecs_stage
    report = result.report
    for hypothesis in report.ranked_hypotheses:
        for ref in hypothesis.source_refs:
            assert "execut" not in ref.explanation.lower(), \
                f"execution claim in source ref: {ref.explanation}"
    assert report.confirmed == [], \
        "temporal acceptance must not force confirmation"
