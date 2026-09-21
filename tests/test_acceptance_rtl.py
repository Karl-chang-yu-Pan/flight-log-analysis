"""Deterministic full-system acceptance: RTL weird-height.

Slice A1 — fixture + raw evidence calibration through the real
production parser path. Later slices (stage, numerics, report,
sidecar) build on the calibration recorded here.
"""
import subprocess
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RTL_LOG = REPO_ROOT / (
    "uploads/053127df84724a75998ff355b5a2519a/RTL-wierd-height-1.ulg")
SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
PINNED_COMMIT = "1dacb4cdef2d7145754fc788fa8dc482eed74b40"


def test_a1_rtl_fixture_loads_and_calibrates():
    """A1: real .ulg parses via production inventory; pinned source
    snapshot verified; raw RTL parameters/signals calibrated."""
    if not RTL_LOG.exists():
        raise AssertionError(
            "RTL acceptance log missing: "
            f"{RTL_LOG} (gitignored fixture; provision the 16 MB "
            "RTL-wierd-height-1.ulg at this exact path)")
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    started = time.perf_counter()
    inventory = parse_ulog_inventory(RTL_LOG, SOURCE_ROOT)
    elapsed = time.perf_counter() - started
    print(f"\nA1 inventory parse: {elapsed:.1f}s")
    assert inventory, "production inventory is empty"

    parameters = dict((inventory or {}).get("parameters") or {})
    assert parameters.get("RTL_RETURN_ALT") == 10.0
    assert parameters.get("RTL_CONE_ANG") == 45
    assert parameters.get("NAV_ACC_RAD") == 10.0
    assert parameters.get("RTL_MIN_DIST") == 10.0

    head = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert head == PINNED_COMMIT, f"source snapshot {head} != pin"

    # Logged firmware identifies the same snapshot production uses.
    assert inventory.get("git_hash") == PINNED_COMMIT
    assert inventory.get("firmware_version") == "v1.14.3"

    # Signal convention is topic[instance].field; the acceptance
    # scenario's signals must be exposed through production parsing.
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
    )
    observed = set(observed_signals_from_inventory(inventory))
    for signal in (
            "vehicle_global_position[0].alt",
            "home_position[0].alt",
            "position_setpoint_triplet[0].current.alt",
            "vehicle_status[0].nav_state",
            "vehicle_land_detected[0].landed",
            "vehicle_local_position[0].z",
            "trajectory_setpoint[0].position[2]"):
        assert signal in observed, f"signal not surfaced: {signal}"


RTL_QUESTION = ("Why did RTL command a climb to about 104 m — roughly "
                "20 m above home — rather than remaining near the "
                "destination altitude around 84 m?")


def _rtl_stub_runner(calls: list):
    """Deterministic stand-in for LLM-owned seeder/judge decisions.

    Controls decisions only: fixed seeds/questioned target plus the
    verdict selecting the grounded slice. Fails loudly on any
    unexpected semantic invocation; injects no mechanism evidence.
    """
    from flight_log_agent.analysis.mechanism_judge import (
        DiscoverySeeds,
        DiscoveryVerdict,
        TerminalCandidate,
        judge_agent,
        seeder_agent,
    )

    async def run(agent, payload):
        calls.append(agent.name)
        if agent is seeder_agent:
            return DiscoverySeeds(
                seeds=["RTL_RETURN_ALT", "rtl_alt",
                       "RTL::find_RTL_destination"],
                candidate_terminals=[TerminalCandidate(
                    terminal="_rtl_alt",
                    terminal_file="src/modules/navigator/rtl.cpp")],
            )
        if agent is judge_agent:
            return DiscoveryVerdict(
                sufficient=True,
                selected_terminal="_rtl_alt",
                explaining_branches=[],
                reasoning="acceptance stub selects the grounded slice",
            )
        raise AssertionError(
            f"unexpected LLM-owned invocation: {agent.name}")

    return run


def test_a2_stub_drives_judge_stage(tmp_path):
    """A2: controlled run_agent drives discover_with_judge through
    seeder → deterministic expansion → judge, selecting _rtl_alt."""
    import asyncio

    from flight_log_agent.analysis.mechanism_judge import (
        discover_with_judge,
        judge_agent,
        seeder_agent,
    )
    from flight_log_agent.px4.mechanism_source_profiler import (
        MechanismSourceProfiler,
    )
    calls: list = []
    profiler = MechanismSourceProfiler(
        SOURCE_ROOT, source_parser_backend="tree_sitter")
    result = asyncio.run(discover_with_judge(
        profiler, tmp_path / "cache", RTL_QUESTION, "acceptance",
        run_agent=_rtl_stub_runner(calls)))
    assert seeder_agent.name in calls
    assert judge_agent.name in calls
    assert result.verdict.selected_terminal == "_rtl_alt"


async def _run_acceptance_stage(tmp_path, calls: list):
    """Shared A3/A5 driver: real log + pinned source + stubbed
    decisions through the production DAG stage."""
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
    inventory = parse_ulog_inventory(RTL_LOG, SOURCE_ROOT)
    schema = load_px4_msg_schema(SOURCE_ROOT)
    schema_signals = sorted(
        f"{topic}.{field}" for topic, fields in schema.items()
        for field in fields)
    profiler = MechanismSourceProfiler(
        SOURCE_ROOT, source_parser_backend="tree_sitter")
    return await run_dag_discovery_stage(
        profiler, tmp_path / "cache", RTL_QUESTION, "acceptance",
        RTL_LOG, inventory=inventory,
        run_agent=_rtl_stub_runner(calls),
        logged_signals=set(observed_signals_from_inventory(inventory)),
        schema_signals=set(schema_signals))


import pytest


@pytest.fixture(scope="module")
def _rtl_stage(tmp_path_factory):
    """One shared production stage run for A3/A5 (each run costs
    ~8 min of real source expansion; the fixture, not the
    assertions, is shared)."""
    import asyncio
    import time

    tmp_path = tmp_path_factory.mktemp("rtl_stage")
    calls: list = []
    started = time.perf_counter()
    result = asyncio.run(_run_acceptance_stage(tmp_path, calls))
    return result, calls, time.perf_counter() - started


def test_a3_stage_recovers_rtl_slice(_rtl_stage):
    """A3: real stage recovers the _rtl_alt slice from real
    fixtures — rtl.cpp evidence, selected terminal, replay."""
    result, calls, elapsed = _rtl_stage
    print(f"\nA3 stage runtime: {elapsed:.1f}s")
    dag = result.annotated_dag
    assert dag is not None and dag.vertices, "no DAG built"
    assert result.judged.verdict.selected_terminal == "_rtl_alt"
    files = {getattr(v, "file", "") or (v.metadata or {}).get("file", "")
             for v in dag.vertices}
    assert any("rtl.cpp" in str(f) for f in files), \
        f"rtl.cpp evidence missing: {sorted(str(f) for f in files)[:5]}"
    assert result.replay is not None
    print(f"\nA3 replay status: {result.replay.get('status')}, "
          f"complete: {result.replay.get('complete')}")


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance, same R=6371000 PX4 uses (geo.cpp)."""
    import math
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    arc = math.sin((phi2 - phi1) / 2) ** 2 + math.cos(phi1) * \
        math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(arc))


def _triplet_frame():
    """Destination/commanded frame derived from raw triplet columns
    (LOG_DERIVED): LAND sample carries the RTL destination
    (altitude + coordinates + acceptance radius); max valid type-2
    sample is the logged command."""
    from pyulog import ULog
    log = ULog(str(RTL_LOG))
    data = next(
        m.data for m in log.data_list
        if m.name == "position_setpoint_triplet")
    n = len(data["timestamp"])
    lands = [i for i in range(n)
             if data["current.valid"][i] and data["current.type"][i] == 4]
    assert lands, "no valid LAND setpoint in fixture"
    land = lands[0]
    commands = [float(data["current.alt"][i]) for i in range(n)
                if data["current.valid"][i]
                and data["current.type"][i] == 2]
    return {
        "dest_alt": float(data["current.alt"][land]),
        "dest_lat": float(data["current.lat"][land]),
        "dest_lon": float(data["current.lon"][land]),
        "acceptance_radius": float(
            data["current.acceptance_radius"][land]),
        "logged_command": max(commands),
    }


def _global_series():
    from pyulog import ULog
    log = ULog(str(RTL_LOG))
    data = next(
        m.data for m in log.data_list
        if m.name == "vehicle_global_position")
    n = len(data["timestamp"])
    return [
        {"timestamp": int(data["timestamp"][i]),
         "lat": float(data["lat"][i]), "lon": float(data["lon"][i]),
         "alt": float(data["alt"][i])}
        for i in range(n)]


def test_a4_numeric_reconstruction_matches_command():
    """A4: floor-wins reconstruction from independently recovered
    inputs predicts the logged RTL command within 0.02 m."""
    from flight_log_agent.ulog.inventory import parse_ulog_inventory
    inventory = parse_ulog_inventory(RTL_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})

    return_alt = float(parameters["RTL_RETURN_ALT"])
    cone_ang = float(parameters["RTL_CONE_ANG"])
    acceptance_radius = float(parameters["NAV_ACC_RAD"])
    min_dist = float(parameters["RTL_MIN_DIST"])
    assert cone_ang > 0, "outer cone branch gate requires CONE_ANG > 0"

    frame = _triplet_frame()
    dest_alt = frame["dest_alt"]
    logged_command = frame["logged_command"]
    # Commanded per-sample acceptance radius corroborates the
    # parameter used by the source equation.
    assert abs(frame["acceptance_radius"] - acceptance_radius) <= 1.0
    # Decisive-point distance: entry fix vs destination fix.
    series = _global_series()
    entry = next(s for s in series if s["timestamp"] >= 1937044585)
    dist = _haversine_m(
        entry["lat"], entry["lon"], frame["dest_lat"], frame["dest_lon"])

    # Inner distance-gated candidates must be inactive here.
    assert dist > acceptance_radius
    assert dist > min_dist

    base = dest_alt + return_alt
    floor = dest_alt + 2 * acceptance_radius
    predicted = max(base, floor, entry["alt"])
    print(f"\nA4 base={base:.5f} floor={floor:.5f} "
          f"entry_alt={float(entry['alt']):.5f} dist={dist:.2f}")
    print(f"A4 predicted={predicted:.5f} logged_command={logged_command:.5f}")
    assert abs(predicted - logged_command) <= 0.02

    peak = max(s["alt"] for s in series)
    assert abs(peak - logged_command) <= 0.5, \
        "observed never corroborates the command"


def test_a5_report_validates_and_binds_evidence(_rtl_stage):
    """A5: real report validates; mechanism/evidence bound through
    structures, never blob-presence; honest classification.

    Post Workstream A the report carries retained causal runtime
    refs (rtl.cpp computation candidates) ahead of the declaration
    anchor. Candidate marking below is calibrated to today's
    assumed path: if real freshness later derives the branch, the
    same computation must appear as ordinary surviving evidence
    and this marking pin is updated as legitimate contract
    evolution."""
    from flight_log_agent.analysis.report_validation import (
        validate_report,
    )
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )
    result, _calls, _elapsed = _rtl_stage
    report = result.report
    assert report.ranked_hypotheses, "no hypotheses built"
    top = report.ranked_hypotheses[0]
    print(f"\nA5 mech={top.known_px4_mechanism!r} conf={top.confidence}")
    print(f"A5 confirmed={report.confirmed} "
          f"unconfirmed={report.unconfirmed}")
    assert top.known_px4_mechanism == \
        result.judged.verdict.selected_terminal
    assert top.source_refs, "mechanism without source evidence"
    for ref in top.source_refs:
        assert (SOURCE_ROOT / ref.file).exists(), \
            f"fabricated source reference: {ref.file}"
    assert any(ref.file.endswith("navigator/rtl.h") for ref in top.source_refs), \
        "declaration anchor for _rtl_alt missing"
    computation = [
        ref for ref in top.source_refs
        if ref.file.endswith("navigator/rtl.cpp")
        and ref.start_line in (245, 248)]
    assert {ref.start_line for ref in computation} == {245, 248}, \
        "retained causal runtime refs missing"
    assert all(ref.explanation.startswith(
        "candidate terminal write excluded by assumed feasibility "
        "condition:") for ref in computation), \
        "causal refs must be candidate-marked while assumed"
    helpers = [
        ref for ref in top.source_refs
        if ref.explanation.startswith("upstream helper contribution:")]
    # STOP H topology limitation: the cone helper's only qualifying
    # consumers (rtl.cpp:245 runtime terminal ops) are pruned, so the
    # surviving DAG has no helper→terminal data path. The honest
    # selector output is NO helper representative — never a
    # synthesized one. Absence is a surviving-topology limitation,
    # not evidence the helper did not contribute.
    assert not helpers, \
        f"fabricated helper ref without surviving helper→terminal path: {helpers}"
    # Wording allowlist: every ref must carry a known semantic
    # kind. The helper wording stays allowlisted for future cases
    # with a surviving helper→terminal path; RTL itself must emit
    # none (asserted above).
    assert all(
        ref.explanation.startswith((
            "terminal write:",
            "candidate terminal write excluded by assumed "
            "feasibility condition:",
            "upstream helper contribution:",
        )) for ref in top.source_refs), \
        "unexpected source-ref kind"
    assert top.expected_logged_signature, "no expected log signature"
    inventory = parse_ulog_inventory(RTL_LOG, SOURCE_ROOT)
    observed = set(observed_signals_from_inventory(inventory))
    for item in top.expected_logged_signature:
        assert item.signal in observed, \
            f"nonexistent observed signal: {item.signal}"
    assert hasattr(top, "unresolved_evidence")
    print(f"A5 unresolved={len(top.unresolved_evidence)} "
          f"numeric_checks={len(top.numeric_checks)}")
    validation = validate_report(report)
    assert validation.passed, \
        f"report invalid: {[i.message for i in validation.issues]}"


SIDECAR = REPO_ROOT / "tests/acceptance/rtl_weird_height.json"


def test_a6_sidecar_contract_enforced(_rtl_stage):
    """A6: semantic sidecar enforced against the live stage result
    — fixtures pinned, family selected, numerics re-derived from
    the fixture, forbidden outcomes absent."""
    import json

    result, _calls, _elapsed = _rtl_stage
    assert SIDECAR.exists(), "semantic sidecar missing"
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    for key in ("scenario_id", "question", "inputs", "expected",
                "strength", "unavailable", "forbidden",
                "normalization"):
        assert key in sidecar, f"sidecar missing category: {key}"

    inputs = sidecar["inputs"]
    assert inputs["log"] == str(RTL_LOG.relative_to(REPO_ROOT))
    assert inputs["source_snapshot"] == PINNED_COMMIT
    assert inputs["backend"] == "tree_sitter"

    expected = sidecar["expected"]
    assert expected["mechanism_family"] == \
        "rtl_cone_branch_acceptance_floor_wins"
    assert result.judged.verdict.selected_terminal == "_rtl_alt"

    # Numerics re-derived from the fixture, compared against the
    # sidecar contract (not against themselves).
    frame = _triplet_frame()
    assert abs(frame["dest_alt"] - 83.93812) <= 0.02
    assert abs(expected["predicted_command_m"]
               - frame["logged_command"]) <= expected["tolerance_m"]
    assert abs(frame["logged_command"] - 103.93812) <= 0.02

    report = result.report
    titles = [h.title for h in report.ranked_hypotheses]
    assert any("Mechanism slice for _rtl_alt" in t for t in titles)
    assert "Mechanism slice for _rtl_alt" in \
        report.confirmed + report.unconfirmed
    for hyp in report.ranked_hypotheses:
        if hyp.confidence in ("high", "medium"):
            assert hyp.source_refs and hyp.numeric_checks, \
                "unsupported high-authority claim"
            assert not hyp.applicability.missing_required_signals
        for ref in hyp.source_refs:
            assert (SOURCE_ROOT / ref.file).exists(), \
                f"fabricated source evidence: {ref.file}"
    assert sidecar["strength"] == "PROVEN"
