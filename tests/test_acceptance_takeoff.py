"""Deterministic full-system acceptance: Takeoff minimum-altitude behavior.

W3C case driven by the landed W3A temporal capability. The real log
carries an initial +20 m takeoff target (MIS_TAKEOFF_ALT above home)
followed later by the retained +7 m mission waypoint: intentional
mission-state behavior, not an overshoot or waypoint loss. Exact
internal state attribution beyond the evidence is unavailable and
non-decisive by design; the mechanism-level conclusion does not
depend on it.
"""
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TAKEOFF_LOG = REPO_ROOT / (
    "uploads/takeoff-inconsistent-hgt/inconsistent-takeoff-hgt.ulg")
SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
PINNED_COMMIT = "1dacb4cdef2d7145754fc788fa8dc482eed74b40"
TAKEOFF_LOG_MD5 = "864264eb3ceadc25c5e0b0703788be61"

TAKEOFF_QUESTION = ("Why did the vehicle initially command a climb to "
                    "about 20 m above home rather than the ~7 m mission "
                    "waypoint altitude?")


def test_k1_takeoff_fixture_loads_and_calibrates():
    """K1: real .ulg present with verified identity; pinned source
    snapshot verified; Takeoff/mission signals calibrated."""
    if not TAKEOFF_LOG.exists():
        raise AssertionError(
            "Takeoff acceptance log missing: "
            f"{TAKEOFF_LOG} (gitignored fixture; provision the "
            "inconsistent-takeoff-hgt.ulg at this exact path)")
    digest = hashlib.md5(TAKEOFF_LOG.read_bytes()).hexdigest()
    assert digest == TAKEOFF_LOG_MD5, \
        f"takeoff log identity mismatch: {digest}"
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
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
            "position_setpoint_triplet[0].current.alt",
            "vehicle_status[0].nav_state",
            "home_position[0].alt",
            "navigator_mission_item[0].nav_cmd"):
        assert signal in observed, f"signal not surfaced: {signal}"

    parameters = dict((inventory or {}).get("parameters") or {})
    assert parameters.get("MIS_TAKEOFF_ALT") == 20.0


def _takeoff_transition_intent():
    """Case-specific Window A intent (test data, never production)."""
    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    return TransitionEventSpec(
        transition_signal="vehicle_status.nav_state",
        from_value=2,
        to_value=3,
        event_selection="first",
        relation="after",
        first_sample_of="position_setpoint_triplet.current.alt",
    )


def _takeoff_resume_intent():
    """Case-specific Window B intent (test data, never production)."""
    from flight_log_agent.analysis.mechanism_judge import TransitionEventSpec

    return TransitionEventSpec(
        transition_signal="navigator_mission_item.nav_cmd",
        from_value=22,
        to_value=16,
        event_selection="first",
        relation="after",
        first_sample_of="position_setpoint_triplet.current.alt",
    )


def _takeoff_frame():
    """Phase anchors derived from raw log columns.

    Independent ground truth for production derivation: raw pyulog
    samples, no pipeline structures involved.
    """
    from pyulog import ULog

    log = ULog(str(TAKEOFF_LOG))
    nav = next(m.data for m in log.data_list if m.name == "vehicle_status")
    stamps = [t / 1e6 for t in nav["timestamp"]]
    states = [int(v) for v in nav["nav_state"]]
    mission_start = next(
        stamps[i + 1] for i in range(len(stamps) - 1)
        if states[i] == 2 and states[i + 1] == 3
    )
    triplet = next(m.data for m in log.data_list
                   if m.name == "position_setpoint_triplet")
    valid = sorted(
        (t / 1e6, float(v))
        for t, v, ok, typ in zip(triplet["timestamp"],
                                 triplet["current.alt"],
                                 triplet["current.valid"],
                                 triplet["current.type"])
        if ok and typ == 0
    )
    items = next(m.data for m in log.data_list
                 if m.name == "navigator_mission_item")
    item_seq = sorted(
        (t / 1e6, int(cmd), float(alt))
        for t, cmd, alt in zip(items["timestamp"], items["nav_cmd"],
                               items["altitude"])
    )
    return {
        "mission_start": mission_start,
        "triplet": valid,
        "items": item_seq,
    }


def _derive_window(intent, logged):
    from flight_log_agent.analysis.dag_pipeline import (
        evaluate_transition_windows,
    )

    result = evaluate_transition_windows(
        intent,
        logged_set=logged,
        log_path=TAKEOFF_LOG,
        signal_policies={},
    )
    assert result["windows"] is not None, result.get("error")
    (window,) = result["windows"]
    return window


def test_k2_initial_takeoff_window_derived_from_real_log(tmp_path):
    """K2: W3A derives the initial takeoff-command window from the
    real log, matching the independently parsed frame."""
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
    logged = set(observed_signals_from_inventory(inventory))
    start, end = _derive_window(_takeoff_transition_intent(), logged)
    frame = _takeoff_frame()
    assert start == frame["mission_start"]
    assert end == frame["triplet"][0][0]
    assert abs(frame["triplet"][0][1] - 32.183) <= 1e-3
    print(f"\nK2 window=[{start:.3f}, {end:.3f}] "
          f"alt={frame['triplet'][0][1]:.3f}")


def test_k3_later_resume_window_derived_from_real_log(tmp_path):
    """K3: W3A derives the later mission-resume window; mission item
    command 22 → 16 marks the phase change (seq_current is NOT
    used: it is constant here)."""
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
    logged = set(observed_signals_from_inventory(inventory))
    start, end = _derive_window(_takeoff_resume_intent(), logged)
    frame = _takeoff_frame()
    assert (start, end) == (frame["items"][1][0], frame["triplet"][1][0])
    assert frame["items"][1][1] == 16
    assert abs(frame["triplet"][1][1] - 19.183) <= 1e-3
    print(f"\nK3 window=[{start:.3f}, {end:.3f}] "
          f"alt={frame['triplet'][1][1]:.3f}")


def test_k4_ordered_windows_use_plural_scope(tmp_path):
    """K4: early window precedes later window through the existing
    plural EvaluationScope.windows representation (no phase model).
    Both bounds are derived, not remembered."""
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
    logged = set(observed_signals_from_inventory(inventory))
    window_a = _derive_window(_takeoff_transition_intent(), logged)
    window_b = _derive_window(_takeoff_resume_intent(), logged)
    scope = EvaluationScope.from_result(
        {"windows": [window_a, window_b]})
    assert scope.windows == (tuple(window_a), tuple(window_b))
    assert scope.error == ""
    assert window_a[1] < window_b[0], \
        f"takeoff window {window_a} must precede resume window {window_b}"
    print(f"\nK4 A=[{window_a[0]:.3f}, {window_a[1]:.3f}] "
          f"B=[{window_b[0]:.3f}, {window_b[1]:.3f}]")


def _takeoff_home_samples():
    """All home_position.alt samples (raw pyulog ground truth).

    Independent of production inventory: exact timestamps drive the
    hold-last home-at-command rule.
    """
    from pyulog import ULog

    log = ULog(str(TAKEOFF_LOG))
    home = next(m.data for m in log.data_list if m.name == "home_position")
    return sorted(
        (t / 1e6, float(v))
        for t, v in zip(home["timestamp"], home["alt"])
    )


def _takeoff_home_at_or_before(command_time):
    """Hold-last home sample at/before the command timestamp.

    Latest home sample with time <= command time; the command
    uses the home estimate available when it was issued.
    """
    candidates = [
        (t, alt) for t, alt in _takeoff_home_samples() if t <= command_time
    ]
    assert candidates, \
        f"no home sample at/before command time {command_time}"
    return max(candidates)


def test_k5_initial_target_is_home_plus_takeoff_alt():
    """K5: initial commanded target ≈ home-at-command + MIS_TAKEOFF_ALT.

    Benchmark-side numeric evidence only (acceptance, never generic
    temporal selection): the fmaxf takeoff rule output pinned
    against logged values with log-quantization tolerance.
    """
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    mis_takeoff_alt = parameters.get("MIS_TAKEOFF_ALT")
    assert mis_takeoff_alt == 20.0

    frame = _takeoff_frame()
    command_time, command_alt = frame["triplet"][0]
    assert round(command_time * 1e6) == 791476832

    home_time, home_alt = _takeoff_home_at_or_before(command_time)
    assert round(home_time * 1e6) == 682506126
    assert abs(home_alt - 12.183) <= 1e-6
    # Hold-last rule: no newer home sample before the command.
    assert all(
        t <= home_time or t > command_time
        for t, _alt in _takeoff_home_samples()
    )

    predicted = home_alt + mis_takeoff_alt
    assert abs(predicted - command_alt) <= 1e-3, \
        f"home {home_alt} + MIS_TAKEOFF_ALT {mis_takeoff_alt} = " \
        f"{predicted} vs commanded {command_alt}"
    print(f"\nK5 home={home_alt:.6f} + {mis_takeoff_alt} = "
          f"{predicted:.6f} vs commanded {command_alt:.6f}")


def _takeoff_logged_items():
    """All navigator_mission_item samples (raw pyulog ground truth).

    Returns (time_s, nav_cmd, altitude, altitude_is_relative,
    sequence_current) in ascending time order.
    """
    from pyulog import ULog

    log = ULog(str(TAKEOFF_LOG))
    items = next(
        m.data for m in log.data_list if m.name == "navigator_mission_item")
    return sorted(
        (t / 1e6, int(cmd), float(alt), bool(rel), int(seq))
        for t, cmd, alt, rel, seq in zip(
            items["timestamp"], items["nav_cmd"], items["altitude"],
            items["altitude_is_relative"], items["sequence_current"])
    )


def test_k6_later_target_is_home_plus_waypoint_alt():
    """K6: later commanded target ≈ home-at-command + mission relative alt.

    The later item is a relative WAYPOINT (altitude_is_relative);
    the relative-to-home conversion rule is confirmed from pinned
    source (mission_block.cpp), not assumed. Log quantization
    tolerance throughout.
    """
    conversion = (
        SOURCE_ROOT / "src/modules/navigator/mission_block.cpp"
    ).read_text()
    assert "if (mission_item.altitude_is_relative)" in conversion
    assert ("return mission_item.altitude + "
            "_navigator->get_home_position()->alt;") in conversion

    frame = _takeoff_frame()
    later_time, later_alt = frame["triplet"][1]
    assert round(later_time * 1e6) == 805313516

    logged_items = _takeoff_logged_items()
    assert len(logged_items) == 2
    item_time, nav_cmd, item_alt, is_relative, _seq = logged_items[1]
    assert round(item_time * 1e6) == 805313512
    assert nav_cmd == 16, \
        f"later item must be a normal WAYPOINT (16), got {nav_cmd}"
    assert is_relative, \
        "later item must carry altitude_is_relative for home conversion"
    assert item_alt == 7.0

    home_time, home_alt = _takeoff_home_at_or_before(later_time)
    assert round(home_time * 1e6) == 682506126

    predicted = home_alt + item_alt
    assert abs(predicted - later_alt) <= 1e-3, \
        f"home {home_alt} + waypoint {item_alt} = {predicted} " \
        f"vs commanded {later_alt}"
    print(f"\nK6 home={home_alt:.6f} + waypoint {item_alt} = "
          f"{predicted:.6f} vs commanded {later_alt:.6f}")


def test_k7_takeoff_altitude_mechanism_pinned_in_source():
    """K7: minimum takeoff-altitude computation grounded in pinned source.

    Anchors use file + symbol + containing content (never exact line
    equality): MIS_TAKEOFF_ALT parameter definition, the
    get_takeoff_min_alt binding, calculate_takeoff_altitude with the
    fmaxf minimum-altitude rule over both landed and in-air
    branches, and the vertical-takeoff gate. Snapshot identity is
    pinned by K1; these assertions re-pin the mechanism sites.
    """
    params = (
        SOURCE_ROOT / "src/modules/navigator/mission_params.c"
    ).read_text()
    assert "PARAM_DEFINE_FLOAT(MIS_TAKEOFF_ALT" in params

    navigator_h = (
        SOURCE_ROOT / "src/modules/navigator/navigator.h"
    ).read_text()
    assert "get_takeoff_min_alt()" in navigator_h
    assert "_param_mis_takeoff_alt" in navigator_h

    mission = (
        SOURCE_ROOT / "src/modules/navigator/mission.cpp"
    ).read_text()
    assert "calculate_takeoff_altitude(" in mission
    assert "get_takeoff_min_alt()" in mission
    assert "fmaxf(takeoff_alt," in mission
    # Landed branch uses live global altitude; in-air uses home.
    assert ("get_global_position()->alt + "
            "_navigator->get_takeoff_min_alt()") in mission
    assert ("get_home_position()->alt + "
            "_navigator->get_takeoff_min_alt()") in mission
    # Vertical-takeoff gate and takeoff-item rewrite sites exist.
    assert "do_need_vertical_takeoff()" in mission
    assert "_mission_item.nav_cmd = NAV_CMD_TAKEOFF;" in mission
    print("\nK7 takeoff-altitude mechanism pinned "
          "(param + binding + fmaxf rule + gate + rewrite)")


def test_k8_waypoint_retention_and_resume_grounded():
    """K8: retained/resumed +7 m waypoint claim, honestly scoped.

    Three deterministic legs: source preservation of the original
    item, logged item continuity (TAKEOFF 22 -> relative WAYPOINT
    16), and later target continuity (K6 relation). The pre-rewrite
    mission-item byte identity is unlogged and NOT claimed, so this
    is MODERATE identity evidence; per spec the observed-command
    claim plus source retention logic carries the mechanism, and
    the frozen benchmark is not weakened. seq_current is constant
    here and MUST NOT be cited as retention evidence (pinned).
    """
    mission = (
        SOURCE_ROOT / "src/modules/navigator/mission.cpp"
    ).read_text()
    # Leg 1: source preserves the original item for later resume.
    assert "mission_item_next_position = _mission_item;" in mission
    assert ("mission_item_next_position.nav_cmd = NAV_CMD_WAYPOINT;"
            in mission)
    # Resume path: retained item flows back into the triplet, plus
    # the just-did-takeoff waypoint-conversion logic.
    assert ("mission_item_to_position_setpoint(mission_item_next_position,"
            in mission)
    assert "if we just did a normal takeoff" in mission

    # Leg 2: logged item continuity (ordered, exact command ids).
    logged_items = _takeoff_logged_items()
    assert len(logged_items) == 2
    (first_time, first_cmd, first_alt, first_rel, _s0) = logged_items[0]
    (later_time, later_cmd, later_alt, later_rel, _s1) = logged_items[1]
    assert round(first_time * 1e6) == 791476669
    assert first_cmd == 22, \
        f"initial item must be TAKEOFF (22), got {first_cmd}"
    assert first_rel is False
    assert abs(first_alt - 32.183) <= 1e-3
    assert round(later_time * 1e6) == 805313512
    assert later_cmd == 16, \
        f"later item must be WAYPOINT (16), got {later_cmd}"
    assert later_rel is True
    assert later_alt == 7.0
    assert first_time < later_time

    # seq_current carries no retention information here: pinned
    # constant so no future reader may cite it as evidence.
    assert {item[4] for item in logged_items} == {0}, \
        "seq_current must stay constant (NOT retention evidence)"

    # Leg 3: later target continuity (same relation K6 pins).
    frame = _takeoff_frame()
    _later_triplet_time, later_triplet_alt = frame["triplet"][1]
    home_time, home_alt = _takeoff_home_at_or_before(later_time)
    assert round(home_time * 1e6) == 682506126
    assert abs((home_alt + later_alt) - later_triplet_alt) <= 1e-3
    print(f"\nK8 retained waypoint: TAKEOFF@{first_time:.3f} -> "
          f"WAYPOINT@{later_time:.3f} (rel {later_alt}) -> "
          f"triplet {later_triplet_alt:.3f} (MODERATE identity, "
          "mechanism carried)")


def _takeoff_logged_text():
    """Navigator/commander logged text (raw pyulog, corroboration).

    Returns (time_s, message) in ascending time order.
    """
    from pyulog import ULog

    log = ULog(str(TAKEOFF_LOG))
    return sorted(
        (message.timestamp / 1e6, str(message.message))
        for message in log.logged_messages
    )


def _takeoff_candidate_inventory(window_a, window_b):
    """Per-phase candidate-domain record (acceptance data, not production).

    Takeoff-phase vs resume-phase mechanisms with their logged
    evidence timestamps and window overlap. Replay/branch domains
    are N/A: no Takeoff DAG stage is built (spec: replay optional;
    ordered windows + source/log evidence suffice).
    """
    def overlaps(window, stamp):
        return window[0] <= stamp <= window[1]

    return [
        {
            "phase": "takeoff",
            "operation": "calculate_takeoff_altitude + fmaxf rule + "
                         "takeoff-item rewrite (mission.cpp)",
            "role": "value-producing initial +20 m target",
            "markers": {
                "nav_transition": 791408357,
                "takeoff_text": 791475648,
                "takeoff_item": 791476669,
                "takeoff_triplet": 791476832,
            },
            "replay_domain": None,
            "branch_windows": None,
            "window_a_overlap": all(
                overlaps(window_a, stamp / 1e6) for stamp in (
                    791408357, 791475648, 791476669, 791476832)),
            "window_b_overlap": any(
                overlaps(window_b, stamp / 1e6) for stamp in (
                    791408357, 791475648, 791476669, 791476832)),
        },
        {
            "phase": "resume",
            "operation": "mission_item_to_position_setpoint(retained "
                         "item) + takeoff-done resume (mission.cpp)",
            "role": "value-producing later +7 m target",
            "markers": {
                "waypoint_item": 805313512,
                "waypoint_triplet": 805313516,
            },
            "replay_domain": None,
            "branch_windows": None,
            "window_a_overlap": any(
                overlaps(window_a, stamp / 1e6) for stamp in (
                    805313512, 805313516)),
            "window_b_overlap": all(
                overlaps(window_b, stamp / 1e6) for stamp in (
                    805313512, 805313516)),
        },
    ]


def test_k9_ordered_windows_discriminate_phases():
    """K9: takeoff vs resume mechanisms separate across ordered windows.

    Each window admits a different single phase (takeoff markers in
    A only, resume markers in B only); A strictly precedes B with
    disjoint bounds. No source-order tiebreak: derivation order is
    permuted and must give identical windows. Replay/branch
    domains are honestly N/A (no Takeoff DAG stage); the
    discrimination rests on ordered logged windows plus distinct
    source roles (ADR-0004 observable-temporal-ordering plus
    ordered-window phase separation). STOP F/G do not fire: the
    mechanism-different phases ARE temporally distinguished, and
    no equivalence is invoked.
    """
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
        parse_ulog_inventory,
    )

    inventory = parse_ulog_inventory(TAKEOFF_LOG, SOURCE_ROOT)
    logged = set(observed_signals_from_inventory(inventory))
    # Permuted derivation order: B first, then A (T11 analog).
    window_b = _derive_window(_takeoff_resume_intent(), logged)
    window_a = _derive_window(_takeoff_transition_intent(), logged)
    window_a2 = _derive_window(_takeoff_transition_intent(), logged)
    window_b2 = _derive_window(_takeoff_resume_intent(), logged)
    assert window_a == window_a2 and window_b == window_b2

    scope = EvaluationScope.from_result(
        {"windows": [window_a, window_b]})
    assert scope.windows == (tuple(window_a), tuple(window_b))
    assert window_a[1] < window_b[0]
    assert not (window_a[0] <= window_b[1] and window_b[0] <= window_a[1]), \
        "ordered windows must be disjoint"

    candidates = _takeoff_candidate_inventory(window_a, window_b)
    assert len(candidates) == 2
    takeoff, resume = candidates
    assert takeoff["window_a_overlap"] and not takeoff["window_b_overlap"]
    assert resume["window_b_overlap"] and not resume["window_a_overlap"]
    assert takeoff["operation"] != resume["operation"], \
        "phases must be materially different mechanisms"
    assert takeoff["replay_domain"] is None
    assert resume["replay_domain"] is None
    print(f"\nK9 takeoff@{takeoff['markers']['takeoff_triplet']}us in A "
          f"only; resume@{resume['markers']['waypoint_triplet']}us in B "
          "only; replay N/A")


def test_k10_takeoff_text_corroborates_command():
    """K10: logged takeoff text present as observation corroboration only.

    Never a CodeRef, never proof authority: the mechanism stays
    supported by K5-K9 if text is ignored.
    """
    text = _takeoff_logged_text()
    takeoff_lines = [
        (t, message) for t, message in text
        if "Takeoff to 20.0 meters above home" in message
    ]
    assert len(takeoff_lines) == 1
    takeoff_time, _message = takeoff_lines[0]
    assert round(takeoff_time * 1e6) == 791475648

    executing = [
        (t, message) for t, message in text
        if "Executing Mission" in message
    ]
    assert len(executing) == 1
    executing_time, _exec_message = executing[0]
    assert round(executing_time * 1e6) == 791475483
    assert executing_time < takeoff_time

    frame = _takeoff_frame()
    assert executing_time < takeoff_time < frame["triplet"][0][0], \
        "mission-start text must precede the commanded target"
    print(f"\nK10 text @{takeoff_time:.3f} corroborates "
          f"(executing @{executing_time:.3f})")


def _takeoff_global_altitude_max(start_s, end_s):
    """Max vehicle_global_position.alt in [start, end] with its time."""
    from pyulog import ULog

    log = ULog(str(TAKEOFF_LOG))
    global_pos = next(
        m.data for m in log.data_list if m.name == "vehicle_global_position")
    best = None
    for timestamp, alt in zip(global_pos["timestamp"], global_pos["alt"]):
        stamp = timestamp / 1e6
        if start_s <= stamp <= end_s and (
                best is None or float(alt) > best[1]):
            best = (stamp, float(alt))
    assert best is not None, f"no global altitude in [{start_s}, {end_s}]"
    return best


def test_k11_rival_hypotheses_excluded():
    """K11: deterministic rivals A/B/C excluded where evidence permits.

    A (waypoint itself requested +20): the logged waypoint asks
    7.0 m relative, and only the takeoff fmaxf rule yields 32.183.
    B (+7 waypoint overwritten/lost): the later 19.183 command is
    consistent with the retained item (K8) plus source retention.
    C (+20 uncontrolled overshoot): the 32.183 target is commanded
    first (K5 + text), while observed tracking peaks lower later
    (19.289 above home) — command precedes achievement.
    """
    logged_items = _takeoff_logged_items()
    assert logged_items[1][1] == 16 and logged_items[1][2] == 7.0, \
        "rival A needs the waypoint itself at +20; it logs +7"
    frame = _takeoff_frame()
    assert abs(frame["triplet"][0][1] - 32.183) <= 1e-3
    assert abs(frame["triplet"][1][1] - 19.183) <= 1e-3
    assert frame["triplet"][0][1] - frame["triplet"][1][1] > 10.0, \
        "rivals A/B need the two commands to coincide; they differ by ~13 m"

    peak_time, peak_alt = _takeoff_global_altitude_max(785.0, 815.0)
    assert round(peak_time * 1e6) == 805815086
    assert abs(peak_alt - 31.472) <= 1e-3
    _home_time, home_alt = _takeoff_home_at_or_before(
        frame["triplet"][0][0])
    achieved_above_home = peak_alt - home_alt
    assert abs(achieved_above_home - 19.289) <= 1e-3
    assert frame["triplet"][0][0] < peak_time, \
        "rival C needs achievement without prior command; the +20 m " \
        "command precedes the observed peak"
    assert achieved_above_home < frame["triplet"][0][1] - home_alt, \
        "observed tracking stays below the commanded target"
    print(f"\nK11 rivals excluded: waypoint={logged_items[1][2]}m, "
          f"commands differ by "
          f"{frame['triplet'][0][1] - frame['triplet'][1][1]:.3f}m, "
          f"achieved {achieved_above_home:.3f}m above home after command")


SIDECAR = REPO_ROOT / "tests/acceptance/takeoff_minimum_altitude.json"


def test_k12_sidecar_benchmark_contract():
    """K12: sidecar preserves PROVEN strength, pins expected evidence
    against the real log, classifies unavailable evidence, and
    forbids overclaims (TECS T19 pattern, Takeoff values)."""
    import json

    assert SIDECAR.exists(), "semantic sidecar missing"
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    for key in ("scenario_id", "question", "inputs", "expected",
                "strength", "unavailable", "forbidden",
                "normalization"):
        assert key in sidecar, f"sidecar missing category: {key}"
    assert sidecar["strength"] == "PROVEN"
    assert sidecar["inputs"]["log"] == str(TAKEOFF_LOG.relative_to(REPO_ROOT))
    assert sidecar["inputs"]["log_md5"] == TAKEOFF_LOG_MD5
    assert sidecar["inputs"]["source_snapshot"] == PINNED_COMMIT
    assert sidecar["question"] == TAKEOFF_QUESTION

    expected = sidecar["expected"]
    tolerance = float(expected["tolerance"])
    frame = _takeoff_frame()
    assert abs(frame["triplet"][0][1]
               - expected["initial_commanded_target_m"]) <= tolerance
    assert abs(frame["triplet"][1][1]
               - expected["later_commanded_target_m"]) <= tolerance
    assert abs(frame["triplet"][0][0]
               - expected["window_a_s"][1]) <= tolerance
    assert abs(frame["triplet"][1][0]
               - expected["window_b_s"][1]) <= tolerance

    unavailable = " ".join(sidecar["unavailable"])
    assert "NON_DECISIVE" in unavailable
    assert "byte identity" in unavailable
    forbidden = " ".join(sidecar["forbidden"])
    assert "seq_current_as_retention_evidence" in forbidden
    assert "execution_attribution_claim" in forbidden
    assert "confidence_upgrade_from_temporal_selection" in forbidden
    print("\nK12 sidecar contract enforced (PROVEN, pinned, scoped)")


def _takeoff_module_tree():
    """AST of this acceptance module (harness introspection, not production)."""
    import ast

    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def test_k13_no_live_status_upgrade():
    """K13: temporal acceptance never forces confirmed/high confidence.

    Benchmark strength (sidecar PROVEN) stays separate from pipeline
    verification status: the sidecar carries no confirmed/confidence
    claims beyond the shared unstable-fields list, and this module
    never reads pipeline status fields.
    """
    import ast
    import json

    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    assert "confirmed" not in sidecar
    assert "confidence" not in sidecar
    flattened = json.dumps(sidecar)
    assert "confirmed" not in flattened
    # "confidence" appears only in the forbidden-upgrade entry and
    # the shared unstable-fields list (no status claim).
    assert flattened.count("confidence") == 2
    assert "confidence_upgrade_from_temporal_selection" in flattened
    assert "confidence_values" in flattened

    status_reads = [
        node.attr for node in ast.walk(_takeoff_module_tree())
        if isinstance(node, ast.Attribute)
        and node.attr in ("confirmed", "confidence")
    ]
    assert status_reads == [], \
        f"acceptance must not read pipeline status: {status_reads}"
    print("\nK13 no forced confirmation (sidecar + module)")


def test_k14_proof_source_replay_isolation():
    """K14: no proof/source/replay semantic changes from W3C acceptance.

    This module interacts with production only through the generic
    W3A seams (transition-window derivation, plural scope,
    inventory parsing); report/replay/checkpoint/proof internals
    are never imported or touched.
    """
    import ast

    allowed = {
        "flight_log_agent.analysis.mechanism_judge": {"TransitionEventSpec"},
        "flight_log_agent.analysis.dag_pipeline": {"evaluate_transition_windows"},
        "flight_log_agent.analysis.dag_replay": {"EvaluationScope"},
        "flight_log_agent.ulog.inventory": {
            "parse_ulog_inventory", "observed_signals_from_inventory"},
    }
    forbidden_names = {
        "build_report_from_dag", "replay_terminal_expressions",
        "replay_dag_roots", "discriminate_candidates", "checkpoint",
        "CodeRef", "branches_verified",
    }
    tree = _takeoff_module_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and str(
                node.module or "").startswith("flight_log_agent"):
            module = str(node.module)
            assert module in allowed, \
                f"unexpected production seam: {module}"
            names = {alias.name for alias in node.names}
            assert names <= allowed[module], \
                f"unexpected names from {module}: {sorted(names)}"
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
    assert not (used & forbidden_names), \
        f"proof/replay/report coupling forbidden: {sorted(used & forbidden_names)}"
    print("\nK14 isolation: generic W3A seams only")


def test_k15_full_suite_still_collects():
    """K15: W3C acceptance coexists with the whole suite.

    Full collection must succeed with no test disappearance: at
    least the pre-W3C baseline (1504 passed + 31 xfailed) plus
    every test in this module. Execution-level regression for
    temporal/TECS/RTL/proof scopes is recorded alongside, not
    inside, this fast guard.
    """
    import ast
    import subprocess
    import sys

    module_tests = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_takeoff_module_tree())
    )
    assert module_tests >= 15, \
        f"W3C module must carry K1-K15, found {module_tests}"

    collected = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
    )
    assert collected.returncode == 0, \
        f"suite collection broken:\n{collected.stderr[-2000:]}"
    total = 0
    for line in collected.stdout.splitlines():
        if "test collected" in line:
            total = int(line.split()[0])
        if "tests collected" in line:
            total = int(line.split()[0])
    assert "test_acceptance_takeoff" in collected.stdout
    assert total >= 1504 + 31 + module_tests, \
        f"collected {total}, expected baseline plus {module_tests} W3C tests"
    print(f"\nK15 suite collects: {total} tests "
          f"(baseline 1535 + {module_tests} W3C)")
