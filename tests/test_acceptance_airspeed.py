"""Deterministic full-system acceptance: Airspeed load-factor adaptation.

BEST_SUPPORTED case on the landed W3A temporal capability. The real
log carries an above-trim equivalent-airspeed setpoint excursion
best explained by bank/load-factor adaptation of the minimum
airspeed in FixedwingPositionControl: the adapted minimum rises
with setpoint-bank load factor and binds the command. Exact
internal state attribution (slew entry history, per-sample NPFG
branches, full trajectory) is unavailable and DISCRIMINATING by
design; the mechanism-level conclusion does not depend on it.
Strength stays BEST_SUPPORTED: never PROVEN.
"""
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AIRSPEED_LOG = REPO_ROOT / (
    "uploads/8a8aa57fedf94caf833ab8b4ade11d25/airspeed-load-factor.ulg")
SOURCE_ROOT = REPO_ROOT / "ref" / "PX4-Autopilot"
PINNED_COMMIT = "1dacb4cdef2d7145754fc788fa8dc482eed74b40"
AIRSPEED_LOG_MD5 = "2acb5b821e1dffed3edabb805f40dbaa"

AIRSPEED_QUESTION = ("Why did the equivalent airspeed setpoint rise to "
                     "about 23.7 m/s, above the 21 m/s trim value, during "
                     "the ~50-degree banked turn, rather than remaining at "
                     "trim or following a mission, wind, or another "
                     "persistent command?")


def test_a1_airspeed_fixture_loads_and_calibrates():
    """A1: real .ulg present with verified identity; pinned source
    snapshot verified; Airspeed signals/parameters calibrated."""
    if not AIRSPEED_LOG.exists():
        raise AssertionError(
            "Airspeed acceptance log missing: "
            f"{AIRSPEED_LOG} (gitignored fixture; provision the "
            "airspeed-load-factor.ulg at this exact path)")
    digest = hashlib.md5(AIRSPEED_LOG.read_bytes()).hexdigest()
    assert digest == AIRSPEED_LOG_MD5, \
        f"airspeed log identity mismatch: {digest}"
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
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
            "tecs_status[0].equivalent_airspeed_sp",
            "vehicle_attitude_setpoint[0].roll_body",
            "vehicle_status[0].nav_state",
            "vehicle_land_detected[0].landed",
            "position_setpoint_triplet[0].current.cruising_speed"):
        assert signal in observed, f"signal not surfaced: {signal}"

    parameters = dict((inventory or {}).get("parameters") or {})
    assert parameters.get("FW_AIRSPD_TRIM") == 21.0
    assert parameters.get("FW_AIRSPD_MIN") == 19.0
    assert parameters.get("FW_AIRSPD_MAX") == 35.0
    assert parameters.get("FW_WIND_ARSP_SC") == 0.0


def _pinned_git_grep(pattern):
    """Fixed-string repo-wide source search at the pinned snapshot."""
    found = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "grep", "-n", "-F", pattern,
         "--", "*.cpp", "*.h", "*.hpp"],
        capture_output=True, text=True, check=True).stdout.strip()
    return [line for line in found.splitlines() if line.strip()]


def test_a2_publication_writer_sole_and_by_value():
    """A2: tecs_status.equivalent_airspeed_sp has one repository
    writer (FixedwingPositionControl publication, by value); TECS
    internal load-factor handling is a separate consumer, never
    the source of the published field. STOP F decision: grounded."""
    writers = _pinned_git_grep("equivalent_airspeed_sp =")
    assert len(writers) == 1, \
        f"expected a sole EAS publication writer, found: {writers}"
    path, text = writers[0].split(":", 1)[0], writers[0]
    assert "fw_pos_control/FixedwingPositionControl.cpp" in path
    assert "tecs_status.equivalent_airspeed_sp = equivalent_airspeed_sp;" \
        in text

    publish = (SOURCE_ROOT / "src/modules/fw_pos_control"
               / "FixedwingPositionControl.cpp").read_text()
    assert ("FixedwingPositionControl::tecs_status_publish("
            "float alt_sp, float equivalent_airspeed_sp") in publish, \
        "publication must take airspeed_sp by value"
    assert "_tecs.update(" in publish
    assert "tecs_status_publish(alt_sp, airspeed_sp" in publish, \
        "publish uses the caller-local airspeed_sp after the TECS call"
    # The TECS-internal load-factor consumer exists but writes no
    # published EAS field (covered by the sole-writer grep above).
    assert "set_load_factor" in publish
    print("\nA2 sole writer pinned; TECS consumer distinguished")


def _airspeed_eas_series():
    """All tecs_status.equivalent_airspeed_sp samples (raw pyulog).

    Independent ground truth: exact timestamps drive peak
    extraction and roll alignment.
    """
    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    tecs = next(m.data for m in log.data_list if m.name == "tecs_status")
    return sorted(
        (t / 1e6, float(v))
        for t, v in zip(tecs["timestamp"],
                        tecs["equivalent_airspeed_sp"])
    )


def test_a3_peak_extraction():
    """A3: exact peak equivalent_airspeed_sp + timestamp parsed from
    the real fixture (re-derived, never trusted reminders)."""
    series = _airspeed_eas_series()
    assert len(series) > 100, "excursion log must carry EAS history"
    peak_time, peak_eas = max(series, key=lambda item: item[1])
    assert peak_eas > 21.0, "peak must clear the trim value"
    assert round(peak_time * 1e6) == 297960673
    assert peak_eas == 23.69844627380371
    print(f"\nA3 peak EAS={peak_eas!r} @ {peak_time:.3f}s")


def _airspeed_setpoint_roll_series():
    """All vehicle_attitude_setpoint.roll_body samples (raw pyulog).

    This is the exact signal the source mechanism consumes
    (adapt_airspeed_setpoint reads _att_sp.roll_body).
    """
    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    attitude_sp = next(
        m.data for m in log.data_list
        if m.name == "vehicle_attitude_setpoint")
    return sorted(
        (t / 1e6, float(v))
        for t, v in zip(attitude_sp["timestamp"],
                        attitude_sp["roll_body"])
    )


def _airspeed_measured_roll_series():
    """Measured vehicle roll from the attitude quaternion (raw pyulog).

    Corroboration only: never the formula input.
    """
    import math

    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    attitude = next(
        m.data for m in log.data_list if m.name == "vehicle_attitude")
    series = []
    for i in range(len(attitude["timestamp"])):
        w = float(attitude["q[0]"][i])
        x = float(attitude["q[1]"][i])
        y = float(attitude["q[2]"][i])
        z = float(attitude["q[3]"][i])
        roll = math.atan2(2.0 * (w * x + y * z),
                          1.0 - 2.0 * (x * x + y * y))
        series.append((attitude["timestamp"][i] / 1e6, roll))
    return sorted(series)


def _nearest_before_or_at(series, stamp):
    """Nearest sample to stamp (exact timestamp alignment)."""
    return min(series, key=lambda item: abs(item[0] - stamp))


def test_a4_correct_roll_alignment():
    """A4: attitude-SETPOINT roll aligned to the EAS peak; measured
    roll pinned as corroboration that must NOT be substituted.
    If setpoint roll cannot be aligned: STOP B."""
    import math

    peak_time, _peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    roll_time, roll_sp = _nearest_before_or_at(
        _airspeed_setpoint_roll_series(), peak_time)
    assert abs(roll_time - peak_time) <= 0.02, \
        f"setpoint roll must align to peak: dt={roll_time - peak_time}"
    assert roll_sp == 0.8726646304130554
    assert abs(math.degrees(roll_sp) - 50.0) <= 0.05

    _meas_time, roll_meas = _nearest_before_or_at(
        _airspeed_measured_roll_series(), peak_time)
    assert roll_meas == 0.7728081561909079
    assert abs(roll_sp - roll_meas) > 0.05, \
        "setpoint and measured roll must be distinct inputs: " \
        "substituting measured roll would change the mechanism"
    print(f"\nA4 setpoint roll={roll_sp!r} "
          f"({math.degrees(roll_sp):.2f}deg) vs measured "
          f"{math.degrees(roll_meas):.2f}deg")


def test_a5_parameters_and_weight_defaults():
    """A5: logged parameters plus weight-default source semantics.

    Historical FW_WGT_SCA does not exist here: the real weight
    path uses WEIGHT_BASE/GROSS, absent from the log (defaults
    -1.0 fail the epsilon gate), so weight_ratio = 1.0 by source
    semantics, never by hard-coding."""
    from pyulog import ULog

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    assert parameters.get("FW_AIRSPD_TRIM") == 21.0
    assert parameters.get("FW_AIRSPD_MIN") == 19.0
    assert parameters.get("FW_AIRSPD_MAX") == 35.0
    assert parameters.get("FW_WIND_ARSP_SC") == 0.0
    assert parameters.get("FW_GND_SPD_MIN") == 5.0

    log = ULog(str(AIRSPEED_LOG))
    # Logged explicitly at -1.0: the epsilon gate below fails on
    # logged values alone (defaults agree, but are not needed).
    assert log.initial_parameters.get("WEIGHT_BASE") == -1.0
    assert log.initial_parameters.get("WEIGHT_GROSS") == -1.0
    assert log.initial_parameters.get("FW_WGT_SCA") is None

    params_c = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "fw_path_navigation_params.c"
    ).read_text()
    assert "PARAM_DEFINE_FLOAT(WEIGHT_BASE, -1.0f)" in params_c
    assert "PARAM_DEFINE_FLOAT(WEIGHT_GROSS, -1.0f)" in params_c

    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "float weight_ratio = 1.0f;" in control
    assert "_param_weight_base.get() > FLT_EPSILON" in control
    assert "_param_weight_gross.get() > FLT_EPSILON" in control

    absent = subprocess.run(
        ["git", "-C", str(SOURCE_ROOT), "grep", "-F", "FW_WGT_SCA",
         "--", "*.cpp", "*.h", "*.hpp", "*.c"],
        capture_output=True, text=True, check=False)
    assert absent.returncode == 1 and absent.stdout.strip() == "", \
        f"FW_WGT_SCA must not exist in this snapshot: {absent.stdout[:200]}"
    print("\nA5 params pinned; weight_ratio = 1.0 by gate semantics")


def test_a6_load_factor_numeric_reconstruction():
    """A6: FW_AIRSPD_MIN * sqrt(load_factor) == observed peak.

    Expected value built ONLY from independently parsed MIN
    parameter, setpoint roll, and weight-gate semantics; compared
    against the separately parsed published peak. If the formula
    cannot reconstruct the peak: STOP C."""
    import math

    from pyulog import ULog

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    airspd_min = parameters.get("FW_AIRSPD_MIN")
    assert airspd_min == 19.0

    peak_time, peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    _roll_time, roll_sp = _nearest_before_or_at(
        _airspeed_setpoint_roll_series(), peak_time)

    log = ULog(str(AIRSPEED_LOG))
    weight_base = log.initial_parameters.get("WEIGHT_BASE")
    weight_gross = log.initial_parameters.get("WEIGHT_GROSS")
    assert weight_base is not None and weight_gross is not None
    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "float weight_ratio = 1.0f;" in control
    weight_ratio = 1.0
    assert not (weight_base > 1e-6 and weight_gross > 1e-6), \
        "weight gate active: ratio 1.0 assumption invalid (STOP C)"

    load_factor = 1.0 / math.cos(roll_sp)
    expected = airspd_min * math.sqrt(load_factor * weight_ratio)
    error = abs(expected - peak_eas)
    # Tolerance 1e-5: ~5x observed float32 chain noise (~1e-6)
    # yet 1e5x smaller than the rival gaps it must discriminate
    # (trim 21 vs 23.7; measured-roll hypothesis 22.45 vs 23.70).
    assert error <= 1e-5, \
        f"adapted min {expected!r} vs peak {peak_eas!r}: STOP C"
    print(f"\nA6 MIN {airspd_min} * sqrt(1/cos({roll_sp!r})) = "
          f"{expected!r} vs peak {peak_eas!r} (err {error:.2e})")


def _airspeed_cruise_series():
    """All position_setpoint_triplet current cruising_speed samples."""
    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    triplet = next(
        m.data for m in log.data_list
        if m.name == "position_setpoint_triplet")
    return sorted(
        (t / 1e6, float(v))
        for t, v in zip(triplet["timestamp"],
                        triplet["current.cruising_speed"])
    )


def test_a7_constrain_binding():
    """A7: requested/fallback < adapted minimum < maximum, so the
    source constrain() must select the adapted minimum. This is
    the causal binding proof: coincidence alone is insufficient.
    If binding cannot be established: STOP D."""
    import math

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    trim = parameters.get("FW_AIRSPD_TRIM")
    airspd_min = parameters.get("FW_AIRSPD_MIN")
    airspd_max = parameters.get("FW_AIRSPD_MAX")
    assert (trim, airspd_min, airspd_max) == (21.0, 19.0, 35.0)

    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "calibrated_airspeed_setpoint = _param_fw_airspd_trim.get();" \
        in control, "invalid/unset request must fall back to trim"
    assert ("calibrated_airspeed_setpoint = constrain("
            "calibrated_airspeed_setpoint, calibrated_min_airspeed,") \
        in control

    cruise = _airspeed_cruise_series()
    assert cruise, "no triplet cruising_speed samples"
    assert all(value == -1.0 for _t, value in cruise), \
        "mission cruising input must be unset throughout"
    peak_time, peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    _ct, cruise_at_peak = _nearest_before_or_at(cruise, peak_time)
    assert cruise_at_peak == -1.0
    requested = trim

    _roll_time, roll_sp = _nearest_before_or_at(
        _airspeed_setpoint_roll_series(), peak_time)
    adapted_min = airspd_min * math.sqrt(
        (1.0 / math.cos(roll_sp)) * 1.0)
    assert requested < adapted_min < airspd_max, \
        f"binding needs {requested} < {adapted_min} < {airspd_max}: STOP D"
    assert abs(adapted_min - peak_eas) <= 1e-5
    print(f"\nA7 binding: requested {requested} < adapted "
          f"{adapted_min:.6f} < max {airspd_max} (STOP D clear)")


def test_a8_mission_command_rival_excluded():
    """A8: unset cruising input falls back to trim, which cannot
    explain the peak. Rejects Rival B (mission requested ~23.7)
    via source semantics, not via absence of a constant."""
    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "calibrated_airspeed_setpoint <= FLT_EPSILON" in control

    cruise = _airspeed_cruise_series()
    peak_time, peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    _ct, cruise_at_peak = _nearest_before_or_at(cruise, peak_time)
    assert cruise_at_peak == -1.0
    assert cruise_at_peak <= 1e-6, \
        "unset cruising input must hit the trim-fallback gate"

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    trim = dict((inventory or {}).get("parameters") or {})[
        "FW_AIRSPD_TRIM"]
    assert trim == 21.0
    assert peak_eas - trim > 2.0, \
        "mission fallback (trim) must fall well short of the peak"
    print(f"\nA8 mission rival excluded: cruise -1 → trim {trim} "
          f"vs peak {peak_eas:.3f}")


def _airspeed_wind_series():
    """All airspeed_wind estimate samples (raw pyulog)."""
    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    wind = next(
        m.data for m in log.data_list if m.name == "airspeed_wind")
    return sorted(
        (t / 1e6, float(north), float(east))
        for t, north, east in zip(
            wind["timestamp"], wind["windspeed_north"],
            wind["windspeed_east"])
    )


def test_a9_wind_rival_excluded():
    """A9: FW_WIND_ARSP_SC = 0 disables the wind-scaling gate even
    though a healthy wind estimate exists. Wind scaling is
    INACTIVE, not merely unlikely."""
    import math

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    wind_sc = dict((inventory or {}).get("parameters") or {})[
        "FW_WIND_ARSP_SC"]
    assert wind_sc == 0.0

    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "_param_fw_wind_arsp_sc.get() > FLT_EPSILON" in control
    assert not wind_sc > 1e-6, "wind gate must be false: INACTIVE"

    peak_time, _peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    wind = _airspeed_wind_series()
    assert len(wind) > 100, "wind estimate must be present (healthy)"
    _wt, north, east = _nearest_before_or_at(wind, peak_time)
    assert math.isfinite(north) and math.isfinite(east)
    assert math.hypot(north, east) > 0.1, \
        "wind must be nonzero so the exclusion is meaningful"
    print(f"\nA9 wind rival excluded: SC=0 gate off despite "
          f"wind ({north:.2f}, {east:.2f}) m/s")


def test_a10_weight_rival_excluded():
    """A10: weight scaling INACTIVE — logged -1.0 weights fail the
    epsilon gate, so weight_ratio = 1.0 enters the load-factor
    root. Rejects Rival D without mentioning FW_WGT_SCA, which
    does not exist (pinned in A5)."""
    from pyulog import ULog

    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "calibrated_min_airspeed *= sqrtf(" in control
    assert "load_factor_from_bank_angle * weight_ratio" in control

    log = ULog(str(AIRSPEED_LOG))
    base = log.initial_parameters.get("WEIGHT_BASE")
    gross = log.initial_parameters.get("WEIGHT_GROSS")
    assert base == -1.0 and gross == -1.0
    assert not (base > 1e-6 and gross > 1e-6), \
        "weight gate must be false: INACTIVE, ratio exactly 1.0"
    print("\nA10 weight rival excluded: gate off, ratio 1.0")


def _airspeed_mode_at(stamp):
    """(nav_state, landed) nearest the stamp (raw pyulog)."""
    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    status = next(
        m.data for m in log.data_list if m.name == "vehicle_status")
    nav = sorted(
        (t / 1e6, int(v))
        for t, v in zip(status["timestamp"], status["nav_state"]))
    landed_msg = next(
        m.data for m in log.data_list
        if m.name == "vehicle_land_detected")
    landed = sorted(
        (t / 1e6, int(v))
        for t, v in zip(landed_msg["timestamp"], landed_msg["landed"]))
    return (_nearest_before_or_at(nav, stamp),
            _nearest_before_or_at(landed, stamp))


def _airspeed_body_forward_at(stamp):
    """Body-forward horizontal speed at stamp via yaw rotation.

    Yaw-only approximation of the source R-transpose projection;
    asserted with 2x margin over the gate so approximation error
    (pitch/roll coupling ~1-2 m/s at cruise) cannot matter.
    """
    import math

    from pyulog import ULog

    log = ULog(str(AIRSPEED_LOG))
    attitude = next(
        m.data for m in log.data_list if m.name == "vehicle_attitude")
    att = sorted(
        (t / 1e6, float(attitude["q[0]"][i]), float(attitude["q[1]"][i]),
         float(attitude["q[2]"][i]), float(attitude["q[3]"][i]))
        for i, t in enumerate(attitude["timestamp"]))
    _at, w, x, y, z = _nearest_before_or_at(att, stamp)
    yaw = math.atan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    local = next(
        m.data for m in log.data_list
        if m.name == "vehicle_local_position")
    vel = sorted(
        (t / 1e6, float(vx), float(vy))
        for t, vx, vy in zip(
            local["timestamp"], local["vx"], local["vy"]))
    _vt, vx, vy = _nearest_before_or_at(vel, stamp)
    return vx * math.cos(yaw) + vy * math.sin(yaw)


def test_a11_normal_mode_gating():
    """A11: mission/airborne mode at peak; takeoff/landing
    adjusted-min paths excluded; ground-speed undershoot
    inactive on body-forward velocity. STOP E decision follows
    A8-A11 jointly."""
    peak_time, _peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    (nav_time, nav_state), (land_time, landed) = _airspeed_mode_at(
        peak_time)
    assert nav_state == 3, f"peak must be mission mode, got {nav_state}"
    assert abs(nav_time - peak_time) <= 1.0
    assert landed == 0, "peak must be airborne"
    assert abs(land_time - peak_time) <= 1.0

    declaration = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.hpp"
    ).read_text()
    assert "bool in_takeoff_situation = false" in declaration, \
        "mission auto-position call uses the false default"
    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "adjusted_min_airspeed = takeoff_airspeed" in control
    assert "adjusted_min_airspeed = airspeed_land" in control

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    gnd_min = dict((inventory or {}).get("parameters") or {})[
        "FW_GND_SPD_MIN"]
    assert gnd_min == 5.0
    body_forward = _airspeed_body_forward_at(peak_time)
    assert body_forward > 2.0 * gnd_min, \
        f"undershoot gate needs body velocity < min: {body_forward}"
    print(f"\nA11 mode mission/airborne; takeoff/landing paths off; "
          f"body-forward {body_forward:.1f} m/s kills undershoot")


def _airspeed_excursion_condition():
    """Above-trim excursion intent (test data, never production).

    Transition-free questioned condition: the existing W3A
    backward-compatible seam for bounding an excursion span.
    """
    from flight_log_agent.analysis.mechanism_judge import QuestionedCondition

    return QuestionedCondition(
        signal_hint="tecs_status.equivalent_airspeed_sp",
        op=">",
        reference="FW_AIRSPD_TRIM",
        units="signal: m/s; reference: m/s",
        frame="equivalent airspeed",
    )


def test_a12_excursion_window_derived():
    """A12: existing questioned-condition machinery bounds the
    above-trim excursion containing the peak. The window scopes
    fingerprints/rivals/anchors, never writer identity; no new
    production event type."""
    from flight_log_agent.analysis.dag_pipeline import (
        evaluate_questioned_condition_windows,
    )
    from flight_log_agent.analysis.dag_replay import EvaluationScope
    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    from flight_log_agent.ulog.inventory import (
        observed_signals_from_inventory,
    )
    logged = set(observed_signals_from_inventory(inventory))
    parameters = dict((inventory or {}).get("parameters") or {})
    from flight_log_agent.px4.msg_schema import load_px4_signal_policies

    policies = load_px4_signal_policies(SOURCE_ROOT)
    result = evaluate_questioned_condition_windows(
        _airspeed_excursion_condition(),
        candidates=None,
        logged_set=logged,
        log_path=AIRSPEED_LOG,
        parameter_values=parameters,
        signal_policies=policies,
    )
    assert result["windows"], \
        f"excursion window derivation failed: {result.get('error')}"
    scope = EvaluationScope.from_result({"windows": result["windows"]})
    assert scope.windows
    assert scope.error == ""

    peak_time, _peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    assert any(
        start <= peak_time <= end for start, end in scope.windows), \
        "excursion window must contain the peak"
    peak_windows = [
        (start, end) for start, end in scope.windows
        if start <= peak_time <= end
    ]
    assert len(peak_windows) == 1
    peak_start, peak_end = peak_windows[0]
    # Fingerprint scope needs rise coverage before the peak and
    # fall coverage after it (A13 works inside this span only).
    assert peak_start <= peak_time - 0.3
    assert peak_end >= peak_time + 2.0
    print(f"\nA12 excursion windows: "
          f"{[(round(s, 3), round(e, 3)) for s, e in scope.windows]}")


def _airspeed_slew_rate_limit():
    """Pinned ASPD_SP_SLEW_RATE from source (must be the exact
    constant the fall-rate fingerprint is judged against)."""
    header = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.hpp"
    ).read_text()
    assert "static constexpr float ASPD_SP_SLEW_RATE = 1.f;" in header
    return 1.0


def test_a13_trajectory_fingerprints():
    """A13: excursion shape fingerprints support adaptation.

    Peak convergence (state == adapted min), slew-limited fall
    (≈ -rate), forced-up rise (faster than the slew limit
    allows), and converged anchors — all inside the A12 peak
    window, all from logged samples plus pinned source. No
    arbitrary entry-state history is reconstructed: STOP H/I
    stay clear by construction (no new machinery used)."""
    import math

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    slew_rate = _airspeed_slew_rate_limit()
    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    airspd_min = parameters["FW_AIRSPD_MIN"]
    assert airspd_min == 19.0

    series = _airspeed_eas_series()
    peak_time, peak_eas = max(series, key=lambda item: item[1])
    rolls = _airspeed_setpoint_roll_series()

    def adapted_min_at(stamp):
        _rt, roll = _nearest_before_or_at(rolls, stamp)
        return airspd_min * math.sqrt(1.0 / math.cos(roll))

    # 1. Peak convergence: plateau samples equal the adapted input.
    plateau = [(t, v) for t, v in series
               if peak_time <= t <= peak_time + 1.0]
    assert len(plateau) >= 3, "flat peak region must be sampled"
    for t, v in plateau:
        assert abs(v - adapted_min_at(t)) <= 1e-5, \
            f"peak convergence broken @{t}: {v} vs {adapted_min_at(t)}"

    # 2. Regime partition of the fall: plateau extension (flat,
    # at peak), one regime-crossing transient, then exact
    # slew-limited fall. The transient pair is disclosed, not
    # hidden: single-sample input reconstruction there is
    # unavailable (the named DISCRIMINATING NPFG/state gap).
    trim = parameters["FW_AIRSPD_TRIM"]
    regime = []
    for (t1, v1), (t2, v2) in zip(series, series[1:]):
        if (peak_time - 1.0 <= t1 and t2 <= peak_time + 4.0
                and min(v1, v2) > trim + 0.5 and 0 < t2 - t1 < 0.5):
            regime.append((t1, v1, t2, v2, (v2 - v1) / (t2 - t1)))
    assert regime, "no above-trim regime pairs found"
    flat = [item for item in regime if abs(item[4]) <= 0.1]
    for t1, v1, _t2, _v2, _rate in flat:
        assert abs(v1 - peak_eas) <= 1e-5, \
            "flat pairs must extend the converged peak"
    falling = [item for item in regime if item[4] < -0.1]
    assert falling, "no falling pairs found"
    drop = falling[0]
    # Transient pair: within the slew envelope (|rate| <= limit)
    # but not exactly on it — input-path reconstruction for
    # this single sample is the disclosed DISCRIMINATING gap.
    assert -1.05 * slew_rate <= drop[4] <= 0.0
    rest = [item for item in falling[1:]]
    assert len(rest) >= 3, "slew-regime fall must be sampled"
    for _t1, _v1, _t2, _v2, rate in rest:
        assert -1.05 * slew_rate <= rate <= -0.95 * slew_rate, \
            f"post-transient fall rate {rate} must match -slew-rate"
    # 3. Forced-up rise: faster than any slew-limited climb allows
    # (|Δstate| <= rate * Δt always, unless setForcedValue fired).
    rise_pairs = [
        ((v2 - v1) / (t2 - t1))
        for (t1, v1), (t2, v2) in zip(series, series[1:])
        if peak_time - 1.0 <= t1 < peak_time and 0 < t2 - t1 < 0.5
    ]
    assert rise_pairs, "no pre-peak rise pairs found"
    assert max(rise_pairs) > 1.2 * slew_rate, \
        f"rise must exceed the slew limit (forced): {max(rise_pairs)}"

    # 4. Converged anchors: plateau samples double as local
    # acceptance-side anchors (analysis only, never infra).
    anchors = [t for t, v in plateau
               if abs(v - adapted_min_at(t)) <= 1e-5]
    assert len(anchors) >= 3
    print(f"\nA13 fingerprints: plateau x{len(plateau)}, "
          f"fall {rest[0][4]:.3f} m/s/s x{len(rest)} "
          f"(+1 transient disclosed), "
          f"max rise {max(rise_pairs):.3f} m/s/s, "
          f"anchors x{len(anchors)} (STOP H/I clear)")


def test_a14_rival_bundle():
    """A14: all deterministic rival conclusions jointly, with
    residual uncertainty preserved (transient pair, per-sample
    NPFG proof stay DISCRIMINATING — never collapsed into
    proof, never claimed impossible)."""
    import math

    from flight_log_agent.ulog.inventory import parse_ulog_inventory

    inventory = parse_ulog_inventory(AIRSPEED_LOG, SOURCE_ROOT)
    parameters = dict((inventory or {}).get("parameters") or {})
    trim = parameters["FW_AIRSPD_TRIM"]
    airspd_min = parameters["FW_AIRSPD_MIN"]
    airspd_max = parameters["FW_AIRSPD_MAX"]

    peak_time, peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    _roll_time, roll_sp = _nearest_before_or_at(
        _airspeed_setpoint_roll_series(), peak_time)
    adapted_min = airspd_min * math.sqrt(1.0 / math.cos(roll_sp))

    control = (
        SOURCE_ROOT / "src/modules/fw_pos_control"
        / "FixedwingPositionControl.cpp"
    ).read_text()
    assert "1.0f / cosf(_att_sp.roll_body)" in control, \
        "formula input must be setpoint roll (Rival E)"

    # Rival A (fixed trim): adapted min exceeds trim and binds.
    assert trim < adapted_min < airspd_max
    # Rival B (mission 23.7): unset cruise falls back to trim.
    cruise = _airspeed_cruise_series()
    _ct, cruise_at_peak = _nearest_before_or_at(cruise, peak_time)
    assert cruise_at_peak == -1.0
    assert peak_eas - trim > 2.0
    # Rival C (wind): gate off.
    assert parameters["FW_WIND_ARSP_SC"] == 0.0
    assert "_param_fw_wind_arsp_sc.get() > FLT_EPSILON" in control
    # Rival D (weight): gate off, ratio 1.
    assert "float weight_ratio = 1.0f;" in control
    # Rival E (measured bank): distinct, wrong input.
    _mt, roll_meas = _nearest_before_or_at(
        _airspeed_measured_roll_series(), peak_time)
    assert abs(roll_sp - roll_meas) > 0.05
    assert abs(airspd_min * math.sqrt(1.0 / math.cos(roll_meas))
               - peak_eas) > 1.0, \
        "measured roll must NOT reconstruct the peak"
    # Rival F (persistent unrelated 23.7): the value occurs only
    # as one excursion event, never as a standing command.
    series = _airspeed_eas_series()
    near_peak = [t for t, v in series if abs(v - peak_eas) <= 0.5]
    assert near_peak, "peak neighborhood must exist"
    assert min(near_peak) >= peak_time - 2.0
    assert max(near_peak) <= peak_time + 6.0
    print(f"\nA14 rivals A-F decided; "
          f"23.7-valued samples span "
          f"[{min(near_peak):.1f}, {max(near_peak):.1f}] only")


SIDECAR = REPO_ROOT / "tests/acceptance/airspeed_load_factor.json"


def test_a15_sidecar_best_supported_contract():
    """A15: sidecar pins BEST_SUPPORTED (never PROVEN), expected
    evidence against the real log, missing-evidence classes, and
    forbidden claims (RTL/TECS/Takeoff sidecar pattern)."""
    import json

    assert SIDECAR.exists(), "semantic sidecar missing"
    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    for key in ("scenario_id", "question", "inputs", "expected",
                "strength", "unavailable", "forbidden",
                "normalization"):
        assert key in sidecar, f"sidecar missing category: {key}"
    assert sidecar["strength"] == "BEST_SUPPORTED"
    assert "PROVEN" not in json.dumps(sidecar), \
        "BEST_SUPPORTED benchmark must never claim PROVEN"
    assert sidecar["inputs"]["log"] == str(
        AIRSPEED_LOG.relative_to(REPO_ROOT))
    assert sidecar["inputs"]["log_md5"] == AIRSPEED_LOG_MD5
    assert sidecar["inputs"]["source_snapshot"] == PINNED_COMMIT
    assert sidecar["question"] == AIRSPEED_QUESTION

    expected = sidecar["expected"]
    tolerance = float(expected["tolerance"])
    assert tolerance == 1e-5
    peak_time, peak_eas = max(
        _airspeed_eas_series(), key=lambda item: item[1])
    assert abs(peak_time - expected["peak_time_s"]) <= 1e-6
    assert abs(peak_eas - expected["peak_eas_m_s"]) <= tolerance
    _roll_time, roll_sp = _nearest_before_or_at(
        _airspeed_setpoint_roll_series(), peak_time)
    assert abs(roll_sp - expected["attitude_setpoint_roll_rad"]) == 0.0

    unavailable = " ".join(sidecar["unavailable"])
    assert "DISCRIMINATING" in unavailable
    assert "NON_DECISIVE" in unavailable
    assert "CRITICAL" not in unavailable, \
        "no CRITICAL gap may be recorded: STOP G on discovery"
    forbidden = " ".join(sidecar["forbidden"])
    for term in ("proven_strength_claim",
                 "measured_roll_as_formula_input",
                 "fw_wgt_sca_as_active_parameter",
                 "tecs_writes_published_eas"):
        assert term in forbidden, f"forbidden claim missing: {term}"
    print("\nA15 sidecar contract enforced "
          "(BEST_SUPPORTED, pinned, scoped)")


def _airspeed_module_tree():
    """AST of this acceptance module (harness introspection)."""
    import ast

    return ast.parse(Path(__file__).read_text(encoding="utf-8"))


def test_a16_proof_status_isolation():
    """A16: BEST_SUPPORTED never touches live confidence/proof.

    The sidecar carries no confirmed/confidence claims beyond
    the shared unstable-fields list; this module never reads
    pipeline status and interacts with production only through
    the generic seams (inventory, msg schema, questioned
    windows, plural scope)."""
    import ast
    import json

    sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))
    assert "confirmed" not in sidecar
    assert "confidence" not in sidecar
    flattened = json.dumps(sidecar)
    # "confirmed" appears exactly once, inside the eas2tas
    # rationale phrase "source-confirmed" (verified-from-source),
    # never as a pipeline verification-status claim.
    assert flattened.count("confirmed") == 1
    assert "source-confirmed set/get cancellation" in flattened
    assert flattened.count("confidence") == 2
    assert "strength_equals_live_confidence" in flattened
    assert "confidence_values" in flattened

    status_reads = [
        node.attr for node in ast.walk(_airspeed_module_tree())
        if isinstance(node, ast.Attribute)
        and node.attr in ("confirmed", "confidence")
    ]
    assert status_reads == [], \
        f"acceptance must not read pipeline status: {status_reads}"

    allowed = {
        "flight_log_agent.analysis.mechanism_judge":
            {"QuestionedCondition"},
        "flight_log_agent.analysis.dag_pipeline":
            {"evaluate_questioned_condition_windows"},
        "flight_log_agent.analysis.dag_replay": {"EvaluationScope"},
        "flight_log_agent.ulog.inventory": {
            "parse_ulog_inventory", "observed_signals_from_inventory"},
        "flight_log_agent.px4.msg_schema": {"load_px4_signal_policies"},
    }
    forbidden_names = {
        "build_report_from_dag", "replay_terminal_expressions",
        "replay_dag_roots", "discriminate_candidates", "checkpoint",
        "CodeRef", "branches_verified", "PROVEN",
    }
    tree = _airspeed_module_tree()
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
        f"proof/replay/report coupling forbidden: " \
        f"{sorted(used & forbidden_names)}"
    print("\nA16 isolation: BEST_SUPPORTED, generic seams only")


def _run_pytest(*args):
    """Subprocess pytest at the repo root (regression driver)."""
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        capture_output=True, text=True, check=False, cwd=str(REPO_ROOT),
    )


def test_a17_existing_acceptance_regressions():
    """A17: completed benchmark suites still execute/collect.

    Fast suites run to GREEN here; slow stage suites
    (RTL/TECS/Takeoff) prove collection here while their full
    execution is covered by the full-suite regression run.
    Nothing in this module may modify them."""
    fast = _run_pytest(
        "tests/test_temporal_qualification.py",
        "tests/test_no_case_specific_terms.py", "-q",
        "-p", "no:cacheprovider",
    )
    assert fast.returncode == 0, \
        f"temporal/guard regression failed:\n{fast.stdout[-1500:]}"
    slow = _run_pytest(
        "tests/test_acceptance_rtl.py",
        "tests/test_acceptance_tecs.py",
        "tests/test_acceptance_takeoff.py",
        "--collect-only", "-q", "-p", "no:cacheprovider",
    )
    assert slow.returncode == 0, \
        f"acceptance collection broken:\n{slow.stderr[-1500:]}"
    print("\nA17 fast suites GREEN; slow suites collect")


def test_a18_full_suite_still_collects():
    """A18: Airspeed acceptance coexists with the whole suite.

    Full collection must succeed with no test disappearance: at
    least the pre-Airspeed baseline (1519 passed + 31 xfailed)
    plus every test in this module."""
    import ast

    module_tests = sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(_airspeed_module_tree())
    )
    assert module_tests >= 18, \
        f"W4 module must carry A1-A18, found {module_tests}"

    collected = _run_pytest("--collect-only", "-q", "-p",
                            "no:cacheprovider")
    assert collected.returncode == 0, \
        f"suite collection broken:\n{collected.stderr[-2000:]}"
    total = 0
    for line in collected.stdout.splitlines():
        if "tests collected" in line:
            total = int(line.split()[0])
    assert "test_acceptance_airspeed" in collected.stdout
    assert total >= 1519 + 31 + module_tests, \
        f"collected {total}, expected baseline plus {module_tests} W4 tests"
    print(f"\nA18 suite collects: {total} tests "
          f"(baseline 1550 + {module_tests} W4)")
