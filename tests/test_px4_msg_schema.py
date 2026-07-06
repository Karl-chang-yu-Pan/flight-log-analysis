from flight_log_agent.px4.msg_schema import (
    derive_signal_policy,
    is_valid_topic_field,
    load_px4_msg_enum_registry,
    load_px4_msg_schema,
    load_px4_signal_policies,
    normalize_px4_enum_value,
    resolve_topic_field,
    signal_policy_for,
)
from flight_log_agent.px4.source_snapshot import SourceRepository

import subprocess


def test_px4_msg_schema_expands_nested_topics():
    schema = load_px4_msg_schema()

    assert "position_setpoint_triplet" in schema
    assert "current.alt" in schema["position_setpoint_triplet"]
    assert "previous.valid" in schema["position_setpoint_triplet"]
    assert "vehicle_local_position" in schema
    assert "z" in schema["vehicle_local_position"]


def test_px4_msg_schema_validates_and_resolves_nested_field_suffixes():
    assert is_valid_topic_field("position_setpoint_triplet.current.alt")
    assert resolve_topic_field("position_setpoint_triplet", "current.alt") == "position_setpoint_triplet.current.alt"
    assert resolve_topic_field("position_setpoint_triplet", "alt") is None


def test_px4_msg_enum_registry_maps_constants_to_matching_fields(tmp_path):
    msg_dir = tmp_path / "PX4-Autopilot" / "msg"
    msg_dir.mkdir(parents=True)
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 vehicle_type
uint8 VEHICLE_TYPE_UNKNOWN = 0
uint8 VEHICLE_TYPE_ROTARY_WING = 1
uint8 VEHICLE_TYPE_FIXED_WING = 2
""",
        encoding="utf-8",
    )
    (msg_dir / "VtolVehicleStatus.msg").write_text(
        """
uint8 VEHICLE_VTOL_STATE_UNDEFINED = 0
uint8 VEHICLE_VTOL_STATE_TRANSITION_TO_FW = 1
uint8 VEHICLE_VTOL_STATE_TRANSITION_TO_MC = 2
uint8 VEHICLE_VTOL_STATE_MC = 3
uint8 VEHICLE_VTOL_STATE_FW = 4

uint64 timestamp
uint8 vehicle_vtol_state # current state of the vtol, see VEHICLE_VTOL_STATE
""",
        encoding="utf-8",
    )

    registry = load_px4_msg_enum_registry(tmp_path / "PX4-Autopilot")

    vehicle_type = registry["vehicle_status.vehicle_type"]
    assert vehicle_type["constants"]["VEHICLE_TYPE_ROTARY_WING"] == 1
    assert vehicle_type["aliases"]["VEHICLE_TYPE_ROTARY_WING"] == 1
    assert vehicle_type["aliases"]["ROTARY_WING"] == 1
    assert normalize_px4_enum_value(
        "vehicle_status.vehicle_type",
        "rotary_wing",
        tmp_path / "PX4-Autopilot",
    ) == 1
    assert normalize_px4_enum_value(
        "vehicle_status.vehicle_type",
        "vehicle_status_s::VEHICLE_TYPE_ROTARY_WING",
        tmp_path / "PX4-Autopilot",
    ) == 1

    vtol_state = registry["vtol_vehicle_status.vehicle_vtol_state"]
    assert vtol_state["constants"]["VEHICLE_VTOL_STATE_FW"] == 4
    assert vtol_state["aliases"]["VEHICLE_VTOL_STATE_FW"] == 4
    assert vtol_state["aliases"]["FW"] == 4
    assert vtol_state["aliases"]["TRANSITION_TO_FW"] == 1


def test_px4_msg_enum_registry_keeps_ambiguous_short_aliases_out(tmp_path):
    msg_dir = tmp_path / "PX4-Autopilot" / "msg"
    msg_dir.mkdir(parents=True)
    (msg_dir / "ModeStatus.msg").write_text(
        """
uint64 timestamp
uint8 mode_state
uint8 MODE_STATE_FOO_BAR = 1
uint8 MODE_STATE_BAZ_BAR = 2
""",
        encoding="utf-8",
    )

    registry = load_px4_msg_enum_registry(tmp_path / "PX4-Autopilot")
    aliases = registry["mode_status.mode_state"]["aliases"]

    assert aliases["FOO_BAR"] == 1
    assert aliases["BAZ_BAR"] == 2
    assert "BAR" not in aliases


def test_px4_msg_schema_and_enums_are_keyed_by_snapshot_commit(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    msg_dir = source_path / "msg"
    msg_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=source_path, check=True, capture_output=True)
    message = msg_dir / "ModeStatus.msg"
    message.write_text("uint8 old_state\nuint8 OLD_STATE_ACTIVE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "old"],
        cwd=source_path,
        check=True,
        capture_output=True,
    )
    old_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_path, check=True, capture_output=True, text=True
    ).stdout.strip()
    message.write_text("uint8 new_state\nuint8 NEW_STATE_ACTIVE = 2\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=source_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "new"],
        cwd=source_path,
        check=True,
        capture_output=True,
    )
    new_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_path, check=True, capture_output=True, text=True
    ).stdout.strip()

    repository = SourceRepository(source_path)
    old = repository.resolve_snapshot(old_sha)
    new = repository.resolve_snapshot(new_sha)

    assert load_px4_msg_schema(old)["mode_status"] == ["old_state"]
    assert load_px4_msg_schema(new)["mode_status"] == ["new_state"]
    assert normalize_px4_enum_value("mode_status.old_state", "OLD_STATE_ACTIVE", old) == 1
    assert normalize_px4_enum_value("mode_status.new_state", "NEW_STATE_ACTIVE", new) == 2


# ---------------------------------------------------------------------------
# Per-signal resampling policy — derive_signal_policy is the taxonomy.
# ---------------------------------------------------------------------------


class TestDeriveSignalPolicy:
    def test_bool_is_discrete_hold(self):
        assert derive_signal_policy("bool", "xy_valid", "true if x and y are valid").method == "discrete_hold"

    def test_enum_field_is_discrete_hold(self):
        assert derive_signal_policy("uint8", "vehicle_type", "see VEHICLE_TYPE", is_enum=True).method == "discrete_hold"

    def test_counter_by_name_is_discrete_hold(self):
        assert derive_signal_policy("uint8", "xy_reset_counter", "").method == "discrete_hold"

    def test_bitfield_by_name_is_discrete_hold(self):
        # Not caught by the enum registry (name has an extra _bitfield suffix
        # the constants don't share) but the name shape backstops it.
        assert derive_signal_policy("uint8", "dist_bottom_sensor_bitfield", "bitfield").method == "discrete_hold"

    def test_plain_integer_defaults_to_discrete_hold(self):
        assert derive_signal_policy("uint32", "some_count_of_things", "").method == "discrete_hold"

    def test_quaternion_is_slerp(self):
        p = derive_signal_policy("float32", "q", "Quaternion rotation", array_size=4)
        assert p.method == "quaternion_slerp"

    def test_scalar_named_q_is_not_slerp(self):
        # Only float32[4] is a quaternion; a scalar named q is not.
        assert derive_signal_policy("float32", "q", "", array_size=None).method != "quaternion_slerp"

    def test_reset_delta_quaternion_is_hold_not_slerp(self):
        # delta_q_reset is an impulse (~0 except at reset), not an orientation.
        p = derive_signal_policy("float32", "delta_q_reset", "Amount by which quaternion has changed during last reset", array_size=4)
        assert p.method == "discrete_hold"

    def test_continuous_scalar_with_unit_is_linear(self):
        assert derive_signal_policy("float32", "vx", "North velocity (metres/sec)").method == "linear"
        assert derive_signal_policy("float32", "x", "North position [m]").method == "linear"

    def test_absolute_angle_bracketed_unit_wraps(self):
        p = derive_signal_policy("float32", "heading", "Euler yaw angle ... -PI..+PI,  (radians)")
        assert p.method == "angle_wrap"
        assert p.unit == "rad"

    def test_absolute_angle_bare_prose_unit_wraps(self):
        # Many setpoint yaw fields write the unit as prose, not [rad].
        assert derive_signal_policy("float32", "yaw", "in radians NED -PI..+PI").method == "angle_wrap"

    # --- traps ---------------------------------------------------------

    def test_angular_rate_is_linear_not_wrapped(self):
        # rad/s is a rate: continuous, non-cyclic. Must not wrap.
        assert derive_signal_policy("float32", "yawspeed", "angular rate [rad/s]").method == "linear"
        assert derive_signal_policy("float32", "ang_accel", "angular acceleration in rad/s^2").method == "linear"

    def test_counter_wrap_comment_is_not_angle(self):
        # "allow to wrap if count exceeds 255" is mod-256 rollover on a uint8,
        # NOT angular wrap.
        p = derive_signal_policy("uint8", "reset_count_quat", "number of quaternion reset events (allow to wrap if count exceeds 255)")
        assert p.method == "discrete_hold"

    def test_geodetic_degrees_are_linear_not_wrapped(self):
        # lat/lon in degrees are absolute geodetic, not +/-pi cyclic.
        assert derive_signal_policy("float64", "ref_lat", "Reference point latitude, (degrees)").method == "linear"

    def test_unitless_float_defaults_linear_low_confidence(self):
        p = derive_signal_policy("float32", "throttle", "normalized value")
        assert p.method == "linear"
        assert p.confidence == "low"

    def test_every_policy_is_nan_segmented(self):
        # NaN is a semantic sentinel everywhere in PX4; never bridge it.
        for p in (
            derive_signal_policy("bool", "x", ""),
            derive_signal_policy("float32", "vx", "[m/s]"),
            derive_signal_policy("float32", "yaw", "in rad -PI..+PI"),
            derive_signal_policy("float32", "q", "Quaternion", array_size=4),
        ):
            assert p.nan_segmented is True


class TestSignalPolicyRegistry:
    def test_registry_classifies_real_px4_fields(self):
        pol = load_px4_signal_policies()
        assert pol["vehicle_attitude.q"].method == "quaternion_slerp"
        assert pol["vehicle_attitude.delta_q_reset"].method == "discrete_hold"
        assert pol["vehicle_attitude.quat_reset_counter"].method == "discrete_hold"
        assert pol["vehicle_local_position.heading"].method == "angle_wrap"
        assert pol["vehicle_local_position.xy_valid"].method == "discrete_hold"
        assert pol["vehicle_local_position.vx"].method == "linear"
        assert pol["vehicle_local_position.ref_lat"].method == "linear"
        assert pol["sensor_gps.cog_rad"].method == "angle_wrap"

    def test_signal_policy_for_returns_none_for_unknown(self):
        assert signal_policy_for("nonexistent_topic.no_such_field") is None

    def test_enum_field_from_registry_is_discrete(self, tmp_path):
        msg_dir = tmp_path / "PX4-Autopilot" / "msg"
        msg_dir.mkdir(parents=True)
        (msg_dir / "VehicleStatus.msg").write_text(
            """
uint64 timestamp
uint8 vehicle_type          # see VEHICLE_TYPE
uint8 VEHICLE_TYPE_UNKNOWN = 0
uint8 VEHICLE_TYPE_ROTARY_WING = 1
float32 yaw                 # in radians NED -PI..+PI
""",
            encoding="utf-8",
        )
        pol = load_px4_signal_policies(tmp_path / "PX4-Autopilot")
        assert pol["vehicle_status.vehicle_type"].method == "discrete_hold"
        assert pol["vehicle_status.yaw"].method == "angle_wrap"
