from flight_log_agent.px4.msg_schema import (
    is_valid_topic_field,
    load_px4_msg_enum_registry,
    load_px4_msg_schema,
    normalize_px4_enum_value,
    resolve_topic_field,
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
