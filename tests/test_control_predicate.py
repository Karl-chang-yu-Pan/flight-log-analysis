from flight_log_agent.analysis.control_predicate import lower_control_predicate


def test_control_predicate_lowers_source_field_and_px4_enum_to_log_expression(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleCommand.msg").write_text(
        """
uint64 timestamp
uint16 command
uint16 VEHICLE_CMD_DO_CHANGE_SPEED = 178
float32 param1
""",
        encoding="utf-8",
    )

    lowered = lower_control_predicate(
        "cmd.command == vehicle_command_s::VEHICLE_CMD_DO_CHANGE_SPEED",
        {"cmd.command": "vehicle_command.command"},
        source_path=tmp_path,
    )

    assert lowered.status == "log_verifiable"
    assert lowered.expression == "vehicle_command_command == 178"
    assert lowered.variables == {"vehicle_command_command": "vehicle_command.command"}


def test_control_predicate_classifies_internal_symbols_as_unlogged():
    lowered = lower_control_predicate(
        "reposition_valid && cmd.param1 <= 0",
        {"cmd.param1": "vehicle_command.param1"},
    )

    assert lowered.status == "internal_unlogged"
    assert "reposition_valid" in lowered.unresolved_symbols


def test_control_predicate_classifies_unsupported_calls():
    lowered = lower_control_predicate(
        "mission_item_to_position_setpoint(_mission_item, &triplet.current)",
        {},
    )

    assert lowered.status == "unsupported"
    assert "unsupported call" in lowered.reason


def test_control_predicate_lowers_common_vehicle_status_alias(tmp_path):
    msg_dir = tmp_path / "msg"
    msg_dir.mkdir()
    (msg_dir / "VehicleStatus.msg").write_text(
        """
uint64 timestamp
uint8 nav_state
uint8 NAVIGATION_STATE_AUTO_RTL = 5
""",
        encoding="utf-8",
    )

    lowered = lower_control_predicate(
        "_vstatus.nav_state == vehicle_status_s::NAVIGATION_STATE_AUTO_RTL",
        {"_vstatus.nav_state": "vehicle_status.nav_state"},
        source_path=tmp_path,
    )

    assert lowered.status == "log_verifiable"
    assert lowered.expression == "vehicle_status_nav_state == 5"
