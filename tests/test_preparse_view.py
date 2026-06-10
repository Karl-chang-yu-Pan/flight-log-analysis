from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import flight_log_agent.ulog.preparse_view as preparse_view


def test_build_parameter_rows_classifies_defaults_by_flight_review_order():
    ulog = SimpleNamespace(
        initial_parameters={
            "FW_AIRSPD_TRIM": 15.0,
            "NAV_ACC_RAD": 10.0,
            "SYS_AUTOSTART": 4001,
            "VT_FW_PERM_STAB": 1,
        },
        has_default_parameters=True,
        changed_parameters=[],
    )
    ulog.get_default_parameters = lambda default_type: {
        0: {
            "FW_AIRSPD_TRIM": 15.0,
            "NAV_ACC_RAD": 5.0,
        },
        1: {
            "NAV_ACC_RAD": 10.0,
            "SYS_AUTOSTART": 4001,
        },
    }[default_type]
    metadata = {
        "VT_FW_PERM_STAB": {
            "default": "0",
            "type": "INT",
            "short_desc": "Permanent fixed-wing stabilization",
        },
    }

    rows = {
        row["name"]: row
        for row in preparse_view.build_parameter_rows(ulog, metadata)
    }

    assert rows["NAV_ACC_RAD"]["default_status"] == "default"
    assert rows["NAV_ACC_RAD"]["default_source"] == "ulog_airframe"
    assert rows["FW_AIRSPD_TRIM"]["default_status"] == "default"
    assert rows["FW_AIRSPD_TRIM"]["default_source"] == "ulog_system"
    assert rows["VT_FW_PERM_STAB"]["default_status"] == "non_default"
    assert rows["VT_FW_PERM_STAB"]["default_source"] == "metadata_xml"
    assert rows["SYS_AUTOSTART"]["default_status"] == "default"


def test_build_parameter_rows_marks_unknowns_and_rc_cal_flags():
    ulog = SimpleNamespace(
        initial_parameters={
            "CAL_GYRO0_ID": 123,
            "RC_MAP_ROLL": 1,
            "UNKNOWN_PARAM": 9,
        },
        has_default_parameters=False,
        changed_parameters=[(2_000_000, "UNKNOWN_PARAM", 10)],
    )

    rows = {
        row["name"]: row
        for row in preparse_view.build_parameter_rows(ulog)
    }

    assert rows["CAL_GYRO0_ID"]["is_rc_or_cal"] is True
    assert rows["RC_MAP_ROLL"]["is_rc_or_cal"] is True
    assert rows["UNKNOWN_PARAM"]["default_status"] == "unknown"
    assert rows["UNKNOWN_PARAM"]["changed_during_log"] is True


def test_load_parameter_metadata_extracts_flight_review_fields(tmp_path):
    metadata_path = tmp_path / "parameters.xml"
    metadata_path.write_text(
        """<parameters>
  <group name="Navigation">
    <parameter name="NAV_ACC_RAD" type="FLOAT" default="10.0">
      <min>1.0</min>
      <max>100.0</max>
      <short_desc>Acceptance radius</short_desc>
      <long_desc>Waypoint acceptance radius.</long_desc>
      <decimal>1</decimal>
    </parameter>
  </group>
</parameters>
""",
        encoding="utf-8",
    )

    metadata = preparse_view.load_parameter_metadata(metadata_path)

    assert metadata["NAV_ACC_RAD"] == {
        "default": "10.0",
        "type": "FLOAT",
        "group_name": "Navigation",
        "min": "1.0",
        "max": "100.0",
        "short_desc": "Acceptance radius",
        "long_desc": "Waypoint acceptance radius.",
        "decimal": "1",
    }


def test_build_topic_rows_includes_missing_and_derived_fields():
    rows = preparse_view.build_topic_rows({
        "available_topics": ["vehicle_attitude", "vehicle_status"],
        "topic_fields": {
            "vehicle_attitude": ["timestamp", "roll", "pitch", "yaw"],
            "vehicle_status": ["timestamp", "nav_state"],
        },
        "missing_topics": ["mission_result"],
    })

    by_name = {row["name"]: row for row in rows}

    assert by_name["vehicle_attitude"]["has_derived_attitude_fields"] is True
    assert by_name["vehicle_status"]["field_count"] == 2
    assert by_name["mission_result"]["missing_expected"] is True


def test_build_preparse_payload_uses_default_source_path_when_empty(tmp_path):
    default_source = tmp_path / "ref" / "PX4-Autopilot"
    default_source.mkdir(parents=True)
    log_path = tmp_path / "flight.ulg"
    mission_path = tmp_path / "mission.plan"

    with patch.object(preparse_view, "DEFAULT_PX4_SOURCE_PATH", default_source), patch.object(
        preparse_view,
        "parse_ulog_inventory",
        return_value={"available_topics": [], "topic_fields": {}},
    ) as parse_inventory, patch.object(
        preparse_view,
        "build_basic_timeline",
        return_value=[],
    ), patch.object(
        preparse_view,
        "infer_control_surface",
        return_value={},
    ) as infer_control, patch.object(
        preparse_view,
        "parse_mission_file",
        return_value={"items": []},
    ) as parse_mission, patch.object(
        preparse_view,
        "load_parameter_metadata",
        return_value={},
    ), patch.object(
        preparse_view,
        "build_parameter_payload",
        return_value={},
    ):
        result = preparse_view.build_preparse_payload(
            log_path,
            mission_path=mission_path,
            source_path="",
        )

    parse_inventory.assert_called_once_with(log_path, preparse_view.SOURCE_UNAVAILABLE)
    infer_control.assert_called_once_with(log_path, preparse_view.SOURCE_UNAVAILABLE)
    parse_mission.assert_called_once_with(
        mission_path,
        source_path=preparse_view.SOURCE_UNAVAILABLE,
    )
    assert result["inputs"]["source_path"] == str(default_source)


def test_build_preparse_payload_resolves_logged_revision_before_source_reads(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    log_path = tmp_path / "flight.ulg"
    mission_path = tmp_path / "mission.plan"
    events = []

    def fake_inventory(*args):
        events.append("inventory")
        return {
            "available_topics": [],
            "topic_fields": {},
            "git_hash": "abcdef1234567890",
        }

    snapshot = SimpleNamespace(commit_sha="abcdef1234567890")

    def fake_resolve(*args):
        events.append("resolve")
        return snapshot

    def fake_control(*args):
        events.append("control_surface")
        return {}

    def fake_mission(*args, **kwargs):
        events.append("mission")
        return {"items": []}

    with patch.object(
        preparse_view,
        "parse_ulog_inventory",
        side_effect=fake_inventory,
    ) as parse_inventory, patch.object(
        preparse_view,
        "SourceRepository",
        return_value=SimpleNamespace(resolve_snapshot=fake_resolve),
    ) as source_repository, patch.object(
        preparse_view,
        "build_basic_timeline",
        return_value=[],
    ), patch.object(
        preparse_view,
        "infer_control_surface",
        side_effect=fake_control,
    ) as infer_control, patch.object(
        preparse_view,
        "parse_mission_file",
        side_effect=fake_mission,
    ) as parse_mission, patch.object(
        preparse_view,
        "load_parameter_metadata",
        return_value={},
    ), patch.object(
        preparse_view,
        "build_parameter_payload",
        return_value={},
    ):
        preparse_view.build_preparse_payload(
            log_path,
            mission_path=mission_path,
            source_path=source_path,
        )

    parse_inventory.assert_called_once_with(log_path, preparse_view.SOURCE_UNAVAILABLE)
    source_repository.assert_called_once_with(source_path)
    infer_control.assert_called_once_with(log_path, snapshot)
    parse_mission.assert_called_once_with(mission_path, source_path=snapshot)
    assert events == ["inventory", "resolve", "control_surface", "mission"]
