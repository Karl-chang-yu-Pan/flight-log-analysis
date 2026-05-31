import importlib.util
import asyncio
import json
import math
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import flight_log_agent.ulog.inventory as ulog_inventory
import flight_log_agent.ulog.timeline as ulog_timeline
import flight_log_agent.ulog.control_surface as ulog_control_surface
import flight_log_agent.ulog.metrics as ulog_metrics
import flight_log_agent.ulog.signature_evaluator as ulog_signature_evaluator
import flight_log_agent.px4.source as px4_source
import flight_log_agent.mission.parser as mission_parser
import flight_log_agent.ulog.plots as ulog_plots
from flight_log_agent.px4 import msg_schema as px4_msg_schema


class FakeLoggedMessage:
    def __init__(self, timestamp, level, message, log_level=None):
        self.timestamp = timestamp
        self._level = level
        self.message = message
        self.log_level = log_level

    def log_level_str(self):
        return self._level


class FakePX4Events:
    def set_default_json_definitions_cb(self, callback):
        self.callback = callback

    def get_logged_events(self, ulog):
        return getattr(ulog, "logged_events", [])


def load_runner(tmp_path: Path):
    """Load runner.py with SDK stubs so tool unit tests stay local-only."""
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump_json(self, indent=None):
            return json.dumps(self.model_dump(), indent=indent)

        def model_dump(self, exclude_none=False):
            data = {
                key: self._dump_value(value, exclude_none)
                for key, value in self.__dict__.items()
            }
            if exclude_none:
                data = {key: value for key, value in data.items() if value is not None}
            return data

        @classmethod
        def _dump_value(cls, value, exclude_none=False):
            if isinstance(value, BaseModel):
                return value.model_dump(exclude_none=exclude_none)
            if isinstance(value, list):
                return [cls._dump_value(item, exclude_none) for item in value]
            if isinstance(value, dict):
                return {
                    key: cls._dump_value(item, exclude_none)
                    for key, item in value.items()
                    if not exclude_none or item is not None
                }
            return value

    pydantic_stub.BaseModel = BaseModel

    def Field(default=None, *, default_factory=None, **kwargs):
        if default_factory is not None:
            return default_factory()
        return default

    pydantic_stub.Field = Field

    agents_stub = types.ModuleType("agents")

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Runner:
        pass

    class RunHooks:
        pass

    class WebSearchTool:
        pass

    class RunContextWrapper:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    def function_tool(func):
        return func

    agents_stub.Agent = Agent
    agents_stub.Runner = Runner
    agents_stub.RunHooks = RunHooks
    agents_stub.WebSearchTool = WebSearchTool
    agents_stub.RunContextWrapper = RunContextWrapper
    agents_stub.function_tool = function_tool

    module_name = f"runner_under_test_{tmp_path.name}"
    runner_path = Path(__file__).resolve().parents[1] / "flight_log_agent" / "runner_core.py"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    module = importlib.util.module_from_spec(spec)

    original_pydantic = sys.modules.get("pydantic")
    original_agents = sys.modules.get("agents")
    original_flight_log_agent_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "flight_log_agent" or name.startswith("flight_log_agent.")
    }
    sys.modules["pydantic"] = pydantic_stub
    sys.modules["agents"] = agents_stub
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    finally:
        for name in list(sys.modules):
            if (
                (name == "flight_log_agent" or name.startswith("flight_log_agent."))
                and name not in original_flight_log_agent_modules
            ):
                sys.modules.pop(name, None)
        for name, original_module in original_flight_log_agent_modules.items():
            sys.modules[name] = original_module

        if original_pydantic is None:
            sys.modules.pop("pydantic", None)
        else:
            sys.modules["pydantic"] = original_pydantic

        if original_agents is None:
            sys.modules.pop("agents", None)
        else:
            sys.modules["agents"] = original_agents

    return module


def make_ctx(runner, tmp_path, source_path=None):
    context = runner.FlightLogContext(
        log_path=tmp_path / "flight.ulg",
        mission_path=None,
        source_path=source_path,
        output_dir=tmp_path / "outputs",
    )
    return SimpleNamespace(context=context)


def test_parse_ulog_inventory_extracts_inventory_from_pyulog(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"
    source_path = tmp_path / "PX4-Autopilot"
    airframe_dir = source_path / "ROMFS" / "px4fmu_common" / "init.d" / "airframes"
    airframe_dir.mkdir(parents=True)
    (airframe_dir / "4001_quad_x").write_text(
        "\n".join(
            [
                "#!/bin/sh",
                "# @name Generic Quadcopter",
                "# @type Quadrotor x",
                "# @class Copter",
            ]
        ),
        encoding="utf-8",
    )

    class FakeULog:
        def __init__(self, path, msg_filter=None, disable_str_exceptions=False):
            self.path = path
            self.disable_str_exceptions = disable_str_exceptions
            self.msg_info_dict = {
                "ver_sw_release": 0x010E00FF,
                "ver_sw": b"abcdef123456\x00",
                "ver_sw_branch": b"release/1.14\x00",
            }
            self.initial_parameters = {
                "SYS_AUTOSTART": 4001,
                "NAV_ACC_RAD": 10.0,
                "UNRELATED_PARAM": 99,
            }
            self.start_timestamp = 1_000_000
            self.last_timestamp = 6_500_000
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_status",
                    field_data=[
                        SimpleNamespace(field_name="timestamp"),
                        SimpleNamespace(field_name="nav_state"),
                    ],
                    data={"timestamp": [1], "nav_state": [3]},
                ),
                SimpleNamespace(
                    name="sensor_accel",
                    multi_id=0,
                    field_data=[
                        SimpleNamespace(field_name="timestamp"),
                        SimpleNamespace(field_name="x"),
                    ],
                    data={"timestamp": [1, 2], "x": [0.1, 0.2]},
                ),
                SimpleNamespace(
                    name="sensor_accel",
                    multi_id=1,
                    field_data=[
                        SimpleNamespace(field_name="timestamp"),
                        SimpleNamespace(field_name="x"),
                        SimpleNamespace(field_name="y"),
                    ],
                    data={"timestamp": [1], "x": [0.3], "y": [0.4]},
                ),
                SimpleNamespace(name="mission_result", data={"timestamp": [1], "seq_current": [1]}),
                SimpleNamespace(name="sensor_combined", data={"timestamp": [1], "gyro_rad[0]": [0.1]}),
                SimpleNamespace(
                    name="debug_topic",
                    field_data=[SimpleNamespace(field_name="custom_field")],
                    data={"timestamp": [1], "custom_field": [42]},
                ),
            ]
            self.logged_messages = [
                FakeLoggedMessage(1_050_000, "EMERGENCY", "hard failure", ord("0")),
                FakeLoggedMessage(1_100_000, "INFO", "armed"),
                FakeLoggedMessage(1_200_000, "WARNING", "duplicate event\t"),
                FakeLoggedMessage(1_300_000, "ERROR", "mission failure"),
            ]
            self.logged_events = [
                (1_250_000, "WARNING", "failsafe event"),
            ]
            self.dropouts = [SimpleNamespace(duration=42)]

        def get_version_info_str(self):
            return "v1.14.0"

        def get_version_info(self):
            return (1, 14, 0, 255)

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)
    monkeypatch.setattr(ulog_inventory, "PX4Events", FakePX4Events)

    result = ulog_inventory.parse_ulog_inventory(log_path, source_path)

    assert result == {
        "firmware_version": "v1.14.0",
        "firmware_branch": "release/1.14",
        "git_hash": "abcdef123456",
        "airframe": {
            "id": 4001,
            "name": "Generic Quadcopter",
            "type": "Quadrotor x",
            "class": "Copter",
            "source": str(source_path),
            "file": "ROMFS/px4fmu_common/init.d/airframes/4001_quad_x",
        },
        "duration_s": 5.5,
        "parameters": {
            "NAV_ACC_RAD": 10.0,
            "SYS_AUTOSTART": 4001,
            "UNRELATED_PARAM": 99,
        },
        "source_path": str(source_path),
        "available_topics": [
            "debug_topic",
            "mission_result",
            "sensor_accel",
            "sensor_combined",
            "vehicle_status",
        ],
        "topic_fields": {
            "debug_topic": ["custom_field", "timestamp"],
            "mission_result": ["seq_current", "timestamp"],
            "sensor_accel": ["timestamp", "x", "y"],
            "sensor_combined": ["gyro_rad[0]", "timestamp"],
            "vehicle_status": ["nav_state", "timestamp"],
        },
        "topic_instances": {
            "debug_topic": [
                {"multi_id": 0, "fields": ["custom_field", "timestamp"], "sample_count": 1}
            ],
            "mission_result": [
                {"multi_id": 0, "fields": ["seq_current", "timestamp"], "sample_count": 1}
            ],
            "sensor_accel": [
                {"multi_id": 0, "fields": ["timestamp", "x"], "sample_count": 2},
                {"multi_id": 1, "fields": ["timestamp", "x", "y"], "sample_count": 1},
            ],
            "sensor_combined": [
                {"multi_id": 0, "fields": ["gyro_rad[0]", "timestamp"], "sample_count": 1}
            ],
            "vehicle_status": [
                {"multi_id": 0, "fields": ["nav_state", "timestamp"], "sample_count": 1}
            ],
        },
        "logged_messages": [
            {
                "timestamp": 1_050_000,
                "time_s": 1.05,
                "level": "EMERGENCY",
                "message": "hard failure",
                "source": "logged_message",
            },
            {
                "timestamp": 1_100_000,
                "time_s": 1.1,
                "level": "INFO",
                "message": "armed",
                "source": "logged_message",
            },
            {
                "timestamp": 1_250_000,
                "time_s": 1.25,
                "level": "WARNING",
                "message": "failsafe event",
                "source": "event",
            },
            {
                "timestamp": 1_300_000,
                "time_s": 1.3,
                "level": "ERROR",
                "message": "mission failure",
                "source": "logged_message",
            },
        ],
        "warnings": [
            "emergency: hard failure",
            "warning: failsafe event",
            "error: mission failure",
            "dropout: 42 ms",
        ],
        "missing_topics": [
            "vehicle_type",
            "vtol_vehicle_status",
        ],
    }


def test_parse_ulog_inventory_resolves_airframe_from_logged_git_revision(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"
    source_path = tmp_path / "PX4-Autopilot"
    airframe_dir = source_path / "ROMFS" / "px4fmu_common" / "init.d" / "airframes"
    airframe_dir.mkdir(parents=True)
    airframe_file = airframe_dir / "4001_quad_x"
    airframe_file.write_text(
        "# @name Logged Revision Quad\n# @type Quadrotor x\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=source_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "add", "."], cwd=source_path, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "old airframe"],
        cwd=source_path,
        check=True,
        capture_output=True,
        text=True,
    )
    logged_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    airframe_file.write_text(
        "# @name Worktree Quad\n# @type Quadrotor changed\n",
        encoding="utf-8",
    )

    class FakeULog:
        def __init__(self, path, msg_filter=None, disable_str_exceptions=False):
            self.msg_info_dict = {"ver_sw": logged_hash}
            self.initial_parameters = {"SYS_AUTOSTART": 4001}
            self.start_timestamp = 1_000_000
            self.last_timestamp = 2_000_000
            self.data_list = []
            self.logged_messages = []
            self.dropouts = []

        def get_version_info_str(self):
            return None

        def get_version_info(self):
            return None

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)

    result = ulog_inventory.parse_ulog_inventory(log_path, source_path)

    assert result["airframe"]["name"] == "Logged Revision Quad"
    assert result["airframe"]["type"] == "Quadrotor x"
    assert result["airframe"]["source"] == f"{source_path}@{logged_hash[:8]}"


def test_parse_ulog_inventory_reports_parse_failure(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path, msg_filter=None, disable_str_exceptions=False):
            raise ValueError("bad log")

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)

    result = ulog_inventory.parse_ulog_inventory(tmp_path / "flight.ulg")

    assert result["parameters"] == {}
    assert result["missing_topics"] == ulog_inventory.EXPECTED_TIMELINE_TOPICS
    assert result["warnings"] == ["failed to parse ULog: bad log"]


def test_parse_ulog_inventory_includes_derived_attitude_plot_fields(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path, msg_filter=None, disable_str_exceptions=False):
            self.msg_info_dict = {}
            self.initial_parameters = {}
            self.start_timestamp = 1_000_000
            self.last_timestamp = 2_000_000
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_attitude",
                    data={
                        "timestamp": [1_000_000],
                        "q[0]": [1.0],
                        "q[1]": [0.0],
                        "q[2]": [0.0],
                        "q[3]": [0.0],
                    },
                ),
                SimpleNamespace(
                    name="vehicle_attitude_setpoint",
                    data={
                        "timestamp": [1_000_000],
                        "q_d[0]": [1.0],
                        "q_d[1]": [0.0],
                        "q_d[2]": [0.0],
                        "q_d[3]": [0.0],
                    },
                ),
            ]
            self.logged_messages = []
            self.dropouts = []

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)

    result = ulog_inventory.parse_ulog_inventory(tmp_path / "flight.ulg")

    assert result["topic_fields"]["vehicle_attitude"] == [
        "pitch",
        "q[0]",
        "q[1]",
        "q[2]",
        "q[3]",
        "roll",
        "timestamp",
        "yaw",
    ]
    assert result["topic_fields"]["vehicle_attitude_setpoint"] == [
        "pitch_d",
        "q_d[0]",
        "q_d[1]",
        "q_d[2]",
        "q_d[3]",
        "roll_d",
        "timestamp",
        "yaw_d",
    ]


def test_build_basic_timeline_returns_state_change_events(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"

    class FakeULog:
        def __init__(self, path):
            self.path = path
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "arming_state": [1, 2, 2],
                        "nav_state": [0, 0, 3],
                    },
                ),
                SimpleNamespace(
                    name="mission_result",
                    data={
                        "timestamp": [1_500_000, 2_500_000, 3_500_000],
                        "seq_current": [0, 1, 1],
                        "mission_finished": [False, False, True],
                    },
                ),
                SimpleNamespace(
                    name="sensor_combined",
                    data={
                        "timestamp": [1_000_000],
                        "gyro_rad": [0.1],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_timeline, "ULog", FakeULog)

    result = ulog_timeline.build_basic_timeline(log_path)

    assert result == [
        {
            "time_s": 1.0,
            "event": "initial_value",
            "topic": "vehicle_status",
            "field": "arming_state",
            "value": 1,
        },
        {
            "time_s": 1.0,
            "event": "initial_value",
            "topic": "vehicle_status",
            "field": "nav_state",
            "value": 0,
        },
        {
            "time_s": 1.5,
            "event": "initial_value",
            "topic": "mission_result",
            "field": "seq_current",
            "value": 0,
        },
        {
            "time_s": 1.5,
            "event": "initial_value",
            "topic": "mission_result",
            "field": "mission_finished",
            "value": False,
        },
        {
            "time_s": 2.0,
            "event": "value_changed",
            "topic": "vehicle_status",
            "field": "arming_state",
            "value": 2,
        },
        {
            "time_s": 2.5,
            "event": "value_changed",
            "topic": "mission_result",
            "field": "seq_current",
            "value": 1,
        },
        {
            "time_s": 3.0,
            "event": "value_changed",
            "topic": "vehicle_status",
            "field": "nav_state",
            "value": 3,
        },
        {
            "time_s": 3.5,
            "event": "value_changed",
            "topic": "mission_result",
            "field": "mission_finished",
            "value": True,
        },
    ]


def test_build_basic_timeline_reports_parse_failure(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            raise ValueError("bad log")

    monkeypatch.setattr(ulog_timeline, "ULog", FakeULog)

    result = ulog_timeline.build_basic_timeline(tmp_path / "flight.ulg")

    assert result == [{
        "time_s": None,
        "event": "timeline_unavailable",
        "details": "failed to parse ULog: bad log",
    }]


def test_runner_parse_ulog_inventory_delegates_to_inventory_module(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    source_path = tmp_path / "PX4-Autopilot"

    with patch.object(
        runner,
        "parse_ulog_inventory_impl",
        return_value={"available_topics": ["vehicle_status"]},
    ) as parse_impl:
        result = runner.parse_ulog_inventory(log_path, source_path)

    parse_impl.assert_called_once_with(log_path, source_path)
    assert result == {"available_topics": ["vehicle_status"]}


def test_resolve_source_path_uses_repo_default_when_available(tmp_path):
    runner = load_runner(tmp_path)
    default_source = tmp_path / "ref" / "PX4-Autopilot"
    default_source.mkdir(parents=True)
    explicit_source = tmp_path / "custom-px4"

    runner.DEFAULT_PX4_SOURCE_PATH = default_source

    assert runner.resolve_source_path(None) == default_source
    assert runner.resolve_source_path("") == default_source
    assert runner.resolve_source_path(explicit_source) == explicit_source

    runner.DEFAULT_PX4_SOURCE_PATH = tmp_path / "missing"
    assert runner.resolve_source_path(None) is None


def test_runner_build_basic_timeline_delegates_to_timeline_module(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"

    with patch.object(
        runner,
        "build_basic_timeline_impl",
        return_value=[{"event": "initial_value"}],
    ) as timeline_impl:
        result = runner.build_basic_timeline(log_path)

    timeline_impl.assert_called_once_with(log_path)
    assert result == [{"event": "initial_value"}]


def test_parse_mission_file_returns_none_without_path():
    assert mission_parser.parse_mission_file(None) is None


def test_parse_mission_file_extracts_qgroundcontrol_plan_items(tmp_path):
    mission_path = tmp_path / "mission.plan"
    mission_path.write_text(
        json.dumps({
            "groundStation": "QGroundControl",
            "version": 1,
            "mission": {
                "cruiseSpeed": 18,
                "firmwareType": 12,
                "hoverSpeed": 5,
                "plannedHomePosition": [47.397742, 8.545594, 488],
                "vehicleType": 2,
                "version": 2,
                "items": [
                    {
                        "AMSLAltAboveTerrain": None,
                        "Altitude": 50,
                        "autoContinue": True,
                        "command": 22,
                        "doJumpId": 1,
                        "frame": 3,
                        "params": [15, 0, 0, None, 47.397742, 8.545594, 50],
                        "type": "SimpleItem",
                    },
                    {
                        "Altitude": 65,
                        "autoContinue": True,
                        "command": 16,
                        "doJumpId": 2,
                        "frame": 3,
                        "params": [0, 10, 0, None, 47.398, 8.546, 65],
                        "type": "SimpleItem",
                    },
                ],
            },
        })
    )

    result = mission_parser.parse_mission_file(mission_path)

    assert result == {
        "mission_file": str(mission_path),
        "format": "qgroundcontrol_plan",
        "version": 1,
        "ground_station": "QGroundControl",
        "planned_home_position": [47.397742, 8.545594, 488],
        "vehicle_type": 2,
        "firmware_type": 12,
        "cruise_speed": 18,
        "hover_speed": 5,
        "items": [
            {
                "sequence": 1,
                "type": "SimpleItem",
                "command": 22,
                "command_name": "MAV_CMD_NAV_TAKEOFF",
                "frame": 3,
                "frame_name": "MAV_FRAME_GLOBAL_RELATIVE_ALT",
                "auto_continue": True,
                "params": [15, 0, 0, None, 47.397742, 8.545594, 50],
                "latitude": 47.397742,
                "longitude": 8.545594,
                "altitude": 50,
            },
            {
                "sequence": 2,
                "type": "SimpleItem",
                "command": 16,
                "command_name": "MAV_CMD_NAV_WAYPOINT",
                "frame": 3,
                "frame_name": "MAV_FRAME_GLOBAL_RELATIVE_ALT",
                "auto_continue": True,
                "params": [0, 10, 0, None, 47.398, 8.546, 65],
                "latitude": 47.398,
                "longitude": 8.546,
                "altitude": 65,
            },
        ],
        "warnings": [],
    }


def test_parse_mission_file_extracts_qgc_wpl_items(tmp_path):
    mission_path = tmp_path / "mission.waypoints"
    mission_path.write_text(
        "QGC WPL 110\n"
        "0\t1\t3\t22\t15\t0\t0\t0\t47.397742\t8.545594\t50\t1\n"
        "1\t0\t3\t16\t0\t10\t0\t0\t47.398\t8.546\t65\t1\n"
    )

    result = mission_parser.parse_mission_file(mission_path)

    assert result["format"] == "qgc_wpl"
    assert result["version"] == "110"
    assert result["items"] == [
        {
            "sequence": 0,
            "current": True,
            "type": "SimpleItem",
            "command": 22,
            "command_name": "MAV_CMD_NAV_TAKEOFF",
            "frame": 3,
            "frame_name": "MAV_FRAME_GLOBAL_RELATIVE_ALT",
            "auto_continue": True,
            "params": [15, 0, 0, 0, 47.397742, 8.545594, 50],
            "latitude": 47.397742,
            "longitude": 8.545594,
            "altitude": 50,
        },
        {
            "sequence": 1,
            "current": False,
            "type": "SimpleItem",
            "command": 16,
            "command_name": "MAV_CMD_NAV_WAYPOINT",
            "frame": 3,
            "frame_name": "MAV_FRAME_GLOBAL_RELATIVE_ALT",
            "auto_continue": True,
            "params": [0, 10, 0, 0, 47.398, 8.546, 65],
            "latitude": 47.398,
            "longitude": 8.546,
            "altitude": 65,
        },
    ]
    assert result["warnings"] == []


def test_parse_mission_file_reports_missing_file(tmp_path):
    mission_path = tmp_path / "missing.plan"

    result = mission_parser.parse_mission_file(mission_path)

    assert result["mission_file"] == str(mission_path)
    assert result["items"] == []
    assert result["warnings"] == [f"mission file does not exist: {mission_path}"]


def test_runner_parse_mission_file_delegates_to_mission_parser(tmp_path):
    runner = load_runner(tmp_path)
    mission_path = tmp_path / "mission.plan"

    with patch.object(
        runner,
        "parse_mission_file_impl",
        return_value={"mission_file": str(mission_path), "items": []},
    ) as parse_impl:
        result = runner.parse_mission_file(mission_path)

    parse_impl.assert_called_once_with(mission_path)
    assert result == {"mission_file": str(mission_path), "items": []}


def _sample_mechanism_candidate(runner, plots=None):
    return runner.MechanismCandidate(
        name="TECS altitude demand",
        summary="TECS altitude setpoint can diverge from measured altitude.",
        source_refs=[
            runner.CodeRef(
                file="src/modules/fw_pos_control_l1/FixedwingPositionControl.cpp",
                function="control_position",
                start_line=100,
                explanation="Applies TECS altitude setpoint.",
            )
        ],
        required_parameters=["SYS_AUTOSTART"],
        required_signals=[
            "vehicle_local_position.z",
            "vehicle_local_position_setpoint.z",
        ],
        expected_logged_signature=[
            runner.ExpectedSignatureItem(
                name="altitude tracking",
                description="actual altitude follows setpoint",
                signal="vehicle_local_position.z",
            )
        ],
        numeric_checks=[
            runner.RelationshipCheckSpec(
                type="tracks_setpoint",
                actual="vehicle_local_position.z",
                setpoint="vehicle_local_position_setpoint.z",
                window="altitude_drop",
                max_error=3.0,
                supports="Altitude remained close to setpoint.",
            )
        ],
        plot_requests=plots or [],
    )


def _sample_applicability(runner, candidate):
    return runner.ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=True,
        supported_conditions=["Parameter present: SYS_AUTOSTART=4001"],
        relevant_parameters={"SYS_AUTOSTART": 4001},
        candidate_windows=[
            runner.WindowSpec(
                name="altitude_drop",
                start_s=10.0,
                end_s=20.0,
                reason="transition window",
            )
        ],
        available_required_signals=candidate.required_signals,
        missing_required_signals=[],
    )


def test_applicability_checks_required_signal_fields(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Missing field mechanism",
        summary="Needs a specific logged field.",
        source_refs=[],
        required_signals=["vehicle_local_position.z", "vehicle_status.nav_state"],
    )

    result = runner.evaluate_candidate_applicability(
        candidate,
        {
            "available_topics": ["vehicle_local_position", "vehicle_status"],
            "topic_fields": {
                "vehicle_local_position": ["x", "y"],
                "vehicle_status": ["nav_state"],
            },
        },
        [],
        None,
    )

    assert result.available_required_signals == ["vehicle_local_position.z", "vehicle_status.nav_state"]
    assert result.missing_required_signals == []


def test_applicability_uses_px4_schema_when_inventory_lacks_field(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Nested field mechanism",
        summary="Needs a nested PX4 message field.",
        source_refs=[],
        required_signals=["position_setpoint_triplet.current.alt"],
    )

    result = runner.evaluate_candidate_applicability(
        candidate,
        {
            "available_topics": ["position_setpoint_triplet"],
            "topic_fields": {"position_setpoint_triplet": ["timestamp"]},
        },
        [],
        None,
    )

    assert result.available_required_signals == ["position_setpoint_triplet.current.alt"]
    assert result.missing_required_signals == []


def test_applicability_treats_flattened_struct_prefix_as_present(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Nested prefix mechanism",
        summary="Needs a nested PX4 message prefix.",
        source_refs=[],
        required_signals=["position_setpoint_triplet.current"],
    )

    result = runner.evaluate_candidate_applicability(
        candidate,
        {
            "available_topics": ["position_setpoint_triplet"],
            "topic_fields": {"position_setpoint_triplet": ["timestamp", "current.alt", "current.valid"]},
        },
        [],
        None,
    )

    assert result.available_required_signals == ["position_setpoint_triplet.current"]
    assert result.missing_required_signals == []


def test_applicability_derives_fallback_windows_from_check_names(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Windowed mechanism",
        summary="Needs a dynamic check window.",
        source_refs=[],
        numeric_checks=[
            runner.RelationshipCheckSpec(
                type="threshold",
                signal="vehicle_local_position.z",
                window="source_window",
                metric="mean",
                op=">=",
                value=0,
            )
        ],
    )

    result = runner.evaluate_candidate_applicability(
        candidate,
        {"topic_fields": {"vehicle_local_position": ["timestamp", "z"]}},
        [
            {"time_s": 2.0, "topic": "vehicle_status", "field": "nav_state", "value": 1},
            {"time_s": 7.5, "topic": "vehicle_status", "field": "nav_state", "value": 2},
        ],
        None,
    )

    assert [window.model_dump() for window in result.candidate_windows] == [
        {
            "name": "source_window",
            "start_s": 2.0,
            "end_s": 7.5,
            "reason": "Fallback verification window from full timeline span for check window 'source_window'.",
        }
    ]
    assert "No candidate verification window" not in " ".join(result.unresolved_conditions)


def test_source_mechanism_conversion_emits_executable_checks(tmp_path):
    runner = load_runner(tmp_path)
    source_candidate = runner.SourceMechanismCandidate(
        title="RTL altitude source path",
        source_mechanism="PX4 gates this path on VT_TYPE.",
        controlling_parameters=[
            runner.ParameterRequirement(
                name="VT_TYPE",
                role="branch_selector",
                source_predicate="_param_vt_type.get() == 2",
                actual_value=2,
                gate_result="satisfied",
                effect="VT_TYPE satisfies the VTOL RTL branch.",
                source_file="src/modules/navigator/rtl.cpp",
                source_line=10,
            )
        ],
        relevant_fields=[
            runner.SourceFieldRef(
                topic="position_setpoint",
                field="alt",
                source_file="src/modules/navigator/rtl.cpp",
                source_line=11,
            )
        ],
        required_log_evidence=["Fetch time-series for position_setpoint.alt."],
        verification_checks=[
            runner.SourceBackedVerificationCheck(
                check=runner.RelationshipCheckSpec(
                    type="tracks_parameter_value",
                    signal="position_setpoint.alt",
                    parameter="RTL_RETURN_ALT",
                    window="rtl",
                    max_error=0.5,
                ),
                source_file="src/modules/navigator/rtl.cpp",
                source_line=12,
            )
        ],
    )

    candidate = runner.source_mechanism_to_candidate(source_candidate)

    assert candidate.exclusion_checks[0].type == "branch_parameter_satisfied"
    assert candidate.exclusion_checks[0].parameter == "VT_TYPE"
    assert candidate.exclusion_checks[0].op == "=="
    assert candidate.exclusion_checks[0].value == 2
    assert candidate.required_signals == ["position_setpoint.alt"]
    assert candidate.source_relevant_fields == ["position_setpoint.alt"]
    assert candidate.numeric_checks[0].type == "tracks_parameter_value"
    assert candidate.numeric_checks[0].parameter == "RTL_RETURN_ALT"
    assert candidate.numeric_checks[1].type == "topic_field_present"
    assert candidate.numeric_checks[1].signal == "position_setpoint.alt"


def test_source_mechanism_conversion_preserves_derived_expression_checks(tmp_path):
    runner = load_runner(tmp_path)
    source_candidate = runner.SourceMechanismCandidate(
        title="RTL altitude expression source path",
        source_mechanism="PX4 computes a return altitude from logged altitude and RTL_RETURN_ALT.",
        verification_checks=[
            runner.SourceBackedVerificationCheck(
                check=runner.RelationshipCheckSpec(
                    type="derived_expression",
                    expression="current_alt + RTL_RETURN_ALT",
                    variables=[{"name": "current_alt", "source": "vehicle_local_position.z"}],
                    window="rtl",
                    op=">=",
                    value=100,
                ),
                source_file="src/modules/navigator/rtl.cpp",
                source_line=42,
            )
        ],
    )

    candidate = runner.source_mechanism_to_candidate(source_candidate)

    assert candidate.required_signals == []
    assert candidate.numeric_checks[0].type == "derived_expression"
    assert candidate.numeric_checks[0].expression == "current_alt + RTL_RETURN_ALT"
    assert [
        variable.model_dump() if hasattr(variable, "model_dump") else variable
        for variable in candidate.numeric_checks[0].variables
    ] == [{"name": "current_alt", "source": "vehicle_local_position.z"}]


def test_source_mechanism_conversion_preserves_helper_dependencies(tmp_path):
    runner = load_runner(tmp_path)
    source_candidate = runner.SourceMechanismCandidate(
        title="Helper expression source path",
        source_mechanism="Source formula depends on a helper that still needs translation.",
        verification_checks=[
            runner.SourceBackedVerificationCheck(
                check=runner.RelationshipCheckSpec(
                    type="derived_expression",
                    expression="helper_result",
                    helper_dependencies=[
                        {
                            "name": "source_helper",
                            "args": ["input_a", "input_b"],
                            "source_file": "src/modules/example.cpp",
                            "source_line": 42,
                            "unresolved_reason": "helper body uses unsupported runtime state",
                        }
                    ],
                ),
                source_file="src/modules/example.cpp",
                source_line=42,
            )
        ],
    )

    candidate = runner.source_mechanism_to_candidate(source_candidate)
    dependency = candidate.numeric_checks[0].helper_dependencies[0]

    dumped = dependency.model_dump() if hasattr(dependency, "model_dump") else dependency
    assert dumped["name"] == "source_helper"
    assert dumped["args"] == ["input_a", "input_b"]
    assert dumped["unresolved_reason"] == "helper body uses unsupported runtime state"


def test_source_relevant_fields_are_not_hard_required_without_checks(tmp_path):
    runner = load_runner(tmp_path)
    source_candidate = runner.SourceMechanismCandidate(
        title="Source-only field",
        source_mechanism="Source touched a field that is not yet a required log signal.",
        relevant_fields=[
            runner.SourceFieldRef(
                topic="vehicle_local_position_setpoint",
                field="z",
                source_file="src/modules/example.cpp",
                source_line=20,
            )
        ],
    )

    candidate = runner.source_mechanism_to_candidate(source_candidate)

    assert candidate.required_signals == []
    assert candidate.source_relevant_fields == ["vehicle_local_position_setpoint.z"]


def test_source_relevant_fields_resolve_nested_px4_schema_fields(tmp_path):
    runner = load_runner(tmp_path)
    source_candidate = runner.SourceMechanismCandidate(
        title="Nested source field",
        source_mechanism="Source touched nested triplet fields.",
        relevant_fields=[
            runner.SourceFieldRef(
                topic="position_setpoint_triplet",
                field="current.alt",
                source_file="src/modules/example.cpp",
                source_line=20,
            )
        ],
        verification_checks=[
            runner.SourceBackedVerificationCheck(
                check=runner.RelationshipCheckSpec(
                    type="topic_field_present",
                    signal="position_setpoint_triplet.current.alt",
                ),
                source_file="src/modules/example.cpp",
                source_line=21,
            )
        ],
    )

    candidate = runner.source_mechanism_to_candidate(source_candidate)

    assert candidate.required_signals == ["position_setpoint_triplet.current.alt"]
    assert candidate.source_relevant_fields == ["position_setpoint_triplet.current.alt"]


def test_extract_signal_reference_keeps_nested_scalar_field(tmp_path):
    runner = load_runner(tmp_path)

    assert (
        runner.extract_signal_reference(
            "Check whether position_setpoint_triplet.current.alt is logged before verification."
        )
        == "position_setpoint_triplet.current.alt"
    )


def _sample_report(runner, candidate, applicability, plots=None):
    report_applicability = runner.ApplicabilityReport(
        applicable=applicability.applicable,
        supported_conditions=applicability.supported_conditions,
        excluded_by=applicability.excluded_by,
        unresolved_conditions=applicability.unresolved_conditions,
        relevant_parameters=[
            runner.ParameterValue(name=name, value=str(value))
            for name, value in applicability.relevant_parameters.items()
        ],
        available_required_signals=applicability.available_required_signals,
        missing_required_signals=applicability.missing_required_signals,
    )
    return runner.FlightLogReport(
        airframe_summary="fixed_wing SYS_AUTOSTART=4001",
        question_intent_summary="altitude drop after transition",
        ranked_hypotheses=[
            runner.HypothesisReportItem(
                title="Altitude drop after transition",
                known_px4_mechanism=candidate.name,
                mechanism="TECS changed altitude demand during transition.",
                source_refs=candidate.source_refs,
                expected_logged_signature=candidate.expected_logged_signature,
                applicability=report_applicability,
                exclusion_checks=[],
                numeric_checks=candidate.numeric_checks,
                evidence=["Altitude remained close to setpoint."],
                contradicting_evidence=[],
                confidence="medium",
                plots=plots or [],
            )
        ],
        excluded_mechanisms=[],
        confirmed=[],
        unconfirmed=[],
        final_summary="answered",
    )


def test_analyze_flight_log_runs_v3_mechanism_first_workflow(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    mission_path = tmp_path / "mission.plan"
    source_path = tmp_path / "PX4-Autopilot"
    output_dir = tmp_path / "outputs"
    dev_log_root = tmp_path / "dev_logs"
    captured = []

    question_intent = runner.QuestionIntent(
        original_question="Why did it loiter before the waypoint?",
        problem_domain="fixed_wing_altitude",
        concise_intent="explain altitude drop",
        source_queries=["TECS altitude setpoint"],
    )
    source_candidate = runner.SourceMechanismCandidate(
        title="Source mechanism path for rtl",
        source_mechanism="PX4 source can command an altitude setpoint before the waypoint.",
        source_chain=[
            runner.CodeRef(
                file="src/modules/navigator/rtl.cpp",
                start_line=10,
                snippet="sp.alt = target_alt;",
                explanation="source match",
            )
        ],
        source_files=["src/modules/navigator/rtl.cpp"],
        controlling_parameters=[
            runner.ParameterRequirement(
                name="SYS_AUTOSTART",
                role="branch_selector",
                source_predicate="SYS_AUTOSTART == 4001",
                actual_value=4001,
                gate_result="satisfied",
                effect="SYS_AUTOSTART satisfies the fixed-wing source branch.",
                source_file="src/modules/navigator/rtl.cpp",
                source_line=10,
            )
        ],
        published_topics=[],
        subscribed_topics=[],
        relevant_fields=[
            runner.SourceFieldRef(
                topic="vehicle_local_position_setpoint",
                field="z",
                variable="sp",
                source_file="src/modules/navigator/rtl.cpp",
                source_line=11,
            )
        ],
        branch_conditions=["SYS_AUTOSTART == 4001"],
        expected_log_signature=[
            "Verify later whether vehicle_local_position_setpoint.z follows the source-expected behavior."
        ],
        required_log_evidence=["Fetch time-series for vehicle_local_position_setpoint.z."],
        contradiction_checks=[],
        source_confidence="medium",
        resolver_notes=["No time-series signal comparison was computed."],
    )
    source_candidate_set = runner.SourceMechanismCandidateSet(
        candidates=[source_candidate],
        expansion_queries=["TECS altitude setpoint"],
        unresolved_questions=[],
    )
    candidate = runner.source_mechanism_to_candidate(source_candidate)
    applicability = _sample_applicability(runner, candidate)
    evaluation = runner.SignatureEvaluation(
        candidate_name=candidate.name,
        verdict="supported",
        confidence_ceiling="medium",
        evidence=["Altitude remained close to setpoint."],
        contradictions=[],
        check_results=[],
        warnings=[],
        raw={},
    )
    final_report = _sample_report(runner, candidate, applicability)

    async def fake_run(agent, input, context, max_turns, hooks=None):
        captured.append(
            {
                "agent": agent,
                "input": json.loads(input),
                "context": context,
                "max_turns": max_turns,
                "hooks": hooks,
            }
        )
        if agent is runner.question_intent_agent:
            return SimpleNamespace(final_output=question_intent, new_items=[])
        if agent is runner.final_report_agent:
            return SimpleNamespace(final_output=final_report, new_items=[])
        raise AssertionError(f"unexpected agent: {agent}")

    runner.Runner.run = fake_run

    with patch.object(
        runner,
        "parse_ulog_inventory",
        return_value={
            "available_topics": [
                "vehicle_local_position",
                "vehicle_local_position_setpoint",
            ],
            "parameters": {"SYS_AUTOSTART": 4001},
        },
    ) as parse_inventory, patch.object(
        runner,
        "build_basic_timeline",
        return_value=[{"event": "initial_value"}],
    ) as build_timeline, patch.object(
        runner,
        "infer_control_surface",
        return_value={"vehicle_type": "fixed_wing"},
    ) as infer_surface, patch.object(
        runner,
        "parse_mission_file",
        return_value={"mission_file": str(mission_path), "items": []},
    ) as parse_mission, patch.object(
        runner,
        "discover_source_mechanisms",
        return_value=source_candidate_set,
    ) as source_discovery, patch.object(
        runner,
        "write_resolved_mechanisms_to_cache",
        return_value=[],
    ) as write_cache, patch.object(
        runner,
        "evaluate_candidate_applicability",
        return_value=applicability,
    ) as evaluate_applicability, patch.object(
        runner,
        "evaluate_candidate_log_signature",
        return_value=evaluation,
    ) as evaluate_signature:
        result = asyncio.run(
            runner.analyze_flight_log(
                log_path=str(log_path),
                mission_path=str(mission_path),
                source_path=str(source_path),
                output_dir=str(output_dir),
                dev_log_root=str(dev_log_root),
                dev_run_id="web_run_001",
                user_question="Why did it loiter before the waypoint?",
                max_candidates=1,
                mechanism_cache_dir=str(tmp_path / "mechanism_cache"),
                force_mechanism_refresh=True,
            )
        )

    source_discovery.assert_called_once()
    write_cache.assert_called_once()
    evaluate_applicability.assert_called_once()
    evaluate_signature.assert_called_once()
    parse_inventory.assert_called_once_with(log_path)
    build_timeline.assert_called_once_with(log_path)
    infer_surface.assert_called_once_with(log_path, source_path)
    parse_mission.assert_called_once_with(mission_path)

    assert result is final_report
    assert [item["agent"] for item in captured] == [
        runner.question_intent_agent,
        runner.final_report_agent,
    ]
    assert [item["max_turns"] for item in captured] == [2, 4]
    assert all(item["hooks"] is not None for item in captured)
    assert captured[0]["input"]["user_question"] == "Why did it loiter before the waypoint?"
    assert "log_inventory" not in captured[0]["input"]
    discovery_args = source_discovery.call_args.args
    assert discovery_args[0] == source_path
    assert discovery_args[1] is question_intent
    assert discovery_args[2].parameters == {"SYS_AUTOSTART": 4001}
    assert discovery_args[2].available_topics == [
        "vehicle_local_position",
        "vehicle_local_position_setpoint",
    ]
    assert list(captured[1]["input"].keys()) == ["verified_mechanism_results"]
    assert captured[1]["input"]["verified_mechanism_results"][0]["final_confidence"] == "medium"
    assert "raw" not in captured[1]["input"]["verified_mechanism_results"][0]["evaluation"]
    assert "airframe_context" not in captured[1]["input"]
    assert "question_intent" not in captured[1]["input"]
    assert "mechanism_cache" not in captured[1]["input"]
    assert final_report.airframe_summary == "vehicle_type=fixed_wing; SYS_AUTOSTART=4001"
    assert final_report.question_intent_summary == "fixed_wing_altitude: explain altitude drop"
    run_dir = dev_log_root / "web_run_001"
    assert output_dir.is_dir()
    assert run_dir.is_dir()
    assert (run_dir / "run_events.jsonl").is_file()
    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["runner_version"] == "v3_mechanism_first"
    assert metadata["report_path"] == str(output_dir / "report.json")


def test_shape_report_evidence_moves_unresolved_items_out_of_contradictions(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Cone helper mechanism",
        summary="Source formula depends on helper translation.",
        source_refs=[],
    )
    applicability = runner.ApplicabilityResult(
        candidate_name=candidate.name,
        applicable=True,
    )
    evaluation = runner.SignatureEvaluation(
        candidate_name=candidate.name,
        verdict="unresolved",
        confidence_ceiling="unresolved",
        evidence=["vehicle_status.vehicle_type matched rotary-wing condition."],
        contradictions=[],
        check_results=[
            {
                "type": "derived_expression",
                "status": "unresolved",
                "message": "cannot evaluate helper source_helper: helper body uses unsupported runtime state",
            }
        ],
    )
    report = runner.FlightLogReport(
        airframe_summary="placeholder",
        question_intent_summary="placeholder",
        ranked_hypotheses=[
            runner.HypothesisReportItem(
                title="Cone helper mechanism",
                known_px4_mechanism=candidate.name,
                mechanism="Source helper path.",
                source_refs=[],
                expected_logged_signature=[],
                applicability=runner.ApplicabilityReport(applicable=True),
                evidence=[],
                contradicting_evidence=[
                    "Direct numeric verification was unresolved.",
                    "The vehicle type contradicted the cone branch.",
                ],
                exclusion_checks=[],
                numeric_checks=[],
                confidence="low",
            )
        ],
        excluded_mechanisms=[],
        confirmed=[],
        unconfirmed=[candidate.name],
        final_summary="placeholder",
    )

    runner.shape_report_evidence(
        report,
        [
            runner.VerifiedMechanismResult(
                candidate=candidate,
                applicability=applicability,
                evaluation=evaluation,
                final_confidence="unresolved",
            )
        ],
    )

    hypothesis = report.ranked_hypotheses[0]
    assert hypothesis.contradicting_evidence == ["The vehicle type contradicted the cone branch."]
    assert hypothesis.unresolved_evidence == [
        "Direct numeric verification was unresolved.",
        "cannot evaluate helper source_helper: helper body uses unsupported runtime state",
    ]


def test_generate_report_plots_updates_complete_v3_plot_specs(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    output_dir = tmp_path / "outputs"
    expected_plot_path = output_dir / "plots" / "altitude_drop.png"

    plot = runner.PlotRef(
        title="Altitude Drop",
        path="",
        purpose="Compare altitude estimate and setpoint during transition.",
        start_s=10.0,
        end_s=20.0,
        signals=[
            "vehicle_local_position.z",
            "vehicle_local_position_setpoint.z",
        ],
        overlays=[
            runner.PlotOverlay(
                start_s=12.5,
                label="transition",
                color="#d55e00",
            )
        ],
    )
    candidate = _sample_mechanism_candidate(runner)
    applicability = _sample_applicability(runner, candidate)
    report = _sample_report(runner, candidate, applicability, plots=[plot])
    ctx = runner.FlightLogContext(
        log_path=log_path,
        mission_path=None,
        source_path=None,
        output_dir=output_dir,
    )

    with patch.object(
        runner,
        "generate_signal_plot_impl",
        return_value={
            "title": "Altitude Drop",
            "path": str(expected_plot_path),
            "purpose": "Compare altitude estimate and setpoint during transition.",
            "window_s": [10.0, 20.0],
            "signals": [
                "vehicle_local_position.z",
                "vehicle_local_position_setpoint.z",
            ],
            "plot_type": "timeseries",
            "overlays": [
                {
                    "start_s": 12.5,
                    "label": "transition",
                    "color": "#d55e00",
                }
            ],
            "missing_signals": [],
            "warnings": [],
        },
    ) as plot_impl:
        result = runner.generate_report_plots(report, ctx, audit_logger=None)

    plot_impl.assert_called_once_with(
        log_path,
        output_dir,
        "Altitude Drop",
        10.0,
        20.0,
        [
            "vehicle_local_position.z",
            "vehicle_local_position_setpoint.z",
        ],
        "Compare altitude estimate and setpoint during transition.",
        plot_type="timeseries",
        bins=50,
        overlays=[
            {
                "start_s": 12.5,
                "label": "transition",
                "color": "#d55e00",
            }
        ],
    )
    assert result.ranked_hypotheses[0].plots[0].path == str(expected_plot_path)
    assert result.ranked_hypotheses[0].plots[0].missing_signals == []
    assert result.ranked_hypotheses[0].plots[0].warnings == []


def test_generate_report_plots_skips_incomplete_v3_plot_specs(tmp_path):
    runner = load_runner(tmp_path)
    candidate = _sample_mechanism_candidate(runner)
    applicability = _sample_applicability(runner, candidate)
    report = _sample_report(
        runner,
        candidate,
        applicability,
        plots=[
            runner.PlotRef(
                title="Incomplete Plot",
                path="",
                purpose="Missing start/end/signals.",
            )
        ],
    )
    ctx = runner.FlightLogContext(
        log_path=tmp_path / "flight.ulg",
        mission_path=None,
        source_path=None,
        output_dir=tmp_path / "outputs",
    )

    with patch.object(runner, "generate_signal_plot_impl") as plot_impl:
        result = runner.generate_report_plots(report, ctx, audit_logger=None)

    plot_impl.assert_not_called()
    assert result.unconfirmed == []


def test_v3_agents_enforce_source_mechanism_context_boundaries(tmp_path):
    runner = load_runner(tmp_path)

    intent_instructions = runner.question_intent_agent.kwargs["instructions"]
    source_discovery_instructions = runner.source_discovery_agent.kwargs["instructions"]
    report_instructions = runner.final_report_agent.kwargs["instructions"]

    assert runner.question_intent_agent.kwargs["model"] == "gpt-5.5"
    assert runner.source_discovery_agent.kwargs["model"] == "gpt-5.5"
    assert runner.final_report_agent.kwargs["model"] == "gpt-5.4"
    assert "Do not use parameter values" in intent_instructions
    assert "Do not perform final dynamic log verification" in source_discovery_instructions
    assert "Candidate drafts must be source-level mechanisms" in source_discovery_instructions
    assert "MechanismVerifier checks" in source_discovery_instructions
    assert "verifiedmechanismresult" in report_instructions.lower()
    assert "Confidence cannot exceed evaluation.confidence_ceiling" in report_instructions
    assert "Do not introduce new mechanisms" in report_instructions


def test_sanitize_mechanism_candidate_contract_separates_parameters_from_signals(tmp_path):
    runner = load_runner(tmp_path)
    candidate = runner.MechanismCandidate(
        name="Candidate",
        summary="summary",
        source_refs=[],
        required_parameters=["SYS_AUTOSTART"],
        required_signals=[
            "vehicle_status.nav_state",
            "MISPLACED_PARAM",
            "position_setpoint_triplet.current.alt",
            "position_setpoint_triplet.current.alt + MISPLACED_PARAM",
            "MISPLACED_PARAM",
        ],
        expected_logged_signature=[
            runner.ExpectedSignatureItem(
                name="parameter",
                description="parameter value",
                signal="MISPLACED_PARAM",
            ),
            runner.ExpectedSignatureItem(
                name="signal",
                description="logged signal",
                signal="vehicle_status.nav_state",
            ),
        ],
    )

    result = runner.sanitize_mechanism_candidate_contract(candidate)

    assert result.required_parameters == ["SYS_AUTOSTART", "MISPLACED_PARAM"]
    assert result.required_signals == [
        "vehicle_status.nav_state",
        "position_setpoint_triplet.current.alt",
    ]
    assert result.expected_logged_signature[0].signal is None
    assert result.expected_logged_signature[1].signal == "vehicle_status.nav_state"


def test_final_report_input_is_compact_and_summaries_are_deterministic(tmp_path):
    runner = load_runner(tmp_path)
    candidate = _sample_mechanism_candidate(runner)
    applicability = _sample_applicability(runner, candidate)
    evaluation = runner.SignatureEvaluation(
        candidate_name=candidate.name,
        verdict="supported",
        confidence_ceiling="medium",
        evidence=["numeric support"],
        contradictions=[],
        check_results=[{"status": "passed", "message": "check passed"}],
        warnings=[],
        raw={"large": "debug payload"},
    )
    verified = runner.VerifiedMechanismResult(
        candidate=candidate,
        applicability=applicability,
        evaluation=evaluation,
        final_confidence="medium",
    )

    payload = runner.build_final_report_input([verified])

    assert list(payload.keys()) == ["verified_mechanism_results"]
    compact = payload["verified_mechanism_results"][0]
    assert compact["candidate"]["name"] == candidate.name
    assert compact["evaluation"]["evidence"] == ["numeric support"]
    assert "raw" not in compact["evaluation"]

    airframe_context = runner.AirframeContext(
        px4_git_hash="abcdef1234567890",
        px4_version="v1.14.3",
        vehicle_type="vtol_standard",
        sys_autostart=13000,
        airframe_name="Test VTOL",
    )
    question_intent = runner.QuestionIntent(
        original_question="Why did it climb?",
        problem_domain="RTL altitude",
        concise_intent="explain RTL climb",
        source_queries=[],
    )
    report = _sample_report(runner, candidate, applicability)

    runner.apply_deterministic_report_summaries(report, airframe_context, question_intent)

    assert report.airframe_summary == (
        "PX4 v1.14.3; git abcdef123456; vehicle_type=vtol_standard; "
        "SYS_AUTOSTART=13000; airframe=Test VTOL"
    )
    assert report.question_intent_summary == "RTL altitude: explain RTL climb"


def test_v3_agent_output_schemas_are_strict_json_compatible():
    from agents import AgentOutputSchema

    import runner

    for output_type in (
        runner.QuestionIntent,
        runner.SourceDiscoveryDecision,
        runner.MechanismCandidateSet,
        runner.FlightLogReport,
    ):
        AgentOutputSchema(output_type)


def test_run_agent_serializes_payload_and_returns_final_output(tmp_path):
    runner = load_runner(tmp_path)
    ctx = runner.FlightLogContext(
        log_path=tmp_path / "flight.ulg",
        mission_path=None,
        source_path=None,
        output_dir=tmp_path / "outputs",
    )
    calls = []

    async def fake_run(agent, input, context, max_turns, hooks=None):
        calls.append(
            {
                "agent": agent,
                "input": json.loads(input),
                "context": context,
                "max_turns": max_turns,
                "hooks": hooks,
            }
        )
        return SimpleNamespace(final_output={"ok": True}, new_items=[])

    runner.Runner.run = fake_run

    result = asyncio.run(
        runner._run_agent(
            audit_logger=None,
            stage_name="question_intent",
            agent=runner.question_intent_agent,
            payload={"user_question": "why"},
            ctx=ctx,
            max_turns=2,
        )
    )

    assert result == {"ok": True}
    assert calls == [
        {
            "agent": runner.question_intent_agent,
            "input": {"user_question": "why"},
            "context": ctx,
            "max_turns": 2,
            "hooks": None,
        }
    ]


def test_run_agent_does_not_retry_errors(tmp_path):
    runner = load_runner(tmp_path)
    ctx = runner.FlightLogContext(
        log_path=tmp_path / "flight.ulg",
        mission_path=None,
        source_path=None,
        output_dir=tmp_path / "outputs",
    )
    calls = []

    async def fake_run(agent, input, context, max_turns, hooks=None):
        calls.append(json.loads(input))
        raise RuntimeError("schema validation failed")

    runner.Runner.run = fake_run

    try:
        asyncio.run(
            runner._run_agent(
                audit_logger=None,
                stage_name="question_intent",
                agent=runner.question_intent_agent,
                payload={"user_question": "why"},
                ctx=ctx,
                max_turns=2,
            )
        )
    except RuntimeError as exc:
        assert "schema validation failed" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert len(calls) == 1


def test_runner_imports_with_real_sdk_function_tool_schema():
    result = subprocess.run(
        [sys.executable, "-c", "import runner; print(runner.analyze_flight_log.__name__)"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "analyze_flight_log" in result.stdout


def test_infer_control_surface_maps_ca_servo_types_to_pwm_outputs(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"

    class FakeULog:
        def __init__(self, path):
            self.path = path
            self.initial_parameters = {
                "CA_SV_CS_COUNT": 3,
                "CA_SV_CS0_TYPE": 5,
                "CA_SV_CS1_TYPE": 6,
                "CA_SV_CS2_TYPE": 4,
                "PWM_MAIN_FUNC1": 201,
                "PWM_MAIN_FUNC2": 202,
                "PWM_AUX_FUNC1": 203,
                "FW_AIRSPD_TRIM": 17.0,
            }
            self.data_list = [
                SimpleNamespace(name="actuator_servos", data={}),
            ]

    monkeypatch.setattr(ulog_control_surface, "ULog", FakeULog)

    result = ulog_control_surface.infer_control_surface(log_path)

    assert result["vehicle_type"] == "fixed_wing"
    assert result["confidence"] == "medium"
    assert result["assumed_actuator_mapping"] == {
        "servo_1": {
            "control_surface": "left_elevon",
            "source_parameter": "CA_SV_CS0_TYPE",
            "source_value": 5,
            "output_channels": [
                {
                    "parameter": "PWM_MAIN_FUNC1",
                    "function_id": 201,
                    "function": "servo_1",
                },
            ],
        },
        "servo_2": {
            "control_surface": "right_elevon",
            "source_parameter": "CA_SV_CS1_TYPE",
            "source_value": 6,
            "output_channels": [
                {
                    "parameter": "PWM_MAIN_FUNC2",
                    "function_id": 202,
                    "function": "servo_2",
                },
            ],
        },
        "servo_3": {
            "control_surface": "rudder",
            "source_parameter": "CA_SV_CS2_TYPE",
            "source_value": 4,
            "output_channels": [
                {
                    "parameter": "PWM_AUX_FUNC1",
                    "function_id": 203,
                    "function": "servo_3",
                },
            ],
        },
    }
    assert "ULog contains actuator_servos topic." in result["evidence"]
    assert "CA_SV_CS_COUNT=3" in result["evidence"]
    assert result["warning"] == (
        "Control-surface mapping is inferred from logged PX4 parameters only; "
        "physical wiring and linkage direction are not confirmed."
    )


def test_infer_control_surface_reports_parse_failure(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            raise ValueError("bad log")

    monkeypatch.setattr(ulog_control_surface, "ULog", FakeULog)

    result = ulog_control_surface.infer_control_surface(tmp_path / "flight.ulg")

    assert result == {
        "vehicle_type": "unknown",
        "assumed_actuator_mapping": {},
        "evidence": [],
        "confidence": "low",
        "warning": "Control-surface mapping unavailable: failed to parse ULog: bad log",
    }


def test_runner_infer_control_surface_delegates_to_control_surface_module(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    source_path = tmp_path / "PX4-Autopilot"

    with patch.object(
        runner,
        "infer_control_surface_impl",
        return_value={"vehicle_type": "fixed_wing"},
    ) as infer_impl:
        result = runner.infer_control_surface(log_path, source_path)

    infer_impl.assert_called_once_with(log_path, source_path)
    assert result == {"vehicle_type": "fixed_wing"}


def test_search_source_runs_rg_and_limits_output(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    stdout = "\n".join(f"match-{index}" for index in range(20))

    with patch.object(px4_source.subprocess, "run") as run:
        run.return_value = SimpleNamespace(stdout=stdout)

        result = px4_source.search_source(source_path, "mission_result", max_results=2)

    run.assert_called_once_with(
        [
            "rg",
            "-n",
            "--context",
            "3",
            "mission_result",
            str(source_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result == [
        {
            "query": "mission_result",
            "matches": [f"match-{index}" for index in range(16)],
        }
    ]


def test_search_source_returns_subprocess_errors(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()

    with patch.object(px4_source.subprocess, "run", side_effect=TimeoutError("too slow")):
        result = px4_source.search_source(source_path, "vehicle_status")

    assert result == [{"error": "too slow"}]


def test_read_source_file_returns_requested_line_range(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    file_path = source_path / "src" / "modules" / "navigator" / "mission.cpp"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("line 1\nline 2\nline 3\nline 4\n")

    result = px4_source.read_source_file(
        source_path,
        "src/modules/navigator/mission.cpp",
        start_line=2,
        end_line=3,
    )

    assert result == {
        "file": "src/modules/navigator/mission.cpp",
        "path": str(file_path.resolve()),
        "start_line": 2,
        "end_line": 3,
        "total_lines": 4,
        "truncated": True,
        "lines": [
            {"line": 2, "text": "line 2"},
            {"line": 3, "text": "line 3"},
        ],
    }


def test_read_source_file_rejects_path_traversal(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()

    result = px4_source.read_source_file(source_path, "../secret.txt")

    assert result == {"error": "relative_path escapes the PX4 source path."}


def test_read_source_file_accepts_absolute_path_inside_source_tree(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    file_path = source_path / "src" / "modules" / "navigator" / "mission.cpp"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("line 1\nline 2\n")

    result = px4_source.read_source_file(source_path, str(file_path), start_line=1)

    assert result["file"] == "src/modules/navigator/mission.cpp"
    assert result["path"] == str(file_path.resolve())
    assert result["lines"] == [
        {"line": 1, "text": "line 1"},
        {"line": 2, "text": "line 2"},
    ]


def test_read_source_file_caps_large_requested_ranges(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    file_path = source_path / "src" / "large.cpp"
    file_path.parent.mkdir(parents=True)
    file_path.write_text("\n".join(f"line {index}" for index in range(1, 251)))

    result = px4_source.read_source_file(
        source_path,
        "src/large.cpp",
        start_line=10,
        end_line=250,
    )

    assert result["start_line"] == 10
    assert result["end_line"] == 209
    assert result["total_lines"] == 250
    assert result["truncated"] is True
    assert len(result["lines"]) == 200
    assert result["lines"][0] == {"line": 10, "text": "line 10"}
    assert result["lines"][-1] == {"line": 209, "text": "line 209"}


def test_checkout_px4_source_revision_refuses_dirty_tree(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()

    def fake_git(path, args):
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout=" M src/file.cpp\n", stderr="")
        raise AssertionError(args)

    with patch.object(px4_source, "_git", side_effect=fake_git):
        result = px4_source.checkout_px4_source_revision(source_path, "v1.14.0")

    assert result == {
        "error": "PX4 source tree has local changes; refusing to checkout.",
        "status": [" M src/file.cpp"],
    }


def test_checkout_px4_source_revision_detaches_for_hash_or_tag(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    calls = []

    def fake_git(path, args):
        calls.append(args)
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["rev-parse", "HEAD"]:
            commit = "before\n" if calls.count(args) == 1 else "after\n"
            return SimpleNamespace(returncode=0, stdout=commit, stderr="")
        if args == ["show-ref", "--verify", "--quiet", "refs/heads/v1.14.0"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if args == ["checkout", "--detach", "v1.14.0"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["branch", "--show-current"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(args)

    with patch.object(px4_source, "_git", side_effect=fake_git):
        result = px4_source.checkout_px4_source_revision(source_path, "v1.14.0")

    assert ["checkout", "--detach", "v1.14.0"] in calls
    assert result == {
        "source_path": str(source_path),
        "requested_revision": "v1.14.0",
        "before_commit": "before",
        "after_commit": "after",
        "active_branch": None,
        "checked_out": True,
    }


def test_checkout_px4_source_revision_uses_local_branch(tmp_path):
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    calls = []

    def fake_git(path, args):
        calls.append(args)
        if args == ["rev-parse", "--is-inside-work-tree"]:
            return SimpleNamespace(returncode=0, stdout="true\n", stderr="")
        if args == ["status", "--porcelain"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["rev-parse", "HEAD"]:
            commit = "before\n" if calls.count(args) == 1 else "after\n"
            return SimpleNamespace(returncode=0, stdout=commit, stderr="")
        if args == ["show-ref", "--verify", "--quiet", "refs/heads/main"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["checkout", "main"]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if args == ["branch", "--show-current"]:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")
        raise AssertionError(args)

    with patch.object(px4_source, "_git", side_effect=fake_git):
        result = px4_source.checkout_px4_source_revision(source_path, "main")

    assert ["checkout", "main"] in calls
    assert result["active_branch"] == "main"
    assert result["checked_out"] is True


def test_compute_log_metrics_returns_numeric_and_discrete_signal_metrics(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.path = path
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_attitude",
                    data={
                        "timestamp": [
                            12_000_000,
                            12_500_000,
                            13_000_000,
                            18_000_000,
                            18_500_000,
                        ],
                        "roll": [0.0, 0.1, 0.4, 0.7, 1.0],
                    },
                ),
                SimpleNamespace(
                    name="vehicle_status",
                    data={
                        "timestamp": [
                            12_000_000,
                            13_000_000,
                            14_000_000,
                            18_000_000,
                        ],
                        "nav_state": [3, 3, 4, 4],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_metrics, "ULog", FakeULog)

    result = ulog_metrics.compute_log_metrics(
        tmp_path / "flight.ulg",
        start_s=12.5,
        end_s=18.0,
        signals=["vehicle_attitude.roll", "vehicle_status.nav_state"],
    )

    assert result == {
        "window_s": [12.5, 18.0],
        "signals": {
            "vehicle_attitude.roll": {
                "count": 3,
                "first_time_s": 12.5,
                "last_time_s": 18.0,
                "sample_rate_hz": 0.363636,
                "start": 0.1,
                "end": 0.7,
                "unit": "rad",
                "min": 0.1,
                "max": 0.7,
                "mean": 0.4,
                "median": 0.4,
                "std": 0.244949,
                "delta": 0.6,
            },
            "vehicle_status.nav_state": {
                "count": 3,
                "first_time_s": 13.0,
                "last_time_s": 18.0,
                "sample_rate_hz": 0.4,
                "start": 3,
                "end": 4,
                "unit": None,
                "value_counts": {"3": 1, "4": 2},
                "transitions": [
                    {"time_s": 13.0, "value": 3},
                    {"time_s": 14.0, "value": 4},
                ],
            },
        },
        "missing_signals": [],
        "warnings": [],
    }


def test_compute_log_metrics_reports_missing_signals(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.path = path
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_attitude",
                    data={
                        "timestamp": [1_000_000],
                        "roll": [0.1],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_metrics, "ULog", FakeULog)

    result = ulog_metrics.compute_log_metrics(
        tmp_path / "flight.ulg",
        start_s=1.0,
        end_s=2.0,
        signals=["vehicle_local_position.z"],
    )

    assert result == {
        "window_s": [1.0, 2.0],
        "signals": {},
        "missing_signals": ["vehicle_local_position.z"],
        "warnings": [
            "missing topic for signal 'vehicle_local_position.z': vehicle_local_position",
        ],
    }


def test_evaluate_log_signature_supports_threshold_transition_and_divergence(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"

    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="vtol_vehicle_status",
                    data={
                        "timestamp": [10_000_000, 12_000_000, 14_000_000],
                        "vehicle_vtol_state": [1, 1, 4],
                    },
                ),
                SimpleNamespace(
                    name="vehicle_air_data",
                    data={
                        "timestamp": [12_000_000, 13_000_000, 14_000_000, 15_000_000],
                        "baro_alt_meter": [100.0, 98.0, 94.0, 90.0],
                    },
                ),
                SimpleNamespace(
                    name="tecs_status",
                    data={
                        "timestamp": [12_000_000, 13_000_000, 14_000_000, 15_000_000],
                        "altitude_sp": [100.0, 100.0, 100.0, 100.0],
                        "throttle_sp": [0.9, 0.95, 0.92, 0.91],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        log_path,
        mechanism="FW takeover had insufficient altitude tracking after transition.",
        expected_signature=[
            {"name": "transition", "description": "vtol_vehicle_status.vehicle_vtol_state changes 1->4"},
            {"name": "altitude", "description": "actual altitude drops below TECS altitude setpoint"},
        ],
        candidate_windows=[
            {"name": "transition", "start_s": 10.0, "end_s": 14.5},
            {"name": "post_transition", "start_s": 12.0, "end_s": 15.0},
        ],
        required_signals=[
            "vtol_vehicle_status.vehicle_vtol_state",
            "vehicle_air_data.baro_alt_meter",
            "tecs_status.altitude_sp",
            "tecs_status.throttle_sp",
        ],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "transition_occurs",
                "signal": "vtol_vehicle_status.vehicle_vtol_state",
                "window": "transition",
                "from": 1,
                "to": 4,
                "supports": "Transition to fixed-wing occurred in the selected window.",
            },
            {
                "type": "diverges_from_setpoint",
                "actual": "vehicle_air_data.baro_alt_meter",
                "setpoint": "tecs_status.altitude_sp",
                "window": "post_transition",
                "direction": "below",
                "min_error": 6.0,
                "supports": "Altitude fell materially below the TECS altitude setpoint.",
            },
            {
                "type": "threshold",
                "signal": "tecs_status.throttle_sp",
                "window": "post_transition",
                "metric": "mean",
                "op": ">=",
                "value": 0.8,
                "supports": "TECS commanded high throttle during the altitude loss.",
            },
        ],
    )

    assert result["required_signals"]["missing"] == []
    assert result["confidence"] == "high"
    assert result["confidence_score"] == 1.0
    assert result["evidence"] == [
        "Transition to fixed-wing occurred in the selected window.",
        "Altitude fell materially below the TECS altitude setpoint.",
        "TECS commanded high throttle during the altitude loss.",
    ]
    assert result["contradicting_evidence"] == []
    assert result["unresolved"] == []
    assert result["numeric_checks"][1]["value"]["max_relevant_error"] == 10.0


def test_evaluate_log_signature_normalizes_px4_enum_labels(tmp_path, monkeypatch):
    source_path = tmp_path / "PX4-Autopilot"
    msg_dir = source_path / "msg"
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

    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "vehicle_type": [1, 1],
                    },
                ),
                SimpleNamespace(
                    name="vtol_vehicle_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "vehicle_vtol_state": [3, 4],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)
    monkeypatch.setattr(
        ulog_signature_evaluator,
        "normalize_px4_enum_value",
        lambda signal, value: px4_msg_schema.normalize_px4_enum_value(signal, value, source_path),
    )

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="VTOL state checks use source enum labels.",
        expected_signature=[],
        candidate_windows=[{"name": "event", "start_s": 1.0, "end_s": 2.0}],
        required_signals=[
            "vehicle_status.vehicle_type",
            "vtol_vehicle_status.vehicle_vtol_state",
        ],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "state_equals",
                "signal": "vehicle_status.vehicle_type",
                "window": "event",
                "value": "ROTARY_WING",
            },
            {
                "type": "transition_occurs",
                "signal": "vtol_vehicle_status.vehicle_vtol_state",
                "window": "event",
                "from_value": "MC",
                "to_value": "FW",
            },
            {
                "type": "state_not_equals",
                "signal": "vehicle_status.vehicle_type",
                "window": "event",
                "value": "VEHICLE_TYPE_FIXED_WING",
            },
        ],
    )

    assert result["numeric_checks"][0]["status"] == "passed"
    assert result["numeric_checks"][0]["value"]["target"] == 1
    assert result["numeric_checks"][1]["status"] == "passed"
    assert result["numeric_checks"][1]["value"]["transitions"] == [
        {"time_s": 2.0, "from": 3, "to": 4}
    ]
    assert result["numeric_checks"][2]["status"] == "passed"


def test_evaluate_log_signature_evaluates_derived_expression(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"RTL_RETURN_ALT": 20.0}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "z": [90.0, 95.0],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="RTL derived return altitude should stay above threshold.",
        expected_signature=[],
        candidate_windows=[{"name": "rtl", "start_s": 1.0, "end_s": 2.0}],
        required_signals=[],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "derived_expression",
                "expression": "max(current_alt + RTL_RETURN_ALT, 100)",
                "variables": {"current_alt": "vehicle_local_position.z"},
                "window": "rtl",
                "op": ">=",
                "value": 110.0,
                "mode": "all",
                "supports": "Derived RTL altitude expression stayed above the expected threshold.",
            }
        ],
    )

    check = result["numeric_checks"][0]
    assert check["status"] == "passed"
    assert check["value"]["values"] == [110.0, 115.0]
    assert result["evidence"] == ["Derived RTL altitude expression stayed above the expected threshold."]


def test_evaluate_log_signature_reports_missing_derived_expression_input(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"RTL_RETURN_ALT": 20.0}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={
                        "timestamp": [1_000_000],
                        "z": [90.0],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="RTL derived return altitude references unresolved runtime state.",
        expected_signature=[],
        candidate_windows=[{"name": "rtl", "start_s": 1.0, "end_s": 2.0}],
        required_signals=[],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "derived_expression",
                "expression": "current_alt + terrain_clearance + RTL_RETURN_ALT",
                "variables": {"current_alt": "vehicle_local_position.z"},
                "window": "rtl",
                "op": ">=",
                "value": 110.0,
            }
        ],
    )

    check = result["numeric_checks"][0]
    assert check["status"] == "unresolved"
    assert check["message"] == (
        "cannot evaluate expression 'current_alt + terrain_clearance + RTL_RETURN_ALT': "
        "missing input terrain_clearance"
    )
    assert check["value"]["missing_inputs"] == ["terrain_clearance"]


def test_evaluate_log_signature_reports_untranslated_helper_dependency(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    data={
                        "timestamp": [1_000_000],
                        "lat": [47.0],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="Source expression depends on a helper.",
        expected_signature=[],
        candidate_windows=[{"name": "rtl", "start_s": 1.0, "end_s": 2.0}],
        required_signals=[],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "derived_expression",
                "expression": "helper_result",
                "helper_dependencies": [
                    {
                        "name": "source_helper",
                        "args": ["vehicle_global_position.lat"],
                        "unresolved_reason": "helper body uses unsupported runtime state",
                    }
                ],
                "window": "rtl",
            }
        ],
    )

    check = result["numeric_checks"][0]
    assert check["status"] == "unresolved"
    assert check["message"] == (
        "cannot evaluate helper source_helper: helper body uses unsupported runtime state"
    )
    assert check["value"]["helper_dependency"]["args"] == ["vehicle_global_position.lat"]


def test_evaluate_log_signature_evaluates_lowered_conditional_expression(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {"radius": 10.0, "floor_value": 15.0}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "dist": [4.0, 20.0],
                        "current_value": [5.0, 30.0],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="Source expression lowered from assignment control flow.",
        expected_signature=[],
        candidate_windows=[{"name": "window", "start_s": 1.0, "end_s": 2.0}],
        required_signals=[],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "derived_expression",
                "expression": (
                    "max((distance * 2 if distance <= radius else max(floor_value, radius * 2)), "
                    "current_value)"
                ),
                "variables": {
                    "distance": "vehicle_global_position.dist",
                    "current_value": "vehicle_global_position.current_value",
                },
                "window": "window",
                "op": ">=",
                "value": 8.0,
                "mode": "all",
            }
        ],
    )

    check = result["numeric_checks"][0]
    assert check["status"] == "passed"
    assert check["value"]["values"] == [8.0, 30.0]


def test_evaluate_log_signature_canonicalizes_safe_math_calls_by_rule(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {}
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_global_position",
                    data={
                        "timestamp": [1_000_000],
                        "x": [0.25],
                        "y": [9.0],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="Source expression uses C++ math spellings.",
        expected_signature=[],
        candidate_windows=[{"name": "window", "start_s": 1.0, "end_s": 1.0}],
        required_signals=[],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "derived_expression",
                "expression": "ceilf(std::sqrt(acosf(math::constrain(x, -1.0, 1.0)))) + fminf(y, 5.0)",
                "variables": {
                    "x": "vehicle_global_position.x",
                    "y": "vehicle_global_position.y",
                },
                "window": "window",
                "op": ">=",
                "value": 6.0,
            }
        ],
    )

    check = result["numeric_checks"][0]
    assert check["status"] == "passed"
    assert check["value"]["values"] == [7.0]


def test_evaluate_log_signature_reports_threshold_contradictions_and_missing_signals(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="tecs_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "throttle_sp": [0.2, 0.3],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="TECS commanded maximum throttle.",
        expected_signature=[],
        candidate_windows=[{"name": "event", "start_s": 1.0, "end_s": 2.0}],
        required_signals=["tecs_status.throttle_sp", "vehicle_air_data.baro_alt_meter"],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "threshold",
                "signal": "tecs_status.throttle_sp",
                "window": "event",
                "metric": "mean",
                "op": ">=",
                "value": 0.8,
                "contradicts": "Throttle was not high enough to support the mechanism.",
            }
        ],
    )

    assert result["required_signals"]["present"] == ["tecs_status.throttle_sp"]
    assert result["required_signals"]["missing"] == ["vehicle_air_data.baro_alt_meter"]
    assert result["contradicting_evidence"] == [
        "Throttle was not high enough to support the mechanism.",
    ]
    assert result["unresolved"] == [
        "required signal is missing: vehicle_air_data.baro_alt_meter",
    ]
    assert result["confidence"] == "low"


def test_evaluate_log_signature_supports_threshold_and_setpoint_tracking(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "z": [100.0, 99.0, 98.0],
                    },
                ),
                SimpleNamespace(
                    name="vehicle_local_position_setpoint",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "z": [100.0, 100.0, 100.0],
                    },
                ),
                SimpleNamespace(
                    name="tecs_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "throttle_sp": [0.8, 0.85, 0.9],
                    },
                ),
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="TECS demanded high throttle while altitude tracked setpoint.",
        expected_signature=[],
        candidate_windows=[{"name": "event", "start_s": 1.0, "end_s": 3.0}],
        required_signals=[
            "vehicle_local_position.z",
            "vehicle_local_position_setpoint.z",
            "tecs_status.throttle_sp",
        ],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "threshold",
                "signal": "tecs_status.throttle_sp",
                "window": "event",
                "metric": "mean",
                "op": ">=",
                "value": 0.8,
                "supports": "Throttle demand was high.",
            },
            {
                "type": "tracks_setpoint",
                "actual": "vehicle_local_position.z",
                "setpoint": "vehicle_local_position_setpoint.z",
                "window": "event",
                "max_error": 2.1,
                "supports": "Altitude remained close to setpoint.",
            }
        ],
    )

    assert result["verdict"] == "supported"
    assert result["confidence_ceiling"] == "high"
    assert result["missing_required_signals"] == []
    assert result["evidence"] == [
        "Throttle demand was high.",
        "Altitude remained close to setpoint.",
    ]
    assert result["contradictions"] == []
    assert result["check_results"][1]["value"]["max_abs_error"] == 2.0


def test_evaluate_log_signature_supports_parameter_and_field_checks(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.initial_parameters = {
                "VT_TYPE": 2,
                "RTL_RETURN_ALT": 35.0,
            }
            self.data_list = [
                SimpleNamespace(
                    name="position_setpoint",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "alt": [35.0, 35.2, 34.9],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="RTL selected return altitude.",
        expected_signature=[],
        candidate_windows=[{"name": "rtl", "start_s": 1.0, "end_s": 3.0}],
        required_signals=["position_setpoint.alt"],
        exclusion_checks=[
            {
                "type": "branch_parameter_satisfied",
                "parameter": "VT_TYPE",
                "source_predicate": "_param_vt_type.get() == 2",
                "supports": "VT_TYPE satisfied the source branch.",
            }
        ],
        numeric_checks=[
            {
                "type": "topic_field_present",
                "signal": "position_setpoint.alt",
                "supports": "Position setpoint altitude was logged.",
            },
            {
                "type": "tracks_parameter_value",
                "signal": "position_setpoint.alt",
                "parameter": "RTL_RETURN_ALT",
                "window": "rtl",
                "max_error": 0.25,
                "supports": "Altitude setpoint tracked RTL_RETURN_ALT.",
            },
        ],
    )

    assert result["verdict"] == "supported"
    assert result["missing_required_signals"] == []
    assert result["evidence"] == [
        "Position setpoint altitude was logged.",
        "Altitude setpoint tracked RTL_RETURN_ALT.",
        "VT_TYPE satisfied the source branch.",
    ]
    assert result["check_results"][1]["value"]["max_abs_error"] == 0.2


def test_evaluate_log_signature_treats_flattened_prefix_as_present_only(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="position_setpoint_triplet",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "current.alt": [35.0, 35.2, 34.9],
                        "current.valid": [True, True, True],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="RTL selected a current setpoint.",
        expected_signature=[],
        candidate_windows=[{"name": "rtl", "start_s": 1.0, "end_s": 3.0}],
        required_signals=["position_setpoint_triplet.current"],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "topic_field_present",
                "signal": "position_setpoint_triplet.current",
                "supports": "Current position setpoint struct was logged.",
            },
            {
                "type": "threshold",
                "signal": "position_setpoint_triplet.current",
                "window": "rtl",
                "metric": "mean",
                "op": ">=",
                "value": 30.0,
            },
        ],
    )

    assert result["required_signals"]["present"] == ["position_setpoint_triplet.current"]
    assert result["missing_required_signals"] == []
    assert result["numeric_checks"][0]["status"] == "passed"
    assert result["numeric_checks"][0]["value"]["field_present"] is True
    assert result["numeric_checks"][1]["status"] == "unresolved"
    assert result["numeric_checks"][1]["message"] == "missing field for signal: position_setpoint_triplet.current"


def test_evaluate_log_signature_reports_contradictions_and_missing_signals(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="tecs_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000],
                        "throttle_sp": [0.2, 0.25],
                    },
                )
            ]

    monkeypatch.setattr(ulog_signature_evaluator, "ULog", FakeULog)

    result = ulog_signature_evaluator.evaluate_log_signature(
        tmp_path / "flight.ulg",
        mechanism="High throttle caused altitude loss.",
        expected_signature=[],
        candidate_windows=[{"name": "event", "start_s": 1.0, "end_s": 2.0}],
        required_signals=["tecs_status.throttle_sp", "vehicle_air_data.baro_alt_meter"],
        exclusion_checks=[],
        numeric_checks=[
            {
                "type": "threshold",
                "signal": "tecs_status.throttle_sp",
                "window": "event",
                "metric": "mean",
                "op": ">=",
                "value": 0.8,
                "contradicts": "Throttle demand was low.",
            }
        ],
    )

    assert result["verdict"] == "contradicted"
    assert result["confidence_ceiling"] == "low"
    assert result["required_signals"]["present"] == ["tecs_status.throttle_sp"]
    assert result["missing_required_signals"] == ["vehicle_air_data.baro_alt_meter"]
    assert result["contradictions"] == ["Throttle demand was low."]


def test_resolve_signal_extracts_windowed_numeric_values():
    ulog = SimpleNamespace(
        data_list=[
            SimpleNamespace(
                name="vehicle_local_position",
                data={
                    "timestamp": [1_000_000, 2_000_000, 3_000_000],
                    "x": [0.0, 1.5, 3.0],
                },
            )
        ],
        get_dataset=lambda topic: next(
            dataset
            for dataset in ulog.data_list
            if dataset.name == topic
        ),
    )

    result = ulog_plots.resolve_signal(
        ulog,
        "vehicle_local_position.x",
        start_s=1.5,
        end_s=3.0,
    )

    assert result == {
        "signal": "vehicle_local_position.x",
        "topic": "vehicle_local_position",
        "field": "x",
        "unit": "m",
        "axis_label": "[m]",
        "time_s": [2.0, 3.0],
        "values": [1.5, 3.0],
    }


def test_resolve_signal_converts_flight_review_angle_units_to_degrees():
    ulog = SimpleNamespace(
        data_list=[
            SimpleNamespace(
                name="vehicle_attitude",
                data={
                    "timestamp": [1_000_000, 2_000_000],
                    "roll": [0.0, math.pi / 2],
                },
            )
        ],
        get_dataset=lambda topic: next(
            dataset
            for dataset in ulog.data_list
            if dataset.name == topic
        ),
    )

    result = ulog_plots.resolve_signal(
        ulog,
        "vehicle_attitude.roll",
        start_s=1.0,
        end_s=2.0,
    )

    assert result["axis_label"] == "[deg]"
    assert result["unit"] == "deg"
    assert result["values"] == [0.0, 90.0]


def test_prepare_ulog_for_plotting_adds_flight_review_attitude_fields():
    pitch_rad = math.radians(60.0)
    ulog = SimpleNamespace(
        data_list=[
            SimpleNamespace(
                name="vehicle_attitude",
                data={
                    "timestamp": [1_000_000, 2_000_000],
                    "q[0]": np.array([1.0, math.cos(pitch_rad / 2.0)]),
                    "q[1]": np.array([0.0, 0.0]),
                    "q[2]": np.array([0.0, math.sin(pitch_rad / 2.0)]),
                    "q[3]": np.array([0.0, 0.0]),
                },
            )
        ],
        get_dataset=lambda topic: next(
            dataset
            for dataset in ulog.data_list
            if dataset.name == topic
        ),
    )

    ulog_plots.prepare_ulog_for_plotting(ulog)
    result = ulog_plots.resolve_signal(
        ulog,
        "vehicle_attitude.pitch",
        start_s=1.0,
        end_s=2.0,
    )

    assert result["axis_label"] == "[deg]"
    assert result["unit"] == "deg"
    assert result["values"][0] == 0.0
    assert math.isclose(result["values"][1], 60.0)


def test_resolve_signal_accepts_common_px4_plot_aliases():
    ulog = SimpleNamespace(
        data_list=[
            SimpleNamespace(
                name="airspeed",
                data={
                    "timestamp": [1_000_000, 2_000_000],
                    "true_airspeed_m_s": [11.0, 12.5],
                },
            ),
            SimpleNamespace(
                name="tecs_status",
                data={
                    "timestamp": [1_000_000, 2_000_000],
                    "altitude_sp": [100.0, 101.0],
                },
            ),
        ],
        get_dataset=lambda topic: next(
            dataset
            for dataset in ulog.data_list
            if dataset.name == topic
        ),
    )

    airspeed = ulog_plots.resolve_signal(
        ulog,
        "airspeed.true_airspeed",
        start_s=1.0,
        end_s=2.0,
    )
    altitude_sp = ulog_plots.resolve_signal(
        ulog,
        "tecs_status.hgt_setpoint",
        start_s=1.0,
        end_s=2.0,
    )

    assert airspeed["field"] == "true_airspeed_m_s"
    assert airspeed["values"] == [11.0, 12.5]
    assert altitude_sp["field"] == "altitude_sp"
    assert altitude_sp["values"] == [100.0, 101.0]


def test_align_signals_interpolates_y_values_to_x_timestamps():
    x_signal = {
        "signal": "vehicle_local_position.x",
        "time_s": [1.0, 2.0, 3.0],
        "values": [10.0, 20.0, 30.0],
    }
    y_signal = {
        "signal": "vehicle_local_position.y",
        "time_s": [1.0, 3.0],
        "values": [100.0, 300.0],
    }

    result = ulog_plots.align_signals(x_signal, y_signal)

    assert result == {
        "x_signal": "vehicle_local_position.x",
        "y_signal": "vehicle_local_position.y",
        "x": [10.0, 20.0, 30.0],
        "y": [100.0, 200.0, 300.0],
    }


def test_generate_signal_plot_writes_hist2d_png_with_fake_pyplot(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"
    output_dir = tmp_path / "outputs"

    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "x": [0.0, 1.0, 2.0],
                        "y": [0.0, 2.0, 4.0],
                    },
                )
            ]

        def get_dataset(self, topic):
            return next(dataset for dataset in self.data_list if dataset.name == topic)

    class FakeAxes:
        transAxes = object()

        def __init__(self):
            self.hist2d_calls = []
            self.spans = []
            self.lines = []

        def hist2d(self, x, y, bins):
            self.hist2d_calls.append((x, y, bins))
            return (None, None, None, "image")

        def set_xlabel(self, value):
            self.xlabel = value

        def set_ylabel(self, value):
            self.ylabel = value

        def set_title(self, value):
            self.title = value

        def grid(self, *args, **kwargs):
            pass

        def legend(self):
            pass

        def axvspan(self, start_s, end_s, color, alpha, label):
            self.spans.append((start_s, end_s, color, alpha, label))

        def axvline(self, time_s, color, alpha, linestyle, label):
            self.lines.append((time_s, color, alpha, linestyle, label))

        def get_legend_handles_labels(self):
            return ["handle"], ["label"]

    class FakeFigure:
        def __init__(self, axes):
            self.axes = axes

        def colorbar(self, image, ax, label):
            self.colorbar_args = (image, ax, label)

        def tight_layout(self):
            pass

        def savefig(self, path):
            Path(path).write_bytes(b"fake-png")

    class FakePyplot:
        def __init__(self):
            self.axes = FakeAxes()

        def subplots(self, figsize):
            return FakeFigure(self.axes), self.axes

        def close(self, fig):
            self.closed = fig

    fake_pyplot = FakePyplot()
    monkeypatch.setattr(ulog_plots, "ULog", FakeULog)
    monkeypatch.setattr(ulog_plots, "_load_pyplot", lambda: fake_pyplot)

    result = ulog_plots.generate_signal_plot(
        log_path,
        output_dir,
        title="Position Relationship",
        start_s=1.0,
        end_s=3.0,
        signals=["vehicle_local_position.x", "vehicle_local_position.y"],
        purpose="Compare local x and y.",
        plot_type="hist2d",
        bins=12,
        overlays=[
            {
                "start_s": 1.25,
                "end_s": 2.25,
                "label": "analysis window",
                "color": "#56b4e9",
            }
        ],
    )

    expected_plot_path = output_dir / "plots" / "position_relationship.png"
    assert result == {
        "title": "Position Relationship",
        "path": str(expected_plot_path),
        "purpose": "Compare local x and y.",
        "window_s": [1.0, 3.0],
        "signals": ["vehicle_local_position.x", "vehicle_local_position.y"],
        "plot_type": "hist2d",
        "overlays": [
            {
                "start_s": 1.25,
                "end_s": 2.25,
                "label": "analysis window",
                "color": "#56b4e9",
            }
        ],
        "missing_signals": [],
        "warnings": [],
    }
    assert expected_plot_path.read_bytes() == b"fake-png"
    assert fake_pyplot.axes.hist2d_calls == [([0.0, 1.0, 2.0], [0.0, 2.0, 4.0], 12)]
    assert fake_pyplot.axes.xlabel == "vehicle_local_position.x [m]"
    assert fake_pyplot.axes.ylabel == "vehicle_local_position.y [m]"
    assert fake_pyplot.axes.spans == [(1.25, 2.25, "#56b4e9", 0.12, "analysis window")]


def test_generate_signal_plot_adds_flight_review_style_mode_backgrounds(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"
    output_dir = tmp_path / "outputs"

    class FakeULog:
        def __init__(self, path):
            self.data_list = [
                SimpleNamespace(
                    name="vehicle_local_position",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "x": [0.0, 1.0, 2.0],
                    },
                ),
                SimpleNamespace(
                    name="vehicle_status",
                    data={
                        "timestamp": [1_000_000, 2_000_000, 3_000_000],
                        "nav_state": [3, 3, 4],
                    },
                ),
                SimpleNamespace(
                    name="vtol_vehicle_status",
                    data={
                        "timestamp": [1_000_000, 2_500_000, 3_000_000],
                        "vehicle_vtol_state": [3, 1, 4],
                    },
                ),
            ]

        def get_dataset(self, topic):
            return next(dataset for dataset in self.data_list if dataset.name == topic)

    class FakeAxes:
        transAxes = object()

        def __init__(self):
            self.spans = []

        def plot(self, *args, **kwargs):
            pass

        def set_xlabel(self, value):
            pass

        def set_ylabel(self, value):
            pass

        def set_title(self, value):
            pass

        def grid(self, *args, **kwargs):
            pass

        def legend(self):
            pass

        def axvspan(self, start_s, end_s, **kwargs):
            self.spans.append((start_s, end_s, kwargs))

        def get_legend_handles_labels(self):
            return [], []

    class FakeFigure:
        def tight_layout(self):
            pass

        def savefig(self, path):
            Path(path).write_bytes(b"fake-png")

    class FakePyplot:
        def __init__(self):
            self.axes = FakeAxes()

        def subplots(self, figsize):
            return FakeFigure(), self.axes

        def close(self, fig):
            pass

    fake_pyplot = FakePyplot()
    monkeypatch.setattr(ulog_plots, "ULog", FakeULog)
    monkeypatch.setattr(ulog_plots, "_load_pyplot", lambda: fake_pyplot)

    result = ulog_plots.generate_signal_plot(
        log_path,
        output_dir,
        title="Mode Backgrounds",
        start_s=1.0,
        end_s=3.0,
        signals=["vehicle_local_position.x"],
        purpose="Check automatic background context.",
    )

    background_overlays = [
        overlay
        for overlay in result["overlays"]
        if overlay.get("kind") in {"mode_background", "vtol_background"}
    ]
    assert [
        (overlay["source"], overlay["start_s"], overlay["end_s"], overlay["label"])
        for overlay in background_overlays
    ] == [
        ("vehicle_status.nav_state", 1.0, 3.0, "Mission"),
        ("vtol_vehicle_status.vehicle_vtol_state", 1.0, 2.5, "Multicopter"),
        ("vtol_vehicle_status.vehicle_vtol_state", 2.5, 3.0, "Transition"),
    ]
    assert fake_pyplot.axes.spans[0] == (
        1.0,
        3.0,
        {
            "color": "#6600cc",
            "alpha": 0.08,
            "label": "_Mission",
            "ymin": 0.0,
            "ymax": 1.0,
        },
    )
    assert fake_pyplot.axes.spans[1][2]["ymax"] == 0.14


def test_generate_signal_plot_delegates_to_plot_module(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path)

    expected_plot_path = tmp_path / "outputs" / "plots" / "mission___acceptance.png"
    with patch.object(
        runner,
        "generate_signal_plot_impl",
        return_value={
            "title": "Mission / Acceptance",
            "path": str(expected_plot_path),
            "plot_type": "xy",
            "overlays": [],
        },
    ) as plot_impl:
        result = runner.generate_signal_plot(
            ctx,
            title="Mission / Acceptance",
            start_s=4.0,
            end_s=9.5,
            signals=["mission_result.seq_current", "vehicle_status.nav_state"],
            purpose="Compare mission progress with navigation state.",
            plot_type="xy",
            bins=30,
            overlays=[runner.PlotOverlay(start_s=4.5, end_s=5.5, label="window")],
        )

    plot_impl.assert_called_once_with(
        ctx.context.log_path,
        ctx.context.output_dir,
        "Mission / Acceptance",
        4.0,
        9.5,
        ["mission_result.seq_current", "vehicle_status.nav_state"],
        "Compare mission progress with navigation state.",
        plot_type="xy",
        bins=30,
        overlays=[{"start_s": 4.5, "end_s": 5.5, "label": "window"}],
    )
    assert result == {
        "title": "Mission / Acceptance",
        "path": str(expected_plot_path),
        "plot_type": "xy",
        "overlays": [],
    }
