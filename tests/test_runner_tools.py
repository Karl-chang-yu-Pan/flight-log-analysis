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

import ulog_inventory
import ulog_timeline
import ulog_control_surface
import ulog_metrics
import px4_source
import mission_parser
import ulog_plots


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

    agents_stub = types.ModuleType("agents")

    class Agent:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Runner:
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
    agents_stub.WebSearchTool = WebSearchTool
    agents_stub.RunContextWrapper = RunContextWrapper
    agents_stub.function_tool = function_tool

    module_name = f"runner_under_test_{tmp_path.name}"
    runner_path = Path(__file__).resolve().parents[1] / "runner.py"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    module = importlib.util.module_from_spec(spec)

    original_pydantic = sys.modules.get("pydantic")
    original_agents = sys.modules.get("agents")
    sys.modules["pydantic"] = pydantic_stub
    sys.modules["agents"] = agents_stub
    sys.modules[module_name] = module

    try:
        spec.loader.exec_module(module)
    finally:
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

    class FakeULog:
        def __init__(self, path):
            self.path = path
            self.msg_info_dict = {
                "ver_sw_release": b"1.14.0\x00",
                "ver_sw": b"abcdef123456\x00",
            }
            self.initial_parameters = {
                "SYS_AUTOSTART": 4001,
                "NAV_ACC_RAD": 10.0,
                "UNRELATED_PARAM": 99,
            }
            self.start_timestamp = 1_000_000
            self.last_timestamp = 6_500_000
            self.data_list = [
                SimpleNamespace(name="vehicle_status", data={"timestamp": [1], "nav_state": [3]}),
                SimpleNamespace(name="mission_result", data={"timestamp": [1], "seq_current": [1]}),
                SimpleNamespace(name="sensor_combined", data={"timestamp": [1], "gyro_rad[0]": [0.1]}),
            ]
            self.logged_messages = [
                SimpleNamespace(log_level_str="INFO", message="armed"),
                SimpleNamespace(log_level_str="ERROR", message="mission failure"),
            ]
            self.dropouts = [SimpleNamespace(duration=42)]

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)

    result = ulog_inventory.parse_ulog_inventory(log_path)

    assert result == {
        "firmware_version": "1.14.0",
        "git_hash": "abcdef123456",
        "duration_s": 5.5,
        "important_parameters": {
            "NAV_ACC_RAD": 10.0,
            "SYS_AUTOSTART": 4001,
        },
        "available_topics": [
            "mission_result",
            "sensor_combined",
            "vehicle_status",
        ],
        "topic_fields": {
            "mission_result": ["seq_current", "timestamp"],
            "sensor_combined": ["gyro_rad[0]", "timestamp"],
            "vehicle_status": ["nav_state", "timestamp"],
        },
        "warnings": [
            "error: mission failure",
            "dropout: 42 ms",
        ],
        "missing_topics": [
            "vehicle_type",
            "vtol_vehicle_status",
        ],
    }


def test_parse_ulog_inventory_reports_parse_failure(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
            raise ValueError("bad log")

    monkeypatch.setattr(ulog_inventory, "ULog", FakeULog)

    result = ulog_inventory.parse_ulog_inventory(tmp_path / "flight.ulg")

    assert result["important_parameters"] == {}
    assert result["missing_topics"] == ulog_inventory.EXPECTED_TIMELINE_TOPICS
    assert result["warnings"] == ["failed to parse ULog: bad log"]


def test_parse_ulog_inventory_includes_derived_attitude_plot_fields(tmp_path, monkeypatch):
    class FakeULog:
        def __init__(self, path):
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

    with patch.object(
        runner,
        "parse_ulog_inventory_impl",
        return_value={"available_topics": ["vehicle_status"]},
    ) as parse_impl:
        result = runner.parse_ulog_inventory(log_path)

    parse_impl.assert_called_once_with(log_path)
    assert result == {"available_topics": ["vehicle_status"]}


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


def test_analyze_flight_log_v1_sends_user_question_and_context_to_agent(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    mission_path = tmp_path / "mission.plan"
    source_path = tmp_path / "PX4-Autopilot"
    output_dir = tmp_path / "outputs"
    captured = {}

    async def fake_run(agent, input, context, max_turns):
        captured["agent"] = agent
        captured["input"] = json.loads(input)
        captured["context"] = context
        captured["max_turns"] = max_turns
        return SimpleNamespace(final_output={"final_summary": "answered"})

    runner.Runner.run = fake_run

    with patch.object(
        runner,
        "parse_ulog_inventory",
        return_value={"available_topics": ["vehicle_status"]},
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
    ) as parse_mission:
        result = asyncio.run(
            runner.analyze_flight_log_v1(
                log_path=str(log_path),
                mission_path=str(mission_path),
                source_path=str(source_path),
                output_dir=str(output_dir),
                user_question="Why did it loiter before the waypoint?",
            )
        )

    parse_inventory.assert_called_once_with(log_path)
    build_timeline.assert_called_once_with(log_path)
    infer_surface.assert_called_once_with(log_path, source_path)
    parse_mission.assert_called_once_with(mission_path)

    assert result == {"final_summary": "answered"}
    assert captured["agent"] is runner.flight_log_agent
    assert captured["max_turns"] == 20
    assert captured["context"] == runner.FlightLogContext(
        log_path=log_path,
        mission_path=mission_path,
        source_path=source_path,
        output_dir=output_dir,
    )
    assert captured["input"] == {
        "user_question": "Why did it loiter before the waypoint?",
        "log_inventory": {"available_topics": ["vehicle_status"]},
        "flight_timeline": [{"event": "initial_value"}],
        "detected_assumptions": {"vehicle_type": "fixed_wing"},
        "mission_summary": {"mission_file": str(mission_path), "items": []},
        "source_path": str(source_path),
        "suggestions_requested": False,
    }
    assert output_dir.is_dir()


def test_analyze_flight_log_v1_generates_plots_from_hypothesis_specs(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"
    output_dir = tmp_path / "outputs"
    expected_plot_path = output_dir / "plots" / "altitude_drop.png"

    report = runner.FlightLogReport(
        assumption_header="assumptions",
        log_inventory_summary="inventory",
        timeline_summary="timeline",
        relevant_windows=["10-20s"],
        ranked_hypotheses=[
            runner.Hypothesis(
                title="Altitude drop after transition",
                mechanism="Pitch demand changed during transition.",
                evidence=["local z changed"],
                contradicting_evidence=[],
                confidence="medium",
                plots=[
                    runner.PlotRef(
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
                ],
                code_references=[],
            )
        ],
        confirmed=[],
        unconfirmed=[],
        final_summary="summary",
    )

    async def fake_run(agent, input, context, max_turns):
        return SimpleNamespace(final_output=report)

    runner.Runner.run = fake_run

    with patch.object(
        runner,
        "parse_ulog_inventory",
        return_value={"available_topics": ["vehicle_local_position"]},
    ), patch.object(
        runner,
        "build_basic_timeline",
        return_value=[],
    ), patch.object(
        runner,
        "infer_control_surface",
        return_value={},
    ), patch.object(
        runner,
        "parse_mission_file",
        return_value=None,
    ), patch.object(
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
        result = asyncio.run(
            runner.analyze_flight_log_v1(
                log_path=str(log_path),
                output_dir=str(output_dir),
                user_question="Why did altitude drop after transition?",
            )
        )

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
    assert result.unconfirmed == []
    assert json.loads((output_dir / "report.json").read_text())["ranked_hypotheses"][0]["plots"][0]["path"] == str(expected_plot_path)


def test_generate_report_plots_warns_when_hypothesis_has_no_plot_spec(tmp_path):
    runner = load_runner(tmp_path)
    report = runner.FlightLogReport(
        assumption_header="assumptions",
        log_inventory_summary="inventory",
        timeline_summary="timeline",
        relevant_windows=[],
        ranked_hypotheses=[
            runner.Hypothesis(
                title="Unplotted hypothesis",
                mechanism="No complete plot spec was returned.",
                evidence=[],
                contradicting_evidence=[],
                confidence="low",
                plots=[
                    runner.PlotRef(
                        title="Incomplete Plot",
                        path="",
                        purpose="Missing start/end/signals.",
                    )
                ],
                code_references=[],
            )
        ],
        confirmed=[],
        unconfirmed=[],
        final_summary="summary",
    )
    ctx = runner.FlightLogContext(
        log_path=tmp_path / "flight.ulg",
        mission_path=None,
        source_path=None,
        output_dir=tmp_path / "outputs",
    )

    with patch.object(runner, "generate_signal_plot_impl") as plot_impl:
        result = runner.generate_report_plots(report, ctx)

    plot_impl.assert_not_called()
    assert result.unconfirmed == [
        (
            "Plot generation was not attempted for hypothesis "
            "'Unplotted hypothesis' because no complete plot spec was returned."
        )
    ]


def test_flight_log_agent_instructions_require_hypothesis_plots_and_selected_overlays(tmp_path):
    runner = load_runner(tmp_path)

    instructions = runner.flight_log_agent.kwargs["instructions"]

    assert "For every hypothesis, include at least one plots entry" in instructions
    assert "Set path to an empty string" in instructions
    assert "generate the PNG after your final report" in instructions
    assert "Decide which overlays are useful for each hypothesis plot" in instructions
    assert "flight_timeline" in instructions
    assert "compute_log_metrics transitions" in instructions
    assert "avoid unrelated clutter" in instructions


def test_runner_imports_with_real_sdk_function_tool_schema():
    result = subprocess.run(
        [sys.executable, "-c", "import runner; print(runner.flight_log_agent.name)"],
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert "PX4 Flight Log Analyst V1" in result.stdout


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


def test_runner_infer_control_mapping_uses_control_surface_inference(tmp_path):
    runner = load_runner(tmp_path)
    log_path = tmp_path / "flight.ulg"

    with patch.object(
        runner,
        "infer_control_surface",
        return_value={"vehicle_type": "fixed_wing"},
    ) as infer:
        result = runner.infer_control_mapping(log_path, None)

    infer.assert_called_once_with(log_path, None)
    assert result == {"vehicle_type": "fixed_wing"}


def test_search_px4_source_reports_missing_source_path(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path, source_path=None)

    result = runner.search_px4_source(ctx, "NAV_ACC_RAD")

    assert result == [{"error": "No PX4 source path provided."}]


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


def test_runner_search_px4_source_delegates_to_source_module(tmp_path):
    runner = load_runner(tmp_path)
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    ctx = make_ctx(runner, tmp_path, source_path=source_path)

    with patch.object(
        runner,
        "search_source",
        return_value=[{"query": "vehicle_status", "matches": []}],
    ) as search:
        result = runner.search_px4_source(ctx, "vehicle_status", max_results=3)

    search.assert_called_once_with(source_path, "vehicle_status", max_results=3)
    assert result == [{"query": "vehicle_status", "matches": []}]


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


def test_read_px4_source_file_reports_missing_source_path(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path, source_path=None)

    result = runner.read_px4_source_file(ctx, "src/modules/navigator/mission.cpp")

    assert result == {"error": "No PX4 source path provided."}


def test_runner_read_px4_source_file_delegates_to_source_module(tmp_path):
    runner = load_runner(tmp_path)
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    ctx = make_ctx(runner, tmp_path, source_path=source_path)

    with patch.object(
        runner,
        "read_source_file",
        return_value={"file": "src/modules/navigator/mission.cpp"},
    ) as read_file:
        result = runner.read_px4_source_file(
            ctx,
            "src/modules/navigator/mission.cpp",
            start_line=10,
            end_line=20,
        )

    read_file.assert_called_once_with(
        source_path,
        "src/modules/navigator/mission.cpp",
        10,
        20,
    )
    assert result == {"file": "src/modules/navigator/mission.cpp"}


def test_checkout_px4_source_reports_missing_source_path(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path, source_path=None)

    result = runner.checkout_px4_source(ctx, "v1.14.0")

    assert result == {"error": "No PX4 source path provided."}


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


def test_runner_checkout_px4_source_delegates_to_source_module(tmp_path):
    runner = load_runner(tmp_path)
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    ctx = make_ctx(runner, tmp_path, source_path=source_path)

    with patch.object(
        runner,
        "checkout_px4_source_revision",
        return_value={"checked_out": True},
    ) as checkout:
        result = runner.checkout_px4_source(ctx, "main")

    checkout.assert_called_once_with(source_path, "main")
    assert result == {"checked_out": True}


def test_compute_log_metrics_returns_numeric_and_discrete_signal_metrics(tmp_path, monkeypatch):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path)

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

    result = runner.compute_log_metrics(
        ctx,
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
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path)

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

    result = runner.compute_log_metrics(
        ctx,
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
