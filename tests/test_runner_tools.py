import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_runner(tmp_path: Path):
    """Load runner.py with SDK stubs so tool unit tests stay local-only."""
    pydantic_stub = types.ModuleType("pydantic")

    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump_json(self, indent=None):
            return "{}"

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


def test_search_px4_source_reports_missing_source_path(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path, source_path=None)

    result = runner.search_px4_source(ctx, "NAV_ACC_RAD")

    assert result == [{"error": "No PX4 source path provided."}]


def test_search_px4_source_runs_rg_and_limits_output(tmp_path):
    runner = load_runner(tmp_path)
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    ctx = make_ctx(runner, tmp_path, source_path=source_path)
    stdout = "\n".join(f"match-{index}" for index in range(20))

    with patch.object(runner.subprocess, "run") as run:
        run.return_value = SimpleNamespace(stdout=stdout)

        result = runner.search_px4_source(ctx, "mission_result", max_results=2)

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


def test_search_px4_source_returns_subprocess_errors(tmp_path):
    runner = load_runner(tmp_path)
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    ctx = make_ctx(runner, tmp_path, source_path=source_path)

    with patch.object(runner.subprocess, "run", side_effect=TimeoutError("too slow")):
        result = runner.search_px4_source(ctx, "vehicle_status")

    assert result == [{"error": "too slow"}]


def test_compute_log_metrics_returns_requested_window_and_signals(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path)

    result = runner.compute_log_metrics(
        ctx,
        start_s=12.5,
        end_s=18.0,
        signals=["vehicle_attitude.roll", "vehicle_local_position.z"],
    )

    assert result["window_s"] == [12.5, 18.0]
    assert result["signals"] == [
        "vehicle_attitude.roll",
        "vehicle_local_position.z",
    ]
    assert result["metrics"] == {"TODO": "replace with real metrics"}


def test_generate_signal_plot_returns_stable_plot_reference(tmp_path):
    runner = load_runner(tmp_path)
    ctx = make_ctx(runner, tmp_path)

    result = runner.generate_signal_plot(
        ctx,
        title="Mission / Acceptance",
        start_s=4.0,
        end_s=9.5,
        signals=["mission_result.seq_current", "vehicle_status.nav_state"],
        purpose="Compare mission progress with navigation state.",
    )

    expected_plot_path = tmp_path / "outputs" / "plots" / "mission___acceptance.png"
    assert result == {
        "title": "Mission / Acceptance",
        "path": str(expected_plot_path),
        "purpose": "Compare mission progress with navigation state.",
        "window_s": [4.0, 9.5],
        "signals": ["mission_result.seq_current", "vehicle_status.nav_state"],
    }
    assert expected_plot_path.parent.is_dir()
