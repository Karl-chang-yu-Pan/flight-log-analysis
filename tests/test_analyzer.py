import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agents import Usage

import flight_log_agent.analysis.analyzer as analyzer
from flight_log_agent.models import FlightLogReport
from flight_log_agent.px4.source_snapshot import SourceRepository, SourceSnapshot


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _commit(path: Path, message: str) -> str:
    _git(path, "add", ".")
    _git(
        path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-m",
        message,
    )
    return _git(path, "rev-parse", "HEAD")


def _empty_report() -> FlightLogReport:
    return FlightLogReport(
        airframe_summary="unknown",
        question_intent_summary="investigate the requested behavior",
        ranked_hypotheses=[],
        excluded_mechanisms=[],
        confirmed=[],
        unconfirmed=[],
        final_summary="Insufficient evidence for a supported hypothesis.",
    )


def test_base_instructions_define_lean_single_agent_contract():
    instructions = " ".join(analyzer.BASE_INSTRUCTIONS.split())

    assert "sole analyst" in instructions
    assert "complete investigation in this one agent context" in instructions
    assert "Write and run your own Python scripts" in instructions
    assert "do not rely on a fixed mechanism catalog" in instructions
    assert "commit-pinned Git commands" in instructions
    assert "Actively try to falsify the leading explanation" in instructions
    assert "plausible alternatives" in instructions
    assert "use it only after the ULog and exact source" in instructions
    assert "Web results cannot replace flight evidence" in instructions
    assert "lower confidence rather than forcing an answer" in instructions
    assert "source broker" not in instructions


def test_base_instructions_check_completion_before_more_tool_calls():
    instructions = " ".join(analyzer.BASE_INSTRUCTIONS.split())

    assert "Before every additional tool call" in instructions
    assert "could the proposed call materially change" in instructions
    assert "return FlightLogReport immediately" in instructions
    assert (
        "Continue only to resolve a specific material uncertainty"
        in instructions
    )
    assert "artifact housekeeping" in instructions


@pytest.mark.parametrize(
    "removed_instruction",
    [
        "Mandatory behavior:",
        "Inspect the supplied source for the exact code path",
        "compare the firmware/version information",
        "Keep shell output compact",
        "Never print complete time-series arrays",
        "Use compact statistics, transition timestamps, extrema",
        "Do not provide tuning, code changes, or flight-test suggestions",
        "previous ChatGPT sessions",
        "Run one command per shell-tool command",
        "Complete only the assigned evidence stage",
        "execution receipt",
    ],
)
def test_base_instructions_omit_prototype_behavior_rules(removed_instruction):
    assert removed_instruction not in analyzer.BASE_INSTRUCTIONS


def test_snapshot_git_commands_read_exact_commit_without_touching_worktree(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    source_file = repository_path / "src" / "module.cpp"
    source_file.parent.mkdir()
    source_file.write_text("int old_value = 1;\n", encoding="utf-8")
    schema_file = repository_path / "msg" / "Output.msg"
    schema_file.parent.mkdir()
    schema_file.write_text("float32 value\n", encoding="utf-8")
    logged_commit = _commit(repository_path, "logged")
    source_file.write_text("int dirty_value = 2;\n", encoding="utf-8")

    snapshot = SourceRepository(repository_path).resolve_snapshot(logged_commit)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    show_command = executor._parse_command(
        "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp"
    )
    schema_command = executor._parse_command(
        "git -C PX4-Autopilot show SNAPSHOT:msg/Output.msg"
    )
    grep_command = executor._parse_command(
        'git -C PX4-Autopilot grep -n -F "old_value" SNAPSHOT -- src/'
    )
    list_command = executor._parse_command(
        "git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/"
    )

    assert show_command == [
        "git",
        "-C",
        str(repository_path),
        "show",
        f"{logged_commit}:src/module.cpp",
    ]
    assert logged_commit in grep_command
    assert logged_commit in list_command
    assert schema_command == [
        "git",
        "-C",
        str(repository_path),
        "show",
        f"{logged_commit}:msg/Output.msg",
    ]
    assert subprocess.run(
        show_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "int old_value = 1;\n"
    assert "old_value" in subprocess.run(
        grep_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert subprocess.run(
        schema_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "float32 value\n"
    assert subprocess.run(
        list_command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip() == "src/module.cpp"
    assert source_file.read_text(encoding="utf-8") == "int dirty_value = 2;\n"
    assert _git(repository_path, "rev-parse", "HEAD") == logged_commit
    assert _git(repository_path, "status", "--short") == " M src/module.cpp"


@pytest.mark.parametrize(
    "command",
    [
        "git status",
        "git -C PX4-Autopilot checkout SNAPSHOT",
        "git -C PX4-Autopilot show HEAD:src/module.cpp",
        "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp HEAD:src/module.cpp",
        "git -C PX4-Autopilot grep term HEAD SNAPSHOT -- src/",
        "git -C PX4-Autopilot show --output=result SNAPSHOT:src/module.cpp",
    ],
)
def test_snapshot_git_commands_reject_unpinned_or_mutating_access(
    tmp_path,
    command,
):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "file.txt").write_text("value\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    with pytest.raises(PermissionError):
        executor._parse_command(command)


def test_snapshot_git_grep_allows_snapshot_as_literal_pattern(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "file.txt").write_text("SNAPSHOT\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    command = executor._parse_command(
        "git -C PX4-Autopilot grep -n -F SNAPSHOT SNAPSHOT -- ."
    )

    assert command == [
        "git",
        "-C",
        str(repository_path),
        "grep",
        "-n",
        "-F",
        "SNAPSHOT",
        commit_sha,
        "--",
        ".",
    ]


def test_snapshot_git_ls_tree_allows_snapshot_as_path(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "SNAPSHOT").write_text("value\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    command = executor._parse_command(
        "git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- SNAPSHOT"
    )

    assert command == [
        "git",
        "-C",
        str(repository_path),
        "ls-tree",
        "-r",
        "--name-only",
        commit_sha,
        "--",
        "SNAPSHOT",
    ]


def test_snapshot_git_commands_resolve_recorded_submodule_gitlink(tmp_path):
    child = tmp_path / "mavlink"
    child.mkdir()
    _git(child, "init")
    child_file = child / "message_definitions" / "common.xml"
    child_file.parent.mkdir()
    child_file.write_text("<old />\n", encoding="utf-8")
    old_child_commit = _commit(child, "old child")

    parent = tmp_path / "PX4-Autopilot"
    parent.mkdir()
    _git(parent, "init")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(parent),
            "submodule",
            "add",
            str(child),
            "src/modules/mavlink/mavlink",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parent_commit = _commit(parent, "parent")

    child_file.write_text("<new />\n", encoding="utf-8")
    _commit(child, "new child")
    checked_out_submodule = parent / "src/modules/mavlink/mavlink"
    _git(checked_out_submodule, "fetch")
    _git(checked_out_submodule, "checkout", "FETCH_HEAD")

    snapshot = SourceRepository(parent).resolve_snapshot(parent_commit)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    command = executor._parse_command(
        "git -C PX4-Autopilot/src/modules/mavlink/mavlink "
        "show SNAPSHOT:message_definitions/common.xml"
    )

    assert command == [
        "git",
        "-C",
        str(checked_out_submodule),
        "show",
        f"{old_child_commit}:message_definitions/common.xml",
    ]
    assert subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == "<old />\n"


@pytest.mark.parametrize(
    "command",
    [
        "/tmp/python script.py",
        "rg --pre command term .",
        "rg term /tmp",
        "rg term generated/../outside",
        "sed -i '1d' notes.txt",
        "sed -n '1e whoami' notes.txt",
        "sed -n '1,20p' ../notes.txt",
        "find . -delete",
        "find . -exec ls",
        "find /tmp -type f",
    ],
)
def test_workspace_commands_reject_execution_and_external_paths(
    tmp_path,
    command,
):
    executor = analyzer.RestrictedShellExecutor(tmp_path)

    with pytest.raises(PermissionError):
        executor._parse_command(command)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -n term .", ["rg", "-n", "term", "."]),
        (
            "sed -n '1,20p' notes.txt",
            ["sed", "-n", "1,20p", "notes.txt"],
        ),
        (
            "find . -type f -name '*.txt'",
            ["find", ".", "-type", "f", "-name", "*.txt"],
        ),
    ],
)
def test_workspace_commands_allow_read_only_analysis_operations(
    tmp_path,
    command,
    expected,
):
    executor = analyzer.RestrictedShellExecutor(tmp_path)

    assert executor._parse_command(command) == expected


def test_python_sandbox_excludes_source_checkout_and_network(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "file.txt").write_text("value\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)

    output_dir = tmp_path / "outputs" / "web_run"
    work_dir = output_dir / "work"
    plots_dir = output_dir / "plots"
    work_dir.mkdir(parents=True)
    plots_dir.mkdir()
    upload_path = tmp_path / "uploads" / "flight.ulg"
    upload_path.parent.mkdir()
    upload_path.write_bytes(b"ULog")
    executor = analyzer.RestrictedShellExecutor(
        work_dir,
        snapshot,
        input_paths={"flight.ulg": upload_path},
    )

    command = executor._build_sandbox_command(
        ["python", "-c", "print('ok')"]
    )

    assert "--unshare-net" in command
    assert str(repository_path) not in command
    assert ["--bind", str(work_dir), "/work"] == command[
        command.index("--bind") : command.index("--bind") + 3
    ]
    assert [
        "--ro-bind",
        str(upload_path),
        "/inputs/flight.ulg",
    ] == command[
        command.index(str(upload_path)) - 1 : command.index(str(upload_path)) + 2
    ]
    assert "/broker" not in command
    assert "FLIGHT_LOG_SOURCE_BROKER" not in command
    assert "FLIGHT_LOG_PX4_REPOSITORY" not in command
    assert "FLIGHT_LOG_PX4_COMMIT" not in command
    assert "FLIGHT_LOG_ULOG" in command
    assert "/inputs/flight.ulg" in command
    assert command[-3:] == [
        "/venv/bin/python",
        "-c",
        "print('ok')",
    ]


def test_preflight_exercises_read_only_input_mounts(
    tmp_path,
    monkeypatch,
):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "file.txt").write_text("value\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    upload_path = tmp_path / "uploads" / "flight.ulg"
    upload_path.parent.mkdir()
    upload_path.write_bytes(b"ULog")
    mission_path = upload_path.with_name("mission.plan")
    mission_path.write_text("{}", encoding="utf-8")
    executor = analyzer.RestrictedShellExecutor(
        tmp_path,
        snapshot,
        input_paths={
            "flight.ulg": upload_path,
            "mission.plan": mission_path,
        },
    )
    captured = {}

    async def fake_sandboxed_process(argv, *, timeout_s):
        captured["argv"] = argv
        captured["timeout_s"] = timeout_s
        return b"", b"", 0, False

    monkeypatch.setattr(
        executor,
        "_run_sandboxed_process",
        fake_sandboxed_process,
    )

    asyncio.run(executor.preflight())

    assert captured["argv"][:2] == ["python", "-c"]
    assert "FLIGHT_LOG_ULOG" in captured["argv"][2]
    assert "FLIGHT_LOG_MISSION" in captured["argv"][2]
    assert "FLIGHT_LOG_PX4_REPOSITORY" not in captured["argv"][2]
    assert "ls-tree" not in captured["argv"][2]
    assert captured["timeout_s"] == 10


@pytest.mark.skipif(
    os.environ.get("FLIGHT_LOG_RUN_BWRAP_TESTS") != "1",
    reason="set FLIGHT_LOG_RUN_BWRAP_TESTS=1 for the real sandbox smoke",
)
def test_real_python_sandbox_scopes_inputs_and_output_without_source_copy(
    tmp_path,
):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    source_file = repository_path / "src" / "module.cpp"
    source_file.parent.mkdir()
    source_file.write_text("old\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    source_file.write_text("dirty\n", encoding="utf-8")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    output_dir = tmp_path / "outputs" / "web_run"
    work_dir = output_dir / "work"
    plots_dir = output_dir / "plots"
    work_dir.mkdir(parents=True)
    plots_dir.mkdir()
    upload_path = tmp_path / "uploads" / "flight.ulg"
    upload_path.parent.mkdir()
    upload_path.write_bytes(b"ULog")
    executor = analyzer.RestrictedShellExecutor(
        work_dir,
        snapshot,
        input_paths={"flight.ulg": upload_path},
    )
    checkout_literal = json.dumps(str(repository_path))
    script = f"""
import json
import os
from pathlib import Path

input_path = Path(os.environ["FLIGHT_LOG_ULOG"])
try:
    descriptor = os.open(input_path, os.O_WRONLY | os.O_APPEND)
except OSError:
    input_is_read_only = True
else:
    os.close(descriptor)
    input_is_read_only = False
Path("/work/generated.txt").write_text("work", encoding="utf-8")
Path("/plots/generated.txt").write_text("plot", encoding="utf-8")
print(json.dumps({{
    "input": input_path.read_bytes().decode("ascii"),
    "input_is_read_only": input_is_read_only,
    "checkout_visible": Path({checkout_literal}).exists(),
    "source_environment_visible": any(
        name.startswith("FLIGHT_LOG_PX4_") for name in os.environ
    ),
}}))
"""

    stdout, stderr, returncode, timed_out = asyncio.run(
        executor._run_sandboxed_process(
            ["python", "-c", script],
            timeout_s=10,
        )
    )

    assert timed_out is False
    assert returncode == 0, stderr.decode("utf-8", errors="replace")
    result = json.loads(stdout.decode("utf-8"))
    assert result == {
        "input": "ULog",
        "input_is_read_only": True,
        "checkout_visible": False,
        "source_environment_visible": False,
    }
    assert source_file.read_text(encoding="utf-8") == "dirty\n"
    assert (work_dir / "generated.txt").read_text(encoding="utf-8") == "work"
    assert (plots_dir / "generated.txt").read_text(encoding="utf-8") == "plot"


def test_restricted_environment_does_not_expose_source_checkout(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    (repository_path / "file.txt").write_text("value\n", encoding="utf-8")
    commit_sha = _commit(repository_path, "snapshot")
    snapshot = SourceRepository(repository_path).resolve_snapshot(commit_sha)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    environment = executor._restricted_env()

    assert str(repository_path) not in environment.values()
    assert "FLIGHT_LOG_PX4_REPOSITORY" not in environment
    assert "FLIGHT_LOG_PX4_COMMIT" not in environment
    assert "FLIGHT_LOG_SOURCE_BROKER" not in environment
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def test_run_process_cancellation_kills_and_reaps_child(
    tmp_path,
    monkeypatch,
):
    executor = analyzer.RestrictedShellExecutor(tmp_path)
    created = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.killed = False
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            return b"", b""

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.finished.set()

    async def fake_create_subprocess_exec(*argv, **kwargs):
        process = FakeProcess()
        created["process"] = process
        return process

    monkeypatch.setattr(
        analyzer.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    async def exercise():
        task = asyncio.create_task(
            executor._run_process(
                ["fake-process"],
                timeout_s=60,
                cwd=tmp_path,
                env=executor._restricted_env(),
            )
        )
        await asyncio.sleep(0.05)
        task.cancel()
        done, pending = await asyncio.wait({task}, timeout=1)
        assert task in done
        assert not pending
        with pytest.raises(asyncio.CancelledError):
            task.result()

    asyncio.run(exercise())
    assert created["process"].killed is True
    assert created["process"].returncode == -9


def test_run_process_timeout_kills_and_reaps_child(
    tmp_path,
    monkeypatch,
):
    executor = analyzer.RestrictedShellExecutor(tmp_path)
    created = {}

    class FakeProcess:
        def __init__(self):
            self.returncode = None
            self.killed = False
            self.finished = asyncio.Event()

        async def communicate(self):
            await self.finished.wait()
            return b"partial stdout", b"partial stderr"

        def kill(self):
            self.killed = True
            self.returncode = -9
            self.finished.set()

    async def fake_create_subprocess_exec(*argv, **kwargs):
        process = FakeProcess()
        created["process"] = process
        return process

    monkeypatch.setattr(
        analyzer.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    result = asyncio.run(
        executor._run_process(
            ["fake-process"],
            timeout_s=0.01,
            cwd=tmp_path,
            env=executor._restricted_env(),
        )
    )

    assert result == (b"partial stdout", b"partial stderr", 124, True)
    assert created["process"].killed is True
    assert created["process"].returncode == -9


def test_source_alias_resource_bounds_apply_before_submodule_resolution(
    tmp_path,
):
    executor = analyzer.RestrictedShellExecutor(tmp_path)
    oversized_alias = "/".join(
        [analyzer.SOURCE_ALIAS]
        + ["submodule"] * (analyzer.SOURCE_ALIAS_MAX_PARTS + 1)
    )

    with pytest.raises(PermissionError):
        executor._source_for_alias(oversized_alias)


def test_async_submodule_ls_tree_timeout_is_preserved(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "PX4-Autopilot"
    repository.mkdir()
    snapshot = SourceSnapshot(repository, "a" * 40)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    async def fake_run_process(_argv, **_kwargs):
        return b"", b"", 124, True

    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor,
        "_run_process",
        staticmethod(fake_run_process),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            executor._source_for_alias_async(
                "PX4-Autopilot/modules/vendor",
                timeout_s=10,
            )
        )


def test_async_submodule_cat_file_timeout_is_preserved(
    tmp_path,
    monkeypatch,
):
    repository = tmp_path / "PX4-Autopilot"
    submodule = repository / "modules" / "vendor"
    submodule.mkdir(parents=True)
    snapshot = SourceSnapshot(repository, "a" * 40)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    calls = 0

    async def fake_run_process(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return (
                f"160000 commit {'b' * 40}\tmodules/vendor\n".encode(),
                b"",
                0,
                False,
            )
        return b"", b"", 124, True

    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor,
        "_run_process",
        staticmethod(fake_run_process),
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            executor._source_for_alias_async(
                "PX4-Autopilot/modules/vendor",
                timeout_s=10,
            )
        )


def test_prepare_workspace_uses_analysis_output_without_copying_source(tmp_path):
    log_path = tmp_path / "upload" / "flight.ulg"
    mission_path = tmp_path / "upload" / "mission.plan"
    log_path.parent.mkdir()
    log_path.write_bytes(b"ULog")
    mission_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "web_run"

    work_dir = analyzer.prepare_workspace(log_path, output_dir, mission_path)

    assert work_dir == output_dir.resolve() / "work"
    assert not (work_dir / "flight.ulg").exists()
    assert not (work_dir / "mission.plan").exists()
    assert log_path.read_bytes() == b"ULog"
    assert mission_path.read_text(encoding="utf-8") == "{}"
    assert (output_dir / "plots").is_dir()
    assert not (work_dir / "PX4-Autopilot").exists()


def test_prepare_workspace_refuses_to_replace_unmanaged_input(tmp_path):
    log_path = tmp_path / "flight.ulg"
    log_path.write_bytes(b"ULog")
    output_dir = tmp_path / "outputs" / "web_run"
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True)
    (work_dir / "flight.ulg").write_text("keep me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="unmanaged input"):
        analyzer.prepare_workspace(log_path, output_dir)

    assert (work_dir / "flight.ulg").read_text(encoding="utf-8") == "keep me"


def test_report_plot_paths_map_only_to_produced_artifacts(tmp_path):
    plots_dir = tmp_path / "outputs" / "web_run" / "plots"
    plots_dir.mkdir(parents=True)
    produced = plots_dir / "analysis.png"
    produced.write_bytes(b"png")
    kept = SimpleNamespace(path="/plots/analysis.png", warnings=[])
    rejected = SimpleNamespace(path="/plots/not-created.png", warnings=[])
    report = SimpleNamespace(
        ranked_hypotheses=[
            SimpleNamespace(plots=[kept, rejected]),
        ]
    )

    result = analyzer._normalize_report_plot_paths(
        report,
        plots_dir=plots_dir,
    )

    assert result is report
    assert kept.path == str(produced.resolve())
    assert kept.warnings == []
    assert rejected.path == ""
    assert rejected.warnings == [
        "The report referenced a plot that was not produced during this analysis."
    ]


@pytest.mark.parametrize("enable_web_fallback", [True, False])
def test_single_agent_analyzer_preserves_shared_report_and_run_contract(
    tmp_path,
    monkeypatch,
    assert_analysis_engine_contract,
    enable_web_fallback,
):
    log_path = tmp_path / "uploads" / "upload_1" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    _git(source_path, "init")
    (source_path / "file.txt").write_text("source\n", encoding="utf-8")
    logged_commit = _commit(source_path, "source")
    output_dir = tmp_path / "outputs" / "web_run_1"
    dev_log_root = tmp_path / "dev_logs"
    report = _empty_report()
    parse_inventory = Mock(
        return_value={
            "firmware_version": "v1.0.0",
            "git_hash": logged_commit,
            "parameters": {},
            "available_topics": ["vehicle_status"],
            "missing_topics": [],
            "warnings": [],
        }
    )
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

    class FakeShellTool:
        def __init__(self, **kwargs):
            self.executor = kwargs["executor"]
            captured["shell_tool"] = self

    class FakeWebSearchTool:
        def __init__(self, **kwargs):
            captured.setdefault("web_tools", []).append(kwargs)

    async def fake_run(agent, prompt, max_turns, hooks):
        captured["agent"] = agent
        captured["prompt"] = prompt
        captured["max_turns"] = max_turns
        captured["hooks"] = hooks
        return SimpleNamespace(
            final_output=report,
            new_items=[],
            context_wrapper=SimpleNamespace(usage=Usage()),
        )

    monkeypatch.setattr(analyzer, "parse_ulog_inventory", parse_inventory)
    monkeypatch.setattr(analyzer, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer, "ShellTool", FakeShellTool)
    monkeypatch.setattr(analyzer, "WebSearchTool", FakeWebSearchTool)
    monkeypatch.setattr(analyzer.Runner, "run", fake_run)

    async def fake_preflight(executor):
        captured["preflight_executor"] = executor
        return None

    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor,
        "preflight",
        fake_preflight,
    )

    result = asyncio.run(
        analyzer.analyze_flight_log(
            log_path=str(log_path),
            source_path=str(source_path),
            output_dir=str(output_dir),
            dev_log_root=str(dev_log_root),
            dev_run_id="web_run_1",
            user_question="What happened?",
            model="gpt-5.6",
            max_turns=30,
            max_total_requests=12,
            project_instructions="Keep this project context.",
            enable_web_fallback=enable_web_fallback,
        )
    )

    parse_inventory.assert_called_once_with(log_path, analyzer.SOURCE_UNAVAILABLE)
    assert result is report
    source_executor = captured["shell_tool"].executor
    assert (
        captured["preflight_executor"].source_snapshot.commit_sha
        == logged_commit
    )
    assert captured["preflight_executor"].input_paths == {
        "flight.ulg": log_path.resolve()
    }
    assert source_executor.source_snapshot.commit_sha == logged_commit
    assert source_executor.cwd == (output_dir / "work").resolve()
    assert source_executor.input_paths == {"flight.ulg": log_path.resolve()}
    assert captured["agent_kwargs"]["output_type"] is FlightLogReport
    assert captured["agent_kwargs"]["model"] == "gpt-5.6"
    assert (
        captured["agent_kwargs"]["model_settings"].reasoning.effort
        == "high"
    )
    assert "SNAPSHOT" in captured["agent_kwargs"]["instructions"]
    assert "Keep this project context." in captured["agent_kwargs"]["instructions"]
    assert captured["max_turns"] == 12
    assert "What happened?" in captured["prompt"]
    assert "try to falsify it" in captured["prompt"]
    assert (
        "Web search is available" in captured["prompt"]
    ) is enable_web_fallback
    assert len(captured.get("web_tools", [])) == int(enable_web_fallback)
    if enable_web_fallback:
        assert captured["web_tools"] == [
            {
                "search_context_size": "medium",
                "external_web_access": True,
            }
        ]
    assert not (output_dir / "work" / "PX4-Autopilot").exists()
    metadata = json.loads(
        (dev_log_root / "web_run_1" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["runner_version"] == "shell_snapshot_v1"
    assert metadata["analysis_architecture"] == "single_agent_v1"
    assert metadata["model"] == "gpt-5.6"
    assert metadata["reasoning_effort"] == "high"
    assert metadata["max_turns"] == 12
    assert metadata["max_total_requests"] == 12
    assert metadata["enable_web_fallback"] is enable_web_fallback
    assert_analysis_engine_contract(
        report,
        output_dir,
        dev_log_root,
        "web_run_1",
    )
