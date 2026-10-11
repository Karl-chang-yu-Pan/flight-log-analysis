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


def _shell_request(*commands: str, timeout_ms: int | None = None):
    return SimpleNamespace(
        data=SimpleNamespace(
            action=SimpleNamespace(
                commands=list(commands),
                timeout_ms=timeout_ms,
            )
        )
    )


def test_default_max_turns_is_shared_by_api_and_cli(monkeypatch):
    assert analyzer.DEFAULT_MAX_TURNS == 30
    assert (
        analyzer.analyze_flight_log.__kwdefaults__["max_turns"]
        == analyzer.DEFAULT_MAX_TURNS
    )

    monkeypatch.setattr(
        analyzer.sys,
        "argv",
        [
            "analyzer.py",
            "--ulog",
            "flight.ulg",
            "--question",
            "What happened?",
        ],
    )

    assert analyzer.parse_args().max_turns == analyzer.DEFAULT_MAX_TURNS


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
    assert "read_px4_source_file" in instructions
    assert "bounded region" in instructions
    assert "oversized" in instructions


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
    ("first_command", "duplicate_command"),
    [
        (
            "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp",
            "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp",
        ),
        (
            'git -C PX4-Autopilot grep -n -F "term" SNAPSHOT -- src/',
            "git -C PX4-Autopilot grep -n -F 'term' SNAPSHOT -- src/",
        ),
        (
            "git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/",
            "git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/",
        ),
    ],
)
def test_duplicate_snapshot_git_reads_are_suppressed(
    tmp_path,
    monkeypatch,
    first_command,
    duplicate_command,
):
    snapshot = SourceSnapshot(tmp_path / "PX4-Autopilot", "a" * 40)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    process_calls = []

    async def fake_run_process(argv, **_kwargs):
        process_calls.append(argv)
        return b"source result\n", b"", 0, False

    monkeypatch.setattr(executor, "_run_process", fake_run_process)

    result = asyncio.run(
        executor(_shell_request(first_command, duplicate_command))
    )

    assert len(process_calls) == 1
    assert len(result.output) == 2
    assert result.output[0].command == first_command
    assert result.output[0].stdout == "source result\n"
    assert result.output[0].exit_code == 0
    assert result.output[1].command == duplicate_command
    assert (
        result.output[1].stdout
        == analyzer.DUPLICATE_SNAPSHOT_GIT_READ_MESSAGE
    )
    assert result.output[1].stderr == ""
    assert result.output[1].exit_code == 0
    assert result.provider_data["duplicate_snapshot_git_reads_suppressed"] == 1


def test_duplicate_snapshot_git_grep_no_match_preserves_exit_status(
    tmp_path,
    monkeypatch,
):
    snapshot = SourceSnapshot(tmp_path / "PX4-Autopilot", "a" * 40)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    process_calls = []
    command = 'git -C PX4-Autopilot grep -n -F "missing" SNAPSHOT -- src/'

    async def fake_run_process(argv, **_kwargs):
        process_calls.append(argv)
        return b"", b"", 1, False

    monkeypatch.setattr(executor, "_run_process", fake_run_process)

    first = asyncio.run(executor(_shell_request(command)))
    duplicate = asyncio.run(executor(_shell_request(command)))

    assert len(process_calls) == 1
    assert first.output[0].stdout == ""
    assert first.output[0].exit_code == 1
    assert duplicate.output[0].stdout == analyzer.DUPLICATE_SNAPSHOT_GIT_READ_MESSAGE
    assert duplicate.output[0].exit_code == 1
    assert duplicate.provider_data["duplicate_snapshot_git_reads_suppressed"] == 1


@pytest.mark.parametrize(
    ("exit_code", "timed_out"),
    [
        (2, False),
        (124, True),
    ],
)
def test_snapshot_git_failures_and_timeouts_remain_retryable(
    tmp_path,
    monkeypatch,
    exit_code,
    timed_out,
):
    snapshot = SourceSnapshot(tmp_path / "PX4-Autopilot", "a" * 40)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    process_calls = []
    command = "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp"

    async def fake_run_process(argv, **_kwargs):
        process_calls.append(argv)
        return b"", b"read failed", exit_code, timed_out

    monkeypatch.setattr(executor, "_run_process", fake_run_process)

    first = asyncio.run(executor(_shell_request(command)))
    second = asyncio.run(executor(_shell_request(command)))

    assert len(process_calls) == 2
    assert first.output[0].exit_code == exit_code
    assert second.output[0].exit_code == exit_code
    assert second.output[0].stdout == ""
    assert second.provider_data["duplicate_snapshot_git_reads_suppressed"] == 0


def test_workspace_commands_are_never_deduplicated(tmp_path, monkeypatch):
    executor = analyzer.RestrictedShellExecutor(tmp_path)
    process_calls = []
    command = 'python -c "print(1)"'

    async def fake_sandboxed_process(argv, **_kwargs):
        process_calls.append(argv)
        return f"run {len(process_calls)}\n".encode(), b"", 0, False

    monkeypatch.setattr(
        executor,
        "_run_sandboxed_process",
        fake_sandboxed_process,
    )

    result = asyncio.run(executor(_shell_request(command, command)))

    assert len(process_calls) == 2
    assert [item.stdout for item in result.output] == ["run 1\n", "run 2\n"]
    assert result.provider_data["duplicate_snapshot_git_reads_suppressed"] == 0


def test_snapshot_git_read_suppression_is_per_executor(tmp_path, monkeypatch):
    snapshot = SourceSnapshot(tmp_path / "PX4-Autopilot", "a" * 40)
    first_executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    second_executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)
    process_calls = []
    command = "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp"

    async def fake_run_process(argv, **_kwargs):
        process_calls.append(argv)
        return b"source result\n", b"", 0, False

    monkeypatch.setattr(first_executor, "_run_process", fake_run_process)
    monkeypatch.setattr(second_executor, "_run_process", fake_run_process)

    asyncio.run(first_executor(_shell_request(command)))
    asyncio.run(second_executor(_shell_request(command)))

    assert len(process_calls) == 2


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


def test_inventory_write_failure_stops_before_model_request(tmp_path, monkeypatch):
    log_path = tmp_path / "flight.ulg"
    log_path.write_bytes(b"ULog")
    output_dir = tmp_path / "output"
    (output_dir / "work" / "ulog_inventory.json").mkdir(parents=True)
    monkeypatch.setattr(analyzer, "resolve_source_path", lambda _: None)
    monkeypatch.setattr(analyzer, "parse_ulog_inventory", lambda *_: {"warnings": []})
    run = Mock()
    monkeypatch.setattr(analyzer.Runner, "run", run)

    with pytest.raises(IsADirectoryError):
        asyncio.run(analyzer.analyze_flight_log(
            str(log_path), "What happened?", output_dir=str(output_dir),
            dev_log_root=str(tmp_path / "audit"), dev_run_id="write_failure",
        ))

    run.assert_not_called()
    events = [json.loads(line) for line in (
        tmp_path / "audit" / "write_failure" / "run_events.jsonl"
    ).read_text().splitlines()]
    assert events[-1]["event"] == "run.failed"
    assert not (output_dir / "report.json").exists()


def test_model_inventory_preserves_full_data_without_host_source_paths():
    repository = Path("/host/private/PX4-Autopilot")
    original = {
        "source_path": str(repository),
        "source_commit": "a" * 40,
        "airframe": {"source": f"{repository}@{'a' * 40}", "name": "Example"},
        "warnings": [f"Source unavailable: {repository}/module"],
        "logged_messages": [{"message": "long message " * 6000}],
        "parameters": {"ANY_NEW_PARAM": 3},
        "future_metadata": {"nested": [1, 2, 3]},
    }
    before = json.dumps(original)

    sanitized = analyzer._model_inventory(original, repository)

    assert "source_path" not in sanitized
    assert str(repository) not in json.dumps(sanitized)
    assert sanitized["airframe"]["source"] == f"PX4-Autopilot@{'a' * 40}"
    assert sanitized["warnings"] == ["Source unavailable: PX4-Autopilot/module"]
    for key in ("source_commit", "logged_messages", "parameters", "future_metadata"):
        assert sanitized[key] == original[key]
    assert json.dumps(original) == before
    sanitized["future_metadata"]["nested"].append(4)
    assert original["future_metadata"]["nested"] == [1, 2, 3]
    assert analyzer._model_inventory({"source_path": None, "warnings": []}, None) == {
        "warnings": [],
    }


@pytest.mark.parametrize("enable_web_fallback", [True, False])
@pytest.mark.parametrize("source_available", [True, False])
def test_single_agent_analyzer_preserves_shared_report_and_run_contract(
    tmp_path,
    monkeypatch,
    assert_analysis_engine_contract,
    enable_web_fallback,
    source_available,
):
    log_path = tmp_path / "uploads" / "upload_1" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    _git(source_path, "init")
    (source_path / "file.txt").write_text("source\n", encoding="utf-8")
    airframe_dir = source_path / "ROMFS/px4fmu_common/init.d/airframes"
    airframe_dir.mkdir(parents=True)
    (airframe_dir / "777_example").write_text("# @name Example airframe\n", encoding="utf-8")
    logged_commit = _commit(source_path, "source")
    output_dir = tmp_path / "outputs" / "web_run_1"
    dev_log_root = tmp_path / "dev_logs"
    report = _empty_report()
    parse_inventory = Mock(
        return_value={
            "firmware_version": "v1.0.0",
            "git_hash": logged_commit if source_available else None,
            "parameters": {"EXAMPLE_PARAM": 42, "SYS_AUTOSTART": 777},
            "topic_fields": {"vehicle_status": ["timestamp", "nav_state"]},
            "topic_instances": {"vehicle_status": [
                {"multi_id": 0, "fields": ["timestamp", "nav_state"], "sample_count": 12},
                {"multi_id": 1, "fields": ["timestamp", "nav_state"], "sample_count": 7},
            ]},
            "logged_messages": [{"message": "example flight message", "time_s": 2.0}],
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

    async def fake_run(agent, prompt, max_turns, hooks, run_config=None):
        artifact = json.loads((output_dir / "work" / "ulog_inventory.json").read_text())
        assert artifact["parameters"] == {"EXAMPLE_PARAM": 42, "SYS_AUTOSTART": 777}
        if source_available:
            assert artifact["airframe"]["name"] == "Example airframe"
            assert artifact["airframe"]["source"] == f"PX4-Autopilot@{logged_commit}"
            assert parse_inventory.return_value["source_path"] == str(source_path)
        for key in ("topic_fields", "topic_instances", "logged_messages", "firmware_version"):
            assert artifact[key] == parse_inventory.return_value[key]
        assert "source_path" not in artifact
        assert str(source_path) not in json.dumps(artifact)
        assert artifact.get("source_commit") == (logged_commit if source_available else None)
        assert bool(artifact["warnings"]) is (not source_available)
        assert "/work/ulog_inventory.json" in prompt
        assert "/inputs/flight.ulg" in prompt
        assert "EXAMPLE_PARAM" not in prompt
        assert "example flight message" not in prompt
        assert str(source_path) not in prompt
        captured["agent"] = agent
        captured["prompt"] = prompt
        captured["max_turns"] = max_turns
        captured["hooks"] = hooks
        captured["run_config"] = run_config
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
        assert (output_dir / "work" / "ulog_inventory.json").is_file()
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
        captured["preflight_executor"].source_snapshot
        == (SourceSnapshot(source_path, logged_commit) if source_available else None)
    )
    assert captured["preflight_executor"].input_paths == {
        "flight.ulg": log_path.resolve()
    }
    assert source_executor.source_snapshot == (
        SourceSnapshot(source_path, logged_commit) if source_available else None
    )
    assert source_executor.cwd == (output_dir / "work").resolve()
    assert source_executor.input_paths == {"flight.ulg": log_path.resolve()}
    assert captured["agent_kwargs"]["output_type"] is FlightLogReport
    assert captured["agent_kwargs"]["model"] == "gpt-5.6"
    # Default path resolves no explicit provider: Runner.run receives no
    # RunConfig, preserving today's provider behavior.
    assert captured["run_config"] is None
    assert (
        captured["agent_kwargs"]["model_settings"].reasoning.effort
        == "high"
    )
    assert "SNAPSHOT" in captured["agent_kwargs"]["instructions"]
    assert any(
        getattr(tool, "name", None) == "read_px4_source_file"
        for tool in captured["agent_kwargs"]["tools"]
    ) is source_available
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


def _provider_harness(tmp_path, monkeypatch, extra_result_kwargs=None, **run_kwargs):
    """Minimal fakes driving analyze_flight_log for provider-wiring tests."""
    from agents import RunConfig

    log_path = tmp_path / "uploads" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    output_dir = tmp_path / "outputs" / "run"
    dev_log_root = tmp_path / "dev_logs"
    report = _empty_report()
    captured = {}
    parse_inventory = Mock(
        return_value={
            "firmware_version": "v1.0.0",
            "git_hash": None,
            "parameters": {},
            "topic_fields": {},
            "topic_instances": {},
            "logged_messages": [],
            "available_topics": [],
            "missing_topics": [],
            "warnings": [],
        }
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            captured["agent_kwargs"] = kwargs

    class FakeShellTool:
        def __init__(self, **kwargs):
            captured["shell_tool"] = kwargs

    class FakeWebSearchTool:
        def __init__(self, **kwargs):
            captured.setdefault("web_tools", []).append(kwargs)

    async def fake_run(agent, prompt, max_turns, hooks, run_config=None):
        captured["prompt"] = prompt
        captured["run_config"] = run_config
        return SimpleNamespace(
            final_output=report,
            new_items=[],
            context_wrapper=SimpleNamespace(usage=Usage()),
            **(extra_result_kwargs or {}),
        )

    async def fake_preflight(executor):
        return None

    monkeypatch.setattr(analyzer, "parse_ulog_inventory", parse_inventory)
    monkeypatch.setattr(analyzer, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer, "ShellTool", FakeShellTool)
    monkeypatch.setattr(analyzer, "WebSearchTool", FakeWebSearchTool)
    monkeypatch.setattr(analyzer.Runner, "run", fake_run)
    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor, "preflight", fake_preflight
    )
    kwargs = {
        "log_path": str(log_path),
        "source_path": None,
        "output_dir": str(output_dir),
        "dev_log_root": str(dev_log_root),
        "dev_run_id": "provider_run",
        "user_question": "What happened?",
        **run_kwargs,
    }
    result = asyncio.run(analyzer.analyze_flight_log(**kwargs))
    events = [
        json.loads(line)
        for line in (
            dev_log_root / "provider_run" / "run_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    metadata = json.loads(
        (dev_log_root / "provider_run" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    return result, captured, events, metadata, output_dir, dev_log_root


def _served_events(events):
    return [e for e in events if e.get("event") == "model.served"]


def test_default_path_passes_no_run_config_and_records_requested_values(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("FLIGHT_LOG_MODEL_PROVIDER", raising=False)
    monkeypatch.delenv("FLIGHT_LOG_REASONING_EFFORT", raising=False)
    _, captured, events, metadata, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6"
    )
    assert captured["run_config"] is None
    assert captured["agent_kwargs"]["model"] == "gpt-5.6"
    assert (
        captured["agent_kwargs"]["model_settings"].reasoning.effort == "high"
    )
    assert captured["agent_kwargs"]["output_type"] is FlightLogReport
    assert len(captured.get("web_tools", [])) == 1
    assert metadata["model"] == "gpt-5.6"
    assert metadata["requested_model_provider"] == "openai"
    assert metadata["requested_model"] == "gpt-5.6"
    assert metadata["requested_reasoning_effort"] == "high"
    assert metadata["model_transport"] == "responses"
    assert metadata["web_search_requested"] is True
    assert metadata["web_search_effective"] is True
    served = _served_events(events)
    assert len(served) == 1
    assert served[0]["requested_model"] == "gpt-5.6"
    assert served[0]["served_model"] == "unavailable"
    assert served[0]["served_reasoning_effort"] == "unavailable"
    assert metadata["served_model"] == "unavailable"
    assert metadata["served_reasoning_effort"] == "unavailable"


def test_openrouter_chat_forces_web_search_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        enable_web_fallback=True,
    )
    from agents import RunConfig

    assert isinstance(captured["run_config"], RunConfig)
    assert captured.get("web_tools", []) == []
    assert "Web search is disabled" in captured["prompt"]
    assert metadata["requested_model_provider"] == "openrouter"
    assert metadata["model_transport"] == "chat"
    assert metadata["web_search_requested"] is True
    assert metadata["web_search_effective"] is False


def test_openrouter_responses_uses_function_adapter(tmp_path, monkeypatch):
    from agents import FunctionTool
    from agents.models.chatcmpl_converter import Converter

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        model_transport="responses",
        enable_web_fallback=True,
    )
    tools = captured["agent_kwargs"]["tools"]
    function_tools = [tool for tool in tools if isinstance(tool, FunctionTool)]
    assert any(
        tool.name == analyzer.SHELL_FUNCTION_TOOL_NAME
        for tool in function_tools
    )
    assert not any(type(tool).__name__ == "ShellTool" for tool in tools)
    # Every function tool on the OpenRouter Responses list must serialize
    # on the real Chat Completions converter: no hosted ShellTool may
    # remain. (The hosted web-search tool travels the Responses
    # transport, not this converter.)
    converted = [Converter.tool_to_openai(tool) for tool in function_tools]
    assert converted
    assert {entry["type"] for entry in converted} == {"function"}
    assert len(captured.get("web_tools", [])) == 1
    assert metadata["web_search_effective"] is True


def test_openrouter_responses_keeps_web_search(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        model_transport="responses",
        enable_web_fallback=True,
    )
    assert len(captured.get("web_tools", [])) == 1
    assert metadata["web_search_effective"] is True


def test_routed_fixture_records_served_model(tmp_path, monkeypatch):
    class Raw:
        model = "served-router-choice"

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    # served identity differs from the requested router ID
    _, _, events, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        {"raw_responses": [Raw()]},
        model="typesafe/jev-router",
        model_provider="openrouter",
        enable_web_fallback=False,
    )
    served = [e for e in events if e.get("event") == "model.served"]
    assert len(served) == 1
    assert served[0]["requested_model"] == "typesafe/jev-router"
    assert served[0]["served_model"] == "served-router-choice"
    assert metadata["served_model"] == "served-router-choice"
    assert metadata["served_reasoning_effort"] == "unavailable"


def test_openrouter_without_key_fails_before_model_request(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    called = {}

    async def fake_run(*args, **kwargs):
        called["ran"] = True
        raise AssertionError("must not reach the model")

    monkeypatch.setattr(analyzer.Runner, "run", fake_run)
    log_path = tmp_path / "uploads" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        asyncio.run(
            analyzer.analyze_flight_log(
                log_path=str(log_path),
                output_dir=str(tmp_path / "outputs" / "run"),
                dev_log_root=str(tmp_path / "dev_logs"),
                user_question="What happened?",
                model="openai/gpt-5-nano",
                model_provider="openrouter",
            )
        )
    assert "ran" not in called


def test_usage_record_written_for_configured_provider(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, _, _, metadata, _, dev_log_root = _provider_harness(
        tmp_path,
        monkeypatch,
        model="openai/gpt-5-nano",
        model_provider="openrouter",
    )
    usage_path = dev_log_root / "provider_run" / "usage.json"
    assert usage_path.is_file()
    assert json.loads(usage_path.read_text(encoding="utf-8")) is not None
    assert metadata["served_model"] == "unavailable"


def test_openai_chat_transport_forces_web_search_off(tmp_path, monkeypatch):
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="gpt-5.6",
        model_provider="openai",
        model_transport="chat",
        enable_web_fallback=True,
    )
    assert captured.get("web_tools", []) == []
    assert "Web search is disabled" in captured["prompt"]
    assert metadata["model_transport"] == "chat"
    assert metadata["web_search_requested"] is True
    assert metadata["web_search_effective"] is False


def test_provider_side_error_flows_through_existing_failure_path(
    tmp_path, monkeypatch
):
    """Remote errors propagate; resolution must not swallow or reclassify."""
    log_path = tmp_path / "uploads" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    dev_log_root = tmp_path / "dev_logs"
    parse_inventory = Mock(
        return_value={
            "firmware_version": "v1.0.0",
            "git_hash": None,
            "parameters": {},
            "topic_fields": {},
            "topic_instances": {},
            "logged_messages": [],
            "available_topics": [],
            "missing_topics": [],
            "warnings": [],
        }
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    class FakeShellTool:
        def __init__(self, **kwargs):
            pass

    async def fake_run(agent, prompt, max_turns, hooks, run_config=None):
        raise RuntimeError("provider: model not found")

    async def fake_preflight(executor):
        return None

    monkeypatch.setattr(analyzer, "parse_ulog_inventory", parse_inventory)
    monkeypatch.setattr(analyzer, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer, "ShellTool", FakeShellTool)
    monkeypatch.setattr(analyzer.Runner, "run", fake_run)
    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor, "preflight", fake_preflight
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    with pytest.raises(RuntimeError, match="model not found"):
        asyncio.run(
            analyzer.analyze_flight_log(
                log_path=str(log_path),
                source_path=None,
                output_dir=str(tmp_path / "outputs" / "run"),
                dev_log_root=str(dev_log_root),
                dev_run_id="provider_run",
                user_question="What happened?",
                model="openai/does-not-exist",
                model_provider="openrouter",
            )
        )
    events = [
        json.loads(line)
        for line in (
            dev_log_root / "provider_run" / "run_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert events[-1]["event"] == "run.failed"
    assert "model not found" in events[-1]["error"]


def test_cli_exposes_provider_transport_effort_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyzer",
            "--ulog",
            "flight.ulg",
            "--question",
            "q",
            "--model-provider",
            "openrouter",
            "--model-transport",
            "chat",
            "--reasoning-effort",
            "low",
        ],
    )
    args = analyzer.parse_args()
    assert args.model_provider == "openrouter"
    assert args.model_transport == "chat"
    assert args.reasoning_effort == "low"
    monkeypatch.setattr("sys.argv", ["analyzer"])
    with pytest.raises(SystemExit):
        analyzer.parse_args()


def _chat_tools_via_harness(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, _, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="openai/gpt-5-nano",
        model_provider="openrouter",
        enable_web_fallback=True,
    )
    return captured["agent_kwargs"]["tools"]


def test_chat_tool_list_serializes_with_real_sdk_converter(
    tmp_path, monkeypatch
):
    from agents.models.chatcmpl_converter import Converter

    tools = _chat_tools_via_harness(tmp_path, monkeypatch)
    converted = [Converter.tool_to_openai(tool) for tool in tools]
    assert converted, "expected a non-empty chat tool list"
    for entry in converted:
        assert entry["type"] == "function"
        assert entry["function"]["name"]
        assert entry["function"]["parameters"]
    names = {entry["function"]["name"] for entry in converted}
    assert analyzer.SHELL_FUNCTION_TOOL_NAME in names


def test_source_excerpt_tool_serializes_with_real_sdk_converter(tmp_path):
    from agents.models.chatcmpl_converter import Converter

    from flight_log_agent.analysis.source_tools import (
        make_read_px4_source_file_tool,
    )

    tool = make_read_px4_source_file_tool(lambda alias: None)
    entry = Converter.tool_to_openai(tool)
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "read_px4_source_file"


def test_shell_function_adapter_executes_snapshot_pinned_git_read(
    tmp_path,
):
    from agents.tool_context import ToolContext

    from flight_log_agent.px4.source_snapshot import SourceRepository

    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    _git(source_path, "init")
    (source_path / "note.txt").write_text("pinned-content\n", encoding="utf-8")
    commit = _commit(source_path, "snapshot")
    snapshot = SourceRepository(source_path).resolve_snapshot(commit)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor = analyzer.RestrictedShellExecutor(
        work_dir, snapshot, input_paths={}
    )
    tool = analyzer.make_local_shell_function_tool(executor)
    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call_1",
        tool_arguments='{"command": "x"}',
    )
    output = asyncio.run(
        tool.on_invoke_tool(
            ctx,
            '{"command": "git -C PX4-Autopilot show SNAPSHOT:note.txt"}',
        )
    )
    assert "pinned-content" in output
    assert "exit code: 0" in output


def test_shell_function_adapter_rejects_disallowed_program(tmp_path):
    from agents.tool_context import ToolContext

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor = analyzer.RestrictedShellExecutor(
        work_dir, None, input_paths={}
    )
    tool = analyzer.make_local_shell_function_tool(executor)
    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="call_2",
        tool_arguments='{"command": "x"}',
    )
    output = asyncio.run(
        tool.on_invoke_tool(ctx, '{"command": "ls /"}')
    )
    assert "exit code: 126" in output


def test_chat_transport_offline_loop_with_mocked_http(tmp_path, monkeypatch):
    from agents import Usage
    from agents.agent_output import AgentOutputSchema
    from agents.model_settings import ModelSettings
    from agents.models.chatcmpl_converter import Converter
    from agents.models.interface import ModelTracing
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.tool_context import ToolContext
    from openai.types.chat.chat_completion import ChatCompletion, Choice
    from openai.types.chat.chat_completion_message import (
        ChatCompletionMessage,
    )
    from openai.types.chat.chat_completion_message_tool_call import (
        ChatCompletionMessageToolCall,
        Function as ToolFunction,
    )
    from openai.types.completion_usage import CompletionUsage

    from flight_log_agent.analysis.report_validation import validate_report
    from flight_log_agent.px4.source_snapshot import SourceRepository

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    _git(source_path, "init")
    (source_path / "note.txt").write_text("pinned-content\n", encoding="utf-8")
    commit = _commit(source_path, "snapshot")
    snapshot = SourceRepository(source_path).resolve_snapshot(commit)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor = analyzer.RestrictedShellExecutor(
        work_dir, snapshot, input_paths={}
    )
    shell_tool = analyzer.make_local_shell_function_tool(executor)
    tools = [shell_tool]

    # 1. Serialize the complete tool list with the real SDK transport.
    converted = [Converter.tool_to_openai(tool) for tool in tools]
    assert [entry["function"]["name"] for entry in converted] == [
        shell_tool.name
    ]

    report_json = _empty_report().model_dump_json()

    def make_completion(content, tool_calls):
        message = ChatCompletionMessage(
            role="assistant", content=content, tool_calls=tool_calls
        )
        choice = Choice(
            finish_reason="tool_calls" if tool_calls else "stop",
            index=0,
            message=message,
        )
        return ChatCompletion(
            id="chatcmpl-test",
            choices=[choice],
            created=0,
            model="openai/gpt-5-nano",
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=100, completion_tokens=20, total_tokens=120
            ),
        )

    call_args = (
        '{"command": "git -C PX4-Autopilot show SNAPSHOT:note.txt"}'
    )
    responses = [
        make_completion(
            None,
            [
                ChatCompletionMessageToolCall(
                    id="call_1",
                    function=ToolFunction(
                        name=shell_tool.name, arguments=call_args
                    ),
                    type="function",
                )
            ],
        ),
        make_completion(report_json, None),
    ]

    class FakeCompletions:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return responses[len(self.calls) - 1]

    class FakeClient:
        base_url = "https://openrouter.ai/api/v1"

        def __init__(self):
            self.chat = SimpleNamespace(completions=FakeCompletions())

    model = OpenAIChatCompletionsModel(
        model="openai/gpt-5-nano", openai_client=(client := FakeClient())
    )
    settings = ModelSettings()
    schema = AgentOutputSchema(output_type=FlightLogReport)
    first = asyncio.run(
        model.get_response(
            "system",
            "investigate",
            settings,
            tools,
            schema,
            [],
            ModelTracing.DISABLED,
        )
    )
    # 2-3. Receive the function call, execute locally, replay the result.
    function_calls = [
        item
        for item in first.output
        if getattr(item, "type", None) == "function_call"
    ]
    assert len(function_calls) == 1
    assert function_calls[0].name == shell_tool.name
    ctx = ToolContext(
        context=None,
        tool_name=shell_tool.name,
        tool_call_id=function_calls[0].call_id,
        tool_arguments=call_args,
    )
    tool_output = asyncio.run(
        shell_tool.on_invoke_tool(ctx, call_args)
    )
    assert "pinned-content" in tool_output
    replay_input = [
        function_calls[0].model_dump(exclude_none=True),
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": tool_output,
        },
    ]
    converted_replay = Converter.items_to_messages(replay_input)
    assert converted_replay, "expected replayable tool output messages"
    assert converted_replay[-1]["role"] == "tool"

    # 4. Structured report parses and validates; usage captured per request.
    # The follow-up input replays history as response items (as the real
    # run loop does), not as already-converted chat messages.
    history_input = [
        {"role": "user", "content": "investigate"},
        function_calls[0].model_dump(exclude_none=True),
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": tool_output,
        },
    ]
    second = asyncio.run(
        model.get_response(
            "system",
            history_input,
            settings,
            tools,
            schema,
            [],
            ModelTracing.DISABLED,
        )
    )
    first_kwargs, second_kwargs = client.chat.completions.calls
    for kwargs in (first_kwargs, second_kwargs):
        assert kwargs["model"] == "openai/gpt-5-nano"
        assert kwargs["tools"][0]["type"] == "function"
        assert "response_format" in kwargs
    texts = [
        item
        for item in second.output
        if getattr(item, "type", None) == "message"
    ]
    assert texts, "expected a final assistant message"
    def _message_text(item):
        content = item.content
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            return content.get("text", "")
        parts = []
        for part in content or []:
            text = getattr(part, "text", None)
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)

    report = FlightLogReport.model_validate_json(_message_text(texts[0]))
    assert validate_report(report).passed
    total = Usage()
    total.add(first.usage)
    total.add(second.usage)
    assert total.requests == 2
    assert total.input_tokens == 200
    assert total.output_tokens == 40


def test_default_path_sends_no_router_plugins(tmp_path, monkeypatch):
    monkeypatch.delenv("FLIGHT_LOG_ROUTER_COST_TIER", raising=False)
    _, captured, _, _, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6"
    )
    assert captured["agent_kwargs"]["model_settings"].extra_body is None


def test_router_tier_attaches_plugins_only_for_openrouter(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="typesafe/jev-router",
        model_provider="openrouter",
        router_cost_tier="low",
    )
    assert captured["agent_kwargs"]["model_settings"].extra_body == {
        "plugins": [{"id": "jev-router", "cost_tier": "low"}]
    }
    assert metadata["requested_router_cost_tier"] == "low"


def test_router_tier_never_reaches_other_providers(tmp_path, monkeypatch):
    _, captured_openai, _, _, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="gpt-5.6",
        model_provider="openai",
        router_cost_tier="low",
    )
    assert captured_openai["agent_kwargs"]["model_settings"].extra_body is None


def test_router_plugins_reach_transmitted_request_body(tmp_path):
    import asyncio
    import json

    import httpx
    from agents.agent_output import AgentOutputSchema
    from agents.model_settings import ModelSettings
    from agents.models.interface import ModelTracing
    from openai.types.shared.reasoning import Reasoning

    from flight_log_agent.analysis import model_provider as mpm
    from flight_log_agent.models import FlightLogReport

    seen = []

    def handler(request):
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "id": "gen-x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                        "index": 0,
                    }
                ],
                "created": 0,
                "model": "served-x",
                "object": "chat.completion",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
            request=request,
        )

    runtime = mpm.resolve_model_runtime(
        model="typesafe/jev-router",
        model_provider="openrouter",
        router_cost_tier="low",
        openrouter_api_key="repro-key",
        http_transport=httpx.MockTransport(handler),
    )
    model = runtime.run_config.model_provider.get_model("typesafe/jev-router")
    asyncio.run(
        model.get_response(
            "system",
            "investigate",
            ModelSettings(
                reasoning=Reasoning(effort="high"),
                extra_body=mpm.router_plugins_body(runtime.router_cost_tier),
            ),
            [],
            AgentOutputSchema(output_type=FlightLogReport),
            [],
            ModelTracing.DISABLED,
        )
    )
    assert seen[0]["plugins"] == [{"id": "jev-router", "cost_tier": "low"}]
    assert seen[0]["model"] == "typesafe/jev-router"


def test_executor_overflow_limit_saves_full_output_and_cites_path(
    tmp_path, monkeypatch
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor = analyzer.RestrictedShellExecutor(
        work_dir, None, input_paths={}, output_limit=200
    )
    full = "0123456789" * 100

    async def fake_sandboxed_process(argv, **_kwargs):
        return full.encode(), b"", 0, False

    monkeypatch.setattr(
        executor, "_run_sandboxed_process", fake_sandboxed_process
    )
    result = asyncio.run(executor(_shell_request('python -c "print(1)"')))

    shown = result.output[0].stdout
    assert len(shown) < len(full)
    assert "shell_outputs/" in shown
    saved = sorted((work_dir / "shell_outputs").glob("*.txt"))
    assert len(saved) == 1
    assert saved[0].read_text(encoding="utf-8") == full


def test_executor_default_limit_delivers_output_without_overflow_file(
    tmp_path, monkeypatch
):
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    executor = analyzer.RestrictedShellExecutor(
        work_dir, None, input_paths={}
    )
    full = "0123456789" * 500

    async def fake_sandboxed_process(argv, **_kwargs):
        return full.encode(), b"", 0, False

    monkeypatch.setattr(
        executor, "_run_sandboxed_process", fake_sandboxed_process
    )
    result = asyncio.run(executor(_shell_request('python -c "print(1)"')))

    assert result.output[0].stdout == full
    assert not (work_dir / "shell_outputs").exists()


def test_shell_output_limit_rejects_nonpositive_value(tmp_path, monkeypatch):
    log_path = tmp_path / "uploads" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    with pytest.raises(ValueError, match="shell_output_limit"):
        asyncio.run(
            analyzer.analyze_flight_log(
                log_path=str(log_path),
                output_dir=str(tmp_path / "outputs" / "run"),
                dev_log_root=str(tmp_path / "dev_logs"),
                user_question="What happened?",
                shell_output_limit=0,
            )
        )


def test_configured_output_limit_reaches_executor_and_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("FLIGHT_LOG_SHELL_OUTPUT_LIMIT", raising=False)
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6", shell_output_limit=2000
    )
    assert captured["shell_tool"]["executor"].output_limit == 2000
    assert metadata["shell_output_limit"] == 2000


def test_jev_consult_defaults_off_with_no_tool_or_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("FLIGHT_LOG_JEV_CONSULT", raising=False)
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6"
    )
    tool_names = [getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]]
    assert "consult_investigation_direction" not in tool_names
    assert metadata["requested_jev_consult"] is False
    assert "consult_investigation_direction" not in captured["agent_kwargs"]["instructions"]


def test_jev_consult_enabled_attaches_tool_instructions_and_metadata(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, dev_log_root = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6", jev_consult=True
    )
    tools = captured["agent_kwargs"]["tools"]
    consult = next(
        tool
        for tool in tools
        if getattr(tool, "name", None) == "consult_investigation_direction"
    )
    assert "consult_investigation_direction" in captured["agent_kwargs"]["instructions"]
    assert "advisory" in captured["agent_kwargs"]["instructions"].lower()
    assert metadata["requested_jev_consult"] is True


def test_jev_consult_audit_sink_writes_decision_events(tmp_path, monkeypatch):
    import json

    from flight_log_agent.analysis import jev_consult as jc

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    seen = {}
    real_builder = jc.build_jev_consult_tool

    def spy_builder(**kwargs):
        seen.update(kwargs)
        return real_builder(**kwargs)

    monkeypatch.setattr(analyzer, "build_jev_consult_tool", spy_builder)
    _, _, _, _, _, dev_log_root = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6", jev_consult=True
    )
    seen["audit_sink"](
        {
            "selected_action": "compute_derivation",
            "effective_recommendation": "compute_derivation",
            "tool_call_id": "call-9",
        }
    )
    events = [
        json.loads(line)
        for line in (
            dev_log_root / "provider_run" / "run_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    consults = [e for e in events if e.get("event") == "decision.consult"]
    assert len(consults) == 1
    assert consults[0]["selected_action"] == "compute_derivation"
    assert consults[0]["tool_call_id"] == "call-9"


def test_sole_mode_attaches_only_the_sole_selector(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6", jev_consult="sole"
    )
    names = [
        getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]
    ]
    assert "select_and_run_check" in names
    assert "consult_investigation_direction" not in names
    assert captured["agent_kwargs"]["model_settings"].tool_choice is None
    assert metadata["requested_jev_mode"] == "sole"
    assert metadata["requested_jev_consult"] is True


def test_default_path_has_no_sole_selector(tmp_path, monkeypatch):
    monkeypatch.delenv("FLIGHT_LOG_JEV_CONSULT", raising=False)
    _, captured, _, _, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6"
    )
    names = [
        getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]
    ]
    assert "select_and_run_check" not in names


def test_sole_mode_exposes_no_direct_shell_or_source_tools(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, metadata, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="gpt-5.6",
        model_provider="openrouter",
        jev_consult="sole",
    )
    names = [
        getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]
    ]
    assert names == ["select_and_run_check"]
    assert metadata["requested_jev_mode"] == "sole"


def test_sole_mode_keeps_web_search_policy_unchanged(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    _, captured, _, _, _, _ = _provider_harness(
        tmp_path,
        monkeypatch,
        model="gpt-5.6",
        model_provider="openrouter",
        model_transport="responses",
        enable_web_fallback=True,
        jev_consult="sole",
    )
    names = [
        getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]
    ]
    assert "select_and_run_check" in names
    assert len(captured.get("web_tools", [])) == 1


def test_other_modes_keep_direct_execution_tools(tmp_path, monkeypatch):
    from flight_log_agent.analysis import sole_selector as ssm

    _, captured, _, _, _, _ = _provider_harness(
        tmp_path, monkeypatch, model="gpt-5.6"
    )
    names = [
        getattr(tool, "name", None) for tool in captured["agent_kwargs"]["tools"]
    ]
    assert ssm.SOLE_SELECTOR_TOOL_NAME not in names
    assert len(names) >= 1


def test_analyzer_records_source_commit_in_report(tmp_path, monkeypatch):
    import subprocess

    log_path = tmp_path / "uploads" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ULog")
    source_path = tmp_path / "PX4-Autopilot"
    source_path.mkdir()
    subprocess.run(["git", "-C", str(source_path), "init"], check=True, capture_output=True)
    (source_path / "file.txt").write_text("source\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(source_path), "add", "."], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(source_path), "-c", "user.name=Test",
         "-c", "user.email=test@example.com", "commit", "-m", "source"],
        check=True, capture_output=True,
    )
    logged_commit = subprocess.run(
        ["git", "-C", str(source_path), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    output_dir = tmp_path / "outputs" / "run"
    dev_log_root = tmp_path / "dev_logs"
    report = _empty_report()
    parse_inventory = Mock(
        return_value={
            "firmware_version": "v1.0.0",
            "git_hash": logged_commit,
            "parameters": {},
            "topic_fields": {},
            "topic_instances": {},
            "logged_messages": [],
            "available_topics": [],
            "missing_topics": [],
            "warnings": [],
        }
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

    class FakeShellTool:
        def __init__(self, **kwargs):
            pass

    async def fake_run(agent, prompt, max_turns, hooks, run_config=None):
        return SimpleNamespace(
            final_output=report,
            new_items=[],
            context_wrapper=SimpleNamespace(usage=Usage()),
        )

    async def fake_preflight(executor):
        return None

    monkeypatch.setattr(analyzer, "parse_ulog_inventory", parse_inventory)
    monkeypatch.setattr(analyzer, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer, "ShellTool", FakeShellTool)
    monkeypatch.setattr(analyzer.Runner, "run", fake_run)
    monkeypatch.setattr(
        analyzer.RestrictedShellExecutor, "preflight", fake_preflight
    )
    asyncio.run(
        analyzer.analyze_flight_log(
            log_path=str(log_path),
            source_path=str(source_path),
            output_dir=str(output_dir),
            dev_log_root=str(dev_log_root),
            dev_run_id="commit_run",
            user_question="What happened?",
            model="gpt-5.6",
        )
    )
    saved = json.loads(
        (output_dir / "report.json").read_text(encoding="utf-8")
    )
    assert saved["source_commit"] == logged_commit


def test_instructions_request_code_support_and_evidence_plots():
    normalized = " ".join(analyzer.BASE_INSTRUCTIONS.split())
    assert "quote the decisive lines in the reference snippet" in normalized
    assert "Generate one evidence plot per ranked hypothesis" in normalized
    assert "state the reason plainly in the hypothesis evidence" in normalized
