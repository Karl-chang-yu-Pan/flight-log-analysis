import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import flight_log_agent.analysis.analyzer as analyzer
from flight_log_agent.models import FlightLogReport
from flight_log_agent.px4.source_snapshot import SourceRepository


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


def test_snapshot_git_commands_read_exact_commit_without_touching_worktree(tmp_path):
    repository_path = tmp_path / "PX4-Autopilot"
    repository_path.mkdir()
    _git(repository_path, "init")
    source_file = repository_path / "src" / "module.cpp"
    source_file.parent.mkdir()
    source_file.write_text("int old_value = 1;\n", encoding="utf-8")
    logged_commit = _commit(repository_path, "logged")
    source_file.write_text("int dirty_value = 2;\n", encoding="utf-8")

    snapshot = SourceRepository(repository_path).resolve_snapshot(logged_commit)
    executor = analyzer.RestrictedShellExecutor(tmp_path, snapshot)

    show_command = executor._parse_command(
        "git -C PX4-Autopilot show SNAPSHOT:src/module.cpp"
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


def test_prepare_workspace_uses_analysis_output_without_copying_source(tmp_path):
    log_path = tmp_path / "upload" / "flight.ulg"
    mission_path = tmp_path / "upload" / "mission.plan"
    log_path.parent.mkdir()
    log_path.write_bytes(b"ULog")
    mission_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "outputs" / "web_run"

    work_dir = analyzer.prepare_workspace(log_path, output_dir, mission_path)

    assert work_dir == output_dir.resolve() / "work"
    assert (work_dir / "flight.ulg").resolve() == log_path.resolve()
    assert (work_dir / "mission.plan").resolve() == mission_path.resolve()
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


def test_shell_analyzer_preserves_shared_report_and_run_contract(
    tmp_path,
    monkeypatch,
    assert_analysis_engine_contract,
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
            captured["shell_kwargs"] = kwargs

    async def fake_run(agent, prompt, max_turns, hooks):
        captured["prompt"] = prompt
        captured["max_turns"] = max_turns
        captured["hooks"] = hooks
        return SimpleNamespace(
            final_output=report,
            new_items=[],
            context_wrapper=SimpleNamespace(usage=None),
        )

    monkeypatch.setattr(analyzer, "parse_ulog_inventory", parse_inventory)
    monkeypatch.setattr(analyzer, "Agent", FakeAgent)
    monkeypatch.setattr(analyzer, "ShellTool", FakeShellTool)
    monkeypatch.setattr(analyzer.Runner, "run", fake_run)

    result = asyncio.run(
        analyzer.analyze_flight_log(
            log_path=str(log_path),
            source_path=str(source_path),
            output_dir=str(output_dir),
            dev_log_root=str(dev_log_root),
            dev_run_id="web_run_1",
            user_question="What happened?",
        )
    )

    parse_inventory.assert_called_once_with(log_path, analyzer.SOURCE_UNAVAILABLE)
    assert result is report
    assert captured["agent_kwargs"]["output_type"] is FlightLogReport
    executor = captured["shell_kwargs"]["executor"]
    assert executor.source_snapshot.commit_sha == logged_commit
    assert executor.cwd == (output_dir / "work").resolve()
    assert "SNAPSHOT" in captured["agent_kwargs"]["instructions"]
    assert not (output_dir / "work" / "PX4-Autopilot").exists()
    metadata = json.loads(
        (dev_log_root / "web_run_1" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["runner_version"] == "shell_snapshot_v1"
    assert_analysis_engine_contract(
        report,
        output_dir,
        dev_log_root,
        "web_run_1",
    )
