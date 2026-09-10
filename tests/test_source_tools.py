import asyncio
import json
import subprocess
from pathlib import Path

from agents.tool_context import ToolContext

from flight_log_agent.analysis.analyzer import RestrictedShellExecutor
from flight_log_agent.analysis.source_tools import (
    MAX_SOURCE_EXCERPT_CHARS,
    make_read_px4_source_file_tool,
)
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


def _invoke(tool, **arguments):
    encoded = json.dumps(arguments)
    context = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id="source_call",
        tool_arguments=encoded,
    )
    output = asyncio.run(tool.on_invoke_tool(context, encoded))
    return json.loads(output)


def _source_tool(tmp_path: Path, lines: list[str]):
    repository = tmp_path / "PX4-Autopilot"
    repository.mkdir()
    _git(repository, "init")
    source_file = repository / "src" / "module.cpp"
    source_file.parent.mkdir()
    source_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    commit = _commit(repository, "snapshot")
    snapshot = SourceRepository(repository).resolve_snapshot(commit)
    executor = RestrictedShellExecutor(tmp_path, snapshot)
    tool = make_read_px4_source_file_tool(executor.resolve_source_alias)
    return repository, source_file, commit, tool


def test_source_line_tool_reads_bounded_committed_lines_without_checkout(
    tmp_path,
):
    old_lines = [f"committed line {index}" for index in range(1, 251)]
    repository, source_file, commit, tool = _source_tool(tmp_path, old_lines)
    source_file.write_text("dirty checkout\n", encoding="utf-8")

    result = _invoke(
        tool,
        repository="PX4-Autopilot",
        path="src/module.cpp",
        start_line=10,
        end_line=250,
    )

    assert result["file"] == "PX4-Autopilot:src/module.cpp"
    assert result["range"] == [10, 209]
    assert result["total_lines"] == 250
    assert result["next_start_line"] == 210
    assert result["text"].splitlines()[0] == "10: committed line 10"
    assert result["text"].splitlines()[-1] == "209: committed line 209"
    assert "dirty checkout" not in result["text"]
    assert str(repository) not in json.dumps(result)
    assert commit not in json.dumps(result)
    assert source_file.read_text(encoding="utf-8") == "dirty checkout\n"
    assert _git(repository, "status", "--short") == " M src/module.cpp"


def test_source_line_tool_caps_text_on_line_boundaries(tmp_path):
    lines = [f"line {index} " + ("x" * 1000) for index in range(1, 31)]
    _repository, _source_file, _commit_sha, tool = _source_tool(tmp_path, lines)

    result = _invoke(
        tool,
        repository="PX4-Autopilot",
        path="src/module.cpp",
        start_line=1,
        end_line=30,
    )

    assert len(result["text"]) <= MAX_SOURCE_EXCERPT_CHARS
    assert result["range"][0] == 1
    assert result["range"][1] < 30
    assert result["next_start_line"] == result["range"][1] + 1


def test_source_line_tool_resolves_recorded_submodule_commit(tmp_path):
    child = tmp_path / "mavlink"
    child.mkdir()
    _git(child, "init")
    child_file = child / "message_definitions" / "common.xml"
    child_file.parent.mkdir()
    child_file.write_text("old one\nold two\n", encoding="utf-8")
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

    child_file.write_text("new one\nnew two\n", encoding="utf-8")
    _commit(child, "new child")
    checked_out_child = parent / "src/modules/mavlink/mavlink"
    _git(checked_out_child, "fetch")
    _git(checked_out_child, "checkout", "FETCH_HEAD")

    snapshot = SourceRepository(parent).resolve_snapshot(parent_commit)
    executor = RestrictedShellExecutor(tmp_path, snapshot)
    tool = make_read_px4_source_file_tool(executor.resolve_source_alias)

    result = _invoke(
        tool,
        repository="PX4-Autopilot/src/modules/mavlink/mavlink",
        path="message_definitions/common.xml",
        start_line=2,
        end_line=2,
    )

    assert result == {
        "file": (
            "PX4-Autopilot/src/modules/mavlink/mavlink:"
            "message_definitions/common.xml"
        ),
        "range": [2, 2],
        "total_lines": 2,
        "next_start_line": None,
        "text": "2: old two",
    }
    assert old_child_commit != _git(checked_out_child, "rev-parse", "HEAD")


def test_source_line_tool_suppresses_only_exact_successful_ranges(tmp_path):
    _repository, _source_file, _commit_sha, tool = _source_tool(
        tmp_path,
        ["one", "two", "three"],
    )
    common = {
        "repository": "PX4-Autopilot",
        "path": "src/module.cpp",
        "start_line": 1,
    }

    first = _invoke(tool, **common, end_line=2)
    duplicate = _invoke(tool, **common, end_line=2)
    adjacent = _invoke(
        tool,
        repository="PX4-Autopilot",
        path="src/module.cpp",
        start_line=2,
        end_line=3,
    )

    assert first["text"] == "1: one\n2: two"
    assert first["next_start_line"] is None
    assert duplicate["duplicate"] is True
    assert "suppressed" in duplicate["message"]
    assert adjacent["text"] == "2: two\n3: three"


def test_source_line_tool_returns_recoverable_errors_and_does_not_cache_them(
    tmp_path,
):
    _repository, _source_file, _commit_sha, tool = _source_tool(tmp_path, ["one"])
    arguments = {
        "repository": "PX4-Autopilot",
        "path": "../outside.cpp",
        "start_line": 1,
        "end_line": 1,
    }

    first = _invoke(tool, **arguments)
    second = _invoke(tool, **arguments)

    assert "escapes" in first["error"]
    assert second == first
    assert "duplicate" not in second


def test_source_line_tool_schema_requires_explicit_bounded_read(tmp_path):
    _repository, _source_file, _commit_sha, tool = _source_tool(tmp_path, ["one"])

    assert tool.name == "read_px4_source_file"
    assert tool.strict_json_schema is True
    assert set(tool.params_json_schema["required"]) == {
        "repository",
        "path",
        "start_line",
        "end_line",
    }
    assert tool.params_json_schema["additionalProperties"] is False
