import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from flight_log_agent.px4.source_snapshot import SourceRepository, SourceResolutionError
from flight_log_agent.px4.msg_schema import load_px4_msg_schema
from flight_log_agent.mission.parser import load_mavlink_command_names


def _git(path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _commit(path: Path, message: str) -> str:
    _git(path, "add", ".")
    _git(path, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", message)
    return _git(path, "rev-parse", "HEAD")


def test_snapshot_reads_lists_and_searches_exact_commit(tmp_path):
    repo = tmp_path / "PX4-Autopilot"
    repo.mkdir()
    _git(repo, "init")
    source = repo / "src" / "module.cpp"
    source.parent.mkdir()
    source.write_text("int old_value = 1;\n", encoding="utf-8")
    old_sha = _commit(repo, "old")
    source.write_text("int new_value = 2;\n", encoding="utf-8")
    new_sha = _commit(repo, "new")

    repository = SourceRepository(repo)
    old = repository.resolve_snapshot(old_sha)
    new = repository.resolve_snapshot(new_sha)

    assert old.read_text("src/module.cpp") == "int old_value = 1;\n"
    assert new.read_text("src/module.cpp") == "int new_value = 2;\n"
    assert old.list_files("src", patterns=["*.cpp"]) == ["src/module.cpp"]
    assert [(match.file, match.line, match.text) for match in old.search("old_value")] == [
        ("src/module.cpp", 1, "int old_value = 1;")
    ]


def test_snapshot_ignores_dirty_worktree(tmp_path):
    repo = tmp_path / "PX4-Autopilot"
    repo.mkdir()
    _git(repo, "init")
    source = repo / "file.txt"
    source.write_text("committed\n", encoding="utf-8")
    sha = _commit(repo, "committed")
    source.write_text("dirty\n", encoding="utf-8")

    snapshot = SourceRepository(repo).resolve_snapshot(sha)

    assert snapshot.read_text("file.txt") == "committed\n"


def test_snapshot_resolves_exact_submodule_gitlink(tmp_path):
    child = tmp_path / "mavlink"
    child.mkdir()
    _git(child, "init")
    xml = child / "message_definitions" / "v1.0" / "common.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text("<old />\n", encoding="utf-8")
    old_child_sha = _commit(child, "old child")

    parent = tmp_path / "PX4-Autopilot"
    parent.mkdir()
    _git(parent, "init")
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-C", str(parent), "submodule", "add", str(child), "src/modules/mavlink/mavlink"],
        check=True,
        capture_output=True,
        text=True,
    )
    parent_sha = _commit(parent, "parent")

    xml.write_text("<new />\n", encoding="utf-8")
    _commit(child, "new child")
    submodule_path = parent / "src/modules/mavlink/mavlink"
    _git(submodule_path, "fetch")
    _git(submodule_path, "checkout", "FETCH_HEAD")

    mavlink = SourceRepository(parent).resolve_snapshot(parent_sha).submodule("src/modules/mavlink/mavlink")

    assert mavlink.commit_sha == old_child_sha
    assert mavlink.read_text("message_definitions/v1.0/common.xml") == "<old />\n"


def test_repository_rejects_unavailable_revision(tmp_path):
    repo = tmp_path / "PX4-Autopilot"
    repo.mkdir()
    _git(repo, "init")

    with pytest.raises(SourceResolutionError) as exc_info:
        SourceRepository(repo).resolve_snapshot("missing")

    assert exc_info.value.status == "revision_unavailable"


def test_repository_rejects_non_git_directory(tmp_path):
    repo = tmp_path / "PX4-Autopilot"
    repo.mkdir()

    with pytest.raises(SourceResolutionError) as exc_info:
        SourceRepository(repo).resolve_snapshot("HEAD")

    assert exc_info.value.status == "repository_unavailable"


def test_concurrent_snapshots_keep_different_revisions_isolated(tmp_path):
    repo = tmp_path / "PX4-Autopilot"
    repo.mkdir()
    _git(repo, "init")
    message = repo / "msg" / "Status.msg"
    message.parent.mkdir()
    message.write_text("uint8 old_field\n", encoding="utf-8")
    old_sha = _commit(repo, "old")
    message.write_text("uint8 new_field\n", encoding="utf-8")
    new_sha = _commit(repo, "new")
    repository = SourceRepository(repo)
    old = repository.resolve_snapshot(old_sha)
    new = repository.resolve_snapshot(new_sha)

    def inspect(snapshot):
        return (
            load_px4_msg_schema(snapshot)["status"],
            snapshot.read_text("msg/Status.msg"),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        old_result, new_result = executor.map(inspect, [old, new])

    assert old_result == (["old_field"], "uint8 old_field\n")
    assert new_result == (["new_field"], "uint8 new_field\n")


def test_mavlink_enums_use_parent_gitlink_revision(tmp_path):
    child = tmp_path / "mavlink"
    child.mkdir()
    _git(child, "init")
    xml = child / "message_definitions" / "v1.0" / "common.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(
        '<mavlink><enums><enum name="MAV_CMD"><entry value="16" name="MAV_CMD_OLD" /></enum></enums></mavlink>',
        encoding="utf-8",
    )
    _commit(child, "old child")

    parent = tmp_path / "PX4-Autopilot"
    parent.mkdir()
    _git(parent, "init")
    subprocess.run(
        ["git", "-c", "protocol.file.allow=always", "-C", str(parent), "submodule", "add", str(child), "src/modules/mavlink/mavlink"],
        check=True,
        capture_output=True,
        text=True,
    )
    parent_sha = _commit(parent, "parent")

    xml.write_text(
        '<mavlink><enums><enum name="MAV_CMD"><entry value="16" name="MAV_CMD_NEW" /></enum></enums></mavlink>',
        encoding="utf-8",
    )
    _commit(child, "new child")
    submodule_path = parent / "src/modules/mavlink/mavlink"
    _git(submodule_path, "fetch")
    _git(submodule_path, "checkout", "FETCH_HEAD")

    snapshot = SourceRepository(parent).resolve_snapshot(parent_sha)

    assert load_mavlink_command_names(snapshot)[16] == "MAV_CMD_OLD"
