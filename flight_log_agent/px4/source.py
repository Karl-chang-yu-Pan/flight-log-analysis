from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_READ_LINE_LIMIT = 200


def search_source(source_path: Path, query: str, max_results: int = 8) -> list[dict]:
    cmd = [
        "rg",
        "-n",
        "--context",
        "3",
        query,
        str(source_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except Exception as exc:
        return [{"error": str(exc)}]

    lines = result.stdout.splitlines()[:max_results * 8]

    return [{
        "query": query,
        "matches": lines,
    }]


def read_source_file(
    source_path: Path,
    relative_path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> dict:
    if start_line < 1:
        return {"error": "start_line must be >= 1."}

    if end_line is not None and end_line < start_line:
        return {"error": "end_line must be >= start_line."}

    try:
        root = source_path.resolve()
        requested_path = Path(relative_path)
        file_path = (
            requested_path.resolve()
            if requested_path.is_absolute()
            else (root / requested_path).resolve()
        )
    except Exception as exc:
        return {"error": str(exc)}

    if not _is_relative_to(file_path, root):
        return {"error": "relative_path escapes the PX4 source path."}

    normalized_relative_path = str(file_path.relative_to(root))

    if not file_path.exists():
        return {"error": f"PX4 source file does not exist: {normalized_relative_path}"}

    if not file_path.is_file():
        return {"error": f"PX4 source path is not a file: {normalized_relative_path}"}

    requested_end_line = end_line or start_line + DEFAULT_READ_LINE_LIMIT - 1
    capped_end_line = min(requested_end_line, start_line + DEFAULT_READ_LINE_LIMIT - 1)

    try:
        file_lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return {"error": str(exc)}

    selected_lines = [
        {
            "line": line_number,
            "text": text,
        }
        for line_number, text in enumerate(file_lines, start=1)
        if start_line <= line_number <= capped_end_line
    ]

    return {
        "file": normalized_relative_path,
        "path": str(file_path),
        "start_line": start_line,
        "end_line": min(capped_end_line, len(file_lines)),
        "total_lines": len(file_lines),
        "truncated": capped_end_line < requested_end_line or capped_end_line < len(file_lines),
        "lines": selected_lines,
    }


def checkout_px4_source_revision(source_path: Path, revision: str) -> dict:
    if not source_path.exists():
        return {"error": f"PX4 source path does not exist: {source_path}"}

    is_repo = _git(source_path, ["rev-parse", "--is-inside-work-tree"])
    if is_repo.returncode != 0 or is_repo.stdout.strip() != "true":
        return {"error": f"PX4 source path is not a git repository: {source_path}"}

    status = _git(source_path, ["status", "--porcelain"])
    if status.returncode != 0:
        return {"error": status.stderr.strip() or "Failed to inspect PX4 source status."}

    if status.stdout.strip():
        return {
            "error": "PX4 source tree has local changes; refusing to checkout.",
            "status": status.stdout.splitlines(),
        }

    before_commit = _current_commit(source_path)
    branch_exists = _git(
        source_path,
        ["show-ref", "--verify", "--quiet", f"refs/heads/{revision}"],
    ).returncode == 0

    checkout_args = ["checkout", revision] if branch_exists else ["checkout", "--detach", revision]
    checkout = _git(source_path, checkout_args)
    if checkout.returncode != 0:
        return {
            "error": checkout.stderr.strip() or f"Failed to checkout {revision}.",
            "requested_revision": revision,
            "before_commit": before_commit,
        }

    return {
        "source_path": str(source_path),
        "requested_revision": revision,
        "before_commit": before_commit,
        "after_commit": _current_commit(source_path),
        "active_branch": _active_branch(source_path),
        "checked_out": True,
    }


def _git(source_path: Path, args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(source_path), *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _current_commit(source_path: Path) -> str:
    result = _git(source_path, ["rev-parse", "HEAD"])
    return result.stdout.strip()


def _active_branch(source_path: Path) -> str | None:
    result = _git(source_path, ["branch", "--show-current"])
    branch = result.stdout.strip()
    return branch or None


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False

    return True
