from __future__ import annotations

import fnmatch
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Optional, Protocol, Union, runtime_checkable

from flight_log_agent.source_path import DEFAULT_PX4_SOURCE_PATH, SOURCE_UNAVAILABLE

class SourceResolutionError(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class SourceMatch:
    file: str
    line: int
    text: str


@runtime_checkable
class SourceHandle(Protocol):
    @property
    def identity(self) -> str: ...

    def read_text(self, relative_path: str, *, errors: str = "replace") -> str: ...

    def file_exists(self, relative_path: str) -> bool: ...

    def list_files(self, prefix: str = "", *, patterns: Iterable[str] = ()) -> list[str]: ...

    def search(
        self,
        query: str,
        *,
        patterns: Iterable[str] = (),
        ignore_case: bool = False,
        fixed_strings: bool = True,
    ) -> list[SourceMatch]: ...

    def submodule(self, relative_path: str) -> "SourceHandle": ...


@dataclass(frozen=True)
class UnavailableSource:
    @property
    def identity(self) -> str:
        return "source-unavailable"

    def read_text(self, relative_path: str, *, errors: str = "replace") -> str:
        raise SourceResolutionError("repository_unavailable", "PX4 source is unavailable.")

    def file_exists(self, relative_path: str) -> bool:
        return False

    def list_files(self, prefix: str = "", *, patterns: Iterable[str] = ()) -> list[str]:
        return []

    def search(
        self,
        query: str,
        *,
        patterns: Iterable[str] = (),
        ignore_case: bool = False,
        fixed_strings: bool = True,
    ) -> list[SourceMatch]:
        return []

    def submodule(self, relative_path: str) -> "SourceHandle":
        raise SourceResolutionError("submodule_unavailable", "PX4 source is unavailable.")


UNAVAILABLE_SOURCE = UnavailableSource()


@dataclass(frozen=True)
class DirectorySource:
    """Explicit adapter for a source directory supplied by a caller."""

    root: Path
    commit_sha: Optional[str] = None

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        if not root.is_dir():
            raise SourceResolutionError("repository_unavailable", f"Source directory does not exist: {root}")
        object.__setattr__(self, "root", root)

    @property
    def identity(self) -> str:
        return str(self.root)

    def read_text(self, relative_path: str, *, errors: str = "replace") -> str:
        path = self._resolve(relative_path)
        if not path.is_file():
            raise SourceResolutionError("file_unavailable", f"Source file is unavailable: {relative_path}")
        return path.read_text(encoding="utf-8", errors=errors)

    def file_exists(self, relative_path: str) -> bool:
        try:
            return self._resolve(relative_path).is_file()
        except SourceResolutionError:
            return False

    def list_files(self, prefix: str = "", *, patterns: Iterable[str] = ()) -> list[str]:
        base = self._resolve(prefix) if prefix else self.root
        if not base.exists():
            return []
        files = [
            path.relative_to(self.root).as_posix()
            for path in base.rglob("*")
            if path.is_file()
        ]
        globs = tuple(patterns)
        if not globs:
            return files
        return [
            path
            for path in files
            if any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(PurePosixPath(path).name, pattern) for pattern in globs)
        ]

    def search(
        self,
        query: str,
        *,
        patterns: Iterable[str] = (),
        ignore_case: bool = False,
        fixed_strings: bool = True,
    ) -> list[SourceMatch]:
        flags = re.IGNORECASE if ignore_case else 0
        pattern = re.escape(query) if fixed_strings else query
        matcher = re.compile(pattern, flags)
        matches: list[SourceMatch] = []
        for relative_path in self.list_files(patterns=patterns):
            try:
                text = self.read_text(relative_path, errors="ignore")
            except SourceResolutionError:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if matcher.search(line):
                    matches.append(SourceMatch(relative_path, line_number, line))
        return matches

    def submodule(self, relative_path: str) -> "DirectorySource":
        return DirectorySource(self._resolve(relative_path))

    def _resolve(self, relative_path: str) -> Path:
        normalized = _normalize_relative_path(relative_path)
        path = (self.root / normalized).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise SourceResolutionError("file_unavailable", f"Source file path escapes repository: {relative_path}") from exc
        return path


@dataclass(frozen=True)
class SourceRepository:
    repository_path: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_path", Path(self.repository_path).expanduser().resolve())

    def resolve_snapshot(self, revision: str) -> "SourceSnapshot":
        revision = str(revision or "").strip()
        if not revision:
            raise SourceResolutionError("revision_missing", "PX4 source revision is missing.")
        if not self.repository_path.exists():
            raise SourceResolutionError(
                "repository_unavailable",
                f"PX4 source repository does not exist: {self.repository_path}",
            )
        is_repository = self._git(["rev-parse", "--is-inside-work-tree"])
        if is_repository.returncode != 0 or is_repository.stdout.strip() != "true":
            raise SourceResolutionError(
                "repository_unavailable",
                f"PX4 source path is not a Git repository: {self.repository_path}",
            )
        result = self._git(["rev-parse", "--verify", f"{revision}^{{commit}}"])
        if result.returncode != 0:
            raise SourceResolutionError(
                "revision_unavailable",
                result.stderr.strip() or f"PX4 source revision is unavailable: {revision}",
            )
        return SourceSnapshot(self.repository_path, result.stdout.strip())

    def _git(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repository_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )


@dataclass(frozen=True)
class SourceSnapshot:
    repository_path: Path
    commit_sha: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_path", Path(self.repository_path).expanduser().resolve())

    @property
    def identity(self) -> str:
        return f"{self.repository_path}@{self.commit_sha}"

    def read_text(self, relative_path: str, *, errors: str = "replace") -> str:
        path = _normalize_relative_path(relative_path)
        result = self._git_bytes(["show", f"{self.commit_sha}:{path}"])
        if result.returncode != 0:
            raise SourceResolutionError(
                "file_unavailable",
                result.stderr.decode("utf-8", errors="replace").strip()
                or f"Source file is unavailable at {self.commit_sha}: {path}",
            )
        return result.stdout.decode("utf-8", errors=errors)

    def file_exists(self, relative_path: str) -> bool:
        path = _normalize_relative_path(relative_path)
        return self._git(["cat-file", "-e", f"{self.commit_sha}:{path}"]).returncode == 0

    def list_files(
        self,
        prefix: str = "",
        *,
        patterns: Iterable[str] = (),
    ) -> list[str]:
        normalized_prefix = _normalize_relative_path(prefix) if prefix else ""
        args = ["ls-tree", "-r", "--name-only", self.commit_sha]
        if normalized_prefix:
            args.extend(["--", normalized_prefix])
        result = self._git(args)
        if result.returncode != 0:
            raise SourceResolutionError(
                "revision_unavailable",
                result.stderr.strip() or f"Failed to list source files at {self.commit_sha}.",
            )
        files = [line for line in result.stdout.splitlines() if line]
        globs = tuple(patterns)
        if not globs:
            return files
        return [
            path
            for path in files
            if any(fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(PurePosixPath(path).name, pattern) for pattern in globs)
        ]

    def search(
        self,
        query: str,
        *,
        patterns: Iterable[str] = (),
        ignore_case: bool = False,
        fixed_strings: bool = True,
    ) -> list[SourceMatch]:
        args = ["grep", "-n", "--full-name"]
        if ignore_case:
            args.append("-i")
        if fixed_strings:
            args.append("-F")
        args.extend([query, self.commit_sha])
        globs = tuple(patterns)
        if globs:
            args.append("--")
            args.extend(globs)
        result = self._git(args)
        if result.returncode not in {0, 1}:
            raise SourceResolutionError(
                "revision_unavailable",
                result.stderr.strip() or f"Failed to search source at {self.commit_sha}.",
            )
        matches: list[SourceMatch] = []
        for line in result.stdout.splitlines():
            match = re.match(r"^[^:]+:(?P<file>.*?):(?P<line>\d+):(?P<text>.*)$", line)
            if not match:
                continue
            matches.append(
                SourceMatch(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    text=match.group("text"),
                )
            )
        return matches

    def submodule(self, relative_path: str) -> "SubmoduleSnapshot":
        path = _normalize_relative_path(relative_path)
        result = self._git(["ls-tree", self.commit_sha, "--", path])
        fields = result.stdout.strip().split()
        if result.returncode != 0 or len(fields) < 4 or fields[0] != "160000" or fields[1] != "commit":
            raise SourceResolutionError(
                "submodule_unavailable",
                f"Submodule Gitlink is unavailable at {self.commit_sha}: {path}",
            )
        submodule_path = self.repository_path / path
        if not submodule_path.exists():
            raise SourceResolutionError(
                "submodule_unavailable",
                f"Submodule repository is unavailable: {submodule_path}",
            )
        commit_sha = fields[2]
        available = subprocess.run(
            ["git", "-C", str(submodule_path), "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if available.returncode != 0:
            raise SourceResolutionError(
                "submodule_unavailable",
                f"Submodule commit is unavailable: {path}@{commit_sha}",
            )
        return SubmoduleSnapshot(submodule_path, commit_sha, path)

    def _git(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repository_path), *args],
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _git_bytes(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repository_path), *args],
            capture_output=True,
            timeout=30,
        )


@dataclass(frozen=True)
class SubmoduleSnapshot(SourceSnapshot):
    parent_path: Optional[str] = None


SourceInput = Union[str, Path, SourceHandle, None]
DEFAULT_SOURCE_DIRECTORY = DEFAULT_PX4_SOURCE_PATH


def source_handle(source: SourceInput) -> Optional[SourceHandle]:
    """Apply the centralized source fallback policy and normalize the result."""
    if source is None or source == "":
        return DirectorySource(DEFAULT_SOURCE_DIRECTORY) if DEFAULT_SOURCE_DIRECTORY.is_dir() else None
    if source is SOURCE_UNAVAILABLE:
        return UNAVAILABLE_SOURCE
    if isinstance(source, SourceHandle):
        return source
    if str(source).strip():
        return DirectorySource(Path(source))
    return None


def source_from_inventory(inventory: dict) -> Optional[SourceHandle]:
    """Rebuild the exact source handle recorded in a serializable inventory."""
    repository_path = inventory.get("source_path")
    commit_sha = inventory.get("source_commit")
    if repository_path and commit_sha:
        return SourceSnapshot(Path(repository_path), str(commit_sha))
    return source_handle(repository_path)


def _normalize_relative_path(path: str) -> str:
    normalized = str(PurePosixPath(str(path).replace("\\", "/")))
    if normalized in {"", "."}:
        raise SourceResolutionError("file_unavailable", "Source file path is empty.")
    if normalized.startswith("/") or normalized == ".." or normalized.startswith("../"):
        raise SourceResolutionError("file_unavailable", f"Source file path escapes repository: {path}")
    return normalized
