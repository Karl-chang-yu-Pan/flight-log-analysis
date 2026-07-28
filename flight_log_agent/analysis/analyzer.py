#!/usr/bin/env python3
"""
PX4 ULog analysis agent using the OpenAI Agents SDK and a local shell.

The agent receives:
  1. A PX4 ULog path
  2. A PX4-Autopilot Git repository
  3. A user question

It may run only:
  python, python3, rg, git, sed, find

PX4 source reads are pinned to the commit recorded in the ULog. The executor
maps the virtual ``PX4-Autopilot`` Git directory and ``SNAPSHOT`` revision to
the resolved repository and commit without checking out or copying the tree.

This is an internal-development example, not a hardened security sandbox.
Run it under a dedicated user or container if the input or prompt is untrusted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shlex
import shutil
import stat
import sys
import tempfile
import time
from contextlib import suppress
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import Any

from agents import (
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
)

from flight_log_agent.analysis.report_validation import (
    enforce_validation_downgrades,
    validate_report,
)
from flight_log_agent.analysis.staged_workflow import (
    reconcile_report_with_evidence_state,
    run_staged_analysis,
)
from flight_log_agent.audit import (
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
)
from flight_log_agent.models import FlightLogReport
from flight_log_agent.px4.source_snapshot import (
    SourceRepository,
    SourceResolutionError,
    SourceSnapshot,
)
from flight_log_agent.source_path import SOURCE_UNAVAILABLE, resolve_source_path
from flight_log_agent.ulog.inventory import (
    enrich_inventory_from_source,
    parse_ulog_inventory,
)


ALLOWED_PROGRAMS = {"python", "python3", "rg", "git", "sed", "find"}
FORBIDDEN_SHELL_TOKENS = {
    "|", "||", "&", "&&", ";",
    ">", ">>", "<", "<<", "<<<",
}
MAX_OUTPUT_CHARS = 600_000
DEFAULT_TIMEOUT_S = 1200
SOURCE_ALIAS = "PX4-Autopilot"
SNAPSHOT_TOKEN = "SNAPSHOT"
READ_ONLY_GIT_SUBCOMMANDS = {"grep", "show", "ls-tree"}
FORBIDDEN_GIT_OPTIONS = {
    "--ext-diff",
    "--no-index",
    "--open-files-in-pager",
    "--output",
    "--textconv",
    "-O",
}
GIT_GREP_FLAG_OPTIONS = {
    "--break",
    "--column",
    "--files-with-matches",
    "--files-without-match",
    "--fixed-strings",
    "--full-name",
    "--ignore-case",
    "--invert-match",
    "--line-number",
    "--name-only",
    "--only-matching",
    "--perl-regexp",
    "--text",
    "--word-regexp",
    "-F",
    "-I",
    "-P",
    "-i",
    "-l",
    "-n",
    "-o",
    "-v",
    "-w",
}
GIT_GREP_VALUE_OPTIONS = {
    "--after-context",
    "--before-context",
    "--context",
    "--max-count",
    "-A",
    "-B",
    "-C",
    "-m",
}
FORBIDDEN_RG_OPTIONS = {
    "--hostname-bin",
    "--pre",
}
FORBIDDEN_FIND_ACTIONS = {
    "-delete",
    "-exec",
    "-execdir",
    "-fls",
    "-fprint",
    "-fprint0",
    "-fprintf",
    "-ok",
    "-okdir",
}
ALLOWED_SED_FLAGS = {
    "-n",
    "--quiet",
    "--silent",
}
SED_PRINT_EXPRESSION = re.compile(r"(?:[0-9]+|\$)(?:,(?:[0-9]+|\$))?p")
SOURCE_BROKER_REQUEST_LIMIT = 64 * 1024
SOURCE_BROKER_MAX_REQUESTS = 128
SOURCE_BROKER_BATCH_SIZE = 16
SOURCE_ALIAS_MAX_CHARS = 4096
SOURCE_ALIAS_MAX_PARTS = 64
SOURCE_ALIAS_MAX_RESOLUTION_ATTEMPTS = 128
SOURCE_BROKER_REQUEST_NAME = re.compile(r"request-[0-9a-f]{32}\.json")
SNAPSHOT_GIT_PROXY = Path(__file__).with_name("snapshot_git_proxy.py")


class RestrictedShellExecutor:
    """Run an allowlist of commands and pin PX4 Git reads to one snapshot."""

    def __init__(
        self,
        cwd: Path,
        source_snapshot: SourceSnapshot | None = None,
        input_paths: dict[str, Path] | None = None,
    ):
        self.cwd = cwd.resolve()
        self.source_snapshot = source_snapshot
        self.input_paths = {
            name: path.expanduser().resolve()
            for name, path in (input_paths or {}).items()
        }
        unknown_inputs = set(self.input_paths) - {
            "flight.ulg",
            "mission.plan",
        }
        if unknown_inputs:
            raise ValueError(
                f"Unsupported analysis input names: {sorted(unknown_inputs)}"
            )

    def _parse_command(
        self,
        command: str,
        *,
        bind_snapshot_git: bool = True,
    ) -> list[str]:
        try:
            argv = shlex.split(command, posix=True)
        except ValueError as exc:
            raise ValueError(f"Invalid command syntax: {exc}") from exc

        if not argv:
            raise ValueError("Empty command.")

        program = Path(argv[0]).name
        if program not in ALLOWED_PROGRAMS:
            raise PermissionError(
                f"Command '{program}' is not allowed. "
                f"Allowed: {sorted(ALLOWED_PROGRAMS)}"
            )
        if argv[0] != program:
            raise PermissionError(
                "Executables must be invoked by their allowlisted name, not "
                "through an absolute or relative executable path."
            )

        if any(token in FORBIDDEN_SHELL_TOKENS for token in argv):
            raise PermissionError(
                "Shell operators, redirection, pipes, and command chaining "
                "are disabled. Run one command at a time."
            )

        if program == "git":
            return (
                self._snapshot_git_command(argv)
                if bind_snapshot_git
                else argv
            )
        if program in {"rg", "sed", "find"}:
            self._validate_workspace_read_command(program, argv[1:])
        return argv

    @classmethod
    def _validate_workspace_read_command(
        cls,
        program: str,
        args: list[str],
    ) -> None:
        if program == "rg":
            for token in args:
                option = token.split("=", 1)[0]
                if option in FORBIDDEN_RG_OPTIONS:
                    raise PermissionError(
                        f"rg option '{option}' can execute another program."
                    )
                cls._reject_external_path_token(token)
            return

        if program == "find":
            for token in args:
                if token in FORBIDDEN_FIND_ACTIONS:
                    raise PermissionError(
                        f"find action '{token}' is not a read-only listing action."
                    )
                cls._reject_external_path_token(token)
            return

        cls._validate_sed_read(args)

    @classmethod
    def _validate_sed_read(cls, args: list[str]) -> None:
        expressions: list[str] = []
        files: list[str] = []
        index = 0
        while index < len(args):
            token = args[index]
            if token in ALLOWED_SED_FLAGS:
                index += 1
                continue
            if token in {"-e", "--expression"}:
                if index + 1 >= len(args):
                    raise PermissionError(
                        f"sed option '{token}' requires a print expression."
                    )
                expressions.append(args[index + 1])
                index += 2
                continue
            if token.startswith("-"):
                raise PermissionError(
                    f"sed option '{token}' is not allowed for read-only excerpts."
                )
            if not expressions:
                expressions.append(token)
            else:
                files.append(token)
            index += 1

        if not expressions or not files:
            raise PermissionError(
                "sed must use a numeric print expression and a workspace file."
            )
        for expression in expressions:
            if SED_PRINT_EXPRESSION.fullmatch(expression.strip()) is None:
                raise PermissionError(
                    "sed is limited to numeric print expressions such as "
                    "'1,120p'."
                )
        for file_name in files:
            cls._reject_external_path_token(file_name)

    @staticmethod
    def _reject_external_path_token(token: str) -> None:
        normalized = token.replace("\\", "/")
        if (
            normalized.startswith("/")
            or normalized == ".."
            or normalized.startswith("../")
            or "/../" in normalized
            or normalized.endswith("/..")
        ):
            raise PermissionError(
                "rg, sed, and find may access only paths inside the analysis "
                "workspace."
            )

    def _snapshot_git_command(self, argv: list[str]) -> list[str]:
        if self.source_snapshot is None:
            raise PermissionError(
                "PX4 source is unavailable because the logged revision was not resolved."
            )
        if len(argv) < 4 or argv[1] != "-C":
            raise PermissionError(
                f"Git source reads must use 'git -C {SOURCE_ALIAS} <command> ...'."
            )

        source = self._source_for_alias(argv[2])
        return self._snapshot_git_command_for_source(argv, source)

    def _snapshot_git_command_for_source(
        self,
        argv: list[str],
        source: SourceSnapshot,
    ) -> list[str]:
        subcommand = argv[3]
        if subcommand not in READ_ONLY_GIT_SUBCOMMANDS:
            raise PermissionError(
                f"Git command '{subcommand}' is not allowed. "
                f"Read-only commands: {sorted(READ_ONLY_GIT_SUBCOMMANDS)}"
            )

        args = argv[4:]
        self._validate_git_options(args)
        rewritten_args = self._bind_snapshot_revision(
            subcommand,
            args,
            source.commit_sha,
        )
        return [
            "git",
            "-C",
            str(source.repository_path),
            subcommand,
            *rewritten_args,
        ]

    async def _snapshot_git_command_async(
        self,
        argv: list[str],
        *,
        timeout_s: float,
    ) -> list[str]:
        if self.source_snapshot is None:
            raise PermissionError(
                "PX4 source is unavailable because the logged revision was "
                "not resolved."
            )
        if len(argv) < 4 or argv[1] != "-C":
            raise PermissionError(
                f"Git source reads must use 'git -C {SOURCE_ALIAS} "
                "<command> ...'."
            )
        source = await self._source_for_alias_async(
            argv[2],
            timeout_s=timeout_s,
        )
        return self._snapshot_git_command_for_source(argv, source)

    def _source_for_alias(self, alias: str) -> SourceSnapshot:
        parts = self._validate_source_alias(alias)
        source = self.source_snapshot
        remaining = list(parts[1:])
        resolution_attempts = 0
        while remaining:
            resolved = None
            for end in range(len(remaining), 0, -1):
                resolution_attempts += 1
                if (
                    resolution_attempts
                    > SOURCE_ALIAS_MAX_RESOLUTION_ATTEMPTS
                ):
                    raise PermissionError(
                        "PX4 submodule alias resolution limit exceeded."
                    )
                candidate = "/".join(remaining[:end])
                try:
                    resolved = source.submodule(candidate)
                except SourceResolutionError:
                    continue
                remaining = remaining[end:]
                break
            if resolved is None:
                raise PermissionError(
                    f"Source alias is not a resolved PX4 submodule: {alias}"
                )
            source = resolved
        return source

    @staticmethod
    def _validate_source_alias(alias: str) -> tuple[str, ...]:
        if len(alias) > SOURCE_ALIAS_MAX_CHARS:
            raise PermissionError("PX4 source alias is too long.")
        normalized = PurePosixPath(str(alias).replace("\\", "/"))
        parts = normalized.parts
        if (
            normalized.is_absolute()
            or not parts
            or len(parts) > SOURCE_ALIAS_MAX_PARTS
            or parts[0] != SOURCE_ALIAS
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise PermissionError(
                f"Git -C must name {SOURCE_ALIAS} or a resolved submodule beneath it."
            )
        return parts

    async def _source_for_alias_async(
        self,
        alias: str,
        *,
        timeout_s: float,
    ) -> SourceSnapshot:
        parts = self._validate_source_alias(alias)
        source = self.source_snapshot
        remaining = list(parts[1:])
        resolution_attempts = 0
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while remaining:
            resolved = None
            for end in range(len(remaining), 0, -1):
                resolution_attempts += 1
                if (
                    resolution_attempts
                    > SOURCE_ALIAS_MAX_RESOLUTION_ATTEMPTS
                ):
                    raise PermissionError(
                        "PX4 submodule alias resolution limit exceeded."
                    )
                candidate = "/".join(remaining[:end])
                remaining_time = deadline - loop.time()
                if remaining_time <= 0:
                    raise TimeoutError(
                        "PX4 submodule alias resolution timed out."
                    )
                stdout, _stderr, returncode, timed_out = (
                    await self._run_process(
                        [
                            "git",
                            "-C",
                            str(source.repository_path),
                            "ls-tree",
                            source.commit_sha,
                            "--",
                            candidate,
                        ],
                        timeout_s=remaining_time,
                        cwd=self.cwd,
                        env=self._restricted_env(),
                    )
                )
                fields = stdout.decode(
                    "utf-8",
                    errors="replace",
                ).strip().split()
                if timed_out:
                    raise TimeoutError(
                        "PX4 submodule alias resolution timed out."
                    )
                if (
                    returncode != 0
                    or len(fields) < 4
                    or fields[0] != "160000"
                    or fields[1] != "commit"
                ):
                    continue
                submodule_path = source.repository_path / candidate
                if not submodule_path.exists():
                    continue
                commit_sha = fields[2]
                remaining_time = deadline - loop.time()
                if remaining_time <= 0:
                    raise TimeoutError(
                        "PX4 submodule alias resolution timed out."
                    )
                _stdout, _stderr, returncode, timed_out = (
                    await self._run_process(
                        [
                            "git",
                            "-C",
                            str(submodule_path),
                            "cat-file",
                            "-e",
                            f"{commit_sha}^{{commit}}",
                        ],
                        timeout_s=remaining_time,
                        cwd=self.cwd,
                        env=self._restricted_env(),
                    )
                )
                if timed_out:
                    raise TimeoutError(
                        "PX4 submodule alias resolution timed out."
                    )
                if returncode != 0:
                    continue
                resolved = SourceSnapshot(submodule_path, commit_sha)
                remaining = remaining[end:]
                break
            if resolved is None:
                raise PermissionError(
                    f"Source alias is not a resolved PX4 submodule: {alias}"
                )
            source = resolved
        return source

    @staticmethod
    def _validate_git_options(args: list[str]) -> None:
        for token in args:
            option = token.split("=", 1)[0]
            if option in FORBIDDEN_GIT_OPTIONS:
                raise PermissionError(
                    f"Git option '{option}' is not allowed for snapshot reads."
                )

    @staticmethod
    def _bind_snapshot_revision(
        subcommand: str,
        args: list[str],
        commit_sha: str,
    ) -> list[str]:
        if subcommand == "show":
            if len(args) != 1:
                raise PermissionError(
                    "Git show accepts exactly one SNAPSHOT:relative/path object."
                )
            index = 0
            token = args[index]
            prefix = f"{SNAPSHOT_TOKEN}:"
            if not token.startswith(prefix) or token == prefix:
                raise PermissionError(
                    f"Git show must read a file as {SNAPSHOT_TOKEN}:relative/path."
                )
            relative_path = token[len(prefix):]
            normalized_path = PurePosixPath(relative_path.replace("\\", "/"))
            if (
                normalized_path.is_absolute()
                or str(normalized_path) in {"", "."}
                or any(part in {"", ".", ".."} for part in normalized_path.parts)
            ):
                raise PermissionError(
                    f"Git show path escapes the snapshot: {relative_path}"
                )
            replacement = f"{commit_sha}:{normalized_path.as_posix()}"
        else:
            matches: list[int] = []
            for candidate_index, token in enumerate(args):
                if token != SNAPSHOT_TOKEN:
                    continue
                try:
                    if subcommand == "grep":
                        RestrictedShellExecutor._validate_git_grep_args(
                            args,
                            candidate_index,
                        )
                    else:
                        RestrictedShellExecutor._validate_git_ls_tree_args(
                            args,
                            candidate_index,
                        )
                except PermissionError:
                    continue
                matches.append(candidate_index)

            if len(matches) != 1:
                raise PermissionError(
                    f"Git {subcommand} must contain exactly one unambiguous "
                    f"{SNAPSHOT_TOKEN} revision."
                )
            index = matches[0]
            replacement = commit_sha

        rewritten = list(args)
        rewritten[index] = replacement
        return rewritten

    @staticmethod
    def _validate_git_grep_args(args: list[str], revision_index: int) -> None:
        suffix = args[revision_index + 1:]
        if suffix and suffix[0] != "--":
            raise PermissionError(
                "Git grep paths must follow '--' after the SNAPSHOT revision."
            )

        prefix = args[:revision_index]
        pattern_count = 0
        index = 0
        while index < len(prefix):
            token = prefix[index]
            if token in GIT_GREP_FLAG_OPTIONS:
                index += 1
                continue
            if token in GIT_GREP_VALUE_OPTIONS:
                if index + 1 >= len(prefix):
                    raise PermissionError(
                        f"Git grep option '{token}' requires a value."
                    )
                index += 2
                continue
            if token in {"-e", "--regexp"}:
                if index + 1 >= len(prefix):
                    raise PermissionError(
                        f"Git grep option '{token}' requires a pattern."
                    )
                pattern_count += 1
                index += 2
                continue
            if token.startswith("-"):
                raise PermissionError(
                    f"Git grep option '{token}' is not allowed."
                )
            pattern_count += 1
            index += 1

        if pattern_count != 1:
            raise PermissionError(
                "Git grep must specify exactly one search pattern before SNAPSHOT."
            )

    @staticmethod
    def _validate_git_ls_tree_args(args: list[str], revision_index: int) -> None:
        if any(not token.startswith("-") for token in args[:revision_index]):
            raise PermissionError(
                "Git ls-tree accepts only options before SNAPSHOT."
            )
        suffix = args[revision_index + 1:]
        if suffix and suffix[0] != "--":
            raise PermissionError(
                "Git ls-tree paths must follow '--' after SNAPSHOT."
            )

    async def __call__(self, request: ShellCommandRequest) -> ShellResult:
        action = request.data.action
        outputs: list[ShellCommandOutput] = []

        requested_timeout_s = (
            (action.timeout_ms or 0) / 1000 if action.timeout_ms else DEFAULT_TIMEOUT_S
        )
        timeout_s = min(max(requested_timeout_s, 1), DEFAULT_TIMEOUT_S)

        for command in action.commands:
            timed_out = False

            try:
                argv = self._parse_command(
                    command,
                    bind_snapshot_git=False,
                )
                if argv[0] == "git":
                    loop = asyncio.get_running_loop()
                    started_at = loop.time()
                    argv = await self._snapshot_git_command_async(
                        argv,
                        timeout_s=timeout_s,
                    )
                    remaining = timeout_s - (loop.time() - started_at)
                    if remaining <= 0:
                        raise TimeoutError(
                            "Snapshot Git request expired during validation."
                        )
                    (
                        stdout_bytes,
                        stderr_bytes,
                        exit_code,
                        timed_out,
                    ) = await self._run_process(
                        argv,
                        timeout_s=remaining,
                        cwd=self.cwd,
                        env=self._restricted_env(),
                    )
                else:
                    (
                        stdout_bytes,
                        stderr_bytes,
                        exit_code,
                        timed_out,
                    ) = await self._run_sandboxed_process(
                        argv,
                        timeout_s=timeout_s,
                    )

                stdout = stdout_bytes.decode("utf-8", errors="replace")
                stderr = stderr_bytes.decode("utf-8", errors="replace")

                stdout = self._truncate(stdout)
                stderr = self._truncate(stderr)

                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout=stdout,
                        stderr=stderr,
                        outcome=ShellCallOutcome(
                            type="timeout" if timed_out else "exit",
                            exit_code=exit_code,
                        ),
                    )
                )

            except TimeoutError as exc:
                timed_out = True
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout="",
                        stderr=f"TimeoutError: {exc}",
                        outcome=ShellCallOutcome(
                            type="timeout",
                            exit_code=124,
                        ),
                    )
                )
            except Exception as exc:
                outputs.append(
                    ShellCommandOutput(
                        command=command,
                        stdout="",
                        stderr=f"{type(exc).__name__}: {exc}",
                        outcome=ShellCallOutcome(
                            type="exit",
                            exit_code=126,
                        ),
                    )
                )

            if timed_out:
                break

        return ShellResult(
            output=outputs,
            provider_data={
                "working_directory": str(self.cwd),
                "source_snapshot": (
                    self.source_snapshot.identity if self.source_snapshot else None
                ),
            },
        )

    async def _run_sandboxed_process(
        self,
        argv: list[str],
        *,
        timeout_s: float,
    ) -> tuple[bytes, bytes, int, bool]:
        if argv[0] not in {"python", "python3"}:
            sandbox_argv = self._build_sandbox_command(argv)
            return await self._run_process(
                sandbox_argv,
                timeout_s=timeout_s,
                cwd=None,
                env=self._restricted_env(),
            )

        with tempfile.TemporaryDirectory(
            prefix="flight-source-broker-",
            dir="/tmp",
        ) as broker_root:
            broker_path = Path(broker_root)
            (broker_path / "requests").mkdir()
            (broker_path / "responses").mkdir()
            stop_broker = asyncio.Event()
            broker_task = asyncio.create_task(
                self._serve_snapshot_git_requests(
                    broker_path,
                    stop_broker,
                    deadline=asyncio.get_running_loop().time() + timeout_s,
                )
            )
            try:
                sandbox_argv = self._build_sandbox_command(
                    argv,
                    broker_root=broker_path,
                )
                return await self._run_process(
                    sandbox_argv,
                    timeout_s=timeout_s,
                    cwd=None,
                    env=self._restricted_env(),
                )
            finally:
                stop_broker.set()
                broker_task.cancel()
                with suppress(asyncio.CancelledError):
                    await broker_task

    async def _serve_snapshot_git_requests(
        self,
        broker_root: Path,
        stop: asyncio.Event,
        *,
        deadline: float | None = None,
    ) -> None:
        request_dir = broker_root / "requests"
        response_dir = broker_root / "responses"
        request_count = 0
        while True:
            if stop.is_set():
                for request_path in request_dir.glob("request-*.json"):
                    request_path.unlink(missing_ok=True)
                return

            request_paths = list(
                islice(
                    request_dir.glob("request-*.json"),
                    SOURCE_BROKER_BATCH_SIZE,
                )
            )
            for request_path in request_paths:
                if stop.is_set():
                    request_path.unlink(missing_ok=True)
                    continue
                if SOURCE_BROKER_REQUEST_NAME.fullmatch(request_path.name) is None:
                    request_path.unlink(missing_ok=True)
                    continue
                response_path = response_dir / request_path.name.replace(
                    "request-",
                    "response-",
                    1,
                )
                request_count += 1
                remaining = (
                    deadline - asyncio.get_running_loop().time()
                    if deadline is not None
                    else DEFAULT_TIMEOUT_S
                )
                if request_count > SOURCE_BROKER_MAX_REQUESTS:
                    response = {
                        "stdout": "",
                        "stderr": (
                            "PermissionError: source-broker request limit "
                            "exceeded\n"
                        ),
                        "returncode": 126,
                    }
                elif remaining <= 0:
                    response = {
                        "stdout": "",
                        "stderr": "TimeoutError: source-broker deadline expired\n",
                        "returncode": 124,
                    }
                else:
                    response = await self._snapshot_git_response(
                        request_path,
                        timeout_s=min(remaining, DEFAULT_TIMEOUT_S),
                    )
                temporary_response = response_path.with_suffix(".tmp")
                temporary_response.write_text(
                    json.dumps(response),
                    encoding="utf-8",
                )
                temporary_response.replace(response_path)
                request_path.unlink(missing_ok=True)

            await asyncio.sleep(0.01)

    async def _snapshot_git_response(
        self,
        request_path: Path,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> dict[str, Any]:
        try:
            request = json.loads(
                self._read_source_broker_request(request_path)
            )
            requested_args = request.get("argv")
            if (
                not isinstance(requested_args, list)
                or not all(isinstance(arg, str) for arg in requested_args)
            ):
                raise ValueError("Source-broker argv must be a list of strings.")

            loop = asyncio.get_running_loop()
            started_at = loop.time()
            git_argv = await self._snapshot_git_command_async(
                ["git", *requested_args],
                timeout_s=timeout_s,
            )
            remaining = timeout_s - (loop.time() - started_at)
            if remaining <= 0:
                raise TimeoutError(
                    "Snapshot Git request expired during validation."
                )
            stdout, stderr, returncode, timed_out = await self._run_process(
                git_argv,
                timeout_s=remaining,
                cwd=self.cwd,
                env=self._restricted_env(),
            )
            return {
                "stdout": self._truncate(
                    stdout.decode("utf-8", errors="replace")
                ),
                "stderr": self._truncate(
                    stderr.decode("utf-8", errors="replace")
                ),
                "returncode": 124 if timed_out else returncode,
            }
        except TimeoutError as exc:
            return {
                "stdout": "",
                "stderr": f"TimeoutError: {exc}\n",
                "returncode": 124,
            }
        except Exception as exc:
            return {
                "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}\n",
                "returncode": 126,
            }

    async def preflight(self) -> None:
        checks = ["import os", "import subprocess", "from pathlib import Path"]
        input_environments = {
            "flight.ulg": "FLIGHT_LOG_ULOG",
            "mission.plan": "FLIGHT_LOG_MISSION",
        }
        for input_name in self.input_paths:
            environment_name = input_environments[input_name]
            checks.append(
                f"""
input_path = Path(os.environ["{environment_name}"])
input_path.open("rb").read(1)
try:
    input_descriptor = os.open(input_path, os.O_WRONLY | os.O_APPEND)
except OSError:
    pass
else:
    os.close(input_descriptor)
    raise RuntimeError(
        "The analysis input {input_name} is writable inside the sandbox."
    )
""".strip()
            )
        if self.source_snapshot is not None:
            checks.append(
                "subprocess.run("
                "['git', '-C', os.environ['FLIGHT_LOG_PX4_REPOSITORY'], "
                "'ls-tree', '--name-only', "
                "os.environ['FLIGHT_LOG_PX4_COMMIT'], '--'], "
                "check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)"
            )
        stdout, stderr, returncode, timed_out = (
            await self._run_sandboxed_process(
                ["python", "-c", "\n".join(checks)],
                timeout_s=10,
            )
        )
        if timed_out or returncode != 0:
            detail = (stderr or stdout).decode(
                "utf-8",
                errors="replace",
            ).strip()
            raise RuntimeError(
                "The analysis shell sandbox or snapshot source bridge is "
                "unavailable"
                + (f": {detail}" if detail else ".")
            )

    @staticmethod
    def _read_source_broker_request(request_path: Path) -> str:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(request_path, flags)
        try:
            file_status = os.fstat(descriptor)
            if not stat.S_ISREG(file_status.st_mode):
                raise ValueError(
                    "Source-broker request must be a regular file."
                )
            if file_status.st_size >= SOURCE_BROKER_REQUEST_LIMIT:
                raise ValueError("Oversized source-broker request.")

            chunks: list[bytes] = []
            bytes_read = 0
            while True:
                remaining = SOURCE_BROKER_REQUEST_LIMIT - bytes_read
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                bytes_read += len(chunk)
                if bytes_read >= SOURCE_BROKER_REQUEST_LIMIT:
                    raise ValueError("Oversized source-broker request.")
            return b"".join(chunks).decode("utf-8")
        finally:
            os.close(descriptor)

    def _build_sandbox_command(
        self,
        argv: list[str],
        *,
        broker_root: Path | None = None,
    ) -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError(
                "bubblewrap is required for workspace-scoped shell execution."
            )
        if not SNAPSHOT_GIT_PROXY.is_file():
            raise RuntimeError(
                f"Snapshot Git proxy is unavailable: {SNAPSHOT_GIT_PROXY}"
            )

        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--unshare-net",
            "--clearenv",
        ]
        for system_path in (Path("/usr"), Path("/lib"), Path("/lib64")):
            if system_path.exists():
                command.extend(
                    ["--ro-bind", str(system_path), str(system_path)]
                )
        if Path("/bin").is_symlink():
            command.extend(["--symlink", "usr/bin", "/bin"])
        elif Path("/bin").exists():
            command.extend(["--ro-bind", "/bin", "/bin"])

        command.extend(
            [
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--bind",
                str(self.cwd),
                "/work",
                "--bind",
                str(self.cwd.parent / "plots"),
                "/plots",
                "--ro-bind",
                str(Path(sys.prefix).resolve()),
                "/venv",
                "--dir",
                "/sandbox-bin",
                "--ro-bind",
                str(SNAPSHOT_GIT_PROXY),
                "/sandbox-bin/git",
            ]
        )

        if self.input_paths:
            command.extend(["--dir", "/inputs"])
        for input_name, input_path in self.input_paths.items():
            target = input_path.resolve(strict=True)
            command.extend(
                ["--ro-bind", str(target), f"/inputs/{input_name}"]
            )

        if broker_root is not None:
            request_dir = broker_root / "requests"
            response_dir = broker_root / "responses"
            if not request_dir.is_dir() or not response_dir.is_dir():
                raise RuntimeError(
                    "Snapshot Git broker request/response directories are "
                    "not initialized."
                )
            command.extend(
                [
                    "--dir",
                    "/broker",
                    "--dir",
                    "/broker/requests",
                    "--dir",
                    "/broker/responses",
                    "--bind",
                    str(request_dir),
                    "/broker/requests",
                    "--ro-bind",
                    str(response_dir),
                    "/broker/responses",
                    "--setenv",
                    "FLIGHT_LOG_SOURCE_BROKER",
                    "/broker",
                ]
            )

        command.extend(
            [
                "--setenv",
                "PATH",
                "/sandbox-bin:/venv/bin:/usr/bin:/bin",
                "--setenv",
                "HOME",
                "/work",
                "--setenv",
                "TMPDIR",
                "/tmp",
                "--setenv",
                "VIRTUAL_ENV",
                "/venv",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                "--setenv",
                "MPLBACKEND",
                "Agg",
                "--setenv",
                "MPLCONFIGDIR",
                "/tmp/matplotlib",
                "--setenv",
                "LANG",
                os.environ.get("LANG", "C.UTF-8"),
            ]
        )
        if self.source_snapshot is not None:
            command.extend(
                [
                    "--setenv",
                    "FLIGHT_LOG_PX4_REPOSITORY",
                    SOURCE_ALIAS,
                    "--setenv",
                    "FLIGHT_LOG_PX4_COMMIT",
                    SNAPSHOT_TOKEN,
                ]
            )
        if "flight.ulg" in self.input_paths:
            command.extend(
                [
                    "--setenv",
                    "FLIGHT_LOG_ULOG",
                    "/inputs/flight.ulg",
                ]
            )
        if "mission.plan" in self.input_paths:
            command.extend(
                [
                    "--setenv",
                    "FLIGHT_LOG_MISSION",
                    "/inputs/mission.plan",
                ]
            )

        command.extend(["--chdir", "/work"])
        if argv[0] in {"python", "python3"}:
            command.extend(["/venv/bin/python", *argv[1:]])
        else:
            executable = shutil.which(argv[0])
            if executable is None:
                raise RuntimeError(
                    f"Allowlisted executable is unavailable: {argv[0]}"
                )
            command.extend([executable, *argv[1:]])
        return command

    @staticmethod
    async def _run_process(
        argv: list[str],
        *,
        timeout_s: float,
        cwd: Path | None,
        env: dict[str, str],
    ) -> tuple[bytes, bytes, int, bool]:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communication = asyncio.create_task(process.communicate())
        try:
            stdout, stderr = await asyncio.wait_for(
                asyncio.shield(communication),
                timeout=timeout_s,
            )
            return stdout, stderr, process.returncode, False
        except TimeoutError:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await communication
            return stdout, stderr, 124, True
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await asyncio.shield(communication)
            raise

    @staticmethod
    def _truncate(text: str) -> str:
        if len(text) <= MAX_OUTPUT_CHARS:
            return text
        omitted = len(text) - MAX_OUTPUT_CHARS
        return (
            text[:MAX_OUTPUT_CHARS]
            + f"\n\n[output truncated; {omitted} characters omitted]"
        )

    def _restricted_env(self) -> dict[str, str]:
        # Keep only the variables normally needed to execute Python and tools.
        keep = {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "PYTHONPATH",
            "VIRTUAL_ENV",
            "SSL_CERT_FILE",
            "SSL_CERT_DIR",
        }
        env = {key: value for key, value in os.environ.items() if key in keep}
        env["GIT_NO_REPLACE_OBJECTS"] = "1"
        if self.source_snapshot is not None:
            env.update(
                {
                    "FLIGHT_LOG_PX4_REPOSITORY": SOURCE_ALIAS,
                    "FLIGHT_LOG_PX4_COMMIT": SNAPSHOT_TOKEN,
                }
            )
        return env


BASE_INSTRUCTIONS = """
You are analyzing a PX4 ULog and, when available, its exact resolved
PX4-Autopilot source snapshot.

Available workspace entries:
- /inputs/flight.ulg: the read-only input log (also FLIGHT_LOG_ULOG)
- /inputs/mission.plan: the optional read-only mission input, when supplied
- the current directory: writable per-run scripts and intermediate data
- /plots/: persistent plot output

Use Python with pyulog to inspect the actual ULog. Decide which commands and
analyses are needed from the user's question.

PX4 source is not exposed as a working-tree directory. Read it through the
executor's commit-pinned Git interface:
- Search: git -C PX4-Autopilot grep -n -F "term" SNAPSHOT -- src/
- Read: git -C PX4-Autopilot show SNAPSHOT:src/module/file.cpp
- List: git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/
- Schema read: git -C PX4-Autopilot show SNAPSHOT:msg/VehicleStatus.msg

Use other snapshot-relative path scopes, including msg/, when the evidence
requires them; the src/ forms above are examples, not a source-path limit.

SNAPSHOT is a virtual revision token. The executor replaces it with the
resolved full commit SHA. Never use HEAD, a branch, a tag, or a literal hash.
For an initialized submodule, use its path as the virtual Git directory, for
example git -C PX4-Autopilot/src/modules/mavlink/mavlink show SNAPSHOT:path.
The executor resolves SNAPSHOT to the gitlink commit recorded by the parent.

Python, rg, sed, and find run in a filesystem sandbox containing only this
analysis workspace, its read-only input files, the plot directory, and runtime
libraries. They cannot see the PX4 working tree. For more complex in-memory
source processing, Python may invoke the same three Git commands using
FLIGHT_LOG_PX4_REPOSITORY and FLIGHT_LOG_PX4_COMMIT. Those variables contain
the virtual PX4-Autopilot and SNAPSHOT tokens; a broker binds every request to
the resolved commit and permits no checkout or write operation.

Required analysis behavior:
- Treat the deterministic prepass and resolved-source status supplied with
  this run as authoritative. Do not repeat source resolution or select a
  different source revision.
- Ground flight-specific claims in values read from /inputs/flight.ulg. Source code
  describes possible firmware behavior; it does not prove that behavior was
  active in this flight.
- Choose the investigation appropriate to the user's question. When making a
  causal claim, do not stop at the first plausible source match. Start from
  the logged value or event being explained and trace upstream through the
  assignments, constraints, branches, and helpers that can change it. Confirm
  that the relevant runtime conditions were active.
- Test causal explanations with evidence aligned to the same time interval
  and expressed in compatible units or representations. An enabled parameter,
  a matching downstream value, correlation, or independently computed extrema
  do not by themselves establish causality.
- Consider competing explanations discovered along the relevant source path.
  If the available log cannot distinguish them, report the result as
  unresolved or lower confidence.
- Only present numeric or exclusion checks as performed evidence when they
  were actually evaluated during this run. A check that merely restates the
  observed symptom is not causal verification.
- Clearly distinguish ULog observations, source-confirmed behavior, inference,
  contradicting evidence, and unresolved evidence. Calibrate confidence to
  the available evidence.
- Complete only the assigned evidence stage. The final report stage answers
  the user's question using the required FlightLogReport schema.

Tool use:
- Choose commands, statistics, scripts, and plots based on the question.
- Analyze complete data or source when needed, but keep returned excerpts and
  printed results relevant and bounded. Do not omit evidence merely for
  compactness.
- Reuse successful results instead of repeating identical commands.
- If a plot is useful, save it under /plots/.
- The executor rejects pipes, shell redirection, and command chaining.
- To create a temporary script, use Python, for example:
  python -c "from pathlib import Path; Path('analysis.py').write_text('...')"
"""


def prepare_workspace(
    ulog_path: Path,
    output_dir: Path,
    mission_path: Path | None = None,
) -> Path:
    output_dir = output_dir.expanduser().resolve()
    work_dir = output_dir / "work"
    plots_dir = output_dir / "plots"
    work_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    _remove_legacy_input_link(work_dir / "flight.ulg")
    _remove_legacy_input_link(work_dir / "mission.plan")
    return work_dir


def _remove_legacy_input_link(link: Path) -> None:
    if link.is_symlink():
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"Analysis workspace contains an unmanaged input path: {link}"
        )


async def analyze_flight_log(
    log_path: str,
    user_question: str,
    mission_path: str | None = None,
    source_path: str | None = None,
    output_dir: str = "outputs/run_001",
    dev_log_root: str = str(DEFAULT_DEV_LOG_ROOT),
    dev_run_id: str | None = None,
    max_candidates: int = 5,
    mechanism_cache_dir: str = ".flightlog_cache/mechanisms",
    force_mechanism_refresh: bool = False,
    dag_discovery: bool | None = None,
    dag_cache_dir: str = ".flightlog_cache",
    *,
    model: str | None = None,
    max_turns: int = 20,
    max_total_requests: int | None = None,
    project_instructions: str = "",
    enable_web_fallback: bool = True,
    web_search_context: str = "medium",
) -> FlightLogReport:
    """Run the shell-driven analyzer behind the stable web analysis contract."""

    log_path_obj = Path(log_path).expanduser().resolve()
    mission_path_obj = (
        Path(mission_path).expanduser().resolve() if mission_path else None
    )
    output_dir_obj = Path(output_dir).expanduser().resolve()
    source_repository_path = resolve_source_path(source_path)
    source_snapshot: SourceSnapshot | None = None

    if not log_path_obj.is_file():
        raise FileNotFoundError(f"ULog not found: {log_path_obj}")
    if mission_path_obj is not None and not mission_path_obj.is_file():
        raise FileNotFoundError(f"Mission file not found: {mission_path_obj}")
    if web_search_context not in {"low", "medium", "high"}:
        raise ValueError(
            "web_search_context must be one of: low, medium, high"
        )
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if max_total_requests is not None and max_total_requests < 1:
        raise ValueError("max_total_requests must be at least 1")

    work_dir = prepare_workspace(
        log_path_obj,
        output_dir_obj,
        mission_path_obj,
    )
    analysis_inputs = {"flight.ulg": log_path_obj}
    if mission_path_obj is not None:
        analysis_inputs["mission.plan"] = mission_path_obj
    report_path = output_dir_obj / "report.json"
    audit_logger = DeveloperAuditLogger(Path(dev_log_root), run_id=dev_run_id)
    audit_logger.save_metadata(
        {
            "runner_version": "shell_snapshot_v1",
            "log_path": str(log_path_obj),
            "mission_path": str(mission_path_obj) if mission_path_obj else None,
            "source_path": (
                str(source_repository_path) if source_repository_path else None
            ),
            "output_dir": str(output_dir_obj),
            "work_dir": str(work_dir),
            "report_path": str(report_path),
            "max_candidates": max_candidates,
            "mechanism_cache_dir": mechanism_cache_dir,
            "force_mechanism_refresh": force_mechanism_refresh,
            "dag_discovery": dag_discovery,
            "dag_cache_dir": dag_cache_dir,
            "analysis_architecture": "staged_evidence_v1",
            "enable_web_fallback": enable_web_fallback,
            "web_search_context": web_search_context,
            "max_turns_per_stage": max_turns,
            "max_total_requests": (
                max_total_requests
                if max_total_requests is not None
                else max_turns
            ),
        }
    )
    audit_logger.log_event("run.started")

    try:
        inventory = _run_audited_stage(
            audit_logger,
            "prepass",
            "parse_ulog_inventory",
            parse_ulog_inventory,
            {"log_path": str(log_path_obj), "source_path": None},
            log_path_obj,
            SOURCE_UNAVAILABLE,
        )
        logged_px4_git_hash = (
            inventory.get("git_hash")
            or inventory.get("px4_git_hash")
            or inventory.get("firmware_git_hash")
        )

        if source_repository_path is not None and logged_px4_git_hash:
            try:
                source_snapshot = _run_audited_stage(
                    audit_logger,
                    "source",
                    "resolve_px4_source_snapshot",
                    SourceRepository(source_repository_path).resolve_snapshot,
                    {"revision": logged_px4_git_hash},
                    logged_px4_git_hash,
                )
                enrich_inventory_from_source(inventory, source_snapshot)
            except Exception as exc:
                status = getattr(exc, "status", "repository_unavailable")
                inventory.setdefault("warnings", []).append(
                    f"Exact PX4 source is unavailable ({status}): {exc}"
                )
        elif source_repository_path is not None:
            inventory.setdefault("warnings", []).append(
                "Exact PX4 source is unavailable (revision_missing): "
                "log has no PX4 git hash."
            )
        else:
            inventory.setdefault("warnings", []).append(
                "Exact PX4 source is unavailable (repository_unavailable): "
                "no PX4 source repository was provided."
            )

        await RestrictedShellExecutor(
            work_dir,
            source_snapshot,
            input_paths=analysis_inputs,
        ).preflight()

        started_at = time.perf_counter()
        audit_logger.log_event(
            "agent.shell_analysis.started",
            input={
                "user_question": user_question,
                "source_snapshot": (
                    source_snapshot.identity if source_snapshot else None
                ),
            },
        )

        def shell_tool_factory(source_access: bool) -> ShellTool:
            return ShellTool(
                executor=RestrictedShellExecutor(
                    work_dir,
                    source_snapshot if source_access else None,
                    input_paths=analysis_inputs,
                ),
                needs_approval=False,
            )

        staged_result = await run_staged_analysis(
            user_question=user_question,
            inventory=inventory,
            source_snapshot=source_snapshot,
            mission_path=mission_path_obj,
            work_dir=work_dir,
            plots_dir=output_dir_obj / "plots",
            model=model or os.environ.get("OPENAI_MODEL", "gpt-5.6"),
            max_turns=max_turns,
            max_total_requests=max_total_requests,
            project_instructions=project_instructions,
            base_instructions=BASE_INSTRUCTIONS,
            audit_logger=audit_logger,
            shell_tool_factory=shell_tool_factory,
            enable_web_fallback=enable_web_fallback,
            web_search_context=web_search_context,
        )
        report = staged_result.report
        usage = staged_result.usage
        audit_logger.log_event(
            "agent.shell_analysis.finished",
            output=report.model_dump(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            usage=usage,
        )
        audit_logger.save_usage(usage)

        validation = validate_report(report)
        validation_required_downgrade = not validation.passed
        audit_logger.log_event(
            "validation.finished",
            output=validation.model_dump(),
        )
        if not validation.passed:
            report = enforce_validation_downgrades(report, validation)
            validation = validate_report(report)
            audit_logger.log_event(
                "validation_after_downgrade.finished",
                output=validation.model_dump(),
            )

        report = reconcile_report_with_evidence_state(
            report,
            evidence_state=staged_result.evidence_state,
            unresolved_reason=(
                "The generated structured report did not retain the evidence "
                "required for a confirmed causal conclusion."
                if validation_required_downgrade
                else None
            ),
        )
        _save_report(report, report_path)
        audit_logger.log_event(
            "run.finished",
            output={
                "report_path": str(report_path),
                "validation_passed": validation.passed,
                "dev_log_dir": str(audit_logger.run_dir),
            },
        )
        return report
    except Exception as exc:
        audit_logger.log_event("run.failed", error=repr(exc))
        raise


def _run_audited_stage(
    audit_logger: DeveloperAuditLogger,
    event_prefix: str,
    name: str,
    func: Any,
    input_payload: dict[str, Any],
    *args: Any,
) -> Any:
    started_at = time.perf_counter()
    audit_logger.log_event(
        f"{event_prefix}.started",
        name=name,
        input=input_payload,
    )
    try:
        result = func(*args)
    except Exception as exc:
        audit_logger.log_event(
            f"{event_prefix}.failed",
            name=name,
            error=repr(exc),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
        raise
    audit_logger.log_event(
        f"{event_prefix}.finished",
        name=name,
        output=result,
        duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
    )
    return result


def _save_report(report: FlightLogReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


async def run_analysis(args: argparse.Namespace) -> None:
    project_instructions = ""
    if args.instructions:
        instructions_path = Path(args.instructions).expanduser().resolve()
        project_instructions = instructions_path.read_text(encoding="utf-8")

    report = await analyze_flight_log(
        log_path=args.ulog,
        mission_path=args.mission,
        source_path=args.source,
        output_dir=args.output_dir,
        dev_log_root=args.dev_log_root,
        user_question=args.question,
        model=args.model,
        max_turns=args.max_turns,
        max_total_requests=args.max_total_requests,
        project_instructions=project_instructions,
        enable_web_fallback=not args.no_web_fallback,
        web_search_context=args.web_context,
    )
    print(report.model_dump_json(indent=2))
    print(f"\n[output] {Path(args.output_dir).expanduser().resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a PX4 ULog with local shell access."
    )
    parser.add_argument("--ulog", required=True, help="Path to the .ulg file")
    parser.add_argument(
        "--source",
        help=(
            "Path to a PX4-Autopilot Git repository containing the logged "
            "revision; defaults to the repository reference checkout"
        ),
    )
    parser.add_argument(
        "--mission",
        help="Optional path to a mission file",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to investigate",
    )
    parser.add_argument(
        "--instructions",
        help="Optional path to exported PX4 project instructions",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/run_001",
        help="Persistent run directory; defaults to outputs/run_001",
    )
    parser.add_argument(
        "--dev-log-root",
        default=str(DEFAULT_DEV_LOG_ROOT),
        help="Developer audit-log root",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6"),
        help="API model ID; defaults to OPENAI_MODEL or gpt-5.6",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum turns for each evidence stage",
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        help=(
            "Maximum model requests across all stages; defaults to --max-turns"
        ),
    )
    parser.add_argument(
        "--no-web-fallback",
        action="store_true",
        help="Do not run the bounded web-assisted retry after an insufficient local pass",
    )
    parser.add_argument(
        "--web-context",
        choices=("low", "medium", "high"),
        default="medium",
        help="Context size for the conditional web-search stage",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_analysis(parse_args()))
