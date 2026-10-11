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
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from agents import (
    Agent,
    FunctionTool,
    ModelSettings,
    RunContextWrapper,
    Runner,
    ShellActionRequest,
    ShellCallData,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
    WebSearchTool,
)
from agents.tool_context import ToolContext
from openai.types.shared.reasoning import Reasoning

from flight_log_agent.analysis.jev_consult import (
    build_jev_consult_tool,
    resolve_jev_mode,
)
from flight_log_agent.analysis.model_provider import (
    UNAVAILABLE,
    resolve_model_runtime,
    router_plugins_body,
    served_model_from_result,
)
from flight_log_agent.analysis.sole_selector import build_sole_selector_tool

from flight_log_agent.analysis.report_validation import (
    enforce_validation_downgrades,
    validate_report,
)
from flight_log_agent.analysis.source_tools import make_read_px4_source_file_tool
from flight_log_agent.audit import (
    AgentRunAuditHooks,
    DEFAULT_DEV_LOG_ROOT,
    DeveloperAuditLogger,
    log_run_items,
    make_json_safe,
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
from flight_log_agent.utils import json_safe_value


ALLOWED_PROGRAMS = {"python", "python3", "rg", "git", "sed", "find"}
FORBIDDEN_SHELL_TOKENS = {
    "|", "||", "&", "&&", ";",
    ">", ">>", "<", "<<", "<<<",
}
MAX_OUTPUT_CHARS = 600_000
DEFAULT_TIMEOUT_S = 1200
DEFAULT_MAX_TURNS = 30
SOURCE_ALIAS = "PX4-Autopilot"
SNAPSHOT_TOKEN = "SNAPSHOT"
SHELL_FUNCTION_TOOL_NAME = "run_shell_command"
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
SOURCE_ALIAS_MAX_CHARS = 4096
SOURCE_ALIAS_MAX_PARTS = 64
SOURCE_ALIAS_MAX_RESOLUTION_ATTEMPTS = 128
DUPLICATE_SNAPSHOT_GIT_READ_MESSAGE = (
    "Duplicate immutable snapshot Git read suppressed; reuse the earlier "
    "result for this exact command."
)


SHELL_FUNCTION_TOOL_DESCRIPTION = (
    "Run one allowlisted shell command in the analysis workspace and "
    "return its output. Use it to inspect the read-only ULog with pyulog, "
    "run Python checks, and read the commit-pinned PX4 source with "
    "'git -C PX4-Autopilot <command> ...' (SNAPSHOT is a virtual revision "
    "the runner resolves; never use HEAD, a branch, a tag, or a literal "
    "hash). Programs allowed: python, python3, rg, git, sed, find."
)

SHELL_FUNCTION_TOOL_INSTRUCTIONS = """
Run shell work with the run_shell_command function tool. Pass one command
at a time as {"command": "..."} with an optional timeout_ms. The same
workspace rules apply as with direct shell access: one command per call, no
pipes, redirection, or chaining, and read-only workspace inspection for rg,
sed, and find. Save useful plots under /plots/ and reuse results already
obtained.
"""


JEV_CONSULT_INSTRUCTIONS = """
When choosing among next investigation steps, you may call the
consult_investigation_direction function tool with your current
hypotheses (your claims), model-reported evidence (facts as you observed
them, with tool or artifact references; they are your claims until
checks verify them), uncertainties, and 2-5 candidate next checks. The
returned direction is advisory only: consider it, but only evidence and
validation decide the report. It never raises confidence and never
replaces a numeric check or source read.
"""


def _render_shell_result_text(result: ShellResult) -> str:
    lines: list[str] = []
    for output in result.output:
        lines.append(f"$ {output.command}")
        if output.stdout:
            lines.append(output.stdout)
        if output.stderr:
            lines.append(output.stderr)
        if output.outcome.type == "timeout":
            lines.append(f"timed out (exit code: {output.outcome.exit_code})")
        else:
            lines.append(f"exit code: {output.outcome.exit_code}")
    return "\n".join(lines)


def make_local_shell_function_tool(
    executor: RestrictedShellExecutor,
) -> FunctionTool:
    """Adapt the sandboxed shell executor to chat-completions transports.

    The Responses ShellTool has no Chat Completions serialization, so chat
    transports call this plain function tool instead. Execution stays local:
    every call runs through the existing executor, keeping command
    restrictions, snapshot pinning, timeouts, output limits, and the
    sandboxed working directory. Rejections and timeouts return as result
    text so the model can adjust; nothing is raised to the run loop.
    """

    async def _invoke(ctx: Any, args_json: str) -> str:
        try:
            args = json.loads(args_json)
        except ValueError:
            return "error: arguments must be a JSON object with a 'command' string"
        if (
            not isinstance(args, dict)
            or not isinstance(args.get("command"), str)
            or not args["command"].strip()
        ):
            return "error: arguments must be a JSON object with a 'command' string"
        timeout_ms = args.get("timeout_ms")
        if timeout_ms is not None and (
            isinstance(timeout_ms, bool)
            or not isinstance(timeout_ms, int)
            or timeout_ms <= 0
        ):
            return "error: 'timeout_ms' must be a positive integer of milliseconds"
        request = ShellCommandRequest(
            ctx_wrapper=RunContextWrapper(context=None),
            data=ShellCallData(
                call_id=getattr(ctx, "tool_call_id", None) or "local",
                action=ShellActionRequest(
                    commands=[args["command"]],
                    timeout_ms=timeout_ms,
                ),
            ),
        )
        result = await executor(request)
        return _render_shell_result_text(result)

    return FunctionTool(
        name=SHELL_FUNCTION_TOOL_NAME,
        description=SHELL_FUNCTION_TOOL_DESCRIPTION,
        params_json_schema={
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "One shell command, e.g. "
                        "\"python -c '...'\" or "
                        "\"git -C PX4-Autopilot grep -n -F 'term' SNAPSHOT -- src/\""
                    ),
                },
                "timeout_ms": {
                    "type": ["integer", "null"],
                    "description": (
                        "Optional per-command timeout in milliseconds; "
                        "null uses the runner default"
                    ),
                },
            },
            "required": ["command", "timeout_ms"],
            "additionalProperties": False,
        },
        on_invoke_tool=_invoke,
        needs_approval=False,
    )


class RestrictedShellExecutor:
    """Run an allowlist of commands and pin PX4 Git reads to one snapshot."""

    def __init__(
        self,
        cwd: Path,
        source_snapshot: SourceSnapshot | None = None,
        input_paths: dict[str, Path] | None = None,
        output_limit: int = MAX_OUTPUT_CHARS,
    ):
        self.cwd = cwd.resolve()
        self.output_limit = output_limit
        self._overflow_count = 0
        self.source_snapshot = source_snapshot
        self.input_paths = {
            name: path.expanduser().resolve()
            for name, path in (input_paths or {}).items()
        }
        self._completed_snapshot_git_reads: dict[tuple[str, ...], int] = {}
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

    def resolve_source_alias(self, alias: str) -> SourceSnapshot:
        """Resolve a virtual PX4 Git directory to its pinned snapshot."""

        if self.source_snapshot is None:
            raise PermissionError(
                "PX4 source is unavailable because the logged revision was not resolved."
            )
        return self._source_for_alias(alias)

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
        duplicate_snapshot_git_reads_suppressed = 0

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
                    cache_key = tuple(argv)
                    cached_exit_code = self._completed_snapshot_git_reads.get(
                        cache_key
                    )
                    if cached_exit_code is not None:
                        outputs.append(
                            ShellCommandOutput(
                                command=command,
                                stdout=DUPLICATE_SNAPSHOT_GIT_READ_MESSAGE,
                                stderr="",
                                outcome=ShellCallOutcome(
                                    type="exit",
                                    exit_code=cached_exit_code,
                                ),
                            )
                        )
                        duplicate_snapshot_git_reads_suppressed += 1
                        continue
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
                if (
                    argv[0] == "git"
                    and not timed_out
                    and (
                        exit_code == 0
                        or (argv[3] == "grep" and exit_code == 1)
                    )
                ):
                    self._completed_snapshot_git_reads[tuple(argv)] = exit_code

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
                "duplicate_snapshot_git_reads_suppressed": (
                    duplicate_snapshot_git_reads_suppressed
                ),
            },
        )

    async def _run_sandboxed_process(
        self,
        argv: list[str],
        *,
        timeout_s: float,
    ) -> tuple[bytes, bytes, int, bool]:
        sandbox_argv = self._build_sandbox_command(argv)
        return await self._run_process(
            sandbox_argv,
            timeout_s=timeout_s,
            cwd=None,
            env=self._restricted_env(),
        )

    async def preflight(self) -> None:
        checks = ["import os", "from pathlib import Path"]
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
                "The analysis shell sandbox is unavailable"
                + (f": {detail}" if detail else ".")
            )

    def _build_sandbox_command(
        self,
        argv: list[str],
    ) -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError(
                "bubblewrap is required for workspace-scoped shell execution."
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
            ]
        )

        if self.input_paths:
            command.extend(["--dir", "/inputs"])
        for input_name, input_path in self.input_paths.items():
            target = input_path.resolve(strict=True)
            command.extend(
                ["--ro-bind", str(target), f"/inputs/{input_name}"]
            )

        command.extend(
            [
                "--setenv",
                "PATH",
                "/venv/bin:/usr/bin:/bin",
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
        except asyncio.TimeoutError:
            if process.returncode is None:
                process.kill()
            stdout, stderr = await communication
            return stdout, stderr, 124, True
        except asyncio.CancelledError:
            if process.returncode is None:
                process.kill()
            await asyncio.shield(communication)
            raise

    def _truncate(self, text: str) -> str:
        if len(text) <= self.output_limit:
            return text
        omitted = len(text) - self.output_limit
        self._overflow_count += 1
        overflow_dir = self.cwd / "shell_outputs"
        overflow_dir.mkdir(parents=True, exist_ok=True)
        overflow_path = overflow_dir / f"cmd_{self._overflow_count:03d}.txt"
        overflow_path.write_text(text, encoding="utf-8")
        return (
            text[: self.output_limit]
            + f"\n\n[output truncated; {omitted} characters omitted; "
            f"full output saved to shell_outputs/{overflow_path.name} - "
            "read bounded slices with Python]"
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
        return env


BASE_INSTRUCTIONS = """
You are the sole analyst for a PX4 flight log. Keep the complete investigation
in this one agent context: inspect the log, inspect the exact source snapshot,
form explanations, run checks, reconsider them, and write the final report.

Available workspace entries:
- /inputs/flight.ulg: the read-only input log (also FLIGHT_LOG_ULOG)
- /inputs/mission.plan: the optional read-only mission input, when supplied
- the current directory: writable per-run scripts and intermediate data
- /plots/: persistent plot output

Use Python with pyulog to inspect the actual ULog. Write and run your own
Python scripts whenever calculations, time alignment, plots, or hypothesis
tests would help answer the user's question. Choose the investigation from the
question and the evidence you discover; do not rely on a fixed mechanism
catalog, topic list, or module-specific procedure.

When an exact PX4 source snapshot is available, read it with the separate
commit-pinned Git commands:
- Search: git -C PX4-Autopilot grep -n -F "term" SNAPSHOT -- src/
- Scoped search with context: git -C PX4-Autopilot grep -n -F -m 5 -C 20 "term" SNAPSHOT -- src/module/
- Full-file read: git -C PX4-Autopilot show SNAPSHOT:src/module/file.cpp
- List: git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/
- Full-schema read: git -C PX4-Autopilot show SNAPSHOT:msg/VehicleStatus.msg

SNAPSHOT is a virtual revision token. The executor replaces it with the
resolved full commit SHA. Never use HEAD, a branch, a tag, or a literal hash.
The examples are not a source-path limit; inspect any snapshot-relative source,
schema, build, or configuration file that is relevant.
For an initialized submodule, use its path as the virtual Git directory, for
example git -C PX4-Autopilot/src/modules/mavlink/mavlink show SNAPSHOT:path.
The executor resolves SNAPSHOT to the gitlink commit recorded by the parent.

After Git search identifies a relevant file and line, prefer the
read_px4_source_file tool when only a bounded region is needed. Give it the
same virtual Git directory, a path relative to that repository, and inclusive
start and end lines. Prefer scoped git grep with -C when matching lines and
their context can answer the source question in one call. Request multiple
independent narrow excerpts in the same turn when their locations are known.
Avoid sequential 200-line paging and oversized ranges: read the relevant
function or region around a hit. next_start_line only identifies a requested
excerpt clipped by output limits; follow it only if the omitted lines matter.
Use git show when the complete file is materially useful.

Python, rg, sed, and find run in a filesystem sandbox containing only this
analysis workspace, its read-only input files, the plot directory, and runtime
libraries. They cannot see the PX4 checkout. Use the separate Git tool calls
for source and Python for ULog analysis.

Start with local evidence. Quantify the symptom in the ULog, trace a relevant
source mechanism when source is available, and run calculations that could
distinguish that mechanism from plausible alternatives. Actively try to
falsify the leading explanation instead of merely collecting supporting facts.
Source code establishes possible firmware behavior, not that the behavior
occurred in this flight. Keep ULog observations, source facts, calculations,
inferences, contradicting evidence, and unresolved questions distinct.

If web search is available, use it only after the ULog and exact source leave a
material external-context gap. Web results cannot replace flight evidence or
the resolved source snapshot. If the available evidence cannot discriminate
between explanations, say so and lower confidence rather than forcing an
answer.

Before every additional tool call, perform a completion check: can the
available evidence already answer the user's question with uncertainty
represented honestly, and could the proposed call materially change the
conclusion or confidence? If the question can already be answered and the call
is not material, stop using tools and return FlightLogReport immediately.
Continue only to resolve a specific material uncertainty; do not use tools for
redundant confirmation, cosmetic improvements, or artifact housekeeping.

Save useful plots under /plots/. Keep command output relevant enough to reason
from, and reuse results already obtained. The executor accepts one command at
a time and rejects pipes, redirection, and command chaining.
- To create a temporary script, use Python, for example:
  python -c "from pathlib import Path; Path('analysis.py').write_text('...')"

For every hypothesis grounded in source behavior, cite the exact file,
function, and line range in its source references, quote the decisive
lines in the reference snippet, and explain what the code does in the
reference explanation. Generate one evidence plot per ranked hypothesis
whenever the log contains the needed signals: time-series windows
around the key event work well. Reference each produced plot from the
hypothesis it supports, describing its purpose. When a plot cannot be
produced (missing signals, empty windows, or a rendering failure),
leave its reference out and state the reason plainly in the hypothesis
evidence instead of inventing a plot.
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
    model_provider: str | None = None,
    model_transport: str | None = None,
    reasoning_effort: str | None = None,
    openrouter_api_key: str | None = None,
    router_cost_tier: str | None = None,
    shell_output_limit: int | None = None,
    jev_consult: bool | str | None = None,
    jev_api_key: str | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
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
    if shell_output_limit is None:
        shell_output_env = os.environ.get("FLIGHT_LOG_SHELL_OUTPUT_LIMIT")
        shell_output_limit = (
            int(shell_output_env) if shell_output_env else MAX_OUTPUT_CHARS
        )
    if shell_output_limit < 1000:
        raise ValueError("shell_output_limit must be at least 1000 characters")
    jev_mode = resolve_jev_mode(
        jev_consult, os.environ.get("FLIGHT_LOG_JEV_CONSULT")
    )

    model_name = model or os.environ.get("OPENAI_MODEL", "gpt-5.6")
    work_dir = prepare_workspace(
        log_path_obj,
        output_dir_obj,
        mission_path_obj,
    )
    audit_logger = DeveloperAuditLogger(Path(dev_log_root), run_id=dev_run_id)
    # Local configuration errors (unknown provider/transport/effort,
    # empty model, missing key) raise here, before any model request.
    # Remote errors (authentication, unknown model, unreachable endpoint)
    # surface from the provider at run time through run.failed.
    model_runtime = resolve_model_runtime(
        model=model_name,
        model_provider=model_provider,
        model_transport=model_transport,
        reasoning_effort=reasoning_effort,
        openrouter_api_key=openrouter_api_key,
        router_cost_tier=router_cost_tier,
        response_sink=lambda fields: audit_logger.log_event(
            "model.provider_response", **fields
        ),
    )
    web_search_effective = (
        enable_web_fallback and model_runtime.web_search_supported
    )
    turn_limit = (
        min(max_turns, max_total_requests)
        if max_total_requests is not None
        else max_turns
    )
    analysis_inputs = {"flight.ulg": log_path_obj}
    if mission_path_obj is not None:
        analysis_inputs["mission.plan"] = mission_path_obj
    report_path = output_dir_obj / "report.json"
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
            "analysis_architecture": "single_agent_v1",
            "model": model_name,
            "reasoning_effort": model_runtime.reasoning_effort,
            "requested_model_provider": model_runtime.provider,
            "requested_model": model_runtime.model,
            "requested_reasoning_effort": model_runtime.reasoning_effort,
            "model_transport": model_runtime.transport,
            "requested_router_cost_tier": model_runtime.router_cost_tier,
            "web_search_requested": enable_web_fallback,
            "web_search_effective": web_search_effective,
            # Served identity is known only after the run; "unavailable"
            # until the post-run update below records what was exposed.
            "served_model": UNAVAILABLE,
            "served_reasoning_effort": UNAVAILABLE,
            "enable_web_fallback": enable_web_fallback,
            "web_search_context": web_search_context,
            "max_turns": turn_limit,
            "shell_output_limit": shell_output_limit,
            "requested_jev_consult": jev_mode != "off",
            "requested_jev_mode": jev_mode,
            "max_total_requests": max_total_requests,
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

        model_inventory = _model_inventory(inventory, source_repository_path)
        (work_dir / "ulog_inventory.json").write_text(
            json.dumps(model_inventory, default=json_safe_value, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )

        shell_executor = RestrictedShellExecutor(
            work_dir,
            source_snapshot,
            input_paths=analysis_inputs,
            output_limit=shell_output_limit,
        )
        await shell_executor.preflight()

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

        instructions = BASE_INSTRUCTIONS
        if project_instructions:
            instructions += (
                "\n\nPX4 PROJECT INSTRUCTIONS\n"
                "========================\n"
                + project_instructions
            )

        # Native ShellTool lives only on the established default OpenAI
        # Responses path. Every OpenRouter configuration and every chat
        # transport runs the same sandboxed executor behind a plain
        # function tool with matching call instructions.
        use_responses_shell = (
            model_runtime.provider == "openai"
            and model_runtime.transport == "responses"
        )
        if not use_responses_shell:
            instructions += SHELL_FUNCTION_TOOL_INSTRUCTIONS

        # Sole mode removes direct shell/source execution: every command
        # Luna runs must pass through one select_and_run_check decision
        # (JEV-selected or audited Luna fallback pick). Ordinary tools
        # remain for exploration, plotting, and unsubmitted checks in all
        # other modes.
        tools: list[Any] = []
        if jev_mode != "sole":
            tools.extend(
                [
                    ShellTool(
                        executor=shell_executor,
                        needs_approval=False,
                    )
                ]
                if use_responses_shell
                else [make_local_shell_function_tool(shell_executor)]
            )
            if source_snapshot is not None:
                tools.append(
                    make_read_px4_source_file_tool(
                        shell_executor.resolve_source_alias,
                    )
                )
        if enable_web_fallback and model_runtime.web_search_supported:
            tools.append(
                WebSearchTool(
                    search_context_size=web_search_context,
                    external_web_access=True,
                )
            )
        if jev_mode == "voluntary":
            instructions += JEV_CONSULT_INSTRUCTIONS
            tools.append(
                build_jev_consult_tool(
                    api_key=jev_api_key or os.environ.get("OPENROUTER_API_KEY"),
                    audit_sink=lambda fields: audit_logger.log_event(
                        "decision.consult", **fields
                    ),
                )
            )

        if jev_mode == "sole":
            instructions += (
                "\n\nYou have no direct shell or source-reading tools: every "
                "shell command you run must go through one "
                "select_and_run_check call as a candidate check, and only "
                "the selected check executes. When several next checks are "
                "possible, call it once with the candidate shell commands. "
                "Pass luna_selection with a candidate id only after a "
                "fallback to run your own pick. Evidence plots, including "
                "matplotlib figures saved under /plots/, are also produced "
                "through candidate checks: propose a plot-generating command "
                "whenever a hypothesis needs visual evidence."
            )
            tools.append(
                build_sole_selector_tool(
                    executor=shell_executor,
                    api_key=jev_api_key or os.environ.get("OPENROUTER_API_KEY"),
                    audit_sink=lambda fields: audit_logger.log_event(
                        "decision.select", **fields
                    ),
                    # Sole mode bounds total work through model turns, not
                    # through a consultation count: votes stay unbounded.
                    max_consults=None,
                )
            )

        # Router plugin fragments travel only to OpenRouter; other
        # providers never receive unknown body fields.
        router_body = (
            router_plugins_body(model_runtime.router_cost_tier)
            if model_runtime.provider == "openrouter"
            else None
        )
        agent = Agent(
            name="PX4 ULog Analyst",
            model=model_name,
            instructions=instructions,
            tools=tools,
            model_settings=ModelSettings(
                reasoning=Reasoning(effort=model_runtime.reasoning_effort),
                extra_body=router_body,
            ),
            output_type=FlightLogReport,
        )
        prompt = _analysis_prompt(
            user_question=user_question,
            inventory=model_inventory,
            source_snapshot=source_snapshot,
            mission_path=mission_path_obj,
            plots_dir=output_dir_obj / "plots",
            web_search_available=web_search_effective,
        )
        result = await Runner.run(
            agent,
            prompt,
            max_turns=turn_limit,
            hooks=AgentRunAuditHooks(audit_logger),
            run_config=model_runtime.run_config,
        )
        log_run_items(audit_logger, getattr(result, "new_items", []) or [])
        served_model = served_model_from_result(result)
        audit_logger.log_event(
            "model.served",
            requested_provider=model_runtime.provider,
            requested_model=model_runtime.model,
            requested_reasoning_effort=model_runtime.reasoning_effort,
            # Result-level identity where the SDK exposes it; per-request
            # served values are not exposed, so record them as unavailable
            # rather than inferring them.
            served_model=served_model,
            served_reasoning_effort=UNAVAILABLE,
        )
        audit_logger.update_metadata(
            {
                "served_model": served_model,
                "served_reasoning_effort": UNAVAILABLE,
            }
        )
        report = (
            result.final_output
            if isinstance(result.final_output, FlightLogReport)
            else FlightLogReport.model_validate(result.final_output)
        )
        report = _normalize_report_plot_paths(
            report,
            plots_dir=output_dir_obj / "plots",
        )
        usage = getattr(getattr(result, "context_wrapper", None), "usage", None)
        audit_logger.log_event(
            "agent.shell_analysis.finished",
            output=report.model_dump(),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 3),
            usage=usage,
        )
        audit_logger.save_usage(usage)

        validation = validate_report(report)
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

        if source_snapshot is not None:
            report.source_commit = source_snapshot.commit_sha
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
    finally:
        if model_runtime.aclose is not None:
            await model_runtime.aclose()


def _model_inventory(
    inventory: dict[str, Any],
    source_repository_path: Path | None,
) -> dict[str, Any]:
    """Copy the full inventory for the model without host repository paths."""

    repository_paths = sorted(
        {
            str(path)
            for path in (source_repository_path, inventory.get("source_path"))
            if path
        },
        key=len,
        reverse=True,
    )

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        if isinstance(value, str):
            for path in repository_paths:
                value = value.replace(path, SOURCE_ALIAS)
            return value
        return value

    return sanitize({
        key: value for key, value in inventory.items() if key != "source_path"
    })


def _analysis_prompt(
    *,
    user_question: str,
    inventory: dict[str, Any],
    source_snapshot: SourceSnapshot | None,
    mission_path: Path | None,
    plots_dir: Path,
    web_search_available: bool,
) -> str:
    compact_inventory = {
        "firmware_version": inventory.get("firmware_version"),
        "firmware_branch": inventory.get("firmware_branch"),
        "logged_git_hash": inventory.get("git_hash"),
        "resolved_source_commit": (
            source_snapshot.commit_sha if source_snapshot else None
        ),
        "airframe": inventory.get("airframe"),
        "duration_s": inventory.get("duration_s"),
        "available_topics": inventory.get("available_topics") or [],
        "missing_topics": inventory.get("missing_topics") or [],
        "warnings": inventory.get("warnings") or [],
    }
    source_note = (
        "The exact source snapshot is available through SNAPSHOT."
        if source_snapshot is not None
        else (
            "No exact source snapshot is available. Do not make claims that "
            "require source confirmation."
        )
    )
    mission_note = (
        "/inputs/mission.plan is available."
        if mission_path is not None
        else "No mission file was supplied."
    )
    web_note = (
        "Web search is available if local evidence leaves an external-context gap."
        if web_search_available
        else "Web search is disabled for this run."
    )
    return f"""
Use pyulog and the available tools to answer the user's question about
/inputs/flight.ulg.

User question:
{user_question}

Deterministic prepass:
{json.dumps(make_json_safe(compact_inventory), indent=2, sort_keys=True)}

The full parsed inventory is available at /work/ulog_inventory.json. Query it
selectively with Python for parameters, topic fields, topic instances and sample
counts, logged messages, warnings, firmware metadata, and the resolved source
commit when available. Avoid printing the entire inventory. Use
/inputs/flight.ulg with pyulog for actual time-series calculations.

{source_note}
{mission_note}
{web_note}

Do the analysis yourself in this context. Generate and run Python checks as
needed, inspect the exact source with the pinned Git commands, test the leading
explanation against alternatives, and try to falsify it before concluding.
Return the existing FlightLogReport schema. Save plots under /plots/ and use
/plots/<relative-path> in plot references; the runner maps produced artifacts
to {plots_dir.resolve()}. Do not write report.json yourself.
"""


def _normalize_report_plot_paths(
    report: FlightLogReport,
    *,
    plots_dir: Path,
) -> FlightLogReport:
    plot_root = plots_dir.resolve()
    available = {
        f"/plots/{path.resolve().relative_to(plot_root).as_posix()}": str(
            path.resolve()
        )
        for path in sorted(plot_root.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(plot_root)
    }
    for hypothesis in report.ranked_hypotheses:
        for plot in hypothesis.plots:
            if not plot.path:
                continue
            host_path = available.get(plot.path)
            if host_path is None:
                plot.warnings.append(
                    "The report referenced a plot that was not produced "
                    "during this analysis."
                )
                plot.path = ""
            else:
                plot.path = host_path
    return report


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
        model_provider=args.model_provider,
        model_transport=args.model_transport,
        reasoning_effort=args.reasoning_effort,
        router_cost_tier=args.router_cost_tier,
        shell_output_limit=args.shell_output_limit,
        jev_consult=args.jev_consult,
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
        "--model-provider",
        choices=("auto", "openai", "openrouter"),
        default=None,
        help=(
            "Model provider; defaults to FLIGHT_LOG_MODEL_PROVIDER or "
            "'auto' (current OpenAI behavior)"
        ),
    )
    parser.add_argument(
        "--model-transport",
        choices=("auto", "responses", "chat"),
        default=None,
        help=(
            "Provider transport; 'auto' keeps the SDK default for OpenAI "
            "and uses chat completions for OpenRouter"
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default=None,
        help=(
            "Reasoning effort; defaults to FLIGHT_LOG_REASONING_EFFORT "
            "or 'high'"
        ),
    )
    parser.add_argument(
        "--router-cost-tier",
        choices=("low", "medium", "high"),
        default=None,
        help=(
            "Router cost-tier policy; defaults to "
            "FLIGHT_LOG_ROUTER_COST_TIER or unset"
        ),
    )
    parser.add_argument(
        "--shell-output-limit",
        type=int,
        default=None,
        help=(
            "Per-command tool-output cap in characters sent to the model; "
            "full output is saved under work/shell_outputs/. Defaults to "
            "FLIGHT_LOG_SHELL_OUTPUT_LIMIT or 600000"
        ),
    )
    parser.add_argument(
        "--jev-consult",
        nargs="?",
        const="voluntary",
        default=None,
        choices=("off", "voluntary", "sole"),
        help=(
            "Attach JEV direction consultation; 'sole' routes all command "
            "execution through select_and_run_check. Defaults to "
            "FLIGHT_LOG_JEV_CONSULT or off"
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Maximum turns in the single analysis run",
    )
    parser.add_argument(
        "--max-total-requests",
        type=int,
        help=(
            "Optional request cap; the lower of this and --max-turns is used"
        ),
    )
    parser.add_argument(
        "--no-web-fallback",
        action="store_true",
        help="Do not make web search available to the analysis agent",
    )
    parser.add_argument(
        "--web-context",
        choices=("low", "medium", "high"),
        default="medium",
        help="Context size when the agent chooses web search",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_analysis(parse_args()))
