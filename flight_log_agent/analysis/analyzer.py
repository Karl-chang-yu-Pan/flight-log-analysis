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
import shlex
import time
from pathlib import Path, PurePosixPath
from typing import Any

from agents import (
    Agent,
    Runner,
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
from flight_log_agent.audit import (
    DEFAULT_DEV_LOG_ROOT,
    AgentRunAuditHooks,
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


class RestrictedShellExecutor:
    """Run an allowlist of commands and pin PX4 Git reads to one snapshot."""

    def __init__(
        self,
        cwd: Path,
        source_snapshot: SourceSnapshot | None = None,
    ):
        self.cwd = cwd.resolve()
        self.source_snapshot = source_snapshot

    def _parse_command(self, command: str) -> list[str]:
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

        if any(token in FORBIDDEN_SHELL_TOKENS for token in argv):
            raise PermissionError(
                "Shell operators, redirection, pipes, and command chaining "
                "are disabled. Run one command at a time."
            )

        if program == "git":
            return self._snapshot_git_command(argv)
        return argv

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

    def _source_for_alias(self, alias: str) -> SourceSnapshot:
        normalized = PurePosixPath(str(alias).replace("\\", "/"))
        parts = normalized.parts
        if (
            normalized.is_absolute()
            or not parts
            or parts[0] != SOURCE_ALIAS
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise PermissionError(
                f"Git -C must name {SOURCE_ALIAS} or a resolved submodule beneath it."
            )

        source = self.source_snapshot
        remaining = list(parts[1:])
        while remaining:
            resolved = None
            for end in range(len(remaining), 0, -1):
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
        matches = [
            index
            for index, token in enumerate(args)
            if token == SNAPSHOT_TOKEN or token.startswith(f"{SNAPSHOT_TOKEN}:")
        ]
        if len(matches) != 1:
            raise PermissionError(
                f"Git {subcommand} must contain exactly one {SNAPSHOT_TOKEN} revision."
            )

        index = matches[0]
        token = args[index]
        if subcommand == "show":
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
            if len(args) != 1:
                raise PermissionError(
                    "Git show accepts exactly one SNAPSHOT:relative/path object."
                )
            replacement = f"{commit_sha}:{normalized_path.as_posix()}"
        else:
            if token != SNAPSHOT_TOKEN:
                raise PermissionError(
                    f"Git {subcommand} must use {SNAPSHOT_TOKEN} as its revision."
                )
            if subcommand == "grep":
                RestrictedShellExecutor._validate_git_grep_args(args, index)
            elif subcommand == "ls-tree":
                RestrictedShellExecutor._validate_git_ls_tree_args(args, index)
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
                argv = self._parse_command(command)

                proc = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=self.cwd,
                    env=self._restricted_env(),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(),
                        timeout=timeout_s,
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    stdout_bytes, stderr_bytes = await proc.communicate()
                    timed_out = True

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
                            exit_code=proc.returncode,
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
        if self.source_snapshot is not None:
            env.update(
                {
                    "FLIGHT_LOG_PX4_REPOSITORY": str(
                        self.source_snapshot.repository_path
                    ),
                    "FLIGHT_LOG_PX4_COMMIT": self.source_snapshot.commit_sha,
                }
            )
        return env


BASE_INSTRUCTIONS = """
You are analyzing a PX4 ULog against its exact PX4-Autopilot source revision.

Available workspace entries:
- flight.ulg: the input log
- mission.plan: the optional mission input, when supplied
- the current directory: writable per-run scripts and intermediate data
- ../plots/: persistent plot output

Use Python with pyulog to inspect the actual ULog. Decide which commands and
analyses are needed from the user's question.

PX4 source is not exposed as a working-tree directory. Read it through the
executor's commit-pinned Git interface:
- Search: git -C PX4-Autopilot grep -n -F "term" SNAPSHOT -- src/
- Read: git -C PX4-Autopilot show SNAPSHOT:src/module/file.cpp
- List: git -C PX4-Autopilot ls-tree -r --name-only SNAPSHOT -- src/

SNAPSHOT is a virtual revision token. The executor replaces it with the
resolved full commit SHA. Never use HEAD, a branch, a tag, or a literal hash.
For an initialized submodule, use its path as the virtual Git directory, for
example git -C PX4-Autopilot/src/modules/mavlink/mavlink show SNAPSHOT:path.
The executor resolves SNAPSHOT to the gitlink commit recorded by the parent.

rg, sed, and find operate only on ordinary files in this analysis workspace.
They do not read PX4 source. Python may invoke commit-addressed Git operations
using FLIGHT_LOG_PX4_REPOSITORY and FLIGHT_LOG_PX4_COMMIT when a more complex,
in-memory source calculation is necessary. Such Python must use the provided
commit exactly and must not modify the repository.

Mandatory behavior:
1. Inspect the ULog before making claims about available topics or fields.
2. Read numeric data with pyulog; do not infer log values from generic PX4
   knowledge.
3. Inspect the supplied source for the exact code path relevant to the
   question. Do not silently substitute another PX4 version.
4. Where possible, compare the firmware/version information in the ULog with
   the source repository commit.
5. Keep shell output compact. Never print complete time-series arrays,
   complete parameter sets, or broad recursive source dumps.
6. Use compact statistics, transition timestamps, extrema, and small
   time-window samples. Save plots under ../plots/.
7. Clearly distinguish:
   - observed in the ULog
   - confirmed from supplied source
   - inference
   - unknown or missing evidence
8. Answer the user's actual question directly.
9. Do not provide tuning, code changes, or flight-test suggestions unless the
   user asks for them.
10. You have no access to previous ChatGPT sessions unless their instructions
    or content are included in this run.

Command constraints:
- Run one command per shell-tool command.
- No pipes, shell redirection, or command chaining.
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

    _ensure_input_link(work_dir / "flight.ulg", ulog_path)
    mission_link = work_dir / "mission.plan"
    if mission_path is not None:
        _ensure_input_link(mission_link, mission_path)
    elif mission_link.is_symlink():
        mission_link.unlink()
    elif mission_link.exists():
        raise FileExistsError(
            f"Analysis workspace contains an unmanaged mission input: {mission_link}"
        )
    return work_dir


def _ensure_input_link(link: Path, target: Path) -> None:
    target = target.expanduser().resolve()
    if link.is_symlink():
        if link.resolve(strict=False) == target:
            return
        link.unlink()
    elif link.exists():
        raise FileExistsError(
            f"Analysis workspace contains an unmanaged input path: {link}"
        )
    link.symlink_to(target, target_is_directory=target.is_dir())


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
    project_instructions: str = "",
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

    work_dir = prepare_workspace(
        log_path_obj,
        output_dir_obj,
        mission_path_obj,
    )
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

        instructions = BASE_INSTRUCTIONS
        if project_instructions:
            instructions += (
                "\n\nPX4 PROJECT INSTRUCTIONS\n"
                "===========================\n"
                + project_instructions
            )

        agent = Agent(
            name="PX4 ULog Analyst",
            model=model or os.environ.get("OPENAI_MODEL", "gpt-5.6"),
            instructions=instructions,
            tools=[
                ShellTool(
                    executor=RestrictedShellExecutor(work_dir, source_snapshot),
                    needs_approval=False,
                )
            ],
            output_type=FlightLogReport,
        )
        prompt = _analysis_prompt(
            user_question=user_question,
            inventory=inventory,
            source_snapshot=source_snapshot,
            mission_path=mission_path_obj,
            plots_dir=output_dir_obj / "plots",
        )

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
        result = await Runner.run(
            agent,
            prompt,
            max_turns=max_turns,
            hooks=AgentRunAuditHooks(audit_logger),
        )
        log_run_items(audit_logger, getattr(result, "new_items", []) or [])
        report = (
            result.final_output
            if isinstance(result.final_output, FlightLogReport)
            else FlightLogReport.model_validate(result.final_output)
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


def _analysis_prompt(
    *,
    user_question: str,
    inventory: dict[str, Any],
    source_snapshot: SourceSnapshot | None,
    mission_path: Path | None,
    plots_dir: Path,
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
    mission_note = (
        "mission.plan is available in the workspace."
        if mission_path is not None
        else "No mission file was supplied."
    )
    source_note = (
        f"Source snapshot resolved to {source_snapshot.commit_sha}."
        if source_snapshot is not None
        else "No exact source snapshot is available; do not make source-backed claims."
    )
    return f"""
Analyze flight.ulg and answer the user's question.

User question:
{user_question}

Deterministic prepass:
{json.dumps(make_json_safe(compact_inventory), indent=2, sort_keys=True)}

{source_note}
{mission_note}
Persistent plot directory: {plots_dir}

Use shell analysis to gather the necessary numeric and source evidence. Return
the existing FlightLogReport schema. Put material limitations in unresolved
evidence and the final summary. Plot references must point into the persistent
plot directory. Do not write report.json yourself; the runner validates and
saves the structured response.
"""


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
        project_instructions=project_instructions,
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
        help="Maximum agent turns",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run_analysis(parse_args()))
