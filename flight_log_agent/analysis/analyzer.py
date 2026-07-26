#!/usr/bin/env python3
"""
Minimal PX4 ULog analysis agent using the OpenAI Agents SDK and a local shell.

The agent receives:
  1. A PX4 ULog path
  2. A PX4-Autopilot source directory
  3. A user question

It may run only:
  python, python3, rg, git, sed, find

This is an internal-development example, not a hardened security sandbox.
Run it under a dedicated user or container if the input or prompt is untrusted.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shlex
import shutil
import tempfile
from pathlib import Path

from agents import (
    Agent,
    Runner,
    ShellCallOutcome,
    ShellCommandOutput,
    ShellCommandRequest,
    ShellResult,
    ShellTool,
)


ALLOWED_PROGRAMS = {"python", "python3", "rg", "git", "sed", "find"}
FORBIDDEN_SHELL_TOKENS = {
    "|", "||", "&", "&&", ";",
    ">", ">>", "<", "<<", "<<<",
}
MAX_OUTPUT_CHARS = 600_000
DEFAULT_TIMEOUT_S = 1200


class RestrictedShellExecutor:
    """Run a small allowlist of commands without invoking a shell."""

    def __init__(self, cwd: Path):
        self.cwd = cwd.resolve()

    @staticmethod
    def _parse_command(command: str) -> list[str]:
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

        return argv

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
            provider_data={"working_directory": str(self.cwd)},
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

    @staticmethod
    def _restricted_env() -> dict[str, str]:
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
        return {key: value for key, value in os.environ.items() if key in keep}


BASE_INSTRUCTIONS = """
You are analyzing a PX4 ULog against a supplied PX4-Autopilot source tree.

Available workspace entries:
- flight.ulg: the input log
- PX4-Autopilot: the source tree
- output/: writable analysis output

Use Python with pyulog to inspect the actual ULog. Use rg, git, sed, and find
to inspect the supplied source. Decide which commands and analyses are needed
from the user's question.

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
   time-window samples. Save large intermediate data or plots under output/.
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
  python -c "from pathlib import Path; Path('work/a.py').write_text('...')"
"""


def prepare_workspace(
    ulog_path: Path,
    source_path: Path,
    workspace_root: Path | None,
) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if workspace_root is None:
        temporary = tempfile.TemporaryDirectory(prefix="px4-ulog-agent-")
        workspace = Path(temporary.name)
    else:
        temporary = None
        workspace = workspace_root.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

    output_dir = workspace / "output"
    work_dir = workspace / "work"
    output_dir.mkdir(exist_ok=True)
    work_dir.mkdir(exist_ok=True)

    log_link = workspace / "flight.ulg"
    source_link = workspace / "PX4-Autopilot"

    for link in (log_link, source_link):
        if link.is_symlink() or link.exists():
            if link.is_dir() and not link.is_symlink():
                shutil.rmtree(link)
            else:
                link.unlink()

    log_link.symlink_to(ulog_path.resolve())
    source_link.symlink_to(source_path.resolve(), target_is_directory=True)

    return workspace, temporary


async def run_analysis(args: argparse.Namespace) -> None:
    ulog_path = Path(args.ulog).expanduser().resolve()
    source_path = Path(args.source).expanduser().resolve()

    if not ulog_path.is_file():
        raise FileNotFoundError(f"ULog not found: {ulog_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"PX4 source directory not found: {source_path}")

    project_instructions = ""
    if args.instructions:
        instructions_path = Path(args.instructions).expanduser().resolve()
        project_instructions = instructions_path.read_text(encoding="utf-8")

    workspace_root = (
        Path(args.workspace).expanduser().resolve()
        if args.workspace
        else None
    )
    workspace, temporary = prepare_workspace(
        ulog_path=ulog_path,
        source_path=source_path,
        workspace_root=workspace_root,
    )

    try:
        instructions = BASE_INSTRUCTIONS
        if project_instructions:
            instructions += (
                "\n\nPX4 PROJECT INSTRUCTIONS\n"
                "===========================\n"
                + project_instructions
            )

        agent = Agent(
            name="PX4 ULog Analyst",
            model=args.model,
            instructions=instructions,
            tools=[
                ShellTool(
                    executor=RestrictedShellExecutor(workspace),
                    needs_approval=False,
                )
            ],
        )

        prompt = f"""
Analyze the supplied flight.ulg and PX4-Autopilot source tree.

User question:
{args.question}

Provide the final analysis in the response. Mention any material limitation,
including missing topics, unmatched source revision, or insufficient evidence.
"""

        result = await Runner.run(
            agent,
            prompt,
            max_turns=args.max_turns,
        )

        print(result.final_output)
        print(f"\n[workspace] {workspace}")

        # Keep a permanent copy of the final answer when --workspace is used.
        if workspace_root is not None:
            (workspace / "output" / "final_answer.md").write_text(
                str(result.final_output),
                encoding="utf-8",
            )

    finally:
        # A TemporaryDirectory is removed after the run. Pass --workspace to
        # preserve generated scripts, plots, and outputs.
        if temporary is not None:
            temporary.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a PX4 ULog with local shell access."
    )
    parser.add_argument("--ulog", required=True, help="Path to the .ulg file")
    parser.add_argument(
        "--source",
        required=True,
        help="Path to the matching PX4-Autopilot source tree",
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
        "--workspace",
        help="Optional persistent workspace directory",
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
