"""Sole-selector direction tool: JEV picks, the tool runs the winner.

Unlike the advisory consult tool (Luna may ignore the vote), a valid
JEV selection here executes directly through the existing sandboxed
executor within the same decision: only the winner runs, and the other
candidates cannot execute through this path. Luna proposes feasible
checks; it never ranks or approves the selection afterward.

Fallbacks return the candidate set for Luna to choose from with its
ordinary tools; a follow-up invocation carrying Luna's explicit
selection executes it through the same safeguards without spending
consultation budget or re-calling JEV.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agents import FunctionTool, ShellActionRequest, ShellCallData, ShellCommandRequest
from agents.run_context import RunContextWrapper

from flight_log_agent.analysis.jev_consult import (
    JEV_DECISIONS_MODEL,
    MAX_JEV_CONSULTS,
    UNAVAILABLE,
    _post_decision_vote,
    build_decision_state,
)

SOLE_SELECTOR_TOOL_NAME = "select_and_run_check"


def build_sole_selector_tool(
    *,
    executor: Any,
    api_key: Optional[str],
    audit_sink: Callable[[dict[str, Any]], None],
    model: str = JEV_DECISIONS_MODEL,
    http_transport: Any = None,
    max_consults: Optional[int] = MAX_JEV_CONSULTS,
    calls_made: Optional[dict[str, int]] = None,
) -> FunctionTool:
    """Build the select-and-run direction tool around an executor.

    max_consults=None means unbounded: every vote is attempted while
    the run continues. Advisory modes keep the default cap.
    """
    if calls_made is None:
        calls_made = {"n": 0}
    # Pending fallback decision: {id: command} recorded when a JEV vote
    # was attempted but unusable, opening exactly one Luna selection from
    # the same candidate set. Consumed by dispatch, cleared by resolution.
    pending_state: dict[str, Any] = {"pending": None}

    async def _invoke(ctx: Any, args_json: str) -> str:
        return await _select_and_run(
            ctx,
            args_json,
            executor=executor,
            api_key=api_key,
            model=model,
            http_transport=http_transport,
            max_consults=max_consults,
            calls_made=calls_made,
            pending_state=pending_state,
            audit_sink=audit_sink,
        )

    return FunctionTool(
        name=SOLE_SELECTOR_TOOL_NAME,
        description=(
            "Propose 2-5 feasible next checks as runnable shell commands. "
            "JEV selects exactly one and this tool runs only the winner "
            "locally through the sandboxed executor, returning its output. "
            "Pass luna_selection with a candidate id to run your own pick "
            "after a fallback instead of calling JEV again."
        ),
        params_json_schema={
            "type": "object",
            "properties": {
                "hypotheses": {
                    "type": "array",
                    "description": "Candidate explanations (model claims).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {"type": "string"},
                            "context": {"type": ["string", "null"]},
                        },
                        "required": ["claim", "context"],
                        "additionalProperties": False,
                    },
                },
                "verified_evidence": {
                    "type": "array",
                    "description": (
                        "Model-reported facts with artifact or tool "
                        "references; verified only by later checks."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "fact": {"type": "string"},
                            "reference": {"type": ["string", "null"]},
                        },
                        "required": ["fact", "reference"],
                        "additionalProperties": False,
                    },
                },
                "uncertainties": {
                    "type": "array",
                    "description": "Open questions and missing signals.",
                    "items": {"type": "string"},
                },
                "candidate_checks": {
                    "type": "array",
                    "description": (
                        "Next-step options, each with the exact shell "
                        "command to run if selected."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "command": {
                                "type": "string",
                                "description": (
                                    "One allowlisted shell command; same "
                                    "rules as direct shell access."
                                ),
                            },
                            "expected_evidence": {
                                "type": ["string", "null"],
                            },
                        },
                        "required": [
                            "id",
                            "description",
                            "command",
                            "expected_evidence",
                        ],
                        "additionalProperties": False,
                    },
                },
                "luna_selection": {
                    "type": ["string", "null"],
                    "description": (
                        "After a fallback, Luna's own pick from the same "
                        "candidate set; runs without calling JEV."
                    ),
                },
            },
            "required": [
                "hypotheses",
                "verified_evidence",
                "uncertainties",
                "candidate_checks",
                "luna_selection",
            ],
            "additionalProperties": False,
        },
        on_invoke_tool=_invoke,
        needs_approval=False,
    )


def _option_ids(checks: Any) -> list[str]:
    return [
        c.get("id")
        for c in (checks if isinstance(checks, list) else [])
        if isinstance(c, dict) and c.get("id")
    ]


def _command_for(checks: Any, check_id: str) -> Optional[str]:
    for check in checks if isinstance(checks, list) else []:
        if isinstance(check, dict) and check.get("id") == check_id:
            command = check.get("command")
            if isinstance(command, str) and command.strip():
                return command
    return None


async def _select_and_run(
    ctx: Any,
    args_json: str,
    *,
    executor: Any,
    api_key: Optional[str],
    model: str,
    http_transport: Any,
    max_consults: Optional[int],
    calls_made: dict[str, int],
    pending_state: dict[str, Any],
    audit_sink: Callable[[dict[str, Any]], None],
) -> str:
    call_id = getattr(ctx, "tool_call_id", None) or "unavailable"
    checks: list[dict[str, Any]] = []

    def record(event: dict[str, Any]) -> None:
        audit_sink(event)

    def pending_mapping(valid_checks: list[dict[str, Any]]) -> dict[str, str]:
        return {
            c["id"]: c["command"]
            for c in valid_checks
            if isinstance(c, dict)
            and isinstance(c.get("id"), str)
            and isinstance(c.get("command"), str)
            and c["command"].strip()
        }

    def audited_fallback(reason: str) -> str:
        record(
            {
                "selection_source": "luna",
                "selected_action": None,
                "fallback_reason": reason,
                "tool_call_id": call_id,
            }
        )
        return json.dumps(
            {
                "selected_action": None,
                "selected_by": "luna",
                "executed": False,
                "fallback_reason": reason,
                "candidates": _option_ids(checks),
            }
        )

    async def dispatch_luna_pick(
        pending: Optional[dict[str, str]], submitted: Any, luna_pick: Any
    ) -> str:
        if pending is None:
            return audited_fallback(
                "no pending fallback decision; "
                "luna_selection requires a recorded unusable JEV outcome"
            )
        if not isinstance(luna_pick, str) or luna_pick not in pending:
            return audited_fallback(f"invalid Luna selection {luna_pick!r}")
        mapping = {
            c["id"]: c.get("command")
            for c in (submitted if isinstance(submitted, list) else [])
            if isinstance(c, dict)
        }
        if mapping.get(luna_pick) != pending[luna_pick]:
            return audited_fallback(
                f"Luna selection {luna_pick!r} changed command text"
            )
        pending_state["pending"] = None
        return await _dispatch(
            executor,
            record,
            call_id,
            selection_source="luna",
            check_id=luna_pick,
            command=pending[luna_pick],
            probabilities={},
            confidence=None,
            fallback_reason=None,
        )
    try:
        args = json.loads(args_json)
    except ValueError:
        pending_state["pending"] = None
        return audited_fallback("arguments must be a JSON object")
    if not isinstance(args, dict):
        pending_state["pending"] = None
        return audited_fallback("arguments must be a JSON object")
    checks = args.get("candidate_checks")
    if not isinstance(checks, list) or not checks:
        pending_state["pending"] = None
        return audited_fallback("candidate_checks must be a non-empty list")
    option_ids = _option_ids(checks)
    if len(set(option_ids)) != len(checks):
        pending_state["pending"] = None
        return audited_fallback("candidate check ids must be unique strings")

    luna_pick = args.get("luna_selection")
    if luna_pick is not None:
        return await dispatch_luna_pick(
            pending_state["pending"], checks, luna_pick
        )

    if not api_key:
        pending_state["pending"] = pending_mapping(checks)
        return audited_fallback("no JEV API key configured")
    # max_consults=None means unbounded; any other value caps attempts.
    if max_consults is not None and calls_made["n"] >= max_consults:
        pending_state["pending"] = pending_mapping(checks)
        return audited_fallback("Decisions budget exhausted for this run")
    # Reserve before any awaited I/O so parallel invocations cannot
    # jointly exceed the cap (single-threaded check-and-increment).
    calls_made["n"] += 1
    state = build_decision_state(
        args.get("hypotheses"),
        args.get("verified_evidence"),
        args.get("uncertainties"),
        checks,
    )
    payload = {
        "model": model,
        "state": state,
        "questions": {
            "next_step": {
                "type": "choice",
                "instructions": (
                    "Which single investigation step should the analyst "
                    "take next? Prefer the step whose expected evidence "
                    "most reduces uncertainty per turn spent."
                ),
                "criteria": {
                    c["id"]: c.get("description", c["id"])
                    for c in checks
                    if isinstance(c, dict)
                },
            }
        },
    }
    outcome, detail = await _post_decision_vote(
        api_key=api_key,
        model=model,
        payload=payload,
        http_transport=http_transport,
    )
    if outcome != "ok":
        pending_state["pending"] = pending_mapping(checks)
        return audited_fallback(detail["reason"])
    body = detail["body"]
    try:
        answer = body["answers"]["next_step"]
    except (KeyError, AttributeError):
        pending_state["pending"] = pending_mapping(checks)
        return audited_fallback("unparseable Decisions response")
    response_meta = {
        "generation_id": body.get("id")
        if isinstance(body.get("id"), str)
        else UNAVAILABLE,
        "served_model": body.get("model")
        if isinstance(body.get("model"), str)
        else UNAVAILABLE,
        "served_provider": body.get("provider")
        if isinstance(body.get("provider"), str)
        else UNAVAILABLE,
        "latency_ms": detail["latency_ms"],
        "returned_cost": _response_cost(body),
    }
    outcome, detail = _validate_vote_answer(answer, option_ids)
    if outcome != "selected":
        pending_state["pending"] = pending_mapping(checks)
        return audited_fallback(detail["reason"])
    pending_state["pending"] = None
    return await _dispatch(
        executor,
        record,
        call_id,
        selection_source="jev",
        check_id=detail["choice"],
        command=_command_for(checks, detail["choice"]),
        probabilities=detail["probabilities"],
        confidence=detail["confidence"],
        top=detail["top"],
        margin=detail["margin"],
        fallback_reason=None,
        response_meta=response_meta,
    )


def _validate_vote_answer(
    answer: Any, option_ids: list[str]
) -> tuple[str, dict[str, Any]]:
    """Validate a Decisions answer without applying ambiguity thresholds.

    Sole-selector policy executes every valid selection and records the
    margin as uncertainty instead of blocking on closeness. Returns
    ("selected", detail) or ("fallback", {"reason": ...}).
    """
    if not isinstance(answer, dict):
        return "fallback", {"reason": "unparseable Decisions response"}
    choice = answer.get("choice")
    probabilities = answer.get("probabilities") or {}
    if not isinstance(probabilities, dict):
        probabilities = {}
    if choice not in option_ids:
        return "fallback", {"reason": f"invalid selection {choice!r}"}
    try:
        ranked = sorted(
            [float(probabilities.get(o, 0.0)) for o in option_ids],
            reverse=True,
        )
    except (TypeError, ValueError):
        return "fallback", {"reason": "non-numeric option probabilities"}
    top = ranked[0] if ranked else 0.0
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else top
    return (
        "selected",
        {
            "choice": choice,
            "probabilities": {
                o: probabilities.get(o, 0.0) for o in option_ids
            },
            "confidence": answer.get("confidence"),
            "top": top,
            "margin": margin,
        },
    )


def _response_cost(body: Any) -> Any:
    usage = body.get("usage") if isinstance(body, dict) and isinstance(
        body.get("usage"), dict
    ) else {}
    cost = usage.get("cost")
    return cost if isinstance(cost, (int, float)) else UNAVAILABLE


async def _dispatch(
    executor: Any,
    record: Callable[[dict[str, Any]], None],
    call_id: str,
    *,
    selection_source: str,
    check_id: str,
    command: Optional[str],
    probabilities: Optional[dict[str, Any]] = None,
    confidence: Any = None,
    top: Optional[float] = None,
    margin: Optional[float] = None,
    fallback_reason: Optional[str] = None,
    response_meta: Optional[dict[str, Any]] = None,
) -> str:
    """Run exactly the selected command; rejection is an ordinary result."""

    if not command:
        record(
            {
                "selection_source": selection_source,
                "selected_action": check_id,
                "fallback_reason": "selected check has no runnable command",
                "tool_call_id": call_id,
            }
        )
        return json.dumps(
            {
                "selected_action": check_id,
                "selected_by": selection_source,
                "executed": False,
                "fallback_reason": "selected check has no runnable command",
            }
        )
    request = ShellCommandRequest(
        ctx_wrapper=RunContextWrapper(context=None),
        data=ShellCallData(
            call_id=call_id,
            action=ShellActionRequest(commands=[command], timeout_ms=None),
        ),
    )
    result = await executor(request)
    output = result.output[0] if result.output else None
    stdout = getattr(output, "stdout", "")
    stderr = getattr(output, "stderr", "")
    exit_code = getattr(getattr(output, "outcome", None), "exit_code", None)
    event: dict[str, Any] = {
        "selection_source": selection_source,
        "selected_action": check_id,
        "dispatched_command": command,
        "exit_code": exit_code,
        "tool_call_id": call_id,
    }
    if probabilities is not None:
        event["option_probabilities"] = probabilities
    if confidence is not None:
        event["returned_confidence"] = confidence
    if top is not None:
        event["top"] = top
    if margin is not None:
        event["margin"] = margin
    if fallback_reason is not None:
        event["fallback_reason"] = fallback_reason
    if response_meta is not None:
        event.update(response_meta)
    record(event)
    return json.dumps(
        {
            "selected_action": check_id,
            "selected_by": selection_source,
            "executed": True,
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "top": top,
            "margin": margin,
        }
    )
