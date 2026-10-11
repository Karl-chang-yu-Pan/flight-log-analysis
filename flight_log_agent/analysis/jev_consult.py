"""Optional JEV consultation tool: Luna proposes, JEV advises.

The single Runner.run loop and FlightLogReport contract are unchanged;
this tool only adds an advisory direction vote Luna may consult mid-run.
It never promotes confidence and never bypasses validation: the returned
selection is advisory text, and the report path is untouched.

Model-supplied claims and independently observed evidence travel in
separate parameters so the decision state never mixes them.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from agents import FunctionTool

JEV_CONSULT_TOOL_NAME = "consult_investigation_direction"
JEV_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_DECISIONS_MODEL = "typesafe/jev-1.13"
UNAVAILABLE = "unavailable"

# Experimental thresholds for treating a JEV answer as decisive rather
# than ambiguous. Tuned against three evaluation snapshots, not production
# traffic; revisit with measured ambiguity/fallback rates.
MIN_TOP_PROBABILITY = 0.55
MIN_PROBABILITY_MARGIN = 0.15

# separately-accounted Decisions budget per run; ordinary tool execution
# never spends it.
MAX_JEV_CONSULTS = 3

_REQUEST_TIMEOUT_S = 30.0


def _check_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def build_decision_state(
    hypotheses: Any,
    verified_evidence: Any,
    uncertainties: Any,
    candidate_checks: Any,
) -> dict[str, Any]:
    """Split Luna's input into model claims vs observed evidence."""
    hypotheses = hypotheses if isinstance(hypotheses, list) else []
    verified = verified_evidence if isinstance(verified_evidence, list) else []
    # Origin discipline: everything Luna supplies is model-sourced. Evidence
    # summaries are labeled model-reported even when Luna presents them as
    # observed; only pipeline checks (validation, tool results) verify.
    return {
        "hypotheses": [
            {**_check_object(h), "origin": "model"} for h in hypotheses
        ],
        "verified_evidence": [
            {**_check_object(e), "origin": "model-reported"} for e in verified
        ],
        "uncertainties": uncertainties if isinstance(uncertainties, list) else [],
        "candidate_checks": (
            candidate_checks if isinstance(candidate_checks, list) else []
        ),
    }


def build_jev_consult_tool(
    *,
    api_key: Optional[str],
    audit_sink: Callable[[dict[str, Any]], None],
    model: str = JEV_DECISIONS_MODEL,
    http_transport: Any = None,
    max_consults: int = MAX_JEV_CONSULTS,
) -> FunctionTool:
    """Build the advisory direction-consultation function tool."""
    calls_made = {"n": 0}

    async def _invoke(ctx: Any, args_json: str) -> str:
        return await _consult(
            ctx,
            args_json,
            api_key=api_key,
            model=model,
            http_transport=http_transport,
            max_consults=max_consults,
            calls_made=calls_made,
            audit_sink=audit_sink,
        )

    return FunctionTool(
        name=JEV_CONSULT_TOOL_NAME,
        description=(
            "Ask the JEV decision model which investigation step to take "
            "next. Pass structured hypotheses (your claims), "
            "model-reported evidence (facts you observed, with artifact "
            "or tool references where known), "
            "uncertainties, and 2-5 candidate next checks with expected "
            "evidence. Returns the selected direction with confidence; the "
            "selection is advisory and never replaces evidence or "
            "validation."
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
                            "claim": {
                                "type": "string",
                                "description": "The candidate explanation.",
                            },
                            "context": {
                                "type": ["string", "null"],
                                "description": (
                                    "Supporting context, if known."
                                ),
                            },
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
                            "fact": {
                                "type": "string",
                                "description": "The reported fact.",
                            },
                            "reference": {
                                "type": ["string", "null"],
                                "description": (
                                    "Artifact or tool reference, if known."
                                ),
                            },
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
                    "description": "Next-step options to choose among.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "description": {"type": "string"},
                            "expected_evidence": {
                                "type": ["string", "null"],
                                "description": (
                                    "What evidence this check should "
                                    "produce, if known."
                                ),
                            },
                        },
                        "required": ["id", "description", "expected_evidence"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": [
                "hypotheses",
                "verified_evidence",
                "uncertainties",
                "candidate_checks",
            ],
            "additionalProperties": False,
        },
        on_invoke_tool=_invoke,
        needs_approval=False,
    )


async def _consult(
    ctx: Any,
    args_json: str,
    *,
    api_key: Optional[str],
    model: str,
    http_transport: Any,
    max_consults: int,
    calls_made: dict[str, int],
    audit_sink: Callable[[dict[str, Any]], None],
) -> str:
    import httpx

    def fallback(reason: str) -> str:
        audit_sink(
            {
                "selected_action": "deferred_to_luna",
                "effective_recommendation": "deferred_to_luna",
                "fallback_reason": reason,
                "tool_call_id": getattr(ctx, "tool_call_id", None)
                or "unavailable",
            }
        )
        return json.dumps(
            {
                "selected_action": "deferred_to_luna",
                "effective_recommendation": (
                    "JEV unavailable or undecided; use your own ranking "
                    f"({reason})."
                ),
                "fallback_reason": reason,
            }
        )

    try:
        args = json.loads(args_json)
    except ValueError:
        return fallback("arguments must be a JSON object")
    if not isinstance(args, dict):
        return fallback("arguments must be a JSON object")
    checks = args.get("candidate_checks")
    if not isinstance(checks, list) or not checks:
        return fallback("candidate_checks must be a non-empty list")
    option_ids = [
        c.get("id") for c in checks if isinstance(c, dict) and c.get("id")
    ]
    if len(set(option_ids)) != len(checks):
        return fallback("candidate check ids must be unique strings")
    if not api_key:
        return fallback("no JEV API key configured")
    if calls_made["n"] >= max_consults:
        return fallback("Decisions budget exhausted for this run")
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
    import time as _time

    started_at = _time.perf_counter()
    try:
        client_kwargs: dict[str, Any] = {"timeout": _REQUEST_TIMEOUT_S}
        if http_transport is not None:
            client_kwargs["transport"] = http_transport
        async with httpx.AsyncClient(**client_kwargs) as client:
            response = await client.post(
                JEV_DECISIONS_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
    except Exception as exc:
        return fallback(f"Decisions request failed: {type(exc).__name__}")
    latency_ms = round((_time.perf_counter() - started_at) * 1000, 1)
    if response.status_code != 200:
        return fallback(f"Decisions returned HTTP {response.status_code}")
    try:
        body = response.json()
        answer = body["answers"]["next_step"]
        choice = answer.get("choice")
        probabilities = answer.get("probabilities") or {}
        if not isinstance(probabilities, dict):
            probabilities = {}
        confidence = answer.get("confidence")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    except (ValueError, KeyError, AttributeError):
        return fallback("unparseable Decisions response")
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
        "latency_ms": latency_ms,
        "returned_usage": usage if isinstance(usage, dict) else UNAVAILABLE,
        "returned_cost": usage.get("cost")
        if isinstance(usage.get("cost"), (int, float))
        else UNAVAILABLE,
    }
    if choice not in option_ids:
        return fallback(f"invalid selection {choice!r}")
    try:
        ranked = sorted(
            [float(probabilities.get(o, 0.0)) for o in option_ids],
            reverse=True,
        )
    except (TypeError, ValueError):
        return fallback("non-numeric option probabilities")
    top = ranked[0] if ranked else 0.0
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else top
    if top < MIN_TOP_PROBABILITY or margin < MIN_PROBABILITY_MARGIN:
        return fallback(
            f"ambiguous result (top {top:.2f}, margin {margin:.2f})"
        )
    audit_sink(
        {
            **response_meta,
            "selected_action": choice,
            "option_probabilities": {
                o: probabilities.get(o, 0.0) for o in option_ids
            },
            "returned_confidence": confidence,
            "effective_recommendation": choice,
            "tool_call_id": getattr(ctx, "tool_call_id", None) or "unavailable",
        }
    )
    return json.dumps(
        {
            "selected_action": choice,
            "option_probabilities": {
                o: probabilities.get(o, 0.0) for o in option_ids
            },
            "returned_confidence": confidence,
            "effective_recommendation": (
                f"Advisory only: consider {choice} next, but only evidence "
                "and validation decide the report."
            ),
        }
    )
