"""Sole-selector tool: JEV picks, the tool itself runs the winner."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from agents.tool_context import ToolContext

from flight_log_agent.analysis import sole_selector as ss
from flight_log_agent.analysis.analyzer import RestrictedShellExecutor


def _ctx(call_id="call-sole-1"):
    return ToolContext(
        context=None,
        tool_name=ss.SOLE_SELECTOR_TOOL_NAME,
        tool_call_id=call_id,
        tool_arguments="{}",
    )


def _args(**overrides):
    base = {
        "hypotheses": [{"claim": "cone floor", "context": None}],
        "verified_evidence": [{"fact": "params", "reference": None}],
        "uncertainties": ["geometry?"],
        "candidate_checks": [
            {"id": "check_a", "description": "Run A",
             "command": "python -c \"print('MARKER_A')\"", "expected_evidence": None},
            {"id": "check_b", "description": "Run B",
             "command": "python -c \"print('MARKER_B')\"", "expected_evidence": None},
        ],
        "luna_selection": None,
    }
    base.update(overrides)
    return base


def _executor(tmp_path, markers):
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    executor = RestrictedShellExecutor(work, None, input_paths={})

    async def fake_sandboxed_process(argv, **_kwargs):
        text = argv[2] if len(argv) > 2 else ""
        for marker, payload in markers.items():
            if marker in text:
                (work / f"{marker.lower()}_ran.txt").write_text(
                    marker, encoding="utf-8"
                )
                return payload.encode(), b"", 0, False
        raise AssertionError(f"unexpected command: {argv}")

    executor._run_sandboxed_process = fake_sandboxed_process
    return executor, work


def _decisions_transport(choice="check_b", probs=None, status=200):
    def handler(request):
        return httpx.Response(
            status,
            json={
                "id": "gen-dec-sole",
                "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "usage": {"cost": 0.00002},
                "answers": {
                    "next_step": {
                        "type": "choice",
                        "choice": choice,
                        "confidence": 0.9,
                        "probabilities": probs
                        if probs is not None
                        else {"check_a": 0.1, "check_b": 0.9},
                    }
                },
            },
            request=request,
        )

    return httpx.MockTransport(handler)


def test_close_but_valid_choice_executes_and_records_margin(tmp_path):
    executor, _ = _executor(tmp_path, {"MARKER_B": "marker-b-ok\n"})
    transport = _decisions_transport(
        probs={"check_a": 0.48, "check_b": 0.52}
    )
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=lambda _e: None,
        http_transport=transport,
    )
    out = json.loads(asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args()))))
    assert out["selected_action"] == "check_b"
    assert out["selected_by"] == "jev"
    assert out["executed"] is True
    assert out["exit_code"] == 0
    assert "marker-b-ok" in out["stdout"]


def test_only_the_winner_executes(tmp_path):
    executor, work = _executor(
        tmp_path, {"MARKER_A": "a\n", "MARKER_B": "b\n"}
    )
    transport = _decisions_transport(choice="check_b")
    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=transport,
    )
    asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args())))
    assert (work / "marker_b_ran.txt").read_text() == "MARKER_B"
    assert not (work / "marker_a_ran.txt").exists()
    assert events[0]["dispatched_command"] == (
        "python -c \"print('MARKER_B')\""
    )


def test_fallback_returns_candidates_then_luna_selection_executes(tmp_path):
    executor, work = _executor(tmp_path, {"MARKER_A": "a-ok\n"})
    hits = []

    def fail(request):
        hits.append(1)
        return httpx.Response(500, json={"error": "x"}, request=request)

    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=httpx.MockTransport(fail),
    )
    first = json.loads(asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args()))))
    assert first["selected_action"] is None
    assert first["executed"] is False
    assert first["fallback_reason"] is not None
    assert not (work / "marker_a_ran.txt").exists()

    second = json.loads(
        asyncio.run(
            tool.on_invoke_tool(
                _ctx("call-sole-2"),
                json.dumps(_args(luna_selection="check_a")),
            )
        )
    )
    assert second["selected_action"] == "check_a"
    assert second["selected_by"] == "luna"
    assert second["executed"] is True
    assert (work / "marker_a_ran.txt").exists()
    assert len(hits) == 1
    assert events[-1]["selected_action"] == "check_a"
    assert events[-1]["selection_source"] == "luna"


def test_executor_rejection_is_an_ordinary_result(tmp_path):
    executor, work = _executor(tmp_path, {})
    transport = _decisions_transport(choice="bad", probs={"bad": 1.0})
    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=transport,
    )
    args = _args()
    args["candidate_checks"] = [
        {"id": "bad", "description": "Disallowed program",
         "command": "ls /", "expected_evidence": None},
    ]
    out = json.loads(asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(args))))
    assert out["selected_action"] == "bad"
    assert out["selected_by"] == "jev"
    assert out["executed"] is True
    assert out["exit_code"] == 126
    assert "not allowed" in (out["stdout"] + out["stderr"]).lower()
    assert events[0]["exit_code"] == 126


def test_concurrent_votes_preserve_consultation_limit(tmp_path):
    executor, _ = _executor(tmp_path, {"MARKER_B": "b\n"})

    import asyncio as _asyncio

    hits = []

    async def slow_handler(request):
        hits.append(1)
        await _asyncio.sleep(0.05)
        return httpx.Response(
            200, json={
                "id": "gen-x", "model": "m", "provider": "p",
                "usage": {"cost": 0.1},
                "answers": {"next_step": {"type": "choice", "choice": "check_b",
                                          "confidence": 0.9,
                                          "probabilities": {"check_a": 0.1, "check_b": 0.9}}},
            }, request=request)

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await slow_handler(request)

    from flight_log_agent.analysis import sole_selector as ssm

    events = []
    tool = ssm.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=SlowTransport(), max_consults=1,
    )

    async def _main():
        return await _asyncio.gather(
            *[_invoke_one(tool, _args(), f"call-{i}") for i in range(2)]
        )

    async def _invoke_one(tool, args, call_id):
        from agents.tool_context import ToolContext

        ctx = ToolContext(context=None, tool_name=tool.name,
                          tool_call_id=call_id, tool_arguments=json.dumps(args))
        return await tool.on_invoke_tool(ctx, json.dumps(args))

    results = _asyncio.run(_main())
    executed = sum(1 for r in results if json.loads(r)["executed"])
    assert executed == 1
    assert len(hits) == 1


def test_sole_tool_serializes_on_chat_transport(tmp_path):
    from agents.models.chatcmpl_converter import Converter

    executor, _ = _executor(tmp_path, {})
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=lambda _e: None
    )
    entry = Converter.tool_to_openai(tool)
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "select_and_run_check"
    assert entry["function"].get("strict") is True
    params = entry["function"]["parameters"]

    def check_object(node, path):
        assert node.get("additionalProperties") is False, path
        properties = node.get("properties")
        assert isinstance(properties, dict) and properties, path
        assert set(node.get("required", [])) == set(properties), path
        for key, value in properties.items():
            check_value(value, f"{path}.{key}")

    def check_value(node, path):
        if isinstance(node, dict):
            if node.get("type") == "object":
                check_object(node, path)
            for key, value in node.items():
                check_value(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                check_value(value, f"{path}[{i}]")

    check_value(params, "params")


def test_luna_selection_without_prior_fallback_is_rejected(tmp_path):
    executor, work = _executor(tmp_path, {"MARKER_A": "a\n"})

    def fail(request):  # pragma: no cover - must not be called
        raise AssertionError("no JEV call expected")

    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=httpx.MockTransport(fail),
    )
    out = json.loads(
        asyncio.run(
            tool.on_invoke_tool(
                _ctx(), json.dumps(_args(luna_selection="check_a"))
            )
        )
    )
    assert out["executed"] is False
    assert "fallback" in out["fallback_reason"].lower()
    assert not (work / "marker_a_ran.txt").exists()


def test_luna_selection_with_changed_command_is_rejected(tmp_path):
    executor, work = _executor(tmp_path, {"MARKER_A": "a\n"})

    def fail(request):
        return httpx.Response(500, json={"error": "x"}, request=request)

    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=httpx.MockTransport(fail),
    )
    asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args())))
    swapped = _args(luna_selection="check_a")
    swapped["candidate_checks"][0]["command"] = "python -c \"print('OTHER')\""
    out = json.loads(
        asyncio.run(tool.on_invoke_tool(_ctx("call-2"), json.dumps(swapped)))
    )
    assert out["executed"] is False
    assert "command" in out["fallback_reason"].lower()


def test_luna_selection_after_executed_jev_choice_is_rejected(tmp_path):
    executor, work = _executor(
        tmp_path, {"MARKER_A": "a\n", "MARKER_B": "b\n"}
    )
    transport = _decisions_transport(choice="check_b")
    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=transport,
    )
    first = json.loads(asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args()))))
    assert first["executed"] is True
    second = json.loads(
        asyncio.run(
            tool.on_invoke_tool(
                _ctx("call-2"), json.dumps(_args(luna_selection="check_a"))
            )
        )
    )
    assert second["executed"] is False
    assert "no pending fallback" in second["fallback_reason"].lower()
    assert not (work / "marker_a_ran.txt").exists()


def test_close_valid_records_margin_and_top(tmp_path):
    executor, _ = _executor(tmp_path, {"MARKER_B": "b\n"})
    transport = _decisions_transport(
        probs={"check_a": 0.48, "check_b": 0.52}
    )
    events = []
    tool = ss.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=events.append,
        http_transport=transport,
    )
    out = json.loads(asyncio.run(tool.on_invoke_tool(_ctx(), json.dumps(_args()))))
    assert out["executed"] is True
    assert events[0]["margin"] == pytest.approx(0.04)
    assert events[0]["top"] == pytest.approx(0.52)


def test_sole_mode_allows_consultations_beyond_three(tmp_path):
    executor, _ = _executor(tmp_path, {"MARKER_B": "b\n"})
    hits = []

    def ok(request):
        hits.append(1)
        return httpx.Response(
            200,
            json={
                "id": "gen-sole",
                "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "usage": {"cost": 0.00002},
                "answers": {
                    "next_step": {
                        "type": "choice",
                        "choice": "check_b",
                        "confidence": 0.9,
                        "probabilities": {"check_a": 0.1, "check_b": 0.9},
                    }
                },
            },
            request=request,
        )

    from flight_log_agent.analysis import sole_selector as ssm

    tool = ssm.build_sole_selector_tool(
        executor=executor, api_key="k", audit_sink=lambda _e: None,
        http_transport=httpx.MockTransport(ok), max_consults=None,
    )
    for i in range(4):
        out = json.loads(
            asyncio.run(tool.on_invoke_tool(_ctx(f"call-{i}"), json.dumps(_args())))
        )
        assert out["selected_action"] == "check_b", f"call {i} fell back"
    assert len(hits) == 4


def test_advisory_modes_keep_three_consultation_cap(tmp_path):
    from flight_log_agent.analysis import jev_consult as jc

    assert jc.MAX_JEV_CONSULTS == 3
