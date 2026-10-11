"""JEV consultation tool: Luna proposes, JEV disposes (advisory only)."""

from __future__ import annotations


def test_builder_exposes_structured_consultation_tool():
    from flight_log_agent.analysis import jev_consult as jc

    tool = jc.build_jev_consult_tool(api_key="k", audit_sink=lambda _e: None)
    assert tool.name == "consult_investigation_direction"
    schema = tool.params_json_schema
    assert schema["type"] == "object"
    assert set(schema["required"]) == {
        "hypotheses",
        "verified_evidence",
        "uncertainties",
        "candidate_checks",
    }
    assert schema["additionalProperties"] is False


def test_consult_tool_serializes_on_chat_transport():
    from agents.models.chatcmpl_converter import Converter

    from flight_log_agent.analysis import jev_consult as jc

    tool = jc.build_jev_consult_tool(api_key="k", audit_sink=lambda _e: None)
    entry = Converter.tool_to_openai(tool)
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "consult_investigation_direction"


def _decisions_payload(choice="compute_derivation", probs=None, conf=0.9):
    return {
        "id": "gen-dec-1",
        "model": "typesafe/jev-1.13-20260917",
        "provider": "TypeSafe",
        "usage": {"cost": 0.00002, "input_tokens": 100, "output_tokens": 10},
        "answers": {
            "next_step": {
                "type": "choice",
                "choice": choice,
                "confidence": conf,
                "probabilities": probs
                if probs is not None
                else {"compute_derivation": 0.9, "verify_coords": 0.1},
            }
        },
    }


def _invoke_tool(tool, args, transport, call_id="call-9"):
    import asyncio
    import json

    import httpx
    from agents.tool_context import ToolContext

    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id=call_id,
        tool_arguments=json.dumps(args),
    )
    return asyncio.run(tool.on_invoke_tool(ctx, json.dumps(args)))


def _tool_with_transport(transport, sink):
    from flight_log_agent.analysis import jev_consult as jc

    return jc.build_jev_consult_tool(
        api_key="k", audit_sink=sink.append, http_transport=transport
    )


def _args():
    return {
        "hypotheses": [{"title": "cone floor"}],
        "verified_evidence": [{"fact": "params", "artifact": "tool-3"}],
        "uncertainties": ["distance geometry"],
        "candidate_checks": [
            {"id": "compute_derivation", "description": "Compute the max() derivation"},
            {"id": "verify_coords", "description": "Extract landing coordinates"},
        ],
    }


def test_successful_consult_returns_selection_and_audits_it():
    import json

    import httpx

    seen = []

    def handler(request):
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_decisions_payload(), request=request)

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(handler), sink)
    out = json.loads(_invoke_tool(tool, _args(), None))

    assert out["selected_action"] == "compute_derivation"
    assert out["option_probabilities"]["compute_derivation"] == 0.9
    assert out["returned_confidence"] == 0.9
    assert "advisory" in out["effective_recommendation"].lower()
    body = seen[0]
    assert body["model"] == "typesafe/jev-1.13"
    assert set(body["questions"]) == {"next_step"}
    assert set(body["questions"]["next_step"]["criteria"]) == {
        "compute_derivation",
        "verify_coords",
    }
    assert body["state"]["hypotheses"][0]["origin"] == "model"
    assert body["state"]["verified_evidence"][0]["origin"] == "model-reported"
    assert len(sink) == 1
    assert sink[0]["selected_action"] == "compute_derivation"
    assert sink[0]["tool_call_id"] == "call-9"
    assert "requested_" not in json.dumps(sink[0])


def test_fallback_when_no_key_configured():
    import json

    from flight_log_agent.analysis import jev_consult as jc

    sink = []
    tool = jc.build_jev_consult_tool(api_key=None, audit_sink=sink.append)
    out = json.loads(_invoke_tool(tool, _args(), None))
    assert out["selected_action"] == "deferred_to_luna"
    assert "no JEV API key" in out["fallback_reason"]
    assert sink[0]["effective_recommendation"] == "deferred_to_luna"


def test_fallback_on_http_error_and_invalid_selection():
    import json

    import httpx

    def boom(request):
        return httpx.Response(500, json={"error": "x"}, request=request)

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(boom), sink)
    out = json.loads(_invoke_tool(tool, _args(), None))
    assert out["selected_action"] == "deferred_to_luna"
    assert "HTTP 500" in out["fallback_reason"]

    def wrong_choice(request):
        return httpx.Response(
            200, json=_decisions_payload(choice="nope"), request=request
        )

    sink2 = []
    tool2 = _tool_with_transport(httpx.MockTransport(wrong_choice), sink2)
    out2 = json.loads(_invoke_tool(tool2, _args(), None))
    assert out2["selected_action"] == "deferred_to_luna"
    assert "invalid selection" in out2["fallback_reason"]


def test_ambiguous_result_falls_back_with_thresholds_labeled():
    import json

    import httpx

    from flight_log_agent.analysis import jev_consult as jc

    assert (jc.MIN_TOP_PROBABILITY, jc.MIN_PROBABILITY_MARGIN) == (0.55, 0.15)

    def close(request):
        return httpx.Response(
            200,
            json=_decisions_payload(
                choice="compute_derivation",
                probs={"compute_derivation": 0.43, "verify_coords": 0.57},
                conf=0.43,
            ),
            request=request,
        )

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(close), sink)
    out = json.loads(_invoke_tool(tool, _args(), None))
    assert out["selected_action"] == "deferred_to_luna"
    assert "ambiguous" in out["fallback_reason"]


def test_decisions_budget_capped_separately_from_tool_execution():
    import json

    import httpx

    hits = []

    def ok(request):
        hits.append(1)
        return httpx.Response(200, json=_decisions_payload(), request=request)

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(ok), sink)
    for _ in range(4):
        out = json.loads(_invoke_tool(tool, _args(), None))
    assert len(hits) == 3
    assert out["selected_action"] == "deferred_to_luna"
    assert "budget exhausted" in out["fallback_reason"]
    assert len(sink) == 4


def _decisions_payload_with_id(response_id="gen-dec-1", model="typesafe/jev-1.13-20260917"):
    payload = _decisions_payload()
    payload["id"] = response_id
    payload["model"] = model
    payload["provider"] = "TypeSafe"
    return payload


def test_success_event_carries_decisions_response_metadata():
    import json

    import httpx

    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200, json=_decisions_payload_with_id(), request=request
        )

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(handler), sink)
    out = json.loads(_invoke_tool(tool, _args(), None))
    assert out["selected_action"] == "compute_derivation"
    event = sink[0]
    assert event["generation_id"] == "gen-dec-1"
    assert event["served_model"] == "typesafe/jev-1.13-20260917"
    assert event["served_provider"] == "TypeSafe"
    assert isinstance(event["latency_ms"], (int, float))
    assert event["returned_cost"] == 0.00002
    assert event["selected_action"] == "compute_derivation"
    assert event["effective_recommendation"] == "compute_derivation"


def test_parallel_invocations_cannot_exceed_budget():
    import asyncio
    import json

    import httpx

    hits = []

    async def slow_handler(request):
        hits.append(1)
        await asyncio.sleep(0.05)
        return httpx.Response(
            200, json=_decisions_payload_with_id(), request=request
        )

    class SlowTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return await slow_handler(request)

    from flight_log_agent.analysis import jev_consult as jc

    sink = []
    tool = jc.build_jev_consult_tool(
        api_key="k",
        audit_sink=sink.append,
        http_transport=SlowTransport(),
        max_consults=1,
    )
    async def _both():
        return await asyncio.gather(
            *[
                _invoke_tool_async(tool, _args(), f"call-{i}")
                for i in range(2)
            ]
        )

    results = asyncio.run(_both())
    selected = sorted(json.loads(r)["selected_action"] for r in results)
    assert selected == ["compute_derivation", "deferred_to_luna"]
    assert len(hits) == 1


def _invoke_tool_async(tool, args, call_id):
    import json

    from agents.tool_context import ToolContext

    ctx = ToolContext(
        context=None,
        tool_name=tool.name,
        tool_call_id=call_id,
        tool_arguments=json.dumps(args),
    )
    return tool.on_invoke_tool(ctx, json.dumps(args))


def test_non_numeric_probabilities_fall_back():
    import json

    import httpx

    def bad_probs(request):
        payload = _decisions_payload()
        payload["answers"]["next_step"]["probabilities"] = {
            "compute_derivation": "high",
            "verify_coords": "low",
        }
        return httpx.Response(200, json=payload, request=request)

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(bad_probs), sink)
    out = json.loads(_invoke_tool(tool, _args(), None))
    assert out["selected_action"] == "deferred_to_luna"
    assert "non-numeric" in out["fallback_reason"]


def _strict_complete(node, path="root"):
    """Every object declares non-empty properties, additionalProperties
    false, and required containing exactly those properties."""
    if isinstance(node, dict):
        if node.get("type") == "object":
            properties = node.get("properties")
            assert isinstance(properties, dict) and properties, (
                f"{path}: object items must declare non-empty properties"
            )
            assert node.get("additionalProperties") is False, path
            assert set(node.get("required", [])) == set(properties), (
                f"{path}: required={node.get('required')}"
            )
        for key, value in node.items():
            _strict_complete(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _strict_complete(value, f"{path}[{i}]")


def test_consult_schema_is_recursively_strict_complete():
    from agents.models.chatcmpl_converter import Converter

    from flight_log_agent.analysis import jev_consult as jc

    tool = jc.build_jev_consult_tool(api_key="k", audit_sink=lambda _e: None)
    entry = Converter.tool_to_openai(tool)
    params = entry["function"]["parameters"]
    assert entry["function"].get("strict") is True
    _strict_complete(params)
    item_props = params["properties"]["candidate_checks"]["items"]["properties"]
    assert set(item_props["expected_evidence"]["type"]) == {"string", "null"}


def test_null_optional_fields_survive_invocation():
    import json

    import httpx

    seen = []

    def handler(request):
        seen.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, json=_decisions_payload(), request=request)

    sink = []
    tool = _tool_with_transport(httpx.MockTransport(handler), sink)
    args = _args()
    args["candidate_checks"] = [
        {"id": "compute_derivation", "description": "Compute it", "expected_evidence": None}
    ]
    args["hypotheses"] = [{"claim": "cone floor", "context": None}]
    args["verified_evidence"] = [{"fact": "params read", "reference": None}]
    out = json.loads(_invoke_tool(tool, args, None))
    assert out["selected_action"] == "compute_derivation"
    sent_checks = seen[0]
    assert sent_checks, "expected the request to be transmitted"


def test_mode_resolution_preserves_legacy_configuration():
    from flight_log_agent.analysis import jev_consult as jc

    assert jc.resolve_jev_mode(None, None) == "off"
    assert jc.resolve_jev_mode(False, None) == "off"
    assert jc.resolve_jev_mode(True, None) == "voluntary"
    assert jc.resolve_jev_mode(None, "1") == "voluntary"
    assert jc.resolve_jev_mode(None, "yes") == "voluntary"


def test_unknown_mode_is_rejected():
    import pytest

    from flight_log_agent.analysis import jev_consult as jc

    with pytest.raises(ValueError, match="jev_consult"):
        jc.resolve_jev_mode("always", None)


def test_sole_mode_value_resolves():
    from flight_log_agent.analysis import jev_consult as jc

    assert jc.resolve_jev_mode("sole", None) == "sole"
    assert jc.resolve_jev_mode(None, None) == "off"
