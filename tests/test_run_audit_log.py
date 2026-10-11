import asyncio
import json
from types import SimpleNamespace

from flight_log_agent.audit import AgentRunAuditHooks, DeveloperAuditLogger, log_run_items


def read_events(run_dir):
    return [
        json.loads(line)
        for line in (run_dir / "run_events.jsonl").read_text().splitlines()
    ]


def test_agent_run_audit_hooks_log_tool_inputs_and_outputs(tmp_path):
    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    hooks = AgentRunAuditHooks(audit_logger)
    context = SimpleNamespace(
        tool_call_id="call_123",
        tool_name="compute_log_metrics",
        tool_arguments='{"start_s": 1.0, "end_s": 2.0, "signals": ["vehicle_status.nav_state"]}',
    )
    agent = SimpleNamespace(name="PX4 Flight Log Analyst V1")

    asyncio.run(hooks.on_tool_start(context, agent, SimpleNamespace(name="unused")))
    asyncio.run(
        hooks.on_tool_end(
            context,
            agent,
            SimpleNamespace(name="unused"),
            '{"metrics": {"vehicle_status.nav_state": {"count": 3}}}',
        )
    )

    events = read_events(audit_logger.run_dir)
    assert events[0]["event"] == "tool.started"
    assert events[0]["tool_name"] == "compute_log_metrics"
    assert events[0]["input"] == {
        "start_s": 1.0,
        "end_s": 2.0,
        "signals": ["vehicle_status.nav_state"],
    }
    assert events[1]["event"] == "tool.finished"
    assert events[1]["output"] == {
        "metrics": {"vehicle_status.nav_state": {"count": 3}}
    }
    assert events[1]["duration_ms"] >= 0


def test_developer_audit_logger_writes_usage_summary(tmp_path):
    usage = SimpleNamespace(
        requests=2,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=SimpleNamespace(cached_tokens=40),
        output_tokens_details=SimpleNamespace(reasoning_tokens=5),
        request_usage_entries=[
            SimpleNamespace(
                requests=1,
                input_tokens=60,
                output_tokens=10,
                total_tokens=70,
                input_tokens_details=SimpleNamespace(cached_tokens=20),
                output_tokens_details=SimpleNamespace(reasoning_tokens=2),
                request_usage_entries=[],
            )
        ],
    )
    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")

    audit_logger.save_usage(usage)

    assert json.loads(audit_logger.usage_path.read_text()) == {
        "cache_write_tokens": 0,
        "cached_input_tokens": 40,
        "input_tokens": 100,
        "output_tokens": 20,
        "reasoning_tokens": 5,
        "request_usage_entries": [
            {
                "cache_write_tokens": 0,
                "cached_input_tokens": 20,
                "input_tokens": 60,
                "output_tokens": 10,
                "reasoning_tokens": 2,
                "request_usage_entries": [],
                "requests": 1,
                "total_tokens": 70,
            }
        ],
        "requests": 2,
        "total_tokens": 120,
    }


def test_developer_audit_logger_update_metadata_preserves_identity(tmp_path):
    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    audit_logger.save_metadata({"model": "gpt-5.6", "served_model": "unavailable"})
    before = json.loads(audit_logger.metadata_path.read_text())

    audit_logger.update_metadata(
        {
            "served_model": "served-choice",
            "run_id": "rewritten",
            "created_at": "rewritten",
        }
    )

    after = json.loads(audit_logger.metadata_path.read_text())
    assert after["run_id"] == before["run_id"]
    assert after["created_at"] == before["created_at"]
    assert after["model"] == "gpt-5.6"
    assert after["served_model"] == "served-choice"


def test_log_run_items_captures_hosted_tool_search_items(tmp_path):
    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    items = [
        SimpleNamespace(type="message_output_item", raw_item={"type": "message"}),
        SimpleNamespace(type="tool_call_item", raw_item={"type": "function_call"}),
        SimpleNamespace(
            type="tool_call_item",
            raw_item={"type": "web_search_call", "id": "ws_123", "status": "completed"},
        ),
        SimpleNamespace(
            type="tool_search_call_item",
            raw_item={"type": "web_search_call", "query": "PX4 NAV_ACC_RAD"},
        ),
        SimpleNamespace(
            type="tool_search_output_item",
            raw_item={"type": "web_search_output", "results": [{"title": "PX4"}]},
        ),
        SimpleNamespace(
            type="tool_call_item",
            raw_item={"type": "shell_call", "call_id": "shell_123"},
        ),
        SimpleNamespace(
            type="tool_call_output_item",
            raw_item={"type": "shell_call_output", "call_id": "shell_123"},
        ),
    ]

    log_run_items(audit_logger, items)

    events = read_events(audit_logger.run_dir)
    assert [event["event"] for event in events] == [
        "hosted_tool.item",
        "hosted_tool.item",
        "hosted_tool.item",
        "hosted_tool.item",
        "hosted_tool.item",
    ]
    assert events[0]["raw_item"]["id"] == "ws_123"
    assert events[1]["raw_item"]["query"] == "PX4 NAV_ACC_RAD"
    assert events[2]["raw_item"]["results"] == [{"title": "PX4"}]
    assert events[3]["raw_item"]["type"] == "shell_call"
    assert events[4]["raw_item"]["type"] == "shell_call_output"


def _model_response(response_id=None, request_id=None):
    from agents import Usage
    from agents.items import ModelResponse

    return ModelResponse(
        output=[], usage=Usage(), response_id=response_id,
        request_id=request_id,
    )


def test_model_request_event_records_observed_ids_without_requested_settings(
    tmp_path,
):
    import asyncio

    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    hooks = AgentRunAuditHooks(audit_logger)
    asyncio.run(
        hooks.on_llm_end(
            None, None, _model_response(response_id="gen-123", request_id="req-456")
        )
    )
    events = read_events(audit_logger.run_dir)
    recorded = [e for e in events if e.get("event") == "model.request"]
    assert len(recorded) == 1
    assert recorded[0]["response_id"] == "gen-123"
    assert recorded[0]["request_id"] == "req-456"
    assert recorded[0]["served_model"] == "unavailable"
    assert recorded[0]["served_provider"] == "unavailable"
    assert recorded[0]["charged_cost"] == "unavailable"
    assert recorded[0]["routing"] == "unavailable"
    assert recorded[0]["served_reasoning_effort"] == "unavailable"
    assert not any(key.startswith("requested_") for key in recorded[0])


def test_model_request_event_marks_missing_ids_unavailable(tmp_path):
    import asyncio

    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    hooks = AgentRunAuditHooks(audit_logger)
    asyncio.run(hooks.on_llm_end(None, None, _model_response()))
    events = read_events(audit_logger.run_dir)
    recorded = [e for e in events if e.get("event") == "model.request"]
    assert len(recorded) == 1
    assert recorded[0]["response_id"] == "unavailable"
    assert recorded[0]["request_id"] == "unavailable"


def test_model_request_events_accumulate_across_requests(tmp_path):
    import asyncio

    from flight_log_agent.audit import extract_response_metadata

    first = extract_response_metadata(
        _model_response(response_id="gen-1"),
        raw=SimpleNamespace(id="gen-1", model="served-a"),
    )
    second = extract_response_metadata(
        _model_response(response_id="gen-2"),
        raw=SimpleNamespace(
            id="gen-2",
            model="served-b",
            provider="prov-b",
            cost=0.001,
            routing={"selected_model": "served-b", "effort": "low"},
        ),
    )
    assert first["served_model"] == "served-a"
    assert first["served_provider"] == "unavailable"
    assert first["charged_cost"] == "unavailable"
    assert second["served_model"] == "served-b"
    assert second["served_provider"] == "prov-b"
    assert second["charged_cost"] == 0.001
    assert second["routing"] == {"selected_model": "served-b", "effort": "low"}
    # Effort is never inferred from token counts or settings.
    assert second["served_reasoning_effort"] == "unavailable"

    audit_logger = DeveloperAuditLogger(tmp_path / "dev_logs", run_id="run_001")
    audit_logger.log_event("model.request", **first)
    audit_logger.log_event("model.request", **second)
    audit_logger.log_event("run.failed", error="boom")
    events = read_events(audit_logger.run_dir)
    recorded = [e for e in events if e.get("event") == "model.request"]
    assert [e["served_model"] for e in recorded] == ["served-a", "served-b"]
