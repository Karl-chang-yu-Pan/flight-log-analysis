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
