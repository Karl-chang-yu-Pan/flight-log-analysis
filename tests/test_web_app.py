from io import BytesIO
from types import SimpleNamespace

import pytest
from pyulog import ULog

import web_app
from web_app import (
    analysis_event_message,
    build_analysis_progress,
    public_ngrok_url,
    safe_upload_filename,
    save_upload_form,
    validate_ulog_file,
)


def test_public_ngrok_url_prefers_https():
    payload = {
        "tunnels": [
            {"public_url": "http://example.ngrok-free.app"},
            {"public_url": "https://example.ngrok-free.app"},
        ]
    }

    assert public_ngrok_url(payload) == "https://example.ngrok-free.app"


def test_public_ngrok_url_falls_back_to_first_public_url():
    payload = {"tunnels": [{"public_url": "http://example.ngrok-free.app"}]}

    assert public_ngrok_url(payload) == "http://example.ngrok-free.app"


def test_public_ngrok_url_handles_missing_tunnels():
    assert public_ngrok_url({}) is None


def test_safe_upload_filename_strips_paths_and_forces_suffix():
    assert safe_upload_filename("../../flight log", ".ulg") == "flight_log.ulg"
    assert safe_upload_filename("mission.plan", None) == "mission.plan"


def test_validate_ulog_file_checks_header(tmp_path):
    valid_log = tmp_path / "valid.ulg"
    invalid_log = tmp_path / "invalid.ulg"
    valid_log.write_bytes(ULog.HEADER_BYTES + b"payload")
    invalid_log.write_bytes(b"not-a-log")

    validate_ulog_file(valid_log)
    with pytest.raises(ValueError, match="not a valid ULog"):
        validate_ulog_file(invalid_log)


def test_save_upload_form_writes_files_under_unique_run_dir(tmp_path):
    form = {
        "log_file": SimpleNamespace(
            filename="../flight.ulg",
            file=BytesIO(ULog.HEADER_BYTES + b"payload"),
        ),
        "mission_file": SimpleNamespace(
            filename="mission.plan",
            file=BytesIO(b"{}"),
        ),
    }

    saved = save_upload_form(form, upload_root=tmp_path)

    assert saved["log_file"].name == "flight.ulg"
    assert saved["log_file"].parent.parent == tmp_path
    assert saved["log_file"].read_bytes().startswith(ULog.HEADER_BYTES)
    assert saved["mission_file"].read_bytes() == b"{}"


def test_analysis_event_message_maps_run_progress_events():
    assert analysis_event_message({
        "ts": "2026-05-11T00:00:00Z",
        "event": "prepass.started",
        "name": "parse_ulog_inventory",
    }) == {
        "ts": "2026-05-11T00:00:00Z",
        "event": "prepass.started",
        "message": "Parsing log inventory",
    }
    assert analysis_event_message({
        "event": "tool.started",
        "tool_name": "compute_log_metrics",
    })["message"] == "Computing log metrics"
    assert analysis_event_message({
        "event": "tool.started",
        "tool_name": "verify_hypothesis_against_log",
    })["message"] == "Verifying hypothesis against log"
    assert analysis_event_message({
        "event": "postprocess_plot.started",
        "name": "generate_signal_plot",
    })["message"] == "Generating report plots"


def test_build_analysis_progress_uses_latest_progress_message():
    progress = build_analysis_progress(
        "running",
        [
            {"event": "run.started"},
            {"event": "llm.started"},
            {"event": "tool.started", "tool_name": "search_px4_source"},
        ],
    )

    assert progress["phase"] == "Searching PX4 source"
    assert [message["message"] for message in progress["messages"]] == [
        "Started analysis run",
        "Analyzing with agent",
        "Searching PX4 source",
    ]


def test_resolve_artifact_path_allows_outputs_and_rejects_other_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path / "outputs")
    plot_dir = web_app.OUTPUT_ROOT / "test_artifacts"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "plot.png"
    plot_path.write_bytes(b"png")

    assert web_app.resolve_artifact_path(str(plot_path)) == plot_path.resolve()

    with pytest.raises(ValueError, match="not allowed"):
        web_app.resolve_artifact_path(str(tmp_path / "plot.png"))
