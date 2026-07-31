from io import BytesIO
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest
from pyulog import ULog

import flight_log_agent.web.server as web_app
from flight_log_agent.web.log_downloads import (
    extract_gps_coordinates,
    serialize_kml_track,
    serialize_parameter_rows,
)
from flight_log_agent.web.server import (
    analysis_event_message,
    browse_sync_config,
    build_analysis_progress,
    public_ngrok_url,
    safe_upload_filename,
    save_upload_form,
    validate_ulog_file,
)


def _json_request_handler(payload):
    body = json.dumps(payload).encode("utf-8")
    handler = object.__new__(web_app.FlightLogWebHandler)
    handler.headers = {"Content-Length": str(len(body))}
    handler.rfile = BytesIO(body)
    responses = []
    handler._send_json = lambda value, status=200: responses.append(
        (status, value)
    )
    return handler, responses


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


def test_browse_sync_config_uses_request_scoped_storage_override(tmp_path):
    original = web_app.BrowseConfig(
        browse_db_path=tmp_path / "browse.sqlite",
        flight_review_storage_path=tmp_path / "configured-storage",
        flight_review_db_path=tmp_path / "configured.sqlite",
        flight_review_log_dir=tmp_path / "configured-logs",
        airframe_image_root=tmp_path / "airframes",
    )

    assert browse_sync_config(original, {}) is original

    overridden = browse_sync_config(
        original,
        {"flight_review_storage_path": "  ~/flight-review-data  "},
    )

    assert overridden.browse_db_path == original.browse_db_path
    assert overridden.flight_review_storage_path == (
        Path.home() / "flight-review-data"
    )
    assert overridden.flight_review_db_path is None
    assert overridden.flight_review_log_dir is None
    assert overridden.airframe_image_root == original.airframe_image_root
    assert original.flight_review_db_path == tmp_path / "configured.sqlite"

    with pytest.raises(ValueError, match="must be a string"):
        browse_sync_config(original, {"flight_review_storage_path": 123})
    with pytest.raises(ValueError, match="must not be empty"):
        browse_sync_config(original, {"flight_review_storage_path": "  "})


def test_browse_sync_handler_passes_path_and_releases_lock(tmp_path, monkeypatch):
    configured = web_app.BrowseConfig(
        browse_db_path=tmp_path / "browse.sqlite",
        flight_review_storage_path=tmp_path / "configured",
    )
    captured = {}

    def fake_sync(config):
        captured["config"] = config
        return {"imported": 1, "added": 1}

    monkeypatch.setattr(web_app, "BROWSE_CONFIG", configured)
    monkeypatch.setattr(web_app, "sync_flight_review", fake_sync)
    handler, responses = _json_request_handler({
        "flight_review_storage_path": str(tmp_path / "entered"),
    })

    handler._handle_browse_import()

    assert responses == [(200, {"imported": 1, "added": 1})]
    assert captured["config"].flight_review_storage_path == tmp_path / "entered"
    assert web_app.FLIGHT_REVIEW_SYNC_LOCK.acquire(blocking=False)
    web_app.FLIGHT_REVIEW_SYNC_LOCK.release()


def test_browse_sync_handler_rejects_overlapping_request():
    handler, responses = _json_request_handler({})
    assert web_app.FLIGHT_REVIEW_SYNC_LOCK.acquire(blocking=False)
    try:
        handler._handle_browse_import()
    finally:
        web_app.FLIGHT_REVIEW_SYNC_LOCK.release()

    assert responses == [(
        409,
        {"error": "Flight Review synchronization is already running"},
    )]


def test_browse_sync_handler_releases_lock_after_failure(monkeypatch):
    def fail_sync(_config):
        raise RuntimeError("sync failed")

    monkeypatch.setattr(web_app, "sync_flight_review", fail_sync)
    handler, responses = _json_request_handler({})

    handler._handle_browse_import()

    assert responses == [(500, {"error": "RuntimeError('sync failed')"})]
    assert web_app.FLIGHT_REVIEW_SYNC_LOCK.acquire(blocking=False)
    web_app.FLIGHT_REVIEW_SYNC_LOCK.release()


def test_browse_sync_handler_returns_value_error_and_releases_lock(monkeypatch):
    def fail_sync(_config):
        raise ValueError("source schema is invalid")

    monkeypatch.setattr(web_app, "sync_flight_review", fail_sync)
    handler, responses = _json_request_handler({})

    handler._handle_browse_import()

    assert responses == [(400, {"error": "source schema is invalid"})]
    assert web_app.FLIGHT_REVIEW_SYNC_LOCK.acquire(blocking=False)
    web_app.FLIGHT_REVIEW_SYNC_LOCK.release()


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
        "event": "agent.question_intent.started",
    })["message"] == "Normalizing question intent"
    assert analysis_event_message({
        "event": "agent.resolve_mechanisms.finished",
    })["message"] == "Resolved PX4 source mechanisms"
    assert analysis_event_message({
        "event": "agent.final_report.retrying",
    })["message"] == "Retrying writing final report"
    assert analysis_event_message({
        "event": "agent.shell_analysis.started",
    })["message"] == "Analyzing log and source"
    assert analysis_event_message({
        "event": "agent.shell_analysis.finished",
    })["message"] == "Analyzed log and source"
    assert analysis_event_message({
        "event": "source.started",
        "name": "bounded_source_search",
    })["message"] == "Searching PX4 source"
    assert analysis_event_message({
        "event": "source.finished",
        "name": "resolve_px4_source_snapshot",
    })["message"] == "Resolved exact PX4 source snapshot"
    assert analysis_event_message({
        "event": "mechanism_cache.started",
        "name": "retrieve_mechanisms",
    })["message"] == "Retrieving cached mechanisms"
    assert analysis_event_message({
        "event": "mechanism_cache.finished",
        "name": "validate_mechanism_source:tecs",
    })["message"] == "Finished mechanism cache step: validate_mechanism_source:tecs"
    assert analysis_event_message({
        "event": "applicability.started",
        "name": "evaluate_applicability:TECS",
    })["message"] == "Running mechanism applicability check: evaluate_applicability:TECS"
    assert analysis_event_message({
        "event": "verification.started",
    })["message"] == "Verifying log signature"
    assert analysis_event_message({
        "event": "verification.finished",
    })["message"] == "Verified log signature"
    assert analysis_event_message({
        "event": "validation_after_downgrade.finished",
    })["message"] == "Validated downgraded report"
    assert analysis_event_message({
        "event": "postprocess_plot.started",
        "name": "generate_signal_plot",
    })["message"] == "Generating report plots"
    assert analysis_event_message({
        "event": "hosted_tool.item",
        "raw_item": {"type": "web_search_call"},
    })["message"] == "Using web search"
    for raw_item in [
        {"type": "web_search_output"},
        {"type": "shell_call"},
        {"type": "shell_call_output"},
        {"type": "file_search_call"},
        {},
        None,
    ]:
        assert analysis_event_message({
            "event": "hosted_tool.item",
            "raw_item": raw_item,
        }) is None


def test_build_analysis_progress_uses_latest_progress_message():
    progress = build_analysis_progress(
        "running",
        [
            {"event": "run.started"},
            {"event": "agent.question_intent.started"},
            {"event": "source.started", "name": "bounded_source_search"},
        ],
    )

    assert progress["phase"] == "Searching PX4 source"
    assert [message["message"] for message in progress["messages"]] == [
        "Started analysis run",
        "Normalizing question intent",
        "Searching PX4 source",
    ]


def test_build_analysis_progress_distinguishes_shell_from_web_search():
    progress = build_analysis_progress(
        "running",
        [
            {"event": "run.started"},
            {"event": "llm.started"},
            {
                "event": "hosted_tool.item",
                "raw_item": {"type": "shell_call"},
            },
            {
                "event": "hosted_tool.item",
                "raw_item": {"type": "shell_call_output"},
            },
            {
                "event": "hosted_tool.item",
                "raw_item": {"type": "web_search_call"},
            },
            {
                "event": "hosted_tool.item",
                "raw_item": {"type": "web_search_output"},
            },
        ],
    )

    assert progress["phase"] == "Using web search"
    assert [message["message"] for message in progress["messages"]] == [
        "Started analysis run",
        "Analyzing with agent",
        "Using web search",
    ]


def test_analysis_job_uses_shell_analyzer_and_preserves_run_contract(
    tmp_path,
    monkeypatch,
):
    import flight_log_agent.analysis.analyzer as shell_analyzer

    run_id = "shell_web_run"
    output_dir = tmp_path / "outputs" / f"web_{run_id}"
    report = {
        "airframe_summary": "unknown",
        "question_intent_summary": "question",
        "ranked_hypotheses": [],
        "excluded_mechanisms": [],
        "confirmed": [],
        "unconfirmed": [],
        "final_summary": "done",
    }
    captured = {}

    async def fake_analyze_flight_log(**kwargs):
        captured.update(kwargs)
        return report

    monkeypatch.setattr(shell_analyzer, "analyze_flight_log", fake_analyze_flight_log)
    monkeypatch.setattr(web_app, "WEB_DEV_LOG_ROOT", tmp_path / "dev_logs")
    with web_app.ANALYSIS_RUNS_LOCK:
        web_app.ANALYSIS_RUNS[run_id] = {
            "run_id": run_id,
            "status": "queued",
            "started_at": None,
            "finished_at": None,
            "error": None,
            "traceback": None,
            "log_path": str(tmp_path / "uploads" / "flight.ulg"),
            "mission_path": None,
            "source_path": str(tmp_path / "PX4-Autopilot"),
            "output_dir": str(output_dir),
            "report_path": str(output_dir / "report.json"),
            "dev_log_dir": str(web_app.WEB_DEV_LOG_ROOT / run_id),
            "report": None,
        }

    try:
        web_app._run_analysis_job(run_id, "What happened?")
        snapshot = web_app.analysis_run_snapshot(run_id)
    finally:
        with web_app.ANALYSIS_RUNS_LOCK:
            web_app.ANALYSIS_RUNS.pop(run_id, None)

    assert snapshot["status"] == "completed"
    assert snapshot["report"] == report
    assert captured == {
        "log_path": str(tmp_path / "uploads" / "flight.ulg"),
        "user_question": "What happened?",
        "mission_path": None,
        "source_path": str(tmp_path / "PX4-Autopilot"),
        "output_dir": str(output_dir),
        "dev_log_root": str(web_app.WEB_DEV_LOG_ROOT),
        "dev_run_id": run_id,
    }


def test_analysis_runs_keep_distinct_work_and_report_directories(
    tmp_path,
    monkeypatch,
):
    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)
    first = web_app.start_analysis_run({
        "log_path": str(tmp_path / "uploads" / "upload_1" / "flight.ulg"),
        "user_question": "First question",
    })
    second = web_app.start_analysis_run({
        "log_path": str(tmp_path / "uploads" / "upload_1" / "flight.ulg"),
        "user_question": "Second question",
    })

    try:
        assert first["run_id"] != second["run_id"]
        assert first["output_dir"] != second["output_dir"]
        assert first["report_path"] == str(
            Path(first["output_dir"]) / "report.json"
        )
        assert second["report_path"] == str(
            Path(second["output_dir"]) / "report.json"
        )
    finally:
        with web_app.ANALYSIS_RUNS_LOCK:
            web_app.ANALYSIS_RUNS.pop(first["run_id"], None)
            web_app.ANALYSIS_RUNS.pop(second["run_id"], None)


def test_analysis_run_uses_indexed_upload_and_persists_history(
    tmp_path,
    monkeypatch,
):
    class FakeThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            return None

    upload_root = tmp_path / "uploads"
    log_path = upload_root / "upload-123" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ulog")
    browse_row = {
        "id": "browse-123",
        "log_path": str(log_path),
        "review_inputs": {
            "mission_path": str(log_path.parent / "mission.plan"),
            "source_path": str(tmp_path / "PX4-Autopilot"),
        },
    }
    monkeypatch.setattr(web_app, "UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(web_app, "get_log", lambda _db_path, _log_id: browse_row)
    monkeypatch.setattr(web_app.threading, "Thread", FakeThread)

    run = web_app.start_analysis_run({
        "browse_log_id": "browse-123",
        "log_path": str(tmp_path / "untrusted.ulg"),
        "user_question": "Why did RTL start?",
    })

    try:
        assert run["log_path"] == str(log_path)
        assert run["mission_path"] == browse_row["review_inputs"]["mission_path"]
        assert run["source_path"] == browse_row["review_inputs"]["source_path"]
        history_path = (
            log_path.parent
            / "analysis_history"
            / f"{run['run_id']}.json"
        )
        queued = web_app.load_history_records(history_path.parent)
        assert queued[0]["question"] == "Why did RTL start?"
        assert queued[0]["status"] == "queued"

        report = {
            "final_summary": "RTL was triggered by the configured failsafe.",
            "ranked_hypotheses": [{"title": "Failsafe"}],
        }
        web_app._update_analysis_run(
            run["run_id"],
            status="completed",
            finished_at=30.0,
            report=report,
        )

        persisted = json.loads(history_path.read_text(encoding="utf-8"))
        assert persisted["question"] == "Why did RTL start?"
        assert persisted["report"] == report
        history = web_app.analysis_history_snapshot("browse-123")
        assert history == {
            "browse_log_id": "browse-123",
            "entries": [persisted],
        }
    finally:
        with web_app.ANALYSIS_RUNS_LOCK:
            web_app.ANALYSIS_RUNS.pop(run["run_id"], None)


def test_resolve_artifact_path_allows_outputs_and_rejects_other_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path / "outputs")
    plot_dir = web_app.OUTPUT_ROOT / "test_artifacts"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_path = plot_dir / "plot.png"
    plot_path.write_bytes(b"png")

    assert web_app.resolve_artifact_path(str(plot_path)) == plot_path.resolve()

    with pytest.raises(ValueError, match="not allowed"):
        web_app.resolve_artifact_path(str(tmp_path / "plot.png"))


def test_parameter_download_serialization_supports_all_and_non_default_rows():
    rows = [
        {"name": "COM_ARM_WO_GPS", "raw_value": 1, "default_status": "default"},
        {"name": "MPC_XY_CRUISE", "raw_value": 8.5, "default_status": "non_default"},
    ]

    assert serialize_parameter_rows(rows, non_default_only=False).decode("utf-8") == (
        "1\t1\tCOM_ARM_WO_GPS\t1\t6\n"
        "1\t1\tMPC_XY_CRUISE\t8.5\t9\n"
    )
    assert serialize_parameter_rows(rows, non_default_only=True).decode("utf-8") == (
        "1\t1\tMPC_XY_CRUISE\t8.5\t9\n"
    )


def test_gps_download_supports_new_and_legacy_field_units():
    new_fields = {
        "latitude_deg": [25.0, 25.1],
        "longitude_deg": [121.0, 121.1],
        "altitude_msl_m": [100.0, 101.0],
        "fix_type": [2, 3],
    }
    legacy_fields = {
        "lat": [250000000],
        "lon": [1210000000],
        "alt": [100000],
        "fix_type": [3],
    }

    assert extract_gps_coordinates(new_fields) == [(121.1, 25.1, 101.0)]
    assert extract_gps_coordinates(legacy_fields) == [(121.0, 25.0, 100.0)]


def test_kml_download_contains_flight_track_coordinates():
    payload = serialize_kml_track([(121.0, 25.0, 100.0)])
    root = ET.fromstring(payload)
    coordinates = root.find(".//{http://www.opengis.net/kml/2.2}coordinates")

    assert coordinates is not None
    assert coordinates.text == "121.000000000,25.000000000,100.000"


def test_upload_review_and_browse_pages_have_separate_navigation_contracts():
    upload_html = (web_app.WEB_DIR / "index.html").read_text(encoding="utf-8")
    review_html = (web_app.WEB_DIR / "review.html").read_text(encoding="utf-8")
    browse_html = (web_app.WEB_DIR / "browse.html").read_text(encoding="utf-8")
    browse_js = (web_app.WEB_DIR / "browse.js").read_text(encoding="utf-8")

    assert 'id="uploadForm"' in upload_html
    assert 'id="uploadProgress"' in upload_html
    assert 'id="appShell"' not in upload_html
    assert 'id="uploadForm"' not in review_html
    assert 'id="downloadMenu"' in review_html
    assert 'id="reviewLoadProgress"' in review_html
    assert 'id="logTagSearch"' in review_html
    assert 'id="plotNavigationMenu"' in review_html
    assert 'id="plotNavigation"' in review_html
    assert 'id="flightReviewStoragePath"' in browse_html
    assert 'id="flightReviewSyncStatus"' in browse_html
    assert "Synchronize" in browse_html
    assert "flight_review_storage_path: storagePath" in browse_js
    assert 'count("imported")' in browse_js
    assert 'fetchJson("/api/browse-import-flight-review"' in browse_js
    assert 'count("newly_unavailable")' in browse_js
    assert 'count("unavailable")' in browse_js
    assert "/review?browse_id=" in browse_js


def test_review_analysis_uses_durable_independent_history_timeline():
    review_html = (web_app.WEB_DIR / "review.html").read_text(encoding="utf-8")
    app_js = (web_app.WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (web_app.WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert 'id="analysisTimeline"' in review_html
    assert "Ask about this log" in review_html
    assert "/api/analysis-history?browse_log_id=" in app_js
    assert "browse_log_id: currentBrowseLogId()" in app_js
    assert "state.analysisHistory" in app_js
    assert "renderAnalysisTurn" in app_js
    assert "View full structured analysis" in app_js
    assert ".analysis-message-user" in styles
    assert ".analysis-message-assistant" in styles


def test_review_plot_controls_use_dropdown_navigation_and_tracker_overlays():
    review_html = (web_app.WEB_DIR / "review.html").read_text(encoding="utf-8")
    app_js = (web_app.WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (web_app.WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert '<summary>Navigation</summary>' in review_html
    assert 'id="plotNavigation"' in review_html
    assert 'plotTrackerCanvas-${escapeAttr(plot.id)}' in app_js
    assert "requestAnimationFrame(flushSharedPlotTracker)" in app_js
    assert "drawInteractivePlotTrackers(plots, { visibleOnly: true })" in app_js
    assert "new window.IntersectionObserver" in app_js
    assert 'root: els.plotSidebar' in app_js
    assert 'els.plotSidebar.addEventListener("scroll"' not in app_js
    assert "plotTrackerDrawnRevisions" in app_js
    assert "state.plotVisibilityObserver !== observer" in app_js
    assert "plotSectionIsNearViewport(section)" in app_js
    assert "drawInteractivePlots(plots, { visibleOnly: observerActive })" in app_js
    assert "if (!isPlotVisibleOrUnobserved(plot.id)) return;" in app_js
    assert ".plot-navigation-menu" in styles
    assert "position: absolute" in styles

    tracker_handler = app_js.split("function setSharedPlotTracker", 1)[1].split(
        "function schedulePlotResize",
        1,
    )[0]
    left_sidebar_handler = app_js.split(
        'els.sidebarToggle.addEventListener("click"',
        1,
    )[1].split('els.plotSidebarToggle.addEventListener("click"', 1)[0]
    right_sidebar_handler = app_js.split(
        'els.plotSidebarToggle.addEventListener("click"',
        1,
    )[1].split('window.addEventListener("resize"', 1)[0]

    assert "drawInteractivePlots" not in tracker_handler
    assert "drawInteractivePlotTrackers(plots, { visibleOnly: true })" in tracker_handler
    assert "drawInteractivePlots" not in left_sidebar_handler
    assert "drawInteractivePlots" not in right_sidebar_handler


def test_review_plot_renderer_supports_additive_flight_review_contract():
    app_js = (web_app.WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert 'const showTimeControls = plot.kind !== "spectrum";' in app_js
    assert 'plot.kind !== "local_position" && plot.kind !== "spectrum"' in app_js
    assert 'if (plot.kind === "spectrum") return;' in app_js
    assert 'series.interpolation === "step_after" ? "previous" : "linear"' in app_js
    assert 'interpolation === "step_after"' in app_js
    assert "series.frequencies_hz || []" in app_js
    assert "series.values || []" in app_js
    assert "marker.frequency_hz" in app_js
    assert "band.min == null ? yRange[0]" in app_js
    assert "band.max == null ? yRange[1]" in app_js
    assert "span.start_x" in app_js
    assert "span.end_x" in app_js
    assert "!span.series_key" in app_js
    assert "isPlotSourceVisible(plot.id, span.series_key)" in app_js

    timeseries_renderer = app_js.split(
        "function drawTimeseriesPlot",
        1,
    )[1].split("function drawTimeseriesTracker", 1)[0]
    band_index = timeseries_renderer.index(
        "drawHorizontalBands(ctx, bounds, yRange, plot.horizontal_bands || [])"
    )
    grid_index = timeseries_renderer.index("drawGrid(ctx, bounds)")
    series_index = timeseries_renderer.index("visibleSeries.forEach")
    span_index = timeseries_renderer.index("drawHorizontalSpans")

    assert band_index < grid_index < series_index < span_index

    spectrum_renderer = app_js.split(
        "function drawSpectrumPlot",
        1,
    )[1].split("function drawSpectrogramImage", 1)[0]
    assert "drawHorizontalSpans(" in spectrum_renderer
    assert "drawFrequencyMarkers(" in spectrum_renderer


def test_review_tags_are_searchable_and_load_progress_covers_plot_generation():
    app_js = (web_app.WEB_DIR / "app.js").read_text(encoding="utf-8")
    upload_js = (web_app.WEB_DIR / "upload.js").read_text(encoding="utf-8")

    assert 'logTagSearch.addEventListener("input", renderLogTags)' in app_js
    assert "tag.toLowerCase().includes(query)" in app_js
    assert 'setReviewLoadProgress("Loading flight log")' in app_js
    assert 'setReviewLoadProgress("Generating plots")' in app_js
    assert "await waitForPlotPaint()" in app_js
    assert 'setUploadProcessing("Processing log")' in upload_js
    assert 'setUploadProcessing("Processing local log")' in upload_js
