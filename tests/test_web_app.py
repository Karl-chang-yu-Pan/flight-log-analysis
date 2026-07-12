from io import BytesIO
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
        "event": "agent.question_intent.started",
    })["message"] == "Normalizing question intent"
    assert analysis_event_message({
        "event": "agent.resolve_mechanisms.finished",
    })["message"] == "Resolved PX4 source mechanisms"
    assert analysis_event_message({
        "event": "agent.final_report.retrying",
    })["message"] == "Retrying writing final report"
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
    assert "/review?browse_id=" in browse_js


def test_review_plot_controls_use_dropdown_navigation_and_tracker_overlays():
    review_html = (web_app.WEB_DIR / "review.html").read_text(encoding="utf-8")
    app_js = (web_app.WEB_DIR / "app.js").read_text(encoding="utf-8")
    styles = (web_app.WEB_DIR / "styles.css").read_text(encoding="utf-8")

    assert '<summary>Navigation</summary>' in review_html
    assert 'id="plotNavigation"' in review_html
    assert 'plotTrackerCanvas-${escapeAttr(plot.id)}' in app_js
    assert "requestAnimationFrame(flushSharedPlotTracker)" in app_js
    assert "drawInteractivePlotTrackers(plots, { visibleOnly: true })" in app_js
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
