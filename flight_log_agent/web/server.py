from __future__ import annotations

import argparse
import asyncio
import cgi
import json
import mimetypes
import os
import subprocess
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import urlopen

from pyulog import ULog
from flight_log_agent.ulog.preparse_view import build_preparse_payload
from flight_log_agent.audit import DEFAULT_DEV_LOG_ROOT, make_json_safe
from flight_log_agent.ulog.interactive_plots import build_interactive_plot_payload
from flight_log_agent.web.analysis_history import (
    load_history_records,
    resolve_history_directory,
    write_history_record,
)
from flight_log_agent.web.browse_index import (
    BrowseConfig,
    add_log_tag,
    create_tag,
    ensure_browse_db,
    get_log,
    import_flight_review,
    list_tags,
    load_browse_airframe_metadata,
    query_logs,
    refresh_airframe_image_keys,
    remove_log_tag,
    resolve_airframe_image_root,
    upsert_log_from_path,
)
from flight_log_agent.web.log_downloads import (
    build_kml_download,
    build_parameter_download,
)


def _optional_env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"
UPLOAD_ROOT = ROOT_DIR / "uploads"
OUTPUT_ROOT = ROOT_DIR / "outputs"
WEB_DEV_LOG_ROOT = ROOT_DIR / DEFAULT_DEV_LOG_ROOT
BROWSE_CONFIG = BrowseConfig(
    browse_db_path=Path(os.environ.get("FLIGHT_LOG_BROWSE_DB_PATH", OUTPUT_ROOT / "browse.sqlite")),
    flight_review_storage_path=_optional_env_path("FLIGHT_REVIEW_STORAGE_PATH"),
    flight_review_db_path=_optional_env_path("FLIGHT_REVIEW_DB_PATH"),
    flight_review_log_dir=_optional_env_path("FLIGHT_REVIEW_LOG_DIR"),
    airframe_image_root=Path(os.environ.get("AIRFRAME_IMAGE_ROOT", WEB_DIR / "airframes")),
)
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
UPLOAD_FIELDS = {
    "log_file": ".ulg",
    "mission_file": None,
    "parameters_xml_file": ".xml",
}

ANALYSIS_RUNS: dict[str, dict[str, Any]] = {}
ANALYSIS_RUNS_LOCK = threading.Lock()


class FlightLogWebHandler(BaseHTTPRequestHandler):
    server_version = "FlightLogPreparseHTTP/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_file(WEB_DIR / "index.html")
            return

        if path == "/upload":
            self._serve_file(WEB_DIR / "index.html")
            return

        if path == "/review":
            self._serve_file(WEB_DIR / "review.html")
            return

        if path == "/browse":
            self._serve_file(WEB_DIR / "browse.html")
            return

        if path == "/api/browse-logs":
            self._handle_browse_logs(parsed.query)
            return

        if path == "/api/browse-log":
            self._handle_browse_log(parsed.query)
            return

        if path == "/api/browse-tags":
            self._send_json({"tags": list_tags(BROWSE_CONFIG.browse_db_path)})
            return

        if path == "/api/browse-config":
            self._send_json(browse_config_payload())
            return

        if path == "/api/browse-download":
            self._handle_browse_download(parsed.query)
            return

        if path.startswith("/airframe_img/"):
            self._handle_airframe_image(path.removeprefix("/airframe_img/"))
            return

        if path.startswith("/api/analyze-runs/"):
            self._handle_analysis_status(path.removeprefix("/api/analyze-runs/"))
            return

        if path == "/api/analysis-history":
            self._handle_analysis_history(parsed.query)
            return

        if path == "/artifacts":
            self._handle_artifact(parsed.query)
            return

        requested = (WEB_DIR / unquote(path.lstrip("/"))).resolve()
        try:
            requested.relative_to(WEB_DIR.resolve())
        except ValueError:
            self._send_json({"error": "invalid path"}, status=400)
            return

        self._serve_file(requested)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/preparse":
            self._handle_preparse_json()
            return

        if parsed.path == "/api/upload-preparse":
            self._handle_upload_preparse()
            return

        if parsed.path == "/api/analyze-runs":
            self._handle_start_analysis()
            return

        if parsed.path == "/api/interactive-plots":
            self._handle_interactive_plots()
            return

        if parsed.path == "/api/browse-import-flight-review":
            self._handle_browse_import()
            return

        if parsed.path == "/api/browse-tags":
            self._handle_create_browse_tag()
            return

        if parsed.path == "/api/browse-log-tags":
            self._handle_browse_log_tag()
            return

        self._send_json({"error": "not found"}, status=404)

    def _handle_preparse_json(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        browse_row = None
        browse_log_id = str(payload.get("browse_log_id") or "").strip()
        if browse_log_id:
            try:
                browse_row = get_log(BROWSE_CONFIG.browse_db_path, browse_log_id)
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=400)
                return
            except Exception as exc:
                self._send_json({"error": repr(exc)}, status=500)
                return
            log_path = str(browse_row["log_path"] or "").strip()
        else:
            log_path = str(payload.get("log_path") or "").strip()
            if not log_path:
                self._send_json({"error": "log_path is required"}, status=400)
                return

        try:
            stored_inputs = (browse_row or {}).get("review_inputs") or {}
            mission_path = _optional_payload_path(payload, "mission_path") or stored_inputs.get(
                "mission_path"
            )
            source_path = _optional_payload_path(payload, "source_path") or stored_inputs.get(
                "source_path"
            )
            parameters_xml_path = _optional_payload_path(
                payload,
                "parameters_xml_path",
            ) or stored_inputs.get("parameters_xml_path")
            result = build_preparse_payload(
                log_path,
                mission_path=mission_path,
                source_path=source_path,
                parameters_xml_path=parameters_xml_path,
            )
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return

        if browse_row is not None:
            result["browse"] = {"indexed": True, "log_id": browse_row["id"], "row": browse_row}
        else:
            result["browse"] = index_browse_log(
                Path(log_path),
                source_kind="local_path",
                original_filename=Path(log_path).name,
                review_inputs={
                    "mission_path": mission_path,
                    "source_path": source_path,
                    "parameters_xml_path": parameters_xml_path,
                },
            )
        self._send_json(result)

    def _handle_upload_preparse(self) -> None:
        content_type = self.headers.get("Content-Type", "")
        if not content_type.startswith("multipart/form-data"):
            self._send_json({"error": "multipart/form-data is required"}, status=400)
            return

        try:
            form = self._read_multipart_form()
            saved_files = save_upload_form(form)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        log_path = saved_files.get("log_file")
        if log_path is None:
            self._send_json({"error": "log_file is required"}, status=400)
            return

        try:
            validate_ulog_file(log_path)
            result = build_preparse_payload(
                log_path,
                mission_path=saved_files.get("mission_file"),
                source_path=_form_value(form, "source_path"),
                parameters_xml_path=saved_files.get("parameters_xml_file"),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return

        result["upload"] = {
            "run_id": log_path.parent.name,
            "log_path": str(log_path),
            "mission_path": str(saved_files["mission_file"]) if saved_files.get("mission_file") else None,
            "parameters_xml_path": (
                str(saved_files["parameters_xml_file"])
                if saved_files.get("parameters_xml_file")
                else None
            ),
        }
        result["browse"] = index_browse_log(
            log_path,
            source_kind="upload",
            source_log_id=log_path.parent.name,
            original_filename=log_path.name,
            review_inputs={
                "mission_path": saved_files.get("mission_file"),
                "source_path": _form_value(form, "source_path"),
                "parameters_xml_path": saved_files.get("parameters_xml_file"),
            },
        )
        self._send_json(result)

    def _handle_browse_logs(self, query: str) -> None:
        params = parse_qs(query)
        try:
            payload = query_logs(
                BROWSE_CONFIG.browse_db_path,
                search=_first_param(params, "search"),
                tags=_tag_params(params),
                upload_start=_first_param(params, "upload_start"),
                upload_end=_first_param(params, "upload_end"),
                log_start=_first_param(params, "log_start"),
                log_end=_first_param(params, "log_end"),
                sort=_first_param(params, "sort") or "upload_date",
                direction=_first_param(params, "direction") or "desc",
                limit=_int_param(params, "limit", 50),
                offset=_int_param(params, "offset", 0),
            )
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json(payload)

    def _handle_browse_log(self, query: str) -> None:
        params = parse_qs(query)
        try:
            row = get_log(BROWSE_CONFIG.browse_db_path, _first_param(params, "log_id"))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json({"log": row})

    def _handle_browse_download(self, query: str) -> None:
        params = parse_qs(query)
        log_id = _first_param(params, "log_id")
        download_type = _first_param(params, "type") or "ulog"
        try:
            row = get_log(BROWSE_CONFIG.browse_db_path, log_id)
            log_path = Path(str(row.get("log_path") or ""))
            if not log_path.is_file():
                raise FileNotFoundError("log file does not exist")
            original_name = safe_upload_filename(
                str(row.get("original_filename") or log_path.name)
            )
            filename_stem = Path(original_name).stem or "flight-log"

            if download_type == "ulog":
                self._send_download_file(log_path, original_name)
                return
            if download_type == "parameters":
                data = build_parameter_download(log_path, non_default_only=False)
                self._send_download_bytes(data, f"{filename_stem}.params")
                return
            if download_type == "parameters_non_default":
                data = build_parameter_download(log_path, non_default_only=True)
                self._send_download_bytes(data, f"{filename_stem}-non-default.params")
                return
            if download_type == "kml":
                data = build_kml_download(log_path)
                self._send_download_bytes(
                    data,
                    f"{filename_stem}.kml",
                    content_type="application/vnd.google-earth.kml+xml",
                )
                return
            raise ValueError("unsupported download type")
        except FileNotFoundError as exc:
            self._send_json({"error": str(exc)}, status=404)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)

    def _handle_browse_import(self) -> None:
        try:
            payload = import_flight_review(BROWSE_CONFIG)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json(payload)

    def _handle_create_browse_tag(self) -> None:
        try:
            payload = self._read_json_body()
            tag = create_tag(BROWSE_CONFIG.browse_db_path, str(payload.get("name") or ""))
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json({"tag": tag, "tags": list_tags(BROWSE_CONFIG.browse_db_path)})

    def _handle_browse_log_tag(self) -> None:
        try:
            payload = self._read_json_body()
            log_id = str(payload.get("log_id") or "")
            tag = str(payload.get("tag") or "")
            action = str(payload.get("action") or "add")
            if action == "remove":
                remove_log_tag(BROWSE_CONFIG.browse_db_path, log_id, tag)
            else:
                add_log_tag(BROWSE_CONFIG.browse_db_path, log_id, tag)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json({"ok": True, "tags": list_tags(BROWSE_CONFIG.browse_db_path)})

    def _handle_airframe_image(self, image_name: str) -> None:
        image_name = unquote(image_name).strip().lstrip("/")
        if not image_name:
            self._send_json({"error": "image is required"}, status=400)
            return
        image_root = resolve_airframe_image_root(BROWSE_CONFIG)
        requested = (image_root / image_name).resolve()
        try:
            requested.relative_to(image_root.resolve())
        except ValueError:
            self._send_json({"error": "invalid image path"}, status=400)
            return
        if not requested.is_file():
            fallback = (image_root / "AirframeUnknown.svg").resolve()
            requested = fallback if fallback.is_file() else requested
        self._serve_file(requested)

    def _handle_start_analysis(self) -> None:
        try:
            payload = self._read_json_body()
            run = start_analysis_run(payload)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return

        self._send_json(analysis_run_snapshot(run["run_id"]))

    def _handle_analysis_status(self, run_id: str) -> None:
        run_id = run_id.strip("/")
        if not run_id:
            self._send_json({"error": "run_id is required"}, status=400)
            return

        snapshot = analysis_run_snapshot(run_id)
        if snapshot is None:
            self._send_json({"error": "analysis run not found"}, status=404)
            return

        self._send_json(snapshot)

    def _handle_analysis_history(self, query: str) -> None:
        browse_log_id = _first_param(parse_qs(query), "browse_log_id")
        try:
            snapshot = analysis_history_snapshot(browse_log_id)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json(snapshot)

    def _handle_interactive_plots(self) -> None:
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        log_path = str(payload.get("log_path") or "").strip()
        if not log_path:
            self._send_json({"error": "log_path is required"}, status=400)
            return

        try:
            result = build_interactive_plot_payload(log_path)
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return

        self._send_json(result)

    def _handle_artifact(self, query: str) -> None:
        path_values = parse_qs(query).get("path") or []
        if not path_values:
            self._send_json({"error": "path is required"}, status=400)
            return

        try:
            artifact_path = resolve_artifact_path(path_values[0])
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return

        self._serve_file(artifact_path)

    def _read_multipart_form(self) -> cgi.FieldStorage:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc

        if content_length <= 0:
            raise ValueError("empty upload")
        if content_length > MAX_UPLOAD_BYTES:
            raise ValueError(
                f"upload is too large; max is {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB"
            )

        return cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                "CONTENT_LENGTH": str(content_length),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _read_json_body(self) -> dict[str, Any]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc

        if content_length <= 0:
            return {}
        if content_length > 1_000_000:
            raise ValueError("request body too large")

        raw_body = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def _serve_file(self, path: Path) -> None:
        if not path.is_file():
            self._send_json({"error": "not found"}, status=404)
            return

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_download_file(self, path: Path, filename: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        with path.open("rb") as file:
            while chunk := file.read(1024 * 1024):
                self.wfile.write(chunk)

    def _send_download_bytes(
        self,
        data: bytes,
        filename: str,
        *,
        content_type: str = "application/octet-stream",
    ) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: Any, *, status: int = 200) -> None:
        data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _optional_payload_path(payload: dict[str, Any], key: str) -> str | None:
    value = str(payload.get(key) or "").strip()
    return value or None


def start_analysis_run(payload: dict[str, Any]) -> dict[str, Any]:
    browse_log_id = str(payload.get("browse_log_id") or "").strip()
    browse_row = (
        get_log(BROWSE_CONFIG.browse_db_path, browse_log_id)
        if browse_log_id
        else None
    )
    if browse_row is not None:
        log_path = str(browse_row.get("log_path") or "").strip()
    else:
        log_path = str(payload.get("log_path") or "").strip()
    if not log_path:
        raise ValueError("log_path is required")

    user_question = str(payload.get("user_question") or "").strip()
    if not user_question:
        raise ValueError("user_question is required")

    stored_inputs = (browse_row or {}).get("review_inputs") or {}
    run_id = uuid.uuid4().hex
    output_dir = OUTPUT_ROOT / f"web_{run_id}"
    history_dir = resolve_history_directory(
        log_path,
        upload_root=UPLOAD_ROOT,
        fallback_root=OUTPUT_ROOT / "analysis_history",
        history_key=browse_log_id or None,
    )
    run = {
        "run_id": run_id,
        "browse_log_id": browse_log_id or None,
        "status": "queued",
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "history_error": None,
        "traceback": None,
        "log_path": log_path,
        "mission_path": (
            _optional_payload_path(payload, "mission_path")
            or stored_inputs.get("mission_path")
        ),
        "source_path": (
            _optional_payload_path(payload, "source_path")
            or stored_inputs.get("source_path")
        ),
        "user_question": user_question,
        "output_dir": str(output_dir),
        "report_path": str(output_dir / "report.json"),
        "dev_log_dir": str(WEB_DEV_LOG_ROOT / run_id),
        "history_dir": str(history_dir),
        "report": None,
    }

    _persist_analysis_run_history(run)
    with ANALYSIS_RUNS_LOCK:
        ANALYSIS_RUNS[run_id] = run

    thread = threading.Thread(
        target=_run_analysis_job,
        args=(run_id,),
        name=f"flight-log-analysis-{run_id[:8]}",
        daemon=True,
    )
    thread.start()
    return run


def analysis_run_snapshot(run_id: str) -> dict[str, Any] | None:
    with ANALYSIS_RUNS_LOCK:
        run = ANALYSIS_RUNS.get(run_id)
        if run is None:
            return None
        snapshot = {
            key: value
            for key, value in run.items()
            if key not in {"history_dir", "traceback"}
        }

    events = read_analysis_events(run_id)
    snapshot["events"] = events
    snapshot["progress"] = build_analysis_progress(snapshot["status"], events)
    return snapshot


def analysis_history_snapshot(browse_log_id: str) -> dict[str, Any]:
    clean_browse_log_id = str(browse_log_id or "").strip()
    if not clean_browse_log_id:
        raise ValueError("browse_log_id is required")
    browse_row = get_log(BROWSE_CONFIG.browse_db_path, clean_browse_log_id)
    log_path = str(browse_row.get("log_path") or "").strip()
    if not log_path:
        raise ValueError("indexed log has no log_path")
    history_dir = resolve_history_directory(
        log_path,
        upload_root=UPLOAD_ROOT,
        fallback_root=OUTPUT_ROOT / "analysis_history",
        history_key=clean_browse_log_id,
    )
    return {
        "browse_log_id": clean_browse_log_id,
        "entries": load_history_records(history_dir),
    }


def read_analysis_events(run_id: str) -> list[dict[str, Any]]:
    events_path = WEB_DEV_LOG_ROOT / run_id / "run_events.jsonl"
    if not events_path.is_file():
        return []

    events = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def build_analysis_progress(status: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    messages = [analysis_event_message(event) for event in events]
    messages = [message for message in messages if message is not None]
    phase = messages[-1]["message"] if messages else status_label(status)
    return {
        "phase": phase,
        "messages": messages[-80:],
    }


def analysis_event_message(event: dict[str, Any]) -> dict[str, Any] | None:
    event_name = event.get("event")
    name = event.get("name")
    tool_name = event.get("tool_name")
    message = None

    if event_name == "run.started":
        message = "Started analysis run"
    elif event_name == "run.finished":
        message = "Analysis complete"
    elif event_name == "run.failed":
        message = "Analysis failed"
    elif event_name.startswith("agent."):
        message = agent_progress_message(event_name)
    elif event_name == "prepass.started":
        message = {
            "parse_ulog_inventory": "Parsing log inventory",
            "build_basic_timeline": "Building flight timeline",
            "infer_control_surface": "Inferring control-surface assumptions",
            "parse_mission_file": "Parsing mission file",
        }.get(str(name), f"Running pre-pass: {name}")
    elif event_name == "prepass.finished":
        message = {
            "parse_ulog_inventory": "Parsed log inventory",
            "build_basic_timeline": "Built flight timeline",
            "infer_control_surface": "Inferred control-surface assumptions",
            "parse_mission_file": "Parsed mission file",
        }.get(str(name), f"Finished pre-pass: {name}")
    elif event_name == "llm.started":
        message = "Analyzing with agent"
    elif event_name == "llm.finished":
        message = "Agent reasoning step complete"
    elif event_name == "tool.started":
        message = tool_progress_message(str(tool_name), started=True)
    elif event_name == "tool.finished":
        message = tool_progress_message(str(tool_name), started=False)
    elif event_name == "verification.started":
        message = "Verifying log signature"
    elif event_name == "verification.finished":
        message = "Verified log signature"
    elif event_name.startswith("source."):
        message = named_stage_progress_message(
            event_name,
            str(name),
            {
                "resolve_px4_source_snapshot": (
                    "Resolving exact PX4 source snapshot",
                    "Resolved exact PX4 source snapshot",
                ),
                "bounded_source_search": (
                    "Searching PX4 source",
                    "Searched PX4 source",
                ),
            },
            "source step",
        )
    elif event_name.startswith("mechanism_cache."):
        message = named_stage_progress_message(
            event_name,
            str(name),
            {
                "retrieve_mechanisms": (
                    "Retrieving cached mechanisms",
                    "Retrieved cached mechanisms",
                ),
                "write_resolved_mechanisms": (
                    "Writing mechanism cache",
                    "Wrote mechanism cache",
                ),
            },
            "mechanism cache step",
        )
    elif event_name.startswith("applicability."):
        message = named_stage_progress_message(
            event_name,
            str(name),
            {},
            "mechanism applicability check",
        )
    elif event_name == "validation.finished":
        message = "Validated report"
    elif event_name == "validation_after_repair.finished":
        message = "Validated repaired report"
    elif event_name == "validation_after_downgrade.finished":
        message = "Validated downgraded report"
    elif event_name == "postprocess_plot.started":
        message = "Generating report plots"
    elif event_name == "postprocess_plot.finished":
        message = "Generated report plots"
    elif event_name == "hosted_tool.item":
        raw_item = event.get("raw_item")
        if (
            isinstance(raw_item, dict)
            and raw_item.get("type") == "web_search_call"
        ):
            message = "Using web search"

    if message is None:
        return None

    return {
        "ts": event.get("ts"),
        "event": event_name,
        "message": message,
    }


def agent_progress_message(event_name: str) -> str | None:
    parts = event_name.split(".")
    if len(parts) < 3:
        return None

    stage = parts[1]
    status = parts[2]
    started = status == "started"
    finished = status == "finished"
    retrying = status == "retrying"
    failed = status == "failed"

    stage_messages = [
        ("question_intent", "Normalizing question intent", "Normalized question intent"),
        ("resolve_mechanisms", "Resolving PX4 source mechanisms", "Resolved PX4 source mechanisms"),
        ("draft_hypotheses", "Drafting candidate hypotheses", "Drafted candidate hypotheses"),
        ("resolve_mechanism_", "Resolving PX4 source mechanism", "Resolved PX4 source mechanism"),
        ("build_signature_", "Building log signature checks", "Built log signature checks"),
        ("final_report", "Writing final report", "Wrote final report"),
        ("repair_report", "Repairing report", "Repaired report"),
        ("shell_analysis", "Analyzing log and source", "Analyzed log and source"),
    ]
    for prefix, started_message, finished_message in stage_messages:
        if stage == prefix or stage.startswith(prefix):
            if retrying:
                return f"Retrying {started_message[0].lower()}{started_message[1:]}"
            if failed:
                return f"Failed while {started_message[0].lower()}{started_message[1:]}"
            if started:
                return started_message
            if finished:
                return finished_message

    return None


def named_stage_progress_message(
    event_name: str,
    name: str,
    known_messages: dict[str, tuple[str, str]],
    fallback_label: str,
) -> str | None:
    status = event_name.rsplit(".", 1)[-1]
    started = status == "started"
    finished = status == "finished"
    failed = status == "failed"

    started_message, finished_message = known_messages.get(
        name,
        (
            f"Running {fallback_label}: {name}",
            f"Finished {fallback_label}: {name}",
        ),
    )
    if started:
        return started_message
    if finished:
        return finished_message
    if failed:
        return f"Failed while {started_message[0].lower()}{started_message[1:]}"
    return None


def tool_progress_message(tool_name: str, *, started: bool) -> str:
    action = "Finished" if not started else None
    if tool_name == "compute_log_metrics":
        return "Computing log metrics" if started else "Computed log metrics"
    if tool_name == "generate_signal_plot":
        return "Generating plot" if started else "Generated plot"
    if tool_name == "evaluate_log_signature":
        return "Verifying log signature" if started else "Verified log signature"
    if tool_name == "search_px4_source":
        return "Searching PX4 source" if started else "Searched PX4 source"
    if tool_name == "read_px4_source_file":
        return "Reading PX4 source file" if started else "Read PX4 source file"
    return f"{action or 'Running'} tool: {tool_name}"


def status_label(status: str) -> str:
    return {
        "queued": "Queued",
        "running": "Running",
        "completed": "Complete",
        "failed": "Failed",
    }.get(status, status)


def resolve_artifact_path(path_value: str) -> Path:
    raw_path = Path(path_value)
    requested = raw_path.resolve() if raw_path.is_absolute() else (ROOT_DIR / raw_path).resolve()
    allowed_roots = (OUTPUT_ROOT.resolve(),)

    if not any(_is_relative_to(requested, root) for root in allowed_roots):
        raise ValueError("artifact path is not allowed")

    if not requested.is_file():
        raise ValueError("artifact does not exist")

    return requested


def browse_config_payload() -> dict[str, Any]:
    return {
        "browse_db_path": str(BROWSE_CONFIG.browse_db_path),
        "flight_review_storage_path": (
            str(BROWSE_CONFIG.flight_review_storage_path)
            if BROWSE_CONFIG.flight_review_storage_path
            else None
        ),
        "flight_review_db_path": (
            str(BROWSE_CONFIG.flight_review_db_path)
            if BROWSE_CONFIG.flight_review_db_path
            else None
        ),
        "flight_review_log_dir": (
            str(BROWSE_CONFIG.flight_review_log_dir)
            if BROWSE_CONFIG.flight_review_log_dir
            else None
        ),
        "airframe_image_root": (
            str(BROWSE_CONFIG.airframe_image_root)
            if BROWSE_CONFIG.airframe_image_root
            else None
        ),
    }


def index_browse_log(
    log_path: Path,
    *,
    source_kind: str,
    source_log_id: str | None = None,
    original_filename: str | None = None,
    review_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        airframes = load_browse_airframe_metadata(BROWSE_CONFIG)
        record = upsert_log_from_path(
            BROWSE_CONFIG.browse_db_path,
            log_path,
            upload_date=datetime.now(timezone.utc),
            source_kind=source_kind,
            source_log_id=source_log_id,
            original_filename=original_filename,
            airframes=airframes,
            airframe_image_root=resolve_airframe_image_root(BROWSE_CONFIG),
            review_inputs=review_inputs,
        )
    except Exception as exc:
        return {"indexed": False, "error": repr(exc)}
    try:
        row = get_log(BROWSE_CONFIG.browse_db_path, record["id"])
    except Exception:
        row = None
    return {"indexed": True, "log_id": record["id"], "row": row}


def _first_param(params: dict[str, list[str]], name: str) -> str:
    values = params.get(name) or []
    return values[0].strip() if values else ""


def _tag_params(params: dict[str, list[str]]) -> list[str]:
    tags: list[str] = []
    for value in params.get("tag", []) + params.get("tags", []):
        for tag in str(value or "").split(","):
            clean = tag.strip()
            if clean and clean not in tags:
                tags.append(clean)
    return tags


def _int_param(params: dict[str, list[str]], name: str, default: int) -> int:
    try:
        return int(_first_param(params, name) or default)
    except ValueError:
        return default


def _run_analysis_job(run_id: str, user_question: str | None = None) -> None:
    _update_analysis_run(run_id, status="running", started_at=time.time())
    try:
        from flight_log_agent.analysis.analyzer import analyze_flight_log

        with ANALYSIS_RUNS_LOCK:
            run = dict(ANALYSIS_RUNS[run_id])
        question = user_question or run["user_question"]

        report = asyncio.run(
            analyze_flight_log(
                log_path=run["log_path"],
                user_question=question,
                mission_path=run["mission_path"],
                source_path=run["source_path"],
                output_dir=run["output_dir"],
                dev_log_root=str(WEB_DEV_LOG_ROOT),
                dev_run_id=run_id,
            )
        )
        _update_analysis_run(
            run_id,
            status="completed",
            finished_at=time.time(),
            report=make_json_safe(report),
        )
    except Exception as exc:
        _update_analysis_run(
            run_id,
            status="failed",
            finished_at=time.time(),
            error=repr(exc),
            traceback=traceback.format_exc(),
        )


def _update_analysis_run(run_id: str, **fields: Any) -> None:
    with ANALYSIS_RUNS_LOCK:
        run = ANALYSIS_RUNS.get(run_id)
        if run is None:
            return
        run.update(fields)
        snapshot = dict(run)

    try:
        _persist_analysis_run_history(snapshot)
    except Exception as exc:
        with ANALYSIS_RUNS_LOCK:
            if run_id in ANALYSIS_RUNS:
                ANALYSIS_RUNS[run_id]["history_error"] = repr(exc)


def _persist_analysis_run_history(run: dict[str, Any]) -> None:
    history_dir = run.get("history_dir")
    if not history_dir:
        return
    write_history_record(
        history_dir,
        {
            "schema_version": 1,
            "run_id": run["run_id"],
            "browse_log_id": run.get("browse_log_id"),
            "status": run["status"],
            "question": run["user_question"],
            "created_at": run.get("created_at"),
            "started_at": run.get("started_at"),
            "finished_at": run.get("finished_at"),
            "output_dir": run["output_dir"],
            "report_path": run["report_path"],
            "report": run.get("report"),
            "error": run.get("error"),
        },
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False

    return True


def save_upload_form(form: cgi.FieldStorage, upload_root: Path = UPLOAD_ROOT) -> dict[str, Path]:
    run_dir = upload_root / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)

    saved_files: dict[str, Path] = {}
    for field_name, forced_suffix in UPLOAD_FIELDS.items():
        field = _form_file(form, field_name)
        if field is None:
            continue

        filename = safe_upload_filename(field.filename or field_name, forced_suffix)
        destination = run_dir / filename
        save_upload_file(field, destination)
        saved_files[field_name] = destination

    return saved_files


def save_upload_file(field: cgi.FieldStorage, destination: Path) -> None:
    with destination.open("wb") as output:
        while True:
            chunk = field.file.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)


def validate_ulog_file(path: Path) -> None:
    with path.open("rb") as file:
        header = file.read(len(ULog.HEADER_BYTES))
    if header != ULog.HEADER_BYTES:
        raise ValueError("uploaded log is not a valid ULog file")


def safe_upload_filename(filename: str, forced_suffix: str | None = None) -> str:
    name = Path(filename).name.strip().replace("\x00", "")
    if not name or name in {".", ".."}:
        name = "upload"

    safe_chars = []
    for char in name:
        if char.isalnum() or char in {"-", "_", "."}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    safe_name = "".join(safe_chars).strip("._") or "upload"
    if forced_suffix and not safe_name.lower().endswith(forced_suffix):
        safe_name = f"{safe_name}{forced_suffix}"
    return safe_name


def _form_file(form: cgi.FieldStorage, name: str) -> cgi.FieldStorage | None:
    if name not in form:
        return None
    field = form[name]
    if isinstance(field, list):
        field = field[0]
    if not getattr(field, "filename", None):
        return None
    return field


def _form_value(form: cgi.FieldStorage, name: str) -> str | None:
    if name not in form:
        return None
    field = form[name]
    if isinstance(field, list):
        field = field[0]
    value = getattr(field, "value", None)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    value = str(value or "").strip()
    return value or None


def start_ngrok_tunnel(
    port: int,
    *,
    ngrok_bin: str = "ngrok",
    timeout_s: float = 10.0,
) -> tuple[subprocess.Popen[bytes], str | None]:
    process = subprocess.Popen(
        [ngrok_bin, "http", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    public_url = wait_for_ngrok_url(timeout_s=timeout_s, process=process)
    return process, public_url


def wait_for_ngrok_url(
    *,
    timeout_s: float = 10.0,
    process: subprocess.Popen[bytes] | None = None,
) -> str | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            return None

        try:
            with urlopen("http://127.0.0.1:4040/api/tunnels", timeout=0.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception:
            time.sleep(0.25)
            continue

        public_url = public_ngrok_url(payload)
        if public_url:
            return public_url

        time.sleep(0.25)

    return None


def public_ngrok_url(payload: dict[str, Any]) -> str | None:
    for tunnel in payload.get("tunnels", []) or []:
        public_url = tunnel.get("public_url")
        if isinstance(public_url, str) and public_url.startswith("https://"):
            return public_url

    for tunnel in payload.get("tunnels", []) or []:
        public_url = tunnel.get("public_url")
        if isinstance(public_url, str) and public_url:
            return public_url

    return None


def main() -> None:
    global BROWSE_CONFIG

    parser = argparse.ArgumentParser(description="Run the flight-log preparse web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--ngrok", action="store_true", help="Expose the local UI through ngrok.")
    parser.add_argument("--ngrok-bin", default="ngrok", help="Path to the ngrok executable.")
    parser.add_argument("--browse-db-path", default=str(BROWSE_CONFIG.browse_db_path))
    parser.add_argument("--flight-review-storage-path", default=None)
    parser.add_argument("--flight-review-db-path", default=None)
    parser.add_argument("--flight-review-log-dir", default=None)
    parser.add_argument("--airframe-image-root", default=str(BROWSE_CONFIG.airframe_image_root or WEB_DIR / "airframes"))
    args = parser.parse_args()

    BROWSE_CONFIG = BrowseConfig(
        browse_db_path=Path(args.browse_db_path).expanduser(),
        flight_review_storage_path=_optional_cli_path(
            args.flight_review_storage_path,
            BROWSE_CONFIG.flight_review_storage_path,
        ),
        flight_review_db_path=_optional_cli_path(
            args.flight_review_db_path,
            BROWSE_CONFIG.flight_review_db_path,
        ),
        flight_review_log_dir=_optional_cli_path(
            args.flight_review_log_dir,
            BROWSE_CONFIG.flight_review_log_dir,
        ),
        airframe_image_root=Path(args.airframe_image_root).expanduser(),
    )
    ensure_browse_db(BROWSE_CONFIG.browse_db_path)
    refresh_airframe_image_keys(BROWSE_CONFIG)

    ngrok_process = None
    server = ThreadingHTTPServer((args.host, args.port), FlightLogWebHandler)
    print(f"Serving flight-log preparse UI at http://{args.host}:{args.port}")
    if args.ngrok:
        try:
            ngrok_process, public_url = start_ngrok_tunnel(
                args.port,
                ngrok_bin=args.ngrok_bin,
            )
        except FileNotFoundError:
            print(f"ngrok executable not found: {args.ngrok_bin}")
        else:
            if public_url:
                print(f"ngrok public URL: {public_url}")
            else:
                print("ngrok started, but no public URL was reported yet.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if ngrok_process is not None and ngrok_process.poll() is None:
            ngrok_process.terminate()
            try:
                ngrok_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ngrok_process.kill()


def _optional_cli_path(value: str | None, fallback: Path | None) -> Path | None:
    if value:
        return Path(value).expanduser()
    return fallback


if __name__ == "__main__":
    main()
