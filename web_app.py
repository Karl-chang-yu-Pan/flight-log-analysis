from __future__ import annotations

import argparse
import cgi
import json
import mimetypes
import subprocess
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import urlopen

from pyulog import ULog
from preparse_view import build_preparse_payload


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
UPLOAD_ROOT = ROOT_DIR / "uploads"
MAX_UPLOAD_BYTES = 250 * 1024 * 1024
UPLOAD_FIELDS = {
    "log_file": ".ulg",
    "mission_file": None,
    "parameters_xml_file": ".xml",
}


class FlightLogWebHandler(BaseHTTPRequestHandler):
    server_version = "FlightLogPreparseHTTP/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._serve_file(WEB_DIR / "index.html")
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

        self._send_json({"error": "not found"}, status=404)

    def _handle_preparse_json(self) -> None:
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
            result = build_preparse_payload(
                log_path,
                mission_path=_optional_payload_path(payload, "mission_path"),
                source_path=_optional_payload_path(payload, "source_path"),
                parameters_xml_path=_optional_payload_path(payload, "parameters_xml_path"),
            )
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return

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
        self._send_json(result)

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
    parser = argparse.ArgumentParser(description="Run the flight-log preparse web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--ngrok", action="store_true", help="Expose the local UI through ngrok.")
    parser.add_argument("--ngrok-bin", default="ngrok", help="Path to the ngrok executable.")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
