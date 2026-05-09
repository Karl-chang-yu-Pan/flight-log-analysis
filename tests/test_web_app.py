from io import BytesIO
from types import SimpleNamespace

import pytest
from pyulog import ULog

from web_app import (
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
