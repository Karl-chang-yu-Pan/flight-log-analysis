import json
from pathlib import Path

from flight_log_agent.web.analysis_history import (
    load_history_records,
    resolve_history_directory,
    write_history_record,
)


def test_uploaded_log_history_is_stored_in_its_upload_directory(tmp_path):
    upload_root = tmp_path / "uploads"
    log_path = upload_root / "upload-123" / "flight.ulg"
    log_path.parent.mkdir(parents=True)
    log_path.write_bytes(b"ulog")

    history_dir = resolve_history_directory(
        log_path,
        upload_root=upload_root,
        fallback_root=tmp_path / "outputs" / "analysis_history",
        history_key="browse-id",
    )

    assert history_dir == log_path.parent / "analysis_history"


def test_local_log_history_uses_deterministic_output_fallback(tmp_path):
    log_path = tmp_path / "local-logs" / "flight.ulg"
    fallback_root = tmp_path / "outputs" / "analysis_history"
    log_path.parent.mkdir()
    log_path.write_bytes(b"ulog")

    first = resolve_history_directory(
        log_path,
        upload_root=tmp_path / "uploads",
        fallback_root=fallback_root,
        history_key="browse-id",
    )
    second = resolve_history_directory(
        tmp_path / "another-path.ulg",
        upload_root=tmp_path / "uploads",
        fallback_root=fallback_root,
        history_key="browse-id",
    )

    assert first == second
    assert first.parent == fallback_root
    assert first != log_path.parent


def test_history_records_keep_exact_question_and_full_report(tmp_path):
    history_dir = tmp_path / "history"
    earlier = {
        "schema_version": 1,
        "run_id": "run-earlier",
        "status": "completed",
        "question": "What caused the airspeed response?",
        "created_at": 10.0,
        "report": {
            "final_summary": "The response followed the measured load factor.",
            "ranked_hypotheses": [{"title": "Load factor"}],
        },
    }
    later = {
        "schema_version": 1,
        "run_id": "run-later",
        "status": "queued",
        "question": "Why did RTL start?",
        "created_at": 20.0,
        "report": None,
    }

    write_history_record(history_dir, later)
    write_history_record(history_dir, earlier)
    (history_dir / "malformed.json").write_text("{", encoding="utf-8")

    records = load_history_records(history_dir)

    assert [record["run_id"] for record in records] == ["run-earlier", "run-later"]
    assert records[0]["question"] == earlier["question"]
    assert records[0]["report"] == earlier["report"]


def test_updating_a_run_atomically_replaces_its_history_record(tmp_path):
    history_dir = tmp_path / "history"
    queued = {
        "schema_version": 1,
        "run_id": "run-123",
        "status": "queued",
        "question": "What happened?",
        "created_at": 10.0,
        "report": None,
    }
    completed = {
        **queued,
        "status": "completed",
        "finished_at": 15.0,
        "report": {"final_summary": "A complete answer."},
    }

    record_path = write_history_record(history_dir, queued)
    write_history_record(history_dir, completed)

    assert json.loads(record_path.read_text(encoding="utf-8")) == completed
    assert load_history_records(history_dir) == [completed]
    assert list(history_dir.glob("*.tmp")) == []
