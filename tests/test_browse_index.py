from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from flight_log_agent.web import browse_index
from flight_log_agent.web.airframe_assets import (
    DEFAULT_AIRFRAME_IMAGE_KEY,
    load_airframe_metadata,
    resolve_airframe_image_key,
)
from flight_log_agent.web.browse_index import (
    BrowseConfig,
    add_log_tag,
    create_tag,
    get_log,
    import_flight_review,
    list_tags,
    load_browse_airframe_metadata,
    query_logs,
    refresh_airframe_image_keys,
    remove_log_tag,
    resolve_flight_review_db_path,
    resolve_flight_review_log_dir,
    sync_flight_review,
    upsert_log_from_path,
)


def test_flight_review_path_resolution_prefers_explicit_paths(tmp_path):
    storage = tmp_path / "storage"
    config = BrowseConfig(
        browse_db_path=tmp_path / "browse.sqlite",
        flight_review_storage_path=storage,
    )

    assert resolve_flight_review_db_path(config) == storage / "logs.sqlite"
    assert resolve_flight_review_log_dir(config) == storage / "log_files"

    explicit = BrowseConfig(
        browse_db_path=tmp_path / "browse.sqlite",
        flight_review_storage_path=storage,
        flight_review_db_path=tmp_path / "other.sqlite",
        flight_review_log_dir=tmp_path / "logs",
    )

    assert resolve_flight_review_db_path(explicit) == tmp_path / "other.sqlite"
    assert resolve_flight_review_log_dir(explicit) == tmp_path / "logs"


def test_import_flight_review_builds_browse_rows_and_searches_metadata(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)

    summary = import_flight_review(config)

    assert summary["imported"] == 2
    assert summary["skipped_missing_logs"] == 1

    result = query_logs(config.browse_db_path, search="pixhawk")
    assert result["filtered"] == 1
    row = result["rows"][0]
    assert row["id"] == "flight_review:log1"
    assert row["log_path"].endswith("log1.ulg")
    assert row["airframe_id"] == "4001"
    assert row["airframe_name"] == "Generic Quadrotor"
    assert row["airframe_group"] == "Multicopter"
    assert row["hardware"] == "Pixhawk"
    assert row["software_version"] == "v1.16.0 255"
    assert row["error_count"] == 2
    assert row["flight_modes"] == ["2", "5"]
    assert row["airframe_image_key"] == "QuadRotorX"
    assert set(row) == {
        "id",
        "source_kind",
        "source_log_id",
        "log_path",
        "original_filename",
        "upload_date",
        "log_date",
        "vehicle_type",
        "airframe_name",
        "airframe_group",
        "airframe_id",
        "airframe_image_key",
        "hardware",
        "software",
        "software_version",
        "duration_s",
        "error_count",
        "flight_modes",
        "tags",
        "review_inputs",
    }

    vtol_row = get_log(config.browse_db_path, "flight_review:log2")
    assert vtol_row["airframe_id"] == "13000"
    assert vtol_row["airframe_group"] == "Standard VTOL"
    assert vtol_row["airframe_image_key"] == "VTOLPlane"


def test_query_logs_combines_must_include_tags(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    import_flight_review(config)

    create_tag(config.browse_db_path, "reviewed")
    create_tag(config.browse_db_path, "regression")
    add_log_tag(config.browse_db_path, "flight_review:log1", "reviewed")
    add_log_tag(config.browse_db_path, "flight_review:log1", "regression")
    add_log_tag(config.browse_db_path, "flight_review:log2", "reviewed")

    result = query_logs(config.browse_db_path, tags=["reviewed", "regression"])

    assert [row["id"] for row in result["rows"]] == ["flight_review:log1"]
    assert list_tags(config.browse_db_path) == [
        {"name": "regression", "log_count": 1},
        {"name": "reviewed", "log_count": 2},
    ]

    remove_log_tag(config.browse_db_path, "flight_review:log1", "regression")
    result = query_logs(config.browse_db_path, tags=["reviewed", "regression"])
    assert result["filtered"] == 0
    assert list_tags(config.browse_db_path) == [
        {"name": "regression", "log_count": 0},
        {"name": "reviewed", "log_count": 2},
    ]


def test_get_log_returns_browse_row_with_tags(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    import_flight_review(config)
    add_log_tag(config.browse_db_path, "flight_review:log1", "reviewed")

    row = get_log(config.browse_db_path, "flight_review:log1")

    assert row["id"] == "flight_review:log1"
    assert row["tags"] == ["reviewed"]


def test_query_logs_filters_upload_and_log_dates_separately(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    import_flight_review(config)

    result = query_logs(
        config.browse_db_path,
        upload_start="2026-07-09",
        upload_end="2026-07-09",
        log_start="2026-07-08",
        log_end="2026-07-08",
    )
    assert [row["id"] for row in result["rows"]] == ["flight_review:log1"]

    result = query_logs(
        config.browse_db_path,
        upload_start="2026-07-09",
        upload_end="2026-07-09",
        log_start="2026-07-09",
        log_end="2026-07-09",
    )
    assert result["filtered"] == 0


def test_uploaded_log_uses_metadata_group_image(tmp_path, monkeypatch):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    log_path = tmp_path / "standard-vtol.ulg"
    log_path.write_bytes(b"test")

    monkeypatch.setattr(
        browse_index,
        "parse_ulog_inventory",
        lambda _path: {
            "airframe": {"id": 13000},
            "available_topics": [],
            "duration_s": 12.0,
            "firmware_version": "v1.16.0",
            "git_hash": "abc123",
            "logged_messages": [],
            "parameters": {"VT_TYPE": 2},
        },
    )
    monkeypatch.setattr(browse_index, "_read_ulog_info", lambda _path: {})
    monkeypatch.setattr(browse_index, "build_basic_timeline", lambda _path: [])
    monkeypatch.setattr(browse_index, "_extract_log_start_utc_s", lambda _path: None)

    record = upsert_log_from_path(
        config.browse_db_path,
        log_path,
        source_kind="upload",
        source_log_id="upload-id",
        airframes=load_browse_airframe_metadata(config),
        airframe_image_root=config.airframe_image_root,
        review_inputs={
            "mission_path": tmp_path / "mission.plan",
            "source_path": tmp_path / "PX4-Autopilot",
        },
    )

    assert record["airframe_name"] == "Generic Standard VTOL"
    assert record["airframe_group"] == "Standard VTOL"
    assert record["airframe_image_key"] == "VTOLPlane"
    stored = get_log(config.browse_db_path, "upload-id")
    assert stored["airframe_image_key"] == "VTOLPlane"
    assert stored["review_inputs"] == {
        "mission_path": str(tmp_path / "mission.plan"),
        "source_path": str(tmp_path / "PX4-Autopilot"),
    }


def test_missing_airframe_asset_uses_unknown_fallback(tmp_path):
    metadata_path = tmp_path / "airframes.xml"
    metadata_path.write_text(
        """
        <airframes>
          <airframe_group name="Standard VTOL" image="MissingVTOLImage">
            <airframe id="13000" name="Generic Standard VTOL" />
          </airframe_group>
        </airframes>
        """,
        encoding="utf-8",
    )
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "AirframeUnknown.svg").write_text("<svg />", encoding="utf-8")

    airframes = load_airframe_metadata([metadata_path])

    assert (
        resolve_airframe_image_key(13000, airframes, image_root)
        == DEFAULT_AIRFRAME_IMAGE_KEY
    )


def test_bundled_qgc_metadata_resolves_standard_vtol_asset(tmp_path):
    config = BrowseConfig(browse_db_path=tmp_path / "browse.sqlite")
    airframes = load_browse_airframe_metadata(config)

    assert airframes["13000"]["group"] == "Standard VTOL"
    assert airframes["13000"]["image"] == "VTOLPlane"
    assert resolve_airframe_image_key(
        13000,
        airframes,
        browse_index.resolve_airframe_image_root(config),
    ) == "VTOLPlane"


def test_refresh_airframe_image_keys_backfills_existing_rows(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    import_flight_review(config)
    with sqlite3.connect(config.browse_db_path) as con:
        con.execute(
            "UPDATE browse_logs SET airframe_image_key = ? WHERE id = ?",
            (DEFAULT_AIRFRAME_IMAGE_KEY, "flight_review:log2"),
        )
        con.commit()

    assert refresh_airframe_image_keys(config) == 1
    assert get_log(config.browse_db_path, "flight_review:log2")["airframe_image_key"] == "VTOLPlane"


def test_sync_flight_review_reports_first_and_unchanged_second_sync(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)

    first = sync_flight_review(config)
    with sqlite3.connect(config.browse_db_path) as con:
        first_updated_at = dict(
            con.execute("SELECT id, updated_at FROM browse_logs ORDER BY id")
        )

    second = sync_flight_review(config)
    with sqlite3.connect(config.browse_db_path) as con:
        second_updated_at = dict(
            con.execute("SELECT id, updated_at FROM browse_logs ORDER BY id")
        )

    assert {
        key: first[key]
        for key in (
            "imported",
            "added",
            "updated",
            "unchanged",
            "skipped_missing_logs",
            "newly_unavailable",
            "unavailable",
        )
    } == {
        "imported": 2,
        "added": 2,
        "updated": 0,
        "unchanged": 0,
        "skipped_missing_logs": 1,
        "newly_unavailable": 0,
        "unavailable": 0,
    }
    assert first["missing"] == first["skipped_missing_logs"] == 1
    assert {
        key: second[key]
        for key in (
            "imported",
            "added",
            "updated",
            "unchanged",
            "skipped_missing_logs",
            "newly_unavailable",
            "unavailable",
        )
    } == {
        "imported": 2,
        "added": 0,
        "updated": 0,
        "unchanged": 2,
        "skipped_missing_logs": 1,
        "newly_unavailable": 0,
        "unavailable": 0,
    }
    assert second["missing"] == second["skipped_missing_logs"] == 1
    assert second_updated_at == first_updated_at


def test_sync_flight_review_updates_metadata_and_preserves_tags(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    sync_flight_review(config)
    add_log_tag(config.browse_db_path, "flight_review:log1", "reviewed")
    with sqlite3.connect(config.browse_db_path) as con:
        created_at = con.execute(
            "SELECT created_at FROM browse_logs WHERE id = ?",
            ("flight_review:log1",),
        ).fetchone()[0]

    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute(
            "UPDATE LogsGenerated SET Hardware = ? WHERE Id = ?",
            ("Cube Orange", "log1"),
        )
        con.commit()

    summary = sync_flight_review(config)

    assert summary["updated"] == 1
    assert summary["unchanged"] == 1
    row = get_log(config.browse_db_path, "flight_review:log1")
    assert row["hardware"] == "Cube Orange"
    assert row["tags"] == ["reviewed"]
    with sqlite3.connect(config.browse_db_path) as con:
        assert con.execute(
            "SELECT created_at FROM browse_logs WHERE id = ?",
            ("flight_review:log1",),
        ).fetchone()[0] == created_at


def test_sync_soft_hides_removed_and_missing_logs_without_touching_uploads(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    sync_flight_review(config)
    _insert_minimal_browse_row(config.browse_db_path, "upload-id", "upload")
    for log_id in ("flight_review:log1", "flight_review:log2", "upload-id"):
        add_log_tag(config.browse_db_path, log_id, "reviewed")

    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute("DELETE FROM Logs WHERE Id = ?", ("log1",))
        con.commit()
    (storage / "log_files" / "log2.ulg").unlink()

    first = sync_flight_review(config)

    assert first["imported"] == 0
    assert first["skipped_missing_logs"] == 2
    assert first["newly_unavailable"] == 2
    assert first["unavailable"] == 2
    result = query_logs(config.browse_db_path)
    assert [row["id"] for row in result["rows"]] == ["upload-id"]
    assert result["total"] == result["filtered"] == 1
    assert list_tags(config.browse_db_path) == [{"name": "reviewed", "log_count": 1}]
    with sqlite3.connect(config.browse_db_path) as con:
        assert con.execute(
            "SELECT id, source_available FROM browse_logs ORDER BY id"
        ).fetchall() == [
            ("flight_review:log1", 0),
            ("flight_review:log2", 0),
            ("upload-id", 1),
        ]
        assert con.execute("SELECT COUNT(*) FROM log_tags").fetchone()[0] == 3

    second = sync_flight_review(config)

    assert second["newly_unavailable"] == 0
    assert second["unavailable"] == 2
    assert second["skipped_missing_logs"] == 2
    assert [row["id"] for row in query_logs(config.browse_db_path)["rows"]] == [
        "upload-id"
    ]


def test_sync_restores_available_row_without_losing_identity_or_tags(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    sync_flight_review(config)
    log_id = "flight_review:log1"
    add_log_tag(config.browse_db_path, log_id, "reviewed")
    with sqlite3.connect(config.browse_db_path) as con:
        created_at = con.execute(
            "SELECT created_at FROM browse_logs WHERE id = ?",
            (log_id,),
        ).fetchone()[0]

    log_path = storage / "log_files" / "log1.ulg"
    original_bytes = log_path.read_bytes()
    log_path.unlink()
    hidden = sync_flight_review(config)

    assert hidden["newly_unavailable"] == 1
    assert hidden["unavailable"] == 1
    assert query_logs(config.browse_db_path)["total"] == 1

    log_path.write_bytes(original_bytes)
    restored = sync_flight_review(config)

    assert restored["updated"] == 1
    assert restored["unchanged"] == 1
    assert restored["newly_unavailable"] == 0
    assert restored["unavailable"] == 0
    result = query_logs(config.browse_db_path)
    assert result["total"] == result["filtered"] == 2
    assert get_log(config.browse_db_path, log_id)["tags"] == ["reviewed"]
    with sqlite3.connect(config.browse_db_path) as con:
        assert con.execute(
            "SELECT created_at, source_available FROM browse_logs WHERE id = ?",
            (log_id,),
        ).fetchone() == (created_at, 1)


def test_flight_review_source_connection_is_read_only(tmp_path):
    storage = _flight_review_fixture(tmp_path)

    with browse_index._connect_flight_review_source(storage / "logs.sqlite") as con:
        assert con.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("UPDATE Logs SET Description = 'changed'")

    with sqlite3.connect(storage / "logs.sqlite") as con:
        assert con.execute(
            "SELECT Description FROM Logs WHERE Id = ?", ("log1",)
        ).fetchone()[0] == "first flight"


def test_ensure_browse_db_migrates_legacy_v0_schema_without_data_loss(tmp_path):
    db_path = tmp_path / "browse.sqlite"
    _legacy_browse_fixture(db_path)

    browse_index.ensure_browse_db(db_path)

    with sqlite3.connect(db_path) as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == 1
        columns = {
            row[1]: row for row in con.execute("PRAGMA table_info('browse_logs')")
        }
        assert columns["source_available"][3] == 1
        assert con.execute(
            "SELECT source_available FROM browse_logs WHERE id = ?",
            ("legacy-upload",),
        ).fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM log_tags").fetchone()[0] == 1

    assert [row["id"] for row in query_logs(db_path)["rows"]] == ["legacy-upload"]
    assert list_tags(db_path) == [{"name": "legacy", "log_count": 1}]


@pytest.mark.parametrize("generated_table", [False, True])
def test_sync_accepts_missing_or_older_logs_generated_schema(
    tmp_path,
    generated_table,
):
    storage = _minimal_flight_review_fixture(
        tmp_path,
        generated_table=generated_table,
    )
    config = _browse_config(tmp_path, storage)

    summary = sync_flight_review(config)

    assert summary["added"] == 1
    row = get_log(config.browse_db_path, "flight_review:minimal")
    assert row["original_filename"] == "minimal.ulg"
    assert row["hardware"] is None
    assert row["software_version"] is None
    assert row["duration_s"] == (12.0 if generated_table else None)


def test_sync_rejects_flight_review_log_id_that_can_escape_log_directory(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    (storage / "escape.ulg").write_bytes(b"outside")
    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute(
            "INSERT INTO Logs VALUES (?, ?, ?, ?)",
            ("../escape", "2027-01-01 00:00:00", "escape.ulg", "invalid"),
        )
        con.commit()

    with pytest.raises(ValueError, match="invalid Flight Review log id"):
        sync_flight_review(config)

    with sqlite3.connect(config.browse_db_path) as con:
        assert con.execute("SELECT COUNT(*) FROM browse_logs").fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM browse_sync_sources"
        ).fetchone()[0] == 0


def test_sync_rejects_using_source_database_as_destination(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    source_db = storage / "logs.sqlite"
    config = BrowseConfig(
        browse_db_path=source_db,
        flight_review_storage_path=storage,
    )

    with pytest.raises(ValueError, match="source database must differ"):
        sync_flight_review(config)

    with sqlite3.connect(source_db) as con:
        assert {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } == {"Logs", "LogsGenerated"}


def test_sync_rejects_hard_linked_source_database_as_destination(tmp_path):
    storage = _flight_review_fixture(tmp_path)
    source_db = storage / "logs.sqlite"
    browse_db = tmp_path / "browse-hardlink.sqlite"
    browse_db.hardlink_to(source_db)
    config = BrowseConfig(
        browse_db_path=browse_db,
        flight_review_storage_path=storage,
    )

    with pytest.raises(ValueError, match="source database must differ"):
        sync_flight_review(config)

    with sqlite3.connect(source_db) as con:
        assert {
            row[0]
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        } == {"Logs", "LogsGenerated"}


def test_sync_rejects_switching_browse_database_to_another_source(tmp_path):
    first_storage = _flight_review_fixture(tmp_path / "first")
    second_storage = _flight_review_fixture(tmp_path / "second")
    first_config = _browse_config(tmp_path, first_storage)
    second_config = _browse_config(tmp_path, second_storage)
    sync_flight_review(first_config)

    with pytest.raises(ValueError, match="different Flight Review source"):
        sync_flight_review(second_config)

    assert get_log(
        first_config.browse_db_path,
        "flight_review:log1",
    )["log_path"].startswith(str(first_storage))


def test_sync_rejects_rebinding_unbound_existing_flight_review_rows(tmp_path):
    first_storage = _flight_review_fixture(tmp_path / "first")
    second_storage = _flight_review_fixture(tmp_path / "second")
    first_config = _browse_config(tmp_path, first_storage)
    second_config = _browse_config(tmp_path, second_storage)
    sync_flight_review(first_config)
    add_log_tag(first_config.browse_db_path, "flight_review:log1", "reviewed")
    with sqlite3.connect(first_config.browse_db_path) as con:
        con.execute("DELETE FROM browse_sync_sources")
        before = con.execute(
            "SELECT log_path, hardware FROM browse_logs WHERE id = ?",
            ("flight_review:log1",),
        ).fetchone()
        con.commit()

    with pytest.raises(ValueError, match="different Flight Review source"):
        sync_flight_review(second_config)

    with sqlite3.connect(first_config.browse_db_path) as con:
        assert con.execute(
            "SELECT log_path, hardware FROM browse_logs WHERE id = ?",
            ("flight_review:log1",),
        ).fetchone() == before
        assert con.execute("SELECT COUNT(*) FROM browse_sync_sources").fetchone()[0] == 0
    assert get_log(
        first_config.browse_db_path,
        "flight_review:log1",
    )["tags"] == ["reviewed"]


def test_sync_accepts_case_insensitive_flight_review_column_names(tmp_path):
    storage = tmp_path / "lowercase"
    log_dir = storage / "log_files"
    log_dir.mkdir(parents=True)
    (log_dir / "lowercase.ulg").write_bytes(b"minimal")
    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute("CREATE TABLE logs(id TEXT PRIMARY KEY, date TEXT)")
        con.execute(
            "CREATE TABLE logsgenerated(id TEXT PRIMARY KEY, duration INT)"
        )
        con.execute(
            "INSERT INTO logs(id, date) VALUES ('lowercase', '2026-01-01')"
        )
        con.execute(
            "INSERT INTO logsgenerated(id, duration) VALUES ('lowercase', 12)"
        )
        con.commit()
    config = _browse_config(tmp_path, storage)

    summary = sync_flight_review(config)

    assert summary["added"] == 1
    assert get_log(
        config.browse_db_path,
        "flight_review:lowercase",
    )["duration_s"] == 12.0


def test_sync_response_preserves_configured_path_values(tmp_path, monkeypatch):
    _flight_review_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = BrowseConfig(
        browse_db_path=Path("browse.sqlite"),
        flight_review_storage_path=Path("flight_review"),
    )

    summary = sync_flight_review(config)

    assert summary["source_db"] == "flight_review/logs.sqlite"
    assert summary["log_dir"] == "flight_review/log_files"
    assert summary["browse_db"] == "browse.sqlite"


def test_sync_rolls_back_partial_upsert_and_does_not_sweep_on_failure(
    tmp_path,
    monkeypatch,
):
    storage = _flight_review_fixture(tmp_path)
    config = _browse_config(tmp_path, storage)
    sync_flight_review(config)
    add_log_tag(config.browse_db_path, "flight_review:log1", "reviewed")
    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute("DELETE FROM Logs WHERE Id = ?", ("log1",))
        con.execute(
            "UPDATE LogsGenerated SET Hardware = ? WHERE Id = ?",
            ("Changed after initial sync", "log2"),
        )
        con.commit()

    original_upsert = browse_index._upsert_log

    def failing_upsert(con, record):
        original_upsert(con, record)
        raise RuntimeError("injected sync failure")

    monkeypatch.setattr(browse_index, "_upsert_log", failing_upsert)

    with pytest.raises(RuntimeError, match="injected sync failure"):
        sync_flight_review(config)

    with sqlite3.connect(config.browse_db_path) as con:
        assert con.execute(
            "SELECT hardware FROM browse_logs WHERE id = ?",
            ("flight_review:log2",),
        ).fetchone()[0] == "FMU"
        assert con.execute(
            "SELECT source_available FROM browse_logs WHERE id = ?",
            ("flight_review:log1",),
        ).fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM log_tags").fetchone()[0] == 1


def _insert_minimal_browse_row(db_path, log_id, source_kind):
    record = {
        "id": log_id,
        "source_kind": source_kind,
        "source_log_id": log_id,
        "log_path": f"/logs/{log_id}.ulg",
        "flight_modes": [],
        "metadata": {},
    }
    with browse_index._connect(db_path) as con:
        browse_index._ensure_schema(con)
        browse_index._upsert_log(con, record)
        con.commit()


def _legacy_browse_fixture(db_path):
    with sqlite3.connect(db_path) as con:
        con.executescript(
            """
            CREATE TABLE browse_logs(
                id TEXT PRIMARY KEY,
                source_kind TEXT NOT NULL,
                source_log_id TEXT,
                log_path TEXT NOT NULL,
                original_filename TEXT,
                upload_date TEXT,
                log_date TEXT,
                vehicle_type TEXT,
                airframe_name TEXT,
                airframe_group TEXT,
                airframe_id TEXT,
                airframe_image_key TEXT,
                hardware TEXT,
                software TEXT,
                software_version TEXT,
                duration_s REAL,
                error_count INTEGER DEFAULT 0,
                flight_modes_json TEXT NOT NULL DEFAULT '[]',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                search_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE tags(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE log_tags(
                log_id TEXT NOT NULL,
                tag_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(log_id, tag_id),
                FOREIGN KEY(log_id) REFERENCES browse_logs(id) ON DELETE CASCADE,
                FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
            );
            INSERT INTO browse_logs(
                id, source_kind, source_log_id, log_path, original_filename,
                flight_modes_json, metadata_json, search_text, created_at, updated_at
            ) VALUES (
                'legacy-upload', 'upload', 'legacy-upload', '/logs/legacy.ulg',
                'legacy.ulg', '[]', '{}', 'legacy upload',
                '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z'
            );
            INSERT INTO tags(name, created_at)
            VALUES ('legacy', '2026-01-01T00:00:00Z');
            INSERT INTO log_tags(log_id, tag_id, created_at)
            VALUES ('legacy-upload', 1, '2026-01-01T00:00:00Z');
            PRAGMA user_version=0;
            """
        )


def _minimal_flight_review_fixture(tmp_path, *, generated_table):
    storage = tmp_path / (
        "minimal_with_generated" if generated_table else "minimal_without_generated"
    )
    log_dir = storage / "log_files"
    log_dir.mkdir(parents=True)
    (log_dir / "minimal.ulg").write_bytes(b"minimal")
    with sqlite3.connect(storage / "logs.sqlite") as con:
        con.execute("CREATE TABLE Logs(Id TEXT PRIMARY KEY)")
        con.execute("INSERT INTO Logs(Id) VALUES ('minimal')")
        if generated_table:
            con.execute(
                "CREATE TABLE LogsGenerated(Id TEXT PRIMARY KEY, Duration INT)"
            )
            con.execute(
                "INSERT INTO LogsGenerated(Id, Duration) VALUES ('minimal', 12)"
            )
        con.commit()
    return storage


def _browse_config(tmp_path, storage):
    return BrowseConfig(
        browse_db_path=tmp_path / "browse.sqlite",
        flight_review_storage_path=storage,
        airframe_image_root=storage / "airframes",
    )


def _flight_review_fixture(tmp_path):
    storage = tmp_path / "flight_review"
    log_dir = storage / "log_files"
    log_dir.mkdir(parents=True)
    cache_dir = storage / "cache"
    cache_dir.mkdir()
    image_dir = storage / "airframes"
    image_dir.mkdir()
    (log_dir / "log1.ulg").write_bytes(b"ulog-one")
    (log_dir / "log2.ulg").write_bytes(b"ulog-two")
    for image_name in ("AirframeUnknown", "QuadRotorX", "VTOLPlane"):
        (image_dir / f"{image_name}.svg").write_text("<svg />", encoding="utf-8")
    (cache_dir / "airframes.xml").write_text(
        """
        <airframes>
          <airframe_group name="Multicopter" image="QuadRotorX">
            <airframe id="4001" name="Generic Quadrotor">
              <type>Quadrotor</type>
            </airframe>
          </airframe_group>
          <airframe_group name="Standard VTOL" image="VTOLPlane">
            <airframe id="13000" name="Generic Standard VTOL">
              <type>Standard VTOL</type>
            </airframe>
          </airframe_group>
        </airframes>
        """,
        encoding="utf-8",
    )

    con = sqlite3.connect(storage / "logs.sqlite")
    con.executescript(
        """
        CREATE TABLE Logs(
            Id TEXT PRIMARY KEY,
            Date TIMESTAMP,
            OriginalFilename TEXT,
            Description TEXT
        );
        CREATE TABLE LogsGenerated(
            Id TEXT PRIMARY KEY,
            Duration INT,
            MavType TEXT,
            AutostartId INT,
            Hardware TEXT,
            Software TEXT,
            NumLoggedErrors INT,
            FlightModes TEXT,
            SoftwareVersion TEXT,
            StartTime INT,
            FlightModeDurations TEXT,
            UUID TEXT
        );
        """
    )
    con.executemany(
        "INSERT INTO Logs VALUES (?, ?, ?, ?)",
        [
            ("log1", "2026-07-09 10:00:00", "one.ulg", "first flight"),
            ("log2", "2026-07-10 10:00:00", "two.ulg", "second flight"),
            ("missing", "2026-07-11 10:00:00", "missing.ulg", "missing file"),
        ],
    )
    con.executemany(
        "INSERT INTO LogsGenerated VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (
                "log1",
                125,
                "Quadrotor",
                4001,
                "Pixhawk",
                "abcdef1234",
                2,
                "2,5",
                "v1.16.0 255",
                1783497600,
                "2:30,5:95",
                "uuid-1",
            ),
            (
                "log2",
                90,
                "VTOL",
                13000,
                "FMU",
                "1234567890",
                0,
                "3",
                "v1.15.0 255",
                1783584000,
                "3:90",
                "uuid-2",
            ),
            (
                "missing",
                45,
                "Rover",
                5001,
                "FMU",
                "111111",
                0,
                "4",
                "v1.14.0 255",
                1783670400,
                "4:45",
                "uuid-3",
            ),
        ],
    )
    con.commit()
    con.close()
    return storage
