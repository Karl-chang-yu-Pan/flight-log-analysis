from __future__ import annotations

import sqlite3

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
