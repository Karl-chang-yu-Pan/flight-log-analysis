from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from pyulog import ULog

from flight_log_agent.ulog.inventory import parse_ulog_inventory
from flight_log_agent.ulog.timeline import build_basic_timeline
from flight_log_agent.web.airframe_assets import (
    DEFAULT_AIRFRAME_IMAGE_KEY,
    load_airframe_metadata,
    metadata_for_airframe,
    resolve_airframe_image_key,
)


ERROR_LEVELS = {"EMERGENCY", "ALERT", "CRITICAL", "ERROR"}
DEFAULT_LIMIT = 50
MAX_LIMIT = 500
BROWSE_SCHEMA_VERSION = 1
DEFAULT_AIRFRAME_ASSET_ROOT = Path(__file__).resolve().parents[2] / "web" / "airframes"
BUNDLED_AIRFRAME_METADATA_PATH = DEFAULT_AIRFRAME_ASSET_ROOT / "AirframeFactMetaData.xml"
FLIGHT_REVIEW_LOG_COLUMNS = ("Date", "OriginalFilename", "Description")
FLIGHT_REVIEW_GENERATED_COLUMNS = (
    "Duration",
    "MavType",
    "AutostartId",
    "Hardware",
    "Software",
    "NumLoggedErrors",
    "FlightModes",
    "SoftwareVersion",
    "StartTime",
    "FlightModeDurations",
    "UUID",
)


@dataclass(frozen=True)
class BrowseConfig:
    browse_db_path: Path
    flight_review_storage_path: Path | None = None
    flight_review_db_path: Path | None = None
    flight_review_log_dir: Path | None = None
    airframe_image_root: Path | None = None


def with_flight_review_storage_path(
    config: BrowseConfig,
    storage_path: str | Path,
) -> BrowseConfig:
    clean_path = str(storage_path).strip()
    if not clean_path:
        raise ValueError("flight_review_storage_path must not be empty")
    return BrowseConfig(
        browse_db_path=config.browse_db_path,
        flight_review_storage_path=Path(clean_path).expanduser(),
        flight_review_db_path=None,
        flight_review_log_dir=None,
        airframe_image_root=config.airframe_image_root,
    )


def resolve_flight_review_db_path(config: BrowseConfig) -> Path | None:
    if config.flight_review_db_path is not None:
        return config.flight_review_db_path
    if config.flight_review_storage_path is not None:
        return config.flight_review_storage_path / "logs.sqlite"
    return None


def resolve_flight_review_log_dir(config: BrowseConfig) -> Path | None:
    if config.flight_review_log_dir is not None:
        return config.flight_review_log_dir
    if config.flight_review_storage_path is not None:
        return config.flight_review_storage_path / "log_files"
    return None


def resolve_airframe_image_root(config: BrowseConfig) -> Path:
    return config.airframe_image_root or DEFAULT_AIRFRAME_ASSET_ROOT


def resolve_airframe_metadata_paths(config: BrowseConfig) -> list[Path]:
    paths = [BUNDLED_AIRFRAME_METADATA_PATH]
    if config.flight_review_storage_path is not None:
        paths.append(config.flight_review_storage_path / "cache" / "airframes.xml")
    return paths


def load_browse_airframe_metadata(config: BrowseConfig) -> dict[str, dict[str, str]]:
    return load_airframe_metadata(resolve_airframe_metadata_paths(config))


def ensure_browse_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as con:
        _ensure_schema(con)


def refresh_airframe_image_keys(config: BrowseConfig) -> int:
    airframes = load_browse_airframe_metadata(config)
    image_root = resolve_airframe_image_root(config)
    now = _iso_utc(datetime.now(timezone.utc))
    updates: list[tuple[str, str, str]] = []

    with _connect(config.browse_db_path) as con:
        _ensure_schema(con)
        rows = con.execute(
            "SELECT id, airframe_id, airframe_image_key FROM browse_logs"
        ).fetchall()
        for log_id, airframe_id, current_key in rows:
            resolved_key = resolve_airframe_image_key(airframe_id, airframes, image_root)
            if resolved_key == DEFAULT_AIRFRAME_IMAGE_KEY and current_key:
                continue
            if resolved_key != current_key:
                updates.append((resolved_key, now, log_id))

        con.executemany(
            "UPDATE browse_logs SET airframe_image_key = ?, updated_at = ? WHERE id = ?",
            updates,
        )
        con.commit()

    return len(updates)


def upsert_log_from_path(
    db_path: Path,
    log_path: Path,
    *,
    upload_date: datetime | None = None,
    source_kind: str = "local",
    source_log_id: str | None = None,
    original_filename: str | None = None,
    airframes: Mapping[str, Mapping[str, str]] | None = None,
    airframe_image_root: Path | None = None,
    review_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record = build_log_record_from_path(
        log_path,
        upload_date=upload_date,
        source_kind=source_kind,
        source_log_id=source_log_id,
        original_filename=original_filename,
        airframes=airframes,
        airframe_image_root=airframe_image_root,
        review_inputs=review_inputs,
    )
    with _connect(db_path) as con:
        _ensure_schema(con)
        _upsert_log(con, record)
        con.commit()
    return record


def sync_flight_review(config: BrowseConfig) -> dict[str, Any]:
    source_db = resolve_flight_review_db_path(config)
    log_dir = resolve_flight_review_log_dir(config)
    if source_db is None:
        raise ValueError("flight_review_db_path or flight_review_storage_path is required")
    if log_dir is None:
        raise ValueError("flight_review_log_dir or flight_review_storage_path is required")

    source_db_path = source_db.expanduser().resolve()
    log_dir_path = log_dir.expanduser().resolve()
    browse_db_path = config.browse_db_path.expanduser().resolve()
    if not source_db_path.is_file():
        raise ValueError(f"Flight Review database does not exist: {source_db_path}")
    if not log_dir_path.is_dir():
        raise ValueError(f"Flight Review log directory does not exist: {log_dir_path}")
    if _paths_refer_to_same_file(source_db_path, browse_db_path):
        raise ValueError("Flight Review source database must differ from browse database")

    skipped_missing_logs = 0
    airframes = load_browse_airframe_metadata(config)
    airframe_image_root = resolve_airframe_image_root(config)
    counts = {"added": 0, "updated": 0, "unchanged": 0}
    with _connect(browse_db_path) as dest_con:
        _ensure_schema(dest_con)
        dest_con.execute("BEGIN IMMEDIATE")
        try:
            _claim_flight_review_source(dest_con, source_db_path, log_dir_path)
            with _connect_flight_review_source(source_db_path) as source_con:
                source_con.execute("BEGIN")
                rows = _flight_review_rows(source_con)
                source_con.rollback()

            records: list[dict[str, Any]] = []
            for row in rows:
                log_id = _flight_review_log_id(row["Id"])
                if log_id is None:
                    continue
                log_path = _flight_review_log_path(log_dir_path, log_id)
                if not log_path.is_file():
                    skipped_missing_logs += 1
                    continue
                records.append(
                    _record_from_flight_review_row(
                        row,
                        log_path,
                        airframes,
                        airframe_image_root,
                        log_id=log_id,
                    )
                )

            dest_con.execute(
                "CREATE TEMP TABLE flight_review_sync_seen(id TEXT PRIMARY KEY)"
            )
            for record in records:
                dest_con.execute(
                    "INSERT INTO flight_review_sync_seen(id) VALUES (?)",
                    (record["id"],),
                )
                outcome = _upsert_log(dest_con, record)
                counts[outcome] += 1

            now = _iso_utc(datetime.now(timezone.utc))
            newly_unavailable = dest_con.execute(
                """
                UPDATE browse_logs
                SET source_available = 0, updated_at = ?
                WHERE source_kind = 'flight_review'
                  AND source_available = 1
                  AND NOT EXISTS (
                      SELECT 1
                      FROM flight_review_sync_seen
                      WHERE flight_review_sync_seen.id = browse_logs.id
                  )
                """,
                (now,),
            ).rowcount
            unavailable = dest_con.execute(
                """
                SELECT COUNT(*)
                FROM browse_logs
                WHERE source_kind = 'flight_review'
                  AND source_available = 0
                """
            ).fetchone()[0]
            dest_con.commit()
        except Exception:
            dest_con.rollback()
            raise

    return {
        "imported": len(records),
        **counts,
        "missing": skipped_missing_logs,
        "skipped_missing_logs": skipped_missing_logs,
        "newly_unavailable": newly_unavailable,
        "unavailable": unavailable,
        "source_db": str(source_db),
        "log_dir": str(log_dir),
        "browse_db": str(config.browse_db_path),
    }


def import_flight_review(config: BrowseConfig) -> dict[str, Any]:
    """Compatibility wrapper for the original manual import entry point."""
    return sync_flight_review(config)


def query_logs(
    db_path: Path,
    *,
    search: str = "",
    tags: Iterable[str] = (),
    upload_start: str = "",
    upload_end: str = "",
    log_start: str = "",
    log_end: str = "",
    sort: str = "upload_date",
    direction: str = "desc",
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> dict[str, Any]:
    ensure_browse_db(db_path)
    clean_tags = [_clean_tag(tag) for tag in tags if _clean_tag(tag)]
    limit = max(1, min(int(limit or DEFAULT_LIMIT), MAX_LIMIT))
    offset = max(0, int(offset or 0))

    where, params = _build_filter_clause(
        search=search,
        tags=clean_tags,
        upload_start=upload_start,
        upload_end=upload_end,
        log_start=log_start,
        log_end=log_end,
    )
    order_by = _order_by(sort, direction)

    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        total = con.execute(
            "SELECT COUNT(*) FROM browse_logs WHERE source_available = 1"
        ).fetchone()[0]
        filtered = con.execute(f"SELECT COUNT(*) FROM browse_logs {where}", params).fetchone()[0]
        rows = con.execute(
            f"""
            SELECT *
            FROM browse_logs
            {where}
            {order_by}
            LIMIT ? OFFSET ?
            """,
            [*params, limit, offset],
        ).fetchall()
        row_payloads = [_row_payload(con, row) for row in rows]

    return {
        "total": total,
        "filtered": filtered,
        "limit": limit,
        "offset": offset,
        "rows": row_payloads,
    }


def get_log(db_path: Path, log_id: str) -> dict[str, Any]:
    clean_log_id = str(log_id or "").strip()
    if not clean_log_id:
        raise ValueError("log_id is required")
    ensure_browse_db(db_path)
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM browse_logs WHERE id = ?", (clean_log_id,)).fetchone()
        if row is None:
            raise ValueError("log not found")
        return _row_payload(con, row)


def list_tags(db_path: Path) -> list[dict[str, Any]]:
    ensure_browse_db(db_path)
    with _connect(db_path) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute(
            """
            SELECT
                tags.name,
                COUNT(
                    CASE WHEN browse_logs.source_available = 1
                    THEN log_tags.log_id END
                ) AS log_count
            FROM tags
            LEFT JOIN log_tags ON log_tags.tag_id = tags.id
            LEFT JOIN browse_logs ON browse_logs.id = log_tags.log_id
            GROUP BY tags.id
            ORDER BY tags.name COLLATE NOCASE
            """
        ).fetchall()
    return [{"name": row["name"], "log_count": row["log_count"]} for row in rows]


def create_tag(db_path: Path, name: str) -> dict[str, Any]:
    clean_name = _clean_tag(name)
    if not clean_name:
        raise ValueError("tag name is required")
    with _connect(db_path) as con:
        _ensure_schema(con)
        tag_id = _ensure_tag(con, clean_name)
        con.commit()
    return {"id": tag_id, "name": clean_name}


def add_log_tag(db_path: Path, log_id: str, tag_name: str) -> None:
    clean_log_id = str(log_id or "").strip()
    clean_tag = _clean_tag(tag_name)
    if not clean_log_id:
        raise ValueError("log_id is required")
    if not clean_tag:
        raise ValueError("tag name is required")
    with _connect(db_path) as con:
        _ensure_schema(con)
        if not _log_exists(con, clean_log_id):
            raise ValueError("log not found")
        tag_id = _ensure_tag(con, clean_tag)
        con.execute(
            "INSERT OR IGNORE INTO log_tags(log_id, tag_id, created_at) VALUES (?, ?, ?)",
            (clean_log_id, tag_id, _iso_utc(datetime.now(timezone.utc))),
        )
        con.commit()


def remove_log_tag(db_path: Path, log_id: str, tag_name: str) -> None:
    clean_log_id = str(log_id or "").strip()
    clean_tag = _clean_tag(tag_name)
    if not clean_log_id:
        raise ValueError("log_id is required")
    if not clean_tag:
        raise ValueError("tag name is required")
    with _connect(db_path) as con:
        _ensure_schema(con)
        con.execute(
            """
            DELETE FROM log_tags
            WHERE log_id = ?
              AND tag_id IN (SELECT id FROM tags WHERE name = ? COLLATE NOCASE)
            """,
            (clean_log_id, clean_tag),
        )
        con.commit()


def build_log_record_from_path(
    log_path: Path,
    *,
    upload_date: datetime | None = None,
    source_kind: str = "local",
    source_log_id: str | None = None,
    original_filename: str | None = None,
    airframes: Mapping[str, Mapping[str, str]] | None = None,
    airframe_image_root: Path | None = None,
    review_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    log_path = Path(log_path)
    inventory = parse_ulog_inventory(log_path)
    ulog_info = _read_ulog_info(log_path)
    timeline = build_basic_timeline(log_path)

    log_id = source_log_id or _local_log_id(log_path)
    airframe = inventory.get("airframe") or {}
    airframe_metadata = metadata_for_airframe(airframe.get("id"), airframes or {})
    flight_modes = _flight_modes_from_timeline(timeline)
    hardware = _first_clean(
        ulog_info.get("ver_hw"),
        ulog_info.get("sys_hw"),
        ulog_info.get("board"),
    )
    software = _first_clean(
        inventory.get("firmware_version"),
        inventory.get("git_hash"),
        ulog_info.get("ver_sw"),
    )

    record_metadata = {
        "inventory": inventory,
        "ulog_info": ulog_info,
    }
    normalized_review_inputs = _normalize_review_inputs(review_inputs)
    if normalized_review_inputs:
        record_metadata["review_inputs"] = normalized_review_inputs

    record = {
        "id": log_id,
        "source_kind": source_kind,
        "source_log_id": source_log_id,
        "log_path": str(log_path),
        "original_filename": original_filename or log_path.name,
        "upload_date": _iso_utc(upload_date or datetime.now(timezone.utc)),
        "log_date": _iso_from_timestamp(_extract_log_start_utc_s(log_path)),
        "vehicle_type": _first_clean(_vehicle_type_from_inventory(inventory)),
        "airframe_name": _first_clean(airframe.get("name"), airframe_metadata.get("name")),
        "airframe_group": _first_clean(
            airframe.get("type"),
            airframe.get("class"),
            airframe_metadata.get("group"),
            airframe_metadata.get("type"),
        ),
        "airframe_id": _clean_optional(airframe.get("id")),
        "airframe_image_key": resolve_airframe_image_key(
            airframe.get("id"),
            airframes or {},
            airframe_image_root,
        ),
        "hardware": hardware,
        "software": software,
        "software_version": _clean_optional(inventory.get("firmware_version")),
        "duration_s": inventory.get("duration_s"),
        "error_count": _error_count(inventory.get("logged_messages") or []),
        "flight_modes": flight_modes,
        "metadata": record_metadata,
    }
    record["search_text"] = _build_search_text(record)
    return record


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, timeout=30)
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=30000")
    return con


def _ensure_schema(con: sqlite3.Connection) -> None:
    schema_version = int(con.execute("PRAGMA user_version").fetchone()[0])
    if schema_version > BROWSE_SCHEMA_VERSION:
        raise ValueError(
            "browse database schema is newer than this application "
            f"({schema_version} > {BROWSE_SCHEMA_VERSION})"
        )

    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS browse_logs(
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
            source_available INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tags(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS log_tags(
            log_id TEXT NOT NULL,
            tag_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(log_id, tag_id),
            FOREIGN KEY(log_id) REFERENCES browse_logs(id) ON DELETE CASCADE,
            FOREIGN KEY(tag_id) REFERENCES tags(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS browse_sync_sources(
            source_kind TEXT PRIMARY KEY,
            source_db_path TEXT NOT NULL,
            log_dir_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_browse_logs_upload_date ON browse_logs(upload_date);
        CREATE INDEX IF NOT EXISTS idx_browse_logs_log_date ON browse_logs(log_date);
        CREATE INDEX IF NOT EXISTS idx_browse_logs_error_count ON browse_logs(error_count);
        CREATE INDEX IF NOT EXISTS idx_log_tags_tag_id ON log_tags(tag_id);
        """
    )
    browse_log_columns = {
        row[1] for row in con.execute("PRAGMA table_info('browse_logs')")
    }
    if "source_available" not in browse_log_columns:
        con.execute(
            "ALTER TABLE browse_logs "
            "ADD COLUMN source_available INTEGER NOT NULL DEFAULT 1"
        )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_browse_logs_source_available "
        "ON browse_logs(source_available)"
    )
    con.execute(f"PRAGMA user_version={BROWSE_SCHEMA_VERSION}")
    con.commit()


def _upsert_log(con: sqlite3.Connection, record: dict[str, Any]) -> str:
    now = _iso_utc(datetime.now(timezone.utc))
    values = _serialized_log_values(record)
    existing = con.execute(
        """
        SELECT
            source_kind, source_log_id, log_path, original_filename,
            upload_date, log_date, vehicle_type, airframe_name, airframe_group,
            airframe_id, airframe_image_key, hardware, software,
            software_version, duration_s, error_count, flight_modes_json,
            metadata_json, search_text, source_available, created_at
        FROM browse_logs
        WHERE id = ?
        """,
        (record["id"],),
    ).fetchone()
    if existing is not None and tuple(existing[:-1]) == values:
        return "unchanged"

    created_at = existing[-1] if existing else now
    con.execute(
        """
        INSERT INTO browse_logs(
            id, source_kind, source_log_id, log_path, original_filename,
            upload_date, log_date, vehicle_type, airframe_name, airframe_group,
            airframe_id, airframe_image_key, hardware, software, software_version,
            duration_s, error_count, flight_modes_json, metadata_json, search_text,
            source_available, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            source_kind = excluded.source_kind,
            source_log_id = excluded.source_log_id,
            log_path = excluded.log_path,
            original_filename = excluded.original_filename,
            upload_date = excluded.upload_date,
            log_date = excluded.log_date,
            vehicle_type = excluded.vehicle_type,
            airframe_name = excluded.airframe_name,
            airframe_group = excluded.airframe_group,
            airframe_id = excluded.airframe_id,
            airframe_image_key = excluded.airframe_image_key,
            hardware = excluded.hardware,
            software = excluded.software,
            software_version = excluded.software_version,
            duration_s = excluded.duration_s,
            error_count = excluded.error_count,
            flight_modes_json = excluded.flight_modes_json,
            metadata_json = excluded.metadata_json,
            search_text = excluded.search_text,
            source_available = excluded.source_available,
            updated_at = excluded.updated_at
        """,
        (
            record["id"],
            *values,
            created_at,
            now,
        ),
    )
    return "updated" if existing is not None else "added"


def _serialized_log_values(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        record.get("source_kind") or "local",
        record.get("source_log_id"),
        record["log_path"],
        record.get("original_filename"),
        record.get("upload_date"),
        record.get("log_date"),
        record.get("vehicle_type"),
        record.get("airframe_name"),
        record.get("airframe_group"),
        record.get("airframe_id"),
        record.get("airframe_image_key") or DEFAULT_AIRFRAME_IMAGE_KEY,
        record.get("hardware"),
        record.get("software"),
        record.get("software_version"),
        record.get("duration_s"),
        int(record.get("error_count") or 0),
        json.dumps(record.get("flight_modes") or [], sort_keys=True),
        json.dumps(record.get("metadata") or {}, sort_keys=True),
        record.get("search_text") or _build_search_text(dict(record)),
        1,
    )


def _connect_flight_review_source(source_db: Path) -> sqlite3.Connection:
    source_uri = f"{source_db.resolve().as_uri()}?mode=ro"
    con = sqlite3.connect(source_uri, uri=True, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA query_only=ON")
    con.execute("PRAGMA busy_timeout=30000")
    return con


def _paths_refer_to_same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    if not right.exists():
        return False
    try:
        return left.samefile(right)
    except OSError:
        return False


def _claim_flight_review_source(
    con: sqlite3.Connection,
    source_db: Path,
    log_dir: Path,
) -> None:
    source_db_path = str(source_db.resolve())
    log_dir_path = str(log_dir.resolve())
    existing = con.execute(
        """
        SELECT source_db_path, log_dir_path
        FROM browse_sync_sources
        WHERE source_kind = 'flight_review'
        """
    ).fetchone()
    if existing is None:
        _validate_unbound_flight_review_rows(con, log_dir)
    elif tuple(existing) != (source_db_path, log_dir_path):
        _raise_different_flight_review_source()

    now = _iso_utc(datetime.now(timezone.utc))
    con.execute(
        """
        INSERT INTO browse_sync_sources(
            source_kind, source_db_path, log_dir_path, created_at, updated_at
        )
        VALUES ('flight_review', ?, ?, ?, ?)
        ON CONFLICT(source_kind) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (source_db_path, log_dir_path, now, now),
    )


def _validate_unbound_flight_review_rows(
    con: sqlite3.Connection,
    log_dir: Path,
) -> None:
    requested_log_dir = log_dir.resolve()
    existing_paths = con.execute(
        """
        SELECT log_path
        FROM browse_logs
        WHERE source_kind = 'flight_review'
        """
    ).fetchall()
    for (path_value,) in existing_paths:
        if not path_value:
            _raise_different_flight_review_source()
        existing_log_dir = Path(str(path_value)).expanduser().resolve().parent
        if existing_log_dir != requested_log_dir:
            _raise_different_flight_review_source()


def _raise_different_flight_review_source() -> None:
    raise ValueError(
        "browse database is already synchronized with a different "
        "Flight Review source; use the original storage path or a "
        "different browse database"
    )


def _flight_review_rows(con: sqlite3.Connection) -> list[sqlite3.Row]:
    logs_columns = _sqlite_table_columns(con, "Logs")
    if not logs_columns:
        raise ValueError("Flight Review database is missing the Logs table")
    if "id" not in logs_columns:
        raise ValueError("Flight Review Logs table is missing required Id column")

    generated_columns = _sqlite_table_columns(con, "LogsGenerated")
    if generated_columns and "id" not in generated_columns:
        raise ValueError(
            "Flight Review LogsGenerated table is missing required Id column"
        )

    select_parts = ['Logs."Id" AS "Id"']
    select_parts.extend(
        _optional_source_column("Logs", column, logs_columns)
        for column in FLIGHT_REVIEW_LOG_COLUMNS
    )
    select_parts.extend(
        _optional_source_column(
            "LogsGenerated",
            column,
            generated_columns,
        )
        for column in FLIGHT_REVIEW_GENERATED_COLUMNS
    )
    join = (
        'LEFT JOIN LogsGenerated ON Logs."Id" = LogsGenerated."Id"'
        if generated_columns
        else ""
    )
    order_by = 'Logs."Date" DESC' if "date" in logs_columns else 'Logs."Id" ASC'
    sql = (
        f"SELECT {', '.join(select_parts)} "
        f"FROM Logs {join} ORDER BY {order_by}"
    )
    return con.execute(sql).fetchall()


def _sqlite_table_columns(con: sqlite3.Connection, table_name: str) -> set[str]:
    quoted_name = table_name.replace("'", "''")
    return {
        str(row[1]).casefold()
        for row in con.execute(f"PRAGMA table_info('{quoted_name}')")
    }


def _optional_source_column(
    table_name: str,
    column_name: str,
    available_columns: set[str],
) -> str:
    if column_name.casefold() in available_columns:
        return f'{table_name}."{column_name}" AS "{column_name}"'
    return f'NULL AS "{column_name}"'


def _flight_review_log_id(value: Any) -> str | None:
    log_id = "" if value is None else str(value).strip()
    if not log_id:
        return None
    if (
        log_id in {".", ".."}
        or "/" in log_id
        or "\\" in log_id
        or "\x00" in log_id
    ):
        raise ValueError(f"invalid Flight Review log id: {log_id!r}")
    return log_id


def _flight_review_log_path(log_dir: Path, log_id: str) -> Path:
    root = log_dir.resolve()
    candidate = (root / f"{log_id}.ulg").resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Flight Review log path escapes configured directory: {log_id!r}"
        ) from exc
    return candidate


def _record_from_flight_review_row(
    row: sqlite3.Row,
    log_path: Path,
    airframes: Mapping[str, Mapping[str, str]],
    airframe_image_root: Path,
    *,
    log_id: str,
) -> dict[str, Any]:
    upload_date = _coerce_datetime_iso(row["Date"])
    log_date = _iso_from_timestamp(row["StartTime"])
    flight_modes = _split_csv_values(row["FlightModes"])
    airframe = airframes.get(str(row["AutostartId"] or ""))
    record = {
        "id": f"flight_review:{log_id}",
        "source_kind": "flight_review",
        "source_log_id": log_id,
        "log_path": str(log_path),
        "original_filename": _clean_optional(row["OriginalFilename"]) or log_path.name,
        "upload_date": upload_date,
        "log_date": log_date,
        "vehicle_type": _clean_optional(row["MavType"]),
        "airframe_name": (airframe or {}).get("name"),
        "airframe_group": (airframe or {}).get("group") or (airframe or {}).get("type"),
        "airframe_id": _clean_optional(row["AutostartId"]),
        "airframe_image_key": resolve_airframe_image_key(
            row["AutostartId"],
            airframes,
            airframe_image_root,
        ),
        "hardware": _clean_optional(row["Hardware"]),
        "software": _clean_optional(row["Software"]),
        "software_version": _clean_optional(row["SoftwareVersion"]),
        "duration_s": _float_or_none(row["Duration"]),
        "error_count": int(row["NumLoggedErrors"] or 0),
        "flight_modes": flight_modes,
        "metadata": {
            "flight_review": {
                "id": log_id,
                "description": row["Description"],
                "flight_mode_durations": row["FlightModeDurations"],
                "uuid": row["UUID"],
            }
        },
    }
    record["search_text"] = _build_search_text(record)
    return record


def _build_filter_clause(
    *,
    search: str,
    tags: list[str],
    upload_start: str,
    upload_end: str,
    log_start: str,
    log_end: str,
) -> tuple[str, list[Any]]:
    clauses = ["source_available = 1"]
    params: list[Any] = []

    clean_search = str(search or "").strip().lower()
    if clean_search:
        for term in clean_search.split():
            clauses.append("LOWER(search_text) LIKE ? ESCAPE '\\'")
            params.append(f"%{_escape_like(term)}%")

    _add_date_clause(clauses, params, "upload_date", upload_start, upload_end)
    _add_date_clause(clauses, params, "log_date", log_start, log_end)

    if tags:
        placeholders = ", ".join("?" for _ in tags)
        normalized_tags = [tag.lower() for tag in tags]
        clauses.append(
            f"""
            id IN (
                SELECT log_tags.log_id
                FROM log_tags
                JOIN tags ON tags.id = log_tags.tag_id
                WHERE LOWER(tags.name) IN ({placeholders})
                GROUP BY log_tags.log_id
                HAVING COUNT(DISTINCT LOWER(tags.name)) = ?
            )
            """
        )
        params.extend(normalized_tags)
        params.append(len(set(normalized_tags)))

    return "WHERE " + " AND ".join(f"({clause})" for clause in clauses), params


def _add_date_clause(
    clauses: list[str],
    params: list[Any],
    column: str,
    start_date: str,
    end_date: str,
) -> None:
    start = _date_bound(start_date, end=False)
    end = _date_bound(end_date, end=True)
    if start:
        clauses.append(f"{column} IS NOT NULL AND {column} >= ?")
        params.append(start)
    if end:
        clauses.append(f"{column} IS NOT NULL AND {column} < ?")
        params.append(end)


def _order_by(sort: str, direction: str) -> str:
    columns = {
        "upload_date": "upload_date",
        "log_date": "log_date",
        "hardware": "hardware",
        "software": "software",
        "duration": "duration_s",
        "error_count": "error_count",
    }
    column = columns.get(str(sort or ""), "upload_date")
    order_dir = "ASC" if str(direction or "").lower() == "asc" else "DESC"
    return f"ORDER BY {column} IS NULL, {column} {order_dir}, id ASC"


def _row_payload(con: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    metadata = _json_object(row["metadata_json"])
    tags = [
        tag_row[0]
        for tag_row in con.execute(
            """
            SELECT tags.name
            FROM tags
            JOIN log_tags ON log_tags.tag_id = tags.id
            WHERE log_tags.log_id = ?
            ORDER BY tags.name COLLATE NOCASE
            """,
            (row["id"],),
        )
    ]
    return {
        "id": row["id"],
        "source_kind": row["source_kind"],
        "source_log_id": row["source_log_id"],
        "log_path": row["log_path"],
        "original_filename": row["original_filename"],
        "upload_date": row["upload_date"],
        "log_date": row["log_date"],
        "vehicle_type": row["vehicle_type"],
        "airframe_name": row["airframe_name"],
        "airframe_group": row["airframe_group"],
        "airframe_id": row["airframe_id"],
        "airframe_image_key": row["airframe_image_key"] or DEFAULT_AIRFRAME_IMAGE_KEY,
        "hardware": row["hardware"],
        "software": row["software"],
        "software_version": row["software_version"],
        "duration_s": row["duration_s"],
        "error_count": row["error_count"],
        "flight_modes": json.loads(row["flight_modes_json"] or "[]"),
        "tags": tags,
        "review_inputs": _normalize_review_inputs(metadata.get("review_inputs")),
    }


def _read_ulog_info(log_path: Path) -> dict[str, Any]:
    try:
        ulog = ULog(str(log_path), None, disable_str_exceptions=True)
    except Exception:
        return {}
    return {
        str(key): _clean_optional(value)
        for key, value in (getattr(ulog, "msg_info_dict", {}) or {}).items()
    }


def _extract_log_start_utc_s(log_path: Path) -> int | None:
    try:
        ulog = ULog(str(log_path), ["vehicle_gps_position"], disable_str_exceptions=True)
        data = ulog.get_dataset("vehicle_gps_position").data
    except Exception:
        return None
    values = data.get("time_utc_usec")
    if values is None:
        return None
    for value in values:
        try:
            integer = int(value)
        except (TypeError, ValueError):
            continue
        if integer > 0:
            return integer // 1_000_000
    return None


def _flight_modes_from_timeline(timeline: Iterable[dict[str, Any]]) -> list[str]:
    modes = []
    seen = set()
    for event in timeline or []:
        if event.get("topic") != "vehicle_status" or event.get("field") != "nav_state":
            continue
        value = str(event.get("value"))
        if value not in seen:
            seen.add(value)
            modes.append(value)
    return modes


def _vehicle_type_from_inventory(inventory: dict[str, Any]) -> str | None:
    parameters = inventory.get("parameters") or {}
    if "VT_TYPE" in parameters:
        return "vtol"
    topics = set(inventory.get("available_topics") or [])
    if "vtol_vehicle_status" in topics:
        return "vtol"
    return None


def _error_count(messages: Iterable[dict[str, Any]]) -> int:
    count = 0
    for message in messages:
        if str(message.get("level") or "").upper() in ERROR_LEVELS:
            count += 1
    return count


def _build_search_text(record: dict[str, Any]) -> str:
    values = [
        record.get("id"),
        record.get("source_log_id"),
        record.get("log_path"),
        record.get("original_filename"),
        record.get("vehicle_type"),
        record.get("airframe_name"),
        record.get("airframe_group"),
        record.get("airframe_id"),
        record.get("hardware"),
        record.get("software"),
        record.get("software_version"),
        " ".join(str(mode) for mode in record.get("flight_modes") or []),
    ]
    return " ".join(str(value) for value in values if value not in (None, ""))


def _ensure_tag(con: sqlite3.Connection, name: str) -> int:
    now = _iso_utc(datetime.now(timezone.utc))
    con.execute("INSERT OR IGNORE INTO tags(name, created_at) VALUES (?, ?)", (name, now))
    row = con.execute("SELECT id FROM tags WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    if row is None:
        raise ValueError("failed to create tag")
    return int(row[0])


def _log_exists(con: sqlite3.Connection, log_id: str) -> bool:
    return con.execute("SELECT 1 FROM browse_logs WHERE id = ?", (log_id,)).fetchone() is not None


def _local_log_id(log_path: Path) -> str:
    parent = log_path.parent.name
    if parent:
        return f"local:{parent}:{log_path.name}"
    return f"local:{log_path.name}"


def _split_csv_values(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _first_clean(*values: Any) -> str | None:
    for value in values:
        clean = _clean_optional(value)
        if clean:
            return clean
    return None


def _clean_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().rstrip("\x00")
    return text or None


def _clean_tag(value: Any) -> str:
    return str(value or "").strip()


def _normalize_review_inputs(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized = {}
    for key in ("mission_path", "source_path", "parameters_xml_path"):
        clean_value = _clean_optional(value.get(key))
        if clean_value:
            normalized[key] = clean_value
    return normalized


def _json_object(value: Any) -> dict[str, Any]:
    try:
        payload = json.loads(value or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_datetime_iso(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return _iso_utc(value)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return _iso_utc(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc))
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    return _iso_utc(parsed)


def _iso_from_timestamp(value: Any) -> str | None:
    if value in (None, "", 0, "0"):
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None
    if timestamp <= 0:
        return None
    try:
        return _iso_utc(datetime.fromtimestamp(timestamp, tz=timezone.utc))
    except (OverflowError, OSError, ValueError):
        return None


def _iso_utc(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _date_bound(value: str, *, end: bool) -> str | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    try:
        parsed = datetime.strptime(clean, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if end:
        parsed = parsed + timedelta(days=1)
    return _iso_utc(parsed)


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
