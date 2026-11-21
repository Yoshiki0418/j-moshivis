"""Read-only, incrementally refreshed catalogue for J-MoshiVis datasets.

The catalogue deliberately stores JSONL byte locators rather than JSON bodies.
This keeps the SQLite database small while still making a single sample cheap to
open.  Source files are re-indexed only when their size or mtime changes.  A
trailing line without a newline is treated as an in-progress write and ignored.

Only paths below ``<repo>/data`` may be exposed by :meth:`resolve_media`.
"""

from __future__ import annotations

import base64
import copy
import datetime as _dt
import json
import logging
import math
import os
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ssvd.ai_suggestion_store import (
    AISuggestionIntegrityError, default_ai_suggestion_store, human_touched_uids,
)
from ssvd.review_store import (
    MAX_REVIEW_NOTE_CHARS,
    REVIEW_DECISIONS,
    ReviewConflictError,
    content_fingerprint,
    default_review_store,
    image_file_sha256,
    requires_review,
    review_metadata,
)

SCHEMA_VERSION = "4"
REGENERATION_PENDING_SQL = """review_status='approved'
    AND json_type(summary_json,'$.human_revision')='object'
    AND COALESCE(json_extract(summary_json,'$.human_revision.had_media'),
        json_extract(summary_json,'$.source_media_paths[0]') IS NOT NULL
        OR json_extract(summary_json,'$.source_media_paths[1]') IS NOT NULL)
    AND (has_audio=0 OR has_alignment=0)"""

COLLECTIONS: dict[str, str] = {
    "visual_v3": "画像対話 v3",
    "japanese_daily_v1": "日本の日常画像 v1",
    "counting_v4": "カウント画像 v4",
    "ocr_v4": "日本語 OCR 画像 v4",
    "nonvisual_v1": "非画像対話 v1",
}


@dataclass(frozen=True)
class _SourceSpec:
    source_id: str
    collection: str
    role: str
    relative_path: str
    priority: int


class _RefreshInterrupted(Exception):
    """Internal control flow: yield the catalogue writer to an annotation."""


_SOURCE_SPECS: tuple[_SourceSpec, ...] = (
    _SourceSpec(
        "japanese_daily_v1.images",
        "japanese_daily_v1",
        "daily_image_bank",
        "data/images/japanese_daily_v1/accepted_images.jsonl",
        10,
    ),
    _SourceSpec(
        "visual_v3.dialogues",
        "visual_v3",
        "visual_dialogue",
        "data/dialogues_merged/dialogues_final_v3.jsonl",
        20,
    ),
    _SourceSpec(
        "visual_v3.assets",
        "visual_v3",
        "asset_manifest",
        "data/speech_v3/asset_manifest.v1.jsonl",
        30,
    ),
    _SourceSpec(
        "counting_v4.images",
        "counting_v4",
        "targeted_images",
        "data/images/targeted_v4/image_pool_v1/counting_images.jsonl",
        10,
    ),
    _SourceSpec(
        "counting_v4.seed",
        "counting_v4",
        "counting_dialogue",
        "data/dialogues_counting/full/qwen3vl_30b_counting_seed5.jsonl",
        20,
    ),
    _SourceSpec(
        "counting_v4.shard0",
        "counting_v4",
        "counting_dialogue",
        "data/dialogues_counting/full/qwen3vl_30b_counting_s0.jsonl",
        21,
    ),
    _SourceSpec(
        "counting_v4.shard1",
        "counting_v4",
        "counting_dialogue",
        "data/dialogues_counting/full/qwen3vl_30b_counting_s1.jsonl",
        22,
    ),
    _SourceSpec(
        "counting_v4.normal_shard0",
        "counting_v4",
        "counting_dialogue",
        "data/dialogues_counting/full/qwen3vl_30b_normal_s0.jsonl",
        23,
    ),
    _SourceSpec(
        "counting_v4.normal_shard1",
        "counting_v4",
        "counting_dialogue",
        "data/dialogues_counting/full/qwen3vl_30b_normal_s1.jsonl",
        24,
    ),
    _SourceSpec(
        "ocr_v4.images",
        "ocr_v4",
        "targeted_images",
        "data/images/targeted_v4/image_pool_v1/ocr_images.jsonl",
        10,
    ),
    _SourceSpec(
        "nonvisual_v1.dialogues",
        "nonvisual_v1",
        "nonvisual_dialogue",
        "data/dialogues_nonvisual/claude_sonnet5_nonvisual_tts_v1.jsonl",
        20,
    ),
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _has_both_auxiliary_kinds(summary: Mapping[str, Any]) -> bool:
    task = summary.get("auxiliary_task")
    if not isinstance(task, Mapping) or task.get("kind") not in ("counting", "ocr"):
        return False
    additional = task.get("additional_tasks")
    if not isinstance(additional, list):
        return False
    other = "ocr" if task["kind"] == "counting" else "counting"
    return any(isinstance(entry, Mapping) and entry.get("kind") == other
               for entry in additional)


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_duration(value: Any) -> float | None:
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return duration if math.isfinite(duration) and duration >= 0 else None


class DatasetCatalog:
    """Persistent catalogue over the four canonical dataset collections.

    Parameters
    ----------
    repo_root:
        Repository root that contains ``data``.
    db_path:
        SQLite file.  It may live outside the repository (for example in a
        temporary directory during tests), but indexed media is always confined
        to ``repo_root/data``.
    """

    def __init__(self, repo_root: str | os.PathLike[str], db_path: str | os.PathLike[str]):
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.data_root = (self.repo_root / "data").resolve()
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_store = default_review_store(self.repo_root)
        self.ai_suggestion_store = default_ai_suggestion_store(self.repo_root)
        self._refresh_lock = threading.Lock()
        self._interactive_lock = threading.Lock()
        self._interactive_waiters = 0
        self._interactive_pending = threading.Event()
        self._interactive_pause_until = 0.0
        self._initialise()

    def request_interactive_pause(self, seconds: float = 30.0) -> None:
        """Nonblocking hint from the HTTP loop: postpone rebuildable scanning."""
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("interactive pause must be finite and nonnegative")
        with self._interactive_lock:
            self._interactive_pause_until = max(self._interactive_pause_until, time.monotonic() + seconds)

    def resume_background_refresh(self) -> None:
        """Honor an explicit refresh, while queued annotation writes still win."""
        with self._interactive_lock:
            self._interactive_pause_until = 0.0

    def _refresh_should_yield(self) -> bool:
        return self._interactive_pending.is_set() or time.monotonic() < self._interactive_pause_until

    @contextmanager
    def _interactive_write(self) -> Iterator[None]:
        """Let a human save interrupt an ongoing, rebuildable background scan.

        The event is set *before* waiting for the catalogue writer. All queued
        saves keep it set until the last one finishes, so a new refresh cannot
        steal the lock between edits. The approval ledger is still fsynced and
        the exact image/text fingerprint is verified under the same writer lock.
        """
        with self._interactive_lock:
            self._interactive_waiters += 1
            self._interactive_pending.set()
        try:
            with self._refresh_lock:
                yield
        finally:
            with self._interactive_lock:
                self._interactive_waiters -= 1
                if self._interactive_waiters == 0:
                    self._interactive_pending.clear()

    def _check_refresh_interruption(self) -> None:
        if self._refresh_should_yield():
            raise _RefreshInterrupted()

    @contextmanager
    def _background_write(self) -> Iterator[sqlite3.Connection]:
        with self._refresh_lock, self._connect() as db:
            self._check_refresh_interruption()
            # SQL aggregations/deletes can themselves take seconds. Cooperate
            # during SQL too; _connect rolls the transaction back on exception.
            db.set_progress_handler(lambda: int(self._refresh_should_yield()), 10000)
            try:
                yield db
            except sqlite3.OperationalError as error:
                if (getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT
                        and self._refresh_should_yield()):
                    raise _RefreshInterrupted() from None
                raise
            finally:
                # Rollback must run without a handler that would interrupt it.
                db.set_progress_handler(None, 0)

    # ------------------------------------------------------------------
    # Database lifecycle
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=60.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=60000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialise(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sources (
                    source_id TEXT PRIMARY KEY,
                    collection TEXT NOT NULL,
                    role TEXT NOT NULL,
                    path TEXT NOT NULL,
                    mtime_ns INTEGER,
                    size_bytes INTEGER,
                    status TEXT NOT NULL,
                    row_count INTEGER NOT NULL DEFAULT 0,
                    ignored_partial INTEGER NOT NULL DEFAULT 0,
                    error_count INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    indexed_at TEXT,
                    indexed_bytes INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS components (
                    source_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    uid TEXT NOT NULL,
                    role TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    offset_bytes INTEGER NOT NULL,
                    length_bytes INTEGER NOT NULL,
                    source_uid TEXT,
                    category TEXT,
                    image_path TEXT,
                    audio_path TEXT,
                    alignment_path TEXT,
                    speech_dialogue_path TEXT,
                    has_image INTEGER NOT NULL DEFAULT 0,
                    has_dialogue INTEGER NOT NULL DEFAULT 0,
                    has_audio INTEGER NOT NULL DEFAULT 0,
                    has_alignment INTEGER NOT NULL DEFAULT 0,
                    eligible INTEGER,
                    search_text TEXT NOT NULL DEFAULT '',
                    summary_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (source_id, uid, role),
                    FOREIGN KEY (source_id) REFERENCES sources(source_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS components_collection_uid
                    ON components(collection, uid);
                CREATE TABLE IF NOT EXISTS items (
                    collection TEXT NOT NULL,
                    uid TEXT NOT NULL,
                    source_uid TEXT NOT NULL,
                    category TEXT NOT NULL,
                    directory TEXT NOT NULL,
                    image_path TEXT,
                    audio_path TEXT,
                    alignment_path TEXT,
                    speech_dialogue_path TEXT,
                    metadata_source TEXT,
                    metadata_offset INTEGER,
                    metadata_length INTEGER,
                    dialogue_source TEXT,
                    dialogue_offset INTEGER,
                    dialogue_length INTEGER,
                    has_image INTEGER NOT NULL,
                    has_dialogue INTEGER NOT NULL,
                    has_audio INTEGER NOT NULL,
                    has_alignment INTEGER NOT NULL,
                    eligible INTEGER,
                    stage TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    verification_status TEXT,
                    auxiliary_kind TEXT,
                    has_both_auxiliary_kinds INTEGER NOT NULL DEFAULT 0,
                    review_required INTEGER NOT NULL DEFAULT 0,
                    review_id TEXT,
                    review_status TEXT NOT NULL DEFAULT 'not_required',
                    dataset_usable INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(collection, uid)
                );
                CREATE INDEX IF NOT EXISTS items_filter
                    ON items(collection, category, stage, uid);
                CREATE INDEX IF NOT EXISTS items_order
                    ON items(collection, category, uid);
                """
            )
            # CREATE TABLE IF NOT EXISTS does not add fields to a v3 cache.
            # Additive guards make the cache upgrade safe before repopulation.
            db.execute("BEGIN IMMEDIATE")
            item_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(items)")
            }
            for name, declaration in (
                ("verification_status", "TEXT"),
                ("auxiliary_kind", "TEXT"),
                ("review_required", "INTEGER NOT NULL DEFAULT 0"),
                ("review_id", "TEXT"),
                ("review_status", "TEXT NOT NULL DEFAULT 'not_required'"),
                ("dataset_usable", "INTEGER NOT NULL DEFAULT 1"),
            ):
                if name not in item_columns:
                    db.execute(f"ALTER TABLE items ADD COLUMN {name} {declaration}")
            if "has_both_auxiliary_kinds" not in item_columns:
                db.execute("ALTER TABLE items ADD COLUMN has_both_auxiliary_kinds INTEGER NOT NULL DEFAULT 0")
                db.execute("""UPDATE items SET has_both_auxiliary_kinds=1
                    WHERE auxiliary_kind IN ('counting','ocr') AND EXISTS (
                        SELECT 1 FROM json_each(items.summary_json, '$.auxiliary_task.additional_tasks') AS t
                        WHERE t.type='object' AND json_extract(t.value, '$.kind') =
                            CASE items.auxiliary_kind WHEN 'counting' THEN 'ocr' ELSE 'counting' END)""")
            source_columns = {
                row["name"] for row in db.execute("PRAGMA table_info(sources)")
            }
            if "indexed_bytes" not in source_columns:
                db.execute(
                    "ALTER TABLE sources ADD COLUMN indexed_bytes "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            db.execute(
                """CREATE INDEX IF NOT EXISTS items_review
                   ON items(review_required,review_status,auxiliary_kind,collection,uid)"""
            )
            db.execute("CREATE INDEX IF NOT EXISTS items_review_uid ON items(uid)")
            db.execute("""CREATE INDEX IF NOT EXISTS items_both_auxiliary ON items(has_both_auxiliary_kinds)
                          WHERE has_both_auxiliary_kinds=1""")
            version = db.execute(
                "SELECT value FROM meta WHERE key='schema_version'"
            ).fetchone()
            if (
                version is not None
                and version[0] not in {SCHEMA_VERSION, "3"}
            ):
                db.executescript(
                    "DELETE FROM components; DELETE FROM items; DELETE FROM sources; DELETE FROM meta;"
                )
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (SCHEMA_VERSION,),
            )

    # ------------------------------------------------------------------
    # Refresh / source parsing
    # ------------------------------------------------------------------
    def _source_specs(self) -> tuple[_SourceSpec, ...]:
        """Return fixed sources plus deterministically discovered review inputs."""

        specs = list(_SOURCE_SPECS)
        fixed_paths = {spec.relative_path for spec in specs}

        normal_root = self.repo_root / "data/dialogues_counting/full"
        if normal_root.is_dir():
            extra_normal = sorted(normal_root.glob("qwen3vl_30b_normal*.jsonl"))
            for index, path in enumerate(extra_normal):
                relative = path.relative_to(self.repo_root).as_posix()
                if relative in fixed_paths:
                    continue
                specs.append(
                    _SourceSpec(
                        f"dynamic.normal:{relative}",
                        "counting_v4",
                        "counting_dialogue",
                        relative,
                        25 + index,
                    )
                )

        candidate_root = self.repo_root / "data/dialogues_review_candidates"
        candidate_roles = {
            "visual_v3": "visual_dialogue",
            "counting_v4": "counting_dialogue",
            "ocr_v4": "counting_dialogue",
        }
        discovered: list[tuple[str, Path]] = []
        for collection in sorted(candidate_roles):
            directory = candidate_root / collection
            if directory.is_dir():
                discovered.extend((collection, path) for path in directory.glob("*.jsonl"))
        for index, (collection, path) in enumerate(
            sorted(discovered, key=lambda value: (value[0], value[1].name))
        ):
            relative = path.relative_to(self.repo_root).as_posix()
            specs.append(
                _SourceSpec(
                    f"dynamic.review_candidate:{relative}",
                    collection,
                    candidate_roles[collection],
                    relative,
                    40 + index,
                )
            )
        return tuple(specs)

    def refresh(self, force: bool = False) -> dict[str, Any]:
        """Refresh changed JSONL sources and return :meth:`status`.

        Import is transactional.  If a source is being appended to, its final
        incomplete line is not visible until a later refresh completes it.
        """

        try:
            return self._refresh_transactional(force=force)
        except _RefreshInterrupted:
            logging.getLogger(__name__).info("Catalog refresh yielded to an annotation save; uncommitted scan rolled back")
            return {**self.status(), "refresh_interrupted": True}

    def _refresh_transactional(self, force: bool = False) -> dict[str, Any]:
        changed_collections: set[str] = set()
        appended_uids: dict[str, set[str]] = {}
        with self._background_write() as db:
            phase_start = time.monotonic()
            timings = {}
            progress_before = self._progress_signature(db)
            source_specs = self._source_specs()
            current_dynamic = {
                spec.source_id
                for spec in source_specs
                if spec.source_id.startswith("dynamic.")
            }
            for row in db.execute(
                "SELECT source_id,collection FROM sources WHERE source_id LIKE 'dynamic.%'"
            ).fetchall():
                if row["source_id"] not in current_dynamic:
                    db.execute("DELETE FROM sources WHERE source_id=?", (row["source_id"],))
                    changed_collections.add(row["collection"])

            for spec in source_specs:
                self._check_refresh_interruption()
                path = self.repo_root / spec.relative_path
                previous = db.execute(
                    "SELECT * FROM sources WHERE source_id=?",
                    (spec.source_id,),
                ).fetchone()

                if not path.is_file():
                    if previous is None:
                        db.execute(
                            """INSERT INTO sources(
                                   source_id,collection,role,path,status,indexed_at
                               ) VALUES(?,?,?,?,?,?)""",
                            (
                                spec.source_id,
                                spec.collection,
                                spec.role,
                                spec.relative_path,
                                "missing",
                                _utc_now(),
                            ),
                        )
                    elif force or previous["status"] != "missing":
                        db.execute("DELETE FROM components WHERE source_id=?", (spec.source_id,))
                        db.execute(
                            """UPDATE sources SET mtime_ns=NULL,size_bytes=NULL,status='missing',
                               row_count=0,ignored_partial=0,error_count=0,error=NULL,
                               indexed_at=?,indexed_bytes=0
                               WHERE source_id=?""",
                            (_utc_now(), spec.source_id),
                        )
                        changed_collections.add(spec.collection)
                    continue

                stat = path.stat()
                unchanged = (
                    previous is not None
                    and previous["status"] in {"ok", "ok_with_errors"}
                    and previous["mtime_ns"] == stat.st_mtime_ns
                    and previous["size_bytes"] == stat.st_size
                )
                if unchanged and not force:
                    continue

                # accepted_images.jsonl grows while collection is running.
                # Resume at the last complete newline rather than rescanning a
                # potentially 100k+ record manifest every ten seconds.  A
                # rewrite, truncation, or forced refresh still performs a full
                # conservative scan.
                append_only = bool(
                    not force
                    and spec.role == "daily_image_bank"
                    and previous is not None
                    and previous["status"] in {"ok", "ok_with_errors"}
                    and previous["size_bytes"] is not None
                    and stat.st_size > previous["size_bytes"]
                    and 0 <= int(previous["indexed_bytes"] or 0) <= stat.st_size
                )

                # Ensure the FK parent exists before inserting components.
                db.execute(
                    """INSERT INTO sources(source_id,collection,role,path,status,indexed_at)
                       VALUES(?,?,?,?,?,?)
                       ON CONFLICT(source_id) DO UPDATE SET
                         collection=excluded.collection, role=excluded.role,
                         path=excluded.path, status=excluded.status,
                         indexed_at=excluded.indexed_at""",
                    (
                        spec.source_id,
                        spec.collection,
                        spec.role,
                        spec.relative_path,
                        "indexing",
                        _utc_now(),
                    ),
                )
                if not append_only:
                    db.execute(
                        "DELETE FROM components WHERE source_id=?", (spec.source_id,)
                    )

                rows = int(previous["row_count"] or 0) if append_only else 0
                partial = 0
                error_count = int(previous["error_count"] or 0) if append_only else 0
                errors = (
                    str(previous["error"] or "").splitlines()[:5]
                    if append_only
                    else []
                )
                indexed_bytes = (
                    int(previous["indexed_bytes"] or 0) if append_only else 0
                )
                batch: list[tuple[Any, ...]] = []
                source_changed_uids: set[str] = set()
                for offset, length, record, parse_error in self._iter_jsonl(
                    path, start_offset=indexed_bytes
                ):
                    self._check_refresh_interruption()
                    if parse_error == "partial":
                        partial += 1
                        continue
                    # Any complete physical line is safe to skip next time,
                    # including blank or malformed lines.
                    indexed_bytes = offset + length
                    if parse_error == "blank":
                        continue
                    if parse_error is not None:
                        error_count += 1
                        if len(errors) < 5:
                            errors.append(f"byte {offset}: {parse_error}")
                        continue
                    assert record is not None
                    component = self._component_from_record(spec, record, offset, length)
                    if component is None:
                        continue
                    batch.append(component)
                    rows += 1
                    source_changed_uids.add(str(component[2]))
                    if len(batch) >= 2_000:
                        self._insert_components(db, batch)
                        batch.clear()
                if batch:
                    self._insert_components(db, batch)

                if append_only and source_changed_uids:
                    appended_uids.setdefault(spec.collection, set()).update(
                        source_changed_uids
                    )
                else:
                    changed_collections.add(spec.collection)

                status = "ok_with_errors" if error_count else "ok"
                db.execute(
                    """UPDATE sources SET mtime_ns=?,size_bytes=?,status=?,row_count=?,
                       ignored_partial=?,error_count=?,error=?,indexed_at=?,indexed_bytes=?
                       WHERE source_id=?""",
                    (
                        stat.st_mtime_ns,
                        stat.st_size,
                        status,
                        rows,
                        partial,
                        error_count,
                        "\n".join(errors) or None,
                        _utc_now(),
                        indexed_bytes,
                        spec.source_id,
                    ),
                )

            timings["sources"] = round(time.monotonic() - phase_start, 3)
            phase_start = time.monotonic()
            for collection, uids in self._refresh_targeted_speech_roots(db).items():
                appended_uids.setdefault(collection, set()).update(uids)
            timings["media_scan"] = round(time.monotonic() - phase_start, 3)
            phase_start = time.monotonic()

            for collection in changed_collections:
                self._check_refresh_interruption()
                self._rebuild_collection(db, collection)
            for collection, uids in appended_uids.items():
                self._check_refresh_interruption()
                if collection not in changed_collections:
                    self._rebuild_items_for_uids(db, collection, uids)
            self._refresh_pending_media(db)
            timings["rebuild_and_pending"] = round(time.monotonic() - phase_start, 3)
            phase_start = time.monotonic()
            duration_updates = self._backfill_audio_durations(db)
            review_updates = self._apply_review_decisions(db)
            timings["review"] = round(time.monotonic() - phase_start, 3)
            logging.getLogger(__name__).info("Catalog refresh phases (seconds): %s", timings)

            refreshed_at = _utc_now()
            self._check_refresh_interruption()
            progress_after = self._progress_signature(db)
            if progress_after != progress_before or duration_updates or review_updates:
                revision_row = db.execute(
                    "SELECT value FROM meta WHERE key='catalog_revision'"
                ).fetchone()
                try:
                    revision = int(revision_row[0]) + 1 if revision_row else 1
                except (TypeError, ValueError):
                    revision = 1
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('catalog_revision',?)",
                    (str(revision),),
                )
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('last_changed',?)",
                    (refreshed_at,),
                )
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('last_refresh',?)",
                (refreshed_at,),
            )

        # A first build writes hundreds of megabytes in one transaction.  WAL
        # mode keeps readers responsive, but its log otherwise remains roughly
        # as large as the catalogue after the commit.  Checkpoint only after the
        # writer transaction has closed so the temporary space is released.
        if not self._refresh_should_yield():
            with self._connect() as checkpoint_db:
                # A reader/annotation may start just after the scan commits.
                # Keep checkpoint lock acquisition opportunistic, not a second
                # minute-long wait in front of an interactive writer.
                checkpoint_db.execute("PRAGMA busy_timeout=0")
                checkpoint_db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

        return self.status()

    @staticmethod
    def _progress_signature(db: sqlite3.Connection) -> tuple[tuple[Any, ...], ...]:
        """Return the lightweight dashboard/tree state used for change notices."""

        return tuple(
            tuple(row)
            for row in db.execute(
                """SELECT collection,category,COUNT(*) AS items,
                          COALESCE(SUM(has_image),0) AS images,
                          COALESCE(SUM(has_dialogue),0) AS dialogues,
                          COALESCE(SUM(has_audio),0) AS audio,
                          COALESCE(SUM(has_alignment),0) AS alignments,
                          COALESCE(SUM(CASE WHEN eligible=1 THEN 1 ELSE 0 END),0) AS eligible,
                          COALESCE(SUM(review_required),0) AS review_required,
                          COALESCE(SUM(CASE WHEN review_status='approved' THEN 1 ELSE 0 END),0)
                              AS review_approved,
                          COALESCE(SUM(CASE WHEN review_status='rejected' THEN 1 ELSE 0 END),0)
                              AS review_rejected,
                          COALESCE(SUM(dataset_usable),0) AS dataset_usable
                   FROM items GROUP BY collection,category
                   ORDER BY collection,category"""
            )
        )

    def _refresh_review_identity(
        self,
        db: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        force_hash: bool,
        persist: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        """Refresh a review ID when the current source image has changed."""

        try:
            summary = json.loads(row["summary_json"])
        except (TypeError, json.JSONDecodeError):
            summary = {}
        if not isinstance(summary, dict):
            summary = {}

        current_size: int | None = None
        current_mtime_ns: int | None = None
        try:
            if not row["image_path"]:
                raise FileNotFoundError("missing image path")
            stat = self._safe_media_path(row["image_path"]).stat()
            current_size = stat.st_size
            current_mtime_ns = stat.st_mtime_ns
        except (OSError, ValueError):
            pass

        stat_changed = (
            summary.get("review_image_size_bytes") != current_size
            or summary.get("review_image_mtime_ns") != current_mtime_ns
        )
        base_id = summary.get("source_review_id") or row["review_id"]
        revision = self.review_store.latest_revisions().get(base_id)
        if not force_hash and not stat_changed:
            if not revision or revision["fingerprint"] == row["review_id"]:
                return summary, False

        dialogue_record = self._read_locator(
            row["dialogue_source"],
            row["dialogue_offset"],
            row["dialogue_length"],
        )
        if isinstance(dialogue_record, Mapping):
            metadata = self._review_metadata_for_record(
                dialogue_record, row["image_path"], verify_image=True
            )
            base_id = metadata["review_id"]
            revision = self.review_store.latest_revisions().get(base_id)
            summary["source_review_id"] = base_id
            summary.pop("human_revision", None)
            if revision:
                dialogue_record = revision["revised_record"]
                metadata = self._review_metadata_for_record(dialogue_record, row["image_path"], verify_image=True)
                summary["human_revision"] = dialogue_record.get("human_revision")
            summary.update(metadata)
            review_id = metadata.get("review_id")
            verification_status = metadata.get("verification_status")
            auxiliary_kind = metadata.get("auxiliary_kind")
        else:
            summary.update(
                {
                    "current_image_sha256": None,
                    "image_integrity_status": "unavailable",
                    "review_image_size_bytes": current_size,
                    "review_image_mtime_ns": current_mtime_ns,
                }
            )
            review_id = row["review_id"]
            verification_status = row["verification_status"]
            auxiliary_kind = row["auxiliary_kind"]

        changed = (
            review_id != row["review_id"]
            or summary.get("human_revision") != self._summary_value(row["summary_json"], "human_revision")
            or summary.get("source_review_id") != self._summary_value(row["summary_json"], "source_review_id")
            or summary.get("image_integrity_status")
            != self._summary_value(row["summary_json"], "image_integrity_status")
            or stat_changed
        )
        if changed and persist:
            # Different text needs different audio/alignment. Use a new storage
            # directory, never overwrite or expose speech for the old version.
            media_paths = [row[key] for key in ("audio_path", "alignment_path", "speech_dialogue_path")]
            if revision:
                summary.setdefault("source_media_paths", media_paths)
                from ssvd.tts.dataset_keys import speech_sample_uid
                storage_uid = speech_sample_uid(dialogue_record)
                media_paths = [str(Path(path).parent.parent / storage_uid / Path(path).name)
                               if path else None for path in media_paths]
            elif self._summary_value(row["summary_json"], "human_revision"):
                media_paths = summary.pop("source_media_paths", [None, None, None])
            if review_id != row["review_id"]:
                summary.pop("duration_seconds", None)
                summary.pop("duration", None)
            has_audio = int(bool(media_paths[0] and self._stored_path_exists(media_paths[0])))
            has_alignment = int(bool(media_paths[1] and self._stored_path_exists(media_paths[1])))
            if has_audio and media_paths[2]:
                speech = self._read_json_media(media_paths[2])
                if isinstance(speech, Mapping):
                    duration = _as_duration(speech.get("duration_seconds", speech.get("duration")))
                    if duration is not None:
                        summary["duration_seconds"] = duration
            stage = "complete" if has_audio and has_alignment else "audio" if has_audio else "dialogue"
            db.execute(
                """UPDATE items SET review_id=?,verification_status=?,
                       auxiliary_kind=?,has_both_auxiliary_kinds=?,summary_json=?,review_status=?,
                       dataset_usable=?,audio_path=?,alignment_path=?,speech_dialogue_path=?,
                       has_audio=?,has_alignment=?,stage=?
                   WHERE collection=? AND uid=?""",
                (
                    review_id,
                    verification_status,
                    auxiliary_kind,
                    int(_has_both_auxiliary_kinds(summary)),
                    _json_dumps(summary),
                    "pending" if row["review_required"] else "not_required", int(not row["review_required"]),
                    *media_paths, has_audio, has_alignment, stage,
                    row["collection"],
                    row["uid"],
                ),
            )
        return summary, changed

    @staticmethod
    def _summary_value(raw: str, key: str) -> Any:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            return None
        return value.get(key) if isinstance(value, Mapping) else None

    def _apply_review_decisions(self, db: sqlite3.Connection) -> int:
        """Resolve exact-content decisions; stale approvals never match a new hash."""

        latest = self.review_store.latest_decisions()
        affected_uids = {str(event["uid"]) for event in latest.values() if event.get("uid") is not None}
        identity_updates = 0
        # Optional samples need tracking only after a human decision. Avoid
        # hashing/walking hundreds of thousands of ordinary samples every tick.
        db.execute("CREATE TEMP TABLE IF NOT EXISTS review_affected(uid TEXT PRIMARY KEY)")
        db.execute("DELETE FROM review_affected")
        db.executemany("INSERT INTO review_affected(uid) VALUES(?)", [(uid,) for uid in affected_uids])
        for row in db.execute(
            "SELECT * FROM items WHERE has_dialogue=1 AND (review_required=1 OR uid IN (SELECT uid FROM review_affected))"
        ).fetchall():
            self._check_refresh_interruption()
            _, changed = self._refresh_review_identity(
                db, row, force_hash=not bool(self._summary_value(row["summary_json"], "source_review_id"))
            )
            identity_updates += int(changed)

        rows = db.execute(
            """SELECT collection,uid,review_id,review_status,dataset_usable,review_required,
                      summary_json
               FROM items WHERE has_dialogue=1 AND (review_required=1 OR uid IN (SELECT uid FROM review_affected))"""
        ).fetchall()
        updates: list[tuple[str, int, str, str]] = []
        for row in rows:
            self._check_refresh_interruption()
            integrity = self._summary_value(
                row["summary_json"], "image_integrity_status"
            )
            event = latest.get(row["review_id"]) if row["review_id"] else None
            status, usable = self._review_state(bool(row["review_required"]), event, integrity)
            if status != row["review_status"] or usable != row["dataset_usable"]:
                updates.append((status, usable, row["collection"], row["uid"]))
        if updates:
            db.executemany(
                """UPDATE items SET review_status=?,dataset_usable=?
                   WHERE collection=? AND uid=?""",
                updates,
            )
        return identity_updates + len(updates)

    @staticmethod
    def _review_state(required: bool, event: Mapping | None, integrity: str | None) -> tuple[str, int]:
        decision = event.get("decision") if event and integrity != "unavailable" else None
        status = decision if decision in {"approved", "rejected"} else "pending" if required else "not_required"
        return status, int(status != "rejected" and (not required or status == "approved"))

    def _refresh_pending_media(self, db: sqlite3.Connection) -> int:
        """Promote only unfinished deterministic media paths that now exist.

        Visual-v3 and nonvisual-v1 records already know where their generated
        WAV/alignment files will be written.  Checking only rows whose stage is
        incomplete keeps a periodic refresh cheap, while still noticing files
        added without a corresponding JSONL modification.
        """

        rows = db.execute(
            """SELECT collection,uid,audio_path,alignment_path,
                      has_image,has_dialogue,has_audio,has_alignment
               FROM items
               WHERE (has_audio=0 AND audio_path IS NOT NULL)
                  OR (has_alignment=0 AND alignment_path IS NOT NULL)"""
        ).fetchall()
        updates: list[tuple[int, int, str, str, str]] = []
        for row in rows:
            self._check_refresh_interruption()
            has_audio = int(
                bool(row["has_audio"])
                or bool(row["audio_path"] and self._stored_path_exists(row["audio_path"]))
            )
            has_alignment = int(
                bool(row["has_alignment"])
                or bool(
                    row["alignment_path"]
                    and self._stored_path_exists(row["alignment_path"])
                )
            )
            if has_audio == row["has_audio"] and has_alignment == row["has_alignment"]:
                continue
            if has_alignment:
                stage = "complete"
            elif has_audio:
                stage = "audio"
            elif row["has_dialogue"]:
                stage = "dialogue"
            elif row["has_image"]:
                stage = "image"
            else:
                stage = "empty"
            updates.append(
                (has_audio, has_alignment, stage, row["collection"], row["uid"])
            )
        if updates:
            db.executemany(
                """UPDATE items SET has_audio=?,has_alignment=?,stage=?
                   WHERE collection=? AND uid=?""",
                updates,
            )
        return len(updates)

    def _backfill_audio_durations(self, db: sqlite3.Connection) -> int:
        """Populate legacy catalogue rows from their small dialogue metadata.

        New and rebuilt rows are handled in ``_merge_components``.  This
        one-time pass lets an existing 725 MB catalogue gain the aggregate
        without forcing all source JSONL files through a full rebuild.
        """

        marker = db.execute(
            "SELECT value FROM meta WHERE key='audio_duration_backfill'"
        ).fetchone()
        if marker is not None and marker[0] == "1":
            return 0

        rows = db.execute(
            """SELECT collection,uid,speech_dialogue_path,summary_json
               FROM items
               WHERE has_audio=1
                 AND json_extract(summary_json,'$.duration_seconds') IS NULL"""
        ).fetchall()
        updates: list[tuple[str, str, str]] = []
        unresolved = 0
        for row in rows:
            self._check_refresh_interruption()
            metadata = self._read_json_media(row["speech_dialogue_path"])
            duration = (
                _as_duration(metadata.get("duration_seconds", metadata.get("duration")))
                if isinstance(metadata, Mapping)
                else None
            )
            if duration is None:
                unresolved += 1
                continue
            try:
                summary = json.loads(row["summary_json"])
            except (TypeError, json.JSONDecodeError):
                summary = {}
            if not isinstance(summary, dict):
                summary = {}
            summary["duration_seconds"] = duration
            updates.append((_json_dumps(summary), row["collection"], row["uid"]))
        if updates:
            db.executemany(
                "UPDATE items SET summary_json=? WHERE collection=? AND uid=?",
                updates,
            )
        if unresolved == 0:
            db.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('audio_duration_backfill','1')"
            )
        return len(updates)

    @staticmethod
    def _iter_jsonl(
        path: Path,
        start_offset: int = 0,
    ) -> Iterator[tuple[int, int, dict[str, Any] | None, str | None]]:
        with path.open("rb") as stream:
            stream.seek(max(0, start_offset))
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    return
                length = len(line)
                if not line.endswith(b"\n"):
                    yield offset, length, None, "partial"
                    return
                if not line.strip():
                    yield offset, length, None, "blank"
                    continue
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError("JSON value is not an object")
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                    yield offset, length, None, str(exc)
                    continue
                yield offset, length, value, None

    @staticmethod
    def _insert_components(db: sqlite3.Connection, rows: Sequence[tuple[Any, ...]]) -> None:
        db.executemany(
            """INSERT OR REPLACE INTO components(
                   source_id,collection,uid,role,priority,offset_bytes,length_bytes,
                   source_uid,category,image_path,audio_path,alignment_path,
                   speech_dialogue_path,has_image,has_dialogue,has_audio,
                   has_alignment,eligible,search_text,summary_json
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _review_metadata_for_record(
        self, record: Mapping[str, Any], image_path: str | None, *, verify_image: bool = False
    ) -> dict[str, Any]:
        """Build review identity from the current source image, fail closed."""

        if not requires_review(record) and not verify_image:
            return review_metadata(record, image=image_path)

        if not image_path and record.get("kind") == "nonvisual":
            return {**review_metadata(record), "current_image_sha256": None,
                    "image_integrity_status": "not_applicable",
                    "review_image_size_bytes": None, "review_image_mtime_ns": None}

        declared_hash = record.get("image_sha256")
        current_hash: str | None = None
        image_size: int | None = None
        image_mtime_ns: int | None = None
        integrity = "unavailable"
        try:
            if not image_path:
                raise FileNotFoundError("review item has no image path")
            image = self._safe_media_path(image_path)
            stat = image.stat()
            current_hash = image_file_sha256(image)
            image_size = stat.st_size
            image_mtime_ns = stat.st_mtime_ns
            integrity = (
                "verified"
                if isinstance(declared_hash, str) and declared_hash == current_hash
                else "mismatch" if isinstance(declared_hash, str) else "computed"
            )
        except (OSError, ValueError):
            # Use a sentinel in the content fingerprint so a declared hash can
            # never make an unavailable image inherit an old approval.
            current_hash_for_fingerprint = "__image_unavailable__"
        else:
            current_hash_for_fingerprint = current_hash

        metadata = review_metadata(
            record,
            image=image_path,
            image_sha256=current_hash_for_fingerprint,
        )
        metadata.update(
            {
                "declared_image_sha256": declared_hash,
                "current_image_sha256": current_hash,
                "image_integrity_status": integrity,
                "review_image_size_bytes": image_size,
                "review_image_mtime_ns": image_mtime_ns,
            }
        )
        return metadata

    def _refresh_targeted_speech_roots(
        self, db: sqlite3.Connection
    ) -> dict[str, set[str]]:
        """Index small future counting/OCR speech roots by their source UID.

        The canonical 170k v3 root is intentionally handled by its asset
        manifest.  Only directories whose name explicitly contains counting,
        OCR or targeted are scanned here, so a refresh never walks historical
        speech_v2/v3 trees.  Each ``dialogue.json`` carries ``source_uid`` and
        ``task_mode`` as written by ``build_speech_dataset.py``.
        """

        discovered: dict[str, tuple[Path, Path]] = {}
        if self.data_root.is_dir():
            for root in self.data_root.iterdir():
                self._check_refresh_interruption()
                if not root.is_dir() or not root.name.startswith("speech"):
                    continue
                lowered = root.name.lower()
                if not any(marker in lowered for marker in ("count", "ocr", "target", "japanese_daily")):
                    continue
                stereo_root = root / "data_stereo"
                if not stereo_root.is_dir():
                    continue
                relative = root.relative_to(self.repo_root).as_posix()
                discovered[f"targeted_speech:{relative}"] = (root, stereo_root)

        # A media scan must not rebuild every dialogue/image in its collection.
        # Compare exact component rows and rebuild only changed source UIDs.
        changed: dict[str, set[str]] = {}
        existing = {
            row["source_id"]: row
            for row in db.execute(
                "SELECT source_id,path FROM sources WHERE role='targeted_speech_root'"
            )
        }
        for source_id in set(existing) - set(discovered):
            self._check_refresh_interruption()
            for row in db.execute("SELECT collection,uid FROM components WHERE source_id=?", (source_id,)):
                changed.setdefault(row["collection"], set()).add(row["uid"])
            db.execute("DELETE FROM sources WHERE source_id=?", (source_id,))

        for source_id, (root, stereo_root) in discovered.items():
            self._check_refresh_interruption()
            # ``lowered`` from the discovery loop may refer to a different
            # speech root by the time indexing starts.  Recompute it for this
            # root so normal dialogues inherit the correct counting/OCR
            # collection from their output directory.
            lowered = root.name.lower()
            relative = root.relative_to(self.repo_root).as_posix()
            db.execute(
                """INSERT INTO sources(source_id,collection,role,path,status,indexed_at)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(source_id) DO UPDATE SET
                     collection=excluded.collection,role=excluded.role,
                     path=excluded.path,status=excluded.status,
                     indexed_at=excluded.indexed_at""",
                (
                    source_id,
                    "targeted_tasks",
                    "targeted_speech_root",
                    relative,
                    "indexing",
                    _utc_now(),
                ),
            )
            old_components = {
                (row["uid"], row["role"]): tuple(row)
                for row in db.execute("SELECT * FROM components WHERE source_id=?", (source_id,))
            }
            db.execute("DELETE FROM components WHERE source_id=?", (source_id,))

            batch: list[tuple[Any, ...]] = []
            errors: list[str] = []
            scanned = accepted = 0
            try:
                entries = os.scandir(stereo_root)
            except OSError as exc:
                entries = None
                errors.append(str(exc))
            if entries is not None:
                with entries:
                    for entry in entries:
                        self._check_refresh_interruption()
                        if not entry.is_dir(follow_symlinks=False):
                            continue
                        scanned += 1
                        sample_dir = Path(entry.path)
                        metadata_path = sample_dir / "dialogue.json"
                        try:
                            with metadata_path.open("r", encoding="utf-8") as stream:
                                metadata = json.load(stream)
                            if not isinstance(metadata, Mapping):
                                raise ValueError("dialogue.json is not an object")
                        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                            if len(errors) < 10:
                                errors.append(f"{sample_dir.name}: {exc}")
                            continue

                        storage_uid = str(metadata.get("uid") or sample_dir.name)
                        task_mode = str(metadata.get("task_mode") or "").lower()
                        if not task_mode and "__task_" in storage_uid:
                            task_mode = storage_uid.rsplit("__task_", 1)[1].lower()
                        if task_mode == "japanese_ocr":
                            task_mode = "ocr"
                        collection = {
                            "counting": "counting_v4",
                            "ocr": "ocr_v4",
                            "ocr_document": "ocr_v4",
                        }.get(task_mode)
                        if task_mode == "normal":
                            requested_collection = metadata.get("dataset_collection")
                            if requested_collection in {
                                "visual_v3",
                                "counting_v4",
                                "ocr_v4",
                                "japanese_daily_v1",
                            }:
                                collection = str(requested_collection)
                            elif "count" in lowered:
                                collection = "counting_v4"
                            elif "ocr" in lowered:
                                collection = "ocr_v4"
                            elif "japanese_daily" in lowered:
                                collection = "japanese_daily_v1"
                        if collection is None:
                            continue
                        source_uid = str(
                            metadata.get("source_uid")
                            or storage_uid.rsplit("__task_", 1)[0]
                        )
                        audio = sample_dir / "stereo_dialogue.wav"
                        alignment = sample_dir / "assistant_dialogue.refined.json"
                        audio_path = self._normalise_stored_path(audio) if audio.is_file() else None
                        alignment_path = (
                            self._normalise_stored_path(alignment)
                            if alignment.is_file()
                            else None
                        )
                        speech_dialogue_path = self._normalise_stored_path(metadata_path)
                        summary = {
                            "task_mode": task_mode,
                            "storage_uid": storage_uid,
                            "duration_seconds": _as_duration(
                                metadata.get("duration_seconds", metadata.get("duration"))
                            ),
                            "user_voice": metadata.get("user_voice"),
                            "assistant_voice": metadata.get("assistant_voice"),
                        }
                        for key in (
                            "verification_status",
                            "auxiliary_task",
                            "auxiliary_kind",
                            "review_required",
                            "review_id",
                            "fingerprint",
                            "content_sha256",
                            "image_sha256",
                        ):
                            if key in metadata:
                                summary[key] = metadata[key]
                        batch.append(
                            (
                                source_id,
                                collection,
                                source_uid,
                                "targeted_speech",
                                30,
                                0,
                                0,
                                source_uid,
                                task_mode,
                                None,
                                audio_path,
                                alignment_path,
                                speech_dialogue_path,
                                0,
                                0,
                                int(audio_path is not None),
                                int(alignment_path is not None),
                                None,
                                f"{source_uid} {storage_uid} {task_mode}",
                                _json_dumps(summary),
                            )
                        )
                        accepted += 1
                        if len(batch) >= 2_000:
                            self._insert_components(db, batch)
                            batch.clear()
            if batch:
                self._insert_components(db, batch)
            new_components = {
                (row["uid"], row["role"]): tuple(row)
                for row in db.execute("SELECT * FROM components WHERE source_id=?", (source_id,))
            }
            for key in old_components.keys() | new_components.keys():
                before, after = old_components.get(key), new_components.get(key)
                if before != after:
                    # A changed collection assignment invalidates both sides.
                    for component in (before, after):
                        if component is not None:
                            changed.setdefault(component[1], set()).add(component[2])
            stat = stereo_root.stat()
            db.execute(
                """UPDATE sources SET mtime_ns=?,size_bytes=?,status=?,row_count=?,
                   ignored_partial=0,error_count=?,error=?,indexed_at=? WHERE source_id=?""",
                (
                    stat.st_mtime_ns,
                    scanned,
                    "ok_with_errors" if errors else "ok",
                    accepted,
                    len(errors),
                    "\n".join(errors) or None,
                    _utc_now(),
                    source_id,
                ),
            )
        return changed

    def _component_from_record(
        self,
        spec: _SourceSpec,
        record: Mapping[str, Any],
        offset: int,
        length: int,
    ) -> tuple[Any, ...] | None:
        role = spec.role
        uid: str | None
        source_uid: str | None = None
        category = ""
        image_path: str | None = None
        audio_path: str | None = None
        alignment_path: str | None = None
        speech_dialogue_path: str | None = None
        has_image = has_dialogue = has_audio = has_alignment = 0
        eligible: int | None = None
        summary: dict[str, Any] = {}

        if role == "asset_manifest":
            if record.get("type") != "sample":
                return None
            uid = str(record.get("sample_id") or "").strip() or None
            if uid is None:
                return None
            for asset in record.get("assets") or []:
                if not isinstance(asset, Mapping):
                    continue
                asset_path = self._normalise_stored_path(asset.get("path"))
                asset_role = asset.get("role")
                if asset_role == "image" and asset_path:
                    image_path, has_image = asset_path, 1
                elif asset_role == "audio" and asset_path:
                    audio_path, has_audio = asset_path, 1
                    for key in ("duration_seconds", "sample_rate", "channels"):
                        if key in asset:
                            summary[key] = asset[key]
                elif asset_role == "alignment" and asset_path:
                    alignment_path, has_alignment = asset_path, 1
                    summary["alignment_entries"] = asset.get("entries")
            category = ""
            summary["split"] = record.get("split")

        elif role == "visual_dialogue":
            uid = str(record.get("uid") or "").strip() or None
            if uid is None:
                return None
            source_uid = uid
            image_path = self._normalise_stored_path(record.get("image"))
            category = self._visual_category(image_path)
            has_image = int(bool(image_path))
            has_dialogue = int(isinstance(record.get("dialogue"), list))
            speech_dir = f"data/speech_v3/data_stereo/{uid}"
            audio_path = f"{speech_dir}/stereo_dialogue.wav"
            alignment_path = f"{speech_dir}/assistant_dialogue.refined.json"
            speech_dialogue_path = f"{speech_dir}/dialogue.json"
            summary = {
                "generator": record.get("generator"),
                "task_mode": record.get("task_mode", "normal"),
                "turns": len(record.get("dialogue") or []),
            }
            summary.update(self._review_metadata_for_record(record, image_path))

        elif role == "targeted_images":
            uid = str(record.get("candidate_uid") or record.get("uid") or "").strip() or None
            if uid is None:
                return None
            source_uid = str(record.get("source_photo_uid") or uid)
            image_path = self._normalise_stored_path(record.get("path") or record.get("image"))
            has_image = int(bool(image_path))
            capability = str(record.get("capability") or "")
            if spec.collection == "counting_v4":
                category = "counting"
                counts = [
                    _as_int(item.get("answer_count"))
                    for item in (record.get("reference_annotations") or [])
                    if isinstance(item, Mapping)
                ]
                eligible = int(any(value is not None and value >= 2 for value in counts))
                summary["answer_counts"] = [value for value in counts if value is not None]
            else:
                category = "ocr"
                refs = record.get("reference_annotations") or []
                eligible_refs: list[tuple[Mapping[str, Any], str, int]] = []
                for annotation in refs:
                    if not isinstance(annotation, Mapping):
                        continue
                    if annotation.get("kind") not in (None, "exact_transcription"):
                        continue
                    text = annotation.get("text")
                    if not isinstance(text, str) or not text.strip():
                        continue
                    compact = "".join(unicodedata.normalize("NFKC", text).split())
                    if 2 <= len(compact) <= 24:
                        eligible_refs.append((annotation, text, len(compact)))
                eligible = int(bool(eligible_refs))
                if refs and isinstance(refs[0], Mapping):
                    text = refs[0].get("text")
                    compact = (
                        "".join(unicodedata.normalize("NFKC", text).split())
                        if isinstance(text, str)
                        else ""
                    )
                    summary["reference_chars"] = len(compact) if compact else None
                    summary["orientation"] = refs[0].get("orientation")
            source = record.get("source")
            summary.update(
                {
                    "capability": capability,
                    "source_dataset": source.get("dataset") if isinstance(source, Mapping) else None,
                    "training_ready": record.get("training_ready"),
                }
            )

        elif role == "daily_image_bank":
            uid = str(record.get("uid") or record.get("source_uid") or "").strip() or None
            if uid is None:
                return None
            source_uid = str(record.get("source_uid") or uid)
            image_path = self._normalise_stored_path(
                record.get("path") or record.get("image")
            )
            has_image = int(bool(image_path))
            category = str(record.get("assigned_bucket") or "japanese_everyday")
            # accepted_images.jsonl contains records that already passed the
            # license, image-quality, and duplicate gates.  They are adopted
            # source images, but become complete multimodal training samples
            # only after dialogue, audio, and alignment are added.
            eligible = 1
            download = record.get("download")
            fingerprints = record.get("fingerprints")
            summary = {
                "source_dataset": record.get("source"),
                "title": record.get("title"),
                "source_url": record.get("source_url"),
                "license": record.get("license"),
                "license_url": record.get("license_url"),
                "artist": record.get("artist"),
                "credit": record.get("credit"),
                "collection_buckets": record.get("collection_buckets"),
                "collection_seeds": record.get("collection_seeds"),
                "task_routing": record.get("task_routing"),
                "source_width": record.get("source_width"),
                "source_height": record.get("source_height"),
                "decoded_width": (
                    download.get("decoded_width")
                    if isinstance(download, Mapping)
                    else None
                ),
                "decoded_height": (
                    download.get("decoded_height")
                    if isinstance(download, Mapping)
                    else None
                ),
                "file_sha256": (
                    fingerprints.get("file_sha256")
                    if isinstance(fingerprints, Mapping)
                    else None
                ),
                "collection_status": "approved_image_bank",
                "training_ready": False,
            }

        elif role == "counting_dialogue":
            uid = str(record.get("uid") or "").strip() or None
            if uid is None:
                return None
            source_uid = uid
            default_mode = "ocr" if spec.collection == "ocr_v4" else "counting"
            task_mode = str(record.get("task_mode") or default_mode).lower()
            category = task_mode if task_mode in {"normal", "counting", "ocr"} else default_mode
            image_path = self._normalise_stored_path(record.get("image"))
            has_image = int(bool(image_path))
            has_dialogue = int(isinstance(record.get("dialogue"), list))
            grounding = record.get("grounding")
            summary = {
                "generator": record.get("generator"),
                "task_mode": task_mode,
                "turns": len(record.get("dialogue") or []),
                "answer_count": (
                    grounding.get("answer_count") if isinstance(grounding, Mapping) else None
                ),
                "question_en": (
                    grounding.get("question_en") if isinstance(grounding, Mapping) else None
                ),
            }
            summary.update(self._review_metadata_for_record(record, image_path))

        elif role == "nonvisual_dialogue":
            uid = str(record.get("uid") or "").strip() or None
            if uid is None:
                return None
            source_uid = uid
            category = "nonvisual"
            has_dialogue = int(isinstance(record.get("dialogue"), list))
            speech_dir = f"data/speech_nonvisual_v1/data_stereo/{uid}"
            audio_path = f"{speech_dir}/stereo_dialogue.wav"
            alignment_path = f"{speech_dir}/assistant_dialogue.refined.json"
            speech_dialogue_path = f"{speech_dir}/dialogue.json"
            # Unlike visual-v3 there is no asset manifest; existence is checked
            # while rebuilding (and again when opening a detail record).
            summary = {
                "generator": record.get("generator"),
                "kind": record.get("kind", "nonvisual"),
                "topic": record.get("topic"),
                "angle": record.get("angle"),
                "scene": record.get("scene"),
                "turns": len(record.get("dialogue") or []),
            }
        else:  # pragma: no cover - guarded by the fixed source table
            return None

        search_values = [uid, source_uid, category, image_path]
        for key in (
            "generator",
            "topic",
            "source_dataset",
            "title",
            "license",
            "artist",
            "question_en",
            "verification_status",
            "auxiliary_kind",
            "review_id",
        ):
            search_values.append(summary.get(key))
        search_text = " ".join(str(value) for value in search_values if value)
        return (
            spec.source_id,
            spec.collection,
            uid,
            role,
            spec.priority,
            offset,
            length,
            source_uid,
            category,
            image_path,
            audio_path,
            alignment_path,
            speech_dialogue_path,
            has_image,
            has_dialogue,
            has_audio,
            has_alignment,
            eligible,
            search_text,
            _json_dumps(summary),
        )

    def _rebuild_collection(self, db: sqlite3.Connection, collection: str) -> None:
        db.execute("DELETE FROM items WHERE collection=?", (collection,))
        cursor = db.execute(
            """SELECT * FROM components WHERE collection=?
               ORDER BY uid, priority, source_id""",
            (collection,),
        )
        current_uid: str | None = None
        group: list[sqlite3.Row] = []
        output: list[tuple[Any, ...]] = []
        for row in cursor:
            self._check_refresh_interruption()
            if current_uid is not None and row["uid"] != current_uid:
                output.append(self._merge_components(collection, current_uid, group))
                if len(output) >= 2_000:
                    self._insert_items(db, output)
                    output.clear()
                group = []
            current_uid = row["uid"]
            group.append(row)
        if current_uid is not None:
            output.append(self._merge_components(collection, current_uid, group))
        if output:
            self._insert_items(db, output)

    def _rebuild_items_for_uids(
        self, db: sqlite3.Connection, collection: str, uids: Iterable[str]
    ) -> None:
        """Rebuild only records appended to a live image-bank manifest."""

        ordered = sorted(set(uids))
        for index in range(0, len(ordered), 500):
            self._check_refresh_interruption()
            chunk = ordered[index : index + 500]
            if not chunk:
                continue
            db.executemany(
                "DELETE FROM items WHERE collection=? AND uid=?",
                ((collection, uid) for uid in chunk),
            )
            placeholders = ",".join("?" for _ in chunk)
            cursor = db.execute(
                f"""SELECT * FROM components
                    WHERE collection=? AND uid IN ({placeholders})
                    ORDER BY uid,priority,source_id""",
                [collection, *chunk],
            )
            current_uid: str | None = None
            group: list[sqlite3.Row] = []
            output: list[tuple[Any, ...]] = []
            for row in cursor:
                self._check_refresh_interruption()
                if current_uid is not None and row["uid"] != current_uid:
                    output.append(
                        self._merge_components(collection, current_uid, group)
                    )
                    group = []
                current_uid = row["uid"]
                group.append(row)
            if current_uid is not None:
                output.append(self._merge_components(collection, current_uid, group))
            if output:
                self._insert_items(db, output)

    @staticmethod
    def _insert_items(db: sqlite3.Connection, rows: Sequence[tuple[Any, ...]]) -> None:
        db.executemany(
            """INSERT INTO items(
                   collection,uid,source_uid,category,directory,image_path,audio_path,
                   alignment_path,speech_dialogue_path,metadata_source,metadata_offset,
                   metadata_length,dialogue_source,dialogue_offset,dialogue_length,
                   has_image,has_dialogue,has_audio,has_alignment,eligible,stage,
                   search_text,summary_json,verification_status,auxiliary_kind,has_both_auxiliary_kinds,
                   review_required,review_id,review_status,dataset_usable
               ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )

    def _merge_components(
        self, collection: str, uid: str, rows: Sequence[sqlite3.Row]
    ) -> tuple[Any, ...]:
        source_uid = uid
        category = {
            "counting_v4": "counting",
            "ocr_v4": "ocr",
            "nonvisual_v1": "nonvisual",
        }.get(collection, "other")
        image_path = audio_path = alignment_path = speech_dialogue_path = None
        metadata_locator: tuple[str, int, int] | None = None
        dialogue_locator: tuple[str, int, int] | None = None
        has_image = has_dialogue = has_audio = has_alignment = 0
        eligible: int | None = None
        search: list[str] = []
        summary: dict[str, Any] = {}
        dialogue_review: dict[str, Any] | None = None
        review_fields = {
            "verification_status",
            "auxiliary_task",
            "auxiliary_kind",
            "review_required",
            "review_id",
            "fingerprint",
            "content_sha256",
            "image_sha256",
            "declared_image_sha256",
            "current_image_sha256",
            "image_integrity_status",
            "review_image_size_bytes",
            "review_image_mtime_ns",
        }

        metadata_roles = {
            "visual_dialogue",
            "targeted_images",
            "daily_image_bank",
            "nonvisual_dialogue",
        }
        for row in rows:
            source_uid = row["source_uid"] or source_uid
            category = row["category"] or category
            # Keep the original image referenced by the dialogue/target manifest.
            # The asset manifest also contains a copied image inside each speech
            # UID directory; allowing it to win would create 170k apparent
            # directories in the browser tree.
            image_path = image_path or row["image_path"]
            audio_path = row["audio_path"] or audio_path
            alignment_path = row["alignment_path"] or alignment_path
            speech_dialogue_path = row["speech_dialogue_path"] or speech_dialogue_path
            has_image = max(has_image, row["has_image"])
            has_dialogue = max(has_dialogue, row["has_dialogue"])
            has_audio = max(has_audio, row["has_audio"])
            has_alignment = max(has_alignment, row["has_alignment"])
            if row["eligible"] is not None:
                eligible = row["eligible"]
            search.append(row["search_text"])
            try:
                row_summary = json.loads(row["summary_json"])
                summary.update(row_summary)
                if row["has_dialogue"]:
                    # The highest-priority generated dialogue is authoritative.
                    # Speech metadata may belong to an older generation and
                    # must never carry its approval onto replacement content.
                    dialogue_review = {
                        key: row_summary[key]
                        for key in review_fields
                        if key in row_summary
                    }
            except (TypeError, json.JSONDecodeError):
                pass
            locator = (row["source_id"], row["offset_bytes"], row["length_bytes"])
            if row["role"] in metadata_roles and metadata_locator is None:
                metadata_locator = locator
            if row["has_dialogue"]:
                # Higher-priority dialogue shards intentionally win duplicates.
                dialogue_locator = locator

        if dialogue_review is not None:
            for key in review_fields:
                if key in dialogue_review:
                    summary[key] = dialogue_review[key]
                else:
                    summary.pop(key, None)

        # The asset manifest covers 170,021 visual-v3 rows.  The remaining 120
        # CER-excluded rows still have deterministic files on disk, so stat only
        # those rows rather than walking 170k directories.
        if audio_path and not has_audio:
            has_audio = int(self._stored_path_exists(audio_path))
        if alignment_path and not has_alignment:
            has_alignment = int(self._stored_path_exists(alignment_path))

        duration_seconds = _as_duration(
            summary.get("duration_seconds", summary.get("duration"))
        )
        if duration_seconds is None and has_audio and speech_dialogue_path:
            speech_metadata = self._read_json_media(speech_dialogue_path)
            if isinstance(speech_metadata, Mapping):
                duration_seconds = _as_duration(
                    speech_metadata.get(
                        "duration_seconds", speech_metadata.get("duration")
                    )
                )
        if duration_seconds is not None:
            summary["duration_seconds"] = duration_seconds

        verification_status = summary.get("verification_status")
        auxiliary_kind = summary.get("auxiliary_kind")
        review_required = int(bool(summary.get("review_required")))
        review_id = summary.get("review_id") or summary.get("fingerprint")
        review_status = "pending" if review_required else "not_required"
        dataset_usable = int(not review_required)

        directory = self._display_directory(collection, image_path)
        if has_alignment:
            stage = "complete"
        elif has_audio:
            stage = "audio"
        elif has_dialogue:
            stage = "dialogue"
        elif has_image:
            stage = "image"
        else:
            stage = "empty"

        return (
            collection,
            uid,
            source_uid,
            category,
            directory,
            image_path,
            audio_path,
            alignment_path,
            speech_dialogue_path,
            *(metadata_locator or (None, None, None)),
            *(dialogue_locator or (None, None, None)),
            has_image,
            has_dialogue,
            has_audio,
            has_alignment,
            eligible,
            stage,
            " ".join(search),
            _json_dumps(summary),
            verification_status,
            auxiliary_kind,
            int(_has_both_auxiliary_kinds(summary)),
            review_required,
            review_id,
            review_status,
            dataset_usable,
        )

    # ------------------------------------------------------------------
    # Public queries
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._connect() as db:
            last = db.execute("SELECT value FROM meta WHERE key='last_refresh'").fetchone()
            last_changed = db.execute(
                "SELECT value FROM meta WHERE key='last_changed'"
            ).fetchone()
            revision = db.execute(
                "SELECT value FROM meta WHERE key='catalog_revision'"
            ).fetchone()
            source_rows = db.execute(
                "SELECT * FROM sources ORDER BY collection,source_id"
            ).fetchall()
            collections = [
                dict(row)
                for row in db.execute(
                    "SELECT collection,COUNT(*) AS items FROM items GROUP BY collection ORDER BY collection"
                )
            ]
        sources = [dict(row) for row in source_rows]
        errors = [
            {"source_id": row["source_id"], "status": row["status"], "error": row["error"]}
            for row in sources
            if row["status"] not in {"ok", "missing"} or row["error"]
        ]
        try:
            ai_suggestion_count = self.ai_suggestion_store.count()
        except AISuggestionIntegrityError:
            ai_suggestion_count = 0
        return {
            "ready": any(row["items"] for row in collections),
            "db_path": str(self.db_path),
            "last_refresh": last[0] if last else None,
            "last_changed": last_changed[0] if last_changed else None,
            "catalog_revision": int(revision[0]) if revision else 0,
            "ai_suggestion_count": ai_suggestion_count,
            "sources": sources,
            "errors": errors,
            "collections": collections,
        }

    def dashboard(
        self, collection: str | None = None, category: str | None = None
    ) -> dict[str, Any]:
        where, params = self._where(collection=collection, category=category)
        with self._connect() as db:
            total = db.execute(
                f"""SELECT COUNT(*) AS items,
                           COALESCE(SUM(has_image),0) AS images,
                           COALESCE(SUM(has_dialogue),0) AS dialogues,
                           COALESCE(SUM(has_audio),0) AS audio,
                           COALESCE(SUM(has_alignment),0) AS alignments,
                           COALESCE(SUM(CASE WHEN has_audio=1 THEN
                               COALESCE(
                                   CAST(json_extract(summary_json,'$.duration_seconds') AS REAL),
                                   CAST(json_extract(summary_json,'$.duration') AS REAL),0.0
                               ) ELSE 0.0 END),0.0) AS audio_duration_seconds,
                           COALESCE(SUM(CASE WHEN has_audio=1 AND COALESCE(
                               json_extract(summary_json,'$.duration_seconds'),
                               json_extract(summary_json,'$.duration')
                           ) IS NOT NULL THEN 1 ELSE 0 END),0) AS audio_duration_known,
                           COALESCE(SUM(CASE WHEN eligible=1 THEN 1 ELSE 0 END),0) AS eligible
                    FROM items {where}""",
                params,
            ).fetchone()
            stages = {
                row["stage"]: row["count"]
                for row in db.execute(
                    f"SELECT stage,COUNT(*) AS count FROM items {where} GROUP BY stage",
                    params,
                )
            }
            collections: list[dict[str, Any]] = []
            for row in db.execute(
                f"""SELECT collection,COUNT(*) AS items,SUM(has_image) AS images,
                           SUM(has_dialogue) AS dialogues,SUM(has_audio) AS audio,
                           SUM(has_alignment) AS alignments,
                           COALESCE(SUM(CASE WHEN has_audio=1 THEN
                               COALESCE(
                                   CAST(json_extract(summary_json,'$.duration_seconds') AS REAL),
                                   CAST(json_extract(summary_json,'$.duration') AS REAL),0.0
                               ) ELSE 0.0 END),0.0) AS audio_duration_seconds,
                           COALESCE(SUM(CASE WHEN has_audio=1 AND COALESCE(
                               json_extract(summary_json,'$.duration_seconds'),
                               json_extract(summary_json,'$.duration')
                           ) IS NOT NULL THEN 1 ELSE 0 END),0) AS audio_duration_known,
                           SUM(CASE WHEN eligible=1 THEN 1 ELSE 0 END) AS eligible
                    FROM items {where} GROUP BY collection ORDER BY collection""",
                params,
            ):
                value = dict(row)
                value["label"] = COLLECTIONS.get(row["collection"], row["collection"])
                if row["collection"] == "counting_v4":
                    value["planned"] = row["eligible"] or 0
                collections.append(value)
            categories = [
                dict(row)
                for row in db.execute(
                    f"""SELECT category AS id,category AS label,COUNT(*) AS count
                        FROM items {where} GROUP BY category ORDER BY category""",
                    params,
                )
            ]
            review = dict(db.execute(f"""SELECT
                COALESCE(SUM(review_required=1 AND has_dialogue=1),0) AS required_total,
                COALESCE(SUM(review_required=1 AND review_status='pending' AND has_dialogue=1),0) AS required_pending,
                COALESCE(SUM(review_status='approved' AND has_dialogue=1),0) AS certified_total,
                COALESCE(SUM(CASE WHEN review_status='approved' THEN json_type(summary_json,'$.human_revision')='object' ELSE 0 END),0) AS edited_certified_total,
                COALESCE(SUM(CASE WHEN review_status='approved' THEN ({REGENERATION_PENDING_SQL}) ELSE 0 END),0) AS regeneration_pending,
                COALESCE(SUM(CASE WHEN review_status='approved' THEN ({REGENERATION_PENDING_SQL}) AND has_audio=0 ELSE 0 END),0) AS regeneration_audio_pending,
                COALESCE(SUM(CASE WHEN review_status='approved' THEN ({REGENERATION_PENDING_SQL}) AND has_alignment=0 ELSE 0 END),0) AS regeneration_alignment_pending,
                COALESCE(SUM(review_status='rejected'),0) AS rejected_total,
                COALESCE(SUM(review_required=0 AND review_status='not_required' AND has_dialogue=1),0) AS optional_unreviewed,
                COALESCE(SUM(review_status='approved' AND has_audio=1 AND has_alignment=1),0) AS certified_complete,
                COALESCE(SUM(review_status='approved' AND has_audio=0),0) AS certified_audio_pending
                FROM items {where}""", params).fetchone())
        totals = dict(total)
        totals["logical_uid_count"] = totals["items"]

        # Keep the legacy all-collection totals above for API compatibility, but
        # expose visual and non-visual pipelines independently.  A non-visual
        # dialogue has no image by design, so combining it with the visual
        # pipeline makes the dialogue/audio/alignment cards look further along
        # than image-backed data actually is.
        integer_metric_keys = (
            "items",
            "images",
            "dialogues",
            "audio",
            "alignments",
            "audio_duration_known",
            "eligible",
        )

        def grouped_totals(nonvisual: bool) -> dict[str, int | float]:
            values: dict[str, int | float] = {
                **{key: 0 for key in integer_metric_keys},
                "audio_duration_seconds": 0.0,
            }
            for row in collections:
                collection_id = str(row.get("collection", "")).replace("-", "_")
                is_nonvisual = collection_id.startswith("nonvisual")
                if is_nonvisual != nonvisual:
                    continue
                for key in integer_metric_keys:
                    values[key] += int(row.get(key) or 0)
                values["audio_duration_seconds"] += float(
                    row.get("audio_duration_seconds") or 0.0
                )
            values["logical_uid_count"] = values["items"]
            return values

        visual_totals = grouped_totals(nonvisual=False)
        nonvisual_totals = grouped_totals(nonvisual=True)
        groups = {
            "visual": {
                "id": "visual",
                "label": "画像付きデータ",
                "collection_ids": [
                    row["collection"]
                    for row in collections
                    if not str(row["collection"]).replace("-", "_").startswith("nonvisual")
                ],
                **visual_totals,
            },
            "nonvisual": {
                "id": "nonvisual",
                "label": "非画像対話データ",
                "collection_ids": [
                    row["collection"]
                    for row in collections
                    if str(row["collection"]).replace("-", "_").startswith("nonvisual")
                ],
                **nonvisual_totals,
            },
        }
        return {
            "totals": totals,
            "review": review,
            "groups": groups,
            "stages": stages,
            "collections": collections,
            "categories": categories,
            "filters": {"collection": collection, "category": category},
            "counting_note": "eligible/planned は answer_count >= 2 の論理 UID 数",
        }

    def tree(self) -> dict[str, Any]:
        with self._connect() as db:
            rows = db.execute(
                """SELECT collection,category,directory,COUNT(*) AS count,
                          SUM(has_image) AS images,SUM(has_dialogue) AS dialogues,
                          SUM(has_audio) AS audio,SUM(has_alignment) AS alignments,
                          SUM(CASE WHEN eligible=1 THEN 1 ELSE 0 END) AS eligible
                   FROM items GROUP BY collection,category,directory
                   ORDER BY collection,category,directory"""
            ).fetchall()
        by_collection: dict[str, dict[str, Any]] = {}
        by_category: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            collection = row["collection"]
            category = row["category"]
            cnode = by_collection.setdefault(
                collection,
                {
                    "id": collection,
                    "label": COLLECTIONS.get(collection, collection),
                    "count": 0,
                    "images": 0,
                    "dialogues": 0,
                    "audio": 0,
                    "alignments": 0,
                    "eligible": 0,
                    "generated": 0,
                    "filter": {"collection": collection},
                    "categories": [],
                },
            )
            key = (collection, category)
            cat = by_category.get(key)
            if cat is None:
                cat = {
                    "id": f"{collection}:{category}",
                    "label": category,
                    "count": 0,
                    "images": 0,
                    "dialogues": 0,
                    "audio": 0,
                    "alignments": 0,
                    "eligible": 0,
                    "generated": 0,
                    "filter": {"collection": collection, "category": category},
                    "directories": [],
                }
                by_category[key] = cat
                cnode["categories"].append(cat)
            count = int(row["count"] or 0)
            cnode["count"] += count
            cat["count"] += count
            for field in ("images", "dialogues", "audio", "alignments", "eligible"):
                value = int(row[field] or 0)
                cnode[field] += value
                cat[field] += value
            cnode["generated"] = cnode["dialogues"]
            cat["generated"] = cat["dialogues"]
            cat["directories"].append(
                {
                    "id": f"{collection}:{category}:{row['directory']}",
                    "label": row["directory"],
                    "count": count,
                    "images": int(row["images"] or 0),
                    "dialogues": int(row["dialogues"] or 0),
                    "audio": int(row["audio"] or 0),
                    "alignments": int(row["alignments"] or 0),
                    "eligible": int(row["eligible"] or 0),
                    "generated": int(row["dialogues"] or 0),
                    "filter": {
                        "collection": collection,
                        "category": category,
                        "directory": row["directory"],
                    },
                }
            )
        return {"collections": list(by_collection.values())}

    def list_items(
        self,
        collection: str | None = None,
        category: str | None = None,
        stage: str | None = None,
        query: str | None = None,
        directory: str | None = None,
        has_image: bool | None = None,
        has_dialogue: bool | None = None,
        has_audio: bool | None = None,
        has_alignment: bool | None = None,
        review_status: str | None = None,
        review_required: bool | None = None,
        auxiliary_kind: str | None = None,
        ai_status: str | None = None,
        page: int = 1,
        limit: int = 50,
        offset: int | None = None,
    ) -> dict[str, Any]:
        page = max(1, int(page))
        limit = min(200, max(1, int(limit)))
        where, params = self._where(
            collection,
            category,
            stage,
            query,
            directory=directory,
            has_image=has_image,
            has_dialogue=has_dialogue,
            has_audio=has_audio,
            has_alignment=has_alignment,
            review_status=review_status,
            review_required=review_required,
            auxiliary_kind=auxiliary_kind,
        )
        if offset is None:
            offset = (page - 1) * limit
        else:
            offset = max(0, int(offset))
            page = offset // limit + 1
        with self._connect() as db:
            where = self._with_ai_filter(db, where, ai_status)
            items_source = self._query_items(ai_status)
            total = db.execute(f"SELECT COUNT(*) FROM {items_source} {where}", params).fetchone()[0]
            rows = db.execute(
                f"""SELECT items.* FROM {items_source} {where}
                    ORDER BY collection,category,uid LIMIT ? OFFSET ?""",
                [*params, limit, offset],
            ).fetchall()
        return {
            "items": [self._summary_from_row(row) for row in rows],
            "page": page,
            "limit": limit,
            "total": total,
            "pages": math.ceil(total / limit) if total else 0,
            "filters": {
                "collection": collection,
                "category": category,
                "stage": stage,
                "query": query,
                "directory": directory,
                "review_status": review_status,
                "review_required": review_required,
                "auxiliary_kind": auxiliary_kind,
                "ai_status": ai_status,
            },
        }

    def get_item(
        self, token: str, filters: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        collection, uid = self._decode_token(token)
        # Validate exact bytes without taking the background writer's lock.
        # Ordinary unchanged reads stay on the WAL snapshot; changed/stale
        # content still takes the serialized fail-closed update path below.
        needs_update = False
        with self._connect() as db:
            selected = db.execute("SELECT * FROM items WHERE collection=? AND uid=?", (collection, uid)).fetchone()
            if selected is not None and selected["has_dialogue"]:
                summary, changed = self._refresh_review_identity(db, selected, force_hash=True, persist=False)
                status, usable = self._review_state(bool(selected["review_required"]),
                    self.review_store.decision_for(summary.get("review_id")), summary.get("image_integrity_status"))
                needs_update = changed or status != selected["review_status"] or usable != selected["dataset_usable"]
        if needs_update:
            with self._interactive_write(), self._connect() as db:
                selected = db.execute("SELECT * FROM items WHERE collection=? AND uid=?", (collection, uid)).fetchone()
                if selected is not None and selected["has_dialogue"]:
                    summary, _ = self._refresh_review_identity(db, selected, force_hash=True)
                    status, usable = self._review_state(bool(selected["review_required"]),
                        self.review_store.decision_for(summary.get("review_id")), summary.get("image_integrity_status"))
                    db.execute("UPDATE items SET review_status=?,dataset_usable=? WHERE collection=? AND uid=?",
                               (status, usable, collection, uid))
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM items WHERE collection=? AND uid=?", (collection, uid)
            ).fetchone()
            if row is None:
                raise KeyError("unknown dataset item")

        detail = self._summary_from_row(row)
        metadata = self._read_locator(
            row["metadata_source"], row["metadata_offset"], row["metadata_length"]
        )
        dialogue_record = self._read_locator(
            row["dialogue_source"], row["dialogue_offset"], row["dialogue_length"]
        )
        summary = json.loads(row["summary_json"] or "{}")
        revision = self.review_store.latest_revisions().get(summary.get("source_review_id"))
        if revision and revision["fingerprint"] == row["review_id"]:
            dialogue_record = revision["revised_record"]
        dialogue = []
        if isinstance(dialogue_record, Mapping):
            dialogue = self._normalise_dialogue(dialogue_record.get("dialogue"))
        speech_metadata = self._read_json_media(row["speech_dialogue_path"])
        if not dialogue and isinstance(speech_metadata, Mapping):
            dialogue = self._normalise_dialogue(speech_metadata.get("segments"))
        alignment = self._normalise_alignment(self._read_json_media(row["alignment_path"]))

        review = dict(detail.get("review") or {})
        if row["review_id"]:
            event = self.review_store.decision_for(row["review_id"])
            if event is not None:
                review.update(
                    {
                        "status": row["review_status"],
                        "decision": event.get("decision"),
                        "note": event.get("note", ""),
                        "reviewed_at": event.get("reviewed_at"),
                        "reviewer": event.get("reviewer"),
                    }
                )

        availability = dict(detail["availability"])
        for kind, column in (("image", "image_path"), ("audio", "audio_path")):
            availability[kind] = bool(row[column] and self._stored_path_exists(row[column]))
        availability["alignment"] = bool(
            row["alignment_path"] and self._stored_path_exists(row["alignment_path"])
        )
        availability["dialogue"] = bool(dialogue)
        detail.update(
            {
                "availability": availability,
                "dialogue": dialogue,
                "metadata": self._without_dialogue(metadata),
                "speech_metadata": speech_metadata,
                "alignment": alignment,
                "review": review,
                "human_revision": summary.get("human_revision"),
                "ai_suggestion": None,
                "navigation": self._navigation(row, filters or {}),
            }
        )
        suggestion = self._active_ai_suggestion(row, summary)
        if suggestion is not None:
            # Suggestions are a display-only overlay. The content/review ID
            # continues to identify the original until a human saves approval.
            detail["original_dialogue"] = dialogue
            detail["original_auxiliary_task"] = detail.get("auxiliary_task")
            detail["dialogue"] = self._normalise_dialogue(suggestion["dialogue"])
            if suggestion.get("auxiliary_task") is not None:
                detail["auxiliary_task"] = suggestion["auxiliary_task"]
            detail["ai_suggestion"] = {
                key: suggestion.get(key) for key in (
                    "suggestion_id", "model", "created_at", "changes", "concerns", "verdict",
                )
            }
        return detail

    def _active_ai_suggestion(self, row: sqlite3.Row, summary: Mapping[str, Any]) -> dict[str, Any] | None:
        """Offer AI drafts only on untouched, required, exact pending content."""
        if (not row["review_required"] or row["review_status"] != "pending"
                or not row["has_dialogue"] or summary.get("human_revision")
                or not summary.get("current_image_sha256")
                or summary.get("image_integrity_status") == "unavailable"):
            return None
        if (row["collection"], row["uid"]) in human_touched_uids(self.review_store):
            return None
        try:
            return self.ai_suggestion_store.get_suggestion(
                collection=row["collection"], uid=row["uid"],
                base_fingerprint=summary.get("review_id") or row["review_id"],
                image_sha256=summary["current_image_sha256"],
            )
        except AISuggestionIntegrityError:
            logging.getLogger(__name__).warning("Ignoring invalid AI suggestion for %s/%s", row["collection"], row["uid"])
            return None

    def edit_dialogue(self, token: str, *, fingerprint: str, dialogue: list,
                      auxiliary_task: Mapping[str, Any] | None = None,
                      note: str = "", reviewer: str | None = None,
                      ai_suggestion_id: str | None = None) -> dict[str, Any]:
        """Save a human correction and its approval in a single durable event."""
        if not isinstance(note, str) or len(note.strip()) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError("確認メモは1000文字以内で入力してください。")
        collection, uid = self._decode_token(token)
        with self._interactive_write():
            with self._connect() as db:
                row = db.execute("SELECT * FROM items WHERE collection=? AND uid=?", (collection, uid)).fetchone()
                if row is None:
                    raise KeyError("unknown dataset item")
                if not row["has_dialogue"]:
                    raise ReviewConflictError("There is no dialogue to edit.")
                summary, changed = self._refresh_review_identity(db, row, force_hash=True)
                if changed:
                    db.commit()
                if summary.get("image_integrity_status") == "unavailable" or fingerprint != summary.get("review_id"):
                    raise ReviewConflictError("画像または対話が変更されました。再読み込みしてから保存してください。")
                if ai_suggestion_id is not None:
                    suggestion = self._active_ai_suggestion(row, summary)
                    if suggestion is None or suggestion.get("suggestion_id") != ai_suggestion_id:
                        raise ReviewConflictError("AI修正案または人手確認の状態が変更されました。再読み込みして確認してください。")
                original = self._read_locator(row["dialogue_source"], row["dialogue_offset"], row["dialogue_length"])
                base = summary.get("source_review_id", fingerprint)
                previous = self.review_store.latest_revisions().get(base)
                revised = copy.deepcopy(previous["revised_record"] if previous else original)
                if not isinstance(revised, dict):
                    raise ReviewConflictError("The source dialogue is unavailable.")
                turns = revised.get("dialogue", [])
                if not isinstance(dialogue, list) or len(dialogue) != len(turns) or not turns:
                    raise ValueError("発話数を変更せず、各発話の本文を編集してください。")
                for old, new in zip(turns, dialogue):
                    if not isinstance(new, Mapping) or not isinstance(new.get("text"), str) or not new["text"].strip():
                        raise ValueError("空の発話は保存できません。")
                    if len(new["text"]) > 10000:
                        raise ValueError("1発話は10000文字以内にしてください。")
                    old["text"] = new["text"].strip()
                auxiliary = revised.get("auxiliary_task")
                if auxiliary_task is not None:
                    if not isinstance(auxiliary_task, Mapping) or not isinstance(auxiliary, dict):
                        raise ValueError("invalid auxiliary_task")
                    # Kind, grounding provenance and turn positions are not client-editable.
                    if auxiliary.get("kind") == "counting" and "answer_count" in auxiliary:
                        count = auxiliary_task.get("answer_count", auxiliary.get("answer_count"))
                        if type(count) is not int or not 1 <= count <= 1000:
                            raise ValueError("数の正解は1〜1000の整数にしてください。")
                        auxiliary["answer_count"] = count
                    key = "counted_target" if auxiliary.get("kind") == "counting" else "recognized_text"
                    if key in auxiliary and key in auxiliary_task:
                        value = auxiliary_task[key]
                        if not isinstance(value, str) or not value.strip() or len(value) > 1000:
                            raise ValueError("対象・読み取り文字は1〜1000文字で入力してください。")
                        auxiliary[key] = value.strip()
                revised["image_sha256"] = summary["current_image_sha256"]
                new_id = content_fingerprint(revised)
                revised["human_revision"] = {
                    "base_fingerprint": base, "parent_fingerprint": fingerprint,
                    "review_id": new_id, "edited_at": _utc_now(), "reviewer": reviewer,
                    "collection": collection,
                    "had_media": bool(row["has_audio"] or row["has_alignment"]
                        or (previous and previous["revised_record"].get("human_revision", {}).get("had_media"))
                        or any(path and self._stored_path_exists(path)
                               for path in (self._summary_value(row["summary_json"], "source_media_paths") or []))),
                }
                event = self.review_store.append(fingerprint=new_id, decision="approved", uid=uid,
                    collection=collection, note=note, reviewer=reviewer,
                    task_mode=str(revised.get("task_mode") or "normal"),
                    revised_record=revised if new_id != fingerprint else None,
                    base_fingerprint=base, parent_fingerprint=fingerprint,
                    # The preview check above is advisory. Recheck under the
                    # ledger's writer lock even when the text is unchanged.
                    require_unreviewed=ai_suggestion_id is not None)
                fresh = db.execute("SELECT * FROM items WHERE collection=? AND uid=?", (collection, uid)).fetchone()
                self._refresh_review_identity(db, fresh, force_hash=True)
                updated = db.execute("""UPDATE items SET review_status='approved',dataset_usable=1
                    WHERE collection=? AND uid=? AND review_id=?""", (collection, uid, new_id))
                if not updated.rowcount:
                    db.commit()  # The new image/content remains pending, never approved by an old edit.
                    raise ReviewConflictError("保存中に画像または元データが変更されました。再読み込みして確認してください。")
                db.execute("UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='catalog_revision'")
                db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES('last_changed',?)", (_utc_now(),))
        return {"ok": True, "item": self.get_item(token), "event": event, "content_changed": new_id != fingerprint}

    def review_item(
        self,
        token: str,
        *,
        fingerprint: str,
        decision: str,
        note: str = "",
        reviewer: str | None = None,
    ) -> dict[str, Any]:
        """Append a decision for the current exact content and update its row."""

        if decision not in REVIEW_DECISIONS:
            raise ValueError("decision must be pending, approved, or rejected")
        if not isinstance(note, str):
            raise ValueError("note must be a string")
        if len(note.strip()) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError(
                f"note must be at most {MAX_REVIEW_NOTE_CHARS} characters"
            )
        collection, uid = self._decode_token(token)
        with self._interactive_write():
            with self._connect() as db:
                row = db.execute(
                    "SELECT * FROM items WHERE collection=? AND uid=?",
                    (collection, uid),
                ).fetchone()
                if row is None:
                    raise KeyError("unknown dataset item")
                if not row["has_dialogue"]:
                    raise ReviewConflictError(
                        "There is no dialogue to review."
                    )
                summary, identity_changed = self._refresh_review_identity(
                    db, row, force_hash=True
                )
                if identity_changed:
                    # Persist fail-closed pending state even though the stale
                    # request below returns HTTP 409.
                    db.commit()
                current_fingerprint = summary.get("review_id") or row["review_id"]
                if summary.get("image_integrity_status") == "unavailable":
                    raise ReviewConflictError(
                        "The current source image cannot be verified."
                    )
                if not fingerprint or fingerprint != current_fingerprint:
                    raise ReviewConflictError(
                        "The dialogue changed after it was opened. Reload before reviewing."
                    )

                event = self.review_store.append(
                    fingerprint=current_fingerprint,
                    decision=decision,
                    uid=uid,
                    collection=collection,
                    note=note,
                    reviewer=reviewer,
                    task_mode=str(summary.get("task_mode") or "normal"),
                )
                status, usable = self._review_state(bool(row["review_required"]), event, summary.get("image_integrity_status"))
                db.execute(
                    """UPDATE items SET review_status=?,dataset_usable=?
                       WHERE collection=? AND uid=? AND review_id=?""",
                    (status, usable, collection, uid, current_fingerprint),
                )
                changed_at = _utc_now()
                revision_row = db.execute(
                    "SELECT value FROM meta WHERE key='catalog_revision'"
                ).fetchone()
                try:
                    revision = int(revision_row[0]) + 1 if revision_row else 1
                except (TypeError, ValueError):
                    revision = 1
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('catalog_revision',?)",
                    (str(revision),),
                )
                db.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('last_changed',?)",
                    (changed_at,),
                )
        return {
            "ok": True,
            "uid": uid,
            "collection": collection,
            "fingerprint": current_fingerprint,
            "review_id": current_fingerprint,
            "content_sha256": current_fingerprint,
            "decision": decision,
            "review_status": status,
            "dataset_usable": bool(usable),
            "review": {
                "required": bool(row["review_required"]),
                "status": status,
                "decision": decision,
                "review_id": current_fingerprint,
                "content_sha256": current_fingerprint,
                "dataset_usable": bool(usable),
                "note": event.get("note", ""),
                "reviewed_at": event.get("reviewed_at"),
            },
            "event": event,
        }

    def resolve_media(self, token: str, kind: str) -> Path:
        collection, uid = self._decode_token(token)
        columns = {
            "image": "image_path",
            "audio": "audio_path",
            "alignment": "alignment_path",
            "speech_metadata": "speech_dialogue_path",
        }
        column = columns.get(kind)
        if column is None:
            raise ValueError(f"unsupported media kind: {kind}")
        with self._connect() as db:
            row = db.execute(
                f"SELECT {column} AS path FROM items WHERE collection=? AND uid=?",
                (collection, uid),
            ).fetchone()
        if row is None:
            raise KeyError("unknown dataset item")
        if not row["path"]:
            raise FileNotFoundError(f"item has no {kind}")
        path = self._safe_media_path(row["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    # ------------------------------------------------------------------
    # Query helpers / normalization
    # ------------------------------------------------------------------
    @staticmethod
    def _query_items(ai_status: str | None) -> str:
        # CROSS JOIN intentionally starts with compact AI keys. Otherwise SQLite
        # prefers items_review and visits every pending item for every page.
        return ("ai_ready_keys CROSS JOIN items ON collection=ai_collection AND uid=ai_uid"
                if ai_status else "items")

    def _with_ai_filter(self, db: sqlite3.Connection, where: str, ai_status: str | None) -> str:
        """Join compact draft keys, without opening every source/image/payload.

        Catalogued image hashes select candidates; get_item and human saves
        independently rehash current image bytes before displaying/accepting one.
        The attached store and temporary human-history index are read snapshots.
        """
        if not ai_status:
            return where
        if ai_status not in {"ready", "needs_attention"}:
            raise ValueError(f"unsupported AI status: {ai_status}")
        prefix = where + " AND " if where else "WHERE "
        db.execute("CREATE TEMP TABLE ai_ready_keys(ai_collection TEXT,ai_uid TEXT,PRIMARY KEY(ai_collection,ai_uid)) WITHOUT ROWID")
        if not self.ai_suggestion_store.path.is_file():
            return prefix + "0"
        db.execute("ATTACH DATABASE ? AS ai_drafts",
                   (self.ai_suggestion_store.path.resolve().as_uri() + "?mode=ro",))
        db.execute("CREATE TEMP TABLE ai_human_touched(collection TEXT,uid TEXT,PRIMARY KEY(collection,uid)) WITHOUT ROWID")
        db.executemany("INSERT INTO ai_human_touched VALUES(?,?)", human_touched_uids(self.review_store))
        db.execute("INSERT OR IGNORE INTO ai_ready_keys SELECT collection,uid FROM ai_drafts.suggestions")
        verdict = " AND json_valid(draft.payload_json) AND json_extract(draft.payload_json,'$.verdict')='needs_attention'" if ai_status == "needs_attention" else ""
        return prefix + """review_required=1 AND review_status='pending' AND has_dialogue=1
            AND json_type(summary_json,'$.human_revision') IS NOT 'object'
            AND COALESCE(json_extract(summary_json,'$.image_integrity_status'),'')!='unavailable'
            AND NOT EXISTS (SELECT 1 FROM ai_human_touched AS touched
                WHERE touched.collection=items.collection AND touched.uid=items.uid)
            AND EXISTS (SELECT 1 FROM ai_drafts.suggestions AS draft
                WHERE draft.collection=items.collection AND draft.uid=items.uid
                AND draft.base_fingerprint=items.review_id
                AND draft.image_sha256=json_extract(items.summary_json,'$.current_image_sha256')""" + verdict + ")"

    @staticmethod
    def _where(
        collection: str | None = None,
        category: str | None = None,
        stage: str | None = None,
        query: str | None = None,
        *,
        directory: str | None = None,
        has_image: bool | None = None,
        has_dialogue: bool | None = None,
        has_audio: bool | None = None,
        has_alignment: bool | None = None,
        review_status: str | None = None,
        review_required: bool | None = None,
        auxiliary_kind: str | None = None,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if collection:
            clauses.append("collection=?")
            params.append(collection)
        if category:
            clauses.append("category=?")
            params.append(category)
        if directory:
            clauses.append("directory=?")
            params.append(directory)
        if stage:
            if stage == "missing_dialogue":
                clauses.append("has_image=1 AND has_dialogue=0")
            elif stage == "planned_missing_dialogue":
                clauses.append("eligible=1 AND has_dialogue=0")
            elif stage == "missing_audio":
                clauses.append("has_dialogue=1 AND has_audio=0")
            elif stage == "missing_alignment":
                clauses.append("has_audio=1 AND has_alignment=0")
            elif stage == "regeneration_pending":
                clauses.append(REGENERATION_PENDING_SQL)
            elif stage in {"empty", "image", "dialogue", "audio", "complete"}:
                clauses.append("stage=?")
                params.append(stage)
            elif stage in {"has_image", "has_dialogue", "has_audio", "has_alignment"}:
                clauses.append(f"{stage}=1")
            else:
                raise ValueError(f"unsupported stage filter: {stage}")
        if query:
            clauses.append("search_text LIKE ? ESCAPE '\\'")
            escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            params.append(f"%{escaped}%")
        if review_status:
            if review_status not in {*REVIEW_DECISIONS, "not_required", "unreviewed"}:
                raise ValueError(f"unsupported review status: {review_status}")
            if review_status == "unreviewed":
                clauses.append("has_dialogue=1 AND review_status IN ('pending','not_required')")
            else:
                clauses.append("review_status=?")
                params.append(review_status)
        if auxiliary_kind:
            if auxiliary_kind not in {"none", "counting", "ocr"}:
                raise ValueError(f"unsupported auxiliary kind: {auxiliary_kind}")
            if auxiliary_kind == "none":
                clauses.append("auxiliary_kind=?")
            else:
                # Keep the main queue/count on its covering review index. The
                # sparse index supplies the extra rowids without loading each
                # dialogue's JSON on every approval and navigation request.
                clauses.append("""auxiliary_kind=? OR items.rowid IN
                    (SELECT rowid FROM items WHERE has_both_auxiliary_kinds=1)""")
            params.append(auxiliary_kind)
        for column, expected in (
            ("has_image", has_image),
            ("has_dialogue", has_dialogue),
            ("has_audio", has_audio),
            ("has_alignment", has_alignment),
            ("review_required", review_required),
        ):
            if expected is not None:
                clauses.append(f"{column}=?")
                params.append(int(expected))
        return ("WHERE " + " AND ".join(f"({clause})" for clause in clauses) if clauses else "", params)

    def _summary_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        try:
            summary = json.loads(row["summary_json"])
        except (TypeError, json.JSONDecodeError):
            summary = {}
        availability = {
            "image": bool(row["has_image"]),
            "dialogue": bool(row["has_dialogue"]),
            "audio": bool(row["has_audio"]),
            "alignment": bool(row["has_alignment"]),
        }
        revision = summary.get("human_revision") or {}
        had_media = revision.get("had_media", any(summary.get("source_media_paths") or []))
        regeneration = bool(revision and had_media and row["review_status"] == "approved"
                            and (not row["has_audio"] or not row["has_alignment"]))
        return {
            "token": self._encode_token(row["collection"], row["uid"]),
            "collection": row["collection"],
            "collection_label": COLLECTIONS.get(row["collection"], row["collection"]),
            "uid": row["uid"],
            "source_uid": row["source_uid"],
            "category": row["category"],
            "directory": row["directory"],
            "stage": row["stage"],
            "eligible": None if row["eligible"] is None else bool(row["eligible"]),
            "availability": availability,
            "media": {
                "image": bool(row["image_path"]),
                "audio": bool(row["audio_path"]),
                "alignment": bool(row["alignment_path"]),
            },
            "summary": summary,
            "verification_status": row["verification_status"],
            "auxiliary_task": summary.get("auxiliary_task"),
            "auxiliary_kind": row["auxiliary_kind"],
            "fingerprint": row["review_id"],
            "review_id": row["review_id"],
            "content_sha256": row["review_id"],
            "review_required": bool(row["review_required"]),
            "review_status": row["review_status"],
            "dataset_usable": bool(row["dataset_usable"]),
            "human_certified": row["review_status"] == "approved",
            "regeneration_pending": regeneration,
            "human_revision": summary.get("human_revision"),
            "review_available": bool(row["has_dialogue"]),
            "review": {
                "required": bool(row["review_required"]),
                "status": row["review_status"],
                "review_id": row["review_id"],
                "content_sha256": row["review_id"],
                "dataset_usable": bool(row["dataset_usable"]),
            },
        }

    def _navigation(self, row: sqlite3.Row, filters: Mapping[str, Any]) -> dict[str, Any]:
        collection = filters.get("collection")
        category = filters.get("category")
        stage = filters.get("stage")
        query = filters.get("query")
        where, params = self._where(
            collection,
            category,
            stage,
            query,
            directory=filters.get("directory"),
            has_image=filters.get("has_image"),
            has_dialogue=filters.get("has_dialogue"),
            has_audio=filters.get("has_audio"),
            has_alignment=filters.get("has_alignment"),
            review_status=filters.get("review_status"),
            review_required=filters.get("review_required"),
            auxiliary_kind=filters.get("auxiliary_kind"),
        )
        earlier = "(collection<? OR (collection=? AND (category<? OR (category=? AND uid<?))))"
        later = "(collection>? OR (collection=? AND (category>? OR (category=? AND uid>?))))"
        order_values = [
            row["collection"],
            row["collection"],
            row["category"],
            row["category"],
            row["uid"],
        ]
        with self._connect() as db:
            where = self._with_ai_filter(db, where, filters.get("ai_status"))
            items_source = self._query_items(filters.get("ai_status"))
            base = where[6:] if where.startswith("WHERE ") else ""
            prefix = f"({base}) AND " if base else ""
            previous = db.execute(
                f"""SELECT collection,uid FROM {items_source} WHERE {prefix}{earlier}
                    ORDER BY collection DESC,category DESC,uid DESC LIMIT 1""",
                [*params, *order_values],
            ).fetchone()
            next_row = db.execute(
                f"""SELECT collection,uid FROM {items_source} WHERE {prefix}{later}
                    ORDER BY collection,category,uid LIMIT 1""",
                [*params, *order_values],
            ).fetchone()
            total = db.execute(
                f"SELECT COUNT(*) FROM {items_source} {where}", params
            ).fetchone()[0]
            before = db.execute(
                f"SELECT COUNT(*) FROM {items_source} WHERE {prefix}{earlier}",
                [*params, *order_values],
            ).fetchone()[0]
        return {
            "previous": (
                self._encode_token(previous["collection"], previous["uid"]) if previous else None
            ),
            "next": self._encode_token(next_row["collection"], next_row["uid"]) if next_row else None,
            "index": before + 1,
            "total": total,
        }

    def _read_locator(
        self, source_id: str | None, offset: int | None, length: int | None
    ) -> dict[str, Any] | None:
        if source_id is None or offset is None or length is None:
            return None
        with self._connect() as db:
            row = db.execute("SELECT path FROM sources WHERE source_id=?", (source_id,)).fetchone()
        if row is None:
            return None
        source = (self.repo_root / row["path"]).resolve()
        try:
            source.relative_to(self.repo_root)
        except ValueError:
            return None
        try:
            with source.open("rb") as stream:
                stream.seek(offset)
                raw = stream.read(length)
            value = json.loads(raw)
            return value if isinstance(value, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _read_json_media(self, stored_path: str | None) -> Any:
        if not stored_path:
            return None
        try:
            path = self._safe_media_path(stored_path)
            with path.open("r", encoding="utf-8") as stream:
                return json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _normalise_dialogue(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        result: list[dict[str, Any]] = []
        for index, turn in enumerate(value):
            if not isinstance(turn, Mapping):
                continue
            raw_speaker = str(turn.get("speaker") or turn.get("role") or "")
            lowered = raw_speaker.lower()
            if raw_speaker in {"ユーザー", "USER"} or lowered == "user":
                speaker = "user"
            elif raw_speaker in {"アシスタント", "ASSISTANT"} or lowered == "assistant":
                speaker = "assistant"
            else:
                speaker = lowered or "unknown"
            result.append(
                {"index": index, "speaker": speaker, "text": str(turn.get("text") or "")}
            )
        return result

    @staticmethod
    def _normalise_alignment(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, Mapping):
            raw_items = value.get("alignments")
            result: dict[str, Any] = {
                "format": value.get("format"),
                "tokenizer": value.get("tokenizer"),
            }
        else:
            raw_items = value
            result = {"format": None, "tokenizer": None}
        if not isinstance(raw_items, list):
            return {**result, "items": [], "count": 0}
        items: list[dict[str, Any]] = []
        for entry in raw_items:
            if isinstance(entry, Mapping):
                text = entry.get("text") or entry.get("token") or entry.get("piece") or ""
                times = entry.get("times") or entry.get("time")
                start = entry.get("start")
                end = entry.get("end")
                speaker = entry.get("speaker") or entry.get("role")
            elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                text, times, speaker = entry[0], entry[1], entry[2]
                start = end = None
            else:
                continue
            if isinstance(times, (list, tuple)) and len(times) >= 2:
                start, end = times[0], times[1]
            items.append(
                {"text": str(text), "start": start, "end": end, "speaker": speaker}
            )
        result.update({"items": items, "count": len(items)})
        return result

    @staticmethod
    def _without_dialogue(value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {key: item for key, item in value.items() if key != "dialogue"}

    # ------------------------------------------------------------------
    # Safe paths and tokens
    # ------------------------------------------------------------------
    @staticmethod
    def _encode_token(collection: str, uid: str) -> str:
        raw = f"{collection}\0{uid}".encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_token(token: str) -> tuple[str, str]:
        if not isinstance(token, str) or not token or len(token) > 8_192:
            raise ValueError("invalid item token")
        try:
            padding = "=" * (-len(token) % 4)
            raw = base64.b64decode(token + padding, altchars=b"-_", validate=True).decode("utf-8")
            collection, uid = raw.split("\0", 1)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("invalid item token") from exc
        if collection not in COLLECTIONS or not uid or "\0" in uid:
            raise ValueError("invalid item token")
        if DatasetCatalog._encode_token(collection, uid) != token.rstrip("="):
            raise ValueError("non-canonical item token")
        return collection, uid

    def _normalise_stored_path(self, value: Any) -> str | None:
        if not isinstance(value, (str, os.PathLike)) or not str(value):
            return None
        raw = Path(value)
        if raw.is_absolute():
            # This method runs for several hundred thousand manifest assets.
            # Keep normalization lexical here: Path.resolve() performs filesystem
            # work for every row and made the initial index unnecessarily slow.
            # Symlink resolution and the strict data-root check still happen in
            # _safe_media_path immediately before a file is served.
            lexical = Path(os.path.abspath(os.path.normpath(os.fspath(raw))))
            try:
                candidate = lexical
                candidate.relative_to(self.repo_root)
            except ValueError:
                # Manifests may be moved with the repository.  Remap the suffix
                # after /data/ but still enforce the current data-root allowlist.
                parts = raw.parts
                if "data" not in parts:
                    return None
                index = parts.index("data")
                candidate = Path(
                    os.path.abspath(
                        os.path.normpath(os.fspath(self.repo_root.joinpath(*parts[index:])))
                    )
                )
        else:
            candidate = Path(
                os.path.abspath(os.path.normpath(os.fspath(self.repo_root / raw)))
            )
        try:
            candidate.relative_to(self.data_root)
        except ValueError:
            return None
        return candidate.relative_to(self.repo_root).as_posix()

    @staticmethod
    def _display_directory(collection: str, image_path: str | None) -> str:
        """Return a useful, bounded tree directory instead of one node per UID."""
        if collection == "nonvisual_v1":
            return "data/speech_nonvisual_v1/data_stereo"
        if collection in {"counting_v4", "ocr_v4"}:
            return "data/images/targeted_v4/image_pool_v1"
        if not image_path:
            return "(画像なし)"
        parts = Path(image_path).parts
        try:
            image_index = parts.index("images")
            # data/images/<dataset> is the stable directory grouping.  It keeps
            # COCO, DOCCI and the Wikimedia families separately without exposing
            # a folder for each copied speech asset.
            end = min(len(parts), image_index + 2)
            return Path(*parts[:end]).as_posix()
        except (ValueError, IndexError):
            return Path(image_path).parent.as_posix()

    def _safe_media_path(self, stored_path: str) -> Path:
        if not isinstance(stored_path, str) or not stored_path:
            raise ValueError("invalid media path")
        raw = Path(stored_path)
        candidate = raw.resolve(strict=False) if raw.is_absolute() else (self.repo_root / raw).resolve(strict=False)
        try:
            candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ValueError("media path escapes data root") from exc
        return candidate

    def _stored_path_exists(self, stored_path: str) -> bool:
        try:
            return self._safe_media_path(stored_path).is_file()
        except ValueError:
            return False

    @staticmethod
    def _visual_category(image_path: str | None) -> str:
        if not image_path:
            return "other"
        parts = Path(image_path).parts
        try:
            raw = parts[parts.index("images") + 1].lower()
        except (ValueError, IndexError):
            raw = Path(image_path).parent.name.lower()
        if raw.startswith("coco"):
            return "coco"
        if raw.startswith("docci"):
            return "docci"
        if raw.startswith("openverse"):
            return "openverse"
        if raw.startswith("wikimedia"):
            return "wikimedia"
        if raw in {"japanese_photos", "alfredplpl_jp"}:
            return raw
        return raw or "other"


__all__ = ["DatasetCatalog", "COLLECTIONS"]
