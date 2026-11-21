"""Read-only index of completed Terra regeneration candidates for human review."""

from __future__ import annotations

import copy
from contextlib import closing
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import re
import sqlite3
from threading import Lock

from ssvd.review_store import REVIEW_REQUIRED_STATUS, ReviewConflictError, content_fingerprint


COLLECTION = "japanese_daily_v1"
MODEL = "gpt-6.1-terra"
_RUNS = Path("data/curation/gpt_regeneration_issue13")
_PRODUCTION_LEDGER = "production_v10_terra/production.sqlite3"
_CLI_LEDGER = Path("data/curation/codex_regeneration_terra_v11/generation.sqlite3")
_LEDGERS = (
    *(_RUNS / f"first100_v10/shard{i}.sqlite3" for i in range(4)),
    _RUNS / _PRODUCTION_LEDGER,
    _CLI_LEDGER,
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")
Key = tuple[str, str, str, str]


@dataclass
class _Index:
    stamp: tuple = ()
    last_rowid: int = 0
    row_count: int = 0
    records: dict[Key, dict] = field(default_factory=dict)


def _stamp(path):
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _outcome_select(db, path):
    """Use the producer's immutable assignment; legacy unassigned rows used v10."""
    if path.name == _CLI_LEDGER.name:
        return """
            SELECT o.rowid, o.uid, o.model, o.outcome,
                strftime('%Y-%m-%dT%H:%M:%SZ', o.finished_at), 'visual_dialogue_v11'
            FROM outcomes o
        """
    version = "'visual_dialogue_v10'"
    join = ""
    if path.parts[-2:] == Path(_PRODUCTION_LEDGER).parts and db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='dialogue_prompt_versions'"
    ).fetchone():
        join = "LEFT JOIN dialogue_prompt_versions AS v ON v.uid=o.uid"
        version = "CASE WHEN v.uid IS NULL THEN 'visual_dialogue_v10' ELSE v.prompt_version END"
    return f"""
        SELECT o.rowid, o.uid, o.model, o.outcome,
            (SELECT strftime('%Y-%m-%dT%H:%M:%SZ', c.finished_at)
             FROM calls c WHERE c.uid=o.uid AND c.model=o.model AND c.charged IS NOT NULL
             ORDER BY c.step DESC LIMIT 1), {version}
        FROM outcomes o {join}
    """


def _record(uid, model, raw, finished_at, expected_version, path):
    outcome = json.loads(raw)
    if outcome["status"] != "pending_review":
        return None
    c, identity = outcome["candidate"], outcome["identity"]
    cli = path.name == _CLI_LEDGER.name
    backend = "codex_cli" if cli else "openai_api"
    # Old API outcomes predate the backend tag. The new CLI producer always writes both tags.
    for record in (outcome, c):
        if record.get("generation_backend", None if cli else backend) != backend:
            raise ValueError("Candidate generation backend differs from its ledger")
    base, image = identity["base_fingerprint"], identity["image_sha256"]
    pairs = identity["pairs"]
    if not (model == identity["model"] == c["generator"] == MODEL
            and uid == identity["uid"] == c["uid"] and identity["collection"] == COLLECTION
            and _SHA.fullmatch(base) and _SHA.fullmatch(image) and c["image_sha256"] == image
            and expected_version in ("visual_dialogue_v10", "visual_dialogue_v11")
            and identity["prompt_version"] == c["prompt_version"] == expected_version
            and c["review_required"] is True and c["verification_status"] == REVIEW_REQUIRED_STATUS
            and c["review_id"] == content_fingerprint(c)
            and type(pairs) is int and 4 <= pairs <= 8 and len(c["dialogue"]) == pairs * 2):
        raise ValueError("Candidate identity or content changed")
    auxiliary = c["auxiliary_task"]
    if (not isinstance(auxiliary, dict) or not isinstance(auxiliary.get("additional_tasks", []), list)
            or any(not isinstance(task, dict) or task.get("kind") not in ("ocr", "counting", "none")
                   for task in (auxiliary, *auxiliary.get("additional_tasks", [])))):
        raise ValueError("Invalid candidate task markers")
    for i, turn in enumerate(c["dialogue"]):
        if (turn["speaker"] != ("アシスタント" if i % 2 else "ユーザー")
                or not isinstance(turn["text"], str) or not turn["text"].strip()):
            raise ValueError("Incomplete candidate dialogue")
    return (COLLECTION, uid, base, image), {
        "candidate": c, "source": path.parent.name, "generation_backend": backend, "created_at": finished_at,
    }


class TerraCandidateStore:
    """Index immutable, committed outcomes in the configured producer-owned ledgers."""

    def __init__(self, repo_root):
        self._paths = [Path(repo_root).resolve() / name for name in _LEDGERS]
        self._indices: dict[Path, _Index] = {}
        self._lock = Lock()
        self._records: dict[Key, dict] = {}
        self._queue: tuple = ()
        self._version = 0

    def _refresh(self, *, invalidate=None):
        # Stat the WAL too, so readers see new commits independently of checkpoints.
        changed = False
        for path in self._paths:
            db_stamp = _stamp(path)
            if db_stamp is None:
                if path in self._indices:
                    del self._indices[path]
                    changed = True
                continue
            stamp = (db_stamp, _stamp(Path(str(path) + "-wal")))
            idx = self._indices.get(path)
            if idx is not None and idx.stamp == stamp and path != invalidate:
                continue
            # A restored/replaced ledger may reuse rowids. Discard its old keys.
            if idx is None or path == invalidate or idx.stamp[0][:2] != db_stamp[:2] or db_stamp[2] < idx.stamp[0][2]:
                idx = _Index()
                changed = True
            try:
                rows, count, rebuilt = self._new_rows(path, idx)
            except sqlite3.DatabaseError as error:
                # A generator can briefly lock its ledger or be creating its tables.
                # Keep known immutable rows during writer contention. On a restore
                # or an unreadable/schema-invalid source, revoke that source's rows.
                if (getattr(error, "sqlite_errorcode", None) not in (sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_BUSY)
                        or idx is not self._indices.get(path)):
                    if self._indices.pop(path, None) is not None:
                        changed = True
                    logging.getLogger(__name__).warning("Could not read Terra outcomes from %s", path.name)
                continue
            if rebuilt:
                idx = _Index()
                changed = True
            for rowid, *row in rows:
                try:
                    entry = _record(*row, path)
                except (TypeError, KeyError, ValueError, AttributeError, RecursionError):
                    logging.getLogger(__name__).warning("Skipping invalid Terra outcome %s:%d", path.name, rowid)
                    entry = None
                if entry:
                    key, record = entry
                    idx.records[key] = record
                    changed = True
                idx.last_rowid = rowid
            idx.stamp = stamp
            idx.row_count = count
            self._indices[path] = idx
        if changed:
            records = {}
            ambiguous = set()
            for idx in self._indices.values():
                for key, record in idx.records.items():
                    previous = records.get(key)
                    if previous and (previous["generation_backend"] != record["generation_backend"]
                            or any(previous["candidate"][field] != record["candidate"][field]
                                   for field in ("review_id", "prompt_version"))):
                        ambiguous.add(key)
                    records[key] = record
            for key in ambiguous:
                del records[key]
            self._records = records
            queue = []
            for key, record in records.items():
                task = record["candidate"]["auxiliary_task"]
                kinds = {item.get("kind") for item in (task, *task.get("additional_tasks", []))}
                queue.append((*key, int("ocr" in kinds), int("counting" in kinds)))
            self._queue = tuple(queue)
            self._version += 1

    @staticmethod
    def _new_rows(path, idx):
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.25)) as db:
            db.execute("BEGIN")
            count = db.execute("SELECT count(*) FROM outcomes").fetchone()[0]
            statement = _outcome_select(db, path) + " WHERE o.rowid>? ORDER BY o.rowid"
            rows = db.execute(statement, (idx.last_rowid,)).fetchall()
            rebuilt = idx.row_count + len(rows) != count
            if rebuilt:
                rows = db.execute(statement, (0,)).fetchall()
            return rows, count, rebuilt

    def keys(self) -> tuple[Key, ...]:
        with self._lock:
            self._refresh()
            return tuple(self._records)

    def queue_keys(self) -> tuple:
        """Keys followed by OCR and counting flags for the newly generated dialogue."""
        with self._lock:
            self._refresh()
            return self._queue

    def get(self, *, collection, uid, base_fingerprint, image_sha256) -> dict | None:
        with self._lock:
            self._refresh()
            key = (collection, uid, base_fingerprint, image_sha256)
            record = self._records.get(key)
            if record is None:
                return None
            # Generators append outcomes. If an operator corrects an existing row
            # in place, recheck the selected UID before it can be shown or approved.
            for path, idx in self._indices.items():
                if key not in idx.records:
                    continue
                try:
                    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=0.25)) as db:
                        db.execute("BEGIN")
                        row = db.execute(_outcome_select(db, path) + " WHERE o.uid=?", (uid,)).fetchone()
                    current = _record(*row[1:], path) if row else None
                except sqlite3.DatabaseError as error:
                    raise ReviewConflictError("生成台帳を照合できませんでした。再読み込みしてから確認してください。") from error
                except (TypeError, KeyError, ValueError, AttributeError, RecursionError):
                    current = None
                if not current or current[0] != key or current[1]["candidate"] != idx.records[key]["candidate"]:
                    # Only the affected ledger is reindexed, and only on divergence.
                    self._refresh(invalidate=path)
                    record = self._records.get(key)
                    break
            return copy.deepcopy(record) if record else None

    def version(self) -> int:
        with self._lock:
            self._refresh()
            return self._version
