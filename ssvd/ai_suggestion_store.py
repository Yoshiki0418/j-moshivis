"""Immutable AI dialogue suggestions, independent of human review decisions.

A suggestion applies only to the exact collection, source dialogue and image
that were inspected. This module never updates source records, catalogues or
the human review log. Readers do not create a missing suggestion database.
"""

from __future__ import annotations

from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any


DEFAULT_AI_SUGGESTIONS = Path("data/curation/ai_dialogue_suggestions.sqlite3")
MAX_PAYLOAD_BYTES = 512 * 1024
MAX_DIALOGUE_TURNS = 128
_BUSY_TIMEOUT_MS = 5_000
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SPEAKERS = frozenset({"ユーザー", "アシスタント", "USER", "ASSISTANT", "user", "assistant"})
_VERDICTS = frozenset({"corrected", "unchanged", "needs_attention"})
_CONTENT_FIELDS = frozenset({
    "schema_version", "collection", "uid", "base_fingerprint", "image_sha256",
    "dialogue", "auxiliary_task", "model", "changes", "concerns", "verdict", "provenance",
})
_KEY_FIELDS = ("collection", "uid", "base_fingerprint", "image_sha256")
_HUMAN_HISTORY_LOCK = threading.Lock()
_HUMAN_HISTORY_CACHE: dict[str, tuple[tuple[int, ...], frozenset[tuple[str, str]]]] = {}
_SCHEMA = """
CREATE TABLE IF NOT EXISTS suggestions (
    collection TEXT NOT NULL,
    uid TEXT NOT NULL,
    base_fingerprint TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    suggestion_id TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_sha256 TEXT NOT NULL,
    PRIMARY KEY (collection, uid, base_fingerprint, image_sha256)
) WITHOUT ROWID
"""


class AISuggestionConflictError(RuntimeError):
    """An immutable suggestion already exists for these exact source bytes."""


class AISuggestionIntegrityError(RuntimeError):
    """Stored data failed its content hash or schema validation."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("suggestion must contain valid UTF-8 JSON") from error


def _text(value: Any, name: str, maximum: int, *, nonempty: bool = True) -> None:
    if (not isinstance(value, str) or len(value) > maximum or "\x00" in value
            or (nonempty and not value.strip())):
        raise ValueError(f"{name} must be a valid string of at most {maximum} characters")


def _key(collection: Any, uid: Any, base_fingerprint: Any, image_sha256: Any) -> tuple:
    _text(collection, "collection", 512)
    _text(uid, "uid", 2048)
    for name, value in (("base_fingerprint", base_fingerprint), ("image_sha256", image_sha256)):
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 hex string")
    return collection, uid, base_fingerprint, image_sha256


def _bounded_json(value: Any, name: str) -> None:
    """Bound nested metadata before serialization, including non-JSON inputs."""
    remaining = 10_000

    def visit(item: Any, depth: int) -> None:
        nonlocal remaining
        remaining -= 1
        if depth > 12 or remaining < 0:
            raise ValueError(f"{name} is too deeply nested or has too many values")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if not -(2 ** 63) <= item < 2 ** 63:
                raise ValueError(f"{name} contains an out-of-range integer")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError(f"{name} contains a nonfinite number")
            return
        if isinstance(item, str):
            _text(item, name, 16_000, nonempty=False)
            return
        if isinstance(item, list):
            for child in item:
                visit(child, depth + 1)
            return
        if isinstance(item, dict):
            for key, child in item.items():
                _text(key, name + " key", 256)
                visit(child, depth + 1)
            return
        raise ValueError(f"{name} must contain JSON values only")

    visit(value, 0)


def _validate_content(content: dict[str, Any]) -> None:
    if set(content) != _CONTENT_FIELDS or type(content["schema_version"]) is not int or content["schema_version"] != 1:
        raise ValueError("invalid suggestion schema")
    _key(*(content[field] for field in _KEY_FIELDS))
    _text(content["model"], "model", 200)
    dialogue = content["dialogue"]
    if not isinstance(dialogue, list) or not 1 <= len(dialogue) <= MAX_DIALOGUE_TURNS:
        raise ValueError("dialogue must contain 1 to 128 turns")
    for turn in dialogue:
        if not isinstance(turn, dict) or set(turn) != {"speaker", "text"}:
            raise ValueError("each dialogue turn must contain only speaker and text")
        if not isinstance(turn["speaker"], str) or turn["speaker"] not in _SPEAKERS:
            raise ValueError("invalid dialogue speaker")
        _text(turn["text"], "dialogue text", 16_000)
    changes = content["changes"]
    if not isinstance(changes, list) or len(changes) > MAX_DIALOGUE_TURNS:
        raise ValueError("changes must be a list of at most 128 entries")
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"turn_index", "reason"}:
            raise ValueError("each change must contain only turn_index and reason")
        index = change["turn_index"]
        if type(index) is not int or not 0 <= index < len(dialogue):
            raise ValueError("change turn_index is outside the dialogue")
        _text(change["reason"], "change reason", 4_000)
    concerns = content["concerns"]
    if not isinstance(concerns, list) or len(concerns) > 128:
        raise ValueError("concerns must be a list of at most 128 strings")
    for concern in concerns:
        _text(concern, "concern", 4_000)
    if not isinstance(content["verdict"], str) or content["verdict"] not in _VERDICTS:
        raise ValueError("invalid suggestion verdict")
    for name in ("auxiliary_task", "provenance"):
        if content[name] is not None and not isinstance(content[name], dict):
            raise ValueError(f"{name} must be an object or null")
        _bounded_json(content[name], name)
    if len(_canonical_json(content)) > MAX_PAYLOAD_BYTES:
        raise ValueError("suggestion exceeds the payload size limit")


class AISuggestionStore:
    """One durable proposal per source version, with short SQLite transactions."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def _connect(self, *, write: bool) -> sqlite3.Connection | None:
        if write:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.path, timeout=_BUSY_TIMEOUT_MS / 1000)
        else:
            if not self.path.is_file():
                return None
            connection = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro",
                                         uri=True, timeout=_BUSY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
            if write:
                # SQLite may reject two simultaneous initial journal-mode
                # changes immediately, without invoking its busy handler.
                deadline = time.monotonic() + _BUSY_TIMEOUT_MS / 1000
                while True:
                    try:
                        connection.execute("PRAGMA journal_mode=WAL")
                        break
                    except sqlite3.OperationalError as error:
                        if (getattr(error, "sqlite_errorcode", None) not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                                or time.monotonic() >= deadline):
                            raise
                        time.sleep(0.01)
                connection.execute("PRAGMA synchronous=FULL")
            else:
                connection.execute("PRAGMA query_only=ON")
        except BaseException:
            connection.close()
            raise
        return connection

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        try:
            raw = row["payload_json"]
            if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                raise ValueError("invalid stored payload size")
            if hashlib.sha256(raw.encode("utf-8")).hexdigest() != row["payload_sha256"]:
                raise ValueError("stored payload digest differs")
            payload = json.loads(raw)
            if not isinstance(payload, dict) or set(payload) != _CONTENT_FIELDS | {"suggestion_id", "created_at"}:
                raise ValueError("invalid stored payload fields")
            content = {key: payload[key] for key in _CONTENT_FIELDS}
            _validate_content(content)
            expected = hashlib.sha256(_canonical_json(content)).hexdigest()
            if expected != row["suggestion_id"] or expected != payload["suggestion_id"]:
                raise ValueError("suggestion content digest differs")
            for name in (*_KEY_FIELDS, "created_at"):
                if payload[name] != row[name]:
                    raise ValueError("stored identity differs")
            timestamp = dt.datetime.fromisoformat(payload["created_at"])
            if timestamp.utcoffset() != dt.timedelta(0):
                raise ValueError("invalid creation timestamp")
            return payload
        except (KeyError, IndexError, TypeError, ValueError, UnicodeError, OverflowError, RecursionError) as error:
            raise AISuggestionIntegrityError("Stored AI suggestion failed integrity validation") from error

    def get_suggestion(self, *, collection: str, uid: str,
                       base_fingerprint: str, image_sha256: str) -> dict[str, Any] | None:
        key = _key(collection, uid, base_fingerprint, image_sha256)
        connection = self._connect(write=False)
        if connection is None:
            return None
        try:
            row = connection.execute(
                "SELECT * FROM suggestions WHERE collection=? AND uid=? "
                "AND base_fingerprint=? AND image_sha256=?", key).fetchone()
            return self._decode(row) if row is not None else None
        except sqlite3.DatabaseError as error:
            raise AISuggestionIntegrityError("AI suggestion database could not be read") from error
        finally:
            connection.close()

    def put_suggestion(self, *, collection: str, uid: str, base_fingerprint: str,
                       image_sha256: str, dialogue: list[dict[str, str]],
                       auxiliary_task: dict[str, Any] | None, model: str,
                       changes: list[dict[str, Any]], concerns: list[str], verdict: str,
                       provenance: dict[str, Any] | None = None) -> dict[str, Any]:
        content = {
            "schema_version": 1, "collection": collection, "uid": uid,
            "base_fingerprint": base_fingerprint, "image_sha256": image_sha256,
            "dialogue": dialogue, "auxiliary_task": auxiliary_task, "model": model,
            "changes": changes, "concerns": concerns, "verdict": verdict,
            "provenance": provenance,
        }
        _validate_content(content)
        encoded_content = _canonical_json(content)
        # Detach the stored return value from mutable caller-owned lists/maps.
        payload = json.loads(encoded_content)
        suggestion_id = hashlib.sha256(encoded_content).hexdigest()
        payload.update(suggestion_id=suggestion_id,
                       created_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds"))
        raw = _canonical_json(payload)
        if len(raw) > MAX_PAYLOAD_BYTES:
            raise ValueError("suggestion exceeds the payload size limit")
        key = tuple(content[field] for field in _KEY_FIELDS)
        connection = self._connect(write=True)
        assert connection is not None
        try:
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(_SCHEMA)
                existing = connection.execute(
                    "SELECT * FROM suggestions WHERE collection=? AND uid=? "
                    "AND base_fingerprint=? AND image_sha256=?", key).fetchone()
                if existing is not None:
                    saved = self._decode(existing)
                    if saved["suggestion_id"] != suggestion_id:
                        raise AISuggestionConflictError("An AI suggestion already exists for this exact source version")
                    return saved
                connection.execute(
                    "INSERT INTO suggestions (collection, uid, base_fingerprint, image_sha256, "
                    "suggestion_id, created_at, payload_json, payload_sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (*key, suggestion_id, payload["created_at"], raw.decode("utf-8"), hashlib.sha256(raw).hexdigest()))
            return payload
        finally:
            connection.close()

    def count(self, *, collection: str | None = None) -> int:
        """Count saved proposals without loading dialogue text or creating a DB."""
        if collection is not None:
            _text(collection, "collection", 512)
        connection = self._connect(write=False)
        if connection is None:
            return 0
        try:
            if collection is None:
                return connection.execute("SELECT COUNT(*) FROM suggestions").fetchone()[0]
            return connection.execute("SELECT COUNT(*) FROM suggestions WHERE collection=?", (collection,)).fetchone()[0]
        except sqlite3.DatabaseError as error:
            raise AISuggestionIntegrityError("AI suggestion database could not be read") from error
        finally:
            connection.close()


def default_ai_suggestion_store(root: str | os.PathLike[str]) -> AISuggestionStore:
    return AISuggestionStore(Path(root) / DEFAULT_AI_SUGGESTIONS)


@contextmanager
def locked_human_uids(review_store: Any):
    """Keep a shared lock on the decision log while comparing and publishing a candidate."""
    path = Path(review_store.path).resolve()
    key = str(path)
    with path.open("r", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_SH)
        with _HUMAN_HISTORY_LOCK:
            stat = os.fstat(stream.fileno())
            stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = _HUMAN_HISTORY_CACHE.get(key)
            if cached is None or cached[0] != stamp:
                result = set()
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (ValueError, UnicodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    fingerprint = event.get("fingerprint") or event.get("review_id")
                    decision = event.get("decision") or event.get("status")
                    collection, uid = event.get("collection"), event.get("uid")
                    if (isinstance(fingerprint, str) and _SHA256.fullmatch(fingerprint)
                            and isinstance(decision, str) and decision in {"pending", "approved", "rejected"}
                            and isinstance(collection, str) and collection.strip()
                            and isinstance(uid, str) and uid.strip()):
                        result.add((collection, uid))
                if len(_HUMAN_HISTORY_CACHE) >= 64:
                    _HUMAN_HISTORY_CACHE.pop(next(iter(_HUMAN_HISTORY_CACHE)))
                cached = (stamp, frozenset(result))
                _HUMAN_HISTORY_CACHE[key] = cached
        yield cached[1]


def human_touched_uids(review_store: Any) -> set[tuple[str, str]]:
    """All human decisions, including superseded decisions and decisions in other collections."""
    try:
        with locked_human_uids(review_store) as touched:
            return set(touched)
    except FileNotFoundError:
        with _HUMAN_HISTORY_LOCK:
            _HUMAN_HISTORY_CACHE.pop(str(Path(review_store.path).resolve()), None)
        return set()
