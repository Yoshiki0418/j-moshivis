"""Content-bound, append-only review decisions for generated dialogues.

The review key is deliberately derived from the generated content rather than
from the UID.  Regenerating a dialogue therefore makes an earlier approval
stale automatically and returns the new version to ``pending``.
"""

from __future__ import annotations

import datetime as _dt
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


REVIEW_REQUIRED_STATUS = "unverified_vlm_visual"
REVIEW_DECISIONS = frozenset({"pending", "approved", "rejected"})
MAX_REVIEW_NOTE_CHARS = 1_000
DEFAULT_REVIEW_EVENTS = Path("data/curation/dialogue_review_events.jsonl")

_FINGERPRINT_FIELDS = (
    "uid",
    "image",
    "image_sha256",
    "task_mode",
    "dialogue",
    "auxiliary_task",
    "verification_status",
    "generator",
)


class ReviewConflictError(RuntimeError):
    """The requested decision does not match the current reviewable content."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def image_file_sha256(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Hash source image bytes without loading the whole image into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def content_fingerprint(
    record: Mapping[str, Any],
    *,
    image: str | None = None,
    image_sha256: str | None = None,
) -> str:
    """Return the SHA-256 for the exact reviewable dialogue content."""

    payload = {key: record.get(key) for key in _FINGERPRINT_FIELDS}
    # Older records have no warning field and retain their original review ID.
    if "review_warnings" in record:
        payload["review_warnings"] = record["review_warnings"]
    if payload["image"] is None and image is not None:
        payload["image"] = image
    # An explicit hash is an independently recomputed current-image hash and
    # therefore intentionally overrides a hash declared by the JSONL record.
    if image_sha256 is not None:
        payload["image_sha256"] = image_sha256
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def requires_review(record: Mapping[str, Any]) -> bool:
    """Whether this record contains an unverified VLM-visual answer."""

    if record.get("review_required") is True or record.get("verification_status") == REVIEW_REQUIRED_STATUS:
        return True
    auxiliary = record.get("auxiliary_task")
    return bool(
        isinstance(auxiliary, Mapping)
        and auxiliary.get("grounding_source") == "vlm_visual"
    )


def review_metadata(
    record: Mapping[str, Any],
    *,
    image: str | None = None,
    image_sha256: str | None = None,
) -> dict[str, Any]:
    """Build catalogue/training metadata for a generated dialogue record."""

    auxiliary = record.get("auxiliary_task")
    auxiliary_kind = (
        str(auxiliary.get("kind"))
        if isinstance(auxiliary, Mapping) and auxiliary.get("kind") is not None
        else None
    )
    required = requires_review(record)
    fingerprint = content_fingerprint(
        record, image=image, image_sha256=image_sha256
    )
    return {
        "verification_status": record.get("verification_status"),
        "auxiliary_task": dict(auxiliary) if isinstance(auxiliary, Mapping) else None,
        "auxiliary_kind": auxiliary_kind,
        "review_required": required,
        "review_id": fingerprint,
        "fingerprint": fingerprint,
        "content_sha256": fingerprint,
    }


class ReviewStore:
    """Append and read content-bound decisions from a JSONL event log."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._revision_stamp = None
        self._revisions: dict[str, dict[str, Any]] = {}
        self._decision_stamp = None
        self._decisions = {}

    def latest_revisions(self) -> dict[str, dict[str, Any]]:
        """Latest edited content per original fingerprint, including rejected edits.

        Decisions and revised records share one durable event: a crash can never
        acknowledge an approval without saving its text. Later rejection changes
        eligibility, not the saved text, so it remains available for regeneration.
        """
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return {}
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._revision_stamp:
            return self._revisions
        revisions = {}
        with self.path.open(encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            try:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    record = event.get("revised_record")
                    base = event.get("base_fingerprint")
                    if (isinstance(record, dict) and isinstance(base, str)
                            and len(base) == 64
                            and content_fingerprint(record) == event.get("fingerprint")):
                        revisions[base] = event
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self._revision_stamp, self._revisions = stamp, revisions
        return revisions

    def resolve_record(self, record: Mapping[str, Any], *, image_sha256=None) -> dict[str, Any]:
        """Overlay an edit only on its exact original text AND source image."""
        base = content_fingerprint(record, image_sha256=image_sha256)
        revision = self.latest_revisions().get(base)
        return copy.deepcopy(revision["revised_record"] if revision else dict(record))

    def latest_decisions(self) -> dict[str, dict[str, Any]]:
        """Return the latest valid physical event for each exact fingerprint."""

        if not self.path.is_file():
            return {}
        stat = self.path.stat()
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._decision_stamp:
            return self._decisions
        latest: dict[str, dict[str, Any]] = {}
        with self.path.open("r", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH)
            try:
                for line in stream:
                    try:
                        event = json.loads(line)
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    fingerprint = event.get("fingerprint") or event.get("review_id")
                    decision = event.get("decision") or event.get("status")
                    if (
                        isinstance(fingerprint, str)
                        and len(fingerprint) == 64
                        and decision in REVIEW_DECISIONS
                    ):
                        latest.pop(fingerprint, None)
                        latest[fingerprint] = event
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        self._decision_stamp, self._decisions = stamp, latest
        return latest

    def decision_for(self, fingerprint: str) -> dict[str, Any] | None:
        return self.latest_decisions().get(fingerprint)

    def append(
        self,
        *,
        fingerprint: str,
        decision: str,
        uid: str,
        collection: str,
        note: str = "",
        reviewer: str | None = None,
        revised_record: Mapping[str, Any] | None = None,
        base_fingerprint: str | None = None,
        parent_fingerprint: str | None = None,
        task_mode: str | None = None,
        require_unreviewed: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("fingerprint must be a 64-character SHA-256 hex string")
        try:
            int(fingerprint, 16)
        except ValueError as exc:
            raise ValueError("fingerprint must be a SHA-256 hex string") from exc
        if decision not in REVIEW_DECISIONS:
            raise ValueError("decision must be pending, approved, or rejected")
        if not isinstance(note, str):
            raise ValueError("note must be a string")
        note = note.strip()
        if len(note) > MAX_REVIEW_NOTE_CHARS:
            raise ValueError(
                f"note must be at most {MAX_REVIEW_NOTE_CHARS} characters"
            )

        event: dict[str, Any] = {
            "schema_version": 1,
            "review_id": fingerprint,
            "fingerprint": fingerprint,
            "content_sha256": fingerprint,
            "uid": uid,
            "collection": collection,
            "decision": decision,
            "note": note,
            "reviewed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"
            ),
        }
        if reviewer:
            event["reviewer"] = str(reviewer)[:200]
        if task_mode is not None:
            event["task_mode"] = str(task_mode)
        if revised_record is not None:
            if decision != "approved" or content_fingerprint(revised_record) != fingerprint:
                raise ValueError("a saved revision must approve its exact revised content")
            if not base_fingerprint or not parent_fingerprint:
                raise ValueError("a revision requires original and parent fingerprints")
            event.update(revised_record=dict(revised_record),
                         base_fingerprint=base_fingerprint,
                         parent_fingerprint=parent_fingerprint,
                         action="edit_and_approve")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = _canonical_json(event) + b"\n"
        # O_APPEND plus an exclusive advisory lock prevents two viewer workers
        # from interleaving JSONL records.  fsync makes an acknowledged approval
        # durable before it can affect dataset eligibility.
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if revised_record is not None or require_unreviewed:
                # Compare while holding the same lock used by all writers. A
                # second tab/server cannot slip in a decision between this
                # check and the append/fsync below. Read the locked descriptor,
                # not cached history or a separately opened shared-lock reader.
                current = base_fingerprint
                with os.fdopen(os.dup(fd), "r", encoding="utf-8") as stream:
                    stream.seek(0)
                    for line in stream:
                        try:
                            previous = json.loads(line)
                        except (UnicodeError, json.JSONDecodeError):
                            continue
                        if not isinstance(previous, dict):
                            continue
                        if (require_unreviewed and previous.get("collection") == collection
                                and previous.get("uid") == uid):
                            previous_id = previous.get("fingerprint") or previous.get("review_id")
                            previous_decision = previous.get("decision") or previous.get("status")
                            if (isinstance(previous_id, str) and len(previous_id) == 64
                                    and all(c in "0123456789abcdefABCDEF" for c in previous_id)
                                    and isinstance(previous_decision, str)
                                    and previous_decision in REVIEW_DECISIONS):
                                raise ReviewConflictError(
                                    "A human decision already exists for this dialogue. "
                                    "Reload before approving the AI suggestion.")
                        if (previous.get("base_fingerprint") == base_fingerprint
                                and isinstance(previous.get("revised_record"), dict)):
                            current = previous.get("fingerprint")
                if revised_record is not None and current != parent_fingerprint:
                    raise ReviewConflictError("Another editor saved a newer revision. Reload before saving.")
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                # Preserve an interrupted trailing fragment as an invalid event,
                # but never let it corrupt the next durable record.
                os.write(fd, b"\n")
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        self._revision_stamp = None
        self._decision_stamp = None
        return event


def default_review_store(repo_root: str | os.PathLike[str]) -> ReviewStore:
    return ReviewStore(Path(repo_root) / DEFAULT_REVIEW_EVENTS)
