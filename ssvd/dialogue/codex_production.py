"""Run the frozen CLI cohort with the same v11 image-backed turns as Terra API."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import threading
import time

from ssvd.ai_suggestion_store import locked_human_uids
from ssvd.dialogue.codex_cohort import BACKEND, CLI_MANIFEST, load_cli_manifest, verify
from ssvd.dialogue.codex_transport import CodexRateLimit, CodexTransport, CodexUsageLimit
from ssvd.dialogue.pilot import encode, operation
from ssvd.dialogue.preview_regeneration import COLLECTION, ROOT, preview
from ssvd.dialogue.production_budget import private_write
from ssvd.dialogue.production_cohort import MODEL
from ssvd.dialogue.regeneration_roles import VISUAL_V11
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, review_metadata

DB_NAME = "generation.sqlite3"


class Throttle:
    """Pause new turns across all workers when the service temporarily throttles one."""

    def __init__(self):
        self.lock = threading.Lock()
        self.until = 0.0

    def defer(self, seconds):
        with self.lock:
            self.until = max(self.until, time.monotonic() + seconds)

    def wait(self, stop):
        while not stop.is_set():
            with self.lock:
                remaining = self.until - time.monotonic()
            if remaining <= 0:
                return
            stop.wait(remaining)


def _connect(directory, *, readonly=False):
    mode = "ro" if readonly else "rwc"
    uri = (directory / DB_NAME).resolve().as_uri() + "?mode=" + mode
    return sqlite3.connect(uri, uri=True, timeout=60)


def _setup(directory, digest):
    marker = directory / "run_contract.json"
    contract = dict(manifest_sha256=digest, model=MODEL, generation_backend=BACKEND, prompt_version=VISUAL_V11)
    if marker.exists() and (json.loads(marker.read_bytes()) != contract or not (directory / DB_NAME).is_file()):
        raise RuntimeError("Restore the CLI ledger and its original contract")
    with closing(_connect(directory)) as db, db:
        db.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS run(id INTEGER PRIMARY KEY CHECK(id=1), contract TEXT NOT NULL, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS turns(uid TEXT NOT NULL, step INTEGER NOT NULL, request_hash TEXT NOT NULL,
                reply TEXT, receipt TEXT, PRIMARY KEY(uid,step));
            CREATE TABLE IF NOT EXISTS outcomes(uid TEXT PRIMARY KEY, model TEXT NOT NULL, outcome TEXT NOT NULL,
                finished_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
        """)
        saved = db.execute("SELECT contract FROM run WHERE id=1").fetchone()
        if saved is None and marker.exists():
            raise RuntimeError("Restore the original CLI ledger")
        if saved is not None and saved != (encode(contract),):
            raise RuntimeError("CLI ledger belongs to a different cohort or generation configuration")
        if not marker.exists() and db.execute("SELECT 1 FROM turns LIMIT 1").fetchone():
            raise RuntimeError("Restore the original CLI contract before resuming these turns")
        db.execute("INSERT OR IGNORE INTO run VALUES(1,?,'ready')", (encode(contract),))
        # If a process died after dispatch, the remote model may already have served the request.
        db.execute("""INSERT OR IGNORE INTO outcomes(uid,model,outcome)
                   SELECT uid,?,? FROM turns WHERE reply IS NULL""",
                   (MODEL, encode(dict(status="hold", reason="unconfirmed_cli_turn"))))
    if not marker.exists():
        private_write(marker, (encode(contract) + "\n").encode())


def _finish(db, store, uid, outcome):
    with locked_human_uids(store) as touched, db:
        if (COLLECTION, uid) in touched:
            outcome = {"status": "protected", "reason": "human_review"}
        db.execute("INSERT INTO outcomes(uid,model,outcome) VALUES(?,?,?)", (uid, MODEL, encode(outcome)))


def _dialogue(root, db, store, sample, transport, stop, throttle):
    uid = sample["uid"]
    identity = {k: sample[k] for k in ("uid", "kind", "pairs", "image_sha256", "base_fingerprint")}
    identity.update(model=MODEL, prompt_version=VISUAL_V11, collection=COLLECTION)
    replies = []
    while True:
        try:
            stage = preview(root, uid, model=MODEL, pairs=sample["pairs"], prompt_version=VISUAL_V11,
                            transcript={"identity": identity, "replies": replies})
        except (ValueError, FileNotFoundError):
            _finish(db, store, uid, {"status": "protected", "reason": "source_or_decision_changed"})
            return
        if stage["status"] != "request":
            if stage["status"] == "pending_review":
                candidate = stage["candidate"]
                candidate["generation_backend"] = BACKEND
                candidate.update(review_metadata(candidate))
                stage["generation_backend"] = BACKEND
            _finish(db, store, uid, stage)
            return
        step = len(replies)
        request = stage["request"]
        request_hash = hashlib.sha256(encode(request).encode()).hexdigest()
        saved = db.execute("SELECT request_hash,reply FROM turns WHERE uid=? AND step=?", (uid, step)).fetchone()
        if saved:
            if saved[0] != request_hash or saved[1] is None:
                raise RuntimeError("CLI request no longer matches its stored turn")
            replies.append(json.loads(saved[1]))
            continue
        throttle.wait(stop)
        if stop.is_set():
            return
        with locked_human_uids(store) as touched, db:
            if (COLLECTION, uid) in touched:
                db.execute("INSERT INTO outcomes(uid,model,outcome) VALUES(?,?,?)",
                           (uid, MODEL, encode(dict(status="protected", reason="human_review"))))
                return
            db.execute("INSERT INTO turns(uid,step,request_hash) VALUES(?,?,?)", (uid, step, request_hash))
        try:
            reply, receipt = transport.generate(request, root / sample["image_path"])
        except CodexUsageLimit:
            # A definitive quota refusal produced no turn. A later login-window reset can retry this slot.
            with db:
                db.execute("DELETE FROM turns WHERE uid=? AND step=? AND reply IS NULL", (uid, step))
            raise
        if receipt["model"] != MODEL or receipt["auth_method"] != "chatgpt" or receipt["transport"] != BACKEND:
            raise RuntimeError("Unexpected CLI receipt; review this turn before retrying")
        with db:
            db.execute("UPDATE turns SET reply=?,receipt=? WHERE uid=? AND step=?",
                       (encode(reply), encode(receipt), uid, step))
        replies.append(reply)


def report(root, digest):
    directory = (root / CLI_MANIFEST).parent
    size = load_cli_manifest(directory / "manifest.json", digest)["sample_size"]
    if not (directory / DB_NAME).is_file():
        if (directory / "run_contract.json").exists():
            raise RuntimeError("CLI ledger is missing")
        return dict(state="frozen", queue=size, manifest_sha256=digest)
    with closing(_connect(directory, readonly=True)) as db:
        db.execute("BEGIN")
        contract, state = db.execute("SELECT contract,state FROM run WHERE id=1").fetchone()
        if json.loads(contract)["manifest_sha256"] != digest:
            raise RuntimeError("CLI ledger does not match its manifest")
        statuses = dict(db.execute("SELECT json_extract(outcome,'$.status'),COUNT(*) FROM outcomes GROUP BY 1"))
        count, received = db.execute("SELECT COUNT(*),COUNT(reply) FROM turns").fetchone()
    return dict(state=state, model=MODEL, generation_backend=BACKEND, prompt_version=VISUAL_V11,
                queue=size, outcomes=statuses, remaining=size-sum(statuses.values()),
                received_turns=received, outstanding_turns=count-received)


def run(root, digest, *, workers=2, limit=None, transport=None, stop=None, retry_delays=(30, 90, 300)):
    if workers not in range(1, 5) or (limit is not None and limit < 1):
        raise ValueError("Expected 1–4 workers and a positive optional limit")
    stop = stop if stop is not None else threading.Event()
    transport = transport if transport is not None else CodexTransport()
    directory = (root / CLI_MANIFEST).parent
    store = ReviewStore(root / DEFAULT_REVIEW_EVENTS)
    with operation(directory):
        samples = load_cli_manifest(directory / "manifest.json", digest)["samples"]
        if not Path(store.path).is_file():
            raise RuntimeError("Human decision history is unavailable")
        transport.preflight()
        if not (directory / "run_contract.json").exists():
            verify(root, digest)
        _setup(directory, digest)
        with closing(_connect(directory)) as control, control:
            complete = {uid for uid, in control.execute("SELECT uid FROM outcomes")}
            started = {uid for uid, in control.execute("SELECT DISTINCT uid FROM turns")}
            remaining = sorted((s for s in samples if s["uid"] not in complete), key=lambda s: s["uid"] not in started)
            pending = iter(remaining[:limit])
            lock = threading.Lock()
            reasons = []
            throttle = Throttle()
            control.execute("UPDATE run SET state='running' WHERE id=1")
            control.commit()

            def worker():
                try:
                    with closing(_connect(directory)) as db:
                        while not stop.is_set():
                            with lock:
                                sample = next(pending, None)
                            if sample is None:
                                return
                            delays = iter(retry_delays)
                            while not stop.is_set():
                                try:
                                    _dialogue(root, db, store, sample, transport, stop, throttle)
                                    break
                                except CodexRateLimit as exc:
                                    delay = next(delays, None)
                                    if delay is None:
                                        raise
                                    throttle.defer(max(delay, getattr(exc, "retry_after", None) or 0))
                except BaseException as exc:
                    stop.set()
                    with lock:
                        reasons.append(exc)

            try:
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = [pool.submit(worker) for _ in range(min(workers, len(remaining), limit or len(remaining)))]
                    for future in futures:
                        future.result()
            finally:
                state = "queue_complete" if limit is None or limit >= len(remaining) else "segment_complete"
                if stop.is_set():
                    state = "operator_stop"
                if reasons:
                    state = "usage_limit" if all(isinstance(e, CodexUsageLimit) for e in reasons) else "invocation_error"
                    if all(isinstance(e, CodexRateLimit) for e in reasons):
                        state = "rate_limited"
                with control:
                    control.execute("UPDATE run SET state=? WHERE id=1", (state,))
            if reasons and state == "invocation_error":
                raise next(e for e in reasons if not isinstance(e, CodexUsageLimit))
    return report(root, digest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "status"))
    parser.add_argument("--root", default=ROOT, type=Path)
    parser.add_argument("--manifest-sha", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    root = args.root.resolve()
    if args.action == "status":
        result = report(root, args.manifest_sha)
    else:
        stop = threading.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.set())
        result = run(root, args.manifest_sha, limit=args.limit, workers=args.workers, stop=stop)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
