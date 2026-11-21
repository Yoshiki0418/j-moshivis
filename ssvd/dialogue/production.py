"""Generate human-review candidates from the frozen Terra queue. Only 'run' calls the API."""

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import threading

from ssvd.ai_suggestion_store import locked_human_uids
from ssvd.dialogue import pilot, preview_regeneration
from ssvd.dialogue.pilot_api import OpenAIPilotAPI, PilotRequestRejected
from ssvd.dialogue.production_budget import (
    CREDIT, CUTOFF, MODEL, TIER, connect, prior_commitment, private_write, setup, validate_existing,
)
from ssvd.dialogue.production_cohort import DEFAULT_MANIFEST, PILOTS, load_manifest, verify_manifest
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore, content_fingerprint

RUN_PATH = DEFAULT_MANIFEST.parent
RETRY_DELAYS = (5, 15, 30, 60, 120, 300)


class _RunControl:
    def __init__(self, stop):
        self.stop = stop
        self.reason = "running"
        self.lock = threading.Lock()

    def finish(self, reason):
        with self.lock:
            if self.reason == "running":
                self.reason = "operator_stop" if self.stop.is_set() else reason
            self.stop.set()

    def before_reserve(self):
        if datetime.now(timezone.utc).timestamp() >= CUTOFF:
            self.finish("credit_deadline")
        if self.stop.is_set():
            raise InterruptedError("Generation stopped before the next reservation")


def _save_outcome(root, db, sample, outcome):
    uid = sample["uid"]
    if outcome["status"] == "pending_review":
        cand = outcome["candidate"]
        if (cand["uid"] != uid or cand["image_sha256"] != sample["image_sha256"]
                or cand["review_required"] is not True or cand["review_id"] != content_fingerprint(cand)
                or cand["generator"] != MODEL or cand["prompt_version"] != pilot.PROMPT_VERSION
                or len(cand["dialogue"]) != 2 * sample["pairs"]):
            raise RuntimeError("Generated candidate does not match the frozen source")
        try:
            _, kind, _, base = preview_regeneration._load_source(root, uid)
        except (ValueError, FileNotFoundError):
            kind, base = None, None
        if (kind, base) != (sample["kind"], sample["base_fingerprint"]):
            outcome = {"status": "protected", "reason": "Source or human decision changed"}
    # ReviewStore writers lock the same file exclusively. No network or image IO under this lock.
    with locked_human_uids(ReviewStore(root / DEFAULT_REVIEW_EVENTS)) as touched, db:
        if (pilot.COLLECTION, uid) in touched:
            outcome = {"status": "protected", "reason": "Human review recorded before publication"}
        db.execute("INSERT INTO outcomes(uid,model,outcome) VALUES(?,?,?)", (uid, MODEL, pilot.encode(outcome)))


def _generate(root, db, sample, previous, api, control, retry_delays):
    retries = 0
    failed_step = None
    while True:
        try:
            return pilot.generate_one(root, db, sample, MODEL, api, previous_commitment=previous,
                                      budget_nanos=CREDIT, before_reserve=control.before_reserve)
        except PilotRequestRejected as exc:
            # Only definitive capacity rejections can be retried. A confirmed quota failure needs intervention.
            if "(HTTP 429)" not in str(exc) or exc.code == "insufficient_quota":
                raise
            step = db.execute("SELECT count(*) FROM calls WHERE uid=? AND model=?",
                              (sample["uid"], MODEL)).fetchone()[0]
            if step != failed_step:
                retries, failed_step = 0, step
            if retries == len(retry_delays):
                control.finish("rate_limit")
                raise RuntimeError("Flex retry limit reached; resume reuses previously settled turns") from exc
            delay = retry_delays[retries]
            retries += 1
            print(f"Flex capacity unavailable; retry {retries}/{len(retry_delays)} in {delay}s", flush=True)
            if control.stop.wait(delay):
                raise InterruptedError("Stopped during capacity backoff") from None


def _worker(root, run_dir, pending, previous, api_factory, control, retry_delays):
    try:
        api = api_factory()
        with closing(connect(run_dir)) as db:
            while not control.stop.is_set():
                with control.lock:
                    sample = next(pending, None)
                if sample is None:
                    return
                try:
                    outcome = _generate(root, db, sample, previous, api, control, retry_delays)
                    _save_outcome(root, db, sample, outcome)
                except pilot.SpendingLimitReached:
                    control.finish("budget_reached")
                    return
                except sqlite3.IntegrityError as error:
                    reasons = {"production_budget_limit": "budget_reached", "production_expired": "credit_deadline"}
                    if str(error) not in reasons:
                        raise
                    control.finish(reasons[str(error)])
                    return
                except InterruptedError:
                    if control.stop.is_set():
                        return
                    raise
                except RuntimeError:
                    unresolved = db.execute("SELECT 1 FROM calls WHERE uid=? AND charged IS NULL LIMIT 1",
                                            (sample["uid"],)).fetchone()
                    if unresolved:
                        control.finish("transport_needs_reconciliation")
                        with db:
                            db.execute("INSERT INTO outcomes(uid,model,outcome) VALUES(?,?,?)", (
                                sample["uid"], MODEL, pilot.encode({"status": "hold", "reason": "unresolved_billing"})))
                    raise
    except BaseException:
        control.finish("request_error")
        raise


def snapshot(root, samples, previous, state="snapshot", *, publish=False):
    run_dir = root / RUN_PATH
    with closing(connect(run_dir, readonly=True)) as db:
        db.execute("BEGIN")
        settled, charged, uncertain, reserved = db.execute("""
            SELECT COUNT(charged), COALESCE(SUM(charged),0), COUNT(*)-COUNT(charged),
                COALESCE(SUM(CASE WHEN charged IS NULL THEN reserved ELSE 0 END),0)
            FROM calls""").fetchone()
        results = dict(db.execute("SELECT uid,outcome FROM outcomes"))
    with locked_human_uids(ReviewStore(root / DEFAULT_REVIEW_EVENTS)) as touched:
        protected = {uid for collection, uid in touched if collection == pilot.COLLECTION}
    counts = Counter()
    pairs = 0
    lines = []
    for sample in samples:
        uid = sample["uid"]
        if uid not in results:
            continue
        outcome = json.loads(results[uid])
        status = "protected" if uid in protected else outcome["status"]
        counts[status] += 1
        if status == "pending_review":
            pairs += len(outcome["candidate"]["dialogue"]) // 2
            if publish:
                lines.append(pilot.encode(outcome["candidate"]) + "\n")
    total = previous + charged + reserved
    report = dict(model=MODEL, service_tier=TIER, state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                  queue=len(samples), outcomes=dict(counts), pairs=pairs, unprocessed=len(samples) - sum(counts.values()),
                  paid_receipts=settled, charged_usd=charged / pilot.NANOS, outstanding_calls=uncertain,
                  reserved_usd=reserved / pilot.NANOS, cumulative_committed_usd=total / pilot.NANOS,
                  remaining_usd=(CREDIT - total) / pilot.NANOS)
    if publish:
        private_write(run_dir / "pending_candidates.jsonl", "".join(lines).encode())
        private_write(run_dir / "summary.json", (json.dumps(report, indent=2) + "\n").encode())
    return report


def run(root, manifest_sha, api_factory, *, workers=4, retry_delays=RETRY_DELAYS, publish_interval=15, stop=None):
    """Start or resume the fixed queue; a worker owns its API client and one dialogue at a time."""
    if workers not in range(1, 5) or publish_interval <= 0 or any(delay < 0 for delay in retry_delays):
        raise ValueError("Invalid production worker or delay settings")
    root = Path(root).resolve()
    run_dir = root / RUN_PATH
    control = _RunControl(stop if stop is not None else threading.Event())
    with pilot.operation(root / PILOTS), pilot.operation(run_dir.parent), pilot.operation(run_dir):
        samples = load_manifest(run_dir / "manifest.json", manifest_sha)["samples"]
        if not (root / DEFAULT_REVIEW_EVENTS).is_file():
            raise RuntimeError("Human decision history is unavailable")
        if datetime.now(timezone.utc).timestamp() >= CUTOFF:
            raise RuntimeError("The credited generation period has ended")
        previous = prior_commitment(root, run_dir)
        if not (run_dir / "billing-contract.json").is_file():
            verify_manifest(root, run_dir / "manifest.json", manifest_sha)
        setup(run_dir, manifest_sha, previous)
        with closing(connect(run_dir, readonly=True)) as db:
            completed = {uid for uid, in db.execute("SELECT uid FROM outcomes")}
            started = {uid for uid, in db.execute("SELECT DISTINCT uid FROM calls")}
        # Restore settled partial dialogues before opening any fresh ones.
        pending = iter(sorted((s for s in samples if s["uid"] not in completed), key=lambda s: s["uid"] not in started))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    futures = {pool.submit(_worker, root, run_dir, pending, previous, api_factory, control, retry_delays)
                               for _ in range(workers)}
                    while futures:
                        done, futures = wait(futures, timeout=publish_interval, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                        if futures and not control.stop.is_set():
                            snapshot(root, samples, previous, "running", publish=True)
                except BaseException:
                    control.finish("interrupted")
                    raise
            control.finish("queue_complete")
        finally:
            report = snapshot(root, samples, previous, control.reason, publish=True)
            print(json.dumps(report, ensure_ascii=False), flush=True)
        return report


def inspect_run(root, manifest_sha, *, verify_sources=False):
    """Read the current contract and live ledger without taking the writer's operation locks."""
    root = Path(root).resolve()
    run_dir = root / RUN_PATH
    manifest_path = run_dir / "manifest.json"
    samples = load_manifest(manifest_path, manifest_sha)["samples"]
    previous = prior_commitment(root, run_dir)
    validate_existing(run_dir, manifest_sha, previous)
    report = dict(model=MODEL, service_tier=TIER, state="prepared", queue=len(samples), unprocessed=len(samples),
                  cumulative_committed_usd=previous / pilot.NANOS, remaining_usd=(CREDIT - previous) / pilot.NANOS)
    if (run_dir / "production.sqlite3").is_file():
        with closing(connect(run_dir, readonly=True)) as db:
            initialized = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='limits'").fetchone()
        if initialized:
            report = snapshot(root, samples, previous)
    if verify_sources:
        report["cohort"] = verify_manifest(root, manifest_path, manifest_sha)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "preflight", "run"))
    parser.add_argument("--root", type=Path, default=pilot.ROOT)
    parser.add_argument("--manifest-sha", required=True, help="SHA-256 of the frozen manifest.json")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4, help="Concurrent dialogues for run")
    parser.add_argument("--api-key-file", type=Path, help="File containing only the project API key")
    args = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if args.action != "run":
            report = inspect_run(args.root, args.manifest_sha, verify_sources=args.action == "preflight")
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0
        keyfile = args.api_key_file or args.root / ".config/openai_api_key.env"
        if keyfile.stat().st_mode & 0o077:
            raise ValueError("The API key file must be private")
        key = keyfile.read_text().strip()
        stop = threading.Event()
        def request_stop(_signum, _frame):
            stop.set()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        run(args.root, args.manifest_sha, lambda: OpenAIPilotAPI(key, service_tier=TIER),
            workers=args.workers, stop=stop)
        return 0
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
