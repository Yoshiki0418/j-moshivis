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
from ssvd.dialogue.regeneration_roles import VISUAL_V11
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


def _enrolled_dialogues(db):
    """None means this ledger still uses its original prompt for every dialogue."""
    tables = db.execute("""SELECT name FROM sqlite_master WHERE type='table'
                        AND name IN ('dialogue_prompt_rollout','dialogue_prompt_versions')""").fetchall()
    if not tables:
        return None
    if len(tables) != 2:
        raise RuntimeError("Incomplete dialogue prompt rollout")
    policy = db.execute("SELECT prompt_version FROM dialogue_prompt_rollout WHERE id=1").fetchone()
    if policy != (VISUAL_V11,):
        raise RuntimeError("Incomplete or unsupported dialogue prompt rollout")
    enrolled = set()
    for uid, version in db.execute("SELECT uid,prompt_version FROM dialogue_prompt_versions"):
        if version != VISUAL_V11:
            raise RuntimeError("Unexpected prompt assigned to a production dialogue")
        enrolled.add(uid)
    return enrolled


def _activate_prompt(db, samples):
    """Enroll previously untouched UIDs once. An older generator must never spend against this rollout."""
    with db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("""CREATE TABLE dialogue_prompt_rollout(
                   id INTEGER PRIMARY KEY CHECK(id=1), prompt_version TEXT NOT NULL)""")
        db.execute("INSERT INTO dialogue_prompt_rollout VALUES(1,?)", (VISUAL_V11,))
        db.execute("CREATE TABLE dialogue_prompt_versions(uid TEXT PRIMARY KEY, prompt_version TEXT NOT NULL)")
        db.executemany("""INSERT INTO dialogue_prompt_versions(uid,prompt_version)
                       SELECT :uid,:version WHERE NOT EXISTS(SELECT 1 FROM calls WHERE uid=:uid)
                       AND NOT EXISTS(SELECT 1 FROM outcomes WHERE uid=:uid)""",
                       ({"uid": s["uid"], "version": VISUAL_V11} for s in samples))
        # Legacy writers do not register generation_prompt() on their SQLite connection.
        # A reservation from such a process must fail before a paid request leaves the machine.
        db.execute("""CREATE TRIGGER require_assigned_prompt BEFORE INSERT ON calls BEGIN
                   SELECT CASE WHEN NOT EXISTS (SELECT 1 FROM dialogue_prompt_versions
                       WHERE uid=NEW.uid AND prompt_version=generation_prompt())
                       THEN RAISE(ABORT,'production_dialogue_prompt_mismatch') END;
                   END""")


def _save_outcome(root, db, sample, outcome, prompt_version):
    uid = sample["uid"]
    if outcome["status"] == "pending_review":
        cand = outcome["candidate"]
        expected_identity = {k: sample[k] for k in ("uid", "kind", "pairs", "base_fingerprint", "image_sha256")}
        expected_identity.update(collection=pilot.COLLECTION, model=MODEL, prompt_version=prompt_version)
        if (outcome.get("identity") != expected_identity or cand["uid"] != uid or cand["image_sha256"] != sample["image_sha256"]
                or cand["review_required"] is not True or cand["review_id"] != content_fingerprint(cand)
                or cand["generator"] != MODEL or cand["prompt_version"] != prompt_version
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


def _generate(root, db, sample, previous, api, control, retry_delays, prompt_version):
    retries = 0
    failed_step = None
    while True:
        try:
            return pilot.generate_one(root, db, sample, MODEL, api, previous_commitment=previous,
                                      budget_nanos=CREDIT, before_reserve=control.before_reserve,
                                      prompt_version=prompt_version)
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


def _worker(root, run_dir, pending, previous, api_factory, control, retry_delays, prompt_version):
    try:
        api = api_factory()
        with closing(connect(run_dir)) as db:
            db.create_function("generation_prompt", 0, lambda: prompt_version)
            while not control.stop.is_set():
                with control.lock:
                    sample = next(pending, None)
                if sample is None:
                    return
                try:
                    outcome = _generate(root, db, sample, previous, api, control, retry_delays, prompt_version)
                    _save_outcome(root, db, sample, outcome, prompt_version)
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
        enrolled = _enrolled_dialogues(db)
        if enrolled is not None:
            started = {uid for uid, in db.execute("SELECT DISTINCT uid FROM calls")}
            deferred = len(started - results.keys() - enrolled)
    with locked_human_uids(ReviewStore(root / DEFAULT_REVIEW_EVENTS)) as touched:
        protected = {uid for collection, uid in touched if collection == pilot.COLLECTION}
    counts = Counter()
    user_turns_by_prompt = {}
    pairs = 0
    lines = []
    for sample in samples:
        uid = sample["uid"]
        if uid not in results:
            continue
        outcome = json.loads(results[uid])
        if outcome["status"] == "pending_review":
            candidate = outcome["candidate"]
            user_turns = candidate["dialogue"][::2]
            questions = sum(turn["text"].rstrip().endswith(("?", "？")) for turn in user_turns)
            metrics = user_turns_by_prompt.setdefault(candidate["prompt_version"], {
                "dialogues": 0, "user_turns": 0, "question_mark_endings": 0,
                "all_question_mark_endings": 0, "no_question_mark_endings": 0,
            })
            metrics["dialogues"] += 1
            metrics["user_turns"] += len(user_turns)
            metrics["question_mark_endings"] += questions
            metrics["all_question_mark_endings"] += int(questions == len(user_turns))
            metrics["no_question_mark_endings"] += int(questions == 0)
        status = "protected" if uid in protected else outcome["status"]
        counts[status] += 1
        if status == "pending_review":
            pairs += len(outcome["candidate"]["dialogue"]) // 2
            if publish:
                lines.append(pilot.encode(outcome["candidate"]) + "\n")
    total = previous + charged + reserved
    report = dict(model=MODEL, service_tier=TIER, state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
                  queue=len(samples), outcomes=dict(counts), pairs=pairs, unprocessed=len(samples) - sum(counts.values()),
                  user_turns_by_prompt=user_turns_by_prompt,
                  paid_receipts=settled, charged_usd=charged / pilot.NANOS, outstanding_calls=uncertain,
                  reserved_usd=reserved / pilot.NANOS, cumulative_committed_usd=total / pilot.NANOS,
                  remaining_usd=(CREDIT - total) / pilot.NANOS)
    report["new_dialogue_prompt"] = pilot.PROMPT_VERSION if enrolled is None else VISUAL_V11
    report["deferred_previous_prompt"] = 0 if enrolled is None else deferred
    if publish:
        private_write(run_dir / "pending_candidates.jsonl", "".join(lines).encode())
        private_write(run_dir / "summary.json", (json.dumps(report, indent=2) + "\n").encode())
    return report


def run(root, manifest_sha, api_factory, *, workers=4, retry_delays=RETRY_DELAYS, publish_interval=15, stop=None,
        new_dialogue_prompt=None):
    """Start or resume the fixed queue; a worker owns its API client and one dialogue at a time."""
    if workers not in range(1, 5) or publish_interval <= 0 or any(delay < 0 for delay in retry_delays):
        raise ValueError("Invalid production worker or delay settings")
    if new_dialogue_prompt not in (None, VISUAL_V11):
        raise ValueError("Unsupported prompt for newly started production dialogues")
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
        with closing(connect(run_dir)) as db:
            enrolled = _enrolled_dialogues(db)
            summary_path = run_dir / "summary.json"
            if enrolled is None and summary_path.is_file():
                published = json.loads(summary_path.read_bytes()).get("new_dialogue_prompt")
                if published == VISUAL_V11:
                    raise RuntimeError("Restore the production ledger that contains the published prompt rollout")
            if new_dialogue_prompt == VISUAL_V11 and enrolled is None:
                _activate_prompt(db, samples)
                enrolled = _enrolled_dialogues(db)
            completed = {uid for uid, in db.execute("SELECT uid FROM outcomes")}
            started = {uid for uid, in db.execute("SELECT DISTINCT uid FROM calls")}
        remaining = [s for s in samples if s["uid"] not in completed]
        deferred = 0
        prompt_version = pilot.PROMPT_VERSION
        if enrolled is not None:
            if any(s["uid"] not in enrolled and s["uid"] not in started for s in remaining):
                raise RuntimeError("Missing prompt assignment for an unstarted dialogue")
            deferred = len(started - completed - enrolled)
            remaining = [s for s in remaining if s["uid"] in enrolled]
            prompt_version = VISUAL_V11
        # Restore eligible partial dialogues before opening any fresh ones.
        pending = iter(sorted(remaining, key=lambda s: s["uid"] not in started))
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                try:
                    futures = {pool.submit(_worker, root, run_dir, pending, previous, api_factory, control,
                                           retry_delays, prompt_version) for _ in range(min(workers, len(remaining)))}
                    while futures:
                        done, futures = wait(futures, timeout=publish_interval, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                        if futures and not control.stop.is_set():
                            snapshot(root, samples, previous, "running", publish=True)
                except BaseException:
                    control.finish("interrupted")
                    raise
            control.finish("previous_prompt_deferred" if deferred else "queue_complete")
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
                  new_dialogue_prompt=pilot.PROMPT_VERSION, deferred_previous_prompt=0,
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
    parser.add_argument("--new-dialogue-prompt", choices=(VISUAL_V11,),
                        help="With run, persistently switch unstarted dialogues to this prompt")
    args = parser.parse_args(argv)
    if args.new_dialogue_prompt is not None and args.action != "run":
        parser.error("--new-dialogue-prompt applies to run only")
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
            workers=args.workers, stop=stop, new_dialogue_prompt=args.new_dialogue_prompt)
        return 0
    except (OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
