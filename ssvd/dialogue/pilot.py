"""Run the fixed Issue #12 comparison, with resumable, budgeted Responses calls."""

import argparse
from collections import Counter
from contextlib import closing, contextmanager
from decimal import Decimal, ROUND_CEILING
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
from uuid import uuid4

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue.pilot_api import MODELS, OpenAIPilotAPI, PilotRequestRejected
from ssvd.dialogue.pilot_review import write_review
from ssvd.dialogue.pilot_selection import prepare_manifest
from ssvd.dialogue.preview_regeneration import COLLECTION, PROMPT_VERSION, ROOT, preview
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore

LEGACY_VERSION = "grounded_roles_v2"
LEGACY_RUN_DIR = ROOT / "data/curation/gpt_pilot_issue12"
RUN_DIR = LEGACY_RUN_DIR / PROMPT_VERSION
BUDGET_USD = Decimal("2.00")
NANOS = 10**9


def encode(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def nanos(value):
    amount = Decimal(str(value))
    if not amount.is_finite() or amount < 0:
        raise ValueError("Invalid billing amount")
    return int((amount * NANOS).to_integral_value(rounding=ROUND_CEILING))


@contextmanager
def operation(run_dir):
    run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (run_dir / "pilot.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another pilot operation is running") from None
        yield


def legacy_snapshot(run_dir):
    """Read the fixed predecessor's cohort and include even unsettled reservations."""
    if not (run_dir / "pilot.sqlite3").is_file():
        if (run_dir / "manifest.json").exists():
            raise RuntimeError("Legacy pilot ledger missing. Restore it before spending or preparing again.")
        return None, 0
    with closing(open_db(run_dir, readonly=True)) as legacy:
        legacy.execute("BEGIN")
        manifest = load_manifest(legacy, prompt_version=LEGACY_VERSION)
        if legacy.execute("SELECT 1 FROM calls WHERE charged>reserved LIMIT 1").fetchone():
            raise RuntimeError("Reconcile the legacy pilot price discrepancy before continuing.")
        return manifest, committed(legacy)


@contextmanager
def shared_operation(run_dir):
    # A process running the old generator locks the parent for its entire operation.
    with operation(run_dir.parent), operation(run_dir):
        yield legacy_snapshot(run_dir.parent)


def prepare(root, run_dir):
    with shared_operation(run_dir) as (previous_manifest, _):
        if (run_dir / "manifest.json").exists() and not (run_dir / "pilot.sqlite3").exists():
            raise RuntimeError("Pilot ledger missing. Restore the original ledger before preparing again.")
        with closing(sqlite3.connect(run_dir / "pilot.sqlite3")) as db:
            return prepare_cohort(root, run_dir, db, previous_manifest)


def prepare_cohort(root, run_dir, db, previous_manifest):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS cohort (
            id INTEGER PRIMARY KEY CHECK(id = 1), manifest TEXT NOT NULL,
            prompt TEXT NOT NULL, models TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calls (
            uid TEXT, model TEXT, step INTEGER, request_hash TEXT NOT NULL, client_id TEXT NOT NULL UNIQUE,
            reserved INTEGER NOT NULL, charged INTEGER, reply TEXT, receipt TEXT,
            started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, finished_at TEXT,
            PRIMARY KEY(uid, model, step));
        CREATE TABLE IF NOT EXISTS outcomes (
            uid TEXT, model TEXT, outcome TEXT NOT NULL, PRIMARY KEY(uid, model));
    """)
    if not db.execute("SELECT 1 FROM cohort").fetchone():
        if previous_manifest is None:
            manifest = prepare_manifest(root)
        else:
            manifest = {**previous_manifest, "budget_predecessor": LEGACY_VERSION}
        samples = manifest["samples"]
        if (manifest["schema_version"] != 1 or len({x["uid"] for x in samples}) != 20
                or Counter(x["kind"] for x in samples) != {"ocr": 14, "counting": 6}
                or any(type(x["pairs"]) is not int or not 4 <= x["pairs"] <= 8 for x in samples)):
            raise ValueError("Expected 20 different images (14 OCR, 6 counting), 4–8 pairs each")
        with db:
            db.execute("INSERT INTO cohort VALUES (1,?,?,?)",
                       (encode(manifest), PROMPT_VERSION, encode(MODELS)))
    manifest = load_manifest(db)
    check_predecessor(manifest, previous_manifest)
    # The database is authoritative. Re-export the same frozen list on a second prepare.
    temp = run_dir / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(run_dir / "manifest.json")
    return manifest


def open_db(run_dir, *, readonly=False):
    mode = "ro" if readonly else "rw"
    return sqlite3.connect((run_dir / "pilot.sqlite3").resolve().as_uri() + "?mode=" + mode, uri=True)


def load_manifest(db, *, prompt_version=PROMPT_VERSION):
    row = db.execute("SELECT manifest,prompt,models FROM cohort WHERE id=1").fetchone()
    if row is None or row[1:] != (prompt_version, encode(MODELS)):
        raise RuntimeError("Pilot not prepared for this prompt version and model pair")
    return json.loads(row[0])


def check_predecessor(manifest, previous_manifest):
    if manifest.get("budget_predecessor") and previous_manifest is None:
        raise RuntimeError("The legacy pilot ledger used by this cohort must be restored.")


def committed(db):
    return db.execute("SELECT COALESCE(sum(COALESCE(charged,reserved)),0) FROM calls").fetchone()[0]


def shared_budget_report(run_dir, db, manifest, prompt_version):
    if prompt_version == LEGACY_VERSION:
        return {}
    previous_manifest, previous_commitment = legacy_snapshot(run_dir.parent)
    check_predecessor(manifest, previous_manifest)
    total = previous_commitment + committed(db)
    return {"shared_budget": {"legacy_committed_usd": previous_commitment / NANOS,
                              "committed_usd": total / NANOS,
                              "remaining_usd": max(nanos(BUDGET_USD) - total, 0) / NANOS}}


def read_outcomes(db):
    return {(uid, model): json.loads(outcome)
            for uid, model, outcome in db.execute("SELECT uid,model,outcome FROM outcomes")}


def status(run_dir, *, prompt_version=PROMPT_VERSION):
    with closing(open_db(run_dir, readonly=True)) as db, db:
        db.execute("BEGIN")
        manifest = load_manifest(db, prompt_version=prompt_version)
        budget = shared_budget_report(run_dir, db, manifest, prompt_version)
        outcomes = read_outcomes(db)
        has_phase = any(col[1] == "phase" for col in db.execute("PRAGMA table_info(calls)"))
        initial = "phase='initial'" if has_phase else "1"
        settled, actual, uncertain, reserved = db.execute(f"""
            SELECT count(charged), COALESCE(sum(charged),0), count(*)-count(charged),
                   COALESCE(sum(CASE WHEN charged IS NULL THEN reserved END),0) FROM calls
            WHERE {initial}
        """).fetchone()
        comparison = {}
        for model in MODELS:
            results = [v for (_, m), v in outcomes.items() if m == model]
            task_mix, modes = Counter(), Counter()
            for outcome in results:
                if outcome["status"] == "pending_review":
                    tasks = outcome["grounding"]["tasks"]
                    task_mix["+".join(sorted(t["kind"] for t in tasks)) or "none"] += 1
                    modes.update(f"{t['kind']}/{t['mode']}" for t in tasks)
            receipts = db.execute(f"SELECT charged,receipt FROM calls WHERE model=? AND charged IS NOT NULL AND {initial}",
                                  (model,))
            tokens = Counter()
            cost = 0
            for charge, payload in receipts:
                cost += charge
                usage = json.loads(payload)["usage"]
                tokens.update({k: usage[k] for k in ("input_tokens", "output_tokens")})
            comparison[model] = dict(outcomes=dict(Counter(v["status"] for v in results)),
                                     generated_tasks=dict(task_mix), answer_modes=dict(modes),
                                     tokens=dict(tokens), cost_usd=cost/NANOS)
    return dict(images=len(manifest["samples"]), old_tasks=dict(Counter(s["kind"] for s in manifest["samples"])),
                models=comparison, remaining=len(manifest["samples"])*len(MODELS)-len(outcomes),
                settled_requests=settled, generation_cost_usd=actual/NANOS, unresolved_requests=uncertain,
                unresolved_reserve_usd=reserved/NANOS, budget_usd=float(BUDGET_USD), **budget)


def generate_one(root, db, sample, model, api, *, previous_commitment, recovery=False):
    uid = sample["uid"]
    store = ReviewStore(root / DEFAULT_REVIEW_EVENTS)
    identity = {key: sample[key] for key in ("uid", "kind", "pairs", "base_fingerprint", "image_sha256")}
    identity.update(collection=COLLECTION, model=model, prompt_version=PROMPT_VERSION)
    replies = []
    existing = 0
    if recovery:
        existing, last = db.execute("SELECT count(*),max(step) FROM calls WHERE uid=? AND model=?",
                                    (uid, model)).fetchone()
        if not existing:
            raise RuntimeError("Recovery requires the retained initial image observation")
        if last != existing - 1:
            raise RuntimeError("Stored pilot transcript has missing steps")
    while True:
        if not Path(store.path).is_file():
            raise RuntimeError("Human decision history is unavailable")
        try:
            stage = preview(root, uid, model=model, pairs=sample["pairs"],
                            transcript={"identity": identity, "replies": replies},
                            **({"recovery": True} if recovery else {}))
        except (ValueError, FileNotFoundError) as exc:
            # preview also validates the source and the immutable transcript identity.
            return {"status": "protected", "reason": str(exc)}
        if stage["status"] != "request":
            if recovery and len(replies) < existing:
                raise RuntimeError("Recovery stopped before consuming the stored pilot transcript")
            return stage
        body = api.prepare(stage["request"])
        request_hash = hashlib.sha256(encode(body).encode()).hexdigest()
        key = (uid, model, len(replies))
        saved = db.execute("SELECT request_hash,reply FROM calls WHERE uid=? AND model=? AND step=?", key).fetchone()
        if saved:
            if saved[0] != request_hash or saved[1] is None:
                raise RuntimeError("Unresolved or changed request. Reconcile the original call before continuing.")
            replies.append(json.loads(saved[1]))
            continue
        reserve = nanos(api.reserve_usd(body))
        if reserve <= 0:
            raise ValueError("Expected a positive reservation for a paid request")
        if previous_commitment + committed(db) + reserve > nanos(BUDGET_USD):
            raise RuntimeError("Pilot spending limit reached. No additional generation request was sent.")
        if not Path(store.path).is_file():
            raise RuntimeError("Human decision history is unavailable")
        if (COLLECTION, uid) in human_touched_uids(store):
            return {"status": "protected", "reason": "Human review recorded before submission"}
        # A crash or ambiguous network failure must leave a durable reservation.
        client_id = str(uuid4())
        with db:
            db.execute("INSERT INTO calls(uid,model,step,request_hash,client_id,reserved) VALUES(?,?,?,?,?,?)",
                       (*key, request_hash, client_id, reserve))
            if recovery:
                db.execute("UPDATE calls SET phase='recovery' WHERE uid=? AND model=? AND step=?", key)
        try:
            reply, receipt = api.generate(body, request_id=client_id)
        except PilotRequestRejected:
            with db:
                db.execute("DELETE FROM calls WHERE uid=? AND model=? AND step=? AND charged IS NULL", key)
            raise
        charge = nanos(receipt["cost_usd"])
        with db:
            db.execute("UPDATE calls SET reply=?,receipt=?,charged=?,finished_at=CURRENT_TIMESTAMP "
                       "WHERE uid=? AND model=? AND step=?",
                       (encode(reply), encode(receipt), charge, *key))
        if charge > reserve:
            raise RuntimeError("Reported cost exceeded the reservation. Check pricing before continuing.")
        replies.append(reply)


def run_pilot(root, run_dir, api):
    with shared_operation(run_dir) as (previous_manifest, previous_commitment), closing(open_db(run_dir)) as db:
        manifest = load_manifest(db)
        check_predecessor(manifest, previous_manifest)
        if db.execute("SELECT 1 FROM calls WHERE charged>reserved LIMIT 1").fetchone():
            raise RuntimeError("Reconcile the price discrepancy before resuming.")
        # On explicit restart, retire the interrupted chain. The original call can
        # never be submitted again, and its full reservation still counts toward the cap.
        with db:
            db.execute("""
                INSERT OR IGNORE INTO outcomes(uid,model,outcome)
                SELECT uid,model,? FROM calls WHERE charged IS NULL
            """, (encode({"status": "hold", "reason": "unresolved_billing"}),))
        finished = read_outcomes(db)
        for sample in manifest["samples"]:
            for model in MODELS:
                key = (sample["uid"], model)
                if key in finished:
                    continue
                result = generate_one(root, db, sample, model, api, previous_commitment=previous_commitment)
                with db:
                    db.execute("INSERT INTO outcomes VALUES(?,?,?)", (*key, encode(result)))
                print(f"{len(finished)+1}/{len(manifest['samples'])*len(MODELS)} {model} {result['status']}", flush=True)
                finished[key] = result
        return write_review(root, run_dir, manifest, finished)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    for name in ("status", "review", "recovery-status", "recovery-review"):
        commands.add_parser(name).add_argument("--legacy", action="store_true",
                                               help="Read the original observation-only pilot")
    for name in ("run", "recover"):
        commands.add_parser(name).add_argument("--api-key-file", type=Path,
                                               help="Private file containing only the project API key")
    args = parser.parse_args()
    legacy = getattr(args, "legacy", False)
    run_dir = LEGACY_RUN_DIR if legacy else RUN_DIR
    prompt_version = LEGACY_VERSION if legacy else PROMPT_VERSION
    os.umask(0o077)
    try:
        if args.command == "prepare":
            prepare(ROOT, RUN_DIR)
        elif args.command in ("run", "recover"):
            key_file = args.api_key_file or (Path.home() / ".config/j-moshivis/openai_api_key")
            key = key_file.read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("OPENAI_API_KEY", "").strip()
            if not key and key_file.is_file():
                key = key_file.read_text(encoding="utf-8").strip()
            if not key:
                raise RuntimeError("Set OPENAI_API_KEY or supply --api-key-file for the credited project")
            if args.command == "run":
                run_pilot(ROOT, RUN_DIR, OpenAIPilotAPI(key))
            else:
                from ssvd.dialogue.pilot_recovery import recover
                recover(ROOT, RUN_DIR, OpenAIPilotAPI(key))
        elif legacy and args.command in ("review", "recovery-review"):
            directory = run_dir / "recovery" if args.command == "recovery-review" else run_dir
            path = directory / "review.html"
            if not path.is_file():
                raise FileNotFoundError(f"Archived comparison page is unavailable: {path}")
            print(path)
        elif args.command == "review":
            with operation(run_dir), closing(open_db(run_dir, readonly=True)) as db:
                print(write_review(ROOT, run_dir, load_manifest(db), read_outcomes(db)))
        elif args.command == "recovery-review":
            from ssvd.dialogue.pilot_recovery import recovery_review
            print(recovery_review(ROOT, run_dir))
        if args.command in ("recover", "recovery-status", "recovery-review"):
            from ssvd.dialogue.pilot_recovery import recovery_status
            report = recovery_status(run_dir, prompt_version=prompt_version)
        else:
            report = status(run_dir, prompt_version=prompt_version)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
