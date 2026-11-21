"""Durable, shared accounting for the first $70 Issue 13 credit (all amounts in nano-USD)."""
from contextlib import closing
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from zoneinfo import ZoneInfo

from ssvd.dialogue import pilot
from ssvd.dialogue.pilot_api import _FLEX_RATES


MODEL = "gpt-6.1-terra"
TIER = "flex"
UNIT_NANOS = dict(ordinary=500, cached=25, written=625, output=2500)
CREDIT = pilot.nanos("70")
_JST = ZoneInfo("Asia/Tokyo")
CUTOFF = int(datetime(2026, 12, 10, tzinfo=_JST).timestamp())


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def private_write(path: Path, data: bytes):
    """Atomically replace a private export and sync the directory entry."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as temp:
        tmp = Path(temp.name)
        try:
            temp.write(data)
            temp.flush()
            os.fsync(temp.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _open(path, mode):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=" + mode, uri=True, timeout=60)


def connect(run_dir, readonly=False):
    db = _open(Path(run_dir) / "production.sqlite3", "ro" if readonly else "rwc")
    if not readonly:
        db.execute("PRAGMA synchronous=FULL")
    return db


def _history_calls(db, clients, responses):
    """Check archived charges against receipts, retaining every unresolved maximum."""
    due = 0
    paid = 0
    for client, reserved, charged, raw in db.execute("SELECT client_id,reserved,charged,receipt FROM calls"):
        _require(isinstance(client, str) and client and client not in clients, "Duplicate or missing historical request")
        clients.add(client)
        _require(type(reserved) is int and reserved > 0, "Invalid historical reservation")
        if charged is None:
            _require(raw is None, "Unreconciled historical receipt")
            due += reserved
            continue
        _require(type(charged) is int and 0 <= charged <= reserved and raw, "Invalid historical charge")
        receipt = json.loads(raw)
        response = receipt["response_id"]
        _require(isinstance(response, str) and response and response not in responses
                 and pilot.nanos(receipt["cost_usd"]) == charged, "Duplicate or inconsistent historical receipt")
        responses.add(response)
        paid += 1
        due += charged
    return due, paid


def prior_commitment(root, run_dir):
    """Read the authoritative pilot and first-100 ledgers once, never their backup copies."""
    root, run_dir = Path(root), Path(run_dir)
    pilots = root / "data/curation/gpt_pilot_issue12"
    first = run_dir.parent / "first100_v10"
    versions = (*pilot.ARCHIVED_VERSIONS, pilot.PROMPT_VERSION)
    directories = {v: (pilots if v == pilot.LEGACY_VERSION else pilots / v) for v in versions}
    # initial_snapshot is an explicitly archived copy of the same requests.
    backup = pilots / "initial_snapshot/pilot.sqlite3"
    allowed = {d / "pilot.sqlite3" for d in directories.values()} | {backup}
    _require(set(pilots.rglob("*.sqlite3")) <= allowed, "Reconcile additional pilot ledgers before spending")
    clients, responses = set(), set()
    manifests = {}
    prior = 0
    for version, directory in directories.items():
        path = directory / "pilot.sqlite3"
        if not path.is_file():
            _require(version not in (pilot.LEGACY_VERSION, pilot.PROMPT_VERSION)
                     and not (directory / "manifest.json").exists(), "Restore missing pilot ledger")
            continue
        with closing(_open(path, "ro")) as db:
            db.execute("BEGIN")
            manifest = pilot.load_manifest(db, prompt_version=version)
            pilot.check_predecessors(manifest, manifests)
            manifests[version] = manifest
            prior += _history_calls(db, clients, responses)[0]
            if version == pilot.LEGACY_VERSION and backup.exists():
                saved_clients, saved_responses = set(), set()
                with closing(_open(backup, "ro")) as snapshot:
                    _history_calls(snapshot, saved_clients, saved_responses)
                _require(saved_clients <= clients and saved_responses <= responses,
                         "Restore historical calls present only in the pilot snapshot")

    contract = json.loads((first / "billing-contract.json").read_bytes())
    report = json.loads((first / "summary.json").read_bytes())
    quotas = contract["allocated_nano_usd"]
    _require(contract["model"] == MODEL and contract["all_runs_max_nano_usd"] == CREDIT
             and contract["shards"] == len(quotas) == 4 and type(contract["pilot_committed_nano_usd"]) is int
             and prior >= contract["pilot_committed_nano_usd"] > 0
             and all(type(q) is int and q > 0 for q in quotas)
             and sum(quotas) <= contract["issue13_max_nano_usd"] <= CREDIT, "Invalid first-100 billing contract")
    shards = [first / f"shard{i}.sqlite3" for i in range(len(quotas))]
    _require(set(run_dir.parent.rglob("*.sqlite3")) <= {*shards, run_dir / "production.sqlite3"}
             and all(path.is_file() for path in shards), "Restore missing or reconcile additional Issue 13 ledgers")
    first_total = receipts = 0
    for i, path in enumerate(shards):
        with closing(_open(path, "ro")) as db:
            db.execute("BEGIN")
            row = db.execute("SELECT manifest_sha, model, shard, cap, prior FROM limits WHERE id=1").fetchone()
            _require(row == (contract["manifest_sha"], MODEL, i, quotas[i], contract["pilot_committed_nano_usd"]),
                     "Unexpected first-100 shard contract")
            amount, paid = _history_calls(db, clients, responses)
            _require(amount <= quotas[i], "First-100 shard overspent")
            first_total += amount
            receipts += paid
    _require(report["remaining"] == 0 and report["paid_receipts"] == receipts
             and report["pairs"] * 2 == receipts and pilot.nanos(report["receipt_usd"]) == first_total
             and report["unresolved_reserve_usd"] == 0, "First-100 receipts differ from the completed wave")
    return prior + first_total


def _contract(digest, prior):
    _require(type(prior) is int and 0 <= prior < CREDIT, "No valid remaining project budget")
    _require(isinstance(digest, str) and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
             "Invalid frozen manifest checksum")
    _require(_FLEX_RATES[MODEL]._asdict() == UNIT_NANOS, "Flex rate card changed")
    return dict(model=MODEL, service_tier=TIER, unit_nano_usd_per_token=UNIT_NANOS, manifest_sha=digest,
                project_credit_nano_usd=CREDIT, prior_nano_usd=prior, production_max_nano_usd=CREDIT-prior,
                stop_before_jst=datetime.fromtimestamp(CUTOFF, _JST).isoformat(), prompt=pilot.PROMPT_VERSION)


def _limit_values(digest, prior):
    return (digest, MODEL, prior, CREDIT - prior, int(CUTOFF), TIER, pilot.encode(UNIT_NANOS))


def validate_existing(run_dir, manifest_sha, prior):
    """Perform preflight without creating or repairing billing state."""
    run_dir = Path(run_dir)
    _contract(manifest_sha, prior)
    path, bill = run_dir / "production.sqlite3", run_dir / "billing-contract.json"
    reports = (run_dir / "summary.json", run_dir / "pending_candidates.jsonl")
    if not path.is_file():
        _require(not (bill.exists() or any(p.exists() for p in reports)), "Restore the missing production ledger")
        return
    # Read the exported checkpoint before taking the SQLite snapshot. A
    # concurrent read-only preflight can then only see the same or newer receipts.
    summary = json.loads(reports[0].read_bytes()) if reports[0].exists() else None
    with closing(connect(run_dir, readonly=True)) as db:
        db.execute("BEGIN")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not tables:
            _require(not bill.exists() and not any(p.exists() for p in reports),
                     "Restore the empty production ledger")
            return
        old = db.execute("SELECT manifest_sha,model,prior,available,cutoff,service_tier,unit_nanos FROM limits WHERE id=1").fetchone()
        _require(old and type(old[2]) is int and 0 <= old[2] <= prior
                 and old == _limit_values(manifest_sha, old[2]), "Production billing contract differs")
        bad = db.execute("""SELECT 1 FROM calls WHERE typeof(reserved)!='integer' OR reserved<=0 OR
                (charged IS NOT NULL AND (typeof(charged)!='integer' OR charged<0 OR charged>reserved))
                OR model IS NOT ? LIMIT 1""", (MODEL,)).fetchone()
        _require(not bad and pilot.committed(db) <= CREDIT-prior, "Reconcile production spending before resuming")
        # SQLite is committed before the external contract is atomically published. A
        # first-time crash in this window may recover only while no money was reserved.
        if not bill.is_file():
            _require(not db.execute("SELECT 1 FROM calls LIMIT 1").fetchone() and not any(p.exists() for p in reports),
                     "Restore the production billing contract")
        else:
            saved = json.loads(bill.read_bytes())
            prior_saved = saved["prior_nano_usd"]
            _require(type(prior_saved) is int and 0 <= prior_saved <= old[2] and saved == _contract(manifest_sha, prior_saved),
                     "Published billing contract differs")
        if summary is not None:
            # Refuse to resume from a restored database older than already-published charges.
            n, charged = db.execute("SELECT COUNT(charged), COALESCE(SUM(charged),0) FROM calls").fetchone()
            _require(n >= summary["paid_receipts"] and charged >= pilot.nanos(summary["charged_usd"]),
                     "Production ledger is older than its published receipts")


def setup(run_dir, manifest_sha, prior):
    """Initialize/resume the existing SQLite format under the caller's exclusive run locks."""
    run_dir = Path(run_dir)
    contract = _contract(manifest_sha, prior)
    validate_existing(run_dir, manifest_sha, prior)
    with closing(connect(run_dir)) as db, db:
        db.executescript("""
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS limits(id INTEGER PRIMARY KEY CHECK(id=1),manifest_sha TEXT,model TEXT,
                prior INTEGER,available INTEGER,cutoff INTEGER,service_tier TEXT,unit_nanos TEXT);
            CREATE TABLE IF NOT EXISTS calls(uid TEXT,model TEXT,step INTEGER,request_hash TEXT NOT NULL,
                client_id TEXT NOT NULL UNIQUE,reserved INTEGER NOT NULL CHECK(reserved>0),charged INTEGER,
                reply TEXT,receipt TEXT,started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,finished_at TEXT,
                PRIMARY KEY(uid,model,step));
            CREATE TABLE IF NOT EXISTS outcomes(uid TEXT PRIMARY KEY,model TEXT NOT NULL,outcome TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS charges_cover ON calls(COALESCE(charged,reserved));
            DROP TRIGGER IF EXISTS cap;
            CREATE TRIGGER cap BEFORE INSERT ON calls BEGIN
                SELECT CASE WHEN NOT EXISTS(SELECT 1 FROM limits WHERE id=1 AND model=NEW.model)
                    OR typeof(NEW.reserved)!='integer' OR NEW.reserved<=0 OR
                    NEW.reserved+(SELECT COALESCE(SUM(COALESCE(charged,reserved)),0) FROM calls INDEXED BY charges_cover)
                    > (SELECT available FROM limits WHERE id=1)
                    THEN RAISE(ABORT,'production_budget_limit') END;
                SELECT CASE WHEN CAST(strftime('%s','now') AS INTEGER)>=(SELECT cutoff FROM limits WHERE id=1)
                    THEN RAISE(ABORT,'production_expired') END;
            END;
        """)
        db.execute("""INSERT INTO limits VALUES(1,?,?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                   prior=excluded.prior, available=excluded.available""", _limit_values(manifest_sha, prior))
        db.execute("""INSERT OR IGNORE INTO outcomes(uid, model, outcome)
                   SELECT uid,model,? FROM calls WHERE charged IS NULL""",
                   (pilot.encode({"status": "hold", "reason": "unresolved_billing"}),))
    private_write(run_dir/"billing-contract.json", (json.dumps(contract, indent=2)+"\n").encode())
