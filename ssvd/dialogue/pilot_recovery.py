"""Continue the original pilot from verified paid replies using revised offline validation."""

from collections import Counter
from contextlib import closing
import json

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue import pilot
from ssvd.dialogue.pilot_review import write_review
from ssvd.dialogue.preview_regeneration import _load_source
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, ReviewStore


def _initial(db, *, prompt_version=pilot.PROMPT_VERSION):
    manifest = pilot.load_manifest(db, prompt_version=prompt_version)
    samples = manifest["samples"]
    expected = {(s["uid"], model) for s in samples for model in pilot.MODELS}
    initial = pilot.read_outcomes(db)
    if (len(samples) != 20 or len(expected) != 40 or initial.keys() != expected
            or Counter(s["kind"] for s in samples) != {"ocr": 14, "counting": 6}):
        raise RuntimeError("Recovery requires all 40 original outcomes for the fixed 20 images")
    if not all(row["status"] in ("pending_review", "hold", "protected") for row in initial.values()):
        raise RuntimeError("Unexpected original pilot outcome")
    return manifest, initial


def _started(db):
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='recovery_outcomes'").fetchone()


def _outcomes(db):
    if not _started(db):
        return {}
    return {(uid, model): json.loads(result)
            for uid, model, result in db.execute("SELECT uid,model,outcome FROM recovery_outcomes")}


def recover(root, run_dir, api):
    store = ReviewStore(root / DEFAULT_REVIEW_EVENTS)
    with pilot.shared_operation(run_dir) as (previous_manifest, previous_commitment), closing(pilot.open_db(run_dir)) as db:
        manifest, initial = _initial(db)
        pilot.check_predecessor(manifest, previous_manifest)
        if not store.path.is_file():
            raise RuntimeError("Human decision history is unavailable")
        if db.execute("SELECT 1 FROM calls WHERE charged>reserved LIMIT 1").fetchone():
            raise RuntimeError("Reconcile the price discrepancy before resuming.")
        # Install the recovery-only schema atomically, after checking the frozen original.
        with db:
            db.execute("BEGIN IMMEDIATE")
            if not _started(db):
                db.execute("ALTER TABLE calls ADD COLUMN phase TEXT NOT NULL DEFAULT 'initial' "
                           "CHECK(phase IN ('initial','recovery'))")
                db.execute("CREATE TABLE recovery_outcomes(uid TEXT, model TEXT, outcome TEXT NOT NULL, "
                           "PRIMARY KEY(uid,model))")
        finished = _outcomes(db)
        for sample in manifest["samples"]:
            uid = sample["uid"]
            for model in pilot.MODELS:
                key = uid, model
                if key in finished:
                    continue
                if not store.path.is_file():
                    raise RuntimeError("Human decision history is unavailable")
                original = initial[key]
                if (pilot.COLLECTION, uid) in human_touched_uids(store):
                    result = {"status": "protected", "reason": "Previously reviewed by a human"}
                elif original["status"] == "pending_review":
                    try:
                        source, kind, _, base = _load_source(root, uid)
                    except (ValueError, FileNotFoundError):
                        result = {"status": "protected", "reason": "Original source unavailable or reviewed"}
                    else:
                        if (kind, base, source["image_sha256"]) == (sample["kind"], sample["base_fingerprint"],
                                                                   sample["image_sha256"]):
                            result = original
                        else:
                            result = {"status": "protected", "reason": "Original source changed"}
                elif original["status"] == "protected":
                    result = original
                elif db.execute("SELECT 1 FROM calls WHERE uid=? AND model=? AND charged IS NULL", key).fetchone():
                    result = {"status": "hold", "reason": "unresolved_billing"}
                else:
                    result = pilot.generate_one(root, db, sample, model, api, recovery=True,
                                                previous_commitment=previous_commitment)
                with db:
                    db.execute("INSERT INTO recovery_outcomes VALUES(?,?,?)", (*key, pilot.encode(result)))
                finished[key] = result
                print(f"{len(finished)}/40 {model} {result['status']}", flush=True)
        return write_review(root, run_dir / "recovery", manifest, finished)


def recovery_status(run_dir, *, prompt_version=pilot.PROMPT_VERSION):
    with closing(pilot.open_db(run_dir, readonly=True)) as db, db:
        db.execute("BEGIN")
        manifest, initial = _initial(db, prompt_version=prompt_version)
        budget = pilot.shared_budget_report(run_dir, db, manifest, prompt_version)
        started = _started(db)
        finished = _outcomes(db)
        comparison = {}
        all_initial_cost = all_added_cost = all_extra_calls = 0
        for model in pilot.MODELS:
            original_counts, current_counts = Counter(), Counter()
            transitions = Counter()
            for sample in manifest["samples"]:
                key = sample["uid"], model
                original = initial[key]["status"]
                original_counts[original] += 1
                if key in finished:
                    current = finished[key]["status"]
                    current_counts[current] += 1
                    transitions[f"{original}->{current}"] += 1
            phase = "phase" if started else "'initial'"
            cost_rows = db.execute(f"SELECT {phase} AS cohort_phase,count(*),coalesce(sum(charged),0) "
                                   "FROM calls WHERE model=? GROUP BY cohort_phase", (model,))
            by_phase = {name: (count, cost) for name, count, cost in cost_rows}
            original_count, original_cost = by_phase.get("initial", (0, 0))
            new_count, new_cost = by_phase.get("recovery", (0, 0))
            all_initial_cost += original_cost
            all_added_cost += new_cost
            all_extra_calls += new_count
            comparison[model] = dict(initial=dict(original_counts), recovery=dict(current_counts),
                                     transitions=dict(transitions), initial_requests=original_count,
                                     additional_requests=new_count, additional_cost_usd=new_cost/pilot.NANOS)
        uncertain, reserve = db.execute("SELECT count(*),coalesce(sum(reserved),0) FROM calls WHERE charged IS NULL").fetchone()
        return dict(images=len(manifest["samples"]), remaining=40-len(finished), models=comparison,
                    initial_generation_cost_usd=all_initial_cost/pilot.NANOS,
                    additional_generation_cost_usd=all_added_cost/pilot.NANOS,
                    total_generation_cost_usd=(all_initial_cost + all_added_cost)/pilot.NANOS,
                    additional_requests=all_extra_calls, unresolved_requests=uncertain,
                    unresolved_reserve_usd=reserve/pilot.NANOS, budget_usd=float(pilot.BUDGET_USD), **budget)


def recovery_review(root, run_dir):
    with pilot.operation(run_dir), closing(pilot.open_db(run_dir, readonly=True)) as db:
        manifest, _ = _initial(db)
        return write_review(root, run_dir / "recovery", manifest, _outcomes(db))
