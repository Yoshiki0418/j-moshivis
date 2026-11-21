"""Prepare or audit the Issue 13 frozen Terra image queue, without API calls."""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import NamedTemporaryFile

from ssvd.ai_suggestion_store import human_touched_uids
from ssvd.dialogue import pilot_selection as selection
from ssvd.dialogue.preview_regeneration import COLLECTION, ROOT
from ssvd.dialogue.regeneration_roles import PROMPT_VERSION, default_pair_count
from ssvd.review_store import DEFAULT_REVIEW_EVENTS, REVIEW_REQUIRED_STATUS, ReviewStore, content_fingerprint
from jmoshivis.evaluation.schema import sha256_file

MODEL = "gpt-6.1-terra"
SEED = "japanese-daily-issue13-terra-production-v10-20261121"
PILOTS = Path("data/curation/gpt_pilot_issue12")
WAVES = Path("data/curation/gpt_regeneration_issue13")
DEFAULT_MANIFEST = WAVES / "production_v10_terra/manifest.json"
KINDS = ("counting", "ocr")
SHA256 = re.compile("[a-f0-9]{64}")


def _readonly(path):
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def load_manifest(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Frozen production manifest checksum differs")
    manifest = json.loads(raw)
    if (manifest.get("schema_version") != 1 or manifest.get("model") != MODEL
            or manifest.get("collection") != COLLECTION or manifest.get("prompt_version") != PROMPT_VERSION):
        raise ValueError("Unexpected production manifest contract")
    samples = manifest["samples"]
    seen_aliases, hashes, bases = set(), set(), set()
    kinds = Counter()
    for sample in samples:
        uid, kind = sample["uid"], sample["kind"]
        image = Path(sample["image_path"])
        digest, base = sample["image_sha256"], sample["base_fingerprint"]
        aliases = {key for key in (uid, sample.get("source_uid"), sample.get("split_group_key")) if key}
        if (not isinstance(uid, str) or not uid or kind not in KINDS
                or not isinstance(digest, str) or not SHA256.fullmatch(digest)
                or not isinstance(base, str) or not SHA256.fullmatch(base)
                or image.is_absolute() or ".." in image.parts or not image.name
                or type(sample["pairs"]) is not int or sample["pairs"] != default_pair_count(uid)):
            raise ValueError("Invalid source identity or dialogue length")
        if aliases & seen_aliases or digest in hashes or base in bases:
            raise ValueError("Repeated image, dialogue or group in production manifest")
        hashes.add(digest)
        bases.add(base)
        seen_aliases.update(aliases)
        kinds[kind] += 1
    if (not samples or manifest.get("sample_size") != len(samples)
            or manifest.get("pairs") != sum(s["pairs"] for s in samples)
            or kinds != Counter(manifest["sampling_target"])):
        raise ValueError("Manifest totals disagree with samples")
    return manifest


def _touched(root):
    path = root / DEFAULT_REVIEW_EVENTS
    if not path.is_file():
        raise FileNotFoundError(f"Missing human decision history: {path}")
    return {uid for collection, uid in human_touched_uids(ReviewStore(path)) if collection == COLLECTION}


def _previous(root, current):
    pilot, waves = root / PILOTS, root / WAVES
    first = waves / "first100_v10/manifest.json"
    # These preceding cohorts used paid calls. An incomplete history can select a duplicate.
    if not (pilot / "manifest.json").is_file() or len(json.loads(first.read_bytes())["samples"]) != 100:
        raise ValueError("Restore the original comparison and first-100 manifests")
    manifests = [*pilot.rglob("manifest.json"), *pilot.rglob("sample.json"), *waves.glob("*/manifest.json")]
    previous = {}
    for path in sorted(manifests):
        if path.resolve() == current.resolve():
            continue
        data = json.loads(path.read_bytes())
        for sample in data.get("samples", [data]):
            if "image_path" not in sample:
                continue
            prior = previous.setdefault(sample["uid"], sample)
            if prior["image_sha256"] != sample["image_sha256"]:
                raise ValueError("Conflicting source image in earlier queues")
    for path in [*pilot.rglob("pilot.sqlite3"), *waves.glob("*/*.sqlite3")]:
        if path.parent.resolve() == current.parent.resolve():
            continue
        with closing(_readonly(path)) as db:
            used = db.execute("SELECT uid FROM calls UNION SELECT uid FROM outcomes")
            if any(uid not in previous for uid, in used):
                raise ValueError("An earlier generation ledger has no corresponding source identity")
    return previous


def _source(db, root, uid):
    row = db.execute("""
        SELECT i.uid,i.source_uid,i.image_path,i.review_id,i.review_required,i.review_status,
               i.has_image,i.has_dialogue,i.auxiliary_kind,i.summary_json,
               i.dialogue_offset,i.dialogue_length,d.path AS dialogue_path,
               i.metadata_offset,i.metadata_length,m.path AS metadata_path
        FROM items i JOIN sources m ON i.metadata_source=m.source_id
        LEFT JOIN sources d ON i.dialogue_source=d.source_id WHERE i.collection=? AND i.uid=?
    """, (COLLECTION, uid)).fetchone()
    if row is None:
        raise ValueError("Missing original image in Japanese Daily catalog")

    def record(which):
        with (root / row[f"{which}_path"]).open("rb") as stream:
            stream.seek(row[f"{which}_offset"])
            return json.loads(stream.read(row[f"{which}_length"]))

    metadata = record("metadata")
    path = (root / row["image_path"]).resolve()
    fp = metadata["fingerprints"]
    if (metadata["uid"] not in (uid, row["source_uid"]) or (root / metadata["path"]).resolve() != path
            or not all(fp.get(k) for k in ("file_sha256", "pixel_sha256", "dhash", "phash"))
            or fp.get("error") or sha256_file(path) != fp["file_sha256"]):
        raise ValueError("Collected image identity has changed")
    source = record("dialogue") if row["dialogue_path"] else {}
    if source and (source["uid"] != uid or (root / source["image"]).resolve() != path):
        raise ValueError("Source dialogue no longer identifies the catalogued image")
    return row, fp, source, str(path.relative_to(root))


def _protected(db, root, current):
    previous = _previous(root, current)
    heldout, index, fixed = selection._fixed_evaluation(root)
    touched = _touched(root)
    blocked = heldout | touched | previous.keys()
    for uid in sorted(touched | previous.keys()):
        row, fp, source, path = _source(db, root, uid)
        if uid in previous and previous[uid]["image_sha256"] != fp["file_sha256"]:
            raise ValueError("An earlier queue used a different source image")
        blocked.update(key for key in (row["source_uid"], source.get("split_group_key")) if key)
        index.add(uid, fp, path)
    return blocked, index, fixed, touched, len(previous)


def prepare_manifest(root: Path, current: Path, limit: int, seed: str = SEED) -> dict:
    if limit < 1 or not seed:
        raise ValueError("Specify a positive image count and a nonempty seed")
    root = root.resolve()
    started = datetime.now(timezone.utc).isoformat()
    with closing(_readonly(root / "data/.dataset_viewer/catalog.sqlite3")) as db:
        db.row_factory = sqlite3.Row
        blocked, known, fixed, touched, previous = _protected(db, root, current)
        rows = selection._candidates(root, {(COLLECTION, uid) for uid in touched})
        pool = Counter(row["auxiliary_kind"] for row in rows)
        queues = {kind: [] for kind in KINDS}
        for row in rows:
            if row["uid"] not in blocked and row["source_uid"] not in blocked:
                queues[row["auxiliary_kind"]].append(row)
        available = {k: len(v) for k, v in queues.items()}
        total = sum(available.values())
        if total < limit:
            raise ValueError(f"Requested {limit} images, only {total} eligible before pixel matching")
        queues = {k: iter(sorted(v, key=lambda r: hashlib.sha256(f"{seed}:{r['uid']}".encode()).digest()))
                  for k, v in queues.items()}
        n_count = (limit * available["counting"] + total // 2) // total
        kept, skipped, samples = Counter(), Counter(), []
        selected_fingerprints = []
        for n in range(1, limit + 1):
            kind = "counting" if kept["counting"] < (n * n_count + limit // 2) // limit else "ocr"
            for hint_row in queues[kind]:
                uid = hint_row["uid"]
                row, fp, source, path = _source(db, root, uid)
                aliases = {key for key in (uid, row["source_uid"], source.get("split_group_key")) if key}
                if blocked & aliases:
                    skipped["identity"] += 1
                    continue
                digest = fp["file_sha256"]
                if (source.get("review_required") is not True or source["verification_status"] != REVIEW_REQUIRED_STATUS
                        or "qwen" not in source["generator"].lower() or source["auxiliary_task"]["kind"] != kind
                        or source["image_sha256"] != digest or content_fingerprint(source) != row["review_id"]
                        or Path(path).suffix.lower() not in (".jpeg", ".jpg", ".png", ".webp")):
                    raise ValueError("Candidate source eligibility changed")
                match = known.match(fp)["status"]
                if match != "accepted":
                    skipped[match] += 1
                    continue
                samples.append(dict(uid=uid, kind=kind, image_path=path, image_sha256=digest,
                    base_fingerprint=row["review_id"], pairs=default_pair_count(uid),
                    sampling_hint=selection._sampling_hints(root, hint_row)[0], source_uid=row["source_uid"],
                    split_group_key=source.get("split_group_key")))
                known.add(uid, fp, path)
                selected_fingerprints.append(fp)
                blocked.update(aliases)
                kept[kind] += 1
                break
            else:
                raise ValueError(f"Insufficient eligible {kind} images after overlap exclusions")
        newly_reviewed = _touched(root) - touched
        if newly_reviewed:
            selected = selection.KnownImageIndex(near_threshold=4, review_threshold=8)
            aliases = set()
            for sample, fp in zip(samples, selected_fingerprints):
                selected.add(sample["uid"], fp, sample["image_path"])
                aliases.update(k for k in (sample["uid"], sample["source_uid"], sample["split_group_key"]) if k)
            for uid in newly_reviewed:
                row, fp, source, _ = _source(db, root, uid)
                if (any(key in aliases for key in (uid, row["source_uid"], source.get("split_group_key")))
                        or selected.match(fp)["status"] != "accepted"):
                    raise ValueError("A selected image was reviewed during selection; rerun before freezing")
    return dict(schema_version=1, collection=COLLECTION, seed=seed, prompt_version=PROMPT_VERSION, model=MODEL,
                sampled_from_utc=started, created_utc=datetime.now(timezone.utc).isoformat(),
                candidate_pool=dict(pool), pool_without_previous_or_eval_id=available,
                sampling_target={"counting": n_count, "ocr": limit - n_count}, sample_size=limit,
                pairs=sum(s["pairs"] for s in samples), previous_pilot_and_first100_excluded=previous,
                human_reviewed_ids_excluded=len(touched), fixed_evaluation=fixed,
                overlap_thresholds=dict(near=4, review=8), overlap_exclusions_encountered=dict(skipped), samples=samples)


def _publish(path, manifest):
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
        try:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.link(name, path)  # Publish atomically without replacing an existing queue.
            fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            name.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def verify_manifest(root, path, digest):
    """Verify the original bytes and report current review eligibility separately."""
    manifest = load_manifest(path, digest)
    catalogue_pending = set()
    previous = _previous(root, path)
    heldout, index, fixed = selection._fixed_evaluation(root)
    if fixed != manifest["fixed_evaluation"] or manifest.get("overlap_thresholds") != {"near": 4, "review": 8}:
        raise ValueError("The frozen evaluation reference changed")
    with closing(_readonly(root / "data/.dataset_viewer/catalog.sqlite3")) as db:
        db.row_factory = sqlite3.Row
        for uid, sample in previous.items():
            row, fp, source, image = _source(db, root, uid)
            if fp["file_sha256"] != sample["image_sha256"]:
                raise ValueError("An earlier queue used a different image")
            heldout.update(k for k in (uid, row["source_uid"], source.get("split_group_key")) if k)
            index.add(uid, fp, image)
        for sample in manifest["samples"]:
            uid = sample["uid"]
            row, fp, source, image = _source(db, root, uid)
            group = source.get("split_group_key")
            aliases = {key for key in (uid, row["source_uid"], group) if key}
            if (image != sample["image_path"] or fp["file_sha256"] != sample["image_sha256"]
                    or source.get("image_sha256") != fp["file_sha256"]
                    or content_fingerprint(source) != sample["base_fingerprint"]
                    or source.get("auxiliary_task", {}).get("kind") != sample["kind"]
                    or source.get("review_required") is not True or source.get("verification_status") != REVIEW_REQUIRED_STATUS
                    or "qwen" not in source.get("generator", "").lower()
                    or group != sample["split_group_key"] or sample["source_uid"] != row["source_uid"]
                    or heldout & aliases or index.match(fp)["status"] != "accepted"):
                raise ValueError("Frozen source, duplicate, or held-out identity mismatch")
            heldout.update(aliases)
            index.add(uid, fp, image)
            summary = json.loads(row["summary_json"])
            if (row["review_status"] == "pending" and row["review_required"] and row["has_image"] and row["has_dialogue"]
                    and row["auxiliary_kind"] == sample["kind"] and row["review_id"] == sample["base_fingerprint"]
                    and summary.get("image_integrity_status") == "verified" and not summary.get("human_revision")):
                catalogue_pending.add(uid)
    ids = {sample["uid"] for sample in manifest["samples"]}
    human = _touched(root) & ids
    return dict(verified_samples=len(ids), sampling_target=manifest["sampling_target"],
                human_decisions_since_freeze=len(human), eligible_in_catalog=len(catalogue_pending - human),
                catalog_ineligible_without_human_decision=len(ids - catalogue_pending - human), sha256=digest)


def freeze_manifest(root, path, limit, seed):
    """Serialize new cohort publications while an earlier queue can continue generating."""
    waves = root / WAVES
    waves.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (waves / "cohort.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise FileExistsError("Manifest already exists; verify it with its recorded SHA-256")
        if any((path.parent / name).exists() for name in ("production.sqlite3", "billing-contract.json")):
            raise ValueError("Restore the original manifest for this existing production ledger")
        manifest = prepare_manifest(root, path, limit, seed)
        return manifest, _publish(path, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "freeze", "verify"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Absolute or relative to --root")
    parser.add_argument("--limit", type=int, default=3600)
    parser.add_argument("--seed", default=SEED)
    parser.add_argument("--sha256", help="Issue's recorded manifest digest (required by verify)")
    args = parser.parse_args()
    root = args.root.resolve()
    path = (root / args.manifest).resolve()
    if args.action == "verify":
        if not args.sha256:
            parser.error("verify requires --sha256")
        result = verify_manifest(root, path, args.sha256)
    else:
        if args.action == "freeze":
            manifest, digest = freeze_manifest(root, path, args.limit, args.seed)
        else:
            manifest = prepare_manifest(root, path, args.limit, args.seed)
        result = {k: v for k, v in manifest.items() if k != "samples"}
        if args.action == "freeze":
            result["sha256"] = digest
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
