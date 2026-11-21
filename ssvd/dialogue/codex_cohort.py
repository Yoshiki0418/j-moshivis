"""Select the remaining Japanese Daily images for the ChatGPT-authenticated Terra CLI."""

import argparse
import fcntl
import json
from pathlib import Path

from ssvd.dialogue import production_cohort as cohort
from ssvd.dialogue.production_cohort import CLI_MANIFEST
from ssvd.dialogue.regeneration_roles import VISUAL_V11

SEED = "japanese-daily-terra-codex-v11"
BACKEND = "codex_cli"


def load_cli_manifest(path: Path, expected_sha: str) -> dict:
    manifest = cohort.load_manifest(path, expected_sha, prompt_version=VISUAL_V11)
    if (manifest.get("backend") != BACKEND
            or not cohort.SHA256.fullmatch(manifest.get("excluded_api_manifest_sha256") or "")):
        raise ValueError("Unexpected Codex CLI manifest contract")
    return manifest


def _api_digest(root):
    manifest = root / cohort.DEFAULT_MANIFEST
    billing = json.loads((manifest.parent / "billing-contract.json").read_bytes())
    digest = billing["manifest_sha"]
    cohort.load_manifest(manifest, digest)
    return digest


def prepare(root):
    digest = _api_digest(root)
    manifest = cohort.prepare_manifest(root, root / CLI_MANIFEST, limit=None, seed=SEED)
    manifest.update(prompt_version=VISUAL_V11, backend=BACKEND, excluded_api_manifest_sha256=digest)
    return manifest


def freeze(root):
    root = root.resolve()
    path = root / CLI_MANIFEST
    # Both backends reserve their work while holding this same cohort lock.
    # The running API workers hold separate locks and continue undisturbed.
    with (root / cohort.WAVES / "cohort.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            raise FileExistsError("CLI cohort is already frozen; verify it with the recorded SHA-256")
        if any((path.parent / name).exists() for name in ("generation.sqlite3", "run_contract.json")):
            raise ValueError("Restore the CLI manifest for the existing generation ledger")
        manifest = prepare(root)
        return manifest, cohort._publish(path, manifest)


def verify(root, digest):
    path = root / CLI_MANIFEST
    manifest = load_cli_manifest(path, digest)
    if manifest["excluded_api_manifest_sha256"] != _api_digest(root):
        raise ValueError("Paid API cohort changed since CLI selection")
    return cohort.verify_manifest(root, path, digest, prompt_version=VISUAL_V11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("dry-run", "freeze", "verify"))
    parser.add_argument("--root", type=Path, default=cohort.ROOT)
    parser.add_argument("--sha256", help="Recorded CLI digest for verify")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "verify":
        if not args.sha256:
            parser.error("--sha256 is required for verify")
        summary = verify(root, args.sha256)
    else:
        if args.action == "freeze":
            manifest, digest = freeze(root)
        else:
            manifest, digest = prepare(root), None
        summary = {key: value for key, value in manifest.items() if key != "samples"}
        if digest:
            summary["sha256"] = digest
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
