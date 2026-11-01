#!/usr/bin/env python3
"""Collect license-auditable Japanese everyday images from Commons.

The pipeline is deliberately split in two:

1. ``harvest`` records Wikimedia Commons file page IDs using category trees
   and geosearch. No image is downloaded at this stage.
2. ``download`` resolves current per-file licensing metadata, downloads a
   1536-pixel derivative, validates it, and excludes exact/perceptual overlap
   with v3, targeted-v4, and already accepted images. Every eligible image is
   retained; there is no fixed 50k acceptance cap.

Collection buckets are *sampling quotas*, not task labels. OCR/counting
questions are chosen later by the adaptive VLM dialogue generator. A Commons
category or location alone is never treated as a canonical OCR/count answer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import random
import re
import sys
import tempfile
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import requests
from PIL import Image, ImageStat, UnidentifiedImageError


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "j-moshivis"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from jmoshivis.evaluation.overlap.fingerprints import (  # noqa: E402
    fingerprint_provenance,
    hamming,
    image_fingerprints,
)


SCHEMA_VERSION = "jmoshivis.japanese-daily-images.v1"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = (
    "J-MoshiVis-DatasetBuilder/2.0 "
    "(Japanese multimodal research; yoshikiyamamoto06@gmail.com)"
)
ALLOWED_MIME = {"image/jpeg", "image/png"}
ALLOWED_LICENSE = re.compile(
    r"^(?:CC0(?:\s|$)|CC BY(?:-SA)?(?:\s|$)|PUBLIC DOMAIN(?:\s|$)|PD(?:\s|$))",
    re.IGNORECASE,
)
TERMINAL_STATUSES = {
    "accepted",
    "excluded_license",
    "excluded_license_metadata",
    "excluded_rights_restrictions",
    "excluded_mime",
    "excluded_source_dimensions",
    "excluded_decoded_dimensions",
    "excluded_aspect_ratio",
    "excluded_low_information",
    "excluded_exact_overlap",
    "excluded_near_overlap",
    "review_perceptual_overlap",
    "excluded_api_missing",
    "excluded_download_failure",
}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", buffering=8 * 1024 * 1024) as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_attribution_csv(manifest_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(name)
    fields = [
        "uid",
        "title",
        "artist",
        "credit",
        "attribution",
        "license",
        "license_url",
        "usage_terms",
        "attribution_required",
        "source_url",
        "original_url",
        "source_pageid",
        "source_lastrevid",
        "source_timestamp",
        "source_sha1",
        "path",
    ]
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in read_jsonl(manifest_path):
                writer.writerow({field: row.get(field, "") for field in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def clean_html(value: str | None) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extmetadata_value(info: Mapping[str, Any], key: str) -> str:
    value = info.get("extmetadata", {}).get(key, {})
    return clean_html(value.get("value") if isinstance(value, dict) else "")


def source_uid(pageid: int | str) -> str:
    return hashlib.sha256(f"wikimedia_commons:{pageid}".encode()).hexdigest()


class CommonsClient:
    def __init__(self, *, sleep_seconds: float = 0.12) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.sleep_seconds = sleep_seconds

    def get(self, params: Mapping[str, Any]) -> dict[str, Any]:
        query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        last_error: BaseException | None = None
        for attempt in range(7):
            try:
                response = self.session.get(COMMONS_API, params=query, timeout=45)
                if response.status_code in {429, 500, 502, 503, 504}:
                    retry = float(response.headers.get("Retry-After", 2 ** attempt))
                    time.sleep(min(60.0, max(1.0, retry)))
                    continue
                response.raise_for_status()
                value = response.json()
                if "error" in value:
                    code = value["error"].get("code", "unknown")
                    if code == "maxlag":
                        time.sleep(min(60.0, 2 ** attempt))
                        continue
                    raise RuntimeError(f"Commons API {code}: {value['error']}")
                time.sleep(self.sleep_seconds)
                return value
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = exc
                time.sleep(min(60.0, 2 ** attempt))
        raise RuntimeError(f"Commons API failed after retries: {last_error}")

    def category_members(self, category: str) -> Iterator[dict[str, Any]]:
        continuation: dict[str, Any] = {}
        while True:
            value = self.get(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmtype": "file|subcat",
                    "cmlimit": 500,
                    **continuation,
                }
            )
            yield from value.get("query", {}).get("categorymembers", [])
            if "continue" not in value:
                return
            continuation = value["continue"]

    def geosearch_files(self, lat: float, lon: float) -> Iterator[dict[str, Any]]:
        continuation: dict[str, Any] = {}
        while True:
            value = self.get(
                {
                    "action": "query",
                    "list": "geosearch",
                    "gscoord": f"{lat}|{lon}",
                    "gsradius": 10_000,
                    "gslimit": 500,
                    "gsnamespace": 6,
                    "gsprimary": "all",
                    **continuation,
                }
            )
            yield from value.get("query", {}).get("geosearch", [])
            if "continue" not in value:
                return
            continuation = value["continue"]

    def image_info(self, pageids: list[int], width: int) -> list[dict[str, Any]]:
        value = self.get(
            {
                "action": "query",
                "pageids": "|".join(str(pageid) for pageid in pageids),
                "prop": "imageinfo|info",
                "iiprop": "url|size|mime|sha1|timestamp|extmetadata",
                "iiurlwidth": width,
                "iiextmetadatalanguage": "en",
                "iiextmetadatafilter": "|".join(
                    [
                        "LicenseShortName",
                        "LicenseUrl",
                        "Artist",
                        "Credit",
                        "Attribution",
                        "AttributionRequired",
                        "UsageTerms",
                        "Restrictions",
                        "Copyrighted",
                        "DateTimeOriginal",
                        "DateTime",
                        "Categories",
                        "ImageDescription",
                    ]
                ),
            }
        )
        return value.get("query", {}).get("pages", [])


def load_plan(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if sum(int(bucket["accepted_target"]) for bucket in value["buckets"]) != int(
        value["accepted_target"]
    ):
        raise ValueError("bucket accepted targets do not sum to accepted_target")
    return value


def add_candidate(
    candidates: dict[int, dict[str, Any]],
    member: Mapping[str, Any],
    *,
    bucket: str,
    method: str,
    seed: str,
    discovered_in: str,
) -> tuple[bool, bool]:
    pageid = int(member["pageid"])
    is_new_candidate = pageid not in candidates
    row = candidates.setdefault(
        pageid,
        {
            "source": "wikimedia_commons",
            "source_pageid": pageid,
            "source_uid": source_uid(pageid),
            "title": str(member["title"]),
            "collection_buckets": [],
            "collection_seeds": [],
            "discoveries": [],
        },
    )
    is_new_bucket = bucket not in row["collection_buckets"]
    if is_new_bucket:
        row["collection_buckets"].append(bucket)
    if seed not in row["collection_seeds"]:
        row["collection_seeds"].append(seed)
    discovery = {"method": method, "seed": seed, "container": discovered_in}
    if discovery not in row["discoveries"] and len(row["discoveries"]) < 8:
        row["discoveries"].append(discovery)
    return is_new_candidate, is_new_bucket


def harvest(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    client = CommonsClient(sleep_seconds=args.api_sleep)
    candidates: dict[int, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    missing_roots: list[str] = []
    bucket_memberships: Counter[str] = Counter()

    # Secure geographic diversity first. Otherwise a very large category tree
    # such as railway stations can consume the entire everyday candidate quota.
    everyday_bucket = "japanese_everyday"
    for location in plan.get("geosearch_locations", []):
        if (
            len(candidates) >= args.max_total_candidates
            or bucket_memberships[everyday_bucket] >= args.max_geosearch_candidates
        ):
            break
        seed = f"geosearch:{location['name']}"
        found = 0
        try:
            for member in client.geosearch_files(
                float(location["lat"]), float(location["lon"])
            ):
                is_new, is_new_bucket = add_candidate(
                    candidates,
                    member,
                    bucket=everyday_bucket,
                    method="geosearch",
                    seed=seed,
                    discovered_in=str(location["name"]),
                )
                found += int(is_new)
                bucket_memberships[everyday_bucket] += int(is_new_bucket)
                if (
                    len(candidates) >= args.max_total_candidates
                    or bucket_memberships[everyday_bucket]
                    >= args.max_geosearch_candidates
                    or found >= args.max_geosearch_per_location
                ):
                    break
        except RuntimeError as exc:
            print(f"[warn] geosearch failed: {seed}: {exc}", flush=True)
            stats["geosearch_errors"] += 1
        print(
            f"[harvest] {seed}: new_files={found:,}, unique_total={len(candidates):,}",
            flush=True,
        )
        stats["geosearch_locations_completed"] += 1

    # Interleave roots from all three buckets. This prevents one large tree
    # from consuming the global limit and makes 125,000 refer to unique file
    # page IDs, not the sum of overlapping bucket memberships.
    bucket_configs = list(plan["buckets"])
    root_rounds = max(len(bucket["category_roots"]) for bucket in bucket_configs)
    for root_index in range(root_rounds):
        for bucket in bucket_configs:
            if len(candidates) >= args.max_total_candidates:
                break
            roots = bucket["category_roots"]
            if root_index >= len(roots):
                continue
            bucket_name = str(bucket["name"])
            depth_limit = int(bucket["category_depth"])
            root = str(roots[root_index])
            queue: deque[tuple[str, int]] = deque([(str(root), 0)])
            visited: set[str] = set()
            seed_files = 0
            while (
                queue
                and seed_files < args.max_per_seed
                and len(candidates) < args.max_total_candidates
            ):
                category, depth = queue.popleft()
                if category in visited or len(visited) >= args.max_categories_per_seed:
                    continue
                visited.add(category)
                found_any = False
                try:
                    for member in client.category_members(category):
                        found_any = True
                        if int(member["ns"]) == 6:
                            is_new, is_new_bucket = add_candidate(
                                candidates,
                                member,
                                bucket=bucket_name,
                                method="category",
                                seed=str(root),
                                discovered_in=category,
                            )
                            seed_files += int(is_new)
                            bucket_memberships[bucket_name] += int(is_new_bucket)
                            if (
                                seed_files >= args.max_per_seed
                                or len(candidates) >= args.max_total_candidates
                            ):
                                break
                        elif int(member["ns"]) == 14 and depth < depth_limit:
                            queue.append((str(member["title"]), depth + 1))
                except RuntimeError as exc:
                    print(f"[warn] category failed: {category}: {exc}", flush=True)
                    stats["category_errors"] += 1
                    continue
                if category == root and not found_any:
                    missing_roots.append(str(root))
                if len(visited) % 100 == 0:
                    print(
                        f"[harvest] {bucket_name} root={root!r} categories={len(visited):,} "
                        f"new_files={seed_files:,} unique_total={len(candidates):,}",
                        flush=True,
                    )
                if len(candidates) >= args.max_total_candidates:
                    break
            stats["category_seeds_completed"] += 1
            print(
                f"[harvest] completed {root!r}: categories={len(visited):,}, "
                f"new_files={seed_files:,}, unique_total={len(candidates):,}",
                flush=True,
            )
        if len(candidates) >= args.max_total_candidates:
            break

    rows = sorted(candidates.values(), key=lambda row: int(row["source_pageid"]))
    for row in rows:
        row["collection_buckets"].sort()
        row["collection_seeds"].sort()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "candidates.jsonl"
    atomic_write_jsonl(candidates_path, rows)
    bucket_counts = {
        str(bucket["name"]): sum(
            str(bucket["name"]) in row["collection_buckets"] for row in rows
        )
        for bucket in plan["buckets"]
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "stage": "metadata_harvest",
        "source": "wikimedia_commons",
        "unique_candidates": len(rows),
        "unique_candidate_target": min(
            int(plan["candidate_target"]), args.max_total_candidates
        ),
        "target_reached": len(rows)
        == min(int(plan["candidate_target"]), args.max_total_candidates),
        "bucket_candidate_counts": bucket_counts,
        "missing_or_empty_roots": sorted(set(missing_roots)),
        "stats": dict(stats),
        "plan": str(args.plan),
        "output": str(candidates_path),
    }
    atomic_write_json(args.output_dir / "harvest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not summary["target_reached"]:
        raise RuntimeError(
            f"unique candidate target not reached: {len(rows):,}/"
            f"{summary['unique_candidate_target']:,}; add more collection roots"
        )
    return 0


class HammingMultiIndex:
    def __init__(self, threshold: int) -> None:
        chunks = threshold + 1
        base, remainder = divmod(64, chunks)
        widths = [base + (index < remainder) for index in range(chunks)]
        offset = 64
        self.parts: list[tuple[int, int]] = []
        for width in widths:
            offset -= width
            self.parts.append((offset, (1 << width) - 1))
        self.buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
        self.hashes: dict[str, str] = {}

    def add(self, uid: str, value: str) -> None:
        self.hashes[uid] = value
        integer = int(value, 16)
        for index, (offset, mask) in enumerate(self.parts):
            self.buckets[(index, (integer >> offset) & mask)].append(uid)

    def candidates(self, value: str) -> set[str]:
        integer = int(value, 16)
        result: set[str] = set()
        for index, (offset, mask) in enumerate(self.parts):
            result.update(self.buckets.get((index, (integer >> offset) & mask), ()))
        return result


class KnownImageIndex:
    def __init__(self, *, near_threshold: int, review_threshold: int) -> None:
        self.near_threshold = near_threshold
        self.review_threshold = review_threshold
        self.file_hashes: dict[str, str] = {}
        self.pixel_hashes: dict[str, str] = {}
        self.phash = HammingMultiIndex(review_threshold)
        self.dhash = HammingMultiIndex(review_threshold)
        self.paths: dict[str, str] = {}
        self.count = 0

    def add(self, uid: str, fp: Mapping[str, Any], path: str) -> None:
        if fp.get("error") or not fp.get("phash") or not fp.get("dhash"):
            return
        key = uid
        if key in self.phash.hashes:
            key = f"{uid}:{self.count}"
        self.file_hashes[str(fp["file_sha256"])] = key
        self.pixel_hashes[str(fp["pixel_sha256"])] = key
        self.phash.add(key, str(fp["phash"]))
        self.dhash.add(key, str(fp["dhash"]))
        self.paths[key] = path
        self.count += 1

    def match(self, fp: Mapping[str, Any]) -> dict[str, Any]:
        for field, lookup in (
            ("file_sha256", self.file_hashes),
            ("pixel_sha256", self.pixel_hashes),
        ):
            uid = lookup.get(str(fp[field]))
            if uid:
                return {
                    "status": "excluded_exact_overlap",
                    "match": {"uid": uid, "path": self.paths.get(uid), "kind": field},
                }
        candidate_uids = self.phash.candidates(str(fp["phash"])) & self.dhash.candidates(
            str(fp["dhash"])
        )
        matches = []
        for uid in candidate_uids:
            pd = hamming(str(fp["phash"]), self.phash.hashes[uid])
            dd = hamming(str(fp["dhash"]), self.dhash.hashes[uid])
            if pd <= self.review_threshold and dd <= self.review_threshold:
                matches.append(
                    {
                        "uid": uid,
                        "path": self.paths.get(uid),
                        "phash_distance": pd,
                        "dhash_distance": dd,
                    }
                )
        matches.sort(
            key=lambda row: (
                max(row["phash_distance"], row["dhash_distance"]),
                row["phash_distance"] + row["dhash_distance"],
            )
        )
        strong = [
            row
            for row in matches
            if min(row["phash_distance"], row["dhash_distance"])
            <= self.near_threshold
        ]
        if strong:
            return {"status": "excluded_near_overlap", "matches": strong[:5]}
        if matches:
            return {"status": "review_perceptual_overlap", "matches": matches[:5]}
        return {"status": "accepted", "matches": []}


def load_known_index(args: argparse.Namespace, manifest_path: Path) -> KnownImageIndex:
    index = KnownImageIndex(
        near_threshold=args.near_threshold,
        review_threshold=args.review_threshold,
    )
    print("[index] loading frozen v3 fingerprints", flush=True)
    for row in read_jsonl(args.v3_index):
        index.add(str(row.get("uid", "v3")), row, str(row.get("path", "")))
    print(f"[index] loaded {index.count:,} v3 images", flush=True)
    if args.targeted_fingerprints.is_file():
        for number, row in enumerate(read_jsonl(args.targeted_fingerprints), 1):
            fp = row.get("fingerprints") or {}
            index.add(
                f"targeted:{number}", fp, str(row.get("path", ""))
            )
        print(f"[index] loaded through {index.count:,} known images", flush=True)
    for row in read_jsonl(manifest_path):
        path = Path(str(row["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"accepted manifest image is missing: {path}")
        index.add(str(row["uid"]), row["fingerprints"], str(path))
    print(f"[index] total known images={index.count:,}", flush=True)
    return index


def parse_image_page(page: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    infos = page.get("imageinfo") or []
    if page.get("missing") or not infos:
        return {"status": "excluded_api_missing"}
    info = infos[0]
    mime = str(info.get("mime", ""))
    if mime not in ALLOWED_MIME:
        return {"status": "excluded_mime", "mime": mime}
    width, height = int(info.get("width", 0)), int(info.get("height", 0))
    if min(width, height) < 448:
        return {
            "status": "excluded_source_dimensions",
            "source_width": width,
            "source_height": height,
        }
    license_name = extmetadata_value(info, "LicenseShortName")
    if not ALLOWED_LICENSE.match(license_name):
        return {"status": "excluded_license", "license": license_name}
    license_url = extmetadata_value(info, "LicenseUrl")
    if license_name.upper().startswith("CC ") and not license_url:
        return {
            "status": "excluded_license_metadata",
            "license": license_name,
            "reason": "Creative Commons record has no license URL",
        }
    restrictions = extmetadata_value(info, "Restrictions")
    if restrictions and restrictions.casefold() not in {"none", "no known restrictions"}:
        return {
            "status": "excluded_rights_restrictions",
            "license": license_name,
            "restrictions": restrictions,
        }
    return {
        "status": "download_ready",
        "source_pageid": int(page["pageid"]),
        "title": str(page.get("title") or candidate.get("title", "")),
        "source_url": str(info.get("descriptionurl", "")),
        "download_url": str(info.get("thumburl") or info.get("url") or ""),
        "original_url": str(info.get("url", "")),
        "source_mime": mime,
        "source_width": width,
        "source_height": height,
        "source_sha1": str(info.get("sha1", "")),
        "source_timestamp": str(info.get("timestamp", "")),
        "source_lastrevid": page.get("lastrevid"),
        "license": license_name,
        "license_url": license_url,
        "usage_terms": extmetadata_value(info, "UsageTerms"),
        "attribution_required": extmetadata_value(info, "AttributionRequired"),
        "copyrighted": extmetadata_value(info, "Copyrighted"),
        "rights_restrictions": restrictions,
        "artist": extmetadata_value(info, "Artist"),
        "credit": extmetadata_value(info, "Credit"),
        "attribution": extmetadata_value(info, "Attribution"),
        "source_categories": extmetadata_value(info, "Categories"),
        "source_description": extmetadata_value(info, "ImageDescription"),
        "source_datetime": extmetadata_value(info, "DateTimeOriginal")
        or extmetadata_value(info, "DateTime"),
    }


def download_one(
    ready: Mapping[str, Any], candidate: Mapping[str, Any], temporary_dir: Path
) -> dict[str, Any]:
    uid = str(candidate["source_uid"])
    extension = ".png" if ready["source_mime"] == "image/png" else ".jpg"
    temporary = temporary_dir / f"{uid}{extension}.part"
    try:
        response = requests.get(
            str(ready["download_url"]),
            headers={"User-Agent": USER_AGENT},
            timeout=(15, 90),
            stream=True,
        )
        response.raise_for_status()
        digest = hashlib.sha256()
        size = 0
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if not chunk:
                    continue
                size += len(chunk)
                if size > 40 * 1024 * 1024:
                    raise ValueError("download exceeded 40 MiB safety limit")
                digest.update(chunk)
                handle.write(chunk)
        with Image.open(temporary) as image:
            image.load()
            width, height = image.size
            entropy = float(image.convert("L").entropy())
            extrema = ImageStat.Stat(image.convert("L")).extrema[0]
        fp = image_fingerprints(temporary)
        if digest.hexdigest() != fp["file_sha256"]:
            raise ValueError("stream and file SHA-256 disagree")
        return {
            "status": "downloaded",
            "temporary_path": str(temporary),
            "extension": extension,
            "download_bytes": size,
            "decoded_width": width,
            "decoded_height": height,
            "entropy": entropy,
            "grayscale_extrema": list(extrema),
            "fingerprints": fp,
        }
    except (requests.RequestException, OSError, ValueError, UnidentifiedImageError) as exc:
        temporary.unlink(missing_ok=True)
        return {"status": "download_error", "error": f"{type(exc).__name__}: {exc}"}


def choose_bucket(
    candidate: Mapping[str, Any],
    accepted_counts: Mapping[str, int],
    bucket_targets: Mapping[str, int],
) -> str | None:
    available = [
        bucket
        for bucket in candidate["collection_buckets"]
        if accepted_counts.get(bucket, 0) < bucket_targets[bucket]
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda bucket: accepted_counts.get(bucket, 0) / bucket_targets[bucket],
    )


def choose_weighted_bucket(
    candidate: Mapping[str, Any],
    accepted_counts: Mapping[str, int],
    bucket_weights: Mapping[str, int],
) -> str | None:
    available = [
        bucket
        for bucket in candidate["collection_buckets"]
        if bucket in bucket_weights
    ]
    if not available:
        return None
    return min(
        available,
        key=lambda bucket: accepted_counts.get(bucket, 0) / bucket_weights[bucket],
    )


@dataclass
class CandidateScheduler:
    candidates: list[dict[str, Any]]
    bucket_targets: dict[str, int]
    completed: set[str]

    def __post_init__(self) -> None:
        all_rows = [
            row
            for row in self.candidates
            if str(row["source_uid"]) not in self.completed
        ]
        all_rows.sort(
            key=lambda row: hashlib.sha256(
                f"jmoshivis-v1:all:{row['source_uid']}".encode()
            ).digest()
        )
        self.all_candidates: deque[dict[str, Any]] = deque(all_rows)
        self.by_bucket: dict[str, deque[dict[str, Any]]] = {}
        for bucket in self.bucket_targets:
            rows = [
                row
                for row in self.candidates
                if bucket in row["collection_buckets"]
                and str(row["source_uid"]) not in self.completed
            ]
            rows.sort(
                key=lambda row: hashlib.sha256(
                    f"jmoshivis-v1:{bucket}:{row['source_uid']}".encode()
                ).digest()
            )
            self.by_bucket[bucket] = deque(rows)
        self.scheduled: set[str] = set()

    def take_all_batch(self, batch_size: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        while self.all_candidates and len(result) < batch_size:
            row = self.all_candidates.popleft()
            uid = str(row["source_uid"])
            if uid in self.scheduled or uid in self.completed:
                continue
            self.scheduled.add(uid)
            result.append(row)
        return result

    def retry_all(self, row: dict[str, Any]) -> None:
        uid = str(row["source_uid"])
        self.scheduled.discard(uid)
        self.all_candidates.append(row)

    def take_batch(
        self, accepted_counts: Mapping[str, int], batch_size: int
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        pending_counts = Counter(accepted_counts)
        while len(result) < batch_size:
            choices = [
                bucket
                for bucket, target in self.bucket_targets.items()
                if pending_counts[bucket] < target and self.by_bucket[bucket]
            ]
            if not choices:
                break
            bucket = min(
                choices,
                key=lambda name: pending_counts[name] / self.bucket_targets[name],
            )
            row = None
            while self.by_bucket[bucket]:
                possible = self.by_bucket[bucket].popleft()
                uid = str(possible["source_uid"])
                if uid not in self.scheduled and uid not in self.completed:
                    row = possible
                    break
            if row is None:
                continue
            self.scheduled.add(str(row["source_uid"]))
            pending_counts[bucket] += 1
            result.append(row)
        return result


def current_status(
    *,
    plan: Mapping[str, Any],
    candidates_path: Path,
    manifest_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    candidate_count = sum(1 for _ in read_jsonl(candidates_path))
    accepted_rows = list(read_jsonl(manifest_path))
    audit_rows = list(read_jsonl(audit_path))
    audit_counts = Counter(str(row.get("status")) for row in audit_rows)
    terminal_uids = {
        str(row.get("source_uid"))
        for row in audit_rows
        if str(row.get("status")) in TERMINAL_STATUSES
    }
    bucket_counts = Counter(str(row.get("assigned_bucket")) for row in accepted_rows)
    accept_all = bool(plan.get("accept_all_eligible"))
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "download_and_deduplicate",
        "acceptance_policy": "all_eligible_candidates" if accept_all else "fixed_target",
        "accepted_target": None if accept_all else int(plan["accepted_target"]),
        "maximum_candidate_count": candidate_count,
        "accepted": len(accepted_rows),
        "terminally_reviewed_candidates": len(terminal_uids),
        "remaining_candidates_to_review": max(0, candidate_count - len(terminal_uids)),
        "bucket_targets": {
            str(row["name"]): int(row["accepted_target"]) for row in plan["buckets"]
        },
        "bucket_accepted": dict(sorted(bucket_counts.items())),
        "candidate_count": candidate_count,
        "audit_status_counts": dict(sorted(audit_counts.items())),
        "paths": {
            "candidates": str(candidates_path),
            "accepted_manifest": str(manifest_path),
            "audit": str(audit_path),
        },
    }


def download(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    accept_all = bool(plan.get("accept_all_eligible"))
    candidates_path = args.output_dir / "candidates.jsonl"
    manifest_path = args.output_dir / "accepted_images.jsonl"
    audit_path = args.output_dir / "download_audit.jsonl"
    progress_path = args.output_dir / "progress.json"
    attribution_path = args.output_dir / "ATTRIBUTION.csv"
    image_root = args.output_dir / "images"
    temporary_dir = args.output_dir / ".tmp"
    image_root.mkdir(parents=True, exist_ok=True)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(read_jsonl(candidates_path))
    if not candidates:
        raise ValueError(f"no candidates in {candidates_path}; run harvest first")

    terminal: set[str] = set()
    attempts: Counter[str] = Counter()
    audit_counts: Counter[str] = Counter()
    for row in read_jsonl(audit_path):
        uid = str(row.get("source_uid"))
        event_status = str(row.get("status"))
        attempts[uid] += 1
        audit_counts[event_status] += 1
        if event_status in TERMINAL_STATUSES:
            terminal.add(uid)
    accepted_counts: Counter[str] = Counter()
    accepted_total = 0
    for row in read_jsonl(manifest_path):
        accepted_total += 1
        accepted_counts[str(row["assigned_bucket"])] += 1
        terminal.add(str(row["source_uid"]))
    targets = {
        str(bucket["name"]): int(bucket["accepted_target"])
        for bucket in plan["buckets"]
    }
    scheduler = CandidateScheduler(candidates, targets, terminal)
    index = load_known_index(args, manifest_path)
    client = CommonsClient(sleep_seconds=args.api_sleep)
    status_counts: Counter[str] = Counter()

    def write_progress() -> dict[str, Any]:
        progress = {
            "schema_version": SCHEMA_VERSION,
            "stage": "download_and_deduplicate",
            "acceptance_policy": (
                "all_eligible_candidates" if accept_all else "fixed_target"
            ),
            "accepted_target": None if accept_all else int(plan["accepted_target"]),
            "maximum_candidate_count": len(candidates),
            "accepted": accepted_total,
            "terminally_reviewed_candidates": len(terminal),
            "remaining_candidates_to_review": max(
                0, len(candidates) - len(terminal)
            ),
            "bucket_targets": targets,
            "bucket_accepted": dict(sorted(accepted_counts.items())),
            "candidate_count": len(candidates),
            "audit_status_counts": dict(sorted(audit_counts.items())),
            "current_run_status_counts": dict(sorted(status_counts.items())),
            "fingerprint_provenance": fingerprint_provenance(),
            "paths": {
                "candidates": str(candidates_path),
                "accepted_manifest": str(manifest_path),
                "audit": str(audit_path),
                "attribution": str(attribution_path),
            },
        }
        atomic_write_json(progress_path, progress)
        return progress

    manifest_handle = manifest_path.open("a", encoding="utf-8", buffering=1)
    audit_handle = audit_path.open("a", encoding="utf-8", buffering=1)
    try:
        while accept_all or accepted_total < int(plan["accepted_target"]):
            batch = (
                scheduler.take_all_batch(args.batch_size)
                if accept_all
                else scheduler.take_batch(accepted_counts, args.batch_size)
            )
            if not batch:
                print("[download] all candidate queues exhausted", flush=True)
                break
            pages = client.image_info(
                [int(row["source_pageid"]) for row in batch],
                int(plan["thumbnail_width"]),
            )
            pages_by_id = {int(page["pageid"]): page for page in pages}
            ready_jobs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for candidate in batch:
                uid = str(candidate["source_uid"])
                page = pages_by_id.get(int(candidate["source_pageid"]), {"missing": True})
                parsed = parse_image_page(page, candidate)
                if parsed["status"] == "download_ready":
                    ready_jobs.append((parsed, candidate))
                    continue
                event = {
                    "schema_version": SCHEMA_VERSION,
                    "source_uid": uid,
                    "source_pageid": candidate["source_pageid"],
                    "title": candidate["title"],
                    "collection_buckets": candidate["collection_buckets"],
                    "attempt": attempts[uid] + 1,
                    **parsed,
                }
                attempts[uid] += 1
                audit_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                terminal.add(uid)
                status_counts[str(parsed["status"])] += 1
                audit_counts[str(parsed["status"])] += 1

            with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
                downloaded = list(
                    executor.map(
                        lambda job: download_one(job[0], job[1], temporary_dir),
                        ready_jobs,
                    )
                )
            for result_index, ((ready, candidate), result) in enumerate(
                zip(ready_jobs, downloaded)
            ):
                uid = str(candidate["source_uid"])
                attempts[uid] += 1
                base_event = {
                    "schema_version": SCHEMA_VERSION,
                    "source_uid": uid,
                    "source_pageid": candidate["source_pageid"],
                    "title": candidate["title"],
                    "collection_buckets": candidate["collection_buckets"],
                    "attempt": attempts[uid],
                }
                if result["status"] != "downloaded":
                    failure_status = (
                        "download_error"
                        if attempts[uid] < args.max_download_attempts
                        else "excluded_download_failure"
                    )
                    event = {**base_event, **result, "status": failure_status}
                    audit_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    status_counts[failure_status] += 1
                    audit_counts[failure_status] += 1
                    if failure_status == "download_error" and accept_all:
                        scheduler.retry_all(candidate)
                    elif failure_status == "excluded_download_failure":
                        terminal.add(uid)
                    continue
                temporary = Path(str(result["temporary_path"]))
                decoded_width = int(result["decoded_width"])
                decoded_height = int(result["decoded_height"])
                aspect = max(
                    decoded_width / decoded_height, decoded_height / decoded_width
                )
                terminal_result: dict[str, Any] | None = None
                if min(decoded_width, decoded_height) < args.min_short_side:
                    terminal_result = {"status": "excluded_decoded_dimensions"}
                elif aspect > args.max_aspect_ratio:
                    terminal_result = {
                        "status": "excluded_aspect_ratio",
                        "aspect_ratio": aspect,
                    }
                elif float(result["entropy"]) < args.min_entropy:
                    terminal_result = {
                        "status": "excluded_low_information",
                        "entropy": result["entropy"],
                    }
                else:
                    terminal_result = index.match(result["fingerprints"])

                assigned_bucket = (
                    choose_weighted_bucket(candidate, accepted_counts, targets)
                    if accept_all
                    else choose_bucket(candidate, accepted_counts, targets)
                )
                if terminal_result["status"] == "accepted" and assigned_bucket is None:
                    temporary.unlink(missing_ok=True)
                    continue
                if terminal_result["status"] != "accepted":
                    temporary.unlink(missing_ok=True)
                    event = {
                        **base_event,
                        **terminal_result,
                        "source": ready,
                        "download": {
                            key: value
                            for key, value in result.items()
                            if key not in {"temporary_path", "fingerprints", "status"}
                        },
                        "fingerprints": result["fingerprints"],
                    }
                    audit_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    terminal.add(uid)
                    status_counts[str(terminal_result["status"])] += 1
                    audit_counts[str(terminal_result["status"])] += 1
                    continue

                final_dir = image_root / uid[:2]
                final_dir.mkdir(parents=True, exist_ok=True)
                final_path = final_dir / f"{uid}{result['extension']}"
                os.replace(temporary, final_path)
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "uid": uid,
                    "source_uid": uid,
                    "path": str(final_path.resolve()),
                    "source": "wikimedia_commons",
                    "assigned_bucket": assigned_bucket,
                    "collection_buckets": candidate["collection_buckets"],
                    "collection_seeds": candidate["collection_seeds"],
                    "discoveries": candidate["discoveries"],
                    "task_routing": "vlm_adaptive",
                    "has_canonical_count_annotation": False,
                    "has_canonical_ocr_annotation": False,
                    "requires_review_if_self_grounded": True,
                    **ready,
                    "download": {
                        key: value
                        for key, value in result.items()
                        if key not in {"temporary_path", "fingerprints", "status"}
                    },
                    "fingerprints": result["fingerprints"],
                    "collected_at_unix": int(time.time()),
                }
                manifest_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                audit_handle.write(
                    json.dumps({**base_event, "status": "accepted"}, ensure_ascii=False)
                    + "\n"
                )
                index.add(uid, result["fingerprints"], str(final_path))
                terminal.add(uid)
                accepted_counts[assigned_bucket] += 1
                accepted_total += 1
                status_counts["accepted"] += 1
                audit_counts["accepted"] += 1
                if not accept_all and accepted_total >= int(plan["accepted_target"]):
                    for leftover in downloaded[result_index + 1 :]:
                        if leftover.get("status") == "downloaded":
                            Path(str(leftover["temporary_path"])).unlink(missing_ok=True)
                    break

            write_progress()
            if accept_all:
                print(
                    f"[download] reviewed={len(terminal):,}/{len(candidates):,} "
                    f"accepted={accepted_total:,}",
                    flush=True,
                )
            else:
                print(
                    f"[download] accepted={accepted_total:,}/{plan['accepted_target']:,} "
                    + " ".join(
                        f"{name}={accepted_counts[name]:,}/{target:,}"
                        for name, target in targets.items()
                    ),
                    flush=True,
                )
    finally:
        manifest_handle.close()
        audit_handle.close()
    progress = write_progress()
    write_attribution_csv(manifest_path, attribution_path)
    print(json.dumps(progress, ensure_ascii=False, indent=2), flush=True)
    if accept_all:
        return 0 if len(terminal) >= len(candidates) else 2
    return 0 if accepted_total >= int(plan["accepted_target"]) else 2


def status(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    value = current_status(
        plan=plan,
        candidates_path=args.output_dir / "candidates.jsonl",
        manifest_path=args.output_dir / "accepted_images.jsonl",
        audit_path=args.output_dir / "download_audit.jsonl",
    )
    print(json.dumps(value, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=["harvest", "download", "status"]
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO_ROOT / "ssvd/images/japanese_daily_v1_plan.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/images/japanese_daily_v1",
    )
    parser.add_argument("--api-sleep", type=float, default=0.12)
    parser.add_argument("--max-total-candidates", type=int, default=125_000)
    parser.add_argument("--max-per-seed", type=int, default=8_000)
    parser.add_argument("--max-categories-per-seed", type=int, default=3_000)
    parser.add_argument("--max-geosearch-candidates", type=int, default=15_000)
    parser.add_argument("--max-geosearch-per-location", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--max-download-attempts", type=int, default=3)
    parser.add_argument("--min-short-side", type=int, default=448)
    parser.add_argument("--max-aspect-ratio", type=float, default=3.5)
    parser.add_argument("--min-entropy", type=float, default=2.5)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--review-threshold", type=int, default=8)
    parser.add_argument(
        "--v3-index",
        type=Path,
        default=REPO_ROOT / "data/evaluation/v3_image_index.jsonl",
    )
    parser.add_argument(
        "--targeted-fingerprints",
        type=Path,
        default=REPO_ROOT
        / "data/images/targeted_v4/image_pool_v1/fingerprints.jsonl",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "harvest":
        return harvest(args)
    if args.command == "download":
        return download(args)
    return status(args)


if __name__ == "__main__":
    raise SystemExit(main())
