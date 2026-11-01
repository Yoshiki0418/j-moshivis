"""Dataset viewer entrypoint with live JSSODa OCR shard support.

This compatibility entrypoint leaves the established viewer API and UI intact
while registering the two append-only OCR dialogue shards produced by the full
JSSODa generation job.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

from aiohttp import web

from . import catalog as catalog_module


LOGGER = logging.getLogger("dataset_viewer")
PATCH_VERSION = "jssoda-live-v1"


def _install_jssoda_support() -> None:
    source_ids = {spec.source_id for spec in catalog_module._SOURCE_SPECS}
    additions = []
    if "ocr_v4.jssoda_shard0" not in source_ids:
        additions.append(
            catalog_module._SourceSpec(
                "ocr_v4.jssoda_shard0",
                "ocr_v4",
                "counting_dialogue",
                "data/dialogues_ocr/jssoda_v1/shard0.jsonl",
                20,
            )
        )
    if "ocr_v4.jssoda_shard1" not in source_ids:
        additions.append(
            catalog_module._SourceSpec(
                "ocr_v4.jssoda_shard1",
                "ocr_v4",
                "counting_dialogue",
                "data/dialogues_ocr/jssoda_v1/shard1.jsonl",
                21,
            )
        )
    if additions:
        catalog_module._SOURCE_SPECS = (*catalog_module._SOURCE_SPECS, *additions)

    original = catalog_module.DatasetCatalog._component_from_record
    if getattr(original, "_jssoda_live_patch", False):
        return

    def component_with_jssoda(
        self: catalog_module.DatasetCatalog,
        spec: catalog_module._SourceSpec,
        record: Mapping[str, Any],
        offset: int,
        length: int,
    ) -> tuple[Any, ...] | None:
        component = original(self, spec, record, offset, length)
        if component is None or spec.collection != "ocr_v4":
            return component
        values = list(component)
        if spec.role == "targeted_images":
            refs = record.get("reference_annotations") or []
            has_exact_transcription = any(
                isinstance(annotation, Mapping)
                and annotation.get("kind") in (None, "exact_transcription")
                and isinstance(annotation.get("text"), str)
                and bool(annotation["text"].strip())
                for annotation in refs
            )
            values[17] = int(has_exact_transcription)
        elif spec.role == "counting_dialogue":
            try:
                summary = json.loads(values[19])
            except (TypeError, json.JSONDecodeError):
                summary = {}
            grounding = record.get("grounding")
            if isinstance(grounding, Mapping):
                summary["ocr_target_count"] = len(grounding.get("targets") or [])
                summary["all_ocr_turns_grounded"] = grounding.get(
                    "all_ocr_turns_grounded"
                )
                summary["canonical_transcription_characters"] = grounding.get(
                    "canonical_transcription_characters"
                )
            values[19] = catalog_module._json_dumps(summary)
        return tuple(values)

    component_with_jssoda._jssoda_live_patch = True  # type: ignore[attr-defined]
    catalog_module.DatasetCatalog._component_from_record = component_with_jssoda


def _prepare_catalog(catalog: catalog_module.DatasetCatalog) -> None:
    """Invalidate only OCR image metadata once so long pages become eligible."""
    with catalog._connect() as database:
        marker = database.execute(
            "SELECT value FROM meta WHERE key='jssoda_live_patch'"
        ).fetchone()
        if marker is not None and marker[0] == PATCH_VERSION:
            return
        database.execute(
            "DELETE FROM sources WHERE source_id='ocr_v4.images'"
        )
        database.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('jssoda_live_patch',?)",
            (PATCH_VERSION,),
        )


def main(argv: Sequence[str] | None = None) -> None:
    _install_jssoda_support()
    from . import app as base_app

    args = base_app._parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if args.refresh_interval < 0:
        raise SystemExit("--refresh-interval must be zero or greater")

    root = Path(args.repo_root).resolve()
    database = Path(args.db or root / "data/.dataset_viewer/catalog.sqlite3")
    if not database.is_absolute():
        database = root / database
    catalog = catalog_module.DatasetCatalog(repo_root=root, db_path=database)
    _prepare_catalog(catalog)
    web.run_app(
        base_app.create_app(
            repo_root=root,
            db_path=database,
            catalog=catalog,
            rebuild=args.rebuild,
            refresh_interval=args.refresh_interval,
        ),
        host=args.host,
        port=args.port,
        print=lambda message: LOGGER.info("%s", message),
    )


if __name__ == "__main__":
    main()
