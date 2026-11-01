"""Dataset viewer with live JSSODa OCR and Japanese Daily v1 dialogues.

The daily dialogue source uses the catalogue's append-only image-bank refresh
path, while its records are interpreted as dialogues. This keeps 10-second
refreshes cheap even when the generation file grows beyond 100,000 rows.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from . import app_jssoda
from . import catalog as catalog_module


DAILY_SOURCE_ID = "japanese_daily_v1.dialogues"
DAILY_DIALOGUE_PATH = (
    "data/dialogues_japanese_daily_v1/full/qwen3vl_30b_normal_adaptive.jsonl"
)


def _install_daily_support() -> None:
    app_jssoda._install_jssoda_support()
    if DAILY_SOURCE_ID not in {
        spec.source_id for spec in catalog_module._SOURCE_SPECS
    }:
        catalog_module._SOURCE_SPECS = (
            *catalog_module._SOURCE_SPECS,
            catalog_module._SourceSpec(
                DAILY_SOURCE_ID,
                "japanese_daily_v1",
                "daily_image_bank",  # use the existing append-only refresh path
                DAILY_DIALOGUE_PATH,
                20,
            ),
        )

    original = catalog_module.DatasetCatalog._component_from_record
    if getattr(original, "_daily_live_patch", False):
        return

    def component_with_daily(
        self: catalog_module.DatasetCatalog,
        spec: catalog_module._SourceSpec,
        record: Mapping[str, Any],
        offset: int,
        length: int,
    ) -> tuple[Any, ...] | None:
        if spec.source_id != DAILY_SOURCE_ID:
            return original(self, spec, record, offset, length)

        dialogue_spec = catalog_module._SourceSpec(
            spec.source_id,
            spec.collection,
            "counting_dialogue",
            spec.relative_path,
            spec.priority,
        )
        component = original(self, dialogue_spec, record, offset, length)
        if component is None:
            return None
        values = list(component)
        values[3] = "daily_dialogue"
        # Retain the image bank's assigned_bucket as the browser category.
        values[8] = ""
        try:
            summary = json.loads(values[19])
        except (TypeError, json.JSONDecodeError):
            summary = {}
        summary["normal_strategy"] = record.get("normal_strategy")
        values[19] = catalog_module._json_dumps(summary)
        return tuple(values)

    component_with_daily._daily_live_patch = True  # type: ignore[attr-defined]
    # app_jssoda.main() calls its installer again; preserve its idempotence.
    component_with_daily._jssoda_live_patch = True  # type: ignore[attr-defined]
    catalog_module.DatasetCatalog._component_from_record = component_with_daily


def main(argv: Sequence[str] | None = None) -> None:
    _install_daily_support()
    app_jssoda.main(argv)


if __name__ == "__main__":
    main()
