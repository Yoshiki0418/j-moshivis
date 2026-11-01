"""Final full-dataset runner normalizing overlong VLM output to five exchanges."""

from __future__ import annotations

from typing import Any

from ssvd.dialogue import generate_jssoda_ocr_dataset as production
from ssvd.dialogue import generate_jssoda_ocr_dataset_all as adaptive
from ssvd.dialogue import generate_jssoda_ocr_dataset_runner as runner
from ssvd.dialogue import generate_jssoda_ocr_pilot as pilot


ORIGINAL_PARSE_PAYLOAD = pilot.parse_payload


def normalized_parse_payload(raw: str) -> dict[str, Any] | None:
    payload = ORIGINAL_PARSE_PAYLOAD(raw)
    if payload is None:
        return None
    dialogue = payload.get("dialogue")
    if isinstance(dialogue, list) and len(dialogue) > 10:
        payload["dialogue"] = dialogue[:10]
    return payload


if __name__ == "__main__":
    pilot.parse_payload = normalized_parse_payload
    production.load_entries = adaptive.load_all_entries
    production.build_prompt = runner.fixed_adaptive_prompt
    production.validate = adaptive.adaptive_validate
    production.main()
