"""Stable executable entrypoint for the adaptive full JSSODa generator."""

from __future__ import annotations

from typing import Any, Mapping

from ssvd.dialogue import generate_jssoda_ocr_dataset as production
from ssvd.dialogue import generate_jssoda_ocr_dataset_all as adaptive


ORIGINAL_BUILD_PROMPT = production.build_prompt


def fixed_adaptive_prompt(entry: Mapping[str, Any]) -> str:
    target_count = int(entry["_target_count"])
    prompt = ORIGINAL_BUILD_PROMPT(entry)
    if target_count == 4:
        return prompt
    general_count = 4 - target_count
    return prompt.replace(
        "- 残る4組はすべてOCR質問応答です。",
        f"- 残る4組のうち、{target_count}組をOCR質問応答にします。\n"
        f"- 残り{general_count}組は文書の概要に関する短い質問応答にします。\n"
        "- 概要質問では、段落・行・最初・最後などの位置指定や読み上げ依頼を使いません。",
    )


if __name__ == "__main__":
    production.load_entries = adaptive.load_all_entries
    production.build_prompt = fixed_adaptive_prompt
    production.validate = adaptive.adaptive_validate
    production.main()
