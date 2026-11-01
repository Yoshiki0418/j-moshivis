#!/usr/bin/env python3
"""Structural checks for the experiment-plan presentation deliverables."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import fitz
from pptx import Presentation

import build_experiment_plan_deck as deck


def media_hashes(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if name.startswith("ppt/media/"):
                hashes[Path(name).name] = hashlib.sha256(archive.read(name)).hexdigest()
    return hashes


def slide_text(slide) -> str:
    return "\n".join(
        shape.text for shape in slide.shapes
        if getattr(shape, "has_text_frame", False)
    )


def check(condition: bool, label: str):
    if not condition:
        raise AssertionError(label)
    print(f"PASS  {label}")


def main():
    for path in (deck.OUTPUT_PPTX, deck.OUTPUT_PDF, deck.ADDED_PDF):
        check(path.exists() and path.stat().st_size > 0, f"exists: {path.name}")

    prs = Presentation(deck.OUTPUT_PPTX)
    check(len(prs.slides) == 60, "PowerPoint has 60 slides")
    check((prs.slide_width, prs.slide_height) == (12192000, 6858000), "16:9 slide size preserved")

    expected = [
        "v3を用いた論文化実験計画",
        "MoshiVis本家に合わせた評価方針",
        "J-MoshiVisで答える4つのResearch Question",
        "日本語音声画像理解を、どう測るか",
        "共通基準から一要因ずつ変える — 固有8 run",
        "v3だけで、データ量25 / 50 / 100%を比較",
        "英語の視覚Adapterは、日本語へ転移するか",
        "実行順序と計算資源",
    ]
    for index, title in enumerate(expected, start=41):
        check(title in slide_text(prs.slides[index]), f"inserted slide {index + 1}: {title}")
        notes = prs.slides[index].notes_slide.notes_text_frame.text.strip()
        check(bool(notes), f"speaker notes present on slide {index + 1}")

    numbered_pages = list(range(50, 56)) + list(range(57, 60))
    for index, expected_number in zip(numbered_pages, numbered_pages):
        numbers = []
        for shape in prs.slides[index].shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            raw = shape.text.strip()
            if raw.isdigit() and shape.top > prs.slide_height * 0.90:
                numbers.append(int(raw))
        check(expected_number in numbers, f"footer number {expected_number} on slide {index + 1}")
    check("今後の展望" in slide_text(prs.slides[56]), "slide 57 is the original section divider")
    check(not any(
        shape.text.strip().isdigit()
        for shape in prs.slides[56].shapes
        if getattr(shape, "has_text_frame", False)
    ), "slide 57 section divider remains unnumbered")

    # Some shapes in the author's original deck intentionally bleed beyond the
    # canvas.  Bounds checks therefore apply only to the eight newly built pages.
    for index in range(41, 49):
        in_bounds = all(
            shape.left >= 0
            and shape.top >= 0
            and shape.left + shape.width <= prs.slide_width + 2
            and shape.top + shape.height <= prs.slide_height + 2
            for shape in prs.slides[index].shapes
        )
        check(in_bounds, f"all shapes in bounds on inserted slide {index + 1}")

    original_media = media_hashes(deck.base.SOURCE_PPTX)
    output_media = media_hashes(deck.OUTPUT_PPTX)
    check(set(original_media).issubset(output_media), "all original media retained")
    check(all(output_media[name] == digest for name, digest in original_media.items()),
          "all original media byte-identical")

    combined = fitz.open(deck.OUTPUT_PDF)
    added = fitz.open(deck.ADDED_PDF)
    check(len(combined) == 60, "combined PDF has 60 pages")
    check(len(added) == 8, "added-only PDF has 8 pages")
    for offset in range(8):
        left = combined[41 + offset].get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False).samples
        right = added[offset].get_pixmap(matrix=fitz.Matrix(0.5, 0.5), alpha=False).samples
        check(hashlib.sha256(left).digest() == hashlib.sha256(right).digest(),
              f"PDF experiment page {offset + 1} matches added-only PDF")
    combined.close()
    added.close()

    print("\nAll experiment-plan deck checks passed.")


if __name__ == "__main__":
    main()
