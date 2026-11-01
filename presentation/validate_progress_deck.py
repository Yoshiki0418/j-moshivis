#!/usr/bin/env python3
"""Read-only validation for the revised J-MoshiVis progress deck.

The checks deliberately inspect the OOXML package directly.  This keeps the
validator independent from the deck builder and avoids changing the files it
is meant to verify.
"""

from __future__ import annotations

import argparse
import hashlib
import posixpath
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "2026_0423_経過報告.pptx"
DEFAULT_REVISED_PPTX = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_改訂版.pptx"
DEFAULT_REVISED_PDF = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_改訂版.pdf"
DEFAULT_INSERT_PDF = ROOT / "J-MoshiVis_改修進捗_追加スライド_改訂版.pdf"

SOURCE_SLIDES = 36
INSERT_AFTER = 25
INSERTED_SLIDES = 16
REVISED_SLIDES = SOURCE_SLIDES + INSERTED_SLIDES
EXPECTED_SLIDE_SIZE = (12_192_000, 6_858_000)  # 13.333 x 7.5 in, 16:9

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pr": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass
class Results:
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    def check(self, condition: bool, success: str, failure: str) -> None:
        (self.passed if condition else self.failed).append(success if condition else failure)

    def require(self, condition: bool, message: str) -> None:
        self.check(condition, message, message)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_xml(package: zipfile.ZipFile, part_name: str) -> ET.Element:
    try:
        return ET.fromstring(package.read(part_name))
    except KeyError as exc:
        raise ValueError(f"OOXML part is missing: {part_name}") from exc
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML in {part_name}: {exc}") from exc


def _resolve_part(source_part: str, target: str) -> str:
    """Resolve an OOXML relationship target to a package-relative part name."""
    if target.startswith("/"):
        return target.lstrip("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(source_part), target))


def relationship_map(
    package: zipfile.ZipFile, source_part: str, rels_part: str
) -> dict[str, tuple[str, str]]:
    root = _read_xml(package, rels_part)
    relationships: dict[str, tuple[str, str]] = {}
    for rel in root.findall("pr:Relationship", NS):
        rel_id = rel.attrib.get("Id")
        target = rel.attrib.get("Target")
        rel_type = rel.attrib.get("Type", "")
        if rel_id and target and rel.attrib.get("TargetMode") != "External":
            relationships[rel_id] = (_resolve_part(source_part, target), rel_type)
    return relationships


def slide_order(package: zipfile.ZipFile) -> list[str]:
    presentation = _read_xml(package, "ppt/presentation.xml")
    rels = relationship_map(
        package, "ppt/presentation.xml", "ppt/_rels/presentation.xml.rels"
    )
    order: list[str] = []
    for slide_id in presentation.findall("./p:sldIdLst/p:sldId", NS):
        rel_id = slide_id.attrib.get(f"{{{NS['r']}}}id")
        if rel_id not in rels:
            raise ValueError(f"Slide relationship is missing: {rel_id}")
        part_name, rel_type = rels[rel_id]
        if not rel_type.endswith("/slide"):
            raise ValueError(f"Relationship {rel_id} does not target a slide")
        order.append(part_name)
    return order


def slide_size(package: zipfile.ZipFile) -> tuple[int, int]:
    presentation = _read_xml(package, "ppt/presentation.xml")
    size = presentation.find("p:sldSz", NS)
    if size is None:
        raise ValueError("ppt/presentation.xml has no p:sldSz element")
    return int(size.attrib["cx"]), int(size.attrib["cy"])


def slide_statistics(package: zipfile.ZipFile, slide_part: str) -> tuple[int, int, int]:
    """Return (native-shape count, non-empty text chars, picture count)."""
    root = _read_xml(package, slide_part)
    native_shapes = sum(
        len(root.findall(f".//p:{tag}", NS))
        for tag in ("sp", "cxnSp", "graphicFrame", "grpSp")
    )
    text_chars = sum(
        len(node.text.strip())
        for node in root.findall(".//a:t", NS)
        if node.text and node.text.strip()
    )
    pictures = len(root.findall(".//p:pic", NS))
    return native_shapes, text_chars, pictures


def notes_text(package: zipfile.ZipFile, slide_part: str) -> str | None:
    slide_name = posixpath.basename(slide_part)
    rels_part = posixpath.join(
        posixpath.dirname(slide_part), "_rels", f"{slide_name}.rels"
    )
    if rels_part not in package.namelist():
        return None
    rels = relationship_map(package, slide_part, rels_part)
    notes_parts = [part for part, kind in rels.values() if kind.endswith("/notesSlide")]
    if len(notes_parts) != 1:
        return None
    notes = _read_xml(package, notes_parts[0])
    chunks = [
        node.text.strip()
        for node in notes.findall(".//a:t", NS)
        if node.text and node.text.strip()
    ]
    return "\n".join(chunks).strip()


def media_hashes(package: zipfile.ZipFile) -> dict[str, str]:
    return {
        name: sha256(package.read(name))
        for name in package.namelist()
        if name.startswith("ppt/media/") and not name.endswith("/")
    }


def _format_names(names: Iterable[str], limit: int = 5) -> str:
    items = sorted(names)
    shown = ", ".join(items[:limit])
    return shown if len(items) <= limit else f"{shown}, ... (+{len(items) - limit})"


def validate_pptx(source: Path, revised: Path, results: Results) -> None:
    with zipfile.ZipFile(source) as source_zip, zipfile.ZipFile(revised) as revised_zip:
        source_order = slide_order(source_zip)
        revised_order = slide_order(revised_zip)

        results.check(
            len(source_order) == SOURCE_SLIDES,
            f"元PPTX: {SOURCE_SLIDES}スライド",
            f"元PPTX: {len(source_order)}スライド（期待値 {SOURCE_SLIDES}）",
        )
        results.check(
            len(revised_order) == REVISED_SLIDES,
            f"改訂PPTX: {REVISED_SLIDES}スライド",
            f"改訂PPTX: {len(revised_order)}スライド（期待値 {REVISED_SLIDES}）",
        )

        source_dimensions = slide_size(source_zip)
        revised_dimensions = slide_size(revised_zip)
        results.check(
            source_dimensions == EXPECTED_SLIDE_SIZE,
            f"元PPTXの寸法: {source_dimensions[0]} x {source_dimensions[1]} EMU",
            f"元PPTXの寸法: {source_dimensions}（期待値 {EXPECTED_SLIDE_SIZE}）",
        )
        results.check(
            revised_dimensions == EXPECTED_SLIDE_SIZE,
            f"改訂PPTXの寸法: {revised_dimensions[0]} x {revised_dimensions[1]} EMU",
            f"改訂PPTXの寸法: {revised_dimensions}（期待値 {EXPECTED_SLIDE_SIZE}）",
        )

        if len(source_order) >= SOURCE_SLIDES and len(revised_order) >= REVISED_SLIDES:
            prefix_ok = revised_order[:INSERT_AFTER] == source_order[:INSERT_AFTER]
            suffix_start = INSERT_AFTER + INSERTED_SLIDES
            suffix_ok = revised_order[suffix_start:] == source_order[INSERT_AFTER:]
            inserted = revised_order[INSERT_AFTER:suffix_start]
            inserted_are_new = len(inserted) == INSERTED_SLIDES and not (
                set(inserted) & set(source_order)
            )
            results.check(
                prefix_ok and suffix_ok and inserted_are_new,
                "追加16枚は元スライド25の直後（表示順26〜41）",
                "追加スライドの挿入位置または元スライドの表示順が不正",
            )

            shape_failures: list[str] = []
            note_failures: list[str] = []
            minimum_shapes: int | None = None
            minimum_text: int | None = None
            for display_index, slide_part in enumerate(inserted, start=INSERT_AFTER + 1):
                shapes, text_chars, _pictures = slide_statistics(revised_zip, slide_part)
                minimum_shapes = shapes if minimum_shapes is None else min(minimum_shapes, shapes)
                minimum_text = text_chars if minimum_text is None else min(minimum_text, text_chars)
                if shapes < 1 or text_chars < 1:
                    shape_failures.append(
                        f"表示順{display_index}({slide_part}: shapes={shapes}, text={text_chars})"
                    )
                note = notes_text(revised_zip, slide_part)
                if not note:
                    note_failures.append(f"表示順{display_index}({slide_part})")

            results.check(
                not shape_failures,
                f"追加16枚にネイティブ図形・テキストあり（最小 shapes={minimum_shapes}, text={minimum_text}文字）",
                "ネイティブ図形・テキスト不足: " + _format_names(shape_failures),
            )
            results.check(
                not note_failures,
                "追加16枚すべてに発表者ノートあり",
                "発表者ノートなし: " + _format_names(note_failures),
            )

        source_media = media_hashes(source_zip)
        revised_media = media_hashes(revised_zip)
        missing_media = set(source_media) - set(revised_media)
        changed_media = {
            name
            for name, digest in source_media.items()
            if name in revised_media and revised_media[name] != digest
        }
        results.check(
            not missing_media and not changed_media,
            f"元のppt/media全{len(source_media)}件を同名・同一SHA-256で保持",
            "元メディアの欠落/変更: "
            + _format_names(missing_media | changed_media),
        )


def pdf_info(path: Path) -> tuple[int, list[tuple[float, float]]]:
    try:
        import fitz  # PyMuPDF; imported lazily so --help remains dependency-free
    except ImportError as exc:
        raise RuntimeError("PDF検証には PyMuPDF (fitz) が必要です") from exc

    with fitz.open(path) as document:
        sizes = [(float(page.rect.width), float(page.rect.height)) for page in document]
        return document.page_count, sizes


def validate_pdf(
    path: Path,
    expected_pages: int,
    label: str,
    results: Results,
) -> tuple[float, float] | None:
    page_count, sizes = pdf_info(path)
    results.check(
        page_count == expected_pages,
        f"{label}: {expected_pages}ページ",
        f"{label}: {page_count}ページ（期待値 {expected_pages}）",
    )
    if not sizes:
        results.failed.append(f"{label}: ページ寸法を取得できません")
        return None

    first = sizes[0]
    same_size = all(abs(width - first[0]) < 0.01 and abs(height - first[1]) < 0.01 for width, height in sizes)
    ratio_ok = first[1] > 0 and abs(first[0] / first[1] - 16 / 9) < 0.001
    results.check(
        same_size and ratio_ok,
        f"{label}の全ページ寸法: {first[0]:.2f} x {first[1]:.2f} pt（16:9）",
        f"{label}: ページ寸法が不統一または16:9ではありません",
    )
    return first


def validate(args: argparse.Namespace) -> Results:
    results = Results()
    paths = {
        "元PPTX": args.source,
        "改訂PPTX": args.revised_pptx,
        "改訂PDF": args.revised_pdf,
        "追加スライドPDF": args.insert_pdf,
    }
    missing = [f"{label}: {path}" for label, path in paths.items() if not path.is_file()]
    if missing:
        results.failed.append("必要ファイルがありません: " + "; ".join(missing))
        return results

    try:
        validate_pptx(args.source, args.revised_pptx, results)
        revised_pdf_size = validate_pdf(
            args.revised_pdf, REVISED_SLIDES, "改訂PDF", results
        )
        insert_pdf_size = validate_pdf(
            args.insert_pdf, INSERTED_SLIDES, "追加スライドPDF", results
        )
        if revised_pdf_size is not None and insert_pdf_size is not None:
            dimensions_match = all(
                abs(left - right) < 0.01
                for left, right in zip(revised_pdf_size, insert_pdf_size)
            )
            results.check(
                dimensions_match,
                "2つのPDFでページ寸法が一致",
                "改訂PDFと追加スライドPDFでページ寸法が不一致",
            )
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile) as exc:
        results.failed.append(f"検証処理エラー: {exc}")
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--revised-pptx", type=Path, default=DEFAULT_REVISED_PPTX)
    parser.add_argument("--revised-pdf", type=Path, default=DEFAULT_REVISED_PDF)
    parser.add_argument("--insert-pdf", type=Path, default=DEFAULT_INSERT_PDF)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = validate(args)
    for message in results.passed:
        print(f"PASS  {message}")
    for message in results.failed:
        print(f"FAIL  {message}", file=sys.stderr)
    print(
        f"RESULT: {len(results.passed)} passed, {len(results.failed)} failed",
        file=sys.stderr if results.failed else sys.stdout,
    )
    return 1 if results.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
