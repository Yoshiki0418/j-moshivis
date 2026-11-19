from __future__ import annotations

import unittest
import re
from html.parser import HTMLParser
from pathlib import Path


STATIC_ROOT = Path(__file__).resolve().parents[1] / "static"


class _IdCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(str(values["id"]))


class ReviewUiStaticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
        self.javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
        self.styles = (STATIC_ROOT / "styles.css").read_text(encoding="utf-8")

    def test_review_controls_exist_and_ids_are_unique(self) -> None:
        parser = _IdCollector()
        parser.feed(self.html)
        self.assertEqual(len(parser.ids), len(set(parser.ids)))
        for element_id in (
            "review-status-filter",
            "auxiliary-kind-filter",
            "review-required-filter",
            "review-section",
            "review-note",
            "review-approve",
            "review-reject",
            "review-reset",
            "review-edit",
            "dialogue-save",
            "dialogue-cancel",
            "metric-review-required",
            "metric-review-certified",
            "metric-review-regeneration",
            "metric-review-pending",
            "ai-suggestion",
            "ai-suggestion-label",
            "ai-suggestion-diff",
            "ai-status-filter",
        ):
            self.assertIn(element_id, parser.ids)

    def test_review_request_is_content_bound(self) -> None:
        self.assertIn('/review`', self.javascript)
        self.assertIn('method: "PUT"', self.javascript)
        self.assertIn("decision,", self.javascript)
        self.assertIn("content_sha256: review.content_sha256", self.javascript)
        self.assertIn("fingerprint: review.content_sha256", self.javascript)
        self.assertIn("note: dom.reviewNote.value.trim()", self.javascript)
        self.assertIn('/dialogue`', self.javascript)
        self.assertIn('if (state.editingDialogue) return;', self.javascript)

    def test_review_filters_and_status_styles_are_wired(self) -> None:
        for query_key in ("review_status", "auxiliary_kind", "review_required"):
            self.assertIn(query_key, self.javascript)
        for status in ("pending", "approved", "rejected"):
            self.assertIn(f".review-badge.is-{status}", self.styles)
        self.assertIn("message.is-review-target", self.styles)
        self.assertIn('state.filters.review_status = "pending"', self.javascript)
        self.assertIn('data-review-filter="regeneration"', self.html)

    def test_annotation_workspace_reuses_editor_and_supports_inline_small_screens(self) -> None:
        parser = _IdCollector()
        parser.feed(self.html)
        for element_id in ("browse-mode", "annotation-mode", "annotation-collection",
                           "annotation-auto-next", "annotation-list-toggle"):
            self.assertIn(element_id, parser.ids)
        self.assertIn('class="detail-text-pane"', self.html)
        self.assertIn('window.location.hash === "#annotate"', self.javascript)
        self.assertIn('params.set("has_dialogue", "true")', self.javascript)
        self.assertIn('body.annotation-mode .review-dashboard { order: 0;', self.styles)
        self.assertIn('body.annotation-mode .detail-panel { position: relative;', self.styles)
        self.assertIn('body.annotation-list-hidden .list-panel { display: none;', self.styles)

    def test_annotation_dialogue_has_fixed_scroll_area_but_edit_inputs_still_expand(self) -> None:
        for suffix in (".detail-text-pane", ".dialogue-editor", ".detail-scroll"):
            rules = re.findall(re.escape("body.annotation-mode " + suffix) + r"\s*\{([^}]+)\}", self.styles)
            self.assertTrue(rules, suffix)
            for rule in rules:
                self.assertNotRegex(rule, r"(?:max-)?height\s*:\s*\d+px", suffix)
                self.assertNotRegex(rule, r"overflow(?:-y)?\s*:\s*(?:auto|scroll|hidden)", suffix)
        self.assertIn('body.annotation-mode .dialogue-edit-text { max-height: none; overflow: hidden;', self.styles)
        self.assertIn('input.addEventListener("input"', self.javascript)
        self.assertIn('input.scrollHeight', self.javascript)
        self.assertIn('if (!keepVisible) dom.detailContent.hidden = true;', self.javascript)
        self.assertIn('"dialogue-heading-actions"', self.javascript)
        self.assertIn('id="review-context-details"', self.html)
        self.assertIn('--annotation-content-height: 600px;', self.styles)
        self.assertIn('body.annotation-mode .conversation { height: var(--annotation-content-height);', self.styles)
        self.assertRegex(self.styles, r"body\.annotation-mode \.conversation\s*\{[^}]*overflow-y: auto;")
        self.assertIn('--annotation-content-height: 420px;', self.styles)

    def test_annotation_panels_share_geometry_without_sticky_offsets(self) -> None:
        for selector in (".detail-header", ".image-section", ".dialogue-section > .detail-section-heading", ".list-panel"):
            rules = re.findall(re.escape("body.annotation-mode " + selector) + r"\s*\{([^}]+)\}", self.styles)
            self.assertTrue(rules, selector)
            self.assertFalse(any('position: sticky' in rule for rule in rules), selector)
        self.assertIn('grid-template-rows: var(--annotation-heading-height) var(--annotation-content-height);', self.styles)
        self.assertIn('grid-template-rows: var(--annotation-heading-height) var(--annotation-content-height) auto;', self.styles)
        self.assertIn('body.annotation-mode.annotation-editing #review-actions { display: none;', self.styles)
        self.assertNotIn('dom.detailDialogue.scrollIntoView({ behavior: "smooth"', self.javascript)

    def test_audio_and_alignment_are_independent_collapsed_details(self) -> None:
        class DetailCollector(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.details = {}
            def handle_starttag(self, tag, attrs):
                if tag == "details":
                    values = dict(attrs)
                    self.details[values.get("id")] = values
        parser = DetailCollector()
        parser.feed(self.html)
        for section in ("audio-section", "alignment-section"):
            self.assertIn(section, parser.details)
            self.assertNotIn("open", parser.details[section])
            self.assertIn(section, self.javascript)
        self.assertIn('dom.audioSection.open = true;', self.javascript)
        self.assertIn('if (!dom.audioSection.open) dom.detailAudio.pause();', self.javascript)


if __name__ == "__main__":
    unittest.main()
