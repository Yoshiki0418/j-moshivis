#!/usr/bin/env python3
"""Build the revised J-MoshiVis progress-report slides.

The original 36-slide deck is kept intact.  Sixteen new slides are inserted
after original slide 25.  New PowerPoint content is made from native text,
shapes, arrows and pictures; matching PNG renders are used to create the two
PDF deliverables without requiring LibreOffice or Microsoft PowerPoint.

The visual system deliberately follows the source deck: white canvas, a dark
header/footer, UD Digi Kyokasho typefaces, large explanatory diagrams, and a
single red conclusion instead of dashboard-style KPI cards.
"""

from __future__ import annotations

import hashlib
import math
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE, MSO_CONNECTOR
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PPTX = ROOT / "2026_0423_経過報告.pptx"
SOURCE_PDF = ROOT / "2026_0423_経過報告.pdf"
OUTPUT_PPTX = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_改訂版.pptx"
OUTPUT_PDF = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_改訂版.pdf"
ADDED_PDF = ROOT / "J-MoshiVis_改修進捗_追加スライド_改訂版.pdf"

W, H = 1920, 1080
PX_PER_INCH = 144.0
TOP_H = 117
FOOT_Y = 1025

CHARCOAL = "#404040"
BLACK = "#111111"
GRAY = "#777777"
LIGHT_GRAY = "#F2F2F2"
MID_GRAY = "#D9D9D9"
BLUE_FILL = "#E8F1F7"
BLUE = "#304158"
SKY = "#DFECF7"
GREEN_FILL = "#E7F3E2"
GREEN = "#1D472E"
BRIGHT_GREEN = "#36A856"
ORANGE_FILL = "#FFF7E2"
ORANGE = "#F6AA0A"
PEACH = "#FFE5D0"
RED = "#FF2631"
PALE_RED = "#FFEDEE"
WHITE = "#FFFFFF"

FONT_BOLD_NAME = "UD Digi Kyokasho NK-B"
FONT_REG_NAME = "UD Digi Kyokasho NK-R"
FALLBACK_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
FALLBACK_REG = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")


def rgb(value: str) -> RGBColor:
    value = value.lstrip("#")
    return RGBColor(int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))


def emu(px: float):
    return Inches(px / PX_PER_INCH)


def _cmap_codepoints(path: Path) -> set[int]:
    """Read cmap format 4/12 from a TrueType font without fontTools."""
    data = path.read_bytes()
    if len(data) < 12:
        return set()
    num_tables = struct.unpack_from(">H", data, 4)[0]
    cmap_offset = None
    for i in range(num_tables):
        pos = 12 + i * 16
        if pos + 16 > len(data):
            break
        tag, _, offset, _ = struct.unpack_from(">4sIII", data, pos)
        if tag == b"cmap":
            cmap_offset = offset
            break
    if cmap_offset is None or cmap_offset + 4 > len(data):
        return set()
    _, count = struct.unpack_from(">HH", data, cmap_offset)
    offsets: list[int] = []
    for i in range(count):
        pos = cmap_offset + 4 + i * 8
        if pos + 8 <= len(data):
            _, _, sub_offset = struct.unpack_from(">HHI", data, pos)
            offsets.append(cmap_offset + sub_offset)
    chars: set[int] = set()
    for sub in offsets:
        if sub + 2 > len(data):
            continue
        fmt = struct.unpack_from(">H", data, sub)[0]
        if fmt == 4 and sub + 16 <= len(data):
            length = struct.unpack_from(">H", data, sub + 2)[0]
            seg_count = struct.unpack_from(">H", data, sub + 6)[0] // 2
            if not seg_count or sub + length > len(data):
                continue
            end_pos = sub + 14
            start_pos = end_pos + seg_count * 2 + 2
            delta_pos = start_pos + seg_count * 2
            range_pos = delta_pos + seg_count * 2
            for i in range(seg_count):
                end = struct.unpack_from(">H", data, end_pos + i * 2)[0]
                start = struct.unpack_from(">H", data, start_pos + i * 2)[0]
                delta = struct.unpack_from(">h", data, delta_pos + i * 2)[0]
                ro = struct.unpack_from(">H", data, range_pos + i * 2)[0]
                if start > end or start == 0xFFFF:
                    continue
                for cp in range(start, end + 1):
                    if ro == 0:
                        glyph = (cp + delta) & 0xFFFF
                    else:
                        glyph_pos = range_pos + i * 2 + ro + 2 * (cp - start)
                        if glyph_pos + 2 > sub + length:
                            continue
                        glyph = struct.unpack_from(">H", data, glyph_pos)[0]
                        if glyph:
                            glyph = (glyph + delta) & 0xFFFF
                    if glyph:
                        chars.add(cp)
        elif fmt == 12 and sub + 16 <= len(data):
            length = struct.unpack_from(">I", data, sub + 4)[0]
            groups = struct.unpack_from(">I", data, sub + 12)[0]
            if sub + length > len(data):
                continue
            for i in range(groups):
                pos = sub + 16 + i * 12
                if pos + 12 > sub + length:
                    break
                start, end, _ = struct.unpack_from(">III", data, pos)
                if end - start <= 100000:
                    chars.update(range(start, end + 1))
    return chars


class FontManager:
    """Use the source deck's embedded UD fonts, with Noto only for new glyphs."""

    def __init__(self, source_pdf: Path, out_dir: Path):
        doc = fitz.open(source_pdf)
        wanted = {7: "bold_a", 29: "bold_b", 49: "regular"}
        self.bold_paths: list[Path] = []
        self.reg_paths: list[Path] = []
        for xref, stem in wanted.items():
            name, ext, _, payload = doc.extract_font(xref)
            path = out_dir / f"{stem}_{name}.{ext}"
            path.write_bytes(payload)
            (self.bold_paths if stem.startswith("bold") else self.reg_paths).append(path)
        doc.close()
        self.coverage = {p: _cmap_codepoints(p) for p in self.bold_paths + self.reg_paths}

    @lru_cache(maxsize=512)
    def _load(self, path: str, size: int):
        return ImageFont.truetype(path, size=size, index=0)

    def candidates(self, bold: bool):
        paths = self.bold_paths if bold else self.reg_paths
        fallback = FALLBACK_BOLD if bold else FALLBACK_REG
        return [(p, self.coverage.get(p)) for p in paths] + [(fallback, None)]

    def char_font(self, ch: str, size: int, bold: bool):
        cp = ord(ch)
        for path, coverage in self.candidates(bold):
            if coverage is None or cp in coverage or ch.isspace():
                return self._load(str(path), size)
        return self._load(str(FALLBACK_BOLD if bold else FALLBACK_REG), size)

    def length(self, text: str, size: int, bold: bool) -> float:
        return sum(self.char_font(ch, size, bold).getlength(ch) for ch in text)

    def wrap(self, text: str, size: int, bold: bool, width: float) -> list[str]:
        lines: list[str] = []
        for paragraph in text.split("\n"):
            if paragraph == "":
                lines.append("")
                continue
            current = ""
            current_w = 0.0
            for ch in paragraph:
                cw = self.char_font(ch, size, bold).getlength(ch)
                if current and current_w + cw > width:
                    lines.append(current.rstrip())
                    current = "" if ch == " " else ch
                    current_w = 0.0 if ch == " " else cw
                else:
                    current += ch
                    current_w += cw
            if current:
                lines.append(current.rstrip())
        return lines

    def draw_line(self, draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
                  size: int, bold: bool, fill: str):
        x, y = xy
        for ch in text:
            fnt = self.char_font(ch, size, bold)
            draw.text((x, y), ch, font=fnt, fill=fill, stroke_width=0)
            x += fnt.getlength(ch)


@dataclass
class Assets:
    communication: Path
    architecture: Path
    user_icon: Path
    robot_icon: Path
    robot_headset: Path
    chat: Path
    dog: Path
    japan_sea: Path
    akihabara: Path
    washitsu: Path


def extract_assets(out_dir: Path) -> Assets:
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    needed = {"image4.png", "image9.png", "image12.png", "image19.png", "image20.png", "image23.jpeg"}
    with zipfile.ZipFile(SOURCE_PPTX) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if member.startswith("ppt/media/") and name in needed:
                (media_dir / name).write_bytes(archive.read(member))
    return Assets(
        communication=media_dir / "image9.png",
        architecture=media_dir / "image12.png",
        user_icon=media_dir / "image19.png",
        robot_icon=media_dir / "image20.png",
        robot_headset=media_dir / "image4.png",
        chat=media_dir / "image23.jpeg",
        dog=ROOT / "data/speech_v3/data_stereo/b8042e2774215b0d11815e23ea70eae836685f4aacb0a1da02ef7f295f0dd202/image.jpg",
        japan_sea=ROOT / "data/images/japanese_photos/images/japanese_photos_10110.jpg",
        akihabara=ROOT / "data/images/wikimedia_tourist/Aerial_photographs_of_Akihabara/6247778de9b2834bba62c6f3c28417b916eee7d28357c3ef5393e047dd18939f.jpg",
        washitsu=ROOT / "data/images/openverse/images/00012ad12392cb2112f9c6e35ee675bf38ff6aeae3cbda3694eee744f9cc5c8d.jpg",
    )


class Canvas:
    """Small backend-neutral drawing API in 1920x1080 pixel coordinates."""

    def __init__(self, fm: FontManager):
        self.fm = fm

    def rect(self, x, y, w, h, fill=WHITE, outline=None, width=2, radius=0, dash=False):
        raise NotImplementedError

    def ellipse(self, x, y, w, h, fill=WHITE, outline=None, width=2):
        raise NotImplementedError

    def line(self, x1, y1, x2, y2, color=BLACK, width=3, dash=False):
        raise NotImplementedError

    def text(self, x, y, w, h, value, size=40, bold=False, color=BLACK,
             align="left", valign="top", wrap=True, margin=0):
        raise NotImplementedError

    def picture(self, path: Path, x, y, w, h, fit="contain"):
        raise NotImplementedError

    def arrow(self, x1, y1, x2, y2, color=BLACK, width=5, head=18):
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        p1 = (x2 - head * math.cos(angle - 0.55), y2 - head * math.sin(angle - 0.55))
        p2 = (x2 - head * math.cos(angle + 0.55), y2 - head * math.sin(angle + 0.55))
        self.triangle([(x2, y2), p1, p2], color)

    def triangle(self, points, fill):
        raise NotImplementedError

    def waveform(self, x, y, w, h, color=BLUE, strokes=36):
        mid = y + h / 2
        for i in range(strokes):
            phase = i / max(strokes - 1, 1)
            envelope = math.sin(math.pi * phase) ** 0.7
            carrier = 0.35 + 0.65 * abs(math.sin(i * 1.73 + 0.4))
            amp = h * 0.46 * envelope * carrier
            xx = x + w * i / max(strokes - 1, 1)
            self.line(xx, mid - amp, xx, mid + amp, color, 3)

    def bullet(self, x, y, w, text_value, size=38, color=BLACK, bullet_color=BLACK,
               marker="・", bold=False, line_gap=10):
        self.text(x, y, 42, size + 16, marker, size=size, bold=True, color=bullet_color)
        lines = self.fm.wrap(text_value, size, bold, w - 48)
        self.text(x + 46, y, w - 46, len(lines) * (size + line_gap), "\n".join(lines),
                  size=size, bold=bold, color=color, wrap=False)
        return y + len(lines) * (size + line_gap)

    def square_heading(self, x, y, text_value, size=44, color=BLACK):
        self.rect(x, y + 8, 24, 24, fill=color, outline=None)
        self.text(x + 42, y, min(1000, W - x - 70), size + 12,
                  text_value, size=size, bold=True, color=color)

    def token(self, x, y, w, text_value, fill=GREEN_FILL, outline=GREEN,
              size=34, text_color=BLACK):
        self.rect(x, y, w, 58, fill=fill, outline=outline, width=2, radius=6)
        self.text(x, y + 3, w, 50, text_value, size=size, bold=True, color=text_color,
                  align="center", valign="middle", wrap=False)


class ImageCanvas(Canvas):
    def __init__(self, fm: FontManager):
        super().__init__(fm)
        self.image = Image.new("RGB", (W, H), WHITE)
        self.draw = ImageDraw.Draw(self.image)

    def rect(self, x, y, w, h, fill=WHITE, outline=None, width=2, radius=0, dash=False):
        box = (round(x), round(y), round(x + w), round(y + h))
        if dash and outline:
            self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=None)
            dash_len = 16
            for xx in range(round(x), round(x + w), dash_len * 2):
                self.draw.line((xx, y, min(xx + dash_len, x + w), y), fill=outline, width=width)
                self.draw.line((xx, y + h, min(xx + dash_len, x + w), y + h), fill=outline, width=width)
            for yy in range(round(y), round(y + h), dash_len * 2):
                self.draw.line((x, yy, x, min(yy + dash_len, y + h)), fill=outline, width=width)
                self.draw.line((x + w, yy, x + w, min(yy + dash_len, y + h)), fill=outline, width=width)
        else:
            self.draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

    def ellipse(self, x, y, w, h, fill=WHITE, outline=None, width=2):
        self.draw.ellipse((x, y, x + w, y + h), fill=fill, outline=outline, width=width)

    def line(self, x1, y1, x2, y2, color=BLACK, width=3, dash=False):
        if not dash:
            self.draw.line((x1, y1, x2, y2), fill=color, width=width)
            return
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            return
        ux, uy = (x2 - x1) / length, (y2 - y1) / length
        pos = 0.0
        while pos < length:
            end = min(pos + 16, length)
            self.draw.line((x1 + ux * pos, y1 + uy * pos, x1 + ux * end, y1 + uy * end), fill=color, width=width)
            pos += 28

    def triangle(self, points, fill):
        self.draw.polygon(points, fill=fill)

    def text(self, x, y, w, h, value, size=40, bold=False, color=BLACK,
             align="left", valign="top", wrap=True, margin=0):
        lines = self.fm.wrap(value, size, bold, max(w - 2 * margin, 1)) if wrap else value.split("\n")
        line_h = round(size * 1.18)
        total_h = len(lines) * line_h
        ty = y + margin
        if valign == "middle":
            ty = y + (h - total_h) / 2
        elif valign == "bottom":
            ty = y + h - total_h - margin
        for line in lines:
            lw = self.fm.length(line, size, bold)
            tx = x + margin
            if align == "center":
                tx = x + (w - lw) / 2
            elif align == "right":
                tx = x + w - lw - margin
            self.fm.draw_line(self.draw, (tx, ty), line, size, bold, color)
            ty += line_h

    def picture(self, path: Path, x, y, w, h, fit="contain"):
        im = Image.open(path).convert("RGB")
        if fit == "cover":
            fitted = ImageOps.fit(im, (round(w), round(h)), method=Image.Resampling.LANCZOS)
            px, py = round(x), round(y)
        else:
            im.thumbnail((round(w), round(h)), Image.Resampling.LANCZOS)
            fitted = im
            px, py = round(x + (w - im.width) / 2), round(y + (h - im.height) / 2)
        self.image.paste(fitted, (px, py))


class PptCanvas(Canvas):
    def __init__(self, fm: FontManager, slide):
        super().__init__(fm)
        self.slide = slide

    @staticmethod
    def _font_run(run, name: str, size: int, color: str, bold: bool):
        run.font.name = name
        run.font.size = Pt(size / 2)
        run.font.bold = bold
        run.font.color.rgb = rgb(color)
        rpr = run._r.get_or_add_rPr()
        for tag in ("a:latin", "a:ea", "a:cs"):
            existing = rpr.find(tag, rpr.nsmap)
            if existing is not None:
                rpr.remove(existing)
            node = OxmlElement(tag)
            node.set("typeface", name)
            rpr.append(node)

    def rect(self, x, y, w, h, fill=WHITE, outline=None, width=2, radius=0, dash=False):
        kind = MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE if radius else MSO_AUTO_SHAPE_TYPE.RECTANGLE
        sh = self.slide.shapes.add_shape(kind, emu(x), emu(y), emu(w), emu(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
        if outline:
            sh.line.color.rgb = rgb(outline)
            sh.line.width = Pt(width / 2)
            if dash:
                sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        else:
            sh.line.fill.background()
        if radius:
            try:
                sh.adjustments[0] = min(0.25, radius / max(min(w, h), 1))
            except (IndexError, ValueError):
                pass
        return sh

    def ellipse(self, x, y, w, h, fill=WHITE, outline=None, width=2):
        sh = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.OVAL, emu(x), emu(y), emu(w), emu(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
        if outline:
            sh.line.color.rgb = rgb(outline)
            sh.line.width = Pt(width / 2)
        else:
            sh.line.fill.background()
        return sh

    def line(self, x1, y1, x2, y2, color=BLACK, width=3, dash=False):
        sh = self.slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, emu(x1), emu(y1), emu(x2), emu(y2))
        sh.line.color.rgb = rgb(color)
        sh.line.width = Pt(width / 2)
        if dash:
            sh.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        return sh

    def triangle(self, points, fill):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x, y, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
        sh = self.slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ISOSCELES_TRIANGLE, emu(x), emu(y), emu(max(w, 2)), emu(max(h, 2)))
        sh.rotation = 90 + math.degrees(math.atan2(points[0][1] - (y + h / 2), points[0][0] - (x + w / 2)))
        sh.fill.solid()
        sh.fill.fore_color.rgb = rgb(fill)
        sh.line.fill.background()

    def text(self, x, y, w, h, value, size=40, bold=False, color=BLACK,
             align="left", valign="top", wrap=True, margin=0):
        lines = self.fm.wrap(value, size, bold, max(w - 2 * margin, 1)) if wrap else value.split("\n")
        value = "\n".join(lines)
        sh = self.slide.shapes.add_textbox(emu(x), emu(y), emu(w), emu(h))
        tf = sh.text_frame
        tf.clear()
        tf.word_wrap = False
        tf.margin_left = emu(margin)
        tf.margin_right = emu(margin)
        tf.margin_top = emu(margin)
        tf.margin_bottom = emu(margin)
        tf.vertical_anchor = {"top": MSO_ANCHOR.TOP, "middle": MSO_ANCHOR.MIDDLE, "bottom": MSO_ANCHOR.BOTTOM}[valign]
        for i, line in enumerate(lines):
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = {"left": PP_ALIGN.LEFT, "center": PP_ALIGN.CENTER, "right": PP_ALIGN.RIGHT}[align]
            p.space_after = Pt(0)
            p.space_before = Pt(0)
            p.line_spacing = 1.0
            run = p.add_run()
            run.text = line
            self._font_run(run, FONT_BOLD_NAME if bold else FONT_REG_NAME, size, color, bold)
        return sh

    def picture(self, path: Path, x, y, w, h, fit="contain"):
        with Image.open(path) as im:
            iw, ih = im.size
        box_ratio = w / h
        image_ratio = iw / ih
        if fit == "cover":
            sh = self.slide.shapes.add_picture(str(path), emu(x), emu(y), emu(w), emu(h))
            if image_ratio > box_ratio:
                visible = box_ratio / image_ratio
                sh.crop_left = sh.crop_right = (1 - visible) / 2
            else:
                visible = image_ratio / box_ratio
                sh.crop_top = sh.crop_bottom = (1 - visible) / 2
            return sh
        scale = min(w / iw, h / ih)
        pw, ph = iw * scale, ih * scale
        return self.slide.shapes.add_picture(str(path), emu(x + (w - pw) / 2), emu(y + (h - ph) / 2), emu(pw), emu(ph))


def header(c: Canvas, title: str, number: int):
    c.rect(0, 0, W, TOP_H, fill=CHARCOAL, outline=None)
    c.text(32, 14, 1500, 84, title, size=56, bold=True, color=WHITE, wrap=False, valign="middle")
    c.rect(0, FOOT_Y, W, H - FOOT_Y, fill=CHARCOAL, outline=None)
    labels = ["学長AI", "研究概要", "視覚音声対話モデル", "行動生成モデル", "今後の展望"]
    x = 18
    for i, label in enumerate(labels):
        color = WHITE if i == 2 else "#A6A6A6"
        width = c.fm.length(label, 28, False) + 8
        c.text(x, FOOT_Y + 8, width, 40, label, size=28, color=color, wrap=False)
        x += width
        if i < len(labels) - 1:
            c.text(x, FOOT_Y + 8, 42, 40, "→", size=28, color="#A6A6A6", align="center", wrap=False)
            x += 42
    c.text(1765, FOOT_Y + 7, 125, 42, str(number), size=28, color=WHITE, align="right", wrap=False)


def lead(c: Canvas, text_value: str, color=BLACK):
    c.text(32, 145, 1845, 65, text_value, size=48, bold=True, color=color, wrap=False)


def conclusion(c: Canvas, text_value: str, y=945, color=RED, size=40):
    c.text(32, y, 1845, 58, "→ " + text_value, size=size, bold=True, color=color, wrap=False)


def file_box(c: Canvas, x, y, w, text_value, color=BLUE):
    c.rect(x, y, w, 55, fill=WHITE, outline=color, width=2, radius=8)
    c.text(x + 8, y + 4, w - 16, 45, text_value, size=27, bold=True, color=color,
           align="center", valign="middle", wrap=False)


def person_icon(c: Canvas, x, y, scale=1.0, color=BLACK):
    c.ellipse(x + 32 * scale, y, 54 * scale, 54 * scale, fill=color, outline=None)
    c.rect(x, y + 62 * scale, 118 * scale, 85 * scale, fill=color, outline=None, radius=45 * scale)
    c.rect(x, y + 108 * scale, 118 * scale, 50 * scale, fill=WHITE, outline=None)


def server_icon(c: Canvas, x, y, w=300, h=340, fill=LIGHT_GRAY, outline=BLUE):
    c.rect(x, y, w, h, fill=fill, outline=outline, width=4, radius=18)
    for yy in (y + 38, y + 128, y + 218):
        c.rect(x + 32, yy, w - 64, 66, fill=WHITE, outline=outline, width=2, radius=8)
        c.ellipse(x + w - 80, yy + 21, 20, 20, fill=BRIGHT_GREEN, outline=None)
        c.line(x + 58, yy + 23, x + 160, yy + 23, color=outline, width=3)
        c.line(x + 58, yy + 42, x + 135, yy + 42, color=outline, width=3)


def train_folder(c: Canvas, x, y, w, h, label, color=BLUE_FILL, outline=BLUE):
    c.rect(x + 8, y, w * 0.42, 34, fill=color, outline=outline, width=2, radius=6)
    c.rect(x, y + 24, w, h - 24, fill=color, outline=outline, width=3, radius=12)
    c.text(x, y + 42, w, h - 56, label, size=36, bold=True, color=outline,
           align="center", valign="middle", wrap=False)


SLIDE_BUILDERS = []
NOTES: list[str] = []


def slide(note: str):
    def decorator(fn):
        SLIDE_BUILDERS.append(fn)
        NOTES.append(note)
        return fn
    return decorator


@slide("ここからは、別サーバーへの移行を契機に行ったJ-MoshiVisの改修を報告します。特に、旧データを移さずゼロから作り直した新データセットと、音声とSPMトークンの同期改修を中心に説明します。")
def slide_01_section(c: Canvas, a: Assets):
    c.text(420, 420, 1080, 100, "J-MoshiVis 改修進捗", size=80, bold=True,
           align="center", valign="middle", wrap=False)
    c.line(545, 535, 1375, 535, color=CHARCOAL, width=3)


@slide("学習環境を別サーバーへ移す際、旧データセットはあえて移行しませんでした。画像分布と学習信号の品質を設計し直し、画像、対話、音声、検証結果まで追跡できるデータセットをゼロから構築する判断をしました。")
def slide_02_migration(c: Canvas, a: Assets):
    header(c, "学習環境の移行を、データ再設計の機会に", 26)
    lead(c, "旧データをそのまま移さず、品質を定義し直してゼロから構築")

    server_icon(c, 165, 290, 330, 350, fill=LIGHT_GRAY, outline="#777777")
    c.text(125, 650, 410, 58, "旧サーバー", size=46, bold=True, color="#666666", align="center", wrap=False)
    c.text(105, 720, 450, 120, "約5.5万対話\n約1,000時間", size=42, bold=True, color="#555555", align="center", wrap=False)
    c.line(100, 278, 560, 855, color=RED, width=8)
    c.line(560, 278, 100, 855, color=RED, width=8)
    c.text(130, 865, 420, 45, "旧データは移行しない", size=35, bold=True, color=RED, align="center", wrap=False)

    c.arrow(620, 515, 930, 515, color=CHARCOAL, width=8, head=28)
    c.text(635, 425, 280, 65, "コード・重み", size=38, bold=True, align="center", wrap=False)
    c.text(635, 545, 280, 50, "別サーバーへ", size=34, color=GRAY, align="center", wrap=False)

    server_icon(c, 1015, 290, 330, 350, fill=BLUE_FILL, outline=BLUE)
    c.text(975, 650, 410, 58, "新サーバー", size=46, bold=True, color=BLUE, align="center", wrap=False)

    c.square_heading(1430, 275, "再構築するもの", size=42)
    yy = 355
    for label in ("日本を含む画像分布", "自然な多ターン対話", "正しい音声・SPM同期", "中断・再開できる生成工程"):
        yy = c.bullet(1430, yy, 440, label, size=34, bullet_color=BLUE, line_gap=8) + 18

    conclusion(c, "移行を単なるコピーではなく、学習データの品質を作り直す機会にした", y=950)


@slide("旧データは約5.5万対話、約1,000時間でした。一方、画像はPixMo、DOCCI、PixelProseだけで、日本の風景や生活画像はほぼありませんでした。CERは音声と文字列の一致度ですが、モデルが実際に学ぶSPM ID列の正しさまでは確認できません。")
def slide_03_old_dataset(c: Canvas, a: Assets):
    header(c, "以前のデータセット — 規模はあるが、分布と学習信号に課題", 27)
    lead(c, "約5.5万対話・約1,000時間を、3種類の画像データから生成")

    c.square_heading(45, 250, "規模", size=44)
    c.text(85, 315, 500, 85, "約5.5万件", size=72, bold=True, color=BLUE, wrap=False)
    c.text(90, 405, 500, 55, "約1,000時間", size=52, bold=True, color=BLACK, wrap=False)
    c.text(90, 475, 500, 90, "1対話平均：約66秒\n　　　　　　約10ターン", size=34, color=BLACK, wrap=False)

    c.square_heading(655, 250, "対話タスク", size=44)
    bar_x, bar_y, bar_w, bar_h = 700, 335, 1115, 86
    values = [(41.6, "#7897C5"), (32.7, "#8CCAD4"), (14.8, "#F3BE70"), (9.0, "#99C58D"), (1.9, MID_GRAY)]
    pos = bar_x
    for value, color in values:
        ww = bar_w * value / 100
        c.rect(pos, bar_y, ww, bar_h, fill=color, outline=WHITE, width=2)
        pos += ww
    labels = [
        (700, "質問応答\n41.6%", "#5577A8"),
        (1010, "画像説明\n32.7%", "#4B9BA8"),
        (1290, "推測\n14.8%", "#C58B32"),
        (1490, "雑談\n9.0%", "#639B55"),
    ]
    for xx, txt, color in labels:
        c.text(xx, 445, 250, 90, txt, size=31, bold=True, color=color, align="center", wrap=False)
    c.text(705, 555, 1080, 45, "※ 提示値の合計は98.1%。残り1.9%はその他・丸め差", size=27, color=GRAY, wrap=False)

    c.square_heading(45, 620, "当時の品質指標", size=44)
    c.text(90, 690, 560, 120, "CER中央値　約5.8%\n有効トークン　約4,570万\nPAD率　　　　約59%", size=36, bold=True, wrap=False)

    c.square_heading(840, 620, "画像ソース", size=44)
    file_box(c, 885, 692, 230, "PixMo", color=BLUE)
    file_box(c, 1160, 692, 230, "DOCCI", color=BLUE)
    file_box(c, 1435, 692, 300, "PixelProse", color=BLUE)
    c.text(890, 775, 845, 72, "日本の風景・文化・日常画像は\nほぼ含まれていなかったと推測", size=34, bold=True, color=RED, align="center", wrap=False)

    conclusion(c, "CERだけでは、学習時のSPMトークン列が正しいか保証できない", y=946)


@slide("新データでは170,141件を生成し、全件を検査したうえで170,021件を採用しました。利用可能な対話は約3.1倍、音声時間は約2.5倍です。量だけでなく、日本関連画像、正しいSPM列、再開可能な工程を同時に整えました。")
def slide_04_v3_overview(c: Canvas, a: Assets):
    header(c, "新データセット v3 — 量・多様性・検証可能性を同時に改善", 28)
    lead(c, "170,141件を生成し、全件検査後の170,021件を学習・評価に採用")

    c.text(80, 275, 490, 95, "170,021件", size=82, bold=True, color=BLUE, align="center", wrap=False)
    c.text(80, 385, 490, 58, "利用可能な対話", size=42, bold=True, align="center", wrap=False)
    c.text(660, 275, 490, 95, "2,515.1時間", size=75, bold=True, color=BLUE, align="center", wrap=False)
    c.text(660, 385, 490, 58, "train + eval", size=42, bold=True, align="center", wrap=False)
    c.text(1235, 275, 600, 95, "12.29ターン", size=70, bold=True, color=BLUE, align="center", wrap=False)
    c.text(1235, 385, 600, 58, "平均（中央値12）", size=42, bold=True, align="center", wrap=False)
    c.line(600, 265, 600, 460, color=MID_GRAY, width=3)
    c.line(1190, 265, 1190, 460, color=MID_GRAY, width=3)

    c.square_heading(55, 510, "旧データからの変化", size=44)
    c.text(100, 580, 315, 70, "約5.5万件", size=46, bold=True, color=GRAY, align="center", wrap=False)
    c.arrow(430, 616, 610, 616, color=BLUE, width=7, head=24)
    c.text(445, 555, 150, 45, "3.09倍", size=34, bold=True, color=BLUE, align="center", wrap=False)
    c.text(630, 580, 315, 70, "17.0万件", size=52, bold=True, color=BLUE, align="center", wrap=False)
    c.text(100, 695, 315, 70, "約1,000時間", size=46, bold=True, color=GRAY, align="center", wrap=False)
    c.arrow(430, 731, 610, 731, color=BLUE, width=7, head=24)
    c.text(445, 670, 150, 45, "2.52倍", size=34, bold=True, color=BLUE, align="center", wrap=False)
    c.text(630, 695, 315, 70, "2,515時間", size=52, bold=True, color=BLUE, align="center", wrap=False)

    c.square_heading(1050, 510, "画像分布", size=44)
    bx, by, bw, bh = 1095, 590, 690, 76
    parts = [(72.4, "#8CB1D1"), (8.7, "#BFD7E8"), (18.9, "#92C98B")]
    pos = bx
    for val, color in parts:
        ww = bw * val / 100
        c.rect(pos, by, ww, bh, fill=color, outline=WHITE, width=2)
        pos += ww
    c.text(1085, 690, 720, 145,
           "COCO　123,185件（72.4%）\nDOCCI　14,844件（8.7%）\n日本関連7ソース　32,112件（18.9%）",
           size=34, bold=True, wrap=False)

    conclusion(c, "規模だけでなく、画像分布・学習信号・再現性を作り直した", y=950)


@slide("ここでは実際の1件を追います。画像、対話JSON、2チャンネル音声を同じUIDのフォルダへ保存します。これにより、どの画像と生成器、話者から作られた学習例かを後からたどれます。")
def slide_05_one_example(c: Canvas, a: Assets):
    header(c, "1件の学習データを具体的に見る", 29)
    lead(c, "画像・対話・2ch音声を、同じUIDでひも付けて保存")

    c.picture(a.dog, 55, 250, 585, 415, fit="cover")
    c.rect(55, 250, 585, 415, fill=WHITE, outline=BLUE, width=3)
    c.picture(a.dog, 61, 256, 573, 403, fit="cover")
    file_box(c, 140, 690, 420, "image.jpg", color=BLUE)

    c.picture(a.user_icon, 705, 260, 100, 130, fit="contain")
    c.rect(835, 255, 955, 105, fill=BLUE_FILL, outline=BLUE, width=3, radius=18)
    c.text(865, 275, 895, 70, "この写真、誰か生き物とか写ってる？", size=37, bold=True, valign="middle", wrap=False)
    c.picture(a.robot_icon, 705, 425, 110, 145, fit="contain")
    c.rect(835, 415, 955, 145, fill=GREEN_FILL, outline=GREEN, width=3, radius=18)
    c.text(865, 435, 895, 100, "はい、犬がすごく近くでカメラを見つめてますよ。\n顔のアップです", size=35, bold=True, valign="middle", wrap=False)
    file_box(c, 1085, 600, 450, "dialogue.json", color=GREEN)

    c.text(720, 690, 170, 45, "ch0 AI", size=31, bold=True, color=GREEN, wrap=False)
    c.waveform(885, 690, 770, 62, color="#E78A68")
    c.text(720, 770, 170, 45, "ch1 User", size=31, bold=True, color=BLUE, wrap=False)
    c.waveform(885, 770, 770, 62, color="#72BF73")
    file_box(c, 1085, 850, 450, "stereo_dialogue.wav", color=BLUE)

    c.text(80, 880, 720, 50, "Claude Sonnet 5 / user: 小春音アミ / AI: ほのか / 54.017秒", size=28, color=GRAY, wrap=False)
    conclusion(c, "UIDをキーに、画像・生成元・話者・音声・検証結果を追跡できる", y=950)


@slide("新しいパイプラインは、画像収集、対話生成、2チャンネル音声合成、音声とSPMの同期、全件検査と分割の五つの大きな段階です。中間ファイルをUIDとshard単位で保存するため、長時間ジョブが止まっても生成済みの続きから再開できます。")
def slide_06_pipeline(c: Canvas, a: Assets):
    header(c, "新データセット構築パイプライン — 1枚の画像から学習データまで", 30)
    lead(c, "各工程の入出力を残し、170,141件すべてを同じ手順で処理")

    xs = [35, 405, 775, 1145, 1515]
    widths = [300, 300, 300, 300, 360]
    fills = [BLUE_FILL, BLUE_FILL, ORANGE_FILL, GREEN_FILL, GREEN_FILL]
    outlines = [BLUE, BLUE, ORANGE, GREEN, GREEN]
    titles = ["① 画像収集", "② 対話生成", "③ 2ch TTS", "④ 音声・SPM同期", "⑤ 全件検査・分割"]
    for x, w, fill, out, title in zip(xs, widths, fills, outlines, titles):
        c.rect(x, 260, w, 570, fill=fill, outline=out, width=3, radius=18)
        c.text(x + 10, 280, w - 20, 58, title, size=38, bold=True, color=out, align="center", wrap=False)
    for i in range(4):
        c.arrow(xs[i] + widths[i] + 8, 540, xs[i + 1] - 12, 540, color=CHARCOAL, width=6, head=20)

    c.picture(a.dog, 75, 360, 220, 155, fit="cover")
    file_box(c, 75, 565, 220, "image.jpg", color=BLUE)
    c.text(65, 650, 240, 105, "破損・極端な縦横比\n低情報量・近似重複を除外", size=28, align="center", wrap=False)

    c.picture(a.user_icon, 445, 360, 70, 95, fit="contain")
    c.rect(525, 350, 135, 62, fill=WHITE, outline=BLUE, width=2, radius=12)
    c.text(532, 360, 120, 42, "質問", size=28, bold=True, align="center", wrap=False)
    c.picture(a.robot_icon, 445, 480, 74, 100, fit="contain")
    c.rect(525, 475, 135, 62, fill=WHITE, outline=GREEN, width=2, radius=12)
    c.text(532, 485, 120, 42, "応答", size=28, bold=True, align="center", wrap=False)
    file_box(c, 445, 630, 220, "dialogue JSON", color=BLUE)
    c.text(430, 700, 250, 65, "Sonnet + Qwen\nJSON・話者を検査", size=28, align="center", wrap=False)

    c.text(810, 370, 105, 42, "ch0", size=29, bold=True, color=ORANGE, wrap=False)
    c.waveform(895, 365, 135, 54, color="#E78A68", strokes=18)
    c.text(810, 475, 105, 42, "ch1", size=29, bold=True, color=BLUE, wrap=False)
    c.waveform(895, 470, 135, 54, color="#72BF73", strokes=18)
    file_box(c, 815, 630, 220, "stereo WAV", color=ORANGE)
    c.text(800, 700, 250, 65, "24 kHz stereo\n欠損ファイル 0件", size=28, align="center", wrap=False)

    file_box(c, 1180, 350, 230, "WAV + 元のAI文", color=GREEN)
    c.waveform(1180, 430, 230, 62, color=BLUE, strokes=24)
    token_x = 1175
    for txt, ww in [("▁これは", 105), ("犬", 55), ("です", 70)]:
        c.token(token_x, 520, ww, txt, size=25)
        token_x += ww + 8
    file_box(c, 1180, 630, 230, "token ID + time", color=GREEN)
    c.text(1170, 700, 250, 65, "元文SPM +\nWhisperの発音時刻", size=28, align="center", wrap=False)

    c.text(1545, 370, 300, 70, "170,141件", size=50, bold=True, color=GREEN, align="center", wrap=False)
    c.arrow(1695, 445, 1695, 520, color=GREEN, width=6, head=18)
    train_folder(c, 1545, 545, 140, 120, "train", BLUE_FILL, BLUE)
    train_folder(c, 1710, 545, 140, 120, "eval", GREEN_FILL, GREEN)
    c.text(1540, 700, 315, 75, "CER不良120件を除外\n170,021件を採用", size=29, bold=True, align="center", wrap=False)

    c.rect(235, 865, 1450, 62, fill=LIGHT_GRAY, outline=CHARCOAL, width=2, radius=8)
    c.text(250, 874, 1420, 45, "UID / shard ごとに中間生成物を保存　→　中断しても生成済み分から再開", size=34, bold=True, align="center", wrap=False)
    conclusion(c, "生成工程そのものを、再現・検査・再開できるようにした", y=952)


@slide("画像はCOCOとDOCCIに加え、日本写真、Wikimedia、Openverseなど七つの日本関連ソースを追加しました。最終データの18.9パーセントが日本関連画像です。対話は同じルールでSonnetとQwenの二系統から生成し、生成元を記録しています。")
def slide_07_diversity(c: Canvas, a: Assets):
    header(c, "画像と対話の多様化 — 日本の日常を含む分布へ", 31)
    lead(c, "日本の風景・都市・室内を追加し、対話生成は品質系と規模系を併用")

    photos = [
        (a.japan_sea, 45, "日本の風景"),
        (a.akihabara, 355, "秋葉原・都市"),
        (a.washitsu, 665, "和室・日常"),
    ]
    for path, x, label in photos:
        c.picture(path, x, 255, 280, 230, fit="cover")
        c.rect(x, 255, 280, 230, fill=WHITE, outline=BLUE, width=3)
        c.picture(path, x + 5, 260, 270, 220, fit="cover")
        c.text(x, 495, 280, 45, label, size=30, bold=True, align="center", wrap=False)

    c.square_heading(45, 570, "最終データの画像ソース", size=40)
    bx, by, bw, bh = 90, 645, 820, 82
    parts = [(72.4, "#8CB1D1"), (8.7, "#BCD4E5"), (18.9, "#8CC781")]
    pos = bx
    for value, color in parts:
        ww = bw * value / 100
        c.rect(pos, by, ww, bh, fill=color, outline=WHITE, width=2)
        pos += ww
    c.text(90, 750, 830, 125,
           "COCO　123,185件（72.4%）\nDOCCI　14,844件（8.7%）\n日本関連7ソース　32,112件（18.9%）",
           size=31, bold=True, wrap=False)

    c.line(985, 245, 985, 890, color=MID_GRAY, width=3)
    c.square_heading(1040, 255, "対話生成の2系統", size=40)
    c.rect(1085, 345, 300, 150, fill=BLUE_FILL, outline=BLUE, width=3, radius=16)
    c.text(1095, 360, 280, 55, "Claude Sonnet 5", size=34, bold=True, color=BLUE, align="center", wrap=False)
    c.text(1095, 430, 280, 45, "32,068件", size=38, bold=True, align="center", wrap=False)
    c.text(1405, 405, 70, 50, "+", size=48, bold=True, align="center", wrap=False)
    c.rect(1495, 345, 340, 150, fill=GREEN_FILL, outline=GREEN, width=3, radius=16)
    c.text(1505, 360, 320, 55, "Qwen3-VL-30B", size=34, bold=True, color=GREEN, align="center", wrap=False)
    c.text(1505, 430, 320, 45, "138,073件", size=38, bold=True, align="center", wrap=False)

    c.square_heading(1040, 560, "共通する対話ルール", size=40)
    yy = 625
    for item in (
        "ユーザーは画像を見ていない設定",
        "初手で画像について質問する",
        "8〜12ターン・1発話1〜2文",
        "12種類の導入で単調化を抑える",
    ):
        yy = c.bullet(1070, yy, 750, item, size=33, bullet_color=BLUE, line_gap=6) + 10
    conclusion(c, "画像の地域偏りと、対話の言い回しの偏りを同時に減らす", y=950)


@slide("音声は24キロヘルツのステレオです。チャンネル0のアシスタントは、ほのかに固定して出力の一貫性を保ちます。チャンネル1のユーザーは16話者から選び、入力音声の多様性を確保します。7パーセントは軽く発話を重ね、全二重対話らしいタイミングも含めます。")
def slide_08_tts(c: Canvas, a: Assets):
    header(c, "音声合成の再設計 — AIは一貫、ユーザーは多様", 32)
    lead(c, "アシスタントの声を固定しながら、入力話者と会話タイミングを多様化")

    c.picture(a.communication, 650, 245, 620, 350, fit="contain")
    c.text(790, 600, 340, 45, "自然な全二重対話", size=35, bold=True, color=BLUE, align="center", wrap=False)

    c.square_heading(45, 260, "ch0　アシスタント", size=40, color=ORANGE)
    c.picture(a.robot_icon, 95, 340, 125, 160, fit="contain")
    c.text(245, 345, 330, 70, "ほのか", size=60, bold=True, color=ORANGE, wrap=False)
    c.text(245, 425, 330, 110, "AivisHub音声に固定\n内容に応じて韻律を切替", size=33, bold=True, wrap=False)
    c.waveform(85, 575, 470, 85, color="#E78A68", strokes=30)

    c.square_heading(1320, 260, "ch1　ユーザー", size=40, color=GREEN)
    c.picture(a.user_icon, 1380, 340, 115, 150, fit="contain")
    c.text(1520, 345, 330, 70, "16話者", size=60, bold=True, color=GREEN, wrap=False)
    c.text(1520, 425, 330, 110, "男女・複数音声から選択\n入力話者への依存を抑制", size=33, bold=True, wrap=False)
    c.waveform(1350, 575, 470, 85, color="#72BF73", strokes=30)

    c.line(110, 745, 1810, 745, color=MID_GRAY, width=3)
    c.text(145, 785, 420, 75, "24 kHz stereo", size=48, bold=True, color=BLUE, align="center", wrap=False)
    c.text(750, 775, 420, 85, "7%", size=65, bold=True, color=ORANGE, align="center", wrap=False)
    c.text(725, 855, 470, 45, "軽い発話の重なり", size=34, bold=True, align="center", wrap=False)
    c.text(1360, 785, 420, 75, "欠損 0件", size=48, bold=True, color=GREEN, align="center", wrap=False)
    conclusion(c, "出力音声の安定性と、入力・タイミングの多様性を両立", y=950)


@slide("SPMは文章をモデルが扱う小さな部品へ分ける処理です。合成音声なので何を話したかという正解文は既にあります。Whisperに任せるのは文字起こしではなく、その文字がいつ発音されたかを調べる部分です。")
def slide_09_spm_concept(c: Canvas, a: Assets):
    header(c, "SPM同期① — 学習に必要なのは「何を」と「いつ」", 33)
    lead(c, "合成音声なので正解文は既知。Whisperは発音時刻を調べるために使う")

    c.square_heading(55, 270, "何を話したか", size=42, color=GREEN)
    c.rect(85, 350, 610, 105, fill=GREEN_FILL, outline=GREEN, width=3, radius=16)
    c.text(100, 370, 580, 65, "正解文「これは犬です」", size=44, bold=True, align="center", valign="middle", wrap=False)
    c.arrow(390, 475, 390, 560, color=GREEN, width=6, head=20)
    c.text(225, 490, 330, 45, "文全体をSPMへ", size=31, bold=True, color=GREEN, align="center", wrap=False)
    token_x = 105
    for txt, ww in [("▁これは", 210), ("犬", 110), ("です", 155)]:
        c.token(token_x, 590, ww, txt, size=35)
        token_x += ww + 20
    c.text(90, 680, 600, 90, "SPM（SentencePiece）\n＝ 文章をAI用の部品へ分ける", size=33, bold=True, color=GRAY, align="center", wrap=False)

    c.square_heading(765, 270, "いつ話したか", size=42, color=BLUE)
    c.rect(795, 350, 610, 105, fill=BLUE_FILL, outline=BLUE, width=3, radius=16)
    c.waveform(840, 365, 520, 70, color=BLUE, strokes=38)
    c.arrow(1100, 475, 1100, 560, color=BLUE, width=6, head=20)
    c.text(935, 490, 330, 45, "Whisperで発音時刻", size=31, bold=True, color=BLUE, align="center", wrap=False)
    c.rect(860, 590, 480, 80, fill=WHITE, outline=BLUE, width=3, radius=12)
    c.text(875, 605, 450, 50, "こ 0.40　れ 0.48　は 0.56 …", size=32, bold=True, color=BLUE, align="center", wrap=False)
    c.text(820, 700, 560, 68, "文字列は正解として使わず\n時刻情報だけを利用", size=33, bold=True, color=GRAY, align="center", wrap=False)

    c.arrow(705, 620, 770, 620, color=CHARCOAL, width=6, head=20)
    c.arrow(1415, 620, 1480, 620, color=CHARCOAL, width=6, head=20)
    c.rect(1505, 350, 350, 430, fill=ORANGE_FILL, outline=ORANGE, width=3, radius=20)
    c.text(1535, 380, 290, 55, "時刻つきSPM", size=40, bold=True, color="#9B6500", align="center", wrap=False)
    c.waveform(1545, 470, 270, 70, color="#9B6500", strokes=26)
    c.token(1545, 575, 130, "▁これは", size=25, fill=WHITE, outline=ORANGE)
    c.token(1688, 575, 55, "犬", size=25, fill=WHITE, outline=ORANGE)
    c.token(1755, 575, 70, "です", size=25, fill=WHITE, outline=ORANGE)
    c.text(1535, 665, 290, 72, "各部品を発音位置へ\n配置して学習", size=32, bold=True, align="center", wrap=False)

    conclusion(c, "Moshiは80msごとに音声とテキストを同時に学習する", y=950)


@slide("旧方式では、Whisperが区切った一文字ずつを別々にSentencePieceへ入れていました。そのため毎回区切り記号が付き、正しい三トークンが十二トークンへ変わります。見た目の文章は同じでも、モデルが学ぶID列は別物です。CERだけではこの破損を検出できません。")
def slide_10_spm_bug(c: Canvas, a: Assets):
    header(c, "SPM同期② — 旧方式では、同じ文章が別のトークン列に変化", 34)
    lead(c, "Whisperの時間区切りと、モデルが使うSPMの区切りは一致しない")

    c.square_heading(55, 250, "本来の処理：文全体を1回だけSPMへ", size=40, color=GREEN)
    c.text(90, 330, 330, 60, "これは犬です", size=44, bold=True, align="center", wrap=False)
    c.arrow(440, 365, 610, 365, color=GREEN, width=6, head=20)
    token_x = 655
    for txt, ww in [("▁これは", 220), ("犬", 110), ("です", 155)]:
        c.token(token_x, 335, ww, txt, size=36)
        token_x += ww + 20
    c.text(1260, 335, 330, 60, "3トークン", size=42, bold=True, color=GREEN, align="center", wrap=False)
    c.text(1595, 325, 100, 70, "✓", size=58, bold=True, color=BRIGHT_GREEN, align="center", wrap=False)
    c.text(660, 410, 610, 45, "正しいID：444, 1651, 2767", size=29, color=GRAY, align="center", wrap=False)

    c.line(50, 490, 1865, 490, color=MID_GRAY, width=3)
    c.square_heading(55, 530, "旧ローダ：Whisper断片を1つずつ再SPM", size=40, color=RED)
    c.text(85, 605, 260, 55, "Whisper区切り", size=32, bold=True, wrap=False)
    fragments = ["こ", "れ", "は", "犬", "で", "す"]
    x = 405
    for frag in fragments:
        c.token(x, 600, 100, frag, size=34, fill=BLUE_FILL, outline=BLUE)
        x += 120
    c.text(1160, 610, 620, 45, "本データでは区切りの約75%が1文字", size=30, bold=True, color=BLUE, wrap=False)
    c.arrow(885, 680, 885, 735, color=RED, width=6, head=18)
    c.text(85, 760, 260, 55, "個別に再SPM", size=32, bold=True, wrap=False)
    x = 375
    wrong = ["▁", "こ", "▁", "れ", "▁", "は", "▁", "犬", "▁", "で", "▁", "す"]
    for i, tok in enumerate(wrong):
        fill = PALE_RED if tok == "▁" else WHITE
        c.token(x, 750, 78, tok, size=31, fill=fill, outline=RED)
        x += 88
    c.text(1470, 750, 250, 60, "12トークン", size=40, bold=True, color=RED, align="center", wrap=False)
    c.text(1730, 742, 90, 70, "×", size=58, bold=True, color=RED, align="center", wrap=False)
    c.text(385, 835, 590, 45, "▁ ＝ 文頭・区切りを示す印", size=30, bold=True, color=RED, wrap=False)
    c.text(1020, 835, 750, 45, "CERは音声と文字の一致を見るため、この差を検出できない", size=29, color=GRAY, wrap=False)

    conclusion(c, "見た目の文が同じでも、モデルが学ぶ数字の並びは別物になる", y=948)


@slide("改修後は、元の正解文全体から正しいSPM ID列を先に作ります。Whisperは文字ごとの時刻だけを推定し、元文と照合して各SPMトークンへ時刻をまとめます。学習ローダでは再トークナイズせず、保存済みIDをそのまま使います。")
def slide_11_spm_fix(c: Canvas, a: Assets):
    header(c, "SPM同期③ — 正しいトークン列へ、発音時刻だけを対応", 35)
    lead(c, "トークンと時刻を別々に作り、最後に統合する")

    c.square_heading(50, 250, "正解文から「何を」を作る", size=40, color=GREEN)
    c.rect(75, 330, 480, 90, fill=GREEN_FILL, outline=GREEN, width=3, radius=14)
    c.text(90, 348, 450, 55, "これは犬です", size=42, bold=True, align="center", wrap=False)
    c.arrow(575, 375, 735, 375, color=GREEN, width=6, head=20)
    c.text(570, 315, 170, 45, "全文をSPM", size=29, bold=True, color=GREEN, align="center", wrap=False)
    token_x = 770
    for txt, ww in [("▁これは", 205), ("犬", 95), ("です", 140)]:
        c.token(token_x, 345, ww, txt, size=34)
        token_x += ww + 18
    c.text(1280, 342, 500, 60, "ID: 444 / 1651 / 2767", size=33, bold=True, color=GREEN, wrap=False)
    c.text(1280, 405, 520, 65, "語彙外表記はSPM前に読みへ正規化\nJR → じぇいあーる", size=26, bold=True, color=GREEN, wrap=False)

    c.square_heading(50, 500, "合成音声から「いつ」を作る", size=40, color=BLUE)
    c.waveform(85, 585, 470, 80, color=BLUE, strokes=36)
    c.arrow(585, 625, 750, 625, color=BLUE, width=6, head=20)
    c.text(585, 565, 175, 45, "Whisper", size=31, bold=True, color=BLUE, align="center", wrap=False)
    c.rect(785, 575, 480, 100, fill=BLUE_FILL, outline=BLUE, width=3, radius=14)
    c.text(800, 590, 450, 70, "こ 0.40　れ 0.48　は 0.56\n犬 0.68　で 0.83　す 1.02", size=30, bold=True, color=BLUE, align="center", wrap=False)
    c.arrow(1295, 625, 1440, 625, color=BLUE, width=6, head=20)
    c.text(1300, 565, 135, 45, "原文と照合", size=28, bold=True, color=BLUE, align="center", wrap=False)
    c.rect(1470, 565, 380, 120, fill=WHITE, outline=BLUE, width=3, radius=14)
    c.text(1485, 580, 350, 88, "一致しない文字の時刻は\n前後から補間", size=29, bold=True, align="center", wrap=False)

    c.arrow(960, 715, 960, 770, color=CHARCOAL, width=7, head=22)
    c.rect(195, 790, 1530, 115, fill=ORANGE_FILL, outline=ORANGE, width=3, radius=16)
    c.text(225, 810, 320, 55, "時刻つき正解トークン", size=34, bold=True, color="#8A5A00", valign="middle", wrap=False)
    c.token(565, 818, 300, "▁これは  0.40-0.67", size=29, fill=WHITE, outline=ORANGE)
    c.token(885, 818, 245, "犬  0.68-0.82", size=29, fill=WHITE, outline=ORANGE)
    c.token(1150, 818, 290, "です  0.83-1.10", size=29, fill=WHITE, outline=ORANGE)
    c.text(1455, 815, 240, 60, "IDをそのまま\nローダへ", size=29, bold=True, color=RED, align="center", wrap=False)

    conclusion(c, "Whisperの認識結果は時刻合わせだけ。学習IDは元文全体から固定", y=950)


@slide("もう一つ、日本語では短いトークンが同じ80ミリ秒枠へ複数入る問題がありました。旧方式は先のトークンを捨てていたため文章が欠けます。keep and shiftでは順序を保ったまま次の空き枠へ送り、復元率を99.97パーセントまで改善しました。シフトなしの0.00パーセントは最終全件ログの値で、引き継ぎ資料中の0.61パーセントは以前の集計です。")
def slide_12_keep_shift(c: Canvas, a: Assets):
    header(c, "SPM同期④ — 80ms枠で衝突したトークンを捨てない", 36)
    lead(c, "1フレームには1トークンだけ。重なったら、順序を保って次の空き枠へ送る")

    c.text(75, 260, 310, 55, "発音された順番", size=36, bold=True, wrap=False)
    c.token(410, 255, 115, "は", size=38, fill=GREEN_FILL, outline=GREEN)
    c.token(510, 255, 115, "、", size=38, fill=ORANGE_FILL, outline=ORANGE)
    c.text(650, 268, 320, 45, "ほぼ同じ時刻", size=31, bold=True, color=GRAY, wrap=False)

    frame_x, frame_w = 420, 195
    for i in range(6):
        x = frame_x + i * frame_w
        c.rect(x, 375, frame_w, 80, fill=LIGHT_GRAY, outline=CHARCOAL, width=2)
        c.text(x, 320, frame_w, 45, f"{i * 80}-{(i + 1) * 80}ms", size=26, color=GRAY, align="center", wrap=False)
    c.text(65, 390, 300, 55, "旧方式", size=38, bold=True, color=RED, wrap=False)
    c.text(frame_x, 385, frame_w, 55, "、", size=40, bold=True, color=BLACK, align="center", wrap=False)
    c.text(1615, 385, 230, 55, "「は」が消える", size=34, bold=True, color=RED, wrap=False)

    for i in range(6):
        x = frame_x + i * frame_w
        c.rect(x, 555, frame_w, 80, fill=WHITE, outline=GREEN, width=3)
        c.text(x, 500, frame_w, 45, f"{i * 80}-{(i + 1) * 80}ms", size=26, color=GRAY, align="center", wrap=False)
    c.text(65, 570, 300, 55, "改修後", size=38, bold=True, color=GREEN, wrap=False)
    c.text(frame_x, 565, frame_w, 55, "は", size=40, bold=True, color=GREEN, align="center", wrap=False)
    c.text(frame_x + frame_w, 565, frame_w, 55, "、", size=40, bold=True, color=GREEN, align="center", wrap=False)
    c.arrow(frame_x + 120, 525, frame_x + frame_w + 70, 525, color=GREEN, width=5, head=16)
    c.text(1600, 550, 245, 90, "次の空き枠へ\n順送り", size=33, bold=True, color=GREEN, align="center", wrap=False)

    c.line(55, 700, 1860, 700, color=MID_GRAY, width=3)
    c.square_heading(55, 740, "全件で確認した結果", size=40)
    c.text(100, 815, 520, 90, "SPM ID一致\n19,771,615 / 19,771,615", size=33, bold=True, color=BLUE, align="center", wrap=False)
    c.text(700, 815, 520, 90, "ローダ復元\n170,087 / 170,141（99.97%）", size=33, bold=True, color=GREEN, align="center", wrap=False)
    c.text(1300, 815, 520, 90, "追加遅延\n平均0.012秒 / p95 0.080秒", size=33, bold=True, color=ORANGE, align="center", wrap=False)
    c.text(455, 910, 1010, 40, "シフトなし（最終全件ログ）：復元成功 1 / 170,141件（0.00%）", size=27, bold=True, color=RED, align="center", wrap=False)
    conclusion(c, "keep_and_shift=Trueで、トークンを捨てず順序も維持", y=958, size=38)


@slide("生成した170,141件は、すべて実際の学習ローダまで通して検査しました。除外条件は平均CERが0.30を超える120件だけです。残るSPM一致、時刻、復元は検証結果であり、追加の除外条件ではありません。最終的にtrain 166,621件、eval 3,400件へ分割しました。")
def slide_13_validation(c: Canvas, a: Assets):
    header(c, "全件検査後、train / evalへ分割", 37)
    lead(c, "170,141件すべてを実ローダまで確認し、CER不良120件だけを除外")

    c.rect(60, 275, 300, 150, fill=BLUE_FILL, outline=BLUE, width=3, radius=18)
    c.text(75, 292, 270, 58, "生成済み", size=38, bold=True, color=BLUE, align="center", wrap=False)
    c.text(75, 350, 270, 60, "170,141件", size=48, bold=True, align="center", wrap=False)
    c.arrow(380, 350, 565, 350, color=CHARCOAL, width=7, head=24)

    c.rect(590, 255, 330, 190, fill=PALE_RED, outline=RED, width=4, radius=18)
    c.text(605, 275, 300, 55, "CER品質ゲート", size=38, bold=True, color=RED, align="center", wrap=False)
    c.text(605, 335, 300, 45, "mean CER > 0.30", size=31, bold=True, align="center", wrap=False)
    c.text(605, 385, 300, 45, "120件を除外", size=34, bold=True, color=RED, align="center", wrap=False)
    c.arrow(940, 350, 1125, 350, color=CHARCOAL, width=7, head=24)

    c.rect(1150, 275, 300, 150, fill=GREEN_FILL, outline=GREEN, width=3, radius=18)
    c.text(1165, 292, 270, 58, "採用", size=38, bold=True, color=GREEN, align="center", wrap=False)
    c.text(1165, 350, 270, 60, "170,021件", size=48, bold=True, align="center", wrap=False)
    c.line(1470, 350, 1540, 350, color=CHARCOAL, width=6)
    c.line(1540, 310, 1540, 540, color=CHARCOAL, width=6)
    c.arrow(1540, 310, 1590, 310, color=CHARCOAL, width=6, head=18)
    c.arrow(1540, 540, 1590, 540, color=CHARCOAL, width=6, head=18)

    train_folder(c, 1605, 245, 245, 130, "train", BLUE_FILL, BLUE)
    c.text(1605, 380, 245, 75, "166,621件\n2,464.9時間", size=29, bold=True, align="center", wrap=False)
    train_folder(c, 1605, 475, 245, 130, "eval", GREEN_FILL, GREEN)
    c.text(1605, 610, 245, 75, "3,400件\n50.2時間", size=29, bold=True, align="center", wrap=False)

    c.square_heading(55, 515, "除外には使わない、全170,141件の検証値", size=38, color=BLUE)
    checks = [
        (80, 590, "必要ファイル", "欠損 0件"),
        (560, 590, "SPM ID", "1,046,458 / 1,046,458 発話"),
        (1040, 590, "トークン長", "170,141 / 170,141 対話"),
        (80, 770, "時刻順序", "170,096 / 170,141 対話"),
        (560, 770, "実ローダ復元", "170,087 / 170,141（99.97%）"),
        (1040, 770, "語彙外", "<unk> なし"),
    ]
    for x, y, name, value in checks:
        c.ellipse(x, y + 5, 48, 48, fill=BLUE, outline=None)
        c.text(x, y + 4, 48, 45, "i", size=32, bold=True, color=WHITE, align="center", valign="middle", wrap=False)
        c.text(x + 65, y, 380, 44, name, size=32, bold=True, wrap=False)
        c.text(x + 65, y + 48, 385, 65, value, size=25, color=GRAY, wrap=False)
    c.text(80, 925, 1120, 38, "※ SPM・時刻・ローダ復元は検証値。CERのような除外ゲートではない", size=27, color=GRAY, wrap=False)
    conclusion(c, "最終データ：170,021件・2,515.1時間", y=962, size=38)


@slide("再構築によって次の課題も明確になりました。現在の17万対話はすべて画像に関係するため、話題転換に弱い状態です。また、合成された発話は整いすぎてフィラーが少なくなっています。非画像会話を前後へ連結するp concatで改善する計画です。")
def slide_14_remaining_data(c: Canvas, a: Assets):
    header(c, "再構築後に見えた、次のデータ課題", 38)
    lead(c, "品質検査で終わりではなく、会話の分布そのものを次に改善する")

    c.square_heading(55, 260, "課題1　画像の話題に固着", size=42, color=BLUE)
    c.text(110, 345, 700, 85, "170,141 / 170,141", size=64, bold=True, color=BLUE, align="center", wrap=False)
    c.text(110, 435, 700, 55, "すべて画像関連の対話", size=38, bold=True, align="center", wrap=False)
    c.bullet(100, 520, 750, "画像と無関係な質問でも、画像の話を続けやすい", size=32, bullet_color=BLUE)
    c.bullet(100, 590, 750, "本家MoshiVisの非画像会話連結 p_concat が未実装", size=32, bullet_color=BLUE)

    c.line(955, 250, 955, 690, color=MID_GRAY, width=3)
    c.square_heading(1020, 260, "課題2　フィラー・相槌が少ない", size=42, color=ORANGE)
    c.text(1080, 345, 700, 85, "1.48%", size=70, bold=True, color=ORANGE, align="center", wrap=False)
    c.text(1080, 435, 700, 55, "フィラーを含むAI発話", size=38, bold=True, align="center", wrap=False)
    c.bullet(1065, 520, 750, "「えっと / えーと / ええと / んー」は調査対象で0件", size=32, bullet_color=ORANGE)
    c.bullet(1065, 590, 750, "LLMの整った書き言葉に引っ張られる", size=32, bullet_color=ORANGE)

    c.square_heading(55, 720, "対策　非画像会話を前後へ連結", size=40, color=GREEN)
    blocks = [
        (105, "非画像会話", "雑談・一般知識", GREEN_FILL, GREEN),
        (695, "画像会話", "v3主データ", BLUE_FILL, BLUE),
        (1285, "非画像会話", "話題転換・フィラー", GREEN_FILL, GREEN),
    ]
    for i, (x, title, sub, fill, out) in enumerate(blocks):
        c.rect(x, 805, 470, 115, fill=fill, outline=out, width=3, radius=16)
        c.text(x + 10, 818, 450, 45, title, size=36, bold=True, color=out, align="center", wrap=False)
        c.text(x + 10, 868, 450, 38, sub, size=29, align="center", wrap=False)
        if i < 2:
            c.arrow(x + 485, 862, x + 570, 862, color=CHARCOAL, width=5, head=17)
    c.text(1285, 935, 470, 35, "非画像対話 8,787 / 12,000件 生成済み", size=26, bold=True, color=GREEN, align="center", wrap=False)
    conclusion(c, "p_concatで、画像以外へ自然に話題転換できる学習分布へ", y=976, size=35)


@slide("データ以外にも、学習基盤と推論経路を改修しました。学習ではDDPや学習率、保存間隔、最大ステップを設定化しました。ローダは保存済みSPM IDを直接受け取ります。推論では画像のクロスアテンション情報を事前計算し、DepformerをCUDA Graph化して、1フレームを90.9ミリ秒から52.1ミリ秒へ短縮しました。")
def slide_15_other_improvements(c: Canvas, a: Assets):
    header(c, "データ以外の主要改修", 39)
    lead(c, "学習の再現性と、80ms以内のリアルタイム推論を両立")

    c.picture(a.architecture, 40, 260, 1080, 490, fit="contain")
    c.text(110, 760, 940, 38, "既存J-MoshiVisアーキテクチャ（元資料の図を再利用）", size=27, color=GRAY, align="center", wrap=False)

    c.square_heading(1165, 255, "学習基盤", size=39, color=BLUE)
    yy = 315
    for item in ("DDPのGPU割当てを修正", "実効バッチ・LR・保存間隔をconfig化", "warmup → cosine減衰 / max_steps対応"):
        yy = c.bullet(1190, yy, 650, item, size=30, bullet_color=BLUE, line_gap=5) + 8

    c.square_heading(1165, 525, "データローダ", size=39, color=GREEN)
    yy = 585
    for item in ("pretokenized SPM IDを直接利用", "USERテキスト混入を防止", "keep_and_shift=True"):
        yy = c.bullet(1190, yy, 650, item, size=30, bullet_color=GREEN, line_gap=5) + 8

    c.line(55, 835, 1865, 835, color=MID_GRAY, width=3)
    c.square_heading(55, 870, "リアルタイム推論", size=39, color=ORANGE)
    c.text(420, 870, 350, 55, "90.9 ms", size=48, bold=True, color=RED, align="center", wrap=False)
    c.arrow(785, 900, 1000, 900, color=CHARCOAL, width=7, head=24)
    c.text(1030, 860, 440, 70, "52.1 ms / frame", size=54, bold=True, color=GREEN, align="center", wrap=False)
    c.rect(1505, 855, 315, 76, fill=GREEN_FILL, outline=GREEN, width=3, radius=12)
    c.text(1515, 866, 295, 52, "80ms予算内", size=38, bold=True, color=GREEN, align="center", wrap=False)
    conclusion(c, "画像K/Vの事前計算 + DepformerのCUDA Graph化", y=960, size=38)


@slide("画像対話コーパスv3と推論高速化は完了しています。現在のstep 17,000モデルは、データセット構築中に旧v1で挙動を確認した試作であり、論文の比較や結果には使用しません。論文実験は、v3のmanifestと評価条件を固定した後、すべての条件を新規に学習します。")
def slide_16_status(c: Canvas, a: Assets):
    header(c, "現在地と次のステップ", 40)
    lead(c, "画像対話v3は完成。論文実験は全条件をv3で新規学習")

    c.square_heading(55, 255, "完了・確認済み", size=42, color=GREEN)
    yy = 330
    for item in (
        "v3：170,021対話 / 2,515.1時間",
        "SPM ID一致 100% / ローダ復元 99.97%",
        "v1 step 17,000で挙動確認（論文から除外）",
        "推論 52.1ms / 80msフレーム",
    ):
        yy = c.bullet(90, yy, 800, item, size=33, bullet_color=GREEN, line_gap=6) + 12

    c.line(955, 245, 955, 655, color=MID_GRAY, width=3)
    c.square_heading(1020, 255, "未完了・注意点", size=42, color=RED)
    yy = 330
    for item in (
        "optimizer / schedulerを含む完全再開",
        "Audio / Text JAMMEval評価基盤",
        "v3の25 / 50 / 100% manifest固定",
        "視覚Adapter全初期化のsmoke test",
    ):
        yy = c.bullet(1055, yy, 790, item, size=33, bullet_color=RED, line_gap=6) + 12

    c.square_heading(55, 700, "推奨する実験順序", size=40)
    stages = [
        ("1", "再開機能", "中断テスト", RED),
        ("2", "評価基盤", "指標を固定", ORANGE),
        ("3", "v3固定", "subset作成", GREEN),
        ("4", "一要因比較", "8k step", BLUE),
        ("5", "full比較", "20k step", CHARCOAL),
    ]
    x = 65
    for i, (no, title, sub, color) in enumerate(stages):
        c.ellipse(x, 790, 62, 62, fill=color, outline=None)
        c.text(x, 794, 62, 48, no, size=35, bold=True, color=WHITE, align="center", valign="middle", wrap=False)
        c.text(x + 82, 785, 225, 48, title, size=34, bold=True, color=color, wrap=False)
        c.text(x + 82, 835, 225, 42, sub, size=29, color=GRAY, wrap=False)
        if i < 4:
            c.arrow(x + 285, 820, x + 345, 820, color=CHARCOAL, width=5, head=17)
        x += 355
    conclusion(c, "旧v1は予備確認のみ。論文の比較はv3から同一条件でやり直す", y=950, size=37)


def render_added_slides(out_dir: Path, fm: FontManager, assets: Assets) -> list[Path]:
    paths: list[Path] = []
    for index, builder in enumerate(SLIDE_BUILDERS, 1):
        canvas = ImageCanvas(fm)
        builder(canvas, assets)
        path = out_dir / f"added_{index:02d}.png"
        canvas.image.save(path, "PNG", optimize=True)
        paths.append(path)
    return paths


def _source_slide_signatures(path: Path) -> list[str]:
    prs = Presentation(path)
    signatures = []
    for slide_obj in prs.slides:
        bits = []
        for shape in slide_obj.shapes:
            if getattr(shape, "has_text_frame", False):
                bits.append(shape.text)
        signatures.append("\n".join(bits))
    return signatures


def build_powerpoint(fm: FontManager, assets: Assets):
    prs = Presentation(SOURCE_PPTX)
    if len(prs.slides) != 36:
        raise RuntimeError(f"Expected 36 source slides, found {len(prs.slides)}")
    if (prs.slide_width, prs.slide_height) != (12192000, 6858000):
        raise RuntimeError("Unexpected source slide size")

    # The inserted section consumes page 25.  Keep the source deck's numbering
    # convention (section dividers count but do not display a number) by moving
    # the original post-insertion content forward by sixteen.
    for slide_obj in list(prs.slides)[26:]:
        for shape in slide_obj.shapes:
            if not getattr(shape, "has_text_frame", False):
                continue
            raw = shape.text.strip()
            if not raw.isdigit() or shape.top < Inches(6.9) or shape.left < Inches(9.5):
                continue
            text_frame = shape.text_frame
            text_frame.clear()
            text_frame.margin_left = text_frame.margin_right = 0
            text_frame.margin_top = text_frame.margin_bottom = 0
            text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
            paragraph = text_frame.paragraphs[0]
            paragraph.alignment = PP_ALIGN.RIGHT
            run = paragraph.add_run()
            run.text = str(int(raw) + 16)
            PptCanvas._font_run(run, FONT_REG_NAME, 28, WHITE, False)

    for builder, note in zip(SLIDE_BUILDERS, NOTES):
        slide_obj = prs.slides.add_slide(prs.slide_layouts[6])
        builder(PptCanvas(fm, slide_obj), assets)
        slide_obj.notes_slide.notes_text_frame.text = note

    slide_ids = prs.slides._sldIdLst
    added = list(slide_ids)[-len(SLIDE_BUILDERS):]
    for element in added:
        slide_ids.remove(element)
    insert_at = 25
    for offset, element in enumerate(added):
        slide_ids.insert(insert_at + offset, element)
    prs.save(OUTPUT_PPTX)


def images_to_pdf(images: list[Path], output_path: Path, size=(959.76, 540.0)):
    doc = fitz.open()
    for image_path in images:
        page = doc.new_page(width=size[0], height=size[1])
        page.insert_image(page.rect, filename=str(image_path), keep_proportion=False)
    doc.save(output_path, deflate=True, garbage=4)
    doc.close()


def build_combined_pdf(images: list[Path]):
    source = fitz.open(SOURCE_PDF)
    combined = fitz.open()
    combined.insert_pdf(source, from_page=0, to_page=24)
    for image_path in images:
        page = combined.new_page(width=source[0].rect.width, height=source[0].rect.height)
        page.insert_image(page.rect, filename=str(image_path), keep_proportion=False)
    combined.insert_pdf(source, from_page=25, to_page=len(source) - 1)
    footer_color = (64 / 255, 64 / 255, 64 / 255)
    # Pages 43–52 are original pages 27–36.  Cover their old numerals and draw
    # the new continuous page number (42–51) in the same footer position.
    for page_index, number in zip(range(42, 52), range(42, 52)):
        page = combined[page_index]
        cover = fitz.Rect(page.rect.width - 82, page.rect.height - 28, page.rect.width, page.rect.height)
        page.draw_rect(cover, color=footer_color, fill=footer_color, overlay=True)
        page.insert_text(
            fitz.Point(page.rect.width - 29, page.rect.height - 8),
            str(number), fontsize=14, fontname="helv", color=(1, 1, 1), overlay=True,
        )
    combined.save(OUTPUT_PDF, deflate=True, garbage=4)
    combined.close()
    source.close()


def verify_original_media_preserved():
    def hashes(path: Path):
        result = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("ppt/media/"):
                    result[Path(name).name] = hashlib.sha256(archive.read(name)).hexdigest()
        return result
    source_hashes = hashes(SOURCE_PPTX)
    output_hashes = hashes(OUTPUT_PPTX)
    missing = sorted(set(source_hashes) - set(output_hashes))
    changed = sorted(name for name, digest in source_hashes.items() if output_hashes.get(name) != digest)
    if missing or changed:
        raise RuntimeError(f"Original media were not preserved: missing={missing}, changed={changed}")


def main():
    for required in (SOURCE_PPTX, SOURCE_PDF):
        if not required.exists():
            raise FileNotFoundError(required)
    if len(SLIDE_BUILDERS) != 16 or len(NOTES) != 16:
        raise RuntimeError(f"Expected 16 added slides, found {len(SLIDE_BUILDERS)}")

    with tempfile.TemporaryDirectory(prefix="jmoshivis_progress_revised_") as temp_name:
        temp = Path(temp_name)
        fm = FontManager(SOURCE_PDF, temp)
        assets = extract_assets(temp)
        images = render_added_slides(temp, fm, assets)
        build_powerpoint(fm, assets)
        images_to_pdf(images, ADDED_PDF)
        build_combined_pdf(images)
    verify_original_media_preserved()

    print(f"Wrote {OUTPUT_PPTX}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {ADDED_PDF}")


if __name__ == "__main__":
    main()
