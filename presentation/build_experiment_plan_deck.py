#!/usr/bin/env python3
"""Append the agreed v3 paper experiment plan to the revised progress deck.

The eight slides in this file are inserted immediately after the current-status
slide of ``build_progress_deck.py``.  The existing revised deck remains
untouched; this builder writes a separate PowerPoint, a matching combined PDF,
and a PDF containing only the newly added experiment-plan slides.

All new PowerPoint content uses native text, shapes, arrows and pictures.  A
matching raster pass is used for the PDFs so that the deliverables remain
visually identical even on a machine without Microsoft PowerPoint.
"""

from __future__ import annotations

import hashlib
import tempfile
import zipfile
from pathlib import Path

import fitz
from pptx import Presentation
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches

import build_progress_deck as base


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PPTX = base.OUTPUT_PPTX
SOURCE_PDF = base.OUTPUT_PDF
OUTPUT_PPTX = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_実験計画版_改訂版.pptx"
OUTPUT_PDF = ROOT / "2026_0423_経過報告_J-MoshiVis改修追記_実験計画版_改訂版.pdf"
ADDED_PDF = ROOT / "J-MoshiVis_論文化実験計画_追加スライド_改訂版.pdf"

W, H = base.W, base.H
WHITE = base.WHITE
BLACK = base.BLACK
GRAY = base.GRAY
CHARCOAL = base.CHARCOAL
MID_GRAY = base.MID_GRAY
LIGHT_GRAY = base.LIGHT_GRAY
BLUE = base.BLUE
BLUE_FILL = base.BLUE_FILL
GREEN = base.GREEN
GREEN_FILL = base.GREEN_FILL
ORANGE = base.ORANGE
ORANGE_FILL = base.ORANGE_FILL
RED = base.RED
PALE_RED = base.PALE_RED

Canvas = base.Canvas
ImageCanvas = base.ImageCanvas
PptCanvas = base.PptCanvas
FontManager = base.FontManager
Assets = base.Assets


SLIDE_BUILDERS = []
NOTES: list[str] = []


def slide(note: str):
    def decorator(fn):
        SLIDE_BUILDERS.append(fn)
        NOTES.append(note)
        return fn
    return decorator


def numbered_circle(c: Canvas, x: float, y: float, number: str, color: str):
    c.ellipse(x, y, 62, 62, fill=color, outline=None)
    c.text(x, y + 3, 62, 50, number, size=34, bold=True, color=WHITE,
           align="center", valign="middle", wrap=False)


def metric_row(c: Canvas, x: float, y: float, w: float, title: str,
               metric: str, color: str, fill: str):
    c.rect(x, y, w, 128, fill=fill, outline=color, width=3, radius=14)
    c.rect(x, y, 200, 128, fill=color, outline=None, radius=14)
    c.text(x + 12, y + 14, 176, 96, title, size=31, bold=True, color=WHITE,
           align="center", valign="middle")
    c.text(x + 225, y + 14, w - 245, 96, metric, size=28, bold=True,
           color=BLACK, valign="middle")


@slide(
    "ここからは、完成した画像対話コーパスv3だけを用いる論文化実験計画です。旧v1の"
    "step 17,000モデルはデータ構築中の挙動確認に限り、論文の比較や結果には含めません。"
)
def slide_01_section(c: Canvas, a: Assets):
    c.text(300, 390, 1320, 120, "v3を用いた論文化実験計画", size=72, bold=True,
           align="center", valign="middle", wrap=False)
    c.line(480, 535, 1440, 535, color=CHARCOAL, width=3)
    c.text(430, 575, 1060, 58, "旧v1 runは予備確認のみ　／　論文実験は全条件をv3で新規学習",
           size=31, color=GRAY, align="center", wrap=False)


@slide(
    "本家MoshiVisは、正しい画像に対して視覚課題を解けるかを主評価にしています。COCO、"
    "OCR-VQA、VQAv2をテキスト質問とTTS音声質問の両方で評価し、生成音声をASRせず、"
    "音声と同期して生成されるテキストトークンを直接採点します。加えて音声品質、話題の"
    "切替、推論速度を確認しています。J-MoshiVisもこの評価思想を踏襲します。"
)
def slide_02_original_protocol(c: Canvas, a: Assets):
    base.header(c, "MoshiVis本家に合わせた評価方針", 42)
    base.lead(c, "正しい画像への回答を主結果にし、AudioとTextを同じ問題で測る")

    c.picture(a.dog, 55, 275, 275, 195, fit="cover")
    c.rect(55, 275, 275, 195, fill=WHITE, outline=BLUE, width=3)
    c.picture(a.dog, 61, 281, 263, 183, fit="cover")
    c.text(65, 480, 255, 42, "画像", size=31, bold=True, color=BLUE,
           align="center", wrap=False)

    c.rect(390, 292, 360, 72, fill=BLUE_FILL, outline=BLUE, width=3, radius=12)
    c.text(405, 304, 330, 48, "Text質問", size=32, bold=True, color=BLUE,
           align="center", valign="middle", wrap=False)
    c.rect(390, 390, 360, 72, fill=ORANGE_FILL, outline=ORANGE, width=3, radius=12)
    c.waveform(420, 409, 150, 34, color=ORANGE)
    c.text(575, 402, 155, 45, "TTS音声", size=29, bold=True, color=ORANGE,
           align="center", valign="middle", wrap=False)
    c.arrow(335, 370, 380, 370, color=CHARCOAL, width=5, head=16)
    c.arrow(755, 370, 835, 370, color=CHARCOAL, width=5, head=16)

    c.picture(a.architecture, 845, 260, 430, 235, fit="contain")
    c.text(890, 495, 340, 45, "MoshiVis", size=34, bold=True, color=BLUE,
           align="center", wrap=False)
    c.arrow(1285, 370, 1375, 370, color=CHARCOAL, width=5, head=16)

    c.rect(1390, 285, 465, 170, fill=GREEN_FILL, outline=GREEN, width=3, radius=15)
    c.text(1420, 305, 405, 55, "同期生成テキスト", size=35, bold=True, color=GREEN,
           align="center", wrap=False)
    c.line(1435, 372, 1810, 372, color=GREEN, width=2)
    c.text(1420, 390, 405, 45, "Accuracy / CIDEr", size=31, bold=True,
           align="center", wrap=False)

    c.line(45, 590, 1875, 590, color=MID_GRAY, width=3)
    items = [
        (80, "視覚理解", "COCO / OCR-VQA / VQAv2\nAudio・Textの両方", BLUE),
        (690, "音声品質", "p_audioを変えて視覚性能と\nMOSNetを同時評価", GREEN),
        (1300, "対話・速度", "V→NV / NV→V切替\n長時間contextのframe時間", ORANGE),
    ]
    for x, title, body, color in items:
        c.square_heading(x, 640, title, size=39, color=color)
        c.text(x + 35, 705, 500, 110, body, size=31, bold=True,
               align="center", valign="middle")
    c.text(650, 875, 620, 34, "出典：MoshiVis, arXiv:2503.15633",
           size=24, color=GRAY, align="center", wrap=False)
    base.conclusion(c, "主評価は『正しい画像に、音声で正しく答えられるか』", y=945, size=39)


@slide(
    "論文の中心となるResearch Questionは四つです。音声付き学習の比率、v3のデータ量、"
    "日本語音声backbone、英語MoshiVisからの視覚Adapter転移を一要因ずつ比較します。"
    "SPM同期は本家仕様へ合わせた実装修正なので比較対象にせず、p concatとGate構成も"
    "全条件で固定します。"
)
def slide_03_research_questions(c: Canvas, a: Assets):
    base.header(c, "J-MoshiVisで答える4つのResearch Question", 43)
    base.lead(c, "モデルの目的に直結する4要因を、一度に1つだけ変える")

    rows = [
        (245, "1", "音声を何割混ぜるべきか", "視覚性能と音声品質の両立", BLUE,
         [("0%", 120), ("25%", 140), ("50%", 140), ("100%", 155)]),
        (405, "2", "データ量を増やす効果はあるか", "固定計算量での多様性効果", GREEN,
         [("25%", 155), ("50%", 155), ("100%", 175)]),
        (565, "3", "日本語backboneで差が出るか", "双方を20kまで学習", ORANGE,
         [("J-Moshi", 235), ("LLM-jp-Moshi", 300)]),
        (725, "4", "英語の視覚能力は日本語へ転移するか", "全Adapterを20kで比較", CHARCOAL,
         [("全体を継承", 245), ("全体を初期化", 270)]),
    ]
    fills = {BLUE: BLUE_FILL, GREEN: GREEN_FILL, ORANGE: ORANGE_FILL, CHARCOAL: LIGHT_GRAY}
    for y, no, title, sub, color, chips in rows:
        numbered_circle(c, 50, y + 12, no, color)
        c.text(145, y, 670, 55, title, size=37, bold=True, color=color, wrap=False)
        c.text(145, y + 58, 650, 42, sub, size=27, color=GRAY, wrap=False)
        chip_total = sum(w for _, w in chips) + 22 * (len(chips) - 1)
        xx = 1825 - chip_total
        for label, width in chips:
            c.rect(xx, y + 15, width, 76, fill=fills[color], outline=color,
                   width=3, radius=12)
            c.text(xx + 8, y + 28, width - 16, 48, label, size=29, bold=True,
                   color=color, align="center", valign="middle", wrap=False)
            xx += width + 22
        c.line(45, y + 125, 1875, y + 125, color=MID_GRAY, width=2)

    c.text(235, 885, 1450, 38,
           "共通固定：v3のみ / seed・最適化・評価質問・Gate構成を統一 / p_concatは今回変更しない",
           size=27, color=GRAY, align="center", wrap=False)
    base.conclusion(c, "SPM同期は本家仕様で固定し、4つの主実験に集中", y=945, size=38)


@slide(
    "主評価にはJAMMEvalの単一画像・自然画像系4セットを使います。同じ日本語質問をTextと"
    "TTS音声で入力し、モデルが音声と同期して生成するテキストトークンを公式形式で採点します。"
    "短答はsoft exact accuracy、選択式はexact matchを使い、データセット別スコアとmacro平均を"
    "報告します。最終モデルだけ人手対話評価、文脈切替、音声品質、速度を追加します。"
)
def slide_04_jammeval(c: Canvas, a: Assets):
    base.header(c, "日本語音声画像理解を、どう測るか", 44)
    base.lead(c, "同一問題をAudio / Textで評価し、生成テキストを直接採点")

    c.picture(a.dog, 45, 270, 240, 170, fit="cover")
    c.rect(45, 270, 240, 170, fill=WHITE, outline=BLUE, width=3)
    c.picture(a.dog, 51, 276, 228, 158, fit="cover")
    c.rect(330, 245, 360, 82, fill=BLUE_FILL, outline=BLUE, width=3, radius=12)
    c.text(345, 260, 330, 52, "日本語Text質問", size=31, bold=True, color=BLUE,
           align="center", valign="middle", wrap=False)
    c.rect(330, 365, 360, 82, fill=ORANGE_FILL, outline=ORANGE, width=3, radius=12)
    c.waveform(350, 389, 135, 35, color=ORANGE)
    c.text(505, 379, 165, 50, "同文TTS", size=29, bold=True, color=ORANGE,
           align="center", valign="middle", wrap=False)
    c.arrow(700, 345, 780, 345, color=CHARCOAL, width=5, head=16)
    c.picture(a.architecture, 790, 240, 390, 215, fit="contain")
    c.arrow(1190, 345, 1280, 345, color=CHARCOAL, width=5, head=16)
    c.rect(1295, 255, 305, 180, fill=GREEN_FILL, outline=GREEN, width=3, radius=14)
    c.text(1320, 278, 255, 58, "生成Text token", size=31, bold=True, color=GREEN,
           align="center", valign="middle", wrap=False)
    c.text(1320, 352, 255, 52, "ASRなし", size=30, bold=True, color=RED,
           align="center", valign="middle", wrap=False)
    c.arrow(1610, 345, 1680, 345, color=CHARCOAL, width=5, head=16)
    c.rect(1695, 275, 180, 140, fill=LIGHT_GRAY, outline=CHARCOAL, width=3, radius=14)
    c.text(1710, 294, 150, 98, "公式採点\nAccuracy", size=29, bold=True,
           align="center", valign="middle")

    c.line(45, 520, 1875, 520, color=MID_GRAY, width=3)
    c.square_heading(55, 570, "主結果：JAMMEval 482問", size=40, color=BLUE)
    bench = [
        ("CC-OCR-JA", "145問", "日本語OCR"),
        ("JA-VLM-Bench", "49問", "日本文化・一般画像"),
        ("Heron-Bench", "88問", "日本文化・風景"),
        ("CVQA-JA", "200問", "日本文化知識"),
    ]
    yy = 640
    for name, count, desc in bench:
        c.text(85, yy, 360, 42, name, size=29, bold=True, color=BLUE, wrap=False)
        c.text(450, yy, 110, 42, count, size=28, bold=True, align="right", wrap=False)
        c.text(600, yy, 350, 42, desc, size=27, color=GRAY, wrap=False)
        yy += 58

    c.square_heading(1030, 570, "最終候補だけ追加", size=40, color=GREEN)
    extras = [
        "人手ブラインド評価：画像忠実性・日本語・音声",
        "V→NV / NV→V：画像と一般話題の切替",
        "UTMOS・ASR-CER：音声品質と発音の補助評価",
        "frame p50 / p95 / p99・80ms超過率",
    ]
    yy = 642
    for item in extras:
        yy = c.bullet(1065, yy, 760, item, size=28, bullet_color=GREEN, line_gap=3) + 8

    c.text(85, 885, 1750, 34,
           "v3はCOCOを含むためCOCO/VQAv2を主testにせず、JAMMEval画像との重複も事前監査",
           size=26, color=GRAY, align="center", wrap=False)
    base.conclusion(c, "内容評価にASR誤りを混ぜず、AudioとTextの差を直接測る", y=945, size=38)


@slide(
    "共通基準Rは、LLM-jp-Moshi、v3 100%、speech ratio 25%、英語MoshiVisの視覚Adapter"
    "全体継承です。ここから一要因だけを変えるため、四つの比較に必要な固有runは8本です。"
    "8k比較では停止stepとschedulerの総stepを分離し、すべて同じ20k cosine曲線上のstep 8k"
    "を比較します。"
)
def slide_05_run_matrix(c: Canvas, a: Assets):
    base.header(c, "共通基準から一要因ずつ変える — 固有8 run", 45)
    base.lead(c, "基準Rを共有し、4要因を総当たりにしない")

    c.rect(120, 225, 1680, 82, fill=GREEN_FILL, outline=GREEN, width=3, radius=13)
    c.text(145, 239, 1630, 52,
           "基準 R　LLM-jp-Moshi　／　v3 100%　／　speech_ratio 25%　／　全Adapter継承",
           size=32, bold=True, color=GREEN, align="center", valign="middle", wrap=False)

    x0, y0 = 45, 335
    widths = [105, 285, 175, 215, 300, 750]
    headers = ["Run", "Backbone", "v3量", "speech_ratio", "Adapter", "比較の目的"]
    xs = [x0]
    for width in widths[:-1]:
        xs.append(xs[-1] + width)
    for x, width, label in zip(xs, widths, headers):
        c.rect(x, y0, width, 52, fill=CHARCOAL, outline=WHITE, width=2)
        c.text(x + 5, y0 + 5, width - 10, 42, label, size=24, bold=True,
               color=WHITE, align="center", valign="middle", wrap=False)

    rows = [
        ("R", "LLM-jp-Moshi", "100%", "25%", "継承", "共通基準・20k", None),
        ("S0", "LLM-jp-Moshi", "100%", "0%", "継承", "RQ1：音声なし", 3),
        ("S50", "LLM-jp-Moshi", "100%", "50%", "継承", "RQ1：混合比", 3),
        ("S100", "LLM-jp-Moshi", "100%", "100%", "継承", "RQ1：全て音声", 3),
        ("D25", "LLM-jp-Moshi", "25%", "25%", "継承", "RQ2：データ量", 2),
        ("D50", "LLM-jp-Moshi", "50%", "25%", "継承", "RQ2：データ量", 2),
        ("BJ", "J-Moshi", "100%", "25%", "継承", "RQ3：Backbone・20k", 1),
        ("IR", "LLM-jp-Moshi", "100%", "25%", "全初期化", "RQ4：言語横断転移・20k", 4),
    ]
    highlights = {1: (ORANGE_FILL, ORANGE), 2: (GREEN_FILL, GREEN),
                  3: (BLUE_FILL, BLUE), 4: (LIGHT_GRAY, CHARCOAL)}
    row_h = 58
    for row_index, row in enumerate(rows):
        yy = y0 + 52 + row_index * row_h
        values, changed = row[:6], row[6]
        for col_index, (x, width, value) in enumerate(zip(xs, widths, values)):
            fill, outline = WHITE, MID_GRAY
            if row_index == 0:
                fill, outline = GREEN_FILL, GREEN
            elif changed == col_index:
                fill, outline = highlights[col_index]
            c.rect(x, yy, width, row_h, fill=fill, outline=outline, width=2)
            c.text(x + 5, yy + 6, width - 10, 46, value, size=23,
                   bold=(row_index == 0 or changed == col_index or col_index == 0),
                   color=outline if changed == col_index else BLACK,
                   align="center", valign="middle", wrap=False)

    c.text(165, 870, 1590, 34,
           "8k比較は全条件で同じ20k schedulerを使用　／　内部evalはcheckpoint選択にのみ使用",
           size=26, color=GRAY, align="center", wrap=False)
    base.conclusion(c, "同じ基準Rを共有し、必要な学習を8条件まで圧縮", y=945, size=38)


@slide(
    "データ量比較は旧v1とv3を比べず、v3 trainだけから25、50、100パーセントの入れ子"
    "部分集合を作ります。画像sourceと対話生成器の比率を保ち、UID hashで再現可能に抽出します。"
    "全条件を8k step、実効batch 16に揃えるため、この実験が測るのは固定計算量下でのデータ"
    "多様性の効果です。eval 3,400対話は共通です。"
)
def slide_06_data_scale(c: Canvas, a: Assets):
    base.header(c, "v3だけで、データ量25 / 50 / 100%を比較", 46)
    base.lead(c, "同一母集団の入れ子subsetで、固定計算量下の多様性効果を測る")

    c.square_heading(55, 245, "v3全体", size=40, color=BLUE)
    c.text(85, 310, 420, 80, "170,021対話", size=59, bold=True, color=BLUE,
           align="center", wrap=False)
    c.text(85, 400, 420, 58, "2,515.1時間", size=45, bold=True,
           align="center", wrap=False)
    c.text(85, 480, 420, 82, "train 166,621対話\neval 3,400対話（固定）",
           size=30, color=GRAY, align="center", valign="middle")

    bar_x, bar_y, bar_w = 610, 275, 1120
    bars = [
        ("100%", 1.00, "166,621対話　2,464.9h", "0.77周", BLUE, BLUE_FILL),
        ("50%", 0.50, "83,310対話　約1,232.2h", "1.54周", GREEN, GREEN_FILL),
        ("25%", 0.25, "41,655対話　約615.6h", "3.07周", ORANGE, ORANGE_FILL),
    ]
    for index, (label, ratio, body, epochs, color, fill) in enumerate(bars):
        yy = bar_y + index * 145
        width = bar_w * ratio
        c.text(520, yy + 20, 85, 52, label, size=34, bold=True, color=color,
               align="right", wrap=False)
        c.rect(bar_x, yy, bar_w, 88, fill=WHITE, outline=color, width=3, radius=12)
        c.rect(bar_x, yy, width, 88, fill=fill, outline=None, radius=12)
        c.text(bar_x + 18, yy + 17, 820, 52, body,
               size=29, bold=True, color=color, valign="middle", wrap=False)
        c.text(bar_x + bar_w - 155, yy + 21, 130, 45, epochs, size=28,
               bold=True, color=GRAY, align="right", wrap=False)

    c.line(45, 725, 1875, 725, color=MID_GRAY, width=3)
    c.square_heading(55, 770, "再現可能な抽出規則", size=38, color=GREEN)
    rules = [
        "image source × generatorで層化",
        "UIDのSHA-256順で25% ⊂ 50% ⊂ 100%",
        "8k × batch 16＝128,000例",
    ]
    xx = 90
    for item in rules:
        c.rect(xx, 840, 535, 70, fill=WHITE, outline=GREEN, width=2, radius=10)
        c.text(xx + 12, 851, 511, 48, item, size=25, bold=True,
               align="center", valign="middle", wrap=False)
        xx += 585
    base.conclusion(c, "旧v1は混ぜず、v3内で『量だけ』を変える", y=945, size=39)


@slide(
    "言語横断転移では、凍結するSigLIP画像encoderと日本語音声backboneを両条件で同じにし、"
    "学習対象となる視覚Adapter約2.06億パラメータだけを比較します。継承条件は英語MoshiVis"
    "から画像projection、Cross-Attention、norm、Gateを読み込み、初期化条件ではすべてを"
    "fresh initします。長期学習前に100から500 stepで損失、勾配、応答を確認します。"
)
def slide_07_adapter_transfer(c: Canvas, a: Assets):
    base.header(c, "英語の視覚Adapterは、日本語へ転移するか", 47)
    base.lead(c, "凍結Backboneを揃え、学習対象206.2Mだけを継承／全初期化")

    c.rect(310, 225, 1300, 78, fill=LIGHT_GRAY, outline=CHARCOAL, width=3, radius=13)
    c.text(335, 238, 1250, 50,
           "共通・凍結　SigLIP画像encoder ＋ 日本語音声Backbone",
           size=32, bold=True, color=CHARCOAL, align="center", valign="middle", wrap=False)

    columns = [
        (55, "英語MoshiVisから継承", GREEN, GREEN_FILL,
        ["Image Projection", "Cross-Attention\n+ Norm", "Gate"],
         "既存の視覚接続を日本語へ転移"),
        (995, "すべて新規初期化", ORANGE, ORANGE_FILL,
        ["Image Projection", "Cross-Attention\n+ Norm", "Gate"],
         "構造だけ同じ状態から日本語で獲得"),
    ]
    for x, title, color, fill, modules, caption in columns:
        c.text(x, 350, 870, 52, title, size=41, bold=True, color=color,
               align="center", wrap=False)
        xx = x + 40
        for index, module in enumerate(modules):
            c.rect(xx, 445, 240, 115, fill=fill, outline=color, width=3, radius=13)
            module_size = 22 if module.startswith("Image") or module.startswith("Cross") else 25
            c.text(xx + 8, 463, 224, 78, module, size=module_size, bold=True, color=color,
                   align="center", valign="middle", wrap=("\n" in module))
            if index < len(modules) - 1:
                c.arrow(xx + 248, 502, xx + 292, 502, color=CHARCOAL, width=4, head=13)
            xx += 285
        c.text(x + 75, 590, 720, 48, caption, size=28, bold=True,
               align="center", wrap=False)

    c.line(45, 685, 1875, 685, color=MID_GRAY, width=3)
    c.square_heading(55, 735, "長期runへ進む前のsmoke test", size=39, color=RED)
    checks = [
        ("100–500 step", "lossが有限で低下"),
        ("全module", "非ゼロ・有限gradient"),
        ("推論", "非空のText / Audio応答"),
    ]
    xx = 90
    for title, body in checks:
        c.rect(xx, 810, 510, 95, fill=PALE_RED, outline=RED, width=2, radius=11)
        c.text(xx + 12, 821, 180, 68, title, size=27, bold=True, color=RED,
               align="center", valign="middle", wrap=False)
        c.line(xx + 205, 825, xx + 205, 890, color=RED, width=2)
        c.text(xx + 220, 821, 275, 68, body, size=25, bold=True,
               align="center", valign="middle", wrap=False)
        xx += 585
    base.conclusion(c, "収束速度だけでなく、20k後の最終性能まで比較する", y=945, size=38)


@slide(
    "実験開始前に、v3 manifest、完全checkpoint、評価音声、全Adapter初期化、schedulerを"
    "固定します。全8条件を同じ20k cosine曲線でstep 8,000まで比較し、Backbone二条件と"
    "Adapter二条件はstep 20,000まで継続します。合計は10万optimizer stepで、実測速度では"
    "2枚のA6000を約12.2日間占有します。最終候補だけ人手評価と詳細システム評価を行います。"
)
def slide_08_execution(c: Canvas, a: Assets):
    base.header(c, "実行順序と計算資源", 48)
    base.lead(c, "評価と再現性を固定してから、8条件を同じ学習曲線で比較")

    stages = [
        (45, 270, 390, "1", "実験基盤を固定", "v3 manifest・完全再開\nJAMMEval音声・LR固定", RED, PALE_RED),
        (505, 270, 350, "2", "smoke test", "全条件100 step\n全初期化のみ最大500", ORANGE, ORANGE_FILL),
        (925, 270, 385, "3", "8条件を8k", "同じ20k cosine曲線\n同じseed・評価条件", GREEN, GREEN_FILL),
        (1380, 270, 495, "4", "3条件を20kへ", "R / J-Moshi / 全初期化\n最終評価・必要時追加seed", BLUE, BLUE_FILL),
    ]
    for index, (x, y, w, no, title, body, color, fill) in enumerate(stages):
        c.rect(x, y, w, 245, fill=fill, outline=color, width=3, radius=16)
        numbered_circle(c, x + 18, y + 18, no, color)
        c.text(x + 95, y + 24, w - 110, 52, title, size=31, bold=True,
               color=color, valign="middle", wrap=False)
        c.line(x + 30, y + 98, x + w - 30, y + 98, color=color, width=2)
        c.text(x + 24, y + 118, w - 48, 100, body, size=27, bold=True,
               align="center", valign="middle")
        if index < len(stages) - 1:
            c.arrow(x + w + 8, y + 122, stages[index + 1][0] - 12, y + 122,
                    color=CHARCOAL, width=5, head=17)

    c.square_heading(55, 600, "必要計算量（評価・共有待ちを除く）", size=40, color=CHARCOAL)
    metrics = [
        (95, "8条件", "すべてstep 8kまで"),
        (540, "100k step", "3条件だけ20kへ継続"),
        (985, "約12.2日", "2×A6000の壁時計"),
        (1430, "24.3 GPU日", "A6000消費量"),
    ]
    for x, title, body in metrics:
        c.text(x, 690, 350, 72, title, size=51, bold=True, color=BLUE,
               align="center", wrap=False)
        c.text(x, 770, 350, 48, body, size=27, bold=True, color=GRAY,
               align="center", wrap=False)

    c.rect(120, 850, 1680, 65, fill=LIGHT_GRAY, outline=CHARCOAL, width=2, radius=10)
    c.text(145, 861, 1630, 44,
           "差が小さい比較だけ追加seed　／　testは最終条件確定後に一度だけ実行　／　v1 runは集計外",
           size=27, bold=True, color=CHARCOAL, align="center", wrap=False)
    base.conclusion(c, "次に着手するのは、学習ではなく評価・manifest・再開機能の固定", y=945, size=38)


def render_added_slides(out_dir: Path, fm: FontManager, assets: Assets) -> list[Path]:
    paths: list[Path] = []
    for index, builder in enumerate(SLIDE_BUILDERS, 1):
        canvas = ImageCanvas(fm)
        builder(canvas, assets)
        path = out_dir / f"experiment_{index:02d}.png"
        canvas.image.save(path, "PNG", optimize=True)
        paths.append(path)
    return paths


def _renumber_trailing_slides(prs: Presentation, increment: int):
    # In the revised source deck, slide 42 is the original section divider;
    # numbered post-divider content begins at zero-based slide index 42.
    for slide_obj in list(prs.slides)[42:]:
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
            run.text = str(int(raw) + increment)
            PptCanvas._font_run(run, base.FONT_REG_NAME, 28, WHITE, False)


def build_powerpoint(fm: FontManager, assets: Assets):
    prs = Presentation(SOURCE_PPTX)
    if len(prs.slides) != 52:
        raise RuntimeError(f"Expected 52 revised-source slides, found {len(prs.slides)}")
    _renumber_trailing_slides(prs, len(SLIDE_BUILDERS))

    for builder, note in zip(SLIDE_BUILDERS, NOTES):
        slide_obj = prs.slides.add_slide(prs.slide_layouts[6])
        builder(PptCanvas(fm, slide_obj), assets)
        slide_obj.notes_slide.notes_text_frame.text = note

    slide_ids = prs.slides._sldIdLst
    added = list(slide_ids)[-len(SLIDE_BUILDERS):]
    for element in added:
        slide_ids.remove(element)
    insert_at = 41
    for offset, element in enumerate(added):
        slide_ids.insert(insert_at + offset, element)
    prs.save(OUTPUT_PPTX)


def build_combined_pdf(images: list[Path]):
    source = fitz.open(SOURCE_PDF)
    combined = fitz.open()
    combined.insert_pdf(source, from_page=0, to_page=40)
    for image_path in images:
        page = combined.new_page(width=source[0].rect.width, height=source[0].rect.height)
        page.insert_image(page.rect, filename=str(image_path), keep_proportion=False)
    combined.insert_pdf(source, from_page=41, to_page=len(source) - 1)

    footer_color = (64 / 255, 64 / 255, 64 / 255)
    # Page index 56 is the source deck's unnumbered "今後の展望" divider.
    # It counts in the sequence but deliberately keeps an empty footer.
    numbered_pages = list(range(50, 56)) + list(range(57, 60))
    for page_index, number in zip(numbered_pages, numbered_pages):
        page = combined[page_index]
        cover = fitz.Rect(page.rect.width - 82, page.rect.height - 28,
                          page.rect.width, page.rect.height)
        page.draw_rect(cover, color=footer_color, fill=footer_color, overlay=True)
        page.insert_text(
            fitz.Point(page.rect.width - 29, page.rect.height - 8),
            str(number), fontsize=14, fontname="helv", color=(1, 1, 1), overlay=True,
        )
    combined.save(OUTPUT_PDF, deflate=True, garbage=4)
    combined.close()
    source.close()


def verify_outputs():
    prs = Presentation(OUTPUT_PPTX)
    expected_slides = 52 + len(SLIDE_BUILDERS)
    if len(prs.slides) != expected_slides:
        raise RuntimeError(
            f"Expected {expected_slides} output slides, found {len(prs.slides)}"
        )
    pdf = fitz.open(OUTPUT_PDF)
    if len(pdf) != expected_slides:
        raise RuntimeError(
            f"Expected {expected_slides} output PDF pages, found {len(pdf)}"
        )
    pdf.close()

    # All media present in the original authored deck must remain byte-identical.
    def hashes(path: Path):
        result = {}
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if name.startswith("ppt/media/"):
                    result[Path(name).name] = hashlib.sha256(archive.read(name)).hexdigest()
        return result

    original = hashes(base.SOURCE_PPTX)
    output = hashes(OUTPUT_PPTX)
    missing = sorted(set(original) - set(output))
    changed = sorted(name for name, digest in original.items() if output.get(name) != digest)
    if missing or changed:
        raise RuntimeError(f"Original media changed: missing={missing}, changed={changed}")


def main():
    if not SOURCE_PPTX.exists() or not SOURCE_PDF.exists():
        base.main()
    if len(SLIDE_BUILDERS) != 8 or len(NOTES) != 8:
        raise RuntimeError(f"Expected 8 experiment slides, found {len(SLIDE_BUILDERS)}")

    with tempfile.TemporaryDirectory(prefix="jmoshivis_experiment_plan_") as temp_name:
        temp = Path(temp_name)
        fm = FontManager(base.SOURCE_PDF, temp)
        assets = base.extract_assets(temp)
        images = render_added_slides(temp, fm, assets)
        build_powerpoint(fm, assets)
        base.images_to_pdf(images, ADDED_PDF)
        build_combined_pdf(images)
    verify_outputs()

    print(f"Wrote {OUTPUT_PPTX}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {ADDED_PDF}")


if __name__ == "__main__":
    main()
