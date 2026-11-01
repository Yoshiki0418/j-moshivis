#!/usr/bin/env python3
"""Oct 9 research update; editable PPTX plus matching, raster-rendered PDF.

Original decks remain untouched. Source URLs and qualifications are in notes.
Run with /gpu-server/yamamoto/miniforge3/bin/python.
"""
from pathlib import Path
import hashlib
import tempfile
import zipfile

import fitz
from PIL import Image
from pptx import Presentation
import build_progress_deck as b

ROOT = b.ROOT
ASSET = ROOT / 'presentation/assets/speech_models_20261009/full_duplex.png'
PREVIEW = ROOT / 'presentation/previews/speech_models_20261009'
STEM = ROOT / '2026_1009_経過報告_音声対話モデル調査'
COMBINED = ROOT / '2026_0423_経過報告_J-MoshiVis_音声対話モデル調査追記'
SOURCE = ROOT / '2026_0423_経過報告_J-MoshiVis改修追記_実験計画版_改訂版'
SOURCES = {
    'O1': ('GPT-Live: Getting started', 'https://developers.openai.com/api/docs/guides/live'),
    'O2': ('Voice agents: Live / Realtime / chained', 'https://developers.openai.com/api/docs/guides/voice-agents'),
    'O3': ('GPT-Live-1 model', 'https://developers.openai.com/api/docs/models/gpt-live-1'),
    'O4': ('GPT-Realtime-2.1 model', 'https://developers.openai.com/api/docs/models/gpt-realtime-2.1'),
    'O5': ('Realtime conversations', 'https://developers.openai.com/api/docs/guides/realtime-conversations'),
    'L1': ('Moshi', 'https://github.com/kyutai-labs/moshi'),
    'L2': ('PersonaPlex', 'https://research.nvidia.com/labs/adlr/personaplex/'),
    'L3': ('J-Moshi', 'https://huggingface.co/nu-dialogue/j-moshi'),
    'L4': ('LLM-jp-Moshi', 'https://llm-jp.github.io/llm-jp-moshi/'),
    'V1': ('MoshiVis paper', 'https://arxiv.org/html/2503.15633v1'),
    'V2': ('Qwen3-Omni', 'https://github.com/QwenLM/Qwen3-Omni'),
    'V3': ('MiniCPM-o 4.5', 'https://huggingface.co/openbmb/MiniCPM-o-4_5'),
}


def source(c, keys, extra=''):
    names = ' / '.join(f'[{k}] {SOURCES[k][0]}' for k in keys)
    c.text(38, 895, 1840, 44, '出典：' + names + ('　' + extra if extra else ''), size=23, color=b.GRAY)


def box(c, x, y, w, h, title, body='', fill=b.BLUE_FILL, size=35):
    c.rect(x, y, w, h, fill=fill, outline=b.BLACK, width=3, radius=18)
    c.text(x+20, y+20, w-40, 60, title, size=40, bold=True, align='center')
    if body:
        c.text(x+24, y+96, w-48, h-112, body, size=size, align='center')


def table(c, columns, rows, widths, y=265, row_h=110, size=32):
    x0 = 40
    for ri, row in enumerate([columns] + rows):
        x = x0
        yy = y + ri*row_h
        for ci, (value, w) in enumerate(zip(row, widths)):
            fill = b.CHARCOAL if ri == 0 else (b.BLUE_FILL if ci == 0 else b.WHITE)
            c.rect(x, yy, w, row_h, fill=fill, outline=b.MID_GRAY, width=2)
            c.text(x+16, yy+16, w-32, row_h-25, value, size=size,
                   bold=(ri == 0 or ci == 0), color=b.WHITE if ri == 0 else b.BLACK,
                   valign='middle')
            x += w


def live(c, a, n):
    b.header(c, '音声対話モデルの動向：GPT-Live', n)
    b.lead(c, 'GPT-Live：聞きながら話せる Speech-to-Speech モデル')
    c.picture(a.user_icon, 90, 315, 235, 350)
    c.text(65, 675, 305, 85, 'スマホ等で利用\n（利用イメージ）', size=31, align='center')
    box(c, 455, 315, 610, 355, 'GPT-Live-1', '音声を入力し、音声で応答\n発話中もユーザーを聞く\n会話を続けながら処理を委譲', size=35)
    box(c, 1260, 315, 560, 355, 'バックエンドのエージェント', '検索・推論・ツール実行\n会話モデルとは別に選択\n結果を GPT-Live に返す', fill=b.GREEN_FILL, size=33)
    c.arrow(345, 420, 435, 420, color=b.BLUE)
    c.arrow(435, 540, 345, 540, color=b.ORANGE)
    c.arrow(1085, 420, 1240, 420, color=b.BLUE)
    c.arrow(1240, 540, 1085, 540, color=b.GREEN)
    c.text(1070, 365, 190, 50, '処理を依頼', size=27, align='center')
    c.text(1090, 560, 155, 50, '結果を返す', size=27, align='center')
    c.text(60, 805, 1770, 75, 'ブラウザ／モバイルでも利用可能なAPI。ChatGPTスマホアプリの内部モデルとの同一性は未確認。', size=29)
    source(c, ['O1', 'O3'])
    b.conclusion(c, 'Moshiと共通するのは「全二重で音声対話する性質」', size=38)


def duplex(c, a, n):
    b.header(c, '全二重（Full-duplex）とは？', n)
    b.lead(c, '自分が話している間も、相手の音声を受け取り続ける')
    c.picture(ASSET, 175, 230, 1570, 605)
    c.text(60, 845, 1790, 48, '相槌・発話の重なり・割り込みへの対応を支える。単に音声を少しずつ出力することとは異なる。', size=29)
    source(c, ['O1', 'V1'], '図：画像生成による概念図')
    b.conclusion(c, '「聞く時間」と「話す時間」を必ず分けなくてよい', size=39)


def compare(c, a, n):
    b.header(c, 'GPT-Live と Realtime API の比較', n)
    b.lead(c, '同じ音声APIでも、会話と推論・ツール利用の分担が異なる')
    table(c, ['比較項目', 'GPT-Live-1', 'Realtime API（GPT-Realtime-2.1）'], [
        ['基本構成', '会話モデル ＋ 別のバックエンド', '音声・判断・応答を１セッションで扱う'],
        ['会話中の動作', '発話中も聞く「全二重」を明記', '音声ストリーミング・割り込み制御を提供'],
        ['推論・ツール', '推論・ツール処理をバックエンドへ委譲', '音声モデルが判断し、ツール呼出しを要求'],
        ['視覚入力', '画像・動画は非対応', '画像入力に対応／動画入力は非対応'],
        ['公開されていない点', '内部のネットワーク構造・学習詳細', '内部構造やMoshi型全二重との同一性'],
    ], [325, 700, 815], y=245, row_h=101, size=30)
    c.text(55, 865, 1800, 45, '※「Realtimeは半二重」「GPT-Liveの内部はMoshiと同じ」とは、公式情報だけでは断定しない。', size=28)
    source(c, ['O2', 'O3', 'O4', 'O5'])
    b.conclusion(c, '会話を担うモデルと、難しい処理を担うモデルを分けられる', size=37)


def local(c, a, n):
    b.header(c, 'ローカルで利用できる音声対話モデル', n)
    b.lead(c, '公開モデルにも、音声を直接扱う全二重モデルがある')
    table(c, ['公開モデル', '音声対話の言語', '全二重', '特徴・研究での役割'], [
        ['Moshi', '英語', '対応', '音声・テキストを同期生成する基盤'],
        ['PersonaPlex', '英語を中心に報告', '対応', 'Moshi系。声・役割を条件付ける'],
        ['J-Moshi / J-Moshi-ext', '日本語', '対応', 'Moshiを日本語対話へ適応'],
        ['LLM-jp-Moshi', '日本語', '対応', '日本語の全二重音声対話基盤'],
    ], [515, 360, 200, 765], row_h=118, size=31)
    c.text(60, 865, 1800, 45, '「公開モデル」＝重み等を入手してローカル実行可能。モデルごとに利用条件は異なる。', size=29)
    source(c, ['L1', 'L2', 'L3', 'L4'])
    b.conclusion(c, '日本語の音声基盤から、視覚理解を追加できる', size=39)


def visual(c, a, n):
    b.header(c, '視覚も理解するローカル音声モデル', n)
    b.lead(c, '「視覚対応」「日本語音声」「全二重」は、別々に確認する')
    table(c, ['公開モデル', '視覚入力', '音声の言語', '会話方式'], [
        ['MoshiVis', '画像', '英語', '全二重を明記'],
        ['Qwen3-Omni\n30B-A3B-Instruct', '画像・動画', '入力19言語／出力10言語\n日本語は両方に含む', '音声出力をストリーミング\nMoshi型全二重は未確認'],
        ['MiniCPM-o 4.5', '画像・動画', 'リアルタイム音声対話は\n英語・中国語', '全二重モードを公開'],
    ], [505, 245, 575, 515], y=255, row_h=140, size=31)
    c.text(60, 850, 1800, 60, '※テキストでの多言語対応と、音声入出力での対応言語は同じとは限らない。', size=31)
    source(c, ['V1', 'V2', 'V3'])
    b.conclusion(c, 'J-MoshiVisは「日本語 × 視覚 × 全二重」を検証する研究', size=37)


def architecture(c, a, n):
    b.header(c, 'MoshiVis と Qwen3-Omni：設計の出発点', n)
    c.text(60, 150, 860, 65, 'MoshiVis：音声対話に視覚を追加', size=43, bold=True)
    c.text(1010, 150, 860, 65, 'Qwen3-Omni：マルチモーダル統合', size=42, bold=True)
    c.line(960, 230, 960, 855, color=b.MID_GRAY, width=3)
    box(c, 80, 245, 340, 130, '画像', fill=b.GREEN_FILL)
    box(c, 495, 245, 360, 130, '画像エンコーダ', fill=b.GREEN_FILL)
    c.arrow(435, 310, 480, 310, color=b.GREEN)
    c.arrow(675, 390, 675, 405, color=b.GREEN, head=10)
    c.text(460, 410, 450, 40, 'Gate付き Cross-Attention', size=29, align='center')
    c.arrow(675, 455, 675, 475, color=b.GREEN, head=10)
    box(c, 115, 485, 730, 185, 'Moshi 音声対話基盤', 'ユーザー音声を受けながら、発話を生成', size=32)
    c.arrow(480, 680, 480, 715)
    c.text(100, 735, 785, 75, '同期したテキスト ＋ 音声トークン', size=35, bold=True, align='center')
    c.text(120, 805, 760, 48, '既存基盤を凍結し、視覚アダプタを学習', size=29, align='center')
    box(c, 1040, 245, 785, 130, 'テキスト・音声・画像・動画', fill=b.GREEN_FILL)
    c.arrow(1440, 390, 1440, 450)
    box(c, 1040, 470, 785, 150, 'Thinker（理解・推論）', 'テキストと内部表現を生成', size=31)
    c.arrow(1440, 630, 1440, 685)
    box(c, 1040, 700, 785, 150, 'Talker（音声生成）', '音声トークンを生成し、音声へ復号', fill=b.ORANGE_FILL, size=31)
    source(c, ['V1', 'V2'], '主要な情報の流れのみを示した概念図')
    b.conclusion(c, 'GPT-Live / Realtime の内部構造を、この図から推測しない', size=37)


def position(c, a, n):
    b.header(c, 'J-MoshiVis の研究上の位置づけ', n)
    b.lead(c, '日本語の音声対話を基盤に、画像を見て自然に会話する')
    c.picture(a.user_icon, 70, 330, 220, 330)
    box(c, 410, 285, 540, 360, '日本語の音声基盤', 'J-Moshi\nまたは LLM-jp-Moshi\n\n日本語で聞く・話す', size=34)
    c.text(1010, 380, 70, 85, '＋', size=70, bold=True, align='center')
    box(c, 1140, 285, 670, 360, '視覚アダプタ ＋ 日本語データ', '画像内容・物の数・日本語文字\nについて日本語で対話\n\n本家MoshiVisの設計を出発点にする', fill=b.GREEN_FILL, size=33)
    c.text(80, 720, 1735, 130, '研究で確かめること\n視覚理解の精度 ／ 日本語音声の品質 ／ 全二重対話とリアルタイム性の維持', size=37, bold=True)
    source(c, ['L3', 'L4', 'V1'], 'J-MoshiVis：本プロジェクトの研究対象')
    b.conclusion(c, '競合がないと断定せず、言語・視覚・会話方式の条件を揃えて比較', size=34)


def future(c, a, n):
    b.header(c, '将来像：会話するモデルと考えるエージェント', n)
    b.lead(c, 'GPT-Liveの役割分担は、ローカル構成を考える参考になる')
    box(c, 100, 285, 725, 390, 'J-MoshiVis（会話・知覚）', '音声と画像を理解\n低遅延で応答・相槌\n追加の発話も受け取る', size=37)
    box(c, 1110, 285, 710, 390, 'テキストLLM（推論・行動）', '高度な推論・検索\nツールの選択と実行要求\n結果を会話モデルへ返す', fill=b.GREEN_FILL, size=37)
    c.arrow(850, 405, 1085, 405, color=b.BLUE)
    c.arrow(1085, 545, 850, 545, color=b.GREEN)
    c.text(840, 345, 250, 50, '構造化した依頼', size=28, align='center')
    c.text(850, 565, 235, 45, '処理結果', size=29, align='center')
    c.text(95, 775, 1725, 100, '将来の研究課題（未実装）\n委譲の判断・要求の構造化・結果の読み上げを学習し、権限確認はアプリ側で行う。', size=32)
    source(c, ['O1', 'O2'], 'OpenAIの公開された役割分担を参考にした構想')
    b.conclusion(c, 'まずは視覚音声対話を評価し、ツール利用は次の研究段階へ', size=36)


def references(c, a, n):
    b.header(c, '参照資料・読み取りの範囲', n)
    b.lead(c, '公式ドキュメント・原論文・公開モデルカードを優先')
    items = list(SOURCES.items())
    for i, (key, (name, url)) in enumerate(items):
        col, row = divmod(i, 6)
        x, y = 50+col*940, 260+row*90
        c.text(x, y, 900, 45, f'[{key}] {name}', size=30, bold=True)
        display = url.replace('https://', '')
        c.text(x, y+40, 900, 46, display, size=21, color=b.BLUE)
    c.text(60, 825, 1800, 95, '確認日：2026年10月9日。各スライドのノートにURLと説明を記載。\n内部構造の推測や、消費者向けChatGPTの採用モデルの未確認情報は結論に含めない。', size=29)
    b.conclusion(c, '「公式に確認できる能力」と「今後の構想」を区別する', size=38)


BUILDERS = [live, duplex, compare, local, visual, architecture, position, future, references]
NOTES = [
    'GPT-Liveはスマホ専用製品ではなく、モバイル等でも接続できる音声API。ChatGPTスマホアプリの内部モデルがGPT-Liveであるという一次情報は今回確認できなかった。Moshiと共通とするのは全二重という動作上の性質だけで、内部のTransformer構造等は同じと断定しない。バックエンドの実際のツール実行と権限確認はアプリ側。',
    '上段は交互発話の概念図であり、Realtime APIを半二重と分類する図ではない。下段は入出力音声が重なる全二重の概念図。波形は実測データではない。全二重は継続的な受聴と発話を可能にする性質で、良い割り込み対応や相槌の品質を自動的に保証しない。図は組み込み画像生成で制作。',
    '比較対象をGPT-Live-1とRealtime APIのGPT-Realtime-2.1に明示。Liveは会話をしながら別バックエンドへ委譲。Realtimeは単一音声モデルが音声を解釈し判断し応答する説明。ツール実行自体はどちらもアプリ側。Realtimeはストリーミングや割り込みのAPIを提供するが、内部をMoshi型と判定する根拠ではない。Liveは画像非対応、Realtimeは画像入力対応。いずれも動画非対応というモデルカードの範囲。',
    '代表例に限定した公開モデルの整理で、市場全体の網羅・シェアの主張ではない。PersonaPlexは英語Fisher等での学習・評価を中心に報告し、全言語の非対応を意味しない。J-MoshiはCC BY-NC 4.0で研究用途を想定。LLM-jp-MoshiはApache 2.0。公開モデルの「ローカル」は重み等が利用できる意味であり、任意GPUやスマホで実行できる保証ではない。',
    'Qwen3-Omniの日本語は音声入力・出力とも対象。公式は入力19／出力10言語と記載（列挙数に差があるため列挙は転記しない）。生成ストリーミングとMoshi式全二重は別概念。MiniCPM-o 4.5は全二重の視聴覚ストリーミングを公開し、リアルタイム音声対話は英語・中国語と記載。テキスト多言語理解から日本語音声対話の正式サポートを推定しない。',
    'MoshiVis原論文§3の視覚クロスアテンションを加える設計と、Qwen3-Omni公式Thinker–Talker設計の概略。細かなエンコーダ、Mimi、Depth Transformer、波形復号器、ThinkerからTalkerへの表現の違い等は省略。MoshiVis基盤／画像エンコーダを凍結した視覚適応という出発点と、統合マルチモーダルモデルの出発点を比較する。',
    'J-MoshiVisの研究目的の整理。日本語・画像・全二重の交差領域に焦点を当てる。「世界初」「同条件の競合なし」とは断定しない。現時点の検証結果や音声品質スコアはこの調査スライドでは追加計測せず。',
    '将来構想であり、J-MoshiVisに現在ツール委譲機能が実装されていることを意味しない。会話と高度な推論の分離は公式GPT-Liveの公開されたシステムの役割分担を参考にしたもの。構造化された要求の出力、処理中の会話、結果の取り込みの学習と検証、アプリの権限確認が別途必要。',
    '本資料の確認日2026-10-09。過去チャット内のOpenAI内部アーキテクチャの推測は根拠として採用しない。公式情報で確認できる範囲と未確認点を明示。',
]


def notes(index):
    return NOTES[index] + '\n\n一次資料（確認日2026-10-09）：\n' + '\n'.join(
        f'[{key}] {name}\n{url}' for key, (name, url) in SOURCES.items())


class CheckedCanvas(b.ImageCanvas):
    def text(self, x, y, w, h, value, size=40, bold=False, color=b.BLACK,
             align='left', valign='top', wrap=True, margin=0):
        lines = self.fm.wrap(value, size, bold, w-2*margin) if wrap else value.split('\n')
        if len(lines)*round(size*1.18) > h+8:
            raise ValueError(f'Text too tall: {value!r} ({len(lines)} lines, h={h})')
        if not wrap and any(self.fm.length(line, size, bold) > w+4 for line in lines):
            raise ValueError(f'Text too wide: {value!r}')
        return super().text(x, y, w, h, value, size, bold, color, align, valign, wrap, margin)


def build():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    if not ASSET.exists():
        raise FileNotFoundError('Generate the full-duplex asset first: ' + str(ASSET))
    with tempfile.TemporaryDirectory(prefix='speech-models-') as tmp:
        fm = b.FontManager(b.SOURCE_PDF, Path(tmp))
        assets = b.extract_assets(Path(tmp))
        standalone = Presentation()
        standalone.slide_width, standalone.slide_height = b.emu(b.W), b.emu(b.H)
        combined = Presentation(SOURCE.with_suffix('.pptx'))
        count = len(combined.slides)
        pngs = []
        for i, builder in enumerate(BUILDERS):
            canvas = CheckedCanvas(fm)
            builder(canvas, assets, i+1)
            path = PREVIEW / f'{i+1:02d}.png'
            canvas.image.save(path)
            pngs.append(path)
            for prs, offset in [(standalone, 0), (combined, count)]:
                slide = prs.slides.add_slide(prs.slide_layouts[6])
                builder(b.PptCanvas(fm, slide), assets, offset+i+1)
                slide.notes_slide.notes_text_frame.text = notes(i)
        standalone.save(STEM.with_suffix('.pptx'))
        combined.save(COMBINED.with_suffix('.pptx'))
        b.images_to_pdf(pngs, STEM.with_suffix('.pdf'))
        pdf = fitz.open(SOURCE.with_suffix('.pdf'))
        for i, builder in enumerate(BUILDERS):
            canvas = CheckedCanvas(fm)
            builder(canvas, assets, count+i+1)
            path = Path(tmp) / f'combined-{i}.png'
            canvas.image.save(path)
            page = pdf.new_page(width=pdf[0].rect.width, height=pdf[0].rect.height)
            page.insert_image(page.rect, filename=str(path), keep_proportion=False)
        pdf.save(COMBINED.with_suffix('.pdf'), deflate=True, garbage=4)
        pdf.close()
    # python-pptx may renumber ZIP parts of an already reordered deck.
    # Compare original slides by presentation order, not ZIP filename.
    before_prs = Presentation(SOURCE.with_suffix('.pptx'))
    after_prs = Presentation(COMBINED.with_suffix('.pptx'))
    for i, original in enumerate(before_prs.slides):
        assert original._element.xml == after_prs.slides[i]._element.xml, f'slide {i+1}'
    with zipfile.ZipFile(SOURCE.with_suffix('.pptx')) as before, zipfile.ZipFile(COMBINED.with_suffix('.pptx')) as after:
        old_media = {hashlib.sha256(before.read(name)).digest() for name in before.namelist() if name.startswith('ppt/media/')}
        new_media = {hashlib.sha256(after.read(name)).digest() for name in after.namelist() if name.startswith('ppt/media/')}
        assert old_media <= new_media
    assert len(Presentation(STEM.with_suffix('.pptx')).slides) == 9
    assert len(Presentation(COMBINED.with_suffix('.pptx')).slides) == count+9
    with fitz.open(COMBINED.with_suffix('.pdf')) as doc:
        assert len(doc) == count+9
    sheet = Image.new('RGB', (1440, 810), 'white')
    for i, path in enumerate(pngs):
        thumbnail = Image.open(path)
        thumbnail.thumbnail((480, 270))
        sheet.paste(thumbnail, ((i%3)*480, (i//3)*270))
    sheet.save(PREVIEW / 'contact_sheet.jpg')
    print(f'Created 9 slides; appended deck: {count+9} slides; source slides/media unchanged.')
    print(STEM.with_suffix('.pptx'))
    print(COMBINED.with_suffix('.pptx'))


if __name__ == '__main__':
    build()
