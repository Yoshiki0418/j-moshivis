"""Actual annotated image example and generated validation diagram, one slide."""
import json
from pathlib import Path
from pptx import Presentation
import build_progress_deck as b

ROOT=b.ROOT
ASSETS=ROOT/'presentation/assets/counting_quality_20261013'
STEM=ROOT/'2026_1013_経過報告_カウント対話の品質管理'
IMAGE=ROOT/'data/images/counting/visual_genome/VG_100K/2361354.jpg'
DIAGRAM=ASSETS/'counting_validation_diagram.png'

def draw(c):
    b.header(c,'カウント対話の品質を支える仕組み',39)
    b.lead(c,'正解を与えて自然に生成し、数値の不一致を除外する')
    c.square_heading(44,247,'実例：道路に立つ人を数える',size=36,color=b.BLUE)
    c.picture(IMAGE,48,312,500,375)
    c.rect(48,707,500,160,fill=b.BLUE_FILL,outline=b.BLUE,width=2,radius=10)
    c.text(64,722,468,40,'画像に紐づく正解注釈',size=31,bold=True,color=b.BLUE)
    c.text(64,772,468,80,'対象：道路に立っている人\nanswer_count = 2',size=30,bold=True,wrap=False)
    c.picture(DIAGRAM,574,293,1302,610)
    c.text(582,854,1278,52,'採用した対話の冒頭：\nユーザー「道路に立っている人は何人いますか？」',size=25,color=b.BLUE,wrap=False)
    c.text(582,918,1278,38,'アシスタント「2人いますね。」 → 位置や服装の自然な対話へ続く',size=26,bold=True,color=b.GREEN,wrap=False)
    c.text(48,958,1810,48,'※照合するのは指定された数え上げ回答。注釈や後続の描写の正しさをすべて保証するものではない。',size=24,color=b.GRAY,wrap=False)

def main():
    ASSETS.mkdir(parents=True,exist_ok=True)
    fm=b.FontManager(b.SOURCE_PDF,ASSETS)
    prs=Presentation();prs.slide_width=b.emu(b.W);prs.slide_height=b.emu(b.H)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);draw(b.PptCanvas(fm,slide))
    slide.notes_slide.notes_text_frame.text='実例UID counting:flickr:10002608404。正解出典：TallyQA / imported_genome, source_question_id=30133591, question_en=How many people are standing on the road?, answer_count=2。対話出典 data/dialogues_counting/pilot_v2/qwen3vl_30b_counting_pilot5_v2.jsonl。画像内のバス乗客ではなく道路に立っている人を対象とする。ssvd/dialogue/prompt_counting.mdで正解を生成時に与え、task_profiles.pyのvalidate_task_dialogueで指定回答の数値列が[2]と一致することを検査。曖昧な数量表現や否定も不採用。図中の3人回答は説明用の不採用例で、実際の生成失敗ログではない。生成図はimagegenを使用。図中の検証は数え上げ回答についてのみで、画像の注釈・質問条件・後続描写の正確性まで自動保証しない。'
    prs.save(STEM.with_suffix('.pptx'))
    canvas=b.ImageCanvas(fm);draw(canvas);canvas.image.save(STEM.with_suffix('.png'))
    b.images_to_pdf([STEM.with_suffix('.png')],STEM.with_suffix('.pdf'))
    print(STEM)

if __name__=='__main__':main()
