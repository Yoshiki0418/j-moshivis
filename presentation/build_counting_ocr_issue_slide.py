"""One editable slide, matching the user's dark header/white canvas style."""
import json
from pathlib import Path
from pptx import Presentation
import build_progress_deck as b

ROOT=b.ROOT
AUDIT=ROOT/'eval_results/v3_counting_ocr_audit_20261012/summary.json'
STEM=ROOT/'2026_1012_経過報告_課題3_カウントOCR'

def draw(c, data):
    s=data['statistics'];n=data['train_dialogues'];q=s['user_turns']
    b.header(c,'再構築後に見えた、データ課題',38)
    c.square_heading(48,157,'課題3　数えるタスク・OCRタスクで誤答',size=44,color=b.BLUE)
    c.text(48,235,1800,48,f'v3：{n:,}対話／ユーザー発話 {q:,}ターンを分析',size=34,bold=True)
    widths=[580,440,430,370];x0=48;y0=314;rowh=102
    rows=[['質問文から抽出した候補','含まれる対話数','全対話に対する割合','ユーザー発話に占める割合']]
    for label,key in [('個数を尋ねる質問','counting_direct'),('文字を読む質問','ocr_direct')]:
        rows.append([label,f'{s[key+"_dialogues"]:,}件',f'{s[key+"_dialogues"]/n*100:.2f}%',f'{s[key+"_turns"]/q*100:.2f}%'])
    for i,row in enumerate(rows):
        x=x0
        for j,(value,w) in enumerate(zip(row,widths)):
            c.rect(x,y0+i*rowh,w,rowh,fill=b.BLUE_FILL if i==0 else b.WHITE,outline=b.MID_GRAY,width=2)
            c.text(x+18,y0+i*rowh+26,w-36,65,value,size=(25 if j==3 else 30) if i==0 else 36,bold=True,color=b.BLUE if i==0 else b.BLACK,wrap=False)
            x+=w
    c.bullet(48,644,1810,'関連する対話は存在するが、正解注釈を活用した専用データは未導入',size=34,bold=True)
    c.bullet(48,697,1810,'カウント質問の約74%が概数表現。OCRには英語なども混在し、正確さは未検証',size=31)
    c.rect(48,768,1820,118,fill=b.GREEN_FILL,outline=b.GREEN,width=2,radius=10)
    c.text(72,782,1780,42,'改善：正解の個数・文字列を与えた自然な対話データを追加',size=36,bold=True,color=b.GREEN,wrap=False)
    c.text(72,835,1780,40,'カウント専用画像 ＋ OCR専用画像 ＋ 店名・看板・標識などの日本の日常画像',size=30,color=b.GREEN,wrap=False)
    c.text(48,903,1820,40,'※質問文の規則による候補抽出。全件の画像・回答の正誤を検証した件数ではない。',size=25,color=b.GRAY,wrap=False)
    b.conclusion(c,'「正確に数える・読む」学習例の不足を仮説として、データを補強',y=960,size=37)

def main():
    data=json.loads(AUDIT.read_text())
    assets=ROOT/'presentation/assets/counting_ocr_issue_20261012';assets.mkdir(parents=True,exist_ok=True)
    fm=b.FontManager(b.SOURCE_PDF,assets)
    prs=Presentation();prs.slide_width=b.emu(b.W);prs.slide_height=b.emu(b.H)
    slide=prs.slides.add_slide(prs.slide_layouts[6]);draw(b.PptCanvas(fm,slide),data)
    slide.notes_slide.notes_text_frame.text='集計出典：'+str(AUDIT)+'\n'+json.dumps(data,ensure_ascii=False,indent=2)+'\nカウント・OCRの学習例が皆無だったという意味ではない。原因は未確定。対話数と質問ターン数の分母を区別し、正解注釈の不足・課題別学習機会の少なさを仮説として示す。'
    prs.save(STEM.with_suffix('.pptx'))
    canvas=b.ImageCanvas(fm);draw(canvas,data);canvas.image.save(STEM.with_suffix('.png'))
    b.images_to_pdf([STEM.with_suffix('.png')],STEM.with_suffix('.pdf'))
    assert len(Presentation(STEM.with_suffix('.pptx')).slides)==1
    print(STEM)

if __name__=='__main__':main()
