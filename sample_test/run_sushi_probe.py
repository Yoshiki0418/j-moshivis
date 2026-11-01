"""Visually verified sushi train-member diagnostic, not a benchmark."""
import argparse
import hashlib
import html
import json
import os
from dataclasses import replace
from pathlib import Path
from run_image_audio_probe import ROOT, main as probe

OUT = ROOT/'eval_results/sushi_probe_20261010'
QUESTION = 'この画像に写っている食べ物は何ですか？'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['find', 'prepare', 'infer'])
    parser.add_argument('--indices', type=int, nargs='+')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.mode == 'find':
        from PIL import Image, ImageDraw
        run = json.loads((ROOT/'checkpoints/llmjp_v3_id0w1_run1/run.json').read_text())
        record = run['metadata']['datasets']['train']
        train = Path(record['path'])
        assert hashlib.sha256(train.read_bytes()).hexdigest() == record['sha256']
        rows, seen = [], set()
        for n, line in enumerate(train.open(), 1):
            row = json.loads(line)
            if '寿司' not in row.get('text', ''): continue
            image = Path(row['image'])
            if not image.is_file(): continue
            sha = hashlib.sha256(image.read_bytes()).hexdigest()
            if sha in seen: continue
            seen.add(sha)
            rows.append(dict(row, train_line=n, image_sha256=sha))
        (OUT/'candidates.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        sheet = Image.new('RGB', (1600,1200),'white')
        draw = ImageDraw.Draw(sheet)
        for i,r in enumerate(rows[:20]):
            im=Image.open(r['image']).convert('RGB'); im.thumbnail((390,205))
            x,y=(i%4)*400,(i//4)*240
            sheet.paste(im,(x+(400-im.width)//2,y))
            draw.text((x+10,y+210),str(i+1),fill='black')
        sheet.save(OUT/'candidates_contact.jpg')
        print('Unique candidate images:',len(rows))
        return
    from jmoshivis.evaluation.schema import EvalExample, read_manifest, write_manifest
    if args.mode == 'prepare':
        rows=json.loads((OUT/'candidates.json').read_text())
        selected=[rows[i-1] for i in args.indices]
        (OUT/'selected.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2))
        examples=[EvalExample(sample_id='sushi_'+r['uid'][:16],benchmark='sushi_train_member_diagnostic',
            split='diagnostic_train_member',domain='japan',task='vqa',image_path=r['image'],
            question=QUESTION,answers=['寿司'],source={'training_uid':r['uid'],'training_line':r['train_line'],
            'image_sha256':r['image_sha256']},metadata={'training_dialogue':r['text']}) for r in selected]
        write_manifest(OUT/'manifest.jsonl',examples)
        return
    runtime=None
    cases=[]
    page='<meta charset="utf-8"><h1>寿司の認識テスト：音声入力・greedy</h1><p>9月27日の20kモデル。学習画像を使った診断であり汎化評価ではありません。左＝ユーザー、右＝モデル。質問に答えの名前は含めません。</p>'
    master=OUT/'manifest_audio.jsonl'
    for i,e in enumerate(read_manifest(master,check_files=True),1):
        e=replace(e,image_path=str(e.resolve_image(master).resolve()),
                  audio_path=str(e.resolve_audio(master).resolve()))
        case=OUT/f'case_{i:02d}';case.mkdir(exist_ok=True)
        write_manifest(case/'manifest_audio.jsonl',[e])
        runtime=probe(['infer','--image',e.image_path,'--output-dir',str(case)],runtime=runtime)
        r=json.loads((case/'result.json').read_text())
        item={'case':i,'image':e.image_path,'answer':r['generated_text_all'],
              'mentions_sushi':any(w in r['generated_text_all'] for w in ['寿司','すし','スシ']),
              'training_line':e.source['training_line']}
        cases.append(item)
        page+=f'<h2>{i}枚目</h2><img style="max-width:600px;max-height:350px" src="{os.path.relpath(e.image_path,OUT)}"><p>質問：{QUESTION}</p><p>応答：{html.escape(item["answer"])}</p><audio controls src="case_{i:02d}/conversation_stereo.wav"></audio>'
        (OUT/'review.html').write_text(page)
        (OUT/'summary.json').write_text(json.dumps({'num_tested':len(cases),'num_mentions_sushi':sum(c['mentions_sushi'] for c in cases),'cases':cases,'note':'Mention detection is not automatically an accuracy score.'},ensure_ascii=False,indent=2))
        print(json.dumps(item,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
