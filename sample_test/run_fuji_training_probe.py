"""Five train-member image probes; answers are not supplied to the model."""
import argparse
import json
from dataclasses import replace
from pathlib import Path

from run_image_audio_probe import ROOT, main as infer_probe

OUT = ROOT/'eval_results/fuji_training_probe_20261010'
QUESTION = 'この画像に写っている山は、何という山ですか？'
# Visually checked outdoor photographs, not mere textual mentions of Fuji.
SELECTED = [4, 5, 7, 9, 12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'infer'])
    args = parser.parse_args()
    from jmoshivis.evaluation.schema import EvalExample, read_manifest, write_manifest
    if args.mode == 'prepare':
        candidates = json.loads((OUT/'candidates.json').read_text())
        selected = [candidates[i] for i in SELECTED]
        examples = []
        for i, row in enumerate(selected, 1):
            examples.append(EvalExample(
                sample_id=f'fuji_train_{row["uid"][:16]}', benchmark='train_member_fuji_diagnostic',
                split='diagnostic_train_member', domain='japan', task='vqa',
                image_path=row['image'], question=QUESTION, answers=['富士山'],
                source={'image_sha256': row['image_sha256'], 'training_uid':row['uid'],
                    'training_manifest':str(ROOT/'data/speech_v3/train.jsonl'), 'training_line':row['train_line']},
                metadata={'training_dialogue':row['text'],
                    'note':'Training split membership only; not proof of exact sample exposure.'},
            ))
        write_manifest(OUT/'manifest.jsonl', examples)
        (OUT/'selected_train_rows.json').write_text(json.dumps(selected, ensure_ascii=False, indent=2))
        return
    import html
    master_manifest = OUT/'manifest_audio.jsonl'
    examples = [replace(item,
        image_path=str(item.resolve_image(master_manifest).resolve()),
        audio_path=str(item.resolve_audio(master_manifest).resolve()))
        for item in read_manifest(master_manifest, check_files=True)]
    runtime = None
    results = []
    page = '<!doctype html><meta charset="utf-8"><title>学習画像の富士山テスト</title><h1>学習画像の富士山テスト（音声入力・greedy）</h1>'
    page += '<p>ユーザー音声：jvnv-M1。左＝ユーザー、右＝モデル。各画像で会話状態をリセット。学習画像を使った診断であり、汎化評価ではありません。</p>'
    for i, example in enumerate(examples, 1):
        case = OUT/f'case_{i:02d}'
        case.mkdir(exist_ok=True)
        # TTS emits absolute WAV paths; retain the exact question waveforms.
        write_manifest(case/'manifest_audio.jsonl', [example])
        runtime = infer_probe(['infer', '--image', example.image_path,
            '--output-dir', str(case), '--device','cuda:0'], runtime=runtime)
        result = json.loads((case/'result.json').read_text())
        item = {'case':i, 'uid':example.source['training_uid'],
            'training_line':example.source['training_line'], 'image':example.image_path,
            'question':example.question, 'answer':result['answer_text'],
            'mentions_fuji': '富士山' in result['answer_text'],
            'generated_text_all':result['generated_text_all'],
            'stereo':str(case/'conversation_stereo.wav')}
        results.append(item)
        import os
        page += '<section><h2>'+str(i)+'枚目</h2><img style="max-width:600px;max-height:360px" src="'+os.path.relpath(example.image_path,OUT)+'">'
        page += '<p>質問：'+html.escape(example.question)+'</p><p>応答：'+html.escape(item['answer'])+'</p>'
        page += '<audio controls src="case_'+f'{i:02d}'+'/conversation_stereo.wav"></audio>'
        page += '<details><summary>学習対話・出典を確認</summary><p>train.jsonl '+str(item['training_line'])+'行目／UID '+item['uid']+'</p><p>'+html.escape(example.metadata['training_dialogue'])+'</p></details></section>'
        summary = {'question':QUESTION, 'decoding':'greedy', 'voice':'jvnv-M1',
            'num_tested':len(results), 'num_mentions_fuji':sum(r['mentions_fuji'] for r in results),
            'cases':results,
            'scope':'Visually selected five training-manifest images. Not an independent generalization metric.'}
        (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        (OUT/'review.html').write_text(page)
        print(json.dumps(item,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
