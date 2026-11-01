"""Replay exact training USER channel; never replay assistant answers."""
import argparse
import json
from pathlib import Path

from run_image_audio_probe import ROOT, main as infer_probe

OUT = ROOT/'eval_results/fuji_original_audio_probe_20261010'
SOURCE = ROOT/'eval_results/fuji_training_probe_20261010/selected_train_rows.json'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'infer'])
    p.add_argument('--output-dir', type=Path, default=OUT)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors')
    args = p.parse_args()
    out = args.output_dir.resolve()
    from jmoshivis.evaluation.schema import EvalExample, write_manifest, sha256_file
    rows = json.loads(SOURCE.read_text())
    out.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        import numpy as np
        import soundfile as sf
        for i, row in enumerate(rows, 1):
            case = out/f'case_{i:02d}'
            case.mkdir(exist_ok=True)
            directory = Path(row['image']).parent
            original_wav = directory/'stereo_dialogue.wav'
            dialogue = json.loads((directory/'dialogue.json').read_text())
            original, sr = sf.read(original_wav, dtype='float32', always_2d=True)
            assert sr == 24000 and original.shape[1] == 2
            # Dataset builder explicitly maps ASSISTANT=0, USER=1.
            user_audio = case/'original_training_user.wav'
            sf.write(user_audio, original[:,1], sr, subtype='FLOAT')
            saved, _ = sf.read(user_audio, dtype='float32')
            assert np.array_equal(saved, original[:,1])
            turns = [dict(turn=j+1, question=s['text'], question_start_sec=s['start'],
                question_speech_end_sec=s['end'], question_wav=str(user_audio), voice=s['voice'],
                question_contains_fuji='富士山' in s['text'])
                for j,s in enumerate(s for s in dialogue['segments'] if s['speaker']=='USER')]
            (case/'turns.json').write_text(json.dumps(turns,ensure_ascii=False,indent=2))
            example = EvalExample(sample_id=f'original_user_{row["uid"][:16]}',
                benchmark='original_training_user_audio_diagnostic', split='diagnostic_train_member',
                domain='japan', task='vqa', image_path=row['image'],
                question=' / '.join(t['question'] for t in turns), answers=['富士山'],
                audio_path=str(user_audio), source={'image_sha256':row['image_sha256'],
                    'training_uid':row['uid'], 'training_line':row['train_line']},
                metadata={'input_audio_source':'exact original training USER channel (index 1)',
                    'original_stereo':str(original_wav), 'original_stereo_sha256':sha256_file(original_wav),
                    'user_samples_identical':True, 'user_voice':dialogue['user_voice'],
                    'question_audio':{'speech_end_sec':turns[-1]['question_speech_end_sec']}})
            write_manifest(case/'manifest_audio.jsonl',[example])
            print(i, dialogue['user_voice'], f'{len(original)/sr:.2f}s',len(turns),'USER turns')
        return

    import html
    import os
    from jmoshivis.evaluation.response import SPECIAL_TEXT_IDS
    runtime = None
    cases = []
    page = '<!doctype html><meta charset="utf-8"><title>学習ユーザー音声の再生テスト</title><h1>学習ユーザー音声の再生テスト（greedy）</h1>'
    page += '<p>元のステレオ音声のUSERチャンネルのみを入力。学習用ASSISTANT音声や正解テキストは入力しません。質問と無音区間のタイミングも維持。左＝学習時のユーザー音声、右＝モデルの新しい生成音声。</p>'
    for i, row in enumerate(rows,1):
        case = out/f'case_{i:02d}'
        runtime = infer_probe(['infer','--image',row['image'],'--output-dir',str(case),
            '--turns-json',str(case/'turns.json'),'--device','cuda:0',
            '--checkpoint',str(args.checkpoint)],runtime=runtime)
        result = json.loads((case/'result.json').read_text())
        turns = result['turns']
        hint_time = next((t['question_start_sec'] for t in turns if t['question_contains_fuji']),None)
        prefix = [e['token_id'] for e in result['raw_text_events']
            if e['token_id'] not in SPECIAL_TEXT_IDS and (hint_time is None or e['time_sec']<hint_time)]
        unhinted_text = runtime.tokenizer.decode(prefix).replace('⁇','').strip()
        item = {'case':i, 'uid':row['uid'], 'training_line':row['train_line'],
            'generated_text_all':result['generated_text_all'], 'turns':turns,
            'first_user_fuji_hint_sec':hint_time, 'text_before_fuji_hint':unhinted_text,
            'mentions_fuji_anywhere':'富士山' in result['generated_text_all'],
            'mentions_fuji_before_user_hint':'富士山' in unhinted_text}
        cases.append(item)
        page += '<section><h2>'+str(i)+'枚目</h2><img style="max-width:600px;max-height:350px" src="'+os.path.relpath(row['image'],out)+'">'
        page += '<p>ユーザーが「富士山」と言う時刻：'+('なし' if hint_time is None else str(hint_time)+'秒')+'</p>'
        page += '<audio controls src="case_'+f'{i:02d}'+'/conversation_stereo.wav"></audio>'
        for turn in turns:
            page += '<p><b>USER '+str(turn['turn'])+'：</b>'+html.escape(turn['question'])+'<br><b>MODEL：</b>'+html.escape(turn['answer_text'])+'</p>'
        page += '<details><summary>全生成テキスト</summary>'+html.escape(item['generated_text_all'])+'</details></section>'
        summary = {'checkpoint':str(args.checkpoint.resolve()), 'num_tested':len(cases), 'num_fuji_anywhere':sum(c['mentions_fuji_anywhere'] for c in cases),
            'num_fuji_before_user_hint':sum(c['mentions_fuji_before_user_hint'] for c in cases),
            'cases':cases, 'note':'User hints are a confound; neither mention metric is automatically a visual identification accuracy.'}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
        (out/'review.html').write_text(page)
        print(json.dumps(item,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
