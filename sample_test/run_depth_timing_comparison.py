"""Run saved intermediate checkpoints without retraining or touching GPU0."""
import json
import html
from pathlib import Path
from run_image_audio_probe import main, ROOT
from check_turn_timing import measure

out=ROOT/'eval_results/depth_turn_timing_comparison_20261012'
out.mkdir(parents=True,exist_ok=True)
runtime=None
reports={}
for step in [100,200]:
    directory=out/f'step{step}'
    runtime=main(['infer','--image',str(ROOT/'test3.jpg'),'--output-dir',str(directory),
        '--audio-manifest',str(ROOT/'eval_results/image_probe_test3_greedy_20261012/manifest_audio.jsonl'),
        '--split-turns','--audio-only-sampling','--seed','43',
        '--depth-delta',str(ROOT/f'checkpoints/depth_audio_honoka_pilot_v2_20261012/step_{step:05d}/depth.safetensors')],runtime=runtime)
    reports[str(step)]=measure(directory)
    (out/'timing_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
print('COMPLETED',json.dumps(reports,ensure_ascii=False),flush=True)
cases=[('追加学習前', ROOT/'eval_results/image_probe_test3_audio_sampling_20261012'),
       ('100 step',out/'step100'),('200 step',out/'step200'),
       ('500 step',ROOT/'eval_results/image_probe_test3_depth500_audio_sampling_seed43_20261012')]
page='<!doctype html><meta charset="utf-8"><title>発話タイミング比較</title><style>body{font-family:sans-serif;max-width:1000px;margin:32px auto}section{border:1px solid #ccc;padding:20px;margin:16px 0}audio{width:100%}pre{white-space:pre-wrap}</style><h1>発音・発話タイミングの比較</h1><p>左：同じユーザー音声／右：モデル生成音声。100・200・500 step は seed 43。追加学習前の既存比較音声のみ seed 42。数値は音量による診断であり、相づちの良否は人手で確認してください。</p>'
import os
for label,directory in cases:
    record=json.loads((directory/'result.json').read_text())
    timing=measure(directory)
    page+=f'<section><h2>{label}</h2><audio controls src="{os.path.relpath(directory/"conversation_stereo.wav",out)}"></audio>'
    page+='<h3>質問前を含む全生成テキスト</h3><pre>'+html.escape(record['generated_text_all'])+'</pre>'
    page+='<details><summary>タイミング診断</summary><pre>'+html.escape(json.dumps(timing,ensure_ascii=False,indent=2))+'</pre></details></section>'
(out/'comparison.html').write_text(page)
