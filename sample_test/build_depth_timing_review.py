"""Build the saved diagnostic comparison without rerunning inference."""
import html
import json
import os
from pathlib import Path
from check_turn_timing import measure

ROOT=Path(__file__).resolve().parents[1]

def build():
    out=ROOT/'eval_results/depth_turn_timing_comparison_20261012'
    cases=[('追加学習前（seed 42）',ROOT/'eval_results/image_probe_test3_audio_sampling_20261012'),
           ('100 step（seed 43）',out/'step100'),('200 step（seed 43）',out/'step200'),
           ('500 step（seed 43）',ROOT/'eval_results/image_probe_test3_depth500_audio_sampling_seed43_20261012')]
    page='<!doctype html><meta charset="utf-8"><title>発話タイミング比較</title><style>body{font-family:sans-serif;max-width:1000px;margin:32px auto}section{border:1px solid #ccc;padding:20px;margin:16px 0}audio{width:100%}pre{white-space:pre-wrap}</style><h1>発音・発話タイミングの比較</h1><p>左：同じユーザー音声／右：モデル生成音声。全条件でテキストは greedy、音声のみ sampling。同じ質問・画像です。追加学習前だけ既存の seed 42 の録音です。数値は音量による診断であり、短い相づちを失敗とは判定しません。</p>'
    for label,directory in cases:
        if not (directory/'result.json').is_file():continue
        record=json.loads((directory/'result.json').read_text())
        timing=measure(directory)
        page+=f'<section><h2>{html.escape(label)}</h2><audio controls src="{os.path.relpath(directory/"conversation_stereo.wav",out)}"></audio>'
        page+='<h3>質問前を含む全生成テキスト</h3><pre>'+html.escape(record['generated_text_all'])+'</pre>'
        page+='<details><summary>タイミング診断</summary><pre>'+html.escape(json.dumps(timing,ensure_ascii=False,indent=2))+'</pre></details></section>'
    (out/'comparison.html').write_text(page)
    print(out/'comparison.html')

if __name__=='__main__':build()
