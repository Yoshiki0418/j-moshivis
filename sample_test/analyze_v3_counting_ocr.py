"""Reproducible text-only task-candidate audit of the exact v3 train split.

Never treats keyword hits as verified visual supervision or correct answers.
No model API, no GPU, and no evaluation split in the reported denominator.
"""
import collections
import hashlib
import json
import random
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'eval_results/v3_counting_ocr_audit_20261012'
COUNT=re.compile(r'何(?:人|個|匹|頭|羽|台|本|枚|つ|脚|杯|軒|着|組|輪|冊|棟|体|足|皿)|いくつ|何[がを]?数|数(?:えて|えら|える|は|が|を)|人数|個数')
OCR=re.compile(r'(?:何|なん|なに|どう).{0,8}(?:書[かい]|読)|(?:書[かい]|文字|看板|標識|店名|ロゴ|ラベル|表示|掲示|名前|番号).{0,15}(?:読[めむん]|何|なん|なに|教えて)|(?:読[めむん]).{0,10}(?:文字|看板|標識|店名|ラベル)')
DIRECT_COUNT=re.compile(r'何(?:人|個|匹|頭|羽|台|本|枚|つ|脚|杯|軒|組|輪|冊|棟|体|足|皿)(?!か|も)|何着(?!て|か|も)|いくつ(?!か|も)|数えて|(?:人数|個数).{0,8}(?:何|どれ|教えて|わか)')
DIRECT_OCR=re.compile(r'(?:何[てとがを]|なん[てと]|なに[がとを]?|どんな文字).{0,8}書|(?:何|なん|なに|どう).{0,5}読|(?:文字|看板|標識|店名|ロゴ|ラベル|表示|掲示|名前|番号).{0,12}(?:読[めむん])|(?:読んで|読める|読み取).{0,12}(?:文字|看板|標識|店名|ラベル)')
NUMBER_ANSWER=re.compile(r'[0-9０-９一二三四五六七八九十百]+(?:人|個|匹|頭|羽|台|本|枚|つ|脚|杯|軒|着|組|輪|冊|棟|体|足|皿)')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=ROOT/'data/speech_v3/train.jsonl'
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    run=json.loads((ROOT/'checkpoints/llmjp_v3_id0w1_run1/run.json').read_text())
    assert digest==run['metadata']['datasets']['train']['sha256']
    train={json.loads(l)['uid'] for l in manifest.open()}
    stats=collections.Counter();source_stats=collections.defaultdict(collections.Counter)
    examples={'counting':[],'ocr':[]};covered=set();fields=collections.Counter()
    merged=ROOT/'data/dialogues_merged/dialogues_final_v3.jsonl'
    with (OUT/'candidate_turns.jsonl').open('w') as output:
        for line in merged.open():
            row=json.loads(line)
            if row['uid'] not in train:continue
            assert row['uid'] not in covered
            covered.add(row['uid']);fields.update(row.keys())
            source=row['image'].split('/data/images/')[-1].split('/')[0]
            turns=row['dialogue'];dialogue_flags=set()
            for i,turn in enumerate(turns):
                if turn.get('speaker') not in ['ユーザー','USER']:continue
                stats['user_turns']+=1
                q=turn['text'];answer=turns[i+1]['text'] if i+1<len(turns) and turns[i+1].get('speaker') in ['アシスタント','ASSISTANT'] else ''
                for kind,broad,direct in [('counting',COUNT,DIRECT_COUNT),('ocr',OCR,DIRECT_OCR)]:
                    if not broad.search(q):continue
                    flags=[kind+'_broad']
                    if direct.search(q):flags.append(kind+'_direct')
                    if kind=='counting' and direct.search(q) and re.search(r'くらい|ぐらい|ほど|およそ|ざっと',q):
                        flags.append('counting_approximate_question')
                    if kind=='counting' and direct.search(q) and NUMBER_ANSWER.search(answer):
                        flags.append('counting_direct_numeric_answer')
                    for flag in flags:
                        stats[flag+'_turns']+=1;dialogue_flags.add(flag)
                    record={'uid':row['uid'],'image':row['image'],'source':source,'question':q,'answer':answer,'flags':flags,'turn_index':i}
                    examples[kind].append(record)
                    output.write(json.dumps(record,ensure_ascii=False)+'\n')
            for flag in dialogue_flags:
                stats[flag+'_dialogues']+=1;source_stats[flag][source]+=1
    assert covered==train
    result={'schema':'text-only task-candidate audit v1','train_dialogues':len(train),
        'train_manifest_sha256':digest,'dialogue_file':str(merged),
        'dialogue_sha256':hashlib.sha256(merged.read_bytes()).hexdigest(),
        'statistics':dict(stats),'source_statistics':dict(source_stats),'fields':dict(fields),
        'patterns':{k:v.pattern for k,v in [('counting_broad',COUNT),('counting_direct',DIRECT_COUNT),('ocr_broad',OCR),('ocr_direct',DIRECT_OCR),('numeric_answer',NUMBER_ANSWER)]},
        'limitations':['Automatic text-only candidate counts; no image-grounding verification',
            'Direct questions still include general knowledge, existence questions and approximate counts',
            'Numeric answers are not verified correct counts',
            'OCR includes non-Japanese writing and cannot establish Japanese OCR coverage',
            'Task labels and answer annotations are absent in the merged v3 schema; not proof that no original image source has annotations']}
    (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    samples={k:random.Random(42).sample([r for r in v if k+'_direct' in r['flags']],20) for k,v in examples.items()}
    (OUT/'sample40.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
