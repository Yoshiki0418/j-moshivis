"""Compare alternative backchannel voices at the same audible onset."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from build_backchannel_pilot import ROOT, overlay_pcm, sha, write_json
from backchannel_review_page import write_review


def build_comparison(root, assets, original_root, plan_path=None):
    root, assets, original_root = (Path(path).resolve() for path in (root, assets, original_root))
    plan_path = Path(plan_path) if plan_path is not None else original_root / 'plan.json'
    base_plan = json.loads(plan_path.read_text())
    old = json.loads((original_root / 'manifest.json').read_text())
    voice_times = json.loads((assets / 'timings.json').read_text())
    if any((root / name).exists() or (root / name).is_symlink()
           for name in ['manifest.json', 'index.html', 'previous', 'variants']):
        raise FileExistsError('Refusing to overwrite an existing voice comparison; choose a new --out directory')
    result = copy.deepcopy(old)
    result['samples'] = []
    old_item = copy.deepcopy(old['samples'][0])
    old_item['title'] = '前回の素材：うん'
    old_item['reason'] = '冷たく聞こえた比較基準。原本の音声は変更していません。'
    keys = ['image_url', 'original_url', 'augmented_url', 'snippet_original_url', 'snippet_augmented_url']
    for key in keys:
        old_item[key] = 'previous/' + old_item[key]
    result['samples'].append(old_item)
    source = base_plan['samples'][0]['source']
    src_audio = (ROOT / source['audio_path']).resolve()
    src_image = (ROOT / source['image_path']).resolve()
    src_meta = (ROOT / source['speech_dialogue_path']).resolve()
    src_alignment = (ROOT / source['alignment_path']).resolve()
    original_paths = [src_audio, src_image, src_meta, src_alignment]
    source_hashes = {str(p): sha(p) for p in original_paths}
    current_hashes = {key: sha((ROOT / source[key]).resolve()) for key in
                      ['audio_path', 'image_path', 'speech_dialogue_path', 'alignment_path']}
    if (source['uid'] != old_item['uid']
            or current_hashes != base_plan['samples'][0]['source_sha256']
            or current_hashes != old_item['source_sha256']):
        raise ValueError('Source no longer matches the original pilot; rebuild the baseline before comparison')
    pcm, sr = sf.read(src_audio, dtype='int16', always_2d=True)
    if sf.info(src_audio).subtype != 'PCM_16':
        raise ValueError('Original audio must remain PCM16')
    meta = json.loads(src_meta.read_text())
    user = meta['segments'][source['segment_index']]
    if user != source['segment']:
        raise ValueError('Source segment differs from the selected pilot plan')
    variants = [
        ('un_expressive', 'うん。', '候補1：うん／抑揚を増やす'),
        ('unun_neutral', 'うんうん。', '候補2：うんうん／通常スタイル'),
        ('unun_expressive', 'うんうん。', '候補3：うんうん／うれしいスタイル'),
    ]
    root.mkdir(parents=True, exist_ok=True)
    (root / 'previous').symlink_to(original_root, target_is_directory=True)
    for name, text, title in variants:
        plan = copy.deepcopy(base_plan)
        plan['samples'] = [plan['samples'][0]]
        plan['samples'][0]['title'] = title
        plan['voice_path'] = str(assets / (name + '.flac'))
        plan['text'] = text
        plan_path = root / (name + '.plan.json')
        write_json(plan_path, plan)
        clip, clip_sr = sf.read(plan['voice_path'], dtype='int16')
        if clip_sr != sr or clip.ndim != 1 or sf.info(plan['voice_path']).subtype != 'PCM_16':
            raise ValueError('Voice must be PCM16 mono at the original sample rate')
        if sha(plan['voice_path']) != voice_times[name]['audio_sha256']:
            raise ValueError('ASR provenance mismatch')
        magnitude = np.abs(clip.astype(np.int32))
        if len(magnitude) == 0:
            raise ValueError('Empty backchannel asset')
        active = np.flatnonzero(magnitude > max(100, magnitude.max() * .05))
        if not len(active):
            raise ValueError('Silent backchannel asset')
        onset = int(active[0])
        start = round(old_item['event']['audible_start'] * sr) - onset
        end = start + len(clip)
        augmented = overlay_pcm(pcm, clip, start, sr, meta['segments'], user)
        folder = root / 'variants' / name
        folder.mkdir(parents=True, exist_ok=False)
        (folder / 'original.wav').symlink_to(src_audio)
        (folder / 'image.jpg').symlink_to(src_image)
        sf.write(folder / 'augmented.flac', augmented, sr, subtype='PCM_16')
        reread, rate = sf.read(folder / 'augmented.flac', dtype='int16', always_2d=True)
        mask = np.ones(len(pcm), dtype=bool); mask[start:end] = False
        checks = dict(user_pcm_unchanged=np.array_equal(pcm[:,1],reread[:,1]),
                      assistant_pcm_outside_overlay_unchanged=np.array_equal(pcm[mask,0],reread[mask,0]),
                      original_length_unchanged=len(pcm)==len(reread),
                      rate_unchanged=rate==sr,
                      roundtrip_exact=np.array_equal(augmented,reread),
                      audible_onset_matches_previous=abs((start+onset)/sr-old_item['event']['audible_start'])<=1/sr,
                      source_files_unchanged=all(sha(p)==source_hashes[str(p)] for p in original_paths))
        if not all(checks.values()):
            raise ValueError('Voice audition invariants failed')
        left = round(old_item['snippet_start']*sr)
        right = min(len(pcm),round((user['end']+2)*sr))
        sf.write(folder/'snippet_original.wav',pcm[left:right],sr,subtype='PCM_16')
        sf.write(folder/'snippet_augmented.wav',reread[left:right],sr,subtype='PCM_16')
        s = copy.deepcopy(old_item)
        s.update(title=title, validation=checks,
                 reason='前回と同じ対話・可聴開始時刻で声だけを試聴。合成指示「'+text+'」、補助ASR「'+voice_times[name]['transcript']+'」。追加分の学習用アライメントは未作成。',
                 alignment_status='not_generated_voice_audition',
                 augmented_alignment_sha256=None, training_eligible=False)
        s['event'].update(text=text,start=start/sr,end=end/sr,start_sample=start,end_sample=end,
                          audible_start=(start+onset)/sr)
        filenames = ['image.jpg','original.wav','augmented.flac','snippet_original.wav','snippet_augmented.wav']
        for key, filename in zip(keys,filenames):
            s[key]='variants/'+name+'/'+filename
        s['augmented_audio_sha256']=sha(folder/'augmented.flac')
        write_json(folder/'audition.json',s)
        result['samples'].append(s)
    result['comparison_note'] = 'Same original conversation and estimated audible onset; only added voice/text differs. Pending listening.'
    write_json(root / 'manifest.json', result)
    page = write_review(root, result)
    content = page.read_text().replace('相づち挿入の試聴レビュー', '相づちの声・抑揚の比較')
    content = content.replace('4 件の試作音声を、元音声と聴き比べます。',
        '同じ対話・同じ可聴開始時刻で、前回の素材と新しい3候補を比較します。各カードのBで追加版、Aで相づちなしを再生できます。')
    page.write_text(content)
    print(page)
    return page


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = ROOT / 'eval_results/backchannel_voice_warmth_20261117'
    parser.add_argument('--out', type=Path, default=default_out)
    parser.add_argument('--assets', type=Path, help='Directory containing the three clips and timings.json; defaults to <out>/assets')
    parser.add_argument('--original-root', type=Path, default=ROOT / 'eval_results/backchannel_pilot_20261117')
    parser.add_argument('--plan', type=Path, help='Local pilot plan; defaults to <original-root>/plan.json')
    args = parser.parse_args()
    build_comparison(args.out, args.assets or args.out / 'assets', args.original_root, args.plan)


if __name__ == '__main__':
    main()
