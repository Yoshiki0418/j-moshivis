"""Build review-only backchannel overlays; never mutate production samples.

Run in the existing .venv-train environment. ASR timing is auxiliary, not a replacement
for the original assistant alignments. No model training or inference is started.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unicodedata

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'ssvd' / 'tts'))
sys.path.insert(0, str(ROOT / 'j-moshivis'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def anchor_end(words, prefix):
    """Require the selected semantic prefix to match auxiliary ASR exactly."""
    expected = normalized(prefix)
    heard = ''.join(normalized(w['text']) for w in words)
    if not expected or not heard.startswith(expected):
        raise ValueError(f'ASR prefix mismatch; manually review timing: {expected!r} / {heard!r}')
    n = 0
    for w in words:
        n += len(normalized(w['text']))
        if n == len(expected):
            return float(w['absolute_end'])
        if n > len(expected):
            raise ValueError('Anchor splits an ASR word; choose a verified word boundary')
    raise ValueError('Anchor not found')


def overlay_pcm(base, clip, start, sr, segments, user_segment, guard=.3):
    """Add PCM16 only to exact digital silence, with same-channel guard."""
    if base.dtype != np.int16 or base.ndim != 2 or base.shape[1] != 2:
        raise ValueError('Pilot requires PCM16 stereo')
    if clip.dtype != np.int16 or clip.ndim != 1 or len(clip) == 0:
        raise ValueError('Pilot requires a nonempty mono PCM16 clip')
    end = start + len(clip)
    if not (0 <= start < end <= len(base)):
        raise ValueError('Overlay exceeds original duration')
    if start / sr < user_segment['start'] + 1 or end / sr > user_segment['end'] - .8:
        raise ValueError('Insufficient user onset/tail margin')
    for s in segments:
        if s['speaker'] == 'ASSISTANT' and start / sr < s['end'] + guard and end / sr > s['start'] - guard:
            raise ValueError('Overlay intersects an existing assistant utterance or guard')
    a, b = max(0, start - round(guard * sr)), min(len(base), end + round(guard * sr))
    if np.any(base[a:b, 0]):
        raise ValueError('Assistant channel is not exactly silent, including guard')
    out = base.copy()
    out[start:end, 0] = clip  # Exact zeros + new PCM; no global gain or clipping.
    return out


def extend_alignment(original, refined, text, start, end):
    result = copy.deepcopy(original)
    additions = {
        'alignments': [[p, ts, 'ASSISTANT'] for p, ts, tid in refined],
        'token_alignments': [[[tid], ts, 'ASSISTANT'] for p, ts, tid in refined],
        'utterances': [dict(speaker='ASSISTANT', text=text, text_spm=text,
                            start=start, end=end, n_tokens=len(refined))],
    }
    for key, entries in additions.items():
        result[key].extend(entries)
        result[key].sort(key=(lambda x: x['start']) if key == 'utterances' else (lambda x: x[1][0]))
    return result, additions


def validate_alignment(original, augmented, additions, sp, duration):
    from collections import Counter
    from jmoshivis.datasets.interleaver import Interleaver
    def counts(rows):
        return Counter(json.dumps(x, sort_keys=True, ensure_ascii=False) for x in rows)
    for key in additions:
        if counts(augmented[key]) != counts(original[key]) + counts(additions[key]):
            raise ValueError('Original alignment entries changed')
    for ids, (s, e), role in augmented['token_alignments']:
        if not (math.isfinite(s) and math.isfinite(e) and 0 <= s < e <= duration + .001):
            raise ValueError('Invalid token timing')
        if sp.unk_id() in ids:
            raise ValueError('Unknown token in alignment')
    il = Interleaver(sp, 12.5, 3, 0, -1, keep_main_only=True, keep_and_shift=True, device='cpu')
    def placed(data):
        entries = sorted([a for a in data['token_alignments'] if a[2] == 'ASSISTANT'], key=lambda x: x[1][0])
        stream = il.prepare_item(data['token_alignments'], duration, pretokenized=True)[0, 0].tolist()
        frames = [i for i, t in enumerate(stream) if t not in (3, 0, -1)]
        ids = [stream[i] for i in frames]
        if ids != [t for a in entries for t in a[0]]:
            raise ValueError('Loader loses or changes tokens')
        if any(len(a[0]) != 1 for a in entries):
            raise ValueError('Expected one assistant SPM token per entry')
        return entries, frames
    _, old_frames = placed(original)
    new_entries, new_frames = placed(augmented)
    addition_counts = counts(additions['token_alignments'])
    preserved_frames = []
    for entry, frame in zip(new_entries, new_frames):
        key = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        if addition_counts[key]:
            addition_counts[key] -= 1
        else:
            preserved_frames.append(frame)
    if old_frames != preserved_frames:
        raise ValueError('Added tokens shift original text frames')
    assistant_entries = [a for a in augmented['token_alignments'] if a[2] == 'ASSISTANT']
    offset = 0
    for u in augmented['utterances']:
        if u['speaker'] != 'ASSISTANT':
            continue
        ids = [a[0][0] for a in assistant_entries[offset:offset + u['n_tokens']]]
        if ids != sp.encode(u['text_spm']):
            raise ValueError('Per-utterance SPM ID mismatch')
        offset += u['n_tokens']
    return dict(original_alignment_entries_preserved=True, loader_token_ids_exact=True,
                original_text_token_frames_unchanged=True, per_utterance_spm_exact=True)


def build(plan_path, timings_path, out_dir):
    import sentencepiece as spm
    from refine_alignments import refine_utterance
    from backchannel_review_page import write_review
    plan = json.loads(Path(plan_path).read_text())
    timing = json.loads(Path(timings_path).read_text())
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(plan_path, out_dir / 'plan.json')
    shutil.copyfile(timings_path, out_dir / 'auxiliary_asr.json')
    tokenizer = ROOT / 'models/j-moshi/tokenizer_spm_32k_3.model'
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    voice_path = Path(plan['voice_path'])
    clip, clip_sr = sf.read(voice_path, dtype='int16')
    if clip.ndim == 2 and clip.shape[1] == 2 and not np.any(clip[:, 1]):
        clip = clip[:, 0]  # Existing solo preview stores a silent USER channel.
    if clip.ndim != 1 or sf.info(voice_path).subtype != 'PCM_16':
        raise ValueError('Voice asset must be PCM16 mono')
    shutil.copyfile(voice_path, out_dir / 'backchannel_source.flac')
    shutil.copyfile(voice_path.parent / 'audit.json', out_dir / 'voice_source_audit.json')
    bc_words = timing['backchannel']['words']
    if normalized(timing['backchannel']['transcript']) != normalized(plan['text']):
        raise ValueError('Short backchannel ASR disagrees; review voice before proceeding')
    # Only the inserted clip receives gain and edge fades.
    clip = np.rint(clip.astype(np.float64) * plan.get('gain', 1.)).clip(-32768, 32767).astype(np.int16)
    fade = min(round(.005 * clip_sr), len(clip) // 2)
    clip[:fade] = np.rint(clip[:fade] * np.linspace(0, 1, fade)).astype(np.int16)
    clip[-fade:] = np.rint(clip[-fade:] * np.linspace(1, 0, fade)).astype(np.int16)
    threshold = max(100, float(np.max(np.abs(clip.astype(np.int32)))) * .05)
    audible = np.flatnonzero(np.abs(clip.astype(np.int32)) > threshold)
    if len(audible) == 0:
        raise ValueError('Silent backchannel asset')
    onset = int(audible[0]) / clip_sr
    manifest = dict(schema_version='jmoshivis.backchannel-pilot.v1', preview_only=True,
                    training_eligible=False, review_status='pending', human_listening_performed=False,
                    placement_method='manually selected semantic prefix + auxiliary ASR boundary + fixed delay; not automatic suitability judgment',
                    tokenizer_sha256=sha(tokenizer), voice_sha256=sha(voice_path), samples=[])
    for item in plan['samples']:
        source = item['source']
        uid = source['uid']
        td = next(x for x in timing['users'] if x['uid'] == uid)
        src = {k: (ROOT / source[k]).resolve() for k in ['audio_path','alignment_path','speech_dialogue_path','image_path']}
        before = {k: sha(p) for k, p in src.items()}
        if before != item['source_sha256']:
            raise ValueError('Source no longer matches the selected pilot plan')
        meta = json.loads(src['speech_dialogue_path'].read_text())
        original = json.loads(src['alignment_path'].read_text())
        if original.get('format') != 'spm_pieces_v2' or original['tokenizer'] != tokenizer.name:
            raise ValueError('Unsupported original alignment/tokenizer')
        if meta.get('assistant_voice') != 'honoka':
            raise ValueError('Assistant voice differs from pilot asset')
        segment = meta['segments'][source['segment_index']]
        if segment != source['segment'] or not segment['text'].startswith(item['anchor_prefix']):
            raise ValueError('Source segment changed')
        pcm, sr = sf.read(src['audio_path'], dtype='int16', always_2d=True)
        if sf.info(src['audio_path']).subtype != 'PCM_16' or sr != clip_sr:
            raise ValueError('Resampling or lossy conversion is not allowed')
        anchor = anchor_end(td['words'], item.get('asr_anchor_prefix', item['anchor_prefix']))
        start = round((anchor + plan['audible_delay'] - onset) * sr)
        end = start + len(clip)
        augmented_pcm = overlay_pcm(pcm, clip, start, sr, meta['segments'], segment)
        a, b = start / sr, end / sr
        words = [[w['text'], [a + w['start'], a + w['end']], 'ASSISTANT'] for w in bc_words]
        refined = refine_utterance(sp, plan['text'], words, a, b)
        augmented, additions = extend_alignment(original, refined, plan['text'], a, b)
        validation = validate_alignment(original, augmented, additions, sp, len(pcm) / sr)
        folder = out_dir / 'samples' / uid
        folder.mkdir(parents=True)
        # Baselines are links, never opened for writing. Only 3 augmented files
        # are materialized for reliable browser seeking; production uses none.
        (folder / 'original.wav').symlink_to(src['audio_path'])
        (folder / 'image.jpg').symlink_to(src['image_path'])
        sf.write(folder / 'augmented.flac', augmented_pcm, sr, subtype='PCM_16')
        reread, check_sr = sf.read(folder / 'augmented.flac', dtype='int16', always_2d=True)
        mask = np.ones(len(pcm), dtype=bool); mask[start:end] = False
        checks = dict(user_pcm_unchanged=np.array_equal(pcm[:,1], reread[:,1]),
                      assistant_pcm_outside_overlay_unchanged=np.array_equal(pcm[mask,0], reread[mask,0]),
                      roundtrip_pcm_exact=np.array_equal(augmented_pcm, reread),
                      sample_count_unchanged=len(reread)==len(pcm), sample_rate_unchanged=sr==check_sr)
        if not all(checks.values()):
            raise ValueError('Audio invariants failed')
        validation.update(checks)
        event = dict(text=plan['text'], speaker='ASSISTANT', purpose='backchannel', voice='honoka',
                     start=a, end=b, start_sample=start, end_sample=end,
                     anchor_end=anchor, audible_start=a+onset, audible_delay=plan['audible_delay'])
        new_meta = copy.deepcopy(meta)
        new_meta.update(uid=uid+'__bc_pilot_v1', source_uid=meta.get('source_uid',uid),
                        split_group_key=meta.get('split_group_key',uid), review_required=True,
                        review_id=None, review_status='pending', human_certified=False,
                        preview_only=True, training_eligible=False)
        new_meta['segments'].append({k:event[k] for k in ['speaker','text','start','end','voice','purpose']})
        new_meta['segments'].sort(key=lambda x:x['start'])
        if [s for s in new_meta['segments'] if s.get('purpose') != 'backchannel'] != meta['segments']:
            raise ValueError('Original metadata segments changed')
        validation['original_segments_preserved'] = True
        write_json(folder / 'dialogue.json',new_meta)
        write_json(folder / 'assistant_dialogue.refined.json',augmented)
        write_json(folder / 'alignment_additions.json',additions)
        snippet_a = max(0, round((segment['start'] - .5) * sr))
        snippet_b = min(len(pcm), round((segment['end'] + 2.) * sr))
        sf.write(folder / 'snippet_original.wav',pcm[snippet_a:snippet_b],sr,subtype='PCM_16')
        sf.write(folder / 'snippet_augmented.wav',reread[snippet_a:snippet_b],sr,subtype='PCM_16')
        after = {k:sha(p) for k,p in src.items()}
        if before != after:
            raise ValueError('Source assets changed during build')
        validation['source_files_unchanged'] = True
        base = f'samples/{uid}'
        record = dict(uid=uid,title=item['title'],user_text=segment['text'],anchor_prefix=item['anchor_prefix'],
                      reason=item['reason'],event=event,validation=validation,snippet_start=snippet_a/sr,
                      source_paths={k:str(p) for k,p in src.items()},source_sha256=before,
                      image_url=base+'/image.jpg',original_url=base+'/original.wav',
                      augmented_url=base+'/augmented.flac',snippet_original_url=base+'/snippet_original.wav',
                      snippet_augmented_url=base+'/snippet_augmented.wav',
                      augmented_audio_sha256=sha(folder/'augmented.flac'),
                      augmented_alignment_sha256=sha(folder/'assistant_dialogue.refined.json'))
        write_json(folder/'overlay.json',record)
        manifest['samples'].append(record)
        print(uid, 'OK', f'{a:.3f}-{b:.3f}s', flush=True)
    write_json(out_dir/'manifest.json',manifest)
    write_review(out_dir,manifest)
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--timings', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    build(args.plan, args.timings, args.out)
