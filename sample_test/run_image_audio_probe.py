"""Single-image speech probe using the established TTS and inference runtime.

Left channel is the original user question; right is generated Mimi audio,
not text-to-speech of the predicted text. No assistant audio is teacher-forced.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTION = 'この画像には何が写っていますか？ 手前の動物と、奥に見える山について教えてください。'
SPLIT_QUESTIONS = ['この画像の手前にいる動物は何ですか？', 'では、奥に見える山は何という山ですか？']


def main(argv=None, runtime=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'infer'])
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors')
    p.add_argument('--depth-delta', type=Path, help='Overlay a completed independent audio adaptation checkpoint')
    p.add_argument('--split-turns', action='store_true', help='Ask animal and mountain questions in one continuous session')
    p.add_argument('--pause-seconds', type=float, default=12.0)
    p.add_argument('--sampling', action='store_true', help='Use established audio/text sampling temperatures and top-k')
    p.add_argument('--audio-only-sampling', action='store_true', help='Sample audio while keeping text argmax (temperature zero)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--audio-manifest', type=Path, help='Reuse previously synthesized user question WAVs')
    p.add_argument('--questions-json', type=Path, help='List of question/answer_reference objects for a custom image')
    p.add_argument('--turns-json', type=Path, help='Use an existing user-only WAV with its original turn timings')
    args = p.parse_args(argv)
    if args.sampling and args.audio_only_sampling:
        p.error('--sampling and --audio-only-sampling are mutually exclusive')
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    image = args.image.resolve()
    if args.mode == 'prepare':
        from jmoshivis.evaluation.schema import EvalExample, write_manifest, sha256_file
        questions = SPLIT_QUESTIONS if args.split_turns else [QUESTION]
        references = ['牛', '富士山'] if args.split_turns else ['牛と富士山']
        if args.questions_json:
            custom = json.loads(args.questions_json.read_text())
            questions = [item['question'] for item in custom]
            references = [item['answer_reference'] for item in custom]
        image_id = hashlib.sha256(image.read_bytes()).hexdigest()[:12]
        examples = [EvalExample(
            sample_id=f'image_{image_id}_turn{i+1}_20261010' if args.questions_json else (f'fuji_cows_split_turn{i+1}_20261010' if args.split_turns else 'fuji_cows_user_probe_20261010'),
            benchmark='user_image_probe', split='diagnostic', domain='japan', task='vqa',
            image_path=str(image), question=question, answers=[references[i]],
            source={'image_sha256': sha256_file(image)},
            metadata={'purpose': 'single image audio response demonstration, not a benchmark score'},
        ) for i, question in enumerate(questions)]
        write_manifest(out / 'manifest.jsonl', examples)
        return

    os.environ.setdefault('NO_TORCH_COMPILE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    import numpy as np
    import soundfile as sf
    import torch
    from jmoshivis.evaluation.runtime import JmoshiVisAudioRuntime, RuntimeConfig, SAMPLING_CONFIG
    from jmoshivis.evaluation.schema import read_manifest, sha256_file
    torch.set_num_threads(8)
    free, total = torch.cuda.mem_get_info(args.device)
    print(f'GPU memory free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB', flush=True)
    if runtime is None and free < 23 * 2**30:
        raise RuntimeError('Insufficient free GPU memory; other processes are left untouched')
    audio_manifest = (args.audio_manifest or out / 'manifest_audio.jsonl').resolve()
    examples = read_manifest(audio_manifest, check_files=True)
    example = examples[0]
    if args.split_turns and len(examples) < 2:
        raise ValueError('Split-turn probe requires at least two question WAVs')
    timeline = []
    pieces = []
    offset = 0
    for i, item in enumerate(examples if args.split_turns else [example]):
        wav_path = item.resolve_audio(audio_manifest)
        wav, sr = sf.read(wav_path, dtype='float32')
        assert sr == 24000 and wav.ndim == 1
        start = offset/24000
        timeline.append({'turn': i+1, 'question': item.question, 'question_start_sec': start,
            'question_speech_end_sec': start + item.metadata['question_audio']['speech_end_sec'],
            'question_wav': str(wav_path)})
        pieces.append(wav)
        offset += len(wav)
        if args.split_turns and i < len(examples)-1:
            gap = np.zeros(round(args.pause_seconds*24000), dtype=np.float32)
            pieces.append(gap)
            offset += len(gap)
    audio = example.resolve_audio(audio_manifest)
    if args.split_turns:
        audio = out / 'user_questions_timeline.wav'
        sf.write(audio, np.concatenate(pieces), 24000, subtype='FLOAT')
    if args.turns_json:
        if args.split_turns:
            raise ValueError('Existing timeline cannot be combined with synthesized split turns')
        timeline = json.loads(args.turns_json.read_text())
        if not timeline:
            raise ValueError('Existing timeline is empty')
    checkpoint = args.checkpoint.resolve()
    cfg = RuntimeConfig(
        checkpoint=checkpoint, model_config=ROOT / 'j-moshivis/configs/j-moshi-vis.yaml',
        tokenizer=ROOT / 'models/j-moshi/tokenizer_spm_32k_3.model',
        mimi_weight=str(ROOT / 'models/moshika-vis-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors'),
        device=args.device, response_seconds=20.0, use_sampling=args.sampling or args.audio_only_sampling, seed=args.seed,
    )
    if runtime is None:
        print(f'Loading checkpoint: {checkpoint}', flush=True)
        runtime = JmoshiVisAudioRuntime(cfg)
    elif runtime.config != cfg:
        raise ValueError('Cannot reuse a runtime with a different configuration')
    else:
        print('Reusing loaded model; resetting streaming state for new image', flush=True)
    if args.depth_delta:
        from safetensors.torch import load_file
        from train_depth_audio import audio_parameter
        delta_path = args.depth_delta.resolve()
        marker_path = delta_path.parent / 'COMPLETE.json'
        marker = json.loads(marker_path.read_text())
        if marker['sha256']['depth.safetensors'] != sha256_file(delta_path):
            raise ValueError('Audio adaptation checkpoint checksum mismatch')
        source_config = json.loads((delta_path.parent.parent / 'config.json').read_text())
        if source_config['source_checkpoint_sha256'] != sha256_file(checkpoint):
            raise ValueError('Audio adaptation base checkpoint mismatch')
        delta = load_file(str(delta_path), device='cpu')
        params = dict(runtime.model.lm_model.named_parameters())
        expected = {name for name in params if audio_parameter(name)}
        if set(delta) != expected:
            raise ValueError('Audio adaptation parameter coverage mismatch')
        with torch.no_grad():
            for name in expected:
                if delta[name].shape != params[name].shape or not torch.isfinite(delta[name]).all():
                    raise ValueError(f'Invalid audio parameter: {name}')
                params[name].copy_(delta[name])
        del delta
        print(f'Loaded audio adaptation step {marker["step"]}', flush=True)
    # sample_token uses argmax whenever temperature is zero, even with
    # use_sampling=True. Set directly: update_gen_kwargs uses `or` and drops 0.
    runtime.model.temp_text = 0.0 if args.audio_only_sampling else SAMPLING_CONFIG['text_temperature']
    print(f'Decoding: use_sampling={runtime.model.use_sampling}, audio_temp={runtime.model.temp}, text_temp={runtime.model.temp_text}', flush=True)
    frames = []
    step = runtime.model.step

    def capture(*a, **kw):
        tokens, gate = step(*a, **kw)
        frames.append(None if tokens is None else tokens[:, 1:, :].detach().clone())
        return tokens, gate

    runtime.model.step = capture
    with torch.inference_mode():
        result = runtime.infer(image, audio, sample_id=example.sample_id,
            speech_end_sec=timeline[-1]['question_speech_end_sec'])
        runtime.model.step = step
        # Decode the exact generated audio codebooks in frame order. Include
        # initial LM delay as silence so both channels share the input timeline.
        chunks = []
        frame_samples = round(24000 / runtime.frame_rate)
        with runtime.mimi.streaming(batch_size=1):
            for tokens in frames:
                if tokens is None:
                    chunks.append(np.zeros(frame_samples, dtype=np.float32))
                else:
                    chunks.append(runtime.mimi.decode(tokens).float().cpu().numpy().reshape(-1))
    assistant = np.concatenate(chunks)
    user, sr = sf.read(audio, dtype='float32')
    assert sr == 24000 and user.ndim == 1
    assert np.isfinite(assistant).all()
    length = max(len(user), len(assistant))
    stereo = np.zeros((length, 2), dtype=np.float32)
    stereo[:len(user), 0] = user
    stereo[:len(assistant), 1] = assistant
    sf.write(out / 'conversation_stereo.wav', stereo, 24000, subtype='PCM_16')
    sf.write(out / 'assistant_generated.wav', assistant, 24000, subtype='PCM_16')
    from jmoshivis.evaluation.response import extract_answer_events
    for i, turn in enumerate(timeline):
        end = timeline[i+1]['question_start_sec'] if i+1 < len(timeline) else length/24000
        events = [e for e in result['raw_text_events'] if e['time_sec'] < end]
        selected = extract_answer_events(events,
            question_end_frame=round(turn['question_speech_end_sec']*runtime.frame_rate),
            overlap_frames=round(cfg.response_overlap_seconds*runtime.frame_rate))
        turn['answer_text'] = runtime.tokenizer.decode([e['token_id'] for e in selected]).replace('⁇', '').strip()
        turn['response_window_end_sec'] = end
    result.update(question=' / '.join(t['question'] for t in timeline), turns=timeline,
        continuous_session=True, inter_turn_pause_seconds=args.pause_seconds if args.split_turns else None,
        image=str(image), image_sha256=sha256_file(image),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256_file(checkpoint),
        input_modality='audio', decoding='audio_sampling_text_greedy' if args.audio_only_sampling else ('sampling' if args.sampling else 'greedy'),
        sampling_config=({**SAMPLING_CONFIG, 'text_temperature': 0.0} if args.audio_only_sampling else (SAMPLING_CONFIG if args.sampling else None)),
        question_audio_manifest=str(audio_manifest), question_audio_manifest_sha256=sha256_file(audio_manifest),
        input_turns_json=str(args.turns_json.resolve()) if args.turns_json else None,
        input_audio_source=example.metadata.get('input_audio_source', 'evaluation TTS'),
        user_audio=str(audio), audio_channels={'left': 'user TTS', 'right': 'model-generated audio'},
        assistant_audio_peak=float(np.max(np.abs(assistant))),
        saved_stereo_duration_sec=length/24000,
        note='Single-image diagnostic, no ASR input and no assistant TTS/teacher forcing.')
    if args.depth_delta:
        result['depth_adaptation'] = {'path': str(delta_path), 'sha256': sha256_file(delta_path),
                                     'step': marker['step'], 'base_checkpoint_sha256': sha256_file(checkpoint)}
    (out / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (out / 'response.txt').write_text('\n\n'.join(f"{t['turn']}問目：{t['question']}\n応答：{t['answer_text']}" for t in timeline)+
        '\n全生成テキスト：'+result['generated_text_all']+'\n')
    html = '<!doctype html><meta charset="utf-8"><title>画像・音声応答テスト</title>'
    html += '<h1>画像・音声応答テスト（'+result['decoding']+'）</h1><img style="max-width:640px" src="'+os.path.relpath(image, out)+'">'
    import html as escaping
    for turn in timeline:
        html += '<h2>'+str(turn['turn'])+'問目</h2><p>質問：'+escaping.escape(turn['question'])+'</p><p>生成応答：'+escaping.escape(turn['answer_text'])+'</p>'
    html += '<p>左：ユーザー／右：モデルの生成音声（ヘッドホン推奨）</p><audio controls src="conversation_stereo.wav"></audio>'
    (out / 'review.html').write_text(html)
    print(json.dumps({'turns': timeline, 'all_text':result['generated_text_all'],
        'stereo':str(out / 'conversation_stereo.wav')}, ensure_ascii=False), flush=True)
    return runtime


if __name__ == '__main__':
    main()
