"""Conservative start-preserving stage-2 audio adaptation pilot. Does not call the visual trainer.

Only Depformer, its projections/embeddings and the eight audio heads are
updated. The visual/temporal networks and Mimi are immutable. All eight
assistant codebooks receive an equally weighted masked CE objective.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ('depformer.', 'depformer_in.', 'depformer_emb.',
            'depformer_text_emb.', 'audio_linears.')


def audio_parameter(name):
    return name.startswith(PREFIXES)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    os.replace(tmp, path)


def make_manifests(out, seed, train_limit=4096, eval_limit=128):
    """Subset only the original checkpoint's verified train/eval manifests."""
    record = json.loads((ROOT/'checkpoints/llmjp_v3_id0w1_run1/run.json').read_text())['metadata']
    selected = {}
    for split, limit in [('train', train_limit), ('eval', eval_limit)]:
        source = Path(record['datasets'][split]['path'])
        if sha(source) != record['datasets'][split]['sha256']:
            raise RuntimeError(f'{split} manifest differs from original run')
        rows = [json.loads(line) for line in source.open()]
        random.Random(seed + (split == 'eval')).shuffle(rows)
        accepted = []
        for row in rows:
            directory = Path(row['path']).parent
            dialogue = json.loads((directory/'dialogue.json').read_text())
            if dialogue.get('assistant_voice') != 'honoka':
                continue
            check_path = directory/'asr_check.json'
            if not check_path.is_file():
                continue
            check = json.loads(check_path.read_text())
            if check.get('n_words', 0) < 1 or check.get('mean_cer', 1.0) > 0.10 or check.get('max_cer', 1.0) > 0.30:
                continue
            segments = [s for s in dialogue['segments'] if s['speaker'] == 'ASSISTANT']
            if not segments or not any(s.get('text', '').strip() for s in segments):
                continue
            alignment = directory/'assistant_dialogue.refined.json'
            if not alignment.is_file() or not Path(row['image']).is_file():
                raise RuntimeError(f'Missing training asset: {directory}')
            accepted.append(row)
            if len(accepted) == limit:
                break
        if len(accepted) != limit:
            raise RuntimeError(f'Insufficient honoka examples: {split} {len(accepted)}')
        selected[split] = accepted
        target = out/f'{split}.jsonl'
        target.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in accepted))
    if {r['uid'] for r in selected['train']} & {r['uid'] for r in selected['eval']}:
        raise RuntimeError('Train/eval UID overlap')
    return {k: {'count': len(v), 'sha256': sha(out/f'{k}.jsonl')} for k,v in selected.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/llmjp_v3_id0w1_run1/final_step_20000.safetensors')
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--start-crop-ratio', type=float, default=0.75)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--duration', type=float, default=20.0)
    p.add_argument('--lr', type=float, default=5e-7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--eval-examples', type=int, default=10)
    p.add_argument('--resume', type=Path)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    if not 0 <= args.start_crop_ratio <= 1:
        p.error('--start-crop-ratio must be in [0,1]')
    if min(args.steps, args.accumulation, args.save_every, args.eval_examples) < 1 or args.duration <= 0 or args.lr <= 0:
        p.error('steps/accumulation/duration/lr/intervals must be positive')
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = dict(vars(args), run_dir=str(out), checkpoint=str(args.checkpoint.resolve()),
                  resume=None, prepare_only=False, script_sha256=sha(__file__),
                  source_checkpoint_sha256=sha(args.checkpoint),
                  objective='equal masked CE for assistant codebooks 0..7; audio only',
                  speech_ratio=1.0, trainable_prefixes=list(PREFIXES),
                  crop_policy='deterministic mixture: conversation start / random crop',
                  timing_note='Preserve user-first starts; no inference muting; timing must be evaluated autoregressively',
                  data_quality_filter={'assistant_voice':'honoka', 'mean_cer_max':0.10,
                                       'utterance_cer_max':0.30,
                                       'note':'ASR-based heuristic; not human pronunciation certification'})
    config = json.loads(json.dumps(config, default=str))
    if (out/'config.json').exists():
        previous = json.loads((out/'config.json').read_text())
        if not (out/'audit.json').exists() and args.resume is None:
            # Preparing manifests is not a training run. Refresh only the
            # script fingerprint while developing, never a trained run.
            previous['script_sha256'] = config['script_sha256']
        if previous != config:
            raise RuntimeError('Run configuration mismatch; choose a new run directory')
        write_json(out/'config.json', config)
        if (out/'audit.json').exists() and args.resume is None:
            raise RuntimeError('Existing training run: explicit --resume required')
    else:
        write_json(out/'config.json', config)
        write_json(out/'dataset.json', make_manifests(out, args.seed))
    if args.prepare_only:
        print(json.dumps(config, indent=2), flush=True)
        return

    os.environ.setdefault('NO_TORCH_COMPILE', '1')
    os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
    import numpy as np
    import sentencepiece as spm
    import torch
    import yaml
    from safetensors.torch import save_file, save_model, load_file
    from moshi.models.loaders import get_mimi
    from jmoshivis.config.kyuteye_config import KyuteyeConfig
    from jmoshivis.models.loaders import get_moshi_vis_train
    from jmoshivis.datasets.interleaver import Interleaver, InterleavedTokenizer
    from jmoshivis.datasets.datasets import parse_data_sources, get_dataset_iterator
    from jmoshivis.trainer import require_batched_image_conditions
    from jmoshivis.loss import compute_loss_with_mask
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(8)
    device = torch.device('cuda:0')
    free, total = torch.cuda.mem_get_info(device)
    print(f'GPU free={free/2**30:.1f}GiB total={total/2**30:.1f}GiB', flush=True)
    if free < 35 * 2**30:
        raise RuntimeError('Insufficient free GPU memory; no other processes are stopped')
    kyuteye = KyuteyeConfig(**yaml.safe_load((ROOT/'j-moshivis/configs/j-moshi-vis.yaml').read_text()))
    print('Loading source checkpoint', flush=True)
    model, image = get_moshi_vis_train(kyuteye, moshivis_weight=str(args.checkpoint),
        device=device, dtype=torch.bfloat16, freeze_backbone=True,
        depformer_config={'train':True, 'layers':'all', 'train_embedding':True, 'train_linear':True},
        cross_attention_config={'visual_adapter_init':'inherit_visual_adapter'})
    # The shared training loader relies on Accelerator.prepare to move some
    # submodules. This standalone loop must perform that transfer explicitly.
    model.to(device=device); image.to(device=device)
    # Override the visual trainer's selective-unfreeze policy completely.
    for name, param in model.named_parameters():
        param.requires_grad_(audio_parameter(name))
        if param.requires_grad:
            param.data = param.data.float()  # FP32 master parameters/Adam states.
    image.requires_grad_(False); image.eval(); model.eval()
    model.depformer.train()
    trainable = {n:p for n,p in model.named_parameters() if p.requires_grad}
    assert trainable and all(audio_parameter(n) for n in trainable)
    assert model.dep_q == model.num_audio_codebooks_out == 8
    assert len(model.audio_linears) == 8
    assert not any(p.requires_grad for p in image.parameters())
    for cb, head in enumerate(model.audio_linears):
        assert all(p.requires_grad for p in head.parameters()), cb

    def frozen_digest():
        h = hashlib.sha256()
        for label, net in [('model',model), ('image',image)]:
            for name, param in net.named_parameters():
                if param.requires_grad: continue
                h.update((label+'.'+name).encode())
                h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    frozen_before = frozen_digest()
    audit = {'trainable_parameters':sum(p.numel() for p in trainable.values()),
             'trainable_tensors':list(trainable), 'frozen_sha256_before':frozen_before,
             'supervised_codebooks':list(range(8)), 'status':'initialized'}
    write_json(out/'audit.json',audit)
    print('TRAINABLE',audit['trainable_parameters'],'parameters; all 8 audio heads',flush=True)
    mimi = get_mimi(str(ROOT/'models/moshika-vis-pytorch-bf16/tokenizer-e351c8d8-checkpoint125.safetensors'), device=device)
    mimi.requires_grad_(False); mimi.eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(ROOT/'models/j-moshi/tokenizer_spm_32k_3.model'))
    inter = Interleaver(tokenizer,mimi.frame_rate,model.text_padding_token_id,
        model.end_of_text_padding_id,model.zero_token_id,keep_main_only=True,keep_and_shift=True,device=device)
    tok = InterleavedTokenizer(mimi,inter,args.duration)
    optimizer = torch.optim.AdamW(trainable.values(),lr=args.lr,weight_decay=0.0)
    start_step=0
    if args.resume:
        resume=args.resume.resolve()
        if resume.parent != out or not (resume/'COMPLETE.json').is_file():
            raise RuntimeError('Resume must be a completed checkpoint in this run')
        complete=json.loads((resume/'COMPLETE.json').read_text())
        for name,digest in complete['sha256'].items():
            if sha(resume/name)!=digest: raise RuntimeError('Checkpoint checksum mismatch')
        delta=load_file(str(resume/'depth.safetensors'),device='cuda:0')
        if set(delta)!=set(trainable): raise RuntimeError('Trainable parameter mismatch')
        with torch.no_grad():
            for name,param in trainable.items():param.copy_(delta[name])
        state=torch.load(resume/'state.pt',map_location='cpu',weights_only=False)
        optimizer.load_state_dict(state['optimizer']); start_step=state['step']
        torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'],device)
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if state['frozen_sha256']!=frozen_before:raise RuntimeError('Frozen weights changed')

    # Scope the crop policy to this standalone process only. The original
    # visual trainer and the previous reproducible audio run remain untouched.
    import jmoshivis.datasets.datasets as dataset_module
    original_crop = dataset_module.speech_crop_start_sec
    crop_counts = {'start': 0, 'random': 0}
    def start_preserving_crop(**kw):
        if kw['is_eval']:
            return original_crop(**kw)
        key = f"{args.seed}:{kw['epoch']}:{kw['sample_key']}:start-preserving"
        fraction = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) / 2**64
        if fraction < args.start_crop_ratio:
            crop_counts['start'] += 1
            return 0.0
        crop_counts['random'] += 1
        return original_crop(**kw)
    dataset_module.speech_crop_start_sec = start_preserving_crop

    def iterator(split, skip=0):
        sources,_=parse_data_sources(pretrain_data=str(out/f'{split}.jsonl'))
        assert len(sources)==1
        return get_dataset_iterator(sources[0],tok,image,rank=0,world_size=1,
            is_finite=split=='eval',seed=args.seed,shuffle_at_epoch=split=='train',
            image_root=None,device=device,speech_decode_workers=2,initial_skip_samples=skip)

    def forward(sample):
        codes=sample.codes.to(device)
        condition=require_batched_image_conditions([sample.condition_attributes],batch_size=1,device=device)
        with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
            src=image(condition)['cross_attention_src']
        with torch.autocast('cuda',dtype=torch.bfloat16):
            outputs=model.forward_speech(input_ids=codes,cross_attention_src=src,num_audio_codebooks=8,compute_audio=True)
            target=codes[:,model.audio_offset:model.audio_offset+8]
            mask=outputs['logits_mask']
            logits=outputs['audio_logits']
            assert logits.shape[1]==target.shape[1]==mask.shape[1]==8
            losses=[compute_loss_with_mask(logits[:,i:i+1],target[:,i:i+1],mask[:,i:i+1],mode='audio') for i in range(8)]
            if not all(mask[:,i].any() for i in range(8)): raise RuntimeError('Missing codebook supervision')
            loss=torch.stack(losses).mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite audio loss')
        return loss, [float(x.detach()) for x in losses]

    def evaluate(step):
        model.depformer.eval()
        counts=[]
        with torch.no_grad():
            for i,sample in enumerate(iterator('eval')):
                if i>=args.eval_examples:break
                _,per_cb=forward(sample);counts.append(per_cb)
        model.depformer.train()
        result={'step':step,'eval_audio_loss_per_cb':np.mean(counts,axis=0).tolist(),
                'eval_examples':len(counts),'note':'Teacher-forced audio CE, not pronunciation accuracy'}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        print('EVAL',json.dumps(result),flush=True)

    def save(step, final=False):
        frozen_after=frozen_digest()
        if frozen_after!=frozen_before:raise RuntimeError('Frozen parameter mutation detected')
        target=out/f'step_{step:05d}';tmp=out/f'.step_{step:05d}.tmp'
        if target.exists() or tmp.exists():raise RuntimeError('Refusing to overwrite checkpoint')
        tmp.mkdir()
        save_file({n:p.detach().cpu().contiguous() for n,p in trainable.items()},str(tmp/'depth.safetensors'))
        state={'optimizer':optimizer.state_dict(),'step':step,'torch_rng':torch.get_rng_state(),
               'cuda_rng':torch.cuda.get_rng_state(device),'python_rng':random.getstate(),
               'numpy_rng':np.random.get_state(),'frozen_sha256':frozen_before,
               'sample_cursor':step*args.accumulation}
        torch.save(state,tmp/'state.pt')
        write_json(tmp/'COMPLETE.json',{'step':step,'sha256':{n:sha(tmp/n) for n in ['depth.safetensors','state.pt']}})
        os.rename(tmp,target)
        write_json(out/'latest.json',{'checkpoint':str(target),'step':step})
        audit.update(status='complete' if final else 'running',step=step,frozen_sha256_after=frozen_after)
        write_json(out/'audit.json',audit)
        if final:
            class Bundle(torch.nn.Module):
                def state_dict(self,*a,**kw):
                    sd=model.state_dict(*a,**kw)
                    sd.update({'image_prefix.'+k:v for k,v in image.state_dict(*a,**kw).items()})
                    return sd
            export=out/f'final_step_{step}.safetensors'
            temp=out/f'.final_step_{step}.safetensors.tmp'
            save_model(Bundle(),str(temp));os.rename(temp,export)
            write_json(out/'export.json',{'path':str(export),'sha256':sha(export),'base_sha256':config['source_checkpoint_sha256']})
        print('SAVED',step,'frozen checksum unchanged',flush=True)

    if not args.resume:
        evaluate(start_step)
    data=iterator('train',start_step*args.accumulation)
    started=time.monotonic()
    for step in range(start_step+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        factor=min(step/25,1.0)*(0.1+0.9*0.5*(1+math.cos(math.pi*step/args.steps)))
        for group in optimizer.param_groups:group['lr']=args.lr*factor
        per_step=[]
        for micro in range(args.accumulation):
            loss,per_cb=forward(next(data));(loss/args.accumulation).backward();per_step.append(per_cb)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        assert all(p.grad is None for p in image.parameters())
        for i,head in enumerate(model.audio_linears):
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
                raise RuntimeError(f'No finite gradient in audio head {i}')
        grad_norm=torch.nn.utils.clip_grad_norm_(list(trainable.values()),1.0,error_if_nonfinite=True)
        optimizer.step()
        row={'step':step,'audio_loss_per_cb':np.mean(per_step,axis=0).tolist(),
             'lr':optimizer.param_groups[0]['lr'],'grad_norm':float(grad_norm),
             'elapsed_sec':time.monotonic()-started,'sample_cursor':step*args.accumulation,
             'crop_counts':dict(crop_counts)}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if step==1 or step%10==0:print('TRAIN',json.dumps(row),flush=True)
        if step==1:
            audit.update(status='running',step=1,all_eight_heads_have_finite_gradients=True,
                         frozen_gradients_absent=True)
            write_json(out/'audit.json',audit)
        if step%args.save_every==0 or step==args.steps:
            evaluate(step);save(step,final=step==args.steps)


if __name__=='__main__':main()

