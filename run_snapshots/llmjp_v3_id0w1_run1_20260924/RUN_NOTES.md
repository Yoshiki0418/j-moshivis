# llmjp_v3_id0w1_run1

Purpose: test whether increasing the ID0 end-of-padding / lexical-boundary
target weight restores non-empty autoregressive audio-input greedy responses.

- Base: `model_merged_llmjp_bf16.safetensors`
- Visual adapter: inherited
- Dataset: speech_v3, train 166,621 / eval 3,400
- Speech ratio: 0.25
- ID3 weight: 0.5
- ID0 weight: 1.0
- Physical batch: 1 on GPU 0
- Effective batch: 16 (gradient accumulation 16)
- Optimizer steps: 20,000
- Seed: 42
- Checkpoints: every 1,000 steps, retain all 20
- Expected runtime: approximately 55--65 hours

This is a fresh run from the same original base as `llmjp_v3_run1`, not a
continuation of its step 20,000 checkpoint. The old run used two ranks, while
this run uses one rank; effective batch size and per-step modality counts are
kept equal.

Primary outcome: fixed audio questions decoded autoregressively with strict
greedy, measuring non-empty rate and the ID3-vs-ID0 probability/logit margin.
Teacher-forced ID0 accuracy alone is not sufficient.
