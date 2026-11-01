# ローカルVLMによる画像対話生成スクリプト (Claude CLI版と同一プロンプト・同一改善を適用)
#
# 使い方 (Qwen3-VL-30B AWQ / vLLMバックエンド):
#   CUDA_VISIBLE_DEVICES=1 CUDA_HOME=/gpu-server/yamamoto/miniforge3/envs/vlm \
#   PATH=/gpu-server/yamamoto/miniforge3/envs/vlm/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
#   /gpu-server/yamamoto/miniforge3/envs/vlm/bin/python ssvd/dialogue/generate_local_vlm.py \
#       --model QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ --backend vllm \
#       --meta data/images/filtered_phase2.jsonl \
#       --out data/dialogues/qwen3vl_30b_main.jsonl
#
# - Claude版と同じレジストリ再開: data/dialogues/*.jsonl の全uidをスキャンし、
#   どのモデルで生成済みの画像も再生成しない
# - uidごとの切り出しスタイル注入・内容リーク禁止 (prompt_utils.build_instruction)
# - vLLMはバッチ推論 (--batch_size) で高速化
# - 各レコードに generator フィールド (モデル名) を記録

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

SSVD_DIR = Path(__file__).resolve().parents[1]
if str(SSVD_DIR) not in sys.path:
    sys.path.insert(0, str(SSVD_DIR))

from review_store import (  # noqa: E402
    image_file_sha256,
    requires_review,
    review_metadata,
)

from prompt_utils import (
    ADAPTIVE_PROMPT_VERSION,
    build_instruction,
    collect_done_keys,
    collect_done_uids,
    extract_auxiliary_task,
)
from task_profiles import (
    NORMAL_STRATEGIES,
    TASK_MODES,
    auxiliary_verification_status,
    load_task_entries,
    public_grounding,
    selected_auxiliary_grounding,
    validate_task_dialogue,
)


def parse_dialogue(text: str):
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(text[start:end + 1])
            dialogue = data.get("dialogue")
            if isinstance(dialogue, list) and len(dialogue) >= 4:
                if all(isinstance(t, dict)
                       and t.get("speaker") in ("ユーザー", "アシスタント")
                       and t.get("text")
                       for t in dialogue):
                    return dialogue
        except json.JSONDecodeError:
            pass
    # フォールバック: 「ユーザー: ...」形式の行を拾う (壊れたJSONにも部分対応)
    dialogue = []
    for m in re.finditer(
            r'"?(ユーザー|アシスタント)"?\s*[:：]\s*"?(?:\{"text"\s*[:：]\s*")?([^"\n{}]+)',
            text):
        utt = m.group(2).strip().strip('」「,')
        if len(utt) >= 3:
            dialogue.append({"speaker": m.group(1), "text": utt})
    return dialogue if len(dialogue) >= 4 else None


def load_entries(
    metas,
    registry_dir,
    max_samples,
    registry_glob="*.jsonl",
    task_mode="normal",
    normal_strategy="standard",
    registry_scope="task",
    shard=0,
    num_shards=1,
):
    entries, selection_stats = load_task_entries(
        metas, task_mode, normal_strategy
    )
    if num_shards < 1:
        raise ValueError("num_shards must be at least 1")
    if not 0 <= shard < num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num_shards")
    entries = entries[shard::num_shards]
    selection_stats["eligible_in_shard"] = len(entries)
    if registry_scope == "task":
        done = collect_done_keys(registry_dir, registry_glob)
        entries = [
            entry
            for entry in entries
            if (entry["uid"], entry["task_mode"]) not in done
        ]
    elif registry_scope == "uid":
        done = collect_done_uids(registry_dir, registry_glob)
        entries = [entry for entry in entries if entry["uid"] not in done]
    else:
        raise ValueError(f"unsupported registry scope: {registry_scope}")
    if max_samples is not None:
        entries = entries[:max_samples]
    return entries, len(done), selection_stats


def write_result(fout, entry, answer, model_name, counter):
    dialogue = parse_dialogue(answer)
    if dialogue is None:
        print(f"⚠️ parse failed: {entry['uid'][:12]}", flush=True)
        return 0
    if len(dialogue) > 12:  # ルール(8〜12ターン)超過分は切り詰め
        dialogue = dialogue[:12]
        while dialogue and dialogue[-1]["speaker"] != "アシスタント":
            dialogue.pop()
    auxiliary_task = None
    if entry.get("normal_strategy") == "vlm_adaptive":
        auxiliary_task = extract_auxiliary_task(answer)
    valid, reason = validate_task_dialogue(entry, dialogue, auxiliary_task)
    if not valid:
        print(f"⚠️ validation failed: {entry['uid'][:12]}: {reason}", flush=True)
        return 0
    record = {
        "uid": entry["uid"], "image": entry["path"], "dialogue": dialogue,
        "generator": model_name,
        "task_mode": entry["task_mode"],
    }
    if entry.get("normal_strategy") == "vlm_adaptive":
        record["normal_strategy"] = "vlm_adaptive"
        record["prompt_version"] = ADAPTIVE_PROMPT_VERSION
        record["auxiliary_task"] = dict(auxiliary_task)
        verification_status = auxiliary_verification_status(auxiliary_task)
        if verification_status is not None:
            record["verification_status"] = verification_status
    grounding = public_grounding(entry)
    if grounding is None:
        grounding = selected_auxiliary_grounding(entry, auxiliary_task)
    if grounding is not None:
        record["grounding"] = grounding
    if entry.get("split_group_key") is not None:
        record["split_group_key"] = entry["split_group_key"]
    if requires_review(record):
        try:
            record["image_sha256"] = image_file_sha256(
                entry["absolute_image_path"]
            )
        except OSError as error:
            print(
                f"⚠️ review image hash failed: {entry['uid'][:12]}: {error}",
                flush=True,
            )
            return 0
        record.update(review_metadata(record))
    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
    fout.flush()
    print(f"✅ {counter} {entry['uid'][:12]}", flush=True)
    return 1


def run_vllm(args, entries, fout):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=16384,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
        # 高解像度写真(4000px級)は画像トークンが1万超になるため上限を設定
        mm_processor_kwargs={"max_pixels": 1280 * 28 * 28},
    )
    sampling = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500)

    # 画像ロードはスレッド並列 + 次バッチを生成中に先読みして GPU の待ちを無くす
    from concurrent.futures import ThreadPoolExecutor

    def prep_batch(batch):
        def load(e):
            try:
                return e, Image.open(e["absolute_image_path"]).convert("RGB")
            except Exception as ex:
                print(f"⚠️ image load failed: {e['uid'][:12]}: {ex}", flush=True)
                return e, None
        conversations, valid = [], []
        with ThreadPoolExecutor(8) as loader:
            for e, image in loader.map(load, batch):
                if image is None:
                    continue
                conversations.append([{
                    "role": "user",
                    "content": [
                        {"type": "image_pil", "image_pil": image},
                        {
                            "type": "text",
                            "text": build_instruction(
                                e["uid"],
                                e["task_mode"],
                                e.get("grounding"),
                                e.get("normal_strategy", "standard"),
                                e.get("optional_groundings"),
                            ),
                        },
                    ],
                }])
                valid.append(e)
        return conversations, valid

    ok = 0
    prefetcher = ThreadPoolExecutor(1)
    fut = prefetcher.submit(prep_batch, entries[:args.batch_size])
    for i in range(0, len(entries), args.batch_size):
        conversations, valid = fut.result()
        nxt = i + args.batch_size
        if nxt < len(entries):
            fut = prefetcher.submit(prep_batch, entries[nxt:nxt + args.batch_size])
        if not conversations:
            continue
        outs = llm.chat(conversations, sampling)
        for e, out in zip(valid, outs):
            ok += write_result(fout, e, out.outputs[0].text, args.model,
                               f"{ok + 1}/{len(entries)}")
    return ok


def run_hf(args, entries, fout):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    def generate_content(content: str, images: list) -> str:
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        encoding = processor(text=text, images=images, return_tensors="pt").to(model.device)
        output = model.generate(
            **encoding, max_new_tokens=1500, temperature=0.7, do_sample=True)
        return processor.decode(
            output[0, len(encoding["input_ids"][0]):], skip_special_tokens=True)

    ok = 0
    for e in entries:
        image = Image.open(e["absolute_image_path"]).convert("RGB")
        try:
            answer = generate_content(
                f"<image>\n{build_instruction(e['uid'], e['task_mode'], e.get('grounding'), e.get('normal_strategy', 'standard'), e.get('optional_groundings'))}",
                [image],
            )
        except Exception as ex:
            print(f"⚠️ generation failed: {e['uid'][:12]}: {ex}", flush=True)
            continue
        ok += write_result(fout, e, answer, args.model, f"{ok + 1}/{len(entries)}")
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--meta", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--shard", type=int, default=0,
                        help="0-based shard index applied after task filtering")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="number of disjoint deterministic input shards")
    parser.add_argument(
        "--task-mode",
        choices=TASK_MODES,
        default="normal",
        help="normal/counting/ocrを明示指定。autoはcapabilityから行ごとに推定",
    )
    parser.add_argument(
        "--normal-strategy",
        choices=NORMAL_STRATEGIES,
        default="standard",
        help="normal時の生成戦略。vlm_adaptiveは画像ごとに補助count/OCRを判断",
    )
    parser.add_argument("--registry_dir", default=None,
                        help="生成済みUIDを走査するディレクトリ (デフォルト: --out の親)")
    parser.add_argument("--registry_glob", default="*.jsonl",
                        help="レジストリ対象ファイルのパターン。例: 'claude_*.jsonl' で"
                             "Sonnet出力のみを生成済みとみなし、他モデル生成分を再生成(置き換え)できる")
    parser.add_argument(
        "--registry-scope",
        choices=("task", "uid"),
        default="task",
        help="taskは(uid, task_mode)、uidはタスクを問わず既存UIDを生成済みとみなす",
    )
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry = Path(args.registry_dir) if args.registry_dir else out_path.parent
    entries, n_done, selection_stats = load_entries(
        args.meta,
        registry,
        args.max_samples,
        args.registry_glob,
        args.task_mode,
        args.normal_strategy,
        args.registry_scope,
        args.shard,
        args.num_shards,
    )
    print(f"[info] task selection: {json.dumps(selection_stats, ensure_ascii=False)}", flush=True)
    print(f"[info] generating for {len(entries)} images with {args.model} "
          f"(shard {args.shard}/{args.num_shards}; "
          f"{n_done} already done in registry {registry})", flush=True)
    if not entries:
        sys.exit(3)  # 全画像生成済み

    with out_path.open("a", encoding="utf-8") as fout:
        if args.backend == "vllm":
            ok = run_vllm(args, entries, fout)
        else:
            ok = run_hf(args, entries, fout)
    print(f"Done: {ok}/{len(entries)} succeeded", flush=True)


if __name__ == "__main__":
    main()
