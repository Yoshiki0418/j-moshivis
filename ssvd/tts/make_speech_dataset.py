# make_speech_dataset.py

# sample_command:
# python3 make_speech_dataset.py \
#   --input /workspace/ssvd/filtered.jsonl \
#   --output /workspace/ssvd/test.jsonl \
#   --outdir /workspace/data/speech/data_stereo

import json
import random
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import argparse
import torchaudio.functional as F
from tqdm import tqdm
import re
import shutil

from modules.style_bert_vit2_wrapper import StyleBertVITS2Wrapper
from utils.sample_gap import sample_gap_ms


# === 設定 ===
TARGET_SR = 24000

# アシスタント音声モデル（固定）
assistant_model_dir = "/workspace/ssvd/tts/models/koharune-ami"

# ユーザー音声モデル（複数用意）
user_model_dirs = [
    "/workspace/ssvd/tts/models/amitaro",
    "/workspace/ssvd/tts/models/dami28",
    "/workspace/ssvd/tts/models/jvnv-F1-jp",
    "/workspace/ssvd/tts/models/jvnv-F2-jp",
    "/workspace/ssvd/tts/models/jvnv-M1-jp",
    "/workspace/ssvd/tts/models/jvnv-M2-jp",
    "/workspace/ssvd/tts/models/kouon28",
    "/workspace/ssvd/tts/models/male28",
    "/workspace/ssvd/tts/models/merge28",
    "/workspace/ssvd/tts/models/merge28_ds",
    "/workspace/ssvd/tts/models/richika_v2",
    "/workspace/ssvd/tts/models/sasayaki28",
    "/workspace/ssvd/tts/models/gakucho_ai_v2",
]

device = "cuda" if torch.cuda.is_available() else "cpu"

# === Wrapperの初期化 ===
assistant_tts = StyleBertVITS2Wrapper(model_dir=assistant_model_dir, device=device)
user_tts_wrappers = [
    StyleBertVITS2Wrapper(model_dir=mdir, device=device) for mdir in user_model_dirs
]

# === テキストフィルタ ===
def is_valid_text(text: str) -> bool:
    """
    異常テキストを検出して False を返す。
    条件:
      - 空文字
      - 長すぎる（>300文字）
      - 同一文字やパターンの繰り返し（例: occoccocc...）
      - 文字種の偏りが高い
    """
    if not text or text.strip() == "":
        return False
    if len(text) > 300:
        return False
    if re.search(r"(.)\1{10,}", text):
        return False
    if re.search(r"(occ){5,}", text):
        return False
    # 90%以上が英数字の場合も除外
    if len(re.findall(r"[a-zA-Z0-9]", text)) / max(1, len(text)) > 0.9:
        return False
    return True


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """音声を正規化（±0.99に収める）"""
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.99
    return audio.astype(np.float32)


def synthesize_dialogue(uid: str, dialogue: list, output_dir: Path, sr: int = TARGET_SR) -> dict:
    """
    1つのサンプル(uid)の対話をステレオ音声化して1ファイルにまとめる。
    左チャンネル: アシスタント, 右チャンネル: ユーザー。
    既に音声ファイルが存在する場合はスキップ。
    """
    sample_dir = output_dir / uid
    sample_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_dir / "stereo_dialogue.wav"

    # --- 既存ファイルスキップ ---
    if out_path.exists():
        return {
            "uid": uid,
            "dialogue": dialogue,
            "audio": str(out_path),
            "status": "skipped"
        }

    # --- 異常テキストチェック ---
    for turn in dialogue:
        text = turn.get("text", "")
        if not is_valid_text(text):
            print(f"⚠️ 異常テキスト検出 → スキップ: uid={uid}")
            # UIDフォルダを削除
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
                print(f"🗑️ フォルダ削除: {sample_dir}")
            return {"uid": uid, "dialogue": dialogue, "audio": None, "status": "invalid_text"}

    enriched_dialogue = []
    assistant_track = np.array([], dtype=np.float32)
    user_track = np.array([], dtype=np.float32)

    fixed_user_wrapper = random.choice(user_tts_wrappers)
    fixed_user_model_name = Path(fixed_user_wrapper.model_dir).name

    # --- 各発話を順に音声合成 ---
    for turn in dialogue:
        text = turn["text"]

        if not text:
            print(f"[スキップ] 空の発話を検出しました: uid={uid}")
            continue
        
        speaker = turn["speaker"]
        is_assistant = (speaker == "アシスタント")

        wrapper = assistant_tts if is_assistant else fixed_user_wrapper
        model_name = "koharune-ami" if is_assistant else fixed_user_model_name
        speaker_id_for_log = "assistant" if is_assistant else "user"

        # 音声合成
        current_sr, audio_np = wrapper.tts_model.infer(text=text, language="JP", speaker_id=0)

        # サンプリングレート統一
        if current_sr != sr:
            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
            audio = F.resample(audio_tensor, orig_freq=current_sr, new_freq=sr).squeeze(0).numpy()
        else:
            audio = audio_np

        audio = normalize_audio(audio)

        # 前の発話との間にランダムなギャップを挿入
        if len(assistant_track) > 0:
            gap_ms = sample_gap_ms()
            if gap_ms > 0:
                gap_samples = int(sr * gap_ms / 1000)
                silence = np.zeros(gap_samples, dtype=np.float32)
                assistant_track = np.concatenate([assistant_track, silence])
                user_track = np.concatenate([user_track, silence])

        # ステレオ構成
        silence_for_other_channel = np.zeros_like(audio)
        if is_assistant:
            assistant_track = np.concatenate([assistant_track, audio])
            user_track = np.concatenate([user_track, silence_for_other_channel])
        else:
            user_track = np.concatenate([user_track, audio])
            assistant_track = np.concatenate([assistant_track, silence_for_other_channel])

        enriched_dialogue.append({
            "speaker": speaker,
            "text": text,
            "speaker_id": speaker_id_for_log,
            "voice_model": model_name,
        })

    # --- ステレオ化 & 保存 ---
    max_len = max(len(assistant_track), len(user_track))
    assistant_track = np.pad(assistant_track, (0, max_len - len(assistant_track)))
    user_track = np.pad(user_track, (0, max_len - len(user_track)))

    stereo_audio = np.vstack((assistant_track, user_track)).T
    stereo_audio = normalize_audio(stereo_audio)
    sf.write(out_path, stereo_audio, sr)

    return {
        "uid": uid,
        "dialogue": enriched_dialogue,
        "audio": str(out_path),
        "status": "generated"
    }


def main(input_jsonl: str, output_jsonl: str, output_dir: str):
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    output_dir = Path(output_dir)

    samples = [json.loads(line) for line in input_path.open("r", encoding="utf-8")]
    total = len(samples)
    generated, skipped = 0, 0

    with output_path.open("w", encoding="utf-8") as fout:
        # tqdmで進行状況を表示
        for sample in tqdm(samples, total=total, desc="Synthesizing dialogues", unit="sample"):
            uid = sample["uid"]
            dialogue = sample["dialogue"]

            enriched_sample = synthesize_dialogue(uid, dialogue, output_dir)
            fout.write(json.dumps(enriched_sample, ensure_ascii=False) + "\n")

            if enriched_sample["status"] == "skipped":
                skipped += 1
            else:
                generated += 1

    print(f"\n🎯 完了: {generated}件, ⏭️ スキップ: {skipped}件, 合計: {total}件")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TTS audio dataset from JSONL dialogues.")
    parser.add_argument("--input", required=True, help="Filtered JSONL input file")
    parser.add_argument("--output", required=True, help="Output JSONL with audio paths")
    parser.add_argument("--outdir", required=True, help="Directory to save audio files")
    args = parser.parse_args()

    main(args.input, args.output, args.outdir)
