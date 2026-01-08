import torch
import os
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from collections import defaultdict

from jmoshivis.models.image_projection import ImageProjection
from jmoshivis.config.kyuteye_config import KyuteyeConfig


# =========================
# ユーティリティ関数 (変更なし)
# =========================
def strip_prefix(sd: dict, prefix: str) -> dict:
    """Return a new dict with `prefix` removed from keys that start with it."""
    if prefix and not prefix.endswith("."):
        prefix = prefix + "."
    out = {}
    for k, v in sd.items():
        if prefix == "" or k.startswith(prefix):
            out[k[len(prefix):] if prefix else k] = v
    return out


def find_best_prefix_for_module(sd: dict, expected_keys: set, max_depth: int = 4):
    """
    Find a prefix in sd keys such that stripping it maximizes overlap with expected_keys.
    """
    keys = list(sd.keys())
    if not keys: return "", 0, {}

    candidates = {""}
    for k in keys:
        parts = k.split(".")
        for n in range(1, min(max_depth, len(parts)) + 1):
            candidates.add(".".join(parts[:n]))

    best_prefix = None
    best_score = -1
    best_sd = None

    for p in candidates:
        sd2 = strip_prefix(sd, p)
        score = len(set(sd2.keys()) & expected_keys)
        if score > best_score:
            best_score = score
            best_prefix = p
            best_sd = sd2

    return best_prefix, best_score, best_sd


def load_local_checkpoint(path, device):
    """ローカルファイルをロードしてstate_dictを返す簡易ヘルパー"""
    print(f"Loading local weights from: {path}")
    if path.endswith(".safetensors"):
        return load_file(path, device=device)
    else:
        # .pt / .pth の場合
        data = torch.load(path, map_location=device)
        # PTL等で保存された場合 "state_dict" キーに入っていることが多い
        if isinstance(data, dict) and "state_dict" in data:
            return data["state_dict"]
        return data

# =========================
# 設定
# =========================
device = "cpu"
local_weights_path = "/workspace/j-moshivis/model_merged_bf16.safetensors"

# =========================
# ロード処理
# =========================
print("--- Loading Raw Weights ---")
# 1. HF
hf_path = hf_hub_download(repo_id="kyutai/moshika-vis-pytorch-bf16", filename="model.safetensors")
hf_raw = load_file(hf_path, device=device)
hf_keys = list(hf_raw.keys())
print(f"HF Raw Keys: {len(hf_keys)}")

# 2. Local
if local_weights_path.endswith(".safetensors"):
    local_raw = load_file(local_weights_path, device=device)
    local_keys = list(local_raw.keys())
else:
    dat = torch.load(local_weights_path, map_location=device)
    local_raw = dat["state_dict"] if isinstance(dat, dict) and "state_dict" in dat else dat
    local_keys = list(local_raw.keys())
print(f"Local Raw Keys: {len(local_keys)}")

# =========================
# プレフィックスの自動推定と正規化
# =========================
def guess_best_prefix(keys_list, target_set_sample):
    """
    keys_list の中から、target_set_sample に最も多くヒットするようなプレフィックスを探す
    """
    candidates = [""]
    # 最初のいくつかのキーを使って候補を作る
    for k in keys_list[:10]:
        parts = k.split(".")
        for i in range(1, len(parts)):
            candidates.append(".".join(parts[:i]) + ".")
    
    best_p = ""
    best_hit = -1
    
    # ターゲット側の代表的なキー末尾（suffix）をセットにしておく
    target_suffixes = {k.split(".")[-1] for k in list(target_set_sample)[:100]}

    for p in candidates:
        # プレフィックスpを除去したとき、ターゲットのsuffixセットにどれくらい含まれるか（簡易チェック）
        # ※厳密には全数チェックが良いが、ここではアライメント用
        hits = 0
        for k in keys_list[:100]:
            normalized = k[len(p):] if k.startswith(p) else k
            if normalized.split(".")[-1] in target_suffixes:
                hits += 1
        if hits > best_hit:
            best_hit = hits
            best_p = p
    return best_p

print("\n--- Aligning Prefixes ---")
# 相互にベストなプレフィックスを探る（簡易ロジック）
# 実際には「HFにあってLocalにない」を知りたいので、HF側のキーをLocalに合わせてみる
local_keys_set = set(local_keys)

# 1. HFキーから取り除くべきプレフィックスを探す
# 最も単純に「HFのキーを削ってLocalのキーセットにヒットする数」を最大化する
best_hf_prefix = ""
max_overlap = -1

candidates = ["", "model.", "image_prefix.", "visual_encoder.", "backbone."]
# HFのキー構造から候補を追加
if hf_keys:
    first_key = hf_keys[0]
    parts = first_key.split(".")
    for i in range(1, len(parts)):
        candidates.append(".".join(parts[:i]) + ".")

for p in set(candidates):
    # プレフィックスを剥がしたセットを作成
    stripped_hf = {k[len(p):] if k.startswith(p) else k for k in hf_keys}
    overlap = len(stripped_hf & local_keys_set)
    if overlap > max_overlap:
        max_overlap = overlap
        best_hf_prefix = p

print(f"Best detected HF prefix to strip: '{best_hf_prefix}'")
print(f"Overlap with Local keys: {max_overlap} / {len(local_keys)}")

# 正規化されたセットを作成
normalized_hf_keys = {k[len(best_hf_prefix):] if k.startswith(best_hf_prefix) else k for k in hf_keys}

# =========================
# 差分抽出 (Strict)
# =========================
# HFにはあるが、Localにはないキー（正規化後）
missing_keys = normalized_hf_keys - local_keys_set

print("\n" + "="*40)
print(f" Strict Missing Keys Check: {len(missing_keys)} keys")
print("="*40)

if len(missing_keys) > 0:
    # プレフィックスごとに集計して表示
    grouped = defaultdict(list)
    for k in missing_keys:
        # 最上位の階層名を取得
        parts = k.split(".")
        prefix = parts[0]
        if len(parts) > 1:
             # もう一段階詳しく
             prefix = f"{parts[0]}.{parts[1]}"
        grouped[prefix].append(k)

    for prefix, keys in sorted(grouped.items()):
        print(f"\n[{prefix}]: {len(keys)} keys missing")
        for sample in sorted(keys)[:-1]:
            print(f"  - {sample}")
        if len(keys) > 3:
            print(f"    ... and {len(keys)-3} more")
else:
    print("No missing keys found after strict alignment.")
    print("Check if Local has EXTRA keys instead?")
    extra_keys = local_keys_set - normalized_hf_keys
    print(f"Extra keys in Local: {len(extra_keys)}")