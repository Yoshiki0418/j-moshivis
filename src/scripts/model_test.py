import torch
from PIL import Image

from jmvis.models.jm_vis_model import JMoshiVisModel
from jmvis.data.datasets import JMVisSingleSampleDataset
from jmvis.data.collate import JMVisCollator
from jmvis.utils.audio_codec import encode as mimi_encode

# ======================
# 設定（学習時と揃える）
# ======================
moshi_name = "kyutai/moshiko-pytorch-bf16"
vision_name = "google/siglip-so400m-patch14-384"
jm_ckpt = None
adapter_ckpt = "/workspace/src/checkpoints/adapter_final.pt"

print("Loading JMoshiVis model...")
model = JMoshiVisModel(
    moshi_name=moshi_name,
    jm_checkpoint=jm_ckpt,
    vision_name=vision_name,
    freeze_backbone=True,
    freeze_vision=True,
)
tokenizer = model.tokenizer
collator = JMVisCollator(tokenizer)

# Adapter load
print(f"Loading adapter checkpoint: {adapter_ckpt}")
adapter_state = torch.load(adapter_ckpt, map_location="cpu")
model.adapter.load_state_dict(adapter_state, strict=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

# ======================
# インタラクティブループ
# ======================
print("🟢 Interactive test mode started!")
print("Type your input (or 'exit' to quit).")
print("※画像を使いたい場合は、image=<path> の形式で指定してください。")

current_image = None

def greedy_decode(model, tokenizer, batch, max_new_tokens=64):
    # 入力を初期化
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                pixel_values=batch.get("pixel_values", None),
                attention_mask=attention_mask,
            )
        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)  # 直近トークン
        input_ids = torch.cat([input_ids, next_token], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token)], dim=1
        )

        # もしEOSが出たら終了
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    if user_input.startswith("image="):
        path = user_input.split("=", 1)[1].strip()
        try:
            current_image = Image.open(path).convert("RGB")
            print(f"[OK] Image loaded: {path}")
        except Exception as e:
            print(f"[Error] Failed to load image: {e}")
        continue

    # ==== Dataset と同じ形式に変換 ====
    sample = {
        "text": user_input,
        "image": current_image,
        "audio": None,  # 今回は音声なし
    }
    dataset = JMVisSingleSampleDataset(
        tokenizer=tokenizer,
        text=user_input,
        image=current_image,
        audio=None,
        audio_tokenizer=mimi_encode,
    )
    features = [dataset[0]]
    batch = collator(features)

    # audio_tokens は model.forward が受け取れないので削除
    if "audio_tokens" in batch:
        del batch["audio_tokens"]

    batch = {k: v.to(device) for k, v in batch.items()}

    # ==== forward & greedy decode ====
    with torch.no_grad():
        response = greedy_decode(model, tokenizer, batch, max_new_tokens=64)
    print(f"Model: {response}")
