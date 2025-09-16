import sounddevice as sd
import torch
from jmvis.utils.audio_codec import encode as mimi_encode, decode as mimi_decode
from transformers import AutoModelForCausalLM

moshi_name = "kyutai/moshiko-pytorch-bf16"
device = "cuda" if torch.cuda.is_available() else "cpu"

# モデルロード
model = AutoModelForCausalLM.from_pretrained(
    moshi_name,
    torch_dtype=torch.float16,
    device_map="auto"
).eval()

# ストリーミング設定
sr = 16000
block_size = 1024  # 1ブロックごとのサンプル数
context = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)

def callback(indata, frames, time, status):
    global context
    if status:
        print(status)
    # 音声をトークン化
    tokens = mimi_encode(torch.tensor(indata[:,0], dtype=torch.float32))
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    context = torch.cat([context, tokens], dim=1)

    # モデルに逐次入力
    with torch.no_grad():
        outputs = model(input_ids=context)
        logits = outputs.logits
    pred_token = torch.argmax(logits[:, -1, :], dim=-1).item()

    # 音声デコードしてスピーカーに出力
    out_wav = mimi_decode([pred_token])
    sd.play(out_wav, sr)

# マイク入力を開始
with sd.InputStream(channels=1, samplerate=sr, blocksize=block_size, callback=callback):
    print("🎤 Speak into the microphone (Ctrl+C to stop)")
    import time
    while True:
        time.sleep(0.1)
