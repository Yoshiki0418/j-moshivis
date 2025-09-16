import sounddevice as sd
import torch
from transformers import AutoModelForCausalLM
import os

# ===== ★★★ 修正箇所 ★★★ =====
# rustymimiの代わりに、プロジェクト内のaudio_codec.pyから関数をインポートします
# このスクリプトは、リポジトリのルートディレクトリ(/workspace/src)から実行する必要があります
from jmvis.utils.audio_codec import encode as audio_encode, decode as audio_decode
# ==============================

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
# Moshiの内部処理は24kHzですが、エンコーダーが16kHzを期待するためこのままでOK
block_size = 1024
context = torch.tensor([], dtype=torch.long, device=device).unsqueeze(0)

def callback(indata, frames, time, status):
    global context
    if status:
        print(status)
    
    # 音声をトークン化
    audio_tensor = torch.tensor(indata[:, 0], dtype=torch.float32)
    tokens = audio_encode(audio_tensor) # 修正した関数名を使用
    
    if not tokens:
        return

    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    context = torch.cat([context, tokens], dim=1)

    # 長すぎるコンテキストを切り捨てる（例として過去4096トークンを保持）
    if context.shape[1] > 4096:
        context = context[:, -4096:]

    # モデルに逐次入力
    with torch.no_grad():
        # model.generateを使用して、より安定した出力を得る
        outputs = model.generate(context, max_new_tokens=1, do_sample=False)
        pred_token = outputs[0, -1].item()
        
    # 予測されたトークンをコンテキストに追加
    pred_token_tensor = torch.tensor([[pred_token]], dtype=torch.long, device=device)
    context = torch.cat([context, pred_token_tensor], dim=1)

    # 音声デコードしてスピーカーに出力
    out_wav_tensor = audio_decode([pred_token]) # 修正した関数名を使用
    if out_wav_tensor.numel() > 0:
        sd.play(out_wav_tensor.squeeze(0).cpu().numpy(), sr)

# マイク入力を開始
try:
    # 既存のrustymimiをアンインストール（念のため）
    print("既存のrustymimiをアンインストールしようと試みます（インストールされていない場合は無視されます）...")
    os.system("pip uninstall -y rustymimi")
    
    print("\n🎤 マイクに向かって話してください (Ctrl+Cで停止)")
    with sd.InputStream(channels=1, samplerate=sr, blocksize=block_size, callback=callback):
        import time
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nストリーミングを停止しました。")
except Exception as e:
    print(f"エラーが発生しました: {e}")
    print("\nPyAudioまたはPortAudioがインストールされていない可能性があります。")
    print("Ubuntu/Debianの場合: sudo apt-get install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0")
    print("condaの場合: conda install pyaudio")