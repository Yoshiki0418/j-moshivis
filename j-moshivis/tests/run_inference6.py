import torch
import torchaudio
import soundfile as sf
from pathlib import Path
import sentencepiece as spm
from huggingface_hub import hf_hub_download

# J-MoshiVis imports
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.image_projection import ImageProcessor
from jmoshivis.models.loaders import get_moshi_vis
from moshi.models.loaders import get_mimi


# ======================================================
# Config
# ======================================================
TARGET_SR = 24000
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load configs
config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/moshi-vis.yaml")
kyuteye_config = KyuteyeConfig.from_yml("/workspace/j-moshivis/configs/j-moshi-vis.yaml")

# ======================================================
# Tokenizer
# ======================================================
tokenizer = spm.SentencePieceProcessor()
tokenizer.load("/workspace/j-moshivis/jmoshivis/tokenizer_spm_32k_3.model")

# ======================================================
# MoshiVis & Mimi
# ======================================================
print("Loading models...")

mimi_weight = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="tokenizer-e351c8d8-checkpoint125.safetensors",
)

weights_path = hf_hub_download(
    repo_id="kyutai/moshika-vis-pytorch-bf16",
    filename="model.safetensors"
)

moshi_vis, image_embedder = get_moshi_vis(
    config,
    # moshi_weight=Path("/workspace/j-moshivis/checkpoints/text_token_problem.safetensors"),
    moshi_weight=weights_path,
    device=device,
    dtype=torch.bfloat16,
)

mimi = get_mimi(mimi_weight, device)

moshi_vis.eval()
mimi.eval()

print("Models loaded.")

# ======================================================
# Image encoding
# ======================================================
img_path = "/gpu-server/user/yoshiki/j-moshivis/data/speech/data_stereo/0a38527d6549fe74d89858273a1e12290b36f641993db2714a1668a5d136f881/image.jpg"

image_processor = ImageProcessor()
image_tensor = image_processor(img_path).unsqueeze(0)

with torch.no_grad():
    img_out = image_embedder(image_tensor)
    ca_src = img_out["cross_attention_src"]

print("Image encoded. ca_src:", ca_src.mean(), ca_src.std())

# ======================================================
# Audio loading
# ======================================================
wav_path = "/workspace/data/english.wav"

waveform, sr = torchaudio.load(wav_path)
waveform = waveform.mean(dim=0, keepdim=True)

if sr != TARGET_SR:
    waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)

waveform = waveform.unsqueeze(0).to(device)

# Mimi encode
codes = mimi.encode(waveform).long()
print("Audio tokens:", codes.shape)

# ======================================================
# FORWARD SPEECH TEST
# ======================================================
print("\n====================================")
print(" Running forward_speech test...")
print("====================================\n")

with torch.no_grad():
    outs = moshi_vis.lm_model.forward_speech(
        input_ids=codes.to(device),
        cross_attention_src=ca_src.to(device),
        # cross_attention_src=None,
    )

text_logits = outs["text_logits"]           # [B,1,T,V]
text_ids = text_logits.argmax(dim=-1)[0, 0].cpu().tolist()

# Remove padding tokens
clean_ids = [
    i for i in text_ids
    if i not in {
        moshi_vis.text_padding_token_id,
        moshi_vis.end_of_text_padding_id,
    }
]

decoded_text = tokenizer.decode(clean_ids)

print("==== TEXT IDS (first 100) ====")
print(text_ids[:100])

print("\n==== CLEAN IDS ====")
print(clean_ids[:100])

print("\n==== DECODED TEXT ====")
print(decoded_text)

print("\nDone.")
