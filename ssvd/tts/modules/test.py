from transformers import AutoTokenizer, AutoModelForMaskedLM

import os

model_id = "ku-nlp/deberta-v2-large-japanese-char-wwm"
hf_token = os.environ.get("HF_TOKEN")

# 初回は自動で Hugging Face からダウンロードされる
tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
model = AutoModelForMaskedLM.from_pretrained(model_id, token=hf_token)

print("✅ Model loaded")