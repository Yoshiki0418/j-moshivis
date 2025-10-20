import base64
import io
from pathlib import Path

import soundfile as sf
import style_bert_vits2.nlp.bert_models as bert_models
from style_bert_vits2.constants import Languages

from .base_tts import BaseTTS

bert_models.DEFAULT_BERT_TOKENIZER_PATHS[Languages.JP] = Path(
    "/usr/local/lib/python3.10/dist-packages/style_bert_vits2/bert/deberta-v2-large-japanese-char-wwm"
)


from style_bert_vits2.constants import (
    DEFAULT_LENGTH,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    Languages,
)
from style_bert_vits2.tts_model import TTSModel


class StyleBertVITS2Wrapper(BaseTTS):
    def __init__(self, model_dir: str, device: str = "cuda"):
        model_paths = list(Path(model_dir).glob("*.pth"))
        config_path = Path(model_dir) / "config.json"
        style_vec_path = Path(model_dir) / "style_vectors.npy"
        self.model_dir = model_dir

        print(f"モデルパス: {model_paths[0]}")
        print(f"コンフィグパス: {config_path}")
        print(f"スタイルベクトルパス: {style_vec_path}")

        self.tts_model = TTSModel(
            model_path=model_paths[0],
            config_path=config_path,
            style_vec_path=style_vec_path,
            device=device,
        )
        self.tts_model.load()

    def generate(self, text: str, output_path: str):
        sr, audio = self.tts_model.infer(
            text=text,
            language=Languages.JP,
            speaker_id=0,
            sdp_ratio=DEFAULT_SDP_RATIO,
            noise=DEFAULT_NOISE,
            noise_w=DEFAULT_NOISEW,
            length=DEFAULT_LENGTH,
        )
        import soundfile as sf

        sf.write(output_path, audio, sr)
        return output_path

    def synthesize_to_base64(self, text: str) -> str:
        sr, audio = self.tts_model.infer(text=text)
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @property
    def model_name(self) -> str:
        """
        TTSモデルの名前を返す
        :return: モデル名
        """
        return "style_bert_vit2"


# model_dir = (
#     "/workspace/ssvd/tts/models/koharune-ami"  # モデルのディレクトリを指定
# )
# device = "cuda"  # または "cpu"
# wrapper = StyleBertVITS2Wrapper(model_dir=model_dir, device=device)

# output_wav = "output_test.wav"
# test_text = "こんにちは。私はRuminaです。あなたの声で話しています。"

# wrapper.generate(text=test_text, output_path=output_wav)

# print(f"✅ 音声が生成されました: {output_wav}")
