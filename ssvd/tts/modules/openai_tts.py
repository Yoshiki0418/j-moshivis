import base64
import os

import openai
from dotenv import load_dotenv

from .base_tts import BaseTTS

load_dotenv()


class OpenAI_TTS(BaseTTS):
    def __init__(self, model: str = "tts-1", voice: str = "alloy"):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.tts_model = model
        self.voice = voice
        self._tts_model = f"openai_{model}"  # モデル名を保持するための属性

    def generate(self, text: str) -> bytes:
        """
        テキストから音声（音声バイナリ）を生成する
        :param text: 合成したいテキスト
        :return: 音声データ（mp3形式のバイナリ）
        """
        response = openai.audio.speech.create(
            model=self.tts_model,
            voice=self.voice,
            input=text,
        )
        return response.content  # バイナリデータを返す

    def synthesize_to_base64(self, text: str) -> str:
        """
        テキストをMP3形式で音声合成し、Base64エンコードされた文字列を返す
        :param text: 合成したいテキスト
        :return: Base64文字列（mp3）
        """
        audio_bytes = self.generate(text)
        return base64.b64encode(audio_bytes).decode("utf-8")

    @property
    def model_name(self) -> str:
        """
        TTSモデルの名前を返す
        :return: モデル名
        """
        return self._tts_model


# # 使用例
# if __name__ == "__main__":
#     tts = OpenAI_TTS(model="tts-1", voice="shimmer")
#     audio_data = tts.generate("こんにちは、OpenAIのTTSを使っています。")

#     # ファイルとして保存する場合
#     with open("output.mp3", "wb") as f:
#         f.write(audio_data)
