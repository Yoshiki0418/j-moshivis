import base64
import io
import threading
from collections import defaultdict

import torch
from torch.serialization import add_safe_globals
from TTS.utils.radam import RAdam

add_safe_globals([RAdam, defaultdict, dict])

from TTS.api import TTS

from .base_tts import BaseTTS


class TTSGenerator(BaseTTS):
    _init_lock = threading.Lock()

    def __init__(self):
        self._build_tts()

    # ------------------------
    # ① モデルの初期化を関数化
    # ------------------------
    def _build_tts(self):
        use_gpu = torch.cuda.is_available()
        with self._init_lock:
            # キャッシュ再利用で 1.5 – 2 秒程度でロードされます
            self.tts = TTS(
                model_name="tts_models/ja/kokoro/tacotron2-DDC",
                gpu=use_gpu,
                progress_bar=False,
            )
        print("TTS model (re)initialized")

    def synthesize_to_base64(self, text: str, _retry: bool = True) -> str:
        """
        Tacotron2 の内部 state が壊れて RuntimeError が出た場合は
        1 回だけモデルを再構築してリトライする。
        """
        buffer = io.BytesIO()

        try:
            self.tts.tts_to_file(text, file_path=buffer)

        except Exception as e:
            print("TTS synthesis failed: %s", e)

            if not _retry:  # すでにリトライ済みなら諦める
                raise

            # 1) GPU メモリと PyTorch のキャッシュをクリア
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 2) インスタンスを作り直す
            self._build_tts()

            # 3) もう一度だけ試す
            return self.synthesize_to_base64(text, _retry=False)

        # ---------- 正常終了 ----------
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    @property
    def model_name(self) -> str:
        """
        TTSモデルの名前を返す
        :return: モデル名
        """
        return "kokoro_tts"


if __name__ == "__main__":
    test_message = "こんにちは！調子はどう？"
    tts_model = TTSGenerator()
    result = tts_model.synthesize_to_base64(test_message)
