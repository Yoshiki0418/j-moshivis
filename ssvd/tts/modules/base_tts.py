from abc import ABC, abstractmethod


class BaseTTS(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        """カタログの tts_id に対応する固有キーを返す"""

    @abstractmethod
    def synthesize_to_base64(self, text: str) -> str:
        """テキストを合成して base64 文字列で返す"""
