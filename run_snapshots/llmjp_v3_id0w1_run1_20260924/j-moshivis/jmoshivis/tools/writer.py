import wandb


class WandBMetricsWriter():
    def __init__(
        self,
        project_name: str,
        model_name: str = None,
    ) -> None:
        self.project_name = project_name
        self.name = model_name

        wandb.init(project=project_name,entity="yoshi-ai", name=self.name)

    def log_step(
        self,
        step: int,
        loss: float,
        text_loss: float,
        audio_loss: float,
        # --- 詳細なテキスト精度 ---
        text_acc_global: float = None,   # 全体 (PAD含む)
        text_acc_content: float = None,  # 意味のある文字のみ (★最重要)
        text_acc_pad: float = None,      # text padding (ID 3)のみ; 旧キー互換
        # --- 詳細な音声精度 ---
        audio_acc_global: float = None,    # 全体
        audio_acc_codebook0: float = None, # Codebook 0のみ (★最重要)
        # Appended after the legacy arguments to preserve positional callers.
        text_acc_end_of_text_padding: float = None,  # end/zero padding (ID 0)のみ
        text_target_ratio_content: float = None,
        text_target_ratio_pad: float = None,  # text padding (ID 3)
        text_target_ratio_end_of_text_padding: float = None,  # ID 0
        text_prediction_ratio_content: float = None,
        text_prediction_ratio_pad: float = None,  # predicted ID 3
        text_prediction_ratio_end_of_text_padding: float = None,  # predicted ID 0
        text_content_to_pad_rate: float = None,  # content target predicted as ID 3
        text_content_to_end_of_text_padding_rate: float = None,  # content target -> ID 0
        gate_mean: float = None,  # actual last-10-layer visual gate output
    ):
        """
        詳細なメトリクスをWandBに送信します。
        Noneの項目はログから除外されます。
        """
        data = {
            "train/loss_step": loss,
            "train/text_loss": text_loss,
            "train/audio_loss": audio_loss,
        }

        # 値が渡された場合のみ辞書に追加 (Noneチェック)
        if text_acc_global is not None:
            data["train/text_acc_global"] = text_acc_global
        
        if text_acc_content is not None:
            data["train/text_acc_content"] = text_acc_content  # グラフでこれを注視！
        
        if text_acc_pad is not None:
            data["train/text_acc_pad"] = text_acc_pad
            # Explicit alias: the historical ``text_acc_pad`` key is ID 3 only.
            data["train/text_acc_text_padding"] = text_acc_pad

        if text_acc_end_of_text_padding is not None:
            data["train/text_acc_end_of_text_padding"] = text_acc_end_of_text_padding

        if text_target_ratio_content is not None:
            data["train/text_target_ratio_content"] = text_target_ratio_content

        if text_target_ratio_pad is not None:
            data["train/text_target_ratio_pad"] = text_target_ratio_pad
            data["train/text_target_ratio_text_padding"] = text_target_ratio_pad

        if text_target_ratio_end_of_text_padding is not None:
            data["train/text_target_ratio_end_of_text_padding"] = (
                text_target_ratio_end_of_text_padding
            )

        if text_prediction_ratio_content is not None:
            data["train/text_prediction_ratio_content"] = text_prediction_ratio_content

        if text_prediction_ratio_pad is not None:
            data["train/text_prediction_ratio_pad"] = text_prediction_ratio_pad
            data["train/text_prediction_ratio_text_padding"] = text_prediction_ratio_pad

        if text_prediction_ratio_end_of_text_padding is not None:
            data["train/text_prediction_ratio_end_of_text_padding"] = (
                text_prediction_ratio_end_of_text_padding
            )

        if text_content_to_pad_rate is not None:
            data["train/text_content_to_pad_rate"] = text_content_to_pad_rate
            data["train/text_content_to_text_padding_rate"] = text_content_to_pad_rate

        if text_content_to_end_of_text_padding_rate is not None:
            data["train/text_content_to_end_of_text_padding_rate"] = (
                text_content_to_end_of_text_padding_rate
            )

        if audio_acc_global is not None:
            data["train/audio_acc_global"] = audio_acc_global
        
        if audio_acc_codebook0 is not None:
            data["train/audio_acc_codebook0"] = audio_acc_codebook0 # グラフでこれを注視！

        if gate_mean is not None:
            data["train/gate_mean"] = gate_mean

        wandb.log(data, step=step)

    def finish(self) -> None:
        wandb.finish()
