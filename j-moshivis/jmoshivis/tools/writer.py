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
        text_acc_pad: float = None,      # PADのみ
        # --- 詳細な音声精度 ---
        audio_acc_global: float = None,    # 全体
        audio_acc_codebook0: float = None, # Codebook 0のみ (★最重要)
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

        if audio_acc_global is not None:
            data["train/audio_acc_global"] = audio_acc_global
        
        if audio_acc_codebook0 is not None:
            data["train/audio_acc_codebook0"] = audio_acc_codebook0 # グラフでこれを注視！

        wandb.log(data, step=step)

    def finish(self) -> None:
        wandb.finish()
