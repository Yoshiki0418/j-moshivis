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
        text_acc: float = None,
        audio_acc: float = None,
    ):
        data = {
            "train/loss_step": loss,
            "train/text_loss": text_loss,
            "train/audio_loss": audio_loss,
            "train/text_acc": text_acc,
            "train/audio_acc": audio_acc,
        }

        wandb.log(data, step=step)

    def finish(self) -> None:
        wandb.finish()
