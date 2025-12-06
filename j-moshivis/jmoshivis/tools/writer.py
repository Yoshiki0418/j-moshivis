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
    ):
        print("calling wandb log")
        data = {
            "train/loss_step": loss,
            "train/text_loss": text_loss,
            "train/audio_loss": audio_loss,
        }

        wandb.log(data, step=step)

    def finish(self) -> None:
        wandb.finish()
