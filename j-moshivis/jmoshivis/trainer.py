import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.cuda.amp import autocast
from contextlib import nullcontext
from accelerate import Accelerator

from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.loss import compute_loss_with_mask
from jmoshivis.datasets.args import TrainerArgs


class JmoshiVisTrainer:
    def __init__(
        self,
        model: MoshiVis,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args: TrainerArgs,
        accelerator: Accelerator,
        use_amp: bool = False,
    ):
        """
        MoshiVis Trainer
        - MoshiVis forward_text() を用いて学習
        - condition_attributes.image_embed を cross_attention_src として使用
        """
        self.accelerator = accelerator
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.args = args
        self.use_amp = use_amp and torch.cuda.is_available()

    def train_epoch(self, dataloader, epoch: int, log_interval: int = 50):
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}")

        for step, batch in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            # --- Extract inputs ---
            # batch は jmoshivis.datasets.interleaver.Batch
            # → batch.codes: torch.Size([B, D, T])
            # → batch.condition_attributes: Optional[ConditionAttributes]
            codes = batch.codes.to(self.device)

            # --- optional cross-attention src ---
            cross_attention_src = None
            if batch.condition_attributes is not None:
                # image_embed が存在するなら cross-attend
                if hasattr(batch.condition_attributes, "image_embed"):
                    cross_attention_src = batch.condition_attributes.image_embed.to(self.device)
                elif isinstance(batch.condition_attributes, dict) and "image_embed" in batch.condition_attributes:
                    cross_attention_src = batch.condition_attributes["image_embed"].to(self.device)

            # --- Forward pass ---
            with self.accelerator.autocast():
                # MoshiVis forward_text は Text + Audioトークンの埋め込み系列を入力とする
                # _, text_logits, _ = self.model.forward_text(
                #     input_ids=codes,
                #     cross_attention_src=cross_attention_src,
                # )
                outputs = self.model.forward_speech(
                    input_ids=codes,
                    cross_attention_src=cross_attention_src,
                )

                # text_logits: [B, 1, T, vocab_size]
                # 教師信号: text部分（= input_ids[:, 0, :]）を参照
                text_loss = compute_loss_with_mask(
                    outputs["text_logits"],
                    codes[:, : self.model.audio_offset],
                    outputs["text_logits_mask"],
                    mode="text",
                    text_padding_weight=self.args.text_padding_weight,
                    text_padding_ids={
                        self.model.text_padding_token_id,
                        self.model.end_of_text_padding_id,
                    },
                )
                audio_loss = compute_loss_with_mask(
                    outputs["audio_logits"],
                    codes[:, self.model.audio_offset : self.model.audio_offset + self.model.dep_q],
                    outputs["logits_mask"],
                    mode="audio",
                    first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
                )

                mb_loss = text_loss + audio_loss

            # --- Backprop ---
            self.accelerator.backward(mb_loss)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += mb_loss.item()
            avg_loss = total_loss / (step + 1)
            if step % log_interval == 0:
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} | Avg Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, path: str):
        """AMP安全にcheckpointを保存"""
        torch.save(self.model.state_dict(), path)
        print(f"💾 Saved checkpoint: {path}")
