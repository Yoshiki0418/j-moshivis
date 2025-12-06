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
from jmoshivis.tools import WandBMetricsWriter
from jmoshivis.models.image_projection import ImageProjection


class MoshiVisBundle(nn.Module):
    """MoshiVis + ImageProjection をまとめて safetensors.save_model で保存する用"""

    def __init__(self, moshi_vis: MoshiVis, image_embedder: nn.Module):
        super().__init__()
        self.moshi_vis = moshi_vis
        self.image_embedder = image_embedder

    def state_dict(self, *args, **kwargs):
        # まず MoshiVis 本体
        sd = self.moshi_vis.state_dict(*args, **kwargs)

        # ImageProjection のパラメータに image_prefix. を付けて結合
        img_sd = self.image_embedder.state_dict(*args, **kwargs)
        for k, v in img_sd.items():
            sd[f"image_prefix.{k}"] = v

        return sd


class JmoshiVisTrainer:
    def __init__(
        self,
        model: MoshiVis,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        args: TrainerArgs,
        accelerator: Accelerator,
        image_embedder: ImageProjection,
        use_amp: bool = False,
        writer: WandBMetricsWriter | None = None,
        tokenizer: any = None,
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
        self.writer = writer
        self.global_step = 0
        self.image_embedder = image_embedder
        self.tokenizer = tokenizer

    def train_epoch(self, dataloader, epoch: int, log_interval: int = 1):
        total_loss = 0.0
        total_text_acc = 0.0
        total_audio_acc = 0.0

        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}")

        for step, batch in pbar:
            global_step = self.global_step
            self.global_step += 1

            self.optimizer.zero_grad(set_to_none=True)

            # --- Extract inputs ---
            # batch は jmoshivis.datasets.interleaver.Batch
            # → batch.codes: torch.Size([B, D, T])
            # → batch.condition_attributes: Optional[ConditionAttributes]
            codes = batch.codes.to(self.device)

            # --- optional cross-attention src ---
            cross_attention_src = None

            if isinstance(batch.condition_attributes, list):
                tensors = []
                for ca in batch.condition_attributes:
                    if hasattr(ca, "tensor") and "image" in ca.tensor:
                        tensors.append(ca.tensor["image"].tensor.to(self.device))
                if tensors:
                    cross_attention_src = torch.cat(tensors, dim=0)

            if batch.condition_attributes is not None:
                # Case 1: moshi standard format (image_embed attribute)
                if hasattr(batch.condition_attributes, "image_embed"):
                    cross_attention_src = batch.condition_attributes.image_embed.to(self.device)

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

                text_logits = outputs["text_logits"]             # [B,1,T_text,V]
                audio_logits = outputs["audio_logits"]           # [B,dep_q,T_audio,V]

                with torch.no_grad():
                    # shape
                    # text_logits: [B, 1, T, V]
                    # text_target: [B, 1, T]
                    # mask: [B, 1, T]
                    text_mask = outputs["text_logits_mask"]
                    text_target = codes[:, :self.model.audio_offset]  # [B,1,T]

                    pred = text_logits.argmax(-1)   # [B,1,T]
                    correct = (pred == text_target) & text_mask       # boolean

                    # avoid ZeroDivision
                    if text_mask.sum() > 0:
                        step_text_acc = correct.sum() / text_mask.sum()
                    else:
                        step_text_acc = torch.tensor(0.0)

                # ロギング用
                total_text_acc += float(step_text_acc)

                text_target = codes[:, :self.model.audio_offset]  # [B,T_text]
                audio_target = codes[:,self.model.audio_offset:self.model.audio_offset + self.model.dep_q]

                # decoded = self.tokenizer.decode(codes[0,0].tolist())
                # print(decoded)

                # print("DEBUG: text_target[0, 0, :50]", codes[0, 0])

                # text_logits: [B, 1, T, vocab_size]
                # 教師信号: text部分（= input_ids[:, 0, :]）を参照
                text_loss = compute_loss_with_mask(
                    text_logits,
                    text_target,
                    outputs["text_logits_mask"],
                    mode="text",
                    text_padding_weight=self.args.text_padding_weight,
                    text_padding_ids={
                        self.model.text_padding_token_id,
                        self.model.end_of_text_padding_id,
                    },
                )
                audio_loss = compute_loss_with_mask(
                    audio_logits,
                    audio_target,
                    outputs["logits_mask"],
                    mode="audio",
                    first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
                )

                loss = text_loss*2 + audio_loss

            # --- Backprop ---
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            avg_loss = total_loss / (step + 1)
            avg_text_acc = total_text_acc / (step + 1)
            avg_audio_acc = total_audio_acc / (step + 1)

            # ============================
            #   tqdm update
            # ============================
            if step % log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "text_acc": f"{avg_text_acc:.3f}",
                    "audio_acc": f"{avg_audio_acc:.3f}",
                })

            # --- WandB Logging (STEP) ---
            print(f"writer:{self.writer is not None}")
            if self.writer is not None:
                self.writer.log_step(
                    step=global_step,
                    loss=loss.item(),
                    text_loss=text_loss.item(),
                    audio_loss=audio_loss.item(),
                )

            # --- Step-based Checkpoint Saving ---
            import os
            from safetensors.torch import save_model

            if global_step % 100 == 0 and global_step > 0:
                if self.accelerator.is_main_process:
                    ckpt_path = f"./checkpoints/step_{global_step}.safetensors"
                    os.makedirs("./checkpoints", exist_ok=True)

                    # unwrap MoshiVis 本体
                    model_to_save = self.model
                    if hasattr(model_to_save, "module"):
                        model_to_save = model_to_save.module

                    # MoshiVis + ImageProjection をまとめたバンドルを作成
                    bundle = MoshiVisBundle(model_to_save, self.image_embedder)

                    # shared weight 付きでも安全に保存できる
                    save_model(bundle, ckpt_path)

                    print(f"💾 Saved FULL safetensor checkpoint at step {global_step}: {ckpt_path}")



        # ==================================================
        #   Epoch Summary
        # ==================================================
        avg_loss = total_loss / len(dataloader)
        avg_text_acc = total_text_acc / len(dataloader)
        avg_audio_acc = total_audio_acc / len(dataloader)

        print(f"✅ Epoch {epoch} | Loss={avg_loss:.4f} | TextAcc={avg_text_acc:.3f} | AudioAcc={avg_audio_acc:.3f}")
        return avg_loss

    def save_checkpoint(self, path: str):
        """AMP安全にcheckpointを保存"""
        from safetensors.torch import save_model
        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        bundle = MoshiVisBundle(model_to_save, self.image_embedder)
        save_model(bundle, path)
        print(f"💾 Saved checkpoint (FULL): {path}")
