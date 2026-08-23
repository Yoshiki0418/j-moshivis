import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.cuda.amp import autocast
from collections import deque
from contextlib import nullcontext
from accelerate import Accelerator

from jmoshivis.models.moshivis import MoshiVis
from jmoshivis.loss import compute_loss_with_mask
from jmoshivis.datasets.args import TrainerArgs
from jmoshivis.tools import WandBMetricsWriter
from jmoshivis.models.image_projection import ImageProjection


def register_nan_hooks(model):
    """
    モデル内の全レイヤーにフックを仕掛け、出力がNaNになった瞬間に
    そのレイヤー名を表示して停止させるデバッグ関数
    """
    def hook_fn(module, inputs, output):
        # 出力がTensorの場合
        if isinstance(output, torch.Tensor):
            if torch.isnan(output).any():
                print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__}")
                print(f"   Shape: {output.shape}")
                raise RuntimeError(f"NaN found in {module.__class__.__name__}")

        # 出力がタプルやリストの場合
        elif isinstance(output, (tuple, list)):
            for i, x in enumerate(output):
                if isinstance(x, torch.Tensor) and torch.isnan(x).any():
                    print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__} (Output index {i})")
                    raise RuntimeError(f"NaN found in {module.__class__.__name__}")
        
        # 出力が辞書の場合
        elif isinstance(output, dict):
            for k, v in output.items():
                if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                    print(f"\n🚨 [NaN DETECTED] Layer: {module.__class__.__name__} (Key: {k})")
                    raise RuntimeError(f"NaN found in {module.__class__.__name__}")

    for name, layer in model.named_modules():
        layer.register_forward_hook(hook_fn)
    
    print(f"👀 NaN Hunter hooks registered for {model.__class__.__name__}")
# ==========================================


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
        # if self.accelerator.is_main_process:
        #     print("🕵️‍♀️ Registering NaN Hunter Hooks...")
        #     register_nan_hooks(self.model)
        #     register_nan_hooks(self.image_embedder)

    def train_epoch(self, dataloader, epoch: int, log_interval: int = 1, DEBUG: bool = False):
        target_batch_size = 32  # ※必要に応じて調整 (64~128推奨)
        # 1GPUあたりのバッチサイズ (引数で指定されたもの)
        physical_batch_size = self.args.batch_size
        
        # GPUの枚数
        num_processes = self.accelerator.num_processes
        accumulation_steps = target_batch_size // (physical_batch_size * num_processes)
        if accumulation_steps < 1:
            accumulation_steps = 1
        
        # Acceleratorに蓄積ステップ数を伝える（お作法）
        self.accelerator.gradient_accumulation_steps = accumulation_steps

        if self.accelerator.is_main_process:
            print(f"🚀 Epoch {epoch} Start | Grad Accum: {accumulation_steps} steps")

        total_loss = 0.0
        
        # --- 移動平均用ウィンドウ ---
        loss_window = deque(maxlen=100)
        # Text
        text_acc_global_window = deque(maxlen=100)  # 全体
        text_acc_content_window = deque(maxlen=100) # 意味のある文字のみ (重要!)
        text_acc_pad_window = deque(maxlen=100)     # PADのみ
        # Audio
        audio_acc_global_window = deque(maxlen=100) # 全体
        audio_acc_cb0_window = deque(maxlen=100)    # Codebook 0のみ (重要!)
        
        # 蓄積中のステップカウント用
        processed_samples = 0

        pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", disable=not self.accelerator.is_main_process)

        self.optimizer.zero_grad(set_to_none=True) # ループ前に初期化

        for step, batch in pbar:
            # -----------------------------------------------------
            # ★ 修正: accumulate コンテキストで囲む
            # -----------------------------------------------------
            with self.accelerator.accumulate(self.model, self.image_embedder):
                # --- Extract inputs ---
                codes = batch.codes.to(self.device)

                if DEBUG:
                    print("\n" + "="*60)
                    print(f"🔍 [CHECK 1] Input Codes (Step {step})")
                    print(f"   Shape: {codes.shape} (Batch, Codebooks, Time)")
                    print(f"   Values: Min={codes.min().item()}, Max={codes.max().item()}")
                    # パディング(-1や3)だらけになっていないか確認
                    unique, counts = torch.unique(codes, return_counts=True)
                    print(f"   Top 5 tokens: {list(zip(unique[:5].tolist(), counts[:5].tolist()))}")
                    print("="*60)

                # --- 画像入力の準備 ---
                image_input = None
                if isinstance(batch.condition_attributes, list):
                    tensors = []
                    for ca in batch.condition_attributes:
                        if hasattr(ca, "tensor") and "image" in ca.tensor:
                            tensors.append(ca.tensor["image"].tensor.to(self.device))
                    if tensors:
                        image_input = torch.cat(tensors, dim=0)

                if DEBUG:
                    print(f"🔍 [CHECK 2] Image Input")
                    if image_input is not None:
                        print(f"   Shape: {image_input.shape}")
                        print(f"   Stats: Mean={image_input.mean().item():.3f}, Std={image_input.std().item():.3f}")
                    else:
                        print("   ⚠️ WARNING: No Image Input found in this batch!")

                # --- Forward pass ---
                with self.accelerator.autocast():
                    cross_attention_src = None
                    if image_input is not None:
                        embedder_out = self.image_embedder(image_input)
                        cross_attention_src = embedder_out["cross_attention_src"]

                        if DEBUG:
                            print(f"🔍 [CHECK 3] Embedder Output (Cross Attention Src)")
                            print(f"   Shape: {cross_attention_src.shape}")
                            print(f"   Stats: Mean={cross_attention_src.mean().item():.3f}, Std={cross_attention_src.std().item():.3f}")
                            print(f"   Max Val: {cross_attention_src.max().item():.3f}") # 爆発していないか(±20以内か)

                    if step == 0:
                        print(f"DEBUG: Accelerator Mixed Precision: {self.accelerator.mixed_precision}")
                        # ダミーテンソルを作って型を確認
                        with self.accelerator.autocast():
                            dummy = torch.tensor([1.0], device=self.device)
                            print(f"DEBUG: Real dtype inside autocast: {dummy.dtype}")

                    outputs = self.model(
                        input_ids=codes,
                        cross_attention_src=cross_attention_src,
                    )

                    text_logits = outputs["text_logits"]
                    audio_logits_full = outputs["audio_logits"]
                    audio_logits = audio_logits_full[:, :2, :, :].contiguous()

                    if DEBUG:
                        print(f"🔍 [CHECK 4] Outputs & Masks")
                        print(f"   Text Logits: {text_logits.shape}")
                        print(f"   Audio Logits: {audio_logits.shape}")
                        
                        # NaNチェック
                        if torch.isnan(text_logits).any():
                            print("   🚨 ERROR: Text Logits contain NaN!")
                        
                        # AudioはマスクがTrue(有効)な場所にあるNaNだけをエラーとする
                        real_audio_nan = (torch.isnan(audio_logits) & outputs["logits_mask"].unsqueeze(-1)).any()
                        
                        if real_audio_nan:
                            print("   🚨 FATAL ERROR: Audio Logits contain REAL NaN inside the mask!")
                            # ここで詳細を出して止める
                            raise RuntimeError("Audio logits exploded within the valid mask!")
                        else:
                            # 正常なNaN(パディング)はスルー
                            pass
                        
                        # Maskの確認: 全てFalseになっていないか？
                        t_mask = outputs["text_logits_mask"]
                        a_mask = outputs["logits_mask"]
                        print(f"   Text Mask Valid Ratio: {t_mask.float().mean().item():.2%}")
                        print(f"   Audio Mask Valid Ratio: {a_mask.float().mean().item():.2%}")
                        
                        if t_mask.sum() == 0:
                            print("   🚨 FATAL: Text Mask is empty! Loss will be 0.")
                        if a_mask.sum() == 0:
                            print("   🚨 FATAL: Audio Mask is empty! Loss will be 0.")

                    # --- Loss Calculation ---
                    text_target = codes[:, :self.model.audio_offset]
                    # 🚀 MoshiVis style: Use only first 2 codebooks for audio loss (faster training)
                    # Original uses all dep_q (8), but first 2 carry most of the semantic information
                    num_audio_codebooks_for_loss = 2  # MoshiVis default
                    audio_target = codes[:, self.model.audio_offset:self.model.audio_offset + num_audio_codebooks_for_loss]

                    # =========================================================================
                    # ▼ デバッグ用コード: ここから ▼
                    # =========================================================================
                    if DEBUG or (self.global_step % 100 == 0): # 毎回出すと重いので100ステップ毎などに制限
                        with torch.no_grad():
                            print(f"\n[Step {self.global_step}] Debug Inspection -------------------------")
                            
                            # 1. シェイプの確認
                            # text_logits: [B, 1, T, Vocab] 想定
                            # text_target: [B, 1, T] 想定
                            B, _, T, _ = text_logits.shape
                            
                            # 2. 予測トークン（ID）を取得 (Argmax)
                            # 確率が最大のIDを取得
                            pred_ids = torch.argmax(text_logits, dim=-1) # [B, 1, T]
                            
                            # 3. バッチ内の最初のサンプルの内容を表示
                            sample_idx = 0
                            
                            # ターゲット（正解）のID列
                            target_sample = text_target[sample_idx, 0, :].cpu().numpy()
                            # モデル予測のID列
                            pred_sample = pred_ids[sample_idx, 0, :].cpu().numpy()
                            # マスク（学習対象かどうか）
                            mask_sample = outputs["text_logits_mask"][sample_idx, 0, :].cpu().numpy()

                            print(f"Target IDs (First 50): {target_sample[:50]}")
                            print(f"Pred   IDs (First 50): {pred_sample[:50]}")
                            # print(f"Mask       (First 50): {mask_sample[:50]}")

                            # 4. (もしtokenizerを持っていれば) デコードして文字で確認
                            # self.tokenizer が Trainerにあると仮定しています
                            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                                # Maskされている部分（Paddingなど）を除外してデコードしてみる
                                valid_target = target_sample[mask_sample.astype(bool)]
                                valid_pred = pred_sample[mask_sample.astype(bool)]
                                
                                try:
                                    print(f"Target Text: {self.tokenizer.decode(valid_target)}")
                                    print(f"Pred   Text: {self.tokenizer.decode(valid_pred)}")
                                except Exception as e:
                                    print(f"Decode failed: {e}")
                            
                            print("----------------------------------------------------------\n")

                    text_loss = compute_loss_with_mask(
                        text_logits, text_target, outputs["text_logits_mask"],
                        mode="text",
                        text_padding_weight=self.args.text_padding_weight,
                        text_padding_ids={self.model.text_padding_token_id, self.model.end_of_text_padding_id},
                    )
                    # 🚀 Slice mask to first 2 codebooks (matching audio_target)
                    audio_mask = outputs["logits_mask"]
                    if audio_mask.dim() == 3 and audio_mask.size(1) > 2:
                        audio_mask = audio_mask[:, :2, :]  # [B, 2, T]
                    audio_loss = compute_loss_with_mask(
                        audio_logits, audio_target, audio_mask,
                        mode="audio",
                        first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
                    )

                    loss = 2.0 * text_loss + audio_loss
                    loss = loss / accumulation_steps

                    if DEBUG:
                        print(f"🔍 [CHECK 5] Loss Values")
                        print(f"   Text Loss: {text_loss.item():.4f}")
                        print(f"   Audio Loss: {audio_loss.item():.4f}")
                        print(f"   Total Loss: {loss.item():.4f}")
                        print("="*60 + "\n")

                    # =========================================================
                    # 🔍 [CHECK 6] Alignment & Prediction Preview (Step 0のみ)
                    # =========================================================
                    if DEBUG:
                        print("\n" + "="*60)
                        print("👀 LOGITS vs TARGET ALIGNMENT CHECK")
                        
                        # --- 1. Text Alignment ---
                        # Batch 0, Channel 0, 最初の10トークンを表示
                        t_tgt = text_target[0, 0, :15].cpu().tolist()
                        t_pred = text_logits.argmax(dim=-1)[0, 0, :15].cpu().tolist()
                        
                        print(f" [Text] Target (GT): {t_tgt}")
                        print(f" [Text] Pred (Argmax): {t_pred}")
                        
                        # --- 2. Audio Alignment ---
                        # Batch 0, Channel 0, 最初の10トークン
                        a_tgt = audio_target[0, 0, :15].cpu().tolist()
                        a_pred = audio_logits.argmax(dim=-1)[0, 0, :15].cpu().tolist()
                        
                        print(f" [Audio] Target (GT): {a_tgt}")
                        print(f" [Audio] Pred (Argmax): {a_pred}")

                        # --- 3. Data Leakage Check ---
                        # もし初期状態で「Pred」が「Target」と完全に一致していたら、
                        # 「カンニング（未来のトークンが見えている）」バグです。
                        # 逆に、ランダムな予測になっていれば正常です。
                        text_match_rate = (text_logits.argmax(-1) == text_target).float().mean().item()
                        print(f" [Check] Initial Text Accuracy: {text_match_rate:.2%} (Should be low/random, NOT 100%)")
                        print("="*60 + "\n")

                # --- Backprop (蓄積される) ---
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    # =========================================================
                    # 🔬 勾配ロギング（MHA学習確認用）
                    # =========================================================
                    if self.global_step % 50 == 0 and self.accelerator.is_main_process:
                        print(f"\n🔬 [Gradient Check] Step {self.global_step}")
                        mha_grad_found = False
                        gate_grad_found = False
                        
                        for name, param in self.model.named_parameters():
                            if param.grad is None:
                                continue
                            
                            # MHA勾配チェック (layer 0のみ)
                            if "cross_attention.mha" in name and "layers.0" in name and not mha_grad_found:
                                grad_norm = param.grad.norm().item()
                                grad_mean = param.grad.abs().mean().item()
                                print(f"   MHA ({name[:50]}...): norm={grad_norm:.8f}, mean={grad_mean:.8f}")
                                mha_grad_found = True
                            
                            # Gate勾配チェック (layer 0のみ)
                            if "gate" in name and "layers.0" in name and not gate_grad_found:
                                grad_norm = param.grad.norm().item()
                                grad_mean = param.grad.abs().mean().item()
                                print(f"   Gate ({name[:50]}...): norm={grad_norm:.8f}, mean={grad_mean:.8f}")
                                gate_grad_found = True
                            
                            if mha_grad_found and gate_grad_found:
                                break
                        
                        if not mha_grad_found:
                            print("   ⚠️ WARNING: No MHA gradient found!")
                        if not gate_grad_found:
                            print("   ⚠️ WARNING: No Gate gradient found!")

                        # 🔬 Gate値の監視（閉じていないか確認）
                        for name, param in self.model.named_parameters():
                            if "gate.alpha" in name and "layers.0" in name:
                                gate_alpha = param.data.mean().item()
                                gate_sigmoid = 1 / (1 + (-gate_alpha).__abs__())  # 簡易sigmoid
                                print(f"   Gate alpha: {gate_alpha:.4f} → sigmoid ≈ {gate_sigmoid:.4f}")
                                if gate_alpha < -6:
                                    print(f"   ⚠️ WARNING: Gate is closing! (alpha < -6)")
                                break

                        # 🔬 MHA重みの変化を確認（optimizer.step前に記録）
                        mha_weight_before = None
                        mha_param = None
                        for name, param in self.model.named_parameters():
                            if "cross_attention.mha.in_proj_weight" in name and "layers.0" in name:
                                mha_weight_before = param.data.clone()
                                mha_param = param
                                break
                    
                    # 勾配クリッピング
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.accelerator.clip_grad_norm_(self.image_embedder.parameters(), 1.0)
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)       
                    self.global_step += 1  # 更新した回数だけカウントアップ

                    # 🔬 optimizer.step後のMHA重みを確認
                    if self.global_step % 50 == 1 and self.accelerator.is_main_process:
                        if mha_weight_before is not None:
                            after = mha_param.data
                            diff = (after - mha_weight_before).abs().mean().item()
                            print(f"   📊 MHA weight change after step {self.global_step}:")
                            print(f"      Before mean: {mha_weight_before.mean().item():.10f}")
                            print(f"      After mean:  {after.mean().item():.10f}")
                            print(f"      Abs diff mean: {diff:.12f}")
                            if diff < 1e-12:
                                print(f"      ❌ CRITICAL: MHA weights did NOT change!")

                    # --- Logging (更新時のみ行う) ---
                    # 正解率計算などは負荷削減のため更新時のみでOK
                    with torch.no_grad():
                        pad_id = self.model.text_padding_token_id
                        text_mask = outputs["text_logits_mask"]
                        pred_text = text_logits.argmax(-1)
                        valid_mask = text_mask.bool()
                        is_pad = (text_target == pad_id) & valid_mask
                        is_content = (text_target != pad_id) & valid_mask
                        # (A) Global Acc
                        correct_global = (pred_text == text_target) & valid_mask
                        acc_text_global = correct_global.sum() / valid_mask.sum().clamp(min=1)

                        # (B) Content Acc (★最重要)
                        correct_content = (pred_text == text_target) & is_content
                        acc_text_content = correct_content.sum() / is_content.sum().clamp(min=1)

                        # (C) PAD Acc (楽をしてるかチェック)
                        correct_pad = (pred_text == text_target) & is_pad
                        if is_pad.sum() > 0:
                            acc_text_pad = correct_pad.sum() / is_pad.sum()
                        else:
                            # speechless バッチの場合は NaN (移動平均から除外される)
                            acc_text_pad = torch.tensor(float('nan'))

                        # Audio Acc (マスク考慮)
                        audio_mask = outputs["logits_mask"]
                        # --- 2. Audio Metrics ---
                        pred_audio = audio_logits.argmax(-1) # [B, 2, T] になっています

                        # ★重要: マスクの形状を正規化する処理
                        # audio_mask が [B, T] なのか [B, 8, T] なのかを判定して統一します
                        if audio_mask.dim() == 2:
                            # [B, T] の場合 -> [B, 2, T] に拡張して Global計算用に使う
                            # audio_target は既に [B, 2, T] なので、expand_as で合わせるのが正解
                            real_audio_mask = audio_mask.unsqueeze(1).expand_as(audio_target)
                            # Codebook 0 用はそのまま [B, T] を使う
                            cb0_mask = audio_mask
                        
                        elif audio_mask.dim() == 3:
                            # [B, 8, T] の場合 -> [B, 2, T] にスライスして使う
                            # 修正前: real_audio_mask = audio_mask  <-- これがエラーの原因
                            
                            # ▼▼▼ 修正後 ▼▼▼
                            # 予測(pred_audio)と正解(audio_target)のチャンネル数(2)に合わせる
                            if audio_mask.size(1) > audio_target.size(1):
                                real_audio_mask = audio_mask[:, :audio_target.size(1), :]
                            else:
                                real_audio_mask = audio_mask
                            # ▲▲▲ 修正ここまで ▲▲▲

                            # Codebook 0 用は 0チャンネル目を取り出して [B, T] にする
                            cb0_mask = audio_mask[:, 0, :]
                        
                        elif audio_mask.dim() == 4: # 万が一 [B, 1, 8, T] などの場合
                             # ... (同様にスライスが必要ですが、稀なケースです)
                             real_audio_mask = audio_mask.squeeze(1)[:, :audio_target.size(1), :]
                             cb0_mask = real_audio_mask[:, 0, :]

                        # (A) Global Audio Acc
                        # 形状が [B, 8, T] で揃ったので安全に計算可能
                        correct_audio = (pred_audio == audio_target) & real_audio_mask
                        if real_audio_mask.sum() > 0:
                            acc_audio_global = correct_audio.sum() / real_audio_mask.sum()
                        else:
                            # speechless バッチの場合は NaN (移動平均から除外される)
                            acc_audio_global = torch.tensor(float('nan'))

                        # (B) Codebook 0 Acc (★最重要: 骨格があっているか)
                        # Channel 0 だけを取り出す
                        cb0_target = audio_target[:, 0, :] # [B, T]
                        cb0_pred = pred_audio[:, 0, :]     # [B, T]

                        # マスクも [B, T] になっているのでエラーにならない
                        correct_cb0 = (cb0_pred == cb0_target) & cb0_mask
                        if cb0_mask.sum() > 0:
                            acc_audio_cb0 = correct_cb0.sum() / cb0_mask.sum()
                        else:
                            # speechless バッチの場合は NaN (移動平均から除外される)
                            acc_audio_cb0 = torch.tensor(float('nan'))

                        # --- Update Windows ---
                        current_loss_val = loss.item() * accumulation_steps
                        loss_window.append(current_loss_val)
                        
                        text_acc_global_window.append(float(acc_text_global))
                        text_acc_content_window.append(float(acc_text_content))
                        text_acc_pad_window.append(float(acc_text_pad))
                        
                        audio_acc_global_window.append(float(acc_audio_global))
                        audio_acc_cb0_window.append(float(acc_audio_cb0))

                        if self.global_step % log_interval == 0:
                            # 平均計算 (NaN はスキップ)
                            s_loss = sum(loss_window) / len(loss_window)
                            # NaN をスキップして平均を計算するヘルパー
                            def nanmean(arr):
                                valid = [x for x in arr if not (isinstance(x, float) and x != x)]  # NaN を除外
                                return sum(valid) / len(valid) if valid else 0.0
                            s_txt_gl = nanmean(text_acc_global_window)
                            s_txt_ct = nanmean(text_acc_content_window)
                            s_aud_gl = nanmean(audio_acc_global_window)
                            s_aud_c0 = nanmean(audio_acc_cb0_window)

                            # tqdmには重要なものだけ表示
                            pbar.set_postfix({
                                "L": f"{s_loss:.3f}",
                                "TxCt": f"{s_txt_ct:.1%}", # Text Content (ここを見る！)
                                "AuC0": f"{s_aud_c0:.1%}", # Audio CB0 (ここを見る！)
                                "AuGl": f"{s_aud_gl:.1%}",
                                "TxGl": f"{s_txt_gl:.1%}",
                            })

                            if self.writer is not None:
                                # NaN をフィルタリング (None に変換して W&B は無視するようにする)
                                def to_float_or_none(val):
                                    f = float(val)
                                    return None if (f != f) else f  # NaN check

                                self.writer.log_step(
                                    step=self.global_step,
                                    loss=current_loss_val,
                                    text_loss=text_loss.item(),
                                    audio_loss=audio_loss.item(),
                                    # Text詳細
                                    text_acc_global=to_float_or_none(acc_text_global),
                                    text_acc_content=to_float_or_none(acc_text_content),
                                    text_acc_pad=to_float_or_none(acc_text_pad),
                                    # Audio詳細
                                    audio_acc_global=to_float_or_none(acc_audio_global),
                                    audio_acc_codebook0=to_float_or_none(acc_audio_cb0),
                                )

                    # --- Save Checkpoint ---
                    if self.global_step % 5000 == 0 and self.global_step > 0:
                        if self.accelerator.is_main_process:
                            self.save_checkpoint(f"./checkpoints/step_{self.global_step}.safetensors")

        # End of Epoch
        print(f"✅ Epoch {epoch} finished.")
        return total_loss / max(1, processed_samples)

    # def train_epoch(self, dataloader, epoch: int, log_interval: int = 1):
    #     total_loss = 0.0
    #     total_text_acc = 0.0
    #     total_audio_acc = 0.0

    #     pbar = tqdm(enumerate(dataloader), desc=f"Epoch {epoch}")

    #     for step, batch in pbar:
    #         global_step = self.global_step
    #         self.global_step += 1

    #         self.optimizer.zero_grad(set_to_none=True)

    #         # --- Extract inputs ---
    #         # batch は jmoshivis.datasets.interleaver.Batch
    #         # → batch.codes: torch.Size([B, D, T])
    #         # → batch.condition_attributes: Optional[ConditionAttributes]
    #         codes = batch.codes.to(self.device)

    #         # --- 画像入力の準備 ---
    #         image_input = None

    #         if isinstance(batch.condition_attributes, list):
    #             tensors = []
    #             for ca in batch.condition_attributes:
    #                 if hasattr(ca, "tensor") and "image" in ca.tensor:
    #                     tensors.append(ca.tensor["image"].tensor.to(self.device))
    #             if tensors:
    #                 image_input = torch.cat(tensors, dim=0)

    #                 # 最初のステップ、または一定間隔で確認
    #                 if self.global_step == 1 or self.global_step % 100 == 0:
    #                     print(f"\n🔍 [DEBUG Step {self.global_step}] Image Input Check:")
    #                     print(f"   - Shape: {image_input.shape}") # [B, 3, H, W] になっているか？ (例: [B, 3, 448, 448])
    #                     print(f"   - Min: {image_input.min().item():.3f}, Max: {image_input.max().item():.3f}") # -1.0 ~ 1.0 付近か？
    #                     print(f"   - Mean: {image_input.mean().item():.3f}, Std: {image_input.std().item():.3f}")

    #                     # 画像として保存して目視確認 (最初の1枚だけ)
    #                     try:
    #                         import os
    #                         import torchvision
    #                         os.makedirs("debug_images", exist_ok=True)
                            
    #                         # 正規化を戻す (mean=0.5, std=0.5 を仮定: [-1, 1] -> [0, 1])
    #                         # ※ ImageProcessorの設定に合わせて調整してください
    #                         img_to_save = image_input[0].clone().detach().float().cpu()
    #                         img_to_save = img_to_save * 0.5 + 0.5
    #                         img_to_save = torch.clamp(img_to_save, 0, 1)
                            
    #                         save_path = f"debug_images/step_{self.global_step}.png"
    #                         torchvision.utils.save_image(img_to_save, save_path)
    #                         print(f"   - Saved debug image to: {save_path}")
    #                     except Exception as e:
    #                         print(f"   - Failed to save debug image: {e}")

    #         if image_input is None and batch.condition_attributes is not None:
    #             # Case 1: moshi standard format (image_embed attribute)
    #             if hasattr(batch.condition_attributes, "image_embed"):
    #                 pass

    #         # --- Forward pass ---
    #         with self.accelerator.autocast():
    #             cross_attention_src = None
    #             if image_input is not None:
    #                 # ここでプロジェクション層 (proj_xa) に勾配が流れる！
    #                 # forward戻り値: {"cross_attention_src": ..., "cross_attention_mask": ...}
    #                 embedder_out = self.image_embedder(image_input)

    #                 cross_attention_src = embedder_out["cross_attention_src"]
    #                 # 必要であればマスクも取得 (Pixtralなど画像サイズ可変の場合)
    #                 # cross_attention_mask = embedder_out.get("cross_attention_mask", None)

    #             # MoshiVis forward_text は Text + Audioトークンの埋め込み系列を入力とする
    #             # _, text_logits, _ = self.model.forward_text(
    #             #     input_ids=codes,
    #             #     cross_attention_src=cross_attention_src,
    #             # )
    #             outputs = self.model.forward_speech(
    #                 input_ids=codes,
    #                 cross_attention_src=cross_attention_src,
    #                 # cross_attention_mask=cross_attention_mask # 必要なら追加
    #             )

    #             text_logits = outputs["text_logits"]             # [B,1,T_text,V]
    #             audio_logits = outputs["audio_logits"]           # [B,dep_q,T_audio,V]

    #             with torch.no_grad():
    #                 # shape
    #                 # text_logits: [B, 1, T, V]
    #                 # text_target: [B, 1, T]
    #                 # mask: [B, 1, T]
    #                 text_mask = outputs["text_logits_mask"]
    #                 text_target = codes[:, :self.model.audio_offset]  # [B,1,T]

    #                 pred = text_logits.argmax(-1)   # [B,1,T]
    #                 correct = (pred == text_target) & text_mask       # boolean

    #                 # avoid ZeroDivision
    #                 if text_mask.sum() > 0:
    #                     step_text_acc = correct.sum() / text_mask.sum()
    #                 else:
    #                     step_text_acc = torch.tensor(0.0)

    #             # ロギング用
    #             total_text_acc += float(step_text_acc)

    #             text_target = codes[:, :self.model.audio_offset]  # [B,T_text]
    #             audio_target = codes[:,self.model.audio_offset:self.model.audio_offset + self.model.dep_q]

    #             # decoded = self.tokenizer.decode(codes[0,0].tolist())
    #             # print(decoded)

    #             # print("DEBUG: text_target[0, 0, :50]", codes[0, 0])

    #             # text_logits: [B, 1, T, vocab_size]
    #             # 教師信号: text部分（= input_ids[:, 0, :]）を参照
    #             text_loss = compute_loss_with_mask(
    #                 text_logits,
    #                 text_target,
    #                 outputs["text_logits_mask"],
    #                 mode="text",
    #                 text_padding_weight=self.args.text_padding_weight,
    #                 text_padding_ids={
    #                     self.model.text_padding_token_id,
    #                     self.model.end_of_text_padding_id,
    #                 },
    #             )
    #             audio_loss = compute_loss_with_mask(
    #                 audio_logits,
    #                 audio_target,
    #                 outputs["logits_mask"],
    #                 mode="audio",
    #                 first_codebook_weight_multiplier=self.args.first_codebook_weight_multiplier,
    #             )

    #             loss = text_loss*2 + audio_loss

    #         # --- Backprop ---
    #         self.accelerator.backward(loss)
    #         if self.accelerator.sync_gradients:
    #             self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
    #             # もし image_embedder も学習対象ならそちらもクリップ対象にするのが安全ですが、
    #             # 通常は model 側に含めるか、params リストに対して行います。
    #             # image_embedder のパラメータも optimizer に含まれているので、
    #             # ここで image_embedder.parameters() もクリップするのがベストです。
    #             self.accelerator.clip_grad_norm_(self.image_embedder.parameters(), 1.0)
    #         self.optimizer.step()
    #         self.optimizer.zero_grad(set_to_none=True)

    #         total_loss += loss.item()
    #         avg_loss = total_loss / (step + 1)
    #         avg_text_acc = total_text_acc / (step + 1)
    #         avg_audio_acc = total_audio_acc / (step + 1)

    #         # ============================
    #         #   tqdm update
    #         # ============================
    #         if step % log_interval == 0:
    #             pbar.set_postfix({
    #                 "loss": f"{avg_loss:.4f}",
    #                 "text_acc": f"{avg_text_acc:.3f}",
    #                 "audio_acc": f"{avg_audio_acc:.3f}",
    #             })

    #         # --- WandB Logging (STEP) ---
    #         print(f"writer:{self.writer is not None}")
    #         if self.writer is not None:
    #             self.writer.log_step(
    #                 step=global_step,
    #                 loss=loss.item(),
    #                 text_loss=text_loss.item(),
    #                 audio_loss=audio_loss.item(),
    #             )

    #         # --- Step-based Checkpoint Saving ---
    #         import os
    #         from safetensors.torch import save_model

    #         if global_step % 2000 == 0 and global_step > 0:
    #             if self.accelerator.is_main_process:
    #                 ckpt_path = f"./checkpoints/step_{global_step}.safetensors"
    #                 os.makedirs("./checkpoints", exist_ok=True)

    #                 # unwrap MoshiVis 本体
    #                 model_to_save = self.model
    #                 if hasattr(model_to_save, "module"):
    #                     model_to_save = model_to_save.module

    #                 # MoshiVis + ImageProjection をまとめたバンドルを作成
    #                 bundle = MoshiVisBundle(model_to_save, self.image_embedder)

    #                 # shared weight 付きでも安全に保存できる
    #                 save_model(bundle, ckpt_path)

    #                 print(f"💾 Saved FULL safetensor checkpoint at step {global_step}: {ckpt_path}")



    #     # ==================================================
    #     #   Epoch Summary
    #     # ==================================================
    #     avg_loss = total_loss / len(dataloader)
    #     avg_text_acc = total_text_acc / len(dataloader)
    #     avg_audio_acc = total_audio_acc / len(dataloader)

    #     print(f"✅ Epoch {epoch} | Loss={avg_loss:.4f} | TextAcc={avg_text_acc:.3f} | AudioAcc={avg_audio_acc:.3f}")
    #     return avg_loss

    def save_checkpoint(self, path: str):
        """AMP安全にcheckpointを保存"""
        from safetensors.torch import save_model
        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module
        bundle = MoshiVisBundle(model_to_save, self.image_embedder)
        save_model(bundle, path)
        print(f"💾 Saved checkpoint (FULL): {path}")
