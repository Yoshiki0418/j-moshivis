
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.moshivis import MoshiVis, MoshiVisGen

def diagnose_mismatch():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    
    print(f"🚀 Device: {device}, Dtype: {dtype}")

    # Load Config & Model
    config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
    checkpoint_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"
    
    kyuteye_config = KyuteyeConfig.from_yml(config_path)
    moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
    
    # Load weights
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    loaded_weights = load_file(checkpoint_path, device="cpu")
    model_state = {k.replace("module.", ""): v for k, v in loaded_weights.items() if not k.startswith("image_prefix.")}
    moshi_vis.load_state_dict(model_state, strict=False)
    moshi_vis = moshi_vis.eval().to(device).to(dtype)

    # Create Generator
    gen = MoshiVisGen(moshi_vis=moshi_vis, use_sampling=False)  # Sampling OFF for deterministic check

    # --- Prepare Dummy Input ---
    B, T = 1, 10
    num_codebooks = 17
    
    # Random tokens (simulating input)
    # 0: Text, 1-8: Moshi Audio, 9-16: User Audio
    sequence = torch.randint(0, 1024, (B, num_codebooks, T), device=device)
    
    # Dummy transformer output (simulating Helium features)
    # [B, T, D]
    transformer_out = torch.randn(B, T, moshi_vis.llm.dim, device=device, dtype=dtype)

    print("\n🔍 Checking Forward Pass Consistency...")

    # --- 1. Training Path (forward_depformer_training) ---
    # This processes the whole sequence at once
    with torch.no_grad():
        # [B, Ka, T, Card]
        training_logits = moshi_vis.forward_depformer_training(sequence, transformer_out)
    
    print(f"   Training Logits Shape: {training_logits.shape}")

    # --- 2. Inference Path (depformer_step) ---
    # We need to feed tokens one by one
    inference_logits_list = []
    
    print("   Running Inference Step-by-Step...")
    
    # MoshiVisGen logic simulation
    # The Depformer streaming context must be reset for each time step `t`
    # because Depformer creates a sequence of length 8 (codebooks) for each time frame.
    
    collected_logits = []
    
    with torch.no_grad():
        for t in range(T):
            # Text token at step t
            text_token_t = sequence[:, 0, t] # [B]
            
            # Transformer out at step t
            transformer_out_t = transformer_out[:, t:t+1, :] # [B, 1, D]
            
            step_logits = []
            
            # Enter streaming mode for this time step (resets offset to 0)
            with moshi_vis.depformer.streaming():
                for cb_index in range(8): # 8 audio codebooks
                    # Input token selection (same as before)
                    if cb_index == 0:
                        input_token = sequence[:, 0, t].view(B, 1, 1)
                    else:
                        input_token = sequence[:, cb_index, t].view(B, 1, 1)

                    logit = moshi_vis.forward_depformer(
                        cb_index, 
                        input_token, 
                        transformer_out[:, t:t+1, :]
                    )
                    step_logits.append(logit)
            
            collected_logits.append(torch.cat(step_logits, dim=1)) # [B, 8, 1, Card]

    inference_logits = torch.cat(collected_logits, dim=2) # [B, 8, T, Card]
    print(f"   Inference Logits Shape: {inference_logits.shape}")
    
    # --- Comparison ---
    diff = (training_logits - inference_logits).abs().max().item()
    print(f"\n📊 Max Difference: {diff}")
    
    if diff < 1e-3:
        print("✅ MATCH! Training and Inference logic are consistent.")
    else:
        print("❌ MISMATCH! There is a logic error in streaming or input handling.")
        # Analyze where mismatch starts
        for t in range(T):
            d_t = (training_logits[:, :, t] - inference_logits[:, :, t]).abs().max().item()
            print(f"   Step {t}: diff={d_t}")

if __name__ == "__main__":
    diagnose_mismatch()
