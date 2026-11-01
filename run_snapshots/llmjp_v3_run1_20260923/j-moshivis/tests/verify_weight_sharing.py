"""
Verify if Cross-Attention MHA weights are TRULY shared across layers.

Key question: Does SharedCrossAttention metaclass actually share weights,
or does each layer have separate (randomly initialized) weights?
"""
import torch
from safetensors.torch import load_file

from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.moshivis import MoshiVis

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16

config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
checkpoint_path = "/workspace/j-moshivis/checkpoints/step_2000.safetensors"

kyuteye_config = KyuteyeConfig.from_yml(config_path)

print("="*70)
print("🔍 VERIFICATION: Are Cross-Attention weights TRULY shared?")
print("="*70)

# ===== 1. Build model WITHOUT loading weights =====
print("\n📦 Building model (no weights loaded)...")
moshi_vis_fresh = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)

print("\n🔍 Check 1: Are MHA objects the SAME instance across layers?")
layers = moshi_vis_fresh.llm.transformer.layers

# Get MHA from each layer
mha_objects = []
for i in range(len(layers)):
    if hasattr(layers[i], 'cross_attention') and layers[i].cross_attention is not None:
        mha = layers[i].cross_attention.mha
        mha_objects.append((i, mha, id(mha)))

if mha_objects:
    first_id = mha_objects[0][2]
    all_same = all(obj[2] == first_id for obj in mha_objects)
    
    if all_same:
        print(f"   ✅ All {len(mha_objects)} layers share the SAME MHA object (id={first_id})")
        print("   → Weights ARE truly shared!")
    else:
        print(f"   ❌ MHA objects are DIFFERENT across layers!")
        for i, mha, obj_id in mha_objects[:5]:
            print(f"      Layer {i}: id={obj_id}")
        print("   → Each layer has SEPARATE weights (potential random init)!")

# ===== 2. Check if weight tensors are the same object =====
print("\n🔍 Check 2: Are MHA weight TENSORS identical across layers?")

if len(mha_objects) >= 2:
    mha0 = mha_objects[0][1]
    mha5 = mha_objects[min(5, len(mha_objects)-1)][1]
    
    w0 = mha0.in_proj_weight
    w5 = mha5.in_proj_weight
    
    same_data = w0.data_ptr() == w5.data_ptr()
    print(f"   Layer 0 weight data_ptr: {w0.data_ptr()}")
    print(f"   Layer 5 weight data_ptr: {w5.data_ptr()}")
    
    if same_data:
        print("   ✅ Weight tensors point to the SAME memory!")
    else:
        print("   ❌ Weight tensors are DIFFERENT objects!")
        # Check if values are the same
        values_equal = torch.allclose(w0, w5)
        print(f"   Values equal: {values_equal}")

# ===== 3. Check Gate objects (should be different) =====
print("\n🔍 Check 3: Are Gate objects different (as expected with xa_gate_shared=False)?")

gate_objects = []
for i in range(len(layers)):
    if hasattr(layers[i], 'cross_attention') and layers[i].cross_attention is not None:
        gate = layers[i].cross_attention.gate
        if gate is not None:
            gate_objects.append((i, gate, id(gate)))

if gate_objects:
    first_gate_id = gate_objects[0][2]
    all_gates_same = all(obj[2] == first_gate_id for obj in gate_objects)
    
    if all_gates_same:
        print(f"   ⚠️ All gates are the SAME object (unexpected if xa_gate_shared=False)")
    else:
        unique_gate_ids = len(set(obj[2] for obj in gate_objects))
        print(f"   ✅ Gates are DIFFERENT across layers ({unique_gate_ids} unique instances)")
        print("   → Each layer has its OWN gate weights (correct!)")

# ===== 4. Load weights and verify =====
print("\n" + "="*70)
print("📂 Loading checkpoint and verifying...")
print("="*70)

loaded_weights = load_file(checkpoint_path, device="cpu")

# Clean keys
model_state = {}
for key, v in loaded_weights.items():
    clean_key = key
    if clean_key.startswith("image_prefix."):
        continue
    if clean_key.startswith("module."):
        clean_key = clean_key[len("module."):]
    if clean_key.startswith("_orig_mod."):
        clean_key = clean_key[len("_orig_mod."):]
    model_state[clean_key] = v

# Load into model
moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=dtype)
missing, unexpected = moshi_vis.load_state_dict(model_state, strict=False)

print(f"\n📊 Load results:")
print(f"   Missing keys: {len(missing)}")
print(f"   Unexpected keys: {len(unexpected)}")

# Check MHA weights after loading
print("\n🔍 Check 4: MHA weights after loading checkpoint")

layers_loaded = moshi_vis.llm.transformer.layers

# Compare layer 0 vs layer 15 MHA weights
if len(layers_loaded) > 15:
    mha0 = layers_loaded[0].cross_attention.mha
    mha15 = layers_loaded[15].cross_attention.mha
    
    w0_loaded = mha0.in_proj_weight
    w15_loaded = mha15.in_proj_weight
    
    same_ptr = w0_loaded.data_ptr() == w15_loaded.data_ptr()
    print(f"   Layer 0 MHA weight: mean={w0_loaded.mean().item():.6f}, std={w0_loaded.std().item():.6f}")
    print(f"   Layer 15 MHA weight: mean={w15_loaded.mean().item():.6f}, std={w15_loaded.std().item():.6f}")
    print(f"   Same data pointer: {same_ptr}")
    
    if same_ptr:
        print("   ✅ Weights are truly shared - both layers use SAME tensor")
    else:
        values_equal = torch.allclose(w0_loaded, w15_loaded)
        print(f"   Values numerically equal: {values_equal}")
        if not values_equal:
            print("   ❌ CRITICAL: Layer 15 has DIFFERENT values from Layer 0!")
            print("   → This confirms Gemini Pro's hypothesis!")

# ===== 5. Check Gate weights after loading =====
print("\n🔍 Check 5: Gate weights after loading (should be learned, different per layer)")

gate0 = layers_loaded[0].cross_attention.gate
gate15 = layers_loaded[15].cross_attention.gate

if hasattr(gate0, 'alpha') and hasattr(gate15, 'alpha'):
    # Get first weight from the sequential
    if hasattr(gate0.alpha, '0'):
        w_gate0 = gate0.alpha[0].weight
        w_gate15 = gate15.alpha[0].weight
        
        print(f"   Layer 0 Gate weight: mean={w_gate0.mean().item():.6f}, std={w_gate0.std().item():.6f}")
        print(f"   Layer 15 Gate weight: mean={w_gate15.mean().item():.6f}, std={w_gate15.std().item():.6f}")
        
        gates_equal = torch.allclose(w_gate0, w_gate15)
        if gates_equal:
            print("   ⚠️ Gate weights are identical (might indicate loading issue)")
        else:
            print("   ✅ Gate weights are different per layer (correct!)")

print("\n" + "="*70)
print("📋 Summary")
print("="*70)
