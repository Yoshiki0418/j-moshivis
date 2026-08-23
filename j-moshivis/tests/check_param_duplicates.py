"""
Verify if SharedCrossAttention parameters are being duplicated in optimizer
"""
import torch
from jmoshivis.config.kyuteye_config import KyuteyeConfig
from jmoshivis.models.moshivis import MoshiVis

config_path = "/workspace/j-moshivis/configs/j-moshi-vis.yaml"
kyuteye_config = KyuteyeConfig.from_yml(config_path)

moshi_vis = MoshiVis(**kyuteye_config.moshi_constructor_kwargs, dtype=torch.bfloat16)

print("="*70)
print("🔍 Checking SharedCrossAttention parameter duplication")
print("="*70)

# Collect cross_attn params like train.py does
cross_attn_params = []
gate_params = []

seen_ids = set()
duplicate_count = 0

for name, p in moshi_vis.named_parameters():
    if not p.requires_grad:
        continue
    
    param_id = id(p)
    
    if "gate" in name or "xa_gate" in name:
        if param_id in seen_ids:
            print(f"⚠️ DUPLICATE Gate: {name}")
            duplicate_count += 1
        else:
            gate_params.append(p)
            seen_ids.add(param_id)
    
    elif "cross_attention" in name or "xa" in name or "norm_cross" in name:
        if param_id in seen_ids:
            print(f"⚠️ DUPLICATE CrossAttn: {name}")
            duplicate_count += 1
        else:
            cross_attn_params.append(p)
            seen_ids.add(param_id)

print(f"\n📊 Summary:")
print(f"   Unique CrossAttn params: {len(cross_attn_params)}")
print(f"   Unique Gate params: {len(gate_params)}")
print(f"   Duplicate params skipped: {duplicate_count}")

# Check MHA specifically
print(f"\n🔍 MHA parameter IDs:")
mha_ids = []
for i in range(min(5, len(moshi_vis.llm.transformer.layers))):
    layer = moshi_vis.llm.transformer.layers[i]
    if hasattr(layer, 'cross_attention') and layer.cross_attention is not None:
        mha = layer.cross_attention.mha
        for pname, p in mha.named_parameters():
            print(f"   Layer {i} {pname}: id={id(p)}")
            if id(p) not in mha_ids:
                mha_ids.append(id(p))

print(f"\n   Unique MHA param tensors: {len(mha_ids)}")

if duplicate_count > 0:
    print(f"\n❌ PROBLEM FOUND: {duplicate_count} duplicate params in optimizer!")
    print("   This may cause the optimizer to skip updates!")
else:
    print("\n✅ No duplicates found")
