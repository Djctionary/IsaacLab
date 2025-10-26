# fix_ckpt_norm_keys.py
import torch, sys
ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu")

# 观察一下有哪些归一化相关键
print([k for k in ckpt.keys() if "norm" in k.lower()])

# 常见映射：老 -> 新
mapping = {
    "critic_obs_norm_dict": "obs_norm_state_dict",
    "critic_obs_norm_state_dict": "obs_norm_state_dict",
    # 视情况把 privileged 的也对上（如果你用了非对称观察/蒸馏）
    "privileged_obs_norm_dict": "privileged_obs_norm_state_dict",
}

for old, new in mapping.items():
    if old in ckpt and new not in ckpt:
        ckpt[new] = ckpt[old]
        print(f"mapped {old} -> {new}")

torch.save(ckpt, ckpt_path.replace(".pt", "_fixed.pt"))
print("saved:", ckpt_path.replace(".pt", "_fixed.pt"))
