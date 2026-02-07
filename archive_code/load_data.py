import numpy as np
import torch
import matplotlib.pyplot as plt
from diffop import diffop  # 你把 diffop.py 放在同目录

# 1) 读数据
d = np.load("output_1000_data(2).npz", allow_pickle=True)
Fnet = torch.tensor(d["Fnet_arr"], dtype=torch.float32)     # (S,N)
logp = torch.tensor(d["logp_arr"], dtype=torch.float32)     # (S,N)

print("Fnet shape:", Fnet.shape, "logp shape:", logp.shape)

# 2) 用 diffop 算导数（HR-like）
dF_dlogp = diffop(Fnet, logp)   # (S,N)

print("dF/dlogp stats:",
      dF_dlogp.min().item(),
      dF_dlogp.max().item(),
      dF_dlogp.mean().item())

# 3) 随机抽 3 条画图验收
idx = torch.randperm(Fnet.shape[0])[:3]
for k, i in enumerate(idx.tolist()):
    plt.figure()
    # 常见画法：把 x 轴反过来，看起来更像“高度向上”
    plt.plot(logp[i].numpy(), Fnet[i].numpy(), label="Fnet")
    plt.plot(logp[i].numpy(), dF_dlogp[i].numpy(), label="dF/dlogp (HR-like)")
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.legend()
    plt.title(f"Sample {i}")
    plt.show()
