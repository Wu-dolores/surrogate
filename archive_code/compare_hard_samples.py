import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# ---------- model defs (must match your training scripts) ----------
class LocalGNOBlock(nn.Module):
    def __init__(self, hidden=128, K=6):
        super().__init__()
        self.K = K
        self.msg = nn.Sequential(
            nn.Linear(hidden*2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden*2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, coord):
        B, N, H = h.shape
        agg = torch.zeros_like(h)
        count = torch.zeros((1, N, 1), device=h.device, dtype=h.dtype)

        for off in range(-self.K, self.K + 1):
            if off == 0:
                continue
            if off > 0:
                hi = h[:, :-off, :]
                hj = h[:, off:, :]
                dcoord = (coord[:, off:] - coord[:, :-off]).unsqueeze(-1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, :-off, :] += m
                count[:, :-off, :] += 1.0
            else:
                k = -off
                hi = h[:, k:, :]
                hj = h[:, :-k, :]
                dcoord = (coord[:, :-k] - coord[:, k:]).unsqueeze(-1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, k:, :] += m
                count[:, k:, :] += 1.0

        agg = agg / count.clamp_min(1.0)
        dh = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + dh)

class V1LocalGNO(nn.Module):
    def __init__(self, in_dim=4, hidden=128, K=6, L=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)
        return self.dec(h)

class RelativeBias(nn.Module):
    def __init__(self, n_heads=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )
    def forward(self, dcoord):  # (B,N,N,1) -> (B,N,N,H)
        return self.net(dcoord)

class GlobalAttnBlock(nn.Module):
    def __init__(self, hidden=128, n_heads=4, bias_hidden=64, attn_drop=0.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_head = hidden // n_heads

        self.qkv = nn.Linear(hidden, hidden*3)
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.bias = RelativeBias(n_heads=n_heads, hidden=bias_hidden)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, h, coord, mask=None):
        B, N, H = h.shape
        x = self.norm(h)

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B,heads,N,N)

        dcoord = (coord[:, None, :] - coord[:, :, None]).unsqueeze(-1)      # (B,N,N,1)
        bias = self.bias(dcoord).permute(0, 3, 1, 2)                         # (B,heads,N,N)
        attn = attn + bias

        if mask is not None:
            m = mask[:, None, None, :].to(dtype=attn.dtype)
            attn = attn.masked_fill(m == 0, float("-inf"))

        w = torch.softmax(attn, dim=-1)
        w = self.drop(w)

        out = torch.matmul(w, v)  # (B,heads,N,d)
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        out = self.proj(out)
        return h + out

class V2Model(nn.Module):
    def __init__(self, in_dim=4, hidden=128, K=6, L=4, n_heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.local = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])
        self.attn = GlobalAttnBlock(hidden=hidden, n_heads=n_heads, bias_hidden=64)
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.local:
            h = blk(h, coord)
        h = self.attn(h, coord)
        return self.dec(h)

# ---------- utils ----------
def zapply(x, mu, std):
    return (x - mu) / std

def zinvert(xn, mu, std):
    return xn * std + mu

def rmse(a, b):
    return np.sqrt(np.mean((a - b)**2))

def band_rmse(pred, true, sl):
    return rmse(pred[:, sl], true[:, sl])

@torch.no_grad()
def predict(model, Xn_t, coord_t, Y_mu, Y_std, device):
    model.eval()
    predn = model(Xn_t.to(device), coord_t.to(device)).cpu().numpy()  # (S,N,1)
    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)                    # (S,N)
    return pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--ckpt_v1", type=str, required=True)
    ap.add_argument("--ckpt_v2", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--m", type=int, default=5, help="band size for upper/near-surface metrics")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)        # (S,)
    logp = d["logp_arr"].astype(np.float32)  # (S,N)
    T = d["T_arr"].astype(np.float32)        # (S,N)
    q = d["q_arr"].astype(np.float32)        # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)  # (S,N)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)
    X = np.stack([T, logp, q, Ts_b], axis=-1).astype(np.float32)  # (S,N,4)
    Y = Fnet.astype(np.float32)                                   # (S,N)

    # val split (same rule as training scripts)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xva = X[va]
    Yva = Y[va]
    cva = logp[va]

    # Load checkpoints (each has its own normalization; we evaluate using each model's own norm)
    ck1 = torch.load(args.ckpt_v1, map_location="cpu")
    ck2 = torch.load(args.ckpt_v2, map_location="cpu")

    # Build models using cfg from ckpt
    cfg1 = ck1.get("cfg", {})
    cfg2 = ck2.get("cfg", {})

    v1 = V1LocalGNO(
        in_dim=4,
        hidden=int(cfg1.get("hidden", 128)),
        K=int(cfg1.get("K", 6)),
        L=int(cfg1.get("L", 4))
    ).to(device)
    v1.load_state_dict(ck1["state_dict"])

    v2 = V2Model(
        in_dim=4,
        hidden=int(cfg2.get("hidden", 128)),
        K=int(cfg2.get("K", 6)),
        L=int(cfg2.get("L", 4)),
        n_heads=int(cfg2.get("heads", cfg2.get("n_heads", 4)))
    ).to(device)
    v2.load_state_dict(ck2["state_dict"])

    # Prepare tensors for val
    coord_t = torch.tensor(cva, dtype=torch.float32)

    # ---- v1 normalization and prediction ----
    X_mu1 = np.array(ck1["X_mu"], dtype=np.float32).reshape(1,1,4)
    X_std1 = np.array(ck1["X_std"], dtype=np.float32).reshape(1,1,4)
    Y_mu1 = np.array(ck1["Y_mu"], dtype=np.float32).reshape(1,1,1)
    Y_std1 = np.array(ck1["Y_std"], dtype=np.float32).reshape(1,1,1)

    Xva1n = zapply(Xva, X_mu1, X_std1).astype(np.float32)
    Xva1n_t = torch.tensor(Xva1n, dtype=torch.float32)
    pred1 = predict(v1, Xva1n_t, coord_t, Y_mu1, Y_std1, device)     # (V,N)

    # ---- v2 normalization and prediction ----
    X_mu2 = np.array(ck2["X_mu"], dtype=np.float32).reshape(1,1,4)
    X_std2 = np.array(ck2["X_std"], dtype=np.float32).reshape(1,1,4)
    Y_mu2 = np.array(ck2["Y_mu"], dtype=np.float32).reshape(1,1,1)
    Y_std2 = np.array(ck2["Y_std"], dtype=np.float32).reshape(1,1,1)

    Xva2n = zapply(Xva, X_mu2, X_std2).astype(np.float32)
    Xva2n_t = torch.tensor(Xva2n, dtype=torch.float32)
    pred2 = predict(v2, Xva2n_t, coord_t, Y_mu2, Y_std2, device)     # (V,N)

    true = Yva  # (V,N)

    # ---- per-sample metrics ----
    V = true.shape[0]
    m = args.m
    per = []
    for i in range(V):
        e1 = pred1[i] - true[i]
        e2 = pred2[i] - true[i]
        per.append({
            "i_val": i,
            "rmse1": float(np.sqrt(np.mean(e1**2))),
            "rmse2": float(np.sqrt(np.mean(e2**2))),
            "toa1": float(abs(e1[0])),
            "toa2": float(abs(e2[0])),
            "boa1": float(abs(e1[-1])),
            "boa2": float(abs(e2[-1])),
            "upper1": float(np.sqrt(np.mean(e1[:m]**2))),
            "upper2": float(np.sqrt(np.mean(e2[:m]**2))),
            "near1": float(np.sqrt(np.mean(e1[-m:]**2))),
            "near2": float(np.sqrt(np.mean(e2[-m:]**2))),
        })

    # pick hard samples by v1 rmse
    per_sorted = sorted(per, key=lambda x: x["rmse1"], reverse=True)
    hard = per_sorted[:args.topk]

    # overall summary
    rmse1_all = rmse(pred1, true)
    rmse2_all = rmse(pred2, true)
    toa1_all = rmse(pred1[:,0], true[:,0])
    toa2_all = rmse(pred2[:,0], true[:,0])
    boa1_all = rmse(pred1[:,-1], true[:,-1])
    boa2_all = rmse(pred2[:,-1], true[:,-1])

    print("\n=== Overall VAL (all samples) ===")
    print(f"v1: prof RMSE={rmse1_all:.3f} | TOA RMSE={toa1_all:.3f} | BOA RMSE={boa1_all:.3f}")
    print(f"v2: prof RMSE={rmse2_all:.3f} | TOA RMSE={toa2_all:.3f} | BOA RMSE={boa2_all:.3f}")

    # hard-set summary
    idx = [h["i_val"] for h in hard]
    rmse1_h = rmse(pred1[idx], true[idx])
    rmse2_h = rmse(pred2[idx], true[idx])
    toa1_h = rmse(pred1[idx,0], true[idx,0])
    toa2_h = rmse(pred2[idx,0], true[idx,0])
    boa1_h = rmse(pred1[idx,-1], true[idx,-1])
    boa2_h = rmse(pred2[idx,-1], true[idx,-1])

    print(f"\n=== HARD SET (top-{args.topk} by v1 profile RMSE) ===")
    print(f"v1: prof RMSE={rmse1_h:.3f} | TOA RMSE={toa1_h:.3f} | BOA RMSE={boa1_h:.3f}")
    print(f"v2: prof RMSE={rmse2_h:.3f} | TOA RMSE={toa2_h:.3f} | BOA RMSE={boa2_h:.3f}")

    # save per-sample table
    outdir = "hard_compare_out"
    os.makedirs(outdir, exist_ok=True)
    table_path = os.path.join(outdir, "hard_table.txt")
    with open(table_path, "w") as f:
        f.write("rank,i_val,rmse_v1,rmse_v2,toa_v1,toa_v2,boa_v1,boa_v2,upper_v1,upper_v2,near_v1,near_v2\n")
        for r, h in enumerate(hard, 1):
            f.write(f"{r},{h['i_val']},{h['rmse1']:.6f},{h['rmse2']:.6f},"
                    f"{h['toa1']:.6f},{h['toa2']:.6f},{h['boa1']:.6f},{h['boa2']:.6f},"
                    f"{h['upper1']:.6f},{h['upper2']:.6f},{h['near1']:.6f},{h['near2']:.6f}\n")
    print("\nSaved hard table:", table_path)

    # plot profiles for hard samples
    for r, h in enumerate(hard, 1):
        i = h["i_val"]
        plt.figure()
        plt.plot(cva[i], true[i], label="true")
        plt.plot(cva[i], pred1[i], label="v1", linestyle="--")
        plt.plot(cva[i], pred2[i], label="v2", linestyle=":")
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Fnet (W/m^2)")
        plt.title(f"HARD rank {r} | i_val={i} | v1={h['rmse1']:.3f} v2={h['rmse2']:.3f} (W/m^2)")
        plt.legend()
        plt.savefig(os.path.join(outdir, f"hard_{r:02d}_ival_{i}.png"), dpi=160, bbox_inches="tight")
        plt.close()

    # aggregate error vs level on hard set
    e1 = np.mean(np.abs(pred1[idx] - true[idx]), axis=0)
    e2 = np.mean(np.abs(pred2[idx] - true[idx]), axis=0)
    plt.figure()
    plt.plot(cva[idx[0]], e1, label="v1 mean|err|")
    plt.plot(cva[idx[0]], e2, label="v2 mean|err|")
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Mean |error| (W/m^2)")
    plt.title(f"Hard set mean abs error vs logp (top-{args.topk})")
    plt.legend()
    plt.savefig(os.path.join(outdir, "hard_mean_abs_error_vs_logp.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved {args.topk} hard profile plots + aggregate error plot to: {outdir}/")

if __name__ == "__main__":
    main()
