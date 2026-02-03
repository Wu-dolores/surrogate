import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / std

def zinvert(xn, mu, std):
    return xn * std + mu

class LocalGNOBlock(nn.Module):
    """One Local-GNO block: message passing on K-index neighborhood with delta coord bias."""
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
        # h: (B,N,H), coord: (B,N)
        B, N, H = h.shape
        agg = torch.zeros_like(h)
        count = torch.zeros((1, N, 1), device=h.device, dtype=h.dtype)

        for off in range(-self.K, self.K + 1):
            if off == 0:
                continue
            if off > 0:
                hi = h[:, :-off, :]
                hj = h[:, off:, :]
                dcoord = (coord[:, off:] - coord[:, :-off]).unsqueeze(-1)  # (B,N-off,1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, :-off, :] += m
                count[:, :-off, :] += 1.0
            else:
                k = -off
                hi = h[:, k:, :]
                hj = h[:, :-k, :]
                dcoord = (coord[:, :-k] - coord[:, k:]).unsqueeze(-1)       # (B,N-k,1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, k:, :] += m
                count[:, k:, :] += 1.0

        agg = agg / count.clamp_min(1.0)
        dh = self.upd(torch.cat([h, agg], dim=-1))
        return self.norm(h + dh)  # residual + LN

class V1LocalGNO(nn.Module):
    def __init__(self, in_dim=4, hidden=128, K=6, L=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)
        return self.dec(h)

@torch.no_grad()
def eval_phys(model, Xva_t, Yva_t, cva_t, Y_mu, Y_std):
    model.eval()
    device = next(model.parameters()).device
    predn = model(Xva_t.to(device), cva_t.to(device)).cpu().numpy()
    truen = Yva_t.cpu().numpy()

    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)
    true = zinvert(truen, Y_mu, Y_std).squeeze(-1)

    rmse_prof = float(np.sqrt(np.mean((pred - true)**2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0])**2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1])**2)))
    return rmse_prof, rmse_toa, rmse_boa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)   # (S,N)
    T = d["T_arr"].astype(np.float32)         # (S,N)
    q = d["q_arr"].astype(np.float32)         # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)   # (S,N)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)
    
    c_top = logp[:, 0:1]
    c_surf = logp[:, -1:]
    c_tilde = (logp - c_top) / (c_surf - c_top + 1e-6)

    X = np.stack([T, logp, c_tilde, q, Ts_b], axis=-1)  # (S,N,5)

    Y = Fnet[..., None]                             # (S,N,1)

    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xtr, Ytr, ctr = X[tr], Y[tr], logp[tr]
    Xva, Yva, cva = X[va], Y[va], logp[va]

    # feature normalization (train stats)
    X_mu, X_std = zfit(Xtr.reshape(-1, 4))
    Y_mu, Y_std = zfit(Ytr.reshape(-1, 1))
    Xtrn = zapply(Xtr, X_mu, X_std)
    Xvan = zapply(Xva, X_mu, X_std)
    Ytrn = zapply(Ytr, Y_mu, Y_std)
    Yvan = zapply(Yva, Y_mu, Y_std)

    Xtr_t = torch.tensor(Xtrn, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytrn, dtype=torch.float32)
    Xva_t = torch.tensor(Xvan, dtype=torch.float32)
    Yva_t = torch.tensor(Yvan, dtype=torch.float32)
    ctr_t = torch.tensor(ctr, dtype=torch.float32)   # IMPORTANT: raw coord, not normalized
    cva_t = torch.tensor(cva, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, ctr_t), batch_size=args.batch, shuffle=True)

    model = V1LocalGNO(in_dim=5, hidden=args.hidden, K=args.K, L=args.L)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    # Simple ReduceLROnPlateau
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    outdir = "step4_v1_out"
    os.makedirs(outdir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(outdir, "best_v1.pt")

    # training loop
    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb, cb)

            # Loss modification
            m = 5
            lam_bc = 1.0
            Lf = loss_fn(pred, yb)
            # BOA (Bottom Of Atmosphere) only
            Lbc = loss_fn(pred[:, -m:, :], yb[:, -m:, :])
            loss = Lf + lam_bc * Lbc

            loss.backward()
            opt.step()

        rmse_prof, rmse_toa, rmse_boa = eval_phys(model, Xva_t, Yva_t, cva_t, Y_mu, Y_std)
        sched.step(rmse_prof)

        if rmse_prof < best:
            best = rmse_prof
            torch.save({
                "state_dict": model.state_dict(),
                "X_mu": X_mu.squeeze(0).tolist(),
                "X_std": X_std.squeeze(0).tolist(),
                "Y_mu": float(Y_mu.squeeze()),
                "Y_std": float(Y_std.squeeze()),
                "cfg": vars(args),
                "features": ["T", "logp", "c_tilde", "q", "Ts_broadcast"],
                "target": "Fnet",
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} W/m^2 | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f} W/m^2)")

    # --------- plots (random 5) ----------
    # load best for plotting
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    idx = rng.choice(Xva_t.shape[0], size=5, replace=False)
    with torch.no_grad():
        predn = model(Xva_t[idx].to(device), cva_t[idx].to(device)).cpu().numpy()
    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)
    true = zinvert(Yva_t[idx].cpu().numpy(), Y_mu, Y_std).squeeze(-1)
    coord = cva[idx]

    # profile overlay
    plt.figure()
    for i in range(true.shape[0]):
        plt.plot(coord[i], true[i], alpha=0.9)
        plt.plot(coord[i], pred[i], alpha=0.9, linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Fnet (W/m^2)")
    plt.title("Step4 v1: Fnet profiles (solid=true, dashed=pred)")
    plt.savefig(os.path.join(outdir, "profiles_overlay.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # error vs height (mean abs error per level)
    mae_level = np.mean(np.abs(pred - true), axis=0)
    plt.figure()
    plt.plot(coord[0], mae_level)
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Mean |error| (W/m^2)")
    plt.title("Step4 v1: Mean abs error vs logp")
    plt.savefig(os.path.join(outdir, "mae_vs_logp.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {outdir}/profiles_overlay.png and {outdir}/mae_vs_logp.png")

if __name__ == "__main__":
    main()
