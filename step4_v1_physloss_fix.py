import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.set_num_threads(1)

def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / std

def zinvert_torch(xn, mu, std):
    # mu/std are python floats or 1x1 arrays; promote to tensors on xn.device
    if not torch.is_tensor(mu):
        mu = torch.tensor(mu, dtype=xn.dtype, device=xn.device)
    if not torch.is_tensor(std):
        std = torch.tensor(std, dtype=xn.dtype, device=xn.device)
    return xn * std + mu

def diffop(F, coord):
    """
    F: (B,N,1) or (B,N)
    coord: (B,N)
    returns dF/dcoord with same shape as F (B,N,1)
    """
    if F.dim() == 2:
        F = F.unsqueeze(-1)
    B, N, C = F.shape
    dF = torch.zeros_like(F)

    # interior: central difference
    num = F[:, 2:, :] - F[:, :-2, :]
    den = (coord[:, 2:] - coord[:, :-2]).unsqueeze(-1).clamp_min(1e-6)
    dF[:, 1:-1, :] = num / den

    # boundaries: one-sided
    den0 = (coord[:, 1] - coord[:, 0]).unsqueeze(-1).clamp_min(1e-6)
    denN = (coord[:, -1] - coord[:, -2]).unsqueeze(-1).clamp_min(1e-6)
    dF[:, 0, :] = (F[:, 1, :] - F[:, 0, :]) / den0
    dF[:, -1, :] = (F[:, -1, :] - F[:, -2, :]) / denN
    return dF

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
                hi, hj = h[:, :-off, :], h[:, off:, :]
                dcoord = (coord[:, off:] - coord[:, :-off]).unsqueeze(-1)
                m = self.msg(torch.cat([hi, hj, dcoord], dim=-1))
                agg[:, :-off, :] += m
                count[:, :-off, :] += 1.0
            else:
                k = -off
                hi, hj = h[:, k:, :], h[:, :-k, :]
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

    pred = predn * Y_std + Y_mu
    true = truen * Y_std + Y_mu
    pred = pred.squeeze(-1)
    true = true.squeeze(-1)

    rmse_prof = float(np.sqrt(np.mean((pred - true)**2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0])**2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1])**2)))
    return rmse_prof, rmse_toa, rmse_boa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)

    ap.add_argument("--lam_hr", type=float, default=0.05)  # start small
    ap.add_argument("--lam_bc", type=float, default=1.0)
    ap.add_argument("--m_bc", type=int, default=5)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)
    logp = d["logp_arr"].astype(np.float32)
    T    = d["T_arr"].astype(np.float32)
    q    = d["q_arr"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    # OLD v1 features
    X = np.stack([T, logp, q, Ts_b], axis=-1).astype(np.float32)  # (S,N,4)
    Y = Fnet[..., None].astype(np.float32)                        # (S,N,1)

    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xtr, Ytr, ctr = X[tr], Y[tr], logp[tr]
    Xva, Yva, cva = X[va], Y[va], logp[va]

    Fdim = X.shape[-1]
    X_mu, X_std = zfit(Xtr.reshape(-1, Fdim))
    Y_mu, Y_std = zfit(Ytr.reshape(-1, 1))

    Xtrn = zapply(Xtr, X_mu, X_std)
    Xvan = zapply(Xva, X_mu, X_std)
    Ytrn = zapply(Ytr, Y_mu, Y_std)
    Yvan = zapply(Yva, Y_mu, Y_std)

    Xtr_t = torch.tensor(Xtrn, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytrn, dtype=torch.float32)
    Xva_t = torch.tensor(Xvan, dtype=torch.float32)
    Yva_t = torch.tensor(Yvan, dtype=torch.float32)
    ctr_t = torch.tensor(ctr, dtype=torch.float32)
    cva_t = torch.tensor(cva, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, ctr_t),
                              batch_size=args.batch, shuffle=True)

    model = V1LocalGNO(in_dim=Fdim, hidden=args.hidden, K=args.K, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    mse = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    outdir = "step4_v1_physfix_out"
    os.makedirs(outdir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(outdir, "best_v1_physfix.pt")

    m = int(args.m_bc)

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb, cb in train_loader:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)

            predn = model(xb, cb)  # normalized flux

            # flux losses in normalized space
            Lf  = mse(predn, yb)
            Lbc = mse(predn[:, -m:, :], yb[:, -m:, :])

            # HR consistency loss in physical space (derived from flux, not HR_arr)
            pred_phys = zinvert_torch(predn, float(Y_mu.squeeze()), float(Y_std.squeeze()))
            true_phys = zinvert_torch(yb,   float(Y_mu.squeeze()), float(Y_std.squeeze()))
            hr_pred = diffop(pred_phys, cb)
            hr_true = diffop(true_phys, cb)
            Lhr = mse(hr_pred, hr_true)

            loss = Lf + args.lam_bc * Lbc + args.lam_hr * Lhr
            loss.backward()
            opt.step()

        rmse_prof, rmse_toa, rmse_boa = eval_phys(model, Xva_t, Yva_t, cva_t, float(Y_mu.squeeze()), float(Y_std.squeeze()))
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
                "features": ["T", "logp", "q", "Ts_broadcast"],
                "target": "Fnet",
                "losses": {"lam_bc": args.lam_bc, "m_bc": m, "lam_hr": args.lam_hr, "hr_target": "from_flux"},
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} W/m^2 | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f} W/m^2)")

if __name__ == "__main__":
    main()
