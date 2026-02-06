import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

torch.set_num_threads(1)

# ------------------ normalization utils ------------------
def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / std

def zinvert(xn, mu, std):
    return xn * std + mu

# ------------------ model ------------------
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
        B, N, Hh = h.shape
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

# ------------------ regridding ------------------
def _interp1d_batch(x_old, y_old, x_new):
    """
    Batched linear interpolation.
    x_old: (B,Nold) strictly monotonic (we will assume increasing in index; if decreasing, we flip)
    y_old: (B,Nold) or (B,Nold,C)
    x_new: (B,Nnew)
    returns y_new: (B,Nnew) or (B,Nnew,C)
    """
    # Ensure increasing along axis=1
    if (x_old[:, 1] < x_old[:, 0]).all():
        x_old = torch.flip(x_old, dims=[1])
        if y_old.dim() == 2:
            y_old = torch.flip(y_old, dims=[1])
        else:
            y_old = torch.flip(y_old, dims=[1])

    B, Nold = x_old.shape
    Nnew = x_new.shape[1]

    # searchsorted expects ascending
    idx = torch.searchsorted(x_old, x_new, right=True)  # (B,Nnew)
    idx0 = (idx - 1).clamp(0, Nold - 1)
    idx1 = idx.clamp(0, Nold - 1)

    x0 = torch.gather(x_old, 1, idx0)
    x1 = torch.gather(x_old, 1, idx1)
    denom = (x1 - x0).clamp_min(1e-6)
    w = (x_new - x0) / denom  # (B,Nnew)

    if y_old.dim() == 2:
        y0 = torch.gather(y_old, 1, idx0)
        y1 = torch.gather(y_old, 1, idx1)
        y_new = y0 * (1 - w) + y1 * w
        return y_new
    else:
        C = y_old.shape[-1]
        idx0c = idx0.unsqueeze(-1).expand(-1, -1, C)
        idx1c = idx1.unsqueeze(-1).expand(-1, -1, C)
        y0 = torch.gather(y_old, 1, idx0c)
        y1 = torch.gather(y_old, 1, idx1c)
        y_new = y0 * (1 - w.unsqueeze(-1)) + y1 * w.unsqueeze(-1)
        return y_new

def make_uniform_like_coord(logp_old, Nnew):
    """
    Create a new coord grid per-sample by linear spacing in logp between top and bottom.
    logp_old: (B,Nold)
    return logp_new: (B,Nnew)
    """
    top = logp_old[:, 0:1]
    bot = logp_old[:, -1:]
    t = torch.linspace(0.0, 1.0, steps=Nnew, device=logp_old.device).view(1, Nnew)
    return top * (1 - t) + bot * t

# ------------------ eval ------------------
@torch.no_grad()
def eval_phys(model, Xva, Yva, cva, Y_mu, Y_std, device, batch=256):
    model.eval()
    preds = []
    trues = []
    for i in range(0, Xva.shape[0], batch):
        xb = Xva[i:i+batch].to(device)
        cb = cva[i:i+batch].to(device)
        yb = Yva[i:i+batch]
        predn = model(xb, cb).cpu().numpy()
        preds.append(predn)
        trues.append(yb.cpu().numpy())
    predn = np.concatenate(preds, axis=0)
    truen = np.concatenate(trues, axis=0)

    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)
    true = zinvert(truen, Y_mu, Y_std).squeeze(-1)

    rmse_prof = float(np.sqrt(np.mean((pred - true)**2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0])**2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1])**2)))
    return rmse_prof, rmse_toa, rmse_boa

# ------------------ main ------------------
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

    ap.add_argument("--m_bc", type=int, default=5)
    ap.add_argument("--lam_bc", type=float, default=1.0)

    ap.add_argument("--N_choices", type=str, default="40,60,80,120,160",
                    help="comma-separated target N for regrid augmentation")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- load base data (N=60) -----
    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)     # (S,N0)
    T    = d["T_arr"].astype(np.float32)
    q    = d["q_arr"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)     # (S,N0)

    S, N0 = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N0, axis=1).astype(np.float32)

    # base features
    X0 = np.stack([T, logp, q, Ts_b], axis=-1).astype(np.float32)  # (S,N0,4)
    Y0 = Fnet[..., None].astype(np.float32)                        # (S,N0,1)
    C0 = logp.astype(np.float32)

    # split
    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xtr0, Ytr0, Ctr0 = X0[tr], Y0[tr], C0[tr]
    Xva0, Yva0, Cva0 = X0[va], Y0[va], C0[va]

    # normalization stats from base (N0) train set
    Fdim = Xtr0.shape[-1]
    X_mu, X_std = zfit(Xtr0.reshape(-1, Fdim))
    Y_mu, Y_std = zfit(Ytr0.reshape(-1, 1))

    # pre-normalize base val (keep on cpu tensors)
    Xva0n = zapply(Xva0, X_mu, X_std)
    Yva0n = zapply(Yva0, Y_mu, Y_std)

    Xva_t = torch.tensor(Xva0n, dtype=torch.float32)
    Yva_t = torch.tensor(Yva0n, dtype=torch.float32)
    Cva_t = torch.tensor(Cva0, dtype=torch.float32)

    # train tensors (base)
    Xtr0_t = torch.tensor(Xtr0, dtype=torch.float32)  # keep physical for regrid
    Ytr0_t = torch.tensor(Ytr0, dtype=torch.float32)
    Ctr0_t = torch.tensor(Ctr0, dtype=torch.float32)

    # N choices
    N_choices = [int(x) for x in args.N_choices.split(",")]
    print("Regrid N choices:", N_choices)

    model = V1LocalGNO(in_dim=Fdim, hidden=args.hidden, K=args.K, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    mse = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    outdir = "step4_v1_regrid_out"
    os.makedirs(outdir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(outdir, "best_v1_regrid.pt")

    m = int(args.m_bc)
    lam_bc = float(args.lam_bc)

    # helper to sample a batch of indices
    def sample_batch_indices(bs):
        return torch.tensor(rng.integers(0, Xtr0_t.shape[0], size=bs), dtype=torch.long)

    # training loop
    for ep in range(1, args.epochs + 1):
        model.train()

        # approx number of steps per epoch similar to DataLoader
        steps = int(np.ceil(Xtr0_t.shape[0] / args.batch))
        for _ in range(steps):
            idx = sample_batch_indices(args.batch)
            x_phys = Xtr0_t[idx].to(device)  # (B,N0,4) physical
            y_phys = Ytr0_t[idx].to(device)  # (B,N0,1) physical
            c_old  = Ctr0_t[idx].to(device)  # (B,N0)   physical coord

            # choose target N
            Nnew = int(N_choices[rng.integers(0, len(N_choices))])

            # build new coord grid (per-sample linear in logp)
            c_new = make_uniform_like_coord(c_old, Nnew)  # (B,Nnew)

            # regrid each feature channel (T, logp, q, Ts_broadcast)
            # x_phys includes logp itself as channel 1, but we rebuild it from c_new to keep consistency
            T_old  = x_phys[..., 0]                 # (B,N0)
            q_old  = x_phys[..., 2]                 # (B,N0)
            Ts_old = x_phys[..., 3]                 # (B,N0)
            # interpolate
            T_new  = _interp1d_batch(c_old, T_old, c_new)
            q_new  = _interp1d_batch(c_old, q_old, c_new)
            Ts_new = _interp1d_batch(c_old, Ts_old, c_new)  # constant anyway, but ok
            # target y
            y_new  = _interp1d_batch(c_old, y_phys, c_new)  # (B,Nnew,1)

            # build X_new physical with consistent logp channel
            X_new_phys = torch.stack([T_new, c_new, q_new, Ts_new], dim=-1)  # (B,Nnew,4)

            # normalize
            X_mu_t = torch.tensor(X_mu, dtype=torch.float32, device=device).view(1,1,-1)
            X_std_t = torch.tensor(X_std, dtype=torch.float32, device=device).view(1,1,-1)
            Y_mu_t = torch.tensor(Y_mu, dtype=torch.float32, device=device).view(1,1,1)
            Y_std_t = torch.tensor(Y_std, dtype=torch.float32, device=device).view(1,1,1)

            X_new = (X_new_phys - X_mu_t) / X_std_t
            Y_new = (y_new - Y_mu_t) / Y_std_t

            opt.zero_grad(set_to_none=True)
            pred = model(X_new, c_new)

            Lf = mse(pred, Y_new)
            Lbc = mse(pred[:, -m:, :], Y_new[:, -m:, :])
            loss = Lf + lam_bc * Lbc
            loss.backward()
            opt.step()

        # eval on base val grid (N0)
        rmse_prof, rmse_toa, rmse_boa = eval_phys(model, Xva_t, Yva_t, Cva_t, float(Y_mu.squeeze()), float(Y_std.squeeze()), device)
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
                "train_aug": {"type": "regrid", "N_choices": N_choices},
                "losses": {"lam_bc": lam_bc, "m_bc": m},
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} W/m^2 | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f} W/m^2)")
    print("Now evaluate on real N=40 testset with eval_real_testset.py")

if __name__ == "__main__":
    main()
