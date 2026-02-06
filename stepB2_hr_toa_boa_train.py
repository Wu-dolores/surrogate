import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# ------------------ utils: normalization ------------------
def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / std

def zinvert(xn, mu, std):
    return xn * std + mu

# ------------------ monotonic helpers ------------------
def enforce_toa_to_boa_numpy(logp, T, q, Fnet):
    """
    Enforce ordering: TOA -> BOA so logp increases with index.
    If majority have logp[:,0] > logp[:,-1], flip all arrays.
    """
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T    = T[:, ::-1].copy()
        q    = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()
    return logp, T, q, Fnet

def _ensure_increasing_torch(x, y=None):
    # x: (B,N). If decreasing, flip. Keep y aligned.
    if (x[:, 1] < x[:, 0]).all():
        x = torch.flip(x, dims=[1])
        if y is not None:
            y = torch.flip(y, dims=[1])
    return x, y

# ------------------ batched interpolation on logp ------------------
def interp1d_batch(x_old, y_old, x_new):
    """
    Linear interpolation per sample.
    x_old: (B,Nold)
    y_old: (B,Nold) or (B,Nold,C)
    x_new: (B,Nnew)
    """
    x_old, y_old = _ensure_increasing_torch(x_old, y_old)
    x_new, _ = _ensure_increasing_torch(x_new, None)

    B, Nold = x_old.shape
    idx = torch.searchsorted(x_old, x_new, right=True)
    idx0 = (idx - 1).clamp(0, Nold - 1)
    idx1 = idx.clamp(0, Nold - 1)

    x0 = torch.gather(x_old, 1, idx0)
    x1 = torch.gather(x_old, 1, idx1)
    w = (x_new - x0) / (x1 - x0).clamp_min(1e-6)

    if y_old.dim() == 2:
        y0 = torch.gather(y_old, 1, idx0)
        y1 = torch.gather(y_old, 1, idx1)
        y_new = y0 * (1 - w) + y1 * w
    else:
        C = y_old.shape[-1]
        idx0c = idx0.unsqueeze(-1).expand(-1, -1, C)
        idx1c = idx1.unsqueeze(-1).expand(-1, -1, C)
        y0 = torch.gather(y_old, 1, idx0c)
        y1 = torch.gather(y_old, 1, idx1c)
        y_new = y0 * (1 - w.unsqueeze(-1)) + y1 * w.unsqueeze(-1)

    return y_new

def make_linear_coord_from_old(logp_old, Nnew):
    # logp_old: (B,Nold) increasing (TOA->BOA)
    top = logp_old[:, 0:1]
    bot = logp_old[:, -1:]
    t = torch.linspace(0.0, 1.0, steps=Nnew, device=logp_old.device).view(1, Nnew)
    return top * (1 - t) + bot * t

def scaled_m_bc(Nnew, N0=60, m0=5, m_min=3):
    m = int(round(m0 * float(Nnew) / float(N0)))
    m = max(m_min, m)
    m = min(m, max(1, Nnew // 3))
    return m

# ------------------ differentiable diffop: dF/dlogp ------------------
def diffop_dF_dlogp(F, logp):
    """
    F:    (B,N)  physical
    logp: (B,N)  physical, increasing
    returns dF/dlogp (B,N)
    """
    B, N = F.shape
    d = torch.zeros_like(F)

    num = F[:, 2:] - F[:, :-2]
    den = (logp[:, 2:] - logp[:, :-2]).clamp_min(1e-6)
    d[:, 1:-1] = num / den

    d[:, 0] = (F[:, 1] - F[:, 0]) / (logp[:, 1] - logp[:, 0]).clamp_min(1e-6)
    d[:, -1] = (F[:, -1] - F[:, -2]) / (logp[:, -1] - logp[:, -2]).clamp_min(1e-6)
    return d

def cumtrapz_batch(y, x):
    """
    y,x: (B,N) same grid
    I[:,0]=0, I[:,k]=∫ y dx up to k
    """
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = torch.zeros_like(y)
    I[:, 1:] = torch.cumsum(inc, dim=1)
    return I

# ------------------ model ------------------
class LocalGNOBlock(nn.Module):
    def __init__(self, hidden=128, K=6):
        super().__init__()
        self.K = K
        self.msg = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
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

class HR_TOA_BOA_Model(nn.Module):
    """
    Inputs: per-layer features on arbitrary N
    Outputs:
      - HR_pred(z) = dF/dlogp  (B,N,1)
      - F_TOA_pred (B,1)
      - F_BOA_pred (B,1)
    """
    def __init__(self, in_dim=4, hidden=128, K=6, L=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])

        self.hr_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.toa_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.boa_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)

        hr = self.hr_head(h)            # (B,N,1)
        g = h.mean(dim=1)               # (B,H)
        f_toa = self.toa_head(g)        # (B,1)
        f_boa = self.boa_head(g)        # (B,1)
        return hr, f_toa, f_boa

# ------------------ eval (physical) ------------------
@torch.no_grad()
def eval_recon(model, Xn, logp, Ftrue, X_mu, X_std, F_mu, F_std, H_mu, H_std, device, batch=256):
    """
    Evaluate reconstruction on fixed-grid dataset (no augmentation).
    """
    model.eval()
    S = Xn.shape[0]
    preds = []
    for i in range(0, S, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(logp[i:i+batch], dtype=torch.float32, device=device)
        hrn, ftoan, fboan = model(xb, cb)

        hr = (hrn.squeeze(-1) * H_std + H_mu)            # torch (B,N)
        f_toa = (ftoan.squeeze(-1) * F_std + F_mu)       # torch (B,)
        f_boa = (fboan.squeeze(-1) * F_std + F_mu)       # torch (B,)

        I = cumtrapz_batch(hr, cb)                       # (B,N)
        f_tilde = f_toa.unsqueeze(-1) + I                # (B,N)

        delta = (f_boa - f_tilde[:, -1])                 # (B,)
        alpha = (cb - cb[:, 0:1]) / (cb[:, -1:] - cb[:, 0:1] + 1e-6)
        frec = f_tilde + delta.unsqueeze(-1) * alpha     # (B,N)

        preds.append(frec.cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = Ftrue

    rmse_prof = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1]) ** 2)))
    return rmse_prof, rmse_toa, rmse_boa, pred, true

# ------------------ main ------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="stepB2_hr_toa_boa_out")

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)

    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)

    ap.add_argument("--N_choices", type=str, default="40,60,80,120,160")
    ap.add_argument("--alpha_gamma", type=float, default=1.0)


    # losses
    ap.add_argument("--lam_hr", type=float, default=0.5)
    ap.add_argument("--lam_frec", type=float, default=1.0)
    ap.add_argument("--lam_bc", type=float, default=0.2)
    ap.add_argument("--m0_bc", type=int, default=5)
    ap.add_argument("--N0_ref", type=int, default=60)

    # scalar endpoint loss weights
    ap.add_argument("--lam_toa", type=float, default=1.0)
    ap.add_argument("--lam_boa", type=float, default=1.0)

    # Ts tail oversampling
    ap.add_argument("--Ts_tail", type=float, default=330.0)
    ap.add_argument("--tail_mult", type=float, default=2.0)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)     # (S,N)
    T    = d["T_arr"].astype(np.float32)        # (S,N)
    q    = d["q_arr"].astype(np.float32)        # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)     # (S,N)

    # enforce consistent ordering
    logp, T, q, Fnet = enforce_toa_to_boa_numpy(logp, T, q, Fnet)

    S, Nbase = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], Nbase, axis=1).astype(np.float32)

    # features: [T, logp, q, Ts_broadcast]
    X0 = np.stack([T, logp, q, Ts_b], axis=-1).astype(np.float32)  # (S,N,4)
    F0 = Fnet.astype(np.float32)                                   # (S,N)

    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xtr0, Ftr0, Ctr0, Tstr = X0[tr], F0[tr], logp[tr], Ts[tr]
    Xva0, Fva0, Cva0       = X0[va], F0[va], logp[va]

    # HR_true on base grid for stats
    Ftr0_t = torch.tensor(Ftr0, dtype=torch.float32)
    Ctr0_t = torch.tensor(Ctr0, dtype=torch.float32)
    HRtr0 = diffop_dF_dlogp(Ftr0_t, Ctr0_t).numpy().astype(np.float32)  # (S_tr,N)

    # normalization stats on train only
    X_mu, X_std = zfit(Xtr0.reshape(-1, 4))
    F_mu, F_std = zfit(Ftr0.reshape(-1, 1))         # flux scale
    H_mu, H_std = zfit(HRtr0.reshape(-1, 1))        # HR scale

    # normalize fixed-grid val once
    Xva0n = zapply(Xva0, X_mu, X_std)

    # Ts-tail oversampling weights
    weights = np.ones_like(Tstr, dtype=np.float64)
    weights[Tstr >= args.Ts_tail] *= float(args.tail_mult)
    weights = weights / weights.sum()

    N_choices = [int(x) for x in args.N_choices.split(",")]
    print("Regrid N choices:", N_choices)
    print(f"Ts-tail oversampling: Ts >= {args.Ts_tail}K weight x{args.tail_mult}")
    print(f"Loss weights: lam_hr={args.lam_hr}, lam_toa={args.lam_toa}, lam_boa={args.lam_boa}, lam_frec={args.lam_frec}, lam_bc={args.lam_bc}")

    model = HR_TOA_BOA_Model(in_dim=4, hidden=args.hidden, K=args.K, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    mse = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(outdir, "best_hr_toa_boa.pt")

    # torch constants for normalization
    X_mu_t  = torch.tensor(X_mu, dtype=torch.float32, device=device).view(1, 1, 4)
    X_std_t = torch.tensor(X_std, dtype=torch.float32, device=device).view(1, 1, 4)
    F_mu_t  = torch.tensor(float(F_mu.squeeze()), dtype=torch.float32, device=device)
    F_std_t = torch.tensor(float(F_std.squeeze()), dtype=torch.float32, device=device)
    H_mu_t  = torch.tensor(float(H_mu.squeeze()), dtype=torch.float32, device=device)
    H_std_t = torch.tensor(float(H_std.squeeze()), dtype=torch.float32, device=device)

    steps = int(np.ceil(Xtr0.shape[0] / args.batch))

    for ep in range(1, args.epochs + 1):
        model.train()

        for _ in range(steps):
            idx_np = rng.choice(Xtr0.shape[0], size=args.batch, replace=True, p=weights)

            x0 = torch.tensor(Xtr0[idx_np], dtype=torch.float32, device=device)  # (B,Nbase,4)
            f0 = torch.tensor(Ftr0[idx_np], dtype=torch.float32, device=device)  # (B,Nbase)
            c0 = torch.tensor(Ctr0[idx_np], dtype=torch.float32, device=device)  # (B,Nbase)

            # choose Nnew and build coord
            Nnew = int(N_choices[rng.integers(0, len(N_choices))])
            c_new = make_linear_coord_from_old(c0, Nnew)                         # (B,Nnew)

            # interpolate physical fields onto c_new
            T_old  = x0[..., 0]
            q_old  = x0[..., 2]
            Ts_old = x0[..., 3]
            T_new  = interp1d_batch(c0, T_old,  c_new)
            q_new  = interp1d_batch(c0, q_old,  c_new)
            Ts_new = interp1d_batch(c0, Ts_old, c_new)
            f_new  = interp1d_batch(c0, f0,     c_new)                           # (B,Nnew)

            # HR_true from f_new
            hr_true = diffop_dF_dlogp(f_new, c_new)                              # (B,Nnew)

            # endpoints true (physical)
            f_toa_true = f_new[:, 0]                                              # (B,)
            f_boa_true = f_new[:, -1]                                             # (B,)

            # normalize inputs
            X_phys = torch.stack([T_new, c_new, q_new, Ts_new], dim=-1)           # (B,Nnew,4)
            Xn = (X_phys - X_mu_t) / X_std_t

            # normalize targets
            hr_t = (hr_true - H_mu_t) / H_std_t
            f_toa_t = (f_toa_true - F_mu_t) / F_std_t
            f_boa_t = (f_boa_true - F_mu_t) / F_std_t

            opt.zero_grad(set_to_none=True)
            hrn_pred, ftoan_pred, fboan_pred = model(Xn, c_new)

            hrn_pred   = hrn_pred.squeeze(-1)            # (B,Nnew)
            ftoan_pred = ftoan_pred.squeeze(-1)          # (B,)
            fboan_pred = fboan_pred.squeeze(-1)          # (B,)

            # losses
            Lhr  = mse(hrn_pred, hr_t)
            Ltoa = mse(ftoan_pred, f_toa_t)
            Lboa = mse(fboan_pred, f_boa_t)

            # reconstruct flux with 2-endpoint anchoring
            hr_phys   = hrn_pred * H_std_t + H_mu_t
            f_toa_phys = ftoan_pred * F_std_t + F_mu_t
            f_boa_phys = fboan_pred * F_std_t + F_mu_t

            I = cumtrapz_batch(hr_phys, c_new)
            f_tilde = f_toa_phys.unsqueeze(-1) + I

            delta = f_boa_phys - f_tilde[:, -1]  # (B,)
            alpha = (c_new - c_new[:, 0:1]) / (c_new[:, -1:] - c_new[:, 0:1] + 1e-6)
            alpha = alpha.clamp(0.0, 1.0) ** args.alpha_gamma
            f_rec = f_tilde + delta.unsqueeze(-1) * alpha


            # reconstructed flux loss (normalized in F scale)
            f_rec_n  = (f_rec - F_mu_t) / F_std_t
            f_true_n = (f_new - F_mu_t) / F_std_t
            Lfrec = mse(f_rec_n, f_true_n)

            # bottom band loss (normalized)
            m_bc = scaled_m_bc(Nnew, N0=args.N0_ref, m0=args.m0_bc)
            Lbc = mse(f_rec_n[:, -m_bc:], f_true_n[:, -m_bc:])

            loss = (
                args.lam_hr  * Lhr
                + args.lam_toa * Ltoa
                + args.lam_boa * Lboa
                + args.lam_frec * Lfrec
                + args.lam_bc * Lbc
            )

            loss.backward()
            opt.step()

        # validation on fixed-grid val
        rmse_prof, rmse_toa, rmse_boa, _, _ = eval_recon(
            model=model,
            Xn=Xva0n,
            logp=Cva0,
            Ftrue=Fva0,
            X_mu=X_mu, X_std=X_std,
            F_mu=float(F_mu.squeeze()), F_std=float(F_std.squeeze()),
            H_mu=float(H_mu.squeeze()), H_std=float(H_std.squeeze()),
            device=device,
            batch=256
        )
        sched.step(rmse_prof)

        if rmse_prof < best:
            best = rmse_prof
            torch.save({
                "state_dict": model.state_dict(),
                "X_mu": X_mu.squeeze(0).tolist(),
                "X_std": X_std.squeeze(0).tolist(),
                "F_mu": float(F_mu.squeeze()),
                "F_std": float(F_std.squeeze()),
                "H_mu": float(H_mu.squeeze()),
                "H_std": float(H_std.squeeze()),
                "cfg": vars(args),
                "features": ["T", "logp", "q", "Ts_broadcast"],
                "targets": ["HR=dF/dlogp", "F_TOA", "F_BOA"],
                "reconstruct": "F = (F_TOA + cumtrapz(HR dlogp)) + linear_correction_to_match_F_BOA",
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f})")

    # plots on random val samples
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    idx = rng.choice(Xva0.shape[0], size=5, replace=False)
    Xn = zapply(Xva0[idx], X_mu, X_std)

    rmse_prof, rmse_toa, rmse_boa, pred, true = eval_recon(
        model=model,
        Xn=Xn,
        logp=Cva0[idx],
        Ftrue=Fva0[idx],
        X_mu=X_mu, X_std=X_std,
        F_mu=float(F_mu.squeeze()), F_std=float(F_std.squeeze()),
        H_mu=float(H_mu.squeeze()), H_std=float(H_std.squeeze()),
        device=device,
        batch=256
    )

    coord = Cva0[idx]

    plt.figure()
    for i in range(pred.shape[0]):
        plt.plot(coord[i], true[i], alpha=0.9)
        plt.plot(coord[i], pred[i], alpha=0.9, linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Fnet (W/m^2)")
    plt.title("Route-B2: reconstructed Fnet (solid=true, dashed=pred)")
    plt.savefig(os.path.join(outdir, "profiles_overlay.png"), dpi=160, bbox_inches="tight")
    plt.close()

    err = pred - true
    mean_err = err.mean(axis=0)
    mae = np.abs(err).mean(axis=0)

    plt.figure()
    plt.plot(coord[0], mean_err)
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Mean (pred-true) (W/m^2)")
    plt.title("Mean error vs logp (val samples)")
    plt.savefig(os.path.join(outdir, "mean_error_vs_logp_val.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(coord[0], mae)
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("MAE (W/m^2)")
    plt.title("MAE vs logp (val samples)")
    plt.savefig(os.path.join(outdir, "mae_vs_logp_val.png"), dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Saved plots to {outdir}/profiles_overlay.png and mean/mae val plots")

if __name__ == "__main__":
    main()
