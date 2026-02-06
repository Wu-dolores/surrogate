import argparse, os
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

# -------------------------
# z-norm helpers
# -------------------------
def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-6)

def zinvert_torch(xn, mu, std):
    return xn * (std + 1e-6) + mu

# -------------------------
# DiffOp: dF/dcoord on non-uniform coord (logp)
# -------------------------
def diffop_torch(F, coord):
    """
    F:     (B,N)  physical
    coord: (B,N)  physical (logp)
    returns dF/dcoord: (B,N)
      interior: central
      boundaries: one-sided
    """
    B, N = F.shape
    d = torch.zeros_like(F)

    # interior
    num = F[:, 2:] - F[:, :-2]
    den = coord[:, 2:] - coord[:, :-2]
    d[:, 1:-1] = num / (den + 1e-6)

    # top boundary (0): forward
    d[:, 0] = (F[:, 1] - F[:, 0]) / ((coord[:, 1] - coord[:, 0]) + 1e-6)

    # bottom boundary (N-1): backward
    d[:, -1] = (F[:, -1] - F[:, -2]) / ((coord[:, -1] - coord[:, -2]) + 1e-6)
    return d

# -------------------------
# Regrid (per-sample): interpolate along c_tilde in [0,1]
# -------------------------
def regrid_1d_profile(x, c_old, c_new):
    return np.interp(c_new, c_old, x).astype(np.float32)

def regrid_batch(T, logp, q, Fnet, Ts, N_new):
    """
    Inputs (numpy):
      T,logp,q,Fnet: (B,N_old)
      Ts: (B,)
    Returns regridded (numpy float32):
      Tn, logpn, qn, Fnetn, Ts_broadcast
    """
    B, N_old = Fnet.shape

    c_top = logp[:, 0:1]
    c_sfc = logp[:, -1:]
    c_old = (logp - c_top) / (c_sfc - c_top + 1e-6)  # (B,N_old)

    c_new = np.linspace(0.0, 1.0, N_new, dtype=np.float32)

    Tn = np.zeros((B, N_new), dtype=np.float32)
    logpn = np.zeros((B, N_new), dtype=np.float32)
    qn = np.zeros((B, N_new), dtype=np.float32)
    Fnetn = np.zeros((B, N_new), dtype=np.float32)

    for i in range(B):
        Tn[i] = regrid_1d_profile(T[i], c_old[i], c_new)
        logpn[i] = regrid_1d_profile(logp[i], c_old[i], c_new)
        qn[i] = regrid_1d_profile(q[i], c_old[i], c_new)
        Fnetn[i] = regrid_1d_profile(Fnet[i], c_old[i], c_new)

    Ts_b = np.repeat(Ts[:, None], N_new, axis=1).astype(np.float32)
    return Tn, logpn, qn, Fnetn, Ts_b

# -------------------------
# Torch integration utilities
# -------------------------
def cumtrapz_torch(y, x):
    """
    y,x: (B,N)
    returns integral from 0..i with trapezoid rule, same shape (B,N), with out[:,0]=0
    """
    out = torch.zeros_like(y)
    dx = x[:, 1:] - x[:, :-1]                        # (B,N-1)
    area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx         # (B,N-1)
    out[:, 1:] = torch.cumsum(area, dim=1)
    return out

# -------------------------
# Recon flux: integrate HR -> anchor TOA -> enforce BOA via bottom-window alpha
# -------------------------
def recon_flux_from_hr_toa_boa(
    HR, logp, TOA, BOA,
    alpha_gamma=1.0,
    bot_window_k=12
):
    """
    HR:   (B,N) physical
    logp: (B,N) physical
    TOA:  (B,)  physical
    BOA:  (B,)  physical
    returns Frec: (B,N) physical

    Key change vs old: delta is distributed ONLY in bottom window (last bot_window_k levels).
    """
    # integrate HR over logp, with f_tilde[:,0]=0
    f_tilde = cumtrapz_torch(HR, logp)               # (B,N)

    # anchor TOA
    f1 = f_tilde + TOA[:, None]                      # (B,N)

    # delta needed to match BOA
    delta = (BOA - f1[:, -1])                        # (B,)

    # bottom-window alpha
    B, N = logp.shape
    Kb = int(min(max(1, bot_window_k), N))

    alpha = torch.zeros_like(logp)
    if Kb >= 2:
        ramp = torch.linspace(0.0, 1.0, Kb, device=logp.device, dtype=logp.dtype)[None, :]
        alpha[:, -Kb:] = ramp ** float(alpha_gamma)
    else:
        alpha[:, -1:] = 1.0

    Frec = f1 + delta[:, None] * alpha
    return Frec

# -------------------------
# Model: Local-GNO + heads (HR profile, TOA scalar, BOA scalar)
# -------------------------
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

class GNO_HR_TOA_BOA(nn.Module):
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

        hr = self.hr_head(h)          # (B,N,1)
        g = h.mean(dim=1)             # (B,H)
        toa = self.toa_head(g)        # (B,1)
        boa = self.boa_head(g)        # (B,1)
        return hr, toa, boa

# -------------------------
# Eval (physical RMSE on recon flux) on native val grid
# -------------------------
@torch.no_grad()
def eval_val(
    model, Xn, logp, F_true,
    F_mu, F_std, H_mu, H_std,
    alpha_gamma, bot_window_k,
    device, batch=256
):
    model.eval()
    S, N, _ = Xn.shape

    F_pred_all = []
    for i in range(0, S, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(logp[i:i+batch], dtype=torch.float32, device=device)
        hr_n, toa_n, boa_n = model(xb, cb)

        hr = zinvert_torch(hr_n.squeeze(-1), H_mu, H_std)
        toa = zinvert_torch(toa_n.squeeze(-1), F_mu, F_std)
        boa = zinvert_torch(boa_n.squeeze(-1), F_mu, F_std)

        Frec = recon_flux_from_hr_toa_boa(
            hr, cb, toa, boa,
            alpha_gamma=alpha_gamma,
            bot_window_k=bot_window_k
        )
        F_pred_all.append(Frec.detach().cpu().numpy())

    F_pred = np.concatenate(F_pred_all, axis=0)
    rmse_prof = float(np.sqrt(np.mean((F_pred - F_true) ** 2)))
    rmse_toa  = float(np.sqrt(np.mean((F_pred[:, 0] - F_true[:, 0]) ** 2)))
    rmse_boa  = float(np.sqrt(np.mean((F_pred[:, -1] - F_true[:, -1]) ** 2)))
    return rmse_prof, rmse_toa, rmse_boa

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="stepB2p3b_out")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)

    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)

    ap.add_argument("--alpha_gamma", type=float, default=3.0)
    ap.add_argument("--bot_window_k", type=int, default=12)

    ap.add_argument("--N_choices", type=str, default="40,60,80,120,160")

    ap.add_argument("--Ts_tail", type=float, default=330.0)
    ap.add_argument("--tail_mult", type=float, default=2.0)

    # loss weights
    ap.add_argument("--lam_hr", type=float, default=0.5)
    ap.add_argument("--lam_toa", type=float, default=1.0)
    ap.add_argument("--lam_boa", type=float, default=3.0)
    ap.add_argument("--lam_frec", type=float, default=1.0)

    ap.add_argument("--lam_bc", type=float, default=0.8)
    ap.add_argument("--m0_bc", type=int, default=10)

    # bottom-band profile loss
    ap.add_argument("--bot_k", type=int, default=8)
    ap.add_argument("--lam_bot", type=float, default=0.5)

    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    N_choices = [int(x) for x in args.N_choices.split(",") if len(x.strip()) > 0]
    print("Regrid N choices:", N_choices)
    print(f"Ts-tail oversampling: Ts >= {args.Ts_tail}K weight x{args.tail_mult}")
    print(f"Recon alpha: bottom-window K={args.bot_window_k}, alpha_gamma={args.alpha_gamma}")
    print(f"Bottom-band loss: bot_k={args.bot_k}, lam_bot={args.lam_bot}")

    # ---- load data
    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)   # (S,N)
    T = d["T_arr"].astype(np.float32)         # (S,N)
    q = d["q_arr"].astype(np.float32)         # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)   # (S,N)

    S, N0 = Fnet.shape

    # train/val split
    perm = rng.permutation(S)
    ntr = int(0.8 * S)
    tr_idx = perm[:ntr]
    va_idx = perm[ntr:]

    # fit feature normalization on TRAIN at native grid
    Ts_b_tr = np.repeat(Ts[tr_idx, None], N0, axis=1).astype(np.float32)
    Xtr = np.stack([T[tr_idx], logp[tr_idx], q[tr_idx], Ts_b_tr], axis=-1)  # (S_tr,N0,4)
    X_mu, X_std = zfit(Xtr.reshape(-1, 4))

    # flux stats
    F_mu = float(Fnet[tr_idx].reshape(-1).mean())
    F_std = float(Fnet[tr_idx].reshape(-1).std() + 1e-6)

    # HR stats from TRUE flux
    with torch.no_grad():
        Ftr_t = torch.tensor(Fnet[tr_idx], dtype=torch.float32)
        ctr_t = torch.tensor(logp[tr_idx], dtype=torch.float32)
        HRtr = diffop_torch(Ftr_t, ctr_t).numpy()
    H_mu = float(HRtr.reshape(-1).mean())
    H_std = float(HRtr.reshape(-1).std() + 1e-6)

    # val arrays (native grid)
    Ts_b_va = np.repeat(Ts[va_idx, None], N0, axis=1).astype(np.float32)
    Xva = np.stack([T[va_idx], logp[va_idx], q[va_idx], Ts_b_va], axis=-1).astype(np.float32)
    Xva_n = zapply(Xva, X_mu, X_std).astype(np.float32)
    logp_va = logp[va_idx].astype(np.float32)
    F_va = Fnet[va_idx].astype(np.float32)

    # ---- model / optim
    model = GNO_HR_TOA_BOA(in_dim=4, hidden=args.hidden, K=args.K, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=12, verbose=True)

    best = float("inf")
    best_path = os.path.join(args.outdir, "best_hr_toa_boa.pt")

    # ---- oversampling pool
    tail_mask = (Ts[tr_idx] >= float(args.Ts_tail))
    tail_ids = tr_idx[tail_mask]
    base_ids = tr_idx

    extra_reps = int(max(0, round(args.tail_mult - 1.0)))
    if extra_reps > 0 and len(tail_ids) > 0:
        tr_pool = np.concatenate([base_ids] + [tail_ids for _ in range(extra_reps)], axis=0)
    else:
        tr_pool = base_ids.copy()

    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(tr_pool)

        n_steps = int(np.ceil(len(tr_pool) / args.batch))
        for st in range(n_steps):
            idx = tr_pool[st * args.batch:(st + 1) * args.batch]
            if len(idx) == 0:
                continue

            # choose batch-fixed N
            N_new = int(rng.choice(N_choices))

            Tb = T[idx]
            cb = logp[idx]
            qb = q[idx]
            Fb = Fnet[idx]
            Tsb = Ts[idx]

            # regrid
            Tn, logpn, qn, Fn, Tsbn = regrid_batch(Tb, cb, qb, Fb, Tsb, N_new=N_new)

            # features
            X = np.stack([Tn, logpn, qn, Tsbn], axis=-1).astype(np.float32)
            Xn = zapply(X, X_mu, X_std).astype(np.float32)

            xb = torch.tensor(Xn, dtype=torch.float32, device=device)
            coord = torch.tensor(logpn, dtype=torch.float32, device=device)
            Ftrue = torch.tensor(Fn, dtype=torch.float32, device=device)

            # HR true from TRUE flux
            HRtrue = diffop_torch(Ftrue, coord)

            # forward
            hr_n, toa_n, boa_n = model(xb, coord)

            HR = zinvert_torch(hr_n.squeeze(-1), H_mu, H_std)
            TOA = zinvert_torch(toa_n.squeeze(-1), F_mu, F_std)
            BOA = zinvert_torch(boa_n.squeeze(-1), F_mu, F_std)

            Frec = recon_flux_from_hr_toa_boa(
                HR, coord, TOA, BOA,
                alpha_gamma=args.alpha_gamma,
                bot_window_k=args.bot_window_k
            )

            # losses (physical)
            Lhr = ((HR - HRtrue) ** 2).mean(dim=1)
            Ltoa = (Frec[:, 0] - Ftrue[:, 0]) ** 2
            Lboa = (Frec[:, -1] - Ftrue[:, -1]) ** 2
            Lfrec = ((Frec - Ftrue) ** 2).mean(dim=1)

            m0 = int(min(args.m0_bc, N_new))
            if m0 > 0:
                Lbc = ((Frec[:, -m0:] - Ftrue[:, -m0:]) ** 2).mean(dim=1)
            else:
                Lbc = torch.zeros_like(Lfrec)

            kb = int(min(args.bot_k, N_new))
            if kb > 0:
                Lbot = ((Frec[:, -kb:] - Ftrue[:, -kb:]) ** 2).mean(dim=1)
            else:
                Lbot = torch.zeros_like(Lfrec)

            loss_per = (
                args.lam_hr * Lhr +
                args.lam_toa * Ltoa +
                args.lam_boa * Lboa +
                args.lam_frec * Lfrec +
                args.lam_bc * Lbc +
                args.lam_bot * Lbot
            )
            loss = loss_per.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        # ---- eval
        rmse_prof, rmse_toa, rmse_boa = eval_val(
            model, Xva_n, logp_va, F_va,
            F_mu=F_mu, F_std=F_std,
            H_mu=H_mu, H_std=H_std,
            alpha_gamma=args.alpha_gamma,
            bot_window_k=args.bot_window_k,
            device=device, batch=256
        )
        sched.step(rmse_prof)

        if rmse_prof < best:
            best = rmse_prof
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": vars(args),
                "features": ["T", "logp", "q", "Ts_broadcast"],
                "F_mu": F_mu, "F_std": F_std,
                "H_mu": H_mu, "H_std": H_std,
                "X_mu": X_mu.squeeze(0).tolist(),
                "X_std": X_std.squeeze(0).tolist(),
                "target": "Route-B2 recon (HR+TOA+BOA -> Fnet) bottom-window alpha",
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f} W/m^2)")

if __name__ == "__main__":
    main()
