import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

torch.set_num_threads(1)

# ---------------- utils ----------------
def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-12)

def enforce_toa_to_boa_numpy(logp, T, q, Fnet, HR=None):
    # ensure coord goes from TOA->BOA (logp small->large)
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T    = T[:, ::-1].copy()
        q    = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()
        if HR is not None:
            HR = HR[:, ::-1].copy()
    return logp, T, q, Fnet, HR

def cumtrapz_batch_np(y, x):
    # y,x: (B,N), integrate along axis=1, out[:,0]=0
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = np.zeros_like(y, dtype=np.float32)
    I[:, 1:] = np.cumsum(inc, axis=1)
    return I

def cwp_rw_norm_from_q_logp_np(q, logp):
    """
    cwp(i)=∫_{top→i} q dlogp
    rw(i) =∫_{i→surf} q dlogp
    then normalize by total column: total=cwp[:, -1]
    """
    cwp = cumtrapz_batch_np(q, logp).astype(np.float32)
    total = cwp[:, -1:]  # (B,1)
    rw = (total - cwp).astype(np.float32)
    denom = total + 1e-6
    cwp_n = (cwp / denom).astype(np.float32)
    rw_n  = (rw  / denom).astype(np.float32)
    return cwp_n, rw_n, total

def regrid_profile_batch(x, logp, new_logp):
    """
    Linear interpolation along logp axis for a batch.
    x: (S,N) or (S,N,C) float32
    logp: (S,N)
    new_logp: (S,M)
    returns: (S,M) or (S,M,C)
    """
    S = logp.shape[0]
    if x.ndim == 2:
        out = np.zeros((S, new_logp.shape[1]), dtype=np.float32)
        for i in range(S):
            out[i] = np.interp(new_logp[i], logp[i], x[i]).astype(np.float32)
        return out
    else:
        C = x.shape[2]
        out = np.zeros((S, new_logp.shape[1], C), dtype=np.float32)
        for i in range(S):
            for c in range(C):
                out[i, :, c] = np.interp(new_logp[i], logp[i], x[i, :, c]).astype(np.float32)
        return out

def make_logp_grid_like(logp, M):
    """
    For each sample, create a new logp grid with M points between top and surface (in logp space).
    logp: (S,N) increasing.
    returns new_logp: (S,M)
    """
    top = logp[:, 0:1]
    bot = logp[:, -1:]
    a = np.linspace(0.0, 1.0, M, dtype=np.float32)[None, :]
    return (top + (bot - top) * a).astype(np.float32)

def cumsum_batch_torch(y, x):
    """
    Forward Euler Integration (Cumsum) to match Forward Difference Target.
    y: (B, N, 1) or (B, N) - integrand (HR)
    x: (B, N) - coordinate (logp)
    
    F[i+1] = F[i] + HR[i] * (x[i+1] - x[i])
    """
    if y.shape[-1] == 1 and y.ndim == 3:
        y = y.squeeze(-1) # (B, N)
    
    # Calculate intervals dx
    # Since inputs are usually (B, N) and regridded, we assume x is monotonic
    dx = x[:, 1:] - x[:, :-1] # (B, N-1)
    
    # HR values to use: HR[0]...HR[N-2]
    # We ignore the last HR point for forward integration over N-1 intervals
    inc = y[:, :-1] * dx # (B, N-1)
    
    B, N = y.shape
    I = torch.zeros((B, N), dtype=y.dtype, device=y.device)
    I[:, 1:] = torch.cumsum(inc, dim=1)
    
    return I.unsqueeze(-1) # (B, N, 1)

# ---------------- model ----------------
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
    def __init__(self, in_dim=6, hidden=128, K=6, L=4):
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
        # Boundary heads now take (global_mean + boundary_local) -> hidden*2
        self.toa_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.boa_head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)
        hr = self.hr_head(h)         # (B,N,1)
        g = h.mean(dim=1)            # (B,H)
        
        # Concat global context with local boundary context
        # TOA: index 0, BOA: index -1
        h_toa = h[:, 0, :]           # (B,H)
        h_boa = h[:, -1, :]          # (B,H)
        
        f_toa = self.toa_head(torch.cat([g, h_toa], dim=-1))     # (B,1)
        f_boa = self.boa_head(torch.cat([g, h_boa], dim=-1))     # (B,1)
        return hr, f_toa, f_boa

# ---------------- training/eval ----------------
@torch.no_grad()
def eval_losses(model, Xva_t, logp_va_t, Hva_t, Ftoa_va_t, Fboa_va_t,
                H_mu, H_std, Ftoa_mu, Ftoa_std, Fboa_mu, Fboa_std, device):
    model.eval()
    hrn, ftoan, fboan = model(Xva_t.to(device), logp_va_t.to(device))

    # normalized-space mse
    mse_hr = float(torch.mean((hrn.cpu() - Hva_t) ** 2).item())
    mse_toa = float(torch.mean((ftoan.cpu() - Ftoa_va_t) ** 2).item())
    mse_boa = float(torch.mean((fboan.cpu() - Fboa_va_t) ** 2).item())

    # physical RMSE (HR only for reporting)
    hr = (hrn.squeeze(-1).cpu().numpy() * H_std + H_mu)
    hr_true = (Hva_t.squeeze(-1).cpu().numpy() * H_std + H_mu)
    rmse_hr = float(np.sqrt(np.mean((hr - hr_true) ** 2)))

    return mse_hr, mse_toa, mse_boa, rmse_hr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, default="stepB2p3d_out")
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)

    # regrid augmentation
    ap.add_argument("--regrid_choices", type=str, default="40,60,80,120,160",
                    help="comma-separated N choices for regrid augmentation")
    ap.add_argument("--regrid_mult", type=int, default=1,
                    help="how many augmented draws per original sample per epoch (1 = one draw)")

    # Ts-tail oversampling
    ap.add_argument("--Ts_tail", type=float, default=330.0)
    ap.add_argument("--tail_mult", type=float, default=2.0)

    # loss weights
    ap.add_argument("--lam_toa", type=float, default=20.0)
    ap.add_argument("--lam_boa", type=float, default=20.0)
    ap.add_argument("--lam_hr", type=float, default=1.0)
    ap.add_argument("--lam_phys", type=float, default=2.0, help="weight for Fnet reconstruction loss (Physics)")

    # optional bottom-k pooling target on HR near surface (if you used this in p3c)
    ap.add_argument("--bot_k", type=int, default=0, help="if >0, add HR bottom-k mean loss")
    ap.add_argument("--lam_bot", type=float, default=0.0, help="weight for bottom-k mean HR loss")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------- load data ----------
    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)     # (S,N)
    T    = d["T_arr"].astype(np.float32)        # (S,N)
    q    = d["q_arr"].astype(np.float32)        # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)     # (S,N)

    HR = None
    if "HR_arr" in d.files:
        HR = d["HR_arr"].astype(np.float32)     # (S,N) if present

    logp, T, q, Fnet, HR = enforce_toa_to_boa_numpy(logp, T, q, Fnet, HR=HR)

    S, N0 = Fnet.shape
    Ts_b0 = np.repeat(Ts[:, None], N0, axis=1).astype(np.float32)

    # ----- targets for Route-B2 -----
    # HR target: FORCE FORWARD DIFFERENCE explicitly to match reconstruction integration
    # Do not use the HR from file (which might be centered diff)
    # Target: HR[i] = (F[i+1] - F[i]) / (logp[i+1] - logp[i])
    # For the last point, we just replicate the previous slope or use backward diff, 
    # but it doesn't matter for reconstruction since we integrate N-1 intervals.
    
    print("Recalculating HR targets using Forward Difference for consistency with recon...")
    HR = np.zeros_like(Fnet, dtype=np.float32)
    
    # intervals
    dF = Fnet[:, 1:] - Fnet[:, :-1]
    dp = logp[:, 1:] - logp[:, :-1] + 1e-9
    
    # forward slope
    slope = dF / dp  # (S, N-1)
    
    HR[:, :-1] = slope
    HR[:, -1] = slope[:, -1] # replicate last slope
    
    # We ignore existing HR in file to ensure consistency
    # if HR is None: ... (removed logic)

    F_toa = Fnet[:, 0].astype(np.float32)       # (S,)
    F_boa = Fnet[:, -1].astype(np.float32)      # (S,)

    # features (normalized cwp/rw)
    cwp_n0, rw_n0, tpw0 = cwp_rw_norm_from_q_logp_np(q, logp)
    # tpw0 is (S,1) => broadcast to (S,N)
    tpw_b0 = np.repeat(tpw0, N0, axis=1).astype(np.float32)
    
    X0 = np.stack([T, logp, q, Ts_b0, cwp_n0, rw_n0, tpw_b0], axis=-1).astype(np.float32)  # (S,N,7)

    # ---------- split ----------
    perm = rng.permutation(S)
    tr = perm[:int(0.8 * S)]
    va = perm[int(0.8 * S):]

    # base train/val
    Xtr0, logptr0, HRtr0, Fnettr0 = X0[tr], logp[tr], HR[tr], Fnet[tr]
    Ftoa_tr0, Fboa_tr0 = F_toa[tr], F_boa[tr]

    Xva0, logpva0, HRva0 = X0[va], logp[va], HR[va]
    Ftoa_va0, Fboa_va0 = F_toa[va], F_boa[va]

    # ---------- normalization stats on TRAIN (flatten over levels) ----------
    X_mu, X_std = zfit(Xtr0.reshape(-1, Xtr0.shape[-1]))
    H_mu, H_std = zfit(HRtr0.reshape(-1, 1))
    
    # Calculate separate stats for TOA and BOA
    Ftoa_mu, Ftoa_std = zfit(Ftoa_tr0.reshape(-1, 1))
    Fboa_mu, Fboa_std = zfit(Fboa_tr0.reshape(-1, 1))

    # normalize base val tensors (NO regrid on val)
    Xva_n = zapply(Xva0, X_mu.reshape(1,1,-1), X_std.reshape(1,1,-1)).astype(np.float32)
    HRva_n = zapply(HRva0[..., None], H_mu, H_std).astype(np.float32)
    Ftoa_va_n = zapply(Ftoa_va0[:, None], Ftoa_mu, Ftoa_std).astype(np.float32)
    Fboa_va_n = zapply(Fboa_va0[:, None], Fboa_mu, Fboa_std).astype(np.float32)

    Xva_t = torch.tensor(Xva_n, dtype=torch.float32)
    logp_va_t = torch.tensor(logpva0, dtype=torch.float32)
    HRva_t = torch.tensor(HRva_n, dtype=torch.float32)
    Ftoa_va_t = torch.tensor(Ftoa_va_n, dtype=torch.float32)
    Fboa_va_t = torch.tensor(Fboa_va_n, dtype=torch.float32)

    # ---------- sampling weights (Ts-tail oversampling) ----------
    w = np.ones(len(tr), dtype=np.float32)
    Ts_tr = Ts[tr]
    w[Ts_tr >= float(args.Ts_tail)] *= float(args.tail_mult)

    sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(tr), replacement=True)

    # ---------- model ----------
    model = HR_TOA_BOA_Model(in_dim=7, hidden=args.hidden, K=args.K, L=args.L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    best = float("inf")
    best_path = os.path.join(args.out, "best_hr_toa_boa.pt")

    regrid_choices = [int(x.strip()) for x in args.regrid_choices.split(",") if x.strip()]
    print("Regrid N choices:", regrid_choices)
    print(f"Ts-tail oversampling: Ts >= {args.Ts_tail}K weight x{args.tail_mult}")
    print("Features:", ['T','logp','q','Ts_broadcast','cwp_norm','rw_norm','tpw'])

    # ---------- training loop ----------
    for ep in range(1, args.epochs + 1):
        model.train()

        # each epoch: sample indices via sampler, then apply regrid augmentation per mini-batch
        idx_epoch = np.array(list(sampler), dtype=np.int64)  # indices into tr-set (0..len(tr)-1)
        # build epoch arrays from base train arrays
        Xbase = Xtr0[idx_epoch]          # (B0,N0,7)
        logpbase = logptr0[idx_epoch]    # (B0,N0)
        HRbase = HRtr0[idx_epoch]        # (B0,N0)
        Fnetbase = Fnettr0[idx_epoch]    # (B0,N0)
        Ftoa_base = Ftoa_tr0[idx_epoch]  # (B0,)
        Fboa_base = Fboa_tr0[idx_epoch]  # (B0,)

        # mini-batch
        for st in range(0, Xbase.shape[0], args.batch):
            ed = min(st + args.batch, Xbase.shape[0])
            Xb0 = Xbase[st:ed]
            logpb0 = logpbase[st:ed]
            HRb0 = HRbase[st:ed]
            Fnetb0 = Fnetbase[st:ed]
            Ftoab0 = Ftoa_base[st:ed]
            Fboab0 = Fboa_base[st:ed]

            # regrid augmentation: draw a target N and interpolate everything
            # do multiple draws if regrid_mult>1 (accumulate grads)
            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0

            for _ in range(int(args.regrid_mult)):
                M = int(rng.choice(regrid_choices))
                new_logp = make_logp_grid_like(logpb0, M)

                # interpolate raw physical fields first
                # we need to rebuild cwp_norm/rw_norm on the new grid consistently
                Tb = regrid_profile_batch(Xb0[..., 0], logpb0, new_logp)  # T
                qb = regrid_profile_batch(Xb0[..., 2], logpb0, new_logp)  # q
                Ts_b = np.repeat(Ts[tr][idx_epoch][st:ed][:, None], M, axis=1).astype(np.float32)

                cwp_n, rw_n, tpw = cwp_rw_norm_from_q_logp_np(qb, new_logp)
                tpw_b = np.repeat(tpw, M, axis=1).astype(np.float32)

                Xb = np.stack([Tb, new_logp, qb, Ts_b, cwp_n, rw_n, tpw_b], axis=-1).astype(np.float32)
                HRb = regrid_profile_batch(HRb0, logpb0, new_logp).astype(np.float32)
                Fnetb = regrid_profile_batch(Fnetb0, logpb0, new_logp).astype(np.float32)

                # normalize with TRAIN stats (from original train)
                Xbn = zapply(Xb, X_mu.reshape(1,1,-1), X_std.reshape(1,1,-1)).astype(np.float32)
                HRbn = zapply(HRb[..., None], H_mu, H_std).astype(np.float32)
                Ftoan = zapply(Ftoab0[:, None], Ftoa_mu, Ftoa_std).astype(np.float32)
                Fboan = zapply(Fboab0[:, None], Fboa_mu, Fboa_std).astype(np.float32)

                xb = torch.tensor(Xbn, dtype=torch.float32, device=device)
                cb = torch.tensor(new_logp, dtype=torch.float32, device=device)
                hr_true = torch.tensor(HRbn, dtype=torch.float32, device=device)
                ftoa_true = torch.tensor(Ftoan, dtype=torch.float32, device=device)
                fboa_true = torch.tensor(Fboan, dtype=torch.float32, device=device)
                fnet_true_phys = torch.tensor(Fnetb, dtype=torch.float32, device=device).unsqueeze(-1)

                hr_pred, ftoa_pred, fboa_pred = model(xb, cb)

                L_hr = loss_fn(hr_pred, hr_true)
                L_toa = loss_fn(ftoa_pred, ftoa_true)
                L_boa = loss_fn(fboa_pred, fboa_true)

                # --- Physics Constraint: Reconstruct profile inside training ---
                # Unnormalize predictions to physical space
                hr_phys = hr_pred * torch.tensor(H_std, device=device) + torch.tensor(H_mu, device=device)
                
                # CRITICAL FIX: Anchor the integration at the TRUE TOA to prevent the model 
                # from shifting the TOA prediction to compensate for HR drift.
                # The TOA head is trained separately via L_toa.
                ftoa_true_phys = fnet_true_phys[:, 0, :] # (B, 1)

                # Integrate: F(p) = F_toa_TRUE + cumsum(HR, p) (Forward Euler)
                integ = cumsum_batch_torch(hr_phys, cb)
                fnet_recon_anchored = ftoa_true_phys.unsqueeze(1) + integ 

                # Loss on profile (normalized by Ftoa_std to match scale of other losses approx)
                f_scale = torch.tensor(Ftoa_std, device=device)
                L_phys = loss_fn(fnet_recon_anchored / f_scale, fnet_true_phys / f_scale)

                loss = args.lam_hr * L_hr + args.lam_toa * L_toa + args.lam_boa * L_boa + args.lam_phys * L_phys

                # optional bottom-k mean HR loss
                if args.bot_k > 0 and args.lam_bot > 0:
                    k = min(int(args.bot_k), M)
                    hrp = hr_pred[:, -k:, :]
                    hrt = hr_true[:, -k:, :]
                    loss_bot = loss_fn(hrp.mean(dim=1, keepdim=True), hrt.mean(dim=1, keepdim=True))
                    loss = loss + float(args.lam_bot) * loss_bot

                loss_acc = loss_acc + loss

            loss_acc.backward()
            opt.step()

        # ---------- val ----------
        mse_hr, mse_toa, mse_boa, rmse_hr = eval_losses(
            model, Xva_t, logp_va_t, HRva_t, Ftoa_va_t, Fboa_va_t,
            H_mu=float(H_mu.squeeze()), H_std=float(H_std.squeeze()),
            Ftoa_mu=float(Ftoa_mu.squeeze()), Ftoa_std=float(Ftoa_std.squeeze()),
            Fboa_mu=float(Fboa_mu.squeeze()), Fboa_std=float(Fboa_std.squeeze()),
            device=device
        )

        # use HR RMSE (physical) as plateau metric
        sched.step(rmse_hr)

        if rmse_hr < best:
            best = rmse_hr
            torch.save({
                "state_dict": model.state_dict(),
                "cfg": vars(args),
                "X_mu": X_mu.squeeze(0).tolist(),
                "X_std": X_std.squeeze(0).tolist(),
                "H_mu": float(H_mu.squeeze()),
                "H_std": float(H_std.squeeze()),
                "Ftoa_mu": float(Ftoa_mu.squeeze()),
                "Ftoa_std": float(Ftoa_std.squeeze()),
                "Fboa_mu": float(Fboa_mu.squeeze()),
                "Fboa_std": float(Fboa_std.squeeze()),
                "features": ["T", "logp", "q", "Ts_broadcast", "cwp_norm", "rw_norm", "tpw"],
                "targets": ["HR_dF_dlogp", "F_TOA", "F_BOA"],
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | "
                  f"val HR_RMSE={rmse_hr:.4f} | "
                  f"mse(hr/toa/boa)={mse_hr:.4e}/{mse_toa:.4e}/{mse_boa:.4e} | "
                  f"L_phys_term(batch)={L_phys.item():.4e}")

    print(f"\nSaved best checkpoint: {best_path} (best val HR_RMSE={best:.4f})")

if __name__ == "__main__":
    main()
