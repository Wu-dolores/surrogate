import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

torch.set_num_threads(8)

# ---------------- utils ----------------
def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-12)

def enforce_toa_to_boa_numpy(logp, T, q, Fnet, HR=None):
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T    = T[:, ::-1].copy()
        q    = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()
        if HR is not None:
            HR = HR[:, ::-1].copy()
    return logp, T, q, Fnet, HR

def cumtrapz_batch_np(y, x):
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = np.zeros_like(y, dtype=np.float32)
    I[:, 1:] = np.cumsum(inc, axis=1)
    return I

def cwp_rw_norm_from_q_logp_np(q, logp):
    cwp = cumtrapz_batch_np(q, logp).astype(np.float32)
    total = cwp[:, -1:]
    rw = (total - cwp).astype(np.float32)
    denom = total + 1e-6
    cwp_n = (cwp / denom).astype(np.float32)
    rw_n  = (rw  / denom).astype(np.float32)
    return cwp_n, rw_n, total

def regrid_profile_batch(x, logp, new_logp):
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
    top = logp[:, 0:1]
    bot = logp[:, -1:]
    a = np.linspace(0.0, 1.0, M, dtype=np.float32)[None, :]
    return (top + (bot - top) * a).astype(np.float32)

def cumsum_batch_torch(y, x):
    if y.shape[-1] == 1 and y.ndim == 3:
        y = y.squeeze(-1)
    dx = x[:, 1:] - x[:, :-1]
    inc = y[:, :-1] * dx
    B, N = y.shape
    I = torch.zeros((B, N), dtype=y.dtype, device=y.device)
    I[:, 1:] = torch.cumsum(inc, dim=1)
    return I.unsqueeze(-1)

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
    def __init__(self, in_dim=6, hidden=128, K=6, L=4, ts_idx=3):
        super().__init__()
        self.ts_idx = ts_idx
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
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        # IMPROVEMENT: BOA head now takes (global + local + Ts)
        self.boa_head = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),  # +1 for Ts Skip Connection
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)
        hr = self.hr_head(h)         # (B,N,1)
        g = h.mean(dim=1)            # (B,H)
        
        h_toa = h[:, 0, :]           # (B,H)
        h_boa = h[:, -1, :]          # (B,H)
        
        # Extract Ts (normalized) from input x
        # x is (B, N, in_dim). Ts is broadcasted, so any N index is fine.
        ts_skip = x[:, 0, self.ts_idx:self.ts_idx+1] # (B, 1)

        f_toa = self.toa_head(torch.cat([g, h_toa], dim=-1))
        f_boa = self.boa_head(torch.cat([g, h_boa, ts_skip], dim=-1)) # Skip connection
        
        return hr, f_toa, f_boa

# ---------------- training/eval ----------------
@torch.no_grad()
def eval_losses(model, Xva_t, logp_va_t, Hva_t, Ftoa_va_t, Fboa_va_t,
                H_mu, H_std, Ftoa_mu, Ftoa_std, Fboa_mu, Fboa_std, device):
    model.eval()
    hrn, ftoan, fboan = model(Xva_t.to(device), logp_va_t.to(device))

    # Preds unnormalized
    hr_pred = hrn.cpu().numpy() * H_std + H_mu
    ftoa_pred = ftoan.cpu().numpy() * Ftoa_std + Ftoa_mu
    fboa_pred = fboan.cpu().numpy() * Fboa_std + Fboa_mu

    # Targets unnormalized
    H_true = Hva_t.numpy() * H_std + H_mu
    Ftoa_true = Ftoa_va_t.numpy() * Ftoa_std + Ftoa_mu
    Fboa_true = Fboa_va_t.numpy() * Fboa_std + Fboa_mu

    mse_hr = np.mean((hr_pred.squeeze() - H_true.squeeze())**2)
    mse_toa = np.mean((ftoa_pred.flatten() - Ftoa_true.flatten())**2)
    mse_boa = np.mean((fboa_pred.flatten() - Fboa_true.flatten())**2)
    return mse_hr, mse_toa, mse_boa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--regrid_choices", type=str, default="40,80,120,160")
    ap.add_argument("--regrid_mult", type=float, default=1.0)
    ap.add_argument("--Ts_tail", type=float, default=320.0)
    ap.add_argument("--tail_mult", type=float, default=2.0)
    ap.add_argument("--loss_weights", type=str, default="1,1,1,0", help="w_hr,w_toa,w_boa,w_phys")
    ap.add_argument("--ckpt", type=str, default="", help="Path to pretrained checkpoint to load")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    w_list = [float(x) for x in args.loss_weights.split(",")]
    w_hr, w_toa, w_boa, w_phys = w_list[0], w_list[1], w_list[2], w_list[3]
    print(f"Loss weights: HR={w_hr}, TOA={w_toa}, BOA={w_boa}, PHYS={w_phys}")

    d = np.load(args.data, allow_pickle=True)
    logp = d["logp_arr"].astype(np.float32)
    T    = d["T_arr"].astype(np.float32)
    q    = d["q_arr"].astype(np.float32)
    Ts   = d["Ts_K"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)

    # Need HR targets. Compute using Forward Difference consistent with Forward Euler Recon
    # HR[i] = (F[i+1] - F[i]) / (logp[i+1] - logp[i])
    HR = np.zeros_like(Fnet)
    logp, T, q, Fnet, HR = enforce_toa_to_boa_numpy(logp, T, q, Fnet, HR)

    S, N0 = Fnet.shape
    Ts_b0 = np.repeat(Ts[:, None], N0, axis=1).astype(np.float32)

    dF = Fnet[:, 1:] - Fnet[:, :-1]
    dp = logp[:, 1:] - logp[:, :-1] + 1e-9
    slope = dF / dp
    HR[:, :-1] = slope
    HR[:, -1] = slope[:, -1]

    F_toa = Fnet[:, 0].astype(np.float32)
    F_boa = Fnet[:, -1].astype(np.float32)

    cwp_n0, rw_n0, tpw0 = cwp_rw_norm_from_q_logp_np(q, logp)
    tpw_b0 = np.repeat(tpw0, N0, axis=1).astype(np.float32)
    
    # Feature list: T, logp, q, Ts_broadcast, cwp_norm, rw_norm, tpw
    # Ts is at index 3
    X0 = np.stack([T, logp, q, Ts_b0, cwp_n0, rw_n0, tpw_b0], axis=-1).astype(np.float32)
    TS_IDX = 3

    rng = np.random.RandomState(42)
    perm = rng.permutation(S)
    tr = perm[:int(0.8 * S)]
    va = perm[int(0.8 * S):]

    Xtr0, logptr0, HRtr0, Fnettr0 = X0[tr], logp[tr], HR[tr], Fnet[tr]
    Ftoa_tr0, Fboa_tr0 = F_toa[tr], F_boa[tr]

    Xva0, logpva0, HRva0 = X0[va], logp[va], HR[va]
    Ftoa_va0, Fboa_va0 = F_toa[va], F_boa[va]

    X_mu, X_std = zfit(Xtr0.reshape(-1, Xtr0.shape[-1]))
    H_mu, H_std = zfit(HRtr0.reshape(-1, 1))
    Ftoa_mu, Ftoa_std = zfit(Ftoa_tr0.reshape(-1, 1))
    Fboa_mu, Fboa_std = zfit(Fboa_tr0.reshape(-1, 1))

    # Overwrite stats if loading from checkpoint (Critical for Fine-Tuning)
    if args.ckpt:
        print(f"Loading normalization stats from {args.ckpt}...")
        cdata = torch.load(args.ckpt, map_location="cpu")
        if "X_mu" in cdata: X_mu = cdata["X_mu"]
        if "X_std" in cdata: X_std = cdata["X_std"]
        if "H_mu" in cdata: H_mu = cdata["H_mu"]
        if "H_std" in cdata: H_std = cdata["H_std"]
        if "Ftoa_mu" in cdata: Ftoa_mu = cdata["Ftoa_mu"]
        if "Ftoa_std" in cdata: Ftoa_std = cdata["Ftoa_std"]
        if "Fboa_mu" in cdata: Fboa_mu = cdata["Fboa_mu"]
        if "Fboa_std" in cdata: Fboa_std = cdata["Fboa_std"]
        print("Stats overwritten from checkpoint.")

    Xva_n = zapply(Xva0, X_mu.reshape(1,1,-1), X_std.reshape(1,1,-1)).astype(np.float32)
    HRva_n = zapply(HRva0[..., None], H_mu, H_std).astype(np.float32)
    Ftoa_va_n = zapply(Ftoa_va0[:, None], Ftoa_mu, Ftoa_std).astype(np.float32)
    Fboa_va_n = zapply(Fboa_va0[:, None], Fboa_mu, Fboa_std).astype(np.float32)

    Xva_t = torch.tensor(Xva_n, dtype=torch.float32)
    logp_va_t = torch.tensor(logpva0, dtype=torch.float32)
    HRva_t = torch.tensor(HRva_n, dtype=torch.float32)
    Ftoa_va_t = torch.tensor(Ftoa_va_n, dtype=torch.float32)
    Fboa_va_t = torch.tensor(Fboa_va_n, dtype=torch.float32)

    w = np.ones(len(tr), dtype=np.float32)
    Ts_tr = Ts[tr]
    w[Ts_tr >= float(args.Ts_tail)] *= float(args.tail_mult)
    sampler = WeightedRandomSampler(weights=torch.tensor(w), num_samples=len(tr), replacement=True)

    model = HR_TOA_BOA_Model(in_dim=7, hidden=args.hidden, K=args.K, L=args.L, ts_idx=TS_IDX).to(device)
    
    if args.ckpt:
        print(f"Loading pretrained weights from {args.ckpt}...")
        ckpt_data = torch.load(args.ckpt, map_location=device)
        # Load state dict, but ignore any size mismatches if we changed heads (should be fine here)
        model.load_state_dict(ckpt_data["state_dict"], strict=False)
        
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    best = float("inf")
    best_path = os.path.join(args.out, "best_hr_toa_boa.pt")
    regrid_choices = [int(x.strip()) for x in args.regrid_choices.split(",") if x.strip()]

    print("Starting training with Ts skip connection...")
    for ep in range(1, args.epochs + 1):
        model.train()
        idx_epoch = np.array(list(sampler), dtype=np.int64)
        Xbase = Xtr0[idx_epoch]
        logpbase = logptr0[idx_epoch]
        HRbase = HRtr0[idx_epoch]
        Fnetbase = Fnettr0[idx_epoch]
        Ftoa_base = Ftoa_tr0[idx_epoch]
        Fboa_base = Fboa_tr0[idx_epoch]

        for st in range(0, Xbase.shape[0], args.batch):
            ed = min(st + args.batch, Xbase.shape[0])
            Xb0 = Xbase[st:ed]
            logpb0 = logpbase[st:ed]
            HRb0 = HRbase[st:ed]
            Fnetb0 = Fnetbase[st:ed]
            Ftoab0 = Ftoa_base[st:ed]
            Fboab0 = Fboa_base[st:ed]

            opt.zero_grad(set_to_none=True)
            loss_acc = 0.0

            for _ in range(int(args.regrid_mult)):
                M = int(rng.choice(regrid_choices))
                new_logp = make_logp_grid_like(logpb0, M)

                Tb = regrid_profile_batch(Xb0[..., 0], logpb0, new_logp)
                qb = regrid_profile_batch(Xb0[..., 2], logpb0, new_logp)
                
                # Reconstruct Ts broadcast
                # Original Xb0[..., 3] is Ts_broadcast. 
                # We can just pick the first column since it is constant in N.
                ts_vals = Xb0[:, 0, 3:4] # (B,1)
                Ts_b = np.repeat(ts_vals[:, None, :], M, axis=1).squeeze(-1).astype(np.float32)

                cwp_n, rw_n, tpw = cwp_rw_norm_from_q_logp_np(qb, new_logp)
                tpw_b = np.repeat(tpw, M, axis=1).astype(np.float32)

                # FEATURES: T, logp, q, Ts_b, cwp, rw, tpw
                Xb = np.stack([Tb, new_logp, qb, Ts_b, cwp_n, rw_n, tpw_b], axis=-1).astype(np.float32)
                HRb = regrid_profile_batch(HRb0, logpb0, new_logp).astype(np.float32)
                Fnetb = regrid_profile_batch(Fnetb0, logpb0, new_logp).astype(np.float32)
                
                Ftoab = Ftoab0
                Fboab = Fboab0

                Xb_n = zapply(Xb, X_mu.reshape(1,1,-1), X_std.reshape(1,1,-1))
                HRb_n = zapply(HRb[..., None], H_mu, H_std)
                Ftoab_n = zapply(Ftoab[:, None], Ftoa_mu, Ftoa_std)
                Fboab_n = zapply(Fboab[:, None], Fboa_mu, Fboa_std)

                xt = torch.tensor(Xb_n, dtype=torch.float32, device=device)
                ct = torch.tensor(new_logp, dtype=torch.float32, device=device)
                ht = torch.tensor(HRb_n, dtype=torch.float32, device=device)
                ftt = torch.tensor(Ftoab_n, dtype=torch.float32, device=device)
                fbt = torch.tensor(Fboab_n, dtype=torch.float32, device=device)

                hr_p, ft_p, fb_p = model(xt, ct)

                l_hr = loss_fn(hr_p, ht)
                l_toa = loss_fn(ft_p, ftt)
                l_boa = loss_fn(fb_p, fbt)
                
                loss_step = w_hr*l_hr + w_toa*l_toa + w_boa*l_boa

                if w_phys > 0:
                    # Physics Consistency Loss
                    # Integrate HR_pred (unnormalized) to get Profile (unnormalized)
                    # Use Cumsum (Forward Euler)
                    # hr_unnorm = hr_p * H_std + H_mu
                    # But doing it in normalized space is hard. So unnormalize.
                    H_std_t = torch.tensor(H_std, device=device)
                    H_mu_t = torch.tensor(H_mu, device=device)
                    Ftoa_std_t = torch.tensor(Ftoa_std, device=device)
                    Ftoa_mu_t = torch.tensor(Ftoa_mu, device=device)
                    
                    hr_un = hr_p.squeeze(-1) * H_std_t + H_mu_t
                    I = cumsum_batch_torch(hr_un, ct) # (B, N, 1)
                    
                    ft_un = ft_p * Ftoa_std_t + Ftoa_mu_t
                    
                    # F_recon = F_top + Integral
                    F_rec = ft_un.unsqueeze(1) + I # (B, N, 1)
                    
                    # Target unnormalized
                    Fnet_t = torch.tensor(Fnetb, dtype=torch.float32, device=device).unsqueeze(-1)
                    
                    l_phys = torch.mean((F_rec - Fnet_t)**2)
                    loss_step += w_phys * l_phys

                loss_step.backward()
                loss_acc += loss_step.item()
            
            opt.step()

        if ep % 5 == 0:
            val_hr, val_toa, val_boa = eval_losses(
                model, Xva_t, logp_va_t, HRva_t, Ftoa_va_t, Fboa_va_t,
                H_mu, H_std, Ftoa_mu, Ftoa_std, Fboa_mu, Fboa_std, device
            )
            print(f"Ep {ep}: Loss={loss_acc:.4f} | Val RMSE: HR={np.sqrt(val_hr):.4f} TOA={np.sqrt(val_toa):.4f} BOA={np.sqrt(val_boa):.4f}")
            
            # Simple early stopping on sum of RMSEs
            score = val_hr + val_toa + val_boa 
            sched.step(score)
            
            if score < best:
                best = score
                save_dict = {
                    "state_dict": model.state_dict(),
                    "cfg": vars(args),
                    "X_mu": X_mu, "X_std": X_std,
                    "H_mu": H_mu, "H_std": H_std,
                    "Ftoa_mu": Ftoa_mu, "Ftoa_std": Ftoa_std,
                    "Fboa_mu": Fboa_mu, "Fboa_std": Fboa_std,
                    "features": ['T','logp','q','Ts_broadcast','cwp_norm','rw_norm','tpw'],
                    "ts_idx": TS_IDX
                }
                torch.save(save_dict, best_path)
                print(f"  Saved best to {best_path}")

if __name__ == "__main__":
    main()
