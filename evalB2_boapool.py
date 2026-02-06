import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# -------------------------
# Utils: z-norm
# -------------------------
def zapply(x, mu, std):
    return (x - mu) / (std + 1e-6)

def zinvert(xn, mu, std):
    return xn * (std + 1e-6) + mu

def as_np(x):
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)

# -------------------------
# Model: Local-GNO backbone + 3 heads (HR profile, TOA scalar, BOA scalar)
# NOTE: This must match your training architecture.
# If your ckpt uses a different class name but same tensor keys, it will load.
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
    """
    Outputs:
      hr:  (B,N,1)   predicted HR (normalized space)
      toa: (B,1)     predicted TOA flux (normalized space)
      boa: (B,1)     predicted BOA flux (normalized space)
    """
    def __init__(self, in_dim=4, hidden=128, K=6, L=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])

        # HR head (per-layer)
        self.hr_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # TOA/BOA heads (global)
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
        # x: (B,N,in_dim), coord: (B,N)
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)

        hr = self.hr_head(h)  # (B,N,1)

        # global pooling: mean over N
        g = h.mean(dim=1)     # (B,H)
        toa = self.toa_head(g)
        boa = self.boa_head(g)
        return hr, toa, boa

# -------------------------
# Recon: integrate HR -> flux shape, then enforce TOA+BOA endpoints by alpha blend
# -------------------------
def cumtrapz_np(y, x):
    """
    y,x: (B,N)
    returns integral from 0..i with trapezoid rule, same shape (B,N), with out[:,0]=0
    """
    B, N = y.shape
    out = np.zeros((B, N), dtype=np.float32)
    dx = x[:, 1:] - x[:, :-1]
    area = 0.5 * (y[:, 1:] + y[:, :-1]) * dx
    out[:, 1:] = np.cumsum(area, axis=1)
    return out

@torch.no_grad()
def predict_recon(model, Xn, logp, F_mu, F_std, H_mu, H_std,
                  alpha_gamma=1.0, batch=256, device="cpu"):
    """
    Xn:   (S,N,Fin) normalized features
    logp: (S,N) raw coord (ln(Pa))
    F_mu,F_std: scalars for Fnet
    H_mu,H_std: scalars for HR (dF/dlogp)
    Returns:
      F_pred: (S,N) in physical space (W/m^2)
      HR_pred:(S,N) in physical space (same units as training target)
      TOA_pred:(S,)
      BOA_pred:(S,)
    """
    model.eval()
    S, N, Fin = Xn.shape

    F_all = np.zeros((S, N), dtype=np.float32)
    H_all = np.zeros((S, N), dtype=np.float32)
    TOA_all = np.zeros((S,), dtype=np.float32)
    BOA_all = np.zeros((S,), dtype=np.float32)

    for i in range(0, S, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(logp[i:i+batch], dtype=torch.float32, device=device)

        hr_n, toa_n, boa_n = model(xb, cb)  # normalized space
        hr_n = hr_n.detach().cpu().numpy().squeeze(-1)   # (B,N)
        toa_n = toa_n.detach().cpu().numpy().squeeze(-1) # (B,)
        boa_n = boa_n.detach().cpu().numpy().squeeze(-1) # (B,)

        # invert HR, TOA, BOA to physical space
        HR = zinvert(hr_n, H_mu, H_std).astype(np.float32)   # (B,N)
        TOA = zinvert(toa_n, F_mu, F_std).astype(np.float32) # (B,)
        BOA = zinvert(boa_n, F_mu, F_std).astype(np.float32) # (B,)

        # integrate HR over logp -> flux shape up to constant
        c = logp[i:i+batch].astype(np.float32)
        f_tilde = cumtrapz_np(HR, c)  # (B,N), f_tilde[:,0]=0

        # anchor TOA at top
        f1 = f_tilde + TOA[:, None]

        # enforce BOA by alpha blend (delta distributed from top->bottom)
        denom = (c[:, -1:] - c[:, 0:1] + 1e-6)
        alpha = (c - c[:, 0:1]) / denom
        alpha = np.clip(alpha, 0.0, 1.0) ** float(alpha_gamma)

        delta = (BOA - f1[:, -1]).astype(np.float32)  # (B,)
        Frec = f1 + delta[:, None] * alpha

        F_all[i:i+batch] = Frec
        H_all[i:i+batch] = HR
        TOA_all[i:i+batch] = TOA
        BOA_all[i:i+batch] = BOA

    return F_all, H_all, TOA_all, BOA_all

# -------------------------
# BOA bottom-K pooling (eval-only)
# -------------------------
def boa_pool(pred_F, true_F, K=6, mode="mean"):
    """
    pred_F,true_F: (S,N)
    K: bottom K levels
    mode: "mean" or "linear" (surface heavier)
    Returns pooled scalars: (S,), (S,)
    """
    if K <= 1:
        return pred_F[:, -1], true_F[:, -1]

    K = int(K)
    pred_bot = pred_F[:, -K:]
    true_bot = true_F[:, -K:]

    if mode == "mean":
        return pred_bot.mean(axis=1), true_bot.mean(axis=1)

    if mode == "linear":
        # weights: higher at surface (last index)
        w = np.linspace(0.5, 1.0, K).astype(np.float32)
        w = w / (w.sum() + 1e-6)
        return (pred_bot * w[None, :]).sum(axis=1), (true_bot * w[None, :]).sum(axis=1)

    raise ValueError(f"Unknown boa_pool_mode: {mode}")

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--alpha_gamma", type=float, default=1.0)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--boa_k", type=int, default=6)
    ap.add_argument("--boa_pool_mode", type=str, default="mean", choices=["mean", "linear"])
    ap.add_argument("--topk_worst", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    features = ckpt.get("features", ["T", "logp", "q", "Ts_broadcast"])
    print("Features used:", features)

    # pull dims
    in_dim = len(features)
    hidden = int(cfg.get("hidden", 128))
    K = int(cfg.get("K", 6))
    L = int(cfg.get("L", 4))

    # stats (allow either list or scalar)
    X_mu = np.array(ckpt["X_mu"], dtype=np.float32).reshape(1, 1, -1)
    X_std = np.array(ckpt["X_std"], dtype=np.float32).reshape(1, 1, -1)

    # Flux stats
    F_mu = float(ckpt.get("F_mu", ckpt.get("Y_mu", 0.0)))
    F_std = float(ckpt.get("F_std", ckpt.get("Y_std", 1.0)))

    # HR stats
    H_mu = float(ckpt.get("H_mu", 0.0))
    H_std = float(ckpt.get("H_std", 1.0))

    # ---- build model
    model = GNO_HR_TOA_BOA(in_dim=in_dim, hidden=hidden, K=K, L=L).to(device)
    try:
        model.load_state_dict(ckpt["state_dict"], strict=True)
    except Exception as e:
        print("\n[ERROR] load_state_dict failed. This usually means eval model class != training model class.")
        print("Exception:", repr(e))
        print("Fix: make eval model match your training script architecture exactly.")
        raise

    # ---- load data
    d = np.load(args.data, allow_pickle=True)
    logp = d["logp_arr"].astype(np.float32)   # (S,N)
    T = d["T_arr"].astype(np.float32)         # (S,N)
    q = d["q_arr"].astype(np.float32)         # (S,N)
    Ts = d["Ts_K"].astype(np.float32)         # (S,)
    F_true = d["Fnet_arr"].astype(np.float32) # (S,N)

    S, N = F_true.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    # build feature tensor in same order as "features"
    feat_map = {
        "T": T,
        "logp": logp,
        "q": q,
        "Ts_broadcast": Ts_b,
    }
    X = np.stack([feat_map[k] for k in features], axis=-1).astype(np.float32)  # (S,N,Fin)

    # normalize
    Xn = zapply(X, X_mu, X_std).astype(np.float32)

    # ---- predict + recon
    F_pred, HR_pred, TOA_pred, BOA_pred = predict_recon(
        model,
        Xn,
        logp,
        F_mu=F_mu, F_std=F_std,
        H_mu=H_mu, H_std=H_std,
        alpha_gamma=args.alpha_gamma,
        batch=args.batch,
        device=device,
    )

    # ---- metrics
    rmse_prof = float(np.sqrt(np.mean((F_pred - F_true) ** 2)))
    rmse_toa  = float(np.sqrt(np.mean((F_pred[:, 0] - F_true[:, 0]) ** 2)))

    # BOA last
    rmse_boa_last = float(np.sqrt(np.mean((F_pred[:, -1] - F_true[:, -1]) ** 2)))

    # BOA pool-K (eval-only)
    boa_pred_pool, boa_true_pool = boa_pool(F_pred, F_true, K=args.boa_k, mode=args.boa_pool_mode)
    rmse_boa_pool = float(np.sqrt(np.mean((boa_pred_pool - boa_true_pool) ** 2)))

    print("\n=== REAL testset eval (Route-B2 recon + BOA bottom-K pooling) ===")
    print(f"Data: {os.path.basename(args.data)}")
    print(f"alpha_gamma = {args.alpha_gamma}")
    print(f"BOA pooling: K={args.boa_k}, mode={args.boa_pool_mode}")
    print(f"RMSE profile   = {rmse_prof:.3f} W/m^2")
    print(f"RMSE TOA       = {rmse_toa:.3f} W/m^2")
    print(f"RMSE BOA@last  = {rmse_boa_last:.3f} W/m^2")
    print(f"RMSE BOA@poolK = {rmse_boa_pool:.3f} W/m^2")

    # ---- worst samples by profile RMSE
    prof_rmse_each = np.sqrt(np.mean((F_pred - F_true) ** 2, axis=1))
    worst_idx = np.argsort(-prof_rmse_each)[:args.topk_worst]
    print(f"\nTop-{args.topk_worst} worst samples by profile RMSE:")
    for r, idx in enumerate(worst_idx, 1):
        toa_abs = abs(F_pred[idx, 0] - F_true[idx, 0])
        boa_abs = abs(F_pred[idx, -1] - F_true[idx, -1])
        print(f"  rank {r:02d}: idx={idx:04d} Ts={Ts[idx]:.2f}K  prof_RMSE={prof_rmse_each[idx]:.3f}  TOA_abs={toa_abs:.3f}  BOA_abs={boa_abs:.3f}")

    # ---- diag plots
    if args.diag:
        # mean error & MAE vs logp (use dataset-mean coord)
        err = (F_pred - F_true)  # (S,N)
        mean_err = err.mean(axis=0)
        mae = np.abs(err).mean(axis=0)

        # use average coord across samples (they should be close if same grid)
        c_plot = logp.mean(axis=0)

        plt.figure()
        plt.plot(c_plot, mean_err)
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Mean (pred-true) (W/m^2)")
        plt.title("Mean error vs logp")
        plt.savefig(os.path.join(args.out, "mean_error_vs_logp.png"), dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(c_plot, mae)
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Mean |pred-true| (W/m^2)")
        plt.title("MAE vs logp")
        plt.savefig(os.path.join(args.out, "mae_vs_logp.png"), dpi=160, bbox_inches="tight")
        plt.close()

        # BOA abs error vs Ts (both last and poolK)
        boa_abs_last = np.abs(F_pred[:, -1] - F_true[:, -1])
        boa_abs_pool = np.abs(boa_pred_pool - boa_true_pool)

        plt.figure()
        plt.plot(Ts, boa_abs_last, label="|BOA_last error|")
        plt.plot(Ts, boa_abs_pool, label=f"|BOA_poolK(K={args.boa_k}) error|")
        plt.xlabel("Ts (K)")
        plt.ylabel("|BOA error| (W/m^2)")
        plt.title("BOA absolute error vs Ts")
        plt.legend()
        plt.savefig(os.path.join(args.out, "boa_abs_error_vs_Ts.png"), dpi=160, bbox_inches="tight")
        plt.close()

        print(f"Saved diag plots to {args.out}/")

if __name__ == "__main__":
    main()
