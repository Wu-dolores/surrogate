import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def enforce_toa_to_boa_numpy(logp, T, q, Fnet):
    # ensure coord goes from TOA->BOA (logp small->large)
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T    = T[:, ::-1].copy()
        q    = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()
    return logp, T, q, Fnet

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-12)

def cumtrapz_batch_np(y, x):
    # y,x: (B,N), integrate along axis=1, out[:,0]=0
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = np.zeros_like(y, dtype=np.float32)
    I[:, 1:] = np.cumsum(inc, axis=1)
    return I

def cwp_rw_from_q_logp_np(q, logp):
    """
    cwp(i)=∫_{top→i} q dlogp
    rw(i) =∫_{i→surf} q dlogp
    """
    cwp = cumtrapz_batch_np(q, logp).astype(np.float32)
    total = cwp[:, -1:]  # (B,1)
    rw = (total - cwp).astype(np.float32)
    return cwp, rw

def alpha_full_column(logp, alpha_gamma):
    # alpha in [0,1] from TOA->BOA across whole column
    a = (logp - logp[:, 0:1]) / (logp[:, -1:] - logp[:, 0:1] + 1e-6)
    a = np.clip(a, 0.0, 1.0).astype(np.float32)
    return a ** float(alpha_gamma)

def alpha_bottom_window(logp, alpha_gamma, bot_window_k):
    """
    Only distribute delta in bottom K layers.
    alpha[:, :-K] = 0
    alpha[:, -K:] = ramp(0->1)^gamma
    """
    B, N = logp.shape
    K = int(bot_window_k)
    if K <= 0:
        return alpha_full_column(logp, alpha_gamma)
    K = max(1, min(K, N))

    a = np.zeros((B, N), dtype=np.float32)
    if K == 1:
        a[:, -1] = 1.0
        return a

    ramp = np.linspace(0.0, 1.0, K, dtype=np.float32)[None, :] ** float(alpha_gamma)
    a[:, -K:] = ramp
    return a

# ---- model definition (must match training) ----
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
        hr = self.hr_head(h)
        g = h.mean(dim=1)
        f_toa = self.toa_head(g)
        f_boa = self.boa_head(g)
        return hr, f_toa, f_boa

@torch.no_grad()
def predict_recon(model, Xn, logp, F_mu, F_std, H_mu, H_std,
                  alpha_gamma, bot_window_k, batch, device):
    model.eval()
    S = Xn.shape[0]
    preds = []
    for i in range(0, S, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(logp[i:i+batch], dtype=torch.float32, device=device)

        hrn, ftoan, fboan = model(xb, cb)

        hr = (hrn.squeeze(-1).cpu().numpy() * H_std + H_mu).astype(np.float32)     # (B,N)
        f_toa = (ftoan.squeeze(-1).cpu().numpy() * F_std + F_mu).astype(np.float32)
        f_boa = (fboan.squeeze(-1).cpu().numpy() * F_std + F_mu).astype(np.float32)

        I = cumtrapz_batch_np(hr, logp[i:i+batch])                                 # (B,N)
        f_tilde = f_toa[:, None] + I                                               # anchor at TOA
        delta = f_boa - f_tilde[:, -1]                                             # enforce BOA

        alpha = alpha_bottom_window(logp[i:i+batch], alpha_gamma, bot_window_k)    # (B,N)

        Frec = f_tilde + delta[:, None] * alpha
        preds.append(Frec.astype(np.float32))

    return np.concatenate(preds, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--worst_k", type=int, default=5)
    ap.add_argument("--alpha_gamma", type=float, default=1.0)
    ap.add_argument("--bot_window_k", type=int, default=0,
                    help="If >0, distribute BOA delta only in bottom K levels (recommended). "
                         "If 0, use full-column alpha as before.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden = int(cfg.get("hidden", 128))
    K = int(cfg.get("K", 6))
    L = int(cfg.get("L", 4))

    # auto-detect features / in_dim
    features = ckpt.get("features", ["T", "logp", "q", "Ts_broadcast"])
    in_dim = len(features)

    X_mu = np.array(ckpt["X_mu"], dtype=np.float32).reshape(1, 1, in_dim)
    X_std = np.array(ckpt["X_std"], dtype=np.float32).reshape(1, 1, in_dim)
    F_mu = float(ckpt["F_mu"]); F_std = float(ckpt["F_std"])
    H_mu = float(ckpt["H_mu"]); H_std = float(ckpt["H_std"])

    model = HR_TOA_BOA_Model(in_dim=in_dim, hidden=hidden, K=K, L=L).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)
    logp = d["logp_arr"].astype(np.float32)
    T    = d["T_arr"].astype(np.float32)
    q    = d["q_arr"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)

    # enforce ordering
    logp, T, q, Fnet = enforce_toa_to_boa_numpy(logp, T, q, Fnet)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    # build features in ckpt order
    feat_map = {
        "T": T,
        "logp": logp,
        "q": q,
        "Ts_broadcast": Ts_b,
    }

    if any(f in features for f in ["cwp", "rw", "cwp_norm", "rw_norm"]):
        cwp, rw = cwp_rw_from_q_logp_np(q, logp)
        total = cwp[:, -1:] + 1e-6
        cwp_n = cwp / total
        rw_n  = rw  / total
        feat_map["cwp"] = cwp
        feat_map["rw"] = rw
        feat_map["cwp_norm"] = cwp_n.astype(np.float32)
        feat_map["rw_norm"]  = rw_n.astype(np.float32)


    X_list = [feat_map[f] for f in features]
    X = np.stack(X_list, axis=-1).astype(np.float32)  # (S,N,in_dim)
    Xn = zapply(X, X_mu, X_std).astype(np.float32)

    pred = predict_recon(
        model, Xn, logp,
        F_mu, F_std, H_mu, H_std,
        alpha_gamma=args.alpha_gamma,
        bot_window_k=args.bot_window_k,
        batch=args.batch,
        device=device
    )
    true = Fnet

    rmse_prof = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1]) ** 2)))

    mode = f"bottom-window(K={args.bot_window_k})" if args.bot_window_k > 0 else "full-column"
    print("\n=== REAL testset eval (Route-B2 recon) ===")
    print("Data:", os.path.basename(args.data))
    print("Features used:", features)
    print(f"Recon alpha mode: {mode}, alpha_gamma={args.alpha_gamma}")
    print(f"RMSE profile = {rmse_prof:.3f} W/m^2")
    print(f"RMSE TOA     = {rmse_toa:.3f} W/m^2")
    print(f"RMSE BOA     = {rmse_boa:.3f} W/m^2")

    if args.diag:
        err = pred - true
        mean_err = err.mean(axis=0)
        mae = np.abs(err).mean(axis=0)
        coord = logp[0]

        plt.figure()
        plt.plot(coord, mean_err)
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Mean (pred-true) (W/m^2)")
        plt.title("Mean error vs logp")
        plt.savefig(os.path.join(args.out, "mean_error_vs_logp.png"), dpi=160, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(coord, mae)
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("MAE (W/m^2)")
        plt.title("MAE vs logp")
        plt.savefig(os.path.join(args.out, "mae_vs_logp.png"), dpi=160, bbox_inches="tight")
        plt.close()

        prof_rmse_each = np.sqrt(np.mean((pred - true) ** 2, axis=1))
        worst = np.argsort(-prof_rmse_each)[:args.worst_k]

        plt.figure()
        for i in worst:
            plt.plot(logp[i], true[i], alpha=0.9)
            plt.plot(logp[i], pred[i], alpha=0.9, linestyle="--")
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Fnet (W/m^2)")
        plt.title(f"Worst-{args.worst_k} profiles (solid=true, dashed=pred)")
        plt.savefig(os.path.join(args.out, f"worst{args.worst_k}_overlays.png"), dpi=160, bbox_inches="tight")
        plt.close()

        print(f"Saved diag plots to {args.out}/")

if __name__ == "__main__":
    main()