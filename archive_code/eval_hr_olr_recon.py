import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-12)

def cumtrapz_batch(y, x):
    dx = x[:, 1:] - x[:, :-1]
    avg = 0.5 * (y[:, 1:] + y[:, :-1])
    inc = avg * dx
    I = np.zeros_like(y)
    I[:, 1:] = np.cumsum(inc, axis=1)
    return I

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

class HR_OLR_Model(nn.Module):
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
        self.olr_head = nn.Sequential(
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
        olr = self.olr_head(g)        # (B,1)
        return hr, olr

@torch.no_grad()
def predict_recon(model, Xn, logp, F_mu, F_std, H_mu, H_std, device, batch=256):
    model.eval()
    S = Xn.shape[0]
    preds = []
    for i in range(0, S, batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(logp[i:i+batch], dtype=torch.float32, device=device)
        hrn, olrn = model(xb, cb)
        hr = (hrn.squeeze(-1).cpu().numpy() * H_std + H_mu)      # (B,N)
        olr = (olrn.squeeze(-1).cpu().numpy() * F_std + F_mu)    # (B,)
        I = cumtrapz_batch(hr, logp[i:i+batch])                  # numpy
        Frec = olr[:, None] + I
        preds.append(Frec)
    return np.concatenate(preds, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--diag", action="store_true")
    ap.add_argument("--worst_k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden = int(cfg.get("hidden", 128))
    K = int(cfg.get("K", 6))
    L = int(cfg.get("L", 4))

    X_mu = np.array(ckpt["X_mu"], dtype=np.float32).reshape(1, 1, 4)
    X_std = np.array(ckpt["X_std"], dtype=np.float32).reshape(1, 1, 4)
    F_mu = float(ckpt["F_mu"]); F_std = float(ckpt["F_std"])
    H_mu = float(ckpt["H_mu"]); H_std = float(ckpt["H_std"])

    model = HR_OLR_Model(in_dim=4, hidden=hidden, K=K, L=L).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)
    logp = d["logp_arr"].astype(np.float32)
    T = d["T_arr"].astype(np.float32)
    q = d["q_arr"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)
    # ---- enforce TOA->BOA ordering (logp increasing) ----
    if np.mean(logp[:, 0] > logp[:, -1]) > 0.5:
        logp = logp[:, ::-1].copy()
        T    = T[:, ::-1].copy()
        q    = q[:, ::-1].copy()
        Fnet = Fnet[:, ::-1].copy()


    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    X = np.stack([T, logp, q, Ts_b], axis=-1).astype(np.float32)
    Xn = zapply(X, X_mu, X_std)

    pred = predict_recon(model, Xn, logp, F_mu, F_std, H_mu, H_std, device=device)
    true = Fnet

    rmse_prof = float(np.sqrt(np.mean((pred - true) ** 2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0]) ** 2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1]) ** 2)))

    print("\n=== REAL testset eval (Route-B recon) ===")
    print("Data:", os.path.basename(args.data))
    print("Features used: ['T','logp','q','Ts_broadcast']")
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

        # worst overlays
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
