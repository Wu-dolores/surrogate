import argparse, os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def zapply(x, mu, std):
    return (x - mu) / (std + 1e-12)

def zinvert(xn, mu, std):
    return xn * std + mu

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

def load_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    hidden = int(cfg.get("hidden", 128))
    K = int(cfg.get("K", 6))
    L = int(cfg.get("L", 4))

    features = ckpt.get("features", ["T", "logp", "q", "Ts_broadcast"])
    in_dim = len(features)

    model = V1LocalGNO(in_dim=in_dim, hidden=hidden, K=K, L=L).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    X_mu = np.array(ckpt["X_mu"], dtype=np.float32).reshape(1, 1, -1)
    X_std = np.array(ckpt["X_std"], dtype=np.float32).reshape(1, 1, -1)
    Y_mu = float(ckpt["Y_mu"])
    Y_std = float(ckpt["Y_std"])
    return model, features, X_mu, X_std, Y_mu, Y_std, ckpt

@torch.no_grad()
def predict(model, Xn, coord, batch=256, device="cpu"):
    preds = []
    for i in range(0, Xn.shape[0], batch):
        xb = torch.tensor(Xn[i:i+batch], dtype=torch.float32, device=device)
        cb = torch.tensor(coord[i:i+batch], dtype=torch.float32, device=device)
        predn = model(xb, cb).cpu().numpy()
        preds.append(predn)
    return np.concatenate(preds, axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--worst_k", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model, features, X_mu, X_std, Y_mu, Y_std, ckpt = load_ckpt(args.ckpt, device)
    print("Features used:", features)

    d = np.load(args.data, allow_pickle=True)
    Ts   = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)     # (S,N)
    T    = d["T_arr"].astype(np.float32)        # (S,N)
    q    = d["q_arr"].astype(np.float32)        # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)     # (S,N)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    # Build X in the same order as features in ckpt
    feats = []
    for f in features:
        if f == "T":
            feats.append(T)
        elif f == "logp":
            feats.append(logp)
        elif f == "q":
            feats.append(q)
        elif f == "Ts_broadcast":
            feats.append(Ts_b)
        else:
            raise ValueError(f"Unknown feature in ckpt: {f}")
    X = np.stack(feats, axis=-1).astype(np.float32)  # (S,N,F)

    Y = Fnet[..., None].astype(np.float32)

    # normalize
    Xn = zapply(X, X_mu, X_std)
    Yn = zapply(Y, Y_mu, Y_std)

    # predict normalized
    predn = predict(model, Xn, logp, batch=args.batch, device=device)
    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)
    true = Fnet

    # metrics
    prof_rmse = np.sqrt(np.mean((pred - true)**2, axis=1))          # per-sample
    toa_abs   = np.abs(pred[:, 0]  - true[:, 0])
    boa_abs   = np.abs(pred[:, -1] - true[:, -1])

    rmse_prof = float(np.sqrt(np.mean((pred - true)**2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0])**2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1])**2)))

    print("\n=== REAL testset eval (diag) ===")
    print("Data:", args.data)
    print(f"RMSE profile = {rmse_prof:.3f} W/m^2")
    print(f"RMSE TOA     = {rmse_toa:.3f} W/m^2")
    print(f"RMSE BOA     = {rmse_boa:.3f} W/m^2")

    # ---- level-wise diagnostics ----
    err = pred - true  # (S,N)
    mean_err = err.mean(axis=0)
    mae = np.abs(err).mean(axis=0)

    # Use mean coord (they may vary slightly, but for your generated grids it's usually identical)
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
    plt.ylabel("Mean |pred-true| (W/m^2)")
    plt.title("MAE vs logp")
    plt.savefig(os.path.join(args.out, "mae_vs_logp.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # ---- error vs Ts diagnostics ----
    plt.figure()
    plt.plot(Ts, prof_rmse, marker=".", linestyle="None")
    plt.xlabel("Ts (K)")
    plt.ylabel("Profile RMSE (W/m^2)")
    plt.title("Profile RMSE vs Ts")
    plt.savefig(os.path.join(args.out, "profile_rmse_vs_Ts.png"), dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(Ts, boa_abs, marker=".", linestyle="None")
    plt.xlabel("Ts (K)")
    plt.ylabel("|BOA pred-true| (W/m^2)")
    plt.title("BOA abs error vs Ts")
    plt.savefig(os.path.join(args.out, "boa_abs_error_vs_Ts.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # ---- worst-k overlays ----
    worst_k = int(args.worst_k)
    worst_idx = np.argsort(-prof_rmse)[:worst_k]

    plt.figure()
    for j, i in enumerate(worst_idx):
        plt.plot(logp[i], true[i], alpha=0.9)
        plt.plot(logp[i], pred[i], alpha=0.9, linestyle="--")
    plt.gca().invert_xaxis()
    plt.xlabel("log(p) [ln(Pa)] (inverted)")
    plt.ylabel("Fnet (W/m^2)")
    plt.title(f"Worst-{worst_k} profiles (solid=true, dashed=pred)")
    plt.savefig(os.path.join(args.out, f"worst{worst_k}_overlays.png"), dpi=160, bbox_inches="tight")
    plt.close()

    # Save a small text summary too
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"RMSE_profile {rmse_prof:.6f}\n")
        f.write(f"RMSE_TOA {rmse_toa:.6f}\n")
        f.write(f"RMSE_BOA {rmse_boa:.6f}\n")
        f.write(f"N {N}\n")
        f.write("features " + ",".join(features) + "\n")

    print(f"\nSaved diag plots to: {args.out}/")
    print(" - mean_error_vs_logp.png")
    print(" - mae_vs_logp.png")
    print(" - profile_rmse_vs_Ts.png")
    print(" - boa_abs_error_vs_Ts.png")
    print(f" - worst{worst_k}_overlays.png")

if __name__ == "__main__":
    main()
