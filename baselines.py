import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def zscore_fit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zscore_apply(x, mu, std):
    return (x - mu) / std

def zscore_invert(xn, mu, std):
    return xn * std + mu

def interp_to_fixed(x, coord, n_fixed=128):
    """x: (S,N,F), coord: (S,N) monotone increasing. Return (S,n_fixed,F), (S,n_fixed)."""
    S, N, F = x.shape
    out = np.zeros((S, n_fixed, F), dtype=np.float32)
    c_out = np.zeros((S, n_fixed), dtype=np.float32)
    for i in range(S):
        c = coord[i]
        c_t = np.linspace(c[0], c[-1], n_fixed, dtype=np.float32)
        c_out[i] = c_t
        for f in range(F):
            out[i, :, f] = np.interp(c_t, c, x[i, :, f]).astype(np.float32)
    return out, c_out

class LayerMLP(nn.Module):
    """Baseline A: per-layer MLP (shared weights)."""
    def __init__(self, in_dim=4, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):  # (B,N,F) -> (B,N,1)
        return self.net(x)

class CNN1D(nn.Module):
    """Baseline B: 1D CNN on fixed N (after interpolation)."""
    def __init__(self, in_ch=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU(),
            nn.Conv1d(hidden, 1, 1)
        )
    def forward(self, x):  # (B,N,F)->(B,N,1)
        x = x.permute(0, 2, 1)
        y = self.net(x)
        return y.permute(0, 2, 1)

class LocalGNO(nn.Module):
    """Baseline C: Local GNO (no attention), index-neighborhood message passing with delta coord."""
    def __init__(self, in_dim=4, hidden=96, K=4, layers=2):
        super().__init__()
        self.K = K
        self.embed = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict({
                "msg": nn.Sequential(nn.Linear(hidden*2 + 1, hidden), nn.SiLU(), nn.Linear(hidden, hidden)),
                "upd": nn.Sequential(nn.Linear(hidden*2, hidden), nn.SiLU(), nn.Linear(hidden, hidden)),
                "norm": nn.LayerNorm(hidden),
            }))
        self.dec = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x, coord):
        # x: (B,N,F), coord: (B,N)
        h = self.embed(x)
        for blk in self.blocks:
            agg = torch.zeros_like(h)
            count = torch.zeros((1, h.shape[1], 1), device=h.device, dtype=h.dtype)

            for off in range(-self.K, self.K + 1):
                if off == 0:
                    continue
                if off > 0:
                    hi = h[:, :-off, :]
                    hj = h[:, off:, :]
                    dcoord = (coord[:, off:] - coord[:, :-off]).unsqueeze(-1)
                    m = blk["msg"](torch.cat([hi, hj, dcoord], dim=-1))
                    agg[:, :-off, :] += m
                    count[:, :-off, :] += 1.0
                else:
                    k = -off
                    hi = h[:, k:, :]
                    hj = h[:, :-k, :]
                    dcoord = (coord[:, :-k] - coord[:, k:]).unsqueeze(-1)
                    m = blk["msg"](torch.cat([hi, hj, dcoord], dim=-1))
                    agg[:, k:, :] += m
                    count[:, k:, :] += 1.0

            agg = agg / count.clamp_min(1.0)
            dh = blk["upd"](torch.cat([h, agg], dim=-1))
            h = blk["norm"](h + dh)

        return self.dec(h)

@torch.no_grad()
def eval_rmse(model, X, Y, coord=None, is_cnn=False):
    model.eval()
    device = next(model.parameters()).device
    X = X.to(device)
    Y = Y.to(device)
    if coord is not None:
        coord = coord.to(device)

    if is_cnn:
        predn = model(X)
    else:
        predn = model(X, coord) if coord is not None else model(X)
    rmse = torch.sqrt(((predn - Y) ** 2).mean()).item()
    rmse_toa = torch.sqrt(((predn[:, 0, :] - Y[:, 0, :]) ** 2).mean()).item()
    return rmse, rmse_toa

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot_n", type=int, default=5)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)              # (S,)
    logp = d["logp_arr"].astype(np.float32)        # (S,N)
    T = d["T_arr"].astype(np.float32)              # (S,N)
    q = d["q_arr"].astype(np.float32)              # (S,N)
    Fnet = d["Fnet_arr"].astype(np.float32)        # (S,N)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)

    # X: (S,N,4), Y: (S,N,1)
    X = np.stack([T, logp, q, Ts_b], axis=-1)
    Y = Fnet[..., None]

    # split
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(S)
    tr = perm[:int(0.8 * S)]
    va = perm[int(0.8 * S):]

    Xtr, Ytr, ctr = X[tr], Y[tr], logp[tr]
    Xva, Yva, cva = X[va], Y[va], logp[va]

    # normalize X and Y using train stats
    X_mu, X_std = zscore_fit(Xtr.reshape(-1, 4))
    Y_mu, Y_std = zscore_fit(Ytr.reshape(-1, 1))

    Xtrn = zscore_apply(Xtr, X_mu, X_std)
    Xvan = zscore_apply(Xva, X_mu, X_std)
    Ytrn = zscore_apply(Ytr, Y_mu, Y_std)
    Yvan = zscore_apply(Yva, Y_mu, Y_std)

    # torch tensors
    Xtr_t = torch.tensor(Xtrn, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytrn, dtype=torch.float32)
    Xva_t = torch.tensor(Xvan, dtype=torch.float32)
    Yva_t = torch.tensor(Yvan, dtype=torch.float32)
    ctr_t = torch.tensor(ctr, dtype=torch.float32)
    cva_t = torch.tensor(cva, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    loss_fn = nn.MSELoss()

    # --------------------
    # Baseline A
    A = LayerMLP().to(device)
    opt = torch.optim.Adam(A.parameters(), lr=2e-3)
    loader = DataLoader(TensorDataset(Xtr_t, Ytr_t), batch_size=args.batch, shuffle=True)
    for _ in range(args.epochs):
        A.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(A(xb), yb)
            loss.backward()
            opt.step()
    rmseA, rmseA_toa = eval_rmse(A, Xva_t, Yva_t)

    # --------------------
    # Baseline B (interp -> CNN)
    N_fixed = 128
    Xtr_fix, c_tr_fix = interp_to_fixed(Xtrn, ctr, N_fixed)
    Ytr_fix, _ = interp_to_fixed(Ytrn, ctr, N_fixed)
    Xva_fix, c_va_fix = interp_to_fixed(Xvan, cva, N_fixed)
    Yva_fix, _ = interp_to_fixed(Yvan, cva, N_fixed)

    Xtr_fix_t = torch.tensor(Xtr_fix, dtype=torch.float32)
    Ytr_fix_t = torch.tensor(Ytr_fix, dtype=torch.float32)
    Xva_fix_t = torch.tensor(Xva_fix, dtype=torch.float32)
    Yva_fix_t = torch.tensor(Yva_fix, dtype=torch.float32)

    B = CNN1D().to(device)
    opt = torch.optim.Adam(B.parameters(), lr=2e-3)
    loader_fix = DataLoader(TensorDataset(Xtr_fix_t, Ytr_fix_t), batch_size=args.batch, shuffle=True)
    for _ in range(args.epochs):
        B.train()
        for xb, yb in loader_fix:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(B(xb), yb)
            loss.backward()
            opt.step()
    rmseB, rmseB_toa = eval_rmse(B, Xva_fix_t, Yva_fix_t, is_cnn=True)

    # --------------------
    # Baseline C (Local GNO)
    C = LocalGNO(K=4, layers=2, hidden=96).to(device)
    opt = torch.optim.AdamW(C.parameters(), lr=2e-3, weight_decay=1e-4)
    loader_gno = DataLoader(TensorDataset(Xtr_t, Ytr_t, ctr_t), batch_size=max(args.batch, 64), shuffle=True)
    # 这里为了速度，训练轮数用 epochs//2（你可以调大）
    for _ in range(max(8, args.epochs // 2)):
        C.train()
        for xb, yb, cb in loader_gno:
            xb, yb, cb = xb.to(device), yb.to(device), cb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(C(xb, cb), yb)
            loss.backward()
            opt.step()
    rmseC, rmseC_toa = eval_rmse(C, Xva_t, Yva_t, coord=cva_t)

    # print results
    print("\nBaseline results (val, normalized target space):")
    print(f"A Layer-MLP:        RMSE(profile)={rmseA:.4f}  RMSE(TOA)={rmseA_toa:.4f}")
    print(f"B CNN (N=128):      RMSE(profile)={rmseB:.4f}  RMSE(TOA)={rmseB_toa:.4f}")
    print(f"C Local-GNO K=4 L2: RMSE(profile)={rmseC:.4f}  RMSE(TOA)={rmseC_toa:.4f}")

    # --------------------
    # quick plots (denormalize back to W/m^2)
    os.makedirs("step3_plots", exist_ok=True)
    idx = rng.choice(Xva_t.shape[0], size=min(args.plot_n, Xva_t.shape[0]), replace=False)

    @torch.no_grad()
    def denormY_t(y):
        y = y.cpu().numpy()
        y = zscore_invert(y, Y_mu, Y_std)
        return y.squeeze(-1)

    # A preds on original grid
    predA = denormY_t(A(Xva_t[idx].to(device)))
    trueA = denormY_t(Yva_t[idx])
    cA = cva[idx]

    # B preds on fixed grid
    predB = denormY_t(B(Xva_fix_t[idx].to(device)))
    trueB = denormY_t(Yva_fix_t[idx])
    cB = c_va_fix[idx]

    # C preds on original grid
    predC = denormY_t(C(Xva_t[idx].to(device), cva_t[idx].to(device)))
    trueC = denormY_t(Yva_t[idx])
    cC = cva[idx]

    def plot_set(coord, true, pred, title, fname):
        plt.figure()
        for i in range(true.shape[0]):
            plt.plot(coord[i], true[i], alpha=0.9)
            plt.plot(coord[i], pred[i], alpha=0.9, linestyle="--")
        plt.gca().invert_xaxis()
        plt.xlabel("log(p) [ln(Pa)] (inverted)")
        plt.ylabel("Fnet (W/m^2)")
        plt.title(title + " (solid=true, dashed=pred)")
        plt.savefig(os.path.join("step3_plots", fname), dpi=160, bbox_inches="tight")
        plt.close()

    plot_set(cA, trueA, predA, "Baseline A: Layer-wise MLP", "A_mlp.png")
    plot_set(cB, trueB, predB, "Baseline B: CNN on N=128 (interp grid)", "B_cnn.png")
    plot_set(cC, trueC, predC, "Baseline C: Local GNO", "C_gno.png")

    print("\nSaved plots to ./step3_plots/: A_mlp.png, B_cnn.png, C_gno.png")

if __name__ == "__main__":
    main()
