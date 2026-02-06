import argparse, os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

torch.set_num_threads(1)

def zfit(x):
    mu = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True) + 1e-6
    return mu, std

def zapply(x, mu, std):
    return (x - mu) / std

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

class RelativeBias(nn.Module):
    """MLP that maps Δcoord -> scalar bias per head."""
    def __init__(self, n_heads=4, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_heads),
        )
    def forward(self, dcoord):  # (B,N,N,1)
        return self.net(dcoord) # (B,N,N,H)

class GlobalAttnBlock(nn.Module):
    def __init__(self, hidden=128, n_heads=4, bias_hidden=64, attn_drop=0.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.hidden = hidden
        self.n_heads = n_heads
        self.d_head = hidden // n_heads

        self.qkv = nn.Linear(hidden, hidden*3)
        self.proj = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.bias = RelativeBias(n_heads=n_heads, hidden=bias_hidden)
        self.drop = nn.Dropout(attn_drop)

    def forward(self, h, coord, mask=None):
        # h: (B,N,H), coord: (B,N)
        B, N, H = h.shape
        x = self.norm(h)

        qkv = self.qkv(x)  # (B,N,3H)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        # (B, heads, N, d_head)
        q = q.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.d_head).transpose(1, 2)

        # scaled dot-product
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B,heads,N,N)

        # relative coord bias
        # dcoord: (B,N,N,1) where dcoord_ij = coord_j - coord_i
        dcoord = (coord[:, None, :] - coord[:, :, None]).unsqueeze(-1)
        bias = self.bias(dcoord)  # (B,N,N,heads)
        bias = bias.permute(0, 3, 1, 2)  # (B,heads,N,N)
        attn = attn + bias

        # mask (optional for padding future)
        if mask is not None:
            # mask: (B,N) True=valid
            m = mask[:, None, None, :].to(dtype=attn.dtype)  # (B,1,1,N)
            attn = attn.masked_fill(m == 0, float("-inf"))

        w = torch.softmax(attn, dim=-1)
        w = self.drop(w)

        out = torch.matmul(w, v)  # (B,heads,N,d_head)
        out = out.transpose(1, 2).contiguous().view(B, N, H)
        out = self.proj(out)
        return h + out  # residual

class V2Model(nn.Module):
    def __init__(self, in_dim=4, hidden=128, K=6, L=4, n_heads=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.local = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])
        self.attn = GlobalAttnBlock(hidden=hidden, n_heads=n_heads, bias_hidden=64)
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord, mask=None):
        h = self.embed(x)
        for blk in self.local:
            h = blk(h, coord)
        h = self.attn(h, coord, mask=mask)
        return self.dec(h)

@torch.no_grad()
def eval_phys(model, Xva_t, Yva_t, cva_t, Y_mu, Y_std):
    model.eval()
    predn = model(Xva_t, cva_t).cpu().numpy()
    truen = Yva_t.cpu().numpy()
    pred = zinvert(predn, Y_mu, Y_std).squeeze(-1)
    true = zinvert(truen, Y_mu, Y_std).squeeze(-1)
    rmse_prof = float(np.sqrt(np.mean((pred - true)**2)))
    rmse_toa  = float(np.sqrt(np.mean((pred[:, 0] - true[:, 0])**2)))
    rmse_boa  = float(np.sqrt(np.mean((pred[:, -1] - true[:, -1])**2)))
    return rmse_prof, rmse_toa, rmse_boa

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
    ap.add_argument("--heads", type=int, default=4)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    d = np.load(args.data, allow_pickle=True)
    Ts = d["Ts_K"].astype(np.float32)
    logp = d["logp_arr"].astype(np.float32)
    T = d["T_arr"].astype(np.float32)
    q = d["q_arr"].astype(np.float32)
    Fnet = d["Fnet_arr"].astype(np.float32)

    S, N = Fnet.shape
    Ts_b = np.repeat(Ts[:, None], N, axis=1).astype(np.float32)
    X = np.stack([T, logp, q, Ts_b], axis=-1)
    Y = Fnet[..., None]

    perm = rng.permutation(S)
    tr = perm[:int(0.8*S)]
    va = perm[int(0.8*S):]

    Xtr, Ytr, ctr = X[tr], Y[tr], logp[tr]
    Xva, Yva, cva = X[va], Y[va], logp[va]

    X_mu, X_std = zfit(Xtr.reshape(-1, 4))
    Y_mu, Y_std = zfit(Ytr.reshape(-1, 1))

    Xtrn = zapply(Xtr, X_mu, X_std)
    Xvan = zapply(Xva, X_mu, X_std)
    Ytrn = zapply(Ytr, Y_mu, Y_std)
    Yvan = zapply(Yva, Y_mu, Y_std)

    Xtr_t = torch.tensor(Xtrn, dtype=torch.float32)
    Ytr_t = torch.tensor(Ytrn, dtype=torch.float32)
    Xva_t = torch.tensor(Xvan, dtype=torch.float32)
    Yva_t = torch.tensor(Yvan, dtype=torch.float32)
    ctr_t = torch.tensor(ctr, dtype=torch.float32)  # raw coord
    cva_t = torch.tensor(cva, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    Xtr_t, Ytr_t, ctr_t = Xtr_t.to(device), Ytr_t.to(device), ctr_t.to(device)
    Xva_t, Yva_t, cva_t = Xva_t.to(device), Yva_t.to(device), cva_t.to(device)

    model = V2Model(in_dim=4, hidden=args.hidden, K=args.K, L=args.L, n_heads=args.heads).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=True)

    outdir = "step5_v2_out"
    os.makedirs(outdir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(outdir, "best_v2.pt")

    train_loader = DataLoader(TensorDataset(Xtr_t, Ytr_t, ctr_t), batch_size=args.batch, shuffle=True)

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb, cb in train_loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb, cb)
            m = 5
            lam_bc = 2.0
            lam_reg = 1e-4

            Lf = loss_fn(pred, yb)
            Lbc = loss_fn(pred[:, -m:, :], yb[:, -m:, :])
            smooth = ((pred[:, 2:, :] - 2*pred[:, 1:-1, :] + pred[:, :-2, :])**2).mean()

            loss = Lf + lam_bc * Lbc + lam_reg * smooth

            loss.backward()
            opt.step()

        rmse_prof, rmse_toa, rmse_boa = eval_phys(model, Xva_t, Yva_t, cva_t, Y_mu, Y_std)
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
            }, best_path)

        if ep % 10 == 0 or ep == 1:
            lr = opt.param_groups[0]["lr"]
            print(f"Epoch {ep:04d} | lr={lr:.2e} | val RMSE prof={rmse_prof:.3f} W/m^2 | TOA={rmse_toa:.3f} | BOA={rmse_boa:.3f}")

    print(f"\nSaved best checkpoint: {best_path}  (best val prof RMSE={best:.3f} W/m^2)")

if __name__ == "__main__":
    main()
