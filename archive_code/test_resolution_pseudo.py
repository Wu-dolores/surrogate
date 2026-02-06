import argparse
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

# ------------------ v1 model definition (same as step4_v1.py) ------------------
class LocalGNOBlock(nn.Module):
    def __init__(self, hidden=128, K=6):
        super().__init__()
        self.K = K
        self.msg = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.SiLU(),
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
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.blocks = nn.ModuleList([LocalGNOBlock(hidden=hidden, K=K) for _ in range(L)])
        self.dec = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x, coord):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h, coord)
        return self.dec(h)

# ------------------ helpers ------------------
def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

def ensure_increasing(coord, x):
    """Return coord,x reversed if coord is decreasing so that coord is increasing."""
    if coord[0] > coord[-1]:
        return coord[::-1].copy(), x[::-1].copy()
    return coord, x

def regrid_1d(x, coord, coord_new):
    """
    Safe 1D interpolation:
    - ensures coord increasing for np.interp
    - if coord_new decreasing, interpolate in increasing then flip back
    """
    coord, x = ensure_increasing(coord, x)

    if coord_new[0] > coord_new[-1]:
        coord_new_inc = coord_new[::-1].copy()
        y_inc = np.interp(coord_new_inc, coord, x).astype(np.float32)
        return y_inc[::-1].astype(np.float32)

    return np.interp(coord_new, coord, x).astype(np.float32)

def build_regridded_set(d, N_new, feat_list):
    """
    Build regridded dataset with feature list defined by checkpoint.
    Supported features: T, logp, c_tilde, q, Ts_broadcast
    """
    Ts = d["Ts_K"].astype(np.float32)         # (S,)
    logp = d["logp_arr"].astype(np.float32)   # (S,N)
    T = d["T_arr"].astype(np.float32)         # (S,N)
    q = d["q_arr"].astype(np.float32)         # (S,N)
    F = d["Fnet_arr"].astype(np.float32)      # (S,N)

    S, N = F.shape
    F_new = np.zeros((S, N_new), dtype=np.float32)
    c_new = np.zeros((S, N_new), dtype=np.float32)

    X_new = np.zeros((S, N_new, len(feat_list)), dtype=np.float32)

    for i in range(S):
        c = logp[i]
        cmin, cmax = float(c.min()), float(c.max())
        c_t = np.linspace(cmin, cmax, N_new, dtype=np.float32)
        c_new[i] = c_t

        # Regrid physical profiles onto c_t
        T_i = regrid_1d(T[i], c, c_t)
        q_i = regrid_1d(q[i], c, c_t)
        F_i = regrid_1d(F[i], c, c_t)
        F_new[i] = F_i

        # Build features according to checkpoint definition
        cols = []
        for feat in feat_list:
            if feat == "T":
                cols.append(T_i)
            elif feat == "logp":
                cols.append(c_t)
            elif feat == "c_tilde":
                # normalize to [0,1] on the NEW grid
                c_top = c_t[0]
                c_surf = c_t[-1]
                cols.append((c_t - c_top) / (c_surf - c_top + 1e-6))
            elif feat == "q":
                cols.append(q_i)
            elif feat in ("Ts_broadcast", "Ts"):
                cols.append(np.full((N_new,), Ts[i], dtype=np.float32))
            else:
                raise ValueError(f"Unsupported feature name in ckpt: {feat}")

        X_new[i] = np.stack(cols, axis=-1)

    return X_new, F_new, c_new

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt_v1", default="step4_v1_out/best_v1.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N_coarse", type=int, default=40)
    ap.add_argument("--N_fine", type=int, default=160)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    ck = torch.load(args.ckpt_v1, map_location="cpu")

    # Determine feature list from checkpoint
    feat_list = ck.get("features", ["T", "logp", "q", "Ts_broadcast"])
    if isinstance(feat_list, tuple):
        feat_list = list(feat_list)

    # Build val split indices (same rule as training scripts)
    S = d["Fnet_arr"].shape[0]
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(S)
    va = perm[int(0.8 * S):]

    # Regrid full set then slice val
    Xc, Fc, cc = build_regridded_set(d, args.N_coarse, feat_list)
    Xf, Ff, cf = build_regridded_set(d, args.N_fine, feat_list)
    Xc, Fc, cc = Xc[va], Fc[va], cc[va]
    Xf, Ff, cf = Xf[va], Ff[va], cf[va]

    # Normalization from checkpoint
    X_mu = np.array(ck["X_mu"], dtype=np.float32).reshape(1, 1, -1)
    X_std = np.array(ck["X_std"], dtype=np.float32).reshape(1, 1, -1)
    Y_mu = np.array(ck["Y_mu"], dtype=np.float32).reshape(1, 1, 1)
    Y_std = np.array(ck["Y_std"], dtype=np.float32).reshape(1, 1, 1)

    def normX(x): return (x - X_mu) / X_std
    def denormY(y): return y * Y_std + Y_mu

    # Load model config
    cfg = ck.get("cfg", {})
    hidden = int(cfg.get("hidden", 128))
    K = int(cfg.get("K", 6))
    L = int(cfg.get("L", 4))

    model = V1LocalGNO(in_dim=len(feat_list), hidden=hidden, K=K, L=L)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    @torch.no_grad()
    def eval_on(X, F_true, coord):
        Xn = torch.tensor(normX(X), dtype=torch.float32)
        c = torch.tensor(coord, dtype=torch.float32)  # raw logp grid
        predn = model(Xn, c).numpy()                  # (V,N,1)
        pred = denormY(predn).squeeze(-1)             # (V,N)
        prof = rmse(pred, F_true)
        toa = rmse(pred[:, 0], F_true[:, 0])
        boa = rmse(pred[:, -1], F_true[:, -1])
        return prof, toa, boa

    prof_c, toa_c, boa_c = eval_on(Xc, Fc, cc)
    prof_f, toa_f, boa_f = eval_on(Xf, Ff, cf)

    print("=== PSEUDO resolution test (regridded truth) ===")
    print(f"Coarse N={args.N_coarse}: prof={prof_c:.3f} TOA={toa_c:.3f} BOA={boa_c:.3f} (W/m^2)")
    print(f"Fine   N={args.N_fine}:   prof={prof_f:.3f} TOA={toa_f:.3f} BOA={boa_f:.3f} (W/m^2)")
    print(f"Features used: {feat_list}")

if __name__ == "__main__":
    main()
