import numpy as np

def regrid_interp(x, logp_old, logp_new):
    # x: (N,) or (N, C)
    # returns (M,) or (M, C)
    if x.ndim == 1:
        return np.interp(logp_new, logp_old, x).astype(np.float32)
    else:
        C = x.shape[1]
        out = np.zeros((len(logp_new), C), dtype=np.float32)
        for c in range(C):
            out[:, c] = np.interp(logp_new, logp_old, x[:, c])
        return out

def process_dataset(path, name, target_N=160, limit=None):
    print(f"Loading {name} from {path}...")
    d = np.load(path, allow_pickle=True)
    
    # Extract arrays
    logp = d["logp_arr"] # (S, N)
    T    = d["T_arr"]    # (S, N)
    q    = d["q_arr"]    # (S, N)
    Fnet = d["Fnet_arr"] # (S, N)
    Ts   = d["Ts_K"]     # (S,)
    
    if limit is not None:
        logp = logp[:limit]
        T = T[:limit]
        q = q[:limit]
        Fnet = Fnet[:limit]
        Ts = Ts[:limit]
        print(f"  -> Took first {limit} samples.")
    
    S, N = logp.shape
    print(f"  Result shape: {S} samples, Orig N={N}")
    
    if N == target_N:
        return logp, T, q, Fnet, Ts
    
    # Needs Regridding
    print(f"  -> Regridding to N={target_N}...")
    
    new_logp = np.zeros((S, target_N), dtype=np.float32)
    new_T    = np.zeros((S, target_N), dtype=np.float32)
    new_q    = np.zeros((S, target_N), dtype=np.float32)
    new_Fnet = np.zeros((S, target_N), dtype=np.float32)
    
    for i in range(S):
        # Create target grid: linear in logp space from top to bot
        top, bot = logp[i, 0], logp[i, -1]
        grid = np.linspace(top, bot, target_N).astype(np.float32)
        new_logp[i] = grid
        
        # Interpolate fields
        # Note: Fnet is also interpolated
        new_T[i]    = regrid_interp(T[i], logp[i], grid)
        new_q[i]    = regrid_interp(q[i], logp[i], grid)
        new_Fnet[i] = regrid_interp(Fnet[i], logp[i], grid)
        
    return new_logp, new_T, new_q, new_Fnet, Ts

def main():
    TARGET_N = 160
    
    # 1. Base 10k data (N=60) -> Upsample
    lp1, t1, q1, f1, ts1 = process_dataset("combined_10000_data.npz", "Train 10k")
    
    # 2. Test 160 (N=160) -> Take 500
    lp2, t2, q2, f2, ts2 = process_dataset("test_N160_1000.npz", "Test 1k (N160)", limit=500)
    
    # 3. Test 40 (N=40) -> Upsample, Take 500
    lp3, t3, q3, f3, ts3 = process_dataset("test_N40_1000.npz", "Test 1k (N40)", limit=500)
    
    # Concatenate
    print("\nConcatenating...")
    logp_all = np.concatenate([lp1, lp2, lp3], axis=0).astype(np.float32)
    T_all    = np.concatenate([t1, t2, t3], axis=0).astype(np.float32)
    q_all    = np.concatenate([q1, q2, q3], axis=0).astype(np.float32)
    Fnet_all = np.concatenate([f1, f2, f3], axis=0).astype(np.float32)
    Ts_all   = np.concatenate([ts1, ts2, ts3], axis=0).astype(np.float32)
    
    print(f"Total Combined Shape: {logp_all.shape}")
    
    out_name = "mixed_train_11k.npz"
    print(f"Saving to {out_name}...")
    np.savez_compressed(
        out_name,
        logp_arr=logp_all,
        T_arr=T_all,
        q_arr=q_all,
        Fnet_arr=Fnet_all,
        Ts_K=Ts_all
    )
    print("Done.")

if __name__ == "__main__":
    main()
