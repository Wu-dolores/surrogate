import numpy as np

def main():
    np.random.seed(42) # Ensure reproducibility
    
    print("Loading test_N160_1000.npz...")
    d = np.load("test_N160_1000.npz", allow_pickle=True)
    
    logp = d["logp_arr"]
    T    = d["T_arr"]
    q    = d["q_arr"]
    Fnet = d["Fnet_arr"]
    Ts   = d["Ts_K"]
    
    S = len(logp)
    indices = np.random.permutation(S)
    
    # Split 80% Train (800), 20% Test (200)
    split = int(0.8 * S)
    idx_tr = indices[:split]
    idx_te = indices[split:]
    
    print(f"Total samples: {S}")
    print(f"Train samples: {len(idx_tr)}")
    print(f"Test samples:  {len(idx_te)}")
    
    # Save Train
    print("Saving ft_train_800.npz...")
    np.savez_compressed(
        "ft_train_800.npz",
        logp_arr=logp[idx_tr],
        T_arr=T[idx_tr],
        q_arr=q[idx_tr],
        Fnet_arr=Fnet[idx_tr],
        Ts_K=Ts[idx_tr]
    )
    
    # Save Test
    print("Saving ft_test_200.npz...")
    np.savez_compressed(
        "ft_test_200.npz",
        logp_arr=logp[idx_te],
        T_arr=T[idx_te],
        q_arr=q[idx_te],
        Fnet_arr=Fnet[idx_te],
        Ts_K=Ts[idx_te]
    )
    
    # Stats Check
    print("\n--- Stats Check ---")
    ts_tr = Ts[idx_tr]
    ts_te = Ts[idx_te]
    print(f"Train Ts Mean: {ts_tr.mean():.2f}")
    print(f"Test  Ts Mean: {ts_te.mean():.2f}")

if __name__ == "__main__":
    main()
