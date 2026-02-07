import numpy as np

def print_stats(name, data_path):
    print(f"--- {name} ({data_path}) ---")
    try:
        d = np.load(data_path, allow_pickle=True)
        keys = list(d.keys())
        # print("Keys:", keys)
        
        if "Ts_K" in d:
            Ts = d["Ts_K"]
            print(f"Ts: shape={Ts.shape}, min={Ts.min():.2f}, max={Ts.max():.2f}, mean={Ts.mean():.2f}, std={Ts.std():.2f}")
        
        if "Fnet_arr" in d:
            F = d["Fnet_arr"]
            F_boa = F[:, -1]
            print(f"F_boa: min={F_boa.min():.2f}, max={F_boa.max():.2f}, mean={F_boa.mean():.2f}, std={F_boa.std():.2f}")
            
    except Exception as e:
        print(f"Error reading {data_path}: {e}")

print_stats("Test 1k", "test_N160_1000.npz")
print_stats("Train 10k", "combined_10000_data.npz")
