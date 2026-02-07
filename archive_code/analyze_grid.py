import numpy as np
import matplotlib.pyplot as plt

def analyze(path):
    print(f"--- Analyzing {path} ---")
    try:
        d = np.load(path, allow_pickle=True)
        logp = d["logp_arr"]
        print(f"Shape: {logp.shape}")
        
        # Check linearity in log-p
        # We look at the first sample
        p = logp[0]
        N = len(p)
        x = np.linspace(0, 1, N)
        slope = p[-1] - p[0]
        linear_ref = p[0] + slope * x
        
        diff = np.abs(p - linear_ref)
        max_dev = np.max(diff)
        mean_dev = np.mean(diff)
        
        print(f"Log-p linearity check (Sample 0): Max dev from linear={max_dev:.4f}, Mean dev={mean_dev:.4f}")
        
        # Check dcoord statistics
        dp = np.diff(p)
        print(f"dp (layer thickness): min={dp.min():.4f}, max={dp.max():.4f}, mean={dp.mean():.4f}, std={dp.std():.4f}")
        
    except Exception as e:
        print(f"Error: {e}")

analyze("combined_10000_data.npz")
analyze("test_N160_1000.npz")
analyze("test_N40_1000.npz")
