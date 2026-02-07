import numpy as np

def check_leakage():
    print("Loading datasets...")
    # Train data
    d_train = np.load("combined_10000_data.npz", allow_pickle=True)
    Ts_train = d_train["Ts_K"].astype(np.float32)
    Fnet_train = d_train["Fnet_arr"].astype(np.float32)
    
    # Test data
    d_test = np.load("test_N160_1000.npz", allow_pickle=True)
    Ts_test = d_test["Ts_K"].astype(np.float32)
    Fnet_test = d_test["Fnet_arr"].astype(np.float32)
    
    # 1. Check duplicates within Train
    print("\n--- 1. Internal Duplicates in Train (10k) ---")
    # Use Ts as a simple hash (fast check)
    unique_Ts, counts = np.unique(Ts_train, return_counts=True)
    duplicates = np.sum(counts > 1)
    print(f"Total samples: {len(Ts_train)}")
    print(f"Unique Ts values: {len(unique_Ts)}")
    print(f"Number of Ts values appearing >1 times: {duplicates}")
    if duplicates > 0:
        print(f"Max repetition of a single Ts value: {counts.max()}")
        
    # Check exact duplicates of (Ts, Fnet_boa) pair
    # Construct a proxy feature vector: [Ts, F_boa]
    proxy_train = np.stack([Ts_train, Fnet_train[:, -1]], axis=1)
    unique_rows, u_counts = np.unique(proxy_train, axis=0, return_counts=True)
    print(f"Unique (Ts, F_boa) pairs: {len(unique_rows)} / {len(Ts_train)}")
    
    # 2. Check overlap between Train and Test
    print("\n--- 2. Overlap between Train and Test ---")
    # We check if test samples appear in train
    # This is rough; floating point equality is tricky. We check exact match.
    
    intersection_count = 0
    # Convert to bytes for set operations
    train_set = set([row.tobytes() for row in proxy_train])
    
    proxy_test = np.stack([Ts_test, Fnet_test[:, -1]], axis=1)
    for row in proxy_test:
        if row.tobytes() in train_set:
            intersection_count += 1
            
    print(f"Test samples found EXACTLY in Train: {intersection_count} / {len(Ts_test)}")

if __name__ == "__main__":
    check_leakage()
