import numpy as np

d = np.load("test_N160_1000.npz", allow_pickle=True)
F = d["Fnet_arr"][:, -1] # BOA
Ts = d["Ts_K"]

half = len(F) // 2
f1, f2 = F[:half], F[half:]
t1, t2 = Ts[:half], Ts[half:]

print(f"--- Data Split Check ---")
print(f"First 500: Mean F_boa={f1.mean():.2f}, Mean Ts={t1.mean():.2f}")
print(f"Last  500: Mean F_boa={f2.mean():.2f}, Mean Ts={t2.mean():.2f}")
print(f"Diff F_boa: {abs(f1.mean()-f2.mean()):.2f}")
