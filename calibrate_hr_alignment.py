import numpy as np

def diffop_np(F, coord):
    # F: (S,N), coord: (S,N)
    S, N = F.shape
    dF = np.zeros_like(F)
    # interior
    dF[:, 1:-1] = (F[:, 2:] - F[:, :-2]) / (coord[:, 2:] - coord[:, :-2] + 1e-12)
    # boundaries
    dF[:, 0] = (F[:, 1] - F[:, 0]) / (coord[:, 1] - coord[:, 0] + 1e-12)
    dF[:, -1] = (F[:, -1] - F[:, -2]) / (coord[:, -1] - coord[:, -2] + 1e-12)
    return dF

d = np.load("output_1000_data_final.npz", allow_pickle=True)
logp = d["logp_arr"].astype(np.float64)
Fnet = d["Fnet_arr"].astype(np.float64)
HR = d["HR_arr"].astype(np.float64)

hr_flux = diffop_np(Fnet, logp)

x = hr_flux.reshape(-1)
y = HR.reshape(-1)

# linear fit y ~ a x + b
A = np.vstack([x, np.ones_like(x)]).T
a, b = np.linalg.lstsq(A, y, rcond=None)[0]

# correlation
corr = np.corrcoef(x, y)[0,1]

print("Fit HR_true ~= a * dF/dlogp + b")
print("a =", a)
print("b =", b)
print("corr =", corr)

# also check corr after removing mean
x0 = x - x.mean()
y0 = y - y.mean()
corr0 = np.corrcoef(x0, y0)[0,1]
print("corr (demeaned) =", corr0)
