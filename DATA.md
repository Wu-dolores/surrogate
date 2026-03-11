# Data Preparation Guide

This document explains how to prepare your atmospheric profile data for use with the surrogate model.

## Data Format

The model expects data in NumPy's compressed NPZ format with the following arrays:

### Required Fields

| Field | Shape | Type | Description | Units |
|-------|-------|------|-------------|-------|
| `logp_arr` | (S, N) | float32 | Log-pressure coordinate | log(Pa) |
| `T_arr` | (S, N) | float32 | Temperature profile | K |
| `q_arr` | (S, N) | float32 | Specific humidity profile | kg/kg |
| `Ts_K` | (S,) | float32 | Surface temperature | K |
| `Fnet_arr` | (S, N) | float32 | Net radiative flux profile | W/m² |

Where:
- **S** = Number of atmospheric samples/profiles
- **N** = Number of vertical levels (can vary between samples)

### Coordinate Convention

The model expects vertical coordinates to go from **TOA (top of atmosphere) to BOA (bottom/surface)**:
- `logp_arr[:, 0]` should be small (low pressure, high altitude)
- `logp_arr[:, -1]` should be large (high pressure, surface)

The data loader will automatically detect and correct reversed profiles.

## Creating Your Dataset

### Example: Converting from Raw Data

```python
import numpy as np

# Your raw atmospheric data
# Assume you have: pressure, temperature, humidity, surface_temp, net_flux

# 1. Compute log-pressure
logp = np.log(pressure)  # pressure in Pa

# 2. Ensure TOA -> BOA ordering
# If your data goes BOA -> TOA, reverse it:
if logp[0] > logp[-1]:
    logp = logp[::-1]
    temperature = temperature[::-1]
    humidity = humidity[::-1]
    net_flux = net_flux[::-1]

# 3. Stack multiple profiles
logp_arr = np.stack([logp_profile1, logp_profile2, ...])
T_arr = np.stack([temp_profile1, temp_profile2, ...])
q_arr = np.stack([humid_profile1, humid_profile2, ...])
Ts_K = np.array([surf_temp1, surf_temp2, ...])
Fnet_arr = np.stack([flux_profile1, flux_profile2, ...])

# 4. Save as NPZ
np.savez_compressed(
    'my_dataset.npz',
    logp_arr=logp_arr.astype(np.float32),
    T_arr=T_arr.astype(np.float32),
    q_arr=q_arr.astype(np.float32),
    Ts_K=Ts_K.astype(np.float32),
    Fnet_arr=Fnet_arr.astype(np.float32)
)
```

### Example: Variable Resolution Profiles

The model supports profiles with different vertical resolutions:

```python
# Profile 1: 40 levels
logp1 = np.linspace(2.0, 5.0, 40)
T1 = ...  # shape (40,)

# Profile 2: 160 levels
logp2 = np.linspace(2.0, 5.0, 160)
T2 = ...  # shape (160,)

# Save separately or interpolate to common grid
```

## Data Validation

Use the provided data loader to validate your dataset:

```python
from data import AtmosphericDataLoader

# Load and validate
loader = AtmosphericDataLoader('my_dataset.npz')
raw_data = loader.load_raw_data()

print(f"Loaded {raw_data['logp'].shape[0]} profiles")
print(f"Vertical levels: {raw_data['logp'].shape[1]}")
print(f"Temperature range: {raw_data['T'].min():.1f} - {raw_data['T'].max():.1f} K")
```

## Typical Data Ranges

For reference, typical atmospheric values:

| Variable | Typical Range | Notes |
|----------|---------------|-------|
| log(p) | 2.0 - 5.0 | log(Pa), ~7 hPa to ~150 hPa |
| Temperature | 180 - 320 K | Stratosphere to surface |
| Humidity | 0 - 0.03 kg/kg | Specific humidity |
| Surface Temp | 200 - 330 K | Polar to tropical |
| Net Flux | -200 - 400 W/m² | Radiative heating/cooling |

## Data Sources

Common sources for atmospheric profile data:

1. **Reanalysis Data**: ERA5, MERRA-2, JRA-55
2. **Climate Models**: CMIP6 output, regional models
3. **Radiative Transfer Models**: RRTMG, LBLRTM output
4. **Satellite Retrievals**: AIRS, IASI, CrIS

## Preprocessing Tips

### 1. Quality Control

```python
# Remove invalid profiles
valid = (
    (T_arr > 150) & (T_arr < 350) &  # Reasonable temperature
    (q_arr >= 0) & (q_arr < 0.1) &    # Valid humidity
    np.isfinite(Fnet_arr)              # No NaN/Inf
).all(axis=1)

T_arr = T_arr[valid]
q_arr = q_arr[valid]
# ... filter other arrays
```

### 2. Handling Missing Levels

```python
# Interpolate to fill gaps
from scipy.interpolate import interp1d

def fill_missing(profile, coord):
    valid = np.isfinite(profile)
    if valid.sum() < 2:
        return profile
    f = interp1d(coord[valid], profile[valid],
                 bounds_error=False, fill_value='extrapolate')
    return f(coord)
```

### 3. Normalization

The model handles normalization internally, but you can check data distribution:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.hist(T_arr.flatten(), bins=50)
plt.xlabel('Temperature (K)')

plt.subplot(132)
plt.hist(q_arr.flatten(), bins=50)
plt.xlabel('Humidity (kg/kg)')

plt.subplot(133)
plt.hist(Fnet_arr.flatten(), bins=50)
plt.xlabel('Net Flux (W/m²)')
plt.tight_layout()
plt.savefig('data_distribution.png')
```

## Example Datasets

The repository includes a pretrained model trained on 10,000 atmospheric profiles. To use your own data:

1. Prepare NPZ file following the format above
2. Use `run_finetune.py` to adapt the pretrained model
3. The model will automatically handle different resolutions

## Troubleshooting

### "Missing required fields in data"
Ensure your NPZ file contains all 5 required arrays with exact names.

### "Dataset too small"
You need at least 10 samples for training. For fine-tuning, 100+ samples recommended.

### "Coordinate ordering error"
Check that `logp_arr[:, 0] < logp_arr[:, -1]` (TOA to BOA).

### Memory issues with large datasets
Process data in chunks or reduce the number of samples:

```python
# Load subset
d = np.load('large_dataset.npz')
subset_idx = np.random.choice(len(d['logp_arr']), 5000, replace=False)
np.savez_compressed('subset.npz',
    logp_arr=d['logp_arr'][subset_idx],
    # ... other fields
)
```

## Contact

For questions about data preparation, please open an issue on GitHub.
