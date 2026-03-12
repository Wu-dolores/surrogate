# Atmospheric Radiation Surrogate Model

A deep learning-based surrogate model for fast atmospheric radiative transfer prediction using LocalGNO (Local Graph Neural Operator) architecture.

## Features

- **Multi-task Learning**: Simultaneously predicts heating rates (HR), top-of-atmosphere (TOA) flux, and bottom-of-atmosphere (BOA) flux
- **LocalGNO Architecture**: Graph neural operator with local message passing for vertical profile processing
- **Transfer Learning**: Fine-tuning pipeline for adapting to new atmospheric conditions
- **Modular Design**: Clean separation of models, utilities, data processing, and configuration

## Project Structure

```
surrogate/
├── models.py              # Neural network architectures
├── utils.py               # Utility functions (normalization, integration, etc.)
├── data.py                # Data loading and preprocessing
├── config.py              # Configuration dataclasses
├── run_finetune.py        # Automated fine-tuning pipeline
├── test_utils.py          # Unit tests
├── requirements.txt       # Python dependencies
├── pretrained_ckpt/       # Pretrained model checkpoints
└── .gitignore             # Git ignore rules
```

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install manually
pip install numpy torch matplotlib scipy pytest
```

## Quick Start

### 1. Fine-Tuning (Transfer Learning)

Use the automated pipeline for adapting to new data:

```bash
python run_finetune.py \
  --pretrained_ckpt pretrained_ckpt/base_model_10k.pt \
  --target_data your_data.npz \
  --job_name high_res_adaptation \
  --epochs 50 \
  --lr 1e-4
```

**Pipeline Steps:**
1. Splits target data (80% train / 20% test)
2. Fine-tunes model on training split
3. Evaluates on held-out test split
4. Generates performance reports

### 2. Testing

```bash
# Run unit tests
pytest test_utils.py -v

# Test with your own data
python run_finetune.py --help
```

## Model Architecture

### LocalGNO Block

Processes vertical atmospheric profiles using local message passing:
- **Neighborhood size K**: Exchanges information between levels within ±K distance
- **Message function**: Combines node features and coordinate distances
- **Update function**: Aggregates messages and updates node representations

### Multi-Task Heads

1. **HR Head**: Predicts heating rate at each vertical level
2. **TOA Head**: Predicts net flux at top of atmosphere (global + local context)
3. **BOA Head**: Predicts surface flux with surface temperature skip connection

## Data Format

Input NPZ files should contain:
- `logp_arr`: Log-pressure coordinate (S, N)
- `T_arr`: Temperature profile (S, N) [K]
- `q_arr`: Specific humidity profile (S, N) [kg/kg]
- `Ts_K`: Surface temperature (S,) [K]
- `Fnet_arr`: Net radiative flux profile (S, N) [W/m²]

Where S = number of samples, N = number of vertical levels.

## Configuration

Model and training parameters can be customized via command-line arguments or by modifying `config.py`:

```python
from config import ModelConfig, TrainingConfig

model_cfg = ModelConfig(
    hidden=128,      # Hidden dimension
    K=6,             # LocalGNO neighborhood size
    L=4              # Number of LocalGNO blocks
)

train_cfg = TrainingConfig(
    epochs=100,
    batch_size=1024,
    lr=1e-3,
    loss_weights=[1.0, 1.0, 1.0, 0.0]  # [HR, TOA, BOA, Physics]
)
```

## Advanced Features

### Custom Configuration

You can customize training parameters:

```bash
python run_finetune.py \
  --pretrained_ckpt pretrained_ckpt/base_model_10k.pt \
  --target_data your_data.npz \
  --epochs 100 \
  --lr 1e-4 \
  --train_ratio 0.8
```

## Performance Metrics

The model reports:
- **RMSE Profile**: Root mean square error across all vertical levels
- **RMSE TOA**: Error at top of atmosphere
- **RMSE BOA**: Error at surface

Typical performance on test data:
- Profile RMSE: ~5-10 W/m²
- TOA RMSE: ~3-5 W/m²
- BOA RMSE: ~5-8 W/m²

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python model_train.py --batch 512
```

### Slow Training

Enable multi-threading:
```python
torch.set_num_threads(8)  # In model_train.py
```

### Poor Convergence

Try adjusting learning rate or loss weights:
```bash
python model_train.py --lr 5e-4 --loss_weights "1,2,2,0"
```

## Citation

If you use this code, please cite:

```bibtex
@software{atmospheric_surrogate_2026,
  title={Atmospheric Radiation Surrogate Model},
  author={Your Name},
  year={2026},
  url={https://github.com/Wu-dolores/surrogate}
}
```

## License

MIT License - see LICENSE file for details.

## Contact

For questions or issues, please open a GitHub issue or contact dolores@stu.pku.edu.cn]
