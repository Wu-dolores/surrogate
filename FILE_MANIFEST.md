# File Manifest

Complete list of files in the repository and their purposes.

## Core Python Modules

| File | Size | Purpose |
|------|------|---------|
| `models.py` | 5.4 KB | Neural network architectures (LocalGNO, HR_TOA_BOA_Model) |
| `utils.py` | 7.1 KB | Utility functions (normalization, integration, interpolation) |
| `data.py` | 6.0 KB | Data loading and preprocessing (AtmosphericDataLoader) |
| `config.py` | 1.6 KB | Configuration dataclasses (ModelConfig, TrainingConfig, etc.) |

## Scripts

| File | Size | Purpose |
|------|------|---------|
| `run_finetune.py` | 8.0 KB | Automated fine-tuning pipeline with error handling |
| `quickstart.sh` | 1.5 KB | Quick start script for new users |

## Tests

| File | Size | Purpose |
|------|------|---------|
| `test_utils.py` | 3.8 KB | Unit tests for utility functions (pytest) |

## Documentation

| File | Size | Purpose |
|------|------|---------|
| `README.md` | 4.8 KB | Main project documentation and usage guide |
| `DATA.md` | 5.7 KB | Data preparation and format guide |
| `CONTRIBUTING.md` | 6.0 KB | Contribution guidelines for developers |
| `CLEANUP.md` | 4.6 KB | Summary of cleanup process |
| `FILE_MANIFEST.md` | This file | Complete file listing |

## Configuration

| File | Size | Purpose |
|------|------|---------|
| `requirements.txt` | 315 B | Python package dependencies |
| `setup.py` | 2.0 KB | Package installation configuration |
| `.gitignore` | 600 B | Git ignore rules (data files, caches, etc.) |
| `LICENSE` | 1.1 KB | MIT License |

## Pretrained Models

| File | Size | Purpose |
|------|------|---------|
| `pretrained_ckpt/base_model_10k.pt` | 1.9 MB | Pretrained model on 10k atmospheric profiles |

## Total Repository Size

- **Code**: ~35 KB (Python modules and scripts)
- **Documentation**: ~27 KB (Markdown files)
- **Configuration**: ~4 KB (requirements, setup, gitignore)
- **Pretrained Model**: 1.9 MB
- **Total (excluding .git)**: ~2.0 MB

## File Purposes Summary

### For Users
- `README.md` - Start here for project overview
- `DATA.md` - Learn how to prepare your data
- `quickstart.sh` - Quick setup and test
- `run_finetune.py` - Main script to use

### For Developers
- `models.py`, `utils.py`, `data.py`, `config.py` - Core modules to understand/modify
- `test_utils.py` - Tests to run and extend
- `CONTRIBUTING.md` - Guidelines for contributing
- `setup.py` - Package installation

### For Deployment
- `requirements.txt` - Install dependencies
- `.gitignore` - Exclude unnecessary files
- `LICENSE` - Legal terms (MIT)

## Not Included (Excluded by .gitignore)

The following file types are excluded from the repository:
- `*.npz` - Data files (too large for Git)
- `*.pt`, `*.pth` - Model checkpoints (except pretrained_ckpt/)
- `__pycache__/` - Python cache
- `output_*/` - Training outputs
- `*.png`, `*.jpg` - Generated plots

Users should prepare their own data files or download them separately.

---

Last updated: 2024-03-11
