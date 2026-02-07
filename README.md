# Surrogate Model Fine-Tuning Pipeline

## Project Structure

*   **`model_train.py`**: The main training script. Supports training from scratch and fine-tuning from a checkpoint (`--ckpt`). Implements the LocalGNO architecture with Skip Connections.
*   **`model_eval.py`**: The evaluation script for calculating RMSE (Profile/TOA/BOA).
*   **`run_finetune.py`**: An automated pipeline script for Few-Shot Transfer Learning.
*   **`pretrained_ckpt/`**: Stores the base model (trained on 10k samples).
*   **`archive_code/`**: Archived legacy scripts.
*   **`archive_logs/`**: Logs from previous runs.

## Automated Fine-Tuning Workflow

To adapt the model to a new dataset (e.g., higher resolution `test_N160_1000.npz`), use the automation script:

```bash
python run_finetune.py \
  --pretrained_ckpt pretrained_ckpt/base_model_10k.pt \
  --target_data test_N160_1000.npz \
  --job_name adaptability_test \
  --epochs 50 \
  --lr 1e-4
```

**Pipeline Steps:**
1.  **Split**: Randomly splits target data into Train (80%) and Test (20%).
2.  **Fine-tune**: Trains the model on the 80% split (freezing normalization stats).
3.  **Evaluate**: Tests performance on the 20% unseen split.
4.  **Report**: Saves results and logs to `output_<job_name>/`.

## Manual Usage

**Training/Fine-Tuning:**
```bash
python model_train.py \
  --data <data.npz> \
  --out <out_dir> \
  --ckpt <pretrained_ckpt.pt> \
  --epochs 50 \
  --batch 128 \
  --lr 1e-4
```

**Evaluation:**
```bash
python model_eval.py \
  --ckpt <model.pt> \
  --data <test.npz> \
  --out <eval_dir> \
  --bot_window_k 0
```
