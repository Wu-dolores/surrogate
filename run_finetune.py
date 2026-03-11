"""
Automated fine-tuning pipeline for surrogate model transfer learning.

This script automates the process of:
1. Splitting target domain data into train/test
2. Fine-tuning pretrained model on training split
3. Evaluating on held-out test split
4. Generating performance reports
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np


def run_command(cmd: str, description: str = "") -> None:
    """
    Execute shell command with proper error handling.

    Args:
        cmd: Command string to execute
        description: Human-readable description of the command

    Raises:
        SystemExit: If command fails
    """
    if description:
        print(f"\n{'='*60}")
        print(f"{description}")
        print(f"{'='*60}")

    print(f"Running: {cmd}\n")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"\n{'!'*60}")
        print(f"ERROR: Command failed with exit code {e.returncode}")
        print(f"{'!'*60}")
        if e.stderr:
            print(f"\nError output:\n{e.stderr}")
        if e.stdout:
            print(f"\nStandard output:\n{e.stdout}")
        sys.exit(e.returncode)

    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"ERROR: Unexpected error executing command")
        print(f"{'!'*60}")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


def validate_file_exists(path: str, description: str) -> Path:
    """
    Validate that a file exists.

    Args:
        path: File path to check
        description: Description for error message

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return p


def split_data(
    input_path: Path,
    train_path: Path,
    test_path: Path,
    train_ratio: float = 0.8,
    seed: int = 42
) -> None:
    """
    Split dataset into training and testing subsets.

    Args:
        input_path: Path to input NPZ file
        train_path: Path to save training split
        test_path: Path to save testing split
        train_ratio: Fraction of data for training (default: 0.8)
        seed: Random seed for reproducibility

    Raises:
        ValueError: If data format is invalid
    """
    print(f"\n{'='*60}")
    print(f"Splitting Data: {input_path.name}")
    print(f"{'='*60}")

    np.random.seed(seed)

    try:
        d = np.load(str(input_path), allow_pickle=True)
    except Exception as e:
        raise ValueError(f"Failed to load data file: {e}")

    # Validate required fields
    required_fields = ["logp_arr", "T_arr", "q_arr", "Fnet_arr", "Ts_K"]
    missing = [f for f in required_fields if f not in d]
    if missing:
        raise ValueError(f"Missing required fields in data: {missing}")

    logp = d["logp_arr"]
    S = len(logp)

    if S < 10:
        raise ValueError(f"Dataset too small: only {S} samples")

    perm = np.random.permutation(S)
    split_idx = int(S * train_ratio)
    idx_tr = perm[:split_idx]
    idx_te = perm[split_idx:]

    print(f"Total samples: {S}")
    print(f"Training: {len(idx_tr)} ({len(idx_tr)/S*100:.1f}%)")
    print(f"Testing: {len(idx_te)} ({len(idx_te)/S*100:.1f}%)")

    # Save training split
    np.savez_compressed(
        str(train_path),
        logp_arr=d["logp_arr"][idx_tr],
        T_arr=d["T_arr"][idx_tr],
        q_arr=d["q_arr"][idx_tr],
        Fnet_arr=d["Fnet_arr"][idx_tr],
        Ts_K=d["Ts_K"][idx_tr]
    )
    print(f"✓ Training split saved: {train_path}")

    # Save testing split
    np.savez_compressed(
        str(test_path),
        logp_arr=d["logp_arr"][idx_te],
        T_arr=d["T_arr"][idx_te],
        q_arr=d["q_arr"][idx_te],
        Fnet_arr=d["Fnet_arr"][idx_te],
        Ts_K=d["Ts_K"][idx_te]
    )
    print(f"✓ Testing split saved: {test_path}")


def main() -> None:
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Automated Fine-Tuning Pipeline for Surrogate Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        required=True,
        help="Path to pretrained checkpoint (.pt file)"
    )
    parser.add_argument(
        "--target_data",
        type=str,
        required=True,
        help="Path to target domain dataset (.npz file)"
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default="finetune_job",
        help="Name for output folders and temporary files"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of fine-tuning epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Fraction of data for training (rest for testing)"
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep temporary train/test split files"
    )

    args = parser.parse_args()

    # Validate inputs
    try:
        pretrained_path = validate_file_exists(
            args.pretrained_ckpt,
            "Pretrained checkpoint"
        )
        target_data_path = validate_file_exists(
            args.target_data,
            "Target dataset"
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Define paths
    temp_train = Path(f"temp_{args.job_name}_train.npz")
    temp_test = Path(f"temp_{args.job_name}_test.npz")
    out_dir = Path(f"output_{args.job_name}")
    eval_out_dir = out_dir / "eval_results"

    print("\n" + "="*60)
    print("FINE-TUNING PIPELINE")
    print("="*60)
    print(f"Pretrained model: {pretrained_path}")
    print(f"Target data: {target_data_path}")
    print(f"Output directory: {out_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    try:
        # Step 1: Split data
        split_data(
            target_data_path,
            temp_train,
            temp_test,
            train_ratio=args.train_ratio
        )

        # Step 2: Fine-tune model
        train_cmd = (
            f"python model_train.py "
            f"--data {temp_train} "
            f"--out {out_dir} "
            f"--ckpt {pretrained_path} "
            f"--epochs {args.epochs} "
            f"--batch 128 "
            f"--lr {args.lr} "
            f"--Ts_tail 320 "
            f"--tail_mult 2.0"
        )
        run_command(train_cmd, "STEP 2: Fine-Tuning Model")

        # Step 3: Evaluate on test split
        best_model = out_dir / "best_hr_toa_boa.pt"
        if not best_model.exists():
            raise FileNotFoundError(f"Trained model not found: {best_model}")

        eval_cmd = (
            f"python model_eval.py "
            f"--ckpt {best_model} "
            f"--data {temp_test} "
            f"--out {eval_out_dir} "
            f"--bot_window_k 0"
        )
        run_command(eval_cmd, "STEP 3: Evaluating on Test Split")

        # Step 4: Cleanup temporary files
        if not args.keep_temp:
            if temp_train.exists():
                temp_train.unlink()
                print(f"✓ Removed temporary file: {temp_train}")
            if temp_test.exists():
                temp_test.unlink()
                print(f"✓ Removed temporary file: {temp_test}")

        # Success message
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Results saved in: {eval_out_dir}")
        print(f"Best model: {best_model}")

    except Exception as e:
        print(f"\n{'!'*60}")
        print(f"PIPELINE FAILED")
        print(f"{'!'*60}")
        print(f"{type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
