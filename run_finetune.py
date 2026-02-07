import os
import argparse
import numpy as np
import subprocess
import shutil
import sys

def run_command(cmd):
    print(f"Running: {cmd}")
    ret = os.system(cmd)
    if ret != 0:
        print(f"Error executing command: {cmd}")
        sys.exit(ret)

def split_data(input_path, train_path, test_path, train_ratio=0.8, seed=42):
    print(f"--- Splitting Data: {input_path} ---")
    np.random.seed(seed)
    d = np.load(input_path, allow_pickle=True)
    
    logp = d["logp_arr"]
    S = len(logp)
    perm = np.random.permutation(S)
    
    split_idx = int(S * train_ratio)
    idx_tr = perm[:split_idx]
    idx_te = perm[split_idx:]
    
    print(f"Total: {S} | Train: {len(idx_tr)} | Test: {len(idx_te)}")
    
    # Save Train
    np.savez_compressed(
        train_path,
        logp_arr=d["logp_arr"][idx_tr],
        T_arr=d["T_arr"][idx_tr],
        q_arr=d["q_arr"][idx_tr],
        Fnet_arr=d["Fnet_arr"][idx_tr],
        Ts_K=d["Ts_K"][idx_tr]
    )
    
    # Save Test
    np.savez_compressed(
        test_path,
        logp_arr=d["logp_arr"][idx_te],
        T_arr=d["T_arr"][idx_te],
        q_arr=d["q_arr"][idx_te],
        Fnet_arr=d["Fnet_arr"][idx_te],
        Ts_K=d["Ts_K"][idx_te]
    )
    print("Data split saved.")

def main():
    parser = argparse.ArgumentParser(description="Automated Fine-Tuning Pipeline")
    parser.add_argument("--pretrained_ckpt", type=str, required=True, help="Path to the Stage-1 pretrained checkpoint (.pt)")
    parser.add_argument("--target_data", type=str, required=True, help="Path to the target domain dataset (.npz)")
    parser.add_argument("--job_name", type=str, default="finetune_job", help="Name for output folders")
    parser.add_argument("--epochs", type=int, default=50, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    # Define paths
    temp_train = f"temp_{args.job_name}_train.npz"
    temp_test  = f"temp_{args.job_name}_test.npz"
    out_dir    = f"output_{args.job_name}"
    
    # 1. Clean Data Split
    split_data(args.target_data, temp_train, temp_test)
    
    # 2. Run Fine-Tuning
    # Using model_train.py (renamed from stepB2p3e_train.py)
    train_cmd = (
        f"python model_train.py "
        f"--data {temp_train} "
        f"--out {out_dir} "
        f"--ckpt {args.pretrained_ckpt} "
        f"--epochs {args.epochs} "
        f"--batch 128 "
        f"--lr {args.lr} "
        f"--Ts_tail 320 "
        f"--tail_mult 2.0"
    )
    run_command(train_cmd)
    
    # 3. Run Evaluation
    eval_out_dir = f"{out_dir}/eval_results"
    best_model = os.path.join(out_dir, "best_hr_toa_boa.pt")
    
    eval_cmd = (
        f"python model_eval.py "
        f"--ckpt {best_model} "
        f"--data {temp_test} "
        f"--out {eval_out_dir} "
        f"--bot_window_k 0"
    )
    run_command(eval_cmd)
    
    # 4. Cleanup
    if os.path.exists(temp_train): os.remove(temp_train)
    if os.path.exists(temp_test): os.remove(temp_test)
    
    print("\n=== Pipeline Completed Successfully ===")
    print(f"Results saved in: {eval_out_dir}")

if __name__ == "__main__":
    main()
