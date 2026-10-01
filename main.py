"""
main.py
Orchestrator for NST_Proj V2.1
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_generator(count):
    script = ROOT / "preprocessing" / "generator.py"
    print(f"--- Running Generator ({count} samples) ---")
    subprocess.run([sys.executable, str(script), "--count", str(count)], check=True)

def run_training(args):
    script = ROOT / "training" / "train.py"
    print("--- Running Training ---")
    cmd = [
        sys.executable, str(script),
        "--data-dir", str(ROOT / args.data_dir),
        "--results-dir", args.results_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--accumulate-steps", str(args.accumulate_steps),
        "--workers", str(args.workers)
    ]
    if args.run_name:
        cmd.extend(["--run-name", args.run_name])
    if args.wandb:
        cmd.append("--wandb")
    subprocess.run(cmd, check=True)

def run_evaluation(args):
    script = ROOT / "testing" / "evaluate.py"
    
    targets = ["student", "teacher"] if args.eval_target == "both" else [args.eval_target]
    
    for target in targets:
        print(f"--- Running Evaluation for {target.upper()} ---")
        
        if args.run_name:
            base_dir = ROOT / args.results_dir / args.run_name
        else:
            base_dir = ROOT / args.results_dir
            
        weights_path = base_dir / f"best_{target}.pth"
        if not weights_path.exists() and (base_dir / "best_model.pth").exists():
            print(f"Warning: best_{target}.pth not found, falling back to legacy best_model.pth")
            weights_path = base_dir / "best_model.pth"
            
        out_dir = base_dir / f"predictions_{target}"
            
        cmd = [
            sys.executable, str(script),
            "--data-dir", str(ROOT / args.data_dir),
            "--weights", str(weights_path),
            "--batch-size", str(args.batch_size),
            "--out-dir", str(out_dir)
        ]
        subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="NST Project V2.1 Orchestrator")
    parser.add_argument("--mode", choices=["generate", "train", "evaluate", "all"], default="all", help="What to run")
    parser.add_argument("--count", type=int, default=7000, help="Number of samples to generate")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-name", type=str, default="", help="Experiment run name")
    parser.add_argument("--eval-target", choices=["student", "teacher", "both"], default="student", help="Which model to evaluate")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    
    # Training args
    parser.add_argument("--data-dir", type=str, default="generated_data_v2.1")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulate-steps", type=int, default=8)
    
    args = parser.parse_args()
    
    if args.mode in ["generate", "all"]:
        run_generator(args.count)
        
    if args.mode in ["train", "all"]:
        run_training(args)
        
    if args.mode in ["evaluate", "all"]:
        run_evaluation(args)

if __name__ == "__main__":
    main()
