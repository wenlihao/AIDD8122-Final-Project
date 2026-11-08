#!/usr/bin/env python3
"""
Batch experiment runner: trains all three models on both datasets.

Usage:
    python run_experiments.py [--epochs N] [--quick]

    --quick   Use synthetic data + few epochs for pipeline testing.
"""

import os
import sys
import subprocess
import argparse

def run_experiment(model, dataset, epochs, results_dir="results"):
    """Run a single experiment via train.py."""
    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--results_dir", results_dir,
    ]
    print(f"\n{'='*60}")
    print(f"  Running: {model} on {dataset} ({epochs} epochs)")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with synthetic data")
    args = parser.parse_args()

    if args.quick:
        models = ["deepdta", "attention_dta"]
        datasets = ["synthetic"]
        epochs = 10
    else:
        models = ["deepdta", "graphdta", "attention_dta"]
        datasets = ["davis", "kiba"]
        epochs = args.epochs

    results = {}
    for dataset in datasets:
        for model in models:
            success = run_experiment(model, dataset, epochs)
            results[f"{model}_{dataset}"] = "PASS" if success else "FAIL"

    print(f"\n{'='*60}")
    print("  EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    for exp, status in results.items():
        print(f"  {exp:40s} : {status}")

    # Generate figures
    print(f"\n  Generating figures...")
    subprocess.run(
        [sys.executable, "plot_results.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )


if __name__ == "__main__":
    main()
