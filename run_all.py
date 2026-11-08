#!/usr/bin/env python3
"""
Run all experiments sequentially in background.
Saves results to results/ directory.
"""
import os
import sys
import subprocess

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

experiments = [
    ("deepdta",       "davis", 15),
    ("attention_dta", "davis", 15),
    ("graphdta",      "davis", 15),
]

for model, dataset, epochs in experiments:
    print(f"\n{'='*60}", flush=True)
    print(f"  EXPERIMENT: {model} on {dataset} ({epochs} epochs)", flush=True)
    print(f"{'='*60}\n", flush=True)
    cmd = [
        sys.executable, "train.py",
        "--model", model,
        "--dataset", dataset,
        "--epochs", str(epochs),
        "--batch_size", "512",
        "--patience", "10",
    ]
    subprocess.run(cmd, cwd=BASE)

# Generate figures
print("\n  Generating figures...", flush=True)
subprocess.run([sys.executable, "plot_results.py"], cwd=BASE)
print("\n  ALL EXPERIMENTS COMPLETE!", flush=True)
