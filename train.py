#!/usr/bin/env python3
"""
Main training & evaluation script for DTA prediction.

Usage examples:
    # Train DeepDTA on DAVIS
    python train.py --model deepdta --dataset davis --epochs 100

    # Train GraphDTA on KIBA
    python train.py --model graphdta --dataset kiba --epochs 100

    # Train Attention-DTA (proposed) on DAVIS
    python train.py --model attention_dta --dataset davis --epochs 100

    # Quick test with synthetic data
    python train.py --model deepdta --dataset synthetic --epochs 10
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.data_preprocessing import prepare_data
from models.deepdta import DeepDTA
from models.graphdta import GraphDTA
from models.attention_dta import AttentionDTA
from utils.utils import set_seed, get_device, EarlyStopping
from utils.metrics import all_metrics

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="DTA prediction training")
    p.add_argument("--model", type=str, default="deepdta",
                   choices=["deepdta", "graphdta", "attention_dta"],
                   help="Model architecture")
    p.add_argument("--dataset", type=str, default="davis",
                   choices=["davis", "kiba", "synthetic"],
                   help="Dataset name")
    p.add_argument("--epochs", type=int, default=100,
                   help="Max training epochs")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size")
    p.add_argument("--lr", type=float, default=0.001,
                   help="Learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stopping patience")
    p.add_argument("--results_dir", type=str, default="results",
                   help="Directory to save results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(model_name, data_info, device):
    """Instantiate the requested model and move to device."""
    if model_name == "deepdta":
        model = DeepDTA(
            smiles_vocab_size=data_info["smiles_vocab_size"],
            prot_vocab_size=data_info["prot_vocab_size"],
        )
    elif model_name == "graphdta":
        model = GraphDTA(
            atom_feature_dim=data_info["atom_feature_dim"],
            prot_vocab_size=data_info["prot_vocab_size"],
        )
    elif model_name == "attention_dta":
        model = AttentionDTA(
            smiles_vocab_size=data_info["smiles_vocab_size"],
            prot_vocab_size=data_info["prot_vocab_size"],
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model_name} | Parameters: {n_params:,}")
    return model


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, model_type):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        if model_type == "graph":
            x, edge_index, prot_enc, affinity, batch_vec = [
                b.to(device) for b in batch
            ]
            pred = model(x, edge_index, prot_enc, batch_vec)
        else:
            drug_enc, prot_enc, affinity = [b.to(device) for b in batch]
            pred = model(drug_enc, prot_enc)
        loss = criterion(pred, affinity)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * affinity.size(0)
        n_samples += affinity.size(0)
    return total_loss / n_samples


@torch.no_grad()
def evaluate(model, loader, device, model_type):
    """Evaluate on a data loader, return predictions and metrics."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        if model_type == "graph":
            x, edge_index, prot_enc, affinity, batch_vec = [
                b.to(device) for b in batch
            ]
            pred = model(x, edge_index, prot_enc, batch_vec)
        else:
            drug_enc, prot_enc, affinity = [b.to(device) for b in batch]
            pred = model(drug_enc, prot_enc)
        preds.append(pred.cpu().numpy())
        trues.append(affinity.cpu().numpy())
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    metrics = all_metrics(trues, preds)
    return metrics, preds, trues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    model_type = "graph" if args.model == "graphdta" else "cnn"

    # --- Data ---
    print(f"\n[1/4] Preparing data: {args.dataset}")
    data_info = prepare_data(
        dataset_name=args.dataset,
        model_type=model_type,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    # --- Model ---
    print(f"\n[2/4] Building model: {args.model}")
    model = build_model(args.model, data_info, device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(args.results_dir, exist_ok=True)
    ckpt_path = os.path.join(
        args.results_dir,
        f"{args.model}_{args.dataset}_best.pth",
    )
    early_stop = EarlyStopping(patience=args.patience, save_path=ckpt_path)

    # --- Training ---
    print(f"\n[3/4] Training (max {args.epochs} epochs, patience={args.patience})")
    history = {"train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, data_info["train_loader"], optimizer,
            criterion, device, model_type,
        )
        val_metrics, _, _ = evaluate(
            model, data_info["test_loader"], device, model_type,
        )
        val_loss = val_metrics["MSE"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        elapsed = time.time() - start_time
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train MSE: {train_loss:.4f} | "
              f"Val MSE: {val_loss:.4f} | "
              f"Val Pearson: {val_metrics['Pearson']:.4f} | "
              f"Val CI: {val_metrics['CI']:.4f} | "
              f"{elapsed:.0f}s")

        early_stop(val_loss, model)
        if early_stop.early_stop:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Load best model
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- Final evaluation ---
    print(f"\n[4/4] Final evaluation on test set")
    final_metrics, preds, trues = evaluate(
        model, data_info["test_loader"], device, model_type,
    )
    print(f"\n  ===== Results: {args.model} on {args.dataset} =====")
    for k, v in final_metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    # Save results
    result_file = os.path.join(
        args.results_dir,
        f"{args.model}_{args.dataset}_results.json",
    )
    result = {
        "model": args.model,
        "dataset": args.dataset,
        "metrics": {k: float(v) for k, v in final_metrics.items()},
        "epochs_trained": len(history["train_loss"]),
        "train_size": int(data_info["train_size"]),
        "test_size": int(data_info["test_size"]),
        "seed": args.seed,
    }
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results saved to {result_file}")

    # Save predictions for plotting
    np.savez(
        os.path.join(args.results_dir,
                     f"{args.model}_{args.dataset}_preds.npz"),
        preds=preds, trues=trues,
    )

    # Save training history
    np.savez(
        os.path.join(args.results_dir,
                     f"{args.model}_{args.dataset}_history.npz"),
        train_loss=np.array(history["train_loss"]),
        val_loss=np.array(history["val_loss"]),
    )

    total_time = time.time() - start_time
    print(f"\n  Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
