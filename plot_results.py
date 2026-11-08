#!/usr/bin/env python3
"""
Generate result figures for the project report.

Reads saved predictions and training histories from results/ and
produces:
  - Scatter plots (predicted vs. true affinity)
  - Training loss curves
  - Comparison bar charts across models
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

MODEL_LABELS = {
    "deepdta": "DeepDTA",
    "graphdta": "GraphDTA",
    "attention_dta": "Attention-DTA",
}
MODEL_COLORS = {
    "deepdta": "#4C72B0",
    "graphdta": "#55A868",
    "attention_dta": "#C44E52",
}


def plot_scatter(dataset="davis"):
    """Scatter plots of predicted vs. true affinity for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ["deepdta", "graphdta", "attention_dta"]
    for ax, model in zip(axes, models):
        fpath = os.path.join(RESULTS_DIR, f"{model}_{dataset}_preds.npz")
        if not os.path.exists(fpath):
            ax.set_title(f"{MODEL_LABELS[model]} (no data)")
            ax.axis("off")
            continue
        data = np.load(fpath)
        preds, trues = data["preds"], data["trues"]
        ax.scatter(trues, preds, alpha=0.3, s=8,
                   c=MODEL_COLORS[model], edgecolors="none")
        lim_min = min(trues.min(), preds.min())
        lim_max = max(trues.max(), preds.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", lw=1, alpha=0.5)
        ax.set_xlabel("True Affinity", fontsize=11)
        ax.set_ylabel("Predicted Affinity", fontsize=11)
        ax.set_title(MODEL_LABELS[model], fontsize=13)
        ax.set_aspect("equal")
    plt.suptitle(f"Predicted vs. True Binding Affinity ({dataset.upper()})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"scatter_{dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_loss_curves(dataset="davis"):
    """Training/validation loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = ["deepdta", "graphdta", "attention_dta"]
    for ax, model in zip(axes, models):
        fpath = os.path.join(RESULTS_DIR, f"{model}_{dataset}_history.npz")
        if not os.path.exists(fpath):
            ax.set_title(f"{MODEL_LABELS[model]} (no data)")
            ax.axis("off")
            continue
        data = np.load(fpath)
        train_loss = data["train_loss"]
        val_loss = data["val_loss"]
        epochs = range(1, len(train_loss) + 1)
        ax.plot(epochs, train_loss, "-", color=MODEL_COLORS[model],
                alpha=0.7, label="Train MSE")
        ax.plot(epochs, val_loss, "--", color=MODEL_COLORS[model],
                label="Val MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(MODEL_LABELS[model], fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"Training & Validation Loss Curves ({dataset.upper()})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"loss_curves_{dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def plot_comparison_bar(dataset="davis"):
    """Bar chart comparing metrics across models."""
    models = ["deepdta", "graphdta", "attention_dta"]
    metrics_list = ["MSE", "RMSE", "Pearson", "CI"]
    values = {m: [] for m in metrics_list}

    for model in models:
        fpath = os.path.join(RESULTS_DIR, f"{model}_{dataset}_results.json")
        if os.path.exists(fpath):
            with open(fpath) as f:
                res = json.load(f)
            for metric in metrics_list:
                values[metric].append(res["metrics"][metric])
        else:
            for metric in metrics_list:
                values[metric].append(0.0)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x = np.arange(len(models))
    bar_labels = [MODEL_LABELS[m] for m in models]
    colors = [MODEL_COLORS[m] for m in models]

    for ax, metric in zip(axes, metrics_list):
        bars = ax.bar(x, values[metric], color=colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=9, rotation=15)
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val in zip(bars, values[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle(f"Model Comparison on {dataset.upper()}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, f"comparison_{dataset}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")


def main():
    datasets = ["davis", "kiba"]
    for ds in datasets:
        print(f"\nGenerating figures for {ds.upper()}...")
        plot_scatter(ds)
        plot_loss_curves(ds)
        plot_comparison_bar(ds)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
