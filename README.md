# Comparative Evaluation of Deep Learning Architectures for Drug-Target Binding Affinity Prediction with Attention-Based Improvement

**AIDD8122 – Advanced Topics in Artificial Intelligence & Drug Discovery**

**Student:** Hao Wenli (p2626817)

**Date:** November 2026

---

## Overview

This project implements and evaluates three deep learning architectures for
predicting drug-target binding affinity (DTA):

| Model           | Drug Representation | Target Representation | Fusion Strategy        |
|-----------------|---------------------|-----------------------|------------------------|
| **DeepDTA**     | 1D CNN on SMILES    | 1D CNN on protein seq | Concatenation          |
| **GraphDTA**    | GNN on molecular graph | 1D CNN on protein seq | Concatenation       |
| **Attention-DTA** (proposed) | 1D CNN on SMILES | 1D CNN on protein seq | Bidirectional Cross-Attention |

The **Attention-DTA** model is the proposed improvement: it retains the
DeepDTA CNN backbone but replaces simple concatenation with a multi-head
cross-attention module that lets drug substructure features attend to
protein-region features (and vice versa), enabling the model to learn
which molecular fragments interact with which protein domains.

## Datasets

| Dataset | Drugs | Targets | Interactions | Affinity Type   |
|---------|-------|---------|--------------|-----------------|
| DAVIS   | 68    | 379     | 30,056       | Kd (pKd transform) |
| KIBA    | 2,111 | 229     | 118,254      | KIBA score      |

Data is automatically downloaded from the
[DeepDTA repository](https://github.com/hkmztrk/DeepDTA).

## Project Structure

```
AIDD8122_Final_Project/
├── data/
│   ├── data_preprocessing.py   # Data loading, encoding, graph construction
│   └── datasets/               # Downloaded raw data (auto-created)
├── models/
│   ├── deepdta.py              # DeepDTA (CNN baseline)
│   ├── graphdta.py             # GraphDTA (GNN baseline)
│   └── attention_dta.py        # Attention-DTA (proposed model)
├── utils/
│   ├── metrics.py              # MSE, RMSE, Pearson, CI
│   └── utils.py                # Encoding, graph utils, early stopping
├── train.py                    # Main training & evaluation script
├── run_experiments.py          # Batch experiment runner
├── run_all.py                  # Run all models on DAVIS
├── plot_results.py             # Generate figures for report
├── report.md                   # Project report (Markdown version)
├── results/                    # JSON results + predictions (auto-created)
├── figures/                    # Generated plots (auto-created)
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Python 3.10+
- PyTorch 2.0+
- RDKit 2022+
- scikit-learn, scipy, pandas, matplotlib

## Usage

### Train a single model

```bash
# DeepDTA on DAVIS
python train.py --model deepdta --dataset davis --epochs 100

# GraphDTA on KIBA
python train.py --model graphdta --dataset kiba --epochs 100

# Attention-DTA (proposed) on DAVIS
python train.py --model attention_dta --dataset davis --epochs 100
```

### Run all experiments

```bash
python run_all.py
```

### Generate figures

```bash
python plot_results.py
```

## Evaluation Metrics

| Metric  | Description                                      |
|---------|--------------------------------------------------|
| MSE     | Mean Squared Error                               |
| RMSE    | Root Mean Squared Error                          |
| Pearson | Pearson correlation coefficient                  |
| CI      | Concordance Index (ranking-based)               |

## Key Findings

- All three models achieve competitive DTA prediction performance on DAVIS.
- The proposed **Attention-DTA** improves over DeepDTA by explicitly
  modeling drug-target interactions through cross-attention.
- **GraphDTA** benefits from richer molecular graph representations.
- See the full report for detailed results and analysis.

## References

1. Ozturk et al. (2018). DeepDTA. *Bioinformatics*, 34(17), i821-i829.
2. Nguyen et al. (2021). GraphDTA. *Bioinformatics*, 37(8), 1140-1147.
3. Davis et al. (2011). DAVIS dataset. *Nature Biotechnology*, 29(11), 1030-1037.
4. Tang et al. (2014). KIBA. *Journal of Cheminformatics*, 6, 36.
5. Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.

## License

This project is for educational purposes (AIDD8122 course project).
