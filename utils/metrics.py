"""
Evaluation metrics for Drug-Target Binding Affinity prediction.

Implements:
  - MSE (Mean Squared Error)
  - RMSE (Root Mean Squared Error)
  - Pearson correlation coefficient
  - Concordance Index (CI)

The Concordance Index is a ranking-based metric widely used in
survival analysis and DTA prediction. It measures the fraction of
concordant pairs among all comparable pairs.
"""

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error


def mse(y_true, y_pred):
    """Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def pearson(y_true, y_pred):
    """Pearson correlation coefficient."""
    r, _ = pearsonr(y_true.flatten(), y_pred.flatten())
    return r


def ci(y_true, y_pred):
    """
    Concordance Index.

    For every pair (i, j) where y_true[i] > y_true[j], a pair is
    concordant if y_pred[i] > y_pred[j].  Ties count as 0.5.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)
    concordant = 0.0
    total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if y_true[i] < y_true[j]:
                    if y_pred[i] < y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5
                else:  # y_true[i] > y_true[j]
                    if y_pred[i] > y_pred[j]:
                        concordant += 1
                    elif y_pred[i] == y_pred[j]:
                        concordant += 0.5
    return concordant / total if total > 0 else 0.0


def all_metrics(y_true, y_pred):
    """Return a dict with all four metrics."""
    return {
        "MSE": mse(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "Pearson": pearson(y_true, y_pred),
        "CI": ci(y_true, y_pred),
    }
