"""
Data preprocessing and loading for DTA prediction.

Supports two benchmark datasets:
  - DAVIS  (68 drugs x 379 targets, Kd affinity)
  - KIBA   (2111 drugs x 229 targets, KIBA score)

The raw data is expected as a CSV / TSV with at least three columns:
    smiles | target_sequence | affinity

If the data files are not present locally, this module attempts to
download them automatically.  A synthetic fallback generator is also
provided so the pipeline can be tested without real data.
"""

import os
import io
import urllib.request
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils.utils import (
    create_smiles_vocab,
    encode_smiles_batch,
    create_protein_vocab,
    encode_protein_batch,
    smiles_to_graph,
    ATOM_FEATURE_DIM,
    MAX_SMILES_LEN,
    MAX_PROT_LEN,
)

# ---------------------------------------------------------------------------
# Dataset URLs (DeepDTA preprocessed data hosted on GitHub)
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

# DeepDTA GitHub raw-file base
_GH_BASE = (
    "https://raw.githubusercontent.com/hkmztrk/DeepDTA/master/data"
)

# Per-dataset file lists
DAVIS_FILES = {
    "smiles":      f"{_GH_BASE}/davis/ligands_can.txt",
    "proteins":    f"{_GH_BASE}/davis/proteins.txt",
    "affinity":    f"{_GH_BASE}/davis/"
                   "drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt",
}
KIBA_FILES = {
    "smiles":      f"{_GH_BASE}/kiba/ligands_can.txt",
    "proteins":    f"{_GH_BASE}/kiba/proteins.txt",
    "affinity":    f"{_GH_BASE}/kiba/kiba_binding_affinity_v2.txt",
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download(url, dest):
    """Download *url* to *dest*."""
    print(f"  Downloading {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        f.write(resp.read())
    print(f"  Saved to {dest}")


def _ensure_files(file_dict, prefix):
    """Download all files in *file_dict* if not present locally."""
    paths = {}
    for key, url in file_dict.items():
        dest = os.path.join(DATA_DIR, f"{prefix}_{key}.txt")
        if not os.path.exists(dest):
            _download(url, dest)
        paths[key] = dest
    return paths


def _build_pairs(paths, affinity_transform=None):
    """
    Load SMILES (JSON dict), protein sequences (JSON dict), and affinity
    matrix, then build a flat DataFrame of (smiles, target, affinity).
    """
    import json
    with open(paths["smiles"]) as f:
        drug_dict = json.loads(f.read().strip())
    with open(paths["proteins"]) as f:
        prot_dict = json.loads(f.read().strip())
    aff = np.loadtxt(paths["affinity"])

    smiles_list = list(drug_dict.values())
    prot_list = list(prot_dict.values())
    n_drugs, n_prots = aff.shape

    rows = []
    for i in range(n_drugs):
        for j in range(n_prots):
            val = aff[i, j]
            if not np.isnan(val):
                a = affinity_transform(val) if affinity_transform else val
                rows.append({
                    "smiles": smiles_list[i],
                    "target": prot_list[j],
                    "affinity": a,
                })
    return pd.DataFrame(rows)


def _load_davis_raw():
    """Load DAVIS; download if necessary."""
    paths = _ensure_files(DAVIS_FILES, "davis")
    df = _build_pairs(
        paths,
        affinity_transform=lambda kd: -np.log10(kd * 1e-9) if kd > 0 else 5.0,
    )
    return df


def _load_kiba_raw():
    """Load KIBA; download if necessary."""
    paths = _ensure_files(KIBA_FILES, "kiba")
    df = _build_pairs(paths)  # KIBA scores used directly
    return df


def load_dataset(name="davis"):
    """
    Load a benchmark dataset by name.

    Parameters
    ----------
    name : str  ('davis' or 'kiba')

    Returns
    -------
    pd.DataFrame with columns ['smiles', 'target', 'affinity']
    """
    name = name.lower()
    if name == "davis":
        return _load_davis_raw()
    elif name == "kiba":
        return _load_kiba_raw()
    else:
        raise ValueError(f"Unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Synthetic fallback (for pipeline testing without real data)
# ---------------------------------------------------------------------------

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
SMILES_FRAGMENTS = ["CC", "CCC", "CCO", "c1ccccc1", "CCN", "CC(=O)O",
                    "CCCC", "CC(C)C", "C1CCCCC1", "CCOC"]


def generate_synthetic_data(n=2000, seed=42):
    """Generate a small synthetic DTA-like dataset for testing."""
    rng = np.random.RandomState(seed)
    smiles_list = [rng.choice(SMILES_FRAGMENTS) for _ in range(n)]
    prot_len = rng.randint(50, 200, size=n)
    targets = ["".join(rng.choice(list(AMINO_ACIDS), size=l)) for l in prot_len]
    # affinity: random float around 7-10 (pKd range)
    affinities = rng.normal(loc=8.0, scale=1.5, size=n)
    return pd.DataFrame({
        "smiles": smiles_list,
        "target": targets,
        "affinity": affinities,
    })


# ---------------------------------------------------------------------------
# PyTorch datasets
# ---------------------------------------------------------------------------

class DTADatasetCNN(Dataset):
    """
    Dataset for CNN-based models (DeepDTA, Attention-DTA).

    Each item returns:
      - drug_encoded: LongTensor [smiles_len]
      - target_encoded: LongTensor [prot_len]
      - affinity: FloatTensor [1]
    """

    def __init__(self, smiles_list, target_list, affinity_list,
                 smiles_vocab, prot_vocab,
                 max_smi=MAX_SMILES_LEN, max_prot=MAX_PROT_LEN):
        self.drug_enc = encode_smiles_batch(smiles_list, smiles_vocab, max_smi)
        self.target_enc = encode_protein_batch(target_list, prot_vocab, max_prot)
        self.affinity = np.array(affinity_list, dtype=np.float32)

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        return (
            torch.LongTensor(self.drug_enc[idx]),
            torch.LongTensor(self.target_enc[idx]),
            torch.FloatTensor([self.affinity[idx]]),
        )


class DTADatasetGraph(Dataset):
    """
    Dataset for GNN-based models (GraphDTA).

    Each item returns:
      - x: FloatTensor [num_atoms, 78]
      - edge_index: LongTensor [2, num_edges]
      - target_encoded: LongTensor [prot_len]
      - affinity: FloatTensor [1]
    """

    def __init__(self, smiles_list, target_list, affinity_list,
                 prot_vocab, max_prot=MAX_PROT_LEN):
        self.graphs = [smiles_to_graph(s) for s in smiles_list]
        self.target_enc = encode_protein_batch(target_list, prot_vocab, max_prot)
        self.affinity = np.array(affinity_list, dtype=np.float32)

    def __len__(self):
        return len(self.affinity)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return (
            torch.FloatTensor(g["x"]),
            torch.LongTensor(g["edge_index"]),
            torch.LongTensor(self.target_enc[idx]),
            torch.FloatTensor([self.affinity[idx]]),
        )


def collate_graph(batch):
    """
    Custom collate function for graph datasets.
    Batches graphs by concatenating node features and adjusting edge indices.
    """
    xs, edge_indices, targets, affinities = zip(*batch)
    # Concatenate node features
    x_batch = torch.cat(xs, dim=0)
    # Adjust and concatenate edge indices
    edge_batch = []
    offset = 0
    for ei, sz in zip(edge_indices, [x.size(0) for x in xs]):
        edge_batch.append(ei + offset)
        offset += sz
    edge_batch = torch.cat(edge_batch, dim=1)
    targets = torch.stack(targets)
    affinities = torch.stack(affinities)
    # Also return batch assignment vector for pooling
    batch_vec = torch.cat([
        torch.full((x.size(0),), i, dtype=torch.long)
        for i, x in enumerate(xs)
    ])
    return x_batch, edge_batch, targets, affinities, batch_vec


# ---------------------------------------------------------------------------
# Data preparation pipeline
# ---------------------------------------------------------------------------

def prepare_data(dataset_name="davis", model_type="cnn",
                 test_size=0.2, seed=42, batch_size=256):
    """
    End-to-end data preparation.

    Parameters
    ----------
    dataset_name : str  ('davis', 'kiba', or 'synthetic')
    model_type   : str  ('cnn' for DeepDTA/Attention-DTA,
                         'graph' for GraphDTA)
    test_size    : float  fraction held out for testing
    seed         : int    random seed for split
    batch_size   : int    batch size for DataLoaders

    Returns
    -------
    dict with keys: train_loader, test_loader, vocab info, device
    """
    # 1. Load raw data
    if dataset_name == "synthetic":
        df = generate_synthetic_data(n=2000, seed=seed)
    else:
        try:
            df = load_dataset(dataset_name)
        except Exception as e:
            print(f"  [WARNING] Failed to load {dataset_name}: {e}")
            print(f"  Falling back to synthetic data.")
            df = generate_synthetic_data(n=2000, seed=seed)

    print(f"  Loaded {len(df)} interactions from '{dataset_name}'")
    print(f"  Unique drugs: {df['smiles'].nunique()}, "
          f"Unique targets: {df['target'].nunique()}")
    print(f"  Affinity range: [{df['affinity'].min():.3f}, "
          f"{df['affinity'].max():.3f}]")

    # 2. Build vocabularies
    smiles_vocab = create_smiles_vocab(df["smiles"].tolist())
    prot_vocab = create_protein_vocab(df["target"].tolist())
    print(f"  SMILES vocab size: {len(smiles_vocab[0])}")
    print(f"  Protein vocab size: {len(prot_vocab[0])}")

    # 3. Train / test split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed
    )
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # 4. Build datasets
    if model_type == "graph":
        train_ds = DTADatasetGraph(
            train_df["smiles"].tolist(),
            train_df["target"].tolist(),
            train_df["affinity"].tolist(),
            prot_vocab[0],
        )
        test_ds = DTADatasetGraph(
            test_df["smiles"].tolist(),
            test_df["target"].tolist(),
            test_df["affinity"].tolist(),
            prot_vocab[0],
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=collate_graph, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            collate_fn=collate_graph, num_workers=0,
        )
    else:  # CNN
        train_ds = DTADatasetCNN(
            train_df["smiles"].tolist(),
            train_df["target"].tolist(),
            train_df["affinity"].tolist(),
            smiles_vocab[0], prot_vocab[0],
        )
        test_ds = DTADatasetCNN(
            test_df["smiles"].tolist(),
            test_df["target"].tolist(),
            test_df["affinity"].tolist(),
            smiles_vocab[0], prot_vocab[0],
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
        )

    return {
        "train_loader": train_loader,
        "test_loader": test_loader,
        "smiles_vocab_size": len(smiles_vocab[0]),
        "prot_vocab_size": len(prot_vocab[0]),
        "train_size": len(train_df),
        "test_size": len(test_df),
        "atom_feature_dim": ATOM_FEATURE_DIM,
    }
