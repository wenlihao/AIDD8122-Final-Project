"""
Utility functions: SMILES encoding, protein encoding, molecular graph
construction, reproducibility helpers, and early stopping.
"""

import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# SMILES character-level encoding (for CNN-based models)
# ---------------------------------------------------------------------------

MAX_SMILES_LEN = 85      # standard for DAVIS / KIBA
SMILES_CHARSET = set(
    "()*+-.0123456789@BCFHIKNOPSclnorsuy"
)  # 34 common SMILES tokens

def create_smiles_vocab(smiles_list):
    """
    Build a character vocabulary from a list of SMILES strings.
    Returns (char2idx, idx2char, vocab_size).
    """
    charset = set()
    for s in smiles_list:
        charset.update(list(s))
    charset = sorted(charset)
    char2idx = {" ": 0, "UNK": 1}
    for i, c in enumerate(charset, start=2):
        char2idx[c] = i
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(char2idx)


def encode_smiles(smiles, char2idx, max_len=MAX_SMILES_LEN):
    """Encode a single SMILES string into an integer vector."""
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, c in enumerate(smiles[:max_len]):
        encoded[i] = char2idx.get(c, char2idx["UNK"])
    return encoded


def encode_smiles_batch(smiles_list, char2idx, max_len=MAX_SMILES_LEN):
    """Encode a list of SMILES strings into a 2-D integer array."""
    return np.array(
        [encode_smiles(s, char2idx, max_len) for s in smiles_list]
    )


# ---------------------------------------------------------------------------
# Protein sequence character-level encoding (for CNN-based models)
# ---------------------------------------------------------------------------

MAX_PROT_LEN = 1200       # standard for DAVIS / KIBA

def create_protein_vocab(sequences):
    """Build amino-acid vocabulary from protein sequences."""
    charset = set()
    for seq in sequences:
        charset.update(list(seq))
    charset = sorted(charset)
    char2idx = {" ": 0, "UNK": 1}
    for i, c in enumerate(charset, start=2):
        char2idx[c] = i
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char, len(char2idx)


def encode_protein(seq, char2idx, max_len=MAX_PROT_LEN):
    """Encode a single protein sequence into an integer vector."""
    encoded = np.zeros(max_len, dtype=np.int64)
    for i, c in enumerate(seq[:max_len]):
        encoded[i] = char2idx.get(c, char2idx["UNK"])
    return encoded


def encode_protein_batch(sequences, char2idx, max_len=MAX_PROT_LEN):
    """Encode a list of protein sequences into a 2-D integer array."""
    return np.array(
        [encode_protein(s, char2idx, max_len) for s in sequences]
    )


# ---------------------------------------------------------------------------
# Molecular graph construction (for GNN-based models)
# ---------------------------------------------------------------------------

# Atom features used by GraphDTA
ATOM_FEATURES = {
    "atomic_num": [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 27, 29, 35, 53, 0],
    "degree": [0, 1, 2, 3, 4, 5, 6],
    "formal_charge": [-2, -1, 0, 1, 2],
    "chiral_tag": [0, 1, 2, 3],
    "num_h": [0, 1, 2, 3, 4],
    "hybridization": [0, 1, 2, 3, 4],
    "is_aromatic": [0, 1],
}


def _onehot(val, choices):
    """Convert a value to a one-hot vector given the list of choices."""
    vec = [0.0] * len(choices)
    idx = choices.index(val) if val in choices else len(choices) - 1
    vec[idx] = 1.0
    return vec


def atom_features(atom):
    """Compute feature vector for a single RDKit atom."""
    from rdkit import Chem
    return (
        _onehot(atom.GetAtomicNum(), ATOM_FEATURES["atomic_num"])
        + _onehot(atom.GetDegree(), ATOM_FEATURES["degree"])
        + _onehot(atom.GetFormalCharge(), ATOM_FEATURES["formal_charge"])
        + _onehot(int(atom.GetChiralTag()), ATOM_FEATURES["chiral_tag"])
        + _onehot(atom.GetTotalNumHs(), ATOM_FEATURES["num_h"])
        + _onehot(int(atom.GetHybridization()), ATOM_FEATURES["hybridization"])
        + _onehot(int(atom.GetIsAromatic()), ATOM_FEATURES["is_aromatic"])
    )


ATOM_FEATURE_DIM = (
    len(ATOM_FEATURES["atomic_num"])
    + len(ATOM_FEATURES["degree"])
    + len(ATOM_FEATURES["formal_charge"])
    + len(ATOM_FEATURES["chiral_tag"])
    + len(ATOM_FEATURES["num_h"])
    + len(ATOM_FEATURES["hybridization"])
    + len(ATOM_FEATURES["is_aromatic"])
)  # = 78


def smiles_to_graph(smiles):
    """
    Convert a SMILES string to a molecular graph.

    Returns a dict with:
      - x: node feature matrix [num_atoms, 78]
      - edge_index: edge list [2, num_edges] (undirected, bidirectional)
    """
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # fallback: single carbon atom
        mol = Chem.MolFromSmiles("C")

    atom_list = [atom_features(a) for a in mol.GetAtoms()]
    x = np.array(atom_list, dtype=np.float32)

    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # bidirectional
    if len(edges) == 0:
        edges = [[0, 0]]
    edge_index = np.array(edges, dtype=np.int64).T  # [2, E]
    return {"x": x, "edge_index": edge_index}


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Simple early stopping based on validation MSE."""

    def __init__(self, patience=20, min_delta=0.0, save_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
