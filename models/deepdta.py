"""
DeepDTA: CNN-based Drug-Target Binding Affinity prediction.

Reference:
  Ozturk et al. (2018) "DeepDTA: prediction of drug-target binding
  affinity using deep learning."  Bioinformatics, 34(17), i821-i829.

Architecture:
  Drug branch:   SMILES → embedding → 3×1D-Conv → max-pool
  Target branch: Protein seq → embedding → 3×1D-Conv → max-pool
  Fusion:        concat → 3×FC → scalar affinity
"""

import torch
import torch.nn as nn


class DeepDTA(nn.Module):
    """DeepDTA model for DTA prediction."""

    def __init__(self, smiles_vocab_size, prot_vocab_size,
                 embed_dim=128, filter_num=32,
                 smi_max_len=85, prot_max_len=1200,
                 fc_dims=(1024, 1024, 512)):
        super().__init__()

        # ---- Drug (SMILES) branch ----
        self.drug_embed = nn.Embedding(smiles_vocab_size, embed_dim,
                                       padding_idx=0)
        self.drug_conv = nn.Sequential(
            nn.Conv1d(embed_dim, filter_num, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(filter_num, filter_num * 2, kernel_size=6),
            nn.ReLU(),
            nn.Conv1d(filter_num * 2, filter_num * 3, kernel_size=8),
            nn.ReLU(),
        )
        self.drug_pool = nn.AdaptiveMaxPool1d(1)

        # ---- Target (protein) branch ----
        self.prot_embed = nn.Embedding(prot_vocab_size, embed_dim,
                                       padding_idx=0)
        self.prot_conv = nn.Sequential(
            nn.Conv1d(embed_dim, filter_num, kernel_size=4),
            nn.ReLU(),
            nn.Conv1d(filter_num, filter_num * 2, kernel_size=8),
            nn.ReLU(),
            nn.Conv1d(filter_num * 2, filter_num * 3, kernel_size=12),
            nn.ReLU(),
        )
        self.prot_pool = nn.AdaptiveMaxPool1d(1)

        # ---- Fusion (FC) ----
        drug_dim = filter_num * 3
        prot_dim = filter_num * 3
        fused_dim = drug_dim + prot_dim
        layers = []
        in_dim = fused_dim
        for out_dim in fc_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, drug_enc, prot_enc):
        """
        Parameters
        ----------
        drug_enc : LongTensor [B, smi_max_len]  integer-encoded SMILES
        prot_enc : LongTensor [B, prot_max_len] integer-encoded protein

        Returns
        -------
        FloatTensor [B, 1]  predicted binding affinity
        """
        # Drug: [B, L] → [B, L, E] → [B, E, L] → conv → [B, C, L'] → pool → [B, C]
        d = self.drug_embed(drug_enc)            # [B, L, E]
        d = d.permute(0, 2, 1)                    # [B, E, L]
        d = self.drug_conv(d)                     # [B, C, L']
        d = self.drug_pool(d).squeeze(-1)         # [B, C]

        # Target: same pattern
        p = self.prot_embed(prot_enc)             # [B, L, E]
        p = p.permute(0, 2, 1)                    # [B, E, L]
        p = self.prot_conv(p)                     # [B, C, L']
        p = self.prot_pool(p).squeeze(-1)         # [B, C]

        # Fusion
        fused = torch.cat([d, p], dim=1)          # [B, C_d + C_p]
        out = self.fc(fused)                      # [B, 1]
        return out
