"""
Attention-DTA: DeepDTA with Cross-Attention for Drug-Target Interaction.

This is the *proposed improvement* in this project.

Motivation:
  DeepDTA extracts drug and target features independently with 1-D CNNs,
  then simply concatenates them.  This ignores *which* substructures in
  the drug interact with *which* regions in the protein.  We add a
  cross-attention module so the model can learn drug-target interaction
  patterns explicitly.

Architecture:
  Drug branch:   SMILES → embedding → 3×1D-Conv → feature map [B, C, L_d]
  Target branch: Protein seq → embedding → 3×1D-Conv → feature map [B, C, L_p]
  Cross-Attention:
      Q_drug  = W_q · drug_map      [B, C, L_d]
      K_target= W_k · target_map    [B, C, L_p]
      V_target= W_v · target_map    [B, C, L_p]
      attended_drug = softmax(Q·K^T / √d) · V   → [B, C, L_d]
      (symmetrically for target-attends-drug)
  Pool + Concat attended features
  FC layers → scalar affinity
"""

import math
import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention between drug and target feature maps.

    The drug features act as queries that attend to target key/value
    pairs, and vice versa.  This lets the model capture which drug
    substructures interact with which protein regions.
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Linear projections for drug (query source) and target (key/value source)
        self.W_q_d = nn.Linear(dim, dim)
        self.W_k_t = nn.Linear(dim, dim)
        self.W_v_t = nn.Linear(dim, dim)

        self.W_q_t = nn.Linear(dim, dim)
        self.W_k_d = nn.Linear(dim, dim)
        self.W_v_d = nn.Linear(dim, dim)

        self.out_proj_d = nn.Linear(dim, dim)
        self.out_proj_t = nn.Linear(dim, dim)

    def _multihead(self, q, k, v):
        """Scaled dot-product attention with multi-head reshaping.

        q has length L_q, k and v have length L_kv (may differ in
        cross-attention).
        """
        B, C, L_q = q.shape
        _, _, L_kv = k.shape
        H, Dh = self.num_heads, self.head_dim

        q = q.view(B, H, Dh, L_q).transpose(1, 3)   # [B, L_q, H, Dh]
        k = k.view(B, H, Dh, L_kv).transpose(1, 3)  # [B, L_kv, H, Dh]
        v = v.view(B, H, Dh, L_kv).transpose(1, 3)  # [B, L_kv, H, Dh]

        scores = torch.einsum("blhd,bmhd->blm", q, k) / math.sqrt(Dh)
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("blm,bmhd->blhd", attn, v)
        out = out.transpose(1, 2).contiguous().view(B, C, L_q)
        return out

    def forward(self, drug_feat, target_feat):
        """
        Parameters
        ----------
        drug_feat   : FloatTensor [B, C, L_d]   drug CNN feature map
        target_feat : FloatTensor [B, C, L_p]   target CNN feature map

        Returns
        -------
        (attended_drug, attended_target) each [B, C, L]
        """
        # Drug attends to target
        q_d = self.W_q_d(drug_feat.transpose(1, 2)).transpose(1, 2)
        k_t = self.W_k_t(target_feat.transpose(1, 2)).transpose(1, 2)
        v_t = self.W_v_t(target_feat.transpose(1, 2)).transpose(1, 2)
        attended_d = self._multihead(q_d, k_t, v_t)
        attended_d = self.out_proj_d(attended_d.transpose(1, 2)).transpose(1, 2)

        # Target attends to drug
        q_t = self.W_q_t(target_feat.transpose(1, 2)).transpose(1, 2)
        k_d = self.W_k_d(drug_feat.transpose(1, 2)).transpose(1, 2)
        v_d = self.W_v_d(drug_feat.transpose(1, 2)).transpose(1, 2)
        attended_t = self._multihead(q_t, k_d, v_d)
        attended_t = self.out_proj_t(attended_t.transpose(1, 2)).transpose(1, 2)

        return attended_d, attended_t


class AttentionDTA(nn.Module):
    """
    Proposed model: DeepDTA backbone + cross-attention fusion.

    The CNN backbones are identical to DeepDTA, but instead of
    concatenating pooled representations, we apply bidirectional
    cross-attention between the full feature maps before pooling.
    """

    def __init__(self, smiles_vocab_size, prot_vocab_size,
                 embed_dim=128, filter_num=32,
                 smi_max_len=85, prot_max_len=1200,
                 num_heads=4,
                 fc_dims=(512, 256)):
        super().__init__()

        # ---- Drug (SMILES) branch — same as DeepDTA ----
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

        # ---- Target (protein) branch — same as DeepDTA ----
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

        # ---- Cross-attention ----
        feat_dim = filter_num * 3  # 96
        self.cross_attn = CrossAttention(feat_dim, num_heads=num_heads)

        # Pooling
        self.drug_pool = nn.AdaptiveMaxPool1d(1)
        self.prot_pool = nn.AdaptiveMaxPool1d(1)

        # ---- Fusion ----
        fused_dim = feat_dim * 2  # attended_drug + attended_target
        layers = []
        in_d = fused_dim
        for out_d in fc_dims:
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_d = out_d
        layers.append(nn.Linear(in_d, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, drug_enc, prot_enc):
        # Drug CNN feature map
        d = self.drug_embed(drug_enc).permute(0, 2, 1)  # [B, E, L]
        d = self.drug_conv(d)                            # [B, C, L']

        # Target CNN feature map
        p = self.prot_embed(prot_enc).permute(0, 2, 1)
        p = self.prot_conv(p)                            # [B, C, L'']

        # Cross-attention (the key innovation)
        d_att, p_att = self.cross_attn(d, p)

        # Pool
        d_out = self.drug_pool(d_att).squeeze(-1)        # [B, C]
        p_out = self.prot_pool(p_att).squeeze(-1)        # [B, C]

        # Fuse & predict
        fused = torch.cat([d_out, p_out], dim=1)
        out = self.fc(fused)
        return out
