"""
GraphDTA: GNN-based Drug-Target Binding Affinity prediction.

Reference:
  Nguyen et al. (2021) "GraphDTA: predicting drug-target binding
  affinity with graph neural networks."  Bioinformatics, 37(8),
  1140-1147.

Architecture:
  Drug branch:   SMILES → molecular graph → GCN layers → global mean-pool
  Target branch: Protein seq → embedding → 3×1D-Conv → max-pool
  Fusion:        concat → 3×FC → scalar affinity

This implementation uses a pure-PyTorch GCN (no dependency on
PyTorch Geometric) so it runs on any standard PyTorch install.
Graph operations (scatter-add aggregation + mean pooling) are
implemented manually for portability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional layer (Kipf & Welling 2017).

      H' = D^{-1/2} (A + I) D^{-1/2} H W

    Implemented without PyTorch Geometric for portability.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, batch_vec=None):
        """
        Parameters
        ----------
        x          : FloatTensor [N, in_dim]   node features
        edge_index : LongTensor [2, E]          edge list
        batch_vec  : LongTensor [N]             batch assignment (unused here)

        Returns
        -------
        FloatTensor [N, out_dim]
        """
        N = x.size(0)

        # Add self-loops
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).repeat(2, 1)
        all_edges = torch.cat([edge_index, self_loops], dim=1)

        # Compute normalised adjacency (D^{-1/2} A D^{-1/2})
        src, dst = all_edges[0], all_edges[1]
        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, torch.ones(src.size(0), device=x.device))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0

        # Aggregate: messages from neighbours
        agg = torch.zeros(N, x.size(1), device=x.device)
        msg = x[src] * deg_inv_sqrt[src].unsqueeze(-1) * deg_inv_sqrt[dst].unsqueeze(-1)
        agg.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

        # Transform
        out = self.linear(agg)
        return out


class GraphDTA(nn.Module):
    """GraphDTA model: GNN for drugs + CNN for targets."""

    def __init__(self, atom_feature_dim, prot_vocab_size,
                 gcn_dims=(32, 64, 128),
                 embed_dim=128, filter_num=32,
                 prot_max_len=1200,
                 fc_dims=(128, 128)):
        super().__init__()

        # ---- Drug (graph) branch ----
        gcn_layers = []
        in_dim = atom_feature_dim
        for out_dim in gcn_dims:
            gcn_layers.append(GraphConvLayer(in_dim, out_dim))
            in_dim = out_dim
        self.gcn = nn.ModuleList(gcn_layers)
        self.drug_dim = gcn_dims[-1]

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
        self.prot_dim = filter_num * 3

        # ---- Fusion ----
        fused_dim = self.drug_dim + self.prot_dim
        layers = []
        in_d = fused_dim
        for out_d in fc_dims:
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            in_d = out_d
        layers.append(nn.Linear(in_d, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x, edge_index, prot_enc, batch_vec):
        """
        Parameters
        ----------
        x          : FloatTensor [N, atom_feature_dim]   all node features in batch
        edge_index : LongTensor [2, E]                    all edges in batch
        prot_enc   : LongTensor [B, prot_max_len]
        batch_vec  : LongTensor [N]                       graph assignment per node

        Returns
        -------
        FloatTensor [B, 1]
        """
        # ---- Drug: GCN ----
        h = x
        for layer in self.gcn:
            h = F.relu(layer(h, edge_index, batch_vec))
        # Global mean pooling per graph
        B = prot_enc.size(0)
        drug_emb = torch.zeros(B, self.drug_dim, device=x.device)
        drug_emb.scatter_add_(0,
                              batch_vec.unsqueeze(1).expand(-1, self.drug_dim),
                              h)
        # Count nodes per graph for mean
        counts = torch.zeros(B, device=x.device)
        counts.scatter_add_(0, batch_vec, torch.ones_like(batch_vec, dtype=torch.float))
        drug_emb = drug_emb / counts.unsqueeze(1).clamp(min=1)

        # ---- Target: CNN ----
        p = self.prot_embed(prot_enc)
        p = p.permute(0, 2, 1)
        p = self.prot_conv(p)
        p = self.prot_pool(p).squeeze(-1)

        # ---- Fusion ----
        fused = torch.cat([drug_emb, p], dim=1)
        out = self.fc(fused)
        return out
