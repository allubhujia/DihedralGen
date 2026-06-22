import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from models.gnn import MolecularGNN


# =============================================================================
# ENCODER
# =============================================================================

class MolecularEncoder(nn.Module):
    """
    Encodes a molecular/peptide graph into a fixed-size latent embedding.
    
    Pipeline: node features → MolecularGNN → global_mean_pool → MLP projection → z
    """

    def __init__(self, in_channels, hidden_channels, embedding_dim, num_gnn_layers=3):
        super(MolecularEncoder, self).__init__()
        self.gnn = MolecularGNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_gnn_layers,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, embedding_dim),
        )

    def forward(self, x, edge_index, batch=None):
        # Node-level embeddings from GNN
        node_embeddings = self.gnn(x, edge_index)
        # Aggregate to graph-level
        graph_embedding = global_mean_pool(node_embeddings, batch)
        # Project to latent space
        z = self.projection(graph_embedding)
        return z

    def encode_nodes(self, x, edge_index):
        """Return node-level embeddings (useful for node-level adjacency reconstruction)."""
        return self.gnn(x, edge_index)

    def encode_nodes_projected(self, x, edge_index):
        """Return node-level embeddings projected through the MLP projection head.
        
        Unlike forward(), this does NOT do global_mean_pool — it applies the
        projection head to each node individually.
        Shape: [N_total, embedding_dim]  (e.g. [N, 32])
        """
        node_embeddings = self.gnn(x, edge_index)        # [N, hidden_channels=64]
        projected = self.projection(node_embeddings)      # [N, embedding_dim=32]
        return projected