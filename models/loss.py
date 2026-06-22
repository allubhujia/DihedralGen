import torch
import torch.nn.functional as F


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def reconstruction_loss(adj_logits, adj):
    """
    Binary cross-entropy between original and reconstructed adjacency.
    Handles class imbalance (sparse graphs have many more 0s than 1s).

    Args:
        adj_logits: Raw logits from decoder (before sigmoid) (N, N)
        adj:        Ground truth adjacency matrix (N, N) Here N is the number of atoms in the molecule
        Check how well your model reconstructs the graph (edges)
    """
    # Count edges vs non-edges
    num_nodes = adj.size(0)
    num_edges = adj.sum()
    num_non_edges = num_nodes * num_nodes - num_edges

    if num_edges > 0:
        pos_weight = (num_non_edges / num_edges).to(adj.device) # Fix imbalance between 0s and 1s
    else:
        pos_weight = torch.tensor(1.0, device=adj.device)

    # FIX: Actually use pos_weight + use logits version for numerical stability
    return F.binary_cross_entropy_with_logits(
        adj_logits, adj, pos_weight=pos_weight, reduction='mean'
    )


def cosine_similarity_loss(z):
    """
    Embedding similarity loss using cosine similarity.
    Target: identity matrix — each graph most similar to itself, dissimilar to others.

    NOTE: Requires batch size >= 2. Returns 0 if only 1 embedding is passed.
    Make graph embeddings meaningful
    “How well are graph embeddings separated?”
    """
    if z.size(0) < 2:
        return torch.tensor(0.0, device=z.device)

    z_norm = F.normalize(z, p=2, dim=1)
    sim_matrix = torch.matmul(z_norm, z_norm.t())          # (B, B)
    target = torch.eye(z.size(0), device=z.device)         # (B, B)
    return F.mse_loss(sim_matrix, target)


def combined_loss(adj_list, adj_logits_list, z, lambda_val=0.01):
    """
    Combined loss:
        L = mean(BCE_per_graph) + lambda * cosine_similarity_loss(z)

    Args:
        adj_list:        list of per-graph ground truth adjacency (n_i, n_i)
        adj_logits_list: list of per-graph raw logits from decoder (n_i, n_i)
        z:               FULL batch graph-level embeddings (B, d) — NOT z[i]
        lambda_val:      weight for similarity loss

    Returns:
        total_loss, rec_loss, sim_loss
    """
    rec_losses = []
    for adj_i, logits_i in zip(adj_list, adj_logits_list):
        rec_losses.append(reconstruction_loss(logits_i, adj_i))

    rec_loss = torch.stack(rec_losses).mean()
    sim_loss = cosine_similarity_loss(z)                   # z must be (B, d)
    total    = rec_loss + lambda_val * sim_loss # Reconstruction Loss :  how closely the output's individual pixels or data points match the
                                                # original input. Its job is to ensure the shape, borders, and fine details are sharp and in the right places.
    
                                                # Similarity Loss:  measures the difference between the output and the target in the embedding space.
    return total, rec_loss, sim_loss

 
    """
    By minimizing the Total Loss, the model learns to create embeddings that are unique
    (Similarity) AND meaningful enough to rebuild the molecule (Reconstruction).
    """