import torch
import torch.nn as nn


# =============================================================================
# DECODERS
# =============================================================================

class InnerProductDecoder(nn.Module):
    """
    Reconstructs adjacency matrix via inner product.

    Returns RAW LOGITS (no sigmoid) — sigmoid is handled internally
    by F.binary_cross_entropy_with_logits for numerical stability.

        A_logits = z @ z^T

    Works at node-level: pass node embeddings, get node-node logits.
    This decoder assumes that if two nodes are connected, their embeddings should be similar (have a high dot product).
    sigmoid is not used here because the exp inside the sigmoid function will overflow, leading to NaN losses or numerical instability.
    """

    def forward(self, z):
        # No sigmoid — BCEWithLogits handles it internally
        # Clamp prevents logit explosion from large dot products (causes ~47 loss)
        adj_logits = torch.matmul(z, z.t())
        return torch.clamp(adj_logits, min=-10, max=10)



