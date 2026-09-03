"""
adapter.py — STEP 2. Freeze Stage 1 and cache its graph embeddings.

The Stage-3 propagator is conditioned on z, the 32-d graph embedding that
Stage 1 already learned. z depends only on the static chemical graph, so it is
constant for a peptide across all 9800 frames — computing it once and caching it
saves a full GNN forward pass per training sample.

The encoder is loaded in eval mode with requires_grad=False and is never
updated: Stage 3 must not be able to reshape the Stage-1 latent space.

Writes {peptide_id: z[32]} to cache/embeddings.pt.

Run:
    python -m temporal_dynamics.adapter
"""

import os

import numpy as np
import torch

from models.encoder import MolecularEncoder
from scripts.processing import load_molecule
from temporal_dynamics import config
from temporal_dynamics.prepare_data import peptide_files


def load_frozen_encoder(device="cpu") -> MolecularEncoder:
    """Load encoder.pt in eval mode with gradients disabled."""
    encoder = MolecularEncoder(
        in_channels=config.ENC_IN_CHANNELS,
        hidden_channels=config.ENC_HIDDEN_CHANNELS,
        embedding_dim=config.ENC_EMBEDDING_DIM,
        num_gnn_layers=config.ENC_NUM_LAYERS,
    )
    state = torch.load(config.ENCODER_CKPT, map_location=device)
    encoder.load_state_dict(state)
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


@torch.no_grad()
def encode_graph(encoder: MolecularEncoder, graph, device="cpu") -> torch.Tensor:
    """Embed a single PyG graph -> z [32].

    global_mean_pool needs an explicit batch vector when there is only one
    graph; without it the pooled tensor comes back with the wrong shape.
    """
    x = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
    return encoder(x, edge_index, batch).squeeze(0).cpu()


def cache_all_embeddings(device="cpu"):
    """Embed every peptide present in the transition cache."""
    config.ensure_dirs()

    if not os.path.exists(config.ENCODER_CKPT):
        raise FileNotFoundError(
            f"Stage-1 checkpoint not found at {config.ENCODER_CKPT}. "
            "Run `python -m models.pipeline_auto` first."
        )

    cached = {os.path.splitext(f)[0] for f in os.listdir(config.TRANSITION_DIR)
              if f.endswith(".npz")}
    if not cached:
        raise FileNotFoundError(
            "No transition cache found. Run `python -m temporal_dynamics.prepare_data` first."
        )

    encoder = load_frozen_encoder(device)
    embeddings = {}

    for split in ("train", "val", "test"):
        for pid, npz, pdb in peptide_files(split):
            if pid not in cached or pid in embeddings:
                continue
            graph = load_molecule(npz, pdb)
            embeddings[pid] = encode_graph(encoder, graph, device)

    torch.save(embeddings, config.EMBEDDING_CACHE)
    return embeddings


def main():
    print("=" * 74)
    print("STEP 2 - CACHE FROZEN STAGE-1 EMBEDDINGS")
    print("=" * 74)
    print(f"encoder: {config.ENCODER_CKPT}")

    embeddings = cache_all_embeddings()

    Z = torch.stack(list(embeddings.values())).numpy()
    print(f"\ncached  : {len(embeddings)} peptides -> {config.EMBEDDING_CACHE}")
    print(f"z shape : [{Z.shape[1]}]")
    print(f"z stats : mean={Z.mean():+.4f}  std={Z.std():.4f}  "
          f"range=[{Z.min():+.3f}, {Z.max():+.3f}]")

    # A quick collapse check: if every peptide maps to nearly the same z, the
    # conditioning signal is useless and Stage 3 cannot tell peptides apart.
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)
    sim = Zn @ Zn.T
    off = sim[~np.eye(len(Z), dtype=bool)]
    print(f"pairwise cosine similarity: mean={off.mean():+.3f}  max={off.max():+.3f}")
    if off.mean() > 0.95:
        print("  [warn] embeddings are nearly identical across peptides - "
              "conditioning will be weak.")

    print("\nNext:  python -m temporal_dynamics.train")


if __name__ == "__main__":
    main()
