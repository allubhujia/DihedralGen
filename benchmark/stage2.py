"""
stage2.py — run the Stage-2 generator and return phi/psi in degrees.

Stage 2 is a Transformer-over-frames trained with a Sliced-Wasserstein loss. It
maps a peptide's molecule embedding straight to a set of backbone angles; it has
no starting structure and no time axis.

TWO CONSTRAINTS THIS FILE HAS TO WORK AROUND
--------------------------------------------
1. FIXED LENGTH. The checkpoint's frame positional embedding is [1, 300, 128],
   so the network can only ever emit exactly 300 frames. `generate()` therefore
   produces 300 at a time and reshapes to the requested count:
     * n <= 300 -> evenly-spaced subsample of the 300. Taking the first n would
       also be defensible (the SW loss is permutation-invariant, so frame order
       carries no temporal meaning), but spreading the picks keeps the full
       distribution represented at small n.
     * n >  300 -> draw ceil(n/300) independent samples, each with its own noise
       vector, and concatenate.

2. NO SEED FRAME. Stage 3 can be started from a specific MD frame; Stage 2
   cannot be conditioned on one at all. `start_frame` is accepted and ignored
   here on purpose, and the benchmark reports that asymmetry rather than
   pretending both stages were given the same information.
"""

import importlib.util as ilu
import os

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from benchmark import bench_config as bcfg
from models.encoder import MolecularEncoder
from scripts.processing import load_molecule


def _load_model_class():
    """Import TrajectorySeqModel from train_trajectory.py without running it."""
    spec = ilu.spec_from_file_location("stage2_train_trajectory", bcfg.STAGE2_TRAIN_FILE)
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrajectorySeqModel


def molecule_embedding(npz_path, pdb_path, device="cpu"):
    """Stage-2 conditioning vector: pooled node embeddings [1, 64].

    Note this is `encode_nodes` + global_mean_pool, i.e. the 64-d hidden
    representation — NOT the 32-d projected z that Stage 3 conditions on.
    """
    encoder = MolecularEncoder(8, 64, 32).to(device)
    ckpt = (bcfg.STAGE2_ENCODER_FT if os.path.exists(bcfg.STAGE2_ENCODER_FT)
            else bcfg.ENCODER_CKPT)
    encoder.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    encoder.eval()

    graph = load_molecule(npz_path, pdb_path)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)

    with torch.no_grad():
        node_z = encoder.encode_nodes(graph.x, graph.edge_index)
        mol_z = global_mean_pool(node_z, graph.batch)      # [1, 64]

    return mol_z, os.path.basename(ckpt)


def load_generator(device="cpu"):
    """Rebuild TrajectorySeqModel at its native 300 frames and load the weights."""
    if not os.path.exists(bcfg.STAGE2_CKPT):
        raise FileNotFoundError(
            f"Stage-2 checkpoint not found at {bcfg.STAGE2_CKPT}. "
            "Train it with `python -m trajectory_pre_.train_trajectory` first."
        )
    TrajectorySeqModel = _load_model_class()
    model = TrajectorySeqModel(
        mol_embed_dim=bcfg.STAGE2_MOL_EMBED_DIM,
        d_model=bcfg.STAGE2_D_MODEL,
        nhead=bcfg.STAGE2_NHEAD,
        num_layers=bcfg.STAGE2_NUM_LAYERS,
        dim_feedforward=bcfg.STAGE2_FF,
        num_frames=bcfg.STAGE2_NATIVE_FRAMES,
        dropout=bcfg.STAGE2_DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(bcfg.STAGE2_CKPT, map_location=device,
                                     weights_only=True))
    model.eval()
    return model


@torch.no_grad()
def generate(npz_path, pdb_path, n_frames, seed=bcfg.SEED, device="cpu"):
    """Generate n_frames of Stage-2 phi/psi.

    Returns:
        phi_deg [n_frames], psi_deg [n_frames], info dict
    """
    device = torch.device(device)
    mol_z, enc_name = molecule_embedding(npz_path, pdb_path, device)
    model = load_generator(device)

    native = bcfg.STAGE2_NATIVE_FRAMES
    n_draws = max(1, int(np.ceil(n_frames / native)))

    chunks = []
    for i in range(n_draws):
        torch.manual_seed(seed + i)
        pred = model(mol_z).squeeze(0).cpu().numpy()       # [300, 4]
        chunks.append(pred)
    pred = np.concatenate(chunks, axis=0)                  # [300 * n_draws, 4]

    if n_frames <= native:
        # Spread the picks across the whole generated set rather than truncating,
        # so a small request still samples the full distribution.
        idx = np.linspace(0, native - 1, n_frames).astype(int)
        pred = pred[idx]
    else:
        pred = pred[:n_frames]

    phi = np.degrees(np.arctan2(pred[:, 0], pred[:, 1]))
    psi = np.degrees(np.arctan2(pred[:, 2], pred[:, 3]))

    info = {
        "encoder": enc_name,
        "native_frames": native,
        "draws": n_draws,
        "mode": "subsampled" if n_frames <= native else "tiled",
        "seeded": False,
    }
    return phi, psi, info
