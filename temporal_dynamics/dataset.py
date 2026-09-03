"""
dataset.py — serve cached transitions as flat, batched graph tensors.

Peptides have different atom counts (23-33), so batching uses the PyG
convention: concatenate all graphs into one flat node list and carry a `batch`
vector saying which graph each node belongs to. Edge indices are offset
accordingly.

Edges are FULLY CONNECTED within each molecule. With <=33 atoms that is at most
~1100 edges per graph, which is cheap, and it lets non-bonded atoms exchange
messages — necessary for steric/H-bond effects that a bond-only graph cannot
see. Whether an edge is a covalent bond is passed as a binary edge feature, so
the model still knows the chemical topology.
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset

from temporal_dynamics import config


def _fully_connected(n: int):
    """[2, n*(n-1)] directed edge list over n nodes, self-loops excluded."""
    idx = torch.arange(n)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    mask = src != dst
    return torch.stack([src[mask], dst[mask]])


class TransitionDataset(Dataset):
    """One item = one (state_t -> target) transition for one peptide.

    Item fields:
        pos_t     [N, 3]   centred positions                        input
        vel_t     [N, 3]   velocities, COM removed                  input
        dpos      [N, 3]   aligned displacement to t+1              target
        vel_tp1   [N, 3]   velocity at t+1 in t's frame             target
        atom_type [N]      element index                            input
        z         [32]     frozen Stage-1 embedding                 conditioning
        edge_index[2, E]   fully-connected edges
        bond_flag [E]      1.0 if the pair is covalently bonded
    """

    def __init__(self, split: str, peptides=None, max_frames=None):
        stats = np.load(config.STATS_CACHE, allow_pickle=True)
        if peptides is None:
            peptides = [str(p) for p in stats[split]]
        if not peptides:
            raise ValueError(f"No peptides cached for split '{split}'. Run prepare_data.py.")

        embeddings = torch.load(config.EMBEDDING_CACHE, map_location="cpu")

        self.peptides = []
        self.index = []          # (peptide slot, frame) pairs

        for pid in peptides:
            path = os.path.join(config.TRANSITION_DIR, f"{pid}.npz")
            if not os.path.exists(path) or pid not in embeddings:
                continue

            with np.load(path, allow_pickle=True) as d:
                n_trans = d["pos_t"].shape[0]
                if max_frames is not None:
                    n_trans = min(n_trans, max_frames)
                n_atoms = d["pos_t"].shape[1]

                bonds = torch.from_numpy(d["bonds"])
                edge_index = _fully_connected(n_atoms)

                # Mark which of the fully-connected edges are real covalent bonds.
                key = edge_index[0] * n_atoms + edge_index[1]
                bond_key = bonds[0] * n_atoms + bonds[1]
                bond_flag = torch.isin(key, bond_key).float()

                entry = {
                    "id": pid,
                    "pos_t":     torch.from_numpy(d["pos_t"][:n_trans]),
                    "dpos":      torch.from_numpy(d["dpos"][:n_trans]),
                    "atom_type": torch.from_numpy(d["atom_type"]),
                    "z":         embeddings[pid].float(),
                    "edge_index": edge_index,
                    "bond_flag": bond_flag,
                }
                # Velocity is only in the cache when config.PREDICT_VELOCITY or
                # USE_VELOCITY_INPUT is on, so serve it only if it is there.
                for vkey in ("vel_t", "vel_tp1"):
                    if vkey in d.files:
                        entry[vkey] = torch.from_numpy(d[vkey][:n_trans])
                self.peptides.append(entry)

            slot = len(self.peptides) - 1
            self.index.extend((slot, t) for t in range(n_trans))

        if not self.peptides:
            raise ValueError(f"Split '{split}' loaded 0 peptides — did adapter.py run?")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        slot, t = self.index[i]
        p = self.peptides[slot]
        item = {"pos_t": p["pos_t"][t], "dpos": p["dpos"][t]}
        for key in ("vel_t", "vel_tp1"):
            if key in p:
                item[key] = p[key][t]
        item.update({
            "atom_type":  p["atom_type"],
            "z":          p["z"],
            "edge_index": p["edge_index"],
            "bond_flag":  p["bond_flag"],
        })
        return item


# Fields concatenated along the node axis; the rest are handled specially.
_NODE_FIELDS = ("pos_t", "dpos", "vel_t", "vel_tp1", "atom_type")


def collate(items):
    """Concatenate a list of graphs into one flat batch."""
    first = items[0]
    node_fields = [f for f in _NODE_FIELDS if f in first]

    out = {f: [] for f in node_fields}
    zs, edges, bflags, batch_vec = [], [], [], []
    offset = 0

    for g, item in enumerate(items):
        n = item["pos_t"].shape[0]
        for f in node_fields:
            out[f].append(item[f])
        zs.append(item["z"])
        edges.append(item["edge_index"] + offset)
        bflags.append(item["bond_flag"])
        batch_vec.append(torch.full((n,), g, dtype=torch.long))
        offset += n

    batch = {f: torch.cat(v) for f, v in out.items()}
    batch.update({
        "z":          torch.stack(zs),
        "edge_index": torch.cat(edges, dim=1),
        "bond_flag":  torch.cat(bflags),
        "batch":      torch.cat(batch_vec),
        "num_graphs": len(items),
    })
    return batch


def make_loader(split, batch_size=None, shuffle=True, max_frames=None, peptides=None):
    """DataLoader over a cached split. num_workers=0 — the data is already in RAM."""
    ds = TransitionDataset(split, peptides=peptides, max_frames=max_frames)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size or config.BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=0,
        drop_last=shuffle,
    )
    return ds, loader
