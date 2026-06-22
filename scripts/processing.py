"""
data_utils.py  –  Convert 2AA-complete dipeptide trajectories into PyG graphs.

Each molecular snapshot becomes a `torch_geometric.data.Data` object:
    - x         : [num_atoms, 8]   node features (3 position + 5 element one-hot)
    - edge_index : [2, num_edges]  bond connectivity — built ONCE per molecule,
                                   then reused for every frame (bonds don't change!)
    - pos        : [num_atoms, 3]  raw xyz positions at time t
    - peptide_id : string tag like "AA", "RR", etc.

Node features per atom (8-dim):
    [x, y, z,  H_onehot, C_onehot, N_onehot, O_onehot, S_onehot]
     ↑ 3 dims  ↑ 5-dim one-hot — one slot per element, no grouping

WHY NO VELOCITY?
    Velocities were removed because we are no longer predicting future positions
    (t → t+τ transition learning). Without a target to predict, velocities are
    just random noise that makes the model harder to train.

WHY NO y_pos?
    We removed the "target frame at t+τ" because we're switching to a static
    graph representation. Looking up future frames was slow and wasted memory.

WHY STATIC EDGES?
    Chemical bonds in a dipeptide do NOT break or form during the simulation.
    The molecule bends and vibrates, but Atom 0 is always bonded to Atom 1.
    Building edges from positions every frame caused "flickering" — a bond
    would disappear when atoms vibrated 0.01 nm past the cutoff. We now build
    edges once from the first frame and reuse them forever.
"""

import os
import glob
import numpy as np
import torch
from torch_geometric.data import Data
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Each element gets its own dedicated slot in the one-hot vector.
# Previously N, O, and S were all collapsed into one "heavy other" slot,
# which meant the model couldn't distinguish them. They are chemically very
# different: N donates H-bonds, O accepts them, S is large and soft.
ELEMENT_TO_IDX = {"H": 0, "C": 1, "N": 2, "O": 3, "S": 4}
NUM_ELEMENTS = len(ELEMENT_TO_IDX)  # 5

# Distance cutoff used ONLY for building the initial bond graph from the
# first frame of the trajectory. After that, the same edge_index is reused.
BOND_CUTOFF_NM = 0.2  # nanometers — covers all typical covalent bond lengths


# ──────────────────────────────────────────────────────────────
# PDB parsing helpers
# ──────────────────────────────────────────────────────────────

def parse_elements_from_pdb(pdb_path: str) -> List[str]:
    """Extract the element symbol for every ATOM line in a PDB file.

    The element symbol lives in columns 77-78 of each ATOM record.
    If that field is blank (some older PDB files), we fall back to
    reading the first character of the atom-name field (cols 13-16).

    line[76:78] is the official, reliable field for element symbols in PDB format, while line[12:16] is only a fallback guess.
    """
    elements = []
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                elem = line[76:78].strip()
                if elem == "":
                    # Fallback: first character of the atom name
                    atom_name = line[12:16].strip()
                    elem = atom_name[0]
                elements.append(elem)
    return elements


# ──────────────────────────────────────────────────────────────
# Graph construction
# ──────────────────────────────────────────────────────────────

def build_edge_index_from_positions(positions: np.ndarray,
                                    cutoff: float = BOND_CUTOFF_NM) -> torch.Tensor:
    """Build an undirected edge list using a distance cutoff.

    This function is called ONCE per molecule (using the first trajectory
    frame) and the resulting edge_index is reused for every subsequent frame.
    Calling it per-frame would cause bonds to appear/disappear as atoms
    vibrate across the cutoff boundary — "flickering topology."

    Undirected means every bond A-B appears TWICE:
        A → B  (so B can send messages to A)
        B → A  (so A can send messages to B)
    This is how chemical bonds actually work — information flows both ways.

    Args:
        positions: [num_atoms, 3] atom coordinates in nm (first frame only)
        cutoff:    distance threshold in nm for declaring two atoms bonded

    Returns:
        edge_index: [2, num_edges] LongTensor, already undirected
    """
    num_atoms = positions.shape[0]

    # Compute all pairwise distances at once using broadcasting
    diff = positions[:, None, :] - positions[None, :, :]  # [N, N, 3]
    dist = np.linalg.norm(diff, axis=-1)                  # [N, N]

    # Keep pairs within cutoff; exclude self-loops (dist > 0)
    # np.where returns row indices and column indices of True entries,
    # which correspond to (source atom, destination atom) pairs.
    src, dst = np.where((dist < cutoff) & (dist > 0))

    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)
    return edge_index


def build_node_features(positions: np.ndarray,
                        elements: List[str]) -> torch.Tensor:
    """Construct the node feature matrix for one snapshot.

    Feature vector per atom [8-dim]:
        [x, y, z,  is_H, is_C, is_N, is_O, is_S]
         ↑ 3 dims  ↑ 5-dim full one-hot (one slot per element)

    CHANGES FROM ORIGINAL:
      - Removed velocities (vx, vy, vz): not needed without transition learning.
      - Expanded element encoding from 3-dim to 5-dim: N, O, and S each get
        their own slot instead of sharing one "heavy other" bucket.

    Args:
        positions: [num_atoms, 3] atom coordinates at this frame (nm)
        elements:  list of element symbols, one per atom (e.g. ['H','C','N',...])

    Returns:
        features: [num_atoms, 8] float32 tensor
    """
    num_atoms = positions.shape[0]

    # --- Position features (3 dims) ---
    pos_feat = torch.tensor(positions, dtype=torch.float32)   # [N, 3]

    # --- Element one-hot (5 dims, one slot per element) ---
    # Each atom gets a 1 in exactly one column and 0s everywhere else.
    # Unknown elements default to column 0 (treated as Hydrogen).
    elem_feat = torch.zeros(num_atoms, NUM_ELEMENTS, dtype=torch.float32)
    for i, elem in enumerate(elements):
        idx = ELEMENT_TO_IDX.get(elem, 0)   # default to H if unknown
        elem_feat[i, idx] = 1.0

    # Concatenate → [N, 3+5] = [N, 8]
    features = torch.cat([pos_feat, elem_feat], dim=1)
    return features


def snapshot_to_graph(positions_t: np.ndarray,
                      elements: List[str],
                      edge_index: torch.Tensor,
                      peptide_id: str = "") -> Data:
    """Convert a single MD snapshot into a PyG Data object.

    The edge_index is passed IN from outside — it was built once from the
    first frame and is shared across all frames of this trajectory. This is
    the key fix for the "flickering topology" problem.

    Args:
        positions_t: [num_atoms, 3]  atom positions at time t (nm)
        elements:    list of element symbols per atom
        edge_index:  [2, num_edges]  precomputed bond graph (static!)
        peptide_id:  string identifier for this dipeptide (e.g. "AA")

    Returns:
        PyG Data object with fields:
            .x          [num_atoms, 8]  node features
            .edge_index [2, num_edges]  bond connectivity (same every frame)
            .pos        [num_atoms, 3]  raw positions (useful for equivariant models)
            .peptide_id string
    """
    x = build_node_features(positions_t, elements)

    data = Data(
        x=x,
        edge_index=edge_index,
        pos=torch.tensor(positions_t, dtype=torch.float32),
    )
    data.peptide_id = peptide_id
    return data


# ──────────────────────────────────────────────────────────────
# Trajectory loading
# ──────────────────────────────────────────────────────────────

def load_molecule(npz_path: str,
                  pdb_path: str) -> Data:
    """Load one dipeptide and convert the first frame to a PyG graph.

    KEY CHANGES FROM ORIGINAL:
      - We no longer process time frames. We only extract the first frame (state0).

    Args:
        npz_path:   path to *-traj-arrays.npz  (contains positions, velocities)
        pdb_path:   path to *-traj-state0.pdb  (used to read element symbols)

    Returns:
        A Data object representing the molecule.
    """
    # --- Read element symbols from the PDB topology file ---
    elements = parse_elements_from_pdb(pdb_path)

    # --- Extract peptide ID from filename ---
    basename = os.path.basename(npz_path)
    peptide_id = basename.split("-")[0]

    # --- Load trajectory arrays ---
    arrays = np.load(npz_path)
    positions = arrays["positions"]   # [T, num_atoms, 3]

    # --- Build the bond graph ONCE from the very first frame ---
    static_edge_index = build_edge_index_from_positions(positions[0])

    # --- Create a single graph from the first frame ---
    graph = snapshot_to_graph(
        positions_t=positions[0],
        elements=elements,
        edge_index=static_edge_index,
        peptide_id=peptide_id,
    )

    return graph


