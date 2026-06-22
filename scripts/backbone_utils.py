"""
backbone_utils.py — Backbone dihedral angle computation utilities.

Provides functions to parse PDB backbone atoms and compute phi/psi dihedral
angles from MD trajectory position arrays. Used by both training
(train_diffusion.py) and prediction (predict_diffusion.py) scripts.

Handles two cases:
  1. Standard 2AA dipeptides: 2 residues, each with full N/CA/C backbone
  2. Capped peptides (AD-3 / Ace-Ala-Nme): 3 residues where cap residues
     have incomplete backbone (ACE has only C, NME has only N)

USAGE:
    from scripts.backbone_utils import parse_backbone_from_pdb, compute_phi_psi_from_positions
"""

import numpy as np


# ──────────────────────────────────────────────────────────────
# Dihedral angle computation
# ──────────────────────────────────────────────────────────────

def compute_dihedral(p0, p1, p2, p3):
    """Compute dihedral angle (radians) for 4 points.

    Each point can be shaped [3] (single frame) or [T, 3] (trajectory).
    Returns scalar or [T] array of dihedral angles in radians.
    """
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / np.linalg.norm(n1, axis=-1, keepdims=True).clip(min=1e-10)
    n2 = n2 / np.linalg.norm(n2, axis=-1, keepdims=True).clip(min=1e-10)
    b2_hat = b2 / np.linalg.norm(b2, axis=-1, keepdims=True).clip(min=1e-10)
    m1 = np.cross(n1, b2_hat)
    # Negate to match standard IUPAC right-handed convention
    return -np.arctan2(np.sum(m1 * n2, axis=-1), np.sum(n1 * n2, axis=-1))


# ──────────────────────────────────────────────────────────────
# PDB backbone parsing
# ──────────────────────────────────────────────────────────────

def parse_backbone_from_pdb(pdb_path):
    """Parse backbone N, CA, C atom indices (0-indexed) per residue from a PDB file.

    Only reads lines starting with 'ATOM'. For each residue, extracts the
    0-indexed atom index for backbone atoms N, CA, and C.

    Returns:
        dict: {residue_number: {"N": idx, "CA": idx, "C": idx}}
              Residues may have incomplete backbone (e.g., ACE only has C).
    """
    residues = {}
    with open(pdb_path, "r") as f:
        for line in f:
            # Stop after the first MODEL block if multi-model PDB
            if line.startswith("MODEL") or line.startswith("ENDMDL"):
                if residues:
                    break
                continue
            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            res_num = int(line[22:26].strip())
            atom_idx = int(line[6:11].strip()) - 1  # 0-indexed

            if atom_name in ("N", "CA", "C"):
                if res_num not in residues:
                    residues[res_num] = {}
                residues[res_num][atom_name] = atom_idx

    return residues


# ──────────────────────────────────────────────────────────────
# Phi/Psi from trajectory positions
# ──────────────────────────────────────────────────────────────

def compute_phi_psi_from_positions(positions, backbone):
    """Compute phi and psi dihedral angles from trajectory positions.

    Args:
        positions: numpy array [T, N_atoms, 3] — atom positions per frame
        backbone:  dict from parse_backbone_from_pdb()

    Returns:
        phi, psi: numpy arrays [T] of dihedral angles in radians

    Handles two cases:
      1. Standard 2AA dipeptide (2 residues with full N/CA/C):
           ψ₁ = dihedral(N₁, CA₁, C₁, N₂)
           φ₂ = dihedral(C₁, N₂, CA₂, C₂)

      2. Capped peptide like AD-3 (3 residues: ACE + ALA + NME):
           ACE (res 1): only has C
           ALA (res 2): has N, CA, C  (full backbone)
           NME (res 3): only has N
           φ = dihedral(C_ACE, N_ALA, CA_ALA, C_ALA)
           ψ = dihedral(N_ALA, CA_ALA, C_ALA, N_NME)
    """
    res_nums = sorted(backbone.keys())
    T = positions.shape[0]

    if len(res_nums) < 2:
        print(f"  ⚠ Only {len(res_nums)} residue(s) found — cannot compute dihedrals.")
        return np.zeros(T), np.zeros(T)

    # ── Case 1: Standard 2AA dipeptide (2 residues, both with full backbone)
    if len(res_nums) == 2:
        r1, r2 = backbone[res_nums[0]], backbone[res_nums[1]]
        if not all(k in r1 for k in ["N", "CA", "C"]):
            print(f"  ⚠ Residue {res_nums[0]} missing backbone atoms: {r1}")
            return np.zeros(T), np.zeros(T)
        if not all(k in r2 for k in ["N", "CA", "C"]):
            print(f"  ⚠ Residue {res_nums[1]} missing backbone atoms: {r2}")
            return np.zeros(T), np.zeros(T)

        # ψ₁ = dihedral(N₁, CA₁, C₁, N₂)
        psi = compute_dihedral(
            positions[:, r1["N"]], positions[:, r1["CA"]],
            positions[:, r1["C"]], positions[:, r2["N"]]
        )
        # φ₂ = dihedral(C₁, N₂, CA₂, C₂)
        phi = compute_dihedral(
            positions[:, r1["C"]], positions[:, r2["N"]],
            positions[:, r2["CA"]], positions[:, r2["C"]]
        )
        return phi, psi

    # ── Case 2: Capped peptide (3+ residues — look for ACE/ALA/NME pattern)
    # Find a residue with full backbone (N, CA, C) — this is the central amino acid
    central_res = None
    for rn in res_nums:
        r = backbone[rn]
        if all(k in r for k in ["N", "CA", "C"]):
            central_res = rn
            break

    if central_res is None:
        print(f"  ⚠ No residue with full backbone (N, CA, C) found.")
        return np.zeros(T), np.zeros(T)

    rc = backbone[central_res]  # Central residue (e.g., ALA)

    # Find the previous residue (should have C — e.g., ACE)
    prev_res = None
    for rn in res_nums:
        if rn < central_res and "C" in backbone[rn]:
            prev_res = rn

    # Find the next residue (should have N — e.g., NME)
    next_res = None
    for rn in res_nums:
        if rn > central_res and "N" in backbone[rn]:
            next_res = rn
            break

    if prev_res is None or next_res is None:
        print(f"  ⚠ Cannot find flanking residues for capped peptide.")
        print(f"    Central: {central_res}, Prev: {prev_res}, Next: {next_res}")
        return np.zeros(T), np.zeros(T)

    rp = backbone[prev_res]  # Previous residue (e.g., ACE — has C)
    rn_dict = backbone[next_res]  # Next residue (e.g., NME — has N)

    # φ = dihedral(C_prev, N_central, CA_central, C_central)
    phi = compute_dihedral(
        positions[:, rp["C"]], positions[:, rc["N"]],
        positions[:, rc["CA"]], positions[:, rc["C"]]
    )
    # ψ = dihedral(N_central, CA_central, C_central, N_next)
    psi = compute_dihedral(
        positions[:, rc["N"]], positions[:, rc["CA"]],
        positions[:, rc["C"]], positions[:, rn_dict["N"]]
    )
    return phi, psi
