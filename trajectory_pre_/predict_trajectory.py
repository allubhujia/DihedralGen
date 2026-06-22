"""
predict_trajectory.py — Generate a dihedral trajectory with the trained regression model.

Loads the best TrajectorySeqModel and, for a given peptide, produces its predicted
300-frame phi/psi trajectory in a single forward pass (deterministic). To give the
downstream analysis a small ensemble (matching the multi-sample format produced by
the diffusion pipeline), it can emit several trajectories using test-time dropout.

Outputs (same format/filenames as trajectory_pdb_files/, written under
trajectory_pre_/predictions/{PEPTIDE}/ so existing diffusion outputs are NOT clobbered):
    {PEPTIDE}_ground_truth_dihedrals.npz  → phi/psi, shape [300]   (degrees)
    {PEPTIDE}_predicted_dihedrals.npz     → phi/psi, shape [S, 300] (degrees)
    {PEPTIDE}_ground_truth.pdb            → true 3D multi-MODEL PDB

Point trajectory_analysis(main)/dihedral_comparison.py at this folder (or copy the
files into trajectory_pdb_files/{PEPTIDE}/) to compare against ground truth.

USAGE:
    .venv/bin/python trajectory_pre_/predict_trajectory.py --peptide AC
    .venv/bin/python trajectory_pre_/predict_trajectory.py --peptide AC --samples 5
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict
from torch_geometric.nn import global_mean_pool

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoder import MolecularEncoder
from scripts.processing import load_molecule, parse_elements_from_pdb
from scripts.backbone_utils import parse_backbone_from_pdb, compute_phi_psi_from_positions

# Reuse the model definition from the training script
import importlib.util as ilu


def _load_model_class():
    spec = ilu.spec_from_file_location(
        "train_trajectory",
        os.path.join(os.path.dirname(__file__), "train_trajectory.py"),
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TrajectorySeqModel


def write_multiframe_pdb(filepath, positions, elements, edge_index=None):
    """Write a multi-MODEL PDB file from atomic positions [N_atoms, num_frames, 3] (nm)."""
    num_atoms, num_frames, _ = positions.shape
    pos_a = positions * 10.0  # nm → Å

    adj = defaultdict(set)
    if edge_index is not None:
        en = edge_index.numpy() if hasattr(edge_index, "numpy") else np.array(edge_index)
        for col in range(en.shape[1]):
            a, b = int(en[0, col]), int(en[1, col])
            if a < b:
                adj[a].add(b)

    with open(filepath, "w") as f:
        for frame in range(num_frames):
            f.write(f"MODEL     {frame + 1:4d}\n")
            for i in range(num_atoms):
                elem = elements[i] if i < len(elements) else "C"
                x, y, z = pos_a[i, frame, :]
                f.write(
                    f"ATOM  {i+1:5d}  {elem:<3s} MOL A   1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
                    f"           {elem:>2s}  \n"
                )
            for src in sorted(adj):
                line = f"CONECT{src+1:5d}"
                for dst in sorted(adj[src]):
                    line += f"{dst+1:5d}"
                f.write(line + "\n")
            f.write("ENDMDL\n")
        f.write("END\n")
    print(f"  ✓ PDB saved ({num_frames} frames, {num_atoms} atoms) → {filepath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--peptide", type=str, default="AC", help="Dipeptide ID (e.g. AA, AC)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Number of predicted trajectories (uses test-time dropout if > 1).")
    parser.add_argument("--num_frames", type=int, default=300)
    args = parser.parse_args()
    pid = args.peptide.upper()

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}\n  PREDICT TRAJECTORY (regression) — Peptide: {pid}\n{'='*60}")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # ── Locate data (val, test, train, then AD-3-style standalone dirs / unseen)
    npz_path = pdb_path = None
    for split in ["val", "test", "train", "unseen"]:
        sd_2aa = os.path.join(base_dir, "timewarp_data", "2AA-complete", split)
        sd_pid = os.path.join(base_dir, "timewarp_data", pid, split)
        for sd in [sd_2aa, sd_pid]:
            np_ = os.path.join(sd, f"{pid}-traj-arrays.npz")
            pd_ = os.path.join(sd, f"{pid}-traj-state0.pdb")
            if os.path.exists(np_):
                npz_path, pdb_path = np_, pd_
                print(f"  Found data in: {sd}")
                break
        if npz_path:
            break
    if npz_path is None:
        print(f"Error: could not find data for peptide {pid}."); return

    out_dir = os.path.join(base_dir, "trajectory_pre_", "predictions", pid)
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Saving outputs to: {out_dir}\n")

    # ── GNN encoder (frozen)
    gnn_encoder = MolecularEncoder(8, 64, 32).to(device)
    encoder_path = os.path.join(base_dir, "encoder.pt")
    gnn_encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    gnn_encoder.eval()

    graph = load_molecule(npz_path, pdb_path)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        node_z = gnn_encoder.encode_nodes(graph.x, graph.edge_index)
        mol_z = global_mean_pool(node_z, graph.batch)                      # [1, 64]

    # ── Ground-truth dihedrals
    num_frames = args.num_frames
    positions_nm = np.load(npz_path)["positions"]
    if positions_nm.shape[0] < num_frames:
        pad = np.repeat(positions_nm[-1:], num_frames - positions_nm.shape[0], axis=0)
        positions_nm = np.concatenate([positions_nm, pad], axis=0)
    else:
        positions_nm = positions_nm[:num_frames]

    backbone = parse_backbone_from_pdb(pdb_path)
    gt_phi_rad, gt_psi_rad = compute_phi_psi_from_positions(positions_nm, backbone)
    gt_phi_deg = np.degrees(gt_phi_rad)
    gt_psi_deg = np.degrees(gt_psi_rad)

    gt_npz_path = os.path.join(out_dir, f"{pid}_ground_truth_dihedrals.npz")
    np.savez(gt_npz_path, phi=gt_phi_deg, psi=gt_psi_deg)
    print(f"✓ GT dihedrals saved → {gt_npz_path}")

    elements = parse_elements_from_pdb(pdb_path)
    positions_for_pdb = positions_nm.transpose(1, 0, 2)                     # [N_atoms, num_frames, 3]
    gt_pdb_path = os.path.join(out_dir, f"{pid}_ground_truth.pdb")
    print(f"\n💾 Saving Ground Truth PDB...")
    write_multiframe_pdb(gt_pdb_path, positions_for_pdb, elements, graph.edge_index.cpu())

    # ── Load trained model
    TrajectorySeqModel = _load_model_class()
    model = TrajectorySeqModel(
        mol_embed_dim=64, d_model=128, nhead=4, num_layers=4,
        dim_feedforward=256, num_frames=num_frames, dropout=0.1,
    ).to(device)
    model_path = os.path.join(os.path.dirname(__file__), "best_trajectory.pt")
    if not os.path.exists(model_path):
        print(f"\nError: {model_path} not found. Run train_trajectory.py first."); return
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f"  ✓ Loaded model weights from {model_path}")

    # ── Predict trajectories
    S = max(1, args.samples)
    out_phi = np.zeros((S, num_frames))
    out_psi = np.zeros((S, num_frames))

    # samples==1 → deterministic (eval). samples>1 → keep dropout on for diversity.
    model.train(S > 1)
    print(f"\n💾 Generating {S} predicted trajector{'y' if S == 1 else 'ies'}...")
    with torch.no_grad():
        for i in range(S):
            torch.manual_seed(42 + i)
            pred = model(mol_z).squeeze(0).cpu().numpy()                   # [num_frames, 4]
            out_phi[i] = np.degrees(np.arctan2(pred[:, 0], pred[:, 1]))
            out_psi[i] = np.degrees(np.arctan2(pred[:, 2], pred[:, 3]))
            print(f"  Sample {i+1}/{S} generated.")

    pred_npz_path = os.path.join(out_dir, f"{pid}_predicted_dihedrals.npz")
    np.savez(pred_npz_path, phi=out_phi, psi=out_psi)
    print(f"\n✓ Predicted dihedrals saved → {pred_npz_path}")

    print(f"\n{'='*60}")
    print(f"  ✅ All outputs saved to: {out_dir}")
    print(f"     {pid}_ground_truth_dihedrals.npz  — GT phi/psi [300]")
    print(f"     {pid}_predicted_dihedrals.npz     — Pred phi/psi [{S}, 300]")
    print(f"     {pid}_ground_truth.pdb            — True 3D trajectory ({num_frames} frames)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
