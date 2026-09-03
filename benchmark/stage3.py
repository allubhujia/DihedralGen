"""
stage3.py — run the Stage-3 propagator and return phi/psi in degrees.

Thin wrapper over temporal_dynamics so the benchmark never re-implements the
rollout. Unlike Stage 2 this one IS seeded: the trajectory starts from a real MD
frame (`start_frame`) and every subsequent frame is sampled by the flow-matching
propagator, so the output has a genuine time axis.
"""

import numpy as np
import torch

from scripts.backbone_utils import compute_phi_psi_from_positions, parse_backbone_from_pdb
from temporal_dynamics import config as s3cfg
from temporal_dynamics import flow
from temporal_dynamics.rollout import autoregressive_rollout, build_state, load_propagator
from temporal_dynamics.velocity_utils import bond_length_rmsd


def generate(peptide, n_frames, start_frame=0, seed=0, flow_steps=None,
             temperature=None, bond_fix=None, device="cpu"):
    """Roll out a Stage-3 trajectory and extract its backbone dihedrals.

    Returns:
        phi_deg [n_frames], psi_deg [n_frames], info dict
    """
    device = torch.device(device)

    model, ckpt = load_propagator(device=device)
    scales = flow.load_scales(device)
    state = build_state(peptide, device)

    n_available = state["positions"].shape[0]
    if start_frame >= n_available:
        raise ValueError(
            f"start_frame {start_frame} is beyond the trajectory "
            f"({n_available} frames available for {peptide})."
        )

    traj, vels = autoregressive_rollout(
        model, state, scales,
        n_frames=n_frames, start_frame=start_frame,
        flow_steps=flow_steps, temperature=temperature,
        bond_fix=bond_fix, seed=seed, device=device,
    )

    backbone = parse_backbone_from_pdb(state["pdb_path"])
    phi, psi = compute_phi_psi_from_positions(traj, backbone)

    info = {
        "epoch": ckpt.get("epoch"),
        "val_loss": ckpt.get("val_loss"),
        "bond_rmsd_A": bond_length_rmsd(traj, state["bonds"], state["eq_len"]) * 10.0,
        "n_atoms": state["n_atoms"],
        "split": state["split"],
        "seeded": True,
        "start_frame": start_frame,
        "sim_time_ns": n_frames * s3cfg.FRAME_LAG_PS / 1000.0,
    }
    return np.degrees(phi), np.degrees(psi), info, traj, state


def ground_truth(state, n_frames, start_frame=0, gt_stride=5):
    """Reference dihedrals from the real MD trajectory.

    Returns two views:
      window — the same n_frames the models were asked to produce, starting at
               start_frame. Matched sample count, so it is the fair comparison
               for a seeded model.
      full   — the whole trajectory, subsampled. This is the equilibrium density
               and is the ONLY reference both stages can be scored against
               fairly, because Stage 2 has no start frame.
    """
    backbone = parse_backbone_from_pdb(state["pdb_path"])
    pos = state["positions"]

    end = min(start_frame + n_frames, pos.shape[0])
    win_phi, win_psi = compute_phi_psi_from_positions(pos[start_frame:end], backbone)
    full_phi, full_psi = compute_phi_psi_from_positions(pos[::gt_stride], backbone)

    return {
        "phi_window": np.degrees(win_phi), "psi_window": np.degrees(win_psi),
        "phi_full": np.degrees(full_phi), "psi_full": np.degrees(full_psi),
        "n_total_frames": int(pos.shape[0]),
    }
