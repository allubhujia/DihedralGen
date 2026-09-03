"""
predict_seeded.py — generate from a seed frame with the Stage-2b model.

The question this answers, which the original Stage 2 could not be asked:

    "Given peptide AC as it was at frame 50, what does frame 100 look like?"

    python -m trajectory_pre_.predict_seeded --peptide AC --seed-frame 50 --target-frame 100

Internally that is one window: seed at frame 50, emit 51 frames, read index 50.
Asking for the whole window instead is just as valid and usually more useful,
since the intermediate frames are the trajectory:

    python -m trajectory_pre_.predict_seeded --peptide AC --seed-frame 50 --frames 100

Outputs go to trajectory_pre_/seeded_predictions/{PEPTIDE}/.

Because the model is stochastic, `--samples N` draws N independent futures from
the same seed. Their spread IS the prediction: a single sample is one draw from
p(future | seed), not a deterministic answer, and reporting only one hides the
uncertainty the model is actually expressing.
"""

import argparse
import os

import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from models.encoder import MolecularEncoder
from scripts.backbone_utils import compute_phi_psi_from_positions, parse_backbone_from_pdb
from scripts.processing import load_molecule
from trajectory_pre_.seeded_model import SeededTrajectoryModel, sincos_to_radians

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(ROOT, "timewarp_data", "2AA-complete")
CKPT = os.path.join(HERE, "best_seeded.pt")
OUT_ROOT = os.path.join(HERE, "seeded_predictions")


def find_peptide(pid):
    for split in ("test", "val", "train"):
        npz = os.path.join(DATA_ROOT, split, f"{pid}-traj-arrays.npz")
        pdb = os.path.join(DATA_ROOT, split, f"{pid}-traj-state0.pdb")
        if os.path.exists(npz) and os.path.exists(pdb):
            return npz, pdb, split
    raise FileNotFoundError(f"no trajectory found for peptide {pid}")


def load_seeded_model(device):
    if not os.path.exists(CKPT):
        raise FileNotFoundError(
            f"{CKPT} not found. Train it first:\n"
            "    python -m trajectory_pre_.train_seeded")
    ckpt = torch.load(CKPT, map_location=device)
    model = SeededTrajectoryModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def main():
    ap = argparse.ArgumentParser(description="Seeded Stage-2 generation.")
    ap.add_argument("--peptide", required=True, help="e.g. AC")
    ap.add_argument("--seed-frame", type=int, required=True,
                    help="real MD frame to condition on, e.g. 50")
    ap.add_argument("--target-frame", type=int, default=None,
                    help="absolute frame to report, e.g. 100. Sets the window "
                         "length to (target - seed + 1).")
    ap.add_argument("--frames", type=int, default=None,
                    help="window length instead of --target-frame")
    ap.add_argument("--samples", type=int, default=8,
                    help="independent futures drawn from the same seed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    pid = args.peptide.upper()
    device = torch.device(args.device)

    if args.target_frame is not None:
        if args.target_frame <= args.seed_frame:
            raise ValueError("--target-frame must be after --seed-frame")
        n_frames = args.target_frame - args.seed_frame + 1
    elif args.frames is not None:
        n_frames = args.frames
    else:
        n_frames = 100

    npz_path, pdb_path, split = find_peptide(pid)
    positions = np.load(npz_path)["positions"]
    if args.seed_frame >= positions.shape[0]:
        raise ValueError(f"seed frame {args.seed_frame} is past the end of the "
                         f"trajectory ({positions.shape[0]} frames)")

    backbone = parse_backbone_from_pdb(pdb_path)
    phi_all, psi_all = compute_phi_psi_from_positions(positions, backbone)

    # --- conditioning -----------------------------------------------------
    encoder = MolecularEncoder(8, 64, 32).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ROOT, "encoder.pt"),
                                       map_location=device, weights_only=True))
    encoder.eval()
    graph = load_molecule(npz_path, pdb_path)
    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
    graph = graph.to(device)
    with torch.no_grad():
        mol_z = global_mean_pool(encoder.encode_nodes(graph.x, graph.edge_index),
                                 graph.batch)                       # [1, 64]

    s = args.seed_frame
    seed_state = torch.tensor(
        [np.sin(phi_all[s]), np.cos(phi_all[s]),
         np.sin(psi_all[s]), np.cos(psi_all[s])],
        dtype=torch.float32, device=device).unsqueeze(0)            # [1, 4]

    model, ckpt = load_seeded_model(device)

    print("=" * 66)
    print(f"SEEDED STAGE-2 PREDICTION - {pid}")
    print("=" * 66)
    print(f"checkpoint : epoch {ckpt['epoch']}, val {ckpt['val_loss']:.4f}")
    print(f"peptide    : {pid}  (split={split}, {positions.shape[0]} MD frames)")
    print(f"seed frame : {s}   phi={np.degrees(phi_all[s]):+7.2f}  "
          f"psi={np.degrees(psi_all[s]):+7.2f}")
    print(f"window     : {n_frames} frames  ({args.samples} independent samples)\n")

    # --- generate ---------------------------------------------------------
    torch.manual_seed(args.seed)
    with torch.no_grad():
        z_b = mol_z.expand(args.samples, -1)
        seed_b = seed_state.expand(args.samples, -1)
        out = model(z_b, seed_b, n_frames)                          # [S, F, 4]
    phi_p, psi_p = sincos_to_radians(out)
    phi_p = np.degrees(phi_p.cpu().numpy())
    psi_p = np.degrees(psi_p.cpu().numpy())

    # --- ground truth for the same window ---------------------------------
    end = min(s + n_frames, positions.shape[0])
    gt_phi = np.degrees(phi_all[s:end])
    gt_psi = np.degrees(psi_all[s:end])

    out_dir = os.path.join(OUT_ROOT, pid)
    os.makedirs(out_dir, exist_ok=True)
    tag = f"seed{s}_n{n_frames}"
    np.savez_compressed(os.path.join(out_dir, f"{pid}_{tag}.npz"),
                        pred_phi=phi_p, pred_psi=psi_p,
                        gt_phi=gt_phi, gt_psi=gt_psi,
                        seed_frame=s, n_frames=n_frames)

    # --- report -----------------------------------------------------------
    def circ_mean_deg(a):
        r = np.radians(a)
        return float(np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())))

    def circ_spread_deg(a):
        r = np.radians(a)
        R = np.hypot(np.sin(r).mean(), np.cos(r).mean())
        return float(np.degrees(np.sqrt(max(0.0, -2.0 * np.log(max(R, 1e-12))))))

    print(f"anchor check (frame {s}, should reproduce the seed):")
    print(f"   predicted  phi={circ_mean_deg(phi_p[:, 0]):+7.2f}  "
          f"psi={circ_mean_deg(psi_p[:, 0]):+7.2f}")
    print(f"   actual     phi={np.degrees(phi_all[s]):+7.2f}  "
          f"psi={np.degrees(psi_all[s]):+7.2f}")

    if args.target_frame is not None:
        k = n_frames - 1
        print(f"\nprediction for frame {args.target_frame} "
              f"({args.samples} samples):")
        print(f"   phi = {circ_mean_deg(phi_p[:, k]):+7.2f} "
              f"+/- {circ_spread_deg(phi_p[:, k]):5.2f} deg")
        print(f"   psi = {circ_mean_deg(psi_p[:, k]):+7.2f} "
              f"+/- {circ_spread_deg(psi_p[:, k]):5.2f} deg")
        if args.target_frame < positions.shape[0]:
            print(f"   actual MD:  phi = {np.degrees(phi_all[args.target_frame]):+7.2f}"
                  f"   psi = {np.degrees(psi_all[args.target_frame]):+7.2f}")
            print("   (MD is chaotic - a single frame this far out is a sample "
                  "from a distribution,\n    not a deterministic target. Read the "
                  "spread, not just the centre.)")

    print(f"\nwritten to {os.path.join(out_dir, pid + '_' + tag + '.npz')}")


if __name__ == "__main__":
    main()
