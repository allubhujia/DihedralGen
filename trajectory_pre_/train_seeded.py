"""
train_seeded.py — train the seeded Stage-2 generator (Stage 2b).

Writes `trajectory_pre_/best_seeded.pt`. Leaves the original
`best_trajectory.pt` and train_trajectory.py completely untouched, so the
unconditional Stage 2 keeps working exactly as before.

WHAT IT LEARNS
--------------
p(next N frames of phi/psi | backbone state at a seed frame, molecule).

Each training example is a random (peptide, start index) pair: the seed is the
backbone state at that index, the target is the window of N frames beginning
there. Because the start index is drawn uniformly over the WHOLE trajectory,
the model sees all 9800 frames rather than the first 300 the original Stage 2
was limited to.

WHY WINDOWS ARE PRECOMPUTED AS SIN/COS
--------------------------------------
Dihedral extraction is pure geometry and does not change during training, so it
is done once per peptide up front. Angles are stored as (sin, cos) pairs rather
than degrees: it removes the wrap-around discontinuity at +/-180, which would
otherwise put an enormous artificial jump in the middle of the delta and ACF
loss terms every time a trajectory crossed the branch cut.

Run:
    python -m trajectory_pre_.train_seeded
    python -m trajectory_pre_.train_seeded --frames 100 --epochs 60
"""

import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import global_mean_pool

from models.encoder import MolecularEncoder
from scripts.backbone_utils import compute_phi_psi_from_positions, parse_backbone_from_pdb
from scripts.processing import load_molecule
from trajectory_pre_.seeded_model import SeededTrajectoryModel, seeded_loss

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_ROOT = os.path.join(ROOT, "timewarp_data", "2AA-complete")
CKPT_OUT = os.path.join(HERE, "best_seeded.pt")
CURVE_OUT = os.path.join(HERE, "seeded_loss_curve.png")


# ──────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────

def build_peptide_bank(split, max_peptides, max_frames, encoder, device):
    """Precompute (molecule embedding, sin/cos dihedral track) for each peptide.

    Returns a list of dicts: {"id", "z" [64], "track" [T, 4]}.
    """
    files = sorted(glob.glob(os.path.join(DATA_ROOT, split, "*-traj-arrays.npz")))
    bank = []

    for npz_path in files:
        if max_peptides is not None and len(bank) >= max_peptides:
            break
        pdb_path = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")
        if not os.path.exists(pdb_path):
            continue

        pid = os.path.basename(npz_path).split("-")[0]
        positions = np.load(npz_path)["positions"][:max_frames]

        backbone = parse_backbone_from_pdb(pdb_path)
        phi, psi = compute_phi_psi_from_positions(positions, backbone)
        if not np.any(phi) and not np.any(psi):
            continue                      # dihedral extraction failed for this topology

        track = np.stack([np.sin(phi), np.cos(phi),
                          np.sin(psi), np.cos(psi)], axis=-1).astype(np.float32)

        graph = load_molecule(npz_path, pdb_path)
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)
        graph = graph.to(device)
        with torch.no_grad():
            z = global_mean_pool(encoder.encode_nodes(graph.x, graph.edge_index),
                                 graph.batch).squeeze(0).cpu()

        bank.append({"id": pid, "z": z, "track": torch.from_numpy(track)})

    return bank


def sample_batch(bank, batch_size, n_frames, rng, device):
    """Draw random (peptide, start index) windows.

    Returns z [B, 64], seed [B, 4], target [B, n_frames, 4].
    """
    zs, seeds, targets = [], [], []
    for _ in range(batch_size):
        p = bank[rng.integers(len(bank))]
        T = p["track"].shape[0]
        start = int(rng.integers(0, max(1, T - n_frames)))
        window = p["track"][start:start + n_frames]
        if window.shape[0] < n_frames:            # near the tail, pad by repeat
            pad = window[-1:].repeat(n_frames - window.shape[0], 1)
            window = torch.cat([window, pad], dim=0)
        zs.append(p["z"])
        seeds.append(window[0])
        targets.append(window)

    return (torch.stack(zs).to(device),
            torch.stack(seeds).to(device),
            torch.stack(targets).to(device))


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Train the seeded Stage-2 generator.")
    ap.add_argument("--frames", type=int, default=100,
                    help="window length trained on; inference can ask for any "
                         "length up to --max-frames")
    ap.add_argument("--max-frames", type=int, default=512,
                    help="largest window this checkpoint will ever support")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-train", type=int, default=60, help="peptides")
    ap.add_argument("--max-val", type=int, default=12)
    ap.add_argument("--traj-frames", type=int, default=9800,
                    help="frames read per peptide (the original Stage 2 used 300)")
    ap.add_argument("--w-sw", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    ap.add_argument("--w-delta", type=float, default=0.5)
    ap.add_argument("--w-acf", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=os.cpu_count())
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("TRAIN SEEDED STAGE-2 GENERATOR")
    print("=" * 70)

    encoder = MolecularEncoder(8, 64, 32).to(device)
    encoder.load_state_dict(torch.load(os.path.join(ROOT, "encoder.pt"),
                                       map_location=device, weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    print("building peptide bank ...")
    train_bank = build_peptide_bank("train", args.max_train, args.traj_frames,
                                    encoder, device)
    val_bank = build_peptide_bank("val", args.max_val, args.traj_frames,
                                  encoder, device)
    print(f"  train: {len(train_bank)} peptides, "
          f"{sum(p['track'].shape[0] for p in train_bank):,} frames")
    print(f"  val  : {len(val_bank)} peptides, "
          f"{sum(p['track'].shape[0] for p in val_bank):,} frames")

    if not train_bank or not val_bank:
        print("no usable peptides found - is timewarp_data/2AA-complete populated?")
        return

    model = SeededTrajectoryModel(mol_dim=64, max_frames=args.max_frames).to(device)
    print(f"  model: {model.num_parameters():,} parameters, "
          f"window {args.frames}, max {args.max_frames}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.05)

    weights = dict(w_sw=args.w_sw, w_anchor=args.w_anchor,
                   w_delta=args.w_delta, w_acf=args.w_acf)
    history = {"train": [], "val": []}
    best = float("inf")

    print(f"{'epoch':>6}{'train':>10}{'val':>10}{'sw':>9}{'anchor':>9}"
          f"{'delta':>9}{'acf':>9}{'sec':>7}")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            z, seed, target = sample_batch(train_bank, args.batch_size,
                                           args.frames, rng, device)
            loss, _ = seeded_loss(model(z, seed, args.frames), target, seed, **weights)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        train_loss = running / args.steps_per_epoch

        # Fixed-seed validation: the loss is stochastic in both the SW
        # projections and the window draw, so without pinning the RNG the
        # epoch-to-epoch curve is mostly sampling noise.
        model.eval()
        vrng = np.random.default_rng(12345)
        torch.manual_seed(12345)
        vtot, vparts, nb = 0.0, {}, 20
        with torch.no_grad():
            for _ in range(nb):
                z, seed, target = sample_batch(val_bank, args.batch_size,
                                               args.frames, vrng, device)
                loss, parts = seeded_loss(model(z, seed, args.frames), target,
                                          seed, **weights)
                vtot += loss.item()
                for k, v in parts.items():
                    vparts[k] = vparts.get(k, 0.0) + v / nb
        val_loss = vtot / nb
        sched.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        flag = ""
        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {"mol_dim": 64, "d_model": 128, "nhead": 4,
                           "num_layers": 4, "dim_feedforward": 256,
                           "max_frames": args.max_frames, "noise_dim": 8},
                "train_frames": args.frames,
                "loss_weights": weights,
            }, CKPT_OUT)
            flag = "  *"

        print(f"{epoch:>6}{train_loss:>10.4f}{val_loss:>10.4f}"
              f"{vparts['sw']:>9.4f}{vparts['anchor']:>9.4f}"
              f"{vparts['delta']:>9.4f}{vparts['acf']:>9.4f}"
              f"{time.time() - t0:>7.1f}{flag}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(history["train"], label="train")
    ax.plot(history["val"], label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("seeded loss")
    ax.set_title("Stage 2b - seeded generator")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CURVE_OUT, dpi=140)

    print(f"\nbest checkpoint : {CKPT_OUT}  (val {best:.4f})")
    print(f"loss curve      : {CURVE_OUT}")
    print("\nNext:  python -m trajectory_pre_.predict_seeded "
          "--peptide AC --seed-frame 50 --target-frame 100")


if __name__ == "__main__":
    main()
