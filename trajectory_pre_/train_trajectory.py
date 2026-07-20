"""
train_trajectory.py — Distribution-matching sequence model for dihedral trajectories.

Goal: given a peptide's GNN embedding, generate a 300-frame backbone dihedral
(phi/psi) trajectory whose *distribution* on the Ramachandran plot matches the
ground-truth MD trajectory. This is what trajectory_analysis(main) evaluates
(overlaid Ramachandran + per-angle KL divergence).

Why distribution matching (and not per-frame regression):
    The conditioning signal is ONLY the molecule identity. Each peptide's 300-frame
    MD trajectory is stochastic — the exact frame ordering is not a function of the
    molecule. A per-frame MSE regressor therefore (a) collapses toward the mean
    sin/cos, which lies *inside* the unit circle, so atan2() returns near-random
    angles → a uniform scatter across the Ramachandran plane (the failure we saw),
    and (b) cannot reproduce the basin-hopping that defines the distribution.

Two design choices fix this:
    1. The model L2-normalizes each (sin, cos) pair, so every predicted frame lies
       exactly ON the unit circle → stable, well-defined angles (no scatter to the
       origin).
    2. The loss is a Sliced-Wasserstein distance in the 4-D (sinφ, cosφ, sinψ, cosψ)
       space between the predicted 300-point set and the GT 300-point set. This
       matches the *joint* φ/ψ distribution and is mode-covering (no mode collapse).

Trains on the `train` split, validates on `val`, saves the best model (lowest val
SW distance), a loss curve, and an every-5-epochs Ramachandran + histogram-KL check.

USAGE (from project root, venv activated):
    python -m trajectory_pre_.train_trajectory
"""

import os
import sys
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.processing import load_molecule
from models.encoder import MolecularEncoder
from scripts.backbone_utils import parse_backbone_from_pdb, compute_phi_psi_from_positions


# ──────────────────────────────────────────────────────────────
# 1. Data Loading (phi/psi as sin/cos, [num_frames, 4] per molecule)
# ──────────────────────────────────────────────────────────────

def load_split_with_dihedrals(data_dir, split="train", max_molecules=None, num_frames=300):
    """Load molecules and precompute sin/cos of phi/psi dihedral angles per frame."""
    split_dir = os.path.join(data_dir, split)
    npz_files = sorted(glob.glob(os.path.join(split_dir, "*-traj-arrays.npz")))
    if max_molecules is not None:
        npz_files = npz_files[:max_molecules]

    all_graphs, skipped = [], 0
    for npz_path in npz_files:
        pdb_path = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")
        if not os.path.exists(pdb_path):
            continue

        backbone = parse_backbone_from_pdb(pdb_path)
        res_nums = sorted(backbone.keys())
        if len(res_nums) < 2:
            skipped += 1
            continue
        r1, r2 = backbone[res_nums[0]], backbone[res_nums[1]]
        if not all(k in r1 for k in ["N", "CA", "C"]) or not all(k in r2 for k in ["N", "CA", "C"]):
            skipped += 1
            continue

        graph = load_molecule(npz_path, pdb_path)
        positions = np.load(npz_path)["positions"]

        if positions.shape[0] < num_frames:
            pad = np.repeat(positions[-1:], num_frames - positions.shape[0], axis=0)
            positions = np.concatenate([positions, pad], axis=0)
        else:
            positions = positions[:num_frames]

        phi, psi = compute_phi_psi_from_positions(positions, backbone)
        sincos = np.stack([np.sin(phi), np.cos(phi), np.sin(psi), np.cos(psi)], axis=-1)
        graph.y_dihedrals = torch.tensor(sincos, dtype=torch.float32)
        all_graphs.append(graph)

    print(f"✓ {split}: {len(all_graphs)} molecules with {num_frames}-frame trajectories "
          f"({skipped} skipped).")
    return all_graphs


# ──────────────────────────────────────────────────────────────
# 2. Sequence Model: mol_embed -> [num_frames, 4] on the unit circle
# ──────────────────────────────────────────────────────────────

def normalize_sincos(out, eps=1e-6):
    """L2-normalize each (sin, cos) pair so every frame lies on the unit circle.

    out: [..., 4] = (sinφ, cosφ, sinψ, cosψ). Returns the same shape, with
    (sinφ, cosφ) and (sinψ, cosψ) each having unit norm.
    """
    phi_pair = F.normalize(out[..., 0:2], dim=-1, eps=eps)
    psi_pair = F.normalize(out[..., 2:4], dim=-1, eps=eps)
    return torch.cat([phi_pair, psi_pair], dim=-1)


class TrajectorySeqModel(nn.Module):
    """Transformer-over-frames generator.

    Each of the `num_frames` output frames is a token formed from a learned frame
    positional embedding plus the projected molecule embedding. A Transformer lets
    frames attend to one another; a linear head maps each to 4 channels, which are
    then projected onto the unit circle so the angles are always well-defined.

    A small per-frame latent noise channel is injected so different samples (and
    different dropout draws) can produce diverse trajectories from the same peptide.
    """

    def __init__(self, mol_embed_dim=64, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, num_frames=300, dropout=0.1, noise_dim=8):
        super().__init__()
        self.num_frames = num_frames
        self.d_model = d_model
        self.noise_dim = noise_dim

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_embed_dim, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        self.noise_proj = nn.Linear(noise_dim, d_model) if noise_dim > 0 else None
        self.frame_pos = nn.Parameter(torch.randn(1, num_frames, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4),
        )

    def forward(self, mol_embed, noise=None):
        """
        mol_embed: [B, mol_embed_dim]
        noise:     [B, num_frames, noise_dim] or None (sampled if None and noise_dim>0)
        Returns:   [B, num_frames, 4] unit-circle (sinφ, cosφ, sinψ, cosψ)
        """
        B = mol_embed.size(0)
        m = self.mol_proj(mol_embed).unsqueeze(1)              # [B, 1, d_model]
        tokens = self.frame_pos.expand(B, -1, -1) + m          # [B, F, d_model]
        if self.noise_proj is not None:
            if noise is None:
                noise = torch.randn(B, self.num_frames, self.noise_dim, device=mol_embed.device)
            tokens = tokens + self.noise_proj(noise)
        h = self.transformer(tokens)                           # [B, F, d_model]
        return normalize_sincos(self.head(h))                  # [B, F, 4]


# ──────────────────────────────────────────────────────────────
# 3. Loss: Sliced-Wasserstein distance in 4-D sin/cos space
# ──────────────────────────────────────────────────────────────

def sliced_wasserstein_loss(pred, target, num_proj=64):
    """Sliced-Wasserstein-1 distance between two point sets per sample.

    pred, target: [B, num_frames, 4]
    Projects both 300-point sets onto `num_proj` random directions in R^4, sorts
    each projection, and averages the L1 difference. Matching all 1-D projections
    matches the full joint 4-D distribution (hence the joint φ/ψ distribution).
    """
    D = pred.size(-1)
    proj = torch.randn(D, num_proj, device=pred.device)
    proj = proj / proj.norm(dim=0, keepdim=True)
    p = pred @ proj                                            # [B, F, num_proj]
    t = target @ proj                                          # [B, F, num_proj]
    p_sorted, _ = torch.sort(p, dim=1)
    t_sorted, _ = torch.sort(t, dim=1)
    return (p_sorted - t_sorted).abs().mean()


# ──────────────────────────────────────────────────────────────
# 4. Diagnostics (histogram-KL + Ramachandran scatter)
# ──────────────────────────────────────────────────────────────

def sincos_to_deg(sincos):
    """sincos: [..., 4] numpy -> (phi_deg, psi_deg)."""
    phi = np.degrees(np.arctan2(sincos[..., 0], sincos[..., 1]))
    psi = np.degrees(np.arctan2(sincos[..., 2], sincos[..., 3]))
    return phi, psi


def histogram_kl(gt_deg, pred_deg, bins=36):
    """KL(GT || Pred) between angle histograms over [-180, 180] degrees."""
    edges = np.linspace(-180, 180, bins + 1)
    pg, _ = np.histogram(gt_deg, edges)
    pp, _ = np.histogram(pred_deg, edges)
    pg = pg + 1e-6; pp = pp + 1e-6
    pg /= pg.sum(); pp /= pp.sum()
    return float(np.sum(pg * np.log(pg / pp)))


def diagnostic(epoch, pred_sincos, gt_sincos, peptide_id, out_dir):
    """Print φ/ψ distribution stats + KL, and save a Ramachandran + histogram plot."""
    gt_phi, gt_psi = sincos_to_deg(gt_sincos)
    pr_phi, pr_psi = sincos_to_deg(pred_sincos)

    kl_phi = histogram_kl(gt_phi, pr_phi)
    kl_psi = histogram_kl(gt_psi, pr_psi)

    print(f"\n{'─'*78}")
    print(f"  Epoch {epoch:3d} — Val '{peptide_id}' distribution check")
    print(f"  {'':10} {'GT mean':>9} {'GT std':>8}  │  {'Pred mean':>9} {'Pred std':>8}")
    print(f"  φ (deg)   {gt_phi.mean():>9.1f} {gt_phi.std():>8.1f}  │  "
          f"{pr_phi.mean():>9.1f} {pr_phi.std():>8.1f}")
    print(f"  ψ (deg)   {gt_psi.mean():>9.1f} {gt_psi.std():>8.1f}  │  "
          f"{pr_psi.mean():>9.1f} {pr_psi.std():>8.1f}")
    print(f"  Histogram KL(GT‖Pred):  φ = {kl_phi:.3f}   ψ = {kl_psi:.3f}   (lower is better)")
    print(f"{'─'*78}\n")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(gt_phi, gt_psi, s=8, alpha=0.5, color="#1f77b4", label="GT")
    axes[0].scatter(pr_phi, pr_psi, s=8, alpha=0.5, color="#d62728", label="Pred")
    axes[0].set_xlim(-180, 180); axes[0].set_ylim(-180, 180)
    axes[0].set_xlabel("φ"); axes[0].set_ylabel("ψ")
    axes[0].set_title(f"Ramachandran — {peptide_id} (epoch {epoch})")
    axes[0].legend(loc="upper right")

    axes[1].hist(gt_phi, bins=36, range=(-180, 180), alpha=0.5, color="#1f77b4",
                 density=True, label="GT")
    axes[1].hist(pr_phi, bins=36, range=(-180, 180), alpha=0.5, color="#d62728",
                 density=True, label="Pred")
    axes[1].set_xlabel("φ (deg)"); axes[1].set_title(f"φ marginal (KL={kl_phi:.3f})")
    axes[1].legend()

    axes[2].hist(gt_psi, bins=36, range=(-180, 180), alpha=0.5, color="#1f77b4",
                 density=True, label="GT")
    axes[2].hist(pr_psi, bins=36, range=(-180, 180), alpha=0.5, color="#d62728",
                 density=True, label="Pred")
    axes[2].set_xlabel("ψ (deg)"); axes[2].set_title(f"ψ marginal (KL={kl_psi:.3f})")
    axes[2].legend()

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"epoch_{epoch:03d}_{peptide_id}.png")
    fig.savefig(path, dpi=110); plt.close(fig)
    print(f"  ✓ Distribution plot saved → {path}")
    return kl_phi, kl_psi


# ──────────────────────────────────────────────────────────────
# 5. Training Loop
# ──────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_proj", type=int, default=64,
                        help="Random projections for the sliced-Wasserstein loss.")
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--max_train", type=int, default=208)
    parser.add_argument("--max_val", type=int, default=87)
    parser.add_argument("--num_frames", type=int, default=300)
    parser.add_argument("--noise_dim", type=int, default=8)
    # ── Regularization / generalization knobs ──────────────────────────────
    # These fight the train↔val/test gap (overfitting) on the small (208-peptide)
    # training set. Defaults are raised from the original run.
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Transformer dropout (was 0.1). Higher = more regularization.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay (was 1e-5). Higher = more regularization.")
    parser.add_argument("--embed_noise", type=float, default=0.15,
                        help="Std of Gaussian jitter added to the (frozen) molecule "
                             "conditioning embedding during TRAINING ONLY, as a fraction "
                             "of each dim's batch std. Forces the generator to map nearby "
                             "embeddings to similar distributions → smoother interpolation "
                             "to unseen peptides. 0 disables it.")
    # ── Optional GNN encoder fine-tuning ───────────────────────────────────
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Unfreeze the GNN encoder and fine-tune it at a small LR "
                             "(--encoder_lr) during trajectory training, so embeddings adapt "
                             "to the generation task. OFF by default: it can help embeddings "
                             "but also adds capacity to overfit the same 208 peptides, so "
                             "always compare val/test KL with and without it. When enabled, "
                             "the tuned encoder is saved to best_encoder_finetuned.pt and "
                             "auto-loaded by predict_trajectory.py.")
    parser.add_argument("--encoder_lr", type=float, default=1e-5,
                        help="LR for the encoder when --finetune_encoder is set. Kept far "
                             "smaller than --lr so the pretrained representation is nudged, "
                             "not destroyed.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "timewarp_data", "2AA-complete")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    epoch_check_dir = os.path.join(out_dir, "epoch_checks")

    gnn_encoder = MolecularEncoder(8, 64, 32).to(device)
    encoder_path = os.path.join(base_dir, "encoder.pt")
    if not os.path.exists(encoder_path):
        print(f"Error: {encoder_path} not found."); return
    gnn_encoder.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
    if args.finetune_encoder:
        gnn_encoder.train()
        for p in gnn_encoder.parameters():
            p.requires_grad = True
        print(f"✓ Loaded GNN encoder (FINE-TUNING enabled, encoder_lr={args.encoder_lr}).")
    else:
        gnn_encoder.eval()
        for p in gnn_encoder.parameters():
            p.requires_grad = False
        print("✓ Loaded GNN encoder (frozen).")

    model = TrajectorySeqModel(
        mol_embed_dim=64, d_model=128, nhead=4, num_layers=4,
        dim_feedforward=256, num_frames=args.num_frames, dropout=args.dropout,
        noise_dim=args.noise_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ TrajectorySeqModel initialized ({n_params:,} trainable params)")

    if args.finetune_encoder:
        optimizer = torch.optim.AdamW(
            [
                {"params": model.parameters(), "lr": args.lr},
                {"params": gnn_encoder.parameters(), "lr": args.encoder_lr},
            ],
            lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15)

    print("\nLoading data...")
    train_graphs = load_split_with_dihedrals(data_dir, "train", args.max_train, args.num_frames)
    val_graphs = load_split_with_dihedrals(data_dir, "val", args.max_val, args.num_frames)
    if len(train_graphs) == 0 or len(val_graphs) == 0:
        print("Error: no data loaded."); return
    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

    track_graph = val_graphs[0]
    track_pid = getattr(track_graph, "peptide_id", "val0")
    track_batch = next(iter(DataLoader([track_graph], batch_size=1))).to(device)
    with torch.no_grad():
        node_z = gnn_encoder.encode_nodes(track_batch.x, track_batch.edge_index)
        track_mol_z = global_mean_pool(node_z, track_batch.batch)
    track_gt = track_graph.y_dihedrals.cpu().numpy()           # [F, 4]

    best_val = float("inf")
    epochs_no_improve = 0
    save_path = os.path.join(out_dir, "best_trajectory.pt")
    train_hist, val_hist = [], []

    print("\n" + "=" * 78)
    print("  DISTRIBUTION-MATCHING DIHEDRAL TRAJECTORY MODEL — TRAINING")
    print("=" * 78)
    print(f"  Loss: Sliced-Wasserstein ({args.num_proj} projections) in 4-D sin/cos space")
    print(f"  Epochs: {args.epochs}  Patience: {args.patience}  noise_dim: {args.noise_dim}")
    print(f"  Regularization → dropout: {args.dropout}  weight_decay: {args.weight_decay}  "
          f"embed_noise: {args.embed_noise}")
    print(f"  Encoder: {'FINE-TUNED @ lr=' + str(args.encoder_lr) if args.finetune_encoder else 'frozen'}")
    print("=" * 78 + "\n")

    def encode(batch, grad=False):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            nz = gnn_encoder.encode_nodes(batch.x, batch.edge_index)
            return global_mean_pool(nz, batch.batch)

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.finetune_encoder:
            gnn_encoder.train()
        tot = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            mol_z = encode(batch, grad=args.finetune_encoder)
            B = mol_z.size(0)
            # Conditioning-embedding augmentation (train only): jitter each molecule
            # embedding by Gaussian noise scaled to that dim's batch std. This is the
            # main anti-overfitting lever — it stops the generator from memorizing a
            # brittle one-to-one map from the 208 training embeddings to their exact
            # distributions, and forces a smooth map that interpolates to unseen peptides.
            if args.embed_noise > 0:
                std = mol_z.detach().std(dim=0, keepdim=True, unbiased=False)  # [1, 64]; 0 if B==1
                mol_z = mol_z + torch.randn_like(mol_z) * args.embed_noise * std
            gt = batch.y_dihedrals.view(B, args.num_frames, 4)
            pred = model(mol_z)
            loss = sliced_wasserstein_loss(pred, gt, args.num_proj)
            loss.backward()
            clip_params = (list(model.parameters()) + list(gnn_encoder.parameters())
                           if args.finetune_encoder else list(model.parameters()))
            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            tot += loss.item()
        avg_train = tot / len(train_loader)
        train_hist.append(avg_train)

        model.eval()
        if args.finetune_encoder:
            gnn_encoder.eval()
        v_tot = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                mol_z = encode(batch)
                B = mol_z.size(0)
                gt = batch.y_dihedrals.view(B, args.num_frames, 4)
                pred = model(mol_z)
                v_tot += sliced_wasserstein_loss(pred, gt, args.num_proj).item()
        avg_val = v_tot / len(val_loader)
        val_hist.append(avg_val)

        print(f"Epoch {epoch:3d}/{args.epochs} | Train SW {avg_train:.5f} | Val SW {avg_val:.5f}")

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                pred_track = model(track_mol_z).squeeze(0).cpu().numpy()   # [F, 4]
            diagnostic(epoch, pred_track, track_gt, track_pid, epoch_check_dir)

        scheduler.step(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            if args.finetune_encoder:
                enc_save_path = os.path.join(out_dir, "best_encoder_finetuned.pt")
                torch.save(gnn_encoder.state_dict(), enc_save_path)
            print(f"  ✓ Best model saved (val SW {best_val:.5f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"\n⚠ Early stopping at epoch {epoch}")
                break

    print(f"\nTraining complete. Best val SW: {best_val:.5f} → {save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_hist, label="Train Sliced-Wasserstein")
    plt.plot(val_hist, label="Val Sliced-Wasserstein")
    plt.xlabel("Epoch"); plt.ylabel("Sliced-Wasserstein distance")
    plt.title("Dihedral Distribution Matching — Loss Curve")
    plt.legend(); plt.grid(alpha=0.3)
    loss_path = os.path.join(out_dir, "loss_curve.png")
    plt.savefig(loss_path, dpi=120); plt.close()
    print(f"✓ Loss curve saved → {loss_path}")


if __name__ == "__main__":
    main()
