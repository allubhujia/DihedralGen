"""
dihedral_comparison.py — Backbone dihedral angle analysis: Predicted vs Ground Truth.

Creates a directory `trajectory_analysis(main)` with per-peptide subfolders
`{peptide_name}_plot` containing two comparison plots:

    1. Overlaid Ramachandran plot (φ vs ψ scatter with free-energy contours)
    2. Comparative histogram + KDE of dihedral angle distributions

USAGE (from project root, via WSL or terminal):
    python trajectory_analysis/dihedral_comparison.py

    # or for a specific peptide:
    python trajectory_analysis/dihedral_comparison.py --peptide AC

The script auto-discovers all peptide subdirectories inside
`trajectory_pdb_files/` or can target a single peptide via --peptide.

DEPENDENCIES:
    pip install numpy matplotlib scipy
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde, entropy

# ──────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

PDB_DIR  = os.path.join(BASE_DIR, "trajectory_pdb_files")
OUT_ROOT = os.path.join(BASE_DIR, "trajectory_analysis(main)")

# Ramachandran pi-fraction labels
PI_TICKS  = [-180, -135, -90, -45, 0, 45, 90, 135, 180]
PI_LABELS = ["-π", "-¾π", "-½π", "-¼π", "0", "¼π", "½π", "¾π", "π"]


# ──────────────────────────────────────────────────────────────
#  1. Dihedral computation from PDB backbone
# ──────────────────────────────────────────────────────────────

from scripts.backbone_utils import parse_backbone_from_pdb, compute_phi_psi_from_positions, compute_dihedral

def compute_phi_psi_from_pdb(pdb_path):
    """Compute phi, psi in degrees from a multi-MODEL PDB ground truth file.
    
    Returns phi [T], psi [T] in degrees.
    """
    backbone = parse_backbone_from_pdb(pdb_path)
    
    # Parse all frames from the multi-MODEL PDB
    frames = []
    current_frame = {}
    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM"):
                idx  = int(line[6:11].strip()) - 1
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                current_frame[idx] = np.array([x, y, z])
            elif line.startswith("ENDMDL"):
                if current_frame:
                    frames.append(current_frame)
                    current_frame = {}
    if current_frame:
        frames.append(current_frame)

    if not frames:
        return None, None

    T = len(frames)
    
    # Reconstruct positions array [T, num_atoms, 3]
    # We find the max atom index to size the array
    max_idx = max([max(frame.keys()) for frame in frames]) if frames else 0
    positions = np.zeros((T, max_idx + 1, 3))
    
    for t, frame in enumerate(frames):
        for idx, coords in frame.items():
            positions[t, idx] = coords
            
    phi_rad, psi_rad = compute_phi_psi_from_positions(positions, backbone)
    return np.degrees(phi_rad), np.degrees(psi_rad)


# ──────────────────────────────────────────────────────────────
#  2. Load dihedral data (precomputed .npz or from PDB)
# ──────────────────────────────────────────────────────────────

def load_dihedral_data(peptide_dir, peptide_id):
    """Load GT and predicted phi/psi from precomputed .npz files.
    
    Falls back to computing GT from the PDB if the GT npz is missing.
    
    Returns:
        gt_phi, gt_psi       — 1-D arrays [T]      (degrees)
        pred_phi, pred_psi   — 2-D arrays [S, T]    (degrees, S = num_samples)
    """
    gt_npz   = os.path.join(peptide_dir, f"{peptide_id}_ground_truth_dihedrals.npz")
    pred_npz = os.path.join(peptide_dir, f"{peptide_id}_predicted_dihedrals.npz")
    gt_pdb   = os.path.join(peptide_dir, f"{peptide_id}_ground_truth.pdb")

    # ── Ground truth
    if os.path.exists(gt_npz):
        data = np.load(gt_npz)
        gt_phi, gt_psi = data["phi"], data["psi"]
    elif os.path.exists(gt_pdb):
        gt_phi, gt_psi = compute_phi_psi_from_pdb(gt_pdb)
        if gt_phi is None:
            raise RuntimeError(f"Could not parse backbone from {gt_pdb}")
    else:
        raise FileNotFoundError(f"No GT dihedral data found for {peptide_id}")

    # ── Predictions
    if os.path.exists(pred_npz):
        data = np.load(pred_npz)
        pred_phi, pred_psi = data["phi"], data["psi"]
        # Ensure 2-D: [S, T]
        if pred_phi.ndim == 1:
            pred_phi = pred_phi[np.newaxis, :]
            pred_psi = pred_psi[np.newaxis, :]
    else:
        raise FileNotFoundError(f"No predicted dihedral data found for {peptide_id}")

    return gt_phi, gt_psi, pred_phi, pred_psi


# ──────────────────────────────────────────────────────────────
#  3. Free energy surface helper
# ──────────────────────────────────────────────────────────────

def compute_free_energy(phi, psi, bins=72, kT=2.494):
    """2-D free energy F = −kT ln(P) from dihedral angles in degrees."""
    edges = np.linspace(-180, 180, bins + 1)
    H, xedges, yedges = np.histogram2d(phi, psi, bins=[edges, edges], density=True)
    with np.errstate(divide="ignore"):
        F = -kT * np.log(H)
    F[np.isinf(F)] = np.nan
    F -= np.nanmin(F)
    cx = 0.5 * (xedges[:-1] + xedges[1:])
    cy = 0.5 * (yedges[:-1] + yedges[1:])
    return cx, cy, F


# ──────────────────────────────────────────────────────────────
#  4. Plot 1 — Overlaid Ramachandran
# ──────────────────────────────────────────────────────────────

def plot_ramachandran_overlay(gt_phi, gt_psi, pred_phi, pred_psi, peptide_id, out_dir):
    """Overlaid Ramachandran scatter with free-energy contours for GT and Pred."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    fig.suptitle(f"Peptide {peptide_id} — Overlaid Ramachandran Plot",
                 fontsize=18, fontweight="bold", y=1.0)

    # Flatten predicted samples
    pred_phi_flat = pred_phi.flatten()
    pred_psi_flat = pred_psi.flatten()

    for ax, (phi, psi, title, scatter_c, cmap) in zip(axes, [
        (gt_phi, gt_psi, "Ground Truth", "#d32f2f", "YlGnBu_r"),
        (pred_phi_flat, pred_psi_flat, "Predicted", "#1565c0", "YlOrRd_r"),
    ]):
        cx, cy, F = compute_free_energy(phi, psi)
        F_smooth = gaussian_filter(np.nan_to_num(F, nan=np.nanmax(F)), sigma=1.5)
        finite_vals = F[np.isfinite(F)]
        vmax = np.nanpercentile(finite_vals, 95) if len(finite_vals) > 0 else 30
        if vmax <= 0:
            vmax = 0.1  # Ensure levels are strictly increasing
        levels = np.linspace(0, vmax, 20)

        PHI, PSI = np.meshgrid(cx, cy)
        cf = ax.contourf(PHI, PSI, F_smooth.T, levels=levels, cmap=cmap, extend="max")
        ax.scatter(phi, psi, c=scatter_c, alpha=0.5, s=5, edgecolors="none", zorder=5)

        ax.set_xlabel("φ (phi)", fontsize=14, fontweight="bold")
        ax.set_ylabel("ψ (psi)", fontsize=14, fontweight="bold")
        ax.set_title(title, fontsize=15, fontweight="bold")
        ax.set_xticks(PI_TICKS); ax.set_xticklabels(PI_LABELS, fontsize=9)
        ax.set_yticks(PI_TICKS); ax.set_yticklabels(PI_LABELS, fontsize=9)
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180); ax.set_aspect("equal")
        plt.colorbar(cf, ax=ax, label="Free Energy (kJ/mol)", shrink=0.85)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"{peptide_id}_ramachandran_overlay.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Ramachandran overlay → {path}")


# ──────────────────────────────────────────────────────────────
#  5. Circular statistics helpers
# ──────────────────────────────────────────────────────────────

def circular_mean_deg(angles_deg):
    """Circular mean of angles in degrees."""
    rad = np.radians(angles_deg)
    return np.degrees(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad))))


def circular_std_deg(angles_deg):
    """Circular standard deviation of angles in degrees."""
    rad = np.radians(angles_deg)
    R = np.sqrt(np.mean(np.sin(rad))**2 + np.mean(np.cos(rad))**2)
    R = min(R, 1.0)  # clamp
    return np.degrees(np.sqrt(-2.0 * np.log(R)))


def _kde_on_grid(data, grid, bw_method=None):
    """Compute KDE on a grid, handling edge cases."""
    if len(data) < 5:
        return np.zeros_like(grid)
    try:
        kde = gaussian_kde(data, bw_method=bw_method)
        return kde(grid)
    except np.linalg.LinAlgError:
        return np.zeros_like(grid)


def compute_kl_divergence(p_angles, q_angles, bins=60):
    """Compute KL divergence between two sets of angles using histogram density."""
    edges = np.linspace(-180, 180, bins + 1)
    p_hist, _ = np.histogram(p_angles, bins=edges, density=True)
    q_hist, _ = np.histogram(q_angles, bins=edges, density=True)
    
    # Add small epsilon to avoid division by zero or log(0)
    eps = 1e-8
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    
    # Normalize
    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)
    
    return entropy(p_hist, q_hist)


# ──────────────────────────────────────────────────────────────
#  6. Plot 2 — Comparative dihedral angle distributions
# ──────────────────────────────────────────────────────────────

def plot_dihedral_distributions(gt_phi, gt_psi, pred_phi, pred_psi, peptide_id, out_dir):
    """Improved overlaid histogram + KDE distributions for φ and ψ with
    circular statistics annotations and matched y-axes."""
    pred_phi_flat = pred_phi.flatten()
    pred_psi_flat = pred_psi.flatten()

    # ── Use finer bins for smoother histograms
    n_bins = 60
    bins = np.linspace(-180, 180, n_bins + 1)
    kde_x = np.linspace(-180, 180, 500)

    # ── Compute circular stats for annotation
    kl_phi = compute_kl_divergence(gt_phi, pred_phi_flat, bins=n_bins)
    kl_psi = compute_kl_divergence(gt_psi, pred_psi_flat, bins=n_bins)

    stats = {
        "gt_phi":   (circular_mean_deg(gt_phi),        circular_std_deg(gt_phi)),
        "pred_phi": (circular_mean_deg(pred_phi_flat),  circular_std_deg(pred_phi_flat)),
        "gt_psi":   (circular_mean_deg(gt_psi),         circular_std_deg(gt_psi)),
        "pred_psi": (circular_mean_deg(pred_psi_flat),  circular_std_deg(pred_psi_flat)),
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"Peptide {peptide_id} — Comparative Dihedral Angle Distributions",
                 fontsize=18, fontweight="bold", y=0.98)

    # ── Color palette
    gt_phi_c, pred_phi_c = "#1565c0", "#e53935"
    gt_psi_c, pred_psi_c = "#2e7d32", "#f57c00"

    # ────────────── Row 1: φ distributions ──────────────
    # Left: Histogram only
    ax = axes[0, 0]
    ax.hist(gt_phi, bins=bins, alpha=0.50, color=gt_phi_c, density=True,
            edgecolor="white", linewidth=0.5, label="GT φ", zorder=2)
    ax.hist(pred_phi_flat, bins=bins, alpha=0.40, color=pred_phi_c, density=True,
            edgecolor="white", linewidth=0.5, label="Predicted φ", zorder=2)
    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Phi (φ) — Histogram", fontsize=13, fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, linestyle="--")
    # Stats annotation
    txt = (f"GT:   μ={stats['gt_phi'][0]:+.1f}°  σ={stats['gt_phi'][1]:.1f}°\n"
           f"Pred: μ={stats['pred_phi'][0]:+.1f}°  σ={stats['pred_phi'][1]:.1f}°\n"
           f"KL Div: {kl_phi:.4f}")
    ax.text(0.03, 0.95, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    # Right: KDE only
    ax = axes[0, 1]
    kde_gt = _kde_on_grid(gt_phi, kde_x)
    kde_pr = _kde_on_grid(pred_phi_flat, kde_x)
    ax.fill_between(kde_x, kde_gt, alpha=0.25, color=gt_phi_c, zorder=2)
    ax.fill_between(kde_x, kde_pr, alpha=0.20, color=pred_phi_c, zorder=2)
    ax.plot(kde_x, kde_gt, color=gt_phi_c, linewidth=2.5, label="GT φ", zorder=3)
    ax.plot(kde_x, kde_pr, color=pred_phi_c, linewidth=2.5, linestyle="--",
            label="Predicted φ", zorder=3)
    ax.set_xlabel("φ (degrees)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Phi (φ) — KDE", fontsize=13, fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, linestyle="--")

    # Match y-axes for φ row
    ymax_phi = max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]) * 1.08
    axes[0, 0].set_ylim(0, ymax_phi)
    axes[0, 1].set_ylim(0, ymax_phi)

    # ────────────── Row 2: ψ distributions ──────────────
    # Left: Histogram only
    ax = axes[1, 0]
    ax.hist(gt_psi, bins=bins, alpha=0.50, color=gt_psi_c, density=True,
            edgecolor="white", linewidth=0.5, label="GT ψ", zorder=2)
    ax.hist(pred_psi_flat, bins=bins, alpha=0.40, color=pred_psi_c, density=True,
            edgecolor="white", linewidth=0.5, label="Predicted ψ", zorder=2)
    ax.set_xlabel("ψ (degrees)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Psi (ψ) — Histogram", fontsize=13, fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, linestyle="--")
    txt = (f"GT:   μ={stats['gt_psi'][0]:+.1f}°  σ={stats['gt_psi'][1]:.1f}°\n"
           f"Pred: μ={stats['pred_psi'][0]:+.1f}°  σ={stats['pred_psi'][1]:.1f}°\n"
           f"KL Div: {kl_psi:.4f}")
    ax.text(0.03, 0.95, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85, edgecolor="#ccc"))

    # Right: KDE only
    ax = axes[1, 1]
    kde_gt = _kde_on_grid(gt_psi, kde_x)
    kde_pr = _kde_on_grid(pred_psi_flat, kde_x)
    ax.fill_between(kde_x, kde_gt, alpha=0.25, color=gt_psi_c, zorder=2)
    ax.fill_between(kde_x, kde_pr, alpha=0.20, color=pred_psi_c, zorder=2)
    ax.plot(kde_x, kde_gt, color=gt_psi_c, linewidth=2.5, label="GT ψ", zorder=3)
    ax.plot(kde_x, kde_pr, color=pred_psi_c, linewidth=2.5, linestyle="--",
            label="Predicted ψ", zorder=3)
    ax.set_xlabel("ψ (degrees)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Psi (ψ) — KDE", fontsize=13, fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, linestyle="--")

    # Match y-axes for ψ row
    ymax_psi = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1]) * 1.08
    axes[1, 0].set_ylim(0, ymax_psi)
    axes[1, 1].set_ylim(0, ymax_psi)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, f"{peptide_id}_dihedral_distributions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Distribution plot   → {path}")

    return kl_phi, kl_psi


# ──────────────────────────────────────────────────────────────
#  7. Discover peptides
# ──────────────────────────────────────────────────────────────

def discover_peptides():
    """List all peptide subdirectories in trajectory_pdb_files/."""
    if not os.path.isdir(PDB_DIR):
        return []
    return sorted([
        d for d in os.listdir(PDB_DIR)
        if os.path.isdir(os.path.join(PDB_DIR, d))
    ])


# ──────────────────────────────────────────────────────────────
#  8. Process one peptide
# ──────────────────────────────────────────────────────────────

def process_peptide(peptide_id):
    """Generate comparison plots for a single peptide."""
    peptide_dir = os.path.join(PDB_DIR, peptide_id)
    if not os.path.isdir(peptide_dir):
        print(f"  ⚠ Directory not found: {peptide_dir}")
        return False

    out_dir = os.path.join(OUT_ROOT, f"{peptide_id}_plot")
    os.makedirs(out_dir, exist_ok=True)

    try:
        gt_phi, gt_psi, pred_phi, pred_psi = load_dihedral_data(peptide_dir, peptide_id)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"  ⚠ Skipping {peptide_id}: {e}")
        return False

    n_gt_frames = len(gt_phi)
    n_samples   = pred_phi.shape[0]
    n_pred_frames = pred_phi.shape[1]

    print(f"\n  Peptide: {peptide_id}")
    print(f"  GT: {n_gt_frames} frames | Pred: {n_samples} samples × {n_pred_frames} frames")
    print(f"  Output: {out_dir}")

    # Generate plots
    plot_ramachandran_overlay(gt_phi, gt_psi, pred_phi, pred_psi, peptide_id, out_dir)
    kl_phi, kl_psi = plot_dihedral_distributions(gt_phi, gt_psi, pred_phi, pred_psi, peptide_id, out_dir)

    # Save KL Divergence
    kl_dir = os.path.join(OUT_ROOT, "kl_divergence")
    os.makedirs(kl_dir, exist_ok=True)
    kl_file = os.path.join(kl_dir, f"{peptide_id}_kl_divergence.txt")
    with open(kl_file, "w") as f:
        f.write(f"Peptide: {peptide_id}\n")
        f.write(f"Phi KL Divergence: {kl_phi:.6f}\n")
        f.write(f"Psi KL Divergence: {kl_psi:.6f}\n")
    print(f"  ✓ KL Divergence saved → {kl_file}")

    return True


# ──────────────────────────────────────────────────────────────
#  9. Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dihedral angle comparison: Predicted vs Ground Truth.")
    parser.add_argument("--peptide", type=str, default=None,
                        help="Process a single peptide (e.g., AC). "
                             "If omitted, all peptides in trajectory_pdb_files/ are processed.")
    args = parser.parse_args()

    print(f"\n{'═'*64}")
    print(f"  TRAJECTORY DIHEDRAL ANALYSIS")
    print(f"{'═'*64}")
    print(f"  Source:  {PDB_DIR}")
    print(f"  Output:  {OUT_ROOT}")
    print(f"{'═'*64}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    if args.peptide:
        peptides = [args.peptide.upper()]
    else:
        peptides = discover_peptides()
        if not peptides:
            print("\n  ⚠ No peptide directories found in trajectory_pdb_files/")
            return

    print(f"\n  Peptides to process: {', '.join(peptides)}")

    success, failed = 0, 0
    for pid in peptides:
        ok = process_peptide(pid)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'═'*64}")
    print(f"  ✅ COMPLETE — {success} peptide(s) processed, {failed} skipped")
    print(f"  Output directory: {OUT_ROOT}")
    print(f"{'═'*64}\n")


if __name__ == "__main__":
    main()
