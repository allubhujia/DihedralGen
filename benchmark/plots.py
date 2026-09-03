"""
plots.py — the two benchmark figures.

  figure 1  ramachandran_comparison.png   MD | Stage 2 | Stage 3, side by side
  figure 2  kl_divergence_comparison.png  phi / psi marginals + a KL bar chart

Both figures put the ground-truth MD density behind every panel so the eye
compares each stage against the same reference rather than against each other.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

GT_COLOR = "steelblue"
S2_COLOR = "darkorange"
S3_COLOR = "crimson"

RAMA_GRID = 64
HIST_BINS = 48


def _density(phi, psi, bins=RAMA_GRID, sigma=1.5):
    edges = np.linspace(-180, 180, bins + 1)
    H, _, _ = np.histogram2d(phi, psi, bins=[edges, edges])
    return gaussian_filter(H, sigma), 0.5 * (edges[:-1] + edges[1:])


def ramachandran_figure(pid, gt, s2, s3, metrics, out_dir):
    """Figure 1 — three Ramachandran panels sharing one MD reference density."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    H, centres = _density(gt["phi_full"], gt["psi_full"])

    panels = [
        (axes[0], None, None, GT_COLOR,
         f"Ground truth MD\n({len(gt['phi_full'])} frames, full trajectory)"),
        (axes[1], s2["phi"], s2["psi"], S2_COLOR,
         f"Stage 2 - distribution model\nJS2D = {metrics['s2']['js2d']:.3f} bits"),
        (axes[2], s3["phi"], s3["psi"], S3_COLOR,
         f"Stage 3 - learned propagator\nJS2D = {metrics['s3']['js2d']:.3f} bits"),
    ]

    for ax, phi, psi, colour, title in panels:
        ax.contourf(centres, centres, H.T, levels=12, cmap="Blues", alpha=0.75)
        if phi is None:
            # The reference panel shows the MD samples themselves.
            ax.scatter(gt["phi_window"], gt["psi_window"], s=8, c=GT_COLOR,
                       alpha=0.55, edgecolors="none", label="MD window")
        else:
            ax.scatter(phi, psi, s=10, c=colour, alpha=0.75,
                       edgecolors="none", label="generated")
        ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
        ax.set_xlabel("phi (deg)")
        ax.set_title(title, fontsize=11)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("psi (deg)")
    fig.suptitle(
        f"{pid} - Ramachandran: Stage 2 vs Stage 3   "
        f"(blue contours = full MD density, identical in all three panels)",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "ramachandran_comparison.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def kl_figure(pid, gt, s2, s3, metrics, out_dir):
    """Figure 2 — phi/psi marginal histograms plus a KL bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    bins = np.linspace(-180, 180, HIST_BINS)

    for ax, key, label in ((axes[0], "phi", "phi"), (axes[1], "psi", "psi")):
        ax.hist(gt[f"{key}_full"], bins=bins, density=True, alpha=0.45,
                color=GT_COLOR, label="MD (reference)")
        ax.hist(s2[key], bins=bins, density=True, histtype="step", lw=2.0,
                color=S2_COLOR,
                label=f"Stage 2  (KL {metrics['s2'][f'kl_{key}']:.3f})")
        ax.hist(s3[key], bins=bins, density=True, histtype="step", lw=2.0,
                color=S3_COLOR,
                label=f"Stage 3  (KL {metrics['s3'][f'kl_{key}']:.3f})")
        ax.set_xlabel(f"{label} (deg)")
        ax.set_ylabel("density")
        ax.set_title(f"{label} marginal distribution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)

    # --- KL bar chart -------------------------------------------------------
    ax = axes[2]
    labels = ["KL phi", "KL psi", "JS 2D"]
    s2_vals = [metrics["s2"]["kl_phi"], metrics["s2"]["kl_psi"], metrics["s2"]["js2d"]]
    s3_vals = [metrics["s3"]["kl_phi"], metrics["s3"]["kl_psi"], metrics["s3"]["js2d"]]

    x = np.arange(len(labels))
    w = 0.36
    b2 = ax.bar(x - w / 2, s2_vals, w, color=S2_COLOR, label="Stage 2")
    b3 = ax.bar(x + w / 2, s3_vals, w, color=S3_COLOR, label="Stage 3")

    for bars in (b2, b3):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("divergence  (lower is better)")
    ax.set_title("Divergence from ground-truth MD")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, axis="y")

    fig.suptitle(
        f"{pid} - divergence from MD: Stage 2 vs Stage 3   "
        f"(both scored against the same full-MD reference)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(out_dir, "kl_divergence_comparison.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
