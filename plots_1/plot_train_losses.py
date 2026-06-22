"""
plot_losses.py  –  Line chart of per-batch losses from pipeline_auto.py.

This file does NOT calculate any losses itself.
It imports the per-batch loss lists from models/pipeline_auto.py
and plots them as a line chart.

OUTPUT:
    plots/training_losses.png

USAGE:
    python -m plots.plot_losses
"""

import os
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# 1. Import per-batch loss lists from pipeline_auto.py
# ──────────────────────────────────────────────────────────────

from models.pipeline_auto import total_losses, rec_losses, sim_losses

print(f"\nImported {len(total_losses)} batch losses from pipeline_auto.\n")


# ──────────────────────────────────────────────────────────────
# 2. Plot the three losses as a line chart
# ──────────────────────────────────────────────────────────────

batches = list(range(1, len(total_losses) + 1))
num_epochs = 300
batches_per_epoch = max(1, len(total_losses) // num_epochs)
epochs = [b / batches_per_epoch for b in batches]

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(epochs, total_losses, color="#E63946", linewidth=2.2, marker="o",
        markersize=5, label="Total Loss")
ax.plot(epochs, rec_losses,   color="#457B9D", linewidth=2.2, marker="s",
        markersize=5, label="Reconstruction Loss")
ax.plot(epochs, sim_losses,   color="#2A9D8F", linewidth=2.2, marker="^",
        markersize=5, label="Similarity Loss")

ax.set_xlabel("Epoch Number", fontsize=13)
ax.set_ylabel("Loss Value",   fontsize=13)
ax.set_title("Training Set - Forward Pass Losses",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")

fig.tight_layout()

# ── Save ──
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
save_path = os.path.join(output_dir, "training_losses.png")

fig.savefig(save_path, dpi=150)
print(f"Plot saved -> {save_path}")
plt.close(fig)
