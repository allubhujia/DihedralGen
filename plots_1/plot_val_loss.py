import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ensure project root is on path ────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ── import loss lists from pipeline_auto ──────────────────────────────────
# Running pipeline_auto as a module executes training + validation and
# populates val_total_losses, val_rec_losses, val_sim_losses.
print("Running pipeline_auto (training + validation) …\n")
import models.pipeline_auto as pa

val_total = pa.val_total_losses
val_rec   = pa.val_rec_losses
val_sim   = pa.val_sim_losses

# ── plot ──────────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(11, 6), dpi=130)

batches = range(1, len(val_total) + 1)
batches_per_epoch = len(val_total) if len(val_total) > 0 else 1
epochs = [b / batches_per_epoch for b in batches]
ax.plot(epochs, val_total, label="Total Loss",          color="#ff6f61", linewidth=2)
ax.plot(epochs, val_rec,   label="Reconstruction Loss", color="#4caf50", linewidth=2)
ax.plot(epochs, val_sim,   label="Similarity Loss",     color="#2196f3", linewidth=2)

ax.set_title("Validation Set — Loss Curves", fontsize=17, weight="bold", pad=14)
ax.set_xlabel("Epoch Number",                 fontsize=13, color="#cccccc")
ax.set_ylabel("Loss",                        fontsize=13, color="#cccccc")
ax.tick_params(colors="#aaaaaa")
ax.grid(True, linestyle="--", alpha=0.25)
ax.legend(frameon=False, fontsize=12)
fig.tight_layout()

out = os.path.join("plots_1", "val_losses.png")
fig.savefig(out, bbox_inches="tight")
print(f"\nSaved → {out}")
