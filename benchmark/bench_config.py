"""
bench_config.py — paths and defaults for the Stage 2 vs Stage 3 benchmark.

Stage-specific hyperparameters are NOT duplicated here. Stage 3's live in
temporal_dynamics/config.py and Stage 2's are baked into its checkpoint; this
file only holds what the comparison itself needs.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR    = os.path.join(PROJECT_ROOT, "benchmark")
RESULTS_DIR  = os.path.join(BENCH_DIR, "results")

# ── Stage 2 (trajectory_pre_) ────────────────────────────────────────────────
STAGE2_DIR        = os.path.join(PROJECT_ROOT, "trajectory_pre_")
STAGE2_CKPT       = os.path.join(STAGE2_DIR, "best_trajectory.pt")
STAGE2_TRAIN_FILE = os.path.join(STAGE2_DIR, "train_trajectory.py")
# A fine-tuned encoder is preferred when present, so the embeddings match what
# the Stage-2 generator was actually trained against.
STAGE2_ENCODER_FT = os.path.join(STAGE2_DIR, "best_encoder_finetuned.pt")

# The Stage-2 checkpoint's frame positional embedding is [1, 300, 128], so the
# generator can only ever emit exactly this many frames. Requests for a
# different count are subsampled or tiled from it — see stage2.py.
STAGE2_NATIVE_FRAMES = 300

# Stage 2 conditions on the 64-d pooled NODE embedding (encode_nodes +
# global_mean_pool), not the 32-d projected z that Stage 3 uses. Each stage gets
# the conditioning it was trained with.
STAGE2_MOL_EMBED_DIM = 64
STAGE2_D_MODEL       = 128
STAGE2_NHEAD         = 4
STAGE2_NUM_LAYERS    = 4
STAGE2_FF            = 256
STAGE2_DROPOUT       = 0.1

# ── Shared encoder (Stage 1) ─────────────────────────────────────────────────
ENCODER_CKPT = os.path.join(PROJECT_ROOT, "encoder.pt")

# ── Defaults for the CLI ─────────────────────────────────────────────────────
DEFAULT_PEPTIDE     = "AC"
DEFAULT_FRAMES      = 300
DEFAULT_START_FRAME = 0

# ── Metric / plot settings (kept identical to temporal_dynamics.evaluate) ─────
GT_STRIDE = 5      # subsample stride for the full-MD reference density
SEED      = 0


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def run_dir(peptide, frames, start_frame):
    """Output directory for one benchmark run."""
    return os.path.join(RESULTS_DIR, f"{peptide}_f{frames}_s{start_frame}")
