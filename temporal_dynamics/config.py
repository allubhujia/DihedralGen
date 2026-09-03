"""
config.py — every hyperparameter and path for Stage 3 in one place.

Nothing here is imported from outside `temporal_dynamics/` except the paths to
the Stage-1 checkpoint and the Timewarp dataset.
"""

import os

# ──────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE3_DIR   = os.path.join(PROJECT_ROOT, "temporal_dynamics")

DATA_ROOT    = os.path.join(PROJECT_ROOT, "timewarp_data", "2AA-complete")
ENCODER_CKPT = os.path.join(PROJECT_ROOT, "encoder.pt")

CACHE_DIR       = os.path.join(STAGE3_DIR, "cache")
TRANSITION_DIR  = os.path.join(CACHE_DIR, "transitions")   # per-peptide .npz written by prepare_data.py
EMBEDDING_CACHE = os.path.join(CACHE_DIR, "embeddings.pt") # {peptide_id: z[32]} written by adapter.py
STATS_CACHE     = os.path.join(CACHE_DIR, "stats.npz")     # normalisation scales

CKPT_PATH   = os.path.join(STAGE3_DIR, "best_propagator.pt")
LOSS_CURVE  = os.path.join(STAGE3_DIR, "loss_curve.png")
RESULTS_DIR = os.path.join(STAGE3_DIR, "results")

# ──────────────────────────────────────────────────────────────
# Stage-1 encoder geometry (must match encoder.pt — do not change)
# ──────────────────────────────────────────────────────────────

ENC_IN_CHANNELS     = 8
ENC_HIDDEN_CHANNELS = 64
ENC_EMBEDDING_DIM   = 32
ENC_NUM_LAYERS      = 3

COND_DIM = ENC_EMBEDDING_DIM          # 32-d frozen conditioning vector

# ──────────────────────────────────────────────────────────────
# Physics / data
# ──────────────────────────────────────────────────────────────

# The Timewarp 2AA trajectories save one frame every 10,000 MD steps = 5 ps.
# We learn the transition over exactly that lag; it is NOT an MD integrator.
FRAME_LAG_PS = 5.0

BOND_CUTOFF_NM = 0.2      # covalent bond cutoff, same value Stage 1 uses
ELEMENTS       = ["H", "C", "N", "O", "S"]
NUM_ELEMENTS   = len(ELEMENTS)
ELEMENT_MASS   = [1.008, 12.011, 14.007, 15.999, 32.06]   # amu, indexed like ELEMENTS

# Rigid-body motion between saved frames is ~5x larger than internal motion,
# and is not a function of the molecule. We Kabsch-align every frame onto its
# predecessor and remove the centre-of-mass velocity.
REMOVE_GLOBAL_ROTATION    = True
REMOVE_COM_VELOCITY       = True

# ──────────────────────────────────────────────────────────────
# Velocity: kept as a real state channel, but no longer allowed to
# dominate the gradient
# ──────────────────────────────────────────────────────────────
#
# The propagator models the FULL state (position, velocity): vel_t is fed in as
# context and vel_{t+1} is emitted as a second target channel, so the rollout
# carries a genuine (x, v) state forward exactly as an MD propagator does.
PREDICT_VELOCITY   = True    # emit vel_{t+1} as a second target channel
USE_VELOCITY_INPUT = True    # feed vel_t to the network as an input channel

# THE CATCH, and why CHANNEL_LOSS_WEIGHTS exists.
#
# At a 5 ps lag momentum is fully decorrelated — velocity_utils measured Pearson
# r = -0.005 between vel_t and the frame-to-frame displacement. vel_{t+1} given
# the state at t is therefore close to a fresh Maxwell-Boltzmann draw: the best
# any model can do is reproduce its marginal, and the rest is irreducible noise.
#
# The first run's validation curve shows exactly that: the velocity channel sat
# flat at 1.51 for all 40 epochs and learned nothing, while the displacement
# channel fell 1.09 -> 0.56. The loss averaged the two equally, so HALF the
# gradient was chasing noise — and that noise swamped the subtle conditional
# signal (the torsional restoring drift) that the Ramachandran metrics depend on.
#
# The fix is weighting, not deletion. Velocity stays in the model and in the
# rolled-forward state; it just no longer gets an equal vote in the objective.
# Set to [1.0, 1.0] to reproduce the original equal-average behaviour.
CHANNEL_LOSS_WEIGHTS = [1.0, 0.15]   # [dpos, vel_{t+1}]

# ──────────────────────────────────────────────────────────────
# Dataset size
# ──────────────────────────────────────────────────────────────
#
# Carrying velocity means the cache stores four [T, N, 3] arrays per peptide
# (pos_t, vel_t, dpos, vel_tp1) rather than two, so it costs ~4.2 MB per peptide
# at 3000 frames — roughly 700 MB over 168 peptides. MAX_FRAMES is set by the
# free space on this machine, not by what the model wants; raise it if you have
# the headroom. 168 x 3000 is still ~3x the transitions the first run used.

MAX_TRAIN_PEPTIDES = 140     # how many peptides to pull from timewarp_data/.../train
MAX_VAL_PEPTIDES   = 20
MAX_FRAMES         = 3000    # frames kept per peptide (=> MAX_FRAMES-1 transitions)
FRAME_STRIDE       = 1       # 1 = consecutive saved frames (the 5 ps lag)

# ──────────────────────────────────────────────────────────────
# Model — vector-channel EGNN
# ──────────────────────────────────────────────────────────────

# Deliberately unchanged from the first run. The failure was diagnosed as an
# objective problem (half the loss was unlearnable velocity, and hydrogen
# outvoted the backbone in the rest), not a capacity problem — so the compute
# budget buys gradient steps over the 4.2x larger dataset rather than parameters.
HIDDEN_DIM   = 96      # invariant scalar width
VEC_CHANNELS = 16      # equivariant vector channels [N, C, 3]
NUM_LAYERS   = 4       # message-passing rounds
NUM_RBF      = 24      # Gaussian radial basis functions for edge distances
RBF_CUTOFF_NM = 1.2    # distance range covered by the RBF expansion
TIME_EMB_DIM = 32      # sinusoidal embedding width for the flow time tau

# ──────────────────────────────────────────────────────────────
# Flow matching
# ──────────────────────────────────────────────────────────────

# Rectified-flow / conditional-OT path:  Y_tau = (1-tau)*eps + tau*Y ,  target = Y - eps
FLOW_SIGMA  = 1.0     # std of the source Gaussian (targets are unit-normalised first)
FLOW_STEPS  = 20      # Euler steps used at sampling time

FLOW_TEMP   = 1.0     # >1 widens the sampled distribution, <1 sharpens it


# ──────────────────────────────────────────────────────────────
# Target normalisation and loss weighting
# ──────────────────────────────────────────────────────────────
#
# The displacement scale is strongly element-dependent — measured over the
# cache: H 0.071 nm, O 0.072, S 0.070, N 0.048, C 0.036. Dividing everything by
# ONE global std (0.062) leaves hydrogen with ~4x the squared amplitude of
# carbon, so hydrogen dominates the gradient purely by moving further.
# Per-element scaling puts every atom on unit variance first.
PER_ELEMENT_SCALE = True

# Even at unit variance, atoms are still counted one-for-one, and a dipeptide is
# roughly half hydrogen (11 of 25 atoms for AD). Measured on AD, the six
# backbone atoms that define phi/psi owned 7.1% of the squared-displacement
# signal while hydrogens owned 55%.
#
# Mass weighting is the standard fix and needs no reference to the metric: it
# makes the loss a kinetic-energy-like quantity, and it matches the convention
# that structural accuracy is judged on heavy atoms (hydrogen positions are
# usually not even experimentally resolved). Weights are mass**power, rescaled
# to mean 1 so the loss magnitude stays comparable. power=0 recovers the old
# uniform weighting.
LOSS_MASS_POWER = 1.0

# ──────────────────────────────────────────────────────────────
# State-noise augmentation — where the restoring force comes from
# ──────────────────────────────────────────────────────────────
#
# THE failure of the first run. Measured against ground truth on 400 held-out
# MD states, the correlation between the current phi and its one-step change —
# the restoring pull back into the basin — was:
#
#     ground truth  -0.705        first model  -0.185
#
# The model's per-step noise was actually SMALLER than MD's (24 deg vs 45 deg),
# so it was never too diffusive. Its restoring force was ~4x too weak, and a
# random walk with a weak spring has a stationary width of sqrt(D/k) — which is
# why 300 rollout steps smeared phi over the full circle (std 120 deg vs MD's
# 41 deg) and put a fifth of its mass in the sterically forbidden phi > 0 region.
#
# Teacher-forced training never asks for that restoring force: every training
# state is a real MD frame, already on the manifold, so "stay where you are" is
# never wrong. The standard fix (Sanchez-Gonzalez et al., "Learning to Simulate",
# 2020) is to corrupt the input state and make the target absorb the correction:
#
#     pos_t  <- pos_t + delta          (delta ~ N(0, sigma^2), sigma ~ U(0, STATE_NOISE_NM))
#     dpos   <- dpos  - delta          (so the model must undo delta to land on truth)
#
# Now the model is explicitly trained to walk back onto the data manifold from
# off it, which is exactly the signal the rollout needs and teacher forcing hides.
# 0 disables the augmentation.
STATE_NOISE_NM = 0.02      # upper end of the noise ramp; MD's own one-step dpos std is ~0.06

# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

BATCH_SIZE    = 32
LR            = 4e-4
WEIGHT_DECAY  = 1e-6
EPOCHS        = 100        # the 40-epoch run was still descending when it stopped
GRAD_CLIP     = 1.0
STEPS_PER_EPOCH = 400      # random transitions drawn per epoch (None = full pass)
SEED          = 0

# ──────────────────────────────────────────────────────────────
# Rollout / evaluation
# ──────────────────────────────────────────────────────────────

ROLLOUT_FRAMES = 300       # the 300-frame trajectory Stage 3 must produce
BOND_FIX_ITERS = 3         # SHAKE-like bond-length projection per rollout step (0 = off)
BOND_TOL_NM    = 0.02      # tolerance before the projection kicks in

KDE_GRID = 64              # Ramachandran / KL histogram resolution
KL_BINS  = 36              # 1-D marginal histogram bins (10 deg each)


def ensure_dirs():
    """Create every output directory Stage 3 writes into."""
    for d in (CACHE_DIR, TRANSITION_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
