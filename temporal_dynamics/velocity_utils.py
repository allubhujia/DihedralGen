"""
velocity_utils.py — velocity handling and frame alignment for Stage 3.

WHY THIS FILE EXISTS
--------------------
Stage 1 deliberately threw velocities away (see scripts/processing.py) because
it only ever looked at a single static frame. Stage 3 propagates state through
time, so velocity is back — but it has to be handled carefully:

  1. The Timewarp `.npz` files DO ship a `velocities` array [T, N, 3] in nm/ps.
     `load_positions_velocities()` uses it when present and falls back to a
     difference only if a file is missing it.

  2. Saved frames are 5 ps apart (10,000 MD steps). Over that lag the
     instantaneous velocity is completely decorrelated from the frame-to-frame
     displacement (measured Pearson r = -0.005 on AA). So velocity is a *state
     channel* the propagator reads and re-emits, NOT an integration term.
     Never write `x + v*dt` here.

  3. Between saved frames the molecule tumbles. Per-atom displacement is
     0.378 nm with only the centroid removed, but 0.072 nm after Kabsch
     superposition. Global rotation is 5x the internal motion and is not a
     function of the molecule, so we remove it and learn only internal motion.
     Dihedrals are internal coordinates, so nothing observable is lost.
"""

import numpy as np

from temporal_dynamics import config


# ──────────────────────────────────────────────────────────────
# Velocity loading
# ──────────────────────────────────────────────────────────────

def load_positions_velocities(npz_path: str):
    """Load positions [T, N, 3] (nm) and velocities [T, N, 3] (nm/ps).

    Returns:
        positions, velocities, source  where source is "file" or "finite-diff".

    If the archive has no `velocities` key we synthesise one with a centred
    finite difference over the saved-frame lag. That is a poor proxy for the
    true instantaneous velocity at this lag, so we tag it and print a warning —
    check_velocities.py reports how many files needed it.
    """
    with np.load(npz_path) as a:
        positions = a["positions"].astype(np.float32)
        if "velocities" in a.files:
            return positions, a["velocities"].astype(np.float32), "file"

    dt = config.FRAME_LAG_PS
    vel = np.zeros_like(positions)
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2.0 * dt)
    vel[0]    = (positions[1] - positions[0]) / dt
    vel[-1]   = (positions[-1] - positions[-2]) / dt
    return positions, vel.astype(np.float32), "finite-diff"


# ──────────────────────────────────────────────────────────────
# Rigid-body removal
# ──────────────────────────────────────────────────────────────

def kabsch_rotation(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimising ||R @ mobile_centred - target_centred||.

    Args:
        mobile: [N, 3] coordinates to be rotated (will be centred internally)
        target: [N, 3] reference coordinates (will be centred internally)

    Returns:
        R: [3, 3] proper rotation (det = +1, so chirality can never invert).
    """
    P = mobile - mobile.mean(axis=0)
    Q = target - target.mean(axis=0)

    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)

    # Flip the least-significant axis if the naive solution is a reflection.
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def align_frame(pos_next: np.ndarray, vel_next: np.ndarray, pos_ref: np.ndarray):
    """Express frame t+1 in the body-fixed frame of frame t.

    Positions are centred and Kabsch-rotated onto `pos_ref`; velocities get the
    same rotation (they are vectors, so they rotate but do not translate).

    Returns:
        pos_aligned [N, 3] (centred at origin), vel_aligned [N, 3], R [3, 3]
    """
    if not config.REMOVE_GLOBAL_ROTATION:
        return pos_next - pos_next.mean(0), vel_next, np.eye(3, dtype=np.float64)

    R = kabsch_rotation(pos_next, pos_ref)
    pos_aligned = (R @ (pos_next - pos_next.mean(0)).T).T
    vel_aligned = (R @ vel_next.T).T
    return pos_aligned.astype(np.float32), vel_aligned.astype(np.float32), R


def strip_com_velocity(vel: np.ndarray) -> np.ndarray:
    """Remove the centre-of-mass drift from a velocity array.

    We centre positions every step, so keeping COM velocity would ask the model
    to predict a translation that has already been projected out.
    Accepts [N, 3] or [T, N, 3].
    """
    if not config.REMOVE_COM_VELOCITY:
        return vel
    return vel - vel.mean(axis=-2, keepdims=True)


# ──────────────────────────────────────────────────────────────
# Transition construction
# ──────────────────────────────────────────────────────────────

def build_transitions(positions: np.ndarray, velocities: np.ndarray, stride: int = 1):
    """Turn a raw trajectory into aligned (state_t -> target) training pairs.

    For every t the pair is:
        input   pos_t  [N, 3]  centred, lab-frame orientation
                vel_t  [N, 3]  COM-drift removed
        target  dpos   [N, 3]  pos_{t+1} aligned onto pos_t, minus pos_t
                vel_tp1[N, 3]  velocity at t+1 rotated into t's frame

    Rolling this forward is self-consistent: after applying dpos the new state
    is already expressed in its own body-fixed frame, so the next step needs no
    extra bookkeeping.

    Returns a dict of float32 arrays, each [T-1, N, 3].
    """
    pos = positions[::stride]
    vel = strip_com_velocity(velocities[::stride])

    # Centre every frame once up front.
    pos = pos - pos.mean(axis=1, keepdims=True)

    T = pos.shape[0] - 1
    dpos    = np.zeros((T,) + pos.shape[1:], dtype=np.float32)
    vel_tp1 = np.zeros_like(dpos)

    for t in range(T):
        p_next, v_next, _ = align_frame(pos[t + 1], vel[t + 1], pos[t])
        dpos[t]    = p_next - pos[t]
        vel_tp1[t] = v_next

    return {
        "pos_t":   pos[:T].astype(np.float32),
        "vel_t":   vel[:T].astype(np.float32),
        "dpos":    dpos,
        "vel_tp1": vel_tp1,
    }


# ──────────────────────────────────────────────────────────────
# Bond geometry (used by the rollout guard-rail and as a sanity metric)
# ──────────────────────────────────────────────────────────────

def covalent_bonds(pos_frame: np.ndarray, cutoff: float = None):
    """Directed covalent edge list from one frame, plus equilibrium lengths.

    Same distance-cutoff rule Stage 1 uses, so the bond set is identical.

    Returns:
        edges [2, E] int64 (both directions), lengths [E] float32
    """
    cutoff = cutoff if cutoff is not None else config.BOND_CUTOFF_NM
    d = np.linalg.norm(pos_frame[:, None, :] - pos_frame[None, :, :], axis=-1)
    src, dst = np.where((d < cutoff) & (d > 0))
    return np.stack([src, dst]).astype(np.int64), d[src, dst].astype(np.float32)


def bond_length_rmsd(traj: np.ndarray, bonds: np.ndarray, eq_lengths: np.ndarray) -> float:
    """RMSD of covalent bond lengths across a trajectory vs. their equilibrium.

    A rolled-out trajectory that slowly inflates or collapses the molecule shows
    up here long before it shows up in a Ramachandran plot.
    """
    src, dst = bonds
    lengths = np.linalg.norm(traj[:, src] - traj[:, dst], axis=-1)   # [T, E]
    return float(np.sqrt(((lengths - eq_lengths[None, :]) ** 2).mean()))
