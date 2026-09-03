"""
flow.py — conditional flow matching for the Stage-3 propagator.

WHY NOT JUST REGRESS THE NEXT FRAME
-----------------------------------
The saved frames are 5 ps apart. Over that lag the transition is genuinely
stochastic: the same (pos_t, vel_t) can be followed by many different next
frames, and the velocity carries essentially no information about which one
(measured corr with the frame-to-frame displacement: -0.005).

A model trained with MSE on the next frame therefore learns the *conditional
mean* of that spread. Rolling a conditional mean forward 300 times damps out
all motion and lands on a static average structure — the Ramachandran plot
collapses to one blob. This is the same failure Stage 2 hit with per-frame
regression, for the same reason.

WHAT THIS DOES INSTEAD
----------------------
Learn the whole conditional *distribution* with rectified flow. Define a
straight path from Gaussian noise to the true target,

    Y_tau = (1 - tau) * eps + tau * Y ,     eps ~ N(0, sigma^2),  tau ~ U(0, 1)

whose constant-in-tau velocity is simply (Y - eps). Train the network to
regress that velocity, then sample by integrating

    dY/dtau = u_theta(Y_tau, tau ; pos_t, z)

from tau = 0 to 1 starting at fresh noise. The regression target is still an
MSE, so training is as stable as regression — but because the target depends on
eps, the learned field transports the *entire* noise distribution onto the
*entire* conditional distribution. Different noise draws give different
plausible next frames, so the rollout keeps moving and keeps hopping basins.

WHAT Y CONTAINS
---------------
Two vector channels per atom, so the propagator carries the full (x, v) state:

    channel 0 = dpos    (nm)     displacement to the next frame
    channel 1 = vel_tp1 (nm/ps)  velocity at the next frame

The two are NOT equally weighted in the loss. vel_{t+1} is close to unlearnable
at a 5 ps lag (momentum has fully decorrelated), so an equal average spends half
the gradient on irreducible noise and buries the displacement signal — which is
what happened in the first run. config.CHANNEL_LOSS_WEIGHTS down-weights the
velocity channel instead of removing it: the model still predicts velocity and
still rolls it forward, it just no longer gets an equal vote in the objective.

Targets are divided by a PER-ELEMENT standard deviation before any of this, not
a single global one. Hydrogen moves ~2x further than carbon per frame (and is
several times faster, since thermal speed goes as 1/sqrt(mass)), so one global
scale hands hydrogen several times the squared amplitude and lets it dominate
the gradient purely by amplitude. Losses are additionally mass-weighted, so the
heavy atoms that define the molecule's conformation are not outvoted by the
hydrogens bolted onto them.

STATE-NOISE AUGMENTATION
------------------------
`flow_matching_loss` corrupts pos_t and subtracts the same corruption from the
target. That is what teaches the model to walk back onto the data manifold, and
it is the single change that fixes the weak-restoring-force failure described
in config.STATE_NOISE_NM. Sampling is unaffected.
"""

import numpy as np
import torch

from temporal_dynamics import config


# ──────────────────────────────────────────────────────────────
# Channel bookkeeping
# ──────────────────────────────────────────────────────────────

def num_channels() -> int:
    """Vector channels in the flow target: dpos, plus vel_{t+1} if enabled."""
    return 2 if config.PREDICT_VELOCITY else 1


# ──────────────────────────────────────────────────────────────
# Normalisation
# ──────────────────────────────────────────────────────────────

def load_scales(device="cpu"):
    """Per-element target scales as a [NUM_ELEMENTS, C] tensor.

    Column 0 is the displacement std for that element (nm); column 1, when
    velocity is enabled, is the global velocity std. Indexing this by
    `atom_type` gives every atom its own normaliser.
    """
    s = np.load(config.STATS_CACHE, allow_pickle=True)
    global_std = float(s["dpos_std"])

    if config.PER_ELEMENT_SCALE and "dpos_std_elem" in s.files:
        dpos_col = np.asarray(s["dpos_std_elem"], dtype=np.float32)
    else:
        # Older caches predate the per-element stats; fall back to the global std.
        dpos_col = np.full(config.NUM_ELEMENTS, global_std, dtype=np.float32)

    cols = [dpos_col]
    if config.PREDICT_VELOCITY:
        if config.PER_ELEMENT_SCALE and "vel_std_elem" in s.files:
            cols.append(np.asarray(s["vel_std_elem"], dtype=np.float32))
        else:
            cols.append(np.full(config.NUM_ELEMENTS, float(s["vel_std"]), dtype=np.float32))

    return torch.tensor(np.stack(cols, axis=1), dtype=torch.float32, device=device)


def per_atom_scales(scales, atom_type):
    """[NUM_ELEMENTS, C] -> [N, C, 1], ready to broadcast over [N, C, 3]."""
    return scales[atom_type].unsqueeze(-1)


def loss_weights(device="cpu"):
    """Per-element loss weight, mass**LOSS_MASS_POWER (power 0 = uniform)."""
    m = torch.tensor(config.ELEMENT_MASS, dtype=torch.float32, device=device)
    return m ** config.LOSS_MASS_POWER


def channel_weights(device="cpu"):
    """Relative weight of each target channel in the loss — see config."""
    return torch.tensor(config.CHANNEL_LOSS_WEIGHTS[:num_channels()],
                        dtype=torch.float32, device=device)


def normalise(dpos, vel_tp1, scales, atom_type):
    """Physical units -> unit-scale Y [N, C, 3]."""
    chans = [dpos, vel_tp1] if config.PREDICT_VELOCITY else [dpos]
    return torch.stack(chans, dim=1) / per_atom_scales(scales, atom_type)


def denormalise(Y, scales, atom_type):
    """Y [N, C, 3] -> (dpos [N,3], vel_tp1 [N,3] or None) in physical units."""
    Y = Y * per_atom_scales(scales, atom_type)
    return Y[:, 0], (Y[:, 1] if config.PREDICT_VELOCITY else None)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def centre_per_graph(x, batch_vec, num_graphs):
    """Subtract each graph's own centroid from its rows of `x` [N, 3]."""
    counts = torch.zeros(num_graphs, device=x.device, dtype=x.dtype)
    counts.index_add_(0, batch_vec, torch.ones(x.shape[0], device=x.device, dtype=x.dtype))
    sums = torch.zeros(num_graphs, x.shape[1], device=x.device, dtype=x.dtype)
    sums.index_add_(0, batch_vec, x)
    return x - (sums / counts.unsqueeze(-1))[batch_vec]


def _vel_input(batch):
    """vel_t if the model consumes it, otherwise a zero stand-in of the right shape."""
    if config.USE_VELOCITY_INPUT and "vel_t" in batch:
        return batch["vel_t"]
    return None


# ──────────────────────────────────────────────────────────────
# Training objective
# ──────────────────────────────────────────────────────────────

def flow_matching_loss(model, batch, scales, generator=None, weights=None,
                       state_noise=None):
    """Rectified-flow regression loss for one batch.

    Args:
        state_noise: std ceiling for the input-state corruption, in nm. Each
            graph draws its own sigma ~ U(0, state_noise), so the model sees a
            whole ramp of corruption levels rather than one. Defaults to
            config.STATE_NOISE_NM; pass 0.0 to disable (validation does).

    Returns:
        loss, {"pos": .., "vel": ..} per-channel MSEs
    """
    device = batch["pos_t"].device
    pos_t = batch["pos_t"]
    dpos = batch["dpos"]
    sigma_max = config.STATE_NOISE_NM if state_noise is None else state_noise

    # --- state-noise augmentation -----------------------------------------
    # Corrupt the input state and hand the correction to the target, so the
    # model is trained to steer back onto the manifold rather than only to
    # continue along it. Without this the rollout has no restoring force.
    if sigma_max > 0:
        sig = torch.rand(batch["num_graphs"], device=device, generator=generator) * sigma_max
        delta = torch.randn(pos_t.shape, device=device, generator=generator) \
            * sig[batch["batch"]].unsqueeze(-1)
        # A centre-of-mass shift is re-centred away every rollout step, so asking
        # the model to undo one would be asking for a translation that never
        # reaches it. Keep the corruption internal.
        delta = centre_per_graph(delta, batch["batch"], batch["num_graphs"])
        pos_t = pos_t + delta
        dpos = dpos - delta

    Y = normalise(dpos, batch.get("vel_tp1"), scales, batch["atom_type"])   # [N, C, 3]

    # One tau per GRAPH, not per atom: the whole molecule must sit at the same
    # point along the path, otherwise the field is not integrable at sampling time.
    tau_g = torch.rand(batch["num_graphs"], device=device, generator=generator)
    tau_n = tau_g[batch["batch"]].view(-1, 1, 1)

    eps = torch.randn(Y.shape, device=device, generator=generator) * config.FLOW_SIGMA
    Y_tau = (1.0 - tau_n) * eps + tau_n * Y
    target = Y - eps                                                # constant along the path

    u = model(Y_tau, tau_g, pos_t, _vel_input(batch), batch["atom_type"],
              batch["z"], batch["edge_index"], batch["bond_flag"], batch["batch"])

    sq = ((u - target) ** 2).mean(dim=2)                            # [N, C]

    # Mass weighting, renormalised over the atoms actually in this batch so the
    # loss stays on the same scale as the unweighted version.
    if weights is None:
        weights = loss_weights(device)
    w = weights[batch["atom_type"]]
    w = w / w.mean()
    per_channel = (sq * w.unsqueeze(-1)).mean(dim=0)                # [C]

    # Channel weighting. A plain .mean() here would hand the velocity channel an
    # equal vote, and since vel_{t+1} is close to unlearnable at this lag that
    # vote is spent almost entirely on irreducible noise — which is what buried
    # the displacement signal in the first run. Weights are renormalised so the
    # reported loss stays comparable to the unweighted version.
    cw = channel_weights(device)
    loss = (per_channel * cw).sum() / cw.sum()

    parts = {"pos": per_channel[0].item(),
             "vel": per_channel[1].item() if config.PREDICT_VELOCITY else 0.0}
    return loss, parts


# ──────────────────────────────────────────────────────────────
# Sampling
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_transition(model, state, scales, steps=None, temperature=None,
                      generator=None):
    """Draw one next state by integrating the learned field from noise.

    Args:
        state: dict with pos_t, atom_type, z, edge_index, bond_flag, batch,
               num_graphs (and vel_t if config.USE_VELOCITY_INPUT).
        steps: Euler steps (default config.FLOW_STEPS). More steps = more
               faithful transport; 20 is ample for a straight rectified-flow path.
        temperature: scales the initial noise. >1 broadens the sampled
               distribution, <1 sharpens it toward the conditional mean.

    Returns:
        dpos [N, 3], vel_tp1 [N, 3] or None — in physical units (nm, nm/ps).
    """
    steps = steps or config.FLOW_STEPS
    temp = config.FLOW_TEMP if temperature is None else temperature
    device = state["pos_t"].device

    n = state["pos_t"].shape[0]
    Y = torch.randn((n, num_channels(), 3), device=device, generator=generator) \
        * config.FLOW_SIGMA * temp

    vel_in = _vel_input(state)
    dt = 1.0 / steps
    for k in range(steps):
        tau = torch.full((state["num_graphs"],), k * dt, device=device)
        u = model(Y, tau, state["pos_t"], vel_in, state["atom_type"],
                  state["z"], state["edge_index"], state["bond_flag"], state["batch"])
        Y = Y + dt * u

    return denormalise(Y, scales, state["atom_type"])


# ──────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def equivariance_error(model, batch, scales, seed=0):
    """Rotate the input, run the model, and check the output rotated too.

    Returns the relative error ||R u(x) - u(R x)|| / ||u(x)||. Anything above
    ~1e-4 means a coordinate leaked into an invariant pathway somewhere.
    """
    device = batch["pos_t"].device
    g = torch.Generator(device=device).manual_seed(seed)

    # Random proper rotation via QR.
    A = torch.randn(3, 3, generator=g, device=device)
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R))
    if torch.det(Q) < 0:
        Q[:, 0] *= -1

    Y = normalise(batch["dpos"], batch.get("vel_tp1"), scales, batch["atom_type"])
    tau = torch.rand(batch["num_graphs"], device=device, generator=g)
    vel = _vel_input(batch)

    def run(Yv, pos, v):
        return model(Yv, tau, pos, v, batch["atom_type"], batch["z"],
                     batch["edge_index"], batch["bond_flag"], batch["batch"])

    u_plain = run(Y, batch["pos_t"], vel)
    u_rot = run(Y @ Q.T, batch["pos_t"] @ Q.T, None if vel is None else vel @ Q.T)

    return ((u_plain @ Q.T - u_rot).norm() / u_plain.norm().clamp(min=1e-12)).item()
