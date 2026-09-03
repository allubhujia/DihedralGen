"""
seeded_model.py — Stage 2b: a Stage-2 generator that can be SEEDED with a frame.

WHAT THIS ADDS TO STAGE 2
-------------------------
The original TrajectorySeqModel (train_trajectory.py, untouched) maps a peptide
identity straight to its *equilibrium* phi/psi distribution. It has no starting
structure and no time axis, so "given frame 50, what does frame 100 look like?"
is a question it cannot be asked.

SeededTrajectoryModel answers exactly that. It takes the backbone state at a seed
frame and generates the window that follows it, so it models the CONDITIONAL
distribution p(state at t+k | state at t, molecule) instead of the marginal.

THREE LIMITS OF THE ORIGINAL THIS ALSO LIFTS
--------------------------------------------
1. Frame count was welded to 300 by a [1, 300, d] positional parameter. Here the
   table is allocated to max_frames and sliced, so any length up to that works.
2. Training only ever saw the first 300 frames of a 9800-frame trajectory.
   The window sampler in train_seeded.py draws from the whole thing.
3. There was no notion of temporal roughness, so nothing stopped the generated
   frames from being an unordered cloud. The delta and ACF loss terms below
   penalise that directly.

ON NOVELTY, STATED PLAINLY
--------------------------
Every line here is written for this repository and copied from nothing. The
*idea* of conditioning a sequence generator on an initial state is standard
practice, and claiming otherwise would be false. What is specific to this work is
the combination: seed-anchored generation supervised by a Sliced-Wasserstein term
on states, a second SW term on frame-to-frame increments, and an explicit
circular-autocorrelation match - a loss built around the exact failure modes
measured on this dataset (mode collapse under MSE, and a decorrelation curve that
falls off far faster than the real dynamics).
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
# Angle helpers
# ──────────────────────────────────────────────────────────────

def project_to_circle(raw, eps=1e-6):
    """Force each (sin, cos) pair onto the unit circle.

    raw: [..., 4] laid out as (sin_phi, cos_phi, sin_psi, cos_psi).

    Without this the network can shrink a pair toward the origin, where atan2
    becomes numerically meaningless and the recovered angle is essentially
    random - the documented collapse mode of the earlier regression attempt.
    """
    phi = raw[..., 0:2]
    psi = raw[..., 2:4]
    phi = phi / phi.norm(dim=-1, keepdim=True).clamp(min=eps)
    psi = psi / psi.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.cat([phi, psi], dim=-1)


def sincos_to_radians(sincos):
    """[..., 4] unit-circle pairs -> (phi, psi) in radians."""
    return (torch.atan2(sincos[..., 0], sincos[..., 1]),
            torch.atan2(sincos[..., 2], sincos[..., 3]))


def radians_to_sincos(phi, psi):
    """(phi, psi) in radians -> [..., 4] unit-circle pairs."""
    return torch.stack([torch.sin(phi), torch.cos(phi),
                        torch.sin(psi), torch.cos(psi)], dim=-1)


# ──────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────

class SeededTrajectoryModel(nn.Module):
    """Transformer-over-frames conditioned on a molecule AND a seed state.

    Every output frame k is a token built from four additive parts:

        frame_pos[k]        where in the window this frame sits
        mol_proj(z)         which peptide this is
        seed_proj(s)        the backbone state we were seeded from
        noise_proj(eps_k)   per-frame latent, so samples differ

    Self-attention across frames then lets the window organise itself, and a
    unit-circle head keeps every emitted angle well defined.

    Args:
        mol_dim:    width of the molecule embedding (64, to match Stage 1
                    pooled node embeddings)
        max_frames: longest window this checkpoint can ever emit
    """

    SEED_DIM = 4      # (sin_phi, cos_phi, sin_psi, cos_psi) at the seed frame

    def __init__(self, mol_dim=64, d_model=128, nhead=4, num_layers=4,
                 dim_feedforward=256, max_frames=512, dropout=0.1, noise_dim=8):
        super().__init__()
        self.max_frames = max_frames
        self.d_model = d_model
        self.noise_dim = noise_dim

        self.mol_proj = nn.Sequential(
            nn.Linear(mol_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        # The seed gets its own encoder rather than being concatenated onto the
        # molecule vector: conditioning strength on the two is then independent,
        # and molecule identity cannot drown out where the window was started.
        self.seed_proj = nn.Sequential(
            nn.Linear(self.SEED_DIM, d_model), nn.SiLU(), nn.Linear(d_model, d_model))

        self.frame_pos = nn.Parameter(torch.randn(1, max_frames, d_model) * 0.02)
        self.noise_proj = nn.Linear(noise_dim, d_model) if noise_dim > 0 else None

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4))

    def forward(self, mol_z, seed_state, n_frames, noise=None):
        """
        mol_z:      [B, mol_dim]
        seed_state: [B, 4] unit-circle backbone state at the seed frame
        n_frames:   how many frames to emit (<= max_frames)
        Returns:    [B, n_frames, 4] on the unit circle
        """
        if n_frames > self.max_frames:
            raise ValueError(
                "asked for {} frames but this checkpoint was built for at most {}"
                .format(n_frames, self.max_frames))

        B = mol_z.size(0)
        tokens = self.frame_pos[:, :n_frames].expand(B, -1, -1)
        tokens = tokens + self.mol_proj(mol_z).unsqueeze(1)
        tokens = tokens + self.seed_proj(seed_state).unsqueeze(1)

        if self.noise_proj is not None:
            if noise is None:
                noise = torch.randn(B, n_frames, self.noise_dim, device=mol_z.device)
            tokens = tokens + self.noise_proj(noise)

        return project_to_circle(self.head(self.transformer(tokens)))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ──────────────────────────────────────────────────────────────
# Loss terms
# ──────────────────────────────────────────────────────────────

def sliced_wasserstein(pred, target, num_proj=64):
    """Sliced-Wasserstein-1 between two point sets, per batch element.

    pred, target: [B, F, D]. Projects both onto `num_proj` random directions in
    R^D, sorts each projection and averages the absolute gap. Matching every 1-D
    projection matches the full joint distribution, and because it compares
    sorted sets rather than paired frames it is mode-covering - it cannot be
    minimised by collapsing to the mean the way an MSE can.
    """
    D = pred.size(-1)
    dirs = torch.randn(D, num_proj, device=pred.device)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    p = torch.sort(pred @ dirs, dim=1).values
    t = torch.sort(target @ dirs, dim=1).values
    return (p - t).abs().mean()


def circular_acf(sincos, max_lag):
    """Circular autocorrelation of phi and psi, differentiable.

    For an angle theta represented as (sin, cos),
        C(k) = mean_t cos(theta_t - theta_{t+k})
             = mean_t [ sin_t sin_{t+k} + cos_t cos_{t+k} ]
    which needs no atan2 and so stays smooth everywhere.

    sincos: [B, F, 4]  ->  [B, max_lag + 1, 2]  (phi and psi curves)
    """
    n_frames = sincos.size(1)
    max_lag = min(max_lag, n_frames - 1)
    curves = []
    for k in range(max_lag + 1):
        a, b = sincos[:, :n_frames - k], sincos[:, k:]
        phi = (a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1]).mean(dim=1)
        psi = (a[..., 2] * b[..., 2] + a[..., 3] * b[..., 3]).mean(dim=1)
        curves.append(torch.stack([phi, psi], dim=-1))
    return torch.stack(curves, dim=1)


def seeded_loss(pred, target, seed_state, w_sw=1.0, w_anchor=1.0,
                w_delta=0.5, w_acf=0.5, acf_lag=32, num_proj=64):
    """Composite objective. Each term exists to stop a specific failure.

      sw      distribution of states matches the real window
      anchor  frame 0 actually equals the seed we were given - without this
              nothing forces the model to honour its conditioning, and it
              quietly degenerates back into the unconditional Stage 2
      delta   distribution of frame-to-frame CHANGES matches, which controls how
              rough the path is; a set-only loss is permutation-invariant and so
              is blind to this
      acf     decorrelation curve matches, pinning the timescale rather than
              just the step size

    Returns (total, parts_dict).
    """
    l_sw = sliced_wasserstein(pred, target, num_proj)
    l_anchor = ((pred[:, 0] - seed_state) ** 2).sum(-1).mean()
    l_delta = sliced_wasserstein(pred[:, 1:] - pred[:, :-1],
                                 target[:, 1:] - target[:, :-1], num_proj)
    l_acf = (circular_acf(pred, acf_lag) - circular_acf(target, acf_lag)).abs().mean()

    total = w_sw * l_sw + w_anchor * l_anchor + w_delta * l_delta + w_acf * l_acf
    return total, {"sw": l_sw.item(), "anchor": l_anchor.item(),
                   "delta": l_delta.item(), "acf": l_acf.item()}
