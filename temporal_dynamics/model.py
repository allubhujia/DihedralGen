"""
model.py — the equivariant vector field for the Stage-3 propagator.

WHAT IT COMPUTES
----------------
A rotation-equivariant function

    u_theta( Y_tau, tau ; pos_t, vel_t, z )  ->  [N, 2, 3]

where Y_tau is the partially-noised target (channel 0 = displacement,
channel 1 = next-frame velocity). flow.py wraps this into the flow-matching
objective; on its own this module is just a network.

WHY EQUIVARIANCE MATTERS HERE
-----------------------------
We stripped global rotation out of the data (velocity_utils.align_frame), so
the network never has to *predict* a rotation. But it still has to *respect*
one: if the whole input is rotated by R, the predicted displacement and
velocity must rotate by R too. Baking that in means the model does not have to
spend capacity (or training data) rediscovering it.

HOW IT IS BUILT
---------------
A PaiNN/EGNN hybrid carrying two kinds of features per atom:

  h_i in R^H        invariant scalars  (rotate the molecule -> unchanged)
  V_i in R^{C x 3}  equivariant vectors (rotate the molecule -> they rotate)

Only three operations are allowed on V, and all three are equivariant:
  * linear mixing across the channel axis      V <- W V
  * scaling by an invariant scalar             V <- s(h) * V
  * adding a sum of relative-position vectors  V <- V + sum_j c_ij (x_i - x_j)

Scalars read vectors through rotation-invariant quantities only: channel norms
||V_i^c|| and projections <V_i^c, r_hat_ij> onto edge directions. Nothing ever
leaks a raw coordinate into h, which is what would silently break equivariance.
"""

import math

import torch
import torch.nn as nn

from temporal_dynamics import config


# ──────────────────────────────────────────────────────────────
# Small building blocks
# ──────────────────────────────────────────────────────────────

def mlp(*dims, out_act=False):
    """[Linear, SiLU, Linear, ...] over the given widths."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or out_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Sum `src` rows into `n` buckets given by `index` (works for any trailing shape)."""
    out = torch.zeros((n,) + src.shape[1:], dtype=src.dtype, device=src.device)
    return out.index_add_(0, index, src)


class GaussianRBF(nn.Module):
    """Expand a scalar distance into `num` smooth Gaussian bumps.

    A raw distance is a poor network input — it is one number with an awkward
    scale. An RBF expansion turns it into a soft one-hot over distance shells,
    which lets the first linear layer learn a distance-dependent response
    directly instead of having to synthesise one.
    """

    def __init__(self, num=config.NUM_RBF, cutoff=config.RBF_CUTOFF_NM):
        super().__init__()
        self.register_buffer("centres", torch.linspace(0.0, cutoff, num))
        self.gamma = (num / cutoff) ** 2

    def forward(self, d):
        return torch.exp(-self.gamma * (d.unsqueeze(-1) - self.centres) ** 2)


class SinusoidalTime(nn.Module):
    """Transformer-style sinusoidal embedding of the flow time tau in [0, 1]."""

    def __init__(self, dim=config.TIME_EMB_DIM):
        super().__init__()
        self.dim = dim

    def forward(self, tau):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=tau.device, dtype=tau.dtype) / half
        )
        ang = tau.unsqueeze(-1) * freqs * 1000.0
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


# ──────────────────────────────────────────────────────────────
# Message-passing layer
# ──────────────────────────────────────────────────────────────

class EquivariantLayer(nn.Module):
    """One round of scalar + vector message passing."""

    def __init__(self, hidden, vec_ch, edge_in, cond_dim):
        super().__init__()
        self.vec_ch = vec_ch

        # FiLM: the frozen embedding z and the flow time tau modulate every layer.
        self.film = nn.Linear(cond_dim, 2 * hidden)

        # Invariant view of the vector channels: their per-channel norms.
        self.norm_inject = mlp(vec_ch, hidden, hidden)

        # Edge message. edge_in already contains the RBF distance, the bond flag
        # and the invariant projections <V_i, r_hat>, <V_j, r_hat>.
        self.msg = mlp(2 * hidden + edge_in, hidden, hidden, out_act=True)

        self.h_update = mlp(2 * hidden, hidden, hidden)

        # Coefficients that build new vectors out of edge directions.
        self.vec_coef = nn.Linear(hidden, vec_ch)
        # Channel mixing + invariant gating of the existing vectors.
        self.vec_mix = nn.Linear(vec_ch, vec_ch, bias=False)
        self.vec_gate = mlp(hidden, hidden, vec_ch)

        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, V, edge_index, edge_attr, r_hat, cond):
        src, dst = edge_index
        n = h.shape[0]

        # --- conditioning -------------------------------------------------
        gamma, beta = self.film(cond).chunk(2, dim=-1)
        h = h * (1.0 + gamma) + beta

        # --- let scalars see the vectors (invariantly) --------------------
        h = h + self.norm_inject(V.norm(dim=-1))

        # --- edge features: add directional projections of both endpoints --
        proj_i = torch.einsum("ecd,ed->ec", V[src], r_hat)    # [E, C]
        proj_j = torch.einsum("ecd,ed->ec", V[dst], r_hat)    # [E, C]
        e_feat = torch.cat([edge_attr, proj_i, proj_j], dim=-1)

        m = self.msg(torch.cat([h[src], h[dst], e_feat], dim=-1))   # [E, H]
        agg = scatter_sum(m, dst, n)

        # --- scalar update -------------------------------------------------
        h = self.norm(h + self.h_update(torch.cat([h, agg], dim=-1)))

        # --- vector update (all three terms are equivariant) ---------------
        coef = self.vec_coef(m)                                     # [E, C]
        dV = scatter_sum(coef.unsqueeze(-1) * r_hat.unsqueeze(1), dst, n)  # [N, C, 3]

        mixed = self.vec_mix(V.transpose(1, 2)).transpose(1, 2)     # [N, C, 3]
        gate = torch.tanh(self.vec_gate(h)).unsqueeze(-1)           # [N, C, 1]

        V = V + dV + mixed * gate
        return h, V


# ──────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────

class EquivariantPropagator(nn.Module):
    """Rotation-equivariant vector field over the flow target.

    Channel counts follow config rather than being fixed, because the velocity
    channel is switchable (config.PREDICT_VELOCITY / USE_VELOCITY_INPUT):

        out channels = 1 (dpos)  or 2 (dpos, vel_{t+1})
        in  channels = out channels, plus 1 if vel_t is fed in as context

    In the default single-channel setup the only input vector is the noised
    displacement itself; all the geometry reaches the network through pos_t, via
    the edge distances and directions.
    """

    def __init__(self,
                 hidden=config.HIDDEN_DIM,
                 vec_ch=config.VEC_CHANNELS,
                 num_layers=config.NUM_LAYERS,
                 cond_dim=config.COND_DIM,
                 num_elements=config.NUM_ELEMENTS,
                 num_out_vec=None,
                 use_vel_input=None):
        super().__init__()
        self.vec_ch = vec_ch
        self.num_out_vec = (2 if config.PREDICT_VELOCITY else 1) \
            if num_out_vec is None else num_out_vec
        self.use_vel_input = config.USE_VELOCITY_INPUT if use_vel_input is None \
            else use_vel_input
        self.num_in_vec = self.num_out_vec + (1 if self.use_vel_input else 0)

        self.rbf = GaussianRBF()
        self.time_emb = SinusoidalTime()

        # z (frozen, 32-d) + tau embedding -> per-graph conditioning vector.
        self.cond_mlp = mlp(cond_dim + config.TIME_EMB_DIM, hidden, hidden, out_act=True)

        # Scalars start from the element identity plus the norms of the inputs.
        self.h_init = mlp(num_elements + self.num_in_vec, hidden, hidden)
        # Vectors start as a learned mix of the input vector channels.
        self.v_init = nn.Linear(self.num_in_vec, vec_ch, bias=False)

        edge_in = config.NUM_RBF + 1 + 2 * vec_ch   # rbf + bond flag + 2 projections
        self.layers = nn.ModuleList([
            EquivariantLayer(hidden, vec_ch, edge_in, hidden) for _ in range(num_layers)
        ])

        self.out_gate = mlp(hidden, hidden, self.num_out_vec)
        self.out_mix = nn.Linear(vec_ch, self.num_out_vec, bias=False)
        # Start near zero so the first training steps predict ~no motion rather
        # than a large random field.
        nn.init.zeros_(self.out_mix.weight)

    def forward(self, y_tau, tau, pos_t, vel_t, atom_type, z, edge_index, bond_flag, batch):
        """
        Args:
            y_tau      [N, C, 3]  noised target (channel 0 dpos, channel 1 vel_tp1)
            tau        [B]        flow time per graph, in [0, 1]
            pos_t      [N, 3]     centred positions at t
            vel_t      [N, 3]     velocities at t, or None when not in use
            atom_type  [N]        element index
            z          [B, 32]    frozen Stage-1 embeddings
            edge_index [2, E]
            bond_flag  [E]
            batch      [N]        graph id per node

        Returns:
            u [N, C, 3] — equivariant vector field.
        """
        n = pos_t.shape[0]
        src, dst = edge_index

        # --- geometry: invariant distance, equivariant unit direction ------
        rel = pos_t[src] - pos_t[dst]
        dist = rel.norm(dim=-1)
        r_hat = rel / dist.clamp(min=1e-8).unsqueeze(-1)
        edge_attr = torch.cat([self.rbf(dist), bond_flag.unsqueeze(-1)], dim=-1)

        # --- stack the input vector channels -------------------------------
        if self.use_vel_input:
            if vel_t is None:
                raise ValueError("model was built with use_vel_input=True but vel_t is None")
            V_in = torch.cat([vel_t.unsqueeze(1), y_tau], dim=1)        # [N, 1+C, 3]
        else:
            V_in = y_tau                                                # [N, C, 3]

        onehot = torch.zeros(n, config.NUM_ELEMENTS, device=pos_t.device, dtype=pos_t.dtype)
        onehot.scatter_(1, atom_type.unsqueeze(-1), 1.0)

        h = self.h_init(torch.cat([onehot, V_in.norm(dim=-1)], dim=-1))
        V = self.v_init(V_in.transpose(1, 2)).transpose(1, 2)          # [N, C, 3]

        cond = self.cond_mlp(torch.cat([z, self.time_emb(tau)], dim=-1))[batch]   # [N, H]

        for layer in self.layers:
            h, V = layer(h, V, edge_index, edge_attr, r_hat, cond)

        # --- equivariant read-out ------------------------------------------
        u = self.out_mix(V.transpose(1, 2)).transpose(1, 2)            # [N, 2, 3]
        return u * torch.tanh(self.out_gate(h)).unsqueeze(-1)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
