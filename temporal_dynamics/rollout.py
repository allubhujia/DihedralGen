"""
rollout.py — autoregressive generation of a 300-frame trajectory.

Given the first real MD frame of a dipeptide (positions AND velocities), the
propagator is applied repeatedly:

    for t in 0 .. n_frames-1:
        dpos, vel_next = sample_transition(model | pos_t, vel_t, z)   # stochastic
        pos_{t+1} = centre(pos_t + dpos)
        vel_{t+1} = vel_next - mean(vel_next)

Each step draws fresh noise, so the trajectory is a genuine sample path rather
than a deterministic curve — repeated calls give different (but equally valid)
trajectories, exactly as re-running MD with a different random seed would.

By default the state carried forward is position only. Velocity was dropped
after it was measured to be unlearnable at this lag (config.PREDICT_VELOCITY);
the velocities written to the output file are then finite differences of the
generated positions, recorded for inspection rather than fed back in. With the
velocity channel switched on, `vel_next` is carried forward instead — but note
that even then velocity never integrates positions, because over a 5 ps lag
`x + v*dt` is meaningless.

BOND GUARD-RAIL
---------------
300 stochastic steps give small per-step biases 300 chances to accumulate, and
the usual way that shows up is a molecule that slowly inflates or collapses.
`project_bonds` runs a few SHAKE-style iterations that pull covalent bond
lengths back toward their equilibrium values whenever they drift past
BOND_TOL_NM. It touches only bonded pairs, so it cannot manufacture backbone
dihedrals — those are still entirely the model's output.

Run:
    python -m temporal_dynamics.rollout --peptide AC
    python -m temporal_dynamics.rollout --peptide AC --frames 300 --samples 3 --pdb
"""

import argparse
import os

import numpy as np
import torch

from temporal_dynamics import config, flow
from temporal_dynamics.model import EquivariantPropagator
from temporal_dynamics.velocity_utils import (
    covalent_bonds,
    load_positions_velocities,
    strip_com_velocity,
)


# ──────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────

def load_propagator(ckpt_path=None, device="cpu"):
    """Load best_propagator.pt, rebuilding the architecture it was saved with."""
    ckpt_path = ckpt_path or config.CKPT_PATH
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"No checkpoint at {ckpt_path}. Run `python -m temporal_dynamics.train` first."
        )
    ckpt = torch.load(ckpt_path, map_location=device)
    c = ckpt.get("config", {})
    model = EquivariantPropagator(
        hidden=c.get("hidden", config.HIDDEN_DIM),
        vec_ch=c.get("vec_ch", config.VEC_CHANNELS),
        num_layers=c.get("num_layers", config.NUM_LAYERS),
        cond_dim=c.get("cond_dim", config.COND_DIM),
        num_out_vec=c.get("num_out_vec"),
        use_vel_input=c.get("use_vel_input"),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, ckpt


def find_raw_files(peptide_id):
    """Locate (npz, pdb) for a peptide across all three splits."""
    for split in ("train", "val", "test"):
        npz = os.path.join(config.DATA_ROOT, split, f"{peptide_id}-traj-arrays.npz")
        pdb = os.path.join(config.DATA_ROOT, split, f"{peptide_id}-traj-state0.pdb")
        if os.path.exists(npz) and os.path.exists(pdb):
            return npz, pdb, split
    raise FileNotFoundError(f"No raw trajectory found for peptide '{peptide_id}'.")


def build_state(peptide_id, device="cpu"):
    """Assemble the static (per-peptide) half of the model input.

    Returns a dict with atom_type, z, edge_index, bond_flag, batch, num_graphs,
    plus bonds/eq_len for the guard-rail and the raw file paths.
    """
    npz_path, pdb_path, split = find_raw_files(peptide_id)
    positions, velocities, vel_source = load_positions_velocities(npz_path)

    from scripts.processing import parse_elements_from_pdb
    elements = parse_elements_from_pdb(pdb_path)
    atom_type = torch.tensor(
        [config.ELEMENTS.index(e) if e in config.ELEMENTS else 0 for e in elements],
        dtype=torch.long, device=device,
    )

    n = positions.shape[1]
    pos0_centred = positions[0] - positions[0].mean(0)
    bonds_np, eq_len_np = covalent_bonds(pos0_centred)

    idx = torch.arange(n, device=device)
    src = idx.repeat_interleave(n)
    dst = idx.repeat(n)
    keep = src != dst
    edge_index = torch.stack([src[keep], dst[keep]])

    bonds = torch.from_numpy(bonds_np).to(device)
    key = edge_index[0] * n + edge_index[1]
    bond_flag = torch.isin(key, bonds[0] * n + bonds[1]).float()

    embeddings = torch.load(config.EMBEDDING_CACHE, map_location="cpu")
    if peptide_id not in embeddings:
        raise KeyError(
            f"No cached Stage-1 embedding for '{peptide_id}'. Add it to the transition "
            f"cache (prepare_data.py) and re-run adapter.py."
        )

    return {
        "atom_type": atom_type,
        "z": embeddings[peptide_id].float().unsqueeze(0).to(device),
        "edge_index": edge_index,
        "bond_flag": bond_flag,
        "batch": torch.zeros(n, dtype=torch.long, device=device),
        "num_graphs": 1,
        "bonds": bonds_np,
        "eq_len": eq_len_np,
        "positions": positions,
        "velocities": velocities,
        "vel_source": vel_source,
        "pdb_path": pdb_path,
        "split": split,
        "n_atoms": n,
    }


# ──────────────────────────────────────────────────────────────
# Guard-rail
# ──────────────────────────────────────────────────────────────

def unique_bonds(bonds, eq_len):
    """Collapse the bidirectional edge list to one entry per physical bond."""
    src, dst = bonds
    keep = src < dst
    return np.stack([src[keep], dst[keep]]), eq_len[keep]


def project_bonds(pos, bonds, eq_len, iters=None, tol=None, relax=1.0):
    """Pull over/under-stretched covalent bonds back toward equilibrium.

    A Jacobi-style constraint relaxation. Two details make it stable, and
    getting either wrong makes the rollout diverge within ~20 frames:

      * `bonds` is stored bidirectionally (i->j and j->i), so it is collapsed to
        one entry per physical bond first — otherwise every bond is corrected
        twice per sweep.
      * A carbon has up to 4 bonds, all pulling on it in the same sweep. Summing
        those corrections overshoots and the error oscillates outward. Dividing
        each atom's accumulated correction by its bond count turns the sum into
        an average, which contracts instead.

    Bonds within `tol` are left alone, so a well-behaved trajectory passes
    through untouched.
    """
    iters = config.BOND_FIX_ITERS if iters is None else iters
    tol = config.BOND_TOL_NM if tol is None else tol
    if iters <= 0:
        return pos

    ub, ueq = unique_bonds(bonds, eq_len)
    src = torch.as_tensor(ub[0], dtype=torch.long, device=pos.device)
    dst = torch.as_tensor(ub[1], dtype=torch.long, device=pos.device)
    eq = torch.as_tensor(ueq, dtype=pos.dtype, device=pos.device)

    # Bond count per atom, used to average rather than sum the corrections.
    degree = torch.zeros(pos.shape[0], dtype=pos.dtype, device=pos.device)
    degree.index_add_(0, src, torch.ones_like(eq))
    degree.index_add_(0, dst, torch.ones_like(eq))
    degree = degree.clamp(min=1.0).unsqueeze(-1)

    for _ in range(iters):
        d = pos[src] - pos[dst]
        length = d.norm(dim=-1)
        err = length - eq
        active = (err.abs() > tol).to(pos.dtype)
        if not active.any():
            break
        direction = d / length.clamp(min=1e-8).unsqueeze(-1)
        corr = 0.5 * (err * active).unsqueeze(-1) * direction     # [B, 3]

        delta = torch.zeros_like(pos)
        delta.index_add_(0, src, -corr)
        delta.index_add_(0, dst, corr)
        pos = pos + relax * delta / degree

    return pos


# ──────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def autoregressive_rollout(model, state, scales, n_frames=None, start_frame=0,
                           flow_steps=None, temperature=None, bond_fix=None,
                           seed=None, device="cpu"):
    """Generate an (n_frames, N, 3) trajectory starting from a real MD frame.

    Returns:
        traj [n_frames, N, 3] positions (nm, body-fixed frame, centred)
        vels [n_frames, N, 3] velocities (nm/ps)
    """
    n_frames = n_frames or config.ROLLOUT_FRAMES
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    pos0 = state["positions"][start_frame]
    vel0 = strip_com_velocity(state["velocities"][start_frame])
    pos = torch.tensor(pos0 - pos0.mean(0), dtype=torch.float32, device=device)
    vel = torch.tensor(vel0, dtype=torch.float32, device=device)

    traj = torch.empty((n_frames, state["n_atoms"], 3), dtype=torch.float32)
    vels = torch.empty_like(traj)
    traj[0], vels[0] = pos.cpu(), vel.cpu()

    step_state = {k: state[k] for k in
                  ("atom_type", "z", "edge_index", "bond_flag", "batch", "num_graphs")}

    for t in range(1, n_frames):
        step_state["pos_t"] = pos
        step_state["vel_t"] = vel

        dpos, vel_next = flow.sample_transition(
            model, step_state, scales, steps=flow_steps,
            temperature=temperature, generator=generator,
        )

        pos = pos + dpos
        pos = project_bonds(pos, state["bonds"], state["eq_len"], iters=bond_fix)
        pos = pos - pos.mean(0)                    # stay in the centred frame

        # With the velocity channel switched off the model emits no velocity, so
        # the state carried forward is position only. The recorded velocity is
        # then just the finite difference, kept for the output file's sake.
        if vel_next is None:
            vel = (pos - traj[t - 1].to(device)) / config.FRAME_LAG_PS
        else:
            vel = vel_next - vel_next.mean(0)      # COM drift stays projected out

        # A diverged rollout poisons every downstream metric with NaN, and the
        # metrics alone do not say where it happened. Stop at the first bad
        # frame and report it instead.
        if not torch.isfinite(pos).all():
            raise RuntimeError(
                f"rollout diverged at frame {t}/{n_frames} (non-finite positions). "
                "The model is likely undertrained, or --temperature is too high."
            )

        traj[t], vels[t] = pos.cpu(), vel.cpu()

    return traj.numpy(), vels.numpy()


# ──────────────────────────────────────────────────────────────
# PDB output
# ──────────────────────────────────────────────────────────────

def write_multimodel_pdb(traj_nm, template_pdb, out_path):
    """Write a trajectory as a multi-MODEL PDB, reusing the topology template.

    Coordinates are converted nm -> Angstrom (PDB convention).
    """
    with open(template_pdb) as f:
        atom_lines = [l for l in f if l.startswith(("ATOM", "HETATM"))]

    if len(atom_lines) != traj_nm.shape[1]:
        raise ValueError(f"template has {len(atom_lines)} atoms, "
                         f"trajectory has {traj_nm.shape[1]}")

    with open(out_path, "w") as out:
        for m, frame in enumerate(traj_nm, start=1):
            out.write(f"MODEL     {m:>4}\n")
            for line, (x, y, z) in zip(atom_lines, frame * 10.0):
                out.write(f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:].rstrip()}\n")
            out.write("ENDMDL\n")
        out.write("END\n")


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Roll out a Stage-3 trajectory.")
    ap.add_argument("--peptide", required=True, help="e.g. AC")
    ap.add_argument("--frames", type=int, default=config.ROLLOUT_FRAMES)
    ap.add_argument("--samples", type=int, default=1, help="independent trajectories")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--flow-steps", type=int, default=config.FLOW_STEPS)
    ap.add_argument("--temperature", type=float, default=config.FLOW_TEMP)
    ap.add_argument("--bond-fix", type=int, default=config.BOND_FIX_ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pdb", action="store_true", help="also write a multi-MODEL PDB")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    config.ensure_dirs()

    model, ckpt = load_propagator(device=device)
    scales = flow.load_scales(device)
    state = build_state(args.peptide, device)

    out_dir = os.path.join(config.RESULTS_DIR, args.peptide)
    os.makedirs(out_dir, exist_ok=True)

    print(f"peptide {args.peptide}  ({state['n_atoms']} atoms, split={state['split']}, "
          f"velocities from {state['vel_source']})")
    print(f"checkpoint epoch {ckpt['epoch']}  val {ckpt['val_loss']:.5f}")
    print(f"rolling out {args.samples} x {args.frames} frames "
          f"({args.frames * config.FRAME_LAG_PS / 1000:.2f} ns of simulated time)\n")

    for s in range(args.samples):
        traj, vels = autoregressive_rollout(
            model, state, scales, n_frames=args.frames, start_frame=args.start_frame,
            flow_steps=args.flow_steps, temperature=args.temperature,
            bond_fix=args.bond_fix, seed=args.seed + s, device=device,
        )
        suffix = "" if args.samples == 1 else f"_s{s}"
        np.savez_compressed(
            os.path.join(out_dir, f"{args.peptide}_rollout{suffix}.npz"),
            positions=traj, velocities=vels,
        )
        from temporal_dynamics.velocity_utils import bond_length_rmsd
        rmsd = bond_length_rmsd(traj, state["bonds"], state["eq_len"])
        print(f"  sample {s}: bond-length RMSD {rmsd * 10:.4f} A  -> "
              f"{args.peptide}_rollout{suffix}.npz")

        if args.pdb:
            pdb_out = os.path.join(out_dir, f"{args.peptide}_rollout{suffix}.pdb")
            write_multimodel_pdb(traj, state["pdb_path"], pdb_out)
            print(f"             {os.path.basename(pdb_out)}")

    print(f"\nwritten to {out_dir}")


if __name__ == "__main__":
    main()
