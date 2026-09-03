"""
prepare_data.py — STEP 1. Build the aligned transition cache.

Reads the raw Timewarp trajectories and writes one .npz per peptide into
`temporal_dynamics/cache/transitions/`, containing everything the model needs:

    pos_t     [T, N, 3]  centred positions at t          (input)
    vel_t     [T, N, 3]  velocities at t, COM removed    (input)
    dpos      [T, N, 3]  pos_{t+1} (Kabsch-aligned) - pos_t   (target)
    vel_tp1   [T, N, 3]  velocities at t+1, same rotation     (target)
    atom_type [N]        element index into config.ELEMENTS
    bonds     [2, E]     covalent edges (both directions)
    eq_len    [E]        equilibrium bond lengths

Doing the Kabsch alignment once here rather than in the DataLoader matters:
it is an SVD per transition, which would otherwise dominate training time.

A global `stats.npz` is also written, holding the standard deviations used to
unit-normalise the two target channels before flow matching.

Run:
    python -m temporal_dynamics.prepare_data
    python -m temporal_dynamics.prepare_data --max-train 100 --max-frames 3000
"""

import argparse
import glob
import os

import numpy as np

from scripts.processing import parse_elements_from_pdb
from temporal_dynamics import config
from temporal_dynamics.velocity_utils import (
    build_transitions,
    covalent_bonds,
    load_positions_velocities,
)


def peptide_files(split: str):
    """(peptide_id, npz_path, pdb_path) for every peptide with BOTH files present."""
    pattern = os.path.join(config.DATA_ROOT, split, "*-traj-arrays.npz")
    out = []
    for npz in sorted(glob.glob(pattern)):
        pdb = npz.replace("-traj-arrays.npz", "-traj-state0.pdb")
        if os.path.exists(pdb):
            out.append((os.path.basename(npz).split("-")[0], npz, pdb))
    return out


def prepare_peptide(peptide_id, npz_path, pdb_path, split, max_frames, stride):
    """Convert one raw trajectory into a cached transition file. Returns its stats."""
    positions, velocities, source = load_positions_velocities(npz_path)
    positions  = positions[: max_frames * stride]
    velocities = velocities[: max_frames * stride]

    elements = parse_elements_from_pdb(pdb_path)
    n_atoms = positions.shape[1]
    if len(elements) != n_atoms:
        return None, f"element/atom count mismatch ({len(elements)} vs {n_atoms})"

    atom_type = np.array(
        [config.ELEMENTS.index(e) if e in config.ELEMENTS else 0 for e in elements],
        dtype=np.int64,
    )

    tr = build_transitions(positions, velocities, stride=stride)
    bonds, eq_len = covalent_bonds(tr["pos_t"][0])

    # The velocity arrays are exactly half the cache and, at a 5 ps lag, carry no
    # usable signal (see config.PREDICT_VELOCITY). Drop them unless asked for.
    keep_vel = config.PREDICT_VELOCITY or config.USE_VELOCITY_INPUT
    arrays = dict(tr) if keep_vel else {"pos_t": tr["pos_t"], "dpos": tr["dpos"]}

    out_path = os.path.join(config.TRANSITION_DIR, f"{peptide_id}.npz")
    np.savez_compressed(
        out_path,
        split=split,
        peptide_id=peptide_id,
        vel_source=source,
        atom_type=atom_type,
        bonds=bonds,
        eq_len=eq_len,
        **arrays,
    )

    # Per-element sums of squares, so the global scales can be pooled exactly
    # across peptides rather than averaged over per-peptide standard deviations.
    # Both channels get this: displacement amplitude is element-dependent, and
    # velocity even more so — thermal speeds go as 1/sqrt(mass), so hydrogen is
    # several times faster than carbon and would otherwise dominate the channel.
    sq = np.zeros(config.NUM_ELEMENTS)
    cnt = np.zeros(config.NUM_ELEMENTS)
    vsq = np.zeros(config.NUM_ELEMENTS)
    for i in range(config.NUM_ELEMENTS):
        m = atom_type == i
        if m.any():
            d = tr["dpos"][:, m]
            sq[i] = float((d.astype(np.float64) ** 2).sum())
            cnt[i] = d.size
            v = tr["vel_tp1"][:, m]
            vsq[i] = float((v.astype(np.float64) ** 2).sum())

    return {
        "peptide_id": peptide_id,
        "n_atoms": n_atoms,
        "n_trans": tr["pos_t"].shape[0],
        "n_bonds": bonds.shape[1] // 2,
        "dpos_std": float(tr["dpos"].std()),
        "vel_std": float(tr["vel_tp1"].std()),
        "elem_sq": sq,
        "elem_cnt": cnt,
        "elem_vsq": vsq,
        "source": source,
    }, None


def main():
    ap = argparse.ArgumentParser(description="Build the Stage-3 transition cache.")
    ap.add_argument("--max-train", type=int, default=config.MAX_TRAIN_PEPTIDES)
    ap.add_argument("--max-val", type=int, default=config.MAX_VAL_PEPTIDES)
    ap.add_argument("--max-test", type=int, default=8,
                    help="test peptides to cache (evaluation pulls raw data itself, "
                         "so this is only for reporting)")
    ap.add_argument("--max-frames", type=int, default=config.MAX_FRAMES)
    ap.add_argument("--stride", type=int, default=config.FRAME_STRIDE)
    args = ap.parse_args()

    config.ensure_dirs()

    print("=" * 74)
    print("STEP 1 - BUILD TRANSITION CACHE")
    print("=" * 74)
    print(f"lag       : {config.FRAME_LAG_PS * args.stride:.1f} ps "
          f"({args.stride} saved frame(s))")
    print(f"max frames: {args.max_frames} per peptide")
    print(f"out dir   : {config.TRANSITION_DIR}\n")

    caps = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    all_stats, manifest = [], {"train": [], "val": [], "test": []}

    for split, cap in caps.items():
        files = peptide_files(split)[:cap]
        print(f"[{split}] {len(files)} peptide(s)")
        for pid, npz, pdb in files:
            stats, err = prepare_peptide(pid, npz, pdb, split, args.max_frames, args.stride)
            if err:
                print(f"   skip {pid}: {err}")
                continue
            manifest[split].append(pid)
            all_stats.append(stats)
            print(f"   {pid:<5} atoms={stats['n_atoms']:>3}  "
                  f"transitions={stats['n_trans']:>5}  bonds={stats['n_bonds']:>3}  "
                  f"vel={stats['source']}")

    if not all_stats:
        print("\nNo peptides cached. Is timewarp_data/2AA-complete/ populated?")
        return

    # Normalisation scales. Flow matching interpolates between N(0, I) and the
    # target, so the target must live on a unit scale. Displacement amplitude is
    # strongly element-dependent, so we pool a separate std per element rather
    # than letting hydrogen's larger motion dominate every other atom.
    dpos_std = float(np.mean([s["dpos_std"] for s in all_stats]))
    vel_std  = float(np.mean([s["vel_std"] for s in all_stats]))

    sq = np.sum([s["elem_sq"] for s in all_stats], axis=0)
    cnt = np.sum([s["elem_cnt"] for s in all_stats], axis=0)
    vsq = np.sum([s["elem_vsq"] for s in all_stats], axis=0)
    # Elements absent from the dataset fall back to the global std.
    dpos_std_elem = np.where(cnt > 0, np.sqrt(sq / np.maximum(cnt, 1)), dpos_std)
    vel_std_elem = np.where(cnt > 0, np.sqrt(vsq / np.maximum(cnt, 1)), vel_std)

    np.savez(config.STATS_CACHE, dpos_std=dpos_std, vel_std=vel_std,
             dpos_std_elem=dpos_std_elem.astype(np.float32),
             vel_std_elem=vel_std_elem.astype(np.float32),
             frame_lag_ps=config.FRAME_LAG_PS * args.stride, stride=args.stride,
             train=np.array(manifest["train"]), val=np.array(manifest["val"]),
             test=np.array(manifest["test"]))

    n_fd = sum(s["source"] == "finite-diff" for s in all_stats)
    print()
    print("-" * 74)
    print(f"cached      : {len(all_stats)} peptides, "
          f"{sum(s['n_trans'] for s in all_stats):,} transitions")
    print(f"velocities  : {len(all_stats) - n_fd} from file, {n_fd} finite-difference"
          + ("" if (config.PREDICT_VELOCITY or config.USE_VELOCITY_INPUT)
             else "  (NOT stored — see config.PREDICT_VELOCITY)"))
    print(f"dpos std    : {dpos_std:.5f} nm      (pooled over all atoms)")
    for i, e in enumerate(config.ELEMENTS):
        if cnt[i] > 0:
            print(f"   {e:<2}       : {dpos_std_elem[i]:.5f} nm      "
                  f"({int(cnt[i] // 3):,} atom-frames)")
    print(f"stats       : {config.STATS_CACHE}")
    print()
    print("Next:  python -m temporal_dynamics.adapter")


if __name__ == "__main__":
    main()
