"""
check_velocities.py — STEP 0. Audit the dataset before doing anything else.

Answers three questions and prints them as a table:

  1. Does every .npz actually ship a `velocities` array, or do some need the
     finite-difference fallback in velocity_utils?
  2. What is the saved-frame lag, and is the velocity usable as an integrator
     over that lag? (Correlation between v(t) and the frame-to-frame
     displacement — if this is ~0, `x + v*dt` is meaningless.)
  3. How much of the frame-to-frame motion is global tumbling vs. internal
     motion? (Displacement with the centroid removed vs. after Kabsch.)

Run:
    python -m temporal_dynamics.check_velocities
    python -m temporal_dynamics.check_velocities --split train --max 40
"""

import argparse
import glob
import os

import numpy as np

from temporal_dynamics import config
from temporal_dynamics.velocity_utils import (
    kabsch_rotation,
    load_positions_velocities,
    strip_com_velocity,
)


def audit_split(split: str, max_files: int, probe_frames: int = 200):
    pattern = os.path.join(config.DATA_ROOT, split, "*-traj-arrays.npz")
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        print(f"  no .npz files under {os.path.join(config.DATA_ROOT, split)}")
        return None

    total = len(npz_files)
    npz_files = npz_files[:max_files]

    from_file, from_fd, missing_pdb = 0, 0, 0
    lags, corrs, raw_disp, ali_disp, speeds = [], [], [], [], []

    for path in npz_files:
        pdb = path.replace("-traj-arrays.npz", "-traj-state0.pdb")
        if not os.path.exists(pdb):
            missing_pdb += 1

        pos, vel, source = load_positions_velocities(path)
        from_file += source == "file"
        from_fd   += source == "finite-diff"

        with np.load(path) as a:
            if "time" in a.files and len(a["time"]) > 1:
                lags.append(float(a["time"][1] - a["time"][0]))

        n = min(probe_frames, pos.shape[0] - 1)
        p = pos[: n + 1].astype(np.float64)
        v = strip_com_velocity(vel[: n + 1].astype(np.float64))

        # Is velocity an integrator over this lag?
        fd = (p[1:] - p[:-1]) / config.FRAME_LAG_PS
        corrs.append(np.corrcoef(v[:-1].ravel(), fd.ravel())[0, 1])
        speeds.append(np.linalg.norm(v, axis=-1).mean())

        # Tumbling vs. internal motion.
        for t in range(n):
            a_ref = p[t] - p[t].mean(0)
            b_raw = p[t + 1] - p[t + 1].mean(0)
            R = kabsch_rotation(p[t + 1], p[t])
            b_ali = (R @ b_raw.T).T
            raw_disp.append(np.linalg.norm(b_raw - a_ref, axis=-1).mean())
            ali_disp.append(np.linalg.norm(b_ali - a_ref, axis=-1).mean())

    return {
        "split": split,
        "files_total": total,
        "files_probed": len(npz_files),
        "from_file": from_file,
        "from_fd": from_fd,
        "missing_pdb": missing_pdb,
        "lag_ps": float(np.mean(lags)) if lags else float("nan"),
        "corr": float(np.mean(corrs)),
        "speed": float(np.mean(speeds)),
        "raw_disp": float(np.mean(raw_disp)),
        "ali_disp": float(np.mean(ali_disp)),
    }


def main():
    ap = argparse.ArgumentParser(description="Audit velocities in the Timewarp dataset.")
    ap.add_argument("--split", default=None, help="train/val/test (default: all three)")
    ap.add_argument("--max", type=int, default=20, help="files to probe per split")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["train", "val", "test"]

    print("=" * 74)
    print("STEP 0 - VELOCITY AUDIT")
    print("=" * 74)
    print(f"dataset: {config.DATA_ROOT}\n")

    reports = []
    for split in splits:
        print(f"[{split}] probing up to {args.max} files ...")
        r = audit_split(split, args.max)
        if r:
            reports.append(r)

    if not reports:
        return

    print()
    print(f"{'split':<7}{'files':>7}{'probed':>8}{'in-file':>9}{'fin-diff':>10}{'lag/ps':>9}")
    print("-" * 74)
    for r in reports:
        print(f"{r['split']:<7}{r['files_total']:>7}{r['files_probed']:>8}"
              f"{r['from_file']:>9}{r['from_fd']:>10}{r['lag_ps']:>9.2f}")

    print()
    print(f"{'split':<7}{'corr(v, dx/dt)':>16}{'speed nm/ps':>14}"
          f"{'disp centred':>15}{'disp Kabsch':>14}")
    print("-" * 74)
    for r in reports:
        print(f"{r['split']:<7}{r['corr']:>16.4f}{r['speed']:>14.3f}"
              f"{r['raw_disp']:>15.4f}{r['ali_disp']:>14.4f}")

    total_fd = sum(r["from_fd"] for r in reports)
    corr = np.mean([r["corr"] for r in reports])
    ratio = np.mean([r["raw_disp"] / max(r["ali_disp"], 1e-9) for r in reports])

    print()
    print("VERDICT")
    print("-" * 74)
    if total_fd == 0:
        print("  [ok]   Every probed file ships real velocities - no synthesis needed.")
    else:
        print(f"  [warn] {total_fd} file(s) had no `velocities` key; the finite-difference")
        print("         fallback in velocity_utils.load_positions_velocities() was used.")

    if abs(corr) < 0.1:
        print(f"  [note] corr(v, dx/dt) = {corr:+.4f} ~ 0 over a {reports[0]['lag_ps']:.1f} ps lag.")
        print("         Velocity is NOT an integrator here. Stage 3 treats it as a state")
        print("         channel: read as input, co-predicted as output.")
    else:
        print(f"  [note] corr(v, dx/dt) = {corr:+.4f} - velocity carries integrable signal.")

    print(f"  [note] Global tumbling is {ratio:.1f}x the internal motion, so every frame is")
    print("         Kabsch-aligned onto its predecessor before training.")
    print()
    print("Next:  python -m temporal_dynamics.prepare_data")


if __name__ == "__main__":
    main()
