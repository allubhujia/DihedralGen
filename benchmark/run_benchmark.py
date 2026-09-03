"""
run_benchmark.py — Stage 2 vs Stage 3, head to head, on one peptide.

Asks the terminal for three things (as flags, or interactively if omitted):

    peptide      which dipeptide, e.g. AC
    frames       how many frames to generate, e.g. 100
    start-frame  which real MD frame Stage 3 starts from, e.g. 50

and produces two figures plus a metrics report.

THE ONE ASYMMETRY, STATED UP FRONT
----------------------------------
Stage 3 is a propagator: it is seeded with the real MD frame at `start_frame`
and walks forward from there, so it has a time axis.

Stage 2 is a distribution model: it maps the molecule's identity straight to a
set of backbone angles. It cannot be seeded with a starting structure at all,
and its checkpoint is locked to exactly 300 frames. `start_frame` therefore does
not apply to it, and its frame count is reshaped from 300 (see stage2.py).

Because of that, the headline metrics score BOTH stages against the same
reference: the equilibrium density of the full MD trajectory. That is the only
target Stage 2 can be fairly measured against. Stage 3 additionally gets a
window score against the specific frames it was asked to reproduce, reported
separately and never mixed into the comparison.

Run:
    python -m benchmark.run_benchmark --peptide AC --frames 100 --start-frame 50
    python -m benchmark.run_benchmark            # prompts for all three
"""

import argparse
import os
import sys

import numpy as np
import torch

from benchmark import bench_config as bcfg
from benchmark import plots, stage2, stage3
from temporal_dynamics import config as s3cfg
from temporal_dynamics.evaluate import js_divergence_2d, kl_divergence_1d


# ──────────────────────────────────────────────────────────────
# Terminal input
# ──────────────────────────────────────────────────────────────

def ask(prompt, default, cast=str):
    """Prompt with a default; fall back to the default if stdin is not a TTY."""
    if not sys.stdin.isatty():
        return default
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  could not read '{raw}' — using {default}")
        return default


def collect_inputs(args):
    peptide = args.peptide or ask("Peptide", bcfg.DEFAULT_PEPTIDE)
    frames = args.frames or ask("Frames to generate", bcfg.DEFAULT_FRAMES, int)
    start = args.start_frame
    if start is None:
        start = ask("Stage-3 start frame (real MD frame to seed from)",
                    bcfg.DEFAULT_START_FRAME, int)
    return peptide.upper(), int(frames), int(start)


# ──────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────

def score(pred_phi, pred_psi, ref_phi, ref_psi):
    """KL per angle plus the 2-D Jensen-Shannon, against one reference set."""
    return {
        "kl_phi": kl_divergence_1d(ref_phi, pred_phi),
        "kl_psi": kl_divergence_1d(ref_psi, pred_psi),
        "js2d": js_divergence_2d(ref_phi, ref_psi, pred_phi, pred_psi),
    }


def write_report(path, pid, frames, start, gt, s2, s3_, metrics, s2_info, s3_info):
    with open(path, "w") as f:
        w = f.write
        w(f"Stage 2 vs Stage 3 benchmark - {pid}\n")
        w("=" * 68 + "\n\n")
        w(f"peptide            : {pid}  ({s3_info['n_atoms']} atoms, "
          f"split={s3_info['split']})\n")
        w(f"frames generated   : {frames}\n")
        w(f"Stage-3 start frame: {start}   "
          f"(of {gt['n_total_frames']} real MD frames)\n")
        w(f"simulated time     : {s3_info['sim_time_ns']:.2f} ns "
          f"at {s3cfg.FRAME_LAG_PS} ps/frame\n\n")

        w("REFERENCE\n")
        w("-" * 68 + "\n")
        w(f"  Both stages scored against the full MD equilibrium density\n")
        w(f"  ({len(gt['phi_full'])} frames, stride {bcfg.GT_STRIDE}).\n\n")

        w("RESULTS  (lower is better)\n")
        w("-" * 68 + "\n")
        w(f"  {'metric':<22}{'Stage 2':>12}{'Stage 3':>12}{'winner':>12}\n")
        for key, label in (("kl_phi", "KL phi"), ("kl_psi", "KL psi"),
                           ("js2d", "JS 2-D (bits)")):
            a, b = metrics["s2"][key], metrics["s3"][key]
            win = "Stage 2" if a < b else ("Stage 3" if b < a else "tie")
            w(f"  {label:<22}{a:>12.4f}{b:>12.4f}{win:>12}\n")

        w("\nSTAGE-3 ONLY\n")
        w("-" * 68 + "\n")
        w("  Scores against the specific MD window it was seeded to reproduce.\n")
        w("  Stage 2 has no equivalent number - it cannot be seeded.\n")
        for key, label in (("kl_phi", "KL phi vs window"),
                           ("kl_psi", "KL psi vs window"),
                           ("js2d", "JS 2-D vs window")):
            w(f"  {label:<22}{metrics['s3_window'][key]:>12.4f}\n")
        w(f"  {'bond-length RMSD':<22}{s3_info['bond_rmsd_A']:>12.4f}  A\n")
        w(f"  checkpoint             epoch {s3_info['epoch']}, "
          f"val {s3_info['val_loss']:.5f}\n")

        w("\nHOW EACH STAGE PRODUCED ITS FRAMES\n")
        w("-" * 68 + "\n")
        w(f"  Stage 2: {s2_info['mode']} from {s2_info['draws']} draw(s) of its\n")
        w(f"           native {s2_info['native_frames']} frames; NOT seeded.\n")
        w(f"           encoder = {s2_info['encoder']}\n")
        w(f"  Stage 3: autoregressive rollout seeded at MD frame {start}.\n")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Benchmark Stage 2 against Stage 3 on the same peptide.")
    ap.add_argument("--peptide", default=None, help="e.g. AC")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames to generate, e.g. 100")
    ap.add_argument("--start-frame", type=int, default=None,
                    help="real MD frame Stage 3 is seeded from, e.g. 50")
    ap.add_argument("--seed", type=int, default=bcfg.SEED)
    ap.add_argument("--gt-stride", type=int, default=bcfg.GT_STRIDE,
                    help="subsample stride for the full-MD reference density")
    ap.add_argument("--flow-steps", type=int, default=None,
                    help="Stage-3 Euler steps per transition (default from config)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Stage-3 sampling temperature (<1 sharpens)")
    ap.add_argument("--bond-fix", type=int, default=None,
                    help="Stage-3 bond-constraint sweeps per step (0 disables)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    peptide, frames, start = collect_inputs(args)
    bcfg.ensure_dirs()
    out_dir = bcfg.run_dir(peptide, frames, start)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 68)
    print("STAGE 2 vs STAGE 3 BENCHMARK")
    print("=" * 68)
    print(f"peptide     : {peptide}")
    print(f"frames      : {frames}")
    print(f"start frame : {start}   (Stage 3 only - Stage 2 cannot be seeded)")
    print(f"device      : {args.device}\n")

    # --- Stage 3 (also gives us the peptide state + ground truth) ----------
    print("[Stage 3] rolling out ...")
    s3_phi, s3_psi, s3_info, traj, state = stage3.generate(
        peptide, frames, start_frame=start, seed=args.seed,
        flow_steps=args.flow_steps, temperature=args.temperature,
        bond_fix=args.bond_fix, device=args.device,
    )
    print(f"          done - bond RMSD {s3_info['bond_rmsd_A']:.3f} A, "
          f"{s3_info['sim_time_ns']:.2f} ns simulated")

    gt = stage3.ground_truth(state, frames, start_frame=start,
                             gt_stride=args.gt_stride)

    # --- Stage 2 -----------------------------------------------------------
    print("[Stage 2] generating ...")
    npz_path = os.path.join(s3cfg.DATA_ROOT, state["split"],
                            f"{peptide}-traj-arrays.npz")
    s2_phi, s2_psi, s2_info = stage2.generate(
        npz_path, state["pdb_path"], frames, seed=args.seed, device=args.device,
    )
    print(f"          done - {s2_info['mode']} from {s2_info['draws']} draw(s) "
          f"of {s2_info['native_frames']} frames")

    # --- Score -------------------------------------------------------------
    metrics = {
        "s2": score(s2_phi, s2_psi, gt["phi_full"], gt["psi_full"]),
        "s3": score(s3_phi, s3_psi, gt["phi_full"], gt["psi_full"]),
        "s3_window": score(s3_phi, s3_psi, gt["phi_window"], gt["psi_window"]),
    }

    s2_d = {"phi": s2_phi, "psi": s2_psi}
    s3_d = {"phi": s3_phi, "psi": s3_psi}

    # --- Figures -----------------------------------------------------------
    rama_path = plots.ramachandran_figure(peptide, gt, s2_d, s3_d, metrics, out_dir)
    kl_path = plots.kl_figure(peptide, gt, s2_d, s3_d, metrics, out_dir)

    # --- Data + report -----------------------------------------------------
    np.savez_compressed(
        os.path.join(out_dir, f"{peptide}_benchmark_dihedrals.npz"),
        stage2_phi=s2_phi, stage2_psi=s2_psi,
        stage3_phi=s3_phi, stage3_psi=s3_psi,
        gt_phi_window=gt["phi_window"], gt_psi_window=gt["psi_window"],
        gt_phi_full=gt["phi_full"], gt_psi_full=gt["psi_full"],
        stage3_positions=traj,
    )

    report_path = os.path.join(out_dir, "benchmark_metrics.txt")
    write_report(report_path, peptide, frames, start, gt, s2_d, s3_d,
                 metrics, s2_info, s3_info)

    with open(os.path.join(out_dir, "benchmark_summary.csv"), "w") as f:
        f.write("peptide,frames,start_frame,stage,kl_phi,kl_psi,js2d\n")
        for tag, key in (("stage2", "s2"), ("stage3", "s3")):
            m = metrics[key]
            f.write(f"{peptide},{frames},{start},{tag},"
                    f"{m['kl_phi']:.6f},{m['kl_psi']:.6f},{m['js2d']:.6f}\n")

    # --- Console summary ---------------------------------------------------
    print()
    print(f"{'metric':<16}{'Stage 2':>12}{'Stage 3':>12}{'winner':>12}")
    print("-" * 52)
    for key, label in (("kl_phi", "KL phi"), ("kl_psi", "KL psi"),
                       ("js2d", "JS 2-D")):
        a, b = metrics["s2"][key], metrics["s3"][key]
        win = "Stage 2" if a < b else ("Stage 3" if b < a else "tie")
        print(f"{label:<16}{a:>12.4f}{b:>12.4f}{win:>12}")
    print("-" * 52)
    print("(both scored against the same full-MD equilibrium density)")

    print(f"\nwritten to {out_dir}")
    for p in (rama_path, kl_path, report_path):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
