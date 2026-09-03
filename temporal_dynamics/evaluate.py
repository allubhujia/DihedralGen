"""
evaluate.py — STEP 4. Score generated trajectories against ground-truth MD.

For each test peptide: roll out 300 frames, extract backbone (phi, psi), and
compare against the real MD trajectory on four axes.

  1. DISTRIBUTION - does it visit the right conformational basins?
       Ramachandran overlay + per-angle KL divergence + 2-D Jensen-Shannon.
       This is the same criterion Stage 2 was judged on, so the numbers are
       directly comparable.

  2. DYNAMICS - is it actually simulating, or just sampling?
       Dihedral autocorrelation. A model that draws i.i.d. samples from the
       equilibrium distribution scores perfectly on (1) but has an ACF that
       drops to zero in one frame. Matching MD's decorrelation profile is what
       separates Stage 3 from Stage 2, and no distributional metric can see it.

  3. SHORT-HORIZON TRACKING - MAD on the first few frames.
       MD is chaotic, so a sample path is expected to diverge from any specific
       GT path. But over the first handful of frames a correct propagator should
       still track it. Reported over a short window only; large MAD at long
       horizon is expected and is NOT a failure.

  4. PHYSICALITY - bond-length RMSD over the rollout.
       Catches a molecule that slowly inflates or collapses, which a
       Ramachandran plot will happily hide.

Run:
    python -m temporal_dynamics.evaluate
    python -m temporal_dynamics.evaluate --peptide AC
    python -m temporal_dynamics.evaluate --peptides AC AD AH --frames 300
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import entropy

from scripts.backbone_utils import compute_phi_psi_from_positions, parse_backbone_from_pdb
from temporal_dynamics import config, flow
from temporal_dynamics.rollout import autoregressive_rollout, build_state, load_propagator
from temporal_dynamics.velocity_utils import bond_length_rmsd


# ──────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────

def kl_divergence_1d(gt_deg, pred_deg, bins=config.KL_BINS, alpha=1.0):
    """KL(P_gt || P_pred) over a fixed [-180, 180] binning, Laplace-smoothed.

    WHY LAPLACE AND NOT A TINY EPSILON
    ----------------------------------
    KL is infinite wherever the prediction puts zero mass on a bin the truth
    occupies. The obvious patch is to add a tiny epsilon, but that is far worse
    than it looks: with eps = 1e-8 an empty bin contributes p*log(p/1e-8), which
    is enormous. At 100 samples over 36 bins, ~6 bins come up empty even when
    the "prediction" is genuine MD data, so the metric was reporting a floor of
    KL ~ 0.62 for a perfect model and burying the real signal underneath it.

    Add-alpha (Laplace) smoothing instead spreads one pseudo-count over every
    bin, which is the posterior mean under a uniform Dirichlet prior. That keeps
    empty bins finite and proportionate: the measured floor drops to ~0.17 with
    a tenth of the variance, so differences between models actually show up.

    Always read this number against `null_baseline_kl` for the same sample
    count - an absolute KL is meaningless without knowing its floor.
    """
    edges = np.linspace(-180, 180, bins + 1)
    p = np.histogram(gt_deg, bins=edges)[0].astype(float) + alpha
    q = np.histogram(pred_deg, bins=edges)[0].astype(float) + alpha
    return float(entropy(p / p.sum(), q / q.sum()))


def null_baseline_kl(gt_deg, n_samples, bins=config.KL_BINS, trials=30, seed=0):
    """The KL a PERFECT model would score, given only `n_samples` draws.

    Draws `n_samples` real frames from the ground truth and scores them against
    the full ground truth, `trials` times. Anything at or below mean + std here
    is indistinguishable from perfect at this sample count.

    Returns (mean, std).
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(gt_deg))
    vals = [kl_divergence_1d(gt_deg, rng.choice(gt_deg, n, replace=False), bins=bins)
            for _ in range(trials)]
    return float(np.mean(vals)), float(np.std(vals))


def js_divergence_2d(gt_phi, gt_psi, pr_phi, pr_psi, bins=config.KDE_GRID, sigma=1.5):
    """Jensen-Shannon divergence between the two Ramachandran densities (bits).

    Symmetric and bounded in [0, 1], so it is readable as "fraction of the
    joint distribution that is wrong" — unlike KL, which is unbounded.
    """
    edges = np.linspace(-180, 180, bins + 1)
    P, _, _ = np.histogram2d(gt_phi, gt_psi, bins=[edges, edges])
    Q, _, _ = np.histogram2d(pr_phi, pr_psi, bins=[edges, edges])
    P = gaussian_filter(P, sigma) + 1e-10
    Q = gaussian_filter(Q, sigma) + 1e-10
    P /= P.sum()
    Q /= Q.sum()
    M = 0.5 * (P + Q)
    return float(0.5 * entropy(P.ravel(), M.ravel(), base=2)
                 + 0.5 * entropy(Q.ravel(), M.ravel(), base=2))


def circular_acf(angles_rad, max_lag=50):
    """Circular autocorrelation C(k) = <cos(theta_t - theta_{t+k})>.

    C(0) = 1 by construction. How fast it decays is the decorrelation timescale.
    """
    c, s = np.cos(angles_rad), np.sin(angles_rad)
    max_lag = min(max_lag, len(angles_rad) - 1)
    return np.array([
        float(np.mean(c[:len(c) - k] * c[k:] + s[:len(s) - k] * s[k:]))
        for k in range(max_lag + 1)
    ])


def angular_mad(gt_deg, pred_deg):
    """Mean absolute deviation in degrees, wrapped into [-180, 180]."""
    d = (pred_deg - gt_deg + 180.0) % 360.0 - 180.0
    return float(np.abs(d).mean())


def acf_error(gt_acf, pred_acf):
    """Mean absolute difference between two autocorrelation curves."""
    n = min(len(gt_acf), len(pred_acf))
    return float(np.abs(gt_acf[:n] - pred_acf[:n]).mean())


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_report(pid, gt, pred, metrics, out_dir):
    """Four-panel figure: Ramachandran, marginals, time series, autocorrelation."""
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.30, wspace=0.28)

    # --- 1. Ramachandran overlay -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    edges = np.linspace(-180, 180, config.KDE_GRID + 1)
    H, _, _ = np.histogram2d(gt["phi_full"], gt["psi_full"], bins=[edges, edges])
    H = gaussian_filter(H, 1.5)
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax.contourf(centres, centres, H.T, levels=12, cmap="Blues", alpha=0.75)
    ax.scatter(pred["phi"], pred["psi"], s=9, c="crimson", alpha=0.7,
               edgecolors="none", label="Stage 3 rollout")
    ax.set_xlim(-180, 180); ax.set_ylim(-180, 180)
    ax.set_xlabel("phi (deg)"); ax.set_ylabel("psi (deg)")
    ax.set_title(f"{pid} - Ramachandran\n(blue = full MD density, red = generated)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.2)

    # --- 2/3. Marginal distributions --------------------------------------
    for col, key, label in ((1, "phi", "phi"), (2, "psi", "psi")):
        ax = fig.add_subplot(gs[0, col])
        bins = np.linspace(-180, 180, 48)
        ax.hist(gt[f"{key}_full"], bins=bins, density=True, alpha=0.5,
                color="steelblue", label="MD (full)")
        ax.hist(pred[key], bins=bins, density=True, histtype="step", lw=2,
                color="crimson", label="Stage 3")
        ax.set_xlabel(f"{label} (deg)"); ax.set_ylabel("density")
        ax.set_title(f"{label} marginal  |  KL = {metrics[f'kl_{key}_full']:.3f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # --- 4/5. Time series --------------------------------------------------
    for col, key in ((0, "phi"), (1, "psi")):
        ax = fig.add_subplot(gs[1, col])
        series = pred[f"{key}_first"]          # one path, not the pooled cloud
        t = np.arange(len(series)) * config.FRAME_LAG_PS / 1000.0
        ax.plot(t, gt[f"{key}_window"], lw=0.9, color="steelblue", alpha=0.8, label="MD")
        ax.plot(t, series, lw=0.9, color="crimson", alpha=0.8, label="Stage 3")
        ax.set_xlabel("time (ns)"); ax.set_ylabel(f"{key} (deg)")
        ax.set_ylim(-185, 185)
        ax.set_title(f"{key} trajectory")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # --- 6. Autocorrelation ------------------------------------------------
    ax = fig.add_subplot(gs[1, 2])
    lags = np.arange(len(gt["acf_phi"])) * config.FRAME_LAG_PS
    ax.plot(lags, gt["acf_phi"], color="steelblue", lw=2, label="MD phi")
    ax.plot(lags, gt["acf_psi"], color="steelblue", lw=2, ls="--", label="MD psi")
    ax.plot(lags[:len(pred["acf_phi"])], pred["acf_phi"], color="crimson", lw=2,
            label="Stage 3 phi")
    ax.plot(lags[:len(pred["acf_psi"])], pred["acf_psi"], color="crimson", lw=2,
            ls="--", label="Stage 3 psi")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (ps)"); ax.set_ylabel("circular autocorrelation")
    ax.set_title(f"dihedral ACF  |  error = {metrics['acf_err']:.3f}\n"
                 "(flat-at-zero = i.i.d. sampling, not dynamics)")
    ax.legend(fontsize=7); ax.grid(alpha=0.2)

    fig.suptitle(f"Stage 3 - {pid}: learned propagator vs. ground-truth MD",
                 fontsize=14, y=0.98)
    path = os.path.join(out_dir, f"{pid}_stage3_report.png")
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# Per-peptide evaluation
# ──────────────────────────────────────────────────────────────

def evaluate_peptide(pid, model, scales, args, device):
    state = build_state(pid, device)
    backbone = parse_backbone_from_pdb(state["pdb_path"])

    out_dir = os.path.join(config.RESULTS_DIR, pid)
    os.makedirs(out_dir, exist_ok=True)

    # --- generate ----------------------------------------------------------
    # With --samples > 1 we roll out several independent trajectories from the
    # same start frame and pool them for the DISTRIBUTION metrics only. A single
    # 300-frame path is a small sample of the model's stationary distribution, so
    # pooling lowers the variance of the KL estimate (it does not lower KL if the
    # model is genuinely wrong). Time series, MAD and the ACF stay per-trajectory
    # — pooling those would be meaningless. Default 1 keeps the original metric.
    phis, psis, acfs_phi, acfs_psi, rmsds = [], [], [], [], []
    traj0 = vels0 = None
    for s in range(args.samples):
        traj, vels = autoregressive_rollout(
            model, state, scales, n_frames=args.frames, start_frame=args.start_frame,
            flow_steps=args.flow_steps, temperature=args.temperature,
            bond_fix=args.bond_fix, seed=args.seed + s, device=device,
        )
        if s == 0:
            traj0, vels0 = traj, vels
        p, q = compute_phi_psi_from_positions(traj, backbone)
        phis.append(np.degrees(p))
        psis.append(np.degrees(q))
        acfs_phi.append(circular_acf(p, args.max_lag))
        acfs_psi.append(circular_acf(q, args.max_lag))
        rmsds.append(bond_length_rmsd(traj, state["bonds"], state["eq_len"]))

    traj, vels = traj0, vels0
    pr_phi_d = np.concatenate(phis)          # pooled, for distribution metrics
    pr_psi_d = np.concatenate(psis)

    gt_all = state["positions"]
    s0, n = args.start_frame, args.frames
    gt_win_phi, gt_win_psi = compute_phi_psi_from_positions(gt_all[s0:s0 + n], backbone)
    gt_full_phi, gt_full_psi = compute_phi_psi_from_positions(
        gt_all[::args.gt_stride], backbone)

    gt = {
        "phi_window": np.degrees(gt_win_phi), "psi_window": np.degrees(gt_win_psi),
        "phi_full": np.degrees(gt_full_phi), "psi_full": np.degrees(gt_full_psi),
        "acf_phi": circular_acf(gt_win_phi, args.max_lag),
        "acf_psi": circular_acf(gt_win_psi, args.max_lag),
    }
    pred = {
        # pooled over samples — distribution metrics and the Ramachandran cloud
        "phi": pr_phi_d, "psi": pr_psi_d,
        # first trajectory only — time series and short-horizon tracking
        "phi_first": phis[0], "psi_first": psis[0],
        # ACF is a per-trajectory quantity; average the curves, never the samples
        "acf_phi": np.mean(acfs_phi, axis=0),
        "acf_psi": np.mean(acfs_psi, axis=0),
    }

    # --- metrics -----------------------------------------------------------
    w = args.mad_window
    metrics = {
        "peptide": pid,
        "n_atoms": state["n_atoms"],
        "split": state["split"],
        "frames": n,
        "samples": args.samples,
        # distribution, vs the equilibrium density of the full MD trajectory
        "kl_phi_full": kl_divergence_1d(gt["phi_full"], pred["phi"]),
        "kl_psi_full": kl_divergence_1d(gt["psi_full"], pred["psi"]),
        # distribution, vs the same-length MD window (matched sample count)
        "kl_phi_window": kl_divergence_1d(gt["phi_window"], pred["phi"]),
        "kl_psi_window": kl_divergence_1d(gt["psi_window"], pred["psi"]),
        "js2d_full": js_divergence_2d(gt["phi_full"], gt["psi_full"],
                                      pred["phi"], pred["psi"]),
        # dynamics
        "acf_err": 0.5 * (acf_error(gt["acf_phi"], pred["acf_phi"])
                          + acf_error(gt["acf_psi"], pred["acf_psi"])),
        # short-horizon tracking — first trajectory only
        "mad_phi": angular_mad(gt["phi_window"][:w], pred["phi_first"][:w]),
        "mad_psi": angular_mad(gt["psi_window"][:w], pred["psi_first"][:w]),
        "mad_window": w,
        # physicality — averaged over samples
        "bond_rmsd_A": float(np.mean(rmsds)) * 10.0,
    }

    # --- outputs -----------------------------------------------------------
    np.savez_compressed(
        os.path.join(out_dir, f"{pid}_stage3_dihedrals.npz"),
        pred_phi=pred["phi"], pred_psi=pred["psi"],
        pred_phi_first=pred["phi_first"], pred_psi_first=pred["psi_first"],
        gt_phi_window=gt["phi_window"], gt_psi_window=gt["psi_window"],
        gt_phi_full=gt["phi_full"], gt_psi_full=gt["psi_full"],
        positions=traj, velocities=vels,
    )
    plot_path = plot_report(pid, gt, pred, metrics, out_dir)

    with open(os.path.join(out_dir, f"{pid}_stage3_metrics.txt"), "w") as f:
        f.write(f"Stage 3 evaluation - {pid}\n")
        f.write("=" * 60 + "\n")
        f.write(f"atoms            : {metrics['n_atoms']}\n")
        f.write(f"split            : {metrics['split']}\n")
        f.write(f"frames generated : {n}  "
                f"({n * config.FRAME_LAG_PS / 1000:.2f} ns simulated)\n")
        f.write(f"rollout samples  : {args.samples}"
                f"{'  (pooled for the distribution metrics)' if args.samples > 1 else ''}\n\n")
        f.write("DISTRIBUTION (lower is better)\n")
        f.write(f"  KL phi  vs full MD     : {metrics['kl_phi_full']:.4f}\n")
        f.write(f"  KL psi  vs full MD     : {metrics['kl_psi_full']:.4f}\n")
        f.write(f"  KL phi  vs MD window   : {metrics['kl_phi_window']:.4f}\n")
        f.write(f"  KL psi  vs MD window   : {metrics['kl_psi_window']:.4f}\n")
        f.write(f"  2-D Jensen-Shannon     : {metrics['js2d_full']:.4f} bits\n\n")
        f.write("DYNAMICS (lower is better)\n")
        f.write(f"  dihedral ACF error     : {metrics['acf_err']:.4f}\n\n")
        f.write(f"SHORT-HORIZON TRACKING (first {w} frames, "
                f"{w * config.FRAME_LAG_PS:.0f} ps)\n")
        f.write(f"  MAD phi                : {metrics['mad_phi']:.2f} deg\n")
        f.write(f"  MAD psi                : {metrics['mad_psi']:.2f} deg\n\n")
        f.write("PHYSICALITY\n")
        f.write(f"  bond-length RMSD       : {metrics['bond_rmsd_A']:.4f} A\n")

    return metrics, plot_path


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

def default_test_peptides(limit):
    stats = np.load(config.STATS_CACHE, allow_pickle=True)
    return [str(p) for p in stats["test"]][:limit]


def main():
    ap = argparse.ArgumentParser(description="Evaluate Stage-3 rollouts against MD.")
    ap.add_argument("--peptide", default=None, help="evaluate a single peptide")
    ap.add_argument("--peptides", nargs="+", default=None, help="explicit list")
    ap.add_argument("--limit", type=int, default=6, help="test peptides if none given")
    ap.add_argument("--frames", type=int, default=config.ROLLOUT_FRAMES)
    ap.add_argument("--samples", type=int, default=1,
                    help="independent rollouts pooled for the DISTRIBUTION metrics. "
                         "1 (default) reproduces the original single-trajectory "
                         "metric; higher values only reduce estimator variance")
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--gt-stride", type=int, default=5,
                    help="subsample stride for the full-MD reference density")
    ap.add_argument("--max-lag", type=int, default=50, help="ACF lag in frames")
    ap.add_argument("--mad-window", type=int, default=20,
                    help="frames over which short-horizon MAD is reported")
    ap.add_argument("--flow-steps", type=int, default=config.FLOW_STEPS)
    ap.add_argument("--temperature", type=float, default=config.FLOW_TEMP)
    ap.add_argument("--bond-fix", type=int, default=config.BOND_FIX_ITERS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    config.ensure_dirs()

    if args.peptide:
        peptides = [args.peptide]
    elif args.peptides:
        peptides = args.peptides
    else:
        peptides = default_test_peptides(args.limit)

    model, ckpt = load_propagator(device=device)
    scales = flow.load_scales(device)

    print("=" * 88)
    print("STEP 4 - EVALUATE STAGE-3 ROLLOUTS")
    print("=" * 88)
    print(f"checkpoint : epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.5f}")
    print(f"rollout    : {args.frames} frames "
          f"({args.frames * config.FRAME_LAG_PS / 1000:.2f} ns), "
          f"{args.flow_steps} flow steps, T={args.temperature}")
    print(f"peptides   : {', '.join(peptides)}\n")

    rows = []
    for pid in peptides:
        try:
            m, plot = evaluate_peptide(pid, model, scales, args, device)
        except Exception as e:
            print(f"  {pid}: FAILED - {e}")
            continue
        rows.append(m)
        print(f"  {pid}: KL(phi)={m['kl_phi_full']:.3f}  KL(psi)={m['kl_psi_full']:.3f}  "
              f"JS2D={m['js2d_full']:.3f}  ACFerr={m['acf_err']:.3f}  "
              f"bond={m['bond_rmsd_A']:.3f}A")

    if not rows:
        return

    print()
    print(f"{'pep':<5}{'KLphi':>9}{'KLpsi':>9}{'JS2D':>9}{'ACFerr':>9}"
          f"{'MADphi':>9}{'MADpsi':>9}{'bondA':>9}")
    print("-" * 88)
    for m in rows:
        print(f"{m['peptide']:<5}{m['kl_phi_full']:>9.3f}{m['kl_psi_full']:>9.3f}"
              f"{m['js2d_full']:>9.3f}{m['acf_err']:>9.3f}"
              f"{m['mad_phi']:>9.1f}{m['mad_psi']:>9.1f}{m['bond_rmsd_A']:>9.3f}")
    print("-" * 88)
    print(f"{'mean':<5}"
          f"{np.mean([m['kl_phi_full'] for m in rows]):>9.3f}"
          f"{np.mean([m['kl_psi_full'] for m in rows]):>9.3f}"
          f"{np.mean([m['js2d_full'] for m in rows]):>9.3f}"
          f"{np.mean([m['acf_err'] for m in rows]):>9.3f}"
          f"{np.mean([m['mad_phi'] for m in rows]):>9.1f}"
          f"{np.mean([m['mad_psi'] for m in rows]):>9.1f}"
          f"{np.mean([m['bond_rmsd_A'] for m in rows]):>9.3f}")

    summary = os.path.join(config.RESULTS_DIR, "stage3_summary.csv")
    with open(summary, "w") as f:
        keys = ["peptide", "split", "n_atoms", "frames", "samples",
                "kl_phi_full", "kl_psi_full",
                "kl_phi_window", "kl_psi_window", "js2d_full", "acf_err",
                "mad_phi", "mad_psi", "bond_rmsd_A"]
        f.write(",".join(keys) + "\n")
        for m in rows:
            f.write(",".join(str(m[k]) for k in keys) + "\n")

    print(f"\nper-peptide figures and reports: {config.RESULTS_DIR}/<PEPTIDE>/")
    print(f"summary table                  : {summary}")


if __name__ == "__main__":
    main()
