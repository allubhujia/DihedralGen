"""
train.py — STEP 3. Train the flow-matching propagator.

Straightforward supervised training: the flow-matching objective is an MSE, so
there is no adversarial game, no KL term, and no curriculum needed. Every epoch
draws STEPS_PER_EPOCH random batches of transitions (there are far more
transitions than we need per epoch, so a full pass would be wasteful).

Validation uses a FIXED noise seed. Flow-matching loss is stochastic in both
tau and eps, so without pinning the seed the epoch-to-epoch validation curve is
dominated by sampling noise and "best checkpoint" becomes a lottery.

Run:
    python -m temporal_dynamics.train
    python -m temporal_dynamics.train --epochs 60 --batch-size 48
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from temporal_dynamics import config, flow
from temporal_dynamics.dataset import make_loader
from temporal_dynamics.model import EquivariantPropagator


def to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


@torch.no_grad()
def validate(model, loader, scales, device, weights=None, state_noise=None,
             max_batches=40, seed=1234):
    """Fixed-seed validation loss so the curve is comparable across epochs.

    Validation carries the same state-noise augmentation as training (drawn from
    the fixed generator, so it is identical every epoch). That is deliberate:
    the augmented objective is the one being optimised, and the model's ability
    to correct off-manifold states is precisely what the rollout depends on, so
    it should be part of what selects the checkpoint.
    """
    model.eval()
    gen = torch.Generator(device=device).manual_seed(seed)
    total, n_pos, n_vel, count = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = to_device(batch, device)
        loss, parts = flow.flow_matching_loss(model, batch, scales, generator=gen,
                                              weights=weights, state_noise=state_noise)
        total += loss.item()
        n_pos += parts["pos"]
        n_vel += parts["vel"]
        count += 1
    model.train()
    if count == 0:
        return float("nan"), 0.0, 0.0
    return total / count, n_pos / count, n_vel / count


def main():
    ap = argparse.ArgumentParser(description="Train the Stage-3 propagator.")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=config.LR)
    ap.add_argument("--steps-per-epoch", type=int, default=config.STEPS_PER_EPOCH)
    ap.add_argument("--state-noise", type=float, default=config.STATE_NOISE_NM,
                    help="nm; upper end of the input-state corruption ramp that "
                         "teaches the model to steer back onto the manifold. 0 = off")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--threads", type=int, default=os.cpu_count(),
                    help="CPU threads; torch defaults to half the cores, which "
                         "leaves ~30%% of throughput on the table for this model")
    args = ap.parse_args()

    if args.device == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config.ensure_dirs()

    print("=" * 74)
    print("STEP 3 - TRAIN FLOW-MATCHING PROPAGATOR")
    print("=" * 74)

    train_ds, train_dl = make_loader("train", batch_size=args.batch_size, shuffle=True)
    val_ds, val_dl = make_loader("val", batch_size=args.batch_size, shuffle=False)

    print(f"train : {len(train_ds.peptides):>3} peptides, {len(train_ds):>7,} transitions")
    print(f"val   : {len(val_ds.peptides):>3} peptides, {len(val_ds):>7,} transitions")

    scales = flow.load_scales(device)
    weights = flow.loss_weights(device)

    model = EquivariantPropagator().to(device)
    print(f"model : {model.num_parameters():,} parameters "
          f"({config.NUM_LAYERS} layers, hidden={config.HIDDEN_DIM}, "
          f"vec_ch={config.VEC_CHANNELS})")
    print(f"target: {flow.num_channels()} vector channel(s) — "
          f"dpos{', vel_t+1' if config.PREDICT_VELOCITY else ''}"
          f"   vel_t input: {config.USE_VELOCITY_INPUT}")
    print(f"scale : per-element {config.PER_ELEMENT_SCALE}  "
          f"({', '.join(f'{e}={scales[i,0]:.4f}' for i, e in enumerate(config.ELEMENTS))})")
    print(f"loss  : mass**{config.LOSS_MASS_POWER} weighting  "
          f"({', '.join(f'{e}={weights[i]/weights.mean():.2f}' for i, e in enumerate(config.ELEMENTS))})")
    print(f"noise : state augmentation up to {args.state_noise:.3f} nm")
    print(f"device: {device}\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs,
                                                       eta_min=args.lr * 0.05)

    stream = infinite(train_dl)
    history = {"train": [], "val": [], "val_pos": [], "val_vel": []}
    best = float("inf")

    print(f"{'epoch':>6}{'train':>11}{'val':>11}{'val_pos':>10}{'val_vel':>10}"
          f"{'lr':>10}{'sec':>7}")
    print("-" * 74)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0

        for step in range(args.steps_per_epoch):
            batch = to_device(next(stream), device)
            loss, _ = flow.flow_matching_loss(model, batch, scales,
                                              weights=weights,
                                              state_noise=args.state_noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            opt.step()
            running += loss.item()

            # Heartbeat. An epoch is minutes long, so without this the terminal
            # sits blank and a stalled run is indistinguishable from a slow one.
            # flush=True because stdout is block-buffered whenever it is not a
            # TTY (piped, redirected, or run under some terminals), and a few
            # hundred bytes of header will sit in that buffer indefinitely.
            if step % 25 == 0:
                done = step + 1
                rate = (time.time() - t0) / done
                print(f"\r    epoch {epoch}  step {done}/{args.steps_per_epoch}"
                      f"  loss {running / done:.4f}"
                      f"  {rate:.2f}s/step"
                      f"  eta {rate * (args.steps_per_epoch - done) / 60:.1f} min   ",
                      end="", flush=True)

        print("\r" + " " * 78 + "\r", end="", flush=True)
        train_loss = running / args.steps_per_epoch
        val_loss, val_pos, val_vel = validate(model, val_dl, scales, device,
                                              weights, args.state_noise)
        sched.step()

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["val_pos"].append(val_pos)
        history["val_vel"].append(val_vel)

        flag = ""
        if val_loss < best:
            best = val_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "config": {
                    "hidden": config.HIDDEN_DIM,
                    "vec_ch": config.VEC_CHANNELS,
                    "num_layers": config.NUM_LAYERS,
                    "cond_dim": config.COND_DIM,
                    "num_out_vec": flow.num_channels(),
                    "use_vel_input": config.USE_VELOCITY_INPUT,
                    "state_noise": args.state_noise,
                },
            }, config.CKPT_PATH)
            flag = "  *"

        print(f"{epoch:>6}{train_loss:>11.5f}{val_loss:>11.5f}{val_pos:>10.5f}"
              f"{val_vel:>10.5f}{sched.get_last_lr()[0]:>10.2e}"
              f"{time.time() - t0:>7.1f}{flag}", flush=True)

    # --- sanity: equivariance must survive training ------------------------
    batch = to_device(next(iter(val_dl)), device)
    err = flow.equivariance_error(model, batch, scales)
    print(f"\nrotation-equivariance error: {err:.2e}  "
          f"({'ok' if err < 1e-4 else 'BROKEN'})")

    # --- loss curve ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train"], label="train")
    ax[0].plot(history["val"], label="val")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("flow-matching MSE")
    ax[0].set_title("Stage 3 - flow matching loss"); ax[0].legend(); ax[0].grid(alpha=.3)

    ax[1].plot(history["val_pos"], label="displacement channel")
    if config.PREDICT_VELOCITY:
        ax[1].plot(history["val_vel"], label="velocity channel")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("val MSE")
    ax[1].set_title("per-channel validation loss"); ax[1].legend(); ax[1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(config.LOSS_CURVE, dpi=140)
    print(f"loss curve : {config.LOSS_CURVE}")
    print(f"best ckpt  : {config.CKPT_PATH}  (val {best:.5f})")
    print("\nNext:  python -m temporal_dynamics.evaluate")


if __name__ == "__main__":
    main()
