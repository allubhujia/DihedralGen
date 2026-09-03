# Stage 3 — Learned Trajectory Propagator

Stage 1 learned **what** a peptide is (a 32-d graph embedding `z`).
Stage 2 learned **where** it goes (the equilibrium φ/ψ distribution).
Stage 3 learns **how it moves**: a propagator that carries the full Cartesian
state — positions **and velocities** — forward one saved frame at a time, so a
**300-frame trajectory** can be rolled out autoregressively from a single
starting structure.

---

## What the data actually looks like

Run `python -m temporal_dynamics.check_velocities` and you get this. Three facts
drive every design decision below.

| Question | Answer | Consequence |
|---|---|---|
| Do the `.npz` files contain velocities? | **Yes** — `velocities [T, N, 3]` in nm/ps, present in every file of all three splits | Nothing has to be synthesised. A finite-difference fallback exists in `velocity_utils.load_positions_velocities()` but is never triggered on this dataset. |
| What is the frame spacing? | **5 ps** (10 000 MD steps) | This is not an MD timestep. Stage 3 is a *transition model over a 5 ps lag*, not an integrator. |
| Is velocity usable as an integrator? | **No.** corr(`v(t)`, `Δx/Δt`) = **−0.005** | `x + v·dt` is meaningless here. Velocity is a **state channel**: read as input, co-predicted as output. |
| How much motion is global tumbling? | Per-atom displacement **0.40 nm** centred vs **0.078 nm** after Kabsch | Tumbling is **5×** the internal motion and is not a function of the molecule. Every frame is Kabsch-aligned onto its predecessor. Dihedrals are internal coordinates, so nothing observable is lost. |

---

## The model, and why this one

### Why not regress the next frame

Over a 5 ps lag the transition is genuinely stochastic — the same `(pos, vel)`
can be followed by many different next frames. An MSE-trained model learns the
**conditional mean** of that spread. Roll a conditional mean forward 300 times
and all motion damps out; the Ramachandran plot collapses to a single blob.
This is the same failure Stage 2 hit with per-frame regression, for the same
reason.

### What it does instead — conditional flow matching

Learn the whole conditional distribution with rectified flow. Define a straight
path from Gaussian noise to the true target,

```
Y_τ = (1 − τ)·ε + τ·Y ,    ε ~ N(0, σ²),  τ ~ U(0,1)
```

whose velocity along the path is just `Y − ε`. Train the network to regress
that, then sample by integrating `dY/dτ = u_θ(Y_τ, τ | pos_t, vel_t, z)` from
fresh noise.

The training target is still an MSE, so it is as stable as regression — but
because the target depends on `ε`, the learned field transports the *entire*
noise distribution onto the *entire* conditional distribution. Different noise
draws give different plausible next frames, so the rollout keeps moving and
keeps hopping basins.

`Y` has two vector channels per atom:

| channel | quantity | units |
|---|---|---|
| 0 | `dpos` — displacement to the next frame | nm |
| 1 | `vel_tp1` — velocity at the next frame | nm/ps |

They differ in scale by ~20×, so both are divided by their dataset standard
deviation before training (`cache/stats.npz`) and multiplied back afterwards.

### The network — equivariant vector-channel GNN

A PaiNN/EGNN hybrid carrying two feature types per atom: invariant scalars
`h ∈ R^H` and equivariant vectors `V ∈ R^{C×3}`. Rotate the molecule and `V`
rotates with it; `h` does not. Conditioning on the **frozen** Stage-1 embedding
`z` and on the flow time `τ` enters through FiLM at every layer.

Rotation equivariance is asserted numerically after training (`flow.equivariance_error`,
measured **~10⁻⁷**, i.e. float32 machine precision).

---

## Files

| File | Role |
|---|---|
| `config.py` | Every hyperparameter and path |
| `velocity_utils.py` | Velocity loading, Kabsch alignment, COM removal, transition construction, bond geometry |
| `check_velocities.py` | **Step 0** — dataset audit (velocity presence, frame lag, tumbling ratio) |
| `prepare_data.py` | **Step 1** — writes the aligned transition cache |
| `adapter.py` | **Step 2** — freezes Stage 1 and caches `z` per peptide |
| `dataset.py` | Serves cached transitions as flat batched graph tensors |
| `model.py` | The equivariant vector field |
| `flow.py` | Flow-matching loss, ODE sampler, equivariance check |
| `train.py` | **Step 3** — training loop |
| `rollout.py` | Autoregressive 300-frame generation + multi-MODEL PDB writer |
| `evaluate.py` | **Step 4** — Ramachandran, KL, ACF, MAD, bond RMSD |

---

## How to run

All commands are run from the **repository root** (not from inside
`temporal_dynamics/`), because the modules import `scripts.*` and `models.*`.

```bash
cd MolGraph-AE
source .venv/bin/activate        # WSL / Linux / macOS
```

### Step 0 — audit the data (optional but recommended)

Confirms velocities are present, measures the frame lag, and quantifies tumbling.
Read-only; writes nothing.

```bash
python -m temporal_dynamics.check_velocities
python -m temporal_dynamics.check_velocities --split train --max 40
```

### Step 1 — build the transition cache

Loads raw trajectories, Kabsch-aligns every frame onto its predecessor, removes
COM velocity, and writes one `.npz` per peptide to `cache/transitions/` plus the
normalisation scales to `cache/stats.npz`.

Doing the alignment once here rather than in the DataLoader matters — it is an
SVD per transition and would otherwise dominate training time.

```bash
python -m temporal_dynamics.prepare_data
python -m temporal_dynamics.prepare_data --max-train 100 --max-frames 3000
```

### Step 2 — cache the frozen Stage-1 embeddings

Requires `encoder.pt` in the repo root (Stage 1 must already have been trained).
`z` depends only on the static chemical graph, so it is computed once per
peptide instead of once per training sample.

```bash
python -m temporal_dynamics.adapter
```

### Step 3 — train the propagator

```bash
python -m temporal_dynamics.train
python -m temporal_dynamics.train --epochs 60 --batch-size 48
```

Writes `best_propagator.pt` and `loss_curve.png`. Validation uses a **fixed
noise seed** — the flow-matching loss is stochastic in both `τ` and `ε`, so
without pinning it the validation curve is dominated by sampling noise and
"best checkpoint" becomes a lottery.

### Step 4 — evaluate

```bash
python -m temporal_dynamics.evaluate                        # default test peptides
python -m temporal_dynamics.evaluate --peptide AC
python -m temporal_dynamics.evaluate --peptides AC AD AH --frames 300
```

Writes per-peptide figures, metric reports, and `results/stage3_summary.csv`.

### Generating a trajectory on its own

```bash
python -m temporal_dynamics.rollout --peptide AC
python -m temporal_dynamics.rollout --peptide AC --samples 3 --pdb
```

`--pdb` writes a multi-MODEL PDB viewable in PyMOL/VMD or `scripts/visualization.py`.
Each sample draws fresh noise, so repeated calls give different — equally valid —
trajectories, exactly as re-running MD with a different random seed would.

---

## Metrics, and what each one catches

| Metric | Measures | Why it is here |
|---|---|---|
| **Ramachandran overlay** | Conformational basins visited | The qualitative check; directly comparable to Stage 2 |
| **KL divergence** (φ, ψ) | Marginal distribution match | Same criterion Stage 2 was scored on |
| **2-D Jensen-Shannon** | Joint φ/ψ distribution | Symmetric and bounded in [0,1], unlike KL which is unbounded |
| **Dihedral autocorrelation** | **Decorrelation timescale** | The one that separates Stage 3 from Stage 2. A model drawing i.i.d. samples from the equilibrium distribution scores *perfectly* on every distributional metric, but its ACF collapses to zero in one frame. No distribution metric can see this. |
| **Short-horizon MAD** | Frame-by-frame tracking over the first ~20 frames | MD is chaotic, so a sample path *is expected* to diverge from any specific GT path. Large MAD at long horizon is **not** a failure. Reported over a short window only. |
| **Bond-length RMSD** | Molecular integrity over the rollout | Catches a molecule that slowly inflates or collapses — which a Ramachandran plot will happily hide |

---

## Rollout guard-rail

300 stochastic steps give small per-step biases 300 chances to accumulate.
`rollout.project_bonds` runs a few Jacobi constraint-relaxation sweeps that pull
covalent bond lengths back toward equilibrium when they drift past
`BOND_TOL_NM`. It touches only bonded pairs, so it cannot manufacture backbone
dihedrals — those remain entirely the model's output.

Two implementation details keep it stable, and getting either wrong makes the
rollout diverge within ~20 frames:

* the bond list is stored **bidirectionally**, so it must be collapsed to one
  entry per physical bond, or every bond is corrected twice per sweep;
* a carbon has up to **4** bonds all pulling on it in the same sweep. Summing
  those corrections overshoots and oscillates outward — dividing each atom's
  accumulated correction by its bond count turns the sum into an average, which
  contracts instead.

Disable with `--bond-fix 0` to see the unconstrained model.

---

## Tuning knobs

| Flag | Effect |
|---|---|
| `--temperature` | Scales the sampling noise. `>1` broadens the visited distribution, `<1` sharpens toward the conditional mean. Start at `1.0`. |
| `--flow-steps` | Euler steps per transition (default 20). More steps = more faithful transport, linearly slower. |
| `--bond-fix` | Constraint sweeps per rollout step; `0` disables. |
| `--frames` | Rollout length. 300 frames = 1.5 ns of simulated time at the 5 ps lag. |

## Cost

CPU-only is the assumed default (`MAX_TRAIN_PEPTIDES = 60`, `MAX_FRAMES = 2000`).
Rollout cost is `frames × flow_steps` forward passes — 300 × 20 = 6000 per
trajectory. Drop `--flow-steps` to 10 for quick iteration. Raise the dataset
caps in `config.py` if a GPU is available.
