<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyG-3C2179?style=for-the-badge&logo=pyg&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧬 MolGraph-AE — Graph Autoencoding, Distribution Matching & Equivariant Trajectory Propagation for Dipeptides

> A **three-stage** deep-learning pipeline that learns (1) **permutation-invariant latent embeddings** of dipeptide molecular graphs, (2) their **equilibrium φ/ψ distribution** under a Sliced-Wasserstein objective, and (3) their **actual dynamics** — a rotation-equivariant flow-matching propagator that rolls a real starting structure forward in time.
>
> Research internship at **IISER Pune** under **Dr. Arnab Mukherjee**, using the [Microsoft Timewarp](https://github.com/microsoft/timewarp) `2AA-complete` dataset.

---

## 📋 Table of Contents

- [The Three Stages](#-the-three-stages)
- [What the Data Actually Looks Like](#-what-the-data-actually-looks-like)
- [Repository Structure](#-repository-structure)
- [Stage 1 — Molecular Graph Autoencoder](#-stage-1--molecular-graph-autoencoder)
- [Stage 2 — Generative Dihedral Distribution Model](#-stage-2--generative-dihedral-distribution-model)
- [Stage 2b — Seeded Generation](#-stage-2b--seeded-generation)
- [Stage 3 — Equivariant Flow-Matching Propagator](#-stage-3--equivariant-flow-matching-propagator)
- [Benchmark — Stage 2 vs Stage 3](#-benchmark--stage-2-vs-stage-3)
- [Results](#-results)
- [Evaluation Metrics](#-evaluation-metrics)
- [Getting Started](#-getting-started)
- [Full Command Reference](#-full-command-reference)
- [Report](#-report)
- [Design Decisions](#-design-decisions)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)

---

## 🔬 The Three Stages

Each stage answers a different question:

| Stage | Question | Output | Directory |
|-------|----------|--------|-----------|
| **1** | *What* is this molecule? | latent embedding `z ∈ ℝ³²` | `models/` |
| **2** | *Where* does it go? | equilibrium φ/ψ distribution | `trajectory_pre_/` |
| **3** | *How* does it move? | time-ordered Cartesian trajectory | `temporal_dynamics/` |

Stage 2 is a **distribution model** — no starting structure, no time axis.
Stage 3 is a **propagator** — seeded with a real conformation, steps forward.
That distinction drives the entire evaluation.

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     MolGraph-AE — THREE-STAGE PIPELINE                   │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PDB + NPZ ──► processing.py ──► PyG Data (x, edge_index, pos)          │
│                        │                                                 │
│   ╔════════════════════▼═══════ STAGE 1: REPRESENTATION ═══════════╗     │
│   ║  MolecularGNN (3× GCNConv) ──► global_mean_pool ──► MLP ──► z   ║     │
│   ║  loss = BCE(adjacency) + λ·cosine-similarity separation         ║     │
│   ╚════════════════════│════════════════════════════════════════════╝     │
│                        │ frozen z (32-d)                                 │
│           ┌────────────┴────────────┐                                    │
│           ▼                         ▼                                    │
│  ╔═══ STAGE 2 ═══════╗    ╔═══ STAGE 3 ════════════════════════════╗     │
│  ║ Transformer over  ║    ║ Equivariant vector-channel GNN         ║     │
│  ║ 300 frames        ║    ║ + conditional flow matching            ║     │
│  ║ Sliced-Wasserstein║    ║ p(Δx, v_{t+1} | x_t, v_t, z)           ║     │
│  ╚═══════│═══════════╝    ╚═══════════════│════════════════════════╝     │
│          ▼                                ▼                              │
│   300 φ/ψ frames                  T frames of (x, v)                     │
│   (no time axis)                  (time-ordered, seeded)                 │
│          └──────────────┬─────────────────┘                              │
│                         ▼                                                │
│        benchmark/ ──► Ramachandran · KL · JS · ACF · bond RMSD           │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 What the Data Actually Looks Like

Run `python -m temporal_dynamics.check_velocities` to reproduce these. **Three measurements determined the entire Stage-3 design**, and each is measured, not assumed:

| Question | Answer | Consequence |
|---|---|---|
| Do the `.npz` files contain velocities? | **Yes** — `velocities [9800, N, 3]` nm/ps, in every file of all three splits | Nothing needs synthesising. A finite-difference fallback exists but never fires. |
| What is the frame spacing? | **5 ps** (10 000 MD steps) | Stage 3 is a *transition model over a 5 ps lag*, not an integrator. |
| Is velocity usable as an integrator? | **No.** corr(`v(t)`, `Δx/Δt`) = **−0.005** | `x + v·dt` is meaningless. Velocity is a **state channel**: read in, re-emitted. |
| How much motion is global tumbling? | **0.40 nm** centred vs **0.078 nm** after Kabsch | Tumbling is **5×** the internal motion. Every frame is Kabsch-aligned onto its predecessor. Dihedrals are internal coordinates, so nothing observable is lost. |

---

## 📁 Repository Structure

```
MolGraph-AE/
│
├── 📄 README.md                       # This file
├── 📄 encoder.pt / decoder.pt         # Stage-1 checkpoints (consumed by 2 and 3)
│
├── 🧠 models/                         # STAGE 1
│   ├── gnn.py                         # MolecularGNN — stacked GCNConv
│   ├── encoder.py                     # graph / node → latent embedding
│   ├── decoder.py                     # InnerProductDecoder (z·zᵀ adjacency)
│   ├── loss.py                        # BCE reconstruction + cosine separation
│   └── pipeline_auto.py               # train + validate, writes encoder.pt
│
├── ⚙️ scripts/                        # Shared utilities
│   ├── processing.py                  # PDB+NPZ → PyG graph
│   ├── backbone_utils.py              # φ/ψ dihedral computation
│   └── visualization.py               # py3Dmol 3-D viewer
│
├── 🔮 trajectory_pre_/                # STAGE 2 (+ 2b)
│   ├── train_trajectory.py            # TrajectorySeqModel + Sliced-Wasserstein
│   ├── predict_trajectory.py          # generate φ/ψ ensemble + GT PDB
│   ├── best_trajectory.pt             # Stage-2 checkpoint (300 frames, fixed)
│   ├── seeded_model.py                # ★ Stage 2b: SeededTrajectoryModel
│   ├── train_seeded.py                # ★ Stage 2b training
│   ├── predict_seeded.py              # ★ "given frame 50, predict frame 100"
│   └── best_seeded.pt                 # ★ Stage-2b checkpoint
│
├── 🌊 temporal_dynamics/              # STAGE 3
│   ├── check_velocities.py            # STEP 0 — dataset audit
│   ├── prepare_data.py                # STEP 1 — aligned transition cache
│   ├── adapter.py                     # STEP 2 — frozen Stage-1 z cache
│   ├── train.py                       # STEP 3 — flow-matching training
│   ├── evaluate.py                    # STEP 4 — Ramachandran/KL/ACF/bond
│   ├── config.py                      # every hyperparameter and path
│   ├── velocity_utils.py              # Kabsch, COM removal, bond geometry
│   ├── dataset.py                     # batched graph tensors
│   ├── model.py                       # equivariant vector-channel GNN
│   ├── flow.py                        # flow-matching loss + ODE sampler
│   ├── rollout.py                     # autoregressive generation + PDB writer
│   ├── best_propagator.pt             # current Stage-3 checkpoint
│   ├── best_propagator_singlelag.pt   # preserved reference checkpoint
│   └── README.md                      # Stage-3 detail
│
├── 📊 benchmark/                      # ★ STAGE 2 vs STAGE 3
│   ├── run_benchmark.py               # CLI: peptide, frames, start-frame
│   ├── stage2.py / stage3.py          # thin adapters over each model
│   ├── plots.py                       # Ramachandran + KL comparison figures
│   ├── bench_config.py                # paths and defaults
│   ├── results/{PEPTIDE}_f{N}_s{S}/   # figures, metrics.txt, summary.csv
│   └── README.md
│
├── 📝 report/                         # ★ LaTeX report (IISER Pune)
│   └── main.tex
│
├── 📊 trajectory_analysis(main)/      # Stage-2 standalone analysis
├── 📈 plots_1/                        # Stage-1 loss plots
├── 📖 Theoretical/                    # Research notes
├── 🎨 visualizations/                 # Interactive 3-D HTML viewers
├── 🌐 unseen_peptide_test/            # HuggingFace generalisation pipeline
└── 💾 timewarp_data/                  # 2AA-complete (gitignored)
```

★ = added or substantially reworked in the latest iteration.

---

## 🧠 Stage 1 — Molecular Graph Autoencoder

Self-supervised representation of each dipeptide's chemical graph.

**Node features (8-d):** `x, y, z` coordinates (nm) + one-hot over `H, C, N, O, S`.
**Edges:** static, built once from frame 0 with a 0.2 nm cutoff — chemical bonds don't break during MD, and per-frame edges cause "flickering" connectivity.

```
in_channels=8 → 3× [GCNConv + LayerNorm + SiLU] → 64-d nodes
              → global_mean_pool → MLP(64→64→32) → z ∈ ℝ³²
```

**Objective:**
```
L = BCEWithLogits(ẑ·ẑᵀ, A)   +   λ · ‖cossim(Z, Z) − I‖²      (λ = 0.01)
    ── reconstruction ──          ── separation ──
```

Reconstruction uses `pos_weight` balancing for the sparse-adjacency class imbalance; logits are clamped to `[-10, 10]`. The separation term is what stops all peptides collapsing to a common embedding — without it the conditioning signal for Stages 2 and 3 is useless.

**Verified after training:** mean pairwise cosine similarity across 168 cached peptides ≈ **0.44** — the latent space genuinely separates molecules. Encoder is **15,520** parameters.

---

## 🔮 Stage 2 — Generative Dihedral Distribution Model

### Why distribution matching, not per-frame regression
The only conditioning signal is molecule identity, and MD frame ordering is not a function of the molecule. A per-frame MSE regressor collapses toward the mean of `(sin, cos)` — which lies *inside* the unit circle, making `atan2` ill-defined and turning the Ramachandran plot into uniform scatter. This was observed directly during development.

### Architecture — Transformer over frames
Each of 300 output frames is a token:
```
token_k = frame_pos[k] + proj(z_mol) + proj(noise_k)
        → TransformerEncoder (4 layers, d=128, 4 heads)
        → LayerNorm + Linear→4 → unit-circle projection
```
The unit-circle head L2-normalises each `(sin, cos)` pair, so every emitted angle is well defined.

### Sliced-Wasserstein loss
Both 300-point sets live in ℝ⁴ = `(sinφ, cosφ, sinψ, cosψ)`. Project onto 64 random directions, sort each projection, average the L1 gap. Matching all 1-D projections matches the full joint distribution — and because it compares *sorted sets* rather than paired frames it is permutation-invariant and **mode-covering**.

**595,076** parameters. Two structural limits follow directly from the design:
- **Fixed at 300 frames** — `frame_pos` is `[1, 300, 128]`. Longer requests are served by concatenating independent draws.
- **No seed, no time axis** — cannot be conditioned on a starting conformation, and frame order carries no temporal meaning.

---

## ⭐ Stage 2b — Seeded Generation

`seeded_model.py` lifts all three Stage-2 limits while leaving the original model untouched. It answers a question Stage 2 cannot be asked:

> *"Given peptide AC as it was at frame 50, what does frame 100 look like?"*

```
token_k = frame_pos[k] + proj(z_mol) + proj(seed_state) + proj(noise_k)
                                       ─────────────────
                                       (sinφ, cosφ, sinψ, cosψ) at the seed frame
```

The seed gets its **own** encoder rather than being concatenated onto the molecule vector, so conditioning strength on the two is independent and molecule identity cannot drown out where the window started.

**Composite loss** — each term stops a specific failure:

| Term | Purpose |
|---|---|
| `sw` | state distribution matches the real window |
| `anchor` | frame 0 equals the given seed — without it the model silently reverts to unconditional |
| `delta` | frame-to-frame *change* distribution matches, controlling path roughness |
| `acf` | circular autocorrelation matches, pinning the decorrelation timescale |

Also lifted: window length is now any value up to `max_frames=512`, and training samples from the **whole 9800-frame trajectory** rather than only the first 300.

---

## 🌊 Stage 3 — Equivariant Flow-Matching Propagator

### Why flow matching
Over a 5 ps lag the transition is genuinely stochastic. An MSE-trained model learns the **conditional mean**; rolling a conditional mean forward 300 times damps out all motion and collapses the Ramachandran plot to a point — the same failure Stage 2 avoided, for the same reason.

**Rectified flow** instead defines a straight path from noise to target:
```
Y_τ = (1−τ)·ε + τ·Y ,   ε ~ N(0, σ²),  τ ~ U(0,1)
```
whose velocity along the path is the constant `Y − ε`. The network regresses that; sampling integrates `dY/dτ = u_θ` from fresh noise over 20 Euler steps. The training target stays an MSE — as stable as regression — but because it depends on `ε`, the learned field transports the *entire* noise distribution onto the *entire* conditional distribution.

### The network
A PaiNN/EGNN hybrid carrying two feature types per atom:
```
h_i ∈ ℝ⁹⁶       invariant scalars   (rotate the molecule → unchanged)
V_i ∈ ℝ¹⁶ˣ³     equivariant vectors (rotate the molecule → they rotate)
```
Only three operations touch `V`, all rotation-commuting: channel mixing `WV`, invariant gating `s(h)·V`, and direction sums `Σⱼ c_ij·r̂_ij`. Scalars read vectors **only** through invariant quantities — channel norms and edge-direction projections — so no raw coordinate ever enters an invariant pathway.

Edges are **fully connected** within each molecule (≤ ~1100 for 33 atoms) with a binary covalent-bond flag as an edge feature, so steric effects are representable while topology stays explicit. Distances enter via 24 Gaussian RBFs. Conditioning on frozen `z` and flow time `τ` enters through **FiLM** at every layer.

**Equivariance is asserted numerically after every training run** — rotate all inputs by a random `R` and measure `‖R·u(x) − u(Rx)‖ / ‖u(x)‖`. Measured: **~1.5 × 10⁻⁷** (float32 machine precision).

### Three corrections that mattered

| Correction | Why |
|---|---|
| **Per-element target scaling** | Displacement is element-dependent (H 0.076, O 0.086, S 0.076, N 0.056, C 0.045 nm). One global std gives hydrogen several times carbon's squared amplitude, letting it dominate the gradient purely by moving further. |
| **Mass weighting** (`m^1.0`) | A dipeptide is ~half hydrogen. Mass weighting makes the loss kinetic-energy-like and matches the convention that structural accuracy is judged on heavy atoms. |
| **Channel weighting** (`[1.0, 0.15]`) | `v_{t+1}` is near-unlearnable at 5 ps, so an equal average spends half the gradient on irreducible noise. Velocity stays in the model and the rolled-forward state — it just loses its equal vote. |

### State-noise augmentation
Training corrupts `pos_t` with `σ ~ U(0, 0.02 nm)` and subtracts the same corruption from the target. This teaches the model to steer **back onto the data manifold** — the restoring force a purely on-manifold objective never provides.

### Rollout guard-rail
300 stochastic steps give small biases 300 chances to accumulate, usually showing up as a molecule that slowly inflates or collapses. A Jacobi constraint relaxation pulls covalent bonds back toward equilibrium. **Two details are essential** — getting either wrong made the rollout diverge to NaN within ~20 frames:

1. The bond list is stored **bidirectionally** — collapse to one entry per physical bond, or every bond is corrected twice per sweep.
2. A carbon has up to **4** bonds pulling on it simultaneously. Summing those corrections overshoots and oscillates outward; **dividing each atom's correction by its bond count** turns the sum into an average, which contracts.

Result: bond-length RMSD over 300 frames falls from **1.82 Å → 0.08 Å**.

---

## 📊 Benchmark — Stage 2 vs Stage 3

```bash
python -u -m benchmark.run_benchmark --peptide AC --frames 300 --start-frame 50
```
Or with no flags, it prompts for peptide / frames / start-frame.

**Outputs** → `benchmark/results/{PEPTIDE}_f{N}_s{S}/`:
`ramachandran_comparison.png`, `kl_divergence_comparison.png`, `benchmark_metrics.txt`, `benchmark_summary.csv`, `{PEPTIDE}_benchmark_dihedrals.npz`.

### ⚠️ The asymmetry, stated up front

|  | Stage 2 | Stage 3 |
|---|---|---|
| What it is | distribution model | propagator |
| Conditioned on | molecule identity only | starting structure + frozen `z` |
| Can be seeded? | **No** | Yes (`--start-frame`) |
| Frame count | **locked to 300** | any |
| Has a time axis? | **No** | Yes |

Consequences, handled explicitly in code rather than papered over:
- `--start-frame` is **accepted and discarded** by Stage 2, and the report says so.
- Frame counts are reshaped for Stage 2 — `n ≤ 300` takes an evenly-spaced subsample of its native 300; `n > 300` concatenates independent draws.
- **Both are scored against the full-MD equilibrium density** — the only reference Stage 2 can be fairly measured against.

---

## 📈 Results

300 frames, seed frame 50, scored against the full-MD equilibrium density.
**Noise floor at 300 samples = 0.048** (see [Evaluation Metrics](#-evaluation-metrics)).

| Peptide | Split | KL φ (S2 / S3) | KL ψ (S2 / S3) | JS 2-D (S2 / S3) |
|---------|-------|----------------|----------------|------------------|
| FW | train | **0.086** / 0.238 | **0.139** / 0.889 | **0.083** / 0.401 |
| GG | train | **0.278** / 0.360 | 0.274 / **0.179** | **0.198** / 0.231 |
| CM | val   | **0.098** / 0.274 | **0.111** / 0.220 | **0.086** / 0.173 |
| DR | val   | **0.095** / 0.487 | **0.220** / 0.592 | **0.110** / 0.323 |
| HH | val   | **0.069** / 0.362 | **0.192** / 0.257 | **0.098** / 0.200 |
| AC | test  | **0.171** / 0.387 | 0.223 / **0.175** | **0.127** / 0.168 |
| AD | test  | **0.070** / 0.306 | **0.221** / 0.227 | **0.097** / 0.177 |
| AH | test  | **0.074** / 0.307 | **0.173** / 0.636 | **0.092** / 0.302 |
| AM | test  | **0.154** / 0.438 | **0.186** / 0.430 | **0.112** / 0.267 |
| AN | test  | **0.095** / 0.253 | **0.192** / 0.490 | **0.090** / 0.246 |
| **AP** | test | 0.283 / **0.144** | 0.314 / **0.190** | 0.127 / **0.083** |
| AR | test  | **0.081** / 0.295 | **0.098** / 0.510 | **0.053** / 0.268 |
| AT | test  | **0.143** / 0.256 | 0.308 / **0.149** | 0.128 / **0.121** |
| **Mean** | | **0.130** / 0.316 | **0.204** / 0.380 | **0.108** / 0.228 |

### Reading these honestly

**Stage 2 leads on aggregate — and that is the expected outcome, not a Stage-3 failure.** Stage 2's Sliced-Wasserstein objective *explicitly matches the ground-truth φ/ψ point set*; it is being evaluated on precisely the quantity it was optimised for. Stage 3 is trained on **single-step transitions** — its equilibrium distribution is an emergent by-product of compounding 300 autoregressive steps.

**Neither metric measures dynamics.** A model drawing i.i.d. samples from the exact equilibrium distribution would score *perfectly* on this table while possessing no dynamics whatsoever. Stage 2 is closer to that limit by construction: its frame ordering is arbitrary, so no autocorrelation curve can even be defined for it.

**Stage 3 wins outright on AP** (all three metrics). AP is proline — the pyrrolidine ring sterically restricts φ to a narrow band, so the accessible space is small and well defined, and a propagator has fewer opportunities to drift.

**The open problem:** six Stage-3 KL values exceed 0.35 (FW, DR, AH, AM, AN, AR on ψ; AC, AM, DR, GG, HH on φ), all cases where the rolled-out ψ distribution is **broader** than the reference. The pattern is systematic over-dispersion, not peptide-specific noise.

### Dispersion analysis

Sweeping the sampling temperature (which scales per-step noise) on the reference checkpoint:

| Temperature | 1.00 | **0.85** | 0.70 | 0.55 | 0.40 |
|---|---|---|---|---|---|
| KL φ | 0.467 | **0.269** | 0.502 | 0.794 | 1.168 |
| ACF error | 0.249 | **0.154** | 0.228 | 0.267 | 0.294 |
| φ spread (deg) | 56.9 | 24.0 | 17.3 | 13.2 | 9.4 |

*(Ground-truth φ spread: 29.3°.)*

At the default the chain is **~2× as dispersed as reality**. Lowering temperature improves distribution *and* autocorrelation simultaneously — not a trade-off — then degrades again as the chain over-sharpens and freezes. Clean U-curve.

> **The optimum was not consistent across peptides** — AC favoured 0.85, AD favoured 0.90. The *direction* is robust (mean KL 0.415 → 0.26); the precise value is not, and should be selected on a validation set rather than on test peptides. `FLOW_TEMP` currently ships at **1.0**.

---

## 📏 Evaluation Metrics

| Metric | Measures | Why it's here |
|---|---|---|
| **Ramachandran overlay** | conformational basins visited | the qualitative check |
| **KL divergence** (φ, ψ) | marginal distribution match | same criterion Stage 2 was scored on |
| **2-D Jensen-Shannon** | joint φ/ψ distribution | symmetric and bounded in [0,1], unlike KL |
| **Dihedral ACF** | **decorrelation timescale** | the one metric that separates a propagator from a sampler — a perfect i.i.d. sampler scores flawlessly everywhere else while its ACF collapses in one frame |
| **Short-horizon MAD** | frame-by-frame tracking over ~20 frames | MD is chaotic; long-horizon divergence is **expected**, not a failure |
| **Bond-length RMSD** | molecular integrity | catches slow inflation/collapse a Ramachandran plot hides |

### ⚠️ A measurement artefact worth knowing about

KL is infinite wherever the prediction puts zero mass on an occupied bin. The original implementation patched this with `ε = 1e-8`, so an empty bin contributed `p·log(p/ε)` — an enormous penalty.

Quantified by scoring **genuine MD frames** against the full ground-truth density. At 100 samples over 36 bins, ~6 bins are empty *even for real data*:

| KL φ (100 samples) | `ε = 1e-8` | **Laplace** |
|---|---|---|
| **Real MD (noise floor)** | 0.623 ± 0.311 | **0.166 ± 0.029** |
| Stage 2 | 1.019 | 0.227 |
| Stage 3 | 1.354 | 0.508 |

All KL values in this repo now use **add-one (Laplace) smoothing** — the posterior mean under a uniform Dirichlet prior. `evaluate.null_baseline_kl()` reports the floor for any sample count.

> **Always read a KL against its noise floor.** An absolute KL is uninterpretable without one. The floor is 0.166 at 100 samples but **0.048 at 300**.

---

## 🚀 Getting Started

```bash
git clone <repo> && cd MolGraph-AE
python -m venv .venv && source .venv/bin/activate
pip install torch torch-geometric numpy scipy matplotlib py3Dmol huggingface_hub

mkdir -p timewarp_data
# download 2AA-complete into timewarp_data/2AA-complete/{train,val,test}/
```

Python 3.12, CPU or CUDA. All reported results were produced on **CPU**.
**All commands run from the repository root** — modules import `scripts.*` and `models.*`.

---

## 💻 Full Command Reference

### Stage 1
```bash
python -m models.pipeline_auto                    # → encoder.pt, decoder.pt
```

### Stage 2
```bash
python -m trajectory_pre_.train_trajectory --epochs 300 --batch_size 8
python trajectory_pre_/predict_trajectory.py --peptide AC --samples 5
```

### Stage 2b — seeded
```bash
python -u -m trajectory_pre_.train_seeded --frames 100 --epochs 60
python -u -m trajectory_pre_.predict_seeded --peptide AC --seed-frame 50 --target-frame 100
python -u -m trajectory_pre_.predict_seeded --peptide AC --seed-frame 50 --frames 100
```
Watch the **`anchor`** column while training — it's the term forcing the model to honour its seed. If it plateaus high, raise `--w-anchor`.

### Stage 3 — run in order the first time
```bash
python -u -m temporal_dynamics.check_velocities        # STEP 0  (~30 s, read-only)
python -u -m temporal_dynamics.prepare_data            # STEP 1  (~4 min)
python -u -m temporal_dynamics.adapter                 # STEP 2  (~10 s)
python -u -m temporal_dynamics.train \
       --epochs 100 --batch-size 16 --steps-per-epoch 250    # STEP 3
python -u -m temporal_dynamics.evaluate --peptide AC --frames 300   # STEP 4
```

Generate a trajectory on its own:
```bash
python -u -m temporal_dynamics.rollout --peptide AC --frames 300 --pdb
python -u -m temporal_dynamics.rollout --peptide AC --samples 3
```

### Benchmark
```bash
python -u -m benchmark.run_benchmark --peptide AC --frames 300 --start-frame 50
for p in AC AD AH AM AN AP AR AT; do
  python -u -m benchmark.run_benchmark --peptide $p --frames 300 --start-frame 50
done
```

### 💡 Practical notes

- **Always use `python -u`.** Python block-buffers stdout when it isn't a TTY, and an epoch takes minutes — without `-u` the terminal sits blank and a *working* run is indistinguishable from a hung one. Training also prints a live step/ETA heartbeat.
- **Startup takes minutes before the first epoch line.** `prepare_data` loads 140 peptides / ~420 k transitions from a ~716 MB cache. That silence is loading, not a hang.
- **`train.py` writes straight to `best_propagator.pt`.** Back up a checkpoint you care about before starting a new run.
- **Dependency chain:** Steps 1–3 must run in order the first time. Re-run Steps 1+2 together if you change the peptide set or frame count — `adapter` only embeds peptides present in the transition cache.

### Tuning knobs (no retraining needed)

| Flag | Effect |
|---|---|
| `--temperature` | `>1` broadens the distribution, `<1` sharpens. Currently 1.0; see dispersion analysis. |
| `--flow-steps` | Euler steps per transition (default 20). Drop to 10 for ~2× faster iteration. |
| `--bond-fix` | Constraint sweeps per step; `0` shows the unconstrained model. |
| `--frames` | 300 frames = 1.5 ns simulated at the 5 ps lag. |

---

## 📝 Report

`report/main.tex` — full LaTeX report for IISER Pune. Four TikZ architecture diagrams (drawn in LaTeX, no image files) plus six figures.

```bash
cd report && pdflatex main.tex && pdflatex main.tex     # twice, for the ToC
```

Requires `geometry, graphicx, amsmath, booktabs, tikz, xcolor, caption, subcaption, hyperref, float`. If a package is missing, either `sudo apt install texlive-latex-extra texlive-pictures` or upload to [Overleaf](https://overleaf.com), which has everything preinstalled.

**Figures referenced** (paths relative to repo root — `\graphicspath{{../}}` assumes `main.tex` sits in `report/`):

```
plots_1/training_losses.png
plots_1/val_losses.png
trajectory_pre_/loss_curve.png                                    ⚠ same filename
temporal_dynamics/loss_curve.png                                  ⚠ as the one above
benchmark/results/AC_f300_s50/ramachandran_comparison.png
benchmark/results/AC_f300_s50/kl_divergence_comparison.png
```

> ⚠️ Two different files are both named `loss_curve.png`. Keep them in separate folders — a `graphicspath` search list would silently resolve the Stage-3 figure to the Stage-2 curve.

---

## 🧩 Design Decisions

| Decision | Rationale |
|---|---|
| **Static edge topology** | Chemical bonds don't break during MD; per-frame edges cause flickering artefacts |
| **5-dim element one-hot** | N, O, S are chemically distinct — collapsing them loses critical information |
| **BCEWithLogits + clamp** | Raw clamped logits avoid `exp()` overflow on large dot products |
| **Frozen Stage-1 encoder** | Decouples representation from generation; stabilises both downstream stages |
| **Unit-circle dihedral head** | L2-normalised `(sin, cos)` keeps angles well defined, preventing collapse-to-origin scatter |
| **Sliced-Wasserstein loss** | Matches the *joint* φ/ψ distribution and is mode-covering |
| **Kabsch alignment per transition** | Tumbling is 5× the internal motion and carries no molecular information |
| **Velocity as state, not integrator** | Measured corr = −0.005 at the 5 ps lag |
| **Flow matching over MSE** | An MSE learns the conditional mean; 300 compounding steps of a conditional mean freezes the trajectory |
| **Fully-connected edges** | ≤ ~1100 edges is cheap and lets non-bonded steric effects propagate |
| **Laplace-smoothed KL** | `ε = 1e-8` put the noise floor at 0.62, burying all real signal |

---

## ⚠️ Known Limitations

- **Test-set breadth.** Only 8 test peptides are cached, all beginning with alanine, because `prepare_data.py` truncates each split alphabetically. The full Timewarp test split has 104. Widening it would materially strengthen the evaluation.
- **Systematic over-dispersion.** Stage 3's rolled-out ψ distribution is consistently broader than the reference on several peptides. This is the principal open problem.
- **Single-step training horizon.** Stage 3's objective never observes the consequences of compounding. Multi-timescale training or a stationarity constraint would target this directly.
- **Sample-size sensitivity.** Distributional metrics from a few hundred frames carry real sampling noise. Report the null baseline alongside every value.
- **Novelty.** Graph autoencoders, Sliced-Wasserstein matching, equivariant message passing and conditional flow matching are all **established techniques**. The contribution here is their combination and the empirical analysis of this specific system — not a new method. Lag-conditioned transition models for MD in particular have been explored in the Implicit Transfer Operator literature.

---

## 🔮 Future Work

- [ ] 🕐 **Multi-timescale training** — condition on the transition lag and train across several lags, pinning the diffusion rate at every timescale rather than only the shortest
- [ ] ⚖️ **Stationarity regularisation** — penalise drift of the equilibrium distribution under repeated application of the learned kernel
- [ ] 🌐 **SE(3)-equivariant Stage-1 encoder** (EGNN, SchNet)
- [ ] 🧬 **Side-chain dihedrals** — extend beyond backbone φ/ψ to χ angles
- [ ] 📦 **Longer peptides** — scale from dipeptides to the full Timewarp suite
- [ ] 🔁 **Full 3-D reconstruction** — lift generated dihedral ensembles back to all-atom coordinates

---

## 🙏 Acknowledgements & License

Data: [Microsoft Timewarp](https://github.com/microsoft/timewarp) `2AA-complete` dipeptide dataset.
Developed as a research internship at **IISER Pune** under **Dr. Arnab Mukherjee**. Please refer to institutional guidelines for usage terms (MIT-style for the code in this repository).

<p align="center">
  <sub>Built with PyTorch Geometric, Transformers &amp; py3Dmol</sub>
</p>
