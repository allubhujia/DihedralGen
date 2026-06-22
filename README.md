<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/PyG-3C2179?style=for-the-badge&logo=pyg&logoColor=white" />
  <img src="https://img.shields.io/badge/Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# 🧬 MolGraph-AE — Graph Autoencoding & Generative Dihedral Trajectory Modelling for Dipeptides

> A two-stage deep-learning pipeline that (1) learns **permutation-invariant latent embeddings** of dipeptide molecular graphs with a GNN autoencoder, and (2) uses those embeddings to **generate backbone conformational ensembles** — predicting the φ/ψ dihedral *distribution* of a peptide's molecular-dynamics trajectory with a Transformer trained under a Sliced-Wasserstein objective.
>
> Developed as part of a research internship at **IISER**, using the [Microsoft Timewarp](https://github.com/microsoft/timewarp) `2AA-complete` dataset.

---

## 📋 Table of Contents

- [Abstract](#-abstract)
- [Tech Stack](#-tech-stack)
- [System Architecture](#-system-architecture)
- [Repository Structure](#-repository-structure)
- [Stage 1 — Molecular Graph Autoencoder](#-stage-1--molecular-graph-autoencoder)
- [Stage 2 — Generative Dihedral Trajectory Model](#-stage-2--generative-dihedral-trajectory-model)
- [Evaluation & Trajectory Analysis](#-evaluation--trajectory-analysis)
- [Unseen-Peptide Generalization Pipeline](#-unseen-peptide-generalization-pipeline)
- [Data Pipeline & Feature Engineering](#-data-pipeline--feature-engineering)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Results](#-results)
- [Theoretical Notes](#-theoretical-notes)
- [Design Decisions](#-design-decisions)
- [Future Work](#-future-work)
- [Acknowledgements & License](#-acknowledgements--license)

---

## 🔬 Abstract

Classical molecular dynamics (MD) generates conformational ensembles by integrating Newton's equations over femtosecond time steps — accurate, but computationally expensive. **MolGraph-AE** investigates whether a learned latent representation of a peptide's *static chemical graph* carries enough information to **generate its conformational distribution** directly, bypassing step-by-step integration.

The project is organized as two coupled learning problems:

1. **Representation learning (Stage 1).** A Graph Convolutional Network (GCN) autoencoder compresses each dipeptide graph into a fixed-size latent vector `z`. Training is self-supervised: an inner-product decoder reconstructs the molecular adjacency matrix, while a cosine-similarity regularizer keeps distinct molecules well-separated in latent space.

2. **Conditional generation (Stage 2).** The *frozen* encoder turns a peptide into a conditioning embedding. A Transformer-over-frames generator maps that embedding to a **300-frame backbone trajectory** in (sin φ, cos φ, sin ψ, cos ψ) space. Because the conditioning signal is only the molecule's identity — not a starting frame — the model is trained to match the **joint φ/ψ distribution** of the ground-truth MD trajectory via a **Sliced-Wasserstein distance**, rather than regressing individual frames (which provably collapses to the mean).

Fidelity is assessed against ground-truth MD using **overlaid Ramachandran plots** and **per-angle KL divergence**, and generalization is probed on **unseen peptides** downloaded and reformatted on the fly from HuggingFace.

---

## 🛠 Tech Stack

| Layer | Tools | Role |
|-------|-------|------|
| **Deep learning** | PyTorch | Tensors, autograd, training loops, checkpoints |
| **Graph learning** | PyTorch Geometric (PyG) — `GCNConv`, `global_mean_pool`, `DataLoader`, `to_dense_adj` | Graph construction, message passing, batched graph tensors |
| **Generative model** | `nn.TransformerEncoder` (frame-attention), Sliced-Wasserstein loss | Conditional dihedral-trajectory generation |
| **Scientific computing** | NumPy, SciPy (`gaussian_kde`, `gaussian_filter`, `entropy`) | Dihedral math, KDE, free-energy contours, KL divergence |
| **Structural biology I/O** | Custom PDB / NPZ parsers, multi-MODEL PDB writer | Topology + trajectory parsing, backbone φ/ψ extraction |
| **Visualization** | Matplotlib, py3Dmol | Ramachandran/loss plots, interactive 3D molecular viewers |
| **Data source** | `huggingface_hub`, Microsoft Timewarp `2AA-complete` | MD trajectories (`.npz` positions + `.pdb` topology) |
| **Runtime** | Python 3.12, WSL + venv (CPU or CUDA) | Reproducible execution environment |

---

## 🏗 System Architecture

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        MolGraph-AE — TWO-STAGE PIPELINE                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ╔══════════════════════ STAGE 1: REPRESENTATION ══════════════════════╗     │
│  ║                                                                      ║     │
│  ║   PDB + NPZ ──► processing.py ──► PyG Data (x, edge_index, pos)      ║     │
│  ║                                        │                             ║     │
│  ║                                        ▼                             ║     │
│  ║              MolecularGNN (3× GCNConv + LayerNorm + SiLU)            ║     │
│  ║                     │                          │                     ║     │
│  ║          global_mean_pool + MLP        node-level projection         ║     │
│  ║                     ▼                          ▼                     ║     │
│  ║            graph embedding z          node embeddings                ║     │
│  ║                     │                          │                     ║     │
│  ║          cosine-sim loss (B,B)     InnerProductDecoder  z·zᵀ         ║     │
│  ║                     │                          │                     ║     │
│  ║                     └──────────► combined loss ◄── BCE(adjacency)    ║     │
│  ║                                       │                              ║     │
│  ║                                   encoder.pt / decoder.pt            ║     │
│  ╚═══════════════════════════════════════│══════════════════════════════╝     │
│                                          │ (frozen encoder)                   │
│  ╔══════════════════════ STAGE 2: GENERATION ═══════════════│═══════════╗     │
│  ║                                                          ▼            ║     │
│  ║   peptide ──► frozen encoder ──► mol embedding (64-d) ──► TrajectorySeqModel
│  ║                                                           │          ║     │
│  ║        frame positional emb. + projected mol emb. + noise │          ║     │
│  ║                              TransformerEncoder (4 layers)│          ║     │
│  ║                                          │                           ║     │
│  ║                          unit-circle head → [300, 4]                 ║     │
│  ║                          (sinφ, cosφ, sinψ, cosψ)                    ║     │
│  ║                                          │                           ║     │
│  ║         Sliced-Wasserstein loss vs. ground-truth 300-point set       ║     │
│  ║                                          │                           ║     │
│  ║                                  best_trajectory.pt                   ║     │
│  ╚═══════════════════════════════════════│══════════════════════════════╝     │
│                                          ▼                                    │
│        dihedral_comparison.py ──► Ramachandran overlay + φ/ψ KL divergence    │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
MolGraph-AE/
│
├── 📄 README.md                       # This file
├── 📄 .gitignore                      # Ignores venv, data, caches, generated outputs
├── 📄 encoder.pt / decoder.pt         # Stage-1 trained checkpoints (consumed by Stage 2)
├── 📄 test_output.txt                 # Sample Stage-1 pipeline log
│
├── 🧠 models/                         # Stage-1 neural network modules
│   ├── gnn.py                         # MolecularGNN — stacked GCNConv message passing
│   ├── encoder.py                     # MolecularEncoder — graph/node → latent embedding
│   ├── decoder.py                     # InnerProductDecoder — z·zᵀ adjacency reconstruction
│   ├── loss.py                        # BCE reconstruction + cosine-similarity combined loss
│   └── pipeline_auto.py              # End-to-end train + validation loop (saves checkpoints)
│
├── ⚙️ scripts/                        # Shared data / structural-biology utilities
│   ├── processing.py                 # PDB+NPZ → PyG graph conversion (load_split / load_molecule)
│   ├── backbone_utils.py             # φ/ψ dihedral computation from backbone atoms
│   └── visualization.py              # py3Dmol interactive 3D molecular viewer
│
├── 🔮 trajectory_pre_/                # Stage-2 generative dihedral trajectory model
│   ├── train_trajectory.py           # TrajectorySeqModel (Transformer) + Sliced-Wasserstein training
│   ├── predict_trajectory.py         # Generate φ/ψ ensembles + multi-MODEL PDB from a peptide
│   ├── best_trajectory.pt            # Best Stage-2 checkpoint (lowest val SW distance)
│   ├── loss_curve.png                # Train/val Sliced-Wasserstein curve
│   ├── epoch_checks/                 # Every-5-epoch Ramachandran + histogram diagnostics
│   └── predictions/{PEPTIDE}/        # Per-peptide GT + predicted dihedral .npz and GT .pdb
│
├── 📊 trajectory_analysis(main)/      # Evaluation: predicted vs. ground-truth distributions
│   ├── dihedral_comparison.py        # Ramachandran overlay + dihedral histogram/KDE + KL
│   ├── {PEPTIDE}_plot/               # Per-peptide distribution & Ramachandran figures
│   └── kl_divergence/                # Per-peptide φ/ψ KL-divergence reports (.txt)
│
├── 🧫 trajectory_pdb_files/{PEPTIDE}/ # Trajectories analyzed (GT + predicted dihedrals/PDB)
│
├── 🌐 unseen_peptide_test/            # Generalization to peptides not in the training set
│   ├── preprocess_huggingface.py     # Download + reformat a peptide from HuggingFace Timewarp
│   └── run_unseen_pipeline.sh        # Orchestrator: download → predict → analyze
│
├── 📈 plots_1/                        # Stage-1 training/validation loss visualizations
│   ├── plot_train_losses.py / plot_val_loss.py
│   └── training_losses.png / val_losses.png
│
├── 📖 Theoretical/                    # Research notes & architecture rationale
│   ├── models.txt                    # Detailed module-by-module explanation
│   ├── autoencoder_dimensions.md     # Tensor-shape walkthrough of Stage 1
│   ├── trajectory_mlp_dimensions.md  # Tensor-shape walkthrough of the trajectory model
│   └── implementation_plan_collapse  # Notes on diagnosing & fixing mode collapse
│
├── 🎨 visualizations/                 # Pre-generated interactive 3D HTML viewers
│   └── AA_single.html / AE_single.html / AF_single.html / AW_single.html
│
└── 💾 timewarp_data/                  # 2AA-complete dataset (gitignored — download separately)
    └── 2AA-complete/{train,val,test}/{PEPTIDE}-traj-arrays.npz + -traj-state0.pdb
```

> **Note:** `.venv/`, `timewarp_data/`, Python caches, and large regenerable outputs are excluded via `.gitignore`.

---

## 🧠 Stage 1 — Molecular Graph Autoencoder

Stage 1 learns a self-supervised latent representation of each dipeptide's chemical graph.

### `gnn.py` — `MolecularGNN`
The message-passing backbone: `num_layers` of `GCNConv`, each followed by `LayerNorm`, with `SiLU` activation on all but the final layer. Maps 8-dim atom features to 64-dim node embeddings using the **static bond topology**.

```python
# in_channels → [hidden × (num_layers-2)] → out_channels ; LayerNorm + SiLU per layer
model = MolecularGNN(in_channels=8, hidden_channels=64, out_channels=64, num_layers=3)
```

### `encoder.py` — `MolecularEncoder`
Wraps the GNN and exposes three views of the data:

| Method | Output | Used for |
|--------|--------|----------|
| `forward(x, edge_index, batch)` | graph embedding `z` `[B, embed_dim]` | similarity loss, Stage-2 conditioning |
| `encode_nodes(x, edge_index)` | raw node embeddings `[N, hidden]` | Stage-2 (frozen, pre-projection) |
| `encode_nodes_projected(x, edge_index)` | projected node embeddings `[N, embed_dim]` | adjacency reconstruction |

```python
encoder = MolecularEncoder(in_channels=8, hidden_channels=64, embedding_dim=32)
z = encoder(x, edge_index, batch)  # → [B, 32]
```

### `decoder.py` — `InnerProductDecoder`
Reconstructs the adjacency matrix as **raw logits** `Â = z·zᵀ`, clamped to `[-10, 10]` to prevent logit explosion. No sigmoid — `BCEWithLogits` applies it internally for numerical stability.

### `loss.py` — Combined Objective
A dual-objective loss balancing structural fidelity with embedding separation:

- **`reconstruction_loss`** — `pos_weight`-balanced `BCEWithLogits` on the adjacency matrix (corrects the heavy 0/1 class imbalance of sparse molecular graphs).
- **`cosine_similarity_loss`** — MSE between the batch cosine-similarity matrix and the identity (each graph similar to itself, dissimilar to others; requires batch ≥ 2).
- **`combined_loss`** — `L = mean(BCE_per_graph) + λ · cosine_sim`, with `λ = 0.01`.

### `pipeline_auto.py` — Training & Validation
End-to-end loop: loads the `train` split, optimizes encoder + decoder with `AdamW` (`lr=1e-3`, `weight_decay=1e-5`) over 300 epochs, **saves `encoder.pt` / `decoder.pt`**, then runs a no-grad evaluation on the `val` split and reports mean total/reconstruction/similarity losses.

---

## 🔮 Stage 2 — Generative Dihedral Trajectory Model

Stage 2 (`trajectory_pre_/`) generates a peptide's backbone conformational ensemble from its frozen Stage-1 embedding.

### Why distribution matching, not per-frame regression
The only conditioning signal is the **molecule's identity**. Each peptide's 300-frame MD trajectory is *stochastic* — the exact frame ordering is not a function of the molecule. A per-frame MSE regressor therefore:

1. Collapses toward the mean (sin, cos), which lies *inside* the unit circle, so `atan2()` returns near-random angles → a uniform scatter on the Ramachandran plane (a real failure mode observed during development); and
2. Cannot reproduce the **basin-hopping** that defines the conformational distribution.

### `train_trajectory.py` — `TrajectorySeqModel`
A **Transformer-over-frames** generator:

- Each of the 300 output frames is a token = learned **frame positional embedding** + projected molecule embedding + small per-frame **latent noise** (for sample diversity).
- A 4-layer `TransformerEncoder` (`d_model=128`, `nhead=4`, `dim_feedforward=256`) lets frames attend to one another.
- A unit-circle head L2-normalizes each `(sin, cos)` pair so every predicted frame lies **exactly on the unit circle** → well-defined, scatter-free angles.

**Loss — Sliced-Wasserstein-1.** Both the predicted and ground-truth 300-point sets are projected onto `num_proj=64` random directions in the 4-D `(sinφ, cosφ, sinψ, cosψ)` space; the loss is the mean sorted-projection L1 distance. Matching all 1-D projections matches the full **joint φ/ψ distribution** and is mode-covering (no mode collapse).

Training uses the **frozen Stage-1 encoder** for conditioning, `AdamW` + `ReduceLROnPlateau`, early stopping on validation SW distance, and saves `best_trajectory.pt`, `loss_curve.png`, and every-5-epoch diagnostics to `epoch_checks/`.

### `predict_trajectory.py`
Loads `best_trajectory.pt` and, for a given peptide, emits:

| File | Contents |
|------|----------|
| `{PEPTIDE}_ground_truth_dihedrals.npz` | true φ/ψ `[300]` (degrees) |
| `{PEPTIDE}_predicted_dihedrals.npz` | predicted φ/ψ `[S, 300]` (degrees) |
| `{PEPTIDE}_ground_truth.pdb` | true 3D multi-MODEL trajectory |

`--samples 1` is deterministic; `--samples > 1` keeps test-time dropout on to emit a diverse ensemble.

---

## 📊 Evaluation & Trajectory Analysis

`trajectory_analysis(main)/dihedral_comparison.py` quantifies how well generated ensembles reproduce ground-truth MD distributions. For each peptide it produces:

1. **Overlaid Ramachandran plot** — predicted vs. ground-truth φ/ψ scatter with free-energy contours.
2. **Comparative histograms + KDE** of the φ and ψ marginal distributions.
3. **Per-angle KL divergence** reports written to `kl_divergence/{PEPTIDE}_kl_divergence.txt` (lower = closer match).

Dihedral math lives in `scripts/backbone_utils.py` (`compute_dihedral`, `parse_backbone_from_pdb`, `compute_phi_psi_from_positions`), which handles both standard 2-residue dipeptides and **capped peptides** (e.g. AD-3 / Ace-Ala-Nme, where ACE/NME caps have incomplete backbones).

---

## 🌐 Unseen-Peptide Generalization Pipeline

`unseen_peptide_test/` probes whether the trained models generalize to peptides outside the training split.

- **`preprocess_huggingface.py`** downloads a target peptide from the `microsoft/timewarp` HuggingFace dataset and reformats it to the `2AA-complete` convention: converts `HETATM` → `ATOM` (so caps are parsed), re-keys NPZ arrays to `"positions"`, and assigns residue/backbone labels.
- **`run_unseen_pipeline.sh`** orchestrates the full flow: **download → predict → analyze**, producing Ramachandran and histogram plots for the new peptide.

```bash
./unseen_peptide_test/run_unseen_pipeline.sh AFA
```

> ℹ️ The orchestrator was originally wired for a diffusion variant; point its prediction step at `trajectory_pre_/predict_trajectory.py` to run it against the current generative model.

---

## 🔄 Data Pipeline & Feature Engineering

```
timewarp_data/2AA-complete/{train,val,test}/
├── {PEPTIDE}-traj-arrays.npz   ← positions [T, N, 3] (nm) (+ velocities)
└── {PEPTIDE}-traj-state0.pdb   ← topology / element symbols
```

`scripts/processing.py` converts each pair into a PyG `Data` object:

| Step | Input | Output |
|------|-------|--------|
| PDB parsing | `.pdb` topology | element list `['H','C','N',...]` |
| Edge construction | first-frame positions | static `edge_index` (distance cutoff ≈ 0.2 nm) |
| Feature building | positions + elements | 8-dim node features |
| Graph assembly | all of the above | `torch_geometric.data.Data` (`.x`, `.edge_index`, `.pos`, `.peptide_id`) |

**Node feature vector (8-dim per atom):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0–2 | `x, y, z` | atom coordinates (nm) |
| 3 | `is_H` | hydrogen one-hot |
| 4 | `is_C` | carbon one-hot |
| 5 | `is_N` | nitrogen one-hot |
| 6 | `is_O` | oxygen one-hot |
| 7 | `is_S` | sulfur one-hot |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.12 (developed on WSL; CPU or CUDA)
- The Microsoft Timewarp `2AA-complete` dataset

### Installation

```bash
# Clone
git clone https://github.com/<your-username>/MolGraph-AE.git
cd MolGraph-AE

# Create & activate a virtual environment (WSL / Linux / macOS)
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch torch-geometric
pip install numpy scipy matplotlib py3Dmol huggingface_hub
```

### Dataset Setup

```bash
mkdir -p timewarp_data
# Download the 2AA-complete dataset from https://github.com/microsoft/timewarp
# into timewarp_data/2AA-complete/{train,val,test}/
```

> 💡 On Windows, run Python via **WSL** with the venv activated (`source .venv/bin/activate`).

---

## 💻 Usage

### 1 — Train the Stage-1 autoencoder
```bash
python -m models.pipeline_auto          # → encoder.pt, decoder.pt + val report
```

### 2 — Train the Stage-2 trajectory generator
```bash
python -m trajectory_pre_.train_trajectory --epochs 300 --batch_size 8
# → best_trajectory.pt, loss_curve.png, epoch_checks/
```

### 3 — Generate a trajectory for a peptide
```bash
python trajectory_pre_/predict_trajectory.py --peptide AC            # deterministic
python trajectory_pre_/predict_trajectory.py --peptide AC --samples 5 # ensemble
```

### 4 — Analyze predicted vs. ground truth
```bash
python "trajectory_analysis(main)/dihedral_comparison.py"               # all peptides
python "trajectory_analysis(main)/dihedral_comparison.py" --peptide AC  # one peptide
```

### 5 — Run the full unseen-peptide pipeline
```bash
./unseen_peptide_test/run_unseen_pipeline.sh AFA
```

### Visualize a structure in 3D
```bash
python scripts/visualization.py --peptide AF --split train
```

### Load data programmatically
```python
from scripts.processing import load_split

graphs = load_split("timewarp_data/2AA-complete", split="train", max_molecules=10)
for g in graphs:
    print(f"{g.peptide_id}: {g.x.shape[0]} atoms, {g.edge_index.shape[1] // 2} bonds")
```

---

## 📊 Results

**Stage 1 — autoencoder tensor flow** (batch of 4 dipeptides):

```
┌──────────────────────────────────┬──────────────────────────┐
│  Stage                           │  Tensor Shape            │
├──────────────────────────────────┼──────────────────────────┤
│  Input node features             │  [92, 8]                 │
│  Edge index (bonds)              │  [2, 256]                │
│  GNN node embeddings             │  [92, 64]                │
│  Graph-level embeddings          │  [4, 32]                 │
│  Reconstructed adjacency (each)  │  [n_i, n_i]              │
└──────────────────────────────────┴──────────────────────────┘
```

**Stage 2 — generative evaluation.** Per-peptide Ramachandran overlays and φ/ψ KL-divergence
reports are written to `trajectory_analysis(main)/{PEPTIDE}_plot/` and `kl_divergence/`. The
Sliced-Wasserstein loss curve (`trajectory_pre_/loss_curve.png`) and every-5-epoch diagnostics
(`trajectory_pre_/epoch_checks/`) track distribution matching during training.

---

## 📖 Theoretical Notes

The `Theoretical/` directory documents the reasoning behind the design:

- **`models.txt`** — module-by-module explanation of the full pipeline.
- **`autoencoder_dimensions.md`** — tensor-shape walkthrough of Stage 1.
- **`trajectory_mlp_dimensions.md`** — tensor-shape walkthrough of the trajectory model.
- **`implementation_plan_collapse`** — root-cause analysis of the atom/angle **collapse** failure and the physics-/distribution-aware fixes that resolved it.

---

## 🧩 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Static edge topology** | Chemical bonds don't break during MD — per-frame edges cause "flickering" artifacts |
| **No velocity features** | Without transition learning (t → t+τ), velocities are noise that hurts convergence |
| **5-dim element one-hot** | N, O, S are chemically distinct — collapsing them loses critical information |
| **BCEWithLogits + clamp** | Raw clamped logits avoid `exp()` overflow inside sigmoid on large dot products |
| **Frozen Stage-1 encoder** | Decouples representation learning from generation; stabilizes Stage-2 training |
| **Unit-circle dihedral head** | L2-normalized (sin, cos) keeps angles well-defined and prevents collapse-to-origin scatter |
| **Sliced-Wasserstein loss** | Matches the *joint* φ/ψ distribution and is mode-covering (avoids MSE mode collapse) |

---

## 🔮 Future Work

- [ ] 🌐 **SE(3)-equivariant GNN** — upgrade to rotation/translation-invariant encoders (EGNN, SchNet)
- [ ] 🧬 **Side-chain dihedrals** — extend generation beyond backbone φ/ψ to χ angles
- [ ] 🔁 **Full 3D reconstruction** — lift generated dihedral ensembles back to all-atom coordinates
- [ ] 📦 **Larger peptides** — scale from dipeptides to longer chains and the full Timewarp suite
- [ ] 💾 **Embedding vector DB** — store latent embeddings for similarity retrieval at scale
- [ ] 🧪 **Diffusion baseline** — benchmark the Transformer generator against a diffusion variant

---

## 🙏 Acknowledgements & License

- Data: [Microsoft Timewarp](https://github.com/microsoft/timewarp) `2AA-complete` dipeptide dataset.
- Developed as part of a research internship at **IISER**. Please refer to institutional guidelines for usage terms (MIT-style for the code in this repository).

<p align="center">
  <sub>Built with ❤️ using PyTorch Geometric, Transformers & py3Dmol</sub>
</p>
