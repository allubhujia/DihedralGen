# Trajectory MLP — Full Dimension & Data Flow Walkthrough
## Files: `train_mlp.py` and `predict_trajectory.py`

This document traces every tensor that flows through each function,
showing exactly what shape goes in and what shape comes out.

A typical 2-amino-acid peptide has ~20 atoms.
We use N to denote the number of atoms in a molecule (varies per molecule).

---

## PART 1: train_mlp.py

---

### Function 1: `load_split_with_trajectory(data_dir, split, max_molecules, num_frames)`

**Purpose:** Load raw .npz trajectory files and .pdb topology files from disk
and convert each molecule into a PyTorch Geometric graph object with a
trajectory label attached.

**Inputs:**
| Parameter       | Type   | Value / Description                          |
|-----------------|--------|----------------------------------------------|
| data_dir        | str    | Path to timewarp_data/2AA-complete/          |
| split           | str    | "train", "val", or "test"                    |
| max_molecules   | int    | 208 (train), 84 (val)                        |
| num_frames      | int    | 300 — number of time steps to keep           |

**Internal tensor transformations:**

```
arrays["positions"]  -->  shape: [T, N, 3]
    T = raw number of frames from NPZ (varies per molecule, e.g. 500+)
    N = number of atoms in the molecule
    3 = x, y, z coordinates

After slicing/padding to exactly num_frames:
    positions  -->  shape: [300, N, 3]

After np.transpose((1, 0, 2))  -- swap time and atom axes:
    positions  -->  shape: [N, 300, 3]
    This transpose is critical so that PyTorch DataLoader
    can batch along the atom (node) dimension correctly.

graph.y_traj = torch.tensor(positions)  -->  shape: [N, 300, 3]
    This gets attached to each graph as the training label.
```

**Output:**
```
all_graphs  -->  Python list of PyTorch Geometric Data objects
    Length: up to 208 (train) or 84 (val)
    Each graph has:
        graph.x          shape: [N, 8]     -- 8 node features per atom
        graph.edge_index shape: [2, E]     -- bond connectivity
        graph.y_traj     shape: [N, 300, 3]-- ground truth trajectory
```

---

### Class: `TrajectoryMLP`

**Purpose:** A 3-layer fully connected neural network. Takes per-atom embeddings
from the frozen encoder and predicts the full 3D trajectory for each atom.

**Architecture:**

```
Layer                     Input Dim    Output Dim    Notes
─────────────────────────────────────────────────────────────────
nn.Linear(64, 128)            64          128
nn.ReLU()                    128          128        non-linearity
nn.Dropout(0.3)              128          128        drops 30% of neurons (train only)
nn.Linear(128, 256)          128          256
nn.ReLU()                    256          256        non-linearity
nn.Dropout(0.3)              256          256        drops 30% of neurons (train only)
nn.Linear(256, 900)          256          900        900 = 300 frames × 3 coordinates
```

**Parameters (trainable weights):**
```
Layer 1:  64  × 128 + 128 bias  =   8,320
Layer 2: 128  × 256 + 256 bias  =  33,024
Layer 3: 256  × 900 + 900 bias  = 231,300
─────────────────────────────────────────
Total trainable parameters      = 272,644
```

---

### Method: `TrajectoryMLP.forward(node_z)`

**Input:**
```
node_z  -->  shape: [N_total, 64]
    N_total = total atoms across all molecules in the batch
              e.g. batch_size=4, each molecule ~20 atoms → N_total ≈ 80
    64      = node embedding dimension (output of MolecularEncoder)
```

**Internal flow:**
```
node_z                      [N_total, 64]
    → Linear(64→128) + ReLU + Dropout
out                         [N_total, 128]
    → Linear(128→256) + ReLU + Dropout
out                         [N_total, 256]
    → Linear(256→900)
out                         [N_total, 900]
    → out.view(N_total, 300, 3)         -- reshape flat 900 into frames × xyz
out                         [N_total, 300, 3]
```

**Output:**
```
out  -->  shape: [N_total, 300, 3]
    N_total = total atoms in the batch
    300     = number of predicted time frames
    3       = x, y, z coordinates at each frame
```

---

### Function: `main()` — Training Loop

**Step-by-step dimension trace per training iteration:**

```
Step 1: Load a batch from DataLoader
    batch_size = 4 molecules, each ~23 atoms
    batch.x          -->  shape: [92, 8]       (4 mols × ~23 atoms, 8 features)
    batch.edge_index  -->  shape: [2, 184]    (2 × total bonds in batch)
    batch.y_traj      -->  shape: [92, 300, 3]  (ground truth trajectory)

Step 2: encoder.encode_nodes(batch.x, batch.edge_index)
    Input:   batch.x          [92, 8]
    Output:  node_z           [92, 64]
    The frozen GNN encoder transforms 8 raw features → 64-dim embedding per atom.

Step 3: mlp(node_z)  -- TrajectoryMLP forward pass
    Input:   node_z           [92, 64]
    Output:  pred_traj        [92, 300, 3]
    Each atom's 64-dim embedding is mapped to 300 frames of 3D coordinates.

Step 4: criterion(pred_traj, batch.y_traj)  -- MSELoss
    pred_traj   shape: [92, 300, 3]
    batch.y_traj shape: [92, 300, 3]
    Loss: scalar float — average squared distance between
          predicted and real coordinates across all atoms, frames, and axes.

Step 5: batch_loss.backward() + optimizer.step()
    Only TrajectoryMLP weights are updated.
    MolecularEncoder weights are frozen (requires_grad=False).
```

**Early Stopping:**
```
patience = 20 epochs
    If val loss does not improve for 20 consecutive epochs → training stops.
    Best model weights (lowest val loss seen so far) are saved to trajectory_mlp.pt
    at every improvement and reloaded at the end.
```

---

## PART 2: predict_trajectory.py

---

### Function: `predict_molecule(peptide_id)`

**Purpose:** Inference-only pipeline. Loads a single unseen test molecule,
runs it through the frozen encoder + trained MLP, and prints the predicted trajectory.

**Step-by-step dimension trace:**

```
Step 1: Load the test molecule via load_molecule(npz_path, pdb_path)
    graph.x           -->  shape: [N, 8]
        N = number of atoms in the test peptide (e.g. 20 for a 2-AA peptide)
        8 = raw node features (atom type, mass, charge, etc.)
    graph.edge_index  -->  shape: [2, E]
        E = number of bonds/edges in the molecule

Step 2: encoder.encode_nodes(graph.x, graph.edge_index)
    Input:
        graph.x          [N, 8]
        graph.edge_index [2, E]
    Output:
        node_z           [N, 64]
        Each atom is now represented by a 64-dimensional learned embedding.

Step 3: mlp(node_z)  -- TrajectoryMLP forward pass
    Input:   node_z           [N, 64]
    Internal:
        Linear(64→128) + ReLU + Dropout  →  [N, 128]
        Linear(128→256) + ReLU + Dropout →  [N, 256]
        Linear(256→900)                  →  [N, 900]
        view(N, 300, 3)                  →  [N, 300, 3]
    Output:
        predicted_traj    [N, 300, 3]

Step 4: Output interpretation
    predicted_traj[atom_index, frame_index, :]  -->  [x, y, z]  (3 floats)
    e.g. predicted_traj[0, 0, :]  = x,y,z of atom 0 at time frame 0
         predicted_traj[0, 1, :]  = x,y,z of atom 0 at time frame 1
         predicted_traj[5, 150,:] = x,y,z of atom 5 at the midpoint of the trajectory
```

---

## Summary Table: All Tensor Shapes at a Glance

| Stage                          | Variable         | Shape            | Description                            |
|-------------------------------|------------------|------------------|----------------------------------------|
| Raw NPZ positions              | positions        | [T, N, 3]        | T=raw frames, N=atoms, 3=xyz           |
| After pad/slice                | positions        | [300, N, 3]      | fixed to 300 frames                    |
| After transpose                | positions        | [N, 300, 3]      | atom-first ordering                    |
| Graph node features            | graph.x          | [N, 8]           | 8 raw features per atom                |
| Graph label                   | graph.y_traj     | [N, 300, 3]      | ground truth trajectory                |
| Batched node features          | batch.x          | [N_total, 8]     | N_total = sum of atoms in batch        |
| Batched ground truth           | batch.y_traj     | [N_total, 300, 3]| ground truth for the whole batch       |
| After encoder                  | node_z           | [N_total, 64]    | 64-dim learned embedding per atom      |
| After MLP Layer 1              | —                | [N_total, 128]   | first hidden layer                     |
| After MLP Layer 2              | —                | [N_total, 256]   | second hidden layer                    |
| After MLP Layer 3 (flat)       | —                | [N_total, 900]   | 900 = 300 frames × 3 coords            |
| After reshape                  | pred_traj        | [N_total, 300, 3]| final predicted trajectory             |
| MSE Loss                       | batch_loss       | scalar           | average squared coordinate error       |
| Saved model                   | trajectory_mlp.pt| —                | best val-loss checkpoint               |
