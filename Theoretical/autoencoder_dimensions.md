# Autoencoder Architecture — Dimension & Data Flow Walkthrough
## Based on `pipeline_auto.py` and `encoder.py` (Phase 1 Training)

This document traces the exact tensor shapes flowing through the autoencoder shown in the architecture diagram. It clarifies the two distinct branches: the MLP Projection Head (for Similarity Loss) and the Molecular Decoder (for Reconstruction Loss).

Let `N_total` be the total number of atoms across all molecules in a single batch.
Let `B` be the batch size (number of molecules, e.g., 4).
Let `E_total` be the total number of edges/bonds across all molecules in the batch.

---

## 1. Input Data (Graph Batch)

A single batch contains multiple molecules bundled into one large disjoint graph.

**Inputs:**
- `batch.x`: `[N_total, 8]`
  * The raw node features (8 chemical features per atom).
  * Example from diagram: `[92, 8]` (Here 92 atoms total, 8 features).
- `batch.edge_index`: `[2, E_total]`
  * The connectivity map specifying which atoms share a bond.
- `batch.batch`: `[N_total]`
  * The batch vector. A 1D array telling PyTorch which atom belongs to which molecule (e.g., `[0,0,0... 1,1,1...]`).

---

## 2. Molecular GNN (Shared Backbone)

**Inputs:** `batch.x`, `batch.edge_index`
**Process:** Message passing through GCN/GAT layers to learn local topology.
**Output:** `node_z` (or Node Embeddings)
- Shape: `[N_total, 64]`
- *What it is:* A dense, learned 64-dimensional summary for every single atom.
- *Where it goes:* It splits and goes down TWO completely different paths simultaneously.

---

## 3. PATH A: MLP Projection Head & Similarity Loss
*(Operating at the **Graph** Level)*

**Purpose:** To force the encoder to learn distinct representations for different molecules in the batch.

**Step-by-step flow:**
1. **Global Mean Pooling:** 
   * Averages all the `[64]` atom vectors in a molecule to get a single `[64]` vector per molecule.
   * Input: `[N_total, 64]`
   * Output: `[B, 64]` (e.g., `[4, 64]`)
2. **Linear Layers (Projection):**
   * Passes the `[B, 64]` representation through the projection head to compress it further.
   * Output: `z` `[B, 32]` (e.g., `[4, 32]`)
3. **Similarity Loss Computation:**
   * Calculates how similar each molecule is to the others in the batch (`z @ z.T`).
   * Output is a scalar loss value penalizing molecules that look too similar when they shouldn't.
   * *Note: The output `z` stops here. It does NOT go to the decoder.*

---

## 4. PATH B: Molecular Decoder & Reconstruction Loss
*(Operating at the **Node** Level)*

**Purpose:** To force the learned node embeddings (`node_z`) to retain the exact connectivity (chemical bonds) of the molecule.

**Step-by-step flow:**
1. **Extract Single Molecule (Node Offset):**
   * The pipeline iterates through the batch, isolating one molecule at a time.
   * For molecule `i` with `N_i` atoms, we grab just its rows from `node_z`.
   * `node_z_i` shape: `[N_i, 64]` (e.g., `[23, 64]`)
2. **Decoder (Inner Product):**
   * We predict if a bond exists by taking the dot product of every atom's vector with every other atom's vector in the molecule (`node_z_i @ node_z_i.T`).
   * `adj_logits_i` shape: `[N_i, N_i]` (e.g., `[23, 23]`)
   * *What it is:* A square matrix where `(row, col)` represents the likelihood of a bond between atom `row` and atom `col`.
3. **Reconstruction Loss Computation:**
   * We compare the predicted `[N_i, N_i]` adjacency logits to the true `[N_i, N_i]` adjacency matrix.
   * Output is a scalar loss value penalizing incorrect or missing bonds.

---

## Summary Table of Flow

| Component                 | Variable       | Shape           | Description                                  |
|---------------------------|----------------|-----------------|----------------------------------------------|
| **Input Features**        | `batch.x`      | `[N_total, 8]`  | Raw atom properties                          |
| **GNN Output**            | `node_z`       | `[N_total, 64]` | Learned atom embeddings                      |
| **Global Pool**           | `graph_embed`  | `[B, 64]`       | One vector per molecule                      |
| **MLP Head Output**       | `z`            | `[B, 32]`       | Projected molecule vectors                   |
| **Decoder Input**         | `node_z_i`     | `[N_i, 64]`     | Atom embeddings for *one* molecule           |
| **Decoder Output**        | `adj_logits_i` | `[N_i, N_i]`    | Predicted bond likelihood between all atoms  |

*Where:*
- `N_total` ≈ 92 (Total atoms in a batch)
- `N_i` ≈ 23 (Atoms in a specific molecule)
- `B` = 4 (Molecules in a batch)
