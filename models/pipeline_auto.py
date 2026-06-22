import torch
from torch_geometric.loader import DataLoader
from scripts.processing import load_split

from models.encoder import MolecularEncoder
from models.decoder import InnerProductDecoder
from models.loss import combined_loss

from torch_geometric.utils import to_dense_adj

"""
Encoder → embeddings
Decoder → adjacency prediction
Loss → combines reconstruction + similarity
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = MolecularEncoder(8, 64, 32).to(device)
decoder = InnerProductDecoder().to(device) #Initializes the decoder which will use dot products to rebuild the adjacency matrix.

optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3, weight_decay=1e-5)

data_dir = "timewarp_data/2AA-complete"
graphs = load_split(data_dir, split="train", max_molecules=208, verbose=False)

loader = DataLoader(graphs, batch_size=4, shuffle=False)

# ── Per-batch loss lists (exported for plotting) ──
total_losses = []
rec_losses   = []
sim_losses   = []

mol_idx = 0  # global molecule counter across all batches

num_epochs = 300

for epoch in range(1, num_epochs + 1):
    print(f"\n{'='*20} EPOCH {epoch}/{num_epochs} {'='*20}")
    encoder.train()
    decoder.train()

    for batch_idx, batch in enumerate(loader):
        batch = batch.to(device)

        # Graph-level embeddings — shape (B, d) — used for similarity loss
        z = encoder(batch.x, batch.edge_index, batch=batch.batch)

        # Node-level projected embeddings — shape (N_total, 32) — used for adjacency reconstruction
        # Uses the MLP projection head output (not raw GNN output)
        node_z = encoder.encode_nodes_projected(batch.x, batch.edge_index)

        # Dense adjacency per graph — shape (B, max_N, max_N)
        adj_dense = to_dense_adj(batch.edge_index, batch=batch.batch)

        # Build per-graph lists for combined_loss
        adj_list        = []
        adj_logits_list = []
        node_offset     = 0

        for i in range(batch.num_graphs): #Iterates through each of the 4 molecules.
            mol_idx += 1
            num_nodes_i = (batch.batch == i).sum().item() #Counts the number of nodes in the current molecule.

            node_z_i = node_z[node_offset: node_offset + num_nodes_i]   # (n_i, d) # Extracts the node embeddings for the current molecule.
            node_offset += num_nodes_i #Updates the offset for the next molecule.

            adj_logits_i = decoder(node_z_i)                             # (n_i, n_i) clamped logits # Predicts the adjacency matrix for the current molecule.
            adj_i = adj_dense[i, :num_nodes_i, :num_nodes_i]      # (n_i, n_i) ground truth # Extracts the adjacency matrix for the current molecule.

            # ── Per-molecule reconstruction loss ──
            from models.loss import reconstruction_loss as _rec_loss
            mol_rec = _rec_loss(adj_logits_i, adj_i)
            # print(f"    Molecule {mol_idx:3d}  |  Rec Loss: {mol_rec.item():.4f}  (nodes: {num_nodes_i})")

            adj_list.append(adj_i) # Appends the adjacency matrix for the current molecule.
            adj_logits_list.append(adj_logits_i) # Appends the predicted adjacency matrix for the current molecule.

        # combined_loss handles rec mean + sim loss internally
        # z must be full batch (B, d) — never z[i] — otherwise sim_loss = 0
        loss, rec, sim = combined_loss(adj_list, adj_logits_list, z)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_losses.append(loss.item())
        rec_losses.append(rec.item())
        sim_losses.append(sim.item())

        print(f"  Batch {batch_idx+1:3d}/{len(loader)}  |  Total: {loss.item():.4f}  Rec (mean): {rec.item():.4f}  Sim: {sim.item():.6f}")

encoder.eval()
decoder.eval()

# Save the trained model checkpoints
torch.save(encoder.state_dict(), "encoder.pt")
torch.save(decoder.state_dict(), "decoder.pt")
print("\nModel checkpoints saved to 'encoder.pt' and 'decoder.pt'.\n")

# ──────────────────────────────────────────────────────────────────────────
# Validation evaluation — same model, no retraining
# ──────────────────────────────────────────────────────────────────────────

print("="*65)
print("   VALIDATION SET EVALUATION")
print("="*65)

val_graphs = load_split(data_dir, split="val", max_molecules=84, verbose=False)
val_loader = DataLoader(val_graphs, batch_size=4, shuffle=False)

val_total_losses = []
val_rec_losses = []
val_sim_losses = []

val_mol_idx = 0

with torch.no_grad():
    for batch_idx, batch in enumerate(val_loader):
        batch = batch.to(device)

        z = encoder(batch.x, batch.edge_index, batch=batch.batch)
        node_z = encoder.encode_nodes_projected(batch.x, batch.edge_index)
        adj_dense = to_dense_adj(batch.edge_index, batch=batch.batch)

        adj_list = []
        adj_logits_list = []
        node_offset = 0

        for i in range(batch.num_graphs):
            val_mol_idx+=1
            num_nodes_i = (batch.batch==i).sum().item()
            node_z_i = node_z[node_offset: node_offset+num_nodes_i]
            node_offset+=num_nodes_i

            adj_logits_i = decoder(node_z_i)
            adj_i = adj_dense[i, :num_nodes_i, :num_nodes_i]
            mol_rec = _rec_loss(adj_logits_i, adj_i)

            print(f" [val] molecule {val_mol_idx:3d} | Rec loss: {mol_rec.item():.4f} (nodes: {num_nodes_i})")

            adj_list.append(adj_i)
            adj_logits_list.append(adj_logits_i)

        loss, rec, sim = combined_loss(adj_list, adj_logits_list, z)

        val_total_losses.append(loss.item())
        val_rec_losses.append(rec.item())
        val_sim_losses.append(sim.item())

        print(f" [val] batch {batch_idx+1:3d}/{len(val_loader)} | Total: {loss.item():.4f}  Rec: {rec.item():.4f}  Sim: {sim.item():.6f}")
        print()

# ── Final validation average ───────────────────────────────────────────────

total_mean = sum(val_total_losses) / len(val_total_losses)
rec_mean   = sum(val_rec_losses)   / len(val_rec_losses)
sim_mean   = sum(val_sim_losses)   / len(val_sim_losses)

print("\n" + "="*40)
print("   VAL SET AVERAGE")
print("="*40)
print(f"Total Loss (mean): {total_mean:.4f}")
print(f"Rec Loss   (mean): {rec_mean:.4f}")
print(f"Sim Loss   (mean): {sim_mean:.6f}")
print("="*40)
            
