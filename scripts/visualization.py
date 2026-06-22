"""
visualization.py  –  Interactive 3D molecular visualization using py3Dmol.

Converts PyG graph objects (produced by processing.py's load_split / 
snapshot_to_graph) back into renderable PDB strings and displays them
as interactive ball-and-stick models.

This fulfils the visualization TODO noted in data_utils.py:
    "and visualization of the graph as well"

USAGE:
    # From command line — exports HTML files you can open in a browser
    python scripts/visualization.py

    # From Jupyter — renders inline interactive 3D viewers
    from scripts.visualization import visualize_molecule
    from scripts.processing import load_split
    graphs = load_split("timewarp_data/2AA-complete", "train", max_molecules=1)
    visualize_molecule(graphs[0])

DEPENDENCIES:
    pip install py3Dmol

RUN:
    # To visualize the default "AA" peptide
python scripts/visualization.py

# To search for and visualize the "AF" peptide specifically
python scripts/visualization.py --peptide AF

# To search for and visualize the "WW" peptide specifically
python scripts/visualization.py --peptide WW

"""

import os
import sys
import torch
import numpy as np
from collections import defaultdict
from typing import List, Optional

from torch_geometric.data import Data

# ── Import constants from processing.py ──────────────────────
# Add parent directory so we can import from scripts/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from processing import ELEMENT_TO_IDX, load_split


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

# Reverse lookup: one-hot column index → element symbol
IDX_TO_ELEMENT = {v: k for k, v in ELEMENT_TO_IDX.items()}

# CPK-inspired colours for each element (used in the viewer)
ELEMENT_COLORS = {
    "H": "#FFFFFF",   # white
    "C": "#909090",   # grey
    "N": "#3050F8",   # blue
    "O": "#FF0D0D",   # red
    "S": "#FFFF30",   # yellow
}

# Van-der-Waals-ish radii (Å) for the sphere style — kept small so
# the ball-and-stick representation is easy to read.
ELEMENT_RADII = {
    "H": 0.25,
    "C": 0.40,
    "N": 0.38,
    "O": 0.35,
    "S": 0.45,
}

# Atomic weights for the elements
ELEMENT_WEIGHTS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
}


# ──────────────────────────────────────────────────────────────
# Helper: decode element from one-hot
# ──────────────────────────────────────────────────────────────

def _decode_element(onehot: torch.Tensor) -> str:
    """Recover element symbol from 5-dim one-hot in node features.

    The one-hot occupies columns 3-7 of the node feature vector x.
    If no bit is set (shouldn't happen), default to "C".
    """
    idx = int(onehot.argmax().item())
    return IDX_TO_ELEMENT.get(idx, "C")


# ──────────────────────────────────────────────────────────────
# PyG graph → PDB string conversion
# ──────────────────────────────────────────────────────────────

def graph_to_pdb_string(graph: Data) -> str:
    """Convert a PyG Data object back into a minimal PDB string.

    Uses graph.pos (nm → Å conversion) and the element one-hot
    encoded in graph.x[:, 3:8].

    The resulting PDB string can be loaded directly into py3Dmol.

    Args:
        graph: PyG Data object with .pos [N, 3] and .x [N, 8]

    Returns:
        Multi-line PDB string with ATOM + CONECT records
    """
    positions = graph.pos.numpy()         # [N, 3] in nm
    positions_angstrom = positions * 10.0  # convert nm → Å for PDB/viewer

    num_atoms = positions.shape[0]
    lines = []

    # --- ATOM records ---
    elements = []
    for i in range(num_atoms):
        elem = _decode_element(graph.x[i, 3:8])
        elements.append(elem)
        x, y, z = positions_angstrom[i]
        # PDB ATOM record format (simplified)
        line = (
            f"ATOM  {i+1:5d}  {elem:<3s} MOL A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"  1.00  0.00           {elem:>2s}  "
        )
        lines.append(line)

    # --- CONECT records (from edge_index) ---
    # Build an adjacency list, keeping only one direction (i < j) to
    # avoid duplicate bonds in the viewer.
    if graph.edge_index is not None and graph.edge_index.numel() > 0:
        adjacency = defaultdict(set)
        edge_idx = graph.edge_index.numpy()
        for col in range(edge_idx.shape[1]):
            a, b = int(edge_idx[0, col]), int(edge_idx[1, col])
            if a < b:
                adjacency[a].add(b)

        for src in sorted(adjacency):
            neighbours = sorted(adjacency[src])
            # PDB CONECT records use 1-based indices
            conect_str = f"CONECT{src+1:5d}"
            for dst in neighbours:
                conect_str += f"{dst+1:5d}"
            lines.append(conect_str)

    lines.append("END")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Single molecule visualization
# ──────────────────────────────────────────────────────────────

def visualize_molecule(graph: Data,
                       width: int = 600,
                       height: int = 400,
                       style: str = "ball-and-stick",
                       show_labels: bool = False,
                       save_html: Optional[str] = None):
    """Render a PyG molecular graph as an interactive 3D viewer.

    Produces a py3Dmol viewer with ball-and-stick (default) or other
    styles. Works inside Jupyter notebooks; for command-line use, pass
    ``save_html`` to export the viewer as an HTML file.

    Styles supported:
        - "ball-and-stick" : spheres + sticks (default, best for structure)
        - "stick"          : sticks only (cleaner for large molecules)
        - "sphere"         : CPK space-filling spheres
        - "cartoon"        : backbone ribbon (requires proper PDB residues)

    Args:
        graph:       PyG Data object from snapshot_to_graph
        width:       viewer width in pixels
        height:      viewer height in pixels
        style:       rendering style name
        show_labels: overlay atom index labels on each atom
        save_html:   if set, write the viewer HTML to this file path
    """
    try:
        import py3Dmol
    except ImportError:
        print("❌  py3Dmol is not installed. Install it with:")
        print("      pip install py3Dmol")
        return None

    pdb_str = graph_to_pdb_string(graph)

    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(pdb_str, "pdb")

    # Apply the requested visual style with CPK element colouring
    if style == "ball-and-stick":
        viewer.setStyle({
            "stick": {"radius": 0.15, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.25, "colorscheme": "Jmol"},
        })
    elif style == "stick":
        viewer.setStyle({
            "stick": {"radius": 0.15, "colorscheme": "Jmol"},
        })
    elif style == "sphere":
        viewer.setStyle({
            "sphere": {"colorscheme": "Jmol"},
        })
    elif style == "cartoon":
        viewer.setStyle({
            "cartoon": {"color": "spectrum"},
        })
    else:
        # Fallback
        viewer.setStyle({
            "stick": {"radius": 0.15, "colorscheme": "Jmol"},
            "sphere": {"scale": 0.25, "colorscheme": "Jmol"},
        })

    # Optional: label each atom with its index and atomic weight
    if show_labels:
        positions = graph.pos.numpy() * 10.0  # nm → Å
        for i in range(positions.shape[0]):
            elem = _decode_element(graph.x[i, 3:8])
            weight = ELEMENT_WEIGHTS.get(elem, 0.0)
            
            # Original label: index and element
            viewer.addLabel(
                f"{i}:{elem}",
                {
                    "position": {
                        "x": float(positions[i, 0]),
                        "y": float(positions[i, 1]),
                        "z": float(positions[i, 2]),
                    },
                    "fontSize": 10,
                    "fontColor": "white",
                    "backgroundOpacity": 0.6,
                    "backgroundColor": "black",
                },
            )
            
            # New label: weight on top of the node
            viewer.addLabel(
                f"{weight}",
                {
                    "position": {
                        "x": float(positions[i, 0]),
                        "y": float(positions[i, 1]) + 0.6,  # Shift up slightly
                        "z": float(positions[i, 2]),
                    },
                    "fontSize": 11,
                    "fontColor": "#FFFF00", # Yellow to stand out
                    "backgroundOpacity": 0.0,
                    "alignment": "center",
                },
            )

    # Title overlay
    peptide_id = getattr(graph, "peptide_id", "?")
    num_atoms = graph.pos.shape[0]
    num_edges = graph.edge_index.shape[1] if graph.edge_index is not None else 0
    viewer.addLabel(
        f"{peptide_id}  ({num_atoms} atoms, {num_edges // 2} bonds)",
        {
            "position": {"x": 0, "y": 0, "z": 0},
            "fontSize": 14,
            "fontColor": "white",
            "backgroundOpacity": 0.0,
            "inFront": True,
            "fixed": True,
            "alignment": "topLeft",
        },
    )

    viewer.zoomTo()
    viewer.setBackgroundColor("#1a1a2e")

    # --- Output ---
    if save_html is not None:
        html_content = viewer._make_html()
        os.makedirs(os.path.dirname(os.path.abspath(save_html)), exist_ok=True)
        with open(save_html, "w") as f:
            f.write(html_content)
        print(f"  💾 Saved interactive viewer → {save_html}")

    # In Jupyter this displays the widget; in CLI it returns the viewer
    return viewer


# ──────────────────────────────────────────────────────────────
# CLI entry point — load data + generate visualization HTMLs
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="Visualize py3Dmol molecular graphs.")
    parser.add_argument("--peptide", type=str, default="AA",
                        help="The 2-letter peptide sequence to search for and visualize (e.g. 'AA', 'AF', 'WW').")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split to search in ('train', 'val', 'test').")
    parser.add_argument("--frame", type=int, default=0,
                        help="The frame number to visualize (e.g. 0 to 299).")
    parser.add_argument("--source", type=str, default="real", choices=["real", "predicted"],
                        help="'real' (ground truth data) or 'predicted' (MLP output).")
    
    args = parser.parse_args()
    seq_id = args.peptide.upper()

    DATA_ROOT = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "timewarp_data", "2AA-complete"
    )

    print(f"Loading from: {DATA_ROOT}\n")

    # Step 1: Find the exact file for the requested sequence
    splits_to_search = ["train", "val", "test"]
    if args.split and args.split in splits_to_search:
        splits_to_search = [args.split] + [s for s in splits_to_search if s != args.split]

    matching_files = []
    found_split = None
    
    for split in splits_to_search:
        split_dir = os.path.join(DATA_ROOT, split)
        npz_pattern = os.path.join(split_dir, f"{seq_id}-traj-arrays.npz")
        matches = glob.glob(npz_pattern)
        if matches:
            matching_files = matches
            found_split = split
            break

    if not matching_files:
        print(f"❌ Error: Could not find any data for peptide '{seq_id}' in any split (train, val, test).")
        print(f"   Make sure '{seq_id}' is a valid sequence and try again.")
        sys.exit(1)
        
    print(f"✅ Found '{seq_id}' in split: {found_split}")

    npz_path = matching_files[0]
    pdb_path = npz_path.replace("-traj-arrays.npz", "-traj-state0.pdb")

    # Step 2: Load the specific frame
    from processing import parse_elements_from_pdb, build_edge_index_from_positions, snapshot_to_graph
    import numpy as np
    
    print(f"Loading '{seq_id}' molecule at frame {args.frame} (Source: {args.source.upper()})...")
    
    elements = parse_elements_from_pdb(pdb_path)
    
    # Always load ground truth initially to get edge connections
    real_arrays = np.load(npz_path)
    static_edge_index = build_edge_index_from_positions(real_arrays["positions"][0])

    if args.source == "predicted":
        pred_npy_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "trajectory_prediction", "trajectory_pdb_files", f"{seq_id}_predicted_trajectory.npy"
        )
        if not os.path.exists(pred_npy_path):
            print(f"❌ Error: Predicted data not found at {pred_npy_path}.")
            print("   Run predict_trajectory.py first!")
            sys.exit(1)
            
        # Predicted array is saved as [num_atoms, num_frames, 3]
        # We transpose it to [num_frames, num_atoms, 3] to match ground truth format
        positions = np.load(pred_npy_path)
        positions = np.transpose(positions, (1, 0, 2))
    else:
        # Ground truth array is already [num_frames, num_atoms, 3]
        positions = real_arrays["positions"]
    
    if args.frame < 0 or args.frame >= positions.shape[0]:
        print(f"❌ Error: Frame {args.frame} is out of bounds (0 to {positions.shape[0]-1}).")
        sys.exit(1)
        
    g = snapshot_to_graph(
        positions_t=positions[args.frame],
        elements=elements,
        edge_index=static_edge_index,
        peptide_id=seq_id
    )

    if not g:
        print("❌ No valid molecule could be extracted.")
        sys.exit(1)

    print(f"  Sample graph: x={g.x.shape}, edges={g.edge_index.shape}, "
          f"pos={g.pos.shape}, peptide={g.peptide_id}\n")

    # ── Output directory ────────────────────────────────────────
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "visualizations"
    )
    os.makedirs(output_dir, exist_ok=True)

    # ── 1) Single molecule view ─────────────────────────────────
    print(f"🔬 Visualizing single molecule frame {args.frame}: {g.peptide_id} ({args.source})")
    html_path = os.path.join(output_dir, f"{g.peptide_id}_frame_{args.frame}_{args.source}.html")
    visualize_molecule(
        g,
        style="ball-and-stick",
        show_labels=True,
        save_html=html_path,
    )

    print(f"\n✅ Visualization HTML file saved to: {output_dir}/")
    print("   Open it in a browser for an interactive 3D view.")

