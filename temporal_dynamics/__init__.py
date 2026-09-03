"""
temporal_dynamics — Stage 3 of MolGraph-AE.

Stage 1 learned WHAT a peptide is (a 32-d graph embedding z).
Stage 2 learned WHERE it goes (the equilibrium phi/psi distribution).
Stage 3 learns HOW IT MOVES: a learned Markov propagator that carries the full
Cartesian state (positions AND velocities) forward one saved frame at a time,
so a 300-frame trajectory can be rolled out autoregressively.

See README.md in this directory for the run order.
"""
