"""
benchmark — head-to-head comparison of Stage 2 vs Stage 3 on the same peptide.

Stage 2 (trajectory_pre_/) generates the equilibrium phi/psi distribution from
the molecule's identity alone.
Stage 3 (temporal_dynamics/) propagates a real starting structure forward in
time and reads the distribution off the resulting trajectory.

Both are scored against the same ground-truth MD density so the numbers are
directly comparable. See README.md for the run command and for the one
structural asymmetry between the two stages.
"""
