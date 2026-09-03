# Benchmark — Stage 2 vs Stage 3

Head-to-head comparison of the two generators on the **same peptide**, scored
against the same ground-truth MD.

## Run

```bash
cd MolGraph-AE--main
python -m benchmark.run_benchmark --peptide AC --frames 100 --start-frame 50
```

Or with no flags, and it prompts for all three in the terminal:

```bash
python -m benchmark.run_benchmark
Peptide [AC]: AC
Frames to generate [300]: 100
Stage-3 start frame (real MD frame to seed from) [0]: 50
```

### Inputs

| Input | Meaning | Applies to |
|---|---|---|
| `--peptide` | which dipeptide, e.g. `AC` | both |
| `--frames` | how many frames to generate, e.g. `100` | both |
| `--start-frame` | which **real MD frame** the rollout is seeded from, e.g. `50` | **Stage 3 only** |

Optional: `--temperature`, `--flow-steps`, `--bond-fix` (Stage-3 sampling knobs),
`--seed`, `--gt-stride`, `--device`.

## Outputs

Written to `benchmark/results/{PEPTIDE}_f{frames}_s{start}/`:

| File | Contents |
|---|---|
| `ramachandran_comparison.png` | **Figure 1** — three panels: MD \| Stage 2 \| Stage 3, all over the same MD contour |
| `kl_divergence_comparison.png` | **Figure 2** — φ and ψ marginal histograms + a KL/JS bar chart |
| `benchmark_metrics.txt` | Full report including the Stage-3-only window scores |
| `benchmark_summary.csv` | One row per stage, for collecting across peptides |
| `{PEPTIDE}_benchmark_dihedrals.npz` | Raw φ/ψ for both stages + both GT views, and the Stage-3 positions |

---

## The one asymmetry — read this before quoting the numbers

The two stages are **not** given the same information, and they cannot be:

|  | Stage 2 | Stage 3 |
|---|---|---|
| What it is | distribution model | propagator |
| Conditioned on | molecule identity only (64-d pooled node embedding) | starting structure + 32-d frozen `z` |
| Can be seeded from a frame? | **No** | Yes (`--start-frame`) |
| Frame count | **locked to 300** by its `frame_pos` embedding `[1, 300, 128]` | any |
| Has a time axis? | **No** — the Sliced-Wasserstein loss is permutation-invariant, so frame order carries no meaning | Yes |

Consequences, both handled explicitly in the code rather than papered over:

* **`--start-frame` is ignored by Stage 2.** It is accepted and discarded in
  `stage2.generate()`, and the report says so.
* **Frame counts are reshaped for Stage 2.** For `n ≤ 300` it takes an
  evenly-spaced subsample of its native 300 (spreading the picks keeps the full
  distribution represented at small `n`); for `n > 300` it draws `ceil(n/300)`
  independent samples and concatenates.
* **Both are scored against the full-MD equilibrium density**, not against the
  seeded window. That is the only reference Stage 2 can be fairly measured
  against. Stage 3's window score is reported separately in the `.txt` and never
  mixed into the comparison.

---

## What these two figures do and do not measure

Both figures measure **distribution match only** — does the model visit the
right Ramachandran basins in the right proportions.

That is precisely the objective **Stage 2 was directly trained on** (Sliced-
Wasserstein distance against the ground-truth φ/ψ point set). Stage 3 was
trained on *one-step transitions*; its distribution is an emergent byproduct of
compounding N autoregressive steps. So on these two figures Stage 2 has a
structural advantage, and beating it is not what Stage 3 is for.

**Stage 3's actual claim is temporal**: it produces a trajectory with a
physically meaningful decorrelation timescale, which Stage 2 cannot produce at
all at any quality. Neither the Ramachandran plot nor the KL histogram can see
this — a model drawing i.i.d. samples from the perfect equilibrium distribution
would score flawlessly on both while having no dynamics whatsoever.

If the benchmark needs to show why Stage 3 exists, it needs a third figure:
the **dihedral autocorrelation** comparison (already implemented in
`temporal_dynamics/evaluate.py` as `circular_acf`). On that axis Stage 2 has no
curve to plot, because its frame ordering is arbitrary.

---

## Requirements

* `encoder.pt` (Stage 1) in the repo root
* `trajectory_pre_/best_trajectory.pt` (Stage 2)
* `temporal_dynamics/best_propagator.pt` + `temporal_dynamics/cache/` (Stage 3 —
  run `prepare_data.py` and `adapter.py` first if the cache is missing)

The peptide must be present in the Stage-3 embedding cache; if it is not, add it
to the transition cache and re-run `python -m temporal_dynamics.adapter`.
