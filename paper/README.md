# Piecewise-planar coil paper: scripts and inputs

- `inputs/`: coil starts, near-axis states, equilibria and bounds of every example, copied
  unchanged from the poster-era branch (`inputs/README.md`, `inputs/MANIFEST.sha256`).
- `stage2.py`: stage-2 driver (fixed equilibrium, every engineering bound a hard constraint of
  `lsq-auglag-composite`). `python paper/stage2.py --help`.
- `lib/dense_check.py`: bounds files, equilibrium loading, and the dense check every run reports
  at the start, after each outer iteration and at the end.

Run from the repository root with this DESC first on the path:

```bash
PYTHONPATH=. systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 \
    python paper/stage2.py --eq EQ --start COILS.h5 --bounds BOUNDS.json --out DIR [options]
```

On an 8 GB GPU with a pointwise B.n cap (`--bn-max`), set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`.

Each run directory holds `cmd.txt`, `commit.txt`, `run.log`, `outer.json`, `x_last.npy`,
`result.h5` and `result.json`.

Differences from the poster driver (`coil_sandbox/run_al.py` on js/qa-poster):

- Distances are always `CoilSetDistanceRows` (unsigned) and `PlasmaCoilSetDistanceRows` (plasma
  boundary fixed). The poster runs used the stock hard minimum (`--dist hybrid`) or a distance
  field for the plasma. Plasma rows default to a 128 x 128 half plasma grid: the row gap grows as
  (cell size)^2 x surface curvature, 47 mm at 32 x 32 on design F, 5 mm at 128 x 128.
- No penalty solvers, kink probe, link guard, demountability report, FourierXYZ coils or kappa_MS.
- `--curv-grid arc` bounds |kappa| on n midpoint nodes per arc (`coil_curvature_grid`); the default
  `linear` is the poster's `LinearGrid(N=300)`, whose sampled maximum misses the interior peaks of
  the paper's arc coils by up to 0.1%.
