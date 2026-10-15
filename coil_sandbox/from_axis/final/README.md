# Two quasi-axisymmetric stellarators with demountable two-face coils

Two vacuum QA stellarators (NFP 2) whose coils are each two planar faces joined at two hinges, with the hinges at or
near the midplane, so that the coil set can be taken apart into an upper and a lower half. Found by a near-axis
search for this coil class (Giuliani et al., JCP 2022, adapted), then refined by a single-stage optimization of the
equilibrium and coils together. Each final pair (coils, equilibrium) is the vacuum **free-boundary** solution of the
coils, checked independently by field-line tracing of the coils.

| | **B** (`designB/`) | **D** (`designD/`) |
|---|---|---|
| coils | 4 per half period (16), **every hinge on z = 0** | 5 per half period (20), hinges within +-0.033 m of z = 0, hinge tilt <= 2 deg |
| aspect ratio, R0, a | 7.26, 0.877 m, 0.121 m | 7.56, 0.929 m, 0.123 m |
| iota (traced), axis -> edge | 0.238 -> 0.218 | 0.399 -> 0.350 |
| QS: L2 of breaking Boozer modes, rho .25/.5/.75/1 | 0.23 / 0.28 / 0.49 / 0.90 % | 0.24 / 0.21 / 0.35 / 0.78 % |
| largest breaking mode | 0.12 / 0.14 / 0.22 / 0.44 % | 0.18 / 0.09 / 0.19 / 0.32 % |
| eps_eff, rho .25/.5/.75 | 6.4e-4 / 6.7e-4 / 1.3e-3 | 2.6e-4 / 2.2e-4 / 7.2e-4 |
| Gamma_c Nemov / Velasco, rho .5 | 1.0e-2 / 6.1e-3 | 7.6e-3 / 3.5e-3 |
| coils' B.n rms on the boundary | 8.7e-6 | 1.6e-5 |
| coils against the bounds | feasible (L 0.9999, kappa 0.996, d_cc 1.13, d_pc 1.20) | feasible (L 0.9998, kappa 0.998, d_cc 1.03, d_pc 1.15) |
| island chains (dense tracing) | 2/9 at rho ~0.87, 2-3 mm | 4/11 at rho ~0.815, ~1.7 mm |
| reproducibility | chain recorded; early steps from uncommitted code (gaps listed) | chain fully recorded; near-axis stage rerun bit for bit |

Coil bounds for both: the reactor set of Gil et al. (PRE 114, 025202, Table III), scaled by the near-axis minor radius
0.167 m (coil length <= 3.941 m, curvature <= 10.17 /m, coil-coil >= 0.04 m, coil-plasma >= 0.148 m).

Against the literature (`compare/COMPARE.md`, same script for all six): both designs have breaking modes comparable
to DESC's `reactor_QA`. B's largest mode is lower than reactor_QA's at every radius, and D's L2 is lower inside
rho 0.75. Both are 4-8x better than NCSX and ARIES-CS, and 30-80x worse than `precise_QA`, a fixed-boundary
equilibrium with no coils.

## Layout

| path | what |
|---|---|
| `designB/`, `designD/` | self-contained packages: `README.md`, `data/` (coils.h5, eq.h5, bounds.json), `METRICS.md` + `metrics.json`, `figures/`, `make_figures.py` (regenerates all of it from `data/`), `REPRODUCE.md`, `drivers/` (frozen scripts per step) |
| `compare/` | `COMPARE.md` and overlay plots: B, D, precise_QA, reactor_QA, NCSX, ARIES-CS through the same `make_figures.py` |
| `PROVENANCE_DETAIL.md` | line-by-line sourcing of every laptop command (logs, transcripts, commits) |
| `environment_laptop.txt` | `pip freeze` of the environment the laptop steps and the metrics ran in |
| `freeze_drivers.py` | writes `designX/drivers/` from git, each file at its step's commit |

## How to regenerate

```
cd coil_sandbox/from_axis/final/designD      # or designB
systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0 python make_figures.py     # ~15 min, laptop GPU
for r in precise_QA reactor_QA NCSX ARIES-CS; do python make_figures.py --ref $r --out ../compare/ref_$r.json; done
cd ../compare && python compare.py
```
DESC is the repository checkout four levels up (`$DESC_REPO` overrides it); commit and versions in each `REPRODUCE.md`.

## Not yet done (`../FINALIZE_HANDOFF.md`, steps 3-6)

Coil forces and torques, currents scaled to a reference machine, winding-pack clearance, finite beta; the remaining
figures (hinge close-up, near-axis target vs solved axis, zooms at the rationals); tolerances (coil displacement and
current errors); the stage-2 family of coil sets for each equilibrium.
