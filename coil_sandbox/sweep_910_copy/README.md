# precise_QA error-vs-length sweep — self-contained bundle

Everything needed to run this sweep on a machine that has never seen the DESC repo.
**Move or copy this whole directory; nothing outside it is required.**

```bash
cd sweep_910
python check_bundle.py                              # verify first, always
python tools/run_sweep.py sweep_grid.json --dry-run # see the plan
python tools/run_sweep.py sweep_grid.json           # run it
python tools/results.py                             # see what came out
```

## Read these, in this order

| file | what it is |
|---|---|
| `fixed_configs.md` | every parameter held constant — equilibrium, objective, constraints, bounds, grids, AL settings, and **§11: what is not standard DESC** |
| `RUNS.md` | the 42 runs: representation, start, and length bound |
| `initial starts/README.md` | the seven cold starts, their feasibility, and the coil viewer |
| `initial starts/viewer.html` | **open this** — all seven coilsets, in a browser, offline |

## What is here

```
sweep_910/
  check_bundle.py        verify integrity + that the vendored DESC is the one in use
  sweep_grid.json        the 42 runs as data (generated from tools/sweep_configs.py)
  RUNS.md  fixed_configs.md
  initial starts/        7 cold-start coilsets, pinned by SHA-256, + viewer.html
  tools/                 the driver and everything it imports
  vendor/                the DESC this sweep runs against, + the diff vs master
  runs/                  output; one directory per run (created as it goes)
```

## The DESC in `vendor/` is not optional

The sweep depends on DESC code that is **not in master**: the entire arc representation,
the mean-square curvature constraint, unsigned `|curvature|`, signed coil-coil distance,
and a heavily modified augmented Lagrangian. `fixed_configs.md` §11 lists all of it.
Against stock DESC the sweep does not merely give different numbers — most of it does not
import.

**`vendor/desc/` is a snapshot of the working tree, not of any commit.** Six files under
`desc/` had uncommitted changes when it was taken, so checking out the recorded `head`
does **not** reproduce it. `vendor/MANIFEST.json` records the branch, the commits, which
files were dirty, and a SHA-256 for every vendored and tool file.
`vendor/desc_changes.patch` is the full diff against master, for review or for porting
onto a fresh DESC clone.

Every tool imports `tools/_bundle.py` **before** importing `desc`, which prepends
`vendor/` to `sys.path` so the vendored copy shadows any installed DESC. This matters:
on the machine the bundle was built on, an unaided `import desc` resolved to a *different*
DESC. If `desc` somehow gets imported first, `_bundle` emits a warning rather than
silently using the wrong code — do not ignore it.

You still need the Python dependencies (`jax`, `numpy`, `scipy`, …):
`pip install -r vendor/requirements.txt`.

## Running it

Chains are sequential **within** a representation and the dependency edges never cross
representations, so the seven chains are independent:

```bash
python tools/run_sweep.py sweep_grid.json --rep arcB2     # one chain
python tools/run_sweep.py sweep_grid.json --status        # where am I?
python tools/run_sweep.py sweep_grid.json --resume        # carry on after a crash
```

Resume is simple by design: finished configs are skipped and the crashed one is re-run
from its start — no solver state is saved or restored, so the most you lose is the one run
that was in flight. A solve that finished but was not adjudicated is adjudicated without
re-solving. See `HANDOFF.md` §4a.

`run_sweep.py` is deliberately serial and enforces two things learned the hard way:

* **Adjudication never runs alongside a solve.** Two runs died to a concurrent `verify`
  on a 15 GB machine. Each point is solve-then-adjudicate.
* **A warm run will not start without its donor's final `h5`.** A run stopped by hand
  writes checkpoints but no final coilset, and a warm start from a stale or missing file
  silently seeds the wrong geometry.

Each run is wrapped in a `systemd-run` cgroup at `MemoryMax=12G`. Pass `--no-cgroup` if
that is unavailable, but note WSL2 has **no OOM killer**, so without the cap a run that
grows takes the machine down instead of itself. Change the cap with `--mem`.

**Cost: ≈41 h sequential**, ~8.5 h at seven workers (set by the `xyzN6` chain). At 12 GB
per run, a 15 GB machine fits exactly one at a time.

## Checking your work

```bash
python check_bundle.py --deep     # also rebuilds the 7 starts and compares hashes
python tools/results.py --md RESULTS.md
python tools/make_coil_viewer.py runs/view.html --dir runs   # look at what you made
```

**Look at the coilsets.** `tools/make_coil_viewer.py` builds a dependency-free HTML
viewer for any set of `h5` files, and it is the standard for this campaign — not
`plotly.write_html`, which embeds ~8 MB of library per file. It draws all 16 coils (the
representations nest differently on disk and a naive walk yields 4 for `xyz`, silently),
marks the closest approach, and reports linked pairs from the pairwise matrix.

To compare a solved run against the start it came from:

```bash
python tools/make_coil_viewer.py /tmp/cmp.html \
    --h5 "start=initial starts/start_arcB5.h5" \
    --h5 solved=runs/sw_arcB5_L1/sw_arcB5_L1.h5
```

## Three things that will mislead you

* **`linked pairs` is a hard rejection.** Two points in this campaign were banked as
  improvements before anyone read that field, and both improvements *were* the linking
  (8.6% and 18.4%). A warm start's flux gain is not a result until it reads 0.
* **Never trust the solver's own verdict.** With `ctol=1e-4` a stalled run reports
  `precision` (SUCCESS) for violations in 1e-6..1e-4. Adjudicate with
  `tools/adjudicate.py`, which is what `run_sweep.py` calls.
* **`d_cc` is not grid-converged at `VERIFY_N=400`** on the most contorted coilsets —
  still moving 1.2% at N=800. `run_sweep.py` adds `--dcc-converge` automatically at ×1.4
  and ×1.6; if it reports drift, do not quote `d_cc` for that point.

## Provenance

Built 2026-09-10 from branch `c0-testing` of the DESC repo. The campaign notes that led
to these choices — `CONVENTIONS.md` (what was measured), `SWEEP_BLUEPRINT.md` (what was
decided), `SWEEP_RESULTS.md` (the board) — live in `c0/` in that repo and are **not**
vendored here; `fixed_configs.md` carries everything this sweep actually depends on.

Every point is **one seed**. `--seed` and a smooth-representation perturbation do not
exist yet, so multiseed cannot currently be configured.
