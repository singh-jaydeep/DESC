# Stage-2 coil optimality tooling

Scripts behind the stage-2 coil investigations (precise_QA, then precise_QH with planar
coils) that ended in certified optima, kept for the augmented Lagrangian work on this
branch. Run them from this directory (they import each other) with the `.venvDEV`
interpreter, one heavy job at a time on a small machine. Run outputs (`*.npy`, `*.h5`,
`*.json`, `*.log`) are ignored by git.

`.venvDEV` imports the main checkout's `desc`; point it at this worktree, e.g.
`PYTHONPATH=/Users/singh/PycharmProjects/DESC-coil-auglag python run_al.py ...`.

**Start with `GUIDE.md`**: how to use the new objectives and solvers, choose resolutions
and set up the augmented Lagrangian, with precise_QH as a worked example.

The history, numbers and reasoning are in `notes/`: `stage2_crawl_handoff.md`
(precise_QA: F1-F4, the composite hinge model) and `stage2_qh_planar_handoff.md`
(precise_QH: smooth distance rows, linking, structured secant, exact-Hessian finish).
`auglag_plan.md` is the task list for the next phase (composite solver in DESC, node-node
distance rows with a gap bound, slack-free hinge augmented Lagrangian).
The two investigation notes predate this branch; the names they use map as follows.

| in the notes | now |
|---|---|
| `scratchpad_crawl/` | `devtools/coil_auglag/` |
| `CoilSetCurveRows`, `PlasmaCoilSetCurveRows` (`curverows.py`) | `desc.objectives.CoilSetDistanceRows`, `PlasmaCoilSetDistanceRows` |
| `CoilArclengthResidual` prototype (`setup.py`) | `desc.objectives.CoilArclengthResidual` |
| `CoilCurvature(signed=...)`, default signed | `CoilCurvature(signed=False)` default, `\|curvature\|` |
| `segcheck.py` | `check.py` (brute-force distances, linking, length, curvature) |
| `CoilSetPairDistance`, `CoilSetCutoffPairDistance`, `PlasmaCoilSetPairDistance`, `segrows.py`, `stages.sh`, `missed.py`, `newton.py`, `gauge.py`, soft-hinge and mode-regularization objectives | removed (superseded) |
| builder options `cc="pairs"/"cut"/"seg"/"curve"` | `cc="rows"` or `"hard"` (same for `pc`) |

## Files

| file | what |
|---|---|
| `setup.py`, `setup_qh.py`, `cases.py` | problem builders; `cases.build(case="qa"/"qh", **kw)` |
| `run.py` | stock `lsq-exact` driver (verbose log, `allx`) |
| `run_composite.py TAG KW [OPTS]` | `Optimizer("lsq-composite")` driver (the DESC port of `ctr.py`); `--start X_x.npy` |
| `run_al.py TAG KW [OPTS]` | `Optimizer("lsq-auglag-composite")` driver on the builder's AL mode (`al=True`, node rows); outer-loop log in `TAG_outer.json` |
| `mkstage.py TAG X KW` | make a run tag at a full state (`init` = the builder's coils) |
| `ctr.py ARM TAG MAXITER OUT [Y]` | controlled trust region: `std`, `comp` (composite hinge model), `compS` (+ structured secant S), `comp2` (+ exact S); checkpoints every 50 its |
| `check.py KW FILE...` | feasibility and topology at coilset files or `*_x.npy` states (`feasibility()` is reused by `run_al.py`) |
| `diag.py TAG X [--hess]` | cost per objective; with `--hess` the optimality certificate |
| `sdecomp.py`, `pintest.py` | dropped Gauss-Newton term per objective; gauge pinning test (planar coils) |
| `trsteps.py`, `hspec.py`, `lk.py`, `common.py` | trust-region ladder, Hessian spectrum, linking numbers, loaders |

## Hands-off precise_QH run

```
KW='{"case":"qh"}'                          # rows distances, fixnorm, weights 200/300/100/100/200
python mkstage.py E1 init "$KW"
python ctr.py compS E1 1000 E1o             # composite + structured secant
python ctr.py comp2 E1 8 E1n E1o_y.npy      # a few exact-Newton steps
python check.py "$KW" E1n_x.npy
python diag.py E1 E1n_x.npy --hess
```

Previously certified with this sequence: scaled gradient 7.7e-9, PD Hessian, coil pair
(2, 3) linked at the penalty optimum (unlinking needs AL multipliers).

The same in DESC, secant then exact finish in one solve (exact-curve rows, as above):

```
python run_composite.py LC1 '{"case":"qh"}'   # {"hessian": "secant", "finish_steps": 8}
```

Builder kwargs for the augmented Lagrangian: `al=True` (QuadraticFlux objective, limits as
constraints), `distance="node"` (node-node rows with a gap) and `pair_N=None` (node count
from the gap); `run_al.py` sets all three.

## Known issues

- `tests/test_optimizer.py::test_auglag` fails: the c0-testing `lsq_auglag` stops
  infeasible ("subproblems repeatedly stalled", max violation 2.2e-3) where master and
  `fmin_auglag` converge. To be addressed in the augmented Lagrangian work.
- Open in the c0-testing augmented Lagrangian (from the branch code review):
  `fmin_auglag` resets its stall counter on any accepted step, so ftol/xtol stalls
  never terminate and runs go to `maxiter`; a `step_veto` that keeps rejecting skips
  the termination checks and spins the outer loop to `maxiter`;
  `_trust_region_step_eigh` puts the step on the trust-region boundary for a singular
  PSD model even when the minimizer is inside.
