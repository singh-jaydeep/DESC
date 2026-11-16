# Near-axis stage (PORT_PLAN 9, stages 1-2)

Joint optimization of a near-axis quasisymmetric target (axis, eta_bar, second order) and folded
two-arc coils fitted to its on-axis field and field gradient (after Giuliani et al., JCP 2022), then
conversion to a DESC equilibrium and `PiecewisePlanarArcCoil` coilset. Ported from
`coil_sandbox/from_axis/na` (js/qa-poster) with the imports pointed at `paper/lib`.

| file | what |
|---|---|
| `nae.py` | first-order near-axis QS field in JAX (sigma equation, B, grad B, helicity) |
| `nae2.py` | second order (port of pyQSC 0.1.3 `calculate_r2`, vacuum terms only) |
| `folded.py` | the folded coil class (z = 0 or free hinges), full coilset by symmetry, polyline Biot-Savart |
| `run_na2.py` | the driver: `lsq_composite` (secant) on axis + eta_bar + coils, CPU |
| `cont.py` | print the `run_na2.py` command that continues a saved run with overrides |
| `ncchain.py` | the coil-count chain (QA nfp 2): cold start, weak second order, final second order, hinge band. Reference arguments in `refs/` and `../inputs/near_axis/` |
| `to_desc.py` | near-axis state -> pyQSC surface at r = a -> DESC equilibrium (fixed-boundary solve) + arc coilset; scores |
| `score_fb.py` | free-boundary score of a near-axis coilset (`../lib/bdry_repair` with `--repair`) |
| `test_nae.py` | `python test_nae.py pylib`: first order vs pyQSC on the Landreman-Sengupta configurations (sections 5.1 and 5.2 agree to 1e-9; 5.3 and 5.5 have ingredients `nae.py` does not model) |
| `pylib/qsc` | pyQSC 0.1.3, vendored (MIT; `pylib/qsc-0.1.3.dist-info`) |

Run from this folder with the branch's DESC first on the path, e.g.

```bash
PYTHONPATH=../.. python cont.py ../inputs/near_axis/o2w1_qa2s2_gg1e-1_w1e-2_w1x10 runs/B_cont --maxiter 50
PYTHONPATH=../.. python to_desc.py ../inputs/near_axis/li_qa2_iota0.30 --r 0.12 --LMN 12 --out runs/B_desc
```

`--device gpu|cpu` selects the device (`../lib/device.py`); `run_na2.py` defaults to the CPU, the DESC
solves to the GPU.

**Not yet done (PORT_PLAN 9, stage 1):** the coil bounds (d_pc, d_cc, kappa, L, current) are still
penalty rows of the least-squares objective (`wpen`). The plan moves them to hard constraints of
`lsq_auglag_composite` through its function-level API; that change needs runs to validate.
