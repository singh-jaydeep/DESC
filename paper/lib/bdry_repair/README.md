# bdry_repair: fix a degenerated boundary parametrization without moving the surface

**Symptom.** During free-boundary or single-stage solves the boundary's poloidal angle stalls at a point: the area
element `|e_theta x e_zeta|` goes to zero there, the largest curvature jumps from tens to 1e4-1e5 per metre, `sqrt(g)`
turns negative near the edge, and B.n shows a spike of 10-20% at a mm-scale kink. B.n and QS cannot see a
re-parametrization, so nothing in the objective stops the drift.

**Use** (any script that has `coil_sandbox/sandbox` on `sys.path`, i.e. after `import boot`; depends only on `desc`):

```python
from bdry_repair import repair, reparametrize, needs_repair, displacement

eq, info = repair(eq)                    # gate -> refit -> interior re-solved by continuation -> verify
surf, hist = reparametrize(eq)           # surface only (also takes a FourierRZToroidalSurface); hist[-1]: mm moved, health
bad, health = needs_repair(eq)           # the gate alone
displacement(old, new, NFP)              # parametrization-independent distance between two surfaces, mm (rms, p99, max)
```

`repair` returns `eq` itself, untouched, when the boundary is healthy (`info["action"] == "none"`); pass `force=True`
to refit anyway. Command line with before/after B.n, QS, Jacobian: `from_axis/ss/repair_eq.py EQ COILS OUT`.

## How it works

1. **Sample** the old surface at equal-arclength poloidal labels in each toroidal plane (`relabelled_points`; other
   rules: `"curvature"`, `"centripetal"`). The old theta = 0 curve stays the origin, which keeps stellarator symmetry.
2. **Guarded fit** (`guarded_fit`). A linear fit at those labels has a clean angle but moves the surface by that
   angle's truncation error. DESC PR 2337's joint fit of labels and coefficients removes the truncation error, but run
   to convergence it returns to the creased angle (the one parametrization that reproduces the old surface exactly,
   kink included). So its Levenberg-Marquardt iteration is advanced one step at a time and a step is kept only while
   the smallest area element stays above 0.85x, and the largest curvature below 1.5x, the fixed-label fit's values;
   it stops when a step gains under 3%. One to three steps do nearly all the work.
3. **Interior** (`interior.py`): re-solved from scratch by continuation, so no folded flux surfaces are inherited.
   Other strategies can be added to `INTERIOR` or passed as a callable.
4. **Verify**: if the re-solved edge is still unhealthy, redo with fewer label steps, down to the fixed-label fit.

`_surface_fit_pr2337.py` is PR 2337's `desc/geometry/_surface_fit.py`, verbatim. Only its fixed-zeta path is used, so
the generalized toroidal angle that PR is stacked on is not needed. Replace with the DESC import once it merges.

## Measured (M = N = 12, 2026-10-13)

Surface displacement of the refit, old (fixed equal-arclength labels) -> new (guarded), mm:

| boundary | rms | 99th percentile | max |
|---|---|---|---|
| iota 0.42 start, creased (`ss12_ref042_q15/start/eq0_b12.h5`) | 0.67 -> 0.12 | 3.3 -> 0.56 | 5.2 -> 1.5 |
| same design after k = 1, clean | 0.39 -> 0.016 | 1.6 -> 0.05 | 3.8 -> 0.10 (and the gate skips it) |
| iota 0.30 design B, creased (`li030/fb_norepair_local/eq_fb.h5`) | 0.22 -> 0.09 | 1.0 -> 0.36 | 1.7 -> 1.2 |

The remaining maximum on creased boundaries is at the kink, which is the part meant to be smoothed.

Full repair, before -> after (QS = L2 of symmetry-breaking Boozer modes / <B> at rho 0.25 / 0.5 / 0.75 / 1):

| | iota 0.42 start | design B |
|---|---|---|
| QS before | 0.35 / 0.67 / 1.31 / 2.37 % | 0.25 / 0.28 / 0.47 / 0.89 % |
| QS after, new | 0.29 / 0.71 / 1.34 / 2.37 % | 0.25 / 0.28 / 0.47 / 0.95 % |
| QS after, old repair | 0.90 / 1.09 / 1.78 / 3.62 % | 0.28 / 0.38 / 0.55 / 1.22 % (after a repaired rescore) |
| coils' B.n on the boundary, rms | 0.092 -> 0.174 % | 0.140 -> 0.104 % |
| B.n max | 7.3 -> 0.88 % | 22.6 -> 0.44 % |
| sqrt(g) min/median at rho = 1 | -0.10 -> +0.57 | -0.10 -> +0.65 |

Results: `from_axis/ss/qa2best/ref042_start_repaired/`, `from_axis/ss/li030/repaired/`.

## Limits

* The toroidal angle must be the cylindrical one (zeta = phi); only the poloidal angle is repaired.
* The "after" states are fixed-boundary solves on the repaired surface. B.n is the coils' field on that surface, so
  it is a fair number, but no free-boundary shell has been run from a repaired state with this version yet.
* Memory: the continuation builds an 11 GB Jacobian at L = M = N = 12. On a 16 GB card run with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.93` and nothing else on the device.
* The gate thresholds (`HEALTHY`) are set from this project's QA cases (clean: area 0.55-0.65, curvature 40-65 /m).
