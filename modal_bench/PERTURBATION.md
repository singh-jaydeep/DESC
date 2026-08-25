# How to perturb an equilibrium (and three ways that look right and are not)

**Read this before writing any code that perturbs a DESC equilibrium.**
Every wrong method below produced results that looked plausible -- the solver
ran, reported "Optimization terminated successfully", and returned a cost -- and
all of them were meaningless. Measured at `precise_QA`, `L=M=N=12`, nominal 1%:

| method | start cost | solved geometry |
|---|---|---|
| unperturbed control | `1.5e-09` | valid |
| **`eq.perturb` on `Rb_lmn`/`Zb_lmn`** | **`5.5e-02`** | **valid, `gtol` in 13 iters** |
| interior spectral, absolute | `~1e+24` | **folded flux surfaces** |
| interior spectral, relative | `~1e+24` | **folded flux surfaces** |
| replace `eq.surface`, re-solve | `9.4e+24` | **folded flux surfaces** |

## The right way

```python
rng = np.random.default_rng(seed)
Rb, Zb = np.asarray(eq.Rb_lmn), np.asarray(eq.Zb_lmn)
dRb = pert * rng.standard_normal(Rb.size) * np.abs(Rb)
dZb = pert * rng.standard_normal(Zb.size) * np.abs(Zb)
eq = eq.perturb(
    deltas={"Rb_lmn": dRb, "Zb_lmn": dZb},
    objective=ObjectiveFunction(ForceBalance(eq), jac_chunk_size=chunk),
    constraints=get_fixed_boundary_constraints(eq),
    order=2, copy=True,
)
```

`eq.perturb` solves the *linearised* problem so the interior moves consistently
with the new boundary, and its default `weight="auto"` weights by
`(mode number)**2` so high-order modes are not over-driven.

## Why each wrong way is wrong

**Interior spectral, absolute** -- `eq.R_lmn += pert * randn(size) * abs(eq.R_lmn).max()`.
White noise scaled to `max|R_lmn|`, i.e. to the R00 major-radius coefficient,
applied uniformly with no mode-number weighting. An equilibrium spectrum decays
steeply: at L=12, 98.9% of R modes lie below 1% of max, the perturbation exceeds
the mode's own magnitude for 98.9% of modes, and the median ratio of
perturbation to coefficient is **5.0e+03**. A nominal "1%" is `||delta||/||v|| = 0.32`.
The result is a randomly corrugated surface, not a perturbed equilibrium.
*This is the version in `branch_experiments/solve_bench.py`, and it is the one
that has fooled people. It is plausible-looking, short, and completely wrong.*

**Interior spectral, relative** -- `eq.R_lmn *= (1 + pert*randn)`. Fixes the
hierarchy problem (`||delta||/||v|| == pert` exactly) but still perturbs the
interior *independently of the boundary*, so the state is not near any
equilibrium. Better, still unusable.

**Replace `eq.surface` and re-solve.** Looks the most physical of the three and
is the most misleading. The interior coefficients are left untouched, so
`LinearConstraintProjection` must jump them onto the new boundary in one
discontinuous step. Worse, `eq.compute()` afterwards reads the *interior*
coefficients and therefore reports the state as unchanged and healthy -- the
damage is invisible to diagnostics until the solve starts.

## How to tell you got it wrong

Diagnose the solution, not the reported cost:

* `sqrt(g)` must be single-signed. A sign change means the flux surfaces have
  folded through each other and the "equilibrium" is not one.
* Aspect ratio must be **identical** between the perturbed start and the
  solution, since the boundary is fixed. If `R0/a` moves, something is broken.
* Compare against the unperturbed control at the same resolution. If a "1%"
  perturbation moves the starting cost by more than a few orders of magnitude,
  stop and find out why.

`eqdiag.py` computes all of these.
