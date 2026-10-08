# Fixed configuration

Everything held constant across all 42 runs. Only three things vary: the
**representation**, the **length bound**, and the **start** (see `RUNS.md`).

Emitted by `c0/sweep_configs.py`; the per-run commands come from the same module.

---

## 1. Equilibrium

| | |
|---|---|
| case | `precise_QA` (`desc.examples.get`) |
| boundary | **fixed** — never solved, only its surface is used |
| `R0` | 1.030670011 m |
| `a` | 0.171778344 m |
| aspect ratio | 6.000 |
| NFP | 2 |
| stellarator symmetry | yes |
| spectral resolution | L = M = N = 8 |
| `Psi` | 0.087 Wb |

## 2. Coil staging

| | |
|---|---|
| unique coils `nc` | 4 → **16 total** under NFP × symmetry |
| `r/a` | **3.30** (all representations except `arcB2`, which is **3.60**) |
| coil radius | 0.5669 m at r/a = 3.30 |
| start | **built, never loaded** — `initialize_modular_coils` → `from_values` |
| fit method | **`chord`** |
| currents | pinned by `FixSumCoilCurrent`, a **linear** constraint held during the solve |

Every representation cold-starts at the exact circle's circumference,
**3.5617 m = 0.691 ×** the paper length bound. That sets the floor on how tight the
length axis can go.

`FixSumCoilCurrent` is not optional: `QuadraticFlux(vacuum=True)` is homogeneous of
degree 2 in the currents, so `I → 0` is a free minimiser of the objective.

The `arcB2` exception is forced — see §9.

## 3. What is optimized

**Free variables:** the coil geometry and the coil currents. The equilibrium is **fixed**
(`eq_fixed=True` throughout) and is never a variable.

| rep | shape DOF/coil | `dim_x` | source grid |
|---|---|---|---|
| planarN7 | 21 | 136 | uniform, 801 nodes/coil |
| arcB2 | 14 | 132 | composite GL, 48 nodes/coil |
| arcB3 | 21 | 172 | composite GL, 72 nodes/coil |
| arcB5 | 35 | 252 | composite GL, 120 nodes/coil |
| arcB7 | 49 | 332 | composite GL, 168 nodes/coil |
| xyzN4 | 27 | 160 | uniform, 801 nodes/coil |
| xyzN6 | 39 | 208 | uniform, 801 nodes/coil |

`dim_x` is the **full state vector** and is larger than `4 × shape DOF` because each coil
also carries its frame and current:

```
dim_x = 4 × (shape DOF + rotmat 9 + shift 3 + current 1)      [+ arc_ref 3B for arcs]
```

Shape DOF is `2N+7` for FourierPlanar, `3(2N+1)` for FourierXYZ, `B(4+M)` for the arcs.

## 4. Objective

**One term.** `ObjectiveFunction((QuadraticFlux,))`.

| | |
|---|---|
| class | `QuadraticFlux` |
| `dim_f` | **1326** rows (= plasma eval grid nodes) |
| `eval_grid` | `LinearGrid(M=25, N=25, NFP=2, sym=True)` |
| `field_grid` | the source grid (per rep, above) |
| `vacuum` | `True` |
| `bs_chunk_size` | 10 |

A **sum of squares over the plasma grid**, so it minimises the *mean* normal field error.
**Nothing in it targets the peak**, and a refinement given extra freedom will spend it on
the mean and let the peak get worse. Report both metrics.

`vacuum=True` is why the current constraint in §5 is not optional: the objective is then
homogeneous of degree 2 in the currents, so `I → 0` is a free minimiser.

## 5. Constraints, as passed to the optimizer

Six, in this order. Five are **nonlinear** and go into the augmented Lagrangian; the sixth
is **linear** and is split out by `Optimizer._parse_constraints` and projected out of the
problem, so it is satisfied exactly at every iterate and never enters the AL.

| # | class | type | `dim_f` | bound | grid | `normalize_target` | weight |
|---|---|---|---|---|---|---|---|
| 1 | `CoilLength` | nonlinear | 4 | `(0, L_max)` | source grid | True | 1.0 |
| 2 | `CoilCurvature` | nonlinear | **1204** | `(0, κ_max)` | `LinearGrid(N=150)` | True | 1.0 |
| 3 | `CoilMeanSquaredCurvature` | nonlinear | 4 | `(0, κ_MS_max)` | `LinearGrid(N=150)` | True | 1.0 |
| 4 | `CoilSetMinDistance` | nonlinear | 4 | `(d_cc_min, ∞)` | `LinearGrid(N=150)` | True | 1.0 |
| 5 | `PlasmaCoilSetMinDistance` | nonlinear | 8 | `(d_pc_min, ∞)` | see §7 | True | 1.0 |
| 6 | `FixSumCoilCurrent` | **linear** | 1 | — | — | — | — |
| — | ~~`CoilSetLinkingNumber`~~ | — | — | — | — | — | **dropped** |

`dim_f` is identical for all seven representations. Row counts: one per unique coil for
1/3/4, one per coil per curvature node (4 × 301) for 2.

**Everything is a BOUND, never a target.** A bounds row that is satisfied contributes
exactly zero residual *and* zero Jacobian rows, so an inactive constraint is invisible to
the linearization rather than pulling on it.

`normalize_target=True` on all five nonlinear constraints, so no bound re-anchors on the
start — the numbers in §6 are absolute, in physical units.

`weight=1.0` on every constraint (the `sqrt(mu0)` weighting applies only to the legacy
penalty mode, not to the augmented Lagrangian).

### 5.1 Why the current constraint is there, and why it is linear

`FixSumCoilCurrent` pins the **sum** of the coil currents; the individual currents stay
free. Without it, `QuadraticFlux(vacuum=True)` is minimised by `I → 0`. Being linear it is
projected out rather than penalised, so it costs the AL nothing and holds exactly.

### 5.2 The linking number is dropped, and `d_cc` does not replace it

Its Gauss integral is only spectral for a C1 curve; on a C0 arc with close coils it is
unreadable — measured Jacobian entries of **2.9e+12** while its own residual is ~0, twelve
orders above every other block, which destroys the Gauss-Newton model.

Separation and topology are **independent**: a coilset can sit at `d_cc` margin 1.029 with
**six linked pairs**. Nothing in this constraint set constrains topology. It is guarded
instead by `--dcc-signed` on warm starts, and checked on every result from the **pairwise
matrix** (§10).

### 5.3 The two curvature constraints do not track each other

`CoilCurvature` returns one row **per grid node per coil**, each clamped independently — a
minimum-bend-**radius** (winding) limit set by the single tightest point.
`CoilMeanSquaredCurvature` returns **one row per coil** — a total-bending (stress) limit.
They bind different coils. Both are enforced.

`--curv-ms-max` is the **enable switch**, not merely the bound: omit it and
`CoilMeanSquaredCurvature` is silently absent from the constraint list. Every command
carries it.

### 5.4 Fixed method flags

| flag | value | why |
|---|---|---|
| `--distance-method` | `segment` | `point` samples grid nodes only and manufactures a barrier |
| `--dcc-pair-mode` | `per_coil` | one row per coil = min over all others |
| `--dcc-neighbors` | all | no neighbour restriction |
| `--dcc-signed` | **warm starts only** | negates linked pairs so a crossing is recoverable |
| softmin | **off** | both distance objectives use the exact minimum |
| `--feas-margin` | **0** | scales every bound at once; use per-bound overrides |
| `--dcc-floor` | **off** | livelocks |
| `use_softmin` | `False` | — |
| `dist_chunk_size` | 2 | both distance constraints |

## 6. Bounds

All in **physical units**, with `normalize_target=True`.

| constraint | raw (R0 = 1 m) | rescale | enforced at our R0 | units |
|---|---|---|---|---|
| coil length | 5.0 | `× f` | **5.153350** | m |
| max curvature | 12.0 | `/ f` | **11.642912** | 1/m |
| mean-square curvature | 6.0 | `/ f²` | **5.648225** | 1/m² |
| coil–coil distance | 0.083 | `× f` | **0.085546** | m |
| plasma–coil distance | 0.166 | `× f` | **0.171091** | m |
| linking number | 0 | — | **dropped** | — |

`f = R0 / 1.0 = 1.030670`. **The rescale exponent follows the units.** `kappa_ms` is
`(1/L)∫κ²dl` with no square root, so it is 1/m² and takes `f²`; a single power leaves the
bound 3.07% too loose (5.821456).

Physical scale: κ ≤ 11.642912 m⁻¹ is an **8.59 cm minimum bend radius**, about half the
plasma minor radius. `kappa_MS` ≤ 5.648225 m⁻² is a **42.1 cm RMS bend radius**.
`d_cc` ≥ 0.085546 m = **0.498 a**.

**The length bound is the swept axis** and is the one entry above that changes between
runs — 0.8/1.0/1.2/1.4/1.6 × 5.153350 m. See `RUNS.md`.

### 6.1 Enforced vs adjudicated

The solver is told a **backed-off** bound; the result is judged against the **paper**
bound. Backoffs are an implementation detail and are never reported.

| family | `d_cc` | `kappa` | `kappa_MS` |
|---|---|---|---|
| smooth (planarN7, xyzN4, xyzN6) | paper | paper | paper (5.648225) |
| arc (B=2/3/5/7) | paper + sagitta, per point | ×0.995 (11.584697) | ×0.99 (5.591743) |

The smooth representations take no backoff: their measured enforcement-grid error is
0.01–0.23%, ~25× smaller than the sagitta rule prescribes. The arc `d_cc` backoff depends
on the length bound and so varies per point:

| ×L | `--dcc-min` | backoff |
|---|---|---|
| 0.8 | 0.087085 | +1.80% |
| 1.0 | 0.087951 | +2.81% |
| 1.2 | 0.089009 | +4.05% |
| 1.4 | 0.090259 | +5.51% |
| 1.6 | 0.091702 | +7.20% |

## 7. Grids

| grid | value | used for |
|---|---|---|
| plasma eval | `LinearGrid(M=25, N=25, NFP, sym)` | `QuadraticFlux` |
| Biot–Savart source, arcs | composite GL, **24 nodes/arc** | flux, length |
| Biot–Savart source, smooth | `LinearGrid(N=400)` | flux, length |
| curvature | `LinearGrid(N=150)`, **uniform** | `CoilCurvature`, `CoilMeanSquaredCurvature` |
| coil–coil distance | `LinearGrid(N=150)` | `CoilSetMinDistance` |
| plasma surface for `d_pc` | `LinearGrid(M=64, N=64, NFP, sym)` | `PlasmaCoilSetMinDistance` |
| adjudication | `N = 400` | all geometry, post-solve |
| linking matrix | `LinearGrid(N=100)` | post-solve topology check |

Three of these are load-bearing rather than arbitrary:

**The source grid is composite-GL for arcs.** The arc classes are C0 and the quadrature
path reads `ds = grid.spacing[:,2]`, so GL weights make DESC's own sum a composite rule
across the corners. It is never `None`: DESC would default to `LinearGrid(N=2·coil.N+5)`,
which would move resolution with the swept axis.

**Curvature uses a uniform grid, not the source grid.** `CoilCurvature` is the one coil
objective that does not reset `quad_weights` to 1, so on a composite-GL grid its rows
would inherit weights spanning 10×. Those cancel against `bounds_scaled`, so the feasible
set is unaffected — but a pointwise bound should not carry an arbitrary 10× spread across
its rows.

**The `d_pc` plasma grid is separate and finer** (M=64 against the objective's M=25). A
min-distance is a pointwise extremum, not an integral, so it converges far more slowly
than the flux quadrature: `d_pc` reads 0.195697 at M=25 against 0.191575 at M=128, i.e.
M=25 over-reads clearance by 2.2%.

## 8. Augmented Lagrangian

`Optimizer("lsq-auglag")`, least-squares augmented Lagrangian.

| parameter | value | note |
|---|---|---|
| `maxiter` | **800** | 500 leaves relaxed-length runs still descending |
| `max_inner_iter` | **100** | **not a tuning knob** — fixes must hold at any value |
| `ctol` | **1e-4** | the one non-default tolerance; see below |
| `gtol` | 1e-8 | floor on `gtolk` |
| `ftol` | 1e-6 | |
| `xtol` | 1e-6 | |
| `max_trust_radius` | **0.5** | caps one oversized step; does **not** guard topology |
| `tr_method` | `svd` | |
| `x_scale` | `auto` (Jacobian column norm, ratcheting) | DESC default, kept |
| initial penalty `mu` | 10 | DESC default |
| initial multipliers `y` | 0 | DESC default |
| `tau` | 10 | DESC default, unchanged |
| `max_inner_stalls` | 50 | |
| `max_penalty` | 1e8 | DESC default |
| `alpha_omega` / `alpha_eta` / `beta_eta` | 1.0 / 0.1 / 0.9 | DESC defaults |
| `--check-every` | 50 | checkpoint + KKT diagnostic; never stops the solve |

**Why `ctol = 1e-4` and not the 1e-6 default.** Whenever a subproblem is feasible the AL
tightens its own working tolerance, `ctolk = max(ctolk / mean(mu)^0.9, ctol)`. Once
`ctolk` falls under the constraint's numerical noise floor, every outer update reads noise
as violation and multiplies `mu` by `tau`, and **`mu` never decreases**. Measured: `mu`
ratcheted 5731 → 3.5e+04 → **1.9e+05** on violations of 2e-06 to 2e-05, with `sqrt(mu)`
then scaling the constraint block of the Jacobian by ~435 and collapsing the trust region.
The identical run at `ctol = 1e-4` froze `mu` at **29.78**.

Two consequences to carry:

* `verify` adjudicates independently at `FEAS_RTOL = 1e-8`, so a looser `ctol` **cannot**
  make an infeasible coilset pass.
* The solver reports `precision` (SUCCESS) rather than `stall` for violations in
  1e-6..1e-4, so **its own verdict is less trustworthy**. Always adjudicate with
  `adjudicate.py`.

**`mu` starts at 10 with `y = 0` on every run, including warm starts.** The h5 carries no
multiplier state, so each rung restarts its penalty schedule from scratch.

## 9. The one staging exception: `arcB2`

`arcB2` runs at **r/a = 3.60** and carries **`--arc-bulge 0`**; every other representation
is at r/a = 3.30 with the fitted start.

`from_values` fits each arc as a transverse graph over its **chord**, and at B=2 that
chord is a **diameter** whose target profile is a semicircle — a `√` with vertical
endpoint tangents that a truncated sine series cannot hold. The fit ripples, and since κ
is a second derivative the ripple dominates it: the fitted B=2 start exceeds `kappa_MS` by
**40.4%**, and *more* modes make it worse (7.93 at M=3, 27.14 at M=10). `--arc-bulge 0`
keeps the fitted fundamental and drops the ripple modes, halving `kappa_MS` — but the
ripple was also holding the plasma off, so `d_pc` then needs a larger r/a to recover.

**Consequence:** a B=2-vs-B=3 comparison mixes two changes, facet count and coil radius.
The bottom rung of the facet ladder is not a clean comparison with the rest of it.

## 10. Adjudication

Runs **after** each solve, never alongside one — concurrent `verify` has killed runs on a
15 GB machine.

| | |
|---|---|
| grid | `N = 400`, plus `N = 800` convergence check at ×1.4 and ×1.6 |
| feasibility tolerance | `FEAS_RTOL = 1e-8` |
| bounds judged against | the **paper** values above, never the enforced ones |
| length judged against | the **swept** bound |
| topology | pairwise linking **matrix**, `LinearGrid(N=100)` |
| flux metrics | `verify.bn_stats` on the solve's own source grid |

Backoffs are an implementation detail and are **never reported**. A run is told a tighter
bound and judged against the paper one.

`d_cc` is **not** grid-converged at N=400 on the most contorted coilsets — still moving
1.2% at N=800. Where the convergence check reports drift, `d_cc` should not be quoted.

## 11. What is not standard DESC

Everything below is on branch `c0-testing` and **not in DESC master**. Reproduce the list
with `git diff master -- desc/`.

**A stock DESC install cannot run this sweep at all**: the arc representation and the
mean-square curvature constraint do not exist there.

### 11.1 Classes that do not exist in master

| class | file | used for |
|---|---|---|
| `PiecewisePlanarArcCurve` / `PiecewisePlanarArcCoil` | `geometry/curve.py`, `coils.py` | the entire arc family (`arcB2/3/5/7`) |
| `CoilMeanSquaredCurvature` | `objectives/_coils.py` | the `kappa_MS` constraint |
| `_FrozenArcReferenceMixin`, `FixCurveArcReference` | `geometry/curve.py`, `objectives/` | the frozen arc gauge (§11.4) |
| `PolarPlanarArcCurve` / `PolarPlanarArcCoil` | same | **not used** — polar arcs are excluded |

### 11.2 An existing class whose MEANING changed — read this one

**`CoilCurvature` targets unsigned `|curvature|` here; master targets SIGNED
`curvature`.** (commit `ddfcb1184`)

In master, positive curvature means "convex" and negative means "concave", so a bound of
`(0, κ_max)` there also forbids every concave stretch — a different constraint from the
one this campaign enforces. Here the bound is on the magnitude, i.e. a genuine
minimum-bend-radius limit.

This is the single most misreadable difference in the stack: the class name, the
constructor call and the bound all look identical to master's, and only the meaning moved.
**Any comparison against a result computed with stock DESC's `CoilCurvature` is comparing
two different constraints.**

The motivation is also a defect in the signed quantity: its sign flips wherever the
curvature vanishes, so as a residual it is discontinuous — it jumps by twice the full
magnitude at points that move as the curve moves.

### 11.3 New keyword arguments on existing classes

| class | kwarg | value used | in master? |
|---|---|---|---|
| `CoilSetMinDistance` | `distance_method` | `"segment"` | **no** |
| `CoilSetMinDistance` | `pair_mode` | `"per_coil"` | **no** |
| `CoilSetMinDistance` | `signed` | `True` on warm starts | **no** |
| `CoilSetMinDistance` | `num_neighbors` | `None` | yes |
| `PiecewisePlanarArcCoil.from_values` | `fit_method` | `"chord"` | **no** (class is new) |
| `_CoilObjective.compute` | `override_grid` | `False` | **no** |
| `CoilSet._compute_linking_number` | `indices` | — | **no**, and **not used** here |

Three of these are load-bearing:

**`distance_method="segment"`** minimises over the line *segments* between grid points
rather than over the points themselves. Master has only the point behaviour, which cannot
see a crossing that falls between nodes — it reported 6.4 mm of clearance for coils that
actually intersect.

**`signed=True`** negates the distance of *linked* pairs, so a linked pair goes negative,
wins the min, and reads as a violation that grows as the coils separate on the wrong side.
Without it the unsigned minimum is blind after a crossing: it *recovers* once a different
pair owns the minimum, so the constraint reports the geometry as improving while the
coilset is linked. The sign is `stop_gradient`'d, which is exact rather than an
approximation.

**`override_grid=False`** stops `Curve.compute` silently recomputing `length` — the only
0-d key any coil objective requests — on its own `LinearGrid(N=2N+5)`. Without it the
length constraint stops measuring length on the grid it was given.

### 11.4 Bug fixes on this branch that change results

| commit | fix | why it matters here |
|---|---|---|
| `6691e5a14` | NaN gradient of `|curvature|` where a curve is momentarily straight | **essential** — these arcs have κ = 0 *exactly* at every hinge by construction, so the branch hits this on every arc run |
| `31e5b7f35` | freeze the planar-arc frame reference | removes a step discontinuity in the arc parameterization |
| `a61689fb1` | linking-number **writhe** bug | master's `CoilSetLinkingNumber` summed the full matrix *including the diagonal* — each coil's Gauss integral with itself, i.e. its writhe, which is nonzero for any non-planar curve |
| `8a6bb8346` | deduplicate symmetry-redundant residuals in coil objectives | changes `dim_f` of the coil blocks (§5) |
| `121c1017d` | AL termination, `eta` default, violation reporting | changes when the solve stops |
| `bfae0eea4` | AL outer loop given a path forward from an infeasible point | reachable on any warm start |

### 11.5 Optimizer and augmented Lagrangian

`lsq-auglag` is heavily modified (~585 changed lines). Options **not in master**:

| option | value used | note |
|---|---|---|
| `max_inner_iter` | 100 | cap on inner iterations per subproblem |
| `max_inner_stalls` | 50 | |
| `track_outer` | `True` | per-outer-iteration diagnostics |
| `y_update_gate` | `residual` | |
| `jac_scale_ratchet` | default (on) | flag exists; **not** exercised — the default reproduces master |
| `step_veto` | **off** | `--dcc-floor`; livelocks |
| `min_window_progress` | **off** | `--mu-gate` |

Also branch-only:

* **`callback` is reachable.** Master documents a `callback(xk, *args) -> bool` that ends
  the solve gracefully, but `_desc_wrappers` never forwarded one, so it always defaulted
  to `lambda *args: False`. This is what `--check-every` uses for checkpointing.
* **Two new termination statuses**, `"stall"` and `"precision"`, distinguishing "stalled
  and not a KKT point" from "stalled at a feasible point with `ctol` satisfied". This is
  why §8 warns that the solver's own verdict is not trustworthy on its own.

`tau`, `ctol`, `alpha_eta`, `beta_eta`, `max_penalty` and the initial `mu`/`y` are all
master behaviour, unchanged.

### 11.6 Local tooling (not DESC at all)

Everything under `c0/` is campaign tooling, not library code: `sweep_configs.py` (the
grid), `run_sweep.py` (the runner), `diag_ymask.py` (the driver), `verify.py` and
`adjudicate.py` (adjudication), `bench/bench.py` (the rescaled bounds),
`dump_initial_starts.py` (the starts). None of it is proposed for DESC.

## 12. Reproducibility

* The start is **built, not loaded**, and is fully determined by
  `(rep, nc, r_over_a, B, M, fit_method, fp_N, xyz_N, arc_bulge)`. No coilset file is
  needed to reproduce any run.
* `initial starts/starts.json` pins each start by SHA-256 of its parameter vector;
  `python c0/dump_initial_starts.py "c0/sweep_910/initial starts" --check` rebuilds and
  compares, so a silent change in `initialize_modular_coils` or `from_values` is caught.
* Every point is **one seed**. `--seed` and a smooth-representation perturbation do not
  exist yet, so multiseed cannot currently be configured.
