# Stage-1 clamshell optimization: full specification

Written 2026-09-22. Read with `NOTES.md` §5d/5e. **Status: specified, not yet run.**
§4 below is an adversarial pass over §§1-3 and it changed three things, marked **[ADV-n]**.

## 0. Equilibrium and provenance

`precise_QA` from `vendor/desc/examples/`. Verified: `p = 0`, `current = 0` (both
PowerSeriesProfile, all coefficients zero), `eq.iota is None` (current-constrained), so it
is a **true vacuum, zero-net-current** equilibrium -- `I/G = -2e-17` and stream-function
residual `1.9e-5`, vs `1.8e-2` / `1.15e-3` on Helios. L=M=N=8, NFP=2, sym=True, Psi=0.087,
R0=1.0307 m, a=0.17178 m, A=6.0, iota ~ 0.4199 (flat).

Three arms, identical in every respect except the clamshell objective:

| arm | clamshell objective | purpose |
|---|---|---|
| `lo` | minimise C | the treatment |
| `mid` | C fixed at baseline (`target=C0`) | **[ADV-1]** the provenance control |
| `hi` | maximise C | the dose-response arm |

## 1. Things (optimizable objects)

1. `eq` -- the Equilibrium.
2. `curves[k]`, k = 0..K-1, `FourierRZSurfaceCoil(equilibrium=eq, secular_theta=1,
   secular_zeta=0, NFP=1)`, `zeta_n` over modes -Ns..Ns with Ns = 12, `theta_n = [0.0]`
   on mode [0] so that `theta(s) = s` exactly.
   - **K = 8 over one field period**, convergence-checked against K = 16. Stellarator
     symmetry makes the foliation field-period symmetric, so one period suffices.
   - Current is irrelevant here (the curves are proxies, never used for Biot-Savart), so
     `FixCoilCurrent` on all of them.

## 2. Objective

```
f  =  sigma * ( 1/K ) * sum_k  C_k ,      sigma = +1 (lo) / -1 (hi)
C_k = eps_2(curve_k) / ( eps_1(curve_k) / 4 )
```

- `eps_2` uses a **softmin over P frozen hinge candidates**, not the DP, and not a hard min
  (§5e: 8 well-separated local minima within 5% of the best, so a hard min chatters).
  P = 190 from a stride-12 grid on n = 240 nodes; ~5% of hinge space is within 10% of
  optimal, so this misses by <1%. The exact DP is far too slow for the inner loop --
  **measured 3.9 s for one C evaluation over K = 8 curves** -- which is what forces the
  frozen-candidate softmin in the first place.
- Candidates and the frozen plane normals are refreshed from the exact DP **only at
  augmented-Lagrangian outer-loop boundaries**, never mid-inner-solve, and the objective
  jump at each refresh is logged. **[ADV-2]**
- Softmin temperature annealed; the reported endpoint value is recomputed with the exact DP.
- `normalize=False` (C is dimensionless and O(1)); `weight=1`.
- Range is bounded: `eps_2 <= eps_1` always (segmentations are nested), so `C in [0, 4]`,
  and `C = 4` means the second plane buys nothing. The `hi` arm therefore cannot run away.

## 3. Constraints

### 3a. Equilibrium, hard (equality)

| constraint | form | value |
|---|---|---|
| `ForceBalance(eq)` | equality, handled by the augmented Lagrangian directly -- **NOT proximal**, see below | 0 |
| `FixPressure(eq)` | equality | keeps p = 0 (vacuum) |
| `FixCurrent(eq)` | equality | keeps I = 0 (so I/G = 0 and contours close poloidally) |
| `FixPsi(eq)` | equality | 0.087 Wb |
| `FixBoundaryR/Z(eq, modes=...)` | equality | every boundary mode with \|m\|>3 or \|n\|>3, **plus R(0,0)** |

Free boundary DOF: R_lmn, Z_lmn with |m| <= 3, |n| <= 3, except R(0,0). Fixing R(0,0)
holds R0; AspectRatio below then holds `a`, which matters because the stage-2 coil bounds
scale with `a`.

### 3b. Equilibrium, physics (anti-trivialization)

An axisymmetric torus has `eps_1 = eps_2 = 0` and perfect planar coils, so it is the
trivial optimum. These are what block the slide:

| constraint | objective | target / bounds |
|---|---|---|
| aspect ratio | `AspectRatio(eq)` | `target = 6.0` |
| rotational transform | `GenericObjective("iota", eq, grid=LinearGrid(rho=[0,0.5,1], M=8, N=8, NFP=2))` | `target = iota_0` per node (~0.4199). **No `RotationalTransform` objective exists; `FixIota` does not apply to a current-constrained equilibrium.** A vacuum stellarator with iota != 0 and no net current MUST be 3D, so this is the constraint that really does the work. |
| quasisymmetry | `QuasisymmetryTwoTerm(eq, helicity=(1,0), grid=...)` | `bounds = (0, f_QS0)` -- no worse than baseline |
| overall non-planarity | `eps_1` of every curve | `target = eps_1,0(k)` **[ADV-3]** -- see §4 |

### 3c. Curves, hard (equality)

| constraint | purpose |
|---|---|
| `SurfaceCurveConsistency(eq, curves[k])` | ties each curve's copy of R_lmn/Z_lmn to the equilibrium's |
| `FixParameters(curves[k], {"rotmat": True, "shift": True})` | the PR notes these are optimizable but meaningless for a surface curve |
| `FixParameters(curves[k], {"theta_n": True})` | keeps `theta(s) = s` |
| `FixCoilCurrent(curves[k])` | currents play no role |
| **streamline**: `(x_s . B) / (\|x_s\| \|B\|) = 0` at every node | **a hard equality, NOT a weighted objective** -- see §4 |
| **pin**: `zeta_k(s=0) = 2 pi k / (K * NFP)` | selects WHICH streamline; without it all K curves collapse onto one |

The streamline residual is the cosine of the angle between the curve tangent and B.
`x_s || K  <=>  x_s . B = 0` because `K = n x B / mu0` and `x_s` is tangent to the surface,
so no cross product and no vanishing denominator (`|K| = |B|/mu0`).

`zeta_k(0) = sum of the cos-mode coefficients`, so the pin is LINEAR in `zeta_n` and goes in
as a `LinearObjectiveFromUser`.

### 3d. Not imposed, deliberately

- **`|K| d = const` (equal-current spacing).** Measured worth 0.01-0.04% on the aggregate
  (NOTES §5e), because mean |K| along a contour varies by only 1.4-2.6%. The equally spaced
  `zeta_k(0)` pins do the anti-collapse job for free.
- Coil-coil / coil-plasma clearance, length, curvature: these are proxy curves on rho=1,
  not coils. They belong to stage 2.

## 4. ADVERSARIAL PASS

### [ADV-1] The baseline had no provenance control. FIXED.

As written, `mid` was "precise_QA, untouched". But `lo` and `hi` go through a boundary
re-solve, a proximal projection and N iterations of an augmented Lagrangian, and `mid` did
not. Any difference in the stage-2 gap could then be an artifact of having been optimized
at all -- resolution, re-solve, or the QS/iota constraints shifting the boundary.

**Fix:** `mid` runs the identical pipeline with the clamshell objective as an EQUALITY at
its own baseline value (`target = C0`) instead of a min or max. All three arms then have
the same provenance and the same iteration budget, and differ only in what the clamshell
term asks for.

### [ADV-2] The streamline condition MUST be a hard constraint. This was the real hole.

If `x_s . B = 0` is a weighted objective sitting in the same `ObjectiveFunction` as C, then
the optimizer can lower C by **moving the curve off the streamline** instead of by changing
the equilibrium. The curves have ~25 free parameters each and C depends on them directly,
so this is not a remote risk -- it is the cheapest available descent direction, and it
would make the entire experiment meaningless while looking like a success.

**Fix:** the streamline residual and the pin go in as augmented-Lagrangian **equality
constraints**, never as weighted objective terms. Verification gate: the final
`max |cos(x_s, B)|` must be below 1e-3 in all three arms, and it is reported per arm. If
an arm cannot satisfy it, that arm is void.

Consequence: the curve ansatz must be able to represent the streamline. `x_s . B = 0` at
n = 240 nodes against 2*Ns+1 = 25 zeta DOF is heavily overdetermined, so the curve is a
least-squares streamline, not an exact one. **Gate:** check the baseline residual at and raise Ns until it is below 1e-3 before running anything.

### [ADV-3] C is scale-free but 0/0-unstable, and that is a confound as well as a guard.

I had claimed C "cannot be gamed by flattening toward axisymmetry". That is wrong in two
ways. (i) `C = 4 eps_2 / eps_1` is ill-conditioned as `eps_1 -> 0`; the iota constraint
blocks FULL axisymmetry but not partial flattening, and C's gradient blows up. (ii) More
importantly, two arms that differ in BOTH `eps_1` and C are not a clean dose-response --
the stage-2 gap could move because the coils got more planar overall, not more clamshell-able.

**Fix:** constrain `eps_1` of each curve to its own baseline value (`target = eps_1,0(k)`,
not merely a floor), so the three arms differ *only* in the two-plane preference at fixed
overall non-planarity. This is both the numerical guard and the experimental control.

### Checks that did NOT change anything, recorded so they are not re-litigated

- **Do the proxy contours close poloidally?** They must, or `secular_zeta = 0` is the wrong
  ansatz. For vacuum `I = 0`, so contours of `Phi` close, and `Phi_mod = Phi`. Confirmed
  numerically: `I/G = -2e-17` and `modular_contours` runs (it refuses if `Phi_mod` is not
  monotone in zeta).
- **Is the `hi` arm unbounded?** No: `eps_2 <= eps_1` by nesting, so `C <= 4`.
- **Does `FixIota` work here?** No -- `eq.iota is None`. Hence `GenericObjective("iota")`.
- **Weight/normalization of C.** Dimensionless and O(1); `normalize=False`.
- **Does re-freezing break the AL?** Possibly, hence the outer-loop-only refresh and the
  logged jump. If a refresh moves the objective by more than a few percent, the candidate
  set was too coarse and P must rise.

### Open risks I could not design away

1. **Does C move at all under these constraints?** Unknown. This is the gate: run stage 1
   for all three arms first and report `C0 -> C_lo / C_mid / C_hi`. If the spread is small
   compared with the natural 0.87-1.37 spread across a single foliation (NOTES §5e), there
   is nothing to measure and we stop before spending GPU.
2. **K = 8 curves may not resolve the aggregate.** C is skewed across the foliation (mean
   0.996 vs median 0.970 at 24 contours on QA). Convergence-check in K, do not assume.
3. **The frozen-normal surrogate is exact only at the freeze point.** Danskin gives the
   right gradient there; away from it the surrogate is an upper bound on eps. The
   outer-loop refresh bounds the drift but does not eliminate it.
4. **`eps_1` held fixed may over-constrain.** With aspect ratio, iota at 3 radii, QS, and
   `eps_1` for K curves all pinned, the boundary may have too little freedom left to move C.
   If the gate shows C frozen, relax `eps_1` from an equality to a +-20% band and re-run
   the gate before abandoning.

## 4b. ARCHITECTURE: single-stage augmented Lagrangian, no proximal projection

Checked against the vendored DESC rather than assumed:

- There is **no registered `proximal-*` optimizer**. `proximal-` is a METHOD PREFIX parsed
  by `_parse_method`; `ProximalProjection` exists as a class and `Optimizer` auto-wraps with
  it only when the chosen method does not support nonlinear constraints.
- `lsq-auglag` reports `equality_constraints=True, inequality_constraints=True`, so the
  auto-wrap never fires for it.
- And using the `proximal-lsq-auglag` prefix explicitly would be WRONG here:
  `_maybe_wrap_nonlinear_constraints` absorbs **all** nonlinear constraints into
  `ProximalProjection` and sets `nonlinear_constraints = ()`. Our QS bound, iota target,
  streamline equality and eps_1 targets would be swallowed into the equilibrium projection,
  which is not what it is for.
- The user's own PR #2298, "Getting proximal to work with augmented Lagrangian optimizers",
  is OPEN and is NOT vendored -- consistent with the above.

**Decision: `method="lsq-auglag"` with `ForceBalance(eq)` as one AL equality constraint
among the rest.** This is single-stage: the equilibrium is in force balance only at
convergence, so intermediate C values are meaningless and only the converged point counts.
It is also the code path this sandbox has already hardened (`second_order="constraints"`,
NOTES/CLAUDE.md).

Consequence for [ADV-2]: the streamline condition is satisfied only asymptotically, so the
curves DO drift off the streamlines during the solve and C is partly "gamed" in flight. The
final gate `max |cos(x_s, B)| < 1e-3` is therefore load-bearing, not a formality.

## 5. Stage 2, for later (fixed for all three arms)

Identical budget, identical bounds, identical start procedure. Per arm: arcB2 and
FourierXYZ, same coil count, `paper_bounds` scaled by `a` (which is held fixed, so the
three arms get literally the same numbers). Cold start from circles at a common r/a for
every arm -- no warm starts, no shared lineage, since the handoff measured a warm planar
start as worth 24% of final bn. Response variable: the ratio `bn_arcB2 / bn_FourierXYZ`.
