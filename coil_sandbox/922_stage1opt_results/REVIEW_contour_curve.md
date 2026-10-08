# Adversarial review: `contour_curve.py` (the in-objective contour clamshell metric)

2026-09-22. Reviewed after implementation, as agreed. `contour_curve.py` builds the proxy
coils INSIDE the objective as contours of the current potential, so they are exact
streamlines and equal-current by construction -- which deletes SPEC [ADV-2] (the optimizer
cannot move a curve off its streamline) rather than constraining it.

## A. Verified correct

| check | result |
|---|---|
| `_fourier_eval` vs DESC's own `surface.compute("R"/"Z")` at 64 random (theta, zeta) | **2.2e-16** on precise_QA, **4.4e-16** on NCSX |
| `_fourier_eval_dtheta` / `_dzeta` vs finite differences of DESC's own R | 4.1e-10 / 4.7e-10 |
| whole chain vs the independent `phi_contours` + `clamshell` path | C to 4 decimals, eps_1 identical, curves agree to **2e-6 m** |
| full-chain Jacobian (eq params -> C) vs finite differences | **1e-9** on three different boundary modes |
| Newton convergence | machine precision in **3** iterations; residual 1.7e-16 |
| contours distinct and correctly ordered | min separation 0.72-1.88 m, matching the expected 2 pi R / (K NFP) toroidal spacing |
| monotonicity of Phi_mod in zeta | dPhi/dzeta spans 6.1e5 .. 1.1e6, never crosses zero |

## B. Bugs found and fixed during implementation

1. **`M`, `N` in `_constants` get traced by jit**, so `reshape(M, N)` fails on a traced
   shape. Exactly the trap CLAUDE.md records for the spline `intervals` fix. Shape-defining
   ints now live on `self` and are listed in `_static_attrs`.
2. **Forward-mode Jacobian tried to allocate 207 GB.** The dense `(n, M, N)` intermediate in
   the Phi~ evaluation, times 1462 equilibrium DOF of tangents. Fixed two ways: truncate
   Phi~ to |m| <= Mt, |n| <= Nt, and default `deriv_mode="rev"` (dim_f = K is tiny, dim_x is
   1462, so reverse mode is obviously right). **Jacobian 4.0 s -> 0.15 s.**
3. **`truncate` did numpy indexing on traced arrays** -> `TracerArrayConversionError`. The
   mode selection depends only on (M, N, Mt, Nt), so it is now computed once at build.
4. **No monotonicity check.** `phi_contours.modular_contours` refuses to run when Phi_mod is
   not monotone in zeta, but the objective had no such guard -- the optimizer could walk the
   boundary into a regime where Newton lands on a different branch and C silently becomes
   garbage. Now an `errorif` at build, with the span reported.
5. **No Newton convergence check.** Fixed iteration count with no residual test. Added
   `contour_residual` and an `errorif` at 1e-10.

## C. Findings that change the run configuration

### C1. K = 4 contours is badly under-resolved, and `phase` is not innocuous

| phase | C mean | C range |
|---|---|---|
| 0.00 | 1.0453 | 0.8650 - 1.3720 |
| 0.25 | 0.9791 | 0.8744 - 1.1025 |
| 0.50 | 0.9713 | 0.9120 - 1.0296 |

**A 7% swing in the mean from the contour ladder's phase alone.** NOTES section 3.2 argued
the foliation average is phase-independent -- true in the limit, but a K = 4 SAMPLE of a
distribution that spans 0.87-1.37 (section 5e, 24 contours) is not. Phase 0 happens to pick
up the 1.372 outlier.

**Action: raise K to 16-24 per field period and convergence-check the aggregate in K.** At
0.15 s per Jacobian for K = 4 this is affordable. Do NOT report a C change smaller than the
phase sensitivity until K is resolved.

### C2. The (M, N) potential grid can be much coarser than assumed

C is **identical to 5 decimals** at (M,N) = (32,64), (48,96), (64,128) and (96,192). The
default of (64,128) is 4x more work than needed. Use (32,64) or (48,96).

### C3. Node count has a ~0.3% noise floor

C at n = 120/180/240/360/480 is 1.37011 / 1.37155 / 1.37199 / 1.37227 / 1.37244 for one
contour, and NON-monotone for others (0.96889 / 0.97050 / 0.97313 / 0.97066 / 0.96997) --
the DP breakpoints shift with n. **So C carries a ~0.3% resolution noise floor at n = 240.**
The v0 run measured a -1.54% change; that is above the floor but only by ~5x. Use n = 480
for any reported number, or quote C to 3 significant figures.

### C4. Force-balance contamination is now REAL

1440 of 1462 Jacobian columns are nonzero: C depends on the whole interior solution through
`K_vc = n x B / mu0`, not just the boundary. The v0 reprieve (NOTES 9d: C survived a
re-solve exactly because it only saw `Rb_lmn`) **expires here**. The AL must actually drive
force balance down, or C is being evaluated on something that is not an equilibrium.

## D. Open risks NOT fixed

1. **`gap_seg = 0.0106`** on precise_QA: one frozen 2-segment split is nearly straight
   (lambda_1 is 1% of lambda_2). The frozen-normal formulation survives this, which is
   exactly why that rewrite was kept -- but the frozen NORMAL of a nearly-straight segment
   is poorly determined, so it may be a bad choice once the geometry moves. Watch it across
   rebuilds.
2. **The frozen data never refreshes inside a run.** Build-time only, so C is an upper bound
   on the true C that drifts as the optimizer moves. Mitigation is to chain short legs and
   rebuild, checking the C jump at each rebuild. Untested.
3. **`sym=False` is untested.** Both verification equilibria (precise_QA, NCSX) are
   stellarator-symmetric, so the sin/cos branches of `_fourier_eval` are exercised but the
   asymmetric mode combinations are not. Test before trusting on an asymmetric boundary.
4. **Two surfaces are mixed.** Positions come from `params["Rb_lmn"]`; the field comes from
   the interior solution evaluated at rho = 1. These agree only if `BoundaryRSelfConsistency`
   holds. It is linear and satisfied exactly by projection inside DESC's optimizer, so this
   is safe in the intended use -- but calling this objective standalone on an unconverged
   equilibrium silently mixes two different surfaces.
5. **No guard on `G -> 0`.** A vacuum stellarator always has net poloidal current, but the
   Newton initial guess `zeta = 2 pi level / G` divides by it.
6. **Newton branch safety.** Monotonicity is checked globally and the contours came out
   distinct and ordered here, but the initial guess assumes Phi~ is small compared with G.
   If that ever fails the solver could land a period off and silently relabel curves. The
   distinctness check should be run as a diagnostic, not just once by hand.
7. **`I` is computed but the contours use Phi_mod = Phi - I theta / 2pi implicitly.** Correct
   (verified I = 0 on vacuum), but nothing asserts `I/G` is small, which is the condition
   for the modular proxy to be meaningful at all.

## D2. Two failures found by ACTUALLY RUNNING it (2026-09-22, `runs/v1/`)

Both were invisible to every static check above.

### D2.1 The Python loop over contours unrolls the traced graph

`_curves_from_params` had `for k in range(K)`, so the graph held K copies of the contour
solve, each with `newton` inlined Newton steps over n nodes. At K=16, n=480 that is ~96
inlined solves in one graph and **XLA spent over 7 minutes compiling before iteration 0**.
Fixed with `jax.vmap` over the contour levels. Verified behaviour-preserving: C identical to
6 decimals at K=4 and the K=16 baseline reproduces exactly (0.9933, range 0.8657-1.3724).

Lesson worth generalising: anything that scales with K belongs in a `vmap`, because the
frozen data has to be refreshed periodically and every refresh pays the compile again.

### D2.2 `second_order="constraints"` OOMs at 4.42 GiB, and is not wanted here

Crash was in `lsq_auglag` at `so_lam_min = float(eig_h[0][0])`, forming the dense
constraint Hessian. **CLAUDE.md's justification for `second_order` is specific to the coil
DISTANCE PENALTIES**, which are degenerate at the bound: their gradient vanishes there, so
multipliers grow and the term Gauss-Newton drops comes to dominate. None of the stage-1
constraints has that structure -- force balance, iota, aspect ratio and the eps_1 equality
are all non-degenerate. So it was paying full price for nothing. Now OFF by default in
`stage1_v1.py`, with `--second-order` to restore it.

### The one iteration that did complete is the best signal so far

Cost 8.009 -> 7.240, **-9.6% in a single iteration**, against v0's -2.3% over seven. C is
strongly drivable through the contour formulation -- expected, since it now depends on the
whole interior solution (1440 of 1462 Jacobian columns) rather than only the boundary.

## E. Recommended configuration after this review

```
K      = 16-24 per field period   (was 4; C1)
M, N   = 32, 64                   (was 64, 128; C2)
n      = 480                      (was 240; C3)
Mt, Nt = 16                       (converged; Mt=8 is wrong in the 3rd decimal)
newton = 6                        (converges in 3)
deriv_mode = "rev"                (forward mode OOMs at 207 GB)
second_order = OFF                (OOMs at 4.42 GiB; D2.2)
contours vmapped, never looped    (7 min compile otherwise; D2.1)
```
