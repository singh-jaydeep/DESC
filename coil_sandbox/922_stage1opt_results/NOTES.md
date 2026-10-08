# Stage-1 proxies for "does this equilibrium support planar / piecewise-planar coils?"

Started 2026-09-22. Read with `coil_sandbox/CLAUDE.md` and `work/helios/HANDOFF.md`.

**The question.** Whether an equilibrium admits planar coils is a property of the equilibrium, not
only of the stage-2 solver. The same should be true of arcB2 (two planar faces, "clamshell"). We
want a quantity computable at stage 1, on the rho=1 surface, that predicts it — and eventually one
that can be optimized against.

**Status: T0 passes and the builder works (5b). The OBSERVATIONAL program is dead (5c).\nThe live plan is the intervention on a vacuum equilibrium (5d); sections 6-7 are superseded.** This file is the plan and the reasoning. Everything below the
"Validation suite" heading is unrun (sections 5 ground truth is measured and current).

---

## 1. Where the literature actually stands

Four questions get conflated. They have different answers.

| question | best known answer | use here |
|---|---|---|
| How close must the coils be? | `min L_grad(B)` on the LCFS | control variable only |
| How *non-planar* must they be? | principal-direction rotation rate `pdrot` | **validation partner for `eps_1`** (T4) |
| What *shape* will they be? | contours of the current potential on the LCFS | **the proxy we build on** |
| Where can coils live at all? | the `det(grad B) = 0` manifold | vacuum-QS only; not actionable |

- **`L_grad(B)` = sqrt(2)|B| / ||grad B||_F**, normalized so that for an infinite straight wire it
  equals the distance to the wire — which is why it reads as a distance. Kappel, Landreman &
  Malhotra (PPCF 2024, arXiv:2309.11342) and the follow-up (NF 66, 066021 (2026),
  arXiv:2602.18974) get **R^2 = 0.944** for `min(L_gradB)/a` vs `min(d_cs)/a` over 3027 QUASR
  configurations (only 0.644 if normalized by R0 — the normalization is not cosmetic). The argmin
  *location* also transfers (45% of cases within Delta < 0.1, 63% for the second-order variant
  `L_gradgradB = sqrt(4B/||grad grad B||_F)`, vs 3% for random points).
- **It is the wrong tool for planarity.** Pavone & Warmer (arXiv:2604.26763) regressed coil
  complexity on boundary geometry over 7500 Constellaration QI boundaries with SIMSOPT coil runs:
  `L_gradB` showed **|rho| < 0.2 against every coil-complexity metric**. The winner is
  `pdrot = |grad_S alpha|`, the rate at which the principal-curvature frame rotates across the
  surface (Spearman rho = 0.936, univariate R^2 = 0.700 vs max non-planarity; R^2 = 0.87 with four
  features). Their non-planarity metric is worth stealing: SVD of the Np x 3 matrix of coil points
  about the centroid, `eta_SVD = sigma_3/(sigma_1+sigma_2+sigma_3)`, 0 planar, -> 1/3 maximally
  non-planar.
- **The constructive proxy** is Rodriguez & Sengupta, *Estimating coil features from an equilibrium*
  (arXiv:2604.12339). In the limit of coils on the boundary the required sheet current is
  `K = -(1/mu0) n x B`; its stream function is the restriction of the vacuum scalar potential to the
  boundary, which is a scaled Boozer toroidal angle; so **contours of constant zeta_Boozer on a flux
  surface are proxy modular coils**. They then get closed-form coil curvature from equilibrium
  quantities (normal `kappa_n~ = 2H - kappa_n`, geodesic `kappa_g~ = (1/B) tau tau : grad B`) and a
  non-planarity scaling as `Delta z ~ rho^2`, validated against REGCOIL on precise-QA, precise-QH and
  W7-X, where it **bounds the true non-planarity from below**.

**Agreed position (user, 2026-09-22):** `L_grad(B)` is an indirect, general-purpose smoothing metric
— worth carrying in a stage-1 objective because it should make the eventual coils better behaved,
but it does not attack the planarity question. It is a *covariate* in everything below, never the
response.

---

## 2. What we measured on Helios (`lgradb_helios.py`, CPU, 2026-09-22)

Total field, rho = 1, LinearGrid M = N = 64, sym off.

```
a = 1.7719 m,  R0/a = 4.493,  NFP 2
min  L_gradB = 2.5430 m = 1.435 a   at R = 9.813, Z = 0.000  (outboard midplane, zeta = 0)
 5th pct      3.9531 m = 2.231 a
median        6.7803 m = 3.827 a
harmonic mean 6.3743 m
area-wtd mean 6.7895 m
max           13.137 m
|K_vc| = 4.38e6 .. 6.01e6 A/m   (max/min = 1.37)
```

Two things follow.

1. **Helios is not standoff-limited.** The imposed coil-plasma distance is 1.2 m = 0.68 a, less than
   half of `min L_gradB`. This agrees with a measurement already in HANDOFF ("Constraint
   sensitivity"): relaxing `d_pc` 1.2 -> 1.0 m gained +0.3%, the coils took 7 cm of the 20 offered
   and ended marginally worse. So the 4.04% / 0.9% field errors are *not* "the equilibrium wants
   closer coils". One proxy, one direct shadow price, same answer.
2. **`|K_vc|` is nearly uniform on Helios (1.37x).** This matters for section 3: Helios is a *weak*
   test bed for any current-weighted selection rule, because the weighting barely discriminates.
   Say so in any writeup; a configuration with strongly peaked `|K|` is needed to test that idea
   properly.

Caveat: computed from the **total** field. Kappel computes `L_gradB` from the vacuum field
(virtual-casing-subtracted) since only the external part is the coils' job. Helios' plasma
contributes B.n ~ 2.4% of <|B|>, so the difference should be small, but T6 below checks it.

---

## 3. The selection problem, and why equal-current cutting solves it

**The objection (user's, and it is the right one):** the streamlines of the surface current foliate
rho = 1. That is an infinite one-parameter family of curves, not a coilset. Something has to pick a
finite set, and the pick should know that curves in regions of large current matter more.

**Resolution.** `grad_S . K_vc = -n . J`, and `J` lies in the flux surface (`J . grad rho = 0`), so
`grad_S . K_vc = 0` exactly and `K` has a stream function `Phi`:

```
K = n x grad Phi        <=>        grad_S Phi = K x n
```

Note this is the precise sense in which the user's "J on the LCFS" intuition is right: **it is
`J . n = 0` that makes the proxy coils exist as closed contours at all.** But the curves you draw
are the `K`-lines, which are the orthogonal trajectories of the field lines in the surface (roughly
poloidal, i.e. modular-coil-shaped), not the `J`-lines, which run roughly along `B`.

Now the key identity: `|grad_S Phi| = |K|`, so the perpendicular spacing between two contours
separated by `dPhi` is

```
dl_perp = dPhi / |K|
```

**Cutting the foliation at equal increments of `Phi` gives every proxy coil the same current, and
automatically crowds the curves where `|K|` is large.** The importance weighting the user wants is
not an extra ingredient — it is what equal-current cutting already does. This is the standard
NESCOIL/REGCOIL coil-cutting rule, and it is the only choice that makes the proxy coils physically
comparable to each other.

What is left of the freedom is small and enumerable:
- the number of contours `N_c` (set it to the coil count you are comparing against: 12 or 16);
- one **phase offset** `phi0` in `[0, dPhi)`. Stellarator symmetry pins this to two natural values
  (a coil centred on the symmetry plane, or a gap centred there). Scan it; the spread is itself a
  diagnostic (T6).

### 3.1 Secular terms: strip the toroidal one

`Phi` is multivalued:

```
Phi(theta, zeta) = (G zeta + I theta)/(2 pi) + Phi~(theta, zeta)
```

`G` = net poloidal current linking the surface (what the modular coils must supply),
`I` = net toroidal current (the plasma's own). For Helios, `G ~ 2 pi R0 B0 / mu0 ~ 2.41e8 A` and
`I = 4.6e6 A`, so `I/G ~ 0.019` — use those two numbers as the sign-convention self-check when
extracting the coefficients.

With `I != 0` the level sets of `Phi` do not close poloidally: you cannot replace a plasma carrying
net toroidal current with modular coils. The proxy for the **coils** therefore contours

```
Phi_mod = Phi - I theta / (2 pi)
```

A 2% correction in magnitude, but it is the difference between closed and non-closed curves, so it
is qualitative. Record both; a run that forgets this will produce drifting curves and nonsense
eps_B.

### 3.2 A phase- and count-free version of the metric

Because `dPhi` is the equal-current measure, averaging any per-curve quantity uniformly in `Phi`
over the whole foliation is *exactly* a `|K|`-weighted surface average
(`dA = dl_parallel dPhi / |K|`). So define

```
eps_B_bar^2 = (1/G) * integral_0^G  eps_B(C_Phi)^2  dPhi
```

which is independent of both `phi0` and `N_c`, and is the current-weighted defect the user asked
for. Discretize with ~64 contours per field period; it is cheap. Also report:
- `eps_B^max` over the foliation (the worst coil, often what actually binds);
- `eps_B` of the `N_c`-contour sample (what you would actually build);
- a weight exponent scan `eps_B_bar^(p)` with surface weight `|K|^p`, `p = 1` being equal-current,
  so that "how much does the weighting matter" is measured rather than assumed.

`eps_B` itself is the existing DP defect from `917_results/planarizability.py` (exact best split of
a closed curve into B arcs, arclength-weighted, normalized by the curve's own in-plane extent).
`eps_1` = planar, `eps_2` = clamshell-able, and the decay with B is the signature.

**Differentiability is deliberately out of scope until the values are shown to have content.**
(Recorded for later: the DP value is differentiable where the optimal segmentation is unique
(Danskin) — freeze the breakpoints and differentiate only the lambda_min-of-covariance part. The
division-free alternative is clustering `x' x x''` on the sphere weighted by its own magnitude,
which ignores straight legs automatically rather than dividing by a vanishing curvature the way
`CoilTorsion` does.)

---

## 4. The trap this whole program has to avoid

`planarizability.py`'s own docstring states it, and the Helios data sharpens it:

- **`eps_B` is the distance from a coilset to the piecewise-planar set, not the cost of imposing the
  restriction.** Metres of projection distance corresponded to a 19% optimum gap at 12 coils.
- **The restriction cost is not a property of the equilibrium alone.** On *the same* Helios
  equilibrium the arcB2/FourierXYZ ratio went **1.24x at 12 coils -> 2.30x at 16**, purely from
  crowding. A proxy can at best predict the cost *at a stated coil budget*.

**Protocol rule: every number produced under this program carries its budget
`(N_coils, L, d_cc, d_pc, convexity on/off)`.** A comparison that does not state all five is void.

---

## 5. Ground truth: take it from `viewer_data.json`, never from a handoff

**Protocol rule (user, 2026-09-22).** Any "best config X vs best config Y" claim regenerates its
numbers from `917_results/viewer_data.json`, which is the live inventory (**37 coilsets**, not the
11 hard-coded in `build_data.py` — `add_coilset.py` appends). A handoff table records what was best
*when it was written*. Sort by `bn_mean` and read off the winner for the budget you want.

Verified 2026-09-22. `bn` is <|B.n|>/<|B|> with the plasma field included; `d_cc`/`d_pc` are
*achieved* distances, so check them against the bound the run actually used.

### The B-ladder at 12 coils (complete — this is the test bed)

| B | key | bn | d_cc | d_pc |
|---|---|---|---|---|
| 1 | `planar_N7` | 4.0223e-2 | 2.019 | 1.651 |
| 2 | `polarB2_M5_cut30` | 1.0450e-2 | 1.618 | 1.594 |
| 3 | `arcB3_M5` | 1.0086e-2 | 1.241 | 1.666 |
| inf | `fourierXYZ_N10_ceiling` | 8.4224e-3 | 1.228 | 1.621 |

### The B-ladder at 16 coils (INCOMPLETE — no B=3 exists)

| B | key | bn | note |
|---|---|---|---|
| 1 | `planar_N7_16coils` | 3.3448e-2 | |
| 2 | `polarB2_16_L36_even` | **7.7596e-3** | the L-ladder result, M=5, paper-feasible |
| 3 | — | — | **does not exist**; arcB3 was only ever run at 12 coils |
| inf | `fourierXYZ_16coils` | 3.7978e-3 | stopped on max-iterations, so a lower bound |

### Corrections this forced

1. **The handoff's headline "piecewise-planar costs 2.30x at 16 coils" is stale.** It used
   `polarB2_M5_cut30_16coils` (8.7227e-3). Against the same `fourierXYZ_16coils`:

   | B=2 coilset used | bn | ratio | what is held fixed |
   |---|---|---|---|
   | `polarB2_M5_cut30_16coils` (handoff) | 8.7227e-3 | 2.297x | — |
   | `polarB2_16_L36_even` | 7.7596e-3 | **2.043x** | best paper-feasible at M=5 |
   | `polarB2_16_M10` | 6.8777e-3 | **1.811x** | best paper-feasible at any M |
   | `polarB2_16_noconvex` | 6.4231e-3 | **1.691x** | convexity dropped on BOTH sides |

   The last row is the only convexity-matched comparison, since the FourierXYZ reference never
   carried convexity. Quote **1.69x** for "the cost of two planar faces at 16 coils", and 2.04x only
   when convexity is explicitly part of the restriction. Either way it still grows from 1.24x at
   12 coils, so the coil-count effect survives; its size does not.
2. **A whole `demount_*` family exists** (6 coilsets, 1.73e-2 to 3.10e-2) that post-dates the
   handoff I read: arcB2 at 16 coils with joints constrained to fixed z, i.e. demountability as an
   extra restriction. Directly relevant to this program later, but a different constraint class, so
   it stays out of the B-ladder.
3. **Not yet in the viewer:** `work/helios/arc16/midplane/level/L34/result.h5` (written today
   11:56, after `viewer_data.json` at 10:50) reaches bn 1.5149e-2 at `d_cc 1.031 / d_pc 1.107` —
   a *relaxed-clearance* budget, not paper-feasible, and it beats `demount_level_L28` (2.2867e-2).
   Left for the user to add with `add_coilset.py`; flagged here so it is not silently missed.

---

## 5b. BUILT AND RUN (2026-09-22): the Phi-contour builder and T0

`phi_contours.py` + `t0_checks.py` + `run_helios_proxy.py`. All CPU, seconds per run.

### DESC PR #2285 is now vendored

`FourierRZSurfaceCurve` / `FourierRZSurfaceCoil` / `SurfaceCurveConsistency` are in
`vendor/desc`. It fits the proxy exactly: every contour advances 2pi in theta and returns
in zeta, i.e. `secular_theta=1, secular_zeta=0`, with theta(s) = s and zeta(s) a plain
Fourier series. The proxy curve is then a DESC object, so `compute("x")`,
`planarizability.py` and (later) E3's joint optimization all work on it unchanged.

How it was merged, because it matters for reproducing it: the PR's own base
(a1520784, #2303) and the vendor base (33076c84, #2292) are **identical in every file the
PR touches**, so the PR applies cleanly to the vendor base; the merge is then
vendored-vs-(base+PR) with base 33076c84. Result: **additions only, zero deletions against
the vendored code.** `desc/objectives/_omnigenity.py` was taken as ours (the PR's changes
there are unrelated drive-by edits). `desc/compute/_curve.py` and `desc/geometry/curve.py`
were rebuilt by exact block insertion rather than a textual union, because git's conflict
boundaries cut through `@register_compute_fun` decorators; both files are purely additive
in the PR (528/0 and 500/1), so insertion is exact.

Verification: `check.py` passes; `polarB2_M5_cut30` re-evaluates to bn 1.044989e-2 against
the recorded 1.0450e-2, `arcB3_M5` to 1.008281e-2 against 1.0086e-2. The only shared
infrastructure the PR touches is a `noqa`, a warning, verbose prints, and a `_cholmod`
guard for `lb == 0` that is a no-op whenever `lb != 0`. Backup of the pre-merge vendor
tree is in the session scratchpad only, so **re-vendor from `upstream/js/surface-curve-consolidation`
if it is needed again**; `vendor/desc_changes_vs_master.patch` is now stale.

### T0 — all three checks pass

**T0a, the real zero-check: an analytic pure-TF field on an axisymmetric shaped surface.**
`K = n x B / mu0` is then purely poloidal, `Phi = G zeta / 2pi` exactly, and every contour
is planar. This exercises the surface metric, the secular extraction, the Poisson solve,
the contouring and the curve fit against a closed form:

```
G          2.388000e+08 A vs exact 2.388000e+08    rel 2.5e-16
I          0                                        exact 0
residual   5.9e-16
|Phi~|rms  0 exactly
contours   lie in zeta = const planes to 0.0e+00 rad
eps_1      max 4.5e-08 over 8 proxy coils
```

(DSHAPE/SOLOVEV would NOT have served: a real axisymmetric equilibrium carries plasma
current, and section 5b's T0c shows that is exactly what breaks planarity. The analytic
case is the only one with a known exact answer.)

**T0b:** `eps_1` of `planar_N7` is 0.00e+00.

**T0c, and it quantifies the proxy's known bias.** The sheet must carry the plasma's
toroidal current while real coils do not, so the proxy is contaminated at O(I/G). On
DSHAPE, `I/G = 5.91e-2` and `eps_1 = 4.81e-2` for every proxy coil — a ratio of **0.82**.
So the bias is not merely of order I/G, it is **approximately 0.8 x I/G**, and on Helios
(`I/G = 1.78e-2`) it contributes **eps_1 ~ 0.015**. Against a measured Helios proxy
`eps_1 ~ 0.066` that is 22% of the signal. Subtract or at least quote it.

### Two calibration facts that change how tolerances are set

1. **`eps_B` has a numerical floor of ~1.5e-8**, because it is `sqrt(lambda_min)` of a
   covariance at round-off. Never set a tolerance below ~1e-7. (`EPS_FLOOR = 1e-6` in
   `t0_checks.py`.)
2. **The stream-function residual is an equilibrium-quality diagnostic, not an error.**
   On Helios it is 1.15e-3 and **resolution-independent** (identical from M,N = 48x96 to
   160x320, with G, I and Phi~ converged to 9 digits at the coarsest). It measures the
   equilibrium's own `J . n` at rho = 1, since `grad_S . K = -n . J`: measured
   `sqrt(g)|n.J|/scale` is rms 2.3e-3, max 1.1e-2, bracketing the 1.15e-3 fit error.
   A large residual means the proxy's premise is shaky for that input.

### Helios, first look (NOT yet T1/T2 — no offset mapping, no robustness sweep)

```
G 2.5984e+08 A    I 4.6324e+06 A    I/G 1.783e-2    residual 1.15e-3
monotonic in zeta: True   dPhi/dzeta 3.40e7 .. 4.91e7 (never straddles 0)
|K| 4.377e6 .. 6.008e6 A/m, ratio 1.3725
```

**The extracted `I = 4.632 MA` matches Helios' known 4.6 MA net toroidal current to 0.7%.**
That is an independent confirmation of the sign convention and of the whole extraction,
and it is the self-check section 3.1 asked for.

Proxy coils (lengths 11.3-14.5 m, on the surface, vs 36 m for real coils at 1.2 m standoff):

| N_c | current/coil | eps_1 | eps_2 | eps_3 | ln(e1/e2) | ln(e2/e3) | ratio |
|---|---|---|---|---|---|---|---|
| 12 | 21.65 MA | 0.0655 | 0.0130 | 0.0080 | 1.619 | 0.483 | **3.3** |
| 16 | 16.24 MA | 0.0657 | 0.0133 | 0.0078 | 1.601 | 0.527 | **3.0** |

Three preliminary readings, all needing T1/T6 before they are worth anything:

1. **The B=1 -> 2 collapse is there and dominant** (1.62 vs 0.48), and the Spearman rank
   over B = 1,2,3 is +1. So the qualitative T2 prediction survives first contact.
2. **The ratio is 3.0-3.3. See the REVISED T2 below before reading anything into that.**
   The original bar ("> 5, measured 38") was wrong: `bn` has a floor that `eps` cannot
   have, and the floor-subtracted target is 14-17. Against the proper null for `eps`
   (B^-2, ratio 1.71) Helios PASSES; against the floor-subtracted bn ladder it does not.
   Still owed: the offset mapping (T1) and removal of the 22% I/G bias, both of which
   push the ratio up.
3. **The proxy is blind to coil count by construction.** `eps_B` is a property of the
   contour, and N_c = 12 and 16 sample the same family of curves, so the values are
   identical to three digits. Since coil count moved the measured restriction cost from
   1.24x to 1.69-2.04x, the proxy can only ever address the equilibrium half of the
   question. That is section 4's protocol rule, now demonstrated rather than asserted.

### Useful by-product: a cheap admissibility test

`Phi_mod` is monotone in zeta iff `sqrt(g) K^theta` never changes sign, and that is exactly
the condition for every contour to be a single-valued `zeta(theta)`, i.e. a genuine modular
curve rather than a saddle or windowpane. It is one line on the raw data, needs no
contouring, and `modular_contours` refuses to run without it. Expect it to be the thing
that fails first in Phase B.

---

## 5c. THE OBSERVATIONAL PROGRAM IS DEAD (2026-09-22). Cross-configuration result.

Running the builder on the examples library took two minutes and settled what sections 6
and 7 proposed to spend days on. **Sections 6-7 below are superseded and kept only as a
record of what was tried.**

| config | beta | I/G | residual | eps_1 | eps_2 (B^-2 null) | ratio |
|---|---|---|---|---|---|---|
| precise_QA | 0 | -2e-17 | 1.9e-5 | 0.0669 | 0.0169 (0.0167) | 1.77 |
| precise_QH | 0 | -1e-17 | 4.1e-5 | 0.0543 | 0.0136 (0.0136) | 1.36 |
| HSX | 0 | -7e-13 | 3.8e-5 | 0.0474 | 0.0134 (0.0118) | 1.04 |
| ESTELL | 0 | -8e-16 | 1.5e-4 | 0.0542 | 0.0142 (0.0136) | 2.01 |
| W7-X | 2.0e-2 | -1.8e-4 | 2.3e-4 | 0.0780 | 0.0246 (0.0195) | 0.90 |
| NCSX | 4.1e-2 | -1.5e-2 | 3.8e-4 | 0.1268 | 0.0292 (0.0317) | 2.02 |
| ARIES-CS | 4.1e-2 | -1.5e-2 | 2.0e-4 | 0.1297 | 0.0259 (0.0324) | 1.73 |
| **Helios** | 2.6e-2 | **1.8e-2** | **1.15e-3** | 0.0655 | 0.0130 (0.0164) | **3.3** |

Three conclusions.

1. **Vacuum IS the clean regime, quantitatively.** I/G = 1e-17 (exactly zero, as it must
   be) and the stream-function residual is 2e-5 to 1.5e-4 -- one to two orders of magnitude
   below Helios' 1.15e-3. The proxy's premise holds far better there.
2. **`eps_2` has essentially no cross-configuration dynamic range.** It sits on the generic
   `B^-2` null within +-25% for every configuration, across a set whose coil complexity
   differs enormously. `eps_1` spans only 2.7x, and puts **HSX lowest**, which is hard to
   reconcile with HSX's coils. This is T8, run early and failed.
3. **Helios' 3.3 was noise.** Nothing else in the set exceeds 2.02. There was never a prior
   that Helios prefers clamshells, and building null-model tests on it was extracting a
   result from contamination (Helios has all three finite-beta effects below).

### What finite beta does to the interpretation (the user's question, answered)

SURVIVES: the sheet identity `K = (1/mu0) n x B` reproducing the interior field exactly and
zero outside, for ANY equilibrium; and `grad_S . K = -n . J = 0` because J lies in the
surface.

FAILS: (i) the identification of Phi with a scaled Boozer zeta -- that is the vacuum
scalar-potential argument and needs vacuum; (ii) real coils do not reproduce the interior
field, so the sheet carries the plasma's net toroidal current while modular coils do not
(measured bias 0.8 x I/G, ~22% of Helios' eps_1); (iii) separately and unquantified, even
at I = 0 the sheet absorbs the Pfirsch-Schlueter and diamagnetic currents. W7-X
(beta 2%, I/G 1.8e-4) would isolate (iii) from (ii) if that is ever needed. Helios has all
three.

---

## 5d. THE PLAN NOW: an INTERVENTION on a vacuum equilibrium (agreed 2026-09-22)

The user's design, and 5c strengthens rather than weakens it: observational correlation is
hopeless precisely BECAUSE there is no natural dynamic range. An intervention can push the
variable where nature does not, which is the only way to find out if it has causal content.

**The experiment.** On a vacuum equilibrium: stage-1 optimize the clamshell metric, then fit
arcB2 AND FourierXYZ coils at a FIXED budget to each resulting boundary, and ask whether
the arcB2/FourierXYZ gap moves.

**Target (user's choice): the clamshell ratio**

```
C = eps_2 / (eps_1 / 4)
```

i.e. how much better than the generic `B^-2` decay the proxy curve does. Rationale: it is
scale-free in overall non-planarity, so it **cannot be gamed by flattening toward
axisymmetry** -- and an axisymmetric torus has `eps_1 = eps_2 = 0` with perfect planar
coils, so that is the trivial global optimum raw `eps_2` would slide into. Natural range
over the 7 configs in 5c is 0.79 (ARIES-CS) to 1.26 (W7-X), so there is headroom to push.

**Control (user's choice): 3-point dose-response.** Optimize C DOWN, keep the baseline, and
optimize C UP. If minimizing shrinks the arcB2/FourierXYZ gap and maximizing widens it,
that is causal. A 2-point before/after cannot distinguish a real effect from generic
sensitivity to a boundary perturbation of that size. Cost: 6 stage-2 runs, ~5 h GPU, one
job at a time.

**Anti-trivialization constraints, all of them required:** fix aspect ratio, fix `<beta> = 0`
and net current = 0 (vacuum), hold QS error no worse than baseline, and **fix the rotational
transform** -- a vacuum stellarator with iota != 0 and no net current must be 3D, which is
the constraint that actually blocks the slide to axisymmetry. Hold `a` fixed too, so the
`paper_bounds` scaling does not silently change the coil budget between the three points.

**Gate before any GPU:** run stage 1 for all three points first and check how far C actually
moved at fixed iota/QS/A. If it barely moves there is nothing to measure and we stop cheaply.

### Differentiability is now on the critical path

We had agreed to defer it until the values showed content. The intervention IS the content
test, and it needs a stage-1 objective, so that agreement no longer holds. Build order:

1. **A differentiable `eps_B` on a curve** -- **DONE, `clamshell.py`.** Danskin: the DP runs
   in numpy for the breakpoints, they are frozen, and only `lambda_min` of the
   arclength-weighted covariance is differentiated. Validated: values agree with
   `planarizability.py` to 6e-5 / 1.1e-3 / 9.1e-4 for B = 1,2,3, and `dC/dp` matches central
   finite differences to **7.8e-10**.
   - Two deliberate differences from `planarizability.py`, both required for a clean
     gradient: points live on a UNIFORM PARAMETER grid weighted by `|x_s|` rather than being
     resampled to uniform arclength (resampling is an interpolation whose nonsmoothness
     pollutes the gradient), and segment boundaries are frozen indices.
   - **Trap, cost an hour:** never form the weighted covariance as `(P-mu)*sqrt(ww)`. The
     segment mask zeroes entries and `d/dx sqrt(x)` at 0 is infinite, so every gradient came
     back NaN while the VALUES were all correct. Use `(D * ww[:,None]).T @ D` instead.
   - Two remaining sharp edges, neither hit in practice but worth knowing: `eigvalsh` has a
     NaN gradient if `lambda_min` is degenerate (a segment with < 3 points), and
     `sqrt(tot/W)` has an infinite gradient at an exactly planar segment.
   - It imports `jax.numpy` directly rather than `desc.backend`, so a consumer that forgets
     `boot.setup()` cannot drag in the non-vendored DESC through this module. The project
     rule still applies to the consumer: the first jax import fixes the platform.
2. **The curve as a free DESC object, not a contour solve.** Do NOT differentiate through
   the contour root-find. Instead use PR #2285 as designed: a `FourierRZSurfaceCoil` on the
   rho=1 surface whose own params are optimized, pinned to be a streamline by a residual
   objective `|x_s x K| / (|x_s||K|) = 0`, with `SurfaceCurveConsistency` tying its surface
   params to the equilibrium's. Then `C` acts on the curve's params and the whole thing is
   ordinary DESC optimization.
3. Stage-1 runs, the gate, then the 6 stage-2 runs.

**Equilibrium: precise_QA**, unless the user says otherwise -- vacuum confirmed above,
it is one of the three cases Rodriguez & Sengupta validated the proxy on, and the sandbox
already has QA tooling from the sweep.

---

## 5e. DESIGN of the stage-1 objective (agreed with the user, 2026-09-22)

### The streamline condition is orthogonality to B, and that is simpler than a cross product

`K = n x B / mu0` and `x_s` is tangent to the surface, so `x_s || K  <=>  x_s . B = 0`.
The residual is therefore one scalar per node,

```
r = (x_s . B) / (|x_s| |B|)         (the cosine of the angle; dimensionless, in [-1, 1])
```

no cross product, and no vanishing-denominator worry since `|K| = |B|/mu0`. This replaces
the `|x_s x K| / (|x_s||K|)` form written in 5d.

**It pins direction, not identity.** Every member of the foliation satisfies `x_s . B = 0`,
so a set of curves under this objective alone can all collapse onto the SAME streamline.
Fix: pin each curve's zeta at theta = 0 to a distinct value -- a linear constraint on the
curve's own params, and the natural place to impose equal-current spacing later.

### Hinges: min over a user-chosen candidate set, not the DP, and not lifting

The user's proposal: pick a number of candidate hinge pairs, evaluate eps_2 at each, take
the min. Adopted. The tempting alternative -- lift the hinge positions into the outer
optimizer as free variables, so there is no inner minimisation at all -- **works only for
the minimising arm**. Both minimisations point the same way there. On the MAXIMISING arm of
the dose-response it becomes `max_boundary ( min_hinges )`, a minimax that lifting silently
converts into the wrong problem. A genuine min over candidates is correct in both arms.

`eps_1` has no hinge freedom (one plane, whole curve), so only the numerator of C needs this.

### Measured: how many candidates are enough, and why softmin

Full `eps_2` landscape over all ~57k breakpoint pairs, 240-point proxy curves
(`hinge_landscape.py`):

| | eps_1 | eps_2 | best hinges (s/2pi) | arc split | within 2/5/10% | separated minima <=5% |
|---|---|---|---|---|---|---|
| precise_QA | 0.0462 | 0.0159 | 0.267, 0.875 | 0.61/0.39 | 0.7 / 2.3 / 5.6 % | 8 |
| precise_QH | 0.0558 | 0.0171 | 0.400, 0.896 | 0.50/0.50 | 1.5 / 3.0 / 5.0 % | 8 |

1. **The basin is broad.** ~5% of the whole hinge space is within 10% of optimal, so ~100
   candidates miss that band with probability ~0.6%; a stride-12 grid on n = 240 gives 190
   pairs for negligible cost. A modest candidate set, refreshed from the DP every few outer
   iterations, is plenty.
2. **But there are 8 well-separated local minima within 5% of the best**, so the argmin
   genuinely switches during optimization. **Use a softmin (log-sum-exp) over the
   candidates**, not a hard min, and not a single frozen DP argmin -- both chatter at ties.
   Anneal the temperature toward a hard min if a sharp endpoint is wanted.
3. Per-coil spread is real: coil 0 of precise_QA gives C = 1.372 while the contour-average
   in 5c is near 1.0. Aggregate over the foliation (section 3.2), do not quote one contour.

### Spreading the curves matters; chasing high |K| does not (measured, `foliation_spread.py`)

The user's point: the free-curve formulation LOSES what equal-dPhi cutting gave for free
(section 3), since `x_s . B = 0` is satisfied by every member of the foliation. So the
curves need something pushing them toward high current density and keeping them apart.
One objective delivers both in principle -- equal current per coil <=> `|K| d = const`
with `d` the local perpendicular gap to the neighbour, local and differentiable, no need
to reconstruct Phi. **Measured, only half of it earns its place.**

C over 24 equal-current contours:

| | C range | mean \|K\| ALONG a contour | corr(C, \|K\|) | effect of \|K\|-weighting on the aggregate |
|---|---|---|---|---|
| precise_QA | 0.866-1.372 (**1.58x**) | 1.014x | -0.728 | **0.04%** |
| precise_QH | 0.706-1.228 (**1.74x**) | 1.026x | -0.034 | **0.01%** |

1. **Spreading is essential.** C varies 1.6-1.7x across the foliation, so a bunched or
   collapsed set gives a badly biased aggregate.
2. **High-|K| attraction is not.** The mean |K| along a contour varies by only 1.4-2.6%,
   far flatter than the POINTWISE 1.24-1.38x, because each contour runs through both high-
   and low-|K| regions and the variation averages out. Reweighting moves the aggregate by
   0.01-0.04%. Note QA has corr(C,|K|) = -0.73 with essentially zero leverage: correlation
   is not leverage when the predictor spans 1.4%.

**Decision: pin each curve's zeta at theta = 0 to EQUALLY SPACED values** (a linear
constraint on the curve's own params, zero extra machinery) and **do not build the
`|K| d = const` objective** until a configuration with genuinely peaked |K| turns up.
Equal-zeta spacing differs from equal-current by the few percent that |K| varies.

3. **Convergence-check the aggregate in the NUMBER of curves.** 24 contours gave mean 0.996
   but median 0.970 on QA, so the distribution is skewed and outliers pull the mean. Do not
   guess the count.

### SIGN CONVENTION -- got this wrong once, in `clamshell.py`'s own docstring

`C = eps_2 / (eps_1/4)`. The null is `eps_2 = eps_1/4`, so

```
C < 1   clamshells BETTER than a generic smooth curve   <- minimise this direction
C = 1   generic
C > 1   worse than generic
```

The docstring originally said "> 1 means a real two-plane preference", which is backwards
and would have pointed the optimization the wrong way. Corrected. Note precise_QA (1.37)
and precise_QH (1.23) are both WORSE than generic on coil 0, and Helios' contour-average
0.79 is the only value in the whole set on the good side of the null -- another reason not
to read Helios' number as signal until the finite-beta contamination is removed.

---

## 6. SUPERSEDED by 5c/5d — the observational suite, Phase A (kept as a record)

Helios first (most coilsets on disk), cross-configuration second. All CPU, no GPU, no cgroup.

### T0 — Pipeline zero-checks (do first, ~20 min)

- Run the whole construction on an **axisymmetric** equilibrium (`SOLOVEV` or `DSHAPE` from
  `vendor/desc/examples/`). Contours of `Phi_mod` are exactly planar circles in poloidal planes, so
  the answer is `eps_1 = 0`. **Pass:** `eps_1 < 1e-10`. **Fail:** nothing downstream means anything.
- `planarizability.py` on `917_results/planar_N7.h5`, planar by construction: `eps_1` at round-off.
- `eps_1` vs Pavone & Warmer's `eta_SVD = sigma_3/(sigma_1+sigma_2+sigma_3)` on the same curves.
  These are near-tautologically related (both are normalized smallest-plane-fit residuals), so this
  is an implementation check, not a result — but it is the bridge that lets our `eps_1` be compared
  with their published correlations at all.

### T1 — Do the proxy curves look like the real coils?

Prerequisite: if the proxy curves do not resemble optimized coils, their `eps_B` is meaningless.

Build `Phi_mod` contours at `N_c = 16`, map them outward along the surface normal to the 1.2 m
constant-offset surface (`FourierRZToroidalSurface.constant_offset_surface`,
`vendor/desc/geometry/surface.py:667`), and measure symmetric mean-nearest-point distance to the
nearest coil of `fourierXYZ_16coils`.

**Nulls to beat, same pipeline:** circles at r/a 2.5; `planar_N7_16coils`; the proxy curves with the
contour phase shifted by half a gap. **Pass:** materially closer than circles. Report on-surface and
offset-mapped both, since `eps_B`'s normalization by in-plane extent only partly removes the scale
difference.

### T2 — The B-spectrum test (REVISED 2026-09-22 after the user's objection)

**The objection, and it was right.** `eps_B` is a projection distance of ONE fixed curve
determined entirely by the equilibrium; nothing is optimized. `bn_B` is four SEPARATE
constrained optima, none of them obliged to stay near the K-streamline. Comparing their
ladders is a heuristic across that gap, not a derivation. Two consequences.

**(a) `eps_B` is monotone by construction, so "does it drop" is not the question.**
B-segmentations are nested: cut either arc of the optimal 2-split at any interior point
and the parent's best-fit plane is still available to each child, so the 3-split costs no
more. `eps_1 >= eps_2 >= eps_3` always. Only the SIZE of the drop carries information.

**(b) `bn` has a floor that `eps` structurally cannot have.** `eps_B -> 0` as `B -> oo`,
while `bn_B -> bn_free` (8.4224e-3 at 12 coils): finite coil count and 1.2 m standoff put
a floor under the field error that no shape freedom removes. The original T2 compared the
ladders raw, which charges the proxy for that floor. Subtracting it:

| | ln(.1/.2) | ln(.2/.3) | ratio |
|---|---|---|---|
| raw bn (the ORIGINAL, wrong, target) | 1.348 | 0.035 | 38.0 |
| bn - floor, linear | 2.753 | 0.198 | **13.9** |
| bn - floor, quadrature | 1.850 | 0.109 | **17.0** |
| Helios proxy eps | 1.617 | 0.486 | **3.3** |

The target is **~14-17, not 38**, and spans roughly 10-18 over plausible floors --
`fourierXYZ_N10_ceiling` is itself a lower bound (seeded from the planar optimum) and
carries no convexity constraint while B = 1,2,3 all do, and there is no convexity-matched
12-coil free run to use instead. Report the floor and the subtraction convention with any
T2 number.

**The ratio statistic's only justification** is that if `ln bn = c + p ln eps` (a power
law), the ratio-of-ratios is invariant to both `c` and `p`, so it is the one comparison
that does not require knowing the unknown monotone map. That is an assumption.

#### T2' — the primary test is now against a null model, not against bn

For a generic smooth curve with no clamshell structure, B planar arcs leave out-of-plane
error ~ torsion x (L/B)^2, so `eps_B ~ C/B^2` and the ratio is fixed at
`2 ln 2 / 2 ln 1.5 = 1.71` for ANY curve. This needs no assumption about the map to field
error, so it is the sounder primary test.

Measured on Helios (N_c = 12):

| | eps_1 | eps_2 | eps_3 | ratio |
|---|---|---|---|---|
| proxy | 0.0655 | **0.0130** | **0.0080** | **3.3** |
| B^-2 null from eps_1 | 0.0655 | 0.0164 | 0.0073 | 1.71 |

**The clamshell signal is real, confined to B = 2, and modest.** eps_2 beats the null
(0.0130 vs 0.0164); eps_3 is slightly WORSE than the null (0.0080 vs 0.0073); the ratio is
about 2x the generic decay. So Helios' K-streamlines genuinely prefer two planes over what
an arbitrary smooth curve would give, and gain nothing special from a third.

**Pass (T2'):** ratio meaningfully above 1.71 and `eps_2` below the B^-2 null.
Helios PASSES this, on the values above.
**Pass (T2, the bn comparison):** ratio within a factor ~2 of the floor-subtracted 14-17.
Helios does NOT -- 3.3 against 14-17.

#### Where the remaining gap plausibly lives, and how to test it

The measured 2 -> 3 step is tiny (ln 0.198 floor-subtracted) because at 12 coils B = 2
already sits near the floor and B = 3's extra freedom is wasted against the binding length
cap -- HANDOFF's own "a constraint is expensive when the freedom acts on a much larger
scale than the constraint does". That is a BUDGET effect, and section 5b showed the proxy
is blind to budget by construction (N_c = 12 and 16 give identical eps to three digits).
So the discrepancy sits exactly where the proxy was already known to be useless.

**This is testable, not a just-so story: the 2 -> 3 step of the bn ladder must move with
budget, while eps cannot move at all.** We have B = 2 and B = 3 at 12 coils and **no arcB3
at 16 coils**. That is now the third independent reason to run it (the others: completing
the 16-coil ladder for T2 at all, and HANDOFF next-step 2).

Also still owed before any T2 verdict is final: the curves are on the surface, not mapped
to the 1.2 m offset (T1), and ~22% of `eps_1` is the I/G bias measured in T0c. Both
corrections push the ratio UP.

### T3 — Hinge placement (cheapest test with immediate practical payoff)

The DP returns optimal breakpoints as a by-product. Compare with the hinges the optimized arcB2 runs
converged to; verify the targets against the current files rather than the handoff text. **Pass:**
proxy breakpoints closer to the winning runs' hinges than to the losing runs', and closer than a
random pair of points on the same curve (bootstrap the null, 1000 draws).

Useful even alone: hinge placement is the dominant basin selector (44% bn spread across cut angles
at 12 coils) and `make_start.py --hinge-s` already takes explicit hinge parameters.

### T4 — Does `eps_1` track the principal-direction rotation rate? (promoted, user 2026-09-22)

`pdrot` is the best published predictor of coil non-planarity (Pavone & Warmer: Spearman 0.936 vs
max `eta_SVD`). It is not a rival to `eps_1` — it is the natural **validation partner**: if our
`eps_1` has content, it must correlate with `pdrot`.

The trick that makes this work on a single equilibrium: **do it across the foliation, not across
equilibria.** Both quantities become profiles in `Phi`:

- `eps_1(Phi)`, `eps_2(Phi)` per contour (~64 contours per field period);
- `pdrot_bar(Phi)` = the arclength-average of `|grad_S alpha|` along that same contour.

That is ~64 paired samples from Helios alone.

**Pass:** Spearman `rho(eps_1, pdrot_bar) > 0.7`, matching the sign and rough strength of the
published cross-configuration result.

**Negative control, in the same plot:** `rho(eps_1, L_gradB_bar)` along the contour, which Pavone &
Warmer measured at `|rho| < 0.2`. The discriminating outcome is **correlated with `pdrot` and NOT
with `L_gradB`** — that is our metric behaving like a genuine non-planarity measure. Correlated with
both, or with `L_gradB` only, means something is wrong.

Implementation traps, both real:
- Principal directions are unoriented lines, so `alpha` is defined mod pi and a naive gradient picks
  up branch jumps. Compute with the doubled angle: form the complex field `exp(2 i alpha)` and take
  `grad_S(2 alpha)/2` from it.
- At **umbilic points** (`kappa_1 = kappa_2`) the principal frame is undefined and `pdrot` diverges.
  Mask, and report the masked fraction and `min |kappa_1 - kappa_2|` alongside every `pdrot` number.

### T5 — Dynamic range

`eps_B` is dimensionless and normalized by the curve's own extent, so it may be nearly constant
across everything, in which case no correlation hunting helps. Establish the range first, over the
37 coilsets in `viewer_data.json` (they now span bn 3.80e-3 to 4.02e-2, a factor of 10.6 — a far
better spread than the nine I originally planned on). Known anchor: `fourierXYZ_16coils` is
`eps_1 = 0.089-0.29`, `eps_2 = 0.049-0.095`.

**Pass:** spread across the coilsets at least an order of magnitude larger than the T6 spread.

### T6 — Robustness (most likely to kill the idea quietly; run early)

Recompute the T2/T3/T4 signals while varying, one at a time: surface grid `M,N` (32/64/96);
contours in the foliation average (32/64/128); contour phase `phi0` over a full period; on-surface
vs offset-mapped; total vs vacuum (virtual-casing-subtracted) field for `K`; keeping vs dropping the
`I theta / 2 pi` secular term; `NS` and `BMAX` in `planarizability.py`.

**Pass:** the T2, T3 and T4 signals exceed the total spread over all of these.

### T7 — Weighting (expected inconclusive on Helios; run it anyway and say so)

Compare (a) uniform-in-arclength selection, (b) equal-`dPhi`, (c) `eps_B_bar^(p)` with `|K|^p`,
`p` in [0,2]. Response: the T1 distance and the T2 spectrum ratio.

**`|K_vc|` varies only 1.37x on Helios**, so (a) and (b) will nearly coincide. The honest conclusion
available here is "the weighting does not hurt"; showing it *helps* needs a configuration with
peaked `|K|`, which is a reason to look for one in Phase B.

---

## 7. SUPERSEDED by 5c — Phase B cross-configuration (T8 was run early; see 5c)

`vendor/desc/examples/` ships solved equilibria with strong priors on coil complexity:
`precise_QA`, `precise_QH`, `reactor_QA`, `W7-X`, `HSX`, `NCSX`, `ARIES-CS`, `ESTELL`, `WISTELL-A`,
`HELIOTRON`, `ATF`. `sandbox/common.py` loads by examples name. Compute `eps_1_bar`, `eps_2_bar`,
`pdrot`, `min L_gradB/a` for all of them plus Helios. Two payoffs, no GPU:

1. **Reproduction check.** precise-QA, precise-QH and W7-X are exactly the three cases Rodriguez &
   Sengupta validated on. If the `Phi`-contour construction does not reproduce their qualitative
   ordering, the implementation is wrong before any new claim is made.
2. **Rank test against folklore.** QUASR reports QA at moderate/high aspect ratio admitting very
   simple, near-planar modular coils while QH does not; W7-X, HSX and NCSX are strongly non-planar;
   HELIOTRON/ATF are continuous helical windings and should be outliers by construction.
   **Pass:** `eps_1_bar` separates known-simple from known-hard with no overlap.
3. Also record `max|K|/min|K|` per configuration, to find a proper test bed for T7.

This is also where `pdrot` gets its real test: 12 configurations is enough for a cross-configuration
Spearman against `eps_1_bar`, directly comparable to the published 0.936.

---

## 8. Order and cost

| step | cost | kills the program if it fails |
|---|---|---|
| T0 zero-checks | 20 min | yes |
| build `Phi`, contours, `eps_B_bar` | ~1 day coding | — |
| T6 robustness | 1 h | yes |
| T5 dynamic range | 1 h | yes |
| T2 B-spectrum | 1 h | yes |
| T4 `eps_1` vs `pdrot` | 2 h | yes |
| T1 shape agreement | 2 h | no, but weakens everything |
| T3 hinge placement | 2 h | no — useful even alone |
| T7 weighting | 1 h | no — expected inconclusive here |
| Phase B (T8 cross-config) | 3 h | yes, for the general claim |

All CPU. Nothing needs the 12 GB cgroup or the GPU.

## 9. After validation (sketch, not agreed)

- **E1, the correlation test.** Needs a *family*: >= 5 equilibria differing in the proxy at fixed
  aspect ratio, QA error, <beta> and coil budget. Per the estimates-over-exhaustive rule, run the
  full stage-2 pair (arcB2 + FourierXYZ reference) only on the two extremes and the cheap proxies on
  all members: ~3-4 h GPU instead of 10-12. Response variable is the **ratio** bn_arcB2/bn_free at a
  fixed budget, never bn alone.
- **E2, a seconds-per-equilibrium surrogate** so a real scan becomes possible: REGCOIL (merged, DESC
  PR #579, with current-potential -> coilset discretization) on a 1.2 m offset winding surface, cut
  coils, `eps_2`, project onto 2 planes, re-optimize currents only, `Delta B.n`. It will overestimate
  the cost (section 4); validate the *ranking* against the Helios ladder first.
- **E3, the single-stage experiment.** DESC PR #2285 (`js/surface-curve-consolidation`,
  `FourierRZSurfaceCoil` + `SurfaceCurveConsistency`). Cheapest useful variant first: coils
  constrained to a **1.2 m offset winding surface** with the arcB2 restriction imposed on the surface
  curve — one function's worth of freedom instead of three, and it needs no stage 1 at all. Then true
  joint boundary + surface-curve optimization with `eps_2` as a stage-1 objective, which is the
  experiment nobody has published.

## 9b. Code review of the objective (2026-09-22)

Reviewed `clamshell.py` line by line and tested the suspected failure modes rather than
asserting them. Two real bugs, one false alarm, all now regressions in `selftest`.

**REAL -- duplicate / adjacent breakpoints.** `eps_from_breaks([7,7])/eps_1` measured
**exactly sqrt(2)**: a zero-length segment made the whole curve count TWICE. Harmless under
a hard min (it would never be selected) but a softmin over candidates averages it in. Worse,
breakpoints merely CLOSE leave a segment with < 3 points, whose covariance is rank-deficient.
Fixed: `MIN_SEG = 3` and `_check_breaks` refuses both.

**REAL -- catastrophic cancellation in the DP cost table.** It used the textbook
`sum(x x^T) - sum(x)sum(x)^T/W` form on UNCENTRED points. At a coordinate offset of 1e4 the
answer was wrong by **116%** (eps_2 collapsed to eps_1). Real coil coordinates (R ~ 8 m,
extents ~2 m) are safe, but the failure is silent and it is a one-line fix: centre first.

**FALSE ALARM -- the "30% wrong gradient" on a degenerate in-plane pair.** I reported that
`eigvalsh` gradients corrupted `scale` for a near-circular curve (analytic 0.9959 vs FD
1.3895). **The analytic value was right and my finite difference was wrong.** Richardson
extrapolation on an independent reference implementation gives 0.99586736, stable across
h = 1e-3 .. 1e-7, matching the original analytic gradient. Recorded so it is not
re-litigated.

The rewrite that came out of it is still worth keeping, for a different and real reason:
Danskin now freezes the per-segment plane NORMALS as well as the segmentation, so the
differentiated path contains **no eigendecomposition at all** (the defect is
`sum_i ww_i ((x_i - mu) . nhat)^2`, polynomial in the points). That removes the genuine
`lambda_0 = lambda_1` degeneracy of a nearly STRAIGHT segment, which piecewise-planar-ish
curves will actually produce. `freeze()` now reports `gap_seg` and `gap_scale` so the
condition is monitored rather than assumed.

Still-open sharp edges, documented in the module: `jnp.clip(lam, 0)` has zero gradient at an
exactly planar segment, and `sqrt(tot/W)` has an infinite gradient there.

---

## 9c. DESC API gotchas hit while wiring the stage-1 driver (2026-09-22)

All of these cost a build failure or a wrong number. `stage1_clamshell.py` has them handled.

1. **`ObjectiveFromUser` hands you `data["x"]` in RPZ, not XYZ.** The clamshell metric is
   Euclidean, so applying the frozen plane normals to (R, phi, Z) triples silently gives a
   different number: measured **C = 3.330 instead of 1.372**. Convert with `rpz2xyz`.
   `|x_s|` is safe either way -- the rpz basis is orthonormal, so the norm is unchanged.
2. **Surface-curve compute needs `secular_theta` / `secular_zeta` as compute kwargs.** The
   curve's own `.compute()` injects them (`kwargs.setdefault`, curve.py:2949), but an
   objective calls `compute_fun` directly and does not, so the build dies with
   `KeyError: 'secular_theta'`. Pass `compute_kwargs={"secular_theta": 1, "secular_zeta": 0}`.
3. **`SurfaceCurveConsistency(source, curve)` is positional** -- there is no `eq=` kwarg.
4. **`jac_chunk_size` on `ObjectiveFunction` is only valid for `deriv_mode="batched"`.**
   With several things (eq + K curves) DESC picks "blocked" and raises. Chunk PER
   sub-objective instead.
5. **`GenericObjective` target length must equal its `dim_f`**, which is not necessarily
   `grid.num_nodes` (flux functions get compressed). Robust recipe: build the objective
   with no target, `compute` it, and use that as the target.
6. **No `proximal-*` optimizer is registered** -- see SPEC §4b.

---

## 9d. FIRST STAGE-1 RUN (v0, 2026-09-22): pace is fine, the AL is the problem

`runs/v0/lo.log`. precise_QA, arm `lo`, K=4 curves, mmax=2 (8 free boundary modes), 240
nodes, chunk 100, 15 iterations, RTX 5070, under the 12 GB cap.

**Pace: 8.02 s/iteration (120 s wall, "Solution time 1.74 min").** The 1462 equilibrium DOF
in the augmented Lagrangian are NOT prohibitive -- this is the same 7-10 s/it the arc coil
runs get, and those routinely run 250-400 iterations. A 400-iteration stage-1 run is ~53 min.

| | baseline | after 15 it | change |
|---|---|---|---|
| C (mean over 4 curves) | 1.0453 | 1.0292 | **-1.54%** |
| per curve | 1.3720, 0.9731, 0.8649, 0.9713 | 1.3142, 0.9715, 0.8611, 0.9701 | almost all of it is curve 0, -4.2% |
| eps_1 (mean) | 0.0670 | 0.0666 | -0.6%, held as intended |
| aspect ratio | 6.000 | 6.000 | exact |
| iota | 0.41970-0.42018 | 0.41990-0.42010 | held |
| QS max (normalized) | 2.500e-3 | 1.192e-3 | improved (it is a bound, so slack is fine) |
| SurfaceCurve consistency | 0 | 1.3e-16 .. 2.0e-16 m | exact |

### The AL outer loop never advanced, and force balance degraded 185x

Penalty parameter stayed at 1.000e+01 and multipliers at 0.000e+00 for all 15 iterations --
`n_outer = 1`, the same pathology HANDOFF records for the 16-coil repairs. Force balance
went from rms 7.606e-07 to 1.418e-04 normalized (max 22.5 -> 3550 N), because with
multipliers still zero ForceBalance is acting as a weak penalty, not a constraint.

### But that did NOT contaminate v0, and the reason is structural

Re-solved the optimized boundary to force balance and recomputed:

```
after 15 AL iterations   C mean 1.0292   |force| rms 1.418e-04
  ...then RE-SOLVED      C mean 1.0292   |force| rms 7.935e-07
C change that survives re-solve: -1.54%  (identical to 4 decimals)
```

**In v0, C is a pure function of the BOUNDARY.** The proxy curves lie on rho=1 at fixed
(theta, zeta), so they see only Rb_lmn/Zb_lmn; re-solving fixes the interior without moving
the surface. So the force-balance drift is cosmetic here and the -1.54% is real.

**This reprieve does NOT carry to the real version.** Once the streamline constraint is in,
the curves depend on B on rho=1, which depends on the interior solution -- and then a 185x
force-balance violation contaminates the curves directly. The SPEC §4b gate becomes
load-bearing exactly when v0 becomes v1.

### What to change next

1. **Run far longer.** 15 iterations is not a stage-1 run; the AL needs enough to advance
   n_outer. At 8 s/it, 400 iterations is ~53 min, which is affordable.
2. **Raise mmax to 3** (48 free modes vs 8). -1.54% on 8 modes is a floor, not a ceiling.
3. Only then is the gate ("does C move enough to be worth stage 2?") meaningful.

---

## 9e. v1 RUN: the AL outer loop never fires, and the objective target is the reason (2026-09-22)

`runs/v1/` (crashed on the second-order OOM) and `runs/v1b/` (rerun without it, stopped at
iteration 26). precise_QA, arm `lo`, K=16, M,N=32,64, n=480, mmax=3 (48 free boundary DOF).

| iteration | 0 | 2 | 9 | 26 |
|---|---|---|---|---|
| cost | 8.009 | 6.934 | 5.408 | **4.536** (-43%) |
| constraint violation | 1.26e-3 | 5.65e-2 | 6.29e-2 | **6.76e-2** (54x) |
| optimality | 1.008 | 1.301 | 0.336 | 0.247 |
| penalty / max\|multiplier\| | 10 / 0 | 10 / 0 | 10 / 0 | **10 / 0** |

**`n_outer` never advanced in 27 iterations.** `gtolk` is 1e-3 and optimality plateaus around
0.2-0.4, two to three orders above it, so the inner subproblem never "converges" and the
augmented Lagrangian never updates its multipliers. With multipliers pinned at zero the
constraints are only a fixed penalty of 10, so the optimizer happily buys a 43% cost
reduction by letting constraint violation grow 54x and sit there. **Any C from this run is
void** -- and unlike v0 it cannot be rescued by re-solving, because C now depends on the
whole interior solution.

### CORRECTION (same day): two things above are wrong

**(i) "Any C from this run is void" was an over-call, and so was stopping it.** In an
augmented Lagrangian the inner solve is SUPPOSED to reach optimality WITHOUT satisfying the
constraints; the outer update is what then tightens them. Constraint violation growing
during an inner solve is normal, not evidence of failure. And v1b was not stalled: cost was
still falling 8.9e-3 per iteration at step 26 and optimality had come down 1.008 -> 0.247.
**A true stall requires BOTH flat cost AND flat optimality.** That is the criterion from now
on (user, 2026-09-22).

**(ii) The root cause was SCALING, not the unreachable target.** Fixing the target alone
would not have helped much. Measured: the objective residual was O(1) for C but O(0.01) for
eps_1, while the NORMALISED force-balance violation started at 7.6e-7 -- seven orders apart.
A fixed penalty of 10 cannot balance that, so the inner solve could never reach `gtolk` and
`n_outer` stayed at 1.

**The fix: every weight is 1/tolerance, so a just-unacceptable violation reads as 1.0.**

| term | tolerance | weight |
|---|---|---|
| ForceBalance (normalized) | 1e-4 | 1e4 |
| AspectRatio | 1e-2 | 1e2 |
| iota | 1e-3 | 1e3 |
| eps_1 hold | 1% of baseline | 1/(0.01 eps_1,0) |
| QuasisymmetryTwoTerm | baseline rms | 1/qs_rms |
| **objective** | \|baseline - target\| | 1/\|baseline - target\|, so the residual is **exactly 1.000 at the start and 0 on target** |

**Result, immediately: the outer loop fired.** Penalty 10 -> 33.90, multipliers 0 -> 7.94e-2
at iteration 1, with `gtolk` adapting 1e-3 -> 1.8e-2 -- after 27 iterations of v1b and 15 of
v0 where it never moved at all.

### Secondary: the objective TARGET was also badly chosen

SPEC section 2 set `target = 0` for the `lo` arm ("minimise C"), which asks for a 100%
reduction -- unattainable, so the residual can never reach zero. Now `target = 0.85 * C0`
per contour for `lo` and `1.15 * C0` for `hi` (`--target-frac`). This is a real improvement
but it was NOT the blocker; the scaling above was.

### The user's fallback is now the right next move

Registered 2026-09-22: if this stalls, try simpler experiments -- optimize `eps_1` (planar
compatibility) first, and/or drop the QS constraint. That is now the **positive control**:
`eps_1` is a single quantity with no ratio and no 0/0 risk, and with QS dropped but iota
still pinned it cannot collapse to axisymmetry. If `eps_1` moves cleanly with the outer loop
advancing, the machinery is sound and the difficulty is specific to C or to the constraint
set. If it does not move even then, the problem is architectural.

---

## 9f. THE eps_1 CONTROL: a TRUE stall, and the cause is the architecture (2026-09-22)

`runs/ctrl_eps1/`. precise_QA, minimise eps_1 to 0.85x baseline, QS DROPPED, iota and aspect
ratio held, K=8, n=360, 48 free boundary modes, all weights = 1/tolerance. Stopped at
iteration 165.

| iteration | 0 | 5 | 10 | 85 | 165 |
|---|---|---|---|---|---|
| cost | 4.000 | 4.000 | 4.000 | **3.861** (min) | 3.967 (rising) |
| cost reduction | - | 1.7e-5 | 3.9e-4 | -7.1e-5 | **-1.8e-3** |
| constraint violation | 3.34e-2 | 3.31e-2 | 6.67e-2 | 1.489 | **2.189** |
| optimality | 1.5e-2 | 1.6e-1 | 8.7e-2 | 7.67 | 5.73 |
| penalty | 10 | 2.54e3 | **2.41e6** | 2.41e6 | 2.41e6 |
| max\|multiplier\| | 0 | 5.72 | 2.68e3 | 2.68e3 | 2.68e3 |

**True stall, on the agreed criterion**: cost has been RISING since iteration ~85, optimality
oscillates in 5-12 with no trend toward `gtolk` = 7.1e-2, and constraint violation grows
monotonically 3.3e-2 -> 2.19 (65x). `nfev/iteration` is 1.9, so the line search is failing
most steps. The good news is that the scaling fix worked exactly as intended -- `n_outer`
advanced twice in the first 10 iterations, which had never happened before.

### The entire constraint violation is force balance, and it is measured

Scaled violations at the STARTING equilibrium:

| constraint | rms | max |
|---|---|---|
| **ForceBalance (weight 1e4)** | 7.61e-3 | **3.34e-2** |
| AspectRatio (weight 1e2) | 3.21e-5 | 3.21e-5 |
| iota (weight 1e3) | 4.30e-14 | 8.33e-14 |

The observed initial constraint violation of **3.336e-2 is exactly the ForceBalance max**.
iota and aspect ratio contribute nothing. So the penalty explosion (10 -> 2.41e6 in ten
iterations) is entirely driven by the AL failing to reduce force-balance error, and once the
penalty is 2.4e6 the subproblem is so stiff that the trust region freezes at ~2.2e-4 and the
objective cannot move.

### CORRECTED: this was NOT architectural. It was an INFEASIBLE EQUALITY CONSTRAINT.

I first concluded the single-stage architecture was at fault -- boundary moves break force
balance, only interior DOF restore it, so proximal projection is needed. **That was wrong,
and the user caught it.**

`ForceBalance` defaults to `target = 0`, i.e. EXACT force balance. No equilibrium achieves
that; precise_QA sits at 2.07e-5 max normalized. So the constraint was **infeasible from
iteration 0**, its violation could never reach zero, and ramping the penalty was the only
move the AL had. My `weight = 1/tolerance` scheme then multiplied an already-unsatisfiable
residual by 1e4, making it worse. The `--tol-force` knob was a confusion of my own: it
conflated SCALING (conditioning) with FEASIBILITY (what counts as acceptable).

**The fix (user, 2026-09-22): constraints that mean "stay acceptable" are INEQUALITIES with
bounds and weight 1, never equalities with a target.**

| constraint | was | now |
|---|---|---|
| ForceBalance | `target=0`, weight 1e4 | `bounds=(+-1e-4)` normalized, weight 1 |
| AspectRatio | `target=6.0`, weight 1e2 | `bounds=(5.99, 6.01)`, weight 1 |
| iota | `target=iota0`, weight 1e3 | `bounds=(iota0 +- 1e-3)`, weight 1 |
| eps_1 hold | `target=E0`, weight 1/(0.01 E0) | `bounds=(0.99 E0, 1.01 E0)`, weight 1 |
| QS | bounds, weight 1/qs_rms | bounds, weight 1 |

Measured effect, same problem, K=4, mmax=1:

| | target=0, weight 1e4 | bounds, weight 1 |
|---|---|---|
| initial constraint violation | 3.336e-2 | **3.336e-6** |
| penalty after 10 iterations | 2.41e6 | **10 (unchanged)** |
| max\|multiplier\| | 2.68e3 | 5.75e-3 |
| cost | 4.000 -> 3.861 over 85 it | **2.000 -> 1.352 in 3 it** |
| eps_1 | -0.0% | **-2.59% in 3 iterations** |

The objective keeps `weight = 1/|baseline - target|` so its residual starts at exactly 1.000
-- that part of the scaling argument stands, and the user endorsed it. What does not stand
is using weights to express feasibility.

**Lesson worth carrying**: a constraint the starting point cannot satisfy will always look
like a stall, an architecture problem, or a conditioning problem. Check feasibility of every
constraint AT THE STARTING POINT before diagnosing anything else.

---

## 9g. THE CONTROL PASSES: eps_1 is drivable, nothing architectural is wrong (2026-09-22)

`runs/ctrl_eps1_bounds/`. Identical to 9f except every "stay acceptable" constraint is an
INEQUALITY with bounds and weight 1. precise_QA, minimise eps_1 to 0.85x baseline, QS
dropped, iota and aspect ratio bounded, K=8, n=360, 48 free boundary modes.

| iteration | 0 | 3 | **6** | 27 | 49 |
|---|---|---|---|---|---|
| cost | 4.000 | 2.917 | **1.9e-3** | 2.2e-3 | 1.5e-3 |
| constraint violation | 3.3e-6 | 4.5e-3 | 3.1e-3 | 2.2e-3 | 2.2e-3 |
| optimality | - | 2.139 | 5.6e-2 | 4.1e-2 | 3.3e-2 |
| penalty | 10 | 10 | 19.06 | 19.06 | 19.06 |
| max\|multiplier\| | 0 | 8.0e-3 | 1.6e-2 | 1.6e-2 | 1.6e-2 |

**Cost fell 4.000 -> 1.9e-3 in SIX iterations, a 99.95% reduction: eps_1 reached its 15%
target.** The same metric moved 0.0% in 85 iterations of 9f. The penalty crept 10 -> 19.06
and stopped; multipliers stayed at 1.6e-2. Everything after iteration 6 is polish --
constraints tightening 3.1e-3 -> 2.2e-3 while the objective sits on target.

**Conclusion: the machinery is sound and the metric is drivable. Nothing architectural was
ever wrong.** No proximal projection, no chained re-solves, no PR #2298. Every apparent
stall traced to `ForceBalance` posed as an infeasible equality (`target=0`), which my
`weight = 1/tolerance` scheme then amplified by 1e4.

### Three wrong diagnoses, recorded so the pattern is recognisable

1. "The objective target of 0 is unreachable, so it never stops pulling." Real but secondary.
2. "The objective and constraints differ by seven orders of magnitude in scale." Real, and
   fixing it did make `n_outer` advance for the first time -- but it did not fix the stall.
3. "It is architectural: boundary moves break force balance, only interior DOF restore it,
   so proximal is required." **Wrong.**

The actual fault was feasibility, and it was visible from iteration 0 the whole time.
**Check that every constraint is satisfiable at the starting point before diagnosing
scaling, conditioning, or architecture.**

---

## 9h. THE AUDIT CAUGHT A REAL CHEAT, and it forced proximal (2026-09-22)

`analyze_result.py` on `runs/ctrl_final/eq_lo.h5` (eps_1 control, single-stage AL).

**The metric was NOT gamed the ways I checked for.** eps_1 fell -14.87% on a FRESH DP
(frozen read -14.68%, correctly an upper bound as Danskin requires, gap +0.34%), and it fell
through the NUMERATOR: out-of-plane defect **-14.80%** while in-plane extent moved **+0.09%**
and contour length -0.03%. eps_B is a ratio, so inflating the denominator was the obvious
cheat; it did not happen. The proxy also stayed valid (monotonic, I/G ~ 1e-18, Newton
residual 1.7e-16, G unchanged to 4 digits).

**But the boundary barely moved.** |dR|max 7.4e-6 m and |dZ|max 1.3e-5 m -- **0.004% and
0.01% of the minor radius**. Volume unchanged to 5 digits. Meanwhile:

| | baseline | final | bound |
|---|---|---|---|
| force balance \|.\|max (normalized) | 3.34e-6 | **2.07e-3** | 1e-4 -- **VIOLATED 20x** |
| QS two-term rms (dropped) | 5.18e-4 | **3.96e-2** | **76x worse** |
| aspect ratio | 6.00000 | 5.99990 | OK |
| iota max change | - | 7.71e-4 | OK (1e-3) |

**The mechanism.** The free DOF included 1145 INTERIOR coefficients (R_lmn, Z_lmn, L_lmn),
and eps_1 depends on the interior solution through `K_vc = n x B / mu0` at rho = 1. Force
balance is the only thing tying interior to boundary, and as a soft AL constraint it was
never enforced (penalty froze at 19.06 after one outer update). **So the optimizer reduced
eps_1 by making the equilibrium stop being an equilibrium** -- far cheaper than reshaping
the boundary. That is a cheat, just one level below where I was auditing.

So the control did NOT show eps_1 is drivable by boundary shaping, and the C result from
that run (-0.25%) is measured on an invalid state and must not be used as the independence
check.

### This is the real argument for proximal, and it is not the one I made before

Earlier I claimed proximal was needed because "the AL cannot reduce force-balance error".
Wrong -- that was the infeasible-equality bug. The actual reason: **the interior DOF give
the optimizer a way to cheat, and proximal removes them from the problem** by SOLVING for
the interior at each step instead of penalising it. Fixing the interior DOF instead is not
an alternative -- then force balance cannot be satisfied at all when the boundary moves.

## 9i. PR #2298 VENDORED (proximal + augmented Lagrangian)

Branch `js/proximal-auglag`, merged the same way as #2285 (apply to vendor base 33076c84,
then 3-way merge). Net effect vs the pre-2298 vendor is **four files**:
`_constraint_wrappers.py` (495/320, the `ProximalState` class), `optimizer.py` (66/11),
`objectives/utils.py` (5/4), `optimize/__init__.py` (5/1).

Four conflicts, all resolved as OURS, and the reasoning matters:
`fmin_scalar.py`, `least_squares.py` and `utils.py` conflicts were purely the
`scale_columns -> scale_matrix` rename from #2239 (which the PR author flagged as transient
and which our vendor does not have). `aug_lagrangian_ls.py`'s one real change -- rescaling J
by `sqrt(mu/mu_old)` on an outer update instead of re-evaluating it -- **our vendored copy
already implements independently**, and ours additionally carries the stall-handling fix
that #2298 lacks. Verified afterwards: no dangling `scale_*` references anywhere, all files
compile, `check.py` passes, `polarB2_M5_cut30` re-evaluates to 1.044989e-2 unchanged.

**What it buys.** `_maybe_wrap_nonlinear_constraints` now splits equilibrium constraints
from the rest, wraps the objective AND the other constraints in a shared `ProximalState`,
and keeps the latter as genuine AL constraints. So `proximal-lsq-auglag` handles our QS/iota
/eps_1 bounds while the equilibrium is re-solved each step.

**Trap:** under proximal, `ForceBalance` must be an EQUALITY (`bounds=None`).
`_parse_nonlinear_constraints` only routes bounds-free equilibrium constraints to the
proximal path, and with bounds it raises "requires at least one equilibrium constraint".
This is correct -- proximal SOLVES force balance rather than penalising it, so the
infeasibility that broke the single-stage runs never arises. The bounds fix was right for
single-stage AL; `target=0` is right again under proximal.

**First proximal numbers** (K=4, mmax=1, 3 iterations, vs the same config single-stage):

| | single-stage AL | proximal |
|---|---|---|
| constraint violation at it 0 | 8.0e-4 | **0.000e+00** |
| eps_1 after 3 iterations | -2.59% | **-0.25%** |
| s/iteration | ~11 | ~25 |

Progress is ~10x slower per iteration because the optimizer can now only move the boundary.
**That is the honest rate; the earlier speed was the interior cheat.**

---

## 9j. PROXIMAL WORKS: the cheat is closed and the boundary actually moves (2026-09-22)

> **CORRECTED by 9n:** this run's -1.98% ceiling is blamed below on the tight iota bound.
> It was ASPECT RATIO (6 +- 0.01 here). A later run with loose iota and a different
> optimizer crawled identically.


`runs/prox_eps1/`, audited with `analyze_result.py`. Same problem as 9h, `proximal-lsq-auglag`,
K=8, n=360, 48 free boundary modes, 60 iterations, 240.6 s (**4.01 s/it**).

| | single-stage AL (9h, cheating) | **proximal** |
|---|---|---|
| force balance \|.\|max (normalized) | 2.07e-3 -- **20x over the 1e-4 bound** | **5.44e-6 -- OK** |
| boundary \|dR\|max | 7.4e-6 m (**0.004% of a**) | 1.57e-3 m (**0.92% of a**) |
| boundary \|dZ\|max | 1.3e-5 m (0.01% of a) | 1.74e-3 m (1.01% of a) |
| volume change | +0.00% | -0.69% |
| eps_1 | -14.87% | **-1.98%** |
| largest modes moved | tiny, scattered | `Z(0,-1)` -1.74e-3, `R(0,+1)` +1.57e-3 -- low order |

**The cheat is structurally gone.** Force balance is preserved because proximal SOLVES it,
and the boundary moved ~200x more than before. eps_1 fell honestly: -1.98% fresh DP vs
-1.92% frozen (correct Danskin ordering), with defect -2.30% against extent -0.34%, so the
numerator shrank faster than the denominator. Proxy still valid (monotonic, I/G ~ 1e-18,
Newton residual 1.7e-16, gap_seg 0.0093).

**The run is INCOMPLETE, not finished.** It ended on `maxiter`, and two constraints are
still outside: aspect ratio 6.02404 (bound 6 +- 0.01) and iota drift 3.36e-3 (bound 1e-3).
The penalty had only reached 25 with multipliers 3.56e-3 by iteration 60 -- the AL had not
had time to pull them back. QS degraded 50.8x, which is the expected price of dropping it.

**Behaviour to expect:** the run appeared to stall around iteration 56 (cost flat at 3.330,
45 cheap iterations in 2 minutes = steps being rejected, trust region collapsing) and then
the AL outer update fired (penalty 10 -> 25) and immediately unstuck it, 3.330 -> 3.107 in
one step. **Do not call a stall before the outer loop has had a chance to fire** -- this is
the same lesson as 9e, now seen positively.

**Rate:** -1.98% of a -15% target in 60 iterations, i.e. ~7x slower than the cheating run
suggested. At 4.0 s/it a 300-iteration run is ~20 minutes, which should both converge the
constraints and show how far eps_1 can actually go.

**C moved -0.11% while eps_1 moved -1.98%** -- ~18x apart, consistent with the independence
finding in 9h and this time measured on a VALID equilibrium.

---

## 9k. LOOSE BOUNDS: eps_1 IS strongly drivable, and C moves the OPPOSITE way (2026-09-22)

> **CORRECTED by 9n:** both headline claims are aspect-ratio artifacts. (i) The eps_1/C
> tension REVERSES (C -3.13%) when A is pinned at 6.00 +- 0.01. (ii) leg 2's iota drift of
> 8.01e-2 against a +-0.05 bound, logged here as an unexplained violation, is the `--from`
> re-centring leak, now fixed.


`runs/loose_eps1_L1/` then `runs/loose_eps1_L2/` (leg 2 restarts from leg 1 and RE-FREEZES
the DP). proximal-lsq-auglag, QS dropped, aspect 6 +- 0.25, iota +- 0.05, K=8, 48 free
boundary modes.

| | baseline | leg 1 (300 it) | leg 2 (200 it) |
|---|---|---|---|
| **eps_1** (fresh DP) | 0.06763 | 0.05071 (**-25.0%**) | **0.03959 (-41.5%)** |
| **C** | 1.0077 | 1.0916 (**+8.3%**) | **1.2675 (+25.8%)** |
| out-of-plane defect | 0.00793 m | 0.00588 (-25.9%) | 0.00469 (-40.9%) |
| in-plane extent | 0.11738 m | -1.2% | +0.7% |
| \|dR\|max | - | 7.8% of a | **17.8% of a** |
| force balance \|.\|max | 3.34e-6 | 2.33e-4 | **1.75e-3 (523x)** |
| iota drift | - | 5.15e-2 | 8.01e-2 (bound 0.05) |
| QS rms | 5.18e-4 | 583x | **1880x** |
| frozen-vs-fresh gap | - | 2.45% | 3.73% |

**1. eps_1 is strongly drivable: -41.5%, honestly.** Defect -40.9% against extent +0.7%,
so it is the numerator throughout. The -25.0% from leg 1 survived the leg-2 re-freeze to
five digits, so it is not an artefact of stale frozen data. The earlier -6.85% ceiling was
entirely the tight iota bound, not the metric.

**2. C moves the OPPOSITE way, and the coupling strengthens.** `|dC/d(eps_1)|` went 0.33
(leg 1) -> 0.62 (leg 2), both with opposite sign. Since C = 4 eps_2/eps_1, eps_2 falls more
slowly than eps_1: **optimizing for planarity makes the curves relatively LESS
clamshell-able.** The SPEC [ADV-3] design -- hold eps_1 fixed and move C -- assumed these
were separable. They are not; they are in tension. **Target C directly.**

**3. The AL outer update broke a plateau THREE times** (10->25, 10->40, 190->1840), each
time after 45-55 flat iterations. Never call a stall on this problem before the penalty has
stepped. Registered as a rule.

### The force-balance degradation is RESOLUTION, not convergence -- and my fix was wrong

I added `--solve-ftol/xtol/gtol/maxiter` to tighten the proximal re-solve, on the theory
that it was stopping short. **It was not.** Re-solving the saved equilibria with
`eq.solve(maxiter=100)` changes nothing:

```
leg1 as saved        |force|max 2.327e-4   eps_1 0.05071
  -> after eq.solve()           2.330e-4         0.05072
leg2 as saved                   1.746e-3         0.03959
  -> after eq.solve()           1.746e-3         0.03959
```

So 1.75e-3 is the best achievable AT THAT BOUNDARY with L=M=N=8. The optimized boundary is
far more shaped (|dR|max 17.8% of a) and the fixed spectral resolution no longer resolves
it. ### ANSWERED: part of the eps_1 gain IS under-resolution (`resolution_check.py`)

| resolution | \|force\|max | eps_1 | vs baseline | C | vs baseline |
|---|---|---|---|---|---|
| baseline (L=8) | 3.34e-6 | 0.06763 | - | 1.0077 | - |
| leg2 @ L=M=N=8 | 1.746e-3 | 0.03959 | **-41.46%** | 1.2676 | +25.79% |
| leg2 @ L=M=N=10 | 7.515e-4 | 0.04262 | **-36.98%** | 1.3655 | **+35.51%** |
| leg2 @ L=M=N=12 | 3.457e-4 | 0.04504 | **-33.41%** | 1.3840 | **+37.35%** |

1. **The eps_1 gain is NOT CONVERGED in resolution, and erodes monotonically:**
   -41.5% -> -37.0% -> **-33.4%**, losing ~4 points per step of 2 in L,M,N. Successive
   differences 4.48 then 3.57 (ratio ~0.8), so extrapolating the decay puts the converged
   value near **-27% to -30%**. The L=8 figure overstated the effect by about a third, and
   -33.4% is still an upper bound. An optimizer handed a fixed spectral resolution spends
   part of its effort in the part of the space that resolution cannot represent, and this
   is a clean measurement of how much.
2. **Force balance halves at each step** (1.75e-3 -> 7.5e-4 -> 3.46e-4) but is still ~100x
   the baseline at L=12. The optimized boundary is genuinely harder to resolve than
   precise_QA -- and that is the same fact as (1), seen from the other side.
3. **The C tension is ROBUST and CONVERGING**: +25.8% -> +35.5% -> **+37.4%**, decelerating.
   It strengthened under every check that weakened the eps_1 number, which makes
   "minimising eps_1 degrades clamshell-ability" the most solid result of the session.

**Protocol consequence: run STAGE 1 ITSELF at raised resolution, not merely re-measure
afterwards.** Re-measuring catches the error but does not avoid it: the optimizer has
already spent its effort in the unresolved space.

**Machine note, CORRECTED.** I first concluded L=12 "needs more than 10 GB" after an
uncapped-Jacobian run was OOM-killed. **Wrong** -- with `jac_chunk_size=100` it peaks at
**~3.6 GB** and finishes in 334 s. Two lessons: (a) chunk the re-solve's Jacobian before
concluding a resolution is out of reach; (b) `jac_chunk_size` on an `ObjectiveFunction`
needs `deriv_mode="batched"` (NOTES 9c item 4 -- hit twice, now commented at the call site).

**DO NOT use `tr_method="cho"` to save memory** (user, 2026-09-22). It forms J^T J, which
SQUARES the condition number -- unacceptable when the whole point is an accurately
converged equilibrium. `qr` is the DESC default and the memory must come from chunking.
The original OOM had nothing to do with the factorization method; it was already using qr.

---

## 9l. THE COIL VALIDATION: monotonic, but NOT a validation (2026-09-23)

> **CORRECTED by 9n:** the planar degradation (+5.9%) is equilibrium quality, not eps_1 --
> it becomes -0.45% at matched force balance. The ladder also carries an uncontrolled
> aspect-ratio change (A pinned at ~6.25 in every leg), which is the larger confound.


Six stage-2 runs, identical coil problem (absolute bounds, 16 coils, r/a 2.5 cold start,
`convex_excess: null`, `--vacuum`). `runs/coilval/`, driver `run_coil_validation.sh`.

| equilibrium | eps_1 | bn_planar | bn_free | ratio |
|---|---|---|---|---|
| baseline | 0.06763 | 6.110e-2 | 1.331e-2 | **4.590** |
| leg 1 | 0.05071 | 6.303e-2 | 1.575e-2 | **4.002** |
| leg 2 | 0.03959 | 6.470e-2 | 2.087e-2 | **3.100** |

`ratio ~ eps_1^0.73`, Pearson r = +0.97, monotonic. **I called this a validation. It is
not, and the user's question is what exposed it** ("are we really improving planar coils or
just making Fourier coils worse?").

```
bn_planar   +5.9%    planar coils got WORSE
bn_free    +56.8%    free coils got MUCH worse
ratio      -32.5%    driven ENTIRELY by the denominator
```

**We did not improve planar coils.** And the free optimum barely moved toward planarity
either: eps_1 of the optimised FourierXYZ COILS went 0.23044 -> 0.21868, **-5.1%** for a
-41.5% change in the equilibrium's eps_1. So the ratio fell neither because planar coils
improved nor because the free optimum became planar-like.

**Null hypothesis, untested:** making an equilibrium harder compresses all representations
toward a common floor, so the best one loses most and every ratio falls -- nothing to do
with planarity. Test it by degrading the baseline in an eps_1-unrelated way (random
perturbation, deliberate QS degradation, or `--arm hi`) to match `bn_free` +57% and seeing
whether the ratio drops anyway. Agenda in `HANDOFF.md` section 3.

**What the design DID get right:** on `bn_planar` alone the conclusion would have been
exactly backwards (it rises monotonically). Only the ratio exposed the structure, and only
the decomposition exposed that the ratio was moving for the wrong reason. Keep reporting
numerator and denominator separately, never the ratio alone.

**Confound still open:** QS was dropped for the whole ladder and degrades monotonically with
eps_1 (583x, 1880x), so "cost falls with eps_1" and "cost falls with QS error" are
indistinguishable here.

---

## 9m. THE FROZEN NORMAL, THE AL STALL, AND WHAT eps_1 ACTUALLY REACHES (2026-09-23)

> **PARTLY SUPERSEDED by 9n.** Sections 1-3 (the live normal, the AL stall, the soft
> ForceBalance and the resolution-gap table) stand. Section 4's "the converged eps_1
> reduction is about -27%" is a statement about A = 6.25; at fixed aspect it is ~-10%.
> "Converged" there means RESOLUTION-converged, not optimizer-converged -- no stage-1 run
> has ever reached a stationary point.


Three changes, each forced by the previous one. Net result: **the converged eps_1 reduction
under these constraints is about -27%, not the -41.5% the ladder recorded**, and it is
reachable in 12 iterations instead of 500.

### 1. The frozen normal was wrong for B = 1 (user's objection, and it was right)

`eps_from_frozen` held the best-fit plane normal from build time. For B >= 2 that is Danskin
and it is needed. For B = 1 there is no segmentation to freeze, and the docstring's
justification -- `eigvalsh` has a meaningless gradient on a degenerate eigenspace -- is about
lambda_1 vs lambda_2, the IN-PLANE pair. It does not apply to lambda_0.

**Measured: lambda_0/lambda_1 = 0.0074 .. 0.0150 on the baseline contours**, so lambda_0 is
simple by a factor of 60-130. It has to be: lambda_0/lambda_1 ~ eps_1^2 by construction. And
at the optimized geometry it is BETTER separated (0.0031 .. 0.0080) -- the live path gets
safer as eps_1 falls, not riskier.

**What freezing cost.** eps_1 is invariant under rigidly ROTATING a curve -- same curve, same
planarity -- but the frozen surrogate is not, and prices tilt as non-planarity:

| rigid tilt | eps_1 true | eps_1 frozen | inflation |
|---|---|---|---|
| 2 deg | 0.04623 | 0.04981 | 1.08x |
| 4 deg | 0.04623 | 0.05928 | 1.28x |
| 8 deg | 0.04623 | 0.08752 | 1.89x |

against a whole-ladder signal of 0.60x. The observed normal drift baseline -> leg2 was mean
0.53 deg, max 1.08 deg -- exactly where the penalty switches on. **A planar basin at a
different plane orientation was unreachable.**

**The fix** (`_eps1_live`, `_lam_min` with a custom JVP): `d lambda_0 / dM = v0 v0^T`, first-order
perturbation theory, no division by an eigenvalue difference, so it is safe even when the
in-plane pair is exactly degenerate. The denominator still goes through the matrix invariants,
so lambda_1 and lambda_2 are never separated. Cost: one symmetric 3x3 eigendecomposition per
contour. `quantity="eps1"` uses it by default; `--frozen-eps1` restores the old behaviour;
`quantity="C"` is untouched.

Validated: live == frozen at the freeze point to **7.7e-14**; rigid-rotation invariance to
**1.6e-13** at 1/5/20/90 deg; reverse-mode gradient vs central FD **1.5e-8** through the full
chain; exactly degenerate in-plane pair (lambda_1/lambda_2 = 1.000000000) grad vs FD **1.3e-11**,
finite. **The selftest had no rotation-invariance test -- that gap is what let this through.**

### 2. lsq-auglag stalls, and it is the AL, not the objective

The live-eps_1 rerun of leg 1 (identical settings) led the original on EVERY iteration, then
froze at ~105 and sat there for 195 iterations: cost identical to 4 digits, optimality pinned
at 1.018e+00, step norm decaying 1e-9 -> 4e-12, all steps ACCEPTED. Final -22.77% vs -25.02%.

**The AL could not escape**: the outer loop fires on `optimality < gtolk`, and optimality sat
at 9.3-10.0x gtolk for the entire run (gtolk is reset to ~optimality/10 at each outer update).
Constraint violation went nonzero at iteration 102 and froze; multipliers froze at 101.

**The objective was exonerated by measurement, not by argument.** At the stalled geometry:
gradient vs central FD **2.2e-8**, `|grad|_inf = 0.798` (large and nonzero), contour Newton
residual 1.6e-16, Phi_mod monotone, I/G 2.2e-18. The optimizer was computing 1e-12 steps
against a correct 0.8 gradient.

**Trap recorded:** T3 originally validated the gradient only at the BASELINE geometry. Always
re-validate at a displaced geometry before blaming the optimizer -- or the objective.

### 3. proximal-lsq-exact, soft constraints, and the soft ForceBalance

**The method string is `proximal-lsq-exact`.** There is no bare `lsq` in the registry.
`lsq-exact` reports `eq_constr=False, ineq_constr=False`, so `_maybe_wrap_nonlinear_constraints`
absorbs ALL nonlinear constraints into `ProximalProjection` -- which would silently swallow
aspect and iota, the constraints that block the axisymmetry slide. **So it is NOT a flag-only
change**: the "stay acceptable" terms must move into the `ObjectiveFunction` as bounded terms.
The bounds semantics (zero inside, quadratic outside) comes from the OBJECTIVE, not the AL.

They must also be SCALED. At `weight=1` a bounded iota term contributes 0.05 when a full
bound-width out, vs the clamshell objective's 1.0 per contour -- ~400x weaker, and iota drifts
freely. `--soft-weight W` (default 10) makes one bound-width of violation contribute W.
**Measured good:** aspect walked to its bound and stopped (6.250), iota settled 0.4% of a
bound-width outside in the worst case and inside in the rest.

**The soft ForceBalance term (user's suggestion) is the important one.** Proximal SOLVES force
balance, but between projections the boundary walks into shapes the spectral basis cannot
represent. Adding `ForceBalance` to the OBJECTIVE (in addition to the constraint proximal
solves, `--fb-weight`) removes those directions from the search rather than penalising the
result. Weight picked by measurement, not guess: dim_f = 5346, sum f^2 = 3.09e-9 at baseline
and 2.95e-4 at the degraded state, against an objective that starts at cost 4.000.

**Mechanism, seen live.** Without it the optimizer took ONE lunge of 7.17e-2 and its steps then
collapsed 100x to 5.9e-4 for the rest of the run. With it, steady 1.2e-2 steps and optimality
falling 3x faster over the first seven iterations. The unresolvable directions were ATTRACTIVE
to the bare objective.

### 4. THE RESULT: the resolution gap is a monotonic function of |force| at L=8

| fb-weight | eps_1 @ L=8 | \|f\| @ L=8 | eps_1 @ L=12 | \|f\| @ L=12 | **gap** | iters |
|---|---|---|---|---|---|---|
| 0 | -39.55% | 2.22e-03 | -28.88% | 3.41e-04 | **10.67 p** | 50 |
| 50 | -38.70% | 4.81e-04 | -33.74% | 9.37e-05 | **4.96 p** | 33 |
| 165 | -26.52% | 1.92e-04 | -26.27% | 3.13e-05 | **0.25 p** | 15 |
| 500 | -27.50% | 1.56e-04 | -27.21% | 2.01e-05 | **0.29 p** | 12 |

**How far the boundary drifts outside what the basis can represent is exactly how much of the
eps_1 gain is fictional.** This is 9k's "resolution, not convergence" finding made quantitative
and, for the first time, controllable.

**Three independent lines now agree on ~-27%:** 9k's resolution-ladder extrapolation
(-41.5 -> -37.0 -> -33.4, extrapolating to -27 to -30), and the two resolution-CONVERGED runs
here (-26.27% and -27.21%, gap < 0.3 p). **The ladder's -41.5% was never real.** w=50's
-33.74% is NOT converged either (4.96 p gap) and is the same object as leg 2's -33.4%.

w=500 reaches the converged answer in **12 iterations / 2.8 min**, with |force| 1.56e-04 --
equilibrium quality matching the AL runs -- versus 500 iterations across two legs.

### 5. Open: every lsq-exact run dies on a bad Gauss-Newton model

Three of the four (w = 50, 165, 500) ended on "**a bad approximation caused failure to predict
improvement**" at iteration 33, 15 and 12; the fourth (w=0) stopped earlier on DESC's default
`ftol=1e-2` before it could get there. **Note `--ftol/--xtol/--gtol` were never passed to the
outer optimizer before; DESC's defaults applied.** None of these runs reached a stationary
point (best optimality 0.0352, at w=50).

Survival is NOT monotonic in the weight (33, 15, 12 iterations for w = 50, 165, 500) -- three
points, no story fitted to them. Untried and cheap: under proximal the true cost involves a
nonlinear re-solve that the linearisation only approximates, so if `--solve-ftol` (1e-6) is
loose relative to the step, the cost is itself noisy and no model can match it. Tightening the
proximal re-solve is a different lever from anything pulled so far.

### 6. Consequence for the coil validation (9l)

The 9l ladder's three equilibria carry force-balance errors of 3.34e-06, 2.33e-04 and 1.75e-03
-- spanning 500x -- and eps_1 values now known to be inflated. **Coils were fitted to plasmas
of systematically decreasing quality, monotonically with eps_1.** That is itself a candidate
answer to HANDOFF 3.1's null: force-balance degradation IS an eps_1-unrelated degradation, it
was present, and `bn_free` rising 56.8% is what it would produce.

`runs/live_eps1_L1_lsqfb/eq_lo.h5` (-27.21% converged, |force| 1.56e-04) is the first
equilibrium suitable for a MATCHED-QUALITY comparison against the baseline coil runs.
Caveat: -27% is a smaller lever than 9l thought it had, so any effect is correspondingly
smaller and may not clear the noise.

---

## 9n. [RETRACTED TITLE -- see 9o] (2026-09-23, later the same day)

> **THIS SECTION'S CENTRAL CLAIM IS WRONG. See 9o.** The "eps_1 spends constraint slack"
> synthesis was built on a comparison with a TRUST-REGION-LIMITED run. Restarting that run
> reaches -32.9% with aspect AND iota both properly held, and it shows the same coil-complexity
> damage as every other run. The aspect and iota explanations are both dead. What survives from
> this section: the trust-region artifact (3), the no-run-has-converged table (4), the
> matched-quality coil numbers (5), the umbilic caveat (6), and the iota re-centring BUG (2) --
> which is real and worth fixing but was worth only 0.9 points, not the 23.5 claimed.


**Supersedes parts of 9m, and corrects 9j, 9k and 9l.** Read this before acting on any of them.

### The finding

**eps_1 has a small honest lever (~-10%) and a large dishonest one that consists of
relaxing physics constraints.** It has no cheap direction of its own: it buys reductions by
spending whatever constraint slack is on offer, and it is the SPENT part that damages the
equilibrium for coils.

| run | eps_1 (cum) | A | iota drift | \|f\| @L=8 | res gap | C | L_gradB/a **min** | pdrot*a **median** |
|---|---|---|---|---|---|---|---|---|
| `live_eps1_L1_aspin` | **-10.31%** | **6.010** | **+-0.008** | **1.61e-05** | **0.00 p** | **-3.13%** | **+7.77%** | **-9.26%** |
| `live_eps1_L1_aspin2` | -33.78% | 6.010 | **+0.06** | 1.03e-04 | 0.01 p | +14.16% | -25.81% | +12.02% |
| `live_eps1_L1_lsqfb` | -27.21% | **6.251** | - | 1.56e-04 | 0.29 p | +13.97% | -23.55% | +65.63% |

The run that spent NOTHING came out **better** on both literature coil-complexity predictors.
The two that spent something -- one aspect ratio, one iota -- came out ~-25% on min L_gradB
with pdrot up. **It does not matter which constraint is spent; the damage signature is the
same.** L_gradB/a min is Kappel's actual metric (the MINIMUM, not the median); pdrot median is
the robust statistic (see the umbilic caveat below).

### 1. Aspect ratio was the confound in 9k and 9l

`--bound-aspect 0.25` was used for every ladder run, and **A pinned itself at ~6.25 in every
single one** (6.2501, 6.2510, 6.2510, 6.2876, 6.2320). Pin it at 6.00 +- 0.01, change nothing
else, and:

* eps_1 reaches **-10.31% in 300 iterations** instead of -27.21% in twelve;
* **C REVERSES**, -3.13% instead of +8% .. +25%. 9k called the eps_1/C tension "the most solid
  result of the session" and it is an aspect-ratio effect;
* pdrot median goes **-9.26%** instead of +65.63%, and min L_gradB **+7.77%** instead of -23.55%.

**9j's -1.98% ceiling was misattributed.** It blamed the tight iota bound; that run also had
aspect at 6 +- 0.01. Today's run had LOOSE iota (+-0.05) and the same crawl, with a completely
different optimizer (`proximal-lsq-exact`, no AL). The variable was always aspect.

**Coil-complexity scaling, for why this is not surprising:** the dimensionless coil crowding is
`2 pi A / N`, so at fixed coil count higher A means relatively wider-spaced coils. 6.000 ->
6.251 is +4.2% crowding. Kappel's a-normalization (R^2 0.944 by `a`, 0.644 by R0) says the
minor radius is the natural length for everything coil-related, so changing A changes the whole
coil problem's scale.

### 2. The iota bound RE-CENTRES on `--from`, and that is a real leak

`iota0` was recomputed on the restarted geometry, so **every leg got a fresh +-`bound_iota` of
room**. Measured: a continuation took the entire new +-0.05 within 12 iterations (0.4122..0.4277
-> 0.4587..0.4823), cumulative drift **+0.06** from baseline -- larger than the bound ever
nominally allowed -- and bought -23.5 points of eps_1 with it.

iota is the anti-trivialization constraint (SPEC 3b: a vacuum stellarator with iota != 0 and no
net current MUST be 3D), so relaxing it is the cheapest possible way to flatten the contours.

**This also explains 9k's leg 2**, whose recorded iota drift of 8.01e-2 against a +-0.05 bound
was logged as a violation with no cause. It was this.

**FIXED** in `stage1_v1.py`: `iota0` is saved as `iota0_{arm}.npy` and reloaded under `--from`,
the same way `BASE0` already worked. Bounds now stay tied to the ORIGINAL equilibrium.

### 3. The 300-iteration crawl was a TRUST REGION artifact, not physics

| | initial trust radius | step norms |
|---|---|---|
| `aspin` (fresh) | 2.174e+00 | ~9e-05 |
| `aspin2` (restart) | **5.418e+02** (250x) | ~5e-02 (500x) |

The restart achieved more in **12 iterations** than the first leg managed in 300. A steady,
non-collapsing step norm was read at the time as healthy progress; it was a pinned trust
region. **Lesson: a flat step norm over hundreds of iterations is a trust-region diagnosis, not
a convergence one -- restart and compare the initial radius before believing a rate.**

(Caveat: that restart is also the leg that spent iota, so the two effects are entangled in
aspin2. The trust-radius numbers themselves are not in doubt.)

### 4. No stage-1 run has ever converged

| run | iters | final optimality | stop |
|---|---|---|---|
| AL, live eps_1 | 301 | 1.018e+00 | maxiter (frozen from ~105) |
| lsq, no FB | 51 | 2.322e-01 | `ftol=1e-2`, DESC's loose DEFAULT |
| lsq, FB w=500 | 13 | 4.089e-01 | bad approximation |
| lsq, FB w=165 | 15 | 5.150e-01 | bad approximation |
| lsq, FB w=50 | 34 | **3.518e-02** | bad approximation |
| lsq, aspect-pinned | 301 | 1.159e+00 | maxiter |

Every eps_1 figure here is a LOWER BOUND on what the metric could reach. Note optimality is NOT
comparable across runs with different `bound_aspect`: the soft aspect term's weight is
`soft_weight / bound_aspect`, so 40 when loose and **1000** when pinned.

### 5. The matched-quality coil test (9l's null, half-answered)

Identical coil problem to 9l, on the `lsqfb` equilibrium (resolution-converged, baseline-quality
force balance) with the existing baseline runs as control:

| | planar | xyz (free) | ratio |
|---|---|---|---|
| baseline | 6.1104e-02 | 1.3311e-02 | 4.590 |
| lsqfb (eps_1 -27.21%) | 6.0826e-02 | 1.5616e-02 | 3.895 |
| change | **-0.45%** | **+17.32%** | **-15.15%** |

* **9l's planar degradation was equilibrium quality**: +5.9% along the old ladder, **-0.45%**
  here. That part of 9l is an artifact, confirmed.
* **The ratio still falls entirely through the denominator.** Planar coils did not improve.
* fb500's free coils newly pin against kMS (baseline xyz: `kMS 0.818, active ['L']`), i.e. the
  surface demands more curvature from coils that had slack before.

**But this was run on the A=6.251 equilibrium**, so it inherits the aspect confound. The clean
test is `aspin` (-10.31%, A=6.010, iota held) and it has NOT been run.

### 6. Umbilic caveat on pdrot -- quote the MEDIAN only

`d alpha` carries a `1/(c^2 + s^2)` factor, so points near umbilics (where principal directions
are undefined) inflate pdrot without bound. Closest approach to an umbilic: **0.480** on
baseline, 0.358 (`aspin`), 0.310 (`aspin2`), **0.0125** (`lsqfb`). The maxima of 892 / 798 /
304264 in the earlier sweep are artifacts. The median is robust (a handful of singular points
out of 16384) and is the only statistic to quote. That the optimized surfaces APPROACH umbilics
is itself a real geometric finding.

`loose_eps1_L2` is unusable for this: L_gradB/a min = 0.0068 and pdrot max = 3e5, on an
equilibrium whose force balance is 1.75e-3. It is barely an equilibrium.

### 7. What is actually established, and what is not

**Established:** aspect ratio and iota each account for most of the eps_1 "gain" in every
previous ladder run; at genuinely fixed physics eps_1 moves ~-10% and the coil-complexity
predictors improve slightly; 9k's C tension, 9j's ceiling attribution and 9l's planar
degradation are all artifacts of those confounds.

**NOT established:** how far eps_1 goes at fixed physics (no run has converged); whether a
-10% lever produces any measurable stage-2 effect (untested -- the lever is a third of what
9l worked with, and 9l's own effect at -27% was only -15% in the ratio); whether eps_1 has
incremental predictive value over pdrot, which gets R^2 = 0.700 univariate against coil
non-planarity on 7500 devices.

The last of these is the cheapest and most decisive, and it does not need this sandbox: QUASR
and Constellaration ship equilibria WITH optimized coils, so eps_1 / pdrot / L_gradB can be
regressed against ACHIEVED coil complexity (length, curvature, eta_SVD) over thousands of
devices for the cost of reading files. That inverts 5c's failure, which was 8 hand-picked
configurations with no dynamic range.

---

## 9o. THE DOSE-RESPONSE IS IN eps_1 ITSELF (2026-09-23)

> **Not final -- see 9p.** The dose-response measured here was under PROPORTIONAL targeting,
> which subsidises the already-planar curves. With a worst-offender bound and an L_gradB floor,
> eps_1 -17.8% costs only 1.81% of min L_gradB (vs 25.81% here at -32.9%) and IMPROVES pdrot.
> The damage is a property of how eps_1 was being driven, not only of how far.

**Supersedes 9n's central claim.** Four runs, all resolution-converged, with the constraints
actually verified rather than assumed:

| run | eps_1 | A | iota (bound 0.41970 +- 0.05) | L_gradB/a **min** | pdrot*a **median** |
|---|---|---|---|---|---|
| `aspin` | **-10.31%** | 6.010 | 0.4122..0.4277 (inside) | **+7.77%** | **-9.26%** |
| `lsqfb` | -27.21% | 6.251 | - | -23.55% | +65.63% |
| `aspin3` | **-32.91%** | **6.010** | 0.4558..0.4733 (at bound) | **-23.55%** | **+13.52%** |
| `aspin2` | -33.78% | 6.010 | 0.4587..0.4823 (leaked +0.013) | -25.81% | +12.02% |

**The damage tracks the SIZE of the eps_1 reduction, not which constraint was spent.** `aspin3`
holds aspect at 6.010 AND keeps iota inside its baseline-referenced bound, reaches -32.91%, and
shows the same -23.55% on min L_gradB as the loose-aspect run. Both constraint explanations are
dead.

**The user's original hypothesis stands:** eps_1 may correlate with good planar coils, but
optimizing it produces equilibria that are harder for coils by both literature predictors --
min L_gradB down ~24% (Kappel: coils must come closer) and pdrot up (Pavone & Warmer's best
single predictor of coil NON-planarity, the thing we claim to be reducing).

Corroborated at stage 2 on `lsqfb` vs the baseline control, identical coil problem:
**planar -0.45% (unchanged), free coils +17.32% WORSE, ratio -15.15%.** Planar coils never
improved in any run, at any eps_1. The ratio moves only through the denominator.

### Why this took four wrong turns, and the lesson

The `aspin` run reached only -10.31% in 300 iterations and looked benign. I read that as "aspect
ratio was the confound". **It was trust-region limited**: initial radius 2.174e+00 vs 5.418e+02
on restart, step norms ~9e-05 vs ~5e-02, and a restart beat 300 iterations in 12. Its complexity
metrics looked good because it had barely moved eps_1, not because aspect was pinned.

**Rule: never attribute a difference between two runs to a constraint until the optimizer is
controlled for.** A flat, non-collapsing step norm over hundreds of iterations is a
trust-region diagnosis, not a convergence one. Restart and compare the initial radius BEFORE
believing a rate -- and before building a causal story on the difference.

### Is there a benign regime?

The -10.31% point is the only one with metrics improving (+7.77% / -9.26%), and the next point
is -27.21%. **A threshold somewhere in between is consistent with the data but NOT established
-- it rests on a single point from a run that stopped for optimizer reasons.** Intermediate
targets (`--target-frac` 0.85, 0.80) would test it cheaply and it is the one thing that could
still make eps_1 useful as a mild regularizer rather than an objective.

### Standing conclusions

* eps_1 is **not** a good optimization target beyond ~10%, and possibly not at all.
* Every previously recorded eps_1 number is from a non-converged run; all are lower bounds.
* The 9l ladder is confounded three ways -- force balance (500x), aspect ratio (A pinned at
  ~6.25 in every leg), and QS (dropped throughout) -- and its planar degradation is an artifact.
* The cheapest remaining test of whether eps_1 has ANY content is observational and needs no
  GPU: QUASR and Constellaration ship equilibria WITH optimized coils, so eps_1 / pdrot /
  L_gradB can be regressed against ACHIEVED coil complexity over thousands of devices. If eps_1
  adds nothing over pdrot (R^2 = 0.700 univariate against non-planarity), the question is closed.

---

## 9p. THE OBJECTIVE WAS ASKING FOR THE WRONG THING (2026-09-23, evening)

Two changes, both the user's, and together they give **better coils AND a better equilibrium** --
the first time in this program those have moved in the same direction.

### 1. The proportional target was subsidising the easy curves

`target_k = frac * base_k` asks every contour for the SAME FRACTIONAL reduction, so the
already-planar curves are explicitly ordered to flatten too -- and they are the cheapest to
move, so they dominate the least-squares descent. **A planar coilset is limited by its WORST
coil**, so that is backwards. Measured, change by baseline rank (easiest -> hardest quartile):

| run | mode | mean eps_1 | easiest | | | **hardest** | spread |
|---|---|---|---|---|---|---|---|
| baseline | - | - | - | - | - | - | 1.66x |
| `aspin` | proportional | -10.3% | **-28.8%** | -2.4% | -6.0% | -8.9% | 2.35x |
| `aspin3` | proportional | -32.9% | **-56.0%** | -16.4% | -24.9% | -37.1% | 3.43x |
| **`worst3`** | **worst** | **-17.9%** | -19.9% | -4.2% | -16.9% | **-29.6%** | **1.73x** |

Proportional STRETCHES the foliation (1.66x -> 3.43x); worst-mode preserves it (1.73x) and the
hardest quartile moves MOST. Also a manufacturing point nobody asked for: a real modular set is
built from a few unique coil types, so a foliation spanning 3.4x instead of 1.7x means the
coils become more unlike each other even at equal average planarity.

**`--target-mode worst`**: a single one-sided bound `(0, frac * max(base))`. Curves below it
contribute zero residual AND zero gradient. Deliberately NOT `loss_function="max"` (which the
vendored DESC supports): a hard max kinks whenever the argmax switches, the same failure
CLAUDE.md records for hard distance minimums. The one-sided bound is the C1 analogue.

### 2. The L_gradB floor works, and costs almost nothing

`GenericObjective("L_grad(B)", bounds=(floor, inf))` -- dim_f is PER NODE, so it is one-sided
per node. Floor = the ORIGINAL equilibrium's own min (tied across restarts like iota).

**Backoff must be 1.0, not 1.01.** At 1.01 the floor sits 1% above the baseline's own min, so
14/2401 nodes violate AT THE START and the term costs 4.0 -- as much as the entire eps_1
objective -- to correct a 0.7% grid error. Grid convergence measured: min L_gradB/a is 3.1788
(M=N=8) -> 3.1623 (M=N=64), so M=N=24 costs 0.7% and the "evaded between nodes" worry is ~1%,
not a loophole.

| run | eps_1 | **L_gradB/a min** | L_gradB/a med | **pdrot*a median** |
|---|---|---|---|---|
| `aspin` (proportional) | -10.3% | +7.77% | -3.35% | -9.26% |
| **`worst3` (worst + floor)** | **-17.8%** | **-1.81%** | -15.81% | **-11.85%** |
| `aspin3` (proportional) | -32.9% | -25.81% | -15.76% | +12.02% |

1.7x more eps_1 than `aspin` for 1.81% of min L_gradB, against the 25.81% proportional
destroyed. pdrot median IMPROVED. Resolution gap 0.04 p, |force| 3.609e-05, A 6.010.

### 3. The coils, same problem as 9n.5, baseline runs as control

| equilibrium | eps_1 | planar | xyz | ratio | dplanar | dxyz | **planar%/eps_1 pt** |
|---|---|---|---|---|---|---|---|
| baseline | - | 6.1104e-02 | 1.3311e-02 | 4.590 | - | - | - |
| `fb500` (A free, prop) | -27.2% | 6.0826e-02 | 1.5616e-02 | 3.895 | -0.45% | +17.32% | 0.017 |
| `aspin3` (A pinned, prop) | -32.9% | 5.2057e-02 | 1.6483e-02 | 3.158 | -14.81% | +23.83% | 0.450 |
| **`worst3`** (A pinned, worst+floor) | **-17.8%** | 5.5463e-02 | 1.5078e-02 | 3.678 | **-9.23%** | +13.27% | **0.519** |

Worst-mode converts eps_1 into planar performance more efficiently (0.519 vs 0.450 per point)
and moves the ratio more per point (1.12 vs 0.95). Free-coil damage per point is the SAME
(0.75 vs 0.72), so it is not buying the gain by hurting free coils harder.

**And it stays in the control's constraint regime**: `worst3` planar is `['L','kMS']` at
d_cc 2.235 (baseline `['L','kMS']`, 2.532) and `worst3` xyz is `['L']` (baseline `['L']`).
`aspin3` needed d_cc ACTIVE at 1.008 on planar and gained kMS on xyz -- so part of its
-14.81% was a regime change (crowding the coils onto the clearance bound), not a like-for-like
gain. This matters for how much of 9n.5's result to believe.

### 4. UNEXPLAINED, and it undermines single-run conclusions

**Only 1 leg in 3 was productive**, with identical configuration:

| leg | initial trust radius | result | stop |
|---|---|---|---|
| `worst1` | 4.637e+03 | -0.9% (max) in 300 iters | bad approximation |
| **`worst2`** | **1.354e+06** | **-16.62%** in 15 | **`xtol` -- a REAL convergence criterion** |
| `worst3` | 1.354e+06 | ~0% in 10 | bad approximation |

Legs 2 and 3 have the SAME trust radius and opposite behaviour, so the radius is not it.
Legs 1 and 2 have the same L_gradB floor status (0/2401 active) and opposite behaviour, so the
floor is not it either. **`--tr-ratio` does NOT help**: raising the initial radius 2500x
(1.855e+00 -> 4.637e+03) changed nothing, and leg 1's step at iteration 1 was 1.003e-03 against
a radius of 4637 -- the step was never trust-region limited, the Gauss-Newton step itself was
tiny. Four mechanisms were proposed for this today (proximal re-solve noise, initial trust
radius, the L_gradB floor's zero slack, a kink in the worst-mode bound) and ALL FOUR were
falsified within minutes of being proposed.

**Rule: do not conclude "configuration X does not work" from a single stalled run.** Restart it
first. Several conclusions earlier in this file were drawn that way and are unsafe.

The untried isolation: save `precise_QA` to .h5 and run leg 1's exact config with `--from` on
it. Same equilibrium, same objective, only the fresh-build-vs-restart code path differs.

### 5. Objective-block scaling is badly unbalanced (user's diagnosis)

Weights were chosen to equalise COST contributions. Gauss-Newton builds its step from J^T J, so
what matters is the JACOBIAN. Measured at the starting point, w.r.t. the full state:

| term | dim_f | \|r\| | \|J\|_F | share of \|J\|^2 |
|---|---|---|---|---|
| clamshell eps_1 | 8 | 2.155e+00 | 8.242e+04 | **0.003%** |
| force (w=500) | 5346 | 2.781e-02 | **1.409e+07** | **99.997%** |

The force block's residual is 77x SMALLER and its Jacobian 171x LARGER. Balancing Jacobians
instead of costs would put `--fb-weight` near **3**, not 500. CAVEAT: this is the PRE-proximal
Jacobian; `ProximalProjection` projects onto the tangent space where dF/dx is nearly zero, so
much of that dominance should be removed in the reduced space -- but any leakage is multiplied
by 500. **And there is a real conflict**: the resolution-gap control WANTS a large force weight
(w=500 -> 0.29 p gap, w=50 -> 4.96 p). The weight that fixes the physics is the one that
unbalances the model, which may mean the force term belongs somewhere other than the objective.

### 6. New flags

`--target-mode worst` | `--lgradb-floor/-weight/-grid/-backoff/-chunk` | `--tr-ratio` |
`--fb-chunk`. Viewer: `--streamlines K` (Phi contours, shaded by each curve's own eps_1) and
`--field-ref` (L_gradB heatmap, absolute or diverging ratio-vs-reference).

---

## 10. Framing to keep hold of

Helios' own architecture (planar encircling + 324 shaping coils) *sidesteps* this question, because
the planar coils never have to do the shaping. "Does Helios support planar coils alone" is already
answered — 4.04% vs 0.23%, no. The open question is the intermediate architectures, which is where
the arcB2 ladder lives.

## References

- Kappel, Landreman & Malhotra, *The magnetic gradient scale length explains why certain plasmas
  require close external magnetic coils*, PPCF 66 (2024) 025018. arXiv:2309.11342
- Kappel, Landreman, Jurasic & Henneberg, *How does the magnetic gradient scale length influence
  complexity of filamentary coils in stellarators?*, Nucl. Fusion 66 (2026) 066021. arXiv:2602.18974
- Pavone & Warmer, *The link between coil non-planarity and magnetic surface geometry in QI
  stellarators: a data-driven study*, arXiv:2604.26763
- Rodriguez & Sengupta, *Estimating coil features from an equilibrium*, arXiv:2604.12339
- *Optical analogy for stellarators: ridges as caustics and coils as singularities*, arXiv:2605.21814
- DESC PR #2285, `SurfaceCurve` class and related optimization features
- Kruger, Elder, Gates et al., *Planar coil design for the Helios stellarator fusion power plant*,
  Fusion Eng. Des. 230 (2026) 115893

## Files here

| file | what |
|---|---|
| `NOTES.md` | this |
| `lgradb_helios.py` | the section-2 L_gradB measurement (CPU, ~1 min) |
| `phi_contours.py` | the builder: `current_potential`, `modular_contours`, `to_surface_curve`, `proxy_coilset` |
| `t0_checks.py` | T0a/T0b/T0c; exits nonzero on failure |
| `run_helios_proxy.py` | the Helios first look in section 5b |
| `cross_config.py` | the section-5c sweep over the examples library (CPU, ~2 min) |
| `clamshell.py` | differentiable `eps_B` (Danskin) and `C = eps_2/(eps_1/4)`; `python clamshell.py` self-tests |
| `hinge_landscape.py` | the section-5e hinge-landscape measurement (CPU, ~1 min) |
| `foliation_spread.py` | C across the foliation; whether \|K\|-weighting matters (CPU, ~1 min) |
| `HANDOFF.md` | **state + agenda for the next agent**: stress-testing the eps_1 interpretation |
| `run_coil_validation.sh` | the 6 stage-2 runs (resumable, retries the CUDA fault) |
| `analyze_result.py` | audit an optimized equilibrium: constraints, real-vs-frozen eps_1, gaming checks |
| `view_{baseline,leg1,leg2}.html` | offline 3D viewers: plasma surface + planar + FourierXYZ + circles start |
| `SPEC_stage1.md` | the full stage-1 spec: things, objective, every constraint, and the adversarial pass |
| `stage1_clamshell.py` | the driver (`--arm lo/mid/hi`). **v0: curve `zeta_n` FIXED**, so it does not yet answer the science question |
| `runs/v0/` | first GPU run and its log |
| `contour_curve.py` | **v1**: proxy contours built INSIDE the objective, so they are streamlines by construction. `ContourClamshell` |
| `REVIEW_contour_curve.md` | adversarial review of the above: 5 bugs fixed, 4 config changes, 7 open risks |
