# SurfaceCurve — unified curve-on-surface plan

Unify the two base branches — **#844** (`FourierRZWindingSurfaceCurve`, Panici) and
**#819**/**#2081** (`FourierUmbilicCurve` + NFP-factor, Gaur / Singh) — into one coherent
curve-on-surface family, built on **PR1** (`CurveSurfaceConsistency`, already merged on
`curve-surface-consistency`).

Both branches solve the same problem — *a closed curve constrained to lie on a surface* —
from opposite ends. #844 has the embedding machinery (curve → surface → lab xyz, with
derivatives, a coil, and a working optimization) but only closes in one field period and
couples the surface with a fragile one-way mirror. #819/#2081 has the multi-period closure
and a clean abstraction, but stops at on-surface `(θ,ζ)` with no lab embedding. This plan
takes the embedding from #844, the `(m,n)` umbilic parametrization from #819/#2081
(corrected — see below), and the surface coupling from PR1.

## Settled decisions

- **Two subclasses over a shared base** (`FourierUmbilicCurve`, `FourierRZWindingCurve`).
  Whether the strictly-more-general winding parametrization can match umbilic optimization
  performance is an empirical question, kept open by shipping both.
- **Surface coupling follows PR1.** Each curve carries a read-only double-Fourier surface
  copy (`R_lmn`/`Z_lmn`) so it can `compute()` standalone, and is tied to a live source
  (`Surface` or `Equilibrium`) with `CurveSurfaceConsistency`. No mirror (#844), no
  objective-reaches-into-eq (#819).
- **Base renamed `SurfaceCurve`** (was `FluxSurfaceCurve` in #2081) — the source can be a
  standalone `Surface`, an equilibrium boundary, or an equilibrium interior flux surface,
  so "flux surface" is too narrow.
- **`N_scaling` / `NFP_umbilic_factor` is eliminated, not ported.** It was never on
  `master`; we simply do not carry over the `basis.py`/`grid.py`/`transform.py` changes
  from #819/#2081. See the next section for why it is unnecessary.

## The corrected parametrization

Both curves are the **same family**:

> **angle = (secular slope)·(parameter) + (NFP-periodic Fourier modulation)**

### Umbilic (arXiv:2505.04211, eq. 6 — Form A)

$$\theta(\zeta) = \frac{m\,\text{NFP}}{n}\,\zeta \;+\; \frac{1}{n}\sum_{k} a_k \sin(k\,\text{NFP}\,\zeta),
\qquad \zeta = \varphi,\quad \gcd(m,n)=1$$

- The `n = n_umbilic` factor lives **only** in the secular slope `m·NFP/n` and the `1/n`
  amplitude. The modulation `Σ aₖ sin(k·NFP·ζ)` is a **standard `FourierSeries(N, NFP)`**
  at frequency `k·NFP`.
- Closes after `n/gcd(n, NFP)` toroidal transits; `ζ ∈ [0, 2π·n/gcd(n,NFP))` over one loop.
- **Why no `N_scaling`:** #819/#2081 put the `/n` *inside* the modulation frequency
  (`k·NFP/n`), which is a misplacement. The umbilic edge is the high-curvature locus of an
  `NFP`-symmetric surface, so that locus is `NFP`-periodic — the modulation *must* have
  frequency `k·NFP`. Form B's `k·NFP/n` modes describe a curve whose shape varies across
  the `n` field periods of its closure, which cannot occur on a symmetric surface. So the
  entire `N_scaling` thread through the basis/grid/transform is not merely internalizable,
  it is unnecessary. (The paper prints `sin(NFP·ζ)` with the index `k` dropped inside the
  sine — a typo confirmed against the source.)
- `n = 1` is the ordinary (non-umbilic) flux-surface curve; it closes in one transit and
  `m·NFP` is an integer slope.

### Winding (#844, generalized)

$$\theta(s) = \sigma_\theta\,s + \sum_k \theta_k\,\mathrm{trig}(k\,s),\qquad
  \zeta(s) = \sigma_\zeta\,s + \sum_k \zeta_k\,\mathrm{trig}(k\,s),\qquad s\in[0,2\pi)$$

- Parametrized by an abstract loop parameter `s`; **both** `θ` and `ζ` are free functions.
- Integer secular terms `σ_θ, σ_ζ` set the topology ((1,0)=modular, (0,1)=toroidal,
  (1,1)=helical) and enforce closure. These are **fixed integer metadata, not optimizable**
  (fixing #844's live FIXME, where they were optimizable floats).
- Angle basis is **`FourierSeries(N, NFP=1)`** (frequencies `k`). Unlike #844, we drop NFP
  from the winding curve's own basis: NFP is not meaningful for a coil that need not be a
  graph over ζ, and NFP-in-basis structurally prevents winding across field periods.

**The only real differences** between the two: the umbilic is parametrized by `ζ` with a
rational θ-slope `m·NFP/n` and an `NFP`-periodic basis (physically required); the winding
is parametrized by `s` with integer slopes and an `NFP=1` basis. Everything downstream
(embedding, derivatives, length/curvature/torsion, surface coupling, coils) is shared.

## Class hierarchy & compute architecture

```
Curve (ABC, core.py)                       [existing]
  ├── length, curvature, torsion, frenet_* ← x_s, x_ss, x_sss   [existing, reused as-is]
  └── SurfaceCurve (ABC, core.py)          [NEW]
        ├── x, x_s, x_ss, x_sss, center    ← theta,zeta,theta_s…, R_lmn, Z_lmn (copy)
        │       (the embedding, lifted from #844)
        ├── FourierUmbilicCurve            ← theta, zeta, theta_s… from a_n, m, n
        └── FourierRZWindingCurve          ← theta, zeta, theta_s… from theta_n/zeta_n, σ
                └── FourierRZWindingCoil (_Coil mixin)
```

`SurfaceCurve` **subclasses `Curve`** so length/curvature/torsion/Frenet come for free once
it supplies `x, x_s, x_ss, x_sss` (verified: those generic funcs depend only on the
s-derivatives of `x`, `desc/compute/_curve.py:1077-1207` on master). The embedding is
registered **once** on the `SurfaceCurve` parameterization; each subclass registers only
its `theta`/`zeta`/`theta_s`/… as functions of its own parameter. Wiring via
`_class_inheritance` (`data_index.py:235`):

```python
"desc.geometry.core.SurfaceCurve":                      [Curve]
"desc.geometry.curve.FourierUmbilicCurve":              [SurfaceCurve, Curve]
"desc.geometry.curve.FourierRZWindingCurve":            [SurfaceCurve, Curve]
"desc.coils.FourierRZWindingCoil":  [FourierRZWindingCurve, SurfaceCurve, Curve, _Coil?]
```

Compute layers:

| layer | provides | consumes |
|---|---|---|
| `Curve` (existing) | `length, curvature, torsion, frenet_*` | `x_s, x_ss, x_sss` |
| `SurfaceCurve` (new) | `x, x_s, x_ss, x_sss, center` | `theta,zeta` + s-derivs, `R_lmn, Z_lmn` |
| subclass (new) | `theta, zeta, theta_s, theta_ss, theta_sss, zeta_s, …` | own params |

## Base `SurfaceCurve` API

State it owns (the PR1 "borrower" surface copy + closure metadata):

```python
class SurfaceCurve(Curve):
    _io_attrs_ = Curve._io_attrs_ + ["_R_lmn", "_Z_lmn", "_R_basis", "_Z_basis", "_NFP"]
    _static_attrs = [..., "_R_basis", "_Z_basis"]   # fix #844's missing _static_attrs

    # carried, read-only double-Fourier surface copy (the CurveSurfaceConsistency borrower)
    @optimizable_parameter @property
    def R_lmn(self): ...
    @optimizable_parameter @property
    def Z_lmn(self): ...
    @property
    def R_basis(self): ...       # DoubleFourierSeries, matches the source
    @property
    def Z_basis(self): ...
    @property
    def NFP(self): ...

    # abstract: subclasses define the on-surface angles vs. their parameter
    #   (registered as compute funcs, not python methods)

    # surface coupling (PR1 convenience)
    def surface_consistency(self, source, rho=None):
        """Return a CurveSurfaceConsistency tying this curve's copy to `source`."""
    def fix_surface(self):
        """Return FixParameters for the copy (the source-is-fixed case)."""
```

- The copy (`R_lmn`/`Z_lmn`, named per surface convention) is what `CurveSurfaceConsistency`
  binds — PR1 already accepts any borrower exposing `R_lmn`/`Z_lmn` + `R_basis`/`Z_basis`,
  so `SurfaceCurve` is a valid borrower with no PR1 changes.
- Built from a source at construction (`copy₀ = source₀`, starts on the constraint
  manifold).
- The embedding compute (`x`, …) evaluates the copy at `[ρ=1, θ, ζ]`: `R(θ,ζ)`, `Z(θ,ζ)`,
  `φ=ζ`, giving the curve position directly in `rpz` (lifted from #844's
  `_x_/_x_s_/_x_ss_/_x_sss_FourierRZWindingSurfaceCurve`).
- **No rigid transform** (`shift`/`rotmat`) — the source surface already positions the
  curve, so a rigid transform is redundant and unconstrained. Dropping it also removes
  #844's `rpz→xyz→rpz` round-trip and its two-`φ` subtlety (rotate in `xyz` using the
  *surface* `φ`, return using the *curve* `φ`), so `x_ss`/`x_sss` reduce to the pure
  chain-rule through the surface — simpler and easier to get right.

## Subclass APIs

```python
class FourierUmbilicCurve(SurfaceCurve):
    def __init__(self, source=None, a_n=[0], modes=None,
                 m_umbilic=1, n_umbilic=1, sym="sin", name=""):
        # NFP taken from source; assert gcd(m_umbilic, n_umbilic) == 1
        # UC/angle basis: FourierSeries(N, NFP, sym=sym)   # NFP, no N_scaling
        #   sym default "sin" => stellarator-symmetric umbilic locus (see decision 7)
        # a_n is the sole optimizable angle DOF
    # registers: theta(ζ)=… (Form A), zeta=grid ζ, + s(=ζ) derivatives
    # default grid spans ζ ∈ [0, 2π·n/gcd(n,NFP))  (base Grid, explicit nodes+spacing;
    #   NOT LinearGrid — see "Grid & closure"); compute() passes override_grid=False

class FourierRZWindingCurve(SurfaceCurve):
    def __init__(self, source=None, theta_n=[0], zeta_n=[0],
                 secular_theta=1, secular_zeta=0, modes_theta=None, modes_zeta=None,
                 sym_theta="sin", sym_zeta="sin", name=""):
        # angle bases: FourierSeries(N, NFP=1, sym=…)
        # secular_theta/secular_zeta are fixed integers (topology), NOT optimizable
        # theta_n, zeta_n are the optimizable angle DOFs
    # registers: theta(s), zeta(s), + s-derivatives (ports #844's _theta_/_zeta_ funcs)
```

Integer metadata (`m_umbilic`, `n_umbilic`, `secular_*`) go in `_static_attrs`.

## Surface coupling (PR1 tie)

- Typical umbilic: `source = Equilibrium`, `rho=1` (LCFS, proximal-safe) or `rho<1`
  (interior). Typical winding: `source = FourierRZToroidalSurface`. PR1's three regimes
  (separate `Surface` / eq ρ=1 / eq ρ<1) cover all cases with no new constraint code.
- The convenience `curve.surface_consistency(source, rho=…)` fans out over a `CoilSet` of
  such curves (PR1's deferred `broadcast_tree` fan-out lands here — a real collection
  borrower now exists to test it).
- `UmbilicHighCurvature` still takes the equilibrium separately (curvature needs full eq
  geometry, not just the copy); the curve supplies `(θ,ζ)` sample points.

## Grid & closure

- **No grid-class changes**, but the umbilic's default grid is a **base `Grid` with
  explicit nodes + spacing**, *not* a `LinearGrid`. It samples the whole loop
  `ζ ∈ [0, L)`, `L = 2π·n/gcd(n,NFP)`, with uniform nodes `ζᵢ = i·L/N_z` and explicit
  `spacing[:,2] = L/N_z`. The winding curve keeps its `LinearGrid(zeta=s)` over `s ∈ [0,2π)`.
- **Why not `LinearGrid` for the umbilic** (resolved — grounded in `grid.py:1336-1342`):
  a custom `zeta` array to `LinearGrid` (a) **hard-errors** once any `ζ > 2π/NFP`
  (`"LinearGrid should be defined on 1 field period"`), and (b) even suppressed, wraps
  `ζ mod 2π/NFP` and rescales `dz` by NFP → single-period weights, not loop weights. And
  the tempting "one period × transit count" shortcut is **wrong for n>1**: over one field
  period `θ` advances by `2πm/n` (not a multiple of 2π since `gcd(m,n)=1`), so the surface
  point doesn't return, the metric differs each transit, and `|x_s|` is *not*
  `2π/NFP`-periodic. The whole loop must be sampled.
- **`length` mechanism.** `length = Σ |x_s|·ds` with `ds = grid.spacing[:,2]`
  (`_curve.py`), so a correct loop length needs only correct `spacing[:,2]` weights — we
  *pass* uniform weights `L/N_z` (no bespoke quadrature; for a smooth closed curve
  uniform-node midpoint weights are spectrally exact). Integrate over the **whole loop**
  (decision 3).
- **The 0d-override trap.** `length`/`center` are 0d (`coordinates=""`), and
  `Curve.compute` (`core.py:152-156`) has a 0d path that, unless the grid is a `LinearGrid`
  with enough `N`, rebuilds a **single-period** `LinearGrid(N=2·self.N·NFP+5)` and computes
  the 0d quantity there — silently wrong for our loop. Fix: `FourierUmbilicCurve.compute`
  builds the loop `Grid` when `grid is None` and calls `super().compute(override_grid=False)`
  (our grid is already full-resolution over the loop, so nothing needs overriding). Its
  `compute()` override therefore does **two** injections: the `m_umbilic`/`n_umbilic`/`NFP`
  kwargs (mirroring the winding curve's secular kwargs) **and** this loop-grid default.
- Closure validation lives in the base / subclass `build`/`__init__`: `gcd(m,n)=1` and the
  `n/gcd(n,NFP)` transit count (umbilic); integer `σ_θ,σ_ζ` (winding).

## What we take / drop / fix per branch

**From #844 (take):** the embedding `x/x_s/x_ss/x_sss` chain-rule-through-surface, the
`_core.py` additions exposing surface `φ`-derivatives (`phi_ttt`, etc.) for `Surface`
parameterizations, the coil mixin, `from_values`, and the optimize test pattern.
**From #844 (fix):** add `_static_attrs`; verify the `x_ss`/`x_sss` formulas (marked
`# FIXME: check correct`); make secular terms fixed-integer metadata; drop the one-way
`R_lmn`/`Z_lmn` mirror in favor of the PR1 copy+constraint; drop `shift`/`rotmat` (see open
question); generalize beyond `FourierRZToroidalSurface` where the copy allows.

**From #819/#2081 (take):** the `(m,n)` umbilic concept, `gcd(m,n)=1`, the `theta` compute,
`from_values`, and the corrected `UmbilicHighCurvature` (normalize by minor radius, require
`eq.NFP == curve.NFP`, `theta` key). **Drop:** `N_scaling`/`NFP_umbilic_factor` everywhere
(basis/grid/transform); the `FluxSurfaceCurve` base (replaced by `SurfaceCurve` *with*
embedding); the `UmbilicCurve` core copy of `Curve`. **Fix:** modulation frequency
`k·NFP/n → k·NFP` (Form A); the `UC/a_n/umbilic/fluxsurface` naming seams.

## PR sequencing

Stacked on `curve-surface-consistency` (PR1, merged):

- **PR2 — `SurfaceCurve` base + winding curve.** The base ABC, the embedding compute, the
  PR1 surface-coupling convenience methods, `FourierRZWindingCurve`, and
  `FourierRZWindingCoil`. Fully testable on `master` primitives (a `FourierRZToroidalSurface`
  source). This is #844 re-expressed on the PR1 coupling — the biggest single piece.
- **PR3 — umbilic curve + objective.** `FourierUmbilicCurve`, the `theta` (Form A) compute,
  `from_values`, `UmbilicHighCurvature`, and the `CoilSet` fan-out of
  `CurveSurfaceConsistency` (PR1's deferred piece, now testable).

Rationale: PR2 stands alone with a clean surface source and no umbilic subtleties; PR3 adds
the umbilic specialization and the objective on top of a proven base.

## Tests

- **PR2:** winding curve construction / `change_resolution` / `from_values` (modular,
  helical, toroidal); `x`/length/curvature/torsion vs. analytic on the default surface;
  `x_s`/`x_ss`/`x_sss` vs. finite-difference (closes the #844 FIXME); the surface-copy stays
  bit-equal to the source under a short optimize (`CurveSurfaceConsistency`); coil length
  optimization (port #844's `test_optimize_FourierRZWindingSurfaceCoil_surface`); `save`/
  `load` round-trip.
- **PR3:** umbilic `theta` matches the Form A formula and its `NFP`-periodicity/wavelength
  (the test #2081 lacked — must use `a_n ≠ 0` and `n>1` so the modulation frequency is
  actually exercised, catching a `k·NFP/n` regression); `from_values` round-trip;
  `UmbilicHighCurvature` = `-1/a` on a circular boundary + no-NaN gradient; `CoilSet`
  fan-out keeps every member's copy tied.

## Resolved decisions (from review)

1. **Rigid transform** — omitted from `SurfaceCurve` (see base API). Bonus: also removes the
   `rpz↔xyz` round-trip / two-`φ` subtlety, simplifying `x_ss`/`x_sss`.
2. **Winding basis NFP** — `NFP=1` (full freedom; NFP is not meaningful for a winding curve).
3. **Umbilic `length` integration** — integrate over the **whole loop**, i.e. the full
   `ζ ∈ [0, 2π·n/gcd(n,NFP))` range spanning all toroidal transits. Implemented by a base
   `Grid` with explicit uniform nodes + `spacing[:,2]=L/N_z` and `override_grid=False`
   (not `LinearGrid` — it caps at one field period and mis-weights; see "Grid & closure").
4. **Naming** — `FourierRZWindingCurve` (shortened) for now.
5. **`x_ss`/`x_sss`** — carried from #844 but FD-verified during PR2; walk through what the
   derivatives *are* (chain rule through the surface) at that point. See note below.
6. **Source generality** — keep general; do **not** assume an `Equilibrium`. The curve's
   position already comes from the carried copy (source-agnostic), and `UmbilicHighCurvature`
   already accepts `Equilibrium` *or* `FourierRZToroidalSurface` (#819), so genericity is
   free. Only requirement: the source can supply the curvature the objective reads.
7. **Umbilic symmetry default** — the `UC` modulation basis defaults to
   `FourierSeries(N, NFP, sym="sin")`, **enforcing stellarator symmetry** of the umbilic
   locus (matches arXiv:2505.04211 and the winding curve's `sym_theta="sin"` default,
   `curve.py:1753`). Documented on the class. Expose `sym` to relax (`False`/`"cos"`) for a
   symmetry-broken source; revisit if a use case needs cos terms by default.

### Note on `x_s` / `x_ss` / `x_sss` (for PR2)

The curve position is `x(s) = surface(θ(s), ζ(s))`, so its s-derivatives are chain rule
through the surface:

- `x_s = e_θ·θ_s + e_ζ·ζ_s`  (surface tangents `e_θ = ∂x/∂θ`, `e_ζ = ∂x/∂ζ`)
- `x_ss = e_θθ·θ_s² + 2 e_θζ·θ_s·ζ_s + e_ζζ·ζ_s² + e_θ·θ_ss + e_ζ·ζ_ss`
- `x_sss` = the analogous third-order expansion (needs surface third derivatives `e_θθθ`, …)

`curvature` needs `x_s, x_ss`; `torsion` needs `x_s, x_ss, x_sss`. #844 marked the 2nd/3rd
forms `# FIXME: check correct` — many terms, easy to mis-assemble. With the rigid transform
gone the coordinate-frame complication is removed, so the remaining risk is purely the
term bookkeeping, which the FD test pins down.

## Implementation notes (PR2, in progress)

- **`SurfaceCurve(Curve)`** carries a private `FourierRZToroidalSurface` copy and delegates
  `R_lmn/Z_lmn/R_basis/Z_basis/NFP` to it (single source of truth + valid PR1 borrower).
  `shift`/`rotmat` are redefined without `@optimizable_parameter` so they drop out of the
  DOFs. `optimizable_params` on the winding curve is exactly
  `['R_lmn','Z_lmn','theta_n','zeta_n']`.
- **`#844`'s `x_ss`/`x_sss` were genuinely wrong** (stacked `[d2R, d2φ, d2Z]` as an rpz
  vector, dropping the rotating-frame centripetal/Coriolis terms). Reimplemented correctly:
  scalar coordinate derivatives via chain rule through the surface (`R_t…R_zzz`, all
  available for `Surface` on master), assembled into physical rpz-frame vectors with the
  cylindrical terms (`x_ss=[R''−Rφ'², Rφ''+2R'φ', Z'']`, and the analogous jerk). Since the
  copy is a `FourierRZToroidalSurface`, **φ=ζ exactly**, so `φ'=ζ_s` etc. — no phi
  derivatives needed. **FD-verified**: `x_s`~1e-9, `x_ss`~1e-8, `x_sss` O(h²)→0; a default
  modular curve gives `length=2π`, `curvature=1`, `torsion=0` exactly.
- **φ=ζ caveat**: the copy carries only `R_lmn/Z_lmn`, so it represents `φ=ζ` surfaces
  (all `FourierRZToroidalSurface`, standard `ω=0` equilibria). Tying to an `ω≠0` eq flux
  surface would need the copy to also carry `ω` (a future extension); #844 supported no eq
  sources at all, so this is strictly more general, not a regression.
- **Secular terms** (`secular_theta`/`secular_zeta`) are fixed-int metadata (not DOFs);
  they reach the `theta`/`zeta` compute funcs as **compute kwargs** injected by
  `FourierRZWindingCurve.compute`. This works in objective paths because `_CoilObjective`
  routes through `coil.compute(...)`. The CoilSet-batched path is to be verified in PR3.
- **DONE (commits 4bb991200, 38d4856f6 on `curve-on-surface`):** base + winding subclass +
  embedding (FD-verified) + `FourierRZWindingCoil` + `data_index` wiring + exports +
  `surface_consistency`/`fix_surface` (PR1 tie) + `from_values` (topology inference) +
  tests (`test_curves.py`) + `test_compute_everything` registration & golden-pickle merge +
  CHANGELOG/docs. Linters clean.
- **Stage-2 milestone validated:** 6 modular `FourierRZWindingCoil`s on a fixed winding
  surface, `QuadraticFlux` (vacuum), `FixParameters` fanned over the CoilSet holding the
  surface fixed -> B.n reduced 1.9x (max |B.n| halved) while surface & currents stayed
  fixed and coil shapes moved. Gradients flow through the embedding into B.n.
- **Local-test caveat:** `test_compute_everything` can't fully pass in this env due to a
  pre-existing `Gamma_c`/`nufft` numerical-noise mismatch on the `Equilibrium` (unrelated
  to this work; would fail on master here too). The winding registration + pickle are
  correct (both compute all 37 quantities; pickle preserves existing entries) and pass on
  a clean env.
- **Deferred to PR3:** CoilSet-batched secular-kwargs path; umbilic curve + objective +
  `CurveSurfaceConsistency` fan-out.

## Anchors

- PR1: `desc/objectives/linear_objectives.py` `CurveSurfaceConsistency`,
  `_surface_projection`; borrower needs `R_lmn`/`Z_lmn` + `R_basis`/`Z_basis`.
- #844 embedding: `upstream/dp/curve-winding-surface:desc/compute/_curve.py`
  (`_x_/_x_s_/_x_ss_/_x_sss_FourierRZWindingSurfaceCurve`), `:desc/compute/_core.py`
  (`phi_ttt` + `Surface` parameterizations), class `:desc/geometry/curve.py:1139`,
  coil `:desc/coils.py:796`.
- #2081 umbilic: `origin/NFP_fac_testing:desc/geometry/fluxsurfacecurve.py`,
  `:desc/compute/_fluxsurfacecurve.py` (`theta`/`UC`), objective
  `:desc/objectives/_geometry.py` (`UmbilicHighCurvature`).
- Generic curve compute (reused): `master:desc/compute/_curve.py:1077-1207`
  (`frenet_*`, `curvature`, `torsion`, `length`).
- `_class_inheritance`: `master:desc/compute/data_index.py:235`.
