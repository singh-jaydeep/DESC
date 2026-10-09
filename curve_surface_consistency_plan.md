# CurveSurfaceConsistency — implementation plan

A linear equality constraint that ties the read-only surface copy a curve carries to a
live source surface, so the copy and the source collapse to a single DOF in the reduced
optimization space (via `LinearConstraintProjection`). This lets a curve `compute()`
standalone (it owns the surface params) while staying exactly on the surface during
optimization.

Use cases: a curve on an equilibrium boundary (divertor, ρ=1), on an interior flux
surface (ρ<1), or on a separate winding surface; and coilsets of such curves.

## Scope

- **Explicit, not auto-injected.** The user adds the constraint, like `ShareParameters` /
  `FixParameters`. DESC auto-injects only *intra-thing* constraints (`getters.py:340-344`);
  cross-thing ties are user-added.
- **Specific, not general.** One purpose-named constraint modeled on
  `BoundaryRSelfConsistency` (`linear_objectives.py:446-551`), which likewise builds its
  own projection matrix for one concrete pairing. A general "link any params of any two
  objects" class is deferred until a third borrow case exists.
- **Parameter consistency, not geometry.** It constrains the surface a borrower *carries*
  to equal the source. It does **not** enforce that an arbitrary curve's locus lies on a
  surface (that is nonlinear and out of scope). A plain curve (`R_n`/`Z_n`, `X_n`/`Y_n`/`Z_n`)
  has no `R_lmn`/`Z_lmn` and is rejected at `build`.

## When you need it (single curve)

| Surface is… | Add | Notes |
|---|---|---|
| **Fixed** | `FixParameters(curve, {"R_lmn": True, "Z_lmn": True})` | Copy is definitive; removed from `x` via `remove_fixed_parameters`. No new constraint. |
| **Free & shared** (separate `Surface`/`eq` also optimized) | `CurveSurfaceConsistency` | The reason this constraint exists. |
| **Free & standalone** (copy is the only surface, free to move) | nothing | Copy is a legitimate free DOF. |

Fixing a subset of one thing's params is standard: it is exactly how `FixBoundaryR` fixes
`eq.Rb_lmn` while the interior `R_lmn` of the same thing is optimized. An objective flag
*cannot* remove DOFs — objectives read params, they do not shape `x`; only a linear
constraint (or not declaring a param optimizable) removes columns.

## When you need it (coilset)

A `CoilSet` is one thing whose members are self-contained, so each member carries its own
surface copy — a coilset of N curves has **N copies** in `x`. Untied, they drift onto N
different surfaces. One constraint fixes this for the whole set (fan-out via
`broadcast_tree`), never one per member:

| Surface is… | Add (one, fans out) |
|---|---|
| **Fixed** | `FixParameters(coilset, {"R_lmn": True, "Z_lmn": True})` |
| **Free** | `CurveSurfaceConsistency(coilset, surface)` — surface must be its own thing |

A coilset has no standalone-free variant: `CurveSurfaceConsistency` ties to a *source
thing*, and a member is not a thing, so a shared free surface must be a separate object
that anchors all copies.

Forgetting the tie silently deforms/separates the surface (same failure mode as forgetting
`FixBoundary`); mitigate with a convenience method (below) and docs.

---

## PR structure

- **PR1** — the constraint. Fully testable with plain `Surface`/`Equilibrium` things; a
  `Surface` is a valid borrower fixture (it exposes `R_lmn`/`Z_lmn` + `R_basis`/`Z_basis`).
- **PR2** — the curve-on-surface class that provides a real borrower.

Branches: `master` → `curve-surface-consistency` (PR1) → `curve-on-surface` (PR2).

## The three linear regimes

All are linear in the optimized params — one constraint with an optional matrix `A`:

| Source | Target key | `A` | Under proximal |
|---|---|---|---|
| Separate `Surface` | `R_lmn`/`Z_lmn` | identity | fine (untouched) |
| Eq boundary, ρ=1 | `Rb_lmn`/`Zb_lmn` | identity | fine (Rb/Zb retained) |
| Eq interior, ρ<1 | `R_lmn`/`Z_lmn` | `zernike_radial(ρ)` | error (R_lmn stripped) |

- A surface at fixed ρ is linear in the spectral coeffs: `R_mn(ρ) = Σ_l
  zernike_radial(ρ,l,m)·R_lmn`. `A` is intrinsic for eq sources (Fourier–Zernike volume →
  double-Fourier surface) and identity otherwise.
- Proximal strips the eq interior args `R_lmn, Z_lmn, L_lmn, Ra_n, Za_n` from the outer
  vector but retains `Rb_lmn, Zb_lmn` (`_constraint_wrappers.py:_set_eq_state_vector`), so
  ρ=1 works under proximal and ρ<1 does not.
- The borrower references the **equilibrium object** (a thing), never `eq.surface` (a side
  object not in `things`, not updated during a solve, `equilibrium.py:1539`).

Correctness reuses `ShareParameters`' machinery: `combine_args` aligns the cross-thing
constraint's columns to the full state vector (`utils.py:324` →
`assert A.shape[1] == xp.size`, `:98`); `factorize_linear_constraints` builds the null
space `Z`; source and copy columns collapse to one reduced coordinate; both objectives'
gradients sum through `Z`; equality holds exactly at every iterate.

---

## PR1 — the constraint

New class in `desc/objectives/linear_objectives.py` near `BoundaryRSelfConsistency`
(~line 446), which it parallels. Cross-thing (`[source, borrower]`), taking R and Z
together.

### API

```python
class CurveSurfaceConsistency(_Objective):
    """Constrain a borrower's stored surface copy to match a source surface.

    Enforces  A @ source[R_lmn|Rb_lmn] - borrower[R_lmn] = 0  (and same for Z): the
    read-only double-Fourier copy the borrower carries equals the live source surface
    (identity), the eq boundary at rho=1 (identity), or the eq flux surface at rho<1
    (A = zernike_radial(rho)). Linear.

    Parameters
    ----------
    borrower : Optimizable
        Object carrying the surface copy: a curve-on-surface, a CoilSet of them, or (for
        testing) a Surface. Must expose R_lmn / Z_lmn and R_basis / Z_basis.
    source : Surface or Equilibrium
        Object supplying the reference surface.
    rho : float, optional
        Equilibrium source only. 1.0 (default) -> boundary DOFs Rb_lmn/Zb_lmn (identity,
        proximal-safe); rho < 1 -> interior R_lmn/Z_lmn via A = zernike_radial(rho). Must
        be None/1.0 for a Surface source.
    """
    _scalar = False
    _linear = True
    _fixed = False
    _units = "(m)"
    _print_value_fmt = "Curve-surface consistency error: "
```

`things = [source, borrower]` (fixed order; `compute` relies on it). No key/matrix
overrides — the pairing is fully determined by source type and ρ.

### build

```python
def build(self, use_jit=False, verbose=1):
    source, borrower = self.things
    from desc.equilibrium import Equilibrium

    if isinstance(source, Equilibrium):
        if self._rho is None or self._rho == 1.0:
            self._keys = {"R_lmn": "Rb_lmn", "Z_lmn": "Zb_lmn"}
            self._A = {"R_lmn": None, "Z_lmn": None}
        else:
            errorif(not (0 < self._rho < 1), ValueError, "rho must be in (0, 1].")
            self._keys = {"R_lmn": "R_lmn", "Z_lmn": "Z_lmn"}
            self._A = {"R_lmn": _zernike_projection(source.R_basis, borrower.R_basis, self._rho),
                       "Z_lmn": _zernike_projection(source.Z_basis, borrower.Z_basis, self._rho)}
    else:
        errorif(self._rho not in (None, 1.0), ValueError,
                "rho only applies to an Equilibrium source.")
        self._keys = {"R_lmn": "R_lmn", "Z_lmn": "Z_lmn"}
        self._A = {"R_lmn": None, "Z_lmn": None}

    # borrower must carry a surface
    errorif(any(k not in borrower.dimensions for k in self._keys), ValueError,
            "borrower must carry a double-Fourier surface (R_lmn/Z_lmn); a plain curve "
            "does not — did you mean a curve-on-surface?")
    # per-key dim/shape check (ShareParameters precedent, linear_objectives.py:372-391);
    # identity => dims match, matrix => A.shape matches. Broadcast R/Z selection over
    # collection members. Accumulate self._dim_f.
    super().build(use_jit=use_jit, verbose=verbose)
```

`_zernike_projection(src_basis, dst_basis, rho)` extracts the matrix build in
`BoundaryRSelfConsistency.build` (`linear_objectives.py:507-528`): rows = `dst_basis`
(m,n), cols = `src_basis` (l,m,n), entries `zernike_radial(rho, l, m)` placed by matching
(m,n). It also enforces the ρ<1 resolution requirement (borrower (m,n) must be
representable in the eq boundary (m,n) set).

### compute

```python
def compute(self, params_source, params_borrower, constants=None):
    # [source, borrower] order (objective_funs.py:808-811). For R and Z, subtract
    # (A @ source[src_key]) from the borrower copy; broadcast over collection members.
    ...
```

### Details

- **Jacobian**: constant `[+A | −I]` (identity when `A is None`), zeros elsewhere after
  `combine_args`. Factored exactly.
- **`A` storage**: dense np arrays built once in `build`, one per R/Z, reused across
  fanned-out members (as in `BoundaryRSelfConsistency`).
- **Fan-out**: `broadcast_tree` ties every member of a `CoilSet` borrower to the single
  source.
- **No resync**: the borrower's copy is built from the source, so `copy₀ = source₀` and
  the initial state is already on the constraint manifold. If a source is modified after
  the borrower is built, a resolution change errors at the dim check and a value change is
  resolved by the projection — matching `BoundaryRSelfConsistency`.

### Proximal guard

ρ<1 under proximal would otherwise fail silently (factorize strips the eq interior
columns, `utils.py:82-95`). Guard at the existing proximal branch in
`get_combined_constraint_objectives` (`optimizer.py:~637`):

```python
if isinstance(objective, ProximalProjection):
    for con in linear_constraints:
        if isinstance(con, CurveSurfaceConsistency) and objective._eq in con.things:
            for src_key in con._keys.values():
                errorif(src_key not in objective._args, ValueError,
                    f"CurveSurfaceConsistency ties '{src_key}' of a proximally-projected "
                    "equilibrium, solved internally rather than as an outer DOF. Use a "
                    "non-proximal optimizer or rho=1.")
```

### Usage

```python
from desc.objectives import CurveSurfaceConsistency

CurveSurfaceConsistency(coilset, winding_surface)     # separate surface
CurveSurfaceConsistency(divertor_curve, eq, rho=1.0)  # eq LCFS (proximal-safe)
CurveSurfaceConsistency(curve, eq, rho=0.7)           # eq interior (non-proximal)
```

### Errors

- Identity link, mismatched dims → `build` (source resolution changed since the borrower).
- ρ<1 with a Surface source, or ρ ∉ (0,1] → `build`.
- Borrower missing `R_lmn`/`Z_lmn` (or, for ρ<1, the basis) → `build`.
- ρ<1 under proximal → the proximal guard.
- Source not in `things` → `factorize_linear_constraints` already warns (`utils.py:64-68`).

### Tests (`tests/test_linear_objectives.py`, `tests/test_optimizer.py`)

A `Surface` serves as the borrower fixture — no curve needed:

1. **Separate surface** (two `Surface`s, matching resolution): `factorize_linear_constraints`
   reduces `dim_x` by the linked count; short optimize keeps the copy bit-equal; consistent
   start leaves the source unchanged, inconsistent start snaps to the nearest consistent
   point.
2. **ρ=1** (`Equilibrium` → `Surface`): identity to Rb/Zb; copy tracks the boundary; works
   under a `ProximalProjection` objective.
3. **ρ<1** (`Equilibrium` → `Surface`): built `A` matches a `zernike_radial(ρ)` reference;
   `A @ eq.R_lmn == surface.R_lmn` after solve.
4. **ρ<1 under proximal** → raises.
5. **Resolution mismatch → raises; rho on a Surface source → raises; plain curve → raises.**
6. **Collection fan-out**: one constraint on a `CoilSet`-like borrower yields one block per
   member, all tied to the source; short optimize keeps every member's copy equal.
7. **Gradient**: one objective reads the copy, another the source; the reduced gradient
   through `LinearConstraintProjection.grad` sums both (finite-difference check).

### Files (PR1)

- `desc/objectives/linear_objectives.py` — `CurveSurfaceConsistency`, `_zernike_projection`.
- `desc/objectives/__init__.py` — export.
- `desc/optimize/optimizer.py` — proximal guard.
- `tests/test_linear_objectives.py`, `tests/test_optimizer.py`, `CHANGELOG.rst`, docs API.

`zernike_radial`, `broadcast_tree`, `tree_leaves`, `tree_map`, `errorif` are already
imported in `linear_objectives.py`.

### Non-goals (PR1)

- No general linker, no auto-injection.
- No curve class / coilset convenience (PR2).
- No ρ<1-under-proximal support (explicit error).

---

## PR2 — curve-on-surface (sketch)

A `Curve` subclass owning a read-only double-Fourier copy of its host surface so
`Curve.compute()` works standalone; the user (or a convenience method) adds a
`CurveSurfaceConsistency` to keep the copy tied to the live surface.

- **Class** `FourierRZCurveOnSurface` (or `CurveOnSurface`), subclass of `Curve`.
  - Params: the curve's own DOFs plus a mirror `R_lmn`/`Z_lmn`. Naming the mirror
    `R_lmn`/`Z_lmn` (surface convention) lets the constraint find it with no config.
  - Exposes `R_basis`/`Z_basis` for the mirror (ρ<1 projection).
  - Mirror set from the source at construction (starts consistent).
  - Read-only wrt the surface.
- **Convenience** (highest-value ergonomic): `curve.surface_consistency()` /
  `coilset.surface_consistency()` returning the fanned-out constraint, and a
  `fix_surface()` for the fixed case — so the tie is one call and hard to forget.

Open questions:

1. Curve parametrization on the surface (θ(s), ζ(s)); own DOFs vs. derived.
2. How the mirror basis is chosen and kept matched to the source boundary basis.
3. Whether `compute` needs surface quantities beyond `R_lmn`/`Z_lmn` (NFP, sym).
4. Divertor objective(s) consuming this at ρ=1 — targets, grids.
5. How a `CoilSet` of surface-curves exposes one shared surface vs. per-coil surfaces.

---

## Anchors

- `BoundaryRSelfConsistency` (template; builds `A = zernike_radial(ρ)`, arbitrary ρ via
  `surface_label`): `linear_objectives.py:446,507-528`.
- `ShareParameters` (cross-thing tie, `broadcast_tree`, dim check):
  `linear_objectives.py:246,365-391,432`.
- Auto-injected constraints are intra-thing: `getters.py:340-344`.
- `FixParameters` / `remove_fixed_parameters` (partial-thing fixing): `linear_objectives.py:74`,
  `utils.py:418`. `FixBoundaryR` is the precedent.
- Proximal retains Rb/Zb, strips interior args: `_constraint_wrappers.py:_set_eq_state_vector`;
  special-cased in `optimizer.py:~637`.
- Column alignment: `combine_args` `utils.py:308-328`; `factorize_linear_constraints`
  `utils.py:13`, `:98`; thing-not-in-objective warning `:64-68`.
- `eq.surface` is a side object: `equilibrium.py:1539`. `broadcast_tree` fan-out: `utils.py:691`.
