# Field quality of the actual coils: one complete example (2026-10-09)

**Question (user).** Take the single-stage output, then the coils, then field-line traces, and evaluate the QS error.
Does the coil field have good surfaces, and is its QS acceptable? What does this example do well and what does it do
poorly? Where should effort go next?

**Cases.**

| | coils | target equilibrium (single stage) | B·n |
|---|---|---|---|
| **level** (the example) | `cont/refine/result.h5`: arcB2 M5, 32 coils, per-coil level hinges, coil continuation | `cont/k4b/eq_k4.h5` | 2.266e-3 |
| **free** (reference) | `matrix/on_arcB2/arcB2/result.h5`: arcB2 M5, 32 coils, free hinges | `matrix/ss/arcB2_k4b/eq_k4.h5` | 1.484e-3 |

Both cases are vacuum, precise_QH-derived, NFP 4, with the same precise_QH bounds. The z = 0 case was traced only, and
its traces are poor (ρ spread about 0.04, edge ι 0.025 below target). It was not taken further.

## Summary

* **Surfaces are good.** In both cases every traced line stays on a clean nested surface for 400 field periods, with no
  visible islands or chaos. This holds out to lines started 6% outside the target boundary. Traced ι matches the
  target to ≤ 0.002.
* **QS of the actual coil field is 2–3.5× worse than the target equilibrium's.** For level, the QS error at ρ 0.5 is
  1.95e-3 against 6.9e-4. For free it is 1.73e-3 against 6.8e-4. In both cases the excess is about the size of B·n.
* **Level is about 10% worse than free at mid and outer radii, and about 2× worse in the core (ρ ≤ 0.3).** The level-hinge
  constraint costs little extra QS once both are traced; most of the gap to the target is present in both.
* **The excess has a clear source: field ripple from the 32 discrete coils.** This ripple appears as the n = 8 band,
  dominated by the (0, 8) mirror mode, and it does not decay toward the axis. It accounts for about 70% of level's
  squared symmetry-breaking at ρ 0.5, and about 30% of free's. The target has essentially none (≤ 4e-5).
* **Effective ripple (ε_eff, 1/ν transport) stays small**, below 0.35% everywhere. It is still 1.6–4× the target in level's
  core and 2–3.4× everywhere for free. Level's coil-field ε_eff at ρ 0.75–0.9 is lower than its target's.

![QS and eps_eff, level vs free](level_vs_free.png)

## Method: how the coil field was evaluated

**Step 1: trace and check in target coordinates** (`fieldcheck.py`).
* Coil currents are rescaled to match the target's mean |B| on the boundary. B·n is blind to the current scale, and the
  rescale does not move field lines.
* 13 field lines are traced for 400 field periods (Poincaré plots `*/poincare.png`).
* Punctures are mapped into the target's flux coordinates.

**Step 2: build the coil field's own equilibrium** (`tracedsurf.py`, `polish.py`). The target's coordinates are only an
approximation of the coil field (traced lines spread over 0.013–0.028 in the target's ρ), so the coil field gets its own
equilibrium.
* One line is traced from ρ 0.95. The resulting surface is fitted (FourierRZ M = N = 10; residual 0.02 mm mean,
  0.15 mm max).
* The target equilibrium is deformed onto this surface in 4 steps and re-solved at L = M = N = 12.
* In vacuum, a boundary determines the whole interior, so this equilibrium *is* the coil field's interior, up to fit
  and discretization error.

**Free-boundary solves were dropped.** `VacuumBoundaryError` reached stationary points with a residual floor, and its
equilibria matched the traces worse than the target did (`crosscheck.txt`).

**Validation.**

| check | level | free |
|---|---|---|
| force error (res 12) | 1.4e-4 | 3.2e-4 |
| traced lines' ρ spread in this equilibrium (ρ 0.1–0.9) | 0.0002–0.003 (target: 0.013–0.028) | 0.002–0.011 (target: 0.0006–0.021) |
| ι of this equilibrium vs trace | ≤ 3e-4 | ≤ 1.2e-3 |
| **control**: target's own ρ 0.95 surface through the same pipeline | QS within 1–4% of the target at every ρ | not run |
| **direct**: Boozer fit of the actual Biot–Savart \|B\| on this equilibrium's surfaces, vs the equilibrium's own \|B\| (`directcheck.py`) | breaking L2 agrees within 2% | within 6% |

**Error bars.** About ±5% on level's QS numbers. For free they are about ±25% inside ρ 0.6: its surfaces are less
accurate there (spread up to 0.01), and the free coils sit so close to their target that the method's own error matters.

## Results

### Surfaces and ι

| | level | free |
|---|---|---|
| lines confined (400 field periods) | 13/13, including starts at ρ 1.03 and 1.06 | 13/13 |
| islands / chaos | none visible | none visible. The core sits at ι ≈ 1.250 = 5/4, where lines close on themselves; no island width is resolved. |
| ι (trace) | 1.2058 (edge) – 1.2236 (axis) | 1.2417 – 1.2505 |
| ι trace vs target | ≤ 0.002 | ≤ 0.001 |
| nearest low-order rational | 6/5 = 1.200: 0.006 below the edge, 0.003 at the line started 6% outside | 5/4 = 1.250 in the core |

### QS error (Boozer, |B_nonsym| / ⟨|B|⟩)

The target is evaluated at ρ × 0.95, because the traced surface is the target's ρ 0.95 surface.

| ρ | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| level target | 5.1e-4 | 6.9e-4 | 9.0e-4 | 2.4e-3 |
| level control (method check) | 5.0e-4 | 6.6e-4 | 9.0e-4 | 2.4e-3 |
| **level coil field** | **1.80e-3** (3.5×) | **1.95e-3** (2.8×) | **2.86e-3** (3.2×) | **5.71e-3** (2.4×) |
| free target | 4.3e-4 | 6.8e-4 | 1.44e-3 | 3.9e-3 |
| **free coil field** | **9.8e-4** (2.3×) | **1.73e-3** (2.5×) | **2.65e-3** (1.8×) | **5.25e-3** (1.3×) |

Plots: `*/boozer_L12.png` (|B| in Boozer coordinates), `*/qs_profile.png`, `level_vs_free.png`.

### What breaks the symmetry (`modes.py`, `directcheck.py`)

| ρ 0.5 | level | free |
|---|---|---|
| breaking L2, coil field / target | 2.10e-3 / 8.9e-4 | 1.88e-3 / 8.4e-4 |
| largest mode | (0, 8) **−1.40e-3** (target −6e-6) | (2, 0) +1.21e-3 (target +2.4e-4), then (0, 8) −7.1e-4 |
| ripple band \|n\| 6–9, L2 (coil field) | 1.61e-3 (≈ 70% of the squared breaking) | 1.00e-3 (≈ 30%) |
| low-order \|n\| ≤ 5 vs target | ≈ 1.2e-3 vs 8.8e-4 | ≈ 1.5e-3 vs 8e-4 |

Two features stand out:
* **The ripple band does not decay toward the axis.** Level's ripple band is 1.25e-3 at ρ 0.25, against 3.8e-3 at the
  edge. This is mirror ripple on the axis from 4 coils per half period. In the Boozer plot (`level/boozer_L12.png`) it
  appears as closed |B| contours, i.e. ripple wells, inside the |B|-minimum band at both ρ 0.5 and the edge.
* **The free coils have fewer ripple wells**: none at ρ 0.5, small ones at the edge. Their extra error is partly
  low-order: (2, 0) at ρ 0.5, and (±4, ±2) and (2, 1) at the edge.

The **geometric** difference between the coil field's surface and the target's is about 2.4 mm L2 (2% of a), all in
low-order modes (|n| ≤ 2). The ripple is a large effect in |B| but tiny in shape.

### Effective ripple ε_eff

Computed with DESC `EffectiveRipple`. Resolution was checked: X = Y 32 → 64, Y_B 64 → 256, transits 20 → 40,
pitches 51 → 101 changed values by ≤ 5%.

| ρ | 0.25 | 0.5 | 0.75 | 0.9 |
|---|---|---|---|---|
| level target | 1.3e-4 | 2.7e-4 | 1.07e-3 | 1.93e-3 |
| level control | 1.3e-4 | 2.6e-4 | 9.4e-4 | 1.66e-3 |
| **level coil field** | **5.3e-4** (4×) | **4.2e-4** (1.6×) | **5.6e-4** (0.6×) | **1.29e-3** (0.8×) |
| free target | 2.0e-4 | 5.0e-4 | 8.8e-4 | 1.07e-3 |
| **free coil field** | **3.9e-4** (1.9×) | **1.51e-3** (3×) | **2.99e-3** (3.4×) | **3.44e-3** (3.2×) |

* ε_eff is not monotonic in the QS error. Level's coil field has more breaking but lower ε_eff at ρ ≥ 0.75. The
  breaking modes it picked up partly offset the target's own breaking in the trapped-particle orbits.
* In absolute terms everything is below 0.35%, well under the ~1% usually considered acceptable for a reactor
  (W7-X ~0.5–1%).

### Coil quality (from the dense check; ratios are value / bound)

| | level (example) | free (reference) |
|---|---|---|
| B·n ⟨\|B·n\|⟩/⟨\|B\|⟩ | 2.266e-3 | 1.484e-3 |
| length | 1.000 (binding) | 1.000 (binding) |
| curvature κ | 1.000 (binding) | 1.001 (binding) |
| κ_MS | 0.707 | 0.80 |
| d_cc (coil–coil) | 1.017 (just clear) | 1.24 |
| d_pc (plasma–coil) | 1.268 | 1.33 |
| hinge corner (max tangent jump) | 102° | 111° |
| demountable (vertical lift-out) | yes, +7.4 cm clearance | not applicable |

## Perspective

### What this example does well

1. **The full chain works and closes**, from single-stage boundary through stage-2 coils to a traced field, Boozer QS
   and ε_eff. Every link has an independent check: the control, the direct Biot–Savart fit, and trace vs equilibrium ι
   and ρ spread.
2. **Good, wide flux surfaces.** There are no islands, and good surfaces extend at least 6% beyond the design boundary.
   ι is reproduced to 0.002.
3. **The coil field is genuinely quasi-helically symmetric.** The QS error is about 2e-3 at mid-radius and ε_eff stays
   below 0.35% everywhere. That is far better than any QS experiment built so far (HSX), but it is not "precise QS".
4. **The demountability constraint is cheap where it matters.** Level hinges cost about 10% QS against free arcB2 at
   mid and outer radii, and 1.5× in B·n. The coils are removable with 7 cm clearance.

### What it does poorly

1. **It loses precise QS.** The coil field's QS error is 2.4–3.5× the target's. At this B·n (about 2e-3), the coils
   simply cannot reproduce a 7e-4-level QS target. In both cases the excess QS error is about the B·n value, which
   gives a rough rule of thumb from two cases: **coil-field QS ≈ target QS ⊕ B·n**.
2. **32-coil ripple dominates the excess for level.** The (0, 8) mirror term persists to the axis, roughly doubles core
   QS error and quadruples core ε_eff. This ripple is not targeted anywhere in the optimization. B·n weighs all
   Fourier content equally, and the target equilibrium has no ripple to match.
3. **The coils are tight against their bounds.** Length and κ are binding, d_cc is 1.7% clear, and the hinge corners
   are 102° tangent breaks. These are engineering-hard joints, inherited from the arcB2 representation (C0 arcs).
4. **The core near-rationals**: level's edge is 0.006 above ι = 6/5, and free's core sits at 5/4. Neither causes visible
   islands at this B·n, but neither is guarded.

### Where I would put effort next (in order)

1. **Target the coil-field |B| directly, especially the ripple band.**
   * Add a stage-2/refine term: Boozer breaking modes of |B_coil| on a few *target* surfaces (fixed points, in the
     target's Boozer angles; this is what `directcheck.py` computes). Weight it toward |n| 6–9 or just (0, 8) on the
     axis.
   * It is cheap: Biot–Savart on fixed points, with no equilibrium in the loop.
   * Expected payoff: up to about 70% of level's squared excess at ρ 0.5, and the core ε_eff.
   * Verify with this pipeline afterwards.
2. **Test coil count.** Mirror ripple falls roughly exponentially with coils per period. Run 5 coils per half period on
   precise_QH level (`run_al.py` already handles nc5 from the 1328095 work) through the same pipeline. Keep the length
   budget per coil in mind: length binds already.
3. **Price ripple against plasma–coil distance.** Level's coils sit closer (d_pc 1.27 vs 1.33 × bound), consistent with
   their larger ripple. A d_pc floor raised to about 1.35 is a one-flag experiment.
4. **Adopt the traced-surface pipeline as the standard evaluation** (`pipeline.sh`, about 45 min per case on this GPU),
   with the control once per target. Stop trying to make `VacuumBoundaryError` converge for this purpose.
   * Low-cost improvement: trace from ρ 0.9 instead of 0.95, to stay further from the edge.
   * Add the direct Biot–Savart check to the pipeline.
5. **Then continue the existing queue** (REPORT.md §6): continuation on free hinges for a like-for-like ratio, and the
   single-stage stall rule.
   * Not worth more effort now: equilibrium resolution (res 8 is fine, §7) and free-boundary debugging.

## Files (this directory)

| file | what |
|---|---|
| `fieldcheck.py`, `run.sh` | rescale, trace, (free-boundary), summary; `*/summary.txt`, `*/poincare.png` |
| `tracedsurf.py`, `polish.py`, `pipeline.sh` | traced-surface equilibrium (res 10 → 12): `*/eq_traced_L12.h5`, `*/tracedsurf.txt`, `*/polish.txt` |
| `control.py` | method control: `level/control.txt`, `level/control_L12.h5` |
| `modes.py` | breaking modes and boundary differences: `*/modes.txt` |
| `directcheck.py` | Biot–Savart \|B\| Boozer fit: `*/directcheck.txt` |
| `ripple.py` | ε_eff: `*/ripple.txt`, resolution check `level/ripple_hi.txt` |
| `figures.py` | `*/boozer_L12.png`, `*/qs_profile.png`; `level_vs_free.png` (both cases) |
| `crosscheck.py` | traces vs free-boundary equilibria (why those were dropped): `crosscheck.txt` |
