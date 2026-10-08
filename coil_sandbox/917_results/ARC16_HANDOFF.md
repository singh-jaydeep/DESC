# 16-coil arcB2 on Helios: results, files, drivers

Written 2026-09-21 as a reference for an agent picking up the next piece of work:
**a constraint that forces hinges to lie at prescribed locations, as if the joints were
demountable.** Section 6 is written specifically for that task; sections 1-5 are the context.

Read alongside:
- `coil_sandbox/CLAUDE.md` — machine rules (cgroup cap, one GPU job at a time), the recipe and
  why it is the way it is. **Read this first; it is binding.**
- `coil_sandbox/work/helios/HANDOFF.md` — the full chronological record including the 12-coil
  work. Longer and more detailed than this document.

---

## 1. What the problem is

Stage-2 coil optimization on `sample_equilibria/helios_repro.h5` (Thea Energy's Helios QA
reactor reproduction; NFP 2, R0 7.96 m, a 1.77 m, finite beta 2.6%). 16 modular coils = 4 unique
per half period, stellarator symmetric.

Coils are **polar piecewise-planar arcs, B=2** (`PolarPlanarArcCoil`): two planar arcs joined at
two C0 hinges. The question being answered is how close such a restricted, buildable coil can get
to unrestricted 3D coils (`FourierXYZCoil`).

Bounds are Kruger et al. Table 2 (`bounds/helios_kruger2026.json`): L <= 36 m per coil,
kappa <= 1.5 /m, d_cc >= 1.2 m, d_pc >= 1.2 m, convexity excess <= 1e-3 rad.
Field error metric `bn` = <|B.n|>/<|B|> with the plasma's own field included (virtual casing).

## 2. Results

All paths relative to `coil_sandbox/`. All of these are in the viewer (section 4).

| coilset | bn | max corner | file |
|---|---|---|---|
| FourierXYZ, 16 coils (unrestricted reference) | 3.7991e-3 | n/a | `917_results/fourierXYZ_16coils.h5` |
| arcB2 M=10, convexity OFF (not paper-feasible) | 6.4227e-3 | – | `work/helios/arc16/noconvex_L36/result.h5` |
| arcB2 M=10, corners free | 6.8770e-3 | 144.5 deg | `work/helios/arc16/M10_L36/result.h5` |
| **arcB2 M=10, corners <= 100 deg — BEST BUILDABLE** | **7.2181e-3** | **100.0 deg** | `work/helios/arc16/M10_c100/result.h5` |
| arcB2 M=5, corners free (length ladder) | 7.7549e-3 | 117.1 deg | `work/helios/arc16/warm_L36_paper/result.h5` |
| arcB2 M=5, corners <= 100 deg | 7.7913e-3 | 100.0 deg | `work/helios/arc16/dual_L36_c100/result.h5` |
| arcB2 M=5, session's starting point | 8.7195e-3 | 108.6 deg | `work/helios/arc16/cut30/legB/result.h5` |
| planar N=7, 16 coils (baseline) | 3.3448e-2 | n/a | `work/helios/planar16/stage2/result.h5` |

**Use `work/helios/arc16/M10_c100/result.h5` as the starting point for new work** unless you need
M=5 for a cleaner comparison, in which case use `work/helios/arc16/dual_L36_c100/result.h5`.

Caveat on the best file: its run was killed by the 12 GB cgroup cap at iteration ~294 after it had
converged (bn flat 7.2179/7.2181/7.2184e-3 over iterations 225-275). `result.h5` there is a copy of
`ckpt_it0250.h5`. It is converged and paper-feasible; it just was not written by the solver's own
final-write path.

### The five findings that matter

1. **The restriction costs 1.9x** (7.2181e-3 vs 3.7991e-3), and part of what looked like a shape
   penalty is actually convexity: the FourierXYZ reference ran with `convex_excess: null` because
   convexity is undefined for non-planar coils, so the comparison was never like-for-like. Dropping
   convexity from arcB2 buys 17.2%.
2. **Path dependence is large.** Starting from long crowded coils drives the optimizer to sacrifice
   one coil (down to 5.3 MA of ~20) and lands 11% worse. Starting from SHORT coils and growing into
   the length cap keeps all four coils loaded and is strictly better on the same bounds and the same
   conductor. See section 3, "the ladder".
3. **Corners are cheap at M=5, not at M=10.** Capping hinge turn angles at 100 deg instead of 117
   costs 0.5% at M=5 but 5.0% at M=10. Raising M still buys 7.4% under the cap.
4. **Extra modes cannot relieve the plasma pinch.** Each arc is
   r(phi) = |chord|/2 + sum_m a_m sin(m pi phi), and every mode vanishes at phi = 0, 1, i.e. AT THE
   HINGES. Three of four coils have their closest approach to the plasma exactly at a hinge, so the
   new-mode displacement there is 0.0 mm at any M. **This is the single most relevant fact for the
   demountable-joint work: hinges are where the coils are pinched against the plasma bound.**
5. **Constraint sensitivity is weak.** Relaxing d_pc 1.2 -> 1.0 m gains nothing (the coils decline
   the room); relaxing d_cc 1.2 -> 1.0 m gains 3.2%. Length is the only strongly active constraint
   in most runs.

## 3. Drivers — what generated what

Everything runs with `/home/singh/miniforge3/envs/desc-env2/bin/python`, one GPU job at a time,
under `systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`.

| driver | what it does |
|---|---|
| `run_one.py` | **the single optimizer entry point.** Every result above came from it. `--mode stage2` or `--mode feasibility`; every knob is a flag (`--help`) |
| `make_start.py` | builds a start: circles, or arcs fitted to an existing coilset (`--from-coils`, `--rep polararc --B 2 --M 5`), hinge placement via `--cut-angle` or `--hinge-s` |
| `work/helios/arc16/run_cold.sh` | cold start: circles -> polar-arc fit -> feasibility repair -> free stage. `BND=`, `LABEL=`, `CUT=`, `RA=`, `IT=` overridable |
| `work/helios/arc16/run_dual_ladder.sh` | the length+corner ladder (28/60 -> 30/70 -> 32/80 -> 34/90 -> 36/100), warm-started rung to rung, resumable |
| `work/helios/arc16/run_corner_ladder.sh` | corner-only ladder (60/75/90/105 at L=36). **Built but never run** |
| `work/helios/arc16/run_sensitivity.sh` | the d_cc / d_pc / L shadow-price runs |
| `work/helios/arc16/run_chain.sh` | the older fit -> repair -> legA(hinges held) -> legB chain. Superseded: leg A is neutral, see HANDOFF |
| `work/helios/arc16/raise_M.py` | raises M without changing geometry (zero-pads sine coefficients, verifies 0 um deviation before writing) |

### Analysis helpers (CPU only, cheap)

| script | what it answers |
|---|---|
| `work/helios/arc16/currents.py` | per-coil currents and lengths at every stage of every run — shows where a coil gets starved |
| `work/helios/arc16/whos_crowding.py` | which coil is closest to which, and whether it is a self-image pair |
| `work/helios/arc16/convex_pressure.py` | how much of each coil is pinned flat against the convexity floor |
| `work/helios/arc16/mode_use.py` | per-mode amplitudes, and whether new modes act where the coil is closest to the plasma |
| `work/helios/arc16/check_start.py` | pre-GPU gate: eps_B planarity spectrum, DP breakpoints, polar-arc admissibility |
| `917_results/planarizability.py` | eps_B defect spectrum by exact DP; also gives optimal B-arc breakpoints |

### Bounds files (`bounds/`)

`helios_kruger2026.json` is the paper set. Variants used here: `helios_L28/30/32/34.json` and
`helios_L36_ladder.json` (paper bounds with the length cap changed), `helios_kruger2026_noconvex.json`
(`convex_excess: null` — **the file the FourierXYZ reference used**), `helios_cold_L28_d10.json`
(L 28 + clearances 1.0), `helios_dcc1.0_16.json`, `helios_dpc1.0.json`.
A bound set to `null` drops that constraint entirely.

## 4. The viewer

`917_results/viewer.html` — open it directly, no server needed. 31 coilsets: B.n heat map on the
boundary, coil geometry, per-coilset metrics.

- It reads **`917_results/data.js`**, NOT `viewer_data.json`. `data.js` is just
  `window.HELIOS_DATA=<the json>;`. If you rebuild the json you must regenerate data.js.
- **`917_results/add_coilset.py`** adds one coilset incrementally and writes both files:
  `python 917_results/add_coilset.py "KEY|LABEL|NOTE|PATH"` (pipe-separated — labels contain colons).
  Re-using an existing KEY replaces it in place.
- It caches the virtual-casing plasma field to `917_results/_bplasma_cache.npz`, so adding a coilset
  takes seconds. `917_results/build_data.py` rebuilds everything from scratch and is slow; prefer
  `add_coilset.py`.
- Known wart: the viewer's "Mean / B_axis" column is a plain grid mean, while `HANDOFF.md` quotes an
  area-weighted one (4.279% vs 4.036% for the same planar coilset). The `bn` column agrees in both.

## 5. Traps that cost time in this session

- **`arc_ref` is a PARAMETER, not a constructor argument.** Rebuilding a `PolarPlanarArcCoil` from
  `hinges`/`tilts`/`shape` without copying `arc_ref` (and `shift`, `rotmat`) moves the curve by
  METRES. `raise_M.py` shows the correct pattern and verifies pointwise before writing.
- **The 12 GB cgroup cap now binds** for M=10 plus a corner constraint (second-order Hessian blocks).
  It killed a converged run at iteration ~294. Levers: `--so-chunk`, `--dcc-k/--dpc-k`, `--qf-chunk`.
  `run_one.py` writes no `result.h5` if killed, so **rely on `--ckpt-every` checkpoints**.
- **Validation of a new objective must include a real run.** `CoilCornerAngle` passed value and
  finite-difference checks and still died on first use, on a framework shape convention. See 6.3.
- Intermittent `CUDA_ERROR_ILLEGAL_ADDRESS`, roughly 1 run in 8. Rerun once before debugging; the
  chain scripts do this automatically.
- The feasibility repair often ends far outside the convexity bound (1.3 rad vs a 1e-3 bound) with
  the augmented Lagrangian stuck at `n_outer 1`. Stage 2 clears it every time, so it is not fatal,
  but do not read the repair's numbers as meaningful.

---

## 6. FOR THE NEXT TASK: constraining hinge LOCATIONS (demountable joints)

### 6.1 How hinges are represented

`PolarPlanarArcCoil.params_dict` has `['arc_ref', 'current', 'hinges', 'rotmat', 'shape', 'shift', 'tilts']`.

**`hinges` is a flat array of length 3B holding the B hinge positions in 3D (xyz), reshaped `(B, 3)`.**
For B=2 that is shape `(6,)` -> two points. Example from the best coilset, coil 0:

```
[[ 6.754  0.821  0.212]
 [13.334  2.997  2.921]]
```

They are free optimization parameters and they move a great deal during a free stage. The arc frame
is built from them by `_ppolar_arc_frame(hinges, tilts, B, arc_ref)` in
`vendor/desc/compute/_curve.py`; the arc between two hinges is r(phi) about the chord midpoint.

### 6.2 What already exists that you should reuse

- **`run_one.py --fix-hinges`** already freezes hinges completely, via
  `O.FixParameters(cs0, params=hinge_mask(cs0.params_dict))` where `hinge_mask` returns a pytree of
  booleans selecting the `"hinges"` key (see `run_one.py` around line 276). **If "force the hinges to
  prescribed locations" means "hold them exactly where I put them", you may not need a new objective
  at all** — set the hinges in the coilset, then run with `--fix-hinges`. `make_start.py --hinge-s
  "s0,s1;s0,s1;..."` places hinges at chosen curve parameters when building a start from an existing
  coilset, and `--cut-angle` places them where a line through the centroid crosses the coil.
- **`sandbox/corners.py`** is the closest template for a genuinely new per-hinge constraint. It is
  ~130 lines, subclasses `_CoilObjective`, evaluates on a custom grid of 2B nodes placed EPS either
  side of each hinge, and returns one value per hinge. Copy its structure.
- **`sandbox/convexity.py`** is the other worked example (per-node constraint, with a careful
  discussion of why the smooth form was necessary).

### 6.3 The trap that will bite you

`_CoilObjective.build` sets `self._dim_f` to the **total number of GRID NODES** when
`_broadcast_input == "Node"`, and builds `quad_weights` the same way. If your `compute` returns one
value per HINGE (B per coil) but your grid has 2B nodes per coil, the build dies inside
`objective_funs.py::_scale` with

```
TypeError: mul got incompatible shapes for broadcasting: (8,), (16,)
```

The fix used in `corners.py` is `jnp.repeat(values, 2)` so each hinge's value is emitted at both of
its nodes; the duplicate row is the same constraint twice and the augmented Lagrangian absorbs it.
If your constraint needs exactly B values per coil, either use a B-node grid (if one node per hinge
suffices — for a position constraint it does) or override `_dim_f` and the weights after
`super().build()`.

**A position constraint is simpler than the corner one**: you need the hinge coordinates, which are
parameters, not computed geometry. You may not need `_CoilObjective` at all — a plain `_Objective`
reading `params["hinges"]` is enough, and avoids the grid machinery entirely. Look at how
`FixParameters` does it before writing anything.

### 6.4 How to validate before spending GPU

Follow what was done for `corners.py`:

1. **Values** against an independent computation (for corners: `common.curve_metrics`'s reported
   angle; agreement was 1e-4 deg).
2. **Derivatives** against finite differences over the parameters the constraint actually touches
   (`jax.jacfwd` vs `(f(x+d)-f(x))/d`; agreement was 1.9e-6).
3. **A real one-iteration run.** Steps 1 and 2 both passed while the objective was still broken.
   `obj.compute_scaled_error(params)` is the call that exposed it — include it.

### 6.5 Physics context you will want

- **Hinges are where the coils are pinched.** Three of four coils in the M=10 optimum have their
  closest approach to the plasma exactly at a hinge (`mode_use.py` output in `HANDOFF.md`). So
  pinning hinges will interact directly with `d_pc`, and you should expect the plasma-clearance
  constraint to become active or infeasible if the prescribed locations are close to the plasma.
- **No mode can relieve a hinge.** Since every sin(m pi phi) vanishes there, raising M will not help
  a coilset whose hinges you have fixed badly. Only moving the hinge, or adding more of them (B=3),
  can.
- Current hinge positions vary a lot between coils and between runs — see `currents.py` and the
  per-run tables in `HANDOFF.md`. There is no single "natural" hinge location to snap to; the
  cut-angle scan found finals at chord offsets 0.17-0.52 of the coil half-size.
- If the demountable joints must be at the SAME (R, Z) for every coil, or on a common plane, expect
  that to cost real field error: the corner ladder showed corner *angle* is cheap, but hinge
  *position* is the parameter the optimizer uses most freely.

### 6.6 Suggested first steps

1. Read `CLAUDE.md`, then this, then `work/helios/HANDOFF.md`'s 2026-09-21 sections.
2. `python check.py` — confirms the vendored DESC is the one imported and shows the device.
3. Reproduce a known number cheaply: evaluate `work/helios/arc16/M10_c100/result.h5` with
   `sandbox/common.py: evaluate` and confirm bn = 7.2181e-3.
4. Decide whether `--fix-hinges` on a hand-placed start already answers the question (6.2) before
   writing a new objective.
5. If a new objective is needed, copy `sandbox/corners.py`, validate per 6.4, then run ONE short
   run before queuing anything.

### 6.7 Still open from this session

- **arcB3 at 16 coils** — repeatedly indicated as the most promising direction: more hinges give
  freedom at the scale where the constraints actually act, and B=3 led at 12 coils. Never run at 16.
- `run_corner_ladder.sh` (corner-only ladder) is built and unused.
- The varied cut-angle scan was stopped after 2 of 6 runs (`work/helios/arc16/run_scan.sh`, resumable).
- Total-length budget (sum over coils instead of per-coil cap) — never built.
