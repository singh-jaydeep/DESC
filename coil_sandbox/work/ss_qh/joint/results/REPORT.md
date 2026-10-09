# arcB2 demountable hinges: report on every run so far (2026-10-08 → 09)

Goal (user, 2026-10-08): arcB2 coils whose hinges allow demounting. The strict version puts every hinge on one
horizontal plane, z = 0. The relaxed version (user's idea) keeps each coil's two hinges level, with each coil at its
own height. Question: can single-stage optimization make either cheap?

All B·n values are ⟨|B·n|⟩/⟨|B|⟩ from the dense check. "Converged" means `run_al.py` stopped on gtol (optimality and
violation < 1e-6). "Removable" is the exact vertical lift-out test (`sandbox/demount.py`), checked after each run and
never constrained. Detailed timeline: `../NOTE.txt`. Earlier write-up: `RESULTS.md`. Viewers: `view.html` (precise_QH study) and `view_devices.html` (other equilibria; its margins are always
against precise_QH's bounds, so they are misleading for the 1328095 own-design and precise_QA panels).

## 1. Headline results

| question | answer |
|---|---|
| Single plane z = 0 on precise_QH | 2.8× free arcB2 (fixed eq); 4.3× after single stage. The vertical excursion of the helical axis is never removed. |
| Level hinges per coil on precise_QH | 1.37–1.6× (fixed eq); **1.93× after single stage**; **1.53× with coil continuation**. |
| Does coil continuation help? (user's test) | **Yes, 21%**: refined B·n 2.859e-3 → **2.266e-3**, with better QS and ι, and it stops on tolerances, not maxiter. |
| Best other equilibrium for arcB2 | **QUASR 1328095**: free 2.69e-3 (2× better than precise_QH), level 5.64e-3 (fixed eq). |
| Can arcB2 meet 1328095's own design (5 coils, QUASR bounds)? | **Yes: feasible in 5–7 inner iterations.** The B·n solves stalled at the trade-off, not at feasibility. |
| Candidates for the strict z = 0 plane | **None among QUASR QH** (613 screened). QA only at ι ≤ 0.2: **QUASR 804112 (ι 0.2) has the cheapest joint plane, 1.70× free** (precise_QH 2.8×, precise_QA 5.4×), but its absolute B·n is worse (free 8.9e-3). |

## 2. precise_QH, arcB2 M5, the hinge conditions

| run | boundary | B·n | removable | notes |
|---|---|---|---|---|
| free, warm (matrix) | precise_QH | 5.437e-3 | — | sweep start |
| free, cold lens + link guard | precise_QH | 6.421e-3 | — | `base_free_link` |
| z = 0 | precise_QH | 1.801e-2 | yes | `base_z0` |
| level per coil | precise_QH | 8.768e-3 | yes | `base_level` (from base_z0) |
| free (matrix) | B_arcB2 (free single stage) | **1.484e-3** | — | the reference |
| z = 0 | B_z0 (single stage) | 6.372e-3 | yes | k = 5 refined 6.356e-3 (stalled) |
| level | B_level (single stage) | 2.859e-3 | yes | `on_level/arcB2_level` |
| **level, coil continuation** | B_cont (single stage) | **2.266e-3** | yes (+0.074 m) | `cont/refine` |

### Coil continuation (user's test): it helps

Schedule (`cont/cont.sh`; the user chose the wider ramp): the length bound and the arc modes rise with the
boundary-mode step k. Everything else is identical to the `ss_level` baseline.

| step | length × | M | B·n (ss coils) | QS ρ = 0.5 | QS edge | ι min | baseline B·n | baseline QS ρ = 0.5 |
|---|---|---|---|---|---|---|---|---|
| stage 2 | 0.80 | 3 | 1.85e-2 | — | — | — | 8.77e-3 (×1.0, M5) | — |
| k = 1 | 0.80 | 3 | 1.63e-2 | 5.3e-3 | 1.2e-2 | 1.239 | 8.46e-3 | 1.35e-3 |
| k = 2 | 0.867 | 4 | 7.42e-3 | 3.2e-3 | 9.0e-3 | 1.189 | 5.30e-3 | 1.60e-3 |
| k = 3 | 0.933 | 5 | **3.50e-3** | **8.6e-4** | 3.6e-3 | 1.194 | 4.15e-3 | 1.49e-3 |
| k = 4 | 1.0 | 5 | **2.44e-3** | **7.0e-4** | **2.9e-3** | **1.208** | 3.28e-3 | 9.9e-4 |
| k4b | 1.0 | 5 | 2.44e-3 | 7.0e-4 | 2.9e-3 | 1.208 | 3.22e-3 | 9.7e-4 |
| refined | 1.0 | 5 | **2.266e-3** | | | | 2.859e-3 | |

* It starts behind (shorter coils, fewer modes) and overtakes at k = 3, while its coils are still 7% shorter. It
  ends better on B·n (−21% refined), QS (both radii; better than free arcB2's own single stage, 8.8e-4 / 4.8e-3), and ι
  (above the 1.19 target, which the baseline missed).
* k = 4 stopped on ftol after 69 iterations and k4b on xtol after 5. Every baseline step ran to maxiter = 100. The
  continuation path is better conditioned, not only better.
* **Caveats**: one schedule, so the length and mode ramps cannot be separated. The start also differs (a fresh M3 lens
  vs the converged base_level coils). The free-hinge matrix single stage never had a continuation, so the 1.53× ratio
  compares a continued run with an uncontinued one.

## 3. Other equilibria (fixed equilibrium, arcB2 M5, precise_QH bounds, 4 coils/half period)

| equilibrium | ι | free (warm from level) | level per coil | level ÷ free | level removable? |
|---|---|---|---|---|---|
| precise_QH | 1.25 | 5.437e-3 (matrix) | 8.768e-3 | 1.61 | yes |
| **QUASR 1328095** | 1.31 | **2.693e-3** | **5.644e-3** | 2.10 | no (5% of an arc) |
| QUASR 1632037 | 1.40 | 4.510e-3 | 7.672e-3 | 1.70 | no (3%) |
| QUASR 1632468 | 1.70 | 6.340e-3 | 8.908e-3 | 1.41 | no (30%) |
| QUASR 1775228 | 2.00 | 5.726e-3 | 1.117e-2 | 1.95 | no (33%) |
| WISTELL-A (mirrored, scaled) | 1.06 | 1.346e-2 | 1.457e-2 | 1.08 | yes (+0.245 m) |

* QUASR 1328095 came from the 2026-09-25 two-plane screen (`925_quasr_sweep/AUDIT.md` §10). It is the only device
  picked for arcB2 geometry, and it is the best. The other three QUASR devices were chosen by best QS per ι bin.
* **Removability fails on every QUASR level result.** At high ι the level chords rise up to 0.68 m (5 a), and the arcs
  dip back under the plasma. Accepted for now (user); `demount.py` has `HingeShadowClearance` and `CoilDemountPenalty`
  ready to wire in.
* A cold free start on 1328095 fell into a poor basin in outer iteration 1 (B·n 0.10, hinges 0.75 m up). Every free
  control above is therefore warm-started from the level result. Their chords stayed within ~0–50° of horizontal (vs
  37–74° for precise_QH's independent free run), so **level ÷ free is a lower bound**.

### 1328095 at its own design (5 coils/hp, QUASR's bounds) — user's question: is the representation asked too much?

Bounds (`bounds/quasr1328095_own.json`): length 2.5 m per coil (my translation of QUASR's 12.5 m TOTAL per half
period, which is stricter); d_cc 0.10 and κ_MS 35 (QUASR thresholds); d_pc 0.15 (as built); κ 13.67 (QUASR never
pinned it). All of these were set for FourierXYZ order-16 coils (~99 parameters per coil vs ~19 for arcB2 M5).

| run | result |
|---|---|
| level (B·n), infeasible lens start | STALLED after 4 outer: B·n 8.58e-3, ~1% infeasible, not removable |
| free warm from it | STALLED: B·n 7.22e-3, ~0.2% infeasible |
| **feasibility only, level** (`--obj-weight 1e-6`, coil-coil grid ×2) | **FEASIBLE** in 7 inner iterations, removable |
| **feasibility only, free** | **FEASIBLE** in 5 inner iterations |
| level (B·n) from the feasible point, grid ×2 | **STALLED again** after 4 outer: B·n 1.14e-2, ~1% infeasible, not removable; the inner subproblems never met gtolk (opt 1.7e-2) |

**So the representation can meet these bounds easily, and the stall is the solver's, not the geometry's.** Neither the
infeasible start nor the coarse coil-coil grid explains it: from a feasible point, with the grid doubled, the B·n solve
stalls the same way. With L, κ, d_cc and d_pc all binding at once at 5 coils, the inner subproblems stop converging
(optimality stays at 1e-2 against gtolk 1e-3), and the stall rule ends the run after 4 outer iterations. Untried:
a larger max_stalls, a different penalty growth, or a length continuation (which helped in §2).

## 4. The strict z = 0 plane: candidate search

**Screen** (`z0screen/`, 613 QUASR boundaries, the best 60 by QS per group: QH nfp 2–5, QA nfp 2–3, A 6.67 / 8 / 10). The
metric is the fraction of each cross-section's R extent that z = 0 cuts, minimised over toroidal angle (0 = misses a
section entirely; the reference values are precise_QH 0.00 and WISTELL-A 0.34).

* **QH: z = 0 misses at least one cross-section entirely on every device** (all ι 0.9–2.6, every nfp). The
  vertical excursion carries the transform. WISTELL-A is the unusual QH.
* **QA: z = 0-friendly only at low ι**: 0.80–0.86 at ι = 0.1, 0.55 at 0.2, 0.44 at 0.3, and 0 by ι ≥ 0.4–0.6.
* precise_QA (ι 0.42): 0.36 (median 0.51), comparable to WISTELL-A, so it was tested:

| precise_QA, arcB2 M5, sweep910 bounds ×0.8 length, 4 coils/hp | B·n | removable |
|---|---|---|
| free, old solver (run_one, 2026-09) | 4.54e-3 | — |
| free, new solver from that state (`free_base`) | **4.542e-3** (already a KKT point: 16 inner its) | — |
| free, warm from level | 8.736e-3 | — |
| level per coil | 1.733e-2 | yes |
| z = 0 | 2.432e-2 | yes (+0.314 m) |

**QUASR 804112** (QA nfp 3, A 10, ι 0.2; the best of the R-extreme metric with ι > 0.1: ext_med 0.80, cut_min 0.75),
boundary scaled exactly to a = 0.1243, precise_QH bounds, 4 coils/hp (QUASR's own design used 2):

| QUASR 804112 | B·n | ÷ free | removable |
|---|---|---|---|
| free (warm from level) | 8.922e-3 | 1 | — |
| level per coil | 1.350e-2 | 1.51 | yes |
| **z = 0** | **1.516e-2** | **1.70** | yes (+0.29 m) |

**z = 0-friendly geometry does make the joint plane cheap in relative terms** (1.70×, the lowest of all), and a shared
plane costs only 11% over level hinges there. But the device is harder for arcB2 in absolute terms (free 8.9e-3 vs
precise_QH's 5.4e-3), and ι = 0.2 is low. The R-extreme height (`ext_med`) is the metric that predicted it;
`cut_min` alone did not (precise_QA).

**z = 0 costs 5.4× free on precise_QA (2.432e-2 vs 4.542e-3), and level costs 3.8×.** That is worse than on precise_QH. The z = 0 cut metric
does not predict cost: precise_QA's bean cross-sections keep their R-extremes ~1.6 a off the midplane (as precise_QH's
do), and a horizontal chord cannot follow the bean's tilt. A better metric would score the R-extreme heights
(`ext_med`, which the screen also records); on that metric only low-ι QA devices (ext_med 0.6–0.7) look good.

## 5. Good / bad / untested

**Good**
* Level hinges per coil + single stage + coil continuation: 1.53× free arcB2, removable, QS and ι better than baseline.
* Coil continuation is a real improvement (and more robust: it ends on tolerances).
* QUASR 1328095: the arcB2-friendly QH, as remembered.
* No linked pair in any run since the link guard (`--link-mu 1e8`).

**Bad**
* Removability is not constrained and fails on every QUASR level result.
* Cold free starts can collapse (1328095).
* AL stalls when every coil bound binds at once (1328095 at its own design), even from a feasible start.
* No QUASR QH suits the strict z = 0 plane. On precise_QA, z = 0 costs 5.4×. The cheapest joint plane (804112, 1.70×)
  comes with low ι (0.2) and worse absolute B·n.
* Single stage (proximal-lsq-exact, plain Gauss-Newton) stalls at k = 5 (`K5_STALL.md`). Two intermittent GPU segfaults
  (identical reruns passed).

**Untested**
* Bounds were never tuned per device: everything uses precise_QH's set (Gil Table III, a-scaled), except 1328095/nc5
  and precise_QA. Length is active in every run, so the results depend directly on it.
* Coil count (4/hp everywhere except nc5) and M = 5 were never varied.
* Free controls are warm-started from level (biased). There is one start per device.
* The continuation schedule: one schedule only, and it has never been applied to free hinges.
* No single stage was run on any new equilibrium. QUASR boundaries are unsolved (fine for vacuum fixed-eq B·n).
* Currents were copied from precise_QH with their sum fixed (harmless for normalized vacuum B·n by argument, not tested).

## 6. Suggested next steps

1. Apply the coil continuation to free arcB2 on precise_QH, so the 1.53× compares like with like.
2. Wire `HingeShadowClearance` / `CoilDemountPenalty` into `run_al.py` as AL rows, then rerun the QUASR level cases.
3. Single stage + continuation on QUASR 1328095 (needs an equilibrium re-solve).
4. A cheaper AL stall rule, or `hessian=secant` in the single stage (`K5_STALL.md`).

## Files

`../cont/` (continuation), `../quasr1328095/` (4-coil runs; `nc5/` own design + feasibility), `../devices/` (batch),
`../qa_z0/` (precise_QA), `../z0screen/` (screen.jsonl, 613 rows), `../ss_level*`, `../on_level/` (level single stage).
Code: `run_al.py --horizontal-hinges --obj-weight`, `single_stage.py --horizontal-hinges`,
`make_z0_start.py --chord-z centre`.
