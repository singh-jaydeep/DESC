# Stage-2 optimality on precise_QH + FourierPlanarCoil — handoff

Session of 2026-10-05, the follow-up to `stage2_crawl_handoff.md` (read that first: F1-F4,
the composite hinge model, `ctr.py`). Branch `js/demo-stage2notebook`. Everything here is
prototype code in `scratchpad_crawl/`, **uncommitted**; nothing was changed in `desc/`.
Run scripts from inside `scratchpad_crawl/` with `.venvDEV/bin/python`; one heavy job at a
time (8 GB box).

Goal of the session: make a hard stage-2 problem converge to a certified optimum
(tiny gradient, PD Hessian) because the user's augmented-Lagrangian (AL) outer loop fails
when its weighted subproblems can't reach optimality.

---

## 1. Problem

| item | value |
|---|---|
| equilibrium | `desc.examples.get("precise_QH")`, NFP=4, sym, a = 0.124 m |
| coils | 4 unique `FourierPlanarCoil`, N=7 (32 physical with symmetry) |
| start | `initialize_modular_coils(eq, 4, r_over_a=2.5)` (user's choice; no circular start is feasible: r/a 2.5 gives L 1.95 m, cc 0.042 m, pc 0.096 m) |
| bounds (physical units) | length <= 2.92 m, \|curvature\| <= 13.7 1/m, coil-coil >= 0.058 m, plasma-coil >= 0.11 m |
| linear constraints | `FixSumCoilCurrent`; `fixnorm` (see F-gauge) |
| QuadraticFlux | eval grid `LinearGrid(M=25, N=25, NFP=4, sym=True)`, coil grid `LinearGrid(N=50)` |
| free parameters | 83 (after projection) |

Weights (penalty, not AL): QF 200, length 200, curvature 100, coil-coil 300,
plasma-coil 100. The precise_QA weights (cc 100 / pc 10 / curv 30 / len 20) do not bind
here: coils grew to 4.0-4.5 m. With the weights above the penalty optimum still violates
every bound (length ~3.06-3.27 m, curvature ~14.2, cc ~0.043 m, pc ~0.086-0.096 m);
feasibility is the AL loop's job.

## 2. The hands-off recipe

```
cd scratchpad_crawl
W='"len_w":200,"curv_w":100,"cc_w":300,"pc_w":100'
KW="{\"case\":\"qh\",\"cc\":\"curve\",\"pc\":\"curve\",\"fixnorm\":true,\"pair_N\":128,\"cc_K\":3000,\"pc_K\":6000,$W}"
python mkstage.py E1 QHI_x.npy "$KW"        # tag E1 at the circular coils
python ctr.py compS E1 1000 E1o             # composite + structured secant, from circles
python ctr.py comp2 E1 8 E1n E1o_y.npy      # 3-8 exact-Newton steps to the certificate
python segcheck.py "$KW" E1n_x.npy          # active rows, closest distances, linked pairs
python diag.py E1 E1n_x.npy --hess          # cost breakdown, constraint values, certificate
```

(`QHI_x.npy` = packed circular coils, from `run.py QHI --case qh ... --maxiter 0`.)
Ingredients, all required:

1. **Per-row smooth distances with per-evaluation re-selection** (`curverows.py`):
   coil-coil rows "node of a -> exact curve b" and "node of b -> exact curve a";
   plasma-coil rows "plasma point -> exact coil curve", coils incl. stellarator reflections.
2. **Continuous linking-signed row per coil pair** (in `CoilSetCurveRows`).
3. **Composite hinge model** (`ctr.py`, from the QA session).
4. **Normal-scale gauge pinned** (`fixnorm=True`).
5. **Structured secant S** (`ctr.py compS`) for the large-residual phase.
6. **Exact-Hessian finish** (`ctr.py comp2`, with the fixed damping floor).

**Result of the recipe run end to end (E1, from circles, no intervention):**
compS 172 its / 403 s to scaled opt 1.5e-6, then comp2 5 accepted steps to **7.7e-9**
(\|Dg\|inf 8.3e-9, raw \|g\|inf 1.4e-4, Newton decrement 1.2e-13, exact Hessian PD,
eigenvalues 7.4e-4 .. 11.5). Cost 2.1528329537e5 = QF 1.640e5 + length 4.09e4 + cc 7.12e3
+ pc 3.14e3 + curvature 46. Brute-force (N=1000 segments): coil-coil 0.04536 m,
plasma-coil 0.08597 m; length 3.06-3.27 m, curvature max 14.1-14.3. Coils saved as
`QH_E1.h5`. Coil pair (2,3) is **linked** at this penalty optimum (§3.4); expected, AL's job.
The piecewise chain G3 -> H1 -> C1 -> C1n reached 2.1e-10 at a nearby optimum (cost
2.14635e5); the difference is only where comp2 hits the cost-resolution floor (§5).

## 3. Everything that changed

### 3.1 New files (`scratchpad_crawl/`)

| file | what |
|---|---|
| `setup_qh.py` | precise_QH builder. Switches: `cc` in {hard, pairs, cut, seg, curve}, `pc` in {hard, pairs, seg, curve}, weights, bounds, `pair_N`, `cc_sel/cc_K/pc_sel/pc_K`, `signed`, `fixnorm`, `link_w/link_N` (built-in `CoilSetLinkingNumber`), `ref` (candidate-selection state for `cut`/`pairs`). Also defines `CoilSetCutoffPairDistance` and `PlasmaCoilSetPairDistance` (both superseded, see 3.3). |
| `cases.py` | `build(case="qa"|"qh", **kw)` dispatch; `run.py`/`common.py` use it. |
| `curverows.py` | **final distance objectives**: `CoilSetCurveRows`, `PlasmaCoilSetCurveRows`, plus `planar_curve` (JAX evaluator of a FourierPlanarCoil at arbitrary t, with t-derivatives), `physical_maps` (3x3 map + source coil for every physical coil), `point_curve_distance` (Newton on t, unrolled 4 its, differentiated through). Evaluator matches `_compute_position` to 9e-16 on all 32 coils. |
| `segrows.py` | segment-distance version (`CoilSetSegmentRows`, `PlasmaCoilSetSegmentRows`) and helpers copied from `_vendored_coils.py` (`_segment_segment_distance`, `_point_segment_distance`, `_unique_pairs`, symmetry perms). Superseded by `curverows.py` for optimization, still the best brute-force checker. |
| `_vendored_coils.py` | the user's vendored `_coils.py` from another session (reference only; not importable stand-alone, package-relative imports). |
| `mkstage.py TAG X KW` | writes `TAG.json`, `TAG_allx.npy` ([x]), `TAG_ref.h5`; the way to start `ctr.py` from any full state. |
| `segcheck.py KW FILE...` | `report()` of the cc/pc objectives: active rows, smallest distance, linked pairs. Accepts `.h5`, `*_x.npy`, `init`. |
| `diag.py TAG X [--hess]` | per-objective cost, length/curvature/cc/pc values, currents; with `--hess`: GN vs exact Hessian spectra, S along flat directions, scaled gradient, Newton decrement. |
| `sdecomp.py TAG X` | splits S = sum r_i Hess r_i by objective; worst GN contraction mu = max eig(-S, G); slowest directions and the parameters they move. Saves `TAG_sdecomp.npz`. |
| `pintest.py TAG X` | mu restricted to subspaces with candidate gauges pinned, with random-pin controls. |
| `lk.py` | per-pair linking numbers (unique coil vs all), cheap; `lk.py TAG ALLX` traces linking over iterates. |
| `missed.py`, `stages.sh` | staged runs with candidate refresh (superseded by per-evaluation re-selection). |

### 3.2 Edits to existing files

| file | change |
|---|---|
| `ctr.py` | (a) new arm **`compS`** (§5); (b) **bug fix in `comp2`**: `lam_floor` was taken from eig(B_targetᵀB_target + S), which omits active hinge rows, so it was indefinite wherever constraints hold and damped every step; now eig(BstdᵀBstd + S) (the scaled exact Hessian). On the old QA problem this only changes comp2 runs; (c) checkpoints `OUT_x/y.npy` every 50 its; (d) `KW` env var also applies on the non-PHASE path. |
| `run.py` | `--case`, `--kw JSON`; legacy flags default to None so each builder keeps its own defaults (QA defaults unchanged). |
| `common.py` | imports `build` from `cases`; `load_run(tag, extra_kw=None)`. |
| `diag.py` (new this session) | plasma-coil check includes stellarator reflections (it had the same half-surface bug as `PlasmaCoilSetPairDistance`; fixed at the end, so its earlier printed pc values for runs before E1 are optimistic). |

### 3.3 Objective evolution (why each step was needed)

| objective | defect found | evidence |
|---|---|---|
| hard-min `CoilSetMinDistance` (DESC) | F1 kink, as in QA | trsteps mismatch +2.9 at Delta=1 vs <=2e-3 elsewhere (QH0) |
| QA `CoilSetPairDistance` (2 nearest neighbours by centroid, frozen at build) | coils grew 1.95 -> 4.4 m and crossed unlisted coils | QC1 "converged" (opt 2e-6) with fine-grid gap 0.003 m and many linked pairs |
| node-node pairs, cutoff list refreshed per stage (`cut`, `stages.sh`) | node spacing (7 cm at N=30) > bound; coils approaching at an angle slip between nodes; also needs stage refreshes | lsqtr hard-min reported 0.034 m where the true gap was 0.006 m |
| `PlasmaCoilSetPairDistance` (mine) | **bug**: sym=True plasma grid covers theta in [0, pi] only and I measured from the unique coils only; reflected coils (which face the other half) were never checked | reported pc 0.1045 m where the true value was 0.082 m |
| segment rows (`seg`), two-pass re-selection | correct and resolution-free, but segment-segment distance is only C^1: Hessian jumps where a closest point clamps to a segment end, and optima park on polyline vertices | comp2 stalled at 2e-7 (H1n); 3 of 6 active pairs had exact ties at vertices; same comp2 on C^inf node distances reached 1e-11 (N2) |
| **curve rows** (`curve`) | — | comp2 reaches 1.9e-10 (C1n). Point side sampled: reads up to ~0.2 mm high where coils approach at an angle (error ~ (sin(theta) h/2)^2 / 2d); judge feasibility with the brute-force segment check |

### 3.4 Linking

- `CoilSetLinkingNumber` (built-in, `link_w`): in one replay it prevented the link, but it
  is ~10x slower (15 s/it, 3.8 GB peak; all 32x32 pairs) and unreliable exactly when coils
  nearly touch (unlinked coils read 0.3-1.4 at N=40/64; the integral only converges once
  the grid resolves the gap). It is piecewise constant, so it cannot "see" an approach.
- **Cliff signed row** (`-N (d_min + d*)` when linked, the `CoilSetDistancePenalty`
  form): jump of ~2.6e7 at a crossing; with a weak cc weight the solver pressed two coils
  to 0.5 mm and then stalled at the cliff edge (G1; mismatch +2.7e7 at every Delta >= 1).
- **Continuous signed row** (final): per coil pair, the closest side-a node's row reads
  sign * d, sign = -1 if linked (|Lk| > 0.5 on `LinearGrid(N=100)`, stop_gradient),
  and that node is excluded from the unsigned slots. Continuous through a crossing (flip
  at d = 0) and through argmin switches while linked (tied rows swap {-d, d}). The solver
  may cross; afterwards the row grows as the pair separates and pulls it back
  (G3 reached a smooth optimum with no stall).
- With a **fixed penalty weight** the pull-back is finite: G3 and C1n end with pair (2,3)
  linked at d* ~ 0.043 m, a genuine local minimum of the penalty function. Unlinking needs
  the multiplier on that row to grow, i.e. AL.
- `CoilSet.is_self_intersecting()` (default test) flags 0.034 m gaps as intersecting;
  do not use it. Use `segcheck.py` / `lk.py`.

### 3.5 Gauge

`FourierPlanarCoil`: scaling `normal` is an exact gauge (positions identical to 1e-16 under
normal*1.7). `fixnorm=True` pins the largest normal component per coil.
In-plane centre shift vs the r_{+-1} radius modes is a first-order near-gauge; it does
**not** explain the slow GN rate (pinning it: mu 0.83 -> 0.72; 8 random pins: 0.72-0.82).

## 4. Why Gauss-Newton converges only linearly here (measured at G3o)

S = sum r_i Hess r_i split by objective; mu = max generalized eig(-S, G) is the GN
contraction per step along the worst direction.

| S included | worst mu |
|---|---|
| all | 0.83 |
| without QF | 0.74 |
| without length | **7.9** |
| only QF (against full G) | 7.4 |
| only QF against its own G | ~9500 |

- The field error is irreducible (\|r_QF\| ~ 580 scaled) and in directions where B.n barely
  changes to first order its second-order term dominates its own GN curvature by up to 1e4.
- The engineering rows supply the curvature there; the violated length penalty's
  positive S offsets the negative S of QF, coil-coil, plasma-coil, curvature.
- Prediction (not run): relaxing targets removes the stabilizing length term and would make
  GN worse. AL inner subproblems have shifted residuals that do not vanish either, so the
  same structure appears there. Hence Hessian refinement rather than target changes.

## 5. `compS` (structured secant) and why it stops near 1e-6

- NL2SOL-style: model Hessian G + S~, S~ in unscaled coordinates; sized symmetric rank-2
  update with **y# = dg - G+ s** (gradient change minus the GN part; row-order independent,
  required because selected slots change identity between iterates); update skipped if
  yᵀs <= 0; per-iteration switch between GN and augmented model, whichever predicted the
  last step better; damping floor from eig(Bstdᵀ Bstd + Sd).
- Cost per iteration = GN (no HVPs). From G3o: 1.4e-2 -> ~1e-6 in 77 its (GN would need
  several hundred). From H1 with curve rows: 21 -> 1.7e-6 in 140 its.
- It stops near 1e-6 because its steps stop contracting the gradient (last 10 its: opt
  bounced 1.2e-5 .. 1.7e-6 while cost decreases were still 20-1500 ulp): a secant S~ is
  only right along recent step directions, is invalidated by active-set changes (194 -> 220
  active hinges), and y# gets noisy as steps shrink. Many small accepted steps are needed,
  their decreases approach a few ulp (ulp(cost) = 2.9e-11 at 2.1e5), the ratio test turns
  to noise, ~12 consecutive rejections blow lambda up.
- `comp2` from there: opt 1.6e-6 -> 7.8e-7 -> 3.4e-7 -> 7.6e-8 -> 6.4e-9 -> 1.9e-10
  (quadratic; last two steps accepted on 3-ulp decreases). 5 steps x ~40 s at n = 83.
- comp2's own floor is set by the cost resolution too: it stops once a Newton step cannot
  show a decrease of >= 1 ulp. C1n got two lucky 3-ulp acceptances (1.9e-10); E1 stopped
  at 7.7e-9 with lambda climbing from 1e-4 to 4e4 in 3 steps. Row-paired decrease
  measurement (below) would lift both floors.
- Possible compS improvements (not done): reset S~ on active-set change; periodic exact
  refresh; measure the decrease as sum (r - r_new)(r + r_new)/2 (row-paired) instead of a
  difference of totals.

## 6. Pitfalls hit

- Large composite (LM) steps from circles jump coils through each other within 10 its;
  only per-evaluation selection + signed rows made starting `ctr.py` from circles safe.
- Candidate lists frozen at build time go stale as coils move; re-select every evaluation.
- Any pair-distance grid must resolve the clearance; prefer smooth exact-curve rows.
- Tag collisions: my `P1` and `N2` runs overwrote the QA session's `P1.json`, `P1.log`,
  `N2.log` (restored from git; my copies are `qh_P1.*`, `qh_N2.log`) and QA's untracked
  `P1_allx.npy` (lost; regenerate with `run.py P1 ...` from `P1.json`). Use distinct prefixes.
- `ctr.py` without PHASE reads a *reduced* y file (`*_y.npy`) as its 5th arg; full states
  go through `mkstage.py`.
- `ctr.py` output tag is used verbatim (`H1` -> `H1_x.npy`, not `H1o_x.npy`).
- Memory: the built-in linking objective at N=64 peaks at 3.8 GB; the N=200 all-pairs
  linking check at 32 coils asks for ~4 GB (compute per pair, as `lk.py` does).

## 7. Run log (outputs on disk only)

| tag | what | outcome |
|---|---|---|
| QH0 | lsqtr, QA weights, hard-min | trust radius 2540 -> 3e-10, cost 3.774e4, length 4.0-4.5 m |
| QC1 | comp, QA pairs (frozen), QA weights | opt 2e-6 but crossed/linked coils |
| R, W | staged comp from circles | crossed within the first stage |
| WA -> WB/WC/WD -> N2 | lsqtr hard-min globalization, staged node pairs, comp2 | opt 1.1e-11, but (2,3) linked and pc checked on half the surface only (true 0.082 m) |
| L1 | built-in `CoilSetLinkingNumber` replay of the linking stage | no link; 15 s/it; gaps 4-6 mm |
| G1 | seg rows + cliff signed rows, cc_w 300 | stalled at contact (0.5 mm) |
| G2 | same, cc_w 3000 | progressing; stopped (superseded) |
| G3 | seg rows + continuous signed rows, comp, 500 its from circles | opt 0.013, (2,3) linked at d* 0.043 m |
| H1 / H1n | compS / comp2 from G3o (seg rows) | ~1e-6 in 77 its / comp2 stalls 2e-7 (C^1 vertices) |
| C1 / C1n | curve rows from H1: compS / comp2 | 1.7e-6 in 140 its / **1.9e-10** |
| E1 | hands-off recipe of §2 from circles | compS 172 its (403 s) -> 1.5e-6; comp2 -> **7.7e-9**, PD Hessian; (2,3) linked |

## 8. Next: augmented Lagrangian

To be designed with the user. Pieces to carry over: per-row smooth distances with
re-selection (multipliers per *row* need stable row identities: slot rows change identity,
so the AL needs either multipliers on persistent rows — curve rows are indexed by
(pair, side, node), which is persistent, so store multipliers per candidate rather than per
slot — or aggregation); continuous signed rows (the AL multiplier is what un-links);
composite inner model; compS + comp2 inner solves to ~1e-10.
