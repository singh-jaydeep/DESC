# Production experiments of the piecewise-planar coil paper

One section per experiment, all in the same fields, so that (a) the paper's methods text can be written from the
**Problem** and **Solver** fields and (b) the experiment agent can run each section as written. Sections are
discussed with the user before they are marked READY; nothing is launched from a section that is not READY.

Prior results (poster era, `js/qa-poster`) are references, not targets: the production runs use the ported code on
this branch (`PORT_SUMMARY.md`), and a different B.n can be a different basin (`findings/8a_basins.md`).

## 0. Method conventions (apply to every experiment unless its section says otherwise)

Decided with the user 2026-11-17.

| topic | convention |
|---|---|
| solver | `lsq-auglag-composite` (`stage2.py`): slack-free AL, every engineering bound a hard constraint; gtol 1e-6, ctol 1e-6, ftol = xtol = 0; `--maxiter` counts inner (subproblem) iterations in total. AL defaults (not changed anywhere): initial penalty mu = 10 on every row, multipliers 0; tolerance schedule of Conn-Gould-Toint / Nocedal-Wright alg. 17.4 with omega_0 = min(scaled gradient at x0, 1e-2), eta_0 = min(violation at x0, 1e-2), alpha_omega = beta_omega = 1, alpha_eta = 0.1, beta_eta = 0.9, penalty increase tau = 10; stop after 3 consecutive stalled subproblems (`desc/optimize/aug_lagrangian_composite.py`) |
| objective (stage 2) | DESC `QuadraticFlux`: B.n on the fixed boundary, area-weighted, normalized by B R0 a; weight 1 (the only objective term, so its weight sets no trade-off). Eval grid `LinearGrid(M=25, N=25, NFP, sym=eq.sym)` on rho = 1 (`--eval-mn`). Biot-Savart source: arcs composite Gauss-Legendre 24 nodes per arc; FourierPlanar / FourierXYZ `LinearGrid(N=50)` (`--field-n`). Vacuum: coil field only |
| constraints (stage 2) | per-coil length <= L (composite GL 24 per arc; `LinearGrid(N=400)` for smooth coils); max |kappa| <= kappa (grid row below); sum of coil currents fixed; `--current-band f`: each current within (1 -+ f) x the mean; d_cc, d_pc (rows below); arc options: fold via cos(fold) >= cos(fold_max), tilt via sin, hinge heights linear (eliminated exactly). Gauges: FourierPlanar largest normal component pinned; FourierXYZ phase pinned + relative speed within +-0.5 of the mean (`CoilArclengthResidual`, `LinearGrid(N=100)`). kappa_MS is never enforced |
| inner Hessian | `{"hessian": "secant", "finish_steps": 5}` (default; not revisited) |
| variable scaling | `x_scale="jac"` (default). Basin sensitivity is handled by multistart where it matters, not by a new scaling |
| run-to-run spread | report best-of-starts with the spread where the experiment's claim depends on it: the faces sweep and the Jorge QI stage 2 (comparison with planar). Each section states its starts |
| bounds (decided 2026-11-17) | **one convention for the paper**: Gil et al., PRE 114, 025202 (2026), Table III (reactor a = 1.7 m: 160 m of coil per half period, kappa 1 /m, d_cc 0.8 m, d_pc 1.5 m) scaled by f = a / 1.7 with the plasma's own a, the length split evenly over the unique coils of a half period; kappa_MS not used. Every bound scales, d_cc included. Faces sweep: `rr02_precise_{QA,QH}.json` (DESC's a); designs B, F: `gil_a0p12_nc{4,5}.json` (a = 0.12 m, where the plasma is placed; absolute values, so the plasma's later a does not move them); Jorge: `gil_jorge_L40.json` (DESC's a, with a stated per-coil length exception, 4.2). The poster's near-axis set (`fa_qa2_a0167`, `qi_jorge2022`) is retired. **Because of this the production results are not expected to match the poster's** (the designs' coils get 28-42% less length and ~40% more coil-coil clearance); the agent confirms each bounds file with the user before the first run that uses it, and discusses any major performance regression (B.n, QS, iota, nested surfaces) with the user before going on |
| grids | moderate, what a practitioner would pick, not convergence-driven densities: curvature on `LinearGrid(N=300)` (`--curv-grid linear --curv-n 300`; misses interior |kappa| peaks by up to 0.12%, accepted); the dense check stays the judge of feasibility. Every section states its grids |
| hinge turning-angle cap | **TO-DO (not implemented; deferred until the rest of this doc is done, 2026-11-17)**. New constraint: per hinge, the angle between the incoming and outgoing arc tangents, capped through its cosine (smooth), like `ArcFold`. **Off by default**; an experiment uses it only after it has been discussed for that experiment (2026-11-17). Its value is reported as `corner_deg` in every arc run regardless |
| coil-coil distance | `CoilSetDistanceRows` (unsigned top-k node-pair rows, each a lower bound: node distance minus a gap). Rows exist within `--cc-sel` x d_cc (2.5). `--cc-gap coils`: the node grid is chosen at build so the gap is <= 2% of d_cc (`--row-n` unused); `--cc-gap bounds`: 2 `--row-n` + 1 nodes per coil, gap from `node_gap_estimate`. Every section states `--cc-gap`, `--row-n` where used, `--cc-k` (starting k) |
| plasma-coil distance | upstream hard minimum first (`--pc-mode min`: `PlasmaCoilSetMinDistance`, plasma `LinearGrid(M, N, NFP)` from `--pc-plasma-mn`, default 32 x 32 over the full surface, coil nodes 2 `--pc-row-n` + 1 per coil, default 513). Switch to top-k rows (`--pc-mode rows`) **only with user approval**, after evidence of a stall (inner subproblems ending non-KKT while d_pc is active). Every section states the starting `--pc-k` and plasma grid it would use |
| top-k slot rule | the starting k must hold every in-band candidate at the start with margin; `run.log` prints `active X of k` at build and the dense check flags overflow. On overflow the closest rows are kept, but raise k and rerun rather than accept it |
| link guard | off. The agent may propose `--exp-pair-rows signed --link-mu 1e8` **to the user** after a run ends linked; it is turned on only with approval, and then for every run of that experiment |
| dense check (judge of every run) | `paper/lib/dense_check.py`, independent of the solve grids: length by composite GL 48 per arc; |kappa| on 2000 nodes per arc (smooth coils `LinearGrid(N=2000)`); d_cc and d_pc between coil polylines of 2001 nodes and the full boundary on 256 x 1024 points; linking numbers on 129-node polylines (|Lk| > 0.5). **Feasible = within 1% of every bound and unlinked** (decided by the N = 300 curvature rule: the solve sees |kappa| up to 0.12% low; RR-02 used the same 1% reading); strict margins are always reported |
| B.n metrics (three, never mixed) | (i) **stage-2 / dense**: mean |B.n| / mean |B| and max |B.n| / mean |B| over one field period, unweighted, on a uniform grid of max(96, 4 M_grid + 1) theta x max(72, 4 N_grid + 1) zeta (precise_QA/QH 96 x 72; designs B, F at basis 14: 113 x 113; Jorge 96 x 113), coil source GL 24 per arc / `LinearGrid(N=400)`; (ii) **free boundary**: area-weighted rms of B.n / <|B|> on `LinearGrid(2 M_grid, 2 N_grid)`, coil source GL 64 per arc (`free_boundary.py`); (iii) the single-stage objective (`VacuumBoundaryBn`, section 2). Every table names which one it shows |
| acceptance (every run) | solver success (gtol KKT); dense check feasible (row above); unlinked; coils rendered and judged tidy (no intertwining, leaning or crowding) before a result is called better |
| machine | one heavy job at a time on the laptop under `systemd-run --user --scope -q -p MemoryMax=12G -p MemorySwapMax=0`; Modal only where a section says so; confirm the run list and settings with the user before launching |
| records | each run directory keeps `cmd.txt`, `commit.txt`, `run.log`, `outer.json`, `result.h5`, `result.json`; each section names its summary table |

## Order and pre-launch to-dos

Run order: section 2 (designs B, F) -> section 3 (on F's fold-60 rung) and section 4 (independent) -> section 1 (largest
compute) -> optional experiments. Sections 1 and 4 can start whenever their to-dos are done.

| to-do | for | who |
|---|---|---|
| turning-angle constraint (section 0): library class, `stage2.py --turn-max`, tests | 1 | **deferred**: discuss with the user first |
| `faces/score.py` reports `corner_deg`; `run_sweep.sh` loops over caps, per-class seed lists, N = 300 curvature, QH maxiter 1600 | 1 | agent |
| 5 new QH r/a starts (planar1, modular) | 1 | agent |
| `compare/run.sh` without `pQA`, on the rebuilt F; cold starts on the rebuilt equilibrium (maybe `make_z0_start.py` runnable) | 3 | agent, after 2 |
| `jorge/chain.sh` cut to ladder -> C per seed + planar/free per seed + trace; new r/a starts | 4 | agent |
| confirm every bounds file with the user before its first use (section 0 bounds row) | all | agent + user |
| near-axis continuation of B's and F's states under the new bounds (2.1 step 1) | 2 | agent |
| render Jorge starts / R0 against the bounds; user confirms they look reasonable (4.2) | 4 | agent + user |
| flatten the near-axis MixedCoilSet before the single stage (2.1 step 2) | 2 | agent |
| faces figure layout | 1 | deferred (user) |

## Template (every section has these fields)

1. **Claim**: what the experiment shows, and the figure/table it feeds.
2. **Problem**: equilibrium (vacuum or not), bounds file and its convention/provenance, coils per half period, coil
   classes (representation, B, M, hinge rules, fold cap, current band), objective and its grid.
3. **Starts**: files, how they were built, number and kind of seeds, what varies between seeds.
4. **Solver**: AL settings, maxiter, distance terms with their grids and starting k, link guard, any deviation from 0.
5. **Driver**: exact commands in order, environment variables, memory cap, device and expected wall time.
6. **Outputs and metrics**: run layout, scoring function and grid, the summary table.
7. **Acceptance**: per-run checks, coverage required, poster reference and how to read a difference.
8. **Escalation and open items**: what the agent may change only with approval; what is still undecided.

Status per section: DRAFT (being discussed) / READY (agent may run) / DONE (results in `findings/`).

---

## 1. Faces sweep: field error vs number of planar faces (precise_QA, precise_QH)

Status: **READY once the code items of 1.5 are done** (all settings decided 2026-11-17; figure layout deferred).

**Poster reference**: `DESC2-poster/deliveries/RR-02/README.md` (request `DESC2-poster/requests/RR-02.md`):
110 runs on Modal A100, 22.8 GPU-h, `pareto.csv`.

### 1.1 Claim
B.n falls with the number of faces N toward the smooth-coil floor; knee at small N; seed spread shown at each N.
Poster (RR-02, median over seeds, relative to the FourierXYZ N 8 floor): precise_QA planar 17.9x, N = 2 1.9x,
N = 3 1.12x, N >= 4 within 2.6%. precise_QH: N >= 5 arcs fell below the smooth floor through uncounted ~120 deg
hinge corners. Both configurations are in the paper.

### 1.2 Problem
- Equilibria: `desc.examples` precise_QA (NFP 2) and precise_QH (NFP 4), both vacuum (`--vacuum`).
- Coils: 4 per half period (16 on QA, 32 on QH). Classes: `planar1` FourierPlanarCoil N 8; `arc2`-`arc8`
  PiecewisePlanarArcCoil B = 2..8, M 5, free hinges (no height or tilt terms); `modular` FourierXYZCoil N 8 with
  the port's phase pin and relative-speed bounds +-0.5 (`CoilArclengthResidual`; RR-02's modular coils had no speed
  bound: state it).
- Objective: QuadraticFlux (section 0), eval `LinearGrid(M=25, N=25)`, coil field only; Biot-Savart source GL 24 per
  arc (arcs: 48-192 nodes per coil for B = 2-8) and `LinearGrid(N=50)` (planar1, modular).
- Constraints (hard): L per coil, max |kappa| (LinearGrid N = 300, within faces: hinge corners carry no curvature),
  d_cc, d_pc, sum of currents fixed, each current within 1 +- 0.5 of the mean, turning-angle cap per the design matrix (1.8); free hinges only; no fold cap.
- **Not enforced: kappa_MS** (decision (b), 2026-11-17). The paper uses Gil's set without kappa_MS, as in every
  other example. Consequence to state: RR-02 found kappa_MS the binding budget of the QH smooth coils (relaxing it
  x1.2 lowered their B.n 8.4%), so QH numbers are not comparable with RR-02's.
- Bounds: convention 1 (Gil et al. PRE 114, 025202, Table III, reactor a = 1.7 m, scaled by f = a/1.7 with DESC's
  a: QA 0.171778 m, QH 0.124334 m; 160 m of conductor per half period split over 4 coils; kappa 1 /m / f;
  d_cc 0.8 m f (= 0.47 a); d_pc 1.5 m f). QA: L 4.0418 m, kappa 9.896 /m, d_cc 0.0808 m, d_pc 0.1516 m.
  QH: L 2.9255 m, kappa 13.673 /m, d_cc 0.0585 m, d_pc 0.1097 m. Files `inputs/bounds/rr02_precise_{QA,QH}.json`
  (their kappa_MS entry is ignored by the driver; consider writing `faces_precise_{QA,QH}.json` without it so the
  file says what is enforced). To do before the paper: check the Table III values against the PRE itself.

### 1.3 Starts
Rule (2026-11-17): **every class gets the same number of seeds, each from a distinct start**, even where the seeds
are expected to reach one optimum (planar, modular). Arcs vary by hinge rotation f (r/a seeds of the arcs reached the
same coils in RR-02: 0.06-4 mm); planar and modular, which have no hinges, vary by r/a.

Existing starts (`inputs/starts/faces/`): per seed the circles once, every class converted from them; arcs by chord
fit, M 5, hinges at equal parameter intervals from s = 0 (or s = f/B), flattened to one CoilSet.

| config | seeds per class | arcs | planar1, modular | new starts needed |
|---|---|---|---|---|
| QA | 4 | s0 (f 0) + s5, s6, s7 (f 1/4, 1/2, 3/4), all r/a 3.2 | s0, s1, s2, s3 (r/a 3.2, 3.1, 3.4, 3.3), circles inside every bound | none (proposal) |
| QH | 8 | s0 (f 0) + s3-s9 (f 1/4, 1/2, 3/4, 1/8, 3/8, 5/8, 7/8), all r/a 2.5 | s0, s1, s2 (r/a 2.5, 2.4, 2.6) + 5 new | 5 QH planar1 + modular starts (proposal: r/a 2.3, 2.35, 2.45, 2.55, 2.7, each unlinked; QH has no feasible circle start at 4 coils/hp) |

The builders are provenance copies (`inputs/starts/faces/build_starts.sh`, `flatten.py`) that call the poster's
`coil_sandbox/make_start.py`; new starts are built in the poster worktree with the same script, copied in, and added to
`MANIFEST.sha256`.

### 1.4 Solver
Section 0 defaults, with `--current-band 0.5 --maxiter 800` (QA) / `--maxiter 1600` (QH: 7 of RR-02's 8
non-converged runs were QH arc6-8 at 800), `--curv-grid linear --curv-n 300`, `--eval-mn 25,25`, `--field-n 50`.
Coil-coil rows: `--cc-gap coils --cc-sel 2.5 --cc-k 80000` (node grid set at build, gap <= 2% of d_cc; starting k
RR-02's value). Plasma-coil: hard minimum, `LinearGrid(M=32, N=32)` over the full surface, 513 nodes per coil. If a stall is shown and rows approved: `--pc-mode rows`
on a 128 x 128 half grid, starting `--pc-k 20000`. Link guard off (section 0).

Differences from RR-02 to state in the paper: link guard off (RR-02: `--link-mu 1e8`); plasma-coil hard minimum
(RR-02: the poster's smooth distance field, `--pc-spacing 0.02`); kappa_MS not enforced; turning-angle caps; QH maxiter.

### 1.5 Driver
`[TURN=deg] [SEEDS=...] [CLASSES=...] bash paper/faces/run_sweep.sh QA QH` (TURN replaces FOLD); scoring by `faces/score.py`, table by
`faces/table.py`. Compute: Modal or a cluster, chosen by the experiment agent (RR-02: QA 6-20 min, QH 7-28 min per run
on an A100). Run count: QA 7 arc classes x 4 seeds x 4 caps + 2 x 4 = 120; QH 7 x 8 x 4 + 2 x 8 = 240; 360 runs,
~70 A100-h at RR-02's rate (the capped runs may be slower).

Before launch (code): the turning-angle constraint (section 0) with a `stage2.py --turn-max deg` flag and tests;
`run_sweep.sh` must loop over the caps and take the seed list per class (arcs and planar/modular use different seed
folders) and pass the curvature grid, QH maxiter; `score.py` must report the **tangent turning angle** at each
hinge (`corner_deg`: max over hinges of the angle between the incoming and outgoing tangents, as RR-02) next to
`fold_deg_max`.

### 1.6 Outputs and metrics
`paper/runs/faces/<cfg>_s<seed>/<class>/` with `score.json`: B.n metric (i) of section 0 (bn_mean / bn_max, plus
bn_rms) on one field period, 72 (phi) x 96 (theta), unweighted, coil field only; margins against the bounds (L, kappa, d_cc, d_pc); `fold_deg_max`,
`corner_deg`; dof per coil. Table: one row per run (RR-02's `pareto.csv` columns + `hinge_phase`, `corner_deg`).
Figure: bn_mean vs N, per-seed points, best-of-seeds line, smooth floor at the best modular run.

### 1.7 Acceptance
Per run: converged (gtol KKT, optimality and violation < 1e-6), unlinked, feasible on the dense check (within 1%). Coverage: QA >= 3 converged seeds per class, QH >= 2. Render the best run of each class. Reading against
RR-02: QA should land near the RR-02 numbers in the free-hinge, uncapped cells; QH will differ (no kappa_MS).

### 1.8 Design matrix (decided 2026-11-17)

Free hinges only (no level or z = 0 arms: `HingeLevel` / `FixHingeHeight` put every hinge of a coil at one height,
which collapses faces for B >= 3). No fold cap (measurement below). For each arc class:

    multistart seeds (QA 4, QH 8)  x  turning-angle cap in {none, 90, 60, 30} deg

planar1 and modular: multistart only (no hinges). The cap values are the same for every N. The starts are the
uncapped ones; each run reports the start's `corner_deg` (chord fits to circles start with small corners, so the
caps should hold at the start: check). Figure layout deferred; it shows the caps and the multistart spread together.

Measurement 2026-11-17 (RR-02 arc optima, `fold_cos` on the ported code; corners from RR-02's `score.json`), medians
over seeds:

| | arc2 | arc3 | arc4 | arc5 | arc6 | arc7 | arc8 |
|---|---|---|---|---|---|---|---|
| QA corner (tangent turn) deg | 121 | 73 | 39 | 29 | 13 | 12 | 12 |
| QA fold, oriented (as ArcFold) deg | 80 | 87 | 100 | 152 | 164 | 165 | 164 |
| QA plane angle, unoriented deg | 80 | 87 | 87 | 89 | 82 | 82 | 86 |
| QH corner deg | 113 | 120 | 115 | 131 | 128 | 122 | 121 |
| QH fold, oriented deg | 88 | 140 | 148 | 143 | 150 | 151 | 145 |

For B >= 3 the fold (angle between consecutive face planes) does not track the corner: QA arc6-8 have 12 deg corners
but planes ~80-90 deg apart, because the plane of a nearly straight face is set by its small bulge and can point
anywhere. Every RR-02 optimum has folds >= 80 deg in every class, so any cap below that binds on every class, and for
many faces it restricts how the face planes rotate, not how sharp the corners are.

---

## 2. The two QA designs, B and F (rebuilt through one pipeline)

Status: **READY once the bounds files are confirmed with the user** (decided 2026-11-17; bounds changed to the paper
convention, simplified pipeline). The planar/hinged/free comparison (section 3) runs on design F's equilibrium, so it
comes after this.

**Poster references**: `inputs/references/design{B,F}/` (eq, coils, bounds), full packages and chains in
`DESC2-poster/coil_sandbox/from_axis/final/design{B,F}/` (`REPRODUCE.md`, `METRICS.md`).

| | design B (poster) | design F (poster) |
|---|---|---|
| coils | 4 per half period, B = 2, every hinge on z = 0, no fold cap | 5 per half period, B = 2, every hinge on z = 0, fold <= 60 deg |
| iota (axis -> edge) | 0.237 -> 0.218 | 0.383 -> 0.351 |
| A | 7.26 | 7.59 |
| QS L2 at rho .25/.5/.75/1 | 0.23/0.28/0.49/0.90 % | 0.27/0.32/0.49/0.97 % |
| final B.n rms (free boundary) | 8.7e-6 | 3.8e-5 |
| lineage | near-axis `li_qa2_iota0.30` -> fb basis 12 (repairs) -> single stage k 1..6 -> basis-14 re-solve -> fb14 -> polish x2 -> fb14 | near-axis `nc5_s4` (hinge band) -> D -> E (fold <= 60) -> E snapped to z = 0 -> single stage -> iota-floor redo (0.36) -> fb14 -> polish -> fb14 |

### 2.1 Pipeline (PORT_PLAN 9, simplified 2026-11-17; the same for both designs)

1. **Near-axis state under the paper bounds** (CPU, minutes): continue the saved state with the new bounds,
   `cd paper/near_axis; python cont.py ../inputs/near_axis/STATE runs/STATE_gil --a 0.12 --dpc 0.10588 --dcc 0.05647
   --Lmax L --kmax 14.1667` (L 2.82353 for B, 2.25882 for F; `cont.py` prints the `run_na2.py` command with every
   other argument of the saved state, second order, penalty weight 100; run it). `--a` is also the radius of the
   near-axis surface the plasma-coil clearance is measured on, so it matches the plasma placement below. States:
   B `inputs/near_axis/li_qa2_iota0.30` (4 coils/hp, iota target 0.30); F `inputs/near_axis/nc5_s3` (5 coils/hp,
   iota target 0.42, z = 0, no fold term). Log the coil margins before and after.
2. **Near-axis -> DESC -> free boundary at basis 12** (`near_axis/score_fb.py`, as the poster's B and D):
   `score_fb.py runs/STATE_gil --LMN 12 --a 0.12 --shells 2,4,6,8 --maxiter 60 --repair`. pyQSC surface at
   r = 0.12 m -> fixed-boundary vacuum solve -> free-boundary shells |m|, |n| <= k (`VacuumBoundaryError`,
   proximal-lsq-exact), boundary repair before every shell after the first. Coils: folded near-axis coils ->
   `PiecewisePlanarArcCoil` B = 2, M 5, saved as a MixedCoilSet: **flatten** to one CoilSet(NFP, sym) before step 3
   (`CoilSetDistanceRows` fails on a MixedCoilSet; `inputs/starts/faces/flatten.py`).
3. **Single stage** (`single_stage.py`, 2.3), basis 12, k = 1..6.
4. **Stage-2 polish on the single stage's own equilibrium** (`eq_k6.h5`, basis 12, the one the coils were
   co-optimized with; `stage2.py`, 2.3), best of 3 (2.2.6).
5. **Free boundary at basis 14** with the polished coils: `free_boundary.py EQ_k6 COILS OUT --LMN 14 --shells 14
   --maxiter 60 --solve-tol 1e-8 --chunk 400 --bounds BOUNDS` (`--LMN 14` re-solves at basis 14 first): the final
   equilibrium.
6. **Metrics and trace**: `metrics/make_metrics.py` (all stages, including the dense trace) on 5's equilibrium and
   4's coils.

Poster pipelines, for the paper's record (`DESC2-poster/.../final/design{B,F}/REPRODUCE.md`): B took a 9-run near-axis
lineage, a basis-12 free boundary with a manual repair, the single stage, a basis-14 re-solve, a basis-14 free
boundary, two polishes (the second with bound back-offs) and a final basis-14 free boundary; F went through the
intermediate designs D (hinge band) and E (fold <= 60) before its own z = 0 single stage, an iota-floor redo, two
basis-14 free boundaries and a polish. The simplified pipeline drops the intermediate designs, the free boundary
before the polish, and the back-offs.

### 2.2 Decisions (2026-11-17)

1. **Anchors**: the saved near-axis states (B `li_qa2_iota0.30`, F `nc5_s3`; full arguments in each state's
   `json["args"]`, lineage in `DESC2-poster/.../final/design{B,D}/REPRODUCE.md`), continued once under the paper
   bounds (2.1 step 1; the earlier "not rerun" is reversed because the bounds changed). Near-axis coil bounds stay
   penalties (weight 100), stated as such: the near-axis coils are only a start, and the polish restores hard
   feasibility. (OPEN_ITEMS 4.3, hard near-axis bounds, is not pursued for the paper.)
2. **What distinguishes them**: B is the low-iota minimal design (4 coils/hp, no fold cap). F carries a **fold-cap
   ladder 60 -> 45 -> 30 deg** (how tight the fold can get). F60 runs the full pipeline. The 45 and 30 rungs are
   optional detail (decided 2026-11-17): each is a stage-2 polish at the tighter cap from the previous rung's coils on
   F60's final equilibrium, then the basis-14 free boundary and metrics; a rung gets its own single stage only if
   its QS or B.n drops badly (discuss with the user first). If a rung cannot reach acceptance (2.4), the ladder
   stops there and that is the result.
3. **Bounds** (changed 2026-11-17): `gil_a0p12_nc4.json` (B: L 2.8235 m) and `gil_a0p12_nc5.json` (F: L 2.2588 m),
   both kappa 14.167 /m, d_cc 0.0565 m, d_pc 0.1059 m: the paper convention of section 0 at a = 0.12 m (absolute values). Compared with the poster's `fa_qa2_a0167` (Gil's set at the near-axis a = 0.167 m with
   d_cc fixed at 0.04 m: L 3.9412, kappa 10.174, d_cc 0.04, d_pc 0.1478, in which every poster design sits at the
   length bound with d_cc 0.041-0.045 m), length drops 28% (B) and 43% (F), d_cc rises 41%, d_pc drops 28%,
   kappa loosens 39%. **The results are not expected to match the poster's**; confirm the files with the user
   first, and discuss any major regression (2.4).
4. **Distances**: single stage uses smooth top-k rows for both terms (weighted least squares on a moving boundary;
   the plasma rows follow the boundary): starting `--cc-rows 40000 --pc-rows 32000` (driver defaults), `--rows-select
   1.6`, `--plasma-mn 40,40`. Stage-2 polish follows section 0 (plasma-coil hard minimum first; coil-coil rows).
5. **Uniform settings**: current band +-0.5 in single stage and polish for both; A in [6.6, 7.6]; iota floor =
   (nearest low-order rational below the edge) + 0.025: B 0.22 (edge clear of 1/5), F 0.36 (clear of 1/3);
   k = 1..6 at basis 12; `tau_bn 5e-4`, `tau_qs_rel 0.15` as the baseline (but see 7); final free boundary at basis 14;
   curvature on LinearGrid(N = 300) (section 0).
6. **Multistart at stage 5 only**: best of 3 polishes (different variable scalings or slightly perturbed starts),
   spread reported (8a: two KKT points 15x apart in B.n on F's polish).
7. **QS quality is a goal, not only a constraint.** Poster feedback: the QS of B and F (L2 0.9-1.0% at the edge) is
   not presentation quality. Where there is a trade-off, the agent should **give up some coil B.n to gain QS**: the
   coil field error of these designs is already very low (free-boundary B.n rms 1e-5 to 4e-5), so it has room. Levers
   in the single stage: smaller `--tau-qs-rel`, larger `--tau-bn`, QS surfaces (`--qs-rho`), the A window. Be
   explicit: every variant tried is logged with its (QS L2 at the four radii, free-boundary B.n rms and max, iota,
   A, eps_eff) next to the baseline, and the summary states which trade-off was taken and why. Reference point for
   "presentation quality": precise_QA scored by the same `metrics/make_metrics.py` QS metric. Limits: the polished
   coils must stay feasible, the edge trace clean, and the iota window kept.

### 2.3 Settings of each stage

**Single stage** (`single_stage.py`; `proximal-lsq-composite` on [eq, coils]): proximal constraints ForceBalance,
fixed Psi, p = 0, current = 0, boundary modes outside |m|, |n| <= k and R_00 fixed, sum of coil currents fixed, every
hinge at z = 0 (exact). Per k: maxiter 50, ftol = xtol = 1e-6, gtol 1e-8; inner equilibrium solves at tol 1e-6;
perturbation order 2; secant Hessian; `--repair-steps` (repair before every k, logged). Weighted least squares,
each term's weight 1 / (tau x its scale), so a term whose rows all sit at their tolerance costs 1/2:

| term | definition | tolerance (flag, value) | grid |
|---|---|---|---|
| B.n | `VacuumBoundaryBn`, rms B.n / <|B|> on the moving boundary, normalization frozen at the start | `--tau-bn 5e-4` | `LinearGrid(M=25, N=25)` on rho = 1; coil source GL 64 per arc |
| QS | `QuasisymmetryBoozer` (1, 0), unnormalized, one term per rho in .25, .5, .75, 1, weight / sqrt(4) | `--tau-qs-rel 0.15` x the **start's** rms over the four surfaces (anchor frozen in `anchors.json`) | `LinearGrid(2 M_grid, 2 N_grid)` per surface; Boozer M_booz = 2M, N_booz = 2N (DESC default) |
| aspect ratio | A in [6.6, 7.6] | `--tau-A 0.01` x mean of the window | |
| iota floor | iota >= floor on 20 surfaces rho 0.05..1 | `--tau-iota 0.01` x floor | `LinearGrid(M_grid, N_grid)` |
| volume | within +-2% of the start (`--vol-band 0.02`) | `--tau-V 0.005` x V0 | |
| L, kappa | <= bounds | `--tau-L 0.01`, `--tau-kappa 0.01` | length composite GL 24 per arc; curvature `--coil-grid-n 300` (section 0; driver default 150) |
| d_cc, d_pc | smooth top-k rows, `--rows-select 1.6`, `--cc-rows 40000 --pc-rows 32000` (starting k); plasma rows on `LinearGrid(M=40, N=40, sym)` following the boundary, gap computed at build (`--pc-gap` unset; log it) | `--tau-rows 0.01` x bound | coil-coil node grid chosen at build (gap <= 2% of d_cc) |
| fold (F) | cos(fold) >= cos(fold_max) | `--tau-shape 0.02`: 2% of fold_max in angle (cos tolerance 0.02 fold_max sin fold_max) | |
| current band | each current within (1 -+ 0.5) x mean | `--tau-shape 0.02` x 0.5 | |

QS levers for 2.2.7 are `--tau-qs-rel`, `--tau-bn`, `--qs-rho` and the A window; since tau_qs is relative to the
start, a rung or rerun from a better start asks for proportionally better QS.

**Stage-2 polish** (`stage2.py`, section 0 otherwise; on `eq_k6.h5`, or F60's final equilibrium for the 45/30 rungs):
`--bounds gil_a0p12_nc{4,5}.json --vacuum --fix-hinge-z 0 [--fold-max 60|45|30]
--current-band 0.5 --maxiter 800 --curv-grid linear --curv-n 300 --cc-gap coils --cc-sel 2.5 --cc-k 80000`;
plasma-coil hard minimum on `LinearGrid(M=32, N=32)`, 513 nodes per coil. Fold sides from the start (default). No
bound back-offs (the poster's L x 0.9999, kappa x 0.995 are replaced by the 1% feasibility reading of section 0).
Best of 3: x_scale "jac" plus two variants chosen by the agent (e.g. `--options '{"x_scale": ...}'` or starts
perturbed by ~1 mm), recorded in each `cmd.txt`.

**Free boundary** (step 5): `VacuumBoundaryError` (coil field fixed, GL 64 per arc) with ForceBalance, FixCurrent,
FixPressure, FixPsi as proximal constraints, `proximal-lsq-exact`, ftol 1e-8, xtol 1e-10, gtol 1e-8, inner solve tol
1e-8; one shell k = 14.

**Compute**: Modal or a cluster, the agent's choice (single stage at basis 12 needed an A100 40 GB, ~1 h; basis-14
free-boundary shells 1.4-2.4 h on the laptop). Every boundary repair is logged (`repairs` in result/score json, one
REPAIR line each).

**Metrics** (`metrics/make_metrics.py`): QS = L2 and max of the symmetry-breaking Boozer |B| amplitudes / B_00 (%) at
rho .25/.5/.75/1, M_booz = 2M, N_booz = 2N, on `LinearGrid(2 M_grid, 2 N_grid)`; two-term f_C; eps_eff and Gamma_c
(Bounce2D, two resolutions); iota, shear, well, mirror ratio, elongation; coil margins (dense check); B.n metrics (i)
and (ii); dense field-line trace (traced iota and surfaces vs the equilibrium's).

### 2.4 Acceptance (per design, per F rung)
- Final coils: KKT at stage 5, feasible on the dense check, unlinked, rendered and tidy.
- Equilibrium: free boundary at basis 14 converged; iota window as designed; the zoomed edge trace shows nested
  surfaces out to the boundary (traced lines vs eq rho = 1; poster F's first version had the 1/3 separatrix at
  rho 0.94).
- The rebuild uses different bounds (2.2.3) and a different path (F especially: the poster went through D and E
  with a hinge band), so **the poster numbers are a comparison, not a target**. A major regression against the
  poster design (e.g. QS L2 more than ~2x worse at any radius, B.n more than ~10x, iota outside the window, or lost
  edge surfaces) is reported to the user and discussed before the agent goes on; QS gains per 2.2.7 are welcome.

### 2.5 Outputs
Per design: `paper/runs/design{B,F}/<stage>/` (F: per rung `F60`, `F45`, `F30`); `metrics/make_metrics.py` output
next to the poster's `METRICS.md`; a findings note with the QS-vs-B.n trade-off table.

---

## 3. Planar vs hinged vs free on design F

Status: **READY** (decided 2026-11-17). Runs after section 2.

Decided 2026-11-17: the equilibrium is the **final rebuilt design F at the fold-60 rung** (section 2; the 45 and 30
rungs are optional extra detail). The precise_QA control arm is **retired** (drop the `pQA` set from
`compare/run.sh`; `inputs/starts/compare_pqa/` and `bounds_pqa.json` stay as provenance). Claim: how much of the free
coils' B.n the hinged coils recover, against planar coils, at equal bounds on one equilibrium.

Poster reference (on E's equilibrium, `inputs/starts/compare_E/`): mean B.n planar N 8 1.98e-2, hinged 5.9e-5, free
FourierXYZ N 8 3.1e-5 (planar ~300x worse, hinged within 1.9x of free); free coils from circles converged to E-like
shapes. Mechanism note: `DESC2-poster/.../designE/comparison/why/WHY.md` (m = 2 content of the current potential).

### 3.1 Problem
- Equilibrium: final design F, fold-60 rung (section 2), vacuum, fixed.
- Bounds: `gil_a0p12_nc5.json` (as F, section 2.2.3). Objective, constraints, grids and solver: section 0
  and the stage-2 polish settings of 2.3 (current band +-0.5, maxiter 800, curvature `LinearGrid(N=300)`, plasma-coil
  hard minimum on 32 x 32 first, coil-coil rows `--cc-gap coils --cc-sel 2.5 --cc-k 80000` as starting k), identical
  for the three arms except the coil class and its own terms (hinged: `--fix-hinge-z 0 --fold-max 60`, fold sides from
  the start; free: phase pin and speed bounds; planar: normal pin).
- Arms (5 coils per half period each):
  - planar: FourierPlanarCoil N 8;
  - hinged: design F's class, PiecewisePlanarArcCoil B = 2, M 5, every hinge on z = 0, fold <= 60 deg;
  - free: FourierXYZCoil N 8 with the phase pin and relative-speed bounds +-0.5.

### 3.2 Starts (decided 2026-11-17: every arm cold, 4 seeds each)
- Circles around the rebuilt plasma at 4 r/a values around 3.4 (e.g. 3.2, 3.3, 3.4, 3.5; each unlinked and inside the
  bounds), built with the poster's `make_start --rep planar --N 8`; the free arm is fitted from the same circles
  (`--rep xyz --N 8`).
- Hinged arm: first try two-face arcs fitted to the same circles (`--rep arc --B 2 --M 5 --chord-angle 90`, flattened)
  with the hinges snapped to z = 0 (`stage2.py --fix-hinge-z 0` snaps them itself and prints the largest move and the
  START verdict); check feasibility and linking at the start. If the snap breaks feasibility
  (as it did on precise_QA), build z = 0 lens starts instead (`make_z0_start.py`, which must first be made runnable;
  provenance copy in `inputs/starts/F/`), one per seed.
- New starts are copied into `inputs/starts/compare_F_rebuilt/` and added to `MANIFEST.sha256`.
- **Design point** (reported separately, not an arm): design F's own coils on this equilibrium (co-optimized with the
  plasma, so not a cold start).

### 3.3 Metrics and outputs
- Per arm: best-of-4 mean B.n, B.n metric (i) of section 0 (113 x 113 on F at basis 14), with the spread; max B.n;
  margins.
- Recovery: the fraction of the planar-to-free gap closed in log B.n,
  `R = log(planar/hinged) / log(planar/free)` from the best-of-seeds values (poster on E: R = 0.89). Also report the
  plain ratios planar/hinged and hinged/free.
- Field-line trace (`trace_lines.py`) of the best run of each arm: do the planar coils hold nested surfaces?
- Runs in `paper/runs/compare_F/<arm>/s<k>/`; summary table in a findings note.

### 3.4 Acceptance
Every run KKT, feasible on the dense check, unlinked, rendered; >= 3 converged seeds per arm. The poster ratios are a
comparison, not a target.

---

## 4. Jorge 2022 QI stage 2 (hinged C at a 2% cap; planar and free controls)

Status: **READY once the bounds file is confirmed with the user and the starts and cut-down driver exist** (decided
2026-11-17). Keep it simple: the quickest path to the result.

Equilibrium `inputs/equilibria/jorge_vac.h5` (Jorge 2022 one-period QI, vacuum, NFP 1), 12 unique coils per half period (24 in all), bounds
bounds `gil_jorge_L40.json` (4.2). Driver
`paper/jorge/chain.sh` (to be cut down to the path below). Poster record: memory `paper-scope-decisions` and
PORT_PLAN 9.4 (relax_vs_cap).

### 4.1 Path (decided 2026-11-17)
- **Hinged**: fold ladder with level hinges (+-0.025 m of the axis) 10 -> 20 -> 30 -> 45 -> 60 from `J_arc90`
  (= R0, poster 1.41e-2 / 17.5%), then **directly** tilt <= 20 deg + pointwise |B.n|/|B| cap 2% on R0 (one step;
  replaces the poster's A3 -> C_bn004 -> C_bn002 sequence; poster C_bn002 5.30e-3 mean / 2.3% max).
- **Planar** (FourierPlanar N 8) and **free** (FourierXYZ N 8, phase pin + speed bounds): controls.
- **Multistart** for every arm: 4 seeds (r/a around 3.6; the hinged arm's seeds carry through its ladder).
- Field-line trace (`trace_lines.py`, then `zoom_poincare.py`) of the best hinged run; poster C_bn002 nested out to
  offset ~0.18 m.

### 4.2 Decisions (2026-11-17)
- **Controls uncapped**: planar and free run once each per seed with no pointwise cap. Comparison on mean B.n
  (the cap moved the hinged mean only 6.43e-3 -> 5.30e-3); the hinged 2% max is reported next to the controls'
  uncapped max as a property of the hinged result.
- **Plasma-coil rows pre-approved for planar** on the poster evidence (P0/P0b: the hard minimum stalled every
  subproblem while d_pc was active): `--pc-mode rows --pc-plasma-mn 128,512 --pc-k 40000` (starting k; row gap
  5.8 mm on Jorge's boundary, curvature up to 100 /m), `--row-n 384`. Hinged and free arms: hard minimum first
  (section 0).
- **Bounds (decided 2026-11-17: option (i))**: `inputs/bounds/gil_jorge_L40.json`: the paper convention (section 0)
  at DESC's a = 0.12498 m (kappa 13.602 /m, d_cc 0.05881 m, d_pc 0.11028 m) with one stated **exception, the
  length**: L = 40 m x f = 2.9407 m per coil, Gil's per-coil value for 4 coils per half period. The convention's
  split (160 m x f over 12 unique coils per half period = 0.98 m) cannot enclose the plasma (a coil that clears it
  by d_pc needs >~ 1.5 m). Rationale for the paper: there is no settled bound set for QI coils; Gil's is a QA/QH
  reactor set. Against the poster's `qi_jorge2022.json` (retired; L 2.9495, kappa 13.594, d_cc 0.04, d_pc 0.1106)
  the change is mainly d_cc 0.04 -> 0.0588 m (+47%). Confirm the file with the user before use; results are not
  expected to match the poster's; regressions are discussed with the user (section 0).
- **No published-coil reference (decided 2026-11-17).** The Jorge 2022 paper's coil figure shows 30 coils, and the
  authors' public data (`paper/jorge/reference/QI_NFP1_Jorge2022`, copied by the user, not committed) has no coil
  files; its two coil paths (REGCOIL cut into 20 coils per field period; a simsopt stage-2 tutorial with 12 unique
  coils, a soft length cost, a 0.1 m distance penalty and no curvature term) were switched off for the published run.
  So there is nothing to match. Instead:
  - **Bounds reasonable by eye**: before the first production run, render the starts and the R0 result against
    `gil_jorge_L40.json`, mainly the per-coil length (2.94 m) and coil-coil distance (0.0588 m): coils of that length
    should wrap the plasma without slack loops, and the clearance should look like a real gap between neighbours at
    12 coils per half period. Show the renders to the user and confirm before running the chain.
  - **One bound set for every arm**: hinged, planar and free all run with `gil_jorge_L40.json` (and the same current
    band, distances, grids and maxiter), so the comparison is at equal bounds.

### 4.3 Settings
Common (from `chain.sh` BASE): `--vacuum --current-band 0.5 --maxiter 800 --cc-gap bounds --row-n 512
--cc-k 30000` (1025 nodes per coil, gap from `node_gap_estimate`; coil-coil starting k) `--pc-plasma-mn 48,96
--pc-plasma-sym half --pc-row-n 128` (hard minimum on the stellarator-symmetric half surface, 257 nodes per coil),
`--curv-grid linear --curv-n 300` (section 0). Objective eval grid `LinearGrid(M=25, N=25, NFP=1, sym)` as the poster:
51 toroidal planes around the whole one-period torus for 12 coils (see open item below). Dense-check B.n grid 96
(theta) x 113 (zeta).
- Hinged ladder: `--horizontal-hinges --hinge-zmax-rel 0.025 --fold-max F` for F = 10, 20, 30, 45, 60, each from
  the previous; from F20 on `--fold-sides-from <F10>/result.json`.
- Hinged C (one step from R0 = F60): `--hinge-tilt-max 20 --hinge-zmax-rel 0.025 --fold-max 60 --fold-sides-from
  <F10>/result.json --bn-max 0.02 --bn-max-grid 48,96 --jac-chunk 16`. On an 8 GB GPU
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.92`.
- Planar: as above plus the rows of 4.2. Free: FourierXYZ N 8 with the phase pin and speed bounds +-0.5.
- Trace: `trace_lines.py --ntransit 150 --offsets 0.02,0.04,...,0.20` (axis from the coils, surface fits M 16 N 16,
  residual tolerance 5e-3), then `jorge/zoom_poincare.py --band 0.15:0.20:0.005` (600 transits) at the edge.
- Open: the objective eval grid. 25 x 25 at NFP 1 is ~2 toroidal nodes per coil; the PointwiseBn cap grid is 48 x 96
  and the dense check 96 x 113. The poster ran 25 x 25 and its dense B.n agreed; keep (moderate grids) unless the user
  prefers `--eval-mn 25,64`.

### 4.4 Starts
4 seeds per arm at r/a 3.4, 3.5, 3.6, 3.7 (3.6 is the poster's; each start unlinked, its margins logged; lens starts
broke L by 8-56% on Jorge, so arcs are fitted to circles). Per seed with the poster's `make_start`: circles
`--eq jorge_vac.h5 --nc 12 --r-over-a R --rep planar --N 8 --vacuum --bounds gil_jorge_L40.json`; arcs `--from-coils
... --rep arc --B 2 --M 5 --chord-angle 90`, flattened; free `--from-coils ... --rep xyz --N 8`. Existing: r/a 3.6
planar and arc (`inputs/starts/jorge/`); the rest are new, copied to `inputs/starts/jorge/` and added to
`MANIFEST.sha256`.

### 4.5 Driver (to do before launch)
Cut `jorge/chain.sh` down to: ladder -> C (tilt + 2% cap directly) per seed; planar and free per seed; trace of the
best hinged run. Drop A3, the 4% step and the P cap ladder.

### 4.6 Metrics and acceptance
Per arm: best-of-4 mean and max B.n with the spread, margins; recovery `R = log(planar/hinged) / log(planar/free)`
as in section 3. Every run KKT, feasible on the dense check, unlinked, rendered; >= 3 converged seeds per arm. The
best hinged run's zoomed trace nested to the plasma edge (poster: to offset ~0.18 m). Poster references: R0
1.41e-2 / 17.5%, C_bn002 5.30e-3 / 2.3%, planar P0c 3.34e-2 / 22.4%; the poster has no free-coil reference on Jorge.

---

## Optional experiments (run only if time allows, after everything above)

- **O1. Planar-coil single stage from design F** (poster "fair test 2", on D: planar k6 B.n 8.6e-3 vs E 1.41e-3, QS
  2-4x worse). Answers "what if the plasma were reshaped for planar coils?". Same single-stage settings as F's (section
  2) minus hinge and fold terms, from the planar arm's best coils of section 3. Not yet specified further.
