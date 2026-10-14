# Handoff: finalize designs B and D (written 2026-10-14)

Goal: turn designs B and D into two self-contained, reproducible, fully characterized design packages that can be
shown outside the group, and then map the range of hinge coil sets that can produce each equilibrium.
Read first: `CLAUDE.md` (rules and traps), `BEST_QA_Z0.md` section 0 (design B), `RESULTS_2026-10-13.md` section 8
(design D), `ss/NOTES.md` (single stage, free boundary, repair, Modal). Rules that still hold: coils are primary,
vacuum only; the score is a vacuum free-boundary solve with the coils plus a Poincaré check; use the branch's
objectives (`CoilSetDistanceRows`, `PlasmaCoilSetDistanceRows`, `lsq-composite`); local GPU one job at a time under
`systemd-run` memory caps; Modal jobs with `--detach` plus a watchdog. **Ask the user for a Modal budget before the
first Modal job** (the last stated limit was 3 hours per night).

## The two designs

| | **B** | **D** |
|---|---|---|
| coils | 4 per half period (16), two planar faces, every hinge on z = 0 | 5 per half period (20), hinges within ±0.0334 (0.2 a) of z = 0, hinge tilt ≤ 2° |
| ι | 0.22 (flat) | 0.415 → 0.365 |
| A | 7.26 | 7.56 |
| QS L2 of breaking modes, ρ .25/.5/.75/1 | 0.25 / 0.28 / 0.47 / 0.94% | 0.30 / 0.28 / 0.47 / 0.95% |
| largest breaking mode | 0.13 / 0.14 / 0.22 / 0.45% | 0.19 / 0.11 / 0.17 / 0.32% |
| ε_eff, ρ .25/.5/.75 | 6.9e-4 / 7.4e-4 / 1.3e-3 | 2.6e-4 / 2.8e-4 / 6.8e-4 |
| B·n rms (coils on the equilibrium boundary) | 0.10% (repaired) | 0.097% |
| coils feasible | length 1.1 mm (0.03%) over its bound | yes (d_cc 0.0411 vs 0.04) |
| known weak points | rescore creased; k=5 may be the better state (never scored); old repair used in the single stage | both rescore shells stopped on "bad approximation"; repair after k=6 moved edge QS 0.59 → 0.75% and ι by ~0.02 |

### Files

**B** (`ss/li030/`, harvested from Modal Volume `from-axis-runs:/ss12_li030_q15`, commit 6993369dc):
* coils `ss/coils_k6.h5`; equilibrium to quote `repaired/eq.h5` (= `fb_norepair_local/eq_fb.h5` repaired with
  `bdry_repair`, 0.09 mm rms); scores `fb_norepair_local/score.json`, `repaired/report.txt`; `qs_metrics.json`.
* every single-stage step: `ss/eq_kK.h5`, `raw_eq_kK.h5`, `coils_kK.h5`, `trials_kK.txt`, `result.json`.
  Alternative state `ss/eq_k5.h5` + `coils_k5.h5` (re-solved L2 0.25/0.27/0.50/0.98%, A 7.04): not free-boundary scored.
* rescore with repairs after each shell (superseded, not faithful): `fb_repair_modal/`.
* ε_eff convergence for B was checked on C's run only (`ss/ref042r/eps_conv.json`); run `ss/eps_conv.py` on B itself.

**D** (`ss/nc5lh/`, from `from-axis-runs:/ss12_nc5lh_q15b`, commit 8aa3ed22f; full commands in `ss/nc5lh/meta.json`):
* coils `ss/coils_k6.h5`; equilibrium `fb/eq_fb.h5`; `fb/score.json`, `fb/poincare.png`, `fb/tip.png`;
  `qs_metrics.json`, `eps_conv.json`; single-stage `ss/eq_k6.h5`, `result.json`, `anchors.json`.
* earlier k steps were not harvested; they are on the Volume under `/ss12_nc5lh_q15b/ss/`.

### Provenance (what the reproduction document has to spell out)

Both start from the near-axis reference state `na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10` (lineage in `REPORT.md`
section 10-11 and `NAE_HANDOFF.md`: `s1_qa2_z0_s2` → `o2_qa2s2_gg1e-2_w1e-3` → `o2w1_qa2s2_gg1e-1_w1e-2_w1x10`).
Each `na/runs2/<name>.json` stores the full `run_na2.py` arguments under `"args"`, with `"init"` naming its parent.

**B:**
1. Near-axis: `na/lowiota.sh` → `na/runs2/li_qa2_iota0.30` (ι target 0.30, z = 0 hinges, init = the reference state).
2. Free-boundary state at basis 12 on the laptop: `na/runs2/li_qa2_iota0.30_fb_a0.12_L12`, then repaired/continued
   to `..._L12_rep` (shells 2-8). **The exact command lines were not saved to a file**; reconstruct them from the
   `.log` files, `score_fb.py`'s defaults at commit 6993369dc, and the session transcript
   (`~/.claude/projects/-home-jay-Documents-desc-dev-DESC/*.jsonl`, 2026-10-12). If they cannot be recovered, say so
   in the reproduction document and record the closest equivalent command (`score_fb.py ... --LMN 12 --a 0.12
   --shells 2,4,6,8 --repair`).
3. Single stage + rescore: Modal job `ss12_li030_q15` (`cloud/jobs.py`, `ss_b12("li030_q15", LI030, 0.15, 0.22)`),
   hinges fixed at z = 0, `--cc-rows 12000`. Used the old fixed-label repair (`ss/repair_lib.py` before it became a
   shim for `bdry_repair`).
4. Local rescore without repair → `fb_norepair_local/`; repaired with the new package → `repaired/`.

**D:**
1. Near-axis: `na/ncchain.py 5` → `nc5_s1` … `nc5_s4` (s4 = hinge band 0.0334, tilt ≤ 2°, ι target 0.42).
2. Free-boundary score: Modal job `fb_nc5lh_b12` (`score_fb.py runs2/nc5_s4 --LMN 12 --a 0.12 --shells 2,4,6,8
   --maxiter 60 --repair`, commit da44eb5ec); local copy `relaxed/nc5lh/`.
3. Single stage + rescore: Modal job `ss12_nc5lh_q15b` (`--hinge-zmax 0.0334 --hinge-tilt-max 2.0 --cc-rows 24000`,
   ι ≥ 0.35, A in [6.8, 7.6]). The earlier `ss12_nc5lh_q15` is invalid (coil-coil rows overflowed); leave it out.
4. Metrics: `ss/qs_metrics.py`, `ss/eps_conv.py` (laptop GPU).

Main drivers: `na/run_na2.py`, `na/nae.py`, `na/folded.py`, `na/ncchain.py`, `na/lowiota.sh`, `na/score_fb.py`,
`ss/prep.py`, `../single_stage.py`, `ss/fb_rescore.py`, `ss/repair_eq.py` + `../sandbox/bdry_repair/`,
`../sandbox/demount.py` (HingeBand), `ss/qs_metrics.py`, `ss/eps_conv.py`, `ss/tip.py`, `ss/trace_check.py`,
`cloud/app.py` + `cloud/jobs.py`, `bounds/fa_qa2_a0167.json` (path `../bounds/`).

## Target layout

```
from_axis/final/
  README.md              one page: what B and D are, the side-by-side table, where everything is
  compare/               metrics table and overlay plots for B and D made by the SAME scripts and settings
  designB/
    README.md            summary, headline numbers, caveats
    REPRODUCE.md         every step above with exact commands, commit hashes, versions, wall times, seeds
    METRICS.md           all equilibrium and coil metrics (list below), generated from metrics.json
    metrics.json
    data/                coils.h5, eq.h5 (+ exports, below)
    drivers/             frozen copies of the drivers at the commit used, with a note that the repo is canonical
    figures/
    make_figures.py      regenerates every figure and metrics.json from data/
  designD/               same structure
  stage2_family/         section 5 below; kept apart from the frozen packages
```

## Work, in order

### 1. Settle each design's equilibrium first (do not make figures before this)

* Get a converged free-boundary solve for the final coils, or explain why it does not converge. Both D's rescore
  shells (and C's) stopped on "A bad approximation caused failure to predict improvement" at B·n 0.1-0.2%; ss4
  reached 0.04%. Things to try: a tighter or different `--solve-tol`, a different optimizer for the free-boundary
  solve, a smaller shell step, basis 14 on the final state only.
* Resolution check: re-solve at basis 14 (and 16 if feasible), more coil quadrature (`--src-gl`), and recompute QS,
  ι, ε_eff. Record the change; quote the converged values.
* ι from field-line tracing of the coils (rotation number from the Poincaré traces; `ss/trace_check.py` maps
  punctures into flux coordinates). Compare with the equilibrium's ι before and after repair. This settles the
  open "repair moves ι by 0.02" question (RESULTS sections 7b, 8).
* B: decide between k=6 and k=5 (score k=5 with the same rescore; k=6 shrank the plasma 2.3% and its edge gain did not
  survive the re-solve). B's coils are 1.1 mm over the length bound: a short coil-only polish to feasibility (stage 2
  with the equilibrium fixed, then rescore) would remove that caveat.

### 2. Freeze the packages

Copy the data and drivers, write `REPRODUCE.md` (commit hashes, `python -c "import desc, jax; ..."` versions, GPU
type, wall times, Modal job names and Volume paths), and write `make_figures.py`. If budget allows, rerun one stage
from scratch (the D rescore is the cheapest at ~10 min) to confirm it reproduces; GPU runs need not match bit for
bit, so state a tolerance.

### 3. Metrics (METRICS.md, both designs, same scripts)

Equilibrium: R0, a, A, volume; ι profile and shear; QS (L2 and largest breaking Boozer mode, two-term) vs ρ; ε_eff vs
ρ at two resolutions; vacuum magnetic well V''(ψ); mirror ratio; elongation; B·n rms/max on the boundary; Jacobian
check. **Resonances:** in nfp 2 the dangerous rationals have n a multiple of 2. D's ι range crosses 2/5 = 0.4; B sits
near 2/9 ≈ 0.222. Trace densely around those surfaces and estimate island widths.

Coils: per coil length, max curvature and minimum bend radius at the arcs, face sizes, fold angle, hinge heights and
tilts; d_cc and d_pc (min and map); currents (and in amps after scaling to a reference machine, e.g. R0 and B0 of
ARIES-CS, stating the scaling); forces and torques per coil, including across the hinge; clearance with an assumed
finite winding-pack thickness (state the assumption).

Literature columns as in `BEST_QA_Z0.md` section 00 (precise_QA, reactor_QA, NCSX, ARIES-CS), same scripts.

### 4. Figures (`make_figures.py`)

* Poincaré of the coil field overlaid with the equilibrium's flux surfaces at several toroidal angles (this is the
  "equilibrium vs coil-set reconstruction" comparison), plus zooms near the rational surfaces.
* |B| on Boozer (θ_B, ζ_B) at ρ 0.25 / 0.5 / 0.75 / 1 (straight contours expected for QA).
* Breaking Boozer modes vs ρ; ι profile (equilibrium vs field-line tracing); ε_eff vs ρ with the literature values.
* Cross-sections at several toroidal angles; B·n and coil-plasma distance mapped on the boundary.
* 3D coils with the plasma; a close-up of one hinge; the near-axis target axis vs the solved axis.
* Existing viewer: `ss/viewer_ss.html` (B = entry with `li030`, D = entry 7); reuse its export for the 3D view.

### 5. Tolerances

Random coil displacements (~1 mm rms, several seeds) and current errors: QS change and island widths at the
rationals above. Hinge-specific errors: one face offset or rotated slightly about the hinge line. Score with the
same free-boundary + trace pipeline, or with traces alone if that is shown to be adequate.

### 6. Stage 2 family (separate folder, own budget)

Question: now that these equilibria are known to suit hinge coils, what range of coil sets can make them?
* Fit coils to the fixed final boundary (B·n plus the branch's distance rows and bounds), then rescore each fit by
  vacuum free boundary + Poincaré. A fixed-boundary fit is not the score.
* Variations: number of coils; plasma-coil distance; length and curvature bounds; hinge band and tilt (for D: can
  every hinge sit exactly on z = 0?); regularizers for tidier coils.
* Reference fit with unconstrained smooth coils (FourierXYZ) to measure what the hinge constraint costs.
* Plot field error against coil complexity as a trade-off curve.
* Watch `--cc-rows`: with 5 or more coils per half period 12000 slots overflow silently except for a UserWarning
  ("rows are within select_distance"); use 24000 or more and grep every log for that warning.

### Optional (ask before starting)

Finite-β robustness at fixed coils (e.g. β = 1%); the 6-coil z = 0 counterpart of D (`relaxed/nc6z0/`; same QS as
the 5-coil state before single stage).

## Traps

All of `CLAUDE.md` "Traps learned", plus:
* `python cloud/harvest.py NAME` with the desc-dev python; miniconda's base python has no h5py. Call modal by full
  path `/home/jay/miniconda3/bin/modal` rather than prepending miniconda to PATH.
* A watchdog is `cloud/watchdog.sh APP DEADLINE_EPOCH JOB`; launch it with miniconda's bin on its own PATH.
* `.h5` and `.png` are gitignored: `git add -f`. The repo has autocrlf; check for CRLF in new scripts.
* Repair before a shell, never after the last one (it moves B·n by 0.2-0.4% on some boundaries).
* Another session uses the same Modal app name `from-axis` (job `qi_ss_free` seen on 2026-10-14); do not stop apps
  you did not launch.
