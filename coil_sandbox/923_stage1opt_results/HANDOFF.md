# Handoff: stage-1 eps_1 optimization

Written 2026-09-25 for whoever picks this up next. You need three files:

* **this one** — orientation, workflow, traps
* **`SESSION.md`** (next to this) — the 2026-09-23 results and their caveats
* **`../922_stage1opt_results/NOTES.md` §9m–9p** — the chronology, including every wrong turn

Read section 5 of this file before you run anything, and section 6 of `SESSION.md` before you
quote any number.

---

## 1. The question

**Does an equilibrium's shape determine whether it can be built with planar coils, and can we
optimize for that at stage 1?**

The proxy: on the ρ=1 surface the required sheet current is `K = -(1/mu0) n x B`. It has a stream
function Φ, and **contours of Φ at equal increments are proxy modular coils** — exact streamlines,
equal-current, crowding where |K| is large (Rodriguez & Sengupta arXiv:2604.12339; `NOTES.md` §3).

The metric on each proxy curve:

```
eps_1  =  sqrt(lambda_0) / (lambda_1 lambda_2)^(1/4)
```

λ₀ ≤ λ₁ ≤ λ₂ are the three eigenvalues of the arclength-weighted covariance of the curve's
points. Numerator = RMS distance from the best-fit plane; denominator = the curve's own in-plane
extent. **Dimensionless, scale-free: "out-of-plane thickness ÷ how big the curve is".** For a
circle of radius `a` with an out-of-plane wobble of amplitude `h`, eps_1 = h/a.

`eps_2` is the same for the best 2-plane split (clamshell), and `C = eps_2/(eps_1/4)` compares it
to the generic `B^-2` decay. **C is parked** — the tension with eps_1 that §9k reported turned out
to be an aspect-ratio artifact, so anything built on it needs re-deriving.

**Response variable at stage 2**: the ratio `bn_planar / bn_FourierXYZ` at a fixed coil budget.
Never `bn_planar` alone — it cannot distinguish "better for planar coils" from "easier for all
coils".

---

## 2. Where everything is

```
coil_sandbox/
  922_stage1opt_results/        THE CODE
    stage1_v1.py                the stage-1 driver (every knob is a flag; --help)
    contour_curve.py            ContourClamshell: contours built INSIDE the objective
    clamshell.py                differentiable eps_B (Danskin); `python clamshell.py` self-tests
    coil_complexity.py          L_gradB and pdrot on a boundary
    analyze_result.py           audit an optimized equilibrium
    NOTES.md                    chronology. 9m-9p are this work; 9n's title is RETRACTED
    HANDOFF.md                  older agenda; its banner points forward
    runs/                       every run, with logs
  923_stage1opt_results/        THIS FOLDER — the 2026-09-23 artifacts
    SESSION.md  README.md  equilibria/  coilsets/  viewers/  logs/  diagnostics/
  run_one.py                    stage-2 coil optimization
  make_start.py                 circles / arcs starting coilsets
  view.py                       offline 3D HTML viewer
  bounds/precise_qa_planar.json ABSOLUTE coil bounds — use these, not a-scaled
  sandbox/common.py             equilibrium loading, the dense check
```

### The equilibria you actually want

All are vacuum precise_QA descendants. Baseline: `desc.examples.get("precise_QA")`,
eps_1 = 0.067633, A = 6.0, |force| 2.07e-05.

| file (in `equilibria/`) | eps_1 | A | use it for |
|---|---|---|---|
| **`worst_leg3.h5`** | **−17.8%** | 6.010 | **the best result — start here** |
| `aspin_leg2_iotatied.h5` | −32.9% | 6.010 | largest clean eps_1 reduction, but damaged (min L_gradB −25.8%) |
| `aspin_leg1.h5` | −10.3% | 6.010 | mildest; complexity metrics *improve* |
| `lsq_fb500.h5` | −27.2% | **6.25** | aspect ratio drifted — confounded |
| `aspin_leg2_iotaleak.h5` | −33.8% | 6.010 | **DO NOT USE** — iota exceeded its bound via a now-fixed leak |

Their coilsets are in `coilsets/{baseline,fb500,aspin3,worst3}_{planar,xyz}.h5`, all from an
identical coil problem so `baseline_*` is the control. Viewers in `viewers/`.

---

## 3. The workflow

### Environment, every time

```python
import sys; sys.path.insert(0, "sandbox")   # from the sandbox root
import boot; boot.setup(default="gpu")      # BEFORE anything imports desc
```

JAX fixes its platform at first import. `python check.py` verifies the vendored DESC is the one
in use. Interpreter: `conda run --no-capture-output -n desc-env2 python -u ...` — see trap 5.1.

### Stage 1 — the current best configuration

```bash
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0 \
  conda run --no-capture-output -n desc-env2 python -u 922_stage1opt_results/stage1_v1.py \
  --quantity eps1 --arm lo --target-mode worst --target-frac 0.60 --no-qs \
  --K 8 --nodes 360 --bound-aspect 0.01 --bound-iota 0.05 \
  --method proximal-lsq-exact --fb-weight 500 \
  --lgradb-floor 1.0 --lgradb-weight 5 --lgradb-grid 24 \
  --ftol 1e-8 --xtol 1e-10 --gtol 1e-10 --maxiter 300 --device gpu \
  --out 922_stage1opt_results/runs/<name>  > <name>/log.txt 2>&1
```

Why each piece:

* `--target-mode worst` — a single one-sided bound `(0, frac·max(base))`. Curves below it
  contribute zero residual **and zero gradient**, so effort goes to the offenders. The old
  proportional target asked every curve for the same *fractional* cut, which subsidises the
  already-planar ones because they are cheapest to move.
* `--bound-aspect 0.01` — **non-negotiable**. With ±0.25 the optimizer spends the whole budget on
  aspect ratio, pinning A at ~6.25 in every run, and that confounded months of earlier results.
* `--lgradb-floor 1.0` — floor on min `L_grad(B)` at the *original* equilibrium's value. Keep
  `--lgradb-backoff` at 1.0 (see trap 5.7).
* `--method proximal-lsq-exact` — not `lsq-auglag`, which stalls (§9m.2), and not `proximal-lsq`,
  which does not exist. With a non-AL method the driver automatically moves aspect/iota into the
  objective as bounded terms; do not put them in `constraints` or `ProximalProjection` swallows
  them.
* `--no-qs` — QS is dropped. **This is an open confound**; nothing here has been run with QS held.

Chain legs with `--from <prev>/eq_lo.h5`. Reference files (`iota0_lo.npy`, `lgradb0_lo.npy`,
`BASE0_lo.npy`) propagate so bounds stay tied to the original.

### Stage 2 — coils

Copy `diagnostics/run_coil_matched.sh`, change `tag` and `EQPATH`. It handles the GPU-clear wait
and retries. Identical problem for every equilibrium, which is what makes `baseline_*` a valid
control:

```
--bounds bounds/precise_qa_planar.json --vacuum --mode stage2 --maxiter 200
--qf-chunk 5    # REQUIRED for rep=xyz: 4/4 die without it
--nc 4          # 16 coils, r/a 2.5 cold start
```

### Diagnostics (all in `diagnostics/`, all cheap)

| script | answers |
|---|---|
| `res_gap2.py` | eps_1 at L=8 vs L=12 — **is the gain real or unresolved boundary?** |
| `coil_complexity.py` | min L_gradB/a and pdrot·a vs baseline |
| `eps_dist.py` | per-contour eps_1 by baseline rank — **did the hard curves move?** |
| `local_global.py` | L_gradB ratio field — is the damage local or global? |
| `scales.py` / `blockscale2.py` | weight scales; per-term Jacobian block norms |

### Viewers

```bash
python view.py out.html --eq <eq.h5> --h5 planar=<result.h5> --h5 xyz=<result.h5> \
  --streamlines 24 --field-ref precise_QA --bounds bounds/precise_qa_planar.json --vacuum --device cpu
```

`--streamlines K` draws the proxy coils shaded by their own eps_1; `--field-ref` adds the L_gradB
heatmap (absolute, or diverging ratio vs a reference). Compare `view_baseline.html` against
`view_worst3.html` to see the foliation compress rather than stretch.

---

## 4. What we know

**Works:**

* Targeting the **worst** contours inverts who moves — hardest quartile −29.6% vs easiest −19.9%,
  and the foliation spread stays at 1.73× instead of blowing out to 3.43×.
* An **L_gradB floor** costs almost nothing (min L_gradB −1.81% at eps_1 −17.8%) and keeps pdrot's
  median improving instead of degrading.
* Together: planar coils −9.23%, **0.519% per eps_1 point** vs proportional's 0.450, in the same
  constraint regime as the control.

**Does not work / is confounded:**

* Letting aspect ratio float. It pins at its bound and dominates everything.
* `lsq-auglag` — stalls with optimality frozen at 9.4× `gtolk`, which the outer loop needs to
  cross before it can fire.
* Proportional targeting — stretches the foliation and barely touches the hard curves.
* The observational program (§5c): 8 configurations, no dynamic range, HSX ranked lowest.

**Unresolved:**

* **Only 1 stage-1 leg in 3 is productive**, identical configuration. Four mechanisms proposed,
  all four falsified. See trap 5.2.
* No stage-1 run has reached a stationary point. Every eps_1 number is a lower bound.
* The force term carries 99.997% of ‖J‖² pre-proximal but the resolution-gap control needs its
  large weight. It may belong somewhere other than the objective.
* QS is destroyed everywhere.

---

## 5. Traps — each of these cost real time

**5.1 `conda run` buffers stdout.** Without `--no-capture-output` the log stays empty until the
process exits, so a 20-minute run is invisible and a crash leaves no trail. Always
`conda run --no-capture-output -n desc-env2 python -u`.

**5.2 Never conclude "configuration X does not work" from one stalled run.** Restart it first. A
restart reset something that let an identical problem go from −10.3% in 300 iterations to −32.9%
in 12. Legs with the same trust radius and the same constraint status behaved oppositely. I
proposed four mechanisms for this (proximal re-solve noise, initial trust radius, a floor's zero
slack, a kink in the bound) and **every one was falsified within minutes**. The untried isolation:
save `precise_QA` to `.h5` and run the fresh config through `--from` — same equilibrium, same
objective, only the fresh-build-vs-restart code path differs.

**5.3 `--tr-ratio` does not fix stalls.** A 2500× larger initial trust radius changed nothing. The
tell: leg 1's first step was 1.003e-03 against a radius of 4637 — never trust-region limited at
all, the Gauss-Newton step itself was tiny.

**5.4 Read the saved artifact, not the optimizer's summary line.** DESC's end-of-run
"Maximum absolute Force error" is evaluated mid-step on the objective's own grid. It read
1.028e-02 where the *saved* equilibrium was 1.555e-04. I was caught by this twice.

**5.5 `--from` re-centres bounds unless they are tied to a reference.** `iota0` was recomputed on
the restarted geometry, granting a fresh ±0.05 every leg; a continuation took the whole thing in
12 iterations. Fixed for iota and the L_gradB floor via saved `.npy` references. **If you add
another bounded quantity, tie it the same way.**

**5.6 Weights chosen to equalize *cost* do not equalize the *Jacobian*.** Gauss-Newton builds its
step from JᵀJ. Measured: the force term's residual is 77× smaller than eps_1's and its Jacobian
171× larger. Use `blockscale2.py` before trusting a weight.

**5.7 A bounds objective is invisible to the model until it activates.** `bounds=(floor, inf)`
gives zero residual *and* zero Jacobian rows while satisfied. Set a floor with slack, not exactly
at the current value. And `--lgradb-backoff 1.01` puts the floor 1% above the baseline min, so
14/2401 nodes violate at the start and the term costs as much as the entire objective — keep it at
1.0.

**5.8 Machine rules.** WSL2, ~15 GB RAM, RTX 5070 8 GB. One heavy job at a time, always under
`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`. Verify `nvidia-smi` shows **zero
compute apps** before launching — a leftover process holding VRAM killed two unrelated runs. To
kill by pattern use the bracket trick (`pkill -f 'stage1_v[1][.]py'`) or the command kills its own
shell.

**5.9 `--qf-chunk 5` is required for FourierXYZ coils.** 4/4 die at the default. Planar is fine.
Retry up to 3× regardless — there is a documented ~1-in-8 intermittent CUDA fault.

**5.10 Quote pdrot's *median*.** It carries a `1/(c²+s²)` factor, so points near umbilics (where
principal directions are undefined) inflate the max without bound — we saw maxima of 3e5. The
script reports umbilic proximity; check it.

**5.11 Bounds must be absolute, not a-scaled.** `a` varies across the ladder, and an a-scaled
bound would move with the thing under test.

**5.12 Resolution.** Always run `res_gap2.py`. An eps_1 gain of −39.55% at L=8 was −28.88% at
L=12. The gap is a monotonic function of |force| at L=8 — which is why the soft ForceBalance term
exists. With `--fb-weight 500` the gap is ~0.3 points.

---

## 6. Next, in the order I would do it

1. **Repeat the worst-mode run.** It rests on one productive leg. Until it reproduces, treat
   section 4's "works" list as provisional.
2. **Isolate the fresh-build vs restart difference** (trap 5.2). It is one cheap test and it
   undermines every single-run conclusion until understood.
3. **Intermediate targets** — `--target-frac 0.85, 0.80` — to map how far worst-mode goes before
   the complexity metrics turn. There is a hint of a benign regime below ~10% that has never been
   properly established.
4. **The database sweep — no GPU needed, and the highest information per hour.** QUASR and
   Constellaration ship equilibria *with* optimized coils, so eps_1 / pdrot / L_gradB can be
   regressed against *achieved* coil complexity (length, curvature, `eta_SVD = sigma_3/sum(sigma)`)
   across thousands of devices. If eps_1 adds nothing over pdrot — which already gets R² = 0.700
   univariate against coil non-planarity — the question closes without another intervention. This
   inverts §5c's failure, which was 8 hand-picked configurations.
5. **A pdrot objective**, if the L_gradB floor keeps paying off. pdrot is the predictor most
   directly opposed to what we want and nothing currently blocks it, but it needs the director
   field in `coil_complexity.py` made differentiable — the branch-free `2*alpha = atan2(s, c)`
   formulation is already there, it just needs to go into JAX.
6. **Hold QS** (drop `--no-qs`) and see whether eps_1 still moves. This confound has been open
   since §9l and nothing has tested it.
