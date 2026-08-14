# Accelerating the Levenberg–Marquardt α loop in DESC: what worked, what didn't, and why

**Scope.** DESC's least-squares drivers (`lsqtr`, `lsq_auglag`) default to
`tr_method="qr"`. Per outer iteration the Jacobian `J_a` is factorized once, and the
resulting `(p_newton, Qᵀf_a, R)` triple is reused across an inner loop that re-solves
the trust-region subproblem at successively smaller radii. Inside each subproblem call, a
safeguarded Hebden/Reinsch root-find on φ(α) = ‖p(α)‖ − Δ re-factorizes
`A = [R; √α I]` from scratch at every α. `eq.solve` takes this path by default, and
`ProximalProjection` inherits it because it calls `eq.solve` for every inner
re-solve — so the α loop sits on both sides of proximal optimization.

![Four panels: (a) the structured QR sustains a constant 0.28x of dense QR's achieved efficiency across a 12x range in n; (b) consequently the per-factorization gain saturates near 1.6x rather than approaching the 5x flop bound; (c) batching is nearly free for GEMM and scatter but catastrophic for the panel QR, which hits a cuSOLVER cliff at B=8; (d) no measured option, batched or sequential, clears the break-even bar that speculative alpha evaluation demands.]({{artifact:art_d04f7734-d66e-499e-9b60-8cb9c2b0eb4a}})

This report covers two attempts to make that loop faster on GPU:

- **(a)** replacing the per-α dense QR of `[R; √α I]` with factorizations that exploit
  its structure;
- **(b)** batching several α values into one factorization call.

Both produce measurable per-factorization gains. Neither yields a significant
end-to-end speedup. The purpose of this document is to explain *why* in terms that
predict when they would work.

---

## 0. Hardware, software, and measurement protocol

All GPU numbers are from NVIDIA A100 SXM4 cards on Modal: the **40 GB** variant for the
first scaling sweep, the **80 GB** variant for all later sweeps. fp64 vector peak is
taken as **9.7 TFLOP/s** throughout (the A100 fp64 rate without tensor cores; the
tensor-core fp64 path at 19.5 TFLOP/s is not what `geqrf`/`gemm` dispatch to here).
JAX 0.11.0 for the linear-algebra microbenchmarks, JAX 0.9.2 for the in-DESC runs
(DESC pins `jax < 0.10`). float64 everywhere. CPU comparisons are an 8-core sandbox.

Three protocol points matter for reading the numbers:

1. **Compilation is excluded.** A cold DESC solve on A100 spends **42–46 s** in JIT
   compilation against a warm solve of **1.4–2.4 s**. Any single cold measurement is
   dominated by compilation.
2. **One method per process.** Two `tr_method`s in one process share the compilation
   cache for the objective and Jacobian; whichever runs second sees its wall time
   collapse ~10× while its subproblem time does not. Every end-to-end and profile
   number here comes from a fresh subprocess per method, three passes, warm = pass 3.
   An earlier version of the profiler violated this and produced α-loop shares of
   3.8–7.8% (cold rows) and 64–69% (warm rows in a shared process); **both are
   retracted** in favour of the isolated 16–36% below.
3. **Noise floor.** Three implementations that are algorithmically near-identical in
   cost (V0/V1/V3 of §1.2) differ by a median of **2.9%** and at most **13.3%** at
   fixed `(n, block)`. Read any single ratio below with that tolerance; differences
   under ~5% are not resolved.

**Problem sizes are the reduced ones.** The matrix `lsqtr` actually factorizes is the
constraint-projected reduced factor `R`, not the full DOF vector. Instrumenting real
solves gives `n` = **327** (HELIOTRON L6), **491** (precise_QA L6), **561**
(HELIOTRON L8), **723** (W7-X L6) — substantially smaller than the 543–1413 DOF counts.
This distinction is load-bearing: an early synthetic benchmark chose sizes from the DOF
counts and consequently overstated the achievable gain. Measured κ(R) on real arguments
is **1e9–3.7e12**, and captured α values span **2.9e-21 to 2.2e-14**.

**How much α work is there?** Instrumenting six perturbed equilibrium solves gives 241
boundary-hitting subproblem calls with this distribution of α factorizations per call:

| α factorizations | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| calls | 7 | 60 | 77 | 60 | 16 | 4 | 6 | 10 | 1 |
| share | 2.9% | 24.9% | 32.0% | 24.9% | 6.6% | 1.7% | 2.5% | 4.1% | 0.4% |

Mean **3.46**, mode 3, max observed 9. The hard cap `max_iter=10` was **never reached**:
the `rtol=0.01` test always fires first. Both constants are hard-coded at the call site
in `least_squares.py` and are not reachable through `options`.

---

## 1. (a) Exploiting the structure of `[R; √α I]`

### 1.1 The flop opportunity, correctly counted

`A = [R; √α I]` is `2n × n` with both blocks triangular. A dense Householder QR costs
`10n³/3` and ignores that structure. Eliminating column `j` only needs to touch the `~j`
bottom rows that have filled in while processing columns `0..j-1`, so the structured
minimum is `Σⱼ 4j(n−j) = (2/3)n³` — a **5× flop reduction**, at the same conditioning as
the dense QR (unlike a Cholesky of `RᵀR + αI`, which squares κ).

Two corrections to earlier estimates in this project are worth recording, because both
inflated the apparent prize:

- An initial claim that structure-exploitation is `O(n²)` (an 833×–6667× flop gap,
  growing with `n`) was **wrong**. There are ~n²/2 Givens rotations but each touches
  `O(n)` entries, so the cost is cubic. Implementing the Givens scheme and counting
  flops directly gave measured/n³ = 1.15 → 1.04 over n = 40…160.
- A subsequent claim of a **10×** flop advantage was also wrong: `(2/3)n³` against
  `10n³/3` is **5×**, and this is consistent with the measured Givens count of ~1.0 n³
  once its ~1.5× per-element penalty is accounted for.

### 1.2 What was implemented

A blocked column-panel sweep, LAPACK `dtpqrt`-style: at panel `k` spanning columns
`[c₀, c₁)`, the live rows are the `b` rows of `R`, the `b` diagonal rows of the bottom
block being annihilated, and the accumulated dense frontier of `c₀` previously-filled
rows. Householder per column within the panel, compact-WY update to the trailing
columns. This is exposed on the branch as `tr_method="qr-struct"` with a `tr_qr_block`
option; it shares the existing QR prep, reuses master's root-find unchanged, and
produces identical α iterates. Verified to `Rtil.T@Rtil == R.T@R + alpha*I` at ~1e-15
and steps matching master to ≤6.9e-12 on real captured solve arguments.

Four implementations were then compared to test whether the committed one leaves
performance on the table:

| | change |
|---|---|
| **V0** | the committed version |
| **V1** | stop re-applying `Qᵀ` to the panel columns the panel QR already produced |
| **V2** | true `dtpqrt` structure: panel QR sees only the live `2b × b` part, so the panel term is `O(b³)` not `O(n b²)`; frontier updated as GEMM |
| **V3** | V1 with contiguous top/bottom arrays instead of a gathered index scatter |

**Result: all four land within 2% of each other** (1.19–1.45× over dense, best block
`b`=128 at every size). V2 is best by 1–4%. V1/V2/V3 also improve the Gram residual from
V0's ~1.5e-15 to ~3e-16 by removing the redundant re-application, which is free.

There is no implementation win left in this algorithm class, and the block width matters
more than the algorithm: at n=723, `b` = 16/32/64/128/256 give 7.6/6.5/6.1/**6.0**/6.7 ms.

### 1.3 Why the 5× flops become only ~1.5× time

This is the central result of part (a), and it is not "the problem is too small" in the
naive sense.

Achieved fraction of A100 fp64 peak, computed against each routine's own flop count
(dense `10n³/3`, structured `(2/3)n³`):

| n | dense QR | structured | ratio |
|---|---|---|---|
| 327 | 0.38% | 0.10% | 0.25 |
| 491 | 0.79% | 0.21% | 0.27 |
| 561 | 0.97% | 0.28% | 0.28 |
| 723 | 1.56% | 0.45% | 0.29 |
| 1000 | 2.81% | 0.80% | 0.28 |
| 2000 | 9.56% | 3.05% | 0.32 |
| 4000 | 25.6% | 8.09% | 0.32 |

The structured routine runs at a **constant ≈0.28 ± 0.02 of the dense QR's achieved
efficiency across a 12× range in n**. Hence

> **5× fewer flops × 0.28× the efficiency ≈ 1.4× the speed.**

Check: 5 × 0.284 = 1.42 against a measured mean gain of 1.41. That identity is the whole
explanation, and it predicts the measured saturation: the gain goes 1.24 → 1.34 → 1.41 →
1.45 over n = 327…723, then **flattens** at 1.41, 1.59, 1.58 for n = 1000, 2000, 4000 —
flat between 2000 and 4000 to within the 2.9% noise floor.

**Why the handicap is constant.** Dense QR is a single large call into a tuned kernel.
The structured routine is `n/b` sequential panel steps, each containing a small panel QR
— a latency-bound primitive — plus GEMMs. Growing `n` makes *both* routines more
efficient in step (dense goes 0.38% → 25.6% of peak; structured 0.10% → 8.09%), so the
ratio between them is preserved. Nothing about increasing `n` alone closes it.

**Calibration: a speed floor.** A Cholesky of `RᵀR + αI` does `n³/3` — *half* the
structured minimum and a *tenth* of dense — yet it runs **4.5–9.0× faster than dense**,
i.e. better than its flop ratio predicts, because it is two big kernels with no panel
loop. That is the target a latency-free structured method would approach, and the gap
between 1.4× and 4.5–9× is the cost of the panel loop.

### 1.4 Conditions under which this route *would* pay

- **Not aspect ratio.** The α loop only ever sees `R`, which is `n × n` after the outer
  QR compresses `m → n`. The one dataset that appears to show an m/n dependence (the
  40 GB sweep: 0.99–1.31× at m/n=1.5 vs 1.25–1.59× at m/n=3, same n) is measuring an
  operation that cannot see `m`; the two series differ by ~25% for an unexplained
  run-to-run reason in that job. The m/n=3 series is cross-validated against the 80 GB
  run to within 1–7% and is used throughout. **Aspect ratio is not a lever here** —
  though it matters for the *prep* choice (§3).
- **Larger `n` helps only weakly**, and saturates. Going 327 → 4000 (12×) moves the gain
  1.24 → 1.58. Extrapolating the efficiency-ratio identity, reaching 3× would require the
  ratio to rise from 0.28 to 0.60 — a qualitative change in how the panel loop maps to
  the device, not a size effect.
- **What would actually work: removing the sequential panel structure.** The prize is
  the 0.28 efficiency ratio, not the flop count. A formulation whose inner operations are
  all large GEMMs would approach the Cholesky floor. §2 shows GEMM-only work batches
  essentially for free on this hardware, so the ingredient exists — the obstacle is that
  a QR needs a factorization primitive somewhere.
- **Tiled QR does not help at these sizes.** Tested separately: fp64 GEMM efficiency on
  A100-80GB is 1.5 GFLOP/s at size 64, 11.1 at 128, 77.1 at 256, rising to 10.6 TFLOP/s
  at 2048. Tile widths of 128–256 therefore run at **0.11–0.79% of peak**, and tile-op
  counts grow as ~q³/6 in the grid dimension q = n/b, which is only 4–8 across at
  n≈1000. An optimistic batched estimate built from *measured* TTQRT/TTMQR primitives
  loses to the column-blocked scheme by 1.7–15× at every size tested. This confirms a
  challenge raised externally against the tiled approach.
- **Sequential Givens is definitively out**: 3425 ms at n=500 against 4.49 ms for the
  dense QR — ~760× slower, and worsening with `n`, because it is n²/2 sequential rank-1
  updates.

### 1.5 End-to-end consequence

Even taking the per-call gain at face value, Amdahl's law bounds the payoff. Timing the
subproblem *inside* isolated warm solves on A100:

| case | true n | α-loop share | per-call | end-to-end |
|---|---|---|---|---|
| HELIOTRON L6 | 327 | 16.1% | 1.09× | 1.016× |
| precise_QA L6 | 491 | 22.8% | 1.17× | 1.023× |
| HELIOTRON L8 | 561 | 24.7% | 1.24× | 1.054× |
| W7-X L6 | 723 | 35.8% | 1.34× | **1.126×** |

Amdahl from (share, per-call) predicts the measured end-to-end within **1.2%** on all
four cases — two independently measured quantities, one predicting the other. The
per-call figures also agree within **4.4%** with a completely separate measurement that
captured real subproblem arguments and timed the two implementations on them outside the
solve.

On CPU the same flag is a **7–14% regression** (per-call 0.70–0.84×), because at n=327–723
the 5× flop advantage does not cover the extra kernel/dispatch overhead there. This is
why `qr-struct` should remain opt-in rather than becoming the default.

### 1.6 An unresolved trajectory divergence — read before trusting the flag

The accuracy verification above is **subproblem-level**: given identical
`(p_newton, z, R, Δ, α)`, the two methods return steps agreeing to ≤6.9e-12 and Gram
factors to ~1e-15. That is necessary but **not sufficient** for the full solve, and the
end-to-end runs show it. Relative difference in final cost between `tr_method="qr"` and
`"qr-struct"`, at `maxiter`=15:

| platform | case | iterations | subproblem calls (qr / struct) | final-cost rel. diff |
|---|---|---|---|---|
| A100 | HELIOTRON L6 | 15 / 15 | 26 / 26 | 1.1e-05 |
| A100 | precise_QA L6 | 15 / 15 | 29 / **30** | **6.61** |
| A100 | HELIOTRON L8 | 15 / 15 | 27 / 27 | 4.5e-03 |
| A100 | W7-X L6 | 15 / 15 | 30 / **27** | **0.176** |
| CPU | HELIOTRON L6 | 15 / 15 | 26 / 26 | 4.1e-07 |
| CPU | precise_QA L6 | 15 / 15 | 30 / 30 | 1.0e-05 |
| CPU | HELIOTRON L8 | 15 / 15 | 27 / 27 | 8.7e-04 |
| CPU | W7-X L6 | 15 / 15 | 30 / 30 | 5.9e-05 |

Two of four A100 cases end at materially different points — a **660%** relative
difference on precise_QA L6 and **17.6%** on W7-X L6 — despite identical iterate counts.
Round-off cannot explain that magnitude directly.

What the data does show:

- The two divergent cases are **exactly** the two where the methods made *different
  numbers of subproblem calls* (29 vs 30, 30 vs 27). The inner
  `while actual_reduction <= 0` loop is a data-dependent branch, so a ~1e-12 step
  difference that flips one acceptance test changes the number of trust-radius shrinks
  and sends the two runs down different trajectories. All four CPU cases made equal call
  counts and all four agree to ≤8.7e-04, consistent with this reading.
- These runs are **not converged** — `maxiter`=15 was a timing budget, so "final cost" is
  a mid-trajectory point of a nonlinear solve, where trajectory divergence is expected to
  be visible.
- A longer run (HELIOTRON L6, 200 iterations, isolated processes) gives
  qr = 4529.056899877351 and qr-struct = 4529.0568869344515, a **2.9e-09** relative
  difference. But *both* runs hit the iteration cap without converging, so this shows the
  two trajectories staying close over many more iterations — **not** agreement at a
  solution.

**Status: plausible mechanism, not verified.** The branch-flip explanation is consistent
with every observation but has not been confirmed by instrumenting the acceptance test
itself, and no converged comparison exists on any case. Until both are done, `qr-struct`
should not be treated as numerically interchangeable with `qr` at the level of a solve
trajectory, even though it is at the level of a single subproblem. The concrete tests
needed: log `actual_reduction` and the accept/reject decision per pass for both methods
on precise_QA L6, and run at least one case to genuine convergence
(`gtol`/`ftol` satisfied, not `maxiter`) under both flags.

---

## 2. (b) Batching over α

### 2.1 Why batching should have worked

Batching B α values does **not** add sequential steps. The panel loop still runs `n/b`
times; each step is simply B times wider. If the routine is latency-bound — which §1.3
establishes — then B α's should cost close to *one* sequential call, i.e. a gain
approaching B. Measuring 0.21–0.48× (B=2…8) means B=2 was ~4× *slower* than a single
call. That is a pathology, not a property of the algorithm, and the initial explanation
given for it ("batching just makes more kernels") was wrong.

### 2.2 A native batched implementation

To separate "batching is bad" from "`vmap` emitted something bad", a native batched
version was written: batch as the leading axis on contiguous arrays, every update an
`einsum` so XLA emits batched GEMM rather than gather/scatter, and the compact-WY factor
`T` formed **explicitly** (LAPACK `larft`-style) so applying it is a GEMM instead of a
batched triangular solve. Verified to ~1e-15 against the dense reference on every batch
member.

*Implementation note worth recording:* `T⁻¹ = VᵀV − diag(1/τ)` is **symmetric**, and the
reference path applies it via `solve_triangular(..., lower=True)`, which implicitly
discards the upper triangle. Inverting the full matrix gives 65% relative error; take
`tril` first.

### 2.3 The measurement: one primitive is responsible

Per-α cost of each batched primitive at the exact panel shape used, `(B, 851, 128)`,
A100-80GB:

| B | panel QR (geqrf) | vs B=1 | GEMM | vs B=1 | scatter | vs B=1 |
|---|---|---|---|---|---|---|
| 1 | 1.390 ms | 1.00 | 0.325 ms | 1.00 | 0.286 ms | 1.00 |
| 2 | 1.176 ms | 0.85 | 0.158 ms | 0.49 | 0.194 ms | 0.68 |
| 4 | 1.099 ms | 0.79 | 0.092 ms | 0.28 | 0.085 ms | 0.30 |
| 8 | **7.026 ms** | **5.05** | 0.063 ms | 0.19 | 0.067 ms | 0.23 |

**GEMM and scatter batch as they should** — per-α cost falls 5.1× and 4.3× at B=8, the
GPU absorbing the batch nearly for free. This confirms the batched *formulation* is
sound. **The batched QR does the opposite:** per-α cost *rises* 5.05× at B=8, and the
total for B=8 (56.2 ms) exceeds eight serial calls (11.1 ms) by 5×. There is a cliff
between B=4 (near-linear) and B=8 — a cuSOLVER strategy switch, not arithmetic.

The native version, `vmap`-over-V0, and `vmap`-over-V3 all land **within 4.4% of each
other**. The implementation route is irrelevant; all three bottom out in the same
`geqrf`.

### 2.4 The bar batching has to clear

Batching also requires an **algorithmic** change, and this is what ultimately closes the
route. The Hebden iterates are sequential: α_{k+1} depends on φ(α_k). A batched loop must
therefore evaluate candidate α's *speculatively*, before knowing whether it needs them.
A scheme doing `r` rounds of `B` evaluations performs `r·B` factorizations of work to
replace the 3.46 the sequential loop needs on average, so it pays off only if

> per-α batched cost / per-α sequential cost < 3.46 / (r·B)

which is **0.43×** for r=2, B=4; **0.29×** for r=3, B=4; **0.22×** for r=2, B=8.

Putting every measured option on one scale — per-α cost relative to a single sequential
dense QR at the same `n`, lower is better:

| n | dense, sequential | structured, sequential | dense, batched (best) | structured, batched (best) |
|---|---|---|---|---|
| 327 | 1.000 | 0.804 | 0.691 | 1.98 |
| 491 | 1.000 | 0.746 | 0.753 | 2.46 |
| 723 | 1.000 | 0.692 | 0.773 | 1.87 |
| 1000 | 1.000 | 0.709 | 0.774 | 1.47 |

Sequential structured QR (0.69–0.80) is the best option measured. Dense batching helps
relative to sequential dense (best 0.69) but does not beat sequential structured, and
**nothing measured comes within 1.6× of the 0.43 bar**.

One earlier statement needs correcting: dense-QR batching does **not** collapse at B=8
universally. Gains are 1.38/1.45/0.11 at n=327 and 1.30/1.33/0.03 at n=491 for
B=2/4/8 — but **1.28/1.29/1.29 at n=723 and 1.28/1.28/1.29 at n=1000**, i.e. it holds
through B=8 at the two larger sizes. The collapse is confined to the small-n cases.

### 2.5 Conditions under which batching *would* pay

- **It is a primitive-quality issue first.** The batched `geqrf` cliff is a library
  heuristic. If it were absent — per-α QR cost merely flat in B, as GEMM achieves — then
  B=4 batching of the structured routine would cost ~0.79 of sequential per α. Still
  above the 0.43 bar, but within striking distance.
- **It is a speculation-efficiency issue second.** The bar is set by 3.46/(r·B). Any
  change that reduces the number of speculative evaluations needed — a better φ model,
  a tighter initial bracket, a warm start that reliably lands within one iteration —
  lowers `r·B` and relaxes the bar proportionally. The measured warm-start behaviour
  already gets 2.9% of calls down to a single factorization; if that fraction were the
  norm, batching would have nothing left to amortize and would be moot anyway.
- **It is not a problem-size issue in the same way as (a).** Batching gets *relatively*
  better with `n` (structured batched: 1.98 → 1.47 from n=327 → 1000), but the trend is
  far too weak to reach the bar within DESC's range.
- **A GEMM-only factorization would change the picture entirely.** The one primitive that
  batches well is the one the structured method uses least.

### 2.6 What remains open in part (b)

Two things are diagnosed but not established, and should not be quoted as settled:

1. **The `geqrf` cliff is not demonstrably the sole cause.** A primitive-level model
   probed at a single (largest) panel shape — hence an upper bound per panel — leaves
   **+12.8 ms at B=2 and +53.4 ms at B=4** unexplained against the measured totals at
   n=723. At least one other batched operation degrades with B inside the real loop, and
   it has not been identified.
2. **Chunking is untested.** Issuing B=8 as two batched calls of B=4 is predicted from
   two measured points to recover ~6.4× of the cliff. The test script exists
   (`qr_cliff.py`: sweeps B=1…16 across five panel shapes, tests chunking, compares
   `raw` vs `reduced` mode) but was not run. Note that even a perfect result caps
   batching at the B=4 per-α rate, which is ~0.79 of sequential — still short of the bar.

---

## 3. Two other routes, closed for the record

**SVD of R.** With `R = W S Vᵀ` and `c = Vᵀ Rᵀ z` formed once, ‖p(α)‖² = Σ cᵢ²/(sᵢ²+α)²
and ‖q(α)‖² = Σ cᵢ²/(sᵢ²+α)³ are `O(n)` reductions: the α loop needs **no matrix
operation at all**, and `p` is formed with one matvec after convergence. On CPU this is
the clear winner. **On GPU it inverts:** the SVD of `R` costs 10–16× the outer QR on
A100 versus 1.3–2.0× on CPU (QR parallelizes well; the SVD's iterative bidiagonal phase
does not), which makes the prep more expensive than everything it saves. Worth
revisiting if a fast batched GPU SVD becomes available. Note the ordering question is
genuinely aspect-ratio dependent: QR-then-SVD(R) beats a direct SVD(J_a) once
m/n ≳ 1.5, which is DESC's regime.

**Cholesky with iterative refinement.** Form `G = RᵀR + αI` as a GEMM, Cholesky it, then
repair the squaring loss with refinement whose residual is computed via matvecs against
`R` (never `G`). This looked excellent — KKT residuals equal to or better than dense QR
at κ up to 1e12, at **4.1–5.5× dense speed**. It is nevertheless **unsafe for DESC**:
the shifted Gram is numerically singular when `(α + σ_min²)/σ_max² < ε`, a criterion
that correctly predicted every observed `nan`. Real captured DESC subproblems have
α ∈ [2.9e-21, 2.2e-14] with κ(R) up to 3.7e12, and **13 of 18 boundary-hitting calls
fall inside that regime**; `alpha_lower` is initialized to exactly `0.0`, so α = 0 is
always reachable. This confirms the known failure of the existing `"cho"` route and
locates its mechanism: the *solve* at a given α is fine, but the *root-find* evaluates
φ and φ′ through the squared operator and returns α = 0.

---

## 4. Summary and recommendation

**Open correctness item.** The largest unresolved question in this work is not
performance: it is the solve-trajectory divergence of §1.6. Two of four A100 cases finish
at substantially different points under the two flags. The mechanism is plausibly a
branch flip in the trust-region acceptance test, but that is unverified, and no converged
comparison has been run.

**(a) Structured QR achieves 1.09–1.34× per factorization at DESC's sizes and
1.016–1.126× end-to-end on A100.** The reason it is not 5× is a single quantitative fact:
the routine sustains a constant ~0.28 of dense QR's achieved efficiency, because it
replaces one large tuned kernel call with `n/b` sequential panel steps. 5 × 0.28 ≈ 1.4.
The gain saturates by n≈2000 and does not trend toward the flop bound. It is not an
aspect-ratio effect and only weakly a size effect; it is a consequence of the panel
loop's mapping onto the device.

**(b) Batching over α is worse than sequential at every point measured** (best 1.47× the
sequential per-α cost), against a break-even bar of 0.43× set by the speculation the
sequential root-find forces. The proximate cause is a batched-`geqrf` performance cliff
— GEMM batches ~5× *better* per item while QR gets 5× *worse* — but even removing the
cliff entirely leaves batching short of the bar.

**Recommendation.** Keep `tr_method="qr-struct"` as an **opt-in** flag with `b`=128: it is
a real 1.02–1.13× on GPU and a 7–14% regression on CPU. Its accuracy is verified *at the
subproblem level* — steps match master to ≤6.9e-12, and the Gram residual improves on the
committed variant if V2's panel structure is adopted — but **not at the level of a solve
trajectory**: two of four A100 cases end at materially different points (660% and 17.6%
relative in final cost), almost certainly because a ~1e-12 step difference flips a
trust-region acceptance test and changes the number of subproblem calls (§1.6). That must
be resolved before the flag is enabled by default or used for production physics, and it
is a correctness question independent of speed. Do not pursue tiled QR, sequential
Givens, batched α, or Cholesky-based routes at current problem sizes; the first three are
closed on measurement and the fourth on conditioning.

**The larger point.** With the α loop at 16–36% of a warm solve, the remaining 64–84%
— objective and Jacobian evaluation — is where a materially faster solve has to come
from. Within the α loop itself, the one untested lever with real upside is `rtol`: the
loop terminates on `rtol=0.01` and *never* on `max_iter=10` (0 of 241 calls), so a
cheaper factorization makes a tighter tolerance nearly free, and a more accurately solved
subproblem may reduce outer iteration count. Neither constant is currently exposed
through `options`.

---

## Appendix: data provenance

| file | contents |
|---|---|
| `alpha_loop_results.csv` | end-to-end and per-call results, both platforms, with call counts and compile times |
| `opt_bench_a100.json` | four implementations × five block widths, A100-80GB, plus Cholesky floor |
| `native_batch_a100.json` | batched primitives and the three batched implementations |
| `batch_alpha_a100.json` | batched-α sweep and Cholesky-refinement accuracy |
| `gpu_results_a100.json` | first scaling sweep, A100-40GB, n=500…4000, two aspect ratios |
| `profile_a100_isolated.json`, `profile_cpu_isolated.json` | isolated warm α-loop share |
| `insitu_a100.json` | real captured subproblem arguments: n, κ(R), α, per-call timings |
| `solve_summaries.json` | α-factorization distribution from six instrumented solves |
| `tile_real_a100.json`, `tile_bound_a100.json` | tiled-QR primitives and bound |
| `opt_struct.py`, `native_batch.py`, `qr_cliff.py` | implementations and the unrun cliff test |

Branch `js/lm-alpha-loop` carries the flag, all benchmark scripts under
`branch_experiments/`, and commit messages recording each result including the
retractions noted above.
