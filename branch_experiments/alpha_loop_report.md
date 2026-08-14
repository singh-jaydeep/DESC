# Accelerating the Levenberg–Marquardt α loop in DESC

### What worked, what didn't, and why

---

## The short version

DESC spends part of every equilibrium solve inside a small, tight loop that factorizes
the same kind of matrix over and over. That loop looked like an obvious target for
optimization, and this report describes two attempts to speed it up on GPU: exploiting
the special structure of the matrix being factorized, and factorizing several of them at
once.

Both attempts produced real, measurable improvements to the individual factorization.
Neither made equilibrium solves meaningfully faster. The interesting part is *why*, and
in both cases the answer turns out to be quantitative and predictive rather than a vague
"the problem is too small."

For the structured factorization, the explanation is a single number. The structured
algorithm does five times fewer arithmetic operations than the dense one it replaces, but
it sustains only 28% of the dense routine's efficiency on the hardware — and it does so
consistently, across a twelve-fold range of problem sizes. Five times fewer operations at
28% efficiency is 1.4 times faster, which is what we measure. Because that efficiency
ratio is *constant* rather than improving with size, growing the problem does not close
the gap; the gain saturates near 1.6× and stays there.

For batching, the explanation is a performance cliff in one library primitive. When we
batch the matrix multiplications, the GPU absorbs them almost for free — the cost per
item *falls* fivefold. When we batch the QR factorizations, the cost per item *rises*
fivefold. Everything else about batching works; that one call spoils it. And even a
perfect fix would not be enough, because batching this particular loop requires
speculative work, and the speculation sets a break-even bar that nothing we measured
comes close to.

There is also an unresolved correctness question, described in §1.6, which matters more
than either performance result: on two of four test cases the structured flag and the
existing one end their solves at substantially different points. The mechanism is
probably benign, but it is not verified, and it should be settled before anyone relies on
the flag.

![Four panels summarizing the two attempts. Panel a: the structured QR sustains a constant 0.28× of dense QR's achieved efficiency across a twelve-fold range in problem size. Panel b: consequently the per-factorization gain saturates near 1.6× rather than climbing toward the 5× flop bound. Panel c: batching is nearly free for matrix multiply and scatter but catastrophic for the panel QR, which hits a cuSOLVER cliff at batch size 8. Panel d: no measured option, batched or sequential, clears the break-even bar that speculative α evaluation demands.]({{artifact:art_d04f7734-d66e-499e-9b60-8cb9c2b0eb4a}})

---

## Where the loop lives and why it matters

DESC's least-squares drivers — `lsqtr` and `lsq_auglag` — default to `tr_method="qr"`.
Once per outer iteration they factorize the Jacobian `J_a`, and then reuse the resulting
`(p_newton, Qᵀf_a, R)` triple across an inner loop that re-solves the trust-region
subproblem at successively smaller trust radii. Hoisting the Jacobian factorization out
of that inner loop is already done in master and is the right design.

What remains inside is the α loop. Solving the trust-region subproblem means finding the
Levenberg–Marquardt parameter α at which the step length matches the trust radius, which
is a one-dimensional root-find on

    φ(α) = ‖p(α)‖ − Δ

carried out by a safeguarded Hebden/Reinsch iteration. Every trial value of α requires
factorizing

    A = [R; √α I]

from scratch. That is the factorization this report is about.

The loop is reached by more code than it first appears. `eq.solve` takes this path by
default, and `ProximalProjection` has no trust-region step of its own — it calls
`eq.solve` for each inner re-solve — so the α loop sits on both sides of proximal
optimization. Making it faster would benefit equilibrium solves and stellarator
optimization alike, which is what motivated the work.

Two routes were pursued:

1. Replace the per-α dense QR of `[R; √α I]` with a factorization that exploits the fact
   that both blocks of that matrix are already triangular.
2. Batch several α values into a single factorization call, so the GPU sees one large
   piece of work instead of several small ones.

---

## 0. How the measurements were made

Everything below is measured, and a few protocol details determine whether the numbers
mean anything, so they come first.

**Hardware and software.** All GPU numbers come from NVIDIA A100 SXM4 cards on Modal: the
40 GB variant for the first scaling sweep, the 80 GB variant for every later sweep. Where
the two overlap they agree to within 1–7%, which is a useful cross-check on the whole
apparatus. Throughout, the A100's fp64 vector peak is taken as **9.7 TFLOP/s** — the rate
without tensor cores, since `geqrf` and `gemm` do not dispatch to the 19.5 TFLOP/s
tensor-core fp64 path in these shapes. The linear-algebra microbenchmarks ran under JAX
0.11.0; the in-DESC runs under JAX 0.9.2, because DESC pins `jax < 0.10`. Everything is
float64. CPU comparisons, where they appear, are from an 8-core sandbox.

**Compilation has to be excluded, and this bit us.** A cold DESC solve on the A100 spends
**42–46 seconds** in JIT compilation, against a warm solve of **1.4–2.4 seconds**. Any
single cold measurement is therefore almost entirely a measurement of the compiler. Worse,
running two `tr_method`s in the same process lets the second one inherit the first's
compilation cache for the objective and Jacobian: its wall time collapses by roughly 10×
while its subproblem time does not, which silently inverts any comparison. Every
end-to-end and profile number here comes from a fresh subprocess per method, three passes,
with the warm measurement taken from the third.

An earlier version of the profiler did not isolate the methods, and it produced two
different wrong answers for the same quantity — α-loop shares of 3.8–7.8% from the cold
rows, and 64–69% from the warm rows of a shared process. **Both are retracted.** The
isolated measurement gives 16–36%, and that is what is used below.

**Noise floor.** Three of the implementations compared in §1.2 (V0, V1, V3) are
algorithmically near-identical in cost, so their spread at fixed problem size and block
width is a direct estimate of run-to-run noise. That spread has a median of **2.9%** and a
maximum of **13.3%**. Any single ratio quoted below should be read with that tolerance;
differences under about 5% are not resolved by this data.

### The problem sizes that actually matter

One detail deserves emphasis because getting it wrong cost us a round of misleading
results. The matrix `lsqtr` factorizes is not sized by the equilibrium's degree-of-freedom
count. It is the *constraint-projected reduced factor* `R`, which is considerably smaller.
Instrumenting real solves gives:

| case | reduced size `n` | (DOF count, for contrast) |
|---|---|---|
| HELIOTRON L6 | 327 | 543 |
| precise_QA L6 | 491 | 961 |
| HELIOTRON L8 | 561 | 833 |
| W7-X L6 | 723 | 1413 |

An early synthetic benchmark picked its sizes from the DOF counts, and since the
structured method's advantage grows with `n`, it overstated the achievable gain. All sizes
quoted in this report are reduced sizes. On real captured arguments, κ(R) ranges over
**1e9–3.7e12** and the α values DESC actually visits span **2.9e-21 to 2.2e-14** — both
facts matter later, in §3.

### How much α work is there per solve?

Instrumenting six perturbed equilibrium solves produced 241 subproblem calls that hit the
trust-region boundary (and therefore ran the α loop at all). The number of α
factorizations they needed:

| α factorizations | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| calls | 7 | 60 | 77 | 60 | 16 | 4 | 6 | 10 | 1 |
| share of calls | 2.9% | 24.9% | 32.0% | 24.9% | 6.6% | 1.7% | 2.5% | 4.1% | 0.4% |

The mean is **3.46**, the mode is 3, and the most ever observed was 9. Notably the hard
cap of `max_iter=10` was **never reached** — the `rtol=0.01` convergence test always fired
first. Both constants are hard-coded at the call site in `least_squares.py` and are not
reachable through `options`, a point we return to at the end.

That mean of 3.46 becomes important in §2, where it sets the bar batching has to clear.

---

## 1. Route (a): exploiting the structure of `[R; √α I]`

### 1.1 How much is theoretically available

The matrix `A = [R; √α I]` is `2n × n`, and both of its blocks are triangular — `R` from
the Jacobian factorization above it, a scaled identity below. A dense Householder QR costs
`10n³/3` and throws that information away. If instead you track which entries are
structurally nonzero, eliminating column `j` only requires touching the roughly `j` bottom
rows that have filled in while processing columns `0..j-1`. Summing gives the structured
minimum:

    Σⱼ 4j(n−j) = (2/3)n³

which is a **5× flop reduction**. Crucially it achieves this at the same conditioning as
the dense QR, unlike a Cholesky factorization of `RᵀR + αI`, which squares the condition
number — a distinction that turns out to be decisive in §3.

Two earlier estimates in this project were wrong in the optimistic direction, and both are
worth recording so the corrected figure isn't mistaken for pessimism:

The first claim was that structure-exploitation is `O(n²)`, implying a flop gap of
833×–6667× that would *grow* with problem size. This is wrong. There are indeed about
`n²/2` Givens rotations, but each one touches `O(n)` entries of the working row, so the
total is cubic, not quadratic. Implementing the Givens scheme and counting flops directly
confirmed it: measured flops divided by `n³` came out at 1.15 falling to 1.04 across
n = 40…160.

The second claim was a **10×** flop advantage. Also wrong: `(2/3)n³` against `10n³/3` is
**5×**. This is consistent with the measured Givens count of about `1.0 n³` once that
scheme's roughly 1.5× per-element penalty is taken into account.

So the honest prize going in was 5×, not 10× and certainly not a growing gap.

### 1.2 What was implemented, and whether it was implemented well

The algorithm is a blocked column-panel sweep in the style of LAPACK's `dtpqrt`, the
routine that exists precisely for a dense block stacked on a triangular one. At panel `k`
spanning columns `[c₀, c₁)`, the rows that can have nonzeros in those columns are the `b`
rows of `R`, the `b` diagonal rows of the bottom block currently being annihilated, and
the accumulated dense "frontier" of `c₀` rows filled in by earlier panels. Within a panel
the elimination is Householder-per-column; the trailing columns are then updated with a
compact-WY transformation, which turns what would be `b` separate rank-1 updates into two
matrix multiplies and a triangular solve.

This is exposed on the branch as `tr_method="qr-struct"`, with a `tr_qr_block` option for
the panel width. It reuses the existing QR prep and master's root-find unchanged, so it
produces identical α iterates by construction. It is verified to satisfy
`Rtil.T @ Rtil == R.T @ R + alpha*I` at about 1e-15, with steps matching master to
≤6.9e-12 on real captured solve arguments. (This is subproblem-level verification. §1.6
explains why that is not the same as verifying the solve.)

Before concluding anything about the algorithm, it is worth asking whether the
*implementation* is leaving performance on the table. Three inefficiencies were visible in
the committed code, so three alternatives were written to remove them:

| variant | what it changes |
|---|---|
| **V0** | the committed version |
| **V1** | stops re-applying `Qᵀ` to the panel columns whose transformed values the panel QR already computed |
| **V2** | the true `dtpqrt` structure: the panel QR sees only the genuinely live `2b × b` part, making the panel term `O(b³)` instead of `O(n b²)`, with the frontier updated as a matrix multiply |
| **V3** | V1 with contiguous top and bottom arrays instead of a gathered index scatter |

**All four land within 2% of one another** — 1.19–1.45× over dense, with block width 128
best at every size. V2 wins by 1–4%. V1, V2 and V3 also improve the Gram residual from
V0's ~1.5e-15 to ~3e-16 by removing the redundant re-application, which is free accuracy
and a good reason to adopt V2's structure regardless.

The conclusion is that there is no implementation win left in this algorithm class. Block
width matters more than the algorithm does: at n=723, widths of 16, 32, 64, 128 and 256
give 7.6, 6.5, 6.1, **6.0** and 6.7 ms respectively.

### 1.3 Why 5× fewer flops buys only 1.4× less time

This is the central result of route (a), and it is more specific than "the matrices are
small."

Consider how efficiently each routine uses the hardware. Computing achieved throughput
against each routine's *own* flop count (dense `10n³/3`, structured `(2/3)n³`) and
expressing it as a fraction of the A100's 9.7 TFLOP/s fp64 peak:

| `n` | dense QR | structured QR | ratio |
|---|---|---|---|
| 327 | 0.38% | 0.10% | 0.25 |
| 491 | 0.79% | 0.21% | 0.27 |
| 561 | 0.97% | 0.28% | 0.28 |
| 723 | 1.56% | 0.45% | 0.29 |
| 1000 | 2.81% | 0.80% | 0.28 |
| 2000 | 9.56% | 3.05% | 0.32 |
| 4000 | 25.6% | 8.09% | 0.32 |

The last column is the finding. The structured routine sustains a **constant 0.28 ± 0.02
of the dense QR's achieved efficiency, across a twelve-fold range in `n`**. From that, the
whole result follows as arithmetic:

> **5× fewer flops × 0.28× the efficiency ≈ 1.4× the speed.**

The check works: 5 × 0.284 = 1.42, against a measured mean gain of 1.41. And it predicts
the saturation we observe. The gain climbs 1.24 → 1.34 → 1.41 → 1.45 across n = 327…723,
then **flattens**: 1.41, 1.59 and 1.58 at n = 1000, 2000 and 4000, the last two identical
to within the 2.9% noise floor.

Why should the handicap be constant? Dense QR is a single large call into a heavily tuned
kernel. The structured routine is `n/b` sequential panel steps, each containing a small
panel QR — a latency-bound primitive, since a QR of a narrow panel cannot fill the device
— plus matrix multiplies. Increasing `n` makes *both* routines more efficient in step:
dense goes from 0.38% to 25.6% of peak, structured from 0.10% to 8.09%. The ratio between
them survives. Nothing about increasing `n` alone closes it.

It helps to calibrate against a speed floor. A Cholesky factorization of `RᵀR + αI` does
`n³/3` flops — *half* the structured minimum and a *tenth* of the dense count — and it
runs **4.5–9.0× faster than dense**, which is *better* than its flop ratio alone would
predict. The reason is that it is two big kernel calls with no panel loop at all. That is
roughly what a latency-free structured method would achieve, and the distance between our
1.4× and that 4.5–9× is precisely the cost of the sequential panel structure.

### 1.4 So under what conditions *would* this route pay?

Since the question is not simply "bigger problems," it is worth being specific about which
knobs matter and which don't.

**Aspect ratio is not a lever.** The α loop only ever touches `R`, which is `n × n`
regardless of how tall `J_a` was — the outer QR has already compressed `m → n` before the
loop starts. One dataset appears to contradict this: the 40 GB sweep shows 0.99–1.31× at
m/n = 1.5 against 1.25–1.59× at m/n = 3 for the same `n`. But that is a ~25% difference in
an operation that structurally cannot see `m`, so it is unexplained run-to-run variance in
that one job rather than a real effect. The m/n = 3 series cross-validates against the
80 GB run to within 1–7%, and it is DESC's regime, so it is the series used throughout.
(Aspect ratio *does* matter for choosing the prep, as §3 notes — just not for the α
iteration.)

**Problem size helps, but only weakly, and it saturates.** Going from n = 327 to n = 4000
— a factor of twelve — moves the gain from 1.24 to 1.58. Extrapolating through the
efficiency identity, reaching even 3× would require the efficiency ratio to rise from 0.28
to 0.60. That is a qualitative change in how the panel loop maps onto the device, not
something more `n` delivers.

**What would actually work is removing the sequential panel structure.** The prize is the
0.28 efficiency ratio, not the flop count. A formulation whose inner operations were all
large matrix multiplies would approach the Cholesky floor, and §2 demonstrates that
GEMM-shaped work batches essentially for free on this hardware — so the ingredient exists.
The obstacle is that a QR fundamentally needs a factorization primitive somewhere, and
that primitive is where the efficiency is lost.

**Tiled QR does not help at these sizes**, which is worth stating because it is the
sophisticated thing one would naturally try next. Measured fp64 GEMM efficiency on the
A100-80GB is 1.5 GFLOP/s at size 64, 11.1 at 128 and 77.1 at 256, only reaching
10.6 TFLOP/s at 2048. Tile widths of 128–256 therefore operate at **0.11–0.79% of peak**.
Meanwhile tile-operation counts grow as roughly `q³/6` in the grid dimension `q = n/b`,
which is only 4–8 across at n ≈ 1000 — too few tiles to expose useful parallelism. An
optimistic estimate built from *measured* TTQRT/TTMQR primitives loses to the simple
column-blocked scheme by 1.7–15× at every size tested. This confirms a challenge raised
externally against the tiled approach.

**Sequential Givens rotations are definitively out.** At n = 500 the Givens scheme takes
3425 ms against 4.49 ms for the dense QR — about **760× slower**, and worsening with `n`,
because it is `n²/2` sequential rank-1 updates with essentially no parallelism to exploit.

### 1.5 What this means for a whole solve

Even granting the per-factorization gain, Amdahl's law bounds what it can do to a solve.
Timing the subproblem *inside* isolated warm solves on the A100:

| case | reduced `n` | α-loop share of solve | per-call gain | end-to-end gain |
|---|---|---|---|---|
| HELIOTRON L6 | 327 | 16.1% | 1.09× | 1.016× |
| precise_QA L6 | 491 | 22.8% | 1.17× | 1.023× |
| HELIOTRON L8 | 561 | 24.7% | 1.24× | 1.054× |
| W7-X L6 | 723 | 35.8% | 1.34× | **1.126×** |

Two independent consistency checks support these figures. First, Amdahl's law applied to
the measured share and per-call gain predicts the measured end-to-end gain to within
**1.2%** on all four cases — two separately measured quantities, one predicting the other.
Second, the per-call figures agree to within **4.4%** with an entirely different
measurement, which captured real subproblem arguments from a live solve and timed both
implementations on them outside the solve.

On CPU the same flag is a **7–14% regression** (per-call 0.70–0.84×). At n = 327–723 the
5× flop advantage does not cover the extra kernel and dispatch overhead there. This
platform split is the main reason `qr-struct` should stay opt-in rather than becoming the
default.

### 1.6 An unresolved trajectory divergence — read this before trusting the flag

The accuracy verification in §1.2 is **subproblem-level**: given identical inputs
`(p_newton, z, R, Δ, α)`, the two methods return steps agreeing to ≤6.9e-12 and Gram
factors to ~1e-15. That is necessary but **not sufficient** for the full solve, and the
end-to-end runs make the gap visible. Relative difference in final cost between
`tr_method="qr"` and `"qr-struct"` at `maxiter`=15:

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

Two of the four A100 cases finish at materially different points — a **660%** relative
difference on precise_QA L6 and **17.6%** on W7-X L6 — despite identical iterate counts.
Round-off alone does not explain differences of that magnitude.

What the data does show is suggestive. The two divergent cases are **exactly** the two
where the methods made *different numbers of subproblem calls* (29 versus 30, and 30
versus 27). The inner `while actual_reduction <= 0` loop is a data-dependent branch, so a
step difference of order 1e-12 that flips a single acceptance test changes how many
trust-radius shrinks occur, and from there the two runs follow different trajectories.
Consistent with this, all four CPU cases made equal numbers of calls and all four agree to
≤8.7e-04.

Two caveats keep this from being conclusive. These runs are **not converged** — `maxiter`
was set to 15 as a timing budget — so "final cost" is a mid-trajectory point of a
nonlinear solve, exactly where trajectory divergence would be expected to show. And a
longer run (HELIOTRON L6, 200 iterations, isolated processes) gives
qr = 4529.056899877351 against qr-struct = 4529.0568869344515, a relative difference of
**2.9e-09** — but *both* of those runs hit the iteration cap without converging, so the
comparison shows two trajectories staying close over many more iterations, not agreement
at a solution.

**Status: plausible mechanism, not verified.** The branch-flip explanation is consistent
with every observation, but it has not been confirmed by instrumenting the acceptance test
itself, and no converged comparison exists on any case. Until both are done, `qr-struct`
should not be treated as numerically interchangeable with `qr` at the level of a solve
trajectory, even though it is interchangeable at the level of a single subproblem. Two
concrete tests would settle it: log `actual_reduction` and the accept/reject decision per
pass for both methods on precise_QA L6, and run at least one case to genuine convergence
(satisfying `gtol`/`ftol` rather than hitting `maxiter`) under both flags.

---

## 2. Route (b): batching over α

### 2.1 Why batching should have worked

The reasoning that motivates batching is sound, and worth spelling out because it is what
makes the eventual result surprising.

Batching B values of α does **not** add sequential steps. The panel loop still runs `n/b`
times; each step is simply B times wider. If the routine is latency-bound — which §1.3
establishes it is — then B α's ought to cost not much more than *one* sequential call,
giving a speedup approaching B.

What we measured instead was 0.21–0.48× across B = 2…8. At B = 2 the batched version was
roughly 4× *slower* than a single call. That is not what a latency-bound routine does when
you widen it, so it is a pathology to be diagnosed rather than a property of the algorithm.
An initial explanation — that batching "just makes more kernels" — was simply wrong, since
the number of sequential steps is unchanged.

### 2.2 Ruling out the implementation

The first possibility is that `vmap` emitted something inefficient rather than batching
being inherently bad. To separate those, a native batched implementation was written from
scratch, designed so that nothing about the batching could be blamed on the framework:
batch as the leading axis on contiguous arrays; every update expressed as an `einsum` so
XLA emits a batched matrix multiply rather than a gather/scatter; and the compact-WY
factor `T` formed **explicitly**, LAPACK `larft`-style, so applying it is a matrix multiply
instead of a batched triangular solve. It verifies to ~1e-15 against the dense reference on
every batch member.

One implementation detail is worth recording because it produced a convincing wrong
answer. The matrix `T⁻¹ = VᵀV − diag(1/τ)` is **symmetric**, and the reference
implementation applies it via `solve_triangular(..., lower=True)`, which implicitly
discards the upper triangle. Inverting the full matrix instead gives a 65% relative error
— take `tril` first.

### 2.3 The diagnosis: one primitive is responsible

Timing each batched primitive separately, at the exact panel shape the real loop uses,
`(B, 851, 128)`, on the A100-80GB:

| B | panel QR (`geqrf`) | per-α vs B=1 | GEMM | per-α vs B=1 | scatter | per-α vs B=1 |
|---|---|---|---|---|---|---|
| 1 | 1.390 ms | 1.00 | 0.325 ms | 1.00 | 0.286 ms | 1.00 |
| 2 | 1.176 ms | 0.85 | 0.158 ms | 0.49 | 0.194 ms | 0.68 |
| 4 | 1.099 ms | 0.79 | 0.092 ms | 0.28 | 0.085 ms | 0.30 |
| 8 | **7.026 ms** | **5.05** | 0.063 ms | 0.19 | 0.067 ms | 0.23 |

The matrix multiply and the scatter behave exactly as batched operations should: per-α cost
*falls* by 5.1× and 4.3× respectively at B = 8, with the GPU absorbing the extra work
almost for free. This is important beyond itself — it confirms that the batched
*formulation* is sound, so the problem is localized.

The batched QR does the opposite. Its per-α cost *rises* by 5.05× at B = 8, and the total
for a batch of 8 (56.2 ms) exceeds the cost of eight serial calls (11.1 ms) by a factor of
5. There is a cliff somewhere between B = 4, where scaling is still near-linear, and
B = 8 — the signature of a cuSOLVER strategy switch, not of arithmetic.

Confirming that the route through the framework is irrelevant: the native version,
`vmap` over V0, and `vmap` over V3 all land **within 4.4% of each other**. All three
bottom out in the same `geqrf` call.

### 2.4 The bar batching has to clear

Diagnosing the primitive is not the end of the story, because batching this loop also
demands an *algorithmic* change, and that is what ultimately closes the route.

The Hebden iterates are sequential by nature: α_{k+1} depends on φ(α_k). A batched loop
therefore has to evaluate candidate α values **speculatively**, before knowing whether it
will need them. A scheme performing `r` rounds of `B` evaluations does `r·B` factorizations
of work to replace the 3.46 that the sequential loop needs on average. It pays off only if

> (per-α batched cost) / (per-α sequential cost) < 3.46 / (r·B)

which works out to **0.43×** for r=2, B=4; **0.29×** for r=3, B=4; and **0.22×** for
r=2, B=8.

Putting every measured option on a single scale — per-α cost relative to one sequential
dense QR at the same `n`, where lower is better:

| `n` | dense, sequential | structured, sequential | dense, batched (best) | structured, batched (best) |
|---|---|---|---|---|
| 327 | 1.000 | 0.804 | 0.691 | 1.98 |
| 491 | 1.000 | 0.746 | 0.753 | 2.46 |
| 723 | 1.000 | 0.692 | 0.773 | 1.87 |
| 1000 | 1.000 | 0.709 | 0.774 | 1.47 |

Sequential structured QR, at 0.69–0.80, is the best option measured anywhere in this
project. Dense batching does help relative to sequential dense — best 0.69 — but it does
not beat sequential structured, and **nothing measured comes within 1.6× of the 0.43
bar**.

One earlier statement of ours needs correcting here. Dense-QR batching does **not**
collapse at B = 8 universally. The gains are 1.38/1.45/0.11 at n = 327 and 1.30/1.33/0.03
at n = 491 for B = 2/4/8, so the collapse is real at small `n` — but at the two larger
sizes it holds: **1.28/1.29/1.29 at n = 723 and 1.28/1.28/1.29 at n = 1000**. The collapse
is confined to the small-`n` cases, and the categorical claim was wrong.

### 2.5 Under what conditions would batching pay?

**It is a primitive-quality issue first.** The batched `geqrf` cliff is a library
heuristic, not a hardware limit. If it were absent — if per-α QR cost were merely *flat*
in B, which is less than the matrix multiply already achieves — then B = 4 batching of the
structured routine would cost about 0.79 of sequential per α. Still above the 0.43 bar,
but within striking distance rather than off by a factor of several.

**It is a speculation-efficiency issue second.** The bar itself is `3.46/(r·B)`, so
anything that reduces the number of speculative evaluations required relaxes it
proportionally: a better model of φ, a tighter initial bracket on α, or a warm start that
reliably lands within one iteration. There is a nice tension here — the measured warm-start
behaviour already gets 2.9% of calls down to a single factorization, and if that fraction
were the norm, batching would have almost nothing left to amortize and would be moot
anyway.

**It is not a problem-size issue in the way route (a) is.** Batching does get relatively
better with `n` — structured batched improves from 1.98 to 1.47 going from n = 327 to
n = 1000 — but the trend is far too weak to reach the bar anywhere inside DESC's range.

**A GEMM-only factorization would change the picture entirely**, which is the same
conclusion route (a) reached from the other direction. The one primitive that batches well
is the one the structured method leans on least.

### 2.6 What remains open in route (b)

Two things here are diagnosed but not established, and should not be quoted as settled.

First, **the `geqrf` cliff is not demonstrably the sole cause.** A primitive-level model
probed at a single panel shape — the largest one, so an upper bound on per-panel cost —
still leaves **+12.8 ms at B = 2 and +53.4 ms at B = 4** unexplained against the measured
totals at n = 723. At least one other batched operation degrades with B inside the real
loop, and it has not been identified.

Second, **chunking is untested.** Issuing B = 8 as two batched calls of B = 4 is predicted
from two measured points to recover about 6.4× of the cliff, which would be a trivial fix
if it works. The test script exists — `qr_cliff.py` sweeps B = 1…16 across five panel
shapes, tests chunking, and compares `raw` against `reduced` mode — but it was not run.
Note that even a perfect result caps batching at the B = 4 per-α rate of roughly 0.79 of
sequential, which is still short of the bar.

---

## 3. Two other routes, closed for the record

Two further approaches were implemented and tested before the two above. Neither survived,
but both failed for instructive reasons.

**An SVD of `R` alone.** This is an elegant idea. Writing `R = W S Vᵀ` and forming
`c = Vᵀ Rᵀ z` once, the quantities the root-find needs become

    ‖p(α)‖² = Σᵢ cᵢ²/(sᵢ²+α)²    and    ‖q(α)‖² = Σᵢ cᵢ²/(sᵢ²+α)³

which are `O(n)` reductions over vectors. The α loop then requires **no matrix operation
at all**: α enters only as a diagonal shift, and `p` is formed with a single matvec after
the root-find converges. On CPU this is the clear winner.

**On GPU it inverts.** The SVD of `R` costs 10–16× the outer QR on the A100, against
1.3–2.0× on CPU. QR parallelizes well on GPU; the SVD's iterative bidiagonal phase does
not. The preparation therefore ends up costing more than everything the cheap α loop saves.
This one is worth revisiting if a fast batched GPU SVD becomes available. Note also that
the ordering question here *is* genuinely aspect-ratio dependent: QR-then-SVD(R) beats a
direct SVD of `J_a` once m/n ≳ 1.5, which is DESC's regime.

**Cholesky with iterative refinement.** Form `G = RᵀR + αI` as a matrix multiply, take its
Cholesky factor, then repair the accuracy lost to squaring the condition number by
iterative refinement whose residual is computed with matvecs against `R` — never against
`G` — so the refinement sees the true operator conditioning. This looked excellent: KKT
residuals equal to or better than the dense QR's at κ up to 1e12, at **4.1–5.5× dense
speed**.

It is nevertheless **unsafe for DESC**, and the boundary is sharp. The shifted Gram matrix
is numerically singular when

    (α + σ_min²) / σ_max² < ε

a criterion that correctly predicted every `nan` we observed. Real captured DESC
subproblems have α ∈ [2.9e-21, 2.2e-14] with κ(R) up to 3.7e12, which puts **13 of 18
boundary-hitting calls inside that failing regime** — and since `alpha_lower` is
initialized to exactly `0.0`, α = 0 is always reachable. This confirms the known failure of
the existing `"cho"` route and locates its mechanism precisely: the *solve* at a given α is
fine, but the *root-find* evaluates φ and φ′ through the squared operator, and returns
α = 0.

---

## 4. Summary and recommendation

**The most important open item is not about performance.** It is the solve-trajectory
divergence of §1.6: two of four A100 cases finish at substantially different points under
the two flags. The mechanism is plausibly a branch flip in the trust-region acceptance
test, but that is unverified and no converged comparison has been run.

**Route (a)** achieves 1.09–1.34× per factorization at DESC's sizes, and 1.016–1.126×
end-to-end on the A100. The reason it is not 5× reduces to one quantitative fact: the
routine sustains a constant ~0.28 of dense QR's achieved efficiency, because it replaces
one large tuned kernel call with `n/b` sequential panel steps. Five times fewer flops at
0.28 efficiency is about 1.4× faster. The gain saturates by n ≈ 2000 and does not trend
toward the flop bound. It is not an aspect-ratio effect and only weakly a size effect; it
is a consequence of how the panel loop maps onto the device.

**Route (b)** is worse than sequential at every point measured — the best case is 1.47× the
sequential per-α cost — against a break-even bar of 0.43× set by the speculation the
sequential root-find forces. The proximate cause is a batched-`geqrf` performance cliff, in
which the matrix multiply gets about 5× *better* per item while the QR gets 5× *worse*.
But even removing the cliff entirely leaves batching short of the bar.

**Recommendation.** Keep `tr_method="qr-struct"` as an **opt-in** flag with block width
128. It is a genuine 1.02–1.13× on GPU and a 7–14% regression on CPU. Its accuracy is
verified *at the subproblem level* — steps match master to ≤6.9e-12, and the Gram residual
improves on the committed variant if V2's panel structure is adopted — but **not at the
level of a solve trajectory**, where two of four A100 cases end at materially different
points (660% and 17.6% relative in final cost), almost certainly because a ~1e-12 step
difference flips a trust-region acceptance test and changes the number of subproblem calls.
That must be resolved before the flag is enabled by default or used for production physics;
it is a correctness question independent of speed.

Do not pursue tiled QR, sequential Givens, batched α, or Cholesky-based routes at current
problem sizes. The first three are closed on measurement, the fourth on conditioning.

**Finally, the larger point about where solve time actually goes.** With the α loop at
16–36% of a warm solve, the remaining 64–84% — objective and Jacobian evaluation — is where
a materially faster solve has to come from. Any further work aimed at total solve time
should start there rather than in the linear algebra.

Within the α loop itself, the one untested lever with real upside is `rtol`. The loop
terminates on `rtol=0.01` and *never* on `max_iter=10` — zero of 241 calls reached the cap
— so a cheaper factorization makes a tighter tolerance nearly free, and a more accurately
solved subproblem may reduce the outer iteration count, which would be worth more than the
factorization speedup itself. Neither constant is currently exposed through `options`.

---

## Appendix: data provenance

Every number in this report traces to one of these files.

| file | contents |
|---|---|
| `alpha_loop_results.csv` | end-to-end and per-call results, both platforms, with call counts, compile times and final-cost differences |
| `opt_bench_a100.json` | four implementations × five block widths, A100-80GB, plus the Cholesky speed floor |
| `native_batch_a100.json` | batched primitives and the three batched implementations |
| `batch_alpha_a100.json` | batched-α sweep and Cholesky-refinement accuracy |
| `gpu_results_a100.json` | first scaling sweep, A100-40GB, n = 500…4000, two aspect ratios |
| `profile_a100_isolated.json`, `profile_cpu_isolated.json` | isolated warm α-loop share |
| `insitu_a100.json` | real captured subproblem arguments: `n`, κ(R), α, per-call timings |
| `solve_summaries.json` | α-factorization distribution from six instrumented solves |
| `tile_real_a100.json`, `tile_bound_a100.json` | tiled-QR primitives and the resulting bound |
| `opt_struct.py`, `native_batch.py`, `qr_cliff.py` | implementations, and the unrun cliff test |

Branch `js/lm-alpha-loop` carries the flag itself, all benchmark scripts under
`branch_experiments/`, and commit messages recording each result — including the
retractions noted above, which are kept rather than quietly corrected.
