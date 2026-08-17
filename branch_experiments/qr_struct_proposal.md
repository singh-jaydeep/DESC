# `tr_method="qr-struct"`: a LAPACK-style structured QR for the LM α loop

### Proposal summary, scope of code changes, demonstrated gains, expected weaknesses, and what to validate next

---

## What it is, in one paragraph

Inside the trust-region subproblem, the current `"qr"` path factorizes
`A = [R; √α I]` — a `2n × n` matrix — with a general dense Householder QR, once per
trial value of α. But both blocks of that matrix are already triangular: `R` from the
Jacobian factorization above it, a scaled identity below. A general QR cannot know that
and pays `10n³/3` flops. Tracking which entries are structurally nonzero instead gives
`(2/3)n³`, a **5× flop reduction at the same conditioning** — no squaring of κ(J), unlike
the `"cho"` route. The algorithm that does this is LAPACK's `dtpqrt`, the routine that
exists specifically for a dense block stacked on a triangular one, implemented as a
blocked column-panel sweep with compact-WY updates. `tr_method="qr-struct"` on branch
`js/lm-alpha-loop` is that algorithm, wired in beside `"qr"` with everything else held
fixed.

The measured payoff is **1.09–1.34× per α factorization and 1.016–1.126× end-to-end** on
A100, and a **7–14% regression on CPU**. Whether that is worth merging is a judgment call;
this document lays out what it would cost, what has been shown, and what has not.

---

## (a) What code changes are needed

The change is additive and small. Nothing in the existing `"qr"` path is modified, so
`tr_method="qr"` remains byte-identical in behaviour.

### Already implemented on the branch

**`desc/optimize/tr_subproblems.py` — 161 lines added, 0 removed.** Three new functions:

| function | lines | role |
|---|---|---|
| `_apply_QT_wy` | 24 | applies `Qᵀ` from packed Householder reflectors via the compact-WY identity `T⁻¹ + T⁻ᴴ = VᴴV`, so each panel update is two matmuls and a triangular solve rather than `b` rank-1 updates |
| `structured_retriangularize` | 65 | the blocked panel sweep itself; returns `(Rtil, Qtz)` with `Rtil.T @ Rtil == R.T @ R + alpha*I` |
| `trust_region_step_exact_qr_struct` | 69 | drop-in replacement for `trust_region_step_exact_qr`: same signature plus `block=`, same safeguarded Hebden root-find, same `(step, hits_boundary, alpha)` return |

Plus `import functools` for the `static_argnames=("block",)` jit wrapper.

**`desc/optimize/least_squares.py` — 33 lines changed.** Five small edits:

1. one import;
2. `tr_qr_block = options.pop("tr_qr_block", 128)`;
3. `"qr-struct"` added to the `tr_method` validation list and its error message;
4. `elif tr_method == "qr":` → `elif tr_method in ("qr", "qr-struct"):` at the prep block,
   so both variants share the existing Jacobian factorization unchanged;
5. a four-line `elif` branch at the subproblem call site.

Plus the docstring entry for both new options.

### One deliberate implementation choice worth flagging in review

`_apply_QT_wy` **duplicates** logic that already exists as
`desc.backend._blocked_householder_multiply`. That helper is defined only inside the
`jax < 0.10` fallback branch of `backend.py`, so it is not available on the JAX versions
where `qr_multiply` is native. A reviewer may reasonably prefer that the helper be hoisted
out of the version guard and shared rather than duplicated — that is a cleanup, not a
functional change, and it would shrink the diff.

### Not yet done, and needed for parity

**`lsq_auglag` is not wired.** `desc/optimize/aug_lagrangian_ls.py` has its own
`tr_method` plumbing (lines 320, 328, 407, 440) that still accepts only
`["cho", "svd", "qr"]`. The edits are mechanically identical to the five above — the
subproblem call site is the same function with the same arguments. Until that is done, the
flag reaches equilibrium solves and `proximal-lsq-exact` but not the augmented-Lagrangian
least-squares driver. `fmintr` and `aug_lagrangian` use a **separate** `tr_method`
namespace (`"exact"`/`"dogleg"`/`"subspace"`) and are unaffected by design.

**No tests.** Existing `tr_method` coverage is exercise-only: `test_optimizer.py` runs
`lsqtr` under `"cho"` and `"svd"` on a small vector-valued fit and asserts the solution is
recovered. Adding `"qr-struct"` to that pattern is a two-line change and should be the
minimum bar for merge. See (d) for what stronger testing would look like.

**CHANGELOG entry.**

---

## (b) What speedups have been demonstrated, and on what

Three independent measurements, all on A100. The distinction between them matters, because
they answer different questions.

### 1. Microbenchmark on synthetic `R` (A100-80GB, best block per size)

| `n` | dense QR | structured | gain |
|---|---|---|---|
| 327 | 3.18 ms | 2.56 ms | 1.243× |
| 491 | 5.17 ms | 3.85 ms | 1.341× |
| 561 | 6.25 ms | 4.44 ms | 1.405× |
| 723 | 8.34 ms | 5.77 ms | 1.446× |
| 1000 | 12.25 ms | 8.68 ms | 1.411× |

Extending to larger sizes (A100-40GB, m/n = 3): 1.250× at n=500, 1.396× at n=1000,
**1.590× at n=2000, 1.578× at n=4000** — i.e. the gain saturates rather than climbing
toward the 5× flop bound.

### 2. On real captured subproblem arguments (A100)

Arguments `(p_newton, z, R, Δ, α)` captured from live DESC solves, then both
implementations timed on them outside the solve: **18 boundary-hitting calls, per-call
speedup 1.08–1.33×, median 1.25×**, with maximum step relative error versus master of
**6.9e-12**.

### 3. Inside real equilibrium solves (A100, isolated warm, `maxiter`=15)

| case | `n` | α-loop share | per-call | end-to-end |
|---|---|---|---|---|
| HELIOTRON L6 | 327 | 16.1% | 1.09× | 1.016× |
| precise_QA L6 | 491 | 22.8% | 1.17× | 1.023× |
| HELIOTRON L8 | 561 | 24.7% | 1.24× | 1.054× |
| W7-X L6 | 723 | 35.8% | 1.34× | **1.126×** |

Two consistency checks support these. Measurement 3's per-call figures agree with
measurement 2 to within **4.4%** — independent methodologies, same answer. And Amdahl's law
applied to the measured α-loop share and per-call gain predicts the measured end-to-end
gain to within **1.2%** on all four cases.

**The honest headline is the end-to-end column: 1.02–1.13×, best on the largest case.**
The α loop is only 16–36% of a warm solve, so the per-factorization gain is diluted by
construction. Anyone quoting 1.4× for this change would be quoting the microbenchmark, not
a solve.

### Measurement caveats

All solve numbers are warm (compilation excluded — a cold A100 solve spends **42–46 s**
compiling against a warm solve of 1.4–2.4 s), one method per process (sharing a process
lets the second method inherit the first's compilation cache and collapses its wall time
~10×), median of repeated passes. Run-to-run noise on the microbenchmarks has a median of
**2.9%** and a maximum of **13.3%**, so the 1.016× on HELIOTRON L6 is at the edge of
resolution while the 1.126× on W7-X L6 is not.

---

## (c) Where this should be expected to be worse

### It is a regression on CPU — 7–14%

| case | `n` | per-call | end-to-end |
|---|---|---|---|
| HELIOTRON L6 | 327 | 0.70× | 0.871× |
| precise_QA L6 | 491 | 0.80× | 0.928× |
| HELIOTRON L8 | 561 | 0.84× | 0.911× |
| W7-X L6 | 723 | 0.81× | 0.863× |

At these sizes the 5× flop advantage does not cover the extra kernel and dispatch overhead
on CPU. **This is the strongest argument for keeping the flag opt-in rather than making it
the default**, since DESC runs on both.

### Small problems are the weak regime, and the mechanism is known

The structured routine sustains a constant **≈0.28 of dense QR's achieved efficiency**
across n = 327…4000, because it replaces one large tuned kernel call with `n/b` sequential
panel steps whose panel QR cannot fill the device. Five times fewer flops at 0.28
efficiency is ≈1.4× — which is what we measure, and why the gain does not grow. The
underlying reason small shapes are inefficient is stark: measured fp64 GEMM on A100-80GB
runs at 1.5 GFLOP/s at size 64, 11.1 at 128 and 77.1 at 256, only reaching 10.6 TFLOP/s at
2048. So the smaller the equilibrium, the worse this looks — and DESC's reduced sizes sit
at the unfavourable end.

### Block width is a real tuning knob, and a bad choice costs ~25%

At each size, worst-versus-best block width across `b` ∈ {16, 32, 64, 128, 256} is
**1.25–1.28×**. 128 was best at every size tested on A100, which is why it is the default,
but that is an A100 result. On a different card — including the RTX-class GPUs many users
develop on — the optimum may move, and at `b`=16 the structured routine is **slower than
dense outright** (1.16× at n=327, 1.02–1.03× at n=491/561/1000). A user who tunes
`tr_qr_block` downward on the assumption that fewer flops is better will make things worse.

### Compile time grows as the block width shrinks

The panel loop is a Python `for` over `⌈n/b⌉` panels, unrolled at trace time, so emitted
HLO scales with the panel count. Measured compile deltas versus `"qr"` at `b`=128 are small
(−0.6 s, +0.4 s, +1.7 s on the three cases, i.e. −1.3% to +3.9% of a 42–46 s compile), but
the panel count is 6 at n=723/`b`=128 and would be **46 at `b`=16** — and 32 at
n=4000/`b`=128, 250 at `b`=16. Anyone scaling up should watch compile time, and a
`lax.fori_loop` rewrite would remove the growth at the cost of a more awkward
implementation.

### No differentiability, but no regression either

Neither path can be differentiated: both raise
`NotImplementedError: Differentiation rule for 'geqrf' not implemented`. This is a
pre-existing constraint of the `"qr"` route, not something the new flag introduces, and it
is consistent with the note in #2244 that `qr_multiply` awaits an upstream JAX PR. Worth
knowing if the trust-region step ever needs to be differentiated through.

### What is *not* a concern

Memory is a wash — the structured work array is `2n × (n+1)` against the dense stacked
`2n × n`, e.g. 256.1 MB versus 256.0 MB at n=4000. Aspect ratio is irrelevant: the α loop
only sees `R`, which is `n × n` after the outer QR compresses `m → n`. Wide Jacobians
(`m < n`) work — verified at (m,n) = (400,500) and (250,500) with Gram residuals of
1.35e-15 and 1.41e-15.

---

## (d) What additional validation would most strengthen the case

Ordered by what would change a merge decision.

### 1. Resolve the solve-trajectory divergence — blocking

This is the one item that could sink the change, and it is a correctness question, not a
performance one. On two of four A100 cases the two flags finish at **materially different
points** despite identical iterate counts: final-cost relative differences of **6.61**
(precise_QA L6) and **0.176** (W7-X L6), against 1.1e-05 and 4.5e-03 on the other two.

The likely mechanism is benign but unproven. The two divergent cases are **exactly** the two
where the methods made different numbers of subproblem calls (29 vs 30, and 30 vs 27). The
inner `while actual_reduction <= 0` loop is a data-dependent branch, so a ~1e-12 step
difference that flips one acceptance test changes the number of trust-radius shrinks and
sends the runs down different paths. Consistent with that, all four CPU cases made equal
call counts and agree to ≤8.7e-04. The runs are also unconverged (`maxiter`=15 was a timing
budget), so "final cost" is a mid-trajectory point where divergence is expected to show.

Two cheap tests would settle it, neither needing a GPU:

- Log `actual_reduction` and the accept/reject decision per inner pass for both methods on
  precise_QA L6, and confirm the divergence begins at a single flipped comparison.
- Run at least one case to **genuine convergence** — `gtol`/`ftol` satisfied, not
  `maxiter` — under both flags, and compare final cost and `‖∇f‖`. A 200-iteration run on
  HELIOTRON L6 gives 4529.056899877351 versus 4529.0568869344515 (2.9e-09 relative), but
  *both* hit the iteration cap, so that shows trajectory proximity, not agreement at a
  solution.

### 2. Confirm on non-datacenter GPU hardware

Every GPU number here is A100. The two things most likely to move on a different card are
the optimal block width and the `geqrf`/`gemm` efficiency balance that sets the 0.28 ratio.
`run_laptop_gpu.sh` on the branch does this in one command against a local `desc-env`; it
aborts if JAX cannot see a GPU or if the wrong DESC is imported, so it cannot silently
return CPU numbers. Given the A100 result I would expect ~1.0–1.1× end-to-end on an
RTX-class card, but the per-call number is the informative one.

### 3. Tests that would justify merge

The existing `tr_method` tests are exercise-only. Three tiers, cheap to expensive:

- **Unit, subproblem level.** Given random `R` across shapes (tall, square, wide), block
  widths and α values spanning `[0, ‖Rᵀz‖/Δ]`, assert
  `Rtil.T @ Rtil ≈ R.T @ R + alpha*I` and that `(step, hits_boundary, alpha)` matches
  `trust_region_step_exact_qr` to ~1e-12. Include rank-deficient `R` (exactly zero
  singular values) and α = 0 exactly, since `alpha_lower` is initialized to `0.0`.
- **Integration.** Add `"qr-struct"` to the `test_optimizer.py` pattern that already runs
  `"cho"` and `"svd"`, asserting the same solution is recovered.
- **Regression.** One equilibrium solved to convergence under both flags, asserting equal
  final cost to a stated tolerance. This is item 1 turned into a test, and it is the one
  that would have caught the divergence.

### 4. Extend to `lsq_auglag` and confirm the proximal path

The wiring is mechanical, but the proximal path deserves an explicit measurement rather than
an inference. `ProximalProjection` calls `eq.solve` for every inner re-solve, so on the
argument that each of those is an `lsqtr` run the gain should carry through — but the
per-solve work there is smaller and more repetitive, and compile-cache effects across many
short solves could behave differently from the single-solve case measured here.

### 5. Adopt V2's panel structure before merging

Four implementations were compared. All land within 2% on speed, but **V2** (panel QR sees
only the live `2b × b` part) is best by 1–4% *and* improves the Gram residual from V0's
~1.5e-15 to ~3e-16 by not re-applying `Qᵀ` to panel columns the panel QR already produced.
The committed version is V0. Switching is a small edit with free accuracy, and it makes the
code a more honest match to the `dtpqrt` structure the docstring claims.

### 6. Measure `rtol` as a coupled knob

The α loop terminates on `rtol=0.01` and **never** on `max_iter=10` (zero of 241 instrumented
calls reached the cap). A cheaper factorization makes a tighter `rtol` nearly free, and a
more accurately solved subproblem may reduce the *outer* iteration count — which would be
worth more than the factorization speedup itself. Neither constant is currently reachable
through `options`. This is the most promising unexplored direction in the whole area, and it
is orthogonal to which factorization is used.

---

## Recommendation

Merge as an **opt-in** flag with `b`=128 — after item 1 above is resolved, because a
correctness question should not ship behind a performance flag. Adopt V2's panel structure,
add the unit and integration tests, wire `lsq_auglag`, and either share
`_blocked_householder_multiply` or note the duplication deliberately.

Set expectations honestly in the docstring: this is **1.02–1.13× on GPU for equilibrium
solves at current problem sizes, and a 7–14% regression on CPU**. The 5× flop reduction is
real but converts to ~1.4× per factorization because of the constant efficiency handicap,
and the α loop is only 16–36% of a solve. The change is most defensible as
scale-preparation — the per-call gain reaches 1.59× at n=2000, so the flag becomes more
attractive if DESC's reduced systems grow — and as a correct, better-conditioned
alternative to the `"cho"` route, which fails outright in the α ≪ σ_min² regime where 13 of
18 real captured subproblems live.

Supporting data and the full analysis are in `alpha_loop_report.md`; all benchmark scripts
are under `branch_experiments/` on `js/lm-alpha-loop`.
