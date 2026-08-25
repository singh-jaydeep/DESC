# `qr-fixed` vs master's `qr` — session handoff

Branch `js/qr-fixed-integrated`, forked from `js/lm-alpha-slim`, with upstream
PRs #2293 and #2286 merged in.

The goal was to compare a new trust-region subproblem method against master's
`qr` on **speed, memory and accuracy**. This file says what is established, what
was retracted, and — most importantly — **which traps to avoid**, because
several of them silently produce plausible-looking numbers that are wrong.

Read `PERTURBATION.md` before writing any benchmark code. Read `ledger.md` for
the chronological record including the false starts.

---

## 1. What the method is

`tr_method="qr-fixed"` replaces the per-alpha dense QR of `[R; sqrt(alpha)*I]`
with a blocked, `dtpqrt`-style retriangularization that exploits both blocks
already being triangular: `(2/3)n^3` flops instead of `10n^3/3`, at the same
conditioning as `qr` (unlike `cho`, which squares `kappa(J)`).

The branch originally carried two other variants. Both were dropped:

* **`qr-struct`** — the structured retriangularization without the
  trailing-columns-only optimisation. Dominated by `qr-fixed` on all three
  axes, so there is no reason to keep it.
* **`qr-slim`** — `qr-struct` plus two changes. Its second change, carrying the
  frontier at its exact live shape instead of in a fixed `2n x (n+1)` buffer,
  turns planned temp memory from `O(n^2)` into **`O(n^3/block)`**, and the peak
  is **non-monotonic in block width**, so no default can be chosen safely.
  Planned totals at the width `least_squares` picks:

  | n | case | `qr` | `qr-struct` | `qr-fixed` | `qr-slim` |
  |---|---|---|---|---|---|
  | 14242 | precise_QA L20 | 6.1 GB | 6.5 | 6.5 | 7.4 |
  | 26896 | precise_QA L25 | 21.6 GB | 23.0 | 23.0 | **48.0** |
  | 38830 | HELIOTRON L25 | 45.0 GB | 47.9 | 47.8 | **187.5 → OOM** |

  `qr-fixed` keeps `qr-slim`'s *first* change — applying `Q^T` only to the
  trailing columns, since the panel QR already produced the rest — which
  carries the accuracy gain and part of the speed, while retaining `O(n^2)`
  memory.

---

## 2. Established results

All on A100, fp64, `precise_QA`, unless stated.

### Speed

| measurement | result |
|---|---|
| per factorization, n=3434 | 1.54x |
| per factorization, n=14242 | **2.26x** (2.29x post-merge) |
| alpha loop in-solve, L=20 | **2.19x** |
| alpha loop in-solve, L=12 | 0.78x — `qr-fixed` is **slower** here |
| per outer iteration, L=20 | ~1.15x |
| per outer iteration, L=12 | ~0.96x |

The per-factorization number is the method-intrinsic one. End-to-end is bounded
by Amdahl: the alpha loop is ~30% of an outer iteration at L=20 and much less at
L=12, which is why `qr-fixed` does not pay below n ~ 5000.

**Block width matters.** At n=14242: b=128 → 655.8 ms, b=512 → 507.9 ms (best),
b=2048 → 577.4 ms. `least_squares` picks 128 below n~10000 and 512 above.

### Accuracy

Gram residual `||Rtil^T Rtil - (R^T R + alpha I)||/||.||`:
`qr` 6.32e-16, `qr-fixed` 6.30e-16, `qr-struct` 9.04e-16. `qr-fixed` matches
dense `qr` and beats the struct-style scheme, because the arithmetic it skips
cannot round. At `alpha = 0` the result is exact: `Rtil == triu(R)`.

At solve level, L=20, both converging on `gtol` from the same start:
13 iterations each, 20 alpha calls each, **cost agreeing to 2.3e-08 and `|x|` to
2.4e-13**, identical peak memory.

At L=12 against `svd` as a yardstick, all three methods reached identical costs
to every printed digit across 3 seeds, with `|qr-fixed - qr|/qr` = 5.4e-09 and
`|svd - qr|/qr` = 4.9e-09 — i.e. **`qr-fixed` is no further from `qr` than a
method DESC already ships**.

### Memory

Peak in a real solve is set by the **Jacobian**, never by the alpha loop
(`peak_set_by=jac` or the perturb step; `alpha_raised_peak=False` in every run).
`qr` and `qr-fixed` had **identical** peak to 0.01 GB at both L=16 and L=20.

---

## 3. Traps — read this section

Each of these produced confident, wrong numbers before being caught.

### 3.1 The perturbation (worst offender)

`branch_experiments/solve_bench.py` and four sibling files use

```python
eq.R_lmn += pert * randn(size) * abs(eq.R_lmn).max()
```

White noise scaled to the R00 major-radius coefficient with **no mode-number
weighting**. At L=12 it exceeds the mode's own magnitude for 98.9% of modes
(median ratio 5.0e+03), and a nominal "1%" is really `||delta||/||v|| = 0.32`.
Solves from such a start reach cost ~1e24 and end with **folded flux surfaces**
while reporting `Optimization terminated successfully`.

Replacing `eq.surface` and re-solving is **also** wrong, and more insidious: the
interior is left untouched so `LinearConstraintProjection` must jump it onto the
new boundary discontinuously (start cost 9.4e24), *and* `eq.compute()` afterwards
reports the state as healthy because it reads the interior coefficients.

**Correct:** `eq.perturb` on `Rb_lmn`/`Zb_lmn` — see `PERTURBATION.md`. Same 1%
converges on `gtol` in 13 iterations to a valid equilibrium. A 1–5% sweep showed
**no tipping point**; continuation is not needed in that range.

All five `branch_experiments` files now carry loud banners; `solve.py` keeps the
broken modes only as `pert_mode="relative"/"absolute"`, which emit a warning and
stamp `INVALID_PERTURBATION` into the result.

### 3.2 Runs are not reproducible without deterministic ops

Identical configuration, same seed, gave final costs **up to 41x apart** and
iteration counts differing by 6. Cause: nondeterministic GPU reductions produce
~1e-12 differences that flip the `actual_reduction > 0` acceptance test. This
was the unresolved "section 1.6 divergence" from the original branch report; it
is now confirmed, and it afflicts a *single* method, not just method-vs-method.

`XLA_FLAGS=--xla_gpu_deterministic_ops=true` makes repeated runs **bit-identical**
(`deterministic=True` in `solve.py`).

**But never time under that flag.** It cost the unhinted scatter 7.2x
(512 → 3695 ms). Determinism for trajectory questions, normal mode for timings —
the two cannot be mixed.

Even with determinism, a single seed compares two arbitrary draws of a chaotic
trajectory. Use an ensemble over seeds.

### 3.3 `lcp.dim_x` is not the size you want

`LinearConstraintProjection.dim_x` is the **full** objective dimension.
`lsqtr` works in `_dim_x_reduced`. `branch_experiments/solve_bench.py` got this
wrong, so the `n` column in `solve_bench_a100.json` is inflated.

| case | L | m | **n reduced** | n full |
|---|---|---|---|---|
| precise_QA / W7-X | 12 | 16562 | 3434 | 4074 |
| precise_QA / W7-X | 16 | 37570 | 7602 | 8710 |
| precise_QA / W7-X | 20 | 71442 | 14242 | 15946 |
| precise_QA / W7-X | 25 | 136552 | 26896 | 29524 |
| HELIOTRON | 12 | 16562 | 5009 | 5661 |
| HELIOTRON | 20 | 71442 | 21007 | — |
| HELIOTRON | 25 | 136552 | 38830 | 43000 |

### 3.4 Compile time contaminates the first alpha call

The first non-trivial alpha call carries JIT compilation — 10–16 s at L=16,
against ~1 s of real work. It inflated the alpha share from 22% to 31% before
being separated out. `solve.py` now reports `alpha_compile_s` apart from
`alpha_time_excl_compile_s`, and `alpha_median_ms` over the compile-free body is
the reproducible metric (it matched to <1% across independent runs while wall
clock swung 30%).

Calls under 10 ms are the `p_newton`-inside-trust-region branch and do no
factorization; they are counted separately. **Note this makes the alpha metrics
meaningless for `svd`**, whose entire alpha loop is sub-10 ms — its cost is the
`jnp.linalg.svd(J_a)` outside the loop.

### 3.5 Wall clock is not the metric

Startup varies 113–302 s run to run (JIT). Runs terminate at different iteration
counts, so `qr-fixed` can do *more* iterations in *less* wall time and the naive
ratio understates it. Use `alpha_median_ms` and the per-iteration cadence
(`iter_median_s`, computed from gaps between successive `jac` starts).

### 3.6 Modal container fan-out

`.map()` starts one container per input unless capped, and these functions use
`single_use_containers=True` (needed: peak memory is a process-lifetime
high-water mark with no reset API). An 11-point sweep once opened 11 concurrent
A100s against a workspace limit of 10. `common.py` now sets
`MAX_GPU_CONTAINERS = 4`. Check with `modal app list`, stop with
`modal app stop -y <id>`.

### 3.7 `desc.set_device` must precede any jax import

`desc/backend.py` calls `set_device("cpu")` at import time when no device has
been chosen, setting `JAX_PLATFORMS=cpu` and `CUDA_VISIBLE_DEVICES=""` *before*
it imports jax. A whole pilot sweep ran on CPU while reporting plausible
numbers. `_bench_core.init_gpu()` handles this and **raises** if the backend is
not `gpu`.

---

## 4. `jac_chunk_size`

Peak memory is **not linear** in chunk — it is a hockey stick. Measured at
precise_QA L=20 on A100-80GB:

```
chunk  125  250  500  750 -> 35.05 GB   (identical)
chunk 1000 -> 36.11   1500 -> 41.33   2000 -> 48.59   3000 -> 62.56

peak = max(35.05, 20.1 + 0.01415*chunk),  knee at chunk ~1057
```

**Below the knee, chunking buys nothing and costs real time** — `chunk=500` gave
the same memory as 1000 but made the Jacobian 65% slower.

### `auto` and its clamp

`objective_funs.py:521`:

```python
estimated_memory_usage = 2.4e-7 * dim_f * dim_x + 1              # GB
max_chunk_size = round((avail_mem / estimated - 0.22) / 0.85 * dim_x)
jac_chunk_size = max([1, max_chunk_size])
```

`2.4e-7` GB/element = 240 bytes ~ 30x fp64, the assumed fwd-mode AD cost. The
`0.22`/`0.85` split asserts 22% of the full-width cost is irreducible baseline.

The **slope is well calibrated** (`0.85*2.4e-7*m` = 0.0146 GB/col predicted vs
0.01415 measured). The **intercept is 1.72x too large** (60.4 GB predicted vs
35.05 measured at L=20).

`chunk < 1` exactly when `estimated > 4.55 * avail_mem`, at which point the model
has concluded the problem **cannot fit at any chunk size** — and `max([1, ...])`
converts that verdict into a value. At L=25 it returns **1**, i.e. 29524
sequential chunks: a hang disguised as a configuration.

What auto picks:

| case | estimate | auto | of dim_x | measured jac peak |
|---|---|---|---|---|
| precise_QA L16 | 79.5 GB | 8052 | 92% | 62.7 GB |
| precise_QA L20 | 274.4 GB | 1342 | 8% | 39.5 GB |
| precise_QA L25 | 968.6 GB | **1** | — | OOM |
| HELIOTRON L25 | 1410.2 GB | **1** | — | — |

Note L=16 uses **more** memory than L=20, because auto only starts chunking once
its estimate exceeds the card.

**Always pass `jac_chunk_size` explicitly in benchmarks.** Leaving it on `auto`
makes the alpha-loop share (and hence the end-to-end speedup) depend on a
heuristic rather than a stated parameter.

### Is L=25 reachable?

**No, not on 80 GB.** There is a genuine chunk-independent floor proportional to
`m * n_reduced` — the Jacobian output, a transpose copy, and `feasible_tangents`,
measured at **34.5 bytes per element (4.3x fp64)**. Scaling gives 126.5 GB at
precise_QA L25 against 75 GB usable, and it OOM'd at chunk 100, 250 and 500.
So auto's *verdict* is right at L=25; only its reporting is wrong.

`_jac` computes `df = jvp(feasible_tangents.T, x)` then `return df.T`. **If that
transpose were avoided the floor would drop by one Jacobian** — a concrete
optimisation worth exploring, though the measured floor is ~4.3x fp64 per
element rather than the ~2x that account alone predicts, so something else is
also resident.

---

## 5. Merged upstream PRs

* **#2293 "Misc"** — LinearConstraintProjection build memory, `lax.scan` rewrite
  of `qr_multiply`'s blocked Householder application, rank-deficiency warning for
  bounded sub-objectives, `initial_alpha` 0 → 1e-6. One conflict in the `lsqtr`
  `tr_method` docstring, resolved keeping both sides.
* **#2286** — `jax.linearize` when chunking forward-mode Jacobians. Clean merge.

Measured effect at n=14242: **runtime unchanged** (`qr` 1157.8 → 1161.9 ms, inside
an 8.3% spread), but **compile time 29.48 s → 2.19 s**, a 13x improvement, which
matches what #2293 advertised. The speedup ratio is unchanged at ~2.29x.

Also ported #2293's improved Householder application into `_apply_QT_wy`
(reflectors as rows, no `(M, k)` transpose formed). Worth ~1%, i.e. inside noise
— XLA was evidently already fusing that transpose. Kept because it removes a
divergence from upstream.

**Two things about that port to know:**
1. It is a *port*, not an import. `desc.backend._householder_multiply` exists
   only inside the `jax < 0.10` fallback branch; on jax >= 0.10 DESC takes
   `qr_multiply` from `jax.scipy.linalg` and the helper is not defined.
2. The upstream version has **no `tau == 0` guard**, and we need one: at
   `alpha = 0` the `sqrt(alpha)*I` block is entirely zero, the panel is already
   triangular, and every tau comes back zero. Without the guard the diagonal
   correction is `1/0`.

### Version caveat worth carrying forward

`requirements.txt` pins `jax < 0.10`, so the pure-JAX `qr_multiply` fallback is
the only path any supported install takes — it **is** our baseline, and the
comparison is valid today. But that block is marked for deletion once DESC
raises its jax floor. On that day the baseline becomes
`jax.scipy.linalg.qr_multiply` (cuSOLVER `ormqr`) and **every speedup ratio here
needs re-measuring**. `desc/backend.py` claims the pure-JAX route beats `ormqr`
on large tall systems, which is exactly our `[R; sqrt(alpha) I]` shape — but that
is the authors' assertion, unmeasured by us, and it is the largest remaining
uncertainty in the speedup figure.

---

## 6. Suggested next steps

The obvious gap is breadth: everything above is `precise_QA`, mostly one seed.

1. **End-to-end reps.** The L=20 end-to-end number is ~1.15x from single runs
   per arm, and container-to-container `jac` variation (0.4 s of a 2.4 s saving)
   is comparable to the effect. 3+ reps per arm would pin it down.
2. **Other equilibria.** `HELIOTRON` (n=5009 at L=12, 21007 at L=20) and `W7-X`
   have different `m/n` ratios — 3.3 vs 4.8 — which changes the alpha share and
   therefore the end-to-end gain.
3. **Resolution scan** L=12/16/20 with the chunk fixed, to map where `qr-fixed`
   crosses from loss to gain. It is a loss at L=12 and a gain at L=20; the
   crossover is somewhere around n ~ 5000–7000 and has not been located.
4. **Chunk sweep per resolution.** The knee moves with `m`; the L=20 knee at
   ~1057 does not transfer.
5. **The `df.T` transpose in `_jac`** (section 4) is the highest-leverage memory
   change for reaching larger problems, and is independent of this work.

### Running things

```bash
# kernel: one alpha factorization
DESC_BENCH_GPU=A100-80GB modal run -m modal_bench.kernel \
    --sizes 14242 --methods qr,qr-fixed --reps 5

# end-to-end solve
DESC_BENCH_GPU=A100-80GB modal run -m modal_bench.solve \
    --name precise_QA --resolutions 20 --methods qr,qr-fixed \
    --maxiter 20 --reps 3 --seeds 0,1,2 --jac-chunk 1000 \
    --no-counted --deterministic --pert 0.03 --pert-mode perturb --gtol 1e-8

# local laptop GPU (8 GB), L=12 fits in 2.3 GB
python modal_bench/perturb_sweep_local.py 0.03 20 250 gpu 12 2
```

`GPU` defaults to `A100-40GB` (`common.py`); override with `DESC_BENCH_GPU`.
**40 GB is not enough at L=20 when `eq.perturb` runs** — the perturb step needs
~23 GiB on top and the peak is 42.07 GB. Use 80 GB there.
