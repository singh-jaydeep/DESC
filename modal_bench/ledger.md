

## Kernel sweep: qr vs qr-struct vs qr-slim (A100-80GB)
*2026-08-21 16:29:56Z*

- **sizes**: [3434, 14242]
- **blocks**: [128, 256, 512, 1024, 2048]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **note**: one container per n; one subprocess per measurement
- **git**: 4482846d0
- **desc_dirty**: True
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
  3434  qr          None        64.0       0.59    4.46e-16   21.2%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-struct    128        48.4       0.58    8.06e-16   17.6%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-slim      128        45.0       0.59    4.50e-16   17.4%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-struct    256        41.5       0.59    8.06e-16   21.8%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-slim      256        47.6       0.88    4.50e-16    9.9%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-struct    512        44.1       0.60    8.06e-16   21.7%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-slim      512        50.0       0.64    4.48e-16    5.8%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-struct   1024        55.2       0.62    8.06e-16   20.2%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-slim     1024        47.2       0.63    4.49e-16   21.3%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-struct   2048        64.2       0.89    8.06e-16   21.6%  NVIDIA A100-SXM4-80GB, 81920 MiB
  3434  qr-slim     2048        54.8       0.62    4.48e-16   13.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr          None      1159.8       7.68    6.32e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    128       686.6       7.95    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      128       488.9      42.30    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    256       570.7       7.97    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      256       461.8      16.44    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    512       533.8       8.00    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      512       447.9       8.97    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   1024       558.2       8.06    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     1024       477.6      11.76    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   2048       659.7       8.20    9.04e-16    0.3%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     2048       551.4       8.87    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```

### Findings from this batch

- **Speed reproduces the docstring.** At n=14242 (spread 0.1%): qr-slim 2.59x over
  dense qr, 1.19x over qr-struct. Interpolating the docstring's own table between
  n=12000 and 16000 predicts ~1100/~530/~450 ms; measured 1159.8/533.8/447.9.
- **n=3434 is noise-dominated** (spread 5-22%). The apparent struct-beats-slim
  ordering there is not resolvable at reps=5. Needs more reps before quoting.
- **The memory claim does NOT reproduce, and inverts.** Docstring: slim peaks
  11.8 GB vs struct 16.6 GB at n=12000 ("29% reduction"). Measured at n=14242:
  struct is FLAT in block width at 7.95-8.20 GB; slim ranges 8.87-42.30 GB and at
  b=128 peaks at 5.5x the dense route it replaces. qr-slim is the most
  memory-hungry route here, not the least.
  Likely mechanism: `block` is static, so the panel loop fully unrolls -- 112
  panels at b=128, n=14242 -- each with its own concatenate temporary for the
  growing frontier.
- **Consequence: the default block rule is a landmine for slim.** least_squares.py
  picks b=128 below n~10000 and 512 above. n=14242 gets 512 (8.97 GB, fine); every
  case below n=10000 gets 128, the setting that blew up here. n=7602 and n=11166
  are untested and sit in that zone.
- **Accuracy: slim is the best of the three.** Gram residual matches dense qr
  (4.50e-16 vs 4.46e-16; 6.30e-16 vs 6.32e-16) and beats struct (8.06e-16,
  9.04e-16) at both sizes.


## Memory-plan diagnosis: why qr-slim's peak scales with panel count
*2026-08-21 16:41:47Z*

- **sizes**: [3434, 7602, 14242]
- **blocks**: [128, 256, 512, 1024, 2048]
- **note**: compile-only; XLA memory_analysis on ShapeDtypeStruct, no allocation
- **git**: 4482846d0
- **desc_dirty**: True
```
     n  method       block  panels     temp_GB    total_GB   temp/panel_MB
--------------------------------------------------------------------------
  3434  qr          None    None        0.18        0.36           nan
  3434  qr-struct    128      27        0.20        0.38           7.6
  3434  qr-slim      128      27        0.21        0.39           8.0
  3434  qr-struct    256      14        0.20        0.38          15.0
  3434  qr-slim      256      14        0.40        0.58          29.6
  3434  qr-struct    512       7        0.21        0.39          31.1
  3434  qr-slim      512       7        0.25        0.43          36.9
  3434  qr-struct   1024       4        0.24        0.41          60.4
  3434  qr-slim     1024       4        0.24        0.42          61.9
  3434  qr-struct   2048       2        0.42        0.59         213.3
  3434  qr-slim     2048       2        0.23        0.40         115.3
  7602  qr          None    None        0.87        1.73           nan
  7602  qr-struct    128      60        0.98        1.84          16.7
  7602  qr-slim      128      60        4.09        4.95          69.8
  7602  qr-struct    256      30        0.98        1.85          33.6
  7602  qr-slim      256      30        1.01        1.87          34.4
  7602  qr-struct    512      15        1.00        1.86          68.2
  7602  qr-slim      512      15        2.13        2.99         145.2
  7602  qr-struct   1024       8        1.03        1.90         132.4
  7602  qr-slim     1024       8        1.35        2.21         173.0
  7602  qr-struct   2048       4        1.19        2.05         303.9
  7602  qr-slim     2048       4        1.07        1.93         274.7
 14242  qr          None    None        3.03        6.06           nan
 14242  qr-struct    128     112        3.41        6.44          31.2
 14242  qr-slim      128     112       37.76       40.78         345.2
 14242  qr-struct    256      56        3.43        6.45          62.7
 14242  qr-slim      256      56       11.90       14.92         217.6
 14242  qr-struct    512      28        3.46        6.48         126.4
 14242  qr-slim      512      28        4.43        7.45         161.9
 14242  qr-struct   1024      14        3.52        6.54         257.2
 14242  qr-slim     1024      14        7.21       10.24         527.6
 14242  qr-struct   2048       7        3.64        6.66         532.8
 14242  qr-slim     2048       7        4.32        7.34         631.2
```

### Diagnosis: qr-slim's memory blowup is a scaling defect, not a tuning issue

Compile-only XLA `memory_analysis()` reproduces the measured runtime peak, so the
blowup is PLANNED liveness in the unrolled graph, not allocator fragmentation:

    n=14242   planned total -> measured peak
    b=128       40.78 GB       42.30 GB
    b=256       14.92 GB       16.44 GB
    b=512        7.45 GB        8.97 GB
    b=1024      10.24 GB       11.76 GB
    b=2048       7.34 GB        8.87 GB

Scaling in n at fixed b=128 (n = 3434 -> 7602 -> 14242):

    qr-struct   0.20 -> 0.98 ->  3.41 GB   ~ O(n^2), and FLAT in b (3.41-3.64 GB)
    qr-slim     0.21 -> 4.09 -> 37.76 GB   ~ O(n^3 / b)

Mechanism: `block` is static so the panel loop fully unrolls.
`structured_retriangularize` threads ONE fixed 2n x (n+1) accumulator, which XLA
updates in place and whose per-panel temporaries it frees immediately.
`structured_retriangularize_slim` carries the frontier "at its exact live shape",
so each of the ceil(n/b) panels allocates a distinct Sub/trail/F that the
scheduler keeps alive. The change the docstring advertises as the memory saving
(change 2) is what costs the memory.

Projected at each route's best block on an A100-80GB:

    n=21007  HELIOTRON L20    qr ~13 GB   struct ~15 GB   slim ~17 GB
    n=26896  precise_QA L25   qr ~22 GB   struct ~23 GB   slim ~44 GB
    n=38830  HELIOTRON L25    qr ~45 GB   struct ~26 GB   slim ~140 GB -> OOM

Recommendation: the two changes in qr-slim are separable. Change 1 (apply Q^T to
trailing columns only) carries most of the speed AND the accuracy improvement
(gram 6.30e-16 vs struct's 9.04e-16, matching dense qr's 6.32e-16) and is
memory-neutral. Change 2 is the defect. Build "struct + change 1": keep the
fixed buffer, drop the exact-live-shape frontier.


## Memory-plan diagnosis: why qr-slim's peak scales with panel count
*2026-08-21 16:52:34Z*

- **sizes**: [3434, 7602, 11166, 14242, 21007, 26896, 38830]
- **blocks**: [128, 256, 512, 1024, 2048]
- **note**: compile-only; XLA memory_analysis on ShapeDtypeStruct, no allocation
- **git**: 4482846d0
- **desc_dirty**: True
```
     n  method       block  panels     temp_GB    total_GB   temp/panel_MB
--------------------------------------------------------------------------

## Session end state (2026-08-21)

**Killed mid-run:** the compile-only memory-plan sweep over all 7 sizes including
qr-fixed (app ap-odCEZaKOALjBjTBhM0pwea). No results from it were recorded. The
earlier 3-size plan run (3434/7602/14242, qr/qr-struct/qr-slim) DID complete and
is in this ledger.

**Done and trustworthy:**
- Kernel timings at n=3434 and n=14242 for qr / qr-struct / qr-slim, 5 block
  widths. n=14242 is solid (spread 0.1%); n=3434 is noise-dominated (5-22%).
- Memory-plan diagnosis: qr-slim is O(n^3/block), qr-struct is O(n^2). Compile
  plan matches runtime peak within 5%.
- `structured_retriangularize_fixed` + `trust_region_step_exact_qr_fixed`
  implemented in desc/optimize/tr_subproblems.py; `tr_method="qr-fixed"`
  registered in least_squares.py. Verified on CPU only: 4 shapes x 5 alphas x
  4 block widths, zero failures, Gram residual identical to qr-slim's.
- Bug fix in least_squares.py:461: `("qr", "qr_struct", "qr_slim")` ->
  hyphenated, so `del R` actually fires for the structured methods.

**Not yet measured (the open work):**
1. qr-fixed on GPU: does it keep qr-struct's flat O(n^2) memory AND most of
   qr-slim's 1.19x speed edge? This is the whole point of the change and is
   completely unverified on hardware. Rerun: `modal run -m modal_bench.memdiag`
   then `modal run -m modal_bench.kernel --sizes 14242`.
2. Kernel sweep at n = 5009, 7602, 11166, 21007, 26896, 38830. The OOM
   projections in the diagnosis section are EXTRAPOLATIONS from a 3-point fit,
   not measurements. Do not quote them as measured.
3. n=3434 with reps>=25 to resolve the small-n noise.
4. End-to-end solves (modal_bench/solve.py, written but NEVER RUN). This is the
   question that decides whether any kernel win matters: the branch's own prior
   A100 data had qr-struct at 0.95-1.01x end-to-end despite a 1.4x kernel win.
5. The section 1.6 trajectory divergence. solve.py has the per-pass accept/reject
   instrumentation for it but has not been run.

**Cost control:** MAX_GPU_CONTAINERS=4 in common.py caps every mapped function.
The workspace GPU limit is 10 and was exceeded once this session by an
uncapped 11-way .map(). Check `modal app list` and `modal app stop -y <id>`.


## Kernel sweep: qr vs qr-struct vs qr-slim vs qr-fixed (A100-80GB)
*2026-08-24 18:55:59Z*

- **sizes**: [14242]
- **blocks**: [128, 256, 512, 1024, 2048]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **note**: one container per n; one subprocess per measurement
- **git**: 0a32c9711
- **desc_dirty**: False
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
 14242  qr          None      1159.5       7.68    6.32e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    128       686.7       7.95    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      128       488.6      42.30    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     128       682.2      10.59    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    256       570.9       7.97    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      256       462.4      16.44    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     256       560.8       7.97    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    512       534.2       8.00    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      512       448.5       8.97    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     512       512.1       8.00    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   1024       559.6       8.06    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     1024       478.4      11.76    6.30e-16    0.2%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    1024       519.2       8.01    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   2048       661.3       8.20    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     2048       552.5       8.87    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    2048       579.1       8.10    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```

### qr-fixed first result (n=14242 = precise_QA/W7-X L=M=N=20), all four in one container

Same card as the earlier n=14242 run (A100-SXM4-80GB); qr/qr-struct/qr-slim
reproduced to <=0.15% (qr 1159.8->1159.5 ms, struct b128 686.6->686.7,
slim b512 447.9->448.5; peaks bit-identical), so this is directly comparable.

    method     best b  time_ms  vs qr   peak_GB  peak range over b   gram
    qr           None   1159.5  1.00x     7.68   7.68                6.32e-16
    qr-struct     512    534.2  2.17x     8.00   7.95 - 8.20         9.04e-16
    qr-fixed      512    512.1  2.26x     8.00   7.97 - 10.59        6.30e-16
    qr-slim       512    448.5  2.59x     8.97   8.87 - 42.30        6.30e-16

MEMORY: qr-fixed removes the pressure. Flat in block width like qr-struct
(7.97/8.00/8.01/8.10 for b=256..2048) against qr-slim's 8.87-42.30. At b=128 --
the width least_squares picks by default for n<10000 -- qr-slim is 42.30 GB and
qr-fixed is 10.59 GB. A mild residual bump remains at b=128 (10.59 vs struct's
7.95, 1.33x) but it is bounded, not the 5.3x blowup. Confirms change 2 was the
sole cause.

ACCURACY: qr-fixed == qr-slim exactly (6.30e-16), both better than qr-struct
(9.04e-16) and matching dense qr (6.32e-16). Change 1 carries all of it.

TIMING -- CORRECTS AN EARLIER CLAIM IN THIS LEDGER. The diagnosis section said
change 1 "carries most of the speed". It does not. Splitting the 85.7 ms
struct->slim gap:

    change 1 only (struct -> fixed):  22.1 ms = 26% of the gap
    change 2 adds  (fixed -> slim) :  63.6 ms = 74% of the gap

So change 2 causes the memory defect AND carries three quarters of the speed
advantage. qr-fixed is not a free win over qr-slim: it costs 12% of kernel time
(512.1 vs 448.5 ms) to buy O(n^2) memory instead of O(n^3/block).

WHAT IS SETTLED: qr-fixed strictly dominates qr-struct -- 4% faster, identical
peak memory, better residual. qr-struct can be dropped from contention.

WHAT IS NOT: qr-fixed vs qr-slim is a genuine trade, not a dominance. At this n
with the default block the memory difference is only 8.00 vs 8.97 GB; the case
against qr-slim rests on the O(n^3/block) SCALING, which is still extrapolated
and unmeasured above n=14242.


## Kernel sweep: qr / qr-struct / qr-slim / qr-fixed / qr-slice (A100-80GB)
*2026-08-24 19:23:51Z*

- **sizes**: [14242]
- **blocks**: [128, 256, 512, 1024, 2048]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **note**: one container per n; one subprocess per measurement
- **git**: 0a32c9711
- **desc_dirty**: True
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
 14242  qr          None      1159.3       7.68    6.32e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    128       689.4       7.95    9.04e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      128       488.8      42.30    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     128       681.4      10.59    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slice     128       878.1      10.66    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    256       571.1       7.97    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      256       462.1      16.44    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     256       560.8       7.97    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slice     256       619.5      13.10    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct    512       534.0       8.00    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim      512       448.0       8.97    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     512       512.3       8.00    6.30e-16    0.2%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slice     512       488.9      15.33    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   1024       558.9       8.06    9.04e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     1024       477.9      11.76    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    1024       518.9       8.01    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slice    1024       489.4      14.50    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-struct   2048       660.2       8.20    9.04e-16    0.2%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slim     2048       552.2       8.87    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    2048       579.1       8.10    6.30e-16    0.3%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-slice    2048       554.5      11.73    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```

### qr-slice: the gather/scatter hypothesis, tested (n=14242, all five in one container)

Proposal was: `idx = concat([arange(c0,c1), n+arange(0,c1)])` is two CONTIGUOUS
ranges, so replacing the general gather/scatter with static slices should recover
qr-slim's speed while keeping the fixed buffer's flat memory.

Result: HALF RIGHT ON SPEED, WRONG ON MEMORY. qr-slice is not adopted.

    method     best b  time_ms  vs qr   peak_GB   peak range over b
    qr           None   1159.3  1.00x      7.68   7.68
    qr-struct     512    534.0  2.17x      8.00   7.95 -  8.20
    qr-fixed      512    512.3  2.26x      8.00   7.97 - 10.59
    qr-slice      512    488.9  2.37x     15.33  10.66 - 15.33
    qr-slim       512    448.0  2.59x      8.97   8.87 - 42.30

Speed: removing the gather/scatter IS worth real time -- 23.4 ms, 27% of the
struct->slim gap. Attribution at each route's best block (gap 85.9 ms):

    change 1            (struct -> fixed)  21.7 ms   25%
    drop gather/scatter (fixed  -> slice)  23.4 ms   27%
    still unexplained   (slice  -> slim )  40.8 ms   48%

Memory: qr-slice LOSES the flat profile that was the whole point. 15.33 GB at
b=512 against qr-fixed's 8.00 GB, and never below 10.66 GB at any width. Likely
cause: three separate static `.at[].set()` chains on M per panel instead of one
scatter, plus an explicit `concatenate` materializing Sub; 15.33 GB is ~5 copies
of the 3.02 GiB M buffer against qr-fixed's ~2.6.

At b=128 qr-slice is also much SLOWER than qr-fixed (878.1 vs 681.4 ms) -- 112
panels x 3 update ops does not pay.

VERDICT: qr-slice is dominated by qr-slim on BOTH axes at this n (slower AND more
memory). Drop it. Also drop qr-struct, which qr-fixed dominates. Note the ~48%
of the gap that neither change explains: qr-slim never writes back to a large
buffer at all -- it carries F forward as a value and reads the top block straight
from R (n x n) rather than from M (2n x (n+1)) -- so it simply touches less
memory per panel. That is intrinsic to the moving frontier, not an addressing
detail, and is not recoverable within a fixed-buffer design.

REMAINING CONTEST: qr-fixed (2.26x, flat O(n^2) memory) vs qr-slim (2.59x,
O(n^3/block)). Decided by whether the scaling actually bites at n >= 21007, which
is still unmeasured.


## Memory plan across all sizes: qr / qr-struct / qr-fixed / qr-slim
*2026-08-24 19:39:20Z*

- **sizes**: [3434, 7602, 11166, 14242, 21007, 26896, 38830]
- **blocks**: [128, 256, 512, 1024, 2048]
- **note**: compile-only; XLA memory_analysis on ShapeDtypeStruct, no allocation
- **git**: 0a32c9711
- **desc_dirty**: True
```
     n  method       block  panels     temp_GB    total_GB   temp/panel_MB
--------------------------------------------------------------------------
  3434  qr          None    None        0.18        0.36           nan
  3434  qr-struct    128      27        0.20        0.38           7.6
  3434  qr-slim      128      27        0.21        0.39           8.0
  3434  qr-fixed     128      27        0.35        0.53          13.4
  3434  qr-struct    256      14        0.20        0.38          15.0
  3434  qr-slim      256      14        0.40        0.58          29.6
  3434  qr-fixed     256      14        0.35        0.53          25.9
  3434  qr-struct    512       7        0.21        0.39          31.1
  3434  qr-slim      512       7        0.25        0.43          36.9
  3434  qr-fixed     512       7        0.21        0.38          30.2
  3434  qr-struct   1024       4        0.24        0.41          60.4
  3434  qr-slim     1024       4        0.24        0.42          61.9
  3434  qr-fixed    1024       4        0.25        0.42          63.0
  3434  qr-struct   2048       2        0.42        0.59         213.3
  3434  qr-slim     2048       2        0.23        0.40         115.3
  3434  qr-fixed    2048       2        0.30        0.48         154.5
  7602  qr          None    None        0.87        1.73           nan
  7602  qr-struct    128      60        0.98        1.84          16.7
  7602  qr-slim      128      60        4.09        4.95          69.8
  7602  qr-fixed     128      60        1.73        2.59          29.5
  7602  qr-struct    256      30        0.98        1.85          33.6
  7602  qr-slim      256      30        1.01        1.87          34.4
  7602  qr-fixed     256      30        0.98        1.84          33.3
  7602  qr-struct    512      15        1.00        1.86          68.2
  7602  qr-slim      512      15        2.13        2.99         145.2
  7602  qr-fixed     512      15        0.99        1.85          67.3
  7602  qr-struct   1024       8        1.03        1.90         132.4
  7602  qr-slim     1024       8        1.35        2.21         173.0
  7602  qr-fixed    1024       8        1.01        1.87         128.9
  7602  qr-struct   2048       4        1.19        2.05         303.9
  7602  qr-slim     2048       4        1.07        1.93         274.7
  7602  qr-fixed    2048       4        1.16        2.02         295.9
 11166  qr          None    None        1.87        3.73           nan
 11166  qr-struct    128      88        2.10        3.96          24.4
 11166  qr-slim      128      88       16.66       18.52         193.8
 11166  qr-fixed     128      88        3.72        5.58          43.3
 11166  qr-struct    256      44        2.11        3.97          49.2
 11166  qr-slim      256      44        4.28        6.14          99.5
 11166  qr-fixed     256      44        2.11        3.97          49.2
 11166  qr-struct    512      22        2.13        3.99          99.4
 11166  qr-slim      512      22        4.08        5.93         189.7
 11166  qr-fixed     512      22        2.11        3.97          98.4
 11166  qr-struct   1024      11        2.18        4.04         203.1
 11166  qr-slim     1024      11        3.58        5.44         333.4
 11166  qr-fixed    1024      11        2.14        4.00         199.5
 11166  qr-struct   2048       6        2.29        4.15         391.3
 11166  qr-slim     2048       6        2.48        4.34         423.0
 11166  qr-fixed    2048       6        2.26        4.12         385.9
 14242  qr          None    None        3.03        6.06           nan
 14242  qr-struct    128     112        3.41        6.44          31.2
 14242  qr-slim      128     112       37.76       40.78         345.2
 14242  qr-fixed     128     112        6.05        9.08          55.3
 14242  qr-struct    256      56        3.43        6.45          62.7
 14242  qr-slim      256      56       11.90       14.92         217.6
 14242  qr-fixed     256      56        3.43        6.45          62.7
 14242  qr-struct    512      28        3.46        6.48         126.4
 14242  qr-slim      512      28        4.43        7.45         161.9
 14242  qr-fixed     512      28        3.46        6.48         126.4
 14242  qr-struct   1024      14        3.52        6.54         257.2
 14242  qr-slim     1024      14        7.21       10.24         527.6
 14242  qr-fixed    1024      14        3.46        6.49         253.3
 14242  qr-struct   2048       7        3.64        6.66         532.8
 14242  qr-slim     2048       7        4.32        7.34         631.2
 14242  qr-fixed    2048       7        3.55        6.57         519.1
 21007  qr          None    None        6.59       13.16           nan
 21007  qr-struct    512      42        7.48       14.06         182.4
 21007  qr-slim      512      42       12.05       18.63         293.9
 21007  qr-fixed     512      42       13.15       19.73         320.7
 21007  qr-struct   1024      21        7.57       14.14         368.9
 21007  qr-slim     1024      21       20.73       27.31        1011.0
 21007  qr-fixed    1024      21        7.49       14.06         365.1
 21007  qr-struct   2048      11        7.75       14.33         721.4
 21007  qr-slim     2048      11       12.01       18.59        1118.2
 21007  qr-fixed    2048      11        7.59       14.17         706.9
 26896  qr          None    None       10.79       21.57           nan
 26896  qr-struct    512      53       12.23       23.01         236.3
 26896  qr-slim      512      53       37.27       48.05         720.0
 26896  qr-fixed     512      53       12.18       22.96         235.4
 26896  qr-struct   1024      27       12.34       23.12         468.0
 26896  qr-slim     1024      27       20.88       31.66         792.0
 26896  qr-fixed    1024      27       12.24       23.02         464.2
 26896  qr-struct   2048      14       12.56       23.34         918.9
 26896  qr-slim     2048      14       23.60       34.38        1725.9
 26896  qr-fixed    2048      14       12.37       23.15         904.9
 38830  qr          None    None       22.49       44.95           nan
 38830  qr-struct    512      76       25.43       47.89         342.6
 38830  qr-slim      512      76      165.00      187.47        2223.2
 38830  qr-fixed     512      76       25.35       47.82         341.6
 38830  qr-struct   1024      38       25.58       48.05         689.3
 38830  qr-slim     1024      38       34.84       57.30         938.7
 38830  qr-fixed    1024      38       25.43       47.90         685.4
 38830  qr-struct   2048      19       25.89       48.36        1395.5
 38830  qr-slim     2048      19       66.88       89.35        3604.5
 38830  qr-fixed    2048      19       25.61       48.08        1380.4
```

### DECISION DATA: memory plan across all 8 sizes (compile-only, A100-80GB)

Planned total GB (temp+args+output) at the block least_squares actually picks
(128 below n~10000, else 512). Device limit at mem_fraction 0.95 is ~75 GB.

       n  case                  b     qr  struct  fixed    slim
    3434  precise_QA/W7-X L12  128    0.4     0.4    0.5     0.4
    7602  precise_QA/W7-X L16  128    1.7     1.8    2.6     5.0
   11166  HELIOTRON L16        512    3.7     4.0    4.0     5.9
   14242  precise_QA/W7-X L20  512    6.1     6.5    6.5     7.4
   21007  HELIOTRON L20        512   13.2    14.1   19.7    18.6
   26896  precise_QA/W7-X L25  512   21.6    23.0   23.0    48.0
   38830  HELIOTRON L25        512   45.0    47.9   47.8   187.5  <-- OOM

1. qr-fixed holds O(n^2). Best-block totals track dense qr within 6% at EVERY
   size: 0.4/1.8/4.0/6.5/14.1/23.0/47.8 against qr's 0.4/1.7/3.7/6.1/13.2/21.6/
   45.0. The O(n^3/block) defect is gone. One bump remains (n=21007, b=512:
   19.7 vs struct's 14.1, +40%) but it is bounded, not a blowup.

2. qr-slim is NON-MONOTONIC in block width, which is worse than merely large.
   There is no rule that picks a safe width:
     n=14242: b128=40.8 b256=14.9 b512=7.4  b1024=10.2 b2048=7.3
     n=38830: b512=187.5           b1024=57.3            b2048=89.3
   The current heuristic picks 512 above n~10000 -- the WORST of the three at
   n=38830, and 128 below, which is the worst at n=7602 (5.0 vs qr's 1.7).
   Even tuned to its best width qr-slim needs 57.3 GB at n=38830 vs 47.8.

3. CAVEAT -- these are alpha-loop plans only; a real solve adds a live Jacobian.
   lsqtr keeps J_h and J_a alive through the alpha loop (`del J_h, J_a` happens
   only after step acceptance), so add AT LEAST one J:
     n=26896  J=27.4 GB -> fixed ~50 GB (fits)  slim ~75 GB (at the limit)
     n=38830  J=39.5 GB -> fixed ~87 GB         slim ~227 GB
   So HELIOTRON L25 is out of reach on an A100-80GB for EVERY method, alpha loop
   or not, and the size where the choice actually decides feasibility is
   precise_QA/W7-X L25 (n=26896).

VERDICT: adopt qr-fixed, drop qr-slim. qr-slim OOMs at the default block at the
largest size, and its memory cannot be tuned safely because it is non-monotonic
in the one parameter available. Cost of the switch is 13% of kernel time
(2.26x vs 2.59x over dense qr at n=14242). Surviving contenders for the timing
sweep: qr (baseline) and qr-fixed.


## End-to-end solve: precise_QA L=M=N=16, qr,qr-fixed (A100-80GB)
*2026-08-24 20:09:24Z*

- **maxiter**: 20
- **block**: default
- **methods**: qr,qr-fixed
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 0a32c9711
- **desc_dirty**: True

```
precise_QA L16 qr [timing ]  wall=165.6s  it=17  cost=2.326416776e+09  conv=True
    alpha    51.7s ( 31.2%)  calls= 35   1477.9 ms/call
    jac      29.6s ( 17.9%)  calls= 18
    fun       3.6s (  2.2%)  calls= 36
    other    80.7s
    peak=54.36 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
```

```
precise_QA L16 qr [counted]  wall=164.3s  it=19  cost=1.298810967e+08  conv=True
    alpha    57.1s ( 34.7%)  calls= 42   1359.0 ms/call
    jac      29.8s ( 18.1%)  calls= 20
    fun       3.6s (  2.2%)  calls= 43
    other    73.8s
    peak=54.36 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    alpha inner iters=190 total, 4.52 per call
```

```
precise_QA L16 qr-fixed [timing ]  wall=109.7s  it=19  cost=7.600405937e+08  conv=True
    alpha    30.6s ( 27.9%)  calls= 38    805.4 ms/call
    jac      24.7s ( 22.5%)  calls= 20
    fun       2.4s (  2.2%)  calls= 39
    other    52.0s
    peak=54.36 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
```

```
precise_QA L16 qr-fixed [counted]  wall=160.1s  it=20  cost=2.648771665e+07  conv=False
    alpha    38.1s ( 23.8%)  calls= 40    952.1 ms/call
    jac      35.2s ( 22.0%)  calls= 21
    fun       4.4s (  2.7%)  calls= 41
    other    82.4s
    peak=54.36 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    alpha inner iters=170 total, 4.25 per call
```


## jac_chunk_size: what auto picks, and what actually fits
*2026-08-24 20:24:13Z*

- **note**: one jac evaluation per point; the L=16 solve showed jac sets the peak and the alpha loop never raises it
- **git**: 0a32c9711
- **desc_dirty**: True
```
case          L  chunk_in  chunk_used   dim_f  n_red   est_GB  jac_peak_GB  jac_s
---------------------------------------------------------------------------------
precise_QA   16      auto        8052   37570   7602     79.5        62.72    23.5
precise_QA   20      auto        1342   71442  14242    274.4        39.53    33.7
precise_QA   20      2000        2000   71442  14242    274.4        50.14    34.9
precise_QA   20      1000        1000   71442  14242    274.4        36.11    28.1
precise_QA   20       500         500   71442  14242    274.4        35.05    33.7
```

### jac_chunk_size: measured, and where the heuristic breaks

One jac evaluation per point on A100-80GB (limit 75 GB), single measurement each
so the times carry ordinary run-to-run noise.

    case          L  chunk_in  chunk_used   dim_f  n_red   est_GB  jac_peak_GB  jac_s
    precise_QA   16      auto        8052   37570   7602     79.5        62.72   23.5
    precise_QA   20      auto        1342   71442  14242    274.4        39.53   33.7
    precise_QA   20      2000        2000   71442  14242    274.4        50.14   34.9
    precise_QA   20      1000        1000   71442  14242    274.4        36.11   28.1
    precise_QA   20       500         500   71442  14242    274.4        35.05   33.7

1. L=20 ALREADY FITS on auto: 39.5 GB of 75. No change was needed to make it run.
2. L=20 uses LESS memory than L=16 (39.5 vs 62.7 GB). The heuristic only starts
   chunking once its estimate exceeds the card, so L=16 sits just under the
   threshold and runs essentially unchunked (8052 of 8710 columns, 92%), while
   L=20 chunks hard (1342 of 15946, 8%). L=16 is the worst case, not L=20.
3. The heuristic's estimate is not accurate enough to size a run by: 79.5 GB
   predicted vs 62.7 GB measured at L=16; 274.4 vs 39.5 at L=20.
4. Reproducing the formula exactly (predicted 8052/1342 vs measured 8052/1342)
   lets it be evaluated ahead of time. At L=25 it DEGENERATES:

       precise_QA L16  est   79.5 GB -> chunk  8052
       precise_QA L20  est  274.4 GB -> chunk  1342
       precise_QA L25  est  968.6 GB -> chunk     1   <-- clamped
       HELIOTRON  L25  est 1410.2 GB -> chunk     1   <-- clamped

   (avail_mem/est - 0.22) goes negative, `max([1, max_chunk_size])` clamps to 1,
   and the Jacobian would be built ONE COLUMN AT A TIME -- 29524 sequential
   chunks. Any L=25 run must pass an explicit jac_chunk_size.

FOR THE SWEEP: pass jac_chunk_size EXPLICITLY, never "auto". Otherwise a method
that perturbs the trajectory could resolve a different chunk and a different
peak, and the arms would differ in chunking rather than in QR variant. At L=20,
chunk=1000 gives 36.1 GB, leaving ~39 GB of headroom against qr-fixed's ~6.5 GB
alpha-loop working set. Timing differences among 28-35 s across chunk sizes are
single measurements and should not be read as a chunk-size optimum.


## End-to-end solve: precise_QA L=M=N=16,20, qr,qr-fixed (A100-80GB)
*2026-08-24 20:40:05Z*

- **maxiter**: 6
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 1
- **jac_chunk**: 1000
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 0a32c9711
- **desc_dirty**: True

```
precise_QA L16 qr [rep0]  wall=131.5s  it=6  cost=1.820340e+17  opt=7.69e+06
    stopped: Maximum number of iterations has been exceeded.
    alpha median=  983.1 ms [p25 885, p75 1241]  calls= 15 (trivial 0)
    alpha    14.1s excl-compile ( 10.7% of wall); compile call 15.4s
    jac      27.4s ( 20.8%)  calls=  7
    fun       3.6s (  2.7%)  calls= 16
    other    71.1s
    peak=13.11 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=113.0s  per-iter median=2.8s [p25 2.8, p75 5.1]  jac median=0.9s (compile 21.9s)
    => projected maxiter=N wall ~ 113 + 2.8*N s
```

```
precise_QA L16 qr-fixed [rep0]  wall=138.9s  it=6  cost=1.819769e+17  opt=7.65e+06
    stopped: Maximum number of iterations has been exceeded.
    alpha median=  584.8 ms [p25 520, p75 746]  calls= 15 (trivial 0)
    alpha     8.4s excl-compile (  6.0% of wall); compile call 18.7s
    jac      35.0s ( 25.2%)  calls=  7
    fun       3.4s (  2.5%)  calls= 16
    other    73.5s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=125.4s  per-iter median=2.5s [p25 2.3, p75 3.6]  jac median=0.9s (compile 29.7s)
    => projected maxiter=N wall ~ 125 + 2.5*N s
```

```
precise_QA L20 qr [rep0]  wall=225.8s  it=6  cost=6.224533e+18  opt=8.76e+08
    stopped: Maximum number of iterations has been exceeded.
    alpha median= 4139.0 ms [p25 3416, p75 5117]  calls= 18 (trivial 0)
    alpha    77.0s excl-compile ( 34.1% of wall); compile call 20.1s
    jac      40.3s ( 17.9%)  calls=  7
    fun       2.3s (  1.0%)  calls= 19
    other    86.1s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=115.3s  per-iter median=22.6s [p25 20.0, p75 24.5]  jac median=3.9s (compile 16.9s)
    => projected maxiter=N wall ~ 115 + 22.6*N s
```

```
precise_QA L20 qr-fixed [rep0]  wall=226.8s  it=6  cost=6.270516e+18  opt=8.84e+08
    stopped: Maximum number of iterations has been exceeded.
    alpha median= 2000.6 ms [p25 1638, p75 2455]  calls= 18 (trivial 0)
    alpha    37.0s excl-compile ( 16.3% of wall); compile call 9.2s
    jac      46.6s ( 20.6%)  calls=  7
    fun       3.7s (  1.7%)  calls= 19
    other   130.2s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=159.3s  per-iter median=14.0s [p25 12.8, p75 15.2]  jac median=3.6s (compile 24.9s)
    => projected maxiter=N wall ~ 159 + 14.0*N s
```

### Calibration: fixed jac_chunk=1000, maxiter=6, precise_QA (A100-80GB)

      L  method     a_med_ms  per_iter_s  startup_s  peak_GB          cost       opt   a_calls/iter
     16  qr              983         2.8      113.0    13.11  1.820340e+17  7.69e+06   2.14
     16  qr-fixed        585         2.5      125.4    13.13  1.819769e+17  7.65e+06   2.14
     20  qr             4139        22.6      115.3    33.83  6.224533e+18  8.76e+08   2.57
     20  qr-fixed       2001        14.0      159.3    33.83  6.270516e+18  8.84e+08   2.57

1. FIXING THE CHUNK CUT L=16's PEAK BY 4.8x: 62.7 GB (auto, ~unchunked) ->
   13.11 GB (chunk=1000). L=20 33.83 GB. Peak identical across arms to 0.02 GB,
   set_by=jac in every run, alpha never raised the high-water mark.

2. ALPHA SPEEDUP IN-SOLVE MATCHES THE KERNEL SWEEP.
     L=16 (n=7602):  983 -> 585 ms  = 1.68x  (1.70x measured at chunk=auto)
     L=20 (n=14242): 4139 -> 2001 ms = 2.07x (kernel sweep standalone: 2.26x)
   The small shortfall at L=20 is dilution: each alpha ITERATION is one
   factorization plus two triangular solves and norms, which are common to both.

3. AT SMALL maxiter THE TRAJECTORIES AGREE. Identical iteration counts, identical
   alpha-call counts, cost rel-diff 3.1e-04 (L=16) and 7.4e-03 (L=20). Contrast
   the maxiter=20 runs, where two passes of the SAME method landed 18x apart.
   Divergence is cumulative, so question (b) -- speed at fixed small maxiter --
   is well posed, while (a) is where trajectory noise lives.

4. PER-ITERATION SPEEDUP: 1.16x (L=16), 1.61x (L=20). CAUTION: these medians come
   from only ~4 inter-jac gaps at maxiter=6 (p25/p75 e.g. 2.8 [2.8, 5.1]), so
   they are indicative, not settled. alpha_median_ms is the reproducible metric.

5. STARTUP IS 113-159 s AND VARIES BY ~40% RUN TO RUN (JIT). At maxiter=6 it is
   most of the wall, which is why raw wall times look equal (225.8 vs 226.8 s at
   L=20) despite a 1.61x per-iteration difference. Any comparison at small
   maxiter must use cadence or alpha medians, never wall clock.

Projected wall per run = startup + per_iter*N:
      L=16  N=50   4.3 / 4.1 min      N=200  11.4 / 10.3 min   (qr / qr-fixed)
      L=20  N=50  20.7 / 14.3 min     N=200  77.2 / 49.4 min


## End-to-end solve: precise_QA L=M=N=16,20, qr,qr-fixed (A100-80GB)
*2026-08-24 20:50:30Z*

- **maxiter**: 40
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 3
- **jac_chunk**: 1000
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 0a32c9711
- **desc_dirty**: True

```
precise_QA L16 qr [rep0]  wall=185.6s  it=18  cost=3.081314e+09  opt=1.32e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 1095.0 ms [p25 662, p75 1321]  calls= 35 (trivial 0)
    alpha    38.9s excl-compile ( 20.9% of wall); compile call 17.6s
    jac      42.0s ( 22.6%)  calls= 19
    fun       4.0s (  2.2%)  calls= 36
    other    83.2s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=124.5s  per-iter median=3.6s [p25 2.8, p75 3.6]  jac median=0.9s (compile 26.0s)
    => projected maxiter=N wall ~ 124 + 3.6*N s
```

```
precise_QA L16 qr [rep1]  wall=177.7s  it=17  cost=8.726425e+09  opt=2.96e+01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  986.2 ms [p25 664, p75 1326]  calls= 35 (trivial 0)
    alpha    39.0s excl-compile ( 22.0% of wall); compile call 16.2s
    jac      39.5s ( 22.2%)  calls= 18
    fun       4.1s (  2.3%)  calls= 36
    other    78.9s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=117.0s  per-iter median=3.7s [p25 2.8, p75 3.9]  jac median=0.9s (compile 24.4s)
    => projected maxiter=N wall ~ 117 + 3.7*N s
```

```
precise_QA L16 qr [rep2]  wall=183.0s  it=19  cost=6.992149e+09  opt=3.45e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  883.1 ms [p25 663, p75 1324]  calls= 38 (trivial 0)
    alpha    42.2s excl-compile ( 23.1% of wall); compile call 16.8s
    jac      41.8s ( 22.9%)  calls= 20
    fun       4.0s (  2.2%)  calls= 39
    other    78.1s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=117.6s  per-iter median=3.6s [p25 2.8, p75 4.3]  jac median=0.9s (compile 25.0s)
    => projected maxiter=N wall ~ 118 + 3.6*N s
```

```
precise_QA L16 qr-fixed [rep0]  wall=144.8s  it=17  cost=3.405362e+09  opt=1.36e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  519.8 ms [p25 392, p75 650]  calls= 40 (trivial 0)
    alpha    23.1s excl-compile ( 16.0% of wall); compile call 16.2s
    jac      35.7s ( 24.7%)  calls= 18
    fun       4.5s (  3.1%)  calls= 41
    other    65.3s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=99.4s  per-iter median=3.0s [p25 2.4, p75 3.3]  jac median=0.9s (compile 20.7s)
    => projected maxiter=N wall ~ 99 + 3.0*N s
```

```
precise_QA L16 qr-fixed [rep1]  wall=145.9s  it=21  cost=1.334856e+09  opt=1.08e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  521.7 ms [p25 392, p75 781]  calls= 40 (trivial 0)
    alpha    25.3s excl-compile ( 17.3% of wall); compile call 13.4s
    jac      37.8s ( 25.9%)  calls= 22
    fun       4.3s (  3.0%)  calls= 41
    other    65.0s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=92.2s  per-iter median=2.7s [p25 2.3, p75 2.8]  jac median=0.9s (compile 19.3s)
    => projected maxiter=N wall ~ 92 + 2.7*N s
```

```
precise_QA L16 qr-fixed [rep2]  wall=140.9s  it=19  cost=1.626558e+10  opt=2.48e+01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  520.8 ms [p25 488, p75 842]  calls= 38 (trivial 1)
    alpha    24.4s excl-compile ( 17.4% of wall); compile call 13.1s
    jac      36.5s ( 25.9%)  calls= 20
    fun       3.1s (  2.2%)  calls= 39
    other    63.8s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=91.3s  per-iter median=2.8s [p25 2.3, p75 3.4]  jac median=0.9s (compile 19.6s)
    => projected maxiter=N wall ~ 91 + 2.8*N s
```

```
precise_QA L20 qr [rep0]  wall=449.0s  it=21  cost=6.764479e+08  opt=3.13e+01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 3409.0 ms [p25 3405, p75 6078]  calls= 38 (trivial 1)
    alpha   164.7s excl-compile ( 36.7% of wall); compile call 28.3s
    jac     103.7s ( 23.1%)  calls= 22
    fun       3.5s (  0.8%)  calls= 39
    other   148.7s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=151.9s  per-iter median=14.9s [p25 13.7, p75 17.0]  jac median=3.9s (compile 21.7s)
    => projected maxiter=N wall ~ 152 + 14.9*N s
```

```
precise_QA L20 qr [rep1]  wall=384.5s  it=15  cost=3.837399e+09  opt=2.42e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 3411.6 ms [p25 2560, p75 4263]  calls= 38 (trivial 0)
    alpha   145.8s excl-compile ( 37.9% of wall); compile call 26.3s
    jac      79.2s ( 20.6%)  calls= 16
    fun       4.4s (  1.2%)  calls= 39
    other   128.8s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=145.7s  per-iter median=14.9s [p25 11.9, p75 22.0]  jac median=3.9s (compile 20.7s)
    => projected maxiter=N wall ~ 146 + 14.9*N s
```

```
precise_QA L20 qr [rep2]  wall=383.1s  it=15  cost=2.799772e+10  opt=8.87e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 3423.2 ms [p25 2571, p75 4287]  calls= 35 (trivial 0)
    alpha   140.1s excl-compile ( 36.6% of wall); compile call 27.5s
    jac      79.7s ( 20.8%)  calls= 16
    fun       3.1s (  0.8%)  calls= 36
    other   132.6s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=150.8s  per-iter median=15.0s [p25 10.6, p75 21.2]  jac median=3.9s (compile 21.2s)
    => projected maxiter=N wall ~ 151 + 15.0*N s
```

```
precise_QA L20 qr-fixed [rep0]  wall=382.1s  it=25  cost=7.854898e+08  opt=1.70e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 1642.9 ms [p25 1642, p75 2053]  calls= 49 (trivial 0)
    alpha    93.9s excl-compile ( 24.6% of wall); compile call 7.7s
    jac     118.9s ( 31.1%)  calls= 26
    fun       3.2s (  0.8%)  calls= 50
    other   158.4s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=137.6s  per-iter median=10.2s [p25 8.6, p75 11.2]  jac median=3.9s (compile 21.4s)
    => projected maxiter=N wall ~ 138 + 10.2*N s
```

```
precise_QA L20 qr-fixed [rep1]  wall=349.7s  it=23  cost=2.219488e+08  opt=2.85e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 1642.2 ms [p25 1232, p75 1913]  calls= 44 (trivial 1)
    alpha    78.4s excl-compile ( 22.4% of wall); compile call 7.1s
    jac     110.3s ( 31.5%)  calls= 24
    fun       3.3s (  0.9%)  calls= 45
    other   150.7s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=125.8s  per-iter median=9.4s [p25 7.9, p75 11.0]  jac median=3.9s (compile 20.6s)
    => projected maxiter=N wall ~ 126 + 9.4*N s
```

```
precise_QA L20 qr-fixed [rep2]  wall=345.1s  it=21  cost=1.496743e+09  opt=4.51e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 1641.9 ms [p25 1640, p75 2933]  calls= 43 (trivial 0)
    alpha    88.4s excl-compile ( 25.6% of wall); compile call 7.1s
    jac     102.9s ( 29.8%)  calls= 22
    fun       3.2s (  0.9%)  calls= 44
    other   143.4s
    peak=33.83 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=132.4s  per-iter median=10.2s [p25 8.9, p75 11.7]  jac median=3.9s (compile 21.0s)
    => projected maxiter=N wall ~ 132 + 10.2*N s
```

### (b) SPEED AT FIXED maxiter=40, jac_chunk=1000, 3 reps, precise_QA (A100-80GB)

12/12 runs succeeded.

      L  method    alpha_med (spread)   per_iter_s (spread)   wall_s (spread)
     16  qr           986 ms (21.5%)      3.64 s ( 0.4%)       183 s ( 4.3%)
     16  qr-fixed     521 ms ( 0.4%)      2.77 s (12.9%)       145 s ( 3.4%)
     20  qr          3412 ms ( 0.4%)     14.95 s ( 0.3%)       384 s (17.2%)
     20  qr-fixed    1642 ms ( 0.1%)     10.16 s ( 7.1%)       350 s (10.6%)

    RATIO qr / qr-fixed:   L=16  alpha 1.89x   per-iter 1.32x
                           L=20  alpha 2.08x   per-iter 1.47x

ANSWER TO (b): qr-fixed is 1.32x faster per outer iteration at L=16 and 1.47x at
L=20. The alpha loop itself is 1.89x / 2.08x faster, matching the standalone
kernel sweep (2.26x at n=14242, diluted in-solve by the two triangular solves and
norms each alpha iteration also performs).

Amdahl closes: alpha is ~50% of a qr iteration at both sizes.
    L=16: 1.90 calls/iter x 0.986 s = 1.87 s of 3.64 s = 51.4%
          predicted per-iter 1.32x, measured 1.32x
    L=20: 2.19 calls/iter x 3.412 s = 7.46 s of 14.95 s = 49.9%
          predicted per-iter 1.35x, measured 1.47x (qr-fixed also took fewer
          alpha calls per iteration here, 1.88 vs 2.19, so it gains twice)

DO NOT USE WALL CLOCK. Runs terminated at different iteration counts (L=20: qr at
15/15/21, qr-fixed at 21/23/25), so wall ratios (1.26x, 1.10x) UNDERSTATE the
difference -- qr-fixed did more iterations in less time. Startup is 126-152 s
with ~20% variance. Cadence and alpha medians are the metrics.

MEMORY: peak 13.13 GB (L=16) and 33.83 GB (L=20), IDENTICAL across all 12 runs to
0.01 GB, always set_by=jac, alpha_raised_peak=False everywhere. With the chunk
fixed, the QR variant has no measurable effect on peak memory at these sizes.

CAVEAT: the L=16 qr alpha median spread is 21.5% (1095/986/883 ms over three
reps) while every other cell is <=0.4%. The alpha median is over calls whose
inner-iteration counts vary with trajectory, so it is not a pure hardware
measure. The L=16 alpha ratio is 1.9x +/- 0.2; the L=20 ratio 2.08x is solid.

### Implication for (a): within-method noise swamps the method effect

Final cost across three reps of the SAME method, same seed, same config:

      L=16  qr         2.8x span   (3.08e+09 .. 8.73e+09)   iters 17,18,19
      L=16  qr-fixed  12.2x span   (1.33e+09 .. 1.63e+10)   iters 17,19,21
      L=20  qr        41.4x span   (6.76e+08 .. 2.80e+10)   iters 15,15,21
      L=20  qr-fixed   6.7x span   (2.22e+08 .. 1.50e+09)   iters 21,23,25

Runs are NOT reproducible: identical configuration gives final costs up to 41x
apart and iteration counts differing by 6. No run reached maxiter=40; all stalled
on xtol earlier. Comparing "final equilibria" between methods is therefore not
meaningful at n=3 reps -- the between-method difference is far inside the
within-method spread. This is the section 1.6 divergence, now measured directly
and shown to afflict a SINGLE method, not just method-vs-method.


## End-to-end solve: precise_QA L=M=N=16, qr,qr-fixed (A100-80GB)
*2026-08-24 21:13:34Z*

- **maxiter**: 40
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 3
- **jac_chunk**: 1000
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 0a32c9711
- **desc_dirty**: True

```
precise_QA L16 qr [rep0]  wall=236.3s  it=26  cost=1.001303e+08  opt=1.79e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  881.2 ms [p25 442, p75 882]  calls= 45 (trivial 2)
    alpha    34.7s excl-compile ( 14.7% of wall); compile call 14.9s
    jac      93.5s ( 39.6%)  calls= 27
    fun      10.9s (  4.6%)  calls= 46
    other    82.3s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=114.5s  per-iter median=4.4s [p25 3.8, p75 5.3]  jac median=2.6s (compile 25.6s)
    => projected maxiter=N wall ~ 114 + 4.4*N s
```

```
precise_QA L16 qr [rep1]  wall=238.7s  it=26  cost=1.001303e+08  opt=1.79e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  881.7 ms [p25 442, p75 885]  calls= 45 (trivial 2)
    alpha    34.7s excl-compile ( 14.5% of wall); compile call 13.8s
    jac      95.5s ( 40.0%)  calls= 27
    fun      11.5s (  4.8%)  calls= 46
    other    83.3s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=113.5s  per-iter median=4.5s [p25 3.9, p75 5.4]  jac median=2.7s (compile 25.5s)
    => projected maxiter=N wall ~ 113 + 4.5*N s
```

```
precise_QA L16 qr [rep2]  wall=237.3s  it=26  cost=1.001303e+08  opt=1.79e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  882.2 ms [p25 442, p75 883]  calls= 45 (trivial 2)
    alpha    34.7s excl-compile ( 14.6% of wall); compile call 14.5s
    jac      95.3s ( 40.2%)  calls= 27
    fun      10.9s (  4.6%)  calls= 46
    other    81.9s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=113.8s  per-iter median=4.5s [p25 3.9, p75 5.4]  jac median=2.7s (compile 25.9s)
    => projected maxiter=N wall ~ 114 + 4.5*N s
```

```
precise_QA L16 qr-fixed [rep0]  wall=582.6s  it=25  cost=1.017455e+08  opt=3.95e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 9297.6 ms [p25 4869, p75 9640]  calls= 45 (trivial 2)
    alpha   375.5s excl-compile ( 64.5% of wall); compile call 21.5s
    jac      92.2s ( 15.8%)  calls= 26
    fun      11.1s (  1.9%)  calls= 46
    other    82.2s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=118.5s  per-iter median=15.6s [p25 8.3, p75 25.0]  jac median=2.7s (compile 25.5s)
    => projected maxiter=N wall ~ 119 + 15.6*N s
```

```
precise_QA L16 qr-fixed [rep1]  wall=603.4s  it=25  cost=1.017455e+08  opt=3.95e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median= 9724.2 ms [p25 4996, p75 10287]  calls= 45 (trivial 2)
    alpha   388.2s excl-compile ( 64.3% of wall); compile call 22.0s
    jac      95.0s ( 15.8%)  calls= 26
    fun      11.4s (  1.9%)  calls= 46
    other    86.8s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=123.8s  per-iter median=15.9s [p25 8.6, p75 25.4]  jac median=2.8s (compile 26.6s)
    => projected maxiter=N wall ~ 124 + 15.9*N s
```

```
precise_QA L16 qr-fixed [rep2]  wall=604.0s  it=25  cost=1.017455e+08  opt=3.95e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=10246.0 ms [p25 5269, p75 11686]  calls= 45 (trivial 2)
    alpha   412.2s excl-compile ( 68.2% of wall); compile call 23.5s
    jac      90.7s ( 15.0%)  calls= 26
    fun      10.4s (  1.7%)  calls= 46
    other    67.3s
    peak=13.13 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=8710 (dim_f=37570)
    cadence: startup=99.2s  per-iter median=16.9s [p25 8.5, p75 26.6]  jac median=2.7s (compile 23.0s)
    => projected maxiter=N wall ~ 99 + 16.9*N s
```

### Determinism check: XLA_FLAGS=--xla_gpu_deterministic_ops=true (L=16, maxiter=40)

IT WORKS -- and it costs qr-fixed dearly.

REPRODUCIBILITY: 6/6 runs. Three reps of each method are BIT-IDENTICAL in final
cost, |x|, iteration count and alpha-call count:

    qr        26 iters, 45 alpha calls, cost 100130322.9761754,  |x| 3.364109276563
    qr-fixed  25 iters, 45 alpha calls, cost 101745508.95145378, |x| 3.364283573529

against the nondeterministic baseline at the same config, where three identical
reps spanned 2.8x (qr) and 12.2x (qr-fixed) in final cost with iteration counts
17/18/19 and 17/19/21. So the section 1.6 divergence IS nondeterministic GPU
reduction noise amplified by the `actual_reduction > 0` acceptance branch --
confirmed, not merely plausible.

(a) ANSWERED, at L=16 maxiter=40: the two methods land at genuinely different but
close points. Both stall on xtol.

    cost rel-diff = 1.61e-02        (1.0013e8 vs 1.0175e8)
    |x|  rel-diff = 5.18e-05
    within-method spread = 0 exactly

qr-fixed reaches a lower optimality (0.395 vs 1.789) in fewer iterations (25 vs
26); qr reaches a marginally lower cost. Neither dominates. The 1.6% cost gap is
a real method difference, not noise -- which only became a meaningful statement
once the within-method spread was driven to zero.

DO NOT TIME UNDER THIS FLAG. Deterministic ops cripple qr-fixed:

    alpha median   qr        986 ->    882 ms   ( 0.89x)
    alpha median   qr-fixed  521 ->   9724 ms   (18.67x)
    per-iteration  qr       3.64 ->   4.46 s
    per-iteration  qr-fixed 2.77 ->  15.92 s

Under determinism qr-fixed is 3.6x SLOWER than qr, exactly reversing the
nondeterministic result. Cause is almost certainly the scatter
`M.at[idx, c1:].set(trail)` with a traced index array: a deterministic scatter
must serialise or sort, while dense qr's cuSOLVER geqrf is already deterministic
and is unaffected (0.89x). This is the same gather/scatter that the qr-slice
experiment showed costs 27% of the struct->slim gap.

USE: determinism for trajectory questions only; normal mode for all timings.


## Kernel sweep: qr / qr-struct / qr-slim / qr-fixed / qr-slice (A100-80GB)
*2026-08-24 22:19:47Z*

- **sizes**: [14242]
- **blocks**: [512]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **deterministic**: False
- **methods**: qr,qr-fixed,qr-hinted
- **note**: one container per n; one subprocess per measurement
- **git**: 0a32c9711
- **desc_dirty**: True
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
 14242  qr          None      1157.8       7.68    6.32e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     512       512.3       8.00    6.30e-16    0.8%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-hinted    512       506.2       8.00    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```


## Kernel sweep: qr / qr-struct / qr-slim / qr-fixed / qr-slice (A100-80GB)
*2026-08-24 22:21:45Z*

- **sizes**: [14242]
- **blocks**: [512]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **deterministic**: True
- **methods**: qr,qr-fixed,qr-hinted
- **note**: one container per n; one subprocess per measurement
- **git**: 0a32c9711
- **desc_dirty**: True
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
 14242  qr          None      1158.4       7.68    6.32e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     512      3695.1       8.00    6.30e-16    2.4%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-hinted    512       506.8       8.00    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```

### qr-hinted: the scatter hints fix determinism at zero cost (n=14242, b=512)

    mode            qr        qr-fixed    qr-hinted
    normal        1157.8 ms    512.3 ms     506.2 ms
    deterministic 1158.4 ms   3695.1 ms     506.8 ms
    penalty          1.00x       7.21x        1.001x

qr-hinted == qr-fixed plus three assertions on the scatter/gather indices:
`indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"`. All are
true by construction: idx = [c0,c1) U [n,n+c1) with c1 <= n, so disjoint, sorted
ascending, unique, in bounds. Verified BITWISE IDENTICAL to qr-fixed on CPU over
3 shapes x 4 alphas x 3 block widths (max |diff| exactly 0 in Rtil and Qtz).

CONFIRMS THE DIAGNOSIS, and separates two independent properties of one op:

  * `unique_indices=True` -> no combiner -> no atomics -> nothing for the
    deterministic path to undo. Penalty 7.21x -> 1.001x. Peak memory UNCHANGED
    at 8.00 GB, because uniqueness says nothing about opacity.
  * Opacity of `scatter` -> scheduling barrier -> still what keeps peak at
    8.00 GB against qr-slice's 15.33 GB. Not a bug; a tradeoff.

CORRECTION: the deterministic penalty is 7.2x at kernel level, not the 18.7x
reported from the solve. That figure came from alpha MEDIANS over calls whose
inner-iteration counts vary with trajectory, so it overstated the
per-factorization cost. 7.2x is the clean measurement.

qr-hinted strictly dominates qr-fixed: 1.2% faster in normal mode, 7.2x faster
under deterministic ops, identical memory, identical numerics. The hints should
simply be folded into qr-fixed.


## End-to-end solve: precise_QA L=M=N=12, qr,qr-fixed (A100-80GB)
*2026-08-24 22:36:31Z*

- **maxiter**: 200
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 2
- **jac_chunk**: 1000
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 qr [rep0]  wall=193.3s  it=147  cost=2.281118e+06  opt=1.12e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   88.6 ms [p25 45, p75 91]  calls=189 (trivial 0)
    alpha    19.0s excl-compile (  9.8% of wall); compile call 5.5s
    jac      94.5s ( 48.9%)  calls=148
    fun      17.5s (  9.0%)  calls=190
    other    56.8s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=63.7s  per-iter median=0.8s [p25 0.7, p75 0.9]  jac median=0.5s (compile 19.9s)
    => projected maxiter=N wall ~ 64 + 0.8*N s
```

```
precise_QA L12 qr [rep1]  wall=186.4s  it=147  cost=2.281118e+06  opt=1.12e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   88.6 ms [p25 45, p75 89]  calls=189 (trivial 0)
    alpha    19.0s excl-compile ( 10.2% of wall); compile call 5.4s
    jac      91.6s ( 49.1%)  calls=148
    fun      16.6s (  8.9%)  calls=190
    other    53.9s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=60.4s  per-iter median=0.8s [p25 0.7, p75 0.9]  jac median=0.5s (compile 17.8s)
    => projected maxiter=N wall ~ 60 + 0.8*N s
```

```
precise_QA L12 qr-fixed [rep0]  wall=90.9s  it=34  cost=1.634202e+06  opt=7.66e-03
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.1 ms [p25 85, p75 141]  calls= 54 (trivial 0)
    alpha     6.5s excl-compile (  7.1% of wall); compile call 6.0s
    jac      35.4s ( 39.0%)  calls= 35
    fun       6.7s (  7.4%)  calls= 55
    other    36.3s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=61.1s  per-iter median=0.9s [p25 0.8, p75 1.0]  jac median=0.5s (compile 18.2s)
    => projected maxiter=N wall ~ 61 + 0.9*N s
```

```
precise_QA L12 qr-fixed [rep1]  wall=95.0s  it=34  cost=1.634202e+06  opt=7.66e-03
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.3 ms [p25 85, p75 142]  calls= 54 (trivial 0)
    alpha     6.5s excl-compile (  6.9% of wall); compile call 6.0s
    jac      39.0s ( 41.0%)  calls= 35
    fun       6.3s (  6.6%)  calls= 55
    other    37.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=67.5s  per-iter median=0.9s [p25 0.7, p75 1.0]  jac median=0.5s (compile 23.2s)
    => projected maxiter=N wall ~ 68 + 0.9*N s
```


## End-to-end solve: precise_QA L=M=N=12, qr,qr-fixed (A100-80GB)
*2026-08-24 22:40:08Z*

- **maxiter**: 200
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 3
- **jac_chunk**: 1000
- **deterministic**: False
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 qr [rep0]  wall=74.4s  it=30  cost=8.828751e+05  opt=1.14e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  176.1 ms [p25 132, p75 220]  calls= 52 (trivial 0)
    alpha     9.7s excl-compile ( 13.1% of wall); compile call 5.7s
    jac      21.7s ( 29.2%)  calls= 31
    fun       2.8s (  3.8%)  calls= 53
    other    34.4s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=59.6s  per-iter median=0.6s [p25 0.4, p75 0.7]  jac median=0.2s (compile 17.2s)
    => projected maxiter=N wall ~ 60 + 0.6*N s
```

```
precise_QA L12 qr [rep1]  wall=79.6s  it=21  cost=1.859566e+07  opt=1.41e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  176.6 ms [p25 134, p75 255]  calls= 46 (trivial 0)
    alpha     9.4s excl-compile ( 11.8% of wall); compile call 5.7s
    jac      23.8s ( 29.9%)  calls= 22
    fun       3.0s (  3.8%)  calls= 47
    other    37.6s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=62.6s  per-iter median=0.7s [p25 0.5, p75 1.0]  jac median=0.2s (compile 20.4s)
    => projected maxiter=N wall ~ 63 + 0.7*N s
```

```
precise_QA L12 qr [rep2]  wall=137.0s  it=200  cost=3.478307e+06  opt=4.02e-02
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   88.7 ms [p25 45, p75 89]  calls=252 (trivial 0)
    alpha    24.8s excl-compile ( 18.1% of wall); compile call 5.5s
    jac      47.0s ( 34.3%)  calls=201
    fun       4.0s (  2.9%)  calls=253
    other    55.5s
    peak=4.65 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=59.0s  per-iter median=0.4s [p25 0.3, p75 0.4]  jac median=0.2s (compile 16.8s)
    => projected maxiter=N wall ~ 59 + 0.4*N s
```

```
precise_QA L12 qr-fixed [rep0]  wall=87.6s  it=23  cost=8.446056e+06  opt=4.98e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.5 ms [p25 86, p75 148]  calls= 45 (trivial 0)
    alpha     5.7s excl-compile (  6.5% of wall); compile call 8.1s
    jac      25.6s ( 29.2%)  calls= 24
    fun       3.6s (  4.2%)  calls= 46
    other    44.5s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=80.1s  per-iter median=0.5s [p25 0.4, p75 0.7]  jac median=0.2s (compile 22.1s)
    => projected maxiter=N wall ~ 80 + 0.5*N s
```

```
precise_QA L12 qr-fixed [rep1]  wall=90.1s  it=27  cost=2.004772e+06  opt=4.56e-03
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  114.0 ms [p25 86, p75 142]  calls= 48 (trivial 0)
    alpha     5.9s excl-compile (  6.6% of wall); compile call 9.0s
    jac      26.0s ( 28.8%)  calls= 28
    fun       3.8s (  4.2%)  calls= 49
    other    45.4s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=80.6s  per-iter median=0.5s [p25 0.4, p75 0.6]  jac median=0.2s (compile 21.9s)
    => projected maxiter=N wall ~ 81 + 0.5*N s
```

```
precise_QA L12 qr-fixed [rep2]  wall=91.8s  it=72  cost=3.268379e+06  opt=3.18e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   57.3 ms [p25 57, p75 113]  calls=103 (trivial 0)
    alpha     9.5s excl-compile ( 10.3% of wall); compile call 6.2s
    jac      28.6s ( 31.2%)  calls= 73
    fun       3.3s (  3.6%)  calls=104
    other    44.1s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=65.2s  per-iter median=0.3s [p25 0.3, p75 0.4]  jac median=0.2s (compile 17.7s)
    => projected maxiter=N wall ~ 65 + 0.3*N s
```

### L=12 (n=3434) end-to-end, jac_chunk=1000, maxiter=200

DETERMINISTIC (2 reps each) -- reps bit-identical again, confirming determinism
holds at this resolution too:

    qr        it=147  cost=2.281118e+06  opt=1.12e-02  alpha_calls=189
              alpha median  88.6 ms   per-iter 0.8 s   wall 193.3 / 186.4 s
    qr-fixed  it= 34  cost=1.634202e+06  opt=7.66e-03  alpha_calls= 54
              alpha median 113.1 ms   per-iter 0.9 s   wall  90.9 /  95.0 s

Two things here contradict the pattern from L=16/L=20:

1. qr-fixed's alpha loop is SLOWER at this size: 113.1 vs 88.6 ms, and per
   iteration it is slower too (0.9 vs 0.8 s). At n=3434 the structured route has
   little to exploit -- the kernel sweep gave only 1.54x there against 2.26x at
   n=14242 -- and the default block at n<10000 is 128.
2. Despite that, qr-fixed finished in 2.1x less wall time, because it took 34
   iterations against qr's 147 and reached a BETTER point (cost 1.63e6 vs 2.28e6,
   opt 7.7e-3 vs 1.1e-2). The end-to-end difference here is a TRAJECTORY effect,
   not a speed effect.

NORMAL MODE (3 reps each) shows that difference is not reliable:

    qr        it = 30, 21, 200(cap)   cost 8.83e5, 1.86e7, 3.48e6
    qr-fixed  it = 23, 27,  72        cost 8.45e6, 2.00e6, 3.27e6

Iteration counts span 21-200 for qr and 23-72 for qr-fixed; final costs span 21x
and 4.2x. So the deterministic 147-vs-34 is ONE DRAW from a highly variable
process, not a systematic property. qr's 147 was a bad draw and qr-fixed's 34 a
good one; in normal mode qr reached 21 and 30 twice.

METHODOLOGICAL CONSEQUENCE: determinism makes each run reproducible but does NOT
make the qr-vs-qr-fixed comparison robust. The trajectory is chaotic with respect
to ~1e-14 perturbations, so a single deterministic seed compares two arbitrary
draws. Comparing the methods requires an ENSEMBLE OVER SEEDS, run deterministically
so that each sample is itself reproducible.


## End-to-end solve: precise_QA L=M=N=12, qr,qr-fixed,svd (A100-80GB)
*2026-08-24 22:44:50Z*

- **maxiter**: 200
- **block**: default
- **methods**: qr,qr-fixed,svd
- **reps**: 1
- **seeds**: 0,1,2,3,4
- **jac_chunk**: 1000
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 qr [s0r0]  wall=187.7s  it=147  cost=2.281118e+06  opt=1.12e-02
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   88.5 ms [p25 45, p75 89]  calls=189 (trivial 0)
    alpha    18.9s excl-compile ( 10.0% of wall); compile call 5.5s
    jac      91.7s ( 48.9%)  calls=148
    fun      16.5s (  8.8%)  calls=190
    other    55.0s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=62.5s  per-iter median=0.8s [p25 0.7, p75 0.9]  jac median=0.5s (compile 18.7s)
    => projected maxiter=N wall ~ 62 + 0.8*N s
```

```
precise_QA L12 qr [s1r0]  wall=174.5s  it=10  cost=1.161761e+11  opt=3.63e+04
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  178.0 ms [p25 135, p75 217]  calls= 26 (trivial 1)
    alpha     4.7s excl-compile (  2.7% of wall); compile call 11.9s
    jac      66.8s ( 38.3%)  calls= 11
    fun       8.1s (  4.7%)  calls= 27
    other    83.0s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=145.6s  per-iter median=2.1s [p25 1.8, p75 2.3]  jac median=0.9s (compile 57.8s)
    => projected maxiter=N wall ~ 146 + 2.1*N s
```

```
precise_QA L12 qr [s2r0]  wall=93.0s  it=17  cost=2.535232e+08  opt=1.10e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  177.4 ms [p25 177, p75 265]  calls= 36 (trivial 2)
    alpha     7.7s excl-compile (  8.3% of wall); compile call 6.7s
    jac      27.7s ( 29.8%)  calls= 18
    fun       5.6s (  6.1%)  calls= 37
    other    45.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=76.3s  per-iter median=1.2s [p25 1.0, p75 1.3]  jac median=0.5s (compile 19.9s)
    => projected maxiter=N wall ~ 76 + 1.2*N s
```

```
precise_QA L12 qr [s3r0]  wall=153.0s  it=103  cost=1.532046e+08  opt=1.77e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   88.4 ms [p25 45, p75 132]  calls=128 (trivial 2)
    alpha    12.9s excl-compile (  8.4% of wall); compile call 5.9s
    jac      67.5s ( 44.1%)  calls=104
    fun      12.0s (  7.8%)  calls=129
    other    54.8s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=70.2s  per-iter median=0.7s [p25 0.7, p75 0.8]  jac median=0.5s (compile 18.3s)
    => projected maxiter=N wall ~ 70 + 0.7*N s
```

```
precise_QA L12 qr [s4r0]  wall=98.0s  it=17  cost=5.349177e+07  opt=3.57e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  176.5 ms [p25 133, p75 183]  calls= 34 (trivial 1)
    alpha     5.8s excl-compile (  5.9% of wall); compile call 6.0s
    jac      28.2s ( 28.8%)  calls= 18
    fun       5.8s (  5.9%)  calls= 35
    other    52.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=74.0s  per-iter median=1.2s [p25 1.1, p75 1.4]  jac median=0.5s (compile 18.0s)
    => projected maxiter=N wall ~ 74 + 1.2*N s
```

```
precise_QA L12 qr-fixed [s0r0]  wall=101.2s  it=34  cost=1.634202e+06  opt=7.66e-03
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.6 ms [p25 85, p75 142]  calls= 54 (trivial 0)
    alpha     6.5s excl-compile (  6.5% of wall); compile call 6.2s
    jac      35.8s ( 35.4%)  calls= 35
    fun       6.6s (  6.6%)  calls= 55
    other    46.0s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=80.2s  per-iter median=0.9s [p25 0.8, p75 1.0]  jac median=0.5s (compile 19.0s)
    => projected maxiter=N wall ~ 80 + 0.9*N s
```

```
precise_QA L12 qr-fixed [s1r0]  wall=87.3s  it=10  cost=1.161761e+11  opt=3.63e+04
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.1 ms [p25 86, p75 141]  calls= 26 (trivial 0)
    alpha     2.9s excl-compile (  3.3% of wall); compile call 7.1s
    jac      27.7s ( 31.7%)  calls= 11
    fun       6.0s (  6.9%)  calls= 27
    other    43.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=75.3s  per-iter median=1.2s [p25 1.1, p75 1.4]  jac median=0.6s (compile 22.0s)
    => projected maxiter=N wall ~ 75 + 1.2*N s
```

```
precise_QA L12 qr-fixed [s2r0]  wall=99.9s  it=16  cost=4.254656e+08  opt=6.55e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  114.4 ms [p25 114, p75 163]  calls= 36 (trivial 1)
    alpha     4.8s excl-compile (  4.8% of wall); compile call 7.2s
    jac      32.4s ( 32.4%)  calls= 17
    fun       7.2s (  7.2%)  calls= 37
    other    48.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=85.5s  per-iter median=1.2s [p25 1.0, p75 1.3]  jac median=0.5s (compile 24.2s)
    => projected maxiter=N wall ~ 85 + 1.2*N s
```

```
precise_QA L12 qr-fixed [s3r0]  wall=106.4s  it=24  cost=1.062278e+08  opt=2.62e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  114.5 ms [p25 87, p75 174]  calls= 43 (trivial 2)
    alpha     5.7s excl-compile (  5.3% of wall); compile call 7.4s
    jac      34.2s ( 32.2%)  calls= 25
    fun       7.0s (  6.5%)  calls= 44
    other    52.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=85.8s  per-iter median=1.0s [p25 0.9, p75 1.0]  jac median=0.5s (compile 21.8s)
    => projected maxiter=N wall ~ 86 + 1.0*N s
```

```
precise_QA L12 qr-fixed [s4r0]  wall=73.5s  it=17  cost=5.350101e+07  opt=3.60e+00
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  113.2 ms [p25 85, p75 114]  calls= 34 (trivial 0)
    alpha     3.7s excl-compile (  5.1% of wall); compile call 5.9s
    jac      26.0s ( 35.4%)  calls= 18
    fun       5.2s (  7.1%)  calls= 35
    other    32.6s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=59.8s  per-iter median=0.9s [p25 0.8, p75 1.1]  jac median=0.5s (compile 17.7s)
    => projected maxiter=N wall ~ 60 + 0.9*N s
```

```
precise_QA L12 svd [s0r0]  wall=372.0s  it=200  cost=8.213780e+06  opt=8.83e-04
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   13.5 ms [p25 13, p75 13]  calls=228 (trivial 226)
    alpha     0.3s excl-compile (  0.1% of wall); compile call 0.6s
    jac     120.0s ( 32.3%)  calls=201
    fun      20.5s (  5.5%)  calls=229
    other   230.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=52.1s  per-iter median=1.5s [p25 1.5, p75 1.5]  jac median=0.5s (compile 17.9s)
    => projected maxiter=N wall ~ 52 + 1.5*N s
```


## End-to-end solve: precise_QA L=M=N=12, svd (A100-80GB)
*2026-08-24 22:57:50Z*

- **maxiter**: 200
- **block**: default
- **methods**: svd
- **reps**: 1
- **seeds**: 1,2,3,4
- **jac_chunk**: 1000
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 svd [s1r0]  wall=121.9s  it=40  cost=1.371340e+15  opt=1.40e+05
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 56 (trivial 55)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.8s
    jac      41.6s ( 34.2%)  calls= 41
    fun       7.6s (  6.3%)  calls= 57
    other    71.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=70.2s  per-iter median=1.4s [p25 1.3, p75 1.4]  jac median=0.5s (compile 21.8s)
    => projected maxiter=N wall ~ 70 + 1.4*N s
```

```
precise_QA L12 svd [s2r0]  wall=93.9s  it=24  cost=8.703431e+07  opt=4.58e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 44 (trivial 43)
    alpha     0.0s excl-compile (  0.1% of wall); compile call 0.7s
    jac      33.6s ( 35.7%)  calls= 25
    fun       6.5s (  6.9%)  calls= 45
    other    53.1s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=63.8s  per-iter median=1.4s [p25 1.4, p75 1.4]  jac median=0.5s (compile 22.1s)
    => projected maxiter=N wall ~ 64 + 1.4*N s
```

```
precise_QA L12 svd [s3r0]  wall=109.4s  it=33  cost=1.343757e+08  opt=2.40e-01
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 50 (trivial 49)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.8s
    jac      35.5s ( 32.5%)  calls= 34
    fun       6.6s (  6.0%)  calls= 51
    other    66.4s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=68.5s  per-iter median=1.3s [p25 1.3, p75 1.4]  jac median=0.5s (compile 20.4s)
    => projected maxiter=N wall ~ 69 + 1.3*N s
```

```
precise_QA L12 svd [s4r0]  wall=152.4s  it=63  cost=4.518811e+06  opt=9.16e-04
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 92 (trivial 91)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.7s
    jac      51.5s ( 33.8%)  calls= 64
    fun      10.5s (  6.9%)  calls= 93
    other    89.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=66.5s  per-iter median=1.4s [p25 1.4, p75 1.4]  jac median=0.5s (compile 21.5s)
    => projected maxiter=N wall ~ 66 + 1.4*N s
```

### RETRACTION: the seed ensemble used an unphysical perturbation

The perturbation inherited from branch_experiments/solve_bench.py was

    eq.R_lmn += pert * randn(size) * abs(eq.R_lmn).max()

i.e. WHITE NOISE scaled to max|R_lmn| -- the R00 major-radius coefficient --
applied uniformly to every mode regardless of mode number. Measured at
precise_QA L=M=N=12:

    R_lmn  98.9% of modes lie below 1% of max
           perturbation std per mode = 1.01e-02
           exceeds the mode's own magnitude for 98.9% of modes
           median (perturbation / |coefficient|) = 5.0e+03
           total ||delta||/||v|| = 0.323      <- a nominal "1%" is really 32%
    Z_lmn  median ratio 1.29e+03,  ||delta||/||v|| = 0.245

The spectrum of an equilibrium decays steeply, so uniform noise at the scale of
the largest coefficient swamps every high-order mode by 3-4 orders of magnitude.
The starting point is not a perturbed equilibrium; it is a randomly corrugated
surface. Consequences visible in the data: seed 1 stalls after 10 iterations at
cost 1.16e11 for BOTH qr and qr-fixed, and svd on the same start reaches
1.37e15 with optimality 1.4e5.

THEREFORE the L=12 seed ensemble above does not support conclusions about the
methods. Final costs there are stall points on unphysical configurations, and
the per-seed agreement (bit-identical on seed 1, 68% apart on seed 2) reflects
the starting points, not the solvers. The qr-fixed-inside-the-qr-vs-svd
yardstick result from seed 0 is likewise withdrawn pending a rerun.

FIX: perturb each coefficient in proportion to ITS OWN magnitude,
`R_lmn *= (1 + pert*randn)`, which preserves the spectral hierarchy and gives
||delta||/||v|| = pert exactly. Kept `pert_mode="absolute"` to reproduce the old
behaviour if ever needed. Also exposed xtol/gtol/ftol: every run so far stopped
on `xtol` at optimality 1e-2 to 1e5 while reporting "terminated successfully",
so xtol must be tightened (1e-14) to let gtol govern convergence.


## End-to-end solve: precise_QA L=M=N=12, qr,qr-fixed,svd (A100-80GB)
*2026-08-24 23:02:17Z*

- **maxiter**: 300
- **block**: default
- **methods**: qr,qr-fixed,svd
- **reps**: 1
- **seeds**: 0,1,2,3,4
- **jac_chunk**: 1000
- **pert**: 0.05
- **pert_mode**: relative
- **xtol**: 1e-14
- **gtol**: 1e-10
- **ftol**: 1e-14
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 qr [s0r0]  wall=301.0s  it=300  cost=3.341510e+02  opt=1.05e-06
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   45.4 ms [p25 45, p75 89]  calls=322 (trivial 1)
    alpha    24.7s excl-compile (  8.2% of wall); compile call 6.4s
    jac     157.6s ( 52.4%)  calls=301
    fun      23.6s (  7.9%)  calls=323
    other    88.6s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=76.7s  per-iter median=0.7s [p25 0.7, p75 0.7]  jac median=0.5s (compile 19.7s)
    => projected maxiter=N wall ~ 77 + 0.7*N s
```

```
precise_QA L12 qr [s1r0]  wall=305.2s  it=300  cost=1.763986e+03  opt=2.01e-03
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   45.3 ms [p25 45, p75 89]  calls=338 (trivial 1)
    alpha    24.7s excl-compile (  8.1% of wall); compile call 6.9s
    jac     158.5s ( 51.9%)  calls=301
    fun      24.7s (  8.1%)  calls=339
    other    90.4s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=78.1s  per-iter median=0.7s [p25 0.7, p75 0.7]  jac median=0.5s (compile 19.4s)
    => projected maxiter=N wall ~ 78 + 0.7*N s
```

```
precise_QA L12 qr [s2r0]  wall=376.1s  it=300  cost=1.200507e+02  opt=1.56e-04
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   89.4 ms [p25 89, p75 90]  calls=372 (trivial 0)
    alpha    36.4s excl-compile (  9.7% of wall); compile call 10.4s
    jac     176.5s ( 46.9%)  calls=301
    fun      33.2s (  8.8%)  calls=373
    other   119.6s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=126.0s  per-iter median=0.8s [p25 0.8, p75 0.8]  jac median=0.5s (compile 29.9s)
    => projected maxiter=N wall ~ 126 + 0.8*N s
```

```
precise_QA L12 qr [s3r0]  wall=321.2s  it=300  cost=3.107870e+01  opt=4.36e-03
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   88.7 ms [p25 88, p75 89]  calls=379 (trivial 0)
    alpha    34.0s excl-compile ( 10.6% of wall); compile call 5.6s
    jac     169.7s ( 52.8%)  calls=301
    fun      30.6s (  9.5%)  calls=380
    other    81.4s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=63.9s  per-iter median=0.8s [p25 0.8, p75 0.8]  jac median=0.5s (compile 17.2s)
    => projected maxiter=N wall ~ 64 + 0.8*N s
```

```
precise_QA L12 qr [s4r0]  wall=104.5s  it=48  cost=1.158428e+02  opt=9.20e-09
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  132.1 ms [p25 89, p75 176]  calls= 78 (trivial 4)
    alpha    10.3s excl-compile (  9.9% of wall); compile call 5.6s
    jac      41.9s ( 40.1%)  calls= 49
    fun       8.4s (  8.0%)  calls= 79
    other    38.2s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=59.1s  per-iter median=0.9s [p25 0.8, p75 1.2]  jac median=0.5s (compile 17.5s)
    => projected maxiter=N wall ~ 59 + 0.9*N s
```

```
precise_QA L12 qr-fixed [s0r0]  wall=325.6s  it=300  cost=3.749809e+02  opt=8.59e-07
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   57.5 ms [p25 31, p75 62]  calls=341 (trivial 0)
    alpha    21.3s excl-compile (  6.5% of wall); compile call 7.6s
    jac     168.8s ( 51.8%)  calls=301
    fun      28.3s (  8.7%)  calls=342
    other    99.5s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=87.5s  per-iter median=0.7s [p25 0.7, p75 0.8]  jac median=0.5s (compile 19.7s)
    => projected maxiter=N wall ~ 88 + 0.7*N s
```

```
precise_QA L12 qr-fixed [s1r0]  wall=302.0s  it=300  cost=2.669607e+03  opt=2.37e-03
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   57.0 ms [p25 29, p75 57]  calls=335 (trivial 1)
    alpha    18.4s excl-compile (  6.1% of wall); compile call 6.3s
    jac     171.8s ( 56.9%)  calls=301
    fun      27.8s (  9.2%)  calls=336
    other    77.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=61.0s  per-iter median=0.7s [p25 0.7, p75 0.8]  jac median=0.5s (compile 18.1s)
    => projected maxiter=N wall ~ 61 + 0.7*N s
```

```
precise_QA L12 qr-fixed [s2r0]  wall=309.0s  it=300  cost=1.202390e+02  opt=5.35e-02
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   57.8 ms [p25 57, p75 58]  calls=377 (trivial 0)
    alpha    23.8s excl-compile (  7.7% of wall); compile call 7.6s
    jac     158.8s ( 51.4%)  calls=301
    fun      26.9s (  8.7%)  calls=378
    other    92.0s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=79.9s  per-iter median=0.7s [p25 0.7, p75 0.8]  jac median=0.5s (compile 20.1s)
    => projected maxiter=N wall ~ 80 + 0.7*N s
```

```
precise_QA L12 qr-fixed [s3r0]  wall=332.6s  it=300  cost=3.753685e+01  opt=1.65e-03
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   57.8 ms [p25 58, p75 58]  calls=365 (trivial 0)
    alpha    21.0s excl-compile (  6.3% of wall); compile call 7.9s
    jac     175.9s ( 52.9%)  calls=301
    fun      32.0s (  9.6%)  calls=366
    other    95.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=84.4s  per-iter median=0.8s [p25 0.8, p75 0.8]  jac median=0.5s (compile 23.2s)
    => projected maxiter=N wall ~ 84 + 0.8*N s
```

```
precise_QA L12 qr-fixed [s4r0]  wall=109.7s  it=27  cost=7.619521e+01  opt=9.37e-09
    stopped: Optimization terminated successfully. `xtol` condition satis
    alpha median=  114.2 ms [p25 86, p75 115]  calls= 50 (trivial 3)
    alpha     5.3s excl-compile (  4.9% of wall); compile call 8.5s
    jac      39.0s ( 35.6%)  calls= 28
    fun       8.5s (  7.7%)  calls= 51
    other    48.3s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=86.8s  per-iter median=1.0s [p25 0.8, p75 1.1]  jac median=0.5s (compile 25.1s)
    => projected maxiter=N wall ~ 87 + 1.0*N s
```

```
precise_QA L12 svd [s0r0]  wall=499.8s  it=300  cost=7.948287e+01  opt=1.54e-02
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls=388 (trivial 387)
    alpha     0.5s excl-compile (  0.1% of wall); compile call 0.8s
    jac     178.4s ( 35.7%)  calls=301
    fun      34.9s (  7.0%)  calls=389
    other   285.3s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=69.6s  per-iter median=1.4s [p25 1.4, p75 1.4]  jac median=0.5s (compile 21.9s)
    => projected maxiter=N wall ~ 70 + 1.4*N s
```

```
precise_QA L12 svd [s1r0]  wall=508.8s  it=300  cost=7.928989e+01  opt=2.81e-07
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls=334 (trivial 333)
    alpha     0.5s excl-compile (  0.1% of wall); compile call 0.8s
    jac     187.9s ( 36.9%)  calls=301
    fun      32.1s (  6.3%)  calls=335
    other   287.5s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=69.2s  per-iter median=1.4s [p25 1.4, p75 1.5]  jac median=0.5s (compile 23.8s)
    => projected maxiter=N wall ~ 69 + 1.4*N s
```

```
precise_QA L12 svd [s2r0]  wall=494.6s  it=300  cost=1.926596e+02  opt=8.88e-01
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls=412 (trivial 411)
    alpha     0.5s excl-compile (  0.1% of wall); compile call 0.8s
    jac     155.0s ( 31.3%)  calls=301
    fun      28.3s (  5.7%)  calls=413
    other   310.0s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=68.2s  per-iter median=1.4s [p25 1.4, p75 1.4]  jac median=0.4s (compile 19.8s)
    => projected maxiter=N wall ~ 68 + 1.4*N s
```

```
precise_QA L12 svd [s3r0]  wall=508.5s  it=300  cost=8.096846e+01  opt=1.88e-05
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls=394 (trivial 393)
    alpha     0.5s excl-compile (  0.1% of wall); compile call 0.7s
    jac     175.5s ( 34.5%)  calls=301
    fun      34.2s (  6.7%)  calls=395
    other   297.6s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=69.6s  per-iter median=1.4s [p25 1.4, p75 1.5]  jac median=0.5s (compile 22.0s)
    => projected maxiter=N wall ~ 70 + 1.4*N s
```

```
precise_QA L12 svd [s4r0]  wall=468.7s  it=300  cost=7.398090e+03  opt=2.05e-05
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls=392 (trivial 391)
    alpha     0.5s excl-compile (  0.1% of wall); compile call 0.8s
    jac     158.6s ( 33.8%)  calls=301
    fun      28.1s (  6.0%)  calls=393
    other   280.7s
    peak=4.64 GB (limit 75)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=69.1s  per-iter median=1.3s [p25 1.3, p75 1.4]  jac median=0.5s (compile 20.4s)
    => projected maxiter=N wall ~ 69 + 1.3*N s
```


## Equilibrium diagnostics
*2026-08-24 23:33:41Z*

- **files**: ['eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5', 'eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5', 'eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5', 'eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5', 'eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5', 'eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5']
- **git**: 493422fac
- **desc_dirty**: False
```
eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5
    <|F|>_vol=-1.854596e+10  max|F|=3.3860e+18  V=0.600325  R0/a=2.551763  W_B=2.77655080e+05
    sqrt(g) in [-1.4097e+00, 7.7161e-01]  sign_change=True  iota 2.677603 -> 2.113589
eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5
    <|F|>_vol=-1.367670e+11  max|F|=2.6786e+18  V=0.600327  R0/a=1.338644  W_B=1.59158353e+05
    sqrt(g) in [-3.0082e+00, 6.6204e-01]  sign_change=True  iota 1.538830 -> 0.361381
eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5
    <|F|>_vol=2.902453e+10  max|F|=3.7130e+18  V=0.600325  R0/a=2.551760  W_B=2.78905027e+05
    sqrt(g) in [-1.4097e+00, 7.7160e-01]  sign_change=True  iota 2.915591 -> 1.987108
eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5
    <|F|>_vol=-1.036034e+11  max|F|=1.8935e+18  V=0.600327  R0/a=1.338644  W_B=1.59316299e+05
    sqrt(g) in [-3.0082e+00, 6.6204e-01]  sign_change=True  iota 1.454125 -> 0.322217
eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5
    <|F|>_vol=-1.704447e+15  max|F|=3.9551e+25  V=0.600325  R0/a=3.553430  W_B=2.63352671e+05
    sqrt(g) in [-8.2655e-01, 6.7502e-01]  sign_change=True  iota 1.643384 -> 1.518752
eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    <|F|>_vol=-4.647142e+11  max|F|=3.0477e+18  V=0.600325  R0/a=3.464338  W_B=7.13444810e+05
    sqrt(g) in [-7.7554e-01, 5.8539e-01]  sign_change=True  iota 1.695568 -> 0.237286

eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5
    dR=1.020e-01 dZ=4.489e-01  dV=3.241e-06 dW_B=4.268e-01 daspect=4.754e-01
    diota axis=1.139e+00 edge=1.752e+00  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5
    dR=4.895e-06 dZ=2.690e-05  dV=4.404e-11 dW_B=4.502e-03 daspect=1.310e-06
    diota axis=2.380e-01 edge=1.265e-01  force ratio=-0.639
eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5
    dR=1.020e-01 dZ=4.489e-01  dV=3.241e-06 dW_B=4.262e-01 daspect=4.754e-01
    diota axis=1.223e+00 edge=1.791e+00  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5
    dR=7.285e-02 dZ=2.517e-01  dV=1.087e-06 dW_B=5.151e-02 daspect=3.925e-01
    diota axis=1.034e+00 edge=5.948e-01  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    dR=7.308e-02 dZ=2.578e-01  dV=1.069e-06 dW_B=1.570e+00 daspect=3.576e-01
    diota axis=9.820e-01 edge=1.876e+00  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5
    dR=1.019e-01 dZ=4.122e-01  dV=3.241e-06 dW_B=7.524e-01 daspect=9.062e-01
    diota axis=1.377e+00 edge=1.626e+00  force ratio=-4.712
eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5
    dR=1.517e-07 dZ=4.776e-07  dV=1.981e-12 dW_B=9.924e-04 daspect=1.738e-07
    diota axis=8.471e-02 edge=3.916e-02  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5
    dR=8.828e-02 dZ=3.377e-01  dV=4.328e-06 dW_B=6.547e-01 daspect=1.655e+00
    diota axis=1.046e-01 edge=1.157e+00  force ratio=-inf
eq_precise_QA_L12_qr_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    dR=8.888e-02 dZ=3.358e-01  dV=4.310e-06 dW_B=3.483e+00 daspect=1.588e+00
    diota axis=1.567e-01 edge=1.241e-01  force ratio=-inf
eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5
    dR=1.020e-01 dZ=4.489e-01  dV=3.241e-06 dW_B=4.288e-01 daspect=4.754e-01
    diota axis=1.461e+00 edge=1.665e+00  force ratio=inf
eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5
    dR=7.285e-02 dZ=2.517e-01  dV=1.086e-06 dW_B=5.576e-02 daspect=3.925e-01
    diota axis=1.272e+00 edge=4.684e-01  force ratio=inf
eq_precise_QA_L12_qr-fixed_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    dR=7.308e-02 dZ=2.578e-01  dV=1.069e-06 dW_B=1.558e+00 daspect=3.576e-01
    diota axis=1.220e+00 edge=1.750e+00  force ratio=inf
eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5
    dR=8.828e-02 dZ=3.377e-01  dV=4.328e-06 dW_B=6.530e-01 daspect=1.654e+00
    diota axis=1.893e-01 edge=1.197e+00  force ratio=-inf
eq_precise_QA_L12_qr-fixed_bNone_seed4_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    dR=8.888e-02 dZ=3.358e-01  dV=4.310e-06 dW_B=3.478e+00 daspect=1.588e+00
    diota axis=2.414e-01 edge=8.493e-02  force ratio=-inf
eq_precise_QA_L12_svd_bNone_seed0_det1_rep0.h5 || eq_precise_QA_L12_svd_bNone_seed4_det1_rep0.h5
    dR=8.297e-03 dZ=5.381e-02  dV=1.735e-08 dW_B=1.709e+00 daspect=2.507e-02
    diota axis=5.218e-02 edge=1.281e+00  force ratio=-inf
```

### Local sweep: eq.perturb boundary perturbation, 1-5%, qr only, L=12, maxiter=20

Run on the laptop RTX 5070 (8 GB), peak 2.3 GB, ~100 s per point.

    pert  |dRb|/|Rb|  iters  stop     cost      opt       <|F|> solved  folded
     1%     0.0033     13    gtol   8.51e-13  7.05e-09      1.80        False
     2%     0.0067     14    gtol   7.43e-13  7.57e-09      1.71        False
     3%     0.0100     15    gtol   1.02e-12  5.29e-09      2.07        False
     4%     0.0134     20*   cap    2.50e-12  1.28e-08      3.39        False
     5%     0.0167     20*   cap    2.62e-12  3.58e-08      3.59        False
    (* hit maxiter=20 at opt within ~4x of gtol=1e-8; a few iterations short)

NO TIPPING POINT in 1-5%. Iteration count goes 13, 14, 15, ~21, ~21. Every
solution is a valid equilibrium: sqrt(g) single-signed throughout, aspect ratio
identical between perturbed start and solution, iota 0.42-0.44, <|F|> 1.7-3.6 N.
Continuation is NOT required in this range.

This closes the question opened by the earlier folded-equilibria result: that was
entirely an artefact of how the perturbation was applied, not of perturbation
size and not of the solver. For the record, three perturbation methods compared
at L=12, nominal 1%:

    interior spectral, absolute (branch_experiments)  start cost ~1e24, folded
    interior spectral, relative                       start cost ~1e24, folded
    replace eq.surface, re-solve                      start cost  9.4e24, folded
    eq.perturb on Rb_lmn/Zb_lmn                       start cost  5.5e-02, clean
    (unperturbed control)                             start cost  1.5e-09

Note |dRb|/|Rb| comes out ~1/3 of nominal because |Rb| is dominated by the R00
major-radius coefficient, so the ratio is pert x |that mode's draw|. The
per-mode perturbation is a true 1%.


## End-to-end solve: precise_QA L=M=N=12,20, qr,qr-fixed,svd (A100-80GB)
*2026-08-25 00:00:07Z*

- **maxiter**: 40
- **block**: default
- **methods**: qr,qr-fixed,svd
- **reps**: 1
- **seeds**: 0,1,2
- **jac_chunk**: 1000
- **pert**: 0.03
- **pert_mode**: perturb
- **xtol**: 1e-14
- **gtol**: 1e-10
- **ftol**: 1e-14
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L12 qr [s0r0]  wall=75.2s  it=40  cost=9.762530e-13  opt=1.62e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   89.2 ms [p25 89, p75 89]  calls= 52 (trivial 7)
    alpha     4.4s excl-compile (  5.9% of wall); compile call 5.4s
    jac      37.7s ( 50.1%)  calls= 41
    fun       5.8s (  7.8%)  calls= 53
    other    21.8s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=121.8s  per-iter median=0.8s [p25 0.8, p75 1.0]  jac median=0.5s (compile 17.0s)
    => projected maxiter=N wall ~ 122 + 0.8*N s
```

```
precise_QA L12 qr [s1r0]  wall=94.1s  it=40  cost=1.797042e-12  opt=8.96e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   89.5 ms [p25 89, p75 123]  calls= 48 (trivial 5)
    alpha     4.6s excl-compile (  4.9% of wall); compile call 7.7s
    jac      47.2s ( 50.1%)  calls= 41
    fun       6.9s (  7.4%)  calls= 49
    other    27.7s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=168.3s  per-iter median=0.9s [p25 0.9, p75 0.9]  jac median=0.6s (compile 24.3s)
    => projected maxiter=N wall ~ 168 + 0.9*N s
```

```
precise_QA L12 qr [s2r0]  wall=76.9s  it=40  cost=2.655080e-12  opt=5.53e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   89.1 ms [p25 89, p75 133]  calls= 47 (trivial 6)
    alpha     4.8s excl-compile (  6.3% of wall); compile call 5.5s
    jac      37.5s ( 48.8%)  calls= 41
    fun       5.5s (  7.1%)  calls= 48
    other    23.6s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=124.7s  per-iter median=0.8s [p25 0.8, p75 0.8]  jac median=0.5s (compile 17.0s)
    => projected maxiter=N wall ~ 125 + 0.8*N s
```

```
precise_QA L12 qr-fixed [s0r0]  wall=92.6s  it=40  cost=9.762530e-13  opt=1.62e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   58.5 ms [p25 58, p75 59]  calls= 52 (trivial 6)
    alpha     2.9s excl-compile (  3.2% of wall); compile call 8.4s
    jac      46.3s ( 50.0%)  calls= 41
    fun       7.3s (  7.8%)  calls= 53
    other    27.7s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=167.4s  per-iter median=0.8s [p25 0.8, p75 1.0]  jac median=0.6s (compile 23.7s)
    => projected maxiter=N wall ~ 167 + 0.8*N s
```

```
precise_QA L12 qr-fixed [s1r0]  wall=91.9s  it=40  cost=1.797042e-12  opt=8.96e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   58.2 ms [p25 58, p75 75]  calls= 48 (trivial 4)
    alpha     3.0s excl-compile (  3.3% of wall); compile call 8.4s
    jac      45.9s ( 50.0%)  calls= 41
    fun       6.8s (  7.4%)  calls= 49
    other    27.8s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=163.3s  per-iter median=0.9s [p25 0.8, p75 0.9]  jac median=0.6s (compile 22.8s)
    => projected maxiter=N wall ~ 163 + 0.9*N s
```

```
precise_QA L12 qr-fixed [s2r0]  wall=73.3s  it=40  cost=2.655080e-12  opt=5.53e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   57.8 ms [p25 58, p75 86]  calls= 47 (trivial 5)
    alpha     3.2s excl-compile (  4.3% of wall); compile call 5.9s
    jac      37.1s ( 50.6%)  calls= 41
    fun       5.5s (  7.6%)  calls= 48
    other    21.7s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=122.8s  per-iter median=0.8s [p25 0.8, p75 0.8]  jac median=0.5s (compile 16.6s)
    => projected maxiter=N wall ~ 123 + 0.8*N s
```

```
precise_QA L12 svd [s0r0]  wall=111.6s  it=40  cost=9.762530e-13  opt=1.62e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 52 (trivial 51)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.7s
    jac      47.1s ( 42.2%)  calls= 41
    fun       7.4s (  6.6%)  calls= 53
    other    56.4s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=165.6s  per-iter median=1.6s [p25 1.6, p75 1.7]  jac median=0.6s (compile 24.9s)
    => projected maxiter=N wall ~ 166 + 1.6*N s
```

```
precise_QA L12 svd [s1r0]  wall=113.3s  it=40  cost=1.797042e-12  opt=8.96e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 48 (trivial 47)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.8s
    jac      46.5s ( 41.0%)  calls= 41
    fun       7.4s (  6.6%)  calls= 49
    other    58.5s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=173.0s  per-iter median=1.6s [p25 1.6, p75 1.7]  jac median=0.5s (compile 24.1s)
    => projected maxiter=N wall ~ 173 + 1.6*N s
```

```
precise_QA L12 svd [s2r0]  wall=110.1s  it=40  cost=2.655080e-12  opt=5.53e-08
    stopped: Maximum number of iterations has been exceeded.
    alpha median=   n/a ms [p25    n/a, p75    n/a]  calls= 47 (trivial 46)
    alpha     0.1s excl-compile (  0.1% of wall); compile call 0.7s
    jac      46.1s ( 41.8%)  calls= 41
    fun       6.8s (  6.2%)  calls= 48
    other    56.5s
    peak=4.20 GB (limit 38)  set_by=jac  alpha_raised=False  jac_raised=True
    jac_chunk=1000 of dim_x=4074 (dim_f=16562)
    cadence: startup=154.9s  per-iter median=1.6s [p25 1.6, p75 1.7]  jac median=0.6s (compile 23.6s)
    => projected maxiter=N wall ~ 155 + 1.6*N s
```

```
precise_QA  L20  qr        count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  qr        count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA L20 qr [s2r0]  wall=815.5s  it=40  cost=2.013191e-15  opt=1.53e-10
    stopped: Maximum number of iterations has been exceeded.
    alpha median= 1705.5 ms [p25 1705, p75 3406]  calls= 54 (trivial 4)
    alpha   115.8s excl-compile ( 14.2% of wall); compile call 31.4s
    jac     456.5s ( 56.0%)  calls= 41
    fun      21.8s (  2.7%)  calls= 55
    other   189.8s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=319.7s  per-iter median=15.5s [p25 14.7, p75 17.6]  jac median=10.7s (compile 31.9s)
    => projected maxiter=N wall ~ 320 + 15.5*N s
```

```
precise_QA  L20  qr-fixed  count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  qr-fixed  count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  qr-fixed  count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  svd       count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  svd       count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```

```
precise_QA  L20  svd       count=0  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 23.38GiB.
```


## End-to-end solve: precise_QA L=M=N=20, qr-fixed (A100-80GB)
*2026-08-25 00:28:48Z*

- **maxiter**: 20
- **block**: default
- **methods**: qr-fixed
- **reps**: 1
- **seeds**: 0
- **jac_chunk**: 500
- **pert**: 0.03
- **pert_mode**: perturb
- **xtol**: 1e-14
- **gtol**: 1e-08
- **ftol**: 1e-14
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L20 qr-fixed [s0r0]  wall=358.4s  it=13  cost=1.376326e-15  opt=3.30e-09
    stopped: `gtol` condition satisfied. (gtol=1.00e-08)
    alpha median= 1543.9 ms [p25 1159, p75 1895]  calls= 20 (trivial 3)
    alpha    24.1s excl-compile (  6.7% of wall); compile call 6.7s
    jac     221.5s ( 61.8%)  calls= 14
    fun       7.5s (  2.1%)  calls= 21
    other    98.6s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=500 of dim_x=15946 (dim_f=71442)
    cadence: startup=255.5s  per-iter median=19.8s [p25 17.4, p75 21.0]  jac median=14.4s (compile 33.2s)
    => projected maxiter=N wall ~ 256 + 19.8*N s
```


## End-to-end solve: precise_QA L=M=N=20, qr,qr-fixed (A100-80GB)
*2026-08-25 00:38:47Z*

- **maxiter**: 20
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 1
- **seeds**: 0
- **jac_chunk**: 500
- **pert**: 0.03
- **pert_mode**: perturb
- **xtol**: 1e-14
- **gtol**: 1e-08
- **ftol**: 1e-14
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L20 qr [s0r0]  wall=428.5s  it=13  cost=1.376326e-15  opt=3.30e-09
    stopped: `gtol` condition satisfied. (gtol=1.00e-08)
    alpha median= 3406.0 ms [p25 2556, p75 4160]  calls= 20 (trivial 3)
    alpha    53.0s excl-compile ( 12.4% of wall); compile call 27.2s
    jac     230.8s ( 53.8%)  calls= 14
    fun       8.1s (  1.9%)  calls= 21
    other   109.5s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=500 of dim_x=15946 (dim_f=71442)
    cadence: startup=302.7s  per-iter median=23.2s [p25 18.1, p75 25.9]  jac median=15.0s (compile 35.5s)
    => projected maxiter=N wall ~ 303 + 23.2*N s
```

```
precise_QA L20 qr-fixed [s0r0]  wall=371.1s  it=13  cost=1.376326e-15  opt=3.30e-09
    stopped: `gtol` condition satisfied. (gtol=1.00e-08)
    alpha median= 1554.9 ms [p25 1167, p75 1905]  calls= 20 (trivial 3)
    alpha    24.3s excl-compile (  6.5% of wall); compile call 8.3s
    jac     223.5s ( 60.2%)  calls= 14
    fun       7.7s (  2.1%)  calls= 21
    other   107.4s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=500 of dim_x=15946 (dim_f=71442)
    cadence: startup=284.9s  per-iter median=20.1s [p25 17.6, p75 21.4]  jac median=14.5s (compile 34.2s)
    => projected maxiter=N wall ~ 285 + 20.1*N s
```

### L=20 pair, eq.perturb, converged: qr vs qr-fixed (A100-80GB, seed 0)

First L=20 comparison where BOTH arms terminate on gtol rather than maxiter.
pert=3% via eq.perturb, jac_chunk=500, gtol=1e-8, deterministic.

                        qr            qr-fixed
    iterations          13            13
    alpha calls         20            20
    stop                gtol          gtol
    cost                1.376326326e-15   1.376326295e-15
    optimality          3.2951e-09    3.2951e-09
    |x|                 1.162171970210    1.162171970210
    alpha median      3406.0 ms       1554.9 ms      <- 2.191x
    per-iteration       23.25 s         20.13 s      <- 1.155x
    peak                42.07 GB        42.07 GB

AGREEMENT: same iteration count, same alpha-call count, cost agreeing to
2.3e-08 relative and |x| to 2.4e-13. qr-fixed also reproduced its own earlier
single run exactly, so determinism holds at L=20 too.

ALPHA SPEEDUP 2.191x, matching the standalone kernel sweep's 2.26x at n=14242;
the small dilution is the two triangular solves and norms each alpha iteration
also performs.

END-TO-END ONLY 1.155x, AND THAT IS A jac_chunk ARTEFACT, NOT THE METHOD.
jac median was 15.0 s of a 23.2 s iteration here at chunk=500, against 3.9 s of
14.95 s at chunk=1000 in earlier runs -- dropping the chunk made the Jacobian
~3.8x slower for barely 1 GB saved, shrinking the alpha share from ~35% to
~22.5%. Amdahl on the measured share predicts 1.14x against 1.155x measured.
The chunk was lowered for safety after the 40 GB OOM, but that margin was free
on an 80 GB card: peak is 42.07 GB either way because it is set by eq.perturb,
not by Jacobian chunking (peak_set_by=None, alpha_raised=False, jac_raised=False
-- neither the alpha loop nor jac evaluation moved the high-water mark).

CONSEQUENCE FOR REPORTING: the end-to-end speedup is not a property of the
method alone. It depends on how well jac_chunk is tuned, since that sets the
Jacobian's share of each iteration. 2.19x on the alpha loop is the
method-intrinsic figure; any end-to-end number must state the chunk it was
measured at.


## End-to-end solve: precise_QA L=M=N=20, qr,qr-fixed (A100-80GB)
*2026-08-25 00:52:59Z*

- **maxiter**: 20
- **block**: default
- **methods**: qr,qr-fixed
- **reps**: 1
- **seeds**: 0
- **jac_chunk**: 1000
- **pert**: 0.03
- **pert_mode**: perturb
- **xtol**: 1e-14
- **gtol**: 1e-08
- **ftol**: 1e-14
- **deterministic**: True
- **note**: two passes per method: clean timing, plus a counted pass whose host callbacks perturb the alpha loop (counts only, not timings)
- **git**: 493422fac
- **desc_dirty**: False

```
precise_QA L20 qr [s0r0]  wall=340.3s  it=13  cost=1.376328e-15  opt=3.30e-09
    stopped: `gtol` condition satisfied. (gtol=1.00e-08)
    alpha median= 3420.8 ms [p25 2567, p75 4175]  calls= 20 (trivial 3)
    alpha    53.2s excl-compile ( 15.6% of wall); compile call 26.2s
    jac     144.9s ( 42.6%)  calls= 14
    fun       7.2s (  2.1%)  calls= 21
    other   108.7s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=287.1s  per-iter median=17.4s [p25 12.4, p75 20.0]  jac median=9.1s (compile 26.8s)
    => projected maxiter=N wall ~ 287 + 17.4*N s
```

```
precise_QA L20 qr-fixed [s0r0]  wall=299.3s  it=13  cost=1.376329e-15  opt=3.30e-09
    stopped: `gtol` condition satisfied. (gtol=1.00e-08)
    alpha median= 1556.4 ms [p25 1168, p75 1905]  calls= 20 (trivial 3)
    alpha    24.3s excl-compile (  8.1% of wall); compile call 8.4s
    jac     151.1s ( 50.5%)  calls= 14
    fun       7.6s (  2.5%)  calls= 21
    other   107.9s
    peak=42.07 GB (limit 75)  set_by=None  alpha_raised=False  jac_raised=False
    jac_chunk=1000 of dim_x=15946 (dim_f=71442)
    cadence: startup=273.8s  per-iter median=15.1s [p25 12.6, p75 16.3]  jac median=9.5s (compile 27.7s)
    => projected maxiter=N wall ~ 274 + 15.1*N s
```


## jac_chunk_size: what auto picks, and what actually fits
*2026-08-25 01:11:33Z*

- **note**: one jac evaluation per point; the L=16 solve showed jac sets the peak and the alpha loop never raises it
- **git**: 493422fac
- **desc_dirty**: False
```
case          L  chunk_in  chunk_used   dim_f  n_red   est_GB  jac_peak_GB  jac_s
---------------------------------------------------------------------------------
precise_QA   20       125         125   71442  14242    274.4        35.05    24.7
precise_QA   20       250         250   71442  14242    274.4        35.05    32.7
precise_QA   20       500         500   71442  14242    274.4        35.05    28.3
precise_QA   20       750         750   71442  14242    274.4        35.05    33.3
precise_QA   20      1000        1000   71442  14242    274.4        36.11    37.5
precise_QA   20      1500        1500   71442  14242    274.4        41.33    28.1
precise_QA   20      2000        2000   71442  14242    274.4        48.59    34.1
precise_QA   20      3000        3000   71442  14242    274.4        62.56    40.3
precise_QA   25       100  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 27.3
precise_QA   25       250  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 27.3
precise_QA   25       500  FAILED JaxRuntimeError: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 27.3
```


## Kernel sweep: qr vs qr-fixed (A100-80GB)
*2026-08-25 02:20:44Z*

- **sizes**: [14242]
- **blocks**: [128, 256, 512, 1024, 2048]
- **alpha**: 2.2e-14
- **cond**: 10000000000.0
- **reps**: 5
- **mem_fraction**: 0.95
- **deterministic**: False
- **methods**: qr,qr-fixed
- **note**: one container per n; one subprocess per measurement
- **git**: c52dfe59d
- **desc_dirty**: False
```
     n  method       block     time_ms    peak_GB    gram_rel  spread  gpu
--------------------------------------------------------------------------
 14242  qr          None      1161.9       7.67    6.32e-16    8.3%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     128       655.8      10.59    6.30e-16    5.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     256       549.3       7.97    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed     512       507.9       8.00    6.30e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    1024       515.5       8.01    6.29e-16    0.0%  NVIDIA A100-SXM4-80GB, 81920 MiB
 14242  qr-fixed    2048       577.4       8.10    6.30e-16    0.1%  NVIDIA A100-SXM4-80GB, 81920 MiB
```
