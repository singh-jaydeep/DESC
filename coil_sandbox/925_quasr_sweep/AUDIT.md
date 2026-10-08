# 2026-09-25: QUASR ingestion, and an audit of the three ways to measure "planar-coil friendly"

New line of work: instead of *optimizing* an equilibrium for planar coils (922/923), rank
*existing* equilibria by how much field error a reduced coil representation costs, then zoom in
on the friendly ones. This file is the audit that had to come first — what each candidate
measurement actually computes, and whether it is stable enough to rank thousands of devices.

Code added: `sandbox/quasr.py`. Scripts used are in `scripts/`.

---

## 1. The data source

**QUASR** is the only large public database that ships equilibria *and* coils: 371,701 vacuum
QA/QH devices, each with an optimized FourierXYZ (order 16) coil set. The alternatives were
surveyed and rejected for this purpose: ConStellaration (182k QI boundaries, **no coils**, but
the only large finite-beta source), Landreman's near-axis QS database (500k, near-axis only),
the near-axis QI database (800k+, near-axis only), and DESC's own 14 bundled examples.

The site's REST scheme is undocumented. It was recovered from the navigator's JS bundle and every
endpoint is verified — see `sandbox/quasr.py`'s docstring for the URL forms. `database.json.gz`
is 37 MB and holds the entire 24-column index, so **the 12.7 GB Zenodo tarball is unnecessary**:
filter the index locally and fetch only the devices you want.

### Two facts from the index that shape any sampling design

* **Aspect ratio is discrete.** The scan targeted 24, 20, 12, 10, 8, 6.67, 6, 5, 4, 3.33, 2.86.
  Good for exactly-matched-A strata; useless for regressing on A as a continuous variable.
* **Coil length is pinned at its threshold in 100% of the 159,426 devices with
  `qs_error` < 1e-3.** QUASR fixed the complexity budget and let field error float, so achieved
  complexity is mostly *not* an outcome variable there — `qs_error` at fixed budget is. This is
  the opposite design from Pavone et al., whose regression therefore does not transfer.
* Coverage after that quality cut: nfp 1 is QA-only (4,295); nfp 2 is QA-heavy (49,893 QA /
  7,090 QH); nfp 3–5 are QH-heavy; nfp 6–8 are sparse (<2k). `nc_per_hp` 2–5 covers 151k of 159k.

### Ingestion is verified, not assumed

Two traps, both absorbed by `sandbox/quasr.py`:

* The namelist carries only LASYM, NFP and RBC/ZBS, so `InputReader` dies with
  "M_pol is not assigned". Prepending MPOL/NTOR/PHIEDGE/NCURR makes DESC's own reader **exact**
  (a, A, V match the index row to 1e-9 … 1e-6). Do not hand-parse RBC/ZBS — an attempt to do so
  silently produced a degenerate cross-section.
* The coil dof ordering is **read from the file**: every simsopt DOFs object carries
  `names` = ("xc(0)", "xs(1)", "xc(1)", …). The base curves are then resampled and refitted by
  DESC and the copies rebuilt with `CoilSet.from_symmetry`, which avoids depending on simsopt's
  `RotatedCurve` sign conventions at all.

Reconstruction check, five devices spanning nfp 2–5, QA and QH, nc/hp 2–5:

| quantity | agreement with the index row |
|---|---|
| coil count | exact |
| `total_coil_length` | 7e-14 … 3e-13 |
| `max_msc` | 2e-13 … 2e-11 |
| `max_kappa` | 2e-4 … 6e-3 (a max, so grid-sensitive) |
| `min_coil2coil_dist` | 1e-3 … 1e-2 |
| `min_coil2surface_dist` | 3e-5 … 2e-2 |
| B·n achieved | 4.3e-08 … 6.3e-04, tracking their `qs_error` |

The length and msc agreement at 1e-13 is what licenses discarding the `RotatedCurve` objects:
a wrong symmetry convention would break the count and both distances at once. The 1e-2 residual
on the distances is the boundary *fit* (their target surface is a `SurfaceXYZTensorFourier` in
Boozer angles; the namelist is an RBC/ZBS fit of it), not the coils.

`CoilSet.from_symmetry` materialises every coil as a leaf, so `common.iter_unique` reports all of
them as independent and `evaluate`'s `L_total` is the whole torus. `quasr.coilset(..., full=False)`
returns the unique coils instead, which is the shape `make_start.py` produces and `run_one.py`
expects.

---

## 2. eps_1 / eps_2 on a QUASR boundary: usable, and knob-free

### It requires a solved interior, and it says so

`ContourClamshell` needs `K_vc = -(1/mu0) n x B`, so it needs the interior field. On the
unsolved namelist equilibrium the contour Newton diverges and `build()` raises:

```
ValueError: contour Newton did not converge (residual 9.64e-02); raise `newton`
```

**No silent wrong number** — the build's own 1e-10 guard catches it. So every device costs a
fixed-boundary vacuum solve first.

### The solve reproduces QUASR's own device

Their ι comes from their coil field; ours from a fixed-boundary vacuum solve of the fitted
boundary. On 1630198 (nfp 4 QH, A 8): ι(edge) 1.20312 vs their 1.20455, **0.11%** at
L=M=N=10, improving to **0.01%** at 12. That is a genuine end-to-end cross-check and it should
be run on every device as a quality filter.

### Resolution: L=M=N=10 is the screening resolution

| resolution | time | ι(edge) error | \|F\|/\|grad(B²)/2μ₀\| | eps_1 mean |
|---|---|---|---|---|
| L8 M10 N10 | 129 s | 0.10% | 1.97e-03 | 0.035380 |
| L10 M10 N10 | 124 s | 0.11% | 1.97e-03 | 0.035356 |
| L12 M12 N12 | 597 s | **0.01%** | **4.59e-04** | 0.035517 |

5× the cost buys 4.3× less force error and **0.46%** on eps_1. Screen at 10, spot-check at 12.
This is 923's trap 5.12 (the resolution gap tracks \|force\|) reappearing, but an order of
magnitude smaller because these boundaries are fixed rather than being optimized.

### The arbitrary knobs do essentially nothing

On 1630198 with K = 2·nc_per_hp = 10:

| knob | range | eps_1 mean varies by | eps_2 mean varies by |
|---|---|---|---|
| contour phase | 0 → 0.5 | **0.07%** | 2.8% |
| contour count K | 10 → 30 | **0.04%** | — |
| nodes per contour | 120 → 480 | 0 (8 digits) | — |
| radial resolution | L 8 → 10 | 0.07% | — |

Class guards pass cleanly: Φ_mod monotone, Newton residual 8.3e-17, I/G −6.5e-19, 99.29% of the
Φ̃ modes kept. The per-contour eps_1 is a palindrome (0.01513, 0.02425, 0.03264, 0.04372,
0.04523, 0.04677, 0.04523, …), as stellarator symmetry demands — a free self-consistency check.

**Why eps_2 is the right B=2 quantity, and the arc projection is not:** `clamshell.best_breaks`
is an exact DP over all 2-segmentations, so eps_2 *optimizes over hinge placement* instead of
picking a rule. That is precisely the freedom the arcB2 projection leaves dangling. `make_start.py`
already has `--hinge-s` to accept DP breakpoints, so the two connect.

Cost: 5 s for K=10 including the eps_2 DP, against ~125 s for the solve. **The solve dominates**,
so ~130 s/device, i.e. ~200 devices in 7 h of GPU.

---

## 3. The projections are NOT usable as a screen

This was the alternative: take QUASR's own optimized FourierXYZ coils, project them onto planar
facets or B=2 arcs, and rank devices by the resulting B·n penalty. It fails on two counts.

### 3a. The planar projection is biased by an order of magnitude

On 1630198, `to_FourierPlanar(N=8)` per coil:

```
QUASR FourierXYZ N=16 (baseline)   bn 6.26e-04
to_FourierPlanar N=8               bn 5.14e-02      82x worse
planar fit deviation: max 0.120 m = 0.96 a
d_cc 0.0995 -> 0.0613     d_pc 0.1474 -> 0.1184
```

The flattened coil sits almost a **full minor radius** from the original. For contrast, 923's
actual planar *optimization* on precise_QA reached 4.59× its FourierXYZ result. The projection is
therefore ~18× more pessimistic than optimization, because it prices "nobody re-optimized the
centers, normals or currents", not "planarity costs field error". A ranking built on it would
order equilibria by how badly an unoptimized fit happens to fail.

Note also that `to_FourierPlanar` fits an **unweighted** SVD plane on parameter-spaced samples,
while eps_1 uses an **arclength-weighted** covariance. QUASR's coils are near-uniform in
arclength by construction, so the two agreed here — but that is a property of this database, not
of the method.

### 3b. The arcB2 projection depends on the hinge rule, and mostly produces invalid coilsets

Polar arcs B=2, M=5, 14 hinge placements on the same device:

| hinge rule | bn | d_cc | κ | linked |
|---|---|---|---|---|
| DP-optimal (min out-of-plane RMS) | **2.32e-02** | 0.050 | **357.9** | 0 |
| equal parameter, phase 0.125 | 3.39e-02 | 0.006 | 5.4 | 0 |
| cut-angle 100° | 3.80e-02 | 0.043 | 5.4 | 0 |
| cut-angle 20° | 3.89e-02 | 0.016 | 6.0 | 0 |
| cut-angle 40° | 4.08e-02 | 0.019 | 5.1 | **8** |
| cut-angle 120° | 4.31e-02 | 0.003 | 6.4 | 0 |
| cut-angle 80° | 4.66e-02 | 0.022 | 5.0 | 0 |
| cut-angle 0° | 4.84e-02 | 0.043 | 5.9 | 0 |
| cut-angle 140° | 4.88e-02 | 0.008 | 6.6 | **12** |
| equal parameter, phase 0.25 | 4.82e-02 | 0.022 | 6.5 | **8** |
| cut-angle 60° | 5.00e-02 | 0.031 | 5.8 | 0 |
| cut-angle 160° | 5.18e-02 | 0.033 | 6.3 | 0 |
| equal parameter, phase 0.0 | 5.20e-02 | 0.003 | 5.8 | **8** |
| long-axis hinges | 5.61e-02 | 0.048 | 4.6 | 0 |

* **bn spread 2.42×** over hinge placement alone, against a cross-device signal that has to be
  larger than that to be visible.
* The DP-optimal hinges *do* minimise bn (1.46× better than the best rule), so the geometric
  optimum and the field optimum do coincide — **but that variant has κ = 358** against a bound
  near 5. **Corrected in §9d:** this is NOT a degenerate segmentation. The DP splits on this device
  are balanced (shorter arc / total = 0.294 … 0.427), so the curvature blow-up comes from
  `PolarPlanarArcCoil.from_values` fitting an arc AT those hinge parameters, not from the
  segmentation. The hinge placement is sound; the arc fit at it is what fails.
* 4 of 14 placements produce **linked** coils, and every one collapses d_cc from 0.0995 to
  0.003–0.050 against a 0.1 m bound. These are not feasible coilsets, let alone optimized ones.

**Verdict: drop the projection screen.** Use eps_1/eps_2 on the boundary and spend the GPU on
real optimizations for the sample it selects.

---

## 4. eps_1 of QUASR's real coils: two findings

### 4a. eps_1 and the literature's eta_SVD are the same metric

sigma_i ∝ sqrt(lambda_i), so when lambda_1 = lambda_2 exactly,

    eta_SVD = sigma_min / (sigma_1 + sigma_2 + sigma_3) = eps_1 / (eps_1 + 2)

Measured on 1630198's five unique coils: eps_1 0.09868 predicts eta 0.04702, measured
**0.04698**; eps_1 0.23119 predicts 0.10362, measured **0.10329**. Agreement to 0.3%, because
modular coils are near-circular in plane; departures measure in-plane elongation and nothing
else.

Consequence: Pavone, Kwak & Warmer (arXiv:2604.26763) already regress eta_SVD of *optimized
coils* on boundary geometry at R² = 0.882, so **that result transfers directly to eps_1-of-coils
and it is not a new metric.** What is new here is eps_1 of the current-potential *proxy
contours* — a stage-1 quantity that needs no coils. Convert with eps_1 = 2 eta/(1 − eta) rather
than fitting an empirical factor, and always label which eps_1 is meant.

### 4b. The proxy contours are much flatter than the real coils

On 1630198: proxy contours eps_1 = 0.0354, QUASR's actual optimized coils eps_1 = 0.1724 —
a factor **4.9**. Same story for the clamshell ratio: C = 0.947 for the proxy against 1.36–1.96
for the real coils, i.e. the proxy looks two-plane-friendly where the real coils are worse than
the generic decay.

That is expected in direction — the real coils sit at a standoff of 1.2 a and carry length and
curvature constraints that push them off the streamline shape — but the *size* matters. A
constant offset is harmless for ranking; a varying one is fatal. Per-coil, the two do span
comparable relative ranges (real 0.099→0.231 = 2.3×; proxy 0.015→0.047 = 3.1×).

*(Cross-device stability of this ratio: see section 5, measurement running.)*

---

## 5. Cross-device: the proxy-to-coil offset is NOT a constant (n = 4)

Four devices, each solved at L=M=N=10, `eps_1` on K = 2·nc_per_hp proxy contours against `eps_1`
of QUASR's own optimized coils on the same device:

| ID | device | A | d_pc/a | proxy eps_1 | coil eps_1 | ratio | η/eps_1 | predicted |
|---|---|---|---|---|---|---|---|---|
| 1630198 | nfp4 QH nc5 | 8 | 1.18 | 0.03536 | 0.17238 | **4.88** | 0.4565 | 0.4603 |
| 197197 | nfp2 QA nc4 | 20 | 10.1 | 0.01018 | 0.20068 | **19.71** | 0.4513 | 0.4544 |
| 1265411 | nfp3 QH nc3 | 24 | 6.42 | 0.02226 | 0.53406 | **23.99** | 0.3900 | 0.3946 |
| 1120872 | nfp5 QH nc2 | 20 | 5.12 | 0.01471 | 0.42487 | **28.88** | 0.4092 | 0.4124 |

ι reproduces QUASR's edge value to 0.00–0.11% on all four, Φ_mod is monotone on all four, worst
Newton residual 9e-17. The measurement itself is sound.

**The offset spans 5.92×** (4.88 … 28.88) and the lowest sits with the lowest aspect ratio and
the smallest coil standoff. That is mechanically sensible: the ρ=1 contours of a thin torus hug a
nearly circular cross-section and look very planar, while coils at d_pc = 5–10 a have to be far
more non-planar. **The proxy lives on the boundary and knows nothing about standoff**, which is
the free parameter QUASR varied most.

**Rank agreement is not established either.** Spearman(proxy, coil) = **−0.200**, Pearson
**−0.207** across these four. At n = 4 that is indistinguishable from noise (p ≈ 0.8) and must not
be read as a negative result — but it is no evidence for a positive relationship, and 197197 is a
concrete counterexample in absolute terms: the flattest proxy of the four (0.0102) sits on coils
as non-planar as the A=8 device (0.2007). Aspect ratio, nfp and nc_per_hp all co-vary across these
four, so nothing can be separated yet.

**Consequences for the recording run:**

* Compare eps_1 **within an aspect-ratio stratum**. QUASR's aspect ratios are discrete, so exact
  strata are available at no cost.
* Record `min_coil2surface_dist` alongside (the recorder does), so the standoff explanation can be
  tested directly rather than assumed.
* Record the coil eps_1/eps_2/η as well as the proxy values. They cost ~2 s, need no solve, and
  they are the only thing that can tell us whether the proxy ranks anything.

The η_SVD identity of §4a holds on all four to 0.8–1.2%, measured consistently just below the
λ₁=λ₂ prediction, which is the in-plane elongation correction showing up.

---

## 6. Where this leaves the program

* **Ingestion: done and verified.** `sandbox/quasr.py`.
* **Measurement: `925_quasr_sweep/record_eps.py`.** No optimization, no screening -- it records
  eps_1/eps_2 (proxy and coil) plus the index row and the class diagnostics, one JSON line per
  device, resumable. ~130 s/device at L=M=N=10, of which ~125 s is the solve.
* **The projections are out** (§3): biased ~18× for planar, 2.42× hinge-dependent for arcB2, and
  mostly infeasible coilsets. Do not use them to rank.
* **eps_1-of-coils is not a new metric** (§4a); eps_1-of-proxy-contours is, and its relationship
  to coils is currently unmeasured beyond n = 4 (§5).
* Open, and cheap to settle once there are numbers: does eps_1 rank within a stratum; how often is
  Φ_mod non-monotone (i.e. how much of QUASR is outside the modular-proxy class at all); and does
  the proxy/coil offset collapse once standoff is controlled for.

---

## 7. THE FIRST RECORDING RUN: 36 devices, A < 10 (2026-09-25)

`record_eps.py --aspect 2,10 --n 36 --seed 0 --qs-max -3.0 --res 10` -> `eps_A10.jsonl`,
analysed by `analyze_eps.py`. 36 devices drawn from the 50,978 matching, ~165 s each.

### 7a. 31% of devices are outside the modular-proxy class, and ELONGATION is why

11 of 36 failed with "Phi_mod is not monotone in zeta", i.e. the poloidal sheet current
`-sqrt(g) K^theta` reverses sign somewhere on the boundary, so the Phi contours are not
single-valued zeta(theta) and are not modular-coil-like. Not marginal: dPhi/dzeta straddles zero
with large magnitude on both sides.

The discriminator is **elongation**, and nothing else:

| quantity | excluded (median) | kept (median) | Mann-Whitney p |
|---|---|---|---|
| **mean_elongation** | **3.519** | **2.512** | **0.000** |
| **max_elongation** | **5.629** | **3.367** | **0.000** |
| aspect_ratio | 4.000 | 6.667 | 0.336 |
| nfp | 2 | 2 | 0.777 |
| helicity | 0 | 0 | 0.982 |
| nc_per_hp | 3 | 3 | 0.592 |
| mean_iota | 0.400 | 0.300 | 0.972 |
| qs_error | −3.431 | −3.494 | 0.837 |

kept elongation spans 1.89–3.49, excluded 2.36–7.47; the best single threshold is 3.3
(23/25 kept below, 8/11 excluded above). **It survives conditioning on aspect ratio**, which is
the obvious confound:

```
A=4:   kept 2.18 2.43 2.79 3.37    excluded 3.00 3.06 3.37 3.52     (overlapping)
A=8:   kept 2.10 2.77 2.97 3.07    excluded 4.47 4.99 5.55          (clean split)
A=10:  kept 1.89 2.18 2.18 2.24 2.81   excluded 4.95 7.47           (clean split)
```

**This is a selection effect that conditions everything below**: eps_1 is undefined for the most
elongated third of low-aspect QUASR, so any statement about eps_1 is a statement about
round-ish cross-sections.

### 7b. The proxy does NOT rank QUASR's achieved coil non-planarity

| comparison | n | Spearman | p |
|---|---|---|---|
| proxy eps_1 vs coil eps_1 | 25 | **−0.207** | 0.321 |
| proxy eps_2 vs coil eps_2 | 25 | −0.022 | 0.919 |
| proxy C vs coil eps_2 | 25 | +0.026 | 0.901 |
| proxy eps_1 vs coil eps_1, \|Δι\| ≤ 0.01 | 16 | −0.100 | 0.713 |
| proxy eps_2 vs coil eps_2, \|Δι\| ≤ 0.01 | 16 | +0.147 | 0.587 |

Spearman(proxy eps_1, coil eps_1) = −0.207 with a 95% CI of **[−0.557, +0.205]**. The CI excludes
any positive rank correlation above ~0.21, so this is not merely "underpowered": **a strong
positive relationship is ruled out.** Restricting to the 16 devices whose ι our solve actually
reproduces does not rescue it.

### 7c. The offset is systematic, and tracks slenderness

ratio = coil eps_1 / proxy eps_1 spans **2.57 … 10.83** (median 8.08, 4.2×):

| vs | Spearman | p |
|---|---|---|
| aspect_ratio | +0.620 | 0.001 |
| d_cc/a | +0.635 | 0.001 |
| nfp | +0.629 | 0.001 |
| minor_radius | −0.630 | 0.001 |
| d_pc/a | +0.395 | 0.050 |
| mean_elongation | −0.455 | 0.022 |

With R0 ≈ 1 these are largely one variable: **the more slender the device and the further the
coils sit in units of a, the more the ρ=1 proxy understates coil non-planarity.** Independently
reproduces §5's aspect-ratio finding on a disjoint sample.

### 7d. Health, and what |Δι| is really flagging

force_rel 9.9e-06 … 1.2e-02 (median 6.6e-04); Newton residual ≤ 2e-16; modes_kept ≥ 0.9869;
monotonic 25/25 by construction; |Δι| median 0.0023, max 0.0826, with 16/25 inside 0.01.

The three worst solves are the *same three devices* as the three largest ι mismatches and the
three lowest mode retentions (2477990: 1.20e-02 / 0.0643 / 0.9884; 1659129: 9.19e-03 / 0.0826 /
0.9869; 1620599: 6.60e-03 / 0.0380 / 0.9914), all QH at A = 6–8. So **|Δι| is a real quality flag
tracking force balance**, and it is the right gate — not the relative percentage, which is
meaningless for the QA devices whose iota target was 0.1.

### 7e. What this does and does not settle

**Settled:** contours of Φ on ρ = 1 do not rank the non-planarity of the coils QUASR actually
built, on devices with A < 10 and round-ish cross-sections. And they are undefined for the
elongated third.

**NOT settled — this is not a verdict on the original question.** Three reasons:

1. **The target is the wrong variable.** QUASR's coil eps_1 is not "the least non-planar coil set
   this boundary admits". Length is pinned at its threshold for 100% of devices (§1), with
   kappa ≤ 5, msc ≤ 5 and d_cc ≥ 0.1 alongside, so coil eps_1 partly reads out those constraints
   rather than the boundary. No boundary-only predictor can track it perfectly.
2. **The surface is probably wrong.** §7c's correlations say the mismatch grows with standoff, and
   §7a's non-monotonicity is the textbook pathology of a current potential evaluated on the plasma
   boundary rather than on a winding surface offset outward. One change addresses both.
3. **The quantity the program actually wants is untouched**: `bn_reduced / bn_xyz` at matched coil
   budget, from real optimizations. That is a different number from coil eps_1 and nobody has
   measured it.

**The one clear methodological next step** is to compute the proxy on an outward-offset winding
surface at the device's own coil standoff instead of on rho = 1 — testable precisely against the
11 devices this run had to exclude, and against the ratio's correlation with d_pc/a. It is a real
change to `contour_curve.py` (the grid and `K_vc` would come from an offset surface, not
`eq.surface`), not a flag.

---

## 8. eps_1 vs eps_2 WITHIN each family: the proxy's eps_2 is redundant, the coils' is not

`analyze_eps12.py` on the same 25 devices. eps_2 is normalised by the same whole-curve in-plane
extent as eps_1, so eps_2/eps_1 is a pure ratio of out-of-plane RMS (best 2-plane vs best single
plane), and C = 4 eps_2/eps_1 measures it against the generic B^-2 decay.

### 8a. Proxy contours: eps_2 ~ eps_1/4, almost always

| | per-device means (n=25) | pooled contours (n=186) |
|---|---|---|
| eps_1 vs eps_2 Spearman | **+0.970** (p 1.3e-15) | **+0.918** (p 5.4e-76) |
| eps_1 vs eps_2 Pearson | +0.983 | +0.916 |
| eps_2/eps_1 median | 0.2567 (0.216 … 0.312) | **0.2480** (0.174 … 0.494) |
| C median | 1.027 (0.865 … 1.248) | 0.992 (0.695 … 1.978) |
| C < 1 | 10/25 | 97/186 |
| log-log slope | 1.162 ± 0.051 | 1.070 ± 0.036 |
| eps_1 vs C Spearman | +0.495 (p 0.012) | +0.305 (p 2.3e-05) |

The pooled eps_2/eps_1 median is **0.2480** against the generic 0.25 exactly, C sits on 1, and
C ≶ 1 is close to a coin flip. The positive eps_1–C correlation says the flattest contours are also
the ones that split best, so C and eps_1 are ALIGNED axes, not independent ones.

**eps_2 carries almost no information beyond eps_1 on the proxy.** That is a direct problem for the
planar-vs-arcB2 question: the distinction eps_1-vs-eps_2 was meant to capture does not exist in
the proxy.

### 8b. QUASR's coils: eps_1 and eps_2 genuinely decouple

| | per-device means (n=25) | pooled coils (n=93) |
|---|---|---|
| eps_1 vs eps_2 Spearman | **+0.757** (p 1.2e-05) | **+0.629** (p 1.5e-11) |
| eps_2/eps_1 median | 0.3318 (**0.180 … 0.547**) | 0.3086 (**0.122 … 0.670**) |
| C median | **1.327** (0.718 … 2.190) | **1.234** (0.489 … 2.681) |
| C < 1 | **5/25** | 28/93 |
| log-log slope | 1.246 ± 0.229 (R² 0.56) | 0.798 ± 0.104 (R² 0.39) |
| eps_1 vs C Spearman | +0.245 (p 0.24) | −0.167 (p 0.11) |

Real coils spread eps_2/eps_1 over 3–5× rather than clustering at 0.25, and their C sits well
ABOVE 1 — only 5 of 25 devices have coils that genuinely prefer a two-plane split, against 10 of
25 for the proxy. **The independent "how two-plane-able is this" axis exists in the coils and is
invisible in the proxy.** (The log-log slopes disagree between the pooled and per-device views,
1.246 vs 0.798, so do not read an exponent off them.)

### 8c. The two families do not agree on C

Spearman(proxy C, coil C) = **+0.295 (p 0.153)**, Pearson +0.232 (p 0.265), n = 25. Coil C's
median is 1.29× the proxy's. So C fails as a stage-1 predictor in the same way eps_1 did in §7b —
and this time by direct measurement, not the aspect-ratio artifact that forced §9k's retraction.

### 8d. Within-device structure

Ranking the curves inside a single device, eps_1 and eps_2 agree only weakly: mean within-device
Spearman +0.280 for the proxy (median +0.600, 18/25 positive) and +0.361 for the coils (median
+0.500, 15/20 positive), with individual devices as extreme as −1.0. **Noisy by construction** —
stellarator symmetry makes the proxy contours degenerate in pairs, and nc_per_hp = 2–3 leaves very
few distinct coils — so this is suggestive only.

Spread across the curves of one device is consistently larger for coils than for the proxy:

| | eps_1 spread (median, max) | eps_2 spread (median, max) |
|---|---|---|
| proxy | 1.30× , 5.06× | 1.29× , 2.56× |
| coils | **1.72× , 6.87×** | **1.86× , 5.60×** |

A planar coil set is limited by its WORST coil, so the proxy systematically understates how
unequal the real coils are. That is relevant to 923 §9p's worst-offender targeting: the foliation
the optimizer was shaping is smoother than the coil set it is meant to stand in for.

### 8e. Summary of §8

* On the proxy, eps_2 is a rescaled eps_1 (ρ = 0.92–0.97, ratio pinned at the generic 0.25).
  Recording both is cheap, but do not expect them to say different things.
* On real coils they are distinct quantities (ρ = 0.63–0.76, ratio spanning 3–5×), and real coils
  are mostly WORSE than generic at two-plane splitting (C median 1.23–1.33).
* Neither eps_1 (§7b) nor C (§8c) ranks the corresponding coil quantity.
* Together with §7e: if the proxy is to distinguish planar from piecewise-planar friendliness, the
  ρ = 1 contour family is too smooth and too generic to do it. The offset-winding-surface variant
  of §7e is the first thing to try, and §8a gives it a sharp acceptance test — on an offset
  surface, eps_2/eps_1 should stop sitting on 0.25 and start spreading the way the coils' does.

---

## 9. The eps_2 / eps_1 ratio: the nesting holds, and where the gap is large

`analyze_eps21_ratio.py`. eps_2 minimises over 2-segmentations, a set containing the degenerate
split that reproduces the single-plane answer, so eps_2 <= eps_1 must hold identically. Both use
the same whole-curve (lambda_1 lambda_2)^(1/4) denominator, and `freeze` recomputes n0 as the true
minimum eigenvector, so the denominators agree exactly and the ratio is a clean comparison of
out-of-plane RMS.

### 9a. eps_2 <= eps_1 in all 279 curves, zero violations

186 proxy contours + 93 coils, no exceptions. This is a real implementation check rather than a
restatement of the nesting: `best_breaks` searches rotation offsets only on a `stride`-subsampled
grid, so it could have missed the near-degenerate split. It never did.

### 9b. The distributions

| eps_2/eps_1 | min | p5 | p25 | median | p75 | p95 | max | <0.25 | <0.20 | <0.15 |
|---|---|---|---|---|---|---|---|---|---|---|
| proxy contours (186) | 0.174 | 0.201 | 0.229 | **0.248** | 0.274 | 0.346 | 0.494 | 97 | 8 | **0** |
| QUASR coils (93) | **0.122** | 0.153 | 0.240 | 0.309 | 0.390 | 0.586 | 0.670 | 28 | 16 | **5** |

As a gain factor: proxy 2.0× … 5.8×, coils 1.5× … **8.2×**. A second plane always buys at least
1.5×, and the ratio never approaches 1.

### 9c. The extremes are GENUINE clamshells, not DP slivers

Segment balance = shorter arc / total, by arclength; 0.5 is two equal arcs, ->0 is a sliver.

```
best    204752 coil 2   eps_1 0.345 -> eps_2 0.042   8.17x   balance 0.398
        204752 coil 3   eps_1 0.372 -> eps_2 0.046   8.07x   balance 0.427
        791647 coil 3   eps_1 0.233 -> eps_2 0.031   7.64x   balance 0.317
       1066871 coil 2   eps_1 0.260 -> eps_2 0.035   7.33x   balance 0.342
worst    10510 coil 1   eps_1 0.245 -> eps_2 0.164   1.49x   balance 0.415
        885868 coil 0   eps_1 0.175 -> eps_2 0.115   1.52x   balance 0.379
       1042631 coil 0   eps_1 0.350 -> eps_2 0.224   1.56x   balance 0.490
```

All balances are 0.24 … 0.50 at both ends, so balance does not discriminate — the difference is
genuinely in the shape, and the large gains are real two-plane structure.

204752 supplies three of the top eight coil gains and 791647 another three, which first looked
like clamshell-ability clustering by device. **§9f walks that back**: the ICC is only 0.33, so the
device effect is real but modest and most of the variation is coil-to-coil within a device.

### 9d. Correction to §3b

On 1630198 the DP segmentations are balanced (0.294, 0.335, 0.312, 0.388, 0.427 for the five unique
coils), with hinges at s/2pi = (0.183, 0.890), (0.035, 0.700), (0.617, 0.929), (0.183, 0.571),
(0.140, 0.567). So §3b's kappa = 358 was NOT caused by a degenerate segmentation as first written;
it is the arc FIT at those hinge parameters that blows up. §3b has been amended.

### 9e. SUPERSEDED by 9f — spread of device means is the wrong comparison

The first version of this section compared the spread of per-device MEAN ratios (proxy 1.44x vs
coils 3.0x). That comparison is contaminated: averaging K curves shrinks the spread of the means by
~sqrt(K), and the proxy averages TWICE as many curves per device as the coils (K = 2 nc_per_hp
contours against nc_per_hp coils). Part of the proxy's tighter means is therefore arithmetic, not
physics. Use the variance decomposition in §9f.

### 9f. The ratio VARIES in the proxy; what it lacks is BETWEEN-DEVICE range

One-way random-effects decomposition of log(eps_2/eps_1):

| | per-curve range | median | sd(log) BETWEEN devices | sd(log) WITHIN device | ICC |
|---|---|---|---|---|---|
| proxy contours (186 curves, 4–12 per device) | 0.174 … 0.494 (2.85x) | **0.248** | **0.0805** (1.084x / sd) | 0.1492 (1.161x / sd) | 0.226 |
| QUASR coils (93 curves, 2–6 per device) | 0.122 … 0.670 (5.48x) | 0.309 | **0.2345** (1.264x / sd) | 0.3349 (1.398x / sd) | 0.329 |

Three conclusions, replacing §9e:

1. **The proxy ratio is not fixed** — it varies 2.85x across contours. What is remarkable is that
   its MEDIAN is 0.248 against the analytic generic 0.25: on average the proxy contours behave
   almost exactly like generic subdivision. The coils' median, 0.309, is shifted toward "two planes
   help LESS than generic".
2. **The between-device variance component is the fair comparison**, since it does not depend on how
   many curves are averaged: log-sd **0.0805 (proxy) vs 0.2345 (coils)**, so the coils carry about
   **2.9x more device-to-device signal**. §9e's conclusion stands; 2.9x is the defensible number.
3. **ICC is low in BOTH families (0.226, 0.329)**, so most of the variation in the two-plane
   advantage is curve-to-curve within a device rather than device-to-device. The two-plane advantage
   is primarily a property of the individual coil, which is why §9c's apparent per-device clustering
   was overstated.

Acceptance test for the offset-surface variant of §7e, restated correctly: on an offset winding
surface the proxy's **between-device log-sd** of eps_2/eps_1 should rise from 0.08 toward the coils'
0.23, and the median should move off 0.25. If it does not, this contour family cannot separate
planar from piecewise-planar friendliness however it is computed.

---

## 10. A worked example: QUASR 1328095, where eps_1 >> eps_2

`show_clamshell.py --id 1328095` -> `figures/clamshell_1328095.png`. The device with the largest
proxy two-plane gain in the §7 sample: nfp 4 QH, A = 8.0, nc/hp = 5, C = 0.863, and |d iota| =
0.0012 so our solve reproduces QUASR's own device.

| curve | eps_1 | eps_2 | gain |
|---|---|---|---|
| proxy contour 3 | 0.03508 | 0.00657 | **5.34x** |
| all 10 proxy contours | 0.0333 … 0.0373 | 0.0066 … 0.0098 | 3.77 … 5.34x |
| QUASR's own coil 0 | 0.45975 | 0.10392 | **4.42x** |

The figure's left panels view each curve edge-on to its single best-fit plane (vertical scale
magnified x43 and x2 respectively, stated on the axis -- at true aspect a contour whose
out-of-plane excursion is ~1% of its in-plane size is a featureless line). The proxy contour is a
clean **saddle**, +0.05 on one side and -0.06 on the other, which is exactly what one plane cannot
follow and two tilted planes can; the DP hinges sit near the two extremes.

The right panels plot the quantity eps_1 and eps_2 are literally the RMS of -- signed distance
from the plane, round the curve. One plane swings the full +-0.05; two planes collapse it to a
ripple with cusps at the hinges. That collapse IS the 5.34x.

Two observations about the object:

* The single-plane residual completes roughly **two full oscillations** round the curve, which is
  why B = 2 is the natural split and why the hinges land where they do.
* The coil row shows the same mechanism at ~13x the amplitude (eps_1 0.46 vs 0.035) with a much
  messier residual: real coils carry short-wavelength structure no number of planes removes, which
  is why the coil gain is 4.42x rather than 5.34x. This is the §8b decoupling seen on one curve.
* Contour gains come in mirror pairs (2<->8, 3<->7, 4<->6 to 3-4 digits) -- stellarator symmetry,
  a free consistency check on the whole chain from namelist to eps.

`show_clamshell.py` caches the extracted curves to `figures/curves_<ID>.npz`, so re-rendering is
instant and only a new device costs the ~3 min solve.
