# precise_QH initial starts

One cold start per representation, pinned in `starts.json` twice: `file_sha256` (the h5 file) and
`params_sha256` (SHA-256 over each unique coil's parameter arrays, the precise_QA bundle's
convention). `check_bundle.py` checks both.

**`viewer.html`** shows all seven in 3D (offline, no dependencies): every one of the 32 coils, the
plasma surface, the closest coil-coil approach and each unique coil's tightest-curvature point.
Regenerate with `python tools/make_coil_viewer.py "initial starts/viewer.html" --dir "initial starts"`. Each was produced by a constraint-only feasibility solve aimed 25% inside
every bound, from a shaped planar parent (the plasma cross-section offset in-plane, fitted with
FourierPlanar N=7), then converted to the representation. Arcs were fitted with
`PiecewisePlanarArcCoil.from_values(..., M=3, fit_method="chord")`.

All seven are adjudicated against the TRUE (paper) bounds with the sweep's dense check
(`tools/qh_common.evaluate`). Margins are value / bound, so **below 1 on length, kappa and
kappa_MS, and above 1 on d_cc and d_pc, is satisfied**.

| rep | length (m) | kappa (1/m) | kappa_MS (1/m²) | d_cc | d_pc | max corner | bn | linked | verdict |
|---|---|---|---|---|---|---|---|---|---|
| `planarN7` | 1.939 (0.663) | 6.30 (0.461) | 13.88 (0.743) | 116.3 mm (1.988) | 149.4 mm (1.362) | 0° | 8.661e-02 | 0 | FEASIBLE |
| `arcB2` | 2.164 (0.740) | 6.87 (0.503) | 7.34 (0.393) | 93.5 mm (1.598) | 139.6 mm (1.273) | 80° | 1.119e-01 | 0 | FEASIBLE |
| `arcB3` | 1.971 (0.674) | 10.24 (0.749) | 12.09 (0.646) | 106.5 mm (1.821) | 139.7 mm (1.274) | 65° | 9.754e-02 | 0 | FEASIBLE |
| `arcB5` | 1.936 (0.662) | 10.09 (0.738) | 13.89 (0.743) | 115.7 mm (1.977) | 149.3 mm (1.361) | 23° | 8.696e-02 | 0 | FEASIBLE |
| `arcB7` | 1.935 (0.662) | 8.96 (0.655) | 12.87 (0.689) | 115.4 mm (1.972) | 149.0 mm (1.358) | 16° | 8.674e-02 | 0 | FEASIBLE |
| `xyzN4` | 1.947 (0.665) | 6.05 (0.442) | 13.88 (0.742) | 116.4 mm (1.989) | 149.2 mm (1.360) | 0° | 8.969e-02 | 0 | FEASIBLE |
| `xyzN6` | 1.937 (0.662) | 6.33 (0.463) | 13.88 (0.743) | 114.4 mm (1.955) | 148.8 mm (1.356) | 0° | 8.706e-02 | 0 | FEASIBLE |

True bounds: length ≤ 2.925511 m per coil, kappa ≤ 13.672824 1/m, kappa_MS ≤ 18.694612 1/m²,
d_cc ≥ 0.058510 m, d_pc ≥ 0.109707 m.

Notes

* The length column matters for the sweep's tight end: at ×0.8 (2.340 m) `arcB2` starts with
  only ~7.5% headroom; the others have 16-17%.
* The corner angle is the kink between neighbouring arcs. No constraint sees it; it is reported
  so it can be tracked across the sweep.
* `arcB2` was fitted with `arc_bulge=0` (the fitted fundamental only), as in the precise_QA sweep.
* Provenance (the feasibility solve behind each start, its margin and iteration count) is in
  `starts.json` under `provenance`.
