# precise_QH boundary × coils matrix

## ⟨|B·n|⟩/⟨B⟩ after run_al.py refinement (max in parentheses); ÷B0 = same column on precise_QH

| boundary ↓ / coils → | planar N7 | arcB2 M5 | FourierXYZ N6 |
|---|---|---|---|
| B0: precise_QH | 2.205e-02 (9.0e-02), ÷B0 1.00 | 5.437e-03 (2.7e-02), ÷B0 1.00 | 9.332e-04 (4.7e-03), ÷B0 1.00 |
| B_planar (ss planar) | 6.866e-03 (3.1e-02), ÷B0 0.31 **(diag)** | 2.335e-03 (1.5e-02), ÷B0 0.43 | 1.097e-03 (1.1e-02), ÷B0 1.18 |
| B_arcB2 (ss arcB2) | 1.687e-02 (6.6e-02), ÷B0 0.76 | 1.484e-03 (6.5e-03), ÷B0 0.27 **(diag)** | 5.171e-04 (2.0e-03), ÷B0 0.55 |
| B_xyz (ss XYZ) | 2.166e-02 (8.7e-02), ÷B0 0.98 | 3.079e-03 (1.5e-02), ÷B0 0.57 | 6.260e-04 (3.0e-03), ÷B0 0.67 **(diag)** |
| ref: old arcB2 ss_k4 | 1.852e-02 (7.8e-02), ÷B0 0.84 | 1.449e-03 (6.3e-03), ÷B0 0.27 | 5.192e-04 (2.1e-03), ÷B0 0.56 |

## Every cell: margins (≤1 for L, κ, κ_MS; ≥1 for d_cc, d_pc), solver

| boundary | coils | B·n | ÷ row XYZ | ÷ col B0 | L | κ | κ_MS | d_cc | d_pc | linked | active | solver | outer/inner | opt | viol | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0: precise_QH | planar N7 | 2.205e-02 | 23.63 | 1.00 | 1.0000 | 1.0008 | 1.000 | 1.039 | 1.005 | 0 | L, kappa, kMS, d_pc | `gtol` condition satisfied. | 11/774 | 6.850e-07 | 7.978e-07 | 833 |
| B0: precise_QH | arcB2 M5 | 5.437e-03 | 5.83 | 1.00 | 1.0000 | 1.0008 | 1.002 | 1.009 | 1.004 | 0 | L, kappa, kMS, d_cc, d_pc | `gtol` condition satisfied. | 12/194 | 5.843e-07 | 3.311e-07 | 367 |
| B0: precise_QH | FourierXYZ N6 | 9.332e-04 | 1.00 | 1.00 | 1.0000 | 1.0000 | 1.000 | 1.022 | 1.185 | 0 | L, kappa, kMS | `gtol` condition satisfied. | 12/214 | 5.427e-07 | 1.365e-07 | 454 |
| B_planar (ss planar) | planar N7 | 6.866e-03 | 6.26 | 0.31 | 1.0000 | 0.9998 | 1.000 | 1.038 | 1.100 | 0 | L, kappa, kMS | `gtol` condition satisfied. | 11/342 | 9.389e-07 | 7.898e-07 | 441 |
| B_planar (ss planar) | arcB2 M5 | 2.335e-03 | 2.13 | 0.43 | 1.0000 | 1.0011 | 0.898 | 1.013 | 1.255 | 0 | L, kappa | `gtol` condition satisfied. | 6/335 | 7.500e-07 | 3.837e-07 | 586 |
| B_planar (ss planar) | FourierXYZ N6 | 1.097e-03 | 1.00 | 1.18 | 1.0000 | 1.0000 | 0.961 | 1.022 | 1.407 | 0 | L, kappa | `gtol` condition satisfied. | 5/17 | 2.862e-07 | 2.192e-07 | 88 |
| B_arcB2 (ss arcB2) | planar N7 | 1.687e-02 | 32.62 | 0.76 | 1.0000 | 1.0019 | 1.000 | 1.039 | 1.005 | 0 | L, kappa, kMS, d_pc | `gtol` condition satisfied. | 7/406 | 9.012e-07 | 2.992e-07 | 496 |
| B_arcB2 (ss arcB2) | arcB2 M5 | 1.484e-03 | 2.87 | 0.27 | 1.0000 | 1.0007 | 0.798 | 1.240 | 1.325 | 0 | L, kappa | `gtol` condition satisfied. | 5/74 | 6.902e-07 | 8.046e-07 | 151 |
| B_arcB2 (ss arcB2) | FourierXYZ N6 | 5.171e-04 | 1.00 | 0.55 | 1.0000 | 0.9109 | 0.681 | 1.022 | 1.387 | 0 | L | `gtol` condition satisfied. | 4/31 | 5.409e-07 | 6.098e-07 | 102 |
| B_xyz (ss XYZ) | planar N7 | 2.166e-02 | 34.60 | 0.98 | 1.0000 | 1.0010 | 1.000 | 1.039 | 1.005 | 0 | L, kappa, kMS, d_pc | `gtol` condition satisfied. | 11/549 | 3.754e-07 | 9.385e-07 | 638 |
| B_xyz (ss XYZ) | arcB2 M5 | 3.079e-03 | 4.92 | 0.57 | 1.0000 | 1.0006 | 1.006 | 1.013 | 1.133 | 0 | L, kappa, kMS | `gtol` condition satisfied. | 8/233 | 6.387e-07 | 6.996e-07 | 421 |
| B_xyz (ss XYZ) | FourierXYZ N6 | 6.260e-04 | 1.00 | 0.67 | 1.0000 | 1.0000 | 1.000 | 1.022 | 1.262 | 0 | L, kappa, kMS | `gtol` condition satisfied. | 12/138 | 6.603e-07 | 6.845e-07 | 338 |
| ref: old arcB2 ss_k4 | planar N7 | 1.852e-02 | 35.67 | 0.84 | 1.0000 | 1.0019 | 1.000 | 1.038 | 1.005 | 0 | L, kappa, kMS, d_pc | `gtol` condition satisfied. | 11/453 | 2.948e-07 | 6.527e-07 | 500 |
| ref: old arcB2 ss_k4 | arcB2 M5 | 1.449e-03 | 2.79 | 0.27 | 1.0000 | 1.0007 | 0.816 | 1.123 | 1.296 | 0 | L, kappa | `gtol` condition satisfied. | 5/72 | 9.502e-07 | 7.942e-08 | 143 |
| ref: old arcB2 ss_k4 | FourierXYZ N6 | 5.192e-04 | 1.00 | 0.56 | 1.0000 | 1.0003 | 0.753 | 1.022 | 1.333 | 0 | L, kappa | `gtol` condition satisfied. | 11/94 | 3.040e-07 | 2.994e-07 | 218 |

## Boundaries (Boozer QS = |B_nonsym|/⟨B⟩; single-stage coils = the k=4 coils before refinement)

| boundary | QS ρ=0.25 | 0.5 | 0.75 | 1 | A | R0 | a | iota | V/V0 | single-stage coils B·n |
|---|---|---|---|---|---|---|---|---|---|---|
| B0: precise_QH | 4.810e-04 | 2.250e-04 | 3.860e-04 | 1.440e-03 | 8.000 | 0.9947 | 0.1243 | [1.2523, 1.2562] | 1.0000 | – |
| B_planar (ss planar) | 1.640e-03 | 3.180e-03 | 5.300e-03 | 1.000e-02 | 7.919 | 0.9813 | 0.1239 | [1.2404, 1.2590] | 0.9799 | 6.953e-03 |
| B_arcB2 (ss arcB2) | 4.400e-04 | 7.180e-04 | 1.690e-03 | 4.780e-03 | 8.176 | 1.0025 | 0.1226 | [1.2400, 1.2516] | 0.9800 | 2.021e-03 |
| B_xyz (ss XYZ) | 1.380e-04 | 1.380e-04 | 3.900e-04 | 1.500e-03 | 8.110 | 0.9970 | 0.1229 | [1.2503, 1.2551] | 0.9800 | 6.648e-04 |
| ref: old arcB2 ss_k4 | 4.380e-04 | 6.740e-04 | 1.490e-03 | 4.540e-03 | 8.165 | 1.0015 | 0.1227 | [1.2435, 1.2569] | 0.9800 | 1.919e-03 |
