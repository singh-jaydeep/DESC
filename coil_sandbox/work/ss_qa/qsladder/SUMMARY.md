# tau_QS ladder: precise_QA arcB2 single stage (k=4, M=5, V +-2%, L x0.8; iota >= 0.40 unless the row says otherwise)

| case | <B.n>/<B> | max | QS Boozer rho .25/.5/.75/1 | A | iota min | V/V0 | disp rms / max (mm) | binding |
|---|---|---|---|---|---|---|---|---|
| start: stage 2 M5 | 4.542e-03 | 1.96e-02 | 8.1e-05 / 4.2e-05 / 5.6e-05 / 2.8e-04 | 6.000 | 0.4197 | 1.0000 | 0.0 / 0.0 | L |
| tau_QS x10 | 3.745e-03 | 1.79e-02 | 1.7e-04 / 1.9e-04 / 2.4e-04 / 7.9e-04 | 6.157 | 0.3995 | 0.9800 | 10.8 / 30.7 | L, iota_floor, volume_band |
| tau_QS x30 | 3.385e-03 | 1.39e-02 | 2.4e-04 / 4.3e-04 / 8.0e-04 / 2.0e-03 | 6.165 | 0.3992 | 0.9800 | 13.1 / 35.7 | L, iota_floor, volume_band |
| tau_QS x100 | 2.829e-03 | 1.20e-02 | 5.2e-04 / 1.0e-03 / 2.2e-03 / 4.8e-03 | 6.109 | 0.3993 | 0.9800 | 14.8 / 41.8 | L, iota_floor, volume_band |
| tau_QS x1000 | 1.814e-03 | 7.50e-03 | 5.9e-03 / 7.5e-03 / 1.1e-02 / 1.7e-02 | 6.012 | 0.3995 | 0.9800 | 16.3 / 51.0 | L, iota_floor, volume_band |
| x1000, iota>=0.30 | 1.495e-03 | 6.05e-03 | 6.8e-03 / 9.8e-03 / 1.5e-02 / 2.1e-02 | 5.980 | 0.3553 | 0.9800 | 15.8 / 51.4 | L, volume_band |
| x1000, iota>=0.20 | 1.241e-03 | 5.61e-03 | 6.0e-03 / 1.1e-02 / 1.7e-02 / 2.5e-02 | 5.935 | 0.3126 | 0.9800 | 19.1 / 52.5 | L, volume_band |

Co-location of the top-10% regions (chance = 0.10; p from random (theta, zeta) shifts; Spearman over the surface):

| case | |disp| vs |B.n| | |disp| vs QS | |B.n| vs QS |
|---|---|---|---|
| start: stage 2 M5 | n/a | n/a | 0.03 (p 0.983, rho +0.12) |
| tau_QS x10 | 0.35 (p 0.000, rho +0.24) | 0.25 (p 0.007, rho +0.11) | 0.20 (p 0.003, rho +0.10) |
| tau_QS x30 | 0.32 (p 0.000, rho +0.22) | 0.23 (p 0.087, rho +0.09) | 0.24 (p 0.007, rho +0.25) |
| tau_QS x100 | 0.32 (p 0.000, rho +0.18) | 0.09 (p 0.527, rho -0.19) | 0.12 (p 0.443, rho +0.16) |
| tau_QS x1000 | 0.10 (p 0.493, rho +0.14) | 0.12 (p 0.397, rho +0.13) | 0.12 (p 0.403, rho +0.15) |
| x1000, iota>=0.30 | 0.15 (p 0.280, rho -0.00) | 0.20 (p 0.083, rho +0.06) | 0.11 (p 0.387, rho +0.10) |
| x1000, iota>=0.20 | 0.14 (p 0.133, rho -0.06) | 0.13 (p 0.287, rho +0.02) | 0.10 (p 0.420, rho +0.04) |

Coil margins (value / bound; L, kappa, kMS <= 1, d_cc, d_pc >= 1):

- start: stage 2 M5: L 1.000, kappa 0.678, kMS 0.916, d_cc 1.752, d_pc 1.285
- tau_QS x10: L 1.000, kappa 0.710, kMS 0.919, d_cc 1.847, d_pc 1.452
- tau_QS x30: L 1.000, kappa 0.684, kMS 0.899, d_cc 1.966, d_pc 1.459
- tau_QS x100: L 1.000, kappa 0.656, kMS 0.736, d_cc 1.889, d_pc 1.409
- tau_QS x1000: L 1.000, kappa 0.662, kMS 0.801, d_cc 1.827, d_pc 1.412
- x1000, iota>=0.30: L 1.000, kappa 0.728, kMS 0.922, d_cc 1.862, d_pc 1.430
- x1000, iota>=0.20: L 1.000, kappa 0.648, kMS 0.867, d_cc 1.768, d_pc 1.465

Figures: xsections.png, maps.png, spectrum.png; 3D: view.html (each case on its own boundary).
