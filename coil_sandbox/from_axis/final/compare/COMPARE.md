# Designs B and D against the literature (same script, same settings)

Equilibria: B and D are vacuum free-boundary states of their coils at basis 14; the references are DESC's example equilibria (fixed boundary, no coils; finite beta where the example has it). Every number below comes from `make_figures.py` (QS in the literature Boozer convention; transport at the `hi` bounce-integral resolution).

| | **B** | **D** | precise_QA | reactor_QA | NCSX | ARIES-CS |
|---|---|---|---|---|---|---|
| aspect ratio | 7.26 | 7.56 | 6.00 | 6.00 | 4.47 | 4.55 |
| NFP | 2 | 2 | 2 | 2 | 3 | 3 |
| iota, rho 0 / .5 / 1 | 0.237 / 0.232 / 0.218 | 0.399 / 0.384 / 0.350 | 0.420 / 0.420 / 0.420 | 0.375 / 0.563 / 0.632 | 0.354 / 0.469 / 0.658 | 0.407 / 0.525 / 0.665 |
| L2 breaking modes %, rho .25/.5/.75/1 | 0.228 / 0.275 / 0.488 / 0.901 | 0.240 / 0.213 / 0.354 / 0.777 | 0.008 / 0.004 / 0.006 / 0.028 | 0.293 / 0.291 / 0.405 / 0.728 | 0.973 / 1.280 / 1.950 / 3.878 | 1.081 / 1.355 / 1.924 / 3.602 |
| largest breaking mode %, rho .25/.5/.75/1 | 0.115 / 0.141 / 0.224 / 0.436 | 0.175 / 0.091 / 0.188 / 0.319 | 0.006 / 0.003 / 0.003 / 0.021 | 0.266 / 0.181 / 0.290 / 0.511 | 0.685 / 0.658 / 1.301 / 2.705 | 0.972 / 1.049 / 1.431 / 2.151 |
| two-term rms(f_C/B^3), rho .25/.5/.75/1 | 1.6e-02 / 2.2e-02 / 3.5e-02 / 5.9e-02 | 1.2e-02 / 1.7e-02 / 2.9e-02 / 5.7e-02 | 2.1e-04 / 9.8e-05 / 1.3e-04 / 5.8e-04 | 4.4e-03 / 4.4e-03 / 6.5e-03 / 1.2e-02 | 4.6e-02 / 5.3e-02 / 7.6e-02 / 1.5e-01 | 2.3e-02 / 2.9e-02 / 5.1e-02 / 1.3e-01 |
| eps_eff, rho .25/.5/.75 | 6.4e-04 / 6.7e-04 / 1.3e-03 | 2.6e-04 / 2.2e-04 / 7.2e-04 | 4.5e-07 / 3.2e-07 / 1.2e-06 | 2.2e-04 / 8.0e-04 / 2.9e-03 | 2.0e-03 / 1.9e-03 / 3.7e-03 | 5.9e-04 / 2.1e-04 / 8.4e-04 |
| Gamma_c (Nemov), rho .25/.5/.75 | 1.3e-02 / 1.0e-02 / 2.2e-02 | 7.5e-03 / 7.6e-03 / 1.5e-02 | 4.0e-05 / 4.8e-06 / 3.0e-06 | 3.1e-03 / 3.4e-03 / 1.7e-02 | 1.7e-02 / 3.2e-02 / 2.4e-02 | 1.6e-02 / 1.5e-02 / 1.5e-02 |
| Gamma_c (Velasco), rho .25/.5/.75 | 2.7e-03 / 6.1e-03 / 1.8e-02 | 8.8e-04 / 3.5e-03 / 1.2e-02 | 5.6e-07 / 2.2e-06 / 3.1e-06 | 2.2e-03 / 2.0e-03 / 8.6e-03 | 4.7e-03 / 3.0e-03 / 1.1e-02 | 4.5e-03 / 4.1e-03 / 5.2e-03 |
| magnetic well (DESC, >0 = well), rho .5 / 1 | -4.8e-03 / -1.6e-02 | -2.7e-03 / -6.0e-03 | -6.4e-03 / -2.6e-02 | +1.4e-02 / +2.1e-02 | +1.6e-02 / +6.7e-02 | +1.4e-02 / +7.0e-02 |
| mirror ratio, rho .5 / 1 | 0.055 / 0.117 | 0.053 / 0.107 | 0.053 / 0.105 | 0.055 / 0.105 | 0.083 / 0.190 | 0.076 / 0.180 |
| max elongation (boundary sections) | 3.29 | 3.65 | 4.19 | 4.19 | 2.80 | 2.49 |
| coils' B.n rms / max | 8.7e-06 / 4.2e-05 | 1.6e-05 / 9.4e-05 | (no coils) | (no coils) | (no coils) | (no coils) |
| coils | 4/hp, FEASIBLE | 5/hp, FEASIBLE |  |  |  |  |
| traced iota vs equilibrium, axis offset | 3e-04, 0.12 mm | 7e-04, 0.04 mm |  |  |  |  |

Figures: `overlay_iota.png`, `overlay_qs.png`, `overlay_transport.png`.
