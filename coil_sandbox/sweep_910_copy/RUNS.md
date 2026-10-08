# Sweep run list

42 runs — 7 representations x 5 length bounds, plus one cold control per representation at x1.4.

Length bounds are relative to the paper value **5.153350 m**. `start` is either a cold build or the run whose solved coilset is transplanted onto it.

| run | representation | DOF/coil | start | xL | L (m) |
|---|---|---|---|---|---|
| `sw_planarN7_L1` | planarN7 | 21 | cold | 1 | 5.1533 |
| `sw_planarN7_L0p8` | planarN7 | 21 | ← `sw_planarN7_L1` | 0.8 | 4.1227 |
| `sw_planarN7_L1p2` | planarN7 | 21 | ← `sw_planarN7_L1` | 1.2 | 6.1840 |
| `sw_planarN7_L1p4` | planarN7 | 21 | ← `sw_planarN7_L1p2` | 1.4 | 7.2147 |
| `sw_planarN7_L1p4_c` | planarN7 | 21 | cold | 1.4 | 7.2147 |
| `sw_planarN7_L1p6` | planarN7 | 21 | ← `sw_planarN7_L1p4_c` | 1.6 | 8.2454 |
| `sw_arcB2_L1` | arcB2 | 14 | cold | 1 | 5.1533 |
| `sw_arcB2_L0p8` | arcB2 | 14 | ← `sw_arcB2_L1` | 0.8 | 4.1227 |
| `sw_arcB2_L1p2` | arcB2 | 14 | ← `sw_arcB2_L1` | 1.2 | 6.1840 |
| `sw_arcB2_L1p4` | arcB2 | 14 | ← `sw_arcB2_L1p2` | 1.4 | 7.2147 |
| `sw_arcB2_L1p4_c` | arcB2 | 14 | cold | 1.4 | 7.2147 |
| `sw_arcB2_L1p6` | arcB2 | 14 | ← `sw_arcB2_L1p4_c` | 1.6 | 8.2454 |
| `sw_arcB3_L1` | arcB3 | 21 | cold | 1 | 5.1533 |
| `sw_arcB3_L0p8` | arcB3 | 21 | ← `sw_arcB3_L1` | 0.8 | 4.1227 |
| `sw_arcB3_L1p2` | arcB3 | 21 | ← `sw_arcB3_L1` | 1.2 | 6.1840 |
| `sw_arcB3_L1p4` | arcB3 | 21 | ← `sw_arcB3_L1p2` | 1.4 | 7.2147 |
| `sw_arcB3_L1p4_c` | arcB3 | 21 | cold | 1.4 | 7.2147 |
| `sw_arcB3_L1p6` | arcB3 | 21 | ← `sw_arcB3_L1p4_c` | 1.6 | 8.2454 |
| `sw_arcB5_L1` | arcB5 | 35 | cold | 1 | 5.1533 |
| `sw_arcB5_L0p8` | arcB5 | 35 | ← `sw_arcB5_L1` | 0.8 | 4.1227 |
| `sw_arcB5_L1p2` | arcB5 | 35 | ← `sw_arcB5_L1` | 1.2 | 6.1840 |
| `sw_arcB5_L1p4` | arcB5 | 35 | ← `sw_arcB5_L1p2` | 1.4 | 7.2147 |
| `sw_arcB5_L1p4_c` | arcB5 | 35 | cold | 1.4 | 7.2147 |
| `sw_arcB5_L1p6` | arcB5 | 35 | ← `sw_arcB5_L1p4_c` | 1.6 | 8.2454 |
| `sw_arcB7_L1` | arcB7 | 49 | cold | 1 | 5.1533 |
| `sw_arcB7_L0p8` | arcB7 | 49 | ← `sw_arcB7_L1` | 0.8 | 4.1227 |
| `sw_arcB7_L1p2` | arcB7 | 49 | ← `sw_arcB7_L1` | 1.2 | 6.1840 |
| `sw_arcB7_L1p4` | arcB7 | 49 | ← `sw_arcB7_L1p2` | 1.4 | 7.2147 |
| `sw_arcB7_L1p4_c` | arcB7 | 49 | cold | 1.4 | 7.2147 |
| `sw_arcB7_L1p6` | arcB7 | 49 | ← `sw_arcB7_L1p4_c` | 1.6 | 8.2454 |
| `sw_xyzN4_L1` | xyzN4 | 27 | cold | 1 | 5.1533 |
| `sw_xyzN4_L0p8` | xyzN4 | 27 | ← `sw_xyzN4_L1` | 0.8 | 4.1227 |
| `sw_xyzN4_L1p2` | xyzN4 | 27 | ← `sw_xyzN4_L1` | 1.2 | 6.1840 |
| `sw_xyzN4_L1p4` | xyzN4 | 27 | ← `sw_xyzN4_L1p2` | 1.4 | 7.2147 |
| `sw_xyzN4_L1p4_c` | xyzN4 | 27 | cold | 1.4 | 7.2147 |
| `sw_xyzN4_L1p6` | xyzN4 | 27 | ← `sw_xyzN4_L1p4_c` | 1.6 | 8.2454 |
| `sw_xyzN6_L1` | xyzN6 | 39 | cold | 1 | 5.1533 |
| `sw_xyzN6_L0p8` | xyzN6 | 39 | ← `sw_xyzN6_L1` | 0.8 | 4.1227 |
| `sw_xyzN6_L1p2` | xyzN6 | 39 | ← `sw_xyzN6_L1` | 1.2 | 6.1840 |
| `sw_xyzN6_L1p4` | xyzN6 | 39 | ← `sw_xyzN6_L1p2` | 1.4 | 7.2147 |
| `sw_xyzN6_L1p4_c` | xyzN6 | 39 | cold | 1.4 | 7.2147 |
| `sw_xyzN6_L1p6` | xyzN6 | 39 | ← `sw_xyzN6_L1p4_c` | 1.6 | 8.2454 |
