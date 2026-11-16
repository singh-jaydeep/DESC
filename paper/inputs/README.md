# paper_min/inputs: the starts, near-axis states and bounds the paper's examples use

Copies, not links: these files are tedious to regenerate, and the port should carry them into the new branch unchanged.
`collect.sh` rebuilds this folder from the poster-era locations; `MANIFEST.sha256` pins every file
(`sha256sum -c MANIFEST.sha256`). 270 files, 18 MB. All coil files are DESC h5 (`desc.io.load`); every coilset is a
`CoilSet` (NFP, sym) unless stated otherwise.

## Bounds (`bounds/`, all absolute)

| file | L (m) | κ (1/m) | κ_MS (1/m²) | d_cc (m) | d_pc (m) | used by |
|---|---|---|---|---|---|---|
| `rr02_precise_QA.json` | 4.0418 | 9.8965 | 9.794 | 0.080837 | 0.15157 | faces sweep (RR-02), precise_QA |
| `rr02_precise_QH.json` | 2.9255 | 13.673 | 18.695 | 0.05851 | 0.10971 | faces sweep (RR-02), precise_QH |
| `precise_qh_gil.json` | 2.9255 | 13.673 | 18.695 | 0.05851 | 0.10971 | precise_QH length sweep (`sweep_qh`), × L multipliers 0.8 to 1.6. **Same numbers as `rr02_precise_QH`** |
| `precise_qa_sweep910.json` | 5.1534 | 11.643 | 5.6482 | 0.085546 | 0.17109 | precise_QA length sweep (`sweep_910`), × L multipliers 0.8 to 1.6 |
| `fa_qa2_a0167.json` | 3.9412 | 10.174 | — | 0.04 | 0.1478 | designs B, D, E, F (identical to each design's `bounds.json`), and planar/hinged/free on E/F |
| `bounds_pqa.json` | 4.0540 | 9.8907 | — | 0.041144 | 0.15202 | precise_QA planar/hinged/free control (`fa_qa2_a0167` × 1.0286, precise_QA's a) |
| `qi_jorge2022.json` | 2.9495 | 13.594 | — | 0.04 | 0.11061 | Jorge 2022 QI stage 2 |

Single-experiment variants, kept so earlier numbers can be traced. They are not paper bounds:

| file | change from its parent | experiment |
|---|---|---|
| `precise_qa_sweep910_L0p8.json` | L × 0.8 | precise_QA single stage, arcB2 |
| `precise_qh_gil_L080.json` | L × 0.8 | QH joint-plane single-stage continuation |
| `precise_qh_gil_kappa2x.json`, `precise_qh_gil_nokappa.json` | κ × 2 / κ off | QH planar optimality tests R2, R3 (2026-10-02) |
| `precise_qh_gil_nc4_L125.json`, `precise_qh_gil_nc5_Lfix.json` | L × 1.25 / × 0.8 | 4 vs 5 coils at equal total conductor (2026-10-10) |
| `precise_qa_planar.json` | L 3.2, κ 8, κ_MS 8, d_cc 0.1, d_pc 0.05 | early precise_QA coil validation, three equilibria |

**Three conventions; the paper must state which one each example uses.**
1. **Gil et al., PRE 114, 025202, Table III (a = 1.7 m reactor), scaled by f = a/1.7** (RR-02 and `sweep_qh`): length
   160 m per half period split over the coils (4 here), κ 1/m, κ_MS 0.1/m² (scales as f⁻²), d_cc 0.8 m, d_pc 1.5 m.
   d_cc is 0.47 a.
2. **The `sweep_910` set, scaled by f = R0/1 m = 1.03067**: L 5.0, κ 12, κ_MS 6, d_cc 0.083, d_pc 0.166. It is a
   different set from (1), so precise_QA has two incompatible bound sets: sweep_910 and RR-02.
3. **The near-axis set** (designs; Jorge): the Gil set scaled by the near-axis a (QA: 0.167), with d_cc fixed at 0.04 m
   and no κ_MS. Jorge uses L 23.6 a, κ 1.699/a, d_pc 0.885 a at a = 0.125, also with d_cc 0.04.

The old length sweeps were *told* backed-off bounds (`sweep_qh/fixed_configs.md` §6.1: d_cc × 1.025 plus an arc
sagitta term, d_pc × 1.02, κ × 0.995 and κ_MS × 0.99 on arcs) and *judged* against the files above. The files hold the
judged values. Nothing on the lsq-auglag-composite path backs bounds off.

## Starts (`starts/`)

| folder | example | what | how it was built |
|---|---|---|---|
| `faces/QA_s0`…`QA_s7`, `faces/QH_s0`…`QH_s9` | faces sweep | per seed: `planar1.h5` (FourierPlanarCoil N 8 circles), `modular.h5` (FourierXYZ N 8), `arc2.h5`…`arc8.h5` (PiecewisePlanarArcCoil B = 2..8, M 5, free hinges); `r_over_a.txt`, `hinge_phase.txt` (f, fraction of a face), `SEED.txt` | `faces/build_starts.sh` + `faces/flatten.py` (one process at a time, peak RSS 2.3 GB). Seeds: QA 0-4 r/a 3.2/3.1/3.4/3.3/3.25; QA 5-7 seed 0's circles, f = 1/4, 1/2, 3/4; QH 0-2 r/a 2.5/2.4/2.6; QH 3-9 r/a 2.5, f = 1/4, 1/2, 3/4, 1/8, 3/8, 5/8, 7/8. Every class is built for every seed (hinge-rotated seeds have no `modular.h5`; their `planar1.h5` is seed 0's circles), though RR-02 ran only some (its coverage table). 4 coils per half period |
| `sweep910_QA/` | precise_QA length sweep | 7 cold starts (planarN7, xyzN4/N6, arcB2/3/5/7), each pinned by SHA-256 in `starts.json` (with margins and feasibility), `README.md` | `sweep_910` bundle (`c0/dump_initial_starts.py`) |
| `sweepqh_QH/` | precise_QH length sweep | the same 7 classes on precise_QH, all FEASIBLE | `../DESC2/sweep_qh` bundle; feasibility solves at margin 0.25 |
| `compare_pqa/` | planar/hinged/free, precise_QA control | circles r/a 3.4 (`start_planar`), FourierXYZ N 8 (`start_xyz`), arc B 2 M 5 with hinges at the horizontal extremes (`start_arc`, flattened); all FEASIBLE; 5 coils/hp | `make_start --nc 5 --r-over-a 3.4 --rep planar --N 8`, then `--from-coils` `--rep xyz --N 8` / `--rep arc --B 2 --M 5 --chord-angle 90` |
| `compare_E/` | planar/hinged/free on E (poster) | circles r/a 3.0 on E (`start_r3.0`), on D (`startD_r3.0`), FourierXYZ (`startxyz_r3.0`) | `make_start --nc 5 --N 8 --r-over-a 3.0` |
| `F/` | F stage 2 (direct at fold 60; solver comparison) | `F_z0_c25_p07.h5`: the **z = 0 lens start**, every hinge on z = 0, feasible, unlinked (L 3.24, κ 9.23, d_cc 77 mm, d_pc 248 mm). `F_planar_r3.4`, `F_xyz_r3.4`, `F_arc_r3.4` (arc fit to the circles, chord 90°, fails L and κ_MS; it is the current donor for the lens). `F10_fold_sides.json` is the `--fold-sides-from` file | lens: `make_z0_start.py --eq designF eq --vacuum --bounds designF bounds --currents F_arc_r3.4.h5 --nc 5 --clear 2.5 --gap-frac 0.75 --power 0.7 --chord-z zero --M 5 --n 720`, then flattened (full args in `F_z0_c25_p07_mixed.json`). Circles: `make_start --nc 5 --r-over-a 3.4 --rep planar --N 8`; xyz and arc `--from-coils` (arc `--B 2 --M 5 --chord-angle 90`) |
| `jorge/` | Jorge 2022 QI stage 2 | `J_planar_r3.6.h5` (12 coils, NFP 1), `J_arc90.h5` (B 2, M 5, chord 90°, flattened): the start of the level-hinge fold ladder | `make_start --eq jorge_vac.h5 --nc 12 --r-over-a 3.6 --rep planar --N 8 --vacuum --bounds qi_jorge2022.json`, then `--from-coils … --rep arc --B 2 --M 5 --chord-angle 90`, flattened. Lens starts break L by 8-56% on Jorge, so they are not kept |

Faces-sweep arcs were flattened from make_start's MixedCoilSet; `CoilSetDistanceRows` crashes on a MixedCoilSet, so
always use the flattened files. The `*_mixed.h5` intermediates are not copied.

## Equilibria and near-axis states

| path | what |
|---|---|
| `equilibria/jorge_vac.h5` | Jorge 2022 one-period QI, vacuum (E2; `1012_QI/cloud/jorge_vac.h5`), NFP 1 |
| `near_axis/o2w1_qa2s2_gg1e-1_w1e-2_w1x10.{json,npz}` | **B's** near-axis reference (4 coils/hp, z = 0). `json["args"]` holds the full `run_na2.py` arguments |
| `near_axis/li_qa2_iota0.30.{json,npz}` | B's low-iota (0.30) near-axis state, made from the one above: the input to B's free boundary |
| `near_axis/nc5_s3.{json,npz}` | 5 coils/hp, second order, iota 0.42, **every hinge on z = 0** (`--mode z0 --zmax 0`). This is the natural near-axis start for **rebuilding F with z = 0 from the start** (PORT_PLAN 9.1) |
| `near_axis/nc5_s4.{json,npz}` | D's state: nc5_s3 relaxed to a hinge band (|z| ≤ 0.0334, tilt ≤ 2°), which led to D, then E, then F |

precise_QA and precise_QH come from `desc.examples` and are not copied.

## References (`references/`): the poster-era results the rebuilds are compared against

`designB/`, `designD/`, `designE/`, `designF/` (each `eq.h5`, `coils.h5`, `bounds.json`, from `from_axis/final/*/data`) and
`jorge_C06_level.h5` (the full-machinery level-hinge Jorge coils, B.n 1.31e-2, `1012_QI/package_jorge_qi`).
