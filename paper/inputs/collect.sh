#!/bin/bash
# Copies every coil start, near-axis state, equilibrium and bounds file the paper's examples need into
# paper_min/inputs/, then writes MANIFEST.sha256. Run from anywhere; re-running overwrites the copies.
# Sources are the poster-era locations (coil_sandbox/... and ../DESC2/sweep_qh); README.md says what each file is.
set -eu
CS=$(cd "$(dirname "$0")/../.." && pwd)      # coil_sandbox
OUT=$CS/paper_min/inputs
QH_SWEEP=${QH_SWEEP:-/home/singh/Documents/DESC2/sweep_qh}
cd "$OUT"
rm -rf bounds starts near_axis equilibria references

# ---- bounds ----
mkdir -p bounds
for f in rr02_precise_QA rr02_precise_QH precise_qa_sweep910 precise_qa_sweep910_L0p8 precise_qa_planar \
         precise_qh_gil precise_qh_gil_L080 precise_qh_gil_kappa2x precise_qh_gil_nokappa \
         precise_qh_gil_nc4_L125 precise_qh_gil_nc5_Lfix fa_qa2_a0167 qi_jorge2022; do
  cp "$CS/bounds/$f.json" bounds/
done

# ---- faces sweep (RR-02) starts: flattened CoilSets only (the *_mixed.h5 intermediates are rebuildable by flatten) ----
for cfg in QA QH; do
  for d in "$CS"/from_axis/rr02/starts/${cfg}_s*; do
    s=$(basename "$d"); mkdir -p starts/faces/$s
    for f in planar1.h5 modular.h5 arc2.h5 arc3.h5 arc4.h5 arc5.h5 arc6.h5 arc7.h5 arc8.h5 r_over_a.txt hinge_phase.txt SEED.txt; do
      [ -f "$d/$f" ] && cp "$d/$f" starts/faces/$s/
    done
  done
done
cp "$CS"/from_axis/rr02/{build_starts.sh,flatten.py} starts/faces/

# ---- length sweeps (sweep_910 = precise_QA, sweep_qh = precise_QH): cold starts + their records ----
mkdir -p starts/sweep910_QA starts/sweepqh_QH
cp "$CS"/sweep_910_copy/"initial starts"/{*.h5,starts.json,README.md} starts/sweep910_QA/
cp "$QH_SWEEP"/"initial starts"/{*.h5,starts.json,README.md} starts/sweepqh_QH/

# ---- planar / hinged / free (design E comparison circles; F-eq starts of paper_min) ----
mkdir -p starts/compare_E starts/F
cp "$CS"/from_axis/final/designE/comparison/runs/{start_r3.0.h5,startD_r3.0.h5,startxyz_r3.0.h5} starts/compare_E/
cp "$CS"/paper_min/starts/{F_z0_c25_p07.h5,F_planar_r3.4.h5,F_xyz_r3.4.h5,F_arc_r3.4.h5} starts/F/
cp "$CS"/paper_min/starts/{F_z0_c25_p07,F_planar,F_xyz,F_arc}.log starts/F/
cp "$CS"/work/ss_qh/joint/make_z0_start.py starts/F/
cp "$CS"/paper_min/starts/F_z0_c25_p07_mixed.json starts/F/
cp "$CS"/paper_min/runs/F_foldlad_topk/F10/result.json starts/F/F10_fold_sides.json   # --fold-sides-from

mkdir -p starts/compare_pqa
cp "$CS"/from_axis/planar_cmp/pqa/{start_planar,start_arc,start_xyz}.{h5,json} starts/compare_pqa/
cp "$CS"/from_axis/planar_cmp/pqa/bounds_pqa.json bounds/

# ---- Jorge 2022 QI ----
mkdir -p starts/jorge equilibria
cp "$CS"/paper_min/jorge/starts/{J_planar_r3.6.h5,J_arc90.h5,J_planar.log,J_arc90.log} starts/jorge/
cp "$CS"/1012_QI/cloud/jorge_vac.h5 equilibria/jorge_vac.h5

# ---- QA designs: near-axis states (rebuild inputs) and final designs (comparison targets) ----
mkdir -p near_axis references
for n in o2w1_qa2s2_gg1e-1_w1e-2_w1x10 li_qa2_iota0.30 nc5_s3 nc5_s4; do
  cp "$CS"/from_axis/na/runs2/$n.{json,npz} near_axis/
done
for D in B D E F; do
  mkdir -p references/design$D; cp "$CS"/from_axis/final/design$D/data/* references/design$D/
done
cp "$CS"/1012_QI/package_jorge_qi/coils/C06_on_E2_stage2_level.h5 references/jorge_C06_level.h5

find bounds starts near_axis equilibria references -type f -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256
echo "collected $(wc -l < MANIFEST.sha256) files, $(du -sh . | cut -f1)"
