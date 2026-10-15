"""Freeze the drivers of each design's chain: every file is extracted from git AT THE COMMIT ITS STAGE USED, into
designX/drivers/<stage>/ with the repository's relative layout (coil_sandbox/...), plus drivers/MANIFEST.md
(stage, commit, path, md5). The repository is canonical; these copies are for reading and diffing. To rerun a stage,
check out its commit (`git worktree add /tmp/desc_<commit> <commit>`) and run there (REPRODUCE.md).

    python freeze_drivers.py            (from final/)
"""
import hashlib
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=HERE, text=True).strip()
SB = "coil_sandbox/sandbox/"
FA = "coil_sandbox/from_axis/"
NA = [FA + f"na/{f}" for f in ("run_na2.py", "nae.py", "nae2.py", "folded.py", "cont.py")]
FB12 = [FA + "na/score_fb.py", FA + "na/convert.py", FA + "na/folded.py", FA + "na/nae.py", FA + "ss/repair_lib.py",
        FA + "ss/tip.py", SB + "boot.py", SB + "common.py", SB + "convexity.py", SB + "planarity.py"]
SS = ["coil_sandbox/single_stage.py", FA + "ss/prep.py", FA + "ss/fb_rescore.py", FA + "ss/tip.py", FA + "ss/repair_lib.py",
      SB + "boot.py", SB + "common.py", SB + "convexity.py", SB + "planarity.py", SB + "bnormal.py", SB + "demount.py",
      SB + "gauge_fix.py", SB + "qs_scale_invariant.py", SB + "subspace_newton.py", "coil_sandbox/bounds/fa_qa2_a0167.json",
      FA + "cloud/app.py", FA + "cloud/jobs.py", FA + "cloud/requirements.txt", FA + "cloud/watchdog.sh"]
FB14 = [FA + "ss/fb_rescore.py", FA + "ss/tip.py", FA + "ss/repair_lib.py", SB + "boot.py", SB + "common.py",
        FA + "cloud/app.py", FA + "cloud/jobs.py", FA + "cloud/requirements.txt"]
ANALYSIS = [FA + f"ss/{f}" for f in ("iota_trace.py", "interior_check.py", "qs_metrics.py", "qs_booz_fix.py",
                                     "gamma_c.py", "resladder.py", "eps_conv.py", "z0geom.py")]
BDRY = "bdry_repair"  # the package directory under sandbox/, at the stage's commit (if it exists there)

CHAINS = {
    "designD": [
        ("1_near_axis", "4e2a72420", NA + [FA + "na/ncchain.py"],
         "nc5_s1..s4 (run_na2.py identical to 49557cd7c, the commit in use at launch)"),
        ("2_fb_basis12", "da44eb5ec", FB12 + [BDRY], "Modal fb_nc5lh_b12: score_fb.py --LMN 12 --repair"),
        ("3_single_stage", "8aa3ed22f", SS + [BDRY], "Modal ss12_nc5lh_q15b: prep.py + single_stage.py k=1..6"),
        ("4_fb_basis14", "29f1c5e36", FB14 + [BDRY], "Modal fb14_D: fb_rescore.py --LMN 14 --solve-tol 1e-8"),
        ("5_analysis", "835590a09", ANALYSIS, "finalize checks (tracing, interior field, ladders, QS, Gamma_c)"),
    ],
    "designB": [
        ("0_near_axis_reference", "846ece7cb", NA + [FA + f"na/{f}" for f in ("sweep1.sh", "reg1.sh", "reg2.sh",
                                                                               "o2.sh", "w1test.sh")],
         "reference lineage s1 -> o2w1 (ran from uncommitted code; 846ece7cb is the closest commit; s1 has no exact "
         "commit)"),
        ("1_near_axis", "e5b0a9ad2", NA + [FA + "na/lowiota.sh"],
         "li_qa2_iota0.30 (ran uncommitted; e5b0a9ad2 adds only --wplanar, default 0)"),
        ("2_fb_basis12", "a322c7f22", FB12 + [FA + "na/fbscreen.sh"],
         "laptop L12 shells 2,4,6 and the resumed shell 8 (a322c7f22 = the run's working copy as --chunk 200)"),
        ("2b_repair_k6", "89f5a39e4", [FA + "ss/repair_eq.py", FA + "ss/repair_lib.py", SB + "boot.py", SB + "common.py"],
         "old equal-arclength refit of shell 6 between the two L12 runs"),
        ("3_single_stage", "6993369dc", SS, "Modal ss12_li030_q15: prep.py + single_stage.py k=1..6 (old repair_lib)"),
        ("3b_repair", "d5c8966e2", [FA + "ss/repair_eq.py", FA + "ss/repair_lib.py", SB + "boot.py", BDRY],
         "laptop: li030/fb_norepair_local (fb_rescore.py --shells 4,8 at 6993369dc, stage 3) repaired with the new "
         "bdry_repair package -> li030/repaired/eq.h5 (ran uncommitted; committed 5 min later as d5c8966e2)"),
        ("4_basis14_and_polish", "835590a09",
         [FA + "ss/resladder.py", FA + "ss/fb_rescore.py", FA + "ss/repair_lib.py", "coil_sandbox/run_al.py",
          SB + "al_problem.py", SB + "boot.py", SB + "common.py", SB + "demount.py", SB + "convexity.py",
          SB + "planarity.py", "coil_sandbox/check.py", "coil_sandbox/bounds/fa_qa2_a0167.json",
          "coil_sandbox/from_axis/final_work/res/polish_Bk6/bounds_backoff.json",
          "coil_sandbox/from_axis/final_work/res/run_B.sh", "coil_sandbox/from_axis/final_work/res/overnight_B.sh",
          "coil_sandbox/from_axis/final_work/res/replan_B.sh", "coil_sandbox/from_axis/final_work/res/polish_B.sh"],
         "laptop: basis-14 re-solve, free-boundary shell, coil polish (run_al.py), free-boundary shell (code unchanged "
         "from the run times to this commit)"),
        ("5_analysis", "835590a09", ANALYSIS, "finalize checks (tracing, interior field, ladders, QS, Gamma_c)"),
    ],
}


def git_files(commit, path):
    out = subprocess.run(["git", "ls-tree", "-r", "--name-only", commit, path], cwd=ROOT, text=True,
                         capture_output=True).stdout.split()
    return out


for design, stages in CHAINS.items():
    base = os.path.join(HERE, design, "drivers")
    rows = []
    for stage, commit, paths, note in stages:
        files = []
        for p in paths:
            files += git_files(commit, SB + "bdry_repair") if p == BDRY else [p]
        for p in files:
            r = subprocess.run(["git", "show", f"{commit}:{p}"], cwd=ROOT, capture_output=True)
            if r.returncode:
                rows.append((stage, commit, p, "NOT IN THIS COMMIT"))
                continue
            dst = os.path.join(base, stage, p)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            open(dst, "wb").write(r.stdout)
            rows.append((stage, commit, p, hashlib.md5(r.stdout).hexdigest()))
    with open(os.path.join(base, "MANIFEST.md"), "w") as f:
        f.write(f"# Frozen drivers of {design} (written by `../../freeze_drivers.py`)\n\n"
                "Each file is `git show <commit>:<path>` for the commit its stage ran at. The repository is canonical; "
                "to rerun a stage, check out its commit and run there (see `../REPRODUCE.md`).\n\n")
        for stage, commit, _, note in stages:
            f.write(f"* `{stage}` @ `{commit}`: {note}\n")
        f.write("\n| stage | commit | path | md5 |\n|---|---|---|---|\n")
        for r in rows:
            f.write("| " + " | ".join(r) + " |\n")
    print(design, len(rows), "files;", sum(r[3] == "NOT IN THIS COMMIT" for r in rows), "missing")
    for r in rows:
        if r[3] == "NOT IN THIS COMMIT":
            print("   missing:", r[:3])
