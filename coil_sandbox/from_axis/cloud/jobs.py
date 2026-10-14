"""The jobs app.py can run. A job = {"what": one line, "steps": [{"cmd": [...], "cwd": repo-relative dir}], "env": {}}.
Placeholders in a cmd: {OUT} this job's directory on the Volume, {ROOT} the Volume root (to read another job's output),
{REPO} the code. Every step must be safe to run again (the wrapper skips finished steps; the scripts resume inside one).
Paths of inputs are relative to the step's cwd and come from the code copy in the image."""

SS = "coil_sandbox/from_axis/ss"
NA = "coil_sandbox/from_axis/na"
CLOUD = "coil_sandbox/from_axis/cloud"
BOUNDS_QA2 = "../../bounds/fa_qa2_a0167.json"
JOBS, GROUPS = {}, {}

# ---- plumbing test: no GPU work, runs in a minute, locally too (python app.py --local smoke)
JOBS["smoke"] = dict(
    what="plumbing: import the shipped DESC, report the device, write into {OUT}; second step proves step-skipping",
    steps=[dict(cmd=["python", "smoke.py", "{OUT}", "a"], cwd=CLOUD), dict(cmd=["python", "smoke.py", "{OUT}", "b"], cwd=CLOUD)])

JOBS["ckpt_test"] = dict(
    what="single_stage.py checkpoint / kill / auto-resume on a tiny basis-6 case (any small GPU)",
    steps=[dict(cmd=["bash", "test_ckpt.sh", "{OUT}"], cwd=CLOUD)])

# ---- acceptance test + benchmark: reproduce a number measured on the laptop before trusting anything new.
# Laptop (RTX 5070 Ti, 16 GB): shell 4 took 55 min incl. compile, shell 8 another 70 min;
# B.n rms 9.1e-4 -> 4.65e-4 -> 4.0e-4; Boozer QS 4.25e-3 / 4.29e-3 / 8.45e-3 / 1.75e-2; iota 0.313-0.356.
JOBS["accept_b14"] = dict(
    what="free-boundary rescore of the best coils (ss7_b14 k=4) at basis 14, shells 4,8; then compare with the laptop",
    steps=[dict(cmd=["python", "fb_rescore.py", "qa2best/ss7_b14/eq_k4.h5", "qa2best/ss7_b14/coils_k4.h5", "{OUT}/fb",
                     "--shells", "4,8"], cwd=SS),
           dict(cmd=["python", "check_accept.py", "{OUT}/fb/score.json", "{OUT}/meta.json"], cwd=CLOUD)])


def screen(run, tag, lmn=14, shells="2,4,6,8,10,14"):
    """Rule-2 score of a near-axis state (na/runs2/<run>) at a = 0.12, then the Jacobian-fold check."""
    JOBS[f"fb_{tag}_b{lmn}"] = dict(
        what=f"free-boundary score of {run} at basis {lmn}, shells {shells} (boundary repaired after every shell; a failed last shell is scored at the previous one)",
        steps=[dict(cmd=["python", "score_fb.py", f"runs2/{run}", "--LMN", str(lmn), "--a", "0.12", "--shells", shells,
                         "--maxiter", "60", "--repair", "--out", "{OUT}/fb"], cwd=NA),
               dict(cmd=["python", "jac.py", "{OUT}/fb/eq_fb.h5"], cwd=SS),
               dict(cmd=["python", "tip.py", "{OUT}/fb/eq_fb.h5", "{OUT}/fb/coils.h5", "{OUT}/fb/tip.png", "--device", "gpu"], cwd=SS)])
    return f"fb_{tag}_b{lmn}"


GROUPS["screen_b14"] = [screen("li_qa2_iota0.30", "li030"), screen("li_qa2_iota0.20", "li020"),
                        screen("lv_qa2_h0.05", "lv005")]
GROUPS["screen_b14_more"] = [screen("lv_qa2_h0.20", "lv020"), screen("li_qa2_iota0.10", "li010"),
                             screen("o2w1_qa2s2_gg1e-1_w1e-2_w1x10", "ref042")]


def single_stage(src, tag, tau_qs_rel, iota_min, a_band=(6.5, 7.5), hinge=("--fix-hinge-z", "0"), kmax=6, lmn=14):
    """Single stage from a screen job's output (its eq_fb.h5 + coils.h5), then the rule-2 rescore of the last k.
    Settings = ss4's (tight B.n, Boozer QS, distance rows), basis 14, a checkpoint and a trial line at every iteration."""
    name = f"ss_{tag}"
    JOBS[name] = dict(
        what=f"single stage from {src} (tau_qs {tau_qs_rel} x start, iota >= {iota_min}), k 1..{kmax}, then free-boundary rescore",
        steps=[
            dict(cmd=["python", "prep.py", "{ROOT}/" + src + "/fb", "{OUT}/start", "--LMN", str(lmn)], cwd=SS),
            dict(cmd=["python", "../../single_stage.py", "--eq", "{OUT}/start/eq0_b" + str(lmn) + ".h5", "--start",
                      "{OUT}/start/coils0.h5", "--out", "{OUT}/ss", "--bounds", BOUNDS_QA2, *hinge,
                      "--dist", "rows", "--rows-select", "1.6", "--cc-rows", str(cc_rows), "--pc-rows", "12000", "--pc-gap", "0.005",
                      "--plasma-m", "40", "--plasma-n", "40", "--helicity", "1,0", "--qs-form", "boozer", "--qs-rho",
                      "0.25,0.5,0.75,1.0", "--A-min", str(a_band[0]), "--A-max", str(a_band[1]), "--iota-min", str(iota_min),
                      "--tau-bn", "5e-4", "--tau-qs-rel", str(tau_qs_rel), "--length-mult", "0.999", "--src-gl", "64",
                      "--force-chunk", "200", "--method", "lsq-composite", "--hessian", "secant", "--solve-tol", "1e-8",
                      "--kmin", "1", "--kmax", str(kmax), "--maxiter", "50", "--track-steps", "--ckpt-every", "1",
                      "--auto-resume"], cwd=SS),
            dict(cmd=["bash", "-c", "k=$(ls {OUT}/ss/eq_k*.h5 | sed 's/.*eq_k\\([0-9]*\\).h5/\\1/' | sort -n | tail -1) && "
                      "python fb_rescore.py {OUT}/ss/eq_k$k.h5 {OUT}/ss/coils_k$k.h5 {OUT}/fb --shells 4,8 --repair"], cwd=SS)])
    return name


# defined for the three screened candidates; run only the screen's winner(s). Two QS weights each.
GROUPS["ss_li030"] = [single_stage("fb_li030_b14", "li030_q10", 0.10, 0.22), single_stage("fb_li030_b14", "li030_q20", 0.20, 0.22)]
GROUPS["ss_li020"] = [single_stage("fb_li020_b14", "li020_q10", 0.10, 0.14), single_stage("fb_li020_b14", "li020_q20", 0.20, 0.14)]
GROUPS["ss_lv005"] = [single_stage("fb_lv005_b14", "lv005_q10", 0.10, 0.30, hinge=("--horizontal-hinges",)),
                      single_stage("fb_lv005_b14", "lv005_q20", 0.20, 0.30, hinge=("--horizontal-hinges",))]


# ---- 2026-10-12: basis 14 dropped as too slow (basis 12 and 14 scores agreed to 1-5% on the reference design).
# Single stage at basis 12 from the LOCAL basis-12 free-boundary state of the iota 0.30 candidate
# (na/runs2/li_qa2_iota0.30_fb_a0.12_L12_rep: shells 2-8, B.n 0.075% rms, QS 0.19 / 0.43 / 0.89 / 1.67%, iota 0.28 -> 0.25),
# with the boundary repaired at the start and after every k step, then the rule-2 rescore with repairs.
def ss_b12(tag, src, tau_qs_rel, iota_min, a_band=(6.6, 7.6), kmax=6, maxiter=50, hinge=("--fix-hinge-z", "0"), cc_rows=12000):
    name = f"ss12_{tag}"
    JOBS[name] = dict(
        what=f"single stage at basis 12 from {src} (tau_qs {tau_qs_rel} x start, iota >= {iota_min}), repairs, then free-boundary rescore",
        steps=[
            dict(cmd=["python", "prep.py", src, "{OUT}/start", "--LMN", "12"], cwd=SS),
            dict(cmd=["python", "../../single_stage.py", "--eq", "{OUT}/start/eq0_b12.h5", "--start", "{OUT}/start/coils0.h5",
                      "--out", "{OUT}/ss", "--bounds", BOUNDS_QA2, *hinge, "--dist", "rows", "--rows-select", "1.6",
                      "--cc-rows", str(cc_rows), "--pc-rows", "12000", "--pc-gap", "0.005", "--plasma-m", "40", "--plasma-n", "40",
                      "--helicity", "1,0", "--qs-form", "boozer", "--qs-rho", "0.25,0.5,0.75,1.0", "--A-min", str(a_band[0]),
                      "--A-max", str(a_band[1]), "--iota-min", str(iota_min), "--tau-bn", "5e-4", "--tau-qs-rel", str(tau_qs_rel),
                      "--length-mult", "0.999", "--src-gl", "64", "--force-chunk", "400", "--method", "lsq-composite",
                      "--hessian", "secant", "--solve-tol", "1e-6", "--kmin", "1", "--kmax", str(kmax), "--maxiter", str(maxiter),
                      "--track-steps", "--ckpt-every", "1", "--auto-resume", "--repair-steps"], cwd=SS),
            dict(cmd=["bash", "-c", "k=$(ls {OUT}/ss/eq_k*.h5 | sed 's/.*eq_k\\([0-9]*\\).h5/\\1/' | sort -n | tail -1) && "
                      "python fb_rescore.py {OUT}/ss/eq_k$k.h5 {OUT}/ss/coils_k$k.h5 {OUT}/fb --shells 4,8 --repair"], cwd=SS),
            dict(cmd=["python", "tip.py", "{OUT}/fb/eq_fb.h5", "{OUT}/ss/coils_k" + str(kmax) + ".h5", "{OUT}/fb/tip.png", "--device", "gpu"], cwd=SS)])
    return name


LI030 = "../na/runs2/li_qa2_iota0.30_fb_a0.12_L12_rep"
ss_b12("li030_q15", LI030, 0.15, 0.22)


# ---- next, agreed 2026-10-12 (after ss12_li030_q15):
# (1) "ss4 redux": the SAME pipeline (basis 12, repairs, same weights) on the iota 0.42 parent that ss4 came from.
#     Separates the two things that changed between ss4 and ss12_li030_q15: a lower-iota parent, and a single stage
#     that converges because the boundary is kept clean. iota floor 0.30 as in ss4.
REF042 = "../na/runs2/o2w1_qa2s2_gg1e-1_w1e-2_w1x10_fb_a0.12_L10"   # Phase 2 free-boundary state (basis 10 -> 12 in prep)
ss_b12("ref042_q15", REF042, 0.15, 0.30, a_band=(6.5, 7.0))
# 2026-10-13: the run above was stopped after k=1 (the old fixed-label repair moved this boundary 2.6-4.2 mm and cost
# 2-3x in QS at the start). Same job under a new name, so it starts fresh with the bdry_repair package (guarded label fit).
ss_b12("ref042_q15r", REF042, 0.15, 0.30, a_band=(6.5, 7.0))

# (2) per-coil level hinges, H = 0.05 (leaves the single joint plane): free-boundary score at basis 12 with repairs,
#     then the single stage with each coil's hinges held level at its own height.
JOBS["fb_lv005_b12"] = dict(
    what="free-boundary score of lv_qa2_h0.05 (level hinges, |z| <= 0.05) at basis 12, shells 2,4,6,8, repairs",
    steps=[dict(cmd=["python", "score_fb.py", "runs2/lv_qa2_h0.05", "--LMN", "12", "--a", "0.12", "--shells", "2,4,6,8",
                     "--maxiter", "60", "--repair", "--out", "{OUT}/fb"], cwd=NA),
           dict(cmd=["python", "tip.py", "{OUT}/fb/eq_fb.h5", "{OUT}/fb/coils.h5", "{OUT}/fb/tip.png", "--device", "gpu"], cwd=SS)])
ss_b12("lv005_q15", "{ROOT}/fb_lv005_b12/fb", 0.15, 0.30, a_band=(6.5, 7.5), hinge=("--horizontal-hinges",))
GROUPS["level_hinges"] = ["fb_lv005_b12", "ss12_lv005_q15"]   # NOT parallel-safe: the second reads the first's output


# ---- 2026-10-13 (evening): rule-2 scores of the two best relaxed near-axis states (RESULTS_2026-10-13.md section 6).
#   nc6_s3: 6 coils per half period, every hinge on z = 0.   nc5_s4: 5 coils, hinges within 0.2 a of z = 0, tilt <= 2 deg.
def score_b12(run, tag):
    JOBS[f"fb_{tag}_b12"] = dict(
        what=f"free-boundary score of {run} at basis 12, shells 2,4,6,8, repairs",
        steps=[dict(cmd=["python", "score_fb.py", f"runs2/{run}", "--LMN", "12", "--a", "0.12", "--shells", "2,4,6,8",
                         "--maxiter", "60", "--repair", "--out", "{OUT}/fb"], cwd=NA),
               dict(cmd=["python", "tip.py", "{OUT}/fb/eq_fb.h5", "{OUT}/fb/coils.h5", "{OUT}/fb/tip.png", "--device", "gpu"], cwd=SS)])
    return f"fb_{tag}_b12"


GROUPS["score_relaxed"] = [score_b12("nc6_s3", "nc6z0"), score_b12("nc5_s4", "nc5lh")]

# ---- 2026-10-13 (overnight): single stage on the 5-coil thin-band state. Start = the Volume's fb_nc5lh_b12 score
# (creased shell-8 boundary; --repair-steps repairs it at the start). Hinges: |z| <= 0.0334 (0.2 a), tilt <= 2 degrees,
# as bounded terms (single_stage.py --hinge-zmax / --hinge-tilt-max, sandbox/demount.HingeBand). iota floor 0.35
# (start 0.38-0.40). The start violates d_cc (0.029 vs 0.04), so the first k steps must also separate the coils.
ss_b12("nc5lh_q15", "{ROOT}/fb_nc5lh_b12/fb", 0.15, 0.35, a_band=(6.8, 7.6),
       hinge=("--hinge-zmax", "0.0334", "--hinge-tilt-max", "2.0"))
# The run above was STOPPED after k=5: with 5 coils 13-16.5 thousand coil-coil rows fall inside the select distance but
# only 12000 slots existed, so the rest (violating ones included) were dropped and d_cc ended at 0.49 of the bound.
# Same job with 24000 slots, under a new name.
ss_b12("nc5lh_q15b", "{ROOT}/fb_nc5lh_b12/fb", 0.15, 0.35, a_band=(6.8, 7.6),
       hinge=("--hinge-zmax", "0.0334", "--hinge-tilt-max", "2.0"), cc_rows=24000)
