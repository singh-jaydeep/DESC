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
                      "--dist", "rows", "--rows-select", "1.6", "--cc-rows", "12000", "--pc-rows", "12000", "--pc-gap", "0.005",
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
