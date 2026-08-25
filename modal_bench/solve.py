"""End-to-end equilibrium solves: master's ``tr_method="qr"`` vs ``"qr-fixed"``.

Every solve runs in a FRESH container (``single_use_containers``). Not optional:
two methods sharing a process also share DESC's and jax's compilation caches, so
whichever runs second skips compilation and looks several times faster.

Both arms run from the same branch install. ``trust_region_step_exact_qr`` here
is byte-identical to master's and the diff to ``tr_subproblems.py`` is pure
addition, so ``tr_method="qr"`` IS master's route -- which controls for every
other difference between the branch and master.

What is measured, beyond wall time and final cost:

* **Where the time goes.** ``lsqtr`` is intercepted so ``fun`` and ``jac`` are
  timed separately, and each alpha subproblem call is timed with an explicit
  ``block_until_ready`` (jax dispatch is async; without the block the timings
  are meaningless). Gives alpha-loop share of the solve directly, rather than
  inferred from a model.
* **Alpha iteration counts.** Two levels: the number of subproblem CALLS, and
  the number of inner Hebden/Reinsch ITERATIONS. The inner loop is a jitted
  ``while_loop``, so Python cannot see it; it is counted by a host callback on
  ``solve_triangular_regularized``, which appears in ``tr_subproblems`` only
  inside alpha-loop bodies, exactly twice per iteration, in every variant.
  Because that callback perturbs the loop, counting runs as a SEPARATE pass --
  timings come from a clean pass with no callbacks.
* **Where peak memory happens.** ``peak_bytes_in_use`` is a monotone high-water
  mark, so reading it either side of every alpha call and every Jacobian
  evaluation says which one actually set the peak.
* **Accept/reject per inner pass**, for the section 1.6 trajectory question.

Perturbation: use ``pert_mode="perturb"`` (the default), which goes through
``eq.perturb``. The other two modes are known-broken and retained only to
reproduce earlier invalid runs -- READ ``PERTURBATION.md`` BEFORE TOUCHING THIS.
"""

import json

from ._bench_core import init_gpu
from .common import GPU, MAX_GPU_CONTAINERS, RESULTS_DIR, app, gpu_image, results


def _instrument(jax, count_alpha_iters):
    """Patch DESC's optimizer entry points. Returns the mutable log dict."""
    import time

    import desc.optimize._desc_wrappers as dw
    import desc.optimize.least_squares as ls
    import desc.optimize.tr_subproblems as trs

    dev = jax.local_devices()[0]
    GB = 1024**3

    def mem():
        st = dev.memory_stats() or {}
        return st.get("peak_bytes_in_use", 0), st.get("bytes_in_use", 0)

    t_origin = time.perf_counter()
    log = {
        "t_origin": t_origin,
        "jac_records": [],
        "fun_records": [],
        "alpha_time_s": 0.0,
        "alpha_calls": 0,
        "alpha_inner_iters": None,
        "fun_time_s": 0.0,
        "fun_calls": 0,
        "jac_time_s": 0.0,
        "jac_calls": 0,
        "alpha_records": [],
        "passes": [],
        "pending": None,
        "peak_set_by": None,
        "peak_GB_seen": 0.0,
        "jac_raised_peak": False,
        "alpha_raised_peak": False,
    }

    def note_peak(p_before, p_after, who):
        if p_after > p_before:
            log["peak_set_by"] = who
            log["peak_GB_seen"] = p_after / GB
            log[f"{who}_raised_peak"] = True

    # --- alpha subproblem: time, and whether it moves the high-water mark ----
    def wrap_sub(fn, name, tr_idx):
        def inner(*a, **kw):
            p0, u0 = mem()
            t0 = time.perf_counter()
            out = fn(*a, **kw)
            jax.block_until_ready(out)  # async dispatch: must block to time
            dt = time.perf_counter() - t0
            p1, u1 = mem()
            note_peak(p0, p1, "alpha")
            log["alpha_time_s"] += dt
            log["alpha_calls"] += 1
            log["alpha_records"].append(
                dict(
                    t_s=dt,
                    t_at=t0 - t_origin,
                    peak_before_GB=p0 / GB,
                    peak_after_GB=p1 / GB,
                    in_use_before_GB=u0 / GB,
                    in_use_after_GB=u1 / GB,
                    raised_peak=bool(p1 > p0),
                    trust_radius_in=float(a[tr_idx]),
                    alpha_out=float(out[2]),
                    hits_boundary=bool(out[1]),
                )
            )
            log["pending"] = dict(
                method=name,
                alpha_out=float(out[2]),
                trust_radius_in=float(a[tr_idx]),
                hits_boundary=bool(out[1]),
            )
            return out

        return inner

    for attr, tr_idx in (
        ("trust_region_step_exact_qr", 3),
        ("trust_region_step_exact_qr_fixed", 3),
        ("trust_region_step_exact_svd", 4),
        ("trust_region_step_exact_cho", 2),
    ):
        setattr(ls, attr, wrap_sub(getattr(ls, attr), attr, tr_idx))

    # --- fun / jac: intercept lsqtr and time what it was handed -------------
    orig_lsqtr = dw.lsqtr

    def lsqtr_wrapped(fun, x0, jac, *a, **kw):
        def tfun(*aa, **kk):
            t0 = time.perf_counter()
            r = fun(*aa, **kk)
            jax.block_until_ready(r)
            log["fun_time_s"] += time.perf_counter() - t0
            log["fun_calls"] += 1
            log["fun_records"].append(
                dict(t_s=time.perf_counter() - t0, t_at=t0 - t_origin)
            )
            return r

        def tjac(*aa, **kk):
            p0, _ = mem()
            t0 = time.perf_counter()
            r = jac(*aa, **kk)
            jax.block_until_ready(r)
            dt = time.perf_counter() - t0
            log["jac_time_s"] += dt
            log["jac_calls"] += 1
            p1, _ = mem()
            note_peak(p0, p1, "jac")
            # jac runs once per accepted outer iteration, so the gaps between
            # successive jac starts ARE the per-iteration wall cadence.
            log["jac_records"].append(
                dict(t_s=dt, t_at=t0 - t_origin, peak_after_GB=p1 / GB)
            )
            return r

        return orig_lsqtr(tfun, x0, tjac, *a, **kw)

    dw.lsqtr = lsqtr_wrapped

    # --- accept/reject per inner pass ---------------------------------------
    orig_utr = ls.update_tr_radius

    def utr(
        trust_radius,
        actual_reduction,
        predicted_reduction,
        step_norm,
        bound_hit,
        *a,
        **kw,
    ):
        new_tr, ratio = orig_utr(
            trust_radius,
            actual_reduction,
            predicted_reduction,
            step_norm,
            bound_hit,
            *a,
            **kw,
        )
        rec = log.get("pending") or {}
        log["pending"] = None
        rec.update(
            actual_reduction=float(actual_reduction),
            predicted_reduction=float(predicted_reduction),
            reduction_ratio=float(ratio),
            step_norm=float(step_norm),
            trust_radius_old=float(trust_radius),
            trust_radius_new=float(new_tr),
            accepted=bool(actual_reduction > 0),  # lsqtr's own accept test
        )
        log["passes"].append(rec)
        return new_tr, ratio

    ls.update_tr_radius = utr

    # --- inner alpha iterations (separate pass; perturbs the loop) ----------
    if count_alpha_iters:
        counter = {"n": 0}

        def _bump():
            counter["n"] += 1

        orig_str = trs.solve_triangular_regularized

        def counted(*a, **kw):
            jax.debug.callback(_bump)
            return orig_str(*a, **kw)

        trs.solve_triangular_regularized = counted
        log["_counter"] = counter

    return log


@app.function(
    image=gpu_image,
    gpu=GPU,
    timeout=21600,
    memory=32768,
    single_use_containers=True,  # cold process: compilation caches leak otherwise
    max_containers=MAX_GPU_CONTAINERS,
    volumes={RESULTS_DIR: results},
    retries=0,
)
def solve_one(cfg: dict):
    """Run one instrumented equilibrium solve."""
    jax = init_gpu(deterministic=cfg.get("deterministic", False))
    import time
    import warnings

    import numpy as np

    import desc
    from desc.examples import get
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )

    name, L, method = cfg["name"], cfg["L"], cfg["tr_method"]
    dev = jax.local_devices()[0]
    GB = 1024**3
    out = dict(
        cfg, device=str(dev.device_kind), jax=jax.__version__, desc=desc.__version__
    )
    try:
        log = _instrument(jax, cfg.get("count_alpha_iters", False))

        eq = get(name)
        eq.change_resolution(L=L, M=L, N=L, L_grid=2 * L, M_grid=2 * L, N_grid=2 * L)
        pert = cfg.get("pert", 0.01)
        mode = cfg.get("pert_mode", "perturb")
        chunk = cfg.get("jac_chunk_size", 1000)
        t_pert = time.perf_counter()
        if pert and mode == "perturb":
            # Perturb the BOUNDARY through eq.perturb, which solves the
            # linearised problem so the interior moves CONSISTENTLY with the new
            # boundary, and weights by (mode number)**2 by default.
            #
            # The alternatives, both measured and both wrong: perturbing the
            # interior spectral coefficients directly (white noise at the scale
            # of R00 swamps high-order modes by 1e3-1e4x), and replacing
            # eq.surface then re-solving (LinearConstraintProjection has to jump
            # the interior onto the new boundary discontinuously). At L=12 a
            # nominal 1% gave a starting cost of 9.4e24 either way, against
            # 1.5e-9 unperturbed, and every solve ended with folded flux
            # surfaces. Through eq.perturb the same 1% converges on gtol in 13
            # iterations to a valid equilibrium.
            rng = np.random.default_rng(cfg.get("seed", 0))
            Rb, Zb = np.asarray(eq.Rb_lmn), np.asarray(eq.Zb_lmn)
            dRb = pert * rng.standard_normal(Rb.size) * np.abs(Rb)
            dZb = pert * rng.standard_normal(Zb.size) * np.abs(Zb)
            eq = eq.perturb(
                deltas={"Rb_lmn": dRb, "Zb_lmn": dZb},
                objective=ObjectiveFunction(ForceBalance(eq), jac_chunk_size=chunk),
                constraints=get_fixed_boundary_constraints(eq),
                order=cfg.get("pert_order", 2),
                verbose=0,
                copy=True,
            )
            out["dRb_rel"] = float(np.linalg.norm(dRb) / np.linalg.norm(Rb))
        elif pert and mode in ("relative", "absolute"):
            # ================================================================
            # BROKEN. Kept ONLY to reproduce the earlier, invalid runs.
            # Both perturb the INTERIOR spectral coefficients independently of
            # the boundary, so the state is not near any equilibrium. Every
            # solve started this way ended with FOLDED FLUX SURFACES
            # (sqrt(g) changing sign) while reporting "terminated
            # successfully". "absolute" additionally ignores mode number, so a
            # nominal 1% hits high-order modes at ~5000x their own amplitude.
            # Measured start cost ~1e24 against 1.5e-9 unperturbed.
            # See PERTURBATION.md. Do not use these for new results.
            # ================================================================
            warnings.warn(
                f"pert_mode={mode!r} is a KNOWN-BROKEN perturbation and produces "
                "equilibria with folded flux surfaces; results are not valid. "
                "See modal_bench/PERTURBATION.md. Use pert_mode='perturb'.",
                stacklevel=2,
            )
            out["INVALID_PERTURBATION"] = mode
            rng = np.random.default_rng(cfg.get("seed", 0))
            if mode == "relative":
                eq.R_lmn = eq.R_lmn * (1 + pert * rng.standard_normal(eq.R_lmn.size))
                eq.Z_lmn = eq.Z_lmn * (1 + pert * rng.standard_normal(eq.Z_lmn.size))
            else:
                eq.R_lmn = (
                    eq.R_lmn
                    + pert * rng.standard_normal(eq.R_lmn.size) * np.abs(eq.R_lmn).max()
                )
                eq.Z_lmn = (
                    eq.Z_lmn
                    + pert * rng.standard_normal(eq.Z_lmn.size) * np.abs(eq.Z_lmn).max()
                )
        elif pert:
            raise ValueError(f"unknown pert_mode {mode}")
        out["perturb_s"] = time.perf_counter() - t_pert

        obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=chunk)
        cons = get_fixed_boundary_constraints(eq)

        options = {"tr_method": method}
        if cfg.get("block") and method == "qr-fixed":
            options["tr_qr_block"] = cfg["block"]

        t_build = time.perf_counter()
        obj.build(verbose=0)
        out["build_s"] = time.perf_counter() - t_build
        # what `jac_chunk_size="auto"` actually resolved to, and out of how many
        out["jac_chunk_resolved"] = getattr(obj, "_jac_chunk_size", None)
        out["obj_dim_f"] = int(obj.dim_f)
        out["obj_dim_x"] = int(obj.dim_x)
        out["peak_after_build_GB"] = (dev.memory_stats() or {}).get(
            "peak_bytes_in_use", 0
        ) / GB

        t0 = time.perf_counter()
        eq_out, res = eq.solve(
            objective=obj,
            constraints=cons,
            ftol=cfg.get("ftol", 1e-10),
            xtol=cfg.get("xtol", 1e-10),
            gtol=cfg.get("gtol", 1e-10),
            maxiter=cfg["maxiter"],
            options=options,
            verbose=0,
            copy=True,
        )
        wall = time.perf_counter() - t0
        st = dev.memory_stats() or {}

        out.update(
            wall_s=wall,
            peak_GB=st.get("peak_bytes_in_use", 0) / GB,
            limit_GB=st.get("bytes_limit", 0) / GB,
            cost=float(res.cost),
            n_iter=int(res.nit),
            nfev=int(getattr(res, "nfev", -1)),
            njev=int(getattr(res, "njev", -1)),
            success=bool(res.success),
            message=str(getattr(res, "message", "")),
            hit_maxiter=int(res.nit) >= cfg["maxiter"],
            optimality=float(getattr(res, "optimality", np.nan)),
            # (a) where the time went
            alpha_time_s=log["alpha_time_s"],
            alpha_frac=log["alpha_time_s"] / wall if wall else None,
            jac_time_s=log["jac_time_s"],
            jac_frac=log["jac_time_s"] / wall if wall else None,
            fun_time_s=log["fun_time_s"],
            fun_frac=log["fun_time_s"] / wall if wall else None,
            other_s=wall - log["alpha_time_s"] - log["jac_time_s"] - log["fun_time_s"],
            # (b) alpha work
            alpha_calls=log["alpha_calls"],
            alpha_time_per_call_ms=(
                1e3 * log["alpha_time_s"] / log["alpha_calls"]
                if log["alpha_calls"]
                else None
            ),
            jac_calls=log["jac_calls"],
            fun_calls=log["fun_calls"],
            # (c) where the peak happened
            peak_set_by=log["peak_set_by"],
            alpha_raised_peak=log["alpha_raised_peak"],
            jac_raised_peak=log["jac_raised_peak"],
            alpha_records=log["alpha_records"],
            passes=log["passes"],
        )
        # The first non-trivial alpha call carries JIT compilation (10-16 s at
        # L=16) and must not be counted as alpha work. Trivial calls (<10 ms) are
        # the p_newton-inside-trust-region branch, which does no factorization.
        ts = sorted(a["t_s"] for a in log["alpha_records"])
        nontrivial = [t for t in ts if t >= 0.010]
        compile_s = nontrivial[-1] if nontrivial else 0.0
        body = nontrivial[:-1]
        alpha_excl = log["alpha_time_s"] - compile_s
        out.update(
            alpha_compile_s=compile_s,
            alpha_time_excl_compile_s=alpha_excl,
            alpha_frac_excl_compile=alpha_excl / wall if wall else None,
            alpha_trivial_calls=len(ts) - len(nontrivial),
            # median over the compile-free body is the reproducible metric:
            # it matched to <1% across independent runs at L=16.
            alpha_median_ms=(1e3 * float(np.median(body)) if body else None),
            alpha_p25_ms=(1e3 * float(np.percentile(body, 25)) if body else None),
            alpha_p75_ms=(1e3 * float(np.percentile(body, 75)) if body else None),
        )
        jr = log["jac_records"]
        if len(jr) >= 3:
            starts = [d["t_at"] for d in jr]
            gaps = [b - a for a, b in zip(starts, starts[1:])]
            # first gap contains JIT compilation of the alpha path; drop it
            steady = gaps[1:] if len(gaps) > 1 else gaps
            out.update(
                startup_s=starts[0] + gaps[0],
                iter_gaps_s=gaps,
                iter_median_s=float(np.median(steady)),
                iter_p25_s=float(np.percentile(steady, 25)),
                iter_p75_s=float(np.percentile(steady, 75)),
                jac_median_s=float(
                    np.median([d["t_s"] for d in jr[1:]] or [jr[0]["t_s"]])
                ),
                jac_compile_s=jr[0]["t_s"],
            )
        if "_counter" in log:
            # two solve_triangular_regularized calls per inner iteration
            out["alpha_inner_iters"] = log["_counter"]["n"] / 2.0
            out["alpha_iters_per_call"] = (
                out["alpha_inner_iters"] / log["alpha_calls"]
                if log["alpha_calls"]
                else None
            )
            out["alpha_ms_per_inner_iter"] = (
                1e3 * alpha_excl / out["alpha_inner_iters"]
                if out["alpha_inner_iters"]
                else None
            )

        x = np.concatenate(
            [
                np.asarray(eq_out.R_lmn),
                np.asarray(eq_out.Z_lmn),
                np.asarray(eq_out.L_lmn),
            ]
        )
        out["x_norm"] = float(np.linalg.norm(x))
        tag = (
            f"{name}_L{L}_{method}_b{cfg.get('block')}"
            f"_seed{cfg.get('seed')}"
            f"_det{int(cfg.get('deterministic', False))}_rep{cfg.get('rep')}"
        )
        np.save(f"{RESULTS_DIR}/x_{tag}.npy", x)
        try:  # full equilibrium, for physics-level comparison later
            eq_out.save(f"{RESULTS_DIR}/eq_{tag}.h5")
            out["eq_file"] = f"eq_{tag}.h5"
        except Exception as e:
            out["eq_save_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        results.commit()
        out["x_file"] = f"x_{tag}.npy"
        out["ok"] = True
    except Exception as e:
        import traceback

        out["ok"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:400]}"
        out["traceback"] = traceback.format_exc()[-1500:]
    return out


def _f(v, spec="7.1f", na="   n/a"):
    """Format a value that may legitimately be absent.

    alpha_median_ms is None when a run made fewer than two non-trivial alpha
    calls, and the cadence fields are absent when there were fewer than three
    Jacobian evaluations. Both happen on runs that stall almost immediately.
    """
    return na if v is None else format(v, spec)


def _report(r):
    """Format one run, flagging it loudly if the perturbation was a broken one."""
    mode = r.get("INVALID_PERTURBATION")
    if mode:
        banner = (
            f"*** INVALID: pert_mode={mode} is known-broken "
            "(see PERTURBATION.md); this row is not a usable result ***"
        )
        return banner + "\n" + _report_body(r)
    return _report_body(r)


def _report_body(r):
    if not r.get("ok"):
        return (
            f"{r['name']:11s} L{r['L']:<3d} {r['tr_method']:9s} "
            f"count={int(r.get('count_alpha_iters', False))}  "
            f"FAILED {str(r.get('error'))[:120]}"
        )
    tag = (
        "counted" if r.get("count_alpha_iters") else f"s{r.get('seed')}r{r.get('rep')}"
    )
    why = str(r.get("message", "")).replace("\n", " ")[:60]
    s = (
        f"{r['name']} L{r['L']} {r['tr_method']} [{tag}]  "
        f"wall={r['wall_s']:.1f}s  it={r['n_iter']}  cost={r['cost']:.6e}  "
        f"opt={r['optimality']:.2e}\n"
        f"    stopped: {why}\n"
        f"    alpha median={_f(r['alpha_median_ms'])} ms "
        f"[p25 {_f(r['alpha_p25_ms'], '.0f')}, "
        f"p75 {_f(r['alpha_p75_ms'], '.0f')}]  "
        f"calls={r['alpha_calls']:3d} (trivial {r['alpha_trivial_calls']})\n"
        f"    alpha {_f(r['alpha_time_excl_compile_s'])}s excl-compile "
        f"({_f((r['alpha_frac_excl_compile'] or 0)*100, '5.1f')}% of wall); "
        f"compile call {r['alpha_compile_s']:.1f}s\n"
        f"    jac   {r['jac_time_s']:7.1f}s ({r['jac_frac']*100:5.1f}%)  "
        f"calls={r['jac_calls']:3d}\n"
        f"    fun   {r['fun_time_s']:7.1f}s ({r['fun_frac']*100:5.1f}%)  "
        f"calls={r['fun_calls']:3d}\n"
        f"    other {r['other_s']:7.1f}s\n"
        f"    peak={r['peak_GB']:.2f} GB (limit {r['limit_GB']:.0f})  "
        f"set_by={r['peak_set_by']}  alpha_raised={r['alpha_raised_peak']}  "
        f"jac_raised={r['jac_raised_peak']}\n"
        f"    jac_chunk={r.get('jac_chunk_resolved')} of dim_x="
        f"{r.get('obj_dim_x')} (dim_f={r.get('obj_dim_f')})"
    )
    if r.get("iter_median_s") is not None:
        s += (
            f"\n    cadence: startup={r['startup_s']:.1f}s  "
            f"per-iter median={r['iter_median_s']:.1f}s "
            f"[p25 {r['iter_p25_s']:.1f}, p75 {r['iter_p75_s']:.1f}]  "
            f"jac median={r['jac_median_s']:.1f}s (compile {r['jac_compile_s']:.1f}s)"
            f"\n    => projected maxiter=N wall ~ "
            f"{r['startup_s']:.0f} + {r['iter_median_s']:.1f}*N s"
        )
    if r.get("alpha_inner_iters") is not None:
        s += (
            f"\n    alpha inner iters={r['alpha_inner_iters']:.0f} total, "
            f"{r['alpha_iters_per_call']:.2f} per call, "
            f"{r.get('alpha_ms_per_inner_iter', float('nan')):.1f} ms/iter"
        )
    return s


@app.local_entrypoint()
def main(
    name: str = "precise_QA",
    resolutions: str = "16",
    methods: str = "qr,qr-fixed",
    maxiter: int = 20,
    block: int = 0,
    reps: int = 3,
    seeds: str = "0",
    jac_chunk: int = 1000,
    counted: bool = True,
    deterministic: bool = False,
    pert: float = 0.01,
    pert_mode: str = "perturb",
    xtol: float = 1e-14,
    gtol: float = 1e-10,
    ftol: float = 1e-14,
    out: str = "modal_bench/solve_a100_80.json",
):
    """Fixed jac_chunk across every arm.

    Never "auto": it resolves from dim_f/dim_x at build time, degenerates to a
    chunk of 1 at L=25, and leaves the whole card budgeted to the Jacobian. A
    fixed value keeps the arms differing in tr_method and nothing else.
    """
    from . import ledger

    cfgs = []
    sd = [int(x) for x in str(seeds).split(",")]
    for res in [int(x) for x in str(resolutions).split(",")]:
        for meth in methods.split(","):
            for seed in sd:
                for rep in range(reps):
                    cfgs.append(
                        dict(
                            name=name,
                            L=res,
                            tr_method=meth,
                            maxiter=maxiter,
                            block=block or None,
                            seed=seed,
                            count_alpha_iters=False,
                            rep=rep,
                            jac_chunk_size=jac_chunk,
                            deterministic=deterministic,
                            pert=pert,
                            pert_mode=pert_mode,
                            xtol=xtol,
                            gtol=gtol,
                            ftol=ftol,
                        )
                    )
            if counted:
                # counts only: the host callbacks perturb the alpha loop, so this
                # pass never contributes a timing.
                cfgs.append(
                    dict(
                        name=name,
                        L=res,
                        tr_method=meth,
                        maxiter=maxiter,
                        block=block or None,
                        seed=0,
                        count_alpha_iters=True,
                        rep="count",
                        jac_chunk_size=jac_chunk,
                        deterministic=deterministic,
                        pert=pert,
                        pert_mode=pert_mode,
                        xtol=xtol,
                        gtol=gtol,
                        ftol=ftol,
                    )
                )

    ledger.open_section(
        f"End-to-end solve: {name} L=M=N={resolutions}, {methods} (A100-80GB)",
        dict(
            maxiter=maxiter,
            block=block or "default",
            methods=methods,
            reps=reps,
            seeds=seeds,
            jac_chunk=jac_chunk or "auto",
            pert=pert,
            pert_mode=pert_mode,
            xtol=xtol,
            gtol=gtol,
            ftol=ftol,
            deterministic=deterministic,
            note="two passes per method: clean timing, plus a counted pass whose "
            "host callbacks perturb the alpha loop (counts only, not timings)",
        ),
    )

    rows = []
    for r in solve_one.map(cfgs, order_outputs=True, return_exceptions=True):
        if isinstance(r, Exception):
            msg = f"container failure: {type(r).__name__}: {str(r)[:200]}"
            print("  " + msg, flush=True)
            ledger.note("  " + msg)
            continue
        rows.append(r)
        text = _report(r)
        print(text, flush=True)
        ledger.note("\n```\n" + text + "\n```")
        with open(out, "w") as fh:
            json.dump(rows, fh, indent=2)
    print(f"\nwrote {out} and modal_bench/ledger.md", flush=True)
