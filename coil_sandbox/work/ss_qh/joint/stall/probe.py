"""Probe the single-stage Gauss-Newton model at one state, term by term (K5_STALL.md diagnostics).

Builds exactly the problem single_stage.py builds for step k (same flags), then, instead of optimizing,
evaluates the wrapped proximal objective along two directions from x0:
  gn  the full Gauss-Newton step (scaled as lsqtr scales it: Jacobian column norms)
  sd  steepest descent in the same scaled space
at several step lengths. For each point and each term: the actual cost, the cost the linear model
predicts, and (small steps) the error of the directional derivative J p against the finite difference.
That separates the hypotheses: a wrong Jacobian (FD mismatch at small t), solve noise (mismatch that
does not shrink with t), and hinges switching on (predicted 0, actual > 0 in a coil term).

    python work/ss_qh/joint/stall/probe.py --probe-out DIR <single_stage.py flags, incl. --kmin K>
"""

import json
import os
import sys
import time

SB = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
sys.path.insert(0, SB)


def _pop(flag, default):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        v = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        return v
    return default


PROBE_OUT = _pop("--probe-out", None)
GN_T = [float(v) for v in _pop("--gn-t", "1,0.3,0.1,0.03,0.01,1e-3,1e-4,1e-6").split(",")]
SD_H = [float(v) for v in _pop("--sd-h", "1e-3,1e-4,1e-5,1e-6,1e-8").split(",")]

import single_stage as SS  # noqa: E402  (runs boot.setup)
import numpy as np  # noqa: E402
from scipy.optimize import OptimizeResult  # noqa: E402

import common as S  # noqa: E402
import desc.objectives as O  # noqa: E402
from desc.io import load  # noqa: E402
from desc.optimize import Optimizer  # noqa: E402
from desc.optimize.optimizer import register_optimizer  # noqa: E402

STATE = {}

# PROX_REG=shift (DESC default: sf += sf[-1]) | none (plain pseudo-inverse). Also prints the dF/dx spectrum once.
PROX_REG = os.environ.get("PROX_REG", "shift")


def _patch_proximal():
    import desc.optimize._constraint_wrappers as CW
    from desc.backend import jnp

    def _eq_tangents(constraint, xf, constants, eq_feasible_tangents, dxdcv, op="scaled_error"):
        dim_x_reduced = eq_feasible_tangents.shape[-1]
        tangents = jnp.concatenate([eq_feasible_tangents.T, dxdcv], axis=0)
        J = getattr(constraint, "jvp_" + op)(tangents, xf, constants)
        Fxh, Fc = J[:dim_x_reduced].T, J[dim_x_reduced:].T
        cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
        uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
        import jax
        if "sv" not in STATE and not isinstance(sf, jax.core.Tracer):
            s = np.asarray(sf)
            STATE["sv"] = s
            print(f"dF/dx (reduced) {Fxh.shape}: s max {s[0]:.3e} min {s[-1]:.3e} cond {s[0] / s[-1]:.3e}; "
                  f"#s < 2 s_min {int(np.sum(s < 2 * s[-1]))}, < 10 s_min {int(np.sum(s < 10 * s[-1]))}, "
                  f"< 100 s_min {int(np.sum(s < 100 * s[-1]))}; smallest 8: {np.array2string(s[-8:], precision=3)}",
                  flush=True)
        if PROX_REG == "shift":
            sf = sf + sf[-1]
        sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
        dfdc = vtf.T @ (sfi[:, None] * (uf.T @ Fc))
        return dxdcv - (eq_feasible_tangents @ dfdc).T

    CW._proximal_eq_tangents = _eq_tangents  # not jitted: called once per Jacobian here
    print(f"PROX_REG={PROX_REG}: proximal (dF/dx)^-1 patched", flush=True)


_patch_proximal()


def _patch_solve():
    """Log every equilibrium re-solve of the proximal evaluation (PROBE_SOLVE_LOG=1)."""
    from desc.equilibrium import Equilibrium

    orig = Equilibrium.solve

    def solve(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        res = out[1] if isinstance(out, tuple) else None
        if res is not None:
            print(f"    [solve] nit {int(res.get('nit', -1))} nfev {int(res.get('nfev', -1))} "
                  f"cost {float(res.get('cost', np.nan)):.3e} |g|inf {float(np.max(np.abs(res.get('grad', [np.nan])))):.2e} "
                  f"{str(res.get('message', ''))[:60]!r}", flush=True)
        return out

    Equilibrium.solve = solve


if os.environ.get("PROBE_SOLVE_LOG"):
    _patch_solve()


@register_optimizer(name="probe", description="model probe", scalar=False, equality_constraints=False,
                    inequality_constraints=False, stochastic=False, hessian=False, GPU=True)
def _probe(objective, constraint, x0, method, x_scale, verbose, stoptol, options=None):
    names, dims = STATE["names"], STATE["dims"]
    edges = np.cumsum([0] + dims)
    fun, jac = objective.compute_scaled_error, objective.jac_scaled_error

    def blocks(f):
        return [0.5 * float(np.sum(np.asarray(f[a:b]) ** 2)) for a, b in zip(edges[:-1], edges[1:])]

    t0 = time.time()
    f0 = np.asarray(fun(x0))
    assert f0.size == edges[-1], (f0.size, edges[-1])
    t_f = time.time() - t0
    t0 = time.time()
    J = np.asarray(jac(x0))
    t_j = time.time() - t0
    scale_inv = np.sqrt(np.sum(J**2, axis=0))
    scale_inv = np.where(scale_inv < np.finfo(float).eps * max(J.shape), 1, scale_inv)
    Jh = J / scale_inv
    g = Jh.T @ f0
    U, s, Vt = np.linalg.svd(Jh, full_matrices=False)
    h_gn = -Vt.T @ ((U.T @ f0) / s)
    h_sd = -g / np.linalg.norm(g)
    c0 = blocks(f0)
    try:
        import jax as _jax
        _st = _jax.devices()[0].memory_stats() or {}
        print(f"device peak so far {_st.get('peak_bytes_in_use', 0) / 1e9:.2f} GB", flush=True)
    except Exception:
        pass
    print(f"\nPROBE  f eval {t_f:.1f} s, jac {t_j:.1f} s; x dim {x0.size}, rows {f0.size}; cost {sum(c0):.6g}; "
          f"|g|_inf {np.abs(g).max():.4g}; |h_gn| {np.linalg.norm(h_gn):.4g}; tr_scipy |x/scale| "
          f"{np.linalg.norm(x0 * scale_inv):.4g}", flush=True)
    print(f"singular values of scaled J: max {s[0]:.3g} min {s[-1]:.3g}  cond {s[0] / s[-1]:.3g}; "
          f"count < 1e-8 max: {int(np.sum(s < 1e-8 * s[0]))}", flush=True)
    # per-term gradient share: which terms drive the step
    gterm = [float(np.linalg.norm(Jh[a:b].T @ f0[a:b])) for a, b in zip(edges[:-1], edges[1:])]
    active0 = [int(np.sum(np.abs(f0[a:b]) > 0)) for a, b in zip(edges[:-1], edges[1:])]
    if os.environ.get("SAVE_TRIALS"):
        os.makedirs(os.environ["SAVE_TRIALS"], exist_ok=True)
        objective._objective._eq.save(os.path.join(os.environ["SAVE_TRIALS"], "base.h5"))
    rec = dict(names=names, dims=dims, cost0=c0, grad_norm_term=gterm, active0=active0, sv=s.tolist(),
               g_inf=float(np.abs(g).max()), h_gn_norm=float(np.linalg.norm(h_gn)), points=[])
    print("terms: " + "; ".join(f"{n[:18]} c {c:.3g} |g| {gg:.3g} act {a}/{d}"
                                for n, c, gg, a, d in zip(names, c0, gterm, active0, dims)), flush=True)

    def evaluate(kind, h):
        x = x0 + h / scale_inv
        t0 = time.time()
        f = np.asarray(fun(x))
        dt = time.time() - t0
        lin = f0 + Jh @ h
        ca, cp = blocks(f), blocks(lin)
        dfa = f - f0
        dfp = Jh @ h
        derr = [float(np.linalg.norm(dfa[a:b] - dfp[a:b]) / max(np.linalg.norm(dfp[a:b]), 1e-300))
                for a, b in zip(edges[:-1], edges[1:])]
        act = [int(np.sum(np.abs(f[a:b]) > 0)) for a, b in zip(edges[:-1], edges[1:])]
        pred = sum(c0) - sum(cp)
        actual = sum(c0) - sum(ca)
        # paired decrease (no cancellation in the total): sum (r0 - r)(r0 + r) / 2
        paired = 0.5 * float(np.sum((f0 - f) * (f0 + f)))
        p = dict(kind=kind, hnorm=float(np.linalg.norm(h)), pred=pred, actual=actual, paired=paired,
                 ratio=actual / pred if pred != 0 else np.nan, cost_act=ca, cost_pred=cp, dir_err=derr,
                 active=act, wall=dt)
        rec["points"].append(p)
        save_dir = os.environ.get("SAVE_TRIALS")
        if save_dir:  # the re-solved equilibrium of this trial (the proximal caches its state vector)
            prox = objective._objective
            e = prox._eq.copy()
            e.params_dict = e.unpack_params(np.asarray(prox._allxeq[-1]))
            os.makedirs(save_dir, exist_ok=True)
            fn = os.path.join(save_dir, f"trial_{len(rec['points']) - 1:02d}_{kind.replace(' ', '').replace('=', '')}"
                              f"_h{p['hnorm']:.0e}.h5")
            e.save(fn)
            p["saved"] = fn
        print(f"{kind} |h| {p['hnorm']:.3e}  pred {pred:+.4e}  actual {actual:+.4e}  paired {paired:+.4e}  "
              f"ratio {p['ratio']:+.4f}  ({dt:.0f} s)", flush=True)
        print("    dc_act/dc_pred per term: " + "  ".join(
            f"{n[:10]} {c0[i] - ca[i]:+.2e}/{c0[i] - cp[i]:+.2e} e{derr[i]:.1e}"
            + (f" act{active0[i]}->{act[i]}" if act[i] != active0[i] else "")
            for i, n in enumerate(names)), flush=True)
        if PROBE_OUT:
            json.dump(rec, open(os.path.join(PROBE_OUT, "probe.json"), "w"), indent=1, default=float)

    for t in GN_T:
        evaluate(f"gn t={t:g}", t * h_gn)
    for hn in SD_H:
        evaluate(f"sd", hn * h_sd)
    return OptimizeResult(x=x0, allx=[x0], success=True, message="probe", nit=0, nfev=1, njev=1, cost=sum(c0))


def main():
    a = SS.parse()
    if a.inner_sn:
        from subspace_newton import install_proximal_hook

        STATE["sn"] = install_proximal_hook(k=a.inner_sn, verbose=1)
    os.makedirs(PROBE_OUT or a.out, exist_ok=True)
    eq, kind = S.load_equilibrium(a.eq)
    cs = load(a.start)
    if a.fix_hinge_z is not None:
        for c in S.iter_unique(cs):
            h0 = np.asarray(c.hinges)
            h = h0.reshape(-1, 3).copy()
            h[:, 2] = a.fix_hinge_z
            c.hinges = h.reshape(h0.shape)
    if a.fix_coils:
        a.fix_current_sum = False
    elif a.field_term == "bn":
        a.fix_current_sum = True
    nc = S.n_unique(cs)
    B = S.paper_bounds(eq, nc, a.a, a.bounds)
    P = {k: B[k] for k in S.UPPER + S.LOWER}
    if P["L"] is not None:
        P["L"] *= a.length_mult
    E = SS.enforced(P, a, S.is_piecewise(cs))
    if a.V0 is None:
        a.V0 = float(eq.compute("V")["V"])
    if a.tau_qs is None:
        a.tau_qs = a.tau_qs_rel * SS.start_qs_rms(eq, a)
    assert a.field_term != "bn" or a.bn_norm is not None, "pass --bn-norm (frozen anchors)"
    k = a.kmin
    cons, n_free = SS.proximal_constraints(eq, k, a.fix_current_sum, cs, a.fix_hinge_z, a.horizontal_hinges,
                                           a.lambda_pin, a.force_chunk)
    terms = SS.build_terms(eq, cs, a, E)[0]
    objective = O.ObjectiveFunction(tuple(terms))
    objective.build(verbose=0)
    STATE["names"] = [t.name for t in terms]
    STATE["dims"] = [int(t.dim_f) for t in terms]
    print(f"probe k={k}: {n_free} boundary modes free; terms {list(zip(STATE['names'], STATE['dims']))}", flush=True)
    solve = {"verbose": 0}
    if a.solve_tol is not None:
        solve.update(ftol=a.solve_tol, xtol=a.solve_tol, gtol=a.solve_tol)
    if a.solve_options:
        solve["options"] = json.loads(a.solve_options)
    opts = {"perturb_options": {"order": a.perturb_order, "verbose": 0}, "solve_options": solve}
    things = [eq] if a.fix_coils else [eq, cs]
    Optimizer("proximal-probe").optimize(things, objective=objective, constraints=cons, maxiter=1, verbose=0,
                                         copy=True, options=opts)


if __name__ == "__main__":
    main()
