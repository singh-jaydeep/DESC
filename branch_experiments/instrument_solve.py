"""Instrument a real DESC equilibrium solve to count alpha-loop work.

Answers question (a): per outer lsqtr iteration, how many times is the
trust-region subproblem actually called (i.e. how many passes of the inner
`while actual_reduction <= 0` loop), and how many alpha iterations does each
call take? Also records the true m/n of J_a, which decides whether the
QR-then-SVD(R) prep beats SVD(J).

Monkeypatches (does not modify DESC source):
  desc.optimize.least_squares.trust_region_step_exact_qr
so every call is logged with its trust radius, incoming alpha and the alpha
returned. The alpha-iteration count is recomputed in numpy from the same
(R, z, Delta, alpha0) with the identical safeguarded Hebden iteration, since
the jitted while_loop does not expose its trip count.
"""

import json
import sys

import numpy as np

CALLS = []


def _count_alpha_iters(R, z, trust_radius, initial_alpha, rtol=0.01, max_iter=10):
    """Replicate the loop in tr_subproblems.trust_region_step_exact_qr."""
    R = np.asarray(R, dtype=float)
    z = np.asarray(z, dtype=float)
    n = R.shape[1]
    zp = np.concatenate([z, np.zeros(n)])
    a_hi = np.linalg.norm(R.T @ z) / trust_radius
    a_lo = 0.0
    alpha = float(np.clip(initial_alpha, a_lo, a_hi))
    phi = np.inf
    k = 0
    while abs(phi) > rtol * trust_radius and k < max_iter:
        A = np.vstack([R, np.sqrt(alpha) * np.eye(n)])
        Q, Rtil = np.linalg.qr(A)
        Qtz = Q.T @ zp
        try:
            p = np.linalg.solve(Rtil, -Qtz)
            p_norm = np.linalg.norm(p)
            q = np.linalg.solve(Rtil.T, p)
            q_norm = np.linalg.norm(q)
        except np.linalg.LinAlgError:
            break
        phi = p_norm - trust_radius
        if phi < 0:
            a_hi = alpha
        if phi > 0:
            a_lo = alpha
        alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
        alpha = float(np.clip(alpha, a_lo, a_hi))
        k += 1
    return k


def install(count_iters=True):
    """Wrap the QR subproblem so every call is recorded."""
    import desc.optimize.least_squares as LS

    orig = LS.trust_region_step_exact_qr

    def wrapped(p_newton, z, R, trust_radius, initial_alpha=0.0, **kw):
        out = orig(p_newton, z, R, trust_radius, initial_alpha, **kw)
        p, hits, alpha = out
        rec = dict(
            n=int(R.shape[1]),
            k_rows_R=int(R.shape[0]),
            trust_radius=float(trust_radius),
            alpha_in=float(initial_alpha),
            alpha_out=float(alpha),
            hits_boundary=bool(hits),
            newton_norm=float(np.linalg.norm(np.asarray(p_newton))),
        )
        if count_iters and bool(hits):
            rec["alpha_iters"] = _count_alpha_iters(
                R, z, float(trust_radius), float(initial_alpha)
            )
        else:
            rec["alpha_iters"] = 0
        CALLS.append(rec)
        return out

    LS.trust_region_step_exact_qr = wrapped
    return orig


def group_into_outer_iterations(calls):
    """Split the call log into outer iterations.

    Within one outer lsqtr iteration the Jacobian is fixed, so R is unchanged
    and the trust radius shrinks monotonically across the inner loop's passes.
    A new outer iteration therefore begins when the trust radius does NOT
    shrink relative to the previous call (it was updated after an accepted
    step), or when the Newton-step norm changes (new Jacobian).
    """
    groups = []
    cur = []
    for c in calls:
        if not cur:
            cur = [c]
            continue
        prev = cur[-1]
        new_jac = not np.isclose(
            c["newton_norm"], prev["newton_norm"], rtol=1e-12, atol=0.0
        )
        if new_jac:
            groups.append(cur)
            cur = [c]
        else:
            cur.append(c)
    if cur:
        groups.append(cur)
    return groups


def summarize(calls, label, extra=None):
    groups = group_into_outer_iterations(calls)
    per_outer = [len(g) for g in groups]
    iters = [c["alpha_iters"] for c in calls if c["hits_boundary"]]
    n_bd = sum(1 for c in calls if c["hits_boundary"])
    out = dict(
        label=label,
        n_calls=len(calls),
        n_outer=len(groups),
        calls_per_outer_mean=float(np.mean(per_outer)) if per_outer else 0.0,
        calls_per_outer_hist={
            str(k): int(v)
            for k, v in zip(*np.unique(per_outer, return_counts=True))
        },
        frac_calls_hitting_boundary=(n_bd / len(calls)) if calls else 0.0,
        alpha_iters_mean=float(np.mean(iters)) if iters else 0.0,
        alpha_iters_hist={
            str(k): int(v) for k, v in zip(*np.unique(iters, return_counts=True))
        },
        n=calls[0]["n"] if calls else None,
        m_rows_R=calls[0]["k_rows_R"] if calls else None,
    )
    if extra:
        out.update(extra)
    return out


def run_case(name, L, pert=1e-3, maxiter=30, N=None):
    """Solve one equilibrium from a perturbed start, return the call summary."""
    global CALLS
    CALLS = []
    import numpy as _np

    import desc.optimize.least_squares as LS
    from desc.examples import get
    from desc.objectives import (
        ForceBalance,
        ObjectiveFunction,
        get_fixed_boundary_constraints,
    )

    orig = LS.trust_region_step_exact_qr
    install()

    eq = get(name)
    NN = eq.N if N is None else N
    eq.change_resolution(L=L, M=L, N=NN, L_grid=2 * L, M_grid=2 * L, N_grid=2 * NN)

    rng = _np.random.default_rng(0)
    if pert:
        eq.R_lmn = eq.R_lmn + pert * rng.standard_normal(
            eq.R_lmn.size
        ) * _np.abs(eq.R_lmn).max()
        eq.Z_lmn = eq.Z_lmn + pert * rng.standard_normal(
            eq.Z_lmn.size
        ) * _np.abs(eq.Z_lmn).max()

    obj = ObjectiveFunction(ForceBalance(eq))
    cons = get_fixed_boundary_constraints(eq)
    # capture the true J_a shape from the built, constraint-projected objective
    from desc.optimize._constraint_wrappers import LinearConstraintProjection

    lcp = LinearConstraintProjection(obj, ObjectiveFunction(cons))
    lcp.build(verbose=0)
    m_res, n_dof = lcp.dim_f, lcp.dim_x

    eq.solve(
        objective=obj,
        constraints=cons,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        maxiter=maxiter,
        verbose=1,
    )
    LS.trust_region_step_exact_qr = orig
    return summarize(
        CALLS,
        f"{name}_L{L}_pert{pert:g}",
        extra=dict(
            m_residuals=int(m_res),
            n_dof=int(n_dof),
            aspect_ratio=float(m_res) / float(n_dof),
            pert=pert,
        ),
    )


if __name__ == "__main__":
    import desc

    desc.set_device("cpu")

    cases = json.loads(sys.argv[1]) if len(sys.argv) > 1 else [["DSHAPE", 6, 1e-3]]
    out = []
    for spec in cases:
        nm, L, pert = spec[0], spec[1], spec[2]
        N = spec[3] if len(spec) > 3 else None
        try:
            s = run_case(nm, L, pert=pert, N=N)
            out.append(s)
            print(
                f"[{s['label']}] m/n={s['aspect_ratio']:.2f} ({s['m_residuals']}"
                f"/{s['n_dof']}) outer={s['n_outer']} calls={s['n_calls']} "
                f"calls/outer={s['calls_per_outer_mean']:.2f} "
                f"bd_frac={s['frac_calls_hitting_boundary']:.2f} "
                f"alpha_iters={s['alpha_iters_hist']}",
                flush=True,
            )
        except Exception as e:  # keep going across the sweep
            print(f"[{nm}_L{L}] FAILED: {type(e).__name__}: {e}", flush=True)
    json.dump(out, open("solve_summaries.json", "w"), indent=2)
