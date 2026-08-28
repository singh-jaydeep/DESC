"""Separate the two reasons an augmented Lagrangian solve can look unconverged.

`stationarity` in the notebook reports |Pg|/|g|, which answers one question: how much
of the objective gradient is *not* absorbed by the active constraint gradients. That
number conflates two very different situations, and they call for opposite fixes:

  * dual error -- g is absorbed by the active set, but the multiplier estimate `y`
    carried by the solver is not yet the true `lambda*`. The iterate then sits at the
    minimizer of the augmented Lagrangian for the *wrong* multipliers, which is
    displaced O(||c||) from the KKT point. More outer (multiplier) iterations fix it;
    more inner iterations do not, because the inner solve is already at the bottom of
    the subproblem it was handed.

  * primal error -- there is a genuine feasible descent direction. Either it lies in
    the null space of the active set (|Pg|/|g| stays large at every threshold), or the
    active set itself is wrong: a constraint is being held at its bound with a
    wrong-sign multiplier, so releasing it decreases the objective. No amount of
    multiplier updating fixes this; the inner solve has to keep going.

`kkt_split` reports both, plus complementarity, so you can tell which one you have.
"""

import numpy as np

from desc.objectives import ObjectiveFunction


def kkt_split(coils, objective, constraints, tols=(1e-6, 1e-4, 1e-3, 1e-2)):
    """Decompose the non-stationarity at `coils` into primal and dual parts.

    Parameters
    ----------
    coils : CoilSet
        Iterate to test, e.g. the ``new_coils`` returned by the solve.
    objective : _Objective
        The objective actually minimized (``obj1``).
    constraints : list of _Objective
        The objectives passed as auglag constraints (``obj2``..``obj6``).
    tols : tuple of float
        Active-set thresholds. As in ``stationarity``, read the trend: the auglag only
        holds constraints to a slack of order ``ctol``, so a threshold below that
        misses constraints that are effectively binding.

    Returns
    -------
    list of dict, one per threshold, with keys ``tol``, ``n_active``, ``null_resid``,
    ``sign_resid``, ``total_resid``, ``n_wrong_sign``, ``max_compl``.
    """
    of = ObjectiveFunction((objective, *constraints))
    of.build(verbose=0)
    x = of.x(coils)
    n = objective.dim_f

    J = np.asarray(of.jac_scaled(x))
    g = J[:n].T @ np.asarray(of.compute_scaled_error(x))[:n]
    c = np.asarray(of.compute_scaled(x))[n:]
    Jc = J[n:]
    lb, ub = (np.asarray(b)[n:] for b in of.bounds_scaled)
    gn = np.linalg.norm(g)

    out = []
    print(f"|g| = {gn:.4e}")
    print(
        f"{'tol':>8}{'active':>8}{'|Pg|/|g|':>11}{'sign/|g|':>11}"
        f"{'total/|g|':>11}{'wrong sgn':>11}{'max compl':>12}"
    )
    for tol in tols:
        at_ub = np.isfinite(ub) & (c >= ub - tol)
        at_lb = np.isfinite(lb) & (c <= lb + tol)
        act = at_ub | at_lb
        if not act.any():
            out.append(
                dict(
                    tol=tol,
                    n_active=0,
                    null_resid=1.0,
                    sign_resid=0.0,
                    total_resid=1.0,
                    n_wrong_sign=0,
                    max_compl=0.0,
                )
            )
            continue
        A = Jc[act]

        # stationarity is  g + A.T @ lam = 0, with lam >= 0 on rows active at their
        # upper bound and lam <= 0 on rows active at their lower bound
        lam, *_ = np.linalg.lstsq(A.T, -g, rcond=None)
        null_resid = np.linalg.norm(g + A.T @ lam) / gn

        sgn = np.where(at_ub[act], 1.0, -1.0)
        wrong = (lam * sgn) < 0
        # descent that is available purely by releasing wrong-sign constraints
        sign_resid = (
            np.linalg.norm(A[wrong].T @ lam[wrong]) / gn if wrong.any() else 0.0
        )

        # complementarity: |lam_i| * (distance of row i from the bound it is held at)
        marg = np.where(at_ub[act], ub[act] - c[act], c[act] - lb[act])
        max_compl = float(np.max(np.abs(lam) * np.abs(marg)))

        total = float(np.hypot(null_resid, sign_resid))
        out.append(
            dict(
                tol=tol,
                n_active=int(act.sum()),
                null_resid=float(null_resid),
                sign_resid=float(sign_resid),
                total_resid=total,
                n_wrong_sign=int(wrong.sum()),
                max_compl=max_compl,
            )
        )
        print(
            f"{tol:8.0e}{int(act.sum()):8d}{null_resid:11.4f}{sign_resid:11.4f}"
            f"{total:11.4f}{int(wrong.sum()):11d}{max_compl:12.3e}"
        )

    print(
        "\nread: null_resid small + sign_resid small  -> KKT point; only the\n"
        "      termination test is wrong, loosen gtol.\n"
        "      null_resid small + sign_resid large   -> active set is wrong, real\n"
        "      descent from releasing a constraint (primal).\n"
        "      null_resid large at every tol         -> real feasible descent left\n"
        "      (primal); the inner solve is genuinely short."
    )
    return out


def restart_test(coils, opt, obj, constraints, **kwargs):
    """Ground truth: re-solve from `coils` and see whether the cost actually moves.

    A converged iterate is a fixed point. If a fresh solve started from `coils`
    (fresh trust radius, fresh y=0, fresh mu) drops the cost materially, the previous
    run stopped early regardless of what any stationarity metric says.
    """
    kwargs.setdefault("maxiter", 200)
    kwargs.setdefault("verbose", 3)
    kwargs.setdefault("copy", True)
    return opt.optimize(coils, objective=obj, constraints=constraints, **kwargs)
