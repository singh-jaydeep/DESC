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

**`|Pg|/|g|` is a DIRECTION quality, not a descent magnitude.** It says what fraction of
the gradient survives projection; it says nothing about how FAR you can move before an
inactive constraint stops you. The reading rule "null_resid large at every tol -> real
feasible descent left" is therefore incomplete, and FourierPlanar is the counterexample
(CONVENTIONS 8j): `|Pg|/|g| = 0.89`, yet the nearest inactive row lies `t = 1.5e-05`
along the projected direction, so the achievable descent is `|Pg| * t_max = 4.7e-04`,
i.e. 4.7e-05 relative. Large direction quality, negligible descent.

So this also reports, per threshold:

  * ``t_max``   -- the smallest ``slack_i / |a_i . d|`` over rows NOT in the active set,
                   i.e. how far the projected step travels before something new binds.
                   Linear, so it costs nothing beyond the Jacobian already computed.
  * ``descent`` -- ``|Pg| * t_max``, the first-order achievable decrease. Verified
                   against a direct step test at the planar iterate: predicted 4.66e-04,
                   measured 3.06e-04.
  * ``binds``   -- which constraint block owns the row that sets ``t_max``.

Read `descent` relative to the objective value. Curvature is a second-order correction
on top and is usually negligible when `t_max` is small: at planar,
`(t_max^2 / 2) * kappa` with the measured `kappa ~ 13` is 1.5e-09.
"""

import os as _os, sys as _sys  # noqa: E402
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import _bundle  # noqa: E402,F401  -- MUST precede any `import desc`

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
    # row index -> which constraint block owns it, for naming what binds
    owner = []
    for c in constraints:
        owner += [c.__class__.__name__] * c.dim_f
    owner = np.array(owner)

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
        f"{'t_max':>11}{'descent':>11}{'desc/f0':>11}  binds"
    )
    _fo = ObjectiveFunction((objective,))
    _fo.build(verbose=0)
    f0 = float(_fo.compute_scalar(_fo.x(coils)))
    print(f"  f0 = {f0:.6e}   (descent is ABSOLUTE; desc/f0 is what to compare)")
    slack_all = np.minimum(
        np.where(np.isfinite(ub), ub - c, np.inf),
        np.where(np.isfinite(lb), c - lb, np.inf),
    )
    for tol in tols:
        at_ub = np.isfinite(ub) & (c >= ub - tol)
        at_lb = np.isfinite(lb) & (c <= lb + tol)
        act = at_ub | at_lb
        if not act.any():
            # Nothing active: the projected direction is plain -g. Silence here would
            # read as "no descent available" when the truth is the opposite -- the
            # cold start has no active row and 27% relative descent in front of it.
            d = -g / gn
            resp = np.abs(Jc @ d)
            free = (resp > 1e-14) & np.isfinite(slack_all)
            t_max = (
                float(np.min(np.maximum(slack_all[free], 0.0) / resp[free]))
                if free.any()
                else float("inf")
            )
            out.append(
                dict(
                    tol=tol,
                    n_active=0,
                    null_resid=1.0,
                    sign_resid=0.0,
                    total_resid=1.0,
                    n_wrong_sign=0,
                    max_compl=0.0,
                    t_max=t_max,
                    descent=gn * t_max,
                    descent_rel=gn * t_max / max(abs(f0), 1e-300),
                    binds="(none active)",
                )
            )
            print(
                f"{tol:8.0e}{0:8d}{1.0:11.4f}{0.0:11.4f}{1.0:11.4f}{0:11d}"
                f"{0.0:12.3e}{t_max:11.2e}{gn * t_max:11.2e}"
                f"{gn * t_max / max(abs(f0), 1e-300):11.2e}  (none active)"
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

        # How far can the projected step go before a row NOT in the active set binds?
        # `A d = 0` for active rows by construction, so only inactive rows can stop it.
        r = g + A.T @ lam
        rn = float(np.linalg.norm(r))
        if rn > 0:
            d = -r / rn
            resp = np.abs(Jc @ d)
            free = (~act) & (resp > 1e-14) & np.isfinite(slack_all)
            if free.any():
                t_all = np.maximum(slack_all[free], 0.0) / resp[free]
                i_min = int(np.argmin(t_all))
                t_max = float(t_all[i_min])
                binds = str(owner[np.flatnonzero(free)[i_min]])[:22]
            else:
                t_max, binds = float("inf"), "-"
        else:
            t_max, binds = 0.0, "-"
        descent = rn * t_max
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
                t_max=t_max,
                descent=descent,
                descent_rel=descent / max(abs(f0), 1e-300),
                binds=binds,
            )
        )
        print(
            f"{tol:8.0e}{int(act.sum()):8d}{null_resid:11.4f}{sign_resid:11.4f}"
            f"{total:11.4f}{int(wrong.sum()):11d}{max_compl:12.3e}"
            f"{t_max:11.2e}{descent:11.2e}{descent / max(abs(f0), 1e-300):11.2e}  "
            f"{binds}"
        )

    print(
        "\nread: null_resid small + sign_resid small  -> KKT point; only the\n"
        "      termination test is wrong, loosen gtol.\n"
        "      null_resid small + sign_resid large   -> active set is wrong, real\n"
        "      descent from releasing a constraint (primal).\n"
        "      null_resid large at every tol         -> real feasible descent left\n"
        "      (primal); the inner solve is genuinely short --\n"
        "      BUT ONLY IF `descent` is also large. |Pg|/|g| is a\n"
        "      direction quality; `descent` = |Pg| * t_max is the\n"
        "      magnitude. Large |Pg|/|g| with tiny `descent` means a\n"
        "      near-active row stops the step almost immediately.\n"
        "\n      READ THE ROW AT THE LARGEST tol WITH wrong_sgn == 0. Beyond that the\n"
        "      active set contains rows that are not really active, the multipliers\n"
        "      go wrong-sign, and both the projection and `descent` are meaningless\n"
        "      (v13 shows sign_resid 2149 at tol=1e-3)."
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
