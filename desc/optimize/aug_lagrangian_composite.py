"""Augmented Lagrangian with slack-free hinge rows and composite inner solves."""

import numpy as np
from scipy.optimize import OptimizeResult

from desc.backend import jnp
from desc.utils import errorif, setdefault

from .composite import composite_resid
from .least_squares_composite import lsq_composite
from .utils import STATUS_MESSAGES

# constraint row ids are offset past every objective row id
_CON_ID_OFFSET = 1 << 40


class _RowValues:
    """Values keyed by row id, with a default for ids never set.

    With ``floor=True`` the default is also a lower bound on every value, so raising
    it raises every row below it.
    """

    def __init__(self, default, floor=False):
        self.default = default
        self.floor = floor
        self.keys = np.zeros(0, dtype=np.int64)
        self.vals = np.zeros(0)

    def get(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if self.keys.size == 0:
            return np.full(ids.shape, self.default, dtype=float)
        pos = np.clip(np.searchsorted(self.keys, ids), 0, self.keys.size - 1)
        hit = (self.keys[pos] == ids) & (ids >= 0)
        vals = np.where(hit, self.vals[pos], self.default)
        return np.maximum(vals, self.default) if self.floor else vals

    def set(self, ids, vals):
        """Set values for ids >= 0; entries equal to the default are dropped."""
        ids = np.asarray(ids, dtype=np.int64)
        vals = np.broadcast_to(np.asarray(vals, dtype=float), ids.shape)
        keep = ~np.isin(self.keys, ids)
        new = ids >= 0
        keys = np.concatenate([self.keys[keep], ids[new]])
        vals = np.concatenate([self.vals[keep], vals[new]])
        order = np.argsort(keys)
        keys, vals = keys[order], vals[order]
        nondefault = (vals > self.default) if self.floor else (vals != self.default)
        self.keys, self.vals = keys[nondefault], vals[nondefault]


def lsq_auglag_composite(  # noqa: C901
    fun,
    x0,
    jac,
    lo,
    hi,
    tgt,
    isb,
    con,
    con_jac,
    lb,
    ub,
    row_ids=None,
    con_row_ids=None,
    keep_rows=None,
    args=(),
    x_scale="jac",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    ctol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options=None,
):
    """Minimize a least squares objective subject to ``lb <= con(x) <= ub``.

    Slack-free augmented Lagrangian: a constraint row ``c`` with multiplier ``y`` and
    penalty ``mu > 0`` enters each subproblem as the row
    ``sqrt(mu) * (max(0, c + y/mu - ub) - max(0, lb - c - y/mu))``, which is
    ``sqrt(mu) * (c + y/mu - t)`` for an equality ``lb = ub = t``. The subproblem
    rows are bound rows of the composite model, so each subproblem is solved by
    ``lsq_composite`` with the hinges exact. After a subproblem, rows violated by less
    than the current constraint tolerance get the first order update
    ``y <- mu * (max(0, c + y/mu - ub) - max(0, lb - c - y/mu))`` and the others a
    larger penalty, with the tolerance schedule of Conn, Gould and Toint, algorithm
    14.4.2 (with our ``mu`` their ``mu^-1``). The schedule uses the mean penalty over
    the active rows (nonzero multiplier or violated), and rows not seen before start
    at that level, so a contact that slides onto new rows keeps its penalty.

    Rows may change identity between evaluations (objectives that re-select their
    rows): multipliers and penalties are then kept per row id, from ``con_row_ids``,
    and ``keep_rows`` keeps rows with a nonzero multiplier selected.

    Parameters
    ----------
    fun, x0, jac, lo, hi, tgt, isb : callable, array-like
        Objective, as in ``lsq_composite``: pre-hinge row values, its Jacobian, and
        per-row bounds, targets and bound-row flags.
    con, con_jac : callable
        Constraint values ``con(x, *args) -> 1d array`` and their Jacobian.
    lb, ub : array-like
        Bounds on the constraint rows, equal for equality rows.
    row_ids, con_row_ids : callable, optional
        Row ids of the objective and constraint rows, ``row_ids(x, *args)``, -1 for
        unused rows. Positional if not given.
    keep_rows : callable, optional
        ``keep_rows(ids)`` with constraint row ids whose multiplier is nonzero, called
        after each multiplier update.
    args : tuple
        Additional arguments passed to every callable.
    x_scale : array_like or ``'jac'``, optional
        As in ``lsq_composite``.
    ftol, xtol : float or None, optional
        Passed to the subproblem solves.
    gtol : float or None, optional
        Stop when the max norm of the scaled gradient of the subproblem merit (the
        Lagrangian gradient with the updated multipliers) is below ``gtol`` and the
        constraints hold to ``ctol``.
    ctol : float, optional
        Tolerance on the max constraint violation, in the units of ``con``.
    verbose : {0, 1, 2}, optional
        * 0 : work silently.
        * 1 (default) : display a termination report and one line per outer iteration.
        * 2 : display progress during the subproblem solves too.
    maxiter : int, optional
        Maximum number of subproblem iterations in total. Defaults to
        ``size(x) * 100``.
    callback : callable, optional
        Called after each accepted step as ``callback(xk, *args)``; returning True
        stops the optimization.
    options : dict, optional
        Optional keyword arguments.

        - ``"initial_penalty_parameter"`` : (float) Initial ``mu`` for every row.
          Default 10.
        - ``"initial_multipliers"`` : (array) Initial ``y`` for the rows of ``con`` at
          ``x0``. Default 0.
        - ``"omega"``, ``"eta"``, ``"alpha_omega"``, ``"beta_omega"``,
          ``"alpha_eta"``, ``"beta_eta"``, ``"tau"`` : Tolerance schedule and penalty
          increase factor of algorithm 14.4.2, as in ``lsq_auglag``. Defaults
          ``min(g, 1e-2)`` and ``min(c, 1e-2)`` with the scaled gradient ``g`` and
          violation ``c`` at ``x0`` (``1e-2`` where zero), 1, 1, 0.1, 0.9, 10.
        - ``"min_inner_gtol"`` : (float) Floor on the subproblem gradient tolerance,
          near what a subproblem solve can reach. Default ``gtol``.
        - ``"max_stalls"`` : (int) Consecutive subproblems stalled before reaching
          their tolerance after which the optimization stops. A stalled subproblem
          raises the penalty on the violated rows, or if none is violated beyond the
          current tolerance updates the multipliers. Default 3.
        - ``"max_multiplier"``, ``"max_penalty_parameter"`` : (float) Caps on
          ``|y|`` and ``mu``. Default ``np.inf``.
        - ``"outer_callback"`` : (callable) Called after each outer iteration with
          its ``outer_history`` entry, which also holds ``x``, and ``y_next`` and
          ``mu_next``, the multipliers and penalties for the next subproblem.
        - Anything else is passed to the subproblem solves (see ``lsq_composite``);
          default ``{"hessian": "secant", "finish_steps": 5}``.

    Returns
    -------
    res : OptimizeResult
        The optimization result. Besides the usual fields: ``y`` and ``penalty_param``,
        the multipliers and penalties of the constraint rows at ``x``, ``con_ids`` their
        ids, ``constr_violation``, and ``outer_history``, one dict per outer
        iteration.

    """
    options = {} if options is None else dict(options)
    mu0 = options.pop("initial_penalty_parameter", 10.0)
    y0 = options.pop("initial_multipliers", None)
    omega = options.pop("omega", None)
    eta = options.pop("eta", None)
    alpha_omega = options.pop("alpha_omega", 1.0)
    beta_omega = options.pop("beta_omega", 1.0)
    alpha_eta = options.pop("alpha_eta", 0.1)
    beta_eta = options.pop("beta_eta", 0.9)
    tau = options.pop("tau", 10.0)
    min_inner_gtol = options.pop("min_inner_gtol", None)
    max_stalls = options.pop("max_stalls", 3)
    max_multiplier = options.pop("max_multiplier", np.inf)
    max_penalty = options.pop("max_penalty_parameter", np.inf)
    outer_callback = options.pop("outer_callback", None)
    inner_options = {"hessian": "secant", "finish_steps": 5}
    inner_options.update(options)
    errorif(
        "initial_jac" in inner_options,
        ValueError,
        "initial_jac is managed by the augmented Lagrangian",
    )
    gtol, ctol = setdefault(gtol, 0.0), setdefault(ctol, 0.0)
    min_inner_gtol = setdefault(min_inner_gtol, gtol)
    maxiter = setdefault(maxiter, x0.size * 100)
    callback = setdefault(callback, lambda *args: False)

    isb_f = np.asarray(isb, dtype=bool)
    m_f = isb_f.size
    lo_f, hi_f, tgt_f = (
        np.broadcast_to(np.asarray(v, dtype=float), (m_f,)) for v in (lo, hi, tgt)
    )
    c0 = np.asarray(con(x0, *args))
    m_c = c0.size
    lb, ub = (np.broadcast_to(np.asarray(v, dtype=float), (m_c,)) for v in (lb, ub))
    eq = lb == ub
    isb_all = np.concatenate([isb_f, ~eq])
    pos_f, pos_c = np.arange(m_f), np.arange(m_c)

    def fun_all(x, *a):
        return jnp.concatenate([fun(x, *a), con(x, *a)])

    def jac_all(x, *a):
        return jnp.vstack([jac(x, *a), con_jac(x, *a)])

    def ids_f(x, *a):
        return pos_f if row_ids is None else np.asarray(row_ids(x, *a))

    def ids_c(x, *a):
        return pos_c if con_row_ids is None else np.asarray(con_row_ids(x, *a))

    def ids_all(x, *a):
        ic = ids_c(x, *a)
        return jnp.asarray(
            np.concatenate([ids_f(x, *a), np.where(ic >= 0, ic + _CON_ID_OFFSET, -1)])
        )

    Y, MU = _RowValues(0.0), _RowValues(float(mu0), floor=True)
    if y0 is not None:
        Y.set(ids_c(x0, *args), np.broadcast_to(y0, (m_c,)))

    def rows(ids):
        ic = np.asarray(ids)[m_f:]
        ic = np.where(ic >= 0, ic - _CON_ID_OFFSET, -1)
        y, mu = Y.get(ic), MU.get(ic)
        shift = y / mu
        lo_c, hi_c = lb - shift, ub - shift
        return (
            jnp.asarray(np.concatenate([lo_f, lo_c])),
            jnp.asarray(np.concatenate([hi_f, hi_c])),
            jnp.asarray(np.concatenate([tgt_f, lb - shift])),
            jnp.asarray(np.concatenate([np.ones(m_f), np.sqrt(mu)])),
        )

    def violation(c):
        return np.abs(np.asarray(composite_resid(c, lb, ub, lb, ~eq)))

    def mean_mu(ic, viol):
        """Mean penalty over the active rows: with a multiplier or violated."""
        active = (ic >= 0) & ((Y.get(ic) != 0) | (viol > 0))
        return float(np.mean(MU.get(ic[active]))) if np.any(active) else MU.default

    def raise_new_rows(ic, viol):
        # rows not yet seen (a contact sliding to new candidates) start at this level
        MU.default = max(MU.default, mean_mu(ic, viol))

    x = jnp.asarray(x0)
    ic = ids_c(x, *args)
    viol = violation(c0)
    constr_violation = float(viol.max(initial=0.0))
    # notation following Conn & Gould, algorithm 14.4.2, with our mu = their mu^-1
    probe = lsq_composite(
        fun_all,
        x,
        jac_all,
        None,
        None,
        None,
        isb_all,
        row_ids=ids_all,
        rows=rows,
        args=args,
        x_scale=x_scale,
        verbose=0,
        maxiter=0,
        options={"initial_jac": None},
    )
    g_norm, J = float(probe.optimality), probe.jac
    # a zero start value would pin the tolerance to gtol or ctol from the outset
    omega = setdefault(omega, min(g_norm, 1e-2) if g_norm > 0 else 1e-2)
    eta = setdefault(eta, min(constr_violation, 1e-2) if constr_violation > 0 else 1e-2)
    gtolk = max(omega / mean_mu(ic, viol) ** alpha_omega, gtol, min_inner_gtol)
    ctolk = max(eta / mean_mu(ic, viol) ** alpha_eta, ctol)

    iteration, nfev, njev, nstalls, outer = 0, 1, probe.njev, 0, 0
    allx, history = [x], []
    success, message = None, None
    if verbose > 0:
        print(
            f"{'Outer':>5} {'Inner its':>9} {'Optimality':>11} {'gtolk':>9} "
            f"{'Violation':>10} {'ctolk':>9} {'mu mean':>9} {'max |y|':>9}"
        )

    while success is None:
        if g_norm < gtol and constr_violation < ctol:
            success, message = True, STATUS_MESSAGES["gtol"]
            break
        if iteration >= maxiter:
            success, message = False, STATUS_MESSAGES["maxiter"]
            break
        res = lsq_composite(
            fun_all,
            x,
            jac_all,
            None,
            None,
            None,
            isb_all,
            row_ids=ids_all,
            rows=rows,
            args=args,
            x_scale=x_scale,
            ftol=ftol,
            xtol=xtol,
            gtol=gtolk,
            verbose=max(verbose - 1, 0),
            maxiter=maxiter - iteration,
            callback=callback,
            options={**inner_options, "initial_jac": J},
        )
        outer += 1
        iteration += res.nit
        nfev += res.nfev
        njev += res.njev
        allx += res.allx[1:]
        x, J = res.x, res.jac
        g_norm = float(res.optimality)
        c = np.asarray(res.s[m_f:])
        r_c = np.asarray(res.fun[m_f:])
        ic = np.asarray(res.ids[m_f:])
        ic = np.where(ic >= 0, ic - _CON_ID_OFFSET, -1)
        viol = violation(c)
        constr_violation = float(viol.max(initial=0.0))
        mu = MU.get(ic)
        converged = res.success or g_norm < gtolk
        stopped = res.message == STATUS_MESSAGES["callback"]
        history.append(
            dict(
                outer=outer,
                inner_iterations=res.nit,
                optimality=g_norm,
                gtolk=gtolk,
                inner_converged=bool(converged),
                constr_violation=constr_violation,
                ctolk=ctolk,
                ids=ic,
                y=Y.get(ic),
                mu=mu,
                violation=viol,
                message=res.message,
                step_log=res.get("step_log", []),
            )
        )

        def report():
            history[-1].update(y_next=Y.get(ic), mu_next=MU.get(ic))
            if outer_callback is not None:
                outer_callback({**history[-1], "x": x})

        if verbose > 0:
            print(
                f"{outer:5d} {res.nit:9d} {g_norm:11.3e} {gtolk:9.2e} "
                f"{constr_violation:10.3e} {ctolk:9.2e} {np.mean(mu):9.2e} "
                f"{np.abs(Y.get(ic)).max(initial=0):9.2e}"
                + ("" if converged else "  (stalled)")
            )
        if stopped:
            success, message = False, STATUS_MESSAGES["callback"]
            report()
            break
        violated = viol >= ctolk
        if not converged:
            # a stall above the subproblem tolerance
            nstalls += 1
            if constr_violation < ctol:
                success, message = False, STATUS_MESSAGES["precision"]
                report()
                break
            if nstalls > max_stalls:
                success, message = False, STATUS_MESSAGES["stall"]
                report()
                break
            if np.any(violated & (ic >= 0)):  # penalize what is violated
                MU.set(ic[violated], np.minimum(tau * mu[violated], max_penalty))
                raise_new_rows(ic, viol)
                report()
                continue
            # nearly feasible: update the multipliers, keep the tolerances
        else:
            nstalls = 0
        if g_norm < gtol and constr_violation < ctol:
            success, message = True, STATUS_MESSAGES["gtol"]
        # first order update where nearly feasible, more penalty elsewhere
        y_new = np.clip(np.sqrt(mu) * r_c, -max_multiplier, max_multiplier)
        upd = ~violated & (ic >= 0)
        Y.set(ic[upd], y_new[upd])
        MU.set(ic[violated], np.minimum(tau * mu[violated], max_penalty))
        raise_new_rows(ic, viol)
        if converged and constr_violation < ctolk:
            ctolk = max(ctolk / mean_mu(ic, viol) ** beta_eta, ctol)
            gtolk = max(gtolk / mean_mu(ic, viol) ** beta_omega, gtol, min_inner_gtol)
        elif converged:
            ctolk = max(eta / mean_mu(ic, viol) ** alpha_eta, ctol)
            gtolk = max(omega / mean_mu(ic, viol) ** alpha_omega, gtol, min_inner_gtol)
        report()
        if keep_rows is not None:
            keep_rows(Y.keys)
            if not np.array_equal(ids_c(x, *args), np.where(ic >= 0, ic, -1)):
                J = None  # the kept rows changed the selection at x
        if success:
            break
        # merit and its gradient at x under the updated multipliers and penalties
        probe = lsq_composite(
            fun_all,
            x,
            jac_all,
            None,
            None,
            None,
            isb_all,
            row_ids=ids_all,
            rows=rows,
            args=args,
            x_scale=x_scale,
            verbose=0,
            maxiter=0,
            options={"initial_jac": J},
        )
        g_norm, J = float(probe.optimality), probe.jac
        nfev += 1
        njev += probe.njev

    ic_final = ids_c(x, *args)
    result = OptimizeResult(
        x=x,
        success=success,
        message=message,
        y=Y.get(ic_final),
        penalty_param=MU.get(ic_final),
        con_ids=ic_final,
        optimality=g_norm,
        constr_violation=constr_violation,
        nit=iteration,
        nfev=nfev,
        njev=njev,
        allx=allx,
        outer_history=history,
    )
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Max scaled gradient: {:.3e}".format(g_norm))
        print("         Max constraint violation: {:.3e}".format(constr_violation))
        print("         Outer iterations: {:d}".format(outer))
        print("         Iterations: {:d}".format(iteration))
        print("         Function evaluations: {:d}".format(nfev))
        print("         Jacobian evaluations: {:d}".format(njev))
    return result
