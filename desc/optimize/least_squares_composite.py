"""Levenberg-Marquardt for least squares with bound rows kept exact as hinges."""

import numpy as np
from scipy.optimize import OptimizeResult

from desc.backend import jax, jit, jnp
from desc.batching import jacfwd_chunked
from desc.utils import errorif, setdefault

from .composite import _model_resid, composite_model, composite_resid, solve_composite
from .utils import (
    STATUS_MESSAGES,
    check_termination,
    compute_jac_scale,
    print_header_nonlinear,
    print_iteration_nonlinear,
    scale_columns,
)


def paired_decrease(r, ids, r_new, ids_new):
    """``(|r|^2 - |r_new|^2) / 2`` summed row by row over rows with the same id.

    Rows are paired by id (``-1`` never pairs); unpaired rows contribute their own
    squares. Pairing cancels the large equal parts of the two costs row by row, so
    the result resolves decreases far below the rounding of either total.
    """
    order = jnp.argsort(ids)
    sid = ids[order]
    pos = jnp.clip(jnp.searchsorted(sid, ids_new), 0, ids.size - 1)
    match = (sid[pos] == ids_new) & (ids_new >= 0)
    old = jnp.where(match, r[order][pos], 0.0)
    paired = jnp.sum(jnp.where(match, (old - r_new) * (old + r_new), 0.0))
    old_matched = jnp.zeros(ids.size, dtype=bool).at[order[pos]].max(match)
    unpaired = jnp.sum(jnp.where(old_matched, 0.0, r**2))
    unpaired -= jnp.sum(jnp.where(match, 0.0, r_new**2))
    return 0.5 * (paired + unpaired)


def lsq_composite(  # noqa: C901
    fun,
    x0,
    jac,
    lo,
    hi,
    tgt,
    isb,
    hess=None,
    row_ids=None,
    rows=None,
    args=(),
    x_scale="jac",
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options=None,
):
    """Minimize half the sum of squared hinge and target residuals.

    The residual of row ``i`` is ``w_i (max(0, s_i - hi_i) - max(0, lo_i - s_i))`` on
    bound rows and ``w_i (s_i - tgt_i)`` on target rows, with ``s = fun(x)`` the
    pre-hinge values and weights ``w = 1`` unless ``rows`` gives them. Each step
    minimizes a model with ``s`` linearized and the hinges kept exact (see
    ``desc.optimize.composite``), damped Levenberg-Marquardt style, and is accepted when
    the cost decreases. Optionally a second order term ``S = sum r_i Hess(r_i)`` is
    added to the model, either as a structured secant estimate (NL2SOL style, at
    Gauss-Newton cost) or exactly.

    Parameters
    ----------
    fun : callable
        Pre-hinge row values, ``fun(x, *args) -> 1d array``.
    x0 : array-like
        Initial guess.
    jac : callable
        Jacobian of ``fun``.
    lo, hi, tgt : array-like
        Per-row lower and upper bounds (used on bound rows) and targets (used on
        target rows), in the units of ``fun``. Ignored if ``rows`` is given.
    isb : array-like of bool
        True for bound rows.
    hess : callable, optional
        Hessian of the cost ``0.5 |r(x)|^2``, ``hess(x, *args) -> 2d array``, used for
        ``hessian="exact"`` and ``finish_steps > 0``. If not given, the exact second
        order term is computed by differentiating ``fun``, which must then be
        traceable by JAX.
    row_ids : callable, optional
        Identity of each row, ``row_ids(x, *args) -> int array`` (-1 for none), used
        to pair rows between iterates when measuring the decrease and passed to
        ``rows``. Rows are paired by position if not given.
    rows : callable, optional
        Per-row data at each evaluation, ``rows(ids) -> (lo, hi, tgt, w)`` with ``ids``
        from ``row_ids`` (None without it) and positive weights ``w``. For rows whose
        bounds or weights follow their identity, as in an augmented Lagrangian. It may
        return a fifth entry, a boolean array: those rows are left out of the
        ``x_scale="jac"`` column norms while they are inactive (zero hinge residual),
        so a heavily weighted guard row does not shrink the steps before it binds.
    args : tuple
        Additional arguments passed to ``fun``, ``jac``, ``hess`` and ``row_ids``.
    x_scale : array_like or ``'jac'``, optional
        Characteristic scale of each variable. If ``'jac'``, the inverse of the largest
        column norm of the weighted Jacobian seen so far.
    ftol, xtol : float or None, optional
        Stop when an accepted step reduces the cost by less than ``ftol * cost``, or
        the scaled step is shorter than ``xtol * (xtol + |x|)``. None disables.
    gtol : float or None, optional
        Stop when the max norm of the scaled gradient is below ``gtol``.
    verbose : {0, 1, 2}, optional
        * 0 : work silently.
        * 1 (default) : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        Maximum number of iterations. Defaults to ``size(x) * 100``.
    callback : callable, optional
        Called after each accepted step as ``callback(xk, *args)``; returning True
        stops the optimization.
    options : dict, optional
        Optional keyword arguments.

        - ``"model"`` : (``"composite"`` or ``"gn"``) Keep the hinges in the model, or
          freeze the active set as the Gauss-Newton model does. Default
          ``"composite"``.
        - ``"hessian"`` : (``"gn"``, ``"secant"`` or ``"exact"``) Second order term
          added to the model. ``"secant"`` updates an estimate of ``S`` from gradient
          changes and uses it only while it predicted the last step better than the
          plain model. Default ``"gn"``.
        - ``"finish_steps"`` : (int >= 0) Once the ``"gn"`` or ``"secant"`` iteration
          stalls (``ftol``, ``xtol``, or the damping exceeding ``max_damping``) or
          reaches ``finish_gtol``, take up to this many accepted steps with the exact
          second order term. Default 0.
        - ``"finish_gtol"`` : (float) Scaled gradient at which to start the exact
          finish. Default 0, i.e. only on a stall.
        - ``"decrease"`` : (``"paired"`` or ``"total"``) Measure the actual decrease
          as ``sum((r - r_new) * (r + r_new)) / 2``, pairing rows by ``row_ids`` (or
          position), or as the difference of the two costs. Near convergence the
          decrease can fall below the rounding of the total cost, where only the
          paired form resolves it. The predicted decrease of the composite model
          is measured the same way. Default ``"paired"``.
        - ``"initial_damping"`` : (float > 0) Initial damping relative to the largest
          squared column norm of the scaled Jacobian. Default 1e-3.
        - ``"max_damping"`` : (float) Stop once rejections push the damping above
          this. Default 1e20.
        - ``"initial_jac"`` : (array) Jacobian of ``fun`` at ``x0``, if known.
        - ``"hess_chunk_size"`` : (int) Batch size of the Hessian-vector products for
          the exact second order term when ``hess`` is not given. Default 1.
        - ``"max_nfev"`` : (int > 0) Maximum number of function evaluations. Default
          ``5*maxiter+1``.
        - ``"track_steps"`` : (bool) Record one dict per attempted step (damping,
          step norm, predicted and actual decrease, ratio, accepted) in
          ``result["step_log"]``. Default False.

    Returns
    -------
    res : OptimizeResult
        The optimization result. Besides the usual fields: ``s`` and ``ids``, the
        pre-hinge values and row ids at ``x``; ``jac``, the Jacobian of ``fun`` at
        ``x``; ``step_history``, one dict per iteration (cost, scaled gradient, damping,
        model used, active hinges).

    """
    options = {} if options is None else dict(options)
    model = options.pop("model", "composite")
    hessian = options.pop("hessian", "gn")
    finish_steps = options.pop("finish_steps", 0)
    finish_gtol = options.pop("finish_gtol", 0.0)
    decrease = options.pop("decrease", "paired")
    initial_damping = options.pop("initial_damping", 1e-3)
    max_damping = options.pop("max_damping", 1e20)
    A = options.pop("initial_jac", None)
    hess_chunk_size = options.pop("hess_chunk_size", 1)
    track_steps = options.pop("track_steps", False)
    maxiter = setdefault(maxiter, x0.size * 100)
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    errorif(len(options) > 0, ValueError, f"Unknown options: {list(options.keys())}")
    errorif(model not in ["composite", "gn"], ValueError, f"Unknown model {model}")
    errorif(
        decrease not in ["paired", "total"],
        ValueError,
        f"Unknown decrease {decrease}",
    )
    errorif(
        hessian not in ["gn", "secant", "exact"],
        ValueError,
        f"Unknown hessian {hessian}",
    )
    errorif(
        isinstance(x_scale, str) and x_scale not in ["jac", "auto"],
        ValueError,
        f"x_scale should be one of 'jac', 'auto' or array-like, got {x_scale}",
    )
    ftol, xtol, gtol = setdefault(ftol, 0.0), setdefault(xtol, 0.0), setdefault(gtol, 0)
    callback = setdefault(callback, lambda *args: False)
    jac_scale = isinstance(x_scale, str)
    if not jac_scale:
        scale = jnp.broadcast_to(x_scale, x0.shape)
    isb = jnp.asarray(isb, dtype=bool)
    if rows is None:
        static_rows = tuple(
            jnp.broadcast_to(jnp.asarray(v, dtype=float), isb.shape)
            for v in (lo, hi, tgt, 1.0)
        )
        rows = lambda ids: static_rows  # noqa: E731
    n = x0.size
    eye = jnp.eye(n)

    @jit
    def exact_S(x, rho):
        """sum_i rho_i Hess(s_i), from Hessian-vector products of fun."""

        def grad_rho(x):
            return jax.vjp(lambda x: fun(x, *args), x)[1](rho)[0]

        return jacfwd_chunked(grad_rho, chunk_size=hess_chunk_size)(x)

    def evaluate(x):
        s = fun(x, *args)
        ids = None if row_ids is None else row_ids(x, *args)
        rd = rows(ids)
        r = rd[3] * composite_resid(s, rd[0], rd[1], rd[2], isb)
        return s, ids, rd, r

    x = jnp.asarray(x0).copy()
    s, ids, rd, r = evaluate(x)
    cost = 0.5 * jnp.dot(r, r)
    f0 = r
    nfev, njev, nhev = 1, 0, 0
    iteration, finish_left = 0, None
    scale_inv = None
    lam, nu = None, 2.0
    # secant state, in unscaled coordinates
    Su, g_prev, step_prev, use_S, nskip = jnp.zeros((n, n)), None, None, False, 0
    success, message = None, None
    actual_reduction, step_norm = jnp.inf, jnp.inf
    allx, history, step_log = [x], [], []

    if verbose > 1:
        print_header_nonlinear()

    need_jac, S_exact, stop = True, None, None
    while True:
        if need_jac:
            if A is None:
                A = jac(x, *args)
                njev += 1
            lo_, hi_, tgt_, w = rd[:4]
            A = A * w[:, None]
            g = jnp.dot(r, A)  # r vanishes on inactive hinge rows
            if jac_scale:
                if len(rd) > 4:  # rows excluded from the scale while inactive
                    out = jnp.asarray(rd[4]) & jnp.asarray(isb) & (r == 0)
                    scale, scale_inv = compute_jac_scale(
                        jnp.where(out[:, None], 0.0, A), scale_inv
                    )
                else:
                    scale, scale_inv = compute_jac_scale(A, scale_inv)
            d = scale
            B = scale_columns(A, d)
            A = None
            sw, low, hiw = w * s, w * lo_, w * hi_
            keep = (~isb | (r != 0)).astype(B.dtype)
            Btr = jnp.dot(r, B)
            GtG = B.T @ (keep[:, None] * B)
            g_norm = jnp.max(jnp.abs(g * d))
            x_norm = jnp.linalg.norm(x / d)
            if hessian == "secant" and step_prev is not None:
                # sized rank-2 update with y# = dg - G+ s, independent of row order
                yv = g - g_prev
                ysh = yv - (GtG @ (step_prev / d)) / d
                ys = yv @ step_prev
                sSs = step_prev @ Su @ step_prev
                if ys > 0:
                    if sSs != 0:
                        Su = Su * min(1.0, abs(step_prev @ ysh) / abs(sSs))
                    w_ = ysh - Su @ step_prev
                    Su = (
                        Su
                        + (jnp.outer(w_, yv) + jnp.outer(yv, w_)) / ys
                        - (w_ @ step_prev) * jnp.outer(yv, yv) / ys**2
                    )
                else:
                    nskip += 1
            need_jac = False
            history.append(
                dict(
                    iteration=iteration,
                    cost=float(cost),
                    optimality=float(g_norm),
                    damping=float(setdefault(lam, jnp.nan)),
                    hessian=hessian,
                    model_S=bool(hessian == "exact" or use_S),
                    active_hinges=int(jnp.sum(isb & (r != 0))),
                )
            )
            if verbose > 1:
                print_iteration_nonlinear(
                    iteration, nfev, cost, actual_reduction, step_norm, g_norm
                )

        # stopping, and switching to the exact finish
        if g_norm < gtol:
            success, message = True, STATUS_MESSAGES["gtol"] + f" ({gtol=:.2e})"
            break
        if stop == "hard":
            break
        start_finish = finish_steps > 0 and finish_left is None
        if stop == "stall" and not start_finish:
            break
        if start_finish and (stop == "stall" or g_norm < finish_gtol):
            if verbose > 1:
                print(f"Switching to at most {finish_steps} exact Hessian steps")
            hessian, finish_left = "exact", finish_steps
            success = message = stop = None
            lam, nu = None, 2.0
        if finish_left == 0:
            success, message = False, STATUS_MESSAGES["maxiter"] + " (finish_steps)"
            break
        if iteration >= maxiter:
            success, message = False, STATUS_MESSAGES["maxiter"]
            break

        if lam is None:
            lam = initial_damping * jnp.max(jnp.diag(GtG))
        with_S = hessian == "exact" or (hessian == "secant" and use_S)
        Sd, lam_floor = None, 0.0
        if hessian == "exact":
            if S_exact is None:
                if hess is not None:
                    Hd = d[:, None] * hess(x, *args) * d[None, :]
                    S_exact = 0.5 * (Hd + Hd.T) - GtG
                else:
                    S = exact_S(x, r * w)
                    S_exact = d[:, None] * 0.5 * (S + S.T) * d[None, :]
                nhev += 1
            Sd = S_exact
        elif hessian == "secant":
            Sd = d[:, None] * Su * d[None, :]
        if with_S:
            # damping floor keeps the model convex, active hinges included
            lam_floor = max(0.0, -float(jnp.linalg.eigvalsh(GtG + Sd)[0])) * 1.01
            lam_floor += 1e-14

        def model_reduction(q):
            if model == "gn":
                return -(q @ Btr + 0.5 * q @ GtG @ q)
            if decrease == "total":
                return cost - composite_model(sw, B, r, q, isb, low, hiw)
            # row by row, as the actual decrease: a difference of totals is noise
            m = _model_resid(sw, B, r, q, isb, low, hiw)
            return 0.5 * jnp.dot(r - m, r + m)

        while True:
            lam_eff = max(lam, lam_floor) if with_S else lam
            if model == "composite":
                q, _ = solve_composite(
                    sw, B, r, lam_eff, isb, low, hiw, S=Sd if with_S else None
                )
            else:
                Hq = GtG + lam_eff * eye + (Sd if with_S else 0)
                q = -jnp.linalg.solve(Hq, Btr)
            pred_plain = model_reduction(q)
            pred = pred_plain - (0.5 * q @ Sd @ q if with_S else 0.0)
            step = d * q
            x_new = x + step
            s_new, ids_new, rd_new, r_new = evaluate(x_new)
            nfev += 1
            cost_new = 0.5 * jnp.dot(r_new, r_new)
            if decrease == "paired" and row_ids is not None:
                actual_reduction = paired_decrease(r, ids, r_new, ids_new)
            elif decrease == "paired":
                # cancels large equal parts row by row instead of between totals
                actual_reduction = 0.5 * jnp.dot(r - r_new, r + r_new)
            else:
                actual_reduction = cost - cost_new
            rho = actual_reduction / pred if pred > 0 else -1.0
            if track_steps:
                step_log.append(
                    dict(
                        iteration=iteration,
                        damping=float(lam_eff),
                        step_norm=float(jnp.linalg.norm(q)),
                        predicted=float(pred),
                        predicted_plain=float(pred_plain),
                        actual=float(actual_reduction),
                        actual_total=float(cost - cost_new),
                        rho=float(rho),
                        accepted=bool(rho > 0),
                    )
                )
            if rho > 0 or lam > max_damping or nfev >= max_nfev:
                break
            lam, nu = lam * nu, 2 * nu

        if rho <= 0:
            if nfev >= max_nfev:
                success, message, stop = False, STATUS_MESSAGES["max_nfev"], "hard"
            else:
                success, message, stop = False, STATUS_MESSAGES["pr_loss"], "stall"
            actual_reduction = step_norm = 0.0
            continue

        if hessian == "secant":
            pred_S = pred_plain - 0.5 * q @ Sd @ q
            use_S = bool(
                abs(pred_S - actual_reduction) < abs(pred_plain - actual_reduction)
            )
            step_prev, g_prev = step, g
        step_norm = jnp.linalg.norm(q)
        success, message = check_termination(
            actual_reduction,
            cost,
            step_norm,
            x_norm,
            jnp.inf,
            rho,
            ftol,
            xtol,
            0.0,
            iteration,
            np.inf,
            nfev,
            max_nfev,
            min_trust_radius=0.0,
        )
        stop = {True: "stall", False: "hard", None: None}[success]
        x, s, ids, rd, r, cost = x_new, s_new, ids_new, rd_new, r_new, cost_new
        allx.append(x)
        lam *= max(1 / 3, 1 - (2 * rho - 1) ** 3)
        nu = 2.0
        iteration += 1
        if finish_left is not None:
            finish_left -= 1
        need_jac, S_exact = True, None
        del B
        if callback(jnp.copy(x), *args):
            success, message, stop = False, STATUS_MESSAGES["callback"], "hard"

    # B holds the weighted, column scaled Jacobian at x
    jac_x = B / (rd[3][:, None] * d[None, :])
    result = OptimizeResult(
        x=x,
        success=success,
        cost=cost,
        fun=r,
        s=s,
        ids=ids,
        jac=jac_x,
        grad=g,
        optimality=g_norm,
        nfev=nfev,
        njev=njev,
        nhev=nhev,
        nit=iteration,
        message=message,
        allx=allx,
        step_history=history,
        step_log=step_log,
        secant_skips=nskip,
    )
    result["fse"] = r
    result["f0se"] = f0
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print("         Current function value: {:.3e}".format(result["cost"]))
        print("         Max scaled gradient: {:.3e}".format(result["optimality"]))
        print(
            "         Total delta_x: {:.3e}".format(jnp.linalg.norm(x0 - result["x"]))
        )
        print("         Iterations: {:d}".format(result["nit"]))
        print("         Function evaluations: {:d}".format(result["nfev"]))
        print("         Jacobian evaluations: {:d}".format(result["njev"]))
    return result
