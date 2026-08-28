"""Augmented Lagrangian for vector valued objectives."""

from scipy.optimize import NonlinearConstraint, OptimizeResult

from desc.backend import jnp, put, qr, qr_multiply
from desc.utils import errorif, safediv, scale_tail_rows, setdefault

from .bound_utils import (
    cl_scaling_vector,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    select_step,
)
from .tr_subproblems import (
    trust_region_step_exact_cho,
    trust_region_step_exact_qr,
    trust_region_step_exact_svd,
    update_tr_radius,
)
from .utils import (
    STATUS_MESSAGES,
    check_termination,
    compute_jac_scale,
    inequality_to_bounds,
    print_header_nonlinear,
    print_iteration_nonlinear,
    solve_triangular_regularized,
)


def lsq_auglag(  # noqa: C901
    fun,
    x0,
    jac,
    bounds=(-jnp.inf, jnp.inf),
    constraint=None,
    args=(),
    x_scale=1,
    ftol=1e-6,
    xtol=1e-6,
    gtol=1e-6,
    ctol=1e-6,
    verbose=1,
    maxiter=None,
    callback=None,
    options={},
):
    """Minimize a function with constraints using an augmented Lagrangian method.

    The objective function is assumed to be vector valued, and is minimized in the least
    squares sense.

    Parameters
    ----------
    fun : callable
        objective to be minimized. Should have a signature like fun(x,*args)-> 1d array
    x0 : array-like
        initial guess
    jac : callable:
        function to compute Jacobian matrix of fun
    bounds : tuple of array-like
        Lower and upper bounds on independent variables. Defaults to no bounds.
        Each array must match the size of x0 or be a scalar, in the latter case a
        bound will be the same for all variables. Use np.inf with an appropriate sign
        to disable bounds on all or some variables.
    constraint : scipy.optimize.NonlinearConstraint
        constraint to be satisfied
    args : tuple
        additional arguments passed to fun, grad, and hess
    x_scale : array_like or ``'hess'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        An alternative view is that the size of a trust region along jth
        dimension is proportional to ``x_scale[j]``. Improved convergence may
        be achieved by setting ``x_scale`` such that a step of a given size
        along any of the scaled variables has a similar effect on the cost
        function. If set to ``'hess'``, the scale is iteratively updated using the
        inverse norms of the columns of the Hessian matrix.
    ftol : float or None, optional
        Tolerance for termination by the change of the cost function.
        The optimization process is stopped when ``dF < ftol * F``,
        and there was an adequate agreement between a local quadratic model and
        the true model in the last step. If None, the termination by this
        condition is disabled.
    xtol : float or None, optional
        Tolerance for termination by the change of the independent variables.
        Optimization is stopped when ``norm(dx) < xtol * (xtol + norm(x))``.
        If None, the termination by this condition is disabled.
    gtol : float or None, optional
        Absolute tolerance for termination by the norm of the gradient.
        Optimizer terminates when ``max(abs(g)) < gtol``., where
        If None, the termination by this condition is disabled.
    ctol : float, optional
        Tolerance for stopping based on infinity norm of the constraint violation.
        This is the violation of the constraints as posed, ``max(lb - c(x), c(x) - ub,
        0)``, not the residual of the internal slack reformulation. Optimizer
        terminates when it is ``< ctol`` AND one or more of the other tolerances are
        met (``ftol``, ``xtol``, ``gtol``)
    verbose : {0, 1, 2}, optional
        * 0 : work silently.
        * 1 (default) : display a termination report.
        * 2 : display progress during iterations
    maxiter : int, optional
        maximum number of iterations. Defaults to size(x)*100
    callback : callable, optional
        Called after each iteration. Should be a callable with
        the signature:

            ``callback(xk, *args) -> bool``

        where ``xk`` is the current parameter vector, and ``args``
        are the same arguments passed to fun and jac. If callback returns True
        the algorithm execution is terminated.
    options : dict, optional
        dictionary of optional keyword arguments to override default solver settings.

        - ``"initial_penalty_parameter"`` : (float or array-like) Initial value for the
          quadratic penalty parameter. May be array like, in which case it should be the
          same length as the number of constraint residuals. Default 10.
        - ``"initial_multipliers"`` : (float or array-like or ``"least_squares"``)
          Initial Lagrange multipliers. May be array like, in which case it should be
          the same length as the number of constraint residuals. If ``"least_squares"``,
          uses an estimate based on the least squares solution of the optimality
          conditions, see ch 14 of [1]_. Default 0.
        - ``"omega"`` : (float) Hyperparameter for determining initial gradient
          tolerance. See algorithm 14.4.2 from [1]_ for details. Default 1.0
        - ``"eta"`` : (float) Hyperparameter for determining initial constraint
          tolerance. See algorithm 14.4.2 from [1]_ for details. Default 1.0, or
          with ``scaled_termination`` (the default), the initial constraint
          violation capped to 1e-2, falling back to 1e-2 if that is ``<= ctol``
          (as it is for a feasible starting point).
        - ``"alpha_omega"`` : (float) Hyperparameter for updating gradient tolerance.
          See algorithm 14.4.2 from [1]_ for details. Default 1.0
        - ``"beta_omega"`` : (float) DEPRECATED, no effect. The successive gradient
          tolerances are now set by ``inner_reduction`` instead. Default 1.0
        - ``"alpha_eta"`` : (float) Hyperparameter for updating constraint tolerance.
          See algorithm 14.4.2 from [1]_ for details. Default 0.1
        - ``"beta_eta"`` : (float) Hyperparameter for updating constraint tolerance.
          See algorithm 14.4.2 from [1]_ for details. Default 0.9
        - ``"tau"`` : (float) Factor to increase penalty parameter by when constraint
          violation doesn't decrease sufficiently. Default 10
        - ``"max_multiplier"`` : (float > 0) Safeguarding box for the Lagrange
          multipliers; each update is clipped to ``[-max_multiplier,
          max_multiplier]``. The update ``y <- y - mu*c`` is only a valid multiplier
          estimate at an approximate minimizer, so subproblems that repeatedly end
          short can integrate a non-zero residual without bound. Clipping degrades the
          method towards a quadratic penalty method, which is still globally
          convergent to a feasible point. This is a guard against blow-up, not a
          tuning knob -- it only binds when something has already gone wrong. Default
          1e6
        - ``"track_residual"`` : (bool) Add ``max|f|`` and ``mean|f|`` columns to the
          iteration printout, where ``f`` is the objective residual as the optimizer
          sees it (scaled by ``weight``/``normalization``, including quadrature
          weights). Costs two reductions per iteration on a vector already in hand, so
          it is free next to a Jacobian evaluation. ``Cost`` is ``1/2*||f||^2``, so
          ``max|f|`` is the new information: whether the error is spread over the grid
          or concentrated at a few nodes. Recorded in the result as ``allf_max`` and
          ``allf_mean`` regardless of this setting. Default False
        - ``"max_inner_stalls"`` : (int >= 0) How many times in a row a subproblem may
          stall (``ftol``/``xtol`` met, or the trust region collapsing) *without real
          progress* at a point that is not yet a KKT point before the solver gives up.
          Such a stall says the *subproblem* is done, not the problem, so instead of
          returning it runs the outer update and restarts the inner solve from the
          initial trust radius. Which outer update depends on why it stalled: at a
          feasible point the remaining optimality is dual error, so the multipliers are
          updated and ``gtolk`` is floored at the accuracy the inner solve has shown it
          can reach; while still infeasible the penalty is too weak, so ``mu`` is raised
          and ``y`` left alone. Restarting without changing ``(y, mu)`` would re-solve a
          bit-identical subproblem forever. Set to 0 to recover the old behaviour, where
          a stalled subproblem ended the whole solve. Default 3
        - ``"inner_reduction"`` : (float > 1) Factor by which each subproblem must
          reduce the optimality before the next multiplier update is taken. After every
          update, ``gtolk`` is set to ``max(g_norm / inner_reduction, gtol)`` --
          relative to where that subproblem actually starts, not from the ``mu``-driven
          schedule of alg 14.4.2. That absolute schedule fails in both directions when
          the reachable optimality is above ``gtol``: divided by ``mean(mu)`` it hits
          ``gtol`` after a few updates and the outer loop dies, and floored at a
          measured ``g_norm`` it latches onto the high excursions of a signal that
          swings an order of magnitude between iterations near convergence, so the
          target is met at once and the outer loop churns. A relative target is
          reachable by construction and always asks for real inner work. Larger values
          mean longer, more thoroughly solved subproblems and fewer multiplier updates.
          Note this supersedes ``beta_omega``, which no longer has any effect.
          Default 10
        - ``"max_inner_iter"`` : (int > 0 or None) EXPERIMENTAL, default None (off).
          Maximum inner iterations spent on a single subproblem before the outer update
          is taken regardless of whether ``gtolk`` was reached. A subproblem can grind
          without ever stalling -- accepted steps too small to trip ``ftol``/``xtol``,
          but nowhere near ``gtolk`` -- so nothing detects that it is finished. Setting
          this caps that wait. Off by default: with ``inner_reduction`` making ``gtolk``
          reachable by construction the grind it guards against should not arise, and an
          earlier version of this option cut healthy early subproblems short.
        - ``"max_nfev"`` : (int > 0) Maximum number of function evaluations (each
          iteration may take more than one function evaluation). Default is
          ``5*maxiter+1``
        - ``"max_dx"`` : (float > 0) Maximum allowed change in the norm of x from its
          starting point. Default np.inf.
        - ``"initial_trust_radius"`` : (``"scipy"``, ``"conngould"``, ``"mix"`` or
          float > 0) Initial trust region radius. ``"scipy"`` uses the scaled norm of
          x0, which is the default behavior in ``scipy.optimize.least_squares``.
          ``"conngould"`` uses the norm of the Cauchy point, as recommended in ch17
          of [1]_. ``"mix"`` uses the geometric mean of the previous two options. A
          float can also be passed to specify the trust radius directly.
          Default is ``"scipy"``.
        - ``"initial_trust_ratio"`` : (float > 0) A extra scaling factor that is
          applied after one of the previous heuristics to determine the initial trust
          radius. Default 1.
        - ``"max_trust_radius"`` : (float > 0) Maximum allowable trust region radius.
          Default ``np.inf``.
        - ``"min_trust_radius"`` : (float >= 0) Minimum allowable trust region radius.
          Optimization is terminated if the trust region falls below this value.
          Default ``np.finfo(x0.dtype).eps``.
        - ``"tr_increase_threshold"`` : (0 < float < 1) Increase the trust region
          radius when the ratio of actual to predicted reduction exceeds this threshold.
          Default 0.75.
        - ``"tr_decrease_threshold"`` : (0 < float < 1) Decrease the trust region
          radius when the ratio of actual to predicted reduction is less than this
          threshold. Default 0.25.
        - ``"tr_increase_ratio"`` : (float > 1) Factor to increase the trust region
          radius by when  the ratio of actual to predicted reduction exceeds threshold.
          Default 2.
        - ``"tr_decrease_ratio"`` : (0 < float < 1) Factor to decrease the trust region
          radius by when  the ratio of actual to predicted reduction falls below
          threshold. Default 0.25.
        - ``"tr_method"`` : (``"qr"``, ``"svd"``, ``"cho"``) Method to use for solving
          the trust region subproblem. ``"qr"`` and ``"cho"`` uses a sequence of QR or
          Cholesky factorizations (generally 2-3), while ``"svd"`` uses one singular
          value decomposition. ``"cho"`` is generally the fastest for large systems,
          especially on GPU, but may be less accurate for badly scaled systems.
          ``"svd"`` is the most accurate but significantly slower. Default ``"qr"``.
        - ``"scaled_termination"`` : Whether to evaluate termination criteria for
          ``xtol`` and ``gtol`` in scaled / normalized units (default) or base units.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a ``OptimizeResult`` object.
        Important attributes are: ``x`` the solution array, ``success`` a
        Boolean flag indicating if the optimizer exited successfully.

    References
    ----------
    .. [1] Conn, Andrew, and Gould, Nicholas, and Toint, Philippe. "Trust-region
           methods" (2000).

    """
    constraint = setdefault(
        constraint,
        NonlinearConstraint(  # create a dummy constraint
            fun=lambda x, *args: jnp.array([0.0]),
            lb=0.0,
            ub=0.0,
            jac=lambda x, *args: jnp.zeros((1, x.size)),
        ),
    )

    (
        z0,
        fun_wrapped,
        jac_wrapped,
        _,
        constraint_wrapped,
        zbounds,
        z2xs,
    ) = inequality_to_bounds(
        x0,
        fun,
        jac,
        None,
        constraint,
        bounds,
        *args,
    )

    # L(x,y,mu) = 1/2 f(x)^2 - y*c(x) + mu/2 c(x)^2 + y^2/(2*mu)
    # = 1/2 f(x)^2 + 1/2 [-y/sqrt(mu) + sqrt(mu) c(x)]^2

    def lagfun(f, c, y, mu):
        sqrt_mu = jnp.sqrt(mu)
        c = -y / sqrt_mu + sqrt_mu * c
        return jnp.concatenate((f, c))

    def lagjac(z, y, mu, *args):
        # NOTE: `y` is unused -- J depends only on (z, mu), and on `mu` only through
        # the sqrt(mu) row scaling of the constraint block. So when `mu` changes at a
        # fixed `z`, J can be rescaled rather than rebuilt. See `mu_J` below.
        Jf = jac_wrapped(z, *args)
        Jc = constraint_wrapped.jac(z, *args)
        Jc = jnp.sqrt(mu)[:, None] * Jc
        return jnp.vstack((Jf, Jc))

    nfev = 0
    njev = 0
    iteration = 0

    z = z0.copy()
    f = fun_wrapped(z, *args)
    f0 = f
    cost = 1 / 2 * jnp.dot(f, f)
    c = constraint_wrapped.fun(z, *args)
    nfev += 1

    # `c` is the residual of the slack-reformulated *equality* problem, c(x) - s.
    # It is the right input to the multiplier update, but it is not the violation of
    # the original constraints: with an interior slack it equals y/mu, so it measures
    # dual convergence and can be driven to zero just by growing mu. Recover the true
    # violation instead -- no extra evaluation needed, since c(x) = c(z) + s.
    _clb, _cub = (jnp.broadcast_to(b, c.shape) for b in (constraint.lb, constraint.ub))
    _ineq = _clb != _cub

    def constr_violation_rows(c_z, z):
        """Per-row violation of the ORIGINAL constraints; 0 where strictly feasible."""
        s = z2xs(z)[1]
        c_x = c_z + put(jnp.zeros_like(c_z), _ineq, s)
        viol = jnp.where(
            _ineq,
            jnp.maximum(_clb - c_x, c_x - _cub),  # <= 0 inside the bounds
            jnp.abs(c_z),  # equality rows already carry the target
        )
        return jnp.maximum(viol, 0.0)

    def constr_violation_fun(c_z, z):
        """Max violation of the ORIGINAL constraints; 0 when strictly feasible."""
        return jnp.max(constr_violation_rows(c_z, z))

    constr_violation = constr_violation_fun(c, z)
    c_norm = jnp.linalg.norm(c, ord=jnp.inf)  # reformulated residual, drives the AL

    lb, ub = zbounds
    bounded = jnp.any(lb != -jnp.inf) | jnp.any(ub != jnp.inf)
    assert in_bounds(z, lb, ub), "x0 is infeasible"
    z = make_strictly_feasible(z, lb, ub)

    max_multiplier = options.pop("max_multiplier", 1e6)
    mu = options.pop("initial_penalty_parameter", 10 * jnp.ones_like(c))
    y = options.pop("initial_multipliers", jnp.zeros_like(c))
    if isinstance(y, str) and y == "least_squares":  # least squares multiplier estimate
        _J = constraint_wrapped.jac(z, *args)
        _g = f @ jac_wrapped(z, *args)
        y = jnp.linalg.lstsq(_J.T, _g)[0]
        y = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        y = jnp.clip(y, -max_multiplier, max_multiplier)
    y, mu, c = jnp.broadcast_arrays(y, mu, c)

    L = lagfun(f, c, y, mu)
    J = lagjac(z, y, mu, *args)
    mu_J = mu  # the `mu` the current `J` was built with
    Lcost = 1 / 2 * jnp.dot(L, L)
    g = L @ J

    allx = []

    maxiter = setdefault(maxiter, z.size * 100)
    max_nfev = options.pop("max_nfev", 5 * maxiter + 1)
    max_dx = options.pop("max_dx", jnp.inf)
    scaled_termination = options.pop("scaled_termination", True)

    jac_scale = isinstance(x_scale, str) and x_scale in ["jac", "auto"]
    if jac_scale:
        scale, scale_inv = compute_jac_scale(J)
    else:
        x_scale = jnp.broadcast_to(x_scale, x0.shape)
        # add ones for slack variables
        x_scale = jnp.concatenate([x_scale, jnp.ones(z0.size - x0.size)])
        scale, scale_inv = x_scale, 1 / x_scale

    v, dv = cl_scaling_vector(z, g, lb, ub)
    v = jnp.where(dv != 0, v * scale_inv, v)
    d = v**0.5 * scale
    diag_h = g * dv * scale

    g_h = g * d
    # TODO: place this function under JIT to use in-place operation (#1669)
    # The unscaled J is kept: an outer update at an unchanged `z` rescales it instead
    # of re-evaluating (see `mu_J`), which is worth one extra J-sized array.
    J_h = J * d
    g_norm = jnp.linalg.norm(
        (g * v * scale if scaled_termination else g * v), ord=jnp.inf
    )

    # conngould : norm of the cauchy point, as recommended in ch17 of Conn & Gould
    # scipy : norm of the scaled x, as used in scipy
    # mix : geometric mean of conngould and scipy
    tr_scipy = jnp.linalg.norm(z * scale_inv / v**0.5)
    conngould = safediv(jnp.sum(g_h**2), jnp.sum((J_h @ g_h) ** 2))
    init_tr = {
        "scipy": tr_scipy,
        "conngould": conngould,
        "mix": jnp.sqrt(conngould * tr_scipy),
    }
    trust_radius = options.pop("initial_trust_radius", "conngould")
    tr_ratio = options.pop("initial_trust_ratio", 1.0)
    trust_radius = init_tr.get(trust_radius, trust_radius)
    trust_radius *= tr_ratio
    trust_radius = trust_radius if (trust_radius > 0) else 1.0
    # kept so the inner solve can be restarted on a fresh radius when (y, mu) change
    init_trust_radius = trust_radius

    max_trust_radius = options.pop("max_trust_radius", jnp.inf)
    min_trust_radius = options.pop("min_trust_radius", jnp.finfo(z.dtype).eps)
    tr_increase_threshold = options.pop("tr_increase_threshold", 0.75)
    tr_decrease_threshold = options.pop("tr_decrease_threshold", 0.5)
    tr_increase_ratio = options.pop("tr_increase_ratio", 4)
    tr_decrease_ratio = options.pop("tr_decrease_ratio", 0.25)
    tr_method = options.pop("tr_method", "qr")

    # notation following Conn & Gould, algorithm 14.4.2, but with our mu = their mu^-1
    omega = options.pop("omega", min(g_norm, 1e-2) if scaled_termination else 1.0)
    # eta is the initial ctolk, which is compared against the reformulated residual,
    # so it keys off c_norm rather than the true violation. A feasible start gives
    # c_norm == 0 exactly, since the slack init sets s = clip(c(x0)) = c(x0). Any
    # eta <= ctol pins ctolk to ctol for the whole run, which disables the schedule,
    # so fall back to the cap in that case.
    eta_default = min(c_norm, 1e-2) if scaled_termination else 1.0
    eta = options.pop("eta", eta_default if eta_default > ctol else 1e-2)
    alpha_omega = options.pop("alpha_omega", 1.0)
    beta_omega = options.pop("beta_omega", 1.0)
    alpha_eta = options.pop("alpha_eta", 0.1)
    beta_eta = options.pop("beta_eta", 0.9)
    tau = options.pop("tau", 10)
    max_inner_stalls = options.pop("max_inner_stalls", 3)
    max_inner_iter = options.pop("max_inner_iter", None)
    inner_reduction = options.pop("inner_reduction", 10.0)
    track_residual = options.pop("track_residual", False)

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )
    errorif(
        tr_method not in ["cho", "svd", "qr"],
        ValueError,
        "tr_method should be one of 'cho', 'svd', 'qr', got {}".format(tr_method),
    )

    callback = setdefault(callback, lambda *args: False)

    z_norm = jnp.linalg.norm(((z * scale_inv) if scaled_termination else z), ord=2)
    success = None
    message = None
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    Lactual_reduction = jnp.inf
    alpha = 0.0  # "Levenberg-Marquardt" parameter
    n_inner_stalls = 0  # consecutive subproblem stalls with no accepted step
    iters_since_update = 0  # inner iterations spent on the current subproblem
    n_outer = 0  # number of augmented Lagrangian (y, mu) updates taken
    # (iteration, reason) for every outer update, so update cadence is recoverable
    # from the result object rather than only from a verbose log
    outer_log = []

    allx = [z]
    alltr = [trust_radius]
    # Per-iteration L-inf and L-1 norms of the objective residual. `f` is already
    # computed every iteration, so these are two reductions on a vector in hand --
    # negligible next to a Jacobian evaluation. `cost` is 1/2*||f||^2, so `allf_max`
    # is the genuinely new information: whether the error is spread over the grid or
    # concentrated at a few nodes.
    allf_max = [jnp.max(jnp.abs(f))]
    allf_mean = [jnp.mean(jnp.abs(f))]
    if g_norm < gtol and constr_violation < ctol:
        success, message = True, STATUS_MESSAGES["gtol"]

    gtolk = max(omega / jnp.mean(mu) ** alpha_omega, gtol)
    ctolk = max(eta / jnp.mean(mu) ** alpha_eta, ctol)

    if verbose > 2:
        print("Solver options:")
        print("-" * 60)
        print(f"{'Maximum Function Evaluations':<35}: {max_nfev}")
        print(f"{'Maximum Allowed Total Δx Norm':<35}: {max_dx:.3e}")
        print(f"{'Scaled Termination':<35}: {scaled_termination}")
        print(f"{'Trust Region Method':<35}: {tr_method}")
        print(f"{'Initial Trust Radius':<35}: {trust_radius:.3e}")
        print(f"{'Maximum Trust Radius':<35}: {max_trust_radius:.3e}")
        print(f"{'Minimum Trust Radius':<35}: {min_trust_radius:.3e}")
        print(f"{'Trust Radius Increase Ratio':<35}: {tr_increase_ratio:.3e}")
        print(f"{'Trust Radius Decrease Ratio':<35}: {tr_decrease_ratio:.3e}")
        print(f"{'Trust Radius Increase Threshold':<35}: {tr_increase_threshold:.3e}")
        print(f"{'Trust Radius Decrease Threshold':<35}: {tr_decrease_threshold:.3e}")
        print(f"{'Alpha Omega':<35}: {alpha_omega:.3e}")
        print(f"{'Beta Omega':<35}: {beta_omega:.3e}")
        print(f"{'Alpha Eta':<35}: {alpha_eta:.3e}")
        print(f"{'Beta Eta':<35}: {beta_eta:.3e}")
        print(f"{'Omega':<35}: {omega:.3e}")
        print(f"{'Eta':<35}: {eta:.3e}")
        print(f"{'Tau':<35}: {tau:.3e}")
        print("-" * 60, "\n")

    if verbose > 1:
        # gtolk is the tolerance the *subproblem* is being solved to; the outer loop
        # only advances when Optimality drops below it. Printing it makes a dead outer
        # loop (gtolk pinned far under any achievable Optimality) visible immediately.
        _extra_hdr = ("max|f|", "mean|f|") if track_residual else ()
        print_header_nonlinear(
            True, "Penalty param", "max(|mltplr|)", "gtolk", *_extra_hdr
        )
        print_iteration_nonlinear(
            iteration,
            nfev,
            cost,
            actual_reduction,
            step_norm,
            g_norm,
            constr_violation,
            jnp.mean(mu),
            jnp.max(jnp.abs(y)),
            gtolk,
            *((allf_max[-1], allf_mean[-1]) if track_residual else ()),
        )

    while iteration < maxiter and success is None:

        # we don't want to factorize the extra stuff if we don't need to
        J_a = jnp.vstack([J_h, jnp.diag(diag_h**0.5)]) if bounded else J_h
        L_a = jnp.concatenate([L, jnp.zeros(diag_h.size)]) if bounded else L

        if tr_method == "svd":
            U, s, Vt = jnp.linalg.svd(J_a, full_matrices=False)
        elif tr_method == "cho":
            B_h = jnp.dot(J_a.T, J_a)
        elif tr_method == "qr":
            # try full newton step
            tall = J_a.shape[0] >= J_a.shape[1]
            if tall:
                Qt_La, R = qr_multiply(J_a, L_a, mode="right")
                p_newton = solve_triangular_regularized(R, -Qt_La)
            else:
                # min-norm Newton step uses the QR of J_a.T
                Q, Rt = qr(J_a.T, mode="economic")
                p_newton = Q @ solve_triangular_regularized(Rt.T, -L_a, lower=True)
                del Q, Rt
                # the tr subproblem still needs the QR of J_a itself
                Qt_La, R = qr_multiply(J_a, L_a, mode="right")

        actual_reduction = -1
        Lactual_reduction = -1
        inner_stall = False

        # theta controls step back step ratio from the bounds.
        theta = max(0.995, 1 - g_norm)

        while Lactual_reduction <= 0 and nfev <= max_nfev:
            # Solve the sub-problem.
            # This gives us the proposed step relative to the current position
            # and it tells us whether the proposed step
            # has reached the trust region boundary or not.
            if tr_method == "svd":
                step_h, hits_boundary, alpha = trust_region_step_exact_svd(
                    L_a, U, s, Vt.T, trust_radius, alpha
                )
            elif tr_method == "cho":
                step_h, hits_boundary, alpha = trust_region_step_exact_cho(
                    g_h, B_h, trust_radius, alpha
                )
            elif tr_method == "qr":
                step_h, hits_boundary, alpha = trust_region_step_exact_qr(
                    p_newton, Qt_La, R, trust_radius, alpha
                )

            step = d * step_h  # Trust-region solution in the original space.

            step, step_h, Lpredicted_reduction = select_step(
                z,
                J_h,
                diag_h,
                g_h,
                step,
                step_h,
                d,
                trust_radius,
                lb,
                ub,
                theta,
                mode="jac",
            )

            step_h_norm = jnp.linalg.norm(step_h, ord=2)
            step_norm = jnp.linalg.norm(step, ord=2)

            z_new = make_strictly_feasible(z + step, lb, ub, rstep=0)
            f_new = fun_wrapped(z_new, *args)
            cost_new = 0.5 * jnp.dot(f_new, f_new)
            c_new = constraint_wrapped.fun(z_new, *args)
            L_new = lagfun(f_new, c_new, y, mu)
            nfev += 1

            Lcost_new = 0.5 * jnp.dot(L_new, L_new)
            actual_reduction = cost - cost_new
            Lactual_reduction = Lcost - Lcost_new

            # update the trust radius according to the actual/predicted ratio
            tr_old = trust_radius
            trust_radius, Lreduction_ratio = update_tr_radius(
                trust_radius,
                Lactual_reduction,
                Lpredicted_reduction,
                step_h_norm,
                hits_boundary,
                max_trust_radius,
                tr_increase_threshold,
                tr_increase_ratio,
                tr_decrease_threshold,
                tr_decrease_ratio,
            )
            alltr.append(trust_radius)
            alpha *= tr_old / trust_radius

            success, message = check_termination(
                actual_reduction,
                cost,
                (step_h_norm if scaled_termination else step_norm),
                z_norm,
                g_norm,
                Lreduction_ratio,
                ftol,
                xtol,
                gtol,
                iteration,
                maxiter,
                nfev,
                max_nfev,
                min_trust_radius=min_trust_radius,
                dx_total=jnp.linalg.norm(z - z0),
                max_dx=max_dx,
                constr_violation=constr_violation,
                ctol=ctol,
            )
            if success is not None:
                # Only a genuine KKT exit ends the algorithm. `ftol`/`xtol` and a
                # collapsed trust region say this *subproblem* stopped improving,
                # which in the nested form of alg 14.4.2 is the cue to update the
                # multipliers and restart the inner solve -- not a global result.
                converged = (g_norm < gtol) and (constr_violation < ctol)
                if (
                    max_inner_stalls  # 0 restores the old single-loop behaviour
                    and not converged
                    and (success or message == STATUS_MESSAGES["approx"])
                ):
                    if n_inner_stalls < max_inner_stalls:
                        n_inner_stalls += 1
                        inner_stall = True
                        success, message = None, None
                    elif constr_violation < ctol:
                        # Feasible, and repeated subproblems have stalled without
                        # further progress, so the multiplier update has stopped
                        # buying anything. This is a KKT point to the accuracy the
                        # problem supports; `gtol` is simply below the noise floor of
                        # the objective. Say so rather than claiming failure.
                        success, message = True, STATUS_MESSAGES["precision"]
                    else:  # repeated stalls: report honestly, don't claim success
                        success, message = False, STATUS_MESSAGES["stall"]
                break

        # if reduction was enough, accept the step
        if Lactual_reduction > 0:
            # Only real progress clears the stall counter. Accepting a ~1e-16
            # reduction scraped from the bottom of a collapsed trust region is not
            # progress, and letting it reset the counter is what allowed a stalled
            # solve to spin for hundreds of iterations instead of stopping.
            if Lactual_reduction > ftol * Lcost:
                n_inner_stalls = 0
            z = z_new
            allx.append(z)
            f = f_new
            c = c_new
            constr_violation = constr_violation_fun(c, z)
            c_norm = jnp.linalg.norm(c, ord=jnp.inf)
            L = L_new
            cost = cost_new
            Lcost = Lcost_new
            J = lagjac(z, y, mu, *args)
            mu_J = mu
            njev += 1
            g = jnp.dot(J.T, L)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)
            v, dv = cl_scaling_vector(z, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )
        else:
            step_norm = step_h_norm = actual_reduction = 0

        # A subproblem is finished when it reaches `gtolk`, but *also* when it stalls:
        # `ftol`/`xtol` or a collapsed trust region mean the inner solve has nothing
        # left to give at this (y, mu). Either way alg 14.4.2 says to update the
        # multipliers/penalty and restart. Leaving (y, mu) untouched on a stall makes
        # the restart re-solve a bit-identical subproblem, which stalls identically --
        # an infinite loop that costs a full trust-region collapse (~25 evaluations)
        # per iteration and looks exactly like "the solver won't accept any steps".
        iters_since_update += 1

        al_changed = False
        force_update = False
        # A subproblem can also be finished without ever stalling: it can simply grind,
        # taking accepted steps too small to trip `ftol`/`xtol` while never getting
        # near `gtolk`. That is the same situation as a stall -- the inner solve has
        # nothing useful left to give at this (y, mu) -- but nothing detects it, so the
        # outer loop waits for a trust-region collapse that may be hundreds of
        # iterations away. Cap how long one subproblem may run before we accept
        # whatever optimality it reached and move the multipliers on.
        stagnated = (
            max_inner_iter is not None
            and iters_since_update >= max_inner_iter
            and g_norm > gtolk
        )
        if (inner_stall and constr_violation < ctol) or stagnated:
            # Feasible and out of road, so what is left in `g_norm` is dual error, not
            # primal: a better multiplier estimate fixes it, a tighter inner solve does
            # not. Firing is structural rather than a re-derived `g_norm <= gtolk`
            # comparison, so it cannot silently stop firing.
            force_update = True

        # updating augmented lagrangian params
        if force_update or (Lactual_reduction > 0 and g_norm <= gtolk):
            al_changed = True
            n_outer += 1
            outer_log.append((iteration, "stall" if force_update else "gtolk"))
            y = jnp.where(jnp.abs(c) < ctolk, y - mu * c, y)
            # safeguard: keep the multipliers in a box. y <- y - mu*c is only a valid
            # estimate at an approximate minimizer, so a subproblem that keeps ending
            # short can integrate a non-zero residual without bound. Clipping degrades
            # the method to a (still globally convergent) penalty method instead.
            y = jnp.clip(y, -max_multiplier, max_multiplier)
            # Grow `mu` only where the ORIGINAL constraint is actually violated. The
            # obvious test, `|c| >= ctolk`, uses the slack-reformulated residual
            # c(x) - s, which is non-zero merely because the slack variable lags c(x)
            # even at a strictly feasible point. Keying on it drove `mu` to 5e7 (and
            # |y| to 350) on a coil solve whose true violation was exactly 0 for every
            # iteration, which badly ill-conditions the subproblem and makes the
            # reported optimality meaningless.
            mu = jnp.where(constr_violation_rows(c, z) >= ctolk, tau * mu, mu)
            if c_norm < ctolk:
                ctolk = max(ctolk / (jnp.mean(mu) ** beta_eta), ctol)
            else:
                ctolk = max(eta / (jnp.mean(mu) ** alpha_eta), ctol)
        elif inner_stall:
            outer_log.append((iteration, "mu"))
            # Stalled while still infeasible. The subproblem is too hard at this
            # penalty, which is exactly what `mu` is for -- Conn & Gould's
            # unsuccessful branch. Raising it on the offending rows makes the restart
            # a genuinely different subproblem instead of the same one again. `y` is
            # deliberately left alone: `y <- y - mu*c` is only a valid estimate at an
            # approximate minimizer, and this point is not one.
            al_changed = True
            mu = jnp.where(constr_violation_rows(c, z) >= ctolk, tau * mu, mu)
            ctolk = max(eta / (jnp.mean(mu) ** alpha_eta), ctol)

        if al_changed:
            # (y, mu) changed, so L, its Jacobian, and every scaling derived from them
            # are stale
            L = lagfun(f, c, y, mu)
            Lcost = 0.5 * jnp.dot(L, L)
            # `lagjac` ignores `y` and depends on `mu` only through the sqrt(mu) row
            # scaling of the constraint block, and `z` has not moved since `J` was
            # built. Re-evaluating here would recompute the objective and constraint
            # Jacobians purely to re-apply a diagonal. Rescaling is exact and removes
            # one Jacobian per outer update: measured `njev = nit + n_outer`, so on a
            # coil solve with frequent updates this was ~35% of all Jacobian work.
            J = scale_tail_rows(J, jnp.sqrt(mu / mu_J), f.size)
            mu_J = mu
            g = jnp.dot(J.T, L)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(J, scale_inv)

            v, dv = cl_scaling_vector(z, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )
            # (y, mu) changed, so this is a new subproblem
            iters_since_update = 0
            # Set its target RELATIVE to where it actually starts, rather than from the
            # mu-driven schedule of alg 14.4.2. That schedule cannot work here in either
            # direction: driven by mean(mu) it reaches `gtol` after ~3 updates and the
            # outer loop dies (no more multiplier updates for the rest of the run),
            # while floored at a measured `g_norm` it latches onto the high excursions
            # of a signal that swings an order of magnitude between iterations near
            # convergence, so the target is met immediately and the outer loop churns --
            # updating `y` every iteration or two at points that are not subproblem
            # minimizers, where `y <- y - mu*c` is not a contraction and the multipliers
            # cycle instead of converging. A fixed relative reduction is reachable by
            # construction and always asks for real inner work.
            gtolk = max(float(g_norm) / inner_reduction, gtol)

        # `J_h`/`g_h`/`d`/`diag_h` define the next trust-region subproblem, so they
        # must be rebuilt whenever `z` changed *or* (y, mu) changed. Rebuilding only
        # on an accepted step left an outer update that fired on a rejected step
        # pointing at the Jacobian of the previous augmented Lagrangian.
        if Lactual_reduction > 0 or al_changed:
            z_norm = jnp.linalg.norm(
                ((z * scale_inv) if scaled_termination else z), ord=2
            )
            d = v**0.5 * scale
            diag_h = g * dv * scale
            g_h = g * d
            # The unscaled J is kept so that an outer update at an unchanged
            # `z` can rescale it instead of re-evaluating (see `mu_J`).
            J_h = J * d

            if g_norm < gtol and constr_violation < ctol:
                success, message = True, STATUS_MESSAGES["gtol"]

            if callback(jnp.copy(z2xs(z)[0]), *args):
                success, message = False, STATUS_MESSAGES["callback"]
        if inner_stall:
            # The subproblem stalled with a collapsed radius. `(y, mu)` were just
            # updated above, so the augmented Lagrangian being minimized has actually
            # changed and a fresh radius gives the new subproblem room to move.
            trust_radius = init_trust_radius
            alpha = 0.0

        iteration += 1
        allf_max.append(jnp.max(jnp.abs(f)))
        allf_mean.append(jnp.mean(jnp.abs(f)))
        if verbose > 1:
            print_iteration_nonlinear(
                iteration,
                nfev,
                cost,
                actual_reduction,
                step_norm,
                g_norm,
                constr_violation,
                jnp.mean(mu),
                jnp.max(jnp.abs(y)),
                gtolk,
                *((allf_max[-1], allf_mean[-1]) if track_residual else ()),
            )

    if g_norm < gtol and constr_violation < ctol:
        success, message = True, STATUS_MESSAGES["gtol"]
    if (iteration == maxiter) and success is None:
        success, message = False, STATUS_MESSAGES["maxiter"]
    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    result = OptimizeResult(
        x=x,
        s=s,
        y=y,
        penalty_param=mu,
        success=success,
        cost=cost,
        fun=f,
        grad=g,
        v=v,
        jac=J_h * 1 / d,  # after overwriting J_h, we have to revert back,
        optimality=g_norm,
        nfev=nfev,
        njev=njev,
        nit=iteration,
        n_outer=n_outer,
        outer_log=outer_log,
        message=message,
        active_mask=active_mask,
        constr_violation=constr_violation,
        allx=[z2xs(x)[0] for x in allx],
        alltr=alltr,
        allf_max=jnp.asarray(allf_max),
        allf_mean=jnp.asarray(allf_mean),
    )
    result["fse"] = f
    result["f0se"] = f0
    if verbose > 0:
        if result["success"]:
            print(result["message"])
        else:
            print("Warning: " + result["message"])
        print(f"""         Current function value: {result["cost"]:.3e}""")
        print(f"""         Constraint violation: {result['constr_violation']:.3e}""")
        print(f"""         Total delta_x: {jnp.linalg.norm(x0 - result["x"]):.3e}""")
        print(f"""         Iterations: {result["nit"]:d}""")
        print(f"""         Function evaluations: {result["nfev"]:d}""")
        print(f"""         Jacobian evaluations: {result["njev"]:d}""")

    return result
