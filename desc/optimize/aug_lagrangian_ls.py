"""Augmented Lagrangian for vector valued objectives."""

import numpy as np
from scipy.optimize import NonlinearConstraint, OptimizeResult

from desc.backend import jnp, put, qr, qr_multiply
from desc.utils import errorif, safediv, scale_tail_rows, setdefault

from .bound_utils import (
    cl_scaling_vector,
    find_active_constraints,
    in_bounds,
    make_strictly_feasible,
    select_step,
    select_step_diag,
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
    scale_columns,
    solve_triangular_regularized,
)


def _trust_region_step_eigh(g, eig, trust_radius, rtol=1e-7, max_iter=400):
    """Exact trust-region step for a symmetric, possibly INDEFINITE model Hessian.

    Solves  min_p  g.p + 1/2 p.B.p  subject to ||p|| <= trust_radius  from the
    eigendecomposition ``eig = (lam, V)`` of ``B`` (lam ascending), following Moré &
    Sorensen (Nocedal & Wright, sec. 4.3): the solution satisfies (B + alpha I) p = -g
    with alpha >= max(0, -lam_min) and alpha (||p|| - trust_radius) = 0, including the
    "hard case" where g has no component along the lowest eigenvector(s).

    ``trust_region_step_exact_cho`` assumes a Cholesky factorization exists, which fails
    for an indefinite B; the second-order augmented-Lagrangian model (``second_order``
    option of ``lsq_auglag``) can be indefinite, so it uses this instead. The
    decomposition is made once per model (on the device, ``jnp.linalg.eigh``) and reused
    for every trust radius tried.

    Where the work happens: ``V`` stays on the device. Only ``lam`` and ``V.T g`` (two
    length-n vectors) come to the host, where the scalar search over alpha runs; norms
    are taken from the coefficients, since ``||V c|| = ||c||``; the step ``V c`` is
    formed back on the device. MEASURED before this split (2026-09-16): with the
    decomposition and the step on the host, precise_QH arcB3 ran 1.8x SLOWER on an RTX
    5070 than on the CPU.

    Returns
    -------
    p : ndarray
        Step.
    hits_boundary : bool
        True if the step lies on the trust-region boundary.
    alpha : float
        The multiplier of the norm constraint (Levenberg-Marquardt parameter).

    """
    lam_d, V = eig
    V = jnp.asarray(V)
    g = jnp.asarray(g)
    gt = np.asarray(V.T @ g)
    lam = np.asarray(lam_d)
    lam_min = lam[0]
    lam_scale = max(float(np.abs(lam).max()), np.finfo(float).tiny)
    g_norm = float(np.linalg.norm(gt))

    def step(coef):
        return V @ jnp.asarray(coef)

    if g_norm == 0.0:
        return jnp.zeros_like(g), False, 0.0

    # interior (Newton) solution, only valid if B is positive definite
    if lam_min > 1e-12 * lam_scale:
        coef = -gt / lam
        if np.linalg.norm(coef) <= trust_radius:
            return step(coef), False, 0.0

    alpha_lo = max(0.0, -float(lam_min))

    # hard case: g (numerically) orthogonal to the lowest eigenspace, and the step built
    # from the remaining eigenvectors at alpha = -lam_min is still inside the region
    if lam_min < 0:
        near = lam <= lam_min + 1e-10 * lam_scale
        if np.linalg.norm(gt[near]) <= 1e-10 * g_norm:
            denom = np.where(near, np.inf, lam - lam_min)
            coef = -gt / denom
            p0 = float(np.linalg.norm(coef))
            if p0 <= trust_radius:
                coef[np.nonzero(near)[0][0]] = np.sqrt(
                    max(trust_radius**2 - p0**2, 0.0)
                )
                return step(coef), True, alpha_lo

    # boundary solution: bisect (geometrically) on t = alpha - alpha_lo so ||p|| =
    # radius. ||p(alpha)|| <= ||g|| / (lam_min + alpha) makes t_hi = ||g||/radius
    # feasible.
    def norm_at(t):
        return float(np.linalg.norm(gt / (lam + alpha_lo + t)))

    t_hi = g_norm / trust_radius
    t_lo = max(t_hi * 1e-16, np.finfo(float).tiny)
    for _ in range(max_iter):
        t = np.sqrt(t_lo * t_hi)
        n = norm_at(t)
        if abs(n - trust_radius) <= rtol * trust_radius:
            t_hi = t
            break
        if n > trust_radius:
            t_lo = t
        else:
            t_hi = t
    alpha = alpha_lo + t_hi
    coef = -gt / (lam + alpha)
    c_norm = float(np.linalg.norm(coef))
    if c_norm > 0:
        coef *= trust_radius / c_norm
    return step(coef), True, alpha


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
        - ``"y_update_gate"`` : (``"residual"`` or ``"violation"``) Which rows are
          eligible for the multiplier update ``y <- y - mu*c``. ``"residual"``
          (default, historical) gates on ``|c| < ctolk``, where ``c`` is the
          slack-reformulated residual ``c(x) - s``. That quantity is NOT the
          violation: at an interior slack it equals ``y/mu``, so the gate reads
          ``|y|/mu < ctolk`` and closes on precisely the rows carrying a non-zero
          multiplier -- the method degrades to a quadratic penalty there, where
          feasibility is floored at ``~|y*|/mu`` and only ``mu`` growth lowers it.
          ``"violation"`` gates on the true per-row violation instead, the same
          quantity the ``mu`` update already uses. EXPERIMENTAL.
        - ``"min_window_progress"`` : (float > 0 or None) EXPERIMENTAL, default None
          (off, unconditional growth). Withhold the ``mu`` increase at an outer
          update when the subproblem just ended reduced its augmented-Lagrangian
          cost by more than this relative amount. Growing ``mu`` is the escape from
          a FROZEN subproblem, but the stagnation cap (``max_inner_iter``) trips on
          elapsed iterations regardless of cause, so it also fires on subproblems
          that are advancing normally -- and there the 10x bump only ill-conditions
          the next solve, since ``sqrt(mu)`` scales the constraint block of ``J``.
          Measured on a cold arc coil solve: an update that fired mid-progress took
          ``mu`` 40 -> 313 and collapsed the step norm from 5.6e-03 to 3.1e-05,
          after which nothing recovered. The multiplier update and the tolerance
          ratchet still fire, so the outer cadence is unchanged.
        - ``"track_outer"`` : (bool) Record per-outer-update diagnostics in
          ``result["outer_diag"]``: ``mu``, ``|y|``, the multiplier change, and how
          many rows each gate admits. Default False.
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
          ``"svd"`` is the most accurate but significantly slower. If the Jacobian
          can become rank-deficient due to the bounds, ``"svd"`` is recommended and
          the default, otherwise the default is ``"qr"``.
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
    # Cap on the penalty parameter. With a stagnation detector enabled, `mu` can be
    # grown on every trip, and tau**n diverges fast; past ~1e8 the augmented Lagrangian
    # is so ill-conditioned that the Gauss-Newton model is meaningless anyway.
    max_penalty = options.pop("max_penalty_parameter", 1e8)
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
    # Reject a trial step on a caller-supplied predicate, BEFORE it is
    # evaluated. `step_veto(x_new, x_old) -> bool` (True = reject) sees the
    # iterates in the same space the `callback` does. Motivation: the coil
    # optimization can change the TOPOLOGY of the coilset -- one coil passing
    # through another -- and no term in the objective can see it. Linking
    # number is integer-valued (zero gradient a.e., unreadable at contact),
    # and coil-coil distance is a smooth quantity that says nothing about
    # topology at the ENDPOINTS of a step. But linking number is a homotopy
    # invariant, so it cannot change unless the curves intersect: a guard that
    # keeps d_cc bounded away from zero ALONG THE PATH is sufficient, and that
    # is what a step predicate can enforce and a residual block cannot.
    # NOTE this is a hard wall: the trust-region model does not know about it,
    # so a rejected step is re-proposed in the same direction on a smaller
    # radius. See `n_veto` in the return value for whether that stalls.
    step_veto = options.pop("step_veto", None)
    # `compute_jac_scale(J, scale_inv)` takes maximum(column_norms, prev_scale_inv), so
    # each variable's scale only ever SHRINKS: one column-norm spike throttles that
    # variable's trust-region allowance for the rest of the solve, even after the
    # Jacobian settles. That is faithful to MINPACK/scipy `x_scale="jac"`, and it is
    # the right default, but it is not free. MEASURED start->final column-norm growth:
    # median 0.93 / max 10.7 for an arc run that converged on `gtol` (1 of 232 columns
    # above 10x, none above 100x) versus median 2.23 / max 959 for a FourierPlanar run
    # that stalled with steps at 1e-07 while 29% of its objective gradient still lay in
    # feasible directions (47 of 192 columns above 10x, 13 above 100x).
    #
    # `jac_scale_ratchet=False` recomputes the scale fresh from the current Jacobian, so
    # a variable recovers once its column norm settles. NOT a substitute for scaling
    # altogether: `x_scale=1` on that same planar case collapsed the trust region by
    # iteration 13 with no constraint active, because FourierPlanar mixes metre-scale
    # `center`/`r_n` with a dimensionless `normal`.
    jac_scale_ratchet = options.pop("jac_scale_ratchet", True)

    # `second_order="constraints"`: model the augmented Lagrangian with H = J^T J + S,
    # S = sum_i L_c,i * d2 L_c,i = d2_x [ w . c(x) ],   w = mu*c - y instead of
    # Gauss-Newton J^T J. Each constraint row of the least-squares form is L_c =
    # sqrt(mu) c - y/sqrt(mu), so the term Gauss-Newton drops is (mu c - y) d2c; slacks
    # enter linearly and contribute nothing. Objective rows keep Gauss-Newton (their
    # dropped term f d2f is small once the fit is good). Why: a degenerate constraint --
    # e.g. the C^1 coil-distance penalty, whose gradient vanishes like the penetration
    # while its curvature does not -- holds a finite multiplier estimate y through a
    # vanishing gradient, so J^T J (which scales with the gradient) is dwarfed by the
    # dropped y*d2c. MEASURED on precise_QH (paper surface): with max|y| >= 3e-4 on an
    # active penalty row the Gauss-Newton model predicted only ~60-70% of the actual
    # reduction at every step size (median ratio 0.70, 55% of steps in the 0.5-0.75 band
    # where the trust radius never moves), stalling the solve; with |y| < 1e-5, median
    # ratio 1.01. `constraint_hess(x, w)` must return the dense n_x-by-n_x Hessian of
    # w.c at x (the lsq wrapper builds one from the constraint objective). The model is
    # then indefinite in general, so the step uses `_trust_region_step_eigh`, whatever
    # `tr_method` says.
    second_order = options.pop("second_order", None)
    constraint_hess = options.pop("constraint_hess", None)
    errorif(
        second_order not in (None, False, "constraints"),
        ValueError,
        f'second_order must be None or "constraints", got {second_order}',
    )
    second_order = bool(second_order)
    errorif(
        second_order and constraint_hess is None,
        ValueError,
        'second_order="constraints" needs a constraint_hess(x, w) callable',
    )

    def _second_order_S(z, c, y, mu):
        """d2_z of sum_i (mu_i c_i - y_i) c_i(z): the x block of it, zeros on slacks."""
        x = z2xs(z)[0]
        w = mu * c - y
        Sx = jnp.asarray(constraint_hess(x, w))
        Sx = 0.5 * (Sx + Sx.T)
        return jnp.zeros((z.size, z.size)).at[: x.size, : x.size].set(Sx)

    S = _second_order_S(z, c, y, mu) if second_order else None
    H_h = None
    so_lam_min = float("nan")  # smallest eigenvalue of the last second-order model

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
    track_outer = options.pop("track_outer", False)
    y_update_gate = options.pop("y_update_gate", "residual")
    min_window_progress = options.pop("min_window_progress", None)
    # Step-level diagnostics, all opt-in (see the setup block below `ctolk`).
    diag_log = options.pop("diag_log", None)  # path of a JSONL file; None = off
    diag_watch = options.pop("diag_watch", None)  # slack indices to watch; None = auto
    diag_blocks = options.pop("diag_blocks", None)  # [(name, n_rows), ...] row labels
    diag_n_auto = int(options.pop("diag_n_auto", 5))  # nearest-to-bound slacks watched
    # per-block model check of every trial from this iteration on; None = off
    diag_block_split_from = options.pop("diag_block_split_from", None)
    # constraint_quad(x, w, v) -> [v . d2_x(w_k . c_k) . v per constraint block]
    constraint_quad = options.pop("constraint_quad", None)

    errorif(
        len(options) > 0,
        ValueError,
        "Unknown options: {}".format([key for key in options]),
    )
    errorif(
        y_update_gate not in ["residual", "violation"],
        ValueError,
        "y_update_gate should be 'residual' or 'violation', got {}".format(
            y_update_gate
        ),
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
    n_veto = 0  # trial steps rejected by `step_veto`, total over the solve
    step_norm = jnp.inf
    actual_reduction = jnp.inf
    Lactual_reduction = jnp.inf
    alpha = 0.0  # "Levenberg-Marquardt" parameter
    n_inner_stalls = 0  # consecutive subproblem stalls with no accepted step
    iters_since_update = 0  # inner iterations spent on the current subproblem
    # Progress made by the CURRENT subproblem, for the `mu`-growth gate. `Lcost` is
    # what the inner solve minimizes and (y, mu) are fixed across a window, so it is
    # comparable within one and non-increasing (steps need Lactual_reduction > 0).
    Lcost_window_start = Lcost
    window_max_step = 0.0
    n_outer = 0  # number of augmented Lagrangian (y, mu) updates taken
    # (iteration, reason) for every outer update, so update cadence is recoverable
    # from the result object rather than only from a verbose log
    outer_log = []
    # Per-outer-update diagnostics (only when `track_outer`). The decisive column
    # is `n_blocked`: rows whose ORIGINAL constraint is satisfied to within
    # `ctolk` -- so they are due a multiplier update -- but which the residual
    # gate `|c| < ctolk` rejects. At an interior slack `c == y/mu`, so that gate
    # reads `|y|/mu < ctolk` and shuts off exactly the rows carrying force.
    outer_diag = []

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

    # ---- step-level diagnostics, opt-in via `diag_log`
    # -------------------------------- One JSONL record per TRIAL step, per ACCEPTED
    # step and per OUTER update, to see how the solve behaves once constraints become
    # active: the trust-radius and model-ratio history, whether a step was truncated at
    # a slack bound (and by which slack), where the scaled gradient norm lives (x block
    # vs slacks), and -- for the slacks nearest their bounds -- the multiplier estimate
    # g[j] = y[i] - mu[i]*(c_i - s) against the unconstrained slack optimum s_free =
    # c_i(x) - y[i]/mu[i]. Off by default. When off, none of this is evaluated and every
    # step is unchanged.
    diag_on = diag_log is not None
    _diag_state = {"watch": [], "prev_sign": None, "n_flipped": 0}
    if diag_on:
        import json as _json

        import numpy as _np

        _dx = int(x0.size)
        _nslack = int(z0.size - x0.size)
        _ineq_idx = _np.nonzero(_np.asarray(_ineq))[0]  # slack k <-> row _ineq_idx[k]
        _labels = [f"row{_r}" for _r in range(int(c.size))]
        _labels_ok = False
        if diag_blocks:
            _lab = [
                f"{_nm}[{_q}]" for _nm, _dim in diag_blocks for _q in range(int(_dim))
            ]
            if len(_lab) == int(c.size):
                _labels, _labels_ok = _lab, True
        _diag_fh = open(diag_log, "a")

        import time as _time

        def _diag_write(rec):
            rec.setdefault(
                "t", _time.time()
            )  # wall clock, for compile vs steady-state timing
            _diag_fh.write(_json.dumps(rec, default=float) + "\n")
            _diag_fh.flush()

        def _slack_label(k):
            return f"s{int(k)}:{_labels[int(_ineq_idx[int(k)])]}"

        # Per-block model check (diag_block_split_from). For each trial, split the
        # actual and predicted change of 0.5|L|^2 by row block, at fractions t of the
        # step. `gn` is the Gauss-Newton prediction; `so` is the second-order term of a
        # constraint block, -0.5 t^2 v.d2(w_b.c_b).v with w = mu c - y, whether or not
        # the model used it. For a smooth block, act - gn - so shrinks like t^3; a kink
        # inside the step shows as t^2 or t. Diagnostic evaluations only: nfev and the
        # iterate are untouched.
        _split_t = (0.25, 0.5, 1.0)
        _nf = int(f.size)
        _blk = [("f", 0, _nf)]
        if _labels_ok:
            _o = _nf
            for _nm, _dim in diag_blocks:
                _blk.append((_nm, _o, _o + int(_dim)))
                _o += int(_dim)

        def _block_split(z, step, L, L_new, J, c, y, mu):
            Jp = _np.asarray(J @ step)
            Ls = {1.0: _np.asarray(L_new)}
            for t in _split_t[:-1]:
                zt = z + t * step
                Ls[t] = _np.asarray(
                    lagfun(
                        fun_wrapped(zt, *args), constraint_wrapped.fun(zt, *args), y, mu
                    )
                )
            L0 = _np.asarray(L)
            w = _np.asarray(mu * c - y)
            quad = None
            if constraint_quad is not None:
                quad = _np.asarray(
                    constraint_quad(z2xs(z)[0], jnp.asarray(w), step[:_dx])
                )
            out = []
            for b, (nm, a, e) in enumerate(_blk):
                h0 = 0.5 * float(L0[a:e] @ L0[a:e])
                lj, jj = float(L0[a:e] @ Jp[a:e]), float(Jp[a:e] @ Jp[a:e])
                rec = dict(
                    name=nm,
                    act=[h0 - 0.5 * float(Ls[t][a:e] @ Ls[t][a:e]) for t in _split_t],
                    gn=[-(t * lj + 0.5 * t * t * jj) for t in _split_t],
                )
                if b > 0:
                    wb = w[a - _nf : e - _nf]
                    rec["w_absmax"] = float(_np.abs(wb).max()) if wb.size else 0.0
                    if quad is not None and quad.size == len(_blk) - 1:
                        rec["so"] = [
                            -0.5 * t * t * float(quad[b - 1]) for t in _split_t
                        ]
                out.append(rec)
            return dict(t=list(_split_t), blocks=out)

        def _z_label(q):
            return f"x{int(q)}" if q < _dx else _slack_label(q - _dx)

        def _diag_watch_set():
            """Watched slacks: `diag_watch` (if given), nearest-to-bound, sign flips."""
            manual = [int(k) for k in (diag_watch if diag_watch is not None else [])]
            s_all = _np.asarray(z[_dx:])
            dmin = _np.minimum(
                s_all - _np.asarray(lb[_dx:]), _np.asarray(ub[_dx:]) - s_all
            )
            near = [int(k) for k in _np.argsort(dmin)[:diag_n_auto]]
            flipped = []
            if _diag_state["prev_sign"] is not None:
                sgn = _np.sign(_np.asarray(g[_dx:]))
                flipped = [
                    int(k) for k in _np.nonzero(sgn != _diag_state["prev_sign"])[0]
                ]
            _diag_state["n_flipped"] = len(flipped)
            return sorted(set(manual) | set(near) | set(flipped[:20]))

        def _diag_scale_summary():
            """Held (possibly ratcheted) scale_inv vs a fresh column-norm scale (x)."""
            if not jac_scale:
                return {}
            fresh = jnp.sqrt(jnp.sum(J**2, axis=0))
            fresh = jnp.where(fresh < jnp.finfo(J.dtype).eps * max(J.shape), 1, fresh)
            rr = _np.asarray(scale_inv)[:_dx] / _np.asarray(fresh)[:_dx]
            return dict(
                ratchet_ratio_median=float(_np.median(rr)),
                ratchet_ratio_max=float(rr.max()),
                ratchet_ratio_argmax=int(_np.argmax(rr)),
                ratchet_n_gt2=int((rr > 2).sum()),
                ratchet_n_gt10=int((rr > 10).sum()),
            )

        def _diag_accept():
            """Record for an accepted step; call after g, v, g_norm are recomputed."""
            comp = _np.asarray(
                jnp.abs(g * v * scale) if scaled_termination else jnp.abs(g * v)
            )
            dvn = _np.asarray(dv)
            am = int(_np.argmax(comp))
            comp_s, dv_s = comp[_dx:], dvn[_dx:]
            pg = _np.abs(_np.asarray(jnp.clip(z - g, lb, ub) - z))
            d_now = _np.asarray(v**0.5 * scale)
            dh_now = _np.asarray(g * dv * scale)
            watch = _diag_watch_set()
            _diag_state["watch"] = watch
            wrec = {}
            for k in watch:
                j, i = _dx + k, int(_ineq_idx[k])
                s_k = float(z[j])
                cms = float(c[i])  # wrapped residual c_i(x) - s at row i
                c_x = cms + s_k  # the constraint value itself (no target on ineq rows)
                yhat = float(g[j])
                s_free = c_x - float(y[i]) / float(mu[i])
                wrec[_slack_label(k)] = dict(
                    s=s_k,
                    d_lb=s_k - float(lb[j]),
                    d_ub=float(ub[j]) - s_k,
                    c_minus_s=cms,
                    yhat=yhat,
                    yhat_check=float(y[i] - mu[i] * c[i]),
                    sign_yhat=float(_np.sign(yhat)),
                    v=float(v[j]),
                    dv=float(dv[j]),
                    d=float(d_now[j]),
                    diag_h=float(dh_now[j]),
                    scale=float(scale[j]),
                    comp=float(comp[j]),
                    s_free=s_free,
                    s_free_minus_lb=s_free - float(lb[j]),  # < 0: wants the lower bound
                    ub_minus_s_free=float(ub[j]) - s_free,  # < 0: wants the upper bound
                    y=float(y[i]),
                    mu=float(mu[i]),
                )
            _diag_write(
                dict(
                    kind="accept",
                    iteration=int(iteration),
                    nfev=int(nfev),
                    g_norm=float(g_norm),
                    gtolk=float(gtolk),
                    ctolk=float(ctolk),
                    constr_violation=float(constr_violation),
                    mu_mean=float(jnp.mean(mu)),
                    y_absmax=float(jnp.max(jnp.abs(y))),
                    comp_argmax=_z_label(am),
                    comp_argmax_is_slack=bool(am >= _dx),
                    comp_max_x=float(comp[:_dx].max()),
                    comp_max_slack_bound_active=(
                        float(comp_s[dv_s != 0].max()) if _np.any(dv_s != 0) else None
                    ),
                    comp_max_slack_unbounded=(
                        float(comp_s[dv_s == 0].max()) if _np.any(dv_s == 0) else None
                    ),
                    proj_grad_inf=float(pg.max()),
                    proj_grad_inf_x=float(pg[:_dx].max()),
                    proj_grad_inf_slack=float(pg[_dx:].max()) if _nslack else None,
                    n_sign_flips=int(_diag_state["n_flipped"]),
                    **_diag_scale_summary(),
                    **(
                        dict(
                            so_S_fro=float(jnp.linalg.norm(S)),
                            so_JtJ_x_fro=float(
                                jnp.linalg.norm(jnp.dot(J[:, :_dx].T, J[:, :_dx]))
                            ),
                            so_lam_min_last=float(so_lam_min),
                        )
                        if second_order
                        else {}
                    ),
                    watched=wrec,
                )
            )
            _diag_state["prev_sign"] = _np.sign(_np.asarray(g[_dx:]))

        _diag_state["watch"] = _diag_watch_set()
        _diag_write(
            dict(
                kind="setup",
                n_x=_dx,
                nslack=_nslack,
                n_rows=int(c.size),
                labels_ok=_labels_ok,
                blocks=[[str(_nm), int(_dim)] for _nm, _dim in (diag_blocks or [])],
                watch=(
                    f"manual {sorted(int(k) for k in (diag_watch or []))}"
                    f" + auto: {diag_n_auto} nearest-to-bound + sign flips"
                ),
                watch_initial=[_slack_label(k) for k in _diag_state["watch"]],
                scaled_termination=bool(scaled_termination),
                tr_method=str(tr_method),
                init_trust_radius=float(trust_radius),
                max_trust_radius=float(max_trust_radius),
                tr_increase_threshold=float(tr_increase_threshold),
                tr_decrease_threshold=float(tr_decrease_threshold),
                tr_increase_ratio=float(tr_increase_ratio),
                tr_decrease_ratio=float(tr_decrease_ratio),
                gtolk=float(gtolk),
                ctolk=float(ctolk),
                mu_mean=float(jnp.mean(mu)),
            )
        )

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

        if second_order:
            # full model in scaled variables; diag_h is the Coleman-Li bound term, which
            # J_a already carries as extra rows in the Gauss-Newton path
            H_h = jnp.dot(J_h.T, J_h) + d[:, None] * S * d[None, :]
            B_h = H_h + jnp.diag(diag_h) if bounded else H_h
            # on the device; _trust_region_step_eigh brings only two vectors to the host
            eig_h = jnp.linalg.eigh(B_h)
            so_lam_min = float(eig_h[0][0])
        elif tr_method == "svd":
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
            if second_order:
                step_h, hits_boundary, alpha = _trust_region_step_eigh(
                    g_h, eig_h, trust_radius
                )
            elif tr_method == "svd":
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

            _alpha_sub = alpha  # the LM parameter this trial was solved with
            if diag_on:
                # identical step; also returns why it was chosen (see select_step_diag)
                step, step_h, Lpredicted_reduction, _sel = select_step_diag(
                    z,
                    H_h if second_order else J_h,
                    diag_h,
                    g_h,
                    step,
                    step_h,
                    d,
                    trust_radius,
                    lb,
                    ub,
                    theta,
                    mode="hess" if second_order else "jac",
                )
            else:
                step, step_h, Lpredicted_reduction = select_step(
                    z,
                    H_h if second_order else J_h,
                    diag_h,
                    g_h,
                    step,
                    step_h,
                    d,
                    trust_radius,
                    lb,
                    ub,
                    theta,
                    mode="hess" if second_order else "jac",
                )

            step_h_norm = jnp.linalg.norm(step_h, ord=2)
            step_norm = jnp.linalg.norm(step, ord=2)

            z_new = make_strictly_feasible(z + step, lb, ub, rstep=0)

            if step_veto is not None and step_veto(
                jnp.copy(z2xs(z_new)[0]), jnp.copy(z2xs(z)[0])
            ):
                # Unsafe step. Reuse the ordinary rejection path: shrink the radius
                # and re-solve. `nfev` is incremented so a veto that never clears
                # cannot spin forever -- it exits the inner solve on max_nfev and
                # hands control to the outer (y, mu) update, which resets the radius.
                n_veto += 1
                tr_old = trust_radius
                trust_radius = max(trust_radius * tr_decrease_ratio, min_trust_radius)
                alpha *= tr_old / trust_radius
                alltr.append(trust_radius)
                nfev += 1
                Lactual_reduction = -1
                if verbose > 2:
                    print(
                        f"  step vetoed ({n_veto}), |step| {step_norm:.3e}, "
                        f"tr {tr_old:.3e} -> {trust_radius:.3e}"
                    )
                continue

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

            _split = None
            if (
                diag_on
                and diag_block_split_from is not None
                and iteration >= diag_block_split_from
            ):
                _split = _block_split(z, step, L, L_new, J, c, y, mu)

            if diag_on:
                _hit_idx = _np.nonzero(_np.asarray(_sel["hits"]))[0]
                _diag_write(
                    dict(
                        kind="trial",
                        iteration=int(iteration),
                        nfev=int(nfev),
                        trust_radius=float(tr_old),
                        trust_radius_new=float(trust_radius),
                        alpha=float(_alpha_sub),
                        hits_boundary=bool(hits_boundary),
                        in_bounds_full=bool(_sel["in_bounds_full"]),
                        p_stride=float(_sel["p_stride"]),
                        hits_idx=[_z_label(q) for q in _hit_idx[:10]],
                        n_hits=int(_hit_idx.size),
                        selected=int(_sel["selected"]),
                        p_value=float(_sel["p_value"]),
                        r_value=float(_sel["r_value"]),
                        ag_value=float(_sel["ag_value"]),
                        step_h_norm=float(step_h_norm),
                        step_norm=float(step_norm),
                        step_h_over_tr=(
                            float(step_h_norm / tr_old) if tr_old > 0 else None
                        ),
                        step_h_norm_x=float(jnp.linalg.norm(step_h[:_dx])),
                        step_h_norm_slack=float(jnp.linalg.norm(step_h[_dx:])),
                        Lpredicted_reduction=float(Lpredicted_reduction),
                        Lactual_reduction=float(Lactual_reduction),
                        Lreduction_ratio=float(Lreduction_ratio),
                        actual_reduction=float(actual_reduction),
                        accepted=bool(Lactual_reduction > 0),
                        block_split=_split,
                        watched={
                            _slack_label(k): dict(
                                step=float(step[_dx + k]),
                                d_lb=float(z[_dx + k] - lb[_dx + k]),
                                d_ub=float(ub[_dx + k] - z[_dx + k]),
                            )
                            for k in _diag_state["watch"]
                        },
                    )
                )

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
            # Delete old arrays before computing new one
            # otherwise the peak is bigger by J-sized arrays
            del J_h, J_a
            if second_order:  # built instead of the tr_method factorization
                del B_h, eig_h
            elif tr_method == "svd":
                del U, s, Vt
            elif tr_method == "cho":
                del B_h
            elif tr_method == "qr":
                del R
            J = lagjac(z, y, mu, *args)
            mu_J = mu
            njev += 1
            g = jnp.dot(L, J)
            if second_order:
                S = _second_order_S(z, c, y, mu)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(
                    J, scale_inv if jac_scale_ratchet else None
                )
            v, dv = cl_scaling_vector(z, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )
            if diag_on:
                _diag_accept()
        else:
            step_norm = step_h_norm = actual_reduction = 0

        # A subproblem is finished when it reaches `gtolk`, but *also* when it stalls:
        # `ftol`/`xtol` or a collapsed trust region mean the inner solve has nothing
        # left to give at this (y, mu). Either way alg 14.4.2 says to update the
        # multipliers/penalty and restart. Leaving (y, mu) untouched on a stall makes
        # the restart re-solve a bit-identical subproblem, which stalls identically --
        # an infinite loop that costs a full trust-region collapse (~25 evaluations)
        # per iteration and looks exactly like "the solver won't accept any steps".
        if jnp.isfinite(step_norm):
            window_max_step = max(window_max_step, float(step_norm))
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
        # `check_termination` can fail to fire indefinitely: steps are nominally
        # accepted with a ~1e-30 reduction, so no tolerance test and no minimum-radius
        # test ever trips, and the iterate freezes with nothing detecting it. Trip on
        # elapsed inner iterations regardless of why.
        stagnated = max_inner_iter is not None and iters_since_update >= max_inner_iter
        # NB `ctolk`, the current working tolerance, not `ctol`, the final target.
        # Alg 14.4.2 branches on the working tolerance: with `ctol` here the
        # multipliers can never update while infeasible, so an infeasible start has
        # no path forward -- measured 811 consecutive frozen iterations.
        if (inner_stall and constr_violation < ctolk) or stagnated:
            # Feasible and out of road, so what is left in `g_norm` is dual error, not
            # primal: a better multiplier estimate fixes it, a tighter inner solve does
            # not. Firing is structural rather than a re-derived `g_norm <= gtolk`
            # comparison, so it cannot silently stop firing.
            force_update = True

        # Relative reduction the current subproblem has achieved so far. `mu` growth
        # is the escape from a FROZEN subproblem (see the 811-iteration freeze that
        # motivated `max_inner_iter`), but the stagnation cap trips on elapsed count
        # "regardless of cause", so it also fires on subproblems that are advancing
        # perfectly well -- and there a 10x penalty bump only ill-conditions the
        # solve. Measure whether this window actually stagnated, and let the caller
        # withhold the bump when it did not.
        window_rel_progress = float(
            (Lcost_window_start - Lcost)
            / jnp.maximum(jnp.abs(Lcost_window_start), jnp.finfo(L.dtype).tiny)
        )
        mu_growth_allowed = (
            min_window_progress is None or window_rel_progress < min_window_progress
        )

        if diag_on:  # snapshot for the outer-update record written after gtolk is reset
            _pre = dict(
                mu_mean=float(jnp.mean(mu)),
                mu_max=float(jnp.max(mu)),
                gtolk=float(gtolk),
                ctolk=float(ctolk),
                g_norm=float(g_norm),
                y=_np.asarray(y),
                mu=_np.asarray(mu),
            )
        # updating augmented lagrangian params
        if force_update or (Lactual_reduction > 0 and g_norm <= gtolk):
            al_changed = True
            n_outer += 1
            outer_log.append((iteration, "stall" if force_update else "gtolk"))
            # `residual` is the historical gate; `violation` keys on the same
            # quantity the `mu` branch below already uses.
            _viol_rows = constr_violation_rows(c, z)
            _pass = jnp.abs(c) < ctolk
            _sat = _viol_rows < ctolk
            y_gate = _sat if y_update_gate == "violation" else _pass
            y_prev = y
            y = jnp.where(y_gate, y - mu * c, y)
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
            if mu_growth_allowed:
                mu = jnp.where(constr_violation_rows(c, z) >= ctolk, tau * mu, mu)
                mu = jnp.minimum(mu, max_penalty)
            if c_norm < ctolk:
                ctolk = max(ctolk / (jnp.mean(mu) ** beta_eta), ctol)
            else:
                ctolk = max(eta / (jnp.mean(mu) ** alpha_eta), ctol)
            if track_outer:
                _blocked = _sat & ~_pass
                outer_diag.append(
                    dict(
                        iteration=iteration,
                        reason="stall" if force_update else "gtolk",
                        window_rel_progress=window_rel_progress,
                        window_max_step=window_max_step,
                        mu_growth_allowed=bool(mu_growth_allowed),
                        ctolk=float(ctolk),
                        constr_violation=float(constr_violation),
                        mu_mean=float(jnp.mean(mu)),
                        mu_max=float(jnp.max(mu)),
                        y_absmax=float(jnp.max(jnp.abs(y))),
                        dy_absmax=float(jnp.max(jnp.abs(y - y_prev))),
                        n_rows=int(c.size),
                        n_sat=int(jnp.sum(_sat)),
                        n_pass=int(jnp.sum(_pass)),
                        n_blocked=int(jnp.sum(_blocked)),
                        y_absmax_blocked=float(
                            jnp.max(jnp.where(_blocked, jnp.abs(y), 0.0))
                        ),
                    )
                )
        elif inner_stall:
            outer_log.append((iteration, "mu"))
            # Stalled while still infeasible. The subproblem is too hard at this
            # penalty, which is exactly what `mu` is for -- Conn & Gould's
            # unsuccessful branch. Raising it on the offending rows makes the restart
            # a genuinely different subproblem instead of the same one again. `y` is
            # deliberately left alone: `y <- y - mu*c` is only a valid estimate at an
            # approximate minimizer, and this point is not one.
            al_changed = True
            if mu_growth_allowed:
                mu = jnp.where(constr_violation_rows(c, z) >= ctolk, tau * mu, mu)
                mu = jnp.minimum(mu, max_penalty)
            ctolk = max(eta / (jnp.mean(mu) ** alpha_eta), ctol)
            if track_outer:
                _viol_rows = constr_violation_rows(c, z)
                _sat = _viol_rows < ctolk
                _pass = jnp.abs(c) < ctolk
                _blocked = _sat & ~_pass
                outer_diag.append(
                    dict(
                        iteration=iteration,
                        reason="mu",
                        window_rel_progress=window_rel_progress,
                        window_max_step=window_max_step,
                        mu_growth_allowed=bool(mu_growth_allowed),
                        ctolk=float(ctolk),
                        constr_violation=float(constr_violation),
                        mu_mean=float(jnp.mean(mu)),
                        mu_max=float(jnp.max(mu)),
                        y_absmax=float(jnp.max(jnp.abs(y))),
                        dy_absmax=0.0,
                        n_rows=int(c.size),
                        n_sat=int(jnp.sum(_sat)),
                        n_pass=int(jnp.sum(_pass)),
                        n_blocked=int(jnp.sum(_blocked)),
                        y_absmax_blocked=float(
                            jnp.max(jnp.where(_blocked, jnp.abs(y), 0.0))
                        ),
                    )
                )

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
            if second_order:  # the weights mu*c - y changed with (y, mu)
                S = _second_order_S(z, c, y, mu)

            if jac_scale:
                scale, scale_inv = compute_jac_scale(
                    J, scale_inv if jac_scale_ratchet else None
                )

            v, dv = cl_scaling_vector(z, g, lb, ub)
            v = jnp.where(dv != 0, v * scale_inv, v)
            g_norm = jnp.linalg.norm(
                (g * v * scale if scaled_termination else g * v), ord=jnp.inf
            )
            # (y, mu) changed, so this is a new subproblem
            iters_since_update = 0
            Lcost_window_start = Lcost
            window_max_step = 0.0
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
            if diag_on:
                _yn, _mn = _np.asarray(y), _np.asarray(mu)
                _diag_write(
                    dict(
                        kind="outer",
                        iteration=int(iteration),
                        reason=str(outer_log[-1][1]),
                        g_norm_trigger=_pre["g_norm"],
                        before=dict(
                            mu_mean=_pre["mu_mean"],
                            mu_max=_pre["mu_max"],
                            gtolk=_pre["gtolk"],
                            ctolk=_pre["ctolk"],
                        ),
                        after=dict(
                            mu_mean=float(jnp.mean(mu)),
                            mu_max=float(jnp.max(mu)),
                            gtolk=float(gtolk),
                            ctolk=float(ctolk),
                        ),
                        n_y_updated=int(_np.sum(_yn != _pre["y"])),
                        n_mu_increased=int(_np.sum(_mn > _pre["mu"])),
                        watched={
                            _slack_label(k): dict(
                                y_before=float(_pre["y"][int(_ineq_idx[k])]),
                                y_after=float(_yn[int(_ineq_idx[k])]),
                                mu_before=float(_pre["mu"][int(_ineq_idx[k])]),
                                mu_after=float(_mn[int(_ineq_idx[k])]),
                            )
                            for k in _diag_state["watch"]
                        },
                    )
                )

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
        if inner_stall or stagnated:
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
    if diag_on:
        _diag_fh.close()
    if (iteration == maxiter) and success is None:
        success, message = False, STATUS_MESSAGES["maxiter"]
    x, s = z2xs(z)
    active_mask = find_active_constraints(z, zbounds[0], zbounds[1], rtol=xtol)
    # after overwriting J_h with J*d, we have to revert back and store the
    # unscaled version
    J_h = scale_columns(J_h, 1 / d)
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
        jac=J_h,
        optimality=g_norm,
        nfev=nfev,
        njev=njev,
        nit=iteration,
        n_outer=n_outer,
        n_veto=n_veto,
        outer_log=outer_log,
        outer_diag=outer_diag,
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
