import numpy as np
from scipy.optimize import NonlinearConstraint

from desc.backend import jnp
from desc.utils import warnif

from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_ls import lsq_auglag
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .optimizer import register_optimizer
from .stochastic import sgd

# List of all optax optimizers to register
# You may use the following test to update the list accordingly
# https://github.com/PlasmaControl/DESC/pull/2041#issuecomment-3813092445
_all_optax_optimizers = [
    "adabelief",
    "adadelta",
    "adafactor",
    "adagrad",
    "adam",
    "adamax",
    "adamaxw",
    "adamw",
    "adan",
    "amsgrad",
    "fromage",
    "lamb",
    "lars",
    "lbfgs",
    "lion",
    "nadam",
    "nadamw",
    "noisy_sgd",
    "novograd",
    "optimistic_adam_v2",
    "optimistic_gradient_descent",
    "polyak_sgd",
    "radam",
    "rmsprop",
    "rprop",
    "sgd",
    "sign_sgd",
    "sm3",
    "yogi",
]


def _warn_if_bounds(objective, constraint, x0, options):
    """Warn if bounds on sub-objectives can make the Jacobian rank-deficient.

    A sub-objective with bounds contributes zero residuals, and hence zero Jacobian
    rows, whenever it is inside its bounds. The full rank of the (m, n) Jacobian is
    min(m, n), so those rows can only cost rank if the ones that always remain are
    fewer than that. Only ``"svd"`` handles the deficient case reliably, so warn if
    the user picked a different method and use ``"svd"`` if they didn't specify one.
    """
    sub = objective
    while hasattr(sub, "_objective"):  # unwrap Proximal/LinearConstraintProjection
        sub = sub._objective
    bounded = [obj for obj in sub.objectives if obj.bounds is not None]
    dim_f_bounded = sum(obj.dim_f for obj in bounded)
    # constraint rows never vanish, bounds there become slack variables instead
    m = objective.dim_f + (0 if constraint is None else constraint.dim_f)
    n = x0.size
    if m - dim_f_bounded < min(m, n):
        # we only want to warn if the user supplied a method that is not svd
        # if user didn't supply, the default qr will be switched to svd
        tr_method = options.get("tr_method", "svd")
        warnif(
            tr_method != "svd",
            UserWarning,
            f"Objectives {[obj.name for obj in bounded]} use bounds instead of target, "
            + f"so they can zero out {dim_f_bounded} of the {m} rows of the ({m}, {n}) "
            + f"Jacobian and drop its rank below {min(m, n)}. The trust region method "
            + f"{tr_method} may then fail to solve the subproblem, it is safer to use "
            + "options={'tr_method': 'svd'}.",
        )
        # if not set by user, use SVD to avoid rank-deficiency issues
        if not hasattr(options, "tr_method"):
            options["tr_method"] = "svd"
    return options


def _make_constraint_hess(
    constraint, chunk_size=16, rtol=1e-6, exclude=None, verbose=0
):
    """Build ``constraint_hess(x, w) -> d2_x [ w . constraint.compute_scaled(x) ]``.

    For ``lsq_auglag(second_order="constraints")``. ``x`` is the optimizer's (possibly
    reduced) state; a ``LinearConstraintProjection`` is differentiated through its
    linear ``recover``. Only sub-objectives whose weights are non-negligible
    (``max|w_block| > rtol * max|w|``) are included, so an expensive constraint the
    multipliers do not load (e.g. a hard-min plasma-coil distance with ~0 weight) is
    never twice-differentiated. That filter is weight-based, and ``mu * (c - s)`` can
    load an expensive block once the slacks move, so ``exclude`` (sub-objective class
    names) keeps such blocks out unconditionally; they are also ignored when finding
    ``max|w|``. Forward-over-reverse (``jacfwd_chunked`` of ``grad``), jitted and cached
    per active set.
    """
    import jax

    from desc.batching import jacfwd_chunked

    from ._constraint_wrappers import LinearConstraintProjection

    if isinstance(constraint, LinearConstraintProjection):
        inner, recover = constraint._objective, constraint.recover
    else:
        inner, recover = constraint, (lambda xr: xr)
    objs = list(inner.objectives)
    dims = [int(o.dim_f) for o in objs]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    constants = inner._get_deprecated_constants(None)
    cache = {}

    def _build(active):
        def phi(xr, w):
            params = inner.unpack_state(recover(xr))
            tot = 0.0
            for k in active:
                fk = objs[k].compute_scaled(*params[k], constants=constants[k])
                tot = tot + jnp.dot(w[offs[k] : offs[k + 1]], fk)
            return tot

        return jax.jit(jacfwd_chunked(jax.grad(phi), argnums=0, chunk_size=chunk_size))

    exclude = set(exclude or ())
    allowed = [k for k, o in enumerate(objs) if type(o).__name__ not in exclude]

    def constraint_hess(x, w):
        w = jnp.asarray(w)
        wabs = np.abs(np.asarray(w))
        # the filter compares against the largest weight of the blocks that can be
        # included, so an excluded block's weight neither enters the Hessian nor
        # switches the others off
        wmax = max(
            (float(wabs[offs[k] : offs[k + 1]].max()) for k in allowed if dims[k]),
            default=0.0,
        )
        if wmax == 0.0:
            return jnp.zeros((x.size, x.size))
        active = tuple(
            k
            for k in allowed
            if dims[k] and wabs[offs[k] : offs[k + 1]].max() > rtol * wmax
        )
        if active not in cache:
            if verbose:
                print(
                    "second_order: compiling constraint Hessian for blocks "
                    + ", ".join(type(objs[k]).__name__ for k in active),
                    flush=True,
                )
            cache[active] = _build(active)
        return cache[active](jnp.asarray(x), w)

    return constraint_hess


def _make_constraint_quad(constraint):
    """Build ``constraint_quad(x, w, v) -> [v . d2_x(w_k . c_k) . v for each block k]``.

    For ``lsq_auglag``'s ``diag_block_split_from`` model check. One Hessian-vector
    product (forward-over-reverse) per sub-objective, so it stays cheap for a block
    whose dense Hessian does not (the hard-min plasma-coil distance). Jitted per block
    on first use.
    """
    import jax

    from ._constraint_wrappers import LinearConstraintProjection

    if isinstance(constraint, LinearConstraintProjection):
        inner, recover = constraint._objective, constraint.recover
    else:
        inner, recover = constraint, (lambda xr: xr)
    objs = list(inner.objectives)
    dims = [int(o.dim_f) for o in objs]
    offs = np.concatenate([[0], np.cumsum(dims)]).astype(int)
    constants = inner._get_deprecated_constants(None)
    fns = {}

    def _build(k):
        a, e = int(offs[k]), int(offs[k + 1])

        def phi(xr, w):
            params = inner.unpack_state(recover(xr))
            return jnp.dot(
                w[a:e], objs[k].compute_scaled(*params[k], constants=constants[k])
            )

        def quad(xr, w, v):
            _, hv = jax.jvp(lambda xx: jax.grad(phi)(xx, w), (xr,), (v,))
            return jnp.dot(v, hv)

        return jax.jit(quad)

    def constraint_quad(x, w, v):
        out = []
        for k in range(len(objs)):
            if dims[k] == 0:
                out.append(0.0)
                continue
            if k not in fns:
                fns[k] = _build(k)
            out.append(float(fns[k](jnp.asarray(x), jnp.asarray(w), jnp.asarray(v))))
        return np.asarray(out)

    return constraint_quad


@register_optimizer(
    name=["fmin-auglag", "fmin-auglag-bfgs"],
    description=[
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
        "Augmented Lagrangian trust region method for minimizing scalar valued "
        + "multivariate function. Uses BFGS to approximate Hessian. "
        + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmin_auglag.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=[True, False],
    GPU=True,
)
def _optimize_desc_aug_lagrangian(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmin_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"fmin-auglag", "fmin-auglag-bfgs"}
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.fmin_auglag`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options["max_nfev"] = stoptol["max_nfev"]
    # local lambdas to handle constants from both objective and constraint
    hess = (lambda x, *c: objective.hess(x)) if "bfgs" not in method else "bfgs"

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x),
        )
        # TODO (#1394): can't pass constants dict into vjp for now
        constraint_wrapped.vjp = lambda v, x, *args: constraint.vjp_scaled(v, x)
    else:
        constraint_wrapped = None

    # Same as the lsq-auglag wrapper below: fmin_auglag takes `callback(xk, *args)` as
    # an argument (and rejects unknown options), so pull it out of `options` and hand it
    # the recovered full state rather than the projected coordinates the optimizer steps
    # in.
    _user_cb = options.pop("callback", None)
    if _user_cb is None:
        _cb = None
    else:
        _recover = getattr(objective, "recover", None)

        def _cb(_x, *_a, _f=_user_cb, _r=_recover):
            return bool(_f(_r(_x) if _r is not None else _x, *_a))

    result = fmin_auglag(
        objective.compute_scalar,
        callback=_cb,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-auglag",
    description="Least squares augmented Lagrangian for constrained optimization"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsq_auglag.html",  # noqa: E501
    scalar=False,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_aug_lagrangian_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsq_auglag.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy
    x0 : ndarray
        Starting point.
    method : {"lsq-auglag"}
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsq_auglag`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    options = _warn_if_bounds(objective, constraint, x0, options)
    options["max_nfev"] = stoptol["max_nfev"]

    if constraint is not None:
        lb, ub = constraint.bounds_scaled
        constraint_wrapped = NonlinearConstraint(
            lambda x, *c: constraint.compute_scaled(x),
            lb,
            ub,
            lambda x, *c: constraint.jac_scaled(x),
        )
    else:
        constraint_wrapped = None

    # `lsq_auglag` documents a `callback(xk, *args) -> bool` that terminates the solve
    # GRACEFULLY (STATUS_MESSAGES["callback"]), so the caller still gets a normal
    # result and can write its record. It was unreachable: this wrapper never
    # forwarded one, so it always defaulted to `lambda *args: False`. Pull it out of
    # `options`, which is already threaded through from `Optimizer.optimize`.
    # The optimizer works in the projected space, so recover the full state first --
    # exactly what `optimizer.py` does to `result["allx"]` after the fact.
    _user_cb = options.pop("callback", None)
    if _user_cb is None:
        _cb = None
    else:
        _recover = getattr(objective, "recover", None)

        def _cb(_x, *_a, _f=_user_cb, _r=_recover):
            return bool(_f(_r(_x) if _r is not None else _x, *_a))

    # Same projected-space problem as `callback` above: `step_veto` inspects the
    # GEOMETRY of a trial iterate, so it must see the recovered full state, not the
    # reduced coordinates the optimizer steps in.
    _user_veto = options.pop("step_veto", None)
    if _user_veto is not None:
        _recover_v = getattr(objective, "recover", None)

        def _veto(_xn, _xo, _f=_user_veto, _r=_recover_v):
            if _r is not None:
                _xn, _xo = _r(_xn), _r(_xo)
            return bool(_f(_xn, _xo))

        options["step_veto"] = _veto

    if (
        options.get("second_order")
        and constraint is not None
        and "constraint_hess" not in options
    ):
        options["constraint_hess"] = _make_constraint_hess(
            constraint,
            chunk_size=options.pop("second_order_chunk", 16),
            rtol=options.pop("second_order_rtol", 1e-3),
            exclude=options.pop("second_order_exclude", None),
            verbose=verbose,
        )
    else:
        options.pop("second_order_chunk", None)
        options.pop("second_order_rtol", None)
        options.pop("second_order_exclude", None)
    if options.get("diag_block_split_from") is not None and constraint is not None:
        options["constraint_quad"] = _make_constraint_quad(constraint)

    result = lsq_auglag(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled_error,
        bounds=(-jnp.inf, jnp.inf),
        constraint=constraint_wrapped,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        callback=_cb,
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-exact",
    description="Trust region least squares, similar to the `trf` method in scipy"
    + "See https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsqtr.html",  # noqa: E501
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_least_squares(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsqtr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : {"lsq-exact"}
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsqtr`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_radius", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options = _warn_if_bounds(objective, constraint, x0, options)
    options["max_nfev"] = stoptol["max_nfev"]

    result = lsqtr(
        objective.compute_scaled_error,
        x0=x0,
        jac=objective.jac_scaled_error,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name=[
        "fmintr",
        "fmintr-bfgs",
    ],
    description=[
        "Trust region method for minimizing scalar valued multivariate function. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
        "Trust region method for minimizing scalar valued multivariate function. Uses "
        + "BFGS to approximate the Hessian. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.fmintr.html",  # noqa: E501
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=[True, False],
    GPU=True,
)
def _optimize_desc_fmin_scalar(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.fmintr.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. An alternative
        view is that the size of a trust region along jth dimension is proportional to
        x_scale[j]. Improved convergence may be achieved by setting x_scale such that
        a step of a given size along any of the scaled variables has a similar effect
        on the cost function. If set to ‘jac’, the scale is iteratively updated using
        the inverse norms of the columns of the Jacobian matrix.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See the code for more details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    hess = objective.hess if "bfgs" not in method else "bfgs"
    if not isinstance(x_scale, str) and jnp.allclose(x_scale, 1):
        options.setdefault("initial_trust_ratio", 1e-3)
        options.setdefault("max_trust_radius", 1.0)
    elif options.get("initial_trust_radius", "scipy") == "scipy":
        options.setdefault("initial_trust_ratio", 0.1)
    options["max_nfev"] = stoptol["max_nfev"]

    result = fmintr(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        hess=hess,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result


@register_optimizer(
    name=["sgd", "optax-custom"] + ["optax-" + opt for opt in _all_optax_optimizers],
    description=[
        "Stochastic gradient descent with Nesterov momentum. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.sgd.html",  # noqa: E501
        "Wrapper for custom ``optax`` optimizer. See "
        + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.sgd.html",  # noqa: E501
    ]
    + [
        f"``optax`` wrapper for {opt}. See "
        + f"https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.{opt}"  # noqa: E501
        for opt in _all_optax_optimizers
    ],
    scalar=True,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=True,
    hessian=False,
    GPU=True,
)
def _optimize_desc_stochastic(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.sgd.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : str
        Name of the method to use. Available options are `'sgd'`.
        Additionally, ``optax`` optimizers can be used by specifying the method as
        ``'optax-<optimizer_name>'``, where ``<optimizer_name>`` is any valid ``optax``
        optimizer. Hyperparameters for the ``optax`` optimizer must be passed via the
        ``'optax-options'`` key of ``options`` dictionary. A custom ``optax``
        optimizer can be used by specifying the method as ``'optax-custom'`` and
        passing the ``optax`` optimizer via the ``'update-rule'`` key of
        ``'optax-options'`` in the ``options`` dictionary.
    x_scale : array_like or 'auto', optional
        Characteristic scale of each variable. Setting x_scale is equivalent to
        reformulating the problem in scaled variables xs = x / x_scale. Improved
        convergence may be achieved by setting x_scale such that a step of a given
        size along any of the scaled variables has a similar effect on the cost
        function. Defaults to 'auto', meaning no scaling.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.sgd`` for details.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    assert constraint is None, f"method {method} doesn't support constraints"
    options = {} if options is None else options
    result = sgd(
        objective.compute_scalar,
        x0=x0,
        grad=objective.grad,
        args=(),
        method=method,
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=None,
        options=options,
    )
    return result
