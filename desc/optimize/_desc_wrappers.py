from scipy.optimize import NonlinearConstraint

from desc.backend import jnp
from desc.utils import warnif

from .aug_lagrangian import fmin_auglag
from .aug_lagrangian_composite import lsq_auglag_composite
from .aug_lagrangian_ls import lsq_auglag
from .fmin_scalar import fmintr
from .least_squares import lsqtr
from .least_squares_composite import lsq_composite
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


def _full_state_callback(callback, objective):
    """Call ``callback`` with the full state; solvers iterate on the reduced one."""
    if callback is None:
        return None
    recover = getattr(objective, "recover", lambda x: x)
    return lambda x, *args: bool(callback(recover(x), *args))


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

    result = fmin_auglag(
        objective.compute_scalar,
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
    name="lsq-composite",
    description="Levenberg-Marquardt least squares with bounded objectives kept exact "
    + "as hinges in the model. See "
    + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsq_composite.html",  # noqa: E501
    scalar=False,
    equality_constraints=False,
    inequality_constraints=False,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_least_squares_composite(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsq_composite.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy - not supported by this method
    x0 : ndarray
        Starting point.
    method : {"lsq-composite"}
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. If set to ‘jac’, the scale is the
        inverse of the largest column norm of the Jacobian seen so far.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report.
        * 2 : display progress during iterations
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev", "max_njev", "max_ngev", "max_nhev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsq_composite`` for details. In addition,
        ``"callback"`` is called with the full state after each accepted step;
        returning True stops the optimization.

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
    options = {} if options is None else dict(options)
    callback = _full_state_callback(options.pop("callback", None), objective)
    options["max_nfev"] = stoptol["max_nfev"]
    lo, hi, tgt, isb = objective.scaled_bounds()

    result = lsq_composite(
        objective.compute_scaled,
        x0=x0,
        jac=objective.jac_scaled,
        lo=lo,
        hi=hi,
        tgt=tgt,
        isb=isb,
        row_ids=getattr(objective, "row_ids", None),
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        maxiter=stoptol["maxiter"],
        verbose=verbose,
        callback=callback,
        options=options,
    )
    return result


@register_optimizer(
    name="lsq-auglag-composite",
    description="Least squares augmented Lagrangian with slack-free hinge rows and "
    + "composite subproblem solves. See "
    + "https://desc-docs.readthedocs.io/en/stable/_api/optimize/desc.optimize.lsq_auglag_composite.html",  # noqa: E501
    scalar=False,
    equality_constraints=True,
    inequality_constraints=True,
    stochastic=False,
    hessian=False,
    GPU=True,
)
def _optimize_desc_aug_lagrangian_composite(
    objective, constraint, x0, method, x_scale, verbose, stoptol, options=None
):
    """Wrapper for desc.optimize.lsq_auglag_composite.

    Parameters
    ----------
    objective : ObjectiveFunction
        Function to minimize.
    constraint : ObjectiveFunction
        Constraint to satisfy.
    x0 : ndarray
        Starting point.
    method : {"lsq-auglag-composite"}
        Name of the method to use.
    x_scale : array_like or ‘jac’, optional
        Characteristic scale of each variable. If set to ‘jac’, the scale is the
        inverse of the largest column norm of the Jacobian seen so far.
    verbose : int
        * 0  : work silently.
        * 1 : display a termination report and one line per outer iteration.
        * 2 : display progress during the subproblem solves too.
    stoptol : dict
        Dictionary of stopping tolerances, with keys {"xtol", "ftol", "gtol", "ctol",
        "maxiter", "max_nfev"}
    options : dict, optional
        Dictionary of optional keyword arguments to override default solver
        settings. See ``desc.optimize.lsq_auglag_composite`` for details. In
        addition, ``"callback"`` is called with the full state after each accepted
        step (returning True stops the optimization), and ``"outer_callback"``
        receives the full state as ``x``.

    Returns
    -------
    res : OptimizeResult
       The optimization result represented as a ``OptimizeResult`` object.
       Important attributes are: ``x`` the solution array, ``success`` a
       Boolean flag indicating if the optimizer exited successfully and
       ``message`` which describes the cause of the termination. See
       `OptimizeResult` for a description of other attributes.

    """
    options = {} if options is None else dict(options)
    callback = _full_state_callback(options.pop("callback", None), objective)
    options.setdefault("max_nfev", stoptol["max_nfev"])
    if options.get("outer_callback") is not None:
        recover = getattr(objective, "recover", lambda x: x)
        options["outer_callback"] = lambda h, f=options["outer_callback"]: f(
            {**h, "x": recover(h["x"])}
        )
    lo, hi, tgt, isb = objective.scaled_bounds()
    if constraint is None:  # no constraint rows: the AL reduces to one subproblem
        n = x0.size
        con, con_jac = (lambda x: jnp.zeros(0)), (lambda x: jnp.zeros((0, n)))
        lb = ub = jnp.zeros(0)
        con_row_ids = keep_rows = None
    else:
        clo, chi, ctgt, cisb = constraint.scaled_bounds()
        con, con_jac = constraint.compute_scaled, constraint.jac_scaled
        lb, ub = jnp.where(cisb, clo, ctgt), jnp.where(cisb, chi, ctgt)
        con_row_ids = getattr(constraint, "row_ids", None)
        keep_rows = getattr(constraint, "keep_rows", None)

    result = lsq_auglag_composite(
        objective.compute_scaled,
        x0=x0,
        jac=objective.jac_scaled,
        lo=lo,
        hi=hi,
        tgt=tgt,
        isb=isb,
        con=con,
        con_jac=con_jac,
        lb=lb,
        ub=ub,
        row_ids=getattr(objective, "row_ids", None),
        con_row_ids=con_row_ids,
        keep_rows=keep_rows,
        args=(),
        x_scale=x_scale,
        ftol=stoptol["ftol"],
        xtol=stoptol["xtol"],
        gtol=stoptol["gtol"],
        ctol=stoptol["ctol"],
        verbose=verbose,
        maxiter=stoptol["maxiter"],
        callback=callback,
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
