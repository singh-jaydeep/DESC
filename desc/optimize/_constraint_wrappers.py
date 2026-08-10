"""Wrappers for doing STELLOPT/SIMSOPT like optimization."""

import functools

import numpy as np

from desc.backend import jit, jnp, put
from desc.objectives import (
    BoundaryRSelfConsistency,
    BoundaryZSelfConsistency,
    ObjectiveFunction,
    get_fixed_boundary_constraints,
    maybe_add_self_consistency,
)
from desc.objectives.utils import (
    _Project,
    _Recover,
    factorize_linear_constraints,
    remove_fixed_parameters,
)
from desc.utils import Timer, errorif, get_instance, setdefault, warnif

from ._profiling import span as prof_span
from .utils import f_where_x


class LinearConstraintProjection(ObjectiveFunction):
    """Remove linear constraints via orthogonal projection.

    Given a problem of the form

    min_x f(x) subject to A*x=b

    We can write any feasible x=xp + Z*x_reduced where xp is a particular solution to
    Ax=b (taken to be the least norm solution), Z is a representation for the null
    space of A (A*Z=0) and x_reduced is unconstrained. This transforms the problem into

    min_x_reduced f(x_reduced)

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Objective function of linear constraints to enforce.
    x_scale : array_like or ``'auto'``, optional
        Characteristic scale of each variable. Setting ``x_scale`` is equivalent
        to reformulating the problem in scaled variables ``xs = x / x_scale``.
        If set to ``'auto'``, the scale is determined from the initial state vector.
        This can be passed through optimizer options as
        solve_options["linear_constraint_options"]["x_scale"].
    name : str
        Name of the objective function.

    """

    def __init__(
        self, objective, constraint, x_scale="auto", name="LinearConstraintProjection"
    ):
        errorif(
            not isinstance(objective, ObjectiveFunction),
            ValueError,
            "Objective should be instance of ObjectiveFunction.",
        )
        errorif(
            not isinstance(constraint, ObjectiveFunction),
            ValueError,
            "Constraint should be instance of ObjectiveFunction.",
        )
        for con in constraint.objectives:
            errorif(
                not con.linear,
                ValueError,
                "LinearConstraintProjection method cannot handle "
                + f"nonlinear constraint {con}.",
            )
            errorif(
                con.bounds is not None,
                ValueError,
                f"Linear constraint {con} must use target instead of bounds.",
            )

        self._objective = objective
        self._constraint = constraint
        self._x_scale = x_scale
        self._built = False
        # don't want to compile this, just use the compiled objective
        self._use_jit = False
        self._compiled = False
        self._name = name

    def build(self, use_jit=None, verbose=1):
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        timer = Timer()
        timer.start(f"{self.name} build")

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(verbose=verbose)
        if not self._constraint.built:
            self._constraint.build(verbose=verbose)

        self._dim_f = self._objective.dim_f
        self._scalar = self._objective.scalar
        (
            self._xp,
            self._A,
            self._b,
            self._Z,
            self._D,
            self._unfixed_idx,
            self._project,
            self._recover,
            self._ADinv,
            self._A_nondegenerate,
            self._degenerate_idx,
        ) = factorize_linear_constraints(
            self._objective,
            self._constraint,
            self._x_scale,
        )
        # inverse of the linear constraint matrix A without any scaling
        self._Ainv = self._D[self._unfixed_idx, None] * self._ADinv
        # nullspace of the linear constraint matrix A without any scaling
        self._ZA = self._D[self._unfixed_idx, None] * self._Z
        self._ZA = self._ZA / jnp.linalg.norm(self._ZA, axis=0)
        self._dim_x = self._objective.dim_x
        self._dim_x_reduced = self._Z.shape[1]

        # equivalent matrix for A[unfixed_idx] @ D @ Z == A @ feasible_tangents
        # Represents the tangent directions of the reduced parameters in full space
        # During optimization, we have the reduced parameters x_reduced, and we need
        # to compute the derivatives for that, but since compute functions are written
        # for the full state vector, we have to compute the derivatives with
        # these tangents.
        # For example, let's say the full state vector X has constraints X1=X2 and
        # X = [X1 X2 X3]. The reduced state vector of this is Y = [Y1 Y2]. We can take
        # Y1=X1=X2 and Y2=X3. Then df/dY1 = df/dX1 + df/dX2 and df/dY2 = df/dX3.
        # in this case, feasible_tangents = [ [1 , 0], [1, 0], [0,1]]
        # and is a shape 3x2 matrix equivalent to dx/dy
        # s.t. df/dy = df/dx @ dx/dy

        # df/dx_reduced = df/dx_full_unscaled @ dx_full_unscaled/dx_reduced # noqa: E800
        # x_full_unscaled = D(xp + Z @ x_reduced)                           # noqa: E800
        # So, the feasible tangents (aka. dx_full_unscaled/dx_reduced) is D@Z
        # Since the fixed parameters stay constant, we add 0 rows by below operation
        self._feasible_tangents = jnp.diag(self._D)[:, self._unfixed_idx] @ self._Z

        self._built = True
        timer.stop(f"{self.name} build")
        if verbose > 1:
            timer.disp(f"{self.name} build")

    def project(self, x):
        """Project full vector x into x_reduced that satisfies constraints."""
        return self._project(x)

    def recover(self, x_reduced):
        """Recover the full state vector from the reduced optimization vector."""
        return self._recover(x_reduced)

    def x(self, *things):
        """Return the reduced state vector from the Equilibrium eq."""
        x = self._objective.x(*things)
        return self.project(x)

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            Reduced state vector (e.g. from calling self.x(*things)).
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : pytree of dict
            if per_objective is True, this is a nested list of of parameters for each
            sub-Objective, such that self.objectives[i] has parameters params[i].
            Otherwise, it is a list of parameters tied to each optimizable thing
            such that params[i] = self.things[i].params_dict

        """
        if x.size != self._dim_x_reduced:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self._dim_x_reduced} got {x.size}."
            )
        x = self.recover(x)
        return self._objective.unpack_state(x, per_objective)

    def update_constraint_target(self, eq_new):
        """Update the target of the constraint.

        Updates the particular solution (xp), nullspace (Z), scaling (D) and
        the inverse of the scaled linear constraint matrix (ADinv) to reflect the new
        equilibrium a.k.a. the new target of the constraint of system Ax=b. This
        also updates the project and recover methods. Updating quantities in this way
        is faster than calling factorize_linear_constraints again.

        Parameters
        ----------
        eq_new : Equilibrium
            New equilibrium to target for the constraints.
        """
        for con in self._constraint.objectives:
            if hasattr(con, "update_target"):
                con.update_target(eq_new)

        dim_x = self._objective.dim_x
        # particular solution to Ax=b
        xp = jnp.zeros(dim_x)
        x0 = jnp.zeros(dim_x)
        A = self._A_nondegenerate
        b = -self._constraint.compute_scaled_error(x0)
        b = np.delete(b, self._degenerate_idx)

        # There is probably a more clever way of doing this, but for now we just
        # remove fixed parameters from A and b again by the same loop as in factorize
        # Actually A (unscaled linear constraint matrix without any degenerate rows)
        # does not change here, but still recompute it while updating others
        A, b, xp, unfixed_idx, fixed_idx = remove_fixed_parameters(A, b, xp)

        # if user specified x_scale, don't dynamically change it
        if self._x_scale == "auto":
            x_scale = self._objective.x(*self._objective.things)
            self._D = jnp.where(jnp.abs(x_scale) < 1e2, 1, jnp.abs(x_scale))

            # since D has changed, we need to update the ADinv
            # as mentioned above A does not change, so we can use the same Ainv
            # pinv(A) = Ainv, ADinv = pinv(A @ D) = Dinv @ Ainv, Dinv = 1 / D
            self._ADinv = (1 / self._D)[unfixed_idx, None] * self._Ainv
            # we also need to update the nullspace Z of AD in a similar way
            # A @ ZA = 0 -> (A @ D) @ ((1 / D) @ ZA) = 0 -> Z = (1 / D) @ ZA
            # where ZA is the nullspace of A, and Z is the nullspace of AD
            self._Z = (1 / self._D)[self._unfixed_idx, None] * self._ZA
            # we also normalize Z to make each column have unit norm
            self._Z = self._Z / jnp.linalg.norm(self._Z, axis=0)

        xp = put(xp, unfixed_idx, self._ADinv @ b)
        xp = put(xp, fixed_idx, ((1 / self._D) * xp)[fixed_idx])
        # cast to jnp arrays
        self._xp = jnp.asarray(xp)

        self._project = _Project(self._Z, self._D, self._xp, self._unfixed_idx)
        self._recover = _Recover(self._Z, self._D, self._xp, self._unfixed_idx, dim_x)

    def compute_unscaled(self, x_reduced, constants=None):
        """Compute the unscaled form of the objective function.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_unscaled(x, constants)
        return f

    def compute_scaled(self, x_reduced, constants=None):
        """Compute the objective function and apply weighting / normalization.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_scaled(x, constants)
        return f

    def compute_scaled_error(self, x_reduced, constants=None):
        """Compute the objective function and apply weighting / bounds.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        x = self.recover(x_reduced)
        f = self._objective.compute_scaled_error(x, constants)
        return f

    def compute_scalar(self, x_reduced, constants=None):
        """Compute the scalar form of the objective function.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : float
            Objective function value.

        """
        x = self.recover(x_reduced)
        return self._objective.compute_scalar(x, constants)

    def grad(self, x_reduced, constants=None):
        """Compute gradient of self.compute_scalar.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        g : ndarray
            gradient vector.

        """
        x = self.recover(x_reduced)
        df = self._objective.grad(x, constants)
        return df[self._unfixed_idx] @ (self._Z * self._D[self._unfixed_idx, None])

    def hess(self, x_reduced, constants=None):
        """Compute Hessian of self.compute_scalar.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        H : ndarray
            Hessian matrix.

        """
        x = self.recover(x_reduced)
        df = self._objective.hess(x, constants)
        return (
            (self._Z.T * (1 / self._D)[None, self._unfixed_idx])
            @ df[self._unfixed_idx, :][:, self._unfixed_idx]
            @ (self._Z * self._D[self._unfixed_idx, None])
        )

    def _jac(self, x_reduced, constants=None, op="scaled"):
        x = self.recover(x_reduced)
        v = self._feasible_tangents
        df = getattr(self._objective, "jvp_" + op)(v.T, x, constants)
        return df.T

    def jac_scaled(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_scaled.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x_reduced, constants, "scaled")

    def jac_scaled_error(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_scaled_error.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x_reduced, constants, "scaled_error")

    def jac_unscaled(self, x_reduced, constants=None):
        """Compute Jacobian of self.compute_unscaled.

        Parameters
        ----------
        x_reduced : ndarray
            Reduced state vector that satisfies linear constraints.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x_reduced, constants, "unscaled")

    def _jvp(self, v, x_reduced, constants=None, op="jvp_scaled"):
        x = self.recover(x_reduced)
        v = self._feasible_tangents @ v
        df = getattr(self._objective, op)(v, x, constants)
        return df

    def jvp_scaled(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._jvp(v, x_reduced, constants, "jvp_scaled")

    def jvp_scaled_error(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._jvp(v, x_reduced, constants, "jvp_scaled_error")

    def jvp_unscaled(self, v, x_reduced, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : tuple of ndarray
            Vectors to right-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._jvp(v, x_reduced, constants, "jvp_unscaled")

    def _vjp(self, v, x_reduced, constants=None, op="vjp_scaled"):
        x = self.recover(x_reduced)
        df = getattr(self._objective, op)(v, x, constants)
        return df[self._unfixed_idx] @ (self._Z * self._D[self._unfixed_idx, None])

    def vjp_scaled(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._vjp(v, x_reduced, constants, "vjp_scaled")

    def vjp_scaled_error(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._vjp(v, x_reduced, constants, "vjp_scaled_error")

    def vjp_unscaled(self, v, x_reduced, constants=None):
        """Compute vector-Jacobian product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray
            Vector to left-multiply the Jacobian by.
        x_reduced : ndarray
            Optimization variables with linear constraints removed.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        return self._vjp(v, x_reduced, constants, "vjp_unscaled")

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        return getattr(self._objective, name)


class ProximalState:
    """Shared state for the equilibrium subproblem of a proximal projection.

    Owns the equilibrium, the equilibrium constraint F, everything built from them,
    the state vector layout, and all of the mutable caches. One of these is shared by
    every ``ProximalProjection`` view of the same subproblem: the view wrapping the
    objective G, and, for the augmented Lagrangian methods, the view wrapping the
    nonlinear constraints H that are not equilibrium constraints. Sharing it is what
    makes two views cost one equilibrium solve per trial point instead of two.

    Parameters
    ----------
    constraint : ObjectiveFunction
        Equilibrium constraint to enforce. Should be an ObjectiveFunction with one or
        more of the following objectives: {ForceBalance, CurrentDensity,
        RadialForceBalance, HelicalForceBalance}
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the objectives.
    perturb_options, solve_options : dict
        dictionary of arguments passed to Equilibrium.perturb and Equilibrium.solve
        during the projection step.

    Notes
    -----
    This is a plain Python object on purpose. It is mutated as the optimization
    proceeds and must never be traced -- see the comment above ``jit_if_possible``
    for why ``ProximalProjection`` itself cannot be jitted with ``self`` either
    static or unstatic. The same reasoning applies to anything hanging off it, so
    this is not ``IOAble``, is not registered as a pytree, and is never passed into
    a jitted function; the jitted helpers take the pieces they need as explicit
    arguments.

    """

    def __init__(self, constraint, eq, perturb_options=None, solve_options=None):
        assert isinstance(
            constraint, ObjectiveFunction
        ), "constraint should be instance of ObjectiveFunction."
        for con in constraint.objectives:
            errorif(
                not con._equilibrium,
                ValueError,
                "ProximalProjection method cannot handle general "
                + f"nonlinear constraint {con}.",
            )
            # can't have bounds on constraint bc if constraint is satisfied then
            # Fx == 0, and that messes with Gx @ Fx^-1 Fc etc.
            errorif(
                con.bounds is not None,
                ValueError,
                "ProximalProjection can only handle equality constraints, "
                + f"got bounds for constraint {con}",
            )

        self.eq = eq
        self.constraint = constraint

        solve_options = {} if solve_options is None else solve_options
        perturb_options = {} if perturb_options is None else perturb_options
        # If user does not want the solve during build, mainly for debug purposes
        self.solve_during_proximal_build = solve_options.pop(
            "solve_during_proximal_build", True
        )
        perturb_options.setdefault("verbose", 0)
        perturb_options.setdefault("include_f", False)
        solve_options.setdefault("verbose", 0)
        self.perturb_options = perturb_options
        self.solve_options = solve_options

        # set by build
        self.eq_linear_constraints = None
        self.eq_solve_objective = None
        self.args = None
        self.dxdc = None
        # set by set_layout
        self.things = None
        self.eq_idx = None
        self.dimx_per_thing = None
        self.dimc_per_thing = None
        # mutable, shared by every view
        self.allx = []
        self.allxopt = []
        self.allxeq = []
        self.x_old = None
        self.history = []
        # eq_is_current <=> eq.params_dict equals the equilibrium params in
        # history[-1], ie. nothing has moved the equilibrium since the last commit
        self.eq_is_current = True
        # one point's worth of tangent matrices, see get_tangents
        self._tangent_x = None
        self._tangent_constants = None
        self._tangents = {}
        # number of tangent matrices actually built, for tests and profiling
        self.n_tangent_builds = 0

        self._built = False

    @property
    def built(self):
        """bool: Whether the equilibrium subproblem has been built."""
        return self._built

    def build(self, use_jit=None, verbose=1):
        """Build the equilibrium subproblem. Runs at most once per state.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the sub-objectives.
        verbose : int, optional
            Level of output.

        """
        if self._built:
            # a second view is attaching to an already built subproblem. The
            # expensive work -- the SVD in factorize_linear_constraints, the pinvs in
            # _set_eq_state_vector, the initial eq.solve -- has already been done.
            # use_jit and verbose from this caller are ignored, as they are for any
            # already built DESC object.
            return

        self.eq_linear_constraints = get_fixed_boundary_constraints(eq=self.eq)
        self.eq_linear_constraints = maybe_add_self_consistency(
            self.eq, self.eq_linear_constraints
        )

        if not self.constraint.built:
            self.constraint.build(use_jit=use_jit, verbose=verbose)
        for constraint in self.eq_linear_constraints:
            constraint.build(use_jit=use_jit, verbose=verbose)

        # Here we create and build the LinearConstraintProjection
        # for the equilibrium subproblem using self.constraint as objective
        # and our fixed-bdry constraints we just made. This will
        # be passed as the objective for the eq subproblem, which saves
        # some time as by building it here we can avoid re-computing the
        # constraint matrix A and its SVD for the feasible direction method
        self.eq_solve_objective = LinearConstraintProjection(
            self.constraint,
            ObjectiveFunction(self.eq_linear_constraints),
            name="Eq Update LinearConstraintProjection",
        )
        self.eq_solve_objective.build(use_jit=use_jit, verbose=verbose)

        errorif(
            self.constraint.things != [self.eq],
            ValueError,
            "ProximalProjection can only handle constraints on the equilibrium.",
        )

        self._set_eq_state_vector()

        # Ensure the equilibrium is solved to the specified tolerances: the
        # derivative identity assumes F = 0 at the point of evaluation, so eq must be
        # solved before anything is recorded against it. This depends only on
        # eq_solve_objective, not on the things list, which is why it belongs here
        # rather than in set_layout. It therefore runs exactly once per subproblem: a
        # second view returns above, and a layout rebuild re-seeds the caches without
        # re-solving.
        if self.solve_during_proximal_build:
            # like the eq_solve in _update_equilibrium, this re-enters
            # Optimizer.optimize, so collapse it to one span at profiling level 1
            with prof_span("eq_solve", collapse=True):
                self.eq.solve(
                    objective=self.eq_solve_objective,
                    constraints=None,
                    **self.solve_options,
                )

        self._built = True

    def _set_eq_state_vector(self):
        full_args = self.eq.optimizable_params.copy()
        self.args = self.eq.optimizable_params.copy()
        # the eq optimizable variables for proximal are the Rb, Zb and profile
        # coefficients. Once these are chosen, we will solve the equilibrium to
        # find the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n. That is why we remove them
        # from the list of optimizable variables. This is accompanied by not including
        # self-consistency constraints (see get_combined_constraint_objectives in
        # desc.optimize.optimizer) and also removing columns corresponding to these
        # variables from the constraint matrix A in
        # desc.objectives.utils.factorize_linear_constraints.
        for arg in ["R_lmn", "Z_lmn", "L_lmn", "Ra_n", "Za_n"]:
            self.args.remove(arg)

        dxdc = []
        xz = {arg: np.zeros(self.eq.dimensions[arg]) for arg in full_args}

        for arg in self.args:
            if arg not in ["Rb_lmn", "Zb_lmn"]:
                x_idx = self.eq.x_idx[arg]
                dxdc.append(np.eye(self.eq.dim_x)[:, x_idx])
            if arg == "Rb_lmn":
                c = get_instance(self.eq_linear_constraints, BoundaryRSelfConsistency)
                # We have A @ R_lmn = Rb_lmn
                A = c.jac_unscaled(xz)[0]["R_lmn"]
                Ainv = np.linalg.pinv(A)
                # Once this is multipled by Rb_lmn, we get the full eq state vector
                # with the R_lmn but rest is 0
                dxdRb = np.eye(self.eq.dim_x)[:, self.eq.x_idx["R_lmn"]] @ Ainv
                dxdc.append(dxdRb)
            if arg == "Zb_lmn":
                c = get_instance(self.eq_linear_constraints, BoundaryZSelfConsistency)
                A = c.jac_unscaled(xz)[0]["Z_lmn"]
                Ainv = np.linalg.pinv(A)
                dxdZb = np.eye(self.eq.dim_x)[:, self.eq.x_idx["Z_lmn"]] @ Ainv
                dxdc.append(dxdZb)
        # dxdc is a matrix that when multiplied by the optimization variables (only
        # Rb_lmn, Zb_lmn) gives the full state vector of the equilibrium (Rb_lmn and
        # Zb_lmn part will be 0, but they will be represented by the equivalent
        # R_lmn and Z_lmn). For example, let's say the eq optimization variables are
        # ceq = [Rb_lmn, Zb_lmn, p_l, i_l].T                      # noqa : E800
        # Then, we will use dxdc for the following:
        # xeq = dxdc @ ceq                                        # noqa : E800
        # And xeq will be,
        # xeq = [                                                 # noqa : E800
        #     R_lmn, Z_lmn, jnp.zeros_like(L_lmn)                 # noqa : E800
        #     jnp.zeros_like(Rb_lmn), jnp.zeros_like(Zb_lmn),     # noqa : E800
        #     p_l, i_l,                                           # noqa : E800
        # ]                                                       # noqa : E800
        self.dxdc = jnp.hstack(dxdc)

    def set_layout(self, things):
        """Set the state vector layout from the things the views are optimizing.

        Called once per view at build, and again from
        ``ProximalProjection._set_things`` if ``combine_args`` widens the things
        list. Every view of one subproblem must agree on this layout, which is why it
        lives here rather than on the views -- with a single copy there is nothing to
        keep in sync.

        Parameters
        ----------
        things : list of Optimizable
            Ordered list of things the views are optimizing.

        """
        things = list(things)
        errorif(
            self.eq not in things,
            ValueError,
            "The equilibrium of the proximal subproblem is not among the things "
            "being optimized.",
        )
        self.things = things
        self.eq_idx = things.index(self.eq)

        # the full state vector includes all the parameters from all the things
        # however, sub-objectives only need the part for their thing. We will
        # use this to split the state vector into its components
        self.dimx_per_thing = [t.dim_x for t in things]
        # we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium params
        # dimc_per_thing accounts for that, don't confuse it with reduced state vector
        dimc_per_thing = [t.dim_x for t in things]
        dimc_per_thing[self.eq_idx] = int(
            np.sum([self.eq.dimensions[arg] for arg in self.args])
        )
        # we will need to pass this as a static argument, only possible if tuple
        self.dimc_per_thing = tuple(dimc_per_thing)

        if self.x_old is None or self.x_old.size != sum(self.dimc_per_thing):
            # (Re)seed the caches against this layout. Discarding an existing cache is
            # only safe because set_layout runs at build time, before any optimization
            # step has been taken, so there is no history worth preserving.
            assert (
                len(self.history) <= 1
            ), "proximal state vector layout changed after optimization started"
            self._seed_caches()

    def _seed_caches(self):
        # build() has already solved the equilibrium; this only records it against the
        # layout. Do not solve here -- _seed_caches runs again on a layout rebuild,
        # and a second nested eq.solve would be pure waste.
        self.x_old = self.pack_x()
        self.allx = [self.x_old]
        self.allxopt = [
            jnp.concatenate([t.pack_params(t.params_dict) for t in self.things])
        ]
        self.allxeq = [self.eq.pack_params(self.eq.params_dict)]
        self.history = [[t.params_dict.copy() for t in self.things]]
        self.eq_is_current = True
        # the layout may have changed, so any cached tangents are the wrong shape
        self._tangent_x = None
        self._tangents = {}

    def get_tangents(self, x, op, constants, build):
        """Return the tangent matrix at x, calling build() on a miss.

        Keyed on the point, the scaling class and the constraint constants.
        ``scaled`` and ``scaled_error`` share a slot: compute_scaled_error differs
        from compute_scaled by a constant target offset, so their JVPs are the same
        function by construction. ``unscaled`` gets its own slot -- it agrees only in
        exact arithmetic, since the SVD in _proximal_eq_tangents sees a differently
        scaled matrix.

        Every view of the subproblem shares this cache, which is the point: the
        objective is differentiated as jac_scaled_error and the nonlinear constraints
        as jac_scaled, both land in the ``scaled`` slot, so one build serves both.

        One point's worth of cache is enough, because the optimizers only ask for
        tangents at the current accepted iterate: trial points get compute, not jac.

        Parameters
        ----------
        x : ndarray
            Point to compute the tangents at.
        op : str
            One of ``scaled``, ``scaled_error``, ``unscaled``.
        constants : any
            Constant parameters for the equilibrium constraint. (Deprecated)
        build : callable
            Zero argument callable returning the tangent matrix, called on a miss.

        """
        key = "scaled" if op in ("scaled", "scaled_error") else op
        x = np.asarray(x)
        # constants is compared by identity rather than value: it is deprecated and
        # in practice always None, and a conservative miss only costs a rebuild
        if (
            self._tangent_x is None
            or not np.array_equal(self._tangent_x, x)
            or constants is not self._tangent_constants
        ):
            self._tangent_x = x
            self._tangent_constants = constants
            self._tangents = {}
        if key not in self._tangents:
            self._tangents[key] = build()
            self.n_tangent_builds += 1
        return self._tangents[key]

    def commit(self, x, x_list):
        """Record x, and the equilibrium solved at it, as the accepted point.

        Parameters
        ----------
        x : ndarray
            Accepted state vector.
        x_list : list of dict
            Params of each thing at x, with the equilibrium params taken from the
            solved equilibrium rather than from x.

        """
        # Derivatives commit, and both the objective view and the constraint view are
        # differentiated at the same accepted point, so this is called more than once
        # per step. Overwrite rather than append in that case, or history gains a
        # duplicate entry per view and desynchronizes from allx.
        # np.array_equal is safe here: these are concrete arrays, never traced, the
        # same reason f_where_x can compare them.
        if self.x_old is not None and np.array_equal(
            np.asarray(x), np.asarray(self.x_old)
        ):
            self.history[-1] = x_list
        else:
            self.history.append(x_list)
        self.x_old = x
        # the caller has just set eq.params_dict to the solved state and written that
        # same state into history, so the two agree by construction
        self.eq_is_current = True

    def rollback(self):
        """Restore the equilibrium to the last committed params, if it has moved."""
        if self.eq_is_current:
            # nothing has perturbed the equilibrium since the last commit or
            # rollback, so eq.params_dict already equals history[-1] and
            # update_constraint_target would redo a pinv based particular solution
            # solve for nothing
            return
        self.eq.params_dict = self.history[-1][self.eq_idx]
        self.eq_solve_objective.update_constraint_target(self.eq)
        self.eq_is_current = True

    def pack_x(self):
        """Return the full proximal state vector from the current params of things.

        Note that we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium
        params.
        """
        xs = []
        for t in self.things:
            if t is self.eq:
                xs += [
                    jnp.concatenate(
                        [jnp.atleast_1d(t.params_dict[arg]) for arg in self.args]
                    )
                ]
            else:
                xs += [t.pack_params(t.params_dict)]
        return jnp.concatenate(xs)


class ProximalProjection(ObjectiveFunction):
    """Remove equilibrium constraint by projecting onto constraint at each step.

    Combines objective and equilibrium constraint into a single objective to then pass
    to an unconstrained optimizer.

    At each iteration, after a step is taken to reduce the objective, the equilibrium
    is perturbed and re-solved to bring it back into force balance. This is analogous
    to a proximal method where each iterate is projected back onto the feasible set.

    Parameters
    ----------
    objective : ObjectiveFunction
        Objective function to optimize.
    constraint : ObjectiveFunction
        Equilibrium constraint to enforce. Should be an ObjectiveFunction with one or
        more of the following objectives: {ForceBalance, CurrentDensity,
        RadialForceBalance, HelicalForceBalance}
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the objectives.
    perturb_options, solve_options : dict
        dictionary of arguments passed to Equilibrium.perturb and Equilibrium.solve
        during the projection step.
    name : str
        Name of the objective function.
    """

    def __init__(
        self,
        objective,
        constraint=None,
        eq=None,
        perturb_options=None,
        solve_options=None,
        state=None,
        name="ProximalProjection",
    ):
        assert isinstance(objective, ObjectiveFunction), (
            "objective should be instance of ObjectiveFunction." ""
        )
        # assign _objective and _state first: __getattr__ forwards every missing
        # attribute to self._objective, so touching either before they exist gives
        # infinite recursion with a useless traceback
        self._objective = objective
        if state is not None:
            errorif(
                constraint is not None and constraint is not state.constraint,
                ValueError,
                "Got both a ProximalState and a different constraint. The equilibrium "
                "constraint belongs to the state, so that every view of one "
                "subproblem is guaranteed to use the same F.",
            )
            errorif(
                eq is not None and eq is not state.eq,
                ValueError,
                "Got both a ProximalState and a different equilibrium. The "
                "equilibrium belongs to the state.",
            )
            errorif(
                perturb_options is not None or solve_options is not None,
                ValueError,
                "perturb_options and solve_options belong to the ProximalState, not "
                "to an individual view.",
            )
            self._state = state
        else:
            self._state = ProximalState(constraint, eq, perturb_options, solve_options)
        self._built = False
        # don't want to compile this, just use the compiled objective and constraint
        self._use_jit = False
        self._compiled = False
        self._name = name

    # The equilibrium subproblem lives on self._state, shared with any other view of
    # it. These delegate so that existing call sites -- and optimizer.py, which reads
    # _dimx_per_thing, _eq_idx and history -- keep working unchanged.

    @property
    def _eq(self):
        return self._state.eq

    @_eq.setter
    def _eq(self, new):
        self._state.eq = new

    @property
    def _constraint(self):
        return self._state.constraint

    @property
    def _eq_linear_constraints(self):
        return self._state.eq_linear_constraints

    @property
    def _eq_solve_objective(self):
        return self._state.eq_solve_objective

    @property
    def _args(self):
        return self._state.args

    @property
    def _dxdc(self):
        return self._state.dxdc

    @property
    def _eq_idx(self):
        return self._state.eq_idx

    @property
    def _dimx_per_thing(self):
        return self._state.dimx_per_thing

    @property
    def _dimc_per_thing(self):
        return self._state.dimc_per_thing

    @property
    def _perturb_options(self):
        return self._state.perturb_options

    @property
    def _solve_options(self):
        return self._state.solve_options

    @property
    def _solve_during_proximal_build(self):
        return self._state.solve_during_proximal_build

    @property
    def _allx(self):
        return self._state.allx

    @_allx.setter
    def _allx(self, new):
        self._state.allx = new

    @property
    def _allxopt(self):
        return self._state.allxopt

    @_allxopt.setter
    def _allxopt(self, new):
        self._state.allxopt = new

    @property
    def _allxeq(self):
        return self._state.allxeq

    @_allxeq.setter
    def _allxeq(self, new):
        self._state.allxeq = new

    @property
    def _x_old(self):
        return self._state.x_old

    @_x_old.setter
    def _x_old(self, new):
        self._state.x_old = new

    @property
    def history(self):
        """list: Params of each thing at each accepted step."""
        return self._state.history

    @history.setter
    def history(self, new):
        self._state.history = new

    def build(self, use_jit=None, verbose=1):  # noqa: C901
        """Build the objective.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
            Note: unused by this class, should pass to sub-objectives directly.
        verbose : int, optional
            Level of output.

        """
        timer = Timer()
        timer.start("Proximal projection build")

        # we don't always build here because in ~all cases the user doesn't interact
        # with this directly, so if the user wants to manually rebuild they should
        # do it before this wrapper is created for them.
        if not self._objective.built:
            self._objective.build(use_jit=use_jit, verbose=verbose)

        # builds F, the eq linear constraints, eq_solve_objective, args and dxdc, and
        # solves the equilibrium. Runs once per subproblem however many views share it
        self._state.build(use_jit=use_jit, verbose=verbose)

        # load-bearing for unpack_state and _proximal_jvp_blocked_pure, do not reduce
        # this to just [self._objective]
        self._objectives = [self._objective, self._constraint]
        # the state is built by now, so this also sets the layout
        # Note: the full state vector version of the feasible tangents used to be
        # built here, but its eq block is just _eq_solve_objective._feasible_tangents
        # and the other blocks are identity, so _proximal_get_tangents uses that.
        self._set_things()

        self._dim_f = self._objective.dim_f
        if self._dim_f == 1:
            self._scalar = True
        else:
            self._scalar = False

        self._built = True
        timer.stop("Proximal projection build")
        if verbose > 1:
            timer.disp("Proximal projection build")

    def _set_things(self, things=None):
        """Tell the wrapper, and the objective it wraps, what things it is optimizing.

        Parameters
        ----------
        things : list, tuple, or nested list, tuple of Optimizable
            Collection of things used by this objective. Defaults to all things from
            all sub-objectives.

        """
        # resolves things=None into the default union and sets self._things. Every
        # line below reads self.things, so this has to come first.
        super()._set_things(things)

        # The inner objective has to agree with the wrapper about the state vector
        # layout: _jvp hands it tangent directions and an xg built in the *wrapper's*
        # X space, so a narrower things list means it either rejects xg outright
        # (batched deriv mode) or silently depends on the per-thing slicing in
        # _proximal_jvp_blocked_pure (blocked). combine_args only calls _set_things on
        # the wrapper, so the wrapper is what has to propagate it. Widening an already
        # built ObjectiveFunction is safe: _set_things recomputes only _things,
        # _things_per_objective_idx and _flatten/_unflatten, and unpack_state still
        # filters each sub-objective down to the things it asked for.
        self._objective._set_things(self.things)

        # deliberately NOT self._constraint. F is evaluated at xf, the equilibrium
        # only state vector (see _proximal_eq_tangents), and ProximalState.build
        # requires F.things == [eq]. Widening it would break that contract.

        # the layout and the caches are derived from self.things, so they have to be
        # rebuilt. Guarded because build() calls _set_things() before the state has
        # the args and dxdc that set_layout needs.
        if self._state.built:
            self._state.set_layout(self.things)
            assert self._objective.dim_x == sum(self._state.dimx_per_thing), (
                "the proximal wrapper and the objective it wraps disagree about the "
                "state vector layout"
            )

    def unpack_state(self, x, per_objective=True):
        """Unpack the state vector into its components.

        Parameters
        ----------
        x : ndarray
            State vector.
        per_objective : bool
            Whether to return param dicts for each objective (default) or for each
            unique optimizable thing.

        Returns
        -------
        params : dict
            Parameter dictionary for equilibrium, with just external degrees of freedom
            visible to the optimizer.

        """
        if not self.built:
            raise RuntimeError("ObjectiveFunction must be built first.")

        x = jnp.atleast_1d(jnp.asarray(x))
        if x.size != self.dim_x:
            raise ValueError(
                "Input vector dimension is invalid, expected "
                + f"{self.dim_x} got {x.size}."
            )

        xs = jnp.split(x, np.cumsum(self._dimc_per_thing))
        params = []
        for t, xi in zip(self.things, xs):
            if t is self._eq:
                xi_splits = np.cumsum([self._eq.dimensions[arg] for arg in self._args])
                p = {arg: xis for arg, xis in zip(self._args, jnp.split(xi, xi_splits))}
                p.update(  # add in dummy values for missing parameters
                    {
                        arg: jnp.zeros_like(xis)
                        for arg, xis in t.params_dict.items()
                        if arg not in self._args  # R_lmn, Z_lmn, L_lmn, Ra_n, Za_n
                    }
                )
                params += [p]
            else:
                params += [t.unpack_params(xi)]

        if per_objective:
            # params is a list of lists of dicts, for each thing and for each objective
            params = self._unflatten(params)
            # this filters out the params of things that are unused by each objective
            params = [
                [par for par, thing in zip(param, self.things) if thing in obj.things]
                for param, obj in zip(params, self.objectives)
            ]
        return params

    def x(self, *things):
        """Return the full state vector from the Optimizable objects things.

        Note that we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium
        params.
        """
        # TODO (#1392): also check resolution etc?
        things = things or self.things
        assert [type(t1) is type(t2) for t1, t2 in zip(things, self.things)]
        xs = []
        for t in self.things:
            if t is self._eq:
                xs += [
                    jnp.concatenate(
                        [jnp.atleast_1d(t.params_dict[arg]) for arg in self._args]
                    )
                ]
            else:
                xs += [t.pack_params(t.params_dict)]

        return jnp.concatenate(xs)

    @property
    def dim_x(self):
        """int: Dimension of the state vector.

        Note that we remove the R_lmn, Z_lmn, L_lmn, Ra_n, Za_n from the equilibrium
        params.
        """
        s = 0
        for t in self.things:
            if t is self._eq:
                s += sum(self._eq.dimensions[arg] for arg in self._args)
            else:
                s += t.dim_x
        return s

    def _update_equilibrium(self, x, store=False):
        """Update the internal equilibrium with new boundary, profile etc.

        Parameters
        ----------
        x : ndarray
            New values of the state vector of equilibrium (except R_lmn, Z_lmn,
            L_lmn, Ra_n, Za_n) and all the parameters of the other things.
        store : bool
            Whether the new x should be stored in self.history

        Notes
        -----
        After updating, if store=False, self._eq will revert back to the previous
        solution when store was True

        """
        # xopt is the full state vector of all the things
        # xeq is the full state vector of the equilibrium only

        with prof_span("update_eq"):
            return self._update_equilibrium_inner(x, store)

    def _update_equilibrium_inner(self, x, store):
        # TODO (#1720): We don't need to check the whole state vector, just the
        # equilibrium parameters should be enough.
        # first check if its something we've seen before, if it is just return
        # cached value, no need to perturb + resolve
        xopt = f_where_x(x, self._allx, self._allxopt)
        xeq = f_where_x(x, self._allx, self._allxeq)
        if xopt.size > 0 and xeq.size > 0:
            pass
        else:
            # After unpack_state, R_lmn, Z_lmn, L_lmn, Ra_n and Za_n in below lists
            # will be 0s
            x_list = self.unpack_state(x, False)
            x_list_old = self.unpack_state(self._x_old, False)
            xeq_dict = x_list[self._eq_idx]
            xeq_dict_old = x_list_old[self._eq_idx]
            deltas = {str(key): xeq_dict[key] - xeq_dict_old[key] for key in xeq_dict}
            # We pass in the LinearConstraintProjection object to skip some redundant
            # computations in the perturb and solve methods
            with prof_span("perturb"):
                self._eq = self._eq.perturb(
                    objective=self._eq_solve_objective,
                    constraints=None,
                    deltas=deltas,
                    **self._perturb_options,
                )
            # this re-enters Optimizer.optimize, so at profiling level 1 it is
            # collapsed to a single opaque span rather than reported in full
            with prof_span("eq_solve", collapse=True):
                self._eq.solve(
                    objective=self._eq_solve_objective,
                    constraints=None,
                    **self._solve_options,
                )
            xeq = self._eq.pack_params(self._eq.params_dict)
            x_list[self._eq_idx] = self._eq.params_dict.copy()
            xopt = jnp.concatenate(
                [t.pack_params(xi) for t, xi in zip(self.things, x_list)]
            )
            self._allx.append(x)
            self._allxopt.append(xopt)
            self._allxeq.append(xeq)
            # the perturb + solve above is the only thing that moves the equilibrium
            # away from the last committed state
            self._state.eq_is_current = False

        if store:
            x_list = self.unpack_state(x, False)
            xeq_dict = self._eq.unpack_params(xeq)
            self._eq.params_dict = xeq_dict
            x_list[self._eq_idx] = xeq_dict
            self._state.commit(x, x_list)
        else:
            # reset to last good params
            self._state.rollback()

        return xopt, xeq

    def compute_scaled(self, x, constants=None):
        """Compute the objective function and apply weights/normalization.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled(xopt, constants[0])

    def compute_scaled_error(self, x, constants=None):
        """Compute the error between target and objective and apply weights etc.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_scaled_error(xopt, constants[0])

    def compute_scalar(self, x, constants=None):
        """Compute the sum of squares error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : float
            Objective function scalar value.

        """
        f = jnp.sum(self.compute_scaled_error(x, constants=constants) ** 2) / 2
        return f

    def compute_unscaled(self, x, constants=None):
        """Compute the raw value of the objective function.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        f : ndarray
            Objective function value(s).

        """
        constants = setdefault(constants, [None, None])
        xopt, _ = self._update_equilibrium(x, store=False)
        return self._objective.compute_unscaled(xopt, constants[0])

    def grad(self, x, constants=None):
        """Compute gradient of self.compute_scalar.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        g : ndarray
            gradient vector.

        """
        # We are looking for the gradient of L = 0.5 * Gᵀ @ G
        # Then, the gradient is ∇L = Gᵀ @ J_of_G
        # where J_of_G is the Jacobian of G with respect to the optimization variables
        # We explained getting J_of_G in the _jvp method. It is basically,
        # J_of_G = ∇G @ [dc_tangents - (∇F @ dx_tangents)⁻¹ @ (∇F @ dc_tangents)]
        # where ∇G is the Jacobian of G with respect to full state vector
        # and ∇F is the Jacobian of F with respect to full state vector. Then,
        # ∇L = Gᵀ @ ∇G @ [dc_tangents - (∇F @ dx_tangents)⁻¹ @ (∇F @ dc_tangents)]
        # We get the part in [] using the _proximal_get_tangents.
        constants = setdefault(constants, [None, None])
        xg, xf = self._update_equilibrium(x, store=True)
        with prof_span("tangents"):
            tangents = self._full_tangents(x, xf, constants, "scaled_error")
        with prof_span("obj_vjp"):
            g = self._objective.compute_scaled_error(xg, constants[0])
            g_vjp = self._objective.vjp_scaled_error(g, xg, constants[0])
        return tangents @ g_vjp

    def _full_tangents(self, x, xf, constants, op):
        """Tangent directions for every column of the Jacobian, cached on the state.

        _proximal_get_tangents is linear in its directions, so this is the matrix T
        with ``_jvp(v) == v @ T``. Building all of it is only worth it when every
        direction is wanted, which is the case for grad and the jac_* methods but not
        for a single JVP, so _jvp keeps its own uncached path.
        """
        state = self._state

        def build():
            return _proximal_get_tangents(
                state.constraint,
                xf,
                jnp.eye(x.shape[0]),
                constants[1],
                state.eq_solve_objective._feasible_tangents,
                state.dxdc,
                state.dimc_per_thing,
                state.eq_idx,
                op,
            )

        return state.get_tangents(x, op, constants[1], build)

    def _apply_objective_jvp(self, tangents, xg, constants, op):
        if self._objective._deriv_mode == "batched":
            # objective's method already know about its jac_chunk_size
            return getattr(self._objective, "jvp_" + op)(tangents, xg, constants[0])
        else:
            return _proximal_jvp_blocked_pure(
                self._objective,
                jnp.split(tangents, np.cumsum(self._dimx_per_thing), axis=-1),
                jnp.split(xg, np.cumsum(self._dimx_per_thing)),
                op,
            )

    def _jac(self, x, constants=None, op="scaled"):
        constants = setdefault(constants, [None, None])
        xg, xf = self._update_equilibrium(x, store=True)
        with prof_span("tangents"):
            tangents = self._full_tangents(x, xf, constants, op)
        with prof_span("obj_jvp"):
            return self._apply_objective_jvp(tangents, xg, constants, op).T

    def hess(self, x, constants=None):
        """Compute Hessian of self.compute_scalar.

        Uses the "small residual approximation" where the Hessian is replaced by
        the square of the Jacobian: H = J.T @ J

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        H : ndarray
            Hessian matrix.

        """
        J = self.jac_scaled_error(x, constants)
        return J.T @ J

    def jac_scaled(self, x, constants=None):
        """Compute Jacobian of self.compute_scaled.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x, constants, "scaled")

    def jac_scaled_error(self, x, constants=None):
        """Compute Jacobian of self.compute_scaled_error.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.

        """
        return self._jac(x, constants, "scaled_error")

    def jac_unscaled(self, x, constants=None):
        """Compute Jacobian of self.compute_unscaled.

        Parameters
        ----------
        x : ndarray
            State vector.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        Returns
        -------
        J : ndarray
            Jacobian matrix.
        """
        return self._jac(x, constants, "unscaled")

    def jvp_scaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "scaled"
        return self._jvp(v, x, constants, op)

    def jvp_scaled_error(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_scaled_error.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "scaled_error"
        return self._jvp(v, x, constants, op)

    def jvp_unscaled(self, v, x, constants=None):
        """Compute Jacobian-vector product of self.compute_unscaled.

        Parameters
        ----------
        v : ndarray or tuple of ndarray
            Vectors to right-multiply the Jacobian by.
            This method only works for first order jvps.
        x : ndarray
            Optimization variables.
        constants : list
            Constant parameters passed to sub-objectives. (Deprecated)

        """
        op = "unscaled"
        return self._jvp(v, x, constants, op)

    def _jvp(self, v, x, constants=None, op="scaled_error"):
        # The goal is to compute the Jacobian of the objective function with respect to
        # the optimization variables (c). Before taking the Jacobian, we update the
        # equilibrium such that
        # F(x+dx, c+dc) = 0 = F(x, c) + dF/dx * dx + dF/dc * dc
        # so that we can set F(x, c) = 0, from here we can solve for dx and get
        # dx = - (dF/dx)⁻¹ * dF/dc * dc     # noqa : E800
        # We can then compute the Jacobian of the objective function with respect to c
        # G(x+dx, c+dc) = G(x, c) + dG/dx * dx + dG/dc * dc
        # substituting in dx we get
        # G(x+dx, c+dc) = G(x, c) + [ dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc ] * dc
        # and the Jacobian we want is dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc

        # Note: This Jacobian can be obtained using JVPs in proper tangent directions.
        # First we will compute the tangent direction (see _proximal_get_tangents),
        # then we will compute the Jacobian.
        v = v[0] if isinstance(v, (tuple, list)) else v
        constants = setdefault(constants, [None, None])
        xg, xf = self._update_equilibrium(x, store=True)

        # we don't need to divide this part into blocked and batched because
        # self._constraint._deriv_mode will handle it
        with prof_span("tangents"):
            tangents = _proximal_get_tangents(
                self._constraint,
                xf,
                v,
                constants[1],
                self._eq_solve_objective._feasible_tangents,
                self._dxdc,
                self._dimc_per_thing,
                self._eq_idx,
                op,
            )

        with prof_span("obj_jvp"):
            return self._apply_objective_jvp(tangents, xg, constants, op)

    @property
    def constants(self):
        """list: constant parameters for each sub-objective."""
        warnif(
            True,
            FutureWarning,
            "constants is deprecated and will be removed in a future "
            "release. Users should not include constants in the arguments "
            "of their objective compute methods. Instead declare all the "
            "constants in the build method and use as obj._constants.",
        )
        return [self._objective.constants, self._constraint.constants]

    def __getattr__(self, name):
        """For other attributes we defer to the base objective."""
        # __getattr__ fires for any attribute missing on self, including while
        # __init__ is still running. Without this guard, touching _objective or
        # _state before they are assigned recurses infinitely instead of raising.
        if name in ("_objective", "_state"):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name}, it is accessed "
                "before __init__ assigned it"
            )
        return getattr(self._objective, name)


# in ProximalProjection we have an explicit state that we keep track of (and add
# to as we go) meaning if we jit anything with self static it doesn't update
# correctly, while if we leave self unstatic then it recompiles every time because
# the pytree structure of ProximalProjection is changing. To get around that we
# define these helper functions that are stateless so we can safely jit them


def jit_if_possible(func=None, *, static_argnames=("op",)):
    """Jit a function if use_jit."""
    if func is None:
        return functools.partial(jit_if_possible, static_argnames=static_argnames)
    jitted_func = functools.partial(jit, static_argnames=list(static_argnames))(func)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # first arg has to be ObjectiveFunction
        obj = args[0]
        if getattr(obj, "_use_jit", False):
            return jitted_func(*args, **kwargs)
        else:
            return func(*args, **kwargs)

    return wrapper


@jit_if_possible(static_argnames=("dimc_per_thing", "eq_idx", "op"))
def _proximal_get_tangents(
    constraint,
    xf,
    v,
    constants,
    eq_feasible_tangents,
    dxdc,
    dimc_per_thing,
    eq_idx,
    op="scaled_error",
):
    # We try to find dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc
    # where G is the objective function. Since DESC stores x and c in the same
    # vector, instead of multiple JVP calls, we will just find a tangent direction
    # that will give us the same result.
    # For making the explanation clear, assume J is the Jacobian of the objective
    # function with respect to the full state vector (both x and c). Then,
    # dG/dc = J @ (tangent vectors in c direction)
    # dG/dx = J @ (tangent vectors in x direction)
    # So, dG/dc - dG/dx * (dF/dx)⁻¹ * dF/dc can be written as
    # J @ [(tangent vectors in c direction) - (tangent vectors in x direction)@dfdc]
    # Note: We will never form full Jacobian J, we will just compute the above
    # expression by JVPs.

    # v contains prox._args DoFs from eq and other objects (like coils, surfaces
    # etc). Only the eq block changes when the equilibrium is re-solved.
    vs = jnp.split(v, np.cumsum(dimc_per_thing)[:-1], axis=-1)
    # JVPs are taken for the dxdc @ v tangents, but at most dimc of them are useful,
    # so take whichever combination needs fewer of them. Applying v after the JVPs
    # only pays off with a lot of coil, surface etc DoFs, ie. single stage.
    if vs[eq_idx].ndim == 2 and vs[eq_idx].shape[0] > dimc_per_thing[eq_idx]:
        eq_tangents = vs[eq_idx] @ _proximal_eq_tangents(
            constraint, xf, constants, eq_feasible_tangents, dxdc.T, op
        )
    else:
        dxdcv = vs[eq_idx] @ dxdc.T
        # atleast_2d and reshape are to also handle a single (1D) direction
        eq_tangents = _proximal_eq_tangents(
            constraint, xf, constants, eq_feasible_tangents, jnp.atleast_2d(dxdcv), op
        )
        eq_tangents = eq_tangents.reshape(dxdcv.shape)
    return jnp.concatenate([*vs[:eq_idx], eq_tangents, *vs[eq_idx + 1 :]], axis=-1)


@jit_if_possible
def _proximal_eq_tangents(
    constraint, xf, constants, eq_feasible_tangents, dxdcv, op="scaled_error"
):
    # Note: dxdcv holds the directions in c, mapped to the full eq state vector, as
    # rows. It is either dxdc.T or v @ dxdc.T, the return has the same shape.

    # here Fxh is dF/dx in the reduced (feasible) eq coordinates and Fc is dF/dc. A
    # single batched JVP gives both, so the SVD below is computed once by
    # construction, instead of relying on the compiler to hoist it out of a loop.
    # Our compute functions never include variables like Rb_lmn, Zb_lmn etc. So,
    # taking the JVP in just dc direction will give 0. To prevent this, we use dxdc
    # which is the dx/dc matrix and convert the Rb_lmn to R_lmn entries etc.
    # For example, if we want the derivative wrt Rb_023, we should take the derivative
    # wrt all R_lmn coefficients that contribute to Rb_023. See BoundaryRSelfConsistency
    # for the relation between Rb_lmn and R_lmn.
    dim_x_reduced = eq_feasible_tangents.shape[-1]
    tangents = jnp.concatenate([eq_feasible_tangents.T, dxdcv], axis=0)
    J = getattr(constraint, "jvp_" + op)(tangents, xf, constants)
    Fxh, Fc = J[:dim_x_reduced].T, J[dim_x_reduced:].T
    cutoff = jnp.finfo(Fxh.dtype).eps * max(Fxh.shape)
    uf, sf, vtf = jnp.linalg.svd(Fxh, full_matrices=False)
    sf += sf[-1]  # add a tiny bit of regularization
    sfi = jnp.where(sf < cutoff * sf[0], 0, 1 / sf)
    # this is (dF/dx)⁻¹ @ dF/dc for all the directions at once  # noqa : E800
    dfdc = vtf.T @ (sfi[:, None] * (uf.T @ Fc))
    # feasible_tangents maps the reduced eq state vector back to the full one
    return dxdcv - (eq_feasible_tangents @ dfdc).T


@jit_if_possible
def _proximal_jvp_blocked_pure(objective, vgs, xgs, op):
    # Note: This function is not vectorized and takes the full set of tangents, and
    # returns a matrix.

    # vgs and xgs are list of arrays (each element of the list is not same size
    # necessarily), that are split by the things in the objective. If there are multiple
    # things for the ObjectiveFunction, each split belongs to a different thing. The
    # information about which thing is used by which sub-objective is stored in
    # _things_per_objective_idx.

    # Note: This function is very similar to _jvp_blocked in ObjectiveFunction with
    # some naming differences to account for ProximalProjection.
    out = []
    for k, obj in enumerate(objective.objectives):
        thing_idx = objective._things_per_objective_idx[k]
        xi = [xgs[i] for i in thing_idx]
        vi = [vgs[i] for i in thing_idx]
        assert len(xi) > 0
        assert len(vi) > 0
        assert len(xi) == len(vi)
        if obj._deriv_mode == "rev":
            # obj might not allow fwd mode, so compute full rev mode jacobian
            # and do matmul manually. This is slightly inefficient, but usually
            # when rev mode is used, dim_f <<< dim_x, so its not too bad.
            Ji = getattr(obj, "jac_" + op)(*xi)
            outi = jnp.array([Jii @ vii.T for Jii, vii in zip(Ji, vi)]).sum(axis=0)
            out.append(outi)
        else:
            outi = getattr(obj, "jvp_" + op)([_vi for _vi in vi], xi).T
            out.append(outi)
    return jnp.concatenate(out).T
