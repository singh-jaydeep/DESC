"""Composite least-squares model with bound rows kept exact as hinges.

A row with pre-hinge scaled value ``s_i`` is either a target row, with residual
``s_i - t_i``, or a bound row, with residual
``max(0, s_i - hi_i) - max(0, lo_i - s_i)``.
The composite model linearizes ``s`` and keeps the hinge, so the model is convex and
piecewise quadratic instead of the Gauss-Newton model's frozen active set.
"""

import functools

from desc.backend import jit, jnp, while_loop


def _hinge(z, lo, hi):
    return jnp.maximum(0, z - hi) - jnp.maximum(0, lo - z)


def composite_resid(s, lo, hi, tgt, isb):
    """Scaled residuals from pre-hinge values.

    Parameters
    ----------
    s : ndarray, shape(m,)
        Pre-hinge scaled row values.
    lo, hi : ndarray, shape(m,)
        Scaled lower and upper bounds, used on bound rows (may be infinite).
    tgt : ndarray, shape(m,)
        Scaled targets, used on target rows.
    isb : ndarray of bool, shape(m,)
        True for bound rows.

    Returns
    -------
    r : ndarray, shape(m,)
        Hinge residual on bound rows, ``s - tgt`` on target rows.

    """
    return jnp.where(isb, _hinge(s, lo, hi), s - tgt)


def _model_resid(a, B, r, q, isb, lo, hi):
    e = B @ q
    return jnp.where(isb, _hinge(a + e, lo, hi), r + e)


def composite_model(a, B, r, q, isb, lo, hi):
    """Composite model value at step ``q``, without damping or second order terms.

    Parameters
    ----------
    a : ndarray, shape(m,)
        Pre-hinge scaled row values at the current point.
    B : ndarray, shape(m, n)
        Jacobian of the pre-hinge values (in the step's coordinates).
    r : ndarray, shape(m,)
        Current residuals; only target rows are used.
    q : ndarray, shape(n,)
        Step.
    isb : ndarray of bool, shape(m,)
        True for bound rows.
    lo, hi : ndarray, shape(m,)
        Scaled bounds on bound rows.

    Returns
    -------
    m : float
        ``0.5 * |rho(q)|^2`` with ``rho`` the hinge of ``a + B q`` on bound rows and
        ``r + B q`` on target rows.

    """
    m = _model_resid(a, B, r, q, isb, lo, hi)
    return 0.5 * jnp.dot(m, m)


@functools.partial(jit, static_argnames=["maxiter"])
def solve_composite(a, B, r, lam, isb, lo, hi, S=None, gtol=1e-12, maxiter=50):
    """Minimize the damped composite model by semismooth Newton.

    Minimizes ``composite_model(a, B, r, q, ...) + 0.5 lam |q|^2 + 0.5 q S q`` from
    ``q = 0``, with an Armijo backtracking line search. The generalized Hessian keeps
    the bound rows whose hinge is nonzero at the current iterate.

    Parameters
    ----------
    a, B, r, isb, lo, hi : ndarray
        As in ``composite_model``.
    lam : float
        Damping parameter. ``B^T B + S + lam I`` must be positive definite.
    S : ndarray, shape(n, n), optional
        Estimate of the second order term dropped by Gauss-Newton.
    gtol : float
        Stop once ``|g| < gtol * (1 + |q|)``.
    maxiter : int
        Maximum number of Newton steps.

    Returns
    -------
    q : ndarray, shape(n,)
        Minimizer of the model.
    nit : int
        Number of Newton steps taken.

    """
    n = B.shape[1]
    S = jnp.zeros((n, n), dtype=B.dtype) if S is None else S
    H0 = S + lam * jnp.eye(n, dtype=B.dtype)

    def merit(q):
        return composite_model(a, B, r, q, isb, lo, hi) + 0.5 * q @ H0 @ q

    def cond_fun(state):
        _, k, done = state
        return (k < maxiter) & ~done

    def body_fun(state):
        q, k, _ = state
        m = _model_resid(a, B, r, q, isb, lo, hi)
        g = B.T @ m + H0 @ q
        done = jnp.linalg.norm(g) < gtol * (1 + jnp.linalg.norm(q))
        w = jnp.where(isb, m != 0, 1.0)
        H = B.T @ (w[:, None] * B) + H0
        dq = -jnp.linalg.solve(H, g)
        m0, slope = merit(q), g @ dq

        def armijo_cond(t):
            return (merit(q + t * dq) > m0 + 1e-4 * t * slope) & (t > 1e-10)

        t = while_loop(armijo_cond, lambda t: 0.5 * t, jnp.ones((), dtype=q.dtype))
        q = jnp.where(done, q, q + t * dq)
        return q, k + jnp.where(done, 0, 1), done

    q, k, _ = while_loop(
        cond_fun, body_fun, (jnp.zeros(n, dtype=B.dtype), 0, jnp.array(False))
    )
    return q, k
