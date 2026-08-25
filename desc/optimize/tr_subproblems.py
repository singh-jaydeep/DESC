"""Functions for solving subproblems arising in trust region methods."""

import functools

import numpy as np

from desc.backend import (
    cho_factor,
    cho_solve,
    cond,
    jit,
    jnp,
    qr,
    qr_multiply,
    solve_triangular,
    while_loop,
)
from desc.utils import setdefault

from .utils import chol, solve_triangular_regularized


@jit
def solve_trust_region_dogleg(g, H, trust_radius, initial_alpha=None, **kwargs):
    """Solve trust region subproblem using the dog-leg method.

    Parameters
    ----------
    g : ndarray
        gradient of objective function
    H : ndarray
        Hessian matrix
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    initial_alpha : float
        initial guess for Levenberg-Marquardt parameter - unused by this method.

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        "Levenberg-Marquardt" parameter - unused by this method.

    """
    L = chol(H)
    # This is the optimum for the quadratic model function.
    # If it is inside the trust radius then return this point.
    p_newton = -cho_solve((L, True), g)
    p_newton_norm = jnp.linalg.norm(p_newton)

    # This is the predicted optimum along the direction of steepest descent.
    gBg = g @ H @ g
    p_cauchy = -(jnp.dot(g, g) / gBg) * g
    # If the Cauchy point is outside the trust region,
    # then return the point where the path intersects the boundary.
    p_cauchy_norm = jnp.linalg.norm(p_cauchy)
    p_boundary1 = p_cauchy * (trust_radius / p_cauchy_norm)

    # Compute the intersection of the trust region boundary
    # and the line segment connecting the Cauchy and Newton points.
    # This requires solving a quadratic equation.
    # ||p_u + t*(p_best - p_u)||**2 == trust_radius**2
    # Solve this for positive time t using the quadratic formula.
    delta = p_newton - p_cauchy
    _, tb = get_boundaries_intersections(p_cauchy, delta, trust_radius)
    p_boundary2 = p_cauchy + tb * delta

    # p_boundary1,2 are super cheap to compute so easier to just compute all of them
    # and then select the one to return
    out = cond(
        p_cauchy_norm >= trust_radius,
        lambda _: (p_boundary1, True),
        lambda _: (p_boundary2, True),
        None,
    )
    out = cond(
        p_newton_norm < trust_radius, lambda _: (p_newton, False), lambda _: out, None
    )
    return *out, initial_alpha


@jit
def solve_trust_region_2d_subspace(g, H, trust_radius, initial_alpha=None, **kwargs):
    """Solve a trust region problem using 2d subspace method.

    Minimizes model function over subspace spanned by the gradient
    and Newton direction

    Parameters
    ----------
    g : ndarray
        gradient of objective function
    H : ndarray
        Hessian of objective function
    trust_radius : float
        We are allowed to wander only this far away from the origin.
    initial_alpha : float
        initial guess for Levenberg-Marquardt parameter - unused by this method

    Returns
    -------
    p : ndarray
        The proposed step.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        "Levenberg-Marquardt" parameter - unused by this method

    """
    L = chol(H)
    # This is the optimum for the quadratic model function.
    p_newton = -cho_solve((L, True), g)

    S = jnp.vstack([g, p_newton]).T
    S, _ = qr(S, mode="economic")
    g = S.T @ g
    B = S.T @ H @ S

    # B = [a b]  g = [d f]
    #     [b c]  q = [x y]
    # p = Sq                    # noqa: E800

    R, lower = cho_factor(B)
    q1 = -cho_solve((R, lower), g)
    p1 = S.dot(q1)

    a = B[0, 0] * trust_radius**2
    b = B[0, 1] * trust_radius**2
    c = B[1, 1] * trust_radius**2

    d = g[0] * trust_radius
    f = g[1] * trust_radius

    coeffs = jnp.array([-b + d, 2 * (a - c + f), 6 * b, 2 * (-a + c + f), -b - d])
    t = jnp.roots(coeffs, strip_zeros=False)
    t = jnp.where(jnp.isreal(t), jnp.real(t), jnp.nan)

    q2 = trust_radius * jnp.vstack((2 * t / (1 + t**2), (1 - t**2) / (1 + t**2)))
    value = 0.5 * jnp.sum(q2 * B.dot(q2), axis=0) + jnp.dot(g, q2)
    i = jnp.argmin(jnp.where(jnp.isnan(value), jnp.inf, value))
    q2 = q2[:, i]
    p2 = S.dot(q2)

    out = cond(
        jnp.dot(q1, q1) <= trust_radius**2,
        lambda _: (p1, True),
        lambda _: (p2, False),
        None,
    )
    return *out, initial_alpha


@jit
def trust_region_step_exact_svd(
    f, u, s, v, trust_radius, initial_alpha=0.0, rtol=0.01, max_iter=10, threshold=None
):
    """Solve a trust-region problem using a semi-exact method.

    Solves problems of the form
        min_p ||J*p + f||^2,  ||p|| < trust_radius

    Parameters
    ----------
    f : ndarray
        Vector of residuals
    u : ndarray
        Left singular vectors of J.
    s : ndarray
        Singular values of J.
    v : ndarray
        Right singular vectors of J (eg transpose of VT).
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy
        ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.
    threshold : float
        relative cutoff for small singular values

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.

    """
    uf = u.T.dot(f)
    suf = s * uf

    def phi_and_derivative(alpha, suf, s, trust_radius):
        """Function of which to find zero.

        It is defined as "norm of regularized (by alpha) least-squares
        solution minus `trust_radius`".
        """
        denom = s**2 + alpha
        denom = jnp.where(denom == 0, 1, denom)
        p = -v.dot(suf / denom)
        p_norm = jnp.linalg.norm(p)
        phi = p_norm - trust_radius
        phi_prime = -jnp.sum(suf**2 / denom**3) / p_norm
        return p, phi, phi_prime

    # Check if J has full rank and try Gauss-Newton step.
    threshold = setdefault(threshold, jnp.finfo(s.dtype).eps * f.size)
    threshold *= s[0]
    large = s > threshold
    s_inv = jnp.where(large, 1 / s, 0)

    p_newton = -v.dot(uf * s_inv)

    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(suf) / trust_radius
        alpha_lower = 0.0
        alpha = initial_alpha
        alpha = jnp.clip(alpha, alpha_lower, alpha_upper)

        _, phi, phi_prime = phi_and_derivative(alpha, suf, s, trust_radius)
        k = 0

        def loop_cond(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state
            return (jnp.abs(phi) > rtol * trust_radius) & (k < max_iter)

        def loop_body(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state

            p, phi, phi_prime = phi_and_derivative(alpha, suf, s, trust_radius)
            alpha_upper = jnp.where(phi < 0, alpha, alpha_upper)
            ratio = phi / phi_prime
            alpha_lower = jnp.maximum(alpha_lower, alpha - ratio)
            alpha -= (phi + trust_radius) * ratio / trust_radius
            alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
            k += 1
            return p, alpha, alpha_lower, alpha_upper, phi, k

        p, alpha, *_ = while_loop(
            loop_cond, loop_body, (p_newton, alpha, alpha_lower, alpha_upper, phi, k)
        )

        # Make the norm of p equal to trust_radius; p is changed only slightly.
        # This is done to prevent p from lying outside the trust region
        # (which can cause problems later).
        p *= trust_radius / jnp.linalg.norm(p)

        return p, True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


@jit
def trust_region_step_exact_cho(
    g, B, trust_radius, initial_alpha=0.0, rtol=0.01, max_iter=10
):
    """Solve a trust-region problem using a semi-exact method.

    Solves problems of the form
        (B + alpha*I)*p = -g,  ||p|| < trust_radius
    for symmetric B. A modified Cholesky factorization is used to deal with indefinite B

    Parameters
    ----------
    g : ndarray
        gradient vector
    B : ndarray
        Hessian or approximate Hessian
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy
        ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        Positive value such that (B + alpha*I)*p = -g.
        Sometimes called Levenberg-Marquardt parameter.

    """
    # try full newton step
    R = chol(B)
    p_newton = cho_solve((R, True), -g)

    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        alpha_upper = jnp.linalg.norm(g) / trust_radius
        alpha_lower = 0.0
        alpha = initial_alpha
        alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
        k = 0
        # algorithm 4.3 from Nocedal & Wright

        def loop_cond(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state
            return (jnp.abs(phi) > rtol * trust_radius) & (k < max_iter)

        def loop_body(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state

            Bi = B + alpha * jnp.eye(B.shape[0])
            R = chol(Bi)
            p = cho_solve((R, True), -g)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            alpha_upper = jnp.where(phi < 0, alpha, alpha_upper)
            alpha_lower = jnp.where(phi > 0, alpha, alpha_lower)

            q = solve_triangular(R.T, p, lower=False)
            q_norm = jnp.linalg.norm(q)

            alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
            alpha = jnp.clip(alpha, alpha_lower, alpha_upper)

            k += 1
            return p, alpha, alpha_lower, alpha_upper, phi, k

        p, alpha, *_ = while_loop(
            loop_cond,
            loop_body,
            (p_newton, alpha, alpha_lower, alpha_upper, jnp.inf, k),
        )

        # Make the norm of p equal to trust_radius; p is changed only slightly.
        # This is done to prevent p from lying outside the trust region
        # (which can cause problems later).
        p *= trust_radius / jnp.linalg.norm(p)

        return p, True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


@jit
def trust_region_step_exact_qr(
    p_newton, z, R, trust_radius, initial_alpha=1e-6, rtol=0.01, max_iter=10
):
    """Solve a trust-region problem using a semi-exact method.

    Solves problems of the form
        min_p ||J*p + f||^2,  ||p|| < trust_radius

    Introduces a Levenberg-Marquardt parameter alpha to make the problem
    well-conditioned.
        min_p ||J*p + f||^2 + alpha*||p||^2,  ||p|| < trust_radius

    The objective function can be written as
        p.T(J.T@J + alpha*I)p + 2f.TJp + f.Tf
    which is equivalent to
        || [J; sqrt(alpha)*I].Tp - [f; 0].T ||^2

    The caller supplies the factorization ``J = Q1@R`` (and ``z = Q1.T@f``), so
    the alpha-loop only retriangularizes the small reduced system
    ``[R; sqrt(alpha)*I]`` instead of refactorizing ``J`` each iteration.

    Parameters
    ----------
    p_newton : ndarray
        The full (unregularized) Newton step, returned as-is if it lies within
        the trust region.
    z : ndarray
        ``Q1.T@f``, where ``J = Q1@R`` is the (economic) QR factorization of J.
    R : ndarray
        The R factor of J, as returned by ``qr_multiply(J, f, mode="right")``.
    trust_radius : float
        Radius of a trust region.
    initial_alpha : float, optional
        Initial guess for alpha, which might be available from a previous
        iteration. If None, determined automatically.
    rtol : float, optional
        Stopping tolerance for the root-finding procedure. Namely, the
        solution ``p`` will satisfy
        ``abs(norm(p) - trust_radius) < rtol * trust_radius``.
    max_iter : int, optional
        Maximum allowed number of iterations for the root-finding procedure.

    Returns
    -------
    p : ndarray, shape (n,)
        Found solution of a trust-region problem.
    hits_boundary : bool
        True if the proposed step is on the boundary of the trust region.
    alpha : float
        Positive value such that (J.T*J + alpha*I)*p = -J.T*f.
        Sometimes called Levenberg-Marquardt parameter.

    """

    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        # J.T@f == R.T@z, so we never need J or f here
        alpha_upper = jnp.linalg.norm(R.T @ z) / trust_radius
        alpha_lower = 0.0
        alpha = initial_alpha
        alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
        k = 0

        n = R.shape[1]
        zp = jnp.concatenate([z, jnp.zeros(n)])

        def loop_cond(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state
            return (jnp.abs(phi) > rtol * trust_radius) & (k < max_iter)

        def loop_body(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state

            A = jnp.vstack([R, jnp.sqrt(alpha) * jnp.eye(n)])
            Qtz, Rtil = qr_multiply(A, zp, mode="right")

            p = solve_triangular_regularized(Rtil, -Qtz)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            alpha_upper = jnp.where(phi < 0, alpha, alpha_upper)
            alpha_lower = jnp.where(phi > 0, alpha, alpha_lower)

            q = solve_triangular_regularized(Rtil.T, p, lower=True)
            q_norm = jnp.linalg.norm(q)

            alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
            alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
            k += 1
            return p, alpha, alpha_lower, alpha_upper, phi, k

        p, alpha, *_ = while_loop(
            loop_cond,
            loop_body,
            (p_newton, alpha, alpha_lower, alpha_upper, jnp.inf, k),
        )

        # Make the norm of p equal to trust_radius; p is changed only slightly.
        # This is done to prevent p from lying outside the trust region
        # (which can cause problems later).
        p *= trust_radius / jnp.linalg.norm(p)

        return p, True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


# The work-array row index is [c0, c1) U [n, n + c1) with c1 <= n: two disjoint
# ranges, so already sorted ascending, unique, and in bounds. XLA cannot infer
# that, and without it a scatter must assume its indices may COLLIDE -- which
# needs a combiner, and on GPU that means atomics. Atomics are the fast path but
# are order-nondeterministic, so under --xla_gpu_deterministic_ops=true XLA
# substitutes a serialised scatter. Asserting uniqueness removes the combiner and
# with it the whole penalty.
_IDX_HINTS = dict(
    indices_are_sorted=True, unique_indices=True, mode="promise_in_bounds"
)


def _apply_QT_wy(at, taus, C):
    """Apply Q.T to C, with Q the product of the Householder reflectors in ``at``.

    ``at`` is ``(k, M)``: the reflectors as ROWS, i.e. ``V.T``, which is exactly
    what ``jnp.linalg.qr(..., mode="raw")`` hands back, so the ``(M, k)``
    transpose is never formed and all three products come from dot dimension
    numbers alone. Uses the compact-WY / UT transform via the identity
    ``T^-1 + T^-H = V^H V`` (Joffrain & Low 2006), so each application is two
    matmuls and a triangular solve rather than ``k`` rank-1 updates.

    Mirrors ``desc.backend``'s ``_householder_multiply``. Duplicated rather than
    imported because that helper exists only on the ``jax < 0.10`` fallback
    path: on ``jax >= 0.10`` DESC takes ``qr_multiply`` from
    ``jax.scipy.linalg`` and the helper is not defined at all, so importing it
    would break as soon as the minimum jax version moves.

    One deliberate difference from the backend version: the ``tau == 0`` guard
    below. A zero tau marks an identity reflector, and that case is not exotic
    here -- at ``alpha = 0`` the ``sqrt(alpha)*I`` block is entirely zero, the
    panel is already triangular, and every tau comes back zero. Without the
    guard the diagonal correction is ``1/0``.
    """
    k, m = at.shape
    # V is unit lower-trapezoidal, so V.T (== at) is unit upper-trapezoidal.
    below = jnp.arange(m)[None, :] - jnp.arange(k)[:, None]
    Vt = jnp.where(below > 0, at, (below == 0).astype(at.dtype))
    live = taus != 0
    Vt = Vt * live[:, None]
    # solve_triangular reads only the relevant triangle, so passing the full
    # Gram matrix V V.T minus the diagonal correction recovers T^-1.
    T_inv = Vt @ Vt.T - jnp.diag(1.0 / jnp.where(live, taus, 1.0))
    return C - Vt.T @ solve_triangular(T_inv, Vt @ C, lower=True)


@functools.partial(jit, static_argnames=("block",))
def structured_retriangularize_fixed(R, z, alpha, block=128):
    """QR of ``[R; sqrt(alpha)*I]``: the fixed work array, trailing columns only.

    This is ``structured_retriangularize`` with the first of
    ``structured_retriangularize_slim``'s two changes and not the second, which
    is the combination that measurement supports:

    * **Kept (slim's change 1).** ``Q.T`` is applied only to the TRAILING
      columns. The panel QR already produced the transformed panel columns --
      they are ``triu(packed[:bk, :bk])`` -- so pushing them back through the
      compact-WY transform is redundant arithmetic and a second floating-point
      path to a known quantity. This carries most of slim's speed advantage and
      all of its accuracy advantage: not doing arithmetic cannot round it.
    * **Dropped (slim's change 2).** The frontier is NOT carried at its exact
      live shape. ``block`` is static, so the panel loop is fully unrolled;
      carrying a differently-shaped frontier per panel gives XLA
      ``ceil(n/block)`` distinct allocations whose liveness its scheduler cannot
      collapse, and planned temp memory becomes ``O(n^3 / block)`` instead of
      ``O(n^2)``. Measured on an A100-80GB at n=14242, b=128, that is 37.8 GB of
      planned temporaries against this routine's ~3.4 GB, and compile-only
      analysis matches the observed runtime peak to within 5%. Keeping the one
      fixed ``2n x (n+1)`` buffer lets XLA update in place and free each panel's
      temporaries immediately.

    The gather and the scatter carry index hints (see ``_IDX_HINTS``). They do
    not change the result -- verified bitwise identical without them -- nor the
    peak memory, but they make the routine immune to deterministic-ops mode:
    measured at n=14242, b=512, the unhinted form goes 512.3 -> 3695.1 ms under
    ``--xla_gpu_deterministic_ops=true`` (7.2x) while this one is unchanged
    (506.2 -> 506.8 ms). Dense ``qr`` is unaffected either way, since cuSOLVER's
    geqrf is already deterministic.

    Writes back in two pieces rather than one: the panel's triangular factor
    goes to the top ``bk`` rows by static slice, and ``Q.T``-transformed trailing
    columns go to the gathered rows. The panel columns of the bottom rows are
    left stale deliberately -- they are never read again (panel ``k+1`` reads
    only columns ``>= c1``) and never returned (only ``M[:n]`` is).

    Parameters
    ----------
    R : ndarray, shape (k, n)
        R factor of J, from ``qr_multiply(J, f, mode="right")``.
    z : ndarray, shape (k,)
        ``Q1.T@f``.
    alpha : float
        Levenberg-Marquardt parameter.
    block : int
        Column-panel width. Must be static under jit.

    Returns
    -------
    Rtil : ndarray, shape (n, n)
        Upper triangular, with ``Rtil.T@Rtil == R.T@R + alpha*I``.
    Qtz : ndarray, shape (n,)
        The transformed right-hand side.

    """
    n = R.shape[1]
    k_ = R.shape[0]
    dt = R.dtype
    # Work array: [R; sqrt(alpha)*I] with the RHS carried as a trailing column.
    M = jnp.zeros((2 * n, n + 1), dt)
    M = M.at[:k_, :n].set(R)
    M = M.at[n:, :n].set(jnp.sqrt(jnp.asarray(alpha, dt)) * jnp.eye(n, dtype=dt))
    M = M.at[:k_, n].set(z)

    for kb in range((n + block - 1) // block):
        c0 = kb * block
        c1 = min(c0 + block, n)
        bk = c1 - c0
        idx = jnp.concatenate([jnp.arange(c0, c1), n + jnp.arange(0, c1)])
        Sub = M.at[idx, c0:].get(**_IDX_HINTS)
        # mode="raw" gives h as (k, M) -- V.T already, so no transpose is formed.
        h, taus = jnp.linalg.qr(Sub[:, :bk], mode="raw")
        # Change 1: take the panel's R from the reflector block, then apply Q.T
        # to the remaining columns only.
        trail = _apply_QT_wy(h, taus, Sub[:, bk:])
        M = M.at[c0:c1, c0:c1].set(jnp.triu(h[:bk, :bk].T))
        M = M.at[idx, c1:].set(trail, **_IDX_HINTS)

    return jnp.triu(M[:n, :n]), M[:n, n]


@functools.partial(jit, static_argnames=("block",))
def trust_region_step_exact_qr_fixed(
    p_newton, z, R, trust_radius, initial_alpha=1e-6, rtol=0.01, max_iter=10, block=128
):
    """Solve a trust-region problem using the fixed-buffer retriangularization.

    Identical to ``trust_region_step_exact_qr_struct`` except that each alpha
    iteration calls ``structured_retriangularize_fixed``. Same safeguarded
    Hebden/Reinsch root-find, same iterates, so it produces the same alpha
    sequence as ``tr_method="qr"`` by construction.

    Parameters and returns are as ``trust_region_step_exact_qr``, plus:

    Parameters
    ----------
    block : int
        Column-panel width passed to ``structured_retriangularize_fixed``.
        Peak memory is flat in this parameter, unlike the ``"qr-slim"`` route,
        so it trades only speed.

    """

    def truefun(*_):
        return p_newton, False, 0.0

    def falsefun(*_):
        # J.T@f == R.T@z, so we never need J or f here
        alpha_upper = jnp.linalg.norm(R.T @ z) / trust_radius
        alpha_lower = 0.0
        alpha = initial_alpha
        alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
        k = 0

        def loop_cond(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state
            return (jnp.abs(phi) > rtol * trust_radius) & (k < max_iter)

        def loop_body(state):
            p, alpha, alpha_lower, alpha_upper, phi, k = state

            Rtil, Qtz = structured_retriangularize_fixed(R, z, alpha, block=block)

            p = solve_triangular_regularized(Rtil, -Qtz)
            p_norm = jnp.linalg.norm(p)
            phi = p_norm - trust_radius
            alpha_upper = jnp.where(phi < 0, alpha, alpha_upper)
            alpha_lower = jnp.where(phi > 0, alpha, alpha_lower)

            q = solve_triangular_regularized(Rtil.T, p, lower=True)
            q_norm = jnp.linalg.norm(q)

            alpha += (p_norm / q_norm) ** 2 * phi / trust_radius
            alpha = jnp.clip(alpha, alpha_lower, alpha_upper)
            k += 1
            return p, alpha, alpha_lower, alpha_upper, phi, k

        p, alpha, *_ = while_loop(
            loop_cond,
            loop_body,
            (p_newton, alpha, alpha_lower, alpha_upper, jnp.inf, k),
        )

        # Make the norm of p equal to trust_radius; p is changed only slightly.
        # This is done to prevent p from lying outside the trust region
        # (which can cause problems later).
        p *= trust_radius / jnp.linalg.norm(p)

        return p, True, alpha

    return cond(jnp.linalg.norm(p_newton) <= trust_radius, truefun, falsefun, None)


def update_tr_radius(
    trust_radius,
    actual_reduction,
    predicted_reduction,
    step_norm,
    bound_hit,
    max_tr=np.inf,
    increase_threshold=0.75,
    increase_ratio=2,
    decrease_threshold=0.25,
    decrease_ratio=0.25,
):
    """Update the radius of a trust region based on the cost reduction.

    Parameters
    ----------
    trust_radius : float
        current trust region radius
    actual_reduction : float
        actual cost reduction from the proposed step
    predicted_reduction : float
        cost reduction predicted by quadratic model
    step_norm : float
        size of the proposed step
    bound_hit : bool
        whether the current step hits the trust region bound
    max_tr : float
        maximum allowed trust region radius
    increase_threshold, increase_ratio : float
        if ratio > increase_threshold, trust radius is increased by a factor
        of increase_ratio
    decrease_threshold, decrease_ratio : float
        if ratio < decrease_threshold, trust radius is decreased by a factor
        of decrease_ratio

    Returns
    -------
    trust_radius : float
        New radius.
    reduction_ratio : float
        Ratio between actual and predicted reductions.
    """
    if predicted_reduction > 0:
        reduction_ratio = actual_reduction / predicted_reduction
    elif predicted_reduction == actual_reduction == 0:
        reduction_ratio = 1
    else:
        reduction_ratio = 0

    if reduction_ratio < decrease_threshold or np.isnan(reduction_ratio):
        trust_radius = decrease_ratio * step_norm
    elif reduction_ratio > increase_threshold:
        trust_radius = max(step_norm * increase_ratio, trust_radius)

    trust_radius = np.clip(trust_radius, 0, max_tr)

    return trust_radius, reduction_ratio


def get_boundaries_intersections(z, d, trust_radius):
    """Solve the scalar quadratic equation ||z + t d|| == trust_radius.

    This is like a line-sphere intersection.
    Return the two values of t, sorted from low to high.
    """
    a = jnp.dot(d, d)
    b = 2 * jnp.dot(z, d)
    c = jnp.dot(z, z) - trust_radius**2
    # abs to catch possible floating point errors for near duplicate roots
    sqrt_discriminant = jnp.sqrt(jnp.abs(b * b - 4 * a * c))

    # The following calculation is mathematically
    # equivalent to:
    # ta = (-b - sqrt_discriminant) / (2*a)    # noqa: E800
    # tb = (-b + sqrt_discriminant) / (2*a)    # noqa: E800
    # but produce smaller round off errors.
    # Look at Matrix Computation p.97
    # for a better justification.
    aux = b + jnp.copysign(sqrt_discriminant, b)
    ta = -aux / (2 * a)
    tb = -2 * c / aux
    return jnp.sort(jnp.array([ta, tb]))
