"""Second-order near-axis QS (Garren-Boozer / Landreman & Sengupta 2019) in JAX, vacuum and stellarator symmetric
(sG = spsi = 1, I2 = p2 = B2s = 0), on top of nae.NearAxis.solve.

second_order(q, B2c) -> dict with X20, X2s, X2c, Y20, Y2s, Y2c, Z20, Z2s, Z2c, B20, B20 anomaly / residual, and
GG: the grad grad B tensor on the axis in Cartesian components, GG[q, i, j, k] = d_i d_j B_k (fully symmetric and
traceless in vacuum).

calculate_r2 is a line-by-line port of pyQSC 0.1.3 (pylib/qsc/calculate_r2.py) with the vacuum terms dropped.
The grad grad B tensor (27 machine-generated expressions in the (normal, binormal, tangent) frame) is taken from
pyQSC's own source at import, evaluated with jax.numpy, then rotated to Cartesian with the Frenet vectors.
"""
import inspect, os, re, sys, types
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "pylib"))
from qsc import grad_B_tensor as _gbt  # noqa: E402


def calculate_r2(q, B2c):
    Dv = q["Dv"]
    X1c, Y1s, Y1c, sigma = q["X1c"], q["Y1s"], q["Y1c"], q["sigma"]
    iota_N, curvature, torsion, etabar, B0 = q["iotaN"], q["kappa"], q["tors"], q["etabar"], q["B0"]
    abs_G0_over_B0 = q["abs_G0"] / B0
    B0_over_abs_G0 = 1 / abs_G0_over_B0
    nphi = X1c.shape[0]
    mm = lambda v: Dv @ v  # noqa: E731

    V1 = X1c * X1c + Y1c * Y1c + Y1s * Y1s
    V2 = 2 * Y1s * Y1c
    V3 = X1c * X1c + Y1c * Y1c - Y1s * Y1s
    factor = -B0_over_abs_G0 / 8
    Z20 = factor * mm(V1)
    Z2s = factor * (mm(V2) - 2 * iota_N * V3)
    Z2c = factor * (mm(V3) + 2 * iota_N * V2)

    qs = -iota_N * X1c - Y1s * torsion * abs_G0_over_B0
    qc = mm(X1c) - Y1c * torsion * abs_G0_over_B0
    rs = mm(Y1s) - iota_N * Y1c
    rc = mm(Y1c) + iota_N * Y1s + X1c * torsion * abs_G0_over_B0

    X2s = B0_over_abs_G0 * (mm(Z2s) - 2 * iota_N * Z2c + B0_over_abs_G0 * ((qc * qs + rc * rs) / 2)) / curvature
    X2c = B0_over_abs_G0 * (mm(Z2c) + 2 * iota_N * Z2s - B0_over_abs_G0 * (
        -abs_G0_over_B0 * abs_G0_over_B0 * B2c / B0 + abs_G0_over_B0 * abs_G0_over_B0 * etabar * etabar / 2
        - (qc * qc - qs * qs + rc * rc - rs * rs) / 4)) / curvature

    Y2s_from_X20 = -curvature * curvature / (etabar * etabar)
    Y2s_inh = -curvature / 2 + curvature * curvature / (etabar * etabar) * (-X2c + X2s * sigma)
    Y2c_from_X20 = -curvature * curvature * sigma / (etabar * etabar)
    Y2c_inh = curvature * curvature / (etabar * etabar) * (X2s + X2c * sigma)

    A = abs_G0_over_B0
    fX0_from_X20 = -4 * A * (Y2c_from_X20 * Z2s - Y2s_from_X20 * Z2c)
    fX0_from_Y20 = -torsion * A - 4 * A * Z2s
    fX0_inh = curvature * A * Z20 - 4 * A * (Y2c_inh * Z2s - Y2s_inh * Z2c)
    fXs_from_X20 = -torsion * A * Y2s_from_X20 - 4 * A * (Y2c_from_X20 * Z20)
    fXs_from_Y20 = -4 * A * (-Z2c + Z20)
    fXs_inh = mm(X2s) - 2 * iota_N * X2c - torsion * A * Y2s_inh + curvature * A * Z2s - 4 * A * (Y2c_inh * Z20)
    fXc_from_X20 = -torsion * A * Y2c_from_X20 - 4 * A * (-Y2s_from_X20 * Z20)
    fXc_from_Y20 = -torsion * A - 4 * A * Z2s
    fXc_inh = mm(X2c) + 2 * iota_N * X2s - torsion * A * Y2c_inh + curvature * A * Z2c - 4 * A * (-Y2s_inh * Z20)
    fY0_from_X20 = torsion * A + 0 * X1c
    fY0_from_Y20 = jnp.zeros(nphi)
    fY0_inh = -4 * A * (X2s * Z2c - X2c * Z2s)
    fYs_from_X20 = -2 * iota_N * Y2c_from_X20 - 4 * A * Z2c
    fYs_from_Y20 = jnp.full(nphi, -2 * iota_N)
    fYs_inh = mm(Y2s_inh) - 2 * iota_N * Y2c_inh + torsion * A * X2s - 4 * A * (-X2c * Z20)
    fYc_from_X20 = 2 * iota_N * Y2s_from_X20 - 4 * A * (-Z2s)
    fYc_from_Y20 = jnp.zeros(nphi)
    fYc_inh = mm(Y2c_inh) + 2 * iota_N * Y2s_inh + torsion * A * X2c - 4 * A * (X2s * Z20)

    # the 2 nphi x 2 nphi system for (X20, Y20), assembled as in pyQSC
    M11 = Y1c[:, None] * Dv * Y2s_from_X20[None] - Y1s[:, None] * Dv * Y2c_from_X20[None]
    M12 = -2 * Y1s[:, None] * Dv
    M21 = -X1c[:, None] * Dv + Y1s[:, None] * Dv * Y2s_from_X20[None] + Y1c[:, None] * Dv * Y2c_from_X20[None]
    M22 = jnp.zeros((nphi, nphi))
    d11 = X1c * fXs_from_X20 - Y1s * fY0_from_X20 + Y1c * fYs_from_X20 - Y1s * fYc_from_X20
    d12 = X1c * fXs_from_Y20 - Y1s * fY0_from_Y20 + Y1c * fYs_from_Y20 - Y1s * fYc_from_Y20
    d21 = -X1c * fX0_from_X20 + X1c * fXc_from_X20 - Y1c * fY0_from_X20 + Y1s * fYs_from_X20 + Y1c * fYc_from_X20
    d22 = -X1c * fX0_from_Y20 + X1c * fXc_from_Y20 - Y1c * fY0_from_Y20 + Y1s * fYs_from_Y20 + Y1c * fYc_from_Y20
    M = jnp.block([[M11 + jnp.diag(d11), M12 + jnp.diag(d12)], [M21 + jnp.diag(d21), M22 + jnp.diag(d22)]])
    rhs = jnp.concatenate([-(X1c * fXs_inh - Y1s * fY0_inh + Y1c * fYs_inh - Y1s * fYc_inh),
                           -(-X1c * fX0_inh + X1c * fXc_inh - Y1c * fY0_inh + Y1s * fYs_inh + Y1c * fYc_inh)])
    sol = jnp.linalg.solve(M, rhs)
    X20, Y20 = sol[:nphi], sol[nphi:]
    Y2s = Y2s_inh + Y2s_from_X20 * X20
    Y2c = Y2c_inh + Y2c_from_X20 * X20 + Y20
    B20 = B0 * (curvature * X20 - B0_over_abs_G0 * mm(Z20) + 0.5 * etabar * etabar
                - 0.25 * B0_over_abs_G0 * B0_over_abs_G0 * (qc * qc + qs * qs + rc * rc + rs * rs))
    w = q["dl"] / jnp.sum(q["dl"])
    B20_mean = jnp.sum(B20 * w)
    return dict(X20=X20, X2s=X2s, X2c=X2c, Y20=Y20, Y2s=Y2s, Y2c=Y2c, Z20=Z20, Z2s=Z2s, Z2c=Z2c, B20=B20,
                B20_mean=B20_mean, B20_anomaly=B20 - B20_mean,
                B20_residual=jnp.sqrt(jnp.sum((B20 - B20_mean) ** 2 * w)) / B0)


# ---- grad grad B: pyQSC's generated expressions, executed with jax.numpy
_src = inspect.getsource(_gbt.calculate_grad_grad_B_tensor)
_body = _src[_src.index("    grad_grad_B = np.zeros"):_src.index("    self.grad_grad_B = grad_grad_B")]
_body = _body.replace("grad_grad_B = np.zeros((s.nphi, 3, 3, 3))", "grad_grad_B = _Collect()")
_body = _body.replace("grad_grad_B_alt = np.zeros((s.nphi, 3, 3, 3))", "grad_grad_B_alt = None")
_body = re.sub(r"^    ", "", _body, flags=re.M)
_head = _src[_src.index("    s = self"):_src.index("    grad_grad_B = np.zeros")]
_head = re.sub(r"^    ", "", _head, flags=re.M)


class _Collect(dict):
    def __setitem__(self, key, value):
        super().__setitem__(tuple(key[1:]), value)


_CODE = compile(_head + _body, "<pyqsc grad_grad_B>", "exec")


def grad_grad_B(q, r2):
    """grad grad B on the axis, (nphi, 3, 3, 3), Cartesian, from pyQSC's (n, b, t)-frame expressions."""
    Dv = q["Dv"]
    s = types.SimpleNamespace(
        X1c=q["X1c"], Y1s=q["Y1s"], Y1c=q["Y1c"], iotaN=q["iotaN"], iota=q["iota"], G0=q["abs_G0"], B0=q["B0"],
        curvature=q["kappa"], torsion=q["tors"], sG=1, spsi=1, I2=0.0, G2=0.0, p2=0.0, B2s=0.0,
        B20=r2["B20"], B2c=r2.get("B2c", 0.0), nphi=q["X1c"].shape[0],
        d_X1c_d_varphi=q["dX1c"], d_Y1s_d_varphi=q["dY1s"], d_Y1c_d_varphi=q["dY1c"],
        d2_X1c_d_varphi2=Dv @ q["dX1c"], d2_Y1s_d_varphi2=Dv @ q["dY1s"], d2_Y1c_d_varphi2=Dv @ q["dY1c"],
        d_curvature_d_varphi=Dv @ q["kappa"], d_torsion_d_varphi=Dv @ q["tors"],
        **{k: r2[k] for k in ("X20", "X2s", "X2c", "Y20", "Y2s", "Y2c", "Z20", "Z2s", "Z2c")},
        **{f"d_{k}_d_varphi": Dv @ r2[k] for k in ("X20", "X2s", "X2c", "Y20", "Y2s", "Y2c", "Z20", "Z2s", "Z2c")})
    ns = {"self": s, "np": jnp, "_Collect": _Collect}
    exec(_CODE, ns)
    C = ns["grad_grad_B"]
    T = jnp.stack([jnp.stack([jnp.stack([jnp.broadcast_to(C[(i, j, k)], q["X1c"].shape) for k in range(3)], -1)
                              for j in range(3)], -1) for i in range(3)], -1)  # T[q, k, j, i]
    T = jnp.transpose(T, (0, 3, 2, 1))  # T[q, i, j, k] in the (n, b, t) frame
    E = jnp.stack([q["n"], q["b"], q["t"]], 1)  # E[q, a, x]: frame vector a, Cartesian component x
    return jnp.einsum("qabc,qai,qbj,qck->qijk", T, E, E, E)


def second_order(q, B2c):
    r2 = calculate_r2(q, B2c)
    r2["B2c"] = B2c
    r2["GG"] = grad_grad_B(q, r2)
    return r2
