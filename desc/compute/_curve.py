from interpax import interp1d

from desc.backend import jnp, sign, vmap

from ..utils import (
    cross,
    dot,
    rotation_matrix,
    rpz2xyz,
    rpz2xyz_vec,
    safearccos,
    safenormalize,
    xyz2rpz,
    xyz2rpz_vec,
)
from .data_index import register_compute_fun


@register_compute_fun(
    name="s",
    label="s",
    units="~",
    units_long="None",
    description="Curve parameter, on [0, 2pi)",
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.core.Curve",
)
def _s(params, transforms, profiles, data, **kwargs):
    data["s"] = transforms["grid"].nodes[:, 2]
    return data


@register_compute_fun(
    name="ds",
    label="ds",
    units="~",
    units_long="None",
    description=(
        "Quadrature weights for integration along the curve,"
        + " i.e. an alias for ``grid.spacing[:,2]``"
    ),
    dim=1,
    params=[],
    transforms={"grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.core.Curve",
)
def _ds(params, transforms, profiles, data, **kwargs):
    data["ds"] = transforms["grid"].spacing[:, 2]
    return data


@register_compute_fun(
    name="X",
    label="X",
    units="m",
    units_long="meters",
    description="Cartesian X coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _X_curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    coords = rpz2xyz(coords)
    data["X"] = coords[:, 0]
    return data


@register_compute_fun(
    name="Y",
    label="Y",
    units="m",
    units_long="meters",
    description="Cartesian Y coordinate.",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _Y_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    coords = rpz2xyz(coords)
    data["Y"] = coords[:, 1]
    return data


@register_compute_fun(
    name="R",
    label="R",
    units="m",
    units_long="meters",
    description="Cylindrical radial position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _R_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["R"] = coords[:, 0]
    return data


@register_compute_fun(
    name="phi",
    label="\\phi",
    units="rad",
    units_long="radians",
    description="Toroidal phi position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _phi_Curve(params, transforms, profiles, data, **kwargs):
    coords = data["x"]
    data["phi"] = coords[:, 1]
    return data


@register_compute_fun(
    name="Z",
    label="Z",
    units="m",
    units_long="meters",
    description="Cylindrical vertical position along curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.core.Curve",
)
def _Z_Curve(params, transforms, profiles, data, **kwargs):
    data["Z"] = data["x"][:, 2]
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["center", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization=[
        "desc.geometry.curve.FourierPlanarCurve",
        "desc.geometry.curve.FourierXYCurve",
    ],
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _center_PlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
    else:
        center = params["center"]
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["r_n", "center", "normal", "rotmat", "shift"],
    transforms={"r": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]
    # create planar curve at Z==0
    r = transforms["r"].transform(params["r_n"], dz=0)
    Z = jnp.zeros_like(r)
    X = r * jnp.cos(data["s"])
    Y = r * jnp.sin(data["s"])
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T) + center
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_s_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    dX = dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    dY = dr * jnp.sin(data["s"]) + r * jnp.cos(data["s"])
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1], [0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_ss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    d2r = transforms["r"].transform(params["r_n"], dz=2)
    d2X = (
        d2r * jnp.cos(data["s"]) - 2 * dr * jnp.sin(data["s"]) - r * jnp.cos(data["s"])
    )
    d2Y = (
        d2r * jnp.sin(data["s"]) + 2 * dr * jnp.cos(data["s"]) - r * jnp.sin(data["s"])
    )
    d2Z = jnp.zeros_like(d2X)
    coords = jnp.array([d2X, d2Y, d2Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["r_n", "center", "normal", "rotmat"],
    transforms={"r": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.FourierPlanarCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_sss_FourierPlanarCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    r = transforms["r"].transform(params["r_n"], dz=0)
    dr = transforms["r"].transform(params["r_n"], dz=1)
    d2r = transforms["r"].transform(params["r_n"], dz=2)
    d3r = transforms["r"].transform(params["r_n"], dz=3)
    d3X = (
        d3r * jnp.cos(data["s"])
        - 3 * d2r * jnp.sin(data["s"])
        - 3 * dr * jnp.cos(data["s"])
        + r * jnp.sin(data["s"])
    )
    d3Y = (
        d3r * jnp.sin(data["s"])
        + 3 * d2r * jnp.cos(data["s"])
        - 3 * dr * jnp.sin(data["s"])
        - r * jnp.cos(data["s"])
    )
    d3Z = jnp.zeros_like(d3X)
    coords = jnp.array([d3X, d3Y, d3Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        center = rpz2xyz(params["center"])
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        center = params["center"]
        normal = params["normal"]
    # create planar curve at Z==0
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = jnp.zeros_like(X)
    coords = jnp.array([X, Y, Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T) + center
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 1]], "Y": [[0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_s_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = jnp.zeros_like(dX)
    coords = jnp.array([dX, dY, dZ]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 2]], "Y": [[0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_ss_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = jnp.zeros_like(d2X)
    coords = jnp.array([d2X, d2Y, d2Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "center", "normal", "rotmat"],
    transforms={"X": [[0, 0, 3]], "Y": [[0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYCurve",
    basis_in="{'rpz', 'xyz'}: Basis for input params vectors, Default 'xyz'",
)
def _x_sss_FourierXYCurve(params, transforms, profiles, data, **kwargs):
    # convert to xyz for displacement and rotation
    if kwargs.get("basis_in", "xyz").lower() == "rpz":
        normal = rpz2xyz_vec(params["normal"], phi=params["center"][1])
    else:
        normal = params["normal"]
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = jnp.zeros_like(d3X)
    coords = jnp.array([d3X, d3Y, d3Z]).T
    # rotate into place
    Zaxis = jnp.array([0.0, 0.0, 1.0])  # 2D curve in X-Y plane has normal = +Z axis
    axis = cross(Zaxis, normal)
    dotprod = dot(Zaxis, safenormalize(normal))
    angle = safearccos(dotprod)
    A = jnp.where(  # handle the case where normal is aligned with the -Z axis
        jnp.allclose(dotprod, -1.0),
        jnp.diag(jnp.array([1.0, -1.0, -1.0])),
        rotation_matrix(axis, angle),
    )
    coords = jnp.matmul(coords, A.T)
    coords = jnp.matmul(coords, params["rotmat"].reshape((3, 3)).T)
    # convert back to rpz
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["R_n", "Z_n", "rotmat", "shift"],
    transforms={"R": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _center_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    idx_Rc = transforms["R"].basis.get_idx(N=1, error=False)
    idx_Rs = transforms["R"].basis.get_idx(N=-1, error=False)
    idx_Z = transforms["Z"].basis.get_idx(N=0, error=False)
    X0 = params["R_n"][idx_Rc] / 2 if isinstance(idx_Rc, int) else 0
    Y0 = params["R_n"][idx_Rs] / 2 if isinstance(idx_Rs, int) else 0
    Z0 = params["Z_n"][idx_Z] if isinstance(idx_Z, int) else 0
    center = jnp.array([X0, Y0, Z0])
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert back to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["R_n", "Z_n", "rotmat", "shift"],
    transforms={"R": [[0, 0, 0]], "Z": [[0, 0, 0]], "grid": []},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R = transforms["R"].transform(params["R_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    phi = transforms["grid"].nodes[:, 2]
    coords = jnp.stack([R, phi, Z], axis=1)
    # convert to xyz for displacement and rotation
    coords = rpz2xyz(coords)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    # convert back to rpz
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={"R": [[0, 0, 0], [0, 0, 1]], "Z": [[0, 0, 1]], "grid": []},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_s_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    dR = transforms["R"].transform(params["R_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dR, R0, dZ], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={"R": [[0, 0, 0], [0, 0, 1], [0, 0, 2]], "Z": [[0, 0, 2]], "grid": []},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_ss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    d1R = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2R - R0, 2 * d1R, d2Z], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["R_n", "Z_n", "rotmat"],
    transforms={
        "R": [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3]],
        "Z": [[0, 0, 3]],
        "grid": [],
    },
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierRZCurve",
)
def _x_sss_FourierRZCurve(params, transforms, profiles, data, **kwargs):
    R0 = transforms["R"].transform(params["R_n"], dz=0)
    d1R = transforms["R"].transform(params["R_n"], dz=1)
    d2R = transforms["R"].transform(params["R_n"], dz=2)
    d3R = transforms["R"].transform(params["R_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3R - 3 * d1R, 3 * d2R - R0, d3Z], axis=1)
    # convert to xyz for rotation using phi=s
    coords = rpz2xyz_vec(coords, phi=transforms["grid"].nodes[:, 2])
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    # convert back to rpz using real phi to account for displacement
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _center_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    idx_X = transforms["X"].basis.get_idx(N=0, error=False)
    idx_Y = transforms["Y"].basis.get_idx(N=0, error=False)
    idx_Z = transforms["Z"].basis.get_idx(N=0, error=False)
    X0 = params["X_n"][idx_X] if isinstance(idx_X, int) else 0
    Y0 = params["Y_n"][idx_Y] if isinstance(idx_Y, int) else 0
    Z0 = params["Z_n"][idx_Z] if isinstance(idx_Z, int) else 0
    center = jnp.array([X0, Y0, Z0])
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat", "shift"],
    transforms={"X": [[0, 0, 0]], "Y": [[0, 0, 0]], "Z": [[0, 0, 0]]},
    profiles=[],
    coordinates="s",
    data=[],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    X = transforms["X"].transform(params["X_n"], dz=0)
    Y = transforms["Y"].transform(params["Y_n"], dz=0)
    Z = transforms["Z"].transform(params["Z_n"], dz=0)
    coords = jnp.stack([X, Y, Z], axis=1)
    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 1]], "Y": [[0, 0, 1]], "Z": [[0, 0, 1]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_s_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    dX = transforms["X"].transform(params["X_n"], dz=1)
    dY = transforms["Y"].transform(params["Y_n"], dz=1)
    dZ = transforms["Z"].transform(params["Z_n"], dz=1)
    coords = jnp.stack([dX, dY, dZ], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_s"] = coords
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 2]], "Y": [[0, 0, 2]], "Z": [[0, 0, 2]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_ss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d2X = transforms["X"].transform(params["X_n"], dz=2)
    d2Y = transforms["Y"].transform(params["Y_n"], dz=2)
    d2Z = transforms["Z"].transform(params["Z_n"], dz=2)
    coords = jnp.stack([d2X, d2Y, d2Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_ss"] = coords
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X_n", "Y_n", "Z_n", "rotmat"],
    transforms={"X": [[0, 0, 3]], "Y": [[0, 0, 3]], "Z": [[0, 0, 3]]},
    profiles=[],
    coordinates="s",
    data=["phi"],
    parameterization="desc.geometry.curve.FourierXYZCurve",
)
def _x_sss_FourierXYZCurve(params, transforms, profiles, data, **kwargs):
    d3X = transforms["X"].transform(params["X_n"], dz=3)
    d3Y = transforms["Y"].transform(params["Y_n"], dz=3)
    d3Z = transforms["Z"].transform(params["Z_n"], dz=3)
    coords = jnp.stack([d3X, d3Y, d3Z], axis=1)
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    coords = xyz2rpz_vec(coords, phi=data["phi"])
    data["x_sss"] = coords
    return data


def _splinexyz_helper(f, transforms, s_query_pts, method, derivative):
    """Used to compute XYZ coordinates for the SplineXYZCurve compute functions.

    Parameters
    ----------
    f : list of ndarray
        X, Y, Z function coords with shape (3, len(transforms["knots"]))
    transforms : dict
        the transforms from the compute function
    s_query_pts : ndarray
        query points that come from s parameterization
    kwargs : dict
        the kwargs from the compute function
    derivative : int
        derivative order used for interpolation

    Returns
    -------
    coords : ndarray
        Interpolated XYZ coords with shape (3, len(s_query_pts))
    """
    f = jnp.asarray(f)
    intervals = jnp.asarray(transforms["intervals"])
    has_break_points = len(intervals[0])
    s_query_pts += transforms["knots"][0]

    def inner_body(f, knots, period=None):
        """Interpolation for spline curves."""
        fq = interp1d(
            s_query_pts,
            knots,
            f.T,
            method=method,
            derivative=derivative,
            period=period,
        )

        return fq.T

    def body(interval, full_f, full_knots, min_interval_idx):
        """Body used if there are break points."""
        istart, istop = interval
        # catch end-point
        istop = jnp.where(istop == min_interval_idx, -1, istop)

        # fill f values outside of interval with break point values so that
        # interpolation only takes into consideration the interval
        f_in_interval = jnp.where(
            full_knots > full_knots[istop], full_f[:, istop][..., None], full_f
        )
        f_in_interval = jnp.where(
            full_knots < full_knots[istart], full_f[:, istart][..., None], f_in_interval
        )
        f_interp = inner_body(f_in_interval, full_knots, period=None)

        # replace values outside of interval with 0 so they don't contribute to the sum
        f_interp = jnp.where(s_query_pts > full_knots[istop], 0, f_interp)
        f_interp = jnp.where(s_query_pts < full_knots[istart], 0, f_interp)
        # covers edge case where the knot is exactly equal to a query point
        # and halves that point to sum to one later.
        f_interp = jnp.where(
            (s_query_pts == full_knots[istop]) & (s_query_pts != full_knots[-1]),
            f_interp / 2,
            f_interp,
        )
        f_interp = jnp.where(
            (s_query_pts == full_knots[istart]) & (s_query_pts != full_knots[0]),
            f_interp / 2,
            f_interp,
        )

        return f_interp

    if has_break_points:
        min_interval_idx = intervals[0][1]
        # manually add endpoint for broken splines so that it is closed
        full_knots = jnp.append(
            transforms["knots"], transforms["knots"][0] + 2 * jnp.pi
        )
        full_f = jnp.append(f, f[:, 0][..., None], axis=1)
        f_interp = vmap(
            lambda interval: body(interval, full_f, full_knots, min_interval_idx)
        )
        f_interp = f_interp(intervals).sum(axis=0)
    else:
        # regular interpolation where the period for interp is 2pi
        f_interp = inner_body(f, transforms["knots"], period=2 * jnp.pi)

    coords = jnp.stack(f_interp, axis=1)

    return coords


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve",
    dim=3,
    params=["X", "Y", "Z", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
)
def _center_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    # center is average of xyz knots
    xyz = jnp.stack([params["X"], params["Y"], params["Z"]], axis=1)
    center = jnp.mean(xyz, axis=0)
    # displacement and rotation
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    # convert to rpz
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["X", "Y", "Z", "rotmat", "shift"],
    transforms={"intervals": [], "knots": []},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_SplineXYZCurve(params, transforms, profiles, data, **kwargs):

    derivative = 0
    xq = data["s"]
    f = [params["X"], params["Y"], params["Z"]]

    coords = _splinexyz_helper(f, transforms, xq, kwargs["method"], derivative)

    coords = (
        coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][jnp.newaxis, :]
    )
    coords = xyz2rpz(coords)
    data["x"] = coords
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"intervals": [], "knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_s_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 1
    xq = data["s"]
    f = [params["X"], params["Y"], params["Z"]]

    coords_s = _splinexyz_helper(f, transforms, xq, kwargs["method"], derivative)
    coords_s = coords_s @ params["rotmat"].reshape((3, 3)).T

    coords_s = xyz2rpz_vec(coords_s, phi=data["phi"])

    data["x_s"] = coords_s
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"intervals": [], "knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_ss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 2
    xq = data["s"]
    f = [params["X"], params["Y"], params["Z"]]

    coords_ss = _splinexyz_helper(f, transforms, xq, kwargs["method"], derivative)
    coords_ss = coords_ss @ params["rotmat"].reshape((3, 3)).T

    coords_ss = xyz2rpz_vec(coords_ss, phi=data["phi"])
    data["x_ss"] = coords_ss
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["X", "Y", "Z", "rotmat"],
    transforms={"intervals": [], "knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _x_sss_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    derivative = 3
    xq = data["s"]
    f = [params["X"], params["Y"], params["Z"]]

    coords_sss = _splinexyz_helper(f, transforms, xq, kwargs["method"], derivative)
    coords_sss = coords_sss @ params["rotmat"].reshape((3, 3)).T

    coords_sss = xyz2rpz_vec(coords_sss, phi=data["phi"])
    data["x_sss"] = coords_sss

    return data


@register_compute_fun(
    name="frenet_tangent",
    label="\\mathbf{T}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Tangent unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_tangent(params, transforms, profiles, data, **kwargs):
    data["frenet_tangent"] = (
        data["x_s"] / jnp.linalg.norm(data["x_s"], axis=-1)[:, None]
    )
    return data


@register_compute_fun(
    name="frenet_normal",
    label="\\mathbf{N}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Normal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_normal(params, transforms, profiles, data, **kwargs):
    normal = cross(data["x_s"], cross(data["x_ss"], data["x_s"]))
    data["frenet_normal"] = normal / jnp.linalg.norm(normal, axis=-1)[:, None]
    return data


@register_compute_fun(
    name="frenet_binormal",
    label="\\mathbf{B}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Binormal unit vector to curve in Frenet-Serret frame",
    dim=3,
    params=["rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["frenet_tangent", "frenet_normal"],
    parameterization="desc.geometry.core.Curve",
)
def _frenet_binormal(params, transforms, profiles, data, **kwargs):
    data["frenet_binormal"] = cross(
        data["frenet_tangent"], data["frenet_normal"]
    ) * jnp.linalg.det(params["rotmat"].reshape((3, 3)))
    return data


@register_compute_fun(
    name="curvature",
    label="\\kappa",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar curvature of the curve, with the sign denoting the convexity/"
    + "concavity relative to the center of the curve (a circle has positive curvature)",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["center", "x", "x_s", "x_ss", "frenet_normal", "phi"],
    parameterization="desc.geometry.core.Curve",
)
def _curvature(params, transforms, profiles, data, **kwargs):
    # magnitude of curvature
    dxn = jnp.linalg.norm(data["x_s"], axis=-1)[:, jnp.newaxis]
    curvature = jnp.linalg.norm(cross(data["x_s"], data["x_ss"]) / dxn**3, axis=-1)
    # sign of curvature (positive = "convex", negative = "concave")
    r = rpz2xyz(data["center"]) - rpz2xyz(data["x"])
    r = xyz2rpz_vec(r, phi=data["phi"])
    data["curvature"] = curvature * sign(dot(r, data["frenet_normal"]))
    return data


@register_compute_fun(
    name="torsion",
    label="\\tau",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar torsion of the curve",
    dim=1,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss", "x_sss"],
    parameterization="desc.geometry.core.Curve",
)
def _torsion(params, transforms, profiles, data, **kwargs):
    dxd2x = cross(data["x_s"], data["x_ss"])
    data["torsion"] = dot(dxd2x, data["x_sss"]) / jnp.linalg.norm(dxd2x, axis=-1) ** 2
    return data


@register_compute_fun(
    name="torsion",
    label="\\tau",
    units="m^{-1}",
    units_long="Inverse meters",
    description="Scalar torsion of the curve",
    dim=1,
    params=[],
    transforms={"intervals": [], "knots": []},
    profiles=[],
    coordinates="s",
    data=["s", "x_s", "x_ss", "x_sss"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _torsion_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    dxd2x = cross(data["x_s"], data["x_ss"])
    data["torsion"] = dot(dxd2x, data["x_sss"]) / jnp.linalg.norm(dxd2x, axis=-1) ** 2
    # set torsion to zero at break points because the curve is just
    # 2 lines that lie in the same plane
    if len(transforms["intervals"][0]):
        is_break_point = (
            data["s"] == transforms["knots"][transforms["intervals"]][:, 1:]
        )
        data["torsion"] = jnp.where(
            is_break_point.any(axis=0),
            0.0,
            data["torsion"],
        )

    return data


@register_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["ds", "x_s"],
    parameterization=["desc.geometry.core.Curve"],
)
def _length(params, transforms, profiles, data, **kwargs):
    T = jnp.linalg.norm(data["x_s"], axis=-1)
    # this is equivalent to jnp.trapz(T, s) for a closed curve,
    # but also works if grid.endpoint is False
    data["length"] = jnp.sum(T * data["ds"])
    return data


@register_compute_fun(
    name="length",
    label="L",
    units="m",
    units_long="meters",
    description="Length of the curve",
    dim=0,
    params=[],
    transforms={},
    profiles=[],
    coordinates="",
    data=["ds", "x", "x_s"],
    parameterization="desc.geometry.curve.SplineXYZCurve",
    method="Interpolation type, Default 'cubic'. See SplineXYZCurve docs for options.",
)
def _length_SplineXYZCurve(params, transforms, profiles, data, **kwargs):
    if kwargs["method"] == "nearest":  # cannot use derivative method as deriv=0
        coords = data["x"]
        if kwargs.get("basis", "rpz").lower() == "rpz":
            coords = rpz2xyz(coords)
        # ensure curve is closed
        # if it's already closed this doesn't add any length since ds will be zero
        coords = jnp.concatenate([coords, coords[:1]])
        X = coords[:, 0]
        Y = coords[:, 1]
        Z = coords[:, 2]
        lengths = jnp.sqrt(jnp.diff(X) ** 2 + jnp.diff(Y) ** 2 + jnp.diff(Z) ** 2)
        data["length"] = jnp.sum(lengths)
    else:
        T = jnp.linalg.norm(data["x_s"], axis=-1)
        # this is equivalent to jnp.trapz(T, s) for a closed curve
        # but also works if grid.endpoint is False
        data["length"] = jnp.sum(T * data["ds"])
    return data


# ---------------------------------------------------------------------------
# PiecewisePlanarArcCurve: B planar arcs joined at shared hinges (C0 corners).
# Parameters: hinges (B,3), tilts (B,), shape (B,M). Curve param s in [0,2pi)
# maps to arc i = floor(s*B/2pi) and local t = frac(s*B/2pi) in [0,1].
# Geometry per arc (xyz):
#   chord = H[i+1]-H[i];  e_par = chord/|chord|
#   perp0 = normalize(ref - (ref.e_par) e_par)   [ref chosen not || chord]
#   perp  = perp0 cos(phi) + (e_par x perp0) sin(phi)    [Rodrigues about e_par]
#   w(t)  = sum_m a[i,m] sin((m+1) pi t)
#   x(t)  = H[i] + t chord + w perp
# ds->dt scale = B/(2pi):  x_s = x_t * (B/2pi);  x_ss = x_tt * (B/2pi)^2
# ---------------------------------------------------------------------------


def _ppa_arc_frame(hinges, tilts, B):
    """Per-arc chord, e_par, perp (unit in-plane normal), for all B arcs.

    Returns chord (B,3), e_par (B,3), perp (B,3).
    """
    Hi = hinges
    Hnext = jnp.roll(hinges, -1, axis=0)
    chord = Hnext - Hi
    L = jnp.linalg.norm(chord, axis=1, keepdims=True)
    e_par = chord / L
    # reference vector not parallel to chord: prefer +Z, fall back to +Y if aligned
    zc = jnp.abs(e_par[:, 2])  # |e_par . zhat|
    ref = jnp.where(
        (zc > 0.9)[:, None],
        jnp.array([0.0, 1.0, 0.0])[None, :],
        jnp.array([0.0, 0.0, 1.0])[None, :],
    )
    perp0 = ref - jnp.sum(ref * e_par, axis=1, keepdims=True) * e_par
    perp0 = perp0 / jnp.linalg.norm(perp0, axis=1, keepdims=True)
    # Rodrigues rotation of perp0 about e_par by tilt (perp0 _|_ e_par so no 3rd term)
    cphi = jnp.cos(tilts)[:, None]
    sphi = jnp.sin(tilts)[:, None]
    perp = perp0 * cphi + jnp.cross(e_par, perp0) * sphi
    return chord, e_par, perp


def _ppa_indices(s, B):
    """Map curve param s in [0,2pi) to arc index i and local t in [0,1]."""
    u = s * B / (2 * jnp.pi)
    i = jnp.floor(u).astype(int)
    i = jnp.clip(i, 0, B - 1)
    t = u - i
    return i, t


def _ppa_coords(params, data, B, M, deriv):
    """Compute xyz coords (deriv=0) or d^deriv x / dt^deriv (deriv=1,2)."""
    hinges = params["hinges"].reshape(B, 3)
    tilts = params["tilts"].reshape(B)
    shape = params["shape"].reshape(B, M)
    chord, e_par, perp = _ppa_arc_frame(hinges, tilts, B)

    i, t = _ppa_indices(data["s"], B)
    Hi = hinges[i]  # (nt,3)
    ci = chord[i]  # (nt,3)
    pi_ = perp[i]  # (nt,3)
    ai = shape[i]  # (nt,M)

    m = jnp.arange(1, M + 1)  # (M,) sine mode numbers
    ang = jnp.pi * jnp.outer(t, m)  # (nt,M)
    if deriv == 0:
        w = jnp.sum(ai * jnp.sin(ang), axis=1)  # (nt,)
        coords = Hi + t[:, None] * ci + w[:, None] * pi_
    elif deriv == 1:
        dw = jnp.sum(ai * (jnp.pi * m) * jnp.cos(ang), axis=1)
        coords = ci + dw[:, None] * pi_
    elif deriv == 2:
        d2w = jnp.sum(ai * (-((jnp.pi * m) ** 2)) * jnp.sin(ang), axis=1)
        coords = d2w[:, None] * pi_
    elif deriv == 3:
        d3w = jnp.sum(ai * (-((jnp.pi * m) ** 3)) * jnp.cos(ang), axis=1)
        coords = d3w[:, None] * pi_
    else:
        raise ValueError(f"deriv must be 0,1,2,3 got {deriv}")
    return coords


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of transverse sine modes per arc",
)
def _x_PiecewisePlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    M = kwargs["arc_M"]
    coords = _ppa_coords(params, data, B, M, deriv=0)
    coords = coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][None, :]
    data["x"] = xyz2rpz(coords)
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of transverse sine modes per arc",
)
def _x_s_PiecewisePlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    M = kwargs["arc_M"]
    scale = B / (2 * jnp.pi)
    coords = _ppa_coords(params, data, B, M, deriv=1) * scale
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_s"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of transverse sine modes per arc",
)
def _x_ss_PiecewisePlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    M = kwargs["arc_M"]
    scale = (B / (2 * jnp.pi)) ** 2
    coords = _ppa_coords(params, data, B, M, deriv=2) * scale
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_ss"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="center",
    label="\\langle\\mathbf{x}\\rangle",
    units="m",
    units_long="meters",
    description="Centroid of the curve (mean of hinges)",
    dim=3,
    params=["hinges", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
    arc_B="int: number of planar arcs",
)
def _center_PiecewisePlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    hinges = params["hinges"].reshape(B, 3)
    center = jnp.mean(hinges, axis=0)
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    data["center"] = xyz2rpz(center) * jnp.ones_like(data["x"])
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of transverse sine modes per arc",
)
def _x_sss_PiecewisePlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    M = kwargs["arc_M"]
    scale = (B / (2 * jnp.pi)) ** 3
    coords = _ppa_coords(params, data, B, M, deriv=3) * scale
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_sss"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="frenet_normal",
    label="\\mathbf{N}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Normal unit vector to curve in Frenet-Serret frame "
    "(safenormalized so the C0 arc-start inflections, where x_ss=0, give a finite "
    "zero vector instead of a 0/0 nan)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss"],
    parameterization="desc.geometry.curve.PiecewisePlanarArcCurve",
)
def _frenet_normal_PiecewisePlanarArcCurve(
    params, transforms, profiles, data, **kwargs
):
    normal = cross(data["x_s"], cross(data["x_ss"], data["x_s"]))
    data["frenet_normal"] = safenormalize(normal, axis=-1)
    return data


# ---------------------------------------------------------------------------
# PolarPlanarArcCurve: B planar arcs, each a POLAR graph r(theta) about its
# chord MIDPOINT. Parameters: hinges (B,3), tilts (B,), shape (B,M).
# Curve param s in [0,2pi) -> arc i = floor(s B/2pi), local phi = frac in [0,1].
# Geometry per arc (xyz), with C = (H[i]+H[i+1])/2, Lc = |chord|:
#   r(phi) = Lc/2 + sum_m a[i,m] sin((m+1) pi phi)
#   theta  = pi phi
#   x      = C + r cos(theta) e_par + r sin(theta) perp
# Both hinges lie ON the polar axis (theta = 0 and pi) because the pole is the
# chord midpoint, so r(0) = r(1) = Lc/2 holds for ANY coefficients: C0 closure is
# structural and a SINGLE series pins both endpoints.
# Derivatives w.r.t. phi, with c = cos(pi phi), sn = sin(pi phi), p = pi, and
# rk = d^k r / dphi^k:
#   dx  = r1 c - p r0 sn                      dy  = r1 sn + p r0 c
#   d2x = r2 c - 2p r1 sn - p^2 r0 c          d2y = r2 sn + 2p r1 c - p^2 r0 sn
#   d3x = r3 c - 3p r2 sn - 3p^2 r1 c + p^3 r0 sn
#   d3y = r3 sn + 3p r2 c - 3p^2 r1 sn - p^3 r0 c
# All six verified symbolically against sympy. ds -> dphi scale = B/(2pi), so
# x_s = dx (B/2pi), x_ss = d2x (B/2pi)^2, x_sss = d3x (B/2pi)^3.
# ---------------------------------------------------------------------------


def _ppolar_arc_frame(hinges, tilts, B):
    """Per-arc chord, e_par, perp, pole (chord midpoint) and chord length."""
    Hi = hinges
    Hnext = jnp.roll(hinges, -1, axis=0)
    chord = Hnext - Hi
    L = jnp.linalg.norm(chord, axis=1, keepdims=True)
    e_par = chord / L
    zc = jnp.abs(e_par[:, 2])
    ref = jnp.where(
        (zc > 0.9)[:, None],
        jnp.array([0.0, 1.0, 0.0])[None, :],
        jnp.array([0.0, 0.0, 1.0])[None, :],
    )
    perp0 = ref - jnp.sum(ref * e_par, axis=1, keepdims=True) * e_par
    perp0 = perp0 / jnp.linalg.norm(perp0, axis=1, keepdims=True)
    cphi = jnp.cos(tilts)[:, None]
    sphi = jnp.sin(tilts)[:, None]
    perp = perp0 * cphi + jnp.cross(e_par, perp0) * sphi
    pole = 0.5 * (Hi + Hnext)
    return chord, e_par, perp, pole, L[:, 0]


def _ppolar_indices(s, B):
    """Map curve param s in [0,2pi) to arc index i and local phi in [0,1]."""
    u = s * B / (2 * jnp.pi)
    i = jnp.clip(jnp.floor(u).astype(int), 0, B - 1)
    return i, u - i


def _ppolar_coords(params, data, B, M, deriv):
    """xyz coords (deriv=0) or the deriv-th phi derivative (deriv=1,2,3)."""
    hinges = params["hinges"].reshape(B, 3)
    tilts = params["tilts"].reshape(B)
    shape = params["shape"].reshape(B, M)
    chord, e_par, perp, pole, Lc = _ppolar_arc_frame(hinges, tilts, B)

    i, phi = _ppolar_indices(data["s"], B)
    ai = shape[i]
    ei = e_par[i]
    pi_ = perp[i]
    Ci = pole[i]
    Li = Lc[i]

    m = jnp.arange(1, M + 1)
    km = jnp.pi * m
    ang = jnp.pi * jnp.outer(phi, m)
    p = jnp.pi
    c = jnp.cos(p * phi)
    sn = jnp.sin(p * phi)

    # NOTE the MINUS on the e_par component. theta is measured from the -e_par end so
    # that phi=0 -> C - (Lc/2) e_par = H[i] and phi=1 -> C + (Lc/2) e_par = H[i+1],
    # i.e. arc i traverses H[i] -> H[i+1] as the CoilSet/arc-index convention requires.
    # Taking theta from +e_par instead puts phi=0 on H[i+1] and runs every arc backwards
    # (endpoint POSITIONS still look right, so this is only caught by checking which
    # hinge phi=0 lands on).
    r0 = Li / 2 + jnp.sum(ai * jnp.sin(ang), axis=1)
    if deriv == 0:
        xl, yl = -r0 * c, r0 * sn
        return Ci + xl[:, None] * ei + yl[:, None] * pi_
    r1 = jnp.sum(ai * km * jnp.cos(ang), axis=1)
    if deriv == 1:
        xl = -(r1 * c - p * r0 * sn)
        yl = r1 * sn + p * r0 * c
        return xl[:, None] * ei + yl[:, None] * pi_
    r2 = -jnp.sum(ai * km**2 * jnp.sin(ang), axis=1)
    if deriv == 2:
        xl = -(r2 * c - 2 * p * r1 * sn - p**2 * r0 * c)
        yl = r2 * sn + 2 * p * r1 * c - p**2 * r0 * sn
        return xl[:, None] * ei + yl[:, None] * pi_
    r3 = -jnp.sum(ai * km**3 * jnp.cos(ang), axis=1)
    if deriv == 3:
        xl = -(r3 * c - 3 * p * r2 * sn - 3 * p**2 * r1 * c + p**3 * r0 * sn)
        yl = r3 * sn + 3 * p * r2 * c - 3 * p**2 * r1 * sn - p**3 * r0 * c
        return xl[:, None] * ei + yl[:, None] * pi_
    raise ValueError(f"deriv must be 0,1,2,3 got {deriv}")


@register_compute_fun(
    name="x",
    label="\\mathbf{x}",
    units="~",
    units_long="not applicable",
    description="Coordinate triplet. "
    "This is not a position vector unless basis is cartesian. "
    "When basis is cartesian, the units are meters.",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s"],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of polar radial sine modes per arc",
)
def _x_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    coords = _ppolar_coords(params, data, kwargs["arc_B"], kwargs["arc_M"], deriv=0)
    coords = coords @ params["rotmat"].reshape((3, 3)).T + params["shift"][None, :]
    data["x"] = xyz2rpz(coords)
    return data


@register_compute_fun(
    name="x_s",
    label="\\partial_{s} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, first derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of polar radial sine modes per arc",
)
def _x_s_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    coords = _ppolar_coords(params, data, B, kwargs["arc_M"], deriv=1)
    coords = coords * (B / (2 * jnp.pi))
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_s"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="x_ss",
    label="\\partial_{ss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, second derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of polar radial sine modes per arc",
)
def _x_ss_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    coords = _ppolar_coords(params, data, B, kwargs["arc_M"], deriv=2)
    coords = coords * (B / (2 * jnp.pi)) ** 2
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_ss"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="x_sss",
    label="\\partial_{sss} \\mathbf{x}",
    units="m",
    units_long="meters",
    description="Position vector along curve, third derivative",
    dim=3,
    params=["hinges", "tilts", "shape", "rotmat"],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["s", "phi"],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
    arc_B="int: number of planar arcs",
    arc_M="int: number of polar radial sine modes per arc",
)
def _x_sss_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    coords = _ppolar_coords(params, data, B, kwargs["arc_M"], deriv=3)
    coords = coords * (B / (2 * jnp.pi)) ** 3
    coords = coords @ params["rotmat"].reshape((3, 3)).T
    data["x_sss"] = xyz2rpz_vec(coords, phi=data["phi"])
    return data


@register_compute_fun(
    name="center",
    label="\\mathbf{x}_{0}",
    units="m",
    units_long="meters",
    description="Centroid of the hinge points",
    dim=3,
    params=["hinges", "rotmat", "shift"],
    transforms={},
    profiles=[],
    coordinates="",
    data=[],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
    arc_B="int: number of planar arcs",
)
def _center_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    B = kwargs["arc_B"]
    center = jnp.mean(params["hinges"].reshape(B, 3), axis=0)
    center = jnp.matmul(center, params["rotmat"].reshape((3, 3)).T) + params["shift"]
    data["center"] = center
    return data


@register_compute_fun(
    name="frenet_normal",
    label="\\mathbf{N}_{\\mathrm{Frenet-Serret}}",
    units="~",
    units_long="None",
    description="Normal unit vector to curve in Frenet-Serret frame "
    "(safenormalized so any point where x_ss vanishes gives a finite zero vector "
    "instead of a 0/0 nan)",
    dim=3,
    params=[],
    transforms={},
    profiles=[],
    coordinates="s",
    data=["x_s", "x_ss"],
    parameterization="desc.geometry.curve.PolarPlanarArcCurve",
)
def _frenet_normal_PolarPlanarArcCurve(params, transforms, profiles, data, **kwargs):
    normal = cross(data["x_s"], cross(data["x_ss"], data["x_s"]))
    data["frenet_normal"] = safenormalize(normal, axis=-1)
    return data
