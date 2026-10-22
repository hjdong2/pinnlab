
import numpy as np
import pytest

from pinnlab.problems import riemann


def test_riemann_t0_is_step():
    x = np.linspace(-1, 1, 401)

    for ul, ur in [(1.0, 0.0), (0.0, 1.0)]:
        u = riemann.riemann_ext(x, [0.0], ul, ur)
        u0 = u[0]

        exp = np.where(x < 0.0, ul, ur)
        np.testing.assert_allclose(u0, exp, rtol=1e-7)


def test_shock_speed():
    x = np.linspace(-1, 1, 401)
    dx = x[1] - x[0]
    ul, ur = 1.0, 0.0
    u_mid = (ul + ur) / 2
    s = (ul + ur) / 2
    ts = [0.2, 0.5, 1.0]

    u = riemann.riemann_ext(x, ts, ul, ur)
    for k, tk in enumerate(ts):
        j = np.argmax(u[k] < u_mid)
        x_jmp = x[j]

        assert abs(x_jmp - s * tk) <= dx, f"t = {tk}: jump at {x_jmp}, expected {s * tk}"


def test_rarefaction_fan():
    x = np.linspace(-1, 1, 401)
    dx = x[1] - x[0]
    ul, ur = 0.0, 1.0
    t = 0.5

    u = riemann.riemann_ext(x, [t], ul, ur)[0]

    inside = (x > ul * t) & (x < ur * t)
    np.testing.assert_allclose(u[inside], x[inside] / t)

    du = np.diff(u)
    assert np.all(du >= 0.0), "rarefaction must be non-decreasing"
    assert du.max() <= dx / t + 1e-12, f"jump of {du.max()} found (an expansion shock?)"


def test_boundary_values():
    """The waves never reach x = ±1 for t <= 1, so the boundary values stay u_l and u_r (the PINN's BC targets)."""
    x = np.linspace(-1, 1, 401)
    ts = np.linspace(0.0, 1.0, 11)

    for ul, ur in [(1.0, 0.0), (0.0, 1.0)]:
        u = riemann.riemann_ext(x, ts, ul, ur)
        np.testing.assert_allclose(u[:, 0], ul)
        np.testing.assert_allclose(u[:, -1], ur)


def test_vis_shk_satisfies_pde():
    nu, ul, ur = 1.0e-2, 1.0, 0.0
    t0, dt = 0.5, 1.0e-5
    x = np.linspace(-1, 1, 20001)
    dx = x[1] - x[0]

    um, u0, up = riemann.viscous_shk(x, [t0 - dt, t0, t0 + dt], ul, ur, nu)
    u_t = (up - um) / (2 * dt)
    u_x = (u0[2:] - u0[:-2]) / (2 * dx)
    u_xx = (u0[2:] - 2 * u0[1:-1] + u0[:-2]) / (dx * dx)
    res = u_t[1:-1] + u0[1:-1] * u_x - nu * u_xx
    assert np.abs(res).max() < 1.0e-3 * np.abs(u_t).max()


def test_vis_shk_l1_dis():
    ul, ur = 1.0, 0.0
    x = np.linspace(-1, 1, 200001)
    for nu in [1.0e-2, 3.0e-3, 1.0e-3]:
        u_vis = riemann.viscous_shk(x, [0.5], ul, ur, nu)[0]
        u_inv = riemann.riemann_ext(x, [0.5], ul, ur)[0]
        l1 = np.trapezoid(np.abs(u_vis - u_inv), x)
        assert l1 == pytest.approx(4 * np.log(2) * nu, rel=1e-6)
