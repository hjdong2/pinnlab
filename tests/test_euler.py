
import numpy as np
import pytest
import torch

from pinnlab.problems import euler


SOD_L = (1.0, 0.0, 1.0)
SOD_R = (0.125, 0.0, 0.1)

def test_sod_str_state():
    """Star states of Sod's problem matches Toro."""
    u_str, p_str, rho_str_l, rho_str_r = euler.str_state(SOD_L, SOD_R)
    assert u_str == pytest.approx(0.92745, abs=1e-5)
    assert p_str == pytest.approx(0.30313, abs=1e-5)
    assert rho_str_l == pytest.approx(0.42632, abs=1e-5)
    assert rho_str_r == pytest.approx(0.26557, abs=1e-5)


def test_wave_speeds():
    """Sod wave speeds: values, and ordering"""
    s = np.array(euler.wave_speeds(SOD_L, SOD_R))
    ref = (-1.18322, -0.07027, 0.92745, 1.75216)
    np.testing.assert_allclose(s, ref, atol=1e-5)
    assert np.all(np.diff(s) > 0), f"wave speeds not ordered: {s}"


def test_t0_is_step():
    """make sure the soln is the initial step @ t=0."""
    x  = np.linspace(0, 1, 101)
    rho, u, p = euler.euler_riemann_exact(x, [0.0], SOD_L, SOD_R)
    for q, ql, qr in zip((rho, u, p), SOD_L, SOD_R):
        qk = np.where(x < 0.5, ql, qr)
        np.testing.assert_allclose(q[0], qk, rtol=0.0)


def test_continuity_across_waves():
    u_str, p_str, rho_str_l, rho_str_r = euler.str_state(SOD_L, SOD_R)
    t, eps = 0.2, 1e-7
    s = np.array(euler.wave_speeds(SOD_L, SOD_R))
    xb = 0.5 + 0.2 * s
    x = np.sort(np.concatenate([xb - eps, xb + eps]))
    rho, u, p = euler.euler_riemann_exact(x, [t], SOD_L, SOD_R)
    d_rho = rho[0, 1::2] - rho[0, 0::2]
    d_u = u[0, 1::2] - u[0, 0::2]
    d_p = p[0, 1::2] - p[0, 0::2]
    assert np.abs(d_rho[0]) < 1e-5
    assert np.abs(d_rho[1]) < 1e-5
    for i in [0, 1, 2]:
        assert np.abs(d_u[i]) < 1e-5
        assert np.abs(d_p[i]) < 1e-5

    assert d_rho[2] == pytest.approx(rho_str_r - rho_str_l, abs=1e-6)
    assert d_rho[3] == pytest.approx(SOD_R[0] - rho_str_r, abs=1e-6)
    assert d_u[3] == pytest.approx(SOD_R[1] - u_str, abs=1e-6)
    assert d_p[3] == pytest.approx(SOD_R[2] - p_str, abs=1e-6)


def test_rankine_hugoniot():
    u_str, p_str, _, rho_str_r = euler.str_state(SOD_L, SOD_R)
    _, _, _, S_R = euler.wave_speeds(SOD_L, SOD_R)
    W_str_r = (rho_str_r, u_str, p_str)
    lhs = euler.invFlux(SOD_R) - euler.invFlux(W_str_r)
    rhs = S_R * (euler.prim2cons(SOD_R) - euler.prim2cons(W_str_r))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_lax_entropy():
    u_str, p_str, _, rho_str_r = euler.str_state(SOD_L, SOD_R)
    _, _, _, S_R = euler.wave_speeds(SOD_L, SOD_R)
    c_str_r = euler.sound_speed(rho_str_r, p_str)
    c_r = euler.sound_speed(SOD_R[0], SOD_R[2])
    lam_str = u_str + c_str_r
    lam_rht = SOD_R[1] + c_r
    assert lam_str > S_R > lam_rht


def mms_fields(x, t, xp):
    rho = 1.0 + 0.2 * xp.sin(2 * np.pi * x - t)
    u = 0.5 + 0.1 * xp.cos(2 * np.pi * x + t)
    p = 1.0 + 0.1 * xp.sin(2 * np.pi * x) * xp.exp(-t)
    return rho, u, p


def test_res_match_fd():
    """audograd residual == central finite difference"""
    gamma, h = 1.3, 1e-4
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 20)
    t = rng.uniform(0, 1, 20)

    xt = torch.tensor(np.column_stack([x, t]), dtype=torch.float64)
    fake_model = lambda xt: mms_fields(xt[:, 0:1], xt[:, 1:2], torch)
    res = euler.residual(fake_model, xt, gamma)

    U = lambda x, t: euler.prim2cons(mms_fields(x, t, np), gamma)
    F = lambda x, t: euler.invFlux(mms_fields(x, t, np), gamma)
    res_fd = (U(x, t + h) - U(x, t - h)) / (2 * h) + (F(x + h, t) - F(x - h, t)) / (2 * h)

    for k in range(3):
        np.testing.assert_allclose(res[k].detach().numpy().ravel(), res_fd[k], atol=1e-6)