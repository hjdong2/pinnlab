
import numpy as np
import torch

from ..utils import grad
from .burgers import _uniform

GAMMA = 1.4

def prim2cons(W, gamma=GAMMA):
    rho, u, p = W
    rhou = rho * u
    rhoE = p / (gamma - 1) + 0.5 * rho * u**2
    return np.array([rho, rhou, rhoE])


def invFlux(W, gamma=GAMMA):
    rho, u, p = W
    _, rhou, rhoE = prim2cons(W, gamma)
    return np.array([rhou, rho * u**2 + p, u * (rhoE + p)])


def sound_speed(rho, p, gamma=GAMMA):
    return np.sqrt(gamma * p / rho)


def pressure_fcn(p, rho_k, p_k, gamma=GAMMA):
    c_k = sound_speed(rho_k, p_k, gamma)
    A_k = 2 / ((gamma + 1) * rho_k)
    B_k = (gamma - 1) / (gamma + 1) * p_k
    if p > p_k:
        f = (p - p_k) * np.sqrt(A_k / (p + B_k))
        df = np.sqrt(A_k / (p + B_k)) * (1 - (p - p_k) / (2 * (B_k + p)))
    else:
        f = (2 * c_k / (gamma - 1)) * ((p / p_k)**((gamma - 1) / (2 * gamma)) - 1)
        df = (1 / (rho_k * c_k)) * (p / p_k)**(-(gamma + 1) / (2 * gamma))

    return f, df


def str_pressure(W_l, W_r, gamma=GAMMA, tol=1e-12, max_it=50):
    rho_l, u_l, p_l = W_l
    rho_r, u_r, p_r = W_r
    c_l = sound_speed(rho_l, p_l, gamma)
    c_r = sound_speed(rho_r, p_r, gamma)
    coeff = (2 * gamma) / (gamma - 1)
    p_TR = ((c_l + c_r - 0.5 * (gamma - 1) * (u_r - u_l)) / (c_l / p_l**(1 / coeff) + c_r / p_r**(1 / coeff)))**(coeff)
    p0 = p_TR
    p = max(tol, p0)
    for it in range(max_it):
        f_l, df_l = pressure_fcn(p, rho_l, p_l, gamma)
        f_r, df_r = pressure_fcn(p, rho_r, p_r, gamma)
        g = f_l + f_r + (u_r - u_l)
        dg = df_l + df_r
        p_new = p - g / dg
        
        if p_new < 0.0:
            p_new = tol

        if (np.abs(p_new - p) / (0.5 * (p_new + p)) < tol):
            return p_new
        p = p_new
    raise RuntimeError("Did not get converged after maximum iterations!")


def str_state(W_l, W_r, gamma=GAMMA):
    rho_l, u_l, p_l = W_l
    rho_r, u_r, p_r = W_r
    p_str = str_pressure(W_l, W_r, gamma)
    f_l, _ = pressure_fcn(p_str, rho_l, p_l, gamma)
    f_r, _ = pressure_fcn(p_str, rho_r, p_r, gamma)
    u_str = (u_l + u_r) * 0.5 + (f_r - f_l) * 0.5
    coeff = (gamma - 1) / (gamma + 1)

    def rho_str_K(p_str, rho_k, p_k):
        if p_str > p_k:
            return rho_k * (p_str / p_k + coeff) / (coeff * p_str / p_k + 1)
        return rho_k * (p_str / p_k)**(1 / gamma)
    
    rho_str_l = rho_str_K(p_str, rho_l, p_l)
    rho_str_r = rho_str_K(p_str, rho_r, p_r)

    return u_str, p_str, rho_str_l, rho_str_r


def wave_speeds(W_l, W_r, gamma=GAMMA):
    rho_l, u_l, p_l = W_l
    rho_r, u_r, p_r = W_r
    u_str, p_str, rho_str_l, rho_str_r = str_state(W_l, W_r, gamma)
    c_l = sound_speed(rho_l, p_l, gamma)
    c_r = sound_speed(rho_r, p_r, gamma)
    coeff = (gamma - 1) / (2 * gamma)
    c_str_l = c_l * (p_str / p_l)**coeff
    S_HL = u_l - c_l
    S_TL = u_str - c_str_l
    S_R = u_r + c_r * np.sqrt((gamma + 1) / (2 * gamma) * p_str / p_r + coeff)
    return S_HL, S_TL, u_str, S_R


def euler_riemann_exact(x, t, W_l, W_r, x0=0.5, gamma=GAMMA):
    rho_l, u_l, p_l = W_l
    rho_r, u_r, p_r = W_r
    u_str, p_str, rho_str_l, rho_str_r = str_state(W_l, W_r, gamma)
    S_HL, S_TL, _, S_R = wave_speeds(W_l, W_r, gamma)

    x = np.asarray(x, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    X, T = np.meshgrid(x, t)
    T = np.where(T > 0, T, 1.0)

    xi = (X - x0) / T
    rho = np.full_like(xi, np.nan)
    u = np.full_like(xi, np.nan)
    p = np.full_like(xi, np.nan)

    m = xi < S_HL
    rho[m], u[m], p[m] = rho_l, u_l, p_l
    m = (xi < u_str) & (xi >= S_TL)
    rho[m], u[m], p[m] = rho_str_l, u_str, p_str
    m = (xi < S_R) & (xi >= u_str)
    rho[m], u[m], p[m] = rho_str_r, u_str, p_str
    m = xi >= S_R
    rho[m], u[m], p[m] = rho_r, u_r, p_r

    c_l = sound_speed(rho_l, p_l, gamma)
    m = (xi >= S_HL) & (xi < S_TL)
    xi_f = xi[m]
    cf = 2 / (gamma + 1) * (c_l + (gamma - 1) * 0.5 * (u_l - xi_f))
    uf = 2 / (gamma + 1) * (c_l + (gamma - 1) * 0.5 * u_l + xi_f)
    rhof = rho_l * (cf / c_l)**(2 / (gamma - 1))
    pf = p_l * (cf / c_l)**(2 * gamma / (gamma - 1))
    rho[m], u[m], p[m] = rhof, uf, pf

    rho[t == 0] = np.where(x < x0, rho_l, rho_r)
    u[t == 0] = np.where(x < x0, u_l, u_r)
    p[t == 0] = np.where(x < x0, p_l, p_r)

    return rho, u, p


def residual(model, xt, gamma=GAMMA):
    xt.requires_grad_(True)
    rho, u, p = model(xt)
    rhoE = p / (gamma - 1) + rho * u**2 * 0.5

    U = [rho, rho * u, rhoE]
    F = [rho * u, rho * u**2 + p, u * (rhoE + p)]
    res = []

    for Uk, Fk in zip(U, F):
        U_t = grad(Uk, xt)[:, 1:2]
        F_x = grad(Fk, xt)[:, 0:1]
        res.append(U_t + F_x)
    return res


def sample_clustered(n, frac, delta, x_c, t_max, gen):
    n_band = int(frac * n)
    t_b = _uniform(n_band, [0], [t_max], gen)
    s = _uniform(n_band, [-1.0], [1.0], gen)
    x_b = x_c(t_b) + delta * s
    band = torch.cat([x_b, t_b], dim=1)

    rest = _uniform(n - n_band, [0.0, 0.0], [1.0, t_max], gen)

    pts = torch.cat([band, rest], dim=0)
    pts[:, 0:1] = pts[:, 0:1].clamp(0.0, 1.0)
    return pts


def _mse3(pred, target):
    return sum((t - p).pow(2).mean() for t, p in zip(target, pred, strict=True))


def make_forward_loss_euler(model, W_l, W_r, x_c, nf, n0, nb, frac, delta, t_max,
                            x0=0.5, gamma=GAMMA, weights=None, seed=0,
                            frac_ic=0.0, delta_ic=0.05):
    gen = torch.Generator().manual_seed(seed)
    weights = weights or {"pde": 1.0, "ic": 1.0, "bc": 1.0}

    xt_f = sample_clustered(nf, frac, delta, x_c, t_max, gen)
    n_ic_band = int(frac_ic * n0)
    s_ic = _uniform(n_ic_band, [-1.0], [1.0], gen)
    x_ic_band = x0 + delta_ic * s_ic
    x_ic_rest = _uniform(n0 - n_ic_band, [0.0], [1.0], gen)
    x_ic = torch.cat([x_ic_band, x_ic_rest], dim=0)
    # x_ic = _uniform(n0, [0.0], [1.0], gen)
    t_ic = torch.zeros_like(x_ic)
    xt_ic = torch.cat([x_ic, t_ic], dim=1)
    W_ic = tuple(torch.where(x_ic < x0, ql, qr) for ql, qr in zip(W_l, W_r))

    t_b = _uniform(nb, [0.0], [t_max], gen)
    xt_lft = torch.cat([torch.zeros_like(t_b), t_b], dim=1)
    xt_rht = torch.cat([torch.ones_like(t_b), t_b], dim=1)
    xt_b = torch.cat([xt_lft, xt_rht], dim=0)
    W_b = tuple(torch.cat([torch.full_like(t_b, ql), torch.full_like(t_b, qr)], dim=0)
                for ql, qr in zip(W_l, W_r))

    def loss_fcn():
        l_pde = sum(r.pow(2).mean() for r in residual(model, xt_f, gamma))
        l_ic = _mse3(model(xt_ic), W_ic)
        l_bc = _mse3(model(xt_b), W_b)
        l_tot = weights["pde"] * l_pde + weights["ic"] * l_ic + weights["bc"] * l_bc

        return l_tot, {"pde": l_pde.item(), "ic": l_ic.item(), "bc": l_bc.item()}

    return loss_fcn

