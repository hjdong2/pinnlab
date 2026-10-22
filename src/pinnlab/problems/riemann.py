
import numpy as np
import torch

from .burgers import LB, UB, _uniform, residual


def riemann_ext(x, t, ul, ur):
    """Exact solution of inviscid Burgers' eqn
       shock if ul > ur, rarefaction if ul < ur
    """
    ul, ur = float(ul), float(ur)

    x = np.asarray(x, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    X, T = x[None, :], t[:, None]

    if ul > ur:
        s = (ul + ur) / 2
        u = np.where(X < s * T, ul, ur)
    else:
        T = np.where(T > 0, T, 1.0)
        u = np.clip(X / T, ul, ur)

    u[t == 0] = np.where(X < 0, ul, ur)
    return u


def viscous_shk(x, t, ul, ur, nu):
    ul, ur = float(ul), float(ur)
    if ul <= ur:
        raise ValueError("a travelling shock exists only for ul > ur")
    x = np.asarray(x, dtype=np.float64)
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    X, T = x[None, :], t[:, None]

    s = (ul + ur) / 2
    z = (ul - ur) * (X - s * T) / (4 * nu)
    return (ul + ur) / 2 - (ul - ur) / 2 * np.tanh(z)


def make_riemann_loss(model, ul, ur, nu, nf, n0, nb, weights, seed=0, smooth_ic=False):
    gen = torch.Generator().manual_seed(seed)

    xt_f = _uniform(nf, LB, UB, gen)
    x0 = _uniform(n0, (LB[0],), (UB[0],), gen)
    t0 = torch.zeros_like(x0)
    xt_0 = torch.cat([x0, t0], dim=1)
    if smooth_ic:
        u0 = (ul + ur) / 2 - (ul - ur) / 2 * torch.tanh(abs((ul - ur)) * x0 / (4 * nu))
    else:
        u0 = (torch.where(x0 < 0, ul, ur))

    tb = _uniform(nb, (LB[1],), (UB[1],), gen)
    xt_lft = torch.cat([-torch.ones_like(tb), tb], dim=1)
    xt_rht = torch.cat([torch.ones_like(tb), tb], dim=1)
    xt_b = torch.cat([xt_lft, xt_rht], dim=0)
    ub = torch.cat([torch.full_like(tb, ul), torch.full_like(tb, ur)], dim=0)

    def loss_fcn():
        l_pde = (residual(model, xt_f, nu)).pow(2).mean()
        l_ic = (model(xt_0) - u0).pow(2).mean()
        l_bc = (model(xt_b) - ub).pow(2).mean()

        l_tot = weights["pde"] * l_pde + weights["ic"] * l_ic + weights["bc"] * l_bc

        return l_tot, {"pde": l_pde.item(), "ic": l_ic.item(), "bc": l_bc.item()}

    return loss_fcn


def crossing(x, u, level):
    below = u < level
    if not below.any():
        raise ValueError(f"profile never drops below {level}")
    j = int(np.argmax(below))
    if j == 0:
        raise ValueError("no segment to interpolate")
    xstar = x[j - 1] + (level - u[j - 1]) * (x[j] - x[j - 1]) / (u[j] - u[j - 1])
    return xstar


def shock_pos(x, u, ul, ur):
    xstar = crossing(x, u, (ul + ur) / 2)
    return xstar


def shock_thickness(x, u, ul, ur):
    d = ul - ur
    xp9 = crossing(x, u, ur + 0.9 * d)
    xp1 = crossing(x, u, ur + 0.1 * d)
    return xp1 - xp9
