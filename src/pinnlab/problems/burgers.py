
import numpy as np
import torch
from torch import nn

from ..utils import grad


class BurgersCoeffs(torch.nn.Module):
    """Unkown PDE coefficients for inverse problem. nu is stored as log(nu) to ensure positivity."""

    def __init__(self, lambda1_init, nu_init):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.tensor(float(lambda1_init)))
        self.log_nu = nn.Parameter(torch.tensor(float(np.log(nu_init))))

    @property
    def nu(self):
        return self.log_nu.exp()


NU = 0.01 / np.pi
LB = (-1.0, 0)
UB = (1.0, 1)

def _uniform(n, lb, ub, gen):
    """" uniformly distributed n points in domain [lb, ub], return shape (n, len(lb))"""
    lb = torch.tensor(lb)
    ub = torch.tensor(ub)
    r = torch.rand(n, len(lb), generator=gen)
    return lb + (ub - lb) * r


def residual(u_fcn, xt, nu, lambda1=1.0):
    """ define residual for burgers' eqn"""

    xt.requires_grad_(True)

    u = u_fcn(xt)
    du = grad(u, xt)

    u_x = du[:, 0:1]
    u_t = du[:, 1:2]

    d2u = grad(u_x, xt)

    u_xx = d2u[:, 0:1]

    return u_t + lambda1 * u * u_x - nu * u_xx


def reference_soln(x, t, nu=NU, n_q=4001):
    """ exact solution for burgers' eqn"""

    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    t = np.atleast_1d(np.asarray(t, dtype=np.float64))
    nt, nx = t.size, x.size

    z = np.linspace(-12, 12, n_q)
    u = np.empty((nt, nx))

    for k, tk in enumerate(t):
        if tk == 0.0:
            row = -np.sin(np.pi * x)
            u[k] = row
            continue

        s = np.sqrt(4 * nu * tk)

        y = x[:, None] - s * z[None, :]
        a = -np.cos(np.pi * y) / (2 * np.pi * nu) - z[None, :]**2
        a_sft = a - a.max(axis=1, keepdims=True)
        w = np.exp(a_sft)

        num = (np.sin(np.pi * y) * w).sum(axis=1)
        den = w.sum(axis=1)
        row = -num / den
        u[k] = row

    return u


def reference_grid(nx=256, nt=101, nu=NU):
    """ define a reference mesh with nx, nt points """

    x = np.linspace(LB[0], UB[0], nx)
    t = np.linspace(LB[1], UB[1], nt)

    return x, t, reference_soln(x, t, nu)


def make_forward_loss(model, nu, nf, n0, nb, weights, seed=0):
    """Forward Burgers problem: PDE, ic, and BCs"""
    gen = torch.Generator().manual_seed(seed)

    xt_f = _uniform(nf, LB, UB, gen)

    x0 = _uniform(n0, (LB[0],), (UB[0],), gen)
    t0 = torch.zeros_like(x0)
    xt_0 = torch.cat([x0, t0], dim=1)
    u0 = -torch.sin(np.pi * x0)

    tb = _uniform(nb, (LB[1],), (UB[1],), gen)
    xt_lft = torch.cat([-torch.ones_like(tb), tb], dim=1)
    xt_rht = torch.cat([torch.ones_like(tb), tb], dim=1)
    xt_b = torch.cat([xt_lft, xt_rht], dim=0)

    def loss_fcn():
        l_pde = (residual(model, xt_f, nu)).pow(2).mean()
        l_ic = (model(xt_0) - u0).pow(2).mean()
        l_bc = (model(xt_b)).pow(2).mean()

        l_tot = weights["pde"] * l_pde + weights["ic"] * l_ic + weights["bc"] * l_bc

        return l_tot, {"pde": l_pde.item(), "ic": l_ic.item(), "bc": l_bc.item()}

    return loss_fcn


def make_inverse_loss(model, coeffs, x, t, u_ref, n_data, noise, n_f, weights, seed=0):
    """Inverse Burgers problem: noisy data with unknowns lambda1, and nu"""
    gen = torch.Generator().manual_seed(seed)
    dtype = torch.get_default_dtype()

    X, T = np.meshgrid(x, t)
    xt = np.stack([X.ravel(), T.ravel()], axis=1)
    all_xt = torch.tensor(xt, dtype=dtype)
    u = u_ref.ravel()[:, None]
    all_u = torch.tensor(u, dtype=dtype)

    idx = torch.randperm(all_xt.shape[0], generator=gen)[:n_data]
    xt_d = all_xt[idx]
    u_d = all_u[idx] + noise * all_u.std() * torch.randn(n_data, 1, generator=gen)

    xt_f = _uniform(n_f, LB, UB, gen)

    def loss_fcn():
        l_data = (model(xt_d) - u_d).pow(2).mean()
        l_pde = (residual(model, xt_f, coeffs.nu, coeffs.lambda1)).pow(2).mean()

        l_tot = weights["data"] * l_data + weights["pde"] * l_pde

        return l_tot, {"data": l_data.item(), "pde": l_pde.item()}

    return loss_fcn

